"""Training-only, fold-local selection for the 17-domain mixture surrogate.

No A6--A15 paths occur in this module. Nested validation evaluates the complete
selection procedure; the final model is selected again using A4/A5 only.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

import numpy as np
import pandas as pd
import sklearn
from scipy.linalg import helmert
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits

SEED = 20260923
VERSION = "q1-solver-2.0"
SELECTION_FILES = ("solver_protocol.json", "solver_cv.csv", "solver_cv_folds.csv",
                   "solver_nested_cv.csv", "solver_nested_inner.csv",
                   "solver_nested_predictions.csv", "solver_folds.csv")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verified_selection(path, fingerprint):
    """Only reuse a complete experiment whose output hashes still agree."""
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
        hashes = saved["artifact_sha256"]
        if (saved["fingerprint"] != fingerprint or set(hashes) != set(SELECTION_FILES)
                or any(file_hash(path.parent / name) != hashes[name] for name in SELECTION_FILES)):
            return None
        return saved
    except (OSError, ValueError, KeyError, TypeError):
        return None


def check_simplex(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) == 0 or x.shape[1] < 2 or not np.isfinite(x).all():
        raise ValueError("Mixtures must be a finite two-dimensional matrix.")
    if np.any(x < 0) or not np.allclose(x.sum(axis=1), 1.0, rtol=0, atol=1e-10):
        raise ValueError("Mixtures must be nonnegative and sum to one.")
    return x


class SimplexFeatures(TransformerMixin, BaseEstimator):
    """Zero-safe Hellinger coordinates or a full-rank orthogonal contrast."""

    def __init__(self, kind: str = "sqrt"):
        self.kind = kind

    def fit(self, x, y=None):
        x = check_simplex(x)
        if self.kind not in {"sqrt", "contrast"}:
            raise ValueError(f"Unknown simplex coordinates: {self.kind}")
        self.n_features_in_ = x.shape[1]
        self.basis_ = helmert(self.n_features_in_, full=False).T
        return self

    def transform(self, x):
        x = check_simplex(x)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Mixture dimension differs from fitted dimension.")
        return np.sqrt(x) if self.kind == "sqrt" else x @ self.basis_


class FoldNormalizedRegressor(RegressorMixin, BaseEstimator):
    """Fit the target scale on fit rows only; predict raw positive-unit Loss."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y):
        x = check_simplex(x)
        y = np.asarray(y, dtype=float)
        if y.ndim != 2 or len(y) != len(x) or not np.isfinite(y).all() or np.any(y <= 0):
            raise ValueError("Loss targets must be a finite positive matrix.")
        self.target_scale_ = y.mean(axis=0)
        self.n_features_in_ = x.shape[1]
        self.estimator_ = clone(self.estimator).fit(x, y / self.target_scale_)
        return self

    def predict(self, x):
        return self.estimator_.predict(check_simplex(x)) * self.target_scale_


def candidate_models(jobs: int = 4) -> dict:
    """Finite candidate set fixed before nested or external evaluation."""
    def trees(n=256, leaf=3, features=1.0):
        return ExtraTreesRegressor(n_estimators=n, min_samples_leaf=leaf,
                                   max_features=features, random_state=SEED, n_jobs=jobs)

    def quadratic(kind, alpha):
        """Degree-2 polynomial in raw, Helmert contrast, or sqrt coordinates.

        In raw proportions, the sqrt model contains sqrt(p_i) and
        sqrt(p_i * p_j) terms, as well as p_i terms. Finite-difference effects
        are defined at their probes; they are not polynomial coefficients in p.
        """
        if kind not in {"raw", "contrast", "sqrt"}:
            raise ValueError(f"Unknown quadratic coordinates: {kind}")
        steps = [] if kind == "raw" else [SimplexFeatures(kind)]
        return make_pipeline(*steps, PolynomialFeatures(2, include_bias=False),
                             StandardScaler(), Ridge(alpha=alpha))

    return {
        "mean": DummyRegressor(),
        # 以下两个键沿用归档输出中记录的模型标识（data/outputs/q1/*.csv 内同名），改名会破坏复现
        "legacy_extra_trees": trees(160, 3),
        "extra_trees_leaf1": trees(256, 1),
        "extra_trees_leaf3": trees(256, 3),
        "extra_trees_leaf6": trees(256, 6),
        "extra_trees_f07": trees(256, 3, .7),
        "sqrt_extra_trees": make_pipeline(SimplexFeatures("sqrt"), trees()),
        "sqrt_rbf_a001_g05": make_pipeline(SimplexFeatures("sqrt"), KernelRidge(alpha=.01, kernel="rbf", gamma=.5)),
        "sqrt_rbf_a01_g05": make_pipeline(SimplexFeatures("sqrt"), KernelRidge(alpha=.1, kernel="rbf", gamma=.5)),
        "sqrt_rbf_a001_g2": make_pipeline(SimplexFeatures("sqrt"), KernelRidge(alpha=.01, kernel="rbf", gamma=2.)),
        "legacy_quadratic_100": quadratic("raw", 100.),
        "simplex_quadratic_10": quadratic("contrast", 10.),
        "simplex_quadratic_100": quadratic("contrast", 100.),
        # Additional fixed candidates; selection compares them with every
        # existing model under the same fold-local protocol.
        "sqrt_quadratic_a01": quadratic("sqrt", 0.1),
        "sqrt_quadratic_a1": quadratic("sqrt", 1.),
        "sqrt_quadratic_a10": quadratic("sqrt", 10.),
    }


def prediction_metrics(y_raw, pred_raw, scale):
    y, pred = np.asarray(y_raw) / scale, np.asarray(pred_raw) / scale
    error = y - pred
    actual, forecast = y.mean(axis=1), pred.mean(axis=1)
    selected, oracle = int(np.argmin(forecast)), int(np.argmin(actual))
    order = np.argsort(forecast, kind="stable")
    return {
        "pooled_rmse": float(np.sqrt(np.mean(error**2))),
        "macro_rmse": float(np.sqrt(np.mean(error**2, axis=0)).mean()),
        "mae": float(np.mean(abs(error))),
        "objective_rmse": float(np.sqrt(np.mean((actual - forecast)**2))),
        "objective_spearman": (float(spearmanr(actual, forecast).statistic)
                                if np.ptp(forecast) > 1e-14 else None),
        "top1_regret_pct": float(100 * (actual[selected] / actual[oracle] - 1)),
        "top10_mean_regret_pct": float(100 * (actual[order[:10]].mean() / np.sort(actual)[:10].mean() - 1)),
        "selected_local_row": selected,
        "oracle_local_row": oracle,
    }


class FoldEvaluator:
    """Reuse scores/predictions for identical model/split pairs, never models."""
    def __init__(self, x, y, models):
        self.x, self.y, self.models = x, y, models
        self.cache = {}
        self.fit_count = 0
        self.cache_hits = 0

    def evaluate(self, name, train, test):
        key = (name, tuple(train), tuple(test))
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        model = FoldNormalizedRegressor(self.models[name])
        start = perf_counter()
        with threadpool_limits(limits=1):
            model.fit(self.x[train], self.y[train])
            fit_seconds = perf_counter() - start
            start = perf_counter()
            pred = model.predict(self.x[test])
            predict_seconds = perf_counter() - start
        result = prediction_metrics(self.y[test], pred, model.target_scale_)
        result.update(fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                      n_train=len(train), n_test=len(test))
        self.cache[key] = (result, pred)
        self.fit_count += 1
        return result, pred


def summarize(rows: list[dict]) -> pd.DataFrame:
    records = pd.DataFrame(rows)
    return records.groupby("model", sort=False).agg(
        n_folds=("macro_rmse", "size"),
        cv_macro_rmse=("macro_rmse", "mean"),
        cv_macro_rmse_sd=("macro_rmse", "std"),
        cv_pooled_rmse=("pooled_rmse", "mean"),
        cv_objective_rmse=("objective_rmse", "mean"),
        cv_objective_spearman=("objective_spearman", "mean"),
        cv_top1_regret_pct=("top1_regret_pct", "mean"),
        cv_top10_regret_pct=("top10_mean_regret_pct", "mean"),
        mean_fit_seconds=("fit_seconds", "mean"),
        mean_predict_seconds=("predict_seconds", "mean"),
    ).reset_index().sort_values(["cv_macro_rmse", "cv_objective_rmse", "model"], kind="stable")


@dataclass
class TrainingResult:
    selected_name: str
    quadratic_name: str
    selected: FoldNormalizedRegressor
    quadratic: FoldNormalizedRegressor
    models: dict
    cv: pd.DataFrame
    metadata: dict


def train_solver(x, y_raw, out_dir: Path, *, jobs=4, repeats=3, outer_folds=5,
                 inner_folds=3, reuse=True) -> TrainingResult:
    x, y_raw = check_simplex(x), np.asarray(y_raw, dtype=float)
    if (y_raw.ndim != 2 or y_raw.shape[1] == 0 or len(x) != len(y_raw)
            or not np.isfinite(y_raw).all() or np.any(y_raw <= 0)):
        raise ValueError("Loss targets must be a finite positive matrix aligned to mixtures.")
    if repeats < 1 or jobs < 1 or min(outer_folds, inner_folds) < 2:
        raise ValueError("repeats/jobs must be positive; nested fold counts must be at least two.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = candidate_models(jobs)
    config = dict(version=VERSION, seed=SEED, jobs=jobs, full_folds=5,
                  full_repeats=repeats, outer_folds=outer_folds, inner_folds=inner_folds,
                  selection="mean fold macro RMSE; objective RMSE then name break exact ties",
                  normalization="13 target means fitted separately inside every training fold",
                  external_evaluation="A6-A15 are historical reference sets, never tuning inputs",
                  candidates={name: str(model) for name, model in models.items()},
                  python=platform.python_version(), sklearn=sklearn.__version__)
    data_hash = hashlib.sha256(x.tobytes() + y_raw.tobytes()).hexdigest()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    fingerprint = hashlib.sha256((data_hash + source_hash + json.dumps(config, sort_keys=True)).encode()).hexdigest()
    protocol = dict(config, training_array_sha256=data_hash, solver_source_sha256=source_hash,
                    fingerprint=fingerprint, training_rows=len(x), input_domains=x.shape[1],
                    loss_domains=y_raw.shape[1])
    selection_path = out_dir / "solver_selection.json"
    metadata = None
    if reuse and selection_path.exists():
        metadata = verified_selection(selection_path, fingerprint)
        if metadata is not None:
            print("Reusing verified A4/A5 model-selection results", flush=True)
    if metadata is None:
        # Persist the protocol before any folds run, so the experiment is auditable.
        (out_dir / "solver_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        evaluator = FoldEvaluator(x, y_raw, models)
        start = perf_counter()
        nested_rows, nested_inner_rows, split_rows, nested_predictions = [], [], [], []
        for outer, (train, test) in enumerate(KFold(outer_folds, shuffle=True, random_state=SEED + 101).split(x)):
            rows = []
            for inner, (a, b) in enumerate(KFold(inner_folds, shuffle=True, random_state=SEED + 1001 + outer).split(train)):
                for name in models:
                    metrics, _ = evaluator.evaluate(name, train[a], train[b])
                    rows.append(dict(model=name, outer_fold=outer, inner_fold=inner, **metrics))
            inner_table = summarize(rows)
            chosen = str(inner_table.iloc[0].model)
            nested_inner_rows.extend(rows)
            for method, name in (("optimized_selection", chosen), ("legacy_fixed", "legacy_extra_trees")):
                metrics, predictions = evaluator.evaluate(name, train, test)
                nested_rows.append(dict(method=method, selected_model=name, outer_fold=outer, **metrics))
                for position, raw_row in zip(test, predictions):
                    nested_predictions.append(dict(method=method, outer_fold=outer, training_row=int(position),
                                                   **{f"pred_loss_{j}": float(v) for j, v in enumerate(raw_row)}))
            split_rows.extend(dict(stage="nested_outer", repeat=0, fold=outer, validation_row=int(i)) for i in test)
            print(f"Nested outer fold {outer + 1}/{outer_folds}: {chosen}", flush=True)
            pd.DataFrame(nested_rows).to_csv(out_dir / "solver_nested_cv.csv", index=False)
        pd.DataFrame(nested_inner_rows).to_csv(out_dir / "solver_nested_inner.csv", index=False)
        pd.DataFrame(nested_predictions).to_csv(out_dir / "solver_nested_predictions.csv", index=False)
        rows = []
        # One repeated pass supplies both the first five folds and stability.
        for fold, (train, test) in enumerate(RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=SEED).split(x)):
            for name in models:
                metrics, _ = evaluator.evaluate(name, train, test)
                rows.append(dict(model=name, repeat=fold // 5, fold=fold % 5, **metrics))
            split_rows.extend(dict(stage="final_selection", repeat=fold // 5, fold=fold % 5,
                                   validation_row=int(i)) for i in test)
            print(f"Final selection fold {fold + 1}/{5 * repeats}", flush=True)
        pd.DataFrame(rows).to_csv(out_dir / "solver_cv_folds.csv", index=False)
        pd.DataFrame(split_rows).to_csv(out_dir / "solver_folds.csv", index=False)
        cv = summarize(rows)
        cv.to_csv(out_dir / "solver_cv.csv", index=False)
        selected_name = str(cv.iloc[0].model)
        quadratic_name = str(cv[cv.model.str.contains("quadratic")].iloc[0].model)
        nested = pd.DataFrame(nested_rows)
        baseline = nested[nested.method == "legacy_fixed"].set_index("outer_fold")
        optimized = nested[nested.method == "optimized_selection"].set_index("outer_fold")
        metric_columns = ["macro_rmse", "pooled_rmse", "mae", "objective_rmse",
                          "objective_spearman", "top1_regret_pct", "top10_mean_regret_pct",
                          "fit_seconds", "predict_seconds"]
        nested_summary = {
            method: {key: (float(value) if pd.notna(value) and np.isfinite(value) else None)
                     for key, value in group[metric_columns].mean().items()}
            for method, group in nested.groupby("method")}
        metadata = dict(fingerprint=fingerprint, selected_model=selected_name,
                        quadratic_model=quadratic_name, training_array_sha256=data_hash,
                        fit_count=evaluator.fit_count, cached_fold_hits=evaluator.cache_hits,
                        selection_wall_seconds=perf_counter() - start,
                        nested_cv=nested_summary,
                        artifact_sha256={name: file_hash(out_dir / name) for name in SELECTION_FILES},
                        outer_folds_macro_improved=int((optimized.macro_rmse < baseline.macro_rmse).sum()),
                        nested_cv_is_new_external_blind_test=False)
        selection_path.write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    cv = pd.read_csv(out_dir / "solver_cv.csv")
    selected_name, quadratic_name = metadata["selected_model"], metadata["quadratic_model"]
    with threadpool_limits(limits=1):
        selected = FoldNormalizedRegressor(models[selected_name]).fit(x, y_raw)
        quadratic = FoldNormalizedRegressor(models[quadratic_name]).fit(x, y_raw)
    return TrainingResult(selected_name, quadratic_name, selected, quadratic, models, cv, metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=paths.A)
    parser.add_argument("--output-dir", type=Path, default=paths.Q1)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    # The proven pair loader enforces index-based matching and column names.
    import solve_q1
    solve_q1.DATA = args.data_dir.resolve()
    solve_q1.OUT = args.output_dir.resolve()
    x, raw, _, _, _, _ = solve_q1.load_pair("train_mixture_1m.csv", "train_pile_loss_1m.csv")
    result = train_solver(x, raw, args.output_dir, jobs=args.jobs, repeats=args.repeats, reuse=not args.refresh)
    print(result.cv.to_string(index=False))
    print("Selected:", result.selected_name, "Quadratic:", result.quadratic_name)


if __name__ == "__main__":
    main()
