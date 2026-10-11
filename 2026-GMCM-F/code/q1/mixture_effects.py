"""Reusable simplex probes and batched local response calculations.

The probe definitions, bootstrap resamples, target averaging, scaler fits and
Ridge penalty are unchanged from the original Q1 implementation. Only
deterministic feature construction and prediction dispatch are shared.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from mixture_solver import SimplexFeatures


DEFAULT_SEED = 20260923


@dataclass(frozen=True)
class ProbeDesign:
    """Ordered anchor, single reallocations, then lexicographic pair probes."""

    domains: tuple[str, ...]
    anchor_index: int
    delta: float
    points: np.ndarray
    pair_indices: np.ndarray

    @property
    def anchor(self) -> np.ndarray:
        return self.points[0]

    @property
    def n_domains(self) -> int:
        return len(self.domains)


def build_probe_design(p_train: np.ndarray, domains: list[str],
                       delta: float = 0.05) -> ProbeDesign:
    """Build feasible probes once, using the nearest observed mean recipe."""
    p_train = np.asarray(p_train, dtype=float)
    if (p_train.ndim != 2 or len(p_train) == 0 or p_train.shape[1] == 0
            or p_train.shape[1] != len(domains)):
        raise ValueError("training recipes and domain names must have matching dimensions")
    if len(set(domains)) != len(domains):
        raise ValueError("domain names must be unique")
    if (not np.all(np.isfinite(p_train)) or np.any(p_train < 0)
            or not np.allclose(p_train.sum(axis=1), 1, rtol=0, atol=1e-10)):
        raise ValueError("training recipes must be finite nonnegative simplex points")
    if not np.isfinite(delta) or not 0 <= delta <= 0.5:
        raise ValueError("delta must be in [0, 0.5] for feasible pair reallocations")
    anchor_index = int(np.argmin(np.linalg.norm(p_train - p_train.mean(axis=0), axis=1)))
    anchor = p_train[anchor_index]
    n_domains = len(domains)
    identity = np.eye(n_domains)
    pairs = np.asarray(list(combinations(range(n_domains), 2)), dtype=int).reshape(-1, 2)
    singles = (1 - delta) * anchor + delta * identity
    pair_points = ((1 - 2 * delta) * anchor
                   + delta * (identity[pairs[:, 0]] + identity[pairs[:, 1]]))
    points = np.vstack((anchor, singles, pair_points))
    points.setflags(write=False)
    pairs.setflags(write=False)
    return ProbeDesign(tuple(domains), anchor_index, float(delta), points, pairs)


def _design(p_train: np.ndarray, domains: list[str], probes: ProbeDesign | None) -> ProbeDesign:
    if probes is None:
        return build_probe_design(p_train, domains)
    if tuple(domains) != probes.domains or np.shape(p_train)[1] != probes.n_domains:
        raise ValueError("the supplied probe design does not match the training domains")
    return probes


def predict_objective(model: object, points: np.ndarray) -> np.ndarray:
    """Evaluate the equal-weight mean normalized Loss in one model call."""
    predictions = np.asarray(model.predict(points), dtype=float)
    if predictions.ndim == 1:
        values = predictions
    elif predictions.ndim == 2:
        values = predictions.mean(axis=1)
    else:
        raise ValueError("model prediction must have one or two dimensions")
    if values.shape != (len(points),) or not np.all(np.isfinite(values)):
        raise ValueError("model returned invalid local objective predictions")
    return values


def _effect_values(values: np.ndarray, probes: ProbeDesign) -> np.ndarray:
    n_domains = probes.n_domains
    pairs = probes.pair_indices
    singles = values[1:1 + n_domains] - values[0]
    interactions = (values[1 + n_domains:]
                    - (values[1 + pairs[:, 0]] + values[1 + pairs[:, 1]]) + values[0])
    return np.concatenate((singles, interactions))


def _write(table: pd.DataFrame, out_dir: str | Path | None, filename: str) -> None:
    if out_dir is not None:
        destination = Path(out_dir)
        destination.mkdir(parents=True, exist_ok=True)
        table.to_csv(destination / filename, index=False, encoding="utf-8-sig")


def domain_effects(quadratic: object, p_train: np.ndarray, domains: list[str], *,
                   probes: ProbeDesign | None = None,
                   out_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the original single/pair tables using one batched prediction."""
    probes = _design(p_train, domains, probes)
    effects = _effect_values(predict_objective(quadratic, probes.points), probes)
    single = pd.DataFrame({
        "domain": domains,
        "anchor_share": probes.anchor,
        "loss_change_5pct_reallocation": effects[:probes.n_domains],
    }).sort_values("loss_change_5pct_reallocation")
    pairs = pd.DataFrame({
        "domain_i": [domains[i] for i in probes.pair_indices[:, 0]],
        "domain_j": [domains[j] for j in probes.pair_indices[:, 1]],
        "interaction_loss": effects[probes.n_domains:],
    }).sort_values("interaction_loss")
    _write(single, out_dir, "domain_reallocation_effects.csv")
    _write(pairs, out_dir, "domain_pair_interactions.csv")
    return single, pairs


def _bootstrap_features(model_template: object, p_train: np.ndarray,
                        points: np.ndarray, cache_features: bool) -> tuple[object, np.ndarray, np.ndarray]:
    # PolynomialFeatures.fit learns dimensionality only. Its expansion can be
    # shared across resamples. StandardScaler and Ridge MUST remain inside the
    # refitted suffix: freezing the scaler would alter the Ridge penalty.
    if (cache_features and isinstance(model_template, Pipeline)
            and len(model_template.steps) > 1
            and type(model_template.steps[0][1]) is PolynomialFeatures
            and getattr(model_template, "transform_input", None) is None):
        expansion = clone(model_template.steps[0][1])
        expanded_train = expansion.fit_transform(p_train)
        expanded_points = expansion.transform(points)
        return model_template[1:], expanded_train, expanded_points
    if (cache_features and isinstance(model_template, Pipeline)
            and len(model_template.steps) > 2
            and type(model_template.steps[0][1]) is SimplexFeatures
            and type(model_template.steps[1][1]) is PolynomialFeatures
            and getattr(model_template, "transform_input", None) is None):
        expansion = clone(model_template[:2])
        expanded_train = expansion.fit_transform(p_train)
        return model_template[2:], expanded_train, expansion.transform(points)
    return model_template, p_train, points


def bootstrap_effects(model_template: object, p_train: np.ndarray, y_train: np.ndarray,
                      domains: list[str], n_boot: int = 80, *,
                      probes: ProbeDesign | None = None,
                      out_dir: str | Path | None = None,
                      seed: int = DEFAULT_SEED, cache_features: bool = True) -> pd.DataFrame:
    """Replay recipe bootstrap with resample-specific scaling and Ridge fits.

    The known data-independent SimplexFeatures/PolynomialFeatures prefix can
    be cached. All other estimator types use the complete fit pipeline.
    """
    if not isinstance(n_boot, (int, np.integer)) or n_boot < 1:
        raise ValueError("n_boot must be a positive integer")
    probes = _design(p_train, domains, probes)
    p_train, y_train = np.asarray(p_train), np.asarray(y_train)
    if len(y_train) != len(p_train):
        raise ValueError("training recipes and targets must have equal lengths")
    template, train_features, point_features = _bootstrap_features(
        model_template, p_train, probes.points, cache_features)
    rng = np.random.default_rng(seed)
    effects = np.empty((n_boot, probes.n_domains + len(probes.pair_indices)))
    for b in range(n_boot):
        ix = rng.integers(0, len(p_train), len(p_train))
        model = clone(template).fit(train_features[ix], y_train[ix])
        effects[b] = _effect_values(predict_objective(model, point_features), probes)
    low, high = np.quantile(effects, [0.025, 0.975], axis=0)
    names = [(name, "") for name in domains] + [
        (domains[i], domains[j]) for i, j in probes.pair_indices]
    result = pd.DataFrame({
        "domain_i": [a for a, _ in names],
        "domain_j": [b for _, b in names],
        "effect_median": np.median(effects, axis=0),
        "effect_ci025": low,
        "effect_ci975": high,
        "fraction_negative": np.mean(effects < 0, axis=0),
    })
    _write(result, out_dir, "effect_bootstrap.csv")
    return result


def response_surface_export(selected_model: object, quadratic: object,
                            p_train: np.ndarray, domains: list[str],
                            pstar: np.ndarray, *,
                            probes: ProbeDesign | None = None) -> pd.DataFrame:
    """Preserve the Q2 exchange schema, with one prediction per model."""
    probes = _design(p_train, domains, probes)
    pstar = np.asarray(pstar, dtype=float)
    if (pstar.shape != (probes.n_domains,) or not np.all(np.isfinite(pstar))
            or np.any(pstar < 0) or not np.isclose(pstar.sum(), 1, rtol=0, atol=1e-10)):
        raise ValueError("selected recipe must be a finite vector matching the domains")
    points = np.vstack((probes.points, pstar))
    kinds = (["anchor"] + ["single_reallocation"] * probes.n_domains
             + ["pair_reallocation"] * len(probes.pair_indices) + ["selected_1m_candidate"])
    first = ([""] + list(domains)
             + [domains[i] for i in probes.pair_indices[:, 0]] + [""])
    second = ([""] * (1 + probes.n_domains)
              + [domains[j] for j in probes.pair_indices[:, 1]] + [""])
    frames = []
    for label, model in (("selected", selected_model), ("quadratic", quadratic)):
        values = predict_objective(model, points)
        frames.append(pd.DataFrame({
            "model": label, "point_kind": kinds, "domain_i": first, "domain_j": second,
            "J_hat": values, "loss_change_vs_anchor": values - values[0],
        }))
    return pd.concat(frames, ignore_index=True)
