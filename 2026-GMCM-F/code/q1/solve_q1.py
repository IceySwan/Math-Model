"""Reproducible solution of Question 1 of the 2026 F problem.

Run from any directory with Python 3.12 and the packages in requirements.txt.

Role of each attachment, stated exactly as implemented:
  * A1 alone fits the fixed 1%/99% quality scales. Direction rules are fixed
    modelling choices. Mixture target scales are fitted on A5 training folds.
  * A2/A3 are read for the extended arxiv/github domain quality summaries. Those
    summaries are pooled with A1 by document id before the domain-quality table
    that later questions consume, so A2/A3 do enter the cross-question quality
    map. They never enter any fitted parameter.
  * A6--A15 are never used to tune the quality transforms or the mixture model;
    they are read after fitting, purely for reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from time import perf_counter
from threadpoolctl import threadpool_limits
from quality_io import read_quality as read_quality_cached
from q1_pipeline import analyze_mixtures as run_mixture_pipeline


HERE = Path(__file__).resolve().parent
DATA = paths.A
OUT = paths.Q1
CACHE_DIR = paths.DATA / "cache" / "quality"   # 质量阶段缓存，可删除后重建
CACHE_MODE = "use"
SEED = 20260923
# ExtraTrees fits are parallelised through joblib. Some confined sandboxes deny
# the named-pipe creation joblib needs, so the worker count is overridable:
# set Q1_N_JOBS=1 to run single-threaded. The fitted model is identical either
# way (fixed random_state), so this only affects wall-clock time.
N_JOBS = int(os.environ.get("Q1_N_JOBS", "4"))

SIGNALS = [
    "fineweb_edu", "fluency_en", "modernbert_cleanliness",
    "modernbert_readability", "modernbert_reasoning",
    "modernbert_professionalism", "dsir_books", "dsir_wiki", "dsir_math",
    "qurater", "ad_en", "rps_doc_word_count", "rps_doc_num_sentences",
    "rps_doc_unigram_entropy", "rps_doc_frac_unique_words",
    "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
    "rps_lines_ending_with_terminal_punctution_mark",
    "rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length",
]
MODEL = list(range(0, 6)) + [9, 10]
DSIR = [6, 7, 8]
SURFACE = list(range(11, 22))
NEGATIVE = {
    "rps_doc_frac_no_alph_words", "rps_doc_frac_chars_top_2gram",
    "rps_doc_frac_chars_top_3gram", "rps_lines_uppercase_letter_fraction",
}
TWO_SIDED = {"rps_lines_numerical_chars_fraction", "rps_doc_mean_word_length"}
AD_RE = re.compile(r"\b(?:buy now|subscribe|coupon|click here|affiliate|sponsored|free trial)\b", re.I)
URL_RE = re.compile(r"https?://", re.I)
# Two distinct lower bounds on the word count, kept deliberately separate:
#   * DSIR_NORMALIZER_FLOOR is applied to rps_doc_word_count itself, because the
#     DSIR signals are divided by the (transformed) word count and a handful of
#     near-empty records would otherwise produce exploding ratios.  The value 20
#     matches the floor used when the DSIR signals were published.
#   * TOKEN_WEIGHT_FLOOR is the minimum weight for the token-weighted domain mean.
#     A document still counts as at least one unit of volume.
DSIR_NORMALIZER_FLOOR = 20.0
TOKEN_WEIGHT_FLOOR = 1.0


def softmax_expected(values: object) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) != 6 or not np.all(np.isfinite(arr)):
        return float("nan")
    z = np.exp(arr - arr.max())
    return float(np.dot(np.arange(6) / 5.0, z / z.sum()))


def binary_positive(values: object) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or len(arr) != 2 or not np.all(np.isfinite(arr)):
        return float("nan")
    delta = float(np.clip(arr[1] - arr[0], -50, 50))
    return 1.0 / (1.0 + math.exp(-delta))


def scalarize(row: dict) -> list[float]:
    vals = []
    word_count = row.get("rps_doc_word_count")
    try:
        word_count = max(float(word_count), DSIR_NORMALIZER_FLOOR)
    except (TypeError, ValueError):
        word_count = DSIR_NORMALIZER_FLOOR
    for name in SIGNALS:
        v = row.get(name)
        try:
            if name.startswith("modernbert_"):
                val = softmax_expected(v)
            elif name in {"ad_en", "fluency_en"}:
                # Dataset card: class 1 means no advertisement / fluent.
                val = binary_positive(v)
            elif name == "fineweb_edu":
                val = float(v[0]) if isinstance(v, list) and len(v) == 1 else float("nan")
            elif name == "qurater":
                arr = np.asarray(v, dtype=float)
                val = float(arr.mean()) if len(arr) == 4 and np.all(np.isfinite(arr)) else float("nan")
            elif name.startswith("dsir_"):
                # DSIR values are strongly length-dependent in A1; words proxy tokens.
                val = float(v) / word_count
            elif name in {"rps_doc_word_count", "rps_doc_num_sentences"}:
                val = math.log1p(max(float(v), 0.0))
            else:
                val = float(v)
        except (TypeError, ValueError, IndexError, OverflowError):
            val = float("nan")
        vals.append(val if np.isfinite(val) else float("nan"))
    return vals


def text_audit(content: str) -> tuple[int, int, float, str]:
    lines = [x.strip() for x in content.splitlines() if len(x.strip()) >= 20]
    repeat_rate = 1.0 - len(set(lines)) / len(lines) if lines else 0.0
    excerpt = re.sub(r"\s+", " ", content)[:110]
    return int(bool(AD_RE.search(content))), int(len(URL_RE.findall(content)) >= 3), repeat_rate, excerpt


def read_quality() -> tuple[pd.DataFrame, dict]:
    return read_quality_cached(DATA, signals=SIGNALS,
                               dsir_normalizer_floor=DSIR_NORMALIZER_FLOOR,
                               token_weight_floor=TOKEN_WEIGHT_FLOOR,
                               cache_dir=CACHE_DIR, cache_mode=CACHE_MODE)


def normalize_quality(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Freeze 1%/99% winsorized min-max scales on A1 and apply to every row."""
    fit = frame.loc[frame.source == "A1", SIGNALS].to_numpy(dtype=float)
    all_values = frame[SIGNALS].to_numpy(dtype=float)
    z = np.empty_like(all_values)
    specs = {}
    for j, name in enumerate(SIGNALS):
        x = fit[:, j]
        valid = x[np.isfinite(x)]
        median = float(np.median(valid))
        lo, hi = np.quantile(valid, [0.01, 0.99])
        if hi <= lo:
            hi = lo + 1.0
        y = np.where(np.isfinite(all_values[:, j]), all_values[:, j], median)
        u = np.clip((y - lo) / (hi - lo), 0.0, 1.0)
        if name in NEGATIVE:
            u = 1.0 - u
            direction = "negative"
        elif name in TWO_SIDED:
            # Extremes can reflect genre rather than quality. Middle A1 range is preferred.
            u = 1.0 - np.abs(2.0 * u - 1.0)
            direction = "two_sided"
        else:
            direction = "positive"
        z[:, j] = u
        specs[name] = {"p01": float(lo), "p99": float(hi), "impute": median, "direction": direction}
    means = np.column_stack([z[:, MODEL].mean(axis=1), z[:, DSIR].mean(axis=1), z[:, SURFACE].mean(axis=1)])
    baseline = means.mean(axis=1)
    spread = means.max(axis=1) - means.min(axis=1)
    conflict = (means.max(axis=1) >= 0.70) & (means.min(axis=1) <= 0.30)
    # DSIR measures affinity to three reference domains. A low DSIR score on
    # otherwise clear specialist text is disagreement, not contamination.
    pollution = (means[:, 0] >= 0.70) & (means[:, 2] <= 0.30)
    edu_ad = (z[:, SIGNALS.index("fineweb_edu")] >= 0.70) & (z[:, SIGNALS.index("ad_en")] <= 0.30)
    penalty = 0.25 * np.maximum(0.0, means[:, 0] - means[:, 2] - 0.40) * pollution + 0.05 * edu_ad
    quality = np.clip(0.50 * means[:, 0] + 0.15 * means[:, 1] + 0.35 * means[:, 2] - penalty, 0.0, 1.0)
    for j, group in enumerate(("model", "dsir", "surface")):
        frame[f"score_{group}"] = means[:, j]
    names = np.asarray(["model", "dsir", "surface"])
    frame["conflict_high_group"] = names[np.argmax(means, axis=1)]
    frame["conflict_low_group"] = names[np.argmin(means, axis=1)]
    frame["Q_equal"] = baseline
    frame["Q"] = quality
    # spread is descriptive only: it is exported so the disagreement between the
    # three signal groups can be inspected, and it never enters Q or any model.
    frame["spread"] = spread
    frame["conflict"] = conflict
    frame["pollution_conflict"] = pollution
    frame["edu_ad_conflict"] = edu_ad
    frame["Q_no_dsir"] = 0.60 * means[:, 0] + 0.40 * means[:, 2]
    return frame, specs


def weighted_mean(values: object, weights: object) -> float:
    """Word-count-weighted mean; raw word counts proxy token volume."""
    w = np.clip(np.asarray(weights, dtype=float), TOKEN_WEIGHT_FLOOR, None)
    return float(np.average(np.asarray(values, dtype=float), weights=w))


def summarize_quality(frame: pd.DataFrame, diagnostics: dict, specs: dict) -> dict:
    frame[["id", "source", "domain", "Q", "Q_equal", "Q_no_dsir",
           "score_model", "score_dsir", "score_surface", "conflict",
           "pollution_conflict", "edu_ad_conflict"]].to_csv(
               OUT / "quality_document_scores.csv.gz", index=False, compression="gzip")
    summary = []
    for (source, domain), sub in frame.groupby(["source", "domain"], sort=True):
        n = len(sub)
        se = float(sub.Q.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
        summary.append({"source": source, "domain": domain, "n": n,
                        "Q_mean": float(sub.Q.mean()), "Q_median": float(sub.Q.median()),
                        "Q_se": se, "Q_equal_mean": float(sub.Q_equal.mean()),
                        "Q_no_dsir_mean": float(sub.Q_no_dsir.mean()),
                        "Q_token_weighted_mean": weighted_mean(sub.Q, sub.word_count_raw),
                        "conflict_rate": float(sub.conflict.mean()),
                        "pollution_conflict_rate": float(sub.pollution_conflict.mean()),
                        "edu_ad_conflict_rate": float(sub.edu_ad_conflict.mean()),
                        "spread_mean": float(sub.spread.mean())})
    domains = pd.DataFrame(summary)
    domains.to_csv(OUT / "quality_domains.csv", index=False, encoding="utf-8-sig")
    frame.loc[frame.conflict].groupby(
        ["source", "domain", "conflict_high_group", "conflict_low_group"]
    ).size().rename("n").reset_index().to_csv(OUT / "conflict_types.csv", index=False, encoding="utf-8-sig")
    # The group-disagreement cut-off is a modelling choice, so report how the
    # flagged share responds to it alongside the share that actually enters the
    # explicit pollution/edu-ad penalty.
    group_matrix = frame[["score_model", "score_dsir", "score_surface"]].to_numpy(float)
    penalized = (frame["pollution_conflict"] | frame["edu_ad_conflict"]).to_numpy(bool)
    source_values = frame["source"].to_numpy()
    threshold_rows = []
    for hi, lo in [(0.65, 0.35), (0.70, 0.30), (0.75, 0.25)]:
        flagged = (group_matrix.max(axis=1) >= hi) & (group_matrix.min(axis=1) <= lo)
        for tag, mask in [("all", np.ones(len(frame), dtype=bool))] + [
            (t, source_values == t) for t in ("A1", "A2", "A3")
        ]:
            size = int(mask.sum())
            threshold_rows.append({
                "high_threshold": hi, "low_threshold": lo, "subset": tag,
                "n": size,
                "flagged_n": int((flagged & mask).sum()),
                "flagged_rate": float((flagged & mask).sum() / size) if size else float("nan"),
                "explicit_penalty_n": int((penalized & mask).sum()),
                "explicit_penalty_rate": float((penalized & mask).sum() / size) if size else float("nan"),
            })
    threshold_sensitivity = pd.DataFrame(threshold_rows)
    threshold_sensitivity.to_csv(OUT / "conflict_threshold_sensitivity.csv", index=False, encoding="utf-8-sig")
    # A2/A3 contain A1 records; do not double count in the pooled estimate.
    pooled = frame.sort_values("source", ascending=False).drop_duplicates("id", keep="first")
    pooled_summary = pooled.groupby("domain").agg(
        n=("id", "size"), Q_mean=("Q", "mean"), conflict_rate=("conflict", "mean")).reset_index()
    pooled_summary["Q_token_weighted_mean"] = [
        weighted_mean(pooled.Q[pooled.domain.eq(d)], pooled.word_count_raw[pooled.domain.eq(d)])
        for d in pooled_summary.domain
    ]
    pooled_summary.to_csv(OUT / "quality_unique_domains.csv", index=False, encoding="utf-8-sig")
    a1 = frame[frame.source == "A1"].copy()
    # Content-based checks are descriptive proxies, never human ground-truth labels.
    a1["Q_quartile_in_domain"] = a1.groupby("domain")["Q"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 4, labels=False))
    text_summary = a1.groupby("Q_quartile_in_domain").agg(
        n=("id", "size"), ad_marker_rate=("ad_marker", "mean"),
        many_urls_rate=("many_urls", "mean"), repeat_line_rate=("repeat_line_rate", "mean"))
    text_summary.to_csv(OUT / "text_audit_quartiles.csv", encoding="utf-8-sig")
    a1.groupby(["domain", "Q_quartile_in_domain"]).agg(
        n=("id", "size"), ad_marker_rate=("ad_marker", "mean"),
        many_urls_rate=("many_urls", "mean"), repeat_line_rate=("repeat_line_rate", "mean")
    ).reset_index().to_csv(OUT / "text_audit_by_domain.csv", index=False, encoding="utf-8-sig")
    examples = pd.concat([a1.nsmallest(8, "Q"), a1.nlargest(8, "Q"),
                          a1[a1.conflict].nlargest(8, "spread")])
    examples[["id", "domain", "Q", "score_model", "score_dsir", "score_surface",
              "conflict", "excerpt"]].to_csv(OUT / "text_examples_for_review.csv", index=False, encoding="utf-8-sig")
    within_domain = pd.concat([
        pd.concat([part.nsmallest(3, "Q"), part.nlargest(3, "Q")])
        for _, part in a1.groupby("domain", sort=True)
    ])
    within_domain[["id", "domain", "Q", "score_model", "score_dsir", "score_surface",
                   "conflict", "excerpt"]].to_csv(
                       OUT / "text_examples_within_domain.csv", index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    piv = domains.pivot(index="domain", columns="source", values="Q_mean")
    piv.plot.bar(ax=ax, rot=40, color={"A1": "#5074a0", "A2": "#d8914d", "A3": "#68a690"})
    ax.set(ylabel="Mean quality Q", xlabel="Quality domain", ylim=(0, 1))
    ax.legend(title="File")
    fig.tight_layout()
    fig.savefig(OUT / "quality_by_domain.pdf")
    plt.close(fig)
    return {"diagnostics": diagnostics, "normalization": specs,
            "quality_domains": summary,
            "unique_document_count": int(len(pooled)),
            "text_audit_quartiles": text_summary.reset_index().to_dict(orient="records"),
            "conflict_threshold_sensitivity": threshold_sensitivity.to_dict(orient="records"),
            "domain_quality_token_weighted": pooled_summary.to_dict(orient="records")}


def load_pair(mixture_name: str, loss_name: str, train_columns: list[str] | None = None,
              loss_columns: list[str] | None = None) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray, dict]:
    base = DATA / "regmix_tables"
    mix = pd.read_csv(base / mixture_name)
    loss = pd.read_csv(base / loss_name)
    mcols = [c for c in mix if c.startswith("train_the_pile_")]
    ycols = [c for c in loss if c.endswith("_val_loss")]
    if train_columns is not None and mcols != train_columns:
        raise ValueError(f"Mixture columns differ in {mixture_name}")
    if loss_columns is not None and ycols != loss_columns:
        raise ValueError(f"Loss columns differ in {loss_name}")
    if not mcols or not ycols or mix.empty or loss.empty:
        raise ValueError("Nonempty mixture and Loss columns are required")
    for table in (mix, loss):
        if "index" not in table or table["index"].isna().any():
            raise ValueError("Each attachment must contain nonmissing index values")
    pair = mix.merge(loss, on="index", validate="one_to_one", how="inner", sort=False)
    if len(pair) != len(mix) or len(pair) != len(loss):
        raise ValueError(f"Unpaired index rows: {mixture_name}, {loss_name}")
    p_raw = pair[mcols].to_numpy(dtype=float)
    y = pair[ycols].to_numpy(dtype=float)
    if not np.all(np.isfinite(p_raw)) or not np.all(np.isfinite(y)) or np.any(p_raw < 0) or np.any(y <= 0):
        raise ValueError(f"Missing, invalid, or negative data in {mixture_name} / {loss_name}")
    totals = p_raw.sum(axis=1)
    if np.any(totals <= 0):
        raise ValueError(f"Zero-total mixture row in {mixture_name}")
    p = p_raw / totals[:, None]
    meta = {"n": int(len(pair)), "raw_sum_min": float(totals.min()), "raw_sum_max": float(totals.max()),
            "zero_fraction": float(np.mean(p_raw == 0)),
            "source_sha256": {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                              for name in (mixture_name, loss_name)}}
    return p, y, mcols, ycols, pair["index"].to_numpy(), meta


def finite_or_none(value: object) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def quality_mapping(frame: pd.DataFrame, domains: list[str]) -> dict:
    mapping = pd.read_csv(DATA / "domain_mapping_guide.csv")
    means = frame.set_index("domain")["Q_mean"]
    token_means = frame.set_index("domain")["Q_token_weighted_mean"]
    mapping["Q_from_quality_domain"] = mapping.quality_domain.map(means)
    mapping["Q_token_weighted_from_quality_domain"] = mapping.quality_domain.map(token_means)
    mapping["has_empirical_Q"] = mapping.Q_from_quality_domain.notna()
    mapping.to_csv(OUT / "quality_to_mixture_mapping.csv", index=False, encoding="utf-8-sig")
    return {"direct_or_near_direct": int(mapping.has_empirical_Q.sum()),
            "unmapped_domains": mapping.loc[~mapping.has_empirical_Q, "mixture_domain"].tolist()}


def analyze_mixtures(frame: pd.DataFrame, *, repeats=3, refresh=False, n_boot=80) -> dict:
    return run_mixture_pipeline(load_pair, quality_mapping, frame, OUT, jobs=N_JOBS,
                                repeats=repeats, refresh=refresh, n_boot=n_boot)


def json_safe(value: object) -> object:
    """Recursively replace non-finite floats with None before strict JSON dumping.

    pandas stores an all-NaN column as float64, so a None written into a row
    comes back as NaN; this pass runs on the finished result tree and keeps
    allow_nan=False (which is what catches genuinely broken numbers) usable.
    """
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return finite_or_none(value)
    return value


def main() -> None:
    global DATA, OUT, CACHE_DIR, CACHE_MODE, N_JOBS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--stage", choices=("all", "quality", "mixture"), default="all")
    parser.add_argument("--quality-summary", type=Path, help="Explicit domain table for mixture-only runs")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--cache-mode", choices=("use", "refresh", "off"), default="use")
    parser.add_argument("--refresh-selection", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-boot", type=int, default=80)
    parser.add_argument("--jobs", type=int, default=N_JOBS)
    args = parser.parse_args()
    if min(args.repeats, args.n_boot, args.jobs) < 1:
        parser.error("repeats, n-boot and jobs must be positive")
    DATA, OUT = args.data_dir.resolve(), args.output_dir.resolve()
    CACHE_DIR, CACHE_MODE, N_JOBS = args.cache_dir.resolve(), args.cache_mode, args.jobs
    OUT.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    result = {}
    timings = {}
    with threadpool_limits(limits=1):
        if args.stage in ("all", "quality"):
            mark = perf_counter()
            quality, diagnostics = read_quality()
            timings["quality_read_seconds"] = perf_counter() - mark
            mark = perf_counter()
            quality, specs = normalize_quality(quality)
            result["quality"] = summarize_quality(quality, diagnostics, specs)
            timings["quality_score_export_seconds"] = perf_counter() - mark
            del quality
            (OUT / "quality_results.json").write_text(
                json.dumps(json_safe(result["quality"]), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        if args.stage in ("all", "mixture"):
            quality_path = (args.quality_summary or OUT / "quality_unique_domains.csv").resolve()
            if not quality_path.is_file():
                raise FileNotFoundError("Run --stage quality first or supply --quality-summary")
            mark = perf_counter()
            result["mixture"] = analyze_mixtures(pd.read_csv(quality_path), repeats=args.repeats,
                                                 refresh=args.refresh_selection, n_boot=args.n_boot)
            timings["mixture_seconds"] = perf_counter() - mark
            result["quality_map_source"] = str(quality_path)
            (OUT / "mixture_results.json").write_text(
                json.dumps(json_safe(result["mixture"]), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    # Per-stage files are authoritative; a partial run never pretends to refresh both stages.
    from mixture_solver import file_hash
    result["run"] = dict(stage=args.stage, data_dir=str(DATA), output_dir=str(OUT), jobs=N_JOBS,
                         blas_threads=1, cache_mode=CACHE_MODE, repeats=args.repeats, n_boot=args.n_boot,
                         seconds=perf_counter() - started, timings=timings,
                         source_sha256={name: file_hash(HERE / name) for name in
                                        ("solve_q1.py", "quality_io.py", "mixture_solver.py", "mixture_effects.py", "q1_pipeline.py")})
    filename = "results.json" if args.stage == "all" else f"run_{args.stage}.json"
    (OUT / filename).write_text(json.dumps(json_safe(result), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print("Results written to", OUT, flush=True)
    print(json.dumps(result["run"]["timings"], indent=2), flush=True)


if __name__ == "__main__":
    main()

