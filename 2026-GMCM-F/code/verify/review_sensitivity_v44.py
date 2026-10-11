"""Conditional Q1 mapping/conflict and Q3 saturation checks for review v44.

This reads frozen Q1/Q2 outputs. It does not refit B6 or claim proxy mappings
measure the missing domains; every alternative is a stated sensitivity case.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

sys.path.insert(0, str(paths.CODE / "q3"))
from core import Params, Problem  # noqa: E402

OUT = paths.REPORTS / "review_v44"
OUT.mkdir(parents=True, exist_ok=True)
P0 = pd.read_csv(paths.Q1 / "domain_reallocation_effects.csv").set_index("domain").anchor_share
MAPPING = pd.read_csv(paths.Q1 / "quality_to_mixture_mapping.csv").set_index("mixture_domain")
DOMAINS = pd.read_csv(paths.Q1 / "quality_unique_domains.csv").set_index("domain")
SCORES_PATH = paths.Q1 / "quality_document_scores.csv.gz"
HANDOFF = paths.Q2 / "q2_handoff_v26.json"
DRAW_PATH = paths.Q2 / "joint_parameter_draws.csv"
BOUND_PATH = paths.Q3 / "saturation_bound_draws.csv"

PROXY = {
    "dm_mathematics": "arxiv",
    "pubmed_central": "arxiv",
    "pubmed_abstracts": "arxiv",
    "philpapers": "book",
    "freelaw": "book",
    "hackernews": "commoncrawl",
    "europarl": "commoncrawl",
    "uspto_backgrounds": "commoncrawl",
    "enron_emails": "commoncrawl",
    "ubuntu_irc": "commoncrawl",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def q0_for(proxy: dict[str, str], domain_mean: pd.Series) -> tuple[float, float, float, float]:
    numerator = 0.0
    coverage = 0.0
    for mix_domain, quality_domain in proxy.items():
        weight = float(P0.loc[mix_domain])
        numerator += weight * float(domain_mean.loc[quality_domain])
        coverage += weight
    q0 = numerator / coverage
    return coverage, q0, coverage * q0, coverage * q0 + 1.0 - coverage


def write_csv(name: str, rows: list[dict]) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    result.to_csv(OUT / name, index=False, encoding="utf-8-sig")
    return result


def main() -> None:
    base_proxy = {
        mix: row.quality_domain
        for mix, row in MAPPING.iterrows()
        if row.has_empirical_Q
    }
    assert len(base_proxy) == 6 and len(PROXY) == 10
    full_proxy = {**base_proxy, **PROXY}
    assert len(full_proxy) == 16 and "nih_exporter" not in full_proxy
    for quality_domain in full_proxy.values():
        assert quality_domain in DOMAINS.index

    frozen = DOMAINS.Q_mean.astype(float)
    base = q0_for(base_proxy, frozen)
    assert abs(base[0] - 0.3193612774451095) < 1e-12
    assert abs(base[1] - 0.6074742221215453) < 1e-12
    mapping_rows = []
    for label, mapping in [("six_a16", base_proxy), ("semantic_proxy_16", full_proxy)]:
        coverage, q0, lower, upper = q0_for(mapping, frozen)
        mapping_rows.append(dict(scenario=label, mapped_domains=len(mapping),
                                 mapped_mass=coverage, Q0=q0,
                                 conditional_lower=lower, conditional_upper=upper))
    write_csv("mapping_sensitivity.csv", mapping_rows)

    scores = pd.read_csv(SCORES_PATH)
    scores = scores.sort_values("source", ascending=False).drop_duplicates("id", keep="first")
    assert len(scores) == 261086
    recovered = scores.groupby("domain").Q.mean().reindex(DOMAINS.index)
    assert np.max(np.abs(recovered.to_numpy() - frozen.to_numpy())) < 1e-12
    signals = scores[["score_model", "score_dsir", "score_surface"]].to_numpy(float)
    spread = signals.max(axis=1) - signals.min(axis=1)
    base_score = 0.50 * signals[:, 0] + 0.15 * signals[:, 1] + 0.35 * signals[:, 2]
    quality_rows = []
    for alpha in (0.1, 0.25, 0.5):
        adjusted = base_score * (1.0 - alpha * spread)
        assert np.all((adjusted >= 0) & (adjusted <= 1))
        mean = pd.Series(adjusted, index=scores.domain).groupby(level=0).mean().reindex(DOMAINS.index)
        coverage, q0, lower, upper = q0_for(base_proxy, mean)
        rho = float(mean.corr(frozen, method="spearman"))
        quality_rows.append(dict(scenario=f"continuous_alpha_{alpha:g}", alpha=alpha,
                                 n_documents=len(scores), Q0=q0, delta_Q0=q0-base[1],
                                 mapped_mass=coverage, rank_spearman=rho,
                                 conditional_lower=lower, conditional_upper=upper))
    write_csv("conflict_sensitivity.csv", quality_rows)

    handoff = json.loads(HANDOFF.read_text(encoding="utf-8"))
    pars = Params.from_dict(handoff["parameters"])
    scenarios = [("six_a16", base[1]), ("semantic_proxy_16", mapping_rows[1]["Q0"])]
    scenarios += [(r["scenario"], r["Q0"]) for r in quality_rows]
    q3_rows = []
    for name, q0 in scenarios:
        model = Problem([pars], 8192, "exponential", q0)
        for budget in (1e19, 1e22, 1e24):
            row = model.solve(budget, tol=1e-8)
            q3_rows.append(dict(scenario=name, budget_FLOPs=budget, Q0=q0,
                                N_B=row["N_B"], D_B=row["D_B"], Q_star=row["Q_star"],
                                objective_gap=row["optimality_gap"]))
    write_csv("q3_conditional_configurations.csv", q3_rows)

    draws = pd.read_csv(DRAW_PATH)
    draws = draws[draws.model.eq("curved")].sort_values("draw")
    assert len(draws) == 200 and draws.draw.tolist() == list(range(200))
    slope_ratio = draws.delta.to_numpy() / (5.0 * draws.beta.to_numpy())
    bound = pd.read_csv(BOUND_PATH)
    bound = bound[bound.cost_model.eq("exponential")]
    assert len(bound) == 1000
    local_summary = []
    global_rows = []
    for context in (2048, 4096, 8192, 32768, 131072):
        sub = bound[bound.context_tokens.eq(context)].sort_values("draw")
        assert len(sub) == 200 and sub.draw.tolist() == list(range(200))
        local = sub.local_boundary_candidate_FLOPs.to_numpy(float)
        local_summary.append(dict(context_tokens=context, n_draws=200,
                                  median_local_candidate_FLOPs=float(np.median(local)),
                                  p95_local_candidate_FLOPs=float(np.quantile(local, 0.95)),
                                  max_local_candidate_FLOPs=float(local.max()),
                                  below_1e19=int((local < 1e19).sum())))
        for item in draws.itertuples(index=False):
            p = Params.from_dict(item._asdict())
            model = Problem([p], context, "exponential", base[1])
            for budget in (1e19, 1e22, 1e24):
                solved = model.solve(budget, tol=1e-7)
                global_rows.append(dict(draw=int(item.draw), context_tokens=context,
                                        budget_FLOPs=budget, Q_star=solved["Q_star"],
                                        objective_gap=solved["optimality_gap"]))
        print(f"global saturation checked: context={context}, draws=200, budgets=3", flush=True)
    write_csv("local_saturation_draws_summary.csv", local_summary)
    global_df = write_csv("global_saturation_checks.csv", global_rows)
    assert len(global_df) == 3000
    assert bool(global_df.Q_star.eq(1.0).all())
    assert global_df.objective_gap.max() < 1e-6

    metadata = {
        "schema": "review-v44-conditional-sensitivity/1",
        "mapping_rule": {"six_a16": base_proxy, "semantic_proxy_extra": PROXY},
        "mapping_caveat": "Semantic proxy is a hypothetical assignment, not a measured quality value for those mixture domains.",
        "continuous_score": "(1-alpha*spread)*(0.50*M+0.15*U+0.35*R), applied to all deduplicated documents",
        "continuous_caveat": "Alternative score scale; the frozen main pipeline and B6 fit are not re-estimated.",
        "saturation_caveat": "Global checks use the same B6 curved-law draw set, fixed six-domain Q0, and exponential quality cost; no independent training evidence.",
        "gamma_bound_ratio_min": float(slope_ratio.min()),
        "gamma_bound_ratio_max": float(slope_ratio.max()),
        "gamma_bound_active_count": int((slope_ratio >= 1.0 - 1e-10).sum()),
        "global_saturation_count": int(global_df.Q_star.eq(1.0).sum()),
        "global_saturation_max_gap": float(global_df.objective_gap.max()),
        "input_sha256": {str(path.relative_to(paths.PACKAGE)).replace("\\", "/"): digest(path)
                         for path in (SCORES_PATH, HANDOFF, DRAW_PATH, BOUND_PATH,
                                      paths.Q1 / "domain_reallocation_effects.csv",
                                      paths.Q1 / "quality_to_mixture_mapping.csv",
                                      paths.Q1 / "quality_unique_domains.csv")},
    }
    (OUT / "receipt.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print("PASS: conditional mapping/conflict scenarios and 3000 global saturation checks")


if __name__ == "__main__":
    main()
