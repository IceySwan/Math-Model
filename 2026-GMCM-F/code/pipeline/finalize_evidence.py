"""Freeze the final numerical interfaces and regenerate evidence for main claims.

Run after the isolated Q1 rerun. This script reads the shipped Q2/Q3/Q4 outputs
without modifying them. It writes all new evidence into data/outputs/evidence.
Numerical examples support the stated theorems; they do not replace proofs.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

OUT = paths.EVIDENCE
OUT.mkdir(parents=True, exist_ok=True)
Q1 = paths.Q1
RBF = paths.VALIDATION / "q1_rbf_comparison"
Q2 = paths.Q2
Q3 = paths.Q3
Q4 = paths.Q4
PAPER = paths.PAPER


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(name: str, value) -> None:
    (OUT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, tol=1e-10) -> bool:
    return bool(np.isclose(a, b, rtol=tol, atol=tol))


def q1_evidence() -> dict:
    selection = read_json(Q1 / "solver_selection.json")
    handoff = read_json(Q1 / "q2_handoff_metadata.json")
    rbf_handoff = read_json(RBF / "q2_handoff_metadata.json")
    assert selection["selected_model"] == "sqrt_quadratic_a01"
    assert handoff["selected_model"] == "sqrt_quadratic_a01"
    assert rbf_handoff["selected_model"] == "sqrt_rbf_a001_g05"
    assert handoff["anchor_index"] == rbf_handoff["anchor_index"] == 170
    assert handoff["selected_candidate"]["index"] == 239
    assert handoff["output_target"] == rbf_handoff["output_target"]

    hist = pd.read_csv(Q1 / "historical_predictions.csv")
    h = hist.loc[
        hist.scale.eq("1M")
        & hist.provenance.eq("measured")
        & hist.model.eq("sqrt_quadratic_a01")
    ].sort_values("index")
    assert len(h) == 256 and h["index"].is_unique
    actual = h.observed_J_fixed_a5.to_numpy(float)
    predicted = h.predicted_J_fixed_a5.to_numpy(float)
    epsilon = float(np.max(np.abs(actual - predicted)))
    chosen = int(h.iloc[np.argmin(predicted)]["index"])
    oracle = int(h.iloc[np.argmin(actual)]["index"])
    regret = float(actual[np.argmin(predicted)] - actual[np.argmin(actual)])
    assert chosen == oracle == 239 and close(regret, 0)
    assert close(epsilon, 0.048135492002841174)

    mapping = pd.read_csv(Q1 / "quality_to_mixture_mapping.csv")
    anchor = pd.read_csv(Q1 / "domain_reallocation_effects.csv")
    mapped = anchor.merge(
        mapping, left_on="domain", right_on="mixture_domain",
        how="left", validate="one_to_one",
    )
    known = mapped.has_empirical_Q.fillna(False).astype(bool)
    mapped_mass = float(mapped.loc[known, "anchor_share"].sum())
    quality_lower = float(
        (mapped.loc[known, "anchor_share"] * mapped.loc[known, "Q_from_quality_domain"]).sum()
    )
    quality_upper = quality_lower + 1 - mapped_mass
    q0 = quality_lower / mapped_mass
    assert close(q0, 0.6074742221215453)

    keys = ["point_kind", "domain_i", "domain_j"]
    a = pd.read_csv(Q1 / "response_surface_evaluations.csv")
    b = pd.read_csv(RBF / "response_surface_evaluations.csv")
    for frame in (a, b):
        frame[["domain_i", "domain_j"]] = frame[["domain_i", "domain_j"]].fillna("")
    a = a.loc[a.model.eq("selected"), keys + ["loss_change_vs_anchor"]]
    b = b.loc[b.model.eq("selected"), keys + ["loss_change_vs_anchor"]]
    a = a.rename(columns={"loss_change_vs_anchor": "sqrt_quadratic_change"})
    b = b.rename(columns={"loss_change_vs_anchor": "sqrt_rbf_change"})
    candidates = a.merge(b, on=keys, validate="one_to_one")
    assert len(candidates) == 155
    candidates["worst_model_change"] = candidates[
        ["sqrt_quadratic_change", "sqrt_rbf_change"]
    ].max(axis=1)
    candidates["safe_in_declared_pair"] = candidates.worst_model_change < -1e-12
    assert int(candidates.safe_in_declared_pair.sum()) == 66
    candidates.to_csv(OUT / "q1_two_model_candidates.csv", index=False, float_format="%.16g")

    pieces = []
    for directory, label in ((Q1, "sqrt_quadratic"), (RBF, "sqrt_rbf")):
        singles = pd.read_csv(directory / "domain_reallocation_effects_selected.csv")
        singles = singles.rename(
            columns={"domain": "domain_i", "loss_change_5pct_reallocation": "effect"}
        )
        singles["domain_j"] = ""
        pairs = pd.read_csv(directory / "domain_pair_interactions_selected.csv")
        pairs = pairs.rename(columns={"interaction_loss": "effect"})
        effects = pd.concat(
            [singles[["domain_i", "domain_j", "effect"]],
             pairs[["domain_i", "domain_j", "effect"]]],
            ignore_index=True,
        )
        boot = pd.read_csv(directory / "effect_bootstrap_selected.csv")
        boot["domain_j"] = boot.domain_j.fillna("")
        effects = effects.merge(
            boot[["domain_i", "domain_j", "effect_ci025", "effect_ci975"]],
            on=["domain_i", "domain_j"], validate="one_to_one",
        )
        effects = effects.rename(
            columns={c: label + "_" + c for c in
                     ["effect", "effect_ci025", "effect_ci975"]}
        )
        pieces.append(effects)
    compare = pieces[0].merge(pieces[1], on=["domain_i", "domain_j"], validate="one_to_one")
    assert len(compare) == 153
    compare["kind"] = np.where(compare.domain_j.eq(""), "single", "pair")
    s1 = np.sign(compare.sqrt_quadratic_effect)
    s2 = np.sign(compare.sqrt_rbf_effect)
    compare["sign_agrees"] = s1.eq(s2)
    qci = (compare.sqrt_quadratic_effect_ci025 * compare.sqrt_quadratic_effect_ci975 > 0)
    rci = (compare.sqrt_rbf_effect_ci025 * compare.sqrt_rbf_effect_ci975 > 0)
    compare["both_intervals_same_side"] = (
        compare.sign_agrees & qci & rci
        & (np.sign(compare.sqrt_quadratic_effect_ci025) == s1)
        & (np.sign(compare.sqrt_rbf_effect_ci025) == s2)
    )
    single_same = int(compare.loc[compare.kind.eq("single"), "sign_agrees"].sum())
    pair_same = int(compare.loc[compare.kind.eq("pair"), "sign_agrees"].sum())
    pair_strong = int(compare.loc[compare.kind.eq("pair"), "both_intervals_same_side"].sum())
    assert (single_same, pair_same, pair_strong) == (12, 112, 15)
    compare.to_csv(OUT / "q1_cross_model_effects.csv", index=False, float_format="%.16g")

    return {
        "selected_model": "sqrt_quadratic_a01",
        "comparison_model": "sqrt_rbf_a001_g05",
        "candidate_count": 256,
        "selected_index": chosen,
        "oracle_index": oracle,
        "anchor_index": 170,
        "historical_uniform_error": epsilon,
        "two_epsilon_bound": 2 * epsilon,
        "observed_regret": regret,
        "mapped_mass_anchor": mapped_mass,
        "Q0_conditional_mapped": q0,
        "full_quality_lower": quality_lower,
        "full_quality_upper": quality_upper,
        "declared_two_model_candidates": len(candidates),
        "declared_two_model_safe": int(candidates.safe_in_declared_pair.sum()),
        "single_sign_agreement": single_same,
        "pair_sign_agreement": pair_same,
        "pair_strong_interval_agreement": pair_strong,
        "scope": "historical finite candidate set; not prospective uniform error or causal effect",
    }


def q2_evidence(q0: float) -> dict:
    selected = read_json(Q2 / "model_selection.json")
    pars = read_json(Q2 / "full_fit_parameters.json")["curved"]
    handoff = read_json(Q2 / "q2_handoff_v26.json")
    assert selected["selected_model"] == handoff["selected_model"] == "curved"
    assert close(handoff["Q1_anchor_Q"], q0, 1e-12)
    assert close(pars["delta_min"], 0)
    q = np.geomspace(0.1, 1, 401)
    gamma = (pars["delta"] - 2 * pars["kappa"] * np.log(q)) / pars["beta"]
    assert gamma.min() >= -1e-10 and gamma.max() <= 5 + 1e-10

    def g(value):
        u = math.log(value)
        return -pars["delta"] * u + pars["kappa"] * u * u

    q1, q2 = 0.5, 0.8
    z1, z2, lam = 0.01, -0.01, 1.3
    d1 = 100.0
    logG1 = -g(q1) / pars["beta"]
    logG2 = -g(q2) / pars["beta"]
    d2 = d1 * math.exp(-(logG2 - logG1) + lam * (z2 - z1))
    eff1 = d1 * math.exp(logG1 - lam * z1)
    eff2 = d2 * math.exp(logG2 - lam * z2)
    assert close(eff1, eff2, 1e-13)
    n = 1.5
    def loss(d, quality, psi):
        return (pars["E"] + pars["A"] * n ** (-pars["alpha"])
                + pars["B"] * d ** (-pars["beta"])
                * math.exp(g(quality) + pars["beta"] * lam * psi))
    assert close(loss(d1, q1, z1), loss(d2, q2, z2), 1e-13)
    comparison = selected["primary_comparisons"]
    assert comparison["leave_Q"]["passes"] and comparison["leave_cell"]["passes"]
    assert comparison["leave_Q"]["curved_rmse"] < comparison["leave_Q"]["power_rmse"]
    assert comparison["leave_cell"]["curved_rmse"] < comparison["leave_cell"]["power_rmse"]
    transfer = pd.read_csv(Q2 / "q1_transfer_sensitivity.csv")
    arxiv = transfer.loc[
        transfer.model.eq("power") & transfer.quality_baseline.eq("doc_equal")
        & transfer.domain.eq("arxiv") & transfer.lambda_p.eq(0)
    ].iloc[0]
    q1_effects = pd.read_csv(Q1 / "domain_reallocation_effects_selected.csv")
    q1_arxiv = float(q1_effects.loc[q1_effects.domain.eq("arxiv"), "loss_change_5pct_reallocation"].iloc[0])
    assert close(arxiv.psi, q1_arxiv, 1e-12)
    return {
        "selected_model": "curved",
        "Q0": q0,
        "gamma_inf_fitted_full_support": float(gamma.min()),
        "gamma_sup_fitted_full_support": float(gamma.max()),
        "gamma_at_Q0": float(
            (pars["delta"] - 2 * pars["kappa"] * math.log(q0)) / pars["beta"]
        ),
        "gamma_scope": "Full-support extrema are on Q in [0.1,1]; Q3 optimizes on [Q0,1].",
        "finite_equivalence_d2": d2,
        "finite_equivalence_loss_error": abs(loss(d1, q1, z1) - loss(d2, q2, z2)),
        "leave_Q_relative_gain": comparison["leave_Q"]["relative_gain"],
        "leave_cell_relative_gain": comparison["leave_cell"]["relative_gain"],
        "Q1_arxiv_psi": q1_arxiv,
        "scope": "B6 source conditional response; finite identity is algebraic, validation is retrospective",
    }


def q3_evidence(q0: float) -> dict:
    rows = pd.read_csv(Q3 / "nominal_configurations.csv")
    assert len(rows) == 45 and np.allclose(rows.Q0, q0)
    assert rows.Q_star.eq(1).all()
    assert rows.budget_relative_residual.abs().max() < 1e-8
    assert rows.optimality_gap.min() >= -1e-8
    assert rows.optimality_gap.max() < 1e-5
    scan = pd.read_csv(Q3 / "budget_scan.csv")
    worst_drop = 0.0
    for _, group in scan.groupby(["context_tokens", "cost_model", "cost_scale"]):
        group = group.sort_values("budget_FLOPs")
        worst_drop = min(worst_drop, float(np.diff(group.Q_star).min()))
    assert worst_drop > -1e-5

    sys.path.insert(0, str(paths.CODE / "q3"))
    from core import Params, Problem

    handoff = read_json(Q2 / "q2_handoff_v26.json")
    problem = Problem([Params.from_dict(handoff["parameters"])], 8192, "exponential", q0)
    fresh = problem.solve(1e19, tol=2e-8)
    saved = rows.loc[
        rows.context_tokens.eq(8192) & rows.cost_model.eq("exponential")
        & np.isclose(rows.budget_FLOPs, 1e19)
    ].iloc[0]
    assert close(fresh["Q_star"], saved.Q_star, 1e-6)
    assert close(fresh["N_B"], saved.N_B, 1e-5)
    assert close(fresh["D_B"], saved.D_B, 1e-5)
    transitions = pd.read_csv(Q3 / "saturation_transitions.csv")
    checked = transitions.loc[transitions.classification.eq("continuous_candidate_globally_checked")]
    assert (checked.left_Q < 1).all() and checked.right_Q.eq(1).all()
    return {
        "nominal_scenarios": len(rows),
        "nominal_upper_quality_count": int(rows.Q_star.eq(1).sum()),
        "max_budget_relative_residual": float(rows.budget_relative_residual.abs().max()),
        "max_nominal_optimality_gap": float(rows.optimality_gap.max()),
        "budget_scan_worst_quality_drop": worst_drop,
        "representative_resolved_N_B": float(fresh["N_B"]),
        "representative_resolved_D_B": float(fresh["D_B"]),
        "checked_continuous_transition_rows": len(checked),
        "scope": "finite numerical certificates on declared scenarios; global theorems require their analytic hypotheses",
    }


def q4_evidence(q0: float) -> dict:
    selected = read_json(Q4 / "selection.json")
    assert selected["selected"] == "joint_size_time"
    assert selected["independent_holdout"] is False
    assert close(read_json(Q4 / "upstream_contract.json")["Q0"], q0, 1e-12)
    ordinary = read_json(Q4 / "fit_diagnostics.json")
    joint = read_json(Q4 / "joint_fit_diagnostics.json")
    full = read_json(Q4 / "full_fit.json")
    assert (len(ordinary), len(joint)) == (79, 199)
    all_fits = ordinary + joint + [full]
    gaps = [float(r["diagnostics"]["optimality_gap"]) for r in all_fits]
    assert len(gaps) == 279 and max(gaps) < 1e-7

    sys.path.insert(0, str(paths.CODE / "q4"))
    from ordered_model import JointFit
    from protocol import BOX, GROUPS

    fit = JointFit(
        full["model"], np.array(full["coef"]), np.array(full["center"]),
        np.array(full["scale"]), full["names"], np.array(full["aux_slope"]),
        full["diagnostics"],
    )
    vertices = list(itertools.product(*(BOX[name] for name in fit.names)))
    vertex_frame = pd.DataFrame([
        {**dict(zip(fit.names, vertex)), "group": group}
        for group in GROUPS for vertex in vertices
    ])
    predictions = fit.matrix(vertex_frame)
    assert predictions.min() >= -1e-7 and predictions.max() <= 100 + 1e-7
    assert (np.diff(predictions, axis=1) >= -1e-7).all()
    metrics = pd.read_csv(Q4 / "rolling_metrics.csv")
    role = metrics.loc[
        metrics.role.eq("reused_late") & metrics.input_mode.eq("resource_forecast")
        & metrics.model.eq("joint_size_time") & metrics.weighting.eq("row")
    ].iloc[0]
    assert close(role.pinball85, 1.189958, 1e-6)
    assert close(role.coverage90, 0.75)
    raw = pd.read_csv(
        paths.C / "epoch_all_ai_models.csv",
        usecols=["Domain", "Open model weights?", "Publication date",
                 "Training compute (FLOP)", "Parameters", "Training dataset size (total)"],
        low_memory=False,
    )
    c4 = raw.loc[
        raw.Domain.eq("Language")
        & raw["Open model weights?"].fillna("").str.lower().eq("yes")
        & pd.to_numeric(raw["Training compute (FLOP)"], errors="coerce").gt(0)
    ]
    all_date_count = len(c4)
    c4 = c4.loc[pd.to_datetime(c4["Publication date"], errors="coerce").le("2025-03-13")]
    complete = c4.loc[
        pd.to_numeric(c4.Parameters, errors="coerce").gt(0)
        & pd.to_numeric(c4["Training dataset size (total)"], errors="coerce").gt(0)
    ]
    assert (all_date_count, len(c4), len(complete)) == (426, 346, 247)
    return {
        "selected_model": selected["selected"],
        "fit_count_without_full": len(ordinary) + len(joint),
        "fit_count_with_full": len(all_fits),
        "max_recorded_objective_gap": max(gaps),
        "vertex_predictions_min": float(predictions.min()),
        "vertex_predictions_max": float(predictions.max()),
        "vertex_min_quantile_spacing": float(np.diff(predictions, axis=1).min()),
        "reused_late_pinball85": float(role.pinball85),
        "reused_late_coverage90": float(role.coverage90),
        "C4_before_date_filter": all_date_count,
        "C4_asof_origin": len(c4),
        "C4_complete_N_D": len(complete),
        "scope": "numerical QP/shape evidence and reused retrospective validation; future coverage requires exchangeability",
    }


def source_manifest() -> dict:
    files = [
        PAPER / "main.tex",
        PAPER / "content" / "04_question1.tex",
        PAPER / "content" / "05_question2.tex",
        PAPER / "content" / "06_question3.tex",
        PAPER / "content" / "07_question4.tex",
        paths.CODE / "q1" / "solve_q1.py",
        paths.CODE / "q1" / "q1_pipeline.py",
        paths.CODE / "q1" / "mixture_solver.py",
        paths.CODE / "q1" / "mixture_effects.py",
        paths.CODE / "q2" / "quality_solver.py",
        paths.CODE / "q2" / "run_study.py",
        paths.CODE / "q3" / "core.py",
        paths.CODE / "q3" / "run_study.py",
        paths.CODE / "q4" / "ordered_model.py",
        paths.CODE / "q4" / "run_study.py",
        Q1 / "solver_selection.json",
        Q1 / "historical_predictions.csv",
        RBF / "historical_predictions.csv",
        Q2 / "q2_handoff_v26.json",
        Q3 / "nominal_configurations.csv",
        Q4 / "full_fit.json",
    ]
    return {
        str(path.relative_to(paths.PACKAGE)).replace("\\", "/"): {
            "sha256": digest(path), "bytes": path.stat().st_size
        }
        for path in files
    }


def main() -> None:
    sections = {}
    for name, fn in (
        ("q1", lambda: q1_evidence()),
        ("q2", lambda: q2_evidence(sections["q1"]["Q0_conditional_mapped"])),
        ("q3", lambda: q3_evidence(sections["q1"]["Q0_conditional_mapped"])),
        ("q4", lambda: q4_evidence(sections["q1"]["Q0_conditional_mapped"])),
    ):
        sections[name] = fn()
        print(name.upper(), "PASS", flush=True)
    save_json("algorithm_checks.json", sections)
    save_json("source_manifest.json", source_manifest())
    contract = {
        "schema": "final-algorithms/1",
        "q1": {
            "quality_program": "code/q1/solve_q1.py",
            "selected_model": sections["q1"]["selected_model"],
            "comparison_model": sections["q1"]["comparison_model"],
            "anchor_index": 170,
            "historical_candidate_index": 239,
            "loss_target": "equal mean raw loss divided by full A5 training mean per domain",
        },
        "q2": {"selected_model": "curved", "Q0": sections["q2"]["Q0"],
               "mixture_psi": "J_hat(p)-J_hat(p0)"},
        "q3": {"quality_model": "curved", "main_mixture": "p0",
               "resource_problem": "N,D,Q with Q in [Q0,1]"},
        "q4": {"selected_model": "joint_size_time",
               "resource_transfer": "training FLOPs only",
               "historical_origin": "2025-03-13"},
        "theorem_evidence": {
            "q1_partial_identification": "q1.mapped_mass_anchor and full_quality_lower/upper",
            "q1_regret_bound": "q1.historical_uniform_error and two_epsilon_bound",
            "q1_model_set_safe_selection": "q1_two_model_candidates.csv",
            "q2_finite_equivalence": "q2.finite_equivalence_loss_error",
            "q2_shape_and_validation": "q2.gamma_inf_fitted_full_support, gamma_sup_fitted_full_support, gamma_at_Q0 and leave_Q/leave_cell_relative_gain",
            "q3_unique_scale_and_global_comparison": "q3.representative_resolved_N_B/D_B and max_nominal_optimality_gap",
            "q3_monotonicity_saturation": "q3.budget_scan_worst_quality_drop and checked_continuous_transition_rows",
            "q4_ordered_QP": "q4.vertex_min_quantile_spacing and max_recorded_objective_gap",
            "q4_conditional_coverage": "q4.reused_late_coverage90; assumption not empirically verified",
        },
    }
    save_json("final_algorithm_contract.json", contract)
    print("Evidence written to", OUT)


if __name__ == "__main__":
    main()


