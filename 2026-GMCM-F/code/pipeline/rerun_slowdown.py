"""Reproduce the Q4 slowdown forecasts from the isolated final joint draws.

Only the half-growth, one-year exponential-cost path needs a new Q3 solve:
zero growth is each draw's saved Q3 anchor, and half growth over two years
has the same training-FLOPs target as the saved full-growth one-year path.
All point forecasts and full-growth draws are checked against the shipped outputs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

OUT = paths.SLOWDOWN
OUT.mkdir(parents=True, exist_ok=True)
Q4 = paths.Q4
Q2 = paths.Q2
Q3 = paths.CODE / "q3"

sys.path.insert(0, str(Q3))
spec = importlib.util.spec_from_file_location("q3_theory_core", Q3 / "core.py")
q3 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = q3
spec.loader.exec_module(q3)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(fit: dict, group: str, logn: float, horizon: int) -> tuple[float, float, float]:
    assert fit["model"] == "joint_size_time" and fit["names"] == ["logN", "tau"]
    coef = np.asarray(fit["coef"], float)
    center = np.asarray(fit["center"], float)
    scale = np.asarray(fit["scale"], float)
    z = (np.array([logn, float(horizon)]) - center) / scale
    offset = 2 + (0 if group == "pretrained" else 3)
    result = 100 * (z @ coef[:2] + coef[offset:offset + 3])
    if not (result[0] >= -1e-7 and result[2] <= 100 + 1e-7
            and result[0] <= result[1] + 1e-7
            and result[1] <= result[2] + 1e-7):
        raise AssertionError(f"Unordered score: {result}")
    return tuple(map(float, result))


def half_growth_one_year(p: q3.Params, q0: float, growth: float,
                         baseline: pd.Series) -> dict:
    prob = q3.Problem([p], 8192, "exponential", q0)
    f0 = float(baseline.baseline_training_FLOPs)
    target = f0 * 10 ** (0.5 * growth)
    t = target / 1e18
    k = prob.k
    h1 = prob.cost.h(1.0)
    rhs = math.log(p.alpha * p.A / (p.beta * p.B * k)) + p.beta * math.log(t / 6)

    def eq(z: float) -> float:
        return (p.alpha + p.beta + 1) * z - math.log(k * math.exp(z) + h1) - rhs

    z = brentq(eq, -40, 40, xtol=1e-13)
    n = math.exp(z)
    d = t / (6 * n)
    budget = 1e18 * d * (k * n + h1)
    sol = prob.solve(budget, tol=1e-7)
    gap = abs(math.log10(6e18 * sol["N_B"] * sol["D_B"] / target))
    if sol["Q_star"] != 1 or gap > 1e-8:
        def residual(logc: float) -> float:
            s = prob.solve(10 ** logc, tol=1e-7)
            return math.log10(6e18 * s["N_B"] * s["D_B"] / target)

        logc = brentq(residual, 20, 26, xtol=2e-12)
        sol = prob.solve(10 ** logc, tol=1e-7)
        gap = abs(residual(logc))
    if gap > 1e-8 or sol["budget_relative_residual"] > 1e-9:
        raise AssertionError(f"Q3 path mismatch: {gap}, {sol['budget_relative_residual']}")
    return {
        "N_multiplier": sol["N_B"] / float(baseline.baseline_N_B),
        "D_multiplier": sol["D_B"] / float(baseline.baseline_D_B),
        "future_Q": sol["Q_star"],
        "future_total_FLOPs": sol["budget_FLOPs"],
        "total_budget_multiplier": sol["budget_FLOPs"] / 1e22,
        "training_target_log10_residual": gap,
        "budget_relative_residual": sol["budget_relative_residual"],
        "Q3_optimality_gap": sol["optimality_gap"],
        "path_source": "new_Q3_full_quality_solve",
    }


def main() -> None:
    saved = pd.read_csv(Q4 / "joint_resource_paths.csv")
    saved = saved[saved.conversion.eq("q3_exponential")]
    fits = json.loads((Q4 / "joint_fit_diagnostics.json").read_text(encoding="utf-8"))
    fit_by_draw = {int(f["draw_id"]): f for f in fits}
    coeff = pd.read_csv(Q4 / "joint_coefficients.csv").set_index("draw_id")
    frozen_forecasts = pd.read_csv(Q4 / "joint_forecast_draws.csv")
    frozen_forecasts = frozen_forecasts[frozen_forecasts.conversion.eq("q3_exponential")]
    frozen_lookup = {(int(r.draw_id), int(r.horizon_years), r.group): r
                     for r in frozen_forecasts.itertuples(index=False)}
    point = pd.read_csv(Q4 / "forecast_scenarios.csv")
    point = point[(point.model.eq("joint_size_time")) &
                  (point.conversion.eq("q3_exponential")) &
                  (point.trend_decay.eq(0))]
    points = {(float(r.growth_factor), int(r.horizon_years), r.group): r
              for r in point.itertuples(index=False)}
    joint_q2 = pd.read_csv(Q2 / "joint_parameter_draws.csv")
    joint_q2 = joint_q2[joint_q2.model.eq("curved")].set_index("draw")
    q0 = float(saved.Q0.dropna().iloc[0])
    draw_ids = sorted(fit_by_draw)
    assert len(draw_ids) == 199 and set(draw_ids) == set(coeff.index)
    assert len(frozen_lookup) == 199 * 2 * 2 and len(points) == 3 * 2 * 2

    path_rows = []
    forecast_rows = []
    max_old_score_diff = 0.0
    max_old_resource_diff = 0.0
    for index, draw in enumerate(draw_ids):
        old = saved[saved.draw_id.eq(draw)].set_index("horizon_years")
        assert set(old.index) == {1, 2}
        base = old.loc[1]
        assert abs(float(base.baseline_total_FLOPs) / 1e22 - 1) < 1e-12
        par = q3.Params.from_dict(joint_q2.loc[draw].to_dict())
        g = float(coeff.loc[draw, "g"])
        half = half_growth_one_year(par, q0, g, base)
        fit = fit_by_draw[draw]
        for zeta in (0.0, 0.5, 1.0):
            for horizon in (1, 2):
                if zeta == 0:
                    path = {
                        "N_multiplier": 1.0, "D_multiplier": 1.0,
                        "future_Q": float(base.baseline_Q),
                        "future_total_FLOPs": 1e22,
                        "total_budget_multiplier": 1.0,
                        "training_target_log10_residual": 0.0,
                        "budget_relative_residual": 0.0,
                        "Q3_optimality_gap": float(base.Q3_optimality_gap),
                        "path_source": "saved_Q3_anchor",
                    }
                elif zeta == 0.5 and horizon == 1:
                    path = half
                else:
                    source = old.loc[1 if zeta == 0.5 else horizon]
                    path = {k: float(source[k]) for k in (
                        "N_multiplier", "D_multiplier", "future_Q",
                        "future_total_FLOPs", "total_budget_multiplier",
                        "training_target_log10_residual", "budget_relative_residual",
                        "Q3_optimality_gap")}
                    path["path_source"] = "saved_full_growth_one_year" if zeta == 0.5 else "saved_full_growth"
                check_training = path["N_multiplier"] * path["D_multiplier"]
                target_training = 10 ** (zeta * g * horizon)
                residual = abs(math.log10(check_training / target_training))
                if residual > 1e-8 or path["future_Q"] != 1:
                    raise AssertionError(f"Draw {draw}, zeta={zeta}, H={horizon}: {residual}")
                path_rows.append({"draw_id": draw, "zeta": zeta, "horizon_years": horizon,
                                  "growth_log10_per_year": g, "training_multiplier_residual": residual,
                                  **path})
                for group in ("pretrained", "posttrained"):
                    nref = float(coeff.loc[draw, "Nref_" + group])
                    logn = math.log10(nref * path["N_multiplier"])
                    low, q85, high = score(fit, group, logn, horizon)
                    forecast_rows.append({"draw_id": draw, "zeta": zeta,
                                          "horizon_years": horizon, "group": group,
                                          "N_reference_B": nref, "future_logN_B": logn,
                                          "q05": low, "q85": q85, "q95": high})
                    if zeta == 1:
                        old_pred = frozen_lookup[(draw, horizon, group)]
                        for field, value in (("q05", low), ("q85", q85), ("q95", high)):
                            max_old_score_diff = max(max_old_score_diff,
                                                     abs(value - float(getattr(old_pred, field))))
                        max_old_resource_diff = max(max_old_resource_diff,
                                                    abs(logn - float(old_pred.future_logN_B)))
        if (index + 1) % 50 == 0:
            print(f"Solved/checked {index + 1}/{len(draw_ids)} draws", flush=True)

    if max_old_score_diff > 1e-9 or max_old_resource_diff > 1e-10:
        raise AssertionError("Frozen full-growth reconstruction mismatch")
    paths = pd.DataFrame(path_rows)
    draws = pd.DataFrame(forecast_rows)
    paths.to_csv(OUT / "q4_slowdown_resource_draws.csv", index=False)
    draws.to_csv(OUT / "q4_slowdown_forecast_draws.csv", index=False)
    ci = draws.groupby(["zeta", "horizon_years", "group"], sort=True).q85.agg(
        curve_5=lambda x: x.quantile(0.05),
        curve_95=lambda x: x.quantile(0.95),
        curve_median="median", draw_count="size").reset_index()
    data = []
    for row in ci.itertuples(index=False):
        point_row = points[(row.zeta, row.horizon_years, row.group)]
        data.append({"zeta": row.zeta, "horizon_years": row.horizon_years,
                     "group": row.group, "point_q85": float(point_row.q85),
                     "curve_5": row.curve_5, "curve_95": row.curve_95,
                     "individual_q05": float(point_row.q05),
                     "individual_q95": float(point_row.q95),
                     "draw_count": row.draw_count})
    table = pd.DataFrame(data).sort_values(["group", "horizon_years", "zeta"])
    table.to_csv(OUT / "q4_slowdown_forecast_table.csv", index=False)
    manifest = {
        "status": "PASS", "valid_draws": len(draw_ids),
        "new_global_Q3_solves": len(draw_ids),
        "forecast_rows": len(draws), "resource_rows": len(paths),
        "max_frozen_score_difference": max_old_score_diff,
        "max_frozen_resource_logN_difference": max_old_resource_diff,
        "max_new_Q3_optimality_gap": float(paths[paths.path_source.eq("new_Q3_full_quality_solve")].Q3_optimality_gap.max()),
        "max_training_multiplier_residual": float(paths.training_multiplier_residual.max()),
        "max_budget_relative_residual": float(paths.budget_relative_residual.max()),
        "input_sha256": {str(p.relative_to(paths.PACKAGE)).replace("\\", "/"): digest(p) for p in (
            Q4 / "joint_resource_paths.csv", Q4 / "joint_fit_diagnostics.json",
            Q4 / "joint_coefficients.csv", Q4 / "joint_forecast_draws.csv",
            Q4 / "forecast_scenarios.csv", Q2 / "joint_parameter_draws.csv")},
        "interpretation": "Fixed 2025-03-13 origin; same 199 retained publisher draws; conditional retrospective scenarios, not future validation.",
    }
    (OUT / "q4_slowdown_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
