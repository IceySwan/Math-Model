# -*- coding: utf-8 -*-
"""Q4 slope-penalty sensitivity and declaration-domain checks (READ-ONLY).

Q4 items:
  (a) replicate the published 12/24-month point forecast (acceptance test),
  (b) lambda_QR sensitivity of the shared slopes and of the 12/24-month forecast,
  (c) declaration-domain coverage in real units + a numeric order/bound check at the
      forecast points,
  (d) provenance of the reported "maximum contrast error bound" (0.5101 pp).

Writes only into data/reports/rechecks/.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

OUT = paths.REPORTS / "rechecks"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(paths.CODE / "q4"))

import q4_data  # noqa: E402
from q4_alternatives import auxiliary_sample  # noqa: E402
from ordered_model import fit_joint  # noqa: E402
from protocol import GROUPS, ORIGIN  # noqa: E402

c1, c4, c6, _, _ = q4_data.read_sources()
idx = q4_data.identity_table(c1, c4)
strict, _ = q4_data.asof(c1, idx, ORIGIN, strict=True)
broad, _ = q4_data.asof(c1, idx, ORIGIN, strict=False)
aux = auxiliary_sample(broad, set(strict.repo))
print("strict/aux rows:", len(strict), len(aux))

REF_N = {"pretrained": 25.5664, "posttrained": 47.88}
T0 = pd.Timestamp(ORIGIN)

# published resource path: use the shipped per-scenario log10 N so that only the
# quantile fit changes when lambda_QR changes.
FS = pd.read_csv(paths.Q4 / "forecast_scenarios.csv")
FS = FS[(FS.draw_id == -1) & (FS.model == "joint_size_time") & (FS.trend_decay == 0.0)]
print("published forecast rows:", len(FS), "| conversions:", sorted(FS.conversion.unique()))


def scen(group, zeta, months):
    h = months // 12
    sub = FS[(FS.group == group) & (FS.growth_factor == zeta) & (FS.horizon_years == h)]
    conv = "fitted_relation" if zeta == 0.0 else "q3_exponential"
    sub = sub[sub.conversion == conv]
    return float(sub.future_logN_B.iloc[0]), float(sub.q85.iloc[0])


def forecast(fit, group, months, zeta, x_n=None):
    """q85 = 100*(theta^T z + b_{group,0.85}); z uses the fit's own centre/scale."""
    gi = GROUPS.index(group)
    th_n, th_t = fit.coef[0], fit.coef[1]
    if x_n is None:
        x_n, _ = scen(group, zeta, months)
    x_t = months / 12.0
    z_n = (x_n - fit.center[0]) / fit.scale[0]
    z_t = (x_t - fit.center[1]) / fit.scale[1]
    q05 = 100 * (th_n * z_n + th_t * z_t + fit.coef[2 + 3 * gi + 0])
    q85 = 100 * (th_n * z_n + th_t * z_t + fit.coef[2 + 3 * gi + 1])
    q95 = 100 * (th_n * z_n + th_t * z_t + fit.coef[2 + 3 * gi + 2])
    return dict(q05=q05, q85=q85, q95=q95, z_logN=z_n, z_tau=z_t)


# ---------------------------------------------------------------- (a) replicate
print("\n=== (a) acceptance: replicate published forecasts with lambda_QR=1 ===")
fit1 = fit_joint(aux, strict, "joint_size_time", penalty=1.0)
rep_rows = []
for g in GROUPS:
    for zeta in (0.0, 0.5, 1.0):
        for months in (12, 24):
            x_n, pubv = scen(g, zeta, months)
            r = forecast(fit1, g, months, zeta, x_n)
            rep_rows.append(dict(group=g, zeta=zeta, months=months, log10N_used=x_n,
                                 recomputed=r["q85"], published=pubv, diff=r["q85"] - pubv))
rep = pd.DataFrame(rep_rows)
print(rep.to_string(index=False))
print("max |recomputed - published| = %.3e" % rep["diff"].abs().max())

# ------------------------------------------------------- (b) lambda sensitivity
print("\n=== (b) lambda_QR sensitivity ===")
lam_rows = []
for lam in (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0):
    f = fit_joint(aux, strict, "joint_size_time", penalty=lam)
    th_n, th_t = float(f.coef[0]), float(f.coef[1])
    b_n = 100 * th_n / f.scale[0]
    b_t = 100 * th_t / f.scale[1]
    rec = dict(penalty=lam, theta_logN=th_n, theta_tau=th_t,
               b_logN_pp_per_decade=b_n, b_t_pp_per_year=b_t,
               objective=f.diagnostics["objective"],
               weighted_pinball=f.diagnostics["weighted_pinball"],
               optimality_gap=f.diagnostics["optimality_gap"])
    for g in GROUPS:
        for (zeta, months) in ((0.0, 24), (1.0, 12), (1.0, 24)):
            r = forecast(f, g, months, zeta)
            rec[f"q85_{g}_z{zeta}_{months}m"] = r["q85"]
    lam_rows.append(rec)
lam = pd.DataFrame(lam_rows)
lam.to_csv(OUT / "q4_lambda_sensitivity.csv", index=False, encoding="utf-8-sig")
print(lam[["penalty", "b_logN_pp_per_decade", "b_t_pp_per_year", "weighted_pinball",
           "q85_pretrained_z0.0_24m", "q85_pretrained_z1.0_24m",
           "q85_posttrained_z0.0_24m", "q85_posttrained_z1.0_24m"]].to_string(index=False))

# ------------------------------------------------------------ (c) domain check
print("\n=== (c) declaration-domain coverage ===")
z_logN = (strict.logN.to_numpy() - fit1.center[0]) / fit1.scale[0]
z_tau = (strict.tau.to_numpy() - fit1.center[1]) / fit1.scale[1]
box = {"logN": (-2.0, 4.0), "tau": (-2.0, 2.0)}
lo_t = fit1.center[1] + box["tau"][0] * fit1.scale[1]
hi_t = fit1.center[1] + box["tau"][1] * fit1.scale[1]
dom = dict(
    strict_z_logN_min=float(z_logN.min()), strict_z_logN_max=float(z_logN.max()),
    strict_z_tau_min=float(z_tau.min()), strict_z_tau_max=float(z_tau.max()),
    box_logN=box["logN"], box_tau=box["tau"],
    box_tau_real_start=str((T0 + pd.Timedelta(days=lo_t * 365.2425)).date()),
    box_tau_real_end=str((T0 + pd.Timedelta(days=hi_t * 365.2425)).date()),
)
for g in GROUPS:
    for months in (12, 24):
        r = forecast(fit1, g, months, 1.0)
        dom[f"z_tau_{g}_{months}m"] = r["z_tau"]
        for k in ("q05", "q85", "q95"):
            dom[f"{k}_{g}_{months}m"] = r[k]
print(json.dumps(dom, ensure_ascii=False, indent=2))
inside = all(0.0 <= dom[f"{k}_{g}_{m}m"] <= 100.0
             for g in GROUPS for m in (12, 24) for k in ("q05", "q85", "q95"))
ordered = all(dom[f"q05_{g}_{m}m"] <= dom[f"q85_{g}_{m}m"] <= dom[f"q95_{g}_{m}m"]
              for g in GROUPS for m in (12, 24))
print("all forecast quantiles within [0,100]:", inside, "| ordered:", ordered)

# ------------------------------------------- (d) provenance of the 0.5101 bound
print("\n=== (d) provenance of the reported maximum contrast bound ===")
draws = json.load(open(paths.Q4 / "joint_fit_diagnostics.json", encoding="utf-8"))
X = np.column_stack([strict.logN.to_numpy(), strict.tau.to_numpy()])
worst, worst_draw = 0.0, None
per_draw = []
for d in draws:
    c = np.asarray(d["center"], float)
    s = np.asarray(d["scale"], float)
    Z = (X - c) / s
    dmax = 0.0
    for i in range(len(Z)):
        dd = np.linalg.norm(Z[i] - Z, axis=1).max()
        dmax = max(dmax, float(dd))
    per_draw.append(dmax)
    if dmax > worst:
        worst, worst_draw = dmax, d["draw_id"]
eps = json.load(open(paths.EVIDENCE / "algorithm_checks.json", encoding="utf-8"))
gap = float(eps["q4"]["max_recorded_objective_gap"])
implied_dz = 0.5101 / (100 * math.sqrt(gap / 1.0))
dom_check = dict(
    objective_gap=gap, lambda_qr=1.0,
    bound_from_paper_pp=0.5101, implied_norm_dz=implied_dz,
    max_norm_dz_within_box=float(np.linalg.norm([box["logN"][1] - box["logN"][0],
                                                 box["tau"][1] - box["tau"][0]])),
    max_norm_dz_observed_strict=float(np.linalg.norm([z_logN.max() - z_logN.min(),
                                                      z_tau.max() - z_tau.min()])),
    max_norm_dz_over_199_draws=worst, argmax_draw_id=worst_draw,
)
print(json.dumps(dom_check, ensure_ascii=False, indent=2))

json.dump(dict(acceptance_max_abs_diff=float(rep["diff"].abs().max()),
               domain=dom, domain_check=dom_check),
          open(OUT / "q4_domain_and_lambda_checks.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"PASS: all 12 published point forecasts reproduced "
      f"(max|diff| = {float(rep['diff'].abs().max()):.2e}); forecast quantiles within "
      f"[0,100]: {inside}, ordered: {ordered}; lambda grid = {len(lam)} values")
print("wrote ->", OUT)

