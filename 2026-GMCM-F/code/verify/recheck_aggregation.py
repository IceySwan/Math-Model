# -*- coding: utf-8 -*-
"""Q4 aggregation check (READ-ONLY on the shipped per-record predictions).

The paper weights only the two joint candidates on the development window.  This
script repeats the same development-period weighting with the ordered auxiliary trend
baseline added to the candidate set, using the shipped per-record predictions
(rolling_with_baselines.csv), and reports the resulting weights and the development /
reused-late / straddling Pinball losses.

Writes only into data/reports/rechecks/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

OUT = paths.REPORTS / "rechecks"
OUT.mkdir(parents=True, exist_ok=True)
RB = paths.Q4 / "rolling_with_baselines.csv"
TAU = 0.85


def pinball(y, q, tau=TAU):
    e = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.where(e >= 0, tau * e, (tau - 1) * e)))


d = pd.read_csv(RB)
d = d[d.model.isin(["joint_size_time", "joint_time", "auxiliary_size_time_ordered"])]
d = d[d.input_mode == "resource_forecast"]
print(d.groupby(["role", "model"]).size().to_string())

roles = {"development": "开发期", "reused_late": "复用晚期", "straddling": "跨分界"}
order = ["joint_size_time", "joint_time", "auxiliary_size_time_ordered"]
labels = ["联合规模--时间", "联合时间", "有序辅助趋势"]

# align rows across models on (role, cutoff, end, repo, months)
key = ["role", "cutoff", "end", "repo", "months"]
parts = []
for i, m in enumerate(order):
    sub = d[d.model == m][key + ["prediction", "actual"]].copy()
    sub = sub.rename(columns={"prediction": f"p{i}", "actual": "y"})
    parts.append(sub.set_index(key))
wide = pd.concat(parts, axis=1, join="inner")
wide = wide[~wide.index.duplicated()]
y = wide["y"]
y = np.asarray(y.iloc[:, 0] if hasattr(y, "columns") else y, float)
P = np.column_stack([wide[f"p{i}"].to_numpy(float) for i in range(len(order))])
act = y
role = wide.index.get_level_values("role")
mask_dev = role == "development"
print("aligned rows:", P.shape, "| development rows:", int(mask_dev.sum()))
print(pd.Series(role).value_counts().to_string())

# side conditions: quantiles must stay ordered across the three candidates
best = None
grid = np.arange(0, 1.0001, 0.01)
for w1 in grid:
    for w2 in grid:
        if w1 + w2 > 1 + 1e-12:
            continue
        w = np.array([w1, w2, 1 - w1 - w2])
        v = pinball(act[mask_dev], P[mask_dev] @ w)
        if best is None or v < best[0] - 1e-15:
            best = (v, w.copy())
# local refinement on the simplex
v0, w0 = best
for _ in range(4):
    step = 0.01 if _ == 0 else step / 5
    for i in range(3):
        for s in (-1, 1):
            w = w0.copy()
            w[i] += s * step
            w = np.clip(w, 0, None)
            w = w / w.sum()
            v = pinball(act[mask_dev], P[mask_dev] @ w)
            if v < v0 - 1e-15:
                v0, w0 = v, w
print("development-optimal weights:", dict(zip(labels, np.round(w0, 4))),
      "| dev pinball = %.6f" % v0)

rows = []
for r, lab in roles.items():
    m = role == r
    rec = {"role": r, "n": int(m.sum())}
    for i, name in enumerate(labels):
        rec[f"pinball_{name}"] = pinball(act[m], P[m, i])
    rec["pinball_aggregate"] = pinball(act[m], P[m] @ w0)
    rows.append(rec)
res = pd.DataFrame(rows)
print(res.to_string(index=False))

# two-candidate reference (the paper's current rule)
P2 = np.column_stack([P[:,0], P[:,1]])
best2 = None
for w in np.arange(0, 1.0001, 0.001):
    v = pinball(act[mask_dev], P2[mask_dev] @ np.array([w, 1 - w]))
    if best2 is None or v < best2[0]:
        best2 = (v, w)
print("\ntwo-candidate reference: weight on joint_size_time = %.3f, dev pinball = %.6f" % (best2[1], best2[0]))

res.to_csv(OUT / "q4_aggregation_with_baseline.csv", index=False, encoding="utf-8-sig")
json.dump({
    "candidates": labels,
    "weights_development_optimal": {k: float(v) for k, v in zip(labels, w0)},
    "development_pinball_aggregate": float(v0),
    "two_candidate_rule": {"weight_joint_size_time": float(best2[1]),
                           "development_pinball": float(best2[0])},
    "per_role": res.to_dict(orient="records"),
    "note": "weights fitted on the 16 development predictions only; late/straddling shown as out-of-sample checks",
}, open(OUT / "q4_aggregation_with_baseline.json", "w", encoding="utf-8"),
    ensure_ascii=False, indent=2)
print(f"PASS: three-candidate development weighting = "
      f"{ {k: float(v) for k, v in zip(labels, w0)} } with dev pinball {v0:.6f}; "
      f"two-candidate reference {best2[0]:.6f}")
print("\nwrote ->", OUT)

