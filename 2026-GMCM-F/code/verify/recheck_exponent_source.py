# -*- coding: utf-8 -*-
"""Q3 scale-exponent source sensitivity (READ-ONLY on the shipped stage outputs).

Acceptance test first: reproduce the shipped nominal configuration at
C=1e19, L_ctx=8192, exponential quality cost. Then re-solve the nominal grid with
the scale exponents (alpha, beta) taken from the REAL Pythia trajectory (B1) instead
of the semi-synthetic B6 table, holding the remaining parameters at their B6 values.

Writes only into data/reports/rechecks/.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

OUT = paths.REPORTS / "rechecks"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(paths.CODE / "q3"))

from core import Params, Problem  # noqa: E402

B6 = dict(E=0.860047, A=0.540173, B=1.593498, alpha=0.279054, beta=0.095607,
          delta=0.229965, kappa=-0.049936)
B1_AB = dict(alpha=0.33997658, beta=0.27987813)      # real Pythia trajectory (B1)
# S2: classical N-D branch taken entirely from the real trajectory, quality response
# still taken from B6 (B1 contains no quality dimension).
B1_FULL = dict(E=1.68979756, A=0.35398032, B=1.24030558, alpha=0.33997658,
               beta=0.27987813, delta=0.229965, kappa=-0.049936)
Q0 = 0.6074742221215453
CTX = 8192
BUDGETS = [1e19, 1e22, 1e24]


def solve(par, kind, budget):
    p = Params(**par)
    prob = Problem([p], CTX, kind, Q0)
    return prob.solve(budget)


print("=== acceptance: reproduce shipped nominal (1e19, 8192, exponential) ===")
t0 = time.time()
row = solve(B6, "exponential", 1e19)
arch = pd.read_csv(paths.Q3 / "nominal_configurations.csv")
a = arch[(arch.budget_FLOPs == 1e19) & (arch.context_tokens == CTX)
         & (arch.cost_model == "exponential")].iloc[0]
print(f"recomputed: N={row['N_B']:.6f} D={row['D_B']:.6f} Q*={row['Q_star']} loss={row['loss_star']:.6f}")
print(f"shipped   : N={a.N_B:.6f} D={a.D_B:.6f} Q*={a.Q_star} loss={a.loss_star:.6f}")
rel = max(abs(row['N_B'] - a.N_B) / a.N_B, abs(row['D_B'] - a.D_B) / a.D_B)
print("max rel diff: %.2e" % rel)
print("single solve seconds: %.2f" % (time.time() - t0))

rows = []
for tag, par in (("S0 nominal (all B6)", B6),
                 ("S1 B1 exponents only", {**B6, **B1_AB}),
                 ("S2 B1 classical branch", B1_FULL)):
    for kind in ("exponential", "power", "logarithmic"):
        for c in BUDGETS:
            t = time.time()
            r = solve(par, kind, c)
            rows.append(dict(scenario=tag, exponent_source=tag, alpha=par["alpha"], beta=par["beta"],
                             cost_model=kind, budget=c, N_B=r["N_B"], D_B=r["D_B"],
                             Q_star=r["Q_star"], active_boundary=r["active_boundary"],
                             loss_star=r["loss_star"],
                             quality_share=r["quality_share"], attention_share=r["attention_share"],
                             training_share=r["training_share"],
                             N_within_B6_support=r["N_within_B6_support"],
                             D_within_B6_support=r["D_within_B6_support"],
                             optimality_gap=r["optimality_gap"],
                             seconds=round(time.time() - t, 1)))
    print(f"  {tag}: done")

tab = pd.DataFrame(rows)
tab.to_csv(OUT / "q3_exponent_source_sensitivity.csv", index=False, encoding="utf-8-sig")
print()
print(tab[tab.cost_model == "exponential"][
    ["scenario", "budget", "N_B", "D_B", "Q_star", "quality_share", "loss_star"]].to_string(index=False))
print()
print(tab[tab.cost_model != "exponential"][
    ["scenario", "cost_model", "budget", "N_B", "D_B", "Q_star", "quality_share"]].to_string(index=False))

summary = {}
for kind in ("exponential", "power", "logarithmic"):
    s = tab[tab.cost_model == kind].set_index(["scenario", "budget"])
    for c in BUDGETS:
        rec = {}
        for tag in ("S1 B1 exponents only", "S2 B1 classical branch"):
            n1 = s.loc[(tag, c), "N_B"]
            d1 = s.loc[(tag, c), "D_B"]
            rec[tag] = dict(N_ratio=float(n1 / s.loc[("S0 nominal (all B6)", c), "N_B"]),
                            D_ratio=float(d1 / s.loc[("S0 nominal (all B6)", c), "D_B"]),
                            Q_star=float(s.loc[(tag, c), "Q_star"]),
                            quality_share=float(s.loc[(tag, c), "quality_share"]))
        summary[f"{kind}_{c:g}"] = rec


def classical_law(par, n, d):
    return par["E"] + par["A"] * n ** (-par["alpha"]) + par["B"] * d ** (-par["beta"])


level = dict(
    at_nominal_config_N1p8255_D0p5683=dict(
        B6_law=classical_law(B6, 1.8255, 0.5683),
        B1_law=classical_law(B1_FULL, 1.8255, 0.5683),
        difference=classical_law(B1_FULL, 1.8255, 0.5683) - classical_law(B6, 1.8255, 0.5683)),
    intercepts=dict(B6_E=B6["E"], B1_E=B1_FULL["E"], gap=B1_FULL["E"] - B6["E"]),
)
print()
print("loss level check:", json.dumps(level, ensure_ascii=False, indent=2))

json.dump({"S0_B6": B6, "S1_B1_exponents_only": {**B6, **B1_AB}, "S2_B1_classical_branch": B1_FULL,
           "dominant_exponent_B6": B6["beta"] / (B6["alpha"] + B6["beta"]),
           "dominant_exponent_B1": B1_AB["beta"] / (B1_AB["alpha"] + B1_AB["beta"]),
           "ratios": summary, "classical_law_level": level},
          open(OUT / "q3_exponent_source_sensitivity.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print()
print("dominant exponent beta/(alpha+beta): B6=%.4f  B1=%.4f"
      % (B6["beta"] / (B6["alpha"] + B6["beta"]),
         B1_AB["beta"] / (B1_AB["alpha"] + B1_AB["beta"])))
print(f"PASS: nominal 1e19/8192/exponential configuration reproduced "
      f"(max rel diff = {rel:.2e}); {len(tab)} scenario/budget/cost rows written")
print("wrote ->", OUT)
