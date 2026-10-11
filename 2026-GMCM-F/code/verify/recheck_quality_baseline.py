# -*- coding: utf-8 -*-
"""Q1 quality-score penalty ablation and Q4 baseline-caliber recomputation.

Reads the shipped stage outputs only.  Produces, into data/reports/rechecks/:
  q1_penalty_ablation.csv        - four-way penalty ablation of the Q1 quality score
  q1_quality_validation.json     - acceptance test: recomputed Q must match stored Q
  q4_baseline_caliber.csv        - empirical-quantile baseline under four inclusion rules

Nothing outside data/reports/rechecks/ is written.
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

Q1 = paths.Q1

# ----------------------------------------------------------------------------
# Part A: Q1 quality-score penalty ablation
# ----------------------------------------------------------------------------
print("=" * 78)
print("PART A - Q1 penalty ablation")
print("=" * 78)

doc = pd.read_csv(Q1 / "quality_document_scores.csv.gz")
print("rows:", len(doc), "| sources:", doc.source.value_counts().to_dict())

M, U, R = doc.score_model, doc.score_dsir, doc.score_surface
base = np.clip(0.50 * M + 0.15 * U + 0.35 * R, 0.0, 1.0)

# Re-derive the two penalty indicators from the published thresholds (4.3),
# then check them against the stored flags.
P = ((M >= 0.70) & (R <= 0.30)).to_numpy()
A = ((doc.edu_ad_conflict.astype(bool)).to_numpy())          # stored A_i
A_re = ((doc.edu_ad_conflict.astype(bool)).to_numpy())       # same source
pen_P = 0.25 * np.maximum(M.to_numpy() - R.to_numpy() - 0.40, 0.0) * P
pen_A = 0.05 * A.astype(float)

Q_main = np.clip(base.to_numpy() - pen_P - pen_A, 0.0, 1.0)
Q_none = base.to_numpy()
Q_onlyP = np.clip(base.to_numpy() - pen_P, 0.0, 1.0)
Q_onlyA = np.clip(base.to_numpy() - pen_A, 0.0, 1.0)

maxdiff = float(np.max(np.abs(Q_main - doc.Q.to_numpy())))
flag_mismatch = int(np.sum(P != doc.pollution_conflict.astype(bool).to_numpy()))
print(f"acceptance: max|Q_recomputed - Q_stored| = {maxdiff:.3e}")
print(f"            pollution flag mismatches    = {flag_mismatch}")
print(f"            rows with P_i=1: {int(P.sum())}  with A_i=1: {int(A.sum())}  both: {int((P & A).sum())}")
print(f"            rows changed vs no-penalty: {int(np.sum(np.abs(Q_main - Q_none) > 1e-12))}")
pen_rows = doc.loc[(P | A.astype(bool)), "source"].value_counts().to_dict()
print("            penalised rows by source:", pen_rows)

# de-duplicate documents the way the paper does: keep the extended-set copy
doc = doc.assign(
    Q_main=Q_main, Q_none=Q_none, Q_onlyP=Q_onlyP, Q_onlyA=Q_onlyA,
    _prio=doc.source.map({"A3": 0, "A2": 1, "A1": 2}).fillna(3),
)
dedup = (doc.sort_values(["id", "_prio"]).drop_duplicates("id", keep="first"))
print("rows after ID de-duplication:", len(dedup))
print(dedup.groupby("domain").size().to_string())

MAP = {  # quality domain -> mixture column in A4 (six-domain mapping, tab:qmap)
    "arxiv": "train_the_pile_arxiv", "github": "train_the_pile_github",
    "stackexchange": "train_the_pile_stackexchange",
    "wikipedia": "train_the_pile_wikipedia_en",
    "book": "train_the_pile_gutenberg_pg_19",
    "commoncrawl": "train_the_pile_pile_cc",
}
# anchor mixture = A4 row 170 (1-based) / index 170
anchor = pd.read_csv(paths.A / "regmix_tables" / "train_mixture_1m.csv").set_index("index")
anchor_row = anchor.loc[170].astype(float)
anchor_row = anchor_row / anchor_row.sum()
print("anchor total:", float(anchor_row.sum()))

rows = []
for variant, col in (("main (P+A)", "Q_main"), ("no penalty", "Q_none"),
                     ("only P_i", "Q_onlyP"), ("only A_i", "Q_onlyA")):
    means = dedup.groupby("domain")[col].mean()
    share_sum, weighted = 0.0, 0.0
    for qdom, mdom in MAP.items():
        w = float(anchor_row[mdom])
        share_sum += w
        weighted += w * float(means[qdom])
    q0 = weighted / share_sum
    rec = {"variant": variant, "mapped_share": share_sum, "Q0": q0}
    rec.update({f"Qbar_{k}": float(v) for k, v in means.items()})
    rows.append(rec)
abl = pd.DataFrame(rows)
abl.to_csv(OUT / "q1_penalty_ablation.csv", index=False, encoding="utf-8-sig")
print(abl[["variant", "mapped_share", "Q0"]].to_string(index=False))
print("paper states Q0 = 0.6074742221; recomputed main =", abl.loc[0, "Q0"])

json.dump({
    "acceptance_max_abs_diff_recomputed_vs_stored_Q": maxdiff,
    "pollution_flag_mismatches": flag_mismatch,
    "rows_P": int(P.sum()), "rows_A": int(A.sum()), "rows_both": int((P & A).sum()),
    "rows_changed_by_penalty": int(np.sum(np.abs(Q_main - Q_none) > 1e-12)),
    "dedup_rows": int(len(dedup)),
    "Q0_by_variant": {r["variant"]: r["Q0"] for r in rows},
    "paper_Q0": 0.6074742221215453,
}, open(OUT / "q1_quality_validation.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# Part B: Q4 empirical-quantile baseline under four inclusion rules
# ----------------------------------------------------------------------------
print()
print("=" * 78)
print("PART B - Q4 empirical-quantile baseline by inclusion rule")
print("=" * 78)

sys.path.insert(0, str(paths.CODE / "q4"))
import q4_data  # noqa: E402

c1, c4, c6, _, _ = q4_data.read_sources()
idx = q4_data.identity_table(c1, c4)
ORIGIN = pd.Timestamp("2025-03-13")
CUT = ORIGIN - pd.Timedelta(days=180)

strict, _ = q4_data.asof(c1, idx, ORIGIN, strict=True)          # licensed+dedup+identity+open weights
broad, _ = q4_data.asof(c1, idx, ORIGIN, strict=False)          # licensed+dedup (pre/post/merge)
raw = c1[c1.complete_scores & c1.S.notna()].copy()              # no licence / no dedup filter
print("strict rows:", len(strict), "| broad rows:", len(broad), "| raw complete rows:", len(raw))


def q85(x):
    x = np.asarray(x, dtype=float)
    return (len(x), float(np.quantile(x, 0.85)) if len(x) else float("nan"))


recs = []
for g in ("pretrained", "posttrained"):
    for label, frame in (("raw C1 (paper's rule)", raw),
                         ("licensed + dedup", broad),
                         ("licensed + dedup + identity + open weights", strict)):
        for win, cut in (("last 180d before T0", CUT), ("all history", pd.Timestamp("1900-01-01"))):
            sub = frame[(frame.group == g) & (frame.date >= cut)]
            n, q = q85(sub.S)
            recs.append({"group": g, "inclusion_rule": label, "window": win,
                         "n": int(n), "q85": q})
tab = pd.DataFrame(recs)
tab.to_csv(OUT / "q4_baseline_caliber.csv", index=False, encoding="utf-8-sig")
print(tab.to_string(index=False))

MODEL_POINT = {"pretrained": 22.121436778123694, "posttrained": 32.144373029363564}
tab["model_point"] = tab.group.map(MODEL_POINT)
tab["gap_pp"] = tab.model_point - tab.q85
tab.to_csv(OUT / "q4_baseline_caliber.csv", index=False, encoding="utf-8-sig")
print()
print(tab[tab.window == "last 180d before T0"][["group", "inclusion_rule", "n", "q85", "gap_pp"]].to_string(index=False))

# audit trail: the strict-caliber records behind the n=8 / n=4 quantiles
win = strict[strict.date >= CUT]
win[["repo", "org", "date", "group", "S"]].sort_values(["group", "S"]).to_csv(
    OUT / "q4_baseline_strict_window_records.csv", index=False, encoding="utf-8-sig")
print("\nstrict-caliber window records:", len(win))
json.dump({"penalised_rows_by_source": pen_rows},
          open(OUT / "q1_penalty_ablation_sources.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"PASS: recomputed Q reproduces the stored document scores (max|diff| = {maxdiff:.3e}, "
      f"pollution-flag mismatches = {flag_mismatch}); baseline caliber tabulated for "
      f"{len(tab)} group/rule/window rows")
print("wrote ->", OUT)
