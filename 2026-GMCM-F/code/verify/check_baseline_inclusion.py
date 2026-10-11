# -*- coding: utf-8 -*-
"""Check whether the retained empirical-quantile baseline is computed on the same
sample as the fitted strict/auxiliary samples.

The baseline table used the rule: complete six-task row + type + date >= 2024-09-14.
It did NOT apply the licence allowlist, the C4 open-weights identity match, or the
per-repository de-duplication used by q4_data.asof().

This script recomputes the same 180-day and all-history quantiles under the
licence allowlist (+ per-repo dedup) so the two sample definitions can be compared.
Read-only: writes nothing.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

C1 = paths.C / "leaderboard_cleaned.csv"
TASKS = ["IFEval", "BBH", "MATH Lvl 5", "GPQA", "MUSR", "MMLU-PRO"]
PRE = {"🟢 pretrained", "🟩 continuously pretrained"}
POST = {"💬 chat models (RLHF, DPO, IFT, ...)",
        "🔶 fine-tuned on domain-specific datasets"}
LICENSES = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "bsd-3-clause-clear",
            "cc-by-4.0", "cc-by-sa-4.0", "cc0-1.0", "isc", "mpl-2.0"}
CUT = "2024-09-14"
T0 = "2025-03-13"

rows = list(csv.DictReader(open(C1, newline="", encoding="utf-8-sig")))
recs = []
for r in rows:
    try:
        v = float(np.mean([float(r[t]) for t in TASKS]))
    except (TypeError, ValueError):
        continue
    recs.append(dict(date=(r["Submission Date"] or "").strip()[:10], S=v,
                     type=r["Type"], lic=(r["Hub License"] or "").strip().lower(),
                     repo=(r.get("Model") or "").strip()))

print("raw C1 rows with complete six-task scores: %d" % len(recs))


def q85(sel, label):
    if not sel:
        print("%-58s n=0" % label)
        return
    a = np.array([x["S"] for x in sel])
    print("%-58s n=%5d  q85=%8.4f  mean=%8.4f" % (label, len(a), np.quantile(a, .85), a.mean()))


def dedup(sel):
    """keep the latest submission per repository, as asof() does"""
    best = {}
    for x in sel:
        k = x["repo"]
        if k not in best or x["date"] > best[k]["date"]:
            best[k] = x
    return list(best.values())


for name, types in (("pretrained", PRE), ("posttrained", POST)):
    win = [x for x in recs if x["type"] in types and x["date"] >= CUT]
    hist = [x for x in recs if x["type"] in types]
    q85(win, f"{name}: 180d window, no licence/dedup (reported baseline)")
    q85(dedup(win), f"{name}: 180d window, per-repo dedup")
    q85([x for x in win if x["lic"] in LICENSES],
        f"{name}: 180d window, licence allowlist")
    q85(dedup([x for x in win if x["lic"] in LICENSES]),
        f"{name}: 180d window, licence allowlist + dedup")
    q85(hist, f"{name}: all history, no licence/dedup (reported baseline)")
    q85(dedup([x for x in hist if x["lic"] in LICENSES]),
        f"{name}: all history, licence allowlist + dedup")
    print()

print("distinct licences in the 180d window (all types):")
from collections import Counter
c = Counter(x["lic"] or "(empty)" for x in recs if x["date"] >= CUT)
for k, v in c.most_common(12):
    print("   %-28s %5d  %s" % (k, v, "IN allowlist" if k in LICENSES else ""))
