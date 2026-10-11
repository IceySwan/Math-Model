# -*- coding: utf-8 -*-
"""Register the three paper figures that are produced outside figures_*.py::record().

They are still fully traceable, but their provenance lives in their own inputs and
receipts rather than in figure_manifest.json.  This script writes
data/figures/external_figures.json so that verify_figures.py and sync_paper_figures.py
can check them without weakening the checks on the other 24.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

ROOT = paths.PACKAGE


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


EXTERNAL = {
    "fig_roadmap.pdf": {
        "built_by": paths.PAPER / "tools" / "build_chapter1_roadmap.py",
        "provenance": [
            paths.PAPER / "reports" / "roadmap_data_provenance.json",
            paths.PAPER / "figures" / "fig_roadmap.tex",
        ],
        "note": "第一章技术路线图；所引数值与来源文件摘要见其自带 receipt。",
    },
    "q1_extrapolation_diagnostic.pdf": {
        "built_by": paths.Q1 / "extrapolation_diagnostic.csv",
        "provenance": [
            paths.Q1 / "extrapolation_diagnostic.csv",
            paths.Q1 / "extrapolation_domain_correlation.csv",
        ],
        "note": "第一问跨尺度迁移诊断；绘图驱动不随包发布，成图由两份归档 CSV 直接确定。",
    },
    "q3_v41_transition_census.pdf": {
        "built_by": paths.CODE / "q3" / "core.py",
        "provenance": [
            paths.CODE / "q3" / "core.py",
            paths.CODE / "q3" / "q3_core.py",
            paths.Q3 / "v41_structural_transfer.csv",
        ],
        "note": "第三问转移点普查；绘图驱动不随包发布，转移点由随包的冻结模型逐点重解得到。",
    },
}

rows = {}
for name, meta in EXTERNAL.items():
    pdf_paper = paths.PAPER_FIGURES / name
    pdf_stage = paths.FIGURE_STAGE / name
    assert pdf_paper.is_file(), name
    entry = {
        "pdf_sha256": sha(pdf_paper),
        "pdf_bytes": pdf_paper.stat().st_size,
        "in_bundle": pdf_stage.is_file(),
        # The generating scripts write the figure twice (staging + paper); matplotlib
        # stamps a fresh PDF /ID and CreationDate on each write, so the two copies are
        # visually identical but not byte-identical. Record both.
        "bundle_sha256": sha(pdf_stage) if pdf_stage.is_file() else None,
        "built_by": rel(meta["built_by"]),
        "note": meta["note"],
        "provenance_sha256": {},
    }
    if pdf_stage.is_file():
        assert pdf_stage.stat().st_size == entry["pdf_bytes"], f"bundle/paper size differ: {name}"
    for p in meta["provenance"]:
        assert p.is_file(), rel(p)
        entry["provenance_sha256"][rel(p)] = sha(p)
    rows[name] = entry

out = paths.FIGURE_STAGE / "external_figures.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"registered {len(rows)} external figures -> {out.relative_to(ROOT)}")
for name, row in rows.items():
    print(f"  {name:<34} {row['pdf_bytes']:>7} bytes  provenance files: {len(row['provenance_sha256'])}")
