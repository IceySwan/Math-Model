"""Single source of truth for every filesystem path used by the final programs.

This source publication uses local competition attachments, which are
never redistributed.  Point ``F_ATTACHMENTS`` at the directory holding
``A_data_value``, ``B_scaling_laws`` and ``C_efficiency_evolution`` to re-run the
stages that read raw data. Later stages consume locally generated outputs; Q2 also needs the local reference files documented in README.md.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

CODE = Path(__file__).resolve().parent          # published repository/code
PACKAGE = CODE.parent                           # published repository
PAPER = PACKAGE / "paper"                    # LaTeX project (GMCM template)
DATA = PACKAGE / "data"
OUT = DATA / "outputs"                          # Q1..Q4 final outputs
VALIDATION = DATA / "validation"                # auxiliary and preserved-run tables
FIGURE_STAGE = DATA / "figures"                 # regenerated figure PDFs + manifest
FIGURE_DATA = OUT / "plot_data"                 # per-figure plotted data
PAPER_FIGURES = PAPER / "figures"               # what LaTeX includes
REPORTS = DATA / "reports"

ATTACHMENTS = Path(os.environ.get("F_ATTACHMENTS", PACKAGE / "real_attachments"))
A = ATTACHMENTS / "A_data_value"
B = ATTACHMENTS / "B_scaling_laws"
C = ATTACHMENTS / "C_efficiency_evolution"

Q1 = OUT / "q1"
Q2 = OUT / "q2"
Q3 = OUT / "q3"
Q4 = OUT / "q4"
SLOWDOWN = OUT / "slowdown"
EVIDENCE = OUT / "evidence"

for _d in (OUT, VALIDATION, FIGURE_STAGE, FIGURE_DATA, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)


def digest(path: str | Path) -> str:
    """SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def read_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
