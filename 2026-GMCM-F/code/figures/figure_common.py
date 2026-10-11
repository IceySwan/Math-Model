"""Figure recording and source fingerprinting for the submitted figure set."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402
from plot_style import save_pdf  # noqa: E402

ROOT = paths.PACKAGE                    # manifest paths stay relative to the package
FIG = paths.FIGURE_STAGE
DATA = paths.FIGURE_DATA
MANIFEST = FIG / "figure_manifest.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tex(text: str) -> str:
    """Escape ordinary data labels that are not math expressions."""
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def record(fig, name: str, sources: list[Path], plotted: pd.DataFrame | None = None) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    target = FIG / name
    save_pdf(fig, target)
    if plotted is not None:
        data_path = DATA / (Path(name).stem + ".csv")
        plotted.to_csv(data_path, index=False, encoding="utf-8-sig", float_format="%.16g")
    else:
        data_path = None
    entries = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    entries[name] = {
        "pdf_sha256": sha(target),
        "pdf_bytes": target.stat().st_size,
        "font_policy": "SimSun text via xeCJK; formula labels via XeLaTeX math",
        "source_sha256": {
            str(p.relative_to(ROOT)).replace("\\", "/"): sha(p)
            for p in sources
        },
        "plot_data": (
            {"path": str(data_path.relative_to(ROOT)).replace("\\", "/"),
             "sha256": sha(data_path)}
            if data_path else None
        ),
    }
    MANIFEST.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print("FIGURE", name, flush=True)
