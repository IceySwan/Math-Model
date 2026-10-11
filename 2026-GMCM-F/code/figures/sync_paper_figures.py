"""Back up the installed paper figures, then install the verified final PDFs."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

PAPER = paths.PAPER
FIG = paths.PAPER_FIGURES
STAGE = paths.FIGURE_STAGE
MANIFEST = STAGE / "figure_manifest.json"
EXTERNAL = STAGE / "external_figures.json"
BACKUP = paths.DATA / "provenance" / "previous_paper_figures"
RECEIPT = paths.DATA / "provenance" / "paper_figure_sync_receipt.json"
# Title-page artwork shipped with the GMCM template; it lives beside the figures but
# is not produced by the figure pipeline.
TEMPLATE_ASSETS = {"title.pdf", "logo.pdf", "title2025.pdf", "logo2025.pdf"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
    external = json.loads(EXTERNAL.read_text(encoding="utf-8"))
    actual = {p.name for p in FIG.glob("*.pdf")} - TEMPLATE_ASSETS
    assert actual == set(expected) | set(external), \
        f"Unexpected paper figures: {actual ^ (set(expected) | set(external))}"
    BACKUP.mkdir(parents=True, exist_ok=True)
    prior_receipt = (json.loads(RECEIPT.read_text(encoding="utf-8"))
                     if RECEIPT.exists() else {})
    receipt = {}
    for name, row in expected.items():
        old = FIG / name
        new = STAGE / name
        assert sha(new) == row["pdf_sha256"], name
        saved = BACKUP / name
        if saved.exists():
            # Permit a later synchronization when the paper already contains
            # the previously recorded final PDF; the backup remains immutable.
            already_recorded = prior_receipt.get(name, {}).get("final_sha256") == sha(old)
            assert sha(saved) == sha(old) or sha(old) == sha(new) or already_recorded, name
        else:
            shutil.copy2(old, saved)
        receipt[name] = {"previous_sha256": sha(saved), "final_sha256": sha(new)}
    # All previous files are now recoverable. Copying is intentionally non-destructive.
    for name in expected:
        shutil.copy2(STAGE / name, FIG / name)
        assert sha(FIG / name) == expected[name]["pdf_sha256"], name
    # External figures are already installed by their own scripts; verify, do not copy.
    for name, row in external.items():
        assert sha(FIG / name) == row["pdf_sha256"], f"external figure drifted: {name}"
        receipt[name] = {"previous_sha256": None, "final_sha256": row["pdf_sha256"],
                         "external": True}
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: backed up and synchronized {len(expected)} paper figures "
          f"(+{len(external)} external verified)")


if __name__ == "__main__":
    main()
