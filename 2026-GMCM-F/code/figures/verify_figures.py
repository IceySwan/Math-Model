"""Check every figure referenced by the paper against the final figure manifest."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

ROOT = paths.PACKAGE
PAPER = paths.PAPER
MANIFEST = paths.FIGURE_STAGE / "figure_manifest.json"
EXTERNAL = paths.FIGURE_STAGE / "external_figures.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    external = json.loads(EXTERNAL.read_text(encoding="utf-8"))
    tex_files = [PAPER / "main.tex", *sorted((PAPER / "content").glob("*.tex"))]
    referenced = []
    for path in tex_files:
        referenced.extend(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+\.pdf)\}",
                                     path.read_text(encoding="utf-8")))
    if len(referenced) != len(set(referenced)):
        raise AssertionError("Repeated figure reference")
    known = set(entries) | set(external)
    if set(referenced) != known:
        raise AssertionError(f"Figure mismatch: paper-only={set(referenced)-known}, "
                             f"registered-only={known-set(referenced)}")
    if set(entries) & set(external):
        raise AssertionError("A figure is registered twice")

    findings = []
    for name in sorted(entries):
        row = entries[name]
        path = paths.FIGURE_STAGE / name
        assert path.is_file() and sha(path) == row["pdf_sha256"], name
        assert path.stat().st_size == row["pdf_bytes"], name
        for relative, expected in row["source_sha256"].items():
            source = ROOT / relative
            assert source.is_file() and sha(source) == expected, relative
        plot = row["plot_data"]
        assert plot and sha(ROOT / plot["path"]) == plot["sha256"], name
        info = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", check=True).stdout
        fonts = subprocess.run(["pdffonts", str(path)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", check=True).stdout
        assert re.search(r"^Pages:\s+1\s*$", info, re.M), name
        assert "SimSun" in fonts, name
        assert "Type 3" not in fonts, name
        findings.append({"figure": name, "one_page": True, "embedded_simsun": True,
                         "source_hashes_match": True, "plot_data_hash_matches": True})

    # Figures produced outside record(): checked against their own provenance instead.
    for name, row in sorted(external.items()):
        path = PAPER / "figures" / name
        assert path.is_file() and sha(path) == row["pdf_sha256"], name
        assert path.stat().st_size == row["pdf_bytes"], name
        assert (ROOT / row["built_by"]).is_file(), row["built_by"]
        for relative, expected in row["provenance_sha256"].items():
            source = ROOT / relative
            assert source.is_file() and sha(source) == expected, relative
        info = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", check=True).stdout
        fonts = subprocess.run(["pdffonts", str(path)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace", check=True).stdout
        assert re.search(r"^Pages:\s+1\s*$", info, re.M), name
        assert "Type 3" not in fonts, name
        findings.append({"figure": name, "one_page": True,
                         "embedded_simsun": "SimSun" in fonts,
                         "source_hashes_match": True, "plot_data_hash_matches": None,
                         "provenance": "external receipt"})

    report = {"schema": "figure-verification/2", "count": len(findings),
              "registered": len(entries), "external": len(external),
              "paper_references_match": True, "findings": findings}
    target = paths.REPORTS / "figure_verification.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {len(findings)} paper figures ({len(entries)} pipeline + {len(external)} external), "
          f"source data hashes, one-page PDF and SimSun")


if __name__ == "__main__":
    main()
