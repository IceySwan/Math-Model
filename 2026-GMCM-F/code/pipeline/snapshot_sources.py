"""Preserve byte-identical snapshots of the shipped core programs.

The snapshot covers the migrated analysis modules under ``code/q1`` .. ``code/q4``
together with the Q1 entry point, so later edits cannot silently change the
numbers reported in the paper.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

DEST = paths.DATA / "provenance" / "core_sources"
RECEIPT = paths.DATA / "provenance" / "core_source_receipt.json"
SOURCES = [source
           for pattern in ("q1/*.py", "q2/*.py", "q3/*.py", "q4/*.py")
           for source in sorted(paths.CODE.glob(pattern))]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    receipt = {}
    for source in SOURCES:
        relative = source.relative_to(paths.CODE).as_posix()
        saved = DEST / relative
        assert source.is_file(), relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        if saved.exists():
            assert digest(saved) == digest(source), f"Source changed after snapshot: {relative}"
        else:
            shutil.copy2(source, saved)
        receipt[relative] = {"sha256": digest(saved), "bytes": saved.stat().st_size}
    RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    RECEIPT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {len(receipt)} shipped core source files saved without modifying originals")


if __name__ == "__main__":
    main()
