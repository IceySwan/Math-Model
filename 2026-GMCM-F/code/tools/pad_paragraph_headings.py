# -*- coding: utf-8 -*-
"""Add a \\quad gap after every run-in \\paragraph{...} heading.

The heading is typeset run-in, so the text starts immediately after the full stop.
Inserting \\quad before the closing brace opens a 1em gap, matching the heading the
user adjusted by hand in 03_mathematical_background.tex.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

TARGETS = sorted((paths.PAPER / "content").glob("*.tex")) + \
    sorted((paths.PAPER / "tables").glob("*.tex"))
PATTERN = re.compile(r"\\paragraph(\*?)\{")


def rewrite(text: str) -> tuple[str, int, list[str]]:
    out, pos, changed, skipped = [], 0, 0, []
    for m in PATTERN.finditer(text):
        start = m.end()                       # just after '{'
        depth, i = 1, start
        while i < len(text) and depth:
            if text[i] == "\\":               # skip escaped brace
                i += 2
                continue
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[start:i - 1]
        closing = i - 1
        if body.rstrip().endswith(r"\quad"):
            skipped.append(body[:40])
            continue
        out.append(text[pos:closing])
        out.append(r"\quad}")
        pos = i
        changed += 1
    out.append(text[pos:])
    return "".join(out), changed, skipped


def main() -> None:
    total = 0
    for path in TARGETS:
        text = path.read_text(encoding="utf-8")
        if r"\paragraph" not in text:
            continue
        new, changed, skipped = rewrite(text)
        if changed:
            path.write_text(new, encoding="utf-8")
            total += changed
            print(f"  {path.name:<32} +{changed}" + (f"  (已跳过 {len(skipped)})" if skipped else ""))
    print(f"total headings updated: {total}")


if __name__ == "__main__":
    main()
