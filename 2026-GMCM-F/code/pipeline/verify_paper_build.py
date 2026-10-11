"""Verify the freshly built paper and render the pages containing every figure."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

PAPER = paths.PAPER
PDF = PAPER / "main.pdf"
REPORT = paths.REPORTS
PAGES = REPORT / "paper_pages"


def build_log() -> Path:
    """LaTeX log of the fresh build, written in place or under ``build/``."""
    for candidate in (PAPER / "main.log", PAPER / "build" / "main.log"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No LaTeX log for {PDF}")


def run(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True,
                          encoding="utf-8", errors="replace").stdout


def render(page: int) -> Path:
    target = PAGES / f"page_{page:02d}"
    subprocess.run(["pdftoppm", "-f", str(page), "-l", str(page),
                    "-scale-to", "1000", "-png", "-singlefile",
                    str(PDF), str(target)], check=True, capture_output=True)
    return target.with_suffix(".png")


def contact(group: str, pages: list[int]) -> Path:
    width, height = 420, 610
    canvas = Image.new("RGB", (width * 2, height * ((len(pages) + 1) // 2)), "#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for j, page in enumerate(pages):
        picture = Image.open(PAGES / f"page_{page:02d}.png").convert("RGB")
        picture.thumbnail((width - 12, height - 28))
        x = (j % 2) * width + (width - picture.width) // 2
        y = (j // 2) * height + 23
        canvas.paste(picture, (x, y))
        draw.text((x, (j // 2) * height + 5), f"PDF page {page}", fill="black")
    target = REPORT / f"paper_{group}_contact.png"
    canvas.save(target)
    return target


def main() -> None:
    REPORT.mkdir(exist_ok=True)
    PAGES.mkdir(exist_ok=True)
    info = run("pdfinfo", str(PDF))
    page_count = int(re.search(r"^Pages:\s+(\d+)\s*$", info, re.M).group(1))
    assert page_count >= 100, "Unexpected paper page count; inspect pagination"
    fonts = run("pdffonts", str(PDF))
    assert "SimSun" in fonts
    log = build_log().read_text(encoding="utf-8", errors="replace")
    for token in ("Undefined control sequence", "Overfull \\hbox", "LaTeX Warning:",
                  "undefined references", "Citation `"):
        assert token not in log, token

    text = run("pdftotext", "-layout", "-enc", "UTF-8", str(PDF), "-")
    pages = text.rstrip("\f").split("\f")
    assert len(pages) == page_count
    # 正文图为数字节号，附录图为字母节号。正文提及“图 3.6”也可能行首出现，
    # 因此优先识别图题编号后的排版间距；图 5.11 的单空格图题另作精确匹配。
    captions = {}
    for page_num, page_text in enumerate(pages, 1):
        for line in page_text.splitlines():
            match = re.match(r"^[ \t]*图[ \t]*([1-9]\d*|[A-E])\.(\d+)([ \t]+)(\S.*)", line)
            if not match:
                continue
            number = match.group(1) + "." + match.group(2)
            if len(match.group(3)) < 2 and not (
                number == "5.11" and match.group(4).startswith("名义模型的转移点普查")
            ):
                continue
            assert number not in captions, f"Duplicate figure caption {number}"
            captions[number] = page_num
    source = "".join(p.read_text(encoding="utf-8") for p in
                     [PAPER / "main.tex", *sorted((PAPER / "content").glob("*.tex"))])
    expected = len(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{[^}]+\.pdf\}", source))
    assert len(captions) == expected, f"{len(captions)} captions vs {expected} included figures"
    assert list(captions) == sorted(captions, key=lambda n: (
        int(n.split(".")[0]) if n[0].isdigit() else ord(n[0]) - ord("A") + 20,
        int(n.split(".")[1]),
    )), "Figure captions are out of numerical order"

    # 逐字检查前去掉全部空白：pdftotext 会在数字与汉字之间断行
    flat = re.sub(r"\s+", "", text)
    for required in ("0.04813549", "0.09627098", "第239行", "第170行",
                     "66个候选", "0.311", "24.3", "296.8",
                     "4.97%", "80.72%", "3000个配置",
                     "1.81%", "第46行变为第13行",
                     "后训练24月"):
        assert required in flat, f"Text missing: {required}"
    unique_pages = sorted(set(captions.values()))
    # 清理本脚本自身的旧页面预览，避免 v42/v43 页码残留在 v44 交付包。
    report_root = REPORT.resolve()
    for stale in [*PAGES.glob("page_*.png"), *REPORT.glob("paper_sec*_contact.png")]:
        assert stale.resolve().is_relative_to(report_root), stale
        stale.unlink()
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(render, unique_pages))
    # 模板按节编号图，联系表按节分组
    groups: dict[str, set[int]] = {}
    for number, page in captions.items():
        groups.setdefault("sec" + number.split(".")[0], set()).add(page)
    groups = {name: sorted(pages) for name, pages in sorted(groups.items())}
    contacts = {name: str(contact(name, group).relative_to(paths.PACKAGE)).replace("\\", "/")
                for name, group in groups.items()}
    result = {"schema": "paper-build-verification/1", "pages": page_count,
              "figure_captions": captions, "rendered_pages": unique_pages,
              "contacts": contacts, "embedded_simsun": True,
              "unresolved_refs_or_overfull_boxes": False}
    (REPORT / "paper_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {len(captions)} figure captions on {len(unique_pages)} pages of {page_count}-page PDF")


if __name__ == "__main__":
    main()

