# -*- coding: utf-8 -*-
"""Generate the code-listing fragment used by Appendix D of the paper.

The excerpts are read straight from the shipped programs, so the listing can never
drift from the code that produced the results.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

SNIPPETS = [
    ("D.1", "第一问：平方根坐标二次岭与固定候选集",
     "q1/mixture_solver.py", "def candidate_models", 18,
     "候选模型集合在任何嵌套或外部评价之前固定；\\texttt{sqrt} 坐标给出 $\\sqrt{p_i}$ 与 $\\sqrt{p_ip_k}$ 项。"),
    ("D.2", "第三问：质量区间的解析下界与全局比较骨架",
     "q3/core.py", "def node(self,a,b,budget):", 28,
     "区间下界取 $h(a)$ 与 $g(\\ln b)$ 的组合，用凸支撑线给出该区间的目标下界；堆结构按最小下界优先分裂。"),
    ("D.3", "第四问：有序分位规划的顶点约束与上图变量",
     "q4/ordered_model.py", "vertices=(domain_vertices", 18,
     "把形状要求化为矩形四个顶点上的线性约束，与 Pinball 损失的上图变量一起构成凸二次规划。"),
    ("D.4", "定稿核验：顶点预测与全局间隙断言",
     "pipeline/finalize_evidence.py", "predictions = fit.matrix(vertex_frame)", 8,
     "对全部矩形顶点逐点检查取值区间与分位间距，并断言归档的目标间隙上界。"),
]


def main() -> None:
    out = []
    for tag, title, rel, anchor, span, note in SNIPPETS:
        src = paths.CODE / rel
        if not src.is_file():
            print(f"  skip {rel} (missing)")
            continue
        lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
        starts = [i for i, ln in enumerate(lines) if anchor in ln]
        if not starts:
            print(f"  ERROR anchor not found in {rel}: {anchor!r}")
            continue
        lo = starts[0] + 1
        hi = min(lo + span - 1, len(lines))
        body = [ln[:96] for ln in lines[lo - 1:hi]]
        esc = rel.replace("_", chr(92) + "_")
        heading = (f"\\paragraph{{{tag} {title}（\\texttt{{{esc}}} 第 {lo}--{hi} 行）"
                   + "\\quad}")
        out.append(heading)
        out.append(note)
        out.append("")
        out.append("\\begin{lstlisting}[language=Python,breaklines=true,basicstyle=\\small\\ttfamily,"
                   "showstringspaces=false,"
                   "frame=single,framesep=4pt,rulecolor=\\color{gray!50},"
                   "backgroundcolor=\\color{gray!5},title={" + Path(rel).name.replace("_", chr(92) + "_") + "}]")
        out.extend(body)
        out.append("\\end{lstlisting}")
        out.append("")
        print(f"  {tag} {rel} lines {lo}-{hi} ({hi - lo + 1} lines)")
    target = paths.PAPER / "tables" / "appendix_program_snippets.tex"
    target.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("wrote", target.relative_to(paths.PACKAGE))


if __name__ == "__main__":
    main()


