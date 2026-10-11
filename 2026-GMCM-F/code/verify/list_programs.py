# -*- coding: utf-8 -*-
"""Build a hash-verified inventory of every shipped program for the code appendix."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

ROOT = paths.PACKAGE
GROUPS = [
    ("路径契约", "*.py"),
    ("Q1 质量评分与配比响应面", "q1/*.py"),
    ("Q2 广义标度律与质量响应", "q2/*.py"),
    ("Q3 预算优化与稳健配置", "q3/*.py"),
    ("Q4 有序分位前沿与情景预测", "q4/*.py"),
    ("定稿复算与核验流水线", "pipeline/*.py"),
    ("图件生成与同步", "figures/*.py"),
    ("复核与验收脚本", "verify/*.py"),
]
ROLE = {
    "code/paths.py": "全包路径契约：目录定义与 JSON、摘要工具",
    "code/q1/solve_q1.py": "第一问主入口：评分、冲突标记、平方根响应面与交叉验证",
    "code/q1/quality_io.py": "质量输入的分批解析与逐附件可核验缓存",
    "code/q1/mixture_solver.py": "仅训练集的折内选择与嵌套验证",
    "code/q1/mixture_effects.py": "单纯形探针与批量局部响应计算",
    "code/q1/q1_pipeline.py": "第一问配比流程：拟合 A4/A5、冻结并评估历史表",
    "code/q2/run_study.py": "第二问固定协议比较、嵌套验证与联合不确定性",
    "code/q2/quality_solver.py": "变量投影拟合与分组验证",
    "code/q2/scaling_solver.py": "带解析导数的有界标度律求解",
    "code/q3/run_study.py": "第三问名义、有界最大后悔与条件重抽样研究",
    "code/q3/core.py": "固定质量唯一尺度、质量全局比较与饱和判据",
    "code/q3/q3_core.py": "质量成本函数族与幂成本工具",
    "code/q4/run_study.py": "第四问按时间与组织切分的候选验证",
    "code/q4/protocol.py": "第四问固定协议：分析口径在评估前声明",
    "code/q4/ordered_model.py": "有序联合分位凸二次规划与对偶下界",
    "code/q4/resource_forecast.py": "训练算力匹配、联合抽样与情景预测",
    "code/q4/data_config.py": "数据定位与分析口径常量",
    "code/q4/q4_data.py": "C1/C4 身份匹配、许可白名单与去重口径",
    "code/q4/q4_kernels.py": "加权 SVD 与加权稀疏分位线性规划等数值核",
    "code/q4/q4_frontier.py": "冻结前沿候选、时间切分评估与算力回归",
    "code/q4/q4_bridge.py": "损失尺度分离、显式基线与折外预测",
    "code/q4/q4_uncertainty.py": "组织重抽样与训练算力一致的情景生成",
    "code/q4/q4_alternatives.py": "辅助趋势基准与迁移探索",
    "code/pipeline/rerun_core.py": "四问定稿重算驱动",
    "code/pipeline/rerun_slowdown.py": "第四问减速情景的第三问资源重解",
    "code/pipeline/finalize_evidence.py": "定稿数值证据与顶点检查汇总",
    "code/pipeline/verify_paper_build.py": "论文数值与归档一致性核验",
    "code/pipeline/compare_preserved_runs.py": "定稿重算与保留运行的逐文件比对",
    "code/pipeline/snapshot_sources.py": "冻结核心程序源的只读快照",
    "code/figures/plot_style.py": "统一图件样式：SimSun 正文与 XeLaTeX 数学",
    "code/figures/figure_common.py": "图件登记、源文件摘要与绘图数据落盘",
    "code/figures/figures_q1.py": "第一问出图",
    "code/figures/figures_q2_q3.py": "第二、三问出图",
    "code/figures/figures_q4.py": "第四问出图",
    "code/figures/verify_figures.py": "图件、绘图数据与正文引用一致性核验",
    "code/figures/sync_paper_figures.py": "论文图件目录的备份与同步",
    "code/verify/recheck_quality_baseline.py": "复核：冲突惩罚消融与基准入选口径",
    "code/verify/recheck_lambda_domain.py": "复核：斜率惩罚敏感性与声明域",
    "code/verify/recheck_exponent_source.py": "复核：规模指数来源敏感性",
    "code/verify/recheck_aggregation.py": "复核：含有序基准的凸组合定权",
    "code/verify/verify_paper_numbers.py": "复核：正文数值与复算结果逐项比对",
    "code/verify/review_sensitivity_v44.py": "评审复核：条件映射、连续冲突权重与饱和重解",
    "code/verify/annotate_review_scope_v44.py": "评审复核：历史数值键名和证据范围注释",
    "code/verify/list_programs.py": "复核：随包程序清单与 LaTeX 附录表",
    "code/verify/register_external_figures.py": "复核：外部图件来源登记",
    "code/verify/check_baseline_inclusion.py": "复核：基准入选规则对照",
}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_lines(p: Path) -> int:
    with open(p, encoding="utf-8", errors="ignore") as fh:
        return sum(1 for _ in fh)


rows = []
for group, pattern in GROUPS:
    for p in sorted(paths.CODE.glob(pattern)):
        if "__pycache__" in p.parts:
            continue
        shown = str(p.relative_to(ROOT)).replace("\\", "/")
        rows.append(dict(group=group, path=shown, bytes=p.stat().st_size,
                         lines=count_lines(p), sha256=sha(p),
                         role=ROLE.get(shown, "")))
print("inventoried:", len(rows), "programs under", Path(paths.CODE).name + "/")
for r in rows:
    print(f"  {r['path']:<46} {r['lines']:>5} lines  {r['sha256'][:16]}")

paths.write_json(paths.REPORTS / "program_inventory.json",
                 {"root": "code", "count": len(rows), "files": rows})

tex = [r"{\scriptsize", r"\setlength{\tabcolsep}{3pt}",
       r"\begin{longtable}{P{5.2cm}P{5.2cm}rP{3.4cm}}",
       r"\caption{程序清单、行数与 SHA-256 摘要（前 8 位）；完整摘要见 \texttt{program\_inventory.json}。}\label{tab:v42-program-inventory}\\",
       r"\toprule 程序文件 & 作用 & 行数 & SHA-256（前 8 位）\\\midrule", r"\endfirsthead",
       r"\toprule 程序文件 & 作用 & 行数 & SHA-256（前 8 位）\\\midrule", r"\endhead"]
for r in rows:
    tex.append(f"\\path{{{r['path']}}} & {r['role']} & {r['lines']} & "
               f"\\texttt{{{r['sha256'][:8]}}} \\\\")
tex += [r"\bottomrule", r"\end{longtable}", r"}"]
target = paths.PAPER / "tables" / "appendix_program_inventory.tex"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text("\n".join(tex) + "\n", encoding="utf-8")
print("wrote appendix fragment ->", target.relative_to(ROOT))
