"""Regenerate all eight Question 4 paper figures from the final outputs."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

from figure_common import record  # noqa: E402
from plot_style import plt, BLUE, ORANGE, TEAL, GREY  # noqa: E402

Q4 = paths.Q4
SLOW = paths.SLOWDOWN
DECOMP = paths.Q4 / "decomposition_sensitivity.csv"
TASKS = paths.Q4 / "c8_c1_normalized_comparison.csv"
ROLE = {
    "development": "开发期", "straddling": "跨界期",
    "reused_late": "复用晚期", "publisher_holdout": "整组织留出",
}


def task_alignment() -> None:
    d = pd.read_csv(TASKS)
    p = d.loc[d.layer.eq("all_licensed")].copy()
    x = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(7.2, 3.1), layout="constrained")
    ax.bar(x - .18, p.raw_RMSE, width=.36, color=BLUE, label="原始 C8 标尺")
    ax.bar(x + .18, p.normalized_RMSE, width=.36, color=ORANGE,
           label="任务归一化后")
    ax.set_yscale("log")
    ax.set_xticks(x, [r"\texttt{IFEval}", r"\texttt{BBH}", "MATH Lvl 5",
                       r"\texttt{GPQA}", r"\texttt{MUSR}", "MMLU-PRO"])
    ax.set_ylabel("与 C1 的 RMSE（分，对数刻度）")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    record(fig, "q4_v26_task_alignment.pdf", [TASKS], p)


def observational_decomposition() -> None:
    d = pd.read_csv(DECOMP)
    specs = [
        ("pooled_type_controls", "类型控制"),
        ("publisher_fixed_effects", "组织固定效应"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2), sharey=True, layout="constrained")
    for ax, (key, title) in zip(axes, specs):
        p = d.loc[d.specification.eq(key)].sort_values("window_days")
        assert p.window_days.tolist() == [30, 60, 90, 120]
        x = np.arange(len(p))
        ax.bar(x - .18, p.scale_pp, width=.34, color=BLUE, label="规模分量")
        ax.bar(x + .18, p.time_residual_pp, width=.34, color=ORANGE,
               label="时间残差")
        ax.plot(x, p.total_pp, color="black", lw=1.0, marker="o", ms=4,
                label="两项合计")
        ax.axhline(0, color=GREY, lw=.7)
        ax.set_xticks(x, [str(v) for v in p.window_days])
        ax.set_xlabel("首尾窗口长度（日）")
        ax.text(.03, .96, title, transform=ax.transAxes, va="top")
        ax.grid(axis="y", alpha=.18)
    axes[0].set_ylabel("拟合分数变化（分）")
    axes[0].legend(frameon=False, ncol=3, fontsize=7, loc="lower center",
                   bbox_to_anchor=(1.04, 1.01))
    record(fig, "q4_observational_decomposition.pdf", [DECOMP], d)


def metric_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    rolling = pd.read_csv(Q4 / "rolling_metrics.csv")
    publisher = pd.read_csv(Q4 / "publisher_metrics.csv")
    rolling = rolling.loc[rolling.input_mode.eq("resource_forecast")
                          & rolling.weighting.eq("row")].copy()
    publisher = publisher.loc[publisher.weighting.eq("row")].copy()
    publisher["role"] = "publisher_holdout"
    return rolling, publisher


def validation() -> None:
    rolling, publisher = metric_data()
    models = [
        ("type_empirical_180d", "经验分位", GREY),
        ("auxiliary_size_time_ordered", "辅助规模--时间", TEAL),
        ("joint_time", "联合时间", ORANGE),
        ("joint_size_time", "联合规模--时间", BLUE),
    ]
    roles = ["development", "straddling", "reused_late", "publisher_holdout"]
    frame = pd.concat([rolling, publisher], ignore_index=True)
    p = frame.loc[frame.model.isin([m for m, _, _ in models])]
    p = p.drop_duplicates(["role", "model"]).set_index(["role", "model"])
    fig, ax = plt.subplots(figsize=(8.3, 3.4), layout="constrained")
    x = np.arange(len(roles))
    for j, (model, label, color) in enumerate(models):
        y = [p.loc[(role, model), "pinball85"] for role in roles]
        ax.bar(x + (j - 1.5) * .19, y, width=.18, color=color, label=label)
    ax.set_xticks(x, [ROLE[r] for r in roles])
    ax.set_ylabel(r"$0.85$ 分位 Pinball 损失（分）")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    record(fig, "q4_v26_validation.pdf",
           [Q4 / "rolling_metrics.csv", Q4 / "publisher_metrics.csv"],
           p.reset_index())


def calibration() -> None:
    src = Q4 / "rolling_metrics.csv"
    d = pd.read_csv(src)
    models = [
        ("joint_size_time", "全历史拟合"),
        ("joint_size_time_split_raw", "切分未校准"),
        ("joint_size_time_split_row90", "记录校准"),
        ("joint_size_time_split_publisher90", "组织校准"),
    ]
    p = d.loc[d.role.eq("reused_late") & d.input_mode.eq("resource_forecast")
              & d.weighting.eq("row") & d.model.isin([m for m, _ in models])]
    p = p.set_index("model").loc[[m for m, _ in models]]
    x = np.arange(len(models))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.2, 3.1), layout="constrained")
    ax.bar(x, 100 * p.coverage90, color=BLUE)
    bx.bar(x, p.width_pp, color=ORANGE)
    for one in (ax, bx):
        one.set_xticks(x, [label for _, label in models], rotation=30, ha="right")
        one.grid(axis="y", alpha=.18)
        one.set_axisbelow(True)
    ax.set_ylim(0, 108)
    ax.set_ylabel("90\\% 区间经验覆盖率")
    bx.set_ylabel("平均区间宽度（分）")
    record(fig, "q4_v26_calibration.pdf", [src], p.reset_index())


def ordered_aggregation() -> None:
    rolling, publisher = metric_data()
    roles = ["development", "straddling", "reused_late", "publisher_holdout"]
    frame = pd.concat([rolling, publisher], ignore_index=True)
    p = frame.loc[frame.model.isin(["joint_size_time", "joint_time"])]
    p = p.drop_duplicates(["role", "model"]).set_index(["role", "model"])
    rows = [
        {"role": r, "size_time": p.loc[(r, "joint_size_time"), "pinball85"],
         "time": p.loc[(r, "joint_time"), "pinball85"]}
        for r in roles
    ]
    d = pd.DataFrame(rows)
    d["development_weighted_blend"] = d.size_time  # frozen weight is 1
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.0, 3.2), layout="constrained")
    ax.bar(x - .19, d.size_time, width=.38, color=BLUE, label="规模--时间联合")
    ax.bar(x + .19, d.time, width=.38, color=ORANGE, label="时间联合")
    ax.scatter(x - .19, d.development_weighted_blend, color="black", marker="_",
               s=110, label="开发期定权组合")
    ax.set_xticks(x, [ROLE[r] for r in d.role])
    ax.set_ylabel(r"$0.85$ 分位 Pinball 损失（分）")
    ax.legend(frameon=False, fontsize=7, ncol=3)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    record(fig, "q4_ordered_aggregation_retrospective.pdf",
           [Q4 / "rolling_metrics.csv", Q4 / "publisher_metrics.csv"], d)


def resource_paths() -> None:
    src = Q4 / "resource_paths_point.csv"
    d = pd.read_csv(src)
    keys = [
        ("q3_exponential", "指数成本", BLUE),
        ("q3_power", "幂成本", ORANGE),
        ("q3_logarithmic", "对数成本", TEAL),
    ]
    d = d.loc[d.growth_factor.eq(1) & d.conversion.isin([k for k, _, _ in keys])]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.2, 3.3), layout="constrained")
    for key, label, color in keys:
        p = d.loc[d.conversion.eq(key)].sort_values("horizon_years")
        assert p.horizon_years.tolist() == [1, 2]
        x = np.r_[0, p.horizon_years]
        ax.plot(x, np.r_[1, p.N_multiplier], color=color, marker="o",
                lw=1.6, label=label)
        bx.plot(x, np.r_[1, p.D_multiplier], color=color, marker="o", lw=1.6)
    for one in (ax, bx):
        one.set_xticks([0, 1, 2])
        one.set_xlabel("距 2025-03-13 的年数")
        one.grid(alpha=.18)
    ax.set_ylabel(r"参数规模 $N$ 相对倍率")
    bx.set_ylabel(r"训练量 $D$ 相对倍率")
    ax.legend(frameon=False)
    record(fig, "q4_v26_resource_paths.pdf", [src], d)


def slowdown_frontier() -> None:
    src = SLOW / "q4_slowdown_forecast_table.csv"
    d = pd.read_csv(src)
    assert len(d) == 12 and d.draw_count.eq(199).all()
    groups = [("pretrained", "预训练"), ("posttrained", "后训练")]
    growth = [(0.0, "零增长", GREY), (0.5, "增速减半", ORANGE),
              (1.0, "历史增速", TEAL)]
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.2), sharey=True, layout="constrained")
    for ax, (group, label) in zip(axes, groups):
        for zeta, name, color in growth:
            p = d.loc[d.group.eq(group) & d.zeta.eq(zeta)].sort_values("horizon_years")
            x = 12 * p.horizon_years.to_numpy(float)
            ax.plot(x, p.point_q85, color=color, marker="o", lw=1.5, label=name)
            ax.fill_between(x, p.curve_5, p.curve_95, color=color, alpha=.13)
        ax.set_xticks([12, 24])
        ax.set_xlim(9, 27)
        ax.set_xlabel("距历史起点的月数")
        ax.text(.03, .96, label, transform=ax.transAxes, va="top")
        ax.grid(alpha=.18)
    axes[0].set_ylabel(r"$0.85$ 分位条件前沿分数")
    axes[0].legend(frameon=False, fontsize=7, ncol=1, loc="lower right")
    record(fig, "q4_slowdown_frontier.pdf", [src], d)


def conditional_forecast() -> None:
    src = Q4 / "forecast_curve_intervals.csv"
    d = pd.read_csv(src)
    p = d.loc[d.model.eq("joint_size_time")
              & d.conversion.eq("q3_exponential")
              & d.growth_factor.eq(1)
              & d.trend_decay.eq(0)].copy()
    assert len(p) == 4 and p.draws.eq(199).all()
    fig, ax = plt.subplots(figsize=(6.7, 3.3), layout="constrained")
    for group, name, color in [
        ("pretrained", "预训练", BLUE), ("posttrained", "后训练", ORANGE)
    ]:
        s = p.loc[p.group.eq(group)].sort_values("horizon_years")
        x = 12 * s.horizon_years.to_numpy(float)
        ax.plot(x, s.q85, color=color, marker="o", lw=1.7, label=name + r" $q_{0.85}$")
        ax.fill_between(x, s.curve_q05, s.curve_q95, color=color, alpha=.16)
        ax.plot(x, s.q05, color=color, ls=":", lw=.9)
        ax.plot(x, s.q95, color=color, ls=":", lw=.9)
    ax.set_xticks([12, 24])
    ax.set_xlim(9, 27)
    ax.set_xlabel("距历史起点的月数")
    ax.set_ylabel("条件分数")
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=7)
    record(fig, "q4_v26_conditional_forecast.pdf", [src], p)


def build() -> None:
    for fn in (
        task_alignment, observational_decomposition, validation, calibration,
        ordered_aggregation, resource_paths, slowdown_frontier,
        conditional_forecast,
    ):
        fn()


if __name__ == "__main__":
    build()
