"""Regenerate four Q2 and five Q3 paper figures after the final algorithm reruns."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

from figure_common import record  # noqa: E402
from plot_style import plt, BLUE, ORANGE, TEAL, GREY  # noqa: E402

Q2 = paths.Q2
Q3 = paths.Q3
V37 = paths.VALIDATION / "q2_q3_checks"
COSTS = [
    ("exponential", "指数成本", BLUE),
    ("power", "幂成本", ORANGE),
    ("logarithmic", "对数成本", TEAL),
]


def q2_validation() -> None:
    src = Q2 / "validation_metrics.csv"
    d = pd.read_csv(src)
    order = [("leave_Q", "留出质量等级"), ("leave_cell", "留出完整资源单元"),
             ("leave_N", "留出参数规模"), ("leave_D", "留出数据量"),
             ("B7_new_Q", "B7 新质量等级")]
    p = d.pivot(index="protocol", columns="model", values="rmse")
    p = p.loc[[key for key, _ in order]]
    y = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(6.9, 3.2), layout="constrained")
    ax.barh(y + .18, p.power, height=.34, color=BLUE, label="常指数")
    ax.barh(y - .18, p.curved, height=.34, color=ORANGE, label="单调曲率")
    ax.set_yticks(y, [label for _, label in order])
    ax.invert_yaxis()
    ax.set_xlabel("原始损失 RMSE（越低越好）")
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    record(fig, "q2_v26_validation_rmse_comparison.pdf", [src], p.reset_index())


def q2_grouped_resource() -> None:
    src = V37 / "q2_grouped_cv_folds.csv"
    d = pd.read_csv(src)
    p = d.pivot(index="fold", columns="model", values="rmse")
    assert len(p) == 5
    x = np.arange(len(p))
    fig, ax = plt.subplots(figsize=(6.4, 3.0), layout="constrained")
    ax.bar(x - .18, p.curved_refit, width=.36, color=BLUE, label="曲率响应重拟合")
    ax.bar(x + .18, p.integrated_slope_refit, width=.36, color=TEAL,
           label="积分斜率重拟合")
    ax.set_xticks(x, [str(i + 1) for i in x])
    ax.set_xlabel("留出资源单元折次")
    ax.set_ylabel("原始损失 RMSE")
    ax.legend(frameon=False, ncol=2, fontsize=7)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    record(fig, "q2_grouped_resource_validation.pdf", [src], p.reset_index())


def q2_quality_response() -> None:
    src = Q2 / "quality_response_curves.csv"
    d = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=(6.2, 3.2), layout="constrained")
    for model, color, label in [
        ("power", BLUE, "常指数"),
        ("curved", ORANGE, "单调曲率"),
    ]:
        part = d.loc[d.model.eq(model)].sort_values("Q")
        ax.plot(part.Q, part.data_term_ratio, color=color, lw=1.8, label=label)
        ax.fill_between(part.Q, part.ratio_q025, part.ratio_q975,
                        color=color, alpha=.16, lw=0)
    ax.set_xscale("log")
    ax.set_xlim(.1, 1)
    ax.axhline(1, color=GREY, lw=.8)
    ax.set_xlabel(r"质量变量 $Q$")
    ax.set_ylabel(r"数据项相对因子 $\exp(g(\log Q))$")
    ax.grid(alpha=.18)
    ax.legend(frameon=False)
    record(fig, "q2_v26_quality_response_with_joint_band.pdf", [src], d)


def q2_parameter_profiles() -> None:
    src = Q2 / "parameter_profiles.csv"
    d = pd.read_csv(src)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), layout="constrained")
    for ax, model, color in zip(axes, ["power", "curved"], [BLUE, ORANGE]):
        part = d.loc[d.model.eq(model)]
        for parameter, ls in (("beta", "-"), ("delta", "--")):
            p = part.loc[part.parameter.eq(parameter)].sort_values("value")
            if len(p):
                ax.plot(p.value, p.excess_SSE, color=color, ls=ls, lw=1.6,
                        label=r"$\beta$" if parameter == "beta" else r"$\delta$")
        ax.axhline(0, color=GREY, lw=.8)
        ax.set_xlabel("固定参数值")
        ax.set_ylabel("相对 SSE")
        ax.legend(frameon=False)
        ax.text(.03, .96, "常指数" if model == "power" else "单调曲率",
                transform=ax.transAxes, va="top")
        ax.grid(alpha=.18)
    record(fig, "q2_v26_parameter_profiles.pdf", [src], d)


def q3_configuration() -> None:
    src = Q3 / "model_policy_comparison.csv"
    d = pd.read_csv(src)
    d = d.loc[d.context_tokens.eq(8192)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), layout="constrained")
    for key, label, color in COSTS:
        p = d.loc[d.cost_model.eq(key)].sort_values("budget_FLOPs")
        x = np.log10(p.budget_FLOPs.to_numpy(float))
        for ax, column in zip(axes, ["N_B", "D_B"]):
            ax.plot(x, p["new_" + column], "o-", color=color, lw=1.5, ms=4,
                    label=label + "：曲率")
            ax.plot(x, p["old_" + column], "x--", color=color, lw=1.0, ms=4,
                    label=label + "：常指数")
    for ax, ylabel in zip(
        axes, [r"最优参数规模 $N$（十亿）", r"最优训练量 $D$（十亿 Token）"]
    ):
        ax.set_yscale("log")
        ax.set_xticks([19, 22, 24], [r"$10^{19}$", r"$10^{22}$", r"$10^{24}$"])
        ax.set_xlabel("总预算（FLOPs）")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.17)
    axes[0].legend(frameon=False, fontsize=6.7, ncol=2, loc="upper left")
    record(fig, "q3_v26_configuration_comparison.pdf", [src], d)


def q3_regret_tradeoff() -> None:
    src = Q3 / "policy_regret_summary.csv"
    d = pd.read_csv(src)
    p = d.pivot(index=["parameter_set", "budget_FLOPs", "context_tokens", "cost_model"],
                columns="policy", values=["worst_regret", "mean_regret"])
    sets = [
        ("design_curved", "设计抽样"),
        ("evaluation_curved", "独立曲率抽样"),
        ("evaluation_power", "常指数形式敏感性"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), layout="constrained", sharey=True)
    rng = np.random.default_rng(20260925)
    saved = []
    for ax, metric, xlabel in zip(
        axes, ["worst_regret", "mean_regret"],
        ["最大后悔相对下降（百分比）", "平均后悔相对下降（百分比）"],
    ):
        for i, (key, name) in enumerate(sets):
            part = p.loc[key]
            ratio = 100 * (1 - part[(metric, "robust")] / part[(metric, "nominal")])
            ypos = i + rng.uniform(-.11, .11, len(ratio))
            ax.scatter(ratio, ypos, s=12, alpha=.65, color=BLUE)
            ax.scatter([float(np.median(ratio))], [i], marker="D", s=38, color=ORANGE)
            if metric == "worst_regret":
                saved.extend({"set": key, "relative_worst_reduction_pct": float(v)}
                             for v in ratio)
        ax.axvline(0, color=GREY, lw=.8)
        ax.set_yticks(range(3), [name for _, name in sets])
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=.18)
    axes[0].invert_yaxis()
    record(fig, "q3_v26_regret_tradeoff_across_scenarios.pdf", [src], pd.DataFrame(saved))


def q3_cost_of_robustness() -> None:
    src = Q3 / "robust_configurations.csv"
    d = pd.read_csv(src)
    fig, ax = plt.subplots(figsize=(6.1, 3.5), layout="constrained")
    for key, label, color in COSTS:
        p = d.loc[d.cost_model.eq(key)]
        ax.scatter(100 * p.relative_regret_reduction, p.nominal_model_penalty,
                   color=color, s=33, alpha=.8, label=label)
    ax.axhline(0, color=GREY, lw=.8)
    ax.set_xlabel("最大后悔值相对下降（百分比）")
    ax.set_ylabel("名义损失代价")
    ax.grid(alpha=.18)
    ax.legend(frameon=False)
    record(fig, "q3_v26_nominal_cost_of_robustness.pdf", [src], d)


def q3_cvar() -> None:
    src = V37 / "q3_policy_evaluation.csv"
    d = pd.read_csv(src)
    d = d.loc[d.budget_FLOPs.eq(1e19) & d.cost_model.eq("power")].copy()
    labels = {
        "nominal": "名义", "minimax": "最大后悔", "CVaR_omega_0": "均值",
        "CVaR_omega_0.5": "均值--CVaR", "CVaR_omega_1": "CVaR",
    }
    fig, ax = plt.subplots(figsize=(5.8, 3.3), layout="constrained")
    for _, r in d.iterrows():
        color = BLUE if r.policy == "nominal" else ORANGE
        ax.scatter(r.evaluation_mean, r.evaluation_cvar90, color=color, s=36)
        ax.annotate(labels[r.policy], (r.evaluation_mean, r.evaluation_cvar90),
                    xytext=(3, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("第二组重抽样平均后悔值")
    ax.set_ylabel(r"第二组重抽样 $\mathrm{CVaR}_{0.9}$")
    ax.grid(alpha=.17)
    record(fig, "q3_cvar_tradeoff_power_1e19.pdf", [src], d)


def q3_transition_sensitivity() -> None:
    new_src = Q3 / "saturation_transitions.csv"
    old_src = paths.Q3 / "structure_transitions.csv"
    new = pd.read_csv(new_src)
    old = pd.read_csv(old_src)
    contexts = sorted(new.context_tokens.unique())
    fig, ax = plt.subplots(figsize=(7.2, 3.2), layout="constrained")
    for key, label, color in COSTS:
        a = new.loc[new.cost_model.eq(key)].sort_values("context_tokens")
        b = old.loc[old.cost_model.eq(key)].sort_values("context_tokens")
        ax.plot(contexts, np.log10(a.transition_estimate_FLOPs),
                "o-", color=color, lw=1.5, label=label + "：曲率")
        ax.plot(contexts, np.log10(b.transition_budget_FLOPs),
                "x--", color=color, lw=1.0, label=label + "：常指数")
    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts, [str(v) for v in contexts])
    ax.set_xlabel("上下文长度（Token）")
    ax.set_ylabel(r"转移预算 $\log_{10}(C/\mathrm{FLOPs})$")
    ax.grid(alpha=.18)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    record(fig, "q3_v26_transition_model_sensitivity.pdf", [new_src, old_src],
           new.merge(old, on=["context_tokens", "cost_model"], suffixes=("_curved", "_power")))


def build() -> None:
    for fn in (
        q2_validation, q2_grouped_resource, q2_quality_response,
        q2_parameter_profiles, q3_configuration, q3_regret_tradeoff,
        q3_cost_of_robustness, q3_cvar, q3_transition_sensitivity,
    ):
        fn()


if __name__ == "__main__":
    build()
