"""Regenerate all seven Question 1 figures from finalized output."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

from figure_common import record, tex  # noqa: E402
from plot_style import plt, BLUE, ORANGE, TEAL, GREY  # noqa: E402

Q1 = paths.Q1
EXT = paths.VALIDATION / "q1_sensitivity"
SENS = paths.VALIDATION / "q1_advice" / "sensitivity_summary.csv"


def quality_by_domain() -> None:
    src = Q1 / "quality_unique_domains.csv"
    d = pd.read_csv(src).sort_values("Q_mean")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(7.3, 3.6), layout="constrained")
    ax.barh(y - 0.18, d.Q_mean, height=0.34, color=BLUE, label="文档等权")
    ax.barh(y + 0.18, d.Q_token_weighted_mean, height=0.34, color=ORANGE, label="词数加权")
    ax.set_yticks(y, [tex(x) for x in d.domain])
    ax.set_xlabel(r"域级质量均值 $Q$")
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2)
    record(fig, "q1_quality_by_domain.pdf", [src], d)


def weight_sensitivity() -> None:
    src = EXT / "quality_weight_sensitivity_domains.csv"
    rank_src = EXT / "quality_weight_sensitivity_rank_corr.csv"
    d = pd.read_csv(src)
    cols = list(d.columns[1:])
    labels = ["主口径", "三组等权", "去除 DSIR", "模型主导", "表面主导",
              "纯模型", "纯 DSIR", "纯表面"]
    assert len(cols) == len(labels) == 8
    base = cols[0]
    # Spearman against the main caliber, recomputed here and checked against the
    # shipped rank-correlation matrix so the axis labels cannot drift from the data.
    rho = {c: float(pd.Series(d[base]).corr(pd.Series(d[c]), method="spearman")) for c in cols}
    ref = pd.read_csv(rank_src, index_col=0)
    for c in cols:
        assert abs(rho[c] - float(ref.loc[base, c])) < 1e-12, c
    assert abs(rho[base] - 1.0) < 1e-12
    # one colour per quality domain, reused by the legend and the extreme annotations
    palette = ["#245C85", "#B86736", "#17816F", "#7A4E9E", "#C2A02C", "#5B8C3A", "#9E3B54"]
    colour = dict(zip(d.domain, palette))
    dots = [colour[x] for x in d.domain]

    cmp_cols, cmp_labels = cols[1:], labels[1:]
    y = np.arange(len(cmp_cols))[::-1]
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(8.8, 3.9), layout="constrained",
        gridspec_kw={"width_ratios": [1.24, 1.0]},
    )
    ax.tick_params(labelsize=9)
    bx.tick_params(labelsize=8.5)
    # left: how far each caliber moves every domain's mean away from the main caliber
    ax.axvline(0, color=GREY, ls="--", lw=1)
    delta_all = d[cmp_cols].to_numpy(float) - d[[base]].to_numpy(float)
    lo, hi = float(delta_all.min()), float(delta_all.max())
    pad = 0.062
    ax.set_xlim(lo - pad, hi + pad)
    for yi, col in zip(y, cmp_cols):
        ax.axhline(yi, color=GREY, lw=.6, alpha=.35, zorder=0)
        delta = (d[col] - d[base]).to_numpy(float)
        ax.scatter(delta, np.full(len(delta), yi), s=24, zorder=3, c=dots,
                   edgecolor="white", linewidth=.4)
        j = int(np.argmax(np.abs(delta)))
        frac = (delta[j] - (lo - pad)) / ((hi + pad) - (lo - pad))
        right = frac < 0.28 or (frac <= 0.72 and delta[j] >= 0)   # annotate where there is room
        ax.annotate(f"{tex(d.domain.iloc[j])} {delta[j]:+.3f}", (delta[j], yi),
                    xytext=(6 if right else -6, 0), textcoords="offset points",
                    fontsize=8, va="center", color=colour[d.domain.iloc[j]],
                    ha="left" if right else "right", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=.82))
    ax.set_yticks(y, [f"{lab}（$\\rho$={rho[c]:.3f}）" for lab, c in zip(cmp_labels, cmp_cols)])
    ax.set_xlabel(r"相对主口径的域级均值变化 $\Delta\bar Q_d$")
    ax.set_ylim(-0.8, len(cmp_cols) - 0.2)
    ax.grid(axis="x", alpha=.2)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", ms=4.5, color=colour[x], label=tex(x))
               for x in d.domain]
    fig.legend(handles=handles, frameon=False, fontsize=7.5, ncol=7,
               loc="outside lower center", handletextpad=.25, columnspacing=1.1)

    # right: the levels themselves, unchanged
    image = bx.imshow(d[cols].to_numpy(float), vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    bx.set_xticks(range(8), labels, rotation=45, ha="right")
    bx.set_yticks(range(len(d)), [tex(x) for x in d.domain])
    fig.colorbar(image, ax=bx, label=r"域级质量均值 $\bar Q_d$", shrink=0.88)

    # tidy record of everything drawn: one row per (caliber, domain) mark
    plotted = pd.DataFrame([
        {"caliber": c, "caliber_label": lab, "domain": dom,
         "mean": float(d.loc[i, c]), "delta_vs_main": float(d.loc[i, c] - d.loc[i, base]),
         "spearman_vs_main": rho[c]}
        for c, lab in zip(cols, labels)
        for i, dom in enumerate(d.domain)
    ])
    record(fig, "q1_quality_weight_sensitivity.pdf", [src, rank_src], plotted)


def group_score_distributions() -> None:
    src = Q1 / "quality_document_scores.csv.gz"
    d = pd.read_csv(src)
    a = d.loc[d.source.eq("A1")].copy()
    groups = [("score_model", r"模型组 $M_i$"), ("score_dsir", r"DSIR 组 $U_i$"),
              ("score_surface", r"文本表面组 $R_i$")]
    domains = sorted(a.domain.unique())
    rows = []
    for domain, g in a.groupby("domain"):
        for col, label in groups:
            x = g[col].to_numpy(float)
            rows.append({"domain": domain, "group": col, "q10": np.quantile(x, .1),
                         "median": np.median(x), "q90": np.quantile(x, .9), "n": len(x)})
    summary = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(10.1, 3.3), layout="constrained", sharey=True)
    for ax, (col, label), color in zip(axes, groups, [BLUE, ORANGE, TEAL]):
        part = summary.loc[summary.group.eq(col)].set_index("domain").loc[domains]
        x = np.arange(len(domains))
        ax.vlines(x, part.q10, part.q90, color=color, lw=2.2)
        ax.scatter(x, part["median"], color=color, s=21, zorder=3)
        for threshold in (.3, .7):
            ax.axhline(threshold, color=GREY, ls="--", lw=.8)
        ax.set_xticks(x, [tex(v) for v in domains], rotation=55, ha="right")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=.15)
    record(fig, "q1_group_score_distributions.pdf", [src], summary)


def quality_function_sensitivity() -> None:
    d = pd.read_csv(SENS)
    labels = {
        "baseline": "正式主口径",
        "both_triangle_center_0.6": "三角中心 0.6",
        "both_triangle_center_0.7": "三角中心 0.7",
        "word_length_plateau_s5": "词长平台 5",
        "numeric_domain_q80_q95": "数字域内 80/95",
        "both_plateau_s5_domain_q80_q95": "联合主比较",
        "both_plateau_s10_domain_q80_q95": "联合平台 10",
        "both_plateau_s5_global_q80_q95": "联合全局阈值",
        "both_plateau_s5_domain_q75_q90": "联合域内 75/90",
        "both_plateau_s5_domain_q85_q975": "联合域内 85/97.5",
        "drop_two_surface_features": "删除两项表面信号",
    }
    assert set(d.scenario) == set(labels)
    y = np.arange(len(d))[::-1]
    colors = [BLUE if v == "baseline" else ORANGE if v == "both_plateau_s5_domain_q80_q95"
              else GREY for v in d.scenario]
    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(9.3, 4.5), sharey=True, layout="constrained",
        gridspec_kw={"width_ratios": [1, 1]},
    )
    ax.axvline(float(d.loc[d.scenario.eq("baseline"), "Q0_doc"].iloc[0]),
               color=BLUE, ls="--", lw=1)
    ax.scatter(d.Q0_doc, y, c=colors, s=28)
    bx.scatter(d.spearman_documents, y, c=colors, s=28)
    ax.set_yticks(y, [labels[x] for x in d.scenario])
    ax.set_xlabel(r"六映射域条件质量基线 $Q_0$")
    bx.set_xlabel("与正式主口径的逐文档 Spearman 相关")
    ax.set_xlim(.59, .66)
    bx.set_xlim(.98, 1.002)
    for one in (ax, bx):
        one.grid(axis="x", alpha=.2)
    record(fig, "q1_quality_function_sensitivity.pdf", [SENS], d)


def conflict_geometry() -> None:
    src = Q1 / "quality_document_scores.csv.gz"
    d = pd.read_csv(src)
    a = d.loc[d.source.eq("A1"),
              ["score_model", "score_dsir", "score_surface", "conflict",
               "pollution_conflict", "edu_ad_conflict"]].copy()
    penalized = a.pollution_conflict | a.edu_ad_conflict
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(9.1, 3.8), layout="constrained")
    cloud = ax.scatter(a.score_model, a.score_dsir, c=a.score_surface,
                       cmap="viridis", vmin=0, vmax=1, s=1.7, alpha=.50,
                       lw=0, rasterized=True)
    fig.colorbar(cloud, ax=ax, label=r"文本表面组 $R_i$", shrink=.85)
    ax.set_xlabel(r"模型组 $M_i$")
    ax.set_ylabel(r"DSIR 组 $U_i$")
    bx.scatter(a.score_model, a.score_surface, color=GREY, s=1.5, alpha=.12,
               lw=0, rasterized=True, label="A1 全部文档")
    bx.scatter(a.loc[a.conflict, "score_model"], a.loc[a.conflict, "score_surface"],
               color=ORANGE, s=5, alpha=.55, lw=0, label="组间分歧")
    bx.scatter(a.loc[penalized, "score_model"], a.loc[penalized, "score_surface"],
               color=TEAL, marker="*", s=45, lw=.4, edgecolor="white", label="显式惩罚")
    bx.set_xlabel(r"模型组 $M_i$")
    bx.set_ylabel(r"文本表面组 $R_i$")
    bx.legend(frameon=False, fontsize=7, loc="lower left")
    for one in (ax, bx):
        for threshold in (.3, .7):
            one.axvline(threshold, color=GREY, ls="--", lw=.7)
            one.axhline(threshold, color=GREY, ls="--", lw=.7)
        one.set_xlim(0, 1)
        one.set_ylim(0, 1)
    record(fig, "q1_conflict_geometry.pdf", [src], a)


def cv_model_comparison() -> None:
    src1 = Q1 / "mixture_cv_repeated.csv"
    src2 = Q1 / "solver_cv_folds.csv"
    summary = pd.read_csv(src1).set_index("model")
    folds = pd.read_csv(src2)
    models = [
        ("sqrt_quadratic_a01", "平方根二次岭"),
        ("sqrt_rbf_a001_g05", "平方根 RBF"),
        ("legacy_extra_trees", "固定随机树"),
        ("simplex_quadratic_10", "对比坐标二次岭"),
    ]
    qa = folds.loc[folds.model.eq(models[0][0]), ["repeat", "fold", "macro_rmse"]]
    rb = folds.loc[folds.model.eq(models[1][0]), ["repeat", "fold", "macro_rmse"]]
    paired = qa.merge(rb, on=["repeat", "fold"], suffixes=("_q", "_r"), validate="one_to_one")
    assert len(paired) == 15
    paired["difference"] = paired.macro_rmse_q - paired.macro_rmse_r
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(8.7, 3.4), layout="constrained")
    ypos = np.arange(4)[::-1]
    for y, (key, _) in zip(ypos, models):
        row = summary.loc[key]
        ax.errorbar(row.cv_macro_rmse, y, xerr=row.cv_macro_rmse_sd, fmt="o",
                    color=BLUE if key == models[0][0] else ORANGE if key == models[1][0] else GREY,
                    capsize=3)
    ax.set_yticks(ypos, [name for _, name in models])
    ax.set_xlabel("重复 CV 宏平均 RMSE")
    ax.set_xlim(.038, .095)
    ax.grid(axis="x", alpha=.2)
    bx.axhline(0, color=GREY, lw=.8)
    bx.scatter(np.arange(1, 16), paired.difference,
               c=np.where(paired.difference < 0, BLUE, ORANGE), s=25)
    bx.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
    bx.set_xlabel("共享验证折次")
    bx.set_ylabel("二次岭减 RBF 的 RMSE")
    bx.grid(axis="y", alpha=.2)
    record(fig, "q1_cv_model_comparison.pdf", [src1, src2], paired)


def mixture_validation() -> None:
    src = Q1 / "historical_predictions.csv"
    d = pd.read_csv(src)
    d = d.loc[d.scale.eq("1M") & d.provenance.eq("measured")
              & d.model.eq("sqrt_quadratic_a01")].sort_values("index")
    assert len(d) == 256
    fig, ax = plt.subplots(figsize=(4.6, 4.2), layout="constrained")
    x = d.observed_J_fixed_a5
    y = d.predicted_J_fixed_a5
    low = min(x.min(), y.min()) - .01
    high = max(x.max(), y.max()) + .01
    ax.scatter(x, y, color=BLUE, alpha=.65, s=17, edgecolor="none")
    ax.plot([low, high], [low, high], color=GREY, lw=.8)
    chosen = d.loc[d["index"].eq(239)].iloc[0]
    ax.scatter([chosen.observed_J_fixed_a5], [chosen.predicted_J_fixed_a5],
               color=ORANGE, s=62, marker="*", label="第 239 行")
    ax.set_xlabel(r"实测固定 A5 相对损失 $\widetilde J$")
    ax.set_ylabel(r"平方根二次面预测 $\widehat J$")
    ax.legend(frameon=False)
    ax.grid(alpha=.18)
    record(fig, "q1_mixture_1m_validation.pdf", [src], d[[
        "index", "observed_J_fixed_a5", "predicted_J_fixed_a5"
    ]])


def build() -> None:
    for fn in (quality_by_domain, weight_sensitivity, group_score_distributions,
               quality_function_sensitivity, conflict_geometry, cv_model_comparison,
               mixture_validation):
        fn()


if __name__ == "__main__":
    build()

