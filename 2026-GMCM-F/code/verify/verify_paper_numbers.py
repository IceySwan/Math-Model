"""Recompute review-critical paper claims from the shipped evidence tables.

Each assertion names its source and the manuscript file in which it appears.
This is an audit of stated values, not a claim that every number is verified.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

CONTENT = paths.PAPER / "content"
Q1 = (CONTENT / "04_question1.tex").read_text(encoding="utf-8")
Q3 = (CONTENT / "06_question3.tex").read_text(encoding="utf-8")
Q4 = (CONTENT / "07_question4.tex").read_text(encoding="utf-8")
OVERVIEW = (CONTENT / "01_problem_overview.tex").read_text(encoding="utf-8")
checks: list[str] = []


def close(label: str, actual: float, expected: float, tol: float = 1e-6) -> None:
    assert math.isclose(float(actual), expected, abs_tol=tol, rel_tol=0), (
        f"{label}: {actual} != {expected}"
    )
    checks.append(label)


def equal(label: str, actual: object, expected: object) -> None:
    assert actual == expected, f"{label}: {actual} != {expected}"
    checks.append(label)


def states(label: str, manuscript: str, fragment: str) -> None:
    assert fragment in manuscript, f"{label}: manuscript lacks {fragment!r}"
    checks.append(label)


def only(frame: pd.DataFrame, **fields: object) -> pd.Series:
    selected = frame
    for column, value in fields.items():
        selected = selected.loc[selected[column] == value]
    assert len(selected) == 1, f"Expected one row for {fields}, found {len(selected)}"
    return selected.iloc[0]


# Q1: a declared two-model set and a separate historical finite set.
evidence = paths.read_json(paths.VALIDATION / "q2_q3_checks" / "q1_checks.json")
source = pd.read_csv(paths.EVIDENCE / "q1_two_model_candidates.csv")
legacy = pd.read_csv(paths.VALIDATION / "q2_q3_checks" / "q1_two_model_candidate_checks.csv")
equal("Q1 declared candidate rows", len(source), 155)
assert source.select_dtypes(include="number").shape[1] > 0
equal("Q1 declared pair count recorded in archive", evidence["declared_two_model_set"]["negative_worst_model_rows"], 66)
equal("Q1 legacy pair count recorded in archive", evidence["legacy_two_model_set"]["negative_worst_model_rows"], 42)
equal("Q1 declared pair count recomputed", int(source.safe_in_declared_pair.sum()), 66)
equal("Q1 legacy pair count recomputed", int(legacy.safe_in_two_model_set.sum()), 42)
close("Q1 selected model historical uniform error", evidence["primary_sqrt_quadratic_uniform_error"], 0.048135492002841174, 1e-14)
close("Q1 historical regret bound", 2 * evidence["primary_sqrt_quadratic_uniform_error"], 0.09627098400568235, 1e-14)
states("Q1 declared count in body", Q1, "其中 66 个候选")
states("Q1 declared path in body", Q1, "q1_two_model_candidates.csv")
states("Q1 conditional mass equation", Q1, "0.320$ 除以该行总和 $1.002")
states("Q1 added conditional trials", Q1, "扩展域映射与连续冲突权重的条件试算")
close("Q1 fixed six-domain mass", pd.read_csv(paths.REPORTS / "review_v44" / "mapping_sensitivity.csv").iloc[0].mapped_mass, 0.319361277445, 1e-12)
equal("Q1 legacy table retained", len(legacy) > 0, True)

# Q3: the D/N table, budget-specific attention range, and saturation checks.
nominal = pd.read_csv(paths.Q3 / "nominal_configurations.csv")
exp = nominal.loc[nominal.cost_model == "exponential"]
anchor = exp.loc[exp.context_tokens == 8192]
for budget, displayed, limit in [(1e19, 0.311, 0.001), (1e22, 24.3, 0.1), (1e24, 296.8, 0.1)]:
    row = only(anchor, budget_FLOPs=budget)
    close(f"Q3 D/N at {budget:g}", row.D_B / row.N_B, displayed, limit)
states("Q3 high-budget ratio in chapter", Q3, r"$296.8$")
states("Q3 high-budget ratio in overview", OVERVIEW, r"$296.8$")
at22 = exp.loc[exp.budget_FLOPs == 1e22].attention_share * 100
close("Q3 attention minimum at 1e22", at22.min(), 5.94, 0.01)
close("Q3 attention maximum at 1e22", at22.max(), 79.36, 0.01)
close("Q3 attention full minimum", exp.attention_share.min() * 100, 4.97, 0.01)
close("Q3 attention full maximum", exp.attention_share.max() * 100, 80.72, 0.01)
states("Q3 attention scope", Q3, "三档预算与五个上下文看（指数质量成本）")
global_checks = pd.read_csv(paths.REPORTS / "review_v44" / "global_saturation_checks.csv")
receipt = paths.read_json(paths.REPORTS / "review_v44" / "receipt.json")
equal("Q3 global solve row count", len(global_checks), 3000)
equal("Q3 all drawn global qualities at upper bound", bool((global_checks.Q_star == 1).all()), True)
close("Q3 largest reported global gap", global_checks.objective_gap.max(), receipt["global_saturation_max_gap"], 1e-15)
equal("Q3 gamma cap active count", receipt["gamma_bound_active_count"], 0)
states("Q3 global checks in body", Q3, "200\\times5\\times3=3000")
states("Q3 local versus global caveat", Q3, "局部候选不被当作逐次全局切换点")
structure = pd.read_csv(paths.Q3 / "structural_sensitivity.csv")
equal("Q3 cost-half solve count", int((structure.scenario == "cost_half").sum()), 18)
equal("Q3 cost-double solve count", int((structure.scenario == "cost_double").sum()), 18)
equal("Q3 cost perturbation certificate", bool((structure.loc[structure.scenario.isin(["cost_half", "cost_double"]), "optimality_gap"] < 1e-6).all()), True)
states("Q3 cost perturbations in body", Q3, "36 个变成本")

# Q4: separate comparison strata and penalty sensitivity.
bridge = pd.read_csv(paths.Q4 / "bridge_metrics.csv")
high = {name: only(bridge, comparability="High", model=name).RMSE_pp for name in
        ("training_mean", "loss_linear", "loss_ridge10")}
medium = {name: only(bridge, comparability="Medium", model=name).RMSE_pp for name in
          ("training_mean", "loss_linear", "loss_ridge10")}
close("Q4 high mean baseline", high["training_mean"], 0.424448, 1e-6)
equal("Q4 both high Loss models worse", all(high[name] > high["training_mean"] for name in ("loss_linear", "loss_ridge10")), True)
equal("Q4 both medium Loss models improve", all(medium[name] < medium["training_mean"] for name in ("loss_linear", "loss_ridge10")), True)
close("Q4 medium linear relative improvement", 100 * (medium["loss_linear"] / medium["training_mean"] - 1), -1.94, 0.01)
close("Q4 medium ridge relative improvement", 100 * (medium["loss_ridge10"] / medium["training_mean"] - 1), -3.56, 0.01)
states("Q4 high-only verdict", Q4, "高可比层三个候选均未优于训练均值基线")
states("Q4 medium counterevidence", Q4, "Loss 线性与岭回归却分别比本层均值基线低")
penalty = pd.read_csv(paths.REPORTS / "rechecks" / "q4_lambda_sensitivity.csv")
pre = penalty["q85_pretrained_z1.0_24m"]
post = penalty["q85_posttrained_z1.0_24m"]
close("Q4 pretraining penalty spread", pre.max() - pre.min(), 0.4916, 0.0001)
close("Q4 posttraining penalty spread", post.max() - post.min(), 0.9861, 0.0001)
states("Q4 penalty stability claim withdrawn", Q4, "不能称为对惩罚强度稳定")
sensitivity = pd.read_csv(paths.Q4 / "sensitivity_metrics.csv")
lambda01 = only(sensitivity, role="reused_late", weighting="row", model="joint_size_time_lambda0.1")
lambda10 = only(sensitivity, role="reused_late", weighting="row", model="joint_size_time_lambda10.0")
close("Q4 lambda 0.1 interval score", lambda01.interval_score90, 30.318614, 1e-6)
close("Q4 lambda 10 interval score", lambda10.interval_score90, 45.90053, 1e-5)
states("Q4 lambda 0.1 score in body", Q4, "$30.318614$")
states("Q4 high-touch count in body", Q4, "其中 8 条")
aggregate = pd.read_csv(paths.VALIDATION / "q2_q3_checks" / "q4_ordered_aggregate_predictions.csv")
equal("Q4 aggregate prediction rows", len(aggregate), 64)
equal("Q4 high-touch rows at stated tolerance",
      int((np.abs(aggregate.blended_q85 - aggregate.blended_high) < 5e-14).sum()), 8)

result = {"schema": "paper-number-verification-v44/1",
          "assertion_count": len(checks), "assertions": checks,
          "scope": "Review-critical source-derived calculations and their manuscript statements; not every numeral in the paper."}
paths.write_json(paths.REPORTS / "paper_number_verification_v44.json", result)
print(f"PASS: {len(checks)} named review-critical checks")
