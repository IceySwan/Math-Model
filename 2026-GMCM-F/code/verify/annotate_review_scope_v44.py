"""Clarify scopes of older evidence fields without changing their values."""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def migrate_q1(obj: dict) -> None:
    if "after_the_fact_uniform_error" in obj:
        err = obj.pop("after_the_fact_uniform_error")
        bound = obj.pop("two_epsilon_bound")
        assert math.isclose(err, 0.06510834627020068, abs_tol=1e-15)
        assert math.isclose(bound, 2 * err, abs_tol=1e-15)
        obj["uniform_error_worst_of_two_models"] = err
        obj["two_epsilon_bound_worst_of_two_models"] = bound
    obj["primary_sqrt_quadratic_uniform_error"] = 0.048135492002841174
    obj["uniform_error_scope"] = (
        "0.065108346 is the larger historical error of the older two-model comparison, "
        "driven by sqrt_rbf_a001_g05; the manuscript's single sqrt_quadratic_a01 "
        "uses 0.048135492002841174 and its bound 0.09627098400568235."
    )
    if "finite_model_set" in obj:
        old = obj.pop("finite_model_set")
        assert old["common_candidate_rows"] == 155
        assert old["negative_worst_model_rows"] == 42
        old["models"] = ["older_quadratic_response", "sqrt_rbf_a001_g05"]
        old["source"] = "data/validation/q2_q3_checks/q1_two_model_candidate_checks.csv"
        old["scope"] = (
            "Historical alternative pair only. The declared manuscript pair "
            "sqrt_quadratic_a01 versus sqrt_rbf_a001_g05 has 66/155 safe rows "
            "in data/outputs/evidence/q1_two_model_candidates.csv."
        )
        obj["legacy_two_model_set"] = old
    obj["declared_two_model_set"] = {
        "models": ["sqrt_quadratic_a01", "sqrt_rbf_a001_g05"],
        "common_candidate_rows": 155,
        "negative_worst_model_rows": 66,
        "source": "data/outputs/evidence/q1_two_model_candidates.csv",
    }


def main() -> None:
    q1 = paths.VALIDATION / "q2_q3_checks" / "q1_checks.json"
    manifest = paths.VALIDATION / "q2_q3_checks" / "run_manifest.json"
    evidence = paths.OUT / "evidence" / "algorithm_checks.json"
    contract = paths.OUT / "evidence" / "final_algorithm_contract.json"
    targets = [q1, manifest, evidence, contract]
    before = {p.name: sha(p) for p in targets}

    checked = json.loads(q1.read_text(encoding="utf-8"))
    migrate_q1(checked)
    write(q1, checked)
    run = json.loads(manifest.read_text(encoding="utf-8"))
    migrate_q1(run["checks"]["q1"])
    write(manifest, run)

    out = json.loads(evidence.read_text(encoding="utf-8"))
    q2 = out["q2"]
    infimum = q2.pop("gamma_min", None)
    supremum = q2.pop("gamma_max", None)
    if infimum is not None:
        assert math.isclose(infimum, 0.0, abs_tol=1e-15)
        assert math.isclose(supremum, 2.4053237313207445, abs_tol=1e-12)
        q2["gamma_inf_fitted_full_support"] = infimum
        q2["gamma_sup_fitted_full_support"] = supremum
    handoff = json.loads((paths.Q2 / "q2_handoff_v26.json").read_text(encoding="utf-8"))
    par = handoff["parameters"]
    q0 = handoff["Q1_anchor_Q"]
    q2["gamma_at_Q0"] = (
        par["delta"] - 2 * par["kappa"] * math.log(q0)
    ) / par["beta"]
    q2["gamma_scope"] = (
        "Full-support extrema are on Q in [0.1,1]; Q3 optimizes on [Q0,1], "
        "whose lower fitted slope ratio is gamma_at_Q0. Shape cap is gamma<=5."
    )
    write(evidence, out)

    doc = json.loads(contract.read_text(encoding="utf-8"))
    doc["theorem_evidence"]["q2_shape_and_validation"] = (
        "q2.gamma_inf_fitted_full_support, gamma_sup_fitted_full_support, "
        "gamma_at_Q0 and leave_Q/leave_cell_relative_gain"
    )
    write(contract, doc)

    receipt = {
        "schema": "review-v44-scope-annotation/1",
        "changes": [
            "Renamed historical two-model Q1 epsilon and candidate-set fields with scope.",
            "Distinguished declared sqrt-quadratic/RBF pair (66) from older pair (42).",
            "Renamed full-support gamma extrema and added gamma at Q0.",
        ],
        "before_sha256": before,
        "after_sha256": {p.name: sha(p) for p in targets},
    }
    output = paths.REPORTS / "review_v44" / "archive_scope_migration.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write(output, receipt)
    print("PASS: scope annotations written without changing stored numeric values")


if __name__ == "__main__":
    main()
