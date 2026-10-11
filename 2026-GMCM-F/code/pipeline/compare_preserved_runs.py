"""Compare key isolated final numerical outputs with the preserved prior runs.

The rerun outputs live in ``data/outputs``; the preserved earlier-run files were
copied to ``data/validation/preserved_runs``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths

PRESERVED = paths.VALIDATION / "preserved_runs"
RERUN = {"q1": paths.Q1, "q2": paths.Q2, "q3": paths.Q3, "q4": paths.Q4}

PAIRS = {
    "q1": (PRESERVED / "q1",
           ["historical_predictions.csv", "solver_selection.json"]),
    "q2": (PRESERVED / "q2",
           ["validation_metrics.csv", "full_fit_parameters.json",
            "quality_response_curves.csv"]),
    "q3": (PRESERVED / "q3",
           ["nominal_configurations.csv", "robust_configurations.csv",
            "saturation_transitions.csv"]),
    "q4": (PRESERVED / "q4",
           ["rolling_metrics.csv", "forecast_scenarios.csv", "full_fit.json"]),
}


def compare_csv(new: Path, old: Path) -> dict:
    a, b = pd.read_csv(new), pd.read_csv(old)
    assert a.shape == b.shape and a.columns.tolist() == b.columns.tolist(), new.name
    max_difference = 0.0
    for col in a.columns:
        left, right = a[col], b[col]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            av, bv = left.to_numpy(float), right.to_numpy(float)
            assert np.allclose(av, bv, atol=1e-6, rtol=1e-8, equal_nan=True), (new.name, col)
            valid = np.isfinite(av) & np.isfinite(bv)
            if valid.any():
                max_difference = max(max_difference, float(np.max(np.abs(av[valid] - bv[valid]))))
        else:
            assert left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)), (new.name, col)
    return {"rows": len(a), "columns": len(a.columns), "max_numeric_difference": max_difference}


def compare_json(new: Path, old: Path) -> dict:
    a = json.loads(new.read_text(encoding="utf-8"))
    b = json.loads(old.read_text(encoding="utf-8"))
    if new.name == "solver_selection.json":
        assert a["selected_model"] == b["selected_model"]
        return {"selected_model": a["selected_model"]}
    if new.name == "full_fit_parameters.json":
        assert a == b
        return {"identical_parsed_json": True}
    if new.name == "full_fit.json":
        for key in ("model", "coef", "center", "scale", "names", "aux_slope"):
            if isinstance(a[key], list) and a[key] and isinstance(a[key][0], (int, float)):
                assert np.allclose(a[key], b[key], atol=1e-8, rtol=1e-8)
            else:
                assert a[key] == b[key]
        return {"fit_parameters_match": True}
    raise AssertionError(new.name)


def main() -> None:
    result = {"schema": "isolated-rerun-comparison/1", "pairs": {}}
    for question, (older, filenames) in PAIRS.items():
        for name in filenames:
            new = RERUN[question] / name
            old = older / name
            assert new.is_file() and old.is_file(), name
            result["pairs"][f"{question}/{name}"] = (
                compare_csv(new, old) if name.endswith(".csv") else compare_json(new, old)
            )
    target = paths.REPORTS / "preserved_run_comparison.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"PASS: {len(result['pairs'])} isolated numerical outputs match preserved runs")


if __name__ == "__main__":
    main()
