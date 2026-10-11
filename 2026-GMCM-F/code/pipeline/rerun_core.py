"""Isolated rerun of the shipped Q2/Q3/Q4 core programs.

Each stage imports the shipped module and then points its output globals at the
package output directory (``data/outputs/q2`` .. ``data/outputs/q4``); the module
sources are never modified.  Run the stages in order: q2, q3, q4_fit,
q4_forecast.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths


def module_at(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def q2() -> None:
    folder = paths.CODE / "q2"
    sys.path[:0] = [str(folder), str(paths.CODE)]
    run = module_at("final_q2_run", folder / "run_study.py")
    run.OUT = paths.Q2
    with threadpool_limits(limits=1):
        run.main()


def q3() -> None:
    folder = paths.CODE / "q3"
    sys.path[:0] = [str(folder), str(paths.CODE)]
    run = module_at("final_q3_run", folder / "run_study.py")
    run.OUT = paths.Q3
    run.Q2 = paths.Q2
    if not (run.Q2 / "run_manifest.json").is_file():
        raise FileNotFoundError("Run q2 first")
    with threadpool_limits(limits=1):
        run.main()


def q4_fit() -> None:
    folder = paths.CODE / "q4"
    sys.path[:0] = [str(folder), str(paths.CODE)]
    config = module_at("data_config", folder / "data_config.py")
    config.OUT = paths.Q4
    run = module_at("final_q4_run", folder / "run_study.py")
    run.OUT = config.OUT
    with threadpool_limits(limits=1):
        run.main()


def q4_forecast() -> None:
    folder = paths.CODE / "q4"
    sys.path[:0] = [str(folder), str(paths.CODE)]
    config = module_at("data_config", folder / "data_config.py")
    config.OUT = paths.Q4
    forecast = module_at("final_q4_forecast", folder / "resource_forecast.py")
    forecast.OUT = config.OUT
    q2out = paths.Q2
    q3out = paths.Q3
    if not (config.OUT / "selection.json").is_file():
        raise FileNotFoundError("Run q4_fit first")
    if not (q2out / "q2_handoff_v26.json").is_file() or not (q3out / "q3_handoff_v26.json").is_file():
        raise FileNotFoundError("Run q2 and q3 first")

    def final_upstream():
        q2h = json.loads((q2out / "q2_handoff_v26.json").read_text(encoding="utf-8"))
        q3h = json.loads((q3out / "q3_handoff_v26.json").read_text(encoding="utf-8"))
        if q2h["selected_model"] != "curved" or q3h["schema"] != "q3-curved-budget-v26/1":
            raise ValueError("Unexpected Q2/Q3 final models")
        if (not all(q2h["parameters"][key] == value for key, value in q3h["params"].items())
                or q2h["Q1_anchor_Q"] != q3h["Q0"]):
            raise ValueError("Q2 to Q3 final handoff mismatch")
        draws = pd.read_csv(q2out / "joint_parameter_draws.csv")
        draws = draws.loc[draws.model.eq("curved")].sort_values("draw")
        if len(draws) != 200 or draws.draw.tolist() != list(range(200)):
            raise ValueError("Incomplete paired Q2 parameter draws")
        params = [forecast.q3.Params.from_dict(row) for row in draws.to_dict("records")]
        contract = {
            "Q3_schema": q3h["schema"],
            "Q2_handoff_sha256": paths.digest(q2out / "q2_handoff_v26.json"),
            "Q2_draws_sha256": paths.digest(q2out / "joint_parameter_draws.csv"),
            "Q3_handoff_sha256": paths.digest(q3out / "q3_handoff_v26.json"),
            "Q3_manifest_sha256": paths.digest(q3out / "run_manifest.json"),
            "Q0": q3h["Q0"],
            "parameters": q3h["params"],
            "source_support": q2h["support"],
            "score_bridge": "B1 real Pythia only; B6 Loss is not mapped to benchmark scores",
            "rerun_scope": "new isolated Q2/Q3 outputs",
        }
        paths.write_json(config.OUT / "upstream_contract.json", contract)
        return forecast.q3.Params.from_dict(q3h["params"]), params, q3h["Q0"], q2h

    forecast.upstream = final_upstream
    with threadpool_limits(limits=1):
        forecast.main()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("q2", "q3", "q4_fit", "q4_forecast"))
    stage = parser.parse_args().stage
    globals()[stage]()


if __name__ == "__main__":
    main()
