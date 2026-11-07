"""Subprocess entry point for the heavy MCP operations.

``python -m pear_sim.mcp.worker`` reads one JSON request on stdin, runs it
through the public Python API and prints one JSON result on stdout. Running in
a child process gives the server two things it cannot get in-process: a hard
wall-clock limit (the parent kills the child) and isolation (a pathological
model cannot take the server down with it). The code path is the same one the
CLI uses, so a run is bit-identical whichever surface started it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _simulate(request: dict[str, Any]) -> dict[str, Any]:
    from pear_sim.api.facade import simulate

    bundle = simulate(request["model"], request["out"], request.get("overrides") or None)
    return {"bundle": str(bundle.output_dir)}


def _steady_state(request: dict[str, Any]) -> dict[str, Any]:
    from pear_sim.api.facade import steady_state

    result = steady_state(request["model"], request.get("overrides") or None).as_dict()
    if result.get("converged") and request.get("cycle_out"):
        result["steady_cycle"] = _steady_cycle(request, result)
    return result


def _steady_cycle(request: dict[str, Any], solved: dict[str, Any]) -> dict[str, Any]:
    """Re-run the model for whole periods starting exactly on the periodic orbit.

    A transient's "last half" still carries whatever start-up oscillation has not
    decayed, which on a lightly damped LC filter can exceed the switching ripple
    many times over. Starting from the shooting solution removes that: what is
    left is the steady-state cycle itself.
    """

    import yaml

    from pear_sim.api.facade import simulate
    from pear_sim.spec.model import _ModelSafeLoader

    document = yaml.load(Path(request["model"]).read_text(encoding="utf-8"), Loader=_ModelSafeLoader)
    components = {c.get("id"): c for c in document.get("circuit", {}).get("components", [])}
    fields = {"inductor": ("initial_current", "A"), "capacitor": ("initial_voltage", "V")}
    for name, value in zip(solved["state_names"], solved["state"]):
        component = components.get(name)
        if component is None or component.get("type") not in fields:
            return {
                "status": "not_run",
                "reason": f"state {name!r} is not a top-level inductor or capacitor",
            }
        field, unit = fields[component["type"]]
        component[field] = f"{value!r} {unit}"
    periods = int(request.get("cycle_periods", 2))
    # The shooting map simulates each period from t = 0 (one_cycle_model), so
    # the solved state belongs to the PWM phase at t = 0, whatever the model's
    # own start time is.
    transient = document["analysis"]["transient"]
    transient["start"] = "0 s"
    transient["stop"] = f"{periods * solved['period_s']!r} s"
    cycle_model = Path(request["cycle_out"]) / "steady-cycle-model.json"
    cycle_model.parent.mkdir(parents=True, exist_ok=True)
    cycle_model.write_text(json.dumps(document), encoding="utf-8")
    bundle = simulate(cycle_model, Path(request["cycle_out"]) / "run", request.get("overrides") or None)
    return {"status": "run", "periods": periods, "bundle": str(bundle.output_dir)}


def _stability(request: dict[str, Any]) -> dict[str, Any]:
    from pear_sim.evidence.stability import run_stability_analysis

    artifacts = run_stability_analysis(
        request["model"], request["out"], request=request.get("request")
    )
    return {"bundle": str(Path(artifacts.output_dir))}


def _sweep_parameter(request: dict[str, Any]) -> dict[str, Any]:
    from pear_sim.evidence.parameter_sweep import run_parameter_sweep

    artifacts = run_parameter_sweep(request["model"], request["request_path"], request["out"])
    return {
        "bundle": str(artifacts.output_dir),
        "summary": str(artifacts.sweep_summary),
        "execution_status": artifacts.execution_status,
    }


def _tolerance_corners(request: dict[str, Any]) -> dict[str, Any]:
    from pear_sim.evidence.tolerance_corners import run_tolerance_corners

    artifacts = run_tolerance_corners(request["model"], request["request_path"], request["out"])
    return {
        "bundle": str(artifacts.output_dir),
        "summary": str(artifacts.summary),
        "execution_status": artifacts.execution_status,
    }


def _evaluate_designs(request: dict[str, Any]) -> dict[str, Any]:
    from pear_sim.evidence.batch import run_batch

    artifacts = run_batch(request["model"], request["request_path"], request["out"])
    return {"bundle": str(artifacts.output_dir), "summary": str(artifacts.summary)}


OPERATIONS = {
    "simulate": _simulate,
    "steady_state": _steady_state,
    "stability": _stability,
    "sweep_parameter": _sweep_parameter,
    "tolerance_corners": _tolerance_corners,
    "evaluate_designs": _evaluate_designs,
}


def main() -> int:
    from pear_sim.api.facade import error_code

    try:
        request = json.loads(sys.stdin.read())
        handler = OPERATIONS[request["op"]]
        payload = {"ok": True, "result": handler(request)}
    except Exception as exc:  # noqa: BLE001 - reported to the parent, never swallowed
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "code": error_code(exc),
        }
    sys.stdout.write(json.dumps(payload) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
