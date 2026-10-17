"""Tolerance-corner sweeps: worst-case-of-the-batch evidence."""

import json
from pathlib import Path

import pytest

from pear_sim.evidence.parameter_sweep import ParameterSweepError
from pear_sim.evidence.tolerance_corners import run_tolerance_corners

MODEL = """
schema: pear-sim/v0.1
name: rc_tolerances

parameters:
  r: 1 kohm
  c: 100 nF

circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 10 V}
    - {id: r1, type: resistor, p: in, n: out, value: "${r}"}
    - {id: c1, type: capacitor, p: out, n: gnd, value: "${c}", initial_voltage: 0 V}

analysis:
  transient:
    start: 0 s
    stop: 1 ms
    output_step: 5 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11

outputs:
  - v(out)
"""


def _write_inputs(tmp_path: Path, *, tolerances, band_metrics=None) -> tuple[Path, Path]:
    model_path = tmp_path / "model.yaml"
    model_path.write_text(MODEL, encoding="utf-8")
    request = {
        "schema": "pear-sim/tolerance-corners-request/v0.1",
        "tolerances": tolerances,
    }
    if band_metrics is not None:
        request["band_metrics"] = band_metrics
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    return model_path, request_path


def test_two_parameter_corners_produce_the_expected_band(tmp_path: Path) -> None:
    model_path, request_path = _write_inputs(
        tmp_path,
        tolerances=[
            {"parameter": "r", "percent": 20},
            {"parameter": "c", "percent": 20},
        ],
        band_metrics=[{"signal": "v(out)", "field": "sample_final"}],
    )
    artifacts = run_tolerance_corners(model_path, request_path, tmp_path / "out")
    summary = json.loads(artifacts.summary.read_text())
    assert summary["execution_status"] == "pass"
    assert summary["acceptance_status"] == "not_run"  # no oracle -> never positive
    assert summary["corner_count"] == 5  # nominal + 4 corners
    ids = [row["corner_id"] for row in summary["corners"]]
    assert ids[0] == "nominal"
    assert set(ids[1:]) == {"corner-r-c-", "corner-r-c+", "corner-r+c-", "corner-r+c+"}

    band = summary["bands"]["v(out)::sample_final"]
    # RC step at t = 1 ms: v = 10*(1-exp(-t/RC)). Slowest charging (max tau =
    # 1.2R * 1.2C) gives the LOWEST final voltage; fastest gives the highest.
    assert band["min_corner"] == "corner-r+c+"
    assert band["max_corner"] == "corner-r-c-"
    assert band["min"] < band["nominal"] < band["max"]
    import math

    tau = 1e3 * 100e-9
    assert band["nominal"] == pytest.approx(10.0 * (1.0 - math.exp(-1e-3 / tau)), rel=1e-6)
    tau_slow = (1.2e3) * (120e-9)
    assert band["min"] == pytest.approx(10.0 * (1.0 - math.exp(-1e-3 / tau_slow)), rel=1e-6)


def test_asymmetric_tolerances_and_run_retention(tmp_path: Path) -> None:
    model_path, request_path = _write_inputs(
        tmp_path,
        tolerances=[{"parameter": "r", "minus_percent": 10, "plus_percent": 50}],
    )
    artifacts = run_tolerance_corners(model_path, request_path, tmp_path / "out")
    summary = json.loads(artifacts.summary.read_text())
    assert summary["corner_count"] == 3
    for row in summary["corners"]:
        run_dir = artifacts.output_dir / row["run"]
        assert (run_dir / "run.json").is_file()
        assert (run_dir / "metrics.json").is_file()
    overrides = {
        row["corner_id"]: row["overrides_si"]["r"] for row in summary["corners"]
    }
    assert overrides["corner-r-"] == pytest.approx(900.0)
    assert overrides["corner-r+"] == pytest.approx(1500.0)


def test_worst_corner_blocks_acceptance_and_is_named(tmp_path: Path) -> None:
    model_path, request_path = _write_inputs(
        tmp_path,
        tolerances=[{"parameter": "r", "percent": 20}],
    )
    expectations_path = tmp_path / "expectations.json"
    # Final voltages: nominal ~9.99955, r-20% ~9.99996, r+20% ~9.99760.
    # A minimum of 9.999 passes nominal and r-, fails the r+ corner.
    expectations_path.write_text(
        json.dumps(
            {
                "schema": "pear-sim/expectations/v0.1",
                "criteria": [
                    {
                        "id": "final-voltage-floor",
                        "signal": "v(out)",
                        "metric": "final",
                        "unit": "V",
                        "minimum": 9.999,
                        "blocking": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifacts = run_tolerance_corners(
        model_path,
        request_path,
        tmp_path / "out",
        expectations_file=expectations_path,
    )
    summary = json.loads(artifacts.summary.read_text())
    assert summary["acceptance_status"] == "fail"
    assert summary["binding_corner"] == "corner-r+"
    assert summary["overall_status"] == "fail"
    statuses = {row["corner_id"]: row["acceptance_status"] for row in summary["corners"]}
    assert statuses["nominal"] == "pass"
    assert statuses["corner-r-"] == "pass"
    assert statuses["corner-r+"] != "pass"


def test_request_validation_fails_closed(tmp_path: Path) -> None:
    model_path, request_path = _write_inputs(
        tmp_path, tolerances=[{"parameter": "ghost", "percent": 10}]
    )
    with pytest.raises(ParameterSweepError, match="unknown model parameter"):
        run_tolerance_corners(model_path, request_path, tmp_path / "out")

    (tmp_path / "b").mkdir()
    _, bad_request = _write_inputs(
        tmp_path / "b",
        tolerances=[{"parameter": "r", "percent": 150}],
    )
    with pytest.raises(ParameterSweepError, match="strictly between 0 and 100"):
        run_tolerance_corners(tmp_path / "b" / "model.yaml", bad_request, tmp_path / "out2")
