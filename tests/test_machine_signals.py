"""Rotor angle, speed and torque as recorded output signals.

A field-oriented drive controller is built from the rotor angle. Until it is
a signal the model can read, the controller cannot exist as anything but
Python. These tests pin what the signal MEANS, not merely that it appears:
its origin, its rate, its unit, the boundary convention it is sampled under,
and — most of all — that the quantities the pipeline cannot honestly compute
for it are ABSENT rather than filled with plausible numbers.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]
DRIVE = ROOT / "examples" / "pmsm_inverter_drive.yaml"


@pytest.fixture(scope="module")
def drive_result():
    return HybridSimulator(load_model(DRIVE)).simulate()


def _model_dict() -> dict:
    import yaml

    return yaml.safe_load(DRIVE.read_text(encoding="utf-8"))


def _load_variant(tmp_path: Path, data: dict):
    """Write a modified model out and load it through the normal path."""

    import yaml

    target = tmp_path / "variant.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_model(target)


def test_angle_starts_at_the_declared_initial_angle(drive_result) -> None:
    model = drive_result.model
    spec = model.machines[0]
    angle = drive_result.signals["angle_elec(motor)"]
    assert angle[0] == pytest.approx(spec.initial_angle_rad, abs=1e-12)


def test_electrical_angle_advances_at_pole_pairs_times_speed(drive_result) -> None:
    """The angle increment over one output step is p * omega * dt.

    Not exactly, and the reason is worth stating: several integration
    segments can close inside one output step, each nudging the held speed,
    so the recorded increment is a piecewise sum rather than one product.
    The MEASURED residual is 4.9e-7 rad against a typical increment of
    6.3e-3 rad — four orders of magnitude down. The gate is set at 1e-3
    relative, which is loose enough to survive segment reshuffling and tight
    enough that the bug this test was written for (the increment being added
    twice, residual == the whole increment) fails it by three orders.
    """

    time = drive_result.time
    angle = drive_result.signals["angle_elec(motor)"]
    speed = drive_result.signals["speed_mech(motor)"]
    pole_pairs = drive_result.model.machines[0].pole_pairs
    increments = np.diff(angle)
    steps = np.diff(time)
    predicted = pole_pairs * speed[:-1] * steps
    residual = np.abs(increments - predicted)
    assert np.max(residual) < 1e-3 * float(np.median(np.abs(increments)))
    # Monotone: the rotor turns one way in this run, so a sign slip shows up.
    assert np.all(increments > 0.0)


def test_mechanical_and_electrical_quantities_differ_by_pole_pairs(drive_result) -> None:
    model = load_model(DRIVE)
    model = replace(
        model,
        outputs=model.outputs + ("angle_mech(motor)", "speed_elec(motor)"),
    )
    result = HybridSimulator(model).simulate()
    pole_pairs = model.machines[0].pole_pairs
    assert np.allclose(
        result.signals["angle_mech(motor)"] * pole_pairs,
        result.signals["angle_elec(motor)"],
        rtol=0.0,
        atol=1e-12,
    )
    assert np.allclose(
        result.signals["speed_elec(motor)"],
        result.signals["speed_mech(motor)"] * pole_pairs,
        rtol=0.0,
        atol=1e-12,
    )


def test_units_are_declared_not_guessed(drive_result) -> None:
    metrics = drive_result.metrics()
    assert metrics["torque(motor)"]["unit"] == "N*m"
    assert metrics["speed_mech(motor)"]["unit"] == "rad/s"
    assert metrics["angle_elec(motor)"]["unit"] == "rad"
    # The electrical outputs keep theirs.
    assert metrics["v(ta)"]["unit"] == "V"
    assert metrics["i(motor.a.l)"]["unit"] == "A"


def test_continuous_time_metrics_are_absent_not_zeroed(drive_result) -> None:
    payload = drive_result.metrics()["torque(motor)"]
    assert payload["time_integrated_status"] == "not_computed"
    assert "reason" not in payload or payload["time_integrated_reason"]
    # The point of the exercise: indexing a quantity that was not computed
    # must RAISE, not return a plausible zero.
    for key in (
        "mean",
        "rms",
        "integral",
        "square_integral",
        "mean_last_half",
        "rms_last_half",
        "full_window_duration_s",
    ):
        with pytest.raises(KeyError):
            payload[key]
    # Output-grid statistics are real measurements of the recorded grid and
    # stay, under their explicit sample_* names.
    assert payload["sample_count"] > 0
    assert math.isfinite(payload["sample_mean"])


def test_torque_agrees_with_the_independent_segment_quadrature(drive_result) -> None:
    # Two independent paths: point samples through the output pipeline, and
    # Gauss quadrature per segment through the machine trace. They integrate
    # the same physical quantity and must land on the same number.
    trace = drive_result.machine_trace
    time = drive_result.time
    torque = drive_result.signals["torque(motor)"]
    quadrature = sum(
        row["mean_torque_nm"] * (row["segment_end_s"] - row["segment_start_s"])
        for row in trace
    )
    trapezoid = float(np.trapezoid(torque, time))
    assert quadrature == pytest.approx(trapezoid, rel=5e-3)


def test_sampled_speed_is_the_speed_held_over_its_segment(drive_result) -> None:
    """A sample carries the speed HELD over the segment it falls in.

    This is the property the whole machine model rests on ("the mechanical
    speed is HELD within one integration segment while the rotor angle still
    advances exactly"), so the recorded signal has to agree with it. It also
    pins the ordering bug directly: `end_segment` advances the mechanical
    state BEFORE the solver records the samples inside the segment just
    integrated, so a sampler reading the live state reports the NEXT
    segment's speed instead of this one's.
    """

    import bisect

    trace = sorted(drive_result.machine_trace, key=lambda row: row["segment_start_s"])
    starts = [row["segment_start_s"] for row in trace]
    time = drive_result.time
    speed = drive_result.signals["speed_mech(motor)"]
    inside = 0
    mismatched = 0
    for index, stamp in enumerate(time):
        moment = float(stamp)
        position = bisect.bisect_right(starts, moment) - 1
        if position < 0:
            continue
        row = trace[position]
        if not row["segment_start_s"] <= moment <= row["segment_end_s"]:
            continue
        inside += 1
        if abs(speed[index] - row["speed_rad_s"]) > 1e-12:
            # A sample sitting exactly ON a boundary belongs to both
            # segments; carrying the closing segment's speed is correct.
            if abs(moment - row["segment_start_s"]) > 0.0:
                mismatched += 1
    assert inside > 1000, "the held-speed property was not actually exercised"
    assert mismatched == 0


def test_unknown_machine_is_named_not_resolved_against_the_circuit(tmp_path: Path) -> None:
    data = _model_dict()
    data["outputs"] = data["outputs"] + ["torque(nosuch)"]
    with pytest.raises(ModelError, match="not a declared machine"):
        _load_variant(tmp_path, data)


def test_machine_signal_without_any_machine_fails_closed(tmp_path: Path) -> None:
    data = _model_dict()
    # A machine signal whose target happens to name an expanded branch id
    # must not resolve: the branch is real, the reading would be nonsense.
    data["outputs"] = data["outputs"] + ["speed_mech(motor.a.l)"]
    with pytest.raises(ModelError, match="not a declared machine"):
        _load_variant(tmp_path, data)


def test_harmonics_refuse_machine_signals(tmp_path: Path) -> None:
    data = _model_dict()
    data["outputs"] = data["outputs"] + ["torque(motor)"]
    data.setdefault("analysis", {})["harmonics"] = {
        "fundamental_frequency": "100 Hz",
        "period_count": 1,
        "max_harmonic": 3,
        "signals": [{"signal": "torque(motor)", "semantic": "phase_current_a"}],
    }
    # Rejected by the schema before the loader is reached, which is the
    # right layer: Fourier analysis evaluates the expression against the
    # compiled circuit, and a machine signal has no representation there.
    with pytest.raises(ModelError, match="does not match"):
        _load_variant(tmp_path, data)


def test_electrical_outputs_are_unaffected_by_machine_outputs() -> None:
    # Backward compatibility, measured: adding machine signals to a model
    # must not disturb one number of the electrical result.
    base = load_model(DRIVE)
    stripped = replace(
        base,
        outputs=tuple(
            name
            for name in base.outputs
            if name.startswith(("v(", "i("))
        ),
    )
    with_machine = HybridSimulator(base).simulate()
    without_machine = HybridSimulator(stripped).simulate()
    for name in stripped.outputs:
        assert np.array_equal(
            with_machine.signals[name], without_machine.signals[name]
        ), f"{name} changed when machine signals were added"
    assert json.dumps(without_machine.metrics()["v(ta)"], sort_keys=True) == json.dumps(
        with_machine.metrics()["v(ta)"], sort_keys=True
    )
