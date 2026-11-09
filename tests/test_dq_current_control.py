# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Closed-loop dq current control with in-the-loop SRF-PLL (grid tie).

The suite variant runs a reduced 4 kHz carrier with gains chosen from an
averaged discrete-loop design study, so the end-to-end grid-following case
stays affordable; the 10 kHz flagship lives in
examples/grid_following_converter.yaml (measured: PLL 50.001 Hz,
i_d 5.0007 A, i_q -0.0023 A, P 2548 W, Q 21 var over 60 ms).
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "grid_following_converter.yaml"


def _fast_variant(tmp_path: Path, **overrides) -> Path:
    document = yaml.safe_load(EXAMPLE.read_text())
    document["name"] = "grid_following_fast"
    document["parameters"]["fsw"] = "4 kHz"
    control = document["controls"]["ctrl"]
    control["current_kp"] = 10.0
    control["current_ki"] = 2000
    control["pll_bandwidth"] = "15 Hz"
    control["dead_time"] = "2 us"
    document["analysis"]["transient"]["stop"] = "40 ms"
    document["analysis"]["transient"]["output_step"] = "100 us"
    document["analysis"]["transient"]["rtol"] = 1e-7
    document["analysis"]["transient"]["atol"] = 1e-9
    for key, value in overrides.items():
        control[key] = value
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loader_expands_dq_block_to_six_gates(tmp_path: Path) -> None:
    model = load_model(_fast_variant(tmp_path))
    gate_ids = sorted(model.sampled_controls)
    assert gate_ids == [
        f"ctrl.{phase}.{role}" for phase in "abc" for role in ("high", "low")
    ]
    high = model.sampled_controls["ctrl.a.high"]
    assert high.kind == "sampled_dq_current_pwm"
    assert high.partner_id == "ctrl.a.low"
    assert high.injection == "min_max"


def test_loader_requires_senses_in_outputs(tmp_path: Path) -> None:
    path = _fast_variant(tmp_path, sense_currents=["i(lfa)", "i(lfb)", "i(rearth)"])
    with pytest.raises(ModelError, match="sense"):
        load_model(path)


def test_loader_rejects_dead_time_exceeding_margins(tmp_path: Path) -> None:
    path = _fast_variant(tmp_path, dead_time="20 us", duty_min="2 %")
    with pytest.raises(ModelError, match="dead_time"):
        load_model(path)


def test_loader_rejects_duplicate_sense_expressions(tmp_path: Path) -> None:
    path = _fast_variant(tmp_path, sense_voltages=["v(pa)", "v(pa)", "v(pc)"])
    with pytest.raises(ModelError, match="distinct"):
        load_model(path)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid_result(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("dq")
    model = load_model(_fast_variant(tmp_path))
    return HybridSimulator(model).simulate()


def test_pll_locks_to_the_simulated_grid(grid_result) -> None:
    trace = [e for e in grid_result.control_trace if e["control_id"] == "ctrl"]
    assert len(trace) == 160  # 40 ms * 4 kHz
    tail = trace[len(trace) // 2 :]
    frequency = float(np.mean([e["frequency_hz"] for e in tail]))
    assert frequency == pytest.approx(50.0, abs=0.1)


def test_dq_currents_track_their_references(grid_result) -> None:
    trace = [e for e in grid_result.control_trace if e["control_id"] == "ctrl"]
    tail = trace[len(trace) // 2 :]
    i_d = float(np.mean([e["i_d"] for e in tail]))
    i_q = float(np.mean([e["i_q"] for e in tail]))
    assert i_d == pytest.approx(5.0, abs=0.25)
    assert i_q == pytest.approx(0.0, abs=0.25)
    assert grid_result.physics_audit["full"]["balance_status"] == (
        "within_numerical_tolerance"
    )


def test_power_at_pcc_matches_the_current_reference(grid_result) -> None:
    from pear_sim.analysis.reference_frames import instantaneous_power

    t = grid_result.time
    mask = t > 0.025
    power = instantaneous_power(
        grid_result.signals["v(pa)"][mask],
        grid_result.signals["v(pb)"][mask],
        grid_result.signals["v(pc)"][mask],
        grid_result.signals["i(lfa)"][mask],
        grid_result.signals["i(lfb)"][mask],
        grid_result.signals["i(lfc)"][mask],
    )
    active = float(np.mean(power["p_w"]))
    reactive = float(np.mean(power["q_var"]))
    # P ~ 1.5 * |V_pcc| * id with |V_pcc| between the 325 V EMF and the
    # slightly boosted PCC level; Q near zero for iq_ref = 0.
    assert 2200.0 < active < 2900.0
    assert abs(reactive) < 0.1 * active


def test_gate_events_alternate_without_overlap(grid_result) -> None:
    states = {f"ctrl.{p}.{r}": (r == "low") for p in "abc" for r in ("high", "low")}
    for event in grid_result.events:
        if event.kind != "scheduled_gate" or not event.target.startswith("ctrl."):
            continue
        assert states[event.target] == event.previous, event
        states[event.target] = event.current
        block, phase, _role = event.target.rsplit(".", 2)
        high_on = states[f"{block}.{phase}.high"]
        low_on = states[f"{block}.{phase}.low"]
        assert not (high_on and low_on), f"interlock violated at {event.time}"


def test_dq_bundle_writes_with_gate_drive_observation(tmp_path: Path) -> None:
    import json

    from pear_sim.run.io import write_result

    document = yaml.safe_load(EXAMPLE.read_text())
    document["name"] = "grid_following_bundle"
    document["parameters"]["fsw"] = "4 kHz"
    control = document["controls"]["ctrl"]
    control["current_kp"] = 10.0
    control["current_ki"] = 2000
    control["dead_time"] = "2 us"
    document["analysis"]["transient"]["stop"] = "5 ms"
    document["analysis"]["transient"]["output_step"] = "100 us"
    document["analysis"]["transient"]["rtol"] = 1e-7
    document["analysis"]["transient"]["atol"] = 1e-9
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = HybridSimulator(load_model(path)).simulate()
    destination = write_result(result, tmp_path / "bundle")
    gate_drive = json.loads((destination / "gate-drive.json").read_text())
    dq_outputs = [
        output
        for output in gate_drive["gate_outputs"]
        if output["control_type"] == "sampled_dq_current_pwm"
    ]
    assert len(dq_outputs) == 6
    assert all(o["command_source"] == "closed_loop_sampled_dq" for o in dq_outputs)
    assert (destination / "control-trace.jsonl").exists()
    resolved = json.loads((destination / "model.resolved.json").read_text())
    dq_controls = [
        control
        for control in resolved["physics"]["controls"]
        if control["type"] == "sampled_dq_current_pwm"
    ]
    assert len(dq_controls) == 1
    assert dq_controls[0]["pll"]["nominal_frequency_hz"] == pytest.approx(50.0)
