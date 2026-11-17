# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""state_discontinuity: a mode change that would make a state jump is named.

Closing a switch across capacitors at different voltages, or opening the
last path of an inductor current, would move charge or flux in zero time.
The ideal kernel refuses; these tests pin WHAT it says: the loop or cut, the
jump, the instant and the event, each against a closed form computed here
by hand (exponential RC/RL decay, ramp of an inductor on a DC source) rather
than read back from the kernel.

The check must also stay silent where nothing jumps: the same circuits with
consistent states run to completion, and a consistent capacitor loop keeps
its plain topology error (it is algebraic, not a jump).

One case used to be silent and WRONG: a switch opening between two
inductors carrying different currents. The floating-star reduction keeps
the KCL residual constant instead of restoring it, so the run went on with
a phantom current into the node; the only trace was an energy-balance miss
blamed on stiffness. test_the_star_check_is_what_catches_it shows that.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import yaml

from pear_sim.api.cli import main
from pear_sim.core import solver
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import StateDiscontinuityError, TopologyError
from pear_sim.spec.model import load_model

CLOSE_S = 0.5e-3  # the pwm edge at which the switch changes state
BUDGET = 1e-9


def _run(tmp_path: Path, components: list[dict], *, phase: str, name: str = "case",
         outputs: tuple[str, ...] = ("v(a)",), stop: str = "2 ms"):
    document = {
        "schema": "pear-sim/v0.1",
        "name": name,
        "circuit": {"ground": "gnd", "components": components},
        "controls": {"pwm": {"type": "pwm", "frequency": "1 kHz", "duty": 0.5, "phase": phase}},
        "analysis": {
            "transient": {"start": "0 s", "stop": stop, "output_step": "10 us",
                          "solver": "dopri5", "rtol": 1e-10, "atol": 1e-12},
        },
        "outputs": list(outputs),
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path, HybridSimulator(load_model(path)).simulate


def _refusal(tmp_path: Path, components: list[dict], *, phase: str) -> StateDiscontinuityError:
    _path, simulate = _run(tmp_path, components, phase=phase)
    with pytest.raises(StateDiscontinuityError) as caught:
        simulate()
    error = caught.value
    assert error.code == "model.state_discontinuity"
    assert isinstance(error, TopologyError)  # Python callers catching the parent keep working
    assert error.time == pytest.approx(CLOSE_S, abs=1e-15)
    return error


BOTH = ("i(l1)", "i(l2)")
CLOSE = "0.5 ms"  # switch initially off, closes at 0.5 ms
OPEN = "0 s"  # switch initially on, opens at 0.5 ms


def test_two_charged_capacitors_closed_together(tmp_path: Path) -> None:
    error = _refusal(
        tmp_path,
        [
            {"id": "c1", "type": "capacitor", "p": "a", "n": "gnd", "value": "1 uF",
             "initial_voltage": "10 V"},
            {"id": "r1", "type": "resistor", "p": "a", "n": "gnd", "value": "1 kohm"},
            {"id": "s1", "type": "ideal_switch", "p": "a", "n": "b", "control": "pwm",
             "initial": "off"},
            {"id": "c2", "type": "capacitor", "p": "b", "n": "gnd", "value": "3 uF",
             "initial_voltage": "2 V"},
            {"id": "r2", "type": "resistor", "p": "b", "n": "gnd", "value": "1 kohm"},
        ],
        phase=CLOSE,
    )
    (finding,) = error.findings
    expected = 10.0 * math.exp(-CLOSE_S / 1e-3) - 2.0 * math.exp(-CLOSE_S / 3e-3)
    assert finding["kind"] == "capacitor_loop"
    assert finding["members"] == ["c1", "c2", "s1"]
    assert finding["closed_by"] == "s1"
    assert abs(finding["jump"] - expected) < BUDGET * expected
    assert error.event == "gate 'pwm' off->on"
    assert "4.37234 V" in str(error)


def test_a_capacitor_closed_onto_a_sine_source(tmp_path: Path) -> None:
    error = _refusal(
        tmp_path,
        [
            {"id": "v1", "type": "voltage_source", "p": "in", "n": "gnd",
             "waveform": {"type": "sine", "amplitude": "10 V", "frequency": "200 Hz",
                          "phase_deg": 30}},
            {"id": "s1", "type": "ideal_switch", "p": "in", "n": "a", "control": "pwm",
             "initial": "off"},
            {"id": "c1", "type": "capacitor", "p": "a", "n": "gnd", "value": "1 uF",
             "initial_voltage": "1 V"},
            {"id": "r1", "type": "resistor", "p": "a", "n": "gnd", "value": "1 kohm"},
        ],
        phase=CLOSE,
    )
    (finding,) = error.findings
    source = 10.0 * math.sin(2.0 * math.pi * 200.0 * CLOSE_S + math.radians(30.0))
    expected = abs(source - math.exp(-CLOSE_S / 1e-3))
    assert finding["members"] == ["c1", "s1", "v1"]
    assert abs(finding["jump"] - expected) < BUDGET * 10.0


def test_one_inductor_cut_by_an_opening_switch(tmp_path: Path) -> None:
    error = _refusal(
        tmp_path,
        [
            {"id": "v1", "type": "voltage_source", "p": "in", "n": "gnd", "value": "10 V"},
            {"id": "l1", "type": "inductor", "p": "in", "n": "a", "value": "1 mH",
             "initial_current": "1 A"},
            {"id": "s1", "type": "ideal_switch", "p": "a", "n": "gnd", "control": "pwm",
             "initial": "on"},
        ],
        phase=OPEN,
    )
    (finding,) = error.findings
    assert finding == {"kind": "inductor_cut", "members": ["l1"], "node": "a",
                       "jump": pytest.approx(1.0 + 10.0 / 1e-3 * CLOSE_S, rel=BUDGET),
                       "unit": "A", "open_switches": ["s1"]}
    assert error.event == "gate 'pwm' on->off"
    assert "antiparallel diode across s1" in str(error)


def _two_inductors(l2_initial: str) -> list[dict]:
    # l1 ramps on the source while s1 shorts node a; l2 (a -> gnd) is shorted
    # too and keeps its current. When s1 opens, l1 and l2 end up in series.
    return [
        {"id": "v1", "type": "voltage_source", "p": "in", "n": "gnd", "value": "10 V"},
        {"id": "l1", "type": "inductor", "p": "in", "n": "a", "value": "1 mH",
         "initial_current": "1 A"},
        {"id": "s1", "type": "ideal_switch", "p": "a", "n": "gnd", "control": "pwm",
         "initial": "on"},
        {"id": "l2", "type": "inductor", "p": "a", "n": "gnd", "value": "3 mH",
         "initial_current": l2_initial},
    ]


def test_two_inductors_with_different_currents_are_refused(tmp_path: Path) -> None:
    error = _refusal(tmp_path, _two_inductors("0 A"), phase=OPEN)
    (finding,) = error.findings
    assert finding["kind"] == "inductor_cut"
    assert finding["members"] == ["l1", "l2"]
    assert finding["jump"] == pytest.approx(6.0, rel=BUDGET)


def test_the_star_check_is_what_catches_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the check the run 'succeeds' and breaks KCL at node a while s1
    is open: i(l1) - i(l2) stays at the 6 A it had at the cut."""

    monkeypatch.setattr(
        solver.HybridSimulator, "_assert_star_continuity", lambda *args, **kwargs: None
    )
    _path, simulate = _run(
        tmp_path, _two_inductors("0 A"), phase=OPEN, outputs=BOTH, stop="0.9 ms"
    )
    result = simulate()
    i1, i2 = result.signals["i(l1)"], result.signals["i(l2)"]
    open_phase = [a - b for t, a, b in zip(result.time, i1, i2) if t > CLOSE_S]
    assert open_phase and all(abs(gap - 6.0) < 1e-6 for gap in open_phase)
    assert result.physics_audit["full"]["balance_within_tolerance"] is False


def test_silent_when_the_inductor_currents_already_agree(tmp_path: Path) -> None:
    # i(l1) reaches 1 + 10/1mH * 0.5 ms = 6 A exactly when s1 opens; l2 holds 6 A.
    _path, simulate = _run(
        tmp_path, _two_inductors("6 A"), phase=OPEN, outputs=BOTH, stop="0.9 ms"
    )
    result = simulate()
    i1, i2 = result.signals["i(l1)"], result.signals["i(l2)"]
    open_phase = [a - b for t, a, b in zip(result.time, i1, i2) if t > CLOSE_S]
    assert open_phase and max(abs(gap) for gap in open_phase) < 1e-6


def test_a_consistent_capacitor_loop_is_not_a_jump(tmp_path: Path) -> None:
    """Equal voltages: nothing jumps, but the loop is still algebraic, which
    the ideal kernel cannot integrate — the plain topology error, not this one."""

    _path, simulate = _run(
        tmp_path,
        [
            {"id": "c1", "type": "capacitor", "p": "a", "n": "gnd", "value": "1 uF",
             "initial_voltage": "5 V"},
            {"id": "s1", "type": "ideal_switch", "p": "a", "n": "b", "control": "pwm",
             "initial": "off"},
            {"id": "c2", "type": "capacitor", "p": "b", "n": "gnd", "value": "3 uF",
             "initial_voltage": "5 V"},
        ],
        phase=CLOSE,
    )
    with pytest.raises(TopologyError) as caught:
        simulate()
    assert not isinstance(caught.value, StateDiscontinuityError)


def test_a_switch_on_resistance_removes_the_jump(tmp_path: Path) -> None:
    _path, simulate = _run(
        tmp_path,
        [
            {"id": "c1", "type": "capacitor", "p": "a", "n": "gnd", "value": "1 uF",
             "initial_voltage": "10 V"},
            {"id": "s1", "type": "ideal_switch", "p": "a", "n": "b", "control": "pwm",
             "initial": "off", "on_resistance": "1 ohm"},
            {"id": "c2", "type": "capacitor", "p": "b", "n": "gnd", "value": "3 uF",
             "initial_voltage": "2 V"},
            {"id": "rx", "type": "resistor", "p": "a", "n": "gnd", "value": "1 kohm"},
        ],
        phase=CLOSE,
    )
    simulate()


def test_the_cli_reports_the_code_and_the_findings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path, _simulate = _run(tmp_path, _two_inductors("0 A"), phase=OPEN)
    status = main(["simulate", str(path), "--out", str(tmp_path / "out"), "--json"])
    assert status == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "model.state_discontinuity"
    details = payload["details"]
    assert details["diagnostic"] == "state_discontinuity"
    assert details["time_s"] == pytest.approx(CLOSE_S)
    assert details["event"] == "gate 'pwm' on->off"
    assert details["findings"][0]["members"] == ["l1", "l2"]


def test_winding_star_drift_is_not_a_jump() -> None:
    """Gate measured on the existing examples (2026-11-17): the coupled-winding
    star of this rectifier drifts 9.5e-8 A under rtol 1e-7 — integration error,
    not a cut. A bound of 1e-9 relative refused it; the run's own atol + rtol*|i|
    does not."""

    root = Path(__file__).resolve().parents[1]
    HybridSimulator(
        load_model(root / "examples" / "rectifier_6pulse_transformer.yaml"), evidence_mode="lean"
    ).simulate()
