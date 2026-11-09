# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from dataclasses import replace
import math
from pathlib import Path

import numpy as np
import pytest

from pear_sim.circuit import Circuit
from pear_sim.errors import TopologyError
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator


ROOT = Path(__file__).parents[1]


def _dcm_buck_model(tmp_path: Path):
    path = tmp_path / "buck_dcm.yaml"
    path.write_text(
        """
schema: pear-sim/v0.1
name: buck_dcm_state_reduction
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 12 V}
    - {id: q1, type: ideal_switch, p: in, n: sw, control: pwm, initial: on}
    - {id: d1, type: diode, p: gnd, n: sw, initial: off}
    - {id: l1, type: inductor, p: sw, n: out, value: 100 uH, initial_current: 0 A}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, initial_voltage: 5 V}
    - {id: load, type: resistor, p: out, n: gnd, value: 100 ohm}
controls:
  pwm: {type: pwm, frequency: 10 kHz, duty: 20 %, phase: 0 s}
analysis:
  transient:
    start: 0 s
    stop: 300 us
    output_step: 0.5 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-12
    max_step: 2 us
outputs: [v(out), i(l1), i(d1), v(q1), v(d1)]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return load_model(path)


def test_zero_current_off_off_mode_reduces_and_reactivates_inductor(tmp_path: Path) -> None:
    model = _dcm_buck_model(tmp_path)
    circuit = Circuit(model)
    state = circuit.initial_state.copy()
    inductor_index = circuit.state_index["l1"]

    diodes, frozen = circuit.stable_mode(
        {"pwm": False},
        {"d1": False},
        state,
        1e-10,
    )
    assert diodes == {"d1": False}
    assert frozen.constrained_inductors == ("l1",)
    assert frozen.constrained_state_indices == (inductor_index,)
    assert inductor_index not in frozen.active_state_indices
    assert frozen.mode_signature == (
        "gates[pwm=off];diodes[d1=off];constrained_inductors[l1]"
    )
    assert frozen.a.shape == (len(state), len(state))
    assert np.all(frozen.a[inductor_index] == 0.0)
    assert frozen.c[inductor_index] == 0.0
    assert frozen.derivative(0.0, state)[inductor_index] == 0.0
    assert frozen.component_voltage(circuit.component_by_id["l1"], state) == 0.0
    assert frozen.constrained_branch_current("l1", state) == 0.0
    assert frozen.node_voltage("sw", state) == frozen.node_voltage("out", state)

    on_diodes, active = circuit.stable_mode(
        {"pwm": True},
        diodes,
        state,
        1e-10,
    )
    assert on_diodes == {"d1": False}
    assert active.constrained_inductors == ()
    assert active.active_state_indices == tuple(range(len(state)))
    assert active.derivative(0.0, state)[inductor_index] > 0.0
    assert state[inductor_index] == 0.0


def test_nonzero_inductor_current_rejects_off_off_constraint(tmp_path: Path) -> None:
    model = _dcm_buck_model(tmp_path)
    circuit = Circuit(model)
    state = circuit.initial_state.copy()
    state[circuit.state_index["l1"]] = 0.25

    diodes, mode = circuit.stable_mode(
        {"pwm": False},
        {"d1": False},
        state,
        1e-10,
    )
    assert diodes == {"d1": True}
    assert mode.constrained_inductors == ()
    assert mode.component_current(circuit.component_by_id["d1"], state) > 0.0


def test_reduction_rejects_zero_volt_branch_that_would_carry_current(tmp_path: Path) -> None:
    path = tmp_path / "invalid_constraint_current.yaml"
    path.write_text(
        """
schema: pear-sim/v0.1
name: invalid_constraint_current
circuit:
  ground: gnd
  components:
    - {id: forcing, type: current_source, p: gnd, n: sw, value: 1 A}
    - {id: l1, type: inductor, p: sw, n: out, value: 1 H, initial_current: 0 A}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 1 F, initial_voltage: 0 V}
    - {id: r1, type: resistor, p: out, n: gnd, value: 1 ohm}
analysis:
  transient: {start: 0 s, stop: 1 ms, output_step: 0.1 ms}
outputs: [i(l1)]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    circuit = Circuit(load_model(path))
    state = circuit.initial_state.copy()
    algebraically_closed = circuit.compile({}, {}, ("l1",))

    assert abs(algebraically_closed.constrained_branch_current("l1", state)) == 1.0
    with pytest.raises(TopologyError, match="No stable ideal-switch configuration"):
        circuit.stable_mode({}, {}, state, 1e-10)


def test_reduction_rejects_hidden_branch_current_that_drifts_from_zero(
    tmp_path: Path,
) -> None:
    path = tmp_path / "three_inductor_hidden_branch_drift.yaml"
    path.write_text(
        """
schema: pear-sim/v0.1
name: three_inductor_hidden_branch_drift
circuit:
  ground: gnd
  components:
    - {id: va, type: voltage_source, p: a, n: gnd, value: 1 V}
    - {id: vb, type: voltage_source, p: b, n: gnd, value: 2 V}
    - {id: l0, type: inductor, p: x, n: gnd, value: 1 H, initial_current: 0 A}
    - {id: l1, type: inductor, p: x, n: a, value: 1 H, initial_current: 1 A}
    - {id: l2, type: inductor, p: x, n: b, value: 1 H, initial_current: -1 A}
analysis:
  transient: {start: 0 s, stop: 1 ms, output_step: 0.1 ms}
outputs: [i(l0)]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    circuit = Circuit(load_model(path))
    state = circuit.initial_state.copy()

    # The original protection this test pins down: the l0-frozen mode, whose
    # hidden branch current would drift from zero, must never be usable. With
    # multi-inductor freezing the rejection happens already at compile time — freezing l0
    # leaves a two-member live star at x whose reduced row no longer
    # determines the frozen branch current, so the mode is singular.
    with pytest.raises(TopologyError, match="Singular"):
        circuit.compile({}, {}, ("l0",))

    # Since the floating-star reduction, this three-inductor junction is
    # genuinely solvable: stable_mode picks the pure star-reduced mode (KCL
    # held as an exact invariant), with no inductor frozen.
    _, mode = circuit.stable_mode({}, {}, state, 1e-10)
    assert mode.constrained_inductors == ()
    assert [node for node, _ in mode.star_constraints] == ["x"]
    # Freezing a pair is structurally inconsistent here (both 0 V branches
    # from x fight the two different source potentials) — still fail closed.
    with pytest.raises(TopologyError):
        circuit.compile({}, {}, ("l0", "l1"))
    # The multi-constraint bound itself is four per mode.
    with pytest.raises(TopologyError, match="at most four constrained"):
        circuit.compile({}, {}, ("l0", "l1", "l2", "ghost1", "ghost2"))


def test_loose_ode_atol_does_not_erase_current_at_gate_turnoff(
    tmp_path: Path,
) -> None:
    path = tmp_path / "loose_atol_gate_turnoff.yaml"
    path.write_text(
        """
schema: pear-sim/v0.1
name: loose_atol_gate_turnoff
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 7 V}
    - {id: q1, type: ideal_switch, p: in, n: sw, control: pwm, initial: on}
    - {id: d1, type: diode, p: gnd, n: sw, initial: off}
    - {id: l1, type: inductor, p: sw, n: out, value: 100 uH, initial_current: 0.05 A}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, initial_voltage: 5 V}
    - {id: load, type: resistor, p: out, n: gnd, value: 100 ohm}
controls:
  pwm: {type: pwm, frequency: 100 kHz, duty: 10 %, phase: 0 s}
analysis:
  transient:
    start: 0 s
    stop: 1.5 us
    output_step: 0.5 us
    solver: dopri5
    rtol: 1.0e-3
    atol: 1.0e-3
    max_step: 50 ns
outputs: [i(l1), i(d1), v(out)]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = HybridSimulator(load_model(path)).simulate()
    turnoff_index = int(np.flatnonzero(np.isclose(result.time, 1e-6, atol=1e-15))[0])
    freewheel = next(
        interval
        for interval in result.mode_intervals
        if interval["start_s"] == pytest.approx(1e-6, abs=1e-15)
        and interval["gates"] == {"pwm": False}
    )
    on_interval = result.mode_intervals[result.mode_intervals.index(freewheel) - 1]

    assert result.signals["i(l1)"][0] == 0.05
    assert result.signals["i(l1)"][turnoff_index] > 0.06
    assert result.signals["i(d1)"][turnoff_index] == pytest.approx(
        result.signals["i(l1)"][turnoff_index]
    )
    assert freewheel["diodes"] == {"d1": True}
    assert freewheel["constrained_inductors"] == []
    assert freewheel["state_start"]["l1"] == on_interval["state_end"]["l1"]
    assert not any(
        interval["constrained_inductors"] for interval in result.mode_intervals
    )


def test_dcm_natural_turnoff_freezes_zero_current_without_chattering(tmp_path: Path) -> None:
    model = _dcm_buck_model(tmp_path)
    result = HybridSimulator(model).simulate()
    inductor = result.signals["i(l1)"]
    diode = result.signals["i(d1)"]
    natural_turnoffs = [
        event
        for event in result.events
        if event.kind == "natural_commutation"
        and event.target == "d1"
        and event.previous is True
        and event.current is False
    ]

    assert len(natural_turnoffs) == 3
    assert result.stats["natural_events"] == 3
    assert result.stats["reduced_topologies"] == 1
    assert result.stats["state_constraint_events"] == 3
    assert result.stats["constrained_state_projections"] <= 3
    assert float(np.min(inductor)) >= -1e-11
    assert float(np.min(diode)) >= -1e-11

    for event in natural_turnoffs:
        plateau = (result.time >= event.time + 2e-6) & (
            result.time <= min(event.time + 35e-6, model.transient.stop)
        )
        assert np.any(plateau)
        assert np.max(np.abs(inductor[plateau])) == 0.0
        assert np.max(np.abs(diode[plateau])) == 0.0

    reduced_intervals = [
        interval for interval in result.mode_intervals if interval["constrained_inductors"]
    ]
    assert len(reduced_intervals) == 3
    assert all(interval["constrained_inductors"] == ["l1"] for interval in reduced_intervals)
    assert all(interval["gates"] == {"pwm": False} for interval in reduced_intervals)
    assert all(interval["diodes"] == {"d1": False} for interval in reduced_intervals)
    assert all(set(interval["state_start"]) == {"c1", "l1"} for interval in reduced_intervals)
    assert all(interval["state_start"]["l1"] == 0.0 for interval in reduced_intervals)
    assert all(interval["state_end"]["l1"] == 0.0 for interval in reduced_intervals)
    assert all(interval["end_s"] > interval["start_s"] for interval in result.mode_intervals)
    assert result.stats["mode_intervals"] == len(result.mode_intervals)


def test_canonical_dcm_fixture_matches_event_and_steady_state_fingerprints() -> None:
    result = HybridSimulator(load_model(ROOT / "examples" / "buck_dcm.yaml")).simulate()
    turnoffs = [
        event
        for event in result.events
        if event.kind == "natural_commutation" and event.target == "d1"
    ]

    assert len(turnoffs) == 200
    assert turnoffs[0].time == pytest.approx(
        20.762398257774295e-6,
        rel=0.0,
        abs=5e-13,
    )
    assert result.metrics()["v(out)"]["mean_last_half"] == pytest.approx(
        92.65159667014042,
        rel=0.0,
        abs=5e-8,
    )
    assert result.stats["state_constraint_events"] == 200
    assert result.stats["constrained_state_projections"] == 200
    assert float(np.min(result.signals["i(l1)"])) == 0.0
    assert float(np.min(result.signals["i(d1)"])) == 0.0


def test_natural_event_just_before_scheduled_edge_has_no_time_gap() -> None:
    base = load_model(
        ROOT / "examples" / "buck_dcm.yaml",
        {"load": "6.089232079759 ohm"},
    )
    model = replace(
        base,
        transient=replace(base.transient, stop=400e-6),
    )
    simulator = HybridSimulator(model)
    simulator.time_tolerance = 1e-14

    result = simulator.simulate()
    natural_turnoffs = [
        event
        for event in result.events
        if event.kind == "natural_commutation"
        and event.target == "d1"
        and event.previous is True
        and event.current is False
    ]
    rising_edges = [
        event
        for event in result.events
        if event.kind == "scheduled_gate"
        and event.target == "pwm"
        and event.current is True
    ]

    assert any(
        0.0 < edge.time - root.time < simulator.time_tolerance
        for root in natural_turnoffs
        for edge in rising_edges
    )
    assert math.isclose(
        result.time_integrals["full"]["duration_s"],
        model.transient.stop - model.transient.start,
        rel_tol=0.0,
        abs_tol=1e-18,
    )
    assert all(
        abs(right["start_s"] - left["end_s"]) <= 1e-18
        for left, right in zip(
            result.mode_intervals,
            result.mode_intervals[1:],
            strict=False,
        )
    )
    assert result.stats["scheduled_events"] == 16
    assert result.time[-1] == model.transient.stop


def test_last_half_dense_state_extrema_find_unrequested_interior_minimum(
    tmp_path: Path,
) -> None:
    path = tmp_path / "coarse_grid_lc_extrema.yaml"
    path.write_text(
        """
schema: pear-sim/v0.1
name: coarse_grid_lc_extrema
circuit:
  ground: gnd
  components:
    - {id: l1, type: inductor, p: tank, n: gnd, value: 1 H, initial_current: 0 A}
    - {id: c1, type: capacitor, p: tank, n: gnd, value: 1 F, initial_voltage: 1 V}
analysis:
  transient:
    start: 0 s
    stop: 6.283185307179586 s
    output_step: 3.141592653589793 s
    solver: dopri5
    rtol: 1.0e-10
    atol: 1.0e-12
    max_step: 250 ms
outputs: [v(tank)]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = HybridSimulator(load_model(path)).simulate()
    inductor_extrema = result.state_extrema["states"]["l1"]

    assert "i(l1)" not in result.signals
    assert result.state_extrema["window"] == "last_half"
    assert "derivative_root_search_all_real_roots" in result.state_extrema["method"]
    assert len(result.time) == 3
    assert inductor_extrema["minimum"] == pytest.approx(-1.0, abs=2e-9)
    assert inductor_extrema["minimum_time_s"] == pytest.approx(
        1.5 * math.pi,
        abs=2e-8,
    )
    assert not np.any(
        np.isclose(
            result.time,
            inductor_extrema["minimum_time_s"],
            rtol=0.0,
            atol=1e-8,
        )
    )
    assert result.mode_intervals[0]["state_extrema"]["l1"][
        "minimum"
    ] == pytest.approx(-1.0, abs=2e-9)
