# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Closed-loop periodic steady state and Floquet multipliers.

The plant and both gain pairs are the ones the averaged loop-gain tests in
``test_closed_loop_control.py`` use: a 24 V -> 12 V buck at 50 kHz with a
5.45 ohm load (2.2 A, CCM), a PI written in signal blocks. kp=0.002/ki=30 is
the healthy design; kp=0.08/ki=150 produced a ~4 kHz limit cycle in the
switching kernel.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pear_sim.analysis import closed_loop
from pear_sim.analysis.closed_loop import (
    ClosedLoopSteadyStateError,
    solve_closed_loop_steady_state,
)
from pear_sim.core.solver import HybridSimulator
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]
PERIOD_S = 20e-6

BODY = """
schema: pear-sim/v0.1
name: buck_closed_loop_floquet
parameters: {{vin: 24 V, fsw: 50 kHz}}
circuit:
  ground: gnd
  components:
    - {{id: vin, type: voltage_source, p: in, n: gnd, value: "${{vin}}"}}
    - {{id: q1, type: ideal_switch, p: in, n: sw, control: ctrl, initial: on}}
    - {{id: d1, type: diode, p: gnd, n: sw, initial: off}}
    - {{id: l1, type: inductor, p: sw, n: out, value: 47 uH, initial_current: 0.92 A}}
    - {{id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, initial_voltage: 12 V}}
    - {{id: load, type: resistor, p: out, n: gnd, value: 5.45 ohm}}
signals:
  task: {{period: 20 us}}
  blocks:
    vout: {{type: probe, source: v(out), unit: V}}
    vref: {{type: constant, value: 12.0, unit: V}}
    err: {{type: sum, inputs: [vref, vout], signs: [1, -1], unit: V}}
    err_dt: {{type: gain, input: err, gain: 2.0e-5, unit: V}}
    int_prev: {{type: unit_delay, input: integral, initial: {integral0}, unit: V}}
    integral: {{type: sum, inputs: [int_prev, err_dt], unit: V}}
    kp_term: {{type: gain, input: err, gain: {kp}, unit: V}}
    ki_term: {{type: gain, input: integral, gain: {ki}, unit: V}}
    u: {{type: sum, inputs: [kp_term, ki_term], unit: V}}
    duty: {{type: saturate, input: u, min: 0.05, max: 0.95, unit: V}}
controls:
  ctrl: {{type: signal_pwm, frequency: "${{fsw}}", duty_signal: s(duty), duty_min: 5 %, duty_max: 95 %}}
analysis:
  transient: {{start: 0 s, stop: 1 ms, output_step: 2 us, solver: dopri5, rtol: 1.0e-9, atol: 1.0e-11}}
outputs: [v(out), i(l1)]
"""

HEALTHY = (0.002, 30.0)
HOT = (0.08, 150.0)


def _model(tmp_path: Path, gains: tuple[float, float], *, integral0: float | None = None):
    kp, ki = gains
    # Near the orbit: the integrator holds duty / ki at zero error.
    start = 0.5 / ki if integral0 is None else integral0
    path = tmp_path / f"loop_{kp}_{ki}.yaml"
    path.write_text(BODY.format(kp=kp, ki=ki, integral0=start), encoding="utf-8")
    return load_model(path)


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("floquet")
    return {
        name: (model, solve_closed_loop_steady_state(model))
        for name, model in (
            ("healthy", _model(tmp_path, HEALTHY)),
            ("hot", _model(tmp_path, HOT)),
        )
    }


def test_orbit_regulates_and_is_differentiable(solved) -> None:
    for name in ("healthy", "hot"):
        _model_, result = solved[name]
        assert result.converged, name
        state = dict(zip(result.state_names, result.state, strict=True))
        # Zero average error: the integrator settles the output at 12 V.
        assert state["c1"] == pytest.approx(12.0, abs=0.05)
        assert all(item.held for item in result.assumptions), result.assumptions
        assert len(result.multipliers) == 3  # l1, c1, one register


def test_floquet_verdict_discriminates_the_two_designs(solved) -> None:
    """The gate must stay quiet on the healthy design and ring on the hot one,
    and the hot design must oscillate where the kernel's limit cycle did."""

    healthy = solved["healthy"][1]
    hot = solved["hot"][1]
    assert healthy.stability == "stable"
    assert healthy.max_multiplier_magnitude < 1.0
    assert hot.stability == "unstable"
    critical = hot.multipliers[0]
    assert critical.magnitude > 1.0
    assert critical.frequency_hz == pytest.approx(4_000.0, rel=0.15)


def test_floquet_agrees_with_the_discrete_aware_averaged_margins(solved) -> None:
    """A different method (state-space averaging + ZOH, a declared
    compensator) must agree on WHICH design is unstable and roughly where."""

    from pear_sim.analysis.small_signal import (
        SamplingModel,
        averaged_small_signal,
        loop_gain,
    )
    from pear_sim.analysis.steady_state import solve_periodic_steady_state
    from tests.test_closed_loop_control import OPEN_LOOP_PLANT_BODY

    path = Path(solved["healthy"][0].source_path).with_name("open_loop.yaml")
    path.write_text(OPEN_LOOP_PLANT_BODY, encoding="utf-8")
    plant = load_model(path)
    signal = averaged_small_signal(plant, solve_periodic_steady_state(plant), "v(out)")
    sampling = SamplingModel(update_frequency_hz=signal.switching_frequency_hz)
    for name, (kp, ki) in (("healthy", HEALTHY), ("hot", HOT)):
        margins = loop_gain(
            signal,
            compensator_numerator=(kp, ki),
            compensator_denominator=(1.0, 0.0),
            sensor_gain=1.0,
            f_min_hz=5.0,
            f_max_hz=5_000.0,
            points=600,
            sampling=sampling,
        )
        averaged_unstable = margins.phase_margin_deg < 0.0
        assert averaged_unstable == (solved[name][1].stability == "unstable"), name
    hot_crossover = loop_gain(
        signal,
        compensator_numerator=HOT,
        compensator_denominator=(1.0, 0.0),
        sensor_gain=1.0,
        f_min_hz=5.0,
        f_max_hz=5_000.0,
        points=600,
        sampling=sampling,
    ).gain_crossover_hz
    assert solved["hot"][1].multipliers[0].frequency_hz == pytest.approx(
        hot_crossover, rel=0.15
    )


def _trajectory_multipliers(model, result, cycles: int) -> np.ndarray:
    """Identify the monodromy from ONE long run started off the orbit.

    Independent of the probe arithmetic: no finite differences, no per-cycle
    restarts — the controller memory is carried by the kernel tick to tick,
    and the matrix is the least-squares fit of d_{k+1} = M d_k over the run.
    """

    plan = closed_loop._plan(model)
    orbit = np.asarray(result.state)
    rng = np.random.default_rng(1)
    offset = 1e-4 * np.maximum(np.abs(orbit), 1e-3) * rng.standard_normal(len(orbit))
    start = closed_loop._cycle_model(plan, orbit + offset)
    run_model = replace(
        start,
        transient=replace(start.transient, stop=cycles * PERIOD_S, output_step=PERIOD_S),
        outputs=tuple(dict.fromkeys((*start.outputs, "i(l1)", "v(c1)"))),
    )
    run = HybridSimulator(run_model).simulate()
    ticks = {entry["tick_index"]: entry["values"] for entry in run.signal_trace}
    rows = []
    for cycle in range(cycles):
        index = int(np.argmin(np.abs(run.time - cycle * PERIOD_S)))
        rows.append(
            [run.signals["i(l1)"][index], run.signals["v(c1)"][index], ticks[cycle]["int_prev"]]
        )
    deviations = np.asarray(rows) - orbit
    fitted = deviations[1:].T @ np.linalg.pinv(deviations[:-1].T)
    return np.sort_complex(np.linalg.eigvals(fitted))


@pytest.mark.parametrize("name", ["healthy", "hot"])
def test_multipliers_match_a_monodromy_identified_from_one_long_run(solved, name) -> None:
    """Measured 2026-11-15: max |difference| 1.2e-8 (healthy) and 3.5e-6 (hot,
    whose perturbation grows over the 60 cycles). Carrying the register's OLD
    value instead of its input — the mapping error this guards — gives 6.9e-3.
    The 1e-4 budget sits between the two by two orders of magnitude each way."""

    model, result = solved[name]
    expected = np.sort_complex(
        np.asarray([complex(item.real, item.imag) for item in result.multipliers])
    )
    identified = _trajectory_multipliers(model, result, cycles=60)
    assert np.max(np.abs(identified - expected)) < 1e-4


def test_a_limiter_engaged_during_newton_is_reported_not_raised(tmp_path: Path) -> None:
    """Starting the integrator at 0 clamps the duty at 5 %: the map is flat in
    the controller direction. That is a non-converged RESULT naming the
    limiter, not a 'singular matrix' crash or a fabricated orbit."""

    result = solve_closed_loop_steady_state(_model(tmp_path, HEALTHY, integral0=0.0))
    assert not result.converged
    assert result.stability == "not_assessed"
    assert not result.multipliers
    assert "limiter" in result.notes[0] and "duty" in result.notes[0]


def test_result_payload_states_its_claim(solved) -> None:
    payload = solved["hot"][1].as_dict()
    assert payload["claim"] == "local stability of the computed periodic orbit only"
    assert payload["stability"] == "unstable"
    assert set(payload["state"]) == {"l1", "c1", "signal:int_prev"}


@pytest.mark.parametrize(
    ("example", "fragment"),
    [
        # Built-in PI: its integrator lives in the kernel and cannot be set.
        ("buck_closed_loop.yaml", "not signal_pwm"),
        # Anti-windup with compare/select: a threshold is a kink.
        ("buck_closed_loop_antiwindup.yaml", "non-smooth"),
        # The shipped signal-block example has a load-step waveform source.
        ("buck_closed_loop_signals.yaml", "DC sources only"),
        # Open loop belongs to the ordinary shooting.
        ("buck.yaml", "signal layer"),
    ],
)
def test_out_of_scope_models_are_refused_by_name(example: str, fragment: str) -> None:
    with pytest.raises(ClosedLoopSteadyStateError, match=fragment):
        solve_closed_loop_steady_state(load_model(ROOT / "examples" / example))
