# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Floating inductor-star nodes: reduction, invariant, and boundaries.

A node whose incident components are all plain inductors gets its degenerate
KCL row replaced by the invariant derivative, pinning the star voltage. The
KCL invariant itself is validated at t=0 and preserved exactly by RK.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import SimulationError, TopologyError
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]

STAR_BODY = """
schema: pear-sim/v0.1
name: three_wire_star

circuit:
  ground: gnd
  components:
    - {{id: ea, type: voltage_source, p: a, n: gnd, {source_a}}}
    - {{id: eb, type: voltage_source, p: b, n: gnd, {source_b}}}
    - {{id: ec, type: voltage_source, p: c, n: gnd, {source_c}}}
    - {{id: ra, type: resistor, p: a, n: ma, value: {r_a}}}
    - {{id: rb, type: resistor, p: b, n: mb, value: {r_b}}}
    - {{id: rc, type: resistor, p: c, n: mc, value: {r_c}}}
    - {{id: la, type: inductor, p: ma, n: star, value: 5 mH, initial_current: {i_a}}}
    - {{id: lb, type: inductor, p: mb, n: star, value: 5 mH, initial_current: {i_b}}}
    - {{id: lc, type: inductor, p: mc, n: star, value: 5 mH, initial_current: {i_c}}}

analysis:
  transient:
    start: 0 s
    stop: {stop}
    output_step: 20 us
    solver: dopri5
    rtol: 1.0e-10
    atol: 1.0e-12

outputs:
  - i(la)
  - i(lb)
  - i(lc)
  - v(star)
"""


def _write(tmp_path: Path, **kwargs) -> Path:
    defaults = dict(
        source_a="value: 10 V",
        source_b="value: 5 V",
        source_c="value: -2 V",
        r_a="1 ohm",
        r_b="2 ohm",
        r_c="4 ohm",
        i_a="0 A",
        i_b="0 A",
        i_c="0 A",
        stop="50 ms",
    )
    defaults.update(kwargs)
    path = tmp_path / "model.yaml"
    path.write_text(STAR_BODY.format(**defaults), encoding="utf-8")
    return path


def test_balanced_ac_star_matches_per_phase_closed_form(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        source_a="waveform: {type: sine, amplitude: 100 V, frequency: 50 Hz}",
        source_b="waveform: {type: sine, amplitude: 100 V, frequency: 50 Hz, phase_deg: -120}",
        source_c="waveform: {type: sine, amplitude: 100 V, frequency: 50 Hz, phase_deg: 120}",
        r_a="10 ohm",
        r_b="10 ohm",
        r_c="10 ohm",
        stop="100 ms",
    )
    result = HybridSimulator(load_model(path)).simulate()
    t = result.time
    ia = result.signals["i(la)"]
    total = ia + result.signals["i(lb)"] + result.signals["i(lc)"]
    assert np.max(np.abs(total)) < 1e-12
    # Balanced equal impedances: the star floats at exactly 0 V and each phase
    # is an independent series RL.
    assert np.max(np.abs(result.signals["v(star)"])) < 1e-9
    omega = 2.0 * np.pi * 50.0
    impedance = 10.0 + 1j * omega * 5e-3
    magnitude, phase = 100.0 / abs(impedance), -np.angle(impedance)
    reference = magnitude * (
        np.sin(omega * t + phase) - np.sin(phase) * np.exp(-t * 10.0 / 5e-3)
    )
    assert np.max(np.abs(ia - reference)) < 5e-8
    assert result.physics_audit["full"]["balance_status"] == "within_numerical_tolerance"


def test_unbalanced_dc_star_matches_matrix_exponential(tmp_path: Path) -> None:
    result = HybridSimulator(load_model(_write(tmp_path))).simulate()
    t = result.time
    inductance = 5e-3
    emf = np.array([10.0, 5.0, -2.0])
    resistance = np.array([1.0, 2.0, 4.0])
    # Equal inductances: v_star = mean(e - R i); di/dt = (e - R i - v_star)/L.
    a = (-np.diag(resistance) + np.tile(resistance, (3, 1)) / 3.0) / inductance
    b = (emf - np.mean(emf)) / inductance
    x_inf = -np.linalg.solve(a, b)
    reference = np.array([x_inf + expm(a * tt) @ (-x_inf) for tt in t])
    for index, name in enumerate(["i(la)", "i(lb)", "i(lc)"]):
        assert np.max(np.abs(result.signals[name] - reference[:, index])) < 1e-8
    v_star_reference = np.mean(emf - resistance * reference, axis=1)
    assert np.max(np.abs(result.signals["v(star)"] - v_star_reference)) < 1e-8
    total = (
        result.signals["i(la)"] + result.signals["i(lb)"] + result.signals["i(lc)"]
    )
    assert np.max(np.abs(total)) < 1e-12


def test_initial_currents_violating_kcl_fail_closed(tmp_path: Path) -> None:
    path = _write(tmp_path, i_a="1 A", i_b="0 A", i_c="0 A", stop="1 ms")
    with pytest.raises(SimulationError, match="violate KCL at the floating"):
        HybridSimulator(load_model(path)).simulate()


def test_current_source_at_floating_node_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """
schema: pear-sim/v0.1
name: bad_star

circuit:
  ground: gnd
  components:
    - {id: ea, type: voltage_source, p: a, n: gnd, value: 10 V}
    - {id: ra, type: resistor, p: a, n: ma, value: 1 ohm}
    - {id: la, type: inductor, p: ma, n: star, value: 5 mH}
    - {id: lb, type: inductor, p: star, n: gnd, value: 5 mH}
    - {id: isrc, type: current_source, p: star, n: gnd, value: 1 A}

analysis:
  transient: {start: 0 s, stop: 1 ms, output_step: 1 us, solver: dopri5, rtol: 1e-9, atol: 1e-11}

outputs: [i(la)]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TopologyError, match="no algebraic support"):
        HybridSimulator(load_model(path)).simulate()


def test_two_inductor_chain_node_is_reduced_too(tmp_path: Path) -> None:
    """Two series inductors: the shared node is a two-member 'star'."""

    path = tmp_path / "model.yaml"
    path.write_text(
        """
schema: pear-sim/v0.1
name: series_inductors

circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 10 V}
    - {id: r1, type: resistor, p: in, n: m1, value: 1 ohm}
    - {id: l1, type: inductor, p: m1, n: mid, value: 3 mH, initial_current: 0 A}
    - {id: l2, type: inductor, p: mid, n: gnd, value: 7 mH, initial_current: 0 A}

analysis:
  transient: {start: 0 s, stop: 50 ms, output_step: 20 us, solver: dopri5, rtol: 1e-10, atol: 1e-12}

outputs: [i(l1), i(l2), v(mid)]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result = HybridSimulator(load_model(path)).simulate()
    t = result.time
    # Series RL with L = L1 + L2; v(mid) divides the inductive voltage by ratio.
    tau = 10e-3 / 1.0
    reference = 10.0 * (1.0 - np.exp(-t / tau))
    assert np.max(np.abs(result.signals["i(l1)"] - reference)) < 5e-8
    assert np.max(np.abs(result.signals["i(l1)"] - result.signals["i(l2)"])) < 1e-12
    v_l_total = 10.0 - 1.0 * reference
    assert np.max(np.abs(result.signals["v(mid)"] - 0.7 * v_l_total)) < 1e-6


def test_floating_neutral_inverter_runs_and_conserves_kcl() -> None:
    example = ROOT / "examples" / "inverter_three_phase_floating_neutral.yaml"
    from dataclasses import replace

    model = load_model(example)
    model = replace(
        model, transient=replace(model.transient, stop=4e-3), harmonics=None
    )
    result = HybridSimulator(model).simulate()
    total = (
        result.signals["i(la)"] + result.signals["i(lb)"] + result.signals["i(lc)"]
    )
    assert np.max(np.abs(total)) < 1e-9
    # Common-mode voltage of a two-level bridge: +/-Vdc/6*2 and +/-Vdc/2 levels.
    cmv = result.signals["v(vsn)"]
    assert np.max(cmv) == pytest.approx(200.0, abs=1.0)
    assert np.max(np.abs(cmv[np.abs(np.abs(cmv) - 200.0 / 3.0) < 5.0])) > 0.0
    assert result.physics_audit["full"]["balance_status"] == "within_numerical_tolerance"


def test_dead_time_at_floating_star_freezes_and_recovers(tmp_path: Path) -> None:
    """Dead time at the floating star runs via state freezing.

    A leg whose current sits at zero during its dead-time window has no
    conduction path: the active-set search freezes that inductor (zero-
    current reduction) and the star reduces over the remaining live legs.
    The first floating-star version livelocked here; the livelock guard stays for genuinely
    unsupported shapes.
    """

    import yaml

    document = yaml.safe_load(
        (ROOT / "examples" / "inverter_three_phase_floating_neutral.yaml").read_text()
    )
    document["parameters"]["dead_time"] = "500 ns"
    document["analysis"]["transient"]["stop"] = "2 ms"
    document["analysis"].pop("harmonics", None)
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = HybridSimulator(load_model(path)).simulate()
    total = (
        result.signals["i(la)"] + result.signals["i(lb)"] + result.signals["i(lc)"]
    )
    assert np.max(np.abs(total)) < 1e-9
    # Cold start actually exercises the freeze path: dead-time windows with
    # all currents at zero must appear as constrained-state events.
    assert result.stats["state_constraint_events"] > 0
    assert result.physics_audit["full"]["balance_status"] == "within_numerical_tolerance"
