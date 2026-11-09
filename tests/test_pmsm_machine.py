# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Surface-magnet synchronous machine: expansion, physics, and the splitting error.

Every reference here is closed form. A non-salient PMSM feeding a symmetric
resistive star at held speed is a balanced three-phase source behind
``R_s + jw*L_s``:

    E_peak = omega_e * psi_f,  omega_e = p * omega_mechanical
    I_peak = E_peak / |R_s + R_load + j*omega_e*L_s|
    P_electrical = 3 * (I_peak/sqrt(2))**2 * (R_s + R_load)
    T_braking = P_electrical / omega_mechanical

The kernel holds the speed within one integration segment while the rotor
angle still advances exactly, so the splitting error must SHRINK with the
segment length — that is what makes the approximation measurable rather
than merely declared, and the convergence test below asserts it.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pear_sim.core.solver import HybridSimulator
from pear_sim.run.io import write_result
from pear_sim.spec.machines import PHASE_OFFSETS, electromagnetic_torque, parse_machines
from pear_sim.spec.model import load_model

FLUX = 0.1
POLE_PAIRS = 4
STATOR_RESISTANCE = 0.05
STATOR_INDUCTANCE = 5e-4
INERTIA = 0.01


def _generator_model(
    tmp_path: Path,
    *,
    load_ohm: float = 2.0,
    speed: float = 100.0,
    inertia: float = INERTIA,
    stop: str = "20 ms",
    max_step: str = "",
):
    step = f"\n    max_step: {max_step}" if max_step else ""
    body = f"""
schema: pear-sim/v0.1
name: pmsm_generator
circuit:
  ground: gnd
  components:
    - {{id: ra, type: resistor, p: ta, n: gnd, value: {load_ohm} ohm}}
    - {{id: rb, type: resistor, p: tb, n: gnd, value: {load_ohm} ohm}}
    - {{id: rc, type: resistor, p: tc, n: gnd, value: {load_ohm} ohm}}
machines:
  m1:
    type: pmsm
    terminals: [ta, tb, tc]
    neutral: gnd
    pole_pairs: {POLE_PAIRS}
    stator_resistance_ohm: {STATOR_RESISTANCE}
    stator_inductance_h: {STATOR_INDUCTANCE}
    flux_linkage_wb: {FLUX}
    inertia_kg_m2: {inertia}
    initial_speed_rad_s: {speed}
analysis:
  transient:
    start: 0 s
    stop: {stop}
    output_step: 20 us
    solver: dopri5
    rtol: 1.0e-10
    atol: 1.0e-12{step}
outputs: [i(m1.a.l), i(m1.b.l), i(m1.c.l), v(ta)]
"""
    path = tmp_path / "machine.yaml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return load_model(path)


def test_declaration_expands_into_auditable_branches(tmp_path: Path) -> None:
    model = _generator_model(tmp_path)
    ids = {component.id for component in model.components}
    for label in ("a", "b", "c"):
        assert {f"m1.{label}.r", f"m1.{label}.l", f"m1.{label}.e"} <= ids
    windings = [c for c in model.components if c.id.endswith(".l")]
    assert all(c.value == pytest.approx(STATOR_INDUCTANCE) for c in windings)
    resistances = [c for c in model.components if c.id.endswith(".r")]
    assert all(c.value == pytest.approx(STATOR_RESISTANCE) for c in resistances)
    assert len(model.machines) == 1
    assert model.machines[0].pole_pairs == POLE_PAIRS


def test_saliency_is_now_modelled_rather_than_refused() -> None:
    """Contract change: saliency.

    This test used to assert the opposite: that declaring Ld != Lq was a
    load-time error. It is now a modelled machine — the winding set becomes
    one coupled group whose matrix is re-evaluated every integration segment.
    The refusal is kept only for a machine with NEITHER magnets NOR saliency,
    which has no torque mechanism at all. See tests/test_salient_machine.py
    for the physics.
    """

    spec = parse_machines(
        {
            "m1": {
                "type": "pmsm",
                "terminals": ["a", "b", "c"],
                "neutral": "n",
                "pole_pairs": 2,
                "stator_resistance_ohm": 0.1,
                "stator_inductance_h": 1e-3,
                "d_axis_inductance_h": 1e-3,
                "q_axis_inductance_h": 2e-3,
                "flux_linkage_wb": 0.1,
                "inertia_kg_m2": 0.01,
            }
        }
    )[0]
    assert spec.is_salient
    assert spec.d_axis_inductance_h == pytest.approx(1e-3)
    assert spec.q_axis_inductance_h == pytest.approx(2e-3)


def test_terminal_current_matches_the_closed_form(tmp_path: Path) -> None:
    load = 2.0
    speed = 100.0
    # A large inertia keeps the speed effectively constant so the closed form
    # for a held-speed source applies over the whole window.
    model = _generator_model(
        tmp_path, load_ohm=load, speed=speed, inertia=1e4, stop="20 ms"
    )
    result = HybridSimulator(model).simulate()

    electrical_speed = POLE_PAIRS * speed
    emf_peak = electrical_speed * FLUX
    impedance = math.hypot(
        STATOR_RESISTANCE + load, electrical_speed * STATOR_INDUCTANCE
    )
    expected_peak = emf_peak / impedance

    current = np.asarray(result.signals["i(m1.a.l)"])
    tail = current[len(current) // 2 :]
    assert float(np.max(np.abs(tail))) == pytest.approx(expected_peak, rel=1e-3)
    voltage = np.asarray(result.signals["v(ta)"])[len(current) // 2 :]
    assert float(np.max(np.abs(voltage))) == pytest.approx(
        expected_peak * load, rel=1e-3
    )

    # Three-wire consistency: the phase currents sum to zero.
    total = (
        current
        + np.asarray(result.signals["i(m1.b.l)"])
        + np.asarray(result.signals["i(m1.c.l)"])
    )
    assert float(np.max(np.abs(total))) < 1e-9


def test_braking_torque_matches_the_dissipated_power(tmp_path: Path) -> None:
    load = 2.0
    speed = 100.0
    model = _generator_model(
        tmp_path, load_ohm=load, speed=speed, inertia=1e4, stop="20 ms"
    )
    result = HybridSimulator(model).simulate()

    electrical_speed = POLE_PAIRS * speed
    impedance = math.hypot(
        STATOR_RESISTANCE + load, electrical_speed * STATOR_INDUCTANCE
    )
    peak = electrical_speed * FLUX / impedance
    dissipated = 3.0 * (peak / math.sqrt(2.0)) ** 2 * (STATOR_RESISTANCE + load)
    expected_torque = -dissipated / speed  # braking

    trace = result.machine_trace
    assert trace, "the machine must record a mechanical trace"
    tail = [row["mean_torque_nm"] for row in trace[len(trace) // 2 :]]
    assert float(np.mean(tail)) == pytest.approx(expected_torque, rel=2e-3)


def test_torque_and_back_emf_satisfy_the_power_identity() -> None:
    """sum(e_x*i_x) must equal T_e*omega_mechanical for any current set."""

    spec = parse_machines(
        {
            "m1": {
                "type": "pmsm",
                "terminals": ["a", "b", "c"],
                "neutral": "n",
                "pole_pairs": POLE_PAIRS,
                "stator_resistance_ohm": STATOR_RESISTANCE,
                "stator_inductance_h": STATOR_INDUCTANCE,
                "flux_linkage_wb": FLUX,
                "inertia_kg_m2": INERTIA,
            }
        }
    )[0]
    speed = 137.0
    electrical_speed = POLE_PAIRS * speed
    for angle in (0.0, 0.7, 2.5, 4.1):
        currents = (11.0, -4.0, -7.0)  # sums to zero, as a three-wire set must
        emf = tuple(
            -electrical_speed * FLUX * math.sin(angle - offset)
            for _label, offset in PHASE_OFFSETS
        )
        electrical_power = sum(e * i for e, i in zip(emf, currents, strict=True))
        torque = electromagnetic_torque(spec, currents, angle)
        assert electrical_power == pytest.approx(torque * speed, rel=1e-12)


def test_frozen_speed_error_shrinks_with_the_segment(tmp_path: Path) -> None:
    """The declared approximation must be measurable, not merely asserted."""

    def final_speed(max_step: str) -> float:
        directory = tmp_path / max_step.replace(" ", "")
        directory.mkdir(parents=True, exist_ok=True)
        model = _generator_model(
            directory,
            load_ohm=2.0,
            speed=100.0,
            inertia=INERTIA,
            stop="5 ms",
            max_step=max_step,
        )
        return HybridSimulator(model).simulate().machine_trace[-1]["speed_end_rad_s"]

    coarse = final_speed("50 us")
    medium = final_speed("10 us")
    fine = final_speed("2 us")
    # Successive halving of the segment must move the answer monotonically
    # less: the splitting error is converging, not wandering.
    assert abs(medium - fine) < abs(coarse - fine)
    assert abs(medium - fine) / abs(fine) < 1e-3


def test_bundle_records_the_machine_trace(tmp_path: Path) -> None:
    model = _generator_model(tmp_path, stop="2 ms")
    result = HybridSimulator(model).simulate()
    destination = tmp_path / "bundle"
    write_result(result, destination)

    rows = [
        json.loads(line)
        for line in (destination / "machine-trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows and rows[0]["machine_id"] == "m1"
    assert rows[0]["speed_rad_s"] == pytest.approx(100.0)
    # Generator braking: the machine slows down.
    assert rows[-1]["speed_end_rad_s"] < rows[0]["speed_rad_s"]

    run = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    assert run["files"]["machine_trace"] == "machine-trace.jsonl"


def _ideal_supply_model(tmp_path: Path, *, sequence: str) -> Path:
    """Machine on a balanced ideal supply, positive or reversed sequence."""

    phases = {"abc": (0, -120, 120), "acb": (0, 120, -120)}[sequence]
    body = f"""
schema: pear-sim/v0.1
name: pmsm_ideal_supply
circuit:
  ground: gnd
  components:
    - id: va
      type: voltage_source
      p: ta
      n: gnd
      waveform: {{type: sine, amplitude: 63.28 V, frequency: 100 Hz, phase_deg: {phases[0]}}}
    - id: vb
      type: voltage_source
      p: tb
      n: gnd
      waveform: {{type: sine, amplitude: 63.28 V, frequency: 100 Hz, phase_deg: {phases[1]}}}
    - id: vc
      type: voltage_source
      p: tc
      n: gnd
      waveform: {{type: sine, amplitude: 63.28 V, frequency: 100 Hz, phase_deg: {phases[2]}}}
machines:
  motor:
    type: pmsm
    terminals: [ta, tb, tc]
    neutral: nstar
    pole_pairs: {POLE_PAIRS}
    stator_resistance_ohm: {STATOR_RESISTANCE}
    stator_inductance_h: 0.002
    flux_linkage_wb: {FLUX}
    inertia_kg_m2: {INERTIA}
    viscous_friction_nm_s: 0.01
    load_torque_nm: 1.0
    initial_speed_rad_s: 157.0796
    initial_angle_rad: 3.05646
analysis:
  transient:
    start: 0 s
    stop: 60 ms
    output_step: 20 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11
outputs: [i(motor.a.l), i(motor.b.l), i(motor.c.l)]
"""
    path = tmp_path / f"{sequence}.yaml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return path


def test_positive_sequence_supply_makes_the_machine_motor(tmp_path: Path) -> None:
    """A supply leading the rotor must MOTOR, at the load's own torque.

    This is the sequence-sensitive check. A reversed b/c offset keeps the
    internal power identity intact — both the EMF and the torque would use
    the same wrong offsets — while silently turning a motor into a brake, so
    the identity alone cannot catch it and this test must.
    """

    model = load_model(_ideal_supply_model(tmp_path, sequence="abc"))
    result = HybridSimulator(model).simulate()
    trace = result.machine_trace
    torque = float(np.mean([row["mean_torque_nm"] for row in trace[len(trace) // 2 :]]))

    # Load plus viscous friction at speed. The window runs long enough for
    # the rotor's hunting transient to settle; over a shorter one the mean
    # still carries the oscillation and lands a few percent low.
    expected = 1.0 + 0.01 * 157.0796
    assert torque > 0.0, "a positive-sequence supply must produce motoring torque"
    assert torque == pytest.approx(expected, rel=5e-3)

    # Synchronism holds: the rotor hunts around the supply frequency instead
    # of slipping away from it.
    speeds = np.array([row["speed_end_rad_s"] for row in trace])
    assert float(np.max(np.abs(speeds - 157.0796))) < 2.0

    current = np.asarray(result.signals["i(motor.a.l)"])
    tail = current[len(current) // 2 :]
    assert float(np.max(np.abs(tail))) == pytest.approx(4.285, rel=0.1)


def test_reversed_sequence_supply_brakes_instead(tmp_path: Path) -> None:
    model = load_model(_ideal_supply_model(tmp_path, sequence="acb"))
    result = HybridSimulator(model).simulate()
    trace = result.machine_trace
    torque = float(np.mean([row["mean_torque_nm"] for row in trace[len(trace) // 2 :]]))
    assert torque < 0.0, "a reversed-sequence supply cannot drive this rotor"
