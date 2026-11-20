# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Switching-transition post-processing: format, closed-form verification, mutations.

VERIFICATION, not validation: the oracles here are closed forms evaluated by
EXACT piecewise-polynomial integration of literal tables written in this file
(never the producer's parser or its CossCurve), so a shared mistake in table
handling cannot make both sides agree. The producer integrates the transition
in time with an adaptive ODE solver and reads the table through its own
interpolation; agreement between the two is agreement between two methods.
The independent engine (ngspice) lives in test_switching_transitions_ngspice.py.

Every gate here is shown both ways: silent on the correct code, firing on a
mutation (Coss x 1.2, inverted commutating current).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from numpy.polynomial import polynomial as P

from pear_sim.analysis import switching_transitions as st
from pear_sim.analysis.switching_transitions import (
    CossCurve,
    build_switching_transitions_observation,
    capacitive_turn_on_energy,
    instant_cut_overshoot,
    ramp_overshoot,
    simulate_transition,
)
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.device_cards import CardProvenance
from pear_sim.spec.model import load_model
from pear_sim.spec.switching_transitions import CossCard

ROOT = Path(__file__).parents[1]
BUCK = ROOT / "examples" / "synchronous_buck_gan_transition.yaml"

# Literal tables, independent of every example file.
AXIS = (0.0, 10.0, 20.0, 40.0, 60.0, 80.0)
COSS = (1.34e-9, 1.23e-9, 0.64e-9, 0.535e-9, 0.49e-9, 0.445e-9)
QOSS_AXIS = (0.0, 10.0, 20.0, 40.0, 48.0, 80.0)
QOSS = (0.0, 12.5e-9, 19.0e-9, 33.5e-9, 37.0e-9, 50.0e-9)

# Declared verification budget: relative 1e-6 against exact closed forms
# (adaptive RK45 at rtol 1e-10 plus event location; measured margin ~1e-8).
CLOSED_FORM_RTOL = 1e-6


# --- independent closed forms -------------------------------------------------------


class _Exact:
    """Exact integrals of weight(w)·C(w) for a piecewise-linear or -constant table."""

    def __init__(self, axis, values, law):
        self.axis = list(axis)
        if law == "coss":
            self.segments = [
                (a, b, np.array([ca - (cb - ca) / (b - a) * a, (cb - ca) / (b - a)]))
                for a, b, ca, cb in zip(axis, axis[1:], values, values[1:])
            ]
            tail = values[-1]
        else:
            slopes = [(qb - qa) / (b - a) for a, b, qa, qb in zip(axis, axis[1:], values, values[1:])]
            self.segments = [(a, b, np.array([c])) for a, b, c in zip(axis, axis[1:], slopes)]
            tail = slopes[-1]
        self.segments.append((axis[-1], math.inf, np.array([tail])))

    def integral(self, lo, hi, weight=(1.0,)):
        total = 0.0
        for a, b, c in self.segments:
            x0, x1 = max(lo, a), min(hi, b)
            if x1 <= x0:
                continue
            antiderivative = P.polyint(P.polymul(c, np.array(weight, dtype=float)))
            total += P.polyval(x1, antiderivative) - P.polyval(x0, antiderivative)
        return float(total)

    def capacitance(self, w):
        for a, b, c in self.segments:
            if a <= w < b:
                return float(P.polyval(w, c))
        return float(P.polyval(w, self.segments[-1][2]))


def _card(axis=AXIS, coss=COSS, qoss=None, scale=1.0):
    return CossCard(
        id="t",
        voltage_axis_v=tuple(axis),
        coss_f=None if coss is None else tuple(scale * c for c in coss),
        qoss_c=None if qoss is None else tuple(scale * q for q in qoss),
        provenance=CardProvenance("literal test table"),
    )


def _node(curve, bus, extra=0.0):
    return st._node_capacitance(curve, bus, extra)


def _closed_form_swing_time(exact, bus, threshold, current, extra=0.0):
    charge = exact.integral(threshold, bus) + exact.integral(0.0, bus - threshold)
    return (charge + extra * (bus - threshold)) / current


def _closed_form_energy_needed(exact, bus, threshold, far_u):
    # ∫_thr^V (u_x − u)·[Cin(u) + Cout(V − u)] du, the second term with w = V − u.
    first = exact.integral(threshold, bus, (far_u, -1.0))
    second = exact.integral(0.0, bus - threshold, (far_u - bus, 1.0))
    return first + second


# --- format -------------------------------------------------------------------------


def _write(tmp_path: Path, transitions: str, *, outputs_extra: str = "") -> Path:
    body = f"""
schema: pear-sim/v0.1
name: transition_format
circuit:
  ground: gnd
  components:
    - {{id: vin, type: voltage_source, p: in, n: gnd, value: 48 V}}
    - {{id: qh, type: ideal_switch, p: in, n: sw, control: leg.high}}
    - {{id: ql, type: ideal_switch, p: sw, n: gnd, control: leg.low}}
    - {{id: dh, type: diode, p: sw, n: in, initial: off}}
    - {{id: dl, type: diode, p: gnd, n: sw, initial: off}}
    - {{id: l1, type: inductor, p: sw, n: out, value: 4.7 uH, initial_current: 10 A}}
    - {{id: c1, type: capacitor, p: out, n: gnd, value: 47 uF, initial_voltage: 12 V}}
    - {{id: rload, type: resistor, p: out, n: gnd, value: 1.2 ohm}}
controls:
  leg: {{type: complementary_pwm, frequency: 500 kHz, duty: 25 %, phase: 0 s, dead_time: 20 ns}}
switching_transitions:
{transitions}
analysis:
  transient: {{start: 0 s, stop: 10 us, output_step: 2 ns, solver: dop853, rtol: 1.0e-9, atol: 1.0e-11}}
outputs: [v(in), v(sw), i(l1), i(qh), i(ql), i(dh), i(dl){outputs_extra}]
"""
    path = tmp_path / "model.yaml"
    path.write_text(body, encoding="utf-8")
    return path


_CARDS = """  coss_cards:
    c:
      voltage_axis_v: [0, 10, 80]
      coss_f: [1.0e-9, 0.6e-9, 0.4e-9]
      provenance: {source: test}
"""
_LEG = """  legs:
    leg: {high: qh, low: ql, high_diode: dh, low_diode: dl, card: c, bus_voltage: v(in)}
"""


def test_valid_section_loads(tmp_path):
    model = load_model(_write(tmp_path, _CARDS + _LEG))
    spec = model.switching_transitions
    assert spec.legs[0].regime == "constant_current"
    assert spec.cards["c"].law == "coss_piecewise_linear"
    assert spec.zvs_threshold == pytest.approx(0.01)


@pytest.mark.parametrize(
    ("cards", "legs", "outputs_extra", "message"),
    [
        (_CARDS.replace("[0, 10, 80]", "[1, 10, 80]"), _LEG, "", "must start at 0 V"),
        (_CARDS.replace("coss_f: [1.0e-9, 0.6e-9, 0.4e-9]", "coss_f: [1.0e-9, 0.6e-9, 0.4e-9]\n      qoss_c: [0, 1e-9, 2e-9]"), _LEG, "", "exactly one of|valid under each"),
        (_CARDS.replace("      provenance: {source: test}\n", ""), _LEG, "", "provenance"),
        (_CARDS.replace("[1.0e-9, 0.6e-9, 0.4e-9]", "[1.0e-9, 0.0, 0.4e-9]"), _LEG, "", "positive"),
        (_CARDS, _LEG.replace(", low_diode: dl", ""), "", "low_diode"),
        (_CARDS, _LEG.replace("bus_voltage: v(in)}", "bus_voltage: v(in), commutation_inductance: 1 uH}"), "", "go together"),
        (_CARDS, _LEG.replace("high: qh, low: ql", "high: ql, low: qh"), "", "switching node"),
        (_CARDS, _LEG.replace("high_diode: dh, low_diode: dl", "high_diode: dl, low_diode: dh"), "", "antiparallel"),
        (_CARDS, _LEG.replace("bus_voltage: v(in)}", "bus_voltage: v(out)}"), "", "must be listed in outputs"),
        (_CARDS, _LEG.replace("bus_voltage: v(in)}", "bus_voltage: v(in), bridge_partner: leg}"), "", "another declared leg"),
        (_CARDS, _LEG.replace("bus_voltage: v(in)}", "bus_voltage: v(in), current_fall_time: 2 ns}"), "", "needs loop_inductance"),
    ],
)
def test_invalid_sections_fail_closed(tmp_path, cards, legs, outputs_extra, message):
    with pytest.raises(ModelError, match=message):
        load_model(_write(tmp_path, cards + legs, outputs_extra=outputs_extra))


# --- closed-form verification ---------------------------------------------------------


@pytest.mark.parametrize("law", ["coss", "qoss"])
@pytest.mark.parametrize("bus", [48.0, 80.0, 120.0])  # 120 V exercises the held tail
@pytest.mark.parametrize("current", [0.7, 3.0, 25.0])
def test_constant_current_swing_time_matches_charge_over_current(law, bus, current):
    axis, values = (AXIS, COSS) if law == "coss" else (QOSS_AXIS, QOSS)
    card = _card(axis, coss=values if law == "coss" else None, qoss=values if law == "qoss" else None)
    exact = _Exact(axis, values, law)
    threshold = 0.01 * bus
    extra = 47e-12
    outcome = simulate_transition(
        [_node(CossCurve(card), bus, extra)], bus, current, 1e-6, threshold
    )
    expected = _closed_form_swing_time(exact, bus, threshold, current, extra)
    assert outcome.minimum_dead_time_s == pytest.approx(expected, rel=CLOSED_FORM_RTOL)


def test_partial_swing_voltage_matches_charge_balance():
    # Constant current, dead time shorter than the swing: the node voltage u_r at
    # turn-on satisfies ∫_{u_r}^{V} C_node du = I·t_dead.
    bus, current, dead = 48.0, 2.0, 20e-9
    exact = _Exact(AXIS, COSS, "coss")
    outcome = simulate_transition([_node(CossCurve(_card()), bus)], bus, current, dead, 0.48)
    residual = outcome.voltage_at_turn_on_v[0]
    delivered = exact.integral(residual, bus) + exact.integral(0.0, bus - residual)
    assert delivered == pytest.approx(current * dead, rel=CLOSED_FORM_RTOL)
    assert outcome.minimum_dead_time_s is None or outcome.minimum_dead_time_s > dead


@pytest.mark.parametrize("far_u", [-20.0, 10.0, 30.0, 70.0])
def test_lumped_lc_zvs_boundary_matches_energy_condition(far_u):
    # ZVS (u reaches the threshold) iff ½·L·I² ≥ ∫_thr^V (u_x − u)·C_node(u) du.
    bus, inductance, threshold = 48.0, 200e-9, 0.48
    exact = _Exact(AXIS, COSS, "coss")
    needed = _closed_form_energy_needed(exact, bus, threshold, far_u)
    curve = CossCurve(_card())
    if needed <= 0.0:
        outcome = simulate_transition([_node(curve, bus)], bus, 1e-3, 5e-6, threshold, inductance, far_u)
        assert outcome.minimum_dead_time_s is not None
        return
    critical = math.sqrt(2.0 * needed / inductance)
    above = simulate_transition([_node(curve, bus)], bus, 1.01 * critical, 5e-6, threshold, inductance, far_u)
    below = simulate_transition([_node(curve, bus)], bus, 0.99 * critical, 5e-6, threshold, inductance, far_u)
    assert above.minimum_dead_time_s is not None
    assert below.minimum_dead_time_s is None


def test_lumped_lc_maximum_dead_time_after_clamp():
    # After the clamp at 0 the current falls as di/dt = -u_x/L; the node leaves
    # the threshold just after it reverses: t_max ≈ t_clamp + L·i_clamp/u_x.
    bus, inductance, far_u = 48.0, 100e-9, 30.0
    outcome = simulate_transition(
        [_node(CossCurve(_card()), bus)], bus, 6.0, 40e-9, 0.48, inductance, far_u
    )
    assert outcome.minimum_dead_time_s is not None
    assert outcome.maximum_dead_time_s is not None
    assert outcome.minimum_dead_time_s < outcome.maximum_dead_time_s < 40e-9
    assert outcome.voltage_at_turn_on_v[0] > 0.48  # dead time too long: ZVS lost again


@pytest.mark.parametrize("residual", [48.0, 20.0, 5.0])
def test_capacitive_turn_on_energy_closed_form(residual):
    bus = 48.0
    exact = _Exact(AXIS, COSS, "coss")
    expected = exact.integral(0.0, residual, (0.0, 1.0)) + exact.integral(bus - residual, bus, (bus, -1.0))
    got = capacitive_turn_on_energy(CossCurve(_card()), bus, residual, 0.0)
    assert got == pytest.approx(expected, rel=1e-9)
    if residual == bus:
        # identical devices, full hard turn-on: Qoss(V)·V
        assert got == pytest.approx(exact.integral(0.0, bus) * bus, rel=1e-9)


@pytest.mark.parametrize("loop_nh", [0.4, 5.0])
def test_instant_cut_linear_capacitance(loop_nh):
    capacitance = 500e-12
    card = _card((0.0, 1000.0), coss=(capacitance, capacitance))
    current, loop = 10.0, loop_nh * 1e-9
    delta = instant_cut_overshoot(CossCurve(card), 48.0, current, loop)
    assert delta == pytest.approx(current * math.sqrt(loop / capacitance), rel=1e-9)


@pytest.mark.parametrize("fall_ns", [0.5, 2.0, 5.0, 20.0])
def test_ramp_overshoot_linear_capacitance(fall_ns):
    # Linear C, current falling linearly over t_f: a = L·I/t_f, θ = ω·t_f,
    # peak = a·max(1 − cos(min(θ, π)), 2·|sin(θ/2)|).
    capacitance, loop, current, bus = 500e-12, 5e-9, 10.0, 48.0
    fall = fall_ns * 1e-9
    card = _card((0.0, 1000.0), coss=(capacitance, capacitance))
    theta = fall / math.sqrt(loop * capacitance)
    a = loop * current / fall
    expected = a * max(1.0 - math.cos(min(theta, math.pi)), 2.0 * abs(math.sin(theta / 2.0)))
    got = ramp_overshoot(CossCurve(card), bus, current, loop, fall)
    assert got == pytest.approx(expected, rel=1e-6)


# --- end to end on the shipped example ---------------------------------------------------


@pytest.fixture(scope="module")
def buck_result():
    return HybridSimulator(load_model(BUCK)).simulate()


def _buck_gate(observation, exact, bus=48.0):
    """Closed-form gate over the example's high->low edges. Returns the worst
    relative deviation of minimum_dead_time_s from Q_node/I (inf when an edge
    the closed form calls ZVS-reachable reports none)."""

    worst = 0.0
    for edge in observation["legs"][0]["edges"]:
        if edge["direction"] != "high_to_low":
            continue
        current = edge["commutating_current_a"]
        expected = _closed_form_swing_time(exact, bus, 0.01 * bus, current)
        got = edge["minimum_dead_time_s"]
        if got is None:
            return math.inf
        worst = max(worst, abs(got - expected) / expected)
    return worst


def test_example_matches_closed_form_and_inductor_current(buck_result):
    observation = build_switching_transitions_observation(buck_result)
    exact = _Exact(
        (0, 10, 20, 40, 60, 80), (1.34e-9, 1.23e-9, 0.64e-9, 0.535e-9, 0.49e-9, 0.445e-9), "coss"
    )
    assert _buck_gate(observation, exact) < CLOSED_FORM_RTOL
    leg = observation["legs"][0]
    assert leg["summary"]["by_direction"]["high_to_low"]["zvs_edges"] == 10
    assert leg["summary"]["by_direction"]["low_to_high"]["zvs_edges"] == 0
    # The commutating current is the inductor current at the edge (independent read).
    time = np.asarray(buck_result.time)
    inductor = np.asarray(buck_result.signals["i(l1)"])
    for edge in leg["edges"]:
        expected = np.interp(edge["time_s"], time, inductor)
        sign = 1.0 if edge["direction"] == "high_to_low" else -1.0
        assert sign * edge["commutating_current_a"] == pytest.approx(expected, rel=2e-3)
    # Hard low->high edges: E = Qoss(48 V)·48 V for identical devices.
    for edge in leg["edges"]:
        if edge["direction"] == "low_to_high":
            assert edge["capacitive_turn_on_energy_j"] == pytest.approx(
                exact.integral(0.0, 48.0) * 48.0, rel=1e-9
            )
    assert observation["validation_status"].startswith("verified")
    assert "MEASURED" not in str(observation)


def test_gate_fires_on_coss_scaled_by_1_2(buck_result, monkeypatch):
    exact = _Exact(
        (0, 10, 20, 40, 60, 80), (1.34e-9, 1.23e-9, 0.64e-9, 0.535e-9, 0.49e-9, 0.445e-9), "coss"
    )
    original = CossCurve.__call__
    monkeypatch.setattr(CossCurve, "__call__", lambda self, v: 1.2 * original(self, v))
    deviation = _buck_gate(build_switching_transitions_observation(buck_result), exact)
    assert deviation > 0.15  # the mutation is a 20 % charge error; the gate sees it


def test_gate_fires_on_inverted_commutating_current(buck_result, monkeypatch):
    exact = _Exact(
        (0, 10, 20, 40, 60, 80), (1.34e-9, 1.23e-9, 0.64e-9, 0.535e-9, 0.49e-9, 0.445e-9), "coss"
    )
    original = st.simulate_transition

    def flipped(capacitances, bus, current, *args, **kwargs):
        return original(capacitances, bus, -current, *args, **kwargs)

    monkeypatch.setattr(st, "simulate_transition", flipped)
    assert _buck_gate(build_switching_transitions_observation(buck_result), exact) == math.inf


def test_models_without_section_have_no_artifact(tmp_path):
    from pear_sim.run.io import write_result

    model = load_model(ROOT / "examples" / "synchronous_buck_complementary.yaml")
    out = tmp_path / "run"
    write_result(HybridSimulator(model).simulate(), out)
    assert not (out / "switching-transitions.json").exists()
