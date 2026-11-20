# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Switching transitions against ngspice: the independent-engine verification.

ngspice integrates the SAME physical case with its own engine: behavioural
capacitors ``C='pwl(V, table)'`` written from literal tables in this file, its
own time stepping, and its own initial operating point (sources step at t=0
so the DC solution sets the starting node voltages and inductor current —
behavioural capacitors ignore ``ic=``). Agreement is VERIFICATION of the
post-processing integrator, not validation against hardware.

Declared budget: relative 2e-3 (ngspice transient tolerances at the options
below, plus the 1 ps source steps used to set the initial state). Each gate is
shown silent on correct code and firing on a mutation (Coss x 1.2).

Runs in CI's ``ngspice-oracle`` job, where a skip is a failure.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from pear_sim.analysis import switching_transitions as st
from pear_sim.analysis.switching_transitions import (
    CossCurve,
    instant_cut_overshoot,
    ramp_overshoot,
    simulate_transition,
)
from pear_sim.oracle.ngspice import find_ngspice
from pear_sim.spec.device_cards import CardProvenance
from pear_sim.spec.switching_transitions import CossCard

BUDGET = 2e-3

_live = pytest.mark.skipif(find_ngspice() is None, reason="no ngspice binary on this machine")

AXIS = (0.0, 10.0, 20.0, 40.0, 60.0, 80.0)
COSS = (1.34e-9, 1.23e-9, 0.64e-9, 0.535e-9, 0.49e-9, 0.445e-9)
HV_AXIS = (0.0, 10.0, 50.0, 100.0, 200.0, 400.0, 650.0)
HV_COSS = (1.5e-9, 6.0e-10, 2.0e-10, 1.2e-10, 8.0e-11, 6.5e-11, 5.5e-11)

_OPTIONS = "* ngspice default transient tolerances"


def _pwl(probe: str, axis, values, scale: float = 1.0) -> str:
    pairs = ", ".join(f"{v:g},{scale * c:.6e}" for v, c in zip(axis, values))
    return f"pwl({probe}, {pairs})"


def _card(axis=AXIS, coss=COSS, scale=1.0) -> CossCard:
    return CossCard(
        "t", tuple(axis), tuple(scale * c for c in coss), None, CardProvenance("literal test table")
    )


def _ngspice(tmp_path: Path, name: str, netlist: str) -> dict[str, float]:
    path = tmp_path / f"{name}.cir"
    path.write_text(netlist, encoding="utf-8")
    completed = subprocess.run(
        [find_ngspice(), "-b", str(path)], capture_output=True, text=True, timeout=120
    )
    values = {}
    for line in (completed.stdout + completed.stderr).splitlines():
        match = re.match(r"^\s*(\w+)\s*=\s*([-+0-9.eE]+)", line)
        if match:
            values[match.group(1).lower()] = float(match.group(2))
    if not values:
        raise AssertionError(f"ngspice produced no measurements:\n{completed.stdout}\n{completed.stderr}")
    return values


def _leg_caps(node: str, bus: str, axis, values, scale=1.0) -> str:
    return (
        f"Cl_{node} {node} 0 C='{_pwl(f'V({node})', axis, values, scale)}'\n"
        f"Ch_{node} {bus} {node} C='{_pwl(f'V({bus},{node})', axis, values, scale)}'\n"
    )


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-30)


# --- constant current -------------------------------------------------------------------


def _constant_current_netlist(bus, current, stop, threshold, scale=1.0, axis=AXIS, values=COSS):
    return f"""* constant-current dead-time swing
VB bus 0 DC {bus}
RL n bus 1e12
{_leg_caps('n', 'bus', axis, values, scale)}
I1 n 0 PWL(0 0 1p {current})
{_OPTIONS}
.tran 1p {stop} 0 2p
.control
run
meas tran tcross WHEN V(n)={threshold} FALL=1
quit
.endc
.end
"""


@_live
@pytest.mark.parametrize(("bus", "current"), [(48.0, 10.0), (48.0, 2.0), (80.0, 5.0)])
def test_constant_current_swing_time(tmp_path, bus, current):
    threshold = 0.01 * bus
    curve = CossCurve(_card())
    pear = simulate_transition([st._node_capacitance(curve, bus, 0.0)], bus, current, 1e-6, threshold)
    spice = _ngspice(
        tmp_path, "cc", _constant_current_netlist(bus, current, 3 * pear.minimum_dead_time_s, threshold)
    )
    # ngspice's current source ramps over 1 ps: half a picosecond of delay.
    assert _rel(pear.minimum_dead_time_s, spice["tcross"] - 0.5e-12) < BUDGET


@_live
def test_constant_current_gate_fires_on_coss_mutation(tmp_path):
    bus, current = 48.0, 10.0
    spice = _ngspice(tmp_path, "cc", _constant_current_netlist(bus, current, 30e-9, 0.48))
    mutated = CossCurve(_card(scale=1.2))
    pear = simulate_transition([st._node_capacitance(mutated, bus, 0.0)], bus, current, 1e-6, 0.48)
    assert _rel(pear.minimum_dead_time_s, spice["tcross"] - 0.5e-12) > 0.15


# --- lumped LC, single leg -------------------------------------------------------------


def _lc_netlist(bus, current, far, inductance, at, scale=1.0):
    return f"""* lumped LC dead-time swing, initial state from the DC operating point
VB bus 0 DC {bus}
VX far 0 PWL(0 {bus} 1p {far})
IINIT 0 n PWL(0 {current} 1p 0)
{_leg_caps('n', 'bus', AXIS, COSS, scale)}
L1 n far {inductance}
{_OPTIONS}
.tran 1p {at * 1.2} 0 2p
.control
run
meas tran vat FIND V(n) AT={at}
meas tran iat FIND i(L1) AT={at}
quit
.endc
.end
"""


@_live
@pytest.mark.parametrize(("current", "far", "at"), [(2.0, 30.0, 40e-9), (1.0, 10.0, 15e-9), (3.0, 45.0, 25e-9)])
def test_lumped_lc_node_voltage_and_current(tmp_path, current, far, at):
    bus, inductance = 48.0, 100e-9
    curve = CossCurve(_card())
    pear = simulate_transition(
        [st._node_capacitance(curve, bus, 0.0)], bus, current, at, 0.48, inductance, far
    )
    if pear.minimum_dead_time_s is not None and pear.minimum_dead_time_s < at:
        pytest.fail("case reaches the clamp; choose a free-swing case for the engine comparison")
    spice = _ngspice(tmp_path, "lc", _lc_netlist(bus, current, far, inductance, at))
    assert abs(pear.voltage_at_turn_on_v[0] - spice["vat"]) < BUDGET * bus
    assert abs(pear.current_at_end_a - spice["iat"]) < BUDGET * max(abs(current), 1.0)


@_live
def test_lumped_lc_gate_fires_on_coss_mutation(tmp_path):
    bus, inductance, current, far, at = 48.0, 100e-9, 2.0, 30.0, 40e-9
    spice = _ngspice(tmp_path, "lc", _lc_netlist(bus, current, far, inductance, at))
    mutated = CossCurve(_card(scale=1.2))
    pear = simulate_transition(
        [st._node_capacitance(mutated, bus, 0.0)], bus, current, at, 0.48, inductance, far
    )
    assert abs(pear.voltage_at_turn_on_v[0] - spice["vat"]) > 10 * BUDGET * bus


# --- bridge pair (full bridge, both legs swing together) ----------------------------------


@_live
def test_bridge_pair_against_ngspice(tmp_path):
    # Leg a falls (A: V -> 0), leg b rises (B: 0 -> V); the path from A to B is
    # L in series with a source v_t. u-frame: u_x = V + v_t.
    bus, inductance, current, v_t, at = 380.0, 50e-6, 1.5, 150.0, 50e-9
    netlist = f"""* full-bridge pair on one commutation current
VB bus 0 DC {bus}
* The pair floats as a whole (only L and VT join a to b): it needs DC paths.
* 100 Mohm leaks set a = V, b = 0 at the operating point; they carry ~4 uA
* against 1.5 A (3e-6) and shift the start by ~0.04 V (1e-4 of V): in budget.
RA a bus 1e8
RB b 0 1e8
{_leg_caps('a', 'bus', HV_AXIS, HV_COSS)}
{_leg_caps('b', 'bus', HV_AXIS, HV_COSS)}
L1 a x {inductance}
* The loop current is set by a source ACROSS the inductor, stepped off at t=0.
ILOOP x a PWL(0 {current} 1p 0)
VT x b PWL(0 {bus} 1p {v_t})
.options rshunt=1e12
.tran 1p {at * 1.2} 0 5p
.control
run
meas tran va FIND V(a) AT={at}
meas tran vb FIND V(b) AT={at}
meas tran iat FIND i(L1) AT={at}
quit
.endc
.end
"""
    curve = CossCurve(_card(HV_AXIS, HV_COSS))
    node = st._node_capacitance(curve, bus, 0.0)
    pear = simulate_transition([node, node], bus, current, at, 4.0, inductance, bus + v_t)
    assert pear.minimum_dead_time_s is None or pear.minimum_dead_time_s > at, "free-swing case"
    spice = _ngspice(tmp_path, "pair", netlist)
    u_a, u_b = pear.voltage_at_turn_on_v
    assert abs(u_a - spice["va"]) < BUDGET * bus
    assert abs(u_b - (bus - spice["vb"])) < BUDGET * bus
    assert abs(pear.current_at_end_a - spice["iat"]) < BUDGET * current


# --- overshoot -------------------------------------------------------------------------


def _overshoot_netlist(bus, current, loop, fall, stop, scale=1.0):
    return f"""* commutation-loop overshoot: device current falls linearly over t_f
VB bus 0 DC {bus}
L1 bus d {loop}
Cd d 0 C='{_pwl('V(d)', AXIS, COSS, scale)}'
ID d 0 PWL(0 {current} {fall} 0)
{_OPTIONS}
.tran 0.1p {stop} 0 1p
.control
run
meas tran vmax MAX V(d)
quit
.endc
.end
"""


@_live
@pytest.mark.parametrize(("loop_nh", "fall_ns"), [(0.4, 2.0), (2.0, 1.0), (5.0, 10.0)])
def test_ramp_overshoot_against_ngspice(tmp_path, loop_nh, fall_ns):
    bus, current = 48.0, 10.0
    loop, fall = loop_nh * 1e-9, fall_ns * 1e-9
    pear = ramp_overshoot(CossCurve(_card()), bus, current, loop, fall)
    stop = fall + 4 * 3.1416 * (loop * 1.34e-9) ** 0.5
    spice = _ngspice(tmp_path, "ovs", _overshoot_netlist(bus, current, loop, fall, stop))
    assert abs(pear - (spice["vmax"] - bus)) < BUDGET * max(pear, 1.0) + 1e-3


@_live
def test_instant_cut_matches_very_fast_ramp(tmp_path):
    bus, current, loop = 48.0, 10.0, 2e-9
    delta = instant_cut_overshoot(CossCurve(_card()), bus, current, loop)
    stop = 4 * 3.1416 * (loop * 1.34e-9) ** 0.5
    spice = _ngspice(tmp_path, "cut", _overshoot_netlist(bus, current, loop, 1e-12, stop))
    # A 1 ps fall is not instant; the residual is first order in t_f·ω (~1e-3 here).
    assert _rel(delta, spice["vmax"] - bus) < 5e-3


@_live
def test_overshoot_gate_fires_on_coss_mutation(tmp_path):
    bus, current, loop, fall = 48.0, 10.0, 2e-9, 1e-9
    stop = fall + 4 * 3.1416 * (loop * 1.34e-9) ** 0.5
    spice = _ngspice(tmp_path, "ovs", _overshoot_netlist(bus, current, loop, fall, stop))
    mutated = ramp_overshoot(CossCurve(_card(scale=1.2)), bus, current, loop, fall)
    assert abs(mutated - (spice["vmax"] - bus)) > 10 * BUDGET * (spice["vmax"] - bus)
