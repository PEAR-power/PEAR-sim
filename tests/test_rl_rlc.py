# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""VAL-001: analytic RL and series-RLC step goldens (partial).

Closed-form references, documented tolerances. The RLC case covers the
underdamped and overdamped regimes; critical damping is deliberately not
asserted exactly (it is a measure-zero parameter point).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator

RL_TOLERANCE_A = 2e-7
RLC_TOLERANCE_V = 1e-5


def _simulate(tmp_path: Path, name: str, body: str):
    path = tmp_path / f"{name}.yaml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return HybridSimulator(load_model(path)).simulate()


def test_rl_step_current_matches_analytic_solution(tmp_path: Path) -> None:
    result = _simulate(
        tmp_path,
        "rl",
        """
schema: pear-sim/v0.1
name: rl_step
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 10 V}
    - {id: r1, type: resistor, p: in, n: mid, value: 100 ohm}
    - {id: l1, type: inductor, p: mid, n: gnd, value: 10 mH, initial_current: 0 A}
analysis:
  transient: {start: 0 s, stop: 0.5 ms, output_step: 1 us, solver: dopri5, rtol: 1e-10, atol: 1e-12}
outputs: [i(l1), v(mid)]
""",
    )
    tau = 10e-3 / 100.0
    expected_current = 0.1 * (1.0 - np.exp(-result.time / tau))
    assert np.max(np.abs(result.signals["i(l1)"] - expected_current)) < RL_TOLERANCE_A
    expected_inductor_voltage = 10.0 * np.exp(-result.time / tau)
    assert np.max(np.abs(result.signals["v(mid)"] - expected_inductor_voltage)) < 100.0 * RL_TOLERANCE_A


def _rlc_body(resistance_ohm: float) -> str:
    return f"""
schema: pear-sim/v0.1
name: rlc_step
circuit:
  ground: gnd
  components:
    - {{id: vin, type: voltage_source, p: in, n: gnd, value: 10 V}}
    - {{id: r1, type: resistor, p: in, n: a, value: {resistance_ohm} ohm}}
    - {{id: l1, type: inductor, p: a, n: b, value: 10 mH, initial_current: 0 A}}
    - {{id: c1, type: capacitor, p: b, n: gnd, value: 1 uF, initial_voltage: 0 V}}
analysis:
  transient: {{start: 0 s, stop: 2 ms, output_step: 1 us, solver: dopri5, rtol: 1e-10, atol: 1e-12}}
outputs: [v(b), i(l1)]
"""


def test_series_rlc_step_underdamped_matches_analytic_solution(tmp_path: Path) -> None:
    resistance = 100.0
    result = _simulate(tmp_path, "rlc_under", _rlc_body(resistance))
    inductance, capacitance, source = 10e-3, 1e-6, 10.0
    alpha = resistance / (2.0 * inductance)
    omega0 = 1.0 / np.sqrt(inductance * capacitance)
    assert alpha < omega0, "fixture must be underdamped"
    omega_d = np.sqrt(omega0**2 - alpha**2)
    t = result.time
    expected = source * (
        1.0 - np.exp(-alpha * t) * (np.cos(omega_d * t) + (alpha / omega_d) * np.sin(omega_d * t))
    )
    assert np.max(np.abs(result.signals["v(b)"] - expected)) < RLC_TOLERANCE_V


def test_series_rlc_step_overdamped_matches_analytic_solution(tmp_path: Path) -> None:
    resistance = 300.0
    result = _simulate(tmp_path, "rlc_over", _rlc_body(resistance))
    inductance, capacitance, source = 10e-3, 1e-6, 10.0
    alpha = resistance / (2.0 * inductance)
    omega0 = 1.0 / np.sqrt(inductance * capacitance)
    assert alpha > omega0, "fixture must be overdamped"
    root = np.sqrt(alpha**2 - omega0**2)
    s1, s2 = -alpha + root, -alpha - root
    t = result.time
    expected = source * (1.0 - (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1))
    assert np.max(np.abs(result.signals["v(b)"] - expected)) < RLC_TOLERANCE_V
