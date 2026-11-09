# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Magnetically coupled GROUPS, not just pairs.

A three-winding transformer — the arrangement behind a Ddy supply feeding two
six-pulse bridges — is declared as three pairwise mutuals. Every winding
reachable through a mutual then shares one inductance matrix whose inverse
maps branch voltages to di/dt, and two windings is simply the special case.

Pairwise k < 1 is necessary but not sufficient once three windings share a
core: the assembled matrix must be positive definite, or some current set
would carry negative stored energy. That is what Cholesky decides.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from pear_sim.core.circuit import Circuit, _coupling_groups
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.model import load_model


def _three_winding(
    tmp_path: Path,
    *,
    primary_h: float = 1.0,
    secondary_h: float = 0.25,
    tertiary_h: float = 0.25,
    k: float = 0.99,
    load_ohm: float = 200.0,
    stop: str = "40 ms",
):
    body = f"""
schema: pear-sim/v0.1
name: three_winding
circuit:
  ground: gnd
  components:
    - id: vsrc
      type: voltage_source
      p: in
      n: gnd
      waveform: {{type: sine, amplitude: 100 V, frequency: 50 Hz}}
    - {{id: rsrc, type: resistor, p: in, n: pri, value: 1 ohm}}
    - {{id: lp, type: inductor, p: pri, n: gnd, value: {primary_h} H, initial_current: 0 A}}
    - {{id: ls, type: inductor, p: seca, n: gnd, value: {secondary_h} H, initial_current: 0 A}}
    - {{id: rs, type: resistor, p: seca, n: gnd, value: {load_ohm} ohm}}
    - {{id: lt, type: inductor, p: terc, n: gnd, value: {tertiary_h} H, initial_current: 0 A}}
    - {{id: rt, type: resistor, p: terc, n: gnd, value: {load_ohm} ohm}}
  couplings:
    - {{inductors: [lp, ls], coupling: {k}}}
    - {{inductors: [lp, lt], coupling: {k}}}
    - {{inductors: [ls, lt], coupling: {k}}}
analysis:
  transient:
    start: 0 s
    stop: {stop}
    output_step: 20 us
    solver: radau
    rtol: 1.0e-8
    atol: 1.0e-10
outputs: [i(lp), v(seca), v(terc), i(rs), i(rt)]
"""
    path = tmp_path / "transformer.yaml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return load_model(path)


def test_windings_joined_by_mutuals_form_one_group() -> None:
    class _Coupling:
        def __init__(self, first, second):
            self.first, self.second = first, second

    groups = _coupling_groups(
        [_Coupling("lp", "ls"), _Coupling("lp", "lt"), _Coupling("lx", "ly")]
    )
    assert groups == (("lp", "ls", "lt"), ("lx", "ly"))


def test_group_matrix_carries_declared_mutuals_and_zeros_elsewhere(
    tmp_path: Path,
) -> None:
    model = _three_winding(tmp_path, k=0.9)
    circuit = Circuit(model)
    assert circuit.coupling_groups == (("lp", "ls", "lt"),)
    matrix = circuit.coupling_matrix(("lp", "ls", "lt"))

    expected_ps = 0.9 * math.sqrt(1.0 * 0.25)
    expected_st = 0.9 * math.sqrt(0.25 * 0.25)
    assert matrix[0, 0] == pytest.approx(1.0)
    assert matrix[1, 1] == pytest.approx(0.25)
    assert matrix[0, 1] == pytest.approx(expected_ps)
    assert matrix[1, 2] == pytest.approx(expected_st)
    # Symmetric, and positive definite as the loader proved.
    assert np.allclose(matrix, matrix.T)
    np.linalg.cholesky(matrix)


def test_undeclared_pair_inside_a_group_stays_uncoupled(tmp_path: Path) -> None:
    """A group member pair with no declared mutual is genuinely uncoupled.

    The couplings must be loose enough for that to be PHYSICAL, which is a
    real constraint rather than a test artifact: two secondaries each tightly
    coupled to a common primary are necessarily coupled to EACH OTHER, and
    declaring their mutual as zero makes the assembled matrix indefinite. At
    k = 0.5 the zero is admissible and the kernel must not invent a value.
    """

    body = """
schema: pear-sim/v0.1
name: partial_group
circuit:
  ground: gnd
  components:
    - {id: vsrc, type: voltage_source, p: pri, n: gnd, value: 10 V}
    - {id: lp, type: inductor, p: pri, n: gnd, value: 1 H, initial_current: 0 A}
    - {id: ls, type: inductor, p: seca, n: gnd, value: 0.25 H, initial_current: 0 A}
    - {id: rs, type: resistor, p: seca, n: gnd, value: 50 ohm}
    - {id: lt, type: inductor, p: terc, n: gnd, value: 0.25 H, initial_current: 0 A}
    - {id: rt, type: resistor, p: terc, n: gnd, value: 50 ohm}
  couplings:
    - {inductors: [lp, ls], coupling: 0.5}
    - {inductors: [lp, lt], coupling: 0.5}
analysis:
  transient: {start: 0 s, stop: 1 ms, output_step: 10 us}
outputs: [i(lp)]
"""
    path = tmp_path / "partial.yaml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    circuit = Circuit(load_model(path))

    # All three windings share ONE group, reached through the primary.
    assert circuit.coupling_groups == (("lp", "ls", "lt"),)
    matrix = circuit.coupling_matrix(("lp", "ls", "lt"))
    assert matrix[1, 2] == pytest.approx(0.0)
    np.linalg.cholesky(matrix)


def test_two_tight_secondaries_need_their_own_mutual(tmp_path: Path) -> None:
    """The physical corollary, asserted: omitting it is refused, not fudged."""

    body = """
schema: pear-sim/v0.1
name: tight_without_secondary_mutual
circuit:
  ground: gnd
  components:
    - {id: vsrc, type: voltage_source, p: pri, n: gnd, value: 10 V}
    - {id: lp, type: inductor, p: pri, n: gnd, value: 1 H, initial_current: 0 A}
    - {id: ls, type: inductor, p: seca, n: gnd, value: 0.25 H, initial_current: 0 A}
    - {id: rs, type: resistor, p: seca, n: gnd, value: 50 ohm}
    - {id: lt, type: inductor, p: terc, n: gnd, value: 0.25 H, initial_current: 0 A}
    - {id: rt, type: resistor, p: terc, n: gnd, value: 50 ohm}
  couplings:
    - {inductors: [lp, ls], coupling: 0.9}
    - {inductors: [lp, lt], coupling: 0.9}
analysis:
  transient: {start: 0 s, stop: 1 ms, output_step: 10 us}
outputs: [i(lp)]
"""
    path = tmp_path / "tight.yaml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    with pytest.raises(ModelError, match="positive definite"):
        load_model(path)


def test_indefinite_group_fails_closed(tmp_path: Path) -> None:
    # Every pair is individually admissible (k < 1), but the combination is
    # not: with L = 1 H each, k(p,s) = k(p,t) = 0.99 and k(s,t) = 0.5 gives a
    # determinant of -0.23, so some winding-current set would store negative
    # energy. Pairwise checks cannot see this; Cholesky on the group does.
    with pytest.raises(ModelError, match="positive definite"):
        body = """
schema: pear-sim/v0.1
name: indefinite
circuit:
  ground: gnd
  components:
    - {id: vsrc, type: voltage_source, p: pri, n: gnd, value: 10 V}
    - {id: lp, type: inductor, p: pri, n: gnd, value: 1 H, initial_current: 0 A}
    - {id: ls, type: inductor, p: seca, n: gnd, value: 1 H, initial_current: 0 A}
    - {id: rs, type: resistor, p: seca, n: gnd, value: 10 ohm}
    - {id: lt, type: inductor, p: terc, n: gnd, value: 1 H, initial_current: 0 A}
    - {id: rt, type: resistor, p: terc, n: gnd, value: 10 ohm}
  couplings:
    - {inductors: [lp, ls], coupling: 0.99}
    - {inductors: [lp, lt], coupling: 0.99}
    - {inductors: [ls, lt], coupling: 0.5}
analysis:
  transient: {start: 0 s, stop: 1 ms, output_step: 10 us}
outputs: [i(lp)]
"""
        path = tmp_path / "indefinite.yaml"
        path.write_text(body.strip() + "\n", encoding="utf-8")
        load_model(path)


def test_both_secondaries_receive_power_and_the_balance_closes(
    tmp_path: Path,
) -> None:
    model = _three_winding(tmp_path, load_ohm=200.0, stop="40 ms")
    result = HybridSimulator(model).simulate()

    secondary = np.asarray(result.signals["v(seca)"])
    tertiary = np.asarray(result.signals["v(terc)"])
    tail = slice(len(secondary) // 2, None)

    # Both secondaries carry a real voltage, and identical windings with
    # identical loads must carry the SAME one.
    secondary_rms = float(np.sqrt(np.mean(secondary[tail] ** 2)))
    tertiary_rms = float(np.sqrt(np.mean(tertiary[tail] ** 2)))
    assert secondary_rms > 5.0
    assert tertiary_rms == pytest.approx(secondary_rms, rel=1e-6)

    # The turns ratio bounds the transfer: with L2/L1 = 1/4 the open-circuit
    # ratio is k*sqrt(L2/L1) = 0.495, and loading can only reduce it.
    primary = np.asarray(result.signals["v(seca)"])  # placeholder guard
    assert primary is not None
    assert secondary_rms < 0.5 * 100.0 / math.sqrt(2.0)

    # The energy audit is the independent check that the group's inverse
    # inductance matrix and the stored-energy cross terms agree.
    audit = result.physics_audit["full"]
    assert audit["balance_within_tolerance"], audit["balance_residual_j"]
