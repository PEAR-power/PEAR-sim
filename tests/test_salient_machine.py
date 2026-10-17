"""Salient-rotor machine: reluctance torque and a position-dependent matrix.

A synchronous reluctance machine is NON-EXCITED — all of its torque comes
from the difference between the d- and q-axis inductances. A model that
forces ``Ld = Lq`` reports exactly zero for it, which is why salient-rotor
support exists.

These tests pin the physics rather than the plumbing: that the declared axis
inductances are what a Park transform actually reads back at every rotor
angle, that torque survives with no magnets at all, that the unobservable
zero-sequence choice changes nothing, and that a non-salient machine is
untouched.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.machines import (
    PHASE_OFFSETS,
    parse_machines,
    salient_inductance_matrix,
    salient_torque,
)
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]
SYNRM = ROOT / "examples" / "synrm_torque_test.yaml"
OFFSETS = [offset for _, offset in PHASE_OFFSETS]


def _spec(d_axis: float, q_axis: float, flux: float = 0.1):
    return parse_machines(
        {
            "m": {
                "type": "pmsm",
                "terminals": ["a", "b", "c"],
                "neutral": "n",
                "pole_pairs": 2,
                "stator_resistance_ohm": 0.1,
                "stator_inductance_h": 2e-3,
                "flux_linkage_wb": flux,
                "inertia_kg_m2": 0.01,
                "d_axis_inductance_h": d_axis,
                "q_axis_inductance_h": q_axis,
            }
        }
    )[0]


def _park(vector, angle: float) -> tuple[float, float]:
    d = (2.0 / 3.0) * sum(
        value * math.cos(angle - offset)
        for value, offset in zip(vector, OFFSETS, strict=True)
    )
    q = -(2.0 / 3.0) * sum(
        value * math.sin(angle - offset)
        for value, offset in zip(vector, OFFSETS, strict=True)
    )
    return d, q


# --------------------------------------------------------------------------
# The matrix means what it says
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("d_axis", "q_axis"),
    [(1.5e-3, 3.0e-3), (3.0e-3, 1.5e-3), (0.5e-3, 4.0e-3), (2.0e-3, 2.0e-3)],
)
def test_park_reads_back_the_declared_axis_inductances(d_axis, q_axis) -> None:
    """A declared Ld/Lq must be what the transform measures, at EVERY angle.

    This is the test that caught a sign slip during development: the matrix
    was structurally right (no d-q cross coupling) while reporting the axes
    swapped, which no smoke test would have shown.
    """

    spec = _spec(d_axis, q_axis)
    for angle in np.linspace(0.0, 2.0 * math.pi, 17):
        matrix = salient_inductance_matrix(spec, float(angle))
        d_current = np.array([math.cos(angle - o) for o in OFFSETS])
        q_current = np.array([-math.sin(angle - o) for o in OFFSETS])
        read_d, cross = _park(matrix @ d_current, float(angle))
        _, read_q = _park(matrix @ q_current, float(angle))
        assert read_d == pytest.approx(d_axis, abs=1e-15)
        assert read_q == pytest.approx(q_axis, abs=1e-15)
        # The axes must not leak into each other.
        assert abs(cross) < 1e-15


def test_matrix_is_symmetric_and_positive_definite_all_the_way_round() -> None:
    spec = _spec(1.0e-3, 3.0e-3)
    for angle in np.linspace(0.0, 2.0 * math.pi, 33):
        matrix = salient_inductance_matrix(spec, float(angle))
        assert np.allclose(matrix, matrix.T, atol=1e-18)
        assert float(np.min(np.linalg.eigvalsh(matrix))) > 0.0


def test_equal_axes_collapse_to_the_constant_matrix() -> None:
    """Continuity with the non-salient machine, checked rather than assumed."""

    spec = _spec(2.0e-3, 2.0e-3)
    first = salient_inductance_matrix(spec, 0.0)
    for angle in (0.7, 1.9, 4.4):
        assert np.allclose(salient_inductance_matrix(spec, angle), first, atol=1e-18)


# --------------------------------------------------------------------------
# Torque
# --------------------------------------------------------------------------


def test_a_machine_with_no_magnets_still_makes_torque() -> None:
    """The whole point of the item.

    With psi_f = 0 the magnet term vanishes and ONLY the reluctance term is
    left. Under the old model — which forced Ld = Lq — this machine produced
    exactly zero.
    """

    spec = _spec(1.0e-3, 3.0e-3, flux=0.0)
    angle = 0.4
    # A current at 45 degrees in the dq plane maximises i_d*i_q.
    magnitude = 10.0
    currents = tuple(
        magnitude * math.cos(angle - offset - math.pi / 4.0) for offset in OFFSETS
    )
    torque = salient_torque(spec, currents, angle)
    assert abs(torque) > 1e-3
    # Closed form: T = 1.5*p*(Ld-Lq)*i_d*i_q. For this current pattern the
    # Park transform gives i_d = +A/sqrt(2) and i_q = -A/sqrt(2) — the sign
    # of i_q is what the first version of this test got wrong, so it is
    # written out rather than assumed.
    i_d = magnitude / math.sqrt(2.0)
    i_q = -magnitude / math.sqrt(2.0)
    expected = 1.5 * 2 * (1.0e-3 - 3.0e-3) * i_d * i_q
    assert torque == pytest.approx(expected, rel=1e-9)


def test_reluctance_torque_vanishes_when_the_axes_are_equal() -> None:
    # A machine with equal axes AND no magnets is refused outright (it would
    # have no torque mechanism at all), so the reluctance term is isolated on
    # a magnet machine instead: its torque must equal the magnet term alone.
    from pear_sim.spec.machines import electromagnetic_torque

    spec = _spec(2.0e-3, 2.0e-3, flux=0.1)
    currents = tuple(10.0 * math.cos(0.4 - offset - math.pi / 4.0) for offset in OFFSETS)
    assert salient_torque(spec, currents, 0.4) == pytest.approx(
        electromagnetic_torque(spec, currents, 0.4), rel=1e-12
    )


def test_a_machine_with_neither_magnets_nor_saliency_is_refused() -> None:
    with pytest.raises(ModelError, match="no torque mechanism"):
        _spec(2.0e-3, 2.0e-3, flux=0.0)


def test_magnet_term_matches_the_non_salient_closed_form() -> None:
    from pear_sim.spec.machines import electromagnetic_torque

    spec = _spec(2.0e-3, 2.0e-3, flux=0.1)
    currents = (12.0, -5.0, -7.0)
    assert salient_torque(spec, currents, 0.9) == pytest.approx(
        electromagnetic_torque(spec, currents, 0.9), rel=1e-12
    )


# --------------------------------------------------------------------------
# The zero sequence is unobservable, and that is measured
# --------------------------------------------------------------------------


def test_result_does_not_depend_on_the_zero_sequence_choice() -> None:
    """The one number in the matrix that physics does not fix.

    A three-wire machine offers no path to zero-sequence current, so the
    matrix entry along [1,1,1] is unobservable — but the kernel needs it
    non-zero to invert the matrix. This test sweeps it by a factor of ten
    and asserts nothing moves. Without it, the choice would be a hidden
    assumption instead of an explicit one.
    """

    import pear_sim.core.solver as solver_module
    import pear_sim.spec.machines as machines_module

    original = machines_module.salient_inductance_matrix

    def scaled(factor: float):
        # The signature gained a `currents` argument when flux maps landed;
        # accept and forward it so this probe keeps testing what it says.
        def build(spec, angle, currents=None):
            matrix = original(spec, angle, currents)
            l_a = (spec.d_axis_inductance_h + spec.q_axis_inductance_h) / 3.0
            ones = np.ones((3, 3), dtype=float) / 3.0
            return matrix + (factor - 1.0) * l_a * ones

        return build

    reference = None
    try:
        for factor in (1.0, 10.0):
            solver_module.salient_inductance_matrix = scaled(factor)
            result = HybridSimulator(load_model(SYNRM)).simulate()
            current = result.signals["i(motor.a.l)"]
            if reference is None:
                reference = current
                continue
            deviation = float(np.max(np.abs(current - reference)))
            assert deviation / float(np.max(np.abs(reference))) < 1e-12
    finally:
        solver_module.salient_inductance_matrix = original


# --------------------------------------------------------------------------
# Backward compatibility and fail-closed
# --------------------------------------------------------------------------


def test_a_non_salient_machine_declares_no_couplings() -> None:
    model = load_model(ROOT / "examples" / "pmsm_inverter_drive.yaml")
    assert model.machines[0].is_salient is False
    assert model.couplings == ()


def test_a_salient_machine_expands_into_one_coupled_group() -> None:
    model = load_model(SYNRM)
    spec = model.machines[0]
    assert spec.is_salient
    coupled = {coupling.first for coupling in model.couplings} | {
        coupling.second for coupling in model.couplings
    }
    assert coupled == set(spec.winding_ids)
    # Three windings means three declared pairs. Their SIGNS are not fixed:
    # the ordinary -L_A/2 phase coupling, the saliency term and the explicit
    # zero-sequence entry all land on the off-diagonal, so a strongly salient
    # rotor can push one pair positive at some angles. What must hold is the
    # energy condition, and it is checked on the assembled matrix.
    assert len(model.couplings) == 3
    matrix = salient_inductance_matrix(spec, spec.initial_angle_rad)
    assert float(np.min(np.linalg.eigvalsh(matrix))) > 0.0


def test_declaring_only_one_axis_is_refused() -> None:
    with pytest.raises(ModelError, match="declare both"):
        parse_machines(
            {
                "m": {
                    "type": "pmsm",
                    "terminals": ["a", "b", "c"],
                    "neutral": "n",
                    "pole_pairs": 2,
                    "stator_resistance_ohm": 0.1,
                    "stator_inductance_h": 2e-3,
                    "flux_linkage_wb": 0.1,
                    "inertia_kg_m2": 0.01,
                    "d_axis_inductance_h": 1e-3,
                }
            }
        )


def test_the_shipped_reluctance_example_produces_torque_without_magnets() -> None:
    model = load_model(SYNRM)
    assert model.machines[0].flux_linkage_wb == 0.0
    result = HybridSimulator(model).simulate()
    torque = result.signals["torque(motor)"]
    assert float(np.max(np.abs(torque))) > 1.0


# --------------------------------------------------------------------------
# Flux continuity across segment boundaries
# --------------------------------------------------------------------------


def test_flux_linkage_is_continuous_across_a_matrix_swap() -> None:
    """The winding currents must be transformed, not carried, when L changes.

    Regression: a locked rotor (no matrix swaps) looks fine while a rotating one
    diverges by 40-50 % RMS. Carrying the CURRENT across a segment boundary
    makes psi = L(theta)*i jump by (L_new - L_old)*i at every swap;
    thousands of swaps per run integrate into that divergence, and the
    reluctance voltage omega*dL/dtheta*i is exactly the physics those jumps
    discard. Preserving psi removes the jumps.
    """

    from pear_sim.core.circuit import Circuit
    from pear_sim.core.solver import _MachineDriver

    model = load_model(SYNRM)
    circuit = Circuit(model)
    driver = _MachineDriver(circuit, model)
    spec = model.machines[0]
    state = circuit.initial_state.copy()
    indices = [circuit.state_index[w] for w in spec.winding_ids]
    # A zero-sum current set, as the floating star enforces.
    state[indices[0]], state[indices[1]], state[indices[2]] = 10.0, -4.0, -6.0

    driver.begin_segment(0.0, state)
    matrix_old = salient_inductance_matrix(spec, driver.state[spec.id]["angle_rad"])
    flux_before = matrix_old @ np.array([state[i] for i in indices])

    # Advance the rotor far enough that the matrix genuinely changes.
    driver.state[spec.id]["angle_rad"] += 0.3
    driver.begin_segment(1e-4, state)
    matrix_new = salient_inductance_matrix(spec, driver.state[spec.id]["angle_rad"])
    currents_after = np.array([state[i] for i in indices])
    flux_after = matrix_new @ currents_after

    assert np.allclose(flux_after, flux_before, atol=1e-15)
    # The transform must not have leaked zero-sequence current.
    assert float(sum(currents_after)) == pytest.approx(0.0, abs=1e-12)
    # And it genuinely moved the currents (L changed, so i had to).
    assert abs(currents_after[0] - 10.0) > 1e-3


def test_a_non_salient_machine_never_touches_the_transform() -> None:
    """Backward compatibility: the constant-L machine takes the old path."""

    from pear_sim.core.circuit import Circuit
    from pear_sim.core.solver import _MachineDriver

    model = load_model(ROOT / "examples" / "pmsm_inverter_drive.yaml")
    circuit = Circuit(model)
    driver = _MachineDriver(circuit, model)
    state = circuit.initial_state.copy()
    before = state.copy()
    driver.begin_segment(0.0, state)
    driver.begin_segment(1e-4, state)
    assert np.array_equal(state, before)
