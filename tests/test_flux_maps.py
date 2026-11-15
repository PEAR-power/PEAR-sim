# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Saturating flux maps psi(id, iq) in the machine.

A real reluctance or interior-magnet machine works deep in saturation, where
constant Ld/Lq express neither the falling inductance nor the cross-coupling
between axes. The map states the flux linkage directly.

THE ORACLE IS INTERNAL AND EXACT: a map that happens to be LINEAR must
reproduce the constant-Ld/Lq machine bit for bit, because it is the same
physics written differently. That equivalence is the gate; everything a
saturating map does beyond it is the new capability.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.machines import (
    mapped_torque,
    park_currents,
    parse_machines,
    salient_inductance_matrix,
    salient_torque,
)
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]
LINEAR_REF = ROOT / "examples" / "pmsm_foc_current_control.yaml"
SATURATING = ROOT / "examples" / "pmsm_saturating_flux_map.yaml"
LD, LQ, PSI_F = 0.0015, 0.003, 0.1
AXIS = [-40.0, -20.0, 0.0, 20.0, 40.0]


def _machine(psi_d, psi_q, axis=AXIS):
    return parse_machines(
        {
            "m": {
                "type": "pmsm", "terminals": ["a", "b", "c"], "neutral": "n",
                "pole_pairs": 4, "stator_resistance_ohm": 0.05,
                "inertia_kg_m2": 0.01,
                "flux_map": {
                    "id_axis": list(axis), "iq_axis": list(axis),
                    "psi_d": [[psi_d(i, q) for q in axis] for i in axis],
                    "psi_q": [[psi_q(i, q) for q in axis] for i in axis],
                },
            }
        }
    )[0]


def _linear_map():
    return _machine(lambda i, q: LD * i + PSI_F, lambda i, q: LQ * q)


# --------------------------------------------------------------------------
# The linear map IS the constant-inductance machine
# --------------------------------------------------------------------------


def test_a_linear_map_reports_the_inductances_it_encodes() -> None:
    spec = _linear_map()
    assert spec.is_saturating and spec.is_salient
    assert spec.d_axis_inductance_h == pytest.approx(LD, rel=1e-9)
    assert spec.q_axis_inductance_h == pytest.approx(LQ, rel=1e-9)
    assert spec.flux_linkage_wb == pytest.approx(PSI_F, rel=1e-12)


def test_a_linear_map_builds_the_same_inductance_matrix() -> None:
    mapped = _linear_map()
    constant = parse_machines(
        {
            "m": {
                "type": "pmsm", "terminals": ["a", "b", "c"], "neutral": "n",
                "pole_pairs": 4, "stator_resistance_ohm": 0.05,
                "stator_inductance_h": 0.5 * (LD + LQ), "flux_linkage_wb": PSI_F,
                "inertia_kg_m2": 0.01,
                "d_axis_inductance_h": LD, "q_axis_inductance_h": LQ,
            }
        }
    )[0]
    for angle in np.linspace(0.0, 2.0 * math.pi, 9):
        from_map = salient_inductance_matrix(mapped, float(angle), (10.0, -4.0, -6.0))
        from_constants = salient_inductance_matrix(constant, float(angle))
        assert np.allclose(from_map, from_constants, atol=1e-15)


def test_mapped_torque_collapses_to_the_closed_form_when_flux_is_linear() -> None:
    """T = 1.5p(psi_d*iq - psi_q*id) must reduce to the familiar formula."""

    mapped = _linear_map()
    constant = parse_machines(
        {
            "m": {
                "type": "pmsm", "terminals": ["a", "b", "c"], "neutral": "n",
                "pole_pairs": 4, "stator_resistance_ohm": 0.05,
                "stator_inductance_h": 0.5 * (LD + LQ), "flux_linkage_wb": PSI_F,
                "inertia_kg_m2": 0.01,
                "d_axis_inductance_h": LD, "q_axis_inductance_h": LQ,
            }
        }
    )[0]
    for currents, angle in (((10.0, -4.0, -6.0), 0.7), ((-8.0, 15.0, -7.0), 2.1)):
        assert mapped_torque(mapped, currents, angle) == pytest.approx(
            salient_torque(constant, currents, angle), rel=1e-9
        )


def test_a_linear_map_reproduces_the_whole_drive_run() -> None:
    """End-to-end equivalence: same currents, same torque, to numerical zero."""

    source = yaml.safe_load(LINEAR_REF.read_text(encoding="utf-8"))
    machine = source["machines"]["motor"]
    for field in ("d_axis_inductance_h", "q_axis_inductance_h",
                  "flux_linkage_wb", "stator_inductance_h"):
        machine.pop(field, None)
    machine["flux_map"] = {
        "id_axis": list(AXIS), "iq_axis": list(AXIS),
        "psi_d": [[LD * i + PSI_F for _ in AXIS] for i in AXIS],
        "psi_q": [[LQ * q for q in AXIS] for _ in AXIS],
    }
    variant = ROOT / "tests" / "_flux_linear.yaml"
    try:
        variant.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
        mapped = HybridSimulator(load_model(variant)).simulate()
    finally:
        variant.unlink(missing_ok=True)
    reference = HybridSimulator(load_model(LINEAR_REF)).simulate()
    for signal in ("i(motor.a.l)", "torque(motor)"):
        deviation = float(
            np.max(np.abs(np.asarray(mapped.signals[signal])
                          - np.asarray(reference.signals[signal])))
        )
        scale = float(np.max(np.abs(reference.signals[signal])))
        assert deviation / scale < 1e-10


# --------------------------------------------------------------------------
# What saturation adds
# --------------------------------------------------------------------------


def test_saturation_lowers_the_inductance_as_current_grows() -> None:
    """The defining behaviour: the slope falls where the map bends."""

    isat = 18.0
    spec = _machine(
        lambda i, q: LD * isat * math.tanh(i / isat) + PSI_F,
        lambda i, q: LQ * isat * math.tanh(q / isat),
    )
    near_zero = spec.flux_map.differential_inductances(0.0, 0.0)
    deep = spec.flux_map.differential_inductances(35.0, 35.0)
    assert near_zero[0] > deep[0] > 0.0
    assert near_zero[1] > deep[1] > 0.0


def test_the_shipped_saturating_example_produces_less_torque_than_its_linear_twin() -> None:
    """Measured: 7.67 vs 7.85 N*m at the same current operating point."""

    saturating = HybridSimulator(load_model(SATURATING)).simulate()
    linear = HybridSimulator(load_model(LINEAR_REF)).simulate()
    count = len(saturating.signals["torque(motor)"])
    tail = slice(int(count * 0.7), count)
    sat_torque = float(np.mean(saturating.signals["torque(motor)"][tail]))
    lin_torque = float(np.mean(linear.signals["torque(motor)"][tail]))
    # The controller closes on the same currents...
    assert float(np.mean(saturating.signals["s(iq_meas)"][tail])) == pytest.approx(
        float(np.mean(linear.signals["s(iq_meas)"][tail])), abs=0.05
    )
    # ...and the saturated machine makes measurably less torque for them.
    assert sat_torque < lin_torque
    assert 0.01 < (lin_torque - sat_torque) / lin_torque < 0.10


def test_cross_saturation_is_expressible() -> None:
    """The d axis may depend on i_q — impossible with two constants."""

    spec = _machine(
        lambda i, q: LD * i + PSI_F - 2e-5 * abs(q),
        lambda i, q: LQ * q,
    )
    without_q = spec.flux_map.flux(10.0, 0.0)[0]
    with_q = spec.flux_map.flux(10.0, 30.0)[0]
    assert with_q < without_q


# --------------------------------------------------------------------------
# Fail-closed
# --------------------------------------------------------------------------


def test_declaring_a_map_and_the_constants_it_encodes_is_refused() -> None:
    for extra in ("flux_linkage_wb", "d_axis_inductance_h", "stator_inductance_h"):
        with pytest.raises(ModelError, match="second, possibly disagreeing"):
            parse_machines(
                {
                    "m": {
                        "type": "pmsm", "terminals": ["a", "b", "c"], "neutral": "n",
                        "pole_pairs": 4, "stator_resistance_ohm": 0.05,
                        "inertia_kg_m2": 0.01, extra: 0.001,
                        "flux_map": {
                            "id_axis": list(AXIS), "iq_axis": list(AXIS),
                            "psi_d": [[LD * i + PSI_F for _ in AXIS] for i in AXIS],
                            "psi_q": [[LQ * q for q in AXIS] for _ in AXIS],
                        },
                    }
                }
            )


def test_a_map_whose_flux_falls_with_its_own_current_is_refused() -> None:
    """Not saturation but a broken table: it would make L indefinite.

    Caught at LOAD time, because the loader already reads the map's slopes
    at zero current to record the machine's nominal axis values — so a table
    like this never reaches a run.
    """

    with pytest.raises(ModelError, match="non-positive differential inductance"):
        _machine(lambda i, q: -LD * i + PSI_F, lambda i, q: LQ * q)


def test_axes_must_be_strictly_increasing() -> None:
    with pytest.raises(ModelError, match="strictly increasing"):
        _machine(lambda i, q: LD * i, lambda i, q: LQ * q, axis=[0.0, 5.0, 5.0])


def test_grid_shape_must_match_the_axes() -> None:
    with pytest.raises(ModelError, match="one row per id_axis"):
        parse_machines(
            {
                "m": {
                    "type": "pmsm", "terminals": ["a", "b", "c"], "neutral": "n",
                    "pole_pairs": 4, "stator_resistance_ohm": 0.05,
                    "inertia_kg_m2": 0.01,
                    "flux_map": {
                        "id_axis": [0.0, 1.0, 2.0], "iq_axis": [0.0, 1.0],
                        "psi_d": [[0.1, 0.1]], "psi_q": [[0.0, 0.0]],
                    },
                }
            }
        )


def test_park_currents_share_the_machine_convention() -> None:
    currents, angle = (10.0, -4.0, -6.0), 0.7
    from pear_sim.spec.machines import PHASE_OFFSETS

    offsets = [offset for _, offset in PHASE_OFFSETS]
    expected_d = (2.0 / 3.0) * sum(
        c * math.cos(angle - o) for c, o in zip(currents, offsets, strict=True)
    )
    assert park_currents(currents, angle)[0] == pytest.approx(expected_d, abs=1e-15)


# --------------------------------------------------------------------------
# Flux conservation across a segment boundary IN SATURATION
# --------------------------------------------------------------------------


def _map_flux_abc(spec, currents, angle):
    from pear_sim.spec.machines import inverse_park

    i_d, i_q = park_currents(tuple(currents), angle)
    return np.array(inverse_park(*spec.flux_map.flux(i_d, i_q), angle))


def test_flux_inversion_round_trips_deep_in_saturation() -> None:
    spec = load_model(SATURATING).machines[0]
    for i_d, i_q in [(0.0, 0.0), (-6.0, 12.0), (-30.0, 35.0), (25.0, -38.0)]:
        psi_d, psi_q = spec.flux_map.flux(i_d, i_q)
        found = spec.flux_map.currents_for_flux(psi_d, psi_q, (0.0, 0.0))
        assert found == pytest.approx((i_d, i_q), abs=1e-9)


def test_flux_inversion_refuses_a_flux_beyond_the_map() -> None:
    from pear_sim.errors import SimulationError

    spec = load_model(SATURATING).machines[0]
    psi_max = max(max(row) for row in spec.flux_map.psi_d)
    with pytest.raises(SimulationError, match="cannot produce psi_d"):
        spec.flux_map.currents_for_flux(psi_max + 0.5, 0.0, (0.0, 0.0))


def _boundary(model, currents, turn):
    """Run one segment boundary; return (tracked flux at the end of the old
    segment, map flux of the transformed currents, transformed currents)."""

    from pear_sim.core.circuit import Circuit
    from pear_sim.core.solver import _MachineDriver
    from pear_sim.spec.machines import inverse_park

    circuit = Circuit(model)
    driver = _MachineDriver(circuit, model)
    spec = model.machines[0]
    state = circuit.initial_state.copy()
    indices = [circuit.state_index[w] for w in spec.winding_ids]
    for index, value in zip(indices, currents, strict=True):
        state[index] = value
    driver.begin_segment(0.0, state)
    held = driver._salient_matrices[spec.id].copy()
    rotor_flux = driver._rotor_flux[spec.id]
    driver.state[spec.id]["angle_rad"] += turn
    angle = driver.state[spec.id]["angle_rad"]
    tracked = held @ np.array(currents) + np.array(inverse_park(*rotor_flux, angle))
    driver.begin_segment(1e-4, state)
    after = np.array([state[i] for i in indices])
    return tracked, _map_flux_abc(spec, after, angle), after, held


def test_a_saturated_boundary_conserves_the_map_flux() -> None:
    """Regression for the saturation runaway (motulator comparison, PR #27).

    The flux the kernel integrated up to the boundary must be the flux the
    MAP gives for the transformed currents. The old transform carried
    L_diff_old*i to L_diff_new*i, which equals the map flux only when the
    map is linear; here, deep in saturation, the same check on the old
    transform misses by far more than round-off — so this gate can fail.
    """

    model = load_model(SATURATING)
    spec = model.machines[0]
    angle0 = spec.initial_angle_rad
    from pear_sim.spec.machines import inverse_park

    currents = inverse_park(-30.0, 32.0, angle0)
    tracked, conserved, after, held = _boundary(model, currents, 0.3)
    assert np.allclose(conserved, tracked, rtol=0.0, atol=1e-11)
    assert float(sum(after)) == pytest.approx(0.0, abs=1e-12)
    # At the segment START the held representation is the true flux.
    assert np.allclose(
        held @ np.array(currents)
        + np.array(inverse_park(*_rotor_flux_at_start(model, currents), angle0)),
        _map_flux_abc(spec, currents, angle0),
        atol=1e-12,
    )

    # The defect, measured: the old L_old*i -> L_new*i transform.
    angle1 = angle0 + 0.3
    new_matrix = salient_inductance_matrix(spec, angle1, tuple(currents))
    old_way = np.linalg.solve(new_matrix, held @ np.array(currents))
    miss = np.max(np.abs(_map_flux_abc(spec, old_way, angle1) - tracked))
    assert miss > 1e-3


def _rotor_flux_at_start(model, currents):
    from pear_sim.core.circuit import Circuit
    from pear_sim.core.solver import _MachineDriver

    circuit = Circuit(model)
    driver = _MachineDriver(circuit, model)
    spec = model.machines[0]
    state = circuit.initial_state.copy()
    for w, value in zip(spec.winding_ids, currents, strict=True):
        state[circuit.state_index[w]] = value
    driver.begin_segment(0.0, state)
    return driver._rotor_flux[spec.id]


def test_a_linear_map_rotor_flux_is_the_magnet() -> None:
    """For a linear map the flux the held matrix does not carry is exactly
    the magnet, (psi_f, 0), so the flux-map boundary reduces to the salient
    transform L_new*i_new = L_old*i_old."""

    from pear_sim.spec.machines import rotor_fixed_flux

    spec = _linear_map()
    i_d, i_q = park_currents((10.0, -4.0, -6.0), 0.4)
    axes = spec.flux_map.differential_inductances(i_d, i_q)
    assert rotor_fixed_flux(spec, i_d, i_q, axes) == pytest.approx(
        (PSI_F, 0.0), abs=1e-12
    )
