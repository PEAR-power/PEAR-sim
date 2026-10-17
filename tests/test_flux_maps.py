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
