# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Liquid cold-plate cooling path.

What a cold plate changes is not only a resistance but the REFERENCE
temperature: an air-cooled device sits at cabinet ambient, a liquid-cooled
one sits at the coolant inlet. Using ambient for a liquid path would
silently overstate the thermal margin, so the reference is asserted here.

The pump curve is deliberately not modelled — unlike the air path, where the
fan PQ curve and the heatsink pressure drop genuinely intersect at one
operating point. The declared flow and the recorded omission are both part
of the contract.
"""

from pathlib import Path

import pytest

from pear_sim.analysis.losses_thermal import build_losses_thermal_observation
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.device_cards import parse_loss_sections
from pear_sim.spec.model import load_model

EXAMPLE = Path(__file__).parents[1] / "examples" / "traction_inverter_liquid_cooled.yaml"

_PLATE = {
    "flow_axis_l_per_min": [2, 4, 8, 16],
    "rth_k_per_w": [0.075, 0.048, 0.033, 0.026],
    "coolant_inlet_c": 65,
    "provenance": {"source": "test plate"},
}
_CARD = {
    "kind": "switch",
    "provenance": {"source": "test"},
    "validity": {"tj_min_c": 25, "tj_max_c": 175, "current_max_a": 800},
    "conduction": {
        "current_axis_a": [0, 100],
        "temperature_axis_c": [25, 125],
        "voltage_table_v": [[0.0, 1.0], [0.0, 1.2]],
        "provenance": {"source": "test"},
    },
    "thermal": {
        "foster_stages": [{"resistance_k_per_w": 0.05, "time_constant_s": 0.01}],
        "provenance": {"source": "test"},
    },
}


def _sections(cooling: dict):
    return parse_loss_sections(
        {"c1": _CARD},
        {"cold_plates": {"plate": _PLATE}, "heatsinks": {}, "fans": {}, "tims": {}},
        {
            "ambient_c": 40,
            "tj_evaluation_c": 125,
            "devices": {"q1": {"card": "c1", "cooling": cooling}},
        },
    )


def test_declared_flow_is_required_with_a_cold_plate() -> None:
    with pytest.raises(ModelError, match="coolant_flow_l_per_min"):
        _sections({"cold_plate": "plate"})


def test_flow_without_a_plate_fails_closed() -> None:
    with pytest.raises(ModelError, match="requires a cold_plate"):
        _sections({"coolant_flow_l_per_min": 8})


def test_air_and_liquid_paths_are_mutually_exclusive() -> None:
    with pytest.raises(ModelError, match="excludes heatsink/fan"):
        parse_loss_sections(
            {"c1": _CARD},
            {
                "cold_plates": {"plate": _PLATE},
                "heatsinks": {
                    "hs": {
                        "airflow_axis_m3_per_h": [0, 100],
                        "rth_k_per_w": [0.5, 0.2],
                        "pressure_drop_pa": [0, 30],
                        "provenance": {"source": "test"},
                    }
                },
            },
            {
                "ambient_c": 40,
                "tj_evaluation_c": 125,
                "devices": {
                    "q1": {
                        "card": "c1",
                        "cooling": {
                            "cold_plate": "plate",
                            "coolant_flow_l_per_min": 8,
                            "heatsink": "hs",
                        },
                    }
                },
            },
        )


def test_plate_resistance_must_not_rise_with_flow() -> None:
    with pytest.raises(ModelError, match="non-increasing with flow"):
        parse_loss_sections(
            {"c1": _CARD},
            {
                "cold_plates": {
                    "plate": {**_PLATE, "rth_k_per_w": [0.03, 0.05, 0.06, 0.07]}
                }
            },
            {
                "ambient_c": 40,
                "tj_evaluation_c": 125,
                "devices": {
                    "q1": {
                        "card": "c1",
                        "cooling": {
                            "cold_plate": "plate",
                            "coolant_flow_l_per_min": 8,
                        },
                    }
                },
            },
        )


def test_example_uses_the_coolant_inlet_as_its_reference() -> None:
    model = load_model(EXAMPLE)
    result = HybridSimulator(model).simulate()
    observation = build_losses_thermal_observation(result)

    device = next(
        entry for entry in observation["devices"] if entry["component_id"] == "qah"
    )
    cooling = device["cooling"]
    plate = cooling["cold_plate"]

    # 8 l/min is an exact table point, so no interpolation and no clamping.
    assert plate["coolant_flow_l_per_min"] == pytest.approx(8.0)
    assert plate["rth_k_per_w"] == pytest.approx(0.033)
    assert plate["flow_below_axis"] == 0 and plate["flow_above_axis"] == 0

    # The reference is the coolant inlet (65 C), NOT the 45 C cabinet ambient.
    assert cooling["reference_temperature_source"] == "coolant_inlet"
    assert cooling["reference_temperature_c"] == pytest.approx(65.0)
    assert model.loss_analysis.ambient_c == pytest.approx(45.0)

    # Case temperature follows from that reference and the declared path.
    path_rth = 0.008 + 0.033  # TIM + plate
    totals = device["switching"]["totals_j"]
    window = float(result.time[-1]) - float(result.time[0])
    mean_power = device["conduction"]["mean_power_w"] + sum(totals.values()) / window
    assert device["thermal"]["case_c_static"] == pytest.approx(
        65.0 + path_rth * mean_power, rel=1e-6
    )

    # Both liquid-specific omissions are declared, not assumed away.
    omissions = " ".join(cooling["omissions"])
    assert "pump and plumbing curve not modelled" in omissions
    assert "coolant temperature rise" in omissions


def test_flow_below_the_table_is_clamped_and_counted() -> None:
    spec = _sections({"cold_plate": "plate", "coolant_flow_l_per_min": 1.0})
    assert spec.cold_plates["plate"].flow_axis_l_per_min[0] == pytest.approx(2.0)
    binding = spec.devices[0]
    assert binding.cooling.coolant_flow_l_per_min == pytest.approx(1.0)


@pytest.mark.parametrize(("first_airflow", "accepted"), [(0, True), (50, False)])
def test_a_fanless_heatsink_needs_its_zero_airflow_point(
    first_airflow: float, accepted: bool
) -> None:
    """Without a fan the estimate reads the heatsink's natural-convection point. A
    table starting at forced air must not be read as natural convection."""

    def sections():
        return parse_loss_sections(
            {"c1": _CARD},
            {
                "heatsinks": {
                    "hs": {
                        "airflow_axis_m3_per_h": [first_airflow, 100],
                        "rth_k_per_w": [0.5, 0.2],
                        "pressure_drop_pa": [0, 30],
                        "provenance": {"source": "test"},
                    }
                }
            },
            {
                "ambient_c": 40,
                "tj_evaluation_c": 125,
                "devices": {"q1": {"card": "c1", "cooling": {"heatsink": "hs"}}},
            },
        )

    if accepted:
        sections()
    else:
        with pytest.raises(ModelError, match="natural-convection point"):
            sections()
