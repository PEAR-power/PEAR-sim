# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Self-consistent junction temperature (loss_analysis.tj_mode) against algebra.

The oracle solves the loop by a DIFFERENT method than the code: the code
iterates trajectories (with an Aitken step), the oracle is one line of
algebra. A constant current I through a device whose card is
v_on = I*R0*(1 + alpha*(T - T0)) — bilinear in (i, T), so the table
reproduces it exactly — with a cooling path of total resistance Rth settles
at Tj = Ta + Rth*I*v_on(Tj), whose solution is

    Tj* = (Ta + Rth*I^2*R0*(1 - alpha*T0)) / (1 - g),   g = Rth*I^2*R0*alpha.

g is the electrothermal loop gain; g >= 1 is thermal runaway and has no
operating point. The gate has to ring both ways: g = 0.3 and 0.9 converge to
Tj* within 1e-6 K, g = 1.1 returns a status and NO number, a negative
temperature coefficient converges too, and a loop that ignores Tj in the
table misses Tj* by orders of magnitude.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from pear_sim.analysis import losses_thermal
from pear_sim.analysis.gradient.losses import LossGradientError, evaluate_losses
from pear_sim.analysis.losses_thermal import build_losses_thermal_observation
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]
BUCK = ROOT / "examples" / "buck_with_loss_cards.yaml"

CURRENT_A = 20.0
R0_OHM = 0.05
T0_C = 25.0
AMBIENT_C = 40.0
FOSTER_R = 0.5  # the rest of Rth is a static TIM
BUDGET_C = 1e-6
TEMPERATURE_AXIS = (-200.0, 5000.0)


def _loop_gain(rth: float, alpha: float) -> float:
    return rth * CURRENT_A**2 * R0_OHM * alpha


def _closed_form_tj(rth: float, alpha: float) -> float:
    heating = rth * CURRENT_A**2 * R0_OHM
    return (AMBIENT_C + heating * (1.0 - alpha * T0_C)) / (1.0 - _loop_gain(rth, alpha))


def _v_on(current: float, temperature: float, alpha: float) -> float:
    return current * R0_OHM * (1.0 + alpha * (temperature - T0_C))


def _model(tmp_path: Path, *, rth: float, alpha: float, tj_mode: str = "self_consistent",
           temperature_axis: tuple[float, ...] = TEMPERATURE_AXIS):
    currents = (0.0, 40.0)
    table = [[_v_on(i, t, alpha) for i in currents] for t in temperature_axis]
    document = {
        "schema": "pear-sim/v0.1",
        "name": "self_consistent_tj_oracle",
        "circuit": {
            "ground": "gnd",
            "components": [
                {"id": "isrc", "type": "current_source", "p": "gnd", "n": "mid",
                 "value": f"{CURRENT_A} A"},
                {"id": "d1", "type": "diode", "p": "mid", "n": "out", "initial": "on"},
                {"id": "rload", "type": "resistor", "p": "out", "n": "gnd", "value": "1 ohm"},
                {"id": "cpar", "type": "capacitor", "p": "out", "n": "gnd", "value": "1 uF",
                 "initial_voltage": f"{CURRENT_A} V"},
            ],
        },
        "device_cards": {
            "card": {
                "kind": "diode",
                "provenance": {"source": "synthetic"},
                "validity": {"tj_min_c": TEMPERATURE_AXIS[0], "tj_max_c": TEMPERATURE_AXIS[1],
                             "current_max_a": 40},
                "conduction": {
                    "current_axis_a": list(currents),
                    "temperature_axis_c": list(temperature_axis),
                    "voltage_table_v": table,
                    "provenance": {"source": "synthetic"},
                },
                "thermal": {
                    "foster_stages": [{"resistance_k_per_w": FOSTER_R, "time_constant_s": 0.002}],
                    "provenance": {"source": "synthetic"},
                },
            }
        },
        "cooling": {
            "tims": {"pad": {"rth_k_per_w": rth - FOSTER_R, "provenance": {"source": "synthetic"}}},
        },
        "loss_analysis": {
            "ambient_c": AMBIENT_C,
            "tj_evaluation_c": 75,
            "tj_mode": tj_mode,
            "devices": {"d1": {"card": "card", "cooling": {"tim": "pad"}}},
        },
        "analysis": {
            "transient": {"start": "0 s", "stop": "5 ms", "output_step": "10 us",
                          "solver": "dopri5", "rtol": 1e-9, "atol": 1e-11},
        },
        "outputs": ["i(d1)", "v(out)"],
    }
    path = tmp_path / f"oracle_{tj_mode}_{rth}_{alpha}.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_model(path)


def _observe(tmp_path: Path, **kw) -> dict:
    return build_losses_thermal_observation(HybridSimulator(_model(tmp_path, **kw)).simulate())


# rth chosen so that g = rth * 20 W/K-of-alpha hits the target gain.
CONVERGING = [
    pytest.param(3.75, 0.004, TEMPERATURE_AXIS, id="gain_0.3"),
    pytest.param(11.25, 0.004, TEMPERATURE_AXIS, id="gain_0.9"),
    # v_on must stay positive over the table: 1 - 0.004*(250 - 25) = 0.1.
    pytest.param(11.25, -0.004, (-200.0, 250.0), id="gain_-0.9_negative_tempco"),
]


@pytest.mark.parametrize(("rth", "alpha", "axis"), CONVERGING)
def test_the_loop_converges_to_the_closed_form(
    tmp_path: Path, rth: float, alpha: float, axis: tuple[float, float]
) -> None:
    observation = _observe(tmp_path, rth=rth, alpha=alpha, temperature_axis=axis)
    (device,) = observation["devices"]
    expected = _closed_form_tj(rth, alpha)
    loop = device["electrothermal"]
    assert loop["status"] == "converged"
    assert observation["electrothermal"]["status"] == "converged"
    assert loop["loop_gain"] == pytest.approx(_loop_gain(rth, alpha), rel=1e-9)
    thermal = device["thermal"]
    for key in ("tj_max_c", "tj_min_c", "tj_final_c"):
        assert abs(thermal[key] - expected) < BUDGET_C, (key, thermal[key], expected)
    assert thermal["tj_evaluation_gap_c"] < BUDGET_C
    assert thermal["balance_within_tolerance"]
    power = CURRENT_A * _v_on(CURRENT_A, expected, alpha)
    assert device["conduction"]["mean_power_w"] == pytest.approx(power, rel=1e-9)
    assert observation["total_mean_power_w"] == pytest.approx(power, rel=1e-9)
    assert device["conduction"]["tj_evaluation_c"] == "self_consistent_trajectory"


def test_runaway_returns_a_status_and_no_number(tmp_path: Path) -> None:
    rth, alpha = 13.75, 0.004  # g = 1.1
    observation = _observe(tmp_path, rth=rth, alpha=alpha)
    (device,) = observation["devices"]
    loop = device["electrothermal"]
    assert loop["status"] == "thermal_runaway"
    assert loop["loop_gain"] == pytest.approx(1.1, rel=1e-9)
    assert observation["electrothermal"]["status"] == "thermal_runaway"
    # Nothing that reads as a loss or a temperature result.
    assert "total_mean_power_w" not in observation
    assert not {"conduction", "switching", "thermal", "mean_power_w", "tj_envelope"} & set(device)
    assert any("thermal_runaway" in warning for warning in observation["warnings"])


def test_a_runaway_cut_off_by_a_short_table_is_still_not_a_number(tmp_path: Path) -> None:
    """Clamped at the top of the temperature axis, v_on stops growing and the
    loop finds a 'fixed point' that belongs to the clamp, not to the device."""

    observation = _observe(tmp_path, rth=13.75, alpha=0.004, temperature_axis=(-200.0, 300.0))
    (device,) = observation["devices"]
    assert device["electrothermal"]["status"] == "beyond_card_temperature_axis"
    assert device["electrothermal"]["temperature_clamped_lookups"] > 0
    assert "total_mean_power_w" not in observation
    assert "thermal" not in device and "conduction" not in device


def test_a_loop_that_ignores_tj_in_the_table_misses_the_closed_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rth, alpha = 3.75, 0.004
    original = losses_thermal._conduction_voltage_trajectory

    def frozen(card, current, junction_c):
        return original(card, current, np.full_like(junction_c, 75.0))

    monkeypatch.setattr(losses_thermal, "_conduction_voltage_trajectory", frozen)
    (device,) = _observe(tmp_path, rth=rth, alpha=alpha)["devices"]
    miss = abs(device["thermal"]["tj_max_c"] - _closed_form_tj(rth, alpha))
    assert miss > 1e6 * BUDGET_C, miss


def test_a_temperature_independent_card_reproduces_the_declared_estimate(tmp_path: Path) -> None:
    """With one temperature row there is nothing to iterate: one pass, and the
    same numbers as the declared path (which takes a different code route)."""

    kwargs = {"rth": 3.75, "alpha": 0.004, "temperature_axis": (25.0,)}
    declared = _observe(tmp_path, tj_mode="declared", **kwargs)
    consistent = _observe(tmp_path, **kwargs)
    (d_dev,), (c_dev,) = declared["devices"], consistent["devices"]
    assert c_dev["electrothermal"]["evaluations"] == 2
    assert c_dev["conduction"]["mean_power_w"] == pytest.approx(
        d_dev["conduction"]["mean_power_w"], rel=1e-12
    )
    for key in ("tj_max_c", "tj_min_c", "tj_final_c"):
        assert c_dev["thermal"][key] == pytest.approx(d_dev["thermal"][key], rel=1e-12)
    assert "electrothermal" not in declared and "electrothermal" not in d_dev


def test_the_buck_example_converges_and_moves_away_from_the_declared_tj(tmp_path: Path) -> None:
    document = yaml.safe_load(BUCK.read_text(encoding="utf-8"))
    document["loss_analysis"]["tj_mode"] = "self_consistent"
    path = tmp_path / "buck_sc.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    consistent = build_losses_thermal_observation(HybridSimulator(load_model(path)).simulate())
    declared = build_losses_thermal_observation(HybridSimulator(load_model(BUCK)).simulate())
    assert consistent["electrothermal"]["status"] == "converged"
    by_id = {device["component_id"]: device for device in declared["devices"]}
    for c_dev in consistent["devices"]:
        d_dev = by_id[c_dev["component_id"]]
        loop = c_dev["electrothermal"]
        assert loop["status"] == "converged"
        assert loop["estimated_error_c"] < 0.01
        # The declared 125 C is far from where these devices run, so the
        # conduction loss has to move; switching stays at the card condition.
        assert d_dev["thermal"]["tj_evaluation_gap_c"] > 20.0
        assert c_dev["conduction"]["mean_power_w"] != pytest.approx(
            d_dev["conduction"]["mean_power_w"], rel=1e-3
        )
        assert c_dev["switching"]["totals_j"] == d_dev["switching"]["totals_j"]
    assert any("used unchanged" in warning for warning in consistent["warnings"])


def test_an_unknown_tj_mode_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="tj_mode"):
        _model(tmp_path, rth=3.75, alpha=0.004, tj_mode="iterate")


def test_the_gradient_engine_refuses_the_self_consistent_mode(tmp_path: Path) -> None:
    model = _model(tmp_path, rth=3.75, alpha=0.004)
    empty = np.array([])
    with pytest.raises(LossGradientError, match="tj_mode='self_consistent'"):
        evaluate_losses(model, None, (), (), (), (), 1.0, empty, empty, None)
