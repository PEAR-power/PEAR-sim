# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Imposed shaft speed: the dynamometer case.

With `driven_speed` the mechanical loop is NOT integrated. The shaft is
turned from outside and the TORQUE is the measurement. Two reasons this is
not a convenience:

* a synchronous machine on a fixed-frequency supply cannot start from
  standstill, so a run with free mechanics cannot ask a start-up question
  at all;
* our own mechanical loop is not independently verified. Imposing
  the speed removes that loop from the answer, which is why a torque test
  stays meaningful while that loop is unverified.

The tolerances below are DERIVED from the holding scheme, not tuned until
green. The speed is read at each segment boundary and held across the
segment, so both errors are consequences of the segment length.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.machines import DrivenSpeed, parse_machines
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]
TORQUE_TEST = ROOT / "examples" / "pmsm_torque_test.yaml"

RAMP_FINAL = 157.0796
RAMP_STOP = 0.03
OUTPUT_STEP = 1e-5
POLE_PAIRS = 4


@pytest.fixture(scope="module")
def torque_test_result():
    return HybridSimulator(load_model(TORQUE_TEST)).simulate()


def _profile(time: np.ndarray) -> np.ndarray:
    return RAMP_FINAL * np.minimum(1.0, time / RAMP_STOP)


def test_speed_follows_the_declared_profile_within_one_held_segment(
    torque_test_result,
) -> None:
    # The speed is read at a segment boundary and held. The worst a held
    # value can be behind a linear ramp is the ramp's own drift across one
    # segment, and segments are capped at the output step.
    bound = (RAMP_FINAL / RAMP_STOP) * OUTPUT_STEP
    time = torque_test_result.time
    speed = torque_test_result.signals["speed_mech(motor)"]
    error = np.abs(speed - _profile(time))
    assert np.max(error) <= bound * 1.01
    # ... and it is genuinely a lag, not noise: the held value never LEADS
    # the ramp.
    assert np.all(speed <= _profile(time) + 1e-12)


def test_angle_is_the_integral_of_the_imposed_speed(torque_test_result) -> None:
    # Holding the speed makes the angle a LEFT Riemann sum of the ramp, which
    # undershoots the exact integral by half the drift per step, summed:
    #   0.5 * (dOmega/dt) * step * duration * pole_pairs.
    angle = torque_test_result.signals["angle_elec(motor)"]
    duration = float(torque_test_result.time[-1])
    exact = POLE_PAIRS * (
        0.5 * RAMP_FINAL * RAMP_STOP + RAMP_FINAL * (duration - RAMP_STOP)
    )
    predicted_undershoot = (
        0.5 * (RAMP_FINAL / RAMP_STOP) * OUTPUT_STEP * RAMP_STOP * POLE_PAIRS
    )
    shortfall = exact - float(angle[-1])
    assert shortfall == pytest.approx(predicted_undershoot, rel=0.2)
    assert angle[0] == 0.0


def test_torque_does_not_feed_back_into_the_speed(torque_test_result) -> None:
    """The defining property: the shaft does not care what the machine does."""

    data = yaml.safe_load(TORQUE_TEST.read_text(encoding="utf-8"))
    # Quadruple the commanded voltage: a completely different torque history.
    data["parameters"]["mi"] = "20 %"
    variant = ROOT / "tests" / "_driven_variant.yaml"
    try:
        variant.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        louder = HybridSimulator(load_model(variant)).simulate()
    finally:
        variant.unlink(missing_ok=True)
    baseline_torque = torque_test_result.signals["torque(motor)"]
    louder_torque = louder.signals["torque(motor)"]
    # The torque really did change: 27 N*m of it.
    assert np.max(np.abs(louder_torque - baseline_torque)) > 1.0
    # The speed did not follow. NOT bit-identical, and the reason is worth
    # stating precisely rather than hiding behind a loose tolerance: the
    # profile is READ at segment boundaries, and boundaries include natural
    # diode commutations, whose timing does depend on the currents. So a
    # different torque history reshuffles WHEN the profile is sampled,
    # never WHAT it returns. The residue is therefore bounded by the
    # profile's drift across one segment — the same bound as the tracking
    # test — while the torque moved by 27 N*m.
    bound = (RAMP_FINAL / RAMP_STOP) * OUTPUT_STEP
    speed_change = np.abs(
        louder.signals["speed_mech(motor)"]
        - torque_test_result.signals["speed_mech(motor)"]
    )
    assert np.max(speed_change) <= bound
    assert np.max(speed_change) / RAMP_FINAL < 1e-3


def test_fields_with_no_effect_are_refused_not_ignored() -> None:
    base = {
        "type": "pmsm",
        "terminals": ["a", "b", "c"],
        "neutral": "n",
        "pole_pairs": 2,
        "stator_resistance_ohm": 0.1,
        "stator_inductance_h": 1e-3,
        "flux_linkage_wb": 0.1,
        "driven_speed": {"initial_rad_s": 100.0},
    }
    for field, value in (
        ("inertia_kg_m2", 0.01),
        ("viscous_friction_nm_s", 0.05),
        ("load_torque_nm", 1.0),
        ("initial_speed_rad_s", 10.0),
    ):
        with pytest.raises(ModelError, match="would have no effect"):
            parse_machines({"m": {**base, field: value}})


def test_half_a_ramp_is_refused() -> None:
    base = {
        "type": "pmsm",
        "terminals": ["a", "b", "c"],
        "neutral": "n",
        "pole_pairs": 2,
        "stator_resistance_ohm": 0.1,
        "stator_inductance_h": 1e-3,
        "flux_linkage_wb": 0.1,
    }
    for profile in (
        {"initial_rad_s": 0.0, "final_rad_s": 100.0},
        {"initial_rad_s": 0.0, "ramp_stop_s": 0.01},
    ):
        with pytest.raises(ModelError, match="both final_rad_s and ramp_stop_s"):
            parse_machines({"m": {**base, "driven_speed": profile}})
    with pytest.raises(ModelError, match="meaningless without a ramp"):
        parse_machines(
            {
                "m": {
                    **base,
                    "driven_speed": {"initial_rad_s": 5.0, "ramp_start_s": 0.001},
                }
            }
        )
    with pytest.raises(ModelError, match="later than ramp_start_s"):
        parse_machines(
            {
                "m": {
                    **base,
                    "driven_speed": {
                        "initial_rad_s": 0.0,
                        "final_rad_s": 10.0,
                        "ramp_start_s": 0.02,
                        "ramp_stop_s": 0.01,
                    },
                }
            }
        )


def test_profile_evaluation_is_clamped_at_both_ends() -> None:
    ramp = DrivenSpeed(
        initial_rad_s=10.0, final_rad_s=110.0, ramp_start_s=1.0, ramp_stop_s=2.0
    )
    assert ramp.speed_at(0.0) == 10.0
    assert ramp.speed_at(1.0) == 10.0
    assert ramp.speed_at(1.5) == pytest.approx(60.0)
    assert ramp.speed_at(2.0) == 110.0
    assert ramp.speed_at(99.0) == 110.0
    constant = DrivenSpeed(initial_rad_s=7.0)
    assert constant.speed_at(0.0) == 7.0
    assert constant.speed_at(1e6) == 7.0


def test_free_mechanics_remains_the_default() -> None:
    # The shipped free-mechanics drive must be untouched by this feature.
    model = load_model(ROOT / "examples" / "pmsm_inverter_drive.yaml")
    assert model.machines[0].driven_speed is None
