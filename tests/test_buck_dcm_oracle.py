# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pear_sim.mode_coverage import build_mode_coverage
from pear_sim.model import Model, load_model
from pear_sim.solver import HybridSimulator, SimulationResult
from tests.reference.buck_dcm_oracle import (
    ORACLE_KIND,
    ORACLE_SCHEMA,
    BuckDcmOracleResult,
    OracleModelError,
    evaluate_buck_dcm,
    textbook_dcm_conversion_ratio,
)


ROOT = Path(__file__).parents[1]
BUCK_DCM = ROOT / "examples" / "buck_dcm.yaml"
BOOST = ROOT / "examples" / "boost.yaml"


@pytest.fixture(scope="module")
def oracle_case() -> tuple[Model, BuckDcmOracleResult]:
    model = load_model(BUCK_DCM)
    return model, evaluate_buck_dcm(model)


@pytest.fixture(scope="module")
def solver_vs_oracle(
    oracle_case: tuple[Model, BuckDcmOracleResult],
) -> tuple[SimulationResult, dict[str, object], BuckDcmOracleResult]:
    model, oracle = oracle_case
    result = HybridSimulator(model).simulate()
    return result, build_mode_coverage(result), oracle


def test_fixture_is_canonical_light_load_buck_with_zero_initial_conditions(
    oracle_case: tuple[Model, BuckDcmOracleResult],
):
    model, result = oracle_case
    parameters = result.parameters_si

    assert model.name == "buck_dcm_zero_ic"
    assert parameters == pytest.approx(
        {
            "vin_v": 100.0,
            "duty": 0.2,
            "switching_frequency_hz": 20_000.0,
            "period_s": 50e-6,
            "inductance_h": 100e-6,
            "capacitance_f": 1e-6,
            "load_ohm": 1000.0,
        },
        rel=0.0,
        abs=1e-15,
    )
    assert result.cycles[0].start_state.inductor_current_a == 0.0
    assert result.cycles[0].start_state.output_voltage_v == 0.0
    assert len(result.cycles) == 200
    assert all(cycle.diode_zero_s is not None for cycle in result.cycles)


def test_first_zero_current_event_and_boundary_states_are_golden(
    oracle_case: tuple[Model, BuckDcmOracleResult],
):
    _model, result = oracle_case
    first = result.cycles[0]

    assert first.switch_off_s == pytest.approx(10e-6, rel=0.0, abs=1e-15)
    assert first.diode_zero_s == pytest.approx(
        20.762398257774295e-6, rel=0.0, abs=5e-13
    )
    assert first.switch_off_state.inductor_current_a == pytest.approx(
        8.418598326775745, rel=0.0, abs=1e-10
    )
    assert first.switch_off_state.output_voltage_v == pytest.approx(
        45.81956077915752, rel=0.0, abs=1e-10
    )
    assert first.diode_zero_state is not None
    assert first.diode_zero_state.inductor_current_a == 0.0
    assert first.diode_zero_state.output_voltage_v == pytest.approx(
        95.13259319893591, rel=0.0, abs=1e-10
    )
    assert first.end_state.inductor_current_a == 0.0
    assert first.conduction_fraction == pytest.approx(
        0.4152479651554859, rel=0.0, abs=1e-11
    )


def test_steady_dcm_mean_voltage_diode_fraction_and_zero_current_offset(
    oracle_case: tuple[Model, BuckDcmOracleResult],
):
    _model, result = oracle_case
    last = result.last_cycle
    mean_vout = result.tail_mean_output_voltage(20)
    conversion_ratio = mean_vout / result.parameters_si["vin_v"]

    assert mean_vout == pytest.approx(92.65159667014042, rel=0.0, abs=1e-9)
    assert conversion_ratio == pytest.approx(0.9265159667014042, rel=0.0, abs=1e-11)
    assert last.diode_conduction_fraction == pytest.approx(
        0.01695354940971154, rel=0.0, abs=1e-11
    )
    assert last.conduction_fraction == pytest.approx(
        0.21695354940971154, rel=0.0, abs=1e-11
    )
    assert last.idle_fraction == pytest.approx(0.7830464505902885, rel=0.0, abs=1e-11)
    assert last.diode_zero_state is not None
    assert last.diode_zero_state.inductor_current_a == 0.0
    assert last.start_state.inductor_current_a == 0.0
    assert last.end_state.inductor_current_a == 0.0
    assert last.end_state.output_voltage_v == pytest.approx(
        last.start_state.output_voltage_v, rel=0.0, abs=1e-11
    )

    textbook_ratio = textbook_dcm_conversion_ratio(result.parameters_si)
    assert textbook_ratio == pytest.approx(0.9160797830996159, rel=0.0, abs=1e-12)
    assert abs(conversion_ratio - textbook_ratio) < 0.02


def test_oracle_record_is_machine_readable_and_disclaims_validation(
    oracle_case: tuple[Model, BuckDcmOracleResult],
):
    _model, result = oracle_case

    first = result.as_dict()
    second = result.as_dict()

    assert first == second
    assert first["schema"] == ORACLE_SCHEMA
    assert first["kind"] == ORACLE_KIND
    assert first["claim"]["classification"] == "verification"
    assert first["claim"]["validation_claim"] is False
    assert first["method"]["pear_sim_kernel_imported"] is False
    assert first["first_diode_zero_s"] == result.first_diode_zero.diode_zero_s
    assert math.isfinite(first["tail_mean_output_voltage_v"])
    assert json.dumps(first, allow_nan=False, sort_keys=True) == json.dumps(
        second, allow_nan=False, sort_keys=True
    )


def test_oracle_rejects_noncanonical_converter():
    with pytest.raises(OracleModelError):
        evaluate_buck_dcm(load_model(BOOST), cycle_count=1)


def test_oracle_rejects_reverse_current_inside_the_commanded_on_interval():
    model = load_model(BUCK_DCM)
    components = tuple(
        replace(component, value=5e-6)
        if component.id == "l1"
        else replace(component, value=2e-8)
        if component.id == "c1"
        else component
        for component in model.components
    )

    with pytest.raises(OracleModelError, match="on interval contains reverse inductor current"):
        evaluate_buck_dcm(replace(model, components=components))


def test_reference_module_has_no_kernel_or_solver_dependency():
    source = (Path(__file__).parent / "reference" / "buck_dcm_oracle.py").read_text(
        encoding="utf-8"
    )
    assert "pear_sim.circuit" not in source
    assert "pear_sim.solver" not in source
    assert "HybridSimulator" not in source


def test_solver_natural_turnoffs_match_every_oracle_root(
    solver_vs_oracle: tuple[SimulationResult, dict[str, object], BuckDcmOracleResult],
):
    result, _coverage, oracle = solver_vs_oracle
    observed = [
        event
        for event in result.events
        if event.kind == "natural_commutation"
        and event.target == "d1"
        and event.previous is True
        and event.current is False
    ]
    expected = [cycle.diode_zero_s for cycle in oracle.cycles]

    assert len(observed) == len(expected) == 200
    assert result.stats["natural_events"] == 200
    errors = [
        abs(event.time - expected_time)
        for event, expected_time in zip(observed, expected, strict=True)
        if expected_time is not None
    ]
    assert len(errors) == 200
    assert errors[0] < 1e-12
    assert max(errors) < 2e-12
    assert observed[0].time == pytest.approx(
        oracle.first_diode_zero.diode_zero_s, rel=0.0, abs=1e-12
    )


def test_solver_tail_mean_and_mode_fractions_match_oracle(
    solver_vs_oracle: tuple[SimulationResult, dict[str, object], BuckDcmOracleResult],
):
    result, coverage, oracle = solver_vs_oracle
    metrics = result.metrics()
    fractions = coverage["summary"]["fraction_by_mode"]
    last = oracle.last_cycle

    expected_tail_mean = oracle.tail_mean_output_voltage(len(oracle.cycles) // 2)
    assert metrics["v(out)"]["mean_last_half"] == pytest.approx(
        expected_tail_mean, rel=0.0, abs=5e-8
    )
    assert fractions["on"] == pytest.approx(oracle.parameters_si["duty"], abs=1e-10)
    assert fractions["freewheel"] == pytest.approx(
        last.diode_conduction_fraction, rel=0.0, abs=1e-10
    )
    assert fractions["idle"] == pytest.approx(last.idle_fraction, rel=0.0, abs=1e-10)
    assert fractions["unclassified"] == 0.0
    assert fractions["on"] + fractions["freewheel"] == pytest.approx(
        last.conduction_fraction, rel=0.0, abs=1e-10
    )
    assert coverage["diode_conduction"]["fraction"] == pytest.approx(
        last.diode_conduction_fraction, rel=0.0, abs=1e-10
    )
    assert coverage["summary"]["natural_commutation"] == {
        "exercised": True,
        "event_count": 100,
        "diode_turn_off_event_count": 100,
    }
    assert result.physics_audit["last_half"]["balance_within_tolerance"] is True
    assert result.physics_audit["last_half"]["balance_residual_relative"] < 1e-8


def test_solver_observes_zero_clamp_without_reverse_current(
    solver_vs_oracle: tuple[SimulationResult, dict[str, object], BuckDcmOracleResult],
):
    result, coverage, _oracle = solver_vs_oracle
    inductor_observation = coverage["inductor_current"]

    assert inductor_observation["minimum_a"] == 0.0
    assert inductor_observation["maximum_forbidden_reverse_current_a"] == 0.0
    assert float(np.min(result.signals["i(l1)"])) == 0.0
    assert float(np.min(result.signals["i(d1)"])) == 0.0
    assert result.stats["state_constraint_events"] == 200
    assert result.stats["constrained_state_projections"] == 200

    idle_intervals = [
        interval for interval in result.mode_intervals if interval["constrained_inductors"]
    ]
    assert len(idle_intervals) == 200
    assert all(interval["constrained_inductors"] == ["l1"] for interval in idle_intervals)
    assert all(interval["state_start"]["l1"] == 0.0 for interval in idle_intervals)
    assert all(interval["state_end"]["l1"] == 0.0 for interval in idle_intervals)
