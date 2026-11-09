# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np
import pytest

from pear_sim.benchmarks.buck_boundary import (
    BCM_COINCIDENCE_FRACTION_TOLERANCE,
    BUCK_BOUNDARY_ORACLE_SCHEMA,
    BUCK_BOUNDARY_RESULT_SCHEMA,
    DEFAULT_LOAD_RATIOS,
    BuckBoundaryBenchmarkError,
    BuckBoundaryObservation,
    BuckBoundaryParameters,
    compare_buck_boundary_point,
    compare_buck_boundary_sweep,
    extract_buck_boundary_parameters,
    find_bcm_load,
    observe_simulation_result,
    run_buck_boundary_sweep,
    solve_periodic_buck,
    _active_matrix,
    _minimum_current,
)
from pear_sim.model import Model, load_model
from pear_sim.solver import HybridSimulator


ROOT = Path(__file__).parents[1]
BUCK_DCM = ROOT / "examples" / "buck_dcm.yaml"
BOOST = ROOT / "examples" / "boost.yaml"
EXPECTED_BCM_LOAD_OHM = 6.089232079756019


@pytest.fixture(scope="module")
def base_model() -> Model:
    return load_model(BUCK_DCM)


@pytest.fixture(scope="module")
def boundary_sweep(base_model: Model):
    return run_buck_boundary_sweep(extract_buck_boundary_parameters(base_model))


def _periodic_model(base: Model, point, parameters: BuckBoundaryParameters) -> Model:
    components = []
    for component in base.components:
        if component.type == "resistor":
            component = replace(component, value=point.load_ohm)
        elif component.type == "inductor":
            component = replace(
                component,
                initial_state=point.start_state.inductor_current_a,
                initial_state_explicit=True,
            )
        elif component.type == "capacitor":
            component = replace(
                component,
                initial_state=point.start_state.output_voltage_v,
                initial_state_explicit=True,
            )
        components.append(component)
    period = parameters.period_s
    transient = replace(
        base.transient,
        start=0.0,
        stop=period,
        output_step=period / 200.0,
        max_step=period / 100.0,
        rtol=1e-10,
        atol=1e-12,
    )
    return replace(base, components=tuple(components), transient=transient)


def test_exact_boundary_and_default_regime_order_are_independently_reproducible(
    boundary_sweep,
):
    assert boundary_sweep.load_ratios == DEFAULT_LOAD_RATIOS
    assert boundary_sweep.bcm_load_ohm == pytest.approx(EXPECTED_BCM_LOAD_OHM, rel=0.0, abs=3e-13)
    assert [point.regime for point in boundary_sweep.points] == [
        "CCM",
        "CCM",
        "CCM",
        "CCM",
        "BCM",
        "DCM",
        "DCM",
        "DCM",
        "DCM",
    ]
    assert [point.load_ratio_to_bcm for point in boundary_sweep.points] == list(DEFAULT_LOAD_RATIOS)


def test_boundary_is_not_the_small_ripple_textbook_approximation(base_model: Model):
    parameters = extract_buck_boundary_parameters(base_model)
    textbook = (
        2.0 * parameters.inductance_h * parameters.switching_frequency_hz / (1.0 - parameters.duty)
    )

    assert textbook == pytest.approx(5.0, rel=0.0, abs=1e-14)
    assert find_bcm_load(parameters) == pytest.approx(EXPECTED_BCM_LOAD_OHM, abs=3e-13)
    assert abs(find_bcm_load(parameters) - textbook) > 1.0


def test_periodic_oracle_states_modes_and_health_are_self_consistent(boundary_sweep):
    current_base = boundary_sweep.parameters.current_base_a
    for point in boundary_sweep.points:
        assert point.periodicity_residual_normalized <= 1e-10
        assert sum(point.fraction_by_mode.values()) == pytest.approx(1.0, abs=1e-14)
        assert point.fraction_by_mode["unclassified"] == 0.0
        assert point.minimum_inductor_current_a >= -1e-12 * current_base
        assert {gate["status"] for gate in point.oracle_gates} == {"pass"}

        if point.regime == "CCM":
            assert point.start_state.inductor_current_a > 0.0
            assert point.diode_zero_s is None
            assert point.fraction_by_mode["idle"] == 0.0
        elif point.regime == "BCM":
            assert point.start_state.inductor_current_a == 0.0
            assert point.end_state.inductor_current_a == 0.0
            assert point.diode_zero_s == pytest.approx(
                boundary_sweep.parameters.period_s, abs=1e-15
            )
            assert point.fraction_by_mode["idle"] <= BCM_COINCIDENCE_FRACTION_TOLERANCE
        else:
            assert point.start_state.inductor_current_a == 0.0
            assert point.end_state.inductor_current_a == 0.0
            assert point.diode_zero_s is not None
            assert point.diode_zero_s < boundary_sweep.parameters.period_s
            assert point.fraction_by_mode["idle"] > BCM_COINCIDENCE_FRACTION_TOLERANCE


def test_oracle_summary_is_versioned_deterministic_and_truthful(boundary_sweep):
    first = boundary_sweep.as_dict()
    second = run_buck_boundary_sweep(boundary_sweep.parameters).as_dict()

    assert first == second
    assert first["schema"] == BUCK_BOUNDARY_ORACLE_SCHEMA
    assert first["claim"]["classification"] == "verification"
    assert first["claim"]["validation_claim"] is False
    assert first["method"]["pear_sim_circuit_or_solver_imported"] is False
    assert first["oracle_health"] == {
        "status": "pass",
        "point_count": 9,
        "failed_gate_count": 0,
    }
    assert len(first["comparison_gate_contract"]) == 11
    assert json.dumps(first, allow_nan=False, sort_keys=True) == json.dumps(
        second, allow_nan=False, sort_keys=True
    )


def test_oracle_module_has_no_kernel_solver_or_event_manager_import():
    source_path = ROOT / "src" / "pear_sim" / "benchmarks" / "buck_boundary.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "pear_sim.circuit" not in imported_modules
    assert "pear_sim.solver" not in imported_modules
    assert "pear_sim.mode_coverage" not in imported_modules
    assert (
        imported_modules
        & {
            "pear_sim.circuit",
            "pear_sim.solver",
            "pear_sim.mode_coverage",
        }
        == set()
    )


def test_parameter_and_topology_recognition_fail_closed(base_model: Model):
    with pytest.raises(BuckBoundaryBenchmarkError, match="duty"):
        BuckBoundaryParameters(100.0, 100e-6, 1e-6, 20e3, 1.0)
    with pytest.raises(BuckBoundaryBenchmarkError, match="strictly increasing"):
        run_buck_boundary_sweep(extract_buck_boundary_parameters(base_model), [1.0, 0.9])
    with pytest.raises(BuckBoundaryBenchmarkError, match="exact BCM ratio"):
        run_buck_boundary_sweep(extract_buck_boundary_parameters(base_model), [0.9, 1.1])
    with pytest.raises(BuckBoundaryBenchmarkError, match="coincidence band"):
        run_buck_boundary_sweep(
            extract_buck_boundary_parameters(base_model),
            [1.0, 1.0000000000001],
        )
    with pytest.raises(BuckBoundaryBenchmarkError, match="switch must connect input"):
        extract_buck_boundary_parameters(load_model(BOOST))


def test_every_default_point_matches_pear_sim_from_independent_periodic_ic(
    base_model: Model,
    boundary_sweep,
):
    observations = []
    for point in boundary_sweep.points:
        model = _periodic_model(base_model, point, boundary_sweep.parameters)
        result = HybridSimulator(model).simulate()
        observations.append(
            observe_simulation_result(
                result,
                point,
                parameters=boundary_sweep.parameters,
                bcm_load_ohm=boundary_sweep.bcm_load_ohm,
            )
        )

    comparison = compare_buck_boundary_sweep(boundary_sweep, observations)

    assert comparison["schema"] == BUCK_BOUNDARY_RESULT_SCHEMA
    assert comparison["claim"]["validation_claim"] is False
    assert comparison["summary"] == {
        "status": "pass",
        "point_count": 9,
        "gate_count": 99,
        "failed_gate_count": 0,
        "not_run_gate_count": 0,
    }
    assert [point["status"] for point in comparison["points"]] == ["pass"] * 9
    assert json.loads(json.dumps(comparison, allow_nan=False, sort_keys=True)) == comparison

    observed_bcm = observations[4]
    period = boundary_sweep.parameters.period_s
    assert observed_bcm.regime == "BCM"
    assert observed_bcm.diode_zero_s is not None
    assert abs(observed_bcm.diode_zero_s - period) / period <= 5e-4
    assert observed_bcm.fraction_by_mode["idle"] <= BCM_COINCIDENCE_FRACTION_TOLERANCE


def test_oracle_rejects_dcm_branch_with_reverse_current_inside_on_interval():
    parameters = BuckBoundaryParameters(
        vin_v=1.0,
        inductance_h=1e-4,
        capacitance_f=1e-4,
        switching_frequency_hz=1.0,
        duty=0.1,
    )
    boundary = find_bcm_load(parameters)

    with pytest.raises(BuckBoundaryBenchmarkError, match="ON interval contains forbidden reverse"):
        solve_periodic_buck(
            parameters,
            1.01 * boundary,
            bcm_load_ohm=boundary,
            load_ratio_to_bcm=1.01,
        )


def test_oracle_fails_closed_before_materializing_unbounded_extrema():
    parameters = BuckBoundaryParameters(
        vin_v=1.0,
        inductance_h=1e-9,
        capacitance_f=1e-9,
        switching_frequency_hz=1.0,
        duty=0.1,
    )
    matrix = _active_matrix(parameters, 1.0)
    offset = np.asarray([parameters.vin_v / parameters.inductance_h, 0.0])

    with pytest.raises(BuckBoundaryBenchmarkError, match="extrema count exceeds"):
        _minimum_current(
            matrix,
            offset,
            np.zeros(2),
            parameters.on_duration_s,
        )


def test_observer_rejects_equal_duration_gap_and_overlap(
    base_model: Model,
    boundary_sweep,
):
    point = boundary_sweep.points[-1]
    result = HybridSimulator(
        _periodic_model(base_model, point, boundary_sweep.parameters)
    ).simulate()
    intervals = deepcopy(result.mode_intervals)
    assert len(intervals) == 3
    displacement = boundary_sweep.parameters.period_s * 1e-6
    intervals[0]["end_s"] -= displacement
    intervals[1]["end_s"] += displacement
    tampered = replace(result, mode_intervals=intervals)

    with pytest.raises(BuckBoundaryBenchmarkError, match="contain a gap"):
        observe_simulation_result(
            tampered,
            point,
            parameters=boundary_sweep.parameters,
            bcm_load_ohm=boundary_sweep.bcm_load_ohm,
        )


def test_observer_binds_reference_to_all_scalar_physics_parameters(
    base_model: Model,
    boundary_sweep,
):
    point = boundary_sweep.points[0]
    result = HybridSimulator(
        _periodic_model(base_model, point, boundary_sweep.parameters)
    ).simulate()
    changed_components = tuple(
        replace(component, value=float(component.value) * 1.01)
        if component.type == "voltage_source"
        else component
        for component in result.model.components
    )
    mismatched = replace(result, model=replace(result.model, components=changed_components))

    with pytest.raises(BuckBoundaryBenchmarkError, match="vin_v expected"):
        observe_simulation_result(
            mismatched,
            point,
            parameters=boundary_sweep.parameters,
            bcm_load_ohm=boundary_sweep.bcm_load_ohm,
        )


def test_comparison_fails_a_deliberately_wrong_regime_without_nonfinite_json(
    base_model: Model,
    boundary_sweep,
):
    point = boundary_sweep.points[0]
    result = HybridSimulator(
        _periodic_model(base_model, point, boundary_sweep.parameters)
    ).simulate()
    observation = observe_simulation_result(
        result,
        point,
        parameters=boundary_sweep.parameters,
        bcm_load_ohm=boundary_sweep.bcm_load_ohm,
    )
    wrong = replace(observation, regime="DCM", diode_zero_state=None, diode_zero_s=0.5e-5)
    replacements = [
        wrong
        if candidate.point_id == point.point_id
        else _perfect_observation(candidate, boundary_sweep.parameters)
        for candidate in boundary_sweep.points
    ]

    comparison = compare_buck_boundary_sweep(boundary_sweep, replacements)

    assert comparison["summary"]["status"] == "fail"
    assert comparison["summary"]["failed_gate_count"] == 2
    failed_ids = {
        gate["id"] for gate in comparison["points"][0]["gates"] if gate["status"] == "fail"
    }
    assert failed_ids == {"regime", "natural_event_time"}
    json.dumps(comparison, allow_nan=False, sort_keys=True)


def test_public_single_point_comparison_supports_partial_runner_results(boundary_sweep):
    point = boundary_sweep.points[-1]
    comparison = compare_buck_boundary_point(
        boundary_sweep.parameters,
        point,
        _perfect_observation(point, boundary_sweep.parameters),
    )

    assert comparison["point_id"] == point.point_id
    assert comparison["status"] == "pass"
    assert comparison["regime"] == "DCM"
    assert comparison["metrics"]["mean_output_voltage_v"] == point.mean_output_voltage_v
    assert [gate["status"] for gate in comparison["gates"]] == ["pass"] * 11


def test_cycle_drift_is_blocking_below_the_looser_boundary_state_limit(boundary_sweep):
    point = boundary_sweep.points[0]
    observation = _perfect_observation(point, boundary_sweep.parameters)
    drifted = replace(
        observation,
        end_state=replace(
            observation.end_state,
            output_voltage_v=(
                observation.end_state.output_voltage_v + boundary_sweep.parameters.vin_v * 1e-6
            ),
        ),
    )

    comparison = compare_buck_boundary_point(boundary_sweep.parameters, point, drifted)
    statuses = {gate["id"]: gate["status"] for gate in comparison["gates"]}

    assert statuses["boundary_state_nonzero"] == "pass"
    assert statuses["cycle_periodicity"] == "fail"
    assert comparison["status"] == "fail"


def test_unclassified_or_nonunit_mode_fractions_are_blocking(boundary_sweep):
    point = boundary_sweep.points[0]
    observation = _perfect_observation(point, boundary_sweep.parameters)
    fractions = dict(observation.fraction_by_mode)
    fractions["unclassified"] = 9e-4
    malformed = replace(observation, fraction_by_mode=fractions)

    comparison = compare_buck_boundary_point(boundary_sweep.parameters, point, malformed)
    statuses = {gate["id"]: gate["status"] for gate in comparison["gates"]}

    assert statuses["mode_fractions"] == "pass"
    assert statuses["unclassified_fraction"] == "fail"
    assert statuses["mode_fraction_sum"] == "fail"
    assert comparison["status"] == "fail"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"mean_output_voltage_v": math.nan}, "mean_output_voltage_v must be finite"),
        ({"minimum_inductor_current_a": math.inf}, "minimum_inductor_current_a must be finite"),
        ({"energy_balance_relative": -1.0}, "energy_balance_relative must be nonnegative"),
    ],
)
def test_public_comparison_rejects_nonfinite_or_negative_observations(
    boundary_sweep,
    changes,
    message,
):
    point = boundary_sweep.points[0]
    malformed = replace(
        _perfect_observation(point, boundary_sweep.parameters),
        **changes,
    )

    with pytest.raises(BuckBoundaryBenchmarkError, match=message):
        compare_buck_boundary_point(boundary_sweep.parameters, point, malformed)


def _perfect_observation(
    point,
    parameters: BuckBoundaryParameters,
) -> BuckBoundaryObservation:
    return BuckBoundaryObservation(
        point_id=point.point_id,
        parameters=parameters,
        load_ratio_to_bcm=point.load_ratio_to_bcm,
        load_ohm=point.load_ohm,
        regime=point.regime,
        start_state=point.start_state,
        switch_off_state=point.switch_off_state,
        diode_zero_state=point.diode_zero_state,
        end_state=point.end_state,
        diode_zero_s=point.diode_zero_s,
        mean_output_voltage_v=point.mean_output_voltage_v,
        minimum_inductor_current_a=point.minimum_inductor_current_a,
        fraction_by_mode=dict(point.fraction_by_mode),
        energy_balance_relative=0.0,
    )
