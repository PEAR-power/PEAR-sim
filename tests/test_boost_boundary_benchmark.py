# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

import ast
from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from pear_sim.benchmarks.boost_boundary import (
    BOOST_BOUNDARY_ORACLE_SCHEMA,
    BOOST_BOUNDARY_RESULT_SCHEMA,
    BOOST_DEFAULT_LOAD_RATIOS,
    BoostBoundaryBenchmarkError,
    BoostBoundaryObservation,
    BoostBoundaryParameters,
    compare_boost_boundary_point,
    compare_boost_boundary_sweep,
    extract_boost_boundary_parameters,
    find_boost_bcm_load,
    observe_boost_simulation_result,
    run_boost_boundary_sweep,
    solve_periodic_boost,
)
from pear_sim.model import Model, load_model
from pear_sim.solver import HybridSimulator


ROOT = Path(__file__).parents[1]
BOOST = ROOT / "examples" / "boost.yaml"
BUCK = ROOT / "examples" / "buck_dcm.yaml"
EXPECTED_BCM_LOAD_OHM = 159.8749935034275


@pytest.fixture(scope="module")
def base_model() -> Model:
    return load_model(BOOST)


@pytest.fixture(scope="module")
def boundary_sweep(base_model: Model):
    return run_boost_boundary_sweep(extract_boost_boundary_parameters(base_model))


def _periodic_model(
    base: Model,
    point,
    parameters: BoostBoundaryParameters,
    *,
    output_step_fraction: float = 1.0 / 200.0,
) -> Model:
    components = []
    for component in base.components:
        if component.type == "resistor":
            component = replace(component, value=point.load_ohm)
        elif component.type == "inductor":
            component = replace(
                component,
                initial_state=point.start_state.inductor_current_a,
                initial_state_explicit=False,
            )
        elif component.type == "capacitor":
            component = replace(
                component,
                initial_state=point.start_state.output_voltage_v,
                initial_state_explicit=False,
            )
        components.append(component)
    period = parameters.period_s
    return replace(
        base,
        components=tuple(components),
        transient=replace(
            base.transient,
            start=0.0,
            stop=period,
            output_step=period * output_step_fraction,
            max_step=period / 100.0,
            rtol=1e-10,
            atol=1e-12,
        ),
        outputs=("v(out)", "i(l1)", "i(d1)", "i(load)"),
    )


def _observe(base: Model, sweep, point, *, output_step_fraction=1.0 / 200.0):
    result = HybridSimulator(
        _periodic_model(
            base,
            point,
            sweep.parameters,
            output_step_fraction=output_step_fraction,
        )
    ).simulate()
    observation = observe_boost_simulation_result(
        result,
        point,
        parameters=sweep.parameters,
        bcm_load_ohm=sweep.bcm_load_ohm,
    )
    return result, observation


def test_numerical_boundary_and_default_regime_order_are_reproducible(boundary_sweep):
    assert boundary_sweep.load_ratios == BOOST_DEFAULT_LOAD_RATIOS
    assert boundary_sweep.bcm_load_ohm == pytest.approx(
        EXPECTED_BCM_LOAD_OHM, rel=0.0, abs=3e-12
    )
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


def test_boundary_is_not_silently_replaced_by_textbook_small_ripple_value(
    base_model: Model,
):
    parameters = extract_boost_boundary_parameters(base_model)
    textbook = (
        2.0
        * parameters.inductance_h
        * parameters.switching_frequency_hz
        / (parameters.duty * (1.0 - parameters.duty) ** 2)
    )
    assert textbook == pytest.approx(160.0, abs=1e-13)
    assert find_boost_bcm_load(parameters) == pytest.approx(
        EXPECTED_BCM_LOAD_OHM, abs=3e-12
    )
    assert abs(textbook - find_boost_bcm_load(parameters)) > 0.1


def test_nominal_ccm_and_bcm_goldens(base_model: Model):
    parameters = extract_boost_boundary_parameters(base_model)
    nominal = solve_periodic_boost(parameters, 24.0)
    assert nominal.load_ratio_to_bcm == pytest.approx(0.15011728522438048, abs=2e-15)
    assert nominal.regime == "CCM"
    assert nominal.start_state.inductor_current_a == pytest.approx(
        3.3968395377978284, abs=3e-13
    )
    assert nominal.start_state.output_voltage_v == pytest.approx(
        48.224426106644685, abs=3e-12
    )
    assert nominal.switch_off_state.inductor_current_a == pytest.approx(
        4.596839537797829, abs=3e-13
    )
    assert nominal.switch_off_state.output_voltage_v == pytest.approx(
        47.7246956163483, abs=3e-12
    )
    assert nominal.mean_output_voltage_v == pytest.approx(47.98706353422631, abs=3e-12)
    assert nominal.inductor_current_peak_to_peak_a == pytest.approx(1.2, abs=3e-13)
    assert nominal.output_voltage_peak_to_peak_v == pytest.approx(
        0.49973049029638617, abs=3e-12
    )

    boundary = find_boost_bcm_load(parameters)
    bcm = solve_periodic_boost(
        parameters,
        boundary,
        bcm_load_ohm=boundary,
        load_ratio_to_bcm=1.0,
    )
    assert bcm.regime == "BCM"
    assert bcm.start_state.inductor_current_a == 0.0
    assert bcm.start_state.output_voltage_v == pytest.approx(48.01249476668009, abs=3e-12)
    assert bcm.switch_off_state.inductor_current_a == pytest.approx(1.2, abs=3e-13)
    assert bcm.switch_off_state.output_voltage_v == pytest.approx(
        47.937475255835786, abs=3e-12
    )
    assert bcm.diode_zero_s == pytest.approx(parameters.period_s, abs=1e-15)
    assert bcm.mean_output_voltage_v == pytest.approx(47.98748761772755, abs=3e-12)


def test_dcm_goldens_cover_finite_idle_and_unbounded_voltage_bracket(boundary_sweep):
    ratio_two = boundary_sweep.points[7]
    assert ratio_two.load_ratio_to_bcm == 2.0
    assert ratio_two.load_ohm == pytest.approx(319.749987006855, abs=6e-12)
    assert ratio_two.start_state.output_voltage_v == pytest.approx(
        61.46932419831573, abs=3e-11
    )
    assert ratio_two.switch_off_state.output_voltage_v == pytest.approx(
        61.42128252300115, abs=3e-11
    )
    assert ratio_two.diode_zero_state is not None
    assert ratio_two.diode_zero_state.inductor_current_a == 0.0
    assert ratio_two.diode_zero_state.output_voltage_v == pytest.approx(
        61.48660373223086, abs=3e-11
    )
    assert ratio_two.diode_zero_s == pytest.approx(41.01282732079056e-6, abs=3e-16)
    assert ratio_two.mean_output_voltage_v == pytest.approx(61.45906871892248, abs=3e-11)
    assert ratio_two.fraction_by_mode["idle"] == pytest.approx(
        0.17974345358418872, abs=3e-13
    )

    light_load = boundary_sweep.points[-1]
    assert light_load.load_ratio_to_bcm == 200.0
    assert light_load.start_state.output_voltage_v == pytest.approx(
        491.9625545354141, rel=5e-12
    )
    assert light_load.diode_zero_s == pytest.approx(26.28215062662586e-6, abs=3e-16)
    assert light_load.start_state.output_voltage_v > 20.0 * boundary_sweep.parameters.vin_v


def test_oracle_modes_health_sequences_and_periodicity_are_self_consistent(
    boundary_sweep,
):
    for point in boundary_sweep.points:
        assert point.periodicity_residual_normalized <= 1e-10
        assert sum(point.fraction_by_mode.values()) == pytest.approx(1.0, abs=1e-14)
        assert point.fraction_by_mode["unclassified"] == 0.0
        assert point.minimum_inductor_current_a >= -1e-12 * boundary_sweep.parameters.current_base_a
        assert point.inductor_current_peak_to_peak_a > 0.0
        assert point.output_voltage_peak_to_peak_v > 0.0
        assert {gate["status"] for gate in point.oracle_gates} == {"pass"}
        if point.regime == "CCM":
            assert point.mode_sequence == ("on", "freewheel")
            assert point.diode_zero_s is None
            assert not any("natural_commutation" in token for token in point.event_sequence)
        elif point.regime == "BCM":
            assert point.mode_sequence == ("on", "freewheel")
            assert point.diode_zero_s == pytest.approx(boundary_sweep.parameters.period_s)
            assert point.event_sequence[2].startswith("natural_commutation")
        else:
            assert point.mode_sequence == ("on", "freewheel", "idle")
            assert point.diode_zero_s is not None
            assert point.diode_zero_s < boundary_sweep.parameters.period_s
            assert point.fraction_by_mode["idle"] > 0.0


def test_oracle_artifact_is_versioned_deterministic_and_finite(boundary_sweep):
    first = boundary_sweep.as_dict()
    second = run_boost_boundary_sweep(boundary_sweep.parameters).as_dict()
    assert first == second
    assert first["schema"] == BOOST_BOUNDARY_ORACLE_SCHEMA
    assert first["claim"]["validation_claim"] is False
    assert first["method"]["pear_sim_circuit_or_solver_imported"] is False
    assert len(first["comparison_gate_contract"]) == 13
    assert first["oracle_health"] == {
        "status": "pass",
        "point_count": 9,
        "failed_gate_count": 0,
    }
    assert json.loads(json.dumps(first, allow_nan=False, sort_keys=True)) == first


def test_oracle_module_has_no_kernel_solver_or_event_manager_import():
    source_path = ROOT / "src" / "pear_sim" / "benchmarks" / "boost_boundary.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported & {
        "pear_sim.circuit",
        "pear_sim.solver",
        "pear_sim.mode_coverage",
    } == set()


def test_parameter_request_and_topology_recognition_fail_closed(base_model: Model):
    with pytest.raises(BoostBoundaryBenchmarkError, match="duty"):
        BoostBoundaryParameters(24.0, 500e-6, 100e-6, 20e3, 1.0)
    parameters = extract_boost_boundary_parameters(base_model)
    with pytest.raises(BoostBoundaryBenchmarkError, match="strictly increasing"):
        run_boost_boundary_sweep(parameters, [1.0, 0.9])
    with pytest.raises(BoostBoundaryBenchmarkError, match="exact BCM ratio"):
        run_boost_boundary_sweep(parameters, [0.9, 1.1])
    with pytest.raises(BoostBoundaryBenchmarkError, match="coincidence band"):
        run_boost_boundary_sweep(parameters, [1.0, 1.0000000000001])
    with pytest.raises(BoostBoundaryBenchmarkError):
        extract_boost_boundary_parameters(load_model(BUCK))


@pytest.mark.parametrize(
    "component_type,changes,message",
    [
        ("voltage_source", {"p": "wrong"}, "inductor must connect input"),
        ("inductor", {"p": "wrong"}, "inductor must connect input"),
        ("ideal_switch", {"n": "out"}, "switch must connect"),
        ("diode", {"p": "in"}, "diode must be oriented"),
        ("capacitor", {"p": "sw"}, "capacitor must connect"),
        ("resistor", {"p": "sw"}, "load resistor must connect"),
    ],
)
def test_each_canonical_connectivity_role_is_strict(
    base_model: Model,
    component_type: str,
    changes: dict[str, str],
    message: str,
):
    components = tuple(
        replace(component, **changes) if component.type == component_type else component
        for component in base_model.components
    )
    with pytest.raises(BoostBoundaryBenchmarkError, match=message):
        extract_boost_boundary_parameters(replace(base_model, components=components))


def test_every_default_point_matches_kernel_with_117_blocking_gates(
    base_model: Model,
    boundary_sweep,
):
    observations = []
    for point in boundary_sweep.points:
        _result, observation = _observe(base_model, boundary_sweep, point)
        observations.append(observation)
    comparison = compare_boost_boundary_sweep(boundary_sweep, observations)
    assert comparison["schema"] == BOOST_BOUNDARY_RESULT_SCHEMA
    assert comparison["claim"]["validation_claim"] is False
    assert comparison["summary"] == {
        "status": "pass",
        "point_count": 9,
        "gate_count": 117,
        "failed_gate_count": 0,
        "not_run_gate_count": 0,
    }
    assert [point["status"] for point in comparison["points"]] == ["pass"] * 9
    assert json.loads(json.dumps(comparison, allow_nan=False, sort_keys=True)) == comparison

    bcm = observations[4]
    dcm = observations[7]
    assert bcm.regime == "BCM"
    assert bcm.event_sequence[2] == "natural_commutation:diode:1>0@diode_zero"
    assert bcm.event_sequence[3] == "scheduled_gate:control:0>1@cycle_end"
    assert dcm.regime == "DCM"
    assert dcm.mode_sequence == ("on", "freewheel", "idle")


def test_state_ripple_is_output_grid_independent(base_model: Model, boundary_sweep):
    point = boundary_sweep.points[7]
    _coarse_result, coarse = _observe(
        base_model, boundary_sweep, point, output_step_fraction=1.0 / 20.0
    )
    _fine_result, fine = _observe(
        base_model, boundary_sweep, point, output_step_fraction=1.0 / 400.0
    )
    assert coarse.inductor_current_peak_to_peak_a == pytest.approx(
        fine.inductor_current_peak_to_peak_a, rel=1e-11
    )
    assert coarse.output_voltage_peak_to_peak_v == pytest.approx(
        fine.output_voltage_peak_to_peak_v, rel=1e-9
    )
    assert compare_boost_boundary_point(boundary_sweep.parameters, point, coarse)[
        "status"
    ] == "pass"
    assert compare_boost_boundary_point(boundary_sweep.parameters, point, fine)[
        "status"
    ] == "pass"


def test_observer_rejects_interval_gap(base_model: Model, boundary_sweep):
    point = boundary_sweep.points[7]
    result, _observation = _observe(base_model, boundary_sweep, point)
    intervals = deepcopy(result.mode_intervals)
    intervals[0]["end_s"] -= boundary_sweep.parameters.period_s * 1e-6
    tampered = replace(result, mode_intervals=intervals)
    with pytest.raises(BoostBoundaryBenchmarkError, match="contain a gap"):
        observe_boost_simulation_result(
            tampered,
            point,
            parameters=boundary_sweep.parameters,
            bcm_load_ohm=boundary_sweep.bcm_load_ohm,
        )


def test_event_semantics_and_ripple_are_independently_blocking(boundary_sweep):
    point = boundary_sweep.points[7]
    perfect = _perfect_observation(point, boundary_sweep.parameters)
    wrong_event = replace(
        perfect,
        event_sequence=(*perfect.event_sequence[:-1], "scheduled_gate:control:0>1@wrong"),
    )
    event_comparison = compare_boost_boundary_point(
        boundary_sweep.parameters, point, wrong_event
    )
    event_statuses = {gate["id"]: gate["status"] for gate in event_comparison["gates"]}
    assert event_statuses["event_semantics"] == "fail"
    assert event_statuses["natural_event_time"] == "pass"

    wrong_ripple = replace(
        perfect,
        output_voltage_peak_to_peak_v=perfect.output_voltage_peak_to_peak_v * 1.01,
    )
    ripple_comparison = compare_boost_boundary_point(
        boundary_sweep.parameters, point, wrong_ripple
    )
    ripple_statuses = {gate["id"]: gate["status"] for gate in ripple_comparison["gates"]}
    assert ripple_statuses["state_ripple"] == "fail"
    assert ripple_statuses["boundary_state_nonzero"] == "pass"


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"mean_output_voltage_v": math.nan}, "mean_output_voltage_v must be finite"),
        ({"output_voltage_peak_to_peak_v": math.inf}, "output_voltage_peak_to_peak_v must be finite"),
        ({"energy_balance_relative": -1.0}, "energy_balance_relative must be nonnegative"),
    ],
)
def test_public_comparison_rejects_invalid_observations(
    boundary_sweep,
    changes,
    message,
):
    point = boundary_sweep.points[0]
    malformed = replace(
        _perfect_observation(point, boundary_sweep.parameters),
        **changes,
    )
    with pytest.raises(BoostBoundaryBenchmarkError, match=message):
        compare_boost_boundary_point(boundary_sweep.parameters, point, malformed)


def _perfect_observation(
    point,
    parameters: BoostBoundaryParameters,
) -> BoostBoundaryObservation:
    return BoostBoundaryObservation(
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
        inductor_current_peak_to_peak_a=point.inductor_current_peak_to_peak_a,
        output_voltage_peak_to_peak_v=point.output_voltage_peak_to_peak_v,
        fraction_by_mode=dict(point.fraction_by_mode),
        mode_sequence=point.mode_sequence,
        event_sequence=point.event_sequence,
        energy_balance_relative=0.0,
    )
