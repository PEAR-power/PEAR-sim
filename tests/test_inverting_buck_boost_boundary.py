from __future__ import annotations

import ast
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from pear_sim.benchmarks.inverting_buck_boost_boundary import (
    INVERTING_BUCK_BOOST_BOUNDARY_ORACLE_SCHEMA,
    INVERTING_BUCK_BOOST_BOUNDARY_RESULT_SCHEMA,
    INVERTING_BUCK_BOOST_DEFAULT_LOAD_RATIOS,
    InvertingBuckBoostBoundaryBenchmarkError,
    InvertingBuckBoostBoundaryObservation,
    InvertingBuckBoostBoundaryParameters,
    compare_inverting_buck_boost_boundary_point,
    compare_inverting_buck_boost_boundary_sweep,
    extract_inverting_buck_boost_boundary_parameters,
    find_inverting_buck_boost_bcm_load,
    observe_inverting_buck_boost_simulation_result,
    run_inverting_buck_boost_boundary_sweep,
    solve_periodic_inverting_buck_boost,
)
from pear_sim.model import Component, Model, load_model
from pear_sim.solver import HybridSimulator


ROOT = Path(__file__).parents[1]
INVERTING_BUCK_BOOST = ROOT / "examples" / "inverting_buck_boost.yaml"
BOOST = ROOT / "examples" / "boost.yaml"
EXPECTED_BCM_LOAD_OHM = 79.91663630292852


@pytest.fixture(scope="module")
def base_model() -> Model:
    return load_model(INVERTING_BUCK_BOOST)


@pytest.fixture(scope="module")
def boundary_sweep(base_model: Model):
    return run_inverting_buck_boost_boundary_sweep(
        extract_inverting_buck_boost_boundary_parameters(base_model)
    )


def _periodic_model(
    base: Model,
    point,
    parameters: InvertingBuckBoostBoundaryParameters,
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
                initial_state=point.start_state.output_magnitude_voltage_v,
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
    observation = observe_inverting_buck_boost_simulation_result(
        result,
        point,
        parameters=sweep.parameters,
        bcm_load_ohm=sweep.bcm_load_ohm,
    )
    return result, observation


def _perfect_observation(
    point,
    parameters: InvertingBuckBoostBoundaryParameters,
) -> InvertingBuckBoostBoundaryObservation:
    return InvertingBuckBoostBoundaryObservation(
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
        signed_mean_output_voltage_v=point.signed_mean_output_voltage_v,
        maximum_signed_output_voltage_v=point.maximum_signed_output_voltage_v,
        minimum_inductor_current_a=point.minimum_inductor_current_a,
        inductor_current_peak_to_peak_a=point.inductor_current_peak_to_peak_a,
        output_voltage_peak_to_peak_v=point.output_voltage_peak_to_peak_v,
        fraction_by_mode=dict(point.fraction_by_mode),
        mode_sequence=point.mode_sequence,
        event_sequence=point.event_sequence,
        energy_balance_relative=0.0,
    )


def test_numerical_boundary_and_regime_order_are_reproducible(boundary_sweep):
    assert boundary_sweep.load_ratios == INVERTING_BUCK_BOOST_DEFAULT_LOAD_RATIOS
    assert boundary_sweep.bcm_load_ohm == pytest.approx(EXPECTED_BCM_LOAD_OHM, rel=0.0, abs=3e-12)
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


def test_boundary_is_numeric_not_silently_the_small_ripple_seed(base_model: Model):
    parameters = extract_inverting_buck_boost_boundary_parameters(base_model)
    textbook = (
        2.0
        * parameters.inductance_h
        * parameters.switching_frequency_hz
        / (1.0 - parameters.duty) ** 2
    )
    assert textbook == pytest.approx(80.0, abs=1e-13)
    assert find_inverting_buck_boost_bcm_load(parameters) == pytest.approx(
        EXPECTED_BCM_LOAD_OHM, abs=3e-12
    )
    assert abs(textbook - find_inverting_buck_boost_bcm_load(parameters)) > 0.08


def test_nominal_and_bcm_goldens(base_model: Model):
    parameters = extract_inverting_buck_boost_boundary_parameters(base_model)
    nominal = solve_periodic_inverting_buck_boost(parameters, 24.0)
    assert nominal.regime == "CCM"
    assert nominal.start_state.inductor_current_a == pytest.approx(1.3978994887121678, abs=3e-13)
    assert nominal.start_state.output_magnitude_voltage_v == pytest.approx(
        24.099645467664033, abs=3e-12
    )
    assert nominal.switch_off_state.inductor_current_a == pytest.approx(
        2.5978994887121676, abs=3e-13
    )
    assert nominal.signed_mean_output_voltage_v == pytest.approx(-23.987280589027776, abs=3e-12)

    boundary = find_inverting_buck_boost_bcm_load(parameters)
    bcm = solve_periodic_inverting_buck_boost(
        parameters,
        boundary,
        bcm_load_ohm=boundary,
        load_ratio_to_bcm=1.0,
    )
    assert bcm.regime == "BCM"
    assert bcm.start_state.inductor_current_a == 0.0
    assert bcm.start_state.output_magnitude_voltage_v == pytest.approx(
        24.012475179385394, abs=3e-12
    )
    assert bcm.switch_off_state.inductor_current_a == pytest.approx(1.2, abs=3e-13)
    assert bcm.diode_zero_s == pytest.approx(parameters.period_s, abs=1e-15)
    assert bcm.signed_mean_output_voltage_v == pytest.approx(-23.987477841431755, abs=3e-12)


def test_dcm_goldens_cover_idle_and_unbounded_voltage_bracket(boundary_sweep):
    ratio_two = boundary_sweep.points[7]
    assert ratio_two.load_ohm == pytest.approx(159.83327260585705, abs=6e-12)
    assert ratio_two.start_state.output_magnitude_voltage_v == pytest.approx(
        33.93592951576501, abs=3e-11
    )
    assert ratio_two.diode_zero_s == pytest.approx(42.680927685954535e-6, abs=3e-16)
    assert ratio_two.signed_mean_output_voltage_v == pytest.approx(-33.92342961793503, abs=3e-11)
    assert ratio_two.fraction_by_mode["idle"] == pytest.approx(0.14638144628090932, abs=3e-13)

    light_load = boundary_sweep.points[-1]
    assert light_load.start_state.output_magnitude_voltage_v == pytest.approx(
        339.2344928732591, rel=5e-12
    )
    assert light_load.diode_zero_s == pytest.approx(26.768679825534054e-6, abs=3e-16)
    assert light_load.signed_mean_output_voltage_v < -10.0 * boundary_sweep.parameters.vin_v


def test_oracle_is_deterministic_finite_and_has_four_health_gates(boundary_sweep):
    first = boundary_sweep.as_dict()
    second = run_inverting_buck_boost_boundary_sweep(boundary_sweep.parameters).as_dict()
    assert first == second
    assert first["schema"] == INVERTING_BUCK_BOOST_BOUNDARY_ORACLE_SCHEMA
    assert first["claim"]["validation_claim"] is False
    assert first["method"]["pear_sim_circuit_or_solver_imported"] is False
    assert len(first["comparison_gate_contract"]) == 14
    assert all(
        len(point["oracle_gates"]) == 4 and point["metrics"]["signed_mean_output_voltage_v"] < 0.0
        for point in first["points"]
    )
    assert json.loads(json.dumps(first, allow_nan=False, sort_keys=True)) == first


def test_oracle_module_has_no_kernel_solver_or_event_manager_import():
    source_path = ROOT / "src" / "pear_sim" / "benchmarks" / "inverting_buck_boost_boundary.py"
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
    assert imported & {"pear_sim.circuit", "pear_sim.solver", "pear_sim.mode_coverage"} == set()


def test_request_and_noncanonical_model_fail_closed(base_model: Model):
    with pytest.raises(InvertingBuckBoostBoundaryBenchmarkError, match="duty"):
        InvertingBuckBoostBoundaryParameters(24.0, 500e-6, 100e-6, 20e3, 1.0)
    parameters = extract_inverting_buck_boost_boundary_parameters(base_model)
    with pytest.raises(InvertingBuckBoostBoundaryBenchmarkError, match="strictly increasing"):
        run_inverting_buck_boost_boundary_sweep(parameters, [1.0, 0.9])
    with pytest.raises(InvertingBuckBoostBoundaryBenchmarkError, match="exact BCM ratio"):
        run_inverting_buck_boost_boundary_sweep(parameters, [0.9, 1.1])
    with pytest.raises(InvertingBuckBoostBoundaryBenchmarkError, match="CCM ratio below"):
        run_inverting_buck_boost_boundary_sweep(parameters, [1.0])
    with pytest.raises(
        InvertingBuckBoostBoundaryBenchmarkError,
        match="supported bounded evidence envelope",
    ):
        run_inverting_buck_boost_boundary_sweep(parameters, [1.0e-6, 1.0, 1.1])
    with pytest.raises(
        InvertingBuckBoostBoundaryBenchmarkError,
        match="regime-separation contract",
    ):
        run_inverting_buck_boost_boundary_sweep(
            parameters,
            [0.9, 1.0, 1.000000001],
        )
    with pytest.raises(
        InvertingBuckBoostBoundaryBenchmarkError,
        match="regime-separation contract",
    ):
        run_inverting_buck_boost_boundary_sweep(
            parameters,
            [0.999999999, 1.0, 1.1],
        )
    with pytest.raises(InvertingBuckBoostBoundaryBenchmarkError):
        extract_inverting_buck_boost_boundary_parameters(load_model(BOOST))


@pytest.mark.parametrize(
    "component_type,changes,message",
    [
        ("voltage_source", {"n": "wrong"}, "source must be positive"),
        ("ideal_switch", {"p": "wrong"}, "switch must connect input"),
        ("inductor", {"p": "wrong"}, "inductor must connect"),
        ("diode", {"n": "in"}, "diode must be oriented"),
        ("capacitor", {"n": "sw"}, "capacitor must be oriented"),
        ("resistor", {"n": "sw"}, "load resistor must be oriented"),
    ],
)
def test_every_canonical_connectivity_role_is_strict(
    base_model: Model,
    component_type: str,
    changes: dict[str, str],
    message: str,
):
    components = tuple(
        replace(component, **changes) if component.type == component_type else component
        for component in base_model.components
    )
    with pytest.raises(InvertingBuckBoostBoundaryBenchmarkError, match=message):
        extract_inverting_buck_boost_boundary_parameters(replace(base_model, components=components))


def test_voltage_probes_are_optional_but_extra_functional_parts_are_rejected(
    base_model: Model,
):
    with_probe = replace(
        base_model,
        components=(
            *base_model.components,
            Component("vp", "voltage_probe", "out", "gnd"),
        ),
    )
    assert extract_inverting_buck_boost_boundary_parameters(with_probe)

    extra_load = replace(
        base_model,
        components=(
            *base_model.components,
            Component("r2", "resistor", "gnd", "out", value=100.0),
        ),
    )
    with pytest.raises(InvertingBuckBoostBoundaryBenchmarkError, match="exactly one"):
        extract_inverting_buck_boost_boundary_parameters(extra_load)


def test_every_default_point_matches_kernel_with_126_blocking_gates(
    base_model: Model, boundary_sweep
):
    observations = []
    for point in boundary_sweep.points:
        _result, observation = _observe(base_model, boundary_sweep, point)
        observations.append(observation)
    comparison = compare_inverting_buck_boost_boundary_sweep(boundary_sweep, observations)
    assert comparison["schema"] == INVERTING_BUCK_BOOST_BOUNDARY_RESULT_SCHEMA
    assert comparison["claim"]["validation_claim"] is False
    assert comparison["summary"] == {
        "status": "pass",
        "point_count": 9,
        "gate_count": 126,
        "failed_gate_count": 0,
        "not_run_gate_count": 0,
    }
    assert [point["status"] for point in comparison["points"]] == ["pass"] * 9
    assert json.loads(json.dumps(comparison, allow_nan=False, sort_keys=True)) == comparison
    assert observations[4].regime == "BCM"
    assert observations[7].mode_sequence == ("on", "freewheel", "idle")


def test_sign_event_and_ripple_mutations_are_independently_blocking(boundary_sweep):
    point = boundary_sweep.points[7]
    perfect = _perfect_observation(point, boundary_sweep.parameters)

    wrong_sign = replace(
        perfect,
        signed_mean_output_voltage_v=-perfect.signed_mean_output_voltage_v,
        maximum_signed_output_voltage_v=abs(perfect.maximum_signed_output_voltage_v),
    )
    sign_gates = {
        gate["id"]: gate["status"]
        for gate in compare_inverting_buck_boost_boundary_point(
            boundary_sweep.parameters, point, wrong_sign
        )["gates"]
    }
    assert sign_gates["signed_cycle_mean_output"] == "fail"
    assert sign_gates["signed_output_polarity"] == "fail"
    assert sign_gates["boundary_state_nonzero"] == "pass"

    wrong_event = replace(
        perfect,
        event_sequence=(*perfect.event_sequence[:-1], "scheduled_gate:control:0>1@wrong"),
    )
    event_gates = {
        gate["id"]: gate["status"]
        for gate in compare_inverting_buck_boost_boundary_point(
            boundary_sweep.parameters, point, wrong_event
        )["gates"]
    }
    assert event_gates["event_semantics"] == "fail"
    assert event_gates["natural_event_time"] == "pass"

    wrong_ripple = replace(
        perfect,
        output_voltage_peak_to_peak_v=perfect.output_voltage_peak_to_peak_v * 1.01,
    )
    ripple_gates = {
        gate["id"]: gate["status"]
        for gate in compare_inverting_buck_boost_boundary_point(
            boundary_sweep.parameters, point, wrong_ripple
        )["gates"]
    }
    assert ripple_gates["state_ripple"] == "fail"
    assert ripple_gates["boundary_state_nonzero"] == "pass"

    impossible_voltage_extreme = replace(
        perfect,
        maximum_signed_output_voltage_v=-1.0e9,
    )
    voltage_extreme_gates = {
        gate["id"]: gate["status"]
        for gate in compare_inverting_buck_boost_boundary_point(
            boundary_sweep.parameters,
            point,
            impossible_voltage_extreme,
        )["gates"]
    }
    assert voltage_extreme_gates["state_ripple"] == "fail"

    impossible_current_extreme = replace(
        perfect,
        minimum_inductor_current_a=1.0e9,
    )
    current_extreme_gates = {
        gate["id"]: gate["status"]
        for gate in compare_inverting_buck_boost_boundary_point(
            boundary_sweep.parameters,
            point,
            impossible_current_extreme,
        )["gates"]
    }
    assert current_extreme_gates["state_ripple"] == "fail"


def test_invalid_public_observation_is_rejected(boundary_sweep):
    point = boundary_sweep.points[0]
    malformed = replace(
        _perfect_observation(point, boundary_sweep.parameters),
        signed_mean_output_voltage_v=math.nan,
    )
    with pytest.raises(
        InvertingBuckBoostBoundaryBenchmarkError,
        match="signed_mean_output_voltage_v must be finite",
    ):
        compare_inverting_buck_boost_boundary_point(boundary_sweep.parameters, point, malformed)

    dcm_point = boundary_sweep.points[7]
    missing_zero_state = replace(
        _perfect_observation(dcm_point, boundary_sweep.parameters),
        diode_zero_state=None,
    )
    with pytest.raises(
        InvertingBuckBoostBoundaryBenchmarkError,
        match="time and state must either both be present",
    ):
        compare_inverting_buck_boost_boundary_point(
            boundary_sweep.parameters,
            dcm_point,
            missing_zero_state,
        )

    ccm_with_zero_event = replace(
        _perfect_observation(point, boundary_sweep.parameters),
        diode_zero_s=boundary_sweep.parameters.period_s,
        diode_zero_state=point.end_state,
    )
    with pytest.raises(
        InvertingBuckBoostBoundaryBenchmarkError,
        match="CCM observation must not contain",
    ):
        compare_inverting_buck_boost_boundary_point(
            boundary_sweep.parameters,
            point,
            ccm_with_zero_event,
        )

    oversized_modes = replace(
        _perfect_observation(point, boundary_sweep.parameters),
        mode_sequence=("on",) * 17,
    )
    with pytest.raises(
        InvertingBuckBoostBoundaryBenchmarkError,
        match="at most 16 entries",
    ):
        compare_inverting_buck_boost_boundary_point(
            boundary_sweep.parameters,
            point,
            oversized_modes,
        )

    oversized_events = replace(
        _perfect_observation(point, boundary_sweep.parameters),
        event_sequence=("scheduled",) * 33,
    )
    with pytest.raises(
        InvertingBuckBoostBoundaryBenchmarkError,
        match="at most 32 entries",
    ):
        compare_inverting_buck_boost_boundary_point(
            boundary_sweep.parameters,
            point,
            oversized_events,
        )
