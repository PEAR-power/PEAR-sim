from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pear_sim.mode_coverage import (
    MODE_COVERAGE_SCHEMA,
    _normalized_intervals,
    build_mode_coverage,
)
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator


ROOT = Path(__file__).parents[1]


def test_canonical_buck_ccm_records_modes_without_claiming_natural_coverage():
    result = HybridSimulator(load_model(ROOT / "examples" / "buck.yaml")).simulate()

    coverage = build_mode_coverage(result)

    assert coverage["schema"] == MODE_COVERAGE_SCHEMA
    assert coverage["claim"]["validation_claim"] is False
    assert coverage["status"] == "observed"
    assert coverage["recognition"]["status"] == "matched"
    assert coverage["summary"]["fraction_by_mode"] == pytest.approx(
        {"on": 0.4, "freewheel": 0.6, "idle": 0.0, "unclassified": 0.0},
        abs=1e-12,
    )
    assert coverage["summary"]["natural_commutation"] == {
        "exercised": False,
        "event_count": 0,
        "diode_turn_off_event_count": 0,
    }
    assert coverage["diode_conduction"]["fraction"] == pytest.approx(0.6, abs=1e-12)
    assert coverage["inductor_current"]["state_constraint_tolerance_a"] == 1e-9
    assert all(interval["mode"] in {"on", "freewheel"} for interval in coverage["intervals"])
    json.dumps(coverage, allow_nan=False, sort_keys=True)


def test_canonical_boost_ccm_records_exact_modes():
    result = HybridSimulator(load_model(ROOT / "examples" / "boost.yaml")).simulate()

    coverage = build_mode_coverage(result)

    assert coverage["status"] == "observed"
    assert coverage["recognition"]["topology"] == "canonical_single_switch_boost"
    assert coverage["recognition"]["status"] == "matched"
    assert coverage["summary"]["fraction_by_mode"] == pytest.approx(
        {"on": 0.5, "freewheel": 0.5, "idle": 0.0, "unclassified": 0.0},
        abs=1e-12,
    )
    assert coverage["diode_conduction"]["fraction"] == pytest.approx(0.5, abs=1e-12)
    assert coverage["inductor_current"]["minimum_a"] > 0.0
    assert coverage["claim"]["validation_claim"] is False


def test_canonical_inverting_buck_boost_dcm_records_idle_and_natural_event():
    result = HybridSimulator(
        load_model(
            ROOT / "examples" / "inverting_buck_boost.yaml",
            {"load": "160 ohm"},
        )
    ).simulate()

    coverage = build_mode_coverage(result)

    assert coverage["status"] == "observed"
    assert coverage["recognition"]["topology"] == (
        "canonical_single_switch_inverting_buck_boost"
    )
    assert coverage["recognition"]["status"] == "matched"
    assert coverage["summary"]["fraction_by_mode"]["idle"] > 0.0
    assert coverage["summary"]["fraction_by_mode"]["unclassified"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert coverage["summary"]["natural_commutation"]["exercised"] is True
    assert coverage["summary"]["natural_commutation"]["diode_turn_off_event_count"] > 0
    assert coverage["inductor_current"]["maximum_forbidden_reverse_current_a"] <= 1e-9
    assert coverage["claim"]["validation_claim"] is False


def test_single_period_benchmark_run_uses_the_complete_mode_window():
    source = load_model(ROOT / "examples" / "boost.yaml")
    period = source.controls["pwm"].period
    model = replace(
        source,
        components=tuple(
            replace(component, initial_state_explicit=False)
            if component.type in {"capacitor", "inductor"}
            else component
            for component in source.components
        ),
        transient=replace(
            source.transient,
            start=0.0,
            stop=period,
            output_step=period / 100.0,
            max_step=period / 100.0,
        ),
    )

    coverage = build_mode_coverage(HybridSimulator(model).simulate())

    assert coverage["window"]["selection"] == (
        "complete_run_at_most_one_switching_period_with_derived_initial_conditions"
    )
    assert coverage["window"]["duration_s"] == pytest.approx(period, abs=1e-15)
    assert coverage["summary"]["fraction_by_mode"] == pytest.approx(
        {"on": 0.5, "freewheel": 0.5, "idle": 0.0, "unclassified": 0.0},
        abs=1e-12,
    )


def test_noncanonical_converter_gets_explicit_not_applicable_observation():
    result = HybridSimulator(
        load_model(ROOT / "examples" / "inverter_three_phase_six_step.yaml")
    ).simulate()

    coverage = build_mode_coverage(result)

    assert coverage["status"] == "not_applicable"
    assert coverage["recognition"]["status"] == "not_applicable"
    assert coverage["recognition"]["topology"] == (
        "canonical_single_switch_buck_or_boost_or_inverting_buck_boost"
    )
    assert coverage["recognition"]["subjects"] is None
    assert coverage["summary"] is None
    assert coverage["intervals"] is None
    assert coverage["claim"]["validation_claim"] is False


def test_positive_sub_tolerance_bcm_interval_remains_in_coverage() -> None:
    tolerance = 1e-12
    boundary = 0.4
    idle_duration = 0.25 * tolerance
    source = [
        {
            "start_s": 0.0,
            "end_s": boundary,
            "mode_signature": "freewheel",
            "gates": {"pwm": False},
            "diodes": {"d1": True},
            "constrained_inductors": [],
        },
        {
            "start_s": boundary,
            "end_s": boundary + idle_duration,
            "mode_signature": "idle",
            "gates": {"pwm": False},
            "diodes": {"d1": False},
            "constrained_inductors": ["l1"],
        },
        {
            "start_s": boundary + idle_duration,
            "end_s": 1.0,
            "mode_signature": "on",
            "gates": {"pwm": True},
            "diodes": {"d1": False},
            "constrained_inductors": [],
        },
    ]

    intervals = _normalized_intervals(
        source,
        0.0,
        1.0,
        2.0,
        0.0,
        tolerance,
        "pwm",
        "d1",
        "l1",
    )

    assert [interval["mode"] for interval in intervals] == [
        "freewheel",
        "idle",
        "on",
    ]
    assert sum(float(interval["duration_s"]) for interval in intervals) == pytest.approx(
        1.0,
        rel=0.0,
        abs=1e-16,
    )
    assert intervals[1]["duration_s"] == pytest.approx(idle_duration, rel=0.0, abs=1e-16)


def test_inductor_extrema_do_not_depend_on_requested_outputs():
    source = load_model(ROOT / "examples" / "buck_dcm.yaml")
    components = tuple(
        replace(component, value=1.0)
        if component.id == "vin"
        else replace(component, value=10e-3, initial_state=-0.7259323042)
        if component.id == "l1"
        else replace(component, value=10e-3, initial_state=1.6877661592)
        if component.id == "c1"
        else component
        for component in source.components
    )
    control = replace(source.controls["pwm"], frequency=10.0, duty=0.9)
    transient = replace(
        source.transient,
        stop=0.1,
        output_step=100e-6,
        rtol=1e-10,
        atol=1e-12,
        max_step=100e-6,
    )
    voltage_only = replace(
        source,
        name="buck_interior_reverse_current",
        components=components,
        controls={"pwm": control},
        transient=transient,
        outputs=("v(out)",),
    )
    with_current = replace(voltage_only, outputs=("v(out)", "i(l1)"))

    first = build_mode_coverage(HybridSimulator(voltage_only).simulate())
    second = build_mode_coverage(HybridSimulator(with_current).simulate())

    assert first["inductor_current"] == second["inductor_current"]
    assert first["inductor_current"]["minimum_a"] == pytest.approx(
        -0.99645836,
        rel=0.0,
        abs=1e-7,
    )
    assert first["inductor_current"]["maximum_forbidden_reverse_current_a"] > 0.99
    assert "derivative_root_search" in first["inductor_current"]["method"]
