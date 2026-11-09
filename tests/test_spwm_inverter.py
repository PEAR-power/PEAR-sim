# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import math
from pathlib import Path

import pytest

from pear_sim.model import Model, load_model
from pear_sim.solver import HybridSimulator, SimulationResult, scheduled_edges


EXAMPLE = Path(__file__).parents[1] / "examples" / "inverter_three_phase_spwm.yaml"


@pytest.fixture(scope="module")
def spwm_model() -> Model:
    return load_model(EXAMPLE)


@pytest.fixture(scope="module")
def spwm_result(spwm_model: Model) -> SimulationResult:
    return HybridSimulator(spwm_model).simulate()


def test_spwm_example_exercises_bridge_events_without_public_round_trips(
    spwm_result: SimulationResult,
) -> None:
    assert len(spwm_result.modulation_cycles) == 252
    assert [row["cycle_index"] for row in spwm_result.modulation_cycles] == list(range(252))
    assert spwm_result.stats["scheduled_events"] == 252 * 12
    assert spwm_result.stats["natural_events"] == sum(
        event.kind == "natural_commutation" for event in spwm_result.events
    )
    assert sum(event.kind == "switch_manager" for event in spwm_result.events) > 0

    for interval in spwm_result.mode_intervals:
        gates = interval["gates"]
        for phase in ("a", "b", "c"):
            assert not (
                gates[f"bridge.{phase}.high"]
                and gates[f"bridge.{phase}.low"]
            )

    # A diode root candidate that the active-set manager immediately rejects is not
    # a physical commutation and must not leak as an on/off round trip at one time.
    diode_events: dict[tuple[float, str], list[object]] = defaultdict(list)
    for event in spwm_result.events:
        if event.kind != "scheduled_gate":
            diode_events[(event.time, event.target)].append(event)
    for events in diode_events.values():
        assert events[0].previous != events[-1].current

    audit = spwm_result.physics_audit["last_half"]
    assert audit["balance_within_tolerance"] is True
    assert audit["balance_residual_relative"] < 1e-8


def test_full_example_schedule_matches_an_independent_252_cell_oracle(
    spwm_model: Model,
) -> None:
    high_a = spwm_model.controls["bridge.a.high"]
    carrier_period = 1.0 / high_a.frequency
    fundamental = high_a.fundamental_frequency
    modulation_index = high_a.modulation_index
    reference_phase = high_a.reference_phase
    assert fundamental is not None
    assert modulation_index is not None
    assert reference_phase is not None

    phase_offsets = {
        "a": 0.0,
        "b": -2.0 * math.pi / 3.0,
        "c": 2.0 * math.pi / 3.0,
    }
    expected: list[tuple[float, str, bool]] = []
    for cycle_index in range(252):
        cell_start = cycle_index * carrier_period
        sample = cell_start + 0.5 * carrier_period
        base_angle = math.tau * ((fundamental * (sample - reference_phase)) % 1.0)
        for phase, offset in phase_offsets.items():
            reference = math.sin(base_angle + offset)
            duty = 0.5 * (1.0 + modulation_index * reference)
            command_start = cell_start + 0.5 * (1.0 - duty) * carrier_period
            command_stop = cell_start + 0.5 * (1.0 + duty) * carrier_period
            expected.extend(
                (
                    (command_start, f"bridge.{phase}.low", False),
                    (command_start + high_a.dead_time, f"bridge.{phase}.high", True),
                    (command_stop, f"bridge.{phase}.high", False),
                    (command_stop + high_a.dead_time, f"bridge.{phase}.low", True),
                )
            )
    expected.sort(key=lambda edge: (edge[0], edge[2], edge[1]))

    actual = scheduled_edges(
        spwm_model.controls,
        spwm_model.transient.start,
        spwm_model.transient.stop,
    )
    assert len(actual) == len(expected) == 252 * 12
    timing_tolerance = 0.0005 * carrier_period
    for actual_edge, expected_edge in zip(actual, expected, strict=True):
        assert actual_edge[1:] == expected_edge[1:]
        assert abs(actual_edge[0] - expected_edge[0]) <= timing_tolerance


def test_spwm_symmetric_phase_edges_are_atomic_without_ulp_mode_slivers(
    spwm_model: Model,
    spwm_result: SimulationResult,
) -> None:
    cycles = {row["cycle_index"]: row for row in spwm_result.modulation_cycles}

    # At these N=42 carrier samples the named phase references are mathematically
    # equal.  Raw libm results differ by one ULP in selected frames, so all four
    # matching transition roles must use one exact timestamp.
    for cycle_index, left, right in ((3, "a", "c"), (31, "b", "c")):
        pulses = cycles[cycle_index]["command_pulses"]
        assert pulses[left] == pulses[right]

    # A one-ULP split used to materialize four nonphysical topology intervals in
    # this fixture.  Every retained mode must span more than the scheduler's
    # existing eight-ULP coincidence envelope.
    for interval in spwm_result.mode_intervals:
        start = interval["start_s"]
        end = interval["end_s"]
        assert end - start > 8.0 * max(math.ulp(start), math.ulp(end))

    dead_time = spwm_model.controls["bridge.a.high"].dead_time
    for cycle_index in (3, 31):
        for pulse in cycles[cycle_index]["command_pulses"].values():
            assert pulse["high_turn_on_s"] > pulse["start_s"]
            assert pulse["low_turn_on_s"] > pulse["end_s"]
            assert pulse["high_turn_on_s"] - pulse["start_s"] == pytest.approx(
                dead_time,
                rel=0.0,
                abs=8.0 * math.ulp(pulse["high_turn_on_s"]),
            )
            assert pulse["low_turn_on_s"] - pulse["end_s"] == pytest.approx(
                dead_time,
                rel=0.0,
                abs=8.0 * math.ulp(pulse["low_turn_on_s"]),
            )


def test_spwm_dense_harmonic_observation_is_balanced_and_explicit(
    spwm_result: SimulationResult,
) -> None:
    observation = spwm_result.harmonic_analysis
    assert observation is not None
    assert observation["schema"] == "pear-sim/harmonic-analysis/v0.1"
    assert observation["window"] == {
        "start_s": 0.008,
        "stop_s": 0.012,
        "duration_s": 0.004,
        "endpoint_convention": "half_open_[start,stop)",
        "fundamental_period_count": 2,
        "source": "solver_dense_output_on_event_aligned_topology_intervals",
    }
    assert observation["frequency"]["fundamental_hz"] == 500.0
    assert observation["frequency"]["max_harmonic"] == 42
    assert observation["frequency"]["bandwidth_hz"] == 21_000.0
    assert observation["method"]["dc_policy"] == (
        "retained_as_coefficient_0_excluded_from_thd"
    )
    assert observation["summary"]["acceptance_status"] == "not_evaluated"

    line_records = [
        row for row in observation["signals"] if row["semantic"].startswith("line_voltage")
    ]
    line_fundamentals = [row["fundamental"]["rms_amplitude"] for row in line_records]
    assert (max(line_fundamentals) - min(line_fundamentals)) / max(
        line_fundamentals
    ) < 1e-12
    assert all(row["thd"]["status"] == "observed" for row in observation["signals"])

    current_sequence = next(
        row for row in observation["sequence_components"] if row["family"] == "phase_current"
    )
    assert current_sequence["negative_to_positive_fraction"] < 1e-5
    assert current_sequence["zero_to_positive_fraction"] < 1e-5


def test_zero_dead_time_spwm_fundamental_matches_linear_range_relation(
    spwm_model: Model,
) -> None:
    zero_dead_time = replace(
        spwm_model,
        controls={
            control_id: replace(control, dead_time=0.0)
            for control_id, control in spwm_model.controls.items()
        },
    )
    observation = HybridSimulator(zero_dead_time).simulate().harmonic_analysis
    assert observation is not None

    # For a two-level bridge in linear SPWM, V_LL,1,rms = m*Vdc*sqrt(3/8).
    # The offline midpoint-sampled scheduler has a finite carrier-ratio sampling
    # error; N=42 remains inside the VAL-005 0.1% amplitude gate.
    expected_rms = 0.8 * 400.0 * math.sqrt(3.0 / 8.0)
    measured = [
        row["fundamental"]["rms_amplitude"]
        for row in observation["signals"]
        if row["semantic"].startswith("line_voltage")
    ]
    assert len(measured) == 3
    assert max(abs(value / expected_rms - 1.0) for value in measured) <= 1e-3
    assert max(measured) - min(measured) < 1e-9


def test_harmonic_coefficients_do_not_depend_on_requested_csv_grid(
    spwm_model: Model,
    spwm_result: SimulationResult,
) -> None:
    coarse_grid_model = replace(
        spwm_model,
        transient=replace(spwm_model.transient, output_step=4e-6),
    )
    coarse = HybridSimulator(coarse_grid_model).simulate()
    assert coarse.harmonic_analysis is not None
    assert spwm_result.harmonic_analysis is not None

    for baseline_signal, coarse_signal in zip(
        spwm_result.harmonic_analysis["signals"],
        coarse.harmonic_analysis["signals"],
        strict=True,
    ):
        assert baseline_signal["signal"] == coarse_signal["signal"]
        for baseline_row, coarse_row in zip(
            baseline_signal["coefficients"],
            coarse_signal["coefficients"],
            strict=True,
        ):
            assert coarse_row["coefficient_real"] == pytest.approx(
                baseline_row["coefficient_real"], rel=0.0, abs=1e-12
            )
            assert coarse_row["coefficient_imag"] == pytest.approx(
                baseline_row["coefficient_imag"], rel=0.0, abs=1e-12
            )
