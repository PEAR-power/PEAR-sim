# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

import pytest

import pear_sim.scheduling as scheduling_module
from pear_sim.errors import SimulationError
from pear_sim.limits import MAX_SPWM_CARRIER_RATIO
from pear_sim.model import PwmControl, load_model
from pear_sim.solver import (
    HybridSimulator,
    _build_gate_schedule,
    _initial_gate_values,
    scheduled_edges,
)


ROOT = Path(__file__).parents[1]
Q = 2.0**-19
PERIOD = 64.0 * Q
FREQUENCY = 1.0 / PERIOD
RATIO = 4


def _spwm_controls(
    *,
    modulation_index: float = 0.0,
    dead_time: float = Q,
    sequence: str = "abc",
    carrier_phase: float = 0.0,
    reference_phase: float = 0.0,
    carrier_ratio: int = RATIO,
) -> dict[str, PwmControl]:
    common = {
        "frequency": FREQUENCY,
        "duty": 0.5,
        "phase": carrier_phase,
        "kind": "three_phase_spwm",
        "block_id": "bridge",
        "dead_time": dead_time,
        "normalized_phase": carrier_phase % PERIOD,
        "sequence": sequence,
        "fundamental_frequency": FREQUENCY / carrier_ratio,
        "modulation_index": modulation_index,
        "reference_phase": reference_phase,
        "carrier_ratio": carrier_ratio,
    }
    controls: dict[str, PwmControl] = {}
    for phase_label in ("a", "b", "c"):
        high_id = f"bridge.{phase_label}.high"
        low_id = f"bridge.{phase_label}.low"
        controls[high_id] = PwmControl(
            id=high_id,
            role="high",
            partner_id=low_id,
            phase_label=phase_label,
            **common,
        )
        controls[low_id] = PwmControl(
            id=low_id,
            role="low",
            partner_id=high_id,
            phase_label=phase_label,
            **common,
        )
    return controls


def _expected_group(time: float, role: str, value: bool) -> list[tuple[float, str, bool]]:
    return [(time, f"bridge.{phase}.{role}", value) for phase in ("a", "b", "c")]


def test_zero_index_spwm_has_binary_exact_center_aligned_edges_and_cycle_record() -> None:
    schedule = _build_gate_schedule(_spwm_controls(), 0.0, PERIOD)

    assert schedule.edges == [
        *_expected_group(16.0 * Q, "low", False),
        *_expected_group(17.0 * Q, "high", True),
        *_expected_group(48.0 * Q, "high", False),
        *_expected_group(49.0 * Q, "low", True),
    ]
    assert len(schedule.modulation_cycles) == 1
    cycle = schedule.modulation_cycles[0]
    assert cycle["block_id"] == "bridge"
    assert cycle["cycle_index"] == cycle["frame_index"] == 0
    assert cycle["start_s"] == 0.0
    assert cycle["end_s"] == PERIOD
    assert cycle["sample_s"] == 32.0 * Q
    assert set(cycle["references"]) == {"a", "b", "c"}
    assert cycle["duties"] == {"a": 0.5, "b": 0.5, "c": 0.5}
    assert cycle["command_pulses"] == {
        phase: {
            "start_s": 16.0 * Q,
            "end_s": 48.0 * Q,
            "high_turn_on_s": 17.0 * Q,
            "low_turn_on_s": 49.0 * Q,
        }
        for phase in ("a", "b", "c")
    }
    assert cycle["phase_sequence"] == "abc"
    assert cycle["sampling"] == "offline_midpoint_sampled_center_aligned"
    assert cycle["dead_time_s"] == Q


def test_spwm_initial_state_and_schedule_share_the_exact_start_boundary() -> None:
    controls = _spwm_controls()
    high_on = 17.0 * Q
    stop = 81.0 * Q

    assert _initial_gate_values(controls, high_on) == {
        f"bridge.{phase}.{role}": role == "high"
        for phase in ("a", "b", "c")
        for role in ("high", "low")
    }
    assert scheduled_edges(controls, high_on, stop) == [
        *_expected_group(48.0 * Q, "high", False),
        *_expected_group(49.0 * Q, "low", True),
        *_expected_group(80.0 * Q, "low", False),
        *_expected_group(81.0 * Q, "high", True),
    ]


def test_zero_dead_time_spwm_swaps_each_leg_in_one_exact_atomic_group() -> None:
    controls = _spwm_controls(dead_time=0.0)
    edges = scheduled_edges(controls, 0.0, PERIOD)

    assert edges[:6] == [
        *_expected_group(16.0 * Q, "low", False),
        *_expected_group(16.0 * Q, "high", True),
    ]
    assert edges[6:] == [
        *_expected_group(48.0 * Q, "high", False),
        *_expected_group(48.0 * Q, "low", True),
    ]
    assert _initial_gate_values(controls, 16.0 * Q) == {
        f"bridge.{phase}.{role}": role == "high"
        for phase in ("a", "b", "c")
        for role in ("high", "low")
    }


def test_abc_and_acb_swap_only_phase_b_and_c_reference_frames() -> None:
    abc = _build_gate_schedule(
        _spwm_controls(modulation_index=0.75, sequence="abc", dead_time=0.0),
        0.0,
        PERIOD,
    ).modulation_cycles[0]
    acb = _build_gate_schedule(
        _spwm_controls(modulation_index=0.75, sequence="acb", dead_time=0.0),
        0.0,
        PERIOD,
    ).modulation_cycles[0]

    assert acb["references"]["a"] == abc["references"]["a"]
    assert acb["references"]["b"] == abc["references"]["c"]
    assert acb["references"]["c"] == abc["references"]["b"]
    assert acb["duties"]["a"] == abc["duties"]["a"]
    assert acb["duties"]["b"] == abc["duties"]["c"]
    assert acb["duties"]["c"] == abc["duties"]["b"]
    assert acb["command_pulses"]["b"] == abc["command_pulses"]["c"]
    assert acb["command_pulses"]["c"] == abc["command_pulses"]["b"]


def test_precomputed_spwm_frame_repeats_without_accumulated_phase_drift() -> None:
    cycles = _build_gate_schedule(
        _spwm_controls(modulation_index=0.75, dead_time=0.0),
        0.0,
        5.0 * PERIOD,
    ).modulation_cycles

    first = cycles[0]
    repeated = cycles[RATIO]
    assert repeated["cycle_index"] == RATIO
    assert repeated["frame_index"] == first["frame_index"] == 0
    assert repeated["electrical_angle_rad"] == first["electrical_angle_rad"]
    assert repeated["references"] == first["references"]
    assert repeated["duties"] == first["duties"]
    for phase in ("a", "b", "c"):
        for field in ("start_s", "end_s", "high_turn_on_s", "low_turn_on_s"):
            assert (
                repeated["command_pulses"][phase][field]
                == first["command_pulses"][phase][field] + RATIO * PERIOD
            )


def test_carrier_phase_before_run_retains_the_signed_overlapping_cycle_index() -> None:
    carrier_phase = 8.0 * Q
    cycle = _build_gate_schedule(
        _spwm_controls(carrier_phase=carrier_phase),
        0.0,
        8.0 * Q,
    ).modulation_cycles[0]

    assert cycle["cycle_index"] == -1
    assert cycle["frame_index"] == RATIO - 1
    assert cycle["start_s"] == carrier_phase - PERIOD
    assert cycle["end_s"] == carrier_phase


def test_spwm_rejects_unrepresentable_absolute_carrier_origin() -> None:
    controls = {
        control_id: replace(
            control,
            frequency=1.0,
            fundamental_frequency=0.25,
        )
        for control_id, control in _spwm_controls(dead_time=0.0).items()
    }

    with pytest.raises(SimulationError, match="not representable|loses distinct"):
        _build_gate_schedule(controls, 1e16, 1e16 + 2.0)


def test_positive_spwm_dead_time_must_survive_absolute_timestamp_rounding() -> None:
    dead_time = math.ulp(16.0 * Q) / 4.0

    with pytest.raises(SimulationError, match="pulse.*not representable"):
        _build_gate_schedule(
            _spwm_controls(dead_time=dead_time),
            0.0,
            PERIOD,
        )


def test_spwm_precomputed_table_has_a_fail_closed_ratio_limit() -> None:
    ratio = MAX_SPWM_CARRIER_RATIO + 1
    controls = {
        control_id: replace(
            control,
            fundamental_frequency=FREQUENCY / ratio,
            carrier_ratio=ratio,
        )
        for control_id, control in _spwm_controls(dead_time=0.0).items()
    }

    with pytest.raises(SimulationError, match="precomputed-table limit"):
        _build_gate_schedule(controls, 0.0, PERIOD)


def test_maximum_legal_spwm_ratio_schedules_one_fundamental_period() -> None:
    ratio = MAX_SPWM_CARRIER_RATIO
    schedule = _build_gate_schedule(
        _spwm_controls(carrier_ratio=ratio, dead_time=0.0),
        0.0,
        ratio * PERIOD,
    )

    assert len(schedule.edges) == 12 * ratio == 99_996
    assert len(schedule.modulation_cycles) == ratio


def test_maximum_legal_spwm_ratio_rejects_one_extra_carrier_period() -> None:
    ratio = MAX_SPWM_CARRIER_RATIO

    with pytest.raises(SimulationError, match="global limit"):
        _build_gate_schedule(
            _spwm_controls(carrier_ratio=ratio, dead_time=0.0),
            0.0,
            (ratio + 1) * PERIOD,
        )


def test_spwm_edge_cap_stops_bounded_schedule_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scheduling_module, "MAX_SCHEDULED_GATE_EDGES", 35)

    with pytest.raises(SimulationError, match="global limit"):
        _build_gate_schedule(_spwm_controls(dead_time=0.0), 0.0, 3.0 * PERIOD)


def test_spwm_solver_and_output_grid_do_not_change_command_schedule() -> None:
    model = load_model(ROOT / "examples" / "inverter_three_phase_spwm.yaml")
    short = replace(
        model,
        transient=replace(
            model.transient,
            stop=2.0 / model.controls["bridge.a.high"].frequency,
            output_step=1.0e-6,
            solver="dopri5",
        ),
        harmonics=None,
    )
    coarse = replace(
        short,
        transient=replace(
            short.transient,
            output_step=2.0e-6,
            solver="dop853",
        ),
    )

    fine_result = HybridSimulator(short).simulate()
    coarse_result = HybridSimulator(coarse).simulate()
    fine_events = [
        event.as_dict() for event in fine_result.events if event.kind == "scheduled_gate"
    ]
    coarse_events = [
        event.as_dict() for event in coarse_result.events if event.kind == "scheduled_gate"
    ]
    assert fine_events == coarse_events
    assert fine_result.modulation_cycles == coarse_result.modulation_cycles
    assert len(fine_result.modulation_cycles) == 2
    assert all(
        not (interval["gates"][f"bridge.{phase}.high"] and interval["gates"][f"bridge.{phase}.low"])
        for interval in fine_result.mode_intervals
        for phase in ("a", "b", "c")
    )


def test_nonzero_modulation_duties_follow_the_independent_sample_formula() -> None:
    modulation_index = 0.625
    reference_phase = 3.0 * Q
    cycle = _build_gate_schedule(
        _spwm_controls(
            modulation_index=modulation_index,
            dead_time=0.0,
            reference_phase=reference_phase,
        ),
        0.0,
        PERIOD,
    ).modulation_cycles[0]

    sample = 0.5 * PERIOD
    angle = math.tau * (((sample - reference_phase) * FREQUENCY / RATIO) % 1.0)
    expected_references = {
        "a": math.sin(angle),
        "b": math.sin(angle - 2.0 * math.pi / 3.0),
        "c": math.sin(angle + 2.0 * math.pi / 3.0),
    }
    assert cycle["electrical_angle_rad"] == angle
    assert cycle["references"] == expected_references
    assert cycle["duties"] == {
        phase: 0.5 * (1.0 + modulation_index * reference)
        for phase, reference in expected_references.items()
    }
