from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pear_sim.solver as solver_module
from pear_sim.errors import SimulationError
from pear_sim.model import PwmControl, load_model
from pear_sim.solver import (
    HybridSimulator,
    _initial_gate_values,
    _pwm_value,
    scheduled_edges,
)


ROOT = Path(__file__).parents[1]


class _StableCircuit:
    @staticmethod
    def stable_mode(
        _gates: dict[str, bool],
        diodes: dict[str, bool],
        _state: np.ndarray,
        _tolerance: float,
        time: float | None = None,
    ) -> tuple[dict[str, bool], object]:
        return dict(diodes), object()


def _scheduler_simulator(controls: dict[str, PwmControl]) -> HybridSimulator:
    simulator = object.__new__(HybridSimulator)
    simulator.model = SimpleNamespace(controls=controls, forced_gate_windows=())
    simulator.circuit = _StableCircuit()
    simulator.guard_tolerance = 1e-10
    return simulator


def test_pwm_uses_right_continuous_state_and_exact_open_closed_run_window() -> None:
    control = PwmControl("pwm", frequency=1_000.0, duty=0.4)
    start = -5e-13
    stop = 1e-3 - 5e-13

    assert _pwm_value(control, start) is False
    assert _pwm_value(control, 0.0) is True
    assert _pwm_value(control, 0.4e-3) is False
    assert scheduled_edges({"pwm": control}, start, stop) == [
        (0.0, "pwm", True),
        (0.4e-3, "pwm", False),
    ]
    assert scheduled_edges({"pwm": control}, 0.0, 1e-3) == [
        (0.4e-3, "pwm", False),
        (1e-3, "pwm", True),
    ]


def test_exactly_coincident_edges_are_ordered_off_before_on() -> None:
    controls = {
        "a_on": PwmControl("a_on", frequency=1.0, duty=0.5, phase=0.5),
        "z_off": PwmControl("z_off", frequency=1.0, duty=0.5, phase=0.0),
    }

    assert scheduled_edges(controls, 0.0, 0.5) == [
        (0.5, "z_off", False),
        (0.5, "a_on", True),
    ]


def test_legacy_edges_within_eight_ulps_are_canonicalized_without_span_tolerance() -> None:
    boundary = 0.5
    nearby = boundary
    for _ in range(4):
        nearby = math.nextafter(nearby, math.inf)
    controls = {
        "on": PwmControl("on", frequency=1.0, duty=0.25, phase=nearby),
        "off": PwmControl("off", frequency=1.0, duty=0.5, phase=0.0),
    }

    assert scheduled_edges(controls, 0.0, nearby) == [
        (boundary, "off", False),
        (boundary, "on", True),
    ]


def test_nearby_distinct_edges_are_not_merged_or_retimed() -> None:
    controls = {
        "first": PwmControl("first", frequency=1.0, duty=0.5),
        "second": PwmControl("second", frequency=1.0, duty=0.5),
    }
    simulator = _scheduler_simulator(controls)
    first_time = 0.5
    second_time = math.nextafter(first_time, math.inf)
    scheduled = [
        (first_time, "first", True),
        (second_time, "second", True),
    ]

    _, gates, _, _, events, edge_index = simulator._apply_scheduled_edges(
        first_time,
        np.empty(0),
        {"first": False, "second": False},
        {},
        scheduled,
        0,
    )

    assert edge_index == 1
    assert gates == {"first": True, "second": False}
    assert [event.as_dict() for event in events] == [
        {
            "time": first_time,
            "kind": "scheduled_gate",
            "target": "first",
            "previous": False,
            "current": True,
        }
    ]


def test_complementary_interlock_is_checked_after_the_atomic_timestamp_group() -> None:
    high = PwmControl(
        "leg.high",
        frequency=1.0,
        duty=0.5,
        kind="complementary_pwm",
        block_id="leg",
        role="high",
        partner_id="leg.low",
    )
    low = PwmControl(
        "leg.low",
        frequency=1.0,
        duty=0.5,
        kind="complementary_pwm",
        block_id="leg",
        role="low",
        partner_id="leg.high",
    )
    simulator = _scheduler_simulator({high.id: high, low.id: low})

    with pytest.raises(SimulationError, match="interlock violation"):
        simulator._assert_complementary_interlocks(
            {high.id: True, low.id: True},
            time=0.0,
        )

    _, gates, _, _, events, edge_index = simulator._apply_scheduled_edges(
        0.5,
        np.empty(0),
        {high.id: False, low.id: True},
        {},
        [
            (0.5, low.id, False),
            (0.5, high.id, True),
        ],
        0,
    )

    assert edge_index == 2
    assert gates == {high.id: True, low.id: False}
    assert [(event.target, event.current) for event in events] == [
        (low.id, False),
        (high.id, True),
    ]


def test_scheduler_rejects_unbounded_edge_materialization() -> None:
    control = PwmControl("pwm", frequency=1e308, duty=0.5)

    with pytest.raises(SimulationError, match="global limit"):
        scheduled_edges({"pwm": control}, 0.0, 1.0)


def test_simulator_rejects_an_event_log_larger_than_portable_consumers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(solver_module, "MAX_RUN_EVENT_ROWS", 0)

    with pytest.raises(SimulationError, match="portable artifact limit"):
        HybridSimulator(load_model(ROOT / "examples" / "buck.yaml")).simulate()


def test_scheduler_rejects_an_unrepresentable_absolute_cycle_origin() -> None:
    control = PwmControl("pwm", frequency=1.0, duty=0.5)

    with pytest.raises(SimulationError, match="not representable"):
        scheduled_edges({"pwm": control}, 1e16, 1e16 + 2.0)


def test_positive_complementary_dead_time_must_survive_absolute_timestamp_rounding() -> None:
    common = {
        "frequency": 1.0,
        "kind": "complementary_pwm",
        "block_id": "leg",
        "dead_time": 1e-16,
        "command_duty": 0.5,
        "command_phase": 0.0,
        "normalized_phase": 0.0,
    }
    high = PwmControl(
        "leg.high",
        duty=0.5 - 1e-16,
        phase=1e-16,
        role="high",
        partner_id="leg.low",
        **common,
    )
    low = PwmControl(
        "leg.low",
        duty=0.5 - 1e-16,
        phase=0.5 + 1e-16,
        role="low",
        partner_id="leg.high",
        **common,
    )

    with pytest.raises(SimulationError, match="dead time.*not representable"):
        scheduled_edges({high.id: high, low.id: low}, 1.0, 2.0)


def test_complementary_initial_state_uses_the_absolute_scheduler_boundary() -> None:
    frequency = 33156.55335968379
    period = 1.0 / frequency
    duty = 0.125
    dead_time = 3.0994415283203125e-6
    phase = 9.5367431640625e-7
    common = {
        "frequency": frequency,
        "kind": "complementary_pwm",
        "block_id": "leg",
        "dead_time": dead_time,
        "command_duty": duty,
        "command_phase": phase,
        "normalized_phase": phase % period,
    }
    high = PwmControl(
        "leg.high",
        duty=duty - dead_time / period,
        phase=(phase % period) + dead_time,
        role="high",
        partner_id="leg.low",
        **common,
    )
    low = PwmControl(
        "leg.low",
        duty=(1.0 - duty) - dead_time / period,
        phase=(phase % period) + duty * period + dead_time,
        role="low",
        partner_id="leg.high",
        **common,
    )
    controls = {high.id: high, low.id: low}
    command_rising = phase + 4 * period
    start = command_rising + dead_time

    assert _initial_gate_values(controls, start) == {
        high.id: True,
        low.id: False,
    }
    assert scheduled_edges(controls, start, start + period)[0] == (
        command_rising + duty * period,
        high.id,
        False,
    )
