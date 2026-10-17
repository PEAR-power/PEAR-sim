"""Topology-specific observations derived from accepted hybrid solver segments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from pear_sim.analysis.analytic_checks import buck_parameters, match_ideal_buck
from pear_sim.spec.canonical_topologies import (
    BoostBoundaryBenchmarkError,
    InvertingBuckBoostBoundaryBenchmarkError,
    extract_boost_boundary_parameters,
    extract_inverting_buck_boost_boundary_parameters,
)
from pear_sim.core.circuit import STATE_CONSTRAINT_TOLERANCE
from pear_sim.errors import SimulationError
from pear_sim.spec.model import Component, Model, PwmControl

if TYPE_CHECKING:
    from pear_sim.core.solver import SimulationResult


MODE_COVERAGE_SCHEMA = "pear-sim/mode-coverage/v0.1"
MODE_COVERAGE_KIND = "hybrid-mode-coverage-observation"
_MODE_NAMES = ("on", "freewheel", "idle", "unclassified")
_CLASSIFICATION_METHOD = "accepted_hybrid_segments_adjacent_equal_modes_merged"
_BOUNDARY_CONVENTION = "[start_s,end_s), with the simulation endpoint owned by the final interval"


@dataclass(frozen=True)
class _CoverageFixture:
    topology_id: str
    match_reason: str
    source: Component
    switch: Component
    diode: Component
    inductor: Component
    capacitor: Component
    load: Component
    pwm: PwmControl
    parameters_si: dict[str, float]


def _match_coverage_fixture(model: Model) -> tuple[_CoverageFixture | None, str]:
    buck, buck_reason = match_ideal_buck(model)
    if buck is not None:
        return (
            _CoverageFixture(
                topology_id="canonical_single_switch_buck",
                match_reason="Exact canonical ideal single-switch buck topology matched.",
                source=buck.source,
                switch=buck.switch,
                diode=buck.diode,
                inductor=buck.inductor,
                capacitor=buck.capacitor,
                load=buck.load,
                pwm=buck.pwm,
                parameters_si=buck_parameters(buck),
            ),
            "",
        )
    try:
        boost_parameters = extract_boost_boundary_parameters(model)
    except BoostBoundaryBenchmarkError as exc:
        boost_reason = str(exc)
    else:
        by_type = {component.type: component for component in model.components}
        pwm = next(iter(model.controls.values()))
        return (
            _CoverageFixture(
                topology_id="canonical_single_switch_boost",
                match_reason="Exact canonical ideal single-switch boost topology matched.",
                source=by_type["voltage_source"],
                switch=by_type["ideal_switch"],
                diode=by_type["diode"],
                inductor=by_type["inductor"],
                capacitor=by_type["capacitor"],
                load=by_type["resistor"],
                pwm=pwm,
                parameters_si=boost_parameters.as_dict(),
            ),
            "",
        )
    try:
        buck_boost_parameters = extract_inverting_buck_boost_boundary_parameters(model)
    except InvertingBuckBoostBoundaryBenchmarkError as exc:
        return (
            None,
            f"buck: {buck_reason}; boost: {boost_reason}; inverting buck-boost: {exc}",
        )
    by_type = {component.type: component for component in model.components}
    pwm = next(iter(model.controls.values()))
    return (
        _CoverageFixture(
            topology_id="canonical_single_switch_inverting_buck_boost",
            match_reason=(
                "Exact canonical ideal single-switch inverting buck-boost topology matched."
            ),
            source=by_type["voltage_source"],
            switch=by_type["ideal_switch"],
            diode=by_type["diode"],
            inductor=by_type["inductor"],
            capacitor=by_type["capacitor"],
            load=by_type["resistor"],
            pwm=pwm,
            parameters_si=buck_boost_parameters.as_dict(),
        ),
        "",
    )


def build_mode_coverage(result: SimulationResult) -> dict[str, Any]:
    """Build a conservative, non-validating mode-coverage observation.

    Only the exact canonical single-switch buck, boost, and inverting buck-boost are
    classified. Durations come from the solver's accepted mode segments, never from
    output samples or event-log reconstruction.
    """

    claim = {
        "classification": "verification-coverage-observation",
        "validation_claim": False,
        "statement": (
            "Observed accepted hybrid modes and event coverage only; this is not a "
            "physical-validation or general DCM-correctness claim."
        ),
    }
    topology, reason = _match_coverage_fixture(result.model)
    if topology is None:
        return {
            "schema": MODE_COVERAGE_SCHEMA,
            "kind": MODE_COVERAGE_KIND,
            "claim": claim,
            "status": "not_applicable",
            "recognition": {
                "topology": (
                    "canonical_single_switch_buck_or_boost_or_inverting_buck_boost"
                ),
                "status": "not_applicable",
                "reason": reason,
                "subjects": None,
            },
            "window": None,
            "classification": None,
            "summary": None,
            "diode_conduction": None,
            "inductor_current": None,
            "event_timing": None,
            "intervals": None,
        }

    settings = result.model.transient
    period = topology.pwm.period
    cycle_origin = topology.pwm.phase
    tolerance = max(1e-15, (settings.stop - settings.start) * 1e-12)
    window_end = settings.stop
    simulated_duration = settings.stop - settings.start
    reactive_components = tuple(
        component
        for component in result.model.components
        if component.type in {"capacitor", "inductor", "sat_inductor"}
    )
    derived_initial_conditions = bool(reactive_components) and all(
        not component.initial_state_explicit for component in reactive_components
    )
    if simulated_duration <= period + tolerance and derived_initial_conditions:
        window_start = settings.start
        window_selection = (
            "complete_run_at_most_one_switching_period_with_derived_initial_conditions"
        )
    else:
        window_start = 0.5 * (settings.start + settings.stop)
        window_selection = "last_half_of_multi_period_run"
    window_duration = window_end - window_start

    raw_intervals = _normalized_intervals(
        result.mode_intervals,
        window_start,
        window_end,
        period,
        cycle_origin,
        tolerance,
        topology.pwm.id,
        topology.diode.id,
        topology.inductor.id,
    )
    intervals = _merge_adjacent(raw_intervals, tolerance)
    covered_duration = sum(float(interval["duration_s"]) for interval in intervals)
    if not math.isclose(
        covered_duration,
        window_duration,
        rel_tol=1e-10,
        abs_tol=tolerance * 10.0,
    ):
        raise SimulationError(
            "Accepted mode intervals do not cover the mode-observation window: "
            f"got {covered_duration:.12g} s, expected {window_duration:.12g} s"
        )

    event_rows = _event_rows(
        result,
        window_start,
        window_end,
        cycle_origin,
        period,
        tolerance,
        topology.pwm.id,
        topology.diode.id,
    )
    _bind_boundary_events(intervals, event_rows, tolerance)
    for index, interval in enumerate(intervals):
        interval["index"] = index

    durations = {mode: 0.0 for mode in _MODE_NAMES}
    diode_duration = 0.0
    for interval in intervals:
        duration = float(interval["duration_s"])
        durations[str(interval["mode"])] += duration
        if interval["diode_conducting"] is True:
            diode_duration += duration
    fractions = {mode: durations[mode] / window_duration for mode in _MODE_NAMES}
    natural_events = [row for row in event_rows if row["kind"] == "natural_commutation"]
    turn_off_events = [
        row
        for row in natural_events
        if row["target"] == topology.diode.id
        and row["previous"] is True
        and row["current"] is False
    ]
    minimum, minimum_time, extrema_method = _minimum_inductor_current(
        result,
        topology.inductor.id,
        window_start,
        window_end,
        tolerance,
    )

    first_cycle = _cycle_position(window_start, cycle_origin, period, tolerance)[0]
    last_cycle = _cycle_position(window_end, cycle_origin, period, tolerance)[0]
    return {
        "schema": MODE_COVERAGE_SCHEMA,
        "kind": MODE_COVERAGE_KIND,
        "claim": claim,
        "status": "observed",
        "recognition": {
            "topology": topology.topology_id,
            "status": "matched",
            "reason": topology.match_reason,
            "subjects": {
                "control_id": topology.pwm.id,
                "source_id": topology.source.id,
                "switch_id": topology.switch.id,
                "freewheel_diode_id": topology.diode.id,
                "inductor_id": topology.inductor.id,
                "capacitor_id": topology.capacitor.id,
                "load_id": topology.load.id,
            },
            "parameters_si": topology.parameters_si,
        },
        "window": {
            "start_s": window_start,
            "end_s": window_end,
            "duration_s": window_duration,
            "selection": window_selection,
            "switching_period_s": period,
            "cycle_origin_s": cycle_origin,
            "first_cycle_index": first_cycle,
            "end_boundary_cycle_index": last_cycle,
        },
        "classification": {
            "method": _CLASSIFICATION_METHOD,
            "boundary_convention": _BOUNDARY_CONVENTION,
            "rules": {
                "on": "gate_on=true and diode_conducting=false",
                "freewheel": "gate_on=false and diode_conducting=true",
                "idle": (
                    "gate_on=false and diode_conducting=false and the recognized "
                    "inductor is constrained at zero current"
                ),
                "unclassified": "all other accepted state combinations",
            },
        },
        "summary": {
            "interval_count": len(intervals),
            "covered_duration_s": covered_duration,
            "duration_by_mode_s": durations,
            "fraction_by_mode": fractions,
            "natural_commutation": {
                "exercised": bool(natural_events),
                "event_count": len(natural_events),
                "diode_turn_off_event_count": len(turn_off_events),
            },
        },
        "diode_conduction": {
            "component_id": topology.diode.id,
            "conducting_duration_s": diode_duration,
            "observed_duration_s": window_duration,
            "fraction": diode_duration / window_duration,
            "method": "sum accepted mode intervals with diode_conducting=true",
        },
        "inductor_current": {
            "component_id": topology.inductor.id,
            "sign_convention": "positive_p_to_n",
            "minimum_a": minimum,
            "minimum_time_s": minimum_time,
            "maximum_forbidden_reverse_current_a": (
                None if minimum is None else max(0.0, -minimum)
            ),
            "definition": "max(0, -minimum_a)",
            "state_constraint_tolerance_a": STATE_CONSTRAINT_TOLERANCE,
            "method": extrema_method,
        },
        "event_timing": {
            "time_reference": "absolute_simulation_time_s",
            "cycle_reference": "recognized PWM origin and period",
            "event_merge_tolerance_s": tolerance,
            "events": event_rows,
        },
        "intervals": intervals,
    }


def _normalized_intervals(
    source: list[dict[str, Any]],
    window_start: float,
    window_end: float,
    period: float,
    origin: float,
    tolerance: float,
    control_id: str,
    diode_id: str,
    inductor_id: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    previous_end: float | None = None
    for raw in source:
        start = max(window_start, float(raw["start_s"]))
        end = min(window_end, float(raw["end_s"]))
        # Solver mode intervals are authoritative accepted integration segments.
        # A strictly positive interval must not disappear merely because it is
        # shorter than the event-binding tolerance.  That occurs at BCM when a
        # natural diode turn-off is immediately followed by a scheduled edge; if
        # one tiny interval per cycle were discarded, their durations would
        # accumulate into an artificial coverage gap over a long observation
        # window.
        if end <= start:
            continue
        if previous_end is not None and start < previous_end - tolerance:
            raise SimulationError("Accepted mode intervals overlap or are out of order")
        gates = raw.get("gates")
        diodes = raw.get("diodes")
        if not isinstance(gates, dict) or control_id not in gates:
            raise SimulationError(f"Mode interval lacks gate state for control {control_id!r}")
        if not isinstance(diodes, dict) or diode_id not in diodes:
            raise SimulationError(f"Mode interval lacks diode state for {diode_id!r}")
        gate_on = bool(gates[control_id])
        diode_conducting = bool(diodes[diode_id])
        constraints = tuple(sorted(str(value) for value in raw.get("constrained_inductors", [])))

        for part_start, part_end in _split_at_cycle_boundaries(
            start, end, origin, period, tolerance
        ):
            cycle_index = _cycle_position(0.5 * (part_start + part_end), origin, period, tolerance)[
                0
            ]
            mode = _classify_mode(
                gate_on,
                diode_conducting,
                inductor_id in constraints,
            )
            normalized.append(
                {
                    "start_s": part_start,
                    "end_s": part_end,
                    "duration_s": part_end - part_start,
                    "mode": mode,
                    "gate_on": gate_on,
                    "diode_conducting": diode_conducting,
                    "constrained_inductors": list(constraints),
                    "mode_signature": raw.get("mode_signature"),
                    "cycle_index": cycle_index,
                    "start_event_indices": [],
                    "end_event_indices": [],
                }
            )
        previous_end = end
    return normalized


def _split_at_cycle_boundaries(
    start: float,
    end: float,
    origin: float,
    period: float,
    tolerance: float,
) -> list[tuple[float, float]]:
    relative_start = (start - origin) / period
    first_boundary_index = math.floor(relative_start) + 1
    boundaries: list[float] = []
    index = first_boundary_index
    while True:
        boundary = origin + index * period
        if boundary >= end - tolerance:
            break
        if boundary > start + tolerance:
            boundaries.append(boundary)
        index += 1
    points = [start, *boundaries, end]
    return list(zip(points[:-1], points[1:], strict=True))


def _classify_mode(
    gate_on: bool,
    diode_conducting: bool,
    inductor_constrained: bool,
) -> str:
    if gate_on and not diode_conducting and not inductor_constrained:
        return "on"
    if not gate_on and diode_conducting and not inductor_constrained:
        return "freewheel"
    if not gate_on and not diode_conducting and inductor_constrained:
        return "idle"
    return "unclassified"


def _merge_adjacent(intervals: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    identity_fields = (
        "mode",
        "gate_on",
        "diode_conducting",
        "constrained_inductors",
        "mode_signature",
        "cycle_index",
    )
    for interval in intervals:
        if (
            merged
            and all(merged[-1][field] == interval[field] for field in identity_fields)
            and math.isclose(
                float(merged[-1]["end_s"]),
                float(interval["start_s"]),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        ):
            merged[-1]["end_s"] = interval["end_s"]
            merged[-1]["duration_s"] = float(merged[-1]["end_s"]) - float(merged[-1]["start_s"])
        else:
            merged.append(dict(interval))
    return merged


def _event_rows(
    result: SimulationResult,
    start: float,
    end: float,
    origin: float,
    period: float,
    tolerance: float,
    control_id: str,
    diode_id: str,
) -> list[dict[str, Any]]:
    relevant = {control_id, diode_id}
    selected = [
        event
        for event in result.events
        if event.target in relevant and start - tolerance <= event.time <= end + tolerance
    ]
    rows: list[dict[str, Any]] = []
    for index, event in enumerate(selected):
        cycle_index, offset, phase = _cycle_position(event.time, origin, period, tolerance)
        rows.append(
            {
                "index": index,
                "time_s": event.time,
                "kind": event.kind,
                "target": event.target,
                "previous": event.previous,
                "current": event.current,
                "cycle_index": cycle_index,
                "offset_from_cycle_start_s": offset,
                "phase_fraction": phase,
            }
        )
    return rows


def _cycle_position(
    time: float,
    origin: float,
    period: float,
    tolerance: float,
) -> tuple[int, float, float]:
    relative = (time - origin) / period
    nearest = round(relative)
    if abs(relative - nearest) <= tolerance / period:
        relative = float(nearest)
    cycle_index = math.floor(relative)
    offset = time - (origin + cycle_index * period)
    if abs(offset) <= tolerance:
        offset = 0.0
    elif abs(offset - period) <= tolerance:
        cycle_index += 1
        offset = 0.0
    return cycle_index, offset, offset / period


def _bind_boundary_events(
    intervals: list[dict[str, Any]],
    events: list[dict[str, Any]],
    tolerance: float,
) -> None:
    for interval in intervals:
        interval["start_event_indices"] = [
            int(event["index"])
            for event in events
            if math.isclose(
                float(event["time_s"]),
                float(interval["start_s"]),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        ]
        interval["end_event_indices"] = [
            int(event["index"])
            for event in events
            if math.isclose(
                float(event["time_s"]),
                float(interval["end_s"]),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        ]


def _minimum_inductor_current(
    result: SimulationResult,
    component_id: str,
    start: float,
    end: float,
    tolerance: float,
) -> tuple[float | None, float | None, str]:
    aggregate = getattr(result, "state_extrema", None)
    if isinstance(aggregate, dict):
        states = aggregate.get("states")
        record = states.get(component_id) if isinstance(states, dict) else None
        aggregate_start = aggregate.get("start_s")
        aggregate_end = aggregate.get("end_s")
        if (
            isinstance(record, dict)
            and record.get("minimum") is not None
            and record.get("minimum_time_s") is not None
            and isinstance(aggregate_start, (int, float))
            and isinstance(aggregate_end, (int, float))
            and math.isclose(float(aggregate_start), start, rel_tol=0.0, abs_tol=tolerance)
            and math.isclose(float(aggregate_end), end, rel_tol=0.0, abs_tol=tolerance)
        ):
            return (
                float(record["minimum"]),
                float(record["minimum_time_s"]),
                str(aggregate.get("method", "event_aligned_dense_trajectory_extrema")),
            )

    candidates: list[tuple[float, float]] = []
    dense_observations = 0
    for interval in result.mode_intervals:
        raw_start = float(interval["start_s"])
        raw_end = float(interval["end_s"])
        extrema = interval.get("state_extrema")
        if (
            isinstance(extrema, dict)
            and raw_start >= start - tolerance
            and raw_end <= end + tolerance
        ):
            record = extrema.get(component_id)
            if isinstance(record, dict) and record.get("minimum") is not None:
                candidates.append((float(record["minimum"]), float(record["minimum_time_s"])))
                dense_observations += 1
        for field, time_field in (("state_start", raw_start), ("state_end", raw_end)):
            values = interval.get(field)
            if (
                isinstance(values, dict)
                and component_id in values
                and start - tolerance <= time_field <= end + tolerance
            ):
                candidates.append((float(values[component_id]), time_field))

    expression = f"i({component_id})"
    values = result.signals.get(expression)
    if values is not None:
        mask = (result.time >= start - tolerance) & (result.time <= end + tolerance)
        indices = np.flatnonzero(mask)
        candidates.extend((float(values[index]), float(result.time[index])) for index in indices)

    if not candidates:
        return None, None, "not_available_without_state_or_requested_current_observations"
    minimum, minimum_time = min(candidates, key=lambda item: (item[0], item[1]))
    if dense_observations:
        method = "event_aligned_dense_trajectory_extrema"
    elif values is not None:
        method = "event_aligned_mode_boundaries_and_requested_output_grid"
    else:
        method = "event_aligned_mode_boundary_states"
    return minimum, minimum_time, method
