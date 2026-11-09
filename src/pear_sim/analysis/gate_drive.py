# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Deterministic human-facing gate-drive observation builder.

The raw scheduler event log remains authoritative.  This module produces a small,
bounded presentation projection: an explicit gate-output-to-device map and the first
command period of every logical PWM block.  It deliberately does not infer bridge
legs, phases, or device roles from component or node names.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
from typing import Any

from pear_sim.run.contracts import model_hashes
from pear_sim.spec.model import Model, PwmControl
from pear_sim.core.solver import SimulationResult


GATE_DRIVE_SCHEMA = "pear-sim/gate-drive-observation/v0.1"
GATE_DRIVE_KIND = "gate-drive-observation"

_CONTROL_KINDS = frozenset(
    {
        "pwm",
        "complementary_pwm",
        "sampled_pi_pwm",
        "sampled_dq_current_pwm",
        "signal_pwm",
        "signal_svpwm",
    }
)
_ROLE_ORDER = {"single": 0, "high": 1, "low": 2}


class GateDriveError(ValueError):
    """Raised when a simulation result cannot support an exact gate-drive view."""

    code = "analysis.gate_drive_invalid"


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GateDriveError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise GateDriveError(f"{name} must be a finite number")
    return result


def _control_metadata(control: PwmControl) -> dict[str, Any]:
    kind = control.kind
    if kind not in _CONTROL_KINDS:
        raise GateDriveError(f"Gate output {control.id!r} has unsupported kind {kind!r}")

    if kind == "sampled_dq_current_pwm":
        frequency = _finite_number(control.frequency, name=f"{control.id}.frequency")
        if frequency <= 0.0:
            raise GateDriveError(f"{control.id}.frequency must be positive")
        dead_time = _finite_number(control.dead_time, name=f"{control.id}.dead_time")
        if dead_time < 0.0:
            raise GateDriveError(f"{control.id}.dead_time must be non-negative")
        return {
            "gate_output_id": control.id,
            "logical_control_id": f"{control.block_id}.{control.phase_label}",
            "control_type": kind,
            "role": control.role,
            "partner_gate_output_id": control.partner_id,
            "frequency_hz": frequency,
            "period_s": 1.0 / frequency,
            # Closed loop: per-cycle center-aligned duties are decided by the
            # sampled dq controller; the record is control-trace.jsonl.
            "effective_duty": None,
            "effective_phase_s": 0.0,
            "command_duty": None,
            "command_phase_s": 0.0,
            "dead_time_s": dead_time,
            "command_source": "closed_loop_sampled_dq",
        }

    if kind == "signal_svpwm":
        frequency = _finite_number(control.frequency, name=f"{control.id}.frequency")
        if frequency <= 0.0:
            raise GateDriveError(f"{control.id}.frequency must be positive")
        return {
            "gate_output_id": control.id,
            "logical_control_id": f"{control.block_id}.{control.phase_label}",
            "control_type": kind,
            "role": control.role,
            "partner_gate_output_id": control.partner_id,
            "frequency_hz": frequency,
            "period_s": 1.0 / frequency,
            "effective_duty": None,
            "effective_phase_s": 0.0,
            "command_duty": None,
            "command_phase_s": 0.0,
            "dead_time_s": _finite_number(
                control.dead_time, name=f"{control.id}.dead_time"
            ),
            "command_source": "declared_signal_graph_space_vector",
        }

    if kind == "signal_pwm":
        frequency = _finite_number(control.frequency, name=f"{control.id}.frequency")
        if frequency <= 0.0:
            raise GateDriveError(f"{control.id}.frequency must be positive")
        return {
            "gate_output_id": control.id,
            "logical_control_id": control.id,
            "control_type": kind,
            "role": "single",
            "partner_gate_output_id": None,
            "frequency_hz": frequency,
            "period_s": 1.0 / frequency,
            # The duty is decided per tick by the declared signal graph; the
            # authoritative record is signal-trace.jsonl.
            "effective_duty": None,
            "effective_phase_s": 0.0,
            "command_duty": None,
            "command_phase_s": 0.0,
            "dead_time_s": 0.0,
            "command_source": "declared_signal_graph",
        }

    if kind == "sampled_pi_pwm":
        frequency = _finite_number(control.frequency, name=f"{control.id}.frequency")
        if frequency <= 0.0:
            raise GateDriveError(f"{control.id}.frequency must be positive")
        return {
            "gate_output_id": control.id,
            "logical_control_id": control.id,
            "control_type": kind,
            "role": "single",
            "partner_gate_output_id": None,
            "frequency_hz": frequency,
            "period_s": 1.0 / frequency,
            # Closed loop: the duty command is decided per cycle by the sampled
            # controller; the authoritative record is control-trace.jsonl.
            "effective_duty": None,
            "effective_phase_s": 0.0,
            "command_duty": None,
            "command_phase_s": 0.0,
            "dead_time_s": 0.0,
            "command_source": "closed_loop_sampled_pi",
        }

    if kind == "pwm":
        block_id = control.id
        role = "single"
        partner_id = None
        command_duty = control.duty
        command_phase = control.phase
        dead_time = 0.0
    else:
        block_id = control.block_id
        role = control.role
        partner_id = control.partner_id
        command_duty = control.command_duty
        command_phase = control.command_phase
        dead_time = control.dead_time
        if not isinstance(block_id, str) or not block_id:
            raise GateDriveError(
                f"Complementary gate output {control.id!r} has no logical block id"
            )
        if role not in {"high", "low"}:
            raise GateDriveError(
                f"Complementary gate output {control.id!r} has invalid role {role!r}"
            )
        if not isinstance(partner_id, str) or not partner_id:
            raise GateDriveError(
                f"Complementary gate output {control.id!r} has no partner id"
            )
        if command_duty is None or command_phase is None:
            raise GateDriveError(
                f"Complementary gate output {control.id!r} lacks command timing metadata"
            )

    frequency = _finite_number(control.frequency, name=f"{control.id}.frequency")
    effective_duty = _finite_number(control.duty, name=f"{control.id}.duty")
    effective_phase = _finite_number(control.phase, name=f"{control.id}.phase")
    command_duty_value = _finite_number(
        command_duty,
        name=f"{control.id}.command_duty",
    )
    command_phase_value = _finite_number(
        command_phase,
        name=f"{control.id}.command_phase",
    )
    dead_time_value = _finite_number(dead_time, name=f"{control.id}.dead_time")
    if frequency <= 0.0:
        raise GateDriveError(f"{control.id}.frequency must be positive")
    if not 0.0 <= effective_duty <= 1.0:
        raise GateDriveError(f"{control.id}.duty must be in [0, 1]")
    if not 0.0 <= command_duty_value <= 1.0:
        raise GateDriveError(f"{control.id}.command_duty must be in [0, 1]")
    if dead_time_value < 0.0:
        raise GateDriveError(f"{control.id}.dead_time must be non-negative")

    return {
        "gate_output_id": control.id,
        "logical_control_id": block_id,
        "control_type": kind,
        "role": role,
        "partner_gate_output_id": partner_id,
        "frequency_hz": frequency,
        "period_s": 1.0 / frequency,
        "effective_duty": effective_duty,
        "effective_phase_s": effective_phase,
        "command_duty": command_duty_value,
        "command_phase_s": command_phase_value,
        "dead_time_s": dead_time_value,
    }


def _gate_outputs(model: Model) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    devices_by_output: dict[str, list[dict[str, str]]] = {
        control_id: []
        for control_id in (*model.controls, *model.sampled_controls)
    }
    for component in sorted(model.components, key=lambda item: item.id):
        if component.type != "ideal_switch":
            continue
        if component.control not in devices_by_output:
            raise GateDriveError(
                f"Switch {component.id!r} references unknown gate output {component.control!r}"
            )
        devices_by_output[component.control].append(
            {
                "device_id": component.id,
                "p": component.p,
                "n": component.n,
            }
        )

    outputs: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    all_controls = list(model.controls.values()) + list(
        model.sampled_controls.values()
    )
    for control in sorted(all_controls, key=lambda item: item.id):
        record = _control_metadata(control)
        record["devices"] = devices_by_output[control.id]
        record["device_ids"] = [item["device_id"] for item in devices_by_output[control.id]]
        outputs.append(record)
        by_id[control.id] = record

    for record in outputs:
        partner_id = record["partner_gate_output_id"]
        if partner_id is None:
            continue
        partner = by_id.get(partner_id)
        if partner is None:
            raise GateDriveError(
                f"Gate output {record['gate_output_id']!r} references unknown partner "
                f"{partner_id!r}"
            )
        if partner["partner_gate_output_id"] != record["gate_output_id"]:
            raise GateDriveError(
                f"Gate outputs {record['gate_output_id']!r} and {partner_id!r} are not "
                "reciprocal partners"
            )
        if partner["logical_control_id"] != record["logical_control_id"]:
            raise GateDriveError("Complementary partner outputs belong to different blocks")

    return outputs, by_id


def _logical_controls(
    outputs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for output in outputs:
        grouped.setdefault(output["logical_control_id"], []).append(output)

    logical: list[dict[str, Any]] = []
    for block_id, members in sorted(grouped.items()):
        members.sort(key=lambda item: (_ROLE_ORDER[item["role"]], item["gate_output_id"]))
        kinds = {item["control_type"] for item in members}
        frequencies = {item["frequency_hz"] for item in members}
        command_duties = {item["command_duty"] for item in members}
        command_phases = {item["command_phase_s"] for item in members}
        dead_times = {item["dead_time_s"] for item in members}
        if any(
            len(values) != 1
            for values in (kinds, frequencies, command_duties, command_phases, dead_times)
        ):
            raise GateDriveError(f"Logical control {block_id!r} has inconsistent output metadata")

        kind = next(iter(kinds))
        roles = [item["role"] for item in members]
        if kind in {"pwm", "sampled_pi_pwm", "signal_pwm"}:
            if len(members) != 1 or roles != ["single"]:
                raise GateDriveError(
                    f"PWM logical control {block_id!r} must have one single output"
                )
        elif kind == "sampled_dq_current_pwm":
            if len(members) != 2 or set(roles) != {"high", "low"}:
                raise GateDriveError(
                    f"Sampled dq logical control {block_id!r} must have high and "
                    "low outputs"
                )
        elif len(members) != 2 or set(roles) != {"high", "low"}:
            raise GateDriveError(
                f"Complementary logical control {block_id!r} must have high and low outputs"
            )

        frequency = next(iter(frequencies))
        logical.append(
            {
                "logical_control_id": block_id,
                "control_type": kind,
                "frequency_hz": frequency,
                "period_s": 1.0 / frequency,
                "command_duty": next(iter(command_duties)),
                "command_phase_s": next(iter(command_phases)),
                "dead_time_s": next(iter(dead_times)),
                "edge_policy": (
                    "turn_off_immediate_turn_on_after_dead_time"
                    if kind in {"complementary_pwm", "sampled_dq_current_pwm"}
                    else "single_pwm_edges"
                ),
                "interlock": (
                    "intrinsic_non_disableable"
                    if kind in {"complementary_pwm", "sampled_dq_current_pwm"}
                    else "not_applicable"
                ),
                "gate_output_ids": [item["gate_output_id"] for item in members],
            }
        )
    return logical


def _initial_gate_states(result: SimulationResult) -> dict[str, bool]:
    expected = set(result.model.controls) | set(result.model.sampled_controls)
    if not result.mode_intervals:
        if expected:
            raise GateDriveError("Simulation result has controls but no accepted mode interval")
        return {}
    first = min(
        result.mode_intervals,
        key=lambda item: _finite_number(item.get("start_s"), name="mode interval start_s"),
    )
    start = _finite_number(first.get("start_s"), name="first mode interval start_s")
    if start != result.model.transient.start:
        raise GateDriveError("First accepted mode interval does not start at transient.start")
    raw_gates = first.get("gates")
    if not isinstance(raw_gates, Mapping):
        raise GateDriveError("First accepted mode interval has no gate-state mapping")
    if set(raw_gates) != expected:
        raise GateDriveError("First accepted mode interval gate IDs differ from the model")
    states: dict[str, bool] = {}
    for control_id in sorted(expected):
        value = raw_gates[control_id]
        if not isinstance(value, bool):
            raise GateDriveError(f"Initial gate state for {control_id!r} is not boolean")
        states[control_id] = value
    return states


def _scheduled_transactions(
    result: SimulationResult,
    outputs_by_id: dict[str, dict[str, Any]],
    initial_states: dict[str, bool],
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    start = result.model.transient.start
    stop = result.model.transient.stop
    scheduled: list[tuple[int, Any]] = []
    previous_time = -math.inf
    for event_index, event in enumerate(result.events):
        if event.kind != "scheduled_gate":
            continue
        time = _finite_number(event.time, name=f"event {event_index} time")
        if time < previous_time:
            raise GateDriveError("Scheduled gate events are not time-monotonic")
        if not start <= time <= stop:
            raise GateDriveError(f"Scheduled gate event {event_index} is outside the run window")
        previous_time = time
        if event.target not in outputs_by_id:
            raise GateDriveError(
                f"Scheduled gate event {event_index} targets unknown output {event.target!r}"
            )
        if not isinstance(event.previous, bool) or not isinstance(event.current, bool):
            raise GateDriveError(f"Scheduled gate event {event_index} has non-boolean state")
        if event.previous == event.current:
            raise GateDriveError(f"Scheduled gate event {event_index} records no transition")
        scheduled.append((event_index, event))

    grouped: list[list[tuple[int, Any]]] = []
    for indexed_event in scheduled:
        if not grouped or indexed_event[1].time != grouped[-1][0][1].time:
            grouped.append([indexed_event])
        else:
            grouped[-1].append(indexed_event)

    states = dict(initial_states)
    _require_no_complementary_overlap(
        states,
        outputs_by_id,
        context="accepted initial gate state",
    )
    transactions: list[dict[str, Any]] = []
    for transaction_index, group in enumerate(grouped, start=1):
        time = float(group[0][1].time)
        seen_outputs: set[str] = set()
        changes: list[dict[str, Any]] = []
        for event_index, event in sorted(
            group,
            # The transaction is atomic; this order is presentation-only and makes
            # break-before-make intent legible without implying an intermediate mode.
            key=lambda item: (item[1].current, item[1].target),
        ):
            if event.target in seen_outputs:
                raise GateDriveError(
                    f"Atomic gate transaction at {time:.17g} s repeats {event.target!r}"
                )
            seen_outputs.add(event.target)
            output = outputs_by_id[event.target]
            if time > start and states[event.target] != event.previous:
                raise GateDriveError(
                    f"Gate event continuity mismatch for {event.target!r} at {time:.17g} s"
                )
            if time == start and initial_states[event.target] != event.current:
                raise GateDriveError(
                    f"Gate event at transient.start disagrees with accepted initial mode for "
                    f"{event.target!r}"
                )
            changes.append(
                {
                    "source_event_index": event_index,
                    "gate_output_id": event.target,
                    "logical_control_id": output["logical_control_id"],
                    "role": output["role"],
                    "previous": event.previous,
                    "current": event.current,
                    "edge_role": "turn_on" if event.current else "turn_off",
                    "device_ids": list(output["device_ids"]),
                }
            )
        for change in changes:
            states[change["gate_output_id"]] = change["current"]
        _require_no_complementary_overlap(
            states,
            outputs_by_id,
            context=f"atomic gate transaction at {time:.17g} s",
        )
        transactions.append(
            {
                "transaction_id": f"gate-transaction-{transaction_index:06d}",
                "time_s": time,
                "atomicity": "all_changes_applied_before_topology_stabilization",
                "source_event_indices": [item["source_event_index"] for item in changes],
                "changes": changes,
            }
        )
    return transactions, states


def _require_no_complementary_overlap(
    states: dict[str, bool],
    outputs_by_id: dict[str, dict[str, Any]],
    *,
    context: str,
) -> None:
    for output_id, output in outputs_by_id.items():
        if output["control_type"] != "complementary_pwm" or output["role"] != "high":
            continue
        partner_id = output["partner_gate_output_id"]
        if states.get(output_id, False) and states.get(partner_id, False):
            raise GateDriveError(
                f"Complementary gate overlap in {context}: {output_id!r} and "
                f"{partner_id!r} are both on"
            )


def _block_state(
    control: dict[str, Any],
    states: dict[str, bool],
    outputs_by_id: dict[str, dict[str, Any]],
) -> str:
    output_ids = control["gate_output_ids"]
    if control["control_type"] in {"pwm", "sampled_pi_pwm", "signal_pwm"}:
        return "on" if states[output_ids[0]] else "off"
    by_role = {outputs_by_id[output_id]["role"]: states[output_id] for output_id in output_ids}
    high = by_role["high"]
    low = by_role["low"]
    if high and low:
        return "overlap"
    if high:
        return "high_on"
    if low:
        return "low_on"
    return "dead_time"


def _timeline(
    control: dict[str, Any],
    transactions: list[dict[str, Any]],
    initial_states: dict[str, bool],
    outputs_by_id: dict[str, dict[str, Any]],
    *,
    run_start: float,
    run_stop: float,
    source_transaction_count: int,
) -> dict[str, Any]:
    window_start = run_start
    requested_end = run_start + control["period_s"]
    window_end = min(run_stop, requested_end)
    selection = (
        "first_command_period_from_run_start"
        if requested_end <= run_stop
        else "complete_run_shorter_than_one_command_period"
    )
    output_ids = set(control["gate_output_ids"])
    states = {output_id: initial_states[output_id] for output_id in sorted(output_ids)}
    selected_transactions: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    cursor = window_start

    def append_segment(end: float) -> None:
        if end <= cursor:
            return
        segments.append(
            {
                "start_s": cursor,
                "end_s": end,
                "duration_s": end - cursor,
                "state": _block_state(control, states, outputs_by_id),
                "gate_states": dict(states),
            }
        )

    for transaction in transactions:
        time = transaction["time_s"]
        relevant = [
            change for change in transaction["changes"] if change["gate_output_id"] in output_ids
        ]
        if not relevant:
            continue
        if time <= window_start:
            # The accepted initial mode is already the post-transaction state at
            # transient.start.  Earlier events are impossible for a valid result.
            continue
        if time >= window_end:
            break
        append_segment(time)
        for change in relevant:
            states[change["gate_output_id"]] = change["current"]
        selected_transactions.append(
            {
                "transaction_id": transaction["transaction_id"],
                "time_s": time,
                "offset_s": time - window_start,
                "atomicity": transaction["atomicity"],
                "source_event_indices": [
                    change["source_event_index"] for change in relevant
                ],
                "changes": relevant,
            }
        )
        cursor = time
    append_segment(window_end)

    state_counts = Counter(segment["state"] for segment in segments)
    state_durations: dict[str, float] = {}
    for segment in segments:
        state = segment["state"]
        state_durations[state] = state_durations.get(state, 0.0) + segment["duration_s"]
    return {
        "selection": selection,
        "retention": {
            "source_artifact": "events.jsonl",
            "source_scope": "full_run_[start,stop]",
            "source_atomic_transactions_for_control": source_transaction_count,
            "retained_atomic_transactions": len(selected_transactions),
            "truncated": len(selected_transactions) < source_transaction_count,
            "statement": (
                "This bounded projection retains transactions strictly inside the "
                "selected half-open display window; events.jsonl remains authoritative "
                "for the full run, including boundary events."
            ),
        },
        "window_start_s": window_start,
        "window_end_s": window_end,
        "window_duration_s": window_end - window_start,
        "boundary_convention": "segments_are_half_open_[start,end)",
        "boundary_coincidence_tolerance_s": 0.0,
        "initial_gate_states": {
            output_id: initial_states[output_id] for output_id in sorted(output_ids)
        },
        "transactions": selected_transactions,
        "segments": segments,
        "final_gate_states": dict(states),
        "observed_state_segment_counts": dict(sorted(state_counts.items())),
        "observed_duration_by_state_s": dict(sorted(state_durations.items())),
        "acceptance_status": "not_evaluated",
    }


def build_gate_drive_observation(result: SimulationResult) -> dict[str, Any]:
    """Build a deterministic gate map and bounded timeline from one accepted result.

    Exact-equal scheduled-event timestamps form one atomic transaction, matching the
    solver's apply-all-edges-before-stabilization semantics.  The returned timelines
    cover at most one command period per logical block; ``events.jsonl`` remains the
    complete event record.
    """

    run_start = _finite_number(result.model.transient.start, name="transient.start")
    run_stop = _finite_number(result.model.transient.stop, name="transient.stop")
    if run_stop <= run_start:
        raise GateDriveError("Transient stop must be greater than start")
    outputs, outputs_by_id = _gate_outputs(result.model)
    logical_controls = _logical_controls(outputs)
    initial_states = _initial_gate_states(result)
    transactions, final_states = _scheduled_transactions(
        result,
        outputs_by_id,
        initial_states,
    )
    transaction_counts_by_control: Counter[str] = Counter()
    for transaction in transactions:
        transaction_counts_by_control.update(
            {
                change["logical_control_id"]
                for change in transaction["changes"]
            }
        )
    for control in logical_controls:
        control["timeline"] = _timeline(
            control,
            transactions,
            initial_states,
            outputs_by_id,
            run_start=run_start,
            run_stop=run_stop,
            source_transaction_count=transaction_counts_by_control[
                control["logical_control_id"]
            ],
        )

    hashes = model_hashes(result.model)
    rendered_overlap_segments = sum(
        control["timeline"]["observed_state_segment_counts"].get("overlap", 0)
        for control in logical_controls
    )
    if rendered_overlap_segments:
        raise GateDriveError(
            "Accepted gate timeline contains a complementary high/low overlap segment"
        )
    return {
        "schema": GATE_DRIVE_SCHEMA,
        "kind": GATE_DRIVE_KIND,
        "claim": {
            "classification": "simulation-observation",
            "validation_claim": False,
            "hardware_sign_off_allowed": False,
            "statement": (
                "Recorded gate scheduling and explicit model bindings only; this artifact "
                "does not establish driver timing, device behavior, or hardware safety."
            ),
        },
        "source_binding": {
            "model_sha256": hashes["model_sha256"],
            "request_sha256": hashes["request_sha256"],
            "raw_event_artifact": "events.jsonl",
            "mapping_source": "resolved_model_explicit_ideal_switch_control_references",
            "timeline_source": "accepted_initial_mode_and_scheduled_gate_events",
            "transaction_policy": (
                "bit_identical_event_times_are_atomic_before_topology_stabilization"
            ),
        },
        "run_window": {
            "start_s": run_start,
            "stop_s": run_stop,
        },
        "run_gate_states": {
            "boundary_convention": "initial_is_right_limit;final_includes_events_at_stop",
            "initial": initial_states,
            "final": final_states,
        },
        "gate_outputs": outputs,
        "logical_controls": logical_controls,
        "summary": {
            "logical_control_count": len(logical_controls),
            "gate_output_count": len(outputs),
            "controlled_device_count": sum(len(output["device_ids"]) for output in outputs),
            "scheduled_gate_event_count": sum(
                len(transaction["changes"]) for transaction in transactions
            ),
            "atomic_transaction_count": len(transactions),
            "rendered_timeline_count": len(logical_controls),
            "rendered_overlap_segment_count": rendered_overlap_segments,
            "observed_interlock_violation_count": 0,
            "acceptance_status": "not_evaluated",
        },
    }


__all__ = [
    "GATE_DRIVE_KIND",
    "GATE_DRIVE_SCHEMA",
    "GateDriveError",
    "build_gate_drive_observation",
]
