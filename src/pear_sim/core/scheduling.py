"""Deterministic PWM/complementary/three-phase-SPWM gate scheduling.

Extracted verbatim from ``pear_sim.core.solver`` (restructuring stage 2, move 7).
This module materializes the scheduled gate-edge timeline and SPWM
modulation cycles the simulator executes; it performs no integration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from pear_sim.errors import SimulationError
from pear_sim.limits import (
    MAX_SCHEDULED_GATE_EDGES,
    MAX_SPWM_CARRIER_RATIO,
)
from pear_sim.spec.controls import npc_pair_duties, npc_pair_specs
from pear_sim.spec.model import PwmControl


_MAX_EXACT_SCHEDULE_CYCLE_INDEX = 2**52
_SPWM_PHASES = ("a", "b", "c")
_SPWM_SAMPLING = "offline_midpoint_sampled_center_aligned"
def _npc_pair_labels(levels: int) -> tuple[str, ...]:
    return tuple(pair_label for pair_label, _, _ in npc_pair_specs(levels))


@dataclass(frozen=True)
class _GateSchedule:
    """Materialized gate transitions and their modulation provenance.

    ``_scheduled_edges`` deliberately remains the compatibility view used by the
    original and complementary-PWM scheduler tests.  The richer internal result binds every generated
    three-phase SPWM carrier cell to the exact edge construction consumed by the
    simulator.
    """

    edges: list[tuple[float, str, bool]]
    modulation_cycles: list[dict[str, Any]]


@dataclass(frozen=True)
class _SpwmFrame:
    electrical_angle_rad: float
    references: tuple[float, float, float]
    duties: tuple[float, float, float]


def _ulp_close(left: float, right: float) -> bool:
    return abs(left - right) <= 8.0 * max(math.ulp(left), math.ulp(right))


def _cycle_position(time: float, phase: float, period: float, *, control_id: str) -> float:
    relative = time - phase
    if not math.isfinite(relative):
        raise SimulationError(
            f"PWM phase subtraction for control {control_id!r} is not representable"
        )
    position = relative % period
    if _ulp_close(position, 0.0) or _ulp_close(position, period):
        return 0.0
    return position


def _snap_run_boundary(time: float, start: float, stop: float) -> float:
    if time < start and _ulp_close(time, start):
        return start
    if time > stop and _ulp_close(time, stop):
        return stop
    return time


def _absolute_cycle_at_or_before(
    time: float,
    phase: float,
    period: float,
    *,
    control_id: str,
) -> tuple[int, float]:
    """Return the last absolute scheduler cycle boundary at or before ``time``."""

    relative = (time - phase) / period
    if not math.isfinite(relative) or abs(relative) > _MAX_EXACT_SCHEDULE_CYCLE_INDEX - 2:
        raise SimulationError(
            f"PWM schedule for control {control_id!r} is not representable "
            "at the requested absolute time origin"
        )

    cycle = math.floor(relative)
    # Division and the scheduler's multiply-add construction can round on opposite
    # sides of a cycle boundary.  Reconcile against the actual absolute timestamps
    # the scheduler emits instead of relying on modulo plus a fixed ULP allowance.
    for _ in range(4):
        rising = phase + cycle * period
        next_rising = phase + (cycle + 1) * period
        if not math.isfinite(rising) or not math.isfinite(next_rising) or next_rising <= rising:
            raise SimulationError(
                f"PWM schedule for control {control_id!r} loses distinct edge "
                "timestamps at the requested absolute time origin"
            )
        if rising > time:
            cycle -= 1
            continue
        if next_rising <= time:
            cycle += 1
            continue
        return cycle, rising
    raise SimulationError(
        f"PWM schedule for control {control_id!r} cannot resolve the absolute cycle "
        "at the requested time origin"
    )


def _pwm_value(control: PwmControl, time: float) -> bool:
    if control.duty <= 0.0:
        return False
    if control.duty >= 1.0:
        return True
    on_duration = control.duty * control.period
    _, rising = _absolute_cycle_at_or_before(
        time,
        control.phase,
        control.period,
        control_id=control.id,
    )
    falling = rising + on_duration
    _require_distinct_pwm_cycle(
        control_id=control.id,
        rising=rising,
        falling=falling,
        previous_rising=None,
    )
    return time < falling


def _complementary_cycle_candidates(
    high: PwmControl,
    cycle: int,
    *,
    previous_rising: float | None,
) -> tuple[float, tuple[tuple[float, str, bool], ...]]:
    block_id = high.block_id
    low_id = high.partner_id
    duty = high.command_duty
    phase = high.normalized_phase
    if block_id is None or low_id is None or duty is None or phase is None:
        raise SimulationError(f"Complementary PWM metadata is incomplete for gate {high.id!r}")

    command_rising = phase + cycle * high.period
    command_falling = command_rising + duty * high.period
    next_command_rising = phase + (cycle + 1) * high.period
    _require_distinct_pwm_cycle(
        control_id=block_id,
        rising=command_rising,
        falling=command_falling,
        previous_rising=previous_rising,
    )
    if high.dead_time == 0.0:
        high_turn_on = command_rising
        low_turn_on = command_falling
    else:
        high_turn_on = command_rising + high.dead_time
        low_turn_on = command_falling + high.dead_time
        if not (
            command_rising < high_turn_on < command_falling
            and command_falling < low_turn_on < next_command_rising
            and not _ulp_close(command_rising, high_turn_on)
            and not _ulp_close(command_falling, low_turn_on)
        ):
            raise SimulationError(
                f"Complementary PWM dead time for block {block_id!r} is not "
                "representable at the requested absolute time origin"
            )

    return command_rising, (
        (command_rising, low_id, False),
        (high_turn_on, high.id, True),
        (command_falling, high.id, False),
        (low_turn_on, low_id, True),
    )


def _spwm_blocks(
    controls: dict[str, PwmControl],
) -> dict[str, dict[str, PwmControl]]:
    """Return one high-side representative per phase of every SPWM block."""

    members: dict[str, list[PwmControl]] = {}
    for control in controls.values():
        if control.kind != "three_phase_spwm":
            continue
        block_id = control.block_id
        if not isinstance(block_id, str) or not block_id:
            raise SimulationError(
                f"Three-phase SPWM metadata is incomplete for gate {control.id!r}"
            )
        members.setdefault(block_id, []).append(control)

    blocks: dict[str, dict[str, PwmControl]] = {}
    metadata_fields = (
        "frequency",
        "duty",
        "phase",
        "normalized_phase",
        "fundamental_frequency",
        "modulation_index",
        "reference_phase",
        "carrier_ratio",
        "sequence",
        "dead_time",
    )
    for block_id, outputs in sorted(members.items()):
        if len(outputs) != 6:
            raise SimulationError(
                f"Three-phase SPWM block {block_id!r} must contain exactly six gate outputs"
            )
        representatives: dict[str, PwmControl] = {}
        baseline = outputs[0]
        for control in outputs:
            phase_label = getattr(control, "phase_label", None)
            if phase_label not in _SPWM_PHASES or control.role not in {"high", "low"}:
                raise SimulationError(
                    f"Three-phase SPWM metadata is incomplete for gate {control.id!r}"
                )
            partner_id = control.partner_id
            partner = controls.get(partner_id) if isinstance(partner_id, str) else None
            if (
                partner is None
                or partner.kind != "three_phase_spwm"
                or partner.block_id != block_id
                or getattr(partner, "phase_label", None) != phase_label
                or partner.partner_id != control.id
                or partner.role == control.role
            ):
                raise SimulationError(
                    f"Three-phase SPWM partner metadata is invalid for gate {control.id!r}"
                )
            for field in metadata_fields:
                if getattr(control, field, None) != getattr(baseline, field, None):
                    raise SimulationError(
                        f"Three-phase SPWM block {block_id!r} has inconsistent {field} metadata"
                    )
            if control.role == "high":
                if phase_label in representatives:
                    raise SimulationError(
                        f"Three-phase SPWM block {block_id!r} has duplicate phase {phase_label!r}"
                    )
                representatives[phase_label] = control
        if set(representatives) != set(_SPWM_PHASES):
            raise SimulationError(
                f"Three-phase SPWM block {block_id!r} must contain phases a, b, and c"
            )
        _validate_spwm_metadata(block_id, representatives["a"])
        blocks[block_id] = representatives
    return blocks


def _validate_spwm_metadata(block_id: str, control: PwmControl) -> None:
    frequency = control.frequency
    carrier_phase = control.normalized_phase
    fundamental_frequency = getattr(control, "fundamental_frequency", None)
    modulation_index = getattr(control, "modulation_index", None)
    reference_phase = getattr(control, "reference_phase", None)
    carrier_ratio = getattr(control, "carrier_ratio", None)
    sequence = getattr(control, "sequence", None)
    numeric = (
        frequency,
        carrier_phase,
        fundamental_frequency,
        modulation_index,
        reference_phase,
        control.dead_time,
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in numeric
    ):
        raise SimulationError(f"Three-phase SPWM metadata is incomplete for block {block_id!r}")
    if frequency <= 0.0 or fundamental_frequency <= 0.0:
        raise SimulationError(
            f"Three-phase SPWM frequencies must be positive for block {block_id!r}"
        )
    if isinstance(carrier_ratio, bool) or not isinstance(carrier_ratio, int) or carrier_ratio <= 0:
        raise SimulationError(
            f"Three-phase SPWM carrier ratio must be a positive integer for block {block_id!r}"
        )
    if carrier_ratio > MAX_SPWM_CARRIER_RATIO:
        raise SimulationError(
            f"Three-phase SPWM carrier ratio exceeds the portable precomputed-table "
            f"limit of {MAX_SPWM_CARRIER_RATIO} for block {block_id!r}"
        )
    observed_ratio = frequency / fundamental_frequency
    if not math.isfinite(observed_ratio) or not _ulp_close(observed_ratio, float(carrier_ratio)):
        raise SimulationError(
            f"Three-phase SPWM frequencies are not synchronous for block {block_id!r}"
        )
    injection = getattr(control, "injection", None) or "none"
    reference_peak = math.sqrt(3.0) / 2.0 if injection == "min_max" else 1.0
    if not 0.0 <= modulation_index * reference_peak < 1.0:
        raise SimulationError(
            "Three-phase SPWM modulation index is outside the linear range "
            f"for block {block_id!r} (injection={injection})"
        )
    if sequence not in {"abc", "acb"}:
        raise SimulationError(
            f"Three-phase SPWM sequence must be abc or acb for block {block_id!r}"
        )
    if control.dead_time < 0.0:
        raise SimulationError(
            f"Three-phase SPWM dead time must be non-negative for block {block_id!r}"
        )
    guaranteed_command_margin = (
        0.5 * (1.0 - modulation_index * reference_peak) * control.period
    )
    if control.dead_time >= guaranteed_command_margin:
        raise SimulationError(
            f"Three-phase SPWM dead time deletes a commanded pulse for block {block_id!r}"
        )


def _spwm_frames(block_id: str, high: PwmControl) -> tuple[_SpwmFrame, ...]:
    """Precompute one synchronous electrical period without accumulating phase drift."""

    ratio = getattr(high, "carrier_ratio")
    fundamental_frequency = getattr(high, "fundamental_frequency")
    modulation_index = getattr(high, "modulation_index")
    reference_phase = getattr(high, "reference_phase")
    sequence = getattr(high, "sequence")
    carrier_phase = high.normalized_phase
    offset_cycles = (carrier_phase - reference_phase) * fundamental_frequency
    if not math.isfinite(offset_cycles) or abs(offset_cycles) > _MAX_EXACT_SCHEDULE_CYCLE_INDEX - 2:
        raise SimulationError(
            f"Three-phase SPWM reference phase for block {block_id!r} is not representable"
        )
    offset_cycles %= 1.0
    frames: list[_SpwmFrame] = []
    for frame_index in range(ratio):
        electrical_cycles = (offset_cycles + (frame_index + 0.5) / ratio) % 1.0
        angle = math.tau * electrical_cycles
        reference_a = math.sin(angle)
        reference_b = math.sin(angle - 2.0 * math.pi / 3.0)
        reference_c = math.sin(angle + 2.0 * math.pi / 3.0)
        if sequence == "acb":
            reference_b, reference_c = reference_c, reference_b
        references = (reference_a, reference_b, reference_c)
        if getattr(high, "injection", None) == "min_max":
            # Min-max (SVPWM-equivalent) common-mode centering; the identical
            # expression is recomputed by the three-phase modulation contract.
            offset = -0.5 * (max(references) + min(references))
            references = tuple(value + offset for value in references)
        duties = tuple(0.5 * (1.0 + modulation_index * value) for value in references)
        if any(not math.isfinite(duty) or not 0.0 < duty < 1.0 for duty in duties):
            raise SimulationError(
                f"Three-phase SPWM frame {frame_index} for block {block_id!r} "
                "loses a nonzero commanded pulse at float precision"
            )
        frames.append(_SpwmFrame(angle, references, duties))
    return tuple(frames)


def _spwm_cycle_candidates(
    block_id: str,
    phases: dict[str, PwmControl],
    frames: tuple[_SpwmFrame, ...],
    cycle: int,
    *,
    previous_cell_start: float | None,
) -> tuple[float, tuple[tuple[float, str, bool], ...], dict[str, Any]]:
    high_a = phases["a"]
    carrier_phase = high_a.normalized_phase
    if carrier_phase is None:
        raise SimulationError(f"Three-phase SPWM carrier phase is missing for block {block_id!r}")
    period = high_a.period
    cell_start = carrier_phase + cycle * period
    cell_end = carrier_phase + (cycle + 1) * period
    sample = cell_start + 0.5 * period
    if (
        not math.isfinite(cell_start)
        or not math.isfinite(cell_end)
        or not math.isfinite(sample)
        or not cell_start < sample < cell_end
        or (previous_cell_start is not None and cell_start <= previous_cell_start)
    ):
        raise SimulationError(
            f"Three-phase SPWM schedule for block {block_id!r} loses distinct carrier "
            "timestamps at the requested absolute time origin"
        )

    frame_index = cycle % len(frames)
    frame = frames[frame_index]
    references = dict(zip(_SPWM_PHASES, frame.references, strict=True))
    duties = dict(zip(_SPWM_PHASES, frame.duties, strict=True))
    transition_groups: dict[str, list[tuple[float, str, bool]]] = {
        "low_off": [],
        "high_on": [],
        "high_off": [],
        "low_on": [],
    }
    for phase_label in _SPWM_PHASES:
        high = phases[phase_label]
        low_id = high.partner_id
        if low_id is None:
            raise SimulationError(f"Three-phase SPWM metadata is incomplete for gate {high.id!r}")
        duty = duties[phase_label]
        command_start = cell_start + 0.5 * (1.0 - duty) * period
        command_end = cell_start + 0.5 * (1.0 + duty) * period
        high_turn_on = command_start + high.dead_time
        low_turn_on = command_end + high.dead_time
        if not (
            cell_start < command_start
            and command_start <= high_turn_on < command_end
            and command_end <= low_turn_on < cell_end
        ):
            raise SimulationError(
                f"Three-phase SPWM pulse for block {block_id!r} phase {phase_label!r} "
                "is not representable at the requested absolute time origin"
            )
        if (
            _ulp_close(cell_start, command_start)
            or _ulp_close(command_start, command_end)
            or _ulp_close(command_end, cell_end)
            or (
                high.dead_time > 0.0
                and (
                    _ulp_close(command_start, high_turn_on) or _ulp_close(command_end, low_turn_on)
                )
            )
        ):
            raise SimulationError(
                f"Three-phase SPWM pulse for block {block_id!r} phase {phase_label!r} "
                "is not representable at the requested absolute time origin"
            )
        transition_groups["low_off"].append((command_start, low_id, False))
        transition_groups["high_on"].append((high_turn_on, high.id, True))
        transition_groups["high_off"].append((command_end, high.id, False))
        transition_groups["low_on"].append((low_turn_on, low_id, True))

    # Balanced three-phase references contain exact mathematical coincidences.
    # Independent libm evaluations can nevertheless move one phase by one ULP.
    # Canonicalize only within one transition role, so a command edge can never
    # merge with its paired dead-time edge or with the opposite edge of its pulse.
    candidates: list[tuple[float, str, bool]] = []
    for group in transition_groups.values():
        candidates.extend(_canonicalize_cross_gate_edge_times(group))
    canonical_times = {
        (target, value): event_time for event_time, target, value in candidates
    }

    pulses: dict[str, dict[str, float]] = {}
    for phase_label in _SPWM_PHASES:
        high = phases[phase_label]
        low_id = high.partner_id
        if low_id is None:  # Already validated while constructing the groups.
            raise SimulationError(
                f"Three-phase SPWM metadata is incomplete for gate {high.id!r}"
            )
        command_start = canonical_times[(low_id, False)]
        high_turn_on = canonical_times[(high.id, True)]
        command_end = canonical_times[(high.id, False)]
        low_turn_on = canonical_times[(low_id, True)]
        if not (
            cell_start < command_start
            and command_start <= high_turn_on < command_end
            and command_end <= low_turn_on < cell_end
        ):
            raise SimulationError(
                f"Three-phase SPWM canonical timing for block {block_id!r} phase "
                f"{phase_label!r} would invalidate the commanded pulse"
            )
        if high.dead_time > 0.0 and (
            _ulp_close(command_start, high_turn_on)
            or _ulp_close(command_end, low_turn_on)
        ):
            raise SimulationError(
                f"Three-phase SPWM canonical timing for block {block_id!r} phase "
                f"{phase_label!r} would invalidate the declared dead time"
            )
        pulses[phase_label] = {
            "start_s": command_start,
            "end_s": command_end,
            "high_turn_on_s": high_turn_on,
            "low_turn_on_s": low_turn_on,
        }

    record = {
        "block_id": block_id,
        "cycle_index": cycle,
        "frame_index": frame_index,
        "start_s": cell_start,
        "end_s": cell_end,
        "sample_s": sample,
        "electrical_angle_rad": frame.electrical_angle_rad,
        "references": references,
        "duties": duties,
        "command_pulses": pulses,
        "phase_sequence": getattr(high_a, "sequence"),
        "sampling": _SPWM_SAMPLING,
        "dead_time_s": high_a.dead_time,
    }
    return cell_start, tuple(candidates), record


@dataclass(frozen=True)
class _NpcFrame:
    electrical_angle_rad: float
    references: tuple[float, float, float]
    # pair_duties[phase_index][pair_index] in npc_pair_specs order (top first).
    pair_duties: tuple[tuple[float, ...], ...]


def _npc_blocks(
    controls: dict[str, PwmControl],
) -> dict[str, dict[str, dict[str, PwmControl]]]:
    """One high-side representative per (phase, pair) of every NPC block."""

    members: dict[str, list[PwmControl]] = {}
    for control in controls.values():
        if control.kind != "three_phase_npc_spwm":
            continue
        block_id = control.block_id
        if not isinstance(block_id, str) or not block_id:
            raise SimulationError(
                f"NPC modulator metadata is incomplete for gate {control.id!r}"
            )
        members.setdefault(block_id, []).append(control)

    blocks: dict[str, dict[str, dict[str, PwmControl]]] = {}
    metadata_fields = (
        "frequency",
        "duty",
        "phase",
        "normalized_phase",
        "fundamental_frequency",
        "modulation_index",
        "reference_phase",
        "carrier_ratio",
        "sequence",
        "dead_time",
        "injection",
        "levels",
    )
    for block_id, outputs in sorted(members.items()):
        baseline = outputs[0]
        levels = getattr(baseline, "levels", None)
        if levels not in {3, 5}:
            raise SimulationError(
                f"NPC modulator levels must be 3 or 5 for block {block_id!r}"
            )
        pair_labels = _npc_pair_labels(levels)
        expected_outputs = 6 * (levels - 1)
        if len(outputs) != expected_outputs:
            raise SimulationError(
                f"NPC modulator block {block_id!r} must contain exactly "
                f"{expected_outputs} gate outputs for {levels} levels"
            )
        representatives: dict[str, dict[str, PwmControl]] = {}
        for control in outputs:
            phase_label = getattr(control, "phase_label", None)
            pair_label = getattr(control, "pair_label", None)
            if (
                phase_label not in _SPWM_PHASES
                or pair_label not in pair_labels
                or control.role not in {"high", "low"}
            ):
                raise SimulationError(
                    f"NPC modulator metadata is incomplete for gate {control.id!r}"
                )
            partner_id = control.partner_id
            partner = controls.get(partner_id) if isinstance(partner_id, str) else None
            if (
                partner is None
                or partner.kind != "three_phase_npc_spwm"
                or partner.block_id != block_id
                or getattr(partner, "phase_label", None) != phase_label
                or getattr(partner, "pair_label", None) != pair_label
                or partner.partner_id != control.id
                or partner.role == control.role
            ):
                raise SimulationError(
                    f"NPC modulator partner metadata is invalid for gate {control.id!r}"
                )
            for field in metadata_fields:
                if getattr(control, field, None) != getattr(baseline, field, None):
                    raise SimulationError(
                        f"NPC modulator block {block_id!r} has inconsistent "
                        f"{field} metadata"
                    )
            if control.role == "high":
                phase_slot = representatives.setdefault(phase_label, {})
                if pair_label in phase_slot:
                    raise SimulationError(
                        f"NPC modulator block {block_id!r} has duplicate pair "
                        f"{phase_label!r}.{pair_label!r}"
                    )
                phase_slot[pair_label] = control
        if set(representatives) != set(_SPWM_PHASES) or any(
            set(pairs) != set(pair_labels) for pairs in representatives.values()
        ):
            raise SimulationError(
                f"NPC modulator block {block_id!r} must cover phases a, b, c "
                f"with the {levels}-level pair set"
            )
        _validate_npc_metadata(block_id, representatives["a"][pair_labels[0]])
        blocks[block_id] = representatives
    return blocks


def _validate_npc_metadata(block_id: str, control: PwmControl) -> None:
    frequency = control.frequency
    fundamental_frequency = getattr(control, "fundamental_frequency", None)
    modulation_index = getattr(control, "modulation_index", None)
    carrier_ratio = getattr(control, "carrier_ratio", None)
    levels = getattr(control, "levels", None)
    numeric = (
        frequency,
        control.normalized_phase,
        fundamental_frequency,
        modulation_index,
        getattr(control, "reference_phase", None),
        control.dead_time,
    )
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in numeric
    ):
        raise SimulationError(
            f"NPC modulator metadata is incomplete for block {block_id!r}"
        )
    if frequency <= 0.0 or fundamental_frequency <= 0.0:
        raise SimulationError(
            f"NPC modulator frequencies must be positive for block {block_id!r}"
        )
    if (
        isinstance(carrier_ratio, bool)
        or not isinstance(carrier_ratio, int)
        or carrier_ratio <= 0
    ):
        raise SimulationError(
            f"NPC modulator carrier ratio must be a positive integer for "
            f"block {block_id!r}"
        )
    if levels not in {3, 5}:
        raise SimulationError(
            f"NPC modulator levels must be 3 or 5 for block {block_id!r}"
        )
    max_carrier_ratio = MAX_SCHEDULED_GATE_EDGES // (18 * (levels - 1))
    if carrier_ratio > max_carrier_ratio:
        raise SimulationError(
            f"NPC modulator carrier ratio exceeds the portable limit of "
            f"{max_carrier_ratio} for block {block_id!r}"
        )
    observed_ratio = frequency / fundamental_frequency
    if not math.isfinite(observed_ratio) or not _ulp_close(
        observed_ratio, float(carrier_ratio)
    ):
        raise SimulationError(
            f"NPC modulator frequencies are not synchronous for block {block_id!r}"
        )
    injection = getattr(control, "injection", None) or "none"
    reference_peak = math.sqrt(3.0) / 2.0 if injection == "min_max" else 1.0
    if not 0.0 <= modulation_index * reference_peak < 1.0:
        raise SimulationError(
            "NPC modulator modulation index is outside the linear range for "
            f"block {block_id!r} (injection={injection})"
        )
    if not 0.0 <= control.dead_time < 0.25 * control.period:
        raise SimulationError(
            f"NPC modulator dead time is outside the schedulable range for "
            f"block {block_id!r}"
        )
    if getattr(control, "sequence", None) not in {"abc", "acb"}:
        raise SimulationError(
            f"NPC modulator sequence must be abc or acb for block {block_id!r}"
        )


def _npc_frames(block_id: str, high: PwmControl) -> tuple[_NpcFrame, ...]:
    """One synchronous electrical period of references and per-pair duties."""

    ratio = getattr(high, "carrier_ratio")
    fundamental_frequency = getattr(high, "fundamental_frequency")
    modulation_index = getattr(high, "modulation_index")
    reference_phase = getattr(high, "reference_phase")
    sequence = getattr(high, "sequence")
    levels = getattr(high, "levels")
    carrier_phase = high.normalized_phase
    offset_cycles = (carrier_phase - reference_phase) * fundamental_frequency
    if (
        not math.isfinite(offset_cycles)
        or abs(offset_cycles) > _MAX_EXACT_SCHEDULE_CYCLE_INDEX - 2
    ):
        raise SimulationError(
            f"NPC modulator reference phase for block {block_id!r} is not "
            "representable"
        )
    offset_cycles %= 1.0
    frames: list[_NpcFrame] = []
    for frame_index in range(ratio):
        electrical_cycles = (offset_cycles + (frame_index + 0.5) / ratio) % 1.0
        angle = math.tau * electrical_cycles
        reference_a = math.sin(angle)
        reference_b = math.sin(angle - 2.0 * math.pi / 3.0)
        reference_c = math.sin(angle + 2.0 * math.pi / 3.0)
        if sequence == "acb":
            reference_b, reference_c = reference_c, reference_b
        references = (reference_a, reference_b, reference_c)
        if getattr(high, "injection", None) == "min_max":
            offset = -0.5 * (max(references) + min(references))
            references = tuple(value + offset for value in references)
        pair_duties = tuple(
            _npc_pair_duties_by_label(modulation_index * value, levels)
            for value in references
        )
        frames.append(_NpcFrame(angle, references, pair_duties))
    return tuple(frames)


def _npc_pair_duties_by_label(reference: float, levels: int) -> tuple[float, ...]:
    """Band duties reordered from bottom-first into pair-spec order (top first).

    Pair k's high-side device pulses in band N-1-k, so the top-first pair
    order is exactly the reversed bottom-first band decomposition.
    """

    return tuple(reversed(npc_pair_duties(reference, levels)))


def _npc_cell_class(
    duty: float, period: float, dead_time: float
) -> tuple[str, str | None]:
    """Classify one pair's carrier cell; suppression is declared, not hidden.

    Level-shifted modulation SATURATES pairs for entire half-periods (that is
    the NPC operating principle, not an edge case) and commands arbitrarily
    short pulses at band crossings. A commanded high pulse no wider than the
    dead time is suppressed exactly as a hardware gate driver's minimum-pulse
    interlock would; a low tail too short to host the dead-time turn-on keeps
    the high side conducting through the cell boundary.
    """

    if duty <= 0.0:
        return "sat0", None
    if duty >= 1.0:
        return "sat1", None
    width = duty * period
    if width <= dead_time or _ulp_close(width, dead_time):
        return "sat0", "high_pulse_suppressed"
    half_gap = 0.5 * (1.0 - duty) * period
    if half_gap <= dead_time or _ulp_close(half_gap, dead_time):
        return "sat1", "low_pulse_suppressed"
    return "pulsing", None


def _npc_exit_state(classification: str) -> str:
    return "high" if classification == "sat1" else "low"


def _npc_cycle_candidates(
    block_id: str,
    phases: dict[str, dict[str, PwmControl]],
    frames: tuple[_NpcFrame, ...],
    cycle: int,
    *,
    previous_cell_start: float | None,
) -> tuple[float, tuple[tuple[float, str, bool], ...], dict[str, Any]]:
    pair_labels = _npc_pair_labels(
        getattr(next(iter(phases["a"].values())), "levels")
    )
    representative = phases["a"][pair_labels[0]]
    carrier_phase = representative.normalized_phase
    if carrier_phase is None:
        raise SimulationError(
            f"NPC modulator carrier phase is missing for block {block_id!r}"
        )
    period = representative.period
    dead_time = representative.dead_time
    cell_start = carrier_phase + cycle * period
    cell_end = carrier_phase + (cycle + 1) * period
    sample = cell_start + 0.5 * period
    if (
        not math.isfinite(cell_start)
        or not math.isfinite(cell_end)
        or not math.isfinite(sample)
        or not cell_start < sample < cell_end
        or (previous_cell_start is not None and cell_start <= previous_cell_start)
    ):
        raise SimulationError(
            f"NPC modulator schedule for block {block_id!r} loses distinct "
            "carrier timestamps at the requested absolute time origin"
        )

    frame_index = cycle % len(frames)
    frame = frames[frame_index]
    previous_frame = frames[(cycle - 1) % len(frames)]

    transition_groups: dict[str, list[tuple[float, str, bool]]] = {
        "boundary_off": [],
        "boundary_on": [],
        "low_off": [],
        "high_on": [],
        "high_off": [],
        "low_on": [],
    }
    pair_records: dict[str, dict[str, dict[str, Any]]] = {}
    for phase_index, phase_label in enumerate(_SPWM_PHASES):
        pair_records[phase_label] = {}
        for pair_index, pair_label in enumerate(pair_labels):
            high = phases[phase_label][pair_label]
            low_id = high.partner_id
            if low_id is None:
                raise SimulationError(
                    f"NPC modulator metadata is incomplete for gate {high.id!r}"
                )
            duty = frame.pair_duties[phase_index][pair_index]
            previous_duty = previous_frame.pair_duties[phase_index][pair_index]
            classification, suppressed = _npc_cell_class(duty, period, dead_time)
            previous_class, _ = _npc_cell_class(previous_duty, period, dead_time)
            entry_state = _npc_exit_state(previous_class)

            realized: dict[str, float] = {}
            if entry_state == "high" and classification != "sat1":
                # Band transition: hand over from the held high side before
                # this cell's own layout begins.
                boundary_on = cell_start + dead_time
                transition_groups["boundary_off"].append((cell_start, high.id, False))
                transition_groups["boundary_on"].append((boundary_on, low_id, True))
                realized["boundary_high_off_s"] = cell_start
                realized["boundary_low_on_s"] = boundary_on
            elif entry_state == "low" and classification == "sat1":
                boundary_on = cell_start + dead_time
                transition_groups["boundary_off"].append((cell_start, low_id, False))
                transition_groups["boundary_on"].append((boundary_on, high.id, True))
                realized["boundary_low_off_s"] = cell_start
                realized["boundary_high_on_s"] = boundary_on

            if classification == "pulsing":
                command_start = cell_start + 0.5 * (1.0 - duty) * period
                command_end = cell_start + 0.5 * (1.0 + duty) * period
                high_turn_on = command_start + dead_time
                low_turn_on = command_end + dead_time
                if not (
                    cell_start < command_start
                    and command_start <= high_turn_on < command_end
                    and command_end <= low_turn_on < cell_end
                ) or (
                    dead_time > 0.0
                    and (
                        _ulp_close(command_start, high_turn_on)
                        or _ulp_close(command_end, low_turn_on)
                        or _ulp_close(cell_start + dead_time, command_start)
                    )
                ):
                    raise SimulationError(
                        f"NPC modulator pulse for block {block_id!r} phase "
                        f"{phase_label!r} pair {pair_label!r} is not representable "
                        "at the requested absolute time origin"
                    )
                transition_groups["low_off"].append((command_start, low_id, False))
                transition_groups["high_on"].append((high_turn_on, high.id, True))
                transition_groups["high_off"].append((command_end, high.id, False))
                transition_groups["low_on"].append((low_turn_on, low_id, True))
                realized.update(
                    {
                        "start_s": command_start,
                        "end_s": command_end,
                        "high_turn_on_s": high_turn_on,
                        "low_turn_on_s": low_turn_on,
                    }
                )
            elif suppressed == "high_pulse_suppressed":
                # The commanded window still exists; the high side never fires
                # (minimum-pulse interlock), so the low side drops out for the
                # commanded window plus one dead time — the same both-off
                # interval a hardware driver would produce.
                command_start = cell_start + 0.5 * (1.0 - duty) * period
                command_end = cell_start + 0.5 * (1.0 + duty) * period
                low_turn_on = command_end + dead_time
                if not (
                    cell_start < command_start < command_end <= low_turn_on < cell_end
                ):
                    raise SimulationError(
                        f"NPC modulator suppressed pulse for block {block_id!r} "
                        f"phase {phase_label!r} pair {pair_label!r} is not "
                        "representable at the requested absolute time origin"
                    )
                transition_groups["low_off"].append((command_start, low_id, False))
                transition_groups["low_on"].append((low_turn_on, low_id, True))
                realized.update(
                    {
                        "start_s": command_start,
                        "end_s": command_end,
                        "low_turn_on_s": low_turn_on,
                    }
                )

            pair_records[phase_label][pair_label] = {
                "duty": duty,
                "classification": classification,
                "suppressed": suppressed,
                "entry_state": entry_state,
                "realized": realized,
            }

    candidates: list[tuple[float, str, bool]] = []
    for group in transition_groups.values():
        candidates.extend(_canonicalize_cross_gate_edge_times(group))
    candidates.sort(key=lambda edge: (edge[0], edge[2], edge[1]))

    record = {
        "modulator": "three_phase_npc_spwm",
        "block_id": block_id,
        "cycle_index": cycle,
        "frame_index": frame_index,
        "start_s": cell_start,
        "end_s": cell_end,
        "sample_s": sample,
        "electrical_angle_rad": frame.electrical_angle_rad,
        "references": dict(zip(_SPWM_PHASES, frame.references, strict=True)),
        "pairs": pair_records,
        "phase_sequence": getattr(representative, "sequence"),
        "sampling": _SPWM_SAMPLING,
        "dead_time_s": dead_time,
        "levels": getattr(representative, "levels"),
    }
    return cell_start, tuple(candidates), record


def _npc_scheduled_block(
    block_id: str,
    phases: dict[str, dict[str, PwmControl]],
    start: float,
    stop: float,
    *,
    edge_budget: int,
) -> tuple[list[tuple[float, str, bool]], list[dict[str, Any]]]:
    pair_labels = _npc_pair_labels(
        getattr(next(iter(phases["a"].values())), "levels")
    )
    representative = phases["a"][pair_labels[0]]
    carrier_phase = representative.normalized_phase
    if carrier_phase is None:
        raise SimulationError(
            f"NPC modulator carrier phase is missing for block {block_id!r}"
        )
    first_cycle, last_cycle = _schedule_cycle_bounds(
        phase=carrier_phase,
        period=representative.period,
        start=start,
        stop=stop,
        control_id=block_id,
    )
    frames = _npc_frames(block_id, representative)
    edges: list[tuple[float, str, bool]] = []
    records: list[dict[str, Any]] = []
    previous_cell_start: float | None = None
    for cycle in range(first_cycle, last_cycle + 1):
        cell_start, candidates, record = _npc_cycle_candidates(
            block_id,
            phases,
            frames,
            cycle,
            previous_cell_start=previous_cell_start,
        )
        previous_cell_start = cell_start
        if record["start_s"] < stop and record["end_s"] > start:
            records.append(record)
        for event_time, target, value in candidates:
            event_time = _snap_run_boundary(event_time, start, stop)
            if start < event_time <= stop:
                if len(edges) >= edge_budget:
                    raise SimulationError(
                        "Scheduled gate edges exceed the global limit of "
                        f"{MAX_SCHEDULED_GATE_EDGES}"
                    )
                edges.append((event_time, target, value))
    return edges, records


def _npc_initial_gate_values(
    controls: dict[str, PwmControl],
    time: float,
) -> dict[str, bool]:
    gates: dict[str, bool] = {}
    for block_id, phases in _npc_blocks(controls).items():
        pair_labels = _npc_pair_labels(
            getattr(next(iter(phases["a"].values())), "levels")
        )
        representative = phases["a"][pair_labels[0]]
        carrier_phase = representative.normalized_phase
        if carrier_phase is None:
            raise SimulationError(
                f"NPC modulator carrier phase is missing for block {block_id!r}"
            )
        frames = _npc_frames(block_id, representative)
        cycle, _ = _absolute_cycle_at_or_before(
            time,
            carrier_phase,
            representative.period,
            control_id=block_id,
        )
        _, candidates, record = _npc_cycle_candidates(
            block_id,
            phases,
            frames,
            cycle,
            previous_cell_start=None,
        )
        block_gates: dict[str, bool] = {}
        for phase_label in _SPWM_PHASES:
            for pair_label in pair_labels:
                high = phases[phase_label][pair_label]
                if high.partner_id is None:
                    raise SimulationError(
                        f"NPC modulator metadata is incomplete for gate {high.id!r}"
                    )
                entry_state = record["pairs"][phase_label][pair_label]["entry_state"]
                block_gates[high.id] = entry_state == "high"
                block_gates[high.partner_id] = entry_state == "low"
        for event_time, target, value in sorted(
            candidates,
            key=lambda edge: (edge[0], edge[2], edge[1]),
        ):
            if event_time <= time:
                block_gates[target] = value
        gates.update(block_gates)
    return gates


def _initial_gate_values(
    controls: dict[str, PwmControl],
    time: float,
) -> dict[str, bool]:
    gates = {
        control.id: _pwm_value(control, time)
        for control in controls.values()
        if control.kind == "pwm"
    }
    blocks: dict[str, PwmControl] = {}
    for control in controls.values():
        if control.kind != "complementary_pwm":
            continue
        if control.block_id is None or control.role not in {"high", "low"}:
            raise SimulationError(
                f"Complementary PWM metadata is incomplete for gate {control.id!r}"
            )
        if control.role == "high":
            blocks[control.block_id] = control

    for block_id, high in blocks.items():
        low_id = high.partner_id
        duty = high.command_duty
        phase = high.normalized_phase
        if low_id is None or low_id not in controls or duty is None or phase is None:
            raise SimulationError(
                f"Complementary PWM metadata is incomplete for block {block_id!r}"
            )
        if duty <= 0.0:
            gates[high.id] = False
            gates[low_id] = True
            continue
        if duty >= 1.0:
            gates[high.id] = True
            gates[low_id] = False
            continue
        cycle, _ = _absolute_cycle_at_or_before(
            time,
            phase,
            high.period,
            control_id=block_id,
        )
        _, candidates = _complementary_cycle_candidates(
            high,
            cycle,
            previous_rising=None,
        )
        block_gates = {high.id: False, low_id: True}
        for event_time, target, value in sorted(
            candidates,
            key=lambda edge: (edge[0], edge[2], edge[1]),
        ):
            if event_time <= time:
                block_gates[target] = value
        gates.update(block_gates)

    for block_id, phases in _spwm_blocks(controls).items():
        high_a = phases["a"]
        carrier_phase = high_a.normalized_phase
        if carrier_phase is None:
            raise SimulationError(
                f"Three-phase SPWM carrier phase is missing for block {block_id!r}"
            )
        frames = _spwm_frames(block_id, high_a)
        cycle, _ = _absolute_cycle_at_or_before(
            time,
            carrier_phase,
            high_a.period,
            control_id=block_id,
        )
        _, candidates, _ = _spwm_cycle_candidates(
            block_id,
            phases,
            frames,
            cycle,
            previous_cell_start=None,
        )
        block_gates: dict[str, bool] = {}
        for high in phases.values():
            if high.partner_id is None:
                raise SimulationError(
                    f"Three-phase SPWM metadata is incomplete for gate {high.id!r}"
                )
            block_gates[high.id] = False
            block_gates[high.partner_id] = True
        for event_time, target, value in sorted(
            candidates,
            key=lambda edge: (edge[0], edge[2], edge[1]),
        ):
            if event_time <= time:
                block_gates[target] = value
        gates.update(block_gates)
    gates.update(_npc_initial_gate_values(controls, time))
    return gates


def _schedule_cycle_bounds(
    *,
    phase: float,
    period: float,
    start: float,
    stop: float,
    control_id: str,
) -> tuple[int, int]:
    relative_start = (start - phase) / period
    relative_stop = (stop - phase) / period
    if (
        not math.isfinite(relative_start)
        or not math.isfinite(relative_stop)
        or abs(relative_start) > _MAX_EXACT_SCHEDULE_CYCLE_INDEX - 2
        or abs(relative_stop) > _MAX_EXACT_SCHEDULE_CYCLE_INDEX - 2
    ):
        raise SimulationError(
            f"PWM schedule for control {control_id!r} is not representable "
            "at the requested absolute time origin"
        )
    return math.floor(relative_start) - 1, math.ceil(relative_stop) + 1


def _require_distinct_pwm_cycle(
    *,
    control_id: str,
    rising: float,
    falling: float,
    previous_rising: float | None,
) -> None:
    if (
        not math.isfinite(rising)
        or not math.isfinite(falling)
        or falling <= rising
        or (previous_rising is not None and rising <= previous_rising)
    ):
        raise SimulationError(
            f"PWM schedule for control {control_id!r} loses distinct edge "
            "timestamps at the requested absolute time origin"
        )


def _legacy_scheduled_edges(
    control: PwmControl,
    start: float,
    stop: float,
) -> list[tuple[float, str, bool]]:
    first_cycle, last_cycle = _schedule_cycle_bounds(
        phase=control.phase,
        period=control.period,
        start=start,
        stop=stop,
        control_id=control.id,
    )
    edges: list[tuple[float, str, bool]] = []
    previous_rising: float | None = None
    for cycle in range(first_cycle, last_cycle + 1):
        rising = control.phase + cycle * control.period
        falling = rising + control.duty * control.period
        _require_distinct_pwm_cycle(
            control_id=control.id,
            rising=rising,
            falling=falling,
            previous_rising=previous_rising,
        )
        previous_rising = rising
        rising = _snap_run_boundary(rising, start, stop)
        falling = _snap_run_boundary(falling, start, stop)
        if start < rising <= stop:
            edges.append((rising, control.id, True))
        if start < falling <= stop:
            edges.append((falling, control.id, False))
    return edges


def _complementary_scheduled_edges(
    high: PwmControl,
    start: float,
    stop: float,
) -> list[tuple[float, str, bool]]:
    block_id = high.block_id
    low_id = high.partner_id
    duty = high.command_duty
    phase = high.normalized_phase
    if block_id is None or low_id is None or duty is None or phase is None:
        raise SimulationError(f"Complementary PWM metadata is incomplete for gate {high.id!r}")
    if duty <= 0.0 or duty >= 1.0:
        return []
    first_cycle, last_cycle = _schedule_cycle_bounds(
        phase=phase,
        period=high.period,
        start=start,
        stop=stop,
        control_id=block_id,
    )
    edges: list[tuple[float, str, bool]] = []
    previous_rising: float | None = None
    for cycle in range(first_cycle, last_cycle + 1):
        command_rising, candidates = _complementary_cycle_candidates(
            high,
            cycle,
            previous_rising=previous_rising,
        )
        previous_rising = command_rising
        for event_time, target, value in candidates:
            event_time = _snap_run_boundary(event_time, start, stop)
            if start < event_time <= stop:
                edges.append((event_time, target, value))
    return edges


def _spwm_scheduled_block(
    block_id: str,
    phases: dict[str, PwmControl],
    start: float,
    stop: float,
    *,
    edge_budget: int,
) -> tuple[list[tuple[float, str, bool]], list[dict[str, Any]]]:
    high_a = phases["a"]
    carrier_phase = high_a.normalized_phase
    if carrier_phase is None:
        raise SimulationError(f"Three-phase SPWM carrier phase is missing for block {block_id!r}")
    first_cycle, last_cycle = _schedule_cycle_bounds(
        phase=carrier_phase,
        period=high_a.period,
        start=start,
        stop=stop,
        control_id=block_id,
    )
    frames = _spwm_frames(block_id, high_a)
    edges: list[tuple[float, str, bool]] = []
    records: list[dict[str, Any]] = []
    previous_cell_start: float | None = None
    for cycle in range(first_cycle, last_cycle + 1):
        cell_start, candidates, record = _spwm_cycle_candidates(
            block_id,
            phases,
            frames,
            cycle,
            previous_cell_start=previous_cell_start,
        )
        previous_cell_start = cell_start
        if record["start_s"] < stop and record["end_s"] > start:
            records.append(record)
        for event_time, target, value in candidates:
            event_time = _snap_run_boundary(event_time, start, stop)
            if start < event_time <= stop:
                if len(edges) >= edge_budget:
                    raise SimulationError(
                        "Scheduled gate edges exceed the global limit of "
                        f"{MAX_SCHEDULED_GATE_EDGES}"
                    )
                edges.append((event_time, target, value))
    return edges, records


def _canonicalize_cross_gate_edge_times(
    edges: list[tuple[float, str, bool]],
) -> list[tuple[float, str, bool]]:
    """Merge only ULP-level coincidences between distinct gate outputs."""

    ordered = sorted(edges, key=lambda edge: (edge[0], edge[2], edge[1]))
    canonical: list[tuple[float, str, bool]] = []
    index = 0
    while index < len(ordered):
        representative = ordered[index][0]
        cluster = [ordered[index]]
        controls = {ordered[index][1]}
        index += 1
        while index < len(ordered):
            candidate = ordered[index]
            if candidate[1] in controls or not _ulp_close(candidate[0], representative):
                break
            cluster.append(candidate)
            controls.add(candidate[1])
            index += 1
        canonical.extend((representative, control_id, value) for _, control_id, value in cluster)
    canonical.sort(key=lambda edge: (edge[0], edge[2], edge[1]))
    return canonical


def _build_gate_schedule(
    controls: dict[str, PwmControl], start: float, stop: float
) -> _GateSchedule:
    span = stop - start
    if not math.isfinite(span) or span <= 0.0:
        raise SimulationError("Scheduled-gate interval must be finite and positive")

    spwm_blocks = _spwm_blocks(controls)
    npc_blocks = _npc_blocks(controls)
    conservative_edge_count = 0
    counted_complementary: set[str] = set()
    for control in controls.values():
        if control.kind in {"three_phase_spwm", "three_phase_npc_spwm"}:
            # SPWM/NPC edges are counted exactly while the block schedule is
            # built below.  A conservative pad would reject the documented
            # maximum synchronous table even though it fits the edge budget.
            continue
        if control.kind == "complementary_pwm":
            if control.duty <= 0.0 or control.duty >= 1.0:
                continue
            if control.block_id is None:
                raise SimulationError(
                    f"Complementary PWM metadata is incomplete for gate {control.id!r}"
                )
            if control.block_id in counted_complementary:
                continue
            counted_complementary.add(control.block_id)
            edges_per_cycle = 4
        elif control.kind == "pwm":
            if control.duty <= 0.0 or control.duty >= 1.0:
                continue
            edges_per_cycle = 2
        else:
            raise SimulationError(f"Unsupported PWM control kind {control.kind!r}")
        cycle_count = span * control.frequency
        if (
            not math.isfinite(cycle_count)
            or cycle_count > MAX_SCHEDULED_GATE_EDGES / edges_per_cycle - 2
        ):
            raise SimulationError(
                f"Scheduled gate edges exceed the global limit of {MAX_SCHEDULED_GATE_EDGES}"
            )
        conservative_edge_count += edges_per_cycle * (math.ceil(cycle_count) + 2)
        if conservative_edge_count > MAX_SCHEDULED_GATE_EDGES:
            raise SimulationError(
                f"Scheduled gate edges exceed the global limit of {MAX_SCHEDULED_GATE_EDGES}"
            )

    legacy_edges: list[tuple[float, str, bool]] = []
    complementary_edges: list[tuple[float, str, bool]] = []
    spwm_edges: list[tuple[float, str, bool]] = []
    modulation_cycles: list[dict[str, Any]] = []
    seen_blocks: set[str] = set()
    for control in controls.values():
        if control.kind in {"three_phase_spwm", "three_phase_npc_spwm"}:
            continue
        if control.duty <= 0.0 or control.duty >= 1.0:
            continue
        if control.kind == "pwm":
            legacy_edges.extend(_legacy_scheduled_edges(control, start, stop))
            continue
        if control.kind != "complementary_pwm":
            raise SimulationError(f"Unsupported PWM control kind {control.kind!r}")
        if control.block_id is None:
            raise SimulationError(
                f"Complementary PWM metadata is incomplete for gate {control.id!r}"
            )
        if control.block_id in seen_blocks or control.role != "high":
            continue
        seen_blocks.add(control.block_id)
        complementary_edges.extend(_complementary_scheduled_edges(control, start, stop))

    for block_id, phases in spwm_blocks.items():
        block_edges, block_cycles = _spwm_scheduled_block(
            block_id,
            phases,
            start,
            stop,
            edge_budget=MAX_SCHEDULED_GATE_EDGES - conservative_edge_count - len(spwm_edges),
        )
        spwm_edges.extend(block_edges)
        modulation_cycles.extend(block_cycles)

    for block_id, npc_phases in npc_blocks.items():
        block_edges, block_cycles = _npc_scheduled_block(
            block_id,
            npc_phases,
            start,
            stop,
            edge_budget=MAX_SCHEDULED_GATE_EDGES - conservative_edge_count - len(spwm_edges),
        )
        spwm_edges.extend(block_edges)
        modulation_cycles.extend(block_cycles)

    edges = _canonicalize_cross_gate_edge_times(legacy_edges) + complementary_edges + spwm_edges
    # Every exact-timestamp group is applied atomically.  The stable event-log
    # order records all turn-offs before turn-ons without introducing an
    # intermediate electrical topology.
    edges.sort(key=lambda edge: (edge[0], edge[2], edge[1]))
    if len(edges) > MAX_SCHEDULED_GATE_EDGES:
        raise SimulationError(
            f"Scheduled gate edges exceed the global limit of {MAX_SCHEDULED_GATE_EDGES}"
        )
    modulation_cycles.sort(key=lambda item: (item["block_id"], item["cycle_index"]))
    return _GateSchedule(edges=edges, modulation_cycles=modulation_cycles)


def scheduled_edges(
    controls: dict[str, PwmControl], start: float, stop: float
) -> list[tuple[float, str, bool]]:
    """Materialized gate-edge schedule for the given controls and window.

    Public inspection/regression view of the deterministic gate schedule the
    simulator executes; it does not run a simulation.
    """

    return _build_gate_schedule(controls, start, stop).edges


# Deprecated alias kept for one release; use ``scheduled_edges``.
_scheduled_edges = scheduled_edges
