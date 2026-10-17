"""Bounded evidence for the three-level NPC level-shifted modulator.

The realized gate-event log remains the authoritative record. This module
independently recomputes the deterministic modulation law — sampled
references, level-shifted (PD) band decomposition into per-pair duties,
center-aligned realization with dead time, band-transition hand-overs and
minimum-pulse suppression — from the expanded control metadata alone, and
requires the solver's per-carrier records to match bit-exactly (the same
contract style as the two-level SPWM observation).
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from pear_sim.run.contracts import model_hashes
from pear_sim.spec.controls import npc_pair_duties, npc_pair_specs

NPC_MODULATION_SCHEMA = "pear-sim/npc-modulation-observation/v0.1"
NPC_MODULATION_KIND = "npc-modulation-observation"
NPC_MODULATION_CYCLE_SCHEMA = "pear-sim/modulation-cycle/three-phase-npc-spwm/v0.1"

MAX_MODULATION_CYCLE_ROWS = 100_000

_PHASES = ("a", "b", "c")
_SAMPLING = "offline_midpoint_sampled_center_aligned"


def _pair_labels(levels: int) -> tuple[str, ...]:
    return tuple(pair_label for pair_label, _, _ in npc_pair_specs(levels))


class NpcModulationError(ValueError):
    """Raised when a run cannot support an exact NPC modulation observation."""

    code = "analysis.npc_modulation_invalid"


def has_three_phase_npc_spwm(model: Any) -> bool:
    """Return whether the expanded model contains an NPC modulator gate."""

    controls = getattr(model, "controls", None)
    if not isinstance(controls, Mapping):
        return False
    return any(
        getattr(control, "kind", None) == "three_phase_npc_spwm"
        for control in controls.values()
    )


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NpcModulationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise NpcModulationError(f"{name} must be a finite number")
    return result


def _require_float_match(actual: float, expected: float, *, name: str) -> None:
    tolerance = max(8.0 * math.ulp(actual), 8.0 * math.ulp(expected), 1e-18)
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise NpcModulationError(
            f"{name} disagrees with the deterministic NPC modulation law: "
            f"got {actual:.17g}, expected {expected:.17g}"
        )


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise NpcModulationError(f"{name} must be a non-empty string")
    return value


def _expanded_npc_blocks(model: Any) -> dict[str, dict[str, Any]]:
    controls = getattr(model, "controls", None)
    if not isinstance(controls, Mapping):
        raise NpcModulationError("Model controls must be a mapping")
    grouped: dict[str, list[Any]] = {}
    for control in controls.values():
        if getattr(control, "kind", None) != "three_phase_npc_spwm":
            continue
        block_id = _text(getattr(control, "block_id", None), name="NPC block_id")
        grouped.setdefault(block_id, []).append(control)
    if not grouped:
        raise NpcModulationError("No expanded three_phase_npc_spwm block was found")

    blocks: dict[str, dict[str, Any]] = {}
    for block_id, members in grouped.items():
        baseline = members[0]
        levels = getattr(baseline, "levels", None)
        if levels not in {3, 5}:
            raise NpcModulationError(f"NPC block {block_id!r} levels must be 3 or 5")
        if len(members) != 6 * (levels - 1):
            raise NpcModulationError(
                f"NPC block {block_id!r} must expand to exactly "
                f"{6 * (levels - 1)} gates for {levels} levels"
            )
        for field in (
            "frequency",
            "normalized_phase",
            "fundamental_frequency",
            "modulation_index",
            "reference_phase",
            "carrier_ratio",
            "sequence",
            "dead_time",
            "injection",
            "levels",
        ):
            values = {getattr(member, field, None) for member in members}
            if len(values) != 1:
                raise NpcModulationError(
                    f"NPC block {block_id!r} has inconsistent {field} metadata"
                )
        carrier_frequency = _finite(
            baseline.frequency, name=f"{block_id}.carrier_frequency"
        )
        if carrier_frequency <= 0.0:
            raise NpcModulationError(f"{block_id}.carrier_frequency must be positive")
        blocks[block_id] = {
            "carrier_frequency_hz": carrier_frequency,
            "period_s": 1.0 / carrier_frequency,
            "normalized_carrier_phase_s": _finite(
                baseline.normalized_phase, name=f"{block_id}.normalized_phase"
            ),
            "fundamental_frequency_hz": _finite(
                baseline.fundamental_frequency,
                name=f"{block_id}.fundamental_frequency",
            ),
            "modulation_index": _finite(
                baseline.modulation_index, name=f"{block_id}.modulation_index"
            ),
            "reference_phase_s": _finite(
                baseline.reference_phase, name=f"{block_id}.reference_phase"
            ),
            "carrier_ratio": baseline.carrier_ratio,
            "phase_sequence": _text(
                baseline.sequence, name=f"{block_id}.sequence"
            ),
            "dead_time_s": _finite(baseline.dead_time, name=f"{block_id}.dead_time"),
            "injection": getattr(baseline, "injection", None) or "none",
            "levels": baseline.levels,
        }
        ratio = blocks[block_id]["carrier_ratio"]
        if isinstance(ratio, bool) or not isinstance(ratio, int) or ratio < 2:
            raise NpcModulationError(
                f"NPC block {block_id!r} carrier_ratio must be an integer >= 2"
            )
    return blocks


def _ulp_close(left: float, right: float) -> bool:
    return abs(left - right) <= 8.0 * max(math.ulp(left), math.ulp(right))


def _cell_class(duty: float, period: float, dead_time: float) -> tuple[str, str | None]:
    """The scheduler's cell classification, restated independently."""

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


def _expected_frame(
    config: dict[str, Any], frame_index: int
) -> tuple[float, dict[str, float], dict[str, tuple[float, ...]]]:
    offset_cycles = (
        config["normalized_carrier_phase_s"] - config["reference_phase_s"]
    ) * config["fundamental_frequency_hz"]
    offset_cycles %= 1.0
    electrical_cycles = (
        offset_cycles + (frame_index + 0.5) / config["carrier_ratio"]
    ) % 1.0
    angle = math.tau * electrical_cycles
    reference_a = math.sin(angle)
    reference_b = math.sin(angle - 2.0 * math.pi / 3.0)
    reference_c = math.sin(angle + 2.0 * math.pi / 3.0)
    if config["phase_sequence"] == "acb":
        reference_b, reference_c = reference_c, reference_b
    references = (reference_a, reference_b, reference_c)
    if config["injection"] == "min_max":
        offset = -0.5 * (max(references) + min(references))
        references = tuple(value + offset for value in references)
    by_phase = dict(zip(_PHASES, references, strict=True))
    duties: dict[str, tuple[float, ...]] = {}
    for phase_label, reference in by_phase.items():
        bottom_first = npc_pair_duties(
            config["modulation_index"] * reference, config["levels"]
        )
        # Pair-spec order is top first: the reversed bottom-first band list.
        duties[phase_label] = tuple(reversed(bottom_first))
    return angle, by_phase, duties


def normalized_npc_modulation_cycles(result: Any) -> list[dict[str, Any]]:
    """Validate the solver's NPC per-carrier records against the law."""

    blocks = _expanded_npc_blocks(result.model)
    raw_cycles = getattr(result, "modulation_cycles", None)
    if not isinstance(raw_cycles, list):
        raise NpcModulationError(
            "three_phase_npc_spwm result must expose modulation_cycles as a list"
        )
    if len(raw_cycles) > MAX_MODULATION_CYCLE_ROWS:
        raise NpcModulationError(
            f"modulation_cycles exceeds the {MAX_MODULATION_CYCLE_ROWS}-row limit"
        )
    if not raw_cycles:
        raise NpcModulationError(
            "three_phase_npc_spwm result contains no modulation cycles"
        )

    run_start = _finite(result.model.transient.start, name="transient.start")
    run_stop = _finite(result.model.transient.stop, name="transient.stop")
    previous_by_block: dict[str, tuple[int, float]] = {}
    normalized: list[dict[str, Any]] = []
    for row_index, raw in enumerate(raw_cycles):
        if not isinstance(raw, Mapping):
            raise NpcModulationError(f"modulation cycle {row_index} must be an object")
        if raw.get("modulator") != "three_phase_npc_spwm":
            raise NpcModulationError(
                f"modulation cycle {row_index} is not an NPC modulator record"
            )
        block_id = _text(
            raw.get("block_id"), name=f"modulation cycle {row_index}.block_id"
        )
        config = blocks.get(block_id)
        if config is None:
            raise NpcModulationError(
                f"modulation cycle {row_index} references unknown block {block_id!r}"
            )
        cycle_index = raw.get("cycle_index")
        if isinstance(cycle_index, bool) or not isinstance(cycle_index, int):
            raise NpcModulationError(
                f"modulation cycle {row_index}.cycle_index must be an integer"
            )
        frame_index = raw.get("frame_index")
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or frame_index < 0
        ):
            raise NpcModulationError(
                f"modulation cycle {row_index}.frame_index must be an integer >= 0"
            )
        if frame_index != cycle_index % config["carrier_ratio"]:
            raise NpcModulationError(
                f"modulation cycle {row_index}.frame_index disagrees with "
                "cycle_index modulo carrier_ratio"
            )
        start = _finite(raw.get("start_s"), name=f"modulation cycle {row_index}.start_s")
        end = _finite(raw.get("end_s"), name=f"modulation cycle {row_index}.end_s")
        sample = _finite(
            raw.get("sample_s"), name=f"modulation cycle {row_index}.sample_s"
        )
        if end <= start or not start <= sample < end:
            raise NpcModulationError(
                f"modulation cycle {row_index} has invalid half-open timing"
            )
        if end <= run_start or start >= run_stop:
            raise NpcModulationError(
                f"modulation cycle {row_index} does not overlap the run window"
            )
        previous = previous_by_block.get(block_id)
        if previous is not None and (cycle_index <= previous[0] or start < previous[1]):
            raise NpcModulationError(
                f"modulation cycles for {block_id!r} are not strictly ordered"
            )
        previous_by_block[block_id] = (cycle_index, start)

        period = config["period_s"]
        carrier_phase = config["normalized_carrier_phase_s"]
        _require_float_match(
            start,
            carrier_phase + cycle_index * period,
            name=f"modulation cycle {row_index}.start_s",
        )
        _require_float_match(
            end,
            carrier_phase + (cycle_index + 1) * period,
            name=f"modulation cycle {row_index}.end_s",
        )
        _require_float_match(
            sample,
            (carrier_phase + cycle_index * period) + 0.5 * period,
            name=f"modulation cycle {row_index}.sample_s",
        )
        if raw.get("phase_sequence") != config["phase_sequence"]:
            raise NpcModulationError(
                f"modulation cycle {row_index} disagrees with block phase sequence"
            )
        if raw.get("sampling") != _SAMPLING:
            raise NpcModulationError(
                f"modulation cycle {row_index} has unexpected sampling declaration"
            )
        dead_time = _finite(
            raw.get("dead_time_s"), name=f"modulation cycle {row_index}.dead_time_s"
        )
        if dead_time != config["dead_time_s"]:
            raise NpcModulationError(
                f"modulation cycle {row_index} disagrees with block dead time"
            )
        if raw.get("levels") != config["levels"]:
            raise NpcModulationError(
                f"modulation cycle {row_index} disagrees with block levels"
            )

        expected_angle, expected_references, expected_duties = _expected_frame(
            config, frame_index
        )
        _require_float_match(
            _finite(
                raw.get("electrical_angle_rad"),
                name=f"modulation cycle {row_index}.electrical_angle_rad",
            ),
            expected_angle,
            name=f"modulation cycle {row_index}.electrical_angle_rad",
        )
        references = raw.get("references")
        if not isinstance(references, Mapping) or set(references) != set(_PHASES):
            raise NpcModulationError(
                f"modulation cycle {row_index}.references must map phases a, b, c"
            )
        for phase_label in _PHASES:
            _require_float_match(
                _finite(
                    references[phase_label],
                    name=f"modulation cycle {row_index}.references.{phase_label}",
                ),
                expected_references[phase_label],
                name=f"modulation cycle {row_index}.references.{phase_label}",
            )

        pairs = raw.get("pairs")
        if not isinstance(pairs, Mapping) or set(pairs) != set(_PHASES):
            raise NpcModulationError(
                f"modulation cycle {row_index}.pairs must map phases a, b, c"
            )
        pair_labels = _pair_labels(config["levels"])
        # Entry states chain through the synchronous frame table.
        previous_frame_index = (cycle_index - 1) % config["carrier_ratio"]
        _, _, previous_duties = _expected_frame(config, previous_frame_index)
        for phase_label in _PHASES:
            phase_pairs = pairs[phase_label]
            if not isinstance(phase_pairs, Mapping) or set(phase_pairs) != set(
                pair_labels
            ):
                raise NpcModulationError(
                    f"modulation cycle {row_index}.pairs.{phase_label} must map "
                    f"the {config['levels']}-level pair set"
                )
            for pair_index, pair_label in enumerate(pair_labels):
                entry = phase_pairs[pair_label]
                if not isinstance(entry, Mapping):
                    raise NpcModulationError(
                        f"modulation cycle {row_index}.pairs.{phase_label}."
                        f"{pair_label} must be an object"
                    )
                name = (
                    f"modulation cycle {row_index}.pairs.{phase_label}.{pair_label}"
                )
                duty = _finite(entry.get("duty"), name=f"{name}.duty")
                expected_duty = expected_duties[phase_label][pair_index]
                _require_float_match(duty, expected_duty, name=f"{name}.duty")
                expected_class, expected_suppressed = _cell_class(
                    expected_duty, period, dead_time
                )
                if entry.get("classification") != expected_class:
                    raise NpcModulationError(
                        f"{name}.classification disagrees with the band law: "
                        f"got {entry.get('classification')!r}, expected "
                        f"{expected_class!r}"
                    )
                if entry.get("suppressed") != expected_suppressed:
                    raise NpcModulationError(
                        f"{name}.suppressed disagrees with the minimum-pulse rule"
                    )
                previous_class, _ = _cell_class(
                    previous_duties[phase_label][pair_index], period, dead_time
                )
                expected_entry = "high" if previous_class == "sat1" else "low"
                if entry.get("entry_state") != expected_entry:
                    raise NpcModulationError(
                        f"{name}.entry_state disagrees with the previous cell"
                    )
                realized = entry.get("realized")
                if not isinstance(realized, Mapping):
                    raise NpcModulationError(f"{name}.realized must be an object")
                cell_start = start
                if expected_entry == "high" and expected_class != "sat1":
                    _require_float_match(
                        _finite(
                            realized.get("boundary_high_off_s"),
                            name=f"{name}.realized.boundary_high_off_s",
                        ),
                        cell_start,
                        name=f"{name}.realized.boundary_high_off_s",
                    )
                    _require_float_match(
                        _finite(
                            realized.get("boundary_low_on_s"),
                            name=f"{name}.realized.boundary_low_on_s",
                        ),
                        cell_start + dead_time,
                        name=f"{name}.realized.boundary_low_on_s",
                    )
                if expected_entry == "low" and expected_class == "sat1":
                    _require_float_match(
                        _finite(
                            realized.get("boundary_low_off_s"),
                            name=f"{name}.realized.boundary_low_off_s",
                        ),
                        cell_start,
                        name=f"{name}.realized.boundary_low_off_s",
                    )
                    _require_float_match(
                        _finite(
                            realized.get("boundary_high_on_s"),
                            name=f"{name}.realized.boundary_high_on_s",
                        ),
                        cell_start + dead_time,
                        name=f"{name}.realized.boundary_high_on_s",
                    )
                if expected_class == "pulsing":
                    command_start = cell_start + 0.5 * (1.0 - expected_duty) * period
                    command_end = cell_start + 0.5 * (1.0 + expected_duty) * period
                    for key, expected_value in (
                        ("start_s", command_start),
                        ("end_s", command_end),
                        ("high_turn_on_s", command_start + dead_time),
                        ("low_turn_on_s", command_end + dead_time),
                    ):
                        _require_float_match(
                            _finite(
                                realized.get(key), name=f"{name}.realized.{key}"
                            ),
                            expected_value,
                            name=f"{name}.realized.{key}",
                        )
                if expected_suppressed == "high_pulse_suppressed":
                    command_start = cell_start + 0.5 * (1.0 - expected_duty) * period
                    command_end = cell_start + 0.5 * (1.0 + expected_duty) * period
                    for key, expected_value in (
                        ("start_s", command_start),
                        ("end_s", command_end),
                        ("low_turn_on_s", command_end + dead_time),
                    ):
                        _require_float_match(
                            _finite(
                                realized.get(key), name=f"{name}.realized.{key}"
                            ),
                            expected_value,
                            name=f"{name}.realized.{key}",
                        )
        normalized.append({"schema": NPC_MODULATION_CYCLE_SCHEMA, **dict(raw)})
    return normalized


def build_npc_modulation_observation(
    result: Any,
    *,
    normalized_cycles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compact deterministic projection of the validated NPC command record."""

    blocks = _expanded_npc_blocks(result.model)
    hashes = model_hashes(result.model)
    block_id, config = next(iter(sorted(blocks.items())))
    pair_labels = _pair_labels(config["levels"])
    suppression: dict[str, dict[str, int]] = {
        phase: {pair: 0 for pair in pair_labels} for phase in _PHASES
    }
    band_transitions = 0
    in_window = 0
    for row in normalized_cycles:
        in_window += 1
        for phase_label in _PHASES:
            for pair_label in pair_labels:
                entry = row["pairs"][phase_label][pair_label]
                if entry.get("suppressed") is not None:
                    suppression[phase_label][pair_label] += 1
                realized = entry.get("realized", {})
                if any(key.startswith("boundary_") for key in realized):
                    band_transitions += 1
    return {
        "schema": NPC_MODULATION_SCHEMA,
        "kind": NPC_MODULATION_KIND,
        "cycle_schema": NPC_MODULATION_CYCLE_SCHEMA,
        "source_binding": {
            "model_sha256": hashes["model_sha256"],
            "request_sha256": hashes["request_sha256"],
            "run_sha256": None,
        },
        "block": {
            "block_id": block_id,
            **config,
        },
        "cycles_recorded": in_window,
        "band_transition_handovers": band_transitions,
        "minimum_pulse_suppressions": suppression,
        "claim_boundary": (
            "The observation validates the deterministic modulation COMMAND "
            "record (references, band duties, realized edge times, "
            "suppression) against the level-shifted PD law. It does not "
            "measure electrical output quality; harmonics and midpoint "
            "balance are separate observations."
        ),
    }
