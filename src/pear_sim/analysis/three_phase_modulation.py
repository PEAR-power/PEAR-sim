# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Bounded, human-facing evidence for three-phase sampled SPWM.

The full per-carrier command record and the realized gate-event log remain separate
authoritative artifacts.  This module validates the explicit bridge mapping and
produces a small deterministic projection for reports; it does not infer phase or
device roles from user-facing names.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
from typing import Any

from pear_sim.run.contracts import model_hashes


THREE_PHASE_MODULATION_SCHEMA = "pear-sim/three-phase-modulation-observation/v0.1"
THREE_PHASE_MODULATION_KIND = "three-phase-modulation-observation"
MODULATION_CYCLE_SCHEMA = "pear-sim/modulation-cycle/three-phase-spwm/v0.1"

# These producer bounds intentionally mirror the existing portable event contract.
# A report applies its own, smaller display bounds without weakening this full-run
# command-record limit.
MAX_MODULATION_CYCLE_ROWS = 100_000
MAX_MODULATION_PROJECTION_CYCLES = 4_096

_PHASES = ("a", "b", "c")
_ROLES = ("high", "low")
_SAMPLING = "offline_midpoint_sampled_center_aligned"


class ThreePhaseModulationError(ValueError):
    """Raised when a run cannot support an exact three-phase observation."""

    code = "analysis.modulation_invalid"


def has_three_phase_spwm(model: Any) -> bool:
    """Return whether the expanded model contains a three-phase SPWM gate."""

    controls = getattr(model, "controls", None)
    if not isinstance(controls, Mapping):
        return False
    return any(
        getattr(control, "kind", None) == "three_phase_spwm" for control in controls.values()
    )


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThreePhaseModulationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ThreePhaseModulationError(f"{name} must be a finite number")
    return result


def _float_matches(actual: float, expected: float) -> bool:
    """Compare deterministic schedule values within a tiny round-off envelope."""

    tolerance = max(8.0 * math.ulp(actual), 8.0 * math.ulp(expected), 1e-18)
    return math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance)


def _require_float_match(actual: float, expected: float, *, name: str) -> None:
    if not _float_matches(actual, expected):
        raise ThreePhaseModulationError(
            f"{name} disagrees with the deterministic sampled-SPWM command: "
            f"got {actual:.17g}, expected {expected:.17g}"
        )


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ThreePhaseModulationError(f"{name} must be an integer >= {minimum}")
    return value


def _signed_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ThreePhaseModulationError(f"{name} must be an integer")
    return value


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ThreePhaseModulationError(f"{name} must be a non-empty string")
    return value


def _same(members: list[Any], field: str, *, block_id: str) -> Any:
    values = [getattr(member, field, None) for member in members]
    first = values[0]
    if any(value != first for value in values[1:]):
        raise ThreePhaseModulationError(
            f"Three-phase block {block_id!r} has inconsistent {field} metadata"
        )
    return first


def _expanded_blocks(model: Any) -> list[dict[str, Any]]:
    controls = getattr(model, "controls", None)
    if not isinstance(controls, Mapping):
        raise ThreePhaseModulationError("Model controls must be a mapping")

    grouped: dict[str, list[Any]] = {}
    for control in controls.values():
        if getattr(control, "kind", None) != "three_phase_spwm":
            continue
        block_id = _text(getattr(control, "block_id", None), name="SPWM block_id")
        grouped.setdefault(block_id, []).append(control)
    if not grouped:
        raise ThreePhaseModulationError("No expanded three_phase_spwm block was found")

    components = getattr(model, "components", None)
    if not isinstance(components, tuple | list):
        raise ThreePhaseModulationError("Model components must be a sequence")
    switches_by_gate: dict[str, list[Any]] = {}
    for component in components:
        if getattr(component, "type", None) != "ideal_switch":
            continue
        gate_id = getattr(component, "control", None)
        if isinstance(gate_id, str):
            switches_by_gate.setdefault(gate_id, []).append(component)

    blocks: list[dict[str, Any]] = []
    for block_id, members in sorted(grouped.items()):
        if len(members) != 6:
            raise ThreePhaseModulationError(
                f"Three-phase block {block_id!r} must expand to exactly six gates"
            )
        by_phase_role: dict[tuple[str, str], Any] = {}
        for control in members:
            phase = getattr(control, "phase_label", None)
            role = getattr(control, "role", None)
            if phase not in _PHASES or role not in _ROLES:
                raise ThreePhaseModulationError(
                    f"Three-phase block {block_id!r} has invalid explicit phase/role metadata"
                )
            key = (phase, role)
            if key in by_phase_role:
                raise ThreePhaseModulationError(
                    f"Three-phase block {block_id!r} repeats phase/role {phase}.{role}"
                )
            expected_id = f"{block_id}.{phase}.{role}"
            if getattr(control, "id", None) != expected_id:
                raise ThreePhaseModulationError(
                    f"Three-phase block {block_id!r} gate ID disagrees with its explicit "
                    f"phase/role metadata: expected {expected_id!r}"
                )
            by_phase_role[key] = control
        if set(by_phase_role) != {(phase, role) for phase in _PHASES for role in _ROLES}:
            raise ThreePhaseModulationError(
                f"Three-phase block {block_id!r} lacks a complete a/b/c high/low mapping"
            )

        carrier_frequency = _finite(
            _same(members, "frequency", block_id=block_id),
            name=f"{block_id}.carrier_frequency",
        )
        fundamental_frequency = _finite(
            _same(members, "fundamental_frequency", block_id=block_id),
            name=f"{block_id}.fundamental_frequency",
        )
        modulation_index = _finite(
            _same(members, "modulation_index", block_id=block_id),
            name=f"{block_id}.modulation_index",
        )
        carrier_phase = _finite(
            _same(members, "phase", block_id=block_id),
            name=f"{block_id}.carrier_phase",
        )
        normalized_carrier_phase = _finite(
            _same(members, "normalized_phase", block_id=block_id),
            name=f"{block_id}.normalized_carrier_phase",
        )
        reference_phase = _finite(
            _same(members, "reference_phase", block_id=block_id),
            name=f"{block_id}.reference_phase",
        )
        dead_time = _finite(
            _same(members, "dead_time", block_id=block_id),
            name=f"{block_id}.dead_time",
        )
        carrier_ratio = _integer(
            _same(members, "carrier_ratio", block_id=block_id),
            name=f"{block_id}.carrier_ratio",
            minimum=1,
        )
        sequence = _text(
            _same(members, "sequence", block_id=block_id),
            name=f"{block_id}.sequence",
        )
        injection = _same(members, "injection", block_id=block_id) or "none"
        if injection not in {"none", "min_max"}:
            raise ThreePhaseModulationError(
                f"Three-phase block {block_id!r} has unknown reference "
                f"injection {injection!r}"
            )
        sampling = getattr(members[0], "sampling", _SAMPLING)
        if any(getattr(member, "sampling", _SAMPLING) != sampling for member in members):
            raise ThreePhaseModulationError(
                f"Three-phase block {block_id!r} has inconsistent sampling metadata"
            )
        if sampling != _SAMPLING:
            raise ThreePhaseModulationError(
                f"Three-phase block {block_id!r} has unsupported sampling {sampling!r}"
            )
        if carrier_frequency <= 0.0 or fundamental_frequency <= 0.0:
            raise ThreePhaseModulationError(
                "SPWM carrier and fundamental frequencies must be positive"
            )
        reference_peak = math.sqrt(3.0) / 2.0 if injection == "min_max" else 1.0
        if not 0.0 <= modulation_index * reference_peak < 1.0:
            raise ThreePhaseModulationError(
                "SPWM modulation_index is outside the linear range for "
                f"injection={injection!r}"
            )
        if dead_time < 0.0:
            raise ThreePhaseModulationError("SPWM dead_time must be non-negative")
        if sequence not in {"abc", "acb"}:
            raise ThreePhaseModulationError("SPWM sequence must be abc or acb")
        if not math.isclose(
            carrier_frequency / fundamental_frequency,
            float(carrier_ratio),
            rel_tol=1e-12,
            abs_tol=0.0,
        ):
            raise ThreePhaseModulationError(
                f"Three-phase block {block_id!r} carrier ratio disagrees with its frequencies"
            )
        expected_normalized_phase = carrier_phase % (1.0 / carrier_frequency)
        _require_float_match(
            normalized_carrier_phase,
            expected_normalized_phase,
            name=f"{block_id}.normalized_carrier_phase",
        )

        phase_rows: list[dict[str, Any]] = []
        positive_rail: str | None = None
        negative_rail: str | None = None
        pole_nodes: set[str] = set()
        gate_ids: set[str] = set()
        for phase in _PHASES:
            high = by_phase_role[(phase, "high")]
            low = by_phase_role[(phase, "low")]
            high_id = _text(getattr(high, "id", None), name=f"{block_id}.{phase}.high id")
            low_id = _text(getattr(low, "id", None), name=f"{block_id}.{phase}.low id")
            if (
                getattr(high, "partner_id", None) != low_id
                or getattr(low, "partner_id", None) != high_id
            ):
                raise ThreePhaseModulationError(
                    f"Three-phase block {block_id!r} phase {phase} partners are not reciprocal"
                )
            high_devices = switches_by_gate.get(high_id, [])
            low_devices = switches_by_gate.get(low_id, [])
            if len(high_devices) != 1 or len(low_devices) != 1:
                raise ThreePhaseModulationError(
                    f"Three-phase block {block_id!r} phase {phase} requires exactly one "
                    "explicit ideal switch per gate output"
                )
            high_device = high_devices[0]
            low_device = low_devices[0]
            high_p = _text(getattr(high_device, "p", None), name=f"{high_id} device.p")
            high_n = _text(getattr(high_device, "n", None), name=f"{high_id} device.n")
            low_p = _text(getattr(low_device, "p", None), name=f"{low_id} device.p")
            low_n = _text(getattr(low_device, "n", None), name=f"{low_id} device.n")
            if high_n != low_p:
                raise ThreePhaseModulationError(
                    f"Three-phase block {block_id!r} phase {phase} is not an oriented "
                    "high-rail -> pole -> low-rail half bridge"
                )
            if positive_rail is None:
                positive_rail, negative_rail = high_p, low_n
            elif positive_rail != high_p or negative_rail != low_n:
                raise ThreePhaseModulationError(
                    f"Three-phase block {block_id!r} phases do not share common DC rails"
                )
            if high_n in pole_nodes:
                raise ThreePhaseModulationError(
                    f"Three-phase block {block_id!r} repeats pole node {high_n!r}"
                )
            pole_nodes.add(high_n)
            gate_ids.update((high_id, low_id))
            phase_rows.append(
                {
                    "phase": phase,
                    "pole_node": high_n,
                    "high": {
                        "gate_output_id": high_id,
                        "device_id": _text(
                            getattr(high_device, "id", None), name=f"{high_id} device.id"
                        ),
                        "p": high_p,
                        "n": high_n,
                    },
                    "low": {
                        "gate_output_id": low_id,
                        "device_id": _text(
                            getattr(low_device, "id", None), name=f"{low_id} device.id"
                        ),
                        "p": low_p,
                        "n": low_n,
                    },
                }
            )
        if (
            positive_rail == negative_rail
            or positive_rail in pole_nodes
            or negative_rail in pole_nodes
        ):
            raise ThreePhaseModulationError(
                f"Three-phase block {block_id!r} has inconsistent pole and rail nodes"
            )

        blocks.append(
            {
                "block_id": block_id,
                "config": {
                    "control_type": "three_phase_spwm",
                    "carrier_frequency_hz": carrier_frequency,
                    "fundamental_frequency_hz": fundamental_frequency,
                    "carrier_ratio": carrier_ratio,
                    "modulation_index": modulation_index,
                    "modulation_index_definition": "phase_reference_peak_normalized_to_carrier_peak",
                    "reference_injection": injection,
                    "linear_region_limit": (
                        2.0 / math.sqrt(3.0) if injection == "min_max" else 1.0
                    ),
                    "carrier_phase_s": carrier_phase,
                    "normalized_carrier_phase_s": normalized_carrier_phase,
                    "reference_phase_s": reference_phase,
                    "phase_sequence": sequence,
                    "sampling": sampling,
                    "dead_time_s": dead_time,
                    "edge_policy": "turn_off_immediate_turn_on_after_dead_time",
                    "interlock": "intrinsic_non_disableable",
                },
                "bridge": {
                    "positive_rail_node": positive_rail,
                    "negative_rail_node": negative_rail,
                    "phases": phase_rows,
                    "mapping_source": (
                        "expanded_control_metadata_and_explicit_ideal_switch_p_n_references"
                    ),
                },
                "gate_output_ids": sorted(gate_ids),
            }
        )
    return blocks


def _phase_numbers(value: Any, *, name: str, duties: bool = False) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(_PHASES):
        raise ThreePhaseModulationError(f"{name} must contain exactly phases a, b, c")
    result = {phase: _finite(value[phase], name=f"{name}.{phase}") for phase in _PHASES}
    if duties and any(not 0.0 <= item <= 1.0 for item in result.values()):
        raise ThreePhaseModulationError(f"{name} duties must be in [0, 1]")
    return result


def _command_pulses(value: Any, *, name: str) -> dict[str, dict[str, float]]:
    if not isinstance(value, Mapping) or set(value) != set(_PHASES):
        raise ThreePhaseModulationError(f"{name} must contain exactly phases a, b, c")
    fields = ("start_s", "end_s", "high_turn_on_s", "low_turn_on_s")
    result: dict[str, dict[str, float]] = {}
    for phase in _PHASES:
        record = value[phase]
        if not isinstance(record, Mapping) or set(record) != set(fields):
            raise ThreePhaseModulationError(
                f"{name}.{phase} must contain exactly {', '.join(fields)}"
            )
        normalized = {
            field: _finite(record[field], name=f"{name}.{phase}.{field}") for field in fields
        }
        if normalized["end_s"] <= normalized["start_s"]:
            raise ThreePhaseModulationError(f"{name}.{phase} has a non-positive pulse")
        result[phase] = normalized
    return result


def normalized_modulation_cycles(result: Any) -> list[dict[str, Any]]:
    """Validate and canonicalize the authoritative per-carrier command records."""

    blocks = {block["block_id"]: block for block in _expanded_blocks(result.model)}
    raw_cycles = getattr(result, "modulation_cycles", None)
    if not isinstance(raw_cycles, list):
        raise ThreePhaseModulationError(
            "three_phase_spwm result must expose modulation_cycles as a list"
        )
    if len(raw_cycles) > MAX_MODULATION_CYCLE_ROWS:
        raise ThreePhaseModulationError(
            f"modulation_cycles exceeds the {MAX_MODULATION_CYCLE_ROWS}-row limit"
        )
    if not raw_cycles:
        raise ThreePhaseModulationError("three_phase_spwm result contains no modulation cycles")

    run_start = _finite(result.model.transient.start, name="transient.start")
    run_stop = _finite(result.model.transient.stop, name="transient.stop")
    previous_by_block: dict[str, tuple[int, float]] = {}
    normalized: list[dict[str, Any]] = []
    for row_index, raw in enumerate(raw_cycles):
        if not isinstance(raw, Mapping):
            raise ThreePhaseModulationError(f"modulation cycle {row_index} must be an object")
        block_id = _text(raw.get("block_id"), name=f"modulation cycle {row_index}.block_id")
        block = blocks.get(block_id)
        if block is None:
            raise ThreePhaseModulationError(
                f"modulation cycle {row_index} references unknown block {block_id!r}"
            )
        cycle_index = _signed_integer(
            raw.get("cycle_index"), name=f"modulation cycle {row_index}.cycle_index"
        )
        frame_index = _integer(
            raw.get("frame_index"), name=f"modulation cycle {row_index}.frame_index"
        )
        start = _finite(raw.get("start_s"), name=f"modulation cycle {row_index}.start_s")
        end = _finite(raw.get("end_s"), name=f"modulation cycle {row_index}.end_s")
        sample = _finite(raw.get("sample_s"), name=f"modulation cycle {row_index}.sample_s")
        if end <= start or not start <= sample < end:
            raise ThreePhaseModulationError(
                f"modulation cycle {row_index} has invalid half-open timing"
            )
        if end <= run_start or start >= run_stop:
            raise ThreePhaseModulationError(
                f"modulation cycle {row_index} does not overlap the run window"
            )
        previous = previous_by_block.get(block_id)
        if previous is not None and (cycle_index <= previous[0] or start < previous[1]):
            raise ThreePhaseModulationError(
                f"modulation cycles for {block_id!r} are not strictly ordered"
            )
        previous_by_block[block_id] = (cycle_index, start)

        config = block["config"]
        phase_sequence = _text(
            raw.get("phase_sequence"), name=f"modulation cycle {row_index}.phase_sequence"
        )
        sampling = _text(raw.get("sampling"), name=f"modulation cycle {row_index}.sampling")
        dead_time = _finite(
            raw.get("dead_time_s"), name=f"modulation cycle {row_index}.dead_time_s"
        )
        if phase_sequence != config["phase_sequence"] or sampling != config["sampling"]:
            raise ThreePhaseModulationError(
                f"modulation cycle {row_index} disagrees with block sequence/sampling"
            )
        if dead_time != config["dead_time_s"]:
            raise ThreePhaseModulationError(
                f"modulation cycle {row_index} disagrees with block dead time"
            )
        references = _phase_numbers(
            raw.get("references"), name=f"modulation cycle {row_index}.references"
        )
        duties = _phase_numbers(
            raw.get("duties"),
            name=f"modulation cycle {row_index}.duties",
            duties=True,
        )
        command_pulses = _command_pulses(
            raw.get("command_pulses"),
            name=f"modulation cycle {row_index}.command_pulses",
        )
        electrical_angle = _finite(
            raw.get("electrical_angle_rad"),
            name=f"modulation cycle {row_index}.electrical_angle_rad",
        )

        period = 1.0 / config["carrier_frequency_hz"]
        expected_frame = cycle_index % config["carrier_ratio"]
        if frame_index != expected_frame:
            raise ThreePhaseModulationError(
                f"modulation cycle {row_index}.frame_index disagrees with "
                "cycle_index modulo carrier_ratio"
            )
        carrier_phase = config["normalized_carrier_phase_s"]
        expected_start = carrier_phase + cycle_index * period
        expected_end = carrier_phase + (cycle_index + 1) * period
        expected_sample = expected_start + 0.5 * period
        _require_float_match(start, expected_start, name=f"modulation cycle {row_index}.start_s")
        _require_float_match(end, expected_end, name=f"modulation cycle {row_index}.end_s")
        _require_float_match(sample, expected_sample, name=f"modulation cycle {row_index}.sample_s")

        offset_cycles = (carrier_phase - config["reference_phase_s"]) * config[
            "fundamental_frequency_hz"
        ]
        offset_cycles %= 1.0
        electrical_cycles = (offset_cycles + (expected_frame + 0.5) / config["carrier_ratio"]) % 1.0
        expected_angle = math.tau * electrical_cycles
        _require_float_match(
            electrical_angle,
            expected_angle,
            name=f"modulation cycle {row_index}.electrical_angle_rad",
        )
        expected_references = {
            "a": math.sin(expected_angle),
            "b": math.sin(expected_angle - 2.0 * math.pi / 3.0),
            "c": math.sin(expected_angle + 2.0 * math.pi / 3.0),
        }
        if config["phase_sequence"] == "acb":
            expected_references["b"], expected_references["c"] = (
                expected_references["c"],
                expected_references["b"],
            )
        if config["reference_injection"] == "min_max":
            ordered = (
                expected_references["a"],
                expected_references["b"],
                expected_references["c"],
            )
            offset = -0.5 * (max(ordered) + min(ordered))
            expected_references = {
                phase: value + offset
                for phase, value in expected_references.items()
            }
        expected_duties = {
            phase: 0.5 * (1.0 + config["modulation_index"] * reference)
            for phase, reference in expected_references.items()
        }
        for phase in _PHASES:
            _require_float_match(
                references[phase],
                expected_references[phase],
                name=f"modulation cycle {row_index}.references.{phase}",
            )
            _require_float_match(
                duties[phase],
                expected_duties[phase],
                name=f"modulation cycle {row_index}.duties.{phase}",
            )
            expected_command_start = expected_start + 0.5 * (1.0 - expected_duties[phase]) * period
            expected_command_end = expected_start + 0.5 * (1.0 + expected_duties[phase]) * period
            expected_pulse = {
                "start_s": expected_command_start,
                "end_s": expected_command_end,
                "high_turn_on_s": expected_command_start + dead_time,
                "low_turn_on_s": expected_command_end + dead_time,
            }
            for field, expected in expected_pulse.items():
                _require_float_match(
                    command_pulses[phase][field],
                    expected,
                    name=f"modulation cycle {row_index}.command_pulses.{phase}.{field}",
                )

        normalized.append(
            {
                "schema": MODULATION_CYCLE_SCHEMA,
                "block_id": block_id,
                "cycle_index": cycle_index,
                "frame_index": frame_index,
                "start_s": start,
                "end_s": end,
                "sample_s": sample,
                "electrical_angle_rad": electrical_angle,
                "references": references,
                "duties": duties,
                "command_pulses": command_pulses,
                "phase_sequence": phase_sequence,
                "sampling": sampling,
                "dead_time_s": dead_time,
            }
        )
    missing = sorted(set(blocks) - {row["block_id"] for row in normalized})
    if missing:
        raise ThreePhaseModulationError(
            "No modulation cycles were recorded for block(s): " + ", ".join(missing)
        )
    for block_id in sorted(blocks):
        rows = [row for row in normalized if row["block_id"] == block_id]
        for previous, current in zip(rows[:-1], rows[1:], strict=True):
            if current["cycle_index"] != previous["cycle_index"] + 1:
                raise ThreePhaseModulationError(
                    f"modulation cycles for {block_id!r} are not contiguous"
                )
            _require_float_match(
                current["start_s"],
                previous["end_s"],
                name=f"modulation cycles for {block_id!r} cell boundary",
            )
        first = rows[0]
        last = rows[-1]
        if not first["start_s"] <= run_start < first["end_s"]:
            raise ThreePhaseModulationError(
                f"modulation cycles for {block_id!r} do not cover the run start"
            )
        if not last["start_s"] < run_stop <= last["end_s"]:
            raise ThreePhaseModulationError(
                f"modulation cycles for {block_id!r} do not cover the run stop"
            )
    return normalized


def _evenly_retained(rows: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if len(rows) <= maximum:
        return rows
    indices = sorted({round(index * (len(rows) - 1) / (maximum - 1)) for index in range(maximum)})
    return [rows[index] for index in indices]


def _block_projection(
    block: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    run_start: float,
    run_stop: float,
    edge_counts: Counter[str],
) -> dict[str, Any]:
    config = block["config"]
    fundamental_period = 1.0 / config["fundamental_frequency_hz"]
    requested_start = run_stop - fundamental_period
    window_start = max(run_start, requested_start)
    selection = (
        "last_fundamental_period_ending_at_run_stop"
        if requested_start >= run_start
        else "complete_run_shorter_than_one_fundamental_period"
    )
    in_window = [row for row in rows if window_start <= row["sample_s"] < run_stop]
    retained = _evenly_retained(in_window, MAX_MODULATION_PROJECTION_CYCLES)
    projection_sampling = (
        "all_command_cycles_in_window"
        if len(in_window) <= MAX_MODULATION_PROJECTION_CYCLES
        else "deterministic_even_index_retention"
    )
    if not retained:
        raise ThreePhaseModulationError(
            f"Three-phase block {block['block_id']!r} has no command sample in display window"
        )

    carrier_period = 1.0 / config["carrier_frequency_hz"]
    dead_time = config["dead_time_s"]
    all_duties = [value for row in rows for value in row["duties"].values()]
    all_references = [value for row in rows for value in row["references"].values()]

    def minimum_effective_pulse(row: dict[str, Any]) -> float:
        return min(
            min(duty, 1.0 - duty) * carrier_period - dead_time for duty in row["duties"].values()
        )

    pulse_values = [minimum_effective_pulse(row) for row in rows]
    if min(pulse_values) <= 0.0:
        raise ThreePhaseModulationError(
            f"Three-phase block {block['block_id']!r} contains a deleted effective pulse"
        )
    zoom_index = min(
        range(len(rows)), key=lambda index: (pulse_values[index], rows[index]["sample_s"])
    )
    zoom = rows[zoom_index]
    gate_counts = {gate_id: int(edge_counts[gate_id]) for gate_id in block["gate_output_ids"]}
    return {
        **block,
        "summary": {
            "source_cycle_count": len(rows),
            "scheduled_gate_edge_count": sum(gate_counts.values()),
            "scheduled_gate_edges_by_output": gate_counts,
            "minimum_duty": min(all_duties),
            "maximum_duty": max(all_duties),
            "minimum_reference": min(all_references),
            "maximum_reference": max(all_references),
            "minimum_effective_pulse_s": min(pulse_values),
            "maximum_effective_pulse_s": max(pulse_values),
            "observed_overlap_count": 0,
            "acceptance_status": "not_evaluated",
        },
        "fundamental_period_projection": {
            "selection": selection,
            "window_start_s": window_start,
            "window_end_s": run_stop,
            "window_duration_s": run_stop - window_start,
            "boundary_convention": "cycle_selected_by_sample_in_[start,end)",
            "projection_sampling": projection_sampling,
            "retention": {
                "source_artifact": "modulation-cycles.jsonl",
                "source_scope": "full_run_overlapping_carrier_cells",
                "source_cycles_for_block": len(rows),
                "source_cycles_in_selected_window": len(in_window),
                "retained_cycles": len(retained),
                "truncated": len(retained) < len(rows),
                "display_decimated": len(retained) < len(in_window),
                "statement": (
                    "This bounded command projection is for presentation; "
                    "modulation-cycles.jsonl remains authoritative for every recorded "
                    "carrier command in the run."
                ),
            },
            "cycles": retained,
        },
        "carrier_zoom": {
            "selection": "minimum_effective_pulse_in_full_run_earliest_tie",
            "cycle": zoom,
            "minimum_effective_pulse_s": pulse_values[zoom_index],
            "source_artifact": "modulation-cycles.jsonl",
        },
    }


def _validated_realized_gate_events(
    result: Any,
    blocks: list[dict[str, Any]],
    cycles: list[dict[str, Any]],
    *,
    run_start: float,
    run_stop: float,
) -> Counter[str]:
    """Bind every stored SPWM command edge to the realized scheduled-event log."""

    phase_gates = {
        (block["block_id"], row["phase"]): (
            row["high"]["gate_output_id"],
            row["low"]["gate_output_id"],
        )
        for block in blocks
        for row in block["bridge"]["phases"]
    }
    gate_ids = {gate_id for high_low in phase_gates.values() for gate_id in high_low}
    expected: list[tuple[float, str, bool]] = []
    for cycle in cycles:
        block_id = cycle["block_id"]
        for phase in _PHASES:
            high_id, low_id = phase_gates[(block_id, phase)]
            pulse = cycle["command_pulses"][phase]
            candidates = (
                (pulse["start_s"], low_id, False),
                (pulse["high_turn_on_s"], high_id, True),
                (pulse["end_s"], high_id, False),
                (pulse["low_turn_on_s"], low_id, True),
            )
            expected.extend(
                candidate for candidate in candidates if run_start < candidate[0] <= run_stop
            )
    expected.sort(key=lambda edge: (edge[0], edge[2], edge[1]))

    events = getattr(result, "events", None)
    if not isinstance(events, list):
        raise ThreePhaseModulationError("Simulation result events must be a list")
    actual = [
        event
        for event in events
        if getattr(event, "kind", None) == "scheduled_gate"
        and getattr(event, "target", None) in gate_ids
    ]
    if len(actual) != len(expected):
        raise ThreePhaseModulationError(
            "Realized scheduled SPWM event count disagrees with deterministic "
            f"commands: got {len(actual)}, expected {len(expected)}"
        )

    intervals = getattr(result, "mode_intervals", None)
    if not isinstance(intervals, list) or not intervals:
        raise ThreePhaseModulationError(
            "Simulation result must expose non-empty mode intervals for gate replay"
        )
    first_interval = intervals[0]
    if not isinstance(first_interval, Mapping):
        raise ThreePhaseModulationError("First mode interval must be an object")
    interval_start = _finite(first_interval.get("start_s"), name="first mode interval.start_s")
    _require_float_match(interval_start, run_start, name="first mode interval.start_s")
    initial_gates = first_interval.get("gates")
    if not isinstance(initial_gates, Mapping):
        raise ThreePhaseModulationError("First mode interval has no gate-state mapping")
    replay: dict[str, bool] = {}
    for gate_id in sorted(gate_ids):
        value = initial_gates.get(gate_id)
        if not isinstance(value, bool):
            raise ThreePhaseModulationError(
                f"First mode interval lacks boolean initial state for {gate_id!r}"
            )
        replay[gate_id] = value

    counts: Counter[str] = Counter()
    for event_index, (event, expected_edge) in enumerate(zip(actual, expected, strict=True)):
        expected_time, expected_target, expected_current = expected_edge
        actual_time = _finite(
            getattr(event, "time", None),
            name=f"scheduled SPWM event {event_index}.time",
        )
        actual_target = getattr(event, "target", None)
        actual_current = getattr(event, "current", None)
        actual_previous = getattr(event, "previous", None)
        if actual_target != expected_target or actual_current is not expected_current:
            raise ThreePhaseModulationError(
                f"Scheduled SPWM event {event_index} target/state disagrees with "
                "the deterministic command record"
            )
        _require_float_match(
            actual_time,
            expected_time,
            name=f"scheduled SPWM event {event_index}.time",
        )
        if not isinstance(actual_previous, bool) or actual_previous is not replay[expected_target]:
            raise ThreePhaseModulationError(
                f"Scheduled SPWM event {event_index} previous state disagrees with gate replay"
            )
        if actual_previous is actual_current:
            raise ThreePhaseModulationError(
                f"Scheduled SPWM event {event_index} is not a boolean transition"
            )
        replay[expected_target] = actual_current
        counts[expected_target] += 1
    return counts


def build_three_phase_modulation_observation(
    result: Any,
    *,
    normalized_cycles: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic bounded observation for one accepted SPWM run."""

    blocks = _expanded_blocks(result.model)
    cycles = (
        normalized_modulation_cycles(result) if normalized_cycles is None else normalized_cycles
    )
    run_start = _finite(result.model.transient.start, name="transient.start")
    run_stop = _finite(result.model.transient.stop, name="transient.stop")
    if run_stop <= run_start:
        raise ThreePhaseModulationError("Transient stop must be greater than start")

    controls_by_id = result.model.controls
    for interval_index, interval in enumerate(getattr(result, "mode_intervals", ())):
        gates = interval.get("gates") if isinstance(interval, Mapping) else None
        if not isinstance(gates, Mapping):
            raise ThreePhaseModulationError(
                f"Mode interval {interval_index} has no gate-state mapping"
            )
        for block in blocks:
            for phase_row in block["bridge"]["phases"]:
                high_id = phase_row["high"]["gate_output_id"]
                low_id = phase_row["low"]["gate_output_id"]
                if high_id not in controls_by_id or low_id not in controls_by_id:
                    raise ThreePhaseModulationError("Bridge mapping references an unknown gate")
                if not isinstance(gates.get(high_id), bool) or not isinstance(
                    gates.get(low_id), bool
                ):
                    raise ThreePhaseModulationError(
                        f"Mode interval {interval_index} lacks boolean states for {high_id}/{low_id}"
                    )
                if gates[high_id] and gates[low_id]:
                    raise ThreePhaseModulationError(
                        f"Complementary overlap in mode interval {interval_index}: "
                        f"{high_id!r} and {low_id!r}"
                    )

    edge_counts = _validated_realized_gate_events(
        result,
        blocks,
        cycles,
        run_start=run_start,
        run_stop=run_stop,
    )

    rows_by_block: dict[str, list[dict[str, Any]]] = {block["block_id"]: [] for block in blocks}
    for cycle in cycles:
        rows_by_block[cycle["block_id"]].append(cycle)
    projected_blocks = [
        _block_projection(
            block,
            rows_by_block[block["block_id"]],
            run_start=run_start,
            run_stop=run_stop,
            edge_counts=edge_counts,
        )
        for block in blocks
    ]
    hashes = model_hashes(result.model)
    return {
        "schema": THREE_PHASE_MODULATION_SCHEMA,
        "kind": THREE_PHASE_MODULATION_KIND,
        "claim": {
            "classification": "simulation-observation",
            "validation_claim": False,
            "hardware_sign_off_allowed": False,
            "statement": (
                "Recorded sampled-SPWM commands, realized scheduling counters, and "
                "explicit ideal-switch bindings only; this artifact does not establish "
                "physical driver timing, device behavior, or inverter validation."
            ),
        },
        "source_binding": {
            "model_sha256": hashes["model_sha256"],
            "request_sha256": hashes["request_sha256"],
            "raw_command_artifact": "modulation-cycles.jsonl",
            "raw_gate_event_artifact": "events.jsonl",
            "mapping_source": (
                "expanded_control_metadata_and_explicit_ideal_switch_control_p_n_fields"
            ),
            "command_projection_source": "validated_result_modulation_cycles",
        },
        "run_window": {"start_s": run_start, "stop_s": run_stop},
        "blocks": projected_blocks,
        "summary": {
            "block_count": len(projected_blocks),
            "phase_count": 3 * len(projected_blocks),
            "gate_output_count": 6 * len(projected_blocks),
            "source_cycle_count": len(cycles),
            "scheduled_gate_edge_count": sum(edge_counts.values()),
            "observed_overlap_count": 0,
            "acceptance_status": "not_evaluated",
        },
    }


__all__ = [
    "MAX_MODULATION_CYCLE_ROWS",
    "MAX_MODULATION_PROJECTION_CYCLES",
    "MODULATION_CYCLE_SCHEMA",
    "THREE_PHASE_MODULATION_KIND",
    "THREE_PHASE_MODULATION_SCHEMA",
    "ThreePhaseModulationError",
    "build_three_phase_modulation_observation",
    "has_three_phase_spwm",
    "normalized_modulation_cycles",
]
