"""Bounded, retained one-dimensional parameter sweeps for agent experiments.

The producer is intentionally strict about its request and deliberately separates
simulation execution from user-declared acceptance.  Every requested point receives
an immutable point record, including points whose override or simulation failed.
Human reports only present recorded JSON; they do not recompute acceptance.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
import hashlib
import html
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
from time import perf_counter
from typing import Any

import pear_sim
from pear_sim import __version__
from pear_sim._jsonutil import (
    reject_json_constant as _reject_json_constant,
    unique_json_object as _unique_json_object,
)
from pear_sim.spec.signals import is_signal_expression, signal_unit
from pear_sim.run.contracts import model_hashes, sha256_file, sha256_json
from pear_sim.evidence.expectations import validate_expectations
from pear_sim.evidence.evidence_io import evaluate_run
from pear_sim.evidence.parallel_points import resolve_jobs, run_ordered
from pear_sim.limits import MAX_SCHEDULED_GATE_EDGES
from pear_sim.spec.model import Model, load_model
from pear_sim.reporting.render import render_schematic
from pear_sim.reporting.report_svg import sanitize_schematic_svg
from pear_sim.spec.units import parse_quantity


PARAMETER_SWEEP_REQUEST_SCHEMA = "pear-sim/parameter-sweep-request/v0.1"
PARAMETER_SWEEP_NORMALIZED_REQUEST_SCHEMA = "pear-sim/parameter-sweep-request-normalized/v0.1"
PARAMETER_SWEEP_SOURCE_MODEL_SCHEMA = "pear-sim/parameter-sweep-source-model/v0.1"
PARAMETER_SWEEP_POINT_SCHEMA = "pear-sim/parameter-sweep-point/v0.1"
PARAMETER_SWEEP_SUMMARY_SCHEMA = "pear-sim/parameter-sweep-summary/v0.1"
PARAMETER_SWEEP_PROJECTIONS_SCHEMA = "pear-sim/parameter-sweep-projections/v0.1"
PARAMETER_SWEEP_FAILURES_SCHEMA = "pear-sim/parameter-sweep-failures/v0.1"
PARAMETER_SWEEP_CONFIDENCE_SCHEMA = "pear-sim/confidence/parameter-sweep/v0.1"
PARAMETER_SWEEP_REPORT_SCHEMA = "pear-sim/parameter-sweep-report/v0.1"
MAX_PARAMETER_SWEEP_POINTS = 64
MAX_PARAMETER_SWEEP_PROJECTIONS = 32
MAX_PARAMETER_SWEEP_SAMPLES_PER_POINT = 1_000_001
MAX_PARAMETER_SWEEP_TOTAL_SAMPLES = 4_000_000
MAX_PARAMETER_SWEEP_SCHEDULED_EDGES_PER_POINT = MAX_SCHEDULED_GATE_EDGES
MAX_PARAMETER_SWEEP_TOTAL_SCHEDULED_EDGES = 4_000_000
MAX_PARAMETER_SWEEP_OUTPUTS = 256
MAX_PARAMETER_SWEEP_COMPONENTS = 128
MAX_PARAMETER_SWEEP_STATES = 64
MAX_PARAMETER_SWEEP_CONTROLS = 32
MAX_PARAMETER_SWEEP_SOURCES = 32
MAX_PARAMETER_SWEEP_PARAMETERS = 256
MAX_PARAMETER_SWEEP_MAX_STEP_INTERVALS_PER_POINT = 250_000
MAX_PARAMETER_SWEEP_TOTAL_MAX_STEP_INTERVALS = 1_000_000
MAX_PARAMETER_SWEEP_SAMPLE_OUTPUT_CELLS_PER_POINT = 4_000_000
MAX_PARAMETER_SWEEP_TOTAL_SAMPLE_OUTPUT_CELLS = 16_000_000
MAX_PARAMETER_SWEEP_POINT_WALL_TIME_S = 60.0
MAX_PARAMETER_SWEEP_TOTAL_WALL_TIME_S = 600.0
MAX_INPUT_BYTES = 2 * 1024 * 1024
MAX_METRICS_BYTES = 64 * 1024 * 1024
MAX_TEXT_LENGTH = 4096
MAX_JSON_NESTING_DEPTH = 64

# These are the exact numeric selectors understood by the existing v0.1 evidence
# evaluator.  Projection is observation-only: it performs no aliasing, sign changes,
# interpolation, aggregation, or unit conversion.
_PROJECTION_METRICS = frozenset(
    {
        "final",
        "max",
        "max_last_half",
        "mean",
        "mean_last_half",
        "min",
        "min_last_half",
        "peak_to_peak",
        "peak_to_peak_last_half",
        "ripple",
        "ripple_last_half",
        "rms",
        "rms_last_half",
        "sample_final",
        "sample_max",
        "sample_max_last_half",
        "sample_mean",
        "sample_mean_last_half",
        "sample_min",
        "sample_min_last_half",
        "sample_peak_to_peak",
        "sample_peak_to_peak_last_half",
        "sample_rms",
        "sample_rms_last_half",
    }
)
_REQUEST_FIELDS = frozenset({"schema", "parameter", "values", "projections"})
_PROJECTION_FIELDS = frozenset({"id", "signal", "metric", "unit"})
_PROJECTION_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:-]*$")


class ParameterSweepError(ValueError):
    """A parameter-sweep request or artifact contract is invalid."""

    code = "sweep.parameter_invalid"


@dataclass(frozen=True)
class ParameterSweepArtifacts:
    """Paths and recorded statuses produced by :func:`run_parameter_sweep`."""

    output_dir: Path
    source_request: Path
    normalized_request: Path
    source_model: Path
    sweep_summary: Path
    projections: Path
    failures: Path
    confidence_json: Path
    report_json: Path
    report_html: Path
    artifact_manifest: Path
    execution_status: str
    acceptance_status: str
    accepted: bool | None
    overall_status: str


@dataclass(frozen=True)
class _NormalizedRequest:
    parameter: str
    dimension: str
    base_value_si: float
    values_si: tuple[float, ...]
    projections: tuple[dict[str, str], ...]
    payload: dict[str, Any]


def _require_portable_json(value: Any, *, path: str = "$") -> None:
    pending: list[tuple[Any, str, int]] = [(value, path, 0)]
    while pending:
        item, item_path, depth = pending.pop()
        if depth > MAX_JSON_NESTING_DEPTH:
            raise ValueError(
                f"JSON nesting exceeds the {MAX_JSON_NESTING_DEPTH}-level limit "
                f"at {item_path}"
            )
        if isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    "text is not a valid UTF-8 Unicode scalar sequence "
                    f"at {item_path}"
                ) from exc
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            try:
                number = float(item)
            except (OverflowError, ValueError) as exc:
                raise ValueError(
                    f"number outside the finite float contract at {item_path}"
                ) from exc
            if not math.isfinite(number):
                raise ValueError(f"non-finite number at {item_path}")
        elif isinstance(item, dict):
            for key, child in item.items():
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ValueError(
                        "object key is not a valid UTF-8 Unicode scalar sequence "
                        f"at {item_path}"
                    ) from exc
                pending.append((child, f"{item_path}.{key}", depth + 1))
        elif isinstance(item, list):
            pending.extend(
                (child, f"{item_path}[{index}]", depth + 1)
                for index, child in enumerate(item)
            )


def _read_strict_json(
    path: Path,
    *,
    role: str,
    maximum_bytes: int = MAX_INPUT_BYTES,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_bounded_regular_bytes(
        path,
        role=role,
        maximum_bytes=maximum_bytes,
    )
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise ParameterSweepError(f"Cannot parse strict {role} JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ParameterSweepError(f"{role} root must be an object")
    try:
        _require_portable_json(payload)
    except ValueError as exc:
        raise ParameterSweepError(f"Invalid {role}: {exc}") from exc
    return payload, raw


def _read_bounded_regular_bytes(
    path: Path,
    *,
    role: str,
    maximum_bytes: int,
) -> bytes:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ParameterSweepError(f"Cannot inspect {role} file {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ParameterSweepError(f"{role} must be a regular non-symlink file: {path}")
    if info.st_size > maximum_bytes:
        raise ParameterSweepError(
            f"{role} exceeds the {maximum_bytes}-byte contract limit: {path}"
        )
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ParameterSweepError(f"Cannot read {role} file {path}: {exc}") from exc
    if len(raw) > maximum_bytes:
        raise ParameterSweepError(
            f"{role} exceeds the {maximum_bytes}-byte contract limit: {path}"
        )
    return raw


def _require_exact_fields(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    name: str,
    required: frozenset[str],
) -> None:
    if any(not isinstance(key, str) for key in value):
        raise ParameterSweepError(f"{name}: every field name must be a string")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ParameterSweepError(f"{name}: unknown field(s): {', '.join(unknown)}")
    missing = sorted(required - set(value))
    if missing:
        raise ParameterSweepError(f"{name}: missing field(s): {', '.join(missing)}")


def _require_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ParameterSweepError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ParameterSweepError(f"{name} must not contain leading or trailing whitespace")
    if len(value) > MAX_TEXT_LENGTH:
        raise ParameterSweepError(f"{name} exceeds the {MAX_TEXT_LENGTH}-character limit")
    return value


def _require_projection_id(value: Any, *, name: str) -> str:
    projection_id = _require_text(value, name=name)
    if _PROJECTION_ID_RE.fullmatch(projection_id) is None:
        raise ParameterSweepError(
            f"{name} must match {_PROJECTION_ID_RE.pattern!r}"
        )
    return projection_id


def _expected_signal_unit(signal: str) -> str | None:
    # None means "not a signal expression", which the caller treats as
    # "no unit expectation to check" — distinct from an unknown unit.
    if not is_signal_expression(signal):
        return None
    return signal_unit(signal)


def _normalize_request(
    request: Mapping[str, Any],
    *,
    source_model: Model,
    source_sha256: str,
) -> _NormalizedRequest:
    _require_exact_fields(
        request,
        _REQUEST_FIELDS,
        name="parameter sweep request",
        required=frozenset({"schema", "parameter", "values"}),
    )
    if request["schema"] != PARAMETER_SWEEP_REQUEST_SCHEMA:
        raise ParameterSweepError(
            f"Unsupported request schema {request['schema']!r}; "
            f"expected {PARAMETER_SWEEP_REQUEST_SCHEMA!r}"
        )
    parameter = _require_text(request["parameter"], name="parameter")
    if parameter not in source_model.parameters:
        raise ParameterSweepError(f"Cannot sweep unknown model parameter {parameter!r}")
    base_quantity = source_model.parameters[parameter]
    raw_values = request["values"]
    if not isinstance(raw_values, list):
        raise ParameterSweepError("values must be an array")
    if not 1 <= len(raw_values) <= MAX_PARAMETER_SWEEP_POINTS:
        raise ParameterSweepError(
            f"values must contain 1..{MAX_PARAMETER_SWEEP_POINTS} sweep points"
        )
    values: list[float] = []
    for index, raw in enumerate(raw_values):
        if isinstance(raw, str) and len(raw) > 256:
            raise ParameterSweepError(f"values[{index}] exceeds the 256-character limit")
        try:
            quantity = parse_quantity(
                raw,
                base_quantity.dimension,
                name=f"values[{index}]",
            )
        except Exception as exc:
            raise ParameterSweepError(str(exc)) from exc
        value = float(quantity.value)
        if not math.isfinite(value):
            raise ParameterSweepError(f"values[{index}] must normalize to a finite SI value")
        if any(value == previous for previous in values):
            raise ParameterSweepError(
                f"values[{index}] duplicates an earlier normalized SI value {value:.17g}"
            )
        values.append(value)

    raw_projections = request.get("projections", [])
    if not isinstance(raw_projections, list):
        raise ParameterSweepError("projections must be an array")
    if len(raw_projections) > MAX_PARAMETER_SWEEP_PROJECTIONS:
        raise ParameterSweepError(
            f"projections exceeds the {MAX_PARAMETER_SWEEP_PROJECTIONS}-selector contract limit"
        )
    projections: list[dict[str, str]] = []
    ids: set[str] = set()
    selectors: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(raw_projections):
        name = f"projections[{index}]"
        if not isinstance(raw, Mapping):
            raise ParameterSweepError(f"{name} must be an object")
        _require_exact_fields(
            raw,
            _PROJECTION_FIELDS,
            name=name,
            required=_PROJECTION_FIELDS,
        )
        projection_id = _require_projection_id(raw["id"], name=f"{name}.id")
        signal = _require_text(raw["signal"], name=f"{name}.signal")
        metric = _require_text(raw["metric"], name=f"{name}.metric")
        unit = _require_text(raw["unit"], name=f"{name}.unit")
        if projection_id in ids:
            raise ParameterSweepError(f"projections contains duplicate id {projection_id!r}")
        if signal not in source_model.outputs:
            raise ParameterSweepError(
                f"{name}.signal {signal!r} is not an exact source-model output"
            )
        if metric not in _PROJECTION_METRICS:
            raise ParameterSweepError(
                f"{name}.metric {metric!r} is not a supported exact metric selector"
            )
        expected_unit = _expected_signal_unit(signal)
        if expected_unit is None or unit != expected_unit:
            raise ParameterSweepError(
                f"{name}.unit {unit!r} does not exactly match signal unit {expected_unit!r}"
            )
        selector = (signal, metric, unit)
        if selector in selectors:
            raise ParameterSweepError(
                f"{name} duplicates an earlier exact signal/metric/unit selector"
            )
        ids.add(projection_id)
        selectors.add(selector)
        projections.append({"id": projection_id, "signal": signal, "metric": metric, "unit": unit})

    normalized = {
        "schema": PARAMETER_SWEEP_NORMALIZED_REQUEST_SCHEMA,
        "source": {
            "schema": PARAMETER_SWEEP_REQUEST_SCHEMA,
            "artifact": "request.source.json",
            "sha256": source_sha256,
        },
        "parameter": {
            "name": parameter,
            "dimension": base_quantity.dimension,
            "base_value_si": base_quantity.value,
        },
        "values": [{"index": index, "value_si": value} for index, value in enumerate(values)],
        "projections": projections,
        "limits": {
            "maximum_points": MAX_PARAMETER_SWEEP_POINTS,
            "maximum_projections": MAX_PARAMETER_SWEEP_PROJECTIONS,
            "maximum_samples_per_point": MAX_PARAMETER_SWEEP_SAMPLES_PER_POINT,
            "maximum_total_samples": MAX_PARAMETER_SWEEP_TOTAL_SAMPLES,
            "maximum_scheduled_edges_per_point": (
                MAX_PARAMETER_SWEEP_SCHEDULED_EDGES_PER_POINT
            ),
            "maximum_total_scheduled_edges": MAX_PARAMETER_SWEEP_TOTAL_SCHEDULED_EDGES,
            "maximum_outputs": MAX_PARAMETER_SWEEP_OUTPUTS,
            "maximum_source_model_bytes": MAX_INPUT_BYTES,
            "maximum_components_per_point": MAX_PARAMETER_SWEEP_COMPONENTS,
            "maximum_states_per_point": MAX_PARAMETER_SWEEP_STATES,
            "maximum_controls_per_point": MAX_PARAMETER_SWEEP_CONTROLS,
            "maximum_sources_per_point": MAX_PARAMETER_SWEEP_SOURCES,
            "maximum_parameters": MAX_PARAMETER_SWEEP_PARAMETERS,
            "maximum_requested_max_step_intervals_per_point": (
                MAX_PARAMETER_SWEEP_MAX_STEP_INTERVALS_PER_POINT
            ),
            "maximum_total_requested_max_step_intervals": (
                MAX_PARAMETER_SWEEP_TOTAL_MAX_STEP_INTERVALS
            ),
            "maximum_sample_output_cells_per_point": (
                MAX_PARAMETER_SWEEP_SAMPLE_OUTPUT_CELLS_PER_POINT
            ),
            "maximum_total_sample_output_cells": (
                MAX_PARAMETER_SWEEP_TOTAL_SAMPLE_OUTPUT_CELLS
            ),
            "maximum_point_wall_time_s": MAX_PARAMETER_SWEEP_POINT_WALL_TIME_S,
            "maximum_total_wall_time_s": MAX_PARAMETER_SWEEP_TOTAL_WALL_TIME_S,
        },
    }
    return _NormalizedRequest(
        parameter=parameter,
        dimension=base_quantity.dimension,
        base_value_si=float(base_quantity.value),
        values_si=tuple(values),
        projections=tuple(projections),
        payload=normalized,
    )


def _require_parameter_affects_resolved_semantics(
    source_path: Path,
    source_model: Model,
    parameter: str,
) -> None:
    """Reject declared parameters that do not affect physics or the run request."""

    base_value = float(source_model.parameters[parameter].value)
    base_hashes = model_hashes(source_model)
    candidates = (math.nextafter(base_value, math.inf), math.nextafter(base_value, -math.inf))
    for candidate in candidates:
        if not math.isfinite(candidate) or candidate == base_value:
            continue
        try:
            probe = load_model(source_path, overrides={parameter: candidate})
        except Exception:
            continue
        hashes = model_hashes(probe)
        if (
            hashes["model_sha256"] != base_hashes["model_sha256"]
            or hashes["request_sha256"] != base_hashes["request_sha256"]
        ):
            return
    raise ParameterSweepError(
        f"Parameter {parameter!r} does not affect resolved physics or the simulation request"
    )


def _require_bounded_model_shape(model: Model, *, context: str) -> None:
    component_count = len(model.components)
    state_count = sum(
        component.type in {"capacitor", "inductor", "sat_inductor"}
        for component in model.components
    )
    source_count = sum(
        component.type in {"voltage_source", "current_source"}
        for component in model.components
    )
    counts = (
        ("components", component_count, MAX_PARAMETER_SWEEP_COMPONENTS),
        ("dynamic states", state_count, MAX_PARAMETER_SWEEP_STATES),
        ("controls", len(model.controls), MAX_PARAMETER_SWEEP_CONTROLS),
        ("independent sources", source_count, MAX_PARAMETER_SWEEP_SOURCES),
        ("parameters", len(model.parameters), MAX_PARAMETER_SWEEP_PARAMETERS),
        ("outputs", len(model.outputs), MAX_PARAMETER_SWEEP_OUTPUTS),
    )
    for name, actual, limit in counts:
        if actual > limit:
            raise ParameterSweepError(
                f"{context} has {actual} {name}; bounded sweep limit is {limit}"
            )


def _require_bounded_work(
    source_path: Path,
    normalized: _NormalizedRequest,
) -> None:
    total_samples = 0
    total_edges = 0
    total_max_step_intervals = 0
    total_sample_output_cells = 0
    for index, value_si in enumerate(normalized.values_si):
        try:
            model = load_model(source_path, overrides={normalized.parameter: value_si})
        except Exception:
            # Invalid point-local overrides are retained by the runner. A model that
            # cannot be resolved cannot allocate simulation work during preflight.
            continue
        _require_bounded_model_shape(model, context=f"point {index}")
        settings = model.transient
        span = settings.stop - settings.start
        grid_ratio = span / settings.output_step
        if (
            not math.isfinite(grid_ratio)
            or grid_ratio >= MAX_PARAMETER_SWEEP_SAMPLES_PER_POINT
        ):
            raise ParameterSweepError(
                f"point {index} requests more than the per-point limit of "
                f"{MAX_PARAMETER_SWEEP_SAMPLES_PER_POINT} waveform samples"
            )
        sample_count = int(math.floor(grid_ratio + 1e-12)) + 1
        last_sample = settings.start + (sample_count - 1) * settings.output_step
        if last_sample < settings.stop - max(1e-15, span * 1e-12):
            sample_count += 1
        scheduled_edges = 0
        for control in model.controls.values():
            if control.duty <= 0.0 or control.duty >= 1.0:
                continue
            cycles = span * control.frequency
            if (
                not math.isfinite(cycles)
                or cycles > MAX_PARAMETER_SWEEP_SCHEDULED_EDGES_PER_POINT / 2 - 2
            ):
                raise ParameterSweepError(
                    f"point {index} may schedule more than the per-point limit of "
                    f"{MAX_PARAMETER_SWEEP_SCHEDULED_EDGES_PER_POINT} PWM edges"
                )
            scheduled_edges += 2 * (int(math.ceil(cycles)) + 2)
        if sample_count > MAX_PARAMETER_SWEEP_SAMPLES_PER_POINT:
            raise ParameterSweepError(
                f"point {index} requests {sample_count} waveform samples; per-point limit is "
                f"{MAX_PARAMETER_SWEEP_SAMPLES_PER_POINT}"
            )
        if scheduled_edges > MAX_PARAMETER_SWEEP_SCHEDULED_EDGES_PER_POINT:
            raise ParameterSweepError(
                f"point {index} may schedule {scheduled_edges} PWM edges; per-point limit is "
                f"{MAX_PARAMETER_SWEEP_SCHEDULED_EDGES_PER_POINT}"
            )
        max_step_intervals = 0
        if settings.max_step is not None:
            max_step_ratio = span / settings.max_step
            if (
                not math.isfinite(max_step_ratio)
                or max_step_ratio > MAX_PARAMETER_SWEEP_MAX_STEP_INTERVALS_PER_POINT
            ):
                raise ParameterSweepError(
                    f"point {index} requests more than the per-point limit of "
                    f"{MAX_PARAMETER_SWEEP_MAX_STEP_INTERVALS_PER_POINT} max-step intervals"
                )
            max_step_intervals = int(math.ceil(max_step_ratio))
        sample_output_cells = sample_count * len(model.outputs)
        if sample_output_cells > MAX_PARAMETER_SWEEP_SAMPLE_OUTPUT_CELLS_PER_POINT:
            raise ParameterSweepError(
                f"point {index} requests {sample_output_cells} sample-output cells; "
                "per-point limit is "
                f"{MAX_PARAMETER_SWEEP_SAMPLE_OUTPUT_CELLS_PER_POINT}"
            )
        total_samples += sample_count
        total_edges += scheduled_edges
        total_max_step_intervals += max_step_intervals
        total_sample_output_cells += sample_output_cells
        if total_samples > MAX_PARAMETER_SWEEP_TOTAL_SAMPLES:
            raise ParameterSweepError(
                f"sweep requests at least {total_samples} waveform samples; total limit is "
                f"{MAX_PARAMETER_SWEEP_TOTAL_SAMPLES}"
            )
        if total_edges > MAX_PARAMETER_SWEEP_TOTAL_SCHEDULED_EDGES:
            raise ParameterSweepError(
                f"sweep may schedule at least {total_edges} PWM edges; total limit is "
                f"{MAX_PARAMETER_SWEEP_TOTAL_SCHEDULED_EDGES}"
            )
        if total_max_step_intervals > MAX_PARAMETER_SWEEP_TOTAL_MAX_STEP_INTERVALS:
            raise ParameterSweepError(
                "sweep requests at least "
                f"{total_max_step_intervals} max-step intervals; total limit is "
                f"{MAX_PARAMETER_SWEEP_TOTAL_MAX_STEP_INTERVALS}"
            )
        if total_sample_output_cells > MAX_PARAMETER_SWEEP_TOTAL_SAMPLE_OUTPUT_CELLS:
            raise ParameterSweepError(
                "sweep requests at least "
                f"{total_sample_output_cells} sample-output cells; total limit is "
                f"{MAX_PARAMETER_SWEEP_TOTAL_SAMPLE_OUTPUT_CELLS}"
            )


def _request_semantic_sha256(normalized_payload: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            key: value
            for key, value in normalized_payload.items()
            if key != "source"
        }
    )


def _prepare_destination(output_dir: str | Path) -> Path:
    destination = Path(output_dir).resolve()
    if destination.exists():
        if not destination.is_dir():
            raise ParameterSweepError(f"Sweep output is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ParameterSweepError(
                f"Sweep output directory is not empty: {destination}. "
                "Choose a new directory so prior sweep evidence is never overwritten."
            )
    else:
        destination.mkdir(parents=True)
    return destination


def _write_json(path: Path, payload: Any) -> None:
    try:
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ParameterSweepError(
            f"Cannot serialize finite JSON artifact {path.name}: {exc}"
        ) from exc
    _write_bytes(path, encoded)


def _write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _point_id(parameter: str, dimension: str, value_si: float, index: int) -> str:
    digest = sha256_json({"parameter": parameter, "dimension": dimension, "value_si": value_si})[
        :12
    ]
    return f"point-{index:03d}-{digest}"


def _safe_run_artifact(run_dir: Path, reference: Any, *, role: str) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ParameterSweepError(f"{role} path must be a non-empty string")
    pure = PurePosixPath(reference)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != reference:
        raise ParameterSweepError(f"{role} path must be a safe normalized relative path")
    path = run_dir.joinpath(*pure.parts)
    try:
        info = path.lstat()
    except OSError as exc:
        raise ParameterSweepError(f"Cannot inspect {role} artifact {reference!r}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ParameterSweepError(f"{role} artifact must be a regular non-symlink file")
    try:
        path.resolve().relative_to(run_dir.resolve())
    except ValueError as exc:
        raise ParameterSweepError(f"{role} artifact escapes the run directory") from exc
    return path


def _read_verified_metrics(
    run_dir: Path,
    *,
    expected_model: Model,
) -> tuple[dict[str, Any], str]:
    """Verify and bind a nested run to the exact parent-resolved sweep point."""

    run_manifest, _raw = _read_strict_json(run_dir / "run.json", role="run manifest")
    if run_manifest.get("schema") != "pear-sim/run/v0.1":
        raise ParameterSweepError("run.json has an unsupported schema")
    if run_manifest.get("backend") != "reference-python":
        raise ParameterSweepError("run.json has an unsupported backend")
    expected_hashes = model_hashes(expected_model)
    expected_parameters = {
        name: {
            "dimension": quantity.dimension,
            "value_si": quantity.value,
        }
        for name, quantity in sorted(expected_model.parameters.items())
    }
    expected_fields: dict[str, Any] = {
        "model": expected_model.name,
        "model_file": expected_model.source_path.name,
        "model_schema": expected_model.schema,
        "source_sha256": expected_hashes["source_sha256"],
        "model_sha256": expected_hashes["model_sha256"],
        "request_sha256": expected_hashes["request_sha256"],
        "pear_sim_version": __version__,
        "outputs": list(expected_model.outputs),
        "parameters": expected_parameters,
        "solver": expected_model.transient.solver,
        "rtol": expected_model.transient.rtol,
        "atol": expected_model.transient.atol,
    }
    for field, expected in expected_fields.items():
        if run_manifest.get(field) != expected:
            raise ParameterSweepError(
                f"run.json field {field!r} is not bound to the expected sweep point"
            )
    files = run_manifest.get("files")
    if not isinstance(files, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in files.items()
    ):
        raise ParameterSweepError("run.json files must be a string-to-string object")
    if "metrics" not in files:
        raise ParameterSweepError("run.json does not declare a metrics artifact")
    expected_paths = {"run.json", *files.values()}
    if len(expected_paths) != len(files) + 1:
        raise ParameterSweepError("run.json declares duplicate artifact paths")

    manifest, _raw = _read_strict_json(
        run_dir / "artifact-manifest.json",
        role="run artifact manifest",
    )
    if manifest.get("schema") != "pear-sim/artifact-manifest/v0.1":
        raise ParameterSweepError("run artifact manifest has an unsupported schema")
    run_sha256 = run_manifest.get("run_sha256")
    if not isinstance(run_sha256, str) or len(run_sha256) != 64:
        raise ParameterSweepError("run.json does not contain a valid run_sha256")
    subject = manifest.get("subject")
    if subject != {"kind": "run", "sha256": run_sha256}:
        raise ParameterSweepError("run artifact manifest subject does not match run_sha256")
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise ParameterSweepError("run artifact manifest artifacts must be an array")
    recorded_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"}:
            raise ParameterSweepError(f"run artifact record {index} has invalid fields")
        path = _safe_run_artifact(
            run_dir,
            record["path"],
            role=f"run artifact record {index}",
        )
        relative = str(record["path"])
        if relative in recorded_paths:
            raise ParameterSweepError(f"run artifact manifest repeats path {relative!r}")
        recorded_paths.add(relative)
        if isinstance(record["bytes"], bool) or not isinstance(record["bytes"], int):
            raise ParameterSweepError(f"run artifact record {relative!r} has invalid byte count")
        if record["bytes"] != path.stat().st_size:
            raise ParameterSweepError(f"run artifact byte count mismatch for {relative!r}")
        if record["sha256"] != sha256_file(path):
            raise ParameterSweepError(f"run artifact SHA-256 mismatch for {relative!r}")
    if recorded_paths != expected_paths:
        raise ParameterSweepError("run artifact manifest coverage does not match run.json")

    expected_run_sha256 = sha256_json(
        {
            "model_sha256": expected_hashes["model_sha256"],
            "request_sha256": expected_hashes["request_sha256"],
            "backend": "reference-python",
            "pear_sim_version": __version__,
        }
    )
    if run_sha256 != expected_run_sha256:
        raise ParameterSweepError("run_sha256 does not match the expected sweep-point semantics")
    metrics_path = _safe_run_artifact(run_dir, files["metrics"], role="run metrics")
    metrics, _raw = _read_strict_json(
        metrics_path,
        role="run metrics",
        maximum_bytes=MAX_METRICS_BYTES,
    )
    return metrics, run_sha256


def _record_projections(
    projections: tuple[dict[str, str], ...],
    metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for projection in projections:
        selector = {key: projection[key] for key in ("signal", "metric", "unit")}
        signal_record = metrics.get(projection["signal"])
        reason: str | None = None
        value: float | None = None
        if not isinstance(signal_record, Mapping):
            reason = f"Exact signal {projection['signal']!r} is absent from metrics.json."
        elif signal_record.get("unit") != projection["unit"]:
            reason = (
                f"Recorded unit {signal_record.get('unit')!r} does not exactly match "
                f"{projection['unit']!r}."
            )
        else:
            raw_value = signal_record.get(projection["metric"])
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                reason = f"Exact metric {projection['metric']!r} is absent or non-numeric."
            else:
                try:
                    candidate = float(raw_value)
                except (OverflowError, ValueError):
                    candidate = math.nan
                if not math.isfinite(candidate):
                    reason = f"Exact metric {projection['metric']!r} is non-finite."
                else:
                    value = candidate
        records.append(
            {
                "id": projection["id"],
                "selector": selector,
                "status": "recorded" if reason is None else "not_run",
                "value": value,
                "reason": reason,
            }
        )
    return records


def _error_record(exc: Exception) -> dict[str, str]:
    message = str(exc)
    if len(message) > MAX_TEXT_LENGTH:
        message = message[: MAX_TEXT_LENGTH - 1] + "…"
    return {"type": type(exc).__name__, "message": message}


def _run_point_worker(
    model_path: Path,
    overrides: Mapping[str, float],
    point_dir: Path,
    *,
    timeout_s: float,
    lean: bool = False,
) -> Path:
    """Run one simulation in a killable child process and atomically retain success.

    ``overrides`` maps parameter names to SI values; one entry is a 1-D sweep
    point, several entries are one tolerance corner.
    """

    worker_dir = point_dir / ".run-worker"
    run_dir = point_dir / "run"
    command = [
        sys.executable,
        "-I",
        "-m",
        "pear_sim.evidence.parameter_sweep_worker",
        str(model_path),
    ]
    for parameter, value_si in overrides.items():
        command.extend(
            ["--parameter", parameter, "--value-si", format(value_si, ".17g")]
        )
    command.extend(["--out", str(worker_dir)])
    if lean:
        command.append("--lean")
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            cwd=point_dir,
            env=environment,
            timeout=max(0.001, timeout_s),
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(worker_dir, ignore_errors=True)
        raise ParameterSweepError(
            f"Point simulation exceeded the {timeout_s:.3f}-second worker budget"
        ) from exc
    payload: Any = None
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        payload = None
    if completed.returncode != 0:
        if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
            error_type = str(payload["error"].get("type", "WorkerError"))
            message = str(payload["error"].get("message", "point worker failed"))
            detail = f"{error_type}: {message}"
        else:
            stderr = completed.stderr.decode("utf-8", errors="replace").strip()
            detail = stderr or f"worker exited with status {completed.returncode}"
        shutil.rmtree(worker_dir, ignore_errors=True)
        raise ParameterSweepError(f"Point worker failed: {detail}")
    worker_stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if worker_stderr:
        shutil.rmtree(worker_dir, ignore_errors=True)
        raise ParameterSweepError(
            "Point worker emitted unretained stderr and cannot be accepted: "
            + worker_stderr[:MAX_TEXT_LENGTH]
        )
    package_init = Path(pear_sim.__file__).resolve()
    worker_module = Path(__file__).with_name("parameter_sweep_worker.py").resolve()
    expected_identity = {
        "package_init": str(package_init),
        "package_init_sha256": sha256_file(package_init),
        "worker_module": str(worker_module),
        "worker_module_sha256": sha256_file(worker_module),
        "version": __version__,
    }
    if payload != {"status": "pass", "identity": expected_identity}:
        shutil.rmtree(worker_dir, ignore_errors=True)
        raise ParameterSweepError(
            "Point worker identity does not match the parent PEARSim installation"
        )
    if not worker_dir.is_dir():
        raise ParameterSweepError("Point worker reported success without a run bundle")
    worker_dir.replace(run_dir)
    return run_dir


def _accepted(status: str) -> bool | None:
    if status == "pass":
        return True
    if status == "fail":
        return False
    return None


def _aggregate_acceptance(statuses: list[str], *, expectations_present: bool) -> str:
    if not expectations_present:
        return "not_run"
    for status in ("fail", "not_comparable", "not_run", "not_applicable"):
        if status in statuses:
            return status
    return "pass" if statuses else "not_run"


def _overall_status(
    execution_status: str,
    acceptance_status: str,
    *,
    expectations_present: bool,
) -> str:
    if execution_status != "pass":
        return "fail"
    if not expectations_present:
        return "pass"
    return acceptance_status


def _artifact(path: Path, destination: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(destination).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _root_manifest(destination: Path, summary_path: Path) -> Path:
    manifest_path = destination / "artifact-manifest.json"
    records: list[dict[str, Any]] = []
    for path in sorted(destination.rglob("*")):
        if path == manifest_path:
            continue
        if path.is_symlink():
            raise ParameterSweepError(
                f"Parameter sweep bundle contains an unsupported symlink: {path}"
            )
        if path.is_dir():
            continue
        if not path.is_file():
            raise ParameterSweepError(
                f"Parameter sweep bundle contains a non-regular artifact: {path}"
            )
        records.append(_artifact(path, destination))
    records.sort(key=lambda item: str(item["path"]))
    _write_json(
        manifest_path,
        {
            "schema": "pear-sim/artifact-manifest/v0.1",
            "subject": {
                "kind": "parameter-sweep",
                "sha256": sha256_file(summary_path),
            },
            "artifacts": records,
        },
    )
    return manifest_path


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.9g}"
    return str(value)


def _status_class(value: Any) -> str:
    text = str(value).lower().replace("_", "-")
    if text in {"pass", "recorded"}:
        return "ok"
    if text == "fail":
        return "bad"
    return "muted"


def _render_report_html(report: Mapping[str, Any], schematic_svg: str) -> str:
    recorded = report["recorded"]
    summary = recorded["sweep_summary"]
    projections = recorded["projections"]
    failures = recorded["failures"]
    parameter = summary["parameter"]
    projection_ids = [item["id"] for item in projections["projections"]]
    header_cells = "".join(f"<th>{html.escape(item)}</th>" for item in projection_ids)
    point_rows: list[str] = []
    for point in summary["points"]:
        values = {item["id"]: item for item in point["projections"]}
        projection_cells = "".join(
            (
                f'<td class="number {_status_class(values[item]["status"])}">'
                f"{html.escape(_display(values[item]['value']))}</td>"
            )
            for item in projection_ids
        )
        point_rows.append(
            "<tr>"
            f'<td class="number">{point["index"]}</td>'
            f"<td><code>{html.escape(point['point_id'])}</code></td>"
            f'<td class="number">{html.escape(_display(point["parameter_value_si"]))}</td>'
            f'<td class="{_status_class(point["execution_status"])}">'
            f"{html.escape(point['execution_status'])}</td>"
            f'<td class="{_status_class(point["acceptance_status"])}">'
            f"{html.escape(point['acceptance_status'])}</td>"
            f"<td>{html.escape(_display(point['accepted']))}</td>"
            f"{projection_cells}</tr>"
        )
    failure_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(str(item['id']))}</code></td>"
        f"<td>{html.escape(str(item['kind']))}</td>"
        f"<td>{html.escape(str(item['stage']))}</td>"
        f"<td>{html.escape(str(item['message']))}</td>"
        "</tr>"
        for item in failures["failures"]
    )
    if not failure_rows:
        failure_rows = '<tr><td colspan="4" class="muted">No recorded failures.</td></tr>'
    source_json = html.escape(json.dumps(recorded, ensure_ascii=False, indent=2, sort_keys=True))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; script-src 'none'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <title>PEARSim parameter sweep — {html.escape(parameter["name"])}</title>
  <style>
    :root {{ font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #102a43; background: #f5f7fa; }}
    * {{ box-sizing: border-box; }} body {{ margin: 0; line-height: 1.45; }}
    main {{ max-width: 1240px; margin: auto; padding: 28px 20px 56px; }}
    header, section {{ background: white; border: 1px solid #d9e2ec; border-radius: 12px; padding: 20px; margin-bottom: 18px; }}
    h1, h2 {{ margin-top: 0; }} .claim {{ color: #7c4d00; background: #fff3c4; border-radius: 999px; padding: 5px 9px; font-weight: 700; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; margin-top: 18px; }}
    .card {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 10px; }}
    .card small {{ display: block; color: #627d98; }} .scroll {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .86rem; }}
    th, td {{ border-bottom: 1px solid #e6edf3; padding: 7px; text-align: left; vertical-align: top; }}
    th {{ background: #f5f7fa; }} td.number {{ text-align: right; font-variant-numeric: tabular-nums; }}
    .ok {{ color: #067647; font-weight: 700; }} .bad {{ color: #b42318; font-weight: 700; }} .muted {{ color: #667085; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }} code {{ overflow-wrap: anywhere; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #102a43; color: #f0f4f8; padding: 14px; border-radius: 8px; }}
    .schematic svg {{ width: 100%; height: auto; max-height: 620px; }}
  </style>
</head>
<body><main>
  <header>
    <h1>PEARSim one-dimensional parameter sweep</h1>
    <p><span class="claim">EXPLORATORY — NOT VALIDATION</span></p>
    <p>This offline report presents recorded producer artifacts only. It does not recompute metrics, acceptance, projections, or failures.</p>
    <div class="cards">
      <div class="card"><small>Parameter</small><strong>{html.escape(parameter["name"])}</strong></div>
      <div class="card"><small>Dimension</small><strong>{html.escape(parameter["dimension"])}</strong></div>
      <div class="card"><small>Points</small><strong>{len(summary["points"])}</strong></div>
      <div class="card"><small>Execution</small><strong class="{_status_class(summary["execution_status"])}">{html.escape(summary["execution_status"])}</strong></div>
      <div class="card"><small>Acceptance</small><strong class="{_status_class(summary["acceptance_status"])}">{html.escape(summary["acceptance_status"])}</strong></div>
      <div class="card"><small>Accepted</small><strong>{html.escape(_display(summary["accepted"]))}</strong></div>
    </div>
  </header>
  <section><h2>Source topology</h2><div class="schematic">{schematic_svg}</div><p class="muted">Nominal source model; the named parameter is overridden independently for each retained point.</p></section>
  <section><h2>Recorded points</h2><div class="scroll"><table><thead><tr><th>#</th><th>Point ID</th><th>Value [SI]</th><th>Execution</th><th>Acceptance</th><th>Accepted</th>{header_cells}</tr></thead><tbody>{"".join(point_rows)}</tbody></table></div></section>
  <section><h2>Recorded failures</h2><div class="scroll"><table><thead><tr><th>ID</th><th>Kind</th><th>Stage</th><th>Message</th></tr></thead><tbody>{failure_rows}</tbody></table></div></section>
  <section><h2>Authoritative recorded JSON</h2><details><summary>Show summary, projections, and failures</summary><pre>{source_json}</pre></details></section>
  <p class="muted">Generated by PEARSim {html.escape(__version__)} · no scripts or network resources</p>
</main></body></html>
"""


def _write_report(
    destination: Path,
    summary_path: Path,
    projections_path: Path,
    failures_path: Path,
    *,
    summary: dict[str, Any],
    projections: dict[str, Any],
    failures: dict[str, Any],
    schematic_svg: str,
) -> tuple[Path, Path, Path]:
    report_dir = destination / "report"
    report_dir.mkdir()
    schematic_path = report_dir / "schematic.svg"
    report_json_path = report_dir / "parameter-sweep-report.json"
    report_html_path = report_dir / "parameter-sweep-report.html"
    _write_bytes(schematic_path, schematic_svg.encode("utf-8"))
    report = {
        "schema": PARAMETER_SWEEP_REPORT_SCHEMA,
        "generator": {"name": "pear_sim", "version": __version__},
        "claim": {
            "level": "exploratory",
            "validation_claim": False,
            "hardware_sign_off_allowed": False,
            "statement": (
                "Presentation of recorded parameter-sweep artifacts; the report does "
                "not recompute metrics or acceptance."
            ),
        },
        "source": {
            "sweep_summary": _artifact(summary_path, destination),
            "projections": _artifact(projections_path, destination),
            "failures": _artifact(failures_path, destination),
        },
        "recorded": {
            "sweep_summary": summary,
            "projections": projections,
            "failures": failures,
        },
        "files": {
            "html": report_html_path.relative_to(destination).as_posix(),
            "schematic": schematic_path.relative_to(destination).as_posix(),
        },
    }
    _write_json(report_json_path, report)
    _write_bytes(report_html_path, _render_report_html(report, schematic_svg).encode("utf-8"))
    return report_json_path, report_html_path, schematic_path


def run_parameter_sweep(
    model_file: str | Path,
    request_file: str | Path,
    output_dir: str | Path,
    *,
    expectations_file: str | Path | None = None,
    jobs: int | None = None,
) -> ParameterSweepArtifacts:
    """Execute a strict, retained 1-D sweep over one declared model parameter.

    ``jobs`` point simulations run at the same time (default: all cores but
    one). Retained artifacts do not depend on it; only wall time does.

    Request/model/expectation semantics are validated before the destination is
    created.  Point-local model or solver errors are retained and do not stop later
    points.  Expectations are optional and are evaluated through the existing
    verified-run :func:`pear_sim.evidence.evidence_io.evaluate_run` workflow.
    """

    try:
        worker_jobs = resolve_jobs(jobs)
    except ValueError as exc:
        raise ParameterSweepError(str(exc)) from exc
    source_input = Path(model_file)
    source_bytes = _read_bounded_regular_bytes(
        source_input,
        role="source model",
        maximum_bytes=MAX_INPUT_BYTES,
    )
    source_path = source_input.resolve()
    request_path = Path(request_file)
    try:
        source_model = load_model(source_path)
    except Exception as exc:
        raise ParameterSweepError(f"Invalid source model: {exc}") from exc
    _require_bounded_model_shape(source_model, context="source model")
    if hashlib.sha256(source_bytes).hexdigest() != source_model.source_sha256:
        raise ParameterSweepError("Source model changed while it was being loaded")
    request_payload, request_bytes = _read_strict_json(request_path, role="sweep request")
    request_source_sha256 = hashlib.sha256(request_bytes).hexdigest()
    normalized = _normalize_request(
        request_payload,
        source_model=source_model,
        source_sha256=request_source_sha256,
    )
    _require_parameter_affects_resolved_semantics(
        source_path,
        source_model,
        normalized.parameter,
    )
    _require_bounded_work(source_path, normalized)

    expectations_payload: dict[str, Any] | None = None
    expectations_bytes: bytes | None = None
    expectations_path: Path | None = None
    normalized_expectations: dict[str, Any] | None = None
    if expectations_file is not None:
        expectations_path = Path(expectations_file)
        expectations_payload, expectations_bytes = _read_strict_json(
            expectations_path,
            role="expectations",
        )
        try:
            normalized_expectations = validate_expectations(expectations_payload)
        except ValueError as exc:
            raise ParameterSweepError(f"Invalid expectations: {exc}") from exc

    try:
        schematic_svg = sanitize_schematic_svg(render_schematic(source_model).svg)
    except Exception as exc:
        raise ParameterSweepError(f"Cannot render the source-model schematic: {exc}") from exc

    current_source_bytes = _read_bounded_regular_bytes(
        source_path,
        role="source model",
        maximum_bytes=MAX_INPUT_BYTES,
    )
    if current_source_bytes != source_bytes:
        raise ParameterSweepError("Source model changed while the sweep was being prepared")

    destination = _prepare_destination(output_dir)
    points_root = destination / "points"
    source_root = destination / "source-model"
    points_root.mkdir()
    source_root.mkdir()

    normalized_request_path = destination / "request.normalized.json"
    request_source_path = destination / "request.source.json"
    source_model_metadata_path = destination / "source-model.json"
    projections_path = destination / "projections.json"
    failures_path = destination / "failures.json"
    summary_path = destination / "sweep-summary.json"
    confidence_path = destination / "confidence.json"
    _write_bytes(request_source_path, request_bytes)
    _write_json(normalized_request_path, normalized.payload)

    source_suffix = ".json" if source_path.suffix.lower() == ".json" else ".yaml"
    retained_source_path = source_root / f"model{source_suffix}"
    _write_bytes(retained_source_path, source_bytes)
    source_model_metadata = {
        "schema": PARAMETER_SWEEP_SOURCE_MODEL_SCHEMA,
        "model": source_model.name,
        "source_artifact": _artifact(retained_source_path, destination),
        "hashes": model_hashes(source_model),
        "parameters": {
            name: {"value_si": quantity.value, "dimension": quantity.dimension}
            for name, quantity in sorted(source_model.parameters.items())
        },
        "swept_parameter": {
            "name": normalized.parameter,
            "dimension": normalized.dimension,
            "base_value_si": normalized.base_value_si,
        },
    }
    _write_json(source_model_metadata_path, source_model_metadata)

    retained_expectations_path: Path | None = None
    normalized_expectations_path: Path | None = None
    if expectations_bytes is not None and normalized_expectations is not None:
        retained_expectations_path = destination / "expectations.source.json"
        normalized_expectations_path = destination / "expectations.normalized.json"
        _write_bytes(retained_expectations_path, expectations_bytes)
        _write_json(normalized_expectations_path, normalized_expectations)

    points: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sweep_wall_time_start = perf_counter()

    def _launch(point_dir: Path, value_si: float) -> Path:
        # The remaining budget is read when the child actually starts, so the
        # total wall-time limit means the same with one job or many.
        remaining_sweep_time = (
            MAX_PARAMETER_SWEEP_TOTAL_WALL_TIME_S
            - (perf_counter() - sweep_wall_time_start)
        )
        if remaining_sweep_time <= 0.0:
            raise ParameterSweepError(
                "Sweep exhausted the total point-worker wall-time budget of "
                f"{MAX_PARAMETER_SWEEP_TOTAL_WALL_TIME_S:.0f} seconds"
            )
        return _run_point_worker(
            retained_source_path,
            {normalized.parameter: value_si},
            point_dir,
            timeout_s=min(MAX_PARAMETER_SWEEP_POINT_WALL_TIME_S, remaining_sweep_time),
        )

    # Every point model is loaded first, in process, from the retained
    # byte-for-byte source artifact (closing the time-of-check/time-of-use gap
    # if the caller edits the original model while a bounded sweep is running).
    # Only the child simulations then run concurrently; everything after them
    # is consumed in point order below, exactly as a serial sweep does.
    prepared: list[tuple[str, Path, Model | None, Exception | None]] = []
    for index, value_si in enumerate(normalized.values_si):
        point_id = _point_id(normalized.parameter, normalized.dimension, value_si, index)
        point_dir = points_root / point_id
        point_dir.mkdir()
        try:
            point_model = load_model(
                retained_source_path,
                overrides={normalized.parameter: value_si},
            )
            _require_bounded_model_shape(point_model, context=f"point {index}")
        except Exception as exc:  # retained below at the model_override stage
            prepared.append((point_id, point_dir, None, exc))
        else:
            prepared.append((point_id, point_dir, point_model, None))
    simulations = run_ordered(
        [
            partial(_launch, point_dir, value_si)
            for (_, point_dir, point_model, _), value_si in zip(
                prepared, normalized.values_si, strict=True
            )
            if point_model is not None
        ],
        jobs=worker_jobs,
    )

    for index, value_si in enumerate(normalized.values_si):
        point_id, point_dir, point_model, load_error = prepared[index]
        run_dir: Path | None = None
        run_sha256: str | None = None
        projection_records: list[dict[str, Any]] = [
            {
                "id": item["id"],
                "selector": {key: item[key] for key in ("signal", "metric", "unit")},
                "status": "not_run",
                "value": None,
                "reason": "Point execution did not produce verified metrics.",
            }
            for item in normalized.projections
        ]
        execution_error: dict[str, str] | None = None
        execution_stage = "model_override"
        try:
            if load_error is not None:
                raise load_error
            execution_stage = "simulation"
            run_dir = next(simulations).unwrap()
            execution_stage = "run_bundle"
            metrics, run_sha256 = _read_verified_metrics(
                run_dir,
                expected_model=point_model,
            )
            execution_stage = "projection"
            projection_records = _record_projections(
                normalized.projections,
                metrics,
            )
            missing_projection_ids = [
                projection["id"]
                for projection in projection_records
                if projection["status"] != "recorded"
            ]
            if missing_projection_ids:
                raise ParameterSweepError(
                    "Requested projection(s) were not recorded: "
                    + ", ".join(missing_projection_ids)
                )
            execution_status = "pass"
        except Exception as exc:  # retain this point and continue the bounded request
            execution_status = "fail"
            execution_error = _error_record(exc)
            failure = {
                "id": f"{point_id}:execution",
                "kind": "execution",
                "status": "fail",
                "point_id": point_id,
                "stage": execution_stage,
                "message": f"{execution_error['type']}: {execution_error['message']}",
                "error": execution_error,
            }
            failures.append(failure)

        for projection in projection_records:
            if execution_status == "pass" and projection["status"] != "recorded":
                failures.append(
                    {
                        "id": f"{point_id}:projection:{projection['id']}",
                        "kind": "projection",
                        "status": "not_run",
                        "point_id": point_id,
                        "stage": "projection",
                        "message": projection["reason"],
                        "error": None,
                    }
                )

        acceptance_status = "not_run"
        acceptance_reason = (
            "No expectations document was supplied."
            if expectations_file is None
            else "Point execution did not produce a verified run bundle."
        )
        evidence_path: Path | None = None
        if execution_status == "pass" and retained_expectations_path is not None:
            try:
                evidence_artifacts = evaluate_run(
                    run_dir,
                    retained_expectations_path,
                    point_dir / "evidence",
                )
                acceptance_status = evidence_artifacts.blocking_status.replace("-", "_")
                evidence_path = evidence_artifacts.evidence_json
                acceptance_reason = None
            except Exception as exc:
                error = _error_record(exc)
                acceptance_reason = f"{error['type']}: {error['message']}"
                failures.append(
                    {
                        "id": f"{point_id}:acceptance",
                        "kind": "acceptance",
                        "status": "not_run",
                        "point_id": point_id,
                        "stage": "acceptance",
                        "message": acceptance_reason,
                        "error": error,
                    }
                )
        point_accepted = _accepted(acceptance_status)
        point_overall = _overall_status(
            execution_status,
            acceptance_status,
            expectations_present=expectations_file is not None,
        )
        point_record = {
            "schema": PARAMETER_SWEEP_POINT_SCHEMA,
            "index": index,
            "point_id": point_id,
            "parameter": {
                "name": normalized.parameter,
                "dimension": normalized.dimension,
                "value_si": value_si,
            },
            "execution": {
                "status": execution_status,
                "error": execution_error,
                "run": (
                    None
                    if run_dir is None or run_sha256 is None
                    else {
                        "path": (run_dir / "run.json").relative_to(destination).as_posix(),
                        "run_sha256": run_sha256,
                    }
                ),
            },
            "acceptance": {
                "status": acceptance_status,
                "accepted": point_accepted,
                "reason": acceptance_reason,
                "evidence": (
                    None
                    if evidence_path is None
                    else {"path": evidence_path.relative_to(destination).as_posix()}
                ),
            },
            "overall_status": point_overall,
            "projections": projection_records,
        }
        point_path = point_dir / "point.json"
        _write_json(point_path, point_record)
        points.append(
            {
                "index": index,
                "point_id": point_id,
                "parameter_value_si": value_si,
                "execution_status": execution_status,
                "acceptance_status": acceptance_status,
                "accepted": point_accepted,
                "overall_status": point_overall,
                "projections": projection_records,
                "artifacts": {
                    "point": _artifact(point_path, destination),
                    "run": point_record["execution"]["run"],
                    "evidence": point_record["acceptance"]["evidence"],
                },
            }
        )

    projection_documents: list[dict[str, Any]] = []
    for projection in normalized.projections:
        projection_documents.append(
            {
                "id": projection["id"],
                "selector": {key: projection[key] for key in ("signal", "metric", "unit")},
                "points": [
                    {
                        "index": point["index"],
                        "point_id": point["point_id"],
                        "parameter_value_si": point["parameter_value_si"],
                        **next(
                            {
                                "status": item["status"],
                                "value": item["value"],
                                "reason": item["reason"],
                            }
                            for item in point["projections"]
                            if item["id"] == projection["id"]
                        ),
                    }
                    for point in points
                ],
            }
        )
    projection_counts = Counter(
        record["status"] for projection in projection_documents for record in projection["points"]
    )
    projections_document = {
        "schema": PARAMETER_SWEEP_PROJECTIONS_SCHEMA,
        "parameter": {
            "name": normalized.parameter,
            "dimension": normalized.dimension,
        },
        "summary": {
            "selector_count": len(projection_documents),
            "record_count": sum(projection_counts.values()),
            "status_counts": dict(sorted(projection_counts.items())),
        },
        "projections": projection_documents,
    }
    _write_json(projections_path, projections_document)

    failure_counts = Counter(item["kind"] for item in failures)
    failures_document = {
        "schema": PARAMETER_SWEEP_FAILURES_SCHEMA,
        "summary": {
            "count": len(failures),
            "kind_counts": dict(sorted(failure_counts.items())),
        },
        "failures": failures,
    }
    _write_json(failures_path, failures_document)

    execution_counts = Counter(point["execution_status"] for point in points)
    acceptance_counts = Counter(point["acceptance_status"] for point in points)
    point_overall_counts = Counter(point["overall_status"] for point in points)
    execution_status = "fail" if execution_counts.get("fail", 0) else "pass"
    acceptance_status = _aggregate_acceptance(
        [point["acceptance_status"] for point in points],
        expectations_present=expectations_file is not None,
    )
    accepted = _accepted(acceptance_status)
    overall_status = _overall_status(
        execution_status,
        acceptance_status,
        expectations_present=expectations_file is not None,
    )
    expectations_record: dict[str, Any] | None = None
    if (
        retained_expectations_path is not None
        and normalized_expectations_path is not None
        and normalized_expectations is not None
    ):
        expectations_record = {
            "source": _artifact(retained_expectations_path, destination),
            "normalized": _artifact(normalized_expectations_path, destination),
            "semantic_sha256": sha256_json(normalized_expectations),
        }
    summary = {
        "schema": PARAMETER_SWEEP_SUMMARY_SCHEMA,
        "kind": "one_dimensional_parameter_sweep",
        "overall_status": overall_status,
        "execution_status": execution_status,
        "acceptance_status": acceptance_status,
        "accepted": accepted,
        "parameter": {
            "name": normalized.parameter,
            "dimension": normalized.dimension,
            "base_value_si": normalized.base_value_si,
        },
        "summary": {
            "point_count": len(points),
            "execution_status_counts": dict(sorted(execution_counts.items())),
            "acceptance_status_counts": dict(sorted(acceptance_counts.items())),
            "overall_status_counts": dict(sorted(point_overall_counts.items())),
            "projection_selector_count": len(normalized.projections),
            "projection_record_status_counts": dict(sorted(projection_counts.items())),
            "failure_count": len(failures),
        },
        "points": points,
        "expectations": expectations_record,
        "claim": {
            "classification": "agent_experiment",
            "validation_claim": False,
            "compatibility_claim": False,
            "hardware_sign_off_allowed": False,
            "statement": (
                "A completed generic parameter sweep is an exploratory execution and, "
                "when supplied, user-acceptance observation; it does not validate physics."
            ),
        },
        "artifacts": {
            "source_request": _artifact(request_source_path, destination),
            "normalized_request": _artifact(normalized_request_path, destination),
            "source_model": _artifact(source_model_metadata_path, destination),
            "projections": _artifact(projections_path, destination),
            "failures": _artifact(failures_path, destination),
            "confidence": {"path": confidence_path.name},
            "human_report": {"path": "report/parameter-sweep-report.html"},
            "artifact_manifest": {"path": "artifact-manifest.json"},
        },
        "provenance": {
            "pear_sim_version": __version__,
            "source_model_hashes": model_hashes(source_model),
            "request_source_sha256": request_source_sha256,
            "request_semantic_sha256": _request_semantic_sha256(normalized.payload),
            "network_access": False,
            "point_order": "request_input_order",
        },
    }
    _write_json(summary_path, summary)

    confidence = {
        "schema": PARAMETER_SWEEP_CONFIDENCE_SCHEMA,
        "kind": "agent_confidence_card",
        "level": "EXPLORATORY",
        "acceptance": {
            "execution_status": execution_status,
            "acceptance_status": acceptance_status,
            "accepted": accepted,
            "overall_status": overall_status,
            "sweep_summary_sha256": sha256_file(summary_path),
        },
        "envelope": {
            "status": "requested_points_only",
            "parameter": {
                "name": normalized.parameter,
                "dimension": normalized.dimension,
                "values_si": list(normalized.values_si),
            },
            "point_count": len(points),
            "expectations_supplied": expectations_file is not None,
            "cross_platform_evidence": "not_run",
            "convergence_study": "not_run",
        },
        "claim": {
            "validation_claim": False,
            "compatibility_claim": False,
            "hardware_sign_off_allowed": False,
            "general_parameter_envelope_claimed": False,
        },
    }
    _write_json(confidence_path, confidence)
    report_json_path, report_html_path, _schematic_path = _write_report(
        destination,
        summary_path,
        projections_path,
        failures_path,
        summary=summary,
        projections=projections_document,
        failures=failures_document,
        schematic_svg=schematic_svg,
    )
    manifest_path = _root_manifest(destination, summary_path)
    return ParameterSweepArtifacts(
        output_dir=destination,
        source_request=request_source_path,
        normalized_request=normalized_request_path,
        source_model=source_model_metadata_path,
        sweep_summary=summary_path,
        projections=projections_path,
        failures=failures_path,
        confidence_json=confidence_path,
        report_json=report_json_path,
        report_html=report_html_path,
        artifact_manifest=manifest_path,
        execution_status=execution_status,
        acceptance_status=acceptance_status,
        accepted=accepted,
        overall_status=overall_status,
    )


__all__ = [
    "MAX_PARAMETER_SWEEP_POINTS",
    "MAX_PARAMETER_SWEEP_PROJECTIONS",
    "PARAMETER_SWEEP_CONFIDENCE_SCHEMA",
    "PARAMETER_SWEEP_FAILURES_SCHEMA",
    "PARAMETER_SWEEP_NORMALIZED_REQUEST_SCHEMA",
    "PARAMETER_SWEEP_POINT_SCHEMA",
    "PARAMETER_SWEEP_PROJECTIONS_SCHEMA",
    "PARAMETER_SWEEP_REPORT_SCHEMA",
    "PARAMETER_SWEEP_REQUEST_SCHEMA",
    "PARAMETER_SWEEP_SOURCE_MODEL_SCHEMA",
    "PARAMETER_SWEEP_SUMMARY_SCHEMA",
    "ParameterSweepArtifacts",
    "ParameterSweepError",
    "run_parameter_sweep",
]
