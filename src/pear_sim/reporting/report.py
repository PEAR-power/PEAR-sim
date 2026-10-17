from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import html
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from pear_sim import __version__
from pear_sim._jsonutil import (
    reject_json_constant as _reject_json_constant,
    unique_json_object as _unique_json_object,
)
from pear_sim.spec.signals import SIGNAL_UNITS
from pear_sim.limits import (
    MAX_HARMONIC_ORDER,
    MAX_MODEL_GATE_OUTPUTS,
    MAX_MODEL_LOGICAL_CONTROLS,
    MAX_RUN_EVENT_ROWS,
)
from pear_sim.reporting.report_svg import (
    loss_bars_svg,
    tj_envelope_svg,
    SvgError,
    harmonic_spectrum_svg,
    phase_series_svg,
    sanitize_schematic_svg,
    waveform_svg,
)
from pear_sim.analysis.three_phase_modulation import MAX_MODULATION_CYCLE_ROWS


REPORT_SCHEMA = "pear-sim/report/v0.1"
_MAX_WAVEFORM_ROWS = 1_000_000
MAX_EVENT_ROWS = MAX_RUN_EVENT_ROWS
MAX_EVENT_LINE_BYTES = 64 * 1024
MAX_EVENT_FILE_BYTES = 64 * 1024 * 1024
MAX_EVENT_HTML_ROWS = 2_000
MAX_MODE_COVERAGE_FILE_BYTES = 64 * 1024 * 1024
MAX_MODE_COVERAGE_EVENT_ROWS = 100_000
MAX_MODE_COVERAGE_INTERVAL_ROWS = 100_000
MAX_MODE_COVERAGE_HTML_EVENTS = 2_000
MAX_MODE_COVERAGE_HTML_INTERVALS = 2_000
MAX_GATE_DRIVE_FILE_BYTES = 8 * 1024 * 1024
MAX_GATE_DRIVE_CONTROLS = MAX_MODEL_LOGICAL_CONTROLS
MAX_GATE_DRIVE_OUTPUTS = MAX_MODEL_GATE_OUTPUTS
MAX_GATE_DRIVE_TIMELINE_ROWS = 4_096
MAX_THREE_PHASE_MODULATION_FILE_BYTES = 8 * 1024 * 1024
MAX_THREE_PHASE_MODULATION_BLOCKS = 32
MAX_THREE_PHASE_MODULATION_PROJECTION_ROWS = 4_096
MAX_MODULATION_CYCLE_FILE_BYTES = 64 * 1024 * 1024
MAX_MODULATION_CYCLE_LINE_BYTES = 64 * 1024
MAX_HARMONICS_FILE_BYTES = 8 * 1024 * 1024
MAX_HARMONIC_SIGNALS = 32
MAX_HARMONICS_PER_SIGNAL = MAX_HARMONIC_ORDER + 1
MAX_HARMONIC_HTML_ROWS = 512
MAX_MODULATION_HTML_ROWS = 96
_REPORT_OUTPUT_FILES = {
    "artifact-manifest.json",
    "report.html",
    "report.json",
    "schematic.svg",
}


class ReportError(ValueError):
    """Raised when a run directory cannot produce a trustworthy report."""

    code = "report.invalid"


@dataclass(frozen=True)
class ReportArtifacts:
    report_json: Path
    report_html: Path
    artifact_manifest: Path
    schematic_svg: Path | None


@dataclass(frozen=True)
class _Waveforms:
    time: list[float]
    signals: dict[str, list[float]]


def generate_report(
    run_dir: str | Path,
    output_dir: str | Path | None = None,
    schematic_svg: str | Path | None = None,
) -> ReportArtifacts:
    """Generate deterministic ``report.json`` and self-contained ``report.html``.

    ``output_dir`` defaults to a sibling named ``<run-name>-report``. Report files are
    never written into the immutable run tree. ``schematic_svg`` may be an SVG source
    string or a :class:`~pathlib.Path` to an SVG file. The schematic is reduced to a
    conservative, non-active inline subset. Existing run artifacts are read but never
    modified.
    """

    directory = Path(run_dir).resolve()
    if not directory.is_dir():
        raise ReportError(f"Run directory does not exist: {directory}")
    destination = (
        (directory.parent / f"{directory.name}-report").resolve()
        if output_dir is None
        else Path(output_dir).resolve()
    )
    try:
        destination.relative_to(directory)
    except ValueError:
        pass
    else:
        raise ReportError("Report output directory must be outside the immutable run tree")
    if destination.exists():
        if not destination.is_dir():
            raise ReportError(f"Report output is not a directory: {destination}")
        unexpected = sorted(
            path.name
            for path in destination.iterdir()
            if path.name not in _REPORT_OUTPUT_FILES or path.is_symlink() or not path.is_file()
        )
        if unexpected:
            raise ReportError(
                "Report output contains files not owned by PEARSim: " + ", ".join(unexpected)
            )
    else:
        destination.mkdir(parents=True)

    run_path = directory / "run.json"
    run = _read_json_object(run_path, "run manifest")
    files = run.get("files", {})
    if not isinstance(files, dict):
        raise ReportError("run.json field 'files' must be an object")

    input_paths: dict[str, Path] = {"run": run_path}
    for role, reference in sorted(files.items()):
        input_paths[str(role)] = _artifact_path(directory, reference, str(role))

    artifact_manifest_path = directory / "artifact-manifest.json"
    if artifact_manifest_path.is_file():
        artifact_manifest = _read_and_verify_artifact_manifest(
            artifact_manifest_path,
            directory,
            run,
        )
        input_paths["artifact_manifest"] = artifact_manifest_path
        artifact_manifest_record = {
            "present": True,
            "verified": True,
            "schema": artifact_manifest.get("schema"),
            "subject": artifact_manifest.get("subject"),
        }
    else:
        artifact_manifest = None
        artifact_manifest_record = {"present": False, "verified": False}

    metrics_path = input_paths.get("metrics")
    if metrics_path is None:
        metrics_path = _artifact_path(directory, "metrics.json", "metrics")
        input_paths["metrics"] = metrics_path
    waveforms_path = input_paths.get("waveforms")
    if waveforms_path is None:
        waveforms_path = _artifact_path(directory, "waveforms.csv", "waveforms")
        input_paths["waveforms"] = waveforms_path
    metrics = _read_json_object(metrics_path, "metrics")
    waveforms = _read_waveforms(waveforms_path)
    warnings = _read_warnings(run, input_paths.get("warnings"))
    events_path = input_paths.get("events")
    if events_path is None and (directory / "events.jsonl").is_file():
        events_path = _artifact_path(directory, "events.jsonl", "events")
        input_paths["events"] = events_path
    events = _read_events(events_path)
    source_summary = (
        _read_json_object(input_paths["summary"], "run summary")
        if "summary" in input_paths
        else None
    )
    resolved_model = (
        _read_json_object(input_paths["resolved_model"], "resolved model")
        if "resolved_model" in input_paths
        else None
    )
    analytic_checks = (
        _read_json_object(input_paths["analytic_checks"], "analytic checks")
        if "analytic_checks" in input_paths
        else None
    )
    physics_audit = (
        _read_json_object(input_paths["physics_audit"], "physics audit")
        if "physics_audit" in input_paths
        else None
    )
    losses_thermal = (
        _read_json_object(input_paths["losses_thermal"], "losses and thermal estimate")
        if "losses_thermal" in input_paths
        else None
    )
    mode_coverage = (
        _read_mode_coverage(input_paths["mode_coverage"])
        if "mode_coverage" in input_paths
        else None
    )
    gate_drive = (
        _read_gate_drive(input_paths["gate_drive"]) if "gate_drive" in input_paths else None
    )
    three_phase_modulation = (
        _read_three_phase_modulation(input_paths["three_phase_modulation"])
        if "three_phase_modulation" in input_paths
        else None
    )
    modulation_cycle_source = (
        _inspect_modulation_cycles(input_paths["modulation_cycles"])
        if "modulation_cycles" in input_paths
        else None
    )
    npc_modulation = (
        _read_npc_modulation(input_paths["npc_modulation"])
        if "npc_modulation" in input_paths
        else None
    )
    if three_phase_modulation is not None and modulation_cycle_source is None:
        raise ReportError(
            "three-phase-modulation.json requires authoritative modulation-cycles.jsonl"
        )
    if three_phase_modulation is not None and gate_drive is not None:
        raise ReportError(
            "A three-phase modulation run must not publish fixed-duty gate-drive v0.1"
        )
    if npc_modulation is not None:
        if gate_drive is not None:
            raise ReportError(
                "An NPC modulation run must not publish fixed-duty gate-drive v0.1"
            )
        if three_phase_modulation is not None:
            raise ReportError(
                "A run must not publish both SPWM and NPC modulation observations"
            )
        if modulation_cycle_source is None:
            raise ReportError(
                "npc-modulation.json requires authoritative modulation-cycles.jsonl"
            )
        if modulation_cycle_source["cycle_schema"] != (
            "pear-sim/modulation-cycle/three-phase-npc-spwm/v0.1"
        ):
            raise ReportError(
                "npc-modulation.json requires NPC modulation-cycle rows"
            )
        _require_run_binding(npc_modulation, run, label="npc-modulation.json")
        if npc_modulation["cycles_recorded"] != modulation_cycle_source["row_count"]:
            raise ReportError(
                "NPC modulation cycle count disagrees with modulation-cycles.jsonl"
            )
    if (
        three_phase_modulation is not None
        and modulation_cycle_source is not None
        and modulation_cycle_source["cycle_schema"]
        != "pear-sim/modulation-cycle/three-phase-spwm/v0.1"
    ):
        raise ReportError(
            "three-phase-modulation.json requires SPWM modulation-cycle rows"
        )
    if three_phase_modulation is not None:
        _require_run_binding(
            three_phase_modulation,
            run,
            label="three-phase-modulation.json",
        )
        recorded_count = three_phase_modulation.get("summary", {}).get("source_cycle_count")
        if recorded_count != modulation_cycle_source["row_count"]:
            raise ReportError(
                "three-phase modulation source-cycle count disagrees with modulation-cycles.jsonl"
            )
        for block in three_phase_modulation["blocks"]:
            block_id = block.get("block_id")
            source_count = block.get("summary", {}).get("source_cycle_count")
            if source_count != modulation_cycle_source["rows_by_block"].get(block_id):
                raise ReportError(
                    "three-phase modulation block count disagrees with modulation-cycles.jsonl"
                )
    harmonics = _read_harmonics(input_paths["harmonics"]) if "harmonics" in input_paths else None
    if harmonics is not None:
        _require_run_binding(harmonics, run, label="harmonics.json")

    sanitized_schematic, schematic_record = _load_schematic(schematic_svg)
    provenance = {
        "run_schema": run.get("schema"),
        "source_pear_sim_version": run.get("pear_sim_version"),
        "model_sha256": run.get("model_sha256"),
        "declared_hashes": {
            name: run.get(name)
            for name in ("source_sha256", "model_sha256", "request_sha256", "run_sha256")
            if run.get(name) is not None
        },
        "source_artifact_manifest": artifact_manifest_record,
        "input_artifacts": {
            role: {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for role, path in sorted(input_paths.items())
        },
        "schematic": schematic_record,
    }
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "generator": {"name": "pear_sim", "version": __version__},
        "claim": {
            "level": "alpha-observation-only",
            "statement": (
                "This report presents run artifacts; it does not by itself establish "
                "verification, compatibility, or physical validation."
            ),
        },
        "provenance": provenance,
        "run": run,
        "source_summary": source_summary,
        "resolved_model": resolved_model,
        "analytic_checks": analytic_checks,
        "physics_audit": physics_audit,
        "losses_thermal": losses_thermal,
        "mode_coverage": mode_coverage,
        "gate_drive": gate_drive,
        "three_phase_modulation": three_phase_modulation,
        "npc_modulation": npc_modulation,
        "modulation_cycle_source": modulation_cycle_source,
        "harmonics": harmonics,
        "warnings": warnings,
        "metrics": metrics,
        "events": events,
        "waveforms": {
            "sample_count": len(waveforms.time),
            "time_start_s": waveforms.time[0],
            "time_stop_s": waveforms.time[-1],
            "signals": list(waveforms.signals),
            "source": waveforms_path.name,
            "render_max_points_per_signal": 800,
        },
        "files": {
            "html": "report.html",
            "schematic": "schematic.svg" if sanitized_schematic is not None else None,
        },
    }

    report_json_text = (
        json.dumps(
            report,
            allow_nan=False,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )
    report_html_text = _render_html(report, waveforms, sanitized_schematic)
    report_json_path = destination / "report.json"
    report_html_path = destination / "report.html"
    schematic_path = destination / "schematic.svg"
    _write_text(report_json_path, report_json_text)
    _write_text(report_html_path, report_html_text)
    if sanitized_schematic is not None:
        _write_text(schematic_path, sanitized_schematic)
        schematic_result: Path | None = schematic_path
    else:
        schematic_path.unlink(missing_ok=True)
        schematic_result = None
    artifact_paths = [report_json_path, report_html_path]
    if schematic_result is not None:
        artifact_paths.append(schematic_result)
    artifact_records = [
        {
            "path": path.name,
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(artifact_paths)
    ]
    report_bundle_sha256 = hashlib.sha256(
        json.dumps(
            artifact_records,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    artifact_manifest_path = destination / "artifact-manifest.json"
    _write_text(
        artifact_manifest_path,
        json.dumps(
            {
                "schema": "pear-sim/artifact-manifest/v0.1",
                "subject": {
                    "kind": "report",
                    "sha256": report_bundle_sha256,
                    "run_sha256": run.get("run_sha256"),
                },
                "artifacts": artifact_records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    return ReportArtifacts(
        report_json=report_json_path,
        report_html=report_html_path,
        artifact_manifest=artifact_manifest_path,
        schematic_svg=schematic_result,
    )


def _artifact_path(directory: Path, reference: Any, role: str) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ReportError(f"run.json file reference for {role!r} must be a non-empty string")
    path = (directory / reference).resolve()
    try:
        path.relative_to(directory)
    except ValueError as exc:
        raise ReportError(f"Run artifact for {role!r} escapes the run directory") from exc
    if not path.is_file():
        raise ReportError(f"Required run artifact is missing for {role!r}: {path.name}")
    return path


def _require_run_binding(payload: dict[str, Any], run: dict[str, Any], *, label: str) -> None:
    binding = payload.get("source_binding")
    if not isinstance(binding, dict):
        raise ReportError(f"{label} source_binding must be an object")
    for field in ("model_sha256", "request_sha256", "run_sha256"):
        if binding.get(field) != run.get(field):
            raise ReportError(f"{label} source_binding.{field} disagrees with run.json")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ReportError(f"Required {label} is missing: {path.name}")
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReportError(f"Cannot read {label} {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReportError(f"{label.capitalize()} {path.name} must contain a JSON object")
    return value


def _read_mode_coverage(path: Path) -> dict[str, Any]:
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ReportError(f"Cannot inspect mode coverage {path.name}: {exc}") from exc
    if file_size > MAX_MODE_COVERAGE_FILE_BYTES:
        raise ReportError(
            f"{path.name} is {file_size} bytes; "
            f"mode-coverage limit is {MAX_MODE_COVERAGE_FILE_BYTES} bytes"
        )
    payload = _read_json_object(path, "mode coverage")
    event_timing = payload.get("event_timing")
    if isinstance(event_timing, dict):
        events = event_timing.get("events")
        if isinstance(events, list) and len(events) > MAX_MODE_COVERAGE_EVENT_ROWS:
            raise ReportError(
                f"{path.name} exceeds the {MAX_MODE_COVERAGE_EVENT_ROWS} "
                "mode-coverage event-row limit"
            )
    intervals = payload.get("intervals")
    if isinstance(intervals, list) and len(intervals) > MAX_MODE_COVERAGE_INTERVAL_ROWS:
        raise ReportError(
            f"{path.name} exceeds the {MAX_MODE_COVERAGE_INTERVAL_ROWS} "
            "mode-coverage interval-row limit"
        )
    return payload


def _read_gate_drive(path: Path) -> dict[str, Any]:
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ReportError(f"Cannot inspect gate drive {path.name}: {exc}") from exc
    if file_size > MAX_GATE_DRIVE_FILE_BYTES:
        raise ReportError(
            f"{path.name} is {file_size} bytes; gate-drive limit is "
            f"{MAX_GATE_DRIVE_FILE_BYTES} bytes"
        )
    payload = _read_json_object(path, "gate drive")
    if payload.get("schema") != "pear-sim/gate-drive-observation/v0.1":
        raise ReportError(
            f"{path.name} has unsupported gate-drive schema {payload.get('schema')!r}"
        )
    outputs = payload.get("gate_outputs")
    controls = payload.get("logical_controls")
    if not isinstance(outputs, list) or len(outputs) > MAX_GATE_DRIVE_OUTPUTS:
        raise ReportError(
            f"{path.name} gate_outputs must contain at most {MAX_GATE_DRIVE_OUTPUTS} rows"
        )
    if not isinstance(controls, list) or len(controls) > MAX_GATE_DRIVE_CONTROLS:
        raise ReportError(
            f"{path.name} logical_controls must contain at most {MAX_GATE_DRIVE_CONTROLS} rows"
        )
    timeline_rows = 0
    for index, control in enumerate(controls):
        if not isinstance(control, dict):
            raise ReportError(f"{path.name} logical control {index} must be an object")
        timeline = control.get("timeline")
        if not isinstance(timeline, dict):
            raise ReportError(f"{path.name} logical control {index} has no timeline object")
        for field in ("transactions", "segments"):
            rows = timeline.get(field)
            if not isinstance(rows, list):
                raise ReportError(
                    f"{path.name} logical control {index} timeline.{field} must be an array"
                )
            timeline_rows += len(rows)
    if timeline_rows > MAX_GATE_DRIVE_TIMELINE_ROWS:
        raise ReportError(
            f"{path.name} exceeds the {MAX_GATE_DRIVE_TIMELINE_ROWS} gate-drive timeline-row limit"
        )
    return payload


def _read_three_phase_modulation(path: Path) -> dict[str, Any]:
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ReportError(f"Cannot inspect three-phase modulation {path.name}: {exc}") from exc
    if file_size > MAX_THREE_PHASE_MODULATION_FILE_BYTES:
        raise ReportError(
            f"{path.name} is {file_size} bytes; three-phase-modulation limit is "
            f"{MAX_THREE_PHASE_MODULATION_FILE_BYTES} bytes"
        )
    payload = _read_json_object(path, "three-phase modulation")
    if payload.get("schema") != "pear-sim/three-phase-modulation-observation/v0.1":
        raise ReportError(
            f"{path.name} has unsupported three-phase-modulation schema {payload.get('schema')!r}"
        )
    summary = payload.get("summary")
    blocks = payload.get("blocks")
    if not isinstance(summary, dict):
        raise ReportError(f"{path.name} summary must be an object")
    if (
        not isinstance(blocks, list)
        or not blocks
        or len(blocks) > MAX_THREE_PHASE_MODULATION_BLOCKS
    ):
        raise ReportError(
            f"{path.name} blocks must contain at most {MAX_THREE_PHASE_MODULATION_BLOCKS} rows"
        )
    projection_rows = 0
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise ReportError(f"{path.name} block {index} must be an object")
        bridge = block.get("bridge")
        phases = bridge.get("phases") if isinstance(bridge, dict) else None
        if not isinstance(phases, list) or len(phases) != 3:
            raise ReportError(f"{path.name} block {index} must map exactly three phases")
        projection = block.get("fundamental_period_projection")
        cycles = projection.get("cycles") if isinstance(projection, dict) else None
        retention = projection.get("retention") if isinstance(projection, dict) else None
        if not isinstance(cycles, list) or not isinstance(retention, dict):
            raise ReportError(
                f"{path.name} block {index} must contain cycles and retention metadata"
            )
        if retention.get("retained_cycles") != len(cycles):
            raise ReportError(f"{path.name} block {index} retained cycle count is inconsistent")
        source_count = retention.get("source_cycles_for_block")
        if (
            isinstance(source_count, bool)
            or not isinstance(source_count, int)
            or not 0 <= source_count <= MAX_MODULATION_CYCLE_ROWS
        ):
            raise ReportError(f"{path.name} block {index} has invalid source cycle count")
        projection_rows += len(cycles)
        for cycle_index, cycle in enumerate(cycles):
            if not isinstance(cycle, dict):
                raise ReportError(
                    f"{path.name} block {index} cycle {cycle_index} must be an object"
                )
            sample = cycle.get("sample_s")
            if isinstance(sample, bool) or not isinstance(sample, (int, float)):
                raise ReportError(
                    f"{path.name} block {index} cycle {cycle_index} lacks numeric sample_s"
                )
            for field in ("references", "duties"):
                values = cycle.get(field)
                if not isinstance(values, dict) or set(values) != {"a", "b", "c"}:
                    raise ReportError(
                        f"{path.name} block {index} cycle {cycle_index} {field} "
                        "must contain phases a, b, c"
                    )
                if any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in values.values()
                ):
                    raise ReportError(
                        f"{path.name} block {index} cycle {cycle_index} {field} "
                        "must contain numeric values"
                    )
    if projection_rows > MAX_THREE_PHASE_MODULATION_PROJECTION_ROWS:
        raise ReportError(
            f"{path.name} exceeds the {MAX_THREE_PHASE_MODULATION_PROJECTION_ROWS} "
            "three-phase projection-row limit"
        )
    return payload


_SUPPORTED_MODULATION_CYCLE_SCHEMAS = {
    "pear-sim/modulation-cycle/three-phase-spwm/v0.1": "full_run_sampled_spwm_commands",
    "pear-sim/modulation-cycle/three-phase-npc-spwm/v0.1": (
        "full_run_sampled_npc_commands"
    ),
}
MAX_NPC_MODULATION_FILE_BYTES = 262_144


def _read_npc_modulation(path: Path) -> dict[str, Any]:
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ReportError(f"Cannot inspect NPC modulation {path.name}: {exc}") from exc
    if file_size > MAX_NPC_MODULATION_FILE_BYTES:
        raise ReportError(
            f"{path.name} is {file_size} bytes; npc-modulation limit is "
            f"{MAX_NPC_MODULATION_FILE_BYTES} bytes"
        )
    payload = _read_json_object(path, "NPC modulation")
    if payload.get("schema") != "pear-sim/npc-modulation-observation/v0.1":
        raise ReportError(
            f"{path.name} has unsupported npc-modulation schema {payload.get('schema')!r}"
        )
    block = payload.get("block")
    if not isinstance(block, dict) or not isinstance(block.get("block_id"), str):
        raise ReportError(f"{path.name} block must be an object with a block_id")
    cycles_recorded = payload.get("cycles_recorded")
    if (
        isinstance(cycles_recorded, bool)
        or not isinstance(cycles_recorded, int)
        or not 0 < cycles_recorded <= MAX_MODULATION_CYCLE_ROWS
    ):
        raise ReportError(f"{path.name} cycles_recorded is invalid")
    suppressions = payload.get("minimum_pulse_suppressions")
    if not isinstance(suppressions, dict) or set(suppressions) != {"a", "b", "c"}:
        raise ReportError(
            f"{path.name} minimum_pulse_suppressions must map phases a, b, c"
        )
    if not isinstance(payload.get("claim_boundary"), str):
        raise ReportError(f"{path.name} claim_boundary must be a string")
    return payload


def _inspect_modulation_cycles(path: Path) -> dict[str, Any]:
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ReportError(f"Cannot inspect modulation cycles {path.name}: {exc}") from exc
    if file_size > MAX_MODULATION_CYCLE_FILE_BYTES:
        raise ReportError(
            f"{path.name} is {file_size} bytes; modulation-cycle limit is "
            f"{MAX_MODULATION_CYCLE_FILE_BYTES} bytes"
        )
    row_count = 0
    cycle_schema: str | None = None
    block_counts: Counter[str] = Counter()
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise ReportError(f"Cannot read modulation cycles {path.name}: {exc}") from exc
    with handle:
        for row_count, raw_line in enumerate(handle, start=1):
            if row_count > MAX_MODULATION_CYCLE_ROWS:
                raise ReportError(
                    f"{path.name} exceeds the {MAX_MODULATION_CYCLE_ROWS} "
                    "modulation-cycle row limit"
                )
            if len(raw_line) > MAX_MODULATION_CYCLE_LINE_BYTES:
                raise ReportError(
                    f"{path.name} line {row_count} exceeds the "
                    f"{MAX_MODULATION_CYCLE_LINE_BYTES}-byte line limit"
                )
            try:
                decoded = raw_line.decode("utf-8")
                row = _strict_json_loads(decoded)
            except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise ReportError(f"{path.name} line {row_count} is invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ReportError(f"{path.name} line {row_count} must be an object")
            row_schema = row.get("schema")
            if row_schema not in _SUPPORTED_MODULATION_CYCLE_SCHEMAS:
                raise ReportError(f"{path.name} line {row_count} has unsupported cycle schema")
            if cycle_schema is None:
                cycle_schema = row_schema
            elif row_schema != cycle_schema:
                raise ReportError(f"{path.name} mixes modulation-cycle schemas")
            block_id = row.get("block_id")
            if not isinstance(block_id, str) or not block_id:
                raise ReportError(f"{path.name} line {row_count} lacks block_id")
            block_counts[block_id] += 1
    if row_count == 0:
        raise ReportError(f"{path.name} contains no modulation cycles")
    return {
        "source": path.name,
        "bytes": file_size,
        "row_count": row_count,
        "rows_by_block": dict(sorted(block_counts.items())),
        "cycle_schema": cycle_schema,
        "authoritative_for": _SUPPORTED_MODULATION_CYCLE_SCHEMAS[cycle_schema],
        "retained_in_report_json": False,
    }


def _read_harmonics(path: Path) -> dict[str, Any]:
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise ReportError(f"Cannot inspect harmonic analysis {path.name}: {exc}") from exc
    if file_size > MAX_HARMONICS_FILE_BYTES:
        raise ReportError(
            f"{path.name} is {file_size} bytes; harmonic-analysis limit is "
            f"{MAX_HARMONICS_FILE_BYTES} bytes"
        )
    payload = _read_json_object(path, "harmonic analysis")
    if payload.get("schema") != "pear-sim/harmonic-analysis/v0.1":
        raise ReportError(
            f"{path.name} has unsupported harmonic-analysis schema {payload.get('schema')!r}"
        )
    for field in ("window", "frequency", "method"):
        if not isinstance(payload.get(field), dict):
            raise ReportError(f"{path.name} {field} must be an object")
    signals = payload.get("signals")
    if not isinstance(signals, list) or not signals or len(signals) > MAX_HARMONIC_SIGNALS:
        raise ReportError(f"{path.name} signals must contain at most {MAX_HARMONIC_SIGNALS} rows")
    seen: set[str] = set()
    for index, signal in enumerate(signals):
        if not isinstance(signal, dict):
            raise ReportError(f"{path.name} signal {index} must be an object")
        name = signal.get("signal")
        if not isinstance(name, str) or not name or name in seen:
            raise ReportError(f"{path.name} signal {index} has an invalid or duplicate name")
        seen.add(name)
        unit = signal.get("unit")
        known_units = sorted(set(SIGNAL_UNITS.values()))
        if unit not in known_units:
            raise ReportError(
                f"{path.name} signal {index} unit must be one of "
                f"{', '.join(repr(item) for item in known_units)}"
            )
        if not isinstance(signal.get("fundamental"), dict):
            raise ReportError(f"{path.name} signal {index} lacks fundamental data")
        coefficients = signal.get("coefficients")
        if not isinstance(coefficients, list) or len(coefficients) > MAX_HARMONICS_PER_SIGNAL:
            raise ReportError(
                f"{path.name} signal {index} coefficients must contain at most "
                f"{MAX_HARMONICS_PER_SIGNAL} rows"
            )
        previous_order = -1
        for coefficient_index, coefficient in enumerate(coefficients):
            if not isinstance(coefficient, dict):
                raise ReportError(
                    f"{path.name} signal {index} coefficient {coefficient_index} must be an object"
                )
            order = coefficient.get("harmonic")
            rms = coefficient.get("rms_amplitude")
            if (
                isinstance(order, bool)
                or not isinstance(order, int)
                or order <= previous_order
                or isinstance(rms, bool)
                or not isinstance(rms, (int, float))
                or rms < 0.0
            ):
                raise ReportError(
                    f"{path.name} signal {index} harmonic rows must have strictly "
                    "increasing orders and non-negative RMS"
                )
            previous_order = order
    sequence_components = payload.get("sequence_components")
    if not isinstance(sequence_components, list) or len(sequence_components) > MAX_HARMONIC_SIGNALS:
        raise ReportError(
            f"{path.name} sequence_components must contain at most {MAX_HARMONIC_SIGNALS} rows"
        )
    return payload


def _read_warnings(run: dict[str, Any], path: Path | None) -> list[Any]:
    if path is None:
        return _as_warning_list(run.get("warnings", []))
    try:
        external = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReportError(f"Cannot read warnings artifact {path.name}: {exc}") from exc
    return _as_warning_list(external)


def _as_warning_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value.copy()
    return [value]


def _read_events(path: Path | None) -> dict[str, Any]:
    limits = {
        "max_file_bytes": MAX_EVENT_FILE_BYTES,
        "max_line_bytes": MAX_EVENT_LINE_BYTES,
        "max_rows": MAX_EVENT_ROWS,
        "max_html_rows": MAX_EVENT_HTML_ROWS,
    }
    if path is None:
        return {
            "source": None,
            "count": 0,
            "by_kind": {},
            "by_target": {},
            "time": {
                "numeric_count": 0,
                "start_s": None,
                "stop_s": None,
                "monotonic": None,
            },
            "items": [],
            "limits": limits,
        }
    file_size = path.stat().st_size
    if file_size > MAX_EVENT_FILE_BYTES:
        raise ReportError(
            f"{path.name} is {file_size} bytes; limit is {MAX_EVENT_FILE_BYTES} bytes"
        )

    items: list[dict[str, Any]] = []
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise ReportError(f"Cannot read events artifact {path.name}: {exc}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if line_number > MAX_EVENT_ROWS:
                raise ReportError(f"{path.name} exceeds the {MAX_EVENT_ROWS} event-row limit")
            if len(raw_line) > MAX_EVENT_LINE_BYTES:
                raise ReportError(
                    f"{path.name} line {line_number} is {len(raw_line)} bytes; "
                    f"limit is {MAX_EVENT_LINE_BYTES} bytes"
                )
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ReportError(f"{path.name} line {line_number} is not UTF-8") from exc
            if not line.strip():
                raise ReportError(f"{path.name} line {line_number} is empty")
            try:
                event = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_unique_json_object,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ReportError(f"{path.name} line {line_number} is invalid JSON: {exc}") from exc
            if not isinstance(event, dict):
                raise ReportError(f"{path.name} line {line_number} must contain a JSON object")
            try:
                _require_finite_json(event)
            except ValueError as exc:
                raise ReportError(f"{path.name} line {line_number} is invalid JSON: {exc}") from exc
            items.append(event)

    kind_counts = Counter(_event_group(event.get("kind"), "(missing)") for event in items)
    target_counts = Counter(_event_group(event.get("target"), "(missing)") for event in items)
    times = [
        float(event["time"])
        for event in items
        if isinstance(event.get("time"), (int, float))
        and not isinstance(event.get("time"), bool)
        and math.isfinite(float(event["time"]))
    ]
    all_times_numeric = len(times) == len(items)
    monotonic = (
        all(current >= previous for previous, current in zip(times, times[1:]))
        if items and all_times_numeric
        else None
    )
    return {
        "source": path.name,
        "count": len(items),
        "by_kind": dict(sorted(kind_counts.items())),
        "by_target": dict(sorted(target_counts.items())),
        "time": {
            "numeric_count": len(times),
            "start_s": min(times) if times else None,
            "stop_s": max(times) if times else None,
            "monotonic": monotonic,
        },
        "items": items,
        "limits": limits,
    }


def _strict_json_loads(source: str) -> Any:
    value = json.loads(
        source,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )
    _require_finite_json(value)
    return value


def _require_finite_json(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite_json(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _require_finite_json(item, f"{path}[{index}]")


def _event_group(value: Any, missing: str) -> str:
    return value if isinstance(value, str) and value else missing


def _read_and_verify_artifact_manifest(
    path: Path,
    directory: Path,
    run: dict[str, Any],
) -> dict[str, Any]:
    manifest = _read_json_object(path, "artifact manifest")
    if manifest.get("schema") != "pear-sim/artifact-manifest/v0.1":
        raise ReportError(
            "artifact-manifest.json field 'schema' must be 'pear-sim/artifact-manifest/v0.1'"
        )
    subject = manifest.get("subject")
    if not isinstance(subject, dict) or subject.get("kind") != "run":
        raise ReportError("artifact-manifest.json subject.kind must be 'run'")
    run_sha256 = run.get("run_sha256")
    if not _is_sha256(run_sha256):
        raise ReportError("run.json field 'run_sha256' must be a lowercase SHA-256 digest")
    if subject.get("sha256") != run_sha256:
        raise ReportError(
            "artifact-manifest.json subject.sha256 does not match run.json run_sha256"
        )

    files = run.get("files")
    if not isinstance(files, dict):
        raise ReportError("run.json field 'files' must be an object")
    expected_artifacts = {(directory / "run.json").resolve()}
    expected_artifacts.update(
        _artifact_path(directory, reference, str(role)) for role, reference in files.items()
    )

    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise ReportError("artifact-manifest.json field 'artifacts' must be an array")
    seen: set[Path] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ReportError(f"artifact-manifest.json artifact {index} must be an object")
        reference = record.get("path")
        expected_hash = record.get("sha256")
        expected_bytes = record.get("bytes")
        artifact = _artifact_path(directory, reference, f"artifact manifest entry {index}")
        relative_artifact = artifact.relative_to(directory).as_posix()
        if artifact in seen:
            raise ReportError(f"artifact-manifest.json repeats artifact {relative_artifact!r}")
        seen.add(artifact)
        actual_hash = _sha256_file(artifact)
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            raise ReportError(f"SHA-256 mismatch for source artifact {relative_artifact!r}")
        if not isinstance(expected_bytes, int) or artifact.stat().st_size != expected_bytes:
            raise ReportError(f"Byte-count mismatch for source artifact {relative_artifact!r}")
    if seen != expected_artifacts:
        missing = sorted(
            artifact.relative_to(directory).as_posix() for artifact in expected_artifacts - seen
        )
        unexpected = sorted(
            artifact.relative_to(directory).as_posix() for artifact in seen - expected_artifacts
        )
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(repr(name) for name in missing))
        if unexpected:
            details.append("unexpected " + ", ".join(repr(name) for name in unexpected))
        raise ReportError(
            "artifact-manifest.json must exactly cover run.json and run.json files: "
            + "; ".join(details)
        )
    return manifest


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_waveforms(path: Path) -> _Waveforms:
    try:
        handle = path.open(newline="", encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ReportError(f"Cannot read waveforms {path.name}: {exc}") from exc
    with handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ReportError("waveforms.csv is empty") from exc
        if len(header) < 2 or header[0] != "time":
            raise ReportError("waveforms.csv must start with time and at least one signal column")
        if len(set(header)) != len(header):
            raise ReportError("waveforms.csv contains duplicate column names")
        time: list[float] = []
        signals = {name: [] for name in header[1:]}
        for row_number, row in enumerate(reader, start=2):
            if row_number > _MAX_WAVEFORM_ROWS + 1:
                raise ReportError(
                    f"waveforms.csv exceeds the {_MAX_WAVEFORM_ROWS} row alpha report limit"
                )
            if len(row) != len(header):
                raise ReportError(
                    f"waveforms.csv row {row_number} has {len(row)} fields; expected {len(header)}"
                )
            try:
                parsed = [float(value) for value in row]
            except ValueError as exc:
                raise ReportError(f"waveforms.csv row {row_number} is not numeric") from exc
            if any(not math.isfinite(value) for value in parsed):
                raise ReportError(f"waveforms.csv row {row_number} contains a non-finite value")
            time.append(parsed[0])
            for name, value in zip(header[1:], parsed[1:], strict=True):
                signals[name].append(value)
    if not time:
        raise ReportError("waveforms.csv contains no samples")
    if any(current < previous for previous, current in zip(time, time[1:])):
        raise ReportError("waveforms.csv time must be monotonically non-decreasing")
    return _Waveforms(time=time, signals=signals)


def _load_schematic(source: str | Path | None) -> tuple[str | None, dict[str, Any]]:
    if source is None:
        return None, {"present": False}
    try:
        if isinstance(source, Path):
            raw = source.read_text(encoding="utf-8")
            source_kind = "file"
        elif isinstance(source, str):
            raw = source
            source_kind = "inline"
        else:
            raise ReportError("schematic_svg must be an SVG string, Path, or None")
        sanitized = sanitize_schematic_svg(raw)
    except (OSError, UnicodeError, SvgError) as exc:
        raise ReportError(f"Cannot use schematic SVG: {exc}") from exc
    return sanitized, {
        "present": True,
        "source_kind": source_kind,
        "original_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "sanitized_sha256": hashlib.sha256(sanitized.encode("utf-8")).hexdigest(),
    }


_REPORT_CSS = """
:root { color-scheme: light; --ink: #0f1f33; --ink-2: #3d5066; --ink-3: #667a8f;
  --line: #e3e8ef; --bg: #f3f5f8; --surface: #ffffff; --raised: #f8fafc;
  --accent: #1f5fbf; --warn-bg: #fff8e6; --warn-line: #d99a00;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--ink); background: var(--bg); font-size: 15px; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); line-height: 1.5; -webkit-text-size-adjust: 100%; }
.layout { max-width: 1260px; margin: 0 auto; padding: 24px 20px 64px; display: grid;
  grid-template-columns: 200px minmax(0, 1fr); gap: 24px; align-items: start; }
nav.toc { position: sticky; top: 16px; font-size: .84rem; }
nav.toc .toc-title { font-size: .7rem; text-transform: uppercase; letter-spacing: .08em;
  color: var(--ink-3); margin: 4px 10px 8px; }
nav.toc a { display: block; padding: 5px 10px; color: var(--ink-2); text-decoration: none;
  border-radius: 6px; }
nav.toc a:hover, nav.toc a:focus-visible { background: #e8edf4; color: var(--ink); }
main { min-width: 0; }
header, section { background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
  padding: 22px 24px; margin-bottom: 16px; }
section { scroll-margin-top: 16px; }
h1 { font-size: 1.55rem; margin: 0 0 2px; letter-spacing: -.01em; }
h2 { font-size: 1.12rem; margin: 0 0 14px; padding-bottom: 10px; border-bottom: 1px solid var(--line); }
h3 { font-size: .95rem; margin: 22px 0 10px; color: var(--ink); }
h4 { font-size: .88rem; margin: 16px 0 8px; }
p { margin: 0 0 12px; }
a { color: var(--accent); }
.subtitle, .empty, figcaption, .note { color: var(--ink-3); font-size: .87rem; }
.eyebrow { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: var(--ink-3); margin-bottom: 4px; }
.badge { display: inline-block; margin: 10px 0 0; padding: 4px 10px; border-radius: 999px;
  background: #fff1c2; color: #6e4700; font-weight: 700; font-size: .72rem; letter-spacing: .05em; }
.notice { border-left: 3px solid var(--warn-line); padding: 10px 14px; background: var(--warn-bg);
  border-radius: 0 8px 8px 0; margin: 14px 0; font-size: .89rem; }
.verification-notice { border-left-color: #486581; background: #f0f4f8; }
.physics-notice { border-left-color: #0f766e; background: #ecfdf5; }
.coverage-notice { border-left-color: #7c3aed; background: #f5f3ff; }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px; margin: 14px 0; }
.coverage-cards { grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); }
.coverage-cards .method-card { grid-column: 1 / -1; }
.card { border: 1px solid var(--line); border-radius: 10px; padding: 11px 13px; background: var(--raised); min-width: 0; }
.card small { display: block; color: var(--ink-3); font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; margin-bottom: 3px; }
.card strong { font-size: 1.02rem; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.card code { font-size: .76rem; color: var(--ink-3); overflow-wrap: anywhere; }
.scroll { overflow-x: auto; margin: 8px 0 16px; border: 1px solid var(--line); border-radius: 10px; }
table { width: 100%; border-collapse: collapse; font-size: .85rem; }
.scroll table { min-width: 640px; }
th, td { border-bottom: 1px solid var(--line); padding: 8px 12px; text-align: left; vertical-align: top; }
tbody tr:last-child td { border-bottom: 0; }
tbody tr:hover td { background: #fafbfd; }
th { color: var(--ink-2); background: var(--raised); font-weight: 600; font-size: .74rem;
  text-transform: uppercase; letter-spacing: .04em; white-space: nowrap; }
td.number { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
td code { overflow-wrap: normal; word-break: normal; }
td:first-child > code { white-space: nowrap; }
td .sub { color: var(--ink-3); font-size: .8rem; margin-top: 3px; }
td.tolerance code { font-size: .76rem; color: var(--ink-2); }
td:last-child { min-width: 18ch; }
code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
code { font-size: .9em; }
p code, li code, .note code { overflow-wrap: anywhere; }
pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #102a43; color: #f0f4f8; padding: 14px; border-radius: 8px; font-size: .78rem; }
details { margin: 10px 0; }
details > summary { cursor: pointer; font-weight: 600; color: var(--ink-2); padding: 6px 0; }
details > summary:hover { color: var(--ink); }
.pill { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: .72rem; font-weight: 700;
  letter-spacing: .03em; background: #eef2f7; color: var(--ink-2); white-space: nowrap; vertical-align: 1px; }
.pill.warning, .pill.serious { background: #fff1c2; color: #6e4700; }
.pill.error, .pill.critical { background: #fde4e4; color: #9b1c1c; }
.pill.info { background: #e7f0fb; color: #1f4f8f; }
.pill.good { background: #e3f5e9; color: #166534; }
ul.findings { list-style: none; padding: 0; margin: 0; }
ul.findings li { padding: 9px 0; border-bottom: 1px solid var(--line); font-size: .9rem; }
ul.findings li:last-child { border-bottom: 0; }
ul.findings .code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .78rem; color: var(--ink-3); margin-left: 6px; }
ul.findings .text { display: block; margin-top: 3px; color: var(--ink); }
.plot { margin: 12px 0 20px; }
.plot-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 0 16px; }
.plot svg, .schematic svg { display: block; width: 100%; height: auto; border: 1px solid var(--line); border-radius: 10px; background: #fff; }
.schematic { overflow-x: auto; }
figcaption { margin-top: 6px; font-size: .78rem; }
.mode-timeline { display: flex; gap: 8px; list-style: none; margin: 0; padding: 4px 0 10px; min-width: max-content; }
.mode-timeline li { width: 190px; border: 1px solid #d9e2ec; border-top: 5px solid #829ab1; border-radius: 8px; padding: 9px; background: #fbfcfe; }
.mode-timeline li[data-mode="on"] { border-top-color: #2563eb; }
.mode-timeline li[data-mode="freewheel"] { border-top-color: #0f766e; }
.mode-timeline li[data-mode="idle"] { border-top-color: #f59e0b; }
.mode-timeline li[data-mode="unclassified"] { border-top-color: #dc2626; }
.mode-timeline small { display: block; color: #627d98; }
.warnings { padding-left: 24px; color: #8d2b0b; }
footer { color: var(--ink-3); text-align: center; font-size: .78rem; padding-top: 12px; }
@media (max-width: 900px) {
  .layout { grid-template-columns: 1fr; padding: 0 10px 48px; gap: 0; }
  nav.toc { position: sticky; top: 0; z-index: 5; display: flex; gap: 6px; overflow-x: auto;
    padding: 10px 0; background: var(--bg); }
  nav.toc .toc-title { display: none; }
  nav.toc a { white-space: nowrap; border: 1px solid var(--line); background: var(--surface); }
  header, section { padding: 16px 14px; border-radius: 10px; }
  section { scroll-margin-top: 60px; }
  .plot-grid { grid-template-columns: 1fr; }
  .cards, .coverage-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
  .card { padding: 9px 10px; }
  h1 { font-size: 1.3rem; }
  .stack { border: 0; overflow: visible; }
  .stack table { min-width: 0; }
  .stack thead { display: none; }
  .stack tr { display: block; border: 1px solid var(--line); border-radius: 10px; margin-bottom: 10px; padding: 4px 0; }
  .stack td { display: grid; grid-template-columns: 7.5rem minmax(0, 1fr); gap: 8px; border: 0;
    padding: 5px 12px; text-align: left; white-space: normal; min-width: 0; }
  .stack td::before { content: attr(data-label); color: var(--ink-3); font-size: .72rem;
    text-transform: uppercase; letter-spacing: .05em; padding-top: 2px; }
  .stack td code { white-space: normal; overflow-wrap: anywhere; }
}
@media print {
  body { background: #fff; } nav.toc { display: none; } .layout { display: block; }
  header, section { break-inside: avoid; border-color: #ccc; }
  details { display: block; } details > summary { display: none; }
}
.schematic-svg .canvas { fill: #fbfcfe; }
.schematic-svg .title { fill: #162033; font: 600 18px 'DejaVu Sans', sans-serif; }
.schematic-svg .subtitle { fill: #58667a; font: 12px 'DejaVu Sans', sans-serif; }
.schematic-svg .layout-warning { fill: #9a3412; font: 600 11px 'DejaVu Sans', sans-serif; }
.schematic-svg .wire, .schematic-svg .ground-symbol { fill: none; stroke: #334155; stroke-linecap: round; stroke-linejoin: round; stroke-width: 1.25; }
.schematic-svg .junction { fill: #334155; stroke: #fbfcfe; stroke-width: 1; }
.schematic-svg .ground-junction { fill: #0f766e; }
.schematic-svg .net-label { fill: #0f766e; font: 600 12px 'DejaVu Sans', sans-serif; paint-order: stroke; stroke: #fbfcfe; stroke-width: 4px; stroke-linejoin: round; }
.schematic-svg .component-symbol { fill: #fbfcfe; }
.schematic-svg .symbol-stroke { fill: none; stroke: #0f172a; stroke-linecap: round; stroke-linejoin: round; stroke-width: 1.25; }
.schematic-svg .terminal { fill: #0f172a; }
.schematic-svg .diode-fill { fill: #fbfcfe; stroke: #0f172a; stroke-linejoin: round; stroke-width: 1.25; }
.schematic-svg .probe-letter { fill: #0f172a; font: 700 13px 'DejaVu Sans', sans-serif; }
.schematic-svg .component-label { fill: #172033; font: 700 12px 'DejaVu Sans', sans-serif; paint-order: stroke; stroke: #fbfcfe; stroke-width: 4px; stroke-linejoin: round; }
.schematic-svg .component-value { fill: #58667a; font: 11px 'DejaVu Sans', sans-serif; paint-order: stroke; stroke: #fbfcfe; stroke-width: 4px; stroke-linejoin: round; }
"""


def _with_navigation(document: str) -> str:
    """Give every section an anchor and build the table of contents from them."""

    used: set[str] = set()
    links: list[str] = []

    def anchor(match: re.Match[str]) -> str:
        title = re.sub(r"<[^>]+>", "", match.group(2))
        slug = re.sub(r"[^a-z0-9]+", "-", html.unescape(title).lower()).strip("-") or "section"
        candidate, index = slug, 2
        while candidate in used:
            candidate, index = f"{slug}-{index}", index + 1
        used.add(candidate)
        links.append(f'<a href="#{candidate}">{title}</a>')
        return f'<section id="{candidate}">{match.group(1)}<h2>{match.group(2)}</h2>'

    document = re.sub(r"<section>(\s*)<h2>(.*?)</h2>", anchor, document)
    nav = '<nav class="toc" aria-label="Report sections"><div class="toc-title">Contents</div>' + "".join(links) + "</nav>"
    return document.replace("<!--TOC-->", nav, 1)


def _render_html(
    report: dict[str, Any],
    waveforms: _Waveforms,
    schematic_svg: str | None,
) -> str:
    run = report["run"]
    metrics = report["metrics"]
    warnings = report["warnings"]
    events = report["events"]
    provenance = report["provenance"]
    source_summary = report.get("source_summary")
    resolved_model = report.get("resolved_model")
    analytic_checks = report.get("analytic_checks")
    physics_audit = report.get("physics_audit")
    mode_coverage = report.get("mode_coverage")
    gate_drive = report.get("gate_drive")
    three_phase_modulation = report.get("three_phase_modulation")
    modulation_cycle_source = report.get("modulation_cycle_source")
    harmonics = report.get("harmonics")
    model = run.get("model", "unnamed model")
    solver = run.get("solver", "unknown")
    rtol = run.get("rtol", "unknown")
    atol = run.get("atol", "unknown")
    model_hash = run.get("model_sha256", "unavailable")
    source_version = run.get("pear_sim_version", "unknown")
    backend = run.get("backend", "unknown")
    maturity = run.get("maturity", "unknown")
    stats_value = run.get("stats")
    stats = stats_value if isinstance(stats_value, dict) else {}
    wall_time_s = stats.get("wall_time_s")
    integration_steps = stats.get("integration_steps")
    natural_events = stats.get("natural_events")
    natural_events_detail = (
        "coverage not exercised"
        if isinstance(natural_events, (int, float))
        and not isinstance(natural_events, bool)
        and natural_events == 0
        else "recorded natural commutation count"
        if natural_events is not None
        else "not recorded"
    )

    findings_html, warning_count, note_count = _findings_section(warnings)
    schematic_html = ""
    if schematic_svg is not None:
        schematic_html = (
            '<section><h2>Schematic</h2><div class="schematic">'
            + schematic_svg
            + "</div></section>"
        )

    device_signals = _device_current_signals(resolved_model, waveforms.signals)

    primary_signals = {
        signal: values
        for signal, values in waveforms.signals.items()
        if signal not in device_signals
    }
    zoom = _zoom_window(waveforms.time, primary_signals)

    def figure(signal: str, values: Any, *, with_zoom: bool = False) -> str:
        if len(waveforms.time) <= _PLOT_POINTS:
            caption = f"{_escape(signal)} — all {len(waveforms.time)} raw samples drawn."
            return f'<figure class="plot">{waveform_svg(waveforms.time, values, signal)}' \
                f"<figcaption>{caption}</figcaption></figure>"
        if not with_zoom or zoom is None:
            return (
                '<figure class="plot">'
                + waveform_svg(waveforms.time, values, signal)
                + f"<figcaption>{_escape(signal)} — {len(waveforms.time)} raw samples; "
                "display keeps every bucket's minimum and maximum (at most "
                f"{_PLOT_POINTS} points).</figcaption></figure>"
            )
        first, last = zoom
        t0, t1 = waveforms.time[first], waveforms.time[last]
        window = f"{_format_si(t0, 's')} → {_format_si(t1, 's')}"
        return (
            '<figure class="plot">'
            + waveform_svg(waveforms.time, values, signal, highlight=(t0, t1))
            + waveform_svg(
                [t - t0 for t in waveforms.time[first : last + 1]],
                values[first : last + 1],
                signal,
                title=f"{signal} — zoom {window}",
                time_label=f"time from {_format_si(t0, 's')}",
                height=210.0,
            )
            + f"<figcaption>{_escape(signal)} — top: the whole run, {len(waveforms.time)} raw "
            f"samples, each of {_PLOT_POINTS} display buckets keeps its minimum and maximum, so "
            "fast switching shows as a filled band. Bottom: the shaded window, every raw sample "
            f"drawn ({last - first + 1} samples).</figcaption></figure>"
        )

    primary_plots = "".join(
        figure(signal, values, with_zoom=True) for signal, values in primary_signals.items()
    )
    device_plots = "".join(
        figure(signal, values)
        for signal, values in waveforms.signals.items()
        if signal in device_signals
    )
    plot_html = primary_plots + (
        f"<details><summary>Semiconductor device currents ({len(device_signals)})</summary>"
        f'<div class="plot-grid">{device_plots}</div></details>'
        if device_plots
        else ""
    )
    primary_metrics = {
        name: value for name, value in metrics.items() if name not in device_signals
    }
    metrics_html = (
        _metrics_table(primary_metrics, compact=True)
        + "<details><summary>Every recorded metric for every signal "
        f"({len(metrics)} signals)</summary>"
        f'<div class="scroll">{_metrics_table(metrics)}</div></details>'
    )
    events_html = _events_section(events)
    measurement_initialization_html = _measurement_initialization_section(resolved_model)
    analytic_checks_html = _analytic_checks_section(analytic_checks)
    physics_audit_html = _physics_audit_section(physics_audit)
    losses_thermal_html = _losses_thermal_section(report.get("losses_thermal"))
    mode_coverage_html = _mode_coverage_section(mode_coverage)
    gate_drive_html = _gate_drive_section(gate_drive)
    three_phase_modulation_html = _three_phase_modulation_section(
        three_phase_modulation,
        modulation_cycle_source,
    )
    npc_modulation_html = _npc_modulation_section(
        report.get("npc_modulation"),
        modulation_cycle_source,
    )
    harmonics_html = _harmonics_section(harmonics)
    artifacts_html = "".join(
        "<tr>"
        f"<td>{_escape(role)}</td>"
        f"<td>{_escape(record['file'])}</td>"
        f'<td class="number">{_escape(record["bytes"])}</td>'
        f"<td><code>{_escape(record['sha256'])}</code></td>"
        "</tr>"
        for role, record in provenance["input_artifacts"].items()
    )
    hashes_html = "".join(
        f"<tr><td>{_escape(name)}</td><td><code>{_escape(value)}</code></td></tr>"
        for name, value in provenance["declared_hashes"].items()
    )
    if not hashes_html:
        hashes_html = '<tr><td colspan="2" class="empty">No declared hashes.</td></tr>'
    metadata_json = _escape(json.dumps(run, indent=2, sort_keys=True, ensure_ascii=False))
    source_details = ""
    if source_summary is not None:
        summary_json = _escape(
            json.dumps(source_summary, indent=2, sort_keys=True, ensure_ascii=False)
        )
        source_details += (
            f"<details><summary>Source summary.json</summary><pre>{summary_json}</pre></details>"
        )
    if resolved_model is not None:
        resolved_json = _escape(
            json.dumps(resolved_model, indent=2, sort_keys=True, ensure_ascii=False)
        )
        source_details += (
            f"<details><summary>Resolved model</summary><pre>{resolved_json}</pre></details>"
        )
    source_manifest = provenance["source_artifact_manifest"]
    manifest_status = (
        "present and verified"
        if source_manifest.get("verified")
        else "not present; hashes below were calculated by the report generator"
    )

    return _with_navigation(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; script-src 'none'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <title>PEARSim alpha report — {_escape(model)}</title>
  <style>
{_REPORT_CSS}
  </style>
</head>
<body>
<div class="layout">
<!--TOC-->
<main>
  <header>
    <div class="eyebrow">PEARSim alpha report</div>
    <h1>{_escape(model)}</h1>
    <span class="badge">ALPHA OBSERVATION — NOT A VALIDATION CLAIM</span>
    <p class="notice">This offline report presents the existing run metadata, metrics, warnings, and waveforms. It does not by itself establish numerical verification, cross-tool compatibility, physical validation, or fitness for safety-critical decisions.</p>
    <div class="cards">
      <div class="card"><small>Solver</small><strong>{_escape(solver)}</strong><br><code>rtol={_escape(rtol)}, atol={_escape(atol)}</code></div>
      <div class="card"><small>Samples</small><strong>{len(waveforms.time)}</strong><br><code>{len(waveforms.signals)} signal(s)</code></div>
      <div class="card"><small>Warnings</small><strong>{warning_count}</strong><br><code>{note_count} note(s)</code></div>
      <div class="card"><small>Wall time</small><strong>{_escape(_quantity(wall_time_s, "s"))}</strong><br><code>wall_time_s, recorded</code></div>
      <div class="card"><small>Integration steps</small><strong>{_escape(_quantity(integration_steps, ""))}</strong></div>
      <div class="card"><small>Natural commutations</small><strong>{_escape(_quantity(natural_events, ""))}</strong><br><code>{_escape(natural_events_detail)}</code></div>
      <div class="card"><small>Source PEARSim</small><strong>{_escape(source_version)}</strong></div>
      <div class="card"><small>Backend / maturity</small><strong>{_escape(backend)}</strong><br><code>{_escape(maturity)}</code></div>
      <div class="card"><small>Model SHA-256</small><code>{_escape(model_hash)}</code></div>
    </div>
  </header>
  {findings_html}
  {schematic_html}
  <section>
    <h2>Waveforms</h2>
    <p class="subtitle">Produced by the PEARSim reference kernel ({_escape(backend)}, solver {_escape(solver)}) and read back from this run's own <code>waveforms.csv</code> — no external simulator and no interpolation beyond the deterministic display decimation. Raw samples remain authoritative.</p>
    {plot_html}
  </section>
  {losses_thermal_html}
  {physics_audit_html}
  {gate_drive_html}
  {three_phase_modulation_html}
  {npc_modulation_html}
  {harmonics_html}
  {analytic_checks_html}
  {mode_coverage_html}
  {measurement_initialization_html}
  <section>
    <h2>Recorded metrics</h2>
    <p class="subtitle">Values are copied from the run's metrics artifact; this report does not recompute acceptance KPI.</p>
    {metrics_html}
  </section>
  <section>
    <h2>Events</h2>
    {events_html}
  </section>
  <section>
    <h2>Provenance</h2>
    <p class="subtitle">Source artifact manifest: {_escape(manifest_status)}.</p>
    <h3>Declared semantic hashes</h3>
    <div class="scroll"><table>
      <thead><tr><th>Hash role</th><th>SHA-256</th></tr></thead>
      <tbody>{hashes_html}</tbody>
    </table></div>
    <h3>Input artifacts</h3>
    <div class="scroll"><table>
      <thead><tr><th>Role</th><th>File</th><th>Bytes</th><th>SHA-256</th></tr></thead>
      <tbody>{artifacts_html}</tbody>
    </table></div>
    <details><summary>Complete run metadata</summary><pre>{metadata_json}</pre></details>
    {source_details}
  </section>
  <footer>Generated deterministically by PEARSim {_escape(__version__)} · schema {_escape(REPORT_SCHEMA)} · no scripts or network resources</footer>
</main>
</div>
</body>
</html>
""")


def _npc_modulation_section(
    modulation: Any,
    cycle_source: Any,
) -> str:
    if not isinstance(modulation, dict):
        return ""
    block = modulation.get("block") if isinstance(modulation.get("block"), dict) else {}
    source = cycle_source if isinstance(cycle_source, dict) else {}
    config_rows = "".join(
        f"<tr><td>{_escape(label)}</td><td>{_escape(_display_value(block.get(key)))}</td></tr>"
        for label, key in (
            ("Block", "block_id"),
            ("Levels", "levels"),
            ("Carrier frequency [Hz]", "carrier_frequency_hz"),
            ("Fundamental frequency [Hz]", "fundamental_frequency_hz"),
            ("Modulation index", "modulation_index"),
            ("Dead time [s]", "dead_time_s"),
            ("Phase sequence", "phase_sequence"),
            ("Injection", "injection"),
            ("Carrier ratio", "carrier_ratio"),
        )
    )
    suppressions = modulation.get("minimum_pulse_suppressions")
    suppression_rows = ""
    total_suppressed = 0
    if isinstance(suppressions, dict):
        for phase in sorted(suppressions):
            by_pair = suppressions[phase]
            if not isinstance(by_pair, dict):
                continue
            for pair in sorted(by_pair):
                count = by_pair[pair]
                if isinstance(count, int) and not isinstance(count, bool):
                    total_suppressed += count
                if count:
                    suppression_rows += (
                        f"<tr><td>{_escape(phase)}</td><td>{_escape(pair)}</td>"
                        f'<td class="number">{_escape(count)}</td></tr>'
                    )
    if not suppression_rows:
        suppression_rows = (
            '<tr><td colspan="3" class="empty">No band-edge pulse was narrower '
            "than the dead time in this run.</td></tr>"
        )
    claim = modulation.get("claim_boundary")
    claim_html = (
        f'<p class="note">{_escape(claim)}</p>' if isinstance(claim, str) else ""
    )
    return (
        "<section><h2>NPC level-shifted modulation</h2>"
        "<p>Level-shifted in-phase (PD) carrier command record; the "
        "authoritative per-cell log is "
        f"<code>{_escape(source.get('source', 'modulation-cycles.jsonl'))}</code> "
        f"({_escape(_display_value(source.get('row_count')))} cells; every row is "
        "independently recomputed from the modulation law before this report is "
        "generated).</p>"
        f'<table><tbody>{config_rows}</tbody></table>'
        f"<h3>Recorded cells</h3><p>{_escape(_display_value(modulation.get('cycles_recorded')))} "
        "carrier cells; "
        f"{_escape(_display_value(modulation.get('band_transition_handovers')))} "
        "band-transition hand-overs (pair saturation changes bridged at a cell "
        f"boundary through one dead time); {total_suppressed} minimum-pulse "
        "suppression(s).</p>"
        "<h3>Minimum-pulse suppressions</h3>"
        "<table><thead><tr><th>Phase</th><th>Pair</th><th>Suppressed cells</th>"
        f"</tr></thead><tbody>{suppression_rows}</tbody></table>"
        f"{claim_html}"
        "</section>"
    )


def _three_phase_modulation_section(
    modulation: Any,
    cycle_source: Any,
) -> str:
    if not isinstance(modulation, dict):
        return ""
    claim = modulation.get("claim") if isinstance(modulation.get("claim"), dict) else {}
    summary = modulation.get("summary") if isinstance(modulation.get("summary"), dict) else {}
    source = cycle_source if isinstance(cycle_source, dict) else {}
    blocks = modulation.get("blocks") if isinstance(modulation.get("blocks"), list) else []
    block_sections: list[str] = []
    for raw_block in blocks:
        if not isinstance(raw_block, dict):
            continue
        config = raw_block.get("config") if isinstance(raw_block.get("config"), dict) else {}
        bridge = raw_block.get("bridge") if isinstance(raw_block.get("bridge"), dict) else {}
        block_summary = (
            raw_block.get("summary") if isinstance(raw_block.get("summary"), dict) else {}
        )
        projection = (
            raw_block.get("fundamental_period_projection")
            if isinstance(raw_block.get("fundamental_period_projection"), dict)
            else {}
        )
        retention = (
            projection.get("retention") if isinstance(projection.get("retention"), dict) else {}
        )
        cycles = projection.get("cycles") if isinstance(projection.get("cycles"), list) else []
        phases = bridge.get("phases") if isinstance(bridge.get("phases"), list) else []
        mapping_rows: list[str] = []
        for phase in phases:
            if not isinstance(phase, dict):
                continue
            high = phase.get("high") if isinstance(phase.get("high"), dict) else {}
            low = phase.get("low") if isinstance(phase.get("low"), dict) else {}
            mapping_rows.append(
                "<tr>"
                f"<td><strong>{_escape(_display_value(phase.get('phase'))).upper()}</strong></td>"
                f"<td><code>{_escape(_display_value(phase.get('pole_node')))}</code></td>"
                f"<td><code>{_escape(_display_value(high.get('gate_output_id')))}</code> → "
                f"<code>{_escape(_display_value(high.get('device_id')))}</code></td>"
                f"<td><code>{_escape(_display_value(high.get('p')))} → "
                f"{_escape(_display_value(high.get('n')))}</code></td>"
                f"<td><code>{_escape(_display_value(low.get('gate_output_id')))}</code> → "
                f"<code>{_escape(_display_value(low.get('device_id')))}</code></td>"
                f"<td><code>{_escape(_display_value(low.get('p')))} → "
                f"{_escape(_display_value(low.get('n')))}</code></td>"
                "</tr>"
            )
        mapping_html = (
            '<div class="scroll"><table><thead><tr><th>Phase</th><th>Pole</th>'
            "<th>High gate → device</th><th>High p → n</th>"
            "<th>Low gate → device</th><th>Low p → n</th></tr></thead><tbody>"
            + "".join(mapping_rows)
            + "</tbody></table></div>"
        )

        valid_cycles = [cycle for cycle in cycles if isinstance(cycle, dict)]
        plot_html = ""
        if valid_cycles:
            times = [float(cycle["sample_s"]) for cycle in valid_cycles]
            references = {
                phase: [float(cycle["references"][phase]) for cycle in valid_cycles]
                for phase in ("a", "b", "c")
            }
            duties = {
                phase: [float(cycle["duties"][phase]) for cycle in valid_cycles]
                for phase in ("a", "b", "c")
            }
            plot_html = (
                '<figure class="plot">'
                + phase_series_svg(
                    times,
                    references,
                    "Sampled SPWM phase references",
                    y_label="normalized reference",
                )
                + "<figcaption>Producer-recorded offline midpoint samples; aligned A/B/C "
                "traces. This is not a claim of causal digital sampling, and no carrier "
                "waveform or KPI is reconstructed by the report.</figcaption></figure>"
                '<figure class="plot">'
                + phase_series_svg(
                    times,
                    duties,
                    "Commanded SPWM phase duties",
                    y_label="duty [fraction]",
                )
                + "<figcaption>Bounded display projection from three-phase-modulation.json; "
                "the full command record remains in modulation-cycles.jsonl.</figcaption></figure>"
            )

        cycle_rows: list[str] = []
        for cycle in valid_cycles[:MAX_MODULATION_HTML_ROWS]:
            references = cycle.get("references")
            duties = cycle.get("duties")
            cycle_rows.append(
                "<tr>"
                f'<td class="number">{_escape(_display_value(cycle.get("cycle_index")))}</td>'
                f'<td class="number">{_escape(_display_value(cycle.get("sample_s")))}</td>'
                f'<td class="number">{_escape(_display_value(cycle.get("electrical_angle_rad")))}</td>'
                f"<td><code>{_escape(_display_value(references))}</code></td>"
                f"<td><code>{_escape(_display_value(duties))}</code></td>"
                "</tr>"
            )
        cycle_table = (
            "<details><summary>Bounded carrier-command table</summary>"
            '<div class="scroll"><table><thead><tr><th>Cycle</th><th>Sample [s]</th>'
            "<th>Electrical angle [rad]</th><th>References A/B/C</th>"
            "<th>Duties A/B/C</th></tr></thead><tbody>"
            + "".join(cycle_rows)
            + "</tbody></table></div>"
            f'<p class="subtitle">HTML shows {len(cycle_rows)} of '
            f"{len(valid_cycles)} cycles retained in report.json.</p></details>"
        )
        zoom = raw_block.get("carrier_zoom")
        zoom_record = zoom if isinstance(zoom, dict) else {}
        zoom_cycle = zoom_record.get("cycle")
        zoom_html = (
            "<h4>Carrier-period zoom: minimum effective pulse</h4>"
            '<div class="cards">'
            '<div class="card"><small>Selection</small>'
            f"<code>{_escape(_display_value(zoom_record.get('selection')))}</code></div>"
            '<div class="card"><small>Cycle</small>'
            f"<strong>{_escape(_display_value(zoom_cycle.get('cycle_index') if isinstance(zoom_cycle, dict) else None))}</strong></div>"
            '<div class="card"><small>Window [s]</small>'
            f"<code>{_escape(_display_value(zoom_cycle.get('start_s') if isinstance(zoom_cycle, dict) else None))} → "
            f"{_escape(_display_value(zoom_cycle.get('end_s') if isinstance(zoom_cycle, dict) else None))}</code></div>"
            '<div class="card"><small>Minimum effective pulse [s]</small>'
            f"<strong>{_escape(_display_value(zoom_record.get('minimum_effective_pulse_s')))}</strong></div>"
            "</div>"
            f"<pre>{_escape(json.dumps(zoom_cycle, indent=2, sort_keys=True, ensure_ascii=False))}</pre>"
        )
        block_sections.append(
            f"<h3><code>{_escape(_display_value(raw_block.get('block_id')))}</code></h3>"
            '<div class="cards">'
            '<div class="card"><small>Strategy / sampling</small>'
            f"<strong>{_escape(_display_value(config.get('control_type')))}</strong><br>"
            f"<code>{_escape(_display_value(config.get('sampling')))}</code></div>"
            '<div class="card"><small>Modulation index</small>'
            f"<strong>{_escape(_display_value(config.get('modulation_index')))}</strong><br>"
            f"<code>{_escape(_display_value(config.get('modulation_index_definition')))}</code></div>"
            '<div class="card"><small>f1 / fs [Hz]</small>'
            f"<strong>{_escape(_display_value(config.get('fundamental_frequency_hz')))} / "
            f"{_escape(_display_value(config.get('carrier_frequency_hz')))}</strong><br>"
            f"<code>ratio={_escape(_display_value(config.get('carrier_ratio')))}</code></div>"
            '<div class="card"><small>Sequence</small>'
            f"<strong>{_escape(_display_value(config.get('phase_sequence')))}</strong></div>"
            '<div class="card"><small>Dead time [s]</small>'
            f"<strong>{_escape(_display_value(config.get('dead_time_s')))}</strong></div>"
            '<div class="card"><small>Duty range</small>'
            f"<strong>{_escape(_display_value(block_summary.get('minimum_duty')))} → "
            f"{_escape(_display_value(block_summary.get('maximum_duty')))}</strong></div>"
            '<div class="card"><small>Minimum effective pulse [s]</small>'
            f"<strong>{_escape(_display_value(block_summary.get('minimum_effective_pulse_s')))}</strong></div>"
            '<div class="card"><small>Scheduled edges</small>'
            f"<strong>{_escape(_display_value(block_summary.get('scheduled_gate_edge_count')))}</strong></div>"
            '<div class="card"><small>Observed overlap</small>'
            f"<strong>{_escape(_display_value(block_summary.get('observed_overlap_count')))}</strong></div>"
            "</div>"
            "<h4>Explicit phase → gate → device mapping</h4>"
            f'<p class="subtitle">DC rails: <code>{_escape(_display_value(bridge.get("positive_rail_node")))}</code> / '
            f"<code>{_escape(_display_value(bridge.get('negative_rail_node')))}</code>. "
            "Mapping is copied from producer metadata and explicit p→n switch fields.</p>"
            f"{mapping_html}"
            '<p class="notice"><strong>Bounded fundamental-period projection.</strong> '
            f"Selection: <code>{_escape(_display_value(projection.get('selection')))}</code>; "
            f"window <code>{_escape(_display_value(projection.get('window_start_s')))} → "
            f"{_escape(_display_value(projection.get('window_end_s')))} s</code>. Retained "
            f"<strong>{_escape(_display_value(retention.get('retained_cycles')))}</strong> of "
            f"<strong>{_escape(_display_value(retention.get('source_cycles_for_block')))}</strong> full-run cycles; "
            f"truncated: <strong>{_escape(_display_value(retention.get('truncated')))}</strong>. "
            f"Raw source: <code>{_escape(_display_value(retention.get('source_artifact')))}</code>. "
            f"{_escape(_display_value(retention.get('statement')))}</p>"
            f"{plot_html}{cycle_table}{zoom_html}"
        )
    return f"""
  <section>
    <h2>Three-phase modulation command evidence</h2>
    <p class="notice coverage-notice"><strong>RECORDED SPWM COMMANDS — NOT INVERTER VALIDATION OR A DRIVER CLAIM.</strong> {_escape(_display_value(claim.get("statement")))}</p>
    <div class="cards">
      <div class="card"><small>Blocks</small><strong>{_escape(_display_value(summary.get("block_count")))}</strong></div>
      <div class="card"><small>Phases / gates</small><strong>{_escape(_display_value(summary.get("phase_count")))} / {_escape(_display_value(summary.get("gate_output_count")))}</strong></div>
      <div class="card"><small>Full command rows</small><strong>{_escape(_display_value(source.get("row_count")))}</strong><br><code>{_escape(_display_value(source.get("source")))}</code></div>
      <div class="card"><small>Realized gate edges</small><strong>{_escape(_display_value(summary.get("scheduled_gate_edge_count")))}</strong><br><code>events.jsonl authoritative</code></div>
      <div class="card"><small>Overlap</small><strong>{_escape(_display_value(summary.get("observed_overlap_count")))}</strong></div>
      <div class="card"><small>Acceptance</small><strong>{_escape(_display_value(summary.get("acceptance_status")))}</strong></div>
    </div>
    {"".join(block_sections)}
  </section>"""


def _harmonics_section(harmonics: Any) -> str:
    if not isinstance(harmonics, dict):
        return ""
    claim = harmonics.get("claim") if isinstance(harmonics.get("claim"), dict) else {}
    window = harmonics.get("window") if isinstance(harmonics.get("window"), dict) else {}
    frequency = harmonics.get("frequency") if isinstance(harmonics.get("frequency"), dict) else {}
    method = harmonics.get("method") if isinstance(harmonics.get("method"), dict) else {}
    summary = harmonics.get("summary") if isinstance(harmonics.get("summary"), dict) else {}
    signals = harmonics.get("signals") if isinstance(harmonics.get("signals"), list) else []
    sequence_components = (
        harmonics.get("sequence_components")
        if isinstance(harmonics.get("sequence_components"), list)
        else []
    )
    sequence_rows: list[str] = []
    for record in sequence_components:
        if not isinstance(record, dict):
            continue
        phasors = record.get("rms_phasors")
        phasors = phasors if isinstance(phasors, dict) else {}
        positive = phasors.get("positive") if isinstance(phasors.get("positive"), dict) else {}
        negative = phasors.get("negative") if isinstance(phasors.get("negative"), dict) else {}
        zero = phasors.get("zero") if isinstance(phasors.get("zero"), dict) else {}
        sequence_rows.append(
            "<tr>"
            f"<td>{_escape(_display_value(record.get('family')))}</td>"
            f"<td><code>{_escape(_display_value(record.get('convention')))}</code></td>"
            f'<td class="number">{_escape(_display_value(positive.get("magnitude")))}</td>'
            f'<td class="number">{_escape(_display_value(positive.get("phase_deg")))}</td>'
            f'<td class="number">{_escape(_display_value(negative.get("magnitude")))}</td>'
            f'<td class="number">{_escape(_display_value(zero.get("magnitude")))}</td>'
            f'<td class="number">{_escape(_display_value(record.get("negative_to_positive_fraction")))}</td>'
            f'<td class="number">{_escape(_display_value(record.get("zero_to_positive_fraction")))}</td>'
            "</tr>"
        )
    sequence_html = (
        "<h3>Fundamental sequence components</h3>"
        '<div class="scroll"><table><thead><tr><th>Family</th><th>Convention</th>'
        "<th>Positive RMS</th><th>Positive phase [deg]</th><th>Negative RMS</th>"
        "<th>Zero RMS</th><th>Negative / positive</th><th>Zero / positive</th>"
        "</tr></thead><tbody>" + "".join(sequence_rows) + "</tbody></table></div>"
        if sequence_rows
        else '<p class="empty">No complete three-phase sequence-component family was recorded.</p>'
    )
    signal_sections: list[str] = []
    remaining_rows = MAX_HARMONIC_HTML_ROWS
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        unit = signal.get("unit")
        fundamental = (
            signal.get("fundamental") if isinstance(signal.get("fundamental"), dict) else {}
        )
        coefficients = (
            signal.get("coefficients") if isinstance(signal.get("coefficients"), list) else []
        )
        rendered_coefficients = coefficients[:remaining_rows]
        remaining_rows -= len(rendered_coefficients)
        spectrum = (
            '<figure class="plot">'
            + harmonic_spectrum_svg(
                coefficients,
                _display_value(signal.get("signal")),
                max_bars=64,
            )
            + f"<figcaption>Producer coefficients; spectrum displays at most 64 of "
            f"{len(coefficients)} stored orders.</figcaption></figure>"
            if coefficients
            else '<p class="empty">No harmonic coefficients were recorded.</p>'
        )
        coefficient_rows = "".join(
            "<tr>"
            f'<td class="number">{_escape(_display_value(row.get("harmonic")))}</td>'
            f'<td class="number">{_escape(_display_value(row.get("frequency_hz")))}</td>'
            f'<td class="number">{_escape(_display_value(row.get("rms_amplitude")))}</td>'
            f'<td class="number">{_escape(_display_value(row.get("peak_amplitude")))}</td>'
            f'<td class="number">{_escape(_display_value(row.get("phase_deg")))}</td>'
            "</tr>"
            for row in rendered_coefficients
            if isinstance(row, dict)
        )
        coefficient_table = (
            "<details><summary>Producer harmonic coefficients</summary>"
            '<div class="scroll"><table><thead><tr><th>Order</th><th>Frequency [Hz]</th>'
            f"<th>RMS [{_escape(_display_value(unit))}]</th>"
            f"<th>Peak [{_escape(_display_value(unit))}]</th>"
            "<th>Phase [deg]</th></tr></thead><tbody>" + coefficient_rows + "</tbody></table></div>"
            f'<p class="subtitle">HTML retained {len(rendered_coefficients)} of '
            f"{len(coefficients)} coefficient rows; harmonics.json remains authoritative.</p>"
            "</details>"
        )
        signal_sections.append(
            f"<h3><code>{_escape(_display_value(signal.get('signal')))}</code> — "
            f"{_escape(_display_value(signal.get('semantic')))} "
            f"[{_escape(_display_value(unit))}]</h3>"
            '<div class="cards">'
            f'<div class="card"><small>Fundamental RMS [{_escape(_display_value(unit))}]</small>'
            f"<strong>{_escape(_display_value(fundamental.get('rms_amplitude')))}</strong></div>"
            f'<div class="card"><small>Fundamental peak [{_escape(_display_value(unit))}]</small>'
            f"<strong>{_escape(_display_value(fundamental.get('peak_amplitude')))}</strong></div>"
            '<div class="card"><small>Fundamental phase [deg]</small>'
            f"<strong>{_escape(_display_value(fundamental.get('phase_deg')))}</strong></div>"
            '<div class="card"><small>THD [fraction]</small>'
            f"<strong>{_escape(_display_value(signal.get('thd', {}).get('value_fraction') if isinstance(signal.get('thd'), dict) else None))}</strong></div>"
            f'<div class="card"><small>DC [{_escape(_display_value(unit))}]</small>'
            f"<strong>{_escape(_display_value(signal.get('dc')))} "
            "</strong></div>"
            '<div class="card"><small>Status</small>'
            f"<strong>{_escape(_display_value(signal.get('thd', {}).get('status') if isinstance(signal.get('thd'), dict) else None))}</strong></div>"
            "</div>"
            f"{spectrum}{coefficient_table}"
        )
    return f"""
  <section>
    <h2>Harmonic analysis</h2>
    <p class="notice verification-notice"><strong>PRODUCER-CALCULATED MEASUREMENT — THE REPORT DOES NOT RECOMPUTE FOURIER KPI.</strong> {_escape(_display_value(claim.get("statement")))}</p>
    <div class="cards">
      <div class="card"><small>Status / acceptance</small><strong>{_escape(_display_value(harmonics.get("status")))} / {_escape(_display_value(summary.get("acceptance_status")))}</strong></div>
      <div class="card"><small>Window [s]</small><strong>{_escape(_display_value(window.get("start_s")))} → {_escape(_display_value(window.get("stop_s")))}</strong><br><code>{_escape(_display_value(window.get("endpoint_convention")))}</code></div>
      <div class="card"><small>Fundamental [Hz]</small><strong>{_escape(_display_value(frequency.get("fundamental_hz")))}</strong><br><code>periods={_escape(_display_value(window.get("fundamental_period_count")))}</code></div>
      <div class="card"><small>Bandwidth</small><strong>{_escape(_display_value(frequency.get("bandwidth_hz")))} Hz</strong><br><code>max harmonic={_escape(_display_value(frequency.get("max_harmonic")))}</code></div>
      <div class="card"><small>Method</small><code>{_escape(_display_value(method.get("name")))}</code></div>
      <div class="card"><small>DC / amplitude policy</small><code>{_escape(_display_value(method.get("dc_policy")))}; {_escape(_display_value(method.get("amplitude_convention")))}</code></div>
      <div class="card"><small>Source</small><code>{_escape(_display_value(window.get("source")))}</code></div>
    </div>
    {sequence_html}
    {"".join(signal_sections)}
  </section>"""


def _gate_drive_section(gate_drive: Any) -> str:
    if not isinstance(gate_drive, dict):
        return ""

    claim_value = gate_drive.get("claim")
    claim = claim_value if isinstance(claim_value, dict) else {}
    summary_value = gate_drive.get("summary")
    summary = summary_value if isinstance(summary_value, dict) else {}
    outputs_value = gate_drive.get("gate_outputs")
    outputs = outputs_value if isinstance(outputs_value, list) else []
    controls_value = gate_drive.get("logical_controls")
    controls = controls_value if isinstance(controls_value, list) else []

    output_rows: list[str] = []
    for raw_output in outputs:
        if not isinstance(raw_output, dict):
            continue
        device_ids = raw_output.get("device_ids")
        devices = (
            ", ".join(str(item) for item in device_ids) if isinstance(device_ids, list) else ""
        )
        output_rows.append(
            "<tr>"
            f"<td><code>{_escape(_display_value(raw_output.get('gate_output_id')))}</code></td>"
            f"<td><code>{_escape(_display_value(raw_output.get('logical_control_id')))}</code></td>"
            f"<td>{_escape(_display_value(raw_output.get('role')))}</td>"
            f"<td><code>{_escape(devices or '—')}</code></td>"
            f'<td class="number">{_escape(_display_value(raw_output.get("effective_duty")))}</td>'
            f'<td class="number">{_escape(_display_value(raw_output.get("effective_phase_s")))}</td>'
            "</tr>"
        )
    outputs_html = (
        '<div class="scroll"><table><thead><tr><th>Gate output</th><th>Logical control</th>'
        "<th>Role</th><th>Controlled devices</th><th>Effective duty</th>"
        "<th>Effective phase [s]</th></tr></thead><tbody>"
        + "".join(output_rows)
        + "</tbody></table></div>"
        if output_rows
        else '<p class="empty">No gate outputs were recorded.</p>'
    )

    control_sections: list[str] = []
    for raw_control in controls:
        if not isinstance(raw_control, dict):
            continue
        timeline_value = raw_control.get("timeline")
        timeline = timeline_value if isinstance(timeline_value, dict) else {}
        retention_value = timeline.get("retention")
        retention = retention_value if isinstance(retention_value, dict) else {}
        segments_value = timeline.get("segments")
        segments = segments_value if isinstance(segments_value, list) else []
        transactions_value = timeline.get("transactions")
        transactions = transactions_value if isinstance(transactions_value, list) else []
        segment_rows: list[str] = []
        for raw_segment in segments:
            if not isinstance(raw_segment, dict):
                continue
            segment_rows.append(
                "<tr>"
                f'<td class="number">{_escape(_display_value(raw_segment.get("start_s")))}</td>'
                f'<td class="number">{_escape(_display_value(raw_segment.get("end_s")))}</td>'
                f'<td class="number">{_escape(_display_value(raw_segment.get("duration_s")))}</td>'
                f"<td><strong>{_escape(_display_value(raw_segment.get('state')))}</strong></td>"
                f"<td><code>{_escape(_display_value(raw_segment.get('gate_states')))}</code></td>"
                "</tr>"
            )
        transaction_rows: list[str] = []
        for raw_transaction in transactions:
            if not isinstance(raw_transaction, dict):
                continue
            changes_value = raw_transaction.get("changes")
            changes = changes_value if isinstance(changes_value, list) else []
            change_text = ", ".join(
                f"{change.get('gate_output_id')}: {change.get('previous')}→{change.get('current')}"
                for change in changes
                if isinstance(change, dict)
            )
            transaction_rows.append(
                "<tr>"
                f"<td><code>{_escape(_display_value(raw_transaction.get('transaction_id')))}</code></td>"
                f'<td class="number">{_escape(_display_value(raw_transaction.get("time_s")))}</td>'
                f'<td class="number">{_escape(_display_value(raw_transaction.get("offset_s")))}</td>'
                f"<td><code>{_escape(change_text or '—')}</code></td>"
                "</tr>"
            )
        segment_table = (
            '<div class="scroll"><table><thead><tr><th>Start [s]</th><th>End [s]</th>'
            "<th>Duration [s]</th><th>Leg state</th><th>Gate states</th></tr></thead><tbody>"
            + "".join(segment_rows)
            + "</tbody></table></div>"
            if segment_rows
            else '<p class="empty">No timeline segments were recorded.</p>'
        )
        transaction_table = (
            '<div class="scroll"><table><thead><tr><th>Atomic transaction</th>'
            "<th>Time [s]</th><th>Offset [s]</th><th>Changes (off before on)</th>"
            "</tr></thead><tbody>" + "".join(transaction_rows) + "</tbody></table></div>"
            if transaction_rows
            else '<p class="empty">No transitions occur in the selected timeline window.</p>'
        )
        timeline_projection = (
            '<p class="notice"><strong>Bounded timeline projection.</strong> '
            f"Selection: <code>{_escape(_display_value(timeline.get('selection')))}</code>. "
            "Retained "
            f"<strong>{_escape(_display_value(retention.get('retained_atomic_transactions')))}</strong> "
            "of "
            f"<strong>{_escape(_display_value(retention.get('source_atomic_transactions_for_control')))}</strong> "
            "full-run atomic transactions for this control; truncated: "
            f"<strong>{_escape(_display_value(retention.get('truncated')))}</strong>. "
            f"Raw source: <code>{_escape(_display_value(retention.get('source_artifact')))}</code>. "
            f"{_escape(_display_value(retention.get('statement')))}</p>"
        )
        control_sections.append(
            f"<h3><code>{_escape(_display_value(raw_control.get('logical_control_id')))}</code></h3>"
            '<div class="cards">'
            '<div class="card"><small>Type</small>'
            f"<strong>{_escape(_display_value(raw_control.get('control_type')))}</strong></div>"
            '<div class="card"><small>Frequency [Hz]</small>'
            f"<strong>{_escape(_display_value(raw_control.get('frequency_hz')))}</strong></div>"
            '<div class="card"><small>Command duty</small>'
            f"<strong>{_escape(_display_value(raw_control.get('command_duty')))}</strong></div>"
            '<div class="card"><small>Dead time [s]</small>'
            f"<strong>{_escape(_display_value(raw_control.get('dead_time_s')))}</strong></div>"
            '<div class="card"><small>Interlock</small>'
            f"<strong>{_escape(_display_value(raw_control.get('interlock')))}</strong></div>"
            '<div class="card"><small>Timeline window [s]</small>'
            f"<code>{_escape(_display_value(timeline.get('window_start_s')))} → "
            f"{_escape(_display_value(timeline.get('window_end_s')))}</code></div>"
            "</div>"
            f"{timeline_projection}{segment_table}<h4>Atomic transitions</h4>"
            f"{transaction_table}"
        )

    return f"""
  <section>
    <h2>Gate drive &amp; switching timeline</h2>
    <p class="notice coverage-notice"><strong>RECORDED SCHEDULE — NOT A DRIVER OR SAFETY CLAIM.</strong> {_escape(_display_value(claim.get("statement")))}</p>
    <div class="cards">
      <div class="card"><small>Logical controls</small><strong>{_escape(_display_value(summary.get("logical_control_count")))}</strong></div>
      <div class="card"><small>Gate outputs</small><strong>{_escape(_display_value(summary.get("gate_output_count")))}</strong></div>
      <div class="card"><small>Controlled devices</small><strong>{_escape(_display_value(summary.get("controlled_device_count")))}</strong></div>
      <div class="card"><small>Full-run atomic transactions</small><strong>{_escape(_display_value(summary.get("atomic_transaction_count")))}</strong></div>
      <div class="card"><small>Overlap segments</small><strong>{_escape(_display_value(summary.get("rendered_overlap_segment_count")))}</strong></div>
      <div class="card"><small>Interlock violations</small><strong>{_escape(_display_value(summary.get("observed_interlock_violation_count")))}</strong></div>
      <div class="card"><small>Acceptance</small><strong>{_escape(_display_value(summary.get("acceptance_status")))}</strong></div>
    </div>
    <h3>Gate-to-device mapping</h3>
    {outputs_html}
    {"".join(control_sections)}
  </section>"""


def _measurement_initialization_section(resolved_model: Any) -> str:
    if not isinstance(resolved_model, dict):
        return """
  <section>
    <h2>Measurement &amp; initialization</h2>
    <p class="empty">No resolved-model measurement conventions or initial-condition provenance were recorded for this run.</p>
  </section>"""

    conventions_value = resolved_model.get("conventions")
    conventions = conventions_value if isinstance(conventions_value, dict) else {}
    convention_rows: list[str] = []
    for raw_name, raw_record in sorted(conventions.items(), key=lambda item: str(item[0])):
        name = str(raw_name)
        if isinstance(raw_record, dict):
            definition = raw_record.get("definition")
            orientation = {
                str(key): value for key, value in raw_record.items() if key != "definition"
            }
        else:
            definition = raw_record
            orientation = {}
        convention_rows.append(
            "<tr>"
            f"<td><code>{_escape(name)}</code></td>"
            f"<td>{_escape(_display_value(definition))}</td>"
            f"<td><code>{_escape(_display_value(orientation)) if orientation else '—'}</code></td>"
            "</tr>"
        )
    conventions_html = (
        '<div class="scroll"><table><thead><tr><th>Convention</th><th>Definition</th>'
        "<th>Orientation details</th></tr></thead><tbody>"
        + "".join(convention_rows)
        + "</tbody></table></div>"
        if convention_rows
        else '<p class="empty">Measurement conventions were not recorded.</p>'
    )

    initial_value = resolved_model.get("initial_conditions")
    initial = initial_value if isinstance(initial_value, dict) else {}
    summary_value = initial.get("summary")
    summary = summary_value if isinstance(summary_value, dict) else {}
    warm_start_value = initial.get("warm_start")
    warm_start = warm_start_value if isinstance(warm_start_value, dict) else {}
    summary_fields = (
        ("State count", "state_count"),
        ("Zero seeds", "zero_count"),
        ("Nonzero seeds", "nonzero_count"),
        ("User declared", "user_declared_count"),
        ("Default zero", "default_zero_count"),
        ("Unknown origin", "unknown_origin_count"),
        ("Warm start", "warm_start"),
    )
    summary_cards = "".join(
        '<div class="card">'
        f"<small>{_escape(label)}</small>"
        f"<strong>{_escape(_display_value(summary.get(field)))}</strong>"
        "</div>"
        for label, field in summary_fields
    )
    initial_cards = (
        '<div class="cards">'
        '<div class="card"><small>Initialization strategy</small>'
        f"<strong>{_escape(_display_value(initial.get('strategy')))}</strong></div>"
        '<div class="card"><small>Analytic initialization claimed</small>'
        f"<strong>{_escape(_display_value(initial.get('analytic_initialization_claimed')))}</strong></div>"
        f"{summary_cards}</div>"
        if initial
        else ""
    )

    states_value = initial.get("states")
    states = states_value if isinstance(states_value, list) else []
    state_rows: list[str] = []
    for raw_state in states:
        if not isinstance(raw_state, dict):
            state_rows.append(
                f'<tr><td colspan="8"><code>{_escape(_display_value(raw_state))}</code></td></tr>'
            )
            continue
        seed = _value_and_unit(raw_state.get("value_si"), raw_state.get("unit"))
        state_rows.append(
            "<tr>"
            f"<td><code>{_escape(_display_value(raw_state.get('component_id')))}</code></td>"
            f"<td>{_escape(_display_value(raw_state.get('component_type')))}</td>"
            f"<td><code>{_escape(_display_value(raw_state.get('state_expression')))}</code></td>"
            f'<td class="number">{_escape(seed)}</td>'
            f"<td>{_escape(_display_value(raw_state.get('value_class')))}</td>"
            f"<td>{_escape(_display_value(raw_state.get('origin')))}</td>"
            f"<td>{_escape(_display_value(raw_state.get('source_field')))}</td>"
            f"<td>{_escape(_display_value(raw_state.get('analytic_derivation')))}</td>"
            "</tr>"
        )
    states_html = (
        '<div class="scroll"><table><thead><tr><th>Component</th><th>Type</th>'
        "<th>State expression</th><th>Seed [SI]</th><th>Class</th><th>Origin</th>"
        "<th>Source field</th><th>Analytic derivation</th></tr></thead><tbody>"
        + "".join(state_rows)
        + "</tbody></table></div>"
        if state_rows
        else '<p class="empty">No initial-condition state records were provided.</p>'
    )
    initialization_note = (
        '<p class="subtitle">'
        f"Origin policy: {_escape(_display_value(initial.get('origin_policy')))}. "
        f"Warm-start definition: {_escape(_display_value(warm_start.get('definition')))}. "
        f"Warm-start provenance: {_escape(_display_value(warm_start.get('provenance_claim')))}."
        "</p>"
        if initial
        else '<p class="empty">Initial-condition provenance was not recorded.</p>'
    )
    return f"""
  <section>
    <h2>Measurement &amp; initialization</h2>
    <p class="subtitle">Sign conventions and initialization provenance are copied from <code>resolved_model</code>; the report does not infer either.</p>
    <h3>Measurement conventions</h3>
    {conventions_html}
    <h3>Initial conditions</h3>
    {initial_cards}
    {initialization_note}
    {states_html}
  </section>"""


def _analytic_checks_section(analytic_checks: Any) -> str:
    if not isinstance(analytic_checks, dict):
        return ""

    claim_value = analytic_checks.get("claim")
    claim = claim_value if isinstance(claim_value, dict) else {}
    recognition_value = analytic_checks.get("recognition")
    recognition = recognition_value if isinstance(recognition_value, dict) else {}
    preconditions_value = analytic_checks.get("preconditions")
    preconditions = preconditions_value if isinstance(preconditions_value, list) else []
    checks_value = analytic_checks.get("checks")
    checks = checks_value if isinstance(checks_value, list) else []

    precondition_rows: list[str] = []
    for raw_precondition in preconditions:
        if not isinstance(raw_precondition, dict):
            precondition_rows.append(
                '<tr><td colspan="6"><code>'
                f"{_escape(_display_value(raw_precondition))}</code></td></tr>"
            )
            continue
        precondition_rows.append(
            "<tr>"
            f"<td><code>{_escape(_display_value(raw_precondition.get('id')))}</code></td>"
            f"<td>{_escape(_display_value(raw_precondition.get('description')))}</td>"
            f"<td>{_recorded_measurement_html(raw_precondition.get('measured'))}</td>"
            f"<td><code>{_escape(_display_value(raw_precondition.get('tolerance')))}</code></td>"
            f"<td><strong>{_escape(_display_value(raw_precondition.get('status')))}</strong></td>"
            f"<td>{_escape(_display_value(raw_precondition.get('reason')))}</td>"
            "</tr>"
        )
    preconditions_html = (
        '<div class="scroll"><table><thead><tr><th>ID</th><th>Precondition</th>'
        "<th>Simulated</th><th>Tolerance</th><th>Status</th><th>Reason</th>"
        "</tr></thead><tbody>" + "".join(precondition_rows) + "</tbody></table></div>"
        if precondition_rows
        else '<p class="empty">No analytic preconditions were recorded.</p>'
    )

    any_reason = any(
        isinstance(item, dict) and item.get("reason") not in (None, "") for item in checks
    )
    check_rows: list[str] = []
    for raw_check in checks:
        if not isinstance(raw_check, dict):
            check_rows.append(
                f'<tr><td colspan="7"><code>{_escape(_display_value(raw_check))}</code></td></tr>'
            )
            continue
        delta_value = raw_check.get("delta")
        delta = delta_value if isinstance(delta_value, dict) else {}
        reason_cell = (
            f'<td data-label="Reason">{_escape(_display_value(raw_check.get("reason")))}</td>'
            if any_reason
            else ""
        )
        check_rows.append(
            "<tr>"
            f'<td data-label="Check"><code>{_escape(_display_value(raw_check.get("id")))}</code>'
            f'<div class="sub"><code>{_escape(_display_value(raw_check.get("relation")))}</code></div></td>'
            f'<td data-label="Expected">{_recorded_measurement_html(raw_check.get("expected"))}</td>'
            f'<td data-label="Simulated">{_recorded_measurement_html(raw_check.get("measured"))}</td>'
            f'<td data-label="Δ abs / Δ %" class="number">{_escape(_value_and_unit(delta.get("absolute"), delta.get("unit")))}'
            f'<div class="sub">{_escape(_relative_percent(delta.get("relative")))}</div></td>'
            f'<td data-label="Tolerance" class="tolerance">{_analytic_tolerance_html(raw_check.get("tolerance"))}</td>'
            f'<td data-label="Status">{_status_pill(raw_check.get("status"))}</td>'
            + reason_cell
            + "</tr>"
        )
    checks_html = (
        '<div class="scroll stack"><table><thead><tr><th>Check / relation</th>'
        "<th>Expected</th><th>Simulated</th><th>Δ abs / Δ %</th>"
        "<th>Tolerance</th><th>Status</th>"
        + ("<th>Reason</th>" if any_reason else "")
        + "</tr></thead><tbody>"
        + "".join(check_rows)
        + "</tbody></table></div>"
        if check_rows
        else '<p class="empty">No analytic checks were recorded.</p>'
    )

    validation_claim = claim.get("validation_claim")
    claim_statement = claim.get("statement")
    recognition_summary = _display_value(recognition.get("reason"))
    if analytic_checks.get("overall_status") == "not_applicable" and checks and all(
        isinstance(item, dict) and item.get("status") == "not_applicable" for item in checks
    ):
        reasons = sorted({str(item.get("reason")) for item in checks if item.get("reason")})
        ids = ", ".join(str(item.get("id")) for item in checks)
        return f"""
  <section>
    <h2>Analytic verification self-check</h2>
    <p><span class="pill">not applicable</span> The closed-form checks ({len(checks)}: <code>{_escape(ids)}</code>) cover a topology this model is not.</p>
    <p class="subtitle">{_escape(" ".join(reasons) or recognition_summary)}</p>
    <div class="cards"><div class="card"><small>Artifact status</small><strong>not_applicable</strong></div></div>
  </section>"""
    return f"""
  <section>
    <h2>Analytic verification self-check</h2>
    <p class="notice verification-notice"><strong>VERIFICATION SELF-CHECK — NOT A VALIDATION CLAIM.</strong> Artifact declaration: <code>validation_claim={_escape(_display_value(validation_claim))}</code>. {_escape(_display_value(claim_statement))}</p>
    <div class="cards">
      <div class="card"><small>Overall status</small><strong>{_escape(_display_value(analytic_checks.get("overall_status")))}</strong></div>
      <div class="card"><small>Kind</small><strong>{_escape(_display_value(analytic_checks.get("kind")))}</strong></div>
      <div class="card"><small>Recognized topology</small><strong>{_escape(_display_value(recognition.get("topology")))}</strong><br><code>{_escape(_display_value(recognition.get("status")))}</code></div>
      <div class="card"><small>Recognition</small><code>{_escape(recognition_summary)}</code></div>
    </div>
    <h3>Preconditions</h3>
    {preconditions_html}
    <h3>Expected versus simulated</h3>
    <p class="subtitle">Expected, simulated (<code>measured</code> in the artifact: a value read from this run, not a laboratory measurement), delta, tolerance, and status values come from <code>analytic_checks</code>. Relative delta is displayed as percent; acceptance status is not recomputed here.</p>
    {checks_html}
  </section>"""


def _mode_coverage_section(mode_coverage: Any) -> str:
    if not isinstance(mode_coverage, dict):
        return ""

    claim_value = mode_coverage.get("claim")
    claim = claim_value if isinstance(claim_value, dict) else {}
    recognition_value = mode_coverage.get("recognition")
    recognition = recognition_value if isinstance(recognition_value, dict) else {}
    subjects_value = recognition.get("subjects")
    subjects = subjects_value if isinstance(subjects_value, dict) else {}
    window_value = mode_coverage.get("window")
    window = window_value if isinstance(window_value, dict) else {}
    classification_value = mode_coverage.get("classification")
    classification = classification_value if isinstance(classification_value, dict) else {}
    summary_value = mode_coverage.get("summary")
    summary = summary_value if isinstance(summary_value, dict) else {}
    natural_value = summary.get("natural_commutation")
    natural = natural_value if isinstance(natural_value, dict) else {}
    diode_value = mode_coverage.get("diode_conduction")
    diode = diode_value if isinstance(diode_value, dict) else {}
    inductor_value = mode_coverage.get("inductor_current")
    inductor = inductor_value if isinstance(inductor_value, dict) else {}

    exercised = natural.get("exercised")
    if exercised is True:
        exercised_detail = "coverage exercised; this is an observation, not a pass claim"
    elif exercised is False:
        exercised_detail = "coverage not exercised; this is not an engine failure"
    else:
        exercised_detail = "coverage state not recorded"

    subjects_rows = "".join(
        "<tr>"
        f"<td><code>{_escape(role)}</code></td>"
        f"<td><code>{_escape(_display_value(component_id))}</code></td>"
        "</tr>"
        for role, component_id in sorted(
            ((str(raw_role), raw_id) for raw_role, raw_id in subjects.items()),
            key=lambda item: item[0],
        )
    )
    subjects_html = (
        '<div class="scroll"><table><thead><tr><th>Subject role</th><th>ID</th>'
        "</tr></thead><tbody>" + subjects_rows + "</tbody></table></div>"
        if subjects_rows
        else '<p class="empty">No recognized subject mapping was recorded.</p>'
    )

    rules_value = classification.get("rules")
    rules = rules_value if isinstance(rules_value, dict) else {}
    rules_rows = "".join(
        "<tr>"
        f"<td><code>{_escape(mode)}</code></td>"
        f"<td><code>{_escape(_display_value(rule))}</code></td>"
        "</tr>"
        for mode, rule in _ordered_mode_items(rules)
    )
    rules_html = (
        '<div class="scroll"><table><thead><tr><th>Mode</th><th>Recorded rule</th>'
        "</tr></thead><tbody>" + rules_rows + "</tbody></table></div>"
        if rules_rows
        else '<p class="empty">No mode-classification rules were recorded.</p>'
    )

    durations_value = summary.get("duration_by_mode_s")
    durations = durations_value if isinstance(durations_value, dict) else {}
    fractions_value = summary.get("fraction_by_mode")
    fractions = fractions_value if isinstance(fractions_value, dict) else {}
    mode_names = set(str(name) for name in durations) | set(str(name) for name in fractions)
    mode_rows = "".join(
        "<tr>"
        f"<td><code>{_escape(mode)}</code></td>"
        f'<td class="number">{_escape(_display_value(durations.get(mode)))}</td>'
        f'<td class="number">{_escape(_display_value(fractions.get(mode)))}</td>'
        "</tr>"
        for mode in _ordered_mode_names(mode_names)
    )
    modes_html = (
        '<div class="scroll"><table><thead><tr><th>Mode</th>'
        "<th>Recorded duration [s]</th><th>Recorded fraction</th></tr></thead><tbody>"
        + mode_rows
        + "</tbody></table></div>"
        if mode_rows
        else '<p class="empty">No mode-duration or mode-fraction summary was recorded.</p>'
    )

    event_timing_value = mode_coverage.get("event_timing")
    event_timing = event_timing_value if isinstance(event_timing_value, dict) else {}
    events_value = event_timing.get("events")
    events = events_value if isinstance(events_value, list) else []
    rendered_events = events[:MAX_MODE_COVERAGE_HTML_EVENTS]
    event_rows: list[str] = []
    for raw_event in rendered_events:
        if not isinstance(raw_event, dict):
            event_rows.append(
                f'<tr><td colspan="8"><code>{_escape(_display_value(raw_event))}</code></td></tr>'
            )
            continue
        transition = (
            f"{_display_value(raw_event.get('previous'))} → "
            f"{_display_value(raw_event.get('current'))}"
        )
        event_rows.append(
            "<tr>"
            f'<td class="number">{_escape(_display_value(raw_event.get("index")))}</td>'
            f'<td class="number">{_escape(_display_value(raw_event.get("time_s")))}</td>'
            f"<td>{_escape(_display_value(raw_event.get('kind')))}</td>"
            f"<td><code>{_escape(_display_value(raw_event.get('target')))}</code></td>"
            f"<td>{_escape(transition)}</td>"
            f'<td class="number">{_escape(_display_value(raw_event.get("cycle_index")))}</td>'
            f'<td class="number">{_escape(_display_value(raw_event.get("offset_from_cycle_start_s")))}</td>'
            f'<td class="number">{_escape(_display_value(raw_event.get("phase_fraction")))}</td>'
            "</tr>"
        )
    event_truncation_html = ""
    if len(events) > len(rendered_events):
        event_truncation_html = (
            f'<p class="notice">HTML shows the first {len(rendered_events)} of '
            f"{len(events)} recorded mode-coverage events. report.json retains the "
            "complete bounded artifact.</p>"
        )
    event_summary = (
        f"Recorded event timing table ({len(rendered_events)} of {len(events)} rows)"
        if len(events) > len(rendered_events)
        else (
            f"Recorded event timing table ({len(rendered_events)} "
            f"{'row' if len(rendered_events) == 1 else 'rows'})"
        )
    )
    events_html = (
        f"<details><summary>{event_summary}</summary>"
        '<div class="scroll"><table><thead><tr><th>#</th><th>Time [s]</th>'
        "<th>Kind</th><th>Target</th><th>Transition</th><th>Cycle</th>"
        "<th>Cycle offset [s]</th><th>Phase fraction</th></tr></thead><tbody>"
        + "".join(event_rows)
        + "</tbody></table></div>"
        + event_truncation_html
        + "</details>"
        if event_rows
        else '<p class="empty">No mode-coverage event timing rows were recorded.</p>'
    )

    intervals_value = mode_coverage.get("intervals")
    intervals = intervals_value if isinstance(intervals_value, list) else []
    rendered_intervals = intervals[:MAX_MODE_COVERAGE_HTML_INTERVALS]
    timeline_items: list[str] = []
    interval_rows: list[str] = []
    safe_modes = {"on", "freewheel", "idle", "unclassified"}
    for raw_interval in rendered_intervals:
        if not isinstance(raw_interval, dict):
            rendered = _escape(_display_value(raw_interval))
            timeline_items.append(
                f'<li data-mode="unclassified"><strong>Unstructured interval</strong>'
                f"<small><code>{rendered}</code></small></li>"
            )
            interval_rows.append(f'<tr><td colspan="10"><code>{rendered}</code></td></tr>')
            continue
        mode = _display_value(raw_interval.get("mode"))
        style_mode = mode if mode in safe_modes else "unclassified"
        start = _display_value(raw_interval.get("start_s"))
        end = _display_value(raw_interval.get("end_s"))
        duration = _display_value(raw_interval.get("duration_s"))
        timeline_items.append(
            f'<li data-mode="{style_mode}">'
            f"<strong>#{_escape(_display_value(raw_interval.get('index')))} · {_escape(mode)}</strong>"
            f"<small><code>{_escape(start)} → {_escape(end)} s</code></small>"
            f"<small>duration <code>{_escape(duration)} s</code></small>"
            f"<small>gate=<code>{_escape(_display_value(raw_interval.get('gate_on')))}</code>; "
            f"diode=<code>{_escape(_display_value(raw_interval.get('diode_conducting')))}</code></small>"
            "</li>"
        )
        interval_rows.append(
            "<tr>"
            f'<td class="number">{_escape(_display_value(raw_interval.get("index")))}</td>'
            f"<td><code>{_escape(mode)}</code></td>"
            f'<td class="number">{_escape(start)}</td>'
            f'<td class="number">{_escape(end)}</td>'
            f'<td class="number">{_escape(duration)}</td>'
            f"<td>{_escape(_display_value(raw_interval.get('gate_on')))}</td>"
            f"<td>{_escape(_display_value(raw_interval.get('diode_conducting')))}</td>"
            f'<td class="number">{_escape(_display_value(raw_interval.get("cycle_index")))}</td>'
            f"<td><code>{_escape(_display_value(raw_interval.get('start_event_indices')))}</code></td>"
            f"<td><code>{_escape(_display_value(raw_interval.get('end_event_indices')))}</code></td>"
            "</tr>"
        )
    truncation_html = ""
    if len(intervals) > len(rendered_intervals):
        truncation_html = (
            f'<p class="notice">HTML shows the first {len(rendered_intervals)} of '
            f"{len(intervals)} recorded intervals. report.json retains the complete artifact.</p>"
        )
    timeline_html = (
        '<p class="subtitle">Equal-width interval tiles are deliberately not to scale; '
        "their timestamps, durations, states, and links are copied verbatim from the artifact.</p>"
        '<div class="scroll"><ol class="mode-timeline">'
        + "".join(timeline_items)
        + "</ol></div>"
        + truncation_html
        + '<details><summary>Recorded interval table</summary><div class="scroll"><table>'
        "<thead><tr><th>#</th><th>Mode</th><th>Start [s]</th><th>End [s]</th>"
        "<th>Duration [s]</th><th>Gate on</th><th>Diode conducting</th><th>Cycle</th>"
        "<th>Start events</th><th>End events</th></tr></thead><tbody>"
        + "".join(interval_rows)
        + "</tbody></table></div></details>"
        if timeline_items
        else '<p class="empty">No mode intervals were recorded.</p>'
    )

    unavailable_html = ""
    collapse_open = collapse_close = ""
    if mode_coverage.get("status") != "observed":
        collapse_open = "<details><summary>Recorded coverage fields (none observed)</summary>"
        collapse_close = "</details>"
        unavailable_html = (
            '<p class="empty">No mode coverage is inferred: this artifact is not in '
            "<code>observed</code> status.</p>"
        )

    return f"""
  <section>
    <h2>Hybrid mode coverage</h2>
    <p class="notice coverage-notice"><strong>COVERAGE OBSERVATION — NOT A VALIDATION CLAIM.</strong> Artifact declaration: <code>validation_claim={_escape(_display_value(claim.get("validation_claim")))}</code>. {_escape(_display_value(claim.get("statement")))}</p>
    <div class="cards coverage-cards">
      <div class="card"><small>Artifact status</small><strong>{_escape(_display_value(mode_coverage.get("status")))}</strong></div>
      <div class="card"><small>Recognized topology</small><strong>{_escape(_display_value(recognition.get("topology")))}</strong><br><code>{_escape(_display_value(recognition.get("status")))}</code></div>
      <div class="card"><small>Natural commutation exercised</small><strong>{_escape(_display_value(exercised))}</strong><br><code>{_escape(exercised_detail)}</code></div>
      <div class="card"><small>Natural event count</small><strong>{_escape(_display_value(natural.get("event_count")))}</strong><br><code>diode turn-off={_escape(_display_value(natural.get("diode_turn_off_event_count")))}</code></div>
      <div class="card"><small>Recorded intervals</small><strong>{_escape(_display_value(summary.get("interval_count")))}</strong><br><code>covered duration={_escape(_display_value(summary.get("covered_duration_s")))} s</code></div>
    </div>
    <p class="subtitle">Recognition: {_escape(_display_value(recognition.get("reason")))}</p>
    {unavailable_html}
    {collapse_open}<h3>Recognized subjects</h3>
    {subjects_html}
    <h3>Observation window &amp; classification</h3>
    <div class="cards coverage-cards">
      <div class="card"><small>Window [s]</small><strong>{_escape(_display_value(window.get("start_s")))} → {_escape(_display_value(window.get("end_s")))}</strong><br><code>duration={_escape(_display_value(window.get("duration_s")))}</code></div>
      <div class="card"><small>Switching period [s]</small><strong>{_escape(_display_value(window.get("switching_period_s")))}</strong><br><code>cycle origin={_escape(_display_value(window.get("cycle_origin_s")))} s</code></div>
      <div class="card"><small>Classification method</small><code>{_escape(_display_value(classification.get("method")))}</code></div>
      <div class="card"><small>Boundary convention</small><code>{_escape(_display_value(classification.get("boundary_convention")))}</code></div>
    </div>
    {rules_html}
    <h3>Mode fractions</h3>
    <p class="subtitle">Durations and fractions below are recorded aggregate fields; this report does not derive them from the interval list.</p>
    {modes_html}
    <h3>Diode conduction &amp; unilateral constraint</h3>
    <div class="cards coverage-cards">
      <div class="card"><small>Diode / conduction fraction</small><strong>{_escape(_display_value(diode.get("component_id")))} · {_escape(_display_value(diode.get("fraction")))}</strong><br><code>{_escape(_display_value(diode.get("conducting_duration_s")))} / {_escape(_display_value(diode.get("observed_duration_s")))} s</code></div>
      <div class="card"><small>Conduction method</small><code>{_escape(_display_value(diode.get("method")))}</code></div>
      <div class="card"><small>Inductor / minimum current [A]</small><strong>{_escape(_display_value(inductor.get("component_id")))} · {_escape(_display_value(inductor.get("minimum_a")))}</strong><br><code>at {_escape(_display_value(inductor.get("minimum_time_s")))} s</code></div>
      <div class="card"><small>Maximum forbidden reverse current [A]</small><strong>{_escape(_display_value(inductor.get("maximum_forbidden_reverse_current_a")))}</strong><br><code>{_escape(_display_value(inductor.get("definition")))}</code></div>
      <div class="card method-card"><small>Current sign / extrema method</small><strong>{_escape(_display_value(inductor.get("sign_convention")))}</strong><br><code>{_escape(_display_value(inductor.get("method")))}</code><br><code>state-constraint tolerance={_escape(_display_value(inductor.get("state_constraint_tolerance_a")))} A</code></div>
    </div>
    <p class="subtitle">Constraint values are observations from the producer. This report does not calculate a reverse-current violation or a pass/fail gate.</p>
    <h3>Event timing</h3>
    <p class="subtitle">Reference: {_escape(_display_value(event_timing.get("time_reference")))}; cycle reference: {_escape(_display_value(event_timing.get("cycle_reference")))}; event merge tolerance: {_escape(_display_value(event_timing.get("event_merge_tolerance_s")))} s.</p>
    {events_html}
    <h3>Mode interval timeline</h3>
    {timeline_html}
    {collapse_close}
  </section>"""


def _ordered_mode_names(names: set[str]) -> list[str]:
    preferred = ("on", "freewheel", "idle", "unclassified")
    return [name for name in preferred if name in names] + sorted(names - set(preferred))


def _ordered_mode_items(values: dict[Any, Any]) -> list[tuple[str, Any]]:
    normalized = {str(name): value for name, value in values.items()}
    return [(name, normalized[name]) for name in _ordered_mode_names(set(normalized))]


def _losses_thermal_section(losses: Any) -> str:
    """Device losses and junction temperatures, drawn from ``losses_thermal`` only."""

    if not isinstance(losses, dict) or not isinstance(losses.get("devices"), list):
        return ""
    devices = [item for item in losses["devices"] if isinstance(item, dict)]
    if not devices:
        return ""

    def number(value: Any) -> float | None:
        return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None

    rows = []
    for device in devices:
        conduction = number((device.get("conduction") or {}).get("mean_power_w"))
        switching = number((device.get("switching") or {}).get("mean_power_w"))
        rows.append(
            {
                "id": str(device.get("component_id")),
                "kind": device.get("kind"),
                "card": device.get("card"),
                "conduction": conduction,
                "switching": switching,
                "total": number(device.get("mean_power_w")),
                "tj_max": number((device.get("thermal") or {}).get("tj_max_c")),
                "tj_final": number((device.get("thermal") or {}).get("tj_final_c")),
                "violated": (device.get("thermal") or {}).get("validity_violated"),
                "envelope": device.get("tj_envelope"),
            }
        )
    rows.sort(key=lambda row: (-(row["total"] or 0.0), row["id"]))
    bars = [
        (row["id"], row["conduction"], row["switching"])
        for row in rows
        if row["conduction"] is not None and row["switching"] is not None
    ]
    bars_html = (
        f'<figure class="plot">{loss_bars_svg(bars)}<figcaption>Mean loss per device over the '
        "whole run, largest first. Conduction and switching are the producer's own figures; "
        "hover a bar for its numbers.</figcaption></figure>"
        if bars
        else '<p class="empty">The artifact carries no per-device mean powers; '
        "re-run the simulation with this PEARSim version.</p>"
    )
    hottest = sorted(
        (row for row in rows if row["tj_max"] is not None and isinstance(row["envelope"], dict)),
        key=lambda row: (-row["tj_max"], row["id"]),
    )[:3]
    envelope_html = (
        f'<figure class="plot">{tj_envelope_svg([(row["id"], row["envelope"]) for row in hottest])}'
        "<figcaption>Junction temperature of the three hottest devices: band = minimum to "
        "maximum inside each of the producer's time buckets, line = maximum.</figcaption></figure>"
        if hottest
        else ""
    )
    hottest_row = max(
        (row for row in rows if row["tj_max"] is not None), key=lambda row: row["tj_max"], default=None
    )
    tiles = [
        (
            "Total mean loss",
            "—" if number(losses.get("total_mean_power_w")) is None
            else f"{number(losses.get('total_mean_power_w')):.4g} W",
            "all devices, whole run",
        ),
        ("Devices", str(len(devices)), "with a loss card"),
    ]
    if hottest_row is not None:
        tiles.append(("Hottest junction", f"{hottest_row['tj_max']:.1f} °C", hottest_row["id"]))
    violated = [row["id"] for row in rows if row["violated"]]
    tiles.append(("Card validity", "violated" if violated else "within", ", ".join(violated) or "Tj below every card limit"))
    tiles_html = "".join(
        f'<div class="card"><small>{_escape(label)}</small><strong>{_escape(value)}</strong>'
        f'<br><code>{_escape(hint)}</code></div>'
        for label, value, hint in tiles
    )

    def cell(value: float | None, unit: str) -> str:
        return "—" if value is None else f"{value:.3g} {unit}"

    table_rows = "".join(
        "<tr>"
        f"<td><code>{_escape(row['id'])}</code></td><td>{_escape(_display_value(row['kind']))}</td>"
        f"<td>{_escape(_display_value(row['card']))}</td>"
        f'<td class="number">{cell(row["conduction"], "W")}</td><td class="number">{cell(row["switching"], "W")}</td>'
        f'<td class="number">{cell(row["total"], "W")}</td><td class="number">{cell(row["tj_max"], "°C")}</td>'
        f'<td class="number">{cell(row["tj_final"], "°C")}</td></tr>'
        for row in rows
    )
    notes = sorted({str(item) for item in losses.get("warnings") or []})
    notes_html = (
        "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in notes) + "</ul>" if notes else ""
    )
    return f"""
  <section>
    <h2>Losses and junction temperatures</h2>
    <p class="notice"><strong>Claim:</strong> {_escape(_display_value(losses.get("claim")))}</p>
    <p class="subtitle">Every number is copied from <code>losses-thermal.json</code> (schema <code>{_escape(_display_value(losses.get("schema")))}</code>); the report draws and sorts, it does not recompute.</p>
    <div class="cards">{tiles_html}</div>
    {bars_html}
    {envelope_html}
    {notes_html}
    <details><summary>Every device ({len(rows)})</summary>
    <div class="scroll"><table><thead><tr><th>Device</th><th>Kind</th><th>Card</th><th>Conduction</th><th>Switching</th><th>Total</th><th>Tj max</th><th>Tj final</th></tr></thead><tbody>{table_rows}</tbody></table></div>
    </details>
  </section>"""


def _physics_audit_section(physics_audit: Any) -> str:
    if not isinstance(physics_audit, dict):
        return ""
    sign_convention = physics_audit.get("power_sign_convention")
    if sign_convention == "positive_absorbed_from_p_to_n":
        sign_explanation = (
            "Positive energy is absorbed by a component under the p→n current and "
            "V(p)-V(n) voltage convention; a negative value is net delivered energy."
        )
    else:
        sign_explanation = "Use the artifact-declared sign convention when interpreting energy."
    return f"""
  <section>
    <h2>Physics audit</h2>
    <p class="notice physics-notice"><strong>Absorbed-energy sign convention:</strong> <code>{_escape(_display_value(sign_convention))}</code>. {_escape(sign_explanation)}</p>
    <p class="subtitle">Schema <code>{_escape(_display_value(physics_audit.get("schema")))}</code>; method <code>{_escape(_display_value(physics_audit.get("method")))}</code>. Every number below is copied from <code>physics_audit</code>; the report does not recompute balances, negate source energy, or infer component roles.</p>
    {_physics_window_section("Full interval", "full", physics_audit.get("full"))}
    {_physics_window_section("Last half", "last_half", physics_audit.get("last_half"))}
  </section>"""


def _physics_window_section(title: str, key: str, window_value: Any) -> str:
    if not isinstance(window_value, dict):
        return (
            f"<h3>{_escape(title)} (<code>{_escape(key)}</code>)</h3>"
            '<p class="empty">This physics-audit window was not recorded.</p>'
        )

    if "load_absorbed_energy_j" in window_value:
        load_label = "Load absorbed energy [J]"
        load_field = "load_absorbed_energy_j"
    elif "resistor_absorbed_energy_j" in window_value:
        load_label = "Resistor/load absorbed energy [J]"
        load_field = "resistor_absorbed_energy_j"
    else:
        load_label = "Load absorbed energy [J]"
        load_field = None
    load_value = window_value.get(load_field) if load_field is not None else None
    source_present = "source_delivered_energy_j" in window_value
    source_value = window_value.get("source_delivered_energy_j") if source_present else None
    balance_present = "balance_residual_j" in window_value
    balance_value = window_value.get("balance_residual_j") if balance_present else None

    component_value = window_value.get("component_absorbed_energy_j")
    component_energy = component_value if isinstance(component_value, dict) else {}
    entries = [
        (str(raw_id), raw_value) for raw_id, raw_value in component_energy.items()
    ]
    # Ideal switches and diodes absorb exactly zero by construction (a closed
    # device has no voltage across it, an open one carries no current), so an
    # alphabetical dump buries the real numbers under dozens of zeros. Order
    # by magnitude and state why the zeros are there, instead of hiding them.
    def _magnitude(item: tuple[str, Any]) -> tuple[float, str]:
        value = item[1]
        size = abs(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0
        return (-size, item[0])

    ideal_zero = [
        component_id
        for component_id, value in entries
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == 0.0
    ]
    component_rows = "".join(
        "<tr>"
        f"<td><code>{_escape(component_id)}</code></td>"
        f'<td class="number">{_escape(_display_value(value))}</td>'
        "</tr>"
        for component_id, value in sorted(entries, key=_magnitude)
        if component_id not in set(ideal_zero)
    )
    zero_note = (
        f'<p class="note">{len(ideal_zero)} component(s) absorbed exactly 0 J: '
        "in the ideal-switch paradigm a conducting device has no voltage across "
        "it and a blocking device carries no current, so its absorbed power is "
        "identically zero. That is an invariant of the model, not a loss figure "
        "— semiconductor losses come from device cards (losses-thermal.json), "
        "never from this audit.</p>"
        f"<details><summary>The {len(ideal_zero)} zero-energy components</summary>"
        f"<p class=\"note\"><code>{_escape(', '.join(sorted(ideal_zero)))}</code></p></details>"
        if ideal_zero
        else ""
    )
    shown = len(entries) - len(ideal_zero)
    components_html = (
        f"<details><summary>{shown} component(s) with non-zero absorbed energy</summary>"
        '<div class="scroll"><table><thead><tr><th>Component</th>'
        "<th>Absorbed energy [J]</th></tr></thead><tbody>"
        + component_rows
        + "</tbody></table></div></details>"
        + zero_note
        if component_rows or ideal_zero
        else '<p class="empty">No per-component absorbed-energy map was recorded.</p>'
    )

    detail_fields = (
        "window_start_s",
        "window_end_s",
        "duration_s",
        "quadrature_intervals",
        "storage_energy_start_j",
        "storage_energy_end_j",
        "storage_energy_change_j",
        "delta_storage_energy_j",
        "integrated_storage_power_j",
        "storage_constitutive_residual_j",
        "source_absorbed_energy_j",
        "source_delivered_energy_j",
        "source_delivered_mean_power_w",
        "load_absorbed_energy_j",
        "load_absorbed_mean_power_w",
        "resistor_absorbed_energy_j",
        "resistor_absorbed_mean_power_w",
        "non_storage_absorbed_energy_j",
        "network_power_residual_j",
        "balance_residual_j",
        "balance_residual_relative",
        "balance_tolerance_j",
        "balance_tolerance_relative",
        "balance_within_tolerance",
        "balance_status",
    )
    detail_rows = "".join(
        "<tr>"
        f"<td><code>{_escape(field)}</code></td>"
        f'<td class="number">{_escape(_display_value(window_value.get(field)))}</td>'
        "</tr>"
        for field in detail_fields
        if field in window_value
    )
    source_note = (
        "artifact field source_delivered_energy_j"
        if source_present
        else "not recorded as a dedicated field"
    )
    load_note = (
        f"artifact field {load_field}"
        if load_field is not None
        else "not recorded as a dedicated field"
    )
    balance_note = (
        "artifact field balance_residual_j"
        if balance_present
        else "not recorded as a dedicated field"
    )
    return f"""
    <h3>{_escape(title)} (<code>{_escape(key)}</code>)</h3>
    <div class="cards">
      <div class="card"><small>Window [s]</small><strong>{_escape(_display_value(window_value.get("window_start_s")))} → {_escape(_display_value(window_value.get("window_end_s")))}</strong><br><code>duration={_escape(_display_value(window_value.get("duration_s")))}</code></div>
      <div class="card"><small>Source delivered energy [J]</small><strong>{_escape(_display_value(source_value))}</strong><br><code>{_escape(source_note)}</code></div>
      <div class="card"><small>{_escape(load_label)}</small><strong>{_escape(_display_value(load_value))}</strong><br><code>{_escape(load_note)}</code></div>
      <div class="card"><small>Balance residual [J]</small><strong>{_escape(_display_value(balance_value))}</strong><br><code>{_escape(balance_note)}</code></div>
      <div class="card"><small>Balance status</small><strong>{_escape(_display_value(window_value.get("balance_status")))}</strong><br><code>within tolerance={_escape(_display_value(window_value.get("balance_within_tolerance")))}</code></div>
    </div>
    <h3>Component absorbed energy — {_escape(title.lower())} (largest first)</h3>
    {components_html}
    <details><summary>Recorded physics fields — {_escape(title.lower())}</summary><div class="scroll"><table><thead><tr><th>Field</th><th>Recorded value</th></tr></thead><tbody>{detail_rows}</tbody></table></div></details>"""


def _recorded_measurement_html(value: Any) -> str:
    if not isinstance(value, dict):
        return _escape(_display_value(value))
    primary = _value_and_unit(value.get("value"), value.get("unit"))
    signal = value.get("signal")
    metric = value.get("metric")
    source = ""
    if signal is not None or metric is not None:
        source = (
            f"<br><code>{_escape(_display_value(signal))}.{_escape(_display_value(metric))}</code>"
        )
    return f'<span class="number">{_escape(primary)}</span>{source}'


def _analytic_tolerance_html(value: Any) -> str:
    if not isinstance(value, dict):
        return f"<code>{_escape(_display_value(value))}</code>"
    unit = value.get("unit")
    parts: list[str] = []
    if "absolute" in value:
        parts.append(f"absolute={_value_and_unit(value.get('absolute'), unit)}")
    if "relative" in value:
        parts.append(f"relative={_relative_percent(value.get('relative'))}")
    if "effective_absolute" in value:
        parts.append(f"effective={_value_and_unit(value.get('effective_absolute'), unit)}")
    if "comparison" in value:
        parts.append(str(value.get("comparison")))
    if not parts:
        parts.append(_display_value(value))
    return "<code>" + "<br>".join(_escape(part) for part in parts) + "</code>"


def _value_and_unit(value: Any, unit: Any) -> str:
    rendered = _display_value(value)
    if value is None or unit in (None, ""):
        return rendered
    return f"{rendered} {unit}"


def _relative_percent(value: Any) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value) * 100.0:.9g}%"
    return _display_value(value)


def _events_section(events: dict[str, Any]) -> str:
    count = int(events["count"])
    source = events.get("source") or "not present"
    time_summary = events["time"]
    limits = events["limits"]
    kinds = events["by_kind"]
    kind_summary = (
        ", ".join(f"{name}: {value}" for name, value in kinds.items()) if kinds else "none"
    )
    monotonic = time_summary.get("monotonic")
    monotonic_text = "not assessed" if monotonic is None else ("yes" if monotonic else "no")
    cards = f"""
    <div class="cards">
      <div class="card"><small>Events</small><strong>{count}</strong></div>
      <div class="card"><small>Time range [s]</small><code>{_escape(_display_value(time_summary.get("start_s")))} → {_escape(_display_value(time_summary.get("stop_s")))}</code></div>
      <div class="card"><small>Monotonic time</small><strong>{_escape(monotonic_text)}</strong><br><code>{_escape(time_summary.get("numeric_count"))}/{count} numeric</code></div>
      <div class="card"><small>Kinds</small><code>{_escape(kind_summary)}</code></div>
    </div>
    """
    limit_note = (
        f"Source: {source}. Safety limits: {limits['max_rows']} rows, "
        f"{limits['max_line_bytes']} bytes per row, {limits['max_file_bytes']} bytes per file."
    )
    if not events["items"]:
        return (
            f'<p class="subtitle">{_escape(limit_note)}</p>'
            + cards
            + '<p class="empty">No events were recorded.</p>'
        )

    maximum = int(limits["max_html_rows"])
    rendered_items = events["items"][:maximum]
    known_fields = {"time", "kind", "target", "previous", "current"}
    rows: list[str] = []
    for index, event in enumerate(rendered_items, start=1):
        transition = "—"
        if "previous" in event or "current" in event:
            transition = (
                f"{_display_value(event.get('previous'))} → {_display_value(event.get('current'))}"
            )
        details = {key: value for key, value in event.items() if key not in known_fields}
        rows.append(
            "<tr>"
            f'<td class="number">{index}</td>'
            f'<td class="number">{_escape(_display_value(event.get("time")))}</td>'
            f"<td>{_escape(_display_value(event.get('kind')))}</td>"
            f"<td>{_escape(_display_value(event.get('target')))}</td>"
            f"<td>{_escape(transition)}</td>"
            f"<td><code>{_escape(_display_value(details)) if details else '—'}</code></td>"
            "</tr>"
        )
    truncation = ""
    if count > len(rendered_items):
        truncation = (
            f'<p class="notice">HTML shows the first {len(rendered_items)} of {count} '
            "events. report.json retains the complete accepted list.</p>"
        )
    table = (
        '<details class="event-details"><summary>Show recorded event table '
        f'({len(rendered_items)} row(s))</summary><div class="scroll"><table>'
        "<thead><tr><th>#</th><th>Time [s]</th>"
        "<th>Kind</th><th>Target</th><th>Transition</th><th>Other fields</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></details>"
    )
    return f'<p class="subtitle">{_escape(limit_note)}</p>{cards}{truncation}{table}'


_COMPACT_METRICS = (
    "unit",
    "mean_last_half",
    "rms_last_half",
    "min_last_half",
    "max_last_half",
    "peak_to_peak_last_half",
)


def _metrics_table(metrics: dict[str, Any], *, compact: bool = False) -> str:
    if compact:
        if not metrics:
            return ""
        header = "<tr><th>Signal</th>" + "".join(
            f"<th>{_escape(column.replace('_last_half', '').replace('_', ' '))}</th>"
            for column in _COMPACT_METRICS
        ) + "</tr>"
        body = "".join(
            f"<tr><td><code>{_escape(signal)}</code></td>"
            + "".join(
                f'<td class="number">{_escape(_short_number(values.get(column)))}</td>'
                for column in _COMPACT_METRICS
            )
            + "</tr>"
            for signal, values in sorted(metrics.items())
            if isinstance(values, dict)
        )
        return (
            '<p class="subtitle">Statistics over the last half of the run (last-half mean, '
            "RMS, extrema, peak-to-peak), five significant digits. Full precision and every "
            "other metric are in the table below.</p>"
            f'<div class="scroll"><table><thead>{header}</thead><tbody>{body}</tbody></table></div>'
        )
    rows: dict[str, dict[str, Any]] = {}
    columns: set[str] = set()
    for signal, values in metrics.items():
        if isinstance(values, dict):
            row = {str(name): value for name, value in values.items()}
        else:
            row = {"value": values}
        rows[str(signal)] = row
        columns.update(row)
    ordered_columns = sorted(columns)
    header = (
        "<tr><th>Signal</th>"
        + "".join(f"<th>{_escape(column)}</th>" for column in ordered_columns)
        + "</tr>"
    )
    body = "".join(
        "<tr>"
        f"<td>{_escape(signal)}</td>"
        + "".join(
            f'<td class="number">{_escape(_display_value(values.get(column)))}</td>'
            for column in ordered_columns
        )
        + "</tr>"
        for signal, values in sorted(rows.items())
    )
    if not rows:
        body = '<tr><td colspan="1" class="empty">No metrics were recorded.</td></tr>'
    return f"<table><thead>{header}</thead><tbody>{body}</tbody></table>"


def _quantity(value: Any, unit: str) -> str:
    """A header figure: three significant digits for floats, grouped integers."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—" if value is None else _display_value(value)
    if isinstance(value, int):
        text = f"{value:,}".replace(",", "\u2009")
    elif not math.isfinite(value):
        text = _display_value(value)
    else:
        text = f"{value:.3g}"
    return f"{text} {unit}".strip()


def _status_pill(value: Any) -> str:
    text = _display_value(value)
    tone = {"pass": "good", "passed": "good", "fail": "error", "failed": "error"}.get(str(value), "")
    return f'<span class="pill {tone}">{_escape(text)}</span>'


def _short_number(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _display_value(value) if value is not None else "—"
    if not math.isfinite(value):
        return _display_value(value)
    return f"{value:.5g}"


_SEMICONDUCTOR_TYPES = frozenset({"ideal_switch", "diode", "thyristor"})


def _device_current_signals(resolved_model: Any, signals: Any) -> set[str]:
    """Currents of individual semiconductors: collapsed by default, they can be dozens."""

    physics = resolved_model.get("physics") if isinstance(resolved_model, dict) else None
    components = physics.get("components") if isinstance(physics, dict) else None
    if not isinstance(components, list):
        return set()
    devices = {
        str(item.get("id"))
        for item in components
        if isinstance(item, dict) and item.get("type") in _SEMICONDUCTOR_TYPES
    }
    return {f"i({device})" for device in devices} & set(signals)


_PLOT_POINTS = 800
_ZOOM_SAMPLES = _PLOT_POINTS // 4


def _zoom_window(time: Sequence[float], signals: dict[str, Any]) -> tuple[int, int] | None:
    """One shared raw-sample window drawn undecimated under every decimated overview.

    It spans ``_ZOOM_SAMPLES`` sample steps (few enough that each switching pulse is
    several pixels wide). Its place: the window in the second half of the run (closest
    to steady state) where the first signal varies most — on a PWM node that is
    where the duty cycle is near one half and both edges of every pulse are clear.
    A peak or an edge of the run would land where pulses are narrowest. Shared, so
    voltages and currents are read at the same instant like on a scope. ``None``
    when the overview already draws every sample.
    """

    length = len(time)
    if length <= _PLOT_POINTS or not signals:
        return None
    first_signal = np.asarray(next(iter(signals.values())), dtype=float)
    half = length // 2
    count = (length - half) // _ZOOM_SAMPLES
    if count < 1:
        first = length - 1 - _ZOOM_SAMPLES
    else:
        windows = first_signal[half : half + count * _ZOOM_SAMPLES].reshape(count, _ZOOM_SAMPLES)
        first = half + int(np.argmax(windows.std(axis=1))) * _ZOOM_SAMPLES
    # Both ends inclusive: on a uniform grid the window spans a round number of steps.
    first = min(first, length - 1 - _ZOOM_SAMPLES)
    return first, first + _ZOOM_SAMPLES


def _format_si(value: float, unit: str) -> str:
    magnitude = abs(value)
    for scale, prefix in ((1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n")):
        if magnitude >= scale or scale == 1e-9:
            return f"{value / scale:.6g} {prefix}{unit}"
    return f"{value:g} {unit}"


def _findings_section(warnings: list[Any]) -> tuple[str, int, int]:
    """Warnings first, model notes (severity info) after, as readable text."""

    def parts(value: Any) -> tuple[str, str, str]:
        if isinstance(value, dict):
            return (
                str(value.get("severity") or "warning"),
                str(value.get("code") or ""),
                str(value.get("message") or _display_value(value)),
            )
        return ("warning", "", _display_value(value))

    items = [parts(value) for value in warnings]
    serious = [item for item in items if item[0] != "info"]
    notes = [item for item in items if item[0] == "info"]

    def listing(entries: list[tuple[str, str, str]]) -> str:
        return '<ul class="findings">' + "".join(
            f'<li><span class="pill {_escape(severity)}">{_escape(severity)}</span>'
            + (f'<span class="code">{_escape(code)}</span>' if code else "")
            + f'<span class="text">{_escape(message)}</span></li>'
            for severity, code, message in entries
        ) + "</ul>"

    body = (
        listing(serious)
        if serious
        else '<p class="empty">No warnings were recorded in the run artifacts.</p>'
    )
    if notes:
        body += (
            "<h3>Notes</h3><p class=\"subtitle\">Information, not faults: what the model "
            "does not represent and how the run handled declared behaviour.</p>"
            + listing(notes)
        )
    return f"<section>\n    <h2>Warnings</h2>\n    {body}\n  </section>", len(serious), len(notes)


def _display_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.9g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)
