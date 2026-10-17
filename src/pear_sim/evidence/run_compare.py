"""Fail-closed comparison of two verified, immutable PEARSim run bundles.

The comparator is deliberately narrower than a generic waveform-diff utility.  It
accepts only run bundles whose root manifest verifies every declared artifact, and
it never guesses signal mappings, polarity changes, gains, or event identities.
Numerical equivalence can therefore be reported only for the same resolved
physical model and the same measurement contract.
"""

from __future__ import annotations

from bisect import bisect_right
import csv
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Literal, Mapping, Sequence

from pear_sim import __version__
from pear_sim._jsonutil import (
    reject_json_constant as _reject_json_constant,
    require_finite as _require_finite,
    unique_json_object as _unique_json_object,
)
from pear_sim.errors import ModelError
from pear_sim.run.contracts import sha256_file, sha256_json
from pear_sim.spec.signals import signal_unit
from pear_sim.limits import MAX_RUN_EVENT_ROWS


COMPARE_PROFILE_SCHEMA = "pear-sim/compare-profile/v0.1"
RUN_COMPARISON_SCHEMA = "pear-sim/run-comparison/v0.1"
RUN_COMPARISON_BUNDLE_SCHEMA = "pear-sim/run-comparison-bundle/v0.1"
SUPPORTED_RUN_SCHEMA = "pear-sim/run/v0.1"
SUPPORTED_RESOLVED_MODEL_SCHEMA = "pear-sim/resolved-model/v0.1"
SUPPORTED_BACKEND = "reference-python"

MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_WAVEFORM_BYTES = 256 * 1024 * 1024
MAX_WAVEFORM_ROWS = 1_000_000
MAX_EVENT_BYTES = 64 * 1024 * 1024
MAX_EVENT_LINE_BYTES = 64 * 1024
MAX_EVENT_ROWS = MAX_RUN_EVENT_ROWS
_SIGNAL_EXPRESSION_RE = re.compile(r"^(v|i)\([A-Za-z_][A-Za-z0-9_.:-]*\)$")

Status = Literal["pass", "fail", "not_comparable"]


class RunComparisonError(ValueError):
    """Raised when comparison inputs violate the trusted-artifact contract."""

    code = "compare.invalid"


@dataclass(frozen=True)
class RunComparisonArtifacts:
    output_dir: Path
    resolved_profile: Path
    comparison_json: Path
    report_html: Path
    artifact_manifest: Path
    status: Status


@dataclass(frozen=True)
class _Waveforms:
    time: tuple[float, ...]
    signals: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class _VerifiedRun:
    directory: Path
    run: dict[str, Any]
    resolved: dict[str, Any]
    waveforms: _Waveforms
    events: tuple[dict[str, Any], ...]
    run_manifest_sha256: str
    artifact_manifest_sha256: str
    artifact_manifest_bytes: int
    bundle_sha256: str
    topology: dict[str, Any]
    topology_sha256: str
    conventions_sha256: str


@dataclass(frozen=True)
class _EventGroup:
    time: float
    semantics: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _Series:
    time: tuple[float, ...]
    value: tuple[float, ...]


def compare_runs(
    reference_run: str | Path,
    candidate_run: str | Path,
    profile_file: str | Path | Mapping[str, Any],
    output_dir: str | Path,
) -> RunComparisonArtifacts:
    """Compare two verified run bundles and retain an offline evidence bundle.

    Input corruption raises :class:`RunComparisonError` before an output directory
    is created.  A valid but physically incompatible pair produces retained
    ``not_comparable`` evidence.  Existing output content is never overwritten.
    """

    reference = _read_verified_run(reference_run, "reference")
    candidate = _read_verified_run(candidate_run, "candidate")
    profile, profile_source = _read_profile(profile_file)
    resolved_profile = _resolve_profile(profile, profile_source)
    comparison = _compare(reference, candidate, resolved_profile)

    destination = _prepare_destination(output_dir, (reference.directory, candidate.directory))
    profile_path = destination / "compare-profile.resolved.json"
    comparison_path = destination / "comparison.json"
    report_path = destination / "report.html"
    manifest_path = destination / "artifact-manifest.json"
    _write_json(profile_path, resolved_profile)
    _write_json(comparison_path, comparison)
    _write_text(report_path, _render_report(comparison))

    artifact_records = [
        {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted((profile_path, comparison_path, report_path))
    ]
    _write_json(
        manifest_path,
        {
            "schema": "pear-sim/artifact-manifest/v0.1",
            "bundle_schema": RUN_COMPARISON_BUNDLE_SCHEMA,
            "subject": {
                "kind": "run-comparison",
                "sha256": sha256_json(comparison),
                "status": comparison["status"],
            },
            "sources": {
                "reference_bundle_sha256": reference.bundle_sha256,
                "candidate_bundle_sha256": candidate.bundle_sha256,
            },
            "artifacts": artifact_records,
        },
    )
    return RunComparisonArtifacts(
        output_dir=destination,
        resolved_profile=profile_path,
        comparison_json=comparison_path,
        report_html=report_path,
        artifact_manifest=manifest_path,
        status=comparison["status"],
    )


def _read_profile(
    source: str | Path | Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(source, Mapping):
        profile = _strict_json_copy(source, "compare profile")
        source_record = {"kind": "inline", "sha256": sha256_json(profile)}
    else:
        path = Path(source).resolve()
        profile = _read_json(path, "compare profile", MAX_JSON_BYTES)
        source_record = {
            "kind": "file",
            "file": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    _validate_profile(profile)
    return profile, source_record


def _validate_profile(profile: dict[str, Any]) -> None:
    allowed = {
        "schema",
        "id",
        "alignment",
        "window",
        "edge_exclusion",
        "coverage",
        "signals",
    }
    _require_keys(profile, allowed, allowed - {"window"}, "compare profile")
    if profile["schema"] != COMPARE_PROFILE_SCHEMA:
        raise RunComparisonError(f"compare profile schema must be {COMPARE_PROFILE_SCHEMA!r}")
    if not isinstance(profile["id"], str) or not profile["id"] or len(profile["id"]) > 256:
        raise RunComparisonError(
            "compare profile id must be a non-empty string of at most 256 chars"
        )

    alignment = _object(profile["alignment"], "alignment")
    _require_keys(
        alignment,
        {"mode", "event_time_tolerance_s"},
        {"mode", "event_time_tolerance_s"},
        "alignment",
    )
    if alignment["mode"] not in {"absolute", "matching_events_piecewise"}:
        raise RunComparisonError("alignment.mode must be absolute or matching_events_piecewise")
    _nonnegative_finite(alignment["event_time_tolerance_s"], "event_time_tolerance_s")

    if "window" in profile:
        window = _object(profile["window"], "window")
        _require_keys(window, {"start_s", "stop_s"}, set(), "window")
        if not window:
            raise RunComparisonError("window must declare start_s or stop_s when present")
        for key in ("start_s", "stop_s"):
            if key in window:
                _finite_number(window[key], f"window.{key}")
        if (
            "start_s" in window
            and "stop_s" in window
            and float(window["stop_s"]) <= float(window["start_s"])
        ):
            raise RunComparisonError("window.stop_s must be greater than window.start_s")

    edge = _object(profile["edge_exclusion"], "edge_exclusion")
    _require_keys(edge, {"before_s", "after_s"}, {"before_s", "after_s"}, "edge_exclusion")
    _nonnegative_finite(edge["before_s"], "edge_exclusion.before_s")
    _nonnegative_finite(edge["after_s"], "edge_exclusion.after_s")

    coverage = _object(profile["coverage"], "coverage")
    _require_keys(coverage, {"minimum_fraction"}, {"minimum_fraction"}, "coverage")
    minimum = _finite_number(coverage["minimum_fraction"], "coverage.minimum_fraction")
    if not 0.0 < minimum <= 1.0:
        raise RunComparisonError("coverage.minimum_fraction must be in (0, 1]")

    signals = profile["signals"]
    if not isinstance(signals, list) or not signals or len(signals) > 256:
        raise RunComparisonError("signals must be an array containing 1..256 records")
    names: set[str] = set()
    for index, value in enumerate(signals):
        signal = _object(value, f"signals[{index}]")
        _require_keys(
            signal,
            {"name", "unit", "tolerances"},
            {"name", "unit", "tolerances"},
            f"signals[{index}]",
        )
        name = signal["name"]
        if not isinstance(name, str) or not name or len(name) > 256:
            raise RunComparisonError(f"signals[{index}].name must be a non-empty string")
        if name in names:
            raise RunComparisonError(f"signals repeats {name!r}")
        names.add(name)
        expected_unit = _signal_unit(name)
        if signal["unit"] != expected_unit:
            raise RunComparisonError(
                f"signals[{index}].unit must be {expected_unit!r} for {name!r}"
            )
        tolerances = _object(signal["tolerances"], f"signals[{index}].tolerances")
        _require_keys(
            tolerances,
            {"max_absolute_error", "mean_absolute_error", "nrmse"},
            {"max_absolute_error", "mean_absolute_error", "nrmse"},
            f"signals[{index}].tolerances",
        )
        for metric in ("max_absolute_error", "mean_absolute_error"):
            tolerance = _object(tolerances[metric], f"signals[{index}].tolerances.{metric}")
            _require_keys(
                tolerance,
                {"absolute", "relative"},
                {"absolute", "relative"},
                f"signals[{index}].tolerances.{metric}",
            )
            _nonnegative_finite(tolerance["absolute"], f"{name}.{metric}.absolute")
            _nonnegative_finite(tolerance["relative"], f"{name}.{metric}.relative")
        _nonnegative_finite(tolerances["nrmse"], f"{name}.nrmse")


def _resolve_profile(profile: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    window = profile.get("window", {})
    payload: dict[str, Any] = {
        "schema": COMPARE_PROFILE_SCHEMA,
        "id": profile["id"],
        "alignment": {
            "mode": profile["alignment"]["mode"],
            "event_time_tolerance_s": float(profile["alignment"]["event_time_tolerance_s"]),
        },
        "window": {
            "start_s": float(window["start_s"]) if "start_s" in window else None,
            "stop_s": float(window["stop_s"]) if "stop_s" in window else None,
            "selection": "declared_or_full_reference_window_without_silent_clipping",
        },
        "edge_exclusion": {
            "before_s": float(profile["edge_exclusion"]["before_s"]),
            "after_s": float(profile["edge_exclusion"]["after_s"]),
            "sample_barrier_policy": "exclude_adjacent_sample_span_at_every_event",
        },
        "coverage": {
            "minimum_fraction": float(profile["coverage"]["minimum_fraction"]),
            "denominator": "selected_reference_time_window",
        },
        "signals": [],
        "source": source,
    }
    for signal in profile["signals"]:
        tolerances = signal["tolerances"]
        payload["signals"].append(
            {
                "name": signal["name"],
                "unit": signal["unit"],
                "tolerances": {
                    metric: {
                        "absolute": float(tolerances[metric]["absolute"]),
                        "relative": float(tolerances[metric]["relative"]),
                    }
                    for metric in ("max_absolute_error", "mean_absolute_error")
                }
                | {"nrmse": float(tolerances["nrmse"])},
            }
        )
    semantic = {key: value for key, value in payload.items() if key != "source"}
    payload["sha256"] = sha256_json(semantic)
    return payload


def _read_verified_run(source: str | Path, role: str) -> _VerifiedRun:
    raw_directory = Path(source).absolute()
    try:
        root_stat = raw_directory.lstat()
    except OSError as exc:
        raise RunComparisonError(f"cannot inspect {role} run directory: {exc}") from exc
    if stat.S_ISLNK(root_stat.st_mode):
        raise RunComparisonError(f"{role} run directory must not be a symlink")
    if not stat.S_ISDIR(root_stat.st_mode):
        raise RunComparisonError(f"{role} run directory does not exist or is not a directory")
    directory = raw_directory.resolve()
    run_path = _safe_member(directory, "run.json", f"{role} run manifest")
    manifest_path = _safe_member(directory, "artifact-manifest.json", f"{role} artifact manifest")
    run = _read_json(run_path, f"{role} run manifest", MAX_JSON_BYTES)
    if run.get("schema") != SUPPORTED_RUN_SCHEMA:
        raise RunComparisonError(f"{role} run schema must be {SUPPORTED_RUN_SCHEMA!r}")
    if run.get("backend") != SUPPORTED_BACKEND:
        raise RunComparisonError(f"{role} run backend must be {SUPPORTED_BACKEND!r}")
    manifest = _read_json(manifest_path, f"{role} artifact manifest", MAX_JSON_BYTES)
    artifact_records = _verify_manifest(directory, run, manifest, role)

    files = _object(run.get("files"), f"{role} run files")
    for required in ("resolved_model", "waveforms", "events"):
        if required not in files:
            raise RunComparisonError(f"{role} run does not declare required {required!r} artifact")
    resolved_path = _safe_member(directory, files["resolved_model"], f"{role} resolved model")
    waveform_path = _safe_member(directory, files["waveforms"], f"{role} waveforms")
    events_path = _safe_member(directory, files["events"], f"{role} events")
    resolved = _read_json(resolved_path, f"{role} resolved model", MAX_JSON_BYTES)
    _verify_semantic_hashes(run, resolved, role)
    waveforms = _read_waveforms(waveform_path, role)
    events = _read_events(events_path, role, waveforms.time[0], waveforms.time[-1])

    outputs = run.get("outputs")
    request = _object(resolved.get("request"), f"{role} resolved request")
    if request.get("schema") != "pear-sim/simulation-request/v0.1":
        raise RunComparisonError(
            f"{role} resolved request schema must be 'pear-sim/simulation-request/v0.1'"
        )
    if request.get("analysis") != "transient":
        raise RunComparisonError(f"{role} resolved request analysis must be 'transient'")
    request_start = _finite_number(request.get("start_s"), f"{role} request.start_s")
    request_stop = _finite_number(request.get("stop_s"), f"{role} request.stop_s")
    if request_stop <= request_start:
        raise RunComparisonError(f"{role} resolved request stop_s must exceed start_s")
    output_step = _finite_number(request.get("output_step_s"), f"{role} request.output_step_s")
    if output_step <= 0.0 or output_step > request_stop - request_start:
        raise RunComparisonError(
            f"{role} resolved request output_step_s must be positive and within the span"
        )
    if waveforms.time[0] != request_start or waveforms.time[-1] != request_stop:
        raise RunComparisonError(
            f"{role} waveform domain does not match the hash-bound resolved request"
        )
    try:
        grid_ratio = (request_stop - request_start) / output_step
    except (OverflowError, ZeroDivisionError) as exc:
        raise RunComparisonError(f"{role} request output grid is outside the bounded contract") from exc
    if not math.isfinite(grid_ratio) or grid_ratio > MAX_WAVEFORM_ROWS:
        raise RunComparisonError(
            f"{role} request output grid exceeds the {MAX_WAVEFORM_ROWS}-row contract limit"
        )
    count = int(math.floor(grid_ratio + 1e-12))
    endpoint_tolerance = max(1e-15, (request_stop - request_start) * 1e-12)
    regular_last = request_start + count * output_step
    expected_length = count + 1 + int(regular_last < request_stop - endpoint_tolerance)
    if expected_length > MAX_WAVEFORM_ROWS or expected_length != len(waveforms.time):
        raise RunComparisonError(
            f"{role} waveform sample grid does not match hash-bound output_step_s"
        )
    for index, actual in enumerate(waveforms.time):
        expected = (
            request_stop
            if index == expected_length - 1
            else request_start + index * output_step
        )
        if not math.isclose(
            actual,
            expected,
            rel_tol=0.0,
            abs_tol=max(2.0 * math.ulp(expected), 1e-300),
        ):
            raise RunComparisonError(
                f"{role} waveform sample grid does not match hash-bound output_step_s"
            )
    stats = _object(run.get("stats"), f"{role} run stats")
    samples = stats.get("samples")
    if isinstance(samples, bool) or not isinstance(samples, int) or samples != len(waveforms.time):
        raise RunComparisonError(
            f"{role} run stats.samples does not match the waveform row count"
        )
    resolved_outputs = request.get("outputs")
    if (
        not isinstance(outputs, list)
        or not outputs
        or not all(isinstance(value, str) and value for value in outputs)
        or len(set(outputs)) != len(outputs)
    ):
        raise RunComparisonError(f"{role} run outputs must be unique non-empty strings")
    if outputs != resolved_outputs:
        raise RunComparisonError(f"{role} run outputs differ from resolved request outputs")
    if outputs != list(waveforms.signals):
        raise RunComparisonError(f"{role} run output order differs from waveforms.csv columns")
    for name in outputs:
        _signal_unit(name)

    physics = _object(resolved.get("physics"), f"{role} resolved physics")
    topology = _structural_topology(physics, role)
    conventions = _object(resolved.get("conventions"), f"{role} conventions")
    bundle_payload = {
        "artifact_manifest": {
            "sha256": sha256_file(manifest_path),
            "bytes": manifest_path.stat().st_size,
        },
        "artifacts": artifact_records,
    }
    return _VerifiedRun(
        directory=directory,
        run=run,
        resolved=resolved,
        waveforms=waveforms,
        events=events,
        run_manifest_sha256=sha256_file(run_path),
        artifact_manifest_sha256=sha256_file(manifest_path),
        artifact_manifest_bytes=manifest_path.stat().st_size,
        bundle_sha256=sha256_json(bundle_payload),
        topology=topology,
        topology_sha256=sha256_json(topology),
        conventions_sha256=sha256_json(conventions),
    )


def _verify_manifest(
    directory: Path,
    run: dict[str, Any],
    manifest: dict[str, Any],
    role: str,
) -> list[dict[str, Any]]:
    if manifest.get("schema") != "pear-sim/artifact-manifest/v0.1":
        raise RunComparisonError(f"{role} artifact manifest has an unsupported schema")
    subject = manifest.get("subject")
    run_sha256 = run.get("run_sha256")
    if not _is_sha256(run_sha256):
        raise RunComparisonError(f"{role} run_sha256 must be a lowercase SHA-256 digest")
    if (
        not isinstance(subject, dict)
        or subject.get("kind") != "run"
        or subject.get("sha256") != run_sha256
    ):
        raise RunComparisonError(f"{role} artifact manifest subject does not bind the run")
    files = _object(run.get("files"), f"{role} run files")
    expected = {"run.json"}
    for file_role, reference in files.items():
        path = _safe_member(directory, reference, f"{role} run file {file_role!r}")
        expected.add(path.relative_to(directory).as_posix())
    if len(expected) != len(files) + 1:
        raise RunComparisonError(f"{role} run file roles alias the same artifact")

    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise RunComparisonError(f"{role} artifact manifest artifacts must be an array")
    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(records):
        record = _object(raw, f"{role} artifact manifest record {index}")
        reference = record.get("path")
        path = _safe_member(directory, reference, f"{role} manifest artifact {index}")
        relative = path.relative_to(directory).as_posix()
        if relative in seen:
            raise RunComparisonError(f"{role} artifact manifest repeats {relative!r}")
        seen.add(relative)
        digest = record.get("sha256")
        byte_count = record.get("bytes")
        if not _is_sha256(digest) or digest != sha256_file(path):
            raise RunComparisonError(f"{role} artifact SHA-256 mismatch for {relative!r}")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int):
            raise RunComparisonError(f"{role} artifact byte count is invalid for {relative!r}")
        if byte_count != path.stat().st_size:
            raise RunComparisonError(f"{role} artifact byte-count mismatch for {relative!r}")
        verified.append({"path": relative, "sha256": digest, "bytes": byte_count})
    if seen != expected:
        missing = sorted(expected - seen)
        unexpected = sorted(seen - expected)
        raise RunComparisonError(
            f"{role} artifact manifest coverage mismatch; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    return sorted(verified, key=lambda item: item["path"])


def _verify_semantic_hashes(run: dict[str, Any], resolved: dict[str, Any], role: str) -> None:
    if resolved.get("schema") != SUPPORTED_RESOLVED_MODEL_SCHEMA:
        raise RunComparisonError(
            f"{role} resolved model schema must be {SUPPORTED_RESOLVED_MODEL_SCHEMA!r}"
        )
    hashes = _object(resolved.get("hashes"), f"{role} resolved hashes")
    for name in ("source_sha256", "model_sha256", "request_sha256"):
        if not _is_sha256(run.get(name)) or run.get(name) != hashes.get(name):
            raise RunComparisonError(f"{role} run and resolved {name} do not match")
    physics = _object(resolved.get("physics"), f"{role} resolved physics")
    request = _object(resolved.get("request"), f"{role} resolved request")
    if sha256_json(physics) != run["model_sha256"]:
        raise RunComparisonError(f"{role} resolved physics does not match model_sha256")
    if sha256_json(request) != run["request_sha256"]:
        raise RunComparisonError(f"{role} resolved request does not match request_sha256")
    expected_run_sha256 = sha256_json(
        {
            "model_sha256": run["model_sha256"],
            "request_sha256": run["request_sha256"],
            "backend": run.get("backend"),
            "pear_sim_version": run.get("pear_sim_version"),
        }
    )
    if run.get("run_sha256") != expected_run_sha256:
        raise RunComparisonError(f"{role} run_sha256 does not match its declared semantics")


def _structural_topology(physics: dict[str, Any], role: str) -> dict[str, Any]:
    ground = physics.get("ground")
    components = physics.get("components")
    controls = physics.get("controls")
    if not isinstance(ground, str) or not ground:
        raise RunComparisonError(f"{role} resolved physics ground is invalid")
    if not isinstance(components, list) or not components:
        raise RunComparisonError(f"{role} resolved physics components are invalid")
    if not isinstance(controls, list):
        raise RunComparisonError(f"{role} resolved physics controls are invalid")
    structural_components: list[dict[str, Any]] = []
    component_ids: set[str] = set()
    for index, raw in enumerate(components):
        component = _object(raw, f"{role} component {index}")
        required = ("id", "type", "p", "n")
        if any(not isinstance(component.get(key), str) or not component[key] for key in required):
            raise RunComparisonError(f"{role} component {index} has invalid structural fields")
        if component["id"] in component_ids:
            raise RunComparisonError(f"{role} resolved physics repeats component id")
        component_ids.add(component["id"])
        structural = {key: component[key] for key in required}
        if "control" in component:
            if not isinstance(component["control"], str) or not component["control"]:
                raise RunComparisonError(f"{role} component {index} control is invalid")
            structural["control"] = component["control"]
        structural_components.append(structural)
    structural_controls: list[dict[str, Any]] = []
    control_ids: set[str] = set()
    for index, raw in enumerate(controls):
        control = _object(raw, f"{role} control {index}")
        if (
            not isinstance(control.get("id"), str)
            or not control["id"]
            or not isinstance(control.get("type"), str)
            or not control["type"]
        ):
            raise RunComparisonError(f"{role} control {index} has invalid structural fields")
        if control["id"] in control_ids:
            raise RunComparisonError(f"{role} resolved physics repeats control id")
        control_ids.add(control["id"])
        structural_controls.append({"id": control["id"], "type": control["type"]})
    return {
        "schema": "pear-sim/structural-topology/v0.1",
        "ground": ground,
        "components": sorted(structural_components, key=lambda item: item["id"]),
        "controls": sorted(structural_controls, key=lambda item: item["id"]),
    }


def _read_waveforms(path: Path, role: str) -> _Waveforms:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise RunComparisonError(f"cannot inspect {role} waveforms: {exc}") from exc
    if size > MAX_WAVEFORM_BYTES:
        raise RunComparisonError(f"{role} waveforms exceed the byte limit")
    try:
        handle = path.open(newline="", encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise RunComparisonError(f"cannot read {role} waveforms: {exc}") from exc
    with handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader)
        except (StopIteration, csv.Error) as exc:
            raise RunComparisonError(f"{role} waveforms are empty or invalid CSV") from exc
        if len(header) < 2 or header[0] != "time" or len(set(header)) != len(header):
            raise RunComparisonError(f"{role} waveforms must have time and unique signal columns")
        times: list[float] = []
        signals: dict[str, list[float]] = {name: [] for name in header[1:]}
        try:
            for row_index, row in enumerate(reader, start=2):
                if row_index > MAX_WAVEFORM_ROWS + 1:
                    raise RunComparisonError(f"{role} waveforms exceed the row limit")
                if len(row) != len(header):
                    raise RunComparisonError(
                        f"{role} waveforms row {row_index} has the wrong field count"
                    )
                try:
                    values = [float(value) for value in row]
                except (OverflowError, ValueError) as exc:
                    raise RunComparisonError(
                        f"{role} waveforms row {row_index} is not numeric"
                    ) from exc
                if any(not math.isfinite(value) for value in values):
                    raise RunComparisonError(
                        f"{role} waveforms row {row_index} contains a non-finite value"
                    )
                if times and values[0] <= times[-1]:
                    raise RunComparisonError(f"{role} waveform time must be strictly increasing")
                times.append(values[0])
                for name, value in zip(header[1:], values[1:], strict=True):
                    signals[name].append(value)
        except csv.Error as exc:
            raise RunComparisonError(f"{role} waveforms contain invalid CSV: {exc}") from exc
    if len(times) < 2:
        raise RunComparisonError(f"{role} waveforms require at least two samples")
    return _Waveforms(
        time=tuple(times),
        signals={name: tuple(values) for name, values in signals.items()},
    )


def _read_events(
    path: Path, role: str, time_start: float, time_stop: float
) -> tuple[dict[str, Any], ...]:
    if path.stat().st_size > MAX_EVENT_BYTES:
        raise RunComparisonError(f"{role} events exceed the byte limit")
    events: list[dict[str, Any]] = []
    previous_time = -math.inf
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise RunComparisonError(f"cannot read {role} events: {exc}") from exc
    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if line_number > MAX_EVENT_ROWS:
                raise RunComparisonError(f"{role} events exceed the row limit")
            if len(raw_line) > MAX_EVENT_LINE_BYTES:
                raise RunComparisonError(f"{role} event line {line_number} exceeds the byte limit")
            if not raw_line.strip():
                raise RunComparisonError(f"{role} event line {line_number} is empty")
            try:
                event = _strict_json_loads(raw_line.decode("utf-8"))
            except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                raise RunComparisonError(
                    f"{role} event line {line_number} is invalid JSON: {exc}"
                ) from exc
            if not isinstance(event, dict):
                raise RunComparisonError(f"{role} event line {line_number} must be an object")
            event_time = _finite_number(event.get("time"), f"{role} event time")
            if event_time < previous_time:
                raise RunComparisonError(f"{role} event times must be monotonically non-decreasing")
            if event_time < time_start or event_time > time_stop:
                raise RunComparisonError(f"{role} event time lies outside the waveform domain")
            if (
                not isinstance(event.get("kind"), str)
                or not event["kind"]
                or not isinstance(event.get("target"), str)
                or not event["target"]
                or not isinstance(event.get("previous"), bool)
                or not isinstance(event.get("current"), bool)
            ):
                raise RunComparisonError(f"{role} event line {line_number} has invalid semantics")
            previous_time = event_time
            events.append(event)
    return tuple(events)


def _compare(
    reference: _VerifiedRun,
    candidate: _VerifiedRun,
    profile: dict[str, Any],
) -> dict[str, Any]:
    outputs_equal = list(reference.waveforms.signals) == list(candidate.waveforms.signals)
    conventions_equal = reference.conventions_sha256 == candidate.conventions_sha256
    topology_equal = reference.topology_sha256 == candidate.topology_sha256
    physics_equal = reference.run["model_sha256"] == candidate.run["model_sha256"]
    selected_names = [record["name"] for record in profile["signals"]]
    selected_available = all(name in reference.waveforms.signals for name in selected_names)
    compatibility_gates = [
        _exact_gate(
            "structural_topology",
            reference.topology_sha256,
            candidate.topology_sha256,
            topology_equal,
        ),
        _exact_gate(
            "resolved_physical_model",
            reference.run["model_sha256"],
            candidate.run["model_sha256"],
            physics_equal,
        ),
        _exact_gate(
            "signal_identity_and_order",
            list(reference.waveforms.signals),
            list(candidate.waveforms.signals),
            outputs_equal,
        ),
        _exact_gate(
            "measurement_polarity",
            reference.conventions_sha256,
            candidate.conventions_sha256,
            conventions_equal,
        ),
        _exact_gate(
            "profile_signal_availability",
            selected_names,
            [name for name in selected_names if name in reference.waveforms.signals],
            selected_available,
        ),
    ]
    compatible = all(gate["status"] == "pass" for gate in compatibility_gates)

    groups_reference = _group_events(reference.events)
    groups_candidate = _group_events(candidate.events)
    event_semantics_equal = _event_semantics(groups_reference) == _event_semantics(groups_candidate)
    compatibility_gates.append(
        _exact_gate(
            "event_semantics",
            _event_semantic_hash(groups_reference),
            _event_semantic_hash(groups_candidate),
            event_semantics_equal,
        )
    )
    compatible = compatible and event_semantics_equal

    event_tolerance = float(profile["alignment"]["event_time_tolerance_s"])
    event_deltas = (
        [
            candidate_group.time - reference_group.time
            for reference_group, candidate_group in zip(
                groups_reference, groups_candidate, strict=True
            )
        ]
        if event_semantics_equal
        else []
    )
    maximum_event_delta = max((abs(value) for value in event_deltas), default=0.0)
    alignment_mode = profile["alignment"]["mode"]
    piecewise_domain_equal = alignment_mode != "matching_events_piecewise" or (
        reference.waveforms.time[0] == candidate.waveforms.time[0]
        and reference.waveforms.time[-1] == candidate.waveforms.time[-1]
    )
    piecewise_anchor_available = alignment_mode != "matching_events_piecewise" or (
        piecewise_domain_equal
        and event_semantics_equal
        and any(
            reference.waveforms.time[0] < reference_group.time < reference.waveforms.time[-1]
            and candidate.waveforms.time[0]
            < candidate_group.time
            < candidate.waveforms.time[-1]
            for reference_group, candidate_group in zip(
                groups_reference,
                groups_candidate,
                strict=True,
            )
        )
    )
    event_timing_gate = {
        "id": "event_timing",
        "status": (
            "pass"
            if event_semantics_equal
            and piecewise_anchor_available
            and maximum_event_delta <= event_tolerance
            else "fail"
            if event_semantics_equal and piecewise_anchor_available
            else "not_comparable"
        ),
        "metric": "maximum_absolute_matched_event_time_delta_s",
        "actual": maximum_event_delta if event_semantics_equal else None,
        "limit": event_tolerance,
        "operator": "<=",
        "matched_group_count": len(groups_reference) if event_semantics_equal else 0,
    }

    alignment_record: dict[str, Any] = {
        "mode": alignment_mode,
        "event_groups": {
            "reference_count": len(groups_reference),
            "candidate_count": len(groups_candidate),
            "semantics_exact": event_semantics_equal,
            "maximum_absolute_time_delta_s": (
                maximum_event_delta if event_semantics_equal else None
            ),
        },
        "event_timing_gate": event_timing_gate,
        "window": None,
        "exclusions": None,
    }
    signal_results: list[dict[str, Any]] = []
    comparison_possible = compatible
    series_candidate_time: tuple[float, ...] | None = None
    mapped_candidate_events: list[float] = []
    window: tuple[float, float] | None = None
    exclusions: list[tuple[float, float]] = []
    domain_exclusions: list[tuple[float, float]] = []
    coverage = 0.0
    coverage_duration = 0.0

    if comparison_possible:
        mode = alignment_mode
        if mode == "absolute":
            series_candidate_time = candidate.waveforms.time
            mapped_candidate_events = [group.time for group in groups_candidate]
        else:
            warp = _event_warp(reference, candidate, groups_reference, groups_candidate)
            if warp is None:
                comparison_possible = False
            else:
                series_candidate_time = tuple(
                    _warp_time(value, warp) for value in candidate.waveforms.time
                )
                mapped_candidate_events = [
                    _warp_time(group.time, warp) for group in groups_candidate
                ]
        if comparison_possible and series_candidate_time is not None:
            declared_window = profile["window"]
            reference_start = reference.waveforms.time[0]
            reference_stop = reference.waveforms.time[-1]
            start = (
                float(declared_window["start_s"])
                if declared_window["start_s"] is not None
                else reference_start
            )
            stop = (
                float(declared_window["stop_s"])
                if declared_window["stop_s"] is not None
                else reference_stop
            )
            # A declared window is a request, not a hint to be silently clipped.
            if start < reference_start or stop > reference_stop or stop <= start:
                comparison_possible = False
            else:
                window = (start, stop)
                candidate_start = series_candidate_time[0]
                candidate_stop = series_candidate_time[-1]
                if candidate_start > start:
                    domain_exclusions.append((start, min(stop, candidate_start)))
                if candidate_stop < stop:
                    domain_exclusions.append((max(start, candidate_stop), stop))
                if candidate_stop <= start or candidate_start >= stop:
                    domain_exclusions = [(start, stop)]

    if comparison_possible and window is not None and series_candidate_time is not None:
        before = float(profile["edge_exclusion"]["before_s"])
        after = float(profile["edge_exclusion"]["after_s"])
        event_positions = [group.time for group in groups_reference] + mapped_candidate_events
        user_exclusions = [(value - before, value + after) for value in event_positions]
        sample_exclusions = _sample_barriers(
            reference.waveforms.time, [group.time for group in groups_reference]
        )
        sample_exclusions += _sample_barriers(series_candidate_time, mapped_candidate_events)
        exclusions = _merge_intervals(
            domain_exclusions + user_exclusions + sample_exclusions,
            window,
        )
        duration = window[1] - window[0]
        excluded_duration = sum(stop - start for start, stop in exclusions)
        coverage_duration = max(0.0, duration - excluded_duration)
        coverage = coverage_duration / duration
        alignment_record["window"] = {
            "start_s": window[0],
            "stop_s": window[1],
            "duration_s": duration,
        }
        alignment_record["exclusions"] = {
            "policy_edge_before_s": before,
            "policy_edge_after_s": after,
            "merged_interval_count": len(exclusions),
            "excluded_duration_s": excluded_duration,
            "included_duration_s": coverage_duration,
            "coverage_fraction": coverage,
            "intervals": [{"start_s": start, "stop_s": stop} for start, stop in exclusions],
        }
        minimum_coverage = float(profile["coverage"]["minimum_fraction"])
        for signal_profile in profile["signals"]:
            name = signal_profile["name"]
            signal_results.append(
                _compare_signal(
                    name,
                    signal_profile,
                    _Series(reference.waveforms.time, reference.waveforms.signals[name]),
                    _Series(series_candidate_time, candidate.waveforms.signals[name]),
                    window,
                    exclusions,
                    coverage,
                    coverage_duration,
                    minimum_coverage,
                )
            )

    if not comparison_possible or window is None:
        alignment_record["window"] = {
            "status": "not_comparable",
            "reason": "no valid common comparison domain or event warp",
        }
        for signal_profile in profile["signals"]:
            signal_results.append(
                {
                    "name": signal_profile["name"],
                    "unit": signal_profile["unit"],
                    "status": "not_comparable",
                    "metrics": None,
                    "gates": [],
                    "reason": "run compatibility or alignment preconditions failed",
                }
            )

    if (
        not compatible
        or not comparison_possible
        or any(signal["status"] == "not_comparable" for signal in signal_results)
    ):
        status: Status = "not_comparable"
    elif event_timing_gate["status"] == "fail" or any(
        signal["status"] == "fail" for signal in signal_results
    ):
        status = "fail"
    else:
        status = "pass"

    all_gates = (
        compatibility_gates
        + [event_timing_gate]
        + [gate for signal in signal_results for gate in signal["gates"]]
    )
    counts = {
        value: sum(gate["status"] == value for gate in all_gates)
        for value in ("pass", "fail", "not_comparable")
    }
    return {
        "schema": RUN_COMPARISON_SCHEMA,
        "kind": "verified-run-numerical-comparison",
        "status": status,
        "claim": {
            "classification": "numerical-equivalence-evidence",
            "validation_claim": False,
            "compatibility_claim": False,
            "hardware_sign_off_allowed": False,
            "statement": (
                "A pass is limited to the declared waveform and event tolerances for two "
                "verified runs of the same resolved ideal model; it is not hardware validation."
            ),
        },
        "generator": {"name": "pear-sim", "version": __version__},
        "profile": {"id": profile["id"], "sha256": profile["sha256"]},
        "sources": {
            "reference": _source_record(reference),
            "candidate": _source_record(candidate),
        },
        "compatibility": {
            "status": "pass" if compatible else "not_comparable",
            "gates": compatibility_gates,
        },
        "alignment": alignment_record,
        "signals": signal_results,
        "summary": {
            "status": status,
            "signal_count": len(signal_results),
            "signal_status_counts": {
                value: sum(signal["status"] == value for signal in signal_results)
                for value in ("pass", "fail", "not_comparable")
            },
            "gate_count": len(all_gates),
            "gate_status_counts": counts,
            "coverage_fraction": coverage if window is not None else None,
        },
    }


def _source_record(run: _VerifiedRun) -> dict[str, Any]:
    return {
        "run_sha256": run.run["run_sha256"],
        "model_sha256": run.run["model_sha256"],
        "request_sha256": run.run["request_sha256"],
        "structural_topology_sha256": run.topology_sha256,
        "run_manifest": {
            "sha256": run.run_manifest_sha256,
            "bytes": (run.directory / "run.json").stat().st_size,
        },
        "artifact_manifest": {
            "sha256": run.artifact_manifest_sha256,
            "bytes": run.artifact_manifest_bytes,
            "verified": True,
        },
        "bundle_sha256": run.bundle_sha256,
        "platform": run.run.get("platform"),
    }


def _compare_signal(
    name: str,
    profile: dict[str, Any],
    reference: _Series,
    candidate: _Series,
    window: tuple[float, float],
    exclusions: list[tuple[float, float]],
    coverage: float,
    included_duration: float,
    minimum_coverage: float,
) -> dict[str, Any]:
    coverage_gate = {
        "id": "coverage",
        "status": "pass"
        if coverage >= minimum_coverage and included_duration > 0.0
        else "not_comparable",
        "metric": "included_time_fraction",
        "actual": coverage,
        "limit": minimum_coverage,
        "operator": ">=",
    }
    if coverage_gate["status"] != "pass":
        return {
            "name": name,
            "unit": profile["unit"],
            "status": "not_comparable",
            "metrics": None,
            "gates": [coverage_gate],
            "reason": "included waveform coverage is below the declared minimum",
        }

    breakpoints = {window[0], window[1]}
    breakpoints.update(value for value in reference.time if window[0] < value < window[1])
    breakpoints.update(value for value in candidate.time if window[0] < value < window[1])
    for start, stop in exclusions:
        breakpoints.add(start)
        breakpoints.add(stop)
    ordered = sorted(breakpoints)
    squared_error = 0.0
    absolute_error = 0.0
    reference_square = 0.0
    reference_max = 0.0
    maximum_error = 0.0
    integrated_duration = 0.0
    interval_count = 0
    for start, stop in zip(ordered, ordered[1:]):
        if stop <= start:
            continue
        midpoint = 0.5 * (start + stop)
        if _inside_intervals(midpoint, exclusions):
            continue
        r0 = _interpolate(reference, start)
        r1 = _interpolate(reference, stop)
        c0 = _interpolate(candidate, start)
        c1 = _interpolate(candidate, stop)
        e0 = c0 - r0
        e1 = c1 - r1
        duration = stop - start
        squared_error += duration * (e0 * e0 + e0 * e1 + e1 * e1) / 3.0
        reference_square += duration * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0
        absolute_error += _integral_abs_linear(e0, e1, duration)
        maximum_error = max(maximum_error, abs(e0), abs(e1))
        reference_max = max(reference_max, abs(r0), abs(r1))
        integrated_duration += duration
        interval_count += 1
    duration_tolerance = max(1e-15, included_duration * 1e-12)
    if integrated_duration <= 0.0 or not math.isclose(
        integrated_duration, included_duration, rel_tol=1e-12, abs_tol=duration_tolerance
    ):
        coverage_gate["status"] = "not_comparable"
        return {
            "name": name,
            "unit": profile["unit"],
            "status": "not_comparable",
            "metrics": None,
            "gates": [coverage_gate],
            "reason": "piecewise integration did not cover the retained domain",
        }

    rmse = math.sqrt(max(0.0, squared_error / integrated_duration))
    reference_rms = math.sqrt(max(0.0, reference_square / integrated_duration))
    mean_absolute_error = absolute_error / integrated_duration
    if reference_rms == 0.0:
        if rmse == 0.0:
            nrmse: float | None = 0.0
        else:
            nrmse = None
    else:
        nrmse = rmse / reference_rms
    reference_scale = max(reference_rms, reference_max)
    tolerances = profile["tolerances"]
    max_limit = (
        float(tolerances["max_absolute_error"]["absolute"])
        + float(tolerances["max_absolute_error"]["relative"]) * reference_scale
    )
    mean_limit = (
        float(tolerances["mean_absolute_error"]["absolute"])
        + float(tolerances["mean_absolute_error"]["relative"]) * reference_scale
    )
    gates = [
        coverage_gate,
        _numeric_gate("max_absolute_error", maximum_error, max_limit),
        _numeric_gate("mean_absolute_error", mean_absolute_error, mean_limit),
        {
            "id": "nrmse",
            "status": (
                "not_comparable"
                if nrmse is None
                else "pass"
                if nrmse <= float(tolerances["nrmse"])
                else "fail"
            ),
            "metric": "root_mean_square_error_over_reference_rms",
            "actual": nrmse,
            "limit": float(tolerances["nrmse"]),
            "operator": "<=",
        },
    ]
    if any(gate["status"] == "not_comparable" for gate in gates):
        status: Status = "not_comparable"
    elif any(gate["status"] == "fail" for gate in gates):
        status = "fail"
    else:
        status = "pass"
    return {
        "name": name,
        "unit": profile["unit"],
        "status": status,
        "metrics": {
            "maximum_absolute_error": maximum_error,
            "mean_absolute_error": mean_absolute_error,
            "root_mean_square_error": rmse,
            "reference_rms": reference_rms,
            "reference_maximum_absolute": reference_max,
            "normalization": "reference_rms",
            "nrmse": nrmse,
            "included_duration_s": integrated_duration,
            "interval_count": interval_count,
            "integration": "exact_integrals_of_piecewise_linear_recorded_samples",
        },
        "gates": gates,
        "reason": None,
    }


def _event_warp(
    reference: _VerifiedRun,
    candidate: _VerifiedRun,
    reference_groups: tuple[_EventGroup, ...],
    candidate_groups: tuple[_EventGroup, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]] | None:
    if _event_semantics(reference_groups) != _event_semantics(candidate_groups):
        return None
    if (
        reference.waveforms.time[0] != candidate.waveforms.time[0]
        or reference.waveforms.time[-1] != candidate.waveforms.time[-1]
    ):
        return None
    # Endpoints alone are not semantic alignment anchors: allowing them would turn
    # a different run duration into a free global time-scale fit.  Piecewise event
    # alignment is available only when at least one matching event is strictly
    # inside both waveform domains.
    if not any(
        reference.waveforms.time[0] < reference_group.time < reference.waveforms.time[-1]
        and candidate.waveforms.time[0] < candidate_group.time < candidate.waveforms.time[-1]
        for reference_group, candidate_group in zip(
            reference_groups,
            candidate_groups,
            strict=True,
        )
    ):
        return None
    candidate_anchors = [candidate.waveforms.time[0]]
    reference_anchors = [reference.waveforms.time[0]]
    for reference_group, candidate_group in zip(reference_groups, candidate_groups, strict=True):
        if (
            candidate_group.time == candidate_anchors[-1]
            and reference_group.time == reference_anchors[-1]
        ):
            continue
        if (
            candidate_group.time <= candidate_anchors[-1]
            or reference_group.time <= reference_anchors[-1]
        ):
            return None
        candidate_anchors.append(candidate_group.time)
        reference_anchors.append(reference_group.time)
    candidate_stop = candidate.waveforms.time[-1]
    reference_stop = reference.waveforms.time[-1]
    if candidate_stop == candidate_anchors[-1] and reference_stop == reference_anchors[-1]:
        pass
    elif candidate_stop > candidate_anchors[-1] and reference_stop > reference_anchors[-1]:
        candidate_anchors.append(candidate_stop)
        reference_anchors.append(reference_stop)
    else:
        return None
    if len(candidate_anchors) < 2:
        return None
    return tuple(candidate_anchors), tuple(reference_anchors)


def _warp_time(value: float, warp: tuple[tuple[float, ...], tuple[float, ...]]) -> float:
    source, destination = warp
    index = bisect_right(source, value) - 1
    index = max(0, min(index, len(source) - 2))
    fraction = (value - source[index]) / (source[index + 1] - source[index])
    return destination[index] + fraction * (destination[index + 1] - destination[index])


def _group_events(events: Sequence[dict[str, Any]]) -> tuple[_EventGroup, ...]:
    groups: list[_EventGroup] = []
    current_time: float | None = None
    current: list[dict[str, Any]] = []
    for event in events:
        event_time = float(event["time"])
        semantic = {key: value for key, value in event.items() if key != "time"}
        if current_time is None or event_time == current_time:
            current_time = event_time
            current.append(semantic)
        else:
            groups.append(_EventGroup(current_time, tuple(current)))
            current_time = event_time
            current = [semantic]
    if current_time is not None:
        groups.append(_EventGroup(current_time, tuple(current)))
    return tuple(groups)


def _event_semantics(groups: Sequence[_EventGroup]) -> list[list[dict[str, Any]]]:
    return [[dict(value) for value in group.semantics] for group in groups]


def _event_semantic_hash(groups: Sequence[_EventGroup]) -> str:
    return sha256_json(
        {
            "schema": "pear-sim/event-semantic-sequence/v0.1",
            "groups": _event_semantics(groups),
        }
    )


def _sample_barriers(
    sample_time: Sequence[float], event_time: Sequence[float]
) -> list[tuple[float, float]]:
    barriers: list[tuple[float, float]] = []
    for event in event_time:
        index = bisect_right(sample_time, event)
        left_index = max(0, index - 2 if index and sample_time[index - 1] == event else index - 1)
        right_index = min(len(sample_time) - 1, index)
        if right_index > left_index:
            barriers.append((sample_time[left_index], sample_time[right_index]))
    return barriers


def _merge_intervals(
    intervals: Sequence[tuple[float, float]], window: tuple[float, float]
) -> list[tuple[float, float]]:
    clipped = sorted(
        (max(window[0], start), min(window[1], stop))
        for start, stop in intervals
        if min(window[1], stop) > max(window[0], start)
    )
    merged: list[tuple[float, float]] = []
    for start, stop in clipped:
        if not merged or start > merged[-1][1]:
            merged.append((start, stop))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], stop))
    return merged


def _inside_intervals(value: float, intervals: Sequence[tuple[float, float]]) -> bool:
    return any(start < value < stop for start, stop in intervals)


def _interpolate(series: _Series, value: float) -> float:
    if value < series.time[0] or value > series.time[-1]:
        raise RunComparisonError("internal comparison point lies outside a waveform domain")
    index = bisect_right(series.time, value) - 1
    if index < 0:
        return series.value[0]
    if index >= len(series.time) - 1:
        return series.value[-1]
    if value == series.time[index]:
        return series.value[index]
    fraction = (value - series.time[index]) / (series.time[index + 1] - series.time[index])
    return series.value[index] + fraction * (series.value[index + 1] - series.value[index])


def _integral_abs_linear(start: float, stop: float, duration: float) -> float:
    if start == 0.0 and stop == 0.0:
        return 0.0
    if start * stop >= 0.0:
        return duration * (abs(start) + abs(stop)) / 2.0
    return duration * (start * start + stop * stop) / (2.0 * (abs(start) + abs(stop)))


def _exact_gate(identifier: str, reference: Any, candidate: Any, equal: bool) -> dict[str, Any]:
    return {
        "id": identifier,
        "status": "pass" if equal else "not_comparable",
        "operator": "==",
        "reference": reference,
        "candidate": candidate,
    }


def _numeric_gate(identifier: str, actual: float, limit: float) -> dict[str, Any]:
    return {
        "id": identifier,
        "status": "pass" if actual <= limit else "fail",
        "metric": identifier,
        "actual": actual,
        "limit": limit,
        "operator": "<=",
    }


def _prepare_destination(output_dir: str | Path, source_directories: Sequence[Path]) -> Path:
    destination = Path(output_dir).resolve()
    for source in source_directories:
        try:
            destination.relative_to(source)
        except ValueError:
            pass
        else:
            raise RunComparisonError("comparison output must be outside both immutable run trees")
    if destination.exists():
        try:
            destination_stat = destination.lstat()
        except OSError as exc:
            raise RunComparisonError(f"cannot inspect comparison output: {exc}") from exc
        if stat.S_ISLNK(destination_stat.st_mode) or not stat.S_ISDIR(destination_stat.st_mode):
            raise RunComparisonError("comparison output must be a non-symlink directory")
        if any(destination.iterdir()):
            raise RunComparisonError(
                "comparison output directory is not empty; evidence is never overwritten"
            )
    else:
        destination.mkdir(parents=True)
    return destination


def _safe_member(directory: Path, reference: Any, label: str) -> Path:
    if not isinstance(reference, str) or not reference:
        raise RunComparisonError(f"{label} path must be a non-empty string")
    if "\\" in reference:
        raise RunComparisonError(f"{label} path must use normalized POSIX separators")
    pure = PurePosixPath(reference)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise RunComparisonError(f"{label} path is not a safe normalized relative path")
    if pure.as_posix() != reference:
        raise RunComparisonError(f"{label} path is not normalized")
    current = directory
    for part in pure.parts:
        current = current / part
        try:
            item_stat = current.lstat()
        except OSError as exc:
            raise RunComparisonError(f"cannot inspect {label}: {exc}") from exc
        if stat.S_ISLNK(item_stat.st_mode):
            raise RunComparisonError(f"{label} must not traverse a symlink")
    if not current.is_file():
        raise RunComparisonError(f"{label} is missing or is not a regular file")
    try:
        current.resolve(strict=True).relative_to(directory)
    except (OSError, ValueError) as exc:
        raise RunComparisonError(f"{label} escapes the run directory") from exc
    return current.resolve(strict=True)


def _read_json(path: Path, label: str, maximum_bytes: int) -> dict[str, Any]:
    if path.stat().st_size > maximum_bytes:
        raise RunComparisonError(f"{label} exceeds the byte limit")
    try:
        value = _strict_json_loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RunComparisonError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise RunComparisonError(f"{label} must contain an object")
    return value


def _strict_json_copy(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    try:
        copied = _strict_json_loads(
            json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RunComparisonError(f"cannot normalize {label}: {exc}") from exc
    if not isinstance(copied, dict):
        raise RunComparisonError(f"{label} must contain an object")
    return copied


def _strict_json_loads(source: str) -> Any:
    value = json.loads(
        source,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_unique_json_object,
    )
    _require_finite(value)
    return value


def _require_keys(value: dict[str, Any], allowed: set[str], required: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(required - set(value))
    if unknown:
        raise RunComparisonError(f"{label} has unknown fields: {', '.join(unknown)}")
    if missing:
        raise RunComparisonError(f"{label} is missing fields: {', '.join(missing)}")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RunComparisonError(f"{label} must be an object")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RunComparisonError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise RunComparisonError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise RunComparisonError(f"{label} must be a finite number")
    return result


def _nonnegative_finite(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result < 0.0:
        raise RunComparisonError(f"{label} must be non-negative")
    return result


def _signal_unit(name: str) -> str:
    try:
        return signal_unit(name)
    except ModelError as error:
        raise RunComparisonError(
            f"unsupported signal expression {name!r}; no unit guessing is allowed"
        ) from error


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _write_json(path: Path, payload: Any) -> None:
    _write_text(
        path,
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _render_report(comparison: dict[str, Any]) -> str:
    status = str(comparison["status"])
    status_class = f"status-{status.replace('_', '-')}"
    signal_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(str(signal['name']))}</code></td>"
        f"<td>{html.escape(str(signal['unit']))}</td>"
        f"<td>{html.escape(str(signal['status']))}</td>"
        f"<td>{html.escape(_metric_text(signal.get('metrics'), 'maximum_absolute_error'))}</td>"
        f"<td>{html.escape(_metric_text(signal.get('metrics'), 'mean_absolute_error'))}</td>"
        f"<td>{html.escape(_metric_text(signal.get('metrics'), 'nrmse'))}</td>"
        "</tr>"
        for signal in comparison["signals"]
    )
    compatibility_rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(str(gate['id']))}</code></td>"
        f"<td>{html.escape(str(gate['status']))}</td>"
        "</tr>"
        for gate in comparison["compatibility"]["gates"]
    )
    source_rows = "".join(
        "<tr>"
        f"<td>{html.escape(role)}</td>"
        f"<td><code>{html.escape(str(record['run_sha256']))}</code></td>"
        f"<td><code>{html.escape(str(record['bundle_sha256']))}</code></td>"
        "</tr>"
        for role, record in comparison["sources"].items()
    )
    raw = html.escape(json.dumps(comparison, indent=2, sort_keys=True, ensure_ascii=False))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'none'; connect-src 'none'; img-src data:; object-src 'none'; base-uri 'none'; form-action 'none'">
  <title>PEARSim verified run comparison</title>
  <style>
    body {{ margin: 0; background: #f4f7fb; color: #17324d; font: 15px/1.45 system-ui,sans-serif; }}
    main {{ max-width: 1080px; margin: auto; padding: 28px 20px 56px; }}
    header,section {{ background: white; border: 1px solid #d8e2ec; border-radius: 10px; padding: 20px; margin-bottom: 16px; overflow-x: auto; }}
    h1,h2 {{ margin-top: 0; }} .status {{ font-weight: 800; text-transform: uppercase; }}
    .status-pass {{ color: #067647; }} .status-fail {{ color: #b42318; }} .status-not-comparable {{ color: #9a6700; }}
    .notice {{ border-left: 4px solid #d69e2e; background: #fffaf0; padding: 10px 12px; }}
    table {{ width: 100%; border-collapse: collapse; }} th,td {{ text-align: left; border-bottom: 1px solid #e6edf3; padding: 8px; }}
    code,pre {{ font-family: ui-monospace,monospace; }} code {{ overflow-wrap: anywhere; word-break: break-word; }} pre {{ overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; }}
  </style>
</head>
<body><main>
  <header><h1>Verified run comparison</h1><p class="status {status_class}">{html.escape(status)}</p>
  <p class="notice">Numerical-equivalence evidence only. No physical-validation, compatibility, or hardware sign-off claim is made.</p></header>
  <section><h2>Signals</h2><table><thead><tr><th>Signal</th><th>Unit</th><th>Status</th><th>Max abs.</th><th>Mean abs.</th><th>NRMSE</th></tr></thead><tbody>{signal_rows}</tbody></table></section>
  <section><h2>Compatibility gates</h2><table><thead><tr><th>Gate</th><th>Status</th></tr></thead><tbody>{compatibility_rows}</tbody></table></section>
  <section><h2>Source bundles</h2><table><thead><tr><th>Role</th><th>Run SHA-256</th><th>Bundle SHA-256</th></tr></thead><tbody>{source_rows}</tbody></table></section>
  <section><h2>Machine-readable record</h2><pre>{raw}</pre></section>
  <p>Generated by PEARSim {html.escape(__version__)} · offline static HTML · no scripts or network resources</p>
</main></body></html>"""


def _metric_text(metrics: Any, key: str) -> str:
    if not isinstance(metrics, dict) or metrics.get(key) is None:
        return "—"
    value = metrics[key]
    return f"{value:.8g}" if isinstance(value, (int, float)) else str(value)


__all__ = [
    "COMPARE_PROFILE_SCHEMA",
    "RUN_COMPARISON_BUNDLE_SCHEMA",
    "RUN_COMPARISON_SCHEMA",
    "RunComparisonArtifacts",
    "RunComparisonError",
    "compare_runs",
]
