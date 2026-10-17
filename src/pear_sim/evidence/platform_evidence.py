"""Fail-closed aggregation of the native full-v1 numerical evidence matrix.

The aggregator never runs a simulation and never turns incomplete evidence into a
passing result.  It accepts exactly one immutable Linux, Darwin, and Windows bundle,
verifies their recursive root manifests, validates the complete canonical 9-by-16
matrix, and compares the signed physical values in each cell's semantic signature.
Independent oracle-error gates remain source evidence; they are deliberately not
used as cross-platform comparison values.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Sequence

from pear_sim._jsonutil import (
    reject_json_constant as _reject_json_constant,
    require_finite as _require_finite,
    unique_json_object as _unique_json_object,
    write_json as _write_json,
)
from pear_sim.run.contracts import sha256_file, sha256_json
from pear_sim.reporting.numerical_report import generate_numerical_report


CROSS_PLATFORM_EVIDENCE_SCHEMA = "pear-sim/cross-platform-evidence/v0.1"
PLATFORM_EVIDENCE_SCHEMA = "pear-sim/platform-evidence/canonical-buck-numerical/v0.1"
NUMERICAL_PROFILE_SCHEMA = "pear-sim/numerical-evidence-profile/v0.1"
SEMANTIC_SIGNATURE_SCHEMA = "pear-sim/numerical-semantic-signature/canonical-buck/v0.1"
ORACLE_SCHEMA = "pear-sim/benchmark-oracle/ideal-buck-boundary/v0.1"
CROSS_PLATFORM_POLICY_SCHEMA = "pear-sim/cross-platform-numerical-policy/v0.1"
ARTIFACT_MANIFEST_SCHEMA = "pear-sim/artifact-manifest/v0.1"
PLATFORM_EVIDENCE_FILENAME = "platform-evidence.json"
REQUIRED_SYSTEMS = ("Darwin", "Linux", "Windows")
SUPPORTED_STATUSES = frozenset({"pass", "fail", "not-run"})
MAX_JSON_BYTES = 64 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

_PLATFORM_KIND = "canonical-buck-full-numerical-evidence-matrix"
_MANIFEST_SUBJECT_KIND = "canonical-buck-numerical-evidence"
_ORACLE_KIND = "independent-periodic-affine-buck-boundary-verification"
_ORACLE_BOUNDARY_ABSOLUTE_TOLERANCE_OHM = 1e-12
_ORACLE_BOUNDARY_RELATIVE_TOLERANCE = 1e-12
_LOCAL_GATE_IDS = (
    "regime",
    "boundary_state_nonzero",
    "boundary_state_zero",
    "natural_event_time",
    "reverse_current",
    "cycle_mean_output",
    "mode_fractions",
    "energy_balance",
    "cycle_periodicity",
    "unclassified_fraction",
    "mode_fraction_sum",
)
_SIGNATURE_FIELD_IDS = (
    "regime",
    "state.cycle_start.inductor_current_a",
    "state.cycle_start.output_voltage_v",
    "state.switch_off.inductor_current_a",
    "state.switch_off.output_voltage_v",
    "state.cycle_end.inductor_current_a",
    "state.cycle_end.output_voltage_v",
    "events.diode_zero.present",
    "events.diode_zero.time_s",
    "metrics.mean_output_voltage_v",
    "metrics.minimum_inductor_current_a",
    "modes.fraction.on",
    "modes.fraction.freewheel",
    "modes.fraction.idle",
    "modes.fraction.unclassified",
    "physics.balance_residual_relative_signed",
    "cycle.delta_inductor_current_a",
    "cycle.delta_output_voltage_v",
    "events.semantic_sequence_sha256",
    "events.count",
    "modes.topology_sequence_sha256",
    "modes.interval_count",
)
_STUDY_GATE_SUFFIX = {
    "tolerance": "convergence",
    "max_step": "convergence",
    "output_grid": "independence",
    "solver": "consistency",
}
_UNKNOWN_REVISIONS = frozenset({"unknown", "unavailable", "none", "null"})


class PlatformEvidenceError(ValueError):
    """Raised when native evidence cannot support a trustworthy aggregation."""

    code = "evidence.platform_invalid"


@dataclass(frozen=True)
class CrossPlatformEvidenceArtifacts:
    evidence_json: Path
    artifact_manifest: Path
    report_json: Path
    report_html: Path
    overall_status: str


@dataclass(frozen=True)
class _SignatureField:
    field_id: str
    quantity: str
    unit: str
    policy_id: str
    value: str | float | int | bool
    tolerance_kind: str
    absolute_tolerance: float | None
    relative_tolerance: float | None


@dataclass(frozen=True)
class _SemanticSignature:
    status: str
    fields: Mapping[str, _SignatureField]
    reason: str | None


@dataclass(frozen=True)
class _Cell:
    cell_id: str
    point_id: str
    configuration_id: str
    expected_regime: str
    status: str
    local_gates: tuple[Mapping[str, Any], ...]
    signature: _SemanticSignature


@dataclass(frozen=True)
class _PlatformBundle:
    system: str
    platform: Mapping[str, Any]
    profile: Mapping[str, Any]
    resolved_profile: Mapping[str, Any]
    profile_sha256: str
    source: Mapping[str, Any]
    oracle: Mapping[str, Any]
    oracle_points: tuple[Mapping[str, Any], ...]
    overall_status: str
    cells: Mapping[str, _Cell]
    studies: Mapping[str, Mapping[str, Any]]
    evidence_sha256: str
    manifest_sha256: str


def _read_json_object(path: Path, *, role: str) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise PlatformEvidenceError(f"Cannot inspect {role} file {path}: {exc}") from exc
    if size > MAX_JSON_BYTES:
        raise PlatformEvidenceError(
            f"{role} file exceeds the {MAX_JSON_BYTES}-byte contract limit: {path}"
        )
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise PlatformEvidenceError(f"Cannot read {role} JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlatformEvidenceError(f"{role} JSON must contain an object: {path}")
    try:
        _require_finite(value)
    except ValueError as exc:
        raise PlatformEvidenceError(f"Invalid {role} JSON in {path}: {exc}") from exc
    return value


def _artifact_path(directory: Path, reference: Any, *, role: str) -> Path:
    if not isinstance(reference, str) or not reference:
        raise PlatformEvidenceError(f"{role} artifact path must be a non-empty string")
    pure = PurePosixPath(reference)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise PlatformEvidenceError(f"{role} artifact path escapes its bundle: {reference!r}")
    candidate = directory.joinpath(*pure.parts)
    if candidate.is_symlink():
        raise PlatformEvidenceError(f"{role} artifact must not be a symbolic link: {reference!r}")
    try:
        candidate.resolve(strict=True).relative_to(directory)
    except (OSError, ValueError) as exc:
        raise PlatformEvidenceError(
            f"{role} artifact does not resolve inside its bundle: {reference!r}"
        ) from exc
    if not candidate.is_file():
        raise PlatformEvidenceError(f"{role} artifact is not a regular file: {reference!r}")
    return candidate


def _verified_platform_payload(directory: Path) -> tuple[dict[str, Any], str]:
    manifest_path = directory / "artifact-manifest.json"
    evidence_path = directory / PLATFORM_EVIDENCE_FILENAME
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise PlatformEvidenceError(
            f"Platform bundle requires a regular root artifact-manifest.json: {directory}"
        )
    manifest = _read_json_object(manifest_path, role="platform artifact manifest")
    if manifest.get("schema") != ARTIFACT_MANIFEST_SCHEMA:
        raise PlatformEvidenceError(
            f"Unsupported platform artifact manifest schema in {manifest_path}: "
            f"{manifest.get('schema')!r}"
        )
    subject = manifest.get("subject")
    if not isinstance(subject, dict) or subject.get("kind") != _MANIFEST_SUBJECT_KIND:
        raise PlatformEvidenceError(
            f"Platform manifest subject.kind must be {_MANIFEST_SUBJECT_KIND!r}: {manifest_path}"
        )
    subject_sha256 = _sha256(subject.get("sha256"), path="manifest subject.sha256")
    records = manifest.get("artifacts")
    if not isinstance(records, list):
        raise PlatformEvidenceError(f"Platform manifest artifacts must be an array: {manifest_path}")
    recorded_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise PlatformEvidenceError(
                f"Platform manifest artifact {index} must be an object: {manifest_path}"
            )
        artifact = _artifact_path(
            directory,
            record.get("path"),
            role=f"manifest artifact {index}",
        )
        relative = artifact.relative_to(directory).as_posix()
        if relative in recorded_paths:
            raise PlatformEvidenceError(
                f"Platform manifest repeats artifact {relative!r}: {manifest_path}"
            )
        recorded_paths.add(relative)
        digest = _sha256(record.get("sha256"), path=f"manifest artifact {relative}.sha256")
        if sha256_file(artifact) != digest:
            raise PlatformEvidenceError(
                f"Platform manifest hash mismatch for artifact {relative!r}"
            )
        byte_count = record.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) or byte_count < 0:
            raise PlatformEvidenceError(
                f"Platform manifest artifact {relative!r} has an invalid byte count"
            )
        if artifact.stat().st_size != byte_count:
            raise PlatformEvidenceError(
                f"Platform manifest byte count mismatch for artifact {relative!r}"
            )
    actual_paths = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path != manifest_path
        and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    }
    if actual_paths != recorded_paths:
        raise PlatformEvidenceError(
            "Platform manifest must exactly cover regular bundle artifacts; "
            f"unrecorded={sorted(actual_paths - recorded_paths)}, "
            f"nonexistent={sorted(recorded_paths - actual_paths)}"
        )
    if PLATFORM_EVIDENCE_FILENAME not in recorded_paths or not evidence_path.is_file():
        raise PlatformEvidenceError(
            f"Platform manifest must bind {PLATFORM_EVIDENCE_FILENAME}: {manifest_path}"
        )
    if subject_sha256 != sha256_file(evidence_path):
        raise PlatformEvidenceError(
            "Platform manifest subject.sha256 does not bind platform-evidence.json"
        )
    payload = _read_json_object(evidence_path, role="platform evidence")
    if payload.get("schema") != PLATFORM_EVIDENCE_SCHEMA:
        raise PlatformEvidenceError(
            f"Unsupported platform evidence schema {payload.get('schema')!r}; "
            f"expected {PLATFORM_EVIDENCE_SCHEMA!r}"
        )
    return payload, sha256_file(manifest_path)


def _sha256(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
        raise PlatformEvidenceError(f"{path} must be a lowercase SHA-256")
    return value


def _require_status(value: Any, *, path: str) -> str:
    if value not in SUPPORTED_STATUSES:
        raise PlatformEvidenceError(
            f"{path} must be one of {sorted(SUPPORTED_STATUSES)}, got {value!r}"
        )
    return str(value)


def _nonempty_string(value: Any, *, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PlatformEvidenceError(f"{path} must be a non-empty string")
    return value


def _nonnegative_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlatformEvidenceError(f"{path} must be a finite non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise PlatformEvidenceError(f"{path} must be a finite non-negative number")
    return number


def _positive_number(value: Any, *, path: str) -> float:
    number = _nonnegative_number(value, path=path)
    if number <= 0.0:
        raise PlatformEvidenceError(f"{path} must be strictly positive")
    return number


def _finite_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlatformEvidenceError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise PlatformEvidenceError(f"{path} must be a finite number")
    return number


def _within_ulps(left: float, right: float, *, limit: int) -> bool:
    """Return whether finite values fit a conservative binary64 ULP envelope."""

    return abs(left - right) <= limit * max(math.ulp(left), math.ulp(right))


def _ulp_span(left: float, right: float) -> float:
    return abs(left - right) / max(math.ulp(left), math.ulp(right))


def _combined_status(statuses: Iterable[str]) -> str:
    values = tuple(statuses)
    if not values:
        return "not-run"
    if "fail" in values:
        return "fail"
    if any(value != "pass" for value in values):
        return "not-run"
    return "pass"


def _expected_profile() -> dict[str, Any]:
    # Lazy import avoids coupling module import order to CLI command registration.
    from pear_sim.evidence.numerical_evidence import full_v1_profile

    return full_v1_profile().as_dict()


def _verify_claim(payload: Mapping[str, Any]) -> None:
    if payload.get("kind") != _PLATFORM_KIND:
        raise PlatformEvidenceError(
            f"platform evidence kind must be {_PLATFORM_KIND!r}"
        )
    claim = payload.get("claim")
    if not isinstance(claim, dict):
        raise PlatformEvidenceError("platform evidence claim must be an object")
    if claim.get("classification") != "verification":
        raise PlatformEvidenceError("platform evidence claim.classification must be 'verification'")
    if claim.get("validation_claim") is not False:
        raise PlatformEvidenceError("platform evidence must not make a validation claim")
    if claim.get("hardware_sign_off_allowed") is not False:
        raise PlatformEvidenceError("platform evidence must prohibit hardware sign-off")
    _nonempty_string(claim.get("scope"), path="platform evidence claim.scope")


def _profile_excerpt(resolved: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "sha256",
        "load_ratios",
        "point_count",
        "configuration_count",
        "cell_count",
        "studies",
    )
    return {key: resolved[key] for key in keys}


def _verify_profile_binding(
    directory: Path,
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    profile = payload.get("profile")
    if not isinstance(profile, dict):
        raise PlatformEvidenceError("platform evidence profile must be an object")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("profile"), dict):
        raise PlatformEvidenceError("platform evidence must identify its numerical profile")
    profile_record = artifacts["profile"]
    profile_path = _artifact_path(
        directory,
        profile_record.get("path"),
        role="platform numerical profile",
    )
    profile_file_sha256 = _sha256(
        profile_record.get("sha256"),
        path="platform evidence artifacts.profile.sha256",
    )
    if profile_file_sha256 != sha256_file(profile_path):
        raise PlatformEvidenceError("platform evidence numerical profile artifact hash mismatched")
    resolved = _read_json_object(profile_path, role="resolved numerical profile")
    if resolved.get("schema") != NUMERICAL_PROFILE_SCHEMA:
        raise PlatformEvidenceError(
            f"Unsupported resolved numerical profile schema: {resolved.get('schema')!r}"
        )
    claimed_semantic_sha256 = _sha256(
        resolved.get("sha256"), path="resolved numerical profile sha256"
    )
    without_digest = {key: value for key, value in resolved.items() if key != "sha256"}
    if claimed_semantic_sha256 != sha256_json(without_digest):
        raise PlatformEvidenceError("resolved numerical profile semantic SHA-256 is invalid")
    expected = _expected_profile()
    if resolved != expected:
        raise PlatformEvidenceError(
            "resolved numerical profile is not the normative full-v1 9-by-16 profile"
        )
    expected_excerpt = _profile_excerpt(expected)
    if profile != expected_excerpt:
        raise PlatformEvidenceError(
            "platform evidence profile excerpt differs from the normative full-v1 profile"
        )
    if payload.get("cross_platform_policy") != expected["cross_platform_policy"]:
        raise PlatformEvidenceError(
            "platform evidence cross-platform policy differs from the resolved profile"
        )
    if payload.get("configurations") != expected["configurations"]:
        raise PlatformEvidenceError(
            "platform evidence configurations differ from the resolved full-v1 profile"
        )
    return profile, resolved


def _source_binding(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    source = payload.get("source")
    if not isinstance(source, dict):
        raise PlatformEvidenceError("platform evidence source must be an object")
    revision = _nonempty_string(source.get("revision"), path="source.revision")
    normalized_revision = revision.strip().casefold()
    if normalized_revision in _UNKNOWN_REVISIONS or any(
        normalized_revision.startswith(f"{placeholder}-")
        for placeholder in _UNKNOWN_REVISIONS
    ):
        raise PlatformEvidenceError("platform evidence rejects an unknown source revision")
    revision_method = _nonempty_string(
        source.get("revision_method"), path="source.revision_method"
    )
    pear_sim_version = _nonempty_string(
        source.get("pear_sim_version"), path="source.pear_sim_version"
    )
    model_file = _nonempty_string(source.get("model_file"), path="source.model_file")
    model_hashes = source.get("model_hashes")
    expected_hash_keys = {"source_sha256", "model_sha256", "request_sha256"}
    if not isinstance(model_hashes, dict) or set(model_hashes) != expected_hash_keys:
        raise PlatformEvidenceError(
            "source.model_hashes must contain exactly source/model/request SHA-256 values"
        )
    hashes = {
        key: _sha256(model_hashes[key], path=f"source.model_hashes.{key}")
        for key in sorted(expected_hash_keys)
    }
    return {
        "revision": revision,
        "revision_method": revision_method,
        "pear_sim_version": pear_sim_version,
        "model_file": model_file,
        "model_hashes": hashes,
    }


def _oracle_binding(
    directory: Path,
    payload: Mapping[str, Any],
    resolved_profile: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    tuple[Mapping[str, Any], ...],
    Mapping[str, Any],
]:
    oracle_record = payload.get("oracle")
    if not isinstance(oracle_record, dict):
        raise PlatformEvidenceError("platform evidence oracle must be an object")
    if oracle_record.get("schema") != ORACLE_SCHEMA:
        raise PlatformEvidenceError(f"platform evidence oracle.schema must be {ORACLE_SCHEMA!r}")
    oracle_sha256 = _sha256(oracle_record.get("sha256"), path="oracle.sha256")
    boundary = _positive_number(
        oracle_record.get("boundary_load_ohm"), path="oracle.boundary_load_ohm"
    )
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not isinstance(artifacts.get("oracle"), dict):
        raise PlatformEvidenceError("platform evidence must identify its oracle artifact")
    artifact_record = artifacts["oracle"]
    oracle_path = _artifact_path(
        directory,
        artifact_record.get("path"),
        role="platform oracle",
    )
    artifact_sha256 = _sha256(
        artifact_record.get("sha256"), path="artifacts.oracle.sha256"
    )
    actual_sha256 = sha256_file(oracle_path)
    if oracle_sha256 != actual_sha256 or artifact_sha256 != actual_sha256:
        raise PlatformEvidenceError("platform oracle hashes do not bind the oracle artifact")
    oracle = _read_json_object(oracle_path, role="canonical buck oracle")
    if oracle.get("schema") != ORACLE_SCHEMA:
        raise PlatformEvidenceError(f"oracle artifact schema must be {ORACLE_SCHEMA!r}")
    if oracle.get("kind") != _ORACLE_KIND:
        raise PlatformEvidenceError(f"oracle artifact kind must be {_ORACLE_KIND!r}")
    claim = oracle.get("claim")
    if (
        not isinstance(claim, dict)
        or claim.get("classification") != "verification"
        or claim.get("validation_claim") is not False
    ):
        raise PlatformEvidenceError("oracle artifact must retain its verification-only claim")
    boundary_record = oracle.get("boundary")
    if not isinstance(boundary_record, dict) or boundary_record.get("load_ohm") != boundary:
        raise PlatformEvidenceError("oracle boundary does not match platform evidence")
    boundary_contract = {
        key: value for key, value in boundary_record.items() if key != "load_ohm"
    }
    if not boundary_contract:
        raise PlatformEvidenceError("oracle boundary must retain a non-numeric contract")
    method = oracle.get("method")
    if not isinstance(method, dict) or not method:
        raise PlatformEvidenceError("oracle method contract must be a non-empty object")
    parameters = oracle.get("parameters_si")
    if not isinstance(parameters, dict) or not parameters:
        raise PlatformEvidenceError("oracle parameters_si must be a non-empty object")
    if oracle.get("load_ratios") != resolved_profile["load_ratios"]:
        raise PlatformEvidenceError("oracle load ratios differ from the full-v1 profile")
    contract = oracle.get("comparison_gate_contract")
    if not isinstance(contract, list):
        raise PlatformEvidenceError("oracle comparison_gate_contract must be an array")
    contract_ids = tuple(
        _nonempty_string(record.get("id"), path="oracle comparison gate id")
        if isinstance(record, dict)
        else ""
        for record in contract
    )
    if contract_ids != _LOCAL_GATE_IDS:
        raise PlatformEvidenceError("oracle comparison gate IDs differ from the canonical contract")
    points = oracle.get("points")
    if not isinstance(points, list) or len(points) != 9:
        raise PlatformEvidenceError("oracle must contain exactly nine canonical load points")
    validated_points: list[Mapping[str, Any]] = []
    for index, (raw_point, ratio) in enumerate(
        zip(points, resolved_profile["load_ratios"], strict=True)
    ):
        if not isinstance(raw_point, dict):
            raise PlatformEvidenceError(f"oracle points[{index}] must be an object")
        expected_id = f"load-ratio-{format(float(ratio), '.12g').replace('.', 'p')}"
        expected_regime = "CCM" if ratio < 1.0 else "BCM" if ratio == 1.0 else "DCM"
        if raw_point.get("point_id") != expected_id:
            raise PlatformEvidenceError(
                f"oracle points[{index}].point_id differs from the canonical full-v1 ID"
            )
        if raw_point.get("load_ratio_to_bcm") != ratio:
            raise PlatformEvidenceError(
                f"oracle points[{index}] load ratio differs from the full-v1 profile"
            )
        if raw_point.get("regime") != expected_regime:
            raise PlatformEvidenceError(
                f"oracle point {expected_id!r} has the wrong canonical regime"
            )
        point_load = _positive_number(
            raw_point.get("load_ohm"), path=f"oracle points[{index}].load_ohm"
        )
        if point_load != boundary * float(ratio):
            raise PlatformEvidenceError(
                f"oracle point {expected_id!r} load is inconsistent with its local boundary"
            )
        validated_points.append(raw_point)
    health = oracle.get("oracle_health")
    if not isinstance(health, dict) or health.get("status") != "pass":
        raise PlatformEvidenceError("oracle health must be pass")
    if health.get("point_count") != 9 or health.get("failed_gate_count") != 0:
        raise PlatformEvidenceError("oracle health counts differ from the canonical matrix")
    identity_payload = {
        "schema": ORACLE_SCHEMA,
        "kind": _ORACLE_KIND,
        "claim": claim,
        "method": method,
        "parameters_si": parameters,
        "boundary_contract": boundary_contract,
        "load_ratios": oracle["load_ratios"],
        "comparison_gate_contract": contract,
    }
    return (
        {
            "schema": ORACLE_SCHEMA,
            "oracle_identity_sha256": sha256_json(identity_payload),
            "local_artifact_sha256": oracle_sha256,
            "boundary_load_ohm": boundary,
        },
        tuple(validated_points),
        parameters,
    )


def _parse_local_gates(record: Mapping[str, Any], *, path: str) -> tuple[Mapping[str, Any], ...]:
    raw_gates = record.get("gates")
    if not isinstance(raw_gates, list):
        raise PlatformEvidenceError(f"{path}.gates must be an array")
    gate_ids: list[str] = []
    gates: list[Mapping[str, Any]] = []
    for index, gate in enumerate(raw_gates):
        if not isinstance(gate, dict):
            raise PlatformEvidenceError(f"{path}.gates[{index}] must be an object")
        gate_id = _nonempty_string(gate.get("id"), path=f"{path}.gates[{index}].id")
        _require_status(gate.get("status"), path=f"{path}.gates[{index}].status")
        gate_ids.append(gate_id)
        gates.append(gate)
    if len(gate_ids) != len(set(gate_ids)) or set(gate_ids) != set(_LOCAL_GATE_IDS):
        raise PlatformEvidenceError(
            f"{path} must contain exactly the eleven canonical oracle gate IDs"
        )
    return tuple(gates)


_DYNAMIC_STATE_FIELDS = {
    "state.cycle_start.inductor_current_a": ("cycle_start", "inductor_current_a", "current"),
    "state.cycle_start.output_voltage_v": ("cycle_start", "output_voltage_v", "voltage"),
    "state.switch_off.inductor_current_a": ("switch_off", "inductor_current_a", "current"),
    "state.switch_off.output_voltage_v": ("switch_off", "output_voltage_v", "voltage"),
    "state.cycle_end.inductor_current_a": ("cycle_end", "inductor_current_a", "current"),
    "state.cycle_end.output_voltage_v": ("cycle_end", "output_voltage_v", "voltage"),
}


def _oracle_tolerance_scales(
    parameters: Mapping[str, Any],
) -> tuple[float, float, float]:
    vin_v = _positive_number(parameters.get("vin_v"), path="oracle parameters_si.vin_v")
    inductance_h = _positive_number(
        parameters.get("inductance_h"), path="oracle parameters_si.inductance_h"
    )
    period_s = _positive_number(
        parameters.get("period_s"), path="oracle parameters_si.period_s"
    )
    duty = _positive_number(parameters.get("duty"), path="oracle parameters_si.duty")
    if duty >= 1.0:
        raise PlatformEvidenceError("oracle parameters_si.duty must be less than one")
    current_base_a = vin_v * duty * period_s / inductance_h
    if not math.isfinite(current_base_a) or current_base_a <= 0.0:
        raise PlatformEvidenceError(
            "oracle parameters_si produce an invalid current tolerance scale"
        )
    return vin_v, current_base_a, period_s


def _state_absolute_tolerance(reference: float, scale: float) -> float:
    if abs(reference) <= scale * 1e-10:
        return max(scale * 2e-10, 1e-15)
    return max(abs(reference) * 2e-7, scale * 1e-15)


def _expected_dynamic_absolute_tolerance(
    field_id: str,
    oracle_point: Mapping[str, Any],
    oracle_parameters: Mapping[str, Any],
) -> float:
    """Resolve one profile ``absolute_rule`` from the local oracle artifact."""

    vin_v, current_base_a, period_s = _oracle_tolerance_scales(oracle_parameters)
    if field_id in _DYNAMIC_STATE_FIELDS:
        state_name, quantity_name, scale_name = _DYNAMIC_STATE_FIELDS[field_id]
        states = oracle_point.get("states")
        if not isinstance(states, dict):
            raise PlatformEvidenceError(
                f"oracle point {oracle_point.get('point_id')!r}.states must be an object"
            )
        state = states.get(state_name)
        if not isinstance(state, dict):
            raise PlatformEvidenceError(
                f"oracle point {oracle_point.get('point_id')!r}.states.{state_name} "
                "must be an object"
            )
        reference = _finite_number(
            state.get(quantity_name),
            path=(
                f"oracle point {oracle_point.get('point_id')!r}.states."
                f"{state_name}.{quantity_name}"
            ),
        )
        scale = current_base_a if scale_name == "current" else vin_v
        return _state_absolute_tolerance(reference, scale)
    if field_id == "events.diode_zero.time_s":
        return max(period_s * 1e-5, 1e-15)
    if field_id == "metrics.mean_output_voltage_v":
        metrics = oracle_point.get("metrics")
        if not isinstance(metrics, dict):
            raise PlatformEvidenceError(
                f"oracle point {oracle_point.get('point_id')!r}.metrics must be an object"
            )
        reference = _finite_number(
            metrics.get("mean_output_voltage_v"),
            path=(
                f"oracle point {oracle_point.get('point_id')!r}.metrics."
                "mean_output_voltage_v"
            ),
        )
        return max(abs(reference) * 2e-5, vin_v * 2e-17)
    if field_id in {
        "metrics.minimum_inductor_current_a",
        "cycle.delta_inductor_current_a",
    }:
        return max(current_base_a * 2e-10, 1e-15)
    if field_id == "cycle.delta_output_voltage_v":
        return max(vin_v * 2e-10, 1e-15)
    raise PlatformEvidenceError(
        f"semantic field {field_id!r} has an unsupported dynamic absolute tolerance rule"
    )


def _parse_signature(
    record: Mapping[str, Any],
    *,
    cell_status: str,
    path: str,
    signature_policy: Mapping[str, Any],
    oracle_point: Mapping[str, Any],
    oracle_parameters: Mapping[str, Any],
) -> _SemanticSignature:
    signature = record.get("semantic_signature")
    if not isinstance(signature, dict) or signature.get("schema") != SEMANTIC_SIGNATURE_SCHEMA:
        raise PlatformEvidenceError(
            f"{path}.semantic_signature must use {SEMANTIC_SIGNATURE_SCHEMA!r}"
        )
    status = signature.get("status")
    fields = signature.get("fields")
    if status == "not-run":
        if cell_status != "not-run":
            raise PlatformEvidenceError(
                f"{path} has a not-run signature but its oracle gate status is {cell_status!r}"
            )
        if fields != {}:
            raise PlatformEvidenceError(f"{path} not-run semantic signature fields must be empty")
        reason = _nonempty_string(
            signature.get("reason"), path=f"{path}.semantic_signature.reason"
        )
        return _SemanticSignature(status="not-run", fields={}, reason=reason)
    if status != "available":
        raise PlatformEvidenceError(
            f"{path}.semantic_signature.status must be 'available' or 'not-run'"
        )
    if cell_status == "not-run":
        raise PlatformEvidenceError(f"{path} not-run cell must not claim an available signature")
    if not isinstance(fields, dict) or set(fields) != set(_SIGNATURE_FIELD_IDS):
        raise PlatformEvidenceError(
            f"{path} available semantic signature must contain exactly 22 canonical fields"
        )
    field_policy_ids = signature_policy["field_policy_ids"]
    tolerance_policies = signature_policy["tolerance_policies"]
    parsed: dict[str, _SignatureField] = {}
    for field_id in _SIGNATURE_FIELD_IDS:
        raw = fields[field_id]
        field_path = f"{path}.semantic_signature.fields.{field_id}"
        if not isinstance(raw, dict):
            raise PlatformEvidenceError(f"{field_path} must be an object")
        quantity = _nonempty_string(raw.get("quantity"), path=f"{field_path}.quantity")
        unit = _nonempty_string(raw.get("unit"), path=f"{field_path}.unit")
        policy_id = _nonempty_string(raw.get("policy_id"), path=f"{field_path}.policy_id")
        if policy_id != field_policy_ids[field_id]:
            raise PlatformEvidenceError(
                f"{field_path}.policy_id differs from the full-v1 profile mapping"
            )
        tolerance = raw.get("cross_platform_tolerance")
        if not isinstance(tolerance, dict):
            raise PlatformEvidenceError(f"{field_path}.cross_platform_tolerance must be an object")
        if tolerance.get("policy_id") != policy_id:
            raise PlatformEvidenceError(
                f"{field_path} embedded tolerance does not bind its policy_id"
            )
        policy = tolerance_policies[policy_id]
        kind = tolerance.get("kind")
        if kind != policy.get("kind") or kind not in {"exact", "numeric"}:
            raise PlatformEvidenceError(
                f"{field_path} tolerance kind differs from its full-v1 policy"
            )
        value = raw.get("value")
        if kind == "exact":
            if set(tolerance) != {"kind", "policy_id"}:
                raise PlatformEvidenceError(
                    f"{field_path} exact tolerance must not declare numeric limits"
                )
            if value is None or not isinstance(value, (str, int, float, bool)):
                raise PlatformEvidenceError(f"{field_path}.value must be a JSON scalar")
            if isinstance(value, float) and not math.isfinite(value):
                raise PlatformEvidenceError(f"{field_path}.value must be finite")
            absolute = None
            relative = None
        else:
            if set(tolerance) != {"kind", "policy_id", "absolute", "relative"}:
                raise PlatformEvidenceError(
                    f"{field_path} numeric tolerance must resolve absolute and relative limits"
                )
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise PlatformEvidenceError(f"{field_path}.value must be a finite number")
            value = float(value)
            if not math.isfinite(value):
                raise PlatformEvidenceError(f"{field_path}.value must be finite")
            absolute = _nonnegative_number(
                tolerance.get("absolute"), path=f"{field_path}.tolerance.absolute"
            )
            relative = _nonnegative_number(
                tolerance.get("relative"), path=f"{field_path}.tolerance.relative"
            )
            if "relative" in policy and relative != float(policy["relative"]):
                raise PlatformEvidenceError(
                    f"{field_path}.tolerance.relative differs from the profile policy"
                )
            if "absolute" in policy and absolute != float(policy["absolute"]):
                raise PlatformEvidenceError(
                    f"{field_path}.tolerance.absolute differs from the profile policy"
                )
            if "absolute_rule" in policy:
                expected_absolute = _expected_dynamic_absolute_tolerance(
                    field_id,
                    oracle_point,
                    oracle_parameters,
                )
                if not _within_ulps(absolute, expected_absolute, limit=1):
                    raise PlatformEvidenceError(
                        f"{field_path}.tolerance.absolute differs from its local oracle rule; "
                        f"recorded={absolute!r}, expected={expected_absolute!r}"
                    )
            elif "absolute" not in policy:
                raise PlatformEvidenceError(
                    f"{field_path} numeric policy has no absolute value or rule"
                )
        parsed[field_id] = _SignatureField(
            field_id=field_id,
            quantity=quantity,
            unit=unit,
            policy_id=policy_id,
            value=value,
            tolerance_kind=str(kind),
            absolute_tolerance=absolute,
            relative_tolerance=relative,
        )
    return _SemanticSignature(status="available", fields=parsed, reason=None)


def _validate_signature_policy(resolved_profile: Mapping[str, Any]) -> Mapping[str, Any]:
    policy = resolved_profile.get("cross_platform_policy")
    if not isinstance(policy, dict) or policy.get("schema") != CROSS_PLATFORM_POLICY_SCHEMA:
        raise PlatformEvidenceError("full-v1 cross-platform policy schema is invalid")
    signature = policy.get("semantic_signature")
    if not isinstance(signature, dict) or signature.get("schema") != SEMANTIC_SIGNATURE_SCHEMA:
        raise PlatformEvidenceError("full-v1 semantic signature policy schema is invalid")
    if signature.get("required_field_count") != len(_SIGNATURE_FIELD_IDS):
        raise PlatformEvidenceError("full-v1 semantic signature field count is not 22")
    field_policy_ids = signature.get("field_policy_ids")
    tolerance_policies = signature.get("tolerance_policies")
    if not isinstance(field_policy_ids, dict) or set(field_policy_ids) != set(
        _SIGNATURE_FIELD_IDS
    ):
        raise PlatformEvidenceError("full-v1 semantic signature field policy IDs are invalid")
    if not isinstance(tolerance_policies, dict):
        raise PlatformEvidenceError("full-v1 semantic signature tolerance policies are invalid")
    referenced = set(field_policy_ids.values())
    if not referenced <= set(tolerance_policies):
        raise PlatformEvidenceError("full-v1 signature references an undefined tolerance policy")
    return signature


def _parse_cells(
    payload: Mapping[str, Any],
    resolved_profile: Mapping[str, Any],
    oracle_points: Sequence[Mapping[str, Any]],
    oracle_parameters: Mapping[str, Any],
) -> Mapping[str, _Cell]:
    records = payload.get("cells")
    if not isinstance(records, list) or len(records) != 144:
        raise PlatformEvidenceError("platform evidence must contain exactly 144 full-v1 cells")
    signature_policy = _validate_signature_policy(resolved_profile)
    point_by_id = {str(point["point_id"]): point for point in oracle_points}
    configurations = resolved_profile["configurations"]
    configuration_ids = [str(configuration["id"]) for configuration in configurations]
    expected_pairs = {
        (point_id, configuration_id)
        for point_id in point_by_id
        for configuration_id in configuration_ids
    }
    seen_pairs: set[tuple[str, str]] = set()
    cells: dict[str, _Cell] = {}
    for index, record in enumerate(records):
        path = f"platform evidence cells[{index}]"
        if not isinstance(record, dict):
            raise PlatformEvidenceError(f"{path} must be an object")
        point_id = _nonempty_string(record.get("point_id"), path=f"{path}.point_id")
        configuration_id = _nonempty_string(
            record.get("configuration_id"), path=f"{path}.configuration_id"
        )
        pair = (point_id, configuration_id)
        if pair not in expected_pairs or pair in seen_pairs:
            raise PlatformEvidenceError(
                f"{path} is not a unique member of the full-v1 Cartesian cell matrix"
            )
        seen_pairs.add(pair)
        cell_id = _nonempty_string(record.get("id"), path=f"{path}.id")
        expected_cell_id = f"{point_id}--{configuration_id}"
        if cell_id != expected_cell_id or cell_id in cells:
            raise PlatformEvidenceError(f"{path}.id does not match its canonical point/config pair")
        point = point_by_id[point_id]
        if record.get("load_ratio_to_bcm") != point["load_ratio_to_bcm"]:
            raise PlatformEvidenceError(f"{path}.load_ratio_to_bcm differs from the oracle")
        expected_regime = str(point["regime"])
        if record.get("expected_regime") != expected_regime:
            raise PlatformEvidenceError(f"{path}.expected_regime differs from the oracle")
        local_gates = _parse_local_gates(record, path=path)
        derived_status = _combined_status(str(gate["status"]) for gate in local_gates)
        status = _require_status(record.get("status"), path=f"{path}.status")
        if status != derived_status:
            raise PlatformEvidenceError(f"{path}.status differs from its eleven oracle gates")
        signature = _parse_signature(
            record,
            cell_status=status,
            path=path,
            signature_policy=signature_policy,
            oracle_point=point,
            oracle_parameters=oracle_parameters,
        )
        cells[cell_id] = _Cell(
            cell_id=cell_id,
            point_id=point_id,
            configuration_id=configuration_id,
            expected_regime=expected_regime,
            status=status,
            local_gates=local_gates,
            signature=signature,
        )
    if seen_pairs != expected_pairs:
        raise PlatformEvidenceError("platform evidence does not cover the full-v1 Cartesian matrix")
    return cells


def _counter_payload(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _validate_studies(
    payload: Mapping[str, Any],
    resolved_profile: Mapping[str, Any],
    oracle_points: Sequence[Mapping[str, Any]],
    cells: Mapping[str, _Cell],
) -> Mapping[str, Mapping[str, Any]]:
    raw_studies = payload.get("studies")
    expected_studies = resolved_profile["studies"]
    if not isinstance(raw_studies, dict) or set(raw_studies) != set(expected_studies):
        raise PlatformEvidenceError("platform evidence study IDs differ from full-v1")
    point_ids = [str(point["point_id"]) for point in oracle_points]
    validated: dict[str, Mapping[str, Any]] = {}
    for study_id, configuration_ids in expected_studies.items():
        study = raw_studies[study_id]
        path = f"platform evidence studies.{study_id}"
        if not isinstance(study, dict):
            raise PlatformEvidenceError(f"{path} must be an object")
        if study.get("configuration_ids") != configuration_ids:
            raise PlatformEvidenceError(f"{path}.configuration_ids differ from full-v1")
        if study.get("configuration_count") != len(configuration_ids):
            raise PlatformEvidenceError(f"{path}.configuration_count is inconsistent")
        selected = [
            cell for cell in cells.values() if cell.configuration_id in configuration_ids
        ]
        expected_cell_count = len(point_ids) * len(configuration_ids)
        if len(selected) != expected_cell_count or study.get("cell_count") != expected_cell_count:
            raise PlatformEvidenceError(f"{path}.cell_count is inconsistent")
        family_gates = study.get("gates")
        if not isinstance(family_gates, list) or len(family_gates) != len(point_ids):
            raise PlatformEvidenceError(f"{path}.gates must contain one gate per load point")
        expected_suffix = _STUDY_GATE_SUFFIX[study_id]
        expected_gate_ids = {f"{study_id}:{point_id}:{expected_suffix}" for point_id in point_ids}
        gate_by_id: dict[str, Mapping[str, Any]] = {}
        for gate_index, gate in enumerate(family_gates):
            if not isinstance(gate, dict):
                raise PlatformEvidenceError(f"{path}.gates[{gate_index}] must be an object")
            gate_id = _nonempty_string(
                gate.get("id"), path=f"{path}.gates[{gate_index}].id"
            )
            if gate_id in gate_by_id:
                raise PlatformEvidenceError(f"{path} repeats family gate {gate_id!r}")
            point_id = _nonempty_string(
                gate.get("point_id"), path=f"{path}.gates[{gate_index}].point_id"
            )
            if gate_id != f"{study_id}:{point_id}:{expected_suffix}":
                raise PlatformEvidenceError(f"{path} family gate ID does not match its point")
            if gate.get("configuration_ids") != configuration_ids:
                raise PlatformEvidenceError(
                    f"{path} family gate {gate_id!r} has inconsistent configuration IDs"
                )
            _require_status(gate.get("status"), path=f"{path}.gates[{gate_index}].status")
            gate_by_id[gate_id] = gate
        if set(gate_by_id) != expected_gate_ids:
            raise PlatformEvidenceError(f"{path} family gate IDs differ from full-v1")
        expected_cell_counts = _counter_payload(cell.status for cell in selected)
        expected_oracle_counts = _counter_payload(
            str(gate["status"]) for cell in selected for gate in cell.local_gates
        )
        expected_family_counts = _counter_payload(
            str(gate["status"]) for gate in family_gates
        )
        if study.get("cell_status_counts") != expected_cell_counts:
            raise PlatformEvidenceError(f"{path}.cell_status_counts are inconsistent")
        if study.get("oracle_gate_status_counts") != expected_oracle_counts:
            raise PlatformEvidenceError(f"{path}.oracle_gate_status_counts are inconsistent")
        if study.get("family_gate_status_counts") != expected_family_counts:
            raise PlatformEvidenceError(f"{path}.family_gate_status_counts are inconsistent")
        expected_status = _combined_status(
            [cell.status for cell in selected]
            + [str(gate["status"]) for gate in family_gates]
        )
        status = _require_status(study.get("status"), path=f"{path}.status")
        if status != expected_status:
            raise PlatformEvidenceError(f"{path}.status differs from its cells and family gates")
        validated[study_id] = study
    return validated


def _validate_root_status_and_summary(
    payload: Mapping[str, Any],
    resolved_profile: Mapping[str, Any],
    cells: Mapping[str, _Cell],
    studies: Mapping[str, Mapping[str, Any]],
) -> str:
    status = _require_status(payload.get("status"), path="platform evidence status")
    derived = _combined_status(
        [cell.status for cell in cells.values()]
        + [str(study["status"]) for study in studies.values()]
    )
    if status != derived:
        raise PlatformEvidenceError("platform evidence status differs from cells and studies")
    gate_statuses = [
        str(gate["status"]) for cell in cells.values() for gate in cell.local_gates
    ]
    family_statuses = [
        str(gate["status"]) for study in studies.values() for gate in study["gates"]
    ]
    expected_summary = {
        "status": status,
        "point_count": 9,
        "configuration_count": 16,
        "cell_count": 144,
        "gate_count": 144 * len(_LOCAL_GATE_IDS),
        "family_gate_count": 9 * len(_STUDY_GATE_SUFFIX),
        "cell_status_counts": _counter_payload(cell.status for cell in cells.values()),
        "gate_status_counts": _counter_payload(gate_statuses),
        "family_gate_status_counts": _counter_payload(family_statuses),
    }
    if payload.get("summary") != expected_summary:
        raise PlatformEvidenceError("platform evidence summary counts/statuses are inconsistent")
    if resolved_profile.get("point_count") != 9 or resolved_profile.get(
        "configuration_count"
    ) != 16 or resolved_profile.get("cell_count") != 144:
        raise PlatformEvidenceError("resolved profile is not the normative 9-by-16 matrix")
    return status


def _parse_platform_bundle(directory: Path) -> _PlatformBundle:
    payload, manifest_sha256 = _verified_platform_payload(directory)
    _verify_claim(payload)
    platform = payload.get("platform")
    if not isinstance(platform, dict):
        raise PlatformEvidenceError(f"platform evidence platform must be an object: {directory}")
    system = _nonempty_string(platform.get("system"), path="platform evidence platform.system")
    profile, resolved_profile = _verify_profile_binding(directory, payload)
    source = _source_binding(payload)
    oracle, oracle_points, oracle_parameters = _oracle_binding(
        directory, payload, resolved_profile
    )
    cells = _parse_cells(
        payload,
        resolved_profile,
        oracle_points,
        oracle_parameters,
    )
    studies = _validate_studies(payload, resolved_profile, oracle_points, cells)
    overall_status = _validate_root_status_and_summary(
        payload, resolved_profile, cells, studies
    )
    return _PlatformBundle(
        system=system,
        platform=platform,
        profile=profile,
        resolved_profile=resolved_profile,
        profile_sha256=str(resolved_profile["sha256"]),
        source=source,
        oracle=oracle,
        oracle_points=oracle_points,
        overall_status=overall_status,
        cells=cells,
        studies=studies,
        evidence_sha256=sha256_json(payload),
        manifest_sha256=manifest_sha256,
    )


def _same_binding(
    bundles: Mapping[str, _PlatformBundle],
    *,
    getter: Any,
    role: str,
) -> Mapping[str, Any]:
    values = {system: getter(bundle) for system, bundle in bundles.items()}
    digests = {system: sha256_json(value) for system, value in values.items()}
    if len(set(digests.values())) != 1:
        raise PlatformEvidenceError(f"Cross-platform {role} binding differs: {digests}")
    return values[REQUIRED_SYSTEMS[0]]


def _oracle_cross_platform_binding(
    bundles: Mapping[str, _PlatformBundle],
    systems: Sequence[str],
) -> dict[str, Any]:
    identity = _same_binding(
        bundles,
        getter=lambda bundle: {
            "schema": bundle.oracle["schema"],
            "oracle_identity_sha256": bundle.oracle["oracle_identity_sha256"],
        },
        role="oracle identity",
    )
    values = {
        system: float(bundles[system].oracle["boundary_load_ohm"])
        for system in systems
    }
    numeric_values = list(values.values())
    actual_range = max(numeric_values) - min(numeric_values)
    limit = max(
        _ORACLE_BOUNDARY_ABSOLUTE_TOLERANCE_OHM,
        _ORACLE_BOUNDARY_RELATIVE_TOLERANCE
        * max(abs(value) for value in numeric_values),
    )
    status = "pass" if actual_range <= limit else "fail"
    return {
        **dict(identity),
        "local_artifact_sha256": {
            system: bundles[system].oracle["local_artifact_sha256"]
            for system in systems
        },
        "boundary_load_ohm": {
            "values": values,
            "actual_range": actual_range,
            "limit": limit,
            "absolute_tolerance_ohm": _ORACLE_BOUNDARY_ABSOLUTE_TOLERANCE_OHM,
            "relative_tolerance": _ORACLE_BOUNDARY_RELATIVE_TOLERANCE,
            "formula": (
                "max(value)-min(value) <= "
                "max(absolute_tolerance_ohm, relative_tolerance*max(abs(value)))"
            ),
            "status": status,
        },
    }


def _field_result(
    systems: Sequence[str],
    cells: Mapping[str, _Cell],
    field_id: str,
    signature_policy: Mapping[str, Any],
) -> dict[str, Any]:
    platform_statuses = {system: cells[system].status for system in systems}
    signature_statuses = {system: cells[system].signature.status for system in systems}
    source_status = _combined_status(platform_statuses.values())
    available = {
        system: cells[system].signature.fields[field_id]
        for system in systems
        if cells[system].signature.status == "available"
    }
    expected_policy_id = signature_policy["field_policy_ids"][field_id]
    policy = signature_policy["tolerance_policies"][expected_policy_id]
    if available:
        descriptors = {
            (
                field.quantity,
                field.unit,
                field.policy_id,
                field.tolerance_kind,
                field.relative_tolerance,
            )
            for field in available.values()
        }
        if len(descriptors) != 1:
            rendered = {
                system: {
                    "quantity": field.quantity,
                    "unit": field.unit,
                    "policy_id": field.policy_id,
                    "kind": field.tolerance_kind,
                    "absolute": field.absolute_tolerance,
                    "relative": field.relative_tolerance,
                }
                for system, field in available.items()
            }
            raise PlatformEvidenceError(
                f"Cross-platform semantic field {field_id!r} metadata/tolerances differ: {rendered}"
            )
        exemplar = next(iter(available.values()))
        quantity: str | None = exemplar.quantity
        unit: str | None = exemplar.unit
        kind = exemplar.tolerance_kind
        absolute = exemplar.absolute_tolerance
        relative = exemplar.relative_tolerance
        platform_absolute_tolerances = {
            system: field.absolute_tolerance for system, field in available.items()
        }
        dynamic_absolute_ulp_span: float | None = None
        dynamic_absolute_uses_local_oracle = False
        if kind == "numeric" and "absolute_rule" in policy:
            dynamic_absolutes = [
                value
                for value in platform_absolute_tolerances.values()
                if value is not None
            ]
            if len(dynamic_absolutes) != len(available):
                raise PlatformEvidenceError(
                    f"Numeric semantic field {field_id!r} lacks a resolved absolute tolerance"
                )
            lowest_absolute = min(dynamic_absolutes)
            highest_absolute = max(dynamic_absolutes)
            dynamic_absolute_ulp_span = _ulp_span(
                lowest_absolute,
                highest_absolute,
            )
            dynamic_absolute_uses_local_oracle = True
            # Each local value has already been checked against its own oracle.
            # The minimum is the fail-safe common envelope: aggregation can never
            # loosen a platform's locally resolved tolerance.
            absolute = lowest_absolute
    else:
        # A full-v1 aggregate always retains one row per canonical field.  When all
        # three source cells are not-run there is no recorded field descriptor to
        # copy, so use explicit non-physical metadata while keeping every value and
        # the comparison itself not-run.
        quantity = field_id
        unit = "not-recorded"
        kind = str(policy["kind"])
        absolute = float(policy["absolute"]) if "absolute" in policy else None
        relative = float(policy["relative"]) if "relative" in policy else None
        platform_absolute_tolerances = {}
        dynamic_absolute_ulp_span = None
        dynamic_absolute_uses_local_oracle = False
    values = {
        system: (
            cells[system].signature.fields[field_id].value
            if cells[system].signature.status == "available"
            else None
        )
        for system in systems
    }
    comparison_status = "not-run"
    max_absolute_difference: float | None = None
    allowed_difference: float | None = None
    exact_match: bool | None = None
    if source_status == "pass":
        if len(available) != len(systems) or any(value is None for value in values.values()):
            raise PlatformEvidenceError(
                f"Passing cells are missing semantic field {field_id!r}"
            )
        if kind == "exact":
            exact_match = len({sha256_json(value) for value in values.values()}) == 1
            comparison_status = "pass" if exact_match else "fail"
        else:
            if absolute is None or relative is None:
                raise PlatformEvidenceError(
                    f"Numeric semantic field {field_id!r} lacks resolved tolerances"
                )
            numeric_values = [float(value) for value in values.values()]
            max_absolute_difference = max(numeric_values) - min(numeric_values)
            allowed_difference = max(
                absolute,
                relative * max(abs(value) for value in numeric_values),
            )
            comparison_status = (
                "pass" if max_absolute_difference <= allowed_difference else "fail"
            )
    status = source_status if source_status != "pass" else comparison_status
    formula = (
        "canonical JSON scalar values are identical"
        if kind == "exact"
        else "max(value)-min(value) <= max(absolute, relative*max(abs(value)))"
    )
    return {
        "id": field_id,
        "status": status,
        "source_status": source_status,
        "platform_statuses": platform_statuses,
        "signature_statuses": signature_statuses,
        "platform_values": values,
        "quantity": quantity,
        "unit": unit,
        "policy_id": expected_policy_id,
        "comparison": {
            "status": comparison_status,
            "kind": kind,
            "formula": formula,
            "absolute_tolerance": absolute,
            "platform_absolute_tolerances": platform_absolute_tolerances,
            "dynamic_absolute_tolerance_ulp_span": dynamic_absolute_ulp_span,
            "dynamic_absolute_tolerance_cross_platform_ceiling": None,
            "dynamic_absolute_common_limit_rule": (
                "minimum-locally-verified-platform-tolerance"
                if dynamic_absolute_uses_local_oracle
                else None
            ),
            "relative_tolerance": relative,
            "max_absolute_difference": max_absolute_difference,
            "allowed_difference": allowed_difference,
            "exact_match": exact_match,
        },
    }


def _prepare_destination(output_dir: str | Path, input_dirs: Sequence[Path]) -> Path:
    destination = Path(output_dir).resolve()
    for directory in input_dirs:
        try:
            destination.relative_to(directory)
        except ValueError:
            continue
        raise PlatformEvidenceError(
            "Cross-platform output directory must be outside every immutable input bundle"
        )
    if destination.exists():
        if not destination.is_dir():
            raise PlatformEvidenceError(f"Cross-platform output is not a directory: {destination}")
        if any(destination.iterdir()):
            raise PlatformEvidenceError(
                f"Cross-platform output directory is not empty: {destination}"
            )
    else:
        destination.mkdir(parents=True)
    return destination


def _cross_platform_studies(
    resolved_profile: Mapping[str, Any],
    cells: Sequence[Mapping[str, Any]],
    bundles: Mapping[str, _PlatformBundle],
    systems: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for study_id, configuration_ids in resolved_profile["studies"].items():
        selected = [cell for cell in cells if cell["configuration_id"] in configuration_ids]
        platform_family_evidence = {
            system: dict(bundles[system].studies[study_id]) for system in systems
        }
        platform_statuses = {
            system: str(platform_family_evidence[system]["status"]) for system in systems
        }
        cross_cell_status = _combined_status(str(cell["status"]) for cell in selected)
        comparison_statuses = [
            str(gate["status"]) for cell in selected for gate in cell["gates"]
        ]
        result[study_id] = {
            "status": _combined_status([*platform_statuses.values(), cross_cell_status]),
            "platform_statuses": platform_statuses,
            "cross_platform_cell_status": cross_cell_status,
            "configuration_ids": list(configuration_ids),
            "configuration_count": len(configuration_ids),
            "cell_count": len(selected),
            "cell_status_counts": _counter_payload(
                str(cell["status"]) for cell in selected
            ),
            "signature_comparison_count": len(comparison_statuses),
            "signature_comparison_status_counts": _counter_payload(comparison_statuses),
            "platform_family_evidence": platform_family_evidence,
        }
    return result


def _root_manifest(destination: Path, evidence_path: Path) -> Path:
    manifest_path = destination / "artifact-manifest.json"
    artifacts = [
        {
            "path": path.relative_to(destination).as_posix(),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(destination.rglob("*"))
        if path.is_file()
        and path != manifest_path
        and not path.is_symlink()
        and not any(part.startswith(".") for part in path.relative_to(destination).parts)
    ]
    _write_json(
        manifest_path,
        {
            "schema": ARTIFACT_MANIFEST_SCHEMA,
            "subject": {
                "kind": "cross-platform-evidence",
                "sha256": sha256_file(evidence_path),
                "semantic_sha256": sha256_json(
                    _read_json_object(evidence_path, role="cross-platform evidence")
                ),
            },
            "bundle_schema": CROSS_PLATFORM_EVIDENCE_SCHEMA,
            "artifacts": artifacts,
        },
    )
    return manifest_path


def aggregate_platform_evidence(
    evidence_dirs: Sequence[str | Path],
    output_dir: str | Path,
) -> CrossPlatformEvidenceArtifacts:
    """Verify and aggregate exactly one Linux, Darwin, and Windows full-v1 bundle."""

    directories = tuple(Path(path).resolve() for path in evidence_dirs)
    if len(directories) != len(REQUIRED_SYSTEMS):
        raise PlatformEvidenceError(
            f"Expected exactly {len(REQUIRED_SYSTEMS)} platform bundles, got {len(directories)}"
        )
    for directory in directories:
        if not directory.is_dir():
            raise PlatformEvidenceError(f"Platform evidence directory does not exist: {directory}")
    parsed = [_parse_platform_bundle(directory) for directory in directories]
    bundles: dict[str, _PlatformBundle] = {}
    for bundle in parsed:
        if bundle.system in bundles:
            raise PlatformEvidenceError(f"Duplicate platform evidence for {bundle.system!r}")
        bundles[bundle.system] = bundle
    if set(bundles) != set(REQUIRED_SYSTEMS):
        raise PlatformEvidenceError(
            f"Platform evidence must contain exactly {list(REQUIRED_SYSTEMS)}; "
            f"got {sorted(bundles)}"
        )
    systems = tuple(REQUIRED_SYSTEMS)
    profile_hashes = {bundle.profile_sha256 for bundle in bundles.values()}
    if len(profile_hashes) != 1:
        raise PlatformEvidenceError(
            "Cross-platform evidence was produced with different numerical profiles"
        )
    source = _same_binding(
        bundles,
        getter=lambda bundle: bundle.source,
        role="source/model/version",
    )
    oracle = _oracle_cross_platform_binding(bundles, systems)
    oracle_boundary_status = str(oracle["boundary_load_ohm"]["status"])
    resolved_profile = bundles[systems[0]].resolved_profile
    signature_policy = _validate_signature_policy(resolved_profile)
    point_ids = [str(point["point_id"]) for point in bundles[systems[0]].oracle_points]
    configuration_ids = [
        str(configuration["id"]) for configuration in resolved_profile["configurations"]
    ]
    expected_cell_ids = [
        f"{point_id}--{configuration_id}"
        for configuration_id in configuration_ids
        for point_id in point_ids
    ]
    cells: list[dict[str, Any]] = []
    for cell_id in expected_cell_ids:
        native = {system: bundles[system].cells[cell_id] for system in systems}
        point_ids_for_cell = {cell.point_id for cell in native.values()}
        configuration_ids_for_cell = {cell.configuration_id for cell in native.values()}
        if len(point_ids_for_cell) != 1 or len(configuration_ids_for_cell) != 1:
            raise PlatformEvidenceError(f"Cross-platform metadata differs for cell {cell_id!r}")
        point_id = next(iter(point_ids_for_cell))
        configuration_id = next(iter(configuration_ids_for_cell))
        study_ids = next(
            list(configuration["studies"])
            for configuration in resolved_profile["configurations"]
            if configuration["id"] == configuration_id
        )
        gates = [
            _field_result(systems, native, field_id, signature_policy)
            for field_id in _SIGNATURE_FIELD_IDS
        ]
        platform_statuses = {system: native[system].status for system in systems}
        source_status = _combined_status(platform_statuses.values())
        gate_status = _combined_status(str(gate["status"]) for gate in gates)
        cells.append(
            {
                "id": cell_id,
                "point_id": point_id,
                "configuration_id": configuration_id,
                "study_ids": study_ids,
                "expected_regime": native[systems[0]].expected_regime,
                "status": _combined_status((source_status, gate_status)),
                "source_status": source_status,
                "platform_statuses": platform_statuses,
                "platform_oracle_evidence": {
                    system: {
                        "status": native[system].status,
                        "gates": [dict(gate) for gate in native[system].local_gates],
                    }
                    for system in systems
                },
                "gates": gates,
            }
        )
    comparisons = [
        {
            "cell_id": cell["id"],
            "point_id": cell["point_id"],
            "configuration_id": cell["configuration_id"],
            "study_ids": cell["study_ids"],
            "gate_id": gate["id"],
            "kind": gate["comparison"]["kind"],
            "quantity": gate["quantity"],
            "unit": gate["unit"],
            "policy_id": gate["policy_id"],
            "values": gate["platform_values"],
            "actual_range": gate["comparison"]["max_absolute_difference"],
            "limit": gate["comparison"]["allowed_difference"],
            "status": gate["status"],
        }
        for cell in cells
        for gate in cell["gates"]
    ]
    if len(comparisons) != 144 * len(_SIGNATURE_FIELD_IDS):
        raise PlatformEvidenceError("internal comparison matrix is not the normative 3168 rows")
    platform_overall_statuses = {
        system: bundles[system].overall_status for system in systems
    }
    studies = _cross_platform_studies(
        resolved_profile, cells, bundles, systems
    )
    overall_status = _combined_status(
        [
            *platform_overall_statuses.values(),
            oracle_boundary_status,
            *(str(cell["status"]) for cell in cells),
            *(str(study["status"]) for study in studies.values()),
        ]
    )
    failures: list[dict[str, Any]] = [
        {"kind": "platform-status", "system": system, "status": status}
        for system, status in platform_overall_statuses.items()
        if status == "fail"
    ]
    if oracle_boundary_status == "fail":
        boundary_comparison = oracle["boundary_load_ohm"]
        failures.append(
            {
                "kind": "oracle-boundary-comparison",
                "status": "fail",
                "values": boundary_comparison["values"],
                "actual_range": boundary_comparison["actual_range"],
                "limit": boundary_comparison["limit"],
            }
        )
    for cell in cells:
        for system, evidence in cell["platform_oracle_evidence"].items():
            if evidence["status"] == "fail":
                failures.append(
                    {
                        "kind": "source-cell",
                        "system": system,
                        "cell_id": cell["id"],
                        "status": "fail",
                    }
                )
        for gate in cell["gates"]:
            if gate["comparison"]["status"] == "fail":
                failures.append(
                    {
                        "kind": "semantic-signature-comparison",
                        "cell_id": cell["id"],
                        "gate_id": gate["id"],
                        "comparison_status": "fail",
                        "platform_values": gate["platform_values"],
                    }
                )
    for study_id, study in studies.items():
        for system, native_study in study["platform_family_evidence"].items():
            for gate in native_study["gates"]:
                if gate["status"] == "fail":
                    failures.append(
                        {
                            "kind": "platform-family-gate",
                            "system": system,
                            "study_id": study_id,
                            "gate_id": gate["id"],
                            "status": "fail",
                        }
                    )
    comparison_status_counts = _counter_payload(
        str(comparison["status"]) for comparison in comparisons
    )
    cell_status_counts = _counter_payload(str(cell["status"]) for cell in cells)
    platforms = [
        {
            **dict(bundles[system].platform),
            "status": bundles[system].overall_status,
            "evidence_semantic_sha256": bundles[system].evidence_sha256,
            "artifact_manifest_sha256": bundles[system].manifest_sha256,
        }
        for system in systems
    ]
    profile = bundles[systems[0]].profile
    payload = {
        "schema": CROSS_PLATFORM_EVIDENCE_SCHEMA,
        "kind": "native_cross_platform_numerical_evidence",
        "status": overall_status,
        "profile": profile,
        "profile_semantic_sha256": profile["sha256"],
        "source_revision": source["revision"],
        "required_systems": list(systems),
        "platform_overall_statuses": platform_overall_statuses,
        "platforms": platforms,
        "inputs": [
            {
                "system": system,
                "evidence_semantic_sha256": bundles[system].evidence_sha256,
                "artifact_manifest_sha256": bundles[system].manifest_sha256,
                "profile_semantic_sha256": bundles[system].profile_sha256,
                "source": dict(bundles[system].source),
                "oracle": dict(bundles[system].oracle),
            }
            for system in systems
        ],
        "comparison_policy": {
            "values": "signed physical semantic-signature fields only",
            "signature_schema": SEMANTIC_SIGNATURE_SCHEMA,
            "required_field_ids": list(_SIGNATURE_FIELD_IDS),
            "tolerances": (
                "profile-bound policy IDs; static limits are identical, while dynamic "
                "limits are verified against each local oracle and resolved to their minimum"
            ),
            "dynamic_absolute_tolerance": {
                "local_verification": "within one ULP of the local oracle rule",
                "cross_platform_ceiling": None,
                "common_limit": "minimum locally verified platform tolerance",
                "ulp_span": "recorded as a diagnostic only",
            },
            "numeric_formula": (
                "max(value)-min(value) <= max(absolute, relative*max(abs(value)))"
            ),
            "categorical_formula": "canonical JSON scalar values are identical",
            "oracle_gate_values_compared": False,
            "status_precedence": "fail, then not-run, then pass",
            "source_binding": dict(source),
            "oracle_binding": dict(oracle),
        },
        "cells": cells,
        "comparisons": comparisons,
        "studies": studies,
        "failures": failures,
        "summary": {
            "status": overall_status,
            "platform_count": len(platforms),
            "cell_count": len(cells),
            "comparison_count": len(comparisons),
            "cell_status_counts": cell_status_counts,
            "comparison_status_counts": comparison_status_counts,
            "failure_count": len(failures),
        },
        "artifacts": {
            "human_report": {"path": "report/numerical-report.html"},
            "artifact_manifest": {"path": "artifact-manifest.json"},
        },
        "claim": {
            "validation_claim": False,
            "compatibility_claim": False,
            "hardware_sign_off_allowed": False,
            "statement": (
                "This artifact records native numerical consistency inside the declared "
                "profile and tolerances; it is not hardware or third-party-tool validation."
            ),
        },
    }
    destination = _prepare_destination(output_dir, directories)
    evidence_path = destination / "cross-platform-evidence.json"
    _write_json(evidence_path, payload)
    report = generate_numerical_report(
        payload,
        destination / "report",
        source_artifact_sha256=sha256_file(evidence_path),
    )
    manifest_path = _root_manifest(destination, evidence_path)
    return CrossPlatformEvidenceArtifacts(
        evidence_json=evidence_path,
        artifact_manifest=manifest_path,
        report_json=report.report_json,
        report_html=report.report_html,
        overall_status=overall_status,
    )


__all__ = [
    "ARTIFACT_MANIFEST_SCHEMA",
    "CROSS_PLATFORM_EVIDENCE_SCHEMA",
    "CrossPlatformEvidenceArtifacts",
    "PLATFORM_EVIDENCE_FILENAME",
    "PLATFORM_EVIDENCE_SCHEMA",
    "PlatformEvidenceError",
    "REQUIRED_SYSTEMS",
    "aggregate_platform_evidence",
]
