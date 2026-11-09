# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Alpha, headless human-review packets with content hash binding.

This module deliberately has no function that creates an approval. Agents may
prepare immutable review packets and validate a decision supplied by a human
review channel. Signature trust and reviewer authentication remain orchestration
concerns in the v0.1 alpha contract.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import mimetypes
from pathlib import Path
import re
import shutil
from typing import Any
from uuid import uuid4

from pear_sim.run.contracts import explain_model, physics_model_dict, simulation_request_dict
from pear_sim.spec.model import load_model
from pear_sim.reporting.render import render_svg
from pear_sim.review.review_annotations import append_review_annotation, validate_review_annotations
from pear_sim.review.review_errors import (
    ReviewDecisionError,
    ReviewIntegrityError,
    ReviewValidationError,
    StaleReviewError,
)


REVIEW_SCHEMA = "pear-sim/review/v0.1"
DECISION_SCHEMA = "pear-sim/review-decision/v0.1"
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PACKET_FIELDS = frozenset(
    {
        "schema",
        "review_id",
        "phase",
        "status",
        "created_at",
        "subject",
        "subject_digest",
        "artifacts",
        "files",
        "packet_digest",
    }
)
_ARTIFACT_FIELDS = frozenset(
    {"role", "logical_path", "path", "sha256", "size_bytes", "media_type"}
)
_TOPOLOGY_SUBJECT_FIELDS = frozenset(
    {"kind", "model_digest", "request_digest", "expectations_digest"}
)
_RESULT_SUBJECT_FIELDS = frozenset(
    {
        "kind",
        "run_digest",
        "model_digest",
        "request_digest",
        "run_artifacts_digest",
        "report_digest",
    }
)
_DECISION_FIELDS = frozenset(
    {
        "schema",
        "review_id",
        "phase",
        "subject_digest",
        "packet_digest",
        "decision",
        "reviewer",
        "decided_at",
        "reason_codes",
        "comment",
        "scope",
        "signature",
    }
)
_REVIEWER_FIELDS = frozenset({"kind", "id", "display_name", "authentication"})
_DECISIONS = frozenset({"approve", "request_changes", "reject"})


@dataclass(frozen=True)
class _Material:
    role: str
    logical_path: str
    data: bytes
    media_type: str

    @property
    def digest(self) -> str:
        return _sha256_bytes(self.data)


@dataclass(frozen=True)
class _TreeEntry:
    logical_path: str
    source: Path
    digest: str
    size_bytes: int
    media_type: str


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReviewValidationError(f"Value is not canonical JSON: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _digest_json(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ReviewValidationError(f"Cannot hash {path}: {exc}") from exc
    return "sha256:" + digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _media_type(path: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    return guessed or "application/octet-stream"


def _require_digest(value: Any, *, name: str, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ReviewValidationError(f"{name} must be a sha256:<64 lowercase hex> digest")


def _require_exact_fields(value: Mapping[str, Any], allowed: frozenset[str], *, name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ReviewValidationError(f"{name}: unknown field(s): {', '.join(unknown)}")


def _validate_timestamp(value: Any, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ReviewValidationError(f"{name} must be a non-empty RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReviewValidationError(f"{name} is not a valid RFC 3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ReviewValidationError(f"{name} must include a timezone")


def _safe_relative(value: Any, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReviewValidationError(f"{name} must be a non-empty bundle-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ReviewValidationError(f"{name} must be a safe bundle-relative path")
    return path


def _prepare_destination(out_dir: str | Path | None, *, sources: tuple[Path, ...]) -> Path:
    if out_dir is None:
        raise ReviewValidationError("out_dir is required")
    destination = Path(out_dir).resolve()
    for source in sources:
        try:
            destination.relative_to(source.resolve())
        except ValueError:
            continue
        raise ReviewValidationError(f"Review output cannot be inside source directory {source}")
    if destination.exists():
        if not destination.is_dir():
            raise ReviewValidationError(f"Review output is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ReviewValidationError(f"Review output directory is not empty: {destination}")
    else:
        destination.mkdir(parents=True)
    (destination / "artifacts").mkdir()
    return destination


def _material_from_input(value: Mapping[str, Any] | str | Path, *, role: str) -> _Material:
    if isinstance(value, Mapping):
        data = _canonical_json(dict(value))
        return _Material(role, f"{role}.json", data, "application/json")
    path = Path(value).resolve()
    if path.is_symlink() or not path.is_file():
        raise ReviewValidationError(f"{role} must be a regular file: {path}")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ReviewValidationError(f"Cannot read {role} file {path}: {exc}") from exc
    return _Material(role, path.name, data, _media_type(path.name))


def _topology_materials(
    model_path: str | Path,
    request: Mapping[str, Any] | str | Path | None,
    expectations: Mapping[str, Any] | str | Path | None,
) -> tuple[_Material, _Material, _Material, _Material, _Material, _Material | None]:
    path = Path(model_path).resolve()
    if path.is_symlink() or not path.is_file():
        raise ReviewValidationError(f"model_path must be a regular file: {path}")
    model = load_model(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ReviewValidationError(f"Cannot read model {path}: {exc}") from exc
    source_material = _Material("source", path.name, data, _media_type(path.name))
    model_material = _Material(
        "model",
        "model.json",
        _canonical_json(physics_model_dict(model)),
        "application/json",
    )
    request_material = (
        _Material(
            "request",
            "request.json",
            _canonical_json(simulation_request_dict(model)),
            "application/json",
        )
        if request is None
        else _material_from_input(request, role="request")
    )
    expectation_material = (
        None if expectations is None else _material_from_input(expectations, role="expectations")
    )
    explanation_material = _Material(
        "explanation",
        "explanation.json",
        _canonical_json(explain_model(model)),
        "application/json",
    )
    schematic_material = _Material(
        "schematic",
        "schematic.svg",
        render_svg(model).encode("utf-8"),
        "image/svg+xml",
    )
    return (
        source_material,
        model_material,
        request_material,
        explanation_material,
        schematic_material,
        expectation_material,
    )


def _artifact_record(material: _Material, destination: Path) -> dict[str, Any]:
    relative = Path("artifacts") / material.role / material.logical_path
    target = destination / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_bytes(material.data)
    except OSError as exc:
        raise ReviewValidationError(f"Cannot write review artifact {target}: {exc}") from exc
    return {
        "role": material.role,
        "logical_path": material.logical_path,
        "path": relative.as_posix(),
        "sha256": material.digest,
        "size_bytes": len(material.data),
        "media_type": material.media_type,
    }


def _tree_entries(root: Path) -> tuple[_TreeEntry, ...]:
    if root.is_symlink() or not root.is_dir():
        raise ReviewValidationError(f"Expected a regular directory: {root}")
    entries: list[_TreeEntry] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ReviewValidationError(f"Symlinks are not allowed in review sources: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ReviewValidationError(f"Unsupported filesystem entry in review source: {path}")
        logical = path.relative_to(root).as_posix()
        entries.append(
            _TreeEntry(logical, path, _file_digest(path), path.stat().st_size, _media_type(logical))
        )
    if not entries:
        raise ReviewValidationError(f"Review source directory is empty: {root}")
    return tuple(entries)


def _tree_digest(entries: tuple[_TreeEntry, ...] | list[dict[str, Any]]) -> str:
    if entries and isinstance(entries[0], _TreeEntry):
        normalized = [
            {
                "logical_path": entry.logical_path,
                "sha256": entry.digest,
                "size_bytes": entry.size_bytes,
            }
            for entry in entries
            if isinstance(entry, _TreeEntry)
        ]
    else:
        normalized = [
            {
                "logical_path": entry["logical_path"],
                "sha256": entry["sha256"],
                "size_bytes": entry["size_bytes"],
            }
            for entry in entries
            if isinstance(entry, dict)
        ]
    return _digest_json(normalized)


def _normalize_manifest_digest(value: Any, *, name: str) -> str:
    if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
        return f"sha256:{value}"
    if isinstance(value, str) and _DIGEST_RE.fullmatch(value):
        return value
    raise ReviewValidationError(f"run.json {name} must be a SHA-256 digest")


def _run_manifest_hashes(run_path: Path) -> dict[str, str]:
    manifest_path = run_path / "run.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewValidationError(f"Cannot read run manifest {manifest_path}: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != "pear-sim/run/v0.1":
        raise ReviewValidationError(f"Unsupported or invalid run manifest: {manifest_path}")
    return {
        name: _normalize_manifest_digest(manifest.get(name), name=name)
        for name in ("model_sha256", "request_sha256", "run_sha256")
    }


def _copy_tree_entries(
    entries: tuple[_TreeEntry, ...],
    *,
    role: str,
    destination: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in entries:
        relative = Path("artifacts") / role / entry.logical_path
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(entry.source, target)
        except OSError as exc:
            raise ReviewValidationError(f"Cannot copy review artifact {entry.source}: {exc}") from exc
        records.append(
            {
                "role": role,
                "logical_path": entry.logical_path,
                "path": relative.as_posix(),
                "sha256": entry.digest,
                "size_bytes": entry.size_bytes,
                "media_type": entry.media_type,
            }
        )
    return records


def _subject_digest(subject: Mapping[str, Any]) -> str:
    return _digest_json(dict(subject))


def _write_packet(destination: Path, packet: dict[str, Any]) -> dict[str, Any]:
    packet["packet_digest"] = _digest_json(packet)
    packet_path = destination / "packet.json"
    packet_path.write_text(
        json.dumps(packet, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    template = {
        "schema": DECISION_SCHEMA,
        "review_id": packet["review_id"],
        "phase": packet["phase"],
        "subject_digest": packet["subject_digest"],
        "packet_digest": packet["packet_digest"],
        "decision": None,
        "reviewer": None,
        "decided_at": None,
        "reason_codes": [],
        "comment": "",
        "scope": {},
        "signature": None,
    }
    (destination / packet["files"]["decision_template"]).write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / packet["files"]["annotations"]).touch()
    return check_review_packet(destination)


def compute_topology_subject_digest(
    model_path: str | Path,
    request: Mapping[str, Any] | str | Path | None = None,
    expectations: Mapping[str, Any] | str | Path | None = None,
) -> str:
    """Compute the decision-binding digest for a model and its exact request inputs."""

    _, model, request_material, _, _, expectation_material = _topology_materials(
        model_path,
        request,
        expectations,
    )
    subject = {
        "kind": "model",
        "model_digest": model.digest,
        "request_digest": request_material.digest,
        "expectations_digest": (
            None if expectation_material is None else expectation_material.digest
        ),
    }
    return _subject_digest(subject)


def compute_result_subject_digest(
    run_dir: str | Path,
    report_dir: str | Path | None = None,
) -> str:
    """Compute the decision-binding digest for every file in a run and optional report."""

    run_path = Path(run_dir).resolve()
    run_entries = _tree_entries(run_path)
    hashes = _run_manifest_hashes(run_path)
    report_entries = None
    if report_dir is not None:
        report_path = Path(report_dir).resolve()
        report_entries = _tree_entries(report_path)
    subject = {
        "kind": "run",
        "run_digest": hashes["run_sha256"],
        "model_digest": hashes["model_sha256"],
        "request_digest": hashes["request_sha256"],
        "run_artifacts_digest": _tree_digest(run_entries),
        "report_digest": None if report_entries is None else _tree_digest(report_entries),
    }
    return _subject_digest(subject)


def prepare_topology_review(
    model_path: str | Path,
    request: Mapping[str, Any] | str | Path | None = None,
    expectations: Mapping[str, Any] | str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create a pending, self-contained topology review packet.

    ``request`` and ``expectations`` may be mappings or paths. Mapping inputs are
    serialized as canonical JSON. No approval or decision is created.
    """

    source = Path(model_path).resolve()
    (
        source_material,
        model,
        request_material,
        explanation_material,
        schematic_material,
        expectation_material,
    ) = _topology_materials(source, request, expectations)
    materials = [
        source_material,
        model,
        request_material,
        explanation_material,
        schematic_material,
    ]
    if expectation_material is not None:
        materials.append(expectation_material)
    destination = _prepare_destination(out_dir, sources=(source,))
    artifacts = sorted(
        (_artifact_record(material, destination) for material in materials),
        key=lambda item: item["path"],
    )
    subject = {
        "kind": "model",
        "model_digest": model.digest,
        "request_digest": request_material.digest,
        "expectations_digest": (
            None if expectation_material is None else expectation_material.digest
        ),
    }
    packet = {
        "schema": REVIEW_SCHEMA,
        "review_id": f"review-{uuid4().hex}",
        "phase": "topology",
        "status": "pending",
        "created_at": _utc_now(),
        "subject": subject,
        "subject_digest": _subject_digest(subject),
        "artifacts": artifacts,
        "files": {
            "decision_template": "review-template.json",
            "annotations": "annotations.jsonl",
        },
    }
    return _write_packet(destination, packet)


def prepare_result_review(
    run_dir: str | Path,
    report_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Create a pending result review packet from immutable run/report snapshots."""

    run_path = Path(run_dir).resolve()
    if not (run_path / "run.json").is_file():
        raise ReviewValidationError(f"Run directory has no run.json: {run_path}")
    report_path = None if report_dir is None else Path(report_dir).resolve()
    if report_path is not None and not (report_path / "report.json").is_file():
        raise ReviewValidationError(f"Report directory has no report.json: {report_path}")
    run_entries = _tree_entries(run_path)
    hashes = _run_manifest_hashes(run_path)
    report_entries = None if report_path is None else _tree_entries(report_path)
    sources = (run_path,) if report_path is None else (run_path, report_path)
    destination = _prepare_destination(out_dir, sources=sources)
    artifacts = _copy_tree_entries(run_entries, role="run", destination=destination)
    if report_entries is not None:
        artifacts.extend(_copy_tree_entries(report_entries, role="report", destination=destination))
    artifacts.sort(key=lambda item: item["path"])
    subject = {
        "kind": "run",
        "run_digest": hashes["run_sha256"],
        "model_digest": hashes["model_sha256"],
        "request_digest": hashes["request_sha256"],
        "run_artifacts_digest": _tree_digest(run_entries),
        "report_digest": None if report_entries is None else _tree_digest(report_entries),
    }
    packet = {
        "schema": REVIEW_SCHEMA,
        "review_id": f"review-{uuid4().hex}",
        "phase": "result",
        "status": "pending",
        "created_at": _utc_now(),
        "subject": subject,
        "subject_digest": _subject_digest(subject),
        "artifacts": artifacts,
        "files": {
            "decision_template": "review-template.json",
            "annotations": "annotations.jsonl",
        },
    }
    return _write_packet(destination, packet)


def _load_json_mapping(value: Mapping[str, Any] | str | Path, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        try:
            result = json.loads(_canonical_json(dict(value)))
        except json.JSONDecodeError as exc:  # pragma: no cover - canonical JSON is valid JSON
            raise ReviewValidationError(f"Cannot normalize {name}: {exc}") from exc
        return result
    path = Path(value).resolve()
    if path.is_dir():
        path = path / "packet.json"
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewValidationError(f"Cannot read {name} JSON from {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise ReviewValidationError(f"{name} must be a JSON object")
    return result


def _review_base(value: Mapping[str, Any] | str | Path) -> Path | None:
    if isinstance(value, Mapping):
        return None
    path = Path(value).resolve()
    return path if path.is_dir() else path.parent


def _validate_artifact_record(value: Any, *, index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReviewValidationError(f"artifacts[{index}] must be an object")
    _require_exact_fields(value, _ARTIFACT_FIELDS, name=f"artifacts[{index}]")
    missing = sorted(_ARTIFACT_FIELDS - set(value))
    if missing:
        raise ReviewValidationError(
            f"artifacts[{index}]: missing field(s): {', '.join(missing)}"
        )
    if value["role"] not in {
        "source",
        "model",
        "request",
        "explanation",
        "schematic",
        "expectations",
        "run",
        "report",
    }:
        raise ReviewValidationError(f"artifacts[{index}].role is unsupported")
    _safe_relative(value["logical_path"], name=f"artifacts[{index}].logical_path")
    _safe_relative(value["path"], name=f"artifacts[{index}].path")
    _require_digest(value["sha256"], name=f"artifacts[{index}].sha256")
    if not isinstance(value["size_bytes"], int) or value["size_bytes"] < 0:
        raise ReviewValidationError(f"artifacts[{index}].size_bytes must be non-negative")
    if not isinstance(value["media_type"], str) or not value["media_type"]:
        raise ReviewValidationError(f"artifacts[{index}].media_type must be a string")
    return value


def validate_review_packet(
    review: Mapping[str, Any] | str | Path,
    *,
    verify_artifacts: bool = False,
) -> dict[str, Any]:
    """Validate review JSON and, optionally, the packet's copied artifacts."""

    packet = _load_json_mapping(review, name="review packet")
    _require_exact_fields(packet, _PACKET_FIELDS, name="review packet")
    missing = sorted(_PACKET_FIELDS - set(packet))
    if missing:
        raise ReviewValidationError(f"review packet: missing field(s): {', '.join(missing)}")
    if packet["schema"] != REVIEW_SCHEMA:
        raise ReviewValidationError(f"Unsupported review schema {packet['schema']!r}")
    if not isinstance(packet["review_id"], str) or not packet["review_id"]:
        raise ReviewValidationError("review_id must be a non-empty string")
    if packet["phase"] not in {"topology", "result"}:
        raise ReviewValidationError("phase must be topology or result")
    if packet["status"] != "pending":
        raise ReviewValidationError("Immutable review packets must have status pending")
    _validate_timestamp(packet["created_at"], name="created_at")
    if not isinstance(packet["subject"], dict):
        raise ReviewValidationError("subject must be an object")
    subject = packet["subject"]
    if packet["phase"] == "topology":
        allowed = _TOPOLOGY_SUBJECT_FIELDS
        expected_kind = "model"
        digest_fields = ("model_digest", "request_digest")
        nullable_fields = ("expectations_digest",)
    else:
        allowed = _RESULT_SUBJECT_FIELDS
        expected_kind = "run"
        digest_fields = (
            "run_digest",
            "model_digest",
            "request_digest",
            "run_artifacts_digest",
        )
        nullable_fields = ("report_digest",)
    _require_exact_fields(subject, allowed, name="subject")
    missing_subject = sorted(allowed - set(subject))
    if missing_subject:
        raise ReviewValidationError(f"subject: missing field(s): {', '.join(missing_subject)}")
    if subject["kind"] != expected_kind:
        raise ReviewValidationError(f"{packet['phase']} review has invalid subject.kind")
    for field in digest_fields:
        _require_digest(subject[field], name=f"subject.{field}")
    for field in nullable_fields:
        _require_digest(subject[field], name=f"subject.{field}", nullable=True)
    _require_digest(packet["subject_digest"], name="subject_digest")
    if packet["subject_digest"] != _subject_digest(subject):
        raise ReviewIntegrityError("subject_digest does not match the review subject")
    if not isinstance(packet["artifacts"], list) or not packet["artifacts"]:
        raise ReviewValidationError("artifacts must be a non-empty array")
    artifacts = [
        _validate_artifact_record(value, index=index)
        for index, value in enumerate(packet["artifacts"])
    ]
    paths = [value["path"] for value in artifacts]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ReviewValidationError("Artifact paths must be unique and sorted")
    if packet["phase"] == "topology":
        roles = {
            "source",
            "model",
            "request",
            "explanation",
            "schematic",
            "expectations",
            "run",
            "report",
        }
        by_role = {role: [item for item in artifacts if item["role"] == role] for role in roles}
        if len(by_role["source"]) != 1:
            raise ReviewIntegrityError("Topology packet must contain exactly one source artifact")
        if len(by_role["model"]) != 1 or by_role["model"][0]["sha256"] != subject["model_digest"]:
            raise ReviewIntegrityError("model_digest does not match the packaged model")
        if (
            len(by_role["request"]) != 1
            or by_role["request"][0]["sha256"] != subject["request_digest"]
        ):
            raise ReviewIntegrityError("request_digest does not match the packaged request")
        if len(by_role["explanation"]) != 1 or len(by_role["schematic"]) != 1:
            raise ReviewIntegrityError(
                "Topology packet must contain exactly one explanation and schematic"
            )
        for role, field in (("expectations", "expectations_digest"),):
            values = by_role[role]
            if subject[field] is None and values:
                raise ReviewIntegrityError(f"Unexpected {role} artifact")
            if subject[field] is not None and (
                len(values) != 1 or values[0]["sha256"] != subject[field]
            ):
                raise ReviewIntegrityError(f"{field} does not match the packaged artifact")
        if by_role["run"] or by_role["report"]:
            raise ReviewValidationError("Topology packets cannot contain run/report artifacts")
    else:
        run_records = [item for item in artifacts if item["role"] == "run"]
        report_records = [item for item in artifacts if item["role"] == "report"]
        if not run_records or _tree_digest(run_records) != subject["run_artifacts_digest"]:
            raise ReviewIntegrityError(
                "run_artifacts_digest does not match the packaged run tree"
            )
        if not any(item["logical_path"] == "run.json" for item in run_records):
            raise ReviewValidationError("Result packet run artifacts must contain run.json")
        if subject["report_digest"] is None and report_records:
            raise ReviewIntegrityError("Unexpected report artifacts")
        if subject["report_digest"] is not None:
            if not report_records or _tree_digest(report_records) != subject["report_digest"]:
                raise ReviewIntegrityError("report_digest does not match the packaged report tree")
            if not any(item["logical_path"] == "report.json" for item in report_records):
                raise ReviewValidationError("Report artifacts must contain report.json")
        if any(item["role"] not in {"run", "report"} for item in artifacts):
            raise ReviewValidationError("Result packets can contain only run/report artifacts")
    if not isinstance(packet["files"], dict) or set(packet["files"]) != {
        "decision_template",
        "annotations",
    }:
        raise ReviewValidationError("files must identify decision_template and annotations")
    for name, value in packet["files"].items():
        _safe_relative(value, name=f"files.{name}")
    _require_digest(packet["packet_digest"], name="packet_digest")
    body = {key: value for key, value in packet.items() if key != "packet_digest"}
    if packet["packet_digest"] != _digest_json(body):
        raise ReviewIntegrityError("packet_digest does not match packet.json")
    if verify_artifacts:
        base = _review_base(review)
        if base is None:
            raise ReviewValidationError(
                "Artifact verification requires a packet directory or packet.json path"
            )
        for index, artifact in enumerate(artifacts):
            relative = _safe_relative(artifact["path"], name=f"artifacts[{index}].path")
            path = base / relative
            if path.is_symlink() or not path.is_file():
                raise ReviewIntegrityError(f"Review artifact is missing or not regular: {relative}")
            if path.stat().st_size != artifact["size_bytes"]:
                raise ReviewIntegrityError(f"Review artifact size mismatch: {relative}")
            if _file_digest(path) != artifact["sha256"]:
                raise ReviewIntegrityError(f"Review artifact digest mismatch: {relative}")
    return packet


def validate_review_json(review: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    """Alias for structural and internal-digest validation of review JSON."""

    return validate_review_packet(review)


def _check_template(base: Path, packet: Mapping[str, Any]) -> None:
    relative = _safe_relative(packet["files"]["decision_template"], name="decision_template")
    template = _load_json_mapping(base / relative, name="review decision template")
    expected = {
        "schema": DECISION_SCHEMA,
        "review_id": packet["review_id"],
        "phase": packet["phase"],
        "subject_digest": packet["subject_digest"],
        "packet_digest": packet["packet_digest"],
    }
    for field, value in expected.items():
        if template.get(field) != value:
            raise ReviewIntegrityError(f"Review decision template has stale {field}")
    if template.get("decision") is not None:
        raise ReviewIntegrityError("review-template.json must not contain a decision")
    if template.get("reviewer") is not None or template.get("decided_at") is not None:
        raise ReviewIntegrityError("review-template.json must not impersonate a completed review")


def _check_subject_semantics(base: Path, packet: Mapping[str, Any]) -> None:
    artifacts = packet["artifacts"]
    if packet["phase"] == "topology":
        source_record = next(item for item in artifacts if item["role"] == "source")
        source_path = base / _safe_relative(source_record["path"], name="source artifact")
        model = load_model(source_path)
        semantic_digest = _sha256_bytes(_canonical_json(physics_model_dict(model)))
        if semantic_digest != packet["subject"]["model_digest"]:
            raise ReviewIntegrityError(
                "Packaged model source does not match the canonical model_digest"
            )
        artifacts_by_role = {item["role"]: item for item in artifacts}
        expected_presentation = {
            "explanation": _canonical_json(explain_model(model)),
            "schematic": render_svg(model).encode("utf-8"),
        }
        for role, expected_bytes in expected_presentation.items():
            record = artifacts_by_role[role]
            if record["sha256"] != _sha256_bytes(expected_bytes):
                raise ReviewIntegrityError(
                    f"Packaged {role} does not match the reviewed model source"
                )
        return
    run_record = next(
        item
        for item in artifacts
        if item["role"] == "run" and item["logical_path"] == "run.json"
    )
    run_path = base / _safe_relative(run_record["path"], name="run.json artifact")
    hashes = _run_manifest_hashes(run_path.parent)
    subject = packet["subject"]
    expected = {
        "run_sha256": subject["run_digest"],
        "model_sha256": subject["model_digest"],
        "request_sha256": subject["request_digest"],
    }
    for name, digest in expected.items():
        if hashes[name] != digest:
            raise ReviewIntegrityError(f"Packaged run.json has stale {name}")


def check_review_packet(review: str | Path) -> dict[str, Any]:
    """Validate packet JSON, every copied artifact, template binding, and annotations."""

    path = Path(review).resolve()
    base = path if path.is_dir() else path.parent
    packet = validate_review_packet(review, verify_artifacts=True)
    _check_subject_semantics(base, packet)
    _check_template(base, packet)
    annotation_path = base / _safe_relative(packet["files"]["annotations"], name="annotations")
    validate_review_annotations(annotation_path, review_id=packet["review_id"])
    return packet


def _validate_reviewer(value: Any) -> None:
    if not isinstance(value, dict):
        raise ReviewValidationError("decision.reviewer must be an object")
    _require_exact_fields(value, _REVIEWER_FIELDS, name="decision.reviewer")
    if value.get("kind") != "human":
        raise ReviewValidationError("Only a human reviewer may author a review decision")
    if not isinstance(value.get("id"), str) or not value["id"]:
        raise ReviewValidationError("decision.reviewer.id must be a non-empty string")
    for field in _REVIEWER_FIELDS - {"kind", "id"}:
        if field in value and not isinstance(value[field], str):
            raise ReviewValidationError(f"decision.reviewer.{field} must be a string")


def validate_review_decision(
    decision: Mapping[str, Any] | str | Path,
    review: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    """Validate a pre-existing human decision and bind it to a review subject."""

    packet = validate_review_packet(review)
    value = _load_json_mapping(decision, name="review decision")
    _require_exact_fields(value, _DECISION_FIELDS, name="review decision")
    required = {
        "schema",
        "review_id",
        "phase",
        "subject_digest",
        "decision",
        "reviewer",
        "decided_at",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ReviewValidationError(f"review decision: missing field(s): {', '.join(missing)}")
    if value["schema"] != DECISION_SCHEMA:
        raise ReviewValidationError(f"Unsupported decision schema {value['schema']!r}")
    if value["decision"] not in _DECISIONS:
        raise ReviewValidationError(f"Unsupported review decision {value['decision']!r}")
    _require_digest(value["subject_digest"], name="decision.subject_digest")
    if value["review_id"] != packet["review_id"]:
        raise StaleReviewError("Decision review_id does not match the review packet")
    if value["phase"] != packet["phase"]:
        raise StaleReviewError("Decision phase does not match the review packet")
    if value["subject_digest"] != packet["subject_digest"]:
        raise StaleReviewError("Decision subject_digest is stale")
    if "packet_digest" in value:
        _require_digest(value["packet_digest"], name="decision.packet_digest")
        if value["packet_digest"] != packet["packet_digest"]:
            raise StaleReviewError("Decision packet_digest is stale")
    _validate_reviewer(value["reviewer"])
    _validate_timestamp(value["decided_at"], name="decision.decided_at")
    if "reason_codes" in value and (
        not isinstance(value["reason_codes"], list)
        or any(not isinstance(item, str) or not item for item in value["reason_codes"])
    ):
        raise ReviewValidationError("decision.reason_codes must be an array of strings")
    if "comment" in value and not isinstance(value["comment"], str):
        raise ReviewValidationError("decision.comment must be a string")
    for field in ("scope", "signature"):
        if field in value and value[field] is not None and not isinstance(value[field], dict):
            raise ReviewValidationError(f"decision.{field} must be an object or null")
    _canonical_json(value)
    return value


def check_review_decision(
    decision: Mapping[str, Any] | str | Path,
    review: Mapping[str, Any] | str | Path,
    *,
    current_subject_digest: str | None = None,
    require_approval: bool = True,
) -> dict[str, Any]:
    """Check packet integrity and whether an existing decision authorizes use.

    Pass a freshly computed ``current_subject_digest`` immediately before the
    action to detect a model/run changed after packet preparation.
    """

    if isinstance(review, Mapping):
        packet = validate_review_packet(review)
    else:
        packet = check_review_packet(review)
    value = validate_review_decision(decision, packet)
    if current_subject_digest is not None:
        _require_digest(current_subject_digest, name="current_subject_digest")
        if current_subject_digest != packet["subject_digest"]:
            raise StaleReviewError("Current model/run subject differs from the reviewed subject")
    if require_approval and value["decision"] != "approve":
        raise ReviewDecisionError(
            f"Review decision {value['decision']!r} does not authorize the requested action"
        )
    return value


__all__ = [
    "DECISION_SCHEMA",
    "REVIEW_SCHEMA",
    "ReviewDecisionError",
    "ReviewIntegrityError",
    "ReviewValidationError",
    "StaleReviewError",
    "append_review_annotation",
    "check_review_decision",
    "check_review_packet",
    "compute_result_subject_digest",
    "compute_topology_subject_digest",
    "prepare_result_review",
    "prepare_topology_review",
    "validate_review_annotations",
    "validate_review_decision",
    "validate_review_json",
    "validate_review_packet",
]
