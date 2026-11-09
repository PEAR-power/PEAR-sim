# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Append-only, hash-chained annotations for PEARSim review packets."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from pear_sim.review.review_errors import ReviewIntegrityError, ReviewValidationError


ANNOTATION_SCHEMA = "pear-sim/review-annotation/v0.1"
_PACKET_SCHEMA = "pear-sim/review/v0.1"
_EVENTS = frozenset({"comment", "reply", "acknowledge", "resolve", "reopen"})
_SEVERITIES = frozenset({"info", "warning", "blocker"})
_CATEGORIES = frozenset(
    {
        "topology",
        "polarity",
        "units",
        "initial_condition",
        "control",
        "numerical",
        "performance",
        "provenance",
        "result_interpretation",
        "safety",
    }
)
_INPUT_FIELDS = frozenset(
    {
        "id",
        "event",
        "author",
        "target",
        "severity",
        "category",
        "message",
        "suggested_action",
        "in_reply_to",
    }
)
_RECORD_FIELDS = _INPUT_FIELDS | {
    "schema",
    "review_id",
    "sequence",
    "created_at",
    "previous_digest",
    "annotation_digest",
}
_AUTHOR_FIELDS = frozenset({"kind", "id", "display_name"})
_TARGET_FIELDS = frozenset(
    {
        "kind",
        "artifact",
        "json_pointer",
        "component_id",
        "port",
        "node",
        "output",
        "time_range_s",
        "event_id",
    }
)


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
        raise ReviewValidationError(f"Annotation is not canonical JSON: {exc}") from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical_json(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReviewValidationError(f"Cannot read JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReviewValidationError(f"{path} must contain a JSON object")
    return value


def _packet_context(review_dir: Path) -> tuple[str, Path]:
    packet = _load_json(review_dir / "packet.json")
    if packet.get("schema") != _PACKET_SCHEMA:
        raise ReviewValidationError(f"{review_dir}/packet.json is not a PEARSim review packet")
    review_id = packet.get("review_id")
    if not isinstance(review_id, str) or not review_id:
        raise ReviewValidationError("Review packet has no valid review_id")
    files = packet.get("files")
    relative = files.get("annotations") if isinstance(files, dict) else None
    if not isinstance(relative, str) or not relative:
        raise ReviewValidationError("Review packet has no annotations file")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ReviewValidationError("Review annotations path must be bundle-relative")
    return review_id, review_dir / candidate


def _validate_timestamp(value: Any, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ReviewValidationError(f"{name} must be a non-empty RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ReviewValidationError(f"{name} is not a valid RFC 3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ReviewValidationError(f"{name} must include a timezone")


def _require_exact_fields(value: Mapping[str, Any], allowed: frozenset[str], *, name: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ReviewValidationError(f"{name}: unknown field(s): {', '.join(unknown)}")


def _validate_author(value: Any) -> None:
    if not isinstance(value, dict):
        raise ReviewValidationError("annotation.author must be an object")
    _require_exact_fields(value, _AUTHOR_FIELDS, name="annotation.author")
    if value.get("kind") not in {"human", "agent", "system"}:
        raise ReviewValidationError("annotation.author.kind must be human, agent, or system")
    if not isinstance(value.get("id"), str) or not value["id"]:
        raise ReviewValidationError("annotation.author.id must be a non-empty string")
    if "display_name" in value and not isinstance(value["display_name"], str):
        raise ReviewValidationError("annotation.author.display_name must be a string")


def _validate_target(value: Any) -> None:
    if not isinstance(value, dict):
        raise ReviewValidationError("annotation.target must be an object")
    _require_exact_fields(value, _TARGET_FIELDS, name="annotation.target")
    if not isinstance(value.get("kind"), str) or not value["kind"]:
        raise ReviewValidationError("annotation.target.kind must be a non-empty string")
    for name in _TARGET_FIELDS - {"kind", "time_range_s"}:
        if name in value and (not isinstance(value[name], str) or not value[name]):
            raise ReviewValidationError(f"annotation.target.{name} must be a non-empty string")
    if "time_range_s" in value:
        time_range = value["time_range_s"]
        if (
            not isinstance(time_range, list)
            or len(time_range) != 2
            or any(not isinstance(item, (int, float)) for item in time_range)
            or time_range[0] > time_range[1]
        ):
            raise ReviewValidationError(
                "annotation.target.time_range_s must be an ordered pair of seconds"
            )


def _validate_record(
    record: Any,
    *,
    expected_review_id: str | None,
    expected_sequence: int,
    expected_previous: str | None,
) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ReviewValidationError("Each annotation line must be a JSON object")
    _require_exact_fields(record, _RECORD_FIELDS, name="annotation")
    required = _RECORD_FIELDS - {"suggested_action", "in_reply_to"}
    missing = sorted(required - set(record))
    if missing:
        raise ReviewValidationError(f"annotation: missing field(s): {', '.join(missing)}")
    if record.get("schema") != ANNOTATION_SCHEMA:
        raise ReviewValidationError(f"Unsupported annotation schema {record.get('schema')!r}")
    if not isinstance(record.get("review_id"), str) or not record["review_id"]:
        raise ReviewValidationError("annotation.review_id must be a non-empty string")
    if expected_review_id is not None and record["review_id"] != expected_review_id:
        raise ReviewIntegrityError("Annotation belongs to a different review packet")
    if record.get("sequence") != expected_sequence:
        raise ReviewIntegrityError(
            f"Annotation sequence is not append-only at entry {expected_sequence}"
        )
    if record.get("previous_digest") != expected_previous:
        raise ReviewIntegrityError(f"Annotation hash chain is broken at entry {expected_sequence}")
    if not isinstance(record.get("id"), str) or not record["id"]:
        raise ReviewValidationError("annotation.id must be a non-empty string")
    if record.get("event") not in _EVENTS:
        raise ReviewValidationError(f"Unsupported annotation event {record.get('event')!r}")
    if record.get("severity") not in _SEVERITIES:
        raise ReviewValidationError(f"Unsupported annotation severity {record.get('severity')!r}")
    if record.get("category") not in _CATEGORIES:
        raise ReviewValidationError(f"Unsupported annotation category {record.get('category')!r}")
    if not isinstance(record.get("message"), str) or not record["message"]:
        raise ReviewValidationError("annotation.message must be a non-empty string")
    if "suggested_action" in record and not isinstance(record["suggested_action"], str):
        raise ReviewValidationError("annotation.suggested_action must be a string")
    if record["event"] != "comment":
        if not isinstance(record.get("in_reply_to"), str) or not record["in_reply_to"]:
            raise ReviewValidationError(
                f"annotation.in_reply_to is required for event {record['event']!r}"
            )
    elif "in_reply_to" in record and not isinstance(record["in_reply_to"], str):
        raise ReviewValidationError("annotation.in_reply_to must be a string")
    _validate_author(record["author"])
    _validate_target(record["target"])
    _validate_timestamp(record["created_at"], name="annotation.created_at")
    supplied_digest = record.get("annotation_digest")
    body = {key: value for key, value in record.items() if key != "annotation_digest"}
    if supplied_digest != _digest(body):
        raise ReviewIntegrityError(f"Annotation digest mismatch at entry {expected_sequence}")
    return record


def validate_review_annotations(
    review_or_log: str | Path,
    *,
    review_id: str | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate an annotation JSONL file and its append-only hash chain."""

    path = Path(review_or_log).resolve()
    if path.is_dir():
        packet_review_id, path = _packet_context(path)
        if review_id is not None and review_id != packet_review_id:
            raise ReviewIntegrityError("Requested review_id differs from packet review_id")
        review_id = packet_review_id
    if not path.exists():
        raise ReviewValidationError(f"Annotation log does not exist: {path}")
    records: list[dict[str, Any]] = []
    previous: str | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ReviewValidationError(f"Cannot read annotation log {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ReviewValidationError(f"Blank line in annotation log at line {line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReviewValidationError(
                f"Invalid JSON in annotation log at line {line_number}: {exc}"
            ) from exc
        record = _validate_record(
            value,
            expected_review_id=review_id,
            expected_sequence=line_number,
            expected_previous=previous,
        )
        if review_id is None:
            review_id = record["review_id"]
        previous = record["annotation_digest"]
        records.append(record)
    return tuple(records)


def append_review_annotation(
    review_dir: str | Path,
    annotation: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one validated annotation event to a review packet's hash chain."""

    destination = Path(review_dir).resolve()
    review_id, log_path = _packet_context(destination)
    existing = validate_review_annotations(log_path, review_id=review_id)
    if not isinstance(annotation, Mapping):
        raise ReviewValidationError("annotation must be an object")
    value = dict(annotation)
    _require_exact_fields(value, _INPUT_FIELDS, name="annotation input")
    required = {"event", "author", "target", "severity", "category", "message"}
    missing = sorted(required - set(value))
    if missing:
        raise ReviewValidationError(f"annotation input: missing field(s): {', '.join(missing)}")
    record: dict[str, Any] = {
        "schema": ANNOTATION_SCHEMA,
        "id": value.get("id", f"ann-{uuid4().hex}"),
        "review_id": review_id,
        "sequence": len(existing) + 1,
        "event": value["event"],
        "author": value["author"],
        "target": value["target"],
        "severity": value["severity"],
        "category": value["category"],
        "message": value["message"],
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "previous_digest": existing[-1]["annotation_digest"] if existing else None,
    }
    for optional in ("suggested_action", "in_reply_to"):
        if optional in value:
            record[optional] = value[optional]
    record["annotation_digest"] = _digest(record)
    _validate_record(
        record,
        expected_review_id=review_id,
        expected_sequence=len(existing) + 1,
        expected_previous=record["previous_digest"],
    )
    line = json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    try:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
    except OSError as exc:
        raise ReviewValidationError(f"Cannot append annotation to {log_path}: {exc}") from exc
    return record


__all__ = [
    "ANNOTATION_SCHEMA",
    "append_review_annotation",
    "validate_review_annotations",
]
