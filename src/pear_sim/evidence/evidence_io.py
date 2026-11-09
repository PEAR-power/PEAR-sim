# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Immutable run-to-evidence artifact workflow for agent clients."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from pear_sim._jsonutil import (
    reject_json_constant as _reject_json_constant,
    require_finite as _require_finite,
    unique_json_object as _unique_json_object,
    write_json as _write_json,
)
from pear_sim.run.contracts import sha256_file, sha256_json
from pear_sim.evidence.expectations import evaluate_expectations
from pear_sim.reporting.report import (
    ReportError,
    _artifact_path,
    _read_and_verify_artifact_manifest,
    _read_json_object,
)


CONFIDENCE_SCHEMA = "pear-sim/confidence/v0.1"
EVIDENCE_BUNDLE_SCHEMA = "pear-sim/evidence-bundle/v0.1"
MAX_EXPECTATIONS_BYTES = 2 * 1024 * 1024
MAX_METRICS_BYTES = 64 * 1024 * 1024
SUPPORTED_RUN_SCHEMA = "pear-sim/run/v0.1"
SUPPORTED_RUN_BACKEND = "reference-python"


@dataclass(frozen=True)
class EvidenceArtifacts:
    evidence_json: Path
    confidence_json: Path
    artifact_manifest: Path
    blocking_status: str


def _read_strict_json(path: Path, *, role: str, maximum_bytes: int) -> dict[str, Any]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Cannot inspect {role} file {path}: {exc}") from exc
    if size > maximum_bytes:
        raise ValueError(f"{role} file exceeds the {maximum_bytes}-byte contract limit: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
            object_pairs_hook=_unique_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Cannot read {role} JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} JSON must contain an object")
    _require_finite(value)
    return value


def _prepare_destination(run_dir: Path, output_dir: str | Path) -> Path:
    destination = Path(output_dir).resolve()
    try:
        destination.relative_to(run_dir)
    except ValueError:
        pass
    else:
        raise ValueError("Evidence output directory must be outside the immutable run tree")
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Evidence output is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError(
                f"Evidence output directory is not empty: {destination}. "
                "Choose a new directory so prior evidence is never overwritten."
            )
    else:
        destination.mkdir(parents=True)
    return destination


def evaluate_run(
    run_dir: str | Path,
    expectations_file: str | Path,
    output_dir: str | Path,
) -> EvidenceArtifacts:
    """Evaluate a verified immutable run bundle against user-declared expectations."""

    directory = Path(run_dir).resolve()
    if not directory.is_dir():
        raise ValueError(f"Run directory does not exist: {directory}")
    run_path = directory / "run.json"
    manifest_path = directory / "artifact-manifest.json"
    if not manifest_path.is_file():
        raise ReportError(
            "Evidence evaluation requires artifact-manifest.json; an unverified run "
            "cannot produce trusted evidence."
        )
    run = _read_json_object(run_path, "run manifest")
    _read_and_verify_artifact_manifest(manifest_path, directory, run)
    if run.get("schema") != SUPPORTED_RUN_SCHEMA:
        raise ValueError(
            f"Unsupported run schema {run.get('schema')!r}; expected {SUPPORTED_RUN_SCHEMA!r}"
        )
    if run.get("backend") != SUPPORTED_RUN_BACKEND:
        raise ValueError(
            f"Unsupported run backend {run.get('backend')!r}; expected {SUPPORTED_RUN_BACKEND!r}"
        )
    files = run.get("files")
    if not isinstance(files, dict) or "metrics" not in files:
        raise ValueError("run.json must identify a metrics artifact")
    metrics_path = _artifact_path(directory, files["metrics"], "metrics")
    metrics = _read_strict_json(
        metrics_path,
        role="metrics",
        maximum_bytes=MAX_METRICS_BYTES,
    )
    expectations_path = Path(expectations_file).resolve()
    expectations = _read_strict_json(
        expectations_path,
        role="expectations",
        maximum_bytes=MAX_EXPECTATIONS_BYTES,
    )
    evidence = evaluate_expectations(expectations, metrics)
    destination = _prepare_destination(directory, output_dir)
    run_sha256 = run.get("run_sha256")
    evidence["source"] = {
        "run_sha256": run_sha256,
        "run_manifest_sha256": sha256_file(run_path),
        "run_artifact_manifest_sha256": sha256_file(manifest_path),
        "run_artifact_manifest_verified": True,
        "metrics_file": metrics_path.name,
        "metrics_sha256": sha256_file(metrics_path),
        "expectations_source_sha256": sha256_file(expectations_path),
        "expectations_semantic_sha256": evidence["expectations"]["sha256"],
    }
    evidence_sha256 = sha256_json(evidence)
    confidence = {
        "schema": CONFIDENCE_SCHEMA,
        "kind": "agent_confidence_card",
        "level": "EXPLORATORY",
        "acceptance": {
            "blocking_status": evidence["blocking_status"],
            "overall_status": evidence["overall_status"],
            "evidence_sha256": evidence_sha256,
        },
        "envelope": {
            "status": "not_established_by_user_expectations",
            "statement": (
                "Passing user-declared KPI criteria does not establish a verified "
                "capability or parameter envelope."
            ),
        },
        "claim": {
            "validation_claim": False,
            "compatibility_claim": False,
            "hardware_sign_off_allowed": False,
        },
    }
    evidence_path = destination / "evidence.json"
    confidence_path = destination / "confidence.json"
    artifact_manifest_path = destination / "artifact-manifest.json"
    _write_json(evidence_path, evidence)
    _write_json(confidence_path, confidence)
    _write_json(
        artifact_manifest_path,
        {
            "schema": "pear-sim/artifact-manifest/v0.1",
            "subject": {
                "kind": "evidence",
                "sha256": evidence_sha256,
                "run_sha256": run_sha256,
            },
            "bundle_schema": EVIDENCE_BUNDLE_SCHEMA,
            "artifacts": [
                {
                    "path": path.name,
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for path in (confidence_path, evidence_path)
            ],
        },
    )
    return EvidenceArtifacts(
        evidence_json=evidence_path,
        confidence_json=confidence_path,
        artifact_manifest=artifact_manifest_path,
        blocking_status=str(evidence["blocking_status"]),
    )


__all__ = [
    "CONFIDENCE_SCHEMA",
    "EVIDENCE_BUNDLE_SCHEMA",
    "EvidenceArtifacts",
    "SUPPORTED_RUN_BACKEND",
    "SUPPORTED_RUN_SCHEMA",
    "evaluate_run",
]
