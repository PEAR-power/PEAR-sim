# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Topology-neutral retained-point machinery for verification benchmarks.

The harness owns artifact retention and fail-closed execution semantics only.  It
deliberately does not import the simulator, a benchmark oracle, or topology-specific
equations.  Each benchmark supplies those pieces through callbacks so a shared bug in
the production kernel cannot silently become shared oracle mathematics.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, TypeVar

from pear_sim._jsonutil import write_json as _shared_write_json
from pear_sim.run.contracts import sha256_file


MAX_RETAINED_ERROR_TYPE_BYTES = 256
MAX_RETAINED_ERROR_MESSAGE_BYTES = 1024
MAX_RETAINED_ERROR_REASON_BYTES = 1024


class BoundaryReference(Protocol):
    """Minimum topology-independent identity required by the retained harness."""

    point_id: str
    load_ratio_to_bcm: float
    load_ohm: float
    regime: str

    def as_dict(self) -> dict[str, Any]: ...


class TimedResult(Protocol):
    stats: Mapping[str, Any]


ReferenceT = TypeVar("ReferenceT", bound=BoundaryReference)
ObservationT = TypeVar("ObservationT")
ResultT = TypeVar("ResultT", bound=TimedResult)


@dataclass(frozen=True)
class BoundaryHarnessContract:
    """Versioned point-artifact and fail-closed behavior for one benchmark family."""

    point_schema: str
    comparison_schema: str
    gate_ids: tuple[str, ...]
    failure_includes_comparison_schema: bool = True
    bounded_error_text: bool = False

    def __post_init__(self) -> None:
        if not self.point_schema or not self.comparison_schema:
            raise ValueError("Boundary harness schemas must be non-empty")
        if not self.gate_ids or any(not gate_id for gate_id in self.gate_ids):
            raise ValueError("Boundary harness gate IDs must be non-empty")
        if len(set(self.gate_ids)) != len(self.gate_ids):
            raise ValueError("Boundary harness gate IDs must be unique")


@dataclass(frozen=True)
class RetainedBoundaryPoints:
    """Ordered comparison and human-summary records for every requested point."""

    comparisons: tuple[dict[str, Any], ...]
    summaries: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class BoundaryAggregate:
    """Status counters derived from retained comparisons without recomputation."""

    status: str
    gate_status_counts: dict[str, int]
    point_status_counts: dict[str, int]


def write_json(path: Path, payload: Any) -> None:
    """Atomically write deterministic, finite JSON used by benchmark artifacts."""

    _shared_write_json(path, payload)


def prepare_destination(output_dir: str | Path) -> Path:
    """Create a new evidence directory without overwriting prior evidence."""

    destination = Path(output_dir).resolve()
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Sweep output is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError(
                f"Sweep output directory is not empty: {destination}. "
                "Choose a new directory so prior sweep evidence is never overwritten."
            )
    else:
        destination.mkdir(parents=True)
    return destination


def _truncate_utf8(value: str, maximum_bytes: int) -> str:
    sanitized = value.encode("utf-8", errors="replace").decode("utf-8")
    encoded = sanitized.encode("utf-8")
    if len(encoded) <= maximum_bytes:
        return sanitized
    marker = "…".encode()
    if maximum_bytes < len(marker):
        return encoded[:maximum_bytes].decode("utf-8", errors="ignore")
    prefix = encoded[: maximum_bytes - len(marker)]
    while prefix:
        try:
            return prefix.decode("utf-8") + "…"
        except UnicodeDecodeError:
            prefix = prefix[:-1]
    return "…"


def exception_record(
    exc: Exception,
    *,
    bounded: bool,
) -> tuple[dict[str, str], str]:
    """Project an exception without ever turning failed execution into a pass."""

    if bounded:
        error_type = _truncate_utf8(type(exc).__name__, MAX_RETAINED_ERROR_TYPE_BYTES)
        message = _truncate_utf8(str(exc), MAX_RETAINED_ERROR_MESSAGE_BYTES)
        reason = _truncate_utf8(
            f"{error_type}: {message}",
            MAX_RETAINED_ERROR_REASON_BYTES,
        )
    else:
        error_type = type(exc).__name__
        message = str(exc)
        reason = f"{error_type}: {message}"
    return {"type": error_type, "message": message}, reason


def _not_run_comparison(
    contract: BoundaryHarnessContract,
    reference: BoundaryReference,
    reason: str,
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "point_id": reference.point_id,
        "load_ratio_to_bcm": reference.load_ratio_to_bcm,
        "load_ohm": reference.load_ohm,
        "expected_regime": reference.regime,
        "status": "not-run",
        "gates": [
            {
                "id": gate_id,
                "actual": None,
                "limit": None,
                "status": "not-run",
                "reason": reason,
                "expected_regime": reference.regime,
            }
            for gate_id in contract.gate_ids
        ],
    }
    if contract.failure_includes_comparison_schema:
        comparison = {"schema": contract.comparison_schema, **comparison}
    return comparison


def _run_reference(
    destination: Path,
    points_root: Path,
    index: int,
    reference: ReferenceT,
    *,
    contract: BoundaryHarnessContract,
    simulate: Callable[[ReferenceT], ResultT],
    observe: Callable[[ResultT, ReferenceT], ObservationT],
    compare: Callable[[ReferenceT, ObservationT], dict[str, Any]],
    retain_run: Callable[[ResultT, Path], Path],
    initial_conditions: Callable[[ReferenceT], dict[str, Any]],
    observation_payload: Callable[[ObservationT], dict[str, Any]],
    summary_metrics: Callable[[ObservationT | None], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    point_dir = points_root / f"{index:03d}-{reference.point_id}"
    point_dir.mkdir()
    observation: ObservationT | None = None
    run_path: Path | None = None
    runtime_s: float | None = None
    error: dict[str, str] | None = None
    try:
        result = simulate(reference)
        runtime_s = float(result.stats["wall_time_s"])
        observation = observe(result, reference)
        comparison = compare(reference, observation)
        run_path = retain_run(result, point_dir / "run")
    except Exception as exc:  # retain every point; execution errors can never pass
        error, reason = exception_record(exc, bounded=contract.bounded_error_text)
        comparison = _not_run_comparison(contract, reference, reason)

    run_record: dict[str, Any] | None = None
    if run_path is not None:
        run_manifest = run_path / "run.json"
        run_record = {
            "path": run_manifest.relative_to(destination).as_posix(),
            "run_sha256": json.loads(run_manifest.read_text(encoding="utf-8"))["run_sha256"],
        }
    point_record = {
        "schema": contract.point_schema,
        "index": index,
        "point_id": reference.point_id,
        "status": comparison["status"],
        "initial_conditions": initial_conditions(reference),
        "reference": reference.as_dict(),
        "observation": None if observation is None else observation_payload(observation),
        "comparison": comparison,
        "error": error,
        "run": run_record,
    }
    point_artifact = point_dir / "benchmark-point.json"
    write_json(point_artifact, point_record)

    artifacts: dict[str, Any] = {
        "benchmark_point": {
            "path": point_artifact.relative_to(destination).as_posix(),
            "sha256": sha256_file(point_artifact),
        }
    }
    if run_path is not None:
        run_manifest = run_path / "run.json"
        artifacts["run"] = {
            "path": run_manifest.relative_to(destination).as_posix(),
            "sha256": sha256_file(run_manifest),
        }
    summary: dict[str, Any] = {
        "index": index,
        "point_id": reference.point_id,
        "load_ratio_to_bcm": reference.load_ratio_to_bcm,
        "load_ohm": reference.load_ohm,
        "expected_regime": reference.regime,
        "status": comparison["status"],
        "runtime_s": runtime_s,
        "metrics": summary_metrics(observation),
        "gates": comparison["gates"],
        "artifacts": artifacts,
    }
    if observation is not None:
        observed_regime = getattr(observation, "regime")
        summary["observed_regime"] = observed_regime
        summary["regime"] = observed_regime
    return comparison, summary


def run_retained_boundary_points(
    destination: Path,
    references: Sequence[ReferenceT],
    *,
    contract: BoundaryHarnessContract,
    simulate: Callable[[ReferenceT], ResultT],
    observe: Callable[[ResultT, ReferenceT], ObservationT],
    compare: Callable[[ReferenceT, ObservationT], dict[str, Any]],
    retain_run: Callable[[ResultT, Path], Path],
    initial_conditions: Callable[[ReferenceT], dict[str, Any]],
    observation_payload: Callable[[ObservationT], dict[str, Any]],
    summary_metrics: Callable[[ObservationT | None], dict[str, Any]],
) -> RetainedBoundaryPoints:
    """Execute and retain every point in input order under a fail-closed contract."""

    points_root = destination / "points"
    points_root.mkdir()
    comparisons: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for index, reference in enumerate(references):
        comparison, summary = _run_reference(
            destination,
            points_root,
            index,
            reference,
            contract=contract,
            simulate=simulate,
            observe=observe,
            compare=compare,
            retain_run=retain_run,
            initial_conditions=initial_conditions,
            observation_payload=observation_payload,
            summary_metrics=summary_metrics,
        )
        comparisons.append(comparison)
        summaries.append(summary)
    return RetainedBoundaryPoints(tuple(comparisons), tuple(summaries))


def build_boundary_spec(
    *,
    schema: str,
    benchmark_id: str,
    claim: dict[str, Any],
    source_model: dict[str, Any],
    parameters_si: dict[str, Any],
    load_ratios: Sequence[float],
    limits: dict[str, Any],
    oracle_schema: str,
    comparison_schema: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared one-cycle, oracle-seeded boundary-sweep specification."""

    payload = {
        "schema": schema,
        "benchmark_id": benchmark_id,
        "claim": claim,
        "source_model": source_model,
        "parameters_si": parameters_si,
        "load_ratios": list(load_ratios),
        "initial_conditions": {
            "source": "independent periodic affine oracle",
            "applied_per_point": True,
            "source_model_initial_conditions_used": False,
        },
        "simulation": {
            "cycles": 1,
            "solver": "dopri5",
            "rtol": 1e-10,
            "atol": 1e-12,
            "max_step_over_period": 0.01,
            "output_step_over_period": 0.005,
        },
        "limits": limits,
        "oracle_schema": oracle_schema,
        "comparison_schema": comparison_schema,
    }
    if extra_fields:
        overlap = set(payload).intersection(extra_fields)
        if overlap:
            raise ValueError(
                "Boundary specification extra fields cannot replace shared fields: "
                + ", ".join(sorted(overlap))
            )
        payload.update(extra_fields)
    return payload


def aggregate_boundary_points(retained: RetainedBoundaryPoints) -> BoundaryAggregate:
    gate_counts = Counter(
        gate["status"]
        for comparison in retained.comparisons
        for gate in comparison["gates"]
    )
    point_counts = Counter(point["status"] for point in retained.summaries)
    return BoundaryAggregate(
        status=overall_status(retained.summaries),
        gate_status_counts=dict(sorted(gate_counts.items())),
        point_status_counts=dict(sorted(point_counts.items())),
    )


def build_boundary_runner_result(
    *,
    schema: str,
    benchmark_id: str,
    oracle_schema: str,
    comparison_schema: str,
    claim: dict[str, Any],
    parameters_si: dict[str, Any],
    boundary_load_ohm: float,
    retained: RetainedBoundaryPoints,
    aggregate: BoundaryAggregate,
) -> dict[str, Any]:
    return {
        "schema": schema,
        "benchmark_id": benchmark_id,
        "oracle_schema": oracle_schema,
        "comparison_schema": comparison_schema,
        "claim": claim,
        "parameters_si": parameters_si,
        "boundary_load_ohm": boundary_load_ohm,
        "summary": {
            "status": aggregate.status,
            "point_count": len(retained.comparisons),
            "gate_count": sum(aggregate.gate_status_counts.values()),
            "gate_status_counts": aggregate.gate_status_counts,
        },
        "points": list(retained.comparisons),
    }


def build_boundary_sweep_summary(
    *,
    schema: str,
    benchmark_id: str,
    claim: dict[str, Any],
    boundary_load_ohm: float,
    retained: RetainedBoundaryPoints,
    aggregate: BoundaryAggregate,
    artifacts: dict[str, Any],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    summaries = list(retained.summaries)
    return {
        "schema": schema,
        "benchmark_id": benchmark_id,
        "overall_status": aggregate.status,
        "claim": claim,
        "bcm": {
            "r_bcm_ohm": boundary_load_ohm,
            "method": "independent periodic CCM fixed-point zero-current root",
        },
        "summary": {
            "status": aggregate.status,
            "point_count": len(summaries),
            "point_status_counts": aggregate.point_status_counts,
            "gate_status_counts": aggregate.gate_status_counts,
        },
        "points": summaries,
        "worst_gates": worst_gates(summaries),
        "failures": failures(summaries),
        "artifacts": artifacts,
        "provenance": provenance,
    }


def regime_coverage(
    references: Sequence[BoundaryReference],
    summaries: Sequence[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    regime_order = {"CCM": 0, "BCM": 1, "DCM": 2}
    requested = sorted(
        {reference.regime for reference in references},
        key=regime_order.__getitem__,
    )
    completed = sorted(
        {
            str(point["observed_regime"])
            for point in summaries
            if point.get("observed_regime") in regime_order
        },
        key=regime_order.__getitem__,
    )
    return requested, completed


def build_boundary_confidence(
    *,
    schema: str,
    status: str,
    sweep_summary_sha256: str,
    benchmark_result_sha256: str,
    topology: str,
    requested_regimes: Sequence[str],
    completed_regimes: Sequence[str],
    load_ratios: Sequence[float],
    parameters_si: dict[str, Any],
    envelope_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    envelope = {
        "topology": topology,
        "requested_regimes": list(requested_regimes),
        "completed_regimes": list(completed_regimes),
        "load_ratio_to_bcm": [min(load_ratios), max(load_ratios)],
        "parameters_si": parameters_si,
        "cross_platform_evidence": "not-run",
        "convergence_sweep": "not-run",
        "output_grid_independence_study": "not-run",
    }
    if envelope_extra:
        overlap = set(envelope).intersection(envelope_extra)
        if overlap:
            raise ValueError(
                "Boundary confidence extras cannot replace shared fields: "
                + ", ".join(sorted(overlap))
            )
        envelope.update(envelope_extra)
    return {
        "schema": schema,
        "level": "EXPLORATORY",
        "acceptance": {
            "status": status,
            "sweep_summary_sha256": sweep_summary_sha256,
            "benchmark_result_sha256": benchmark_result_sha256,
        },
        "envelope": envelope,
        "related_regression_coverage": {
            "binding": "informational-only; not evidence for this run parameter set",
            "canonical_fixture_selected_solver_settings": "covered-by-regression-suite",
            "canonical_fixture_output_grid_independence": "covered-by-regression-suite",
        },
        "claim": {
            "validation_claim": False,
            "compatibility_claim": False,
            "hardware_sign_off_allowed": False,
            "general_dae_support_claimed": False,
        },
    }


def gate_unit(gate_id: str) -> str:
    return {
        "regime": "exact",
        "boundary_state_nonzero": "relative",
        "boundary_state_zero": "I_base/Vin normalized",
        "natural_event_time": "T_switch",
        "reverse_current": "I_base",
        "cycle_mean_output": "relative",
        "mode_fractions": "fraction",
        "energy_balance": "relative",
        "cycle_periodicity": "I_base/Vin normalized",
        "unclassified_fraction": "fraction",
        "mode_fraction_sum": "fraction",
        "state_ripple": "relative",
        "event_semantics": "exact",
        "signed_cycle_mean_output": "relative",
        "signed_output_polarity": "Vin normalized",
    }.get(gate_id, "dimensionless")


def _gate_score(gate: dict[str, Any]) -> float:
    if gate.get("status") == "fail":
        base = 1e12
    elif gate.get("status") == "not-run":
        base = 1e9
    else:
        base = 0.0
    actual = gate.get("actual")
    limit = gate.get("limit")
    if (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and math.isfinite(float(actual))
        and isinstance(limit, (int, float))
        and not isinstance(limit, bool)
        and math.isfinite(float(limit))
    ):
        denominator = max(abs(float(limit)), 1e-300)
        return base + abs(float(actual)) / denominator
    return base


def worst_gates(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for point in points:
        for gate in point["gates"]:
            grouped[str(gate["id"])].append((point, gate))
    records: list[dict[str, Any]] = []
    for gate_id in sorted(grouped):
        point, gate = max(
            grouped[gate_id],
            key=lambda item: (_gate_score(item[1]), -int(item[0]["index"])),
        )
        actual = gate.get("actual")
        reference = gate.get("reference")
        detail = gate.get("reason")
        if detail is None and reference is not None:
            detail = f"recorded reference={reference}"
        records.append(
            {
                "gate": gate_id,
                "status": gate["status"],
                "value": actual,
                "limit": gate.get("limit"),
                "unit": gate_unit(gate_id),
                "point_id": point["point_id"],
                "message": detail,
            }
        )
    return records


def failures(points: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for point in points:
        for gate in point["gates"]:
            if gate["status"] == "pass":
                continue
            records.append(
                {
                    "id": f"{point['point_id']}:{gate['id']}",
                    "status": gate["status"],
                    "point_id": point["point_id"],
                    "gate": gate["id"],
                    "message": gate.get("reason")
                    or "Recorded benchmark gate did not pass.",
                    "artifacts": point["artifacts"],
                }
            )
    return records


def overall_status(points: Sequence[dict[str, Any]]) -> str:
    statuses = {point["status"] for point in points}
    if "fail" in statuses:
        return "fail"
    if "not-run" in statuses:
        return "not-run"
    return "pass" if points else "not-run"


def root_manifest(
    destination: Path,
    summary_path: Path,
    *,
    subject_kind: str,
) -> Path:
    """Hash every regular retained artifact after report generation."""

    manifest_path = destination / "artifact-manifest.json"
    artifacts = []
    for path in sorted(destination.rglob("*")):
        if (
            not path.is_file()
            or path == manifest_path
            or path.is_symlink()
            or path.name.startswith(".")
        ):
            continue
        artifacts.append(
            {
                "path": path.relative_to(destination).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    write_json(
        manifest_path,
        {
            "schema": "pear-sim/artifact-manifest/v0.1",
            "subject": {
                "kind": subject_kind,
                "sha256": sha256_file(summary_path),
            },
            "artifacts": artifacts,
        },
    )
    return manifest_path


__all__ = [
    "BoundaryAggregate",
    "BoundaryHarnessContract",
    "RetainedBoundaryPoints",
    "aggregate_boundary_points",
    "build_boundary_confidence",
    "build_boundary_runner_result",
    "build_boundary_spec",
    "build_boundary_sweep_summary",
    "failures",
    "overall_status",
    "prepare_destination",
    "regime_coverage",
    "root_manifest",
    "run_retained_boundary_points",
    "worst_gates",
    "write_json",
]
