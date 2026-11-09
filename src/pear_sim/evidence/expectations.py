# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Strict expectation evaluation over existing PEARSim ``metrics.json`` payloads."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
import math
from typing import Any

from pear_sim.run.contracts import sha256_json


EXPECTATIONS_SCHEMA = "pear-sim/expectations/v0.1"
LEGACY_EXPECTATIONS_SCHEMA = "pear-sim/review-expectations/v0.1-draft"
EVIDENCE_SCHEMA = "pear-sim/evidence/v0.1"
EVIDENCE_STATUSES = ("pass", "fail", "not_applicable", "not_comparable", "not_run")
MAX_CRITERIA = 1000
MAX_HUMAN_CHECKS = 100
MAX_TEXT_LENGTH = 4096


class EvidenceValidationError(ValueError):
    """An expectation document has no unambiguous v0.1 meaning."""

    code = "evidence.expectations_invalid"


_DOCUMENT_FIELDS = frozenset(
    {
        "schema",
        "purpose",
        "measurement_window",
        "criteria",
        "human_checks",
        "automation_status",
    }
)
_WINDOW_FIELDS = frozenset({"kind", "start_s", "end_s"})
_CRITERION_FIELDS = frozenset(
    {
        "id",
        "signal",
        "metric",
        "unit",
        "target",
        "absolute_tolerance",
        "relative_tolerance",
        "minimum",
        "maximum",
        "blocking",
        "applicability",
        "reason",
    }
)
_KNOWN_METRICS = frozenset(
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
_TIME_AGGREGATE_METRICS = frozenset({"mean", "mean_last_half", "rms", "rms_last_half"})
_SAMPLED_ALIAS_METRICS = frozenset(
    {
        "final",
        "max",
        "max_last_half",
        "min",
        "min_last_half",
        "peak_to_peak",
        "peak_to_peak_last_half",
    }
)
_RIPPLE_METRICS = frozenset({"ripple", "ripple_last_half"})
_TIME_AGGREGATION_METHOD = "gauss_legendre_8_on_solve_ivp_dense_output"
_EXTREMA_METHOD = "requested_output_grid_samples"
_RIPPLE_METHOD = "requested_output_grid_peak_to_peak"


def _require_exact_fields(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    name: str,
    required: frozenset[str] = frozenset(),
) -> None:
    non_text_keys = [key for key in value if not isinstance(key, str)]
    if non_text_keys:
        raise EvidenceValidationError(f"{name}: every field name must be a string")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise EvidenceValidationError(f"{name}: unknown field(s): {', '.join(unknown)}")
    missing = sorted(required - set(value))
    if missing:
        raise EvidenceValidationError(f"{name}: missing field(s): {', '.join(missing)}")


def _require_text(value: Any, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        qualifier = "a string" if allow_empty else "a non-empty string"
        raise EvidenceValidationError(f"{name} must be {qualifier}")
    if len(value) > MAX_TEXT_LENGTH:
        raise EvidenceValidationError(f"{name} exceeds the {MAX_TEXT_LENGTH}-character limit")
    return value


def _require_number(value: Any, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceValidationError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise EvidenceValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise EvidenceValidationError(f"{name} must be a finite number")
    if nonnegative and number < 0.0:
        raise EvidenceValidationError(f"{name} must be non-negative")
    return number


def _validate_window(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise EvidenceValidationError("measurement_window must be an object")
    _require_exact_fields(
        raw,
        _WINDOW_FIELDS,
        name="measurement_window",
        required=_WINDOW_FIELDS,
    )
    kind = _require_text(raw["kind"], name="measurement_window.kind")
    if kind not in {"full", "last_half", "explicit"}:
        raise EvidenceValidationError(
            "measurement_window.kind must be full, last_half, or explicit"
        )
    start = _require_number(raw["start_s"], name="measurement_window.start_s")
    end = _require_number(raw["end_s"], name="measurement_window.end_s")
    if end <= start:
        raise EvidenceValidationError("measurement_window.end_s must be greater than start_s")
    return {"kind": kind, "start_s": start, "end_s": end}


def _validate_criterion(raw: Any, *, index: int) -> dict[str, Any]:
    name = f"criteria[{index}]"
    if not isinstance(raw, Mapping):
        raise EvidenceValidationError(f"{name} must be an object")
    _require_exact_fields(
        raw,
        _CRITERION_FIELDS,
        name=name,
        required=frozenset({"id", "signal", "metric", "unit"}),
    )
    normalized: dict[str, Any] = {
        "id": _require_text(raw["id"], name=f"{name}.id"),
        "signal": _require_text(raw["signal"], name=f"{name}.signal"),
        "metric": _require_text(raw["metric"], name=f"{name}.metric"),
        "unit": _require_text(raw["unit"], name=f"{name}.unit"),
    }
    blocking = raw.get("blocking", True)
    if not isinstance(blocking, bool):
        raise EvidenceValidationError(f"{name}.blocking must be a boolean")
    normalized["blocking"] = blocking
    applicability = _require_text(
        raw.get("applicability", "applicable"),
        name=f"{name}.applicability",
    )
    if applicability not in {"applicable", "not_applicable"}:
        raise EvidenceValidationError(f"{name}.applicability must be applicable or not_applicable")
    normalized["applicability"] = applicability
    if "reason" in raw:
        normalized["reason"] = _require_text(raw["reason"], name=f"{name}.reason")
    elif applicability == "not_applicable":
        raise EvidenceValidationError(
            f"{name}.reason is required when applicability is not_applicable"
        )

    has_target = "target" in raw
    has_bounds = "minimum" in raw or "maximum" in raw
    if has_target == has_bounds:
        raise EvidenceValidationError(
            f"{name} must define either target with tolerance or minimum/maximum bounds"
        )
    if has_target:
        normalized["target"] = _require_number(raw["target"], name=f"{name}.target")
        has_tolerance = "absolute_tolerance" in raw or "relative_tolerance" in raw
        if not has_tolerance:
            raise EvidenceValidationError(
                f"{name} target comparison requires absolute_tolerance or relative_tolerance"
            )
        if "absolute_tolerance" in raw:
            normalized["absolute_tolerance"] = _require_number(
                raw["absolute_tolerance"],
                name=f"{name}.absolute_tolerance",
                nonnegative=True,
            )
        if "relative_tolerance" in raw:
            normalized["relative_tolerance"] = _require_number(
                raw["relative_tolerance"],
                name=f"{name}.relative_tolerance",
                nonnegative=True,
            )
        if "minimum" in raw or "maximum" in raw:
            raise EvidenceValidationError(f"{name} target cannot be combined with bounds")
    else:
        if "absolute_tolerance" in raw or "relative_tolerance" in raw:
            raise EvidenceValidationError(f"{name} bounds cannot include target tolerances")
        if "minimum" in raw:
            normalized["minimum"] = _require_number(raw["minimum"], name=f"{name}.minimum")
        if "maximum" in raw:
            normalized["maximum"] = _require_number(raw["maximum"], name=f"{name}.maximum")
        if (
            "minimum" in normalized
            and "maximum" in normalized
            and normalized["minimum"] > normalized["maximum"]
        ):
            raise EvidenceValidationError(f"{name}.minimum must not exceed maximum")
    return normalized


def validate_expectations(expectations: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a v0.1 expectation document.

    The existing review draft is accepted as an explicit compatibility schema. Unknown
    fields are rejected so typos cannot silently weaken a criterion.
    """

    if not isinstance(expectations, Mapping):
        raise EvidenceValidationError("expectations must be an object")
    _require_exact_fields(
        expectations,
        _DOCUMENT_FIELDS,
        name="expectations",
        required=frozenset({"schema", "criteria"}),
    )
    schema = _require_text(expectations["schema"], name="expectations.schema")
    if schema not in {EXPECTATIONS_SCHEMA, LEGACY_EXPECTATIONS_SCHEMA}:
        raise EvidenceValidationError(f"Unsupported expectations schema {schema!r}")
    criteria = expectations["criteria"]
    if not isinstance(criteria, list):
        raise EvidenceValidationError("expectations.criteria must be an array")
    if len(criteria) > MAX_CRITERIA:
        raise EvidenceValidationError(
            f"expectations.criteria exceeds the {MAX_CRITERIA}-entry limit"
        )
    normalized_criteria = [
        _validate_criterion(criterion, index=index) for index, criterion in enumerate(criteria)
    ]
    ids = [criterion["id"] for criterion in normalized_criteria]
    duplicates = sorted(identifier for identifier, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise EvidenceValidationError(
            "expectations.criteria contains duplicate id(s): " + ", ".join(duplicates)
        )
    normalized: dict[str, Any] = {
        "schema": schema,
        "criteria": normalized_criteria,
    }
    if "purpose" in expectations:
        normalized["purpose"] = _require_text(expectations["purpose"], name="expectations.purpose")
    if "measurement_window" in expectations:
        normalized["measurement_window"] = _validate_window(expectations["measurement_window"])
    if "human_checks" in expectations:
        human_checks = expectations["human_checks"]
        if not isinstance(human_checks, list):
            raise EvidenceValidationError("expectations.human_checks must be an array")
        if len(human_checks) > MAX_HUMAN_CHECKS:
            raise EvidenceValidationError(
                f"expectations.human_checks exceeds the {MAX_HUMAN_CHECKS}-entry limit"
            )
        normalized["human_checks"] = [
            _require_text(item, name=f"expectations.human_checks[{index}]")
            for index, item in enumerate(human_checks)
        ]
    if "automation_status" in expectations:
        normalized["automation_status"] = _require_text(
            expectations["automation_status"], name="expectations.automation_status"
        )
    return normalized


def _criterion_requirement(criterion: Mapping[str, Any]) -> dict[str, Any]:
    if "target" in criterion:
        return {
            key: criterion[key]
            for key in ("target", "absolute_tolerance", "relative_tolerance")
            if key in criterion
        }
    return {key: criterion[key] for key in ("minimum", "maximum") if key in criterion}


def _unevaluated(criterion: Mapping[str, Any], status: str, reason: str) -> dict[str, Any]:
    return {
        "id": criterion["id"],
        "status": status,
        "blocking": criterion["blocking"],
        "subject": {
            "signal": criterion["signal"],
            "metric": criterion["metric"],
            "unit": criterion["unit"],
        },
        "requirement": _criterion_requirement(criterion),
        "measured": None,
        "reason": reason,
    }


def _times_match(left: float, right: float) -> bool:
    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) <= max(1e-15, scale * 1e-12)


def _window_mismatch_reason(
    metric_name: str,
    record: Mapping[str, Any],
    expectation_window: Mapping[str, Any] | None,
) -> str | None:
    if expectation_window is None:
        return None
    is_last_half = metric_name.endswith("_last_half")
    metric_kind = "last_half" if is_last_half else "full"
    expected_kind = expectation_window["kind"]
    if expected_kind != "explicit" and expected_kind != metric_kind:
        return (
            f"Metric {metric_name!r} describes the {metric_kind} window, not the "
            f"declared {expected_kind} window."
        )
    prefix = "" if is_last_half else "full_"
    start_raw = record.get(f"{prefix}window_start_s")
    end_raw = record.get(f"{prefix}window_end_s")
    if isinstance(start_raw, bool) or not isinstance(start_raw, (int, float)):
        return "Metric record has no finite measurement-window provenance."
    if isinstance(end_raw, bool) or not isinstance(end_raw, (int, float)):
        return "Metric record has no finite measurement-window provenance."
    try:
        start = float(start_raw)
        end = float(end_raw)
    except (OverflowError, ValueError):
        return "Metric record has no finite measurement-window provenance."
    if not math.isfinite(start) or not math.isfinite(end):
        return "Metric record has no finite measurement-window provenance."
    if not _times_match(start, expectation_window["start_s"]) or not _times_match(
        end, expectation_window["end_s"]
    ):
        return (
            f"Metric window [{start:.12g}, {end:.12g}] s does not match the declared "
            f"window [{expectation_window['start_s']:.12g}, "
            f"{expectation_window['end_s']:.12g}] s."
        )
    return None


def _method_mismatch_reason(metric_name: str, record: Mapping[str, Any]) -> str | None:
    if (
        metric_name in _TIME_AGGREGATE_METRICS
        and record.get("aggregation_method") != _TIME_AGGREGATION_METHOD
    ):
        return f"Metric {metric_name!r} requires aggregation_method {_TIME_AGGREGATION_METHOD!r}."
    if metric_name in _SAMPLED_ALIAS_METRICS and record.get("extrema_method") != _EXTREMA_METHOD:
        return f"Metric {metric_name!r} requires extrema_method {_EXTREMA_METHOD!r}."
    if metric_name in _RIPPLE_METRICS:
        if record.get("ripple_method") != _RIPPLE_METHOD:
            return f"Metric {metric_name!r} requires ripple_method {_RIPPLE_METHOD!r}."
        if record.get("continuity") != "structurally_continuous_across_visited_modes":
            return f"Metric {metric_name!r} requires a structurally continuous signal."
    return None


def _evaluate_criterion(
    criterion: Mapping[str, Any],
    metrics: Mapping[str, Any] | None,
    window: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if criterion["applicability"] == "not_applicable":
        return _unevaluated(
            criterion,
            "not_applicable",
            criterion["reason"],
        )
    metric_name = criterion["metric"]
    if metric_name not in _KNOWN_METRICS:
        return _unevaluated(
            criterion,
            "not_run",
            f"Metric {metric_name!r} is not supported by the v0.1 evidence evaluator.",
        )
    if metrics is None:
        return _unevaluated(criterion, "not_run", "No metrics payload was supplied.")
    signal = criterion["signal"]
    record = metrics.get(signal)
    if not isinstance(record, Mapping):
        return _unevaluated(
            criterion,
            "not_run",
            f"Signal {signal!r} is absent from the metrics payload.",
        )
    producer_status = record.get("status")
    if producer_status == "not_applicable":
        reason = record.get("reason")
        return _unevaluated(
            criterion,
            "not_comparable",
            (
                reason
                if isinstance(reason, str) and reason
                else "Producer marked the required metric not applicable."
            ),
        )
    if producer_status == "not_run":
        reason = record.get("reason")
        return _unevaluated(
            criterion,
            "not_run",
            reason if isinstance(reason, str) and reason else "Producer marked it not run.",
        )
    if producer_status is not None:
        return _unevaluated(
            criterion,
            "not_comparable",
            f"Metric producer status {producer_status!r} has no v0.1 comparison semantics.",
        )
    unit = record.get("unit")
    if unit != criterion["unit"]:
        return _unevaluated(
            criterion,
            "not_comparable",
            f"Metric unit {unit!r} does not match expected unit {criterion['unit']!r}.",
        )
    method_mismatch = _method_mismatch_reason(metric_name, record)
    if method_mismatch is not None:
        return _unevaluated(criterion, "not_comparable", method_mismatch)
    mismatch = _window_mismatch_reason(metric_name, record, window)
    if mismatch is not None:
        return _unevaluated(criterion, "not_comparable", mismatch)
    raw_value = record.get(metric_name)
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        return _unevaluated(
            criterion,
            "not_run" if metric_name not in record else "not_comparable",
            (
                f"Metric {metric_name!r} is absent from signal {signal!r}."
                if metric_name not in record
                else f"Metric {metric_name!r} is not numeric."
            ),
        )
    try:
        value = float(raw_value)
    except (OverflowError, ValueError):
        return _unevaluated(
            criterion,
            "not_comparable",
            f"Metric {metric_name!r} cannot be represented as a finite number.",
        )
    if not math.isfinite(value):
        return _unevaluated(
            criterion,
            "not_comparable",
            f"Metric {metric_name!r} is non-finite.",
        )
    measured = {
        "value": value,
        "unit": unit,
        "signal": signal,
        "metric": metric_name,
    }
    requirement = _criterion_requirement(criterion)
    delta: dict[str, Any] | None = None
    if "target" in criterion:
        signed_delta = value - criterion["target"]
        absolute_delta = abs(signed_delta)
        effective_tolerance = max(
            criterion.get("absolute_tolerance", 0.0),
            criterion.get("relative_tolerance", 0.0) * abs(criterion["target"]),
        )
        if not all(
            math.isfinite(derived)
            for derived in (signed_delta, absolute_delta, effective_tolerance)
        ):
            return _unevaluated(
                criterion,
                "not_comparable",
                "Comparison arithmetic exceeds the finite evidence contract.",
            )
        passed = absolute_delta <= effective_tolerance
        delta = {
            "signed": signed_delta,
            "absolute": absolute_delta,
            "effective_tolerance": effective_tolerance,
        }
    else:
        passed = ("minimum" not in criterion or value >= criterion["minimum"]) and (
            "maximum" not in criterion or value <= criterion["maximum"]
        )
    status = "pass" if passed else "fail"
    return {
        "id": criterion["id"],
        "status": status,
        "blocking": criterion["blocking"],
        "subject": {"signal": signal, "metric": metric_name, "unit": unit},
        "requirement": requirement,
        "measured": measured,
        "delta": delta,
        "reason": None if passed else "Measured value does not satisfy the declared criterion.",
    }


def _aggregate_status(records: list[Mapping[str, Any]]) -> str:
    if not records:
        return "not_run"
    statuses = {record["status"] for record in records}
    for status in ("fail", "not_comparable", "not_run", "not_applicable"):
        if status in statuses:
            return status
    return "pass"


def _aggregate_blocking_status(records: list[Mapping[str, Any]]) -> str:
    applicable = [record for record in records if record["status"] != "not_applicable"]
    return _aggregate_status(applicable)


def evaluate_expectations(
    expectations: Mapping[str, Any],
    metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate declared criteria over a ``metrics.json`` object.

    Missing signals, absent metrics, unknown metric names, unit/window mismatches, and
    non-finite values can never produce ``pass``. The result is deterministic and does
    not mutate either input.
    """

    normalized = validate_expectations(expectations)
    if metrics is not None and not isinstance(metrics, Mapping):
        raise EvidenceValidationError("metrics must be an object or None")
    window = normalized.get("measurement_window")
    records = [
        _evaluate_criterion(criterion, metrics, window) for criterion in normalized["criteria"]
    ]
    blocking_records = [record for record in records if record["blocking"]]
    counts = Counter(record["status"] for record in records)
    return {
        "schema": EVIDENCE_SCHEMA,
        "kind": "expectation_evaluation",
        "expectations": {
            "schema": normalized["schema"],
            "sha256": sha256_json(normalized),
            "purpose": normalized.get("purpose"),
            "measurement_window": window,
            "criterion_count": len(records),
        },
        "claim": {
            "classification": "run_acceptance_observation",
            "validation_claim": False,
            "statement": (
                "Criteria are user-declared checks over recorded metrics; passing does not "
                "establish cross-tool compatibility or physical validation."
            ),
        },
        "overall_status": _aggregate_status(records),
        "blocking_status": _aggregate_blocking_status(blocking_records),
        "summary": {
            "criteria": len(records),
            "blocking": len(blocking_records),
            "counts": {status: counts.get(status, 0) for status in EVIDENCE_STATUSES},
        },
        "criteria": records,
        "human_checks": normalized.get("human_checks", []),
    }


__all__ = [
    "EVIDENCE_SCHEMA",
    "EVIDENCE_STATUSES",
    "EXPECTATIONS_SCHEMA",
    "EvidenceValidationError",
    "evaluate_expectations",
    "validate_expectations",
]
