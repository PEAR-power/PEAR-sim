"""Tolerance-corner sweeps: worst-case-of-the-batch confidence.

The request declares symmetric or asymmetric tolerances on up to six model
parameters. The producer runs the nominal point plus every sign corner
(full factorial, deterministic order), retains every run bundle through the
same hardened worker as the 1-D sweep, reads declared band metrics from each
verified bundle, and — when expectations are supplied — evaluates them on
EVERY corner: acceptance is the WORST case, and the binding corner is named.

Claim boundary, stated in the artifact: corner extremes bound the response
only under monotonicity in each parameter; a non-monotone response can peak
strictly inside the tolerance box, and this producer does not search there.
This is deliberate worst-case-of-the-batch evidence, not a probabilistic
guarantee and not global optimization.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pear_sim.evidence.evidence_io import evaluate_run
from pear_sim.evidence.expectations import validate_expectations
from pear_sim.evidence.parameter_sweep import (
    MAX_PARAMETER_SWEEP_POINT_WALL_TIME_S,
    ParameterSweepError,
    _prepare_destination,
    _read_strict_json,
    _require_bounded_model_shape,
    _require_parameter_affects_resolved_semantics,
    _run_point_worker,
    _write_json,
)
from pear_sim.spec.model import Model, load_model

__all__ = ["ToleranceCornersArtifacts", "run_tolerance_corners"]

REQUEST_SCHEMA = "pear-sim/tolerance-corners-request/v0.1"
SUMMARY_SCHEMA = "pear-sim/tolerance-corners-summary/v0.1"
_MAX_TOLERANCED_PARAMETERS = 6


@dataclass(frozen=True)
class ToleranceCornersArtifacts:
    output_dir: Path
    summary: Path
    execution_status: str
    acceptance_status: str
    overall_status: str


def _require_percent(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterSweepError(f"{name} must be a number (percent)")
    result = float(value)
    if not 0.0 < result < 100.0:
        raise ParameterSweepError(f"{name} must lie strictly between 0 and 100")
    return result


def _normalize_request(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ParameterSweepError("tolerance-corners request must be an object")
    unknown = set(payload) - {"schema", "tolerances", "band_metrics"}
    if unknown:
        raise ParameterSweepError(
            "tolerance-corners request has unknown field(s): "
            + ", ".join(sorted(unknown))
        )
    if payload.get("schema") != REQUEST_SCHEMA:
        raise ParameterSweepError(
            f"tolerance-corners request schema must be {REQUEST_SCHEMA!r}"
        )
    tolerances = payload.get("tolerances")
    if not isinstance(tolerances, list) or not 1 <= len(tolerances) <= _MAX_TOLERANCED_PARAMETERS:
        raise ParameterSweepError(
            "tolerances must list 1.."
            f"{_MAX_TOLERANCED_PARAMETERS} parameter entries"
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, entry in enumerate(tolerances):
        name = f"tolerances[{index}]"
        if not isinstance(entry, dict):
            raise ParameterSweepError(f"{name} must be an object")
        unknown = set(entry) - {"parameter", "percent", "minus_percent", "plus_percent"}
        if unknown:
            raise ParameterSweepError(
                f"{name} has unknown field(s): " + ", ".join(sorted(unknown))
            )
        parameter = entry.get("parameter")
        if not isinstance(parameter, str) or not parameter:
            raise ParameterSweepError(f"{name}.parameter must be a non-empty string")
        if parameter in seen:
            raise ParameterSweepError(
                f"{name}.parameter duplicates {parameter!r}"
            )
        seen.add(parameter)
        if "percent" in entry:
            if "minus_percent" in entry or "plus_percent" in entry:
                raise ParameterSweepError(
                    f"{name}: use either percent or minus/plus_percent, not both"
                )
            minus = plus = _require_percent(entry["percent"], name=f"{name}.percent")
        else:
            minus = _require_percent(
                entry.get("minus_percent"), name=f"{name}.minus_percent"
            )
            plus = _require_percent(
                entry.get("plus_percent"), name=f"{name}.plus_percent"
            )
        normalized.append(
            {"parameter": parameter, "minus_percent": minus, "plus_percent": plus}
        )
    return normalized


def _normalize_band_metrics(payload: Any) -> list[dict[str, str]]:
    metrics = payload.get("band_metrics")
    if metrics is None:
        return []
    if not isinstance(metrics, list) or len(metrics) > 16:
        raise ParameterSweepError("band_metrics must list at most 16 entries")
    normalized = []
    for index, entry in enumerate(metrics):
        name = f"band_metrics[{index}]"
        if not isinstance(entry, dict) or set(entry) != {"signal", "field"}:
            raise ParameterSweepError(
                f"{name} must be an object with exactly signal and field"
            )
        signal = entry["signal"]
        field = entry["field"]
        if not isinstance(signal, str) or not isinstance(field, str):
            raise ParameterSweepError(f"{name}.signal/.field must be strings")
        normalized.append({"signal": signal, "field": field})
    return normalized


def _nominal_si(model: Model, parameter: str) -> float:
    quantity = model.parameters.get(parameter)
    if quantity is None:
        raise ParameterSweepError(
            f"tolerances reference unknown model parameter {parameter!r}"
        )
    return float(quantity.value)


def _corner_points(
    model: Model, tolerances: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Nominal first, then every sign corner in deterministic order."""

    names = [entry["parameter"] for entry in tolerances]
    nominals = {name: _nominal_si(model, name) for name in names}
    points: list[dict[str, Any]] = [
        {
            "corner_id": "nominal",
            "signs": {name: 0 for name in names},
            "overrides": dict(nominals),
        }
    ]
    for signs in itertools.product((-1, 1), repeat=len(names)):
        overrides = {}
        label_parts = []
        for name, sign, entry in zip(names, signs, tolerances):
            percent = entry["minus_percent"] if sign < 0 else entry["plus_percent"]
            overrides[name] = nominals[name] * (1.0 + sign * percent / 100.0)
            label_parts.append(f"{name}{'-' if sign < 0 else '+'}")
        points.append(
            {
                "corner_id": "corner-" + "".join(label_parts),
                "signs": dict(zip(names, signs)),
                "overrides": overrides,
            }
        )
    return points


def _metric_value(metrics_payload: Mapping[str, Any], signal: str, field: str) -> float:
    signal_metrics = metrics_payload.get(signal)
    if not isinstance(signal_metrics, Mapping) or field not in signal_metrics:
        raise ParameterSweepError(
            f"band metric {signal!r}.{field} is missing from a retained run"
        )
    value = signal_metrics[field]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ParameterSweepError(
            f"band metric {signal!r}.{field} is not a number"
        )
    return float(value)


def run_tolerance_corners(
    model_file: str | Path,
    request_file: str | Path,
    output_dir: str | Path,
    *,
    expectations_file: str | Path | None = None,
) -> ToleranceCornersArtifacts:
    """Execute the nominal point plus every tolerance corner, retained."""

    model_path = Path(model_file).resolve()
    request_payload, _raw = _read_strict_json(
        Path(request_file), role="tolerance-corners request"
    )
    tolerances = _normalize_request(request_payload)
    band_metrics = _normalize_band_metrics(request_payload)

    model = load_model(model_path)
    _require_bounded_model_shape(model, context="tolerance-corner model")
    for entry in tolerances:
        if entry["parameter"] not in model.parameters:
            raise ParameterSweepError(
                "tolerances reference unknown model parameter "
                f"{entry['parameter']!r}"
            )
        _require_parameter_affects_resolved_semantics(
            model_path, model, entry["parameter"]
        )
    for metric in band_metrics:
        if metric["signal"] not in model.outputs:
            raise ParameterSweepError(
                f"band metric signal {metric['signal']!r} is not a model output"
            )

    expectations_path: Path | None = None
    if expectations_file is not None:
        expectations_path = Path(expectations_file).resolve()
        payload, _raw = _read_strict_json(
            expectations_path, role="expectations"
        )
        validate_expectations(payload)

    destination = _prepare_destination(output_dir)
    points = _corner_points(model, tolerances)

    rows: list[dict[str, Any]] = []
    execution_failures = 0
    acceptance_statuses: list[str] = []
    for point in points:
        point_dir = destination / "corners" / point["corner_id"]
        point_dir.mkdir(parents=True, exist_ok=False)
        row: dict[str, Any] = {
            "corner_id": point["corner_id"],
            "signs": point["signs"],
            "overrides_si": point["overrides"],
        }
        try:
            run_dir = _run_point_worker(
                model_path,
                point["overrides"],
                point_dir,
                timeout_s=MAX_PARAMETER_SWEEP_POINT_WALL_TIME_S,
            )
            row["execution_status"] = "pass"
            row["run"] = str(run_dir.relative_to(destination))
            metrics_payload = json.loads((run_dir / "metrics.json").read_text())
            row["band_metrics"] = {
                f"{metric['signal']}::{metric['field']}": _metric_value(
                    metrics_payload, metric["signal"], metric["field"]
                )
                for metric in band_metrics
            }
        except ParameterSweepError as exc:
            execution_failures += 1
            row["execution_status"] = "fail"
            row["error"] = str(exc)[:1024]
            row["band_metrics"] = {}
            rows.append(row)
            continue
        if expectations_path is not None:
            evidence_dir = point_dir / "evidence"
            try:
                evidence = evaluate_run(run_dir, expectations_path, evidence_dir)
                row["acceptance_status"] = evidence.blocking_status
            except Exception as exc:  # noqa: BLE001 - retained verbatim
                row["acceptance_status"] = "error"
                row["acceptance_error"] = str(exc)[:1024]
            acceptance_statuses.append(row["acceptance_status"])
        rows.append(row)

    bands: dict[str, Any] = {}
    executed = [row for row in rows if row["execution_status"] == "pass"]
    for metric in band_metrics:
        key = f"{metric['signal']}::{metric['field']}"
        values = [(row["band_metrics"][key], row["corner_id"]) for row in executed]
        if not values:
            continue
        nominal_value = next(
            (value for value, corner in values if corner == "nominal"), None
        )
        low = min(values)
        high = max(values)
        bands[key] = {
            "signal": metric["signal"],
            "field": metric["field"],
            "nominal": nominal_value,
            "min": low[0],
            "min_corner": low[1],
            "max": high[0],
            "max_corner": high[1],
        }

    execution_status = "pass" if execution_failures == 0 else "fail"
    if expectations_path is None:
        acceptance_status = "not_run"
        binding_corner = None
    else:
        failing = [
            row["corner_id"]
            for row in rows
            if row.get("acceptance_status") not in (None, "pass")
        ]
        acceptance_status = "pass" if not failing and execution_status == "pass" else "fail"
        binding_corner = failing[0] if failing else None
    overall_status = (
        "pass"
        if execution_status == "pass"
        and acceptance_status in {"pass", "not_run"}
        else "fail"
    )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "kind": "tolerance-corners-worst-case",
        "claim": (
            "worst-case-of-the-batch over declared tolerance corners; corner "
            "extremes bound the response only under per-parameter "
            "monotonicity — a non-monotone response can peak inside the box; "
            "not a probabilistic guarantee"
        ),
        "method": "full_factorial_sign_corners_plus_nominal",
        "tolerances": tolerances,
        "corner_count": len(points),
        "execution_status": execution_status,
        "acceptance_status": acceptance_status,
        "acceptance_policy": "blocking_on_worst_corner",
        "binding_corner": binding_corner,
        "overall_status": overall_status,
        "bands": bands,
        "corners": rows,
    }
    summary_path = destination / "tolerance-corners-summary.json"
    _write_json(summary_path, summary)
    _write_json(
        destination / "request.normalized.json",
        {
            "schema": REQUEST_SCHEMA,
            "tolerances": tolerances,
            "band_metrics": band_metrics,
        },
    )
    return ToleranceCornersArtifacts(
        output_dir=destination,
        summary=summary_path,
        execution_status=execution_status,
        acceptance_status=acceptance_status,
        overall_status=overall_status,
    )
