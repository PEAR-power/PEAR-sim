# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Evaluate many designs of one model and rank them against a declared goal.

A batch request names the design parameters, how to choose designs (Latin
hypercube, grid or an explicit list), one objective and any number of
constraints. Every design is screened with a lean run: the same metrics as a
full run, less evidence. Feasible designs are ranked by the objective; the top
few (the finalists) are re-run in full and must reproduce their screening
values exactly, or the batch says so.

What this is and is not:

- The winner is the best design OF THE SAMPLE, not an optimum. Nothing here
  searches between samples or certifies that no better design exists.
- Screening runs keep only the files the ranking reads (run.json, its
  artifact manifest, metrics, losses, warnings); waveforms and the other
  evidence are deleted to keep a 2 000-design batch on disk. Every run is
  deterministic, so any design can be re-run bit for bit. Finalists keep the
  full evidence bundle.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from pear_sim.evidence.parallel_points import resolve_jobs
from pear_sim.evidence.worker_pool import run_pooled
from pear_sim.evidence.parameter_sweep import (
    MAX_PARAMETER_SWEEP_POINT_WALL_TIME_S,
    _PROJECTION_METRICS,
    _read_strict_json,
    _read_verified_metrics,
    _require_bounded_model_shape,
    _require_parameter_affects_resolved_semantics,
    _run_point_worker,
    _safe_run_artifact,
)
from pear_sim.core.solver import HybridSimulator
from pear_sim.spec.model import Model, load_model
from pear_sim.spec.units import parse_quantity

REQUEST_SCHEMA = "pear-sim/batch-request/v0.1"
SUMMARY_SCHEMA = "pear-sim/batch-summary/v0.1"

MAX_BATCH_RUNS = 2000
MAX_BATCH_PARAMETERS = 16
MAX_BATCH_CONSTRAINTS = 16
MAX_BATCH_FINALISTS = 10
DEFAULT_FINALISTS = 3
MAX_GRID_POINTS_PER_PARAMETER = 100

_LOSS_FIELDS = {
    # field -> whether a device id is required
    "total_mean_power_w": False,
    "tj_max_c": False,  # without a device: the hottest device
    "mean_power_w": True,
}
# Files a screening run keeps; the ranking reads nothing else.
_SCREENING_KEPT_FILES = ("metrics", "losses_thermal", "warnings")


class BatchError(ValueError):
    """Invalid batch request or a batch that cannot run."""


# ---------------------------------------------------------------------------
# request
# ---------------------------------------------------------------------------


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], *, name: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise BatchError(f"{name}: unknown field(s) {', '.join(unknown)}")


def _require_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BatchError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise BatchError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class _Quantity:
    """One number read from a run: a signal metric or a loss/thermal field."""

    label: str
    signal: str | None = None
    metric: str | None = None
    losses: str | None = None
    device: str | None = None

    def as_dict(self) -> dict[str, Any]:
        if self.signal is not None:
            return {"signal": self.signal, "metric": self.metric}
        payload: dict[str, Any] = {"losses": self.losses}
        if self.device is not None:
            payload["device"] = self.device
        return payload

    def read(self, metrics: Mapping[str, Any], losses: Mapping[str, Any] | None) -> float:
        if self.signal is not None:
            value = metrics.get(self.signal, {}).get(self.metric)
        else:
            value = self._read_losses(losses)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise BatchError(f"{self.label} was not recorded by the run")
        if not math.isfinite(float(value)):
            raise BatchError(f"{self.label} is not finite in the run")
        return float(value)

    def _read_losses(self, losses: Mapping[str, Any] | None) -> Any:
        if losses is None:
            raise BatchError(f"{self.label}: the run has no losses-thermal.json")
        if self.losses == "total_mean_power_w":
            return losses.get("total_mean_power_w")
        devices = losses.get("devices") or []
        if self.device is not None:
            devices = [item for item in devices if item.get("component_id") == self.device]
            if not devices:
                raise BatchError(f"{self.label}: no loss record for device {self.device!r}")
        if self.losses == "mean_power_w":
            return devices[0].get("mean_power_w")
        values = [item.get("thermal", {}).get("tj_max_c") for item in devices]
        if not values or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in values
        ):
            raise BatchError(f"{self.label}: a device has no junction temperature")
        return max(values)


def _parse_quantity_selector(raw: Any, model: Model, *, name: str) -> _Quantity:
    if not isinstance(raw, dict):
        raise BatchError(f"{name} must be an object")
    if "signal" in raw:
        _reject_unknown(raw, {"signal", "metric", "min", "max"}, name=name)
        signal, metric = raw.get("signal"), raw.get("metric")
        if signal not in model.outputs:
            raise BatchError(f"{name}.signal {signal!r} is not a model output")
        if metric not in _PROJECTION_METRICS:
            raise BatchError(
                f"{name}.metric must be one of {', '.join(sorted(_PROJECTION_METRICS))}"
            )
        return _Quantity(label=f"{signal}.{metric}", signal=signal, metric=metric)
    if "losses" in raw:
        _reject_unknown(raw, {"losses", "device", "min", "max"}, name=name)
        field, device = raw.get("losses"), raw.get("device")
        if field not in _LOSS_FIELDS:
            raise BatchError(f"{name}.losses must be one of {', '.join(sorted(_LOSS_FIELDS))}")
        if model.loss_analysis is None:
            raise BatchError(
                f"{name}: the model declares no device cards, so no losses are computed"
            )
        bound = {binding.component_id for binding in model.loss_analysis.devices}
        if device is not None and device not in bound:
            raise BatchError(f"{name}.device {device!r} has no loss card")
        if _LOSS_FIELDS[field] and device is None:
            raise BatchError(f"{name}: losses {field!r} needs a device")
        label = f"losses.{field}" + (f"[{device}]" if device is not None else "")
        return _Quantity(label=label, losses=field, device=device)
    raise BatchError(f"{name} must name a 'signal' (with 'metric') or 'losses'")


@dataclass(frozen=True)
class _Constraint:
    quantity: _Quantity
    minimum: float | None
    maximum: float | None

    def as_dict(self) -> dict[str, Any]:
        payload = self.quantity.as_dict()
        if self.minimum is not None:
            payload["min"] = self.minimum
        if self.maximum is not None:
            payload["max"] = self.maximum
        return payload

    def holds(self, value: float) -> bool:
        return (self.minimum is None or value >= self.minimum) and (
            self.maximum is None or value <= self.maximum
        )


@dataclass(frozen=True)
class _Request:
    method: str
    parameters: tuple[str, ...]
    designs: tuple[dict[str, float], ...]
    objective: _Quantity
    sense: str  # "minimize" | "maximize"
    constraints: tuple[_Constraint, ...]
    finalists: int
    normalized: dict[str, Any]
    backend: str  # "python" | "rust"
    backend_id: str  # the exact engine id every run must record


def resolve_backend(value: Any, model: Model) -> tuple[str, str]:
    """Check a requested engine against the model before anything runs.

    Returns the engine name and the exact id its runs record. The compiled
    kernel refuses unsupported models here, with the reason, rather than
    failing every design one by one.
    """

    name = "python" if value is None else value
    if name not in {"python", "rust"}:
        raise BatchError("backend must be 'python' or 'rust'")
    try:
        backend_id = HybridSimulator(model, evidence_mode="lean", backend=name).backend_id
    except Exception as exc:  # unsupported model or kernel not installed
        raise BatchError(str(exc)) from exc
    return name, backend_id


def _parameter_value(model: Model, parameter: str, raw: Any, *, name: str) -> float:
    quantity = model.parameters[parameter]
    try:
        value = float(parse_quantity(raw, quantity.dimension, name=name).value)
    except Exception as exc:
        raise BatchError(str(exc)) from exc
    if not math.isfinite(value):
        raise BatchError(f"{name} must be finite")
    return value


def _parse_ranges(raw: Any, model: Model, *, grid: bool) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_BATCH_PARAMETERS:
        raise BatchError(f"designs.parameters must list 1..{MAX_BATCH_PARAMETERS} entries")
    ranges = []
    for index, entry in enumerate(raw):
        name = f"designs.parameters[{index}]"
        if not isinstance(entry, dict):
            raise BatchError(f"{name} must be an object")
        allowed = {"name", "from", "to", "scale"} | ({"points"} if grid else set())
        _reject_unknown(entry, allowed, name=name)
        parameter = entry.get("name")
        if parameter not in model.parameters:
            raise BatchError(f"{name}.name {parameter!r} is not a model parameter")
        low = _parameter_value(model, parameter, entry.get("from"), name=f"{name}.from")
        high = _parameter_value(model, parameter, entry.get("to"), name=f"{name}.to")
        if not low < high:
            raise BatchError(f"{name}: 'from' must be below 'to'")
        scale = entry.get("scale", "linear")
        if scale not in {"linear", "log"}:
            raise BatchError(f"{name}.scale must be 'linear' or 'log'")
        if scale == "log" and low <= 0.0:
            raise BatchError(f"{name}: a log scale needs a positive range")
        item: dict[str, Any] = {"name": parameter, "from_si": low, "to_si": high, "scale": scale}
        if grid:
            points = entry.get("points")
            if (
                isinstance(points, bool)
                or not isinstance(points, int)
                or not 2 <= points <= MAX_GRID_POINTS_PER_PARAMETER
            ):
                raise BatchError(
                    f"{name}.points must be an integer in 2..{MAX_GRID_POINTS_PER_PARAMETER}"
                )
            item["points"] = points
        ranges.append(item)
    names = [item["name"] for item in ranges]
    if len(set(names)) != len(names):
        raise BatchError("designs.parameters names a parameter twice")
    return ranges


def _scaled(item: Mapping[str, Any], fraction: np.ndarray) -> np.ndarray:
    low, high = item["from_si"], item["to_si"]
    if item["scale"] == "log":
        return np.exp(math.log(low) + fraction * (math.log(high) - math.log(low)))
    return low + fraction * (high - low)


def latin_hypercube(count: int, dimensions: int, seed: int) -> np.ndarray:
    """``count`` points in the unit cube, one per row-stratum in every dimension.

    Each dimension is cut into ``count`` equal strata; every stratum gets
    exactly one point, at a uniformly random position inside it, and the
    strata are paired across dimensions by independent random permutations.
    """

    rng = np.random.default_rng(seed)
    columns = [
        (rng.permutation(count) + rng.random(count)) / count for _ in range(dimensions)
    ]
    return np.column_stack(columns)


def _expand_designs(raw: Any, model: Model) -> tuple[str, tuple[str, ...], list[dict[str, float]], dict[str, Any]]:
    if not isinstance(raw, dict):
        raise BatchError("designs must be an object")
    method = raw.get("method", "lhs")
    if method == "lhs":
        _reject_unknown(raw, {"method", "count", "seed", "parameters"}, name="designs")
        count = raw.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_BATCH_RUNS:
            raise BatchError(f"designs.count must be an integer in 1..{MAX_BATCH_RUNS}")
        seed = raw.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise BatchError("designs.seed must be a non-negative integer")
        ranges = _parse_ranges(raw.get("parameters"), model, grid=False)
        unit = latin_hypercube(count, len(ranges), seed)
        columns = [_scaled(item, unit[:, index]) for index, item in enumerate(ranges)]
        designs = [
            {item["name"]: float(columns[index][row]) for index, item in enumerate(ranges)}
            for row in range(count)
        ]
        spec = {"method": method, "count": count, "seed": seed, "parameters": ranges}
        return method, tuple(item["name"] for item in ranges), designs, spec
    if method == "grid":
        _reject_unknown(raw, {"method", "parameters"}, name="designs")
        ranges = _parse_ranges(raw.get("parameters"), model, grid=True)
        total = math.prod(item["points"] for item in ranges)
        if total > MAX_BATCH_RUNS:
            raise BatchError(f"the grid has {total} designs, above the {MAX_BATCH_RUNS}-run limit")
        axes = [
            [float(value) for value in _scaled(item, np.linspace(0.0, 1.0, item["points"]))]
            for item in ranges
        ]
        designs = [
            {item["name"]: value for item, value in zip(ranges, combination)}
            for combination in itertools.product(*axes)
        ]
        spec = {"method": method, "parameters": ranges}
        return method, tuple(item["name"] for item in ranges), designs, spec
    if method == "list":
        _reject_unknown(raw, {"method", "rows"}, name="designs")
        rows = raw.get("rows")
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_BATCH_RUNS:
            raise BatchError(f"designs.rows must list 1..{MAX_BATCH_RUNS} designs")
        names: list[str] = []
        designs = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or not row:
                raise BatchError(f"designs.rows[{index}] must be a non-empty object")
            design = {}
            for parameter, value in row.items():
                if parameter not in model.parameters:
                    raise BatchError(
                        f"designs.rows[{index}]: {parameter!r} is not a model parameter"
                    )
                design[parameter] = _parameter_value(
                    model, parameter, value, name=f"designs.rows[{index}].{parameter}"
                )
                if parameter not in names:
                    names.append(parameter)
            designs.append(design)
        if len(names) > MAX_BATCH_PARAMETERS:
            raise BatchError(f"designs.rows set more than {MAX_BATCH_PARAMETERS} parameters")
        return method, tuple(names), designs, {"method": method}
    raise BatchError("designs.method must be 'lhs', 'grid' or 'list'")


def _normalize_request(payload: Any, model: Model) -> _Request:
    if not isinstance(payload, dict) or payload.get("schema") != REQUEST_SCHEMA:
        raise BatchError(f"batch request schema must be {REQUEST_SCHEMA!r}")
    _reject_unknown(
        payload,
        {"schema", "label", "designs", "objective", "constraints", "finalists", "backend"},
        name="batch request",
    )
    method, parameters, designs, design_spec = _expand_designs(payload.get("designs"), model)

    objective_raw = payload.get("objective")
    if not isinstance(objective_raw, dict) or len(objective_raw) != 1:
        raise BatchError("objective must be {'minimize': {...}} or {'maximize': {...}}")
    sense, selector = next(iter(objective_raw.items()))
    if sense not in {"minimize", "maximize"}:
        raise BatchError("objective must be {'minimize': {...}} or {'maximize': {...}}")
    if isinstance(selector, dict) and ({"min", "max"} & set(selector)):
        raise BatchError("objective takes no bounds; put limits in constraints")
    objective = _parse_quantity_selector(selector, model, name=f"objective.{sense}")

    raw_constraints = payload.get("constraints", [])
    if not isinstance(raw_constraints, list) or len(raw_constraints) > MAX_BATCH_CONSTRAINTS:
        raise BatchError(f"constraints must list at most {MAX_BATCH_CONSTRAINTS} entries")
    constraints = []
    for index, raw in enumerate(raw_constraints):
        name = f"constraints[{index}]"
        quantity = _parse_quantity_selector(raw, model, name=name)
        minimum = (
            _require_number(raw["min"], name=f"{name}.min") if "min" in raw else None
        )
        maximum = (
            _require_number(raw["max"], name=f"{name}.max") if "max" in raw else None
        )
        if minimum is None and maximum is None:
            raise BatchError(f"{name} needs 'min', 'max' or both")
        if minimum is not None and maximum is not None and minimum > maximum:
            raise BatchError(f"{name}: 'min' is above 'max'")
        constraints.append(_Constraint(quantity, minimum, maximum))

    finalists = payload.get("finalists", DEFAULT_FINALISTS)
    if (
        isinstance(finalists, bool)
        or not isinstance(finalists, int)
        or not 0 <= finalists <= MAX_BATCH_FINALISTS
    ):
        raise BatchError(f"finalists must be an integer in 0..{MAX_BATCH_FINALISTS}")
    backend, backend_id = resolve_backend(payload.get("backend"), model)

    normalized = {
        "schema": REQUEST_SCHEMA,
        "label": payload.get("label"),
        "designs": design_spec,
        "objective": {sense: objective.as_dict()},
        "constraints": [constraint.as_dict() for constraint in constraints],
        "finalists": finalists,
        "backend": backend,
        "expanded_designs": [
            {"design_id": _design_id(index), "parameters_si": design}
            for index, design in enumerate(designs)
        ],
    }
    return _Request(
        method=method,
        parameters=parameters,
        designs=tuple(designs),
        objective=objective,
        sense=sense,
        constraints=tuple(constraints),
        finalists=finalists,
        normalized=normalized,
        backend=backend,
        backend_id=backend_id,
    )


def _design_id(index: int) -> str:
    return "d" + format(index + 1, "04d")


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------


def _read_run(
    run_dir: Path,
    model_path: Path,
    overrides: Mapping[str, float],
    quantities: Sequence[_Quantity],
    *,
    backend_id: str = "reference-python",
) -> tuple[str, dict[str, float]]:
    """Verify one run bundle and read the requested quantities from it."""

    expected = load_model(model_path, overrides=dict(overrides))
    metrics, run_sha256 = _read_verified_metrics(
        run_dir, expected_model=expected, expected_backend=backend_id
    )
    run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    losses = None
    if "losses_thermal" in run_manifest["files"]:
        losses_path = _safe_run_artifact(
            run_dir, run_manifest["files"]["losses_thermal"], role="run losses"
        )
        losses, _raw = _read_strict_json(losses_path, role="run losses")
    values = {quantity.label: quantity.read(metrics, losses) for quantity in quantities}
    return run_sha256, values


def _prune_screening_run(run_dir: Path) -> None:
    run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    keep = {"run.json", "artifact-manifest.json"} | {
        run_manifest["files"][key]
        for key in _SCREENING_KEPT_FILES
        if key in run_manifest["files"]
    }
    for path in sorted(run_dir.rglob("*"), reverse=True):
        relative = path.relative_to(run_dir).as_posix()
        if path.is_file() and relative not in keep:
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def screen_designs(
    designs: Sequence[Mapping[str, float]],
    model_path: Path,
    root: Path,
    quantities: Sequence[_Quantity],
    *,
    jobs: int,
    backend: str = "python",
    backend_id: str = "reference-python",
) -> list[dict[str, Any]]:
    """Lean-run every design under ``root/<design_id>/run`` and read ``quantities``.

    Runs go concurrently; rows come back in design order. A design that fails to
    run or verify is kept with its error. Each verified run is pruned to the
    files the reading needs.
    """

    point_dirs = []
    for index in range(len(designs)):
        point_dir = root / _design_id(index)
        point_dir.mkdir(parents=True)
        point_dirs.append(point_dir)
    outcomes = run_pooled(
        lambda pool: [
            partial(
                _run_point_worker,
                model_path,
                dict(design),
                point_dir,
                timeout_s=MAX_PARAMETER_SWEEP_POINT_WALL_TIME_S,
                pool=pool,
                lean=True,
                backend=backend,
            )
            for design, point_dir in zip(designs, point_dirs, strict=True)
        ],
        jobs=jobs,
    )
    rows: list[dict[str, Any]] = []
    for index, (design, outcome) in enumerate(zip(designs, outcomes, strict=True)):
        row: dict[str, Any] = {
            "design_id": _design_id(index),
            "parameters_si": dict(design),
            "status": "pass",
            "values": {},
            "violations": [],
            "error": None,
            "run_sha256": None,
        }
        try:
            run_dir = outcome.unwrap()
            row["run_sha256"], row["values"] = _read_run(
                run_dir, model_path, design, quantities, backend_id=backend_id
            )
            _prune_screening_run(run_dir)
        except Exception as exc:  # retained per design; the batch continues
            row["status"] = "fail"
            row["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
        rows.append(row)
    return rows


@dataclass(frozen=True)
class BatchArtifacts:
    output_dir: Path
    summary: Path
    designs_csv: Path
    design_count: int
    feasible_count: int
    failed_count: int
    best_design: str | None
    status: str


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )


def run_batch(
    model_file: str | Path,
    request_file: str | Path,
    output_dir: str | Path,
    *,
    jobs: int | None = None,
) -> BatchArtifacts:
    """Screen every requested design, rank them, and re-run the finalists in full."""

    try:
        worker_jobs = resolve_jobs(jobs)
    except ValueError as exc:
        raise BatchError(str(exc)) from exc
    source_path = Path(model_file).resolve()
    source_bytes = source_path.read_bytes()
    try:
        model = load_model(source_path)
    except Exception as exc:
        raise BatchError(f"Invalid model: {exc}") from exc
    try:
        _require_bounded_model_shape(model, context="batch model")
    except ValueError as exc:
        raise BatchError(str(exc)) from exc
    request_bytes = Path(request_file).read_bytes()
    try:
        payload = json.loads(request_bytes)
    except json.JSONDecodeError as exc:
        raise BatchError(f"batch request is not valid JSON: {exc}") from exc
    request = _normalize_request(payload, model)
    for parameter in request.parameters:
        try:
            _require_parameter_affects_resolved_semantics(source_path, model, parameter)
        except ValueError as exc:
            raise BatchError(str(exc)) from exc

    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise BatchError(f"Batch output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "request.source.json").write_bytes(request_bytes)
    _write_json(destination / "request.normalized.json", request.normalized)
    # Designs run from a retained byte-for-byte copy of the model, so editing
    # the original while the batch runs cannot change what is measured.
    retained_model = destination / "source-model" / f"model{source_path.suffix or '.yaml'}"
    retained_model.parent.mkdir()
    retained_model.write_bytes(source_bytes)

    quantities = [request.objective] + [c.quantity for c in request.constraints]
    quantities = list({quantity.label: quantity for quantity in quantities}.values())
    started = perf_counter()

    # Screening: lean runs, concurrently, consumed in design order.
    rows = screen_designs(
        request.designs,
        retained_model,
        destination / "designs",
        quantities,
        jobs=worker_jobs,
        backend=request.backend,
        backend_id=request.backend_id,
    )
    for row in rows:
        if row["status"] == "pass":
            row["violations"] = [
                constraint.quantity.label
                for constraint in request.constraints
                if not constraint.holds(row["values"][constraint.quantity.label])
            ]
        row["feasible"] = row["status"] == "pass" and not row["violations"]

    feasible = [row for row in rows if row["feasible"]]
    sign = 1.0 if request.sense == "minimize" else -1.0
    feasible.sort(key=lambda row: (sign * row["values"][request.objective.label], row["design_id"]))
    for rank, row in enumerate(feasible, start=1):
        row["rank"] = rank
    for row in rows:
        row.setdefault("rank", None)

    # Finalists: full evidence bundles that must reproduce screening exactly.
    chosen = feasible[: request.finalists]
    finalist_dirs = []
    for row in chosen:
        finalist_dir = destination / "finalists" / row["design_id"]
        finalist_dir.mkdir(parents=True)
        finalist_dirs.append(finalist_dir)
    full_runs = run_pooled(
        lambda pool: [
            partial(
                _run_point_worker,
                retained_model,
                row["parameters_si"],
                finalist_dir,
                timeout_s=MAX_PARAMETER_SWEEP_POINT_WALL_TIME_S,
                pool=pool,
                backend=request.backend,
            )
            for row, finalist_dir in zip(chosen, finalist_dirs, strict=True)
        ],
        jobs=worker_jobs,
    )
    finalists = []
    for row, finalist_dir, outcome in zip(chosen, finalist_dirs, full_runs, strict=True):
        record: dict[str, Any] = {
            "design_id": row["design_id"],
            "rank": row["rank"],
            "run": None,
            "status": "pass",
            "mismatches": [],
            "error": None,
        }
        try:
            run_dir = outcome.unwrap()
            _sha, values = _read_run(
                run_dir,
                retained_model,
                row["parameters_si"],
                quantities,
                backend_id=request.backend_id,
            )
            record["run"] = (run_dir / "run.json").relative_to(destination).as_posix()
            # Lean and full runs must agree exactly; a lean shortcut that
            # changed a ranked number would otherwise pick the wrong winner.
            record["mismatches"] = [
                {"quantity": label, "screening": row["values"][label], "full": value}
                for label, value in values.items()
                if value != row["values"][label]
            ]
            if record["mismatches"]:
                record["status"] = "screening_mismatch"
        except Exception as exc:
            record["status"] = "fail"
            record["error"] = f"{type(exc).__name__}: {str(exc)[:1000]}"
        finalists.append(record)

    failed = [row for row in rows if row["status"] == "fail"]
    if not feasible:
        status = "no_feasible_design"
    elif any(record["status"] != "pass" for record in finalists):
        status = "finalist_check_failed"
    else:
        status = "pass"

    designs_csv = destination / "designs.csv"
    labels = [quantity.label for quantity in quantities]
    with designs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["design_id", "rank", "feasible", "status"]
            + [f"{name} [SI]" for name in request.parameters]
            + labels
            + ["violations", "error"]
        )
        for row in rows:
            writer.writerow(
                [row["design_id"], row["rank"] or "", row["feasible"], row["status"]]
                + [repr(row["parameters_si"].get(name, "")) for name in request.parameters]
                + [repr(row["values"][label]) if label in row["values"] else "" for label in labels]
                + [";".join(row["violations"]), row["error"] or ""]
            )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "label": request.normalized["label"],
        "model": source_path.name,
        "claim": (
            "Best design OF THE SAMPLE, not an optimum, under the declared objective and constraints. "
            "Nothing searches between samples, so a better design may exist. "
            "Screening runs are lean (same metrics, less evidence) and keep only the "
            "files the ranking reads; finalists are full evidence bundles and must "
            "reproduce their screening values exactly."
        ),
        "status": status,
        "backend": request.backend_id,
        "method": request.method,
        "objective": request.normalized["objective"],
        "constraints": request.normalized["constraints"],
        "design_count": len(rows),
        "feasible_count": len(feasible),
        "failed_count": len(failed),
        "best_design": feasible[0]["design_id"] if feasible else None,
        "ranking": [
            {
                "rank": row["rank"],
                "design_id": row["design_id"],
                "parameters_si": row["parameters_si"],
                "values": row["values"],
            }
            for row in feasible[:20]
        ],
        "finalists": finalists,
        "failures": [
            {"design_id": row["design_id"], "error": row["error"]} for row in failed
        ],
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "model_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "wall_time_s": perf_counter() - started,
        "designs": rows,
    }
    summary_path = destination / "batch-summary.json"
    _write_json(summary_path, summary)
    return BatchArtifacts(
        output_dir=destination,
        summary=summary_path,
        designs_csv=designs_csv,
        design_count=len(rows),
        feasible_count=len(feasible),
        failed_count=len(failed),
        best_design=summary["best_design"],
        status=status,
    )
