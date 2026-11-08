"""Global sensitivity: which parameters move which outputs, over declared ranges.

Two methods, both on the batch machinery (lean runs, in parallel):

- **Morris** (screening, r·(k+1) runs). For every parameter, the mean absolute
  change of the output when that parameter steps by a fixed fraction of its
  range, from many random starting points (``mu_star``), scaled to the whole
  range. ``sigma`` (spread of those changes) is large when the effect depends
  on where the other parameters are: interaction or non-linearity.
- **Sobol** (variance-based, N·(k+2) runs, Saltelli design). ``S1``: the share
  of the output's variance a parameter explains alone. ``ST``: its share
  including every interaction. ``ST - S1`` is the interaction part.

Every index carries a bootstrap 95 % interval over base samples or
trajectories. Inputs are independent and uniform over their ranges
(log-uniform on a log scale): the indices describe the model over THIS box,
not the spread of real components unless the ranges are that spread.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import qmc

from pear_sim.evidence.batch import (
    BatchError,
    _parse_quantity_selector,
    _parse_ranges,
    _reject_unknown,
    _scaled,
    _Quantity,
    screen_designs,
)
from pear_sim.evidence.parallel_points import resolve_jobs
from pear_sim.evidence.parameter_sweep import (
    _require_bounded_model_shape,
    _require_parameter_affects_resolved_semantics,
)
from pear_sim.spec.model import load_model

REQUEST_SCHEMA = "pear-sim/sensitivity-request/v0.1"
SUMMARY_SCHEMA = "pear-sim/sensitivity-summary/v0.1"

MAX_SENSITIVITY_RUNS = 2000
MAX_OUTPUTS = 16
DEFAULT_TRAJECTORIES = 10
DEFAULT_LEVELS = 4
DEFAULT_BASE_SAMPLES = 64
DEFAULT_BOOTSTRAP = 1000
CONFIDENCE = 0.95

CLAIM = (
    "Indices describe this model over the declared ranges, with every parameter "
    "independent and uniform over its range (log-uniform on a log scale). They rank "
    "what moves each output inside this box; they say nothing about the spread of real "
    "components unless the ranges are that spread. Intervals are bootstrap 95 % over "
    "the base samples (Sobol) or trajectories (Morris): they measure sampling error of "
    "this design, not model error."
)


class SensitivityError(BatchError):
    """Invalid sensitivity request or a study that cannot be estimated."""


# ---------------------------------------------------------------------------
# designs (unit cube)
# ---------------------------------------------------------------------------


def morris_design(
    trajectories: int, dimensions: int, levels: int, seed: int
) -> tuple[np.ndarray, list[list[tuple[int, float]]]]:
    """Morris trajectories in the unit cube.

    Returns the points (``trajectories * (dimensions + 1)`` rows) and, per
    trajectory, the (factor, step) moved between consecutive points. Each
    coordinate starts on one of ``levels`` grid levels and moves once by
    ``delta = levels / (2 (levels - 1))``: up from the lower half, down from
    the upper half, so every point stays on the grid.
    """

    rng = np.random.default_rng(seed)
    delta = levels / (2.0 * (levels - 1))
    points: list[np.ndarray] = []
    steps: list[list[tuple[int, float]]] = []
    for _ in range(trajectories):
        start = rng.integers(0, levels, size=dimensions)
        x = start / (levels - 1)
        order = rng.permutation(dimensions)
        trajectory_steps = []
        points.append(x.copy())
        for factor in order:
            step = delta if start[factor] < levels // 2 else -delta
            x = x.copy()
            x[factor] += step
            points.append(x)
            trajectory_steps.append((int(factor), step))
        steps.append(trajectory_steps)
    return np.vstack(points), steps


def saltelli_design(base_samples: int, dimensions: int, seed: int) -> np.ndarray:
    """Rows: A (N), B (N), then A with column i taken from B, for each i (N each)."""

    sample = qmc.Sobol(d=2 * dimensions, scramble=True, seed=seed).random(base_samples)
    a, b = sample[:, :dimensions], sample[:, dimensions:]
    blocks = [a, b]
    for column in range(dimensions):
        ab = a.copy()
        ab[:, column] = b[:, column]
        blocks.append(ab)
    return np.vstack(blocks)


# ---------------------------------------------------------------------------
# estimators (pure; tested on functions with known indices)
# ---------------------------------------------------------------------------


def _interval(samples: np.ndarray) -> list[float]:
    tail = (1.0 - CONFIDENCE) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return [float(low), float(high)]


def morris_indices(
    values: np.ndarray,
    steps: Sequence[Sequence[tuple[int, float]]],
    dimensions: int,
    *,
    bootstrap: int,
    seed: int,
) -> list[dict[str, Any]]:
    """mu, mu_star, sigma per factor from outputs along each trajectory.

    ``values`` has one entry per design row; NaN marks a failed run, and any
    trajectory touching one is dropped. Effects are per whole range: an
    output that changes by 3 V when a parameter goes from its lowest to its
    highest value, linearly, has mu_star = 3 V.
    """

    per_trajectory = len(steps[0]) + 1 if steps else 0
    effects = []
    for index, trajectory in enumerate(steps):
        chunk = values[index * per_trajectory : (index + 1) * per_trajectory]
        if np.isnan(chunk).any():
            continue
        row = np.empty(dimensions)
        for position, (factor, step) in enumerate(trajectory):
            row[factor] = (chunk[position + 1] - chunk[position]) / step
        effects.append(row)
    if len(effects) < 2:
        raise SensitivityError("fewer than two complete Morris trajectories ran")
    matrix = np.vstack(effects)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(matrix), size=(bootstrap, len(matrix)))
    mu_star_boot = np.abs(matrix)[picks].mean(axis=1)
    return [
        {
            "mu": float(matrix[:, factor].mean()),
            "mu_star": float(np.abs(matrix[:, factor]).mean()),
            "mu_star_ci95": _interval(mu_star_boot[:, factor]),
            "sigma": float(matrix[:, factor].std(ddof=1)),
            "trajectories_used": len(matrix),
        }
        for factor in range(dimensions)
    ]


def _sobol_point(f_a: np.ndarray, f_b: np.ndarray, f_ab: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    pooled = np.concatenate([f_a, f_b])
    variance = float(np.var(pooled, ddof=1))
    if variance == 0.0:
        return np.full(f_ab.shape[0], np.nan), np.full(f_ab.shape[0], np.nan), 0.0
    # Saltelli et al. 2010 (first order) and Jansen 1999 (total effect). The
    # first-order estimator is unbiased for any offset, but its noise grows with
    # the output's mean: a 40 V output varying by 2 V gave S1 intervals of
    # [-3.6, 5.9]. Centering removes the offset without changing the estimate's
    # expectation; the total-effect estimator uses differences and needs none.
    centre = pooled.mean()
    first = ((f_b - centre) * (f_ab - f_a)).mean(axis=1) / variance
    total = 0.5 * ((f_a - f_ab) ** 2).mean(axis=1) / variance
    return first, total, variance


def sobol_indices(
    values: np.ndarray,
    base_samples: int,
    dimensions: int,
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """S1 and ST per factor from outputs on a :func:`saltelli_design`.

    A base sample whose A, B or any A_B row failed (NaN) is dropped as a whole.
    """

    blocks = values.reshape(dimensions + 2, base_samples)
    keep = ~np.isnan(blocks).any(axis=0)
    used = int(keep.sum())
    if used < 4:
        raise SensitivityError("fewer than four complete Sobol base samples ran")
    f_a, f_b, f_ab = blocks[0, keep], blocks[1, keep], blocks[2:, keep]
    first, total, variance = _sobol_point(f_a, f_b, f_ab)
    if variance == 0.0:
        return {
            "variance": 0.0,
            "base_samples_used": used,
            "factors": [
                {"S1": None, "ST": None, "S1_ci95": None, "ST_ci95": None}
                for _ in range(dimensions)
            ],
            "note": "the output does not vary over the box; indices are undefined",
        }
    rng = np.random.default_rng(seed)
    boot_first = np.empty((bootstrap, dimensions))
    boot_total = np.empty((bootstrap, dimensions))
    for draw in range(bootstrap):
        pick = rng.integers(0, used, size=used)
        boot_first[draw], boot_total[draw], _ = _sobol_point(
            f_a[pick], f_b[pick], f_ab[:, pick]
        )
    return {
        "variance": variance,
        "base_samples_used": used,
        "factors": [
            {
                "S1": float(first[factor]),
                "S1_ci95": _interval(boot_first[:, factor]),
                "ST": float(total[factor]),
                "ST_ci95": _interval(boot_total[:, factor]),
            }
            for factor in range(dimensions)
        ],
    }


# ---------------------------------------------------------------------------
# request and run
# ---------------------------------------------------------------------------


def _positive_int(raw: Mapping[str, Any], key: str, default: int, low: int, high: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise SensitivityError(f"{key} must be an integer in {low}..{high}")
    return value


@dataclass(frozen=True)
class SensitivityArtifacts:
    output_dir: Path
    summary: Path
    runs_csv: Path
    method: str
    run_count: int
    failed_count: int
    status: str


def run_sensitivity(
    model_file: str | Path,
    request_file: str | Path,
    output_dir: str | Path,
    *,
    jobs: int | None = None,
) -> SensitivityArtifacts:
    """Run a Morris or Sobol study and write indices with bootstrap intervals."""

    try:
        worker_jobs = resolve_jobs(jobs)
    except ValueError as exc:
        raise SensitivityError(str(exc)) from exc
    source_path = Path(model_file).resolve()
    source_bytes = source_path.read_bytes()
    try:
        model = load_model(source_path)
        _require_bounded_model_shape(model, context="sensitivity model")
    except Exception as exc:
        raise SensitivityError(f"Invalid model: {exc}") from exc
    request_bytes = Path(request_file).read_bytes()
    try:
        payload = json.loads(request_bytes)
    except json.JSONDecodeError as exc:
        raise SensitivityError(f"sensitivity request is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema") != REQUEST_SCHEMA:
        raise SensitivityError(f"sensitivity request schema must be {REQUEST_SCHEMA!r}")
    _reject_unknown(
        payload,
        {
            "schema", "label", "method", "parameters", "outputs", "seed",
            "trajectories", "levels", "base_samples", "bootstrap",
        },
        name="sensitivity request",
    )
    method = payload.get("method")
    if method not in {"morris", "sobol"}:
        raise SensitivityError("method must be 'morris' or 'sobol'")
    if method == "morris" and "base_samples" in payload:
        raise SensitivityError("base_samples belongs to the sobol method")
    if method == "sobol" and ({"trajectories", "levels"} & set(payload)):
        raise SensitivityError("trajectories and levels belong to the morris method")
    try:
        ranges = _parse_ranges(payload.get("parameters"), model, grid=False)
    except BatchError as exc:
        raise SensitivityError(str(exc).replace("designs.parameters", "parameters")) from exc
    if len(ranges) < 2:
        raise SensitivityError("a sensitivity study needs at least two parameters")
    raw_outputs = payload.get("outputs")
    if not isinstance(raw_outputs, list) or not 1 <= len(raw_outputs) <= MAX_OUTPUTS:
        raise SensitivityError(f"outputs must list 1..{MAX_OUTPUTS} quantities")
    outputs: list[_Quantity] = []
    for index, raw in enumerate(raw_outputs):
        if isinstance(raw, dict) and ({"min", "max"} & set(raw)):
            raise SensitivityError(f"outputs[{index}] takes no bounds")
        outputs.append(_parse_quantity_selector(raw, model, name=f"outputs[{index}]"))
    if len({quantity.label for quantity in outputs}) != len(outputs):
        raise SensitivityError("outputs names the same quantity twice")
    seed = _positive_int(payload, "seed", 0, 0, 2**31 - 1)
    bootstrap = _positive_int(payload, "bootstrap", DEFAULT_BOOTSTRAP, 100, 10000)
    dimensions = len(ranges)

    if method == "morris":
        trajectories = _positive_int(payload, "trajectories", DEFAULT_TRAJECTORIES, 2, 200)
        levels = _positive_int(payload, "levels", DEFAULT_LEVELS, 4, 10)
        if levels % 2:
            raise SensitivityError("levels must be even")
        run_count = trajectories * (dimensions + 1)
        settings: dict[str, Any] = {"trajectories": trajectories, "levels": levels}
    else:
        base_samples = _positive_int(payload, "base_samples", DEFAULT_BASE_SAMPLES, 8, 1024)
        if base_samples & (base_samples - 1):
            raise SensitivityError("base_samples must be a power of two (Sobol sequence balance)")
        run_count = base_samples * (dimensions + 2)
        settings = {"base_samples": base_samples}
    if run_count > MAX_SENSITIVITY_RUNS:
        raise SensitivityError(
            f"the study needs {run_count} runs, above the {MAX_SENSITIVITY_RUNS}-run limit"
        )
    for item in ranges:
        try:
            _require_parameter_affects_resolved_semantics(source_path, model, item["name"])
        except ValueError as exc:
            raise SensitivityError(str(exc)) from exc

    if method == "morris":
        unit, steps = morris_design(trajectories, dimensions, levels, seed)
    else:
        unit = saltelli_design(base_samples, dimensions, seed)
    columns = [_scaled(item, unit[:, index]) for index, item in enumerate(ranges)]
    designs = [
        {item["name"]: float(columns[index][row]) for index, item in enumerate(ranges)}
        for row in range(len(unit))
    ]

    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SensitivityError(f"Sensitivity output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "request.source.json").write_bytes(request_bytes)
    retained_model = destination / "source-model" / f"model{source_path.suffix or '.yaml'}"
    retained_model.parent.mkdir()
    retained_model.write_bytes(source_bytes)

    started = perf_counter()
    rows = screen_designs(designs, retained_model, destination / "runs", outputs, jobs=worker_jobs)
    failed = [row for row in rows if row["status"] != "pass"]

    results = []
    for quantity in outputs:
        values = np.array(
            [row["values"].get(quantity.label, np.nan) if row["status"] == "pass" else np.nan
             for row in rows],
            dtype=float,
        )
        entry: dict[str, Any] = {"output": quantity.label, "selector": quantity.as_dict()}
        try:
            if method == "morris":
                indices = morris_indices(values, steps, dimensions, bootstrap=bootstrap, seed=seed)
                entry["parameters"] = [
                    {"name": item["name"], **index} for item, index in zip(ranges, indices, strict=True)
                ]
                entry["ranking"] = [
                    item["name"]
                    for item, _ in sorted(
                        zip(ranges, indices, strict=True), key=lambda pair: -pair[1]["mu_star"]
                    )
                ]
            else:
                estimate = sobol_indices(values, base_samples, dimensions, bootstrap=bootstrap, seed=seed)
                entry["variance"] = estimate["variance"]
                entry["base_samples_used"] = estimate["base_samples_used"]
                if "note" in estimate:
                    entry["note"] = estimate["note"]
                entry["parameters"] = [
                    {"name": item["name"], **index}
                    for item, index in zip(ranges, estimate["factors"], strict=True)
                ]
                ranked = [p for p in entry["parameters"] if p["ST"] is not None]
                entry["ranking"] = [p["name"] for p in sorted(ranked, key=lambda p: -p["ST"])]
            entry["status"] = "pass"
        except SensitivityError as exc:
            entry["status"] = "not_estimated"
            entry["reason"] = str(exc)
        results.append(entry)

    if any(entry["status"] != "pass" for entry in results):
        status = "incomplete"
    elif failed:
        status = "pass_with_failed_runs"
    else:
        status = "pass"

    runs_csv = destination / "runs.csv"
    with runs_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            ["run_id", "status"]
            + [f"{item['name']} [SI]" for item in ranges]
            + [quantity.label for quantity in outputs]
            + ["error"]
        )
        for row in rows:
            writer.writerow(
                [row["design_id"], row["status"]]
                + [repr(row["parameters_si"][item["name"]]) for item in ranges]
                + [
                    repr(row["values"][q.label]) if q.label in row["values"] else ""
                    for q in outputs
                ]
                + [row["error"] or ""]
            )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "label": payload.get("label"),
        "model": source_path.name,
        "method": method,
        "claim": CLAIM,
        "status": status,
        "settings": {**settings, "seed": seed, "bootstrap": bootstrap, "confidence": CONFIDENCE},
        "parameters": ranges,
        "run_count": len(rows),
        "failed_count": len(failed),
        "failures": [{"run_id": r["design_id"], "error": r["error"]} for r in failed[:50]],
        "outputs": results,
        "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        "model_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "wall_time_s": perf_counter() - started,
    }
    summary_path = destination / "sensitivity-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return SensitivityArtifacts(
        output_dir=destination,
        summary=summary_path,
        runs_csv=runs_csv,
        method=method,
        run_count=len(rows),
        failed_count=len(failed),
        status=status,
    )
