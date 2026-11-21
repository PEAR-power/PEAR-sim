# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Families of runs: parametric corners and injected events, with criteria.

A campaign takes one model plus a request describing **axes**, forms their
cartesian product, materializes one complete model source per point, runs each
in a killable child process, reads declared metrics back out of the retained
bundle, and judges every point against declared criteria.

The product of this module is a **requirement**, not a prediction. Given
criteria the design must meet, ``boundary_search`` bisects one parameter for
the largest drift at which every criterion still passes. That number is an
input for a parts engineer — "this architecture tolerates this much and no
more" — and says nothing about whether any particular component will drift
that far, in any environment, for any reason. Where a drift figure came from
travels with it: every axis carries `source` and `note`, either inherited from
the model's declared envelope or stated in the request.

Criteria take their dimension from what is measured. ``signal`` plus ``field``
fix the unit (``v(out)`` is volts); ``limit`` and ``reference`` are either a
plain number in that SI unit or a unit-bearing string of the same dimension
(``"50 mV"``). ``unit`` is optional and must agree: ``v(out) <= 50 A`` is a
request error, and so is the ambiguous ``limit: 50, unit: mV``. A criterion on
a sampled extremum (``max``, ``min``, ...) is decided only on a grid that
resolves the switching period or when the grid's one-sided error cannot change
the verdict (``pear_sim.evidence.metric_fitness``); otherwise the point is
``not_comparable``, with no margin, and is not accepted.

Claim boundary, restated in every artifact this module writes:

* corner extremes bound the response only under per-parameter monotonicity; a
  non-monotone response can peak strictly inside the box and this producer does
  not search there;
* the bisection assumes the criteria are monotone in the searched parameter
  over the bracket, and reports the bracket it actually used;
* protection behaviour (OCP/UVLO) is **not modelled** by this kernel and is
  reported as `not_modelled`, never as "did not trip";
* nothing here computes a cross-section, a threshold, an event rate, or a
  device's response to any environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import csv
import itertools
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from pear_sim.errors import ModelError
from pear_sim.evidence.metric_fitness import EXTREMA_GRID_FIELD, sampled_bound_unfit_reason
from pear_sim.evidence.parallel_points import resolve_jobs, run_ordered
from pear_sim.spec.model import Model, load_model
from pear_sim.spec.signals import unit_for
from pear_sim.spec.units import UnitError, parse_quantity, unit_scale_and_dimension
from pear_sim.spec.parameter_envelope import PROVENANCE_SOURCES, ParameterEnvelope
from pear_sim.spec.perturbation import (
    PerturbationError,
    PerturbationEvent,
    ForcedGateWindow,
    PerturbationSpec,
    apply_perturbation_to_document,
    parse_perturbation_request,
)


__all__ = [
    "CampaignArtifacts",
    "REQUEST_SCHEMA",
    "SUMMARY_SCHEMA",
    "run_campaign",
]

REQUEST_SCHEMA = "pear-sim/perturbation-campaign-request/v0.1"
SUMMARY_SCHEMA = "pear-sim/perturbation-campaign-summary/v0.1"

MAX_CAMPAIGN_RUNS = 512
MAX_AXES = 4
MAX_POINT_WALL_TIME_S = 120.0
_NOT_MODELLED = "not_modelled"


@dataclass(frozen=True)
class CampaignArtifacts:
    output_dir: Path
    summary: Path
    report_html: Path
    run_count: int
    execution_status: str
    acceptance_status: str
    overall_status: str


# ---------------------------------------------------------------------------
# request parsing
# ---------------------------------------------------------------------------


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], *, name: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise PerturbationError(f"{name}: unknown field(s) " + ", ".join(sorted(unknown)))


def _provenance(raw: Mapping[str, Any], *, name: str) -> dict[str, str]:
    """Provenance is mandatory wherever a request states a drift value itself."""

    source = raw.get("source")
    note = raw.get("note")
    if source not in PROVENANCE_SOURCES:
        raise PerturbationError(
            f"{name}.source must be one of " + ", ".join(PROVENANCE_SOURCES)
        )
    if not isinstance(note, str) or not note.strip():
        raise PerturbationError(f"{name}.note must be a non-empty string")
    return {"source": source, "note": note}


@dataclass(frozen=True)
class _AxisPoint:
    label: str
    parameter_override: tuple[str, float] | None = None
    event_mutation: tuple[str, str, Any] | None = None  # (event_id, field, value)


@dataclass(frozen=True)
class _Axis:
    id: str
    kind: str
    points: tuple[_AxisPoint, ...]
    provenance: dict[str, str]
    detail: dict[str, Any]


def _envelope(model: Model, parameter: str, *, name: str) -> ParameterEnvelope:
    envelope = model.parameter_envelopes.get(parameter)
    if envelope is None:
        raise PerturbationError(
            f"{name}: parameter {parameter!r} declares no drift envelope. "
            "Declare corners/distribution/life on it in the model, or use a "
            "parameter_values axis and state the provenance in the request."
        )
    return envelope


def _parse_axis(raw: Any, index: int, model: Model) -> _Axis:
    name = f"axes[{index}]"
    if not isinstance(raw, dict):
        raise PerturbationError(f"{name} must be an object")
    kind = raw.get("kind")
    axis_id = raw.get("id") or f"axis{index}"

    if kind == "parameter_corners":
        _reject_unknown(raw, {"kind", "id", "parameter"}, name=name)
        parameter = raw.get("parameter")
        envelope = _envelope(model, str(parameter), name=name)
        if not envelope.corners:
            raise PerturbationError(f"{name}: {parameter!r} declares no corners")
        points = [_AxisPoint("nom", (str(parameter), float(envelope.nominal.value)))]
        points += [
            _AxisPoint(corner, (str(parameter), value))
            for corner, value in envelope.corners
        ]
        return _Axis(
            id=str(axis_id), kind=kind, points=tuple(points),
            provenance={"source": envelope.source, "note": envelope.note},
            detail={"parameter": parameter, "from": "model envelope corners"},
        )

    if kind == "parameter_life":
        _reject_unknown(raw, {"kind", "id", "parameter", "life_fractions"}, name=name)
        parameter = str(raw.get("parameter"))
        envelope = _envelope(model, parameter, name=name)
        if envelope.life is None:
            raise PerturbationError(f"{name}: {parameter!r} declares no life model")
        fractions = raw.get("life_fractions")
        if not isinstance(fractions, list) or not fractions:
            raise PerturbationError(f"{name}.life_fractions must be a non-empty list")
        points = []
        for fraction in fractions:
            if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
                raise PerturbationError(f"{name}.life_fractions must hold numbers")
            value = envelope.life.value_at(float(envelope.nominal.value), float(fraction))
            points.append(_AxisPoint(f"life{float(fraction):g}", (parameter, value)))
        return _Axis(
            id=str(axis_id), kind=kind, points=tuple(points),
            provenance={"source": envelope.source, "note": envelope.note},
            detail={"parameter": parameter, "life": envelope.life.as_dict()},
        )

    if kind == "parameter_samples":
        _reject_unknown(raw, {"kind", "id", "parameter", "count"}, name=name)
        parameter = str(raw.get("parameter"))
        envelope = _envelope(model, parameter, name=name)
        if envelope.distribution is None:
            raise PerturbationError(f"{name}: {parameter!r} declares no distribution")
        count = raw.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 64:
            raise PerturbationError(f"{name}.count must be an integer in 1..64")
        draws = envelope.distribution.sample(float(envelope.nominal.value), count)
        points = [
            _AxisPoint(f"s{draw_index}", (parameter, value))
            for draw_index, value in enumerate(draws)
        ]
        return _Axis(
            id=str(axis_id), kind=kind, points=tuple(points),
            provenance={"source": envelope.source, "note": envelope.note},
            detail={
                "parameter": parameter,
                "distribution": envelope.distribution.as_dict(),
            },
        )

    if kind == "parameter_values":
        _reject_unknown(
            raw, {"kind", "id", "parameter", "values", "source", "note"}, name=name
        )
        parameter = str(raw.get("parameter"))
        quantity = model.parameters.get(parameter)
        if quantity is None:
            raise PerturbationError(f"{name}: unknown model parameter {parameter!r}")
        values = raw.get("values")
        if not isinstance(values, list) or not values:
            raise PerturbationError(f"{name}.values must be a non-empty list")
        provenance = _provenance(raw, name=name)
        points = []
        for value_index, value in enumerate(values):
            parsed = parse_quantity(
                value, quantity.dimension, name=f"{name}.values[{value_index}]"
            )
            points.append(
                _AxisPoint(f"{parameter}={value}", (parameter, float(parsed.value)))
            )
        return _Axis(
            id=str(axis_id), kind=kind, points=tuple(points),
            provenance=provenance,
            detail={"parameter": parameter, "from": "explicit request values"},
        )

    if kind in ("event_phase", "event_field"):
        allowed = {"kind", "id", "event", "values", "source", "note"}
        if kind == "event_field":
            allowed.add("field")
        _reject_unknown(raw, allowed, name=name)
        event_id = str(raw.get("event"))
        field = "phase" if kind == "event_phase" else str(raw.get("field"))
        if kind == "event_field" and field not in (
            "width", "amplitude", "tau_rise", "tau_fall", "level", "at"
        ):
            raise PerturbationError(f"{name}.field {field!r} is not sweepable")
        values = raw.get("values")
        if not isinstance(values, list) or not values:
            raise PerturbationError(f"{name}.values must be a non-empty list")
        provenance = _provenance(raw, name=name)
        points = tuple(
            _AxisPoint(f"{field}={value}", event_mutation=(event_id, field, value))
            for value in values
        )
        return _Axis(
            id=str(axis_id), kind=kind, points=points, provenance=provenance,
            detail={"event": event_id, "field": field},
        )

    raise PerturbationError(
        f"{name}.kind must be one of parameter_corners, parameter_life, "
        f"parameter_samples, parameter_values, event_phase, event_field "
        f"(got {kind!r})"
    )


@dataclass(frozen=True)
class _Criterion:
    id: str
    signal: str
    field: str
    op: str  # "max" | "min" | "abs_deviation_max"
    limit: float
    reference: float | None
    unit: str
    provenance: dict[str, str]

    def bounds(self) -> tuple[float | None, float | None]:
        """The requirement as ``lower <= value <= upper``."""

        if self.op == "max":
            return None, self.limit
        if self.op == "min":
            return self.limit, None
        reference = float(self.reference or 0.0)
        return reference - self.limit, reference + self.limit

    def verdict(
        self, metrics: Mapping[str, Any], grids: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Status for one point: pass/fail with a margin, or why there is none.

        A sampled extremum on a grid that does not resolve the switching
        period decides nothing unless its known-sign error cannot change the
        verdict (``pear_sim.evidence.metric_fitness``); it then gets
        ``not_comparable`` and no margin.
        """

        value = metrics.get(f"{self.signal}::{self.field}")
        if value is None:
            return {"status": "not_run", "reason": "metric absent"}
        value = float(value)
        lower, upper = self.bounds()
        unfit = sampled_bound_unfit_reason(
            self.field, grids.get(self.signal), value, lower=lower, upper=upper
        )
        if unfit is not None:
            return {"status": "not_comparable", "value": value, "reason": unfit}
        status, margin = self.evaluate(value)
        return {"status": status, "value": value, "margin": margin}

    def evaluate(self, value: float) -> tuple[str, float]:
        if self.op == "max":
            margin = self.limit - value
        elif self.op == "min":
            margin = value - self.limit
        else:  # abs_deviation_max
            margin = self.limit - abs(value - float(self.reference or 0.0))
        return ("pass" if margin >= 0.0 else "fail"), margin

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "signal": self.signal,
            "field": self.field,
            "op": self.op,
            "limit": self.limit,
            "unit": self.unit,
            **self.provenance,
        }
        if self.reference is not None:
            payload["reference"] = self.reference
        return payload


_SIGNAL_FIELDS = ("min", "max", "mean_last_half", "min_last_half", "max_last_half")
_FSW_FIELD = "fsw_measured_hz"


def _measured_unit(signal: str, field: str, model: Model, *, name: str) -> str:
    """Unit of ``signal::field`` as the producer measures it.

    The dimension of a criterion comes from WHAT IS MEASURED, never from the
    request's own ``unit`` label: a label the producer does not check lets
    ``v(out) <= 50 A`` pass against 40 V.
    """

    if field == _FSW_FIELD:
        return "Hz"
    if field not in _SIGNAL_FIELDS:
        raise PerturbationError(
            f"{name}.field must be one of {', '.join(_SIGNAL_FIELDS)} or "
            f"{_FSW_FIELD} (got {field!r})"
        )
    graph = getattr(model, "signals", None)
    declared = {block.id: block.unit for block in graph.blocks} if graph else {}
    try:
        return unit_for(signal, declared)
    except ModelError as exc:
        raise PerturbationError(f"{name}.signal: {exc}") from exc


def _criterion_value(raw: Any, measured_unit: str, *, name: str) -> float:
    """An SI number, or a unit-bearing string of the measured dimension."""

    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        raise PerturbationError(
            f"{name} must be a number in {measured_unit!r} or a string with a unit"
        )
    if isinstance(raw, str):
        known = unit_scale_and_dimension(measured_unit)
        if known is None:
            raise PerturbationError(
                f"{name}: a unit-bearing string is not supported for a quantity "
                f"measured in {measured_unit!r}; give a plain number in that unit"
            )
        try:
            value = float(parse_quantity(raw, known[1], name=name).value)
        except UnitError as exc:
            raise PerturbationError(str(exc)) from exc
    else:
        value = float(raw)
    if not math.isfinite(value):
        raise PerturbationError(f"{name} must be finite")
    return value


def _check_declared_unit(
    declared: Any, measured_unit: str, *, subject: str, bare_number: bool, name: str
) -> None:
    if declared is None or declared == "":
        return
    if not isinstance(declared, str):
        raise PerturbationError(f"{name} must be a string")
    if declared.strip() == measured_unit:
        return
    given = unit_scale_and_dimension(declared)
    known = unit_scale_and_dimension(measured_unit)
    if given is None or known is None or given[1] != known[1]:
        raise PerturbationError(
            f"{name} {declared!r} does not match {subject}, which is measured in "
            f"{measured_unit!r}"
        )
    if given[0] != known[0] and bare_number:
        raise PerturbationError(
            f"{name} {declared!r} with a bare number is ambiguous (is the number in "
            f"{declared} or in {measured_unit}?); write the value with its unit, "
            f"e.g. \"50 {declared}\", or give an SI number with unit {measured_unit!r}"
        )


def _parse_criteria(raw: Any, model: Model) -> tuple[_Criterion, ...]:
    if not isinstance(raw, list) or not 1 <= len(raw) <= 16:
        raise PerturbationError("criteria must list 1..16 entries")
    criteria = []
    for index, entry in enumerate(raw):
        name = f"criteria[{index}]"
        if not isinstance(entry, dict):
            raise PerturbationError(f"{name} must be an object")
        _reject_unknown(
            entry,
            {"id", "signal", "field", "op", "limit", "reference", "unit", "source", "note"},
            name=name,
        )
        op = entry.get("op")
        if op not in ("max", "min", "abs_deviation_max"):
            raise PerturbationError(
                f"{name}.op must be max, min or abs_deviation_max (got {op!r})"
            )
        signal, field = str(entry.get("signal")), str(entry.get("field"))
        measured_unit = _measured_unit(signal, field, model, name=name)
        raw_limit, raw_reference = entry.get("limit"), entry.get("reference")
        limit = _criterion_value(raw_limit, measured_unit, name=f"{name}.limit")
        if op == "abs_deviation_max" and raw_reference is None:
            raise PerturbationError(f"{name}.reference is required for abs_deviation_max")
        reference = (
            None
            if raw_reference is None
            else _criterion_value(raw_reference, measured_unit, name=f"{name}.reference")
        )
        _check_declared_unit(
            entry.get("unit"),
            measured_unit,
            subject=f"{signal} {field}",
            bare_number=not isinstance(raw_limit, str)
            or (raw_reference is not None and not isinstance(raw_reference, str)),
            name=f"{name}.unit",
        )
        criteria.append(
            _Criterion(
                id=str(entry.get("id") or f"c{index}"),
                signal=signal,
                field=field,
                op=str(op),
                limit=limit,
                reference=reference,
                # Always the unit the limit is stored in: the measured SI unit.
                unit=measured_unit,
                provenance=_provenance(entry, name=name),
            )
        )
    # Verdicts are keyed by id: with a repeated id the later criterion would
    # silently replace the earlier one, and the order would decide acceptance.
    seen: set[str] = set()
    for criterion in criteria:
        if criterion.id in seen:
            raise PerturbationError(
                f"criteria: id {criterion.id!r} is used more than once; every "
                "criterion needs its own id (an omitted id defaults to c<index>)"
            )
        seen.add(criterion.id)
    return tuple(criteria)


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


def _prepare_destination(output_dir: str | Path) -> Path:
    destination = Path(output_dir).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise PerturbationError(
            f"Campaign output directory is not empty: {destination}"
        )
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def _run_point(model_path: Path, point_dir: Path, *, timeout_s: float) -> Path:
    worker_dir = point_dir / ".run-worker"
    command = [
        sys.executable, "-I", "-m",
        "pear_sim.evidence.perturbation_campaign_worker",
        str(model_path), "--out", str(worker_dir),
    ]
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    try:
        completed = subprocess.run(
            command, check=False, capture_output=True, cwd=point_dir,
            env=environment, timeout=max(0.001, timeout_s),
        )
    except subprocess.TimeoutExpired as exc:
        shutil.rmtree(worker_dir, ignore_errors=True)
        raise PerturbationError(
            f"Point exceeded the {timeout_s:.1f}-second worker budget"
        ) from exc
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        try:
            payload = json.loads(completed.stdout.decode("utf-8"))
            error = payload.get("error", {})
            detail = f"{error.get('type', 'WorkerError')}: {error.get('message', detail)}"
        except (UnicodeError, json.JSONDecodeError, AttributeError):
            pass
        shutil.rmtree(worker_dir, ignore_errors=True)
        raise PerturbationError(f"Point worker failed: {detail}")
    run_dir = point_dir / "run"
    worker_dir.rename(run_dir)
    return run_dir


def _read_waveform(run_dir: Path) -> tuple[list[float], dict[str, list[float]]]:
    times: list[float] = []
    signals: dict[str, list[float]] = {}
    with (run_dir / "waveforms.csv").open() as handle:
        reader = csv.DictReader(handle)
        names = [name for name in (reader.fieldnames or []) if name != "time"]
        signals = {name: [] for name in names}
        for row in reader:
            times.append(float(row["time"]))
            for name in names:
                signals[name].append(float(row[name]))
    return times, signals


@dataclass(frozen=True)
class _SwitchingFrequency:
    """Outcome of :func:`_measured_switching_frequency`.

    ``value_hz`` is ``None`` exactly when ``reason`` says why; a number is never
    returned without the bounds that make it checkable.
    """

    value_hz: float | None
    lo_hz: float | None
    hi_hz: float | None
    per_gate_hz: dict[str, float]
    reason: str | None


def _in_window(time: float, windows: Sequence[ForcedGateWindow]) -> bool:
    # Closed on both ends: the release edge is logged exactly at ``to_s``
    # (the window end is an integration breakpoint) and has no command behind it.
    return any(window.from_s <= time <= window.to_s for window in windows)


def _measured_switching_frequency(
    run_dir: Path,
    window_start: float,
    forced_windows: Sequence[ForcedGateWindow],
) -> _SwitchingFrequency:
    """Switching frequency over the settled window, per gate, from events.jsonl.

    Only ``scheduled_gate`` rises count, per gate target, and only those outside
    every declared forced window on that gate: a ``forced_gate`` edge and the
    release rise at a window's end are realizations of the override, not
    command edges. Rises are never pooled across gates — a complementary pair
    rises twice per period. A rise-to-rise interval that spans a forced window
    is dropped (the window may have masked command edges inside it), which
    splits a gate's rises into separate clean runs.

    Estimate per gate: ``n / span`` with ``n`` clean intervals over a total
    clean ``span``. Under the assumption of one rise per carrier period, each
    rise sitting anywhere inside its period, each run's span is ``n_k*T + d_k``
    with ``|d_k| < T``, so the true frequency lies in
    ``((n - c) / span, (n + c) / span)`` for ``c`` clean runs. Gates whose
    intervals do not intersect switch at different frequencies and no single
    number is reported. The assumption (no skipped pulse, no extra pulse) is
    NOT checked here.
    """

    path = run_dir / "events.jsonl"
    if not path.exists():
        return _SwitchingFrequency(None, None, None, {}, "events.jsonl absent")
    windows_by_gate: dict[str, list[ForcedGateWindow]] = {}
    for window in forced_windows:
        windows_by_gate.setdefault(window.control_id, []).append(window)

    rises: dict[str, list[float]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("kind") not in ("scheduled_gate", "forced_gate"):
            continue
        target = str(event.get("target"))
        rises.setdefault(target, [])
        if event.get("kind") != "scheduled_gate" or event.get("current") is not True:
            continue
        time = float(event["time"])
        if time < window_start or _in_window(time, windows_by_gate.get(target, ())):
            continue
        rises[target].append(time)
    if not rises:
        return _SwitchingFrequency(None, None, None, {}, "no gate events in this run")

    per_gate: dict[str, float] = {}
    bounds: dict[str, tuple[float, float]] = {}
    for target in sorted(rises):
        times = sorted(rises[target])
        windows = windows_by_gate.get(target, ())
        intervals = 0
        span = 0.0
        runs = 0
        run_open = False
        previous: float | None = None
        for time in times:
            if previous is not None and not any(
                previous < window.to_s and time > window.from_s for window in windows
            ):
                if not run_open:
                    runs += 1
                intervals += 1
                span += time - previous
                run_open = True
            else:
                run_open = False
            previous = time
        if intervals == 0 or span <= 0.0:
            continue
        per_gate[target] = intervals / span
        bounds[target] = ((intervals - runs) / span, (intervals + runs) / span)

    unmeasured = sorted(set(rises) - set(per_gate))
    if per_gate and unmeasured:
        # A gate with no clean interval gives no evidence that it shares the
        # others' frequency; publishing theirs as "the" frequency would hide it.
        return _SwitchingFrequency(
            None, None, None, per_gate,
            "no clean command interval in the settled window on gate(s) "
            + ", ".join(unmeasured)
            + "; a shared switching frequency is not established",
        )
    if not per_gate:
        return _SwitchingFrequency(
            None, None, None, {},
            "no gate has two command rises without a forced window between them "
            "in the settled window",
        )
    lo = max(low for low, _ in bounds.values())
    hi = min(high for _, high in bounds.values())
    if lo >= hi:
        return _SwitchingFrequency(
            None, None, None, per_gate,
            "per-gate frequencies disagree beyond their resolution bounds; "
            "the gates do not share one switching frequency",
        )
    values = sorted(per_gate.values())
    middle = len(values) // 2
    median = (
        values[middle] if len(values) % 2 else 0.5 * (values[middle - 1] + values[middle])
    )
    # Open interval: the per-run phase offset is strictly under one period.
    if not lo < median < hi:
        # Overlapping bounds do not put the median inside them: a gate with few
        # clean intervals has wide bounds but still pulls the median. Publishing
        # a number its own bounds rule out would be a wrong number.
        return _SwitchingFrequency(
            None, None, None, per_gate,
            "the median of the per-gate estimates lies outside the frequency "
            "interval shared by all gates",
        )
    return _SwitchingFrequency(median, lo, hi, per_gate, None)


def _point_metrics(
    run_dir: Path,
    *,
    nominal_vout: float | None,
    recovery_band: float | None,
    event_onset: float | None,
    forced_windows: Sequence[ForcedGateWindow] = (),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Point metrics, and each signal's ``extrema_grid`` for the criteria."""

    metrics_payload = json.loads((run_dir / "metrics.json").read_text())
    times, signals = _read_waveform(run_dir)
    settled_start = times[len(times) // 2] if times else 0.0

    measured: dict[str, Any] = {}
    grids = {
        signal: fields.get(EXTREMA_GRID_FIELD) for signal, fields in metrics_payload.items()
    }
    for signal, fields in metrics_payload.items():
        for field in ("min", "max", "mean_last_half", "min_last_half", "max_last_half"):
            if field in fields:
                measured[f"{signal}::{field}"] = float(fields[field])

    vout = signals.get("v(out)")
    if vout is not None and nominal_vout is not None:
        window = [
            (time, value)
            for time, value in zip(times, vout)
            if event_onset is None or time >= event_onset
        ]
        deviations = [abs(value - nominal_vout) for _, value in window]
        measured["vout_peak_deviation_v"] = max(deviations) if deviations else 0.0
        measured["vout_settled_error_v"] = (
            float(metrics_payload["v(out)"]["mean_last_half"]) - nominal_vout
        )
        if recovery_band is not None and window:
            band = abs(nominal_vout) * recovery_band / 100.0
            recovery = None
            # Last instant outside the band; recovery is what follows it.
            for time, value in reversed(window):
                if abs(value - nominal_vout) > band:
                    recovery = time
                    break
            if recovery is None:
                measured["recovery_time_s"] = 0.0
            elif recovery >= window[-1][0] - 1e-15:
                # Never came back inside the band before the window closed.
                measured["recovery_time_s"] = None
                measured["recovery_status"] = "not_recovered_within_window"
            else:
                measured["recovery_time_s"] = recovery - (
                    event_onset if event_onset is not None else window[0][0]
                )

    frequency = _measured_switching_frequency(run_dir, settled_start, forced_windows)
    for gate, value in frequency.per_gate_hz.items():
        measured[f"{gate}::fsw_measured_hz"] = value
    if frequency.value_hz is not None:
        measured["fsw_measured_hz"] = frequency.value_hz
        measured["fsw_measured_lo_hz"] = frequency.lo_hz
        measured["fsw_measured_hi_hz"] = frequency.hi_hz
    else:
        measured["fsw_measured_status"] = frequency.reason

    # Explicitly NOT an observation of "protection did not trip".
    measured["protection"] = _NOT_MODELLED
    return measured, grids


def _materialize(
    document: Mapping[str, Any],
    model: Model,
    spec: PerturbationSpec,
    destination: Path,
) -> Path:
    perturbed = apply_perturbation_to_document(document, spec, model)
    path = destination / "model.materialized.yaml"
    path.write_text(yaml.safe_dump(perturbed, sort_keys=False, allow_unicode=True))
    return path


def _mutate_events(
    events: Sequence[PerturbationEvent], mutations: Iterable[tuple[str, str, Any]]
) -> tuple[PerturbationEvent, ...]:
    """Apply ``(event_id, field, value)`` axis mutations to declared events."""

    by_id = {event.id: event for event in events}
    for event_id, field, value in mutations:
        event = by_id.get(event_id)
        if event is None:
            raise PerturbationError(f"axis targets undeclared event {event_id!r}")
        raw = {
            "id": event.id,
            "type": event.kind,
            "target": f"{event.target_kind}:{event.target}",
        }
        if event.at_s is not None:
            raw["at"] = f"{event.at_s!r} s"
        if event.phase is not None:
            raw["phase"] = event.phase
        if event.phase_reference is not None:
            raw["phase_reference"] = event.phase_reference
        if event.phase_anchor_s is not None:
            raw["phase_anchor"] = f"{event.phase_anchor_s!r} s"
        if event.amplitude_a is not None:
            raw["amplitude"] = f"{event.amplitude_a!r} A"
        if event.tau_rise_s is not None:
            raw["tau_rise"] = f"{event.tau_rise_s!r} s"
        if event.tau_fall_s is not None:
            raw["tau_fall"] = f"{event.tau_fall_s!r} s"
        if event.width_s is not None:
            raw["width"] = f"{event.width_s!r} s"
        if event.level is not None:
            raw["level"] = "on" if event.level else "off"
        if field == "phase":
            raw.pop("at", None)
        elif field == "at":
            raw.pop("phase", None)
            raw.pop("phase_reference", None)
            raw.pop("phase_anchor", None)
        raw[field] = value
        (mutated,) = parse_perturbation_request(
            {"schema": "pear-sim/perturbation-request/v0.1", "events": [raw]}
        )
        by_id[event_id] = mutated
    return tuple(by_id[event.id] for event in events)


def _acceptance(verdicts: Mapping[str, Mapping[str, Any]]) -> str:
    """``pass`` only if every criterion passed; a violation outranks an undecided one."""

    statuses = {verdict["status"] for verdict in verdicts.values()}
    if statuses <= {"pass"}:
        return "pass"
    if "fail" in statuses:
        return "fail"
    return "not_comparable" if "not_comparable" in statuses else "fail"


def _event_onset(model: Model, events: Sequence[PerturbationEvent]) -> float | None:
    if not events:
        return None
    from pear_sim.spec.perturbation import _resolve_time

    return min(_resolve_time(event, model) for event in events)


def run_campaign(
    model_file: str | Path,
    request_file: str | Path,
    output_dir: str | Path,
    *,
    jobs: int | None = None,
) -> CampaignArtifacts:
    """Execute one campaign and write its summary and report.

    ``jobs`` grid points run at the same time (default: all cores but one).
    Retained artifacts do not depend on it; only wall time does.
    """

    try:
        worker_jobs = resolve_jobs(jobs)
    except ValueError as exc:
        raise PerturbationError(str(exc)) from exc

    from pear_sim.reporting.perturbation_report import render_campaign_report

    model_path = Path(model_file).resolve()
    payload = json.loads(Path(request_file).read_text())
    if not isinstance(payload, dict) or payload.get("schema") != REQUEST_SCHEMA:
        raise PerturbationError(f"campaign request schema must be {REQUEST_SCHEMA!r}")
    _reject_unknown(
        payload,
        {"schema", "label", "events", "axes", "criteria", "nominal", "boundary_search"},
        name="campaign request",
    )

    model = load_model(model_path)
    document = yaml.safe_load(model_path.read_text())

    events = (
        parse_perturbation_request(
            {"schema": "pear-sim/perturbation-request/v0.1", "events": payload["events"]}
        )
        if payload.get("events")
        else ()
    )

    raw_axes = payload.get("axes")
    if not isinstance(raw_axes, list) or not 1 <= len(raw_axes) <= MAX_AXES:
        raise PerturbationError(f"axes must list 1..{MAX_AXES} entries")
    axes = tuple(_parse_axis(raw, index, model) for index, raw in enumerate(raw_axes))

    total = math.prod(len(axis.points) for axis in axes)
    if total > MAX_CAMPAIGN_RUNS:
        raise PerturbationError(
            f"the requested axes produce {total} runs, above the "
            f"{MAX_CAMPAIGN_RUNS}-run cap"
        )

    criteria = _parse_criteria(payload.get("criteria"), model)
    nominal = payload.get("nominal") or {}
    nominal_vout = nominal.get("vout_v")
    recovery_band = nominal.get("recovery_band_percent")

    destination = _prepare_destination(output_dir)
    rows: list[dict[str, Any]] = []
    failures = 0

    # Each point is materialized first, in process; the point simulations
    # then run concurrently and are consumed in grid order, so every retained
    # file matches a serial campaign. The boundary search below stays serial:
    # each bisection step depends on the previous verdict.
    prepared: list[tuple[dict[str, Any], Path, Any, Path | None, Exception | None]] = []
    for combination in itertools.product(*(axis.points for axis in axes)):
        label = "|".join(point.label for point in combination)
        point_id = "p" + format(len(prepared), "04d")
        point_dir = destination / "points" / point_id
        point_dir.mkdir(parents=True, exist_ok=False)

        overrides = tuple(
            point.parameter_override
            for point in combination
            if point.parameter_override is not None
        )
        mutations = [
            point.event_mutation
            for point in combination
            if point.event_mutation is not None
        ]
        row: dict[str, Any] = {
            "point_id": point_id,
            "label": label,
            "axis_points": {
                axis.id: point.label for axis, point in zip(axes, combination)
            },
            "overrides_si": dict(overrides),
        }
        try:
            point_events = _mutate_events(events, mutations)
            spec = PerturbationSpec(
                label=label, events=point_events, parameter_overrides_si=overrides
            )
            materialized = _materialize(document, model, spec, point_dir)
            row["materialized_model"] = str(materialized.relative_to(destination))
            row["events"] = [event.as_dict() for event in point_events]
        except PerturbationError as exc:
            prepared.append((row, point_dir, None, None, exc))
        else:
            prepared.append((row, point_dir, point_events, materialized, None))

    simulations = run_ordered(
        [
            partial(_run_point, materialized, point_dir, timeout_s=MAX_POINT_WALL_TIME_S)
            for _, point_dir, _, materialized, _ in prepared
            if materialized is not None
        ],
        jobs=worker_jobs,
    )

    for row, point_dir, point_events, materialized, prepare_error in prepared:
        try:
            if prepare_error is not None:
                raise prepare_error
            run_dir = next(simulations).unwrap()
            row["run"] = str(run_dir.relative_to(destination))
            row["execution_status"] = "pass"
            point_model = load_model(materialized)
            row["metrics"], grids = _point_metrics(
                run_dir,
                nominal_vout=float(nominal_vout) if nominal_vout is not None else None,
                recovery_band=(
                    float(recovery_band) if recovery_band is not None else None
                ),
                event_onset=_event_onset(point_model, point_events),
                forced_windows=point_model.forced_gate_windows,
            )
        except PerturbationError as exc:
            failures += 1
            row["execution_status"] = "fail"
            row["error"] = str(exc)[:1024]
            row["metrics"] = {}
            row["criteria"] = {}
            row["acceptance_status"] = "not_run"
            rows.append(row)
            continue

        verdicts = {
            criterion.id: criterion.verdict(row["metrics"], grids) for criterion in criteria
        }
        row["criteria"] = verdicts
        row["acceptance_status"] = _acceptance(verdicts)
        rows.append(row)

    boundary = None
    if payload.get("boundary_search"):
        boundary = _boundary_search(
            payload["boundary_search"], model_path, model, document,
            events, criteria, nominal_vout, recovery_band, destination,
        )

    execution_status = "pass" if failures == 0 else "fail"
    acceptance_status = (
        "pass"
        if all(row.get("acceptance_status") == "pass" for row in rows)
        else "fail"
    )
    binding = next(
        (row["label"] for row in rows if row.get("acceptance_status") != "pass"), None
    )
    overall_status = (
        "pass" if execution_status == "pass" and acceptance_status == "pass" else "fail"
    )

    summary = {
        "schema": SUMMARY_SCHEMA,
        "label": payload.get("label") or destination.name,
        "model": model_path.name,
        "claim": (
            "Declared-input family study. Corner extremes bound the response "
            "only under per-parameter monotonicity — a non-monotone response "
            "can peak strictly inside the box, and this producer does not "
            "search there. The output is a REQUIREMENT (how much drift this "
            "architecture tolerates before a declared criterion fails), never "
            "a prediction of whether any device drifts that far. No cross "
            "section, threshold, event rate or environment response is "
            "computed anywhere in this artifact."
        ),
        "protection_modelling": (
            "OCP/UVLO are NOT modelled by this kernel; every run reports "
            "protection = not_modelled. Absence of a trip in this artifact is "
            "not evidence that a protection would not have tripped."
        ),
        "axes": [
            {
                "id": axis.id, "kind": axis.kind, "point_count": len(axis.points),
                "points": [point.label for point in axis.points],
                **axis.provenance, **axis.detail,
            }
            for axis in axes
        ],
        "criteria": [criterion.as_dict() for criterion in criteria],
        "nominal": dict(nominal),
        "run_count": len(rows),
        "execution_status": execution_status,
        "acceptance_status": acceptance_status,
        "acceptance_policy": "blocking_on_worst_point",
        "binding_point": binding,
        "overall_status": overall_status,
        "boundary_search": boundary,
        "runs": rows,
    }
    summary_path = destination / "campaign-summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    report_path = destination / "campaign-report.html"
    report_path.write_text(render_campaign_report(summary, destination))

    return CampaignArtifacts(
        output_dir=destination,
        summary=summary_path,
        report_html=report_path,
        run_count=len(rows),
        execution_status=execution_status,
        acceptance_status=acceptance_status,
        overall_status=overall_status,
    )


# ---------------------------------------------------------------------------
# boundary search
# ---------------------------------------------------------------------------


def _boundary_search(
    raw: Mapping[str, Any],
    model_path: Path,
    model: Model,
    document: Mapping[str, Any],
    events: Sequence[PerturbationEvent],
    criteria: Sequence[_Criterion],
    nominal_vout: Any,
    recovery_band: Any,
    destination: Path,
) -> dict[str, Any]:
    """Bisect one parameter for the largest drift that still passes everything.

    This is the product: a stated tolerance requirement. It assumes the
    criteria are monotone in this parameter across the declared bracket — the
    assumption is recorded, and the bracket ends are evaluated explicitly so a
    caller can see the assumption was at least not obviously violated.
    """

    _reject_unknown(
        raw,
        {"parameter", "from", "to", "iterations", "source", "note"},
        name="boundary_search",
    )
    parameter = str(raw.get("parameter"))
    quantity = model.parameters.get(parameter)
    if quantity is None:
        raise PerturbationError(f"boundary_search: unknown parameter {parameter!r}")
    provenance = _provenance(raw, name="boundary_search")
    low = float(parse_quantity(raw.get("from"), quantity.dimension, name="boundary_search.from").value)
    high = float(parse_quantity(raw.get("to"), quantity.dimension, name="boundary_search.to").value)
    iterations = raw.get("iterations", 8)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or not 1 <= iterations <= 20:
        raise PerturbationError("boundary_search.iterations must be an integer in 1..20")

    probe_dir = destination / "boundary"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probes: list[dict[str, Any]] = []

    def accepts(value: float, tag: str) -> bool:
        return probe(value, tag) == "pass"

    def probe(value: float, tag: str) -> str:
        point_dir = probe_dir / tag
        point_dir.mkdir(parents=True, exist_ok=True)
        record: dict[str, Any] = {"value_si": value, "tag": tag}
        try:
            spec = PerturbationSpec(
                label=tag, events=tuple(events),
                parameter_overrides_si=((parameter, value),),
            )
            materialized = _materialize(document, model, spec, point_dir)
            run_dir = _run_point(materialized, point_dir, timeout_s=MAX_POINT_WALL_TIME_S)
            point_model = load_model(materialized)
            metrics, grids = _point_metrics(
                run_dir,
                nominal_vout=float(nominal_vout) if nominal_vout is not None else None,
                recovery_band=float(recovery_band) if recovery_band is not None else None,
                event_onset=_event_onset(point_model, events),
                forced_windows=point_model.forced_gate_windows,
            )
        except PerturbationError as exc:
            record.update({"status": "fail", "error": str(exc)[:512]})
            probes.append(record)
            return "fail"
        verdicts = {criterion.id: criterion.verdict(metrics, grids) for criterion in criteria}
        status = _acceptance(verdicts)
        record.update(
            {
                "status": status,
                "criteria": {name: verdict["status"] for name, verdict in verdicts.items()},
            }
        )
        probes.append(record)
        return status

    low_status = probe(low, "bracket-low")
    low_ok = low_status == "pass"
    high_ok = accepts(high, "bracket-high")
    if low_status == "not_comparable":
        # Not "no margin": nothing was shown to fail, the measurement cannot say.
        result_value: float | None = None
        verdict = "not_comparable_at_bracket_low"
    elif not low_ok:
        result_value = None
        verdict = "no_margin_at_bracket_low"
    elif high_ok:
        result_value = high
        verdict = "criteria_still_pass_at_bracket_high"
    else:
        left, right = low, high
        for step in range(iterations):
            middle = 0.5 * (left + right)
            if accepts(middle, f"bisect-{step:02d}"):
                left = middle
            else:
                right = middle
        result_value = left
        verdict = "bisected"

    return {
        "parameter": parameter,
        "dimension": quantity.dimension,
        "nominal_si": float(quantity.value),
        "bracket_si": [low, high],
        "iterations": iterations,
        "verdict": verdict,
        "max_passing_value_si": result_value,
        "max_passing_drift_percent": (
            None
            if result_value is None or quantity.value == 0.0
            else 100.0 * (result_value - float(quantity.value)) / float(quantity.value)
        ),
        "assumption": (
            "criteria are assumed monotone in this parameter across the "
            "bracket; both bracket ends were evaluated explicitly and are "
            "recorded below. A non-monotone response can fail strictly inside "
            "the reported passing range."
        ),
        **provenance,
        "probes": probes,
    }
