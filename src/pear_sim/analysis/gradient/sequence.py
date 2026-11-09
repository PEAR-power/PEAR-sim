# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Explicit mode sequences extracted from an oracle run.

The gradient engine needs the switching cycle as an EXPLICIT object: which
compiled modes occur, in what order, and what ends each one. The solver
already records exactly that in ``result.mode_intervals`` — this module only
reads the artifact and never touches ``core/``.

Provenance is the contract, not a nicety: a sequence extracted
here always carries the ``run_sha256`` of its source run, computed by the same
recipe the bundle writer uses, so the in-memory sequence and a retained bundle
of the same run name the same identity. A hand-authored sequence is the ONLY
path with ``source_run_sha256 = None``, and that absence is visible in every
artifact downstream — an undeclared sequence is a claim without a source.

Disagreement between a declared sequence and an extracted one is DATA, not an
exception: :func:`sequences_agree` returns the verdict with its reasons, so a
caller can turn it into an assumption flag (T3 contract) instead of dying.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

from pear_sim import __version__
from pear_sim.errors import PearSimError
from pear_sim.run.contracts import model_hashes, sha256_json

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pear_sim.core.solver import SimulationResult


class ModeSequenceError(PearSimError):
    """A mode sequence cannot be extracted or is internally inconsistent."""

    code = "gradient.mode_sequence_invalid"


class ExitKind(Enum):
    """How a sequence step ends — this decides the shape of the sensitivity.

    ``SCHEDULED_GATE``  — a PWM edge; the instant is an explicit function of
                          the design vector, so the saltation matrix is I.
    ``NATURAL_GUARD``   — natural commutation; the instant comes from a guard
                          h(x, t) = 0 and needs the full saltation matrix.
    ``SATURATION``      — a |i| = i_sat segment boundary; a state guard.
    ``PERIOD_END``      — the cycle closes with no event at the boundary.
    """

    SCHEDULED_GATE = "scheduled_gate"
    NATURAL_GUARD = "natural_guard"
    SATURATION = "saturation"
    PERIOD_END = "period_end"


@dataclass(frozen=True)
class ExitCondition:
    kind: ExitKind
    target: str
    """Diode/thyristor/sat-inductor id, or the control id for a PWM edge."""

    guard_is_current: bool
    """True: the crossing guard is a branch current (leaving conduction or a
    saturation boundary); False: a voltage guard (entering conduction) or a
    boundary with no guard at all."""


@dataclass(frozen=True)
class ModeStep:
    """One mode inside the period, with its exit condition."""

    signature: str
    """``CompiledMode.mode_signature`` — the comparison key against the oracle."""

    gates: Mapping[str, bool]
    diodes: Mapping[str, bool]
    constrained_inductors: tuple[str, ...]
    saturated_inductors: tuple[str, ...]
    exit_condition: ExitCondition
    nominal_duration_s: float
    """Newton's starting point; exact for scheduled steps."""


@dataclass(frozen=True)
class ModeSequence:
    """The switching cycle as data — the ENGINE'S INPUT, never its output.

    ``source_run_sha256`` carries provenance. Extracted sequences always have
    it; only :func:`manual_mode_sequence` may leave it ``None``, and that
    absence is the visible mark of a hand-authored claim.
    """

    period_s: float
    steps: tuple[ModeStep, ...]
    source_run_sha256: str | None
    extraction_tolerance_s: float

    def signatures(self) -> tuple[str, ...]:
        return tuple(step.signature for step in self.steps)


_SATURATED_RE = re.compile(r";saturated_inductors\[([^\]]*)\]")


def _saturated_from_signature(signature: str) -> tuple[str, ...]:
    """Recover the saturation flags the interval log does not carry.

    ``mode_intervals`` records gates/diodes/constrained inductors but not the
    saturation set; the signature encodes it, and parsing the signature keeps
    this module out of ``core/``.
    """

    match = _SATURATED_RE.search(signature)
    if match is None:
        return ()
    return tuple(name for name in match.group(1).split(",") if name)


def _source_run_sha256(result: "SimulationResult") -> str:
    """The same identity the bundle writer stamps into ``run.json``.

    The kernel is deterministic, so a run is identified by its resolved model,
    its request and the backend — not by wall-clock or file paths. Computing
    the identical recipe here means a sequence extracted in memory names the
    same run as a bundle retained on disk.
    """

    hashes = model_hashes(result.model)
    return sha256_json(
        {
            "model_sha256": hashes["model_sha256"],
            "request_sha256": hashes["request_sha256"],
            "backend": "reference-python",
            "pear_sim_version": __version__,
        }
    )


def extract_mode_sequence(
    result: "SimulationResult",
    *,
    period_s: float,
    cycle_index: int = -1,
    merge_tolerance_s: float = 1e-12,
) -> ModeSequence:
    """Build the explicit sequence of one cycle from ``result.mode_intervals``.

    ``cycle_index`` selects which complete cycle to read (negative indexes
    from the end; the default is the LAST complete cycle, which is the one
    closest to steady state). The window is clipped exactly to the cycle, and
    every step boundary is classified from the run's own event log — never
    guessed. A boundary this function cannot attribute to a scheduled edge, a
    natural commutation, a saturation crossing or the period end is an error,
    because a silently misclassified exit would later produce a plausible but
    wrong saltation matrix.
    """

    if period_s <= 0.0:
        raise ModeSequenceError("period_s must be positive")
    intervals = result.mode_intervals
    if not intervals:
        raise ModeSequenceError("the run recorded no mode intervals")

    start = float(intervals[0]["start_s"])
    stop = float(intervals[-1]["end_s"])
    cycles = int((stop - start + merge_tolerance_s) / period_s)
    if cycles < 1:
        raise ModeSequenceError(
            f"the run covers {stop - start:.6g} s, less than one full period "
            f"of {period_s:.6g} s — no complete cycle to extract"
        )
    index = cycle_index if cycle_index >= 0 else cycles + cycle_index
    if not 0 <= index < cycles:
        raise ModeSequenceError(
            f"cycle_index {cycle_index} is outside the {cycles} complete "
            "cycle(s) this run contains"
        )
    window_start = start + index * period_s
    window_end = window_start + period_s

    # Clip the interval list to the window, merging consecutive fragments that
    # share a signature (a zero-duration stabilisation step splits an interval
    # without changing the topology).
    steps_raw: list[dict[str, Any]] = []
    for interval in intervals:
        begin = max(float(interval["start_s"]), window_start)
        end = min(float(interval["end_s"]), window_end)
        if end - begin <= merge_tolerance_s:
            continue
        if steps_raw and steps_raw[-1]["mode_signature"] == interval["mode_signature"]:
            steps_raw[-1]["end_s"] = end
            continue
        steps_raw.append(
            {
                "mode_signature": interval["mode_signature"],
                "gates": dict(interval["gates"]),
                "diodes": dict(interval["diodes"]),
                "constrained_inductors": tuple(interval["constrained_inductors"]),
                "start_s": begin,
                "end_s": end,
            }
        )
    if not steps_raw:
        raise ModeSequenceError("the selected cycle contains no mode interval")

    events_by_time = sorted(result.events, key=lambda event: event.time)

    def classify(boundary: float) -> ExitCondition:
        if abs(boundary - window_end) <= merge_tolerance_s:
            return ExitCondition(ExitKind.PERIOD_END, "", False)
        at_boundary = [
            event
            for event in events_by_time
            if abs(event.time - boundary) <= merge_tolerance_s
        ]
        for event in at_boundary:
            if event.kind == "natural_commutation":
                # Leaving conduction crosses the current guard; entering
                # conduction crosses the voltage guard.
                return ExitCondition(
                    ExitKind.NATURAL_GUARD, event.target, bool(event.previous)
                )
        for event in at_boundary:
            if event.kind == "saturation_boundary":
                return ExitCondition(ExitKind.SATURATION, event.target, True)
        for event in at_boundary:
            if event.kind == "scheduled_gate":
                return ExitCondition(ExitKind.SCHEDULED_GATE, event.target, False)
        kinds = sorted({event.kind for event in at_boundary}) or ["<none>"]
        raise ModeSequenceError(
            f"cannot classify the mode boundary at t={boundary!r}: events "
            f"there are {kinds}. A misclassified exit would later yield a "
            "plausible but wrong saltation matrix, so this fails closed."
        )

    steps = tuple(
        ModeStep(
            signature=raw["mode_signature"],
            gates=raw["gates"],
            diodes=raw["diodes"],
            constrained_inductors=raw["constrained_inductors"],
            saturated_inductors=_saturated_from_signature(raw["mode_signature"]),
            exit_condition=classify(raw["end_s"]),
            nominal_duration_s=raw["end_s"] - raw["start_s"],
        )
        for raw in steps_raw
    )
    return ModeSequence(
        period_s=period_s,
        steps=steps,
        source_run_sha256=_source_run_sha256(result),
        extraction_tolerance_s=merge_tolerance_s,
    )


def manual_mode_sequence(
    *,
    period_s: float,
    steps: tuple[ModeStep, ...],
) -> ModeSequence:
    """The one legitimate path to a sequence without provenance.

    Allowed as a DOCUMENTARY override only: callers that accept a declared
    sequence must compare it against an extraction with
    :func:`sequences_agree` and surface the disagreement as a flag. The
    ``source_run_sha256 = None`` it carries is the visible mark that a human,
    not a run, is the source.
    """

    if period_s <= 0.0:
        raise ModeSequenceError("period_s must be positive")
    if not steps:
        raise ModeSequenceError("a mode sequence must contain at least one step")
    return ModeSequence(
        period_s=period_s,
        steps=steps,
        source_run_sha256=None,
        extraction_tolerance_s=0.0,
    )


MODE_SEQUENCE_SCHEMA = "pear-sim/mode-sequence/v0.1"


def mode_sequence_as_dict(sequence: ModeSequence) -> dict[str, Any]:
    """Portable form of a sequence — same object, no parallel notation.

    This is THE serialization of a ``ModeSequence``: the topology library
    stores exactly this shape, and
    :func:`mode_sequence_from_dict` reconstructs the identical dataclass.
    Anything that invents its own mode notation creates a second source of
    truth against the gradient engine, which is what this pair exists to
    prevent.
    """

    return {
        "schema": MODE_SEQUENCE_SCHEMA,
        "period_s": float(sequence.period_s),
        "source_run_sha256": sequence.source_run_sha256,
        "extraction_tolerance_s": float(sequence.extraction_tolerance_s),
        "steps": [
            {
                "signature": step.signature,
                "gates": {name: bool(value) for name, value in sorted(step.gates.items())},
                "diodes": {name: bool(value) for name, value in sorted(step.diodes.items())},
                "constrained_inductors": list(step.constrained_inductors),
                "saturated_inductors": list(step.saturated_inductors),
                "exit": {
                    "kind": step.exit_condition.kind.value,
                    "target": step.exit_condition.target,
                    "guard_is_current": step.exit_condition.guard_is_current,
                },
                "nominal_duration_s": float(step.nominal_duration_s),
            }
            for step in sequence.steps
        ],
    }


def _require_number(raw: Any, *, name: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ModeSequenceError(f"{name} must be a number")
    return float(raw)


def _require_bool_map(raw: Any, *, name: str) -> dict[str, bool]:
    if not isinstance(raw, dict):
        raise ModeSequenceError(f"{name} must be an object of booleans")
    parsed: dict[str, bool] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, bool):
            raise ModeSequenceError(f"{name} must map string ids to booleans")
        parsed[key] = value
    return parsed


def _require_name_list(raw: Any, *, name: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
        raise ModeSequenceError(f"{name} must be a list of strings")
    return tuple(raw)


def mode_sequence_from_dict(payload: Any) -> ModeSequence:
    """Rebuild a :class:`ModeSequence` from its portable form, fail closed.

    Unknown fields, a wrong schema tag, a malformed step or an unknown exit
    kind are refusals, never defaults — a silently repaired sequence would
    later drive the engine with a plausible but unsourced claim.
    """

    if not isinstance(payload, dict):
        raise ModeSequenceError("a serialized mode sequence must be an object")
    allowed = {
        "schema",
        "period_s",
        "source_run_sha256",
        "extraction_tolerance_s",
        "steps",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ModeSequenceError(
            "serialized mode sequence has unknown field(s): " + ", ".join(unknown)
        )
    if payload.get("schema") != MODE_SEQUENCE_SCHEMA:
        raise ModeSequenceError(
            f"serialized mode sequence must declare schema {MODE_SEQUENCE_SCHEMA!r} "
            f"(got {payload.get('schema')!r})"
        )
    period_s = _require_number(payload.get("period_s"), name="period_s")
    if period_s <= 0.0:
        raise ModeSequenceError("period_s must be positive")
    source = payload.get("source_run_sha256")
    if source is not None and not isinstance(source, str):
        raise ModeSequenceError("source_run_sha256 must be a string or null")
    tolerance = _require_number(
        payload.get("extraction_tolerance_s"), name="extraction_tolerance_s"
    )
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        raise ModeSequenceError("steps must be a non-empty list")
    steps: list[ModeStep] = []
    for index, raw_step in enumerate(raw_steps):
        prefix = f"steps[{index}]"
        if not isinstance(raw_step, dict):
            raise ModeSequenceError(f"{prefix} must be an object")
        step_allowed = {
            "signature",
            "gates",
            "diodes",
            "constrained_inductors",
            "saturated_inductors",
            "exit",
            "nominal_duration_s",
        }
        step_unknown = sorted(set(raw_step) - step_allowed)
        if step_unknown:
            raise ModeSequenceError(
                f"{prefix} has unknown field(s): " + ", ".join(step_unknown)
            )
        missing = sorted(step_allowed - set(raw_step))
        if missing:
            raise ModeSequenceError(
                f"{prefix} is missing field(s): " + ", ".join(missing)
            )
        signature = raw_step["signature"]
        if not isinstance(signature, str) or not signature:
            raise ModeSequenceError(f"{prefix}.signature must be a non-empty string")
        raw_exit = raw_step["exit"]
        if not isinstance(raw_exit, dict) or set(raw_exit) != {
            "kind",
            "target",
            "guard_is_current",
        }:
            raise ModeSequenceError(
                f"{prefix}.exit must carry exactly kind, target, guard_is_current"
            )
        try:
            kind = ExitKind(raw_exit["kind"])
        except ValueError as exc:
            raise ModeSequenceError(
                f"{prefix}.exit.kind {raw_exit['kind']!r} is not one of "
                + ", ".join(item.value for item in ExitKind)
            ) from exc
        target = raw_exit["target"]
        if not isinstance(target, str):
            raise ModeSequenceError(f"{prefix}.exit.target must be a string")
        guard = raw_exit["guard_is_current"]
        if not isinstance(guard, bool):
            raise ModeSequenceError(f"{prefix}.exit.guard_is_current must be a boolean")
        duration = _require_number(
            raw_step["nominal_duration_s"], name=f"{prefix}.nominal_duration_s"
        )
        if duration <= 0.0:
            raise ModeSequenceError(f"{prefix}.nominal_duration_s must be positive")
        steps.append(
            ModeStep(
                signature=signature,
                gates=_require_bool_map(raw_step["gates"], name=f"{prefix}.gates"),
                diodes=_require_bool_map(raw_step["diodes"], name=f"{prefix}.diodes"),
                constrained_inductors=_require_name_list(
                    raw_step["constrained_inductors"],
                    name=f"{prefix}.constrained_inductors",
                ),
                saturated_inductors=_require_name_list(
                    raw_step["saturated_inductors"],
                    name=f"{prefix}.saturated_inductors",
                ),
                exit_condition=ExitCondition(kind, target, guard),
                nominal_duration_s=duration,
            )
        )
    return ModeSequence(
        period_s=period_s,
        steps=tuple(steps),
        source_run_sha256=source,
        extraction_tolerance_s=tolerance,
    )


#: Exit kinds whose instant is EXACT ARITHMETIC rather than a located root.
#:
#: A scheduled gate edge and a period boundary are explicit functions of the
#: declared frequency, phase and duty, so two machines compute them from the
#: same numbers and get the same answer. A guard-located instant does not
#: work that way: it is a root of h(x, t) = 0 found on the integrated
#: trajectory, so it inherits both the integration tolerance and — because
#: adaptive solvers take different internal steps on different platforms —
#: the platform. That difference is not a defect to be fixed; it is what
#: `README.md` states outright about comparing adaptive runs.
EXACTLY_TIMED_EXITS = frozenset({ExitKind.SCHEDULED_GATE, ExitKind.PERIOD_END})


def _step_is_exactly_timed(steps: tuple[ModeStep, ...], position: int) -> bool:
    """Whether step ``position``'s DURATION is exact arithmetic on both ends.

    A duration spans two instants: this step's exit and the previous one's.
    It is only as exact as the looser of the two, so a scheduled edge that
    happens to follow a natural commutation still carries the commutation's
    noise. Step 0 starts at the cycle boundary, which is exact.
    """

    if steps[position].exit_condition.kind not in EXACTLY_TIMED_EXITS:
        return False
    if position == 0:
        return True
    return steps[position - 1].exit_condition.kind in EXACTLY_TIMED_EXITS


def sequences_agree(
    declared: ModeSequence,
    observed: ModeSequence,
    *,
    duration_rtol: float,
    guard_duration_rtol: float | None = None,
) -> tuple[bool, tuple[str, ...]]:
    """Compare a declared sequence against an observed one — verdict, not raise.

    Disagreement is the T3 contract's raw material: the caller wraps the
    reasons into an assumption flag with a margin. Raising here would turn
    "your CCM assumption does not hold" into a crash instead of a result.

    TWO TOLERANCES, BECAUSE THERE ARE TWO KINDS OF INSTANT. ``duration_rtol``
    applies to steps timed by exact arithmetic; ``guard_duration_rtol``
    applies to steps whose boundaries were located by root-finding. Passing
    only the first keeps the previous single-tolerance behaviour exactly, so
    a caller that has not thought about the distinction is not silently given
    a looser gate than it asked for.

    WHY THE SPLIT EXISTS. A line-commutated rectifier has no scheduled edges
    at all: every boundary is a located root. Holding those to a tolerance
    tighter than the run's own integration ``rtol`` asserts more accuracy
    about where the root is than the trajectory it was found on ever had —
    an assertion that passes or fails by which machine ran it.
    """

    guard_rtol = duration_rtol if guard_duration_rtol is None else guard_duration_rtol
    reasons: list[str] = []
    if abs(declared.period_s - observed.period_s) > duration_rtol * observed.period_s:
        reasons.append(
            f"period differs: declared {declared.period_s!r} s vs observed "
            f"{observed.period_s!r} s"
        )
    if declared.signatures() != observed.signatures():
        reasons.append(
            f"mode signatures differ: declared {len(declared.steps)} step(s) "
            f"{list(declared.signatures())} vs observed "
            f"{len(observed.steps)} step(s) {list(observed.signatures())}"
        )
    else:
        for position, (expected, actual) in enumerate(
            zip(declared.steps, observed.steps)
        ):
            scale = max(abs(actual.nominal_duration_s), 1e-15)
            exact = _step_is_exactly_timed(observed.steps, position)
            applied = duration_rtol if exact else guard_rtol
            if (
                abs(expected.nominal_duration_s - actual.nominal_duration_s)
                > applied * scale
            ):
                reasons.append(
                    f"step {position} duration differs beyond "
                    f"{'rtol' if exact else 'guard rtol'} "
                    f"{applied!r}: declared "
                    f"{expected.nominal_duration_s!r} s vs observed "
                    f"{actual.nominal_duration_s!r} s"
                )
            if expected.exit_condition.kind is not actual.exit_condition.kind:
                reasons.append(
                    f"step {position} exit kind differs: declared "
                    f"{expected.exit_condition.kind.value} vs observed "
                    f"{actual.exit_condition.kind.value}"
                )
    return (not reasons, tuple(reasons))
