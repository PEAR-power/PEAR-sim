# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Discrete signal blocks: the executable control layer.

WHY A SECOND DOMAIN. The kernel models an electrical network. A drive's
DESIGN, however, is its controller — and until now a controller could only
exist as one of five ready-made types compiled into Python, so every new
control law was a change to the simulator rather than a change to the model.
This module makes the signal a first-class modelled object: declared blocks,
declared clock, declared wiring, executed in a deterministic order.

WHAT A BLOCK IS. A pure function of its inputs evaluated at a sampling
instant, plus — for exactly one block kind — one sample of memory. Blocks do
not integrate; the kernel integrates. That split is the whole reason the
layer can be executed in a fixed order and audited afterwards.

THE ORDER IS THE CONTRACT. Blocks form a directed graph. A cycle through
combinational blocks is an algebraic loop and is REFUSED, naming both ends:
its value would depend on the order the author happened to write the blocks
in. A cycle through ``unit_delay`` is not a loop at all — it is a difference
equation, the thing every discrete controller is made of — because the delay
answers with the PREVIOUS sample and therefore cuts the dependency. Both
anti-windup by back-calculation and rotor-angle prediction need exactly this,
which is why the distinction is drawn here rather than left to a solver
heuristic.

TWO KINDS OF BRANCH, AND THEY ARE NOT INTERCHANGEABLE. ``switch`` chooses on
a MODEL-level boolean: which control law this run exercises (decoupling on or
off), a property of the experiment, fixed before the first tick. ``select``
chooses on a SIGNAL: which branch this sample takes, a property of the state,
decided again every tick. Collapsing them would make "what did we run?" and
"what did it do?" the same field, and only the first belongs in a comparison
between two runs.

WHY TRUTH VALUES ARE TYPED. Conditions carry the declared unit ``bool`` and
nothing else may reach a condition input. Without that, any dimensionless
signal — a duty ratio, a per-unit voltage — would be accepted as a condition
and silently read as "true whenever non-zero". The check runs one way only:
a truth value entering arithmetic is legitimate (a 0/1 gate multiplying a
term is how a contribution gets switched off), a quantity entering a
condition is not.

DECLARED SCOPE (v0.1). One periodic task; multi-rate control is out of scope
and says so rather than being implied. No continuous-time blocks: a block
that would need its own integrator belongs to the kernel, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Final, Iterable

from pear_sim.errors import ModelError
from pear_sim.spec.controls import _resolve
from pear_sim.spec.signals import is_signal_expression, signal_unit

MAX_SIGNAL_BLOCKS = 256
MAX_BLOCK_INPUTS = 16
MAX_LOOKUP_POINTS = 1024
MAX_DELAY_SAMPLES = 4096

#: Every block kind and the input fields it reads, in declaration order.
BLOCK_KINDS: dict[str, tuple[str, ...]] = {
    "constant": (),
    "probe": (),
    "time_source": (),
    "gain": ("input",),
    "sum": ("inputs",),
    "product": ("inputs",),
    "saturate": ("input",),
    "unit_delay": ("input",),
    "switch": ("if_true", "if_false"),
    "lookup_1d": ("input",),
    "compare": ("left", "right"),
    "logic": ("inputs",),
    "select": ("condition", "if_true", "if_false"),
    "quantize": ("input",),
    "noise": (),
    "delay": ("input",),
    # Rotating-frame transforms. A block yields ONE value, so the axis is
    # part of the declaration rather than the block having two outputs;
    # a controller declares `park` twice, once per axis, and the pair reads
    # as the transform it is.
    "park": ("a", "b", "c", "angle"),
    "inverse_park": ("d", "q", "angle"),
    "lookup_2d": ("x", "y"),
    #: Vector magnitude. Needed for a CIRCULAR voltage limit, which is what
    #: a three-phase inverter actually has — limiting each axis separately
    #: allows |u| up to sqrt(2) times the real bound.
    "hypot": ("x", "y"),
    "divide": ("numerator", "denominator"),
    #: A perturbation source. Declared here rather than synthesised from
    #: `time_source` + arithmetic because a small-signal sweep needs its
    #: excitation to be EXACTLY a sinusoid of a known frequency and phase —
    #: that is what the correlation at the other end assumes.
    "sine": (),
}

#: Kinds that cut a dependency cycle: they answer with the previous sample.
DELAY_KINDS = frozenset({"unit_delay", "delay"})

#: Kinds that read from outside the signal layer.
SOURCE_KINDS = frozenset({"constant", "probe", "time_source", "noise", "sine"})

#: The unit a logical signal carries, and the reason it is not "1".
#:
#: Dimensionless is not the same claim as logical. A duty ratio, a per-unit
#: voltage and a modulation index are all "1", and any of them wired into a
#: condition slot would be read as "true whenever non-zero" — which is a
#: control law nobody wrote. Giving truth values their own declared unit turns
#: that wiring mistake into a load-time error instead of a plausible run.
LOGICAL_UNIT = "bool"

#: Kinds whose output IS a truth value, and whose declared unit is therefore
#: fixed rather than chosen.
LOGICAL_KINDS = frozenset({"compare", "logic"})

#: Comparison operators. Equality is deliberately absent: two floats that
#: went through different arithmetic are equal by luck, so `eq` would be a
#: branch that works in the test and fails in the run. Compare against a
#: threshold with `ge`/`le` instead — that is what the physical question
#: ("has the limiter engaged?") actually asks.
COMPARE_OPS = ("lt", "le", "gt", "ge")

#: Logical operators, with the arity each one requires.
LOGIC_ARITY: dict[str, tuple[int, int]] = {
    "not": (1, 1),
    "and": (2, MAX_BLOCK_INPUTS),
    "or": (2, MAX_BLOCK_INPUTS),
}


@dataclass(frozen=True)
class SignalBlock:
    """One declared block."""

    id: str
    kind: str
    unit: str
    inputs: tuple[str, ...] = ()
    # Kind-specific, validated at parse time and never re-interpreted later.
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def is_delay(self) -> bool:
        return self.kind in DELAY_KINDS


@dataclass(frozen=True)
class SignalTask:
    """The clock a block set runs on.

    Declared, never defaulted: a controller whose sampling rate is implied by
    whatever else is in the model is a controller whose timing nobody chose.
    """

    period: float
    offset: float = 0.0

    def instants(self, start: float, stop: float) -> list[float]:
        """Sampling instants of this task inside ``[start, stop)``."""

        first = start + self.offset
        if first < start:
            first += self.period
        count = int(math.floor((stop - first) / self.period)) + 1
        return [first + index * self.period for index in range(max(0, count))]


@dataclass(frozen=True)
class SignalGraph:
    """The declared signal layer: a task, its blocks, and their run order."""

    task: SignalTask
    blocks: tuple[SignalBlock, ...]
    order: tuple[str, ...]

    @property
    def by_id(self) -> dict[str, SignalBlock]:
        return {block.id: block for block in self.blocks}

    def unit_of(self, block_id: str) -> str:
        return self.by_id[block_id].unit

    def probe_expressions(self) -> tuple[str, ...]:
        return tuple(
            block.params["source"] for block in self.blocks if block.kind == "probe"
        )


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ModelError(f"{name} must be an object")
    return value


def _require_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ModelError(f"{name} must be finite")
    return result


def _require_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelError(f"{name} must be a non-empty string")
    return value


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], *, name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ModelError(f"{name}: unknown field(s): {', '.join(unknown)}")


def _parse_block(
    block_id: str,
    raw: Any,
    *,
    machine_ids: frozenset[str],
    parameters: dict[str, Any],
) -> SignalBlock:
    name = f"signals.blocks.{block_id}"
    body = _require_mapping(raw, name=name)
    kind = body.get("type")
    if kind not in BLOCK_KINDS:
        raise ModelError(
            f"{name}.type must be one of {', '.join(sorted(BLOCK_KINDS))}"
        )
    # The unit is DECLARED, never inferred. A signal layer carries volts,
    # amperes, newton-metres and dimensionless quantities side by side, so
    # there is no syntax to read it off — and a guessed unit is how a torque
    # becomes an ampere (the failure this repo already refused once).
    unit = _require_text(body.get("unit", ""), name=f"{name}.unit")

    allowed = {"type", "unit"} | set(BLOCK_KINDS[kind])
    params: dict[str, Any] = {}
    inputs: tuple[str, ...] = ()

    if kind == "constant":
        allowed |= {"value"}
        params["value"] = _require_number(body.get("value"), name=f"{name}.value")
    elif kind == "probe":
        allowed |= {"source"}
        source = _require_text(body.get("source"), name=f"{name}.source")
        if not is_signal_expression(source):
            raise ModelError(
                f"{name}.source must be a canonical signal expression, got {source!r}"
            )
        declared_unit = signal_unit(source)
        if unit != declared_unit:
            raise ModelError(
                f"{name}.unit is {unit!r} but {source!r} carries {declared_unit!r}; "
                "a probe may not relabel what it reads"
            )
        params["source"] = source
    elif kind == "time_source":
        allowed |= {"shape", "initial", "final", "start", "stop"}
        shape = body.get("shape")
        if shape not in {"step", "ramp"}:
            raise ModelError(f"{name}.shape must be step or ramp")
        params["shape"] = shape
        params["initial"] = _require_number(
            body.get("initial", 0.0), name=f"{name}.initial"
        )
        params["final"] = _require_number(body.get("final"), name=f"{name}.final")
        params["start"] = _resolve(
            body.get("start", 0.0), parameters, "time", name=f"{name}.start"
        )
        if shape == "ramp":
            params["stop"] = _resolve(
                body.get("stop"), parameters, "time", name=f"{name}.stop"
            )
            if params["stop"] <= params["start"]:
                raise ModelError(f"{name}.stop must be later than start")
        elif "stop" in body:
            raise ModelError(f"{name}.stop is meaningless for a step")
    elif kind == "gain":
        allowed |= {"gain"}
        params["gain"] = _require_number(body.get("gain"), name=f"{name}.gain")
        inputs = (_require_text(body.get("input"), name=f"{name}.input"),)
    elif kind in {"sum", "product"}:
        raw_inputs = body.get("inputs")
        if not isinstance(raw_inputs, list) or len(raw_inputs) < 2:
            raise ModelError(f"{name}.inputs must list at least two blocks")
        if len(raw_inputs) > MAX_BLOCK_INPUTS:
            raise ModelError(
                f"{name}.inputs exceeds the portable limit of {MAX_BLOCK_INPUTS}"
            )
        inputs = tuple(
            _require_text(item, name=f"{name}.inputs[{index}]")
            for index, item in enumerate(raw_inputs)
        )
        if kind == "sum":
            allowed |= {"signs"}
            raw_signs = body.get("signs")
            if raw_signs is None:
                signs = tuple(1.0 for _ in inputs)
            else:
                if not isinstance(raw_signs, list) or len(raw_signs) != len(inputs):
                    raise ModelError(
                        f"{name}.signs must have exactly one entry per input"
                    )
                for index, item in enumerate(raw_signs):
                    if item not in (1, -1):
                        raise ModelError(
                            f"{name}.signs[{index}] must be 1 or -1"
                        )
                signs = tuple(float(item) for item in raw_signs)
            params["signs"] = signs
    elif kind == "saturate":
        allowed |= {"min", "max"}
        params["min"] = _require_number(body.get("min"), name=f"{name}.min")
        params["max"] = _require_number(body.get("max"), name=f"{name}.max")
        if params["min"] >= params["max"]:
            raise ModelError(f"{name}.min must be below max")
        inputs = (_require_text(body.get("input"), name=f"{name}.input"),)
    elif kind == "unit_delay":
        allowed |= {"initial"}
        params["initial"] = _require_number(
            body.get("initial", 0.0), name=f"{name}.initial"
        )
        inputs = (_require_text(body.get("input"), name=f"{name}.input"),)
    elif kind == "switch":
        allowed |= {"flag"}
        # A flag is a MODEL-level boolean, not a signal: it selects which
        # control law the run exercises (decoupling on/off, voltage limit
        # on/off), which is a property of the experiment, not of the state.
        flag = body.get("flag")
        if not isinstance(flag, bool):
            raise ModelError(f"{name}.flag must be true or false")
        params["flag"] = flag
        inputs = (
            _require_text(body.get("if_true"), name=f"{name}.if_true"),
            _require_text(body.get("if_false"), name=f"{name}.if_false"),
        )
    elif kind == "lookup_1d":
        allowed |= {"axis", "values", "extrapolation"}
        axis = body.get("axis")
        values = body.get("values")
        for label, series in (("axis", axis), ("values", values)):
            if not isinstance(series, list) or len(series) < 2:
                raise ModelError(f"{name}.{label} must list at least two numbers")
            if len(series) > MAX_LOOKUP_POINTS:
                raise ModelError(
                    f"{name}.{label} exceeds the portable limit of "
                    f"{MAX_LOOKUP_POINTS} points"
                )
        if len(axis) != len(values):
            raise ModelError(f"{name}.axis and values must have equal length")
        axis_values = tuple(
            _require_number(item, name=f"{name}.axis[{index}]")
            for index, item in enumerate(axis)
        )
        for index in range(1, len(axis_values)):
            if axis_values[index] <= axis_values[index - 1]:
                raise ModelError(f"{name}.axis must be strictly increasing")
        params["axis"] = axis_values
        params["values"] = tuple(
            _require_number(item, name=f"{name}.values[{index}]")
            for index, item in enumerate(values)
        )
        # No default: silently holding the edge value is a claim about
        # behaviour outside the measured range (the same rule as clamped loss tables).
        extrapolation = body.get("extrapolation")
        if extrapolation not in {"hold", "error"}:
            raise ModelError(
                f"{name}.extrapolation must be declared as hold or error"
            )
        params["extrapolation"] = extrapolation
        inputs = (_require_text(body.get("input"), name=f"{name}.input"),)
    elif kind == "compare":
        allowed |= {"op"}
        op = body.get("op")
        if op not in COMPARE_OPS:
            raise ModelError(
                f"{name}.op must be one of {', '.join(COMPARE_OPS)}; equality is "
                "not offered because two floats reached by different arithmetic "
                "are equal only by luck"
            )
        params["op"] = op
        inputs = (
            _require_text(body.get("left"), name=f"{name}.left"),
            _require_text(body.get("right"), name=f"{name}.right"),
        )
    elif kind == "logic":
        allowed |= {"op"}
        op = body.get("op")
        if op not in LOGIC_ARITY:
            raise ModelError(
                f"{name}.op must be one of {', '.join(sorted(LOGIC_ARITY))}"
            )
        params["op"] = op
        raw_inputs = body.get("inputs")
        low, high = LOGIC_ARITY[op]
        if not isinstance(raw_inputs, list):
            raise ModelError(f"{name}.inputs must be a list")
        if not low <= len(raw_inputs) <= high:
            expected = (
                f"exactly {low}" if low == high else f"between {low} and {high}"
            )
            raise ModelError(
                f"{name}.inputs must list {expected} block(s) for op {op!r}, "
                f"got {len(raw_inputs)}"
            )
        inputs = tuple(
            _require_text(item, name=f"{name}.inputs[{index}]")
            for index, item in enumerate(raw_inputs)
        )
    elif kind == "select":
        inputs = (
            _require_text(body.get("condition"), name=f"{name}.condition"),
            _require_text(body.get("if_true"), name=f"{name}.if_true"),
            _require_text(body.get("if_false"), name=f"{name}.if_false"),
        )

    # A truth value is not a quantity the author gets to name. Every other
    # block declares its unit because no syntax can tell volts from newton
    # metres; these two carry one thing only, so a declaration that says
    # otherwise is a mistake worth naming rather than a preference to honour.
    if kind in LOGICAL_KINDS and unit != LOGICAL_UNIT:
        raise ModelError(
            f"{name}.unit is {unit!r} but a {kind} block carries "
            f"{LOGICAL_UNIT!r}; a truth value is not a physical quantity"
        )

    elif kind == "quantize":
        allowed |= {"step", "mode"}
        params["step"] = _require_number(body.get("step"), name=f"{name}.step")
        if params["step"] <= 0.0:
            raise ModelError(f"{name}.step must be positive")
        mode = body.get("mode", "round")
        if mode not in {"round", "truncate"}:
            raise ModelError(f"{name}.mode must be round or truncate")
        params["mode"] = mode
        inputs = (_require_text(body.get("input"), name=f"{name}.input"),)
    elif kind == "noise":
        allowed |= {"stddev", "seed"}
        params["stddev"] = _require_number(body.get("stddev"), name=f"{name}.stddev")
        if params["stddev"] < 0.0:
            raise ModelError(f"{name}.stddev must be non-negative")
        # A seed is REQUIRED, never defaulted. The promise this
        # repository makes is seed -> bit; a generator whose seed the model
        # does not state cannot keep it, and "it was random" is not a
        # reproducible run.
        seed = body.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ModelError(
                f"{name}.seed must be declared as an integer — a noise source "
                "without a stated seed cannot be reproduced"
            )
        if seed < 0:
            raise ModelError(f"{name}.seed must be non-negative")
        params["seed"] = int(seed)
    elif kind == "delay":
        allowed |= {"samples", "initial"}
        samples = body.get("samples")
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ModelError(f"{name}.samples must be an integer >= 1")
        if samples > MAX_DELAY_SAMPLES:
            raise ModelError(
                f"{name}.samples exceeds the portable limit of "
                f"{MAX_DELAY_SAMPLES}"
            )
        params["samples"] = samples
        params["initial"] = _require_number(
            body.get("initial", 0.0), name=f"{name}.initial"
        )
        inputs = (_require_text(body.get("input"), name=f"{name}.input"),)

    elif kind == "park":
        allowed |= {"axis"}
        axis = body.get("axis")
        if axis not in {"d", "q"}:
            raise ModelError(f"{name}.axis must be d or q")
        params["axis"] = axis
        inputs = tuple(
            _require_text(body.get(field), name=f"{name}.{field}")
            for field in ("a", "b", "c", "angle")
        )
    elif kind == "inverse_park":
        allowed |= {"axis"}
        axis = body.get("axis")
        if axis not in {"alpha", "beta"}:
            raise ModelError(f"{name}.axis must be alpha or beta")
        params["axis"] = axis
        inputs = tuple(
            _require_text(body.get(field), name=f"{name}.{field}")
            for field in ("d", "q", "angle")
        )
    elif kind == "sine":
        allowed |= {"amplitude", "frequency", "phase_rad", "offset"}
        params["amplitude"] = _require_number(
            body.get("amplitude"), name=f"{name}.amplitude"
        )
        params["frequency"] = _resolve(
            body.get("frequency"), parameters, "frequency", name=f"{name}.frequency"
        )
        if params["frequency"] <= 0.0:
            raise ModelError(f"{name}.frequency must be positive")
        params["phase_rad"] = _require_number(
            body.get("phase_rad", 0.0), name=f"{name}.phase_rad"
        )
        params["offset"] = _require_number(
            body.get("offset", 0.0), name=f"{name}.offset"
        )
    elif kind == "divide":
        inputs = (
            _require_text(body.get("numerator"), name=f"{name}.numerator"),
            _require_text(body.get("denominator"), name=f"{name}.denominator"),
        )
    elif kind == "hypot":
        inputs = (
            _require_text(body.get("x"), name=f"{name}.x"),
            _require_text(body.get("y"), name=f"{name}.y"),
        )
    elif kind == "lookup_2d":
        allowed |= {"x_axis", "y_axis", "values", "extrapolation"}
        for label in ("x_axis", "y_axis"):
            series = body.get(label)
            if not isinstance(series, list) or len(series) < 2:
                raise ModelError(f"{name}.{label} must list at least two numbers")
            if len(series) > MAX_LOOKUP_POINTS:
                raise ModelError(
                    f"{name}.{label} exceeds the portable limit of "
                    f"{MAX_LOOKUP_POINTS} points"
                )
        x_axis = tuple(
            _require_number(item, name=f"{name}.x_axis[{index}]")
            for index, item in enumerate(body["x_axis"])
        )
        y_axis = tuple(
            _require_number(item, name=f"{name}.y_axis[{index}]")
            for index, item in enumerate(body["y_axis"])
        )
        for label, axis_values in (("x_axis", x_axis), ("y_axis", y_axis)):
            for index in range(1, len(axis_values)):
                if axis_values[index] <= axis_values[index - 1]:
                    raise ModelError(f"{name}.{label} must be strictly increasing")
        grid = body.get("values")
        if not isinstance(grid, list) or len(grid) != len(x_axis):
            raise ModelError(
                f"{name}.values must have one row per x_axis entry "
                f"({len(x_axis)} rows)"
            )
        rows: list[tuple[float, ...]] = []
        for row_index, row in enumerate(grid):
            if not isinstance(row, list) or len(row) != len(y_axis):
                raise ModelError(
                    f"{name}.values[{row_index}] must have one entry per y_axis "
                    f"value ({len(y_axis)} entries)"
                )
            rows.append(
                tuple(
                    _require_number(item, name=f"{name}.values[{row_index}][{col}]")
                    for col, item in enumerate(row)
                )
            )
        params["x_axis"] = x_axis
        params["y_axis"] = y_axis
        params["values"] = tuple(rows)
        extrapolation = body.get("extrapolation")
        if extrapolation not in {"hold", "error"}:
            raise ModelError(
                f"{name}.extrapolation must be declared as hold or error"
            )
        params["extrapolation"] = extrapolation
        inputs = (
            _require_text(body.get("x"), name=f"{name}.x"),
            _require_text(body.get("y"), name=f"{name}.y"),
        )

    _reject_unknown(body, allowed, name=name)
    del machine_ids  # bound by the loader through the probe expression
    return SignalBlock(id=block_id, kind=kind, unit=unit, inputs=inputs, params=params)


def _logical_sinks(block: SignalBlock) -> tuple[str, ...]:
    """Inputs of ``block`` that must be fed a truth value.

    Only SINKS are constrained, never sources. Feeding a truth value into
    arithmetic is a legitimate and common technique — multiplying a term by a
    0/1 gate is how a contribution gets switched off — so a `bool` reaching a
    numeric input is not an error. The reverse is: a volt arriving where a
    condition was expected would be read as "true whenever non-zero", which is
    a control law nobody wrote and no number afterwards would look wrong.
    """

    if block.kind == "logic":
        return block.inputs
    if block.kind == "select":
        return block.inputs[:1]
    return ()


def _check_logical_sinks(blocks: tuple[SignalBlock, ...]) -> None:
    units = {block.id: block.unit for block in blocks}
    for block in blocks:
        for source in _logical_sinks(block):
            if units[source] != LOGICAL_UNIT:
                raise ModelError(
                    f"signals.blocks.{block.id} expects a condition, but "
                    f"{source!r} carries {units[source]!r} rather than "
                    f"{LOGICAL_UNIT!r}; compare it against a threshold first"
                )


def _resolve_order(blocks: tuple[SignalBlock, ...]) -> tuple[str, ...]:
    """Deterministic execution order, or a named algebraic loop.

    Edges are drawn only through COMBINATIONAL inputs. A ``unit_delay``
    consumes the previous sample, so its input edge is deliberately absent
    from the graph — that is precisely what makes a difference equation
    legal while an algebraic loop stays an error.

    Ties are broken by block id so the order is a function of the graph, not
    of the order the author happened to write the blocks in.
    """

    by_id = {block.id: block for block in blocks}
    dependencies: dict[str, set[str]] = {}
    for block in blocks:
        if block.is_delay:
            dependencies[block.id] = set()
            continue
        dependencies[block.id] = set(block.inputs)

    order: list[str] = []
    ready = sorted(
        block_id for block_id, pending in dependencies.items() if not pending
    )
    remaining = {
        block_id: set(pending) for block_id, pending in dependencies.items()
    }
    while ready:
        block_id = ready.pop(0)
        order.append(block_id)
        newly_ready = []
        for other, pending in remaining.items():
            if block_id in pending:
                pending.discard(block_id)
                if not pending and other not in order and other not in ready:
                    newly_ready.append(other)
        for candidate in sorted(newly_ready):
            ready.append(candidate)
        ready.sort()
        remaining.pop(block_id, None)

    if len(order) != len(blocks):
        stuck = sorted(set(by_id) - set(order))
        cycle = _describe_cycle(stuck, dependencies)
        raise ModelError(
            "signals: algebraic loop through "
            f"{cycle} — a value that depends on itself within one sample has "
            "no defined order; break it with a unit_delay, which is what a "
            "difference equation actually is"
        )
    return tuple(order)


def _describe_cycle(stuck: Iterable[str], dependencies: dict[str, set[str]]) -> str:
    """Name one concrete cycle rather than dumping every unresolved block."""

    stuck = list(stuck)
    if not stuck:
        return "the signal graph"
    start = stuck[0]
    path = [start]
    seen = {start}
    current = start
    while True:
        candidates = sorted(dependencies.get(current, set()) & set(stuck))
        if not candidates:
            break
        nxt = candidates[0]
        if nxt in seen:
            path.append(nxt)
            break
        path.append(nxt)
        seen.add(nxt)
        current = nxt
    return " -> ".join(path)


_PI_FIELDS = {"type", "unit", "input", "kp", "ki", "min", "max", "kaw", "initial"}


def _expand_pi_blocks(raw_blocks: dict[str, Any], period: float) -> dict[str, Any]:
    """Rewrite each ``type: pi`` block into the primitive blocks it means.

    A discrete PI controller with backward-Euler integration and, when output
    limits are given, back-calculation anti-windup:

        int[k] = int[k-1] + ki*Ts*e[k] + kaw*(u[k-1] - raw[k-1])
        raw[k] = kp*e[k] + int[k];   u[k] = clamp(raw[k], min, max)

    The saturation gap re-enters the integrator one sample late, exactly as a
    real implementation computes the limiter after the controller. Expansion is
    deterministic and happens before validation, so the expanded graph obeys
    every rule a hand-written one does; the controller output keeps the PI
    block's own id, and its internals are ``<id>__<part>``.
    """

    expanded: dict[str, Any] = {}
    for block_id, body in raw_blocks.items():
        if not isinstance(body, dict) or body.get("type") != "pi":
            expanded[block_id] = body
            continue
        name = f"signals.blocks.{block_id}"
        _reject_unknown(body, _PI_FIELDS, name=name)
        unit = _require_text(body.get("unit", ""), name=f"{name}.unit")
        error = _require_text(body.get("input"), name=f"{name}.input")
        kp = _require_number(body.get("kp", 0.0), name=f"{name}.kp")
        ki = _require_number(body.get("ki", 0.0), name=f"{name}.ki")
        initial = _require_number(body.get("initial", 0.0), name=f"{name}.initial")
        limited = "min" in body or "max" in body
        if limited and not ("min" in body and "max" in body):
            raise ModelError(f"{name}: give both min and max, or neither")
        kaw = _require_number(body.get("kaw", 1.0 if limited else 0.0), name=f"{name}.kaw")
        if kaw and not limited:
            raise ModelError(f"{name}.kaw needs output limits (min and max)")
        part = lambda suffix: f"{block_id}__{suffix}"  # noqa: E731
        for suffix in ("p", "i_in", "int", "int_prev", "raw", "gap", "gap_prev", "aw"):
            if part(suffix) in raw_blocks:
                raise ModelError(f"{name}: block id {part(suffix)!r} is reserved by the pi block")
        expanded[part("p")] = {"type": "gain", "input": error, "gain": kp, "unit": unit}
        expanded[part("i_in")] = {"type": "gain", "input": error, "gain": ki * period, "unit": unit}
        expanded[part("int_prev")] = {
            "type": "unit_delay", "input": part("int"), "initial": initial, "unit": unit,
        }
        integrator_inputs = [part("int_prev"), part("i_in")]
        if kaw:
            expanded[part("gap_prev")] = {
                "type": "unit_delay", "input": part("gap"), "initial": 0.0, "unit": unit,
            }
            expanded[part("aw")] = {"type": "gain", "input": part("gap_prev"), "gain": kaw, "unit": unit}
            integrator_inputs.append(part("aw"))
        expanded[part("int")] = {"type": "sum", "inputs": integrator_inputs, "unit": unit}
        if limited:
            expanded[part("raw")] = {"type": "sum", "inputs": [part("p"), part("int")], "unit": unit}
            expanded[block_id] = {
                "type": "saturate", "input": part("raw"),
                "min": body["min"], "max": body["max"], "unit": unit,
            }
            if kaw:
                expanded[part("gap")] = {
                    "type": "sum", "inputs": [block_id, part("raw")], "signs": [1, -1], "unit": unit,
                }
        else:
            expanded[block_id] = {"type": "sum", "inputs": [part("p"), part("int")], "unit": unit}
    return expanded


def parse_signal_graph(
    raw: Any,
    *,
    machine_ids: frozenset[str],
    parameters: dict[str, Any] | None = None,
) -> SignalGraph | None:
    """Validate the ``signals`` section; returns None when absent."""

    if raw is None:
        return None
    body = _require_mapping(raw, name="signals")
    _reject_unknown(body, {"task", "blocks"}, name="signals")
    raw_task = _require_mapping(body.get("task"), name="signals.task")
    _reject_unknown(raw_task, {"period", "offset"}, name="signals.task")
    if "period" not in raw_task:
        raise ModelError("signals.task requires period")
    # Times carry units like everywhere else in the format; block VALUES do
    # not, because the block already declares its unit and a second spelling
    # of the same fact is a second thing that can disagree.
    period = _resolve(
        raw_task["period"], parameters or {}, "time", name="signals.task.period"
    )
    if period <= 0.0:
        raise ModelError("signals.task.period must be positive")
    offset = _resolve(
        raw_task.get("offset", 0.0), parameters or {}, "time", name="signals.task.offset"
    )
    if not 0.0 <= offset < period:
        raise ModelError("signals.task.offset must be in [0, period)")
    task = SignalTask(period=period, offset=offset)

    raw_blocks = _expand_pi_blocks(
        _require_mapping(body.get("blocks"), name="signals.blocks"), period
    )
    if not raw_blocks:
        raise ModelError("signals.blocks must declare at least one block")
    if len(raw_blocks) > MAX_SIGNAL_BLOCKS:
        raise ModelError(
            f"signals.blocks declares {len(raw_blocks)} blocks; the portable "
            f"limit is {MAX_SIGNAL_BLOCKS}"
        )
    blocks = tuple(
        _parse_block(
            block_id,
            raw_blocks[block_id],
            machine_ids=machine_ids,
            parameters=parameters or {},
        )
        for block_id in sorted(raw_blocks)
    )
    known = {block.id for block in blocks}
    for block in blocks:
        for source in block.inputs:
            if source not in known:
                raise ModelError(
                    f"signals.blocks.{block.id} reads {source!r}, which is not a "
                    f"declared block (declared: {', '.join(sorted(known))})"
                )
            if source == block.id and not block.is_delay:
                raise ModelError(
                    f"signals.blocks.{block.id} reads itself without a delay"
                )
    _check_logical_sinks(blocks)
    order = _resolve_order(blocks)
    return SignalGraph(task=task, blocks=blocks, order=order)


def evaluate_block(
    block: SignalBlock,
    values: dict[str, float],
    previous: dict[str, float],
    *,
    time: float,
    probe_reader: Any,
    tick: int = 0,
    buffers: dict[str, list[float]] | None = None,
) -> float:
    """Value of one block at ``time`` given this sample's resolved inputs.

    ``tick`` is the task tick index; a noise sample is a function of the
    declared seed AND the tick, never of how many times the graph happened
    to be evaluated. ``buffers`` holds the shift register of every multi-tick
    ``delay``; a caller that omits it gets each delay's initial value, which
    is what an isolated single-shot evaluation should see.
    """

    kind = block.kind
    if kind == "constant":
        return float(block.params["value"])
    if kind == "probe":
        return float(probe_reader(block.params["source"]))
    if kind == "time_source":
        return _time_source_value(block, time)
    if kind == "unit_delay":
        # The defining property: answers with the previous sample OF ITS
        # INPUT, which is what lets a difference equation close a loop
        # legally. Reading its own previous value instead is a subtle and
        # very quiet bug — the block still produces numbers, still updates,
        # and an accumulator built from it simply never accumulates.
        return float(previous.get(block.inputs[0], block.params["initial"]))
    if kind == "gain":
        return float(block.params["gain"]) * values[block.inputs[0]]
    if kind == "sum":
        return float(
            sum(
                sign * values[source]
                for sign, source in zip(block.params["signs"], block.inputs, strict=True)
            )
        )
    if kind == "product":
        product = 1.0
        for source in block.inputs:
            product *= values[source]
        return float(product)
    if kind == "saturate":
        return float(
            min(max(values[block.inputs[0]], block.params["min"]), block.params["max"])
        )
    if kind == "switch":
        chosen = block.inputs[0] if block.params["flag"] else block.inputs[1]
        return float(values[chosen])
    if kind == "lookup_1d":
        return _lookup_1d(block, values[block.inputs[0]])
    if kind == "park":
        return _park(block, values)
    if kind == "inverse_park":
        return _inverse_park(block, values)
    if kind == "lookup_2d":
        return _lookup_2d(block, values[block.inputs[0]], values[block.inputs[1]])
    if kind == "sine":
        return float(
            block.params["offset"]
            + block.params["amplitude"]
            * math.sin(
                2.0 * math.pi * block.params["frequency"] * time
                + block.params["phase_rad"]
            )
        )
    if kind == "divide":
        denominator = values[block.inputs[1]]
        if denominator == 0.0:
            # No silent infinity and no silent zero. A controller that
            # divides by zero has a modelling error, and the run should say
            # which block rather than propagate a NaN into a gate command.
            raise ModelError(
                f"signals.blocks.{block.id}: division by zero — guard the "
                "denominator with a saturate block if zero is reachable"
            )
        return float(values[block.inputs[0]] / denominator)
    if kind == "hypot":
        return float(math.hypot(values[block.inputs[0]], values[block.inputs[1]]))
    if kind == "quantize":
        step = float(block.params["step"])
        scaled = values[block.inputs[0]] / step
        rounded = math.floor(scaled) if block.params["mode"] == "truncate" else round(scaled)
        return float(rounded * step)
    if kind == "noise":
        # Deterministic by construction: the sample is a pure function
        # of (seed, tick), so replaying a run — or evaluating the graph twice
        # for any reason — cannot change it.
        stddev = float(block.params["stddev"])
        if stddev == 0.0:
            return 0.0
        generator = random.Random((int(block.params["seed"]) << 32) ^ int(tick))
        return float(stddev * generator.gauss(0.0, 1.0))
    if kind == "delay":
        # Until the shift register is FULL the block reports its declared
        # initial value: a delay of N samples means the first N ticks have
        # nothing that old to show. Returning the oldest available sample
        # instead would silently make the delay shorter at start-up, which
        # is a phase error precisely where the transient is.
        history = (buffers or {}).get(block.id) or []
        if len(history) < int(block.params["samples"]):
            return float(block.params["initial"])
        return float(history[0])
    if kind == "compare":
        left = values[block.inputs[0]]
        right = values[block.inputs[1]]
        op = block.params["op"]
        if op == "lt":
            return 1.0 if left < right else 0.0
        if op == "le":
            return 1.0 if left <= right else 0.0
        if op == "gt":
            return 1.0 if left > right else 0.0
        return 1.0 if left >= right else 0.0
    if kind == "logic":
        op = block.params["op"]
        # Truth is read as "not zero" rather than "equals one" so that a gate
        # arriving through arithmetic still behaves; the blocks in LOGICAL_KINDS
        # only ever emit an exact 0.0 or 1.0 themselves.
        states = [values[source] != 0.0 for source in block.inputs]
        if op == "not":
            return 0.0 if states[0] else 1.0
        if op == "and":
            return 1.0 if all(states) else 0.0
        return 1.0 if any(states) else 0.0
    if kind == "select":
        condition, if_true, if_false = block.inputs
        chosen = if_true if values[condition] != 0.0 else if_false
        return float(values[chosen])
    raise ModelError(f"signal block kind {kind!r} has no evaluation rule")


def _time_source_value(block: SignalBlock, time: float) -> float:
    initial = float(block.params["initial"])
    final = float(block.params["final"])
    start = float(block.params["start"])
    if block.params["shape"] == "step":
        return final if time >= start else initial
    stop = float(block.params["stop"])
    if time <= start:
        return initial
    if time >= stop:
        return final
    return initial + (time - start) / (stop - start) * (final - initial)


def _lookup_1d(block: SignalBlock, argument: float) -> float:
    axis = block.params["axis"]
    values = block.params["values"]
    if argument < axis[0] or argument > axis[-1]:
        if block.params["extrapolation"] == "error":
            raise ModelError(
                f"signals.blocks.{block.id}: input {argument:.17g} is outside the "
                f"declared axis [{axis[0]:.17g}, {axis[-1]:.17g}] and the block "
                "declares extrapolation: error"
            )
        return float(values[0] if argument < axis[0] else values[-1])
    for index in range(1, len(axis)):
        if argument <= axis[index]:
            span = axis[index] - axis[index - 1]
            fraction = (argument - axis[index - 1]) / span
            return float(
                values[index - 1] + fraction * (values[index] - values[index - 1])
            )
    return float(values[-1])


#: Phase offsets of the rotating-frame transforms. IDENTICAL to
#: `pear_sim.spec.machines.PHASE_OFFSETS` by construction, and a test asserts
#: it: a controller that transformed on a different convention than the
#: machine it drives would produce a plausible, wrong torque, and nothing in
#: an aggregate metric would show it. That failure mode has already been hit
#: three times in this repository.
_PARK_OFFSETS: Final[tuple[float, float, float]] = (
    0.0,
    2.0 * math.pi / 3.0,
    -2.0 * math.pi / 3.0,
)


def _park(block: SignalBlock, values: dict[str, float]) -> float:
    """abc -> dq, amplitude-invariant, on the machine's own convention."""

    phases = [values[name] for name in block.inputs[:3]]
    angle = values[block.inputs[3]]
    if block.params["axis"] == "d":
        return (2.0 / 3.0) * sum(
            value * math.cos(angle - offset)
            for value, offset in zip(phases, _PARK_OFFSETS, strict=True)
        )
    return -(2.0 / 3.0) * sum(
        value * math.sin(angle - offset)
        for value, offset in zip(phases, _PARK_OFFSETS, strict=True)
    )


def _inverse_park(block: SignalBlock, values: dict[str, float]) -> float:
    """dq -> alpha/beta, the stationary-frame vector a modulator wants."""

    d_value = values[block.inputs[0]]
    q_value = values[block.inputs[1]]
    angle = values[block.inputs[2]]
    if block.params["axis"] == "alpha":
        return d_value * math.cos(angle) - q_value * math.sin(angle)
    return d_value * math.sin(angle) + q_value * math.cos(angle)


def _lookup_2d(block: SignalBlock, x_value: float, y_value: float) -> float:
    """Bilinear interpolation on a declared grid — the flux-map shape."""

    x_axis = block.params["x_axis"]
    y_axis = block.params["y_axis"]
    grid = block.params["values"]
    outside = (
        x_value < x_axis[0]
        or x_value > x_axis[-1]
        or y_value < y_axis[0]
        or y_value > y_axis[-1]
    )
    if outside and block.params["extrapolation"] == "error":
        raise ModelError(
            f"signals.blocks.{block.id}: input ({x_value:.17g}, {y_value:.17g}) "
            f"is outside the declared grid [{x_axis[0]:.17g}, {x_axis[-1]:.17g}] "
            f"x [{y_axis[0]:.17g}, {y_axis[-1]:.17g}] and the block declares "
            "extrapolation: error"
        )
    x_clamped = min(max(x_value, x_axis[0]), x_axis[-1])
    y_clamped = min(max(y_value, y_axis[0]), y_axis[-1])

    def _bracket(axis: tuple[float, ...], value: float) -> tuple[int, float]:
        for index in range(1, len(axis)):
            if value <= axis[index]:
                span = axis[index] - axis[index - 1]
                return index - 1, (value - axis[index - 1]) / span
        return len(axis) - 2, 1.0

    row, fx = _bracket(x_axis, x_clamped)
    col, fy = _bracket(y_axis, y_clamped)
    lower = grid[row][col] + fy * (grid[row][col + 1] - grid[row][col])
    upper = grid[row + 1][col] + fy * (grid[row + 1][col + 1] - grid[row + 1][col])
    return float(lower + fx * (upper - lower))
