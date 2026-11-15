# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Closed-loop periodic steady state and Floquet multipliers.

WHAT IS SOLVED. A converter under a sampled controller repeats one cycle in
steady state, but the cycle is a function of TWO kinds of state: the circuit
(inductor currents, capacitor voltages) and the controller's memory (every
``unit_delay`` register of the signal layer). Shooting on the circuit alone
restarts the controller from its declared memory each cycle and finds a fixed
point of the wrong map — which is why :mod:`pear_sim.analysis.steady_state`
refuses closed-loop models. Here the state is augmented,

    z = [circuit states ; unit_delay registers],

and the one-cycle map ``F(z)`` is ONE PERIOD OF THE PRODUCTION SIMULATOR
started from ``z``: circuit states through the components' initial values,
registers through each ``unit_delay``'s ``initial`` (which is what the first
tick of a run reads). After the cycle, the circuit state is the state at the
period end and each register is what its ``unit_delay`` will answer at the
NEXT tick, i.e. the value its input block took at tick 0. There is no second
model of the controller or the plant in this module.

WHAT IS REPORTED. The fixed point ``z* = F(z*)`` by Newton with a
forward-difference Jacobian, then the monodromy ``M = dF/dz`` at ``z*`` by
central differences, and its eigenvalues — the Floquet multipliers. All
``|mu| < 1``: a small perturbation of the orbit decays cycle by cycle (local
stability). One ``|mu| > 1``: it grows, at the frequency the multiplier's
angle names. Nothing is claimed about global stability, other orbits, or
robustness to parameter changes.

WHY THE VERDICT CAN BE "NOT RESOLVED". Finite differences on an adaptive
integrator carry noise of order ``rtol / step``. The monodromy is computed at
two step sizes and the spread of the multipliers is reported as their
uncertainty; a multiplier whose distance from the unit circle is inside that
spread gets no verdict. A number that cannot be told from 1 is not "stable".

WHEN THE DERIVATIVE DOES NOT EXIST. A clipping limiter, a clamped duty or a
change of the switching-mode sequence between the orbit and a perturbed run
makes ``F`` non-differentiable at ``z*``: the multipliers would describe a
kink, not the orbit. Each is checked on EVERY run that enters the derivative
and reported as an assumption with its margin; if any fails, no multipliers
are reported.

DECLARED SCOPE (v1). Control laws written in the signal layer and driving
``signal_pwm``; time-invariant smooth blocks only (constant, probe, gain,
sum, product, divide, saturate, unit_delay, model-level switch); DC sources;
no machines, saturable inductors, inductor-star invariants or forced gate
windows. Everything else is refused by name — a refused model is a stated
limit, a silently linearised kink is a wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from pear_sim.core.circuit import Circuit
from pear_sim.core.solver import HybridSimulator, SimulationResult
from pear_sim.errors import ModelError
from pear_sim.spec.model import Model
from pear_sim.spec.signals import declared_signal_target

__all__ = [
    "ClosedLoopAssumption",
    "ClosedLoopSteadyState",
    "ClosedLoopSteadyStateError",
    "FloquetMultiplier",
    "solve_closed_loop_steady_state",
]

DEFAULT_MAX_ITERATIONS = 30
DEFAULT_TOLERANCE = 1e-9
DEFAULT_RELATIVE_STEP = 1e-4
# Floor of a probe step, in the state's own SI unit. A register that sits at
# zero (an integrator at a zero-error orbit) still needs a non-zero probe.
_STEP_FLOOR = 1e-7

#: Block kinds whose output is a time-invariant, smooth (or, for saturate,
#: smooth away from its limits) function of the inputs. `switch` branches on
#: a MODEL-level flag fixed before the first tick, so it is a constant choice.
SUPPORTED_BLOCK_KINDS = frozenset(
    {
        "constant",
        "probe",
        "gain",
        "sum",
        "product",
        "divide",
        "saturate",
        "unit_delay",
        "switch",
    }
)

_UNSUPPORTED_REASON = {
    "time_source": "time-varying (the cycle map is not autonomous)",
    "sine": "time-varying (the cycle map is not autonomous)",
    "noise": "time-varying (the cycle map is not autonomous)",
    "delay": "multi-sample memory is not part of the v1 augmented state",
    "compare": "a threshold makes the map non-smooth",
    "logic": "boolean logic makes the map non-smooth",
    "select": "a state-dependent branch makes the map non-smooth",
    "quantize": "quantisation makes the map non-smooth",
    "lookup_1d": "piecewise-linear breakpoints make the map non-smooth",
    "lookup_2d": "piecewise-linear breakpoints make the map non-smooth",
}


class ClosedLoopSteadyStateError(ModelError):
    """The model is outside the closed-loop steady-state v1 scope."""

    code = "analysis.closed_loop_steady_state_invalid"


@dataclass(frozen=True)
class ClosedLoopAssumption:
    """A differentiability assumption, checked on every derivative run.

    ``margin`` is the worst case over those runs in the guard's own units;
    ``held`` is the verdict on the assumption, never on the physics.
    """

    name: str
    held: bool
    margin: float
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "held": self.held,
            "margin": self.margin,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FloquetMultiplier:
    real: float
    imag: float
    magnitude: float
    #: |angle| * f_cycle / (2 pi): the frequency a perturbation along this
    #: mode oscillates at, aliased into [0, f_cycle / 2].
    frequency_hz: float
    #: Spread of this multiplier between the two probe step sizes.
    uncertainty: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "real": self.real,
            "imag": self.imag,
            "magnitude": self.magnitude,
            "frequency_hz": self.frequency_hz,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class ClosedLoopSteadyState:
    converged: bool
    iterations: int
    residual_inf: float
    period_s: float
    state_names: tuple[str, ...]
    state: tuple[float, ...]
    #: "stable" | "unstable" | "not_resolved" | "not_assessed"
    stability: str
    multipliers: tuple[FloquetMultiplier, ...] = ()
    monodromy: tuple[tuple[float, ...], ...] = ()
    assumptions: tuple[ClosedLoopAssumption, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def max_multiplier_magnitude(self) -> float | None:
        if not self.multipliers:
            return None
        return max(item.magnitude for item in self.multipliers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis": "closed_loop_periodic_steady_state",
            "method": "newton_shooting_on_augmented_state",
            "jacobian": "central_finite_difference_of_production_cycle_map",
            "claim": "local stability of the computed periodic orbit only",
            "converged": self.converged,
            "iterations": self.iterations,
            "residual_inf": self.residual_inf,
            "period_s": self.period_s,
            "state": dict(zip(self.state_names, self.state, strict=True)),
            "stability": self.stability,
            "max_multiplier_magnitude": self.max_multiplier_magnitude,
            "multipliers": [item.as_dict() for item in self.multipliers],
            "monodromy": [list(row) for row in self.monodromy],
            "assumptions": [item.as_dict() for item in self.assumptions],
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class _Plan:
    """What the cycle map needs, resolved and scope-checked once."""

    model: Model
    period: float
    circuit_ids: tuple[str, ...]
    register_ids: tuple[str, ...]
    #: unit_delay id -> the block whose tick-0 value it answers next tick.
    register_sources: dict[str, str]
    saturate_ids: tuple[str, ...]
    #: signal_pwm control id -> (duty block id, duty_min, duty_max)
    duty_sources: dict[str, tuple[str, float, float]]


def _plan(model: Model) -> _Plan:
    graph = model.signals
    sampled = dict(model.sampled_controls)
    other = sorted(
        control_id
        for control_id, control in sampled.items()
        if getattr(control, "kind", "sampled_pi_pwm") != "signal_pwm"
    )
    if other:
        raise ClosedLoopSteadyStateError(
            f"sampled control(s) {other!r} are not signal_pwm: their controller "
            "memory lives inside the kernel and cannot be set from outside, so "
            "the cycle map cannot start from a given controller state. Express "
            "the law with signal blocks (see examples/buck_closed_loop_signals.yaml)"
        )
    if graph is None or not sampled:
        raise ClosedLoopSteadyStateError(
            "closed-loop steady state needs a control law in the signal layer "
            "driving a signal_pwm control; this model has none (an open-loop "
            "model belongs to analysis.steady_state)"
        )
    for block in graph.blocks:
        if block.kind not in SUPPORTED_BLOCK_KINDS:
            reason = _UNSUPPORTED_REASON.get(block.kind, "outside the v1 block set")
            raise ClosedLoopSteadyStateError(
                f"signal block {block.id!r} of type {block.kind!r} is not supported "
                f"by the closed-loop steady state: {reason}"
            )

    circuit = Circuit(model)
    if circuit.waveform_sources:
        raise ClosedLoopSteadyStateError(
            "closed-loop steady state supports DC sources only; a waveform source "
            "makes the cycle map time-varying"
        )
    if model.machines:
        raise ClosedLoopSteadyStateError(
            "machines are outside the closed-loop steady-state scope"
        )
    if circuit.sat_inductors:
        raise ClosedLoopSteadyStateError(
            "saturable inductors are outside the closed-loop steady-state scope "
            "(the map is not differentiable at the saturation knee)"
        )
    if model.forced_gate_windows:
        raise ClosedLoopSteadyStateError(
            "forced gate windows are a one-off perturbation, not a periodic orbit"
        )
    if not circuit.state_components:
        raise ClosedLoopSteadyStateError(
            "closed-loop steady state requires at least one circuit state"
        )
    _diodes, mode = HybridSimulator(model).initial_mode()
    if tuple(getattr(mode, "star_constraints", ())):
        raise ClosedLoopSteadyStateError(
            "inductor-star invariants are outside the closed-loop steady-state "
            "v1 scope (the open-loop shooting handles them; this solver does not "
            "restrict its probes to the invariant subspace)"
        )

    periods = {round(1.0 / control.frequency, 18) for control in sampled.values()}
    periods |= {round(1.0 / control.frequency, 18) for control in model.controls.values()}
    if len(periods) != 1:
        raise ClosedLoopSteadyStateError(
            "closed-loop steady state requires one shared switching frequency; "
            f"found periods {sorted(periods)!r} s"
        )
    period = next(iter(sampled.values())).period
    task = graph.task
    if task.offset != 0.0 or abs(task.period - period) > 1e-12 * period:
        raise ClosedLoopSteadyStateError(
            "closed-loop steady state requires the signal task to tick once per "
            f"switching period at offset 0; task period {task.period!r} s, offset "
            f"{task.offset!r} s, switching period {period!r} s"
        )

    registers = tuple(block for block in graph.blocks if block.kind == "unit_delay")
    duty_sources: dict[str, tuple[str, float, float]] = {}
    for control_id, control in sorted(sampled.items()):
        target = declared_signal_target(control.duty_signal)
        if target is None:
            raise ClosedLoopSteadyStateError(
                f"signal_pwm {control_id!r} duty {control.duty_signal!r} does not "
                "name a signal block"
            )
        duty_sources[control_id] = (target, control.duty_min, control.duty_max)
    return _Plan(
        model=model,
        period=period,
        circuit_ids=tuple(component.id for component in circuit.state_components),
        register_ids=tuple(block.id for block in registers),
        register_sources={block.id: block.inputs[0] for block in registers},
        saturate_ids=tuple(
            block.id for block in graph.blocks if block.kind == "saturate"
        ),
        duty_sources=duty_sources,
    )


def _initial_vector(plan: _Plan) -> np.ndarray:
    circuit = Circuit(plan.model)
    blocks = plan.model.signals.by_id  # type: ignore[union-attr]
    registers = [float(blocks[block_id].params["initial"]) for block_id in plan.register_ids]
    return np.concatenate(
        [np.asarray(circuit.initial_state, dtype=float), np.asarray(registers, dtype=float)]
    )


def _cycle_model(plan: _Plan, z: np.ndarray) -> Model:
    model = plan.model
    count = len(plan.circuit_ids)
    by_id = dict(zip(plan.circuit_ids, z[:count], strict=True))
    components = tuple(
        replace(component, initial_state=float(by_id[component.id]), initial_state_explicit=True)
        if component.id in by_id
        else component
        for component in model.components
    )
    registers = dict(zip(plan.register_ids, z[count:], strict=True))
    graph = model.signals
    blocks = tuple(
        replace(block, params={**block.params, "initial": float(registers[block.id])})
        if block.id in registers
        else block
        for block in graph.blocks  # type: ignore[union-attr]
    )
    transient = replace(
        model.transient,
        start=0.0,
        stop=plan.period,
        # Only the end state and the tick-0 signal values are read; the grid
        # is a recording choice that does not touch the trajectory.
        output_step=plan.period / 4.0,
    )
    return replace(
        model,
        components=components,
        signals=replace(graph, blocks=blocks),  # type: ignore[arg-type]
        transient=transient,
        harmonics=None,
    )


@dataclass(frozen=True)
class _CycleRun:
    end: np.ndarray
    mode_sequence: tuple[str, ...]
    #: Smallest distance of any saturate input / raw duty from its limits.
    limiter_margin: float
    limiter_detail: str


def _run_cycle(plan: _Plan, z: np.ndarray) -> _CycleRun:
    result: SimulationResult = HybridSimulator(_cycle_model(plan, z)).simulate()
    final_circuit = result.mode_intervals[-1]["state_end"]
    first_ticks = [entry for entry in result.signal_trace if entry["tick_index"] == 0]
    if len(first_ticks) != 1:
        raise ClosedLoopSteadyStateError(
            f"expected exactly one signal tick at the cycle start, found {len(first_ticks)}"
        )
    values = first_ticks[0]["values"]
    end = np.concatenate(
        [
            np.asarray([final_circuit[name] for name in plan.circuit_ids], dtype=float),
            np.asarray(
                [values[plan.register_sources[name]] for name in plan.register_ids],
                dtype=float,
            ),
        ]
    )
    graph_blocks = plan.model.signals.by_id  # type: ignore[union-attr]
    margin = float("inf")
    detail = "no limiter in the loop"
    for block_id in plan.saturate_ids:
        block = graph_blocks[block_id]
        value = values[block.inputs[0]]
        low, high = float(block.params["min"]), float(block.params["max"])
        distance = min(value - low, high - value)
        if distance < margin:
            margin, detail = distance, f"saturate {block_id!r} input {value!r} in [{low!r}, {high!r}]"
    for control_id, (block_id, low, high) in plan.duty_sources.items():
        value = values[block_id]
        distance = min(value - low, high - value)
        if distance < margin:
            margin, detail = (
                distance,
                f"signal_pwm {control_id!r} duty {value!r} in [{low!r}, {high!r}]",
            )
    sequence = tuple(str(interval["mode_signature"]) for interval in result.mode_intervals)
    return _CycleRun(end=end, mode_sequence=sequence, limiter_margin=margin, limiter_detail=detail)


def _steps(z: np.ndarray, relative: float) -> np.ndarray:
    return relative * np.maximum(np.abs(z), _STEP_FLOOR)


def _central_monodromy(
    plan: _Plan, z: np.ndarray, relative: float
) -> tuple[np.ndarray, list[_CycleRun]]:
    steps = _steps(z, relative)
    columns = []
    runs: list[_CycleRun] = []
    for index, step in enumerate(steps):
        offset = np.zeros_like(z)
        offset[index] = step
        plus = _run_cycle(plan, z + offset)
        minus = _run_cycle(plan, z - offset)
        runs.extend((plus, minus))
        columns.append((plus.end - minus.end) / (2.0 * step))
    return np.column_stack(columns), runs


def _match(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Pair each reference eigenvalue with its nearest unused counterpart."""

    remaining = list(other)
    paired = []
    for value in reference:
        index = int(np.argmin([abs(value - candidate) for candidate in remaining]))
        paired.append(remaining.pop(index))
    return np.asarray(paired)


def solve_closed_loop_steady_state(
    model: Model,
    *,
    initial_state: dict[str, float] | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    relative_step: float = DEFAULT_RELATIVE_STEP,
) -> ClosedLoopSteadyState:
    """Closed-loop periodic orbit and its Floquet multipliers.

    Newton is local: it starts from the model's declared initial circuit
    state and ``unit_delay`` initial values, or from ``initial_state``
    (keys are circuit state ids and ``signal:<unit_delay id>``) — for
    example the last cycle of a settled transient.
    """

    plan = _plan(model)
    names = plan.circuit_ids + tuple(f"signal:{name}" for name in plan.register_ids)
    z = _initial_vector(plan)
    if initial_state is not None:
        unknown = sorted(set(initial_state) - set(names))
        if unknown:
            raise ClosedLoopSteadyStateError(
                f"initial_state names unknown state(s) {unknown!r}; expected a subset "
                f"of {list(names)!r}"
            )
        for position, name in enumerate(names):
            if name in initial_state:
                z[position] = float(initial_state[name])

    identity = np.eye(len(z))
    converged = False
    residual_inf = float("inf")
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        base = _run_cycle(plan, z)
        residual = base.end - z
        scale = 1.0 + np.maximum(np.abs(z), np.abs(base.end))
        residual_inf = float(np.max(np.abs(residual) / scale))
        if residual_inf <= tolerance:
            converged = True
            break
        if base.limiter_margin <= 0.0:
            # A clamped duty makes the map flat in the controller directions:
            # the Newton matrix is singular for a reason that has nothing to
            # do with the orbit. Say which limiter, not "a multiplier at 1".
            return ClosedLoopSteadyState(
                converged=False,
                iterations=iterations,
                residual_inf=residual_inf,
                period_s=plan.period,
                state_names=names,
                state=tuple(float(value) for value in z),
                stability="not_assessed",
                notes=(
                    "a Newton iterate drives a limiter into its bound ("
                    + base.limiter_detail
                    + "); start closer to the orbit with initial_state, e.g. the "
                    "last cycle of a settled transient",
                ),
            )
        steps = _steps(z, relative_step)
        jacobian = np.column_stack(
            [
                (_run_cycle(plan, z + step * identity[:, index]).end - base.end) / step
                for index, step in enumerate(steps)
            ]
        )
        try:
            update = np.linalg.solve(jacobian - identity, -residual)
        except np.linalg.LinAlgError as error:
            raise ClosedLoopSteadyStateError(
                "the shooting system is singular (a multiplier sits exactly at 1); "
                "the orbit is not isolated"
            ) from error
        # Backtracking: a full Newton step from far away can drive the duty
        # into its clamp, where the map is flat and the step is meaningless.
        damping = 1.0
        current = float(np.max(np.abs(residual) / scale))
        while damping > 1.0 / 64.0:
            trial = z + damping * update
            trial_run = _run_cycle(plan, trial)
            trial_scale = 1.0 + np.maximum(np.abs(trial), np.abs(trial_run.end))
            if float(np.max(np.abs(trial_run.end - trial) / trial_scale)) < current:
                break
            damping *= 0.5
        z = z + damping * update

    if not converged:
        return ClosedLoopSteadyState(
            converged=False,
            iterations=iterations,
            residual_inf=residual_inf,
            period_s=plan.period,
            state_names=names,
            state=tuple(float(value) for value in z),
            stability="not_assessed",
            notes=("shooting did not converge; no multipliers are reported",),
        )

    orbit = _run_cycle(plan, z)
    coarse, coarse_runs = _central_monodromy(plan, z, relative_step)
    fine, fine_runs = _central_monodromy(plan, z, 0.5 * relative_step)
    runs = [orbit, *coarse_runs, *fine_runs]

    worst_limiter = min(runs, key=lambda run: run.limiter_margin)
    limiter = ClosedLoopAssumption(
        name="no_limiter_engaged",
        held=worst_limiter.limiter_margin > 0.0,
        margin=worst_limiter.limiter_margin,
        detail="worst over the orbit and every probe run: "
        + worst_limiter.limiter_detail,
    )
    changed = sum(run.mode_sequence != orbit.mode_sequence for run in runs)
    sequence = ClosedLoopAssumption(
        name="mode_sequence_unchanged",
        held=changed == 0,
        margin=float(-changed),
        detail=(
            f"{changed} of {len(runs) - 1} probe runs switched through a different "
            f"mode sequence than the orbit ({len(orbit.mode_sequence)} intervals)"
        ),
    )
    assumptions = (limiter, sequence)
    if not all(item.held for item in assumptions):
        return ClosedLoopSteadyState(
            converged=True,
            iterations=iterations,
            residual_inf=residual_inf,
            period_s=plan.period,
            state_names=names,
            state=tuple(float(value) for value in z),
            stability="not_assessed",
            assumptions=assumptions,
            notes=(
                "the cycle map is not differentiable at this orbit; no multipliers "
                "are reported",
            ),
        )

    values = np.linalg.eigvals(coarse)
    finer = _match(values, np.linalg.eigvals(fine))
    uncertainty = np.abs(values - finer)
    order = np.argsort(-np.abs(values))
    frequency = 1.0 / plan.period
    multipliers = tuple(
        FloquetMultiplier(
            real=float(values[index].real),
            imag=float(values[index].imag),
            magnitude=float(abs(values[index])),
            frequency_hz=float(abs(np.angle(values[index])) * frequency / (2.0 * np.pi)),
            uncertainty=float(uncertainty[index]),
        )
        for index in order
    )
    if all(item.magnitude + item.uncertainty < 1.0 for item in multipliers):
        stability = "stable"
    elif any(item.magnitude - item.uncertainty > 1.0 for item in multipliers):
        stability = "unstable"
    else:
        stability = "not_resolved"
    return ClosedLoopSteadyState(
        converged=True,
        iterations=iterations,
        residual_inf=residual_inf,
        period_s=plan.period,
        state_names=names,
        state=tuple(float(value) for value in z),
        stability=stability,
        multipliers=multipliers,
        monodromy=tuple(tuple(float(value) for value in row) for row in coarse),
        assumptions=assumptions,
    )
