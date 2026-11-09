# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Periodic steady-state analysis via single-cycle shooting.

Finds the state vector ``x*`` that the production hybrid simulator maps onto
itself over exactly one switching period: ``F(x*) = x*`` where ``F`` simulates
the model for ``T = 1/f_pwm`` from the given initial state. Newton iteration
uses a forward-difference Jacobian of the cycle map, so every claim rests on
the same solver that produces run bundles — there is no separate averaged
model in this step. Convergence status is part of the result and is meant to
be retained as evidence; a non-converged result is a first-class outcome, not
an exception.

Star invariants. A node where only inductors meet — a series Lr+Lm
chain, a floating star — is index-reduced by the kernel into a conservation
law the flow preserves exactly, and the kernel validates every INITIAL state
against it. A finite-difference probe that perturbs one state component
independently therefore violates KCL at that node and the cycle map refuses
to run; the same conservation puts a structural unit eigenvalue in the cycle
map, so the unreduced Newton system is singular by construction. The solve
runs on the invariant subspace instead: probes and Newton steps move only
inside the kernel of the constraint matrix, the invariant values stay pinned
to the model's DECLARED initial state (never projected, never invented), and
the pinning is recorded in the result. Models without such nodes take the
exact code path this module always had.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from pear_sim.core.circuit import Circuit
from pear_sim.core.solver import HybridSimulator, SimulationResult
from pear_sim.errors import ModelError
from pear_sim.spec.model import Model

__all__ = [
    "PeriodicSteadyState",
    "SteadyStateError",
    "one_cycle_model",
    "solve_periodic_steady_state",
]

DEFAULT_MAX_ITERATIONS = 25
DEFAULT_TOLERANCE = 1e-9
DEFAULT_OUTPUT_POINTS = 501


class SteadyStateError(ModelError):
    """The model cannot be driven through the shooting analysis."""

    code = "analysis.steady_state_invalid"


@dataclass(frozen=True)
class PeriodicSteadyState:
    """Shooting result over one switching period."""

    converged: bool
    iterations: int
    residual_inf: float
    tolerance: float
    period_s: float
    state_names: tuple[str, ...]
    state: tuple[float, ...]
    cycle_average_state: tuple[float, ...]
    natural_events_per_cycle: int
    distinct_topologies: int
    mode_intervals: tuple[dict[str, Any], ...]
    #: Star invariants the solve was confined to, each with the level value
    #: pinned from the model's declared initial state. Empty for models
    #: without inductor-star nodes.
    invariants: tuple[dict[str, Any], ...] = ()

    def state_by_name(self) -> dict[str, float]:
        return dict(zip(self.state_names, self.state))

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": "pear-sim/steady-state/v0.1",
            "kind": "periodic-steady-state-shooting",
            "converged": self.converged,
            "iterations": self.iterations,
            "residual_inf": self.residual_inf,
            "tolerance": self.tolerance,
            "period_s": self.period_s,
            "state_names": list(self.state_names),
            "state": list(self.state),
            "cycle_average_state": list(self.cycle_average_state),
            "natural_events_per_cycle": self.natural_events_per_cycle,
            "distinct_topologies": self.distinct_topologies,
        }
        if self.invariants:
            # Additive, and only when present: artifacts of models without
            # star nodes stay byte-identical to the invariant-free format.
            payload["invariants"] = [dict(entry) for entry in self.invariants]
        return payload


def _shooting_period(model: Model) -> float:
    if not model.controls:
        raise SteadyStateError(
            "periodic steady state requires at least one PWM control"
        )
    frequencies = sorted({control.frequency for control in model.controls.values()})
    if len(frequencies) != 1:
        raise SteadyStateError(
            "periodic steady state requires a single shared switching frequency; "
            f"found {frequencies!r}"
        )
    return 1.0 / frequencies[0]


def one_cycle_model(
    model: Model,
    state: np.ndarray,
    *,
    output_points: int = DEFAULT_OUTPUT_POINTS,
) -> Model:
    """Rebuild ``model`` to simulate exactly one period from ``state``."""

    period = _shooting_period(model)
    circuit = Circuit(model)
    if circuit.waveform_sources:
        raise SteadyStateError(
            "periodic steady state supports DC sources only; the one-cycle map "
            "restarts at t=0 and is not autonomous under waveform sources"
        )
    if not circuit.state_components:
        raise SteadyStateError("periodic steady state requires at least one state")
    if len(state) != len(circuit.state_components):
        raise SteadyStateError(
            f"state vector length {len(state)} does not match the "
            f"{len(circuit.state_components)} state components"
        )
    by_id = {
        component.id: float(value)
        for component, value in zip(circuit.state_components, state)
    }
    components = tuple(
        replace(component, initial_state=by_id[component.id], initial_state_explicit=True)
        if component.id in by_id
        else component
        for component in model.components
    )
    transient = replace(
        model.transient,
        start=0.0,
        stop=period,
        output_step=period / (output_points - 1),
    )
    # Record every state on the output grid. Outputs only select what is
    # sampled from the dense solution — the trajectory is unchanged — but a
    # state that is not recorded can only be reconstructed linearly between mode
    # boundaries, which biases the cycle average of a capacitor whose current
    # ramps (its voltage is parabolic inside an interval).
    state_outputs = tuple(
        f"{'i' if component.type == 'inductor' else 'v'}({component.id})"
        for component in circuit.state_components
        if component.type in {"inductor", "capacitor"}
    )
    outputs = model.outputs + tuple(name for name in state_outputs if name not in model.outputs)
    return replace(
        model, components=components, transient=transient, harmonics=None, outputs=outputs
    )


def _cycle_map(model: Model, state: np.ndarray) -> tuple[np.ndarray, SimulationResult]:
    cycle = one_cycle_model(model, state)
    result = HybridSimulator(cycle).simulate()
    circuit = Circuit(cycle)
    final = np.array(
        [
            result.mode_intervals[-1]["state_end"][component.id]
            for component in circuit.state_components
        ],
        dtype=float,
    )
    return final, result


def _star_invariants(
    model: Model, circuit: Circuit
) -> tuple[np.ndarray, tuple[dict[str, Any], ...]]:
    """Constraint matrix of the initial mode's inductor-star invariants.

    Rows come from the SAME compiled mode the kernel validates the declared
    initial state against — so any state satisfying them is exactly a state
    the cycle map accepts. Normalised rows; empty matrix when the model has
    no star nodes.
    """

    _diodes, mode = HybridSimulator(model).initial_mode()
    names = tuple(component.id for component in circuit.state_components)
    rows: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []
    x0 = np.asarray(circuit.initial_state, dtype=float)
    for node_name, terms in getattr(mode, "star_constraints", ()):
        row = np.zeros(len(names))
        for state_index, sign in terms:
            row[state_index] = sign
        row /= float(np.linalg.norm(row))
        rows.append(row)
        entries.append(
            {
                "node": node_name,
                "terms": {names[index]: float(sign) for index, sign in terms},
                "pinned_value_a": float(
                    sum(sign * x0[index] for index, sign in terms)
                ),
                "source": "declared initial state",
            }
        )
    matrix = np.asarray(rows) if rows else np.zeros((0, len(names)))
    return matrix, tuple(entries)


def _nullspace_basis(constraints: np.ndarray, dimension: int) -> np.ndarray:
    """Orthonormal basis of the constraint kernel (columns)."""

    if len(constraints) == 0:
        return np.eye(dimension)
    _u, singular, vt = np.linalg.svd(constraints, full_matrices=True)
    rank = int(np.sum(singular > 1e-12))
    return vt[rank:].T


def solve_periodic_steady_state(
    model: Model,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> PeriodicSteadyState:
    """Newton shooting on the one-period cycle map of the production solver."""

    period = _shooting_period(model)
    circuit = Circuit(model)
    names = tuple(component.id for component in circuit.state_components)
    x = np.asarray(circuit.initial_state, dtype=float).copy()
    scale = 1.0 + np.abs(x)

    constraints, invariant_entries = _star_invariants(model, circuit)
    basis = _nullspace_basis(constraints, len(x))
    if basis.shape[1] == 0:
        raise SteadyStateError(
            "every state direction is pinned by a star invariant; there is "
            "nothing left to solve for"
        )

    converged = False
    iterations = 0
    residual_inf = float("inf")
    last_result: SimulationResult | None = None
    for iterations in range(1, max_iterations + 1):
        fx, last_result = _cycle_map(model, x)
        residual = fx - x
        scale = 1.0 + np.maximum(np.abs(x), np.abs(fx))
        residual_inf = float(np.max(np.abs(residual) / scale))
        if residual_inf < tolerance:
            converged = True
            break
        if len(constraints):
            # The cycle map must conserve the invariant it was probed on; a
            # residual component ALONG the constraint cannot be corrected by
            # in-subspace steps and would mean the level set was wrong.
            drift = float(np.max(np.abs(constraints @ residual)))
            if drift > 1e3 * tolerance * float(np.max(scale)):
                node = invariant_entries[
                    int(np.argmax(np.abs(constraints @ residual)))
                ]["node"]
                raise SteadyStateError(
                    f"the one-cycle map does not conserve the star invariant "
                    f"at node {node!r} (drift {drift:.3e} A per cycle); the "
                    "fixed point is not confined to the declared level set"
                )
        if len(constraints) == 0:
            # Unconstrained path, byte-for-byte the invariant-free iteration.
            jacobian = np.zeros((len(x), len(x)), dtype=float)
            for column in range(len(x)):
                step = 1e-6 * scale[column]
                probe = x.copy()
                probe[column] += step
                f_probe, _ = _cycle_map(model, probe)
                jacobian[:, column] = (f_probe - fx) / step
            system = jacobian - np.eye(len(x))
            rhs = -residual
        else:
            # Probe only along the invariant kernel: every probe satisfies
            # the star KCL the kernel validates, and the reduced system has
            # no structural unit eigenvalue left to be singular about.
            reduced = np.zeros((basis.shape[1], basis.shape[1]), dtype=float)
            step_size = 1e-6 * float(np.max(scale))
            for column in range(basis.shape[1]):
                probe = x + step_size * basis[:, column]
                f_probe, _ = _cycle_map(model, probe)
                reduced[:, column] = basis.T @ ((f_probe - fx) / step_size)
            system = reduced - np.eye(basis.shape[1])
            rhs = -(basis.T @ residual)
        try:
            delta = np.linalg.solve(system, rhs)
        except np.linalg.LinAlgError as exc:
            raise SteadyStateError(
                "shooting Jacobian is singular; the cycle map has no isolated "
                "fixed point for this model"
            ) from exc
        x = x + (delta if len(constraints) == 0 else basis @ delta)

    assert last_result is not None
    # Cycle average from the retained output grid of the last iterate.
    time = last_result.time
    averages = []
    state_traces = _state_traces(model, last_result)
    for name in names:
        trace = state_traces[name]
        averages.append(float(np.trapezoid(trace, time) / (time[-1] - time[0])))
    return PeriodicSteadyState(
        converged=converged,
        iterations=iterations,
        residual_inf=residual_inf,
        tolerance=tolerance,
        period_s=period,
        state_names=names,
        state=tuple(float(value) for value in x),
        cycle_average_state=tuple(averages),
        natural_events_per_cycle=int(last_result.stats["natural_events"]),
        distinct_topologies=len(
            {interval["mode_signature"] for interval in last_result.mode_intervals}
        ),
        mode_intervals=tuple(last_result.mode_intervals),
        invariants=invariant_entries,
    )


def _state_traces(model: Model, result: SimulationResult) -> dict[str, np.ndarray]:
    """State trajectories on the output grid, from signals or reconstruction."""

    circuit = Circuit(model)
    traces: dict[str, np.ndarray] = {}
    for component in circuit.state_components:
        if component.type == "inductor" and f"i({component.id})" in result.signals:
            traces[component.id] = result.signals[f"i({component.id})"]
            continue
        if component.type == "capacitor" and f"v({component.id})" in result.signals:
            traces[component.id] = result.signals[f"v({component.id})"]
            continue
        traces[component.id] = _piecewise_state_trace(component.id, result)
    return traces


def _piecewise_state_trace(component_id: str, result: SimulationResult) -> np.ndarray:
    """Linear per-interval reconstruction from retained mode boundary states."""

    time = result.time
    values = np.empty_like(time)
    for interval in result.mode_intervals:
        start = float(interval["start_s"])
        end = float(interval["end_s"])
        v0 = float(interval["state_start"][component_id])
        v1 = float(interval["state_end"][component_id])
        mask = (time >= start) & (time <= end)
        if end > start:
            values[mask] = v0 + (v1 - v0) * (time[mask] - start) / (end - start)
        else:
            values[mask] = v0
    return values
