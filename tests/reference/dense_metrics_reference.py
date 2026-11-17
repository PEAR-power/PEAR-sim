# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Per-node / per-step metric loops as they were before 2026-11-17, kept verbatim.

pear_sim.core.metrics now evaluates the quadrature as column gathers and the
state extrema as one batch per segment. These are the former Python loops,
copied unchanged, so tests/test_vectorized_metrics.py can demand that the new
code reproduces them BIT FOR BIT. Do not "improve" this file: its only value
is that it is the old arithmetic.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from pear_sim.core.circuit import CompiledMode
from pear_sim.core.metrics import (
    _IntervalIntegral,
    _QUADRATURE_NODES,
    _QUADRATURE_WEIGHTS,
    _STATE_EXTREMA_PROBE_SUBINTERVALS,
    _STATE_EXTREMA_ROOT_BOUNDARY_TOLERANCE,
    _STATE_EXTREMA_ROOT_IMAGINARY_TOLERANCE,
    _dense_step_state_polynomials,
    _mode_output,
    _poly_derivative,
    _poly_roots,
    _poly_value,
    electrical_outputs,
)
from pear_sim.errors import SimulationError


def reference_integrate_dense_interval(
    solution: Callable[[np.ndarray], np.ndarray],
    mode: CompiledMode,
    start: float,
    end: float,
    *,
    component_energy_requested: bool = True,
) -> _IntervalIntegral:
    """Integrate outputs and absorbed component power over one dense-output interval.

    Gauss-Legendre nodes are strictly inside ``(start, end)``. Consequently, an
    algebraic jump at either topology boundary has zero weight, as required by the
    continuous-time integral.

    ``component_energy_requested=False`` (lean evidence mode) skips the
    per-component power quadrature — the dominant cost here, since it visits
    every component at every node. The signal integrals are computed by the
    identical statements either way, so a lean run's means and RMS values are
    bit-for-bit identical to a full run's. The skipped energies are not returned
    as zeros: ``component_absorbed_energy`` is ``None``, and every consumer of it
    fails closed rather than reading a fabricated balance.
    """

    half_width = 0.5 * (end - start)
    midpoint = 0.5 * (start + end)
    times = midpoint + half_width * _QUADRATURE_NODES
    states = np.asarray(solution(times), dtype=float)
    if states.ndim == 1:
        states = states[:, np.newaxis]
    weights = half_width * _QUADRATURE_WEIGHTS
    outputs = electrical_outputs(mode.circuit.model)
    signal_integrals = {name: 0.0 for name in outputs}
    signal_square_integrals = {name: 0.0 for name in outputs}
    component_energy = (
        {component.id: 0.0 for component in mode.circuit.components}
        if component_energy_requested
        else None
    )

    for index, weight in enumerate(weights):
        state = states[:, index]
        sample_time = float(times[index])
        algebraic = mode.algebraic(state, sample_time)
        for name in outputs:
            value = _mode_output(mode, name, state, algebraic, sample_time)
            signal_integrals[name] += float(weight * value)
            signal_square_integrals[name] += float(weight * value * value)
        if component_energy is None:
            continue
        for component in mode.circuit.components:
            voltage = mode.component_voltage(component, state, algebraic, sample_time)
            current = (
                0.0
                if component.type == "voltage_probe"
                else mode.component_current(component, state, algebraic, sample_time)
            )
            component_energy[component.id] += float(weight * voltage * current)

    return _IntervalIntegral(
        duration=end - start,
        signal_integrals=signal_integrals,
        signal_square_integrals=signal_square_integrals,
        component_absorbed_energy=component_energy,
    )


def reference_dense_state_extrema(
    solution: Any,
    mode: CompiledMode,
    window_start: float,
    window_end: float,
) -> dict[str, dict[str, float]]:
    """Evaluate every real dense-polynomial derivative root per accepted step."""

    if solution.sol is None or window_end <= window_start:
        return {}
    interpolants = getattr(solution.sol, "interpolants", None)
    if interpolants is None or len(interpolants) != len(solution.t) - 1:
        raise SimulationError(
            "solve_ivp dense output does not expose one interpolant per accepted step; "
            "refusing to publish incomplete state extrema"
        )

    extrema: dict[str, dict[str, float]] = {}
    state_components = tuple(
        sorted(mode.circuit.state_components, key=lambda component: component.id)
    )

    def consider(component_id: str, value: float, time: float) -> None:
        record = extrema.get(component_id)
        if record is None:
            extrema[component_id] = {
                "minimum": value,
                "minimum_time_s": time,
                "maximum": value,
                "maximum_time_s": time,
            }
            return
        if value < record["minimum"]:
            record["minimum"] = value
            record["minimum_time_s"] = time
        if value > record["maximum"]:
            record["maximum"] = value
            record["maximum_time_s"] = time

    for step_index, (raw_start, raw_end) in enumerate(
        zip(solution.t[:-1], solution.t[1:], strict=True)
    ):
        step_start = float(raw_start)
        step_end = float(raw_end)
        start = max(step_start, window_start)
        end = min(step_end, window_end)
        if end <= start:
            continue
        interpolant = interpolants[step_index]
        dense_start = float(interpolant.t_old)
        dense_step = float(interpolant.t - interpolant.t_old)
        if dense_step <= 0.0 or not math.isclose(
            dense_start,
            step_start,
            rel_tol=0.0,
            abs_tol=max(1e-15, abs(step_start) * 1e-14),
        ):
            raise SimulationError(
                "solve_ivp dense interpolant is not aligned with its accepted step; "
                "refusing to publish incomplete state extrema"
            )
        # A terminal root can truncate solution.t inside the integrator's accepted
        # step. The interpolant still uses its original, longer h, so normalize with
        # interpolant.t rather than the event-truncated raw_end.
        normalized_start = (start - dense_start) / dense_step
        normalized_end = (end - dense_start) / dense_step
        probe_points = np.linspace(
            normalized_start,
            normalized_end,
            _STATE_EXTREMA_PROBE_SUBINTERVALS + 1,
        )
        polynomials = _dense_step_state_polynomials(
            interpolant,
            len(mode.circuit.state_components),
        )

        for component in state_components:
            state_index = mode.circuit.state_index[component.id]
            polynomial = polynomials[state_index]
            candidate_points = [float(point) for point in probe_points]
            for raw_root in _poly_roots(_poly_derivative(polynomial)):
                root = complex(raw_root)
                if not (math.isfinite(root.real) and math.isfinite(root.imag)):
                    continue
                if abs(root.imag) > _STATE_EXTREMA_ROOT_IMAGINARY_TOLERANCE * max(
                    1.0,
                    abs(root.real),
                ):
                    continue
                if (
                    root.real < normalized_start - _STATE_EXTREMA_ROOT_BOUNDARY_TOLERANCE
                    or root.real > normalized_end + _STATE_EXTREMA_ROOT_BOUNDARY_TOLERANCE
                ):
                    continue
                candidate_points.append(
                    min(normalized_end, max(normalized_start, float(root.real)))
                )

            unique_points = np.asarray(sorted(set(candidate_points)), dtype=float)
            values = np.asarray(_poly_value(polynomial, unique_points), dtype=float)
            times = dense_start + dense_step * unique_points
            for time, value in zip(times, values, strict=True):
                consider(component.id, float(value), float(time))
    return extrema
