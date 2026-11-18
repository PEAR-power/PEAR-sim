# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Dense-output metric quadrature, energy bookkeeping, and state extrema.

Extracted verbatim from ``pear_sim.core.solver`` (restructuring stage 2, move 8).
The simulator feeds accepted ``solve_ivp`` dense-output steps into
``_MetricAccumulator``; nothing here integrates the circuit itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable
import weakref

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial import polynomial as _poly
from numpy.polynomial import polyutils

from pear_sim.spec.signals import is_declared_signal, is_machine_signal
from pear_sim.core.circuit import SWITCHING_BRANCH_TYPES, Circuit, CompiledMode
from pear_sim.errors import ModelError, SimulationError
from pear_sim.spec.model import Model


_QUADRATURE_ORDER = 8
_QUADRATURE_NODES, _QUADRATURE_WEIGHTS = np.polynomial.legendre.leggauss(_QUADRATURE_ORDER)
_TIME_METRIC_METHOD = "gauss_legendre_8_on_solve_ivp_dense_output"
_STATE_EXTREMA_PROBE_SUBINTERVALS = 16
_STATE_EXTREMA_ROOT_IMAGINARY_TOLERANCE = 1e-7
_STATE_EXTREMA_ROOT_BOUNDARY_TOLERANCE = 1e-9
_STATE_EXTREMA_METHOD = (
    "solve_ivp_dense_polynomial_derivative_root_search_all_real_roots_"
    "per_accepted_step_plus_17_probe_points"
)
# A lean run never labels an observation with a method it did not execute.
_STATE_EXTREMA_NOT_COMPUTED = "not_computed_in_lean_evidence_mode"

#: Evidence scope of a run. ``full`` substantiates every claim the bundle can
#: make. ``lean`` drops two claims whose cost is dominant and whose consumers a
#: parameter search does not have — the per-component energy balance and the
#: dense state extrema of intervals outside the reported window. Everything a
#: lean run still reports is computed by the identical code path as in a full
#: run, so it is bit-for-bit identical; the difference is what is claimed, never
#: how accurately it was computed.
EVIDENCE_MODES = ("full", "lean")


@dataclass(frozen=True)
class _IntervalIntegral:
    duration: float
    signal_integrals: dict[str, float]
    signal_square_integrals: dict[str, float]
    #: ``None`` when the lean evidence mode skipped the per-component power
    #: quadrature. Never zeros — an absent balance must stay absent.
    component_absorbed_energy: dict[str, float] | None


@dataclass
class _WindowAccumulator:
    start: float
    end: float
    signal_integrals: dict[str, float]
    signal_square_integrals: dict[str, float]
    component_absorbed_energy: dict[str, float] | None
    integrated_interval_s: float = 0.0
    quadrature_intervals: int = 0

    @classmethod
    def create(
        cls,
        model: Model,
        start: float,
        end: float,
        *,
        component_energy_requested: bool = True,
    ) -> "_WindowAccumulator":
        return cls(
            start=start,
            end=end,
            signal_integrals={name: 0.0 for name in electrical_outputs(model)},
            signal_square_integrals={name: 0.0 for name in electrical_outputs(model)},
            component_absorbed_energy=(
                {component.id: 0.0 for component in model.components}
                if component_energy_requested
                else None
            ),
        )

    def add(self, contribution: _IntervalIntegral) -> None:
        self.integrated_interval_s += contribution.duration
        self.quadrature_intervals += 1
        for name, value in contribution.signal_integrals.items():
            self.signal_integrals[name] += value
        for name, value in contribution.signal_square_integrals.items():
            self.signal_square_integrals[name] += value
        if (self.component_absorbed_energy is None) != (
            contribution.component_absorbed_energy is None
        ):
            raise SimulationError(
                "Metric accumulation mixed component-energy contributions with "
                "contributions that skipped them; a partial energy balance must "
                "never be published"
            )
        if self.component_absorbed_energy is None:
            return
        assert contribution.component_absorbed_energy is not None
        for component_id, value in contribution.component_absorbed_energy.items():
            self.component_absorbed_energy[component_id] += value


def _mode_output(
    mode: CompiledMode,
    expression: str,
    state: np.ndarray,
    algebraic: np.ndarray,
    time: float | None = None,
) -> float:
    kind, target = mode.circuit.output_targets[expression]
    if kind == "node_voltage":
        return mode.node_voltage(target, state, algebraic, time)
    component = mode.circuit.component_by_id[target]
    if kind == "component_voltage":
        return mode.component_voltage(component, state, algebraic, time)
    return mode.component_current(component, state, algebraic, time)


def _integrate_dense_interval(
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
    plan = _quadrature_plan(mode)
    # One solve per node, by the identical call as before; everything after it
    # is a gather over (node, column) arrays instead of a Python visit of every
    # output and component at every node (2026-11-17: 30 million calls and a
    # third of a 60 ms grid-following run). Each element is produced by the
    # same float operations in the same order, so results are bit-identical.
    algebraic = np.stack(
        [mode.algebraic(states[:, index], float(times[index])) for index in range(len(times))]
    )
    extended = np.concatenate((np.zeros((len(times), 1)), algebraic), axis=1)
    weighted = weights[:, np.newaxis]

    values = plan.output_values(extended, algebraic, states, times)
    signal_sum = np.zeros(values.shape[1])
    square_sum = np.zeros(values.shape[1])
    weighted_values = weighted * values
    weighted_squares = weighted_values * values
    for index in range(len(times)):
        signal_sum = signal_sum + weighted_values[index]
        square_sum = square_sum + weighted_squares[index]
    signal_integrals = {
        name: float(signal_sum[column]) for column, name in enumerate(plan.outputs)
    }
    signal_square_integrals = {
        name: float(square_sum[column]) for column, name in enumerate(plan.outputs)
    }

    component_energy: dict[str, float] | None = None
    if component_energy_requested:
        voltage, current = plan.component_voltage_current(extended, algebraic, states, times)
        power = (weighted * voltage) * current
        energy_sum = np.zeros(power.shape[1])
        for index in range(len(times)):
            energy_sum = energy_sum + power[index]
        component_energy = {
            component_id: float(energy_sum[column])
            for column, component_id in enumerate(plan.component_ids)
        }

    return _IntervalIntegral(
        duration=end - start,
        signal_integrals=signal_integrals,
        signal_square_integrals=signal_square_integrals,
        component_absorbed_energy=component_energy,
    )


class _CurrentGather:
    """Column gather reproducing ``CompiledMode.component_current`` element-wise."""

    def __init__(self, mode: CompiledMode, components: list[Any], *, probe_is_zero: bool):
        circuit = mode.circuit
        self.count = len(components)
        resistor, resistor_p, resistor_n, resistor_value = [], [], [], []
        inductor, inductor_state = [], []
        constant, constant_value = [], []
        branch, branch_row = [], []
        self.waveform: list[tuple[int, Any]] = []
        self.unsupported: list[str] = []
        for column, component in enumerate(components):
            if probe_is_zero and component.type == "voltage_probe":
                continue
            if component.type == "resistor":
                resistor.append(column)
                resistor_p.append(_extended_index(circuit, component.p))
                resistor_n.append(_extended_index(circuit, component.n))
                resistor_value.append(float(component.value))
            elif component.type in {"inductor", "sat_inductor"}:
                inductor.append(column)
                inductor_state.append(circuit.state_index[component.id])
            elif component.type == "current_source":
                if component.waveform is not None:
                    self.waveform.append((column, component.waveform))
                else:
                    constant.append(column)
                    constant_value.append(float(component.value))
            elif component.id in mode.branch_index:
                branch.append(column)
                branch_row.append(mode.branch_index[component.id])
            elif component.type in SWITCHING_BRANCH_TYPES:
                continue
            else:
                self.unsupported.append(component.type)
        self.resistor = np.asarray(resistor, dtype=int)
        self.resistor_p = np.asarray(resistor_p, dtype=int)
        self.resistor_n = np.asarray(resistor_n, dtype=int)
        self.resistor_value = np.asarray(resistor_value, dtype=float)
        self.inductor = np.asarray(inductor, dtype=int)
        self.inductor_state = np.asarray(inductor_state, dtype=int)
        self.constant = np.asarray(constant, dtype=int)
        self.constant_value = np.asarray(constant_value, dtype=float)
        self.branch = np.asarray(branch, dtype=int)
        self.branch_row = np.asarray(branch_row, dtype=int)

    def __call__(
        self,
        extended: np.ndarray,
        algebraic: np.ndarray,
        states: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        if self.unsupported:
            raise ModelError(
                f"Current output is not implemented for {self.unsupported[0]}"
            )
        current = np.zeros((len(times), self.count))
        if self.resistor.size:
            current[:, self.resistor] = (
                extended[:, self.resistor_p] - extended[:, self.resistor_n]
            ) / self.resistor_value
        if self.inductor.size:
            current[:, self.inductor] = states[self.inductor_state, :].T
        if self.constant.size:
            current[:, self.constant] = self.constant_value
        if self.branch.size:
            current[:, self.branch] = algebraic[:, self.branch_row]
        for column, waveform in self.waveform:
            current[:, column] = [float(waveform.value(float(time))) for time in times]
        return current


def _extended_index(circuit: Circuit, node: str) -> int:
    """Column of ``node`` in [0 (ground), algebraic...]."""

    if node == circuit.ground:
        return 0
    return circuit.node_index[node] + 1


class _QuadraturePlan:
    """Per-mode gather indices for the quadrature; built once per topology."""

    def __init__(self, mode: CompiledMode):
        circuit = mode.circuit
        self.outputs = electrical_outputs(circuit.model)
        output_columns: dict[str, list[int]] = {
            "node": [], "voltage": [], "current": []
        }
        node_index, voltage_p, voltage_n = [], [], []
        current_components = []
        for column, name in enumerate(self.outputs):
            kind, target = circuit.output_targets[name]
            if kind == "node_voltage":
                output_columns["node"].append(column)
                node_index.append(_extended_index(circuit, target))
                continue
            component = circuit.component_by_id[target]
            if kind == "component_voltage":
                output_columns["voltage"].append(column)
                voltage_p.append(_extended_index(circuit, component.p))
                voltage_n.append(_extended_index(circuit, component.n))
            else:
                output_columns["current"].append(column)
                current_components.append(component)
        self._node_columns = np.asarray(output_columns["node"], dtype=int)
        self._node_index = np.asarray(node_index, dtype=int)
        self._voltage_columns = np.asarray(output_columns["voltage"], dtype=int)
        self._voltage_p = np.asarray(voltage_p, dtype=int)
        self._voltage_n = np.asarray(voltage_n, dtype=int)
        self._current_columns = np.asarray(output_columns["current"], dtype=int)
        self._output_current = _CurrentGather(mode, current_components, probe_is_zero=False)

        components = list(circuit.components)
        self.component_ids = tuple(component.id for component in components)
        self._component_p = np.asarray(
            [_extended_index(circuit, component.p) for component in components], dtype=int
        )
        self._component_n = np.asarray(
            [_extended_index(circuit, component.n) for component in components], dtype=int
        )
        self._component_current = _CurrentGather(mode, components, probe_is_zero=True)

    def output_values(
        self,
        extended: np.ndarray,
        algebraic: np.ndarray,
        states: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        values = np.zeros((len(times), len(self.outputs)))
        if self._node_columns.size:
            values[:, self._node_columns] = extended[:, self._node_index]
        if self._voltage_columns.size:
            values[:, self._voltage_columns] = (
                extended[:, self._voltage_p] - extended[:, self._voltage_n]
            )
        if self._current_columns.size:
            values[:, self._current_columns] = self._output_current(
                extended, algebraic, states, times
            )
        return values

    def component_voltage_current(
        self,
        extended: np.ndarray,
        algebraic: np.ndarray,
        states: np.ndarray,
        times: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        voltage = extended[:, self._component_p] - extended[:, self._component_n]
        return voltage, self._component_current(extended, algebraic, states, times)


_QUADRATURE_PLANS: "weakref.WeakKeyDictionary[CompiledMode, _QuadraturePlan]" = (
    weakref.WeakKeyDictionary()
)


def _quadrature_plan(mode: CompiledMode) -> _QuadraturePlan:
    plan = _QUADRATURE_PLANS.get(mode)
    if plan is None:
        plan = _QuadraturePlan(mode)
        _QUADRATURE_PLANS[mode] = plan
    return plan


def electrical_outputs(model: Any) -> tuple[str, ...]:
    """Outputs the continuous-time quadrature can integrate.

    Machine signals are read from the mechanical state and declared signal
    blocks are discrete values held between task instants; the segment
    quadrature has an affine map for neither. They are excluded HERE, in one
    place, rather than guarded at each of the four sites that iterate outputs
    — a signal that cannot be integrated must be absent from the integral,
    not integrated to a wrong number.
    """

    return tuple(
        name
        for name in model.outputs
        if not is_machine_signal(name) and not is_declared_signal(name)
    )


def _output_affine_map(
    mode: CompiledMode, expression: str
) -> tuple[np.ndarray, float, np.ndarray]:
    """Exact (state coefficients, constant, waveform-source coefficients)."""

    return mode.output_affine(expression)


def _storage_energy(component: Any, current_or_voltage: float) -> float:
    """Stored energy of one state component; piecewise for sat_inductor.

    For the saturable inductor E = integral of i dpsi with the piecewise-linear
    flux psi(i): below i_sat the usual 0.5*L*i^2; beyond it the incremental
    segment uses l_sat while the flux stays continuous.
    """

    value = float(component.value)
    magnitude = abs(float(current_or_voltage))
    if component.type == "sat_inductor":
        boundary = float(component.sat_current)
        if magnitude > boundary:
            # E = int_0^i x * L_inc(x) dx with L_inc = value below i_sat and
            # sat_inductance beyond it (energy, not co-energy of the flux map).
            saturated_inductance = float(component.sat_inductance)
            return float(
                0.5 * value * boundary * boundary
                + 0.5
                * saturated_inductance
                * (magnitude * magnitude - boundary * boundary)
            )
        return float(0.5 * value * magnitude * magnitude)
    return float(0.5 * value * magnitude * magnitude)


def _storage_energy_by_component(circuit: Circuit, state: np.ndarray) -> dict[str, float]:
    energies = {
        component.id: _storage_energy(
            component, float(state[circuit.state_index[component.id]])
        )
        for component in circuit.state_components
    }
    for coupling in getattr(circuit, "couplings", ()):
        cross = (
            float(coupling.mutual)
            * float(state[circuit.state_index[coupling.first]])
            * float(state[circuit.state_index[coupling.second]])
        )
        # E = 0.5 i^T L i; the mutual term is attributed half to each winding.
        energies[coupling.first] += 0.5 * cross
        energies[coupling.second] += 0.5 * cross
    return energies


def _state_values_by_component(circuit: Circuit, state: np.ndarray) -> dict[str, float]:
    return {
        component.id: float(state[circuit.state_index[component.id]])
        for component in sorted(circuit.state_components, key=lambda item: item.id)
    }


# The dense-state polynomials are handled as raw coefficient arrays with the
# same numpy.polynomial.polynomial routines the Polynomial class delegates to
# (polyadd, polymul, polyder, polyroots, polyval). The class adds only object
# overhead plus an identity domain map, off + scl * x with off = 0, scl = 1;
# that map is applied literally below so every result is bit-for-bit the one
# the class produced. On a three-phase SPWM run the class overhead was a
# quarter of the whole run. tests/test_dense_polynomials.py pins the equality.
_DOMAIN_OFFSET, _DOMAIN_SCALE = Polynomial([0.0]).mapparms()
_WINDOW_OFFSET, _WINDOW_SCALE = polyutils.mapparms(Polynomial.window, Polynomial.domain)


def _poly_value(coefficients: np.ndarray, points: np.ndarray) -> np.ndarray:
    return _poly.polyval(_DOMAIN_OFFSET + _DOMAIN_SCALE * points, coefficients)


def _poly_derivative(coefficients: np.ndarray) -> np.ndarray:
    return _poly.polyder(coefficients, 1, _DOMAIN_SCALE)


def _poly_roots(coefficients: np.ndarray) -> np.ndarray:
    return _WINDOW_OFFSET + _WINDOW_SCALE * _poly.polyroots(coefficients)


def _dense_step_state_polynomials(
    interpolant: Any,
    state_count: int,
) -> tuple[np.ndarray, ...]:
    """Reconstruct a solve_ivp step's dense state polynomials in x=(t-t0)/h.

    Each entry is a coefficient array, lowest power first.
    """

    kind = type(interpolant).__name__
    if kind in {"RkDenseOutput", "RadauDenseOutput"}:
        factor = float(interpolant.h) if kind == "RkDenseOutput" else 1.0
        return tuple(
            np.array(
                np.concatenate(
                    (
                        np.asarray([interpolant.y_old[index]], dtype=float),
                        factor * np.asarray(interpolant.Q[index, :], dtype=float),
                    )
                ),
                dtype=float,
            )
            for index in range(state_count)
        )

    if kind == "Dop853DenseOutput":
        x = np.array([0.0, 1.0])
        one_minus_x = np.array([1.0, -1.0])
        polynomials: list[np.ndarray] = []
        for state_index in range(state_count):
            polynomial = np.array([0.0])
            for index, coefficient in enumerate(reversed(interpolant.F[:, state_index])):
                polynomial = _poly.polyadd(polynomial, float(coefficient))
                polynomial = _poly.polymul(
                    polynomial, x if index % 2 == 0 else one_minus_x
                )
            polynomial = _poly.polyadd(polynomial, float(interpolant.y_old[state_index]))
            polynomials.append(polynomial)
        return tuple(polynomials)

    if kind == "BdfDenseOutput":
        step = float(interpolant.t - interpolant.t_old)
        polynomials = []
        for state_index in range(state_count):
            polynomial = np.array([float(interpolant.D[0, state_index])])
            product = np.array([1.0])
            for index in range(interpolant.order):
                product = _poly.polymul(
                    product,
                    np.array(
                        [
                            float(
                                (interpolant.t_old - interpolant.t_shift[index])
                                / interpolant.denom[index]
                            ),
                            float(step / interpolant.denom[index]),
                        ]
                    ),
                )
                polynomial = _poly.polyadd(
                    polynomial,
                    _poly.polymul(float(interpolant.D[index + 1, state_index]), product),
                )
            polynomials.append(polynomial)
        return tuple(polynomials)

    raise SimulationError(
        "Dense state extrema are not implemented for solve_ivp interpolant "
        f"{kind!r}; refusing to publish incomplete extrema"
    )


def _batched_derivative_roots(coefficients: np.ndarray) -> np.ndarray:
    """Roots of the derivative of every row of ``coefficients``, batched.

    Row by row this is ``_poly_roots(_poly_derivative(row))`` BIT FOR BIT: the
    same derivative products, the same trailing-zero trim, the same linear
    shortcut, the same companion matrix handed to the same LAPACK routine
    (``eigvals`` on a stack runs it once per matrix). Only the Python loop
    around it is gone; on a 60 ms grid-following run the per-row calls were a
    third of the wall time. Missing roots (lower effective degree) are NaN.
    Shape: (rows, degree - 1), complex.
    """

    rows, length = coefficients.shape
    width = max(length - 2, 0)
    roots = np.full((rows, width), complex(math.nan, math.nan), dtype=complex)
    if width == 0:
        return roots
    # polyder with scl == _DOMAIN_SCALE: c *= scl, then der[j-1] = j * c[j].
    scaled = coefficients * _DOMAIN_SCALE
    derivative = np.empty((rows, length - 1), dtype=float)
    for power in range(length - 1, 0, -1):
        derivative[:, power - 1] = power * scaled[:, power]
    # as_series trims trailing EXACT zeros, keeping at least one coefficient.
    nonzero = derivative != 0.0
    effective = np.where(
        nonzero.any(axis=1),
        derivative.shape[1] - np.argmax(nonzero[:, ::-1], axis=1),
        1,
    )
    for size in np.unique(effective):
        size = int(size)
        if size < 2:
            continue
        selected = np.flatnonzero(effective == size)
        series = derivative[selected, :size]
        if size == 2:
            found = (-series[:, 0] / series[:, 1])[:, np.newaxis].astype(complex)
        else:
            degree = size - 1
            companion = np.zeros((selected.size, degree, degree), dtype=float)
            flat = companion.reshape(selected.size, -1)
            flat[:, degree :: degree + 1] = 1
            companion[:, :, -1] -= series[:, :-1] / series[:, -1:]
            found = np.linalg.eigvals(companion).astype(complex)
        roots[selected, : found.shape[1]] = _WINDOW_OFFSET + _WINDOW_SCALE * found
    return roots


def _batched_poly_value(coefficients: np.ndarray, points: np.ndarray) -> np.ndarray:
    """``_poly_value`` row by row (same Horner statements), NaN points stay NaN."""

    x = _DOMAIN_OFFSET + _DOMAIN_SCALE * points
    value = coefficients[:, -1:] + x * 0
    for index in range(2, coefficients.shape[1] + 1):
        value = coefficients[:, -index : coefficients.shape[1] - index + 1] + value * x
    return value


def _dense_state_extrema(
    solution: Any,
    mode: CompiledMode,
    window_start: float,
    window_end: float,
) -> dict[str, dict[str, float]]:
    """Evaluate every real dense-polynomial derivative root per accepted step.

    Per (step, state) the candidate set is the 17 probe points plus every real
    derivative root inside the window; the extremum is the first candidate, in
    step order and ascending time, that is strictly beyond all earlier ones.
    The whole solution is evaluated as one batch (2026-11-17); the candidate
    set, the arithmetic and the tie rule are the per-step loop's, so the
    result is bit-identical (tests/test_dense_polynomials.py).
    """

    if solution.sol is None or window_end <= window_start:
        return {}
    interpolants = getattr(solution.sol, "interpolants", None)
    if interpolants is None or len(interpolants) != len(solution.t) - 1:
        raise SimulationError(
            "solve_ivp dense output does not expose one interpolant per accepted step; "
            "refusing to publish incomplete state extrema"
        )

    state_count = len(mode.circuit.state_components)
    steps: list[int] = []
    starts: list[float] = []
    ends: list[float] = []
    dense_starts: list[float] = []
    dense_steps: list[float] = []
    for step_index, (raw_start, raw_end) in enumerate(
        zip(solution.t[:-1], solution.t[1:], strict=True)
    ):
        step_start = float(raw_start)
        start = max(step_start, window_start)
        end = min(float(raw_end), window_end)
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
        steps.append(step_index)
        starts.append(start)
        ends.append(end)
        dense_starts.append(dense_start)
        dense_steps.append(dense_step)
    if not steps:
        return {}

    # (step, state, power): every kept step's dense polynomials, one array.
    polynomial_rows = [
        _dense_step_state_polynomials(interpolants[index], state_count) for index in steps
    ]
    length = max(len(row) for polynomials in polynomial_rows for row in polynomials)
    coefficients = np.zeros((len(steps), state_count, length), dtype=float)
    for step_position, polynomials in enumerate(polynomial_rows):
        for state_index, row in enumerate(polynomials):
            # polyadd/polymul trim exact trailing zeros, so a row can be short.
            # Zero padding is exact: the extra Horner steps add +-0 to the
            # leading coefficient and the root search trims them again.
            coefficients[step_position, state_index, : len(row)] = row

    dense_start_array = np.asarray(dense_starts)
    dense_step_array = np.asarray(dense_steps)
    # A terminal root can truncate solution.t inside the integrator's accepted
    # step. The interpolant still uses its original, longer h, so normalize with
    # interpolant.t rather than the event-truncated raw_end.
    normalized_start = (np.asarray(starts) - dense_start_array) / dense_step_array
    normalized_end = (np.asarray(ends) - dense_start_array) / dense_step_array
    probes = np.stack(
        [
            np.linspace(low, high, _STATE_EXTREMA_PROBE_SUBINTERVALS + 1)
            for low, high in zip(normalized_start, normalized_end, strict=True)
        ]
    )

    flat = coefficients.reshape(len(steps) * state_count, length)
    roots = _batched_derivative_roots(flat).reshape(len(steps), state_count, -1)
    real = roots.real
    low = normalized_start[:, np.newaxis, np.newaxis]
    high = normalized_end[:, np.newaxis, np.newaxis]
    keep = (
        np.isfinite(roots.real)
        & np.isfinite(roots.imag)
        & ~(
            np.abs(roots.imag)
            > _STATE_EXTREMA_ROOT_IMAGINARY_TOLERANCE * np.maximum(1.0, np.abs(real))
        )
        & ~(real < low - _STATE_EXTREMA_ROOT_BOUNDARY_TOLERANCE)
        & ~(real > high + _STATE_EXTREMA_ROOT_BOUNDARY_TOLERANCE)
    )
    clamped = np.where(keep, np.minimum(high, np.maximum(low, real)), math.nan)
    candidates = np.concatenate(
        (
            np.broadcast_to(
                probes[:, np.newaxis, :], (len(steps), state_count, probes.shape[1])
            ),
            clamped,
        ),
        axis=2,
    )
    # sorted(set(...)): ascending, duplicates dropped (NaN sorts last).
    candidates = np.sort(candidates, axis=2)
    duplicate = np.zeros(candidates.shape, dtype=bool)
    duplicate[:, :, 1:] = candidates[:, :, 1:] == candidates[:, :, :-1]
    candidates = np.where(duplicate, math.nan, candidates)

    values = _batched_poly_value(
        flat, candidates.reshape(len(steps) * state_count, -1)
    ).reshape(candidates.shape)
    times = (
        dense_start_array[:, np.newaxis, np.newaxis]
        + dense_step_array[:, np.newaxis, np.newaxis] * candidates
    )

    extrema: dict[str, dict[str, float]] = {}
    for component in sorted(mode.circuit.state_components, key=lambda item: item.id):
        state_index = mode.circuit.state_index[component.id]
        # Step-major, then ascending time: the order the per-step loop visited.
        ordered_points = candidates[:, state_index, :].reshape(-1)
        valid = ~np.isnan(ordered_points)
        ordered_values = values[:, state_index, :].reshape(-1)[valid]
        ordered_times = times[:, state_index, :].reshape(-1)[valid]
        # First occurrence wins a tie, exactly as the strict < / > updates did.
        minimum = int(np.argmin(ordered_values))
        maximum = int(np.argmax(ordered_values))
        extrema[component.id] = {
            "minimum": float(ordered_values[minimum]),
            "minimum_time_s": float(ordered_times[minimum]),
            "maximum": float(ordered_values[maximum]),
            "maximum_time_s": float(ordered_times[maximum]),
        }
    return extrema


def _merge_state_extrema(
    target: dict[str, dict[str, float]],
    source: dict[str, dict[str, float]],
) -> None:
    for component_id, candidate in source.items():
        record = target.get(component_id)
        if record is None:
            target[component_id] = dict(candidate)
            continue
        if candidate["minimum"] < record["minimum"]:
            record["minimum"] = candidate["minimum"]
            record["minimum_time_s"] = candidate["minimum_time_s"]
        if candidate["maximum"] > record["maximum"]:
            record["maximum"] = candidate["maximum"]
            record["maximum_time_s"] = candidate["maximum_time_s"]


class _MetricAccumulator:
    def __init__(
        self,
        model: Model,
        circuit: Circuit,
        initial_state: np.ndarray,
        *,
        evidence_mode: str = "full",
    ):
        if evidence_mode not in EVIDENCE_MODES:
            raise SimulationError(
                f"Unknown evidence mode {evidence_mode!r}; expected one of "
                + ", ".join(EVIDENCE_MODES)
            )
        self.model = model
        self.circuit = circuit
        self.evidence_mode = evidence_mode
        self._component_energy_requested = evidence_mode == "full"
        self.midpoint = 0.5 * (model.transient.start + model.transient.stop)
        self.full = _WindowAccumulator.create(
            model,
            model.transient.start,
            model.transient.stop,
            component_energy_requested=self._component_energy_requested,
        )
        self.last_half = _WindowAccumulator.create(
            model,
            self.midpoint,
            model.transient.stop,
            component_energy_requested=self._component_energy_requested,
        )
        self.initial_state = initial_state.copy()
        self.last_half_initial_state: np.ndarray | None = None
        # Impulsive jumps (impulsive_events: conserve), set by the simulator
        # before finalize; empty for every run without one.
        self.impulses: list[dict[str, Any]] = []
        self.topology_segments = 0
        self._seen_topologies: set[Any] = set()
        self._output_maps: dict[str, list[tuple[np.ndarray, float, np.ndarray]]] = {
            name: [] for name in electrical_outputs(model)
        }

    def _record_mode(self, mode: CompiledMode) -> None:
        if mode.topology in self._seen_topologies:
            return
        self._seen_topologies.add(mode.topology)
        for name in electrical_outputs(self.model):
            self._output_maps[name].append(_output_affine_map(mode, name))

    def begin_segment(self, solution: Any, mode: CompiledMode) -> bool:
        """Book-keep one integrated segment; False if it has nothing to integrate.

        Shared by the reference quadrature below and the compiled kernel's
        (pear_sim.core.rust_backend.observe_segment).
        """

        if solution.sol is None or len(solution.t) < 2:
            return False
        segment_start = float(solution.t[0])
        segment_end = float(solution.t[-1])
        if segment_end <= segment_start:
            return False
        self.topology_segments += 1
        self._record_mode(mode)
        if self.last_half_initial_state is None and segment_start <= self.midpoint <= segment_end:
            self.last_half_initial_state = np.asarray(solution.sol(self.midpoint), dtype=float)
        return True

    def add_solution(self, solution: Any, mode: CompiledMode) -> None:
        if not self.begin_segment(solution, mode):
            return

        for raw_start, raw_end in zip(solution.t[:-1], solution.t[1:], strict=True):
            start = float(raw_start)
            end = float(raw_end)
            if end <= start:
                continue
            contribution = _integrate_dense_interval(
                solution.sol,
                mode,
                start,
                end,
                component_energy_requested=self._component_energy_requested,
            )
            self.full.add(contribution)
            if end <= self.midpoint:
                continue
            if start >= self.midpoint:
                self.last_half.add(contribution)
            else:
                self.last_half.add(
                    _integrate_dense_interval(
                        solution.sol,
                        mode,
                        self.midpoint,
                        end,
                        component_energy_requested=self._component_energy_requested,
                    )
                )

    def _continuous_outputs(self) -> frozenset[str]:
        continuous: set[str] = set()
        for name, maps in self._output_maps.items():
            if not maps:
                continue
            reference_coefficients, reference_constant, reference_sources = maps[0]
            if all(
                np.allclose(
                    coefficients,
                    reference_coefficients,
                    rtol=1e-10,
                    atol=1e-12,
                )
                and math.isclose(
                    constant,
                    reference_constant,
                    rel_tol=1e-10,
                    abs_tol=1e-12,
                )
                and np.allclose(
                    source_coefficients,
                    reference_sources,
                    rtol=1e-10,
                    atol=1e-12,
                )
                for coefficients, constant, source_coefficients in maps[1:]
            ):
                continuous.add(name)
        return frozenset(continuous)

    @staticmethod
    def _time_window(window: _WindowAccumulator) -> dict[str, Any]:
        return {
            "start_s": window.start,
            "end_s": window.end,
            "duration_s": window.integrated_interval_s,
            "quadrature_intervals": window.quadrature_intervals,
            "signals": {
                name: {
                    "integral": window.signal_integrals[name],
                    "square_integral": window.signal_square_integrals[name],
                }
                for name in sorted(window.signal_integrals)
            },
        }

    def _physics_window(
        self,
        window: _WindowAccumulator,
        start_state: np.ndarray,
        end_state: np.ndarray,
    ) -> dict[str, Any]:
        start_storage = _storage_energy_by_component(self.circuit, start_state)
        end_storage = _storage_energy_by_component(self.circuit, end_state)
        storage_change = {
            component_id: end_storage[component_id] - start_storage[component_id]
            for component_id in sorted(start_storage)
        }
        component_energy = {
            component_id: window.component_absorbed_energy[component_id]
            for component_id in sorted(window.component_absorbed_energy)
        }
        storage_ids = set(start_storage)
        # A machine is expanded into windings and back-EMF voltage sources; those
        # sources carry the electromagnetic (shaft) power, not a supply, so they
        # are accounted apart from the independent sources.
        machine_prefixes = tuple(f"{machine.id}." for machine in self.model.machines)
        machine_source_ids = {
            component.id
            for component in self.circuit.components
            if component.type in {"voltage_source", "current_source"}
            and component.id.startswith(machine_prefixes or ("\0",))
        }
        source_ids = {
            component.id
            for component in self.circuit.components
            if component.type in {"voltage_source", "current_source"}
        } - machine_source_ids
        resistor_ids = {
            component.id for component in self.circuit.components if component.type == "resistor"
        }
        delta_storage = float(sum(storage_change.values()))
        integrated_storage_power = float(
            sum(component_energy[component_id] for component_id in storage_ids)
        )
        source_absorbed = float(sum(component_energy[component_id] for component_id in source_ids))
        source_delivered = -source_absorbed
        resistor_absorbed = float(
            sum(component_energy[component_id] for component_id in resistor_ids)
        )
        non_storage_absorbed = float(
            sum(
                value
                for component_id, value in component_energy.items()
                if component_id not in storage_ids
            )
        )
        # A jump moves storage energy at one instant, outside every quadrature
        # interval: what the sources delivered in it is added to their energy,
        # and what the vanishing loop resistance dissipated is its own term.
        impulses = [
            impulse
            for impulse in self.impulses
            if window.start <= impulse["time_s"] <= window.end
        ]
        if window is self.last_half:
            impulses = [i for i in impulses if i["time_s"] >= self.midpoint]
        commutation_loss = 0.0
        for impulse in impulses:
            commutation_loss += impulse["commutation_loss_j"]
            for component_id, value in impulse["source_absorbed_energy_j"].items():
                component_energy[component_id] = component_energy.get(component_id, 0.0) + value
        if impulses:
            source_absorbed = float(
                sum(component_energy[component_id] for component_id in source_ids)
            )
            source_delivered = -source_absorbed
            non_storage_absorbed = float(
                sum(
                    value
                    for component_id, value in component_energy.items()
                    if component_id not in storage_ids
                )
            )
        balance_residual = delta_storage + non_storage_absorbed + commutation_loss
        balance_scale = abs(delta_storage) + abs(commutation_loss) + sum(
            abs(value)
            for component_id, value in component_energy.items()
            if component_id not in storage_ids
        )
        balance_relative_tolerance = max(1e-9, self.model.transient.rtol * 10.0)
        balance_tolerance = max(1e-15, balance_scale * balance_relative_tolerance)
        balance_within_tolerance = abs(balance_residual) <= balance_tolerance
        machine_energy = float(sum(component_energy[cid] for cid in machine_source_ids))
        # A rotor-position-dependent inductance exchanges energy through
        # 1/2 i^T (dL/dt) i — the reluctance power — which this audit does not
        # represent (stored energy uses constant inductances). Its residual is
        # then a missing term, not a numerical error, and is reported as such.
        position_dependent = any(machine.is_salient for machine in self.model.machines)
        if position_dependent:
            balance_status = "not_assessed"
        elif balance_within_tolerance:
            balance_status = "within_numerical_tolerance"
        else:
            balance_status = "outside_numerical_tolerance"
        return {
            "window_start_s": window.start,
            "window_end_s": window.end,
            "duration_s": window.integrated_interval_s,
            "quadrature_intervals": window.quadrature_intervals,
            "component_absorbed_energy_j": component_energy,
            "storage_energy_start_j": float(sum(start_storage.values())),
            "storage_energy_end_j": float(sum(end_storage.values())),
            "storage_energy_change_j": storage_change,
            "delta_storage_energy_j": delta_storage,
            "integrated_storage_power_j": integrated_storage_power,
            "storage_constitutive_residual_j": integrated_storage_power - delta_storage,
            "source_absorbed_energy_j": source_absorbed,
            "source_delivered_energy_j": source_delivered,
            "source_delivered_mean_power_w": source_delivered / window.integrated_interval_s,
            "resistor_absorbed_energy_j": resistor_absorbed,
            "resistor_absorbed_mean_power_w": resistor_absorbed / window.integrated_interval_s,
            "non_storage_absorbed_energy_j": non_storage_absorbed,
            "network_power_residual_j": float(sum(component_energy.values())),
            "balance_residual_j": balance_residual,
            "balance_residual_relative": abs(balance_residual) / max(balance_scale, 1e-30),
            "balance_tolerance_j": balance_tolerance,
            "balance_tolerance_relative": balance_relative_tolerance,
            "balance_within_tolerance": None if position_dependent else balance_within_tolerance,
            "balance_status": balance_status,
            **(
                {
                    "balance_not_assessed_reason": (
                        "position-dependent inductance (salient rotor or flux map): the "
                        "reluctance power 1/2 i^T (dL/dt) i is not represented in this audit"
                    )
                }
                if position_dependent
                else {}
            ),
            "machine_electromagnetic_energy_j": machine_energy,
            "machine_electromagnetic_mean_power_w": machine_energy / window.integrated_interval_s,
            **self._parasitic_window(component_energy, window.integrated_interval_s),
            **(
                {
                    "commutation_loss_j": commutation_loss,
                    "impulsive_commutations": len(impulses),
                }
                if self.model.transient.impulsive_events == "conserve"
                else {}
            ),
        }

    def _parasitic_window(
        self, component_energy: dict[str, float], duration: float
    ) -> dict[str, Any]:
        """Group expanded ESR/DCR dissipation under the user's component.

        The expanded resistors are ordinary resistors, so their energy is
        already inside `resistor_absorbed_energy_j` and the balance closes
        without a special case; this only regroups it. A capacitor's RMS
        current follows exactly from its ESR's energy, because the ESR carries
        the capacitor current: I_rms = sqrt(E / (ESR * T)). Absent (not empty)
        when the model declares no `esr`/`dcr`, so older bundles do not move.
        """

        records = getattr(self.model, "passive_parasitics", ())
        if not records:
            return {}
        resistance = {
            component.id: float(component.value)
            for component in self.circuit.components
            if component.type == "resistor"
        }
        loss_w: dict[str, float] = {}
        rms_a: dict[str, float] = {}
        total = 0.0
        for record in sorted(records, key=lambda item: item.component_id):
            energy = float(component_energy[record.resistor_id])
            total += energy
            loss_w[record.component_id] = energy / duration
            if record.kind == "esr":
                # Quadrature noise can leave a vanishing negative energy on a
                # near-idle capacitor; the RMS of a real current is never imaginary.
                rms_a[record.component_id] = math.sqrt(
                    max(energy, 0.0) / (resistance[record.resistor_id] * duration)
                )
        return {
            "parasitic_absorbed_energy_j": total,
            "parasitic_loss_by_component_w": loss_w,
            **({"capacitor_rms_current_a": rms_a} if rms_a else {}),
        }

    def finalize(
        self,
        final_state: np.ndarray,
        tolerance: float,
    ) -> tuple[dict[str, Any], frozenset[str], dict[str, Any]]:
        expected_full = self.model.transient.stop - self.model.transient.start
        expected_tail = self.model.transient.stop - self.midpoint
        duration_tolerance = max(tolerance * 10.0, expected_full * 1e-10)
        if abs(self.full.integrated_interval_s - expected_full) > duration_tolerance:
            raise SimulationError(
                "Metric integration does not cover the full simulation interval: "
                f"got {self.full.integrated_interval_s:.12g} s, expected {expected_full:.12g} s"
            )
        if abs(self.last_half.integrated_interval_s - expected_tail) > duration_tolerance:
            raise SimulationError(
                "Metric integration does not cover the last-half interval: "
                f"got {self.last_half.integrated_interval_s:.12g} s, "
                f"expected {expected_tail:.12g} s"
            )
        if self.last_half_initial_state is None:
            raise SimulationError("Metric integration did not capture the last-half boundary state")

        time_integrals = {
            "method": _TIME_METRIC_METHOD,
            "quadrature_order": _QUADRATURE_ORDER,
            "full": self._time_window(self.full),
            "last_half": self._time_window(self.last_half),
        }
        if self._component_energy_requested:
            physics_audit = {
                "schema": "pear-sim/physics-audit/v0.1",
                "method": _TIME_METRIC_METHOD,
                "power_sign_convention": "positive_absorbed_from_p_to_n",
                "full": self._physics_window(
                    self.full,
                    self.initial_state,
                    final_state,
                ),
                "last_half": self._physics_window(
                    self.last_half,
                    self.last_half_initial_state,
                    final_state,
                ),
            }
        else:
            # A declaration of absence, not an empty balance. The `full` and
            # `last_half` windows are deliberately missing rather than zeroed:
            # a reader that indexes them gets a KeyError instead of a residual
            # that looks conserved because nothing was ever integrated.
            physics_audit = {
                "schema": "pear-sim/physics-audit/v0.1",
                "status": "not_computed",
                "evidence_mode": self.evidence_mode,
                "reason": (
                    "The lean evidence mode skips the per-component power quadrature, "
                    "so no energy balance was integrated for this run."
                ),
                "restore_with": "re-run without --lean (the full evidence mode)",
            }
        if self.impulses:
            # Every jump, with its loop, mismatch, transferred charge and loss —
            # also in lean mode, where no balance is integrated.
            physics_audit["impulsive_commutations"] = [dict(item) for item in self.impulses]
        return time_integrals, self._continuous_outputs(), physics_audit
