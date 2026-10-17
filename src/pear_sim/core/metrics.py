"""Dense-output metric quadrature, energy bookkeeping, and state extrema.

Extracted verbatim from ``pear_sim.core.solver`` (restructuring stage 2, move 8).
The simulator feeds accepted ``solve_ivp`` dense-output steps into
``_MetricAccumulator``; nothing here integrates the circuit itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np
from numpy.polynomial import Polynomial

from pear_sim.spec.signals import is_declared_signal, is_machine_signal
from pear_sim.core.circuit import Circuit, CompiledMode
from pear_sim.errors import SimulationError
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


def _dense_step_state_polynomials(
    interpolant: Any,
    state_count: int,
) -> tuple[Polynomial, ...]:
    """Reconstruct a solve_ivp step's dense state polynomials in x=(t-t0)/h."""

    kind = type(interpolant).__name__
    if kind in {"RkDenseOutput", "RadauDenseOutput"}:
        factor = float(interpolant.h) if kind == "RkDenseOutput" else 1.0
        return tuple(
            Polynomial(
                np.concatenate(
                    (
                        np.asarray([interpolant.y_old[index]], dtype=float),
                        factor * np.asarray(interpolant.Q[index, :], dtype=float),
                    )
                )
            )
            for index in range(state_count)
        )

    if kind == "Dop853DenseOutput":
        x = Polynomial([0.0, 1.0])
        one_minus_x = Polynomial([1.0, -1.0])
        polynomials: list[Polynomial] = []
        for state_index in range(state_count):
            polynomial = Polynomial([0.0])
            for index, coefficient in enumerate(reversed(interpolant.F[:, state_index])):
                polynomial += float(coefficient)
                polynomial *= x if index % 2 == 0 else one_minus_x
            polynomial += float(interpolant.y_old[state_index])
            polynomials.append(polynomial)
        return tuple(polynomials)

    if kind == "BdfDenseOutput":
        step = float(interpolant.t - interpolant.t_old)
        polynomials = []
        for state_index in range(state_count):
            polynomial = Polynomial([float(interpolant.D[0, state_index])])
            product = Polynomial([1.0])
            for index in range(interpolant.order):
                product *= Polynomial(
                    [
                        float(
                            (interpolant.t_old - interpolant.t_shift[index])
                            / interpolant.denom[index]
                        ),
                        float(step / interpolant.denom[index]),
                    ]
                )
                polynomial += float(interpolant.D[index + 1, state_index]) * product
            polynomials.append(polynomial)
        return tuple(polynomials)

    raise SimulationError(
        "Dense state extrema are not implemented for solve_ivp interpolant "
        f"{kind!r}; refusing to publish incomplete extrema"
    )


def _dense_state_extrema(
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
            for raw_root in polynomial.deriv().roots():
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
            values = np.asarray(polynomial(unique_points), dtype=float)
            times = dense_start + dense_step * unique_points
            for time, value in zip(times, values, strict=True):
                consider(component.id, float(value), float(time))
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

    def add_solution(self, solution: Any, mode: CompiledMode) -> None:
        if solution.sol is None or len(solution.t) < 2:
            return
        segment_start = float(solution.t[0])
        segment_end = float(solution.t[-1])
        if segment_end <= segment_start:
            return
        self.topology_segments += 1
        self._record_mode(mode)
        if self.last_half_initial_state is None and segment_start <= self.midpoint <= segment_end:
            self.last_half_initial_state = np.asarray(solution.sol(self.midpoint), dtype=float)

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
        balance_residual = delta_storage + non_storage_absorbed
        balance_scale = abs(delta_storage) + sum(
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
        return time_integrals, self._continuous_outputs(), physics_audit
