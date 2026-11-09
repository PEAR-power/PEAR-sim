# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Independent periodic CCM/BCM/DCM reference for a canonical ideal boost.

The reference equations in this module are derived locally.  The oracle does not
import PEARSim's circuit compiler, transient solver, event manager, or mode
classifier.  ``observe_boost_simulation_result`` is only an adapter from an already
completed public result into the comparison contract.

This is mathematical verification evidence for one exact ideal topology.  It is not
hardware validation, a compatibility claim against another simulator, or evidence for arbitrary DAE/DCM
networks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Literal, Sequence

import numpy as np
from scipy.optimize import brentq

from pear_sim.benchmarks.affine import (
    affine_map as _affine_map,
    propagate as _propagate,
    propagate_with_coordinate_integral as _propagate_with_voltage_integral,
)
from pear_sim.spec.model import Model
from pear_sim.spec.canonical_topologies import (
    BoostBoundaryBenchmarkError,
    BoostBoundaryParameters,
    BoostFixture as _Fixture,
    extract_boost_boundary_parameters,
    extract_boost_fixture as _extract_fixture,
)
from pear_sim.benchmarks._shared import (
    append_state_error as _append_state_error,
    comparison_gate as _comparison_gate,
    exact_gate as _exact_gate,
    oracle_gate as _oracle_gate,
    relative_or_scaled_error as _relative_or_scaled_error,
)


BOOST_BOUNDARY_ORACLE_SCHEMA = "pear-sim/benchmark-oracle/ideal-boost-boundary/v0.1"
BOOST_BOUNDARY_RESULT_SCHEMA = "pear-sim/benchmark-result/ideal-boost-boundary/v0.1"
BOOST_BOUNDARY_KIND = "independent-periodic-affine-boost-boundary-verification"
BOOST_DEFAULT_LOAD_RATIOS = (0.25, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 200.0)

BCM_COINCIDENCE_FRACTION_TOLERANCE = 1e-8
EVENT_TIME_RELATIVE_LIMIT = 5e-4
REVERSE_CURRENT_NORMALIZED_LIMIT = 1e-8
BOUNDARY_ZERO_NORMALIZED_LIMIT = 1e-8
BOUNDARY_STATE_RELATIVE_LIMIT = 1e-5
CYCLE_MEAN_RELATIVE_LIMIT = 1e-3
STATE_RIPPLE_RELATIVE_LIMIT = 1e-3
MODE_FRACTION_ABSOLUTE_LIMIT = 1e-3
ENERGY_BALANCE_RELATIVE_LIMIT = 1e-4
CYCLE_PERIODICITY_NORMALIZED_LIMIT = 1e-8
UNCLASSIFIED_FRACTION_LIMIT = 1e-10
MODE_FRACTION_SUM_ERROR_LIMIT = 1e-10
MAX_ANALYTIC_EXTREMA_PER_INTERVAL = 10_000
_REGIME_BOUNDARY_RELATIVE_TOLERANCE = 2e-12

Regime = Literal["CCM", "BCM", "DCM"]


@dataclass(frozen=True)
class BoostState:
    inductor_current_a: float
    output_voltage_v: float

    def as_vector(self) -> np.ndarray:
        return np.asarray([self.inductor_current_a, self.output_voltage_v], dtype=float)


@dataclass(frozen=True)
class BoostBoundaryPoint:
    point_id: str
    load_ratio_to_bcm: float
    load_ohm: float
    regime: Regime
    start_state: BoostState
    switch_off_state: BoostState
    diode_zero_state: BoostState | None
    end_state: BoostState
    switch_off_s: float
    diode_zero_s: float | None
    mean_output_voltage_v: float
    minimum_inductor_current_a: float
    inductor_current_peak_to_peak_a: float
    output_voltage_peak_to_peak_v: float
    fraction_by_mode: dict[str, float]
    mode_sequence: tuple[str, ...]
    event_sequence: tuple[str, ...]
    periodicity_residual_normalized: float
    oracle_gates: tuple[dict[str, Any], ...]

    @property
    def conduction_fraction(self) -> float:
        return self.fraction_by_mode["on"] + self.fraction_by_mode["freewheel"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "point_id": self.point_id,
            "load_ratio_to_bcm": self.load_ratio_to_bcm,
            "load_ohm": self.load_ohm,
            "regime": self.regime,
            "states": {
                "cycle_start": asdict(self.start_state),
                "switch_off": asdict(self.switch_off_state),
                "diode_zero": (
                    None if self.diode_zero_state is None else asdict(self.diode_zero_state)
                ),
                "cycle_end": asdict(self.end_state),
            },
            "events": {
                "switch_off_s": self.switch_off_s,
                "diode_zero_s": self.diode_zero_s,
                "mode_sequence": list(self.mode_sequence),
                "event_sequence": list(self.event_sequence),
            },
            "metrics": {
                "mean_output_voltage_v": self.mean_output_voltage_v,
                "minimum_inductor_current_a": self.minimum_inductor_current_a,
                "inductor_current_peak_to_peak_a": self.inductor_current_peak_to_peak_a,
                "output_voltage_peak_to_peak_v": self.output_voltage_peak_to_peak_v,
                "conduction_fraction": self.conduction_fraction,
                "fraction_by_mode": dict(self.fraction_by_mode),
                "periodicity_residual_normalized": self.periodicity_residual_normalized,
            },
            "oracle_gates": [dict(gate) for gate in self.oracle_gates],
        }


@dataclass(frozen=True)
class BoostBoundarySweep:
    parameters: BoostBoundaryParameters
    bcm_load_ohm: float
    load_ratios: tuple[float, ...]
    points: tuple[BoostBoundaryPoint, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": BOOST_BOUNDARY_ORACLE_SCHEMA,
            "kind": BOOST_BOUNDARY_KIND,
            "claim": _claim(),
            "method": {
                "periodic_ccm": (
                    "fixed point of independently derived ON/freewheel affine "
                    "matrix-exponential map"
                ),
                "bcm_boundary": (
                    "brentq root of periodic CCM cycle-start inductor current versus load"
                ),
                "periodic_dcm": (
                    "scalar periodic-voltage root with first directed zero-current "
                    "root bracketed by analytical current extrema"
                ),
                "integrals": "augmented matrix exponential and exact idle RC integral",
                "pear_sim_circuit_or_solver_imported": False,
            },
            "parameters_si": self.parameters.as_dict(),
            "boundary": {
                "load_ohm": self.bcm_load_ohm,
                "definition": "periodic CCM cycle-start inductor current equals zero",
                "coincidence_fraction_tolerance": BCM_COINCIDENCE_FRACTION_TOLERANCE,
            },
            "load_ratios": list(self.load_ratios),
            "comparison_gate_contract": _comparison_gate_contract(),
            "oracle_health": {
                "status": "pass",
                "point_count": len(self.points),
                "failed_gate_count": 0,
            },
            "points": [point.as_dict() for point in self.points],
        }


@dataclass(frozen=True)
class BoostBoundaryObservation:
    point_id: str
    parameters: BoostBoundaryParameters
    load_ratio_to_bcm: float
    load_ohm: float
    regime: Regime
    start_state: BoostState
    switch_off_state: BoostState
    diode_zero_state: BoostState | None
    end_state: BoostState
    diode_zero_s: float | None
    mean_output_voltage_v: float
    minimum_inductor_current_a: float
    inductor_current_peak_to_peak_a: float
    output_voltage_peak_to_peak_v: float
    fraction_by_mode: dict[str, float]
    mode_sequence: tuple[str, ...]
    event_sequence: tuple[str, ...]
    energy_balance_relative: float

    @property
    def conduction_fraction(self) -> float:
        return self.fraction_by_mode["on"] + self.fraction_by_mode["freewheel"]


def find_boost_bcm_load(parameters: BoostBoundaryParameters) -> float:
    """Numerically locate the periodic CCM point whose minimum current is zero."""

    textbook_seed = (
        2.0
        * parameters.inductance_h
        * parameters.switching_frequency_hz
        / (parameters.duty * (1.0 - parameters.duty) ** 2)
    )
    if not math.isfinite(textbook_seed) or textbook_seed <= 0.0:
        raise BoostBoundaryBenchmarkError("textbook boundary seed is not finite and positive")

    def start_current(load_ohm: float) -> float:
        return float(_ccm_periodic_state(parameters, load_ohm)[0])

    lower = textbook_seed
    upper = textbook_seed
    lower_value = start_current(lower)
    upper_value = lower_value
    for _ in range(80):
        if lower_value > 0.0 and upper_value < 0.0:
            break
        if lower_value <= 0.0:
            lower *= 0.5
            lower_value = start_current(lower)
        if upper_value >= 0.0:
            upper *= 2.0
            upper_value = start_current(upper)
    else:
        raise BoostBoundaryBenchmarkError("Could not bracket the CCM/BCM load boundary")

    root = float(
        brentq(
            start_current,
            lower,
            upper,
            xtol=max(np.nextafter(0.0, 1.0), textbook_seed * 1e-14),
            rtol=1e-14,
            maxiter=200,
        )
    )
    if not math.isfinite(root) or root <= 0.0:
        raise BoostBoundaryBenchmarkError("BCM load root is not finite and positive")
    return root


def solve_periodic_boost(
    parameters: BoostBoundaryParameters,
    load_ohm: float,
    *,
    bcm_load_ohm: float | None = None,
    load_ratio_to_bcm: float | None = None,
) -> BoostBoundaryPoint:
    if not math.isfinite(load_ohm) or load_ohm <= 0.0:
        raise BoostBoundaryBenchmarkError("load_ohm must be finite and positive")
    computed_boundary = find_boost_bcm_load(parameters)
    boundary = computed_boundary if bcm_load_ohm is None else bcm_load_ohm
    if not math.isfinite(boundary) or boundary <= 0.0:
        raise BoostBoundaryBenchmarkError("bcm_load_ohm must be finite and positive")
    if not math.isclose(boundary, computed_boundary, rel_tol=2e-12, abs_tol=1e-15):
        raise BoostBoundaryBenchmarkError(
            "supplied bcm_load_ohm does not match the independently evaluated boundary"
        )
    ratio = load_ohm / boundary if load_ratio_to_bcm is None else load_ratio_to_bcm
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise BoostBoundaryBenchmarkError("load_ratio_to_bcm must be finite and positive")
    if not math.isclose(load_ohm / boundary, ratio, rel_tol=2e-13, abs_tol=1e-15):
        raise BoostBoundaryBenchmarkError("load ratio and load value are inconsistent")

    band = max(boundary * _REGIME_BOUNDARY_RELATIVE_TOLERANCE, 1e-15)
    if load_ohm < boundary - band:
        point = _solve_ccm_or_bcm(parameters, load_ohm, ratio, "CCM")
    elif load_ohm > boundary + band:
        point = _solve_dcm(parameters, load_ohm, ratio)
    else:
        if ratio != 1.0:
            raise BoostBoundaryBenchmarkError(
                "a load inside the numerical BCM coincidence band must use exact ratio 1.0"
            )
        point = _solve_ccm_or_bcm(parameters, boundary, 1.0, "BCM")
    _require_oracle_health(point)
    return point


def run_boost_boundary_sweep(
    parameters: BoostBoundaryParameters,
    load_ratios: Sequence[float] = BOOST_DEFAULT_LOAD_RATIOS,
) -> BoostBoundarySweep:
    ratios = tuple(float(value) for value in load_ratios)
    if not ratios:
        raise BoostBoundaryBenchmarkError("load_ratios must not be empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in ratios):
        raise BoostBoundaryBenchmarkError("every load ratio must be finite and positive")
    if any(right <= left for left, right in zip(ratios[:-1], ratios[1:], strict=True)):
        raise BoostBoundaryBenchmarkError(
            "load_ratios must be strictly increasing and unique"
        )
    if not any(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-15) for value in ratios):
        raise BoostBoundaryBenchmarkError("load_ratios must contain the exact BCM ratio 1.0")
    if any(
        value != 1.0 and abs(value - 1.0) <= _REGIME_BOUNDARY_RELATIVE_TOLERANCE
        for value in ratios
    ):
        raise BoostBoundaryBenchmarkError(
            "load ratios inside the numerical BCM coincidence band are ambiguous; "
            "retain only exact ratio 1.0"
        )
    point_ids = [_point_id(ratio) for ratio in ratios]
    if len(set(point_ids)) != len(point_ids):
        raise BoostBoundaryBenchmarkError(
            "load ratios are not distinguishable by the versioned point-id format"
        )

    boundary = find_boost_bcm_load(parameters)
    points = tuple(
        solve_periodic_boost(
            parameters,
            boundary * ratio,
            bcm_load_ohm=boundary,
            load_ratio_to_bcm=ratio,
        )
        for ratio in ratios
    )
    return BoostBoundarySweep(parameters, boundary, ratios, points)


def observe_boost_simulation_result(
    result: Any,
    reference: BoostBoundaryPoint,
    *,
    parameters: BoostBoundaryParameters,
    bcm_load_ohm: float,
) -> BoostBoundaryObservation:
    """Project a one-cycle result without using it to generate the reference."""

    model = getattr(result, "model", None)
    if not isinstance(model, Model):
        raise BoostBoundaryBenchmarkError("observation requires a result with a PEARSim Model")
    fixture = _extract_fixture(model)
    observed_parameters = extract_boost_boundary_parameters(model)
    _require_matching_parameters(parameters, observed_parameters, context="observed model")
    if not math.isclose(float(fixture.load.value), reference.load_ohm, rel_tol=2e-12):
        raise BoostBoundaryBenchmarkError("observed load differs from reference")

    start_s = float(model.transient.start)
    stop_s = float(model.transient.stop)
    period = parameters.period_s
    if not math.isclose(stop_s - start_s, period, rel_tol=0.0, abs_tol=period * 1e-10):
        raise BoostBoundaryBenchmarkError("observation must span exactly one switching period")
    intervals = list(getattr(result, "mode_intervals", ()))
    if not intervals:
        raise BoostBoundaryBenchmarkError("observed result has no accepted mode intervals")

    durations = {"on": 0.0, "freewheel": 0.0, "idle": 0.0, "unclassified": 0.0}
    mode_sequence: list[str] = []
    minima = {fixture.inductor.id: math.inf, fixture.capacitor.id: math.inf}
    maxima = {fixture.inductor.id: -math.inf, fixture.capacitor.id: -math.inf}
    covered = 0.0
    previous_end = start_s
    coverage_tolerance = period * 1e-10
    for interval in intervals:
        interval_start = float(interval["start_s"])
        interval_end = float(interval["end_s"])
        if not math.isfinite(interval_start) or not math.isfinite(interval_end):
            raise BoostBoundaryBenchmarkError("accepted interval bounds must be finite")
        if interval_start > previous_end + coverage_tolerance:
            raise BoostBoundaryBenchmarkError("accepted mode intervals contain a gap")
        if interval_start < previous_end - coverage_tolerance:
            raise BoostBoundaryBenchmarkError(
                "accepted mode intervals overlap or are out of order"
            )
        if abs(interval_start - previous_end) <= coverage_tolerance:
            interval_start = previous_end
        duration = interval_end - interval_start
        if duration <= 0.0:
            raise BoostBoundaryBenchmarkError(
                "accepted mode intervals must be ordered and strictly positive"
            )
        previous_end = interval_end
        covered += duration

        gates = interval.get("gates", {})
        diodes = interval.get("diodes", {})
        gate_on = gates.get(fixture.pwm.id)
        diode_on = diodes.get(fixture.diode.id)
        constrained = fixture.inductor.id in interval.get("constrained_inductors", ())
        if gate_on is True and diode_on is False and not constrained:
            mode = "on"
        elif gate_on is False and diode_on is True and not constrained:
            mode = "freewheel"
        elif gate_on is False and diode_on is False and constrained:
            mode = "idle"
        else:
            mode = "unclassified"
        durations[mode] += duration
        mode_sequence.append(mode)

        extrema = interval.get("state_extrema", {})
        for component in (fixture.inductor, fixture.capacitor):
            record = extrema.get(component.id)
            if not isinstance(record, dict) or not {"minimum", "maximum"} <= set(record):
                raise BoostBoundaryBenchmarkError(
                    f"accepted interval lacks extrema for {component.id}"
                )
            minimum = float(record["minimum"])
            maximum = float(record["maximum"])
            if not math.isfinite(minimum) or not math.isfinite(maximum) or maximum < minimum:
                raise BoostBoundaryBenchmarkError("accepted state extrema must be finite and ordered")
            minima[component.id] = min(minima[component.id], minimum)
            maxima[component.id] = max(maxima[component.id], maximum)

    if not math.isclose(previous_end, stop_s, rel_tol=0.0, abs_tol=coverage_tolerance):
        raise BoostBoundaryBenchmarkError("accepted mode intervals do not reach cycle end")
    if not math.isclose(covered, period, rel_tol=1e-10, abs_tol=coverage_tolerance):
        raise BoostBoundaryBenchmarkError("accepted intervals do not cover one complete cycle")
    fractions = {name: duration / period for name, duration in durations.items()}

    natural_turnoffs = [
        event
        for event in getattr(result, "events", ())
        if getattr(event, "kind", None) == "natural_commutation"
        and getattr(event, "target", None) == fixture.diode.id
        and getattr(event, "previous", None) is True
        and getattr(event, "current", None) is False
    ]
    if len(natural_turnoffs) > 1:
        raise BoostBoundaryBenchmarkError("more than one diode turn-off occurred in one cycle")
    diode_zero_s = None if not natural_turnoffs else float(natural_turnoffs[0].time) - start_s

    if fractions["idle"] > BCM_COINCIDENCE_FRACTION_TOLERANCE:
        regime: Regime = "DCM"
    elif diode_zero_s is not None and abs(diode_zero_s - period) / period <= EVENT_TIME_RELATIVE_LIMIT:
        regime = "BCM"
    elif diode_zero_s is None:
        regime = "CCM"
    else:
        regime = "DCM"

    switch_target = start_s + parameters.on_duration_s
    switch_interval = min(
        intervals, key=lambda interval: abs(float(interval["end_s"]) - switch_target)
    )
    if abs(float(switch_interval["end_s"]) - switch_target) > period * EVENT_TIME_RELATIVE_LIMIT:
        raise BoostBoundaryBenchmarkError("could not identify the commanded switch-off boundary")

    diode_zero_state: BoostState | None = None
    if diode_zero_s is not None:
        zero_target = start_s + diode_zero_s
        zero_interval = min(
            intervals, key=lambda interval: abs(float(interval["end_s"]) - zero_target)
        )
        if abs(float(zero_interval["end_s"]) - zero_target) > period * EVENT_TIME_RELATIVE_LIMIT:
            raise BoostBoundaryBenchmarkError("could not identify the natural-event boundary state")
        diode_zero_state = _state_from_mapping(
            zero_interval["state_end"], fixture.inductor.id, fixture.capacitor.id
        )

    output_expression = f"v({fixture.diode.n})"
    metrics = result.metrics()
    if output_expression not in metrics:
        raise BoostBoundaryBenchmarkError(
            f"observation requires requested output {output_expression!r}"
        )
    audit = getattr(result, "physics_audit", {}).get("full")
    if not isinstance(audit, dict) or "balance_residual_relative" not in audit:
        raise BoostBoundaryBenchmarkError("observation requires the full-cycle physics audit")

    observation = BoostBoundaryObservation(
        point_id=reference.point_id,
        parameters=observed_parameters,
        load_ratio_to_bcm=float(fixture.load.value) / bcm_load_ohm,
        load_ohm=float(fixture.load.value),
        regime=regime,
        start_state=_state_from_mapping(
            intervals[0]["state_start"], fixture.inductor.id, fixture.capacitor.id
        ),
        switch_off_state=_state_from_mapping(
            switch_interval["state_end"], fixture.inductor.id, fixture.capacitor.id
        ),
        diode_zero_state=diode_zero_state,
        end_state=_state_from_mapping(
            intervals[-1]["state_end"], fixture.inductor.id, fixture.capacitor.id
        ),
        diode_zero_s=diode_zero_s,
        mean_output_voltage_v=float(metrics[output_expression]["mean"]),
        minimum_inductor_current_a=minima[fixture.inductor.id],
        inductor_current_peak_to_peak_a=(
            maxima[fixture.inductor.id] - minima[fixture.inductor.id]
        ),
        output_voltage_peak_to_peak_v=(
            maxima[fixture.capacitor.id] - minima[fixture.capacitor.id]
        ),
        fraction_by_mode=fractions,
        mode_sequence=tuple(mode_sequence),
        event_sequence=_observed_event_sequence(
            result,
            fixture,
            start_s=start_s,
            reference=reference,
            parameters=parameters,
        ),
        energy_balance_relative=float(audit["balance_residual_relative"]),
    )
    _validate_observation(observation)
    if not math.isclose(
        observation.load_ratio_to_bcm,
        reference.load_ratio_to_bcm,
        rel_tol=2e-12,
        abs_tol=1e-15,
    ):
        raise BoostBoundaryBenchmarkError("observed load ratio differs from reference")
    return observation


def compare_boost_boundary_sweep(
    reference: BoostBoundarySweep,
    observations: Sequence[BoostBoundaryObservation],
) -> dict[str, Any]:
    observed_by_id: dict[str, BoostBoundaryObservation] = {}
    for observation in observations:
        if observation.point_id in observed_by_id:
            raise BoostBoundaryBenchmarkError(f"duplicate observation {observation.point_id!r}")
        observed_by_id[observation.point_id] = observation
    expected_ids = [point.point_id for point in reference.points]
    if set(observed_by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(observed_by_id))
        extra = sorted(set(observed_by_id) - set(expected_ids))
        raise BoostBoundaryBenchmarkError(
            f"observation set does not match sweep; missing={missing}, extra={extra}"
        )
    points = [
        compare_boost_boundary_point(
            reference.parameters, point, observed_by_id[point.point_id]
        )
        for point in reference.points
    ]
    failed = sum(gate["status"] == "fail" for point in points for gate in point["gates"])
    return {
        "schema": BOOST_BOUNDARY_RESULT_SCHEMA,
        "kind": "ideal-boost-boundary-sweep-comparison",
        "claim": _claim(),
        "oracle_schema": BOOST_BOUNDARY_ORACLE_SCHEMA,
        "parameters_si": reference.parameters.as_dict(),
        "boundary_load_ohm": reference.bcm_load_ohm,
        "gate_contract": _comparison_gate_contract(),
        "summary": {
            "status": "pass" if failed == 0 else "fail",
            "point_count": len(points),
            "gate_count": sum(len(point["gates"]) for point in points),
            "failed_gate_count": failed,
            "not_run_gate_count": 0,
        },
        "points": points,
    }


def compare_boost_boundary_point(
    parameters: BoostBoundaryParameters,
    reference: BoostBoundaryPoint,
    observation: BoostBoundaryObservation,
) -> dict[str, Any]:
    _validate_observation(observation)
    if observation.point_id != reference.point_id:
        raise BoostBoundaryBenchmarkError(
            f"point-id mismatch: expected {reference.point_id!r}, got {observation.point_id!r}"
        )
    _require_matching_parameters(parameters, observation.parameters, context=reference.point_id)
    if not math.isclose(observation.load_ohm, reference.load_ohm, rel_tol=2e-12):
        raise BoostBoundaryBenchmarkError(f"load mismatch for {reference.point_id}")
    if not math.isclose(
        observation.load_ratio_to_bcm,
        reference.load_ratio_to_bcm,
        rel_tol=2e-12,
        abs_tol=1e-15,
    ):
        raise BoostBoundaryBenchmarkError(f"load-ratio mismatch for {reference.point_id}")

    nonzero_errors: list[float] = []
    zero_errors: list[float] = []
    pairs = [
        (reference.start_state, observation.start_state),
        (reference.switch_off_state, observation.switch_off_state),
        (reference.end_state, observation.end_state),
    ]
    if reference.diode_zero_state is not None:
        if observation.diode_zero_state is None:
            zero_errors.append(math.inf)
        else:
            pairs.append((reference.diode_zero_state, observation.diode_zero_state))
    for expected, actual in pairs:
        _append_state_error(
            expected.inductor_current_a,
            actual.inductor_current_a,
            parameters.current_base_a,
            nonzero_errors,
            zero_errors,
        )
        _append_state_error(
            expected.output_voltage_v,
            actual.output_voltage_v,
            parameters.vin_v,
            nonzero_errors,
            zero_errors,
        )

    if reference.diode_zero_s is None:
        event_error = 0.0 if observation.diode_zero_s is None else math.inf
    elif observation.diode_zero_s is None:
        event_error = math.inf
    else:
        event_error = abs(observation.diode_zero_s - reference.diode_zero_s) / parameters.period_s
    mean_error = abs(
        observation.mean_output_voltage_v - reference.mean_output_voltage_v
    ) / max(abs(reference.mean_output_voltage_v), parameters.vin_v * 1e-12)
    ripple_error = max(
        _relative_or_scaled_error(
            reference.inductor_current_peak_to_peak_a,
            observation.inductor_current_peak_to_peak_a,
            parameters.current_base_a,
        ),
        _relative_or_scaled_error(
            reference.output_voltage_peak_to_peak_v,
            observation.output_voltage_peak_to_peak_v,
            parameters.vin_v,
        ),
    )
    fraction_error = max(
        abs(observation.fraction_by_mode.get(name, math.inf) - expected)
        for name, expected in reference.fraction_by_mode.items()
    )
    reverse = max(0.0, -observation.minimum_inductor_current_a) / parameters.current_base_a
    periodicity = max(
        abs(
            observation.end_state.inductor_current_a
            - observation.start_state.inductor_current_a
        )
        / parameters.current_base_a,
        abs(observation.end_state.output_voltage_v - observation.start_state.output_voltage_v)
        / parameters.vin_v,
    )
    unclassified = observation.fraction_by_mode["unclassified"]
    fraction_sum_error = abs(sum(observation.fraction_by_mode.values()) - 1.0)
    event_semantics_match = (
        observation.mode_sequence == reference.mode_sequence
        and observation.event_sequence == reference.event_sequence
    )

    gates = [
        _exact_gate("regime", observation.regime, reference.regime),
        _comparison_gate(
            "boundary_state_nonzero",
            max(nonzero_errors, default=0.0),
            BOUNDARY_STATE_RELATIVE_LIMIT,
        ),
        _comparison_gate(
            "boundary_state_zero",
            max(zero_errors, default=0.0),
            BOUNDARY_ZERO_NORMALIZED_LIMIT,
        ),
        _comparison_gate("natural_event_time", event_error, EVENT_TIME_RELATIVE_LIMIT),
        _comparison_gate("reverse_current", reverse, REVERSE_CURRENT_NORMALIZED_LIMIT),
        _comparison_gate("cycle_mean_output", mean_error, CYCLE_MEAN_RELATIVE_LIMIT),
        _comparison_gate("state_ripple", ripple_error, STATE_RIPPLE_RELATIVE_LIMIT),
        _comparison_gate("mode_fractions", fraction_error, MODE_FRACTION_ABSOLUTE_LIMIT),
        {
            "id": "event_semantics",
            "metric": "exact_mode_and_event_sequence",
            "actual": event_semantics_match,
            "reference": True,
            "limit": True,
            "observed_mode_sequence": list(observation.mode_sequence),
            "reference_mode_sequence": list(reference.mode_sequence),
            "observed_event_sequence": list(observation.event_sequence),
            "reference_event_sequence": list(reference.event_sequence),
            "status": "pass" if event_semantics_match else "fail",
        },
        _comparison_gate(
            "energy_balance", observation.energy_balance_relative, ENERGY_BALANCE_RELATIVE_LIMIT
        ),
        _comparison_gate(
            "cycle_periodicity", periodicity, CYCLE_PERIODICITY_NORMALIZED_LIMIT
        ),
        _comparison_gate("unclassified_fraction", unclassified, UNCLASSIFIED_FRACTION_LIMIT),
        _comparison_gate(
            "mode_fraction_sum", fraction_sum_error, MODE_FRACTION_SUM_ERROR_LIMIT
        ),
    ]
    return {
        "point_id": reference.point_id,
        "load_ratio_to_bcm": reference.load_ratio_to_bcm,
        "load_ohm": reference.load_ohm,
        "regime": observation.regime,
        "expected_regime": reference.regime,
        "observed_regime": observation.regime,
        "status": "pass" if all(gate["status"] == "pass" for gate in gates) else "fail",
        "metrics": {
            "mean_output_voltage_v": observation.mean_output_voltage_v,
            "minimum_inductor_current_a": observation.minimum_inductor_current_a,
            "inductor_current_peak_to_peak_a": observation.inductor_current_peak_to_peak_a,
            "output_voltage_peak_to_peak_v": observation.output_voltage_peak_to_peak_v,
            "conduction_fraction": observation.conduction_fraction,
            "fraction_by_mode": dict(observation.fraction_by_mode),
            "energy_balance_relative": observation.energy_balance_relative,
            "cycle_periodicity_normalized": periodicity,
            "mode_fraction_sum_error": fraction_sum_error,
        },
        "reference_metrics": {
            "mean_output_voltage_v": reference.mean_output_voltage_v,
            "minimum_inductor_current_a": reference.minimum_inductor_current_a,
            "inductor_current_peak_to_peak_a": reference.inductor_current_peak_to_peak_a,
            "output_voltage_peak_to_peak_v": reference.output_voltage_peak_to_peak_v,
            "conduction_fraction": reference.conduction_fraction,
            "fraction_by_mode": dict(reference.fraction_by_mode),
        },
        "events": {
            "observed_diode_zero_s": observation.diode_zero_s,
            "reference_diode_zero_s": reference.diode_zero_s,
        },
        "gates": gates,
    }


def _claim() -> dict[str, Any]:
    return {
        "classification": "verification",
        "validation_claim": False,
        "scope": "canonical single-switch ideal boost periodic CCM/BCM/DCM load sweep",
        "statement": (
            "Independent mathematical reference for the declared ideal equations; "
            "not hardware validation, cross-tool compatibility, or general DAE evidence."
        ),
    }


def _comparison_gate_contract() -> list[dict[str, Any]]:
    definitions = (
        ("regime", "exact_match", True),
        ("boundary_state_nonzero", "maximum_relative_error", BOUNDARY_STATE_RELATIVE_LIMIT),
        (
            "boundary_state_zero",
            "maximum_normalized_absolute_error",
            BOUNDARY_ZERO_NORMALIZED_LIMIT,
        ),
        (
            "natural_event_time",
            "absolute_event_time_error_over_Tsw",
            EVENT_TIME_RELATIVE_LIMIT,
        ),
        ("reverse_current", "maximum_forbidden_reverse_current_over_Ibase", REVERSE_CURRENT_NORMALIZED_LIMIT),
        ("cycle_mean_output", "relative_error", CYCLE_MEAN_RELATIVE_LIMIT),
        ("state_ripple", "maximum_relative_peak_to_peak_error", STATE_RIPPLE_RELATIVE_LIMIT),
        ("mode_fractions", "maximum_absolute_fraction_error", MODE_FRACTION_ABSOLUTE_LIMIT),
        ("event_semantics", "exact_mode_and_event_sequence", True),
        ("energy_balance", "relative_energy_balance_residual", ENERGY_BALANCE_RELATIVE_LIMIT),
        (
            "cycle_periodicity",
            "maximum_normalized_cycle_end_minus_start",
            CYCLE_PERIODICITY_NORMALIZED_LIMIT,
        ),
        ("unclassified_fraction", "absolute_fraction", UNCLASSIFIED_FRACTION_LIMIT),
        ("mode_fraction_sum", "absolute_error_from_unity", MODE_FRACTION_SUM_ERROR_LIMIT),
    )
    return [
        {
            "id": gate_id,
            "metric": metric,
            "operator": "==" if isinstance(limit, bool) else "<=",
            "limit": limit,
            "scope": "every load point",
        }
        for gate_id, metric, limit in definitions
    ]


def _mode_matrices(
    parameters: BoostBoundaryParameters, load_ohm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        on = np.asarray(
            [[0.0, 0.0], [0.0, -1.0 / (load_ohm * parameters.capacitance_f)]],
            dtype=float,
        )
        freewheel = np.asarray(
            [
                [0.0, -1.0 / parameters.inductance_h],
                [
                    1.0 / parameters.capacitance_f,
                    -1.0 / (load_ohm * parameters.capacitance_f),
                ],
            ],
            dtype=float,
        )
        offset = np.asarray([parameters.vin_v / parameters.inductance_h, 0.0])
    except (OverflowError, ZeroDivisionError) as exc:
        raise BoostBoundaryBenchmarkError("mode coefficients must be finite") from exc
    if not all(np.all(np.isfinite(value)) for value in (on, freewheel, offset)):
        raise BoostBoundaryBenchmarkError("mode coefficients must be finite")
    return on, freewheel, offset


def _ccm_periodic_state(
    parameters: BoostBoundaryParameters, load_ohm: float
) -> np.ndarray:
    on, freewheel, offset = _mode_matrices(parameters, load_ohm)
    on_transition, on_affine = _affine_map(on, offset, parameters.on_duration_s)
    off_transition, off_affine = _affine_map(
        freewheel, offset, parameters.off_duration_s
    )
    cycle_transition = off_transition @ on_transition
    cycle_affine = off_transition @ on_affine + off_affine
    try:
        state = np.linalg.solve(np.eye(2) - cycle_transition, cycle_affine)
    except np.linalg.LinAlgError as exc:
        raise BoostBoundaryBenchmarkError("periodic CCM map is singular") from exc
    if not np.all(np.isfinite(state)):
        raise BoostBoundaryBenchmarkError("periodic CCM state is non-finite")
    return np.asarray(state, dtype=float)


def _solve_ccm_or_bcm(
    parameters: BoostBoundaryParameters,
    load_ohm: float,
    ratio: float,
    regime: Literal["CCM", "BCM"],
) -> BoostBoundaryPoint:
    on, freewheel, offset = _mode_matrices(parameters, load_ohm)
    start = _ccm_periodic_state(parameters, load_ohm)
    if regime == "BCM":
        start[0] = 0.0
    switch, on_integral = _propagate_with_voltage_integral(
        on, offset, start, parameters.on_duration_s
    )
    end, off_integral = _propagate_with_voltage_integral(
        freewheel, offset, switch, parameters.off_duration_s
    )
    if regime == "BCM":
        end[0] = 0.0
    extrema = _cycle_extrema(
        parameters,
        on,
        freewheel,
        offset,
        start,
        switch,
        parameters.off_duration_s,
        None,
    )
    minimum = extrema[0][0]
    if regime == "BCM":
        if minimum < -parameters.current_base_a * 1e-12:
            raise BoostBoundaryBenchmarkError("BCM fixed point contains reverse current")
        minimum = 0.0
    fractions = {
        "on": parameters.duty,
        "freewheel": 1.0 - parameters.duty,
        "idle": 0.0,
        "unclassified": 0.0,
    }
    diode_zero_s = parameters.period_s if regime == "BCM" else None
    mode_sequence = ("on", "freewheel")
    event_sequence = _expected_event_sequence(regime)
    return _make_point(
        parameters,
        load_ohm,
        ratio,
        regime,
        start,
        switch,
        _state(end) if regime == "BCM" else None,
        end,
        diode_zero_s,
        (on_integral + off_integral) / parameters.period_s,
        minimum,
        extrema,
        fractions,
        mode_sequence,
        event_sequence,
    )


def _solve_dcm(
    parameters: BoostBoundaryParameters, load_ohm: float, ratio: float
) -> BoostBoundaryPoint:
    on, freewheel, offset = _mode_matrices(parameters, load_ohm)
    on_transition, on_affine = _affine_map(on, offset, parameters.on_duration_s)
    off_transition, off_affine = _affine_map(
        freewheel, offset, parameters.off_duration_s
    )

    # At the edge of the DCM domain, current first reaches zero exactly at Tsw.
    # The end current is affine in the scalar starting voltage, so this boundary is
    # obtained directly rather than with a finite voltage scan.
    voltage_coefficient = float((off_transition @ on_transition)[0, 1])
    current_constant = float((off_transition @ on_affine + off_affine)[0])
    if voltage_coefficient == 0.0:
        raise BoostBoundaryBenchmarkError("DCM voltage-domain boundary is singular")
    domain_voltage = -current_constant / voltage_coefficient
    if not math.isfinite(domain_voltage) or domain_voltage <= 0.0:
        raise BoostBoundaryBenchmarkError("DCM voltage-domain boundary is invalid")

    def cycle(output_voltage_v: float) -> tuple[float, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray] | None:
        start = np.asarray([0.0, output_voltage_v])
        switch = on_transition @ start + on_affine
        if switch[0] <= 0.0:
            return None
        zero_delay = _first_directed_current_zero(
            parameters, freewheel, offset, switch, parameters.off_duration_s
        )
        if zero_delay is None or zero_delay > parameters.off_duration_s:
            return None
        zero_state = _propagate(freewheel, offset, switch, zero_delay)
        zero_state[0] = 0.0
        idle_duration = parameters.off_duration_s - zero_delay
        end = np.asarray(
            [
                0.0,
                zero_state[1]
                * math.exp(-idle_duration / (load_ohm * parameters.capacitance_f)),
            ]
        )
        return end[1] - output_voltage_v, start, switch, zero_delay, zero_state, end

    lower = math.nextafter(domain_voltage, math.inf)
    evaluated = cycle(lower)
    if evaluated is None:
        for exponent in range(14, 5, -1):
            lower = domain_voltage + max(domain_voltage, parameters.vin_v) * 10.0**-exponent
            evaluated = cycle(lower)
            if evaluated is not None:
                break
    if evaluated is None:
        raise BoostBoundaryBenchmarkError("could not enter the periodic DCM voltage domain")
    lower_residual = float(evaluated[0])
    if not math.isfinite(lower_residual) or lower_residual <= 0.0:
        raise BoostBoundaryBenchmarkError(
            "periodic DCM voltage residual has no positive lower bracket"
        )

    upper = max(lower * 1.05, lower + parameters.vin_v * 0.05)
    upper_evaluated = cycle(upper)
    for _ in range(100):
        if upper_evaluated is not None and math.isfinite(upper_evaluated[0]) and upper_evaluated[0] < 0.0:
            break
        upper *= 1.5
        if not math.isfinite(upper):
            raise BoostBoundaryBenchmarkError("periodic DCM voltage bracket overflowed")
        upper_evaluated = cycle(upper)
    else:
        raise BoostBoundaryBenchmarkError("could not bracket periodic DCM voltage")

    def residual(voltage: float) -> float:
        candidate = cycle(voltage)
        if candidate is None:
            raise BoostBoundaryBenchmarkError("DCM periodic root left its valid mode domain")
        return float(candidate[0])

    root = float(
        brentq(
            residual,
            lower,
            upper,
            xtol=max(np.nextafter(0.0, 1.0), parameters.vin_v * 1e-14),
            rtol=1e-14,
            maxiter=200,
        )
    )
    final = cycle(root)
    if final is None:
        raise BoostBoundaryBenchmarkError("periodic DCM root is outside its mode domain")
    _residual, start, switch, zero_delay, zero_state, end = final
    if zero_delay >= parameters.off_duration_s:
        raise BoostBoundaryBenchmarkError("periodic DCM point has no finite idle interval")

    switch_integrated, on_integral = _propagate_with_voltage_integral(
        on, offset, start, parameters.on_duration_s
    )
    zero_integrated, off_integral = _propagate_with_voltage_integral(
        freewheel, offset, switch, zero_delay
    )
    zero_integrated[0] = 0.0
    if not np.allclose(switch_integrated, switch, rtol=1e-12, atol=1e-12):
        raise BoostBoundaryBenchmarkError("inconsistent ON propagation in DCM oracle")
    if not np.allclose(zero_integrated, zero_state, rtol=1e-12, atol=1e-12):
        raise BoostBoundaryBenchmarkError("inconsistent freewheel propagation in DCM oracle")
    idle_duration = parameters.off_duration_s - zero_delay
    decay = math.exp(-idle_duration / (load_ohm * parameters.capacitance_f))
    idle_integral = (
        zero_state[1]
        * load_ohm
        * parameters.capacitance_f
        * (1.0 - decay)
    )
    extrema = _cycle_extrema(
        parameters,
        on,
        freewheel,
        offset,
        start,
        switch,
        zero_delay,
        zero_state,
    )
    fractions = {
        "on": parameters.duty,
        "freewheel": zero_delay / parameters.period_s,
        "idle": idle_duration / parameters.period_s,
        "unclassified": 0.0,
    }
    return _make_point(
        parameters,
        load_ohm,
        ratio,
        "DCM",
        start,
        switch,
        _state(zero_state),
        end,
        parameters.on_duration_s + zero_delay,
        (on_integral + off_integral + idle_integral) / parameters.period_s,
        0.0,
        extrema,
        fractions,
        ("on", "freewheel", "idle"),
        _expected_event_sequence("DCM"),
    )


def _stationary_times(
    matrix: np.ndarray,
    offset: np.ndarray,
    state: np.ndarray,
    duration_s: float,
    state_index: int,
) -> list[float]:
    """Analytically enumerate derivative zeros for one state component."""

    derivative = matrix @ state + offset
    g0 = float(derivative[state_index])
    gdot0 = float((matrix @ derivative)[state_index])
    alpha = -0.5 * float(np.trace(matrix))
    natural_squared = float(np.linalg.det(matrix))
    if not all(math.isfinite(value) for value in (g0, gdot0, alpha, natural_squared)):
        raise BoostBoundaryBenchmarkError("analytical extrema coefficients must be finite")
    discriminant = natural_squared - alpha * alpha
    scale = max(abs(natural_squared), alpha * alpha, 1.0)
    roots: list[float] = []
    if discriminant > scale * 1e-14:
        beta = math.sqrt(discriminant)
        sine = (gdot0 + alpha * g0) / beta
        if g0 != 0.0 or sine != 0.0:
            phase = math.atan2(sine, g0)
            base = phase + math.pi / 2.0
            maximum_angle = beta * duration_s
            first_index = math.ceil(-base / math.pi)
            last_index = math.floor((maximum_angle - base) / math.pi)
            count = max(0, last_index - first_index + 1)
            if count > MAX_ANALYTIC_EXTREMA_PER_INTERVAL:
                raise BoostBoundaryBenchmarkError(
                    "analytical extrema count exceeds the bounded benchmark envelope"
                )
            roots.extend(
                (base + index * math.pi) / beta
                for index in range(first_index, last_index + 1)
            )
    elif discriminant < -scale * 1e-14:
        rate = math.sqrt(-discriminant)
        first_rate = -alpha - rate
        second_rate = -alpha + rate
        first_coefficient = (gdot0 - second_rate * g0) / (first_rate - second_rate)
        second_coefficient = g0 - first_coefficient
        if first_coefficient != 0.0:
            ratio = -second_coefficient / first_coefficient
            if ratio > 0.0:
                roots.append(math.log(ratio) / (first_rate - second_rate))
    else:
        linear = gdot0 + alpha * g0
        if linear != 0.0:
            roots.append(-g0 / linear)
    tolerance = duration_s * 1e-13
    return sorted(
        {
            min(duration_s, max(0.0, float(root)))
            for root in roots
            if math.isfinite(root) and -tolerance <= root <= duration_s + tolerance
        }
    )


def _first_directed_current_zero(
    parameters: BoostBoundaryParameters,
    matrix: np.ndarray,
    offset: np.ndarray,
    state: np.ndarray,
    duration_s: float,
) -> float | None:
    candidates = [0.0, *_stationary_times(matrix, offset, state, duration_s, 0), duration_s]
    candidates = sorted(set(candidates))

    def current(time_s: float) -> float:
        return float(_propagate(matrix, offset, state, time_s)[0])

    tolerance = parameters.current_base_a * 1e-12
    left_time = candidates[0]
    left_value = current(left_time)
    for right_time in candidates[1:]:
        right_value = current(right_time)
        if left_value > tolerance and right_value <= tolerance:
            if abs(right_value) <= tolerance:
                return right_time
            return float(
                brentq(
                    current,
                    left_time,
                    right_time,
                    xtol=max(np.nextafter(0.0, 1.0), parameters.period_s * 1e-15),
                    rtol=1e-14,
                    maxiter=200,
                )
            )
        left_time = right_time
        left_value = right_value
    return None


def _interval_extrema(
    matrix: np.ndarray,
    offset: np.ndarray,
    state: np.ndarray,
    duration_s: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    records: list[tuple[float, float]] = []
    for index in (0, 1):
        times = [0.0, duration_s, *_stationary_times(matrix, offset, state, duration_s, index)]
        values = [float(_propagate(matrix, offset, state, time_s)[index]) for time_s in times]
        if not all(math.isfinite(value) for value in values):
            raise BoostBoundaryBenchmarkError("state-extremum evaluation is non-finite")
        records.append((min(values), max(values)))
    return records[0], records[1]


def _cycle_extrema(
    parameters: BoostBoundaryParameters,
    on: np.ndarray,
    freewheel: np.ndarray,
    offset: np.ndarray,
    start: np.ndarray,
    switch: np.ndarray,
    freewheel_duration_s: float,
    zero_state: np.ndarray | None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    on_extrema = _interval_extrema(on, offset, start, parameters.on_duration_s)
    off_extrema = _interval_extrema(
        freewheel, offset, switch, freewheel_duration_s
    )
    current_min = min(on_extrema[0][0], off_extrema[0][0])
    current_max = max(on_extrema[0][1], off_extrema[0][1])
    voltage_min = min(on_extrema[1][0], off_extrema[1][0])
    voltage_max = max(on_extrema[1][1], off_extrema[1][1])
    if zero_state is not None:
        # Idle voltage is monotone.  The exact end voltage is the periodic start
        # voltage, so including both endpoints is sufficient and avoids duplicating R.
        current_min = min(current_min, 0.0)
        current_max = max(current_max, 0.0)
        voltage_min = min(voltage_min, float(start[1]), float(zero_state[1]))
        voltage_max = max(voltage_max, float(start[1]), float(zero_state[1]))
    return (current_min, current_max), (voltage_min, voltage_max)


def _make_point(
    parameters: BoostBoundaryParameters,
    load_ohm: float,
    ratio: float,
    regime: Regime,
    start: np.ndarray,
    switch: np.ndarray,
    diode_zero_state: BoostState | None,
    end: np.ndarray,
    diode_zero_s: float | None,
    mean_output_voltage_v: float,
    minimum_current_a: float,
    extrema: tuple[tuple[float, float], tuple[float, float]],
    fractions: dict[str, float],
    mode_sequence: tuple[str, ...],
    event_sequence: tuple[str, ...],
) -> BoostBoundaryPoint:
    periodicity = max(
        abs(float(end[0] - start[0])) / parameters.current_base_a,
        abs(float(end[1] - start[1])) / parameters.vin_v,
    )
    fraction_sum_error = abs(sum(fractions.values()) - 1.0)
    reverse_normalized = max(0.0, -minimum_current_a) / parameters.current_base_a
    current_ripple = extrema[0][1] - extrema[0][0]
    voltage_ripple = extrema[1][1] - extrema[1][0]
    gates = (
        _oracle_gate("periodicity", periodicity, 1e-10),
        _oracle_gate("unilateral_current", reverse_normalized, 1e-12),
        _oracle_gate("mode_fraction_sum", fraction_sum_error, 1e-12),
    )
    return BoostBoundaryPoint(
        point_id=_point_id(ratio),
        load_ratio_to_bcm=ratio,
        load_ohm=load_ohm,
        regime=regime,
        start_state=_state(start),
        switch_off_state=_state(switch),
        diode_zero_state=diode_zero_state,
        end_state=_state(end),
        switch_off_s=parameters.on_duration_s,
        diode_zero_s=diode_zero_s,
        mean_output_voltage_v=float(mean_output_voltage_v),
        minimum_inductor_current_a=float(minimum_current_a),
        inductor_current_peak_to_peak_a=float(current_ripple),
        output_voltage_peak_to_peak_v=float(voltage_ripple),
        fraction_by_mode=fractions,
        mode_sequence=mode_sequence,
        event_sequence=event_sequence,
        periodicity_residual_normalized=periodicity,
        oracle_gates=gates,
    )


def _require_oracle_health(point: BoostBoundaryPoint) -> None:
    failed = [gate["id"] for gate in point.oracle_gates if gate["status"] != "pass"]
    if failed:
        raise BoostBoundaryBenchmarkError(
            f"independent oracle health gates failed for {point.point_id}: {failed}"
        )
    idle = point.fraction_by_mode["idle"]
    if point.regime == "CCM" and (
        point.minimum_inductor_current_a <= 0.0
        or point.diode_zero_s is not None
        or idle != 0.0
    ):
        raise BoostBoundaryBenchmarkError("CCM oracle point violates its regime domain")
    if point.regime == "BCM" and (
        point.diode_zero_s is None
        or abs(point.diode_zero_s - point.switch_off_s / point.fraction_by_mode["on"])
        > _point_period(point) * BCM_COINCIDENCE_FRACTION_TOLERANCE
        or idle > BCM_COINCIDENCE_FRACTION_TOLERANCE
    ):
        raise BoostBoundaryBenchmarkError("BCM oracle point violates its coincidence domain")
    if point.regime == "DCM" and (
        point.diode_zero_s is None or idle <= BCM_COINCIDENCE_FRACTION_TOLERANCE
    ):
        raise BoostBoundaryBenchmarkError("DCM oracle point lacks a finite idle interval")
    if (
        not math.isfinite(point.inductor_current_peak_to_peak_a)
        or point.inductor_current_peak_to_peak_a <= 0.0
        or not math.isfinite(point.output_voltage_peak_to_peak_v)
        or point.output_voltage_peak_to_peak_v <= 0.0
    ):
        raise BoostBoundaryBenchmarkError("oracle state ripple must be finite and positive")


def _point_period(point: BoostBoundaryPoint) -> float:
    return point.switch_off_s / point.fraction_by_mode["on"]


def _expected_event_sequence(regime: Regime) -> tuple[str, ...]:
    beginning = (
        "scheduled_gate:control:1>0@switch_off",
        "switch_manager:diode:0>1@switch_off",
    )
    if regime == "CCM":
        return (
            *beginning,
            "scheduled_gate:control:0>1@cycle_end",
            "switch_manager:diode:1>0@cycle_end",
        )
    if regime == "BCM":
        return (
            *beginning,
            "natural_commutation:diode:1>0@diode_zero",
            "scheduled_gate:control:0>1@cycle_end",
        )
    return (
        *beginning,
        "natural_commutation:diode:1>0@diode_zero",
        "scheduled_gate:control:0>1@cycle_end",
    )


def _observed_event_sequence(
    result: Any,
    fixture: _Fixture,
    *,
    start_s: float,
    reference: BoostBoundaryPoint,
    parameters: BoostBoundaryParameters,
) -> tuple[str, ...]:
    tokens: list[str] = []
    switch_time = start_s + parameters.on_duration_s
    end_time = start_s + parameters.period_s
    zero_time = (
        None if reference.diode_zero_s is None else start_s + reference.diode_zero_s
    )
    for event in getattr(result, "events", ()):
        time_s = float(getattr(event, "time", math.nan))
        kind = str(getattr(event, "kind", "missing"))
        target = getattr(event, "target", None)
        role = (
            "control"
            if target == fixture.pwm.id
            else "diode"
            if target == fixture.diode.id
            else f"unexpected-target:{target}"
        )
        previous = getattr(event, "previous", None)
        current = getattr(event, "current", None)
        transition = f"{int(previous)}>{int(current)}" if isinstance(previous, bool) and isinstance(current, bool) else "invalid"
        if math.isfinite(time_s) and abs(time_s - switch_time) <= parameters.period_s * 1e-10:
            location = "switch_off"
        elif (
            kind == "natural_commutation"
            and zero_time is not None
            and math.isfinite(time_s)
            and abs(time_s - zero_time)
            <= parameters.period_s * EVENT_TIME_RELATIVE_LIMIT
        ):
            location = "diode_zero"
        elif math.isfinite(time_s) and abs(time_s - end_time) <= parameters.period_s * 1e-10:
            location = "cycle_end"
        else:
            location = "unexpected-time"
        tokens.append(f"{kind}:{role}:{transition}@{location}")
    return tuple(tokens)


def _validate_observation(observation: BoostBoundaryObservation) -> None:
    if not isinstance(observation.point_id, str) or not observation.point_id:
        raise BoostBoundaryBenchmarkError("observation point_id must be a non-empty string")
    if observation.regime not in {"CCM", "BCM", "DCM"}:
        raise BoostBoundaryBenchmarkError("observation regime must be CCM, BCM, or DCM")
    for name, value in {
        "load_ratio_to_bcm": observation.load_ratio_to_bcm,
        "load_ohm": observation.load_ohm,
    }.items():
        if not math.isfinite(value) or value <= 0.0:
            raise BoostBoundaryBenchmarkError(f"observation {name} must be finite and positive")
    states = [observation.start_state, observation.switch_off_state, observation.end_state]
    if observation.diode_zero_state is not None:
        states.append(observation.diode_zero_state)
    for state in states:
        if not math.isfinite(state.inductor_current_a) or not math.isfinite(state.output_voltage_v):
            raise BoostBoundaryBenchmarkError("observation states must be finite")
    if observation.diode_zero_s is not None and (
        not math.isfinite(observation.diode_zero_s) or observation.diode_zero_s < 0.0
    ):
        raise BoostBoundaryBenchmarkError(
            "observation diode_zero_s must be finite and nonnegative when present"
        )
    for name, value in {
        "mean_output_voltage_v": observation.mean_output_voltage_v,
        "minimum_inductor_current_a": observation.minimum_inductor_current_a,
        "inductor_current_peak_to_peak_a": observation.inductor_current_peak_to_peak_a,
        "output_voltage_peak_to_peak_v": observation.output_voltage_peak_to_peak_v,
        "energy_balance_relative": observation.energy_balance_relative,
    }.items():
        if not math.isfinite(value):
            raise BoostBoundaryBenchmarkError(f"observation {name} must be finite")
    if observation.inductor_current_peak_to_peak_a < 0.0 or observation.output_voltage_peak_to_peak_v < 0.0:
        raise BoostBoundaryBenchmarkError("observation state ripple must be nonnegative")
    if observation.energy_balance_relative < 0.0:
        raise BoostBoundaryBenchmarkError("observation energy_balance_relative must be nonnegative")
    expected_modes = {"on", "freewheel", "idle", "unclassified"}
    if set(observation.fraction_by_mode) != expected_modes:
        raise BoostBoundaryBenchmarkError(
            "observation fraction_by_mode must contain on/freewheel/idle/unclassified"
        )
    for name, value in observation.fraction_by_mode.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise BoostBoundaryBenchmarkError(
                f"observation mode fraction {name!r} must be finite and within [0,1]"
            )
    if not all(isinstance(value, str) for value in observation.mode_sequence):
        raise BoostBoundaryBenchmarkError("observation mode_sequence must contain strings")
    if not all(isinstance(value, str) for value in observation.event_sequence):
        raise BoostBoundaryBenchmarkError("observation event_sequence must contain strings")


def _require_matching_parameters(
    expected: BoostBoundaryParameters,
    actual: BoostBoundaryParameters,
    *,
    context: str,
) -> None:
    for field in (
        "vin_v",
        "inductance_h",
        "capacitance_f",
        "switching_frequency_hz",
        "duty",
    ):
        expected_value = float(getattr(expected, field))
        actual_value = float(getattr(actual, field))
        if not math.isclose(expected_value, actual_value, rel_tol=2e-13, abs_tol=1e-18):
            raise BoostBoundaryBenchmarkError(
                f"parameter mismatch for {context}: {field} expected "
                f"{expected_value:.17g}, got {actual_value:.17g}"
            )


def _point_id(ratio: float) -> str:
    return f"load-ratio-{format(ratio, '.12g').replace('.', 'p')}"


def _state(vector: np.ndarray) -> BoostState:
    return BoostState(float(vector[0]), float(vector[1]))


def _state_from_mapping(values: Any, inductor_id: str, capacitor_id: str) -> BoostState:
    if not isinstance(values, dict) or inductor_id not in values or capacitor_id not in values:
        raise BoostBoundaryBenchmarkError("accepted boundary state is incomplete")
    state = BoostState(float(values[inductor_id]), float(values[capacitor_id]))
    if not math.isfinite(state.inductor_current_a) or not math.isfinite(state.output_voltage_v):
        raise BoostBoundaryBenchmarkError("accepted boundary state must be finite")
    return state
