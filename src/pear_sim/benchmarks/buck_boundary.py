# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Independent periodic CCM/BCM/DCM reference for a canonical ideal buck.

The oracle in this module derives the three affine modes locally.  It does not use
PEARSim's circuit compiler, transient solver, event manager, or mode classifier.  A
small adapter can project an already completed PEARSim result into the benchmark's
observation contract; that adapter does not participate in generating the reference.

This is mathematical verification evidence for one explicitly recognized ideal
topology.  It is not a hardware-validation or general converter-correctness claim.
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
    BuckBoundaryBenchmarkError,
    BuckBoundaryParameters,
    extract_buck_boundary_parameters,
    extract_buck_fixture as _extract_fixture,
)
from pear_sim.benchmarks._shared import (
    append_state_error as _append_state_error,
    comparison_gate as _comparison_gate,
)


BUCK_BOUNDARY_ORACLE_SCHEMA = "pear-sim/benchmark-oracle/ideal-buck-boundary/v0.1"
BUCK_BOUNDARY_RESULT_SCHEMA = "pear-sim/benchmark-result/ideal-buck-boundary/v0.1"
BUCK_BOUNDARY_KIND = "independent-periodic-affine-buck-boundary-verification"
DEFAULT_LOAD_RATIOS = (0.25, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 200.0)

# A numerically localized zero-current event may precede the scheduled rising edge by
# a tiny positive sliver.  BCM is therefore a dimensionless coincidence band, not a
# requirement that an idle interval be represented by literal floating-point zero.
BCM_COINCIDENCE_FRACTION_TOLERANCE = 1e-8
EVENT_TIME_RELATIVE_LIMIT = 5e-4
REVERSE_CURRENT_NORMALIZED_LIMIT = 1e-8
BOUNDARY_ZERO_NORMALIZED_LIMIT = 1e-8
BOUNDARY_STATE_RELATIVE_LIMIT = 1e-5
CYCLE_MEAN_RELATIVE_LIMIT = 1e-3
MODE_FRACTION_ABSOLUTE_LIMIT = 1e-3
ENERGY_BALANCE_RELATIVE_LIMIT = 1e-4
CYCLE_PERIODICITY_NORMALIZED_LIMIT = 1e-8
UNCLASSIFIED_FRACTION_LIMIT = 1e-10
MODE_FRACTION_SUM_ERROR_LIMIT = 1e-10
MAX_ANALYTIC_EXTREMA_PER_INTERVAL = 10_000
_REGIME_BOUNDARY_RELATIVE_TOLERANCE = 2e-12

Regime = Literal["CCM", "BCM", "DCM"]


@dataclass(frozen=True)
class BuckState:
    inductor_current_a: float
    output_voltage_v: float

    def as_vector(self) -> np.ndarray:
        return np.asarray([self.inductor_current_a, self.output_voltage_v], dtype=float)


@dataclass(frozen=True)
class BuckBoundaryPoint:
    point_id: str
    load_ratio_to_bcm: float
    load_ohm: float
    regime: Regime
    start_state: BuckState
    switch_off_state: BuckState
    diode_zero_state: BuckState | None
    end_state: BuckState
    switch_off_s: float
    diode_zero_s: float | None
    mean_output_voltage_v: float
    minimum_inductor_current_a: float
    fraction_by_mode: dict[str, float]
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
            },
            "metrics": {
                "mean_output_voltage_v": self.mean_output_voltage_v,
                "minimum_inductor_current_a": self.minimum_inductor_current_a,
                "conduction_fraction": self.conduction_fraction,
                "fraction_by_mode": dict(self.fraction_by_mode),
                "periodicity_residual_normalized": self.periodicity_residual_normalized,
            },
            "oracle_gates": [dict(gate) for gate in self.oracle_gates],
        }


@dataclass(frozen=True)
class BuckBoundarySweep:
    parameters: BuckBoundaryParameters
    bcm_load_ohm: float
    load_ratios: tuple[float, ...]
    points: tuple[BuckBoundaryPoint, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": BUCK_BOUNDARY_ORACLE_SCHEMA,
            "kind": BUCK_BOUNDARY_KIND,
            "claim": _claim(),
            "method": {
                "periodic_ccm": (
                    "fixed point of independently derived ON/OFF affine matrix-exponential map"
                ),
                "bcm_boundary": (
                    "brentq root of periodic CCM cycle-start inductor current versus load"
                ),
                "periodic_dcm": (
                    "scalar periodic-voltage root with an analytical first zero-current "
                    "candidate and independently evaluated ON/freewheel/idle maps"
                ),
                "integrals": "augmented matrix exponential per affine interval",
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
class BuckBoundaryObservation:
    """Solver-independent projection of one completed simulation result."""

    point_id: str
    parameters: BuckBoundaryParameters
    load_ratio_to_bcm: float
    load_ohm: float
    regime: Regime
    start_state: BuckState
    switch_off_state: BuckState
    diode_zero_state: BuckState | None
    end_state: BuckState
    diode_zero_s: float | None
    mean_output_voltage_v: float
    minimum_inductor_current_a: float
    fraction_by_mode: dict[str, float]
    energy_balance_relative: float

    @property
    def conduction_fraction(self) -> float:
        return self.fraction_by_mode["on"] + self.fraction_by_mode["freewheel"]


def find_bcm_load(parameters: BuckBoundaryParameters) -> float:
    """Find the exact numerical load where the periodic CCM minimum reaches zero."""

    textbook_seed = (
        2.0 * parameters.inductance_h * parameters.switching_frequency_hz / (1.0 - parameters.duty)
    )
    if not math.isfinite(textbook_seed) or textbook_seed <= 0.0:
        raise BuckBoundaryBenchmarkError("textbook boundary seed is not finite and positive")

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
        raise BuckBoundaryBenchmarkError("Could not bracket the CCM/BCM load boundary")

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
        raise BuckBoundaryBenchmarkError("BCM load root is not finite and positive")
    return root


def solve_periodic_buck(
    parameters: BuckBoundaryParameters,
    load_ohm: float,
    *,
    bcm_load_ohm: float | None = None,
    load_ratio_to_bcm: float | None = None,
) -> BuckBoundaryPoint:
    """Solve one exact periodic operating point in the recognized ideal topology."""

    if not math.isfinite(load_ohm) or load_ohm <= 0.0:
        raise BuckBoundaryBenchmarkError("load_ohm must be finite and positive")
    computed_boundary = find_bcm_load(parameters)
    boundary = computed_boundary if bcm_load_ohm is None else bcm_load_ohm
    if not math.isfinite(boundary) or boundary <= 0.0:
        raise BuckBoundaryBenchmarkError("bcm_load_ohm must be finite and positive")
    if not math.isclose(boundary, computed_boundary, rel_tol=2e-12, abs_tol=1e-15):
        raise BuckBoundaryBenchmarkError(
            "supplied bcm_load_ohm does not match the independently evaluated boundary"
        )
    ratio = load_ohm / boundary if load_ratio_to_bcm is None else load_ratio_to_bcm
    if not math.isfinite(ratio) or ratio <= 0.0:
        raise BuckBoundaryBenchmarkError("load_ratio_to_bcm must be finite and positive")
    if not math.isclose(load_ohm / boundary, ratio, rel_tol=2e-13, abs_tol=1e-15):
        raise BuckBoundaryBenchmarkError("load ratio and load value are inconsistent")

    boundary_band = max(boundary * _REGIME_BOUNDARY_RELATIVE_TOLERANCE, 1e-15)
    if load_ohm < boundary - boundary_band:
        regime: Regime = "CCM"
        point = _solve_ccm_or_bcm(parameters, load_ohm, ratio, regime)
    elif load_ohm > boundary + boundary_band:
        regime = "DCM"
        point = _solve_dcm(parameters, load_ohm, ratio)
    else:
        if ratio != 1.0:
            raise BuckBoundaryBenchmarkError(
                "a load inside the numerical BCM coincidence band must use exact ratio 1.0"
            )
        regime = "BCM"
        point = _solve_ccm_or_bcm(parameters, boundary, 1.0, regime)
    _require_oracle_health(point)
    return point


def run_buck_boundary_sweep(
    parameters: BuckBoundaryParameters,
    load_ratios: Sequence[float] = DEFAULT_LOAD_RATIOS,
) -> BuckBoundarySweep:
    """Evaluate a deterministic, ordered sweep around the independently found boundary."""

    ratios = tuple(float(value) for value in load_ratios)
    if not ratios:
        raise BuckBoundaryBenchmarkError("load_ratios must not be empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in ratios):
        raise BuckBoundaryBenchmarkError("every load ratio must be finite and positive")
    if any(right <= left for left, right in zip(ratios[:-1], ratios[1:], strict=True)):
        raise BuckBoundaryBenchmarkError("load_ratios must be strictly increasing and unique")
    if not any(math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-15) for value in ratios):
        raise BuckBoundaryBenchmarkError("load_ratios must contain the exact BCM ratio 1.0")
    if any(
        value != 1.0 and abs(value - 1.0) <= _REGIME_BOUNDARY_RELATIVE_TOLERANCE for value in ratios
    ):
        raise BuckBoundaryBenchmarkError(
            "load ratios inside the numerical BCM coincidence band are ambiguous; "
            "retain only exact ratio 1.0"
        )
    point_ids = [_point_id(ratio) for ratio in ratios]
    if len(set(point_ids)) != len(point_ids):
        raise BuckBoundaryBenchmarkError(
            "load ratios are not distinguishable by the versioned point-id format"
        )

    boundary = find_bcm_load(parameters)
    points = tuple(
        solve_periodic_buck(
            parameters,
            boundary * ratio,
            bcm_load_ohm=boundary,
            load_ratio_to_bcm=ratio,
        )
        for ratio in ratios
    )
    return BuckBoundarySweep(parameters, boundary, ratios, points)


def observe_simulation_result(
    result: Any,
    reference: BuckBoundaryPoint,
    *,
    parameters: BuckBoundaryParameters,
    bcm_load_ohm: float,
) -> BuckBoundaryObservation:
    """Project a completed result without using it to generate the oracle.

    The function intentionally relies only on the public result data contract and
    accepted interval records.  It imports no solver, circuit, or event code.
    """

    model = getattr(result, "model", None)
    if not isinstance(model, Model):
        raise BuckBoundaryBenchmarkError("observation requires a result with a PEARSim Model")
    fixture = _extract_fixture(model)
    observed_parameters = extract_buck_boundary_parameters(model)
    _require_matching_parameters(parameters, observed_parameters, context="observed model")
    if not math.isclose(
        parameters.period_s,
        reference.switch_off_s / parameters.duty,
        rel_tol=2e-13,
        abs_tol=1e-18,
    ):
        raise BuckBoundaryBenchmarkError("reference switching period differs from parameters")
    if not math.isclose(float(fixture.load.value), reference.load_ohm, rel_tol=2e-12):
        raise BuckBoundaryBenchmarkError("observed load differs from reference")

    intervals = list(getattr(result, "mode_intervals", ()))
    if not intervals:
        raise BuckBoundaryBenchmarkError("observed result has no accepted mode intervals")
    start_s = float(model.transient.start)
    stop_s = float(model.transient.stop)
    period = parameters.period_s
    if not math.isclose(stop_s - start_s, period, rel_tol=0.0, abs_tol=period * 1e-10):
        raise BuckBoundaryBenchmarkError("observation must span exactly one switching period")

    durations = {"on": 0.0, "freewheel": 0.0, "idle": 0.0, "unclassified": 0.0}
    covered = 0.0
    minimum_current = math.inf
    coverage_tolerance = period * 1e-10
    previous_end = start_s
    for interval in intervals:
        raw_start = float(interval["start_s"])
        raw_end = float(interval["end_s"])
        if not math.isfinite(raw_start) or not math.isfinite(raw_end):
            raise BuckBoundaryBenchmarkError("accepted interval bounds must be finite")
        interval_start = max(start_s, raw_start)
        interval_end = min(stop_s, raw_end)
        if interval_start > previous_end + coverage_tolerance:
            raise BuckBoundaryBenchmarkError("accepted mode intervals contain a gap")
        if interval_start < previous_end - coverage_tolerance:
            raise BuckBoundaryBenchmarkError("accepted mode intervals overlap or are out of order")
        if abs(interval_start - previous_end) <= coverage_tolerance:
            interval_start = previous_end
        duration = interval_end - interval_start
        if duration <= 0.0:
            raise BuckBoundaryBenchmarkError(
                "accepted mode intervals must be ordered and strictly positive"
            )
        covered += duration
        previous_end = interval_end
        gates = interval.get("gates", {})
        diodes = interval.get("diodes", {})
        gate_on = gates.get(fixture.pwm.id)
        diode_on = diodes.get(fixture.diode.id)
        constrained = fixture.inductor.id in interval.get("constrained_inductors", ())
        if gate_on is True and diode_on is False:
            mode = "on"
        elif gate_on is False and diode_on is True:
            mode = "freewheel"
        elif gate_on is False and diode_on is False and constrained:
            mode = "idle"
        else:
            mode = "unclassified"
        durations[mode] += duration
        extrema = interval.get("state_extrema", {}).get(fixture.inductor.id)
        if not isinstance(extrema, dict) or "minimum" not in extrema:
            raise BuckBoundaryBenchmarkError("accepted interval lacks inductor extrema")
        minimum_current = min(minimum_current, float(extrema["minimum"]))
    if not math.isclose(previous_end, stop_s, rel_tol=0.0, abs_tol=coverage_tolerance):
        raise BuckBoundaryBenchmarkError("accepted mode intervals do not reach cycle end")
    if not math.isclose(covered, period, rel_tol=1e-10, abs_tol=coverage_tolerance):
        raise BuckBoundaryBenchmarkError("accepted intervals do not cover one complete cycle")
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
        raise BuckBoundaryBenchmarkError("more than one diode turn-off occurred in one cycle")
    diode_zero_s = None if not natural_turnoffs else float(natural_turnoffs[0].time) - start_s

    idle_fraction = fractions["idle"]
    if idle_fraction > BCM_COINCIDENCE_FRACTION_TOLERANCE:
        regime: Regime = "DCM"
    elif (
        diode_zero_s is not None
        and abs(diode_zero_s - period) / period <= EVENT_TIME_RELATIVE_LIMIT
    ):
        regime = "BCM"
    elif minimum_current / parameters.current_base_a <= REVERSE_CURRENT_NORMALIZED_LIMIT:
        regime = "BCM"
    else:
        regime = "CCM"

    first_interval = intervals[0]
    last_interval = intervals[-1]
    switch_interval = min(
        intervals,
        key=lambda interval: abs(float(interval["end_s"]) - (start_s + parameters.on_duration_s)),
    )
    if (
        abs(float(switch_interval["end_s"]) - (start_s + parameters.on_duration_s))
        > period * EVENT_TIME_RELATIVE_LIMIT
    ):
        raise BuckBoundaryBenchmarkError("could not identify the commanded switch-off boundary")

    diode_zero_state: BuckState | None = None
    if diode_zero_s is not None:
        zero_interval = min(
            intervals,
            key=lambda interval: abs(float(interval["end_s"]) - (start_s + diode_zero_s)),
        )
        if (
            abs(float(zero_interval["end_s"]) - (start_s + diode_zero_s))
            > period * EVENT_TIME_RELATIVE_LIMIT
        ):
            raise BuckBoundaryBenchmarkError("could not identify the natural-event boundary state")
        diode_zero_state = _state_from_mapping(
            zero_interval["state_end"], fixture.inductor.id, fixture.capacitor.id
        )

    output_expression = f"v({fixture.inductor.n})"
    metrics = result.metrics()
    if output_expression not in metrics:
        raise BuckBoundaryBenchmarkError(
            f"observation requires requested output {output_expression!r}"
        )
    audit = getattr(result, "physics_audit", {}).get("full")
    if not isinstance(audit, dict) or "balance_residual_relative" not in audit:
        raise BuckBoundaryBenchmarkError("observation requires the full-cycle physics audit")

    observation = BuckBoundaryObservation(
        point_id=reference.point_id,
        parameters=observed_parameters,
        load_ratio_to_bcm=float(fixture.load.value) / bcm_load_ohm,
        load_ohm=float(fixture.load.value),
        regime=regime,
        start_state=_state_from_mapping(
            first_interval["state_start"], fixture.inductor.id, fixture.capacitor.id
        ),
        switch_off_state=_state_from_mapping(
            switch_interval["state_end"], fixture.inductor.id, fixture.capacitor.id
        ),
        diode_zero_state=diode_zero_state,
        end_state=_state_from_mapping(
            last_interval["state_end"], fixture.inductor.id, fixture.capacitor.id
        ),
        diode_zero_s=diode_zero_s,
        mean_output_voltage_v=float(metrics[output_expression]["mean"]),
        minimum_inductor_current_a=minimum_current,
        fraction_by_mode=fractions,
        energy_balance_relative=float(audit["balance_residual_relative"]),
    )
    _validate_observation(observation)
    if not math.isclose(
        observation.load_ratio_to_bcm,
        reference.load_ratio_to_bcm,
        rel_tol=2e-12,
        abs_tol=1e-15,
    ):
        raise BuckBoundaryBenchmarkError("observed load ratio differs from reference")
    return observation


def compare_buck_boundary_sweep(
    reference: BuckBoundarySweep,
    observations: Sequence[BuckBoundaryObservation],
) -> dict[str, Any]:
    """Apply the versioned, explicit gate contract to a complete observed sweep."""

    observed_by_id: dict[str, BuckBoundaryObservation] = {}
    for observation in observations:
        if observation.point_id in observed_by_id:
            raise BuckBoundaryBenchmarkError(f"duplicate observation {observation.point_id!r}")
        observed_by_id[observation.point_id] = observation
    expected_ids = [point.point_id for point in reference.points]
    if set(observed_by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(observed_by_id))
        extra = sorted(set(observed_by_id) - set(expected_ids))
        raise BuckBoundaryBenchmarkError(
            f"observation set does not match sweep; missing={missing}, extra={extra}"
        )

    point_results = [
        compare_buck_boundary_point(
            reference.parameters,
            point,
            observed_by_id[point.point_id],
        )
        for point in reference.points
    ]
    failed = sum(gate["status"] == "fail" for point in point_results for gate in point["gates"])
    return {
        "schema": BUCK_BOUNDARY_RESULT_SCHEMA,
        "kind": "ideal-buck-boundary-sweep-comparison",
        "claim": _claim(),
        "oracle_schema": BUCK_BOUNDARY_ORACLE_SCHEMA,
        "parameters_si": reference.parameters.as_dict(),
        "boundary_load_ohm": reference.bcm_load_ohm,
        "gate_contract": _comparison_gate_contract(),
        "summary": {
            "status": "pass" if failed == 0 else "fail",
            "point_count": len(point_results),
            "gate_count": sum(len(point["gates"]) for point in point_results),
            "failed_gate_count": failed,
            "not_run_gate_count": 0,
        },
        "points": point_results,
    }


def compare_buck_boundary_point(
    parameters: BuckBoundaryParameters,
    reference: BuckBoundaryPoint,
    observation: BuckBoundaryObservation,
) -> dict[str, Any]:
    """Evaluate one point so a runner can preserve partial and failed sweeps.

    Missing or failed runs belong to the runner's ``not-run``/structured-error
    contract and are intentionally not fabricated here.  This helper evaluates a
    completed observation using exactly the same gates as the all-points function.
    """

    return _compare_point(parameters, reference, observation)


def _claim() -> dict[str, Any]:
    return {
        "classification": "verification",
        "validation_claim": False,
        "scope": "canonical single-switch ideal buck periodic CCM/BCM/DCM load sweep",
        "statement": (
            "Independent mathematical reference for the declared ideal equations; "
            "not hardware validation, cross-tool compatibility, or general DAE evidence."
        ),
    }


def _comparison_gate_contract() -> list[dict[str, Any]]:
    return [
        {
            "id": "regime",
            "metric": "exact_match",
            "limit": True,
            "scope": "every load point",
        },
        {
            "id": "boundary_state_nonzero",
            "metric": "maximum_relative_error",
            "operator": "<=",
            "limit": BOUNDARY_STATE_RELATIVE_LIMIT,
            "scope": "all nonzero reference states at switching and natural boundaries",
        },
        {
            "id": "boundary_state_zero",
            "metric": "maximum_normalized_absolute_error",
            "operator": "<=",
            "limit": BOUNDARY_ZERO_NORMALIZED_LIMIT,
            "scope": "all zero-current boundary states",
        },
        {
            "id": "natural_event_time",
            "metric": "absolute_event_time_error_over_Tsw",
            "operator": "<=",
            "limit": EVENT_TIME_RELATIVE_LIMIT,
            "scope": "BCM and DCM diode turn-off; exact absence in CCM",
        },
        {
            "id": "reverse_current",
            "metric": "maximum_forbidden_reverse_current_over_Ibase",
            "operator": "<=",
            "limit": REVERSE_CURRENT_NORMALIZED_LIMIT,
            "scope": "every load point",
        },
        {
            "id": "cycle_mean_output",
            "metric": "relative_error",
            "operator": "<=",
            "limit": CYCLE_MEAN_RELATIVE_LIMIT,
            "scope": "every load point",
        },
        {
            "id": "mode_fractions",
            "metric": "maximum_absolute_fraction_error",
            "operator": "<=",
            "limit": MODE_FRACTION_ABSOLUTE_LIMIT,
            "scope": "on/freewheel/idle/unclassified at every load point",
        },
        {
            "id": "energy_balance",
            "metric": "relative_energy_balance_residual",
            "operator": "<=",
            "limit": ENERGY_BALANCE_RELATIVE_LIMIT,
            "scope": "every load point",
        },
        {
            "id": "cycle_periodicity",
            "metric": "maximum_normalized_cycle_end_minus_start",
            "operator": "<=",
            "limit": CYCLE_PERIODICITY_NORMALIZED_LIMIT,
            "scope": "inductor current and output voltage at every load point",
        },
        {
            "id": "unclassified_fraction",
            "metric": "absolute_fraction",
            "operator": "<=",
            "limit": UNCLASSIFIED_FRACTION_LIMIT,
            "scope": "every load point",
        },
        {
            "id": "mode_fraction_sum",
            "metric": "absolute_error_from_unity",
            "operator": "<=",
            "limit": MODE_FRACTION_SUM_ERROR_LIMIT,
            "scope": "every load point",
        },
    ]


def _active_matrix(parameters: BuckBoundaryParameters, load_ohm: float) -> np.ndarray:
    try:
        matrix = np.asarray(
            [
                [0.0, -1.0 / parameters.inductance_h],
                [
                    1.0 / parameters.capacitance_f,
                    -1.0 / (load_ohm * parameters.capacitance_f),
                ],
            ],
            dtype=float,
        )
    except (OverflowError, ZeroDivisionError) as exc:
        raise BuckBoundaryBenchmarkError("active-mode coefficients must be finite") from exc
    if not np.all(np.isfinite(matrix)):
        raise BuckBoundaryBenchmarkError("active-mode coefficients must be finite")
    return matrix


def _ccm_periodic_state(parameters: BuckBoundaryParameters, load_ohm: float) -> np.ndarray:
    matrix = _active_matrix(parameters, load_ohm)
    on_offset = np.asarray([parameters.vin_v / parameters.inductance_h, 0.0])
    off_offset = np.zeros(2)
    on_transition, on_affine = _affine_map(matrix, on_offset, parameters.on_duration_s)
    off_transition, off_affine = _affine_map(matrix, off_offset, parameters.off_duration_s)
    cycle_transition = off_transition @ on_transition
    cycle_affine = off_transition @ on_affine + off_affine
    try:
        state = np.linalg.solve(np.eye(2) - cycle_transition, cycle_affine)
    except np.linalg.LinAlgError as exc:
        raise BuckBoundaryBenchmarkError("periodic CCM map is singular") from exc
    if not np.all(np.isfinite(state)):
        raise BuckBoundaryBenchmarkError("periodic CCM state is non-finite")
    return np.asarray(state, dtype=float)


def _solve_ccm_or_bcm(
    parameters: BuckBoundaryParameters,
    load_ohm: float,
    ratio: float,
    regime: Literal["CCM", "BCM"],
) -> BuckBoundaryPoint:
    matrix = _active_matrix(parameters, load_ohm)
    on_offset = np.asarray([parameters.vin_v / parameters.inductance_h, 0.0])
    off_offset = np.zeros(2)
    start = _ccm_periodic_state(parameters, load_ohm)
    if regime == "BCM":
        start[0] = 0.0
    switch, on_integral = _propagate_with_voltage_integral(
        matrix, on_offset, start, parameters.on_duration_s
    )
    end, off_integral = _propagate_with_voltage_integral(
        matrix, off_offset, switch, parameters.off_duration_s
    )
    if regime == "BCM":
        end[0] = 0.0
    raw_minimum = min(
        _minimum_current(matrix, on_offset, start, parameters.on_duration_s),
        _minimum_current(matrix, off_offset, switch, parameters.off_duration_s),
        float(start[0]),
        float(end[0]),
    )
    if regime == "BCM":
        if raw_minimum < -parameters.current_base_a * 1e-12:
            raise BuckBoundaryBenchmarkError(
                "BCM fixed point contains reverse current inside an active interval"
            )
        minimum = 0.0
    else:
        minimum = raw_minimum
    fractions = {
        "on": parameters.duty,
        "freewheel": 1.0 - parameters.duty,
        "idle": 0.0,
        "unclassified": 0.0,
    }
    diode_zero_s = parameters.period_s if regime == "BCM" else None
    diode_zero_state = _state(end) if regime == "BCM" else None
    return _make_point(
        parameters,
        load_ohm,
        ratio,
        regime,
        start,
        switch,
        diode_zero_state,
        end,
        diode_zero_s,
        (on_integral + off_integral) / parameters.period_s,
        minimum,
        fractions,
    )


def _solve_dcm(
    parameters: BuckBoundaryParameters,
    load_ohm: float,
    ratio: float,
) -> BuckBoundaryPoint:
    matrix = _active_matrix(parameters, load_ohm)
    on_offset = np.asarray([parameters.vin_v / parameters.inductance_h, 0.0])
    off_offset = np.zeros(2)
    on_transition, on_affine = _affine_map(matrix, on_offset, parameters.on_duration_s)

    def cycle(output_voltage_v: float) -> tuple[float, np.ndarray, float, np.ndarray] | None:
        start = np.asarray([0.0, output_voltage_v])
        switch = on_transition @ start + on_affine
        if switch[0] <= 0.0:
            return None
        zero_delay = _first_off_zero(parameters, load_ohm, switch)
        if zero_delay is None or zero_delay >= parameters.off_duration_s:
            return None
        zero_state = _propagate(matrix, off_offset, switch, zero_delay)
        zero_state[0] = 0.0
        idle_duration = parameters.off_duration_s - zero_delay
        end_voltage = float(
            zero_state[1] * math.exp(-idle_duration / (load_ohm * parameters.capacitance_f))
        )
        return end_voltage - output_voltage_v, switch, zero_delay, zero_state

    candidates = np.linspace(0.0, parameters.vin_v * (1.0 - 1e-13), 513)
    valid: list[tuple[float, float]] = []
    for candidate in candidates:
        evaluated = cycle(float(candidate))
        if evaluated is not None and math.isfinite(evaluated[0]):
            valid.append((float(candidate), float(evaluated[0])))
    bracket: tuple[float, float] | None = None
    root_value: float | None = None
    for (left, left_value), (right, right_value) in zip(valid[:-1], valid[1:], strict=True):
        if left_value == 0.0:
            root_value = left
            break
        if left_value * right_value < 0.0:
            bracket = (left, right)
            break
    if root_value is None and bracket is None:
        raise BuckBoundaryBenchmarkError(
            f"could not bracket periodic DCM voltage for load {load_ohm:.12g} ohm"
        )
    if root_value is None:
        assert bracket is not None

        def residual(voltage: float) -> float:
            evaluated = cycle(voltage)
            if evaluated is None:
                raise BuckBoundaryBenchmarkError("DCM periodic root left its valid mode domain")
            return float(evaluated[0])

        root_value = float(
            brentq(
                residual,
                bracket[0],
                bracket[1],
                xtol=max(np.nextafter(0.0, 1.0), parameters.vin_v * 1e-14),
                rtol=1e-14,
                maxiter=200,
            )
        )
    evaluated = cycle(root_value)
    if evaluated is None:
        raise BuckBoundaryBenchmarkError("periodic DCM root is outside the three-mode domain")
    _residual, switch, zero_delay, zero_state = evaluated
    start = np.asarray([0.0, root_value])
    on_minimum = _minimum_current(matrix, on_offset, start, parameters.on_duration_s)
    if on_minimum < -parameters.current_base_a * 1e-12:
        raise BuckBoundaryBenchmarkError(
            "periodic DCM ON interval contains forbidden reverse inductor current"
        )
    switch_integrated, on_integral = _propagate_with_voltage_integral(
        matrix, on_offset, start, parameters.on_duration_s
    )
    if not np.allclose(switch_integrated, switch, rtol=1e-12, atol=1e-12):
        raise BuckBoundaryBenchmarkError("inconsistent ON propagation in DCM oracle")
    zero_integrated, off_integral = _propagate_with_voltage_integral(
        matrix, off_offset, switch, zero_delay
    )
    zero_integrated[0] = 0.0
    if not np.allclose(zero_integrated, zero_state, rtol=1e-12, atol=1e-12):
        raise BuckBoundaryBenchmarkError("inconsistent freewheel propagation in DCM oracle")
    idle_duration = parameters.off_duration_s - zero_delay
    decay = math.exp(-idle_duration / (load_ohm * parameters.capacitance_f))
    end = np.asarray([0.0, zero_state[1] * decay])
    idle_integral = zero_state[1] * load_ohm * parameters.capacitance_f * (1.0 - decay)
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
        fractions,
    )


def _first_off_zero(
    parameters: BuckBoundaryParameters,
    load_ohm: float,
    state: np.ndarray,
) -> float | None:
    current = float(state[0])
    voltage = float(state[1])
    if current <= 0.0 or voltage <= 0.0:
        return None
    slope = -voltage / parameters.inductance_h
    try:
        alpha = 1.0 / (2.0 * load_ohm * parameters.capacitance_f)
        natural_squared = (1.0 / parameters.inductance_h) * (1.0 / parameters.capacitance_f)
    except (OverflowError, ZeroDivisionError) as exc:
        raise BuckBoundaryBenchmarkError("natural-event coefficients must be finite") from exc
    if not math.isfinite(alpha) or not math.isfinite(natural_squared):
        raise BuckBoundaryBenchmarkError("natural-event coefficients must be finite")
    discriminant = natural_squared - alpha * alpha
    scale = max(natural_squared, alpha * alpha, 1.0)
    if discriminant > scale * 1e-14:
        beta = math.sqrt(discriminant)
        sine_coefficient = (slope + alpha * current) / beta
        phase = math.atan2(sine_coefficient, current)
        base = phase + math.pi / 2.0
        index = math.floor(-base / math.pi) + 1
        theta = base + index * math.pi
        root = theta / beta
    elif discriminant < -scale * 1e-14:
        rate = math.sqrt(-discriminant)
        first_root = -alpha - rate
        second_root = -alpha + rate
        first_coefficient = (slope - second_root * current) / (first_root - second_root)
        second_coefficient = current - first_coefficient
        if first_coefficient == 0.0:
            return None
        ratio = -second_coefficient / first_coefficient
        if ratio <= 0.0:
            return None
        root = math.log(ratio) / (first_root - second_root)
    else:
        coefficient = slope + alpha * current
        if coefficient >= 0.0:
            return None
        root = -current / coefficient
    tolerance = parameters.period_s * 1e-11
    if not math.isfinite(root) or root <= tolerance:
        return None
    if root > parameters.off_duration_s + tolerance:
        return None
    return min(root, parameters.off_duration_s)


def _minimum_current(
    matrix: np.ndarray,
    offset: np.ndarray,
    state: np.ndarray,
    duration_s: float,
) -> float:
    """Evaluate every analytical current extremum of the second-order affine mode."""

    try:
        equilibrium = np.linalg.solve(matrix, -offset)
    except np.linalg.LinAlgError as exc:
        raise BuckBoundaryBenchmarkError("active affine mode has no unique equilibrium") from exc
    current_equilibrium = float(equilibrium[0])
    displacement = float(state[0]) - current_equilibrium
    initial_derivative = float((matrix @ state + offset)[0])
    alpha = -0.5 * float(matrix[1, 1])
    natural_squared = -float(matrix[0, 1] * matrix[1, 0])
    if (
        not math.isfinite(alpha)
        or not math.isfinite(natural_squared)
        or alpha < 0.0
        or natural_squared <= 0.0
    ):
        raise BuckBoundaryBenchmarkError("active mode is not a passive canonical RLC mode")
    discriminant = natural_squared - alpha * alpha
    scale = max(natural_squared, alpha * alpha, 1.0)
    candidate_times = [0.0, duration_s]

    if discriminant > scale * 1e-14:
        beta = math.sqrt(discriminant)
        sine_coefficient = (initial_derivative + alpha * displacement) / beta
        derivative_cosine = initial_derivative
        derivative_sine = -alpha * sine_coefficient - beta * displacement

        def current(time_s: float) -> float:
            angle = beta * time_s
            homogeneous = math.exp(-alpha * time_s) * (
                displacement * math.cos(angle) + sine_coefficient * math.sin(angle)
            )
            return current_equilibrium + homogeneous

        if derivative_cosine != 0.0 or derivative_sine != 0.0:
            phase = math.atan2(derivative_sine, derivative_cosine)
            first_family_root = phase + math.pi / 2.0
            maximum_angle = beta * duration_s
            if not math.isfinite(maximum_angle):
                raise BuckBoundaryBenchmarkError("analytical extrema count is non-finite")
            first_index = math.ceil((-first_family_root) / math.pi)
            last_index = math.floor((maximum_angle - first_family_root) / math.pi)
            root_count = max(0, last_index - first_index + 1)
            if root_count > MAX_ANALYTIC_EXTREMA_PER_INTERVAL:
                raise BuckBoundaryBenchmarkError(
                    "analytical extrema count exceeds the bounded benchmark envelope: "
                    f"{root_count} > {MAX_ANALYTIC_EXTREMA_PER_INTERVAL}"
                )
            candidate_times.extend(
                (first_family_root + index * math.pi) / beta
                for index in range(first_index, last_index + 1)
            )
    elif discriminant < -scale * 1e-14:
        rate = math.sqrt(-discriminant)
        first_root = -alpha - rate
        second_root = -alpha + rate
        first_coefficient = (initial_derivative - second_root * displacement) / (
            first_root - second_root
        )
        second_coefficient = displacement - first_coefficient

        def current(time_s: float) -> float:
            return (
                current_equilibrium
                + first_coefficient * math.exp(first_root * time_s)
                + second_coefficient * math.exp(second_root * time_s)
            )

        first_derivative_coefficient = first_root * first_coefficient
        second_derivative_coefficient = second_root * second_coefficient
        if first_derivative_coefficient != 0.0:
            root_ratio = -second_derivative_coefficient / first_derivative_coefficient
            if root_ratio > 0.0:
                root = math.log(root_ratio) / (first_root - second_root)
                if 0.0 <= root <= duration_s:
                    candidate_times.append(root)
    else:
        linear_coefficient = initial_derivative + alpha * displacement

        def current(time_s: float) -> float:
            return current_equilibrium + (displacement + linear_coefficient * time_s) * math.exp(
                -alpha * time_s
            )

        denominator = alpha * linear_coefficient
        if denominator != 0.0:
            root = (linear_coefficient - alpha * displacement) / denominator
            if 0.0 <= root <= duration_s:
                candidate_times.append(root)

    bounded_times = [
        min(duration_s, max(0.0, float(time_s)))
        for time_s in candidate_times
        if math.isfinite(time_s) and -duration_s * 1e-13 <= time_s <= duration_s * (1.0 + 1e-13)
    ]
    if not bounded_times:
        raise BuckBoundaryBenchmarkError("current-extremum enumeration produced no candidates")
    values = [current(time_s) for time_s in sorted(set(bounded_times))]
    if not all(math.isfinite(value) for value in values):
        raise BuckBoundaryBenchmarkError("current-extremum evaluation is non-finite")
    return min(values)


def _make_point(
    parameters: BuckBoundaryParameters,
    load_ohm: float,
    ratio: float,
    regime: Regime,
    start: np.ndarray,
    switch: np.ndarray,
    diode_zero_state: BuckState | None,
    end: np.ndarray,
    diode_zero_s: float | None,
    mean_output_voltage_v: float,
    minimum_current_a: float,
    fractions: dict[str, float],
) -> BuckBoundaryPoint:
    current_scale = parameters.current_base_a
    periodicity = max(
        abs(float(end[0] - start[0])) / current_scale,
        abs(float(end[1] - start[1])) / parameters.vin_v,
    )
    fraction_sum_error = abs(sum(fractions.values()) - 1.0)
    reverse_normalized = max(0.0, -minimum_current_a) / current_scale
    oracle_gates = (
        _gate("periodicity", periodicity, 1e-10),
        _gate("unilateral_current", reverse_normalized, 1e-12),
        _gate("mode_fraction_sum", fraction_sum_error, 1e-12),
    )
    return BuckBoundaryPoint(
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
        mean_output_voltage_v=mean_output_voltage_v,
        minimum_inductor_current_a=minimum_current_a,
        fraction_by_mode=fractions,
        periodicity_residual_normalized=periodicity,
        oracle_gates=oracle_gates,
    )


def _require_oracle_health(point: BuckBoundaryPoint) -> None:
    failed = [gate["id"] for gate in point.oracle_gates if gate["status"] != "pass"]
    if failed:
        raise BuckBoundaryBenchmarkError(
            f"independent oracle health gates failed for {point.point_id}: {failed}"
        )
    idle = point.fraction_by_mode["idle"]
    if point.regime == "CCM" and (
        point.minimum_inductor_current_a <= 0.0 or point.diode_zero_s is not None or idle != 0.0
    ):
        raise BuckBoundaryBenchmarkError("CCM oracle point violates its regime domain")
    if point.regime == "BCM" and (
        point.diode_zero_s is None
        or abs(point.diode_zero_s - point.switch_off_s / point.fraction_by_mode["on"])
        > (point.switch_off_s / point.fraction_by_mode["on"]) * BCM_COINCIDENCE_FRACTION_TOLERANCE
        or idle > BCM_COINCIDENCE_FRACTION_TOLERANCE
    ):
        raise BuckBoundaryBenchmarkError("BCM oracle point violates its coincidence domain")
    if point.regime == "DCM" and (
        point.diode_zero_s is None or idle <= BCM_COINCIDENCE_FRACTION_TOLERANCE
    ):
        raise BuckBoundaryBenchmarkError("DCM oracle point lacks a finite idle interval")


def _gate(gate_id: str, actual: float, limit: float) -> dict[str, Any]:
    finite_and_nonnegative = math.isfinite(actual) and actual >= 0.0
    return {
        "id": gate_id,
        "metric": "normalized_absolute_error",
        "actual": actual,
        "operator": "<=",
        "limit": limit,
        "status": "pass" if finite_and_nonnegative and actual <= limit else "fail",
    }


def _compare_point(
    parameters: BuckBoundaryParameters,
    reference: BuckBoundaryPoint,
    observed: BuckBoundaryObservation,
) -> dict[str, Any]:
    _validate_observation(observed)
    if observed.point_id != reference.point_id:
        raise BuckBoundaryBenchmarkError(
            f"point-id mismatch: expected {reference.point_id!r}, got {observed.point_id!r}"
        )
    _require_matching_parameters(parameters, observed.parameters, context=reference.point_id)
    if not math.isclose(observed.load_ohm, reference.load_ohm, rel_tol=2e-12):
        raise BuckBoundaryBenchmarkError(f"load mismatch for {reference.point_id}")
    if not math.isclose(
        observed.load_ratio_to_bcm,
        reference.load_ratio_to_bcm,
        rel_tol=2e-12,
        abs_tol=1e-15,
    ):
        raise BuckBoundaryBenchmarkError(f"load-ratio mismatch for {reference.point_id}")
    nonzero_errors: list[float] = []
    zero_errors: list[float] = []
    pairs = [
        (reference.start_state, observed.start_state),
        (reference.switch_off_state, observed.switch_off_state),
        (reference.end_state, observed.end_state),
    ]
    if reference.diode_zero_state is not None:
        if observed.diode_zero_state is None:
            zero_errors.append(math.inf)
        else:
            pairs.append((reference.diode_zero_state, observed.diode_zero_state))
    for expected_state, actual_state in pairs:
        _append_state_error(
            expected_state.inductor_current_a,
            actual_state.inductor_current_a,
            parameters.current_base_a,
            nonzero_errors,
            zero_errors,
        )
        _append_state_error(
            expected_state.output_voltage_v,
            actual_state.output_voltage_v,
            parameters.vin_v,
            nonzero_errors,
            zero_errors,
        )
    maximum_nonzero_error = max(nonzero_errors, default=0.0)
    maximum_zero_error = max(zero_errors, default=0.0)

    if reference.diode_zero_s is None:
        event_error = 0.0 if observed.diode_zero_s is None else math.inf
    elif observed.diode_zero_s is None:
        event_error = math.inf
    else:
        event_error = abs(observed.diode_zero_s - reference.diode_zero_s) / parameters.period_s
    mean_error = abs(observed.mean_output_voltage_v - reference.mean_output_voltage_v) / max(
        abs(reference.mean_output_voltage_v), parameters.vin_v * 1e-12
    )
    fraction_error = max(
        abs(observed.fraction_by_mode.get(name, math.inf) - expected)
        for name, expected in reference.fraction_by_mode.items()
    )
    reverse_violation = max(0.0, -observed.minimum_inductor_current_a) / (parameters.current_base_a)
    periodicity = max(
        abs(observed.end_state.inductor_current_a - observed.start_state.inductor_current_a)
        / parameters.current_base_a,
        abs(observed.end_state.output_voltage_v - observed.start_state.output_voltage_v)
        / parameters.vin_v,
    )
    unclassified_fraction = observed.fraction_by_mode["unclassified"]
    fraction_sum_error = abs(sum(observed.fraction_by_mode.values()) - 1.0)
    gates = [
        {
            "id": "regime",
            "metric": "exact_match",
            "actual": observed.regime,
            "reference": reference.regime,
            "limit": True,
            "status": "pass" if observed.regime == reference.regime else "fail",
        },
        _comparison_gate(
            "boundary_state_nonzero", maximum_nonzero_error, BOUNDARY_STATE_RELATIVE_LIMIT
        ),
        _comparison_gate("boundary_state_zero", maximum_zero_error, BOUNDARY_ZERO_NORMALIZED_LIMIT),
        _comparison_gate("natural_event_time", event_error, EVENT_TIME_RELATIVE_LIMIT),
        _comparison_gate("reverse_current", reverse_violation, REVERSE_CURRENT_NORMALIZED_LIMIT),
        _comparison_gate("cycle_mean_output", mean_error, CYCLE_MEAN_RELATIVE_LIMIT),
        _comparison_gate("mode_fractions", fraction_error, MODE_FRACTION_ABSOLUTE_LIMIT),
        _comparison_gate(
            "energy_balance", observed.energy_balance_relative, ENERGY_BALANCE_RELATIVE_LIMIT
        ),
        _comparison_gate(
            "cycle_periodicity",
            periodicity,
            CYCLE_PERIODICITY_NORMALIZED_LIMIT,
        ),
        _comparison_gate(
            "unclassified_fraction",
            unclassified_fraction,
            UNCLASSIFIED_FRACTION_LIMIT,
        ),
        _comparison_gate(
            "mode_fraction_sum",
            fraction_sum_error,
            MODE_FRACTION_SUM_ERROR_LIMIT,
        ),
    ]
    return {
        "point_id": reference.point_id,
        "load_ratio_to_bcm": reference.load_ratio_to_bcm,
        "load_ohm": reference.load_ohm,
        "regime": observed.regime,
        "expected_regime": reference.regime,
        "observed_regime": observed.regime,
        "status": "pass" if all(gate["status"] == "pass" for gate in gates) else "fail",
        "metrics": {
            "mean_output_voltage_v": observed.mean_output_voltage_v,
            "minimum_inductor_current_a": observed.minimum_inductor_current_a,
            "conduction_fraction": observed.conduction_fraction,
            "fraction_by_mode": dict(observed.fraction_by_mode),
            "energy_balance_relative": observed.energy_balance_relative,
            "cycle_periodicity_normalized": periodicity,
            "mode_fraction_sum_error": fraction_sum_error,
        },
        "reference_metrics": {
            "mean_output_voltage_v": reference.mean_output_voltage_v,
            "minimum_inductor_current_a": reference.minimum_inductor_current_a,
            "conduction_fraction": reference.conduction_fraction,
            "fraction_by_mode": dict(reference.fraction_by_mode),
        },
        "events": {
            "observed_diode_zero_s": observed.diode_zero_s,
            "reference_diode_zero_s": reference.diode_zero_s,
        },
        "gates": gates,
    }


def _validate_observation(observation: BuckBoundaryObservation) -> None:
    if not isinstance(observation.point_id, str) or not observation.point_id:
        raise BuckBoundaryBenchmarkError("observation point_id must be a non-empty string")
    if observation.regime not in {"CCM", "BCM", "DCM"}:
        raise BuckBoundaryBenchmarkError("observation regime must be CCM, BCM, or DCM")
    positive = {
        "load_ratio_to_bcm": observation.load_ratio_to_bcm,
        "load_ohm": observation.load_ohm,
    }
    for name, value in positive.items():
        if not math.isfinite(value) or value <= 0.0:
            raise BuckBoundaryBenchmarkError(f"observation {name} must be finite and positive")
    states = {
        "start_state": observation.start_state,
        "switch_off_state": observation.switch_off_state,
        "end_state": observation.end_state,
    }
    if observation.diode_zero_state is not None:
        states["diode_zero_state"] = observation.diode_zero_state
    for name, state in states.items():
        if not math.isfinite(state.inductor_current_a) or not math.isfinite(state.output_voltage_v):
            raise BuckBoundaryBenchmarkError(f"observation {name} must contain finite states")
    if observation.diode_zero_s is not None and (
        not math.isfinite(observation.diode_zero_s) or observation.diode_zero_s < 0.0
    ):
        raise BuckBoundaryBenchmarkError(
            "observation diode_zero_s must be finite and nonnegative when present"
        )
    finite_scalars = {
        "mean_output_voltage_v": observation.mean_output_voltage_v,
        "minimum_inductor_current_a": observation.minimum_inductor_current_a,
        "energy_balance_relative": observation.energy_balance_relative,
    }
    for name, value in finite_scalars.items():
        if not math.isfinite(value):
            raise BuckBoundaryBenchmarkError(f"observation {name} must be finite")
    if observation.energy_balance_relative < 0.0:
        raise BuckBoundaryBenchmarkError("observation energy_balance_relative must be nonnegative")
    expected_modes = {"on", "freewheel", "idle", "unclassified"}
    if set(observation.fraction_by_mode) != expected_modes:
        raise BuckBoundaryBenchmarkError(
            "observation fraction_by_mode must contain on/freewheel/idle/unclassified"
        )
    for name, value in observation.fraction_by_mode.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise BuckBoundaryBenchmarkError(
                f"observation mode fraction {name!r} must be finite and within [0,1]"
            )


def _require_matching_parameters(
    expected: BuckBoundaryParameters,
    actual: BuckBoundaryParameters,
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
            raise BuckBoundaryBenchmarkError(
                f"parameter mismatch for {context}: {field} expected "
                f"{expected_value:.17g}, got {actual_value:.17g}"
            )


def _point_id(ratio: float) -> str:
    return f"load-ratio-{format(ratio, '.12g').replace('.', 'p')}"


def _state(vector: np.ndarray) -> BuckState:
    return BuckState(float(vector[0]), float(vector[1]))


def _state_from_mapping(
    values: Any,
    inductor_id: str,
    capacitor_id: str,
) -> BuckState:
    if not isinstance(values, dict) or inductor_id not in values or capacitor_id not in values:
        raise BuckBoundaryBenchmarkError("accepted boundary state is incomplete")
    state = BuckState(float(values[inductor_id]), float(values[capacitor_id]))
    if not math.isfinite(state.inductor_current_a) or not math.isfinite(state.output_voltage_v):
        raise BuckBoundaryBenchmarkError("accepted boundary state must be finite")
    return state
