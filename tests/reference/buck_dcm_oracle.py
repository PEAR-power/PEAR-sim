"""Independent test oracle for the canonical ideal buck converter in DCM.

This module deliberately does not import PEARSim's circuit compiler, mode matrices,
event manager, or transient solver. It derives the three affine modes locally,
propagates them with ``scipy.linalg.expm``, and locates diode current zero with an
independent bracketed ``scipy.optimize.brentq`` solve.

The result is verification evidence for the declared ideal equations. It is not a
physical-validation claim and it is not shipped as a runtime backend.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from pear_sim.model import Component, Model, PwmControl


ORACLE_SCHEMA = "pear-sim/test-oracle/ideal-buck-dcm/v0.1"
ORACLE_KIND = "independent-affine-state-space-verification-oracle"


class OracleModelError(ValueError):
    """Raised when a model is not the exact fixture topology understood by the oracle."""


@dataclass(frozen=True)
class ReferenceState:
    inductor_current_a: float
    output_voltage_v: float

    def as_vector(self) -> np.ndarray:
        return np.asarray([self.inductor_current_a, self.output_voltage_v], dtype=float)


@dataclass(frozen=True)
class ReferenceCycle:
    index: int
    start_s: float
    switch_off_s: float
    diode_zero_s: float | None
    end_s: float
    start_state: ReferenceState
    switch_off_state: ReferenceState
    diode_zero_state: ReferenceState | None
    end_state: ReferenceState
    mean_output_voltage_v: float
    conduction_fraction: float
    diode_conduction_fraction: float
    idle_fraction: float

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mode_at_end"] = "idle" if self.diode_zero_s is not None else "diode_on"
        return value


@dataclass(frozen=True)
class BuckDcmOracleResult:
    parameters_si: dict[str, float]
    cycles: tuple[ReferenceCycle, ...]

    @property
    def first_diode_zero(self) -> ReferenceCycle | None:
        return next((cycle for cycle in self.cycles if cycle.diode_zero_s is not None), None)

    @property
    def last_cycle(self) -> ReferenceCycle:
        return self.cycles[-1]

    def tail_mean_output_voltage(self, cycle_count: int = 20) -> float:
        if cycle_count <= 0:
            raise ValueError("cycle_count must be positive")
        selected = self.cycles[-min(cycle_count, len(self.cycles)) :]
        return sum(cycle.mean_output_voltage_v for cycle in selected) / len(selected)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": ORACLE_SCHEMA,
            "kind": ORACLE_KIND,
            "claim": {
                "classification": "verification",
                "validation_claim": False,
                "statement": (
                    "Independent exact-mode reference for an ideal buck; not a hardware "
                    "or cross-tool validation claim."
                ),
            },
            "method": {
                "state": ["inductor_current_a", "output_voltage_v"],
                "propagation": "scipy.linalg.expm of locally derived affine modes",
                "natural_event": "scipy.optimize.brentq root of iL(t)=0 during diode mode",
                "pear_sim_kernel_imported": False,
            },
            "parameters_si": dict(sorted(self.parameters_si.items())),
            "cycle_count": len(self.cycles),
            "first_diode_zero_s": (
                None if self.first_diode_zero is None else self.first_diode_zero.diode_zero_s
            ),
            "tail_mean_output_voltage_v": self.tail_mean_output_voltage(),
            "cycles": [cycle.as_dict() for cycle in self.cycles],
        }


@dataclass(frozen=True)
class _Fixture:
    source: Component
    switch: Component
    diode: Component
    inductor: Component
    capacitor: Component
    load: Component
    pwm: PwmControl


def evaluate_buck_dcm(model: Model, *, cycle_count: int | None = None) -> BuckDcmOracleResult:
    fixture = _extract_fixture(model)
    period = fixture.pwm.period
    if cycle_count is None:
        raw_cycles = (model.transient.stop - model.transient.start) / period
        cycle_count = round(raw_cycles)
        if not math.isclose(raw_cycles, cycle_count, rel_tol=0.0, abs_tol=1e-10):
            raise OracleModelError("Transient span must be an integer number of PWM periods")
    if cycle_count <= 0:
        raise OracleModelError("Oracle requires at least one PWM period")

    vin = float(fixture.source.value)
    inductance = float(fixture.inductor.value)
    capacitance = float(fixture.capacitor.value)
    resistance = float(fixture.load.value)
    duty = fixture.pwm.duty
    on_duration = duty * period
    off_duration = period - on_duration

    active_matrix = np.asarray(
        [
            [0.0, -1.0 / inductance],
            [1.0 / capacitance, -1.0 / (resistance * capacitance)],
        ],
        dtype=float,
    )
    on_offset = np.asarray([vin / inductance, 0.0], dtype=float)
    off_offset = np.zeros(2, dtype=float)
    idle_matrix = np.asarray(
        [[0.0, 0.0], [0.0, -1.0 / (resistance * capacitance)]], dtype=float
    )

    state = ReferenceState(
        inductor_current_a=fixture.inductor.initial_state,
        output_voltage_v=fixture.capacitor.initial_state,
    )
    cycles: list[ReferenceCycle] = []
    for index in range(cycle_count):
        cycle_start = model.transient.start + index * period
        start_state = state
        after_on_vector, on_voltage_integral = _propagate_with_voltage_integral(
            active_matrix, on_offset, start_state.as_vector(), on_duration
        )
        after_on = _state(after_on_vector)
        minimum_on_current, minimum_on_time = _minimum_current_during_interval(
            active_matrix,
            on_offset,
            start_state.as_vector(),
            on_duration,
        )
        current_scale = max(1.0, abs(after_on.inductor_current_a))
        if minimum_on_current < -1e-12 * current_scale:
            raise OracleModelError(
                f"Cycle {index}: on interval contains reverse inductor current "
                f"{minimum_on_current:.12g} A at offset {minimum_on_time:.12g} s; "
                "outside this DCM oracle's domain"
            )
        if after_on.inductor_current_a <= 0.0:
            raise OracleModelError(
                f"Cycle {index}: on interval did not establish positive inductor current"
            )

        diode_zero_delay = _first_zero_current_delay(
            active_matrix, off_offset, after_on_vector, off_duration
        )
        diode_zero_state: ReferenceState | None = None
        if diode_zero_delay is not None:
            zero_vector, off_voltage_integral = _propagate_with_voltage_integral(
                active_matrix, off_offset, after_on_vector, diode_zero_delay
            )
            zero_vector[0] = 0.0
            diode_zero_state = _state(zero_vector)
            idle_duration = off_duration - diode_zero_delay
            end_vector, idle_voltage_integral = _propagate_with_voltage_integral(
                idle_matrix, off_offset, zero_vector, idle_duration
            )
            end_vector[0] = 0.0
            voltage_integral = (
                on_voltage_integral + off_voltage_integral + idle_voltage_integral
            )
            conduction_fraction = (on_duration + diode_zero_delay) / period
            diode_conduction_fraction = diode_zero_delay / period
        else:
            end_vector, off_voltage_integral = _propagate_with_voltage_integral(
                active_matrix, off_offset, after_on_vector, off_duration
            )
            voltage_integral = on_voltage_integral + off_voltage_integral
            conduction_fraction = 1.0
            diode_conduction_fraction = 1.0 - duty

        state = _state(end_vector)
        switch_off = cycle_start + on_duration
        diode_zero = (
            None if diode_zero_delay is None else switch_off + diode_zero_delay
        )
        cycles.append(
            ReferenceCycle(
                index=index,
                start_s=cycle_start,
                switch_off_s=switch_off,
                diode_zero_s=diode_zero,
                end_s=cycle_start + period,
                start_state=start_state,
                switch_off_state=after_on,
                diode_zero_state=diode_zero_state,
                end_state=state,
                mean_output_voltage_v=voltage_integral / period,
                conduction_fraction=conduction_fraction,
                diode_conduction_fraction=diode_conduction_fraction,
                idle_fraction=1.0 - conduction_fraction,
            )
        )

    parameters = {
        "vin_v": vin,
        "duty": duty,
        "switching_frequency_hz": fixture.pwm.frequency,
        "period_s": period,
        "inductance_h": inductance,
        "capacitance_f": capacitance,
        "load_ohm": resistance,
    }
    return BuckDcmOracleResult(parameters_si=parameters, cycles=tuple(cycles))


def textbook_dcm_conversion_ratio(parameters_si: dict[str, float]) -> float:
    """Return the small-ripple textbook DCM conversion ratio for a resistive load."""

    duty = parameters_si["duty"]
    inductance = parameters_si["inductance_h"]
    resistance = parameters_si["load_ohm"]
    period = parameters_si["period_s"]
    k = 2.0 * inductance / (resistance * period)
    if k == 0.0:
        raise ValueError("DCM parameter k must be positive")
    return (-duty * duty + math.sqrt(duty**4 + 4.0 * k * duty * duty)) / (2.0 * k)


def _extract_fixture(model: Model) -> _Fixture:
    functional = [component for component in model.components if component.type != "voltage_probe"]
    expected = Counter(
        {
            "voltage_source": 1,
            "ideal_switch": 1,
            "diode": 1,
            "inductor": 1,
            "capacitor": 1,
            "resistor": 1,
        }
    )
    found = Counter(component.type for component in functional)
    if found != expected:
        raise OracleModelError(
            "Oracle accepts exactly one voltage source, switch, diode, inductor, "
            f"capacitor, and load resistor; found {dict(sorted(found.items()))}"
        )
    if len(model.controls) != 1:
        raise OracleModelError("Oracle accepts exactly one PWM control")
    by_type = {component.type: component for component in functional}
    source = by_type["voltage_source"]
    switch = by_type["ideal_switch"]
    diode = by_type["diode"]
    inductor = by_type["inductor"]
    capacitor = by_type["capacitor"]
    load = by_type["resistor"]
    pwm = next(iter(model.controls.values()))
    ground = model.ground
    if source.n != ground or source.value is None or source.value <= 0.0:
        raise OracleModelError("Source must be positive and oriented from input to ground")
    if switch.p != source.p or switch.control != pwm.id:
        raise OracleModelError("Switch must connect the source to the switch node under sole PWM")
    if diode.p != ground or diode.n != switch.n:
        raise OracleModelError("Diode must be oriented from ground to the switch node")
    if inductor.p != switch.n:
        raise OracleModelError("Inductor must connect the switch node to the output")
    if capacitor.p != inductor.n or capacitor.n != ground:
        raise OracleModelError("Capacitor must connect output to ground")
    if load.p != inductor.n or load.n != ground:
        raise OracleModelError("Load resistor must connect output to ground")
    if len({ground, source.p, switch.n, inductor.n}) != 4:
        raise OracleModelError("Ground, input, switch, and output nodes must be distinct")
    if not 0.0 < pwm.duty < 1.0 or pwm.phase != 0.0:
        raise OracleModelError("PWM must have zero phase and duty strictly between zero and one")
    if model.transient.start != 0.0:
        raise OracleModelError("Oracle fixture must start at t=0")
    if not switch.initial or diode.initial:
        raise OracleModelError("Fixture must start with switch on and diode off")
    if inductor.initial_state != 0.0 or capacitor.initial_state != 0.0:
        raise OracleModelError("Fixture must use zero inductor-current and capacitor-voltage IC")
    return _Fixture(source, switch, diode, inductor, capacitor, load, pwm)


def _propagate(matrix: np.ndarray, offset: np.ndarray, state: np.ndarray, duration: float) -> np.ndarray:
    augmented = np.zeros((3, 3), dtype=float)
    augmented[:2, :2] = matrix
    augmented[:2, 2] = offset
    transition = expm(augmented * duration)
    return np.asarray(transition @ np.asarray([state[0], state[1], 1.0]))[:2]


def _propagate_with_voltage_integral(
    matrix: np.ndarray,
    offset: np.ndarray,
    state: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, float]:
    augmented = np.zeros((4, 4), dtype=float)
    augmented[:2, :2] = matrix
    augmented[:2, 3] = offset
    augmented[2, 1] = 1.0
    initial = np.asarray([state[0], state[1], 0.0, 1.0], dtype=float)
    propagated = expm(augmented * duration) @ initial
    return np.asarray(propagated[:2], dtype=float), float(propagated[2])


def _first_zero_current_delay(
    matrix: np.ndarray,
    offset: np.ndarray,
    state: np.ndarray,
    maximum: float,
) -> float | None:
    def current(delay: float) -> float:
        return float(_propagate(matrix, offset, state, delay)[0])

    start_current = current(0.0)
    if start_current <= 0.0:
        raise OracleModelError("Diode interval must begin with positive inductor current")
    previous_time = 0.0
    previous_current = start_current
    for index in range(1, 513):
        candidate_time = maximum * index / 512
        candidate_current = current(candidate_time)
        if candidate_current == 0.0:
            return candidate_time
        if candidate_current < 0.0 <= previous_current:
            return float(
                brentq(
                    current,
                    previous_time,
                    candidate_time,
                    xtol=1e-15,
                    rtol=1e-14,
                    maxiter=100,
                )
            )
        previous_time = candidate_time
        previous_current = candidate_current
    return None


def _minimum_current_during_interval(
    matrix: np.ndarray,
    offset: np.ndarray,
    state: np.ndarray,
    maximum: float,
) -> tuple[float, float]:
    """Locate current extrema independently from PEARSim's dense-output machinery."""

    def propagated(delay: float) -> np.ndarray:
        return _propagate(matrix, offset, state, delay)

    def current(delay: float) -> float:
        return float(propagated(delay)[0])

    def current_derivative(delay: float) -> float:
        return float((matrix @ propagated(delay) + offset)[0])

    angular_rate = max((abs(value.imag) for value in np.linalg.eigvals(matrix)), default=0.0)
    estimated_half_waves = maximum * angular_rate / math.pi
    subdivisions = max(64, min(4096, math.ceil(estimated_half_waves * 16)))
    times = np.linspace(0.0, maximum, subdivisions + 1)
    candidates = [(current(float(time)), float(time)) for time in times]
    derivatives = [current_derivative(float(time)) for time in times]
    for left, right, left_value, right_value in zip(
        times[:-1],
        times[1:],
        derivatives[:-1],
        derivatives[1:],
        strict=True,
    ):
        if left_value == 0.0:
            candidates.append((current(float(left)), float(left)))
        if left_value * right_value < 0.0:
            root = float(
                brentq(
                    current_derivative,
                    float(left),
                    float(right),
                    xtol=1e-15,
                    rtol=1e-14,
                    maxiter=100,
                )
            )
            candidates.append((current(root), root))
    return min(candidates, key=lambda item: (item[0], item[1]))


def _state(vector: np.ndarray) -> ReferenceState:
    return ReferenceState(float(vector[0]), float(vector[1]))
