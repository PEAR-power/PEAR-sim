# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Structural recognition of the exact canonical converter topologies.

Extracted verbatim from the boundary-oracle modules (restructuring stage 3,
move 10) so that core observation code (``pear_sim.analysis.mode_coverage``) no longer
imports the benchmark package.  This module contains structural matching and
derived circuit parameters only; the periodic oracle equations and gates
remain local to each ``pear_sim.benchmarks`` module per the AGENTS.md
oracle-independence boundary.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math

from pear_sim.spec.model import Component, Model, PwmControl




class BuckBoundaryBenchmarkError(ValueError):
    """The requested model or numerical result is outside the benchmark contract."""

    code = "benchmark.buck_boundary_invalid"
@dataclass(frozen=True)
class BuckBoundaryParameters:
    """Scalar SI inputs used by the independent reference."""

    vin_v: float
    inductance_h: float
    capacitance_f: float
    switching_frequency_hz: float
    duty: float

    def __post_init__(self) -> None:
        named_positive = {
            "vin_v": self.vin_v,
            "inductance_h": self.inductance_h,
            "capacitance_f": self.capacitance_f,
            "switching_frequency_hz": self.switching_frequency_hz,
        }
        for name, value in named_positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise BuckBoundaryBenchmarkError(f"{name} must be finite and positive")
        if not math.isfinite(self.duty) or not 0.0 < self.duty < 1.0:
            raise BuckBoundaryBenchmarkError("duty must be finite and strictly between 0 and 1")
        try:
            period = 1.0 / self.switching_frequency_hz
            on_duration = self.duty * period
            off_duration = (1.0 - self.duty) * period
            current_base = self.vin_v * on_duration / self.inductance_h
            natural_squared = (1.0 / self.inductance_h) * (1.0 / self.capacitance_f)
        except (OverflowError, ZeroDivisionError) as exc:
            raise BuckBoundaryBenchmarkError(
                "derived period, current scale, and natural frequency must be finite"
            ) from exc
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (
                period,
                on_duration,
                off_duration,
                current_base,
                natural_squared,
            )
        ):
            raise BuckBoundaryBenchmarkError(
                "derived period, current scale, and natural frequency must be finite and positive"
            )

    @property
    def period_s(self) -> float:
        return 1.0 / self.switching_frequency_hz

    @property
    def on_duration_s(self) -> float:
        return self.duty * self.period_s

    @property
    def off_duration_s(self) -> float:
        return (1.0 - self.duty) * self.period_s

    @property
    def current_base_a(self) -> float:
        return self.vin_v * self.on_duration_s / self.inductance_h

    def as_dict(self) -> dict[str, float]:
        return {
            "capacitance_f": self.capacitance_f,
            "duty": self.duty,
            "inductance_h": self.inductance_h,
            "switching_frequency_hz": self.switching_frequency_hz,
            "period_s": self.period_s,
            "vin_v": self.vin_v,
        }
@dataclass(frozen=True)
class BuckFixture:
    source: Component
    switch: Component
    diode: Component
    inductor: Component
    capacitor: Component
    load: Component
    pwm: PwmControl
def extract_buck_boundary_parameters(model: Model) -> BuckBoundaryParameters:
    """Fail-closed extraction of scalar inputs from the exact canonical topology."""

    fixture = extract_buck_fixture(model)
    return BuckBoundaryParameters(
        vin_v=float(fixture.source.value),
        inductance_h=float(fixture.inductor.value),
        capacitance_f=float(fixture.capacitor.value),
        switching_frequency_hz=fixture.pwm.frequency,
        duty=fixture.pwm.duty,
    )
def extract_buck_fixture(model: Model) -> BuckFixture:
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
        raise BuckBoundaryBenchmarkError(
            "benchmark accepts exactly one voltage source, switch, diode, inductor, "
            f"capacitor, and load resistor; found {dict(sorted(found.items()))}"
        )
    if len(model.controls) != 1:
        raise BuckBoundaryBenchmarkError("benchmark accepts exactly one PWM control")
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
        raise BuckBoundaryBenchmarkError("source must be positive and oriented input-to-ground")
    if switch.p != source.p or switch.control != pwm.id:
        raise BuckBoundaryBenchmarkError("switch must connect input to the PWM switch node")
    if diode.p != ground or diode.n != switch.n:
        raise BuckBoundaryBenchmarkError("diode must be oriented from ground to switch node")
    if inductor.p != switch.n:
        raise BuckBoundaryBenchmarkError("inductor must connect switch node to output")
    if capacitor.p != inductor.n or capacitor.n != ground:
        raise BuckBoundaryBenchmarkError("capacitor must connect output to ground")
    if load.p != inductor.n or load.n != ground:
        raise BuckBoundaryBenchmarkError("load resistor must connect output to ground")
    if len({ground, source.p, switch.n, inductor.n}) != 4:
        raise BuckBoundaryBenchmarkError("ground, input, switch, and output nodes must differ")
    if not 0.0 < pwm.duty < 1.0 or pwm.phase != 0.0:
        raise BuckBoundaryBenchmarkError("PWM must have zero phase and 0 < duty < 1")
    if not switch.initial or diode.initial:
        raise BuckBoundaryBenchmarkError("cycle must start with switch on and diode off")
    for component in (inductor, capacitor, load):
        if component.value is None or component.value <= 0.0:
            raise BuckBoundaryBenchmarkError(f"{component.id} must have a positive value")
    return BuckFixture(source, switch, diode, inductor, capacitor, load, pwm)


class BoostBoundaryBenchmarkError(ValueError):
    """The model, reference request, or observation is outside this contract."""

    code = "benchmark.boost_boundary_invalid"
@dataclass(frozen=True)
class BoostBoundaryParameters:
    vin_v: float
    inductance_h: float
    capacitance_f: float
    switching_frequency_hz: float
    duty: float

    def __post_init__(self) -> None:
        positive = {
            "vin_v": self.vin_v,
            "inductance_h": self.inductance_h,
            "capacitance_f": self.capacitance_f,
            "switching_frequency_hz": self.switching_frequency_hz,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise BoostBoundaryBenchmarkError(f"{name} must be finite and positive")
        if not math.isfinite(self.duty) or not 0.0 < self.duty < 1.0:
            raise BoostBoundaryBenchmarkError(
                "duty must be finite and strictly between 0 and 1"
            )
        try:
            derived = (
                self.period_s,
                self.on_duration_s,
                self.off_duration_s,
                self.current_base_a,
            )
        except (OverflowError, ZeroDivisionError) as exc:
            raise BoostBoundaryBenchmarkError(
                "derived period, interval durations, and current scale must be finite"
            ) from exc
        if not all(math.isfinite(value) and value > 0.0 for value in derived):
            raise BoostBoundaryBenchmarkError(
                "derived period, interval durations, and current scale must be finite"
            )

    @property
    def period_s(self) -> float:
        return 1.0 / self.switching_frequency_hz

    @property
    def on_duration_s(self) -> float:
        return self.duty * self.period_s

    @property
    def off_duration_s(self) -> float:
        return (1.0 - self.duty) * self.period_s

    @property
    def current_base_a(self) -> float:
        return self.vin_v * self.on_duration_s / self.inductance_h

    def as_dict(self) -> dict[str, float]:
        return {
            "capacitance_f": self.capacitance_f,
            "duty": self.duty,
            "inductance_h": self.inductance_h,
            "switching_frequency_hz": self.switching_frequency_hz,
            "period_s": self.period_s,
            "vin_v": self.vin_v,
        }
@dataclass(frozen=True)
class BoostFixture:
    source: Component
    inductor: Component
    switch: Component
    diode: Component
    capacitor: Component
    load: Component
    pwm: PwmControl
def extract_boost_boundary_parameters(model: Model) -> BoostBoundaryParameters:
    fixture = extract_boost_fixture(model)
    return BoostBoundaryParameters(
        vin_v=float(fixture.source.value),
        inductance_h=float(fixture.inductor.value),
        capacitance_f=float(fixture.capacitor.value),
        switching_frequency_hz=fixture.pwm.frequency,
        duty=fixture.pwm.duty,
    )
def extract_boost_fixture(model: Model) -> BoostFixture:
    functional = [component for component in model.components if component.type != "voltage_probe"]
    expected = Counter(
        {
            "voltage_source": 1,
            "inductor": 1,
            "ideal_switch": 1,
            "diode": 1,
            "capacitor": 1,
            "resistor": 1,
        }
    )
    found = Counter(component.type for component in functional)
    if found != expected:
        raise BoostBoundaryBenchmarkError(
            "benchmark accepts exactly one voltage source, inductor, switch, diode, "
            f"capacitor, and load resistor; found {dict(sorted(found.items()))}"
        )
    if len(model.controls) != 1:
        raise BoostBoundaryBenchmarkError("benchmark accepts exactly one PWM control")
    by_type = {component.type: component for component in functional}
    source = by_type["voltage_source"]
    inductor = by_type["inductor"]
    switch = by_type["ideal_switch"]
    diode = by_type["diode"]
    capacitor = by_type["capacitor"]
    load = by_type["resistor"]
    pwm = next(iter(model.controls.values()))
    ground = model.ground
    if (
        source.n != ground
        or source.value is None
        or not math.isfinite(float(source.value))
        or source.value <= 0.0
    ):
        raise BoostBoundaryBenchmarkError("source must be positive and oriented input-to-ground")
    if inductor.p != source.p:
        raise BoostBoundaryBenchmarkError("inductor must connect input to the switch node")
    if switch.p != inductor.n or switch.n != ground or switch.control != pwm.id:
        raise BoostBoundaryBenchmarkError(
            "switch must connect the inductor switch node to ground under the sole PWM"
        )
    if diode.p != inductor.n:
        raise BoostBoundaryBenchmarkError("diode must be oriented from switch node to output")
    if capacitor.p != diode.n or capacitor.n != ground:
        raise BoostBoundaryBenchmarkError("capacitor must connect output to ground")
    if load.p != diode.n or load.n != ground:
        raise BoostBoundaryBenchmarkError("load resistor must connect output to ground")
    if len({ground, source.p, inductor.n, diode.n}) != 4:
        raise BoostBoundaryBenchmarkError("ground, input, switch, and output nodes must differ")
    if not 0.0 < pwm.duty < 1.0 or pwm.phase != 0.0:
        raise BoostBoundaryBenchmarkError("PWM must have zero phase and 0 < duty < 1")
    if not switch.initial or diode.initial:
        raise BoostBoundaryBenchmarkError("cycle must start with switch on and diode off")
    for component in (inductor, capacitor, load):
        if (
            component.value is None
            or not math.isfinite(float(component.value))
            or component.value <= 0.0
        ):
            raise BoostBoundaryBenchmarkError(f"{component.id} must have a positive value")
    return BoostFixture(source, inductor, switch, diode, capacitor, load, pwm)


class InvertingBuckBoostBoundaryBenchmarkError(ValueError):
    """The model, request, reference, or observation is outside this contract."""

    code = "benchmark.inverting_buck_boost_boundary_invalid"
@dataclass(frozen=True)
class InvertingBuckBoostBoundaryParameters:
    vin_v: float
    inductance_h: float
    capacitance_f: float
    switching_frequency_hz: float
    duty: float

    def __post_init__(self) -> None:
        positive = {
            "vin_v": self.vin_v,
            "inductance_h": self.inductance_h,
            "capacitance_f": self.capacitance_f,
            "switching_frequency_hz": self.switching_frequency_hz,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise InvertingBuckBoostBoundaryBenchmarkError(
                    f"{name} must be finite and positive"
                )
        if not math.isfinite(self.duty) or not 0.0 < self.duty < 1.0:
            raise InvertingBuckBoostBoundaryBenchmarkError(
                "duty must be finite and strictly between 0 and 1"
            )
        try:
            derived = (
                self.period_s,
                self.on_duration_s,
                self.off_duration_s,
                self.current_base_a,
            )
        except (OverflowError, ZeroDivisionError) as exc:
            raise InvertingBuckBoostBoundaryBenchmarkError(
                "derived period, interval durations, and current scale must be finite"
            ) from exc
        if not all(math.isfinite(value) and value > 0.0 for value in derived):
            raise InvertingBuckBoostBoundaryBenchmarkError(
                "derived period, interval durations, and current scale must be finite"
            )

    @property
    def period_s(self) -> float:
        return 1.0 / self.switching_frequency_hz

    @property
    def on_duration_s(self) -> float:
        return self.duty * self.period_s

    @property
    def off_duration_s(self) -> float:
        return (1.0 - self.duty) * self.period_s

    @property
    def current_base_a(self) -> float:
        return self.vin_v * self.on_duration_s / self.inductance_h

    def as_dict(self) -> dict[str, float]:
        return {
            "capacitance_f": self.capacitance_f,
            "duty": self.duty,
            "inductance_h": self.inductance_h,
            "switching_frequency_hz": self.switching_frequency_hz,
            "period_s": self.period_s,
            "vin_v": self.vin_v,
        }
@dataclass(frozen=True)
class InvertingBuckBoostFixture:
    source: Component
    inductor: Component
    switch: Component
    diode: Component
    capacitor: Component
    load: Component
    pwm: PwmControl
def extract_inverting_buck_boost_boundary_parameters(
    model: Model,
) -> InvertingBuckBoostBoundaryParameters:
    fixture = extract_inverting_buck_boost_fixture(model)
    return InvertingBuckBoostBoundaryParameters(
        vin_v=float(fixture.source.value),
        inductance_h=float(fixture.inductor.value),
        capacitance_f=float(fixture.capacitor.value),
        switching_frequency_hz=fixture.pwm.frequency,
        duty=fixture.pwm.duty,
    )
def extract_inverting_buck_boost_fixture(model: Model) -> InvertingBuckBoostFixture:
    functional = [component for component in model.components if component.type != "voltage_probe"]
    expected = Counter(
        {
            "voltage_source": 1,
            "inductor": 1,
            "ideal_switch": 1,
            "diode": 1,
            "capacitor": 1,
            "resistor": 1,
        }
    )
    found = Counter(component.type for component in functional)
    if found != expected:
        raise InvertingBuckBoostBoundaryBenchmarkError(
            "benchmark accepts exactly one voltage source, inductor, switch, diode, "
            f"capacitor, and load resistor; found {dict(sorted(found.items()))}"
        )
    if len(model.controls) != 1:
        raise InvertingBuckBoostBoundaryBenchmarkError("benchmark accepts exactly one PWM control")
    by_type = {component.type: component for component in functional}
    source = by_type["voltage_source"]
    inductor = by_type["inductor"]
    switch = by_type["ideal_switch"]
    diode = by_type["diode"]
    capacitor = by_type["capacitor"]
    load = by_type["resistor"]
    pwm = next(iter(model.controls.values()))
    ground = model.ground
    if (
        source.n != ground
        or source.value is None
        or not math.isfinite(float(source.value))
        or source.value <= 0.0
    ):
        raise InvertingBuckBoostBoundaryBenchmarkError(
            "source must be positive and oriented input-to-ground"
        )
    if switch.p != source.p or switch.control != pwm.id or switch.n == ground:
        raise InvertingBuckBoostBoundaryBenchmarkError(
            "switch must connect input to the inductor switch node under the sole PWM"
        )
    if inductor.p != switch.n or inductor.n != ground:
        raise InvertingBuckBoostBoundaryBenchmarkError(
            "inductor must connect the switch node to ground"
        )
    if diode.n != switch.n:
        raise InvertingBuckBoostBoundaryBenchmarkError(
            "diode must be oriented from the negative output to the switch node"
        )
    if capacitor.p != ground or capacitor.n != diode.p:
        raise InvertingBuckBoostBoundaryBenchmarkError(
            "capacitor must be oriented ground-to-negative-output"
        )
    if load.p != ground or load.n != diode.p:
        raise InvertingBuckBoostBoundaryBenchmarkError(
            "load resistor must be oriented ground-to-negative-output"
        )
    if len({ground, source.p, switch.n, diode.p}) != 4:
        raise InvertingBuckBoostBoundaryBenchmarkError(
            "ground, input, switch, and output nodes must differ"
        )
    if not 0.0 < pwm.duty < 1.0 or pwm.phase != 0.0:
        raise InvertingBuckBoostBoundaryBenchmarkError("PWM must have zero phase and 0 < duty < 1")
    if not switch.initial or diode.initial:
        raise InvertingBuckBoostBoundaryBenchmarkError(
            "cycle must start with switch on and diode off"
        )
    for component in (inductor, capacitor, load):
        if (
            component.value is None
            or not math.isfinite(float(component.value))
            or component.value <= 0.0
        ):
            raise InvertingBuckBoostBoundaryBenchmarkError(
                f"{component.id} must have a positive value"
            )
    return InvertingBuckBoostFixture(source, inductor, switch, diode, capacitor, load, pwm)
