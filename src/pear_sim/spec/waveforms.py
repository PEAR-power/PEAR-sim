"""Time-varying source waveforms (sine/step/pulse/PWL).

A waveform replaces the constant ``value`` of a voltage/current source with an
exact, deterministic function of time. Discontinuous waveforms expose their
breakpoints so the solver restarts integration precisely at each one — no
step is ever smeared across a discontinuity. Evaluation is closed-form; there
is no interpolation table hiding inside the kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "DoubleExponentialWaveform",
    "PwlWaveform",
    "PulseWaveform",
    "SineWaveform",
    "StepWaveform",
    "Waveform",
]

MAX_PWL_POINTS = 10_000
MAX_PULSE_BREAKPOINTS = 200_000


@dataclass(frozen=True)
class SineWaveform:
    """offset + amplitude * sin(2*pi*frequency*t + phase_rad)."""

    offset: float
    amplitude: float
    frequency_hz: float
    phase_rad: float

    kind = "sine"

    def value(self, time: float) -> float:
        return self.offset + self.amplitude * math.sin(
            2.0 * math.pi * self.frequency_hz * time + self.phase_rad
        )

    def breakpoints(self, start: float, stop: float) -> tuple[float, ...]:
        return ()

    def slope(self, time: float) -> float:
        omega = 2.0 * math.pi * self.frequency_hz
        return self.amplitude * omega * math.cos(omega * time + self.phase_rad)

    def as_dict(self) -> dict:
        return {
            "type": "sine",
            "offset": self.offset,
            "amplitude": self.amplitude,
            "frequency_hz": self.frequency_hz,
            "phase_rad": self.phase_rad,
        }


@dataclass(frozen=True)
class StepWaveform:
    """``initial`` before ``at_s``, ``final`` from ``at_s`` on (right-continuous)."""

    initial: float
    final: float
    at_s: float

    kind = "step"

    def value(self, time: float) -> float:
        return self.final if time >= self.at_s else self.initial

    def breakpoints(self, start: float, stop: float) -> tuple[float, ...]:
        if start < self.at_s < stop:
            return (self.at_s,)
        return ()

    def slope(self, time: float) -> float:
        return 0.0

    def as_dict(self) -> dict:
        return {
            "type": "step",
            "initial": self.initial,
            "final": self.final,
            "at_s": self.at_s,
        }


@dataclass(frozen=True)
class PulseWaveform:
    """Periodic rectangular wave: ``high`` for the first duty fraction."""

    low: float
    high: float
    frequency_hz: float
    duty: float
    phase_s: float

    kind = "pulse"

    def value(self, time: float) -> float:
        period = 1.0 / self.frequency_hz
        position = math.fmod(time - self.phase_s, period)
        if position < 0.0:
            position += period
        return self.high if position < self.duty * period else self.low

    def breakpoints(self, start: float, stop: float) -> tuple[float, ...]:
        period = 1.0 / self.frequency_hz
        first_cycle = math.floor((start - self.phase_s) / period)
        times: list[float] = []
        cycle = first_cycle
        while True:
            base = self.phase_s + cycle * period
            for candidate in (base, base + self.duty * period):
                if start < candidate < stop:
                    times.append(candidate)
            if base > stop:
                break
            cycle += 1
            if len(times) > MAX_PULSE_BREAKPOINTS:
                raise ValueError(
                    "pulse waveform produces more than "
                    f"{MAX_PULSE_BREAKPOINTS} breakpoints in the analysis window"
                )
        return tuple(times)

    def slope(self, time: float) -> float:
        return 0.0

    def as_dict(self) -> dict:
        return {
            "type": "pulse",
            "low": self.low,
            "high": self.high,
            "frequency_hz": self.frequency_hz,
            "duty": self.duty,
            "phase_s": self.phase_s,
        }


@dataclass(frozen=True)
class PwlWaveform:
    """Piecewise-linear points ((t, v), ...); endpoints held outside the range."""

    points: tuple[tuple[float, float], ...]

    kind = "pwl"

    def value(self, time: float) -> float:
        points = self.points
        if time <= points[0][0]:
            return points[0][1]
        if time >= points[-1][0]:
            return points[-1][1]
        for (t0, v0), (t1, v1) in zip(points, points[1:]):
            if t0 <= time <= t1:
                if t1 == t0:
                    return v1
                return v0 + (v1 - v0) * (time - t0) / (t1 - t0)
        raise AssertionError("unreachable: PWL points are sorted")

    def breakpoints(self, start: float, stop: float) -> tuple[float, ...]:
        return tuple(t for t, _ in self.points if start < t < stop)

    def slope(self, time: float) -> float:
        points = self.points
        if time <= points[0][0] or time >= points[-1][0]:
            return 0.0
        for (t0, v0), (t1, v1) in zip(points, points[1:]):
            if t0 <= time < t1:
                return (v1 - v0) / (t1 - t0)
        return 0.0

    def as_dict(self) -> dict:
        return {"type": "pwl", "points": [[t, v] for t, v in self.points]}


@dataclass(frozen=True)
class DoubleExponentialWaveform:
    """Single double-exponential pulse: zero, then rise/fall about ``start_s``.

    Shape ``exp(-t/tau_fall) - exp(-t/tau_rise)`` measured from ``start_s``,
    scaled so its extremum equals ``amplitude``. This is the standard
    circuit-level transient shape; here it is a *declared input waveform*
    supplied by the experimenter, not a quantity this simulator derives. The
    simulator has no opinion on what amplitude or width is physically correct
    for any device — that number comes from test data.

    ``tau_fall`` must strictly exceed ``tau_rise``; equal time constants
    collapse the shape to zero and the degenerate form is rejected rather than
    regularized.
    """

    amplitude: float
    tau_rise_s: float
    tau_fall_s: float
    start_s: float

    kind = "double_exponential"

    @property
    def _peak_offset_s(self) -> float:
        """Time from ``start_s`` to the extremum of the unscaled shape."""

        rise, fall = self.tau_rise_s, self.tau_fall_s
        return (rise * fall / (fall - rise)) * math.log(fall / rise)

    @property
    def _shape_peak(self) -> float:
        offset = self._peak_offset_s
        return math.exp(-offset / self.tau_fall_s) - math.exp(-offset / self.tau_rise_s)

    def value(self, time: float) -> float:
        elapsed = time - self.start_s
        if elapsed <= 0.0:
            return 0.0
        shape = math.exp(-elapsed / self.tau_fall_s) - math.exp(
            -elapsed / self.tau_rise_s
        )
        return self.amplitude * shape / self._shape_peak

    def breakpoints(self, start: float, stop: float) -> tuple[float, ...]:
        # The onset is the only discontinuity in the derivative; the decay is
        # smooth and the integrator handles it without help.
        if start < self.start_s < stop:
            return (self.start_s,)
        return ()

    def slope(self, time: float) -> float:
        elapsed = time - self.start_s
        if elapsed <= 0.0:
            return 0.0
        derivative = (
            math.exp(-elapsed / self.tau_rise_s) / self.tau_rise_s
            - math.exp(-elapsed / self.tau_fall_s) / self.tau_fall_s
        )
        return self.amplitude * derivative / self._shape_peak

    def as_dict(self) -> dict:
        return {
            "type": "double_exponential",
            "amplitude": self.amplitude,
            "tau_rise_s": self.tau_rise_s,
            "tau_fall_s": self.tau_fall_s,
            "start_s": self.start_s,
            "peak_time_s": self.start_s + self._peak_offset_s,
        }


Waveform = (
    SineWaveform
    | StepWaveform
    | PulseWaveform
    | PwlWaveform
    | DoubleExponentialWaveform
)
