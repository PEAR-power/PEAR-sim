# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Independent dead-time oracle for the three-phase SPWM inverter (VAL-005).

WHAT IT SOLVES. A two-level, split-bus inverter whose RL star load is tied to
the DC midpoint, driven by the offline-midpoint-sampled, center-aligned SPWM
law (optionally with min-max injection) WITH dead time. Tying the star to the
midpoint decouples the three phases, so each leg is a first-order RL circuit
driven by a piecewise-constant pole voltage, and the whole run has a closed
form interval by interval.

WHY IT IS INDEPENDENT. Nothing here imports the solver, the circuit builder or
the scheduler. The modulator law is restated from its documented definition
(docs/model-format.md, three_phase_spwm), the pole voltage during each dead
time is decided by the sign of the load current, and the current is advanced
by an exact exponential. The kernel reaches the same answers through MNA, an
ideal-diode active set and an adaptive Runge-Kutta integrator; the only thing
the two share is the circuit definition.

THE CASE THE HALF-BRIDGE GOLDEN EXCLUDES. A dead time is easy while the load
current keeps one sign: one body diode owns the whole gap. Near every current
zero crossing of an inverter the switching ripple makes the current reverse
INSIDE the gap. Then the conducting diode is starved, the current is held at
zero, and the pole floats to the load side (0 V here, because the star sits on
the midpoint). This oracle resolves each of those events exactly:

- current leaving the pole (i > 0) pins the pole to the negative rail through
  the lower diode, and vice versa;
- a current that reaches zero inside the gap stays at zero for the rest of it
  (no diode can carry the opposite sign), with the pole at the load voltage;
- the crossing instant is solved in closed form, s = tau*ln(1 + |i0|*R/Vh).

Fourier coefficients use the producer's convention,
C_k = (1/W) * integral over [start, stop) of x(t) * exp(-j*k*w1*t) dt,
integrated analytically over every piece, so the THD below is a reference
value and not a second estimate of the same samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

SPWM_DEAD_TIME_ORACLE_SCHEMA = "pear-sim/spwm-dead-time-oracle/v0.1"

Injection = Literal["none", "min_max"]


class SpwmDeadTimeOracleError(ValueError):
    """Raised for a case the closed form below does not describe."""


@dataclass(frozen=True)
class SpwmDeadTimeCase:
    """One inverter operating point, in SI units."""

    half_bus_v: float
    fundamental_hz: float
    carrier_ratio: int
    modulation_index: float
    dead_time_s: float
    phase_r_ohm: float
    phase_l_h: float
    stop_s: float
    injection: Injection = "none"

    def __post_init__(self) -> None:
        if self.carrier_ratio < 2:
            raise SpwmDeadTimeOracleError("carrier ratio must be at least two")
        if self.injection not in ("none", "min_max"):
            raise SpwmDeadTimeOracleError(f"unknown injection {self.injection!r}")
        limit = 2.0 / math.sqrt(3.0) if self.injection == "min_max" else 1.0
        if not 0.0 <= self.modulation_index < limit:
            raise SpwmDeadTimeOracleError("modulation index outside the linear range")
        if self.dead_time_s < 0.0 or self.phase_r_ohm <= 0.0 or self.phase_l_h <= 0.0:
            raise SpwmDeadTimeOracleError("dead time, R and L must be physical")

    @property
    def carrier_period_s(self) -> float:
        return 1.0 / (self.fundamental_hz * self.carrier_ratio)

    @property
    def tau_s(self) -> float:
        return self.phase_l_h / self.phase_r_ohm


def leg_duties(case: SpwmDeadTimeCase) -> np.ndarray:
    """Duty of the three high switches in each carrier cell of one period.

    The law as documented: the references are sampled at the MIDDLE of each
    cell, abc sequence, optional min-max centering -(max+min)/2, and
    duty = (1 + m*reference)/2. Shape (carrier_ratio, 3).
    """

    duties = np.empty((case.carrier_ratio, 3))
    for cell in range(case.carrier_ratio):
        angle = 2.0 * math.pi * (cell + 0.5) / case.carrier_ratio
        refs = [
            math.sin(angle),
            math.sin(angle - 2.0 * math.pi / 3.0),
            math.sin(angle + 2.0 * math.pi / 3.0),
        ]
        if case.injection == "min_max":
            offset = -0.5 * (max(refs) + min(refs))
            refs = [value + offset for value in refs]
        duties[cell] = [0.5 * (1.0 + case.modulation_index * value) for value in refs]
    return duties


@dataclass(frozen=True)
class PhaseSolution:
    """One leg as a list of pieces: on [start, stop) the pole sits at
    ``pole`` and the current is settled + (initial - settled)*exp(-(t-start)/tau)."""

    start: np.ndarray
    stop: np.ndarray
    pole: np.ndarray
    initial: np.ndarray
    settled: np.ndarray
    tau_s: float
    reversal_times_s: tuple[float, ...]
    dead_time_clamped_s: float

    @property
    def dead_time_reversals(self) -> int:
        return len(self.reversal_times_s)

    def current(self, times: np.ndarray) -> np.ndarray:
        times = np.asarray(times, dtype=float)
        index = np.clip(np.searchsorted(self.start, times, side="right") - 1, 0, None)
        elapsed = times - self.start[index]
        return self.settled[index] + (self.initial[index] - self.settled[index]) * np.exp(
            -elapsed / self.tau_s
        )

    def _clipped(self, start: float, stop: float):
        a = np.maximum(self.start, start)
        b = np.minimum(self.stop, stop)
        keep = b > a
        return a[keep], b[keep], keep

    def pole_coefficients(self, start: float, stop: float, omega: float, orders: int) -> np.ndarray:
        """C_0..C_orders of the pole voltage over [start, stop)."""

        a, b, keep = self._clipped(start, stop)
        pole = self.pole[keep]
        out = np.empty(orders + 1, dtype=complex)
        out[0] = np.sum(pole * (b - a))
        for k in range(1, orders + 1):
            w = k * omega
            out[k] = np.sum(pole * (np.exp(-1j * w * a) - np.exp(-1j * w * b)) / (1j * w))
        return out / (stop - start)

    def current_coefficients(self, start: float, stop: float, omega: float, orders: int) -> np.ndarray:
        """C_0..C_orders of the leg current over [start, stop)."""

        a, b, keep = self._clipped(start, stop)
        t0 = self.start[keep]
        settled = self.settled[keep]
        # Amplitude of the decaying part referred to the clipped start a.
        amplitude = (self.initial[keep] - settled) * np.exp(-(a - t0) / self.tau_s)
        out = np.empty(orders + 1, dtype=complex)
        for k in range(orders + 1):
            w = k * omega
            p = 1.0 / self.tau_s + 1j * w
            decaying = amplitude * np.exp(-1j * w * a) * (1.0 - np.exp(-p * (b - a))) / p
            if k == 0:
                constant = settled * (b - a)
            else:
                constant = settled * (np.exp(-1j * w * a) - np.exp(-1j * w * b)) / (1j * w)
            out[k] = np.sum(constant + decaying)
        return out / (stop - start)


def _advance(initial: float, settled: float, elapsed: float, tau: float) -> float:
    return settled + (initial - settled) * math.exp(-elapsed / tau)


def solve_phase(
    case: SpwmDeadTimeCase,
    phase: int,
    *,
    initial_current_a: float = 0.0,
    _mutation: str | None = None,
) -> PhaseSolution:
    """Exact trajectory of one leg from t = 0 to ``case.stop_s``.

    ``_mutation`` exists only so the test suite can prove the comparison
    rings when the physics is wrong; it is never used to produce evidence.
    """

    duties = leg_duties(case)[:, phase]
    period = case.carrier_period_s
    vh = case.half_bus_v
    r = case.phase_r_ohm
    tau = case.tau_s
    td = 0.0 if _mutation == "no_dead_time" else case.dead_time_s
    cells = int(math.ceil(case.stop_s / period - 1e-9))

    starts: list[float] = []
    stops: list[float] = []
    poles: list[float] = []
    initials: list[float] = []
    settleds: list[float] = []
    reversals: list[float] = []
    clamped = 0.0
    current = float(initial_current_a)

    def push(start: float, stop: float, pole: float) -> None:
        nonlocal current
        if stop <= start:
            return
        settled = pole / r
        starts.append(start)
        stops.append(stop)
        poles.append(pole)
        initials.append(current)
        settleds.append(settled)
        current = _advance(current, settled, stop - start, tau)

    def dead(start: float, stop: float) -> None:
        nonlocal current, reversals, clamped
        if stop <= start:
            return
        if current == 0.0:
            push(start, stop, 0.0)
            clamped += stop - start
            return
        sign = 1.0 if current > 0.0 else -1.0
        # Current leaving the pole pins it to the NEGATIVE rail (lower diode).
        pole = -sign * vh if _mutation != "flipped_dead_time_pole" else sign * vh
        if _mutation == "no_zero_clamp" or _mutation == "flipped_dead_time_pole":
            push(start, stop, pole)
            return
        crossing = tau * math.log1p(abs(current) * r / vh)
        if start + crossing >= stop:
            push(start, stop, pole)
            return
        push(start, start + crossing, pole)
        current = 0.0  # exactly zero: the diode is starved, nothing reverses it
        reversals.append(start + crossing)
        push(start + crossing, stop, 0.0)
        clamped += stop - (start + crossing)

    for cell in range(cells):
        cell_start = cell * period
        duty = duties[cell % case.carrier_ratio]
        command_on = cell_start + 0.5 * (1.0 - duty) * period
        command_off = cell_start + 0.5 * (1.0 + duty) * period
        cell_stop = cell_start + period
        # low switch -> dead gap -> high switch -> dead gap -> low switch
        push(cell_start, command_on, -vh)
        dead(command_on, command_on + td)
        push(command_on + td, command_off, +vh)
        dead(command_off, command_off + td)
        push(command_off + td, cell_stop, -vh)

    return PhaseSolution(
        start=np.asarray(starts),
        stop=np.asarray(stops),
        pole=np.asarray(poles),
        initial=np.asarray(initials),
        settled=np.asarray(settleds),
        tau_s=tau,
        reversal_times_s=tuple(reversals),
        dead_time_clamped_s=clamped,
    )


def solve(case: SpwmDeadTimeCase, *, _mutation: str | None = None) -> tuple[PhaseSolution, ...]:
    """All three legs (abc), each from zero initial current."""

    return tuple(solve_phase(case, phase, _mutation=_mutation) for phase in range(3))


def thd_fraction(coefficients: np.ndarray) -> float:
    """Bandwidth-limited THD over harmonics 2..K, DC excluded (producer policy)."""

    peaks = 2.0 * np.abs(coefficients)
    return float(math.sqrt(float(np.sum(peaks[2:] ** 2))) / peaks[1])


def leading_order_fundamental_error(case: SpwmDeadTimeCase, current_phase_rad: float) -> complex:
    """Textbook dead-time voltage error on the phase fundamental, as C_1.

    Every carrier cycle loses Td*Vdc volt-seconds against the current sign
    (Vdc = 2*Vh), so the pole error is -Td*fc*Vdc*sign(i): a square wave in
    phase with the current whose fundamental peak is (4/pi)*Td*fc*Vdc. Ignores
    the ripple around the zero crossings, which is exactly what the exact
    solution above resolves; the gap between the two is the size of that
    effect, not an error of either.
    """

    carrier_hz = case.fundamental_hz * case.carrier_ratio
    peak = 4.0 / math.pi * case.dead_time_s * carrier_hz * 2.0 * case.half_bus_v
    # i = |I| sin(w t + phi)  <=>  C_1 = |I|/2 * exp(j*(phi - pi/2))
    return -0.5 * peak * complex(math.cos(current_phase_rad - math.pi / 2.0),
                                 math.sin(current_phase_rad - math.pi / 2.0))
