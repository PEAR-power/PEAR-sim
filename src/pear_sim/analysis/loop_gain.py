# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Loop gain of a closed-loop SWITCHING model, measured by injection.

WHY THIS EXISTS. `analysis/small_signal.py` forms a loop gain from an
AVERAGED plant (one leg, CCM) and a compensator the caller declares a
second time beside the model. That is fast, but it is two approximations
away from the converter: the averaging, and the hope that the declared
compensator is the one the model actually runs. This module measures the
loop the model actually closes, on the switching circuit, the way a bench
does it.

THE METHOD is Middlebrook's injection. At a chosen signal block ``x`` the
loop is opened logically, not physically: every consumer of ``x`` is
rewired to read ``y = x + A sin(2 pi f t)`` instead. The loop stays closed,
so the orbit stays where it is; the injected sinusoid circulates once
around the loop and comes back as the response in ``x``. With both
recorded,

    L(f) = -X(f) / Y(f)

is the loop gain seen from that point (negative-feedback convention: a
well-behaved loop has L > 0 at DC, crosses |L| = 1 with a phase above
-180 degrees). The minus sign is the comparison's sign; it is not a fudge.

WHY THE CONTROLLER'S OWN SAMPLES. The signal layer runs once per tick, so
``x`` and ``y`` are sequences, not waveforms. Fitting the phasor on the
tick samples (least squares on [1, sin, cos] at f) reads the discrete loop
gain L(e^{j 2 pi f T}) the sampled controller actually lives in. The orbit
is periodic with the tick, so its samples are constant: the switching
ripple never reaches the fit at all — there is nothing to filter.

THE START IS THE ORBIT. The run begins on the closed-loop periodic orbit
(`analysis/closed_loop.py`), so the only transient is the one the
injection itself excites. It decays with the closed loop's own
multipliers, which the orbit solve has already computed: the settle time
is ceil(log(tolerance) / log(max |mu|)) cycles, derived, not guessed.
Measured on the canonical buck (|mu| 0.9896): 600 cycles left a 3e-4
error in the gain margin; 1500 cycles, 1.5e-8.

WHAT IS CLAIMED. The small-signal loop gain of the computed orbit at the
injection point. A clamped limiter during the run is refused, not
reported: a clamped loop is not a small-signal loop. An unstable or
unresolved orbit is refused too — a closed loop cannot settle around it,
exactly as on a bench; its multipliers are `analyze floquet`'s answer.

THE GATE IS A DIFFERENT METHOD. Scaling the controller by the measured gain
margin must put a Floquet multiplier on the unit circle at the measured
phase-crossover frequency (tests/test_loop_gain.py). One side is a
time-domain sinusoidal measurement, the other eigenvalues of a saltation
monodromy; they share the simulator kernel, so this is verification, not
validation against hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Iterable

import numpy as np
from scipy.optimize import brentq

from pear_sim.analysis.closed_loop import (
    ClosedLoopSteadyState,
    _cycle_model,
    _plan,
    solve_closed_loop_steady_state,
)
from pear_sim.errors import ModelError
from pear_sim.limits import MAX_RUN_EVENT_ROWS
from pear_sim.spec.model import Model
from pear_sim.spec.signal_graph import LOGICAL_UNIT, SignalBlock, _resolve_order
from pear_sim.spec.signals import declared_signal_target

DRIVE_ID = "_lg_drive"
INJECTED_ID = "_lg_inj"
DEFAULT_SETTLE_TOLERANCE = 1e-6
DEFAULT_MIN_WINDOW_TICKS = 200
DEFAULT_MIN_WINDOW_PERIODS = 4
#: Default amplitude as a fraction of the orbit's distance to the nearest
#: limiter. Measured on the canonical buck (margin 0.45): 1e-2 of it (duty
#: 4.5e-3) moved the switching mode sequence near the LC resonance and read
#: a gain margin of 3.42 instead of 2.45; duty 1e-3..1e-5 agreed to 1e-8.
#: The mode-sequence check below refuses the first case whatever the default.
DEFAULT_AMPLITUDE_FRACTION = 1e-3
#: Relative frequency tolerance of the refined crossovers.
REFINE_RTOL = 1e-7


class LoopGainError(ModelError):
    """Raised when a loop gain cannot be measured as requested."""

    code = "analysis.loop_gain_invalid"


@dataclass(frozen=True)
class LoopGainPoint:
    frequency_hz: float
    real: float
    imag: float
    #: RMS of the least-squares residual of the returned signal relative to
    #: its fitted amplitude: harmonics, leftover transient, round-off. Not a
    #: gate — a number the caller can watch move with the amplitude.
    fit_residual_rel: float

    @property
    def value(self) -> complex:
        return complex(self.real, self.imag)

    @property
    def gain(self) -> float:
        return abs(self.value)

    @property
    def phase_deg(self) -> float:
        return float(np.degrees(np.angle(self.value)))

    def as_dict(self) -> dict[str, Any]:
        return {
            "frequency_hz": self.frequency_hz,
            "gain": self.gain,
            "gain_db": 20.0 * math.log10(self.gain) if self.gain > 0 else None,
            "phase_deg": self.phase_deg,
            "real": self.real,
            "imag": self.imag,
            "fit_residual_rel": self.fit_residual_rel,
        }


@dataclass(frozen=True)
class GainCrossover:
    frequency_hz: float
    phase_deg: float

    @property
    def phase_margin_deg(self) -> float:
        return 180.0 + self.phase_deg

    def as_dict(self) -> dict[str, Any]:
        return {
            "frequency_hz": self.frequency_hz,
            "phase_deg": self.phase_deg,
            "phase_margin_deg": self.phase_margin_deg,
        }


@dataclass(frozen=True)
class PhaseCrossover:
    frequency_hz: float
    gain: float

    @property
    def gain_margin(self) -> float:
        return 1.0 / self.gain

    def as_dict(self) -> dict[str, Any]:
        return {
            "frequency_hz": self.frequency_hz,
            "gain": self.gain,
            "gain_margin": self.gain_margin,
            "gain_margin_db": 20.0 * math.log10(self.gain_margin),
        }


@dataclass(frozen=True)
class LoopGainResult:
    injection_block: str
    amplitude: float
    period_s: float
    settle_cycles: int
    orbit: ClosedLoopSteadyState
    points: tuple[LoopGainPoint, ...]
    gain_crossovers: tuple[GainCrossover, ...]
    phase_crossovers: tuple[PhaseCrossover, ...]
    refined: bool
    notes: tuple[str, ...] = ()

    @property
    def phase_margin_deg(self) -> float | None:
        if not self.gain_crossovers:
            return None
        return min(item.phase_margin_deg for item in self.gain_crossovers)

    @property
    def gain_margin(self) -> float | None:
        if not self.phase_crossovers:
            return None
        return min(item.gain_margin for item in self.phase_crossovers)

    def as_dict(self) -> dict[str, Any]:
        margin = self.gain_margin
        return {
            "analysis": "loop_gain_by_injection",
            "method": (
                "Middlebrook injection at a signal block on the switching model, "
                "started on the closed-loop periodic orbit; phasors fitted on the "
                "controller's tick samples"
            ),
            "convention": "L = -X/Y; X = block value returning, Y = X + injection",
            "claim": (
                "small-signal loop gain of the computed orbit at the injection "
                "point; verification against Floquet multipliers, not validation "
                "against hardware"
            ),
            "injection_block": self.injection_block,
            "amplitude": self.amplitude,
            "period_s": self.period_s,
            "settle_cycles": self.settle_cycles,
            "orbit_max_multiplier_magnitude": self.orbit.max_multiplier_magnitude,
            "refined": self.refined,
            "phase_margin_deg": self.phase_margin_deg,
            "gain_margin": margin,
            "gain_margin_db": None if margin is None else 20.0 * math.log10(margin),
            "gain_crossovers": [item.as_dict() for item in self.gain_crossovers],
            "phase_crossovers": [item.as_dict() for item in self.phase_crossovers],
            "points": [item.as_dict() for item in self.points],
            "notes": list(self.notes),
        }


def default_injection_block(model: Model) -> str:
    """The block the PWM duty reads, when there is exactly one bridge."""

    bridges = [
        control
        for control in model.sampled_controls.values()
        if getattr(control, "kind", None) == "signal_pwm"
    ]
    if len(bridges) != 1:
        raise LoopGainError(
            f"the model has {len(bridges)} signal_pwm controls; name the injection "
            "block explicitly (inject_at)"
        )
    target = declared_signal_target(bridges[0].duty_signal)
    if target is None:
        raise LoopGainError(
            f"signal_pwm duty {bridges[0].duty_signal!r} does not name a signal block"
        )
    return target


def injected_model(
    base: Model, *, block: str, frequency_hz: float, amplitude: float, stop_s: float
) -> Model:
    """``base`` with every consumer of ``block`` reading ``block + A sin``."""

    graph = base.signals
    if graph is None:
        raise LoopGainError("the model has no signal layer to inject into")
    by_id = graph.by_id
    if block not in by_id:
        raise LoopGainError(
            f"injection block {block!r} is not a signal block; blocks: "
            f"{sorted(by_id)!r}"
        )
    if by_id[block].unit == LOGICAL_UNIT:
        raise LoopGainError(
            f"injection block {block!r} carries a condition, not a number"
        )
    for reserved in (DRIVE_ID, INJECTED_ID):
        if reserved in by_id:
            raise LoopGainError(f"signal block id {reserved!r} is reserved")
    consumers = [item.id for item in graph.blocks if block in item.inputs]
    expression = f"s({block})"
    bridges = [
        control_id
        for control_id, control in base.sampled_controls.items()
        if getattr(control, "duty_signal", None) == expression
    ]
    if not consumers and not bridges:
        raise LoopGainError(
            f"nothing reads block {block!r}: injecting there opens no loop"
        )
    unit = by_id[block].unit
    blocks = tuple(
        replace(item, inputs=tuple(INJECTED_ID if name == block else name for name in item.inputs))
        for item in graph.blocks
    ) + (
        SignalBlock(
            id=DRIVE_ID,
            kind="sine",
            unit=unit,
            params={
                "amplitude": amplitude,
                "frequency": frequency_hz,
                "phase_rad": 0.0,
                "offset": 0.0,
            },
        ),
        SignalBlock(
            id=INJECTED_ID,
            kind="sum",
            unit=unit,
            inputs=(block, DRIVE_ID),
            params={"signs": (1.0, 1.0)},
        ),
    )
    controls = dict(base.sampled_controls)
    for control_id in bridges:
        controls[control_id] = replace(controls[control_id], duty_signal=f"s({INJECTED_ID})")
    return replace(
        base,
        signals=replace(graph, blocks=blocks, order=_resolve_order(blocks)),
        sampled_controls=controls,
        transient=replace(base.transient, start=0.0, stop=stop_s),
    )


def _phasor(t: np.ndarray, values: np.ndarray, frequency_hz: float) -> tuple[complex, float]:
    """Least-squares fit of c + a sin + b cos; returns (a + jb, residual RMS)."""

    omega_t = 2.0 * np.pi * frequency_hz * t
    basis = np.column_stack([np.ones_like(t), np.sin(omega_t), np.cos(omega_t)])
    coefficients, *_ = np.linalg.lstsq(basis, values, rcond=None)
    residual = values - basis @ coefficients
    return complex(coefficients[1], coefficients[2]), float(np.sqrt(np.mean(residual**2)))


def _cycle_sequences(
    intervals: list[dict[str, Any]], period: float
) -> dict[int, tuple[str, ...]]:
    """Switching mode sequence of every cycle, keyed by cycle index."""

    cycles: dict[int, list[str]] = {}
    for interval in intervals:
        start, end = float(interval["start_s"]), float(interval["end_s"])
        # Slivers (a segment closed by a tick a few ulps after a switching
        # edge) carry no dynamics and would land in either cycle by rounding.
        if end - start <= 1e-9 * period:
            continue
        index = int(math.floor(0.5 * (start + end) / period))
        sequence = cycles.setdefault(index, [])
        signature = str(interval["mode_signature"])
        if not sequence or sequence[-1] != signature:
            sequence.append(signature)
    return {index: tuple(sequence) for index, sequence in cycles.items()}


def _limiter_violations(model: Model, rows: list[dict[str, Any]]) -> list[str]:
    graph = model.signals
    found = []
    for item in graph.blocks:  # type: ignore[union-attr]
        if item.kind != "saturate":
            continue
        source = item.inputs[0]
        values = [row["values"][source] for row in rows]
        low, high = float(item.params["min"]), float(item.params["max"])
        if min(values) <= low or max(values) >= high:
            found.append(
                f"saturate {item.id!r} input {source!r} spans "
                f"[{min(values):.6g}, {max(values):.6g}] against [{low:.6g}, {high:.6g}]"
            )
    for control_id, control in model.sampled_controls.items():
        source = declared_signal_target(control.duty_signal)
        if source is None:
            continue
        values = [row["values"][source] for row in rows]
        if min(values) <= control.duty_min or max(values) >= control.duty_max:
            found.append(
                f"signal_pwm {control_id!r} duty spans [{min(values):.6g}, "
                f"{max(values):.6g}] against [{control.duty_min:.6g}, {control.duty_max:.6g}]"
            )
    return found


class _Bench:
    """One orbit, many frequencies."""

    def __init__(
        self,
        model: Model,
        *,
        block: str,
        amplitude: float,
        orbit: ClosedLoopSteadyState,
        settle_tolerance: float,
        min_window_ticks: int,
        min_window_periods: int,
    ) -> None:
        plan = _plan(model)
        self.base = _cycle_model(plan, np.asarray(orbit.state, dtype=float))
        self.block = block
        self.amplitude = amplitude
        self.period = plan.period
        largest = orbit.max_multiplier_magnitude
        assert largest is not None and largest < 1.0
        self.settle_cycles = max(
            1, int(math.ceil(math.log(settle_tolerance) / math.log(largest)))
        )
        self.min_window_ticks = min_window_ticks
        self.min_window_periods = min_window_periods
        self.cache: dict[float, LoopGainPoint] = {}
        # The orbit's own per-cycle mode sequence, read off an unperturbed run
        # of a few cycles by the same splitting the injected runs get.
        from pear_sim.core.solver import HybridSimulator

        quiet = HybridSimulator(
            replace(self.base, transient=replace(self.base.transient, stop=3.0 * self.period))
        ).simulate()
        self.orbit_sequence = _cycle_sequences(quiet.mode_intervals, self.period)[1]

    def measure(self, frequency_hz: float) -> LoopGainPoint:
        from pear_sim.core.solver import HybridSimulator

        frequency_hz = float(frequency_hz)
        if frequency_hz in self.cache:
            return self.cache[frequency_hz]
        nyquist = 0.5 / self.period
        if not 0.0 < frequency_hz < nyquist:
            raise LoopGainError(
                f"injection at {frequency_hz:.6g} Hz is outside (0, {nyquist:.6g}) Hz: "
                "the controller samples once per cycle and cannot carry it"
            )
        window = max(
            self.min_window_ticks,
            int(math.ceil(self.min_window_periods / (frequency_hz * self.period))),
        )
        total = self.settle_cycles + window
        if total + 1 > MAX_RUN_EVENT_ROWS:
            raise LoopGainError(
                f"{total} controller ticks exceed the {MAX_RUN_EVENT_ROWS}-row trace "
                f"(settle {self.settle_cycles} + window {window}); raise the lowest "
                "frequency or the settle tolerance"
            )
        model = injected_model(
            self.base,
            block=self.block,
            frequency_hz=frequency_hz,
            amplitude=self.amplitude,
            stop_s=(total + 0.5) * self.period,
        )
        model = replace(
            model, transient=replace(model.transient, output_step=self.period)
        )
        # Lean evidence: the trajectory is the same; only the per-interval
        # extrema audit (about half the run time, never read here) is skipped.
        result = HybridSimulator(model, evidence_mode="lean").simulate()
        rows = [row for row in result.signal_trace if row["tick_index"] >= self.settle_cycles]
        rows = rows[:window]
        if len(rows) != window:
            raise LoopGainError(
                f"expected {window} controller ticks in the window, recorded {len(rows)}"
            )
        clamped = _limiter_violations(model, result.signal_trace)
        if clamped:
            raise LoopGainError(
                f"a limiter engaged during the {frequency_hz:.6g} Hz injection ("
                + "; ".join(clamped)
                + "): a clamped loop is not a small-signal loop; lower the amplitude"
            )
        # A small-signal loop gain is the linearisation around the orbit's
        # switching pattern. An injection that adds or drops a mode (a diode
        # starting to conduct, CCM tipping into DCM) is outside it, and no
        # limiter would say so: the gain read there is a describing function.
        sequences = _cycle_sequences(result.mode_intervals, self.period)
        for index in range(total):
            if sequences.get(index) != self.orbit_sequence:
                raise LoopGainError(
                    f"the {frequency_hz:.6g} Hz injection changed the switching mode "
                    f"sequence in cycle {index} ({len(self.orbit_sequence)} modes on the "
                    f"orbit, {len(sequences.get(index, ()))} there): the response is not "
                    "small-signal; lower the amplitude"
                )
        t = np.array([row["time_s"] for row in rows])
        returned, residual = _phasor(
            t, np.array([row["values"][self.block] for row in rows]), frequency_hz
        )
        sent, _ = _phasor(
            t, np.array([row["values"][INJECTED_ID] for row in rows]), frequency_hz
        )
        value = -returned / sent
        point = LoopGainPoint(
            frequency_hz=frequency_hz,
            real=float(value.real),
            imag=float(value.imag),
            fit_residual_rel=residual / abs(returned) if abs(returned) > 0 else float("inf"),
        )
        self.cache[frequency_hz] = point
        return point


def _refine(function, low: float, high: float) -> float:
    return float(brentq(function, low, high, xtol=REFINE_RTOL * low, rtol=REFINE_RTOL))


def measure_loop_gain(
    model: Model,
    frequencies_hz: Iterable[float],
    *,
    inject_at: str | None = None,
    amplitude: float | None = None,
    initial_state: dict[str, float] | None = None,
    orbit: ClosedLoopSteadyState | None = None,
    refine: bool = True,
    settle_tolerance: float = DEFAULT_SETTLE_TOLERANCE,
    min_window_ticks: int = DEFAULT_MIN_WINDOW_TICKS,
    min_window_periods: int = DEFAULT_MIN_WINDOW_PERIODS,
) -> LoopGainResult:
    """Loop gain at ``inject_at`` over ``frequencies_hz``, plus its margins.

    With ``refine`` the gain and phase crossovers bracketed by the grid are
    located by Brent's method on further injections, so the margins do not
    inherit the grid spacing.
    """

    grid = sorted({float(value) for value in frequencies_hz})
    if len(grid) < 2:
        raise LoopGainError("a loop-gain sweep needs at least two frequencies")
    if not 0.0 < settle_tolerance < 1.0:
        raise LoopGainError("settle_tolerance must lie in (0, 1)")
    block = inject_at or default_injection_block(model)
    if orbit is None:
        orbit = solve_closed_loop_steady_state(model, initial_state=initial_state)
    if orbit.stability != "stable":
        detail = "; ".join(orbit.notes) or (
            f"largest |mu| {orbit.max_multiplier_magnitude!r}"
        )
        raise LoopGainError(
            f"the closed-loop orbit is {orbit.stability} ({detail}); an injection "
            "cannot settle around it, exactly as on a bench. Its multipliers are "
            "the answer: analyze floquet"
        )
    if amplitude is None:
        margin = next(
            (item.margin for item in orbit.assumptions if item.name == "no_limiter_engaged"),
            None,
        )
        if margin is None or not math.isfinite(margin) or margin <= 0.0:
            raise LoopGainError("no limiter margin to size the amplitude; pass amplitude")
        amplitude = DEFAULT_AMPLITUDE_FRACTION * margin
    if amplitude <= 0.0:
        raise LoopGainError("injection amplitude must be positive")

    bench = _Bench(
        model,
        block=block,
        amplitude=float(amplitude),
        orbit=orbit,
        settle_tolerance=settle_tolerance,
        min_window_ticks=min_window_ticks,
        min_window_periods=min_window_periods,
    )
    points = [bench.measure(frequency) for frequency in grid]

    gain_crossovers: list[GainCrossover] = []
    phase_crossovers: list[PhaseCrossover] = []
    for left, right in zip(points, points[1:]):
        if (left.gain - 1.0) * (right.gain - 1.0) < 0.0:
            frequency = (
                _refine(
                    lambda f: math.log(bench.measure(f).gain),
                    left.frequency_hz,
                    right.frequency_hz,
                )
                if refine
                else _interpolate_log(left, right, lambda p: math.log(p.gain))
            )
            point = bench.measure(frequency) if refine else None
            phase = point.phase_deg if point else _lerp_phase(left, right, frequency)
            gain_crossovers.append(GainCrossover(frequency_hz=frequency, phase_deg=phase))
    # The negative real axis, found on the UNWRAPPED phase: between two grid
    # points the phase may sweep through a resonance by more than 90 degrees,
    # so "Im changes sign while Re stays negative" misses crossings a coarse
    # grid straddles. Brent then solves Im(L) = 0 inside the bracket and the
    # root is kept only if Re(L) < 0 there (a phase of -180, not 0).
    unwrapped = np.degrees(np.unwrap(np.radians([item.phase_deg for item in points])))
    notes: list[str] = []
    for index, (left, right) in enumerate(zip(points, points[1:])):
        low, high = sorted((unwrapped[index], unwrapped[index + 1]))
        first = math.ceil((low + 180.0) / 360.0)
        last = math.floor((high + 180.0) / 360.0)
        for _ in range(first, last + 1):
            if refine:
                frequency = _refine(
                    lambda f: bench.measure(f).imag, left.frequency_hz, right.frequency_hz
                )
                point = bench.measure(frequency)
                if point.real >= 0.0:
                    notes.append(
                        f"phase bracket {left.frequency_hz:.6g}-{right.frequency_hz:.6g} Hz "
                        "resolved to Re(L) >= 0; grid too coarse there, add points"
                    )
                    continue
                gain = point.gain
            else:
                frequency = _interpolate_log(left, right, lambda p: p.imag)
                gain = math.exp(_lerp(left, right, frequency, lambda p: math.log(p.gain)))
            phase_crossovers.append(PhaseCrossover(frequency_hz=frequency, gain=gain))

    return LoopGainResult(
        injection_block=block,
        amplitude=float(amplitude),
        period_s=bench.period,
        settle_cycles=bench.settle_cycles,
        orbit=orbit,
        points=tuple(points),
        gain_crossovers=tuple(gain_crossovers),
        phase_crossovers=tuple(phase_crossovers),
        refined=refine,
        notes=tuple(notes),
    )


def _lerp(left: LoopGainPoint, right: LoopGainPoint, frequency: float, value) -> float:
    position = math.log(frequency / left.frequency_hz) / math.log(
        right.frequency_hz / left.frequency_hz
    )
    return value(left) + position * (value(right) - value(left))


def _interpolate_log(left: LoopGainPoint, right: LoopGainPoint, value) -> float:
    a, b = value(left), value(right)
    position = a / (a - b)
    return float(
        math.exp(
            math.log(left.frequency_hz)
            + position * math.log(right.frequency_hz / left.frequency_hz)
        )
    )


def _lerp_phase(left: LoopGainPoint, right: LoopGainPoint, frequency: float) -> float:
    unwrapped = np.unwrap(np.radians([left.phase_deg, right.phase_deg]))
    position = math.log(frequency / left.frequency_hz) / math.log(
        right.frequency_hz / left.frequency_hz
    )
    return float(np.degrees(unwrapped[0] + position * (unwrapped[1] - unwrapped[0])))
