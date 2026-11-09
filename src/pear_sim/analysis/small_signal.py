# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Cycle-averaged small-signal analysis and loop-gain margins.

State-space averaging over the two topologies of a CCM converter at the
periodic operating point found by :mod:`pear_sim.analysis.steady_state`:

    x_dot = [d*A_on + (1-d)*A_off] x + [d*c_on + (1-d)*c_off]

Duty perturbation enters through B_d = (A_on - A_off) x_bar + (c_on - c_off)
with the matching output feedthrough, giving the control-to-output transfer
function G_yd(s). The kernel has no closed-loop control blocks yet, so the
loop gain is formed with a *caller-declared* compensator C(s) and sensor gain
H: L(s) = C(s) * G_yd(s) * H. Margins are therefore an auditable statement
about a declared design, never an inferred one. A declared sampling model
folds the digital controller's ZOH and computation delay into L(s);
without one the result is continuous and flagged ``sampling: not_modelled``.

Validity boundary (fail closed): exactly one modulated control period, a
two-topology CCM cycle with zero natural events, and frequencies well below
the switching frequency. Averaged models are meaningless near f_sw; the
producer records the f_sw/10 guideline alongside the grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pear_sim.analysis.steady_state import PeriodicSteadyState, one_cycle_model
from pear_sim.core.circuit import Circuit
from pear_sim.core.metrics import _output_affine_map
from pear_sim.errors import ModelError
from pear_sim.spec.model import Model

__all__ = [
    "AveragedSmallSignal",
    "LoopGainResult",
    "SamplingModel",
    "SmallSignalError",
    "averaged_small_signal",
    "frequency_response",
    "loop_gain",
]


class SmallSignalError(ModelError):
    """The model is outside the averaged small-signal validity boundary."""

    code = "analysis.small_signal_invalid"


@dataclass(frozen=True)
class AveragedSmallSignal:
    """Averaged LTI model around the periodic operating point."""

    output: str
    duty: float
    period_s: float
    switching_frequency_hz: float
    a_matrix: tuple[tuple[float, ...], ...]
    duty_input: tuple[float, ...]
    output_row: tuple[float, ...]
    output_feedthrough: float
    operating_point: tuple[float, ...]
    state_names: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "pear-sim/small-signal/v0.1",
            "kind": "two-topology-state-space-averaging",
            "output": self.output,
            "duty": self.duty,
            "period_s": self.period_s,
            "switching_frequency_hz": self.switching_frequency_hz,
            "a_matrix": [list(row) for row in self.a_matrix],
            "duty_input": list(self.duty_input),
            "output_row": list(self.output_row),
            "output_feedthrough": self.output_feedthrough,
            "operating_point": list(self.operating_point),
            "state_names": list(self.state_names),
        }


@dataclass(frozen=True)
class SamplingModel:
    """Declared digital sampling/update model.

    The sampled controller updates once per ``1/update_frequency_hz`` and
    holds its command over the interval (zero-order hold), optionally after a
    computation delay. Folded into the loop gain as
    ``(1 - e^(-sT)) / (sT) * e^(-s*tau)`` — the phase lag the continuous
    averaged model has no way to see (up to 180 deg * f/f_update from the
    ZOH alone).
    """

    update_frequency_hz: float
    computation_delay_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "zoh_with_computation_delay",
            "update_frequency_hz": self.update_frequency_hz,
            "computation_delay_s": self.computation_delay_s,
            "note": (
                "loop gain includes (1 - e^(-sT))/(sT) * e^(-s*tau) for "
                "T = 1/update_frequency_hz, tau = computation_delay_s"
            ),
        }


@dataclass(frozen=True)
class LoopGainResult:
    """Loop-gain sweep with declared compensator and margin extraction."""

    frequencies_hz: tuple[float, ...]
    plant_magnitude_db: tuple[float, ...]
    plant_phase_deg: tuple[float, ...]
    loop_magnitude_db: tuple[float, ...]
    loop_phase_deg: tuple[float, ...]
    gain_crossover_hz: float | None
    phase_margin_deg: float | None
    phase_crossover_hz: float | None
    gain_margin_db: float | None
    compensator_numerator: tuple[float, ...]
    compensator_denominator: tuple[float, ...]
    sensor_gain: float
    validity_max_hz: float
    sampling: SamplingModel | None = None
    #: Every 0 dB / -180 deg crossing on the grid. The scalar margins above
    #: are the WORST over these, and the scalar frequencies are where that
    #: worst margin occurs: a loop that crosses 0 dB twice is only as stable
    #: as its weaker crossover.
    gain_crossovers_hz: tuple[float, ...] = ()
    phase_crossovers_hz: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "pear-sim/loop-gain/v0.2",
            "frequencies_hz": list(self.frequencies_hz),
            "plant_magnitude_db": list(self.plant_magnitude_db),
            "plant_phase_deg": list(self.plant_phase_deg),
            "loop_magnitude_db": list(self.loop_magnitude_db),
            "loop_phase_deg": list(self.loop_phase_deg),
            "gain_crossover_hz": self.gain_crossover_hz,
            "phase_margin_deg": self.phase_margin_deg,
            "phase_crossover_hz": self.phase_crossover_hz,
            "gain_margin_db": self.gain_margin_db,
            "gain_crossovers_hz": list(self.gain_crossovers_hz),
            "phase_crossovers_hz": list(self.phase_crossovers_hz),
            "compensator": {
                "numerator": list(self.compensator_numerator),
                "denominator": list(self.compensator_denominator),
            },
            "sensor_gain": self.sensor_gain,
            "validity_max_hz": self.validity_max_hz,
            "sampling": (
                "not_modelled" if self.sampling is None else self.sampling.as_dict()
            ),
            "validity_note": (
                "state-space averaging; magnitudes and phases above "
                "switching_frequency/10 are outside the model's validity"
            ),
        }


def _duty_controlled_modes(grouped: dict[str, dict[str, Any]]) -> tuple[str, str]:
    """(mode lengthened by duty, mode shortened by duty) for ONE PWM leg.

    A plain ``pwm`` gate: on vs off. A complementary leg ``X.high``/``X.low``:
    high-on vs low-on, with both-off dead-time intervals as fixed passengers.
    Anything else (several legs, both gates on) is refused: which interval a
    duty perturbation moves would be a guess.
    """

    channels = sorted({name for entry in grouped.values() for name in entry["gates"]})

    def single(predicate) -> str:
        matches = [sig for sig, entry in grouped.items() if predicate(entry["gates"])]
        if len(matches) != 1:
            raise SmallSignalError(
                "averaged small-signal analysis needs exactly one topology per duty "
                f"state in CCM; found {len(matches)} for one of them"
            )
        return matches[0]

    if len(channels) == 1:
        (gate,) = channels
        dead = []
        rising = single(lambda gates: gates[gate])
        falling = single(lambda gates: not gates[gate])
    elif (
        len(channels) == 2
        and channels[0].endswith(".high")
        and channels[1] == channels[0][: -len(".high")] + ".low"
    ):
        high, low = channels
        if any(entry["gates"][high] and entry["gates"][low] for entry in grouped.values()):
            raise SmallSignalError("both gates of the leg are on in some interval (shoot-through)")
        rising = single(lambda gates: gates[high] and not gates[low])
        falling = single(lambda gates: gates[low] and not gates[high])
        dead = [sig for sig in grouped if sig not in {rising, falling}]
    else:
        raise SmallSignalError(
            "averaged small-signal analysis supports ONE PWM leg (a `pwm` gate or a "
            f"complementary high/low pair); the cycle is driven by {channels}"
        )
    if len(grouped) != 2 + len(dead):
        raise SmallSignalError(
            f"the steady-state cycle visits {len(grouped)} topologies, more than one "
            "PWM leg with dead time explains"
        )
    return rising, falling


def averaged_small_signal(
    model: Model,
    steady_state: PeriodicSteadyState,
    output: str,
) -> AveragedSmallSignal:
    """Two-topology averaged LTI model at the shooting operating point."""

    if any(component.waveform is not None for component in model.components):
        raise SmallSignalError(
            "averaged small-signal analysis supports DC sources only; "
            "waveform sources make the two-topology averaged model invalid"
        )
    if not steady_state.converged:
        raise SmallSignalError(
            "averaged small-signal analysis requires a converged periodic "
            "steady state"
        )
    if steady_state.natural_events_per_cycle != 0:
        raise SmallSignalError(
            "averaged small-signal analysis supports CCM only; the steady-state "
            f"cycle contains {steady_state.natural_events_per_cycle} natural "
            "events (DCM averaging is a follow-up capability)"
        )
    if output not in model.outputs:
        raise SmallSignalError(
            f"output {output!r} is not among the model outputs {list(model.outputs)}"
        )

    cycle = one_cycle_model(model, np.asarray(steady_state.state, dtype=float))
    circuit = Circuit(cycle)
    grouped: dict[str, dict[str, Any]] = {}
    for interval in steady_state.mode_intervals:
        signature = interval["mode_signature"]
        entry = grouped.setdefault(
            signature,
            {"duration": 0.0, "gates": interval["gates"], "diodes": interval["diodes"]},
        )
        entry["duration"] += float(interval["end_s"]) - float(interval["start_s"])
    rising, falling = _duty_controlled_modes(grouped)
    period = steady_state.period_s
    compiled = {
        signature: circuit.compile(entry["gates"], entry["diodes"])
        for signature, entry in grouped.items()
    }
    weights = {signature: entry["duration"] / period for signature, entry in grouped.items()}
    duty = weights[rising]
    on_mode, off_mode = compiled[rising], compiled[falling]

    x_bar = np.asarray(steady_state.cycle_average_state, dtype=float)
    # x' = sum_k d_k (A_k x + c_k). A duty perturbation lengthens the interval
    # with the main gate on and shortens the one with the complementary gate
    # on by the same amount; dead-time intervals keep their duration, so they
    # enter the average and nothing else.
    a_avg = sum(weights[sig] * compiled[sig].a for sig in compiled)
    duty_input = (on_mode.a - off_mode.a) @ x_bar + (on_mode.c - off_mode.c)

    rows = {sig: _output_affine_map(compiled[sig], output) for sig in compiled}
    output_row = sum(weights[sig] * np.asarray(rows[sig][0]) for sig in compiled)
    row_on, const_on, _sources_on = rows[rising]
    row_off, const_off, _sources_off = rows[falling]
    feedthrough = float(
        (np.asarray(row_on) - np.asarray(row_off)) @ x_bar + (const_on - const_off)
    )

    return AveragedSmallSignal(
        output=output,
        duty=float(duty),
        period_s=period,
        switching_frequency_hz=1.0 / period,
        a_matrix=tuple(tuple(float(v) for v in row) for row in a_avg),
        duty_input=tuple(float(v) for v in duty_input),
        output_row=tuple(float(v) for v in output_row),
        output_feedthrough=feedthrough,
        operating_point=tuple(float(v) for v in x_bar),
        state_names=steady_state.state_names,
    )


def frequency_response(
    small_signal: AveragedSmallSignal, frequencies_hz: np.ndarray
) -> np.ndarray:
    """Complex control-to-output response G_yd(j*2*pi*f) on the given grid."""

    a = np.asarray(small_signal.a_matrix, dtype=float)
    b = np.asarray(small_signal.duty_input, dtype=float)
    c = np.asarray(small_signal.output_row, dtype=float)
    d = small_signal.output_feedthrough
    identity = np.eye(a.shape[0])
    response = np.empty(len(frequencies_hz), dtype=complex)
    for index, frequency in enumerate(frequencies_hz):
        s = 2j * np.pi * float(frequency)
        response[index] = c @ np.linalg.solve(s * identity - a, b) + d
    return response


def _polyval_jw(coefficients: tuple[float, ...], frequencies_hz: np.ndarray) -> np.ndarray:
    s = 2j * np.pi * np.asarray(frequencies_hz, dtype=float)
    return np.polyval(np.asarray(coefficients, dtype=float), s)


def _interpolate_crossings(
    frequencies: np.ndarray, values: np.ndarray, target: float
) -> list[float]:
    """Every crossing of ``target`` on the grid, log-frequency interpolated."""

    shifted = values - target
    signs = np.sign(shifted)
    crossings: list[float] = []
    for index in range(len(shifted) - 1):
        if signs[index] == 0.0:
            crossings.append(float(frequencies[index]))
        elif signs[index] * signs[index + 1] < 0.0:
            f0, f1 = np.log10(frequencies[index]), np.log10(frequencies[index + 1])
            v0, v1 = shifted[index], shifted[index + 1]
            crossings.append(float(10.0 ** (f0 - v0 * (f1 - f0) / (v1 - v0))))
    if len(signs) and signs[-1] == 0.0:
        crossings.append(float(frequencies[-1]))
    return crossings


def loop_gain(
    small_signal: AveragedSmallSignal,
    *,
    compensator_numerator: tuple[float, ...] = (1.0,),
    compensator_denominator: tuple[float, ...] = (1.0,),
    sensor_gain: float = 1.0,
    f_min_hz: float | None = None,
    f_max_hz: float | None = None,
    points: int = 200,
    sampling: SamplingModel | None = None,
) -> LoopGainResult:
    """Loop gain L = C * G_yd * H with grid-interpolated PM/GM extraction.

    With a declared ``sampling`` model the ZOH and computation-delay terms
    are folded into L(s); without one the result stays continuous and says
    so (``sampling: not_modelled``) — the omission is visible, never silent.
    """

    if points < 10:
        raise SmallSignalError("loop-gain sweep needs at least 10 frequency points")
    if not np.isfinite(sensor_gain) or sensor_gain == 0.0:
        raise SmallSignalError("sensor_gain must be finite and non-zero")
    f_sw = small_signal.switching_frequency_hz
    f_min = 1.0 if f_min_hz is None else float(f_min_hz)
    f_max = f_sw / 2.0 if f_max_hz is None else float(f_max_hz)
    if not 0.0 < f_min < f_max:
        raise SmallSignalError("frequency window must satisfy 0 < f_min < f_max")
    if sampling is not None:
        update_hz = sampling.update_frequency_hz
        delay_s = sampling.computation_delay_s
        if not np.isfinite(update_hz) or update_hz <= 0.0:
            raise SmallSignalError(
                "sampling update_frequency_hz must be finite and positive"
            )
        if not np.isfinite(delay_s) or delay_s < 0.0:
            raise SmallSignalError(
                "sampling computation_delay_s must be finite and non-negative"
            )
        if f_max >= update_hz:
            raise SmallSignalError(
                "the loop-gain grid must stay below the sampling update "
                f"frequency; f_max {f_max!r} Hz >= update {update_hz!r} Hz"
            )
    frequencies = np.logspace(np.log10(f_min), np.log10(f_max), points)

    plant = frequency_response(small_signal, frequencies)
    compensator = _polyval_jw(compensator_numerator, frequencies) / _polyval_jw(
        compensator_denominator, frequencies
    )
    loop = plant * compensator * sensor_gain
    if sampling is not None:
        s = 2j * np.pi * frequencies
        period = 1.0 / sampling.update_frequency_hz
        zoh = (1.0 - np.exp(-s * period)) / (s * period)
        loop = loop * zoh * np.exp(-s * sampling.computation_delay_s)

    plant_db = 20.0 * np.log10(np.abs(plant))
    plant_phase = np.degrees(np.unwrap(np.angle(plant)))
    loop_db = 20.0 * np.log10(np.abs(loop))
    loop_phase = np.degrees(np.unwrap(np.angle(loop)))

    log_f = np.log10(frequencies)
    gain_crossovers = _interpolate_crossings(frequencies, loop_db, 0.0)
    gain_crossover = None
    phase_margin = None
    for crossing in gain_crossovers:
        margin = 180.0 + float(np.interp(np.log10(crossing), log_f, loop_phase))
        if phase_margin is None or margin < phase_margin:
            gain_crossover, phase_margin = crossing, margin
    phase_crossovers = _interpolate_crossings(frequencies, loop_phase, -180.0)
    phase_crossover = None
    gain_margin = None
    for crossing in phase_crossovers:
        margin = -float(np.interp(np.log10(crossing), log_f, loop_db))
        if gain_margin is None or margin < gain_margin:
            phase_crossover, gain_margin = crossing, margin

    return LoopGainResult(
        frequencies_hz=tuple(float(v) for v in frequencies),
        plant_magnitude_db=tuple(float(v) for v in plant_db),
        plant_phase_deg=tuple(float(v) for v in plant_phase),
        loop_magnitude_db=tuple(float(v) for v in loop_db),
        loop_phase_deg=tuple(float(v) for v in loop_phase),
        gain_crossover_hz=gain_crossover,
        phase_margin_deg=phase_margin,
        phase_crossover_hz=phase_crossover,
        gain_margin_db=gain_margin,
        gain_crossovers_hz=tuple(gain_crossovers),
        phase_crossovers_hz=tuple(phase_crossovers),
        compensator_numerator=tuple(float(v) for v in compensator_numerator),
        compensator_denominator=tuple(float(v) for v in compensator_denominator),
        sensor_gain=float(sensor_gain),
        validity_max_hz=f_sw / 10.0,
        sampling=sampling,
    )
