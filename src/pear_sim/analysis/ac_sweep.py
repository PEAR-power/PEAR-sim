# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Small-signal AC sweep on the SWITCHED model (not on an averaged one).

WHY THIS EXISTS. `analysis/small_signal.py` linearises a two-topology cycle
into an averaged LTI model and evaluates its transfer function in closed
form. That is fast and exact FOR THE AVERAGED MODEL — but the averaged model
is an approximation of the converter, valid below roughly a tenth of the
switching frequency and blind to anything the averaging discards. This
module answers the other question: what does the ACTUAL switching circuit do
when its duty cycle is wiggled?

THE METHOD is the one every bench does. Perturb the duty command with a
small sinusoid at f_p, integrate the real switched model, and correlate the
recorded output against that same sinusoid over an integer number of
perturbation periods. The correlation is a single-bin DFT: it rejects the
switching ripple, the DC operating point and every harmonic that is not
f_p, provided the window is an exact integer multiple of 1/f_p — which is
why the window is constructed rather than chosen.

THE ORACLE IS INTERNAL, AND THAT IS THE POINT. Below f_sw/10 this sweep and
the averaged model must agree; where they diverge, the averaged model is
the one that stopped being valid. No external tool is needed to check it,
and the test suite pins the agreement on the canonical buck.

DECLARED LIMITS. Duty-to-output only (the perturbation enters as duty).
Linear-range amplitudes only: the caller picks the amplitude, and an
amplitude sweep is offered so the caller can SEE whether the answer depends
on it — a response that moves with amplitude is not a small-signal response.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from pear_sim.errors import ModelError
from pear_sim.spec.controls import PwmControl, SignalPwmControl
from pear_sim.spec.model import Model
from pear_sim.spec.signal_graph import SignalBlock, SignalGraph, SignalTask, _resolve_order


class AcSweepError(ModelError):
    """Raised when a switched AC sweep cannot be performed as requested."""

    code = "analysis.ac_sweep_invalid"


@dataclass(frozen=True)
class AcSweepPoint:
    """One correlated frequency point."""

    frequency_hz: float
    gain: float
    phase_deg: float
    real: float
    imag: float
    perturbation_amplitude: float
    periods: int
    settle_periods: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "frequency_hz": self.frequency_hz,
            "gain": self.gain,
            "gain_db": float(20.0 * np.log10(self.gain)) if self.gain > 0 else None,
            "phase_deg": self.phase_deg,
            "real": self.real,
            "imag": self.imag,
            "perturbation_amplitude": self.perturbation_amplitude,
            "periods": self.periods,
            "settle_periods": self.settle_periods,
        }


def _duty_of(model: Model) -> tuple[str, float, float]:
    """The single PWM control this sweep perturbs, with its duty and rate."""

    candidates = [
        control
        for control in model.controls.values()
        if isinstance(control, PwmControl) and control.kind == "pwm"
    ]
    if len(candidates) != 1:
        raise AcSweepError(
            "a switched AC sweep perturbs exactly one plain `pwm` control; this "
            f"model has {len(candidates)}"
        )
    control = candidates[0]
    return control.id, float(control.duty), float(control.frequency)


def perturbed_model(
    model: Model,
    *,
    frequency_hz: float,
    amplitude: float,
    stop_s: float,
) -> Model:
    """Same converter, duty replaced by D + A*sin(2*pi*f*t) from the signal layer.

    Built by REUSING the declared signal layer rather than by teaching the
    scheduler about perturbations: the duty a `signal_pwm` bridge consumes is
    already a signal, so a sweep is an ordinary model whose duty happens to
    be a sinusoid. Nothing in the kernel learns anything new.
    """

    if getattr(model, "signals", None) is not None:
        raise AcSweepError(
            "the model already declares a signal layer; perturbing it would "
            "silently redefine the author's controller"
        )
    control_id, duty, carrier_hz = _duty_of(model)
    blocks = (
        SignalBlock(id="_ac_bias", kind="constant", unit="1",
                    params={"value": duty}),
        SignalBlock(id="_ac_drive", kind="sine", unit="1",
                    params={"amplitude": amplitude, "frequency": frequency_hz,
                            "phase_rad": 0.0, "offset": 0.0}),
        SignalBlock(id="_ac_duty", kind="sum", unit="1",
                    inputs=("_ac_bias", "_ac_drive"),
                    params={"signs": (1.0, 1.0)}),
    )
    graph = SignalGraph(
        task=SignalTask(period=1.0 / carrier_hz, offset=0.0),
        blocks=blocks,
        order=_resolve_order(blocks),
    )
    bridge = SignalPwmControl(
        id=control_id,
        frequency=carrier_hz,
        duty_signal="s(_ac_duty)",
        # The limits exist so a perturbation cannot drive the duty out of
        # range; they are deliberately wide, and the caller sees the
        # amplitude sweep if saturation ever starts to matter.
        duty_min=max(1e-6, duty - 10.0 * amplitude),
        duty_max=min(1.0 - 1e-6, duty + 10.0 * amplitude),
    )
    controls = {k: v for k, v in model.controls.items() if k != control_id}
    sampled = dict(model.sampled_controls)
    sampled[control_id] = bridge
    return replace(
        model,
        controls=controls,
        sampled_controls=sampled,
        signals=graph,
        transient=replace(model.transient, stop=stop_s),
    )


def correlate(
    time: np.ndarray,
    values: np.ndarray,
    *,
    frequency_hz: float,
    window_start_s: float,
) -> complex:
    """Single-bin DFT of ``values`` at ``frequency_hz`` over an integer window.

    Only samples at or after ``window_start_s`` participate, so the start-up
    transient never enters the estimate. The window is trimmed to a whole
    number of perturbation periods before the sum; a partial period would
    leak the operating point into the bin and show up as a phantom gain.
    """

    period = 1.0 / frequency_hz
    mask = time >= window_start_s - 1e-15
    t = np.asarray(time[mask], dtype=float)
    y = np.asarray(values[mask], dtype=float)
    if len(t) < 8:
        raise AcSweepError("correlation window holds fewer than eight samples")
    span = t[-1] - t[0]
    whole = np.floor(span / period + 1e-9) * period
    if whole <= 0.0:
        raise AcSweepError(
            f"correlation window {span:.6g} s is shorter than one perturbation "
            f"period {period:.6g} s"
        )
    keep = t <= t[0] + whole + 1e-15
    t, y = t[keep], y[keep]
    phase = 2.0 * np.pi * frequency_hz * (t - t[0])
    # THE REFERENCE IS THE PERTURBATION ITSELF, WHICH IS A SINE. Projecting
    # onto cos/-sin (the textbook exp(-j*phase) convention) measures phase
    # against a COSINE and reports every result rotated by -90 degrees —
    # amplitudes stay perfect, so only a phase check catches it. For
    # y = |G|*A*sin(w*t + phi):  <y,sin> = |G|*A/2*cos(phi)*T  and
    # <y,cos> = |G|*A/2*sin(phi)*T, hence the pairing below.
    # The DC term integrates to zero over whole periods; it is removed anyway
    # so a truncation error cannot masquerade as gain.
    y = y - np.trapezoid(y, t) / (t[-1] - t[0])
    in_phase = np.trapezoid(y * np.sin(phase), t)
    quadrature = np.trapezoid(y * np.cos(phase), t)
    return complex(2.0 * in_phase / whole, 2.0 * quadrature / whole)


def ac_sweep_point(
    model: Model,
    output: str,
    *,
    frequency_hz: float,
    amplitude: float = 0.01,
    periods: int = 12,
    settle_periods: int = 6,
) -> AcSweepPoint:
    """One frequency of a switched-model duty-to-output sweep."""

    from pear_sim.core.solver import HybridSimulator

    if output not in model.outputs:
        raise AcSweepError(
            f"output {output!r} is not among the model outputs {list(model.outputs)}"
        )
    if amplitude <= 0.0:
        raise AcSweepError("perturbation amplitude must be positive")
    if periods < 2 or settle_periods < 0:
        raise AcSweepError("need at least two measured periods")
    period = 1.0 / frequency_hz
    stop = (periods + settle_periods) * period
    perturbed = perturbed_model(
        model, frequency_hz=frequency_hz, amplitude=amplitude, stop_s=stop
    )
    result = HybridSimulator(perturbed).simulate()
    response = correlate(
        np.asarray(result.time, dtype=float),
        np.asarray(result.signals[output], dtype=float),
        frequency_hz=frequency_hz,
        window_start_s=settle_periods * period,
    )
    transfer = response / amplitude
    return AcSweepPoint(
        frequency_hz=float(frequency_hz),
        gain=float(abs(transfer)),
        phase_deg=float(np.degrees(np.angle(transfer))),
        real=float(transfer.real),
        imag=float(transfer.imag),
        perturbation_amplitude=float(amplitude),
        periods=int(periods),
        settle_periods=int(settle_periods),
    )


def ac_sweep(
    model: Model,
    output: str,
    frequencies_hz: Any,
    *,
    amplitude: float = 0.01,
    periods: int = 12,
    settle_periods: int = 6,
) -> tuple[AcSweepPoint, ...]:
    """Duty-to-output response of the SWITCHED model over a frequency grid."""

    _, _, carrier_hz = _duty_of(model)
    points = []
    for frequency in np.asarray(frequencies_hz, dtype=float):
        if frequency >= 0.5 * carrier_hz:
            raise AcSweepError(
                f"perturbation at {frequency:.6g} Hz is at or above the Nyquist "
                f"rate of the {carrier_hz:.6g} Hz carrier; the duty command is "
                "updated once per cycle and cannot represent it"
            )
        points.append(
            ac_sweep_point(
                model, output, frequency_hz=float(frequency), amplitude=amplitude,
                periods=periods, settle_periods=settle_periods,
            )
        )
    return tuple(points)
