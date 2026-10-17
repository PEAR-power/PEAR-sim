"""Clarke/Park reference-frame transforms and P/Q metrics.

Pure post-processing of recorded three-phase signals — nothing here touches
the kernel. Conventions are stated explicitly because every downstream
grid-interface claim (PLL, dVOC, P/Q export) inherits them:

- Clarke (power-invariant is NOT used): the amplitude-invariant form
  ``alpha = (2/3)*(a - b/2 - c/2)``, ``beta = (a - c)/sqrt(3) * ... `` — see
  the matrix below; a balanced set of amplitude A maps to a rotating vector
  of magnitude A.
- Park: ``d + j q = (alpha + j beta) * exp(-j theta)`` with ``theta`` the
  electrical angle of the a-axis (cosine reference). A balanced positive-
  sequence set ``a = A cos(theta + phi)`` maps to constant ``d = A cos(phi)``,
  ``q = A sin(phi)``.
- Instantaneous power (amplitude-invariant Clarke): ``p = 1.5*(v_alpha*
  i_alpha + v_beta*i_beta) + 3*v_0*i_0``; ``q = 1.5*(v_beta*i_alpha -
  v_alpha*i_beta)`` (positive q = inductive/lagging load absorbing vars).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pear_sim.errors import ModelError

__all__ = [
    "ReferenceFrameError",
    "ThreePhaseFrames",
    "clarke",
    "park",
    "three_phase_frames",
    "instantaneous_power",
]

_SQRT3 = float(np.sqrt(3.0))


class ReferenceFrameError(ModelError):
    """Raised when a reference-frame request is outside the stated contract."""

    code = "analysis.reference_frame_invalid"


@dataclass(frozen=True)
class ThreePhaseFrames:
    """Recorded three-phase signal in stationary and rotating frames."""

    time_s: tuple[float, ...]
    alpha: tuple[float, ...]
    beta: tuple[float, ...]
    zero: tuple[float, ...]
    d: tuple[float, ...]
    q: tuple[float, ...]
    theta_rad: tuple[float, ...]
    convention: str = "amplitude_invariant_clarke_cosine_park"

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "pear-sim/reference-frames/v0.1",
            "convention": self.convention,
            "time_s": list(self.time_s),
            "alpha": list(self.alpha),
            "beta": list(self.beta),
            "zero": list(self.zero),
            "d": list(self.d),
            "q": list(self.q),
            "theta_rad": list(self.theta_rad),
        }


def _as_phases(a: Any, b: Any, c: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase_a = np.asarray(a, dtype=float)
    phase_b = np.asarray(b, dtype=float)
    phase_c = np.asarray(c, dtype=float)
    if not (phase_a.shape == phase_b.shape == phase_c.shape) or phase_a.ndim != 1:
        raise ReferenceFrameError(
            "phase signals must be three one-dimensional arrays of equal length"
        )
    if not (
        np.all(np.isfinite(phase_a))
        and np.all(np.isfinite(phase_b))
        and np.all(np.isfinite(phase_c))
    ):
        raise ReferenceFrameError("phase signals must be finite")
    return phase_a, phase_b, phase_c


def clarke(a: Any, b: Any, c: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Amplitude-invariant Clarke transform abc -> (alpha, beta, zero)."""

    phase_a, phase_b, phase_c = _as_phases(a, b, c)
    alpha = (2.0 * phase_a - phase_b - phase_c) / 3.0
    beta = (phase_b - phase_c) / _SQRT3
    zero = (phase_a + phase_b + phase_c) / 3.0
    return alpha, beta, zero


def park(
    alpha: Any, beta: Any, theta_rad: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (alpha, beta) into the dq frame at electrical angle theta."""

    alpha = np.asarray(alpha, dtype=float)
    beta = np.asarray(beta, dtype=float)
    theta = np.asarray(theta_rad, dtype=float)
    if not (alpha.shape == beta.shape == theta.shape):
        raise ReferenceFrameError("alpha/beta/theta must share one shape")
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    d = alpha * cos_t + beta * sin_t
    q = -alpha * sin_t + beta * cos_t
    return d, q


def three_phase_frames(
    time_s: Any,
    a: Any,
    b: Any,
    c: Any,
    *,
    frequency_hz: float,
    theta0_rad: float = 0.0,
) -> ThreePhaseFrames:
    """Clarke + Park of a recorded abc set at a declared synchronous frequency.

    ``theta = 2*pi*frequency_hz*t + theta0_rad`` — an open-loop angle. A PLL
    (closed-loop angle recovery) is a separate, future capability; using this
    helper on a signal whose actual frequency differs from the declaration
    shows up as a rotation of (d, q) at the difference frequency, which is
    honest and visible rather than silently corrected.
    """

    if not np.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ReferenceFrameError("frequency_hz must be positive and finite")
    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1:
        raise ReferenceFrameError("time_s must be one-dimensional")
    alpha, beta, zero = clarke(a, b, c)
    if alpha.shape != time.shape:
        raise ReferenceFrameError("phase signals must match the time grid")
    theta = 2.0 * np.pi * frequency_hz * time + theta0_rad
    d, q = park(alpha, beta, theta)
    return ThreePhaseFrames(
        time_s=tuple(float(v) for v in time),
        alpha=tuple(float(v) for v in alpha),
        beta=tuple(float(v) for v in beta),
        zero=tuple(float(v) for v in zero),
        d=tuple(float(v) for v in d),
        q=tuple(float(v) for v in q),
        theta_rad=tuple(float(v) for v in theta),
    )


def instantaneous_power(
    va: Any,
    vb: Any,
    vc: Any,
    ia: Any,
    ib: Any,
    ic: Any,
) -> dict[str, np.ndarray]:
    """Instantaneous p/q from abc voltages and currents (amplitude-invariant).

    Sign convention: currents measured INTO the load — positive ``p`` is
    power absorbed by the load, positive ``q`` is inductive (lagging)
    reactive power absorbed. The zero-sequence power appears only in ``p``.
    """

    v_alpha, v_beta, v_zero = clarke(va, vb, vc)
    i_alpha, i_beta, i_zero = clarke(ia, ib, ic)
    p = 1.5 * (v_alpha * i_alpha + v_beta * i_beta) + 3.0 * v_zero * i_zero
    q = 1.5 * (v_beta * i_alpha - v_alpha * i_beta)
    return {"p_w": p, "q_var": q}
