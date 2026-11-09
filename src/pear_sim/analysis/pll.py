# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Offline SRF-PLL over recorded three-phase signals (grid interface).

A standard synchronous-reference-frame PLL run as POST-PROCESSING of a
recorded, uniformly sampled abc voltage set: Clarke -> Park at the estimated
angle, PI on the normalized q component, forward-Euler angle integration.
This recovers the grid angle/frequency from a finished run — an in-the-loop
PLL (feeding a controller during simulation) is a separate, future kernel
capability and is deliberately NOT claimed here.

Gains follow the textbook second-order design: for a target bandwidth f_bw
and damping zeta, omega_n = 2*pi*f_bw, kp = 2*zeta*omega_n, ki = omega_n^2,
acting on vq normalized by the running dq magnitude (locking is then
amplitude-independent).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pear_sim.analysis.reference_frames import ReferenceFrameError, clarke

__all__ = ["PllResult", "srf_pll"]

_MIN_SAMPLES = 32
_MAGNITUDE_FLOOR = 1e-12


@dataclass(frozen=True)
class PllResult:
    """Recovered angle/frequency track plus honest lock diagnostics."""

    time_s: tuple[float, ...]
    theta_rad: tuple[float, ...]
    frequency_hz: tuple[float, ...]
    q_residual: tuple[float, ...]
    locked: bool
    locked_frequency_hz: float | None
    lock_time_s: float | None
    lock_threshold: float
    nominal_frequency_hz: float
    bandwidth_hz: float
    damping: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "pear-sim/srf-pll/v0.1",
            "kind": "offline_post_processing_srf_pll",
            "locked": self.locked,
            "locked_frequency_hz": self.locked_frequency_hz,
            "lock_time_s": self.lock_time_s,
            "lock_threshold": self.lock_threshold,
            "nominal_frequency_hz": self.nominal_frequency_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "damping": self.damping,
            "time_s": list(self.time_s),
            "theta_rad": list(self.theta_rad),
            "frequency_hz": list(self.frequency_hz),
            "q_residual": list(self.q_residual),
        }


def srf_pll(
    time_s: Any,
    va: Any,
    vb: Any,
    vc: Any,
    *,
    nominal_frequency_hz: float,
    bandwidth_hz: float = 50.0,
    damping: float = 0.7071067811865476,
    theta0_rad: float = 0.0,
    lock_threshold: float = 0.01,
) -> PllResult:
    """Track the abc set's angle; report lock only when the tail stays locked.

    ``locked`` requires the normalized q residual to stay below
    ``lock_threshold`` over the entire last half of the record — a transient
    that wanders back out of lock is reported honestly as unlocked.
    """

    time = np.asarray(time_s, dtype=float)
    if time.ndim != 1 or len(time) < _MIN_SAMPLES:
        raise ReferenceFrameError(
            f"srf_pll needs a one-dimensional grid of at least {_MIN_SAMPLES} samples"
        )
    steps = np.diff(time)
    if np.any(steps <= 0.0):
        raise ReferenceFrameError("srf_pll needs a strictly increasing time grid")
    if not np.isfinite(nominal_frequency_hz) or nominal_frequency_hz <= 0.0:
        raise ReferenceFrameError("nominal_frequency_hz must be positive")
    if not np.isfinite(bandwidth_hz) or bandwidth_hz <= 0.0:
        raise ReferenceFrameError("bandwidth_hz must be positive")
    if not 0.0 < damping < 10.0:
        raise ReferenceFrameError("damping must be positive and sane")
    if bandwidth_hz * float(np.max(steps)) > 0.1:
        raise ReferenceFrameError(
            "srf_pll bandwidth is too high for the sample grid; "
            "bandwidth_hz * max(dt) must stay below 0.1"
        )

    alpha, beta, _zero = clarke(va, vb, vc)
    if alpha.shape != time.shape:
        raise ReferenceFrameError("phase signals must match the time grid")

    omega_n = 2.0 * np.pi * bandwidth_hz
    kp = 2.0 * damping * omega_n
    ki = omega_n * omega_n
    omega_nominal = 2.0 * np.pi * nominal_frequency_hz

    theta = float(theta0_rad)
    integrator = 0.0
    theta_track = np.empty_like(time)
    frequency_track = np.empty_like(time)
    residual_track = np.empty_like(time)

    for index in range(len(time)):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        v_q = -alpha[index] * sin_t + beta[index] * cos_t
        v_d = alpha[index] * cos_t + beta[index] * sin_t
        magnitude = float(np.hypot(v_d, v_q))
        normalized_q = v_q / max(magnitude, _MAGNITUDE_FLOOR)
        omega = omega_nominal + kp * normalized_q + integrator
        theta_track[index] = theta
        frequency_track[index] = omega / (2.0 * np.pi)
        residual_track[index] = abs(normalized_q)
        if index < len(time) - 1:
            dt = float(steps[index])
            integrator += ki * normalized_q * dt
            theta = float((theta + omega * dt) % (2.0 * np.pi))

    half = len(time) // 2
    tail_locked = bool(np.all(residual_track[half:] < lock_threshold))
    if tail_locked:
        locked_frequency = float(np.mean(frequency_track[half:]))
        below = residual_track < lock_threshold
        # Lock time: start of the final unbroken locked stretch.
        lock_index = half
        for index in range(half, -1, -1):
            if not below[index]:
                lock_index = index + 1
                break
        else:
            lock_index = 0
        lock_time = float(time[lock_index])
    else:
        locked_frequency = None
        lock_time = None

    return PllResult(
        time_s=tuple(float(v) for v in time),
        theta_rad=tuple(float(v) for v in theta_track),
        frequency_hz=tuple(float(v) for v in frequency_track),
        q_residual=tuple(float(v) for v in residual_track),
        locked=tail_locked,
        locked_frequency_hz=locked_frequency,
        lock_time_s=lock_time,
        lock_threshold=float(lock_threshold),
        nominal_frequency_hz=float(nominal_frequency_hz),
        bandwidth_hz=float(bandwidth_hz),
        damping=float(damping),
    )
