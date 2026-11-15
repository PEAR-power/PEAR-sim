# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Closed-form Floquet oracle: ideal CCM buck under a sampled PI.

Derived on paper, solved by a different method than the product, and it
imports NOTHING from PEARSim — it takes the circuit as plain numbers.

Plant (ideal switch, ideal diode, CCM). With x = [i_L, v_C],

    dx/dt = A x + s(t) b,   A = [[0, -1/L], [1/C, -1/(R C)]],   b = [Vin/L, 0],

where s = 1 while the switch conducts. Both topologies share A — only the
input differs — which is what makes this case closed-form.

Controller (one tick per period, at the cycle start, reading the pre-edge
state; the switch turns on at the tick and off at t_s = d T):

    e = vref - v_C(0),   r' = r + e T,   d = kp e + ki r'.

One period from z = [i_L, v_C, r]:

    x(T) = e^{A T} x0 + e^{A (T - t_s)} W(t_s),   W(t) = int_0^t e^{A u} du b,

so with dx(T)/dt_s = e^{A (T - t_s)} b and dt_s/dz = T dd/dz,

    dd/dz = [0, -(kp + ki T), ki],
    M = [[e^{A T} + p q^T],      p = e^{A (T - t_s)} b T,
         [0, -T, 1        ]],    q = dd/dz.

The orbit has e = 0 (the register is stationary only then), so v_C(0) = vref
and the remaining unknowns i_L(0) and t_s solve the affine fixed point
(I - e^{A T}) x0 = e^{A (T - t_s)} W(t_s) — one scalar root in t_s, found
with a bracketed ``brentq``. The register is r* = d / ki.

Verification evidence for the declared ideal equations only; it is not a
physical-validation claim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq


@dataclass(frozen=True)
class BuckLoop:
    vin_v: float
    inductance_h: float
    capacitance_f: float
    load_ohm: float
    switching_frequency_hz: float
    vref_v: float
    kp: float
    ki: float

    @property
    def period_s(self) -> float:
        return 1.0 / self.switching_frequency_hz

    @property
    def a(self) -> np.ndarray:
        ind, cap, res = self.inductance_h, self.capacitance_f, self.load_ohm
        return np.array([[0.0, -1.0 / ind], [1.0 / cap, -1.0 / (res * cap)]])

    @property
    def b(self) -> np.ndarray:
        return np.array([self.vin_v / self.inductance_h, 0.0])


@dataclass(frozen=True)
class OracleOrbit:
    duty: float
    inductor_current_a: float
    capacitor_voltage_v: float
    register: float
    monodromy: np.ndarray
    multipliers: np.ndarray  # sorted by decreasing magnitude


def _forced_response(loop: BuckLoop, on_time: float) -> np.ndarray:
    """e^{A (T - t_on)} int_0^{t_on} e^{A u} du b, by the van Loan block."""

    block = np.zeros((3, 3))
    block[:2, :2] = loop.a
    block[:2, 2] = loop.b
    w = expm(block * on_time)[:2, 2]
    return expm(loop.a * (loop.period_s - on_time)) @ w


def solve(loop: BuckLoop) -> OracleOrbit:
    period = loop.period_s
    phi = expm(loop.a * period)
    inverse = np.linalg.inv(np.eye(2) - phi)

    def output_error(on_time: float) -> float:
        return float((inverse @ _forced_response(loop, on_time))[1] - loop.vref_v)

    on_time = brentq(output_error, 1e-9 * period, (1.0 - 1e-9) * period, xtol=1e-18, rtol=1e-15)
    x0 = inverse @ _forced_response(loop, on_time)
    if x0[0] <= 0.0:
        raise ValueError("orbit is not in CCM: the valley current is not positive")
    duty = on_time / period
    p = expm(loop.a * (period - on_time)) @ loop.b * period
    q = np.array([0.0, -(loop.kp + loop.ki * period), loop.ki])
    monodromy = np.zeros((3, 3))
    monodromy[:2, :2] = phi
    monodromy[:2, :] += np.outer(p, q)
    monodromy[2, :] = [0.0, -period, 1.0]
    values = np.linalg.eigvals(monodromy)
    return OracleOrbit(
        duty=duty,
        inductor_current_a=float(x0[0]),
        capacitor_voltage_v=float(x0[1]),
        register=duty / loop.ki,
        monodromy=monodromy,
        multipliers=values[np.argsort(-np.abs(values))],
    )


def rlc_multipliers(loop: BuckLoop) -> np.ndarray:
    """Textbook physics, independent of the matrix algebra above: the
    series-L / parallel-RC network's natural frequencies
    lambda = -1/(2RC) +- sqrt(1/(2RC)^2 - 1/(LC)), mapped over one period."""

    alpha = 1.0 / (2.0 * loop.load_ohm * loop.capacitance_f)
    omega0_sq = 1.0 / (loop.inductance_h * loop.capacitance_f)
    root = np.sqrt(complex(alpha * alpha - omega0_sq))
    return np.exp(np.array([-alpha + root, -alpha - root]) * loop.period_s)


# ---------------------------------------------------------------------------
# DCM: the inductor current returns to zero inside every period
# ---------------------------------------------------------------------------
#
# Three intervals from z = [i_L, v_C, r] (switch on, diode on, both off):
#
#   1. [0, t_s]       x1 = e^{A t_s} x0 + W(t_s)                 (source on)
#   2. [t_s, t_s+t_2] x(tau) = e^{A tau} x1 until i_L(t_2) = 0   (free LC)
#   3. the rest       i_L = 0,  v_C decays with RC:  v(T) = v2 e^{-t_3/(RC)}
#
# t_2 is the FIRST root of [e^{A tau} x1]_0, bracketed and solved by brentq.
# Derivatives by the implicit function theorem on that root, written out by
# hand (no saltation matrices, no simulator):
#
#   dx1 = e^{A t_s} dx0 + (A x1 + b) dt_s
#   dt_2 = -[e^{A t_2} dx1]_0 / [A x2]_0
#   dx2 = e^{A t_2} dx1 + A x2 dt_2
#   dv(T) = e^{-t_3/(RC)} dv2 + v(T)/(RC) (dt_s + dt_2),   di(T) = 0
#
# The orbit has e = 0 and i_L(0) = 0 (it starts from the zero-current
# interval), so v_C(0) = vref and t_s is one scalar root of v(T) = vref.


@dataclass(frozen=True)
class DcmOrbit:
    duty: float
    capacitor_voltage_v: float
    register: float
    fall_time_s: float
    monodromy: np.ndarray
    multipliers: np.ndarray  # sorted by decreasing magnitude


def _on_state(loop: BuckLoop, x0: np.ndarray, on_time: float) -> np.ndarray:
    block = np.zeros((3, 3))
    block[:2, :2] = loop.a
    block[:2, 2] = loop.b
    return expm(loop.a * on_time) @ x0 + expm(block * on_time)[:2, 2]


def _fall_time(loop: BuckLoop, x1: np.ndarray, available: float) -> float:
    def current(tau: float) -> float:
        return float((expm(loop.a * tau) @ x1)[0])

    grid = np.linspace(0.0, available, 2001)[1:]
    for low, high in zip(np.concatenate([[0.0], grid[:-1]]), grid, strict=True):
        if current(high) <= 0.0:
            if current(low) <= 0.0:
                return float(low)
            return float(brentq(current, low, high, xtol=1e-20, rtol=1e-15))
    raise ValueError("orbit is not in DCM: the inductor current does not reach zero")


def solve_dcm(loop: BuckLoop) -> DcmOrbit:
    period = loop.period_s
    rc = loop.load_ohm * loop.capacitance_f
    start = np.array([0.0, loop.vref_v])

    def end_voltage(on_time: float) -> float:
        x1 = _on_state(loop, start, on_time)
        fall = _fall_time(loop, x1, period - on_time)
        x2 = expm(loop.a * fall) @ x1
        return float(x2[1] * np.exp(-(period - on_time - fall) / rc))

    # Upper bracket: the longest on-time that still leaves the current room to
    # reach zero (beyond it the orbit is CCM and this oracle does not apply).
    upper = 0.9 * period
    while True:
        try:
            end_voltage(upper)
            break
        except ValueError:
            upper *= 0.9
            if upper < 1e-3 * period:
                raise
    on_time = brentq(
        lambda t: end_voltage(t) - loop.vref_v, 1e-4 * period, upper,
        xtol=1e-20, rtol=1e-15,
    )
    x1 = _on_state(loop, start, on_time)
    fall = _fall_time(loop, x1, period - on_time)
    x2 = expm(loop.a * fall) @ x1
    rest = period - on_time - fall
    decay = np.exp(-rest / rc)
    v_end = float(x2[1] * decay)

    duty = on_time / period
    # dt_s/dz from the controller, as in the CCM oracle.
    dts = period * np.array([0.0, -(loop.kp + loop.ki * period), loop.ki])
    dx0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    dx1 = expm(loop.a * on_time) @ dx0 + np.outer(loop.a @ x1 + loop.b, dts)
    propagated = expm(loop.a * fall) @ dx1
    slope = float((loop.a @ x2)[0])
    dfall = -propagated[0, :] / slope
    dx2 = propagated + np.outer(loop.a @ x2, dfall)
    monodromy = np.zeros((3, 3))
    monodromy[0, :] = 0.0
    monodromy[1, :] = decay * dx2[1, :] + v_end / rc * (dts + dfall)
    monodromy[2, :] = [0.0, -period, 1.0]
    values = np.linalg.eigvals(monodromy)
    return DcmOrbit(
        duty=duty,
        capacitor_voltage_v=v_end,
        register=duty / loop.ki,
        fall_time_s=fall,
        monodromy=monodromy,
        multipliers=values[np.argsort(-np.abs(values))],
    )
