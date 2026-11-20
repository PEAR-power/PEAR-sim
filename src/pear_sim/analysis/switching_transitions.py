# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Post-processed switching transitions: dead-time Coss swing and loop overshoot.

The kernel stays ideal. For every dead-time window of a declared half-bridge
leg this module integrates, OUTSIDE the circuit equations, the swing of the
switching node through the datasheet's nonlinear output capacitance, driven by
the commutating current recorded at the outgoing switch's turn-off.

Frame (per node k, "u" = voltage across the INCOMING device, V at the start):

    C_k(u_k) = Coss_in(u_k) + Coss_out(V - u_k) + C_extra
    du_k/dt  = -i / C_k(u_k)                        (free node)
    L_c di/dt = sum_k u_k - u_x                      (lumped LC regime)
    di/dt    = 0                                     (constant-current regime)

``i`` > 0 drives the node toward the incoming device's rail (ZVS direction).
A node at u = 0 with i >= 0 is clamped by the incoming device's reverse
conduction (its drop is ignored, declared); a node back at u = V with i <= 0
is clamped by the outgoing device's. Derivation and the claim boundary:
docs/specs/2026-11-20-switching-transition.md, docs/model-format.md §7c.

Everything here is a COMPUTED estimate from a datasheet table and declared
lumped parameters. It is verified against closed forms and ngspice; it is
not a measurement and is labelled accordingly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq

from pear_sim.errors import PearSimError
from pear_sim.spec.switching_transitions import CossCard, SwitchingTransitionSpec, TransitionLeg

__all__ = [
    "SCHEMA",
    "SwitchingTransitionError",
    "CossCurve",
    "TransitionResult",
    "build_switching_transitions_observation",
    "instant_cut_overshoot",
    "ramp_overshoot",
    "simulate_transition",
]

SCHEMA = "pear-sim/switching-transitions/v0.1"

CLAIM_BOUNDARY = (
    "Post-processed estimate on the recorded IDEAL run: the circuit equations are "
    "unchanged and the transition never feeds back into the waveforms. Coss/Qoss is the "
    "datasheet small-signal table (typical values; large-signal Qoss and Coss hysteresis "
    "of GaN/superjunction parts make the real charge and soft-switching loss larger). "
    "The commutating current is the recorded one at the outgoing turn-off; the lumped LC "
    "regime holds the declared far voltage constant through the window and uses one "
    "static commutation inductance. Reverse-conduction drop, gate-drive dynamics and "
    "Miller plateau, reverse recovery and EMI are not modelled. capacitive_turn_on_energy_j "
    "is not part of the loss_analysis switching energies (a datasheet E_on usually "
    "already contains it) and must never be summed with them. Overshoot values are "
    "hard-commutation estimates from a declared loop inductance, not envelopes."
)

VALIDATION_STATUS = "verified (closed form + ngspice), NOT validated against measurements"

_MAX_HORIZON_S = 1e-2


class SwitchingTransitionError(PearSimError):
    code = "analysis.switching_transitions_failed"


class CossCurve:
    """Coss(V) from a card, with the card's declared interpolation law.

    ``coss_f``: piecewise linear in V. ``qoss_c``: Qoss piecewise linear, so Coss
    is piecewise constant (right-continuous at breakpoints). Beyond the last axis
    point the last capacitance is held; callers count such lookups.
    """

    def __init__(self, card: CossCard):
        self.card = card
        self.axis = np.asarray(card.voltage_axis_v, dtype=float)
        self.v_max = float(self.axis[-1])
        if card.coss_f is not None:
            self._values = np.asarray(card.coss_f, dtype=float)
            self._piecewise_constant = False
        else:
            charge = np.asarray(card.qoss_c, dtype=float)
            self._values = np.diff(charge) / np.diff(self.axis)
            self._piecewise_constant = True

    def __call__(self, voltage: float) -> float:
        u = max(float(voltage), 0.0)
        if self._piecewise_constant:
            index = int(np.searchsorted(self.axis, u, side="right")) - 1
            return float(self._values[min(max(index, 0), len(self._values) - 1)])
        return float(np.interp(u, self.axis, self._values))

    def breakpoints(self, low: float, high: float) -> list[float]:
        return [float(x) for x in self.axis if low < x < high]

    def integral(self, low: float, high: float, weight: Callable[[float], float] | None = None) -> float:
        """∫ weight(w)·C(w) dw over [low, high] (numerical, breakpoint-aware)."""

        if high <= low:
            return 0.0
        integrand = (lambda w: self(w)) if weight is None else (lambda w: weight(w) * self(w))
        value, _ = quad(
            integrand, low, high, points=self.breakpoints(low, high) or None, limit=200,
            epsabs=0.0, epsrel=1e-10,
        )
        return float(value)

    def charge(self, voltage: float) -> float:
        return self.integral(0.0, voltage)

    def energy(self, voltage: float) -> float:
        return self.integral(0.0, voltage, lambda w: w)


@dataclass(frozen=True)
class TransitionResult:
    voltage_at_turn_on_v: tuple[float, ...]
    minimum_dead_time_s: float | None
    maximum_dead_time_s: float | None
    current_at_end_a: float
    volt_second_deficit_v_s: float
    horizon_s: float


def _node_capacitance(curve: CossCurve, bus: float, extra: float) -> Callable[[float], float]:
    return lambda u: curve(u) + curve(bus - u) + extra


def simulate_transition(
    capacitances: list[Callable[[float], float]],
    bus_voltage: float,
    current_a: float,
    dead_time_s: float,
    threshold_v: float,
    inductance_h: float | None = None,
    far_voltage_u: float = 0.0,
) -> TransitionResult:
    """Integrate the dead-time swing of one node (or a bridge pair) in the u-frame.

    ``current_a`` > 0 drives every node toward its incoming device's rail.
    With ``inductance_h`` None the current is held constant.
    """

    count = len(capacitances)
    v = float(bus_voltage)
    lumped = inductance_h is not None
    # state: u_1..u_N, i, q (q' = sum u: volt-seconds the ideal run did not see)
    state = np.array([v] * count + [float(current_a), 0.0])
    clamp = ["high" if current_a <= 0.0 else None for _ in range(count)]

    charge = sum(quad(c, 0.0, v, limit=200)[0] for c in capacitances)
    swing_time = charge / abs(current_a) if current_a != 0.0 else math.inf
    if lumped:
        c_max = max(c(0.0) for c in capacitances)
        period = 2.0 * math.pi * math.sqrt(inductance_h * c_max)
        horizon = max(3.0 * dead_time_s, 3.0 * period)
        characteristic = min(period, swing_time)
    elif current_a > 0.0:
        horizon = max(3.0 * dead_time_s, 3.0 * swing_time)
        characteristic = swing_time
    else:
        horizon = max(dead_time_s, 1e-15)
        characteristic = horizon
    horizon = min(horizon, _MAX_HORIZON_S)
    # Resolve the swing itself, not just the horizon: the dense output is what
    # the turn-on voltage, the volt-seconds and the threshold crossings read.
    step_cap = characteristic / 200.0
    if dead_time_s > 0.0:
        step_cap = min(step_cap, dead_time_s / 20.0)

    def rhs(_t: float, y: np.ndarray) -> np.ndarray:
        current = y[count]
        derivative = np.zeros_like(y)
        for k in range(count):
            if clamp[k] is None:
                derivative[k] = -current / capacitances[k](y[k])
        if lumped:
            derivative[count] = (float(np.sum(y[:count])) - far_voltage_u) / inductance_h
        derivative[count + 1] = float(np.sum(y[:count]))
        return derivative

    t = 0.0
    segments: list[tuple[float, float, Any]] = []
    guard = 0
    ever_all_low = False
    exited = False
    exit_time: float | None = None
    returned_high = False
    while t < horizon and guard < 64:
        guard += 1
        if t >= dead_time_s and (exited or returned_high):
            break  # everything the report needs has happened (lossless: it repeats)
        if not lumped and all(item is not None for item in clamp):
            break  # constant current, every node clamped: nothing moves any more
        if lumped and all(item is not None for item in clamp):
            # Every node clamped: u is constant and the current is a straight line,
            # di/dt = (sum u - u_x)/L. Step to the release analytically instead of
            # crawling through it at the swing's time step.
            slope = (float(np.sum(state[:count])) - far_voltage_u) / inductance_h
            sum_u = float(np.sum(state[:count]))
            releases_low = all(item == "low" for item in clamp) and slope < 0.0
            releases_high = all(item == "high" for item in clamp) and slope > 0.0
            if releases_low or releases_high:
                t_release = t + abs(state[count]) / abs(slope)
            else:
                t_release = horizon
            segments.append((t, t_release, _linear_segment(state, t, slope, sum_u, count)))
            state = _linear_segment(state, t, slope, sum_u, count)(t_release)
            t = t_release
            if not (releases_low or releases_high):
                break  # the current never turns: the clamp holds to the horizon
            state[count] = 0.0
            clamp = [None] * count
            if t >= dead_time_s and (exited or returned_high):
                break
            continue
        events: list[Callable[[float, np.ndarray], float]] = []
        tags: list[tuple[str, int]] = []
        # Every event level sits a hair OUTSIDE the value the state restarts
        # from (a node restarts at exactly 0 or V, the current at exactly 0):
        # an event function that is exactly zero at the start of a segment is
        # reported by solve_ivp as an event at that start, and the loop would
        # stall on it.
        eps_v = 1e-12 * max(v, 1.0)
        eps_i = 1e-12 * max(abs(current_a), 1e-3)
        for k in range(count):
            if clamp[k] is None:
                for tag, level, direction in (("low", -eps_v, -1), ("high", v + eps_v, 1)):
                    def event(_t, y, k=k, level=level):
                        return y[k] - level

                    event.terminal = True
                    event.direction = direction
                    events.append(event)
                    tags.append((tag, k))
        if lumped and ever_all_low and not exited:
            for k in range(count):
                if clamp[k] is None:
                    def exit_event(_t, y, k=k):
                        return y[k] - threshold_v

                    exit_event.terminal = True
                    exit_event.direction = 1
                    events.append(exit_event)
                    tags.append(("exit", k))
        if lumped and "low" in clamp:
            def release_low(_t, y):
                return y[count] + eps_i  # a low clamp lets go once i < 0

            release_low.terminal = True
            release_low.direction = -1
            events.append(release_low)
            tags.append(("current_zero", -1))
        if lumped and "high" in clamp:
            def release_high(_t, y):
                return y[count] - eps_i  # a high clamp lets go once i > 0

            release_high.terminal = True
            release_high.direction = 1
            events.append(release_high)
            tags.append(("current_zero", 1))
        scale = max(abs(v), 1.0)
        solution = solve_ivp(
            rhs,
            (t, horizon),
            state,
            method="RK45",
            dense_output=True,
            events=events or None,
            rtol=1e-10,
            atol=np.array([1e-9 * scale] * count + [1e-12 + 1e-10 * abs(current_a), 1e-30]),
            max_step=min((horizon - t) / 50.0, step_cap),
        )
        if solution.status < 0:
            raise SwitchingTransitionError(f"transition integration failed: {solution.message}")
        t_end = float(solution.t[-1])
        if t_end > t:
            segments.append((t, t_end, solution.sol))
        state = solution.y[:, -1].copy()
        t = t_end
        if solution.status != 1:
            break
        # Clamp EVERY free node that reached a rail in this step, not only the
        # one whose event stopped the solver: identical nodes of a bridge pair
        # reach the rail together, and an unclamped twin would run past it.
        for k in range(count):
            if clamp[k] is None and state[k] <= 0.0:
                state[k], clamp[k] = 0.0, "low"
            elif clamp[k] is None and state[k] >= v:
                state[k], clamp[k] = v, "high"
                returned_high = True
        if all(item == "low" for item in clamp):
            ever_all_low = True
        for index, times in enumerate(solution.t_events):
            if not (len(times) and abs(times[-1] - t_end) <= 1e-15 + 1e-12 * t_end):
                continue
            tag, k = tags[index]
            if tag == "exit":
                exited = True
                exit_time = t_end
                continue
            if tag == "low" and clamp[k] is None:
                # solve_ivp locates the root to its own tolerance: the state may
                # sit a hair short of the rail. The event, not the value, decides.
                state[k], clamp[k] = 0.0, "low"
                ever_all_low = all(item == "low" for item in clamp)
                continue
            if tag == "high" and clamp[k] is None:
                state[k], clamp[k] = v, "high"
                returned_high = True
                continue
            if tag in {"low", "high"}:
                continue
            else:
                released = "low" if k < 0 else "high"
                for j in range(count):
                    if clamp[j] == released:
                        clamp[j] = None

    if guard >= 64 and t < horizon:
        raise SwitchingTransitionError(
            "transition integration did not settle within 64 segments "
            f"(t = {t:.3e} s of a {horizon:.3e} s horizon); refusing to report a partial result"
        )

    settled_at = t

    def evaluate(at: float) -> np.ndarray:
        for start, end, dense in segments:
            if start - 1e-18 <= at <= end + 1e-18:
                return dense(min(max(at, start), end))
        # Past the last segment nothing moves (the loop only stops early when the
        # state is settled), but the volt-second integral keeps accruing sum(u).
        value = state.copy()
        if at > settled_at:
            value[count + 1] += float(np.sum(state[:count])) * (at - settled_at)
        return value

    def excess(at: float) -> float:
        return float(np.max(evaluate(at)[:count])) - threshold_v

    t_min = _first_crossing(segments, excess, below=True, after=0.0)
    if exit_time is not None:
        t_max = exit_time  # the event itself: a scan would sit on the threshold
    elif lumped and t_min is not None:
        t_max = _first_crossing(segments, excess, below=False, after=t_min)
    else:
        t_max = None
    at_turn_on = evaluate(dead_time_s)
    voltages = tuple(float(min(max(x, 0.0), v)) for x in at_turn_on[:count])
    end_time = min(dead_time_s, t_min) if t_min is not None else dead_time_s
    end_state = evaluate(end_time)
    return TransitionResult(
        voltage_at_turn_on_v=voltages,
        minimum_dead_time_s=t_min,
        maximum_dead_time_s=t_max,
        current_at_end_a=float(end_state[count]),
        volt_second_deficit_v_s=float(at_turn_on[count + 1]),
        horizon_s=horizon,
    )


def _linear_segment(state: np.ndarray, start: float, slope: float, sum_u: float, count: int):
    """Dense output of a fully clamped stretch: u fixed, i linear, q' = sum u."""

    origin = state.copy()

    def dense(at: float) -> np.ndarray:
        value = origin.copy()
        value[count] = origin[count] + slope * (at - start)
        value[count + 1] = origin[count + 1] + sum_u * (at - start)
        return value

    return dense


def _first_crossing(
    segments: list[tuple[float, float, Any]],
    excess: Callable[[float], float],
    *,
    below: bool,
    after: float,
) -> float | None:
    """First time >= ``after`` where ``excess`` is <= 0 (below) or > 0 (not below).

    Scanned on the dense solution of each segment, then refined by bisection;
    the state after the last segment is constant, so no later crossing exists.
    """

    def hit(t: float) -> bool:
        value = excess(t)
        return value <= 0.0 if below else value > 1e-9 * max(abs(value), 1.0)

    if not segments:
        return after if hit(after) else None
    for start, end, _dense in segments:
        if end < after:
            continue
        grid = np.linspace(max(start, after), end, 401)
        previous = grid[0]
        if hit(previous):
            return float(previous)
        for current in grid[1:]:
            if hit(current):
                return float(brentq(lambda x: excess(x), previous, current, xtol=1e-16, rtol=1e-12))
            previous = current
    return None


def capacitive_turn_on_energy(curve: CossCurve, bus: float, residual: float, extra: float) -> float:
    """Energy the incoming channel dissipates discharging the residual node voltage."""

    if residual <= 0.0:
        return 0.0
    incoming = curve.energy(residual)
    outgoing = curve.integral(bus - residual, bus, lambda w: bus - w)
    return incoming + outgoing + 0.5 * extra * residual * residual


def instant_cut_overshoot(curve: CossCurve, bus: float, current: float, loop_h: float) -> float | None:
    """ΔV with ½·Lσ·I² = ∫_V^{V+ΔV} (w − V)·Coss(w) dw; None when no root ≤ 10·V."""

    target = 0.5 * loop_h * current * current
    if target <= 0.0:
        return 0.0

    def residual(delta: float) -> float:
        return curve.integral(bus, bus + delta, lambda w: w - bus) - target

    limit = 10.0 * max(bus, 1.0)
    high = max(bus, 1.0) * 1e-3
    while residual(high) < 0.0:
        if high >= limit:
            return None
        # Doubling may step past the limit: clamp the last step ONTO it and test
        # there, or a root between the last bracket and the limit is lost.
        high = min(2.0 * high, limit)
    return float(brentq(residual, 0.0, high, xtol=1e-9 * max(bus, 1.0), rtol=1e-12))


def ramp_overshoot(curve: CossCurve, bus: float, current: float, loop_h: float, fall_s: float) -> float:
    """Peak ΔV of the lossless Lσ–Coss loop when the device current falls linearly."""

    period = 2.0 * math.pi * math.sqrt(loop_h * curve(bus))
    stop = fall_s + 1.5 * period

    def device_current(t: float) -> float:
        return current * (1.0 - t / fall_s) if t < fall_s else 0.0

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        x, loop_current = y
        return np.array(
            [(loop_current - device_current(t)) / curve(bus + x), -x / loop_h]
        )

    def crest(t: float, y: np.ndarray) -> float:
        return y[1] - device_current(t)  # dx/dt = 0: a local maximum of x

    crest.direction = -1
    peak = 0.0
    state = np.array([0.0, current])
    for start, end in ((0.0, fall_s), (fall_s, stop)):
        solution = solve_ivp(
            rhs,
            (start, end),
            state,
            method="RK45",
            events=[crest],
            rtol=1e-10,
            atol=[1e-9 * max(bus, 1.0), 1e-12 + 1e-10 * abs(current)],
            max_step=min(fall_s, period) / 200.0,
        )
        if solution.status < 0:
            raise SwitchingTransitionError(f"overshoot integration failed: {solution.message}")
        peak = max(peak, float(np.max(solution.y[0])))
        if solution.y_events and len(solution.y_events[0]):
            peak = max(peak, float(np.max(solution.y_events[0][:, 0])))
        state = solution.y[:, -1]
    return peak


# --- window detection --------------------------------------------------------------


@dataclass(frozen=True)
class _Window:
    t_off: float
    t_on: float
    falling: bool  # outgoing high, incoming low


# Every EFFECTIVE gate change: the solver records one event per control and
# timestamp, "forced_gate" when a perturbation window overrides the schedule.
# Reading only "scheduled_gate" silently drops a forced turn-off and its window.
_GATE_EVENT_KINDS = frozenset({"scheduled_gate", "forced_gate"})


def _switch_timeline(result: Any, control: str) -> list[tuple[float, bool]]:
    return [
        (float(event.time), bool(event.current))
        for event in result.events
        if event.kind in _GATE_EVENT_KINDS and event.target == control
        and bool(event.current) != bool(event.previous)
    ]


def _windows(result: Any, high: Any, low: Any) -> tuple[list[_Window], list[tuple[float, str]]]:
    """Dead-time windows of one leg, and every turn-off that opened none, with the reason.

    A turn-off is never dropped silently: either it opens a window, or it is
    returned here and lands in the leg's ``not_evaluated`` counts.
    """

    high_edges = _switch_timeline(result, high.control)
    low_edges = _switch_timeline(result, low.control)
    windows = []
    dropped: list[tuple[float, str]] = []
    for outgoing, incoming, falling in ((high_edges, low_edges, True), (low_edges, high_edges, False)):
        for index, (t_off, state) in enumerate(outgoing):
            if state:
                continue
            next_on = next((t for t, s in incoming if s and t >= t_off), None)
            if next_on is None:
                dropped.append((t_off, "window_cut_by_end_of_run"))
                continue
            outgoing_back = next((t for t, s in outgoing[index + 1:] if s), math.inf)
            if next_on > outgoing_back:
                dropped.append((t_off, "outgoing_back_on_before_incoming"))
                continue
            windows.append(_Window(t_off=t_off, t_on=next_on, falling=falling))
    return sorted(windows, key=lambda w: w.t_off), sorted(dropped)


def _pre_edge(
    signal: np.ndarray, time: np.ndarray, t_edge: float, step: float, event_times: np.ndarray
) -> float:
    """Value just before ``t_edge``: linear extrapolation over the last two pre-edge
    samples when no other event lies between them and the edge, else the last sample."""

    base = int(np.searchsorted(time, t_edge - 0.25 * step))
    pre = max(base - 1, 0)
    if pre < 1:
        return float(signal[pre])
    t0, t1 = float(time[pre - 1]), float(time[pre])
    lo = int(np.searchsorted(event_times, t0, side="right"))
    hi = int(np.searchsorted(event_times, t_edge - 1e-6 * step, side="left"))
    if hi > lo:
        return float(signal[pre])
    slope = (float(signal[pre]) - float(signal[pre - 1])) / (t1 - t0)
    return float(signal[pre]) + slope * (t_edge - t1)


def _voltage(result: Any, terms: tuple[tuple[str, float], ...], at: Callable[[np.ndarray], float]) -> float:
    return float(sum(coefficient * at(np.asarray(result.signals[name], dtype=float)) for name, coefficient in terms))


# --- producer --------------------------------------------------------------------


def build_switching_transitions_observation(result: Any) -> dict[str, Any]:
    model = result.model
    spec: SwitchingTransitionSpec | None = getattr(model, "switching_transitions", None)
    if spec is None:
        raise SwitchingTransitionError("model declares no switching_transitions section")
    time = np.asarray(result.time, dtype=float)
    if len(time) < 3:
        raise SwitchingTransitionError("switching transitions need at least three recorded samples")
    step = float(model.transient.output_step)
    start, stop = float(model.transient.start), float(model.transient.stop)
    window_start = 0.5 * (start + stop)
    tolerance = max(1e-15, 1e-9 * (stop - start))
    event_times = np.array(sorted(float(event.time) for event in result.events))
    components = {component.id: component for component in model.components}
    curves = {card_id: CossCurve(card) for card_id, card in spec.cards.items()}
    legs = {leg.id: leg for leg in spec.legs}
    partners = {leg.bridge_partner: leg.id for leg in spec.legs if leg.bridge_partner}

    leg_windows: dict[str, list[_Window]] = {}
    leg_dropped: dict[str, list[str]] = {}
    for leg in spec.legs:
        windows, dropped = _windows(result, components[leg.high], components[leg.low])
        leg_windows[leg.id] = [w for w in windows if w.t_off >= window_start]
        leg_dropped[leg.id] = [reason for t_off, reason in dropped if t_off >= window_start]

    def at_edge(t_edge: float) -> Callable[[np.ndarray], float]:
        return lambda signal: _pre_edge(signal, time, t_edge, step, event_times)

    def load_current(leg: TransitionLeg, at: Callable[[np.ndarray], float]) -> float:
        def sig(name: str) -> float:
            return at(np.asarray(result.signals[f"i({name})"], dtype=float))

        return (sig(leg.high) - sig(leg.high_diode)) - (sig(leg.low) - sig(leg.low_diode))

    # Simultaneous-window detection across evaluation units (a leg, or a bridge pair).
    units = [leg.id for leg in spec.legs if leg.id not in partners]
    unit_intervals = {
        unit: [(w.t_off, w.t_on) for w in leg_windows[unit]] for unit in units
    }

    def overlaps(unit: str, window: _Window) -> str | None:
        for other in units:
            if other == unit or other == legs[unit].bridge_partner:
                continue
            for t_off, t_on in unit_intervals[other]:
                if t_off <= window.t_on + tolerance and window.t_off <= t_on + tolerance:
                    return other
        return None

    payload_legs = []
    for unit in units:
        leg = legs[unit]
        partner = legs[leg.bridge_partner] if leg.bridge_partner else None
        curve = curves[leg.card_id]
        partner_curve = curves[partner.card_id] if partner else None
        edges = []
        skipped: dict[str, int] = {}
        cut_by_end = 0
        for reason in leg_dropped[unit] + (leg_dropped[partner.id] if partner else []):
            if reason == "window_cut_by_end_of_run":
                # The incoming turn-on lies beyond the run: no transition inside
                # the run was skipped. Counted, but it does not make the verdict partial.
                cut_by_end += 1
                continue
            skipped[reason] = skipped.get(reason, 0) + 1
        extrapolated = 0
        for window in leg_windows[unit]:
            other = overlaps(unit, window)
            if other is not None:
                skipped["simultaneous_transition_unpaired"] = skipped.get(
                    "simultaneous_transition_unpaired", 0
                ) + 1
                continue
            partner_window = None
            if partner is not None:
                partner_window = next(
                    (
                        w
                        for w in leg_windows[partner.id]
                        if abs(w.t_off - window.t_off) <= tolerance
                        and abs(w.t_on - window.t_on) <= tolerance
                    ),
                    None,
                )
                if partner_window is None or partner_window.falling == window.falling:
                    skipped["partner_window_mismatch"] = skipped.get("partner_window_mismatch", 0) + 1
                    continue
            at = at_edge(window.t_off)
            bus = abs(_voltage(result, leg.bus_voltage, at))
            if bus <= 0.0:
                skipped["zero_bus_voltage"] = skipped.get("zero_bus_voltage", 0) + 1
                continue
            i_load = load_current(leg, at)
            current_u = i_load if window.falling else -i_load
            dead = window.t_on - window.t_off
            far = _voltage(result, leg.far_voltage, at) if leg.far_voltage else 0.0
            if partner is None:
                u_x = far if window.falling else bus - far
            else:
                u_x = bus + (far if window.falling else -far)
            capacitances = [_node_capacitance(curve, bus, leg.extra_node_capacitance_f)]
            if partner is not None:
                capacitances.append(
                    _node_capacitance(partner_curve, bus, partner.extra_node_capacitance_f)
                )
            threshold_v = spec.zvs_threshold * bus
            outcome = simulate_transition(
                capacitances,
                bus,
                current_u,
                dead,
                threshold_v,
                inductance_h=leg.commutation_inductance_h,
                far_voltage_u=u_x,
            )
            edge_extrapolated = bus > curve.v_max or (
                partner_curve is not None and bus > partner_curve.v_max
            )
            residuals = outcome.voltage_at_turn_on_v
            energy = capacitive_turn_on_energy(curve, bus, residuals[0], leg.extra_node_capacitance_f)
            eoss = curve.energy(residuals[0])
            if partner is not None:
                energy += capacitive_turn_on_energy(
                    partner_curve, bus, residuals[1], partner.extra_node_capacitance_f
                )
                eoss += partner_curve.energy(residuals[1])
            zvs = max(residuals) <= threshold_v * (1 + 1e-9)
            # The recorded IDEAL run puts each node, for the whole window, on the
            # rail its freewheeling diode picks at the edge: the incoming rail
            # (u = 0) when the current drives toward ZVS, the outgoing rail (u = V)
            # otherwise. The deficit is the transition MINUS that, in the same
            # orientation, so a node both runs hold at the same rail adds nothing.
            ideal_volt_seconds = 0.0 if current_u > 0.0 else len(capacitances) * bus * dead
            edge: dict[str, Any] = {
                "time_s": window.t_off,
                "direction": "high_to_low" if window.falling else "low_to_high",
                "outgoing": [leg.high if window.falling else leg.low]
                + ([partner.low if window.falling else partner.high] if partner else []),
                "incoming": [leg.low if window.falling else leg.high]
                + ([partner.high if window.falling else partner.low] if partner else []),
                "bus_voltage_v": bus,
                "commutating_current_a": current_u,
                "dead_time_s": dead,
                "zvs": bool(zvs),
                "voltage_at_turn_on_v": max(residuals),
                "node_voltages_at_turn_on_v": list(residuals),
                "minimum_dead_time_s": outcome.minimum_dead_time_s,
                "maximum_dead_time_s": outcome.maximum_dead_time_s,
                "capacitive_turn_on_energy_j": energy,
                "incoming_eoss_j": eoss,
                "current_at_end_a": outcome.current_at_end_a,
                "transition_fraction_of_dead_time": (
                    outcome.minimum_dead_time_s / dead
                    if outcome.minimum_dead_time_s is not None and dead > 0.0
                    else None
                ),
                "volt_second_deficit_v_s": outcome.volt_second_deficit_v_s - ideal_volt_seconds,
                "extrapolated_lookup": bool(edge_extrapolated),
                "overshoot": None,
            }
            if leg.regime == "lumped_lc":
                edge["far_voltage_u_v"] = u_x
            if leg.loop_inductance_h is not None:
                # The turn-off at t_off interrupts the current flowing AT t_off:
                # that current decides whether the outgoing device ends up
                # blocking, and sizes the overshoot. A current the resonance
                # builds later in the window belongs to a later commutation.
                edge["overshoot"] = _overshoot(curve, bus, current_u, leg, capacitances[0])
                if edge["overshoot"].get("extrapolated_lookup"):
                    edge_extrapolated = True
            extrapolated += int(edge_extrapolated)
            edges.append(edge)
        payload_legs.append(
            {
                "leg": unit,
                "bridge_partner": leg.bridge_partner,
                "regime": leg.regime,
                "card": leg.card_id,
                "card_law": curve.card.law,
                "provenance": curve.card.provenance.as_dict(),
                "commutation_inductance_h": leg.commutation_inductance_h,
                "loop_inductance_h": leg.loop_inductance_h,
                "current_fall_time_s": leg.current_fall_time_s,
                "edges": edges,
                "not_evaluated": skipped,
                "windows_cut_by_end_of_run": cut_by_end,
                "extrapolated_lookups": extrapolated,
                "summary": _summary(edges, stop - window_start),
            }
        )
    zvs_total = sum(leg["summary"]["zvs_edges"] for leg in payload_legs)
    edge_total = sum(leg["summary"]["edges"] for leg in payload_legs)
    skipped_total = sum(sum(leg["not_evaluated"].values()) for leg in payload_legs)
    # A verdict over ALL edges exists only when every window was evaluated. With
    # windows skipped, a failing evaluated edge still proves "not all ZVS", but
    # "all evaluated passed" proves nothing about the skipped ones: null, partial.
    if edge_total > 0 and zvs_total < edge_total:
        all_zvs: bool | None = False
    elif edge_total > 0 and skipped_total == 0:
        all_zvs = True
    else:
        all_zvs = None
    status = "ok" if skipped_total == 0 and edge_total > 0 else (
        "partial" if edge_total > 0 else "not_evaluated"
    )
    return {
        "schema": SCHEMA,
        "status": status,
        "claim_boundary": CLAIM_BOUNDARY,
        "validation_status": VALIDATION_STATUS,
        "zvs_threshold_fraction": spec.zvs_threshold,
        "evaluation_window_s": [window_start, stop],
        "legs": payload_legs,
        "summary": {
            "edges": edge_total,
            "zvs_edges": zvs_total,
            "not_evaluated_windows": skipped_total,
            "all_zvs": all_zvs,
        },
    }


def _overshoot(
    curve: CossCurve,
    bus: float,
    current: float,
    leg: TransitionLeg,
    node_capacitance: Callable[[float], float],
) -> dict[str, Any]:
    loop = float(leg.loop_inductance_h)
    if current <= 0.0:
        return {
            "status": "not_evaluated",
            "reason": "current commutates into the outgoing device's own diode; the stress is "
            "the incoming device's hard turn-on (reverse recovery: out of scope)",
        }
    delta = instant_cut_overshoot(curve, bus, current, loop)
    payload: dict[str, Any] = {
        "status": "ok",
        "loop_inductance_h": loop,
        "current_a": current,
        "vpeak_instant_cut_v": None if delta is None else bus + delta,
        "extrapolated_lookup": bool(delta is None or bus + delta > curve.v_max),
    }
    fall = leg.current_fall_time_s
    if fall is not None:
        payload["current_fall_time_s"] = fall
        payload["vpeak_conventional_v"] = bus + loop * current / fall
        node_charge, _ = quad(node_capacitance, 0.0, bus, limit=200)
        applicable = fall >= node_charge / current
        payload["ramp_applicable"] = bool(applicable)
        payload["vpeak_ramp_v"] = (
            bus + ramp_overshoot(curve, bus, current, loop, fall) if applicable else None
        )
    return payload


def _summary(edges: list[dict[str, Any]], window_s: float) -> dict[str, Any]:
    if not edges:
        return {"edges": 0, "zvs_edges": 0}
    by_direction: dict[str, Any] = {}
    for direction in ("high_to_low", "low_to_high"):
        subset = [edge for edge in edges if edge["direction"] == direction]
        if not subset:
            continue
        minimums = [edge["minimum_dead_time_s"] for edge in subset]
        maximums = [edge["maximum_dead_time_s"] for edge in subset if edge["maximum_dead_time_s"] is not None]
        by_direction[direction] = {
            "edges": len(subset),
            "zvs_edges": sum(edge["zvs"] for edge in subset),
            "worst_voltage_at_turn_on_v": max(edge["voltage_at_turn_on_v"] for edge in subset),
            "max_minimum_dead_time_s": None if any(m is None for m in minimums) else max(minimums),
            "min_maximum_dead_time_s": min(maximums) if maximums else None,
            "commutating_current_a": [
                min(edge["commutating_current_a"] for edge in subset),
                max(edge["commutating_current_a"] for edge in subset),
            ],
        }
    peaks = {
        key: max(
            (edge["overshoot"][key] for edge in edges if edge["overshoot"] and edge["overshoot"].get(key) is not None),
            default=None,
        )
        for key in ("vpeak_instant_cut_v", "vpeak_conventional_v", "vpeak_ramp_v")
    }
    energy = sum(edge["capacitive_turn_on_energy_j"] for edge in edges)
    return {
        "edges": len(edges),
        "zvs_edges": sum(edge["zvs"] for edge in edges),
        "by_direction": by_direction,
        "capacitive_turn_on_energy_total_j": energy,
        "capacitive_turn_on_power_w": energy / window_s if window_s > 0 else None,
        "max_overshoot": {key: value for key, value in peaks.items() if value is not None} or None,
    }
