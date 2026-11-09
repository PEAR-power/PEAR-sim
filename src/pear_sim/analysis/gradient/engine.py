# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Gradient backend core: monodromy, shooting, metric sensitivities.

For a DECLARED mode sequence whose every exit is a scheduled gate edge, each
mode is LTI with constant input (v1 scope: DC sources only), so the one-period
map is affine and exact:

    x_{k+1} = Phi_k x_k + psi_k,   Phi_k = expm(A_k tau_k),
    psi_k   = (integral_0^tau expm(A_k s) ds) d_k        (van Loan blocks)

The monodromy matrix is the product of the Phi_k and the periodic steady state
is ONE linear solve, (I - M) x* = b — Newton shooting degenerates to its exact
special case. Integral metrics (mean, rms) are evaluated on the closed-form
trajectory: first moments through augmented matrix exponentials, second
moments by Gauss-Legendre quadrature per segment, both at machine precision
for these smooth-exponential segments.

Sensitivities are central finite differences ON THE
COMPILATION STEP — perturb theta, recompile the modes, recompute the segment
durations from the CONTROL LAW (this is where the duty and switching-frequency
derivatives live), redo the affine algebra. The oracle is never re-run inside
the engine; the full oracle appears only as the REFERENCE in the T2 gate.

The engine never issues a verdict. A violated assumption — a diode asked to
conduct backwards by the declared sequence, a schedule whose shape no longer
matches the declaration — comes back as an :class:`AssumptionFlag` with a
signed margin, not as an exception (the T3 contract). The oracle remains the
only judge of physical truth; this module only points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm

from pear_sim.analysis.gradient.design_vector import DesignVector
from pear_sim.analysis.gradient.sequence import ExitKind, ModeSequence
from pear_sim.analysis.steady_state import _shooting_period
from pear_sim.core.circuit import Circuit
from pear_sim.core.scheduling import _initial_gate_values, scheduled_edges
from pear_sim.errors import PearSimError

__all__ = [
    "AssumptionFlag",
    "GradientEngineError",
    "GradientResult",
    "MetricJacobian",
    "MetricSpec",
    "SteadyStatePoint",
    "solve",
]

_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(24)
_RELATIVE_FD_STEP = 1e-6
_ZERO_DURATION = 1e-15
# A declared conduction state whose guard sits EXACTLY on zero at a swap
# instant (a dead-time hand-over) is healthy, and round-off may land it a
# hair below. The flag's verdict therefore carries an explicit tolerance
# instead of an exact sign test; the margin itself is reported unrounded.
_GUARD_MARGIN_TOLERANCE = 1e-9


class GradientEngineError(PearSimError):
    """The model or sequence is outside the engine's declared v1 scope."""

    code = "gradient.engine_invalid"


@dataclass(frozen=True)
class MetricSpec:
    """One integral metric: a model output and a statistic over the period."""

    output: str
    statistic: str  # "mean" | "rms"

    @property
    def name(self) -> str:
        return f"{self.statistic}({self.output})"


@dataclass(frozen=True)
class AssumptionFlag:
    """A validated-or-not assumption with its signed margin.

    ``held`` is the verdict on the ASSUMPTION, never on the physics; a
    negative ``margin`` quantifies how badly it failed in the guard's own
    units. Flags are returned, not raised — an optimizer wants to know it
    walked out of the declared region, not to die there.
    """

    name: str
    held: bool
    margin: float
    detail: str


@dataclass(frozen=True)
class SteadyStatePoint:
    converged: bool
    iterations: int
    residual_inf: float
    period_s: float
    state_names: tuple[str, ...]
    state: tuple[float, ...]
    segment_durations_s: tuple[float, ...]
    segment_boundary_states: tuple[tuple[float, ...], ...]
    """States at every step boundary, cycle start first, cycle end last."""


@dataclass(frozen=True)
class MetricJacobian:
    metric_names: tuple[str, ...]
    parameter_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    """``values[i][j]`` = d metric_i / d theta_j, SI over SI."""

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=float)


@dataclass(frozen=True)
class GradientResult:
    parameter_names: tuple[str, ...]
    parameter_values: tuple[float, ...]
    steady_state: SteadyStatePoint
    metrics: dict[str, float]
    jacobian: MetricJacobian | None
    flags: tuple[AssumptionFlag, ...]
    sequence_source_run_sha256: str | None
    claim_boundary: str
    jacobian_labels: tuple[str, ...] = ()
    """Declared omissions the Jacobian carries, e.g. the open thermal loop."""

    def flag(self, name: str) -> AssumptionFlag:
        for candidate in self.flags:
            if candidate.name == name:
                return candidate
        raise KeyError(name)


def _scheduled_timeline(
    model, period_s: float
) -> tuple[tuple[float, dict[str, bool]], ...]:
    """One period of gate intervals, (duration, gates) each, from the law.

    Recomputed from the model's controls at the CURRENT theta — this, not the
    declared nominal durations, is what makes d/d(duty) and d/d(fsw) real
    derivatives instead of zeros.
    """

    gates = dict(_initial_gate_values(model.controls, 0.0))
    edges = scheduled_edges(model.controls, 0.0, period_s)
    intervals: list[tuple[float, dict[str, bool]]] = []
    cursor = 0.0
    for time, control_id, value in sorted(edges, key=lambda item: item[0]):
        if time >= period_s:
            break
        if time - cursor > _ZERO_DURATION:
            intervals.append((time - cursor, dict(gates)))
            cursor = time
        gates[control_id] = value
    if period_s - cursor > _ZERO_DURATION:
        intervals.append((period_s - cursor, dict(gates)))
    return tuple(intervals)


def _segment_operators(
    matrix: np.ndarray, drive: np.ndarray, duration: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(Phi, psi, S, s2) for one LTI segment via one augmented ``expm``.

    Phi = e^{A tau}; psi advances the state past the constant drive;
    S = integral of e^{A s} ds and s2 = integral of S(t) d dt feed the exact
    first moment of the trajectory, so ``mean`` carries no quadrature error
    at all.
    """

    n = matrix.shape[0]
    augmented = np.zeros((2 * n + 1, 2 * n + 1))
    augmented[:n, :n] = matrix
    augmented[:n, n : 2 * n] = np.eye(n)
    augmented[n : 2 * n, 2 * n] = drive
    block = expm(augmented * duration)
    phi = block[:n, :n]
    s_operator = block[:n, n : 2 * n]
    s2_drive = block[:n, 2 * n]
    psi = s_operator @ drive
    return phi, psi, s_operator, s2_drive


def solve(
    vector: DesignVector,
    sequence: ModeSequence,
    metrics: tuple[MetricSpec, ...],
    *,
    jacobian: bool = True,
    loss_metrics: bool = False,
    smooth_loss_tables: bool = False,
) -> GradientResult:
    """Steady state and metric sensitivities for a declared sequence.

    Fail-closed v1 scope: DC sources, plain scheduled edges, no machines, no
    saturable inductors, no zero-current reductions in the declared steps.
    Everything else is an ERROR here — not because it is unimportant, but
    because a silently wrong sensitivity is worse than a refusal.
    """

    for spec in metrics:
        if spec.statistic not in ("mean", "rms"):
            raise GradientEngineError(
                f"metric statistic {spec.statistic!r} is not an integral "
                "statistic; v1 supports mean and rms only (ripple/min/max have "
                "a different smoothness class and are out of scope)"
            )
    for step in sequence.steps:
        if step.exit_condition.kind is ExitKind.SATURATION:
            raise GradientEngineError(
                "saturation-boundary exits are outside the engine scope "
                "(sat_inductor is refused at model level too)"
            )
        if step.saturated_inductors:
            raise GradientEngineError(
                "declared steps with saturated inductors are outside the "
                "engine scope"
            )

    result = _evaluate(
        vector,
        sequence,
        metrics,
        loss_metrics=loss_metrics,
        smooth_loss_tables=smooth_loss_tables,
    )
    if not jacobian:
        return result

    # Loss metrics ride the same finite-difference loop as the declared
    # integral metrics: everything the base evaluation reports gets a row.
    metric_names = tuple(result.metrics.keys())
    names = vector.names()
    theta = vector.as_array()
    rows: list[list[float]] = [[] for _ in metric_names]
    for j, name in enumerate(names):
        step_size = _RELATIVE_FD_STEP * max(abs(theta[j]), 1e-12)
        plus = _evaluate(
            vector.with_offset(name, +step_size),
            sequence,
            metrics,
            loss_metrics=loss_metrics,
            smooth_loss_tables=smooth_loss_tables,
        )
        minus = _evaluate(
            vector.with_offset(name, -step_size),
            sequence,
            metrics,
            loss_metrics=loss_metrics,
            smooth_loss_tables=smooth_loss_tables,
        )
        for i, metric_name in enumerate(metric_names):
            derivative = (
                plus.metrics[metric_name] - minus.metrics[metric_name]
            ) / (2.0 * step_size)
            rows[i].append(derivative)

    return GradientResult(
        parameter_names=result.parameter_names,
        parameter_values=result.parameter_values,
        steady_state=result.steady_state,
        metrics=result.metrics,
        jacobian=MetricJacobian(
            metric_names=metric_names,
            parameter_names=names,
            values=tuple(tuple(row) for row in rows),
        ),
        flags=result.flags,
        sequence_source_run_sha256=sequence.source_run_sha256,
        claim_boundary=result.claim_boundary,
        jacobian_labels=result.jacobian_labels,
    )



_COINCIDENCE_TOLERANCE_S = 1e-12
_FIXED_POINT_TOLERANCE = 1e-12
_MAX_FIXED_POINT_ITERATIONS = 60


def _flow_invariants(circuit, modes) -> np.ndarray:
    """Rows of exact linear invariants shared by EVERY mode of the sequence.

    A node where only inductors meet (a two-inductor series chain, a floating
    star) is index-reduced by the kernel: its KCL row becomes a conservation
    law the flow preserves EXACTLY, so the monodromy has a structural unit
    eigenvalue along it and the periodic fixed point is isolated only on the
    invariant's level set. The engine must therefore solve on that subspace —
    the level value comes from the model's declared initial state, exactly as
    the oracle validates it.
    """

    n = len(circuit.state_components)
    rows: list[np.ndarray] = []
    for mode in modes:
        for _node, terms in getattr(mode, "star_constraints", ()):
            row = np.zeros(n)
            for state_index, sign in terms:
                row[state_index] = sign
            rows.append(row)
    if not rows:
        return np.zeros((0, n))
    # Deduplicate up to sign and keep only directions conserved by EVERY
    # mode's flow (a constraint private to one mode is not a cycle invariant).
    unique: list[np.ndarray] = []
    for row in rows:
        unit = row / np.linalg.norm(row)
        if any(abs(abs(unit @ keep) - 1.0) < 1e-12 for keep in unique):
            continue
        if all(
            float(np.max(np.abs(unit @ mode.a))) < 1e-9
            and abs(float(unit @ mode.c)) < 1e-9
            for mode in modes
        ):
            unique.append(unit)
    return np.asarray(unique) if unique else np.zeros((0, n))


def _constrained_solve(system, rhs, invariants, values):
    """Solve system @ x = rhs on the affine set invariants @ x = values.

    The stacked least-squares system is consistent by construction (the flow
    preserves the invariants, so rhs already lies in the compatible range);
    lstsq resolves the structural rank deficiency instead of blowing up on
    the exact unit eigenvalue.
    """

    if len(invariants) == 0:
        return np.linalg.solve(system, rhs)
    stacked = np.vstack([system, invariants])
    target = np.concatenate([rhs, values])
    solution, _residual, rank, _sv = np.linalg.lstsq(stacked, target, rcond=None)
    if rank < system.shape[0]:
        raise GradientEngineError(
            "the period map is rank-deficient beyond its declared flow "
            "invariants; the steady state is not isolated"
        )
    return solution


def _match_plan(timeline, sequence):
    """Group declared steps into the scheduled intervals they live in.

    Every scheduled interval may contain several declared steps: all but the
    last exit through a NATURAL guard (the physics splits the interval), the
    last one exits on the clock. Gates must agree throughout, because a
    natural commutation never moves a gate. Returns ``None`` on any mismatch
    — the caller turns that into the declared_sequence_held flag.
    """

    steps = list(sequence.steps)
    cursor = 0
    plan: list[tuple[float, list[int]]] = []
    for duration, gates in timeline:
        group: list[int] = []
        while True:
            if cursor >= len(steps):
                return None
            step = steps[cursor]
            if dict(step.gates) != gates:
                return None
            group.append(cursor)
            cursor += 1
            kind = step.exit_condition.kind
            if kind in (ExitKind.SCHEDULED_GATE, ExitKind.PERIOD_END):
                break
            if kind is not ExitKind.NATURAL_GUARD:
                return None
        plan.append((duration, group))
    if cursor != len(steps):
        return None
    return plan


def _exit_guard_row(circuit, mode, step):
    """Affine row of the guard whose zero crossing ends a natural step.

    ``guard_is_current`` in the declared exit picks the convention: leaving
    conduction crosses the branch-current guard of the still-conducting
    diode; entering conduction crosses its voltage guard.
    """

    target = step.exit_condition.target
    diode = next(d for d in circuit.diodes if d.id == target)
    conducting = bool(step.exit_condition.guard_is_current)
    coefficients, constant, _sources = mode.diode_guard_affine(diode, conducting)
    return np.asarray(coefficients, dtype=float), float(constant)


def _solve_natural_exit(mode, guard_row, guard_const, x0, remaining, healthy_sign):
    """FIRST zero crossing of the exit guard in (0, remaining].

    The guard along an LTI segment is smooth but OSCILLATORY — an
    under-damped LC pair rings, so the guard can cross zero, dive, and
    return above zero before the scheduled edge. Endpoint sign tests
    therefore miss even-count crossings (measured on buck_dcm, where
    g(0) > 0 and g(40 us) > 0 bracket a crossing at 0.9 us); the search
    scans a declared 128-point grid for the first sign change and refines
    it by Brent plus one analytic-derivative Newton polish. A crossing
    thinner than the grid spacing would need the smooth guard to dip and
    recover within one cell — outside this circuit class; the resolution
    is declared here rather than silently assumed.

    Returns (tau, status): "ok", "coincident" (the crossing lands on the
    scheduled edge within tolerance) or "missing" (the guard never leaves
    the healthy side — the declaration does not hold at this theta).
    """

    def guard_at(t: float) -> float:
        phi, psi, _, _ = _segment_operators(mode.a, mode.c, t)
        return float(guard_row @ (phi @ x0 + psi) + guard_const)

    start_value = float(guard_row @ x0 + guard_const)
    if healthy_sign * start_value <= 0.0:
        # The step begins already at/past its own exit: a zero-length step.
        return 0.0, "coincident" if abs(start_value) <= 1e-9 else "ok"

    samples = 128
    grid = np.linspace(0.0, remaining, samples + 1)
    previous_t, previous_g = 0.0, start_value
    bracket = None
    for t in grid[1:]:
        value = guard_at(float(t))
        if healthy_sign * value <= 0.0:
            bracket = (previous_t, float(t), previous_g, value)
            break
        previous_t, previous_g = float(t), value
    if bracket is None:
        end_value = guard_at(remaining)
        if abs(end_value) <= 1e-9:
            return remaining, "coincident"
        return remaining, "missing"

    low, high, _, high_value = bracket
    if high_value == 0.0:
        tau = high
    else:
        from scipy.optimize import brentq

        tau = float(brentq(guard_at, low, high, xtol=1e-18, rtol=8.9e-16))
    # One Newton polish on the analytic slope tightens the bracketed root.
    phi, psi, _, _ = _segment_operators(mode.a, mode.c, tau)
    state = phi @ x0 + psi
    slope = float(guard_row @ (mode.a @ state + mode.c))
    if slope != 0.0:
        tau -= float(guard_row @ state + guard_const) / slope
        tau = min(max(tau, 0.0), remaining)
    if remaining - tau <= _COINCIDENCE_TOLERANCE_S:
        return tau, "coincident"
    return tau, "ok"


def _evaluate(
    vector: DesignVector,
    sequence: ModeSequence,
    metrics: tuple[MetricSpec, ...],
    *,
    loss_metrics: bool = False,
    smooth_loss_tables: bool = False,
) -> GradientResult:
    model = vector.load()
    circuit = Circuit(model)
    if circuit.waveform_sources:
        raise GradientEngineError(
            "the v1 engine supports DC sources only; waveform sources make "
            "the period map time-varying"
        )
    if model.machines:
        raise GradientEngineError("machines are outside the v1 engine scope")
    if circuit.sat_inductors:
        raise GradientEngineError(
            "saturable inductors are outside the v1 engine scope"
        )

    period = _shooting_period(model)
    timeline = _scheduled_timeline(model, period)

    flags: list[AssumptionFlag] = []
    plan = _match_plan(timeline, sequence)
    def infeasible(detail: str, margin: float) -> GradientResult:
        # A declaration that does not hold at this theta is a RESULT, not a
        # crash: the caller's optimizer must see the boundary it walked
        # across, with NaN metrics instead of fiction.
        flags.append(
            AssumptionFlag(
                name="declared_sequence_held",
                held=False,
                margin=margin,
                detail=detail,
            )
        )
        return GradientResult(
            parameter_names=vector.names(),
            parameter_values=tuple(float(v) for v in vector.as_array()),
            steady_state=SteadyStatePoint(
                converged=False,
                iterations=0,
                residual_inf=float("nan"),
                period_s=period,
                state_names=tuple(c.id for c in circuit.state_components),
                state=(),
                segment_durations_s=(),
                segment_boundary_states=(),
            ),
            metrics={spec.name: float("nan") for spec in metrics},
            jacobian=None,
            flags=tuple(flags),
            sequence_source_run_sha256=sequence.source_run_sha256,
            claim_boundary=_CLAIM,
        )

    if plan is None:
        return infeasible(
            f"the control law produces {len(timeline)} scheduled interval(s) "
            f"that do not match the declared {len(sequence.steps)} step(s); "
            "metrics and Jacobian are not computable under this declaration",
            -1.0,
        )

    modes = tuple(
        circuit.compile(
            dict(step.gates), dict(step.diodes), step.constrained_inductors, ()
        )
        for step in sequence.steps
    )
    natural_exits = {
        index: _exit_guard_row(circuit, modes[index], step)
        for index, step in enumerate(sequence.steps)
        if step.exit_condition.kind is ExitKind.NATURAL_GUARD
    }

    n = len(circuit.state_components)
    identity = np.eye(n)

    coincidences: list[str] = []
    missing_crossing: list[str] = []

    def propagate(x_start):
        """One pass of the cycle map: state, monodromy (with saltation),
        realized durations and boundary states."""

        coincidences.clear()
        missing_crossing.clear()
        x = np.asarray(x_start, dtype=float)
        monodromy = identity.copy()
        realized: list[float] = []
        boundaries = [x.copy()]
        for interval_duration, group in plan:
            remaining = interval_duration
            for position, index in enumerate(group):
                mode = modes[index]
                last = position == len(group) - 1
                if last:
                    tau = remaining
                else:
                    guard_row, guard_const = natural_exits[index]
                    healthy = (
                        1.0
                        if sequence.steps[index].exit_condition.guard_is_current
                        else -1.0
                    )
                    tau, status = _solve_natural_exit(
                        mode, guard_row, guard_const, x, remaining, healthy
                    )
                    if status == "missing":
                        phi_r, psi_r, _, _ = _segment_operators(
                            mode.a, mode.c, remaining
                        )
                        end_guard = float(
                            guard_row @ (phi_r @ x + psi_r) + guard_const
                        )
                        missing_crossing.append(
                            f"step {index}: the declared natural exit on "
                            f"{sequence.steps[index].exit_condition.target!r} "
                            f"never fires within its interval (guard still "
                            f"{end_guard:.6g} at the scheduled edge)"
                        )
                        tau = remaining
                    elif status == "coincident":
                        coincidences.append(
                            f"step {index}: natural exit lands on the "
                            f"scheduled edge within "
                            f"{_COINCIDENCE_TOLERANCE_S:.0e} s; ordering "
                            "resolved deterministically (natural first)"
                        )
                phi, psi, _, _ = _segment_operators(mode.a, mode.c, tau)
                x = phi @ x + psi
                monodromy = phi @ monodromy
                if not last and not missing_crossing:
                    # Saltation at the state-continuous, dynamics-jumping
                    # crossing: S = I + (f+ - f-) grad_h / (grad_h . f-).
                    guard_row, _ = natural_exits[index]
                    post = modes[group[position + 1]]
                    f_minus = mode.a @ x + mode.c
                    f_plus = post.a @ x + post.c
                    denominator = float(guard_row @ f_minus)
                    if abs(denominator) < 1e-30:
                        raise GradientEngineError(
                            "grazing crossing: the guard's flow derivative "
                            "vanishes at the exit, so the saltation matrix "
                            "is undefined (a tangential commutation is "
                            "outside the engine's differentiability claim)"
                        )
                    monodromy = (
                        identity
                        + np.outer(f_plus - f_minus, guard_row) / denominator
                    ) @ monodromy
                realized.append(tau)
                boundaries.append(x.copy())
                remaining -= tau
        return x, monodromy, realized, boundaries

    # Seed the Newton from the DECLARATION: freeze each natural step at its
    # declared nominal duration and solve the resulting AFFINE fixed point
    # (exactly the step-1 machinery). At the extraction's own theta the
    # declared durations ARE the answer, and under a finite-difference
    # perturbation they are within O(h) of it — a cold zero seed, by
    # contrast, spends its first passes in the wrong regime and hands the
    # piecewise-smooth Newton a two-cycle to fall into (measured on
    # buck_dcm before this seed existed).
    seed_monodromy = identity.copy()
    seed_offset = np.zeros(n)
    for interval_duration, group in plan:
        remaining = interval_duration
        for position, index in enumerate(group):
            if position == len(group) - 1:
                tau = max(remaining, 0.0)
            else:
                tau = min(sequence.steps[index].nominal_duration_s, remaining)
            phi, psi, _, _ = _segment_operators(
                modes[index].a, modes[index].c, tau
            )
            seed_monodromy = phi @ seed_monodromy
            seed_offset = phi @ seed_offset + psi
            remaining -= tau
    invariants = _flow_invariants(circuit, modes)
    invariant_values = (
        invariants @ np.asarray(circuit.initial_state, dtype=float)
        if len(invariants)
        else np.zeros(0)
    )
    try:
        x0 = _constrained_solve(
            identity - seed_monodromy, seed_offset, invariants, invariant_values
        )
    except (np.linalg.LinAlgError, GradientEngineError):
        x0 = np.zeros(n)

    # Damped Newton on the fixed point of the hybrid cycle map. For an
    # all-scheduled sequence the map is affine, the seed already IS the fixed
    # point, and the loop exits on its first verification pass — the
    # all-scheduled behaviour is preserved, just reached through the general path. A
    # crossing that fails to fire on an INTERMEDIATE iterate is not a
    # verdict; the declaration is judged only at the converged point.
    converged = False
    iterations = 0
    residual = float("inf")
    for iterations in range(1, _MAX_FIXED_POINT_ITERATIONS + 1):
        x_end, monodromy, realized, boundaries = propagate(x0)
        gap = x_end - x0
        residual = float(np.max(np.abs(gap)))
        if residual <= _FIXED_POINT_TOLERANCE * (1.0 + float(np.max(np.abs(x0)))):
            converged = True
            break
        try:
            correction = _constrained_solve(
                identity - monodromy,
                gap,
                invariants,
                np.zeros(len(invariants)),
            )
        except np.linalg.LinAlgError as exc:
            raise GradientEngineError(
                "the period map has no isolated fixed point (I - M is "
                "singular); an undamped or marginal mode has no steady state "
                "to report"
            ) from exc
        # Backtracking guard against the regime two-cycle: accept the step
        # only if it shrinks the residual, halving up to six times before
        # falling back to a plain Picard step (the physical map contracts).
        accepted = False
        scale = 1.0
        for _ in range(6):
            candidate = x0 + scale * correction
            cand_end, _, _, _ = propagate(candidate)
            cand_residual = float(np.max(np.abs(cand_end - candidate)))
            if cand_residual < residual:
                x0 = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            x0 = x_end  # Picard: strictly follows the physical map
    if not converged:
        raise GradientEngineError(
            f"the cycle-map Newton did not converge in "
            f"{_MAX_FIXED_POINT_ITERATIONS} iterations "
            f"(residual {residual:.3e}); the declared sequence may sit on a "
            "regime boundary"
        )
    if missing_crossing:
        # Judged at the CONVERGED point: the physics genuinely never fires
        # the declared exit, so the declaration does not describe this theta.
        return infeasible("; ".join(missing_crossing), -1.0)

    steady = x0
    durations = tuple(realized)
    boundary_states = boundaries
    operators = [
        _segment_operators(mode.a, mode.c, duration)
        for mode, duration in zip(modes, durations)
    ]
    if coincidences:
        flags.append(
            AssumptionFlag(
                name="event_ordering_unambiguous",
                held=False,
                margin=-_COINCIDENCE_TOLERANCE_S,
                detail="; ".join(coincidences),
            )
        )

    metric_values: dict[str, float] = {}
    for spec in metrics:
        integral = 0.0
        for k, (mode, duration) in enumerate(zip(modes, durations)):
            row, const, _sources = mode.output_affine(spec.output)
            row = np.asarray(row, dtype=float)
            x0 = boundary_states[k]
            _, _, s_op, s2 = operators[k]
            if spec.statistic == "mean":
                state_integral = s_op @ x0 + s2
                integral += float(row @ state_integral) + const * duration
            else:  # rms: quadrature on the closed-form trajectory
                half = 0.5 * duration
                for node, weight in zip(_GAUSS_NODES, _GAUSS_WEIGHTS):
                    t = half * (node + 1.0)
                    phi_t, psi_t, _, _ = _segment_operators(mode.a, mode.c, t)
                    value = float(row @ (phi_t @ x0 + psi_t)) + const
                    integral += half * weight * value * value
        if spec.statistic == "mean":
            metric_values[spec.name] = integral / period
        else:
            metric_values[spec.name] = float(np.sqrt(integral / period))

    # Post-hoc assumption check on the STEADY trajectory: every diode of every
    # declared step must be consistent with its declared conduction state along
    # the whole segment, by the mode's own guard. This is what catches a DCM
    # model handed a CCM declaration: the freewheeling current dips below zero
    # and the conducting diode's guard goes negative.
    worst_margin = float("inf")
    worst_detail = "all declared conduction states hold along the steady cycle"
    for k, (mode, duration) in enumerate(zip(modes, durations)):
        x0 = boundary_states[k]
        half = 0.5 * duration
        for node in np.linspace(-1.0, 1.0, 17):
            t = half * (node + 1.0)
            phi_t, psi_t, _, _ = _segment_operators(mode.a, mode.c, t)
            state = phi_t @ x0 + psi_t
            for diode in circuit.diodes:
                conducting = bool(sequence.steps[k].diodes[diode.id])
                guard = float(mode.diode_guard(diode, state, conducting, 0.0))
                # The guard's healthy sign depends on the declared state: a
                # conducting diode must keep its branch CURRENT non-negative
                # (margin = +guard), a blocking one must keep its VOLTAGE
                # non-positive (margin = -guard). Folding both into "guard
                # must be positive" flags every healthily reverse-biased
                # diode — the first draft did exactly that.
                margin = guard if conducting else -guard
                if margin < worst_margin:
                    worst_margin = margin
                    worst_detail = (
                        f"step {k}, diode {diode.id!r} "
                        f"({'conducting' if conducting else 'blocking'}): "
                        f"margin {margin:.6g} at t={t:.3e} s into the step"
                    )
    flags.append(
        AssumptionFlag(
            name="declared_sequence_held",
            held=worst_margin >= -_GUARD_MARGIN_TOLERANCE,
            margin=worst_margin,
            detail=worst_detail,
        )
    )

    jacobian_labels: tuple[str, ...] = ()
    if loss_metrics:
        from pear_sim.analysis.gradient.losses import (
            NO_THERMAL_TERM_LABEL,
            evaluate_losses,
        )

        losses = evaluate_losses(
            model,
            circuit,
            modes,
            operators,
            boundary_states,
            durations,
            period,
            _GAUSS_NODES,
            _GAUSS_WEIGHTS,
            _segment_operators,
            smooth_tables=smooth_loss_tables,
        )
        metric_values["P_loss_total_w"] = losses.total_power_w
        for device_id, power in losses.per_device_power_w.items():
            metric_values[f"P_loss({device_id})_w"] = power
        for device_id, (edge_margin, clipped) in losses.table_margins.items():
            flags.append(
                AssumptionFlag(
                    name=f"loss_tables_interior[{device_id}]",
                    held=clipped == 0,
                    margin=edge_margin,
                    detail=(
                        f"{clipped} clamped lookup(s); worst signed distance "
                        f"to a table edge {edge_margin:.6g} A. Outside the "
                        "axes the tables are FLAT, so a gradient there is a "
                        "clamp artifact, not physics."
                    ),
                )
            )
        if losses.smoothed:
            delta = losses.total_power_w - float(losses.unsmoothed_total_power_w)
            flags.append(
                AssumptionFlag(
                    name="loss_tables_smoothed",
                    held=True,
                    margin=abs(delta),
                    detail=(
                        "monotone PCHIP smoothing is ACTIVE in the gradient "
                        "engine only (never in the oracle); "
                        f"it moves the total loss by {delta:+.6g} W against "
                        f"the unsmoothed evaluation "
                        f"({losses.unsmoothed_total_power_w:.6g} W)"
                    ),
                )
            )
        jacobian_labels = (NO_THERMAL_TERM_LABEL,)

    return GradientResult(
        parameter_names=vector.names(),
        parameter_values=tuple(float(v) for v in vector.as_array()),
        steady_state=SteadyStatePoint(
            converged=True,
            iterations=iterations,
            residual_inf=residual,
            period_s=period,
            state_names=tuple(c.id for c in circuit.state_components),
            state=tuple(float(v) for v in steady),
            segment_durations_s=durations,
            segment_boundary_states=tuple(
                tuple(float(v) for v in state) for state in boundary_states
            ),
        ),
        metrics=metric_values,
        jacobian=None,
        flags=tuple(flags),
        sequence_source_run_sha256=sequence.source_run_sha256,
        claim_boundary=_CLAIM,
        jacobian_labels=jacobian_labels,
    )


_CLAIM = (
    "Direction, not verdict: the engine evaluates the DECLARED sequence's "
    "affine period map exactly and differentiates it by finite differences on "
    "the compilation step. It never simulates, never adjudicates physics, and "
    "a violated declaration returns as a flag with a signed margin. The "
    "oracle (HybridSimulator) remains the only issuer of pass."
)
