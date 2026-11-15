# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Closed-loop periodic steady state and Floquet multipliers.

WHAT IS SOLVED. A converter under a sampled controller repeats one cycle in
steady state, but the cycle is a function of TWO kinds of state: the circuit
(inductor currents, capacitor voltages) and the controller's memory (every
``unit_delay`` register of the signal layer). Shooting on the circuit alone
restarts the controller from its declared memory each cycle and finds a fixed
point of the wrong map — which is why :mod:`pear_sim.analysis.steady_state`
refuses closed-loop models. Here the state is augmented,

    z = [circuit states ; unit_delay registers],

and the one-cycle map ``F(z)`` is ONE PERIOD OF THE PRODUCTION SIMULATOR
started from ``z``: circuit states through the components' initial values,
registers through each ``unit_delay``'s ``initial`` (which is what the first
tick of a run reads). After the cycle, the circuit state is the state at the
period end and each register is what its ``unit_delay`` will answer at the
NEXT tick, i.e. the value its input block took at tick 0. There is no second
model of the controller or the plant in this module.

WHAT IS REPORTED. The fixed point ``z* = F(z*)`` by Newton with a
forward-difference Jacobian, then the monodromy ``M = dF/dz`` at ``z*`` by
central differences, and its eigenvalues — the Floquet multipliers. All
``|mu| < 1``: a small perturbation of the orbit decays cycle by cycle (local
stability). One ``|mu| > 1``: it grows, at the frequency the multiplier's
angle names. Nothing is claimed about global stability, other orbits, or
robustness to parameter changes.

WHY THE VERDICT CAN BE "NOT RESOLVED". Finite differences on an adaptive
integrator carry noise of order ``rtol / step``. The monodromy is computed at
two step sizes and the spread of the multipliers is reported as their
uncertainty; a multiplier whose distance from the unit circle is inside that
spread gets no verdict. A number that cannot be told from 1 is not "stable".

WHEN THE DERIVATIVE DOES NOT EXIST. A clipping limiter, a clamped duty or a
change of the switching-mode sequence between the orbit and a perturbed run
makes ``F`` non-differentiable at ``z*``: the multipliers would describe a
kink, not the orbit. Each is checked on EVERY run that enters the derivative
and reported as an assumption with its margin; if any fails, no multipliers
are reported.

DECLARED SCOPE (v1). Control laws written in the signal layer and driving
``signal_pwm``; time-invariant smooth blocks only (constant, probe, gain,
sum, product, divide, saturate, unit_delay, model-level switch); DC sources;
no machines, saturable inductors, inductor-star invariants or forced gate
windows. Everything else is refused by name — a refused model is a stated
limit, a silently linearised kink is a wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from pear_sim.core.circuit import Circuit
from pear_sim.core.solver import HybridSimulator, SimulationResult
from pear_sim.errors import ModelError
from pear_sim.spec.model import Model
from pear_sim.spec.signals import declared_signal_target

__all__ = [
    "ClosedLoopAssumption",
    "ClosedLoopSteadyState",
    "ClosedLoopSteadyStateError",
    "FloquetMultiplier",
    "LyapunovCertificate",
    "lyapunov_certificate",
    "solve_closed_loop_steady_state",
]

DEFAULT_MAX_ITERATIONS = 30
DEFAULT_TOLERANCE = 1e-9
DEFAULT_RELATIVE_STEP = 1e-4
#: Budget for the analytic vs finite-difference multipliers. Measured
#: 2026-11-15 on the buck/PI gate cases: ~1e-11 to 1e-8 apart; a missing
#: saltation term moves them by >1e-2.
METHOD_AGREEMENT_TOLERANCE = 1e-6
#: A Lyapunov certificate must guarantee V(M z) <= (1 - margin) V(z): a
#: unit-free decrease far above round-off (~1e-16 on these matrices) and far
#: below any multiplier a stability verdict could rest on.
CERTIFICATE_MARGIN = 1e-8
# Floor of a probe step, in the state's own SI unit. A register that sits at
# zero (an integrator at a zero-error orbit) still needs a non-zero probe.
_STEP_FLOOR = 1e-7

#: Block kinds whose output is a time-invariant, smooth (or, for saturate,
#: smooth away from its limits) function of the inputs. `switch` branches on
#: a MODEL-level flag fixed before the first tick, so it is a constant choice.
SUPPORTED_BLOCK_KINDS = frozenset(
    {
        "constant",
        "probe",
        "gain",
        "sum",
        "product",
        "divide",
        "saturate",
        "unit_delay",
        "switch",
    }
)

_UNSUPPORTED_REASON = {
    "time_source": "time-varying (the cycle map is not autonomous)",
    "sine": "time-varying (the cycle map is not autonomous)",
    "noise": "time-varying (the cycle map is not autonomous)",
    "delay": "multi-sample memory is not part of the v1 augmented state",
    "compare": "a threshold makes the map non-smooth",
    "logic": "boolean logic makes the map non-smooth",
    "select": "a state-dependent branch makes the map non-smooth",
    "quantize": "quantisation makes the map non-smooth",
    "lookup_1d": "piecewise-linear breakpoints make the map non-smooth",
    "lookup_2d": "piecewise-linear breakpoints make the map non-smooth",
}


class ClosedLoopSteadyStateError(ModelError):
    """The model is outside the closed-loop steady-state v1 scope."""

    code = "analysis.closed_loop_steady_state_invalid"


@dataclass(frozen=True)
class ClosedLoopAssumption:
    """A differentiability assumption, checked on every derivative run.

    ``margin`` is the worst case over those runs in the guard's own units;
    ``held`` is the verdict on the assumption, never on the physics.
    """

    name: str
    held: bool
    margin: float
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "held": self.held,
            "margin": self.margin,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class FloquetMultiplier:
    real: float
    imag: float
    magnitude: float
    #: |angle| * f_cycle / (2 pi): the frequency a perturbation along this
    #: mode oscillates at, aliased into [0, f_cycle / 2].
    frequency_hz: float
    #: Spread of this multiplier between the two probe step sizes.
    uncertainty: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "real": self.real,
            "imag": self.imag,
            "magnitude": self.magnitude,
            "frequency_hz": self.frequency_hz,
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class LyapunovCertificate:
    """A quadratic Lyapunov function for the one-cycle map, checkable by hand.

    If ``P`` and ``P - M^T P M`` are both positive definite (each confirmed by
    a Cholesky factorisation, not by trusting whatever produced ``P``), then
    ``V(z) = z^T P z`` strictly decreases every cycle, so the orbit is locally
    asymptotically stable — without computing a single eigenvalue. Anyone can
    re-check it from ``floquet.json`` with two factorisations.

    ``contraction_bound`` is the largest ``V(M z) / V(z)``: an upper bound on
    ``|mu|^2`` that the multipliers must respect. ``certifies_other_monodromy``
    says the SAME ``P`` also passes for the second monodromy this solver
    computed (finite differences against the analytic one, or the other step
    size) — a unit-free statement that the certificate is not an artefact of
    one method's error. (A norm-ball radius was tried and dropped: the state
    mixes amperes, volts and volt-seconds, so any norm is a choice of units.)
    """

    found: bool
    detail: str
    #: How P was found; irrelevant to validity, recorded for reproduction.
    construction: str = ""
    p_matrix: tuple[tuple[float, ...], ...] = ()
    contraction_bound: float | None = None
    certifies_other_monodromy: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": "cholesky(P) and cholesky(P - M^T P M) both succeed",
            "found": self.found,
            "construction": self.construction,
            "detail": self.detail,
            "p_matrix": [list(row) for row in self.p_matrix],
            "contraction_bound": self.contraction_bound,
            "certifies_other_monodromy": self.certifies_other_monodromy,
        }


def _positive_definite(matrix: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError:
        return False
    return True


def _check_candidate(m: np.ndarray, p: np.ndarray) -> float | None:
    """Contraction bound if ``p`` certifies ``m``, else None.

    Only Cholesky factorisations decide — no eigenvalues of ``m`` — so the
    verdict is independent of how ``p`` was found.
    """

    from scipy.linalg import eigh

    p = 0.5 * (p + p.T)
    if not np.all(np.isfinite(p)) or not _positive_definite(p):
        return None
    image = m.T @ p @ m
    image = 0.5 * (image + image.T)
    if not _positive_definite(p - image):
        return None
    contraction = float(np.max(eigh(image, p, eigvals_only=True)))
    # Positive definiteness of P - M^T P M alone is decided by round-off when
    # the true decrease is zero: on a lossless rotation it came out
    # [-1.3e-17, +1.3e-17] on x86 (refused) and both positive on ARM
    # (certified a map on the unit circle, CI 2026-11-15). Demand a decrease
    # that round-off cannot fake.
    if contraction > 1.0 - CERTIFICATE_MARGIN:
        return None
    return contraction


def lyapunov_certificate(
    monodromy: np.ndarray, *, other: np.ndarray | None = None
) -> LyapunovCertificate:
    """Search and check a quadratic Lyapunov certificate for ``z -> M z``.

    Candidates: the Lyapunov equation ``M^T P M - P = -I``, and the eigenvector
    metric ``P = Re(V^-H V^-1)`` (contraction ``max |mu|^2``). A candidate that
    also certifies ``other`` is preferred, then the tighter contraction.
    Measured 2026-11-15 on the buck/PI orbit: Q = I gives a valid but nearly
    empty certificate (contraction 0.99999998, the state mixing units), the
    eigenvector metric gives 0.9794 = |mu|^2.
    """

    import warnings

    from scipy.linalg import LinAlgWarning, solve_discrete_lyapunov

    m = np.asarray(monodromy, dtype=float)
    size = m.shape[0]
    candidates: list[tuple[str, np.ndarray]] = []
    try:
        with warnings.catch_warnings():
            # On the unit circle the equation is singular by construction; the
            # Cholesky check below rejects whatever comes out, so the solver's
            # ill-conditioning warning carries no information here.
            warnings.simplefilter("ignore", LinAlgWarning)
            candidates.append(
                ("lyapunov_equation_q_identity", solve_discrete_lyapunov(m.T, np.eye(size)))
            )
    except (np.linalg.LinAlgError, ValueError):
        pass
    _values, vectors = np.linalg.eig(m)
    if np.linalg.cond(vectors) < 1e12:
        inverse = np.linalg.inv(vectors)
        candidates.append(("eigenvector_metric", np.real(inverse.conj().T @ inverse)))
    passing = []
    for name, p in candidates:
        contraction = _check_candidate(m, p)
        if contraction is None:
            continue
        both = None if other is None else _check_candidate(np.asarray(other), p) is not None
        passing.append((not bool(both), contraction, name, 0.5 * (p + p.T), both))
    if not passing:
        return LyapunovCertificate(
            found=False,
            detail=(
                "no candidate P passed: P or P - M^T P M is not positive definite "
                "(a multiplier on or outside the unit circle, or too close to it)"
            ),
        )
    _rank, contraction, name, p, both = min(passing, key=lambda item: item[:2])
    return LyapunovCertificate(
        found=True,
        construction=name,
        detail=(
            f"V(z) = z^T P z decreases every cycle by at least a factor {contraction:.6f}"
            + ("" if both is None else f"; same P certifies the other monodromy: {both}")
        ),
        p_matrix=tuple(tuple(float(value) for value in row) for row in p),
        contraction_bound=contraction,
        certifies_other_monodromy=both,
    )


@dataclass(frozen=True)
class ClosedLoopSteadyState:
    converged: bool
    iterations: int
    residual_inf: float
    period_s: float
    state_names: tuple[str, ...]
    state: tuple[float, ...]
    #: "stable" | "unstable" | "not_resolved" | "not_assessed"
    stability: str
    multipliers: tuple[FloquetMultiplier, ...] = ()
    monodromy: tuple[tuple[float, ...], ...] = ()
    assumptions: tuple[ClosedLoopAssumption, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)
    #: "analytic_saltation" | "central_finite_difference" | "" (none computed)
    monodromy_method: str = ""
    certificate: LyapunovCertificate | None = None

    @property
    def max_multiplier_magnitude(self) -> float | None:
        if not self.multipliers:
            return None
        return max(item.magnitude for item in self.multipliers)

    def as_dict(self) -> dict[str, Any]:
        return {
            "analysis": "closed_loop_periodic_steady_state",
            "method": "newton_shooting_on_augmented_state",
            "monodromy_method": self.monodromy_method,
            "claim": "local stability of the computed periodic orbit only",
            "converged": self.converged,
            "iterations": self.iterations,
            "residual_inf": self.residual_inf,
            "period_s": self.period_s,
            "state": dict(zip(self.state_names, self.state, strict=True)),
            "stability": self.stability,
            "max_multiplier_magnitude": self.max_multiplier_magnitude,
            "multipliers": [item.as_dict() for item in self.multipliers],
            "monodromy": [list(row) for row in self.monodromy],
            "assumptions": [item.as_dict() for item in self.assumptions],
            "notes": list(self.notes),
            "lyapunov_certificate": (
                None if self.certificate is None else self.certificate.as_dict()
            ),
        }


@dataclass(frozen=True)
class _Plan:
    """What the cycle map needs, resolved and scope-checked once."""

    model: Model
    period: float
    circuit_ids: tuple[str, ...]
    register_ids: tuple[str, ...]
    #: unit_delay id -> the block whose tick-0 value it answers next tick.
    register_sources: dict[str, str]
    saturate_ids: tuple[str, ...]
    #: signal_pwm control id -> (duty block id, duty_min, duty_max)
    duty_sources: dict[str, tuple[str, float, float]]


def _plan(model: Model) -> _Plan:
    graph = model.signals
    sampled = dict(model.sampled_controls)
    other = sorted(
        control_id
        for control_id, control in sampled.items()
        if getattr(control, "kind", "sampled_pi_pwm") != "signal_pwm"
    )
    if other:
        raise ClosedLoopSteadyStateError(
            f"sampled control(s) {other!r} are not signal_pwm: their controller "
            "memory lives inside the kernel and cannot be set from outside, so "
            "the cycle map cannot start from a given controller state. Express "
            "the law with signal blocks (see examples/buck_closed_loop_signals.yaml)"
        )
    if graph is None or not sampled:
        raise ClosedLoopSteadyStateError(
            "closed-loop steady state needs a control law in the signal layer "
            "driving a signal_pwm control; this model has none (an open-loop "
            "model belongs to analysis.steady_state)"
        )
    for block in graph.blocks:
        if block.kind not in SUPPORTED_BLOCK_KINDS:
            reason = _UNSUPPORTED_REASON.get(block.kind, "outside the v1 block set")
            raise ClosedLoopSteadyStateError(
                f"signal block {block.id!r} of type {block.kind!r} is not supported "
                f"by the closed-loop steady state: {reason}"
            )

    circuit = Circuit(model)
    if circuit.waveform_sources:
        raise ClosedLoopSteadyStateError(
            "closed-loop steady state supports DC sources only; a waveform source "
            "makes the cycle map time-varying"
        )
    if model.machines:
        raise ClosedLoopSteadyStateError(
            "machines are outside the closed-loop steady-state scope"
        )
    if circuit.sat_inductors:
        raise ClosedLoopSteadyStateError(
            "saturable inductors are outside the closed-loop steady-state scope "
            "(the map is not differentiable at the saturation knee)"
        )
    if model.forced_gate_windows:
        raise ClosedLoopSteadyStateError(
            "forced gate windows are a one-off perturbation, not a periodic orbit"
        )
    if not circuit.state_components:
        raise ClosedLoopSteadyStateError(
            "closed-loop steady state requires at least one circuit state"
        )
    _diodes, mode = HybridSimulator(model).initial_mode()
    if tuple(getattr(mode, "star_constraints", ())):
        raise ClosedLoopSteadyStateError(
            "inductor-star invariants are outside the closed-loop steady-state "
            "v1 scope (the open-loop shooting handles them; this solver does not "
            "restrict its probes to the invariant subspace)"
        )

    periods = {round(1.0 / control.frequency, 18) for control in sampled.values()}
    periods |= {round(1.0 / control.frequency, 18) for control in model.controls.values()}
    if len(periods) != 1:
        raise ClosedLoopSteadyStateError(
            "closed-loop steady state requires one shared switching frequency; "
            f"found periods {sorted(periods)!r} s"
        )
    period = next(iter(sampled.values())).period
    task = graph.task
    if task.offset != 0.0 or abs(task.period - period) > 1e-12 * period:
        raise ClosedLoopSteadyStateError(
            "closed-loop steady state requires the signal task to tick once per "
            f"switching period at offset 0; task period {task.period!r} s, offset "
            f"{task.offset!r} s, switching period {period!r} s"
        )

    registers = tuple(block for block in graph.blocks if block.kind == "unit_delay")
    duty_sources: dict[str, tuple[str, float, float]] = {}
    for control_id, control in sorted(sampled.items()):
        target = declared_signal_target(control.duty_signal)
        if target is None:
            raise ClosedLoopSteadyStateError(
                f"signal_pwm {control_id!r} duty {control.duty_signal!r} does not "
                "name a signal block"
            )
        duty_sources[control_id] = (target, control.duty_min, control.duty_max)
    return _Plan(
        model=model,
        period=period,
        circuit_ids=tuple(component.id for component in circuit.state_components),
        register_ids=tuple(block.id for block in registers),
        register_sources={block.id: block.inputs[0] for block in registers},
        saturate_ids=tuple(
            block.id for block in graph.blocks if block.kind == "saturate"
        ),
        duty_sources=duty_sources,
    )


def _initial_vector(plan: _Plan) -> np.ndarray:
    circuit = Circuit(plan.model)
    blocks = plan.model.signals.by_id  # type: ignore[union-attr]
    registers = [float(blocks[block_id].params["initial"]) for block_id in plan.register_ids]
    return np.concatenate(
        [np.asarray(circuit.initial_state, dtype=float), np.asarray(registers, dtype=float)]
    )


def _cycle_model(plan: _Plan, z: np.ndarray) -> Model:
    model = plan.model
    count = len(plan.circuit_ids)
    by_id = dict(zip(plan.circuit_ids, z[:count], strict=True))
    components = tuple(
        replace(component, initial_state=float(by_id[component.id]), initial_state_explicit=True)
        if component.id in by_id
        else component
        for component in model.components
    )
    registers = dict(zip(plan.register_ids, z[count:], strict=True))
    graph = model.signals
    blocks = tuple(
        replace(block, params={**block.params, "initial": float(registers[block.id])})
        if block.id in registers
        else block
        for block in graph.blocks  # type: ignore[union-attr]
    )
    transient = replace(
        model.transient,
        start=0.0,
        stop=plan.period,
        # Only the end state and the tick-0 signal values are read; the grid
        # is a recording choice that does not touch the trajectory.
        output_step=plan.period / 4.0,
    )
    return replace(
        model,
        components=components,
        signals=replace(graph, blocks=blocks),  # type: ignore[arg-type]
        transient=transient,
        harmonics=None,
    )


@dataclass(frozen=True)
class _CycleRun:
    end: np.ndarray
    mode_sequence: tuple[str, ...]
    #: Smallest distance of any saturate input / raw duty from its limits.
    limiter_margin: float
    limiter_detail: str
    #: The production run itself; the analytic monodromy reads its mode
    #: intervals, events and tick-0 signal values.
    result: SimulationResult | None = None


def _run_cycle(plan: _Plan, z: np.ndarray) -> _CycleRun:
    result: SimulationResult = HybridSimulator(_cycle_model(plan, z)).simulate()
    final_circuit = result.mode_intervals[-1]["state_end"]
    first_ticks = [entry for entry in result.signal_trace if entry["tick_index"] == 0]
    if len(first_ticks) != 1:
        raise ClosedLoopSteadyStateError(
            f"expected exactly one signal tick at the cycle start, found {len(first_ticks)}"
        )
    values = first_ticks[0]["values"]
    end = np.concatenate(
        [
            np.asarray([final_circuit[name] for name in plan.circuit_ids], dtype=float),
            np.asarray(
                [values[plan.register_sources[name]] for name in plan.register_ids],
                dtype=float,
            ),
        ]
    )
    graph_blocks = plan.model.signals.by_id  # type: ignore[union-attr]
    margin = float("inf")
    detail = "no limiter in the loop"
    for block_id in plan.saturate_ids:
        block = graph_blocks[block_id]
        value = values[block.inputs[0]]
        low, high = float(block.params["min"]), float(block.params["max"])
        distance = min(value - low, high - value)
        if distance < margin:
            margin, detail = distance, f"saturate {block_id!r} input {value!r} in [{low!r}, {high!r}]"
    for control_id, (block_id, low, high) in plan.duty_sources.items():
        value = values[block_id]
        distance = min(value - low, high - value)
        if distance < margin:
            margin, detail = (
                distance,
                f"signal_pwm {control_id!r} duty {value!r} in [{low!r}, {high!r}]",
            )
    sequence = tuple(str(interval["mode_signature"]) for interval in result.mode_intervals)
    return _CycleRun(
        end=end,
        mode_sequence=sequence,
        limiter_margin=margin,
        limiter_detail=detail,
        result=result,
    )


#: Events whose instant is either fixed in time (a scheduled gate of a fixed
#: PWM), set by the controller (a signal_pwm turn-off), or a commutation the
#: switch manager applies AT that same instant. A natural commutation has its
#: own state-dependent instant and needs a guard saltation — v1 leaves it to
#: the finite-difference path.
_ANALYTIC_EVENT_KINDS = frozenset({"scheduled_gate", "switch_manager"})


def _graph_jacobian(plan: _Plan, run: _CycleRun, circuit: Circuit) -> dict[str, np.ndarray]:
    """d(block value at tick 0) / dz for every block, by forward differentiation
    of the declared block equations — no simulation involved.

    Probes read the pre-edge state through the cycle-start mode's affine output
    map; a unit_delay answers with its register, which is a coordinate of z.
    """

    assert run.result is not None
    graph = plan.model.signals
    assert graph is not None
    blocks = graph.by_id
    values = next(
        entry["values"] for entry in run.result.signal_trace if entry["tick_index"] == 0
    )
    first = run.result.mode_intervals[0]
    mode = circuit.compile(
        first["gates"], first["diodes"], first["constrained_inductors"], ()
    )
    n = len(plan.circuit_ids)
    size = n + len(plan.register_ids)
    zero = np.zeros(n)
    rows: dict[str, np.ndarray] = {}
    for block_id in graph.order:
        block = blocks[block_id]
        kind = block.kind
        row = np.zeros(size)
        if kind == "probe":
            base = mode.output(block.params["source"], zero, 0.0)
            for index in range(n):
                unit = np.zeros(n)
                unit[index] = 1.0
                row[index] = mode.output(block.params["source"], unit, 0.0) - base
        elif kind == "unit_delay":
            row[n + plan.register_ids.index(block_id)] = 1.0
        elif kind == "gain":
            row = float(block.params["gain"]) * rows[block.inputs[0]]
        elif kind == "sum":
            for sign, source in zip(block.params["signs"], block.inputs, strict=True):
                row = row + float(sign) * rows[source]
        elif kind == "product":
            for index, source in enumerate(block.inputs):
                others = 1.0
                for other_index, other in enumerate(block.inputs):
                    if other_index != index:
                        others *= values[other]
                row = row + others * rows[source]
        elif kind == "divide":
            numerator, denominator = block.inputs
            top, bottom = values[numerator], values[denominator]
            row = (rows[numerator] * bottom - top * rows[denominator]) / (bottom * bottom)
        elif kind == "saturate":
            # Inside its limits (checked as an assumption) the block is identity.
            row = rows[block.inputs[0]]
        elif kind == "switch":
            row = rows[block.inputs[0] if block.params["flag"] else block.inputs[1]]
        elif kind != "constant":  # pragma: no cover - _plan refuses the rest
            raise ClosedLoopSteadyStateError(f"no derivative rule for block kind {kind!r}")
        rows[block_id] = row
    return rows


def _saltation_term(
    after_edge: np.ndarray, jump: np.ndarray, period: float, duty_row: np.ndarray
) -> np.ndarray:
    """dx(T)/dz through the controlled edge: the field jump at the edge,
    carried to the period end, times dt_s/dz = T dd/dz. Without it the
    controller does not act on the plant at all in the linearisation."""

    return np.outer(after_edge @ jump, period * duty_row)


def _analytic_monodromy(plan: _Plan, run: _CycleRun) -> tuple[np.ndarray | None, str]:
    """Monodromy from the orbit's own compiled modes, without probing.

    x(T) = Phi_N ... Phi_1 x0 + (forced terms), Phi_k = expm(a_k tau_k) of the
    production mode matrices along the orbit. A controlled turn-off at
    t_s = d T moves with z: dx(T)/dt_s = Phi_after (f_before - f_after) at
    x(t_s), and dt_s/dz = T dd/dz from the block equations. Registers map to
    the derivative of their input block. Returns (None, reason) when the orbit
    has an event this rule does not cover.
    """

    from scipy.linalg import expm

    result = run.result
    assert result is not None
    kinds = sorted({event.kind for event in result.events} - _ANALYTIC_EVENT_KINDS)
    if kinds:
        return None, f"orbit has {kinds!r} events (state-dependent guard instants)"
    circuit = Circuit(result.model)
    rows = _graph_jacobian(plan, run, circuit)
    n = len(plan.circuit_ids)
    size = n + len(plan.register_ids)
    intervals = result.mode_intervals
    modes = [
        circuit.compile(
            interval["gates"], interval["diodes"], interval["constrained_inductors"], ()
        )
        for interval in intervals
    ]
    for mode in modes:
        if mode.constrained_state_indices:
            return None, "an interval reduces a state (zero-current or inductor-star path)"
    transitions = [
        expm(mode.a * (interval["end_s"] - interval["start_s"]))
        for mode, interval in zip(modes, intervals, strict=True)
    ]

    def after(index: int) -> np.ndarray:
        """Transition from the end of interval `index` to the period end."""

        chain = np.eye(n)
        for transition in transitions[index + 1 :]:
            chain = transition @ chain
        return chain

    total = after(-1)  # product of every interval
    monodromy = np.zeros((size, size))
    monodromy[:n, :n] = total
    tolerance = 1e-9 * plan.period
    for control_id, (block_id, _low, _high) in plan.duty_sources.items():
        duty_row = rows[block_id]
        off_time = float(
            next(
                entry["values"][block_id]
                for entry in result.signal_trace
                if entry["tick_index"] == 0
            )
        ) * plan.period
        boundary = next(
            (
                index
                for index in range(len(intervals) - 1)
                if abs(intervals[index]["end_s"] - off_time) <= tolerance
                and intervals[index]["gates"].get(control_id)
                and not intervals[index + 1]["gates"].get(control_id)
            ),
            None,
        )
        if boundary is None:
            return None, f"no turn-off of {control_id!r} at the commanded instant"
        state_at_edge = np.asarray(
            [intervals[boundary]["state_end"][name] for name in plan.circuit_ids]
        )
        before, later = modes[boundary], modes[boundary + 1]
        jump = (before.a @ state_at_edge + before.c) - (later.a @ state_at_edge + later.c)
        monodromy[:n, :] += _saltation_term(after(boundary), jump, plan.period, duty_row)
    for offset, register_id in enumerate(plan.register_ids):
        monodromy[n + offset, :] = rows[plan.register_sources[register_id]]
    return monodromy, "analytic"


def _steps(z: np.ndarray, relative: float) -> np.ndarray:
    return relative * np.maximum(np.abs(z), _STEP_FLOOR)


def _central_monodromy(
    plan: _Plan, z: np.ndarray, relative: float
) -> tuple[np.ndarray, list[_CycleRun]]:
    steps = _steps(z, relative)
    columns = []
    runs: list[_CycleRun] = []
    for index, step in enumerate(steps):
        offset = np.zeros_like(z)
        offset[index] = step
        plus = _run_cycle(plan, z + offset)
        minus = _run_cycle(plan, z - offset)
        runs.extend((plus, minus))
        columns.append((plus.end - minus.end) / (2.0 * step))
    return np.column_stack(columns), runs


def _match(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Pair each reference eigenvalue with its nearest unused counterpart."""

    remaining = list(other)
    paired = []
    for value in reference:
        index = int(np.argmin([abs(value - candidate) for candidate in remaining]))
        paired.append(remaining.pop(index))
    return np.asarray(paired)


def solve_closed_loop_steady_state(
    model: Model,
    *,
    initial_state: dict[str, float] | None = None,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    relative_step: float = DEFAULT_RELATIVE_STEP,
) -> ClosedLoopSteadyState:
    """Closed-loop periodic orbit and its Floquet multipliers.

    Newton is local: it starts from the model's declared initial circuit
    state and ``unit_delay`` initial values, or from ``initial_state``
    (keys are circuit state ids and ``signal:<unit_delay id>``) — for
    example the last cycle of a settled transient.
    """

    plan = _plan(model)
    names = plan.circuit_ids + tuple(f"signal:{name}" for name in plan.register_ids)
    z = _initial_vector(plan)
    if initial_state is not None:
        unknown = sorted(set(initial_state) - set(names))
        if unknown:
            raise ClosedLoopSteadyStateError(
                f"initial_state names unknown state(s) {unknown!r}; expected a subset "
                f"of {list(names)!r}"
            )
        for position, name in enumerate(names):
            if name in initial_state:
                z[position] = float(initial_state[name])

    identity = np.eye(len(z))
    converged = False
    residual_inf = float("inf")
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        base = _run_cycle(plan, z)
        residual = base.end - z
        scale = 1.0 + np.maximum(np.abs(z), np.abs(base.end))
        residual_inf = float(np.max(np.abs(residual) / scale))
        if residual_inf <= tolerance:
            converged = True
            break
        if base.limiter_margin <= 0.0:
            # A clamped duty makes the map flat in the controller directions:
            # the Newton matrix is singular for a reason that has nothing to
            # do with the orbit. Say which limiter, not "a multiplier at 1".
            return ClosedLoopSteadyState(
                converged=False,
                iterations=iterations,
                residual_inf=residual_inf,
                period_s=plan.period,
                state_names=names,
                state=tuple(float(value) for value in z),
                stability="not_assessed",
                notes=(
                    "a Newton iterate drives a limiter into its bound ("
                    + base.limiter_detail
                    + "); start closer to the orbit with initial_state, e.g. the "
                    "last cycle of a settled transient",
                ),
            )
        steps = _steps(z, relative_step)
        jacobian = np.column_stack(
            [
                (_run_cycle(plan, z + step * identity[:, index]).end - base.end) / step
                for index, step in enumerate(steps)
            ]
        )
        try:
            update = np.linalg.solve(jacobian - identity, -residual)
        except np.linalg.LinAlgError as error:
            raise ClosedLoopSteadyStateError(
                "the shooting system is singular (a multiplier sits exactly at 1); "
                "the orbit is not isolated"
            ) from error
        # Backtracking: a full Newton step from far away can drive the duty
        # into its clamp, where the map is flat and the step is meaningless.
        damping = 1.0
        current = float(np.max(np.abs(residual) / scale))
        while damping > 1.0 / 64.0:
            trial = z + damping * update
            trial_run = _run_cycle(plan, trial)
            trial_scale = 1.0 + np.maximum(np.abs(trial), np.abs(trial_run.end))
            if float(np.max(np.abs(trial_run.end - trial) / trial_scale)) < current:
                break
            damping *= 0.5
        z = z + damping * update

    if not converged:
        return ClosedLoopSteadyState(
            converged=False,
            iterations=iterations,
            residual_inf=residual_inf,
            period_s=plan.period,
            state_names=names,
            state=tuple(float(value) for value in z),
            stability="not_assessed",
            notes=("shooting did not converge; no multipliers are reported",),
        )

    orbit = _run_cycle(plan, z)
    coarse, coarse_runs = _central_monodromy(plan, z, relative_step)
    fine, fine_runs = _central_monodromy(plan, z, 0.5 * relative_step)
    runs = [orbit, *coarse_runs, *fine_runs]

    worst_limiter = min(runs, key=lambda run: run.limiter_margin)
    limiter = ClosedLoopAssumption(
        name="no_limiter_engaged",
        held=worst_limiter.limiter_margin > 0.0,
        margin=worst_limiter.limiter_margin,
        detail="worst over the orbit and every probe run: "
        + worst_limiter.limiter_detail,
    )
    changed = sum(run.mode_sequence != orbit.mode_sequence for run in runs)
    sequence = ClosedLoopAssumption(
        name="mode_sequence_unchanged",
        held=changed == 0,
        margin=float(-changed),
        detail=(
            f"{changed} of {len(runs) - 1} probe runs switched through a different "
            f"mode sequence than the orbit ({len(orbit.mode_sequence)} intervals)"
        ),
    )
    assumptions = (limiter, sequence)
    if not all(item.held for item in assumptions):
        return ClosedLoopSteadyState(
            converged=True,
            iterations=iterations,
            residual_inf=residual_inf,
            period_s=plan.period,
            state_names=names,
            state=tuple(float(value) for value in z),
            stability="not_assessed",
            assumptions=assumptions,
            notes=(
                "the cycle map is not differentiable at this orbit; no multipliers "
                "are reported",
            ),
        )

    fd_values = np.linalg.eigvals(coarse)
    spread = np.abs(fd_values - _match(fd_values, np.linalg.eigvals(fine)))
    analytic, reason = _analytic_monodromy(plan, orbit)
    notes: list[str] = []
    if analytic is None:
        method, primary, values, uncertainty = (
            "central_finite_difference",
            coarse,
            fd_values,
            spread,
        )
        notes.append(f"analytic monodromy not applicable ({reason}); finite differences only")
    else:
        method, primary = "analytic_saltation", analytic
        values = np.linalg.eigvals(analytic)
        fd_matched = _match(values, fd_values)
        disagreement = np.abs(values - fd_matched)
        agree = ClosedLoopAssumption(
            name="analytic_and_finite_difference_agree",
            held=bool(np.max(disagreement) <= METHOD_AGREEMENT_TOLERANCE),
            margin=float(METHOD_AGREEMENT_TOLERANCE - np.max(disagreement)),
            detail=(
                f"max multiplier difference {float(np.max(disagreement)):.3e} between "
                "the analytic monodromy and central finite differences of the "
                f"production cycle map (budget {METHOD_AGREEMENT_TOLERANCE:.0e})"
            ),
        )
        assumptions = (*assumptions, agree)
        if not agree.held:
            # Two methods disagreeing is a defect to diagnose, not noise to
            # average: no verdict rather than a choice between them.
            return ClosedLoopSteadyState(
                converged=True,
                iterations=iterations,
                residual_inf=residual_inf,
                period_s=plan.period,
                state_names=names,
                state=tuple(float(value) for value in z),
                stability="not_assessed",
                monodromy_method=method,
                assumptions=assumptions,
                notes=("the analytic and finite-difference monodromies disagree",),
            )
        uncertainty = np.maximum(
            disagreement, np.abs(_match(values, np.linalg.eigvals(fine)) - fd_matched)
        )
    order = np.argsort(-np.abs(values))
    frequency = 1.0 / plan.period
    multipliers = tuple(
        FloquetMultiplier(
            real=float(values[index].real),
            imag=float(values[index].imag),
            magnitude=float(abs(values[index])),
            frequency_hz=float(abs(np.angle(values[index])) * frequency / (2.0 * np.pi)),
            uncertainty=float(uncertainty[index]),
        )
        for index in order
    )
    if all(item.magnitude + item.uncertainty < 1.0 for item in multipliers):
        stability = "stable"
    elif any(item.magnitude - item.uncertainty > 1.0 for item in multipliers):
        stability = "unstable"
    else:
        stability = "not_resolved"
    other = fine if analytic is None else coarse
    certificate = lyapunov_certificate(primary, other=other)
    return ClosedLoopSteadyState(
        converged=True,
        iterations=iterations,
        residual_inf=residual_inf,
        period_s=plan.period,
        state_names=names,
        state=tuple(float(value) for value in z),
        stability=stability,
        multipliers=multipliers,
        monodromy=tuple(tuple(float(value) for value in row) for row in primary),
        monodromy_method=method,
        assumptions=assumptions,
        notes=tuple(notes),
        certificate=certificate,
    )
