"""Fair comparison of topology finalists.

This is the place where it is easiest to fool yourself, so the rules are
structural rather than advisory:

**Each finalist is pulled to its OWN optimum.** Evaluating every candidate
at one shared operating point silently crowns whichever topology that point
happened to suit. So every finalist gets its own optimisation run, its own
trajectory, and its own landing point.

**The oracle is the only issuer of a verdict.** The gradient engine moves
the search (it is ~20x cheaper — 3 ms against 78 ms on the buck entry), but
what it produces is a DIRECTION. The reported numbers come from a full
transient run at the landing point, and the gradient's own numbers are kept
beside them rather than in place of them.

**The trust region is the mode sequence, and leaving it is a measurement.**
The Jacobian describes the circuit only while the declared sequence holds;
once a mode appears or vanishes (CCM->DCM, lost ZVS, a different commutation
order) the Jacobian describes a circuit that no longer exists and nothing in
the numbers gives that away. So the landing point is re-extracted from the
oracle run and compared with :func:`sequences_agree`. Disagreement is a
recorded, measured mode boundary — never a crash, and never silence.

**A candidate the kernel cannot run is refused, not skipped.** Entries may
declare `kernel_support: not_simulated`. Ranking them would be
fabrication; dropping them would let the shortlist be quietly decided by
what the engine happens to support. They come back with status
``unsupported``, which is a statement about the ENGINE, not a defect of the
candidate.

**Dominated points stay in the record.** A front that shows only its winners
cannot be audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pear_sim.analysis.gradient import (
    DesignVector,
    GradientResult,
    MetricSpec,
    extract_mode_sequence,
    sequences_agree,
    solve,
)
from pear_sim.analysis.steady_state import one_cycle_model, solve_periodic_steady_state
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import PearSimError
from pear_sim.run.contracts import sha256_json
from pear_sim.topology.library import TopologyEntry

TOPOLOGY_COMPARISON_SCHEMA = "pear-sim/topology-comparison/v0.1"

# The evaluator's status set. Five, not four: `unsupported` must never be
# collapsed into `invalid_model`, or the library's decision to list
# not-yet-simulatable topologies honestly would be undone one level down.
STATUSES = ("ok", "infeasible", "not_converged", "invalid_model", "unsupported")

_SENSES = ("minimize", "maximize")


class ComparisonError(PearSimError):
    """The comparison request itself is malformed."""

    code = "topology.comparison_invalid"


@dataclass(frozen=True)
class Objective:
    """One axis of the comparison.

    ``metric`` is a gradient-engine metric name (``mean(v(out))``); ``sense``
    says which direction is better. Objectives are never summed into a single
    score — efficiency, ripple and temperature do not add up to anything an
    engineer would defend, which is why the result is a front and not a
    ranking.
    """

    metric: str
    sense: str

    def __post_init__(self) -> None:
        if self.sense not in _SENSES:
            raise ComparisonError(
                f"objective sense {self.sense!r} must be one of {_SENSES}"
            )

    @property
    def spec(self) -> MetricSpec:
        """Split ``rms(i(l1))`` into its statistic and its output.

        The output itself contains parentheses, so exactly ONE closing paren
        is removed — ``rstrip(')')`` would eat both and silently ask the
        kernel for a signal named ``i(l1``.
        """

        statistic, separator, rest = self.metric.partition("(")
        if not separator or not rest.endswith(")"):
            raise ComparisonError(
                f"objective metric {self.metric!r} must read "
                "'<statistic>(<output>)', e.g. 'mean(v(out))'"
            )
        return MetricSpec(rest[:-1], statistic)

    def better(self, left: float, right: float) -> bool:
        return left < right if self.sense == "minimize" else left > right

    def as_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "sense": self.sense}


@dataclass(frozen=True)
class OptimizationStep:
    """One accepted step, retained so the path can be audited."""

    iteration: int
    theta: dict[str, float]
    objective_value: float
    gradient_inf_norm: float
    step_scale: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "theta": dict(self.theta),
            "objective_value": self.objective_value,
            "gradient_inf_norm": self.gradient_inf_norm,
            "step_scale": self.step_scale,
        }


@dataclass(frozen=True)
class TrustRegionVerdict:
    """Did the landing point stay inside the sequence the Jacobian assumed?

    The region is the mode STRUCTURE — which modes, in what order, each
    ending on what kind of event. It is deliberately NOT the step durations:
    moving those is precisely what optimising ``duty`` or ``fsw`` does, so
    checking them here would flag every successful run as having escaped and
    the gate would cry wolf until nobody read it. Durations are reported
    beside the verdict as information.
    """

    held: bool
    reasons: tuple[str, ...]
    checked: bool
    note: str
    duration_drift: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "held": self.held,
            "checked": self.checked,
            "reasons": list(self.reasons),
            "note": self.note,
            "duration_drift": list(self.duration_drift),
        }


@dataclass(frozen=True)
class TierHealth:
    """Gradient-PREDICTED change vs oracle-MEASURED change, start -> landing.

    Sharper than a rank correlation and free from the first day: it carries
    magnitude and sign. Both deltas span the same two points, so what is
    compared is how far each tier says the optimisation actually moved the
    metric. The pair drifting apart is the earliest signal that the trust
    region is too large.
    """

    metric: str
    predicted_delta: float
    measured_delta: float
    metric_scale: float
    """|metric| at the landing point — the yardstick for "did it move at all"."""

    @property
    def absolute_error(self) -> float:
        return abs(self.predicted_delta - self.measured_delta)

    @property
    def relative_error(self) -> float | None:
        """Error against the measured move, or ``None`` if nothing moved.

        A metric the optimisation left untouched carries no information about
        tier agreement, and dividing two floating-point crumbs would report a
        confident "100 % error" about noise. Below a millionth of the
        metric's own size the move is not a move.
        """

        scale = abs(self.measured_delta)
        if scale <= 1e-6 * max(abs(self.metric_scale), np.finfo(float).tiny):
            return None
        return self.absolute_error / scale

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "predicted_delta": self.predicted_delta,
            "measured_delta": self.measured_delta,
            "metric_scale": self.metric_scale,
            "absolute_error": self.absolute_error,
            "relative_error": self.relative_error,
        }


@dataclass(frozen=True)
class FinalistResult:
    """One finalist, optimised to its own optimum and judged by the oracle."""

    topology_id: str
    status: str
    reason: str
    theta_names: tuple[str, ...] = ()
    theta_start: dict[str, float] = field(default_factory=dict)
    theta_final: dict[str, float] = field(default_factory=dict)
    trajectory: tuple[OptimizationStep, ...] = ()
    gradient_metrics: dict[str, float] = field(default_factory=dict)
    oracle_metrics: dict[str, float] = field(default_factory=dict)
    trust_region: TrustRegionVerdict | None = None
    tier_health: tuple[TierHealth, ...] = ()
    flags: tuple[str, ...] = ()
    jacobian_labels: tuple[str, ...] = ()
    loss_instrumented: bool = False
    """Does this finalist's model declare device cards and a loss analysis?

    An uninstrumented candidate is not merely less documented — it looks
    LOSSLESS, so on any efficiency-flavoured axis it beats an instrumented
    rival for free. The comparison refuses to let that happen quietly.
    """

    @property
    def rankable(self) -> bool:
        """Only an `ok` finalist whose landing point stayed in its trust
        region may be placed on a front against the others."""

        return self.status == "ok" and bool(
            self.trust_region and self.trust_region.held
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "topology_id": self.topology_id,
            "status": self.status,
            "reason": self.reason,
            "rankable": self.rankable,
            "theta_names": list(self.theta_names),
            "theta_start": dict(self.theta_start),
            "theta_final": dict(self.theta_final),
            "trajectory": [step.as_dict() for step in self.trajectory],
            # Both tiers are reported. The oracle numbers are the verdict; the
            # gradient numbers are kept so a disagreement is visible instead
            # of being quietly overwritten.
            "gradient_metrics": dict(self.gradient_metrics),
            "oracle_metrics": dict(self.oracle_metrics),
            "trust_region": None if self.trust_region is None else self.trust_region.as_dict(),
            "tier_health": [item.as_dict() for item in self.tier_health],
            "flags": list(self.flags),
            "jacobian_labels": list(self.jacobian_labels),
            "loss_instrumented": self.loss_instrumented,
        }


def _oracle_metrics(vector: DesignVector, specs: tuple[MetricSpec, ...]) -> dict[str, float]:
    """Metrics from a FULL transient run at periodic convergence.

    The model is driven to its periodic steady state first and then run for
    exactly one settled cycle; measuring on a shared truncated window would
    report the transient instead of the operating point.
    """

    model = vector.load()
    steady = solve_periodic_steady_state(model)
    if not steady.converged:
        raise ComparisonError("the oracle did not reach a periodic steady state")
    cycle = one_cycle_model(model, np.asarray(steady.state))
    run = HybridSimulator(cycle).simulate()
    computed = run.metrics()
    return {
        spec.name: float(computed[spec.output][spec.statistic]) for spec in specs
    }


def _objective_value(metrics: dict[str, float], objective: Objective) -> float:
    """Scalar the optimiser descends: always a MINIMISED quantity."""

    value = metrics[objective.metric]
    return value if objective.sense == "minimize" else -value


def _check_trust_region(
    vector: DesignVector, entry: TopologyEntry, *, duration_rtol: float
) -> TrustRegionVerdict:
    """Re-extract the cycle at the landing point and compare with the entry.

    An entry with no declared sequence cannot be checked; that is reported as
    `checked: False` rather than passed, because an unchecked trust region is
    not a satisfied one.
    """

    if not entry.mode_sequences:
        return TrustRegionVerdict(
            held=False,
            reasons=(),
            checked=False,
            note=(
                "the entry declares no mode sequence, so the region the "
                "Jacobian assumed cannot be checked — unchecked is not "
                "the same as satisfied"
            ),
        )
    declared = entry.mode_sequences[0].sequence
    model = vector.load()
    steady = solve_periodic_steady_state(model)
    cycle = one_cycle_model(model, np.asarray(steady.state))
    run = HybridSimulator(cycle).simulate()
    observed = extract_mode_sequence(run, period_s=declared.period_s)

    # Structure decides the verdict: same modes, same order, same kind of
    # exit event. `sequences_agree` additionally compares durations, which is
    # right for `topology verify` (same theta, so any drift is a real
    # regression) and wrong here (theta moved on purpose).
    reasons: list[str] = []
    if declared.signatures() != observed.signatures():
        reasons.append(
            f"mode structure changed: declared {list(declared.signatures())} "
            f"vs observed {list(observed.signatures())}"
        )
    else:
        for position, (expected, actual) in enumerate(
            zip(declared.steps, observed.steps, strict=True)
        ):
            if expected.exit_condition.kind is not actual.exit_condition.kind:
                reasons.append(
                    f"step {position} now ends on "
                    f"{actual.exit_condition.kind.value} instead of "
                    f"{expected.exit_condition.kind.value}"
                )
    # Duration movement is the optimiser doing its job; recorded, not judged.
    _, drift = sequences_agree(declared, observed, duration_rtol=duration_rtol)
    duration_drift = tuple(reason for reason in drift if "duration differs" in reason)

    held = not reasons
    return TrustRegionVerdict(
        held=held,
        reasons=tuple(reasons),
        checked=True,
        note=(
            "the landing point still runs the declared mode structure; step "
            "durations moved, which is what optimising theta does"
            if held
            else (
                "the optimum LEFT the mode structure the Jacobian was "
                "computed on — a mode appeared, vanished or now ends on a "
                "different event. The gradient described a circuit that no "
                "longer exists here, so this is a measured mode boundary, "
                "not an optimum"
            )
        ),
        duration_drift=duration_drift,
    )


def optimize_finalist(
    entry: TopologyEntry,
    *,
    objective: Objective,
    theta_names: tuple[str, ...],
    bounds: dict[str, tuple[float, float]],
    root: str,
    extra_metrics: tuple[Objective, ...] = (),
    max_iterations: int = 12,
    initial_step_scale: float = 0.1,
    minimum_step_scale: float = 1e-3,
    duration_rtol: float = 1e-6,
) -> FinalistResult:
    """Pull ONE finalist to its own optimum, then let the oracle judge it.

    The inner loop is projected gradient descent with a backtracking step, run
    on the gradient engine; the outer act is a single oracle evaluation at the
    landing point, which produces the reported metrics, the trust-region
    verdict and the tier-health pair.

    Never raises for a candidate's sake: a refusal by the kernel, a
    non-converged solve or a violated declaration comes back as a status.
    """

    if entry.kernel_support.status != "simulated":
        return FinalistResult(
            topology_id=entry.id,
            status="unsupported",
            reason=(
                f"this kernel cannot run the entry: {entry.kernel_support.note} "
                "— ranking it would be fabrication and dropping it would let "
                "the shortlist be decided by engine coverage"
            ),
        )
    if not theta_names:
        raise ComparisonError("theta_names must name at least one parameter")
    for name in theta_names:
        if name not in bounds:
            raise ComparisonError(
                f"parameter {name!r} is optimised but has no bounds; an "
                "unbounded search would wander outside the model's validity"
            )

    model_path = f"{root}/{entry.template.model}"
    specs = tuple(
        dict.fromkeys((objective.spec, *(item.spec for item in extra_metrics)))
    )
    all_objectives = (objective, *extra_metrics)

    try:
        vector = DesignVector.from_model(model_path, theta_names)
        start = dict(zip(vector.names(), vector.as_array().tolist(), strict=True))
        result: GradientResult = solve(
            vector, entry.mode_sequences[0].sequence, specs, jacobian=True
        )
    except PearSimError as exc:
        return FinalistResult(
            topology_id=entry.id,
            status="invalid_model",
            reason=f"{type(exc).__name__}: {exc}",
        )

    lower = np.array([bounds[name][0] for name in theta_names], dtype=float)
    upper = np.array([bounds[name][1] for name in theta_names], dtype=float)
    theta = np.clip(vector.as_array(), lower, upper)
    vector = vector.with_values(theta)
    result = solve(vector, entry.mode_sequences[0].sequence, specs, jacobian=True)

    # The starting point is measured on BOTH tiers before the search moves,
    # so tier health can compare like with like: how far each tier says the
    # optimisation actually moved the metric, over the same two points.
    gradient_start = dict(result.metrics)
    try:
        oracle_start = _oracle_metrics(vector, specs)
    except (PearSimError, ComparisonError) as exc:
        return FinalistResult(
            topology_id=entry.id,
            status="not_converged",
            reason=f"the oracle could not judge the starting point: {exc}",
            theta_names=theta_names,
            theta_start=start,
            theta_final=start,
            gradient_metrics=gradient_start,
        )

    current = _objective_value(result.metrics, objective)
    trajectory: list[OptimizationStep] = []
    step_scale = initial_step_scale
    flags: set[str] = set()
    span = np.maximum(upper - lower, np.finfo(float).tiny)

    for iteration in range(max_iterations):
        assert result.jacobian is not None
        row = result.jacobian.metric_names.index(objective.metric)
        gradient = np.array(result.jacobian.values[row], dtype=float)
        if objective.sense == "maximize":
            gradient = -gradient
        # Scale-free descent: theta entries differ by decades (Hz vs H), so a
        # raw gradient step would move one axis by nothing and another off the
        # map. The step is a fraction of each axis's own bounded span.
        direction = -gradient * span
        norm = float(np.max(np.abs(direction))) if direction.size else 0.0
        if not np.isfinite(norm) or norm == 0.0:
            break
        direction = direction / norm

        improved = False
        while step_scale >= minimum_step_scale:
            candidate_theta = np.clip(theta + step_scale * span * direction, lower, upper)
            if np.allclose(candidate_theta, theta):
                break
            candidate_vector = vector.with_values(candidate_theta)
            try:
                candidate_result = solve(
                    candidate_vector,
                    entry.mode_sequences[0].sequence,
                    specs,
                    jacobian=True,
                )
            except PearSimError:
                # A step that leaves the kernel's supported region is not an
                # error: it is the edge of where this candidate can be taken.
                step_scale /= 2.0
                flags.add("step_refused_by_kernel")
                continue
            value = _objective_value(candidate_result.metrics, objective)
            if value < current:
                theta, vector, result, current = (
                    candidate_theta,
                    candidate_vector,
                    candidate_result,
                    value,
                )
                improved = True
                break
            step_scale /= 2.0
        if not improved:
            break
        trajectory.append(
            OptimizationStep(
                iteration=iteration,
                theta=dict(zip(theta_names, theta.tolist(), strict=True)),
                objective_value=current,
                gradient_inf_norm=float(np.max(np.abs(gradient))),
                step_scale=step_scale,
            )
        )

    for flag in result.flags:
        if not flag.held:
            flags.add(flag.name)
    labels = tuple(result.jacobian_labels)
    if labels:
        # An open thermal loop or a clamped loss table makes every
        # number here optimistic in a named way; the comparison must carry
        # that forward rather than presenting a clean figure.
        flags.add("OPTIMISTIC")

    try:
        oracle = _oracle_metrics(vector, specs)
    except (PearSimError, ComparisonError) as exc:
        return FinalistResult(
            topology_id=entry.id,
            status="not_converged",
            reason=f"the oracle could not judge the landing point: {exc}",
            theta_names=theta_names,
            theta_start=start,
            theta_final=dict(zip(theta_names, theta.tolist(), strict=True)),
            trajectory=tuple(trajectory),
            gradient_metrics=dict(result.metrics),
            flags=tuple(sorted(flags)),
            jacobian_labels=labels,
        )

    instrumented = vector.load().loss_analysis is not None
    trust = _check_trust_region(vector, entry, duration_rtol=duration_rtol)
    health = tuple(
        TierHealth(
            metric=item.metric,
            predicted_delta=result.metrics[item.metric] - gradient_start[item.metric],
            measured_delta=oracle[item.metric] - oracle_start[item.metric],
            metric_scale=abs(oracle[item.metric]),
        )
        for item in all_objectives
    )

    return FinalistResult(
        topology_id=entry.id,
        status="ok",
        reason="optimised to its own optimum; metrics judged by the oracle",
        theta_names=theta_names,
        theta_start=start,
        theta_final=dict(zip(theta_names, theta.tolist(), strict=True)),
        trajectory=tuple(trajectory),
        gradient_metrics=dict(result.metrics),
        oracle_metrics=oracle,
        trust_region=trust,
        tier_health=health,
        flags=tuple(sorted(flags)),
        jacobian_labels=labels,
        loss_instrumented=instrumented,
    )


def _dominates(
    left: dict[str, float], right: dict[str, float], objectives: tuple[Objective, ...]
) -> bool:
    """Pareto dominance: at least as good everywhere, strictly better once."""

    at_least_as_good = all(
        not objective.better(right[objective.metric], left[objective.metric])
        for objective in objectives
    )
    strictly_better = any(
        objective.better(left[objective.metric], right[objective.metric])
        for objective in objectives
    )
    return at_least_as_good and strictly_better


def compare_finalists(
    results: tuple[FinalistResult, ...],
    objectives: tuple[Objective, ...],
) -> dict[str, Any]:
    """Place comparable finalists on a Pareto front — dominated points kept.

    Only `ok` finalists whose landing point stayed inside their trust region
    are ranked. Everything else is reported in `not_ranked` WITH ITS REASON:
    a candidate the kernel cannot run, one whose optimum left the sequence
    its Jacobian assumed, and one that simply lost are three different facts,
    and a shortlist that blurs them is not evidence.
    """

    if not objectives:
        raise ComparisonError("a comparison needs at least one objective")

    rankable = [result for result in results if result.rankable]
    not_ranked = [
        {
            "topology_id": result.topology_id,
            "status": result.status,
            "reason": result.reason,
            "trust_region_held": (
                None if result.trust_region is None else result.trust_region.held
            ),
        }
        for result in results
        if not result.rankable
    ]

    missing = {
        result.topology_id: sorted(
            objective.metric
            for objective in objectives
            if objective.metric not in result.oracle_metrics
        )
        for result in rankable
    }
    incomplete = {name: metrics for name, metrics in missing.items() if metrics}
    if incomplete:
        raise ComparisonError(
            "cannot compare on metrics a finalist does not carry: "
            + "; ".join(f"{name} lacks {', '.join(items)}" for name, items in incomplete.items())
        )

    # "A finalist without complete loss cards must not win silently." An
    # uninstrumented model reports no dissipation at all, so on any
    # efficiency-flavoured axis it beats an instrumented rival for free. When
    # the field is mixed, every uninstrumented finalist is marked OPTIMISTIC
    # — the comparison stays available, but the advantage is named.
    instrumentation = {result.topology_id: result.loss_instrumented for result in rankable}
    mixed_instrumentation = len(set(instrumentation.values())) > 1

    points: list[dict[str, Any]] = []
    for result in rankable:
        dominated_by = sorted(
            other.topology_id
            for other in rankable
            if other.topology_id != result.topology_id
            and _dominates(other.oracle_metrics, result.oracle_metrics, objectives)
        )
        flags = set(result.flags)
        if mixed_instrumentation and not result.loss_instrumented:
            flags.add("OPTIMISTIC")
        points.append(
            {
                "topology_id": result.topology_id,
                # Objective values are the ORACLE's, never the gradient's.
                "objectives": {
                    objective.metric: result.oracle_metrics[objective.metric]
                    for objective in objectives
                },
                "dominated": bool(dominated_by),
                "dominated_by": dominated_by,
                "loss_instrumented": result.loss_instrumented,
                "flags": sorted(flags),
            }
        )
    points.sort(key=lambda item: item["topology_id"])

    optimistic = sorted(
        point["topology_id"] for point in points if "OPTIMISTIC" in point["flags"]
    )
    record = {
        "schema": TOPOLOGY_COMPARISON_SCHEMA,
        "claim": {
            "statement": (
                "Every ranked finalist was optimised to ITS OWN optimum and "
                "its reported metrics come from a full oracle run at that "
                "landing point; gradient values are recorded beside them, "
                "never in place of them. A front position is not a "
                "recommendation and carries no hardware claim. Candidates "
                "that could not be ranked are listed with the reason, "
                "because a candidate the kernel cannot run, one whose "
                "optimum left its trust region, and one that lost are three "
                "different facts."
            ),
        },
        "objectives": [objective.as_dict() for objective in objectives],
        "summary": {
            "finalists": len(results),
            "ranked": len(points),
            "not_ranked": len(not_ranked),
            "non_dominated": sum(1 for point in points if not point["dominated"]),
            "optimistic": optimistic,
            "mixed_loss_instrumentation": mixed_instrumentation,
        },
        "front": [point for point in points if not point["dominated"]],
        "points": points,
        "not_ranked": not_ranked,
        "finalists": [result.as_dict() for result in results],
    }
    record["comparison_sha256"] = sha256_json(record)
    return record


COMPARISON_REQUEST_SCHEMA = "pear-sim/comparison-request/v0.1"


def run_comparison_request(
    request: dict[str, Any], entries: tuple[TopologyEntry, ...]
) -> dict[str, Any]:
    """Execute a declarative comparison request, fail closed on its shape.

    The request is a file rather than a pile of flags because a fair
    comparison needs per-finalist bounds, and a bound quietly defaulted is a
    bound nobody chose.
    """

    if request.get("schema") != COMPARISON_REQUEST_SCHEMA:
        raise ComparisonError(
            f"comparison request must declare schema {COMPARISON_REQUEST_SCHEMA!r}"
        )
    unknown = sorted(
        set(request) - {"schema", "root", "objectives", "finalists", "max_iterations"}
    )
    if unknown:
        raise ComparisonError(
            "comparison request: unknown field(s): " + ", ".join(unknown)
        )
    raw_objectives = request.get("objectives")
    if not isinstance(raw_objectives, list) or not raw_objectives:
        raise ComparisonError("comparison request needs a non-empty objectives list")
    objectives = tuple(
        Objective(metric=item["metric"], sense=item["sense"]) for item in raw_objectives
    )
    raw_finalists = request.get("finalists")
    if not isinstance(raw_finalists, list) or not raw_finalists:
        raise ComparisonError("comparison request needs a non-empty finalists list")

    by_id = {entry.id: entry for entry in entries}
    results: list[FinalistResult] = []
    for item in raw_finalists:
        topology_id = item["topology_id"]
        entry = by_id.get(topology_id)
        if entry is None:
            raise ComparisonError(
                f"finalist {topology_id!r} is not in the library; a comparison "
                "may not invent candidates"
            )
        theta = item.get("theta") or {}
        if not isinstance(theta, dict) or not theta:
            raise ComparisonError(
                f"finalist {topology_id!r} declares no theta to optimise over"
            )
        bounds = {
            name: (float(pair[0]), float(pair[1])) for name, pair in theta.items()
        }
        results.append(
            optimize_finalist(
                entry,
                objective=objectives[0],
                theta_names=tuple(theta),
                bounds=bounds,
                root=str(request.get("root", ".")),
                extra_metrics=objectives[1:],
                max_iterations=int(request.get("max_iterations", 12)),
            )
        )
    return compare_finalists(tuple(results), objectives)


__all__ = [
    "COMPARISON_REQUEST_SCHEMA",
    "STATUSES",
    "TOPOLOGY_COMPARISON_SCHEMA",
    "run_comparison_request",
    "ComparisonError",
    "FinalistResult",
    "Objective",
    "OptimizationStep",
    "TierHealth",
    "TrustRegionVerdict",
    "compare_finalists",
    "optimize_finalist",
]
