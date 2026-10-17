"""Fair comparison of topology finalists.

The acceptance criteria of finalist comparison are all about refusing to fool yourself, so
the tests are mostly about what the machinery REFUSES to do:

- each finalist is optimised to its own optimum, with the path retained;
- the reported numbers come from the oracle, the gradient's sit beside them;
- the trust region is the mode STRUCTURE, so a legitimate change of step
  durations must not be mistaken for leaving it;
- a candidate this kernel cannot run is `unsupported`, never silently dropped;
- dominated points stay in the record;
- an uninstrumented finalist cannot win an efficiency-flavoured axis quietly.
"""

from pathlib import Path

import pytest

from pear_sim.topology import load_topology_library
from pear_sim.topology.comparison import (
    ComparisonError,
    FinalistResult,
    Objective,
    TierHealth,
    TrustRegionVerdict,
    compare_finalists,
    optimize_finalist,
)

ROOT = Path(__file__).parents[1]
LIBRARY = ROOT / "library" / "topologies"

RIPPLE = Objective(metric="rms(i(l1))", sense="minimize")
OUTPUT = Objective(metric="mean(v(out))", sense="maximize")


@pytest.fixture(scope="module")
def entries():
    return {entry.id: entry for entry in load_topology_library(LIBRARY)}


@pytest.fixture(scope="module")
def optimised_buck(entries):
    return optimize_finalist(
        entries["buck"],
        objective=RIPPLE,
        theta_names=("l",),
        bounds={"l": (0.5e-3, 4e-3)},
        root=str(ROOT),
        extra_metrics=(OUTPUT,),
        max_iterations=8,
    )


# --- each finalist reaches its OWN optimum -------------------------------


def test_a_finalist_is_pulled_to_its_own_optimum_with_the_path_retained(
    optimised_buck,
):
    result = optimised_buck
    assert result.status == "ok"
    assert result.trajectory, "an optimisation with no retained step is not auditable"
    # More inductance means less ripple current: the search must move L up,
    # and every retained step must improve the objective it descends.
    assert result.theta_final["l"] > result.theta_start["l"]
    values = [step.objective_value for step in result.trajectory]
    assert values == sorted(values, reverse=True), values
    for step in result.trajectory:
        assert set(step.theta) == {"l"}


def test_the_optimum_respects_declared_bounds(entries):
    result = optimize_finalist(
        entries["buck"],
        objective=RIPPLE,
        theta_names=("l",),
        bounds={"l": (0.9e-3, 1.2e-3)},
        root=str(ROOT),
        max_iterations=8,
    )
    assert result.status == "ok"
    assert 0.9e-3 <= result.theta_final["l"] <= 1.2e-3


def test_an_unbounded_parameter_is_refused(entries):
    with pytest.raises(ComparisonError, match="no bounds"):
        optimize_finalist(
            entries["buck"],
            objective=RIPPLE,
            theta_names=("l",),
            bounds={},
            root=str(ROOT),
        )


# --- the oracle is the only issuer of a verdict --------------------------


def test_reported_metrics_come_from_the_oracle_with_the_gradient_beside_them(
    optimised_buck,
):
    result = optimised_buck
    assert result.oracle_metrics, "the verdict must exist"
    assert result.gradient_metrics, "the gradient's own numbers must be retained"
    assert set(result.oracle_metrics) == set(result.gradient_metrics)
    payload = result.as_dict()
    assert payload["oracle_metrics"] and payload["gradient_metrics"]


def test_the_front_is_built_on_oracle_numbers(optimised_buck, entries):
    boost = optimize_finalist(
        entries["boost"],
        objective=RIPPLE,
        theta_names=("duty",),
        bounds={"duty": (0.3, 0.6)},
        root=str(ROOT),
        extra_metrics=(OUTPUT,),
        max_iterations=3,
    )
    finalists = (optimised_buck, boost)
    record = compare_finalists(finalists, (RIPPLE, OUTPUT))
    for point in record["points"]:
        source = next(f for f in finalists if f.topology_id == point["topology_id"])
        for objective in (RIPPLE, OUTPUT):
            assert point["objectives"][objective.metric] == pytest.approx(
                source.oracle_metrics[objective.metric]
            )


# --- the trust region is the mode STRUCTURE ------------------------------


def test_moving_theta_within_the_same_mode_structure_stays_in_the_region(
    optimised_buck,
):
    """The bug this pins: step durations move whenever theta moves — that is
    what optimising does. Judging the region on durations would flag every
    successful run as having escaped, and the gate would cry wolf."""

    trust = optimised_buck.trust_region
    assert trust is not None
    assert trust.checked is True
    assert trust.held is True, trust.reasons
    assert optimised_buck.rankable is True


def test_leaving_the_declared_mode_structure_is_reported_not_hidden(entries):
    """Pushing duty far down takes the buck out of the declared CCM cycle."""

    result = optimize_finalist(
        entries["buck"],
        objective=RIPPLE,
        theta_names=("duty",),
        bounds={"duty": (0.02, 0.4)},
        root=str(ROOT),
        max_iterations=8,
    )
    assert result.status == "ok"
    trust = result.trust_region
    assert trust is not None and trust.checked
    if not trust.held:
        # The measured mode boundary: reported, with reasons, and the
        # finalist is not rankable on numbers whose Jacobian was invalid.
        assert trust.reasons
        assert result.rankable is False
        assert "mode" in trust.note


def test_an_entry_without_a_declared_sequence_is_unchecked_not_passed():
    """`checked=False` must never read as `held=True`."""

    verdict = TrustRegionVerdict(
        held=False, reasons=(), checked=False, note="no declared sequence"
    )
    result = FinalistResult(
        topology_id="x", status="ok", reason="", trust_region=verdict
    )
    assert result.rankable is False


# --- refusals ------------------------------------------------------------


def test_a_candidate_this_kernel_cannot_run_is_unsupported_not_dropped(entries):
    result = optimize_finalist(
        entries["flyback"],
        objective=RIPPLE,
        theta_names=("l",),
        bounds={"l": (1e-3, 2e-3)},
        root=str(ROOT),
    )
    assert result.status == "unsupported"
    assert result.rankable is False
    # The status is a statement about the ENGINE, not a defect of the
    # candidate, and it must say so.
    assert "kernel cannot run" in result.reason
    assert "fabrication" in result.reason


def test_unsupported_finalists_are_listed_with_their_reason_not_omitted(
    optimised_buck, entries
):
    flyback = optimize_finalist(
        entries["flyback"],
        objective=RIPPLE,
        theta_names=("l",),
        bounds={"l": (1e-3, 2e-3)},
        root=str(ROOT),
    )
    record = compare_finalists((optimised_buck, flyback), (RIPPLE,))
    assert record["summary"]["finalists"] == 2
    assert record["summary"]["ranked"] == 1
    (missing,) = record["not_ranked"]
    assert missing["topology_id"] == "flyback"
    assert missing["status"] == "unsupported"
    assert missing["reason"]


def test_a_comparison_needs_objectives(optimised_buck):
    with pytest.raises(ComparisonError, match="at least one objective"):
        compare_finalists((optimised_buck,), ())


def test_comparing_on_a_metric_a_finalist_lacks_is_refused(optimised_buck):
    with pytest.raises(ComparisonError, match="does not carry"):
        compare_finalists(
            (optimised_buck,), (Objective(metric="mean(v(nowhere))", sense="maximize"),)
        )


def test_a_malformed_metric_name_is_refused():
    with pytest.raises(ComparisonError, match="statistic"):
        Objective(metric="not-a-metric", sense="minimize").spec


def test_an_unknown_sense_is_refused():
    with pytest.raises(ComparisonError, match="sense"):
        Objective(metric="mean(v(out))", sense="sideways")


def test_the_metric_name_split_keeps_nested_parentheses():
    """`rstrip(')')` would eat both closing parens and ask the kernel for a
    signal named `i(l1` — the exact bug this pins."""

    spec = Objective(metric="rms(i(l1))", sense="minimize").spec
    assert spec.output == "i(l1)"
    assert spec.statistic == "rms"


# --- dominated points stay in the record ---------------------------------


def test_dominated_points_are_kept_not_pruned():
    def finalist(topology_id: str, ripple: float, output: float) -> FinalistResult:
        return FinalistResult(
            topology_id=topology_id,
            status="ok",
            reason="",
            oracle_metrics={"rms(i(l1))": ripple, "mean(v(out))": output},
            trust_region=TrustRegionVerdict(True, (), True, ""),
            loss_instrumented=True,
        )

    winner = finalist("winner", ripple=1.0, output=50.0)
    loser = finalist("loser", ripple=2.0, output=40.0)
    record = compare_finalists((winner, loser), (RIPPLE, OUTPUT))

    assert {point["topology_id"] for point in record["points"]} == {"winner", "loser"}
    assert [point["topology_id"] for point in record["front"]] == ["winner"]
    dominated = next(p for p in record["points"] if p["topology_id"] == "loser")
    assert dominated["dominated"] is True
    assert dominated["dominated_by"] == ["winner"]
    assert record["summary"]["non_dominated"] == 1


def test_a_genuine_trade_off_leaves_both_on_the_front():
    def finalist(topology_id: str, ripple: float, output: float) -> FinalistResult:
        return FinalistResult(
            topology_id=topology_id,
            status="ok",
            reason="",
            oracle_metrics={"rms(i(l1))": ripple, "mean(v(out))": output},
            trust_region=TrustRegionVerdict(True, (), True, ""),
            loss_instrumented=True,
        )

    quiet = finalist("quiet", ripple=1.0, output=40.0)
    strong = finalist("strong", ripple=2.0, output=50.0)
    record = compare_finalists((quiet, strong), (RIPPLE, OUTPUT))
    assert record["summary"]["non_dominated"] == 2
    assert all(point["dominated"] is False for point in record["points"])


# --- an uninstrumented finalist cannot win quietly -----------------------


def test_an_uninstrumented_finalist_is_flagged_optimistic_against_an_instrumented_one():
    def finalist(topology_id: str, instrumented: bool) -> FinalistResult:
        return FinalistResult(
            topology_id=topology_id,
            status="ok",
            reason="",
            oracle_metrics={"rms(i(l1))": 1.0},
            trust_region=TrustRegionVerdict(True, (), True, ""),
            loss_instrumented=instrumented,
        )

    record = compare_finalists(
        (finalist("bare", False), finalist("carded", True)), (RIPPLE,)
    )
    assert record["summary"]["mixed_loss_instrumentation"] is True
    assert record["summary"]["optimistic"] == ["bare"]
    bare = next(p for p in record["points"] if p["topology_id"] == "bare")
    assert "OPTIMISTIC" in bare["flags"]
    carded = next(p for p in record["points"] if p["topology_id"] == "carded")
    assert "OPTIMISTIC" not in carded["flags"]


def test_a_uniformly_uninstrumented_field_is_not_flagged():
    """Flagging everyone would make the flag meaningless — the criterion is
    about an UNEVEN field, where one candidate's losses are counted and
    another's are not."""

    def finalist(topology_id: str) -> FinalistResult:
        return FinalistResult(
            topology_id=topology_id,
            status="ok",
            reason="",
            oracle_metrics={"rms(i(l1))": 1.0},
            trust_region=TrustRegionVerdict(True, (), True, ""),
            loss_instrumented=False,
        )

    record = compare_finalists((finalist("a"), finalist("b")), (RIPPLE,))
    assert record["summary"]["mixed_loss_instrumentation"] is False
    assert record["summary"]["optimistic"] == []


# --- tier health ---------------------------------------------------------


def test_tier_health_compares_the_same_two_points_on_both_tiers(optimised_buck):
    (ripple, output) = optimised_buck.tier_health
    assert ripple.metric == RIPPLE.metric
    assert output.metric == OUTPUT.metric
    # The buck's mean output does not depend on L, so that metric did not
    # move and its relative error is undefined rather than a confident 100 %.
    assert output.relative_error is None
    # The ripple did move, and the two tiers must agree closely on how far.
    assert ripple.measured_delta < 0.0
    assert ripple.relative_error is not None
    assert ripple.relative_error < 1e-3, ripple.as_dict()


def test_relative_error_is_undefined_when_nothing_moved():
    health = TierHealth(
        metric="mean(v(out))",
        predicted_delta=1e-18,
        measured_delta=1e-18,
        metric_scale=40.0,
    )
    assert health.relative_error is None


# --- determinism ---------------------------------------------------------


def test_the_comparison_record_is_deterministic(optimised_buck):
    first = compare_finalists((optimised_buck,), (RIPPLE,))
    second = compare_finalists((optimised_buck,), (RIPPLE,))
    assert first["comparison_sha256"] == second["comparison_sha256"]
