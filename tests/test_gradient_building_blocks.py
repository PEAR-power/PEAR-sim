"""Gradient-engine building blocks: sequence extraction and DesignVector.

Acceptance criteria are asserted literally: extraction reproduces the event log for every listed
example with <= 4 modes, provenance is mandatory and matches the bundle
writer's recipe, a manual override disagreement is a VERDICT (not a raise),
and the named-parameter theta contract fails closed with a remedy in the message.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pear_sim import __version__
from pear_sim.analysis.gradient import (
    DesignVector,
    DesignVectorError,
    ExitCondition,
    ExitKind,
    ModeSequenceError,
    ModeStep,
    extract_mode_sequence,
    manual_mode_sequence,
    sequences_agree,
)
from pear_sim.model import load_model
from pear_sim.run.contracts import model_hashes, sha256_json
from pear_sim.solver import HybridSimulator

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"

# Every light example whose cycle visits <= 4 distinct modes. The heavyweight
# models (NPC 3L/5L, PMSM drive, rectifiers) are excluded on runtime budget,
# not on principle — their mode counts exceed four anyway, except where noted.
SMALL_MODE_EXAMPLES = {
    "buck.yaml": 20e3,
    "buck_dcm.yaml": 20e3,
    "boost.yaml": 25e3,
    "synchronous_buck_complementary.yaml": 100e3,
}


@pytest.fixture(scope="module")
def buck_run():
    model = load_model(EXAMPLES / "buck.yaml")
    return model, HybridSimulator(model).simulate()


# ---------------------------------------------------------------------------
# Sequence extraction — extract_mode_sequence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("example,fsw", sorted(SMALL_MODE_EXAMPLES.items()))
def test_extraction_reproduces_the_event_log(example: str, fsw: float) -> None:
    """Acceptance: for every <=4-mode example, the sequence matches the run.

    The comparison is against ``result.events`` and interval durations of the
    same run — never against a stored expectation, so this cannot drift into
    testing a snapshot of itself.
    """

    model = load_model(EXAMPLES / example)
    result = HybridSimulator(model).simulate()
    period = 1.0 / fsw
    sequence = extract_mode_sequence(result, period_s=period)

    distinct = {step.signature for step in sequence.steps}
    assert 1 <= len(distinct) <= 4, example

    # Durations of one cycle add up to the period exactly.
    total = sum(step.nominal_duration_s for step in sequence.steps)
    assert total == pytest.approx(period, rel=1e-9)

    # Every step boundary except the last must coincide with a recorded event
    # of the kind the classifier claims.
    start = float(result.mode_intervals[0]["start_s"])
    stop = float(result.mode_intervals[-1]["end_s"])
    cycles = int((stop - start + 1e-12) / period)
    window_start = start + (cycles - 1) * period
    boundary = window_start
    for step in sequence.steps[:-1]:
        boundary += step.nominal_duration_s
        matching = [
            event for event in result.events if abs(event.time - boundary) <= 1e-9
        ]
        assert matching, (example, boundary)
        kinds = {event.kind for event in matching}
        expected_kind = {
            ExitKind.SCHEDULED_GATE: "scheduled_gate",
            ExitKind.NATURAL_GUARD: "natural_commutation",
            ExitKind.SATURATION: "saturation_boundary",
        }[step.exit_condition.kind]
        assert expected_kind in kinds, (example, boundary, kinds)
    assert sequence.steps[-1].exit_condition.kind is ExitKind.PERIOD_END


def test_provenance_is_the_bundle_writers_run_identity(buck_run) -> None:
    """Acceptance: the sequence carries the run's sha256 — same recipe as run.json."""

    model, result = buck_run
    sequence = extract_mode_sequence(result, period_s=1.0 / 20e3)
    hashes = model_hashes(model)
    expected = sha256_json(
        {
            "model_sha256": hashes["model_sha256"],
            "request_sha256": hashes["request_sha256"],
            "backend": "reference-python",
            "pear_sim_version": __version__,
        }
    )
    assert sequence.source_run_sha256 == expected
    assert sequence.period_s == pytest.approx(1.0 / 20e3)


def test_a_dcm_cycle_shows_the_natural_guard_with_a_current_crossing(
    tmp_path: Path,
) -> None:
    """The DCM buck's diode turn-off must classify as a NATURAL_GUARD exit on
    the branch current — that classification is what the saltation extension will
    consume, so it has to be right before it is ever differentiated."""

    model = load_model(EXAMPLES / "buck_dcm.yaml")
    result = HybridSimulator(model).simulate()
    sequence = extract_mode_sequence(result, period_s=1.0 / 20e3)
    natural = [
        step
        for step in sequence.steps
        if step.exit_condition.kind is ExitKind.NATURAL_GUARD
    ]
    assert natural, "a DCM cycle must contain a natural commutation"
    assert all(step.exit_condition.guard_is_current for step in natural), (
        "leaving conduction crosses the CURRENT guard"
    )
    # And the idle mode really is the frozen-inductor one.
    assert any(step.constrained_inductors for step in sequence.steps)


def test_manual_override_disagreement_is_a_verdict_not_a_raise(buck_run) -> None:
    """Acceptance: a hand-declared sequence is compared, and the discrepancy
    comes back as data. This is the T3 contract one level down."""

    _, result = buck_run
    observed = extract_mode_sequence(result, period_s=1.0 / 20e3)

    # A CCM declaration for a CCM run agrees with itself.
    agree, reasons = sequences_agree(observed, observed, duration_rtol=1e-9)
    assert agree and reasons == ()

    # A wrong declaration (one step instead of two) disagrees with REASONS.
    declared = manual_mode_sequence(
        period_s=observed.period_s,
        steps=(observed.steps[0],),
    )
    assert declared.source_run_sha256 is None, (
        "hand-authored provenance must be visibly absent"
    )
    agree, reasons = sequences_agree(declared, observed, duration_rtol=1e-9)
    assert not agree
    assert any("signatures differ" in reason for reason in reasons)


def test_wrong_durations_disagree_even_with_matching_signatures(buck_run) -> None:
    _, result = buck_run
    observed = extract_mode_sequence(result, period_s=1.0 / 20e3)
    skewed_steps = tuple(
        ModeStep(
            signature=step.signature,
            gates=step.gates,
            diodes=step.diodes,
            constrained_inductors=step.constrained_inductors,
            saturated_inductors=step.saturated_inductors,
            exit_condition=step.exit_condition,
            nominal_duration_s=step.nominal_duration_s * (1.5 if index == 0 else 0.5),
        )
        for index, step in enumerate(observed.steps)
    )
    declared = manual_mode_sequence(period_s=observed.period_s, steps=skewed_steps)
    agree, reasons = sequences_agree(declared, observed, duration_rtol=1e-3)
    assert not agree
    assert any("duration differs" in reason for reason in reasons)


def _exactly_timed(steps, position: int) -> bool:
    from pear_sim.analysis.gradient.sequence import _step_is_exactly_timed

    return _step_is_exactly_timed(steps, position)


def _reskew(sequence, factor_for):
    """Copy a sequence with each step's duration scaled by ``factor_for(index)``."""

    return manual_mode_sequence(
        period_s=sequence.period_s,
        steps=tuple(
            ModeStep(
                signature=step.signature,
                gates=step.gates,
                diodes=step.diodes,
                constrained_inductors=step.constrained_inductors,
                saturated_inductors=step.saturated_inductors,
                exit_condition=step.exit_condition,
                nominal_duration_s=step.nominal_duration_s * factor_for(index),
            )
            for index, step in enumerate(sequence.steps)
        ),
    )


def test_a_located_instant_is_not_held_to_a_tolerance_the_solver_never_had() -> None:
    """The cross-platform failure this split was built for.

    A line-commutated rectifier has no scheduled edges: every boundary is a
    root of h(x, t) = 0 found on the integrated trajectory. CI measured the
    consequence — the 12-pulse library entry re-extracted on ubuntu and
    windows moved its step durations by 3e-9 to 9.1e-9 relative, on a model
    whose own integration rtol is 1e-7. macOS passed only because its
    arithmetic happened to land identically.

    The DCM buck is the small fixture with both kinds of boundary in one
    cycle, so it can separate them: the SAME perturbation must be accepted
    where the instant was located and refused where it was declared.
    """

    result = HybridSimulator(load_model(EXAMPLES / "buck_dcm.yaml")).simulate()
    observed = extract_mode_sequence(result, period_s=1.0 / 20e3)
    exact = [
        position
        for position in range(len(observed.steps))
        if _exactly_timed(observed.steps, position)
    ]
    located = [
        position for position in range(len(observed.steps)) if position not in exact
    ]
    assert exact and located, (
        "this fixture must contain both a declared and a located boundary, "
        f"got kinds {[s.exit_condition.kind.value for s in observed.steps]}"
    )

    noise = 9.1e-9  # the worst relative deviation CI actually measured

    # Perturb ONLY the located steps: accepted, because the trajectory those
    # roots were found on was never integrated more tightly than 1e-7.
    on_located = _reskew(
        observed, lambda index: 1.0 + noise if index in located else 1.0
    )
    agree, reasons = sequences_agree(
        on_located, observed, duration_rtol=1e-9, guard_duration_rtol=1e-7
    )
    assert agree, reasons

    # The same perturbation on a declared boundary is a real change: a
    # scheduled edge is arithmetic on the declared frequency and duty, and
    # two machines computing it disagree by nothing at all.
    on_exact = _reskew(observed, lambda index: 1.0 + noise if index in exact else 1.0)
    agree, reasons = sequences_agree(
        on_exact, observed, duration_rtol=1e-9, guard_duration_rtol=1e-7
    )
    assert not agree
    assert any("beyond rtol" in reason for reason in reasons)


def test_a_duration_that_moved_by_percent_is_still_refused(buck_run) -> None:
    """The loosened gate must not stop being a gate.

    What this audit exists to catch is a stored cycle that no longer matches
    reality. Those move by percent, not by parts per billion.
    """

    _, result = buck_run
    observed = extract_mode_sequence(result, period_s=1.0 / 20e3)
    real_change = _reskew(observed, lambda index: 1.02 if index == 0 else 0.98)

    agree, reasons = sequences_agree(
        real_change, observed, duration_rtol=1e-9, guard_duration_rtol=1e-7
    )

    assert not agree
    assert any("duration differs" in reason for reason in reasons)


def test_a_single_tolerance_still_means_what_it_used_to() -> None:
    """Backward compatibility, stated as an identity rather than a sample.

    A caller that passes one tolerance has not thought about the two kinds of
    instant, and must not be silently handed a looser gate than it asked for.
    Omitting ``guard_duration_rtol`` therefore has to be exactly equivalent to
    passing the same value twice — for every fixture, not just a lucky one.
    """

    result = HybridSimulator(load_model(EXAMPLES / "buck_dcm.yaml")).simulate()
    observed = extract_mode_sequence(result, period_s=1.0 / 20e3)

    for rtol in (1e-12, 1e-9, 1e-6, 1e-3):
        skewed = _reskew(observed, lambda index: 1.0 + 1e-7)
        assert sequences_agree(skewed, observed, duration_rtol=rtol) == sequences_agree(
            skewed, observed, duration_rtol=rtol, guard_duration_rtol=rtol
        )


def test_a_duration_is_only_as_exact_as_both_its_ends() -> None:
    """A scheduled edge FOLLOWING a natural commutation is not exact.

    The duration spans two instants. Treating it as exact because its own
    exit is scheduled would reintroduce the platform dependence through the
    back door, on the step nobody would think to check.
    """

    from pear_sim.analysis.gradient.sequence import _step_is_exactly_timed

    def step(kind: ExitKind) -> ModeStep:
        return ModeStep(
            signature=f"s[{kind.value}]",
            gates={},
            diodes={},
            constrained_inductors=(),
            saturated_inductors=(),
            exit_condition=ExitCondition(kind=kind, target="", guard_is_current=False),
            nominal_duration_s=1e-5,
        )

    steps = (
        step(ExitKind.SCHEDULED_GATE),
        step(ExitKind.NATURAL_GUARD),
        step(ExitKind.SCHEDULED_GATE),
        step(ExitKind.PERIOD_END),
    )

    assert _step_is_exactly_timed(steps, 0) is True
    assert _step_is_exactly_timed(steps, 1) is False
    # Its own exit is scheduled, but it starts where the commutation landed.
    assert _step_is_exactly_timed(steps, 2) is False
    assert _step_is_exactly_timed(steps, 3) is True


def test_extraction_fails_closed_without_a_complete_cycle(buck_run) -> None:
    _, result = buck_run
    with pytest.raises(ModeSequenceError, match="complete cycle"):
        extract_mode_sequence(result, period_s=10.0)
    with pytest.raises(ModeSequenceError, match="cycle_index"):
        extract_mode_sequence(result, period_s=1.0 / 20e3, cycle_index=10_000)


def test_zero_core_changes() -> None:
    """Acceptance: sequence extraction must not touch src/pear_sim/core.

    Guarded structurally: the gradient package must not be imported by core,
    and this module's own imports go one way only.
    """

    core = (ROOT / "src" / "pear_sim" / "core").rglob("*.py")
    for path in core:
        assert "analysis.gradient" not in path.read_text(encoding="utf-8"), path


# ---------------------------------------------------------------------------
# Design vector — DesignVector
# ---------------------------------------------------------------------------

BUCK_THETA = ("l", "c", "fsw", "duty", "load")


def test_round_trip_offsets_change_exactly_one_parameter() -> None:
    """Acceptance: with_offset on every theta of buck.yaml loads a model that
    differs in precisely that parameter."""

    vector = DesignVector.from_model(EXAMPLES / "buck.yaml", BUCK_THETA)
    base = vector.load()
    assert set(vector.names()) == set(BUCK_THETA)
    assert np.all(np.isfinite(vector.as_array()))

    for name in BUCK_THETA:
        shifted = vector.with_offset(name, 0.01 * vector.value(name))
        model = shifted.load()
        for other in BUCK_THETA:
            expected = shifted.value(other)
            assert float(model.parameters[other].value) == pytest.approx(
                expected, rel=1e-12
            ), (name, other)
        # Only the shifted entry moved.
        deltas = shifted.as_array() - vector.as_array()
        moved = {n for n, d in zip(vector.names(), deltas) if d != 0.0}
        assert moved == {name}
        # And the untouched original still loads to the baseline.
        assert float(base.parameters[name].value) == pytest.approx(
            vector.value(name), rel=1e-12
        )


def test_theta_must_be_a_named_parameter(tmp_path: Path) -> None:
    """Acceptance: unknown parameter fails closed, names the gap."""

    with pytest.raises(DesignVectorError, match="not declared") as excinfo:
        DesignVector.from_model(EXAMPLES / "buck.yaml", ("no_such_theta",))
    assert excinfo.value.code == "gradient.design_vector_invalid"


def test_inline_component_value_gets_a_remedy_not_just_a_refusal() -> None:
    """Acceptance: a component id as theta names the fix — move the value
    to parameters: — not merely the failure."""

    with pytest.raises(DesignVectorError, match="parameters: section") as excinfo:
        DesignVector.from_model(EXAMPLES / "buck.yaml", ("d1",))
    assert "${d1}" in str(excinfo.value)


def test_vector_is_immutable_and_validates_shape() -> None:
    vector = DesignVector.from_model(EXAMPLES / "buck.yaml", ("l", "c"))
    replaced = vector.with_values(np.array([2e-3, 2e-4]))
    assert replaced.value("l") == pytest.approx(2e-3)
    assert vector.value("l") != replaced.value("l")
    with pytest.raises(DesignVectorError, match="shape"):
        vector.with_values(np.array([1.0]))
    with pytest.raises(DesignVectorError, match="finite"):
        vector.with_values(np.array([np.nan, 1.0]))
    with pytest.raises(DesignVectorError, match="duplicate"):
        DesignVector.from_model(EXAMPLES / "buck.yaml", ("l", "l"))


def test_overrides_keep_full_precision_and_units() -> None:
    """The override strings must survive the loader's unit check and carry the
    exact float — repr round-trips, a formatted decimal would not."""

    vector = DesignVector.from_model(EXAMPLES / "buck.yaml", ("l", "duty"))
    ugly = vector.with_values(np.array([1.2345678901234567e-3, 0.4123456789012345]))
    model = ugly.load()
    assert float(model.parameters["l"].value) == 1.2345678901234567e-3
    assert float(model.parameters["duty"].value) == 0.4123456789012345
