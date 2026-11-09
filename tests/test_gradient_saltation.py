# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Gradient-engine gate: natural commutation and saltation matrices.

The declared sequence may now contain NATURAL_GUARD exits: the engine solves
each exit instant from the guard's closed form (first-crossing scan — an
underdamped guard RINGS, so endpoint sign tests miss even-count crossings;
measured on buck_dcm, where g(0) > 0 and g(40 us) > 0 bracket a crossing at
0.85 us), composes the monodromy through saltation matrices
S = I + (f+ - f-) grad_h / (grad_h . f-), and Newton-iterates the hybrid cycle
map from a seed built out of the DECLARED durations (a cold zero seed hands
the piecewise-smooth Newton a regime two-cycle; measured before the seed
existed).

The strongest correctness statement here is quantitative: with saltation in
the monodromy, every DCM derivative agrees with finite differences on the
full oracle to 1e-8 or better OF THE GATE — if S were silently identity, the
duty derivatives would be off by whole percents, so the agreement is the
proof, not the decoration.

Per the scoping decided BEFORE implementation: coincident events
resolve deterministically — the natural exit fires first, the trailing step
degenerates to zero length — and the arbitrariness is FLAGGED, never silent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pear_sim.analysis.gradient import (
    DesignVector,
    MetricSpec,
    extract_mode_sequence,
    solve,
)
from pear_sim.analysis.steady_state import one_cycle_model, solve_periodic_steady_state
from pear_sim.benchmarks.buck_boundary import BuckBoundaryParameters, find_bcm_load
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
PERIOD_S = 1.0 / 20e3
DCM_THETA = ("duty", "load", "vin")
METRICS = (
    MetricSpec("v(out)", "mean"),
    MetricSpec("v(out)", "rms"),
    MetricSpec("i(l1)", "mean"),
)

# Measured 1.2e-11 against the production shooting solver; the gate keeps the
# step-1 headroom.
T1_STATE_TOLERANCE = 1e-7
# The T2 criterion, unchanged; measured worst entry 3.5e-8 of the gate.
GATE = 1e-3


@pytest.fixture(scope="module")
def dcm_sequence():
    run = HybridSimulator(load_model(EXAMPLES / "buck_dcm.yaml")).simulate()
    return extract_mode_sequence(run, period_s=PERIOD_S)


@pytest.fixture(scope="module")
def dcm_result(dcm_sequence):
    vector = DesignVector.from_model(EXAMPLES / "buck_dcm.yaml", DCM_THETA)
    return vector, solve(vector, dcm_sequence, METRICS, jacobian=True)


def _oracle_metrics(vector: DesignVector) -> dict[str, float]:
    model = vector.load()
    steady = solve_periodic_steady_state(model)
    assert steady.converged
    run = HybridSimulator(one_cycle_model(model, np.asarray(steady.state))).simulate()
    computed = run.metrics()
    return {
        spec.name: float(computed[spec.output][spec.statistic]) for spec in METRICS
    }


# ---------------------------------------------------------------------------
# Steady state and realized exit times
# ---------------------------------------------------------------------------


def test_dcm_steady_state_matches_the_shooting_solver(dcm_result) -> None:
    _, result = dcm_result
    reference = solve_periodic_steady_state(load_model(EXAMPLES / "buck_dcm.yaml"))
    ours = np.asarray(result.steady_state.state)
    theirs = np.asarray(reference.state)
    deviation = float(np.max(np.abs(ours - theirs) / (1.0 + np.abs(theirs))))
    assert deviation <= T1_STATE_TOLERANCE, deviation
    assert result.steady_state.converged
    assert result.steady_state.iterations <= 5
    assert result.flag("declared_sequence_held").held


def test_the_realized_exit_time_matches_the_oracle_run(
    dcm_sequence, dcm_result
) -> None:
    """The diode's conduction sliver is SOLVED, not copied — and it lands on
    the duration the oracle's own event log recorded (0.8477 us)."""

    _, result = dcm_result
    solved = result.steady_state.segment_durations_s[1]
    recorded = dcm_sequence.steps[1].nominal_duration_s
    assert solved == pytest.approx(recorded, rel=1e-3)
    # And the idle tail completes the interval exactly.
    total = sum(result.steady_state.segment_durations_s)
    assert total == pytest.approx(PERIOD_S, rel=1e-12)


# ---------------------------------------------------------------------------
# The saltation gate: DCM derivatives against the full oracle
# ---------------------------------------------------------------------------


def test_dcm_jacobian_matches_oracle_finite_differences(dcm_result) -> None:
    """Nine derivatives, all within the T2 criterion (measured <= 3.5e-8 of
    it). This is the saltation proof: the exit instant depends on theta and
    on the cycle state, and an identity saltation would corrupt every duty
    entry by percents."""

    vector, result = dcm_result
    jacobian = result.jacobian.as_array()
    theta = vector.as_array()
    for j, name in enumerate(DCM_THETA):
        step = 1e-4 * abs(theta[j])
        plus = _oracle_metrics(vector.with_offset(name, +step))
        minus = _oracle_metrics(vector.with_offset(name, -step))
        for i, spec in enumerate(METRICS):
            reference = (plus[spec.name] - minus[spec.name]) / (2.0 * step)
            ours = jacobian[i, j]
            characteristic = abs(result.metrics[spec.name]) / max(
                abs(theta[j]), 1e-30
            )
            error = abs(ours - reference) / (abs(reference) + GATE * characteristic)
            assert error <= GATE, (name, spec.name, ours, reference, error)


def test_t4_dead_time_derivative_matches_the_oracle() -> None:
    """T4, admitted only once saltation is modelled. Measured 7.3e-9 relative; the
    closed form is -fsw*Vin because the freewheeling diode hands the pole to
    the low rail for exactly the dead-time window."""

    run = HybridSimulator(
        load_model(EXAMPLES / "synchronous_buck_complementary.yaml")
    ).simulate()
    sequence = extract_mode_sequence(run, period_s=PERIOD_S)
    vector = DesignVector.from_model(
        EXAMPLES / "synchronous_buck_complementary.yaml", ("dead_time",)
    )
    result = solve(vector, sequence, (MetricSpec("v(out)", "mean"),), jacobian=True)
    ours = result.jacobian.as_array()[0, 0]

    def oracle_mean(v: DesignVector) -> float:
        model = v.load()
        steady = solve_periodic_steady_state(model)
        cycle = one_cycle_model(model, np.asarray(steady.state))
        return float(HybridSimulator(cycle).simulate().metrics()["v(out)"]["mean"])

    # dead_time enters the schedule LINEARLY, so a large relative step is
    # exact for the reference and clears the oracle's noise floor by orders.
    step = 0.1 * 5e-7
    reference = (
        oracle_mean(vector.with_offset("dead_time", +step))
        - oracle_mean(vector.with_offset("dead_time", -step))
    ) / (2.0 * step)
    assert ours == pytest.approx(reference, rel=1e-6)
    assert ours == pytest.approx(-2.0e6, rel=1e-6)  # -fsw * Vin


# ---------------------------------------------------------------------------
# Coincidence policy and declaration boundaries
# ---------------------------------------------------------------------------


def test_bcm_coincidence_is_deterministic_and_flagged(dcm_sequence) -> None:
    """At the CCM/DCM boundary the natural exit lands exactly on the
    scheduled edge. Policy (decided before implementation): the natural exit
    fires first, the idle step degenerates to zero length, the run is
    bit-repeatable, and the arbitrariness is a FLAG."""

    boundary = find_bcm_load(
        BuckBoundaryParameters(
            vin_v=100.0,
            inductance_h=1e-4,
            capacitance_f=1e-6,
            switching_frequency_hz=20e3,
            duty=0.2,
        )
    )
    vector = DesignVector.from_model(EXAMPLES / "buck_dcm.yaml", ("load",)).with_values(
        np.asarray([boundary])
    )
    first = solve(vector, dcm_sequence, (MetricSpec("v(out)", "mean"),), jacobian=False)
    second = solve(
        vector, dcm_sequence, (MetricSpec("v(out)", "mean"),), jacobian=False
    )
    assert first.steady_state.converged
    assert first.metrics == second.metrics
    assert first.steady_state.state == second.steady_state.state
    flag = first.flag("event_ordering_unambiguous")
    assert not flag.held
    assert "deterministically" in flag.detail
    # The idle step really did degenerate.
    assert first.steady_state.segment_durations_s[-1] == pytest.approx(0.0, abs=1e-9)


def test_a_declaration_the_physics_never_fires_is_a_flag_not_a_raise(
    dcm_sequence,
) -> None:
    """Heavy load pushes the converter into CCM: the declared diode exit
    never fires AT THE CONVERGED POINT. Intermediate Newton iterates are
    allowed to miss it — a cold pass legitimately spends time in the wrong
    regime — but the verdict is only issued at the fixed point."""

    vector = DesignVector.from_model(EXAMPLES / "buck_dcm.yaml", ("load",)).with_values(
        np.asarray([5.0])
    )
    result = solve(vector, dcm_sequence, METRICS, jacobian=False)
    flag = result.flag("declared_sequence_held")
    assert not flag.held
    assert "never fires" in flag.detail
    assert np.isnan(result.metrics["mean(v(out))"])
    assert result.jacobian is None


def test_all_scheduled_sequences_still_take_the_exact_affine_path(
    dcm_sequence,
) -> None:
    """Step-1 behaviour through the general machinery: for the plain CCM buck
    the seed IS the fixed point and Newton exits on its first verification
    pass."""

    run = HybridSimulator(load_model(EXAMPLES / "buck.yaml")).simulate()
    sequence = extract_mode_sequence(run, period_s=PERIOD_S)
    vector = DesignVector.from_model(EXAMPLES / "buck.yaml", ("duty",))
    result = solve(vector, sequence, (MetricSpec("v(out)", "mean"),), jacobian=False)
    assert result.steady_state.converged
    assert result.steady_state.iterations == 1
    assert result.metrics["mean(v(out))"] == pytest.approx(40.0, rel=1e-9)
