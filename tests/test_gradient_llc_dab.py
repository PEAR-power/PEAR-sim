"""Gradient-engine gate: the LLC/DAB vertical slice.

Two resonant/isolated topologies, each exercising a distinct piece of the
engine that the buck family could not:

DAB — all-scheduled but TRANSFORMER-COUPLED: the declared winding resistance
is load-bearing (a lossless DAB conserves the magnetizing DC current, which
puts a unit eigenvalue in the monodromy and destroys the isolated fixed
point). With it, the all-scheduled affine path carries over unchanged and the
production shooting solver provides the oracle.

LLC — the transformer's DC is blocked by the series Cr instead, but the
series pair Lr + Lm creates a KERNEL STAR INVARIANT (i_lres - i_lprim is
conserved by every mode's flow), which is a STRUCTURAL unit eigenvalue: no
resistor can remove it and none should. The engine solves on the invariant
subspace (the invariant's value pinned from the declared initial state).

HISTORY OF THE LLC ORACLE. When the vertical slice was first added, the
production shooting solver could not run this model at all — its FD probes
perturbed states independently and violated the star KCL — so the d/d(lr)
reference was a 14 ms settled transient, whose grid-sampled rms carried a
~1e-3 bias (measured errs 7.34e-6 mean / 3.01e-3 rms, rms gated at 1e-2
with the bias documented). Teaching the shooting solver the same
invariant-subspace treatment gave a better oracle: both errors collapsed to the
noise floor (4.4e-8 / 6.5e-8 measured 2026-07-28), confirming the wider rms
gate had been pure reference bias. The gate below uses the shooting oracle
and the standard T2 criterion for both metrics.

Measured before the gates were declared (2026-07-27, oracle upgrade
2026-07-28):
  DAB   T1 vs shooting 1.43e-9; d/d(phi), d/d(load) err <= 3.6e-8 of the T2
        criterion. d/d(vin) (measured 1.7e-8/6.4e-9) joined the gate when the
        constant-guard livelock was fixed and the shooting oracle survived
        the perturbed operating point — before that the column was refused,
        a gate without an oracle being decoration.
  LLC   engine converges in 2 Newton iterations; T1 vs the invariant-aware
        shooting solver 1.08e-9; d/d(lr) errs 4.4e-8 (mean) / 6.5e-8 (rms)
        of the T2 criterion against the shooting oracle.
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
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
PERIOD_S = 1.0 / 20e3
GATE = 1e-3  # the T2 criterion, unchanged since the buck gate

DAB_METRICS = (MetricSpec("v(outp)", "mean"), MetricSpec("i(lprim)", "rms"))
LLC_METRICS = (MetricSpec("v(op)", "mean"), MetricSpec("i(lres)", "rms"))


@pytest.fixture(scope="module")
def dab_result():
    run = HybridSimulator(load_model(EXAMPLES / "dab_phase_shift.yaml")).simulate()
    sequence = extract_mode_sequence(run, period_s=PERIOD_S)
    vector = DesignVector.from_model(
        EXAMPLES / "dab_phase_shift.yaml", ("phi", "load", "vin")
    )
    return vector, solve(vector, sequence, DAB_METRICS, jacobian=True)


@pytest.fixture(scope="module")
def llc_run():
    return HybridSimulator(load_model(EXAMPLES / "llc_resonant.yaml")).simulate()


@pytest.fixture(scope="module")
def llc_result(llc_run):
    sequence = extract_mode_sequence(llc_run, period_s=PERIOD_S)
    vector = DesignVector.from_model(EXAMPLES / "llc_resonant.yaml", ("lr",))
    return sequence, vector, solve(vector, sequence, LLC_METRICS, jacobian=True)


# ---------------------------------------------------------------------------
# DAB: transformer-coupled, all-scheduled — shooting is the oracle
# ---------------------------------------------------------------------------


def test_dab_steady_state_matches_the_shooting_solver(dab_result) -> None:
    """T1 on the coupled-winding state. Measured 1.43e-9; the declared
    primary resistance is what makes this fixed point exist at all."""

    _, result = dab_result
    reference = solve_periodic_steady_state(load_model(EXAMPLES / "dab_phase_shift.yaml"))
    ours = np.asarray(result.steady_state.state)
    theirs = np.asarray(reference.state)
    deviation = float(np.max(np.abs(ours - theirs) / (1.0 + np.abs(theirs))))
    assert deviation <= 1e-7, deviation
    assert result.steady_state.converged
    # All-scheduled: the affine seed IS the fixed point.
    assert result.steady_state.iterations == 1
    assert result.metrics["mean(v(outp))"] == pytest.approx(271.59, rel=1e-3)


def test_dab_phase_and_load_derivatives_match_the_shooting_oracle(dab_result) -> None:
    """Six derivatives, measured <= 3.6e-8 of the T2 criterion. d/d(phi) is
    the one the leg-swap construction exists for: the phase shift is a single
    named parameter, so this is the DAB power-flow derivative.

    vin was originally excluded: the shooting oracle livelocked on the
    bridge-parallel diode guards at the perturbed operating point — their
    guard expressions sit on an identically-zero special case that finite
    perturbation shaves off to elimination dust. The kernel now holds
    a guard with no state and no source dependence as the constant it is,
    the oracle survives both sides of the old knife (regression in
    test_constant_guard_livelock.py), and the column earned its gate back.
    """

    vector, result = dab_result
    jacobian = result.jacobian.as_array()
    theta = vector.as_array()

    def oracle(v: DesignVector) -> dict[str, float]:
        model = v.load()
        steady = solve_periodic_steady_state(model)
        run = HybridSimulator(one_cycle_model(model, np.asarray(steady.state))).simulate()
        computed = run.metrics()
        return {
            spec.name: float(computed[spec.output][spec.statistic])
            for spec in DAB_METRICS
        }

    for j, name in enumerate(("phi", "load", "vin")):
        step = 1e-4 * abs(theta[j])
        plus = oracle(vector.with_offset(name, +step))
        minus = oracle(vector.with_offset(name, -step))
        for i, spec in enumerate(DAB_METRICS):
            reference = (plus[spec.name] - minus[spec.name]) / (2.0 * step)
            characteristic = abs(result.metrics[spec.name]) / max(abs(theta[j]), 1e-30)
            error = abs(jacobian[i, j] - reference) / (
                abs(reference) + GATE * characteristic
            )
            assert error <= GATE, (name, spec.name, jacobian[i, j], reference, error)


# ---------------------------------------------------------------------------
# LLC: structural star invariant — both engines now solve on the subspace
# ---------------------------------------------------------------------------


def test_llc_engine_converges_on_the_invariant_subspace(llc_result) -> None:
    """The monodromy has a STRUCTURAL unit eigenvalue (i_lres - i_lprim is
    conserved by every mode); the engine pins the invariant to its declared
    initial value and Newton-converges in 2 iterations on the subspace."""

    sequence, _, result = llc_result
    assert result.steady_state.converged
    assert result.steady_state.iterations <= 3
    assert result.flag("declared_sequence_held").held
    assert result.metrics["mean(v(op))"] == pytest.approx(153.14, rel=1e-3)
    # The natural rectifier commutation is SOLVED, and lands on the duration
    # the transient's own event log recorded (2.4803 us into each half).
    solved = result.steady_state.segment_durations_s[0]
    recorded = sequence.steps[0].nominal_duration_s
    assert solved == pytest.approx(recorded, rel=1e-3)
    total = sum(result.steady_state.segment_durations_s)
    assert total == pytest.approx(PERIOD_S, rel=1e-12)


def test_llc_steady_state_matches_the_invariant_aware_shooting_solver(
    llc_result,
) -> None:
    """T1 for the LLC — possible at all only because the shooting solver
    now probes and corrects inside the star-invariant subspace. Measured
    1.08e-9; the two engines share only the kernel, not the fixed-point
    machinery, so the agreement is a genuine cross-check."""

    _, _, result = llc_result
    reference = solve_periodic_steady_state(load_model(EXAMPLES / "llc_resonant.yaml"))
    assert reference.converged
    # The pinning contract, mirrored from the gradient engine: the invariant
    # is confined to the DECLARED initial level set and says so.
    (invariant,) = reference.invariants
    assert invariant["node"] == "pri"
    assert invariant["pinned_value_a"] == pytest.approx(0.0, abs=1e-12)
    ours = np.asarray(result.steady_state.state)
    theirs = np.asarray(reference.state)
    deviation = float(np.max(np.abs(ours - theirs) / (1.0 + np.abs(theirs))))
    assert deviation <= 1e-7, deviation


def test_llc_resonant_inductor_derivative_matches_the_shooting_oracle(
    llc_result,
) -> None:
    """d/d(lr) — the saltation-bearing parameter: lr moves the tank frequency
    and therefore every natural exit instant.

    Before invariant-aware shooting the reference was a settled 14 ms transient whose grid-sampled
    rms carried a ~1e-3 bias (rms gated at 1e-2 with the bias documented).
    With the invariant-aware shooting oracle both errors sit at the noise
    floor — measured 4.4e-8 (mean) / 6.5e-8 (rms) — so both metrics now hold
    the standard T2 criterion, and the old wider gate is confirmed to have
    been reference bias, not engine error.
    """

    _, vector, result = llc_result
    jacobian = result.jacobian.as_array()
    theta = vector.as_array()

    def oracle(v: DesignVector) -> dict[str, float]:
        model = v.load()
        steady = solve_periodic_steady_state(model)
        assert steady.converged
        run = HybridSimulator(one_cycle_model(model, np.asarray(steady.state))).simulate()
        computed = run.metrics()
        return {
            spec.name: float(computed[spec.output][spec.statistic])
            for spec in LLC_METRICS
        }

    step = 1e-4 * abs(theta[0])
    plus = oracle(vector.with_offset("lr", +step))
    minus = oracle(vector.with_offset("lr", -step))
    for i, spec in enumerate(LLC_METRICS):
        reference = (plus[spec.name] - minus[spec.name]) / (2.0 * step)
        characteristic = abs(result.metrics[spec.name]) / max(abs(theta[0]), 1e-30)
        error = abs(jacobian[i, 0] - reference) / (
            abs(reference) + GATE * characteristic
        )
        assert error <= GATE, (spec.name, jacobian[i, 0], reference, error)
