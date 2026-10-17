"""The hard gate of the gradient engine: T1 + T2 must be green.

Every declared tolerance below is a measured value with margin, and the
measurement is quoted next to it, so a future regression is visible as a
NUMBER, not as a feeling.

T1 — the engine's steady state agrees with the oracle, twice over:
     (a) the production shooting solver, (b) the closed-form buck oracle.
T2 — the engine's Jacobian agrees with central finite differences taken on
     the FULL oracle, extended to periodic convergence first.
T3 — a violated declaration returns as a flag with a negative margin, and
     no exception escapes.

T4 (dead time / saltation) is deliberately ABSENT here: in this all-scheduled
scope every edge is forced, the saltation matrix is I, and the test would pass
without proving anything. It lives in tests/test_gradient_saltation.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pear_sim.analysis.gradient import (
    DesignVector,
    GradientEngineError,
    MetricSpec,
    extract_mode_sequence,
    sequences_agree,
    solve,
)
from pear_sim.analysis.steady_state import one_cycle_model, solve_periodic_steady_state
from pear_sim.benchmarks.buck_boundary import BuckBoundaryParameters, solve_periodic_buck
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
PERIOD_S = 1.0 / 20e3
THETA = ("l", "c", "fsw", "duty", "load")
METRICS = (
    MetricSpec("v(out)", "mean"),
    MetricSpec("v(out)", "rms"),
    MetricSpec("i(l1)", "mean"),
)

# T1(a): bounded by the oracle run's rtol=1e-8/atol=1e-10; measured 2.2e-10.
T1A_STATE_TOLERANCE = 1e-7
# T1(b): two closed forms against each other; measured 0.0 (mean) and
# 3.7e-16 (ripple), so the assessment's 1e-9 gate is kept as written.
T1B_RELATIVE_TOLERANCE = 1e-9
# T2: 1e-3 per the assessment — two orders above the central-difference noise
# floor of an rtol=1e-8 oracle, still catching a sign, a scale, or a missing
# term. epsilon_abs implements the assessment's "scale of the metric": the
# characteristic derivative magnitude |metric|/|theta|, taken at one part in
# a thousand. A derivative smaller than that is numerically indistinguishable
# from zero on BOTH sides of the comparison (measured: the c/mean(i(l1))
# pair is ~3e-5 against a characteristic 4e4), and comparing two noise
# floors tightly would test the noise, not the Jacobian.
T2_RELATIVE_TOLERANCE = 1e-3
T2_ORACLE_FD_RELATIVE_STEP = 1e-4


@pytest.fixture(scope="module")
def buck_sequence():
    model = load_model(EXAMPLES / "buck.yaml")
    run = HybridSimulator(model).simulate()
    return extract_mode_sequence(run, period_s=PERIOD_S)


@pytest.fixture(scope="module")
def buck_result(buck_sequence):
    vector = DesignVector.from_model(EXAMPLES / "buck.yaml", THETA)
    return vector, solve(vector, buck_sequence, METRICS, jacobian=True)


# ---------------------------------------------------------------------------
# T1 — steady state agrees with the oracle
# ---------------------------------------------------------------------------


def test_t1a_steady_state_matches_the_shooting_solver(buck_result) -> None:
    _, result = buck_result
    reference = solve_periodic_steady_state(load_model(EXAMPLES / "buck.yaml"))
    ours = np.asarray(result.steady_state.state)
    theirs = np.asarray(reference.state)
    assert tuple(result.steady_state.state_names) == tuple(reference.state_names)
    deviation = float(np.max(np.abs(ours - theirs) / (1.0 + np.abs(theirs))))
    assert deviation <= T1A_STATE_TOLERANCE, deviation
    assert result.steady_state.converged
    assert result.steady_state.residual_inf < 1e-12
    assert result.flag("declared_sequence_held").held


def test_t1b_mean_and_ripple_match_the_closed_form_oracle(buck_result) -> None:
    _, result = buck_result
    parameters = BuckBoundaryParameters(
        vin_v=100.0,
        inductance_h=1e-3,
        capacitance_f=100e-6,
        switching_frequency_hz=20e3,
        duty=0.4,
    )
    point = solve_periodic_buck(parameters, 10.0)

    mean_engine = result.metrics["mean(v(out))"]
    assert abs(mean_engine - point.mean_output_voltage_v) <= (
        T1B_RELATIVE_TOLERANCE * abs(point.mean_output_voltage_v)
    )

    # Ripple from the segment-boundary states: in CCM the inductor current is
    # monotone within each segment, so its extrema sit exactly on the
    # boundaries the engine already carries.
    index = result.steady_state.state_names.index("l1")
    currents = [state[index] for state in result.steady_state.segment_boundary_states]
    ripple_engine = max(currents) - min(currents)
    ripple_oracle = (
        point.switch_off_state.inductor_current_a
        - point.start_state.inductor_current_a
    )
    assert abs(ripple_engine - ripple_oracle) <= (
        T1B_RELATIVE_TOLERANCE * abs(ripple_oracle)
    )


# ---------------------------------------------------------------------------
# T2 — the Jacobian agrees with finite differences on the oracle
# ---------------------------------------------------------------------------


def _oracle_metrics(vector: DesignVector) -> dict[str, float]:
    """Metric evaluation on the FULL oracle at periodic convergence.

    The model is first driven to its periodic steady state by the production
    shooting solver, then simulated for exactly one settled cycle; comparing
    at a shared truncated window instead would measure the transient, not the
    derivative.
    """

    model = vector.load()
    steady = solve_periodic_steady_state(model)
    assert steady.converged
    cycle = one_cycle_model(model, np.asarray(steady.state))
    run = HybridSimulator(cycle).simulate()
    computed = run.metrics()
    return {
        spec.name: float(computed[spec.output][spec.statistic]) for spec in METRICS
    }


def test_t2_jacobian_matches_oracle_finite_differences(buck_result) -> None:
    vector, result = buck_result
    jacobian = result.jacobian.as_array()
    theta = vector.as_array()

    for j, name in enumerate(THETA):
        step = T2_ORACLE_FD_RELATIVE_STEP * abs(theta[j])
        plus = _oracle_metrics(vector.with_offset(name, +step))
        minus = _oracle_metrics(vector.with_offset(name, -step))
        for i, spec in enumerate(METRICS):
            reference = (plus[spec.name] - minus[spec.name]) / (2.0 * step)
            ours = jacobian[i, j]
            characteristic = abs(result.metrics[spec.name]) / max(
                abs(theta[j]), 1e-30
            )
            epsilon_abs = T2_RELATIVE_TOLERANCE * characteristic
            error = abs(ours - reference) / (abs(reference) + epsilon_abs)
            assert error <= T2_RELATIVE_TOLERANCE, (
                name,
                spec.name,
                ours,
                reference,
                error,
            )


def test_t2_headline_derivatives_have_their_textbook_values(buck_result) -> None:
    """Spot checks with closed-form expectations, so a coincidental FD
    agreement cannot hide a systematically wrong engine: d mean(v_out)/d duty
    is Vin, and d mean(i_l)/d load is -mean(v_out)/R^2."""

    _, result = buck_result
    jacobian = result.jacobian.as_array()
    names = result.jacobian.metric_names
    params = result.jacobian.parameter_names
    d_mean_d_duty = jacobian[names.index("mean(v(out))"), params.index("duty")]
    assert d_mean_d_duty == pytest.approx(100.0, rel=1e-6)
    d_il_d_load = jacobian[names.index("mean(i(l1))"), params.index("load")]
    assert d_il_d_load == pytest.approx(-40.0 / 100.0, rel=1e-6)


# ---------------------------------------------------------------------------
# T3 — a violated declaration is a result, not an exception
# ---------------------------------------------------------------------------


def test_t3_violated_declaration_returns_a_flag_not_an_exception(
    buck_sequence,
) -> None:
    """buck_dcm under a declared CCM sequence: the freewheeling diode is asked
    to conduct while the steady cycle drives its current negative. The engine
    must report that with a negative margin and MUST NOT raise; the oracle
    post-hoc shows the same violation as a sequence disagreement."""

    vector = DesignVector.from_model(EXAMPLES / "buck_dcm.yaml", ("duty", "load"))
    result = solve(vector, buck_sequence, METRICS, jacobian=False)

    flag = result.flag("declared_sequence_held")
    assert not flag.held
    assert flag.margin < 0.0
    assert "conducting" in flag.detail

    # The metrics still evaluate — under the DECLARED (wrong) physics. What
    # protects the caller is the flag, not a censored number.
    assert np.isfinite(result.metrics["mean(v(out))"])

    # Oracle post-hoc: the run's own sequence disagrees with the declaration.
    dcm_run = HybridSimulator(load_model(EXAMPLES / "buck_dcm.yaml")).simulate()
    observed = extract_mode_sequence(dcm_run, period_s=PERIOD_S)
    agree, reasons = sequences_agree(buck_sequence, observed, duration_rtol=1e-6)
    assert not agree and reasons


def test_a_schedule_shape_mismatch_is_flagged_without_metrics(tmp_path) -> None:
    """If theta drives the control law to a DIFFERENT step structure, metrics
    under the stale declaration would be fiction — so they come back NaN with
    the flag, still without an exception."""

    body = (EXAMPLES / "buck.yaml").read_text(encoding="utf-8").replace(
        "  duty: 40 %", "  duty: 0 %"
    )
    path = tmp_path / "duty_zero.yaml"
    path.write_text(body, encoding="utf-8")
    model = load_model(EXAMPLES / "buck.yaml")
    run = HybridSimulator(model).simulate()
    declared = extract_mode_sequence(run, period_s=PERIOD_S)

    vector = DesignVector.from_model(path, ("duty",))
    result = solve(vector, declared, METRICS, jacobian=False)
    flag = result.flag("declared_sequence_held")
    assert not flag.held
    assert np.isnan(result.metrics["mean(v(out))"])
    assert result.jacobian is None


# ---------------------------------------------------------------------------
# Scope refusals and determinism
# ---------------------------------------------------------------------------


def test_the_synchronous_buck_three_mode_cycle_is_in_scope() -> None:
    model = load_model(EXAMPLES / "synchronous_buck_complementary.yaml")
    run = HybridSimulator(model).simulate()
    sequence = extract_mode_sequence(run, period_s=PERIOD_S)
    vector = DesignVector.from_model(
        EXAMPLES / "synchronous_buck_complementary.yaml",
        ("duty", "dead_time", "load"),
    )
    result = solve(vector, sequence, (MetricSpec("v(out)", "mean"),), jacobian=True)
    reference = solve_periodic_steady_state(model)
    deviation = float(
        np.max(
            np.abs(np.asarray(result.steady_state.state) - np.asarray(reference.state))
            / (1.0 + np.abs(np.asarray(reference.state)))
        )
    )
    assert deviation <= T1A_STATE_TOLERANCE
    # In all-scheduled scope the dead-time edge is FORCED, so this derivative is the
    # scheduled-duty loss -fsw*Vin — checking it here is legitimate; what may
    # NOT be checked without saltation is the same derivative under natural
    # commutation (T4).
    jacobian = result.jacobian.as_array()
    dead_index = result.jacobian.parameter_names.index("dead_time")
    assert jacobian[0, dead_index] == pytest.approx(-2.0e6, rel=1e-6)


def test_natural_guard_steps_are_in_scope_since_step_three() -> None:
    """Saltation support moved the scope boundary: natural commutation now SOLVES (the
    saltation gates live in tests/test_gradient_saltation.py); what stays
    refused is the saturation machinery."""

    run = HybridSimulator(load_model(EXAMPLES / "buck_dcm.yaml")).simulate()
    dcm_sequence = extract_mode_sequence(run, period_s=PERIOD_S)
    vector = DesignVector.from_model(EXAMPLES / "buck_dcm.yaml", ("duty",))
    result = solve(vector, dcm_sequence, METRICS, jacobian=False)
    assert result.steady_state.converged
    assert result.flag("declared_sequence_held").held


def test_waveform_sources_are_refused_in_v1(buck_sequence) -> None:
    vector = DesignVector.from_model(
        EXAMPLES / "thyristor_phase_controlled.yaml", ("rload",)
    )
    with pytest.raises(GradientEngineError, match="DC sources"):
        solve(vector, buck_sequence, (MetricSpec("i(ll)", "mean"),), jacobian=False)


def test_non_integral_statistics_are_refused(buck_sequence) -> None:
    vector = DesignVector.from_model(EXAMPLES / "buck.yaml", ("duty",))
    with pytest.raises(GradientEngineError, match="integral"):
        solve(vector, buck_sequence, (MetricSpec("v(out)", "ripple"),))


def test_the_engine_is_deterministic(buck_sequence) -> None:
    vector = DesignVector.from_model(EXAMPLES / "buck.yaml", THETA)
    first = solve(vector, buck_sequence, METRICS, jacobian=True)
    second = solve(vector, buck_sequence, METRICS, jacobian=True)
    assert first.metrics == second.metrics
    assert first.jacobian.values == second.jacobian.values
    assert first.steady_state.state == second.steady_state.state
