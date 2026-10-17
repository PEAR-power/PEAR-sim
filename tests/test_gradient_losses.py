"""Gradient-engine gate: device losses inside the Jacobian.

Acceptance criteria, asserted literally:
step-1 results are untouched by the loss machinery, dP_loss/dtheta agrees with
finite differences on the oracle at the T2 tolerance for theta INSIDE the
tables, a theta pushed outside the axes produces a FLAG (never a pretty
gradient), smoothing is engine-only and declared, and the Jacobian carries the
"no d/dT_j term" label.

One measured subtlety governs the duty reference and is worth its comment: the
oracle's loss artifact anchors switching-event samples to the OUTPUT GRID (a
declared convention of losses_thermal), so as a duty perturbation slides the
gate edge across a grid node the artifact moves in a small stair. Its
fine-step duty-derivative therefore does not exist; the honest reference steps
duty so the edge moves by WHOLE grid intervals (h = output_step / period),
where the stair contributions cancel identically. Measured: engine 84.605 vs
oracle 84.613 W per unit duty (9e-5 relative) at that step, while a naive
h = 4e-5 lands on a stair riser and reads 159.7.
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
from pear_sim.analysis.gradient.losses import LossGradientError, NO_THERMAL_TERM_LABEL
from pear_sim.analysis.losses_thermal import build_losses_thermal_observation
from pear_sim.analysis.steady_state import one_cycle_model, solve_periodic_steady_state
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
MODEL = EXAMPLES / "buck_with_loss_cards.yaml"
PERIOD_S = 1.0 / 20e3
OUTPUT_STEP_S = 1e-6

# Engine loss vs the oracle's one-settled-cycle artifact; measured 1.4e-4
# relative (quadrature on the closed form vs trapezoid on the 1 us grid).
VALUE_RELATIVE_TOLERANCE = 5e-4
GATE = 1e-3  # the same T2 gate as the buck gate


@pytest.fixture(scope="module")
def sequence():
    run = HybridSimulator(load_model(MODEL)).simulate()
    return extract_mode_sequence(run, period_s=PERIOD_S)


@pytest.fixture(scope="module")
def result(sequence):
    vector = DesignVector.from_model(MODEL, ("duty", "load", "fsw"))
    return vector, solve(
        vector,
        sequence,
        (MetricSpec("v(out)", "mean"),),
        jacobian=True,
        loss_metrics=True,
    )


def _oracle_loss(vector: DesignVector) -> float:
    model = vector.load()
    steady = solve_periodic_steady_state(model)
    assert steady.converged
    cycle = one_cycle_model(model, np.asarray(steady.state))
    run = HybridSimulator(cycle).simulate()
    return float(build_losses_thermal_observation(run)["total_mean_power_w"])


def test_loss_value_matches_the_oracle_artifact(result) -> None:
    vector, solved = result
    reference = _oracle_loss(vector)
    ours = solved.metrics["P_loss_total_w"]
    assert abs(ours - reference) <= VALUE_RELATIVE_TOLERANCE * reference
    # Per-device split adds up to the total it came from.
    split = solved.metrics["P_loss(q1)_w"] + solved.metrics["P_loss(d1)_w"]
    assert split == pytest.approx(ours, rel=1e-12)


def test_step_one_metrics_are_untouched_by_the_loss_machinery(
    sequence, result
) -> None:
    """Acceptance: adding losses must not move the electrical answer."""

    vector, with_losses = result
    without = solve(
        vector, sequence, (MetricSpec("v(out)", "mean"),), jacobian=False
    )
    assert with_losses.metrics["mean(v(out))"] == without.metrics["mean(v(out))"]
    assert tuple(with_losses.steady_state.state) == tuple(without.steady_state.state)


def test_loss_derivatives_match_oracle_finite_differences(result) -> None:
    """dP_loss/dtheta at the T2 gate, theta inside the tables.

    duty uses the grid-commensurate step (see module docstring); load and fsw
    use the plain relative step, because neither slides an edge across the
    output grid (fsw moves the edges and the one-cycle grid TOGETHER).
    """

    vector, solved = result
    jacobian = solved.jacobian.as_array()
    row = solved.jacobian.metric_names.index("P_loss_total_w")
    theta = vector.as_array()
    total = solved.metrics["P_loss_total_w"]

    steps = {
        "duty": OUTPUT_STEP_S / PERIOD_S,  # edge moves exactly one grid cell
        "load": 1e-4 * theta[list(vector.names()).index("load")],
        "fsw": 1e-4 * theta[list(vector.names()).index("fsw")],
    }
    for name, step in steps.items():
        j = list(vector.names()).index(name)
        reference = (
            _oracle_loss(vector.with_offset(name, +step))
            - _oracle_loss(vector.with_offset(name, -step))
        ) / (2.0 * step)
        ours = jacobian[row, j]
        characteristic = abs(total) / max(abs(theta[j]), 1e-30)
        error = abs(ours - reference) / (abs(reference) + GATE * characteristic)
        assert error <= GATE, (name, ours, reference, error)


def test_theta_outside_the_tables_is_a_flag_not_a_plateau(sequence) -> None:
    """Acceptance: the assertion is that the flag EXISTS with the clamp
    facts — not that the gradient looks nice out there."""

    vector = DesignVector.from_model(MODEL, ("load",)).with_values(
        np.asarray([0.5])  # ~80 A peak against a 50 A table axis
    )
    solved = solve(vector, sequence, (), jacobian=False, loss_metrics=True)
    flag = solved.flag("loss_tables_interior[q1]")
    assert not flag.held
    assert flag.margin < 0.0, "signed distance beyond the edge"
    assert "clamped" in flag.detail
    assert "FLAT" in flag.detail


def test_interior_theta_reports_its_headroom(result) -> None:
    _, solved = result
    for device in ("q1", "d1"):
        flag = solved.flag(f"loss_tables_interior[{device}]")
        assert flag.held
        assert flag.margin > 0.0  # measured 9.4 A of headroom to the 50 A edge


def test_smoothing_is_declared_and_quantified(sequence) -> None:
    """Acceptance: the spline lives in the engine only, off by default,
    and the flag reports the difference against the unsmoothed evaluation."""

    vector = DesignVector.from_model(MODEL, ("duty",))
    plain = solve(vector, sequence, (), jacobian=False, loss_metrics=True)
    assert all(flag.name != "loss_tables_smoothed" for flag in plain.flags)

    smoothed = solve(
        vector,
        sequence,
        (),
        jacobian=False,
        loss_metrics=True,
        smooth_loss_tables=True,
    )
    flag = smoothed.flag("loss_tables_smoothed")
    assert flag.held
    assert "PCHIP" in flag.detail and "never in the oracle" in flag.detail
    # The difference is quantified, small, and honestly nonzero.
    assert 0.0 < flag.margin < 0.05 * plain.metrics["P_loss_total_w"]

    # Structural half of the engine-only rule: the ORACLE loss module knows nothing of splines.
    oracle_source = (
        ROOT / "src" / "pear_sim" / "analysis" / "losses_thermal.py"
    ).read_text(encoding="utf-8")
    assert "Pchip" not in oracle_source and "spline" not in oracle_source.lower()


def test_the_jacobian_carries_the_open_thermal_loop_label(result) -> None:
    _, solved = result
    assert NO_THERMAL_TERM_LABEL in solved.jacobian_labels
    assert "d/dT_j" in NO_THERMAL_TERM_LABEL


def test_loss_metrics_refuse_a_model_without_cards(sequence) -> None:
    vector = DesignVector.from_model(EXAMPLES / "buck.yaml", ("duty",))
    with pytest.raises(LossGradientError, match="loss_analysis"):
        solve(vector, sequence, (), loss_metrics=True)
