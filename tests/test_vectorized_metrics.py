# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""The batched metric code reproduces the former per-node loops bit for bit.

On 2026-11-17 the dense-output quadrature and the state-extremum search were
rewritten as array operations (grid-following example: 147.6 s -> 48.7 s for a
full-evidence run). Nothing about WHAT is computed changed, so nothing about
the numbers may change either: every test here compares against the old loops,
kept verbatim in tests/reference/dense_metrics_reference.py, with exact
equality (repr of every float), not a tolerance.

A bit-identity test cannot tell whether both implementations miss the same
extremum, so the extremum search is also checked against an independent
method — brute-force dense sampling — on a step whose minimum lies between the
probe points, and a search that drops the derivative roots must fail it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pear_sim.core import metrics, solver
from pear_sim.core.solver import HybridSimulator
from pear_sim.spec.model import load_model
from tests.reference.dense_metrics_reference import (
    reference_dense_state_extrema,
    reference_integrate_dense_interval,
)

ROOT = Path(__file__).resolve().parents[1]


def _fake_mode(state_count: int):
    components = [SimpleNamespace(id=f"x{index}") for index in range(state_count)]
    return SimpleNamespace(
        circuit=SimpleNamespace(
            state_components=components,
            state_index={component.id: index for index, component in enumerate(components)},
        )
    )


@pytest.mark.parametrize("method", ["RK45", "RK23", "DOP853", "Radau", "BDF"])
def test_batched_extrema_equal_the_per_step_search_bit_for_bit(method: str) -> None:
    rng = np.random.default_rng(11)
    compared = 0
    for trial in range(12):
        size = int(rng.integers(1, 5))
        matrix = rng.normal(size=(size, size))
        matrix -= np.eye(size) * (np.abs(np.linalg.eigvals(matrix)).max() + 0.5)
        drive = rng.normal(size=size) * 5.0
        initial = rng.normal(size=size)
        if trial % 4 == 0:
            # A state frozen at zero: derivative polynomials trim to nothing.
            matrix[0, :] = 0.0
            initial[0] = 0.0
            drive[0] = 0.0

        def rhs(t, y, matrix=matrix, drive=drive):
            return matrix @ y + 3.0 * np.sin(drive * t)

        solution = solve_ivp(
            rhs, (0.0, 4.0), initial, method=method, dense_output=True, rtol=1e-6
        )
        mode = _fake_mode(size)
        for window in ((0.0, 4.0), (0.37, 2.9), (1.0, 1.0 + 1e-7)):
            expected = reference_dense_state_extrema(solution, mode, *window)
            observed = metrics._dense_state_extrema(solution, mode, *window)
            assert list(observed) == list(expected)
            assert repr(observed) == repr(expected)
            compared += 1
    assert compared == 36


EXAMPLES = (
    "buck.yaml",
    "buck_with_loss_cards.yaml",
    "rl_sat_inductor.yaml",
    "transformer_rl.yaml",
    "thyristor_phase_controlled.yaml",
    "halfwave_rectifier.yaml",
    "inverter_three_phase_six_step.yaml",
    "synchronous_buck_complementary.yaml",
)


def _observables(result) -> dict[str, str]:
    return {
        "time_integrals": repr(result.time_integrals),
        "physics_audit": repr(result.physics_audit),
        "state_extrema": repr(result.state_extrema),
        "mode_intervals": repr(result.mode_intervals),
    }


def _first_difference(left: str, right: str) -> str:
    index = next(
        (i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right))
    )
    return f"at char {index}: {left[max(0, index - 60):index + 40]!r} != {right[max(0, index - 60):index + 40]!r}"


@pytest.mark.parametrize("evidence_mode", ["full", "lean"])
@pytest.mark.parametrize("example", EXAMPLES)
def test_whole_runs_are_bit_identical_to_the_per_node_loops(
    example: str, evidence_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = load_model(ROOT / "examples" / example)
    batched = HybridSimulator(model, evidence_mode=evidence_mode).simulate()
    monkeypatch.setattr(metrics, "_integrate_dense_interval", reference_integrate_dense_interval)
    monkeypatch.setattr(solver, "_dense_state_extrema", reference_dense_state_extrema)
    looped = HybridSimulator(
        load_model(ROOT / "examples" / example), evidence_mode=evidence_mode
    ).simulate()
    expected, observed = _observables(looped), _observables(batched)
    for field, value in expected.items():
        # Explicit message: pytest's own diff of these multi-megabyte reprs
        # takes minutes when it fails.
        if observed[field] != value:
            pytest.fail(f"{field} differs " + _first_difference(observed[field], value))


class RadauDenseOutput:  # the class name selects the polynomial reconstruction
    def __init__(self, t_old: float, t: float, y_old: np.ndarray, q: np.ndarray):
        self.t_old, self.t, self.y_old, self.Q = t_old, t, y_old, q

    def __call__(self, t):
        x = (np.atleast_1d(t) - self.t_old) / (self.t - self.t_old)
        powers = np.vstack([x ** (k + 1) for k in range(self.Q.shape[1])])
        return self.y_old[:, np.newaxis] + self.Q @ powers


def _one_step_solution(y_old: float, q: list[float]):
    interpolant = RadauDenseOutput(0.0, 1.0, np.array([y_old]), np.array([q]))
    return SimpleNamespace(
        t=np.array([0.0, 1.0]), sol=SimpleNamespace(interpolants=[interpolant])
    ), interpolant


def test_extrema_agree_with_dense_sampling_and_a_missing_root_is_caught(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # y(x) = (x - 0.53)^2 - 0.1*x^3: an interior minimum between probe points
    # 0.5 and 0.5625, so the 17 probes alone miss it.
    centre = 0.53
    solution, interpolant = _one_step_solution(centre**2, [-2.0 * centre, 1.0, -0.1])
    mode = _fake_mode(1)
    sampled = interpolant(np.linspace(0.0, 1.0, 100_001))[0]
    found = metrics._dense_state_extrema(solution, mode, 0.0, 1.0)["x0"]
    assert found["minimum"] <= sampled.min() + 1e-15
    assert found["maximum"] >= sampled.max() - 1e-15

    monkeypatch.setattr(
        metrics,
        "_batched_derivative_roots",
        lambda coefficients: np.full(
            (coefficients.shape[0], max(coefficients.shape[1] - 2, 0)), np.nan + 0j
        ),
    )
    probes_only = metrics._dense_state_extrema(solution, mode, 0.0, 1.0)["x0"]
    assert probes_only["minimum"] - sampled.min() > 1e-5
