"""Global sensitivity: estimators checked against closed forms, then a real circuit.

The estimator tests use functions whose indices are known on paper (a linear
function for Morris, the Ishigami function for Sobol), so they measure the
truth rather than the code's agreement with itself. The circuit test uses the
ideal buck law V_out = D * V_in: only the duty can move the mean output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pear_sim.evidence.sensitivity import (
    SensitivityError,
    morris_design,
    morris_indices,
    run_sensitivity,
    saltelli_design,
    sobol_indices,
)

ROOT = Path(__file__).resolve().parents[1]
LOSS_BUCK = ROOT / "examples" / "buck_with_loss_cards.yaml"

# Ishigami (a=7, b=0.1) on [-pi, pi]^3: analytic first-order and total indices.
ISHIGAMI_S1 = (0.3139, 0.4424, 0.0)
ISHIGAMI_ST = (0.5576, 0.4424, 0.2437)


def _ishigami(unit: np.ndarray) -> np.ndarray:
    x = -np.pi + 2.0 * np.pi * unit
    return np.sin(x[:, 0]) + 7.0 * np.sin(x[:, 1]) ** 2 + 0.1 * x[:, 2] ** 4 * np.sin(x[:, 0])


# --- designs ----------------------------------------------------------------


def test_morris_points_stay_on_the_grid_and_move_one_factor_per_step() -> None:
    points, steps = morris_design(12, 4, 6, seed=5)
    assert points.shape == (12 * 5, 4)
    levels = np.round(points * 5, 9)
    assert np.allclose(levels, np.round(levels)) and points.min() >= 0 and points.max() <= 1
    for index, trajectory in enumerate(steps):
        chunk = points[index * 5 : (index + 1) * 5]
        assert sorted(factor for factor, _ in trajectory) == [0, 1, 2, 3]
        for position, (factor, step) in enumerate(trajectory):
            moved = chunk[position + 1] - chunk[position]
            assert np.count_nonzero(moved) == 1 and moved[factor] == pytest.approx(step)


def test_saltelli_design_swaps_exactly_one_column_per_block() -> None:
    n, k = 16, 3
    design = saltelli_design(n, k, seed=2)
    a, b = design[:n], design[n : 2 * n]
    for column in range(k):
        block = design[(2 + column) * n : (3 + column) * n]
        assert np.array_equal(block[:, column], b[:, column])
        others = [c for c in range(k) if c != column]
        assert np.array_equal(block[:, others], a[:, others])


# --- estimators against closed forms -----------------------------------------


def test_morris_recovers_a_linear_function_exactly() -> None:
    points, steps = morris_design(15, 3, 4, seed=1)
    values = 3.0 * points[:, 0] - 2.0 * points[:, 1] + 0.5 * points[:, 2]
    indices = morris_indices(values, steps, 3, bootstrap=200, seed=1)
    assert [item["mu_star"] for item in indices] == pytest.approx([3.0, 2.0, 0.5])
    assert [item["mu"] for item in indices] == pytest.approx([3.0, -2.0, 0.5])
    assert [item["sigma"] for item in indices] == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_morris_sigma_flags_an_interaction() -> None:
    points, steps = morris_design(30, 2, 4, seed=3)
    values = points[:, 0] * points[:, 1]  # each effect depends on the other factor
    indices = morris_indices(values, steps, 2, bootstrap=200, seed=1)
    assert all(item["sigma"] > 0.2 for item in indices)


def test_morris_drops_trajectories_with_a_failed_run() -> None:
    points, steps = morris_design(5, 2, 4, seed=1)
    values = points[:, 0] + points[:, 1]
    values[4] = np.nan  # inside trajectory 1 (rows 3..5)
    indices = morris_indices(values, steps, 2, bootstrap=100, seed=1)
    assert indices[0]["trajectories_used"] == 4


def test_sobol_converges_to_the_ishigami_indices() -> None:
    n = 4096
    estimate = sobol_indices(_ishigami(saltelli_design(n, 3, seed=1)), n, 3, bootstrap=200, seed=1)
    first = [item["S1"] for item in estimate["factors"]]
    total = [item["ST"] for item in estimate["factors"]]
    assert first == pytest.approx(ISHIGAMI_S1, abs=0.02)
    assert total == pytest.approx(ISHIGAMI_ST, abs=0.02)


def test_sobol_intervals_cover_the_truth_at_a_small_sample() -> None:
    n = 256
    estimate = sobol_indices(_ishigami(saltelli_design(n, 3, seed=1)), n, 3, bootstrap=1000, seed=1)
    for item, s1, st in zip(estimate["factors"], ISHIGAMI_S1, ISHIGAMI_ST, strict=True):
        assert item["S1_ci95"][0] <= s1 <= item["S1_ci95"][1]
        assert item["ST_ci95"][0] <= st <= item["ST_ci95"][1]


def test_sobol_does_not_depend_on_an_output_offset() -> None:
    # A 40 V output varying by 2 V must give the same indices as the 2 V
    # variation alone; without centering, the first-order interval was [-3.6, 5.9].
    n = 128
    values = _ishigami(saltelli_design(n, 3, seed=4))
    plain = sobol_indices(values, n, 3, bootstrap=300, seed=1)
    shifted = sobol_indices(values + 1000.0, n, 3, bootstrap=300, seed=1)
    for a, b in zip(plain["factors"], shifted["factors"], strict=True):
        assert a["S1"] == pytest.approx(b["S1"], abs=1e-9)
        assert a["S1_ci95"] == pytest.approx(b["S1_ci95"], abs=1e-9)
        assert a["ST"] == pytest.approx(b["ST"], abs=1e-9)


def test_sobol_says_so_when_the_output_does_not_vary() -> None:
    n = 16
    estimate = sobol_indices(np.full(n * 4, 3.0), n, 2, bootstrap=100, seed=1)
    assert estimate["variance"] == 0.0
    assert all(item["S1"] is None for item in estimate["factors"])
    assert "does not vary" in estimate["note"]


# --- request validation ------------------------------------------------------


def _request(tmp_path: Path, **change) -> Path:
    payload = {
        "schema": "pear-sim/sensitivity-request/v0.1",
        "method": "morris",
        "parameters": [
            {"name": "duty", "from": "35 %", "to": "45 %"},
            {"name": "load", "from": "3 ohm", "to": "6 ohm"},
        ],
        "outputs": [{"signal": "v(out)", "metric": "mean_last_half"}],
        "trajectories": 4,
        "seed": 1,
    }
    payload.update(change)
    if payload["method"] == "sobol" and "trajectories" not in change:
        payload.pop("trajectories")
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload))
    return path


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"method": "fast"}, "method must be"),
        ({"base_samples": 64}, "belongs to the sobol method"),
        ({"method": "sobol", "trajectories": 4}, "belong to the morris method"),
        ({"method": "sobol", "base_samples": 60, "trajectories": None}, "belong to the morris"),
        ({"parameters": [{"name": "duty", "from": "35 %", "to": "45 %"}]}, "at least two"),
        ({"outputs": [{"signal": "v(out)", "metric": "mean", "max": 3}]}, "takes no bounds"),
        ({"levels": 5}, "levels must be even"),
        # 1024 * (3 + 2) = 5120 runs.
        ({"method": "sobol", "base_samples": 1024, "parameters": [
            {"name": n, "from": f, "to": t} for n, f, t in
            [("duty", "35 %", "45 %"), ("load", "3 ohm", "6 ohm"), ("fsw", "10 kHz", "20 kHz")]
        ]}, "above the 2000-run limit"),
    ],
)
def test_invalid_requests_are_refused_before_anything_runs(
    tmp_path: Path, change: dict, message: str
) -> None:
    with pytest.raises(SensitivityError, match=message):
        run_sensitivity(LOSS_BUCK, _request(tmp_path, **change), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_sobol_base_samples_must_be_a_power_of_two(tmp_path: Path) -> None:
    payload = json.loads(_request(tmp_path).read_text())
    payload.pop("trajectories")
    payload.update(method="sobol", base_samples=60)
    path = tmp_path / "sobol.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(SensitivityError, match="power of two"):
        run_sensitivity(LOSS_BUCK, path, tmp_path / "out")


# --- a real circuit against its closed form -------------------------------------


def test_buck_mean_output_moves_only_with_duty(tmp_path: Path) -> None:
    # Ideal buck in CCM: V_out = D * V_in with V_in = 100 V. Moving the duty
    # across its 10-point range moves the mean output by 10 V; load and
    # switching frequency move it only by what the run has not settled.
    payload = json.loads(_request(tmp_path).read_text())
    payload["parameters"].append({"name": "fsw", "from": "10 kHz", "to": "40 kHz", "scale": "log"})
    path = tmp_path / "buck.json"
    path.write_text(json.dumps(payload))
    artifacts = run_sensitivity(LOSS_BUCK, path, tmp_path / "out", jobs=4)
    assert artifacts.status == "pass" and artifacts.run_count == 16
    summary = json.loads(artifacts.summary.read_text())
    output = summary["outputs"][0]
    effect = {item["name"]: item["mu_star"] for item in output["parameters"]}
    assert effect["duty"] == pytest.approx(10.0, abs=0.01)
    assert effect["load"] < 1e-3 and effect["fsw"] < 1e-3
    assert output["ranking"][0] == "duty"
    assert "uniform over its range" in summary["claim"]
