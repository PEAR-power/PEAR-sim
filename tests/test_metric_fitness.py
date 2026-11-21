# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""A sampled extremum decides a bound the same way on every acceptance path.

The buck in ``examples/buck.yaml`` switches at 20 kHz and its inductor current
peaks at 4.6 A. Recorded once per PWM period the grid only ever sees the
valley, 3.4 A. ``i(l1) max <= 4.1 A`` is violated; no path may accept it, and
on the coarse grid no path may report a positive margin or a winner.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pear_sim.core.solver import HybridSimulator
from pear_sim.evidence.batch import run_batch
from pear_sim.evidence.evidence_io import evaluate_run
from pear_sim.evidence.expectations import evaluate_expectations
from pear_sim.evidence.metric_fitness import (
    MIN_SAMPLES_PER_SWITCHING_PERIOD,
    grid_unresolved_reason,
    sampled_bound_unfit_reason,
)
from pear_sim.evidence.perturbation_campaign import REQUEST_SCHEMA, run_campaign
from pear_sim.run.io import write_result
from pear_sim.spec.model import load_model

BUCK = Path(__file__).parents[1] / "examples" / "buck.yaml"
LIMIT_A = 4.1
PEAK_A = 4.6003
VALLEY_A = 3.3997


def _grid(samples: float | None, controls: int = 1) -> dict:
    return {
        "output_step_s": 1e-6,
        "switching_control_count": controls,
        "shortest_switching_period_s": None if samples is None else samples * 1e-6,
        "samples_per_shortest_switching_period": samples,
    }


COARSE, FINE = _grid(1.0), _grid(50.0)


# --- the rule ---------------------------------------------------------------


def test_a_resolved_grid_or_a_non_extremum_is_always_fit() -> None:
    assert sampled_bound_unfit_reason("max", FINE, 3.0, upper=4.1) is None
    assert sampled_bound_unfit_reason("mean_last_half", COARSE, 3.0, upper=4.1) is None
    assert sampled_bound_unfit_reason("final", None, 3.0, upper=4.1) is None


def test_the_floor_is_inclusive_and_survives_float_division() -> None:
    assert grid_unresolved_reason(_grid(1e-5 / 1e-6)) is None
    assert grid_unresolved_reason(_grid(float(MIN_SAMPLES_PER_SWITCHING_PERIOD))) is None
    assert "at least" in grid_unresolved_reason(_grid(9.99))


def test_a_model_without_switching_has_nothing_to_alias_against() -> None:
    assert grid_unresolved_reason(_grid(None, controls=0)) is None


@pytest.mark.parametrize(
    "grid", [None, {}, {"switching_control_count": 1}, _grid(None), _grid(0.0), "fine"]
)
def test_missing_or_malformed_grid_metadata_is_unresolved(grid) -> None:
    assert grid_unresolved_reason(grid) is not None
    assert sampled_bound_unfit_reason("max", grid, 3.0, upper=4.1) is not None


@pytest.mark.parametrize(
    ("metric", "value", "bounds", "decided"),
    [
        # A sampled maximum is a lower bound on the true one.
        ("max", 3.4, {"upper": 4.1}, False),  # below the limit: could still exceed it
        ("max", 4.6, {"upper": 4.1}, True),  # above the limit: so is the true value
        ("max", 4.6, {"lower": 4.1}, True),  # above the floor: so is the true value
        ("max", 3.4, {"lower": 4.1}, False),
        ("ripple_last_half", 1e-8, {"upper": 0.01}, False),
        ("max", 4.6, {"lower": 4.0, "upper": 5.0}, False),  # two-sided needs the value
        ("max", 5.5, {"lower": 4.0, "upper": 5.0}, True),
        # A sampled minimum is an upper bound on the true one.
        ("min", 3.4, {"lower": 3.0}, False),
        ("min", 2.9, {"lower": 3.0}, True),
        ("min", 3.4, {"upper": 3.5}, True),
        ("min", 3.6, {"upper": 3.5}, False),
        # No bound at all: is the number usable for ranking? Not on this grid.
        ("max", 3.4, {}, False),
    ],
)
def test_a_coarse_grid_decides_only_what_its_one_sided_error_cannot_undo(
    metric, value, bounds, decided
) -> None:
    reason = sampled_bound_unfit_reason(metric, COARSE, value, **bounds)
    assert (reason is None) is decided


# --- the producer -----------------------------------------------------------


def _buck(tmp_path: Path, step: str) -> Path:
    path = tmp_path / f"buck_{step.replace(' ', '')}.yaml"
    path.write_text(BUCK.read_text().replace("output_step: 1 us", f"output_step: {step}"))
    return path


@pytest.mark.parametrize(("step", "samples"), [("1 us", 50.0), ("50 us", 1.0)])
def test_metrics_state_the_grid_against_the_carrier(tmp_path, step, samples) -> None:
    metrics = HybridSimulator(load_model(_buck(tmp_path, step))).simulate().metrics()
    for record in metrics.values():
        grid = record["extrema_grid"]
        assert grid["switching_control_count"] == 1
        assert grid["shortest_switching_period_s"] == pytest.approx(50e-6)
        assert grid["samples_per_shortest_switching_period"] == pytest.approx(samples)


# --- the three acceptance paths ----------------------------------------------


def _evaluate(tmp_path: Path, model: Path) -> dict:
    run = tmp_path / "run"
    write_result(HybridSimulator(load_model(model)).simulate(), run)
    expectations = tmp_path / "expectations.json"
    expectations.write_text(
        json.dumps(
            {
                "schema": "pear-sim/expectations/v0.1",
                "purpose": "inductor peak current limit",
                "measurement_window": {"kind": "last_half", "start_s": 0.001, "end_s": 0.002},
                "criteria": [
                    {
                        "id": "ipk", "signal": "i(l1)", "metric": "max_last_half",
                        "unit": "A", "maximum": LIMIT_A, "blocking": True,
                    }
                ],
            }
        )
    )
    artifacts = evaluate_run(run, expectations, tmp_path / "evidence")
    evidence = json.loads(artifacts.evidence_json.read_text())
    (criterion,) = evidence["criteria"]
    return criterion


def _campaign(tmp_path: Path, model: Path) -> dict:
    request = tmp_path / "campaign.json"
    request.write_text(
        json.dumps(
            {
                "schema": REQUEST_SCHEMA,
                "label": "one point",
                "axes": [
                    {
                        "id": "vin", "kind": "parameter_values", "parameter": "vin",
                        "values": ["100 V"], "source": "assumption", "note": "nominal",
                    }
                ],
                "criteria": [
                    {
                        "id": "ipk", "signal": "i(l1)", "field": "max_last_half",
                        "op": "max", "limit": LIMIT_A, "unit": "A",
                        "source": "assumption", "note": "placeholder limit",
                    }
                ],
            }
        )
    )
    artifacts = run_campaign(model, request, tmp_path / "campaign", jobs=1)
    return json.loads(artifacts.summary.read_text())


def _batch(tmp_path: Path, model: Path) -> dict:
    request = tmp_path / "batch.json"
    request.write_text(
        json.dumps(
            {
                "schema": "pear-sim/batch-request/v0.1",
                "designs": {"method": "list", "rows": [{"vin": "100 V"}]},
                "objective": {"minimize": {"signal": "i(l1)", "metric": "mean_last_half"}},
                "constraints": [
                    {"signal": "i(l1)", "metric": "max_last_half", "max": LIMIT_A}
                ],
                "finalists": 1,
            }
        )
    )
    artifacts = run_batch(model, request, tmp_path / "batch", jobs=1)
    return json.loads(artifacts.summary.read_text())


def test_on_a_resolved_grid_every_path_reports_the_violation(tmp_path) -> None:
    model = _buck(tmp_path, "1 us")

    criterion = _evaluate(tmp_path, model)
    assert criterion["status"] == "fail"
    assert criterion["measured"]["value"] == pytest.approx(PEAK_A, abs=1e-3)

    campaign = _campaign(tmp_path, model)
    verdict = campaign["runs"][0]["criteria"]["ipk"]
    assert verdict["status"] == "fail"
    assert verdict["margin"] == pytest.approx(LIMIT_A - PEAK_A, abs=1e-3)
    assert campaign["overall_status"] == "fail"

    batch = _batch(tmp_path, model)
    (design,) = batch["designs"]
    assert design["violations"] == ["i(l1).max_last_half"]
    assert not design["feasible"]
    assert batch["feasible_count"] == 0 and batch["best_design"] is None


def test_on_an_aliased_grid_no_path_accepts_or_reports_a_margin(tmp_path) -> None:
    model = _buck(tmp_path, "50 us")

    criterion = _evaluate(tmp_path, model)
    assert criterion["status"] == "not_comparable"
    assert criterion["measured"] is None
    assert "1 sample(s) per shortest switching period" in criterion["reason"]

    campaign = _campaign(tmp_path, model)
    (run,) = campaign["runs"]
    verdict = run["criteria"]["ipk"]
    assert verdict["status"] == "not_comparable"
    assert "margin" not in verdict
    assert verdict["value"] == pytest.approx(VALLEY_A, abs=1e-3)  # what the grid saw
    assert run["acceptance_status"] == "not_comparable"
    assert campaign["acceptance_status"] != "pass"
    assert campaign["overall_status"] != "pass"

    batch = _batch(tmp_path, model)
    (design,) = batch["designs"]
    assert not design["feasible"] and design["rank"] is None
    assert design["violations"] == []
    assert [item["quantity"] for item in design["not_comparable"]] == ["i(l1).max_last_half"]
    assert batch["feasible_count"] == 0
    assert batch["not_comparable_count"] == 1
    assert batch["best_design"] is None
    assert batch["finalists"] == []
    assert batch["status"] == "no_feasible_design"


def test_a_batch_does_not_rank_by_an_aliased_extremum(tmp_path) -> None:
    request = tmp_path / "batch.json"
    request.write_text(
        json.dumps(
            {
                "schema": "pear-sim/batch-request/v0.1",
                "designs": {"method": "list", "rows": [{"vin": "100 V"}, {"vin": "90 V"}]},
                "objective": {"minimize": {"signal": "i(l1)", "metric": "max_last_half"}},
                "finalists": 1,
            }
        )
    )
    artifacts = run_batch(_buck(tmp_path, "50 us"), request, tmp_path / "batch", jobs=1)
    summary = json.loads(artifacts.summary.read_text())
    assert summary["best_design"] is None and summary["ranking"] == []
    assert summary["not_comparable_count"] == 2


def test_a_bundle_without_grid_metadata_keeps_means_and_loses_extrema() -> None:
    """A run written before ``extrema_grid`` existed: re-simulate to judge extrema."""

    record = {
        "unit": "A",
        "aggregation_method": "gauss_legendre_8_on_solve_ivp_dense_output",
        "extrema_method": "requested_output_grid_samples",
        "window_start_s": 0.001,
        "window_end_s": 0.002,
        "mean_last_half": 4.0,
        "max_last_half": 3.4,
    }
    expectations = {
        "schema": "pear-sim/expectations/v0.1",
        "purpose": "legacy bundle",
        "measurement_window": {"kind": "last_half", "start_s": 0.001, "end_s": 0.002},
        "criteria": [
            {"id": "mean", "signal": "i(l1)", "metric": "mean_last_half", "unit": "A",
             "target": 4.0, "absolute_tolerance": 0.1, "blocking": True},
            {"id": "peak", "signal": "i(l1)", "metric": "max_last_half", "unit": "A",
             "maximum": LIMIT_A, "blocking": True},
            {"id": "peak_low", "signal": "i(l1)", "metric": "max_last_half", "unit": "A",
             "maximum": 3.0, "blocking": True},
        ],
    }
    evidence = evaluate_expectations(expectations, {"i(l1)": record})
    statuses = {item["id"]: item["status"] for item in evidence["criteria"]}
    # 3.4 A sampled already exceeds 3.0 A, so that violation stands on any grid.
    assert statuses == {"mean": "pass", "peak": "not_comparable", "peak_low": "fail"}
