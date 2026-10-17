import copy
import json
from pathlib import Path

import pytest

from pear_sim.evidence import (
    EvidenceValidationError,
    evaluate_expectations,
    validate_expectations,
)
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator


ROOT = Path(__file__).parents[1]


def _window() -> dict[str, object]:
    return {"kind": "last_half", "start_s": 1.0, "end_s": 2.0}


def _metric_record(*, unit: str = "V", **values: float) -> dict[str, object]:
    return {
        "unit": unit,
        "aggregation_method": "gauss_legendre_8_on_solve_ivp_dense_output",
        "extrema_method": "requested_output_grid_samples",
        "ripple_method": "requested_output_grid_peak_to_peak",
        "continuity": "structurally_continuous_across_visited_modes",
        "window_start_s": 1.0,
        "window_end_s": 2.0,
        "full_window_start_s": 0.0,
        "full_window_end_s": 2.0,
        **values,
    }


def _criterion(identifier: str, **overrides: object) -> dict[str, object]:
    criterion: dict[str, object] = {
        "id": identifier,
        "signal": "v(out)",
        "metric": "mean_last_half",
        "unit": "V",
        "target": 40.0,
        "absolute_tolerance": 0.1,
        "blocking": True,
    }
    criterion.update(overrides)
    return criterion


def test_evaluator_emits_every_status_and_never_passes_unknown_or_missing_metrics() -> None:
    expectations = {
        "schema": "pear-sim/expectations/v0.1",
        "purpose": "Exercise fail-closed evidence statuses.",
        "measurement_window": _window(),
        "criteria": [
            _criterion("passes"),
            _criterion("fails", maximum=39.0, target=None, absolute_tolerance=None),
            _criterion(
                "not_applicable",
                applicability="not_applicable",
                reason="This case deliberately excludes the criterion.",
            ),
            _criterion("not_comparable", unit="A"),
            _criterion("missing_signal", signal="v(missing)"),
            _criterion("unknown_metric", metric="made_up_metric"),
        ],
        "human_checks": ["Inspect the schematic."],
    }
    # Remove keys replaced by the bound comparison; ``None`` is intentionally invalid.
    failing = expectations["criteria"][1]
    del failing["target"]
    del failing["absolute_tolerance"]
    metrics = {"v(out)": _metric_record(mean_last_half=40.05, made_up_metric=40.0)}

    evidence = evaluate_expectations(expectations, metrics)

    assert evidence["schema"] == "pear-sim/evidence/v0.1"
    assert evidence["claim"]["validation_claim"] is False
    assert evidence["overall_status"] == "fail"
    assert evidence["blocking_status"] == "fail"
    assert evidence["summary"]["counts"] == {
        "pass": 1,
        "fail": 1,
        "not_applicable": 1,
        "not_comparable": 1,
        "not_run": 2,
    }
    by_id = {record["id"]: record for record in evidence["criteria"]}
    assert by_id["passes"]["status"] == "pass"
    assert by_id["passes"]["delta"]["effective_tolerance"] == pytest.approx(0.1)
    assert by_id["fails"]["status"] == "fail"
    assert by_id["not_applicable"]["status"] == "not_applicable"
    assert by_id["not_comparable"]["status"] == "not_comparable"
    assert by_id["missing_signal"]["status"] == "not_run"
    assert by_id["unknown_metric"]["status"] == "not_run"
    assert "not supported" in by_id["unknown_metric"]["reason"]
    assert evidence["human_checks"] == ["Inspect the schematic."]


def test_existing_buck_expectations_evaluate_against_existing_metric_contract() -> None:
    expectations = json.loads(
        (ROOT / "examples" / "buck_expectations.json").read_text(encoding="utf-8")
    )
    metrics = HybridSimulator(load_model(ROOT / "examples" / "buck.yaml")).simulate().metrics()

    first = evaluate_expectations(expectations, metrics)
    second = evaluate_expectations(copy.deepcopy(expectations), copy.deepcopy(metrics))

    assert first == second
    assert first["expectations"]["schema"] == "pear-sim/expectations/v0.1"
    assert first["overall_status"] == "pass"
    assert first["blocking_status"] == "pass"
    assert [record["status"] for record in first["criteria"]] == ["pass", "pass", "pass"]


def test_legacy_review_expectations_schema_remains_explicitly_compatible() -> None:
    expectations = json.loads(
        (ROOT / "examples" / "buck_expectations.json").read_text(encoding="utf-8")
    )
    expectations["schema"] = "pear-sim/review-expectations/v0.1-draft"
    expectations["automation_status"] = "legacy hash-bound draft"
    metrics = HybridSimulator(load_model(ROOT / "examples" / "buck.yaml")).simulate().metrics()

    result = evaluate_expectations(expectations, metrics)

    assert result["expectations"]["schema"] == "pear-sim/review-expectations/v0.1-draft"
    assert result["blocking_status"] == "pass"


def test_relative_and_range_comparisons_are_inclusive() -> None:
    expectations = {
        "schema": "pear-sim/expectations/v0.1",
        "measurement_window": _window(),
        "criteria": [
            _criterion(
                "relative",
                target=100.0,
                absolute_tolerance=0.01,
                relative_tolerance=0.02,
            ),
            {
                "id": "range",
                "signal": "v(out)",
                "metric": "min_last_half",
                "unit": "V",
                "minimum": 39.0,
                "maximum": 41.0,
                "blocking": True,
            },
        ],
    }
    metrics = {"v(out)": _metric_record(mean_last_half=102.0, min_last_half=39.0)}

    result = evaluate_expectations(expectations, metrics)

    assert result["overall_status"] == "pass"
    assert result["criteria"][0]["delta"]["effective_tolerance"] == 2.0
    assert result["criteria"][1]["status"] == "pass"


def test_window_mismatch_and_nonfinite_value_are_not_comparable() -> None:
    expectations = {
        "schema": "pear-sim/expectations/v0.1",
        "measurement_window": _window(),
        "criteria": [_criterion("window"), _criterion("nonfinite", signal="v(other)")],
    }
    metrics = {
        "v(out)": {
            **_metric_record(mean_last_half=40.0),
            "window_start_s": 0.5,
        },
        "v(other)": _metric_record(mean_last_half=float("nan")),
    }

    result = evaluate_expectations(expectations, metrics)

    assert result["overall_status"] == "not_comparable"
    assert [record["status"] for record in result["criteria"]] == [
        "not_comparable",
        "not_comparable",
    ]


def test_overflowing_window_and_comparison_arithmetic_are_not_comparable() -> None:
    expectations = {
        "schema": "pear-sim/expectations/v0.1",
        "measurement_window": _window(),
        "criteria": [
            _criterion("window"),
            _criterion(
                "delta",
                signal="v(other)",
                target=1e308,
                absolute_tolerance=1e-3,
            ),
        ],
    }
    metrics = {
        "v(out)": {
            **_metric_record(mean_last_half=40.0),
            "window_start_s": 10**400,
        },
        "v(other)": _metric_record(mean_last_half=-1e308),
    }

    result = evaluate_expectations(expectations, metrics)

    assert [record["status"] for record in result["criteria"]] == [
        "not_comparable",
        "not_comparable",
    ]
    json.dumps(result, allow_nan=False)


def test_unknown_metric_producer_status_cannot_pass() -> None:
    expectations = {
        "schema": "pear-sim/expectations/v0.1",
        "criteria": [_criterion("producer_status")],
    }
    metrics = {
        "v(out)": {
            **_metric_record(mean_last_half=40.0),
            "status": "looks_good",
        }
    }

    result = evaluate_expectations(expectations, metrics)

    assert result["overall_status"] == "not_comparable"
    assert result["criteria"][0]["status"] == "not_comparable"


def test_producer_not_applicable_cannot_skip_an_applicable_blocking_criterion() -> None:
    expectations = {
        "schema": "pear-sim/expectations/v0.1",
        "criteria": [_criterion("passes"), _criterion("producer-na", signal="v(other)")],
    }
    metrics = {
        "v(out)": _metric_record(mean_last_half=40.0),
        "v(other)": {
            **_metric_record(mean_last_half=40.0),
            "status": "not_applicable",
            "reason": "Producer cannot supply this metric.",
        },
    }

    result = evaluate_expectations(expectations, metrics)

    assert result["criteria"][0]["status"] == "pass"
    assert result["criteria"][1]["status"] == "not_comparable"
    assert result["blocking_status"] == "not_comparable"


def test_missing_metrics_payload_is_not_run_not_pass() -> None:
    expectations = {
        "schema": "pear-sim/expectations/v0.1",
        "criteria": [_criterion("missing")],
    }

    result = evaluate_expectations(expectations, None)

    assert result["overall_status"] == "not_run"
    assert result["blocking_status"] == "not_run"
    assert result["criteria"][0]["status"] == "not_run"


@pytest.mark.parametrize(
    "mutation, match",
    [
        (lambda document: document.update({"typo": True}), "unknown field"),
        (
            lambda document: document["criteria"][0].update({"targte": 40.0}),
            "unknown field",
        ),
        (lambda document: document["criteria"].append(document["criteria"][0]), "duplicate"),
        (
            lambda document: document["criteria"][0].pop("absolute_tolerance"),
            "requires absolute_tolerance",
        ),
        (
            lambda document: document["measurement_window"].update({"end_s": 1.0}),
            "greater than start_s",
        ),
        (lambda document: document.update({"measurement_window": None}), "must be an object"),
        (lambda document: document.update({"schema": []}), "schema must be"),
        (
            lambda document: document["measurement_window"].update({"kind": []}),
            "measurement_window.kind must be",
        ),
        (
            lambda document: document["criteria"][0].update({"applicability": []}),
            "applicability must be",
        ),
        (
            lambda document: document["criteria"][0].update({"target": 10**400}),
            "finite number",
        ),
    ],
)
def test_expectation_validation_rejects_ambiguous_or_unknown_contract_fields(
    mutation: object, match: str
) -> None:
    document = {
        "schema": "pear-sim/expectations/v0.1",
        "measurement_window": _window(),
        "criteria": [_criterion("criterion")],
    }
    mutation(document)  # type: ignore[operator]

    with pytest.raises(EvidenceValidationError, match=match):
        validate_expectations(document)


def test_empty_criteria_and_only_nonblocking_criteria_remain_fail_closed() -> None:
    empty = evaluate_expectations({"schema": "pear-sim/expectations/v0.1", "criteria": []}, {})
    assert empty["overall_status"] == "not_run"
    assert empty["blocking_status"] == "not_run"

    nonblocking = {
        "schema": "pear-sim/expectations/v0.1",
        "criteria": [_criterion("advisory", blocking=False, maximum=39.0)],
    }
    del nonblocking["criteria"][0]["target"]
    del nonblocking["criteria"][0]["absolute_tolerance"]
    result = evaluate_expectations(
        nonblocking,
        {"v(out)": _metric_record(mean_last_half=40.0)},
    )
    assert result["overall_status"] == "fail"
    assert result["blocking_status"] == "not_run"

    pass_and_not_applicable = {
        "schema": "pear-sim/expectations/v0.1",
        "criteria": [
            _criterion("pass"),
            _criterion(
                "excluded",
                applicability="not_applicable",
                reason="Outside this criterion's declared scope.",
            ),
        ],
    }
    result = evaluate_expectations(
        pass_and_not_applicable,
        {"v(out)": _metric_record(mean_last_half=40.0)},
    )
    assert result["overall_status"] == "not_applicable"
    assert result["blocking_status"] == "pass"

    only_not_applicable = {
        "schema": "pear-sim/expectations/v0.1",
        "criteria": [
            _criterion(
                "excluded",
                applicability="not_applicable",
                reason="No applicable blocking criterion.",
            )
        ],
    }
    result = evaluate_expectations(only_not_applicable, {})
    assert result["blocking_status"] == "not_run"


@pytest.mark.parametrize(
    ("metric", "metadata_field"),
    [
        ("mean_last_half", "aggregation_method"),
        ("min_last_half", "extrema_method"),
        ("ripple_last_half", "ripple_method"),
    ],
)
def test_metric_method_provenance_is_required(metric: str, metadata_field: str) -> None:
    expectations = {
        "schema": "pear-sim/expectations/v0.1",
        "measurement_window": _window(),
        "criteria": [_criterion("method", metric=metric)],
    }
    record = _metric_record(**{metric: 40.0})
    del record[metadata_field]

    result = evaluate_expectations(expectations, {"v(out)": record})

    assert result["blocking_status"] == "not_comparable"
    assert metadata_field in result["criteria"][0]["reason"]
