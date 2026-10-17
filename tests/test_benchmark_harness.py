from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from pathlib import Path

import pytest

from pear_sim.benchmark_harness import (
    BoundaryHarnessContract,
    overall_status,
    prepare_destination,
    run_retained_boundary_points,
    write_json,
)


@dataclass(frozen=True)
class _Reference:
    point_id: str
    load_ratio_to_bcm: float
    load_ohm: float
    regime: str

    def as_dict(self) -> dict[str, object]:
        return {
            "point_id": self.point_id,
            "load_ratio_to_bcm": self.load_ratio_to_bcm,
            "load_ohm": self.load_ohm,
            "regime": self.regime,
        }


@dataclass(frozen=True)
class _Observation:
    regime: str
    value: float


@dataclass(frozen=True)
class _Result:
    reference: _Reference
    stats: dict[str, float]


def _contract(*, bounded: bool = False) -> BoundaryHarnessContract:
    return BoundaryHarnessContract(
        point_schema="test/point/v0.1",
        comparison_schema="test/comparison/v0.1",
        gate_ids=("physics", "energy"),
        bounded_error_text=bounded,
    )


def test_harness_module_has_no_solver_or_oracle_import() -> None:
    source = (Path(__file__).parents[1] / "src" / "pear_sim" / "benchmark_harness.py").read_text(
        encoding="utf-8"
    )
    imported = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    )
    assert "pear_sim.solver" not in imported
    assert "pear_sim.benchmarks" not in imported
    assert "pear_sim.sweep" not in imported


def test_contract_rejects_duplicate_or_empty_gate_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        BoundaryHarnessContract("point", "comparison", ("same", "same"))
    with pytest.raises(ValueError, match="non-empty"):
        BoundaryHarnessContract("point", "comparison", ())


def test_retained_runner_keeps_order_and_fails_closed_with_bounded_error(tmp_path: Path) -> None:
    destination = prepare_destination(tmp_path / "evidence")
    references = (
        _Reference("passing", 0.5, 5.0, "CCM"),
        _Reference("observation-broken", 1.0, 10.0, "BCM"),
        _Reference("simulation-broken", 2.0, 20.0, "DCM"),
    )

    def simulate(reference: _Reference) -> _Result:
        if reference.point_id == "simulation-broken":
            raise RuntimeError("€" * 2000)
        return _Result(reference, {"wall_time_s": 0.25})

    def observe(result: _Result, reference: _Reference) -> _Observation:
        assert result.reference == reference
        if reference.point_id == "observation-broken":
            raise RuntimeError("€" * 2000)
        return _Observation(regime=reference.regime, value=3.0)

    def compare(reference: _Reference, observation: _Observation) -> dict[str, object]:
        return {
            "schema": "test/comparison/v0.1",
            "point_id": reference.point_id,
            "status": "pass",
            "gates": [
                {"id": "physics", "actual": 0.0, "limit": 1.0, "status": "pass"},
                {"id": "energy", "actual": 0.0, "limit": 1.0, "status": "pass"},
            ],
        }

    def retain_run(_result: _Result, run_path: Path) -> Path:
        run_path.mkdir()
        write_json(run_path / "run.json", {"run_sha256": "a" * 64})
        return run_path

    retained = run_retained_boundary_points(
        destination,
        references,
        contract=_contract(bounded=True),
        simulate=simulate,
        observe=observe,
        compare=compare,
        retain_run=retain_run,
        initial_conditions=lambda reference: {"source": "test", "load": reference.load_ohm},
        observation_payload=lambda observation: {
            "regime": observation.regime,
            "value": observation.value,
        },
        summary_metrics=lambda observation: {
            "value": None if observation is None else observation.value
        },
    )

    assert [point["point_id"] for point in retained.summaries] == [
        "passing",
        "observation-broken",
        "simulation-broken",
    ]
    assert retained.comparisons[0]["status"] == "pass"
    assert retained.comparisons[1]["status"] == "not-run"
    assert retained.comparisons[2]["status"] == "not-run"
    assert {gate["status"] for gate in retained.comparisons[1]["gates"]} == {"not-run"}
    assert overall_status(retained.summaries) == "not-run"
    assert retained.summaries[1]["runtime_s"] == 0.25
    assert retained.summaries[2]["runtime_s"] is None

    passing = json.loads(
        (destination / "points" / "000-passing" / "benchmark-point.json").read_text(
            encoding="utf-8"
        )
    )
    broken = json.loads(
        (
            destination
            / "points"
            / "001-observation-broken"
            / "benchmark-point.json"
        ).read_text(encoding="utf-8")
    )
    assert passing["run"]["run_sha256"] == "a" * 64
    assert broken["run"] is None
    assert broken["observation"] is None
    assert len(broken["error"]["message"].encode("utf-8")) <= 1024
    assert len(broken["comparison"]["gates"][0]["reason"].encode("utf-8")) <= 1024


def test_prepare_destination_refuses_to_overwrite_prior_evidence(tmp_path: Path) -> None:
    destination = tmp_path / "evidence"
    destination.mkdir()
    (destination / "prior.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="never overwritten"):
        prepare_destination(destination)
