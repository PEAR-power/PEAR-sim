from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pear_sim.solver import HybridSimulator
from pear_sim.sweep import MAX_SWEEP_POINTS, run_buck_boundary_benchmark


ROOT = Path(__file__).parents[1]
BUCK = ROOT / "examples" / "buck_dcm.yaml"
BOOST = ROOT / "examples" / "boost.yaml"


@pytest.fixture(scope="module")
def completed_sweep(tmp_path_factory: pytest.TempPathFactory):
    return run_buck_boundary_benchmark(
        BUCK,
        tmp_path_factory.mktemp("buck-boundary") / "sweep",
    )


def test_runner_retains_nine_verified_runs_and_human_report(completed_sweep) -> None:
    assert completed_sweep.overall_status == "pass"
    summary = json.loads(completed_sweep.sweep_summary.read_text(encoding="utf-8"))
    result = json.loads(completed_sweep.benchmark_result.read_text(encoding="utf-8"))
    confidence = json.loads(completed_sweep.confidence_json.read_text(encoding="utf-8"))
    report = json.loads(completed_sweep.report_json.read_text(encoding="utf-8"))

    assert summary["schema"] == "pear-sim/sweep-summary/ideal-buck-boundary/v0.1"
    assert summary["bcm"]["r_bcm_ohm"] == pytest.approx(
        6.089232079756019,
        rel=0.0,
        abs=5e-13,
    )
    assert summary["summary"] == {
        "gate_status_counts": {"pass": 99},
        "point_count": 9,
        "point_status_counts": {"pass": 9},
        "status": "pass",
    }
    assert [point["expected_regime"] for point in summary["points"]] == [
        "CCM",
        "CCM",
        "CCM",
        "CCM",
        "BCM",
        "DCM",
        "DCM",
        "DCM",
        "DCM",
    ]
    assert [point["observed_regime"] for point in summary["points"]] == [
        point["expected_regime"] for point in summary["points"]
    ]
    assert result["summary"] == {
        "gate_count": 99,
        "gate_status_counts": {"pass": 99},
        "point_count": 9,
        "status": "pass",
    }
    assert len(list((completed_sweep.output_dir / "points").glob("*/run/run.json"))) == 9
    assert len(summary["worst_gates"]) == 11
    assert summary["failures"] == []
    assert confidence["level"] == "EXPLORATORY"
    assert confidence["envelope"]["requested_regimes"] == ["CCM", "BCM", "DCM"]
    assert confidence["envelope"]["completed_regimes"] == ["CCM", "BCM", "DCM"]
    assert confidence["envelope"]["cross_platform_evidence"] == "not-run"
    assert confidence["envelope"]["convergence_sweep"] == "not-run"
    assert confidence["envelope"]["output_grid_independence_study"] == "not-run"
    assert confidence["related_regression_coverage"] == {
        "binding": "informational-only; not evidence for this run parameter set",
        "canonical_fixture_output_grid_independence": "covered-by-regression-suite",
        "canonical_fixture_selected_solver_settings": "covered-by-regression-suite",
    }
    assert confidence["claim"]["hardware_sign_off_allowed"] is False
    assert (
        report["source"]["sweep_summary_artifact_sha256"]
        == hashlib.sha256(completed_sweep.sweep_summary.read_bytes()).hexdigest()
    )
    assert (
        report["source"]["sweep_summary_artifact_sha256"]
        == confidence["acceptance"]["sweep_summary_sha256"]
    )

    rendered = completed_sweep.report_html.read_text(encoding="utf-8")
    assert "EXPLORATORY — NOT VALIDATION" in rendered
    assert "BCM · R/R_BCM = 1" in rendered
    assert "script-src 'none'" in rendered
    assert "<script" not in rendered.lower()


def test_confidence_reports_only_requested_and_completed_regimes(tmp_path: Path) -> None:
    artifacts = run_buck_boundary_benchmark(
        BUCK,
        tmp_path / "bcm-only",
        load_ratios=[1.0],
    )
    confidence = json.loads(artifacts.confidence_json.read_text(encoding="utf-8"))

    assert confidence["envelope"]["requested_regimes"] == ["BCM"]
    assert confidence["envelope"]["completed_regimes"] == ["BCM"]
    assert confidence["envelope"]["load_ratio_to_bcm"] == [1.0, 1.0]


def test_runner_records_oracle_seed_origin_and_resolved_load(completed_sweep) -> None:
    point_dir = completed_sweep.output_dir / "points" / "004-load-ratio-1"
    point = json.loads((point_dir / "benchmark-point.json").read_text(encoding="utf-8"))
    resolved = json.loads((point_dir / "run" / "model.resolved.json").read_text(encoding="utf-8"))

    assert point["initial_conditions"] == {
        "source": "independent periodic affine oracle",
        "inductor_current_a": 0.0,
        "output_voltage_v": pytest.approx(2.482365578926105, abs=1e-12),
    }
    assert resolved["parameters"]["load"]["value_si"] == pytest.approx(
        point["reference"]["load_ohm"],
        abs=1e-14,
    )
    assert resolved["initial_conditions"]["analytic_initialization_claimed"] is False
    assert (
        point["run"]["run_sha256"]
        == json.loads((point_dir / "run" / "run.json").read_text(encoding="utf-8"))["run_sha256"]
    )


def test_root_manifest_hashes_every_nested_artifact(completed_sweep) -> None:
    manifest = json.loads(completed_sweep.artifact_manifest.read_text(encoding="utf-8"))
    records = manifest["artifacts"]
    paths = [record["path"] for record in records]

    assert paths == sorted(paths)
    assert "sweep-summary.json" in paths
    assert "report/sweep-report.html" in paths
    assert "points/004-load-ratio-1/run/waveforms.csv" in paths
    assert "artifact-manifest.json" not in paths
    for record in records:
        path = completed_sweep.output_dir / record["path"]
        assert record["bytes"] == path.stat().st_size
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_runner_retains_a_failed_point_as_not_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = HybridSimulator.simulate

    def selectively_fail(simulator: HybridSimulator):
        if simulator.model.name.endswith("load-ratio-1p01"):
            raise RuntimeError("deliberate point failure")
        return original(simulator)

    monkeypatch.setattr(HybridSimulator, "simulate", selectively_fail)
    artifacts = run_buck_boundary_benchmark(BUCK, tmp_path / "sweep")
    summary = json.loads(artifacts.sweep_summary.read_text(encoding="utf-8"))

    assert artifacts.overall_status == "not-run"
    assert len(summary["points"]) == 9
    failed = next(point for point in summary["points"] if point["status"] == "not-run")
    assert failed["point_id"] == "load-ratio-1p01"
    assert {gate["status"] for gate in failed["gates"]} == {"not-run"}
    assert "deliberate point failure" in failed["gates"][0]["reason"]
    assert len(list((artifacts.output_dir / "points").glob("*/benchmark-point.json"))) == 9
    assert len(list((artifacts.output_dir / "points").glob("*/run/run.json"))) == 8
    assert summary["summary"]["point_status_counts"] == {"not-run": 1, "pass": 8}
    assert len(summary["failures"]) == 11


def test_runner_rejects_unsupported_or_unbounded_work_before_writing(tmp_path: Path) -> None:
    unsupported = tmp_path / "unsupported"
    with pytest.raises(ValueError):
        run_buck_boundary_benchmark(BOOST, unsupported)
    assert not unsupported.exists()

    too_many = tmp_path / "too-many"
    ratios = [index / 100.0 for index in range(1, MAX_SWEEP_POINTS + 2)]
    with pytest.raises(ValueError, match="point contract limit"):
        run_buck_boundary_benchmark(BUCK, too_many, load_ratios=ratios)
    assert not too_many.exists()


def test_runner_never_overwrites_prior_sweep(tmp_path: Path) -> None:
    destination = tmp_path / "sweep"
    destination.mkdir()
    marker = destination / "human-note.txt"
    marker.write_text("keep\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        run_buck_boundary_benchmark(BUCK, destination)

    assert marker.read_text(encoding="utf-8") == "keep\n"
