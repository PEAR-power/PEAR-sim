from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pear_sim.benchmarks import BOOST_DEFAULT_LOAD_RATIOS
from pear_sim.solver import HybridSimulator
from pear_sim.sweep import MAX_SWEEP_POINTS, run_boost_boundary_benchmark


ROOT = Path(__file__).parents[1]
BOOST = ROOT / "examples" / "boost.yaml"
BUCK = ROOT / "examples" / "buck_dcm.yaml"


@pytest.fixture(scope="module")
def completed_boost_sweep(tmp_path_factory: pytest.TempPathFactory):
    return run_boost_boundary_benchmark(
        BOOST,
        tmp_path_factory.mktemp("boost-boundary") / "sweep",
    )


def test_boost_runner_retains_nine_verified_runs_and_human_report(
    completed_boost_sweep,
) -> None:
    assert completed_boost_sweep.overall_status == "pass"
    summary = json.loads(completed_boost_sweep.sweep_summary.read_text(encoding="utf-8"))
    result = json.loads(completed_boost_sweep.benchmark_result.read_text(encoding="utf-8"))
    confidence = json.loads(completed_boost_sweep.confidence_json.read_text(encoding="utf-8"))
    report = json.loads(completed_boost_sweep.report_json.read_text(encoding="utf-8"))

    assert summary["schema"] == "pear-sim/sweep-summary/ideal-boost-boundary/v0.1"
    assert summary["benchmark_id"] == "VAL-003:canonical-boost-boundary-sweep"
    assert summary["bcm"]["r_bcm_ohm"] == pytest.approx(
        159.87499350342793,
        rel=0.0,
        abs=5e-10,
    )
    assert summary["summary"] == {
        "gate_status_counts": {"pass": 117},
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
    assert result["schema"] == "pear-sim/benchmark-runner-result/ideal-boost-boundary/v0.1"
    assert result["oracle_schema"] == "pear-sim/benchmark-oracle/ideal-boost-boundary/v0.1"
    assert result["comparison_schema"] == "pear-sim/benchmark-result/ideal-boost-boundary/v0.1"
    assert result["summary"] == {
        "gate_count": 117,
        "gate_status_counts": {"pass": 117},
        "point_count": 9,
        "status": "pass",
    }
    assert result["claim"]["validation_claim"] is False
    assert len(list((completed_boost_sweep.output_dir / "points").glob("*/run/run.json"))) == 9
    assert len(summary["worst_gates"]) == 13
    assert summary["failures"] == []

    assert confidence["schema"] == "pear-sim/confidence/boost-boundary-sweep/v0.1"
    assert confidence["level"] == "EXPLORATORY"
    assert confidence["envelope"]["topology"] == "canonical single-switch ideal boost"
    assert confidence["envelope"]["requested_regimes"] == ["CCM", "BCM", "DCM"]
    assert confidence["envelope"]["completed_regimes"] == ["CCM", "BCM", "DCM"]
    assert confidence["envelope"]["cross_platform_evidence"] == "not-run"
    assert confidence["envelope"]["convergence_sweep"] == "not-run"
    assert confidence["envelope"]["output_grid_independence_study"] == "not-run"
    assert confidence["claim"]["hardware_sign_off_allowed"] is False
    assert (
        report["source"]["sweep_summary_artifact_sha256"]
        == hashlib.sha256(completed_boost_sweep.sweep_summary.read_bytes()).hexdigest()
    )
    assert (
        report["source"]["sweep_summary_artifact_sha256"]
        == confidence["acceptance"]["sweep_summary_sha256"]
    )

    rendered = completed_boost_sweep.report_html.read_text(encoding="utf-8")
    assert "EXPLORATORY — NOT VALIDATION" in rendered
    assert "VAL-003:canonical-boost-boundary-sweep" in rendered
    assert "BCM · R/R_BCM = 1" in rendered
    assert "script-src 'none'" in rendered
    assert "<script" not in rendered.lower()
    assert completed_boost_sweep.schematic_svg is not None
    assert completed_boost_sweep.schematic_svg.is_file()
    assert "Nominal source-model topology template" in rendered


def test_boost_runner_records_oracle_seed_and_resolved_load(completed_boost_sweep) -> None:
    point_dir = completed_boost_sweep.output_dir / "points" / "004-load-ratio-1"
    point = json.loads((point_dir / "benchmark-point.json").read_text(encoding="utf-8"))
    resolved = json.loads((point_dir / "run" / "model.resolved.json").read_text(encoding="utf-8"))

    assert point["initial_conditions"]["source"] == "independent periodic affine oracle"
    assert point["initial_conditions"]["inductor_current_a"] == pytest.approx(0.0, abs=1e-12)
    assert point["initial_conditions"]["output_voltage_v"] == pytest.approx(
        48.0124947667,
        abs=5e-9,
    )
    assert resolved["parameters"]["load"]["value_si"] == pytest.approx(
        point["reference"]["load_ohm"],
        abs=1e-12,
    )
    assert resolved["initial_conditions"]["analytic_initialization_claimed"] is False
    assert (
        point["run"]["run_sha256"]
        == json.loads((point_dir / "run" / "run.json").read_text(encoding="utf-8"))[
            "run_sha256"
        ]
    )


def test_boost_root_manifest_hashes_every_nested_artifact(completed_boost_sweep) -> None:
    manifest = json.loads(completed_boost_sweep.artifact_manifest.read_text(encoding="utf-8"))
    records = manifest["artifacts"]
    paths = [record["path"] for record in records]

    assert manifest["subject"] == {
        "kind": "boost-boundary-sweep",
        "sha256": hashlib.sha256(completed_boost_sweep.sweep_summary.read_bytes()).hexdigest(),
    }
    assert paths == sorted(paths)
    assert "sweep-summary.json" in paths
    assert "report/sweep-report.html" in paths
    assert "report/schematic.svg" in paths
    assert "points/004-load-ratio-1/run/waveforms.csv" in paths
    assert "artifact-manifest.json" not in paths
    for record in records:
        path = completed_boost_sweep.output_dir / record["path"]
        assert record["bytes"] == path.stat().st_size
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_boost_runner_retains_a_failed_point_with_thirteen_not_run_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = HybridSimulator.simulate

    def selectively_fail(simulator: HybridSimulator):
        if simulator.model.name.endswith("load-ratio-1p01"):
            raise RuntimeError("deliberate boost point failure")
        return original(simulator)

    monkeypatch.setattr(HybridSimulator, "simulate", selectively_fail)
    artifacts = run_boost_boundary_benchmark(BOOST, tmp_path / "sweep")
    summary = json.loads(artifacts.sweep_summary.read_text(encoding="utf-8"))

    assert artifacts.overall_status == "not-run"
    assert len(summary["points"]) == len(BOOST_DEFAULT_LOAD_RATIOS)
    failed = next(point for point in summary["points"] if point["status"] == "not-run")
    assert failed["point_id"] == "load-ratio-1p01"
    assert [gate["id"] for gate in failed["gates"]] == [
        "regime",
        "boundary_state_nonzero",
        "boundary_state_zero",
        "natural_event_time",
        "reverse_current",
        "cycle_mean_output",
        "state_ripple",
        "mode_fractions",
        "event_semantics",
        "energy_balance",
        "cycle_periodicity",
        "unclassified_fraction",
        "mode_fraction_sum",
    ]
    assert {gate["status"] for gate in failed["gates"]} == {"not-run"}
    assert "deliberate boost point failure" in failed["gates"][0]["reason"]
    assert len(list((artifacts.output_dir / "points").glob("*/benchmark-point.json"))) == 9
    assert len(list((artifacts.output_dir / "points").glob("*/run/run.json"))) == 8
    assert summary["summary"]["point_status_counts"] == {"not-run": 1, "pass": 8}
    assert len(summary["failures"]) == 13


def test_boost_runner_rejects_unsupported_or_unbounded_work_before_writing(
    tmp_path: Path,
) -> None:
    unsupported = tmp_path / "unsupported"
    with pytest.raises(ValueError):
        run_boost_boundary_benchmark(BUCK, unsupported)
    assert not unsupported.exists()

    too_many = tmp_path / "too-many"
    ratios = [index / 100.0 for index in range(1, MAX_SWEEP_POINTS + 2)]
    with pytest.raises(ValueError, match="point contract limit"):
        run_boost_boundary_benchmark(BOOST, too_many, load_ratios=ratios)
    assert not too_many.exists()

    ambiguous = tmp_path / "ambiguous"
    with pytest.raises(ValueError, match="exact BCM ratio"):
        run_boost_boundary_benchmark(BOOST, ambiguous, load_ratios=[0.9, 1.1])
    assert not ambiguous.exists()


def test_boost_runner_never_overwrites_prior_sweep(tmp_path: Path) -> None:
    destination = tmp_path / "sweep"
    destination.mkdir()
    marker = destination / "human-note.txt"
    marker.write_text("keep\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        run_boost_boundary_benchmark(BOOST, destination, load_ratios=[1.0])

    assert marker.read_text(encoding="utf-8") == "keep\n"
