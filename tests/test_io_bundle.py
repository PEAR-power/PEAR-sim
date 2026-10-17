import json
from pathlib import Path

import pytest

from pear_sim.contracts import sha256_file
from pear_sim.io import write_result
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator


ROOT = Path(__file__).parents[1]


def test_run_bundle_contains_portable_snapshot_and_verified_artifact_hashes(tmp_path: Path) -> None:
    model = load_model(ROOT / "examples" / "buck.yaml")
    destination = write_result(HybridSimulator(model).simulate(), tmp_path / "run")

    manifest = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    artifacts = json.loads(
        (destination / "artifact-manifest.json").read_text(encoding="utf-8")
    )

    assert manifest["maturity"] == "experimental"
    assert manifest["model_file"] == "buck.yaml"
    assert "model_path" not in manifest
    assert (destination / "model.resolved.json").is_file()
    assert (destination / "summary.json").is_file()
    assert (destination / "warnings.json").is_file()
    assert (destination / "analytic-checks.json").is_file()
    assert (destination / "physics-audit.json").is_file()
    assert (destination / "mode-coverage.json").is_file()
    metrics = json.loads((destination / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["v(out)"]["unit"] == "V"
    assert metrics["i(l1)"]["unit"] == "A"
    assert metrics["v(out)"]["window_start_s"] < metrics["v(out)"]["window_end_s"]
    analytic_checks = json.loads(
        (destination / "analytic-checks.json").read_text(encoding="utf-8")
    )
    assert analytic_checks["overall_status"] == "pass"
    assert all(check["status"] == "pass" for check in analytic_checks["checks"])
    physics_audit = json.loads(
        (destination / "physics-audit.json").read_text(encoding="utf-8")
    )
    assert physics_audit["power_sign_convention"] == "positive_absorbed_from_p_to_n"
    assert physics_audit["last_half"]["balance_residual_relative"] < 1e-8
    mode_coverage = json.loads(
        (destination / "mode-coverage.json").read_text(encoding="utf-8")
    )
    assert mode_coverage["status"] == "observed"
    assert mode_coverage["claim"]["validation_claim"] is False
    assert mode_coverage["summary"]["natural_commutation"]["exercised"] is False
    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode_coverage"]["artifact"] == "mode-coverage.json"
    assert summary["mode_coverage"]["status"] == "observed"
    assert "intervals" not in summary["mode_coverage"]
    assert "event_timing" not in summary["mode_coverage"]
    warnings = json.loads((destination / "warnings.json").read_text(encoding="utf-8"))
    assert any(
        warning["code"] == "NATURAL_COMMUTATION_NOT_EXERCISED"
        for warning in warnings
    )
    assert manifest["files"]["analytic_checks"] == "analytic-checks.json"
    assert manifest["files"]["physics_audit"] == "physics-audit.json"
    assert manifest["files"]["mode_coverage"] == "mode-coverage.json"
    assert {artifact["path"] for artifact in artifacts["artifacts"]} == {
        "run.json",
        *manifest["files"].values(),
    }
    for artifact in artifacts["artifacts"]:
        path = destination / artifact["path"]
        assert artifact["sha256"] == sha256_file(path)
        assert artifact["bytes"] == path.stat().st_size


def test_write_result_refuses_to_overwrite_existing_run(tmp_path: Path) -> None:
    model = load_model(ROOT / "examples" / "buck.yaml")
    result = HybridSimulator(model).simulate()
    destination = write_result(result, tmp_path / "run")

    with pytest.raises(ValueError, match="never overwritten"):
        write_result(result, destination)

    assert (destination / "artifact-manifest.json").is_file()
