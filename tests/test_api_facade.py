"""The typed agent-facing facade in pear_sim.api."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pear_sim.api as api
from pear_sim.errors import ModelError, SimulationError, TopologyError

ROOT = Path(__file__).parents[1]
BUCK = ROOT / "examples" / "buck.yaml"


def test_lint_returns_a_typed_static_report() -> None:
    report = api.lint(BUCK)
    assert report.model_name == "buck_ccm"
    assert report.components == 6
    assert report.states == 2
    assert set(report.hashes) >= {"source_sha256", "model_sha256", "request_sha256"}
    assert "v(out)" in report.outputs
    assert report.limitations


def test_simulate_returns_a_persisted_bundle_with_metrics(tmp_path: Path) -> None:
    run = api.simulate(BUCK, tmp_path / "run")
    assert run.run_manifest.is_file()
    assert run.summary.is_file()
    metrics = run.metrics()
    assert metrics["v(out)"]["mean"] == pytest.approx(40.0, abs=0.1)
    manifest = json.loads(run.run_manifest.read_text())
    assert manifest["schema"] == "pear-sim/run/v0.1"


def test_simulate_report_evaluate_compose_end_to_end(tmp_path: Path) -> None:
    run = api.simulate(BUCK, tmp_path / "run")
    report = api.report(run.output_dir, tmp_path / "report")
    assert report.report_html.is_file()
    evidence = api.evaluate(
        run.output_dir,
        ROOT / "examples" / "buck_expectations.json",
        tmp_path / "evidence",
    )
    assert evidence.evidence_json.is_file()
    comparison = api.compare(
        run.output_dir,
        run.output_dir,
        ROOT / "examples" / "buck_compare_profile.json",
        tmp_path / "comparison",
    )
    assert comparison.comparison_json.is_file()


def test_explain_render_and_hashes_take_paths_or_models(tmp_path: Path) -> None:
    model = api.load(BUCK)
    assert api.explain(model)["hashes"] == api.explain(BUCK)["hashes"]
    assert api.hashes(model) == api.hashes(BUCK)
    rendered = api.render(model)
    assert rendered.svg.startswith("<svg") or "<svg" in rendered.svg


def test_overrides_require_a_path_not_a_loaded_model() -> None:
    model = api.load(BUCK)
    with pytest.raises(ValueError, match="overrides require a model file path"):
        api.lint(model, {"vin": "120 V"})


def test_error_codes_are_stable_and_machine_readable(tmp_path: Path) -> None:
    assert ModelError.code == "model.invalid"
    assert TopologyError.code == "model.topology"
    assert SimulationError.code == "simulation.failed"
    try:
        api.load(tmp_path / "missing.yaml")
    except ModelError as exc:
        assert api.error_code(exc) == "model.invalid"
    else:  # pragma: no cover
        raise AssertionError("expected ModelError")
    assert api.error_code(ValueError("x")) == "request.invalid"
    assert api.error_code(OSError("x")) == "io.error"
    assert api.error_code(RuntimeError("x")) == "internal.error"


def test_cli_json_error_payload_carries_the_code(tmp_path: Path, capsys) -> None:
    from pear_sim.api.cli import main

    assert main(["lint", str(tmp_path / "missing.yaml"), "--json"]) == 2
    payload = json.loads(capsys.readouterr().err)
    assert payload["ok"] is False
    assert payload["code"] == "model.invalid"


def test_sweep_reexports_are_the_canonical_runners() -> None:
    from pear_sim.evidence.sweep import run_buck_boundary_benchmark

    assert api.sweep_buck_boundary is run_buck_boundary_benchmark
