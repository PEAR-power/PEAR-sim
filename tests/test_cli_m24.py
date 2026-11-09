# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pear_sim.cli as cli_module
from pear_sim.benchmarks import INVERTING_BUCK_BOOST_DEFAULT_LOAD_RATIOS
from pear_sim.cli import main


ROOT = Path(__file__).parents[1]


def test_cli_compare_distinguishes_completed_failure_from_contract_error(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    output = tmp_path / "comparison"

    def fake_compare(reference, candidate, profile, out):
        assert reference == Path("reference")
        assert candidate == Path("candidate")
        assert profile == Path("profile.json")
        assert out == output
        return SimpleNamespace(
            status="fail",
            output_dir=output,
            resolved_profile=output / "compare-profile.resolved.json",
            comparison_json=output / "comparison.json",
            report_html=output / "report.html",
            artifact_manifest=output / "artifact-manifest.json",
        )

    monkeypatch.setattr(cli_module, "compare_runs", fake_compare)
    assert (
        main(
            [
                "compare",
                "reference",
                "candidate",
                "--profile",
                "profile.json",
                "--out",
                str(output),
                "--json",
            ]
        )
        == 1
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["accepted"] is False
    assert payload["status"] == "fail"
    assert payload["comparison_json"].endswith("comparison.json")

    def invalid_compare(*_args):
        raise ValueError("invalid verified-run contract")

    monkeypatch.setattr(cli_module, "compare_runs", invalid_compare)
    assert (
        main(
            [
                "compare",
                "reference",
                "candidate",
                "--profile",
                "profile.json",
                "--out",
                str(tmp_path / "invalid"),
                "--json",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["ok"] is False
    assert error["message"] == "invalid verified-run contract"


def test_cli_parameter_sweep_reports_not_run_acceptance_without_false_approval(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    output = tmp_path / "parameter-sweep"
    captured: dict[str, object] = {}

    def fake_sweep(model, request, out, *, expectations_file, jobs=None):
        captured.update(
            model=model,
            request=request,
            out=out,
            expectations_file=expectations_file,
        )
        return SimpleNamespace(
            execution_status="pass",
            acceptance_status="not_run",
            accepted=None,
            overall_status="pass",
            output_dir=output,
            source_request=output / "request.source.json",
            normalized_request=output / "request.normalized.json",
            sweep_summary=output / "sweep-summary.json",
            projections=output / "projections.json",
            failures=output / "failures.json",
            confidence_json=output / "confidence.json",
            report_html=output / "report" / "sweep-report.html",
            artifact_manifest=output / "artifact-manifest.json",
        )

    monkeypatch.setattr(cli_module, "run_parameter_sweep", fake_sweep)
    assert (
        main(
            [
                "sweep",
                "parameter",
                "model.yaml",
                "--request",
                "request.json",
                "--out",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["execution_status"] == "pass"
    assert payload["acceptance_status"] == "not_run"
    assert payload["accepted"] is None
    assert payload["overall_status"] == "pass"
    assert captured["expectations_file"] is None


def test_cli_parameter_sweep_rejects_unpaired_surrogate_without_output(
    tmp_path: Path,
    capsys,
) -> None:
    request = tmp_path / "surrogate-request.json"
    request.write_text(
        json.dumps(
            {
                "schema": "pear-sim/parameter-sweep-request/v0.1",
                "parameter": "load",
                "values": ["10 ohm"],
                "projections": [
                    {
                        "id": "\ud800",
                        "signal": "v(out)",
                        "metric": "sample_final",
                        "unit": "V",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "surrogate-output"

    assert (
        main(
            [
                "sweep",
                "parameter",
                str(ROOT / "examples" / "buck.yaml"),
                "--request",
                str(request),
                "--out",
                str(output),
                "--json",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["ok"] is False
    assert "valid UTF-8 Unicode scalar sequence" in error["message"]
    assert not output.exists()


def test_cli_parameter_sweep_rejects_excessive_json_depth_without_traceback(
    tmp_path: Path,
    capsys,
) -> None:
    request = tmp_path / "deep-request.json"
    nested_value = "[" * 1500 + "0" + "]" * 1500
    request.write_text(
        '{"schema":"pear-sim/parameter-sweep-request/v0.1",'
        '"parameter":"load","values":'
        + nested_value
        + "}",
        encoding="utf-8",
    )
    output = tmp_path / "deep-output"

    assert (
        main(
            [
                "sweep",
                "parameter",
                str(ROOT / "examples" / "buck.yaml"),
                "--request",
                str(request),
                "--out",
                str(output),
                "--json",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["ok"] is False
    assert "JSON" in error["message"]
    assert not output.exists()


def test_cli_inverting_buck_boost_uses_its_own_profile_and_schematic_pointer(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    output = tmp_path / "inverting-buck-boost"
    captured: dict[str, object] = {}

    def fake_runner(model, out, *, load_ratios):
        captured.update(model=model, out=out, load_ratios=load_ratios)
        return SimpleNamespace(
            overall_status="pass",
            output_dir=output,
            sweep_summary=output / "sweep-summary.json",
            benchmark_result=output / "benchmark-result.json",
            confidence_json=output / "confidence.json",
            report_html=output / "report" / "sweep-report.html",
            schematic_svg=output / "report" / "schematic.svg",
            artifact_manifest=output / "artifact-manifest.json",
        )

    monkeypatch.setattr(
        cli_module,
        "run_inverting_buck_boost_boundary_benchmark",
        fake_runner,
    )
    assert (
        main(
            [
                "sweep",
                "inverting-buck-boost-boundary",
                str(ROOT / "examples" / "inverting_buck_boost.yaml"),
                "--out",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["accepted"] is True
    assert payload["schematic_svg"].endswith("schematic.svg")
    assert captured["load_ratios"] == INVERTING_BUCK_BOOST_DEFAULT_LOAD_RATIOS


def test_cli_inverting_buck_boost_rejects_profile_without_exact_bcm_point(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "invalid"
    assert (
        main(
            [
                "sweep",
                "inverting-buck-boost-boundary",
                str(ROOT / "examples" / "inverting_buck_boost.yaml"),
                "--ratios",
                "0.9,1.1",
                "--out",
                str(output),
                "--json",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["ok"] is False
    assert "exact BCM ratio" in error["message"]
    assert not output.exists()
