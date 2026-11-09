# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pear_sim.cli as cli_module
from pear_sim.benchmarks import BOOST_DEFAULT_LOAD_RATIOS
from pear_sim.cli import build_parser, main


ROOT = Path(__file__).parents[1]


def test_cli_uses_canonical_pear_sim_product_name() -> None:
    parser = build_parser()

    assert parser.prog == "pear-sim"


def test_cli_catalog_lists_every_loader_supported_control(capsys) -> None:
    assert main(["catalog", "--json"]) == 0
    catalog = json.loads(capsys.readouterr().out)

    assert catalog["controls"] == ["pwm", "complementary_pwm", "three_phase_spwm"]


def test_cli_capabilities_is_machine_readable_and_unknown_is_fail_closed(capsys) -> None:
    assert main(["capabilities", "--json"]) == 0
    catalog = json.loads(capsys.readouterr().out)
    assert catalog["schema"] == "pear-sim/capabilities/v0.1"
    assert catalog["default_policy"]["silent_fallback"] is False

    assert main(["capabilities", "--id", "future.magic", "--json"]) == 0
    unknown = json.loads(capsys.readouterr().out)
    assert unknown["status"] == "unsupported"
    assert unknown["reason"] == "fail_closed_unlisted_capability"


def test_cli_evaluate_uses_distinct_exit_for_completed_blocking_failure(
    tmp_path: Path,
    capsys,
) -> None:
    run = tmp_path / "run"
    assert (
        main(
            [
                "simulate",
                str(ROOT / "examples" / "buck.yaml"),
                "--out",
                str(run),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()

    passing = tmp_path / "passing"
    assert (
        main(
            [
                "evaluate",
                str(run),
                "--expectations",
                str(ROOT / "examples" / "buck_expectations.json"),
                "--out",
                str(passing),
                "--json",
            ]
        )
        == 0
    )
    passed = json.loads(capsys.readouterr().out)
    assert passed["ok"] is True
    assert passed["accepted"] is True
    assert passed["blocking_status"] == "pass"

    expectations = tmp_path / "failing.json"
    expectations.write_text(
        json.dumps(
            {
                "schema": "pear-sim/expectations/v0.1",
                "criteria": [
                    {
                        "id": "vout_too_low",
                        "signal": "v(out)",
                        "metric": "mean_last_half",
                        "unit": "V",
                        "maximum": 1.0,
                        "blocking": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    failing = tmp_path / "failing"
    assert (
        main(
            [
                "evaluate",
                str(run),
                "--expectations",
                str(expectations),
                "--out",
                str(failing),
                "--json",
            ]
        )
        == 1
    )
    failed = json.loads(capsys.readouterr().out)
    assert failed["ok"] is True
    assert failed["accepted"] is False
    assert failed["blocking_status"] == "fail"
    assert Path(failed["evidence_json"]).is_file()

    invalid_documents = [
        {"schema": [], "criteria": []},
        {
            "schema": "pear-sim/expectations/v0.1",
            "criteria": [
                {
                    "id": "overflow",
                    "signal": "v(out)",
                    "metric": "mean_last_half",
                    "unit": "V",
                    "target": 10**400,
                    "absolute_tolerance": 0.1,
                }
            ],
        },
    ]
    for index, document in enumerate(invalid_documents):
        invalid_file = tmp_path / f"invalid-{index}.json"
        invalid_file.write_text(json.dumps(document), encoding="utf-8")
        invalid_output = tmp_path / f"invalid-{index}"

        assert (
            main(
                [
                    "evaluate",
                    str(run),
                    "--expectations",
                    str(invalid_file),
                    "--out",
                    str(invalid_output),
                    "--json",
                ]
            )
            == 2
        )
        error = json.loads(capsys.readouterr().err)
        assert error["ok"] is False
        assert not invalid_output.exists()


def test_cli_buck_boundary_sweep_returns_compact_agent_pointers(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "sweep"

    assert (
        main(
            [
                "sweep",
                "buck-boundary",
                str(ROOT / "examples" / "buck_dcm.yaml"),
                "--ratios",
                "0.9,1,1.1",
                "--out",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["accepted"] is True
    assert payload["overall_status"] == "pass"
    assert Path(payload["sweep_summary"]).is_file()
    assert Path(payload["report_html"]).is_file()
    summary = json.loads(Path(payload["sweep_summary"]).read_text(encoding="utf-8"))
    assert summary["summary"]["point_count"] == 3
    assert summary["summary"]["gate_status_counts"] == {"pass": 33}


def test_cli_buck_boundary_rejects_ambiguous_ratios_without_output(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "invalid-sweep"

    assert (
        main(
            [
                "sweep",
                "buck-boundary",
                str(ROOT / "examples" / "buck_dcm.yaml"),
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


def test_cli_boost_boundary_sweep_returns_compact_agent_pointers(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "boost-sweep"

    assert (
        main(
            [
                "sweep",
                "boost-boundary",
                str(ROOT / "examples" / "boost.yaml"),
                "--ratios",
                "0.9,1,1.1",
                "--out",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["ok"] is True
    assert payload["accepted"] is True
    assert payload["overall_status"] == "pass"
    assert Path(payload["sweep_summary"]).is_file()
    assert Path(payload["benchmark_result"]).is_file()
    assert Path(payload["report_html"]).is_file()
    assert Path(payload["schematic_svg"]).is_file()
    assert Path(payload["artifact_manifest"]).is_file()
    summary = json.loads(Path(payload["sweep_summary"]).read_text(encoding="utf-8"))
    assert summary["schema"] == "pear-sim/sweep-summary/ideal-boost-boundary/v0.1"
    assert summary["summary"]["point_count"] == 3
    assert summary["summary"]["gate_status_counts"] == {"pass": 39}


def test_cli_boost_boundary_uses_its_own_default_profile(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    output = tmp_path / "boost-default"
    captured: dict[str, object] = {}

    def fake_runner(model, out, *, load_ratios):
        captured["model"] = model
        captured["out"] = out
        captured["load_ratios"] = load_ratios
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

    monkeypatch.setattr(cli_module, "run_boost_boundary_benchmark", fake_runner)
    assert (
        main(
            [
                "sweep",
                "boost-boundary",
                str(ROOT / "examples" / "boost.yaml"),
                "--out",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert payload["accepted"] is True
    assert captured["load_ratios"] == BOOST_DEFAULT_LOAD_RATIOS


def test_cli_boost_boundary_rejects_ambiguous_ratios_without_output(
    tmp_path: Path,
    capsys,
) -> None:
    output = tmp_path / "invalid-boost-sweep"

    assert (
        main(
            [
                "sweep",
                "boost-boundary",
                str(ROOT / "examples" / "boost.yaml"),
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


def test_cli_numerical_evidence_returns_compact_machine_pointers(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    output = tmp_path / "numerical"
    output.mkdir()
    evidence = output / "platform-evidence.json"
    manifest = output / "artifact-manifest.json"
    report = output / "report" / "numerical-report.html"
    report.parent.mkdir()
    evidence.write_text("{}\n", encoding="utf-8")
    manifest.write_text("{}\n", encoding="utf-8")
    report.write_text("<!doctype html>\n", encoding="utf-8")

    monkeypatch.setattr(
        cli_module,
        "run_buck_numerical_evidence",
        lambda model, out, source_revision=None: SimpleNamespace(
            status="pass",
            output_dir=output,
            platform_evidence=evidence,
            report_html=report,
            artifact_manifest=manifest,
        ),
    )

    assert (
        main(
            [
                "verify",
                "numerics",
                "buck-boundary",
                str(ROOT / "examples" / "buck_dcm.yaml"),
                "--profile",
                "full-v1",
                "--source-revision",
                "test-revision",
                "--out",
                str(output),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "ok": True,
        "accepted": True,
        "status": "pass",
        "profile": "full-v1",
        "output": str(output),
        "platform_evidence": str(evidence),
        "report_html": str(report),
        "artifact_manifest": str(manifest),
    }


def test_cli_platform_aggregation_requires_the_exact_native_set(
    tmp_path: Path,
    capsys,
) -> None:
    assert (
        main(
            [
                "verify",
                "platforms",
                str(tmp_path / "linux"),
                str(tmp_path / "windows"),
                "--require",
                "Linux,Windows",
                "--out",
                str(tmp_path / "cross"),
                "--json",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["ok"] is False
    assert "requires exactly" in error["message"]
