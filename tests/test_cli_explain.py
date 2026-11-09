# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
import json
from pathlib import Path

from pear_sim.cli import main


ROOT = Path(__file__).parents[1]


def test_cli_explain_returns_machine_readable_hashes(capsys) -> None:
    exit_code = main(["explain", str(ROOT / "examples" / "buck.yaml"), "--json"])
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["schema"] == "pear-sim/explanation/v0.1"
    assert payload["summary"]["components"] == 6
    assert len(payload["hashes"]["model_sha256"]) == 64


def test_cli_render_writes_svg_and_metadata(tmp_path: Path, capsys) -> None:
    output = tmp_path / "schematic"
    exit_code = main(
        ["render", str(ROOT / "examples" / "buck.yaml"), "--out", str(output), "--json"]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert Path(payload["svg"]) == output / "schematic.svg"
    metadata = json.loads((output / "schematic.json").read_text(encoding="utf-8"))
    assert metadata["schema"] == "pear-sim/schematic/v0.1"
    assert len(metadata["components"]) == 6


def test_cli_topology_review_contains_human_view_and_accepts_annotation(
    tmp_path: Path, capsys
) -> None:
    review_dir = tmp_path / "topology-review"
    assert (
        main(
            [
                "review",
                "prepare",
                "topology",
                str(ROOT / "examples" / "buck.yaml"),
                "--expectations",
                str(ROOT / "examples" / "buck_expectations.json"),
                "--out",
                str(review_dir),
                "--json",
            ]
        )
        == 0
    )
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["phase"] == "topology"
    assert (review_dir / "artifacts" / "schematic" / "schematic.svg").is_file()
    assert (review_dir / "artifacts" / "explanation" / "explanation.json").is_file()

    annotation_path = tmp_path / "annotation.json"
    annotation_path.write_text(
        json.dumps(
            {
                "event": "comment",
                "author": {"kind": "human", "id": "reviewer:test"},
                "target": {"kind": "model_component", "component_id": "d1"},
                "severity": "warning",
                "category": "polarity",
                "message": "Check diode orientation.",
            }
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "review",
                "annotate",
                str(review_dir),
                "--file",
                str(annotation_path),
                "--json",
            ]
        )
        == 0
    )
    annotation = json.loads(capsys.readouterr().out)
    assert annotation["sequence"] == 1
    assert main(["review", "check", str(review_dir), "--json"]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["ok"] is True
    assert checked["decision"] is None


def test_cli_reviewed_simulation_records_human_decision(tmp_path: Path, capsys) -> None:
    review_dir = tmp_path / "review"
    assert (
        main(
            [
                "review",
                "prepare",
                "topology",
                str(ROOT / "examples" / "buck.yaml"),
                "--expectations",
                str(ROOT / "examples" / "buck_expectations.json"),
                "--out",
                str(review_dir),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    decision = json.loads(
        (review_dir / "review-template.json").read_text(encoding="utf-8")
    )
    decision.update(
        {
            "decision": "approve",
            "reviewer": {
                "kind": "human",
                "id": "reviewer:test",
                "authentication": "test-fixture",
            },
            "decided_at": "2026-07-18T12:00:00Z",
            "comment": "Approved for an exploratory test.",
            "scope": {"purpose": "exploratory"},
        }
    )
    decision_path = review_dir / "decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    run_dir = tmp_path / "reviewed-run"
    assert (
        main(
            [
                "simulate",
                str(ROOT / "examples" / "buck.yaml"),
                "--out",
                str(run_dir),
                "--review",
                str(review_dir),
                "--decision",
                str(decision_path),
                "--require-review",
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert manifest["topology_review"]["decision"] == "approve"
    assert manifest["topology_review"]["reviewer_id"] == "reviewer:test"


def test_cli_simulate_and_report_keep_report_outside_run(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "buck-run"
    schematic_dir = tmp_path / "schematic"
    assert (
        main(
            [
                "render",
                str(ROOT / "examples" / "buck.yaml"),
                "--out",
                str(schematic_dir),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
            [
                "simulate",
                str(ROOT / "examples" / "buck.yaml"),
                "--out",
                str(run_dir),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()

    report_dir = tmp_path / "buck-report"
    assert (
        main(
            [
                "report",
                str(run_dir),
                "--out",
                str(report_dir),
                "--schematic",
                str(schematic_dir / "schematic.svg"),
                "--json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)

    assert Path(payload["report_html"]) == report_dir / "report.html"
    assert (report_dir / "report.json").is_file()
    assert not (run_dir / "report.html").exists()

    result_review = tmp_path / "result-review"
    assert (
        main(
            [
                "review",
                "prepare",
                "result",
                str(run_dir),
                "--report",
                str(report_dir),
                "--out",
                str(result_review),
                "--json",
            ]
        )
        == 0
    )
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["phase"] == "result"
    assert main(["review", "check", str(result_review), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True


def test_cli_does_not_authorize_a_custom_request_it_will_not_execute(
    tmp_path: Path, capsys
) -> None:
    custom_request = tmp_path / "request.json"
    custom_request.write_text(
        json.dumps({"analysis": "transient", "stop_s": 99.0}),
        encoding="utf-8",
    )
    review_dir = tmp_path / "review"
    assert (
        main(
            [
                "review",
                "prepare",
                "topology",
                str(ROOT / "examples" / "buck.yaml"),
                "--request",
                str(custom_request),
                "--out",
                str(review_dir),
                "--json",
            ]
        )
        == 0
    )
    capsys.readouterr()
    decision = json.loads(
        (review_dir / "review-template.json").read_text(encoding="utf-8")
    )
    decision.update(
        {
            "decision": "approve",
            "reviewer": {"kind": "human", "id": "reviewer:test"},
            "decided_at": "2026-07-18T12:00:00Z",
        }
    )
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    run_dir = tmp_path / "run"

    assert (
        main(
            [
                "simulate",
                str(ROOT / "examples" / "buck.yaml"),
                "--out",
                str(run_dir),
                "--review",
                str(review_dir),
                "--decision",
                str(decision_path),
                "--require-review",
                "--json",
            ]
        )
        == 2
    )
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "StaleReviewError"
    assert not run_dir.exists()
