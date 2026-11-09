# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil

import pytest

from pear_sim.evidence_io import evaluate_run
from pear_sim.io import write_result
from pear_sim.model import load_model
from pear_sim.report import ReportError
from pear_sim.solver import HybridSimulator


ROOT = Path(__file__).parents[1]


def _run(tmp_path: Path) -> Path:
    result = HybridSimulator(load_model(ROOT / "examples" / "buck.yaml")).simulate()
    return write_result(result, tmp_path / "run")


def test_evidence_bundle_is_fail_closed_and_bound_to_verified_run(tmp_path: Path) -> None:
    run = _run(tmp_path)
    output = tmp_path / "evidence"

    artifacts = evaluate_run(
        run,
        ROOT / "examples" / "buck_expectations.json",
        output,
    )

    assert artifacts.blocking_status == "pass"
    evidence = json.loads(artifacts.evidence_json.read_text(encoding="utf-8"))
    confidence = json.loads(artifacts.confidence_json.read_text(encoding="utf-8"))
    manifest = json.loads(artifacts.artifact_manifest.read_text(encoding="utf-8"))
    assert evidence["schema"] == "pear-sim/evidence/v0.1"
    assert evidence["source"]["run_artifact_manifest_verified"] is True
    assert (
        evidence["source"]["run_sha256"]
        == json.loads((run / "run.json").read_text(encoding="utf-8"))["run_sha256"]
    )
    assert confidence["level"] == "EXPLORATORY"
    assert confidence["claim"]["hardware_sign_off_allowed"] is False
    assert manifest["subject"]["kind"] == "evidence"
    assert [item["path"] for item in manifest["artifacts"]] == [
        "confidence.json",
        "evidence.json",
    ]


def test_evidence_bundle_retains_completed_blocking_failure(tmp_path: Path) -> None:
    run = _run(tmp_path)
    expectations = tmp_path / "failing.json"
    expectations.write_text(
        json.dumps(
            {
                "schema": "pear-sim/expectations/v0.1",
                "criteria": [
                    {
                        "id": "impossible",
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

    artifacts = evaluate_run(run, expectations, tmp_path / "evidence")

    assert artifacts.blocking_status == "fail"
    payload = json.loads(artifacts.evidence_json.read_text(encoding="utf-8"))
    assert payload["criteria"][0]["status"] == "fail"
    assert payload["criteria"][0]["measured"]["value"] > 39.0


def test_evidence_rejects_tampered_run_and_does_not_create_output(tmp_path: Path) -> None:
    run = _run(tmp_path)
    metrics_path = run / "metrics.json"
    metrics_path.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "evidence"

    with pytest.raises(ReportError, match="SHA-256 mismatch"):
        evaluate_run(
            run,
            ROOT / "examples" / "buck_expectations.json",
            output,
        )

    assert not output.exists()


def test_evidence_never_overwrites_prior_artifact(tmp_path: Path) -> None:
    run = _run(tmp_path)
    output = tmp_path / "evidence"
    output.mkdir()
    marker = output / "human-note.txt"
    marker.write_text("keep\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        evaluate_run(
            run,
            ROOT / "examples" / "buck_expectations.json",
            output,
        )

    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_evidence_content_is_independent_of_input_directory_names(tmp_path: Path) -> None:
    run = _run(tmp_path)
    copied_run = tmp_path / "renamed-run"
    shutil.copytree(run, copied_run)
    copied_expectations = tmp_path / "renamed-expectations.json"
    shutil.copyfile(ROOT / "examples" / "buck_expectations.json", copied_expectations)

    first = evaluate_run(
        run,
        ROOT / "examples" / "buck_expectations.json",
        tmp_path / "first-evidence",
    )
    second = evaluate_run(
        copied_run,
        copied_expectations,
        tmp_path / "second-evidence",
    )

    assert first.evidence_json.read_bytes() == second.evidence_json.read_bytes()
    assert first.confidence_json.read_bytes() == second.confidence_json.read_bytes()
    assert first.artifact_manifest.read_bytes() == second.artifact_manifest.read_bytes()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("schema", "pear-sim/run/v999", "Unsupported run schema"),
        ("backend", "unknown-backend", "Unsupported run backend"),
    ],
)
def test_evidence_rejects_unknown_run_contract_even_when_manifest_is_self_consistent(
    tmp_path: Path,
    field: str,
    value: str,
    match: str,
) -> None:
    run = _run(tmp_path)
    run_path = run / "run.json"
    payload = json.loads(run_path.read_text(encoding="utf-8"))
    payload[field] = value
    run_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_path = run / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = next(item for item in manifest["artifacts"] if item["path"] == "run.json")
    record["bytes"] = run_path.stat().st_size
    record["sha256"] = hashlib.sha256(run_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "evidence"

    with pytest.raises(ValueError, match=match):
        evaluate_run(run, ROOT / "examples" / "buck_expectations.json", output)

    assert not output.exists()


def test_semantically_invalid_expectations_do_not_create_output(tmp_path: Path) -> None:
    run = _run(tmp_path)
    expectations = tmp_path / "invalid-expectations.json"
    expectations.write_text(
        json.dumps({"schema": [], "criteria": []}),
        encoding="utf-8",
    )
    output = tmp_path / "evidence"

    with pytest.raises(ValueError, match="schema must be"):
        evaluate_run(run, expectations, output)

    assert not output.exists()
