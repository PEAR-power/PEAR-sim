from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from pear_sim.contracts import model_hashes
from pear_sim.model import load_model
from pear_sim.review import (
    DECISION_SCHEMA,
    REVIEW_SCHEMA,
    ReviewDecisionError,
    ReviewIntegrityError,
    ReviewValidationError,
    StaleReviewError,
    append_review_annotation,
    check_review_decision,
    check_review_packet,
    compute_result_subject_digest,
    compute_topology_subject_digest,
    prepare_result_review,
    prepare_topology_review,
    validate_review_annotations,
    validate_review_decision,
    validate_review_json,
)


ROOT = Path(__file__).parents[1]
BUCK = ROOT / "examples" / "buck.yaml"


def _decision(packet: dict, action: str = "approve") -> dict:
    return {
        "schema": DECISION_SCHEMA,
        "review_id": packet["review_id"],
        "phase": packet["phase"],
        "subject_digest": packet["subject_digest"],
        "packet_digest": packet["packet_digest"],
        "decision": action,
        "reviewer": {
            "kind": "human",
            "id": "reviewer:test",
            "authentication": "test-fixture",
        },
        "decided_at": "2026-07-18T12:00:00Z",
        "reason_codes": [],
        "comment": "Reviewed for an exploratory test.",
        "scope": {"purpose": "exploratory"},
        "signature": None,
    }


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _make_run(run_dir: Path) -> None:
    run_dir.mkdir()
    _write_json(
        run_dir / "run.json",
        {
            "schema": "pear-sim/run/v0.1",
            "model_sha256": "a" * 64,
            "request_sha256": "b" * 64,
            "run_sha256": "c" * 64,
            "files": {"waveforms": "waveforms.csv", "metrics": "metrics.json"},
        },
    )
    (run_dir / "waveforms.csv").write_text("time,v(out)\n0,0\n1e-6,1\n", encoding="utf-8")
    _write_json(run_dir / "metrics.json", {"v(out)": {"max": 1.0}})


def test_prepare_topology_review_creates_bound_pending_packet(tmp_path: Path):
    out = tmp_path / "topology-review"
    expectations = {"checks": [{"metric": "mean:v(out)", "between": [39.0, 41.0]}]}
    packet = prepare_topology_review(BUCK, expectations=expectations, out_dir=out)

    hashes = model_hashes(load_model(BUCK))
    assert packet["schema"] == REVIEW_SCHEMA
    assert packet["phase"] == "topology"
    assert packet["status"] == "pending"
    assert packet["subject"]["model_digest"] == f"sha256:{hashes['model_sha256']}"
    assert packet["subject"]["request_digest"] == f"sha256:{hashes['request_sha256']}"
    assert packet["subject"]["expectations_digest"].startswith("sha256:")
    assert {item["role"] for item in packet["artifacts"]} == {
        "source",
        "model",
        "request",
        "explanation",
        "schematic",
        "expectations",
    }
    assert all(item["sha256"].startswith("sha256:") for item in packet["artifacts"])
    assert validate_review_json(out / "packet.json") == packet
    assert check_review_packet(out) == packet

    template = json.loads((out / "review-template.json").read_text(encoding="utf-8"))
    assert template["subject_digest"] == packet["subject_digest"]
    assert template["packet_digest"] == packet["packet_digest"]
    assert template["decision"] is None
    assert template["reviewer"] is None
    assert (out / "artifacts" / "schematic" / "schematic.svg").is_file()
    explanation = json.loads(
        (out / "artifacts" / "explanation" / "explanation.json").read_text(
            encoding="utf-8"
        )
    )
    assert explanation["schema"] == "pear-sim/explanation/v0.1"


def test_decisions_are_human_authored_bound_and_stale_safe(tmp_path: Path):
    first_dir = tmp_path / "first"
    first = prepare_topology_review(BUCK, out_dir=first_dir)
    approved = _decision(first)

    assert validate_review_decision(approved, first)["decision"] == "approve"
    assert check_review_decision(approved, first_dir) == approved

    for action in ("request_changes", "reject"):
        decision = _decision(first, action)
        assert validate_review_decision(decision, first)["decision"] == action
        assert check_review_decision(
            decision,
            first_dir,
            require_approval=False,
        ) == decision
        with pytest.raises(ReviewDecisionError):
            check_review_decision(decision, first_dir)

    agent_decision = _decision(first)
    agent_decision["reviewer"] = {"kind": "agent", "id": "agent:test"}
    with pytest.raises(ReviewValidationError, match="human reviewer"):
        validate_review_decision(agent_decision, first)

    changed_model = tmp_path / "changed-buck.yaml"
    shutil.copyfile(BUCK, changed_model)
    changed_model.write_text(
        changed_model.read_text(encoding="utf-8").replace("duty: 40 %", "duty: 45 %"),
        encoding="utf-8",
    )
    current_digest = compute_topology_subject_digest(changed_model)
    with pytest.raises(StaleReviewError, match="Current model/run subject"):
        check_review_decision(
            approved,
            first_dir,
            current_subject_digest=current_digest,
        )

    second_dir = tmp_path / "second"
    second = prepare_topology_review(changed_model, out_dir=second_dir)
    with pytest.raises(StaleReviewError, match="review_id"):
        validate_review_decision(approved, second)


def test_packet_check_detects_artifact_tampering(tmp_path: Path):
    out = tmp_path / "review"
    packet = prepare_topology_review(BUCK, out_dir=out)
    source = next(item for item in packet["artifacts"] if item["role"] == "source")
    source_path = out / source["path"]
    source_path.write_text(source_path.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")

    with pytest.raises(ReviewIntegrityError, match="size mismatch|digest mismatch"):
        check_review_packet(out)


def test_prepare_result_review_binds_manifest_and_all_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    report_dir = tmp_path / "report"
    _make_run(run_dir)
    report_dir.mkdir()
    _write_json(report_dir / "report.json", {"schema": "pear-sim/report/v0.1", "sections": []})
    (report_dir / "report.html").write_text("<!doctype html><title>Review</title>", encoding="utf-8")

    out = tmp_path / "result-review"
    packet = prepare_result_review(run_dir, report_dir, out)
    assert packet["phase"] == "result"
    assert packet["subject"]["run_digest"] == f"sha256:{'c' * 64}"
    assert packet["subject"]["model_digest"] == f"sha256:{'a' * 64}"
    assert packet["subject"]["request_digest"] == f"sha256:{'b' * 64}"
    assert packet["subject"]["run_artifacts_digest"].startswith("sha256:")
    assert packet["subject"]["report_digest"].startswith("sha256:")
    assert check_review_packet(out) == packet

    decision = _decision(packet)
    assert check_review_decision(decision, out) == decision
    (run_dir / "metrics.json").write_text('{"v(out)":{"max":2.0}}\n', encoding="utf-8")
    changed_digest = compute_result_subject_digest(run_dir, report_dir)
    with pytest.raises(StaleReviewError, match="Current model/run subject"):
        check_review_decision(
            decision,
            out,
            current_subject_digest=changed_digest,
        )


def test_annotations_are_append_only_and_hash_chained(tmp_path: Path):
    out = tmp_path / "review"
    prepare_topology_review(BUCK, out_dir=out)
    first = append_review_annotation(
        out,
        {
            "event": "comment",
            "author": {"kind": "human", "id": "reviewer:test"},
            "target": {"kind": "model_component", "component_id": "d1", "port": "p"},
            "severity": "blocker",
            "category": "polarity",
            "message": "Verify the freewheel diode orientation.",
        },
    )
    second = append_review_annotation(
        out,
        {
            "event": "resolve",
            "author": {"kind": "agent", "id": "agent:test"},
            "target": {"kind": "model_component", "component_id": "d1"},
            "severity": "info",
            "category": "polarity",
            "message": "Orientation checked against the declared current convention.",
            "in_reply_to": first["id"],
        },
    )

    records = validate_review_annotations(out)
    assert [item["sequence"] for item in records] == [1, 2]
    assert second["previous_digest"] == first["annotation_digest"]
    assert check_review_packet(out)["status"] == "pending"

    log = out / "annotations.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[0])
    tampered["message"] = "silently changed"
    lines[0] = json.dumps(tampered, sort_keys=True)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ReviewIntegrityError, match="digest mismatch"):
        validate_review_annotations(out)
