from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import pear_sim.report as report_module
from pear_sim.io import write_result
from pear_sim.model import load_model
from pear_sim.report import ReportError, generate_report
from pear_sim.solver import HybridSimulator, SimulationResult


ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "synchronous_buck_complementary.yaml"


@pytest.fixture(scope="module")
def complementary_result() -> SimulationResult:
    return HybridSimulator(load_model(EXAMPLE)).simulate()


def _write_example_variant(
    tmp_path: Path,
    *,
    dead_time: str | None = None,
    add_parallel_gate_load: bool = False,
) -> Path:
    source = EXAMPLE.read_text(encoding="utf-8")
    source = source.replace("stop: 500 us", "stop: 100 us", 1)
    if dead_time is not None:
        source = source.replace("dead_time: 500 ns", f"dead_time: {dead_time}", 1)
    if add_parallel_gate_load:
        marker = "    - {id: qh, type: ideal_switch, p: in, n: sw, control: leg.high}\n"
        assert marker in source
        source = source.replace(
            marker,
            marker
            + "    - {id: qh_aux, type: ideal_switch, p: in, n: aux, "
            "control: leg.high}\n"
            + "    - {id: aux_load, type: resistor, p: aux, n: gnd, value: 100 ohm}\n",
            1,
        )
    path = tmp_path / "synchronous-buck-variant.yaml"
    path.write_text(source, encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh_source_manifest_record(run_dir: Path, artifact_name: str) -> None:
    manifest_path = run_dir / "artifact-manifest.json"
    manifest = _read_json(manifest_path)
    artifact_path = run_dir / artifact_name
    payload = artifact_path.read_bytes()
    records = manifest["artifacts"]
    assert isinstance(records, list)
    matching = [record for record in records if record["path"] == artifact_name]
    assert len(matching) == 1
    matching[0]["sha256"] = hashlib.sha256(payload).hexdigest()
    matching[0]["bytes"] = len(payload)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_gate_drive_artifact_maps_one_output_to_every_explicit_device_and_is_deterministic(
    tmp_path: Path,
) -> None:
    model_path = _write_example_variant(tmp_path, add_parallel_gate_load=True)
    result = HybridSimulator(load_model(model_path)).simulate()

    first = write_result(result, tmp_path / "run-a")
    second = write_result(result, tmp_path / "run-b")

    assert (first / "gate-drive.json").read_bytes() == (
        second / "gate-drive.json"
    ).read_bytes()
    gate_drive = _read_json(first / "gate-drive.json")
    outputs = {
        output["gate_output_id"]: output
        for output in gate_drive["gate_outputs"]
    }
    assert outputs["leg.high"]["device_ids"] == ["qh", "qh_aux"]
    assert outputs["leg.high"]["devices"] == [
        {"device_id": "qh", "n": "sw", "p": "in"},
        {"device_id": "qh_aux", "n": "aux", "p": "in"},
    ]
    assert outputs["leg.low"]["device_ids"] == ["ql"]
    assert gate_drive["summary"]["controlled_device_count"] == 3
    assert set(gate_drive["run_gate_states"]) == {
        "boundary_convention",
        "initial",
        "final",
    }
    timeline = gate_drive["logical_controls"][0]["timeline"]
    retention = timeline["retention"]
    assert retention["source_artifact"] == "events.jsonl"
    assert retention["source_scope"] == "full_run_[start,stop]"
    assert retention["retained_atomic_transactions"] == len(
        timeline["transactions"]
    )
    assert retention["source_atomic_transactions_for_control"] > retention[
        "retained_atomic_transactions"
    ]
    assert retention["truncated"] is True
    assert "events.jsonl remains authoritative" in retention["statement"]

    run = _read_json(first / "run.json")
    assert gate_drive["source_binding"]["run_sha256"] == run["run_sha256"]
    summary = _read_json(first / "summary.json")
    source_manifest = _read_json(first / "artifact-manifest.json")
    assert run["files"]["gate_drive"] == "gate-drive.json"
    assert summary["gate_drive"]["artifact"] == "gate-drive.json"
    gate_record = next(
        record
        for record in source_manifest["artifacts"]
        if record["path"] == "gate-drive.json"
    )
    gate_bytes = (first / "gate-drive.json").read_bytes()
    assert gate_record == {
        "bytes": len(gate_bytes),
        "path": "gate-drive.json",
        "sha256": hashlib.sha256(gate_bytes).hexdigest(),
    }


def test_zero_dead_time_swap_is_one_atomic_off_before_on_transaction(
    tmp_path: Path,
) -> None:
    model_path = _write_example_variant(tmp_path, dead_time="0 s")
    run_dir = write_result(
        HybridSimulator(load_model(model_path)).simulate(),
        tmp_path / "zero-dead-time-run",
    )
    gate_drive = _read_json(run_dir / "gate-drive.json")
    control = gate_drive["logical_controls"][0]
    timeline = control["timeline"]

    assert timeline["observed_state_segment_counts"] == {
        "high_on": 1,
        "low_on": 1,
    }
    assert [segment["state"] for segment in timeline["segments"]] == [
        "high_on",
        "low_on",
    ]
    assert gate_drive["summary"]["rendered_overlap_segment_count"] == 0
    assert len(timeline["transactions"]) == 1
    transaction = timeline["transactions"][0]
    assert transaction["atomicity"] == (
        "all_changes_applied_before_topology_stabilization"
    )
    assert [
        (
            change["gate_output_id"],
            change["previous"],
            change["current"],
            change["edge_role"],
        )
        for change in transaction["changes"]
    ] == [
        ("leg.high", True, False, "turn_off"),
        ("leg.low", False, True, "turn_on"),
    ]


def test_report_renders_static_escaped_gate_drive_section_deterministically(
    tmp_path: Path,
    complementary_result: SimulationResult,
) -> None:
    run_dir = write_result(complementary_result, tmp_path / "run")
    gate_path = run_dir / "gate-drive.json"
    gate_drive = _read_json(gate_path)
    hostile_control = "leg<script>alert('gate')</script>&"
    hostile_device = "<img src=x onerror=alert('device')>"
    gate_drive["logical_controls"][0]["logical_control_id"] = hostile_control
    gate_drive["gate_outputs"][0]["logical_control_id"] = hostile_control
    gate_drive["gate_outputs"][0]["device_ids"] = [hostile_device]
    gate_path.write_text(
        json.dumps(gate_drive, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_source_manifest_record(run_dir, "gate-drive.json")

    artifacts = generate_report(run_dir, tmp_path / "report")
    first_json = artifacts.report_json.read_bytes()
    first_html = artifacts.report_html.read_bytes()
    second = generate_report(run_dir, tmp_path / "report")

    assert second.report_json.read_bytes() == first_json
    assert second.report_html.read_bytes() == first_html
    report = json.loads(first_json)
    assert report["gate_drive"] == gate_drive
    assert report["provenance"]["input_artifacts"]["gate_drive"]["file"] == (
        "gate-drive.json"
    )

    rendered = first_html.decode("utf-8")
    lowered = rendered.lower()
    assert "<h2>Gate drive &amp; switching timeline</h2>" in rendered
    assert "Gate-to-device mapping" in rendered
    assert "Atomic transitions" in rendered
    assert "Full-run atomic transactions" in rendered
    assert "Bounded timeline projection." in rendered
    assert "first_command_period_from_run_start" in rendered
    assert "truncated: <strong>true</strong>" in rendered
    assert "Raw source: <code>events.jsonl</code>" in rendered
    assert "events.jsonl remains authoritative" in rendered
    assert "RECORDED SCHEDULE — NOT A DRIVER OR SAFETY CLAIM." in rendered
    assert "leg&lt;script&gt;alert(&#x27;gate&#x27;)&lt;/script&gt;&amp;" in rendered
    assert "&lt;img src=x onerror=alert(&#x27;device&#x27;)&gt;" in rendered
    assert "<script" not in lowered
    assert "<img " not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "fetch(" not in lowered
    assert "<link" not in lowered
    # Only in-page anchors (the table of contents) may link.
    assert not __import__("re").search(r' href="(?!#[a-z0-9-]+")', lowered)


def test_report_rejects_gate_drive_tampering(
    tmp_path: Path,
    complementary_result: SimulationResult,
) -> None:
    run_dir = write_result(complementary_result, tmp_path / "run")
    gate_path = run_dir / "gate-drive.json"
    gate_path.write_bytes(gate_path.read_bytes() + b" \n")

    with pytest.raises(
        ReportError,
        match="SHA-256 mismatch for source artifact 'gate-drive.json'",
    ):
        generate_report(run_dir, tmp_path / "report")


def test_report_bounds_gate_drive_file_and_nested_timeline_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complementary_result: SimulationResult,
) -> None:
    run_dir = write_result(complementary_result, tmp_path / "run")
    output_dir = tmp_path / "report"

    monkeypatch.setattr(report_module, "MAX_GATE_DRIVE_FILE_BYTES", 10)
    with pytest.raises(ReportError, match="gate-drive limit is 10 bytes"):
        generate_report(run_dir, output_dir)

    monkeypatch.setattr(report_module, "MAX_GATE_DRIVE_FILE_BYTES", 8 * 1024 * 1024)
    monkeypatch.setattr(report_module, "MAX_GATE_DRIVE_TIMELINE_ROWS", 1)
    with pytest.raises(ReportError, match="1 gate-drive timeline-row limit"):
        generate_report(run_dir, output_dir)
