# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import pear_sim.report as report_module
from pear_sim.io import write_result
from pear_sim.model import load_model
from pear_sim.report import ReportError, generate_report
from pear_sim.solver import HybridSimulator, SimulationResult
from pear_sim.three_phase_modulation import (
    ThreePhaseModulationError,
    build_three_phase_modulation_observation,
)


ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "inverter_three_phase_spwm.yaml"


@pytest.fixture(scope="module")
def spwm_result() -> SimulationResult:
    return HybridSimulator(load_model(EXAMPLE)).simulate()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _refresh_manifest(run_dir: Path, artifact_name: str) -> None:
    manifest_path = run_dir / "artifact-manifest.json"
    manifest = _read_json(manifest_path)
    artifact_path = run_dir / artifact_name
    payload = artifact_path.read_bytes()
    record = next(item for item in manifest["artifacts"] if item["path"] == artifact_name)
    record["bytes"] = len(payload)
    record["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_spwm_run_writes_distinct_command_evidence_without_fixed_gate_drive(
    tmp_path: Path,
    spwm_result: SimulationResult,
) -> None:
    first = write_result(spwm_result, tmp_path / "run-a")
    second = write_result(spwm_result, tmp_path / "run-b")

    for name in ("modulation-cycles.jsonl", "three-phase-modulation.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    assert not (first / "gate-drive.json").exists()

    run = _read_json(first / "run.json")
    assert "gate_drive" not in run["files"]
    assert run["files"]["modulation_cycles"] == "modulation-cycles.jsonl"
    assert run["files"]["three_phase_modulation"] == "three-phase-modulation.json"
    lines = (first / "modulation-cycles.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(spwm_result.modulation_cycles)
    assert all(
        json.loads(line)["schema"] == "pear-sim/modulation-cycle/three-phase-spwm/v0.1"
        for line in lines
    )

    observation = _read_json(first / "three-phase-modulation.json")
    assert observation["schema"] == ("pear-sim/three-phase-modulation-observation/v0.1")
    assert observation["claim"]["validation_claim"] is False
    assert observation["claim"]["hardware_sign_off_allowed"] is False
    assert observation["source_binding"]["run_sha256"] == run["run_sha256"]
    assert observation["summary"]["source_cycle_count"] == len(lines)
    assert observation["summary"]["observed_overlap_count"] == 0
    block = observation["blocks"][0]
    assert block["config"]["control_type"] == "three_phase_spwm"
    assert block["config"]["sampling"] == "offline_midpoint_sampled_center_aligned"
    assert block["config"]["phase_sequence"] == "abc"
    assert block["bridge"]["positive_rail_node"] == "dc_plus"
    assert block["bridge"]["negative_rail_node"] == "dc_minus"
    assert {
        row["phase"]: (
            row["high"]["gate_output_id"],
            row["high"]["device_id"],
            row["low"]["gate_output_id"],
            row["low"]["device_id"],
        )
        for row in block["bridge"]["phases"]
    } == {
        "a": ("bridge.a.high", "qah", "bridge.a.low", "qal"),
        "b": ("bridge.b.high", "qbh", "bridge.b.low", "qbl"),
        "c": ("bridge.c.high", "qch", "bridge.c.low", "qcl"),
    }
    projection = block["fundamental_period_projection"]
    retention = projection["retention"]
    assert retention["retained_cycles"] == len(projection["cycles"])
    assert retention["source_cycles_for_block"] == len(lines)
    assert retention["truncated"] is True
    assert "modulation-cycles.jsonl remains authoritative" in retention["statement"]
    assert block["summary"]["minimum_effective_pulse_s"] > 0.0

    manifest = _read_json(first / "artifact-manifest.json")
    paths = {item["path"] for item in manifest["artifacts"]}
    assert {"modulation-cycles.jsonl", "three-phase-modulation.json"} <= paths


def test_modulation_observation_fails_closed_on_bridge_and_overlap_mutations(
    spwm_result: SimulationResult,
) -> None:
    components = tuple(
        replace(component, p="wrong_pole") if component.id == "qbl" else component
        for component in spwm_result.model.components
    )
    wrong_bridge = replace(
        spwm_result,
        model=replace(spwm_result.model, components=components),
    )
    with pytest.raises(ThreePhaseModulationError, match="oriented|pole"):
        build_three_phase_modulation_observation(wrong_bridge)

    intervals = deepcopy(spwm_result.mode_intervals)
    intervals[0]["gates"]["bridge.a.high"] = True
    intervals[0]["gates"]["bridge.a.low"] = True
    overlap = replace(spwm_result, mode_intervals=intervals)
    with pytest.raises(ThreePhaseModulationError, match="overlap"):
        build_three_phase_modulation_observation(overlap)

    controls = {
        control_id: replace(control, normalized_phase=control.normalized_phase + 1.0e-9)
        if control.kind == "three_phase_spwm"
        else control
        for control_id, control in spwm_result.model.controls.items()
    }
    wrong_phase = replace(
        spwm_result,
        model=replace(spwm_result.model, controls=controls),
    )
    with pytest.raises(ThreePhaseModulationError, match="normalized_carrier_phase"):
        build_three_phase_modulation_observation(wrong_phase)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_middle", "not contiguous"),
        ("frame_index", "frame_index disagrees"),
        ("sample", "sample_s disagrees"),
        ("reference", "references.a disagrees"),
        ("duty", "duties.b disagrees"),
        ("pulse", "command_pulses.c.start_s disagrees"),
    ],
)
def test_modulation_cycles_fail_closed_on_deterministic_contract_mutations(
    spwm_result: SimulationResult,
    mutation: str,
    match: str,
) -> None:
    cycles = deepcopy(spwm_result.modulation_cycles)
    middle = len(cycles) // 2
    if mutation == "missing_middle":
        del cycles[middle]
    elif mutation == "frame_index":
        cycles[middle]["frame_index"] += 1
    elif mutation == "sample":
        carrier_frequency = spwm_result.model.controls["bridge.a.high"].frequency
        cycles[middle]["sample_s"] += 0.1 / carrier_frequency
    elif mutation == "reference":
        cycles[middle]["references"]["a"] += 1.0e-3
    elif mutation == "duty":
        cycles[middle]["duties"]["b"] += 1.0e-3
    elif mutation == "pulse":
        cycles[middle]["command_pulses"]["c"]["start_s"] += 1.0e-9
    else:  # pragma: no cover - the parameter table is closed above.
        raise AssertionError(mutation)

    mutated = replace(spwm_result, modulation_cycles=cycles)
    with pytest.raises(ThreePhaseModulationError, match=match):
        build_three_phase_modulation_observation(mutated)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing", "event count disagrees"),
        ("retarget", "target/state disagrees"),
        ("shift", "event .*time disagrees"),
        ("previous", "previous state disagrees"),
        ("extra", "event count disagrees"),
    ],
)
def test_modulation_evidence_binds_commands_to_exact_realized_gate_events(
    spwm_result: SimulationResult,
    mutation: str,
    match: str,
) -> None:
    events = list(spwm_result.events)
    scheduled_indices = [
        index
        for index, event in enumerate(events)
        if event.kind == "scheduled_gate" and event.target.startswith("bridge.")
    ]
    selected = scheduled_indices[len(scheduled_indices) // 2]
    if mutation == "missing":
        del events[selected]
    elif mutation == "retarget":
        replacement_target = (
            "bridge.b.low" if events[selected].target != "bridge.b.low" else "bridge.c.low"
        )
        events[selected] = replace(events[selected], target=replacement_target)
    elif mutation == "shift":
        events[selected] = replace(events[selected], time=events[selected].time + 1.0e-9)
    elif mutation == "previous":
        events[selected] = replace(events[selected], previous=not events[selected].previous)
    elif mutation == "extra":
        events.append(events[selected])
    else:  # pragma: no cover - the parameter table is closed above.
        raise AssertionError(mutation)

    mutated = replace(spwm_result, events=events)
    with pytest.raises(ThreePhaseModulationError, match=match):
        build_three_phase_modulation_observation(mutated)


def test_report_renders_bounded_escaped_modulation_evidence_and_detects_tampering(
    tmp_path: Path,
    spwm_result: SimulationResult,
) -> None:
    run_dir = write_result(spwm_result, tmp_path / "run")
    observation_path = run_dir / "three-phase-modulation.json"
    observation = _read_json(observation_path)
    hostile = "bridge<script>alert('phase')</script>&"
    observation["claim"]["statement"] = hostile
    observation["blocks"][0]["bridge"]["phases"][0]["high"]["device_id"] = (
        "<img src=x onerror=alert('device')>"
    )
    observation_path.write_text(
        json.dumps(observation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_manifest(run_dir, observation_path.name)

    first = generate_report(run_dir, tmp_path / "report")
    first_html = first.report_html.read_bytes()
    first_json = first.report_json.read_bytes()
    second = generate_report(run_dir, tmp_path / "report")
    assert second.report_html.read_bytes() == first_html
    assert second.report_json.read_bytes() == first_json

    report = json.loads(first_json)
    assert report["three_phase_modulation"] == observation
    assert report["modulation_cycle_source"]["row_count"] == len(spwm_result.modulation_cycles)
    assert report["modulation_cycle_source"]["retained_in_report_json"] is False
    rendered = first_html.decode("utf-8")
    lowered = rendered.lower()
    assert "Three-phase modulation command evidence" in rendered
    assert "Explicit phase → gate → device mapping" in rendered
    assert "Sampled SPWM phase references" in rendered
    assert "Commanded SPWM phase duties" in rendered
    assert "minimum_effective_pulse_in_full_run_earliest_tie" in rendered
    assert "modulation-cycles.jsonl remains authoritative" in rendered
    assert "bridge&lt;script&gt;alert(&#x27;phase&#x27;)&lt;/script&gt;&amp;" in rendered
    assert "&lt;img src=x onerror=alert(&#x27;device&#x27;)&gt;" in rendered
    assert "<script" not in lowered
    assert "<img " not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "fetch(" not in lowered

    observation_path.write_bytes(observation_path.read_bytes() + b" \n")
    with pytest.raises(
        ReportError,
        match="SHA-256 mismatch for source artifact 'three-phase-modulation.json'",
    ):
        generate_report(run_dir, tmp_path / "tampered-report")


def test_report_bounds_modulation_projection_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    spwm_result: SimulationResult,
) -> None:
    run_dir = write_result(spwm_result, tmp_path / "run")
    monkeypatch.setattr(report_module, "MAX_THREE_PHASE_MODULATION_PROJECTION_ROWS", 1)
    with pytest.raises(ReportError, match="projection-row limit"):
        generate_report(run_dir, tmp_path / "report")
