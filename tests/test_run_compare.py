from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
import pytest

from pear_sim.contracts import sha256_file, sha256_json
from pear_sim.run_compare import (
    COMPARE_PROFILE_SCHEMA,
    RUN_COMPARISON_SCHEMA,
    RunComparisonError,
    compare_runs,
)


ROOT = Path(__file__).parents[1]


def _physics(*, resistor_p: str = "out") -> dict[str, Any]:
    return {
        "schema": "pear-sim/physics-model/v0.1",
        "ground": "gnd",
        "components": [
            {
                "id": "load",
                "type": "resistor",
                "p": resistor_p,
                "n": "gnd",
                "value_si": 10.0,
                "dimension": "resistance",
            }
        ],
        "controls": [],
    }


def _write_bundle(
    directory: Path,
    *,
    time: list[float] | None = None,
    values: list[float] | None = None,
    events: list[dict[str, Any]] | None = None,
    physics: dict[str, Any] | None = None,
    output: str = "v(out)",
) -> Path:
    directory.mkdir()
    time = [0.0, 0.5, 1.0, 1.5, 2.0] if time is None else time
    values = list(time) if values is None else values
    events = [] if events is None else events
    physics = _physics() if physics is None else physics
    output_step = time[1] - time[0]
    request = {
        "schema": "pear-sim/simulation-request/v0.1",
        "analysis": "transient",
        "start_s": time[0],
        "stop_s": time[-1],
        "output_step_s": output_step,
        "solver": "dopri5",
        "rtol": 1e-8,
        "atol": 1e-10,
        "max_step_s": None,
        "outputs": [output],
    }
    hashes = {
        "source_sha256": "a" * 64,
        "model_sha256": sha256_json(physics),
        "request_sha256": sha256_json(request),
    }
    conventions = {
        "component_voltage": {
            "definition": "v(component)=V(p)-V(n)",
            "positive_terminal": "p",
            "negative_terminal": "n",
        },
        "component_current": {
            "definition": "i(component)>0 means current flows p->n",
            "positive_direction": "p->n",
        },
    }
    resolved = {
        "schema": "pear-sim/resolved-model/v0.1",
        "source_schema": "pear-sim/v0.1",
        "name": "fixture",
        "source_file": "fixture.yaml",
        "hashes": hashes,
        "parameters": {},
        "conventions": conventions,
        "initial_conditions": {},
        "physics": physics,
        "request": request,
    }
    (directory / "model.resolved.json").write_text(
        json.dumps(resolved, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (directory / "waveforms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", output])
        writer.writerows(zip(time, values, strict=True))
    with (directory / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, allow_nan=False, sort_keys=True) + "\n")
    run = {
        "schema": "pear-sim/run/v0.1",
        "pear_sim_version": "0.2.test",
        "maturity": "experimental",
        "backend": "reference-python",
        "model_schema": "pear-sim/v0.1",
        "model": "fixture",
        "model_file": "fixture.yaml",
        **hashes,
        "run_sha256": sha256_json(
            {
                "model_sha256": hashes["model_sha256"],
                "request_sha256": hashes["request_sha256"],
                "backend": "reference-python",
                "pear_sim_version": "0.2.test",
            }
        ),
        "solver": "dopri5",
        "rtol": 1e-8,
        "atol": 1e-10,
        "parameters": {},
        "outputs": [output],
        "platform": {"system": "TestOS", "machine": "test", "python": "3.11"},
        "stats": {"samples": len(time)},
        "files": {
            "resolved_model": "model.resolved.json",
            "waveforms": "waveforms.csv",
            "events": "events.jsonl",
        },
    }
    (directory / "run.json").write_text(
        json.dumps(run, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_manifest(directory)
    return directory


def _rehash_manifest(directory: Path) -> None:
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    names = sorted(["run.json", *run["files"].values()])
    manifest = {
        "schema": "pear-sim/artifact-manifest/v0.1",
        "subject": {"kind": "run", "sha256": run["run_sha256"]},
        "artifacts": [
            {
                "path": name,
                "sha256": sha256_file(directory / name),
                "bytes": (directory / name).stat().st_size,
            }
            for name in names
        ],
    }
    (directory / "artifact-manifest.json").write_text(
        json.dumps(manifest, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8"
    )


def _profile(
    *,
    mode: str = "absolute",
    event_tolerance: float = 0.0,
    minimum_coverage: float = 0.4,
    absolute_tolerance: float = 1e-12,
    relative_tolerance: float = 0.0,
    nrmse_tolerance: float = 1e-12,
) -> dict[str, Any]:
    return {
        "schema": COMPARE_PROFILE_SCHEMA,
        "id": "test-profile",
        "alignment": {
            "mode": mode,
            "event_time_tolerance_s": event_tolerance,
        },
        "edge_exclusion": {"before_s": 0.0, "after_s": 0.0},
        "coverage": {"minimum_fraction": minimum_coverage},
        "signals": [
            {
                "name": "v(out)",
                "unit": "V",
                "tolerances": {
                    "max_absolute_error": {
                        "absolute": absolute_tolerance,
                        "relative": relative_tolerance,
                    },
                    "mean_absolute_error": {
                        "absolute": absolute_tolerance,
                        "relative": relative_tolerance,
                    },
                    "nrmse": nrmse_tolerance,
                },
            }
        ],
    }


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _event(time: float, *, target: str = "pwm") -> dict[str, Any]:
    return {
        "time": time,
        "kind": "scheduled_gate",
        "target": target,
        "previous": True,
        "current": False,
    }


def test_identical_verified_runs_pass_and_emit_hash_bound_offline_bundle(
    tmp_path: Path,
) -> None:
    reference = _write_bundle(tmp_path / "reference")
    candidate = _write_bundle(tmp_path / "candidate")
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(_profile()) + "\n", encoding="utf-8")

    artifacts = compare_runs(reference, candidate, profile_path, tmp_path / "comparison")

    assert artifacts.status == "pass"
    comparison = _load(artifacts.comparison_json)
    assert comparison["schema"] == RUN_COMPARISON_SCHEMA
    assert comparison["status"] == "pass"
    assert comparison["signals"][0]["metrics"]["maximum_absolute_error"] == 0.0
    assert comparison["sources"]["reference"]["artifact_manifest"]["verified"] is True
    resolved_profile = _load(artifacts.resolved_profile)
    assert resolved_profile["source"]["sha256"] == sha256_file(profile_path)
    profile_semantic = {
        key: value for key, value in resolved_profile.items() if key not in {"source", "sha256"}
    }
    assert resolved_profile["sha256"] == sha256_json(profile_semantic)

    profile_schema = _load(ROOT / "schema" / "compare-profile.schema.json")
    comparison_schema = _load(ROOT / "schema" / "run-comparison.schema.json")
    Draft202012Validator(profile_schema).validate(_profile())
    Draft202012Validator(profile_schema).validate(resolved_profile)
    Draft202012Validator(comparison_schema).validate(comparison)

    page = artifacts.report_html.read_text(encoding="utf-8")
    assert "script-src 'none'" in page
    assert "<script" not in page.lower()
    assert 'class="status status-pass"' in page
    assert "overflow-wrap: anywhere" in page
    assert "http://" not in page and "https://" not in page
    manifest = _load(artifacts.artifact_manifest)
    assert manifest["subject"]["sha256"] == sha256_json(comparison)
    assert [record["path"] for record in manifest["artifacts"]] == [
        "compare-profile.resolved.json",
        "comparison.json",
        "report.html",
    ]
    for record in manifest["artifacts"]:
        path = artifacts.output_dir / record["path"]
        assert record["sha256"] == sha256_file(path)
        assert record["bytes"] == path.stat().st_size


def test_piecewise_linear_error_integrals_are_exact_and_fail_closed(tmp_path: Path) -> None:
    reference = _write_bundle(tmp_path / "reference")
    candidate = _write_bundle(tmp_path / "candidate", values=[0.1, 0.6, 1.1, 1.6, 2.1])

    artifacts = compare_runs(
        reference,
        candidate,
        _profile(absolute_tolerance=0.099, nrmse_tolerance=0.08),
        tmp_path / "comparison",
    )

    comparison = _load(artifacts.comparison_json)
    signal = comparison["signals"][0]
    assert artifacts.status == "fail"
    assert signal["status"] == "fail"
    assert signal["metrics"]["maximum_absolute_error"] == pytest.approx(0.1)
    assert signal["metrics"]["mean_absolute_error"] == pytest.approx(0.1)
    assert signal["metrics"]["root_mean_square_error"] == pytest.approx(0.1)
    assert signal["metrics"]["reference_rms"] == pytest.approx(math.sqrt(4.0 / 3.0))
    assert signal["metrics"]["nrmse"] == pytest.approx(0.1 / math.sqrt(4.0 / 3.0))
    assert {gate["status"] for gate in signal["gates"]} == {"pass", "fail"}


def test_relative_tolerance_uses_recorded_reference_scale(tmp_path: Path) -> None:
    reference = _write_bundle(tmp_path / "reference", values=[10.0] * 5)
    candidate = _write_bundle(tmp_path / "candidate", values=[10.5] * 5)

    artifacts = compare_runs(
        reference,
        candidate,
        _profile(
            absolute_tolerance=0.0,
            relative_tolerance=0.05,
            nrmse_tolerance=0.05,
        ),
        tmp_path / "comparison",
    )

    signal = _load(artifacts.comparison_json)["signals"][0]
    assert artifacts.status == "pass"
    assert signal["metrics"]["reference_maximum_absolute"] == 10.0
    assert all(gate["status"] == "pass" for gate in signal["gates"])


def test_matching_events_piecewise_aligns_without_hiding_event_timing_gate(
    tmp_path: Path,
) -> None:
    reference = _write_bundle(
        tmp_path / "reference",
        events=[_event(1.0)],
    )
    candidate = _write_bundle(
        tmp_path / "candidate",
        time=[0.0, 0.5, 1.0, 1.5, 2.0],
        values=[0.0, 5.0 / 11.0, 10.0 / 11.0, 13.0 / 9.0, 2.0],
        events=[_event(1.1)],
    )

    passing = compare_runs(
        reference,
        candidate,
        _profile(mode="matching_events_piecewise", event_tolerance=0.11),
        tmp_path / "passing",
    )
    failing = compare_runs(
        reference,
        candidate,
        _profile(mode="matching_events_piecewise", event_tolerance=0.09),
        tmp_path / "failing",
    )

    pass_payload = _load(passing.comparison_json)
    fail_payload = _load(failing.comparison_json)
    assert passing.status == "pass"
    assert pass_payload["signals"][0]["metrics"]["maximum_absolute_error"] < 1e-14
    assert pass_payload["alignment"]["event_timing_gate"]["actual"] == pytest.approx(0.1)
    assert failing.status == "fail"
    assert fail_payload["signals"][0]["status"] == "pass"
    assert fail_payload["alignment"]["event_timing_gate"]["status"] == "fail"


def test_piecewise_alignment_requires_an_interior_semantic_event_anchor(
    tmp_path: Path,
) -> None:
    reference = _write_bundle(
        tmp_path / "reference",
        time=[0.0, 1.0, 2.0],
        values=[0.0, 0.5, 1.0],
    )
    candidate = _write_bundle(
        tmp_path / "candidate",
        time=[0.0, 2.0, 4.0],
        values=[0.0, 0.5, 1.0],
    )

    artifacts = compare_runs(
        reference,
        candidate,
        _profile(mode="matching_events_piecewise", minimum_coverage=0.1),
        tmp_path / "comparison",
    )
    comparison = _load(artifacts.comparison_json)

    assert artifacts.status == "not_comparable"
    assert comparison["alignment"]["event_timing_gate"]["status"] == "not_comparable"
    assert comparison["alignment"]["event_timing_gate"]["matched_group_count"] == 0
    assert comparison["signals"][0]["status"] == "not_comparable"


def test_piecewise_alignment_never_uses_endpoint_scaling_as_a_free_time_fit(
    tmp_path: Path,
) -> None:
    reference = _write_bundle(
        tmp_path / "reference",
        time=[0.0, 1.0, 2.0],
        values=[0.0, 1.0, 2.0],
        events=[_event(1.0)],
    )
    candidate = _write_bundle(
        tmp_path / "candidate",
        time=[0.0, 1.0, 2.0, 3.0, 4.0],
        values=[0.0, 1.0, 4.0 / 3.0, 5.0 / 3.0, 2.0],
        events=[_event(1.0)],
    )

    artifacts = compare_runs(
        reference,
        candidate,
        _profile(mode="matching_events_piecewise", minimum_coverage=0.1),
        tmp_path / "comparison",
    )
    comparison = _load(artifacts.comparison_json)

    assert artifacts.status == "not_comparable"
    assert comparison["alignment"]["event_timing_gate"]["status"] == "not_comparable"
    assert comparison["signals"][0]["status"] == "not_comparable"


def test_event_semantics_are_exact_and_never_fuzzy_matched(tmp_path: Path) -> None:
    reference = _write_bundle(tmp_path / "reference", events=[_event(1.0)])
    candidate = _write_bundle(tmp_path / "candidate", events=[_event(1.0, target="almost-pwm")])

    artifacts = compare_runs(
        reference,
        candidate,
        _profile(mode="matching_events_piecewise", event_tolerance=1.0),
        tmp_path / "comparison",
    )

    comparison = _load(artifacts.comparison_json)
    assert artifacts.status == "not_comparable"
    assert comparison["compatibility"]["gates"][-1]["id"] == "event_semantics"
    assert comparison["compatibility"]["gates"][-1]["status"] == "not_comparable"
    assert comparison["signals"][0]["metrics"] is None


@pytest.mark.parametrize("difference", ["topology", "polarity", "output"])
def test_physics_signal_and_polarity_contracts_are_exact(tmp_path: Path, difference: str) -> None:
    reference = _write_bundle(tmp_path / "reference")
    candidate = _write_bundle(
        tmp_path / "candidate",
        physics=_physics(resistor_p="other") if difference == "topology" else None,
        output="i(load)" if difference == "output" else "v(out)",
    )
    if difference == "polarity":
        resolved_path = candidate / "model.resolved.json"
        resolved = _load(resolved_path)
        resolved["conventions"]["component_voltage"]["definition"] = "changed"
        resolved_path.write_text(json.dumps(resolved, sort_keys=True) + "\n", encoding="utf-8")
        _rehash_manifest(candidate)

    profile = _profile()
    if difference == "output":
        # The profile remains exact and does not guess that i(load) corresponds to v(out).
        pass
    artifacts = compare_runs(reference, candidate, profile, tmp_path / "comparison")

    comparison = _load(artifacts.comparison_json)
    assert artifacts.status == "not_comparable"
    assert comparison["compatibility"]["status"] == "not_comparable"
    assert comparison["summary"]["signal_status_counts"]["not_comparable"] == 1


def test_insufficient_coverage_is_not_comparable_not_pass(tmp_path: Path) -> None:
    reference = _write_bundle(tmp_path / "reference", events=[_event(1.0)])
    candidate = _write_bundle(tmp_path / "candidate", events=[_event(1.0)])

    artifacts = compare_runs(
        reference,
        candidate,
        _profile(minimum_coverage=0.6),
        tmp_path / "comparison",
    )

    signal = _load(artifacts.comparison_json)["signals"][0]
    assert artifacts.status == "not_comparable"
    assert signal["gates"] == [
        {
            "id": "coverage",
            "status": "not_comparable",
            "metric": "included_time_fraction",
            "actual": pytest.approx(0.5),
            "limit": 0.6,
            "operator": ">=",
        }
    ]


def test_absolute_coverage_uses_selected_reference_window_not_domain_intersection(
    tmp_path: Path,
) -> None:
    reference = _write_bundle(
        tmp_path / "reference",
        time=[0.0, 0.5, 1.0, 1.5, 2.0],
        values=[0.0, 0.5, 1.0, 1.5, 2.0],
    )
    candidate = _write_bundle(
        tmp_path / "candidate",
        time=[1.0, 1.25, 1.5],
        values=[1.0, 1.25, 1.5],
    )

    artifacts = compare_runs(
        reference,
        candidate,
        _profile(mode="absolute", minimum_coverage=0.4),
        tmp_path / "comparison",
    )
    comparison = _load(artifacts.comparison_json)

    assert artifacts.status == "not_comparable"
    assert comparison["alignment"]["window"] == {
        "start_s": 0.0,
        "stop_s": 2.0,
        "duration_s": 2.0,
    }
    assert comparison["summary"]["coverage_fraction"] == pytest.approx(0.25)
    coverage_gate = comparison["signals"][0]["gates"][0]
    assert coverage_gate["id"] == "coverage"
    assert coverage_gate["status"] == "not_comparable"
    assert coverage_gate["actual"] == pytest.approx(0.25)
    assert coverage_gate["limit"] == 0.4


def test_zero_reference_with_nonzero_error_cannot_claim_nrmse_pass(tmp_path: Path) -> None:
    reference = _write_bundle(tmp_path / "reference", values=[0.0] * 5)
    candidate = _write_bundle(tmp_path / "candidate", values=[1e-9] * 5)

    artifacts = compare_runs(
        reference,
        candidate,
        _profile(absolute_tolerance=1.0, nrmse_tolerance=1.0),
        tmp_path / "comparison",
    )

    signal = _load(artifacts.comparison_json)["signals"][0]
    assert artifacts.status == "not_comparable"
    assert signal["metrics"]["nrmse"] is None
    assert signal["gates"][-1]["status"] == "not_comparable"


def test_manifest_tampering_missing_coverage_and_semantic_hashes_are_rejected(
    tmp_path: Path,
) -> None:
    candidate = _write_bundle(tmp_path / "candidate")

    tampered = _write_bundle(tmp_path / "tampered")
    (tampered / "waveforms.csv").write_text("time,v(out)\n0,999\n1,999\n", encoding="utf-8")
    with pytest.raises(RunComparisonError, match="SHA-256 mismatch"):
        compare_runs(tampered, candidate, _profile(), tmp_path / "out-tampered")

    incomplete = _write_bundle(tmp_path / "incomplete")
    manifest = _load(incomplete / "artifact-manifest.json")
    manifest["artifacts"] = manifest["artifacts"][:-1]
    (incomplete / "artifact-manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    with pytest.raises(RunComparisonError, match="coverage mismatch"):
        compare_runs(incomplete, candidate, _profile(), tmp_path / "out-incomplete")

    semantic = _write_bundle(tmp_path / "semantic")
    run = _load(semantic / "run.json")
    run["request_sha256"] = "f" * 64
    (semantic / "run.json").write_text(json.dumps(run) + "\n", encoding="utf-8")
    _rehash_manifest(semantic)
    with pytest.raises(RunComparisonError, match="resolved request_sha256"):
        compare_runs(semantic, candidate, _profile(), tmp_path / "out-semantic")

    truncated = _write_bundle(tmp_path / "truncated")
    with (truncated / "waveforms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "v(out)"])
        writer.writerows([(0.5, 0.5), (1.0, 1.0), (1.5, 1.5)])
    _rehash_manifest(truncated)
    with pytest.raises(RunComparisonError, match="waveform domain"):
        compare_runs(truncated, candidate, _profile(), tmp_path / "out-truncated")

    sparse = _write_bundle(tmp_path / "sparse")
    with (sparse / "waveforms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "v(out)"])
        writer.writerows([(0.0, 0.0), (2.0, 2.0)])
    sparse_run = _load(sparse / "run.json")
    sparse_run["stats"]["samples"] = 2
    (sparse / "run.json").write_text(
        json.dumps(sparse_run, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rehash_manifest(sparse)
    with pytest.raises(RunComparisonError, match="sample grid"):
        compare_runs(sparse, candidate, _profile(), tmp_path / "out-sparse")

    tiny_step = _write_bundle(tmp_path / "tiny-step")
    tiny_resolved = _load(tiny_step / "model.resolved.json")
    tiny_resolved["request"]["output_step_s"] = 1e-15
    tiny_request_sha256 = sha256_json(tiny_resolved["request"])
    tiny_resolved["hashes"]["request_sha256"] = tiny_request_sha256
    (tiny_step / "model.resolved.json").write_text(
        json.dumps(tiny_resolved, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tiny_run = _load(tiny_step / "run.json")
    tiny_run["request_sha256"] = tiny_request_sha256
    tiny_run["run_sha256"] = sha256_json(
        {
            "model_sha256": tiny_run["model_sha256"],
            "request_sha256": tiny_request_sha256,
            "backend": tiny_run["backend"],
            "pear_sim_version": tiny_run["pear_sim_version"],
        }
    )
    (tiny_step / "run.json").write_text(
        json.dumps(tiny_run, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _rehash_manifest(tiny_step)
    with pytest.raises(RunComparisonError, match="row contract limit"):
        compare_runs(tiny_step, candidate, _profile(), tmp_path / "out-tiny-step")


def test_symlink_and_path_escape_are_rejected(tmp_path: Path) -> None:
    candidate = _write_bundle(tmp_path / "candidate")
    symlinked = _write_bundle(tmp_path / "symlinked")
    waveform = symlinked / "waveforms.csv"
    external = tmp_path / "external.csv"
    external.write_bytes(waveform.read_bytes())
    waveform.unlink()
    waveform.symlink_to(external)
    _rehash_manifest(symlinked)
    with pytest.raises(RunComparisonError, match="symlink"):
        compare_runs(symlinked, candidate, _profile(), tmp_path / "out-symlink")

    escaped = _write_bundle(tmp_path / "escaped")
    run = _load(escaped / "run.json")
    run["files"]["waveforms"] = "../external.csv"
    (escaped / "run.json").write_text(json.dumps(run) + "\n", encoding="utf-8")
    manifest = _load(escaped / "artifact-manifest.json")
    manifest["artifacts"] = []
    (escaped / "artifact-manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    with pytest.raises(RunComparisonError, match="safe normalized relative path"):
        compare_runs(escaped, candidate, _profile(), tmp_path / "out-escape")


@pytest.mark.parametrize(
    ("artifact", "replacement", "message"),
    [
        ("waveforms.csv", "time,v(out)\n0,0\n1,nan\n", "non-finite"),
        (
            "events.jsonl",
            '{"time":NaN,"kind":"x","target":"x","previous":false,"current":true}\n',
            "invalid JSON",
        ),
        (
            "events.jsonl",
            '{"time":1,"time":1,"kind":"x","target":"x","previous":false,"current":true}\n',
            "duplicate JSON object key",
        ),
    ],
)
def test_nonfinite_csv_and_strict_event_json_are_rejected(
    tmp_path: Path, artifact: str, replacement: str, message: str
) -> None:
    source = _write_bundle(tmp_path / "source")
    candidate = _write_bundle(tmp_path / "candidate")
    (source / artifact).write_text(replacement, encoding="utf-8")
    _rehash_manifest(source)

    with pytest.raises(RunComparisonError, match=message):
        compare_runs(source, candidate, _profile(), tmp_path / "comparison")


def test_profile_is_closed_unit_strict_and_duplicate_free(tmp_path: Path) -> None:
    reference = _write_bundle(tmp_path / "reference")
    candidate = _write_bundle(tmp_path / "candidate")

    unknown = _profile()
    unknown["surprise"] = True
    with pytest.raises(RunComparisonError, match="unknown fields"):
        compare_runs(reference, candidate, unknown, tmp_path / "unknown")

    wrong_unit = _profile()
    wrong_unit["signals"][0]["unit"] = "A"
    with pytest.raises(RunComparisonError, match="must be 'V'"):
        compare_runs(reference, candidate, wrong_unit, tmp_path / "wrong-unit")

    duplicate = _profile()
    duplicate["signals"].append(duplicate["signals"][0].copy())
    with pytest.raises(RunComparisonError, match="repeats"):
        compare_runs(reference, candidate, duplicate, tmp_path / "duplicate")

    empty_window = _profile()
    empty_window["window"] = {}
    with pytest.raises(RunComparisonError, match="window must declare"):
        compare_runs(reference, candidate, empty_window, tmp_path / "empty-window")

    invalid_signal = _write_bundle(
        tmp_path / "invalid-signal",
        output="v(out space)",
    )
    with pytest.raises(RunComparisonError, match="unsupported signal expression"):
        compare_runs(invalid_signal, candidate, _profile(), tmp_path / "invalid-signal-out")


def test_output_is_non_overwriting_and_outside_immutable_sources(tmp_path: Path) -> None:
    reference = _write_bundle(tmp_path / "reference")
    candidate = _write_bundle(tmp_path / "candidate")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    marker = occupied / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(RunComparisonError, match="never overwritten"):
        compare_runs(reference, candidate, _profile(), occupied)
    assert marker.read_text(encoding="utf-8") == "keep"

    with pytest.raises(RunComparisonError, match="outside both immutable run trees"):
        compare_runs(reference, candidate, _profile(), reference / "comparison")
    assert not (reference / "comparison").exists()


def test_profile_window_must_intersect_and_empty_domain_is_retained_not_comparable(
    tmp_path: Path,
) -> None:
    reference = _write_bundle(tmp_path / "reference")
    candidate = _write_bundle(tmp_path / "candidate")
    profile = _profile()
    profile["window"] = {"start_s": 3.0, "stop_s": 4.0}

    artifacts = compare_runs(reference, candidate, profile, tmp_path / "comparison")

    comparison = _load(artifacts.comparison_json)
    assert artifacts.status == "not_comparable"
    assert comparison["alignment"]["window"]["status"] == "not_comparable"
    Draft202012Validator(_load(ROOT / "schema" / "run-comparison.schema.json")).validate(comparison)
