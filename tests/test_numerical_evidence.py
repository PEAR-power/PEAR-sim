from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from pear_sim.contracts import sha256_json
from pear_sim.numerical_evidence import (
    NUMERICAL_EVIDENCE_SCHEMA,
    SEMANTIC_SIGNATURE_SCHEMA,
    NumericalEvidenceProfile,
    _platform_record,
    _study_summaries,
    full_v1_profile,
    run_buck_numerical_evidence,
)
from pear_sim.solver import HybridSimulator


ROOT = Path(__file__).parents[1]
BUCK = ROOT / "examples" / "buck_dcm.yaml"
BOOST = ROOT / "examples" / "boost.yaml"


@pytest.mark.parametrize(
    "filename",
    [
        "numerical-profile.schema.json",
        "platform-evidence.schema.json",
        "cross-platform-evidence.schema.json",
    ],
)
def test_m22_json_schemas_are_versioned_finite_documents(filename: str) -> None:
    schema = json.loads((ROOT / "schema" / filename).read_text(encoding="utf-8"))

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["$id"].startswith("https://raw.githubusercontent.com/PEAR-power/pear-sim/main/schema/")
    assert schema["title"].startswith("PEARSim")


@pytest.fixture(scope="module")
def completed_evidence(tmp_path_factory: pytest.TempPathFactory):
    return run_buck_numerical_evidence(
        BUCK,
        tmp_path_factory.mktemp("numerical-evidence") / "matrix",
        source_revision="test-revision",
    )


def test_full_v1_profile_is_exact_bounded_and_deduplicated() -> None:
    profile = full_v1_profile()
    payload = profile.as_dict()
    schema = json.loads(
        (ROOT / "schema" / "numerical-profile.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(payload)

    assert len(profile.load_ratios) == 9
    assert len(profile.configurations) == 16
    assert profile.cell_count == 144
    assert {name: len(ids) for name, ids in profile.study_configuration_ids.items()} == {
        "tolerance": 5,
        "max_step": 5,
        "output_grid": 3,
        "solver": 4,
    }
    assert sum(len(ids) for ids in profile.study_configuration_ids.values()) == 17
    assert payload["sha256"] == sha256_json(
        {key: value for key, value in payload.items() if key != "sha256"}
    )

    by_id = {configuration.id: configuration for configuration in profile.configurations}
    assert [
        by_id[identifier].rtol for identifier in profile.study_configuration_ids["tolerance"]
    ] == [
        1e-8,
        1e-9,
        1e-10,
        1e-11,
        1e-12,
    ]
    assert [
        by_id[identifier].atol for identifier in profile.study_configuration_ids["tolerance"]
    ] == [
        1e-10,
        1e-11,
        1e-12,
        1e-13,
        1e-14,
    ]
    assert [
        by_id[identifier].max_step_over_period
        for identifier in profile.study_configuration_ids["max_step"]
    ] == pytest.approx([1 / 40, 1 / 60, 1 / 80, 1 / 120, 1 / 160])
    assert [
        by_id[identifier].output_step_over_period
        for identifier in profile.study_configuration_ids["output_grid"]
    ] == pytest.approx([1 / 13, 1 / 37, 1 / 200])
    assert [
        by_id[identifier].solver for identifier in profile.study_configuration_ids["solver"]
    ] == ["dopri5", "dop853", "radau", "bdf"]
    for identifier in profile.study_configuration_ids["solver"]:
        configuration = by_id[identifier]
        assert configuration.rtol == 1e-10
        assert configuration.atol == 1e-12
        assert configuration.max_step_over_period == 1 / 100
        assert configuration.output_step_over_period == 1 / 200
    signature_policy = profile.cross_platform_policy["semantic_signature"]
    assert signature_policy["schema"] == SEMANTIC_SIGNATURE_SCHEMA
    assert signature_policy["required_field_count"] == 22
    assert len(signature_policy["field_policy_ids"]) == 22


def test_runner_retains_complete_passing_matrix_and_family_evidence(completed_evidence) -> None:
    evidence = json.loads(completed_evidence.platform_evidence.read_text(encoding="utf-8"))
    schema = json.loads(
        (ROOT / "schema" / "platform-evidence.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(evidence)

    assert completed_evidence.status == "pass"
    assert evidence["schema"] == NUMERICAL_EVIDENCE_SCHEMA
    assert evidence["status"] == "pass"
    assert evidence["profile"]["id"] == "full-v1"
    assert evidence["source"]["revision"] == "test-revision"
    assert evidence["source"]["revision_method"] == "explicit"
    assert evidence["platform"]["system"]
    assert evidence["platform"]["python"]
    assert evidence["platform"]["numpy"]
    assert evidence["platform"]["scipy"]
    assert evidence["summary"] == {
        "status": "pass",
        "point_count": 9,
        "configuration_count": 16,
        "cell_count": 144,
        "gate_count": 1584,
        "family_gate_count": 36,
        "cell_status_counts": {"pass": 144},
        "gate_status_counts": {"pass": 1584},
        "family_gate_status_counts": {"pass": 36},
    }
    assert len(evidence["cells"]) == 144
    assert len({cell["id"] for cell in evidence["cells"]}) == 144
    assert len(list(completed_evidence.output_dir.glob("cells/*/*/cell.json"))) == 144
    assert len(list(completed_evidence.output_dir.glob("cells/*/*/run/run.json"))) == 144

    assert {name: study["status"] for name, study in evidence["studies"].items()} == {
        "max_step": "pass",
        "output_grid": "pass",
        "solver": "pass",
        "tolerance": "pass",
    }
    for study in evidence["studies"].values():
        assert len(study["gates"]) == 9
        assert study["family_gate_status_counts"] == {"pass": 9}
    for name in ("tolerance", "max_step"):
        for gate in evidence["studies"][name]["gates"]:
            assert gate["fixed_numeric_gate_set"] is True
            assert gate["oracle_gate_series"]
            assert gate["oracle_gate_series_status_counts"] == {
                "pass": len(gate["oracle_gate_series"])
            }
            for series in gate["oracle_gate_series"].values():
                assert series["fitted_log_log_slope"] is not None
                assert series["positive_trend"] or series["final_three_plateau"]
                assert series["status"] == "pass"
            envelope = gate["worst_envelope_summary"]
            assert envelope["acceptance_role"] == "diagnostic-only"
            assert envelope["normalized_budget_fractions"]
    for gate in evidence["studies"]["output_grid"]["gates"]:
        assert len(set(gate["sample_counts"])) == 3
        assert {check["status"] for check in gate["checks"].values()} == {"pass"}
    for gate in evidence["studies"]["solver"]["gates"]:
        assert gate["checks"]["semantic_event_sequence"]["status"] == "pass"
        assert gate["checks"]["accepted_mode_sequence"]["status"] == "pass"
        assert gate["checks"]["recorded_semantic_signature"]["status"] == "pass"

    expected_signature_ids = set(
        evidence["cross_platform_policy"]["semantic_signature"]["field_policy_ids"]
    )
    for cell in evidence["cells"]:
        assert cell["status"] == "pass"
        assert cell["run"] is not None
        assert cell["numerics"]["sample_count"] > 1
        assert cell["metrics"]["fraction_by_mode"] is not None
        signature = cell["semantic_signature"]
        assert signature["schema"] == SEMANTIC_SIGNATURE_SCHEMA
        assert signature["status"] == "available"
        assert set(signature["fields"]) == expected_signature_ids
        for field_id, field in signature["fields"].items():
            assert field["quantity"]
            assert field["unit"]
            assert (
                field["policy_id"]
                == evidence["cross_platform_policy"]["semantic_signature"]["field_policy_ids"][
                    field_id
                ]
            )
            assert field["cross_platform_tolerance"]["policy_id"] == field["policy_id"]
        for gate in cell["gates"]:
            assert "value" not in gate
            assert "cross_platform_tolerance" not in gate


def test_root_manifest_hashes_platform_evidence_and_nested_runs(completed_evidence) -> None:
    manifest = json.loads(completed_evidence.artifact_manifest.read_text(encoding="utf-8"))
    records = manifest["artifacts"]
    by_path = {record["path"]: record for record in records}

    assert (
        manifest["subject"]["sha256"]
        == hashlib.sha256(completed_evidence.platform_evidence.read_bytes()).hexdigest()
    )
    assert "platform-evidence.json" in by_path
    assert "numerical-profile.json" in by_path
    assert "oracle.json" in by_path
    assert "report/numerical-report.html" in by_path
    assert "report/numerical-report.json" in by_path
    assert "report/artifact-manifest.json" in by_path
    assert completed_evidence.report_html.is_file()
    assert completed_evidence.report_json.is_file()
    nested = "cells/00-tolerance-1e-8/000-load-ratio-0p25/run/run.json"
    assert nested in by_path
    for relative in ("platform-evidence.json", nested):
        path = completed_evidence.output_dir / relative
        assert by_path[relative]["bytes"] == path.stat().st_size
        assert by_path[relative]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_convergence_family_rejects_event_and_mode_signature_mutations(
    completed_evidence,
) -> None:
    evidence = json.loads(completed_evidence.platform_evidence.read_text(encoding="utf-8"))
    cells = deepcopy(evidence["cells"])
    event_cell = next(
        cell
        for cell in cells
        if cell["point_id"] == "load-ratio-1" and cell["configuration_id"] == "tolerance-1e-9"
    )
    event_cell["semantic_signature"]["fields"]["events.semantic_sequence_sha256"]["value"] = (
        "f" * 64
    )
    mode_cell = next(
        cell
        for cell in cells
        if cell["point_id"] == "load-ratio-2" and cell["configuration_id"] == "max-step-t-over-80"
    )
    mode_count = mode_cell["semantic_signature"]["fields"]["modes.interval_count"]
    mode_count["value"] += 1

    studies = _study_summaries(full_v1_profile(), cells)
    tolerance_gate = next(
        gate for gate in studies["tolerance"]["gates"] if gate["point_id"] == "load-ratio-1"
    )
    max_step_gate = next(
        gate for gate in studies["max_step"]["gates"] if gate["point_id"] == "load-ratio-2"
    )

    assert studies["tolerance"]["status"] == "fail"
    assert tolerance_gate["status"] == "fail"
    assert tolerance_gate["stability"]["event_sequence"]["status"] == "fail"
    assert studies["max_step"]["status"] == "fail"
    assert max_step_gate["status"] == "fail"
    assert max_step_gate["stability"]["mode_sequence"]["status"] == "fail"


def test_convergence_family_tracks_fixed_oracle_gates_not_a_changing_worst_gate(
    completed_evidence,
) -> None:
    evidence = json.loads(completed_evidence.platform_evidence.read_text(encoding="utf-8"))
    cells = deepcopy(evidence["cells"])
    profile = full_v1_profile()
    configuration_ids = profile.study_configuration_ids["tolerance"]
    moving_fractions = (0.8, 0.4, 0.2, 0.1, 0.01)

    for configuration_id, moving_fraction in zip(
        configuration_ids,
        moving_fractions,
        strict=True,
    ):
        cell = next(
            candidate
            for candidate in cells
            if candidate["point_id"] == "load-ratio-0p5"
            and candidate["configuration_id"] == configuration_id
        )
        numeric_gates = {}
        for oracle_gate in cell["gates"]:
            limit = oracle_gate.get("limit")
            if (
                isinstance(limit, (int, float))
                and not isinstance(limit, bool)
                and float(limit) > 0.0
            ):
                oracle_gate["actual"] = 0.0
                numeric_gates[oracle_gate["id"]] = oracle_gate
        stationary = numeric_gates["boundary_state_nonzero"]
        stationary["actual"] = stationary["limit"] * 0.02
        moving = numeric_gates["cycle_mean_output"]
        moving["actual"] = moving["limit"] * moving_fraction

    tolerance = _study_summaries(profile, cells)["tolerance"]
    family_gate = next(gate for gate in tolerance["gates"] if gate["point_id"] == "load-ratio-0p5")
    stationary_series = family_gate["oracle_gate_series"]["boundary_state_nonzero"]
    moving_series = family_gate["oracle_gate_series"]["cycle_mean_output"]
    envelope = family_gate["worst_envelope_summary"]

    assert envelope["normalized_budget_fractions"] == pytest.approx([0.8, 0.4, 0.2, 0.1, 0.02])
    assert envelope["positive_trend"] is True
    assert envelope["status"] == "pass"
    assert envelope["acceptance_role"] == "diagnostic-only"
    assert stationary_series["normalized_budget_fractions"] == pytest.approx([0.02] * 5)
    assert stationary_series["positive_trend"] is False
    assert stationary_series["final_three_plateau"] is False
    assert stationary_series["status"] == "fail"
    assert moving_series["status"] == "pass"
    assert family_gate["status"] == "fail"
    assert tolerance["status"] == "fail"
    assert "boundary_state_nonzero" in family_gate["reason"]


def test_solver_family_rejects_signed_physical_value_drift(completed_evidence) -> None:
    evidence = json.loads(completed_evidence.platform_evidence.read_text(encoding="utf-8"))
    cells = deepcopy(evidence["cells"])
    changed = next(
        cell
        for cell in cells
        if cell["point_id"] == "load-ratio-2" and cell["configuration_id"] == "solver-bdf"
    )
    field = changed["semantic_signature"]["fields"]["metrics.mean_output_voltage_v"]
    field["value"] += field["cross_platform_tolerance"]["absolute"] * 3.0
    assert {gate["status"] for gate in changed["gates"]} == {"pass"}

    solver = _study_summaries(full_v1_profile(), cells)["solver"]
    gate = next(item for item in solver["gates"] if item["point_id"] == "load-ratio-2")
    comparison = gate["signature_field_ranges"]["metrics.mean_output_voltage_v"]

    assert solver["status"] == "fail"
    assert gate["status"] == "fail"
    assert comparison["status"] == "fail"
    assert comparison["actual_range"] > comparison["limit"]


def test_platform_record_retains_declared_ci_and_toolchain_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "PEAR_SIM_UV_VERSION": "0.11.28",
        "GITHUB_ACTIONS": "true",
        "RUNNER_OS": "macOS",
        "RUNNER_ARCH": "ARM64",
        "RUNNER_ENVIRONMENT": "github-hosted",
        "ImageOS": "macos15",
        "ImageVersion": "20260701.1",
        "GITHUB_WORKFLOW_REF": "owner/repo/.github/workflows/numerical-evidence.yml@refs/pull/1/merge",
        "GITHUB_SHA": "a" * 40,
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)

    record = _platform_record()

    assert record["toolchain"] == {"uv": "0.11.28"}
    assert record["ci_environment"] == {
        name: value for name, value in values.items() if name != "PEAR_SIM_UV_VERSION"
    }


def test_runner_retains_all_not_run_cells_and_never_converts_them_to_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(_simulator: HybridSimulator):
        raise RuntimeError("deliberate matrix failure")

    monkeypatch.setattr(HybridSimulator, "simulate", fail)
    artifacts = run_buck_numerical_evidence(
        BUCK,
        tmp_path / "failed-matrix",
        source_revision="test-revision",
    )
    evidence = json.loads(artifacts.platform_evidence.read_text(encoding="utf-8"))

    assert artifacts.status == "not-run"
    assert evidence["summary"]["cell_status_counts"] == {"not-run": 144}
    assert len(list(artifacts.output_dir.glob("cells/*/*/cell.json"))) == 144
    assert not list(artifacts.output_dir.glob("cells/*/*/run/run.json"))
    assert all(cell["error"]["type"] == "RuntimeError" for cell in evidence["cells"])
    assert all(gate["status"] == "not-run" for cell in evidence["cells"] for gate in cell["gates"])
    assert all(
        cell["semantic_signature"]["status"] == "not-run"
        and cell["semantic_signature"]["fields"] == {}
        for cell in evidence["cells"]
    )
    assert {study["status"] for study in evidence["studies"].values()} == {"not-run"}


def test_runner_rejects_unsupported_input_and_nonempty_destination(tmp_path: Path) -> None:
    unsupported = tmp_path / "unsupported"
    with pytest.raises(ValueError):
        run_buck_numerical_evidence(BOOST, unsupported, source_revision="test-revision")
    assert not unsupported.exists()

    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "human-note.txt"
    marker.write_text("keep\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not empty"):
        run_buck_numerical_evidence(BUCK, destination, source_revision="test-revision")
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_profile_rejects_more_than_bounded_matrix() -> None:
    profile = full_v1_profile()
    with pytest.raises(ValueError, match="cell numerical evidence limit"):
        replace(profile, load_ratios=tuple(index / 100 for index in range(1, 18)))


def test_profile_rejects_inconsistent_study_index() -> None:
    profile = full_v1_profile()
    broken = dict(profile.study_configuration_ids)
    broken["solver"] = broken["solver"][:-1]
    with pytest.raises(ValueError, match="index differs from membership"):
        NumericalEvidenceProfile(
            id=profile.id,
            load_ratios=profile.load_ratios,
            configurations=profile.configurations,
            study_configuration_ids=broken,
            cross_platform_policy=profile.cross_platform_policy,
        )
