# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from pear_sim.benchmarks import INVERTING_BUCK_BOOST_DEFAULT_LOAD_RATIOS
import pear_sim.sweep as sweep_module
from pear_sim.solver import HybridSimulator
from pear_sim.sweep import (
    MAX_INVERTING_BUCK_BOOST_SOURCE_BYTES,
    MAX_INVERTING_BUCK_BOOST_SOURCE_OUTPUTS,
    MAX_SWEEP_POINTS,
    run_inverting_buck_boost_boundary_benchmark,
)


ROOT = Path(__file__).parents[1]
INVERTING_BUCK_BOOST = ROOT / "examples" / "inverting_buck_boost.yaml"
BOOST = ROOT / "examples" / "boost.yaml"
EVIDENCE_SCHEMA = ROOT / "schema" / "inverting-buck-boost-boundary-evidence.schema.json"
GATE_IDS = [
    "regime",
    "boundary_state_nonzero",
    "boundary_state_zero",
    "natural_event_time",
    "reverse_current",
    "signed_cycle_mean_output",
    "signed_output_polarity",
    "state_ripple",
    "mode_fractions",
    "event_semantics",
    "energy_balance",
    "cycle_periodicity",
    "unclassified_fraction",
    "mode_fraction_sum",
]


@pytest.fixture(scope="module")
def completed_inverting_buck_boost_sweep(tmp_path_factory: pytest.TempPathFactory):
    return run_inverting_buck_boost_boundary_benchmark(
        INVERTING_BUCK_BOOST,
        tmp_path_factory.mktemp("inverting-buck-boost-boundary") / "sweep",
    )


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validator() -> Draft202012Validator:
    schema = _load(EVIDENCE_SCHEMA)
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _write_model_variant(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def test_inverting_buck_boost_runner_retains_nine_runs_and_126_blocking_gates(
    completed_inverting_buck_boost_sweep,
) -> None:
    artifacts = completed_inverting_buck_boost_sweep
    summary = _load(artifacts.sweep_summary)
    result = _load(artifacts.benchmark_result)
    confidence = _load(artifacts.confidence_json)

    assert artifacts.overall_status == "pass"
    assert summary["schema"] == (
        "pear-sim/sweep-summary/ideal-inverting-buck-boost-boundary/v0.1"
    )
    assert summary["benchmark_id"] == (
        "VAL-003:canonical-inverting-buck-boost-boundary-sweep"
    )
    assert summary["bcm"]["r_bcm_ohm"] == pytest.approx(
        79.91663630292852,
        rel=0.0,
        abs=3e-12,
    )
    assert summary["summary"] == {
        "gate_status_counts": {"pass": 126},
        "point_count": 9,
        "point_status_counts": {"pass": 9},
        "status": "pass",
    }
    expected_regimes = ["CCM", "CCM", "CCM", "CCM", "BCM", "DCM", "DCM", "DCM", "DCM"]
    assert [point["expected_regime"] for point in summary["points"]] == expected_regimes
    assert [point["observed_regime"] for point in summary["points"]] == expected_regimes
    assert all([gate["id"] for gate in point["gates"]] == GATE_IDS for point in summary["points"])
    assert len(list((artifacts.output_dir / "points").glob("*/run/run.json"))) == 9

    assert result["schema"] == (
        "pear-sim/benchmark-runner-result/ideal-inverting-buck-boost-boundary/v0.1"
    )
    assert result["oracle_schema"] == (
        "pear-sim/benchmark-oracle/ideal-inverting-buck-boost-boundary/v0.1"
    )
    assert result["comparison_schema"] == (
        "pear-sim/benchmark-result/ideal-inverting-buck-boost-boundary/v0.1"
    )
    assert result["summary"] == {
        "gate_count": 126,
        "gate_status_counts": {"pass": 126},
        "point_count": 9,
        "status": "pass",
    }
    assert result["claim"]["validation_claim"] is False
    assert len(summary["worst_gates"]) == 14
    assert summary["failures"] == []

    assert confidence["schema"] == (
        "pear-sim/confidence/inverting-buck-boost-boundary-sweep/v0.1"
    )
    assert confidence["level"] == "EXPLORATORY"
    assert confidence["envelope"]["topology"] == (
        "canonical single-switch ideal inverting buck-boost"
    )
    assert confidence["envelope"]["output_polarity"] == "v(out) < 0"
    assert confidence["envelope"]["requested_regimes"] == ["CCM", "BCM", "DCM"]
    assert confidence["envelope"]["completed_regimes"] == ["CCM", "BCM", "DCM"]
    assert confidence["claim"] == {
        "compatibility_claim": False,
        "general_dae_support_claimed": False,
        "hardware_sign_off_allowed": False,
        "validation_claim": False,
    }


def test_inverting_buck_boost_runner_preserves_signed_output_contract(
    completed_inverting_buck_boost_sweep,
) -> None:
    artifacts = completed_inverting_buck_boost_sweep
    summary = _load(artifacts.sweep_summary)
    points = summary["points"]

    assert all(point["metrics"]["signed_mean_output_voltage_v"] < 0.0 for point in points)
    assert all(point["metrics"]["maximum_signed_output_voltage_v"] <= 0.0 for point in points)
    assert points[-1]["metrics"]["signed_mean_output_voltage_v"] < -300.0

    point_dir = artifacts.output_dir / "points" / "004-load-ratio-1"
    point = _load(point_dir / "benchmark-point.json")
    resolved = _load(point_dir / "run" / "model.resolved.json")
    assert point["initial_conditions"]["source"] == "independent periodic affine oracle"
    assert point["initial_conditions"]["output_magnitude_voltage_v"] > 0.0
    assert point["initial_conditions"]["signed_output_voltage_v"] == pytest.approx(
        -point["initial_conditions"]["output_magnitude_voltage_v"],
        abs=1e-15,
    )
    assert point["observation"]["signed_mean_output_voltage_v"] < 0.0
    assert resolved["parameters"]["load"]["value_si"] == pytest.approx(
        point["reference"]["load_ohm"],
        abs=1e-12,
    )
    assert resolved["initial_conditions"]["analytic_initialization_claimed"] is False
    assert point["run"]["run_sha256"] == _load(point_dir / "run" / "run.json")["run_sha256"]


def test_inverting_buck_boost_schema_accepts_all_fourteen_producer_documents(
    completed_inverting_buck_boost_sweep,
) -> None:
    artifacts = completed_inverting_buck_boost_sweep
    schema = _load(EVIDENCE_SCHEMA)
    validator = _validator()
    expected_union = [
        "spec",
        "oracle",
        "point",
        "runner_result",
        "summary",
        "confidence",
    ]
    assert schema["oneOf"] == [{"$ref": f"#/$defs/{name}"} for name in expected_union]
    assert all(schema["$defs"][name]["additionalProperties"] is False for name in expected_union)
    for name in ("gate_contract", "gates"):
        ordered = schema["$defs"][name]
        assert ordered["minItems"] == ordered["maxItems"] == 14
        assert ordered["items"] is False
        assert [
            item["allOf"][1]["properties"]["id"]["const"]
            for item in ordered["prefixItems"]
        ] == GATE_IDS

    point_documents = sorted(
        (artifacts.output_dir / "points").glob("*/benchmark-point.json")
    )
    documents = [
        artifacts.benchmark_spec,
        artifacts.oracle_json,
        *point_documents,
        artifacts.benchmark_result,
        artifacts.sweep_summary,
        artifacts.confidence_json,
    ]
    assert len(documents) == 14
    for document in documents:
        validator.validate(_load(document))

    spec = _load(artifacts.benchmark_spec)
    assert spec["source_work"]["output_count"] <= 60
    assert spec["limits"]["maximum_source_outputs"] == 60
    assert spec["limits"]["maximum_point_outputs"] == 64
    assert spec["solver_work"] == _load(artifacts.oracle_json)["solver_work_contract"]
    assert spec["regime_separation"] == _load(artifacts.oracle_json)[
        "regime_separation_contract"
    ]
    assert spec["regime_separation"]["minimum_requested_ccm_margin_normalized"] > 1e-8
    assert spec["regime_separation"]["minimum_requested_dcm_margin_normalized"] > 1e-8


def test_inverting_buck_boost_root_manifest_rehashes_every_nested_artifact(
    completed_inverting_buck_boost_sweep,
) -> None:
    artifacts = completed_inverting_buck_boost_sweep
    manifest = _load(artifacts.artifact_manifest)
    records = manifest["artifacts"]
    paths = [record["path"] for record in records]

    assert manifest["subject"] == {
        "kind": "inverting-buck-boost-boundary-sweep",
        "sha256": hashlib.sha256(artifacts.sweep_summary.read_bytes()).hexdigest(),
    }
    assert paths == sorted(paths)
    assert "sweep-summary.json" in paths
    assert "report/sweep-report.html" in paths
    assert "report/schematic.svg" in paths
    assert "points/004-load-ratio-1/run/waveforms.csv" in paths
    assert "artifact-manifest.json" not in paths
    for record in records:
        path = artifacts.output_dir / record["path"]
        assert record["bytes"] == path.stat().st_size
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_inverting_buck_boost_report_is_offline_and_contains_the_signed_schematic(
    completed_inverting_buck_boost_sweep,
) -> None:
    artifacts = completed_inverting_buck_boost_sweep
    report = _load(artifacts.report_json)
    rendered = artifacts.report_html.read_text(encoding="utf-8")

    assert report["source"]["sweep_summary_artifact_sha256"] == hashlib.sha256(
        artifacts.sweep_summary.read_bytes()
    ).hexdigest()
    assert report["source"]["schematic"]["path"] == "schematic.svg"
    assert report["presentation"]["schematic"]["role"] == (
        "nominal_source_topology_template"
    )
    assert "EXPLORATORY — NOT VALIDATION" in rendered
    assert "VAL-003:canonical-inverting-buck-boost-boundary-sweep" in rendered
    assert "v(out) is negative" in rendered
    assert "script-src 'none'" in rendered
    assert "<script" not in rendered.lower()
    assert artifacts.schematic_svg is not None
    assert artifacts.schematic_svg.is_file()


def test_inverting_buck_boost_runner_retains_failure_with_fourteen_not_run_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = HybridSimulator.simulate

    def selectively_fail(simulator: HybridSimulator):
        if simulator.model.name.endswith("load-ratio-1p01"):
            raise RuntimeError("deliberate inverting buck-boost point failure")
        return original(simulator)

    monkeypatch.setattr(HybridSimulator, "simulate", selectively_fail)
    artifacts = run_inverting_buck_boost_boundary_benchmark(
        INVERTING_BUCK_BOOST,
        tmp_path / "sweep",
    )
    summary = _load(artifacts.sweep_summary)

    assert artifacts.overall_status == "not-run"
    assert len(summary["points"]) == len(INVERTING_BUCK_BOOST_DEFAULT_LOAD_RATIOS)
    failed = next(point for point in summary["points"] if point["status"] == "not-run")
    assert failed["point_id"] == "load-ratio-1p01"
    assert [gate["id"] for gate in failed["gates"]] == GATE_IDS
    assert {gate["status"] for gate in failed["gates"]} == {"not-run"}
    assert "deliberate inverting buck-boost point failure" in failed["gates"][0]["reason"]
    assert len(list((artifacts.output_dir / "points").glob("*/benchmark-point.json"))) == 9
    assert len(list((artifacts.output_dir / "points").glob("*/run/run.json"))) == 8
    assert summary["summary"]["point_status_counts"] == {"not-run": 1, "pass": 8}
    assert summary["summary"]["gate_status_counts"] == {"not-run": 14, "pass": 112}
    assert len(summary["failures"]) == 14

    validator = _validator()
    for document in [
        artifacts.benchmark_spec,
        artifacts.oracle_json,
        *sorted((artifacts.output_dir / "points").glob("*/benchmark-point.json")),
        artifacts.benchmark_result,
        artifacts.sweep_summary,
        artifacts.confidence_json,
    ]:
        validator.validate(_load(document))


def test_inverting_buck_boost_runner_rejects_unsupported_or_unbounded_work_before_writing(
    tmp_path: Path,
) -> None:
    unsupported = tmp_path / "unsupported"
    with pytest.raises(ValueError):
        run_inverting_buck_boost_boundary_benchmark(BOOST, unsupported)
    assert not unsupported.exists()

    too_many = tmp_path / "too-many"
    ratios = [index / 100.0 for index in range(1, MAX_SWEEP_POINTS + 2)]
    with pytest.raises(ValueError, match="point contract limit"):
        run_inverting_buck_boost_boundary_benchmark(
            INVERTING_BUCK_BOOST,
            too_many,
            load_ratios=ratios,
        )
    assert not too_many.exists()

    ambiguous = tmp_path / "ambiguous"
    with pytest.raises(ValueError, match="exact BCM ratio"):
        run_inverting_buck_boost_boundary_benchmark(
            INVERTING_BUCK_BOOST,
            ambiguous,
            load_ratios=[0.9, 1.1],
        )
    assert not ambiguous.exists()

    out_of_envelope = tmp_path / "out-of-envelope"
    with pytest.raises(ValueError, match="supported bounded evidence envelope"):
        run_inverting_buck_boost_boundary_benchmark(
            INVERTING_BUCK_BOOST,
            out_of_envelope,
            load_ratios=[1.0e-6, 1.0, 1.1],
        )
    assert not out_of_envelope.exists()

    bcm_only = tmp_path / "bcm-only"
    with pytest.raises(ValueError, match="CCM ratio below"):
        run_inverting_buck_boost_boundary_benchmark(
            INVERTING_BUCK_BOOST,
            bcm_only,
            load_ratios=[1.0],
        )
    assert not bcm_only.exists()

    near_bcm = tmp_path / "near-bcm"
    with pytest.raises(ValueError, match="regime-separation contract"):
        run_inverting_buck_boost_boundary_benchmark(
            INVERTING_BUCK_BOOST,
            near_bcm,
            load_ratios=[0.9, 1.0, 1.000000001],
        )
    assert not near_bcm.exists()

    stiff_model = _write_model_variant(
        tmp_path,
        "stiff-inverting-buck-boost.yaml",
        INVERTING_BUCK_BOOST.read_text(encoding="utf-8").replace("100 uF", "1 pF"),
    )
    stiff_destination = tmp_path / "stiff"
    with pytest.raises(ValueError, match="resonant-frequency work estimate exceeds"):
        run_inverting_buck_boost_boundary_benchmark(
            stiff_model,
            stiff_destination,
            load_ratios=[0.9, 1.0, 1.1],
        )
    assert not stiff_destination.exists()


def test_inverting_buck_boost_runner_bounds_source_bytes_and_outputs_before_writing(
    tmp_path: Path,
) -> None:
    source_text = INVERTING_BUCK_BOOST.read_text(encoding="utf-8")

    oversized_model = _write_model_variant(
        tmp_path,
        "oversized-inverting-buck-boost.yaml",
        source_text
        + "\n# "
        + "x" * (MAX_INVERTING_BUCK_BOOST_SOURCE_BYTES + 1),
    )
    oversized_destination = tmp_path / "oversized"
    with pytest.raises(ValueError, match="input limit"):
        run_inverting_buck_boost_boundary_benchmark(
            oversized_model,
            oversized_destination,
            load_ratios=[0.9, 1.0, 1.1],
        )
    assert not oversized_destination.exists()

    outputs = ", ".join(
        f"v(output_{index})"
        for index in range(MAX_INVERTING_BUCK_BOOST_SOURCE_OUTPUTS + 1)
    )
    excessive_outputs_model = _write_model_variant(
        tmp_path,
        "too-many-outputs-inverting-buck-boost.yaml",
        source_text.replace(
            "outputs: [v(out), i(l1), v(q1), i(d1)]",
            f"outputs: [{outputs}]",
        ),
    )
    excessive_outputs_destination = tmp_path / "too-many-outputs"
    with pytest.raises(ValueError, match="bounded source-model limit"):
        run_inverting_buck_boost_boundary_benchmark(
            excessive_outputs_model,
            excessive_outputs_destination,
            load_ratios=[0.9, 1.0, 1.1],
        )
    assert not excessive_outputs_destination.exists()


def test_inverting_buck_boost_runner_never_overwrites_prior_sweep(tmp_path: Path) -> None:
    destination = tmp_path / "sweep"
    destination.mkdir()
    marker = destination / "human-note.txt"
    marker.write_text("keep\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        run_inverting_buck_boost_boundary_benchmark(
            INVERTING_BUCK_BOOST,
            destination,
            load_ratios=[0.9, 1.0, 1.1],
        )

    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_inverting_buck_boost_runner_truncates_retained_exception_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = "🧪" * 5000
    below_bcm = [0.25 + (0.99 - 0.25) * index / 31.0 for index in range(32)]
    above_bcm = [1.01 + (2.0 - 1.01) * index / 30.0 for index in range(31)]
    ratios = [*below_bcm, 1.0, *above_bcm]
    assert len(ratios) == MAX_SWEEP_POINTS

    def fail_with_long_message(_simulator: HybridSimulator):
        raise RuntimeError(message)

    monkeypatch.setattr(HybridSimulator, "simulate", fail_with_long_message)
    artifacts = run_inverting_buck_boost_boundary_benchmark(
        INVERTING_BUCK_BOOST,
        tmp_path / "long-error",
        load_ratios=ratios,
    )

    assert artifacts.overall_status == "not-run"
    validator = _validator()
    point_documents = sorted(
        (artifacts.output_dir / "points").glob("*/benchmark-point.json")
    )
    assert len(point_documents) == MAX_SWEEP_POINTS
    for document in point_documents:
        payload = _load(document)
        validator.validate(payload)
        assert len(payload["error"]["message"].encode("utf-8")) <= 1024
        assert payload["error"]["message"].endswith("…")
        assert all(
            len(gate["reason"].encode("utf-8")) <= 1024
            for gate in payload["comparison"]["gates"]
        )

    for document in [
        artifacts.benchmark_spec,
        artifacts.oracle_json,
        artifacts.benchmark_result,
        artifacts.sweep_summary,
        artifacts.confidence_json,
    ]:
        validator.validate(_load(document))
    assert artifacts.report_html.is_file()
    assert artifacts.artifact_manifest.is_file()


def test_inverting_buck_boost_schema_retains_bounded_solver_fail_observation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = sweep_module.observe_inverting_buck_boost_simulation_result

    def mutate_one_observation(*args, **kwargs):
        observation = original(*args, **kwargs)
        reference = args[1]
        if reference.point_id != "load-ratio-0p9":
            return observation
        return replace(
            observation,
            start_state=replace(
                observation.start_state,
                output_magnitude_voltage_v=-1.0,
            ),
            signed_mean_output_voltage_v=abs(
                observation.signed_mean_output_voltage_v
            ),
            mode_sequence=("unclassified",),
            event_sequence=(),
        )

    monkeypatch.setattr(
        sweep_module,
        "observe_inverting_buck_boost_simulation_result",
        mutate_one_observation,
    )
    artifacts = run_inverting_buck_boost_boundary_benchmark(
        INVERTING_BUCK_BOOST,
        tmp_path / "retained-fail",
        load_ratios=[0.9, 1.0, 1.1],
    )

    assert artifacts.overall_status == "fail"
    validator = _validator()
    point_documents = sorted(
        (artifacts.output_dir / "points").glob("*/benchmark-point.json")
    )
    failed = _load(point_documents[0])
    assert failed["status"] == "fail"
    assert failed["error"] is None
    assert failed["run"] is not None
    assert failed["observation"]["signed_mean_output_voltage_v"] > 0.0
    assert failed["observation"]["start_state"]["output_magnitude_voltage_v"] < 0.0
    assert failed["observation"]["mode_sequence"] == ["unclassified"]
    assert failed["observation"]["event_sequence"] == []

    for document in [
        artifacts.benchmark_spec,
        artifacts.oracle_json,
        *point_documents,
        artifacts.benchmark_result,
        artifacts.sweep_summary,
        artifacts.confidence_json,
    ]:
        validator.validate(_load(document))
