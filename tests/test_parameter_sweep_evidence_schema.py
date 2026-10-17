from __future__ import annotations

import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest
from referencing import Registry, Resource

from pear_sim.parameter_sweep import ParameterSweepArtifacts, run_parameter_sweep


ROOT = Path(__file__).parents[1]
SCHEMA = ROOT / "schema"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_model(path: Path) -> Path:
    path.write_text(
        """schema: pear-sim/v0.1
name: parameter_sweep_schema_fixture
parameters:
  load: 10 ohm
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 12 V}
    - {id: r1, type: resistor, p: in, n: out, value: "${load}"}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 1 uF, initial_voltage: 0 V}
controls: {}
analysis:
  transient:
    start: 0 s
    stop: 50 us
    output_step: 1 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11
outputs:
  - v(out)
""",
        encoding="utf-8",
    )
    return path


def _write_request(path: Path, values: list[str]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": "pear-sim/parameter-sweep-request/v0.1",
                "parameter": "load",
                "values": values,
                "projections": [
                    {
                        "id": "vout_final",
                        "signal": "v(out)",
                        "metric": "sample_final",
                        "unit": "V",
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def actual_bundles(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ParameterSweepArtifacts, ParameterSweepArtifacts]:
    root = tmp_path_factory.mktemp("parameter-sweep-evidence-schema")
    model = _write_model(root / "model.yaml")
    passing = run_parameter_sweep(
        model,
        _write_request(root / "passing-request.json", ["10 ohm", "20 ohm"]),
        root / "passing",
    )
    failing = run_parameter_sweep(
        model,
        _write_request(root / "failing-request.json", ["10 ohm", "-1 ohm", "20 ohm"]),
        root / "failing",
    )
    assert passing.execution_status == "pass"
    assert failing.execution_status == "fail"
    return passing, failing


def _evidence_validator() -> Draft202012Validator:
    evidence_schema = _load_json(SCHEMA / "parameter-sweep-evidence.schema.json")
    summary_schema = _load_json(SCHEMA / "parameter-sweep-summary.schema.json")
    registry = Registry().with_resource(
        str(summary_schema["$id"]),
        Resource.from_contents(summary_schema),
    )
    return Draft202012Validator(evidence_schema, registry=registry)


def _evidence_documents(bundle: ParameterSweepArtifacts) -> list[Path]:
    return [
        bundle.normalized_request,
        bundle.source_model,
        *sorted(bundle.output_dir.glob("points/*/point.json")),
        bundle.projections,
        bundle.failures,
        bundle.confidence_json,
        bundle.report_json,
    ]


def test_parameter_sweep_evidence_schema_is_a_closed_seven_document_union() -> None:
    schema = _load_json(SCHEMA / "parameter-sweep-evidence.schema.json")
    expected = [
        "normalized_request",
        "source_model",
        "point",
        "projections",
        "failures",
        "confidence",
        "report",
    ]

    Draft202012Validator.check_schema(schema)
    assert schema["oneOf"] == [{"$ref": f"#/$defs/{name}"} for name in expected]
    for name in expected:
        document = schema["$defs"][name]
        assert document["type"] == "object"
        assert document["additionalProperties"] is False


def test_actual_passing_and_failing_bundles_match_all_published_schemas(
    actual_bundles: tuple[ParameterSweepArtifacts, ParameterSweepArtifacts],
) -> None:
    evidence_validator = _evidence_validator()
    request_validator = Draft202012Validator(
        _load_json(SCHEMA / "parameter-sweep-request.schema.json")
    )
    summary_validator = Draft202012Validator(
        _load_json(SCHEMA / "parameter-sweep-summary.schema.json")
    )

    for bundle in actual_bundles:
        request_validator.validate(_load_json(bundle.source_request))
        summary_validator.validate(_load_json(bundle.sweep_summary))
        for document in _evidence_documents(bundle):
            evidence_validator.validate(_load_json(document))


def test_every_evidence_document_rejects_unknown_root_fields(
    actual_bundles: tuple[ParameterSweepArtifacts, ParameterSweepArtifacts],
) -> None:
    validator = _evidence_validator()

    for bundle in actual_bundles:
        for document in _evidence_documents(bundle):
            payload = copy.deepcopy(_load_json(document))
            payload["unexpected_field"] = "must be rejected"
            assert not validator.is_valid(payload), document


def test_schema_captures_source_artifact_work_limits_and_claim_boundaries() -> None:
    schema = _load_json(SCHEMA / "parameter-sweep-evidence.schema.json")
    definitions = schema["$defs"]
    normalized = definitions["normalized_request"]
    source_model = definitions["source_model"]

    assert normalized["properties"]["source"]["properties"]["artifact"] == {
        "const": "request.source.json"
    }
    assert set(normalized["properties"]["limits"]["required"]) == {
        "maximum_components_per_point",
        "maximum_controls_per_point",
        "maximum_parameters",
        "maximum_point_wall_time_s",
        "maximum_points",
        "maximum_projections",
        "maximum_requested_max_step_intervals_per_point",
        "maximum_sample_output_cells_per_point",
        "maximum_samples_per_point",
        "maximum_total_samples",
        "maximum_scheduled_edges_per_point",
        "maximum_source_model_bytes",
        "maximum_sources_per_point",
        "maximum_states_per_point",
        "maximum_total_requested_max_step_intervals",
        "maximum_total_sample_output_cells",
        "maximum_total_scheduled_edges",
        "maximum_total_wall_time_s",
        "maximum_outputs",
    }
    assert "source_artifact" in source_model["required"]
    projection_request = definitions["projection_request"]["properties"]
    assert projection_request["id"]["pattern"] == (
        r"^[A-Za-z_][A-Za-z0-9_.:-]*(?![\s\S])"
    )
    assert projection_request["signal"]["pattern"] == (
        r"^[vi]\([A-Za-z_][A-Za-z0-9_.:-]*\)(?![\s\S])"
    )

    confidence_claim = definitions["confidence"]["properties"]["claim"]
    report_claim = definitions["report"]["properties"]["claim"]
    for field in (
        "validation_claim",
        "compatibility_claim",
        "hardware_sign_off_allowed",
        "general_parameter_envelope_claimed",
    ):
        assert confidence_claim["properties"][field] == {"const": False}
    assert report_claim["properties"]["validation_claim"] == {"const": False}
    assert report_claim["properties"]["hardware_sign_off_allowed"] == {"const": False}
