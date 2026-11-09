# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest


ROOT = Path(__file__).parents[1]
SCHEMA = ROOT / "schema"


def _schema(filename: str) -> dict[str, object]:
    return json.loads((SCHEMA / filename).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "filename",
    [
        "numerical-profile.schema.json",
        "platform-evidence.schema.json",
        "cross-platform-evidence.schema.json",
        "boost-boundary-evidence.schema.json",
    ],
)
def test_evidence_schema_is_valid_draft_2020_12(filename: str) -> None:
    Draft202012Validator.check_schema(_schema(filename))


@pytest.mark.parametrize(
    ("filename", "required", "constants"),
    [
        (
            "platform-evidence.schema.json",
            {
                "classification",
                "validation_claim",
                "hardware_sign_off_allowed",
                "scope",
            },
            {
                "classification": "verification",
                "validation_claim": False,
                "hardware_sign_off_allowed": False,
            },
        ),
        (
            "cross-platform-evidence.schema.json",
            {
                "validation_claim",
                "compatibility_claim",
                "hardware_sign_off_allowed",
                "statement",
            },
            {
                "validation_claim": False,
                "compatibility_claim": False,
                "hardware_sign_off_allowed": False,
            },
        ),
    ],
)
def test_evidence_schema_claim_boundaries_are_closed(
    filename: str,
    required: set[str],
    constants: dict[str, object],
) -> None:
    claim = _schema(filename)["properties"]["claim"]

    assert claim["type"] == "object"
    assert claim["additionalProperties"] is False
    assert set(claim["required"]) == required
    for field, expected in constants.items():
        assert claim["properties"][field] == {"const": expected}


def test_full_v1_cross_platform_schema_requires_every_semantic_comparison() -> None:
    comparisons = _schema("cross-platform-evidence.schema.json")["properties"]["comparisons"]

    assert comparisons["minItems"] == 9 * 16 * 22
    assert comparisons["maxItems"] == 9 * 16 * 22
    item = comparisons["items"]
    assert item["additionalProperties"] is False
    assert {"gate_id", "quantity", "unit", "policy_id", "values"} <= set(item["required"])


def test_platform_schema_requires_a_closed_twenty_two_field_signature() -> None:
    schema = _schema("platform-evidence.schema.json")
    cell = schema["properties"]["cells"]["items"]
    signature = schema["$defs"]["semantic_signature"]

    assert "semantic_signature" in cell["required"]
    assert signature["additionalProperties"] is False
    available = signature["oneOf"][0]["properties"]
    assert available["status"] == {"const": "available"}
    assert available["fields"] == {"minProperties": 22, "maxProperties": 22}


def test_platform_schema_requires_the_complete_full_v1_matrix() -> None:
    properties = _schema("platform-evidence.schema.json")["properties"]
    profile = properties["profile"]["properties"]

    assert profile["id"] == {"const": "full-v1"}
    assert profile["point_count"] == {"const": 9}
    assert profile["configuration_count"] == {"const": 16}
    assert profile["cell_count"] == {"const": 144}
    assert properties["configurations"]["minItems"] == 16
    assert properties["configurations"]["maxItems"] == 16
    assert properties["cells"]["minItems"] == 144
    assert properties["cells"]["maxItems"] == 144
    gates = properties["cells"]["items"]["properties"]["gates"]
    assert gates["minItems"] == 11
    assert gates["maxItems"] == 11


def test_platform_schema_keeps_local_oracle_gates_separate_from_signature_policy() -> None:
    schema = _schema("platform-evidence.schema.json")
    gate = schema["properties"]["cells"]["items"]["properties"]["gates"]["items"]

    assert set(gate["required"]) == {"id", "actual", "limit", "status"}
    assert "value" not in gate["properties"]
    assert "cross_platform_tolerance" not in gate["properties"]


def test_cross_platform_failure_capacity_covers_every_semantic_comparison() -> None:
    properties = _schema("cross-platform-evidence.schema.json")["properties"]

    assert properties["failures"]["maxItems"] >= 9 * 16 * 22


def test_boost_boundary_schema_is_a_closed_six_document_union() -> None:
    schema = _schema("boost-boundary-evidence.schema.json")
    expected = [
        "spec",
        "oracle",
        "point",
        "runner_result",
        "summary",
        "confidence",
    ]

    assert schema["oneOf"] == [{"$ref": f"#/$defs/{name}"} for name in expected]
    for name in expected:
        assert schema["$defs"][name]["type"] == "object"
        assert schema["$defs"][name]["additionalProperties"] is False


def test_boost_boundary_schema_closes_thirteen_ordered_gate_classes() -> None:
    schema = _schema("boost-boundary-evidence.schema.json")
    expected = [
        "regime",
        "boundary_state_nonzero",
        "boundary_state_zero",
        "natural_event_time",
        "reverse_current",
        "cycle_mean_output",
        "state_ripple",
        "mode_fractions",
        "event_semantics",
        "energy_balance",
        "cycle_periodicity",
        "unclassified_fraction",
        "mode_fraction_sum",
    ]

    for name in ("gate_contract", "gates"):
        gates = schema["$defs"][name]
        assert gates["minItems"] == 13
        assert gates["maxItems"] == 13
        assert gates["items"] is False
        assert [item["allOf"][1]["properties"]["id"]["const"] for item in gates["prefixItems"]] == expected


def test_boost_boundary_schema_keeps_verification_claims_closed() -> None:
    schema = _schema("boost-boundary-evidence.schema.json")
    claim = schema["$defs"]["verification_claim"]
    confidence_claim = schema["$defs"]["confidence"]["properties"]["claim"]

    assert claim["additionalProperties"] is False
    assert claim["properties"]["classification"] == {"const": "verification"}
    assert claim["properties"]["validation_claim"] == {"const": False}
    assert confidence_claim["additionalProperties"] is False
    for field in (
        "validation_claim",
        "compatibility_claim",
        "hardware_sign_off_allowed",
        "general_dae_support_claimed",
    ):
        assert confidence_claim["properties"][field] == {"const": False}


def test_boost_boundary_schema_accepts_every_actual_producer_document(tmp_path: Path) -> None:
    from pear_sim.sweep import run_boost_boundary_benchmark

    artifacts = run_boost_boundary_benchmark(
        ROOT / "examples" / "boost.yaml",
        tmp_path / "boost-boundary",
        load_ratios=[0.9, 1.0, 1.1],
    )
    points = sorted((artifacts.output_dir / "points").glob("*/benchmark-point.json"))
    assert len(points) == 3
    documents = [
        artifacts.benchmark_spec,
        artifacts.oracle_json,
        *points,
        artifacts.benchmark_result,
        artifacts.sweep_summary,
        artifacts.confidence_json,
    ]
    validator = Draft202012Validator(_schema("boost-boundary-evidence.schema.json"))

    for document in documents:
        validator.validate(json.loads(document.read_text(encoding="utf-8")))
