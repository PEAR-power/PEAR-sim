# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Gate for `DesignSpec`.

The acceptance criteria, each pinned by a test:
hard/soft mixing is a validation error; a missing field becomes a question
with context, never a default; a docs-filled value carries provenance (which
unit, who approved) and a human always outranks it; an assumption without a
named author is impossible.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pear_sim.design import (
    DesignSpec,
    DesignSpecError,
    IncompleteDesignSpecError,
)

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "design_specs" / "isolated_aux.yaml"

def _example_dict() -> dict:
    return yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Loading and the closed vocabulary
# ---------------------------------------------------------------------------


def test_the_example_specification_is_complete_and_hashable() -> None:
    spec = DesignSpec.load(EXAMPLE)
    assert spec.complete
    assert spec.questions() == []
    digest = spec.spec_sha256()
    assert digest == DesignSpec.load(EXAMPLE).spec_sha256()  # deterministic
    # Provenance separates the owner's declarations from the assumption.
    assert spec.values["isolation"].basis == "declared"
    ambient = spec.values["environment.ambient_c"]
    assert ambient.basis == "assumption"
    assert ambient.rationale is not None


def test_an_unknown_field_is_an_error_not_a_shrug() -> None:
    document = _example_dict()
    document["hard"]["ripple_mv"] = 50
    with pytest.raises(DesignSpecError, match="unknown specification field 'ripple_mv'"):
        DesignSpec.from_dict(document)


def test_mixing_hard_and_soft_sections_is_a_validation_error() -> None:
    # A soft target in the hard section would silently shrink the candidate
    # space during screening...
    document = _example_dict()
    document["hard"]["targets"] = {"efficiency_pct": 95}
    with pytest.raises(DesignSpecError, match="soft"):
        DesignSpec.from_dict(document)
    # ...and a hard reject in the soft section would let an infeasible
    # design win the Pareto comparison.
    document = _example_dict()
    document["soft"]["isolation"] = "required"
    with pytest.raises(DesignSpecError, match="hard"):
        DesignSpec.from_dict(document)


def test_leaf_values_are_validated_fail_closed() -> None:
    document = _example_dict()
    document["hard"]["input"]["voltage_v"] = {"min": 75, "max": 36}
    with pytest.raises(DesignSpecError, match="min 75.0 exceeds max 36.0"):
        DesignSpec.from_dict(document)


# ---------------------------------------------------------------------------
# Questions instead of defaults
# ---------------------------------------------------------------------------


def test_a_missing_required_field_becomes_a_question_with_context() -> None:
    document = _example_dict()
    del document["hard"]["isolation"]
    spec = DesignSpec.from_dict(document)
    assert not spec.complete
    (question,) = spec.questions()
    assert question.field == "isolation"
    assert "family-level reject" in question.why  # the "what for" context
    # No default was invented anywhere:
    assert "isolation" not in spec.values
    with pytest.raises(IncompleteDesignSpecError, match="isolation"):
        spec.require_complete()
    with pytest.raises(IncompleteDesignSpecError):
        spec.spec_sha256()  # an incomplete spec has no identity


def test_the_grid_scenario_is_required_exactly_for_ac_inputs() -> None:
    # DC input: no grid question (the example proves it).
    assert DesignSpec.load(EXAMPLE).complete
    # AC input without a grid scenario: the grid is a specification
    # parameter, not a run setting — so it becomes a question.
    document = _example_dict()
    document["hard"]["input"]["kind"] = "ac_three_phase"
    spec = DesignSpec.from_dict(document)
    assert [question.field for question in spec.questions()] == ["grid"]
    # Declaring the scenario closes it.
    document["hard"]["grid"] = {"scr": 8.5, "voltage_unbalance_pct": 2.0}
    assert DesignSpec.from_dict(document).complete


# ---------------------------------------------------------------------------
# Assumptions: every value has an author
# ---------------------------------------------------------------------------


def test_an_assumption_without_rationale_is_rejected() -> None:
    document = _example_dict()
    document["assumptions"][0].pop("rationale")
    with pytest.raises(DesignSpecError, match="rationale"):
        DesignSpec.from_dict(document)


def test_an_assumption_shadowing_a_declared_value_is_ambiguous() -> None:
    document = _example_dict()
    document["assumptions"].append(
        {
            "field": "isolation",
            "value": "not_required",
            "author": {"id": "x", "name": "X"},
            "rationale": "contradicts the declared value on purpose",
        }
    )
    with pytest.raises(DesignSpecError, match="both declared and assumed"):
        DesignSpec.from_dict(document)


# ---------------------------------------------------------------------------
# Packaging and CLI
# ---------------------------------------------------------------------------


def test_the_packaged_schema_copy_is_byte_identical() -> None:
    repo = (ROOT / "schema" / "design-spec.schema.json").read_bytes()
    packaged = (
        ROOT / "src" / "pear_sim" / "design" / "design-spec.schema.json"
    ).read_bytes()
    assert repo == packaged


def test_cli_spec_reports_completeness_and_questions(tmp_path, capsys) -> None:
    from pear_sim.api.cli import main

    assert main(["spec", str(EXAMPLE), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["complete"] is True
    assert payload["spec_sha256"]

    document = _example_dict()
    del document["hard"]["isolation"]
    incomplete = tmp_path / "incomplete.yaml"
    incomplete.write_text(yaml.safe_dump(document), encoding="utf-8")
    assert main(["spec", str(incomplete), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["complete"] is False
    assert payload["open_questions"][0]["field"] == "isolation"
    assert "spec_sha256" not in payload
