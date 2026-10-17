"""The JSON Schema is enforced by the loader, not merely documented."""

from __future__ import annotations

from pathlib import Path

import pytest

from pear_sim.errors import ModelError
from pear_sim.model import load_model

ROOT = Path(__file__).parents[1]


def test_packaged_schema_copy_is_byte_identical_to_the_canonical_schema() -> None:
    canonical = (ROOT / "schema" / "model.schema.json").read_bytes()
    packaged = (ROOT / "src" / "pear_sim" / "spec" / "model.schema.json").read_bytes()
    assert canonical == packaged, (
        "schema/model.schema.json and src/pear_sim/spec/model.schema.json diverged; "
        "update both in the same change"
    )


def test_every_repository_example_conforms_to_the_enforced_schema() -> None:
    examples = sorted((ROOT / "examples").glob("*.yaml")) + sorted(
        (ROOT / "examples").glob("*.json")
    )
    models = [path for path in examples if "expectations" not in path.name]
    loaded = []
    for path in models:
        try:
            loaded.append(load_model(path).name)
        except ModelError:
            # Non-model JSON inputs (sweep requests, compare profiles) are not
            # circuit documents; only YAML circuit examples must load.
            if path.suffix == ".yaml":
                raise
    assert len(loaded) >= 6


def test_structurally_invalid_documents_are_rejected_by_the_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
schema: pear-sim/v0.1
surprise: true
circuit:
  components:
    - {id: r1, type: resistor, p: a, n: gnd, value: 1 ohm}
    - {id: v1, type: voltage_source, p: a, n: gnd, value: 1 V}
analysis:
  transient: {stop: 1 us, output_step: 0.5 us}
outputs: [v(a)]
""",
        encoding="utf-8",
    )
    with pytest.raises(ModelError, match="does not conform to schema.*unknown field"):
        load_model(path)


def test_schema_rejection_reports_a_json_pointer_location(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """
schema: pear-sim/v0.1
circuit:
  components:
    - {id: r1, type: resistor, p: a, n: gnd, value: 1 ohm, vendor: acme}
    - {id: v1, type: voltage_source, p: a, n: gnd, value: 1 V}
analysis:
  transient: {stop: 1 us, output_step: 0.5 us}
outputs: [v(a)]
""",
        encoding="utf-8",
    )
    with pytest.raises(ModelError, match="/circuit/components/0.*vendor"):
        load_model(path)
