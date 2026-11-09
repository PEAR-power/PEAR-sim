# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Characterization tests for the model-format input layer.

These tests pin behaviour that predates them, so that any change to the
surface syntax, the loader, or the elaboration step trips a red test instead
of silently moving the contract. They are the safety net required BEFORE any
future model-format work — they assert what IS, not what ought to be.

Three pins and one boundary:

1. the YAML and JSON surfaces resolve the same document to the identical
   model (identical physics/request hashes and resolved snapshot) — the JSON
   branch of the loader had no test at all before this file;
2. the full resolved-model snapshot of the canonical buck example is stable
   field for field, not merely hash for hash;
3. physics/request hashes are pinned for examples exercising the newer
   mechanisms (couplings, NPC 5L, thyristor, PMSM machine expansion) — the
   legacy pin in test_contracts.py covers only pre-coupling models;
4. YAML node reuse (anchors, aliases, merge keys) is rejected at load time.
   Until node reuse was rejected this file only PROBED that reuse was silently accepted;
   the probe is now the regression test for the rejection.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from pear_sim.errors import ModelError
from pear_sim.run.contracts import model_hashes, resolved_model_dict
from pear_sim.spec.model import _ModelSafeLoader, load_model

import pytest


ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
GOLDEN = ROOT / "tests" / "data" / "buck_resolved_golden.json"

# Deliberately spans the format's expressiveness: parameters + ${...}
# references (buck), a multilevel modulator block (npc midpoint), a latching
# component (thyristor) and magnetic couplings (6-pulse transformer).
EQUIVALENCE_EXAMPLES = (
    "buck.yaml",
    "npc_three_phase_midpoint.yaml",
    "thyristor_phase_controlled.yaml",
    "rectifier_6pulse_transformer.yaml",
)

# Pinned with the same intent as test_m32_preserves_legacy_pwm_physics_and_
# request_hashes, extended to models using later mechanisms (couplings, multilevel, machines). A change here
# is a change to canonicalization or elaboration and needs a migration note,
# never a silent re-pin.
PINNED_HASHES = {
    "rectifier_6pulse_transformer.yaml": (
        "ec0fac22f0dc4f1b70cdf3420139aa533f47ee976fc98e6af9ada191212ebd56",
        "0a3d6cc5f7bcab679fae4d95126a28d07a884e902b41123767da3a1172ef41a5",
    ),
    "npc5_three_phase.yaml": (
        "24f1be849b1a241594bbfd13bd3c075c7a1ae8b03bff0a6e0ca9df06a0df4892",
        "256ef0dbe1b435dde6eb240637794c935a27869601bf1273f041234a6056673a",
    ),
    "thyristor_phase_controlled.yaml": (
        "c42eef750d5bfbb614c5c37bf38d7320afeceac456680d891f9eed3ceb18af66",
        "a7800acc5ce758cc1755212dff42c91aa785c2d653088e7f9b8ef30829b22f48",
    ),
    "pmsm_inverter_drive.yaml": (
        "a11e0a99365ded8831bc0797bb17557a40de6b1a39c6ac2d88c043f161248970",
        "34ef2e596dcf14201400a31abe7d2206bd179f49d30ec7646c9fbfaa71fd5d27",
    ),
    "npc_three_phase_midpoint.yaml": (
        "23f88a28a97dbb00f9714ed1f92c2a7b48e1a6010cdbaca129c2b8b44f6183e0",
        "282b39de3d89245f7b329eac87c71724d417c135f03ae286b6c0545edad97646",
    ),
}


def _without_source_identity(resolved: dict) -> dict:
    """Drop the only fields legitimately tied to the source FILE, not the model.

    ``source_sha256`` hashes the bytes of the file and ``source_file`` names
    it; both must differ between a .yaml and its .json rendering while
    everything else must not.
    """

    trimmed = json.loads(json.dumps(resolved))
    del trimmed["source_file"]
    del trimmed["hashes"]["source_sha256"]
    return trimmed


@pytest.mark.parametrize("filename", EQUIVALENCE_EXAMPLES)
def test_yaml_and_json_surfaces_resolve_to_the_identical_model(
    filename: str, tmp_path: Path
) -> None:
    yaml_path = EXAMPLES / filename
    # The conversion must go through the loader's OWN YAML semantics
    # (_ModelSafeLoader adds the exponent-only float resolver); converting
    # with plain safe_load could turn `1e-10` into a string and make this
    # test fail for a reason that is not a surface divergence.
    document = yaml.load(yaml_path.read_bytes(), Loader=_ModelSafeLoader)
    json_path = tmp_path / (yaml_path.stem + ".json")
    json_path.write_text(json.dumps(document), encoding="utf-8")

    from_yaml = load_model(yaml_path)
    from_json = load_model(json_path)

    yaml_hashes = model_hashes(from_yaml)
    json_hashes = model_hashes(from_json)
    assert yaml_hashes["model_sha256"] == json_hashes["model_sha256"]
    assert yaml_hashes["request_sha256"] == json_hashes["request_sha256"]
    # The file bytes differ, so the source hash must differ — if it does not,
    # the test compared a file with itself.
    assert yaml_hashes["source_sha256"] != json_hashes["source_sha256"]

    assert _without_source_identity(
        resolved_model_dict(from_yaml)
    ) == _without_source_identity(resolved_model_dict(from_json))


def test_resolved_model_snapshot_for_buck_is_stable() -> None:
    resolved = resolved_model_dict(load_model(EXAMPLES / "buck.yaml"))
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    # Field-for-field, not hash-for-hash: a change that reshapes the resolved
    # document while accidentally preserving its hash still fails here, and
    # the assertion diff names the exact field that moved.
    assert resolved == golden


def test_physics_and_request_hashes_are_pinned_for_newer_features() -> None:
    assert {
        filename: (
            model_hashes(load_model(EXAMPLES / filename))["model_sha256"],
            model_hashes(load_model(EXAMPLES / filename))["request_sha256"],
        )
        for filename in PINNED_HASHES
    } == PINNED_HASHES


def _write(source: Path, body: str) -> Path:
    source.write_text(body.strip() + "\n", encoding="utf-8")
    return source


# Each document below carries node reuse in a different position. The parser
# expands all of them before the JSON Schema or `_reject_unknown` can see the
# result, which is why the rejection has to live in the loader class itself.
NODE_REUSE_CASES = {
    "anchored_component_reused_by_merge_key": """
schema: pear-sim/v0.1
name: reuse_probe
circuit:
  ground: gnd
  components:
    - &tpl {id: r1, type: resistor, p: in, n: gnd, value: 10 ohm}
    - {<<: *tpl, id: r2, p: in2}
    - {id: s1, type: voltage_source, p: in, n: gnd, value: 1 V}
    - {id: s2, type: voltage_source, p: in2, n: gnd, value: 1 V}
analysis:
  transient: {stop: 1 ms, output_step: 100 us}
outputs: [v(in), v(in2)]
""",
    "anchored_scalar_parameter": """
schema: pear-sim/v0.1
name: reuse_probe
parameters:
  a: &shared 10 ohm
  b: *shared
circuit:
  ground: gnd
  components:
    - {id: r1, type: resistor, p: in, n: gnd, value: "${a}"}
    - {id: s1, type: voltage_source, p: in, n: gnd, value: 1 V}
analysis:
  transient: {stop: 1 ms, output_step: 100 us}
outputs: [v(in)]
""",
    "inline_merge_key_without_any_alias": """
schema: pear-sim/v0.1
name: reuse_probe
circuit:
  ground: gnd
  components:
    - {<<: {type: resistor, value: 10 ohm}, id: r1, p: in, n: gnd}
    - {id: s1, type: voltage_source, p: in, n: gnd, value: 1 V}
analysis:
  transient: {stop: 1 ms, output_step: 100 us}
outputs: [v(in)]
""",
    "alias_with_no_anchor_at_all": """
schema: pear-sim/v0.1
name: reuse_probe
circuit:
  ground: gnd
  components:
    - *ghost
    - {id: s1, type: voltage_source, p: in, n: gnd, value: 1 V}
analysis:
  transient: {stop: 1 ms, output_step: 100 us}
outputs: [v(in)]
""",
}


@pytest.mark.parametrize("case", sorted(NODE_REUSE_CASES))
def test_yaml_node_reuse_is_rejected_with_its_position(
    case: str, tmp_path: Path
) -> None:
    source = _write(tmp_path / f"{case}.yaml", NODE_REUSE_CASES[case])

    with pytest.raises(ModelError) as excinfo:
        load_model(source)

    message = str(excinfo.value)
    assert "node reuse is not part of pear-sim/v0.1" in message
    # The diagnostic must point at a line, or it is useless in a fifty-component
    # model — and it must name the remedy, not merely the prohibition.
    assert "line " in message and "column " in message
    assert "subcircuits:" in message


def test_a_custom_yaml_tag_is_rejected(tmp_path: Path) -> None:
    """The safe loader has no constructor for a custom tag; that stays an error.

    Not the node-reuse message: this one comes from PyYAML and is wrapped as a
    parse failure. Pinned here so the closed door is a tested door.
    """

    source = _write(
        tmp_path / "tagged.yaml",
        """
schema: pear-sim/v0.1
name: tagged
circuit:
  ground: gnd
  components:
    - !mycomponent {id: r1, type: resistor, p: in, n: gnd, value: 10 ohm}
    - {id: s1, type: voltage_source, p: in, n: gnd, value: 1 V}
analysis:
  transient: {stop: 1 ms, output_step: 100 us}
outputs: [v(in)]
""",
    )

    with pytest.raises(ModelError):
        load_model(source)


def test_every_repository_example_still_loads_without_node_reuse() -> None:
    """The prohibition costs nothing today — and this proves it, per example.

    A single example relying on an anchor would make the rejection a breaking
    change rather than a tightening.
    """

    examples = sorted(EXAMPLES.glob("*.yaml"))
    assert len(examples) >= 25
    for path in examples:
        load_model(path)
