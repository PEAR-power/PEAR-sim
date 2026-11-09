# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Component model_meta declaration: parsing, resolved-model echo, run warning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pear_sim.contracts import model_hashes, resolved_model_dict
from pear_sim.errors import ModelError
from pear_sim.io import write_result
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "buck_model_meta.yaml"
PLAIN = ROOT / "examples" / "buck.yaml"


def _write(tmp_path: Path, text: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "model.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _meta_model(tmp_path: Path, meta_yaml: str) -> Path:
    return _write(
        tmp_path,
        f"""
schema: pear-sim/v0.1
name: rc
circuit:
  ground: gnd
  components:
    - {{id: v1, type: voltage_source, p: a, n: gnd, value: 1 V}}
    - id: r1
      type: resistor
      p: a
      n: b
      value: 1 ohm
{meta_yaml}
    - {{id: c1, type: capacitor, p: b, n: gnd, value: 1 uF}}
analysis:
  transient: {{stop: 1 us, output_step: 0.5 us}}
outputs:
  - v(b)
""",
    )


def test_model_meta_is_parsed_and_echoed_in_resolved_model() -> None:
    model = load_model(EXAMPLE)
    by_id = {component.id: component for component in model.components}
    assert by_id["q1"].model_meta is not None
    assert by_id["q1"].model_meta.version == "ideal-switch-datasheet-stub/1.0"
    assert by_id["q1"].model_meta.frequency_max_hz == 250000.0
    assert by_id["q1"].model_meta.temperature_min_c == -40.0
    assert by_id["l1"].model_meta.dvdt_max_v_per_s is None
    assert by_id["d1"].model_meta is None

    resolved = resolved_model_dict(model)
    components = {entry["id"]: entry for entry in resolved["physics"]["components"]}
    q1 = components["q1"]["model_meta"]
    assert q1["version"] == "ideal-switch-datasheet-stub/1.0"
    assert q1["evaluated"] is False
    assert q1["validity"]["frequency_max_hz"] == 250000.0
    l1 = components["l1"]["model_meta"]
    assert l1["validity"] == {"frequency_max_hz": 100000.0}
    assert "model_meta" not in components["d1"]


def test_meta_free_model_hashes_are_unchanged_by_the_feature() -> None:
    plain = load_model(PLAIN)
    resolved = resolved_model_dict(plain)
    assert all(
        "model_meta" not in entry for entry in resolved["physics"]["components"]
    )
    hashes = model_hashes(plain)
    assert set(hashes) >= {"source_sha256", "model_sha256", "request_sha256"}


def test_declaring_meta_changes_the_model_hash(tmp_path: Path) -> None:
    without = _meta_model(tmp_path / "a", "")
    with_meta = _meta_model(
        tmp_path / "b",
        '      model_meta: {version: "r/1", validity: {frequency_max_hz: 1000}}\n',
    )
    assert (
        model_hashes(load_model(without))["model_sha256"]
        != model_hashes(load_model(with_meta))["model_sha256"]
    )


def test_run_bundle_flags_declared_meta_as_not_evaluated(tmp_path: Path) -> None:
    model = load_model(EXAMPLE)
    result = HybridSimulator(model).simulate()
    run_dir = tmp_path / "run"
    write_result(result, run_dir)
    warnings = json.loads((run_dir / "warnings.json").read_text())
    flagged = [w for w in warnings if w["code"] == "MODEL_META_DECLARED_NOT_EVALUATED"]
    assert len(flagged) == 1
    assert "l1" in flagged[0]["message"] and "q1" in flagged[0]["message"]
    resolved = json.loads((run_dir / "model.resolved.json").read_text())
    components = {entry["id"]: entry for entry in resolved["physics"]["components"]}
    assert components["q1"]["model_meta"]["evaluated"] is False


def test_meta_free_run_has_no_model_meta_warning(tmp_path: Path) -> None:
    result = HybridSimulator(load_model(PLAIN)).simulate()
    run_dir = tmp_path / "run"
    write_result(result, run_dir)
    warnings = json.loads((run_dir / "warnings.json").read_text())
    assert not [w for w in warnings if w["code"] == "MODEL_META_DECLARED_NOT_EVALUATED"]


@pytest.mark.parametrize(
    ("meta_yaml", "match"),
    [
        ('      model_meta: {validity: {}}\n', "version"),
        ('      model_meta: {version: "r/1", vendor: x}\n', "unknown field"),
        (
            '      model_meta: {version: "r/1", validity: {frequency_max_hz: 0}}\n',
            "must be positive",
        ),
        (
            '      model_meta: {version: "r/1", validity: {frequency_max_hz: .nan}}\n',
            "finite",
        ),
        (
            '      model_meta: {version: "r/1", validity: {frequency_max_hz: "10 kHz"}}\n',
            "plain finite number",
        ),
        (
            "      model_meta:\n"
            '        version: "r/1"\n'
            "        validity: {temperature_min_c: 100, temperature_max_c: 25}\n",
            "must not exceed",
        ),
        (
            '      model_meta: {version: "r/1", validity: {dvdt_max_v_per_s: -1}}\n',
            "must be positive",
        ),
    ],
)
def test_invalid_model_meta_is_rejected(tmp_path: Path, meta_yaml: str, match: str) -> None:
    path = _meta_model(tmp_path, meta_yaml)
    with pytest.raises(ModelError, match=match):
        load_model(path)
