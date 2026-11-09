# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Hook A: a parameter may be a scalar, a set of corners, or a distribution."""

from __future__ import annotations

import textwrap

import pytest

from pear_sim.errors import ModelError
from pear_sim.spec.model import load_model
from pear_sim.spec.parameter_envelope import Distribution, LifeModel


_TEMPLATE = """schema: pear-sim/v0.1
name: envelope_probe
parameters:
{parameters}
circuit:
  ground: gnd
  components:
    - {{id: vin, type: voltage_source, p: in, n: gnd, value: "${{vin}}"}}
    - {{id: r1, type: resistor, p: in, n: out, value: "${{esr}}"}}
    - {{id: c1, type: capacitor, p: out, n: gnd, value: 100 uF}}
controls: {{}}
analysis:
  transient:
    start: 0 s
    stop: 1 ms
    output_step: 10 us
    solver: dopri5
    rtol: 1.0e-8
    atol: 1.0e-10
outputs:
  - v(out)
"""


def _model(tmp_path, parameters: str):
    block = textwrap.indent(textwrap.dedent(parameters).strip("\n"), "  ")
    path = tmp_path / "probe.yaml"
    path.write_text(_TEMPLATE.format(parameters=block))
    return load_model(path)


ENVELOPE = """
vin: 24 V
esr:
  nominal: 20 mohm
  source: assumption
  note: "Placeholder; no real component is described."
  corners:
    eol_hi: 60 mohm
    eol_lo: 18 mohm
"""


def test_scalar_parameters_declare_no_envelope(tmp_path):
    model = _model(tmp_path, "vin: 24 V\nesr: 20 mohm\n")
    assert model.parameter_envelopes == {}
    assert model.parameters["esr"].value == pytest.approx(0.02)


def test_named_corners_parse_into_si_and_keep_the_nominal(tmp_path):
    model = _model(tmp_path, ENVELOPE)
    envelope = model.parameter_envelopes["esr"]
    # The nominal still lands in `parameters`; the solver sees one number.
    assert model.parameters["esr"].value == pytest.approx(0.02)
    assert envelope.corner_map == {
        "eol_hi": pytest.approx(0.06),
        "eol_lo": pytest.approx(0.018),
    }
    assert envelope.source == "assumption"
    assert envelope.note


def test_envelope_without_provenance_is_rejected(tmp_path):
    """Rejected by the JSON Schema layer; the loader repeats the check."""

    with pytest.raises(ModelError, match=r"source"):
        _model(
            tmp_path,
            """
            vin: 24 V
            esr:
              nominal: 20 mohm
              corners: {eol_hi: 60 mohm}
            """,
        )


def test_envelope_without_any_drift_is_rejected(tmp_path):
    """Provenance alone is a scalar wearing a costume — fail closed."""

    with pytest.raises(ModelError, match="at least one of corners"):
        _model(
            tmp_path,
            """
            vin: 24 V
            esr:
              nominal: 20 mohm
              source: datasheet
              note: "nothing declared"
            """,
        )


def test_unknown_provenance_source_is_rejected(tmp_path):
    with pytest.raises(ModelError, match=r"source.*(must be one of|is not one of)"):
        _model(
            tmp_path,
            """
            vin: 24 V
            esr:
              nominal: 20 mohm
              source: vibes
              note: "n/a"
              corners: {eol_hi: 60 mohm}
            """,
        )


def test_corner_value_must_match_the_nominal_dimension(tmp_path):
    with pytest.raises(ModelError):
        _model(
            tmp_path,
            """
            vin: 24 V
            esr:
              nominal: 20 mohm
              source: assumption
              note: "wrong dimension below"
              corners: {eol_hi: 60 uF}
            """,
        )


def test_distribution_requires_an_explicit_seed(tmp_path):
    with pytest.raises(ModelError, match="seed"):
        _model(
            tmp_path,
            """
            vin: 24 V
            esr:
              nominal: 20 mohm
              source: assumption
              note: "no seed"
              distribution: {type: normal, sigma_percent: 10}
            """,
        )


def test_sampling_is_reproducible_from_the_seed():
    distribution = Distribution(
        kind="normal", seed=20260727, low_si=0.0, high_si=0.04, sigma_si=0.002
    )
    first = distribution.sample(0.02, 8)
    second = distribution.sample(0.02, 8)
    assert first == second
    assert all(0.02 - 3 * 0.002 <= value <= 0.02 + 3 * 0.002 for value in first)


def test_uniform_and_triangular_stay_inside_their_support():
    uniform = Distribution(kind="uniform", seed=7, low_si=1.0, high_si=2.0)
    triangular = Distribution(
        kind="triangular", seed=7, low_si=1.0, high_si=2.0, mode_si=1.2
    )
    assert all(1.0 <= value <= 2.0 for value in uniform.sample(1.5, 32))
    assert all(1.0 <= value <= 2.0 for value in triangular.sample(1.5, 32))


def test_life_model_moves_the_value_across_the_mission():
    linear = LifeModel(kind="linear", end_of_life_percent=10.0)
    assert linear.value_at(100.0, 0.0) == pytest.approx(100.0)
    assert linear.value_at(100.0, 1.0) == pytest.approx(110.0)
    # Leakage grows by decades — the geometric form is the one that fits.
    geometric = LifeModel(kind="geometric", end_of_life_ratio=100.0)
    assert geometric.value_at(1e-9, 0.0) == pytest.approx(1e-9)
    assert geometric.value_at(1e-9, 0.5) == pytest.approx(1e-8)
    assert geometric.value_at(1e-9, 1.0) == pytest.approx(1e-7)


def test_envelopes_reach_the_resolved_model_only_when_declared(tmp_path):
    from pear_sim.run.contracts import resolved_model_dict

    plain = resolved_model_dict(_model(tmp_path, "vin: 24 V\nesr: 20 mohm\n"))
    assert "parameter_envelopes" not in plain

    declared = resolved_model_dict(_model(tmp_path, ENVELOPE))
    payload = declared["parameter_envelopes"]["esr"]
    assert payload["source"] == "assumption"
    assert payload["corners_si"]["eol_hi"] == pytest.approx(0.06)
    # The physics hash must not notice a declaration about drift.
    assert plain["hashes"]["model_sha256"] == declared["hashes"]["model_sha256"]
