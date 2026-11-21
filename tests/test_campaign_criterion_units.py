# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""A campaign criterion takes its dimension from the measured signal.

``unit`` used to be a display label next to a bare number, so a 40 V output
passed ``v(out) <= 50 A`` with a margin of +10.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pear_sim.evidence.perturbation_campaign import (
    REQUEST_SCHEMA,
    _parse_criteria,
    run_campaign,
)
from pear_sim.spec.model import load_model
from pear_sim.spec.perturbation import PerturbationError

BUCK = Path(__file__).parents[1] / "examples" / "buck.yaml"
SYNC_BUCK = Path(__file__).parents[1] / "examples" / "synchronous_buck_complementary.yaml"


def _criterion(**fields) -> dict:
    entry = {
        "id": "c", "signal": "v(out)", "field": "mean_last_half", "op": "max",
        "limit": 50, "source": "assumption", "note": "placeholder",
    }
    entry.update(fields)
    return {key: value for key, value in entry.items() if value is not None}


def _parse(model: Path = BUCK, **fields):
    (criterion,) = _parse_criteria([_criterion(**fields)], load_model(model))
    return criterion


def _run(tmp_path: Path, name: str, **fields) -> dict:
    request = tmp_path / f"{name}.json"
    request.write_text(
        json.dumps(
            {
                "schema": REQUEST_SCHEMA,
                "label": name,
                "axes": [
                    {
                        "id": "vin", "kind": "parameter_values", "parameter": "vin",
                        "values": ["100 V"], "source": "assumption", "note": "nominal",
                    }
                ],
                "criteria": [_criterion(**fields)],
            }
        )
    )
    artifacts = run_campaign(BUCK, request, tmp_path / name, jobs=1)
    return json.loads(artifacts.summary.read_text())


def test_a_voltage_cannot_be_limited_in_amperes(tmp_path) -> None:
    with pytest.raises(PerturbationError, match=r"unit 'A' does not match v\(out\)"):
        _run(tmp_path, "volts_vs_amperes", limit=50, unit="A")
    assert not (tmp_path / "volts_vs_amperes").exists()  # refused before any run


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"limit": "50 mA"}, "expected voltage, got current"),
        ({"limit": "50 mA", "unit": "V"}, "expected voltage, got current"),
        ({"signal": "i(l1)", "limit": 4.1, "unit": "V"}, "does not match i"),
        ({"limit": 50, "unit": "furlong"}, "does not match"),
        ({"limit": "50"}, "expected voltage, got dimensionless"),
        ({"limit": True}, "limit must be a number"),
        ({"limit": float("inf")}, "must be finite"),
        ({"op": "abs_deviation_max", "limit": 1, "reference": "40 A"}, "expected voltage"),
        ({"op": "abs_deviation_max", "limit": 1}, "reference is required"),
        ({"field": "median"}, "field must be one of"),
        ({"signal": "vout"}, "not a canonical signal expression"),
    ],
)
def test_a_criterion_of_the_wrong_dimension_is_a_request_error(fields, message) -> None:
    with pytest.raises(PerturbationError, match=message):
        _parse(**fields)


def test_a_scaled_unit_next_to_a_bare_number_is_ambiguous() -> None:
    """``limit: 50, unit: mV`` — 50 V or 50 mV? The producer will not guess."""

    with pytest.raises(PerturbationError, match="ambiguous"):
        _parse(limit=50, unit="mV")
    assert _parse(limit="50 mV", unit="mV").limit == pytest.approx(0.05)


@pytest.mark.parametrize(
    "fields",
    [
        {"limit": "50 mV", "unit": "V"},
        {"limit": 0.05, "unit": "V"},
        {"limit": "50 mV", "unit": None},
        {"limit": 0.05, "unit": None},
        {"limit": "0.05 V", "unit": "v"},
    ],
)
def test_equivalent_spellings_of_one_limit_normalize_to_the_same_criterion(fields) -> None:
    criterion = _parse(**fields)
    assert criterion.limit == pytest.approx(0.05, rel=1e-12)
    assert criterion.unit == "V"
    assert criterion.as_dict()["unit"] == "V"


def test_reference_and_limit_share_the_measured_unit() -> None:
    criterion = _parse(op="abs_deviation_max", limit="500 mV", reference="40 V", unit="V")
    assert criterion.bounds() == pytest.approx((39.5, 40.5))


def test_a_gate_frequency_criterion_is_measured_in_hertz() -> None:
    criterion = _parse(
        SYNC_BUCK, signal="leg.high", field="fsw_measured_hz", op="min", limit="19 kHz"
    )
    assert criterion.limit == pytest.approx(19e3) and criterion.unit == "Hz"
    with pytest.raises(PerturbationError, match="does not match"):
        _parse(SYNC_BUCK, signal="leg.high", field="fsw_measured_hz", limit=19e3, unit="V")


def test_50_mv_and_0_05_v_give_the_same_negative_margin(tmp_path) -> None:
    """The 40 V output against a 50 mV ceiling fails identically in both spellings."""

    spelled = _run(tmp_path, "spelled", limit="50 mV", unit="V")
    plain = _run(tmp_path, "plain", limit=0.05, unit="V")
    verdicts = [
        summary["runs"][0]["criteria"]["c"] for summary in (spelled, plain)
    ]
    assert verdicts[0] == verdicts[1]
    assert verdicts[0]["status"] == "fail"
    assert verdicts[0]["margin"] == pytest.approx(0.05 - 40.0, abs=1e-3)
    assert spelled["criteria"] == plain["criteria"]
    assert spelled["overall_status"] == plain["overall_status"] == "fail"


def test_a_limit_in_the_measured_unit_still_passes(tmp_path) -> None:
    summary = _run(tmp_path, "volts", limit=50, unit="V")
    verdict = summary["runs"][0]["criteria"]["c"]
    assert verdict["status"] == "pass"
    assert verdict["margin"] == pytest.approx(10.0, abs=1e-3)
