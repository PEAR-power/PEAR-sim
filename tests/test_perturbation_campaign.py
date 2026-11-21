# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Campaign producer: axes, criteria, provenance enforcement, bisection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pear_sim.evidence.perturbation_campaign import (
    REQUEST_SCHEMA,
    run_campaign,
)
from pear_sim.reporting.perturbation_report import render_campaign_report
from pear_sim.spec.perturbation import PerturbationError


EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
WCA = EXAMPLES_DIR / "buck_ccm_wca.yaml"

_CRITERION = {
    "id": "vout_regulation",
    "signal": "v(out)",
    "field": "mean_last_half",
    "op": "abs_deviation_max",
    "reference": 12.0,
    "limit": 1.0,
    "unit": "V",
    "source": "assumption",
    "note": "placeholder budget",
}


def _write(tmp_path, payload, name="request.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def _request(**overrides):
    payload = {
        "schema": REQUEST_SCHEMA,
        "label": "probe",
        "nominal": {"vout_v": 12.0, "recovery_band_percent": 2.0},
        "axes": [{"id": "vref", "kind": "parameter_corners", "parameter": "vref"}],
        "criteria": [_CRITERION],
    }
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------
# request validation — the provenance rule is the load-bearing one
# --------------------------------------------------------------------------


def test_an_axis_that_states_its_own_values_must_state_their_provenance(tmp_path):
    request = _request(
        axes=[
            {
                "id": "load",
                "kind": "parameter_values",
                "parameter": "load",
                "values": ["2 ohm", "4 ohm"],
            }
        ]
    )
    with pytest.raises(PerturbationError, match="source must be one of"):
        run_campaign(WCA, _write(tmp_path, request), tmp_path / "out")


def test_a_criterion_must_state_its_provenance(tmp_path):
    criterion = dict(_CRITERION)
    criterion.pop("note")
    request = _request(criteria=[criterion])
    with pytest.raises(PerturbationError, match="note"):
        run_campaign(WCA, _write(tmp_path, request), tmp_path / "out")


def test_a_corner_axis_requires_a_declared_envelope(tmp_path):
    request = _request(
        axes=[{"id": "vin", "kind": "parameter_corners", "parameter": "vin"}]
    )
    with pytest.raises(PerturbationError, match="declares no drift envelope"):
        run_campaign(WCA, _write(tmp_path, request), tmp_path / "out")


def test_the_run_cap_is_enforced_before_anything_executes(tmp_path):
    request = _request(
        axes=[
            {
                "id": f"a{index}",
                "kind": "parameter_values",
                "parameter": "load",
                "values": [f"{value} ohm" for value in range(2, 8)],
                "source": "assumption",
                "note": "cap probe",
            }
            for index in range(4)
        ]
    )
    with pytest.raises(PerturbationError, match="run cap"):
        run_campaign(WCA, _write(tmp_path, request), tmp_path / "out")


def test_the_request_schema_is_pinned(tmp_path):
    request = _request()
    request["schema"] = "pear-sim/something-else/v9"
    with pytest.raises(PerturbationError, match="schema must be"):
        run_campaign(WCA, _write(tmp_path, request), tmp_path / "out")


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def corner_campaign(tmp_path_factory):
    destination = tmp_path_factory.mktemp("campaign") / "out"
    request = {
        "schema": REQUEST_SCHEMA,
        "label": "vref corners",
        "nominal": {"vout_v": 12.0, "recovery_band_percent": 2.0},
        "axes": [{"id": "vref", "kind": "parameter_corners", "parameter": "vref"}],
        "criteria": [_CRITERION],
    }
    request_path = destination.parent / "request.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    request_path.write_text(json.dumps(request))
    artifacts = run_campaign(WCA, request_path, destination)
    summary = json.loads(artifacts.summary.read_text())
    return artifacts, summary


def test_a_corner_axis_runs_nominal_plus_every_declared_corner(corner_campaign):
    artifacts, summary = corner_campaign
    assert artifacts.run_count == 3
    labels = {run["label"] for run in summary["runs"]}
    assert labels == {"nom", "eol_lo", "eol_hi"}


def test_every_run_retains_the_exact_source_that_was_simulated(corner_campaign):
    artifacts, summary = corner_campaign
    for run in summary["runs"]:
        materialized = artifacts.output_dir / run["materialized_model"]
        assert materialized.exists()
        assert "schema: pear-sim/v0.1" in materialized.read_text()


def test_provenance_travels_from_the_model_envelope_into_the_summary(corner_campaign):
    _artifacts, summary = corner_campaign
    (axis,) = summary["axes"]
    assert axis["source"] == "assumption"
    assert axis["note"]
    assert summary["criteria"][0]["source"] == "assumption"


def test_protection_is_reported_as_not_modelled_never_as_did_not_trip(corner_campaign):
    _artifacts, summary = corner_campaign
    for run in summary["runs"]:
        if run["execution_status"] == "pass":
            assert run["metrics"]["protection"] == "not_modelled"
    assert "not modelled" in summary["protection_modelling"].lower()


def test_the_summary_states_its_claim_boundary(corner_campaign):
    _artifacts, summary = corner_campaign
    claim = summary["claim"].lower()
    assert "monotonicity" in claim
    assert "requirement" in claim
    assert "cross section" in claim


def test_criteria_carry_a_signed_margin(corner_campaign):
    _artifacts, summary = corner_campaign
    for run in summary["runs"]:
        if run["execution_status"] != "pass":
            continue
        verdict = run["criteria"]["vout_regulation"]
        assert verdict["status"] in ("pass", "fail")
        assert (verdict["margin"] >= 0.0) == (verdict["status"] == "pass")


def test_the_report_is_self_contained_and_carries_the_disclaimer(corner_campaign):
    artifacts, summary = corner_campaign
    page = artifacts.report_html.read_text()
    assert page.startswith("<!doctype html>")
    assert "http://" not in page and "https://" not in page.replace(
        "https://json-schema.org", ""
    )
    for phrase in ("cross section", "assumption", "requirement", "not modelled"):
        assert phrase in page.lower()
    # Rendering is a pure projection: it must not need the producer.
    assert render_campaign_report(summary, artifacts.output_dir)


# --------------------------------------------------------------------------
# boundary search — the actual product
# --------------------------------------------------------------------------


def test_bisection_reports_a_tolerable_drift_and_its_assumption(tmp_path):
    request = _request(
        criteria=[{**_CRITERION, "limit": 0.3}],
        axes=[
            {
                "id": "anchor",
                "kind": "parameter_values",
                "parameter": "vin",
                "values": ["24 V"],
                "source": "assumption",
                "note": "single anchor point",
            }
        ],
        boundary_search={
            "parameter": "vref",
            "from": "12 V",
            "to": "16 V",
            "iterations": 4,
            "source": "assumption",
            "note": "bracket chosen to contain the failure",
        },
    )
    artifacts = run_campaign(WCA, _write(tmp_path, request), tmp_path / "out")
    summary = json.loads(artifacts.summary.read_text())
    boundary = summary["boundary_search"]

    assert boundary["verdict"] == "bisected"
    # The reference tolerates a few percent of upward drift, not four volts.
    assert 12.0 < boundary["max_passing_value_si"] < 12.6
    assert 0.0 < boundary["max_passing_drift_percent"] < 5.0
    assert "monotone" in boundary["assumption"]
    # Both bracket ends are probed explicitly so the assumption is inspectable.
    tags = {probe["tag"] for probe in boundary["probes"]}
    assert {"bracket-low", "bracket-high"} <= tags
    assert boundary["source"] == "assumption"


# --------------------------------------------------------------------------
# switching frequency read back from events.jsonl — one period per gate
# --------------------------------------------------------------------------

from pear_sim.evidence.perturbation_campaign import (  # noqa: E402
    _switching_frequency_by_gate,
)

_PERIOD = 50e-6  # 20 kHz


def _events_dir(tmp_path, events):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )
    return tmp_path


def _edge(target, time, current, kind="scheduled_gate"):
    return {
        "kind": kind, "target": target, "time": time,
        "previous": not current, "current": current,
    }


def _complementary_pair(periods=10, duty=0.4, dead=0.5e-6):
    events = []
    for k in range(periods):
        start = k * _PERIOD
        events += [
            _edge("leg.high", start + dead, True),
            _edge("leg.high", start + duty * _PERIOD, False),
            _edge("leg.low", start + duty * _PERIOD + dead, True),
            _edge("leg.low", start + _PERIOD, False),
        ]
    return sorted(events, key=lambda event: event["time"])


def test_a_complementary_pair_reads_the_carrier_not_twice_it(tmp_path):
    """Regression: rising edges of leg.high and leg.low were pooled, so a
    20 kHz synchronous buck reported ~40 kHz."""

    scalar, by_gate = _switching_frequency_by_gate(
        _events_dir(tmp_path, _complementary_pair()), 0.0
    )
    assert scalar == pytest.approx(20e3, rel=1e-9)
    assert by_gate == pytest.approx({"leg.high": 20e3, "leg.low": 20e3}, rel=1e-9)


def test_a_forced_release_inside_the_window_does_not_shift_the_frequency(tmp_path):
    """Regression: a forced off-window ending mid-period adds one rising edge;
    the pooled (n - 1) / span estimate moved with it."""

    events = _complementary_pair()
    # leg.high held off from 300 us to 312 us: its scheduled turn-off at 320 us
    # and the 300.5 us turn-on vanish, the release at 312 us is a forced rise.
    events = [
        event for event in events
        if not (event["target"] == "leg.high" and 300e-6 <= event["time"] < 330e-6)
    ]
    events += [
        _edge("leg.high", 312e-6, True, kind="forced_gate"),
        _edge("leg.high", 320e-6, False),
    ]
    events.sort(key=lambda event: event["time"])
    scalar, by_gate = _switching_frequency_by_gate(_events_dir(tmp_path, events), 0.0)
    assert by_gate["leg.high"] == pytest.approx(20e3, rel=1e-9)
    assert scalar == pytest.approx(20e3, rel=1e-9)


def test_a_forced_glitch_mid_pulse_adds_a_rise_but_not_a_period(tmp_path):
    """A 0.1 us forced turn-off inside a high pulse adds one rising edge and
    removes none. Counting rises, (n - 1) / span, read 22.2 kHz for 20 kHz; the
    median spacing does not move."""

    events = _complementary_pair()
    events += [
        _edge("leg.high", 210e-6, False, kind="forced_gate"),
        _edge("leg.high", 210.1e-6, True),
    ]
    events.sort(key=lambda event: event["time"])
    scalar, by_gate = _switching_frequency_by_gate(_events_dir(tmp_path, events), 0.0)
    assert by_gate["leg.high"] == pytest.approx(20e3, rel=1e-9)
    assert scalar == pytest.approx(20e3, rel=1e-9)


def test_gates_at_different_frequencies_give_no_single_number(tmp_path):
    # The gate must stay silent when gates agree (tests above) and refuse a
    # scalar when they do not: 20 kHz and 100 kHz have no common "fsw".
    events = [_edge("a", k * _PERIOD, True) for k in range(10)]
    events += [_edge("b", k * _PERIOD / 5, True) for k in range(50)]
    scalar, by_gate = _switching_frequency_by_gate(_events_dir(tmp_path, events), 0.0)
    assert scalar is None
    assert by_gate == pytest.approx({"a": 20e3, "b": 100e3}, rel=1e-9)


def test_only_edges_inside_the_settled_window_count(tmp_path):
    events = [_edge("a", k * _PERIOD / 2, True) for k in range(10)]  # 40 kHz start-up
    events += [_edge("a", 500e-6 + k * _PERIOD, True) for k in range(10)]
    scalar, by_gate = _switching_frequency_by_gate(
        _events_dir(tmp_path, events), 500e-6
    )
    assert scalar == pytest.approx(20e3, rel=1e-9)
    assert by_gate == pytest.approx({"a": 20e3}, rel=1e-9)


def test_a_gate_with_a_single_rising_edge_has_no_frequency(tmp_path):
    scalar, by_gate = _switching_frequency_by_gate(
        _events_dir(tmp_path, [_edge("a", 1e-6, True)]), 0.0
    )
    assert scalar is None
    assert by_gate == {}


def test_campaign_reports_the_complementary_buck_carrier(tmp_path):
    model = EXAMPLES_DIR / "synchronous_buck_complementary.yaml"
    request = {
        "schema": REQUEST_SCHEMA,
        "label": "sync buck",
        "nominal": {"vout_v": 40.0, "recovery_band_percent": 2.0},
        "axes": [
            {
                "id": "anchor", "kind": "parameter_values", "parameter": "vin",
                "values": ["100 V"], "source": "assumption", "note": "anchor",
            }
        ],
        "criteria": [
            {
                "id": "vout", "signal": "v(out)", "field": "mean_last_half",
                "op": "abs_deviation_max", "reference": 40.0, "limit": 100.0,
                "unit": "V", "source": "assumption", "note": "placeholder",
            }
        ],
    }
    artifacts = run_campaign(model, _write(tmp_path, request), tmp_path / "out")
    metrics = json.loads(artifacts.summary.read_text())["runs"][0]["metrics"]
    assert metrics["fsw_measured_hz"] == pytest.approx(20e3, rel=1e-6)
    assert set(metrics["fsw_measured_hz_by_gate"]) == {"leg.high", "leg.low"}
    assert "fsw_measured_hz_by_gate" not in artifacts.report_html.read_text()
