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
# fsw_measured_hz — per gate, command edges only, fail closed
# --------------------------------------------------------------------------

COMPLEMENTARY = EXAMPLES_DIR / "synchronous_buck_complementary.yaml"
_DECLARED_FSW_HZ = 20_000.0  # synchronous_buck_complementary.yaml: fsw: 20 kHz


@pytest.fixture(scope="module")
def complementary_forced_campaign(tmp_path_factory):
    """A complementary pair with a forced-off window inside the settled half.

    Regression for two biases of the old estimator, both measured on this
    campaign before the fix: it pooled the rises of ``leg.high`` and
    ``leg.low`` (two per period) and counted the release rise at a window's
    end as a switching edge — 45.5 kHz reported for a 20 kHz converter.
    """

    tmp_path = tmp_path_factory.mktemp("fsw")
    request = _request(
        nominal={"vout_v": 40.0, "recovery_band_percent": 2.0},
        events=[
            {
                "id": "set1", "type": "pulse", "target": "control:leg.high",
                "at": "305 us", "width": "1 us", "level": "off",
            }
        ],
        axes=[
            {
                "id": "width", "kind": "event_field", "event": "set1",
                "field": "width", "values": ["1 us", "60 us", "120 us"],
                "source": "assumption", "note": "window inside the settled half",
            }
        ],
        criteria=[
            {
                "id": "alive", "signal": "v(out)", "field": "min_last_half",
                "op": "min", "limit": 0.0, "unit": "V",
                "source": "assumption", "note": "placeholder",
            }
        ],
    )
    artifacts = run_campaign(COMPLEMENTARY, _write(tmp_path, request), tmp_path / "out")
    return json.loads(artifacts.summary.read_text())


def test_fsw_is_the_switching_frequency_not_the_pooled_rise_rate(
    complementary_forced_campaign,
):
    for run in complementary_forced_campaign["runs"]:
        assert run["execution_status"] == "pass", run.get("error")
        metrics = run["metrics"]
        # An unmodulated carrier: every clean rise-to-rise interval is exactly
        # one period, so the estimate is exact up to float round-off.
        assert metrics["fsw_measured_hz"] == pytest.approx(_DECLARED_FSW_HZ, rel=1e-9)
        for gate in ("leg.high", "leg.low"):
            assert metrics[f"{gate}::fsw_measured_hz"] == pytest.approx(
                _DECLARED_FSW_HZ, rel=1e-9
            )


def test_fsw_bounds_contain_the_estimate(complementary_forced_campaign):
    for run in complementary_forced_campaign["runs"]:
        metrics = run["metrics"]
        assert (
            metrics["fsw_measured_lo_hz"]
            <= metrics["fsw_measured_hz"]
            <= metrics["fsw_measured_hi_hz"]
        )


def _events_run(tmp_path, events):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events)
    )
    return run_dir


def _rise(time, target, kind="scheduled_gate"):
    return {"time": time, "kind": kind, "target": target, "previous": False, "current": True}


def test_a_window_release_rise_is_not_a_switching_edge(tmp_path):
    from pear_sim.evidence.perturbation_campaign import _measured_switching_frequency
    from pear_sim.spec.perturbation import ForcedGateWindow

    period = 1e-5
    rises = [_rise(k * period, "g") for k in range(10)]
    # A forced-off window [3.2, 5.4] periods masks the command rises at 4 and 5
    # and releases at 5.4 — logged as scheduled_gate with no command behind it.
    rises = [r for r in rises if not 3.2 * period <= r["time"] <= 5.4 * period]
    rises.append(_rise(5.4 * period, "g"))
    rises.sort(key=lambda r: r["time"])
    window = ForcedGateWindow(control_id="g", from_s=3.2 * period, to_s=5.4 * period, level=False)

    result = _measured_switching_frequency(_events_run(tmp_path, rises), 0.0, (window,))

    assert result.value_hz == pytest.approx(1.0 / period, rel=1e-9)
    assert result.reason is None


def test_gates_at_different_frequencies_give_no_single_fsw(tmp_path):
    from pear_sim.evidence.perturbation_campaign import _measured_switching_frequency

    events = [_rise(k * 1e-5, "slow") for k in range(20)]
    events += [_rise(k * 0.5e-5, "fast") for k in range(40)]
    events.sort(key=lambda r: r["time"])

    result = _measured_switching_frequency(_events_run(tmp_path, events), 0.0, ())

    assert result.value_hz is None
    assert "disagree" in result.reason
    assert result.per_gate_hz["slow"] == pytest.approx(1e5, rel=1e-9)
    assert result.per_gate_hz["fast"] == pytest.approx(2e5, rel=1e-9)


def test_no_measurable_gate_gives_none_with_a_reason(tmp_path):
    from pear_sim.evidence.perturbation_campaign import _measured_switching_frequency

    result = _measured_switching_frequency(
        _events_run(tmp_path, [_rise(1e-5, "g")]), 0.0, ()
    )

    assert result.value_hz is None
    assert result.reason


def test_a_median_outside_the_shared_bounds_is_withheld(tmp_path):
    from pear_sim.evidence.perturbation_campaign import _measured_switching_frequency

    # "long": 100 clean intervals over 1 s -> 100 Hz, bounds (99, 101).
    # "short": one clean 5 ms interval -> 200 Hz, bounds (0, 400).
    # The bounds intersect in (99, 101); the median, 150 Hz, lies outside it.
    events = [_rise(k * 0.01, "long") for k in range(101)]
    events += [_rise(0.5, "short"), _rise(0.505, "short")]
    events.sort(key=lambda r: r["time"])

    result = _measured_switching_frequency(_events_run(tmp_path, events), 0.0, ())

    assert result.value_hz is None
    assert "outside" in result.reason
    assert result.per_gate_hz["long"] == pytest.approx(100.0, rel=1e-9)
    assert result.per_gate_hz["short"] == pytest.approx(200.0, rel=1e-9)
