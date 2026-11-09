# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Hook B: events declared in a request, materialized source-to-source."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.run.contracts import model_hashes
from pear_sim.spec.model import load_model
from pear_sim.spec.perturbation import (
    REQUEST_SCHEMA,
    PerturbationError,
    PerturbationSpec,
    apply_perturbation_to_document,
    parse_perturbation_request,
)


EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
BUCK = EXAMPLES_DIR / "buck.yaml"


def _request(**event):
    payload = {"schema": REQUEST_SCHEMA, "events": [{"id": "e1", **event}]}
    return parse_perturbation_request(payload)


def _materialize(tmp_path, events, name="perturbed.yaml", overrides=()):
    document = yaml.safe_load(BUCK.read_text())
    model = load_model(BUCK)
    spec = PerturbationSpec(
        label="probe", events=tuple(events), parameter_overrides_si=tuple(overrides)
    )
    perturbed = apply_perturbation_to_document(document, spec, model)
    path = tmp_path / name
    path.write_text(yaml.safe_dump(perturbed, sort_keys=False))
    return load_model(path)


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def test_absolute_time_and_phase_are_mutually_exclusive():
    with pytest.raises(PerturbationError, match="exactly one of 'at'"):
        _request(
            type="pulse", target="control:pwm", at="1 ms", phase=0.25,
            width="1 us", level="on",
        )
    with pytest.raises(PerturbationError, match="exactly one of 'at'"):
        _request(type="pulse", target="control:pwm", width="1 us", level="on")


def test_a_parameter_target_fails_closed_naming_the_kernel_limitation():
    """R/L/C are stamped once per compiled mode; only sources vary with time."""

    with pytest.raises(PerturbationError, match="not supported"):
        _request(type="step", target="parameter:load", at="1 ms", amplitude="1 A")


def test_injection_events_must_target_a_node_and_pulses_a_control():
    with pytest.raises(PerturbationError, match="must target a node"):
        _request(
            type="analog_transient", target="control:pwm", at="1 ms",
            amplitude="1 A", tau_rise="1 ns", tau_fall="100 ns",
        )
    with pytest.raises(PerturbationError, match="must target a control"):
        _request(type="pulse", target="node:out", at="1 ms", width="1 us", level="on")


def test_degenerate_transient_time_constants_are_rejected():
    with pytest.raises(PerturbationError, match="tau_fall must strictly exceed"):
        _request(
            type="analog_transient", target="node:out", at="1 ms",
            amplitude="1 A", tau_rise="100 ns", tau_fall="100 ns",
        )


# --------------------------------------------------------------------------
# materialization
# --------------------------------------------------------------------------


def test_analog_transient_materializes_a_current_source_and_needs_no_core_change(tmp_path):
    # 2 A decaying over ~50 us moves ~100 uC through a 100 uF output cap,
    # i.e. of order 1 V — enough to separate from the 75 mV nominal ripple.
    events = _request(
        type="analog_transient", target="node:out", at="1.5 ms",
        amplitude="2 A", tau_rise="100 ns", tau_fall="50 us",
    )
    model = _materialize(tmp_path, events)
    injected = next(c for c in model.components if c.id == "_pert_e1")
    assert injected.type == "current_source"
    assert injected.p == "out" and injected.n == "gnd"
    assert injected.waveform.kind == "double_exponential"
    # No forced windows: this path touches nothing in the solver.
    assert model.forced_gate_windows == ()

    baseline = HybridSimulator(load_model(BUCK)).simulate()
    perturbed = HybridSimulator(model).simulate()
    window = [
        (t, v) for t, v in zip(perturbed.time, perturbed.signals["v(out)"])
        if 1.5e-3 <= t <= 1.6e-3
    ]
    baseline_window = [
        v for t, v in zip(baseline.time, baseline.signals["v(out)"])
        if 1.5e-3 <= t <= 1.6e-3
    ]
    assert min(v for _, v in window) < min(baseline_window) - 0.5


def test_unknown_node_is_rejected_before_any_run(tmp_path):
    events = _request(
        type="analog_transient", target="node:nowhere", at="1 ms",
        amplitude="1 A", tau_rise="10 ns", tau_fall="1 us",
    )
    with pytest.raises(PerturbationError, match="does not exist"):
        _materialize(tmp_path, events)


def test_forced_gate_window_overrides_the_schedule(tmp_path):
    """A SET on the gate: held off for a window, then handed straight back."""

    events = _request(
        type="pulse", target="control:pwm", at="1.5 ms", width="20 us", level="off",
    )
    model = _materialize(tmp_path, events)
    (window,) = model.forced_gate_windows
    assert window.control_id == "pwm"
    assert window.level is False
    assert window.to_s == pytest.approx(window.from_s + 20e-6)

    result = HybridSimulator(model).simulate()
    baseline_result = HybridSimulator(load_model(BUCK)).simulate()

    def rises(events):
        return [
            event.time
            for event in events
            if event.target == "pwm" and event.current
            and 1.5e-3 <= event.time < 1.52e-3
        ]

    # The window suppresses the rise that the schedule would have produced.
    # No event is recorded for it precisely BECAUSE nothing transitions: the
    # gate was already low and the override keeps it there.
    assert rises(baseline_result.events), "baseline must rise inside this window"
    assert not rises(result.events)

    inductor = [
        v for t, v in zip(result.time, result.signals["i(l1)"])
        if 1.5e-3 <= t <= 1.53e-3
    ]
    baseline_inductor = [
        v for t, v in zip(baseline_result.time, baseline_result.signals["i(l1)"])
        if 1.5e-3 <= t <= 1.53e-3
    ]
    # Holding the switch off drains the inductor below anything the undisturbed
    # steady state reaches.
    assert min(inductor) < min(baseline_inductor) - 0.05


def test_the_gate_returns_to_its_natural_schedule_after_the_window(tmp_path):
    """After the window the gate takes its scheduled cadence back, unchanged.

    Deliberately asserted on the GATE, not on v(out): this buck is open-loop
    and its LC rings for milliseconds after any disturbance, so output recovery
    would be testing the plant, not the override mask.
    """

    events = _request(
        type="pulse", target="control:pwm", at="1.0 ms", width="30 us", level="off",
    )
    model = _materialize(tmp_path, events)
    result = HybridSimulator(model).simulate()

    period = model.controls["pwm"].period
    (window,) = model.forced_gate_windows
    rises_after = sorted(
        event.time
        for event in result.events
        if event.target == "pwm" and event.current and event.time > window.to_s
    )
    assert len(rises_after) >= 5
    gaps = [b - a for a, b in zip(rises_after, rises_after[1:])]
    assert all(gap == pytest.approx(period, rel=1e-9) for gap in gaps)
    # The first rise after the window lands on the undisturbed PWM grid.
    offset = rises_after[0] % period
    assert min(offset, period - offset) < 1e-9 * period + 1e-12


def test_a_perturbed_run_never_carries_the_baseline_physics_hash(tmp_path):
    events = _request(
        type="pulse", target="control:pwm", at="1.0 ms", width="10 us", level="off",
    )
    baseline = model_hashes(load_model(BUCK))
    perturbed = model_hashes(_materialize(tmp_path, events))
    assert perturbed["model_sha256"] != baseline["model_sha256"]
    assert perturbed["source_sha256"] != baseline["source_sha256"]


def test_parameter_overrides_are_written_into_the_materialized_source(tmp_path):
    events = _request(
        type="analog_transient", target="node:out", at="1 ms",
        amplitude="1 A", tau_rise="10 ns", tau_fall="1 us",
    )
    model = _materialize(tmp_path, events, overrides=(("load", 5.0),))
    assert model.parameters["load"].value == pytest.approx(5.0)


def test_phase_requires_an_explicit_anchor(tmp_path):
    """A phase says where in a cycle, never which cycle — so the cycle is declared.

    The first sensible-looking default lands the event in start-up; the last
    leaves no window to observe the response in. Both were wrong, so neither is
    guessed.
    """

    with pytest.raises(PerturbationError, match="phase_anchor"):
        _request(
            type="pulse", target="control:pwm", phase=0.25,
            width="1 us", level="off",
        )


def test_phase_lands_on_the_cycle_at_or_after_the_declared_anchor(tmp_path):
    model_baseline = load_model(BUCK)
    period = model_baseline.controls["pwm"].period
    anchor = 1.0e-3
    for phase in (0.0, 0.5, 0.75):
        events = _request(
            type="pulse", target="control:pwm", phase=phase,
            phase_anchor=f"{anchor} s", width="1 us", level="off",
        )
        model = _materialize(tmp_path, events, name=f"phase_{phase}.yaml")
        (window,) = model.forced_gate_windows
        assert window.from_s == pytest.approx(anchor + phase * period, abs=1e-12)
        # Room is left to observe what the event did.
        assert window.to_s < model.transient.stop


def test_an_anchor_outside_the_window_is_rejected(tmp_path):
    events = _request(
        type="pulse", target="control:pwm", phase=0.0,
        phase_anchor="9 ms", width="1 us", level="off",
    )
    with pytest.raises(PerturbationError, match="outside the analysis window"):
        _materialize(tmp_path, events)


def test_overlapping_windows_on_one_control_are_rejected(tmp_path):
    document = yaml.safe_load(BUCK.read_text())
    document["perturbations"] = {
        "forced_gate_windows": [
            {"control": "pwm", "from": 1.0e-3, "to": 1.1e-3, "level": "off"},
            {"control": "pwm", "from": 1.05e-3, "to": 1.2e-3, "level": "on"},
        ]
    }
    path = tmp_path / "overlap.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(ModelError, match="overlapping windows"):
        load_model(path)


def test_forced_window_on_an_undeclared_control_is_rejected(tmp_path):
    document = yaml.safe_load(BUCK.read_text())
    document["perturbations"] = {
        "forced_gate_windows": [
            {"control": "nope", "from": 1.0e-3, "to": 1.1e-3, "level": "off"}
        ]
    }
    path = tmp_path / "unknown.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(ModelError, match="not a declared control"):
        load_model(path)


def test_request_schema_is_pinned():
    with pytest.raises(PerturbationError, match="schema must be"):
        parse_perturbation_request({"schema": "something/else", "events": []})
    # A well-formed request round-trips its declaration for the report.
    events = _request(
        type="step", target="node:out", at="1 ms", amplitude="-500 mA"
    )
    payload = json.loads(json.dumps([event.as_dict() for event in events]))
    assert payload[0]["type"] == "step"
    assert payload[0]["target"] == "node:out"
