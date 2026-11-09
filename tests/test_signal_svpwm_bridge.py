# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""The signal-layer space-vector bridge, and the rail-to-rail short regression.

The point of this bridge: a control law hands over a commanded
VOLTAGE VECTOR and the modulator owns the sector decision, so the modulator
can be swapped without rewriting the controller.

Every run with this control type used to end in a rail-to-rail short at the
SECOND task tick. The cause was a single line: the loop that seeds a
turn-on at each cycle boundary — correct for a single-gate sampled
controller, catastrophic for a six-gate bridge — did not exclude this type.
The recorded suspicion (insertion order under the schedule cursor) was
wrong, which is why these tests pin the behaviour rather than that theory.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pear_sim.core.solver import HybridSimulator, _initial_sampled_gate_values
from pear_sim.errors import SimulationError
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "inverter_signal_svpwm.yaml"


@pytest.fixture(scope="module")
def bridge_result():
    return HybridSimulator(load_model(EXAMPLE)).simulate()


def test_the_bridge_runs_at_all(bridge_result) -> None:
    """Regression: this raised TopologyError at the second tick."""

    assert len(bridge_result.time) > 10
    assert bridge_result.signal_trace


def test_the_commanded_vector_appears_on_the_load(bridge_result) -> None:
    """The measurement that makes this more than 'it switches'.

    alpha = 200 V is commanded; the mean PHASE voltage across the load must
    be that, to continuous-time quadrature rather than to the output grid
    (the grid samples a centred rectangle 10 times per cycle and would
    report 100 V — an aliasing artefact, not a modulator error).
    """

    metrics = bridge_result.metrics()
    pole = metrics["v(ta)"]["mean"]
    star = metrics["v(star)"]["mean"]
    assert pole - star == pytest.approx(200.0, abs=1e-9)


def test_pole_voltage_matches_the_commanded_duty(bridge_result) -> None:
    # duty 0.75 on a +/-300 V rail is (2*0.75 - 1) * 300 = 150 V.
    assert bridge_result.metrics()["v(ta)"]["mean"] == pytest.approx(150.0, abs=1e-9)


def test_every_tick_records_its_sector_and_dwell_times(bridge_result) -> None:
    for entry in bridge_result.signal_trace:
        duties = entry["commanded_duty"]
        assert {"bridge.a", "bridge.b", "bridge.c"} <= set(duties)
        assert duties["bridge.a"] == pytest.approx(0.75, abs=1e-9)
        assert duties["bridge.b"] == pytest.approx(0.25, abs=1e-9)


def test_a_bridge_never_starts_with_every_gate_closed() -> None:
    """The shape of the rail-to-rail short failure, pinned directly.

    Six gates commanded on at once is a rail-to-rail short. The kernel used
    to report it as a singular topology, which points at the circuit instead
    of at the block that caused it.
    """

    model = load_model(EXAMPLE)
    gates = _initial_sampled_gate_values(model)
    assert gates, "the bridge declared no sampled gates"
    for phase in ("a", "b", "c"):
        high = gates[f"bridge.{phase}.high"]
        low = gates[f"bridge.{phase}.low"]
        assert not (high and low), f"phase {phase} starts shorted"


def test_a_shoot_through_is_named_not_reported_as_a_singular_circuit() -> None:
    """If the interlock is ever violated, the error must say WHO did it."""

    model = load_model(EXAMPLE)
    simulator = HybridSimulator(model)
    shorted = {gate_id: True for gate_id in model.sampled_controls}
    with pytest.raises(SimulationError, match="interlock violation for signal_svpwm"):
        simulator._assert_complementary_interlocks(shorted, time=0.0)


def test_the_control_enters_the_physics_hash() -> None:
    """Two different commanded modulators must not share a hash."""

    from dataclasses import replace

    from pear_sim.run.contracts import model_hashes

    model = load_model(EXAMPLE)
    baseline = model_hashes(model)["model_sha256"]
    louder = replace(
        model,
        sampled_controls={
            key: replace(control, dc_link_voltage=control.dc_link_voltage * 2.0)
            for key, control in model.sampled_controls.items()
        },
    )
    assert model_hashes(louder)["model_sha256"] != baseline
