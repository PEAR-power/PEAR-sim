# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Replayed active-set search plans must not change a single result.

The diode active-set search keeps, per switch configuration, the list of
candidates it visited and what each compiled to (see Circuit._mode_plan).
A repeated configuration replays that list and re-runs only the
state-dependent feasibility check. These tests hold the replayed run to
exact equality with a run that rebuilds every search from scratch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pear_sim.core import circuit as circuit_module
from pear_sim.core.circuit import Circuit, Topology
from pear_sim.core.solver import HybridSimulator, _initial_gate_values
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]


def _run(path: Path, monkeypatch: pytest.MonkeyPatch, *, plans: bool):
    with monkeypatch.context() as patch:
        if not plans:
            # A full plan store is cleared before every insert, so each call
            # searches from scratch, exactly as before plans existed.
            patch.setattr(circuit_module, "MAX_MODE_PLANS", 0)
        simulator = HybridSimulator(load_model(path))
        result = simulator.simulate()
    return simulator, result


@pytest.mark.parametrize(
    "example", ["npc_three_phase.yaml", "npc5_three_phase.yaml", "buck_dcm.yaml"]
)
def test_replayed_plans_reproduce_a_fresh_search_exactly(
    example: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = ROOT / "examples" / example
    fresh_sim, fresh = _run(path, monkeypatch, plans=False)
    plan_sim, planned = _run(path, monkeypatch, plans=True)

    assert np.array_equal(fresh.time, planned.time)
    assert fresh.signals.keys() == planned.signals.keys()
    for name in fresh.signals:
        assert np.array_equal(fresh.signals[name], planned.signals[name]), name
    assert fresh.mode_intervals == planned.mode_intervals
    assert fresh.state_extrema == planned.state_extrema
    # Same candidates examined, same compiled modes: the search effort is
    # identical, only the bookkeeping around it is skipped.
    assert (
        fresh_sim.circuit.active_set_search_stats()
        == plan_sim.circuit.active_set_search_stats()
    )
    assert (
        fresh_sim.circuit.compiled_topology_stats()
        == plan_sim.circuit.compiled_topology_stats()
    )
    # And the store was actually used, otherwise this proves nothing.
    assert len(plan_sim.circuit._plans) < plan_sim.circuit.active_set_search_stats()["calls"]


def test_plan_keys_match_the_public_topology_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # The search builds Topology keys itself (gates sorted once per search).
    # A key in another order would never hit the compile cache: results stay
    # right but every mode recompiles. Hold it to Circuit.topology()'s form.
    seen: list[Topology] = []
    original = Circuit._compile_topology

    def spy(self, topology, **options):
        seen.append(topology)
        return original(self, topology, **options)

    monkeypatch.setattr(Circuit, "_compile_topology", spy)
    HybridSimulator(load_model(ROOT / "examples" / "npc_three_phase.yaml")).simulate()
    assert seen
    probe = Circuit(load_model(ROOT / "examples" / "npc_three_phase.yaml"))
    for topology in seen:
        assert topology == probe.topology(
            topology.gate_dict(),
            topology.diode_dict(),
            topology.constrained_inductors,
            topology.saturated_inductors,
        )


def test_a_caller_editing_the_returned_diodes_cannot_corrupt_a_plan() -> None:
    model = load_model(ROOT / "examples" / "npc_three_phase.yaml")
    circuit = Circuit(model)
    gates = _initial_gate_values(model.controls, model.transient.start)
    state = circuit.initial_state.copy()
    diodes, _ = circuit.stable_mode(gates, dict(circuit.initial_diodes), state, 1e-9)
    expected = dict(diodes)
    for key in diodes:
        diodes[key] = not diodes[key]
    again, _ = circuit.stable_mode(gates, dict(circuit.initial_diodes), state, 1e-9)
    assert len(circuit._plans) == 1  # the second call was a replay
    assert again == expected


def test_new_inductances_drop_the_plans() -> None:
    circuit = Circuit(load_model(ROOT / "examples" / "transformer_rl.yaml"))
    group = circuit.coupling_groups[0]
    circuit._plans[("marker",)] = object()  # type: ignore[assignment]
    circuit.set_group_inductance_matrix(group, circuit.coupling_matrix(group))
    assert not circuit._plans


def test_an_error_inside_a_search_does_not_leave_a_truncated_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The first search dies on an unexpected error; the plan it left behind
    # must not be replayed as if it were complete.
    model = load_model(ROOT / "examples" / "npc_three_phase.yaml")
    gates = _initial_gate_values(model.controls, model.transient.start)
    reference = Circuit(model)
    state = reference.initial_state.copy()
    expected, _ = reference.stable_mode(gates, dict(reference.initial_diodes), state, 1e-9)

    circuit = Circuit(model)
    original = Circuit._compile_topology
    calls = {"n": 0}

    def flaky(self, topology, **options):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient failure")
        return original(self, topology, **options)

    monkeypatch.setattr(Circuit, "_compile_topology", flaky)
    with pytest.raises(RuntimeError, match="transient failure"):
        circuit.stable_mode(gates, dict(circuit.initial_diodes), state, 1e-9)
    found, _ = circuit.stable_mode(gates, dict(circuit.initial_diodes), state, 1e-9)
    assert found == expected
