# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""impulsive_events: conserve — a capacitor loop closed by a switch.

Oracles are charge conservation on paper: capacitors with no resistor keep
their declared voltages exactly until the switch closes, so the expected
voltage after the jump, sum(C*V)/sum(C), and the dissipated energy
1/2*sum(C*V^2) - 1/2*sum(C)*V_after^2 (plus what a source delivers) are
exact numbers, compared at 1e-12.

A second, independent method checks that the jump is the physical limit and
not an invention: the same circuit with an on-resistance R in the switch,
integrated by the ordinary engine (default reject mode, no impulse at all),
dissipates in R an energy that does NOT depend on R for an RC circuit — it
must reach the impulsive loss for R = 1, 0.1 and 0.01 ohm alike.

Two mutations must ring: an "average of the voltages" jump instead of the
charge-conserving one (indistinguishable with equal capacitors, hence
C1 != C2), and an audit without the commutation_loss_j term.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from pear_sim.analysis.steady_state import SteadyStateError, solve_periodic_steady_state
from pear_sim.core import solver
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError, StateDiscontinuityError, TopologyError
from pear_sim.run.contracts import simulation_request_dict
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]
CLOSE_S = 0.5e-3
BUDGET = 1e-12


def _model(tmp_path: Path, components: list[dict], outputs: list[str], *,
           policy: str | None = "conserve", solver_name: str = "dopri5", name: str = "case"):
    transient = {"start": "0 s", "stop": "2 ms", "output_step": "10 us",
                 "solver": solver_name, "rtol": 1e-10, "atol": 1e-12}
    if policy is not None:
        transient["impulsive_events"] = policy
    document = {
        "schema": "pear-sim/v0.1",
        "name": name,
        "circuit": {"ground": "gnd", "components": components},
        "controls": {"pwm": {"type": "pwm", "frequency": "1 kHz", "duty": 0.5,
                             "phase": "0.5 ms"}},
        "analysis": {"transient": transient},
        "outputs": outputs,
    }
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return load_model(path)


def _switch(**extra) -> dict:
    return {"id": "s1", "type": "ideal_switch", "p": "a", "n": "b", "control": "pwm",
            "initial": "off", **extra}


C1, V1, C2, V2 = 1e-6, 10.0, 3e-6, 2.0
TWO_CAPS = [
    {"id": "c1", "type": "capacitor", "p": "a", "n": "gnd", "value": "1 uF",
     "initial_voltage": f"{V1} V"},
    _switch(),
    {"id": "c2", "type": "capacitor", "p": "b", "n": "gnd", "value": "3 uF",
     "initial_voltage": f"{V2} V"},
]
V_AFTER = (C1 * V1 + C2 * V2) / (C1 + C2)  # 4 V
LOSS = 0.5 * C1 * C2 / (C1 + C2) * (V1 - V2) ** 2  # 24 uJ


def _after(result, signal: str) -> np.ndarray:
    time = np.asarray(result.time)
    return np.asarray(result.signals[signal])[time >= CLOSE_S]


def test_two_capacitors_share_their_charge(tmp_path: Path) -> None:
    result = HybridSimulator(_model(tmp_path, TWO_CAPS, ["v(a)", "v(b)"])).simulate()
    for signal in ("v(a)", "v(b)"):
        assert np.max(np.abs(_after(result, signal) - V_AFTER)) < BUDGET * V1
    (impulse,) = result.physics_audit["impulsive_commutations"]
    assert impulse["time_s"] == pytest.approx(CLOSE_S, abs=1e-15)
    assert impulse["event"] == "gate 'pwm' off->on"
    assert impulse["loops"] == [["c1", "c2", "s1"]]
    assert abs(impulse["commutation_loss_j"] - LOSS) < BUDGET * LOSS
    assert abs(impulse["voltage_mismatch_v"][0] - (V1 - V2)) < BUDGET * V1
    audit = result.physics_audit["full"]
    assert audit["balance_status"] == "within_numerical_tolerance"
    assert abs(audit["commutation_loss_j"] - LOSS) < BUDGET * LOSS
    assert result.stats["impulsive_commutations"] == 1


def test_a_capacitor_closed_onto_a_dc_source(tmp_path: Path) -> None:
    c, v0, vs = 1e-6, 2.0, 10.0
    result = HybridSimulator(_model(tmp_path, [
        {"id": "v1", "type": "voltage_source", "p": "a", "n": "gnd", "value": f"{vs} V"},
        _switch(),
        {"id": "c1", "type": "capacitor", "p": "b", "n": "gnd", "value": "1 uF",
         "initial_voltage": f"{v0} V"},
    ], ["v(b)"])).simulate()
    assert np.max(np.abs(_after(result, "v(b)") - vs)) < BUDGET * vs
    (impulse,) = result.physics_audit["impulsive_commutations"]
    # The source delivers V*Q = 10 V * 8 uC = 80 uJ; storage rises 48 uJ; 32 uJ is lost.
    assert abs(impulse["commutation_loss_j"] - 0.5 * c * (vs - v0) ** 2) < BUDGET * 1e-4
    assert abs(impulse["source_absorbed_energy_j"]["v1"] + vs * c * (vs - v0)) < BUDGET * 1e-4
    assert result.physics_audit["full"]["balance_status"] == "within_numerical_tolerance"


def test_a_chain_of_three_capacitors_closed_at_once(tmp_path: Path) -> None:
    caps = ((1e-6, 9.0), (2e-6, 3.0), (3e-6, 0.0))
    result = HybridSimulator(_model(tmp_path, [
        {"id": "c1", "type": "capacitor", "p": "a", "n": "gnd", "value": "1 uF",
         "initial_voltage": "9 V"},
        _switch(),
        {"id": "c2", "type": "capacitor", "p": "b", "n": "gnd", "value": "2 uF",
         "initial_voltage": "3 V"},
        {"id": "s2", "type": "ideal_switch", "p": "b", "n": "c", "control": "pwm",
         "initial": "off"},
        {"id": "c3", "type": "capacitor", "p": "c", "n": "gnd", "value": "3 uF",
         "initial_voltage": "0 V"},
    ], ["v(a)", "v(b)", "v(c)"])).simulate()
    total = sum(c for c, _v in caps)
    v_after = sum(c * v for c, v in caps) / total  # 2.5 V
    loss = 0.5 * sum(c * v * v for c, v in caps) - 0.5 * total * v_after**2  # 30.75 uJ
    for signal in ("v(a)", "v(b)", "v(c)"):
        assert np.max(np.abs(_after(result, signal) - v_after)) < BUDGET * 9.0
    impulses = result.physics_audit["impulsive_commutations"]
    assert len(impulses) == 1 and len(impulses[0]["loops"]) == 2
    assert abs(impulses[0]["commutation_loss_j"] - loss) < BUDGET * loss
    assert result.physics_audit["full"]["balance_status"] == "within_numerical_tolerance"


@pytest.mark.parametrize("resistance", [1.0, 0.1, 0.01])
def test_the_jump_is_the_limit_of_a_vanishing_loop_resistance(
    tmp_path: Path, resistance: float
) -> None:
    """Ordinary engine, no impulse: the switch's R dissipates the same energy
    for every R, and that energy is the impulsive loss."""

    components = [dict(item) for item in TWO_CAPS]
    components[1] = _switch(on_resistance=f"{resistance} ohm")
    result = HybridSimulator(
        _model(tmp_path, components, ["v(a)", "v(b)"], policy=None, solver_name="radau")
    ).simulate()
    dissipated = result.physics_audit["full"]["component_absorbed_energy_j"]["s1"]
    assert abs(dissipated - LOSS) < 1e-6 * LOSS, (resistance, dissipated)
    assert abs(result.signals["v(a)"][-1] - V_AFTER) < 1e-6 * V_AFTER
    assert "impulsive_commutations" not in result.physics_audit


def test_a_jump_weighted_by_the_wrong_capacitances_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The "average of the voltages" mutation cannot even be expressed here:
    the jump moves charge only AROUND loops, so charge is conserved by
    construction (c1 gives exactly what c2 gets). What can go wrong is the
    loop circulation itself; solving it as if the capacitors were equal (the
    algebra that yields the plain average) leaves KVL broken after the jump,
    and the kernel's own re-check refuses instead of integrating it."""

    def as_if_equal(capacitance, incidence, residuals):
        equal = np.full_like(capacitance, float(np.mean(capacitance)))
        return np.linalg.solve((incidence / equal) @ incidence.T, -residuals)

    monkeypatch.setattr(solver, "impulsive_loop_charges", as_if_equal)
    with pytest.raises(solver.SimulationError, match="did not settle"):
        HybridSimulator(_model(tmp_path, TWO_CAPS, ["v(a)", "v(b)"])).simulate()


def test_charge_is_conserved_by_construction(tmp_path: Path) -> None:
    result = HybridSimulator(_model(tmp_path, TWO_CAPS, ["v(a)", "v(b)"])).simulate()
    (impulse,) = result.physics_audit["impulsive_commutations"]
    moved = impulse["transferred_charge_c"]
    assert moved["c1"] == -moved["c2"]
    assert abs(moved["c2"] - C2 * (V_AFTER - V2)) < BUDGET * C1 * V1


def test_an_audit_without_the_commutation_loss_does_not_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(solver, "commutation_loss", lambda storage, sources: 0.0)
    result = HybridSimulator(_model(tmp_path, TWO_CAPS, ["v(a)", "v(b)"])).simulate()
    audit = result.physics_audit["full"]
    assert audit["balance_status"] == "outside_numerical_tolerance"
    assert abs(abs(audit["balance_residual_j"]) - LOSS) < 1e-9 * LOSS


def test_the_default_still_refuses(tmp_path: Path) -> None:
    with pytest.raises(StateDiscontinuityError):
        HybridSimulator(_model(tmp_path, TWO_CAPS, ["v(a)"], policy=None)).simulate()


def test_a_waveform_source_in_the_loop_is_refused(tmp_path: Path) -> None:
    model = _model(tmp_path, [
        {"id": "v1", "type": "voltage_source", "p": "a", "n": "gnd",
         "waveform": {"type": "sine", "amplitude": "10 V", "frequency": "50 Hz"}},
        _switch(),
        {"id": "c1", "type": "capacitor", "p": "b", "n": "gnd", "value": "1 uF"},
    ], ["v(b)"])
    with pytest.raises(TopologyError, match="waveform source"):
        HybridSimulator(model).simulate()


def test_a_diode_in_the_loop_is_refused(tmp_path: Path) -> None:
    model = _model(tmp_path, [
        {"id": "c1", "type": "capacitor", "p": "a", "n": "gnd", "value": "1 uF",
         "initial_voltage": "10 V"},
        {"id": "d1", "type": "diode", "p": "a", "n": "b", "initial": "off"},
        {"id": "c2", "type": "capacitor", "p": "b", "n": "gnd", "value": "3 uF",
         "initial_voltage": "2 V"},
    ], ["v(a)"])
    with pytest.raises(TopologyError, match="d1"):
        HybridSimulator(model).simulate()


def test_an_unknown_policy_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="impulsive_events"):
        _model(tmp_path, TWO_CAPS, ["v(a)"], policy="absorb")


def test_the_default_leaves_existing_contracts_untouched() -> None:
    model = load_model(ROOT / "examples" / "buck.yaml")
    assert model.transient.impulsive_events == "reject"
    assert "impulsive_events" not in simulation_request_dict(model)
    result = HybridSimulator(model).simulate()
    assert "impulsive_commutations" not in result.physics_audit
    assert "commutation_loss_j" not in result.physics_audit["full"]
    assert "impulsive_commutations" not in result.stats


def test_steady_state_shooting_refuses_the_opt_in(tmp_path: Path) -> None:
    with pytest.raises(SteadyStateError, match="impulsive_events"):
        solve_periodic_steady_state(_model(tmp_path, TWO_CAPS, ["v(a)"]))
