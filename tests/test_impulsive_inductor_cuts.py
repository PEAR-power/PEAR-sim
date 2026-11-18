# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""impulsive_events: conserve — inductor current cut by an opening switch.

Dual of tests/test_impulsive_capacitor_loops.py. Inductors shorted by a
closed switch keep their declared currents exactly until it opens, so the
oracle is flux conservation on paper: two inductors left in series end at
(L1*i1 + L2*i2)/(L1 + L2) and lose 1/2*L1*L2/(L1+L2)*(i1 - i2)^2; a lone
inductor ends at 0 and loses 1/2*L*i^2. Compared at 1e-12.

Independent method: the same circuit with a resistor R ACROSS the switch
(parallel conductance G = 1/R -> 0), integrated by the default engine with
no impulse. For an RL circuit the energy R dissipates does not depend on R,
so it must reach the impulsive loss for every R.

Every cut is also a warning in the bundle: in hardware it is an overvoltage
across the switch, and the opt-in must not silence that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from pear_sim.api.cli import main
from pear_sim.core import solver
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import StateDiscontinuityError
from pear_sim.spec.model import load_model

OPEN_S = 0.5e-3
BUDGET = 1e-12
L1, I1, L2, I2 = 1e-3, 6.0, 3e-3, 2.0
I_AFTER = (L1 * I1 + L2 * I2) / (L1 + L2)  # 3 A
LOSS = 0.5 * L1 * L2 / (L1 + L2) * (I1 - I2) ** 2  # 6 mJ


def _path(tmp_path: Path, components: list[dict], outputs: list[str], *,
          policy: str | None = "conserve", solver_name: str = "dopri5") -> Path:
    transient = {"start": "0 s", "stop": "2 ms", "output_step": "10 us",
                 "solver": solver_name, "rtol": 1e-10, "atol": 1e-12}
    if policy is not None:
        transient["impulsive_events"] = policy
    document = {
        "schema": "pear-sim/v0.1",
        "name": "cut",
        "circuit": {"ground": "gnd", "components": components},
        # On from 0, open at 0.5 ms, on again at 1 ms, open at 1.5 ms.
        "controls": {"pwm": {"type": "pwm", "frequency": "1 kHz", "duty": 0.5,
                             "phase": "0 s"}},
        "analysis": {"transient": transient},
        "outputs": outputs,
    }
    path = tmp_path / "cut.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _run(tmp_path: Path, components: list[dict], outputs: list[str], **kw):
    return HybridSimulator(load_model(_path(tmp_path, components, outputs, **kw))).simulate()


SWITCH = {"id": "s1", "type": "ideal_switch", "p": "a", "n": "gnd", "control": "pwm",
          "initial": "on"}
TWO_INDUCTORS = [
    # l1 drives current INTO node a, l2 takes it out; s1 shorts a while closed.
    {"id": "l1", "type": "inductor", "p": "gnd", "n": "a", "value": "1 mH",
     "initial_current": f"{I1} A"},
    SWITCH,
    {"id": "l2", "type": "inductor", "p": "a", "n": "gnd", "value": "3 mH",
     "initial_current": f"{I2} A"},
]


def _after(result, signal: str) -> np.ndarray:
    time = np.asarray(result.time)
    return np.asarray(result.signals[signal])[time > OPEN_S]


def test_two_inductors_share_their_flux(tmp_path: Path) -> None:
    result = _run(tmp_path, TWO_INDUCTORS, ["i(l1)", "i(l2)"])
    for signal in ("i(l1)", "i(l2)"):
        assert np.max(np.abs(_after(result, signal) - I_AFTER)) < BUDGET * I1
    (impulse,) = result.physics_audit["impulsive_commutations"]
    assert impulse["kind"] == "inductor_cut"
    assert impulse["cuts"] == [{"node": "a", "members": ["l1", "l2"]}]
    assert impulse["time_s"] == pytest.approx(OPEN_S, abs=1e-15)
    assert abs(impulse["current_mismatch_a"][0] - (I1 - I2)) < BUDGET * I1
    assert abs(impulse["commutation_loss_j"] - LOSS) < BUDGET * LOSS
    flux = impulse["transferred_flux_wb"]
    assert flux["l1"] == pytest.approx(-flux["l2"], rel=1e-15)  # conserved by construction
    audit = result.physics_audit["full"]
    assert audit["balance_status"] == "within_numerical_tolerance"
    assert abs(audit["commutation_loss_j"] - LOSS) < BUDGET * LOSS


def test_a_lone_inductor_is_cut_to_zero(tmp_path: Path) -> None:
    inductance, current = 2e-3, 5.0
    result = _run(tmp_path, [
        {"id": "l1", "type": "inductor", "p": "gnd", "n": "a", "value": "2 mH",
         "initial_current": f"{current} A"},
        SWITCH,
    ], ["i(l1)"])
    after = _after(result, "i(l1)")
    time = np.asarray(result.time)[np.asarray(result.time) > OPEN_S]
    # Zero while s1 is open; when it closes again at 1 ms the inductor stays at zero.
    assert np.max(np.abs(after[time < 1e-3])) < BUDGET
    (impulse,) = result.physics_audit["impulsive_commutations"]
    assert abs(impulse["commutation_loss_j"] - 0.5 * inductance * current**2) < BUDGET * 0.025
    assert result.physics_audit["full"]["balance_status"] == "within_numerical_tolerance"


def test_a_star_of_three_inductors(tmp_path: Path) -> None:
    """Node a keeps only three inductors to ground when s1 opens; KCL then
    asks the three currents into a to sum to zero."""

    values = ((1e-3, 4.0), (2e-3, -1.0), (4e-3, 0.5))  # (L, current INTO a)
    components = [SWITCH] + [
        {"id": f"l{k}", "type": "inductor", "p": "gnd", "n": "a", "value": f"{L * 1e3} mH",
         "initial_current": f"{i} A"}
        for k, (L, i) in enumerate(values, start=1)
    ]
    result = _run(tmp_path, components, ["i(l1)", "i(l2)", "i(l3)"])
    # Flux impulse lambda at a: i_k += lambda / L_k with sum(i_k) = 0 after.
    lam = -sum(i for _L, i in values) / sum(1.0 / L for L, _i in values)
    expected = [i + lam / L for L, i in values]
    loss = sum(0.5 * L * (i * i - e * e) for (L, i), e in zip(values, expected))
    for k, value in enumerate(expected, start=1):
        assert np.max(np.abs(_after(result, f"i(l{k})")[:10] - value)) < BUDGET * 4.0
    (impulse,) = result.physics_audit["impulsive_commutations"]
    assert abs(impulse["commutation_loss_j"] - loss) < BUDGET * loss


@pytest.mark.parametrize("resistance", [1e3, 1e4, 1e5])
def test_the_jump_is_the_limit_of_a_vanishing_parallel_conductance(
    tmp_path: Path, resistance: float
) -> None:
    components = TWO_INDUCTORS + [
        {"id": "rp", "type": "resistor", "p": "a", "n": "gnd", "value": f"{resistance} ohm"}
    ]
    result = _run(tmp_path, components, ["i(l1)", "i(l2)"], policy=None, solver_name="radau")
    dissipated = result.physics_audit["full"]["component_absorbed_energy_j"]["rp"]
    assert abs(dissipated - LOSS) < 1e-6 * LOSS, (resistance, dissipated)
    assert abs(result.signals["i(l1)"][-1] - I_AFTER) < 1e-6 * I_AFTER
    assert "impulsive_commutations" not in result.physics_audit


def test_a_jump_with_the_wrong_inductances_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flux moves only through the cut node, so an "average of the currents"
    cannot be expressed; a flux impulse solved as if the inductors were equal
    leaves KCL broken, and the star check refuses it."""

    def as_if_equal(inductance, incidence, residuals):
        equal = np.full_like(inductance, float(np.mean(inductance)))
        return np.linalg.solve((incidence / equal) @ incidence.T, -residuals)

    monkeypatch.setattr(solver, "impulsive_cut_fluxes", as_if_equal)
    with pytest.raises((StateDiscontinuityError, solver.SimulationError)):
        _run(tmp_path, TWO_INDUCTORS, ["i(l1)", "i(l2)"])


def test_an_audit_without_the_commutation_loss_does_not_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(solver, "commutation_loss", lambda storage, sources: 0.0)
    audit = _run(tmp_path, TWO_INDUCTORS, ["i(l1)", "i(l2)"]).physics_audit["full"]
    assert audit["balance_status"] == "outside_numerical_tolerance"
    assert abs(abs(audit["balance_residual_j"]) - LOSS) < 1e-9 * LOSS


def test_the_default_still_refuses(tmp_path: Path) -> None:
    with pytest.raises(StateDiscontinuityError):
        _run(tmp_path, TWO_INDUCTORS, ["i(l1)"], policy=None)


def test_every_cut_is_a_warning_in_the_bundle(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    lone = [
        {"id": "l1", "type": "inductor", "p": "in", "n": "a", "value": "1 mH",
         "initial_current": "1 A"},
        {"id": "v1", "type": "voltage_source", "p": "in", "n": "gnd", "value": "10 V"},
        SWITCH,
    ]
    out = tmp_path / "out"
    status = main(["simulate", str(_path(tmp_path, lone, ["i(l1)"])), "--out", str(out), "--json"])
    assert status == 0, capsys.readouterr().err
    warnings = json.loads((out / "warnings.json").read_text(encoding="utf-8"))
    cuts = [w for w in warnings if w["code"] == "IMPULSIVE_INDUCTOR_CUT"]
    # The source re-charges l1 each time s1 closes; it is cut at 0.5 and 1.5 ms.
    assert [round(w["time_s"] * 1e3, 9) for w in cuts] == [0.5, 1.5]
    assert all("overvoltage" in w["message"] for w in cuts)
    assert "6 A" in cuts[0]["message"]  # 1 A + 10 V / 1 mH * 0.5 ms
