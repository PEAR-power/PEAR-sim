"""Constant diode guards must not livelock on float dust.

An open diode antiparallel to a closed ideal switch has a guard with NO state
dependence and NO source coupling — a pure constant along every trajectory of
the mode. When the declared parameters are exactly binary-representable the
MNA elimination cancels that constant to 0.0 and the identically-zero special
case holds it as a non-crossing. Shave the special case off by any finite
perturbation (vin = 400.4 is not representable; on other platforms the same
dust appears at the NOMINAL 400 through a different BLAS) and the constant
becomes ~1e-13 of elimination dust. Reading that dust as a zero crossing
toggles a diode the active-set manager immediately undoes: a livelock at the
first segment, parameter-knife-edge, platform-dependent.

Two kernel rules under test:

1. STRUCTURAL: a blocking diode whose terminals the mode's zero-voltage
   branches (closed switches, conducting diodes, frozen plain inductors)
   short together has an identically-zero guard BY TOPOLOGY. The
   classification is exact graph reasoning, immune to whatever dust each
   platform's arithmetic leaves in the solved affine row — ubuntu left it in
   the constant, windows in the state coefficients too, macOS cancelled all
   but one diode's constant. Any test on the solved numbers is the knife.
2. NUMERIC FALLBACK: a guard whose state AND source coefficients are all
   zero is constant along the flow — it cannot cross inside a segment, so it
   is held at a directed non-crossing value exactly like the
   invariant-subspace constants at a floating star.

The livelock guard itself stays: genuine chatter moves the state, so its
guards have state coefficients and are untouched by both rules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pear_sim.analysis.steady_state import solve_periodic_steady_state
from pear_sim.solver import HybridSimulator
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]
DAB = ROOT / "examples" / "dab_phase_shift.yaml"


@pytest.mark.parametrize("vin", ["399.6 V", "400 V", "400.4 V"])
def test_the_dab_transient_runs_on_both_sides_of_the_knife(vin: str) -> None:
    """Livelock repro: vin = 400.4 livelocked at the first segment while 399.6 and
    the exactly-representable 400 ran clean. All three must run clean, and the
    antiparallel diodes must stay idle at this operating point either way."""

    run = HybridSimulator(load_model(DAB, overrides={"vin": vin})).simulate()
    assert run.stats["natural_events"] == 0


def test_bridge_parallel_blocking_diodes_are_structurally_shorted() -> None:
    """The mechanism itself, platform-independent: in the initial DAB mode
    the four diodes antiparallel to CLOSED switches sit across structurally
    shorted node pairs; the four across OPEN switches do not. This is the
    classification the event machinery now trusts instead of the solved
    affine row's dust."""

    simulator = HybridSimulator(load_model(DAB))
    gates = {"p.high": True, "p.low": False, "s.high": False, "s.low": True}
    _, mode = simulator.circuit.stable_mode(
        gates,
        dict(simulator.circuit.initial_diodes),
        simulator.circuit.initial_state,
        simulator.guard_tolerance,
        time=0.0,
    )
    across_closed = {"dpah": ("pa", "dcp"), "dpbl": ("gnd", "pb"),
                     "dsal": ("gnd", "sa"), "dsbh": ("sb", "outp")}
    across_open = {"dpal": ("gnd", "pa"), "dpbh": ("pb", "dcp"),
                   "dsah": ("sa", "outp"), "dsbl": ("gnd", "sb")}
    for diode_id, (p, n) in across_closed.items():
        assert mode.nodes_structurally_shorted(p, n), diode_id
    for diode_id, (p, n) in across_open.items():
        assert not mode.nodes_structurally_shorted(p, n), diode_id


def test_the_shooting_oracle_survives_the_perturbed_operating_point() -> None:
    """The d/d(vin) oracle of the LLC/DAB gradient gate: solve_periodic_steady_state at the
    perturbed vin was the original livelock failure — a gate column was refused for
    the lack of it. It must converge on both sides of the old knife."""

    for vin in ("399.6 V", "400.4 V"):
        steady = solve_periodic_steady_state(load_model(DAB, overrides={"vin": vin}))
        assert steady.converged
