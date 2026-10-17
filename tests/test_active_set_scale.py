"""Kernel scale: lazy, budget-bounded diode active-set search.

The former search materialized all 2**n diode configurations and hard-failed
above 16 diodes. The lazy search must (a) visit candidates in EXACTLY the
former order — (Hamming distance from preferred, lexicographic) — so every
model the old limit accepted behaves bit-identically, and (b) make a
three-phase three-level NPC bridge (18 ideal diodes) simulatable with a
warm-started search whose per-event work stays near the preferred mode.
"""

import itertools
from pathlib import Path

import pytest

from pear_sim.core.circuit import Circuit, _hamming_ordered_candidates
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import TopologyError
from pear_sim.limits import MAX_IDEAL_DIODES
from pear_sim.spec.model import load_model


def _eager_order(preferred: tuple[bool, ...]) -> list[tuple[bool, ...]]:
    candidates = list(itertools.product((False, True), repeat=len(preferred)))
    candidates.sort(
        key=lambda bits: (sum(a != b for a, b in zip(bits, preferred)), bits)
    )
    return candidates


@pytest.mark.parametrize(
    "preferred",
    [
        (),
        (False,),
        (True,),
        (False, True, False),
        (True, True, False, True),
        (False, True, False, True, True, False, True, False),
    ],
)
def test_lazy_order_is_bit_identical_to_the_former_eager_sort(preferred) -> None:
    lazy = list(_hamming_ordered_candidates(preferred, 2 ** len(preferred)))
    assert lazy == _eager_order(preferred)


def test_budget_truncation_keeps_the_eager_prefix() -> None:
    preferred = (True, False, True, False, True)
    for budget in (1, 3, 7, 12, 31, 32):
        lazy = list(_hamming_ordered_candidates(preferred, budget))
        assert lazy == _eager_order(preferred)[:budget]


def test_large_configuration_space_is_never_materialized() -> None:
    # 40 diodes: 2**40 configurations. The generator must produce the first
    # candidates lazily without building the space.
    preferred = tuple(index % 2 == 0 for index in range(40))
    generator = _hamming_ordered_candidates(preferred, 100)
    first = next(generator)
    assert first == preferred
    count = 1 + sum(1 for _ in generator)
    assert count == 100


def _model(tmp_path: Path, body: str):
    path = tmp_path / "model.yaml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return load_model(path)


_DIODE_RING = """
schema: pear-sim/v0.1
name: diode_ring
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 10 V}
    - {id: r1, type: resistor, p: in, n: a, value: 100 ohm}
    - {id: d1, type: diode, p: a, n: b, initial: off}
    - {id: d2, type: diode, p: b, n: c, initial: off}
    - {id: r2, type: resistor, p: c, n: gnd, value: 100 ohm}
    - {id: rb, type: resistor, p: b, n: gnd, value: 1 kohm}
analysis:
  transient: {start: 0 s, stop: 1 ms, output_step: 10 us}
outputs: [v(c)]
"""


def test_exhausted_budget_falls_back_to_the_anchor_repair(
    tmp_path, monkeypatch
) -> None:
    # The consistent mode (both diodes ON) is at Hamming distance 2 from the
    # preferred all-off guess. A budget of 2 covers distance 0 and one
    # distance-1 candidate only; the deterministic anchor repair must then
    # recover the guard-consistent mode instead of failing.
    model = _model(tmp_path, _DIODE_RING)
    circuit = Circuit(model)
    import pear_sim.core.circuit as circuit_module

    monkeypatch.setattr(circuit_module, "MAX_ACTIVE_SET_CANDIDATES", 2)
    diodes, _mode = circuit.stable_mode(
        {},
        {"d1": False, "d2": False},
        circuit.initial_state,
        1e-9,
        time=0.0,
    )
    assert diodes == {"d1": True, "d2": True}
    assert circuit.active_set_search_stats()["anchor_repairs"] == 1


def test_budget_exhaustion_without_repair_is_a_controlled_diagnostic(
    tmp_path, monkeypatch
) -> None:
    # If the repair cannot resolve a mode either, the search must fail closed
    # with a diagnostic naming both the budget and the repair — never
    # silently pick an infeasible mode.
    model = _model(tmp_path, _DIODE_RING)
    circuit = Circuit(model)
    import pear_sim.core.circuit as circuit_module

    monkeypatch.setattr(circuit_module, "MAX_ACTIVE_SET_CANDIDATES", 2)
    monkeypatch.setattr(
        Circuit, "_anchor_repair", lambda self, *args, **kwargs: None
    )
    with pytest.raises(TopologyError, match="search budget"):
        circuit.stable_mode(
            {},
            {"d1": False, "d2": False},
            circuit.initial_state,
            1e-9,
            time=0.0,
        )


def test_diode_ring_still_resolves_within_the_default_budget(tmp_path) -> None:
    model = _model(tmp_path, _DIODE_RING)
    circuit = Circuit(model)
    diodes, _mode = circuit.stable_mode(
        {},
        {"d1": False, "d2": False},
        circuit.initial_state,
        1e-9,
        time=0.0,
    )
    assert diodes == {"d1": True, "d2": True}
    stats = circuit.active_set_search_stats()
    assert stats["calls"] == 1
    assert 0 < stats["max_candidates_in_call"] <= 4
    assert stats["candidates_examined"] == stats["max_candidates_in_call"]


def test_more_than_the_portable_diode_bound_fails_closed(tmp_path) -> None:
    diodes = "\n".join(
        f"    - {{id: d{index}, type: diode, p: n{index}, n: gnd, initial: off}}"
        for index in range(MAX_IDEAL_DIODES + 1)
    )
    resistors = "\n".join(
        f"    - {{id: r{index}, type: resistor, p: in, n: n{index}, value: 1 kohm}}"
        for index in range(MAX_IDEAL_DIODES + 1)
    )
    body = f"""
schema: pear-sim/v0.1
name: too_many_diodes
circuit:
  ground: gnd
  components:
    - {{id: vin, type: voltage_source, p: in, n: gnd, value: 1 V}}
{resistors}
{diodes}
analysis:
  transient: {{start: 0 s, stop: 1 ms, output_step: 100 us}}
outputs: [v(in)]
"""
    model = _model(tmp_path, body)
    circuit = Circuit(model)
    with pytest.raises(TopologyError, match=str(MAX_IDEAL_DIODES)):
        circuit.stable_mode(
            {},
            {f"d{index}": False for index in range(MAX_IDEAL_DIODES + 1)},
            circuit.initial_state,
            1e-9,
            time=0.0,
        )


def _npc_leg(phase: str, *, pole: str, q1: str, q2: str, q3: str, q4: str) -> str:
    """One three-level NPC leg: 4 switches, 4 antiparallel + 2 clamp diodes."""

    return f"""
    - {{id: q1{phase}, type: ideal_switch, p: dcp, n: n1{phase}, control: {q1}}}
    - {{id: q2{phase}, type: ideal_switch, p: n1{phase}, n: {pole}, control: {q2}}}
    - {{id: q3{phase}, type: ideal_switch, p: {pole}, n: n2{phase}, control: {q3}}}
    - {{id: q4{phase}, type: ideal_switch, p: n2{phase}, n: dcn, control: {q4}}}
    - {{id: d1{phase}, type: diode, p: n1{phase}, n: dcp, initial: off}}
    - {{id: d2{phase}, type: diode, p: {pole}, n: n1{phase}, initial: off}}
    - {{id: d3{phase}, type: diode, p: n2{phase}, n: {pole}, initial: off}}
    - {{id: d4{phase}, type: diode, p: dcn, n: n2{phase}, initial: off}}
    - {{id: d5{phase}, type: diode, p: mid, n: n1{phase}, initial: off}}
    - {{id: d6{phase}, type: diode, p: n2{phase}, n: mid, initial: off}}
    - {{id: l{phase}, type: inductor, p: {pole}, n: load{phase}, value: 2 mH, initial_current: 0 A}}
    - {{id: r{phase}, type: resistor, p: load{phase}, n: mid, value: 5 ohm}}
"""


_NPC_THREE_PHASE = f"""
schema: pear-sim/v0.1
name: npc_three_phase_scale
# 18 ideal diodes: above the former 16-diode wall. Static per-leg operating
# points (positive band: upper pair pulses, inner-high held on; negative
# band mirrored), complementary dead time on the pulsing pair so natural
# commutation through the clamp/antiparallel diodes exercises the search.
circuit:
  ground: mid
  components:
    - {{id: vp, type: voltage_source, p: dcp, n: mid, value: 600 V}}
    - {{id: vn, type: voltage_source, p: mid, n: dcn, value: 600 V}}
{_npc_leg("a", pole="pa", q1="upa.high", q2="hold_on", q3="upa.low", q4="hold_off")}
{_npc_leg("b", pole="pb", q1="hold_off", q2="inb.high", q3="hold_on", q4="inb.low")}
{_npc_leg("c", pole="pc", q1="upc.high", q2="hold_on", q3="upc.low", q4="hold_off")}
controls:
  upa: {{type: complementary_pwm, frequency: 2 kHz, duty: 60 %, phase: 0 s, dead_time: 2 us}}
  inb: {{type: complementary_pwm, frequency: 2 kHz, duty: 45 %, phase: 0 s, dead_time: 2 us}}
  upc: {{type: complementary_pwm, frequency: 2 kHz, duty: 30 %, phase: 0 s, dead_time: 2 us}}
  hold_on: {{type: pwm, frequency: 2 kHz, duty: 100 %, phase: 0 s}}
  hold_off: {{type: pwm, frequency: 2 kHz, duty: 0 %, phase: 0 s}}
analysis:
  transient:
    start: 0 s
    stop: 5 ms
    output_step: 2 us
    solver: dopri5
    rtol: 1.0e-8
    atol: 1.0e-10
outputs: [v(pa), v(pb), v(pc), i(la), i(lb), i(lc), i(d5a), i(d6b)]
"""


def test_three_phase_npc_with_18_diodes_simulates(tmp_path) -> None:
    model = _model(tmp_path, _NPC_THREE_PHASE)
    assert len([c for c in model.components if c.type == "diode"]) == 18
    simulator = HybridSimulator(model)
    result = simulator.simulate()

    # Leg A pulses between the positive rail and the clamped midpoint.
    pole_a = result.signals["v(pa)"]
    assert max(pole_a) > 550.0
    # Leg B operates in the negative band.
    pole_b = result.signals["v(pb)"]
    assert min(pole_b) < -550.0
    # The load currents settle to the band-consistent signs.
    assert result.signals["i(la)"][-1] > 10.0
    assert result.signals["i(lb)"][-1] < -10.0

    stats = simulator.circuit.active_set_search_stats()
    assert stats["calls"] > 50
    # Warm start: the feasible mode stays within a small Hamming ball of the
    # preferred configuration — this is the measured-complexity bound that
    # replaces the former 16-diode wall. Measured on this model: max 494,
    # mean ~91 (synchronized carriers flip several legs at one timestamp,
    # which is the worst case); 2**18 = 262144 would be brute force.
    assert stats["max_candidates_in_call"] <= 1000
    assert stats["candidates_examined"] / stats["calls"] <= 150.0
