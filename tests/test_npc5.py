"""Five-level multilevel: DNPC bridge, generic band law and anchor repair.

The level-shifted law was written level-generically for the three-level NPC; this module
pins its five-level instantiation end to end: 24 interlocked gates, five
pole levels, the staircase gating invariant, the bit-exact law-recompute
contract, and the deterministic anchor repair that resolves the cold-start
mode of a 42-diode bridge no distance-ordered enumeration can reach.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pear_sim.analysis.npc_modulation import normalized_npc_modulation_cycles
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import SimulationError
from pear_sim.run.io import write_result
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "npc5_three_phase.yaml"


@pytest.fixture(scope="module")
def npc5_run():
    model = load_model(EXAMPLE)
    simulator = HybridSimulator(model)
    result = simulator.simulate()
    return model, simulator, result


def test_bridge_shape(npc5_run) -> None:
    model, _simulator, _result = npc5_run
    assert len(model.controls) == 24
    assert sum(1 for c in model.components if c.type == "diode") == 42
    assert sum(1 for c in model.components if c.type == "ideal_switch") == 24


def test_pole_voltage_visits_all_five_levels(npc5_run) -> None:
    _model, _simulator, result = npc5_run
    pole = np.asarray(result.signals["v(pa)"])
    for level in (-600.0, -300.0, 0.0, 300.0, 600.0):
        near = np.abs(pole - level) < 30.0
        assert near.mean() > 0.02, f"level {level} V barely visited"
    assert pole.max() <= 600.0 + 1e-6
    assert pole.min() >= -600.0 - 1e-6


def test_fundamental_follows_the_modulation_index(npc5_run) -> None:
    _model, _simulator, result = npc5_run
    time = np.asarray(result.time)
    signal = np.asarray(result.signals["v(pa)"])
    window = time >= time[-1] - 1.0 / 50.0
    t = time[window]
    y = signal[window]
    cos_part = 2.0 * np.trapezoid(y * np.cos(2 * np.pi * 50.0 * t), t) * 50.0
    sin_part = 2.0 * np.trapezoid(y * np.sin(2 * np.pi * 50.0 * t), t) * 50.0
    amplitude = float(math.hypot(cos_part, sin_part))
    # 3 us dead time at 2 kHz costs ~1% of the fundamental (real physics,
    # documented for the 3L law test as well) — hence the 2% band.
    assert amplitude == pytest.approx(0.9 * 600.0, rel=2e-2)


def test_cold_start_uses_the_anchor_repair_then_stays_warm(npc5_run) -> None:
    _model, simulator, _result = npc5_run
    stats = simulator.circuit.active_set_search_stats()
    assert stats["anchor_repairs"] == 1  # the cold start only
    assert stats["calls"] > 100
    # Warm-started queries stay in a small Hamming ball; measured on this
    # model: max 298, mean ~18 (42 diodes: 2**42 would be brute force).
    assert stats["max_candidates_in_call"] <= 1500
    assert stats["candidates_examined"] / stats["calls"] <= 120.0


def test_law_recompute_contract_holds_for_five_levels(npc5_run) -> None:
    _model, _simulator, result = npc5_run
    normalized = normalized_npc_modulation_cycles(result)
    assert normalized
    row = normalized[len(normalized) // 2]
    assert row["levels"] == 5
    assert set(row["pairs"]["a"]) == {"p1", "p2", "p3", "p4"}
    # Band nesting: a higher pair's duty never exceeds a lower pair's.
    for pairs in (row["pairs"][phase] for phase in ("a", "b", "c")):
        duties = [pairs[f"p{k}"]["duty"] for k in range(1, 5)]
        assert all(duties[k] <= duties[k + 1] for k in range(3))


def test_staircase_interlock_fails_closed(npc5_run) -> None:
    model, _simulator, _result = npc5_run
    simulator = HybridSimulator(model)
    gates = {gate_id: False for gate_id in model.controls}
    gates["mod.a.h1"] = True  # top device without its pole-adjacent neighbor
    with pytest.raises(SimulationError, match="h1 on without h2"):
        simulator._assert_complementary_interlocks(gates, time=0.0)


def test_bundle_confidence_maps_to_val010(npc5_run, tmp_path: Path) -> None:
    _model, _simulator, result = npc5_run
    destination = tmp_path / "bundle"
    write_result(result, destination)
    observation = json.loads(
        (destination / "npc-modulation.json").read_text(encoding="utf-8")
    )
    assert observation["block"]["levels"] == 5
    confidence = json.loads(
        (destination / "confidence-card.json").read_text(encoding="utf-8")
    )
    rows = [
        row for row in confidence["features"] if row["validation_ref"] == "VAL-010"
    ]
    assert len(rows) == 1 and rows[0]["status"] == "partial"
