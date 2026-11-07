"""Floating-node memo in the active-set search changes cost, never outcome.

A diode candidate whose compile fails with "node X has no algebraic support"
fails because of the components touching X alone. The memo remembers that
local state and skips later candidates that repeat it instead of compiling
them. The claim under test is exactness: the same accepted modes, the same
search statistics and bit-identical waveforms as a search that compiles every
candidate — and every skip confirmed by a real compile.
"""

from pathlib import Path

import numpy as np
import pytest

from pear_sim.core import circuit as circuit_module
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import TopologyError
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]


def _short_npc(tmp_path: Path) -> Path:
    # The three-phase NPC bridge has 18 ideal diodes and floating clamp
    # midpoints while switches commutate: the memo's home ground. 4 ms keeps
    # the test fast and still crosses hundreds of gate edges.
    text = (ROOT / "examples" / "npc_three_phase.yaml").read_text(encoding="utf-8")
    assert "stop: 40 ms" in text
    path = tmp_path / "npc_short.yaml"
    path.write_text(text.replace("stop: 40 ms", "stop: 4 ms"), encoding="utf-8")
    return path


def _run(model_path: Path) -> tuple[HybridSimulator, object]:
    simulator = HybridSimulator(load_model(model_path))
    return simulator, simulator.simulate()


def test_memo_is_bit_identical_to_compiling_every_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = _short_npc(tmp_path)

    skipped: list[circuit_module._SkippedFloatingTopology] = []
    original_init = circuit_module._SkippedFloatingTopology.__init__

    def recording_init(self, *args):  # type: ignore[no-untyped-def]
        original_init(self, *args)
        skipped.append(self)

    monkeypatch.setattr(
        circuit_module._SkippedFloatingTopology, "__init__", recording_init
    )
    with_memo, result_memo = _run(model_path)
    monkeypatch.undo()

    monkeypatch.setattr(
        circuit_module.Circuit, "_predicted_floating_node", lambda self, topology: None
    )
    without_memo, result_full = _run(model_path)

    assert skipped, "the memo was never exercised; the test proves nothing"
    assert len(with_memo.circuit._cache) < len(without_memo.circuit._cache)
    assert (
        with_memo.circuit.active_set_search_stats()
        == without_memo.circuit.active_set_search_stats()
    )
    assert (
        with_memo.circuit.compiled_topology_stats()
        == without_memo.circuit.compiled_topology_stats()
    )
    assert list(result_memo.signals) == list(result_full.signals)
    assert np.array_equal(result_memo.time, result_full.time)
    for signal in result_memo.signals:
        assert np.array_equal(result_memo.signals[signal], result_full.signals[signal]), signal

    # Soundness, checked one by one: every skip is a failure the full compile
    # reproduces (exact_message compiles for real and raises if it does not).
    for skip in skipped:
        assert "has no algebraic support" in skip.exact_message()


def test_skipped_candidate_reports_the_full_compile_message(tmp_path: Path) -> None:
    simulator, _ = _run(_short_npc(tmp_path))
    circuit = simulator.circuit
    node, signatures = next(iter(circuit._floating_signatures.items()))
    failed = next(
        topology
        for topology, value in circuit._cache.items()
        if isinstance(value, TopologyError) and getattr(value, "floating_node", None) == node
    )
    expected = str(circuit._cache[failed])
    # Ask again through the memo on a fresh cache: the message the user would
    # see is the full compile's, word for word.
    circuit._cache.clear()
    with pytest.raises(circuit_module._SkippedFloatingTopology) as raised:
        circuit.compile(
            failed.gate_dict(),
            failed.diode_dict(),
            failed.constrained_inductors,
            failed.saturated_inductors,
            skip_known_floating=True,
        )
    assert circuit_module._failure_text(raised.value) == expected
    assert circuit._node_signature(node, failed) in signatures
