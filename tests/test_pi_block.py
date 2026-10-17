"""The ``pi`` signal block is shorthand: it must mean exactly the hand-written graph."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "buck_closed_loop_signals.yaml"

HAND_WRITTEN = """    err_dt:
      type: gain
      input: err
      gain: 2.0e-5           # * T
      unit: V
    int_prev:
      type: unit_delay
      input: integral
      initial: 0.0
      unit: V
    integral:
      type: sum
      inputs: [int_prev, err_dt]
      unit: V
    kp_term:
      type: gain
      input: err
      gain: 0.002
      unit: V
    ki_term:
      type: gain
      input: integral
      gain: 30.0
      unit: V
    u:
      type: sum
      inputs: [kp_term, ki_term]
      unit: V
    duty:
      type: saturate
      input: u
      min: 0.05
      max: 0.95
      unit: V
"""

PI_BLOCK = """    duty:
      type: pi
      input: err
      kp: 0.002
      ki: 30.0
      min: 0.05
      max: 0.95
      kaw: 0
      unit: V
"""


def _variant(tmp_path: Path, replacement: str, name: str) -> Path:
    text = EXAMPLE.read_text(encoding="utf-8")
    assert HAND_WRITTEN in text, "the example's controller changed; update this test"
    text = text.replace(HAND_WRITTEN, replacement).replace("  - s(integral)\n", "")
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_the_pi_block_reproduces_the_hand_written_controller(tmp_path: Path) -> None:
    reference = HybridSimulator(load_model(_variant(tmp_path, HAND_WRITTEN, "hand.yaml"))).simulate()
    candidate = HybridSimulator(load_model(_variant(tmp_path, PI_BLOCK, "pi.yaml"))).simulate()
    # Same law; the integral is accumulated as sum(ki*T*e) instead of
    # ki*sum(T*e), so only floating-point rounding may differ.
    for signal in ("v(out)", "i(l1)", "s(duty)"):
        a, b = reference.signals[signal], candidate.signals[signal]
        assert np.max(np.abs(a - b)) <= 1e-9 * max(1.0, float(np.max(np.abs(a)))), signal


def test_back_calculation_keeps_a_saturated_integrator_bounded(tmp_path: Path) -> None:
    # A unreachable reference (60 V from a 24 V buck) pins the duty at its
    # maximum. Without anti-windup the integrator grows for the whole run;
    # with back-calculation it settles where ki*T*e balances the clamp gap.
    def run(kaw: str) -> np.ndarray:
        block = PI_BLOCK.replace("kaw: 0", f"kaw: {kaw}")
        path = _variant(tmp_path, block, f"aw{kaw}.yaml")
        text = path.read_text().replace("value: 12.0", "value: 60.0").replace(
            "  - s(duty)\n", "  - s(duty)\n  - s(duty__int)\n"
        )
        path.write_text(text)
        return HybridSimulator(load_model(path)).simulate().signals["s(duty__int)"]

    wound = run("0")
    held = run("1")
    # Without anti-windup the integrator keeps climbing through the run.
    assert wound[-1] > 1.5 * wound[len(wound) // 2] > 0
    # With it, int settles where int[k] = int[k-1]: kaw*(u - raw) = -ki*T*e,
    # i.e. int = max + (ki*T - kp)*e. With e ~ 36 V that is ~0.90; the output
    # ripple moves e, so int breathes by a few mV but stays put.
    tail = held[len(held) // 2 :]
    assert float(np.max(tail) - np.min(tail)) < 0.01
    assert float(np.mean(tail)) == pytest.approx(0.95 + (30.0 * 2e-5 - 0.002) * 36.0, abs=0.02)
    assert wound[-1] > 5 * float(np.mean(tail))


def test_pi_block_fields_fail_closed(tmp_path: Path) -> None:
    for bad, message in (
        (PI_BLOCK.replace("      min: 0.05\n", ""), "both min and max"),
        (PI_BLOCK.replace("      kaw: 0\n", "      kd: 1.0\n"), "unknown field"),
    ):
        with pytest.raises(ModelError, match=message):
            load_model(_variant(tmp_path, bad, "bad.yaml"))
