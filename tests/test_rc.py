from pathlib import Path

import numpy as np

from pear_sim.circuit import Circuit
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator


def test_rc_step_matches_analytic_solution(tmp_path: Path):
    model_path = tmp_path / "rc.yaml"
    model_path.write_text(
        """
schema: pear-sim/v0.1
name: rc_step
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 10 V}
    - {id: r1, type: resistor, p: in, n: out, value: 1 kohm}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 1 uF, initial_voltage: 0 V}
analysis:
  transient: {start: 0 s, stop: 5 ms, output_step: 10 us, solver: dopri5, rtol: 1e-10, atol: 1e-12}
outputs: [v(out)]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    model = load_model(model_path)
    result = HybridSimulator(model).simulate()
    expected = 10.0 * (1.0 - np.exp(-result.time / 1e-3))
    assert np.max(np.abs(result.signals["v(out)"] - expected)) < 2e-7
    circuit = Circuit(model)
    mode = circuit.compile({}, {})
    assert abs(mode.power_balance_residual(np.array([4.0]))) < 1e-10
