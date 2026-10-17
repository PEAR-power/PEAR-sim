from pathlib import Path

import numpy as np

from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator


EXAMPLE = Path(__file__).parents[1] / "examples" / "inverter_three_phase_six_step.yaml"


def _fundamental_rms(time: np.ndarray, values: np.ndarray, frequency: float) -> float:
    phasor_peak = 2.0 * np.mean(values * np.exp(-2j * np.pi * frequency * time))
    return float(abs(phasor_peak) / np.sqrt(2.0))


def test_three_phase_six_step_line_voltage_and_balance():
    model = load_model(EXAMPLE)
    assert set(model.controls) == {
        "leg_a.high",
        "leg_a.low",
        "leg_b.high",
        "leg_b.low",
        "leg_c.high",
        "leg_c.low",
    }
    assert all(control.kind == "complementary_pwm" for control in model.controls.values())
    result = HybridSimulator(model).simulate()
    keep = (result.time >= 5e-3) & (result.time < 10e-3)
    time = result.time[keep]
    vab = result.signals["v(vab)"][keep]
    expected_fundamental_rms = np.sqrt(6.0) / np.pi * 400.0
    assert abs(_fundamental_rms(time, vab, 1e3) - expected_fundamental_rms) < 0.8

    phase_rms = [
        float(np.sqrt(np.mean(result.signals[name][keep] ** 2)))
        for name in ("i(la)", "i(lb)", "i(lc)")
    ]
    assert max(phase_rms) - min(phase_rms) < 0.03
    assert result.stats["scheduled_events"] >= 100
    assert result.stats["compiled_topologies"] == 6
    for interval in result.mode_intervals:
        gates = interval["gates"]
        for leg in ("leg_a", "leg_b", "leg_c"):
            assert not (gates[f"{leg}.high"] and gates[f"{leg}.low"])
