"""Modal resonance audit: poles of the accepted modes vs what excites them.

The reference values are closed form. For a filter inductor L feeding a
capacitor C damped by a parallel load R:

    f_n = 1 / (2*pi*sqrt(L*C))          undamped natural frequency
    zeta = (1/(2*R)) * sqrt(L/C)        damping ratio
    f_d = f_n * sqrt(1 - zeta**2)       what an eigenvalue's imaginary part is

so a larger load resistance damps the filter LESS — the case the audit
exists to catch.
"""

import json
import math
from pathlib import Path

import pytest

from pear_sim.analysis.resonance import (
    LIGHTLY_DAMPED_RATIO,
    PROXIMITY_FRACTION,
    build_resonance_observation,
)
from pear_sim.core.solver import HybridSimulator
from pear_sim.run.io import write_result
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"


def _closed_form(inductance: float, capacitance: float, resistance: float):
    natural = 1.0 / (math.tau * math.sqrt(inductance * capacitance))
    damping = 0.5 / resistance * math.sqrt(inductance / capacitance)
    return natural * math.sqrt(max(0.0, 1.0 - damping**2)), damping


def _filtered_buck(
    tmp_path: Path,
    *,
    inductance: str,
    capacitance: str,
    load: str,
    switching: str = "20 kHz",
):
    body = f"""
schema: pear-sim/v0.1
name: filtered_buck
circuit:
  ground: gnd
  components:
    - {{id: vin, type: voltage_source, p: in, n: gnd, value: 100 V}}
    - {{id: q1, type: ideal_switch, p: in, n: sw, control: pwm}}
    - {{id: d1, type: diode, p: gnd, n: sw, initial: off}}
    - {{id: l1, type: inductor, p: sw, n: out, value: {inductance}, initial_current: 0 A}}
    - {{id: c1, type: capacitor, p: out, n: gnd, value: {capacitance}, initial_voltage: 0 V}}
    - {{id: r1, type: resistor, p: out, n: gnd, value: {load}}}
controls:
  pwm: {{type: pwm, frequency: {switching}, duty: 40 %, phase: 0 s}}
analysis:
  transient:
    start: 0 s
    stop: 2 ms
    output_step: 2 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11
outputs: [v(out), i(l1)]
"""
    path = tmp_path / "model.yaml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return load_model(path)


def test_filter_pole_matches_the_closed_form(tmp_path: Path) -> None:
    model = _filtered_buck(
        tmp_path, inductance="1 mH", capacitance="100 uF", load="10 ohm"
    )
    observation = build_resonance_observation(HybridSimulator(model).simulate())
    assert observation["status"] == "ok"
    assert len(observation["resonances"]) == 1
    resonance = observation["resonances"][0]

    expected_frequency, expected_damping = _closed_form(1e-3, 100e-6, 10.0)
    assert resonance["frequency_hz"] == pytest.approx(expected_frequency, rel=1e-9)
    assert resonance["damping_ratio"] == pytest.approx(expected_damping, rel=1e-9)
    # The filter is present in every accepted mode, so it is structural.
    assert resonance["structural"]
    assert resonance["mode_coverage_fraction"] > 0.9


def test_a_lighter_load_damps_the_filter_less(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir(parents=True, exist_ok=True)
    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    weak = build_resonance_observation(
        HybridSimulator(
            _filtered_buck(
                tmp_path / "a", inductance="1 mH", capacitance="100 uF", load="10 ohm"
            )
        ).simulate()
    )["resonances"][0]
    light = build_resonance_observation(
        HybridSimulator(
            _filtered_buck(
                tmp_path / "b", inductance="1 mH", capacitance="100 uF", load="100 ohm"
            )
        ).simulate()
    )["resonances"][0]
    assert light["damping_ratio"] < weak["damping_ratio"]
    assert light["damping_ratio"] == pytest.approx(
        _closed_form(1e-3, 100e-6, 100.0)[1], rel=1e-9
    )


def test_resonance_on_the_carrier_is_flagged(tmp_path: Path) -> None:
    # L = 1 mH with C = 63.3 nF places the pole at ~20 kHz, exactly the
    # carrier, and a 1 kohm load leaves it lightly damped: the design
    # question the audit exists to raise.
    model = _filtered_buck(
        tmp_path, inductance="1 mH", capacitance="63.3 nF", load="1 kohm"
    )
    observation = build_resonance_observation(HybridSimulator(model).simulate())
    assert observation["status"] == "flagged"
    flagged = [entry for entry in observation["resonances"] if entry["flagged"]]
    assert len(flagged) == 1
    entry = flagged[0]
    assert entry["damping_ratio"] < LIGHTLY_DAMPED_RATIO
    assert entry["separation_fraction"] <= PROXIMITY_FRACTION
    assert entry["nearest_excitation_hz"] == pytest.approx(20_000.0)
    # Deep DCM parks the filter pole while the inductor is frozen, so the
    # coverage is well under a half: the case a stricter threshold would hide.
    assert 0.2 <= entry["mode_coverage_fraction"] < 0.5


def test_purely_dissipative_network_reports_no_resonance(tmp_path: Path) -> None:
    body = """
schema: pear-sim/v0.1
name: rl_only
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 10 V}
    - {id: r1, type: resistor, p: in, n: mid, value: 10 ohm}
    - {id: l1, type: inductor, p: mid, n: gnd, value: 1 mH, initial_current: 0 A}
analysis:
  transient: {start: 0 s, stop: 1 ms, output_step: 10 us}
outputs: [i(l1)]
"""
    path = tmp_path / "rl.yaml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    observation = build_resonance_observation(
        HybridSimulator(load_model(path)).simulate()
    )
    assert observation["resonances"] == []
    assert observation["status"] == "ok"


def test_npc_lc_filter_example_is_clear_of_the_carrier_band() -> None:
    model = load_model(EXAMPLES / "npc_three_phase_lc_filter.yaml")
    observation = build_resonance_observation(HybridSimulator(model).simulate())

    assert observation["status"] == "ok"
    structural = [entry for entry in observation["resonances"] if entry["structural"]]
    assert len(structural) == 1
    entry = structural[0]

    expected_frequency, expected_damping = _closed_form(1e-3, 100e-6, 5.0)
    assert entry["frequency_hz"] == pytest.approx(expected_frequency, rel=1e-6)
    assert entry["damping_ratio"] == pytest.approx(expected_damping, rel=1e-6)
    # Textbook placement: well above the fundamental, well below the carrier.
    assert 50.0 < entry["frequency_hz"] < 2000.0
    assert not entry["near_excitation"]
    assert 2000.0 in observation["excitation"]["carrier_frequencies_hz"]


def test_bundle_carries_the_resonance_artifact(tmp_path: Path) -> None:
    model = _filtered_buck(
        tmp_path, inductance="1 mH", capacitance="63.3 nF", load="1 kohm"
    )
    result = HybridSimulator(model).simulate()
    destination = tmp_path / "bundle"
    write_result(result, destination)

    observation = json.loads(
        (destination / "resonance.json").read_text(encoding="utf-8")
    )
    assert observation["schema"] == "pear-sim/resonance-observation/v0.1"
    assert observation["status"] == "flagged"

    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    assert summary["resonance"]["artifact"] == "resonance.json"
    assert summary["resonance"]["status"] == "flagged"
    codes = {warning["code"] for warning in summary["warnings"]}
    assert "RESONANCE_NEAR_EXCITATION" in codes

    run = json.loads((destination / "run.json").read_text(encoding="utf-8"))
    assert run["files"]["resonance"] == "resonance.json"
