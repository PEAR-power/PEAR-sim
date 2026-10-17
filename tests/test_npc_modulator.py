"""Three-level NPC level-shifted (PD) modulator.

The law under test, restated independently of the scheduler: per carrier
cell the sampled [-1, 1] reference falls into one of two stacked bands;
the band's complementary pair realizes a center-aligned pulse with dead
time while the other pair saturates (upper pair duty = clip(r, 0, 1),
lower pair duty = clip(1 + r, 0, 1)); band transitions hand over at the
cell boundary through one dead time; commanded pulses narrower than the
dead time are suppressed exactly like a hardware minimum-pulse interlock.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pear_sim.analysis.npc_modulation import (
    NpcModulationError,
    normalized_npc_modulation_cycles,
)
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError, SimulationError
from pear_sim.run.io import write_result
from pear_sim.spec.controls import npc_pair_duties
from pear_sim.spec.model import load_model


def _npc_body(
    *,
    modulation_index: str = "0.9",
    dead_time: str = "3 us",
    carrier: str = "2 kHz",
    fundamental: str = "100 Hz",
    stop: str = "10 ms",
    extra_control_fields: str = "",
) -> str:
    legs = ""
    for phase, pole in (("a", "pa"), ("b", "pb"), ("c", "pc")):
        legs += f"""
    - {{id: q1{phase}, type: ideal_switch, p: dcp, n: n1{phase}, control: mod.{phase}.outer_high}}
    - {{id: q2{phase}, type: ideal_switch, p: n1{phase}, n: {pole}, control: mod.{phase}.inner_high}}
    - {{id: q3{phase}, type: ideal_switch, p: {pole}, n: n2{phase}, control: mod.{phase}.inner_low}}
    - {{id: q4{phase}, type: ideal_switch, p: n2{phase}, n: dcn, control: mod.{phase}.outer_low}}
    - {{id: d1{phase}, type: diode, p: n1{phase}, n: dcp, initial: off}}
    - {{id: d2{phase}, type: diode, p: {pole}, n: n1{phase}, initial: off}}
    - {{id: d3{phase}, type: diode, p: n2{phase}, n: {pole}, initial: off}}
    - {{id: d4{phase}, type: diode, p: dcn, n: n2{phase}, initial: off}}
    - {{id: d5{phase}, type: diode, p: mid, n: n1{phase}, initial: off}}
    - {{id: d6{phase}, type: diode, p: n2{phase}, n: mid, initial: off}}
    - {{id: l{phase}, type: inductor, p: {pole}, n: load{phase}, value: 1 mH, initial_current: 0 A}}
    - {{id: r{phase}, type: resistor, p: load{phase}, n: mid, value: 5 ohm}}
"""
    return f"""
schema: pear-sim/v0.1
name: npc_modulator_test
circuit:
  ground: mid
  components:
    - {{id: vp, type: voltage_source, p: dcp, n: mid, value: 600 V}}
    - {{id: vn, type: voltage_source, p: mid, n: dcn, value: 600 V}}
{legs}
controls:
  mod:
    type: three_phase_npc_spwm
    carrier_frequency: {carrier}
    fundamental_frequency: {fundamental}
    modulation_index: {modulation_index}
    dead_time: {dead_time}
{extra_control_fields}
analysis:
  transient:
    start: 0 s
    stop: {stop}
    output_step: 5 us
    solver: dopri5
    rtol: 1.0e-8
    atol: 1.0e-10
outputs: [v(pa), v(pb), v(pc), i(la), i(lb), i(lc)]
"""


def _load(tmp_path: Path, body: str):
    path = tmp_path / "model.yaml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return load_model(path)


# ---------------------------------------------------------------------------
# The band law itself
# ---------------------------------------------------------------------------


def test_band_decomposition_matches_the_level_shifted_law() -> None:
    for reference, expected_lower, expected_upper in (
        (-1.0, 0.0, 0.0),
        (-0.5, 0.5, 0.0),
        (0.0, 1.0, 0.0),
        (0.25, 1.0, 0.25),
        (1.0, 1.0, 1.0),
    ):
        lower, upper = npc_pair_duties(reference, 3)
        assert lower == pytest.approx(expected_lower, abs=0.0)
        assert upper == pytest.approx(expected_upper, abs=0.0)


def test_band_decomposition_is_level_generic() -> None:
    # Five levels: four stacked bands of height 0.5. The v0.1 loader refuses
    # levels != 3, but the law itself must already generalize.
    duties = npc_pair_duties(0.25, 5)
    assert duties == (1.0, 1.0, 0.5, 0.0)


# ---------------------------------------------------------------------------
# Loader validation
# ---------------------------------------------------------------------------


def test_expansion_produces_twelve_paired_gates(tmp_path: Path) -> None:
    model = _load(tmp_path, _npc_body())
    assert len(model.controls) == 12
    upper_high = model.controls["mod.a.outer_high"]
    assert upper_high.partner_id == "mod.a.inner_low"
    assert upper_high.pair_label == "upper"
    lower_high = model.controls["mod.a.inner_high"]
    assert lower_high.partner_id == "mod.a.outer_low"
    assert lower_high.pair_label == "lower"
    assert upper_high.levels == 3


def test_unsupported_level_counts_fail_closed(tmp_path: Path) -> None:
    # Three and five levels are supported; anything else fails closed (the
    # JSON Schema rejects it first, the loader guard is the byte-independent
    # second line of defense).
    with pytest.raises(ModelError, match="levels"):
        _load(tmp_path, _npc_body(extra_control_fields="    levels: 4"))


def test_five_level_expansion_produces_24_paired_gates() -> None:
    from pear_sim.spec.controls import parse_controls

    controls = parse_controls(
        {
            "mod": {
                "type": "three_phase_npc_spwm",
                "levels": 5,
                "carrier_frequency": "2 kHz",
                "fundamental_frequency": "50 Hz",
                "modulation_index": 0.9,
                "dead_time": "3 us",
            }
        },
        {},
    )
    assert len(controls) == 24
    top = controls["mod.a.h1"]
    assert top.partner_id == "mod.a.l1"
    assert top.pair_label == "p1"
    assert top.levels == 5
    bottom = controls["mod.b.l4"]
    assert bottom.partner_id == "mod.b.h4"
    assert bottom.role == "low"


def test_npc_must_be_the_only_control_source(tmp_path: Path) -> None:
    body = _npc_body(extra_control_fields="").replace(
        "controls:\n  mod:",
        "controls:\n  aux:\n    type: pwm\n    frequency: 2 kHz\n"
        "    duty: 50 %\n    phase: 0 s\n  mod:",
    )
    # The JSON Schema's exclusive-modulator shape rejects the mix first; the
    # loader's aloneness rule remains as byte-independent defense in depth.
    with pytest.raises(ModelError):
        _load(tmp_path, body)


def test_quarter_period_dead_time_bound(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="quarter"):
        _load(tmp_path, _npc_body(dead_time="130 us"))


def test_modulation_index_linear_range(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="modulation_index"):
        _load(tmp_path, _npc_body(modulation_index="1.0"))


# ---------------------------------------------------------------------------
# Electrical behavior
# ---------------------------------------------------------------------------


def _fundamental_amplitude(
    time: np.ndarray, signal: np.ndarray, frequency: float
) -> float:
    window = time >= time[-1] - 1.0 / frequency
    t = time[window]
    y = signal[window]
    cos_part = 2.0 * np.trapezoid(y * np.cos(2 * np.pi * frequency * t), t) * frequency
    sin_part = 2.0 * np.trapezoid(y * np.sin(2 * np.pi * frequency * t), t) * frequency
    return float(math.hypot(cos_part, sin_part))


def test_pole_voltage_fundamental_follows_the_modulation_index(tmp_path: Path) -> None:
    # Dead time 0 pins the law without the (real, documented) dead-time
    # voltage loss; m = 0.9 on a +-600 V link commands 540 V fundamental.
    model = _load(
        tmp_path,
        _npc_body(modulation_index="0.9", dead_time="0 s", stop="20 ms"),
    )
    result = HybridSimulator(model).simulate()
    time = np.asarray(result.time)
    amplitude = _fundamental_amplitude(
        time, np.asarray(result.signals["v(pa)"]), 100.0
    )
    assert amplitude == pytest.approx(0.9 * 600.0, rel=5e-3)
    # Three-level pole: the waveform must visit +600, 0 and -600.
    pole = np.asarray(result.signals["v(pa)"])
    assert pole.max() > 550.0
    assert pole.min() < -550.0
    near_zero_fraction = float(np.mean(np.abs(pole) < 50.0))
    assert near_zero_fraction > 0.1


def test_dead_time_run_survives_interlocks_and_records_cycles(
    tmp_path: Path,
) -> None:
    model = _load(tmp_path, _npc_body(stop="10 ms"))
    result = HybridSimulator(model).simulate()
    # 2 kHz carrier over 10 ms: 20 in-window cells (plus schedule padding).
    in_window = [
        cycle
        for cycle in result.modulation_cycles
        if cycle["start_s"] < 10e-3 and cycle["end_s"] > 0.0
    ]
    assert len(in_window) >= 20
    normalized = normalized_npc_modulation_cycles(result)
    assert len(normalized) == len(result.modulation_cycles)


def test_npc_interlock_assertion_fails_closed(tmp_path: Path) -> None:
    model = _load(tmp_path, _npc_body())
    simulator = HybridSimulator(model)
    gates = {gate_id: False for gate_id in model.controls}
    gates["mod.a.outer_high"] = True  # outer on without its inner neighbor
    with pytest.raises(SimulationError, match="NPC gating interlock"):
        simulator._assert_complementary_interlocks(gates, time=0.0)


def test_tampered_command_record_is_rejected(tmp_path: Path) -> None:
    model = _load(tmp_path, _npc_body(stop="5 ms"))
    result = HybridSimulator(model).simulate()
    tampered = [json.loads(json.dumps(cycle)) for cycle in result.modulation_cycles]
    tampered[3]["pairs"]["a"]["upper"]["duty"] += 1e-9
    stub = SimpleNamespace(model=result.model, modulation_cycles=tampered)
    with pytest.raises(NpcModulationError, match="duty"):
        normalized_npc_modulation_cycles(stub)


def test_band_edge_pulse_is_suppressed_like_a_gate_driver(tmp_path: Path) -> None:
    # reference_phase places one carrier sample at electrical angle
    # ~ pi - 0.00314 rad: the upper-pair command (0.9 * 0.00314 * 500 us
    # ~ 1.4 us) is narrower than the 3 us dead time and must be suppressed,
    # with the suppression declared per cell.
    model = _load(
        tmp_path,
        _npc_body(
            extra_control_fields="    reference_phase: -0.000245 s",
            stop="10 ms",
        ),
    )
    result = HybridSimulator(model).simulate()
    suppressed = [
        (cycle["cycle_index"], phase, pair)
        for cycle in result.modulation_cycles
        for phase, pairs in cycle["pairs"].items()
        for pair, entry in pairs.items()
        if entry["suppressed"] is not None
    ]
    assert suppressed, "expected at least one suppressed band-edge pulse"
    normalized_npc_modulation_cycles(result)  # law recompute must agree


# ---------------------------------------------------------------------------
# Bundle artifacts
# ---------------------------------------------------------------------------


def test_bundle_carries_npc_modulation_artifacts(tmp_path: Path) -> None:
    model = _load(tmp_path, _npc_body(stop="5 ms"))
    result = HybridSimulator(model).simulate()
    destination = tmp_path / "bundle"
    write_result(result, destination)

    assert (destination / "npc-modulation.json").exists()
    assert (destination / "modulation-cycles.jsonl").exists()
    observation = json.loads(
        (destination / "npc-modulation.json").read_text(encoding="utf-8")
    )
    assert observation["schema"] == "pear-sim/npc-modulation-observation/v0.1"
    assert observation["block"]["levels"] == 3
    assert observation["cycles_recorded"] >= 10
    assert observation["source_binding"]["run_sha256"]

    summary = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    assert summary["npc_modulation"]["artifact"] == "npc-modulation.json"
    warning_codes = {warning["code"] for warning in summary["warnings"]}
    assert "NPC_MODULATOR_EXPERIMENTAL" in warning_codes

    confidence = json.loads(
        (destination / "confidence-card.json").read_text(encoding="utf-8")
    )
    npc_rows = [
        row
        for row in confidence["features"]
        if row["validation_ref"] == "VAL-010"
    ]
    assert len(npc_rows) == 1
    assert npc_rows[0]["status"] == "partial"
    assert "NOT a validated general multilevel claim" in npc_rows[0]["claim_boundary"]
    # The NPC bridge must NOT fall into the non-canonical not_run bucket.
    assert not any(
        row["feature"] == "non-canonical switching topology"
        for row in confidence["features"]
    )


def test_offline_report_renders_the_npc_section(tmp_path: Path) -> None:
    from pear_sim.reporting.report import generate_report

    model = _load(tmp_path, _npc_body(stop="5 ms"))
    result = HybridSimulator(model).simulate()
    destination = tmp_path / "bundle"
    write_result(result, destination)
    artifacts = generate_report(destination, tmp_path / "report")
    html = Path(artifacts.report_html).read_text(encoding="utf-8")
    assert "NPC level-shifted modulation" in html
    assert "band-transition hand-overs" in html
    # The report validates the command record against the law before
    # rendering; the fixed-duty gate-drive section must be absent.
    assert "modulation-cycles.jsonl" in html
    payload = json.loads(Path(artifacts.report_json).read_text(encoding="utf-8"))
    assert payload["npc_modulation"]["block"]["levels"] == 3
    assert payload["gate_drive"] is None
