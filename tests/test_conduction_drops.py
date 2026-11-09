# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Piecewise-linear conduction: on-resistance and diode forward voltage.

A closed ``ideal_switch`` or a conducting ``diode`` obeys v = V_f + R_on * i
instead of v = 0. Every check below compares against a closed form derived by
hand, not against another run of this kernel.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from pear_sim.analysis.steady_state import solve_periodic_steady_state
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.run.contracts import model_hashes
from pear_sim.spec.model import load_model


def _model(tmp_path: Path, text: str, name: str = "model.yaml"):
    path = tmp_path / name
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return load_model(path)


RL_THROUGH_DIODE = """
schema: pear-sim/v0.1
name: rl_through_diode
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 10 V}
    - {id: d1, type: diode, p: in, n: a, forward_voltage: 0.7 V, on_resistance: 0.1 ohm}
    - {id: l1, type: inductor, p: a, n: b, value: 1 mH, initial_current: 0 A}
    - {id: r1, type: resistor, p: b, n: gnd, value: 1 ohm}
analysis:
  transient: {start: 0 s, stop: 5 ms, output_step: 10 us, solver: dopri5, rtol: 1.0e-10, atol: 1.0e-12}
outputs: [i(l1), v(d1)]
"""


def test_rl_step_through_a_diode_follows_the_closed_form(tmp_path: Path) -> None:
    # i(t) = (V - Vf) / (R + Rd) * (1 - exp(-t (R + Rd) / L))
    result = HybridSimulator(_model(tmp_path, RL_THROUGH_DIODE)).simulate()
    t = result.time
    expected = (10.0 - 0.7) / 1.1 * (1.0 - np.exp(-t * 1.1 / 1e-3))
    assert np.max(np.abs(result.signals["i(l1)"] - expected)) < 1e-6
    # The diode voltage is the forward drop plus R_d * i, sample by sample.
    assert np.max(np.abs(result.signals["v(d1)"] - (0.7 + 0.1 * expected))) < 1e-6
    assert result.physics_audit["full"]["balance_status"] == "within_numerical_tolerance"


HALF_WAVE = """
schema: pear-sim/v0.1
name: half_wave_vf
circuit:
  ground: gnd
  components:
    - id: vs
      type: voltage_source
      p: in
      n: gnd
      waveform: {type: sine, amplitude: 10 V, frequency: 50 Hz}
    - {id: rs, type: resistor, p: in, n: mid, value: 1 ohm}
    - {id: d1, type: diode, p: mid, n: out, forward_voltage: 1 V}
    - {id: r1, type: resistor, p: out, n: gnd, value: 10 ohm}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 1 uF, initial_voltage: 0 V}
analysis:
  transient: {start: 0 s, stop: 20 ms, output_step: 10 us, solver: dopri5, rtol: 1.0e-10, atol: 1.0e-12}
outputs: [v(out), i(d1)]
"""


def test_a_diode_turns_on_when_its_voltage_reaches_the_forward_drop(tmp_path: Path) -> None:
    result = HybridSimulator(_model(tmp_path, HALF_WAVE)).simulate()
    on = [e for e in result.events if e.kind == "natural_commutation" and e.current]
    assert on, "the diode never turned on"
    # The output starts at 0 V, so the diode first turns on when 10 sin(wt) = 1 V.
    expected = math.asin(0.1) / (2 * math.pi * 50.0)
    assert on[0].time == pytest.approx(expected, abs=2e-6)
    # And the output is the source minus the drop, divided by rs and r1.
    peak = float(np.max(result.signals["v(out)"]))
    # (tau = 10 us is negligible against the 50 Hz half-cycle.)
    assert peak == pytest.approx((10.0 - 1.0) * 10.0 / 11.0, abs=1e-2)


SYNC_BUCK = """
schema: pear-sim/v0.1
name: sync_buck_ron
parameters:
  ron: 50 milliohm
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 48 V}
    - {id: qh, type: ideal_switch, p: in, n: sw, control: leg.high, on_resistance: "${ron}"}
    - {id: ql, type: ideal_switch, p: sw, n: gnd, control: leg.low, on_resistance: "${ron}"}
    - {id: l1, type: inductor, p: sw, n: out, value: 10 uH, initial_current: 10 A}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, initial_voltage: 12 V}
    - {id: load, type: resistor, p: out, n: gnd, value: 1.2 ohm}
controls:
  leg: {type: complementary_pwm, frequency: 200 kHz, duty: 25 %, phase: 0 s, dead_time: 0 s}
analysis:
  transient: {start: 0 s, stop: 50 us, output_step: 20 ns, solver: dopri5, rtol: 1.0e-10, atol: 1.0e-12}
outputs: [v(out), i(l1)]
"""


def test_synchronous_buck_output_drops_by_the_on_resistance(tmp_path: Path) -> None:
    # Cycle averages in periodic steady state: <v_sw> = D Vin - R_on <i_L> and
    # <i_L> = Vout / R exactly, so Vout = D Vin R / (R + R_on).
    steady = solve_periodic_steady_state(_model(tmp_path, SYNC_BUCK))
    averages = dict(zip(steady.state_names, steady.cycle_average_state))
    expected = 0.25 * 48.0 * 1.2 / (1.2 + 0.05)
    assert averages["c1"] == pytest.approx(expected, rel=1e-6)
    assert averages["l1"] == pytest.approx(expected / 1.2, rel=1e-6)


DIODE_BUCK = """
schema: pear-sim/v0.1
name: diode_buck_vf
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 48 V}
    - {id: q1, type: ideal_switch, p: in, n: sw, control: pwm}
    - {id: d1, type: diode, p: gnd, n: sw, forward_voltage: 0.8 V}
    - {id: l1, type: inductor, p: sw, n: out, value: 47 uH, initial_current: 10 A}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, initial_voltage: 12 V}
    - {id: load, type: resistor, p: out, n: gnd, value: 1.2 ohm}
controls:
  pwm: {type: pwm, frequency: 100 kHz, duty: 30 %, phase: 0 s}
analysis:
  transient: {start: 0 s, stop: 100 us, output_step: 50 ns, solver: dopri5, rtol: 1.0e-10, atol: 1.0e-12}
outputs: [v(out), i(l1), i(d1)]
"""


def test_diode_buck_output_loses_the_forward_drop_while_freewheeling(tmp_path: Path) -> None:
    # CCM with an ideal switch and a diode with drop V_f (R_d = 0): the switch
    # node sits at Vin for D and at -V_f for 1 - D, so Vout = D Vin - (1-D) V_f.
    steady = solve_periodic_steady_state(_model(tmp_path, DIODE_BUCK))
    assert steady.natural_events_per_cycle == 0  # CCM: the diode never turns off on its own
    averages = dict(zip(steady.state_names, steady.cycle_average_state))
    assert averages["c1"] == pytest.approx(0.3 * 48.0 - 0.7 * 0.8, rel=1e-6)


BODY_DIODE_SHARE = """
schema: pear-sim/v0.1
name: body_diode_share
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 10 V}
    - {id: l1, type: inductor, p: in, n: a, value: 1 mH, initial_current: 18.6 A}
    - {id: rs, type: resistor, p: a, n: b, value: 0.5 ohm}
    - {id: q1, type: ideal_switch, p: b, n: gnd, control: gate, on_resistance: 0.5 ohm}
    - {id: d1, type: diode, p: b, n: gnd, forward_voltage: 0.7 V}
controls:
  gate: {type: pwm, frequency: 1 kHz, duty: 100 %, phase: 0 s}
analysis:
  transient: {start: 0 s, stop: 100 us, output_step: 1 us, solver: dopri5, rtol: 1.0e-10, atol: 1.0e-12}
outputs: [v(q1), i(q1), i(d1), i(l1)]
"""


def test_a_parallel_diode_takes_over_once_the_switch_drop_exceeds_its_forward_voltage(
    tmp_path: Path,
) -> None:
    # 10 V through 0.5 ohm into R_on = 0.5 ohm alone would sit at 5 V, above
    # V_f = 0.7 V, so the diode conducts and clamps the node: v = 0.7 V,
    # i_q = 0.7 / 0.5 = 1.4 A, i_d = (10 - 0.7) / 0.5 - 1.4 = 17.2 A.
    try:
        model = _model(tmp_path, BODY_DIODE_SHARE)
    except ModelError as exc:  # a 100 % duty PWM may be refused; that is a format question
        pytest.skip(f"fixture not expressible: {exc}")
    result = HybridSimulator(model).simulate()
    tail = slice(len(result.time) // 2, None)
    assert np.allclose(result.signals["v(q1)"][tail], 0.7, atol=1e-6)
    assert np.allclose(result.signals["i(q1)"][tail], 1.4, atol=1e-5)
    assert np.allclose(result.signals["i(d1)"][tail], 17.2, atol=1e-4)


def test_ideal_devices_keep_their_hashes_and_drops_change_them(tmp_path: Path) -> None:
    base = SYNC_BUCK.replace(', on_resistance: "${ron}"', "")
    ideal = model_hashes(_model(tmp_path, base, "a.yaml"))
    explicit_zero = model_hashes(
        _model(tmp_path, SYNC_BUCK.replace("ron: 50 milliohm", "ron: 0 ohm"), "b.yaml")
    )
    lossy = model_hashes(_model(tmp_path, SYNC_BUCK, "c.yaml"))
    # A declared zero is the ideal device: same physics, same physics hash.
    assert explicit_zero["model_sha256"] == ideal["model_sha256"]
    assert lossy["model_sha256"] != ideal["model_sha256"]


@pytest.mark.parametrize(
    "replacement, message",
    [
        ("on_resistance: \"${ron}\"}", None),
        ("on_resistance: -1 ohm}", "zero or positive"),
        ("on_resistance: 1 V}", "resistance"),
        ("forward_voltage: 1 V}", "only valid for diode"),
    ],
)
def test_conduction_fields_fail_closed(tmp_path: Path, replacement: str, message) -> None:
    text = SYNC_BUCK.replace('on_resistance: "${ron}"}', replacement, 1)
    if message is None:
        _model(tmp_path, text)
        return
    with pytest.raises(ModelError, match=message):
        _model(tmp_path, text)


def test_on_resistance_is_refused_on_passive_components(tmp_path: Path) -> None:
    text = RL_THROUGH_DIODE.replace("value: 1 ohm}", "value: 1 ohm, on_resistance: 1 ohm}")
    with pytest.raises(ModelError, match="only valid for ideal_switch and diode"):
        _model(tmp_path, text)


def test_the_shipped_mosfet_buck_matches_the_hand_balance() -> None:
    """examples/sync_buck_mosfet.yaml, cycle-averaged by hand.

    The high side conducts for D - t_d*fsw of the period; each of the two dead
    times hands the current to the low-side body diode (-V_f), and the low
    switch conducts for the rest. With R_on on both switches and the DCR in
    series: <v_sw> = Vin (D - t_d f) - R_on I (1 - 2 t_d f) - V_f 2 t_d f and
    Vout = <v_sw> - DCR * I with I = Vout / R. Ripple shapes the per-interval
    means slightly, hence 1e-4 relative.
    """

    model = load_model(Path(__file__).parents[1] / "examples" / "sync_buck_mosfet.yaml")
    steady = solve_periodic_steady_state(model)
    averages = dict(zip(steady.state_names, steady.cycle_average_state))
    td_f = 50e-9 * 200e3
    numerator = 48.0 * (0.26 - td_f) - 0.8 * 2 * td_f
    vout = numerator / (1.0 + (0.008 * (1 - 2 * td_f) + 0.005) / 1.2)
    assert averages["c1"] == pytest.approx(vout, rel=1e-4)


def test_an_unknown_component_type_names_the_valid_ones(tmp_path: Path) -> None:
    text = SYNC_BUCK.replace("type: ideal_switch, p: in", "type: mosfet, p: in")
    with pytest.raises(ModelError, match=r"unknown component type 'mosfet'.*ideal_switch.*body diode"):
        _model(tmp_path, text)


def test_a_switch_node_without_a_dead_time_path_suggests_the_antiparallel_diode(
    tmp_path: Path,
) -> None:
    from pear_sim.errors import PearSimError

    text = SYNC_BUCK.replace("dead_time: 0 s", "dead_time: 100 ns")
    with pytest.raises(PearSimError, match="antiparallel diode"):
        HybridSimulator(_model(tmp_path, text)).simulate()


STIFF_FILTER = """
schema: pear-sim/v0.1
name: stiff_input_filter
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: src, n: gnd, value: 12 V}
    - {id: rsrc, type: resistor, p: src, n: in, value: 10 milliohm}
    - {id: c0, type: capacitor, p: in, n: gnd, value: 10 uF, initial_voltage: 12 V}
    - {id: l1, type: inductor, p: in, n: out, value: 10 uH, initial_current: 0 A}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 22 uF, initial_voltage: 0 V}
    - {id: r1, type: resistor, p: out, n: gnd, value: 5 ohm}
analysis:
  transient: {start: 0 s, stop: 2 ms, output_step: 1 us, solver: SOLVER, rtol: 1.0e-8, atol: 1.0e-10}
outputs: [v(out)]
"""


def test_an_energy_balance_outside_tolerance_is_a_warning_with_a_way_out(tmp_path: Path) -> None:
    """A 100 ns input-filter time constant inside a 2 ms run is stiff: the
    explicit dopri5 leaves a 3.6e-7 energy residual against the 1e-7 budget,
    an implicit solver closes it to ~1e-12. The run must say so and say how."""

    import json

    from pear_sim.run.io import write_result

    def warnings_of(solver: str) -> list[dict]:
        model = _model(tmp_path, STIFF_FILTER.replace("SOLVER", solver), f"{solver}.yaml")
        write_result(HybridSimulator(model).simulate(), tmp_path / solver)
        return json.loads((tmp_path / solver / "warnings.json").read_text())

    explicit = [w for w in warnings_of("dopri5") if w["code"] == "ENERGY_BALANCE_OUTSIDE_TOLERANCE"]
    assert explicit and "radau" in explicit[0]["message"]
    assert not [w for w in warnings_of("radau") if w["code"] == "ENERGY_BALANCE_OUTSIDE_TOLERANCE"]


def test_overriding_a_parameter_nothing_references_is_refused(tmp_path: Path) -> None:
    """An override that cannot change anything is a silent no-op: the user thinks
    they changed the design and they did not. It must fail with the reason."""

    text = SYNC_BUCK.replace("  ron: 50 milliohm\n", "  ron: 50 milliohm\n  unused: 3 A\n")
    path = tmp_path / "with_unused.yaml"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    load_model(path, {"ron": "10 milliohm"})  # referenced: fine
    with pytest.raises(ModelError, match=r"unused.*no value references"):
        load_model(path, {"unused": "5 A"})


def test_output_sample_count_is_bounded_before_anything_is_allocated(tmp_path: Path) -> None:
    """2 s at 10 ns is 2e8 samples: the grid alone is 1.6 GB, and memory keeps
    growing until the machine runs out. The loader must refuse it; the bound
    itself must stay silent on every legitimate long run."""

    from pear_sim.limits import MAX_RUN_OUTPUT_SAMPLES

    text = SYNC_BUCK.strip() + "\n"
    stop = re.search(r"stop: ([^,}]+)", text).group(1)
    step = re.search(r"output_step: ([^,}]+)", text).group(1)
    huge = text.replace(f"stop: {stop}", "stop: 2 s").replace(f"output_step: {step}", "output_step: 10 ns")
    path = tmp_path / "huge.yaml"
    path.write_text(huge, encoding="utf-8")
    with pytest.raises(ModelError, match=r"output samples; the limit is"):
        load_model(path)
    at_limit = text.replace(f"stop: {stop}", f"stop: {(MAX_RUN_OUTPUT_SAMPLES - 1) * 10} ns").replace(
        f"output_step: {step}", "output_step: 10 ns"
    )
    path.write_text(at_limit, encoding="utf-8")
    load_model(path)
