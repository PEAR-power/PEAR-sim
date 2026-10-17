"""Waveform sources: loader, kernel, and fail-closed boundaries.

Numerical assertions compare the production solver against independent closed
forms (first-order RC responses). Tolerances follow the requested integrator
tolerances, never exact floats.
"""

from pathlib import Path

import numpy as np
import pytest

from pear_sim.analysis.small_signal import SmallSignalError, averaged_small_signal
from pear_sim.analysis.steady_state import SteadyStateError, solve_periodic_steady_state
from pear_sim.core.circuit import Circuit
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.run.contracts import physics_model_dict
from pear_sim.spec.model import load_model
from pear_sim.spec.waveforms import (
    PulseWaveform,
    PwlWaveform,
    SineWaveform,
    StepWaveform,
)

ROOT = Path(__file__).parents[1]


def _write_model(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def _rc_model(tmp_path: Path, waveform_yaml: str, *, stop: str = "5 ms") -> Path:
    return _write_model(
        tmp_path,
        f"""
schema: pear-sim/v0.1
name: rc_waveform

circuit:
  ground: gnd
  components:
    - id: vin
      type: voltage_source
      p: in
      n: gnd
      waveform: {waveform_yaml}
    - {{id: r1, type: resistor, p: in, n: out, value: 1 kohm}}
    - {{id: c1, type: capacitor, p: out, n: gnd, value: 100 nF, initial_voltage: 0 V}}

analysis:
  transient:
    start: 0 s
    stop: {stop}
    output_step: 5 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11

outputs:
  - v(out)
""",
    )


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loader_parses_sine_waveform_with_units(tmp_path: Path) -> None:
    model = load_model(
        _rc_model(tmp_path, "{type: sine, amplitude: 5 V, frequency: 1 kHz, phase_deg: 90}")
    )
    waveform = model.components[0].waveform
    assert isinstance(waveform, SineWaveform)
    assert waveform.amplitude == pytest.approx(5.0)
    assert waveform.frequency_hz == pytest.approx(1000.0)
    assert waveform.phase_rad == pytest.approx(np.pi / 2.0)
    assert waveform.offset == 0.0


def test_loader_rejects_waveform_on_non_source(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        """
schema: pear-sim/v0.1
name: bad

circuit:
  ground: gnd
  components:
    - id: r1
      type: resistor
      p: in
      n: gnd
      value: 1 kohm
      waveform: {type: step, initial: 1 ohm, final: 2 ohm, at: 1 ms}
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 5 V}
    - {id: c1, type: capacitor, p: in, n: gnd, value: 1 uF}

analysis:
  transient:
    start: 0 s
    stop: 1 ms
    output_step: 1 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11

outputs:
  - v(in)
""",
    )
    with pytest.raises(ModelError):
        load_model(path)


def test_loader_rejects_value_and_waveform_together(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        """
schema: pear-sim/v0.1
name: bad

circuit:
  ground: gnd
  components:
    - id: vin
      type: voltage_source
      p: in
      n: gnd
      value: 5 V
      waveform: {type: sine, amplitude: 5 V, frequency: 1 kHz}
    - {id: r1, type: resistor, p: in, n: gnd, value: 1 kohm}
    - {id: c1, type: capacitor, p: in, n: gnd, value: 1 uF}

analysis:
  transient:
    start: 0 s
    stop: 1 ms
    output_step: 1 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11

outputs:
  - v(in)
""",
    )
    with pytest.raises(ModelError, match="mutually exclusive"):
        load_model(path)


def test_loader_rejects_non_increasing_pwl_points(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="strictly increasing"):
        load_model(
            _rc_model(tmp_path, "{type: pwl, points: [[0 s, 0 V], [1 ms, 5 V], [1 ms, 0 V]]}")
        )


def test_loader_rejects_pulse_duty_out_of_range(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="duty"):
        load_model(
            _rc_model(
                tmp_path,
                "{type: pulse, low: 0 V, high: 5 V, frequency: 1 kHz, duty: 100 %}",
            )
        )


# ---------------------------------------------------------------------------
# Kernel vs closed forms
# ---------------------------------------------------------------------------


def test_rc_sine_matches_closed_form(tmp_path: Path) -> None:
    model = load_model(_rc_model(tmp_path, "{type: sine, amplitude: 5 V, frequency: 1 kHz}"))
    result = HybridSimulator(model).simulate()
    t = result.time
    tau = 1_000.0 * 100e-9
    omega = 2.0 * np.pi * 1_000.0
    gain = 1.0 / (1.0 + 1j * omega * tau)
    magnitude, phase = np.abs(gain), np.angle(gain)
    reference = 5.0 * magnitude * (
        np.sin(omega * t + phase) - np.sin(phase) * np.exp(-t / tau)
    )
    assert np.max(np.abs(result.signals["v(out)"] - reference)) < 5e-8
    assert result.physics_audit["full"]["balance_status"] == "within_numerical_tolerance"


def test_rc_step_matches_closed_form_and_is_not_smeared(tmp_path: Path) -> None:
    model = load_model(
        _rc_model(tmp_path, "{type: step, initial: 0 V, final: 5 V, at: 2 ms}", stop="4 ms")
    )
    result = HybridSimulator(model).simulate()
    t = result.time
    tau = 1_000.0 * 100e-9
    reference = np.where(t < 2e-3, 0.0, 5.0 * (1.0 - np.exp(-(t - 2e-3) / tau)))
    assert np.max(np.abs(result.signals["v(out)"] - reference)) < 5e-8
    # The breakpoint restarts integration exactly at the discontinuity: the
    # sample just before the step is still fully settled at the initial level.
    before = result.signals["v(out)"][t < 2e-3 - 1e-9]
    assert abs(float(before[-1])) < 1e-9


def test_rc_pwl_ramp_matches_closed_form(tmp_path: Path) -> None:
    # Ramp 0->5 V over [1, 3] ms; slope k drives v_c = k*(t' - tau*(1-exp(-t'/tau))).
    model = load_model(
        _rc_model(
            tmp_path,
            "{type: pwl, points: [[0 s, 0 V], [1 ms, 0 V], [3 ms, 5 V], [5 ms, 5 V]]}",
        )
    )
    result = HybridSimulator(model).simulate()
    t = result.time
    tau = 1_000.0 * 100e-9
    slope = 5.0 / 2e-3
    ramp_time = np.clip(t - 1e-3, 0.0, None)
    on_ramp = slope * (ramp_time - tau * (1.0 - np.exp(-ramp_time / tau)))
    mask = t <= 3e-3
    assert np.max(np.abs(result.signals["v(out)"][mask] - on_ramp[mask])) < 5e-8


def test_pulse_breakpoints_restart_each_edge(tmp_path: Path) -> None:
    model = load_model(
        _rc_model(
            tmp_path,
            "{type: pulse, low: 0 V, high: 5 V, frequency: 1 kHz, duty: 25 %}",
            stop="3 ms",
        )
    )
    result = HybridSimulator(model).simulate()
    v = result.signals["v(out)"]
    t = result.time
    tau = 1_000.0 * 100e-9
    # First cycle closed form: charge from 0 during the 0.25 ms high phase,
    # then discharge during the low phase.
    (high_index,) = np.nonzero(np.abs(t - 0.24e-3) < 1e-9)
    (low_index,) = np.nonzero(np.abs(t - 0.99e-3) < 1e-9)
    v_high = 5.0 * (1.0 - np.exp(-0.24e-3 / tau))
    v_peak = 5.0 * (1.0 - np.exp(-0.25e-3 / tau))
    v_low = v_peak * np.exp(-(0.99e-3 - 0.25e-3) / tau)
    assert float(v[high_index[0]]) == pytest.approx(v_high, abs=5e-7)
    assert float(v[low_index[0]]) == pytest.approx(v_low, abs=5e-7)


def test_half_wave_rectifier_commutates_naturally(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        """
schema: pear-sim/v0.1
name: halfwave_rectifier

circuit:
  ground: gnd
  components:
    - id: vin
      type: voltage_source
      p: ac
      n: gnd
      waveform: {type: sine, amplitude: 10 V, frequency: 50 Hz}
    - {id: rs, type: resistor, p: ac, n: mid, value: 1 ohm}
    - {id: d1, type: diode, p: mid, n: out, initial: off}
    - {id: rload, type: resistor, p: out, n: gnd, value: 100 ohm}
    - {id: cfilt, type: capacitor, p: out, n: gnd, value: 470 uF, initial_voltage: 0 V}

analysis:
  transient:
    start: 0 s
    stop: 100 ms
    output_step: 50 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11

outputs:
  - v(out)
  - i(d1)
""",
    )
    result = HybridSimulator(load_model(path)).simulate()
    v = result.signals["v(out)"]
    commutations = [e for e in result.events if e.kind == "natural_commutation"]
    # One on/off pair per 20 ms mains cycle over 5 cycles.
    assert len(commutations) == 10
    tail = v[len(v) // 2 :]
    assert float(np.max(tail)) < 10.0
    assert float(np.max(tail)) > 9.0
    assert float(np.min(tail)) > 6.0
    assert result.physics_audit["full"]["balance_status"] == "within_numerical_tolerance"
    # The diode never conducts backwards.
    assert float(np.min(result.signals["i(d1)"])) > -1e-9


def test_waveform_current_source_matches_closed_form(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        """
schema: pear-sim/v0.1
name: rc_current_sine

circuit:
  ground: gnd
  components:
    - id: iin
      type: current_source
      p: gnd
      n: out
      waveform: {type: sine, amplitude: 1 mA, frequency: 1 kHz}
    - {id: r1, type: resistor, p: out, n: gnd, value: 1 kohm}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 nF, initial_voltage: 0 V}

analysis:
  transient:
    start: 0 s
    stop: 5 ms
    output_step: 5 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11

outputs:
  - v(out)
  - i(iin)
""",
    )
    result = HybridSimulator(load_model(path)).simulate()
    t = result.time
    tau = 1_000.0 * 100e-9
    omega = 2.0 * np.pi * 1_000.0
    gain = 1_000.0 / (1.0 + 1j * omega * tau)
    magnitude, phase = np.abs(gain), np.angle(gain)
    reference = 1e-3 * magnitude * (
        np.sin(omega * t + phase) - np.sin(phase) * np.exp(-t / tau)
    )
    assert np.max(np.abs(result.signals["v(out)"] - reference)) < 5e-8
    assert np.max(
        np.abs(result.signals["i(iin)"] - 1e-3 * np.sin(omega * t))
    ) < 1e-12


# ---------------------------------------------------------------------------
# Contracts and fail-closed boundaries
# ---------------------------------------------------------------------------


def test_dc_model_physics_dict_has_no_waveform_key() -> None:
    model = load_model(ROOT / "examples" / "buck_dcm.yaml")
    payload = physics_model_dict(model)
    for component in payload["components"]:
        assert "waveform" not in component


def test_waveform_model_physics_dict_records_waveform(tmp_path: Path) -> None:
    model = load_model(_rc_model(tmp_path, "{type: sine, amplitude: 5 V, frequency: 1 kHz}"))
    payload = physics_model_dict(model)
    vin = next(c for c in payload["components"] if c["id"] == "vin")
    assert vin["waveform"]["type"] == "sine"
    assert vin["waveform"]["amplitude"] == pytest.approx(5.0)
    assert vin["dimension"] == "voltage"
    assert "value_si" not in vin


def test_algebraic_without_time_fails_closed_for_waveform_circuit(tmp_path: Path) -> None:
    model = load_model(_rc_model(tmp_path, "{type: sine, amplitude: 5 V, frequency: 1 kHz}"))
    circuit = Circuit(model)
    mode = circuit.compile({}, {})
    with pytest.raises(ModelError, match="time is required"):
        mode.algebraic(np.zeros(1))
    assert mode.algebraic(np.zeros(1), 0.0) is not None


def test_steady_state_fails_closed_for_waveform_sources(tmp_path: Path) -> None:
    path = _write_model(
        tmp_path,
        """
schema: pear-sim/v0.1
name: buck_waveform_input

parameters:
  fsw: 20 kHz

circuit:
  ground: gnd
  components:
    - id: vin
      type: voltage_source
      p: in
      n: gnd
      waveform: {type: sine, amplitude: 100 V, frequency: 50 Hz}
    - {id: q1, type: ideal_switch, p: in, n: sw, control: pwm, initial: on}
    - {id: d1, type: diode, p: gnd, n: sw, initial: off}
    - {id: l1, type: inductor, p: sw, n: out, value: 100 uH, initial_current: 0 A}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 10 uF, initial_voltage: 0 V}
    - {id: load, type: resistor, p: out, n: gnd, value: 10 ohm}

controls:
  pwm:
    type: pwm
    frequency: "${fsw}"
    duty: 40 %
    phase: 0 s

analysis:
  transient:
    start: 0 s
    stop: 1 ms
    output_step: 1 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11

outputs:
  - v(out)
""",
    )
    model = load_model(path)
    with pytest.raises(SteadyStateError, match="DC sources only"):
        solve_periodic_steady_state(model)


def test_small_signal_fails_closed_for_waveform_sources(tmp_path: Path) -> None:
    model = load_model(ROOT / "examples" / "buck_dcm.yaml")
    steady = solve_periodic_steady_state(model)

    from dataclasses import replace

    waveform_component = replace(
        model.components[0],
        value=None,
        waveform=StepWaveform(initial=100.0, final=100.0, at_s=1.0),
    )
    tainted = replace(
        model, components=(waveform_component,) + tuple(model.components[1:])
    )
    with pytest.raises(SmallSignalError, match="DC sources only"):
        averaged_small_signal(tainted, steady, "v(out)")


# ---------------------------------------------------------------------------
# Waveform primitives
# ---------------------------------------------------------------------------


def test_pulse_breakpoints_cover_window_edges() -> None:
    pulse = PulseWaveform(low=0.0, high=1.0, frequency_hz=1_000.0, duty=0.25, phase_s=0.0)
    points = pulse.breakpoints(0.0, 2.5e-3)
    assert points == pytest.approx((0.25e-3, 1e-3, 1.25e-3, 2e-3, 2.25e-3), abs=1e-15)
    assert pulse.value(0.0) == 1.0
    assert pulse.value(0.25e-3) == 0.0
    assert pulse.value(1e-3) == 1.0


def test_pwl_holds_endpoints_and_interpolates() -> None:
    pwl = PwlWaveform(points=((1.0, 0.0), (3.0, 10.0)))
    assert pwl.value(0.0) == 0.0
    assert pwl.value(2.0) == pytest.approx(5.0)
    assert pwl.value(4.0) == 10.0
    assert pwl.slope(2.0) == pytest.approx(5.0)
    assert pwl.slope(0.5) == 0.0


def test_sine_slope_is_derivative() -> None:
    sine = SineWaveform(offset=1.0, amplitude=2.0, frequency_hz=100.0, phase_rad=0.3)
    h = 1e-8
    for time in (0.0, 1.234e-3, 7.7e-3):
        numeric = (sine.value(time + h) - sine.value(time - h)) / (2.0 * h)
        assert sine.slope(time) == pytest.approx(numeric, rel=1e-6, abs=1e-4)
