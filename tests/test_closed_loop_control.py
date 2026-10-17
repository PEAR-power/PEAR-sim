"""Sampled PI closed-loop PWM: loader, determinism, and the acceptance gate.

The acceptance-gate scenario is a closed-loop buck regulating v(out) through a
load step, with a deterministic event order (controller samples strictly
before any gate edge at the same timestamp). The PI gains in the example
model were selected with the averaged loop-gain analysis; the cross-check test
asserts the averaged model actually predicts those margins.
"""

from pathlib import Path

import numpy as np
import pytest

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.controls import SampledPiPwmControl
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]

CLOSED_LOOP_BODY = """
schema: pear-sim/v0.1
name: buck_closed_loop

parameters:
  vin: 24 V
  fsw: 50 kHz
  vref: 12 V

circuit:
  ground: gnd
  components:
    - {{id: vin, type: voltage_source, p: in, n: gnd, value: "${{vin}}"}}
    - {{id: q1, type: ideal_switch, p: in, n: sw, control: ctrl, initial: on}}
    - {{id: d1, type: diode, p: gnd, n: sw, initial: off}}
    - {{id: l1, type: inductor, p: sw, n: out, value: 47 uH, initial_current: 1.2 A}}
    - {{id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, initial_voltage: 12 V}}
    - {{id: load, type: resistor, p: out, n: gnd, value: 10 ohm}}
    - id: istep
      type: current_source
      p: out
      n: gnd
      waveform: {{type: step, initial: 0 A, final: 1 A, at: {step_at}}}

controls:
  ctrl:
    type: sampled_pi_pwm
    frequency: "${{fsw}}"
    sense: {sense}
    reference: "${{vref}}"
    kp: 0.002
    ki: 30
    duty_min: 5 %
    duty_max: 95 %

analysis:
  transient:
    start: 0 s
    stop: {stop}
    output_step: 2 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11

outputs:
  - v(out)
  - i(l1)
"""

# Open-loop equivalent of the closed-loop plant at the post-step operating
# point (2.2 A load keeps the cycle in CCM), shared by the loop-gain and
# sampled-loop-gain cross-check tests.
OPEN_LOOP_PLANT_BODY = """
schema: pear-sim/v0.1
name: buck_ol
parameters: {vin: 24 V, fsw: 50 kHz}
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: "${vin}"}
    - {id: q1, type: ideal_switch, p: in, n: sw, control: pwm, initial: on}
    - {id: d1, type: diode, p: gnd, n: sw, initial: off}
    - {id: l1, type: inductor, p: sw, n: out, value: 47 uH, initial_current: 2.2 A}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, initial_voltage: 12 V}
    - {id: load, type: resistor, p: out, n: gnd, value: 5.45 ohm}
controls:
  pwm: {type: pwm, frequency: "${fsw}", duty: 50 %, phase: 0 s}
analysis:
  transient: {start: 0 s, stop: 2 ms, output_step: 1 us, solver: dopri5, rtol: 1.0e-10, atol: 1.0e-12}
outputs: [v(out), i(l1)]
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "model.yaml"
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loader_parses_sampled_pi_control(tmp_path: Path) -> None:
    model = load_model(
        _write(
            tmp_path,
            CLOSED_LOOP_BODY.format(sense="v(out)", step_at="25 ms", stop="1 ms"),
        )
    )
    control = model.sampled_controls["ctrl"]
    assert isinstance(control, SampledPiPwmControl)
    assert control.frequency == pytest.approx(50_000.0)
    assert control.reference == pytest.approx(12.0)
    assert control.duty_min == pytest.approx(0.05)
    assert control.duty_max == pytest.approx(0.95)
    assert "ctrl" not in model.controls


def test_loader_requires_sense_among_outputs(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="sense"):
        load_model(
            _write(
                tmp_path,
                CLOSED_LOOP_BODY.format(sense="v(sw)", step_at="25 ms", stop="1 ms"),
            )
        )


def test_loader_rejects_degenerate_duty_limits(tmp_path: Path) -> None:
    body = CLOSED_LOOP_BODY.format(sense="v(out)", step_at="25 ms", stop="1 ms")
    body = body.replace("duty_max: 95 %", "duty_max: 100 %")
    with pytest.raises(ModelError, match="duty"):
        load_model(_write(tmp_path, body))


# ---------------------------------------------------------------------------
# Acceptance gate: closed-loop buck load step
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gate_result():
    model = load_model(ROOT / "examples" / "buck_closed_loop.yaml")
    return HybridSimulator(model).simulate()


def test_m3_gate_regulates_before_the_step(gate_result) -> None:
    t, v = gate_result.time, gate_result.signals["v(out)"]
    window = v[(t > 20e-3) & (t < 25e-3)]
    assert np.mean(window) == pytest.approx(12.0, abs=0.05)
    assert np.max(window) - np.min(window) < 0.15


def test_m3_gate_recovers_from_the_load_step(gate_result) -> None:
    t, v = gate_result.time, gate_result.signals["v(out)"]
    dip = v[(t > 25e-3) & (t < 30e-3)]
    assert float(np.min(dip)) > 10.5  # bounded transient
    recovered = t[(t > 26e-3) & (np.abs(v - 12.0) < 0.12)]
    assert len(recovered) and float(recovered[0]) < 30e-3  # within 5 ms
    tail = v[t > 40e-3]
    assert np.mean(tail) == pytest.approx(12.0, abs=0.05)
    assert np.max(tail) - np.min(tail) < 0.5


def test_m3_gate_event_order_is_deterministic(gate_result) -> None:
    trace = gate_result.control_trace
    # One update per cycle, contiguous cycle indices from 0.
    assert [entry["cycle_index"] for entry in trace] == list(range(len(trace)))
    assert len(trace) == 2250  # 45 ms * 50 kHz
    assert all(
        0.05 <= entry["duty_command"] <= 0.95 for entry in trace
    )
    # The controller fires strictly before the gate edges at the same time:
    # every cycle boundary with a scheduled_gate ON edge has a trace entry at
    # exactly that timestamp.
    on_edges = {
        event.time
        for event in gate_result.events
        if event.kind == "scheduled_gate" and event.target == "ctrl" and event.current
    }
    trace_times = {entry["time_s"] for entry in trace}
    assert on_edges <= trace_times
    assert gate_result.stats["sampled_control_updates"] == len(trace)


def test_closed_loop_simulation_is_bit_deterministic(tmp_path: Path) -> None:
    body = CLOSED_LOOP_BODY.format(sense="v(out)", step_at="2 ms", stop="4 ms")
    first = HybridSimulator(load_model(_write(tmp_path, body))).simulate()
    second = HybridSimulator(load_model(_write(tmp_path, body))).simulate()
    assert np.array_equal(first.signals["v(out)"], second.signals["v(out)"])
    assert np.array_equal(first.signals["i(l1)"], second.signals["i(l1)"])
    assert first.control_trace == second.control_trace


def test_anti_windup_recovers_from_low_saturation(tmp_path: Path) -> None:
    """Regression: conditional integration must integrate OUT of a limit.

    Start at the reference (zero error -> duty clamps to duty_min), let the
    output collapse, and require the integrator to climb back out. A naive
    'freeze while saturated' anti-windup deadlocks at duty_min forever.
    """

    body = CLOSED_LOOP_BODY.format(sense="v(out)", step_at="90 ms", stop="20 ms")
    result = HybridSimulator(load_model(_write(tmp_path, body))).simulate()
    t, v = result.time, result.signals["v(out)"]
    tail = v[t > 15e-3]
    assert np.mean(tail) == pytest.approx(12.0, abs=0.3)
    duties = [entry["duty_command"] for entry in result.control_trace]
    assert min(duties) == pytest.approx(0.05)  # it did saturate low
    assert duties[-1] > 0.4  # and climbed back out


# ---------------------------------------------------------------------------
# Loop-gain cross-check: the shipped gains are margin-backed
# ---------------------------------------------------------------------------


def test_p8_margins_back_the_shipped_gains(tmp_path: Path) -> None:
    """The averaged model predicts the gains used by the acceptance-gate scenario.

    Same plant at the post-step operating point (2.2 A load), open loop for
    the shooting analysis. kp=0.002/ki=30 must show a healthy margin; the
    kp=0.08/ki=150 pair must show a near-zero margin at ~4 kHz — that pair
    produced an actual ~4 kHz limit cycle in the switching kernel while this
    feature was developed.
    """

    from pear_sim.analysis.small_signal import averaged_small_signal, loop_gain
    from pear_sim.analysis.steady_state import solve_periodic_steady_state

    open_loop = _write(tmp_path, OPEN_LOOP_PLANT_BODY)
    model = load_model(open_loop)
    steady = solve_periodic_steady_state(model)
    assert steady.converged
    signal = averaged_small_signal(model, steady, "v(out)")

    healthy = loop_gain(
        signal,
        compensator_numerator=(0.002, 30.0),
        compensator_denominator=(1.0, 0.0),
        sensor_gain=1.0,
        f_min_hz=5.0,
        f_max_hz=5_000.0,
        points=600,
    )
    assert healthy.phase_margin_deg is not None
    assert healthy.phase_margin_deg > 80.0
    assert healthy.gain_margin_db is not None
    assert healthy.gain_margin_db > 6.0

    marginal = loop_gain(
        signal,
        compensator_numerator=(0.08, 150.0),
        compensator_denominator=(1.0, 0.0),
        sensor_gain=1.0,
        f_min_hz=5.0,
        f_max_hz=5_000.0,
        points=600,
    )
    assert marginal.phase_margin_deg is not None
    assert abs(marginal.phase_margin_deg) < 10.0
    assert marginal.gain_crossover_hz == pytest.approx(4_000.0, rel=0.15)


def test_p32_discrete_aware_margins_call_the_limit_cycle(tmp_path: Path) -> None:
    """ZOH-aware margins must agree with the switching kernel on WHICH
    design is unstable, not only on the stable one.

    The continuous averaged analysis reports PM ~ +2 deg for kp=0.08/ki=150 —
    "stable" — while the switching kernel reproduces a ~4 kHz limit cycle for
    that pair. Folding the sampled controller's ZOH at f_sw into L(s) adds
    the 180 deg * f/f_sw lag the continuous model cannot see (~14 deg at the
    ~4 kHz crossover), turning the verdict decisively unstable. The healthy
    kp=0.002/ki=30 pair crosses over at ~115 Hz where the same ZOH costs
    under half a degree, so its margins survive.
    """

    from pear_sim.analysis.small_signal import (
        SamplingModel,
        averaged_small_signal,
        loop_gain,
    )
    from pear_sim.analysis.steady_state import solve_periodic_steady_state

    model = load_model(_write(tmp_path, OPEN_LOOP_PLANT_BODY))
    steady = solve_periodic_steady_state(model)
    assert steady.converged
    signal = averaged_small_signal(model, steady, "v(out)")
    sampling = SamplingModel(update_frequency_hz=signal.switching_frequency_hz)

    def margins(kp: float, ki: float, declared: SamplingModel | None):
        return loop_gain(
            signal,
            compensator_numerator=(kp, ki),
            compensator_denominator=(1.0, 0.0),
            sensor_gain=1.0,
            f_min_hz=5.0,
            f_max_hz=5_000.0,
            points=600,
            sampling=declared,
        )

    healthy = margins(0.002, 30.0, sampling)
    assert healthy.phase_margin_deg is not None
    assert healthy.phase_margin_deg > 80.0
    assert healthy.gain_margin_db is not None
    assert healthy.gain_margin_db > 6.0

    hot_continuous = margins(0.08, 150.0, None)
    hot_discrete = margins(0.08, 150.0, sampling)
    # Continuous: marginally positive — overstated precisely in the digital
    # regime. Discrete-aware: decisively negative, matching the kernel.
    assert hot_continuous.phase_margin_deg is not None
    assert hot_continuous.phase_margin_deg > 0.0
    assert hot_discrete.phase_margin_deg is not None
    assert hot_discrete.phase_margin_deg < -5.0
    assert hot_discrete.gain_margin_db is not None
    assert hot_discrete.gain_margin_db < 0.0
