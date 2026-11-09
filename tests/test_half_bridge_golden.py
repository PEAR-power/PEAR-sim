# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Half-bridge switching-node golden (VAL-005 row).

The last missing analytical golden. A half-bridge leg with complementary PWM
and a real dead time driving an RL load is the smallest circuit where the
kernel has to get three separate things right at once:

1. the gate schedule — an EXACT rational timeline, asserted against the closed
   form of the modulator law rather than against a recorded snapshot;
2. the switching node — during the dead time no switch conducts, so the pole
   is pinned by whichever body diode the load current forces (here the lower
   one, because the current leaves the pole throughout);
3. the state trajectory — piecewise exponential toward +-Vdc/2 over R with
   tau = L/R, which this test integrates INDEPENDENTLY interval by interval
   from the analytic schedule, never from the events the run produced.

Operating point chosen so the current never reverses (settled band 17.3..21.1
A around a 19.2 A mean): a zero crossing inside a dead-time window would hand
the interval to the opposite diode, which is real physics but a different
golden.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator

UDC_HALF = 200.0
FSW = 10_000.0
DUTY = 0.75
DEAD_TIME = 1e-6
RLOAD = 5.0
LLOAD = 2e-3
I0 = 20.0
STOP = 2000e-6  # 20 periods = 5 tau, so the steady relation below is admissible

PERIOD = 1.0 / FSW
TAU = LLOAD / RLOAD

# Declared tolerances. The schedule is exact rational arithmetic, so its
# budget is float representation only. The current budget covers dop853 at
# rtol 1e-10 accumulated across 60 segments; measured 3.6e-11 A on a ~19 A
# signal (relative 2e-12), with the budget kept well above it.
SCHEDULE_TOLERANCE_S = 1e-12
POLE_TOLERANCE_V = 1e-9
CURRENT_TOLERANCE_A = 1e-8

MODEL = """
schema: pear-sim/v0.1
name: half_bridge_golden
parameters:
  udc_half: 200 V
  fsw: 10 kHz
  duty: 75 %
  dead_time: 1 us
  rload: 5 ohm
  lload: 2 mH
circuit:
  ground: mid
  components:
    - {id: vp, type: voltage_source, p: dcp, n: mid, value: "${udc_half}"}
    - {id: vn, type: voltage_source, p: mid, n: dcn, value: "${udc_half}"}
    - {id: qh, type: ideal_switch, p: dcp, n: pole, control: leg.high}
    - {id: ql, type: ideal_switch, p: pole, n: dcn, control: leg.low}
    - {id: dh, type: diode, p: pole, n: dcp, initial: off}
    - {id: dl, type: diode, p: dcn, n: pole, initial: off}
    - {id: lo, type: inductor, p: pole, n: la, value: "${lload}", initial_current: 20 A}
    - {id: ro, type: resistor, p: la, n: mid, value: "${rload}"}
controls:
  leg:
    type: complementary_pwm
    frequency: "${fsw}"
    duty: "${duty}"
    phase: 0 s
    dead_time: "${dead_time}"
analysis:
  transient:
    start: 0 s
    stop: 2000 us
    output_step: 0.5 us
    solver: dop853
    rtol: 1.0e-10
    atol: 1.0e-12
outputs: [v(pole), i(lo)]
"""


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    path = tmp_path_factory.mktemp("half_bridge") / "model.yaml"
    path.write_text(MODEL.strip() + "\n", encoding="utf-8")
    model = load_model(path)
    return model, HybridSimulator(model).simulate()


def _analytic_schedule() -> dict[str, list[float]]:
    """The modulator law in closed form: high on k*T+dt .. k*T+D*T.

    Each turn-off is followed by a dead-time gap before the opposite switch
    closes, so the low switch owns k*T+D*T+dt .. (k+1)*T.
    """

    high_on, high_off, low_on, low_off = [], [], [], []
    periods = int(round(STOP / PERIOD))
    for k in range(periods):
        start = k * PERIOD
        high_on.append(start + DEAD_TIME)
        high_off.append(start + DUTY * PERIOD)
        low_on.append(start + DUTY * PERIOD + DEAD_TIME)
        low_off.append(start + PERIOD)
    return {
        "leg.high": sorted(high_on + high_off),
        "leg.low": sorted(low_on + low_off),
    }


def _analytic_pole_intervals() -> list[tuple[float, float, float]]:
    """(start, stop, pole voltage) from the schedule and the current sign.

    The pole is +Vdc/2 only while the high switch conducts. Everywhere else —
    low switch on AND both dead-time gaps — the load current still leaves the
    pole, so the lower body diode conducts and the pole sits at -Vdc/2.
    """

    intervals = []
    periods = int(round(STOP / PERIOD))
    for k in range(periods):
        start = k * PERIOD
        intervals.append((start, start + DEAD_TIME, -UDC_HALF))
        intervals.append((start + DEAD_TIME, start + DUTY * PERIOD, +UDC_HALF))
        intervals.append((start + DUTY * PERIOD, start + PERIOD, -UDC_HALF))
    return intervals


def _closed_form_current(times: np.ndarray) -> np.ndarray:
    """Piecewise exponential toward pole/R with tau = L/R, interval by interval."""

    values = np.empty_like(times)
    current = I0
    cursor = 0
    for start, stop, pole in _analytic_pole_intervals():
        settled = pole / RLOAD
        mask = (times >= start) & (times <= stop + 1e-15)
        for index in np.nonzero(mask)[0]:
            elapsed = times[index] - start
            values[index] = settled + (current - settled) * math.exp(-elapsed / TAU)
        span = stop - start
        current = settled + (current - settled) * math.exp(-span / TAU)
        cursor += 1
    assert cursor > 0
    return values


def test_the_gate_schedule_matches_the_modulator_law_in_closed_form(run) -> None:
    _, result = run
    expected = _analytic_schedule()
    for gate, times in expected.items():
        observed = sorted(
            event.time
            for event in result.events
            if event.kind == "scheduled_gate" and event.target == gate
        )
        assert len(observed) == len(times), gate
        for got, want in zip(observed, times):
            assert abs(got - want) < SCHEDULE_TOLERANCE_S, (gate, got, want)


def test_the_switching_node_is_pinned_by_the_diode_through_the_dead_time(run) -> None:
    _, result = run
    time = np.asarray(result.time)
    pole = np.asarray(result.signals["v(pole)"])

    # Only the two rails ever appear: a floating pole would show a third level.
    assert set(np.unique(np.round(pole, 6))) == {-UDC_HALF, UDC_HALF}

    for start, stop, expected in _analytic_pole_intervals():
        # Off-edge samples only: a grid point landing exactly on a transition
        # is attributable to either side.
        inside = (time > start + 1e-9) & (time < stop - 1e-9)
        if not inside.any():
            continue
        assert np.max(np.abs(pole[inside] - expected)) < POLE_TOLERANCE_V

    # Tripwire: the dead-time windows must actually be sampled, otherwise the
    # assertion above would be vacuous for the interesting interval.
    first_gap = (time > 0.0) & (time < DEAD_TIME)
    assert first_gap.sum() >= 1
    assert np.all(pole[first_gap] == pytest.approx(-UDC_HALF, abs=POLE_TOLERANCE_V))


def test_the_current_follows_the_closed_form_piecewise_exponential(run) -> None:
    _, result = run
    time = np.asarray(result.time)
    measured = np.asarray(result.signals["i(lo)"])
    expected = _closed_form_current(time)

    deviation = float(np.max(np.abs(measured - expected)))
    assert deviation < CURRENT_TOLERANCE_A, deviation

    # The ripple has to be there: a constant current would match a closed form
    # built from the same constant and prove nothing.
    assert measured.max() - measured.min() > 4.0
    assert measured.min() > 0.0, "the golden assumes the current never reverses"


def test_the_mean_pole_voltage_matches_the_dead_time_corrected_duty(run) -> None:
    """Dead time is not cosmetic: it biases the mean by 2*dt*fsw*Vdc.

    Each period contains TWO gaps — one before the high switch closes and one
    before the low switch closes — and the load current hands both to the
    lower rail, so the effective duty is D - dt/T. The mean pole voltage is
    then (2*D_eff - 1)*Vdc/2 and the settled mean current is that over R.
    This is the check that would fail if dead time were silently dropped.

    The steady relation mean(i) = mean(v)/R only holds once the L/R transient
    has died, which is why this golden runs 5 tau rather than the few periods
    the switching-node checks would need. The residual at 5 tau is 2.3*exp(-5)
    = 0.015 A, so the declared 3e-3 relative budget is deliberate headroom.
    """

    _, result = run
    effective_duty = DUTY - DEAD_TIME / PERIOD
    expected_pole = (2.0 * effective_duty - 1.0) * UDC_HALF
    time = np.asarray(result.time)
    pole = np.asarray(result.signals["v(pole)"])
    # Mean over whole periods, trapezoid on the uniform grid.
    mean_pole = float(np.trapezoid(pole, time) / (time[-1] - time[0]))
    assert mean_pole == pytest.approx(expected_pole, abs=1.0)

    # And the load agrees: the steady mean current is the mean pole over R.
    current = np.asarray(result.signals["i(lo)"])
    tail = time >= time[-1] - 2 * PERIOD
    mean_current = float(np.mean(current[tail]))
    assert mean_current == pytest.approx(expected_pole / RLOAD, rel=3e-3)


def test_the_golden_model_is_the_one_shipped_in_the_repository() -> None:
    """Guard the golden against silent drift of the analytic constants."""

    assert Path(__file__).is_file()
    assert PERIOD == pytest.approx(100e-6)
    assert TAU == pytest.approx(400e-6)
    assert int(round(STOP / PERIOD)) == 20
