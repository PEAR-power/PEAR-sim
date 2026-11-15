"""Loop gain by injection on the switching model, gated by Floquet.

The gate is a different method: scaling the controller by the measured gain
margin must put a Floquet multiplier on the unit circle, at the measured
phase-crossover frequency. Injection is a time-domain sinusoidal measurement;
Floquet is the eigenvalues of a saltation monodromy (itself pinned to a
closed-form oracle in test_closed_loop_floquet.py). Both share the kernel,
so this is verification, not validation against hardware.

Scaling kp and ki together scales L exactly: the orbit's circuit part does
not move (v(out) = vref, same duty), only the integrator register, which the
loop gain does not depend on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pear_sim.analysis.closed_loop import solve_closed_loop_steady_state
from pear_sim.analysis.loop_gain import (
    LoopGainError,
    _Bench,
    measure_loop_gain,
)
from pear_sim.spec.model import load_model

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "buck_closed_loop_floquet.yaml"
KP, KI = 0.002, 30.0


def _model(scale: float = 1.0):
    return load_model(EXAMPLE, {"kp": str(KP * scale), "ki": str(KI * scale)})


@pytest.fixture(scope="module")
def orbit():
    return solve_closed_loop_steady_state(_model())


@pytest.fixture(scope="module")
def measured(orbit):
    # One bracket around the LC resonance holds the -180 degree crossing.
    return measure_loop_gain(_model(), [2000.0, 3000.0], orbit=orbit)


def _floquet_at(orbit, scale: float):
    start = dict(zip(orbit.state_names, orbit.state, strict=True))
    # Same orbit, integrator rescaled so ki * integral still gives the duty.
    start["signal:int_prev"] /= scale
    return solve_closed_loop_steady_state(_model(scale), initial_state=start, tolerance=1e-12)


def test_gain_margin_puts_a_floquet_multiplier_on_the_unit_circle(orbit, measured) -> None:
    """Measured 2026-11-16: GM 2.4508206 at 2452.7533 Hz by injection;
    Floquet bisection puts |mu| = 1 at 2.4508203, 2452.7532 Hz (1.2e-7)."""

    (crossing,) = measured.phase_crossovers
    margin = crossing.gain_margin
    assert 2.0 < margin < 3.0

    at_margin = _floquet_at(orbit, margin)
    assert at_margin.stability != "not_assessed"
    assert abs(at_margin.max_multiplier_magnitude - 1.0) < 1e-5
    assert at_margin.multipliers[0].frequency_hz == pytest.approx(
        crossing.frequency_hz, rel=1e-4
    )
    # Discrimination: 3 % either side of the margin, the verdict flips.
    assert _floquet_at(orbit, 0.97 * margin).stability == "stable"
    assert _floquet_at(orbit, 1.03 * margin).stability == "unstable"


def test_a_short_settle_is_caught_by_the_floquet_gate(orbit, measured) -> None:
    """Mutation: settling 150 cycles instead of the derived ~1300 leaves the
    injection's own transient in the window. The margin moves by ~1e-3 —
    about 100x the gate's tolerance on |mu| above."""

    frequency = measured.phase_crossovers[0].frequency_hz
    short = _Bench(
        _model(),
        block="duty",
        amplitude=measured.amplitude,
        orbit=orbit,
        settle_tolerance=0.2,
        min_window_ticks=200,
        min_window_periods=4,
    )
    assert short.settle_cycles < 200 < measured.settle_cycles
    moved = 1.0 / short.measure(frequency).gain
    assert abs(moved / measured.gain_margin - 1.0) > 1e-4
    assert abs(_floquet_at(orbit, moved).max_multiplier_magnitude - 1.0) > 1e-5


def test_the_loop_is_the_same_loop_wherever_it_is_broken(orbit, measured) -> None:
    """A single loop has one loop gain: injecting at the error signal reads
    the same L as injecting at the duty (the saturate between them is linear
    inside its limits)."""

    frequency = measured.phase_crossovers[0].frequency_hz
    point = _Bench(
        _model(),
        block="err",
        amplitude=1e-3,
        orbit=orbit,
        settle_tolerance=1e-6,
        min_window_ticks=200,
        min_window_periods=4,
    ).measure(frequency)
    assert abs(point.value - complex(-measured.phase_crossovers[0].gain, 0.0)) < 1e-6


def test_phase_margin_of_the_example(orbit) -> None:
    result = measure_loop_gain(_model(), [80.0, 160.0], orbit=orbit)
    (crossing,) = result.gain_crossovers
    # The example's comment says PM ~92 deg (averaged model); this is the
    # switching model's own reading.
    assert crossing.phase_margin_deg == pytest.approx(92.41, abs=0.05)
    assert result.points[0].gain > 1.0 > result.points[1].gain


def test_an_amplitude_that_moves_the_mode_sequence_is_refused(orbit) -> None:
    """Measured: duty 4.5e-3 at 2.4 kHz tips the buck into DCM for some
    cycles; read as a gain it gave a margin of 3.42 instead of 2.45. No
    limiter engages, so only the mode-sequence check can see it."""

    with pytest.raises(LoopGainError, match="mode sequence"):
        measure_loop_gain(_model(), [2400.0, 2500.0], orbit=orbit, amplitude=4.5e-3)


def test_an_unstable_orbit_is_refused_and_pointed_at_floquet() -> None:
    model = load_model(EXAMPLE, {"kp": "0.08", "ki": "150"})
    with pytest.raises(LoopGainError, match="analyze floquet"):
        measure_loop_gain(model, [1000.0, 2000.0])


def test_refusals(orbit) -> None:
    model = _model()
    with pytest.raises(LoopGainError, match="outside"):
        measure_loop_gain(model, [1000.0, 25000.0], orbit=orbit, refine=False)
    with pytest.raises(LoopGainError, match="not a signal block"):
        measure_loop_gain(model, [1000.0, 2000.0], orbit=orbit, inject_at="nope")
    with pytest.raises(LoopGainError, match="at least two"):
        measure_loop_gain(model, [1000.0], orbit=orbit)
