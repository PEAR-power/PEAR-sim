"""Small-signal AC sweep on the switched model, judged by an internal oracle.

The averaged model in `analysis/small_signal.py` and this sweep answer the
same question by completely different routes: one linearises a two-topology
cycle and evaluates a transfer function in closed form, the other perturbs
the real switching circuit and correlates the recorded output. Below the
averaging's declared validity band they must agree — and where they part
company, the averaged model is the one that stopped being valid.

That makes the agreement a free, fully independent gate. No external tool
is required to check it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pear_sim.analysis.ac_sweep import (
    AcSweepError,
    ac_sweep,
    ac_sweep_point,
    correlate,
    perturbed_model,
)
from pear_sim.analysis.small_signal import averaged_small_signal, frequency_response
from pear_sim.analysis.steady_state import solve_periodic_steady_state
from pear_sim.model import load_model

ROOT = Path(__file__).resolve().parents[1]
BUCK = ROOT / "examples" / "buck.yaml"


@pytest.fixture(scope="module")
def averaged():
    model = load_model(BUCK)
    return model, averaged_small_signal(model, solve_periodic_steady_state(model), "v(out)")


def test_gain_matches_the_averaged_model_inside_its_validity_band(averaged) -> None:
    """The headline gate: two independent derivations, one magnitude.

    Measured on the canonical buck (20 kHz): 0.000 % at 100 Hz and 300 Hz,
    0.004 % at 1 kHz. The band is asserted at f_sw/20, which is where the
    averaged model's own capability entry says it may be trusted.
    """

    model, avg = averaged
    frequencies = np.array([100.0, 300.0, 1000.0])
    reference = frequency_response(avg, frequencies)
    points = ac_sweep(model, "v(out)", frequencies, amplitude=0.01,
                      periods=10, settle_periods=6)
    for point, expected in zip(points, reference, strict=True):
        assert point.gain == pytest.approx(abs(expected), rel=1e-3)


def test_the_switched_model_shows_modulator_delay_the_average_cannot(averaged) -> None:
    """Where they diverge is the POINT, not an error.

    The duty command is held for a carrier cycle; that transport lag costs
    phase in proportion to frequency, and an averaged model has no such
    term. The residual phase tracks a fraction of half a carrier period
    (measured: 0.77-0.83 of it across two decades), so the test asserts the
    SHAPE — proportional to frequency, in the lagging direction — rather
    than a fitted constant.
    """

    model, avg = averaged
    frequencies = np.array([300.0, 1000.0, 2000.0])
    reference = frequency_response(avg, frequencies)
    points = ac_sweep(model, "v(out)", frequencies, amplitude=0.01,
                      periods=10, settle_periods=6)
    ratios = []
    for point, expected in zip(points, reference, strict=True):
        residual = (point.phase_deg - np.degrees(np.angle(expected)) + 180.0) % 360.0 - 180.0
        half_cycle = -360.0 * point.frequency_hz * 0.5 / avg.switching_frequency_hz
        assert residual < 0.0, "the switched model must LAG, never lead"
        ratios.append(residual / half_cycle)
    # Same fraction of the half-cycle at every frequency: a transport lag,
    # not a coincidence.
    assert max(ratios) - min(ratios) < 0.15
    assert 0.5 < float(np.mean(ratios)) < 1.2


def test_the_result_does_not_depend_on_the_perturbation_amplitude(averaged) -> None:
    """A response that moves with amplitude is not a SMALL-signal response."""

    model, _ = averaged
    gains = [
        ac_sweep_point(model, "v(out)", frequency_hz=500.0, amplitude=a,
                       periods=8, settle_periods=5).gain
        for a in (0.005, 0.01, 0.02)
    ]
    assert max(gains) / min(gains) - 1.0 < 0.01


def test_correlation_recovers_a_synthetic_tone_exactly() -> None:
    """The estimator itself, isolated from any converter."""

    frequency = 250.0
    time = np.linspace(0.0, 4.0 / frequency, 20001)
    signal = 7.0 + 3.0 * np.sin(2.0 * np.pi * frequency * time + 0.6)
    recovered = correlate(time, signal, frequency_hz=frequency, window_start_s=0.0)
    assert abs(recovered) == pytest.approx(3.0, rel=1e-4)
    assert np.degrees(np.angle(recovered)) == pytest.approx(np.degrees(0.6), abs=0.05)


def test_a_dc_offset_cannot_masquerade_as_gain() -> None:
    frequency = 100.0
    time = np.linspace(0.0, 3.0 / frequency, 9001)
    recovered = correlate(time, np.full_like(time, 42.0),
                          frequency_hz=frequency, window_start_s=0.0)
    assert abs(recovered) < 1e-9


def test_perturbation_above_nyquist_is_refused(averaged) -> None:
    """The duty is updated once per carrier cycle and cannot carry more."""

    model, avg = averaged
    with pytest.raises(AcSweepError, match="Nyquist"):
        ac_sweep(model, "v(out)", [0.6 * avg.switching_frequency_hz])


def test_a_model_with_its_own_signal_layer_is_refused() -> None:
    """Perturbing it would silently redefine the author's controller."""

    model = load_model(ROOT / "examples" / "buck_closed_loop_signals.yaml")
    with pytest.raises(AcSweepError, match="already declares a signal layer"):
        perturbed_model(model, frequency_hz=100.0, amplitude=0.01, stop_s=0.01)


def test_the_perturbed_model_keeps_the_converter_intact(averaged) -> None:
    """Only the duty source changes; the circuit must be untouched."""

    model, _ = averaged
    perturbed = perturbed_model(model, frequency_hz=500.0, amplitude=0.01, stop_s=0.02)
    assert [c.id for c in perturbed.components] == [c.id for c in model.components]
    assert perturbed.signals is not None
    assert perturbed.sampled_controls, "the duty must now come from the signal layer"
