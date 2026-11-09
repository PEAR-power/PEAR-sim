# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Clarke/Park transforms and P/Q metrics vs closed forms."""

import numpy as np
import pytest

from pear_sim.analysis.reference_frames import (
    ReferenceFrameError,
    clarke,
    instantaneous_power,
    park,
    three_phase_frames,
)


def _balanced(amplitude, frequency, t, phase=0.0):
    omega = 2.0 * np.pi * frequency
    a = amplitude * np.cos(omega * t + phase)
    b = amplitude * np.cos(omega * t + phase - 2.0 * np.pi / 3.0)
    c = amplitude * np.cos(omega * t + phase + 2.0 * np.pi / 3.0)
    return a, b, c


def test_clarke_balanced_set_has_unit_amplitude_and_no_zero_sequence() -> None:
    t = np.linspace(0.0, 0.04, 801)
    a, b, c = _balanced(10.0, 50.0, t)
    alpha, beta, zero = clarke(a, b, c)
    assert np.max(np.abs(np.hypot(alpha, beta) - 10.0)) < 1e-12
    assert np.max(np.abs(zero)) < 1e-12
    assert np.max(np.abs(alpha - a)) < 1e-12  # alpha aligned with phase a


def test_park_locks_balanced_set_to_constant_dq() -> None:
    t = np.linspace(0.0, 0.04, 801)
    phase = 0.7
    a, b, c = _balanced(10.0, 50.0, t, phase=phase)
    frames = three_phase_frames(t, a, b, c, frequency_hz=50.0)
    d = np.asarray(frames.d)
    q = np.asarray(frames.q)
    assert np.max(np.abs(d - 10.0 * np.cos(phase))) < 1e-12
    assert np.max(np.abs(q - 10.0 * np.sin(phase))) < 1e-12


def test_frequency_mismatch_shows_up_as_difference_frequency_rotation() -> None:
    t = np.linspace(0.0, 1.0, 20001)
    a, b, c = _balanced(1.0, 51.0, t)
    frames = three_phase_frames(t, a, b, c, frequency_hz=50.0)
    d = np.asarray(frames.d)
    # dq vector rotates at 1 Hz: d crosses zero twice per beat period.
    signs = np.sign(d[np.abs(d) > 1e-6])
    assert np.sum(signs[1:] != signs[:-1]) == pytest.approx(2, abs=1)


def test_zero_sequence_is_separated() -> None:
    t = np.linspace(0.0, 0.04, 801)
    a, b, c = _balanced(10.0, 50.0, t)
    offset = 3.0
    alpha, beta, zero = clarke(a + offset, b + offset, c + offset)
    assert np.max(np.abs(zero - offset)) < 1e-12
    assert np.max(np.abs(np.hypot(alpha, beta) - 10.0)) < 1e-12


def test_instantaneous_power_matches_phasor_closed_form() -> None:
    t = np.linspace(0.0, 0.1, 5001)
    v_amp, i_amp, phi = 100.0, 5.0, np.pi / 6.0  # current lags by 30 deg
    va, vb, vc = _balanced(v_amp, 50.0, t)
    ia, ib, ic = _balanced(i_amp, 50.0, t, phase=-phi)
    power = instantaneous_power(va, vb, vc, ia, ib, ic)
    p_expected = 1.5 * v_amp * i_amp * np.cos(phi)
    q_expected = 1.5 * v_amp * i_amp * np.sin(phi)
    assert np.max(np.abs(power["p_w"] - p_expected)) < 1e-9
    assert np.max(np.abs(power["q_var"] - q_expected)) < 1e-9


def test_power_convention_resistive_load_has_zero_q() -> None:
    t = np.linspace(0.0, 0.1, 5001)
    va, vb, vc = _balanced(100.0, 50.0, t)
    ia, ib, ic = _balanced(10.0, 50.0, t)
    power = instantaneous_power(va, vb, vc, ia, ib, ic)
    assert np.max(np.abs(power["q_var"])) < 1e-9
    assert np.max(np.abs(power["p_w"] - 1500.0)) < 1e-9


def test_shape_and_finite_validation() -> None:
    with pytest.raises(ReferenceFrameError, match="equal length"):
        clarke([1.0, 2.0], [1.0], [1.0])
    with pytest.raises(ReferenceFrameError, match="finite"):
        clarke([np.nan], [0.0], [0.0])
    with pytest.raises(ReferenceFrameError, match="positive"):
        three_phase_frames([0.0], [0.0], [0.0], [0.0], frequency_hz=0.0)
    with pytest.raises(ReferenceFrameError, match="one shape"):
        park([1.0, 2.0], [1.0, 2.0], [0.0])


def test_frames_from_simulated_inverter_output(tmp_path) -> None:
    """End-to-end: SPWM inverter phase currents lock to near-constant dq."""

    from pathlib import Path

    from pear_sim.model import load_model
    from pear_sim.solver import HybridSimulator

    example = Path(__file__).parents[1] / "examples" / "inverter_three_phase_spwm.yaml"
    if not example.exists():
        pytest.skip("three-phase SPWM example not present")
    result = HybridSimulator(load_model(example)).simulate()
    names = [name for name in result.signals if name.startswith("i(")]
    if len(names) < 3:
        pytest.skip("example does not record three phase currents")
    t = result.time
    mask = t >= 0.5 * float(t[-1])
    frames = three_phase_frames(
        t[mask],
        result.signals[names[0]][mask],
        result.signals[names[1]][mask],
        result.signals[names[2]][mask],
        frequency_hz=50.0,
    )
    d = np.asarray(frames.d)
    q = np.asarray(frames.q)
    magnitude = np.hypot(d, q)
    # The fundamental dominates: dq magnitude ripple stays well below its mean.
    assert np.mean(magnitude) > 0.0
    assert np.std(magnitude) < 0.35 * np.mean(magnitude)
