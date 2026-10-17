"""Offline SRF-PLL vs known synthetic grids and a simulated weak grid."""

from pathlib import Path

import numpy as np
import pytest

from pear_sim.analysis.pll import srf_pll
from pear_sim.analysis.reference_frames import ReferenceFrameError

ROOT = Path(__file__).parents[1]


def _grid(amplitude, frequency, t, phase=0.0):
    omega = 2.0 * np.pi * frequency
    a = amplitude * np.cos(omega * t + phase)
    b = amplitude * np.cos(omega * t + phase - 2.0 * np.pi / 3.0)
    c = amplitude * np.cos(omega * t + phase + 2.0 * np.pi / 3.0)
    return a, b, c


def test_pll_locks_to_a_clean_grid_with_frequency_offset() -> None:
    t = np.linspace(0.0, 1.0, 20001)
    a, b, c = _grid(325.0, 51.3, t, phase=0.9)
    result = srf_pll(t, a, b, c, nominal_frequency_hz=50.0, bandwidth_hz=20.0)
    assert result.locked
    assert result.locked_frequency_hz == pytest.approx(51.3, abs=0.01)
    assert result.lock_time_s is not None and result.lock_time_s < 0.3
    # Locked angle equals the signal angle: theta tracks omega*t + phase.
    tail = slice(len(t) // 2, None)
    theta = np.asarray(result.theta_rad)[tail]
    expected = (2.0 * np.pi * 51.3 * t[tail] + 0.9) % (2.0 * np.pi)
    wrapped = np.angle(np.exp(1j * (theta - expected)))
    assert np.max(np.abs(wrapped)) < 5e-3


def test_pll_lock_is_amplitude_independent() -> None:
    t = np.linspace(0.0, 0.5, 10001)
    small = srf_pll(t, *_grid(1.0, 50.0, t), nominal_frequency_hz=50.0)
    large = srf_pll(t, *_grid(400.0, 50.0, t), nominal_frequency_hz=50.0)
    assert small.locked and large.locked
    assert small.locked_frequency_hz == pytest.approx(large.locked_frequency_hz, abs=1e-9)


def test_pll_survives_harmonic_distortion() -> None:
    t = np.linspace(0.0, 1.0, 40001)
    a, b, c = _grid(325.0, 50.0, t)
    omega = 2.0 * np.pi * 50.0
    # 5% fifth harmonic (negative sequence), a classic grid distortion.
    a = a + 16.0 * np.cos(-5.0 * omega * t)
    b = b + 16.0 * np.cos(-5.0 * omega * t + 2.0 * np.pi / 3.0)
    c = c + 16.0 * np.cos(-5.0 * omega * t - 2.0 * np.pi / 3.0)
    result = srf_pll(
        t, a, b, c, nominal_frequency_hz=50.0, bandwidth_hz=10.0, lock_threshold=0.05
    )
    assert result.locked
    assert result.locked_frequency_hz == pytest.approx(50.0, abs=0.05)


def test_pll_reports_unlocked_honestly() -> None:
    rng_t = np.linspace(0.0, 0.2, 5001)
    # A 70 Hz signal against a 50 Hz nominal with a slow loop: cannot settle
    # within the record; the PLL must say so instead of averaging garbage.
    a, b, c = _grid(100.0, 70.0, rng_t)
    result = srf_pll(rng_t, a, b, c, nominal_frequency_hz=50.0, bandwidth_hz=2.0)
    assert not result.locked
    assert result.locked_frequency_hz is None
    assert result.lock_time_s is None


def test_pll_validation_boundaries() -> None:
    t = np.linspace(0.0, 1.0, 100)
    a, b, c = _grid(1.0, 50.0, t)
    with pytest.raises(ReferenceFrameError, match="bandwidth"):
        srf_pll(t, a, b, c, nominal_frequency_hz=50.0, bandwidth_hz=0.0)
    with pytest.raises(ReferenceFrameError, match="sample grid"):
        srf_pll(t, a, b, c, nominal_frequency_hz=50.0, bandwidth_hz=500.0)
    with pytest.raises(ReferenceFrameError, match="increasing"):
        srf_pll(np.zeros(100), a, b, c, nominal_frequency_hz=50.0)


def test_pll_locks_to_simulated_weak_grid_voltages(tmp_path: Path) -> None:
    """End-to-end: Thevenin weak grid simulated by the kernel, PLL on v(load)."""

    from pear_sim.core.solver import HybridSimulator
    from pear_sim.spec.model import load_model

    path = tmp_path / "weak_grid.yaml"
    path.write_text(
        """
schema: pear-sim/v0.1
name: weak_grid_three_phase

parameters:
  amp: 325 V
  f: 50 Hz
  r_grid: 0.5 ohm
  l_grid: 2 mH

circuit:
  ground: gnd
  components:
    - id: ea
      type: voltage_source
      p: sa
      n: gnd
      waveform: {type: sine, amplitude: "${amp}", frequency: "${f}", phase_deg: 90}
    - id: eb
      type: voltage_source
      p: sb
      n: gnd
      waveform: {type: sine, amplitude: "${amp}", frequency: "${f}", phase_deg: -30}
    - id: ec
      type: voltage_source
      p: sc
      n: gnd
      waveform: {type: sine, amplitude: "${amp}", frequency: "${f}", phase_deg: 210}
    - {id: rga, type: resistor, p: sa, n: ga, value: "${r_grid}"}
    - {id: lga, type: inductor, p: ga, n: pa, value: "${l_grid}", initial_current: 0 A}
    - {id: rgb, type: resistor, p: sb, n: gb, value: "${r_grid}"}
    - {id: lgb, type: inductor, p: gb, n: pb, value: "${l_grid}", initial_current: 0 A}
    - {id: rgc, type: resistor, p: sc, n: gc, value: "${r_grid}"}
    - {id: lgc, type: inductor, p: gc, n: pc, value: "${l_grid}", initial_current: 0 A}
    - {id: loada, type: resistor, p: pa, n: gnd, value: 20 ohm}
    - {id: loadb, type: resistor, p: pb, n: gnd, value: 20 ohm}
    - {id: loadc, type: resistor, p: pc, n: gnd, value: 20 ohm}

analysis:
  transient:
    start: 0 s
    stop: 400 ms
    output_step: 100 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11

outputs:
  - v(pa)
  - v(pb)
  - v(pc)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result = HybridSimulator(load_model(path)).simulate()
    pll = srf_pll(
        result.time,
        result.signals["v(pa)"],
        result.signals["v(pb)"],
        result.signals["v(pc)"],
        nominal_frequency_hz=50.0,
        bandwidth_hz=15.0,
    )
    assert pll.locked
    assert pll.locked_frequency_hz == pytest.approx(50.0, abs=0.01)
    # The PCC voltage lags the source EMF through the grid impedance; the PLL
    # angle must match the SIMULATED pcc voltage, cross-checked via Park:
    from pear_sim.analysis.reference_frames import clarke, park

    alpha, beta, _ = clarke(
        result.signals["v(pa)"], result.signals["v(pb)"], result.signals["v(pc)"]
    )
    tail = slice(len(result.time) // 2, None)
    d, q = park(alpha[tail], beta[tail], np.asarray(pll.theta_rad)[tail])
    assert np.max(np.abs(q)) / np.mean(np.hypot(d, q)) < 0.01
