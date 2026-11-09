# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Twelve-pulse rectifier: harmonic cancellation as the acceptance criterion.

A twelve-pulse rectifier is not "two bridges wired together" — it is two
bridges whose six-pulse harmonics CANCEL. That cancellation is the whole point
of the topology and the only thing worth asserting, because a model can be
wired correctly, run cleanly, produce a plausible DC voltage, and still be a
mediocre six-pulse rectifier in disguise.

So the tests here check the signature on both sides (DC ripple at multiples of
12*f_grid only, line current at 12k+-1 only), and then deliberately BREAK the
turns ratio to show the cancellation is a real physical result and not an
artifact of the analysis window.

The model needs three coupling groups of three windings each — the case that
pairwise coupling could not express at all.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pear_sim.core.circuit import Circuit
from pear_sim.errors import TopologyError
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator

ROOT = Path(__file__).parents[1]
MODEL = ROOT / "examples" / "rectifier_12pulse_yyd.yaml"
GRID_HZ = 50.0


def _spectrum(times: np.ndarray, signal: np.ndarray):
    """Amplitude spectrum over exactly one grid period of the settled tail.

    The window MUST be a whole grid period: half a period puts the fundamental
    between bins, and every odd harmonic then shows up as leakage. That mistake
    produced a convincing-looking spectrum of pure nonsense before this helper
    existed, so the assertion below guards the window length.
    """

    step = float(times[1] - times[0])
    period = 1.0 / GRID_HZ
    mask = times >= times[-1] - period + step / 2
    samples = int(mask.sum())
    assert samples == pytest.approx(round(period / step), abs=1)
    window = np.asarray(signal)[mask]
    window = window - window.mean()
    frequencies = np.fft.rfftfreq(samples, step)
    amplitudes = np.abs(np.fft.rfft(window)) / samples * 2.0
    return frequencies, amplitudes, mask


def _at(frequencies: np.ndarray, amplitudes: np.ndarray, hertz: float) -> float:
    return float(amplitudes[int(np.argmin(np.abs(frequencies - hertz)))])


@pytest.fixture(scope="module")
def run():
    model = load_model(MODEL)
    return model, HybridSimulator(model).simulate()


def test_the_transformer_forms_three_groups_of_three_windings(run) -> None:
    model, _ = run
    circuit = Circuit(model)
    assert circuit.coupling_groups == (
        ("lda", "lpa", "lya"),
        ("ldb", "lpb", "lyb"),
        ("ldc", "lpc", "lyc"),
    )
    for group in circuit.coupling_groups:
        matrix = circuit.coupling_matrix(group)
        assert matrix.shape == (3, 3)
        # Positive definite, or some winding-current set would store negative
        # energy; the loader already refused otherwise, this pins it.
        np.linalg.cholesky(matrix)


def test_the_dc_ripple_carries_only_multiples_of_twelve_times_the_grid(run) -> None:
    _, result = run
    times = np.asarray(result.time)
    dc = np.asarray(result.signals["v(dcp)"]) - np.asarray(result.signals["v(dbot)"])
    frequencies, amplitudes, mask = _spectrum(times, dc)

    ripple_600 = _at(frequencies, amplitudes, 600.0)
    assert ripple_600 > 5.0, "no twelve-pulse ripple at all"
    # The six-pulse harmonics the two bridges are supposed to cancel.
    for hertz in (300.0, 900.0, 1500.0):
        residual = _at(frequencies, amplitudes, hertz)
        assert residual < ripple_600 / 1000.0, (hertz, residual)

    mean_dc = float(dc[mask].mean())
    assert 850.0 < mean_dc < 980.0, mean_dc


def test_the_line_current_carries_only_the_twelve_k_plus_minus_one_family(run) -> None:
    _, result = run
    times = np.asarray(result.time)
    frequencies, amplitudes, _ = _spectrum(times, result.signals["i(vga)"])

    fundamental = _at(frequencies, amplitudes, GRID_HZ)
    assert fundamental > 10.0

    # 12k+-1 survives: the 11th and 13th are the pair a twelve-pulse supply is
    # bought to leave behind, and they are the dominant distortion here.
    assert _at(frequencies, amplitudes, 11 * GRID_HZ) / fundamental > 0.03
    assert _at(frequencies, amplitudes, 13 * GRID_HZ) / fundamental > 0.02

    # 6k+-1 outside that family cancels: these are exactly what the 30 degree
    # shift removes, and they are what a single six-pulse bridge would show.
    for order in (5, 7, 17, 19):
        share = _at(frequencies, amplitudes, order * GRID_HZ) / fundamental
        assert share < 1e-3, (order, share)


def test_the_declared_winding_shunts_stay_negligible(run) -> None:
    """The workaround has to be bounded, not just declared."""

    _, result = run
    times = np.asarray(result.time)
    _, _, mask = _spectrum(times, result.signals["i(rl)"])
    load = float(np.asarray(result.signals["i(rl)"])[mask].mean())
    for name in ("i(rma)", "i(rna)"):
        shunt = float(np.max(np.abs(np.asarray(result.signals[name])[mask])))
        assert shunt / load < 1e-3, (name, shunt, load)


def test_a_wrong_delta_turns_ratio_destroys_the_cancellation(tmp_path: Path) -> None:
    """The cancellation is physics, not an artifact of how it is measured.

    Give the delta secondary the same turns as the star one and the two bridges
    stop contributing equally. Nothing errors, the rectifier still rectifies —
    it just becomes a poor six-pulse rectifier with a 300 Hz ripple, which is
    the failure mode this whole example exists to make visible.
    """

    body = MODEL.read_text(encoding="utf-8").replace(
        "  lsec_d: 37.5 mH", "  lsec_d: 12.5 mH"
    )
    path = tmp_path / "wrong_ratio.yaml"
    path.write_text(body, encoding="utf-8")
    result = HybridSimulator(load_model(path)).simulate()

    times = np.asarray(result.time)
    dc = np.asarray(result.signals["v(dcp)"]) - np.asarray(result.signals["v(dbot)"])
    frequencies, amplitudes, _ = _spectrum(times, dc)
    six_pulse = _at(frequencies, amplitudes, 300.0)
    twelve_pulse = _at(frequencies, amplitudes, 600.0)
    assert six_pulse > twelve_pulse, (six_pulse, twelve_pulse)


def test_without_the_declared_support_the_kernel_refuses_and_says_why(
    tmp_path: Path,
) -> None:
    """Pins the kernel gap (no zero-current reduction for this coupled-winding case) as an executable statement.

    A coupled winding whose branch is opened by a non-conducting diode has no
    zero-current reduction in v0.1, so the transformer-fed bridge cannot start.
    The refusal is the correct behaviour — fail closed, name the node — and
    this test documents it. When that reduction lands, this test is the one that changes.
    """

    body = MODEL.read_text(encoding="utf-8")
    for winding in ("rma", "rmb", "rmc", "rna", "rnb", "rnc"):
        body = "\n".join(
            line
            for line in body.splitlines()
            # The probes on the shunts go with them, or the loader rejects the
            # outputs before the topology is ever compiled.
            if f"id: {winding}," not in line and f"- i({winding})" not in line
        )
    path = tmp_path / "unsupported.yaml"
    path.write_text(body + "\n", encoding="utf-8")

    with pytest.raises(TopologyError, match="no algebraic support"):
        HybridSimulator(load_model(path)).simulate()
