# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""VAL-005 closure items: independent THD golden + retained inverter sweep.

The harmonics producer integrates event-aligned dense output with
Gauss-Legendre quadrature. The golden here recomputes the same window from
the recorded uniform output grid with an FFT — an independent algorithm and
code path — and cross-checks fundamental amplitude and bandwidth-limited
THD. The retained sweep exercises the bounded parameter-sweep harness over
modulation_index on the floating-neutral SPWM inverter with nonzero dead
time, keeping every run as retained evidence.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pear_sim.core.solver import HybridSimulator
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "inverter_three_phase_floating_neutral.yaml"


@pytest.fixture(scope="module")
def inverter_result():
    return HybridSimulator(load_model(EXAMPLE)).simulate()


def _fft_harmonics(result, signal_name: str) -> dict[int, float]:
    """Peak amplitudes of harmonics 1..42 over the declared harmonic window."""

    settings = result.model.harmonics
    time = result.time
    values = result.signals[signal_name]
    step = result.model.transient.output_step
    mask = (time >= settings.start - 1e-12) & (time < settings.stop - 1e-12)
    window = values[mask]
    count = len(window)
    expected = round((settings.stop - settings.start) / step)
    assert count == expected, "uniform half-open window required for the FFT golden"
    spectrum = np.fft.rfft(window) / count
    duration = settings.stop - settings.start
    bin_spacing = 1.0 / duration
    fundamental_bin = round(settings.fundamental_frequency / bin_spacing)
    amplitudes: dict[int, float] = {}
    for order in range(1, settings.max_harmonic + 1):
        amplitudes[order] = 2.0 * float(np.abs(spectrum[order * fundamental_bin]))
    return amplitudes


def test_fft_golden_confirms_fundamental_and_thd(inverter_result) -> None:
    payload = inverter_result.harmonic_analysis
    record = next(
        signal
        for signal in payload["signals"]
        if signal["semantic"] == "line_voltage_ab"
    )
    fft = _fft_harmonics(inverter_result, record["signal"])

    quadrature_fundamental = record["fundamental"]["peak_amplitude"]
    assert fft[1] == pytest.approx(quadrature_fundamental, rel=2e-3)

    fft_thd = math.sqrt(
        sum(fft[order] ** 2 for order in range(2, len(fft) + 1))
    ) / fft[1]
    assert record["thd"]["status"] == "observed"
    assert fft_thd == pytest.approx(record["thd"]["value_fraction"], rel=0.02)
    # Physics anchor: m=0.8 with 500 ns dead time sits just under the ideal
    # m*(Vdc/2)*sqrt(3) line fundamental (dead-time voltage loss ~1.4%).
    ideal = 0.8 * 200.0 * math.sqrt(3.0)
    assert 0.95 * ideal < fft[1] < ideal


def test_fft_golden_covers_phase_currents(inverter_result) -> None:
    payload = inverter_result.harmonic_analysis
    for semantic in ("phase_current_a", "phase_current_b", "phase_current_c"):
        record = next(
            signal for signal in payload["signals"] if signal["semantic"] == semantic
        )
        fft = _fft_harmonics(inverter_result, record["signal"])
        assert fft[1] == pytest.approx(
            record["fundamental"]["peak_amplitude"], rel=2e-3
        ), semantic


def test_retained_modulation_index_sweep(tmp_path: Path) -> None:
    import yaml

    from pear_sim.evidence.parameter_sweep import run_parameter_sweep

    # Shorten for the suite: 6 ms transient, harmonic window 2..6 ms.
    document = yaml.safe_load(EXAMPLE.read_text())
    document["analysis"]["transient"]["stop"] = "6 ms"
    document["analysis"]["harmonics"]["start"] = "2 ms"
    document["analysis"]["harmonics"]["stop"] = "6 ms"
    model_path = tmp_path / "inverter.yaml"
    model_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "schema": "pear-sim/parameter-sweep-request/v0.1",
                "parameter": "modulation_index",
                "values": ["60 %", "75 %", "90 %"],
            }
        ),
        encoding="utf-8",
    )
    artifacts = run_parameter_sweep(
        model_path, request_path, tmp_path / "sweep"
    )
    summary = json.loads(artifacts.sweep_summary.read_text())
    points = summary["points"]
    assert len(points) == 3
    assert all(point["execution_status"] == "pass" for point in points)
    assert all(point["artifacts"]["run"]["path"] for point in points)

    fundamentals = []
    for point in points:
        run_json = artifacts.output_dir / point["artifacts"]["run"]["path"]
        harmonics_path = run_json.parent / "harmonics.json"
        harmonics = json.loads(harmonics_path.read_text())
        record = next(
            signal
            for signal in harmonics["signals"]
            if signal["semantic"] == "line_voltage_ab"
        )
        fundamentals.append(record["fundamental"]["peak_amplitude"])
    # The line fundamental scales with the commanded modulation index.
    assert fundamentals[0] < fundamentals[1] < fundamentals[2]
    ratio = fundamentals[2] / fundamentals[0]
    assert ratio == pytest.approx(90.0 / 60.0, rel=0.05)
