# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

import pear_sim.harmonics as harmonics_module
from pear_sim.errors import SimulationError
from pear_sim.harmonics import HarmonicAccumulator


class _DenseTime:
    def __call__(self, times):
        return np.asarray(times, dtype=float)[np.newaxis, :]


class _AnalyticMode:
    @staticmethod
    def output(
        expression: str, state: np.ndarray, _time: float | None = None
    ) -> float:
        time = float(state[0])
        angle = 2.0 * math.pi * time
        if expression == "v(test)":
            return 3.0 + 2.0 * math.sin(angle) + 0.5 * math.cos(5.0 * angle)
        offsets = {
            "i(a)": 0.0,
            "i(b)": -2.0 * math.pi / 3.0,
            "i(c)": 2.0 * math.pi / 3.0,
        }
        return math.sin(angle + offsets[expression])


def _model() -> SimpleNamespace:
    signals = (
        SimpleNamespace(signal="v(test)", semantic="line_voltage_ab"),
        SimpleNamespace(signal="i(a)", semantic="phase_current_a"),
        SimpleNamespace(signal="i(b)", semantic="phase_current_b"),
        SimpleNamespace(signal="i(c)", semantic="phase_current_c"),
    )
    harmonics = SimpleNamespace(
        start=0.0,
        stop=1.0,
        fundamental_frequency=1.0,
        max_harmonic=5,
        period_count=1,
        signals=signals,
    )
    return SimpleNamespace(harmonics=harmonics)


def test_dense_event_aligned_fourier_recovers_known_dc_harmonics_and_sequence() -> None:
    accumulator = HarmonicAccumulator(_model())
    solution = SimpleNamespace(
        t=np.asarray([0.0, 0.37, 0.71, 1.0]),
        sol=_DenseTime(),
    )
    accumulator.add_solution(solution, _AnalyticMode())
    artifact = accumulator.finalize()

    assert artifact is not None
    voltage = artifact["signals"][0]
    assert voltage["unit"] == "V"
    assert voltage["dc"] == pytest.approx(3.0, abs=1e-12)
    assert voltage["fundamental"]["peak_amplitude"] == pytest.approx(2.0, abs=1e-12)
    assert voltage["fundamental"]["rms_amplitude"] == pytest.approx(
        math.sqrt(2.0), abs=1e-12
    )
    assert voltage["fundamental"]["phase_deg"] == pytest.approx(-90.0, abs=1e-10)
    assert voltage["coefficients"][5]["peak_amplitude"] == pytest.approx(
        0.5, abs=1e-12
    )
    assert voltage["thd"]["value_fraction"] == pytest.approx(0.25, abs=1e-12)
    assert artifact["method"]["report_recomputes_values"] is False
    assert artifact["method"]["topology_segment_count"] == 1
    assert artifact["method"]["dense_output_interval_count"] == 3

    sequence = artifact["sequence_components"][0]
    assert all(record["unit"] == "A" for record in artifact["signals"][1:])
    assert sequence["family"] == "phase_current"
    assert sequence["rms_phasors"]["positive"]["magnitude"] == pytest.approx(
        1.0 / math.sqrt(2.0), abs=1e-12
    )
    assert sequence["negative_to_positive_fraction"] == pytest.approx(0.0, abs=1e-12)
    assert sequence["zero_to_positive_fraction"] == pytest.approx(0.0, abs=1e-12)


def test_harmonic_accumulator_fails_closed_on_incomplete_window() -> None:
    accumulator = HarmonicAccumulator(_model())
    accumulator.add_solution(
        SimpleNamespace(t=np.asarray([0.1, 1.0]), sol=_DenseTime()),
        _AnalyticMode(),
    )

    with pytest.raises(SimulationError, match="does not cover"):
        accumulator.finalize()


def test_harmonic_accumulator_is_disabled_when_not_requested() -> None:
    accumulator = HarmonicAccumulator(SimpleNamespace(harmonics=None))
    accumulator.add_solution(
        SimpleNamespace(t=np.asarray([0.0, 1.0]), sol=_DenseTime()),
        _AnalyticMode(),
    )

    assert accumulator.finalize() is None


def test_harmonic_runtime_panel_budget_fails_before_materializing_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harmonics_module, "MAX_HARMONIC_QUADRATURE_PANELS", 1)
    accumulator = HarmonicAccumulator(_model())

    with pytest.raises(SimulationError, match="runtime limit of 1 panels"):
        accumulator.add_solution(
            SimpleNamespace(t=np.asarray([0.0, 1.0]), sol=_DenseTime()),
            _AnalyticMode(),
        )


def test_harmonic_runtime_work_budget_accounts_for_order_and_signal_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harmonics_module, "MAX_HARMONIC_QUADRATURE_WORK_UNITS", 1)
    accumulator = HarmonicAccumulator(_model())

    with pytest.raises(SimulationError, match="runtime work limit of 1 units"):
        accumulator.add_solution(
            SimpleNamespace(t=np.asarray([0.0, 1.0]), sol=_DenseTime()),
            _AnalyticMode(),
        )
