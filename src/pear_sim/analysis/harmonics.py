# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Event-aligned dense-output Fourier observation for explicit model requests."""

from __future__ import annotations

import cmath
import math
from typing import Any, Callable

import numpy as np

from pear_sim.errors import SimulationError
from pear_sim.spec.signals import signal_unit
from pear_sim.limits import (
    HARMONIC_QUADRATURE_ORDER,
    MAX_HARMONIC_QUADRATURE_PANELS,
    MAX_HARMONIC_QUADRATURE_WORK_UNITS,
)


HARMONIC_ANALYSIS_SCHEMA = "pear-sim/harmonic-analysis/v0.1"
HARMONIC_ANALYSIS_KIND = "harmonic-analysis-observation"
_QUADRATURE_ORDER = HARMONIC_QUADRATURE_ORDER
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(_QUADRATURE_ORDER)
_MAX_PHASE_SPAN_RAD = 0.5 * math.pi


class HarmonicAccumulator:
    """Accumulate requested Fourier coefficients on accepted topology segments.

    Gauss-Legendre nodes lie strictly inside every event-aligned interval, so isolated
    switching-boundary values carry zero measure.  Long dense-output steps are split
    so the highest requested harmonic advances by at most pi/2 per quadrature panel.
    """

    def __init__(self, model: Any):
        self.model = model
        self.settings = getattr(model, "harmonics", None)
        self.enabled = self.settings is not None
        self.panel_count = 0
        self.topology_segment_count = 0
        self.dense_output_interval_count = 0
        self.integrated_duration_s = 0.0
        if not self.enabled:
            self._signals: tuple[Any, ...] = ()
            self._orders = np.empty(0, dtype=float)
            self._integrals = np.empty((0, 0), dtype=np.complex128)
            return

        self._signals = tuple(self.settings.signals)
        self._orders = np.arange(self.settings.max_harmonic + 1, dtype=float)
        self._integrals = np.zeros(
            (len(self._signals), self.settings.max_harmonic + 1),
            dtype=np.complex128,
        )

    def add_solution(self, solution: Any, mode: Any) -> None:
        if not self.enabled or solution.sol is None or len(solution.t) < 2:
            return
        window_start = float(self.settings.start)
        window_stop = float(self.settings.stop)
        segment_overlaps_window = False
        for raw_start, raw_stop in zip(solution.t[:-1], solution.t[1:], strict=True):
            start = max(float(raw_start), window_start)
            stop = min(float(raw_stop), window_stop)
            if stop <= start:
                continue
            segment_overlaps_window = True
            self.dense_output_interval_count += 1
            self._integrate_interval(solution.sol, mode, start, stop)
        if segment_overlaps_window:
            self.topology_segment_count += 1

    def _integrate_interval(
        self,
        dense_output: Callable[[np.ndarray], np.ndarray],
        mode: Any,
        start: float,
        stop: float,
    ) -> None:
        bandwidth_hz = self.settings.fundamental_frequency * self.settings.max_harmonic
        max_panel_s = (
            math.inf
            if bandwidth_hz <= 0.0
            else _MAX_PHASE_SPAN_RAD / (2.0 * math.pi * bandwidth_hz)
        )
        panel_count = max(1, math.ceil((stop - start) / max_panel_s))
        if self.panel_count + panel_count > MAX_HARMONIC_QUADRATURE_PANELS:
            raise SimulationError(
                "Harmonic quadrature exceeds the portable runtime limit of "
                f"{MAX_HARMONIC_QUADRATURE_PANELS} panels"
            )
        projected_work_units = (
            (self.panel_count + panel_count)
            * _QUADRATURE_ORDER
            * (self.settings.max_harmonic + 1)
            * (len(self._signals) + 1)
        )
        if projected_work_units > MAX_HARMONIC_QUADRATURE_WORK_UNITS:
            raise SimulationError(
                "Harmonic quadrature exceeds the portable runtime work limit of "
                f"{MAX_HARMONIC_QUADRATURE_WORK_UNITS} units"
            )
        boundaries = np.linspace(start, stop, panel_count + 1)
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True):
            self._integrate_panel(
                dense_output,
                mode,
                float(left),
                float(right),
            )

    def _integrate_panel(
        self,
        dense_output: Callable[[np.ndarray], np.ndarray],
        mode: Any,
        start: float,
        stop: float,
    ) -> None:
        half_width = 0.5 * (stop - start)
        midpoint = 0.5 * (start + stop)
        times = midpoint + half_width * _NODES
        weights = half_width * _WEIGHTS
        states = np.asarray(dense_output(times), dtype=float)
        if states.ndim == 1:
            states = states[:, np.newaxis]
        basis = np.exp(
            -2j
            * math.pi
            * self.settings.fundamental_frequency
            * self._orders[:, np.newaxis]
            * times[np.newaxis, :]
        )
        for signal_index, signal in enumerate(self._signals):
            values = np.asarray(
                [
                    mode.output(
                        signal.signal,
                        states[:, sample_index],
                        float(times[sample_index]),
                    )
                    for sample_index in range(states.shape[1])
                ],
                dtype=float,
            )
            self._integrals[signal_index, :] += basis @ (weights * values)
        self.integrated_duration_s += stop - start
        self.panel_count += 1

    def finalize(self) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        expected_duration = float(self.settings.stop - self.settings.start)
        duration_tolerance = max(1e-15, expected_duration * 1e-10)
        if abs(self.integrated_duration_s - expected_duration) > duration_tolerance:
            raise SimulationError(
                "Harmonic quadrature does not cover the requested half-open window: "
                f"got {self.integrated_duration_s:.17g} s, expected "
                f"{expected_duration:.17g} s"
            )

        signal_records = [
            _signal_record(
                signal.signal,
                signal.semantic,
                self._integrals[index, :] / expected_duration,
                self.settings.fundamental_frequency,
            )
            for index, signal in enumerate(self._signals)
        ]
        return {
            "schema": HARMONIC_ANALYSIS_SCHEMA,
            "kind": HARMONIC_ANALYSIS_KIND,
            "status": "observed",
            "claim": {
                "classification": "simulation-observation",
                "validation_claim": False,
                "compatibility_claim": False,
                "hardware_sign_off_allowed": False,
                "statement": (
                    "Producer-calculated Fourier observation for the explicitly declared "
                    "window and bandwidth; no reference comparison or validation verdict "
                    "is implied."
                ),
            },
            "source_binding": {},
            "window": {
                "start_s": float(self.settings.start),
                "stop_s": float(self.settings.stop),
                "duration_s": expected_duration,
                "endpoint_convention": "half_open_[start,stop)",
                "fundamental_period_count": int(self.settings.period_count),
                "source": "solver_dense_output_on_event_aligned_topology_intervals",
            },
            "frequency": {
                "fundamental_hz": float(self.settings.fundamental_frequency),
                "max_harmonic": int(self.settings.max_harmonic),
                "bandwidth_hz": float(
                    self.settings.fundamental_frequency * self.settings.max_harmonic
                ),
            },
            "method": {
                "name": "event_aligned_dense_output_gauss_legendre_fourier",
                "quadrature_order": _QUADRATURE_ORDER,
                "max_highest_harmonic_phase_span_per_panel_rad": _MAX_PHASE_SPAN_RAD,
                "window_function": "rectangular",
                "dc_policy": "retained_as_coefficient_0_excluded_from_thd",
                "coefficient_convention": (
                    "C_k=(1/window_duration)*integral(x(t)*exp(-j*2*pi*k*f1*t)dt)"
                ),
                "amplitude_convention": (
                    "for_k>0_peak_phasor=2*C_k_and_rms=abs(peak_phasor)/sqrt(2)"
                ),
                "topology_segment_count": self.topology_segment_count,
                "dense_output_interval_count": self.dense_output_interval_count,
                "quadrature_panel_count": self.panel_count,
                "report_recomputes_values": False,
            },
            "signals": signal_records,
            "sequence_components": _sequence_component_records(signal_records),
            "summary": {
                "signal_count": len(signal_records),
                "undefined_thd_count": sum(
                    record["thd"]["status"] != "observed"
                    for record in signal_records
                ),
                "acceptance_status": "not_evaluated",
            },
        }


def _signal_record(
    signal: str,
    semantic: str,
    coefficients: np.ndarray,
    fundamental_frequency: float,
) -> dict[str, Any]:
    coefficient_rows: list[dict[str, Any]] = []
    rms_values: list[float] = []
    for harmonic, raw_coefficient in enumerate(coefficients):
        coefficient = complex(raw_coefficient)
        if harmonic == 0:
            peak_phasor = coefficient
            peak_amplitude = abs(coefficient)
            rms_amplitude = abs(coefficient)
            phase_deg = 0.0 if peak_amplitude == 0.0 else math.degrees(cmath.phase(coefficient))
        else:
            peak_phasor = 2.0 * coefficient
            peak_amplitude = abs(peak_phasor)
            rms_amplitude = peak_amplitude / math.sqrt(2.0)
            phase_deg = (
                0.0
                if peak_amplitude == 0.0
                else math.degrees(cmath.phase(peak_phasor))
            )
        rms_values.append(float(rms_amplitude))
        coefficient_rows.append(
            {
                "harmonic": harmonic,
                "frequency_hz": harmonic * fundamental_frequency,
                "coefficient_real": float(coefficient.real),
                "coefficient_imag": float(coefficient.imag),
                "cosine_peak": float(peak_phasor.real),
                "sine_peak": float(-peak_phasor.imag),
                "peak_amplitude": float(peak_amplitude),
                "rms_amplitude": float(rms_amplitude),
                "phase_deg": float(phase_deg),
            }
        )

    fundamental_rms = rms_values[1]
    harmonic_rms = math.sqrt(sum(value * value for value in rms_values[2:]))
    thd_defined = fundamental_rms > max(1e-15, harmonic_rms * 1e-15)
    # A generic signal (DC-side ripple, an arbitrary waveform) has no declared
    # fundamental in the THD sense: normalising by whatever small 1st harmonic a
    # start-up leaves would print a meaningless ratio. Amplitudes only.
    thd_status = (
        "not_computed_generic_signal"
        if semantic == "generic"
        else ("observed" if thd_defined else "not_defined_zero_fundamental")
    )
    thd = harmonic_rms / fundamental_rms if thd_status == "observed" else None
    bandlimited_rms = math.sqrt(sum(value * value for value in rms_values))
    return {
        "signal": signal,
        "semantic": semantic,
        "unit": signal_unit(signal),
        "dc": coefficient_rows[0]["coefficient_real"],
        "fundamental": {
            key: coefficient_rows[1][key]
            for key in (
                "frequency_hz",
                "cosine_peak",
                "sine_peak",
                "peak_amplitude",
                "rms_amplitude",
                "phase_deg",
            )
        },
        "thd": {
            "status": thd_status,
            "value_fraction": float(thd) if thd is not None else None,
            "harmonic_rms": float(harmonic_rms),
            "fundamental_rms": float(fundamental_rms),
            "included_harmonics": (
                f"2..{len(coefficients) - 1}"
                if len(coefficients) > 2
                else "none"
            ),
        },
        "bandlimited_rms": float(bandlimited_rms),
        "coefficients": coefficient_rows,
    }


def _sequence_component_records(
    signals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for family in ("phase_voltage", "phase_current"):
        by_phase = {
            record["semantic"].removeprefix(f"{family}_"): record
            for record in signals
            if record["semantic"].startswith(f"{family}_")
        }
        if set(by_phase) != {"a", "b", "c"}:
            continue
        phasors = {
            phase: complex(
                record["fundamental"]["cosine_peak"],
                -record["fundamental"]["sine_peak"],
            )
            / math.sqrt(2.0)
            for phase, record in by_phase.items()
        }
        rotation = cmath.exp(2j * math.pi / 3.0)
        zero = (phasors["a"] + phasors["b"] + phasors["c"]) / 3.0
        positive = (
            phasors["a"] + rotation * phasors["b"] + rotation**2 * phasors["c"]
        ) / 3.0
        negative = (
            phasors["a"] + rotation**2 * phasors["b"] + rotation * phasors["c"]
        ) / 3.0
        positive_amplitude = abs(positive)
        records.append(
            {
                "family": family,
                "convention": "Fortescue_abc_positive_sequence",
                "rms_phasors": {
                    "zero": _complex_record(zero),
                    "positive": _complex_record(positive),
                    "negative": _complex_record(negative),
                },
                "negative_to_positive_fraction": (
                    abs(negative) / positive_amplitude
                    if positive_amplitude > 1e-15
                    else None
                ),
                "zero_to_positive_fraction": (
                    abs(zero) / positive_amplitude
                    if positive_amplitude > 1e-15
                    else None
                ),
            }
        )
    return records


def _complex_record(value: complex) -> dict[str, float]:
    return {
        "real": float(value.real),
        "imag": float(value.imag),
        "magnitude": float(abs(value)),
        "phase_deg": float(0.0 if value == 0.0 else math.degrees(cmath.phase(value))),
    }


__all__ = [
    "HARMONIC_ANALYSIS_KIND",
    "HARMONIC_ANALYSIS_SCHEMA",
    "HarmonicAccumulator",
]
