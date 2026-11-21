# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Core-loss post-processing: the improved Generalized Steinmetz Equation.

For each bound inductor the flux density over the LAST declared period of the
recorded run is ``B(t) = L*i(t)/(N*A_e)`` and the volumetric loss is the iGSE
(J. Li, T. Abdallah, C. R. Sullivan, "Improved calculation of core loss with
nonsinusoidal waveforms", IEEE IAS 2001):

    Pv = (1/T) * integral_0^T ki * |dB/dt|^alpha * dB_pp^(beta-alpha) dt

    ki = k / ( (2*pi)^(alpha-1) * 2^(beta-alpha) * integral_0^2pi |cos x|^alpha dx )

with the |cos|^alpha integral in CLOSED FORM, 2*sqrt(pi)*Gamma((alpha+1)/2) /
Gamma(alpha/2 + 1), so for a sinusoid the iGSE reproduces the Steinmetz
equation exactly (asserted in the tests) — no numerical quadrature constant.

The integral is evaluated on the recorded samples as a piecewise-linear B(t),
which is exact when every waveform corner is a sample and otherwise carries a
grid error that the artifact makes visible through ``samples_per_period``.

Scope (fail closed, per inductor): one MAJOR loop per period. A period with
more than one rise and one fall holds minor loops, which iGSE as written does
not split; that inductor is reported ``not_estimated`` with the reason instead
of being given a number. Nothing here feeds back into the circuit solution.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pear_sim.errors import ModelError
from pear_sim.spec.core_loss_cards import CoreLossBinding, CoreMaterialCard, SteinmetzRange

__all__ = [
    "CORE_LOSSES_SCHEMA",
    "CoreLossError",
    "build_core_losses_observation",
    "igse_ki",
    "igse_volumetric_loss",
]

CORE_LOSSES_SCHEMA = "pear-sim/core-losses/v0.1"
# Below this many samples per period the |dB/dt|^alpha integral is dominated by
# where the grid happens to fall relative to the waveform corners.
MIN_SAMPLES_PER_PERIOD = 64
# B(T) vs B(0) mismatch, relative to the swing, above which the window is not
# one period of a periodic waveform (the run has not settled).
PERIODIC_CLOSURE_TOLERANCE = 1e-3
# Sample-to-sample changes below this fraction of the swing are flat, not a
# direction: round-off on a flat segment must not count as a minor loop.
_FLAT_FRACTION = 1e-9

DECLARED_OMISSIONS = (
    "DC-bias dependence of the loss (B_dc is recorded, not used)",
    "relaxation loss during zero-voltage intervals (i2GSE term)",
    "minor hysteresis loops (refused, not approximated)",
    "feedback of core loss into the circuit solution (post-processing only)",
    "temperature rise of the core (evaluated at a declared temperature)",
)


class CoreLossError(ModelError):
    """The run cannot support the requested core-loss estimate."""

    code = "analysis.core_losses_invalid"


def _cos_power_integral(alpha: float) -> float:
    """integral_0^{2 pi} |cos x|^alpha dx, in closed form."""

    return 2.0 * math.sqrt(math.pi) * math.gamma((alpha + 1.0) / 2.0) / math.gamma(alpha / 2.0 + 1.0)


def igse_ki(k: float, alpha: float, beta: float) -> float:
    return k / (
        (2.0 * math.pi) ** (alpha - 1.0) * 2.0 ** (beta - alpha) * _cos_power_integral(alpha)
    )


def _direction_changes(flux: np.ndarray, swing: float) -> int:
    steps = np.diff(flux)
    signs = np.sign(steps[np.abs(steps) > _FLAT_FRACTION * swing])
    if signs.size == 0:
        return 0
    changes = int(np.count_nonzero(signs[1:] != signs[:-1]))
    # Cyclic: the period wraps, so a change between the last and first slope
    # is a change too. A clean major loop has exactly two.
    if signs[-1] != signs[0]:
        changes += 1
    return changes


def igse_volumetric_loss(
    time_s: np.ndarray,
    flux_density_t: np.ndarray,
    *,
    k: float,
    alpha: float,
    beta: float,
) -> dict[str, Any]:
    """iGSE over samples spanning EXACTLY one period (first to last sample).

    Returns the loss WITHOUT the temperature factor, plus the facts the
    caller needs to judge it. Raises :class:`CoreLossError` for minor loops
    or a degenerate window.
    """

    time = np.asarray(time_s, dtype=float)
    flux = np.asarray(flux_density_t, dtype=float)
    if time.shape != flux.shape or time.ndim != 1 or time.size < 3:
        raise CoreLossError("iGSE needs matching one-dimensional time and flux arrays")
    if not (np.all(np.isfinite(time)) and np.all(np.isfinite(flux))):
        raise CoreLossError("iGSE needs finite time and flux samples")
    steps = np.diff(time)
    if np.any(steps <= 0.0):
        raise CoreLossError("iGSE needs strictly increasing time samples")
    period = float(time[-1] - time[0])
    swing = float(flux.max() - flux.min())
    if swing == 0.0:
        return {"volumetric_loss_w_per_m3": 0.0, "delta_b_t": 0.0, "direction_changes": 0}
    changes = _direction_changes(flux, swing)
    if changes > 2:
        raise CoreLossError(
            f"the flux waveform reverses direction {changes} times per period: minor "
            "loops, which this iGSE (one major loop per period) does not split"
        )
    slope_term = float(np.sum(np.abs(np.diff(flux) / steps) ** alpha * steps))
    loss = igse_ki(k, alpha, beta) * swing ** (beta - alpha) * slope_term / period
    return {"volumetric_loss_w_per_m3": loss, "delta_b_t": swing, "direction_changes": changes}


def _select_range(card: CoreMaterialCard, frequency_hz: float) -> tuple[SteinmetzRange, float]:
    """Range containing f, else the one whose edge is nearest, plus the signed distance.

    The distance is ``f - edge`` to the nearest edge OF THE RETURNED RANGE
    (negative below it, positive above it), so the number and the coefficients
    describe the same decision. A card may leave a gap between ranges: inside
    one, the nearer side wins, and the lower range on an exact tie.
    """

    best: tuple[SteinmetzRange, float] | None = None
    for item in card.ranges:
        if item.frequency_min_hz <= frequency_hz <= item.frequency_max_hz:
            return item, 0.0
        distance = (
            frequency_hz - item.frequency_min_hz
            if frequency_hz < item.frequency_min_hz
            else frequency_hz - item.frequency_max_hz
        )
        if best is None or abs(distance) < abs(best[1]):
            best = (item, distance)
    assert best is not None  # a card always carries at least one range
    return best


def _one_period(time: np.ndarray, values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    stop = float(time[-1])
    start = stop - period
    inside = time > start
    window_time = np.concatenate(([start], time[inside]))
    window_values = np.concatenate(([float(np.interp(start, time, values))], values[inside]))
    return window_time, window_values


def _estimate_inductor(
    binding: CoreLossBinding,
    card: CoreMaterialCard,
    inductance_h: float,
    window_time: np.ndarray,
    window_current: np.ndarray,
    *,
    period: float,
    temperature_c: float,
) -> dict[str, Any]:
    flux = inductance_h * window_current / (binding.turns * binding.effective_area_m2)
    swing = float(flux.max() - flux.min())
    record: dict[str, Any] = {
        "inductor": binding.inductor_id,
        "material": card.id,
        "material_card": card.as_dict(),
        "geometry": {
            "turns": binding.turns,
            "effective_area_m2": binding.effective_area_m2,
            "effective_volume_m3": binding.effective_volume_m3,
            "inductance_h": inductance_h,
        },
        "flux_density": {
            "delta_b_t": swing,
            "b_ac_peak_t": swing / 2.0,
            "b_dc_t": float(0.5 * (flux.max() + flux.min())),
            "b_abs_max_t": float(np.abs(flux).max()),
            "periodic_closure_relative": (
                float(abs(flux[-1] - flux[0]) / swing) if swing > 0.0 else 0.0
            ),
        },
    }
    flags: list[dict[str, Any]] = []
    frequency = 1.0 / period
    selected, distance_hz = _select_range(card, frequency)
    if distance_hz != 0.0:
        flags.append(
            {
                "code": "FREQUENCY_OUTSIDE_RANGES",
                "frequency_hz": frequency,
                "signed_distance_hz": distance_hz,
                "message": "no Steinmetz range covers 1/period; the nearest range was used",
            }
        )
    if record["flux_density"]["b_abs_max_t"] > card.flux_density_peak_max_t:
        flags.append(
            {
                "code": "FLUX_DENSITY_ABOVE_VALIDITY",
                "b_abs_max_t": record["flux_density"]["b_abs_max_t"],
                "limit_t": card.flux_density_peak_max_t,
                "message": "max |B| exceeds the card's validity envelope",
            }
        )
    if record["flux_density"]["periodic_closure_relative"] > PERIODIC_CLOSURE_TOLERANCE:
        flags.append(
            {
                "code": "WINDOW_NOT_PERIODIC",
                "closure_relative": record["flux_density"]["periodic_closure_relative"],
                "tolerance": PERIODIC_CLOSURE_TOLERANCE,
                "message": (
                    "B at the end of the window differs from B at its start: the run "
                    "has not reached periodic steady state or the period is wrong"
                ),
            }
        )
    record["steinmetz_range"] = selected.as_dict()
    try:
        igse = igse_volumetric_loss(
            window_time, flux, k=selected.k, alpha=selected.alpha, beta=selected.beta
        )
    except CoreLossError as error:
        record.update(status="not_estimated", reason=str(error), flags=flags)
        return record
    factor = selected.temperature_factor(temperature_c)
    volumetric = igse["volumetric_loss_w_per_m3"] * factor
    record.update(
        status="estimated",
        direction_changes_per_period=igse["direction_changes"],
        ki=igse_ki(selected.k, selected.alpha, selected.beta),
        temperature_factor=factor,
        volumetric_loss_w_per_m3=volumetric,
        power_w=volumetric * binding.effective_volume_m3,
        flags=flags,
    )
    return record


def build_core_losses_observation(result: Any) -> dict[str, Any]:
    model = result.model
    spec = getattr(model, "core_losses", None)
    if spec is None:
        raise CoreLossError("model declares no core_losses section")
    time = np.asarray(result.time, dtype=float)
    period = spec.period_s
    if time.size < 3 or float(time[-1] - time[0]) < period:
        raise CoreLossError("the recorded run is shorter than one core_losses.period")
    samples = int(np.count_nonzero(time > time[-1] - period)) + 1
    if samples < MIN_SAMPLES_PER_PERIOD:
        raise CoreLossError(
            f"only {samples} recorded samples in the last period; core loss needs at "
            f"least {MIN_SAMPLES_PER_PERIOD} (reduce analysis.transient.output_step)"
        )
    component_by_id = {component.id: component for component in model.components}
    inductors: list[dict[str, Any]] = []
    window_time: np.ndarray | None = None
    for binding in spec.bindings:
        signal = f"i({binding.inductor_id})"
        if signal not in result.signals:
            raise CoreLossError(f"core loss requires {signal} among the recorded outputs")
        current = np.asarray(result.signals[signal], dtype=float)
        window_time, window_current = _one_period(time, current, period)
        inductance = component_by_id[binding.inductor_id].value
        inductors.append(
            _estimate_inductor(
                binding,
                spec.materials[binding.material_id],
                float(inductance),
                window_time,
                window_current,
                period=period,
                temperature_c=spec.evaluation_temperature_c,
            )
        )
    estimated = [item for item in inductors if item["status"] == "estimated"]
    warnings = [
        {
            "code": "CORE_LOSS_NOT_ESTIMATED",
            "inductor": item["inductor"],
            "message": item["reason"],
        }
        for item in inductors
        if item["status"] != "estimated"
    ]
    warnings.extend(
        {"code": flag["code"], "inductor": item["inductor"], "message": flag["message"]}
        for item in inductors
        for flag in item.get("flags", [])
    )
    assert window_time is not None
    return {
        "schema": CORE_LOSSES_SCHEMA,
        "claim": (
            "Post-processing ESTIMATE: iGSE on the recorded inductor current over the "
            "last declared period, with Steinmetz coefficients from a declared material "
            "card at a declared core temperature. Not a measurement; accuracy is bounded "
            "by the card and by the declared omissions."
        ),
        "method": "iGSE (Li, Abdallah, Sullivan, IEEE IAS 2001); B = L*i/(N*Ae)",
        "evaluation_temperature_c": spec.evaluation_temperature_c,
        "period_s": period,
        "window": {
            "start_s": float(window_time[0]),
            "stop_s": float(window_time[-1]),
            "samples_per_period": int(window_time.size),
        },
        "inductors": inductors,
        "total_power_w": (
            float(sum(item["power_w"] for item in estimated))
            if len(estimated) == len(inductors)
            else None
        ),
        "declared_omissions": list(DECLARED_OMISSIONS),
        "warnings": warnings,
    }
