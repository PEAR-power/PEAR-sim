# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Core-material cards and their binding to inductors: the SHAPE of core loss.

A material card carries Steinmetz coefficients in declared frequency ranges,
the temperature factor ``ct(T) = ct0 - ct1*T + ct2*T^2`` (T in degC), a
mandatory provenance and an explicit validity envelope. A binding ties one
linear ``inductor`` to a card through its geometry (turns, effective area,
effective volume), so the flux density is a DERIVED, auditable quantity:

    B(t) = L * i(t) / (N * A_e)

That identity is exact for a linear inductor and nothing else, which is why
saturating and coupled inductors are refused rather than approximated.

Deliberately NOT here (declared omissions, see docs/model-format.md):
DC-bias dependence of the loss, relaxation losses during zero-voltage
intervals (the i2GSE term), minor hysteresis loops, and any feedback of the
core loss into the circuit solution — the estimate is post-processing on the
recorded current, exactly like ``loss_analysis``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pear_sim.errors import ModelError
from pear_sim.spec.controls import _reject_unknown, _require_model_text, _resolve
from pear_sim.spec.device_cards import CardProvenance, _finite_number, _parse_provenance
from pear_sim.spec.units import Quantity

__all__ = [
    "CoreLossBinding",
    "CoreLossSpec",
    "CoreMaterialCard",
    "SteinmetzRange",
    "parse_core_loss_section",
]

_MAX_RANGES = 16
_MAX_MATERIALS = 32
_MAX_BINDINGS = 64


@dataclass(frozen=True)
class SteinmetzRange:
    """``Pv = k * f^alpha * B_peak^beta * ct(T)`` valid for f in [min, max] Hz.

    ``B_peak`` is the sinusoidal AMPLITUDE (half the peak-to-peak swing), Pv
    in W/m^3, f in Hz, B in T — the convention of the MAS material database
    and of manufacturer Pv(f, B) charts.
    """

    frequency_min_hz: float
    frequency_max_hz: float
    k: float
    alpha: float
    beta: float
    ct0: float
    ct1: float
    ct2: float

    def temperature_factor(self, temperature_c: float) -> float:
        return self.ct0 - self.ct1 * temperature_c + self.ct2 * temperature_c**2

    def as_dict(self) -> dict[str, float]:
        return {
            "frequency_min_hz": self.frequency_min_hz,
            "frequency_max_hz": self.frequency_max_hz,
            "k": self.k,
            "alpha": self.alpha,
            "beta": self.beta,
            "ct0": self.ct0,
            "ct1": self.ct1,
            "ct2": self.ct2,
        }


@dataclass(frozen=True)
class CoreMaterialCard:
    id: str
    provenance: CardProvenance
    temperature_min_c: float
    temperature_max_c: float
    flux_density_peak_max_t: float
    ranges: tuple[SteinmetzRange, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.as_dict(),
            "validity": {
                "temperature_min_c": self.temperature_min_c,
                "temperature_max_c": self.temperature_max_c,
                "flux_density_peak_max_t": self.flux_density_peak_max_t,
            },
            "steinmetz_ranges": [item.as_dict() for item in self.ranges],
        }


@dataclass(frozen=True)
class CoreLossBinding:
    inductor_id: str
    material_id: str
    turns: float
    effective_area_m2: float
    effective_volume_m3: float


@dataclass(frozen=True)
class CoreLossSpec:
    evaluation_temperature_c: float
    period_s: float
    materials: dict[str, CoreMaterialCard]
    bindings: tuple[CoreLossBinding, ...]


def _positive(value: Any, *, name: str) -> float:
    number = _finite_number(value, name=name)
    if number <= 0.0:
        raise ModelError(f"{name} must be positive")
    return number


def _parse_range(raw: Any, *, name: str) -> SteinmetzRange:
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    _reject_unknown(
        raw,
        {"frequency_min_hz", "frequency_max_hz", "k", "alpha", "beta", "ct0", "ct1", "ct2"},
        name=name,
    )
    for required in ("frequency_min_hz", "frequency_max_hz", "k", "alpha", "beta"):
        if required not in raw:
            raise ModelError(f"{name}.{required} is required")
    low = _finite_number(raw["frequency_min_hz"], name=f"{name}.frequency_min_hz")
    high = _positive(raw["frequency_max_hz"], name=f"{name}.frequency_max_hz")
    if low < 0.0 or high <= low:
        raise ModelError(f"{name}: need 0 <= frequency_min_hz < frequency_max_hz")
    k = _positive(raw["k"], name=f"{name}.k")
    alpha = _finite_number(raw["alpha"], name=f"{name}.alpha")
    beta = _finite_number(raw["beta"], name=f"{name}.beta")
    # The iGSE coefficient ki needs alpha > 0, and beta < alpha would make the
    # loss DECREASE with swing at fixed dB/dt — no ferrite does that; such a
    # fit is a data error, not a material.
    if not 0.0 < alpha < 4.0 or not 0.0 < beta < 5.0 or beta < alpha:
        raise ModelError(
            f"{name}: Steinmetz exponents out of the physical band "
            "(need 0 < alpha < 4, 0 < beta < 5, beta >= alpha)"
        )
    # Missing temperature coefficients mean "no temperature dependence"
    # (ct = 1), the MAS schema default — stated here so it is not a surprise.
    ct0 = _finite_number(raw.get("ct0", 1.0), name=f"{name}.ct0")
    ct1 = _finite_number(raw.get("ct1", 0.0), name=f"{name}.ct1")
    ct2 = _finite_number(raw.get("ct2", 0.0), name=f"{name}.ct2")
    return SteinmetzRange(low, high, k, alpha, beta, ct0, ct1, ct2)


def _parse_material(material_id: str, raw: Any) -> CoreMaterialCard:
    name = f"core_losses.materials.{material_id}"
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    _reject_unknown(raw, {"provenance", "validity", "steinmetz_ranges"}, name=name)
    provenance = _parse_provenance(raw.get("provenance"), name=name)
    validity = raw.get("validity")
    if not isinstance(validity, dict):
        raise ModelError(f"{name}.validity is required and must be an object")
    _reject_unknown(
        validity,
        {"temperature_min_c", "temperature_max_c", "flux_density_peak_max_t", "notes"},
        name=f"{name}.validity",
    )
    for required in ("temperature_min_c", "temperature_max_c", "flux_density_peak_max_t"):
        if required not in validity:
            raise ModelError(f"{name}.validity.{required} is required")
    t_min = _finite_number(validity["temperature_min_c"], name=f"{name}.validity.temperature_min_c")
    t_max = _finite_number(validity["temperature_max_c"], name=f"{name}.validity.temperature_max_c")
    if t_max <= t_min:
        raise ModelError(f"{name}.validity: temperature_max_c must exceed temperature_min_c")
    b_max = _positive(
        validity["flux_density_peak_max_t"], name=f"{name}.validity.flux_density_peak_max_t"
    )
    raw_ranges = raw.get("steinmetz_ranges")
    if not isinstance(raw_ranges, list) or not 1 <= len(raw_ranges) <= _MAX_RANGES:
        raise ModelError(f"{name}.steinmetz_ranges must list 1..{_MAX_RANGES} ranges")
    ranges = tuple(
        _parse_range(item, name=f"{name}.steinmetz_ranges[{index}]")
        for index, item in enumerate(raw_ranges)
    )
    ordered = sorted(ranges, key=lambda item: item.frequency_min_hz)
    if list(ordered) != list(ranges):
        raise ModelError(f"{name}.steinmetz_ranges must be ordered by frequency")
    for left, right in zip(ranges, ranges[1:]):
        if right.frequency_min_hz < left.frequency_max_hz:
            raise ModelError(f"{name}.steinmetz_ranges overlap")
    # The temperature factor must stay positive over the declared validity
    # envelope: a fit whose ct(T) crosses zero there would turn into a
    # negative or zero loss, which is a broken card, not a cold core.
    for index, item in enumerate(ranges):
        samples = [t_min + (t_max - t_min) * step / 64.0 for step in range(65)]
        if any(item.temperature_factor(value) <= 0.0 for value in samples):
            raise ModelError(
                f"{name}.steinmetz_ranges[{index}]: ct(T) is not positive over "
                "the declared validity temperatures"
            )
    return CoreMaterialCard(
        id=material_id,
        provenance=provenance,
        temperature_min_c=t_min,
        temperature_max_c=t_max,
        flux_density_peak_max_t=b_max,
        ranges=ranges,
    )


def _parse_binding(inductor_id: str, raw: Any, materials: dict[str, CoreMaterialCard]) -> CoreLossBinding:
    name = f"core_losses.inductors.{inductor_id}"
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    _reject_unknown(
        raw, {"material", "turns", "effective_area_m2", "effective_volume_m3"}, name=name
    )
    material_id = _require_model_text(raw.get("material"), name=f"{name}.material")
    if material_id not in materials:
        raise ModelError(f"{name}.material {material_id!r} is not a declared core material")
    turns = _positive(raw.get("turns"), name=f"{name}.turns")
    area = _positive(raw.get("effective_area_m2"), name=f"{name}.effective_area_m2")
    volume = _positive(raw.get("effective_volume_m3"), name=f"{name}.effective_volume_m3")
    return CoreLossBinding(inductor_id, material_id, turns, area, volume)


def parse_core_loss_section(
    raw: Any,
    parameters: dict[str, Quantity],
) -> CoreLossSpec | None:
    """Parse the optional ``core_losses`` section; fail closed on anything odd."""

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ModelError("core_losses must be an object")
    _reject_unknown(
        raw, {"evaluation_temperature_c", "period", "materials", "inductors"}, name="core_losses"
    )
    if "evaluation_temperature_c" not in raw:
        raise ModelError("core_losses.evaluation_temperature_c is required")
    temperature = _finite_number(
        raw["evaluation_temperature_c"], name="core_losses.evaluation_temperature_c"
    )
    if "period" not in raw:
        raise ModelError("core_losses.period is required (one period of the flux waveform)")
    period = _resolve(raw["period"], parameters, "time", name="core_losses.period")
    if not period > 0.0:
        raise ModelError("core_losses.period must be positive")
    raw_materials = raw.get("materials")
    if not isinstance(raw_materials, dict) or not 1 <= len(raw_materials) <= _MAX_MATERIALS:
        raise ModelError(f"core_losses.materials must declare 1..{_MAX_MATERIALS} materials")
    materials = {
        _require_model_text(key, name="core_losses.materials key"): _parse_material(key, value)
        for key, value in raw_materials.items()
    }
    raw_bindings = raw.get("inductors")
    if not isinstance(raw_bindings, dict) or not 1 <= len(raw_bindings) <= _MAX_BINDINGS:
        raise ModelError(f"core_losses.inductors must bind 1..{_MAX_BINDINGS} inductors")
    bindings = tuple(
        _parse_binding(_require_model_text(key, name="core_losses.inductors key"), value, materials)
        for key, value in raw_bindings.items()
    )
    unused = sorted(set(materials) - {binding.material_id for binding in bindings})
    if unused:
        raise ModelError(f"core_losses.materials declared but never bound: {unused}")
    for binding in bindings:
        card = materials[binding.material_id]
        if not card.temperature_min_c <= temperature <= card.temperature_max_c:
            raise ModelError(
                f"core_losses.evaluation_temperature_c={temperature} lies outside the "
                f"validity envelope of material {card.id!r} "
                f"[{card.temperature_min_c}, {card.temperature_max_c}] degC"
            )
    return CoreLossSpec(
        evaluation_temperature_c=temperature,
        period_s=period,
        materials=materials,
        bindings=bindings,
    )
