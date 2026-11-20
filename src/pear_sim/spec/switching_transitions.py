# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""The optional ``switching_transitions`` section: Coss cards and leg bindings.

The section DESCRIBES a post-processing check on the recorded ideal run; it is
never an input to the circuit equations (the kernel stays ideal, see
docs/specs/2026-11-20-switching-transition.md). It carries its own
output-capacitance cards rather than extending ``device_cards``: device cards
are all-or-nothing with ``loss_analysis`` and refuse components that declare
``forward_voltage`` — which is exactly how GaN reverse conduction is described.

Card interpolation laws (declared, documented in docs/model-format.md §7c):
``coss_f`` — Coss piecewise LINEAR in V between table points;
``qoss_c`` — Qoss piecewise linear in V, i.e. Coss piecewise CONSTANT.
Above the last axis point the last capacitance is held and the lookup is
COUNTED as an extrapolation by the producer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pear_sim.errors import ModelError
from pear_sim.spec.controls import _reject_unknown, _require_model_text, _resolve
from pear_sim.spec.device_cards import (
    CardProvenance,
    _number_list,
    _parse_provenance,
    _strictly_increasing,
)
from pear_sim.spec.units import Quantity

__all__ = [
    "CossCard",
    "TransitionLeg",
    "SwitchingTransitionSpec",
    "parse_switching_transition_section",
]

_MAX_CARDS = 8
_MAX_LEGS = 12
_MAX_FAR_TERMS = 6
DEFAULT_ZVS_THRESHOLD = 0.01


@dataclass(frozen=True)
class CossCard:
    """Datasheet output capacitance as Coss(V) or Qoss(V); axis starts at 0 V."""

    id: str
    voltage_axis_v: tuple[float, ...]
    coss_f: tuple[float, ...] | None
    qoss_c: tuple[float, ...] | None
    provenance: CardProvenance

    @property
    def law(self) -> str:
        return "coss_piecewise_linear" if self.coss_f is not None else "qoss_piecewise_linear"


@dataclass(frozen=True)
class TransitionLeg:
    id: str
    high: str
    low: str
    high_diode: str
    low_diode: str
    card_id: str
    bus_voltage: tuple[tuple[str, float], ...]
    extra_node_capacitance_f: float
    commutation_inductance_h: float | None
    far_voltage: tuple[tuple[str, float], ...]
    loop_inductance_h: float | None
    current_fall_time_s: float | None
    bridge_partner: str | None = None

    @property
    def regime(self) -> str:
        return "constant_current" if self.commutation_inductance_h is None else "lumped_lc"

    def required_outputs(self) -> tuple[str, ...]:
        names = [
            f"i({self.high})",
            f"i({self.low})",
            f"i({self.high_diode})",
            f"i({self.low_diode})",
        ]
        names += [term for term, _ in self.bus_voltage]
        names += [term for term, _ in self.far_voltage]
        return tuple(dict.fromkeys(names))


@dataclass(frozen=True)
class SwitchingTransitionSpec:
    zvs_threshold: float
    cards: dict[str, CossCard]
    legs: tuple[TransitionLeg, ...]


def _parse_card(card_id: str, raw: Any) -> CossCard:
    name = f"switching_transitions.coss_cards.{card_id}"
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    _reject_unknown(raw, {"voltage_axis_v", "coss_f", "qoss_c", "provenance"}, name=name)
    axis = _number_list(raw.get("voltage_axis_v"), name=f"{name}.voltage_axis_v")
    _strictly_increasing(axis, name=f"{name}.voltage_axis_v")
    if axis[0] != 0.0:
        raise ModelError(
            f"{name}.voltage_axis_v must start at 0 V: the transition integrates the "
            "capacitance from zero, and an extrapolation below the first point would "
            "be a guess"
        )
    has_coss, has_qoss = raw.get("coss_f") is not None, raw.get("qoss_c") is not None
    if has_coss == has_qoss:
        raise ModelError(f"{name} must declare exactly one of coss_f or qoss_c")
    field = "coss_f" if has_coss else "qoss_c"
    values = _number_list(raw.get(field), name=f"{name}.{field}", minimum_len=len(axis))
    if len(values) != len(axis):
        raise ModelError(f"{name}.{field} must match the voltage axis length")
    if has_coss:
        if any(value <= 0.0 for value in values):
            raise ModelError(f"{name}.coss_f must be positive")
    else:
        if values[0] != 0.0:
            raise ModelError(f"{name}.qoss_c must be 0 at 0 V")
        _strictly_increasing(values, name=f"{name}.qoss_c")
    return CossCard(
        id=card_id,
        voltage_axis_v=axis,
        coss_f=values if has_coss else None,
        qoss_c=None if has_coss else values,
        provenance=_parse_provenance(raw.get("provenance"), name=name),
    )


def _voltage_terms(raw: Any, *, name: str) -> tuple[tuple[str, float], ...]:
    """A recorded v(...) output, or a {v(...): coefficient} linear combination."""

    if isinstance(raw, str):
        raw = {raw: 1.0}
    if not isinstance(raw, dict) or not 1 <= len(raw) <= _MAX_FAR_TERMS:
        raise ModelError(
            f"{name} must be a recorded v(...) output or map 1..{_MAX_FAR_TERMS} of them "
            "to numeric coefficients"
        )
    terms = []
    for term, coefficient in raw.items():
        term = _require_model_text(term, name=f"{name} key")
        if not term.startswith("v("):
            raise ModelError(f"{name}: {term!r} must be a v(...) output")
        if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
            raise ModelError(f"{name}[{term!r}] must be a number")
        if coefficient != coefficient or abs(coefficient) == float("inf"):
            raise ModelError(f"{name}[{term!r}] must be finite")
        terms.append((term, float(coefficient)))
    return tuple(terms)


def _optional_positive(
    raw: dict[str, Any],
    key: str,
    parameters: dict[str, Quantity],
    dimension: str,
    *,
    name: str,
) -> float | None:
    if raw.get(key) is None:
        return None
    value = _resolve(raw[key], parameters, dimension, name=f"{name}.{key}")
    if not value > 0.0:
        raise ModelError(f"{name}.{key} must be positive")
    return value


def _parse_leg(
    leg_id: str,
    raw: Any,
    parameters: dict[str, Quantity],
    cards: dict[str, CossCard],
) -> TransitionLeg:
    name = f"switching_transitions.legs.{leg_id}"
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    _reject_unknown(
        raw,
        {
            "high",
            "low",
            "high_diode",
            "low_diode",
            "card",
            "bus_voltage",
            "extra_node_capacitance",
            "commutation_inductance",
            "far_voltage",
            "loop_inductance",
            "current_fall_time",
            "bridge_partner",
        },
        name=name,
    )
    high = _require_model_text(raw.get("high"), name=f"{name}.high")
    low = _require_model_text(raw.get("low"), name=f"{name}.low")
    # Both antiparallel diodes are required: without a reverse-conduction path
    # the ideal run cuts the commutating current in the dead time, and the
    # recorded waveform the transition is evaluated on is not a usable basis.
    high_diode = _require_model_text(raw.get("high_diode"), name=f"{name}.high_diode")
    low_diode = _require_model_text(raw.get("low_diode"), name=f"{name}.low_diode")
    card_id = _require_model_text(raw.get("card"), name=f"{name}.card")
    if card_id not in cards:
        raise ModelError(f"{name}.card {card_id!r} is not a declared coss card")
    bus_voltage = _voltage_terms(raw.get("bus_voltage"), name=f"{name}.bus_voltage")
    extra = 0.0
    if raw.get("extra_node_capacitance") is not None:
        extra = _resolve(
            raw["extra_node_capacitance"],
            parameters,
            "capacitance",
            name=f"{name}.extra_node_capacitance",
        )
        if extra < 0.0:
            raise ModelError(f"{name}.extra_node_capacitance must be non-negative")
    commutation = _optional_positive(
        raw, "commutation_inductance", parameters, "inductance", name=name
    )
    far_raw = raw.get("far_voltage")
    if (commutation is None) != (far_raw is None):
        raise ModelError(
            f"{name}: commutation_inductance and far_voltage go together — the lumped "
            "LC transition needs the voltage at the inductor's far terminal; omit both "
            "for the constant-current transition"
        )
    far: tuple[tuple[str, float], ...] = ()
    if far_raw is not None:
        far = _voltage_terms(far_raw, name=f"{name}.far_voltage")
    loop = _optional_positive(raw, "loop_inductance", parameters, "inductance", name=name)
    fall = _optional_positive(raw, "current_fall_time", parameters, "time", name=name)
    if fall is not None and loop is None:
        raise ModelError(f"{name}.current_fall_time needs loop_inductance")
    return TransitionLeg(
        id=leg_id,
        high=high,
        low=low,
        high_diode=high_diode,
        low_diode=low_diode,
        card_id=card_id,
        bus_voltage=bus_voltage,
        extra_node_capacitance_f=extra,
        commutation_inductance_h=commutation,
        far_voltage=far,
        loop_inductance_h=loop,
        current_fall_time_s=fall,
        bridge_partner=(
            None
            if raw.get("bridge_partner") is None
            else _require_model_text(raw["bridge_partner"], name=f"{name}.bridge_partner")
        ),
    )


def parse_switching_transition_section(
    raw: Any,
    parameters: dict[str, Quantity],
) -> SwitchingTransitionSpec | None:
    """Parse the optional section; fail closed on anything odd."""

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ModelError("switching_transitions must be an object")
    _reject_unknown(raw, {"zvs_threshold", "coss_cards", "legs"}, name="switching_transitions")
    threshold = DEFAULT_ZVS_THRESHOLD
    if raw.get("zvs_threshold") is not None:
        threshold = _resolve(
            raw["zvs_threshold"],
            parameters,
            "dimensionless",
            name="switching_transitions.zvs_threshold",
        )
        if not 0.0 < threshold < 0.5:
            raise ModelError(
                "switching_transitions.zvs_threshold must lie in (0, 50 %) of the bus voltage"
            )
    raw_cards = raw.get("coss_cards")
    if not isinstance(raw_cards, dict) or not 1 <= len(raw_cards) <= _MAX_CARDS:
        raise ModelError(f"switching_transitions.coss_cards must declare 1..{_MAX_CARDS} cards")
    cards = {
        _require_model_text(key, name="switching_transitions.coss_cards key"): _parse_card(key, value)
        for key, value in raw_cards.items()
    }
    raw_legs = raw.get("legs")
    if not isinstance(raw_legs, dict) or not 1 <= len(raw_legs) <= _MAX_LEGS:
        raise ModelError(f"switching_transitions.legs must declare 1..{_MAX_LEGS} legs")
    legs = tuple(
        _parse_leg(
            _require_model_text(key, name="switching_transitions.legs key"),
            value,
            parameters,
            cards,
        )
        for key, value in raw_legs.items()
    )
    unused = sorted(set(cards) - {leg.card_id for leg in legs})
    if unused:
        raise ModelError(f"switching_transitions.coss_cards declared but never bound: {unused}")
    seen: dict[str, str] = {}
    for leg in legs:
        for device in (leg.high, leg.low):
            if device in seen:
                raise ModelError(
                    f"switching_transitions: switch {device!r} is bound in legs "
                    f"{seen[device]!r} and {leg.id!r}"
                )
            seen[device] = leg.id
        if leg.high == leg.low:
            raise ModelError(f"switching_transitions.legs.{leg.id}: high and low are the same switch")
    by_id = {leg.id: leg for leg in legs}
    partnered: dict[str, str] = {}
    for leg in legs:
        if leg.bridge_partner is None:
            continue
        name = f"switching_transitions.legs.{leg.id}.bridge_partner"
        partner = by_id.get(leg.bridge_partner)
        if partner is None or partner.id == leg.id:
            raise ModelError(f"{name} must name another declared leg")
        if partner.bridge_partner is not None:
            raise ModelError(
                f"{name}: {partner.id!r} declares its own bridge_partner; declare the "
                "pair once, on one leg"
            )
        if partner.id in partnered:
            raise ModelError(f"{name}: {partner.id!r} is already the partner of {partnered[partner.id]!r}")
        if partner.commutation_inductance_h is not None or partner.far_voltage:
            raise ModelError(
                f"{name}: the commutation path is declared once, on {leg.id!r}; remove "
                f"commutation_inductance/far_voltage from {partner.id!r}"
            )
        if partner.bus_voltage != leg.bus_voltage:
            raise ModelError(f"{name}: both legs of a bridge must share bus_voltage")
        partnered[partner.id] = leg.id
    return SwitchingTransitionSpec(zvs_threshold=threshold, cards=cards, legs=legs)
