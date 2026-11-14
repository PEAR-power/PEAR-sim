# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Series parasitics of passive components: capacitor ESR and inductor DCR.

A declared ``esr`` or ``dcr`` EXPANDS source-to-source into an ordinary resistor
in series with the component, before semantic validation (the machine and
subcircuit precedent). The kernel never sees a new element type: the expanded
resistor is stamped, audited and reported like any other resistor, which is
why the energy balance closes without a special case.

Naming. For a capacitor ``c1`` with ``esr``::

    p ──[c1.esr]── c1.core ──| c1 |── n

and for an inductor ``l1`` with ``dcr``::

    p ──( l1 )── l1.core ──[l1.dcr]── n

The user's terminals do not move: ``v(out)`` is still the terminal voltage a
probe would read, including the drop across the ESR. ``initial_voltage`` and
``initial_current`` keep describing the state variable, i.e. the ideal
capacitance (``c1.core`` to ``n``) and the inductor current. Top-level ids
cannot contain ``.``, so the generated ids cannot collide with a user's; a
user NODE literally named ``c1.core`` is rejected rather than silently joined.

Declared limits: one frequency-independent value per component (state it at
the frequency that matters, usually f_sw). No leakage resistance, no ESL.

Audit. Every expansion is recorded as a :class:`PassiveParasitic`, so the
energy audit can group the expanded resistor's dissipation under the user's
component and derive a capacitor's RMS current from it exactly
(I_rms = sqrt(E_esr / (ESR * T)): the ESR carries the capacitor current).
That identity is why ``ripple_current_rating`` requires an ``esr``: without
one there is no resistor whose energy measures the current.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pear_sim.errors import ModelError
from pear_sim.spec.controls import _resolve
from pear_sim.spec.units import Quantity

__all__ = [
    "PARASITIC_FIELDS",
    "RIPPLE_CURRENT_RATING",
    "PassiveParasitic",
    "expand_passive_parasitics",
]

#: Parasitic field -> component types that accept it.
PARASITIC_FIELDS: dict[str, frozenset[str]] = {
    "esr": frozenset({"capacitor"}),
    "dcr": frozenset({"inductor", "sat_inductor"}),
}


#: Optional thermal limit on a capacitor's RMS current; needs an ``esr``.
RIPPLE_CURRENT_RATING = "ripple_current_rating"


@dataclass(frozen=True)
class PassiveParasitic:
    """One expanded series parasitic: which resistor came from which component."""

    component_id: str
    kind: str  # "esr" | "dcr"
    resistor_id: str
    ripple_current_rating_a: float | None = None

    def as_dict(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "component": self.component_id,
            "kind": self.kind,
            "resistor": self.resistor_id,
        }
        if self.ripple_current_rating_a is not None:
            record["ripple_current_rating_a"] = self.ripple_current_rating_a
        return record


def _check_value(
    raw_value: Any,
    parameters: dict[str, Quantity],
    *,
    name: str,
) -> None:
    value = _resolve(raw_value, parameters, "resistance", name=name)
    if not math.isfinite(value) or value <= 0.0:
        raise ModelError(
            f"{name} must be a positive, finite resistance; omit the field for an "
            "ideal component instead of writing zero"
        )


def _ripple_rating(raw: dict[str, Any], parameters: dict[str, Quantity]) -> float | None:
    if RIPPLE_CURRENT_RATING not in raw:
        return None
    component_id = raw.get("id")
    name = f"{component_id}.{RIPPLE_CURRENT_RATING}"
    if raw.get("type") != "capacitor":
        raise ModelError(f"{name}: only valid for capacitor, not {raw.get('type')!r}")
    if "esr" not in raw:
        raise ModelError(
            f"{name}: needs an `esr` on the same capacitor; the RMS current is "
            "measured from the energy the ESR dissipates, and the rating is a "
            "limit on exactly that heating"
        )
    value = _resolve(raw[RIPPLE_CURRENT_RATING], parameters, "current", name=name)
    if not math.isfinite(value) or value <= 0.0:
        raise ModelError(f"{name} must be a positive, finite current")
    return float(value)


def _declares(raw: Any) -> bool:
    return isinstance(raw, dict) and (
        any(field in raw for field in PARASITIC_FIELDS) or RIPPLE_CURRENT_RATING in raw
    )


def expand_passive_parasitics(
    raw_components: list[Any],
    parameters: dict[str, Quantity],
) -> tuple[list[Any], tuple[PassiveParasitic, ...]]:
    """Expand every ``esr``/``dcr`` in series; return components and records.

    Components without a parasitic field pass through as the same object, so a
    model that declares none is unchanged byte for byte (and has no records).
    """

    if not any(_declares(raw) for raw in raw_components):
        return raw_components, ()

    user_nodes = {
        raw.get(terminal)
        for raw in raw_components
        if isinstance(raw, dict)
        for terminal in ("p", "n")
    }
    expanded: list[Any] = []
    records: list[PassiveParasitic] = []
    for raw in raw_components:
        if not _declares(raw):
            expanded.append(raw)
            continue
        rating = _ripple_rating(raw, parameters)
        component_id = raw.get("id")
        component_type = raw.get("type")
        present = [field for field in PARASITIC_FIELDS if field in raw]
        for field in present:
            if component_type not in PARASITIC_FIELDS[field]:
                allowed = ", ".join(sorted(PARASITIC_FIELDS[field]))
                raise ModelError(
                    f"{component_id}.{field}: only valid for {allowed}, "
                    f"not {component_type!r}"
                )
        # Each type accepts exactly one field, so `present` has one entry here.
        field = present[0]
        if not isinstance(component_id, str) or not component_id:
            raise ModelError(f"{field}: the component needs an id")
        _check_value(raw[field], parameters, name=f"{component_id}.{field}")

        core = f"{component_id}.core"
        if core in user_nodes:
            raise ModelError(
                f"{component_id}.{field}: node name {core!r} is reserved for the "
                "expanded parasitic; rename that node"
            )
        main = {
            key: value
            for key, value in raw.items()
            if key not in (field, RIPPLE_CURRENT_RATING)
        }
        resistor = {
            "id": f"{component_id}.{field}",
            "type": "resistor",
            "value": raw[field],
        }
        if field == "esr":
            # p --[esr]-- core --| C |-- n
            resistor["p"], resistor["n"] = raw.get("p"), core
            main["p"] = core
            expanded.extend([main, resistor])
        else:
            # p --( L )-- core --[dcr]-- n
            main["n"] = core
            resistor["p"], resistor["n"] = core, raw.get("n")
            expanded.extend([main, resistor])
        records.append(
            PassiveParasitic(
                component_id=component_id,
                kind=field,
                resistor_id=resistor["id"],
                ripple_current_rating_a=rating,
            )
        )
    return expanded, tuple(records)
