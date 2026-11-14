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
"""

from __future__ import annotations

import math
from typing import Any

from pear_sim.errors import ModelError
from pear_sim.spec.controls import _resolve
from pear_sim.spec.units import Quantity

__all__ = ["PARASITIC_FIELDS", "expand_passive_parasitics"]

#: Parasitic field -> component types that accept it.
PARASITIC_FIELDS: dict[str, frozenset[str]] = {
    "esr": frozenset({"capacitor"}),
    "dcr": frozenset({"inductor", "sat_inductor"}),
}


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


def expand_passive_parasitics(
    raw_components: list[Any],
    parameters: dict[str, Quantity],
) -> list[Any]:
    """Return ``raw_components`` with every ``esr``/``dcr`` expanded in series.

    Components without a parasitic field pass through as the same object, so a
    model that declares none is unchanged byte for byte.
    """

    declared = [
        raw
        for raw in raw_components
        if isinstance(raw, dict) and any(field in raw for field in PARASITIC_FIELDS)
    ]
    if not declared:
        return raw_components

    user_nodes = {
        raw.get(terminal)
        for raw in raw_components
        if isinstance(raw, dict)
        for terminal in ("p", "n")
    }
    expanded: list[Any] = []
    for raw in raw_components:
        if not isinstance(raw, dict) or not any(field in raw for field in PARASITIC_FIELDS):
            expanded.append(raw)
            continue
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
        main = {key: value for key, value in raw.items() if key != field}
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
    return expanded
