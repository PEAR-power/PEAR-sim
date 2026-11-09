# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""The design vector theta as a thin, typed layer over model overrides.

The named-parameter contract is deliberately narrow: **theta must be a
named parameter in the model's ``parameters:`` section**. The engine never
reaches into inline component values — a parameter the model does not name is
a parameter the optimisation cannot see, and it is better for that to be a
loud error with a remedy than a silently missing derivative.

Mechanically this wraps ``load_model(path, overrides=...)``, the same
многократ-override path the tolerance-corner sweeps already trust; nothing new
touches the loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pear_sim.errors import PearSimError
from pear_sim.spec.model import Model, load_model

__all__ = ["DesignVector", "DesignVectorError"]


class DesignVectorError(PearSimError):
    """The design vector violates the named-parameter contract."""

    code = "gradient.design_vector_invalid"


# SI unit spelled per dimension, so an override survives the loader's
# dimension check. Kelvin is correct for temperature: Quantity values are
# already SI-normalized, so no degC offset may be applied a second time.
_SI_UNIT_BY_DIMENSION = {
    "dimensionless": "",
    "time": "s",
    "frequency": "Hz",
    "voltage": "V",
    "current": "A",
    "resistance": "ohm",
    "inductance": "H",
    "capacitance": "F",
    "power": "W",
    "energy": "J",
    "temperature": "K",
}


@dataclass(frozen=True)
class DesignVector:
    """Named theta with SI values, bound to one model file.

    Immutable on purpose: an optimisation step produces a NEW vector via
    :meth:`with_offset` / :meth:`with_values`, so no two evaluations can ever
    share mutated state — the same property the deterministic kernel already
    guarantees one level down.
    """

    model_path: Path
    _names: tuple[str, ...]
    _values: tuple[float, ...]
    _dimensions: tuple[str, ...]

    @classmethod
    def from_model(cls, model_path: str | Path, names: tuple[str, ...] | list[str]) -> "DesignVector":
        """Bind theta to a model, failing closed on the named-parameter contract.

        Every requested name must exist in ``parameters:``. The error message
        says WHAT TO DO, not merely that it failed — if the name matches a
        component id, the remedy (move the value to ``parameters:`` and
        reference it as ``${...}``) is spelled out, because that is the exact
        situation the contract exists to catch.
        """

        if not names:
            raise DesignVectorError("a design vector needs at least one parameter")
        duplicates = {name for name in names if list(names).count(name) > 1}
        if duplicates:
            raise DesignVectorError(
                f"duplicate design parameter(s): {sorted(duplicates)}"
            )
        path = Path(model_path).resolve()
        model = load_model(path)
        component_ids = {component.id for component in model.components}
        values: list[float] = []
        dimensions: list[str] = []
        for name in names:
            quantity = model.parameters.get(name)
            if quantity is None:
                if name in component_ids:
                    raise DesignVectorError(
                        f"{name!r} is a component id, not a named parameter: its "
                        "value is inline in the component and therefore invisible "
                        "to the design vector (named-parameter contract). Move the value to "
                        "the model's parameters: section and reference it as "
                        f"'${{{name}}}' in the component."
                    )
                raise DesignVectorError(
                    f"design parameter {name!r} is not declared in the model's "
                    "parameters: section "
                    f"(declared: {sorted(model.parameters)})"
                )
            dimension = quantity.dimension
            if dimension not in _SI_UNIT_BY_DIMENSION:
                raise DesignVectorError(
                    f"design parameter {name!r} has dimension {dimension!r}, "
                    "which this layer cannot express as an SI override"
                )
            values.append(float(quantity.value))
            dimensions.append(dimension)
        return cls(path, tuple(names), tuple(values), tuple(dimensions))

    def names(self) -> tuple[str, ...]:
        return self._names

    def as_array(self) -> np.ndarray:
        """Theta as a dense SI vector, in ``names()`` order."""

        return np.array(self._values, dtype=float)

    def value(self, name: str) -> float:
        return self._values[self._index(name)]

    def with_offset(self, name: str, delta: float) -> "DesignVector":
        """A new vector with one entry shifted — the finite-difference step."""

        index = self._index(name)
        values = list(self._values)
        values[index] = values[index] + float(delta)
        return DesignVector(self.model_path, self._names, tuple(values), self._dimensions)

    def with_values(self, values: np.ndarray) -> "DesignVector":
        """A new vector with all entries replaced — the optimiser's step."""

        array = np.asarray(values, dtype=float)
        if array.shape != (len(self._names),):
            raise DesignVectorError(
                f"expected {len(self._names)} value(s) in names() order, got "
                f"shape {array.shape}"
            )
        if not np.all(np.isfinite(array)):
            raise DesignVectorError("design values must be finite")
        return DesignVector(
            self.model_path, self._names, tuple(float(v) for v in array), self._dimensions
        )

    def overrides(self) -> dict[str, str]:
        """The exact strings handed to ``load_model(..., overrides=...)``.

        ``repr`` keeps full float precision; the unit keeps the loader's
        dimension check honest instead of bypassing it.
        """

        rendered: dict[str, str] = {}
        for name, value, dimension in zip(self._names, self._values, self._dimensions):
            unit = _SI_UNIT_BY_DIMENSION[dimension]
            rendered[name] = f"{value!r} {unit}".strip()
        return rendered

    def load(self) -> Model:
        """The bound model with this theta applied — one call, no side channel."""

        return load_model(self.model_path, self.overrides())

    def _index(self, name: str) -> int:
        try:
            return self._names.index(name)
        except ValueError:
            raise DesignVectorError(
                f"{name!r} is not part of this design vector "
                f"(names: {list(self._names)})"
            ) from None
