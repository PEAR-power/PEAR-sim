from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from pear_sim.errors import UnitError


@dataclass(frozen=True)
class Quantity:
    value: float
    dimension: str


_UNITS: dict[str, tuple[float, str]] = {
    "": (1.0, "dimensionless"),
    "1": (1.0, "dimensionless"),
    "%": (0.01, "dimensionless"),
    "s": (1.0, "time"),
    "ms": (1e-3, "time"),
    "us": (1e-6, "time"),
    "ns": (1e-9, "time"),
    "hz": (1.0, "frequency"),
    "khz": (1e3, "frequency"),
    "mhz": (1e6, "frequency"),
    "v": (1.0, "voltage"),
    "mv": (1e-3, "voltage"),
    "kv": (1e3, "voltage"),
    "a": (1.0, "current"),
    "ma": (1e-3, "current"),
    "ka": (1e3, "current"),
    "ohm": (1.0, "resistance"),
    "kohm": (1e3, "resistance"),
    "megaohm": (1e6, "resistance"),
    "milliohm": (1e-3, "resistance"),
    "h": (1.0, "inductance"),
    "mh": (1e-3, "inductance"),
    "uh": (1e-6, "inductance"),
    "nh": (1e-9, "inductance"),
    "f": (1.0, "capacitance"),
    "mf": (1e-3, "capacitance"),
    "uf": (1e-6, "capacitance"),
    "nf": (1e-9, "capacitance"),
    "pf": (1e-12, "capacitance"),
    "w": (1.0, "power"),
    "kw": (1e3, "power"),
    "j": (1.0, "energy"),
    "mj": (1e-3, "energy"),
    "k": (1.0, "temperature"),
    "degc": (1.0, "temperature"),
}

# Reverse lookup: the unit symbol whose scale is exactly 1.0 for a dimension.
# Used when a producer has an SI number and must write it back out as a
# unit-bearing string, so a generated model source keeps the dimension it
# started with instead of degrading into a bare (dimensionless) float.
SI_BASE_UNITS: dict[str, str] = {
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


def format_si(value: float, dimension: str) -> str | float:
    """Render an SI value as a unit-bearing string for a generated model."""

    unit = SI_BASE_UNITS.get(dimension)
    if unit is None:
        raise UnitError(f"no base unit is known for dimension {dimension!r}")
    if not unit:
        return float(value)
    return f"{value!r} {unit}"


# Temperature is the one supported dimension whose units differ by an OFFSET
# and not only a factor. Normalizing "25 degc" as 25 K by scale alone is off
# by 273.15 — an error a plausibility check would never catch, because both
# numbers look like temperatures. Kept as an explicit table so a new
# temperature unit cannot be added without deciding its offset.
_ADDITIVE_OFFSETS: dict[str, float] = {
    "degc": 273.15,
}

_VALUE_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([^\s]*)\s*$"
)


def _normalize_unit(unit: str) -> str:
    normalized = unit.strip().replace("µ", "u").replace("μ", "u").replace("Ω", "ohm")
    if normalized in {"MOhm", "Mohm"}:
        return "megaohm"
    if normalized in {"mOhm", "mohm"}:
        return "milliohm"
    return normalized.lower()


def parse_quantity(value: Any, expected: str | None = None, *, name: str = "value") -> Quantity:
    if isinstance(value, bool):
        raise UnitError(f"{name}: boolean is not a numeric quantity")
    if isinstance(value, (int, float)):
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise UnitError(f"{name}: value must fit the finite float contract") from exc
        dimension = expected or "dimensionless"
        if not math.isfinite(numeric):
            raise UnitError(f"{name}: value must be finite")
        return Quantity(numeric, dimension)
    if not isinstance(value, str):
        raise UnitError(f"{name}: expected a number or a number with a unit")
    match = _VALUE_RE.match(value)
    if not match:
        raise UnitError(f"{name}: cannot parse quantity {value!r}")
    numeric = float(match.group(1))
    if not math.isfinite(numeric):
        raise UnitError(f"{name}: value must be finite")
    unit = _normalize_unit(match.group(2))
    if unit not in _UNITS:
        raise UnitError(f"{name}: unsupported unit {match.group(2)!r}")
    factor, dimension = _UNITS[unit]
    if expected is not None and dimension != expected:
        raise UnitError(f"{name}: expected {expected}, got {dimension} ({value!r})")
    normalized = numeric * factor + _ADDITIVE_OFFSETS.get(unit, 0.0)
    if not math.isfinite(normalized):
        raise UnitError(f"{name}: SI-normalized value must be finite")
    if dimension == "temperature" and normalized < 0.0:
        raise UnitError(
            f"{name}: {value!r} is below absolute zero ({normalized:.2f} K)"
        )
    return Quantity(normalized, dimension)


def require_dimension(quantity: Quantity, expected: str, *, name: str) -> float:
    if quantity.dimension != expected:
        raise UnitError(f"{name}: expected {expected}, got {quantity.dimension}")
    return quantity.value
