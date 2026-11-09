# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Parameter envelopes: a declared parameter is a scalar OR a drift envelope.

A model parameter has always been one number. This module adds the second
form — the same number plus a *declared* statement of how far it may move and
where that statement came from:

* **corners** — named operating points (``eol_lo``, ``eol_hi``, ``min`` …),
* **distribution** — uniform / normal / triangular, sampled from an explicit
  seed so a campaign is reproducible bit for bit,
* **life** — ``p = f(p_bol, life_fraction)`` for quantities that grow across a
  mission rather than sitting at a fixed offset (leakage is the motivating
  case).

Two rules are load-bearing and both fail closed.

**A parameter with no declared envelope has no drift.** The loader never
invents a tolerance. Absence means "not stated", never "assume ±x%".

**Every envelope carries its provenance.** ``source`` and ``note`` are required,
not optional, because the number's authority is the whole point: a drift taken
from a qualification report and a drift someone guessed produce identical
arithmetic and must never look identical in a report.

The envelope is a declaration *about* a parameter, not a change *to* it.
``Model.parameters`` still holds the nominal scalar, the solver still sees one
number, and a model that declares no envelope hashes exactly as it did before
this module existed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
import statistics
from typing import Any, Mapping

from pear_sim.errors import ModelError
from pear_sim.spec.units import Quantity, parse_quantity


__all__ = [
    "PROVENANCE_SOURCES",
    "Distribution",
    "LifeModel",
    "ParameterEnvelope",
    "parse_parameter_envelope",
]


# The four things a drift number can be. `assumption` is the honest default for
# a placeholder and must survive all the way into the report.
PROVENANCE_SOURCES = ("datasheet", "test_report", "assumption", "standard")

_CORNER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_CORNER_NAMES = frozenset({"nom", "nominal"})
_MAX_CORNERS = 16
_MAX_NOTE_CHARS = 512

# A normal distribution is unbounded; a component value is not. Samples are
# clamped to this many sigmas so a campaign cannot silently produce a negative
# resistance. The clamp is recorded in the artifact, not hidden.
NORMAL_TRUNCATION_SIGMA = 3.0


@dataclass(frozen=True)
class Distribution:
    """A sampling law with an explicit seed. No seed, no sampling."""

    kind: str
    seed: int
    low_si: float
    high_si: float
    mode_si: float | None = None
    sigma_si: float | None = None

    def sample(self, nominal_si: float, count: int) -> tuple[float, ...]:
        """Return ``count`` draws, deterministic for a given seed.

        Uses inverse-CDF sampling over ``random.Random`` uniforms and
        ``statistics.NormalDist`` so the sequence depends only on the seed —
        not on a numpy version or a platform BLAS.
        """

        rng = random.Random(self.seed)
        draws: list[float] = []
        for _ in range(count):
            u = rng.random()
            if self.kind == "uniform":
                value = self.low_si + u * (self.high_si - self.low_si)
            elif self.kind == "normal":
                sigma = float(self.sigma_si or 0.0)
                z = statistics.NormalDist(0.0, 1.0).inv_cdf(u)
                z = max(-NORMAL_TRUNCATION_SIGMA, min(NORMAL_TRUNCATION_SIGMA, z))
                value = nominal_si + z * sigma
            elif self.kind == "triangular":
                low, high = self.low_si, self.high_si
                mode = float(self.mode_si if self.mode_si is not None else nominal_si)
                split = (mode - low) / (high - low) if high > low else 0.0
                if u < split:
                    value = low + math.sqrt(u * (high - low) * (mode - low))
                else:
                    value = high - math.sqrt((1.0 - u) * (high - low) * (high - mode))
            else:  # pragma: no cover - constructor rejects other kinds
                raise ModelError(f"unsupported distribution kind {self.kind!r}")
            draws.append(float(value))
        return tuple(draws)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.kind,
            "seed": self.seed,
            "low_si": self.low_si,
            "high_si": self.high_si,
        }
        if self.mode_si is not None:
            payload["mode_si"] = self.mode_si
        if self.sigma_si is not None:
            payload["sigma_si"] = self.sigma_si
            payload["truncated_at_sigma"] = NORMAL_TRUNCATION_SIGMA
        return payload


@dataclass(frozen=True)
class LifeModel:
    """``p = f(p_bol, life_fraction)`` on ``life_fraction`` in [0, 1].

    ``linear`` adds a percentage of the beginning-of-life value; ``geometric``
    multiplies by a ratio raised to the life fraction, which is the shape a
    leakage current that grows by decades across a mission actually takes.
    """

    kind: str
    end_of_life_percent: float | None = None
    end_of_life_ratio: float | None = None

    def value_at(self, bol_si: float, life_fraction: float) -> float:
        if not 0.0 <= life_fraction <= 1.0:
            raise ModelError("life_fraction must lie in [0, 1]")
        if self.kind == "linear":
            percent = float(self.end_of_life_percent or 0.0)
            return bol_si * (1.0 + percent / 100.0 * life_fraction)
        if self.kind == "geometric":
            ratio = float(self.end_of_life_ratio or 1.0)
            return bol_si * (ratio**life_fraction)
        raise ModelError(f"unsupported life model {self.kind!r}")  # pragma: no cover

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"type": self.kind}
        if self.end_of_life_percent is not None:
            payload["end_of_life_percent"] = self.end_of_life_percent
        if self.end_of_life_ratio is not None:
            payload["end_of_life_ratio"] = self.end_of_life_ratio
        return payload


@dataclass(frozen=True)
class ParameterEnvelope:
    """Everything declared about one parameter beyond its nominal value."""

    parameter: str
    nominal: Quantity
    source: str
    note: str
    corners: tuple[tuple[str, float], ...] = ()
    distribution: Distribution | None = None
    life: LifeModel | None = None

    @property
    def corner_map(self) -> dict[str, float]:
        return dict(self.corners)

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "parameter": self.parameter,
            "nominal_si": self.nominal.value,
            "dimension": self.nominal.dimension,
            "source": self.source,
            "note": self.note,
        }
        if self.corners:
            payload["corners_si"] = {name: value for name, value in self.corners}
        if self.distribution is not None:
            payload["distribution"] = self.distribution.as_dict()
        if self.life is not None:
            payload["life"] = self.life.as_dict()
        return payload


def _require_text(value: Any, *, name: str, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelError(f"{name} must be a non-empty string")
    if len(value) > max_chars:
        raise ModelError(f"{name} exceeds {max_chars} characters")
    return value


def _reject_unknown_keys(raw: Mapping[str, Any], allowed: set[str], *, name: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ModelError(f"{name}: unknown field(s) " + ", ".join(sorted(unknown)))


def _parse_corners(raw: Any, dimension: str, *, name: str) -> tuple[tuple[str, float], ...]:
    if not isinstance(raw, dict) or not raw:
        raise ModelError(f"{name} must be a non-empty object of corner name to value")
    if len(raw) > _MAX_CORNERS:
        raise ModelError(f"{name} declares more than {_MAX_CORNERS} corners")
    corners: list[tuple[str, float]] = []
    for corner_name, corner_value in raw.items():
        if not isinstance(corner_name, str) or _CORNER_NAME_RE.fullmatch(corner_name) is None:
            raise ModelError(
                f"{name}: corner name must match {_CORNER_NAME_RE.pattern!r}"
            )
        if corner_name in _RESERVED_CORNER_NAMES:
            raise ModelError(
                f"{name}: corner name {corner_name!r} is reserved — the nominal "
                "point is always present and is named 'nom'"
            )
        quantity = parse_quantity(
            corner_value, dimension, name=f"{name}.{corner_name}"
        )
        corners.append((corner_name, float(quantity.value)))
    # Deterministic order: the campaign matrix must not depend on dict ordering.
    return tuple(sorted(corners))


def _parse_seed(raw: Any, *, name: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ModelError(f"{name}.seed must be an integer — sampling is never implicit")
    if not 0 <= raw < 2**63:
        raise ModelError(f"{name}.seed must lie in [0, 2^63)")
    return int(raw)


def _parse_distribution(
    raw: Any, nominal: Quantity, *, name: str
) -> Distribution:
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    kind = raw.get("type")
    dimension = nominal.dimension
    if kind == "uniform":
        _reject_unknown_keys(raw, {"type", "seed", "min", "max"}, name=name)
        low = parse_quantity(raw.get("min"), dimension, name=f"{name}.min").value
        high = parse_quantity(raw.get("max"), dimension, name=f"{name}.max").value
        if not high > low:
            raise ModelError(f"{name}: max must exceed min")
        return Distribution(
            kind="uniform", seed=_parse_seed(raw.get("seed"), name=name),
            low_si=float(low), high_si=float(high),
        )
    if kind == "normal":
        _reject_unknown_keys(raw, {"type", "seed", "sigma", "sigma_percent"}, name=name)
        if ("sigma" in raw) == ("sigma_percent" in raw):
            raise ModelError(f"{name}: declare exactly one of sigma, sigma_percent")
        if "sigma" in raw:
            sigma = float(
                parse_quantity(raw["sigma"], dimension, name=f"{name}.sigma").value
            )
        else:
            percent = raw["sigma_percent"]
            if isinstance(percent, bool) or not isinstance(percent, (int, float)):
                raise ModelError(f"{name}.sigma_percent must be a number")
            sigma = abs(float(nominal.value)) * float(percent) / 100.0
        if not sigma > 0.0 or not math.isfinite(sigma):
            raise ModelError(f"{name}: sigma must be finite and positive")
        span = NORMAL_TRUNCATION_SIGMA * sigma
        return Distribution(
            kind="normal", seed=_parse_seed(raw.get("seed"), name=name),
            low_si=float(nominal.value) - span, high_si=float(nominal.value) + span,
            sigma_si=sigma,
        )
    if kind == "triangular":
        _reject_unknown_keys(raw, {"type", "seed", "min", "max", "mode"}, name=name)
        low = float(parse_quantity(raw.get("min"), dimension, name=f"{name}.min").value)
        high = float(parse_quantity(raw.get("max"), dimension, name=f"{name}.max").value)
        if not high > low:
            raise ModelError(f"{name}: max must exceed min")
        if "mode" in raw:
            mode = float(
                parse_quantity(raw["mode"], dimension, name=f"{name}.mode").value
            )
        else:
            mode = float(nominal.value)
        if not low <= mode <= high:
            raise ModelError(f"{name}: mode must lie within [min, max]")
        return Distribution(
            kind="triangular", seed=_parse_seed(raw.get("seed"), name=name),
            low_si=low, high_si=high, mode_si=mode,
        )
    raise ModelError(
        f"{name}.type must be one of uniform, normal, triangular (got {kind!r})"
    )


def _parse_life(raw: Any, *, name: str) -> LifeModel:
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    kind = raw.get("type")
    if kind == "linear":
        _reject_unknown_keys(raw, {"type", "end_of_life_percent"}, name=name)
        percent = raw.get("end_of_life_percent")
        if isinstance(percent, bool) or not isinstance(percent, (int, float)):
            raise ModelError(f"{name}.end_of_life_percent must be a number")
        if not math.isfinite(float(percent)):
            raise ModelError(f"{name}.end_of_life_percent must be finite")
        return LifeModel(kind="linear", end_of_life_percent=float(percent))
    if kind == "geometric":
        _reject_unknown_keys(raw, {"type", "end_of_life_ratio"}, name=name)
        ratio = raw.get("end_of_life_ratio")
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ModelError(f"{name}.end_of_life_ratio must be a number")
        if not math.isfinite(float(ratio)) or float(ratio) <= 0.0:
            raise ModelError(f"{name}.end_of_life_ratio must be finite and positive")
        return LifeModel(kind="geometric", end_of_life_ratio=float(ratio))
    raise ModelError(f"{name}.type must be one of linear, geometric (got {kind!r})")


def parse_parameter_envelope(
    raw: Mapping[str, Any], *, parameter: str
) -> tuple[Quantity, ParameterEnvelope]:
    """Parse the object form of a parameter into (nominal, envelope).

    Raises if the object declares provenance but no drift: an envelope that
    moves nothing is a scalar wearing a costume, and silently accepting it
    would let a report claim a corner analysis that never ran.
    """

    name = f"parameter {parameter}"
    _reject_unknown_keys(
        raw,
        {"nominal", "source", "note", "corners", "distribution", "life"},
        name=name,
    )
    if "nominal" not in raw:
        raise ModelError(f"{name}: the object form requires 'nominal'")
    nominal = parse_quantity(raw["nominal"], name=f"{name}.nominal")

    source = _require_text(raw.get("source"), name=f"{name}.source", max_chars=32)
    if source not in PROVENANCE_SOURCES:
        raise ModelError(
            f"{name}.source must be one of " + ", ".join(PROVENANCE_SOURCES)
        )
    note = _require_text(raw.get("note"), name=f"{name}.note", max_chars=_MAX_NOTE_CHARS)

    corners = (
        _parse_corners(raw["corners"], nominal.dimension, name=f"{name}.corners")
        if "corners" in raw
        else ()
    )
    distribution = (
        _parse_distribution(raw["distribution"], nominal, name=f"{name}.distribution")
        if "distribution" in raw
        else None
    )
    life = _parse_life(raw["life"], name=f"{name}.life") if "life" in raw else None

    if not corners and distribution is None and life is None:
        raise ModelError(
            f"{name}: an envelope must declare at least one of corners, "
            "distribution, life — otherwise write the parameter as a scalar"
        )

    envelope = ParameterEnvelope(
        parameter=parameter,
        nominal=nominal,
        source=source,
        note=note,
        corners=corners,
        distribution=distribution,
        life=life,
    )
    return nominal, envelope
