# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""`DesignSpec` — the requirements side of a design job.

A specification describes the JOB, not the circuit: what the converter must
do (hard constraints — these later REJECT topology families) and
what the customer would like (soft criteria — these later rank the Pareto
front). The two kinds live in separate sections and mixing them is
a validation error, because a soft wish smuggled into the hard section would
silently shrink the candidate space and a hard limit smuggled into the soft
section would let an infeasible design win a comparison.

Fail-closed policy, carried over from the kernel to the specification layer:

* Unknown fields are errors, never ignored.
* A missing required field NEVER receives a default. It becomes an
  ``OpenQuestion`` carrying the reason the field is needed, and the spec
  refuses to be consumed (``require_complete``) until a human answers.
* Every value has an author. Declared values belong to the spec ``owner``;
  assumptions name their author and rationale. There is no code path that
  manufactures a value out of nothing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import jsonschema
import yaml

from pear_sim.design.errors import DesignSpecError, IncompleteDesignSpecError
from pear_sim.run.contracts import sha256_json

DESIGN_SPEC_SCHEMA = "pear-sim/design-spec/v0.1"

# --------------------------------------------------------------------------
# Field registry — the closed vocabulary of the specification.
#
# ``kind`` decides which section a field may appear in. ``why`` is the
# context attached to the open question when the field is missing ("what is
# this needed for") — required by the item's acceptance criteria, so an
# engineer answering the question knows what the answer will drive.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSpec:
    path: str
    kind: str  # "hard" | "soft"
    required: bool
    why: str
    validate: Callable[[Any], None]
    search_terms: str  # keywords describing the field (for tooling)


def _enum(*allowed: str) -> Callable[[Any], None]:
    def check(value: Any) -> None:
        if value not in allowed:
            raise DesignSpecError(
                f"expected one of {sorted(allowed)}, got {value!r}"
            )

    return check


def _finite_number(value: Any, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DesignSpecError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise DesignSpecError(f"{name} must be finite, got {value!r}")
    if positive and number <= 0.0:
        raise DesignSpecError(f"{name} must be > 0, got {value!r}")
    return number


def _range_v(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"min", "max"}:
        raise DesignSpecError("expected an object with exactly {min, max}")
    low = _finite_number(value["min"], "min")
    high = _finite_number(value["max"], "max")
    if low > high:
        raise DesignSpecError(f"min {low} exceeds max {high}")


def _ambient_range(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"min", "max"}:
        raise DesignSpecError("expected an object with exactly {min, max}")
    low = _finite_number(value["min"], "min", positive=False)
    high = _finite_number(value["max"], "max", positive=False)
    if low > high:
        raise DesignSpecError(f"min {low} exceeds max {high}")
    if low < -273.15:
        raise DesignSpecError(f"min {low} degC is below absolute zero")


def _output_voltage(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"nominal", "tolerance_pct"}:
        raise DesignSpecError("expected an object with exactly {nominal, tolerance_pct}")
    _finite_number(value["nominal"], "nominal")
    _finite_number(value["tolerance_pct"], "tolerance_pct")


def _output_power(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"max"}:
        raise DesignSpecError("expected an object with exactly {max}")
    _finite_number(value["max"], "max")


def _standards_list(value: Any) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise DesignSpecError(
            "expected a list of standard identifiers (an EXPLICIT empty list "
            "is a statement and is allowed; omitting the field is not)"
        )


def _grid_scenario(value: Any) -> None:
    allowed = {"scr", "voltage_unbalance_pct", "background_thd_pct", "dips"}
    required = {"scr"}
    if not isinstance(value, Mapping) or not required <= set(value) <= allowed:
        raise DesignSpecError(
            f"expected an object with at least {sorted(required)} and only "
            f"{sorted(allowed)}"
        )
    _finite_number(value["scr"], "scr")
    for key in ("voltage_unbalance_pct", "background_thd_pct"):
        if key in value:
            _finite_number(value[key], key, positive=False)
    for dip in value.get("dips", ()):
        if not isinstance(dip, Mapping) or set(dip) != {"depth_pct", "duration_ms"}:
            raise DesignSpecError(
                "each dip must be an object with exactly {depth_pct, duration_ms}"
            )
        _finite_number(dip["depth_pct"], "depth_pct")
        _finite_number(dip["duration_ms"], "duration_ms")


def _percent(value: Any) -> None:
    number = _finite_number(value, "value")
    if number > 100.0:
        raise DesignSpecError(f"a percentage cannot exceed 100, got {number}")


def _cost(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != {"value", "currency"}:
        raise DesignSpecError("expected an object with exactly {value, currency}")
    _finite_number(value["value"], "value")
    if not isinstance(value["currency"], str) or len(value["currency"]) != 3:
        raise DesignSpecError("currency must be a 3-letter code")


def _positive_number(value: Any) -> None:
    _finite_number(value, "value")


def _positive_int(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DesignSpecError(f"expected a positive integer, got {value!r}")


def _string_list(value: Any) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise DesignSpecError("expected a list of non-empty strings")


FIELDS: tuple[FieldSpec, ...] = (
    # ------------------------------------------------------------- hard ---
    FieldSpec(
        "input.kind",
        "hard",
        True,
        "topology families are filtered by the input interface before "
        "anything is computed: dc, single-phase or three-phase "
        "ac input each admit a different candidate set",
        _enum("dc", "ac_single_phase", "ac_three_phase"),
        "input interface dc ac phase",
    ),
    FieldSpec(
        "input.voltage_v",
        "hard",
        True,
        "the input voltage window bounds device voltage classes and rules "
        "out topologies whose conversion ratio cannot span it",
        _range_v,
        "input voltage range",
    ),
    FieldSpec(
        "output.voltage_v",
        "hard",
        True,
        "the output voltage and its tolerance are the acceptance band every "
        "candidate is later judged against by the oracle",
        _output_voltage,
        "output voltage tolerance",
    ),
    FieldSpec(
        "output.power_w",
        "hard",
        True,
        "continuous output power picks the power class: it decides which "
        "topology families are even plausible and scales every loss number",
        _output_power,
        "output power rating",
    ),
    FieldSpec(
        "isolation",
        "hard",
        True,
        "galvanic isolation is a family-level reject: "
        "non-isolated topologies cannot be fixed later if it is required — "
        "state 'not_required' explicitly if it does not matter",
        _enum("required", "not_required"),
        "galvanic isolation safety barrier",
    ),
    FieldSpec(
        "direction",
        "hard",
        True,
        "unidirectional topologies cannot serve a bidirectional job; this "
        "rejects whole families before any sizing",
        _enum("unidirectional", "bidirectional"),
        "bidirectional power flow",
    ),
    FieldSpec(
        "standards.emc",
        "hard",
        True,
        "the EMC class drives filter volume and switching-frequency "
        "choices; an explicit empty list means 'no EMC requirement' and is "
        "an owner statement, not a default",
        _standards_list,
        "EMC standard emissions class",
    ),
    FieldSpec(
        "standards.safety",
        "hard",
        True,
        "safety standards decide insulation coordination and creepage — "
        "they must be known before topology selection, not at layout",
        _standards_list,
        "safety standard insulation",
    ),
    FieldSpec(
        "grid",
        "hard",
        False,  # conditionally required for ac inputs — see questions()
        "for grid-fed inputs the grid is a specification parameter, not a "
        "run setting: short-circuit ratio, unbalance and dip profiles "
        "change which topologies and controls survive (owner's AC-DC "
        "review, pre-review source)",
        _grid_scenario,
        "grid scr short circuit ratio unbalance dips",
    ),
    FieldSpec(
        "environment.ambient_c",
        "hard",
        False,
        "the ambient window sets thermal derating; without it any thermal "
        "verdict is conditional",
        _ambient_range,
        "ambient temperature range",
    ),
    # ------------------------------------------------------------- soft ---
    FieldSpec(
        "targets.efficiency_pct",
        "soft",
        False,
        "an efficiency target ranks candidates on the Pareto front; "
        "it does not reject families",
        _percent,
        "efficiency target",
    ),
    FieldSpec(
        "targets.cost",
        "soft",
        False,
        "target unit cost is a Pareto axis, not a filter",
        _cost,
        "cost target budget",
    ),
    FieldSpec(
        "targets.volume_l",
        "soft",
        False,
        "target volume is a Pareto axis, not a filter",
        _positive_number,
        "volume power density",
    ),
    FieldSpec(
        "targets.mass_kg",
        "soft",
        False,
        "target mass is a Pareto axis, not a filter",
        _positive_number,
        "mass weight",
    ),
    FieldSpec(
        "lead_time_weeks",
        "soft",
        False,
        "acceptable lead time softly penalises exotic parts",
        _positive_number,
        "lead time availability",
    ),
    FieldSpec(
        "production_volume_units",
        "soft",
        False,
        "production volume shifts the cost/effort trade-off between "
        "candidates",
        _positive_int,
        "production volume quantity",
    ),
    FieldSpec(
        "preferences",
        "soft",
        False,
        "free-form customer preferences ranked by the engineer, never "
        "enforced mechanically",
        _string_list,
        "customer preference",
    ),
)

_BY_PATH: dict[str, FieldSpec] = {spec.path: spec for spec in FIELDS}


# --------------------------------------------------------------------------
# Value provenance
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenancedValue:
    """A specification value plus where it came from. ``basis`` is one of
    ``declared`` (spec owner wrote it), ``assumption`` (a named human
    assumed it, with rationale)."""

    value: Any
    basis: str
    author: dict[str, str]
    rationale: str | None = None
    flags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "value": self.value,
            "basis": self.basis,
            "author": dict(self.author),
        }
        if self.rationale is not None:
            payload["rationale"] = self.rationale
        if self.flags:
            payload["flags"] = list(self.flags)
        return payload


@dataclass(frozen=True)
class OpenQuestion:
    """A missing answer, not an error: the field and why it is needed."""

    field: str
    why: str

    def as_dict(self) -> dict[str, Any]:
        return {"field": self.field, "why": self.why}


# --------------------------------------------------------------------------
# The spec itself
# --------------------------------------------------------------------------


def _flatten(section: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested mappings down to registry paths. Recursion stops as
    soon as a path is registered, so structured VALUES (e.g. the grid
    scenario object) are not torn apart."""

    flat: dict[str, Any] = {}
    for key, value in section.items():
        if not isinstance(key, str):
            raise DesignSpecError(f"field names must be strings, got {key!r}")
        path = f"{prefix}{key}"
        if path in _BY_PATH:
            flat[path] = value
        elif isinstance(value, Mapping):
            flat.update(_flatten(value, f"{path}."))
        else:
            raise DesignSpecError(
                f"unknown specification field '{path}' — the vocabulary is "
                f"closed; known fields: {sorted(_BY_PATH)}"
            )
    return flat


def _require_actor(value: Any, context: str) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"id", "name"}
        or any(not isinstance(value[k], str) or not value[k].strip() for k in ("id", "name"))
    ):
        raise DesignSpecError(
            f"{context} must name a human as {{id, name}} — a specification "
            "value without an accountable author is not allowed"
        )
    return {"id": value["id"], "name": value["name"]}


_SCHEMA_PATH = Path(__file__).with_name("design-spec.schema.json")


@dataclass(frozen=True)
class DesignSpec:
    name: str
    owner: dict[str, str]
    values: dict[str, ProvenancedValue] = field(default_factory=dict)
    source_path: str | None = None

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: str | Path) -> "DesignSpec":
        raw_text = Path(path).read_text(encoding="utf-8")
        try:
            document = yaml.safe_load(raw_text)
        except yaml.YAMLError as error:
            raise DesignSpecError(f"specification is not valid YAML: {error}") from error
        return cls.from_dict(document, source_path=str(path))

    @classmethod
    def from_dict(cls, document: Any, *, source_path: str | None = None) -> "DesignSpec":
        schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
        try:
            jsonschema.validate(document, schema)
        except jsonschema.ValidationError as error:
            location = "/".join(str(part) for part in error.absolute_path) or "<root>"
            raise DesignSpecError(
                f"specification rejected by schema at {location}: {error.message}"
            ) from error

        owner = _require_actor(document["owner"], "owner")
        values: dict[str, ProvenancedValue] = {}

        for section_name in ("hard", "soft"):
            flat = _flatten(document.get(section_name, {}) or {})
            for path_key, value in flat.items():
                spec = _BY_PATH[path_key]
                if spec.kind != section_name:
                    raise DesignSpecError(
                        f"field '{path_key}' is a {spec.kind} constraint and "
                        f"must not appear in the '{section_name}' section — "
                        "mixing hard rejects with soft criteria would corrupt "
                        "both the screening and the Pareto ranking"
                    )
                try:
                    spec.validate(value)
                except DesignSpecError as error:
                    raise DesignSpecError(f"field '{path_key}': {error}") from error
                values[path_key] = ProvenancedValue(
                    value=value, basis="declared", author=owner
                )

        for entry in document.get("assumptions", ()) or ():
            path_key = entry["field"]
            spec = _BY_PATH.get(path_key)
            if spec is None:
                raise DesignSpecError(
                    f"assumption for unknown field '{path_key}' — the "
                    f"vocabulary is closed; known fields: {sorted(_BY_PATH)}"
                )
            if path_key in values:
                raise DesignSpecError(
                    f"field '{path_key}' is both declared and assumed — "
                    "remove one; an ambiguous provenance is worse than a "
                    "missing value"
                )
            author = _require_actor(entry["author"], f"assumption for '{path_key}'")
            rationale = entry.get("rationale")
            if not isinstance(rationale, str) or not rationale.strip():
                raise DesignSpecError(
                    f"assumption for '{path_key}' needs a non-empty rationale "
                    "— an assumption nobody can defend is a default in "
                    "disguise"
                )
            try:
                spec.validate(entry["value"])
            except DesignSpecError as error:
                raise DesignSpecError(f"assumption for '{path_key}': {error}") from error
            values[path_key] = ProvenancedValue(
                value=entry["value"],
                basis="assumption",
                author=author,
                rationale=rationale,
            )

        return cls(
            name=document["name"],
            owner=owner,
            values=values,
            source_path=source_path,
        )

    # ------------------------------------------------------- completeness
    def _missing_fields(self) -> list[FieldSpec]:
        missing = [
            spec
            for spec in FIELDS
            if spec.required and spec.path not in self.values
        ]
        input_kind = self.values.get("input.kind")
        if (
            input_kind is not None
            and input_kind.value != "dc"
            and "grid" not in self.values
        ):
            missing.append(_BY_PATH["grid"])
        return missing

    def questions(self) -> list[OpenQuestion]:
        """Open questions for every unanswerable field."""

        return [
            OpenQuestion(field=spec.path, why=spec.why)
            for spec in self._missing_fields()
        ]

    @property
    def complete(self) -> bool:
        return not self._missing_fields()

    def require_complete(self) -> None:
        missing = self._missing_fields()
        if missing:
            listing = ", ".join(spec.path for spec in missing)
            raise IncompleteDesignSpecError(
                f"specification '{self.name}' has open questions for: "
                f"{listing} — answer them; "
                "defaults are not available by design"
            )

    def resolved(self) -> dict[str, Any]:
        return {
            "schema": DESIGN_SPEC_SCHEMA,
            "name": self.name,
            "owner": dict(self.owner),
            "complete": self.complete,
            "values": {
                path_key: self.values[path_key].as_dict()
                for path_key in sorted(self.values)
            },
            "open_questions": [q.as_dict() for q in self.questions()],
        }

    def spec_sha256(self) -> str:
        """Content hash of the COMPLETE specification (values + provenance).
        Hashing an incomplete spec is refused: a stable identity for a
        document with holes would let the holes hide."""

        self.require_complete()
        return sha256_json(self.resolved())

    def hard_values(self) -> dict[str, Any]:
        return {
            path_key: value.value
            for path_key, value in self.values.items()
            if _BY_PATH[path_key].kind == "hard"
        }

    def soft_values(self) -> dict[str, Any]:
        return {
            path_key: value.value
            for path_key, value in self.values.items()
            if _BY_PATH[path_key].kind == "soft"
        }
