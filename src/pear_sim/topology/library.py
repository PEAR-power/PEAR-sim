# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Topology library entries — declarative data, never code.

An entry describes one topology the way a selection engine needs it:
prerequisites a `DesignSpec` can be matched against (isolation,
directionality, sensible voltage/power ranges), the switching cycle as a
:class:`~pear_sim.analysis.gradient.sequence.ModeSequence` — THE SAME object
the gradient engine consumes, serialized by the one canonical pair in
``pear_sim.analysis.gradient.sequence`` — and a MANDATORY list of known
limitations.  An empty limitations list is a claim ("this topology has no
known weaknesses"), not an absence of data, so the format refuses it.

Adding a topology is adding a YAML file.  The loader is generic: nothing in
this module names buck, boost or any other entry, which is the acceptance
test for "data, not code".

Entries are validated twice, fail closed: the packaged JSON Schema first
(structure), then the semantic rules the schema cannot express (id equals
the file stem, coherent ranges, unique sequence labels, and every stored
sequence reconstructed through :func:`mode_sequence_from_dict` — a sequence
this module cannot rebuild into the engine's dataclass is refused, not
carried along as opaque text).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pear_sim.analysis.gradient.sequence import (
    ModeSequence,
    ModeSequenceError,
    extract_mode_sequence,
    mode_sequence_from_dict,
    sequences_agree,
)
from pear_sim.errors import PearSimError

TOPOLOGY_ENTRY_SCHEMA = "pear-sim/topology-entry/v0.1"
MAX_ENTRY_SOURCE_BYTES = 256 * 1024


class TopologyLibraryError(PearSimError):
    """A topology entry or library directory is invalid."""

    code = "topology.library_invalid"


@dataclass(frozen=True)
class VoltageRange:
    min_v: float
    max_v: float


@dataclass(frozen=True)
class PowerRange:
    min_w: float
    max_w: float


@dataclass(frozen=True)
class Prerequisites:
    """What a topology needs and where it makes sense — the 3.2 sieve input."""

    isolation: str
    direction: str
    voltage: VoltageRange
    power: PowerRange


@dataclass(frozen=True)
class EntryModeSequence:
    """One labelled cycle of the entry, as the engine's own object."""

    label: str
    source_model: str
    sequence: ModeSequence


@dataclass(frozen=True)
class TopologyTemplate:
    """The circuit this entry describes.

    ``model`` is a repo-relative model file, absent for an entry this kernel
    cannot yet run; ``subcircuit`` optionally names the declared subcircuit
    that IS the topology as a reusable unit — the level at which "the
    same bridge with a different leg" is expressible.
    """

    model: str | None
    subcircuit: str | None


@dataclass(frozen=True)
class KernelSupport:
    """Whether THIS kernel can currently simulate the entry, stated openly.

    A selection library that only listed simulatable topologies would bias
    every recommendation toward what the engine happens to support — the
    tool would reject a flyback because it is missing, not because it is
    worse, and nobody downstream could tell the two apart. So an entry may
    declare ``not_simulated`` with a reason and still be screened (item
    3.2); what it may NOT do is be silently absent.

    The claim and its evidence are coupled at load time: ``simulated``
    requires at least one mode sequence, ``not_simulated`` requires none.
    """

    status: str
    note: str

    @property
    def simulated(self) -> bool:
        return self.status == "simulated"


@dataclass(frozen=True)
class TopologyEntry:
    id: str
    name: str
    source_path: Path
    prerequisites: Prerequisites
    mode_sequences: tuple[EntryModeSequence, ...]
    known_limitations: tuple[str, ...]
    template: TopologyTemplate
    kernel_support: KernelSupport
    notes: str | None


_SCHEMA_VALIDATOR: Any = None


def _schema_validator() -> Any:
    global _SCHEMA_VALIDATOR
    if _SCHEMA_VALIDATOR is None:
        import jsonschema

        schema_path = Path(__file__).with_name("topology-entry.schema.json")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator_cls = jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)
        _SCHEMA_VALIDATOR = validator_cls(schema)
    return _SCHEMA_VALIDATOR


def load_topology_entry(path: str | Path) -> TopologyEntry:
    """Load and validate one entry, fail closed with the file named."""

    source_path = Path(path).resolve()
    try:
        raw_bytes = source_path.read_bytes()
    except OSError as exc:
        raise TopologyLibraryError(f"cannot read {source_path}: {exc}") from exc
    if len(raw_bytes) > MAX_ENTRY_SOURCE_BYTES:
        raise TopologyLibraryError(
            f"{source_path} exceeds the {MAX_ENTRY_SOURCE_BYTES}-byte entry limit"
        )
    try:
        data = yaml.safe_load(raw_bytes)
    except yaml.YAMLError as exc:
        raise TopologyLibraryError(f"cannot parse {source_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise TopologyLibraryError(f"{source_path}: entry root must be an object")

    errors = sorted(
        _schema_validator().iter_errors(data),
        key=lambda error: (len(list(error.absolute_path)),),
        reverse=True,
    )
    if errors:
        error = errors[0]
        pointer = "/".join(str(part) for part in error.absolute_path)
        location = f"/{pointer}" if pointer else "entry root"
        raise TopologyLibraryError(
            f"{source_path.name} does not conform to schema "
            f"{TOPOLOGY_ENTRY_SCHEMA} at {location}: {error.message}"
        )

    entry_id = data["id"]
    if source_path.stem != entry_id:
        raise TopologyLibraryError(
            f"{source_path.name}: id {entry_id!r} must equal the file stem "
            f"{source_path.stem!r} — one entry, one name, no aliases"
        )

    raw_prerequisites = data["prerequisites"]
    voltage = VoltageRange(
        min_v=float(raw_prerequisites["voltage"]["min_v"]),
        max_v=float(raw_prerequisites["voltage"]["max_v"]),
    )
    if voltage.min_v >= voltage.max_v:
        raise TopologyLibraryError(
            f"{entry_id}: prerequisites.voltage.min_v must be below max_v"
        )
    power = PowerRange(
        min_w=float(raw_prerequisites["power"]["min_w"]),
        max_w=float(raw_prerequisites["power"]["max_w"]),
    )
    if power.min_w >= power.max_w:
        raise TopologyLibraryError(
            f"{entry_id}: prerequisites.power.min_w must be below max_w"
        )
    prerequisites = Prerequisites(
        isolation=raw_prerequisites["isolation"],
        direction=raw_prerequisites["direction"],
        voltage=voltage,
        power=power,
    )

    sequences: list[EntryModeSequence] = []
    labels: set[str] = set()
    for index, raw_sequence in enumerate(data.get("mode_sequences", [])):
        label = raw_sequence["label"]
        if label in labels:
            raise TopologyLibraryError(
                f"{entry_id}: duplicate mode-sequence label {label!r}"
            )
        labels.add(label)
        try:
            sequence = mode_sequence_from_dict(raw_sequence["sequence"])
        except ModeSequenceError as exc:
            raise TopologyLibraryError(
                f"{entry_id}: mode_sequences[{index}] ({label!r}): {exc}"
            ) from exc
        sequences.append(
            EntryModeSequence(
                label=label,
                source_model=raw_sequence["source_model"],
                sequence=sequence,
            )
        )

    limitations = tuple(item.strip() for item in data["known_limitations"])
    if any(not item for item in limitations):
        raise TopologyLibraryError(
            f"{entry_id}: known_limitations must not contain blank text"
        )

    raw_template = data["template"]
    template = TopologyTemplate(
        model=raw_template.get("model"),
        subcircuit=raw_template.get("subcircuit"),
    )

    raw_support = data["kernel_support"]
    kernel_support = KernelSupport(
        status=raw_support["status"], note=raw_support["note"].strip()
    )
    if not kernel_support.note:
        raise TopologyLibraryError(
            f"{entry_id}: kernel_support.note must not be blank — a support "
            "claim without a reason cannot be argued with"
        )
    # The claim and its evidence must agree, in BOTH directions: "we can
    # simulate this" with no cycle is an unbacked claim, and "we cannot" with
    # a stored cycle means the entry contradicts itself.
    if kernel_support.simulated and not sequences:
        raise TopologyLibraryError(
            f"{entry_id}: kernel_support.status is 'simulated' but the entry "
            "carries no mode sequence — the claim has no evidence"
        )
    if not kernel_support.simulated and sequences:
        raise TopologyLibraryError(
            f"{entry_id}: kernel_support.status is 'not_simulated' yet the "
            "entry carries a mode sequence extracted from a run"
        )
    if kernel_support.simulated and template.model is None:
        raise TopologyLibraryError(
            f"{entry_id}: a simulated entry must name template.model"
        )

    return TopologyEntry(
        id=entry_id,
        name=data["name"],
        source_path=source_path,
        prerequisites=prerequisites,
        mode_sequences=tuple(sequences),
        known_limitations=limitations,
        template=template,
        kernel_support=kernel_support,
        notes=data.get("notes"),
    )


def load_topology_library(directory: str | Path) -> tuple[TopologyEntry, ...]:
    """Load every ``*.yaml`` entry of a directory, sorted by id, fail closed."""

    root = Path(directory).resolve()
    if not root.is_dir():
        raise TopologyLibraryError(f"{root} is not a directory")
    paths = sorted(root.glob("*.yaml"))
    if not paths:
        raise TopologyLibraryError(f"{root} contains no topology entries")
    entries: list[TopologyEntry] = []
    seen: set[str] = set()
    for path in paths:
        entry = load_topology_entry(path)
        if entry.id in seen:
            raise TopologyLibraryError(f"duplicate topology id {entry.id!r} in {root}")
        seen.add(entry.id)
        entries.append(entry)
    return tuple(entries)


def verify_entry_sequence(
    entry_sequence: EntryModeSequence,
    *,
    root: str | Path,
    duration_rtol: float = 1e-9,
) -> tuple[bool, tuple[str, ...]]:
    """Re-extract the sequence from a fresh run of its source model.

    The stored sequence is a claim with provenance; this function is the
    audit.  It simulates ``source_model`` (resolved against ``root``),
    extracts the last complete cycle at the stored period, and compares via
    :func:`sequences_agree` — content agreement, deliberately not a hash
    comparison, so a pear_sim version bump does not silently invalidate every
    entry while a changed cycle still does.

    GUARD-LOCATED STEPS ARE COMPARED AGAINST THE RUN'S OWN ``rtol``. A step
    whose boundary is a root of h(x, t) = 0 cannot be reproduced more
    accurately than the trajectory it was found on was integrated, so this
    audit never asserts tighter than ``max(duration_rtol, model rtol)`` for
    those. The number is DERIVED from the model rather than chosen: tighten
    the model's tolerance and the audit tightens with it.

    This is not a relaxation of the audit. What the audit exists to catch is
    a stored cycle that no longer matches reality — a changed step order, a
    mode that appeared or vanished, a duration that moved by percent. Mode
    signatures are still compared EXACTLY, and so is the period. Asserting
    located instants below the integration tolerance did not add strength;
    it added a dependence on which machine ran the suite.
    """

    from pear_sim.core.solver import HybridSimulator
    from pear_sim.spec.model import load_model

    model_path = Path(root).resolve() / entry_sequence.source_model
    if not model_path.is_file():
        return (
            False,
            (f"source model {entry_sequence.source_model!r} not found under {root}",),
        )
    model = load_model(model_path)
    result = HybridSimulator(model).simulate()
    observed = extract_mode_sequence(
        result, period_s=entry_sequence.sequence.period_s
    )
    return sequences_agree(
        entry_sequence.sequence,
        observed,
        duration_rtol=duration_rtol,
        guard_duration_rtol=max(duration_rtol, float(model.transient.rtol)),
    )
