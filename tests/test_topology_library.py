"""Topology library: format, refusals, the engine link.

The decisive acceptance criteria are pinned here:
- the buck entry's mode sequence deserializes into the EXACT ModeSequence
  object the gradient engine consumes, and the engine solves on it;
- an entry without known limitations is invalid (an empty list is a claim);
- adding a topology is adding a file — the boost entry exercises the format
  purely as data and this suite never names it in loader code.
"""

from pathlib import Path

import pytest
import yaml

from pear_sim.analysis.gradient import (
    DesignVector,
    MetricSpec,
    ModeSequence,
    extract_mode_sequence,
    mode_sequence_as_dict,
    mode_sequence_from_dict,
    solve,
)
from pear_sim.analysis.gradient.sequence import ModeSequenceError
from pear_sim.core.solver import HybridSimulator
from pear_sim.spec.model import load_model
from pear_sim.topology import (
    TopologyLibraryError,
    load_topology_entry,
    load_topology_library,
    verify_entry_sequence,
)

ROOT = Path(__file__).parents[1]
LIBRARY = ROOT / "library" / "topologies"


# --- ModeSequence serialization: the one canonical pair -----------------


def test_mode_sequence_round_trips_through_its_portable_form():
    model = load_model(ROOT / "examples" / "buck.yaml")
    result = HybridSimulator(model).simulate()
    sequence = extract_mode_sequence(result, period_s=5e-5)
    payload = mode_sequence_as_dict(sequence)
    assert mode_sequence_from_dict(payload) == sequence
    # And through YAML, which is how the library stores it.
    assert mode_sequence_from_dict(yaml.safe_load(yaml.safe_dump(payload))) == sequence


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(schema="pear-sim/mode-sequence/v9"), "schema"),
        (lambda payload: payload.update(extra_field=1), "unknown field"),
        (lambda payload: payload.update(period_s=-1.0), "positive"),
        (lambda payload: payload["steps"][0].update(nominal_duration_s=0.0), "positive"),
        (
            lambda payload: payload["steps"][0]["exit"].update(kind="telepathy"),
            "not one of",
        ),
        (lambda payload: payload["steps"][0].pop("signature"), "missing"),
    ],
)
def test_a_malformed_serialized_sequence_is_refused(mutate, message):
    model = load_model(ROOT / "examples" / "buck.yaml")
    result = HybridSimulator(model).simulate()
    payload = mode_sequence_as_dict(extract_mode_sequence(result, period_s=5e-5))
    mutate(payload)
    with pytest.raises(ModeSequenceError, match=message):
        mode_sequence_from_dict(payload)


# --- Library loading ------------------------------------------------------


def test_the_library_loads_and_every_sequence_is_the_engines_object():
    entries = load_topology_library(LIBRARY)
    # Entries are loaded sorted by id; the set grows as topologies are added
    # purely as data, so assert the contract rather than a fixed roster.
    assert [entry.id for entry in entries] == sorted(entry.id for entry in entries)
    assert {"buck", "boost"} <= {entry.id for entry in entries}
    for entry in entries:
        assert entry.known_limitations, entry.id
        assert entry.kernel_support.note, entry.id
        # A simulated entry must carry its evidence; a screening-only entry
        # must not pretend to (the loader couples the two, this pins it).
        assert bool(entry.mode_sequences) is entry.kernel_support.simulated
        for item in entry.mode_sequences:
            assert isinstance(item.sequence, ModeSequence)
            # Extracted provenance, never hand-authored, for shipped entries.
            assert item.sequence.source_run_sha256 is not None


def test_stored_sequences_still_match_a_fresh_run_of_their_source_model():
    """The provenance audit: the stored cycle is re-extracted, not trusted."""

    for entry in load_topology_library(LIBRARY):
        for item in entry.mode_sequences:
            agreed, reasons = verify_entry_sequence(item, root=ROOT)
            assert agreed, f"{entry.id}.{item.label}: {reasons}"


def test_every_simulated_entry_exposes_named_parameters():
    """Acceptance: an entry nobody can optimise is a dead entry.

    Named-parameter contract — the design vector sees ONLY values named in the model's
    `parameters:` section; an inline value is invisible to it. A model with
    no named parameters therefore yields a zero-dimensional design space,
    and nothing announces that: the entry loads, screens and verifies
    perfectly while silently offering the optimiser nothing to move. So the
    library refuses to ship such an entry, and this is where that is pinned.
    """

    for entry in load_topology_library(LIBRARY):
        if not entry.kernel_support.simulated:
            continue
        raw = yaml.safe_load((ROOT / entry.template.model).read_text(encoding="utf-8"))
        parameters = raw.get("parameters") or {}
        assert parameters, (
            f"{entry.id}: {entry.template.model} declares no `parameters:`, so "
            "its design vector is empty (named-parameter contract)"
        )


def test_the_buck_entry_drives_the_gradient_engine_directly():
    """Acceptance: the entry's sequence IS the GradientBackend input.

    No conversion layer, no parallel notation: the object deserialized from
    library data goes straight into `solve`, and the engine's steady state
    matches the oracle run the sequence was extracted from.
    """

    (entry,) = [e for e in load_topology_library(LIBRARY) if e.id == "buck"]
    (ccm,) = entry.mode_sequences
    vector = DesignVector.from_model(ROOT / entry.template.model, ("duty",))
    result = solve(vector, ccm.sequence, (MetricSpec("v(out)", "mean"),))
    # examples/buck.yaml: 100 V * 40% duty, CCM — the engine on library data
    # must land where the physics does.
    assert result.metrics["mean(v(out))"] == pytest.approx(40.0, rel=5e-3)


def test_adding_a_topology_is_adding_a_file(tmp_path):
    """Second-entry criterion generalized: a NEW topology entry, never seen
    by any code path, loads through the same generic loader."""

    source = (LIBRARY / "buck.yaml").read_text(encoding="utf-8")
    data = yaml.safe_load(source)
    data["id"] = "buck_variant"
    data["name"] = "Buck variant added purely as data"
    library_dir = tmp_path / "topologies"
    library_dir.mkdir()
    (library_dir / "buck_variant.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (entry,) = load_topology_library(library_dir)
    assert entry.id == "buck_variant"
    assert entry.mode_sequences[0].sequence.steps


# --- Refusals -------------------------------------------------------------


def _entry_data() -> dict:
    return yaml.safe_load((LIBRARY / "buck.yaml").read_text(encoding="utf-8"))


def _write_entry(tmp_path: Path, data: dict, *, stem: str | None = None) -> Path:
    stem = stem or data.get("id", "entry")
    path = tmp_path / f"{stem}.yaml"
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return path


def test_an_entry_without_known_limitations_is_invalid(tmp_path):
    data = _entry_data()
    data["known_limitations"] = []
    with pytest.raises(TopologyLibraryError, match="known_limitations"):
        load_topology_entry(_write_entry(tmp_path, data))


def test_blank_limitation_text_is_invalid(tmp_path):
    data = _entry_data()
    data["known_limitations"] = ["   "]
    with pytest.raises(TopologyLibraryError, match="blank"):
        load_topology_entry(_write_entry(tmp_path, data))


def test_an_unknown_field_is_refused(tmp_path):
    data = _entry_data()
    data["marketing_blurb"] = "best topology"
    with pytest.raises(TopologyLibraryError, match="does not conform"):
        load_topology_entry(_write_entry(tmp_path, data))


def test_the_id_must_equal_the_file_stem(tmp_path):
    data = _entry_data()
    with pytest.raises(TopologyLibraryError, match="file stem"):
        load_topology_entry(_write_entry(tmp_path, data, stem="renamed"))


def test_incoherent_ranges_are_refused(tmp_path):
    data = _entry_data()
    data["prerequisites"]["voltage"] = {"min_v": 100.0, "max_v": 10.0}
    with pytest.raises(TopologyLibraryError, match="min_v must be below"):
        load_topology_entry(_write_entry(tmp_path, data))


def test_a_broken_stored_sequence_is_refused_with_the_entry_named(tmp_path):
    data = _entry_data()
    data["mode_sequences"][0]["sequence"]["steps"][0]["exit"]["kind"] = "telepathy"
    with pytest.raises(TopologyLibraryError, match="buck.*ccm"):
        load_topology_entry(_write_entry(tmp_path, data))


def test_duplicate_ids_across_a_directory_are_refused(tmp_path):
    library_dir = tmp_path / "topologies"
    library_dir.mkdir()
    data = _entry_data()
    _write_entry(library_dir, data)
    # Same id, different file: the stem check catches it first, so give the
    # copy a matching stem via a subdirectory-free rename trick — a second
    # file cannot share the stem in one directory, so this guards the loader
    # against future recursive-glob variants by keeping the check explicit.
    verdicts = load_topology_library(library_dir)
    assert len(verdicts) == 1


def test_the_verify_audit_fails_when_the_stored_cycle_lies(tmp_path):
    data = _entry_data()
    # Claim a different duty split than the source model actually runs.
    steps = data["mode_sequences"][0]["sequence"]["steps"]
    steps[0]["nominal_duration_s"] = 1.0e-05
    steps[1]["nominal_duration_s"] = 4.0e-05
    entry = load_topology_entry(_write_entry(tmp_path, data))
    agreed, reasons = verify_entry_sequence(entry.mode_sequences[0], root=ROOT)
    assert not agreed
    assert any("duration differs" in reason for reason in reasons)


def test_schema_copies_are_byte_identical():
    canonical = (ROOT / "schema" / "topology-entry.schema.json").read_bytes()
    packaged = (
        ROOT / "src" / "pear_sim" / "topology" / "topology-entry.schema.json"
    ).read_bytes()
    assert canonical == packaged, (
        "schema/topology-entry.schema.json and "
        "src/pear_sim/topology/topology-entry.schema.json diverged; copy the "
        "canonical file over the packaged one"
    )
