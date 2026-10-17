# Topology library `pear-sim/topology-entry/v0.1`

Status: early-alpha contract
Normative schema: [`../schema/topology-entry.schema.json`](../schema/topology-entry.schema.json)
Entries: [`../library/topologies/`](../library/topologies/)

The topology library is **data, not code**: one YAML file per topology,
read by a generic loader (`pear_sim.topology`). Nothing in the selection code
names a buck or a boost — adding a topology means adding a file, and a test
enforces that.

## What an entry carries

| Field | Meaning |
|---|---|
| `id` | Identifier; **must equal the file name without the extension** (one entry, one name, zero aliases). |
| `prerequisites` | `isolation` (`none`/`provided`/`optional`), `direction` (`unidirectional`/`bidirectional`), and the `voltage`/`power` ranges in which the topology **makes sense as a candidate** (not device limits). |
| `mode_sequences` | The switching cycle as **the same object the gradient engine uses** — see below. |
| `known_limitations` | **Mandatory and non-empty.** An empty list of limitations is a claim ("this topology has no weaknesses"), not missing data — the format rejects it. |
| `template` | A reference model (`model`, a path relative to the repository root) + optionally `subcircuit` — the name of a declared subcircuit, which IS the topology as a reusable unit. |

## Mode sequence = the engine's object, not a parallel notation

The `sequence` field of each entry is the canonical serialization of
`ModeSequence` (`pear_sim.analysis.gradient.sequence`:
`mode_sequence_as_dict` / `mode_sequence_from_dict` — the only pair, schema
`pear-sim/mode-sequence/v0.1`). The library loader reconstructs from it **exactly
the dataclass that the gradient engine's `solve()` accepts**; the test
`test_the_buck_entry_drives_the_gradient_engine_directly` feeds the engine
a sequence straight from the file and checks the physics (100 V x 40 % = 40 V).
A separate mode notation in the library would mean two sources of truth competing with the
engine — exactly what this design forbids.

**Provenance**: `source_run_sha256` is the identity of the run from which the
sequence was extracted (the same recipe as `run.json`). A human-authored
sequence has `null` — and that absence is visible in every
downstream artifact. Entries shipped in the repository always have provenance.

## Audit: the record is a claim, verification is a run

`verify_entry_sequence` (and `pear-sim topology verify`) starts a fresh run of
`source_model`, extracts the last full cycle at the recorded period and
compares via `sequences_agree` — **content (signatures, exit kinds,
times within rtol), deliberately not a hash**: bumping the pear_sim version does not
silently invalidate all entries, while a changed cycle still does.

```bash
uv run pear-sim topology list --dir library/topologies
uv run pear-sim topology verify --dir library/topologies --root . --json
```

The suite performs the same audit on every run
(`tests/test_topology_library.py`), so an entry whose cycle no longer
matches the reality of its source model breaks the tests.

## Fail-closed rules

- Draft 2020-12 schema enforced on every load (packaged copy,
  byte-identity test as for the model schema);
- unknown fields, empty/near-empty `known_limitations`, `id` ≠ file name,
  inconsistent ranges (`min >= max`), duplicate sequence labels and
  duplicate `id` within the directory — all are refused, naming the offending entry;
- a broken serialized sequence is rejected when the entry is loaded
  (it never travels further as opaque text).

## Scope

Start: buck (the first entry of the vertical slice) + boost (the second entry, added
purely as data — the acceptance criterion of the library). The target is 10–15 topologies
in real-world use. **No encyclopedia** — an entry without a real
application is a maintenance cost without a return.

**State as of 2026-07-28: 13 entries, 11 computable.** The expansion order
is derived from **target conversion chains**, not from a taxonomy of topologies:
first the slots of the canonical drive (transformer → rectifier → inverter → machine),
then isolated and non-isolated DC-DC. A new entry is still nothing but a file.

## Gate: an entry nobody can optimize is rejected

The model of a `simulated` entry must have a **non-empty `parameters:` section**.

The reason lies in the design-vector contract: the design vector sees only values
**named** in that section — a value written inline in a component is
invisible to it. A model with only inline values therefore yields a **zero-dimensional
optimization space**, and nothing announces it: such an entry loads
and passes the re-extraction audit, yet offers the optimizer not a single
dimension. A silent loss of search space is worse
than a refusal, so the format refuses.

The first casualty of this gate: `examples/npc5_three_phase.yaml` (zero named
parameters) — the 5L DNPC entry waits for its parameters to be promoted in the example.

## Cost of the audit

A full re-extraction of all computable entries
(`pear-sim topology verify --dir library/topologies --root .`) costs
**~31 s** and runs in the suite on every commit. This is the real price of an entry,
which is why it is recorded here: an entry whose reference model runs for a long time
burdens everyone's gate. For this reason an AFE entry **has not been shipped** —
its natural model (`examples/grid_following_converter.yaml`)
costs ~105 s per run, more than three times the entire current audit.
