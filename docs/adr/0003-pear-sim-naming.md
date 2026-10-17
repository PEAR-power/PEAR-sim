# ADR 0003: PEARSim product naming and package namespace

- Status: Accepted
- Date: 2026-10-03

## Context

During private development the import package, the second executable and every
serialized schema identifier used the short working name `pesim`. The product is
published by PEAR Power as **PEARSim** (Power Electronics Agent-Ready Simulator).

`pesim` collides with an unrelated project already published on PyPI. Shipping a public
release with a top-level `pesim` package would invite import shadowing and installation
confusion for anyone who has both installed.

No public artifact existed before the first release, so renaming everything at once
breaks no external consumer.

## Decision

- Product and human-facing brand: **PEARSim**, published by PEAR Power.
- Python distribution: `pear-sim`; import package: `pear_sim`.
- Executable: `pear-sim` only (no second alias).
- GitHub repository: `PEAR-power/pear-sim`.
- Serialized schema identifiers use the `pear-sim/` prefix, e.g. `pear-sim/v0.1` for
  models and `pear-sim/run/v0.1` for run manifests. JSON Schema `$id` values point at the
  schema files in the repository.

The rename was made once, mechanically, before the first public release. Pinned hashes
in the test suite changed only because the identifier strings are part of the hashed
payloads; the simulated waveforms were checked to be bit-identical before and after.

## Consequences

From the first public release on, a schema namespace change is a wire-format break: it
requires an explicit versioned migration, dual readers for a documented transition
period, and tests against retained artifacts. It must not be hidden inside an unrelated
change.
