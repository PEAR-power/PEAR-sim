# Design specification (`pear-sim/design-spec/v0.1`)

A `DesignSpec` describes the JOB, not the circuit: what a converter must do
and what the customer would like. It is the typed entry point of topology
selection — hard constraints REJECT topology families, soft criteria RANK
candidates on the Pareto front (see [topology comparison](topology-comparison.md)).
This document is the contract; the schema
(`schema/design-spec.schema.json`, packaged byte-identical copy in
`pear_sim/design/`) checks the structural shape, and `pear_sim.design.spec`
enforces the semantics on top.

## File format

```yaml
schema: pear-sim/design-spec/v0.1
name: isolated_aux_supply_48v_to_12v
owner: {id: jdoe, name: "Jane Doe"}

hard:                       # rejects families
  input:
    kind: dc                # dc | ac_single_phase | ac_three_phase
    voltage_v: {min: 36, max: 75}
  output:
    voltage_v: {nominal: 12, tolerance_pct: 5}
    power_w: {max: 60}
  isolation: required       # required | not_required (explicit, never implied)
  direction: unidirectional # unidirectional | bidirectional
  standards:
    emc: ["EN 55032 Class B"]     # explicit [] = "no requirement", a statement
    safety: ["IEC 62368-1"]
  # grid: {scr: 8.5, ...}   # REQUIRED for ac inputs — the grid is a spec
  #                         # parameter, not a run setting (SCR, unbalance,
  #                         # background THD, dip profiles)
  # environment.ambient_c: {min: -10, max: 55}   # optional

soft:                       # Pareto criteria — comparison input
  targets:
    efficiency_pct: 92
    cost: {value: 18, currency: EUR}
  lead_time_weeks: 12
  preferences: ["prefer parts already used in existing BOMs"]

assumptions:                # values a named human assumed, with a why
  - field: environment.ambient_c
    value: {min: -10, max: 55}
    author: {id: jdoe, name: "Jane Doe"}
    rationale: "indoor cabinet assumed; confirm with the customer"
```

## Semantics (all fail-closed)

- **Closed vocabulary.** The field registry in `pear_sim/design/spec.py` is the
  complete list. An unknown field is an error, never ignored.
- **Hard and soft do not mix.** Each field is registered as hard or soft;
  placing it in the other section is a validation error. A soft wish in the
  hard section would silently shrink the candidate space; a hard limit in
  the soft section would let an infeasible design win a comparison.
- **Questions instead of defaults.** A missing required field never gets a
  default. It becomes an *open question* carrying the reason the field is
  needed ("what the answer will drive"). An incomplete spec refuses
  `require_complete()` and has **no `spec_sha256`** — a stable identity for
  a document with holes would let the holes hide.
- **Every value has an author.** Declared values belong to the spec
  `owner`; assumptions require a named human and a rationale. There
  is no code path that manufactures a value out of nothing.
- **Conditional requirement.** `grid` is required exactly when
  `input.kind != dc`: for grid-fed inputs the grid is a specification
  parameter, not a run setting.

## CLI

```
pear-sim spec examples/design_specs/isolated_aux.yaml [--json]
```

Exit 0 = valid and complete (prints `spec_sha256`); exit 1 = valid but
incomplete (prints the open questions with context); exit 2 = invalid
(standard error payload with `code`, e.g. `design_spec.invalid`).
