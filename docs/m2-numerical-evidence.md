# Canonical-buck numerical evidence

PEARSim provides a retained numerical-evidence matrix and native cross-platform aggregation
for one deliberately narrow subject: the recognized, ideal, single-switch canonical
buck used by `examples/buck_dcm.yaml`. It verifies PEARSim against the existing
independent periodic affine oracle. It does not establish general converter support,
descriptor/DAE coverage, agreement with another simulator, or agreement with
hardware.

## Full profile contract

The versioned `full-v1` profile is bounded and immutable for a given source revision.
It evaluates nine load ratios around the independently located BCM boundary:

```text
0.25, 0.5, 0.9, 0.99, 1.0, 1.01, 1.1, 2.0, 200.0
```

The ratios span declared CCM, numerical BCM, and DCM points. Each point is evaluated
with 16 unique numerical configurations:

| Family | Configurations | Family question |
| --- | ---: | --- |
| tolerance | 5 | Do normalized oracle errors show a positive tightening trend or a final-three plateau below 1% of their budgets? |
| maximum step | 5 | Do normalized oracle errors show a positive refinement trend or a final-three plateau below 1% of their budgets? |
| output grid | 3 | Do different requested sample counts preserve the signed state, event, output, mode, and energy fields in the physical semantic signature? |
| solver | 4 | Do DOPRI5, DOP853, Radau, and BDF preserve event and accepted-mode sequences and keep the signed semantic signature inside the separately declared 5% cross-solver envelope? |

The nominal DOPRI5 configuration belongs to both the tolerance and output-grid
families. There are therefore 17 family memberships but 16 executed configurations.
The complete platform matrix contains:

```text
9 operating points × 16 configurations = 144 retained cells
144 cells × 11 independent-oracle gates = 1,584 cell gates
4 families × 9 operating points = 36 family gates
```

Every cell applies the same eleven local independent-oracle gates as the
canonical-buck boundary benchmark:

1. regime;
2. nonzero boundary state;
3. zero boundary state;
4. natural-event time;
5. reverse current;
6. cycle-mean output;
7. mode fractions;
8. energy balance;
9. cycle periodicity;
10. unclassified time fraction;
11. mode-fraction closure.

A family gate cannot turn a failed or incomplete source cell into a pass. Tolerance
and maximum-step gates record convergence or the declared small-error plateau;
output-grid gates require different sample counts and a stable physical semantic
signature; solver gates require consistent event and accepted-mode sequences plus
bounded signed semantic values. These family results remain first-class evidence and
are verified during platform aggregation; they are not inferred from cell counts.

## Native platform run

Run the full profile into a new or empty directory:

```console
uv run pear-sim verify numerics buck-boundary examples/buck_dcm.yaml \
  --profile full-v1 \
  --source-revision REVISION \
  --out numerical-evidence \
  --json
```

`--source-revision` is evidence binding, not display metadata. The three native CI
jobs use the same GitHub commit SHA. Each platform bundle additionally records the
PEARSim tool version, resolved model hashes, an exact hash of its local independent
oracle artifact, and a stable oracle-contract identity. Aggregation requires the
source, tool, model, profile, and oracle-contract identities to agree. Local oracle
artifact hashes may differ because each native platform computes its own reference;
each hash must still bind that platform's verified artifact, and the independently
computed BCM boundaries must agree within the published absolute/relative tolerance.
The aggregator also rejects duplicate or missing platforms, mismatched cell IDs,
altered semantic-field policies, inconsistent family evidence, and manifest/hash
inconsistencies.

The platform bundle contains at least:

| Artifact | Meaning |
| --- | --- |
| `numerical-profile.json` | Fully resolved `full-v1` profile, configuration membership, and cross-platform tolerance policy. |
| `oracle.json` | Independent canonical-buck reference and numerical BCM boundary for the nine points. |
| `platform-evidence.json` | Platform identity, source/model/tool binding, local oracle-artifact hash and stable oracle identity, all cell and family statuses, physical semantic signatures, and summary counts. |
| `cells/<configuration>/<point>/cell.json` | One retained cell record, including configuration, reference, observation, local oracle gates, 22-field semantic signature, numerics, error, and run reference. |
| `cells/<configuration>/<point>/run/` | Complete ordinary PEARSim run bundle when that cell executed successfully. |
| `report/numerical-report.json` | Hash-bound presentation model containing the recorded evidence without recomputing it. |
| `report/numerical-report.html` | Offline, script-free human projection of the same recorded evidence. |
| `artifact-manifest.json` | Exact SHA-256 and byte-count coverage of regular files in the immutable evidence bundle. |

The structural contracts are published in `schema/numerical-profile.schema.json`,
`schema/platform-evidence.schema.json`, and
`schema/cross-platform-evidence.schema.json`. Semantic checks remain fail-closed in
the producers and aggregator; JSON Schema alone is not an acceptance decision.

The destination is non-overwriting. A cell exception is retained as structured
`not-run` evidence; it is not omitted or converted into pass. Successful runs remain
available for waveform and event inspection. Machine consumers should treat the
JSON evidence and hashes as authoritative; any HTML report is a presentation-only
projection and must not recompute gates.

## Cross-platform aggregation

Produce one bundle each on native Linux, macOS, and Windows, then aggregate them
outside all three immutable input directories:

```console
uv run pear-sim verify platforms \
  platform-inputs/Linux \
  platform-inputs/Darwin \
  platform-inputs/Windows \
  --require Linux,Darwin,Windows \
  --out cross-platform-evidence \
  --json
```

The required evidence system names are `Linux`, `Darwin`, and `Windows`; the GitHub
artifact carrying the Darwin bundle is named `m22-macOS`. The aggregate contains
`cross-platform-evidence.json`, `report/numerical-report.json`,
`report/numerical-report.html`, and the root `artifact-manifest.json`. The report is
created before manifest finalization and is therefore hash-covered like every other
regular bundle artifact.

Cross-platform comparison does not reuse normalized local oracle-gate errors as a
proxy for physical agreement. Those eleven gates remain separate, blocking local
evidence. Instead, every cell carries a flat `semantic_signature` with 22 typed,
sign-preserving observed fields:

- operating regime;
- inductor current and output voltage at cycle start, switch-off, and cycle end;
- natural diode-zero-event presence and time;
- cycle-mean output voltage and minimum inductor current;
- ON, freewheel, idle, and unclassified mode fractions;
- signed relative energy-balance residual and signed cycle-state deltas;
- semantic event and accepted-topology sequence digests and counts.

This is a semantic record, not a cryptographic signature; bundle hashes bind its
serialized evidence. Categorical fields must match exactly. Numeric fields retain
their physical sign and pass only when:

```text
max(value) - min(value) <= max(absolute_tolerance,
                               relative_tolerance × max(abs(value)))
```

The complete aggregate therefore retains exactly:

```text
144 cells × 22 semantic fields = 3,168 cross-platform comparisons
```

A failed local oracle gate remains blocking even when physical values agree across
platforms. An unavailable signature produces explicit `not-run` records for its
required fields rather than shortening the matrix. Study entries retain and verify
the contributing platform family evidence and roll up the relevant cross-platform
cell/signature statuses; a missing or inconsistent family record cannot be hidden by
small platform drift.

## CI retention and status semantics

`.github/workflows/numerical-evidence.yml` runs on pull requests, pushes to `main`,
and manual dispatch. It uses Python 3.12.13, single-thread BLAS settings, and fixed
runner labels (`ubuntu-24.04`, `macos-15`, and `windows-2025`). GitHub refreshes the
contents of those hosted images over time, so the evidence records the concrete
runner/image metadata instead of claiming that a label alone is an immutable image.
Third-party actions are pinned to immutable commit SHAs and `uv` is pinned to
`0.11.28`. Each platform upload is attempted with `if: always()` and retained for
30 days under:

```text
m22-Linux
m22-macOS
m22-Windows
```

The aggregation job also uses `if: always()`, downloads those three named artifacts,
runs the fail-closed aggregator, and attempts to retain `m22-cross-platform` for 30
days. This preserves generated failure evidence. It does not label a missing bundle
as pass; infrastructure failure or an absent artifact leaves the workflow non-green.

Evidence statuses are:

- `pass`: all required source cells and applicable family/comparison gates passed;
- `fail`: at least one blocking source or comparison gate failed;
- `not-run`: required evidence did not complete or could not be evaluated;
- invalid request/input/artifact: command error; no engineering pass is published.

The workflow definition is not itself proof that Linux, Darwin, and Windows agree.
Only a concrete run whose three native bundles and aggregate are retained with
`pass` status supplies that evidence for its recorded source revision.

## Claim boundary

This evidence may support this statement only:

> For the exact recognized ideal canonical buck, the recorded source revision passed
> or failed the declared `full-v1` numerical profile and, when an aggregate exists,
> the declared semantic cross-platform tolerances on the named native platforms.

It does not support any of the following statements:

- arbitrary buck schematics, boost or buck-boost DCM/BCM are verified;
- PEARSim is a general descriptor/DAE simulator;
- PEARSim is compatible with SPICE or another third-party tool;
- ideal-switch evidence predicts semiconductor loss, thermal behavior, EMI, or SOA;
- a result is suitable for protection settings, safety decisions, regulatory work,
  production design, or hardware sign-off.

The project remains an experimental early alpha. Cross-platform verification,
third-party compatibility, and physical validation remain separate evidence axes.
