# Agent evidence workflow

PEARSim provides a fail-closed tool surface around the existing experimental simulator. It
does not broaden PEARSim into a general DAE solver and does not make a physical
validation claim. The vertical slice has three parts:

1. a machine-readable capability catalog;
2. evaluation of declared KPI expectations against a verified immutable run; and
3. an independently referenced canonical-buck load sweep through CCM, BCM, and DCM.

A separate producer with the same fail-closed process semantics for the
exact canonical ideal boost exists. Its oracle, topology, 13 gates, artifacts, and limits
are documented in [canonical boost boundary evidence](m2-boost-boundary-evidence.md).

## Agent commands and exit codes

```console
pear-sim capabilities --json
pear-sim capabilities --id converter.buck.bcm.canonical_ideal --json

pear-sim simulate examples/buck.yaml --out work/buck-run --json
pear-sim evaluate work/buck-run \
  --expectations examples/buck_expectations.json \
  --out work/buck-evidence --json

pear-sim sweep buck-boundary examples/buck_dcm.yaml \
  --out work/buck-boundary --json

pear-sim sweep boost-boundary examples/boost.yaml \
  --out work/boost-boundary --json
```

The process contract distinguishes an engineering result from an input/tool error:

| Exit | Meaning |
| ---: | --- |
| `0` | The command completed and all applicable blocking checks passed. Explicit user-declared `not_applicable` criteria are excluded. |
| `1` | Evaluation retained a blocking `fail`, `not_comparable`, or `not_run` (including no applicable blocking criterion), or the sweep retained a `fail`/`not-run` point. |
| `2` | The request, input contract, topology, artifact integrity, or output location is invalid. |

For exit `1`, JSON is still written to standard output and points to the retained
evidence. Invalid requests use standard error. A missing signal, unknown metric,
incompatible unit/window, non-finite value, unsupported topology, or failed run can
never become a pass through fallback.

## Capability catalog

`pear-sim capabilities` emits `pear-sim/capabilities/v0.1`. Every entry is explicitly
`supported`, `experimental`, or `unsupported`. An unlisted identifier is returned as
`unsupported`; absence is not interpreted as future or implicit support. The catalog
also carries allowed and prohibited uses and states that neither physical-validation
nor third-party-compatibility evidence exists for this slice.

## Expectations and evidence

The input schema is `pear-sim/expectations/v0.1`; its structural JSON Schema is
[`../schema/expectations.schema.json`](../schema/expectations.schema.json). The
runtime validator is additionally responsible for semantic rules that JSON Schema
does not conveniently express, including unique IDs, finite values, unambiguous
comparators, ordered windows, and bounded document sizes.

Each criterion declares:

- a signal, recorded metric, and exact unit;
- either a target with absolute/relative tolerance or inclusive min/max bounds;
- whether the criterion is blocking; and
- optional explicit non-applicability with a reason.

`pear-sim evaluate` requires a run-level `artifact-manifest.json` and verifies every
artifact declared by the supported run manifest before reading `metrics.json`. It
writes a new, non-overwriting directory
outside the run:

```text
evidence.json
confidence.json
artifact-manifest.json
```

The v0.1 evaluator accepts only `pear-sim/run/v0.1` bundles produced by the
`reference-python` backend. Time aggregates, sampled extrema aliases, and ripple
criteria must carry their exact recorded aggregation/extrema/ripple method; an unknown
or missing method is `not_comparable`, never an inferred pass.

The evidence records source/run/metrics/expectations hashes, every criterion result,
and `pass`, `fail`, `not_applicable`, `not_comparable`, or `not_run`. The confidence
card remains `EXPLORATORY` even when all user criteria pass: user-selected KPI bounds
do not establish a verified parameter envelope.

The older `pear-sim/review-expectations/v0.1-draft` document remains an explicit legacy
input so existing review packets can still be evaluated. New files should use the
stable v0.1 schema.

## Canonical buck boundary sweep

The default sweep uses:

```text
R/R_BCM = [0.25, 0.5, 0.9, 0.99, 1, 1.01, 1.1, 2, 200]
```

The reference module locally derives the ON, freewheel, and idle affine equations,
propagates them with matrix exponentials, independently solves periodic CCM/DCM states
and zero-current events, and numerically locates their BCM boundary. It does not
import PEARSim's circuit compiler, transient solver, mode classifier, or event manager.
`R_BCM` is the numerical root
where the periodic CCM cycle-start inductor current reaches zero; it is not silently
replaced by the small-ripple textbook approximation.

Each point starts from the independent periodic state, runs one complete switching
period, and is retained in its own run bundle. The comparison applies eleven blocking
gates:

1. expected versus observed regime;
2. nonzero switching-boundary states;
3. zero-current boundary states;
4. natural-event time;
5. forbidden reverse current;
6. cycle-mean output voltage;
7. ON/freewheel/idle/unclassified fractions;
8. full-cycle energy balance;
9. normalized cycle-end versus cycle-start state periodicity;
10. unclassified time fraction; and
11. closure of the recorded mode fractions to one.

The nine-point default sweep therefore records 99 point-comparison gates when every
point executes. The last three gates prevent a numerically plausible waveform from
passing when the one-cycle warm start drifts, time is left outside the canonical mode
partition, or independently reported mode durations do not close the full period.

A failed or unexecuted point does not erase the other runs. An executed point retains
its run and the recorded mix of `pass`/`fail` gates. If execution raises before a run
bundle exists, its `benchmark-point.json`, error, and eleven `not-run` gates remain in
the aggregate. Every completed point retains raw waveforms, events, metrics, physics
audit, mode coverage, and hashes.

The sweep directory contains:

```text
benchmark-spec.json
oracle.json
benchmark-result.json
sweep-summary.json
confidence.json
artifact-manifest.json
points/<ordered-point>/benchmark-point.json
points/<ordered-point>/run/*
report/sweep-report.json
report/sweep-report.html
report/artifact-manifest.json
```

The offline report has no scripts or network resources. It displays the recorded
status, numerical `R_BCM`, mean output voltage and minimum inductor current versus
`R/R_BCM`, the BCM marker, regimes, worst gates, failures, and artifact pointers. It
does not recompute producer gates.

Custom `--ratios` must be finite, positive, strictly increasing, unique, contain exact
`1.0`, remain outside the numerical BCM ambiguity band except at `1.0`, and stay
within the 64-point process limit.

## Canonical boost boundary sweep

The boost producer uses the same default ratios and retention rule, but it is a
separate topology-specific contract and does not reuse buck equations or infer boost
support from a buck result. Its independent affine oracle solves periodic CCM and DCM
states, numerically locates the BCM load, and refines the diode-current zero event.
The producer records the buck sweep's eleven gate classes plus explicit state-ripple
and event-semantics gates: 13 gates per point and 117 for the complete default sweep.

Only the exact recognized source–inductor–switch–diode–capacitor–load topology is in
scope. Completed points retain their raw run bundles, while failed execution retains
the point record and 13 `not-run` gates. The closed structural contract is
[`boost-boundary-evidence.schema.json`](../schema/boost-boundary-evidence.schema.json).
The complete equations, polarity, artifact layout, CI scope, and claim boundary are
in [canonical boost boundary evidence](m2-boost-boundary-evidence.md).

## Claim boundary of the boundary sweeps

These slices are mathematical verification evidence for two separately recognized
canonical single-switch ideal topologies—buck and boost—and each exact declared
parameter set. They are useful for catching regime/event/metric regressions and for
an agent–human review loop. Neither topology inherits evidence from the other. They
are not evidence for:

- arbitrary buck or boost schematics, or a general descriptor/DAE formulation;
- buck-boost DCM/BCM or other unrecognized converter topologies;
- validated three-phase inverter operation (the six-step example has only partial checks);
- semiconductor losses, temperature, parasitics, EMI, or controls;
- agreement with another simulator (no independent-engine comparison exists for these slices);
- agreement with hardware; or
- production, protection, safety, or regulatory sign-off.

The basic sweep's confidence card deliberately records full convergence and cross-platform
evidence as `not-run`; a basic sweep cannot infer those results. A separate,
source-bound producer for the complete declared numerical profile and native platform
aggregate exists. It does not retroactively promote a basic sweep, and its GitHub workflow is
not a pass claim until a concrete run retains passing artifacts. See
[canonical-buck numerical evidence](m2-numerical-evidence.md). The boost slice likewise adds a bounded
canonical-boost producer and native CI smoke, not a cross-platform full numerical
matrix or a physical-validation claim.
