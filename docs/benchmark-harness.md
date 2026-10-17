# Benchmark Harness v1

`pear_sim.benchmark_harness` is the topology-neutral retention layer for bounded
verification sweeps. It exists to stop artifact, failure, and runner behavior from
being reimplemented for every converter while preserving independent oracle physics.

## Shared responsibilities

The harness owns:

- ordered, non-overwriting point directories;
- fail-closed conversion of execution errors to `not-run` gates;
- bounded retained error text where required by a versioned schema;
- `benchmark-point.json` materialization and run-manifest binding;
- topology-neutral point summaries, worst-gate projection, failure projection, and
  root artifact manifests;
- validation that every family declares a unique ordered gate set.

`pear_sim.benchmarks.affine` separately provides dimension-independent matrix-exponential
evaluation for affine oracle equations. It is tested against closed-form scalar and
constant-derivative cases.

## Deliberately not shared

The harness does not import the PEARSim circuit compiler, simulator, event manager,
mode classifier, or any topology oracle. Each benchmark family still owns:

- exact topology recognition and polarity conventions;
- derivation of every mode matrix and forcing vector;
- periodic CCM solution and BCM/DCM event roots;
- state, ripple, event-sequence, and energy expectations;
- comparison gates and their numerical tolerances;
- projection from a completed PEARSim run into its observation contract.

This boundary prevents production equations from generating their own expected
results. Sharing artifact mechanics is not evidence that two topology oracles are
independent of one another; the claim remains only that oracle equations and the
production kernel are independently implemented.

## Adding a benchmark family

Before a new retained family is accepted, it must:

1. define a versioned `BoundaryHarnessContract` with unique ordered gate IDs;
2. implement topology-local oracle, observation, and comparison functions;
3. supply point initial-condition and summary-metric projections;
4. demonstrate that execution errors retain every requested point as `not-run`;
5. retain hashes for point and run artifacts without overwriting prior evidence;
6. pass an import-boundary test proving that oracle utilities do not import the
   production solver or circuit compiler;
7. document its mathematical source, envelope, thresholds, and claim boundary.

The harness is infrastructure, not a new validation result. Existing VAL-003 status
remains partial and all previous topology-specific limitations still apply.
