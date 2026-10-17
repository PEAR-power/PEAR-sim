# Canonical boost boundary evidence

PEARSim provides a bounded, retained-run verification producer for one exact canonical
single-switch ideal boost converter. It exercises CCM, the independently located BCM
point, and DCM without making a claim about arbitrary boost schematics, general
descriptor/DAE support, compatibility with another simulator, or hardware behavior.

## Command and process contract

```console
pear-sim sweep boost-boundary examples/boost.yaml \
  --out work/boost-boundary --json
```

An optional strictly ordered list can replace the default load ratios:

```console
pear-sim sweep boost-boundary examples/boost.yaml \
  --ratios 0.9,1,1.1 --out work/boost-smoke --json
```

The output directory must be empty and is never overwritten. Exit `0` means every
requested point and all applicable blocking gates passed. Exit `1` means evidence was
retained with at least one `fail` or `not-run`. Exit `2` means that the request,
topology, model, or output destination is invalid.

The default contract is:

```text
R/R_BCM = [0.25, 0.5, 0.9, 0.99, 1, 1.01, 1.1, 2, 200]
```

Custom ratios must be finite, positive, unique, strictly increasing, include exact
`1.0`, avoid the numerical BCM ambiguity band except at `1.0`, and contain no more
than 64 points.

## Exact topology and polarity

The recognizer accepts exactly one positive DC source, inductor, PWM-controlled ideal
switch, ideal diode, output capacitor, and load resistor, plus optional voltage probes.
There must be exactly one zero-phase PWM control with `0 < duty < 1`. With `gnd` as
the declared ground, the required directed connections are:

```text
Vin: input -> gnd       L: input -> switch
Q:   switch -> gnd      D: switch -> output
C:   output -> gnd      R: output -> gnd
```

The state is `x = [i_L, v_out]`, where inductor current follows `input -> switch` and
output voltage is measured `output -> gnd`. Both are positive in the normal operating
regimes. The cycle begins with the switch on and diode off.

No alternate orientation, sign correction, inferred node mapping, extra parasitic,
or hidden regularization is accepted. A rejected topology is an input error, never a
weakly matched pass.

## Independent periodic affine oracle

The oracle derives its equations locally and does not call the PEARSim circuit
compiler, transient integrator, active-set selector, event manager, or mode classifier.
For load `R`, inductance `L`, capacitance `C`, and source `V_in`, it uses:

```text
ON (Q on, D off):
    di_L/dt   = V_in/L
    dv_out/dt = -v_out/(R C)

FREEWHEEL (Q off, D on):
    di_L/dt   = (V_in - v_out)/L
    dv_out/dt = (i_L - v_out/R)/C

IDLE (Q off, D off):
    i_L       = 0
    dv_out/dt = -v_out/(R C)
```

Matrix exponentials propagate each affine segment. The oracle solves the periodic CCM
fixed point and numerically finds `R_BCM` where cycle-start current reaches zero. The
textbook boundary expression is only a diagnostic seed; it is not substituted for the
numerical root. In DCM, the oracle separately solves the periodic output state and
brackets/refines the first directed inductor-current zero during freewheel.

Every simulated point is seeded from the independent periodic state and runs exactly
one switching period. Initial conditions in the source fixture are not presented as
evidence of steady-state convergence.

## Thirteen blocking gates

Every point records these gate IDs in a stable order:

1. `regime` — expected CCM/BCM/DCM equals the observed mode classification;
2. `boundary_state_nonzero` — nonzero state values agree at switching boundaries;
3. `boundary_state_zero` — expected zero state values remain within normalized limits;
4. `natural_event_time` — the natural diode-current zero occurs at the oracle time;
5. `reverse_current` — no forbidden ideal-diode/inductor reverse current is introduced;
6. `cycle_mean_output` — continuous-time mean output agrees with the oracle;
7. `state_ripple` — current and voltage peak-to-peak state ripple agree;
8. `mode_fractions` — ON/freewheel/idle/unclassified durations agree;
9. `event_semantics` — scheduled and natural event kind/order match the regime contract;
10. `energy_balance` — the recorded full-cycle energy residual stays within its limit;
11. `cycle_periodicity` — normalized cycle-end and cycle-start states close;
12. `unclassified_fraction` — no material time is outside recognized modes; and
13. `mode_fraction_sum` — recorded mode fractions close to one.

The complete nine-point sweep therefore has 117 point gates. Gate values, references,
operators, limits, and statuses are producer data. Human reports display those values
and do not recompute acceptance.

If a point raises before its run bundle exists, PEARSim retains its
`benchmark-point.json`, structured error, and all 13 gates as `not-run`. Completed
points retain their raw waveforms, events, metrics, physics audit, mode coverage, and
hashes even when a gate fails.

## Artifact and schema contract

The non-overwriting sweep directory contains:

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
report/schematic.svg
report/artifact-manifest.json
```

[`boost-boundary-evidence.schema.json`](../schema/boost-boundary-evidence.schema.json)
is a Draft 2020-12 closed union for the producer's specification, oracle, point,
runner result, summary, and confidence documents. Its accepted schema identifiers are
all topology-specific `v0.1` contracts. The root artifact manifest binds every
retained file and uses `sweep-summary.json` as its subject.

`confidence.json` remains `EXPLORATORY`. Its acceptance hashes bind the retained
summary and runner result, while its claim explicitly sets physical validation,
third-party compatibility, hardware sign-off, and general DAE support to false.

The offline report has no scripts or network resources. It shows the sanitized
nominal source schematic (clearly captioned as a template rather than a point-specific
load), recorded status, the numerical BCM load, regime/metrics by ratio, worst gates,
failures, and artifact pointers; it is a presentation projection rather than a second
evaluator.

## CI scope

The ordinary native CI matrix executes a three-point `0.9,1,1.1` boost boundary smoke
on `ubuntu-24.04`, `macos-15`, and `windows-2025` with the two locked Python lines.
That smoke catches installation, topology, event, artifact, and command regressions.
It does not replace the complete nine-point bundle, the canonical buck's 9-by-16 numerical profile,
or its native cross-platform semantic aggregation. A workflow definition is not
evidence that a specific run passed; only retained output bound to its revision can
carry that status.

## Claim boundary

This producer supports mathematical verification of the exact recognized ideal boost
and declared parameter set. Its capability records are experimental for canonical
boost CCM, BCM, and DCM; the bounded evidence producer contract itself is supported.
The broader `converter.dc_dc.bcm_general` and `converter.dc_dc.dcm_general`
capabilities remain unsupported.

Even a complete passing sweep does not establish:

- correctness for an arbitrary boost, buck, buck-boost, or multi-switch topology;
- a general zero-current reduction, complementarity formulation, or descriptor/DAE;
- semiconductor switching loss, thermal behavior, parasitics, EMI, or controls;
- compatibility with any other simulator;
- agreement with measurements or a physical-validation envelope; or
- suitability for production, protection, safety, regulatory, or hardware sign-off.

This producer advances VAL-003 coverage from one canonical topology to two independently
specified topology slices. VAL-003 remains partial until its remaining cases and
release-bound retained evidence are complete.
