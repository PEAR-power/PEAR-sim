# PEARSim validation suite

This directory defines the acceptance contract for PEARSim. The suite is deliberately
runner-independent: [`manifest.json`](./manifest.json) is the machine-readable source
of truth, while [`../docs/validation-plan.md`](../docs/validation-plan.md) explains the
test design and the limits of the evidence. Current implemented coverage is reported
separately in [`status.json`](./status.json); a case listed in the manifest is a planned
claim, not evidence that the corresponding test already passes.

The word *validation* is used for the directory name in the broad engineering sense.
Individual cases are explicitly classified as:

- **verification** — comparison with equations, exact state maps, or conservation laws;
- **compatibility** — comparison with another simulator (here: ngspice) or a published reference result;
- **physical validation** — comparison with experimental measurements not used for
  calibration.

A match with another simulator is useful compatibility evidence, but it is not evidence
that either simulator represents a real converter correctly.

## Planned benchmark matrix

| ID | Scope | Primary oracle | Default CI tier |
|---|---|---|---|
| VAL-001 | Linear DC and transient kernel | analytical solution / matrix exponential | pull request |
| VAL-002 | Buck, boost, and buck-boost in CCM | ARCH hybrid matrices and exact state maps | pull request |
| VAL-003 | DCM, BCM, and ideal-diode events | piecewise analytical event solution | pull request |
| VAL-004 | Single- and three-phase rectifiers | closed-form averages and commutation equations | pull request |
| VAL-005 | Two-level inverter, PWM, dead time, harmonics | switching schedule and Fourier analysis | pull request |
| VAL-006 | Buck small-signal response | averaged transfer function; optional independent-engine comparison | nightly |
| VAL-007 | Semiconductor conduction and DPT losses | curated datasheet curves and DPT definitions | nightly |
| VAL-008 | Foster thermal impedance and convolution | closed-form thermal response | pull request |
| VAL-009 | End-to-end electrothermal converter | held-out measurements | nightly / manual |

Thresholds in the manifest are preliminary engineering gates. They are neither a
normative accuracy specification nor a claim of metrological traceability.

## Data policy

No third-party dataset, datasheet digitization, or other simulator output is vendored by this
documentation layer. Analytical references may be generated locally. External
artifacts must be acquired separately, recorded in an artifact registry, and pinned by
SHA-256 together with their source URL, revision/access date, license or access terms,
test conditions, and extraction method.

The suite must never silently fetch external material during pull-request CI. Tests
whose reference artifact is unavailable are reported as `not-run`, not `pass`.

## Runner contract

A future runner should:

1. validate `manifest.json` against its declared schema version;
2. resolve only locally approved artifacts and verify their hashes;
3. record the PEARSim version, solver configuration, platform, and random seed;
4. preserve raw waveforms, while also producing event-aligned comparison views;
5. calculate metrics exactly as defined in the manifest;
6. fail a required tier when any applicable gate fails or produces a non-finite value;
7. emit both machine-readable results and a compact human report.

Calibration cases and held-out validation cases must be separate. Changing a model,
digitization, alignment rule, or tolerance after inspecting held-out results requires a
new benchmark revision and review.

## CI policy at a glance

- **Pull request:** deterministic analytical cases VAL-001–005 and VAL-008.
- **Nightly:** VAL-006, VAL-007, and compatibility phase VAL-009-A when approved
  reference artifacts are available.
- **Manual/release evidence:** physical phase VAL-009-B. It cannot be reported as
  passed until the experimental dataset and its provenance record are present.

See the detailed plan for case definitions, formulas, gates, provenance requirements,
and known limitations.
