# PEARSim verification and validation plan

Status: draft acceptance plan, 2026-07-18
Machine-readable contract: [`../validation/manifest.json`](../validation/manifest.json)

## 1. Objective and claim boundary

This plan defines the evidence needed before PEARSim can be treated as a dependable
headless power-electronics simulator. It is intentionally layered:

1. **Numerical and model verification** asks whether the implemented equations are
   solved correctly.
2. **Compatibility testing** asks whether PEARSim reproduces a documented result of
   another simulator under equivalent assumptions.
3. **Physical validation** asks whether the complete model agrees with held-out
   measurements, with uncertainty and boundary conditions recorded.

These claims are not interchangeable. Agreement with another simulator is valuable for
a workflow that must stay compatible with that tool, but it cannot establish agreement
with hardware. Likewise,
a good result for one converter does not validate untested device models, switching
regimes, controls, or thermal boundary conditions.

The terminology follows the verification/validation distinction promoted by
[ASME VVUQ](https://www.asme.org/codes-standards/publications-information/verification-validation-uncertainty).
All tolerances below are provisional engineering gates for continuous integration;
they are not standards-derived, normative accuracy guarantees, or statements of
measurement traceability.

## 2. Common comparison rules

### 2.1 Metrics

For a signal with declared engineering scale `s`:

```text
relative_error = |actual - reference| / max(|reference|, s * 1e-12)

nrmse = rms(actual - reference) /
        max(rms(reference), s * 1e-12)

energy_balance_relative =
    |E_in - E_out - E_loss - delta_E_stored| /
    max(|E_in|, E_scale * 1e-12)

event_time_relative = |t_actual - t_reference| / T_switch
```

Temperature comparisons use absolute kelvin error; a temperature difference in kelvin
is numerically equal to a difference in degrees Celsius. THD is stored as a fraction,
so a tolerance of `0.01` means one percentage point. Bode magnitude is compared in dB,
and phase is unwrapped before comparison in degrees.

Every test case must declare its current, voltage, power, energy, and temperature
scales. A non-finite metric always fails a blocking gate. Aggregate reports must retain
per-case values; a mean must never hide an individual failure.

### 2.2 Discontinuities and event alignment

A tiny time shift at an ideal switching edge can produce a large sample-by-sample
error even when both trajectories are otherwise identical. Therefore the suite uses
two complementary checks:

- event times are compared directly against their reference;
- waveform metrics may align matching, labeled events and exclude a half-width of
  `max(5 * allowed_event_time_error, 1e-4 * T_switch)` around each edge.

Raw, unaligned waveforms must still be stored. Alignment is permitted only between
events with the same semantic label, never by unconstrained cross-correlation or a
free time warp.

PEARSim implements a deliberately narrower `pear-sim compare` subset for two
manifest-verified PEARSim runs with the same resolved physics and measurement
contract. It supports absolute time or bounded piecewise-linear alignment anchored by
matching labeled events, plus explicit edge exclusion and minimum reference-window
coverage. It does not infer signal mappings, polarity, gain, units, event identity, or
compatibility; incompatible valid bundles are retained as `not_comparable`.

### 2.3 Repeatability record

Each result bundle should contain:

- PEARSim commit/version and manifest revision;
- complete circuit/netlist and parameter values, including units;
- initial conditions, stop time, warm-up interval, and measurement window;
- solver, tolerances, maximum step, event localization settings, and random seed;
- operating system, architecture, and floating-point precision;
- raw waveforms, event log, calculated metrics, and pass/not-run/fail status;
- reference-artifact identifiers and SHA-256 hashes.

## 3. Benchmark definitions

### VAL-001 — linear circuit kernel

Purpose: isolate stamping, algebraic constraints, controlled sources, transient
integration, and conservation before introducing switches.

Cases include resistive DC networks, RC charge/discharge, natural and driven series and
parallel RLC networks, and independent plus controlled voltage/current sources. DC
oracles are direct linear-system solutions. Transient oracles are closed form where
simple and independently evaluated matrix exponentials otherwise.

Blocking gates:

| Quantity | Gate |
|---|---:|
| DC node voltage and branch current relative error | `<= 1e-9` |
| Transient state NRMSE | `<= 1e-5` |
| Normalized KCL and KVL residual | `<= 1e-9` |
| Relative energy-balance residual | `<= 1e-5` |

Optional Xyce or ngspice runs can diagnose disagreements, but the analytical oracle is
authoritative. See [Xyce](https://github.com/Xyce/Xyce) and
[ngspice quality assurance](https://ngspice.sourceforge.io/quality.html).

### VAL-002 — buck, boost, and buck-boost in CCM

Purpose: verify piecewise topology construction and state propagation at every switch
boundary.

The primary reference is the
[ARCH DC-DC benchmark](https://cps-vo.org/sites/cps-vo.org/files/cpsvo_file_nodes/arch14_submission_12pdf.pdf),
which publishes hybrid models for buck, boost, and buck-boost converters. The nominal
buck neighborhood includes `Vs = 11.95–12.05 V`, `R = 0.95–1.05 ohm`,
`C = 23.75–26.25 uF`, `L = 47.5–52.5 uH`, switching period
`24.5–25.5 us`, and duty ratio `0.6`. The suite covers nominal and parameter-corner
startup plus periodic steady state.

For each constant-topology interval, an independent evaluator propagates the affine
state equations with a matrix exponential. This avoids using the PEARSim integrator to
generate its own oracle.

Blocking gates:

| Quantity | Gate |
|---|---:|
| Every state at each switching boundary | relative error `<= 1e-5` |
| Cycle mean and peak-to-peak ripple | relative error `<= 0.1%` |
| Cycle energy-balance residual | `<= 0.01%` |

### VAL-003 — DCM, BCM, and ideal-diode events

Purpose: verify unilateral constraints, zero-current event location, topology changes,
and the absence of artificial reverse current.

Sweep the buck and boost load through CCM, the boundary-conduction point, and DCM. Add
an ideal diode turn-off case whose analytical current crosses zero. The reference is a
piecewise analytical mode solution with an independent bracketing/refinement procedure
for the zero-current event.

Blocking gates:

| Quantity | Gate |
|---|---:|
| Zero-current or mode-transition time | `<= 0.05%` of `T_switch` |
| Maximum forbidden ideal-diode reverse current | `<= 1e-8 * I_base` |
| Cycle mean output and conduction interval | relative error `<= 0.1%` |
| Cycle energy-balance residual | `<= 0.01%` |

The load sweep must include points on both sides of the boundary; passing only deeply
CCM or deeply DCM cases is insufficient.

Current status: **partial, early-alpha verification evidence only; not a
validation claim**. Three separate implemented slices recognize the exact canonical
single-switch ideal buck, boost, and inverting buck-boost. Each sweeps declared load
ratios through CCM, its independently found numerical BCM point, and DCM. The default
retained-run sweeps use
`R/R_BCM = [0.25, 0.5, 0.9, 0.99, 1, 1.01, 1.1, 2, 200]`. Each oracle derives the
ON/freewheel/idle affine equations locally, propagates them with matrix exponentials,
and independently solves periodic states and zero-current events without using the
PEARSim circuit compiler, transient solver, event manager, or mode classifier.

Each of the nine default points records 11 blocking comparison gates, for 99
point-gate results in a complete default execution:

1. expected versus observed regime;
2. nonzero switching-boundary state agreement;
3. zero switching-boundary state agreement;
4. natural-event time;
5. forbidden reverse current;
6. cycle-mean output voltage;
7. ON/freewheel/idle/unclassified mode fractions;
8. full-cycle energy balance;
9. **cycle periodicity** — normalized mismatch between the observed cycle-end and
   cycle-start inductor current and output voltage;
10. **unclassified fraction** — time not assigned to a recognized canonical mode;
11. **mode-fraction sum** — absolute closure error of all recorded mode fractions
    relative to one.

Every point is requested from the independently calculated periodic state for one full
switching period. Completed points retain their model, waveform, events, metrics,
physics audit, mode coverage, point result, and hashes. A point that raises before its
run bundle exists still retains `benchmark-point.json`, the error, and eleven
`not-run` gates; it cannot disappear from the aggregate. The aggregate also retains
the human-readable offline sweep report.

The canonical boost is a distinct topology contract and does not inherit a pass from
the buck. In addition to the eleven gate classes above, each boost point checks state
ripple and scheduled/natural event semantics, for 13 gates per point and 117 gates in
the complete default sweep. Its directed topology, affine ON/freewheel/idle equations,
artifact contract, and claim boundary are documented in
[`m2-boost-boundary-evidence.md`](m2-boost-boundary-evidence.md).

The canonical inverting buck-boost is a third, exact topology contract. It uses the
oracle coordinates `x=[iL,u]`, with `u=v(gnd)-v(out)=-v(out)`, and separately checks
the negative output polarity. Every point records 14 blocking comparison/invariant
gates: the boost producer's 13 classes with a sign-preserving cycle-mean definition,
plus an explicit signed-output-polarity gate, for 126 gates in the complete default
sweep. This count must not be interpreted as 14 independent-oracle quantities: the
full-cycle energy-balance gate is a separate physics invariant. The exact topology,
artifact contract, and claim boundary are defined by the oracle module
`src/pear_sim/benchmarks/inverting_buck_boost_boundary.py`.
Its retained request is additionally bounded to `0.25 <= R/R_BCM <= 200`, exact BCM,
at least one point on each side, finite source/solver work caps, and derived
regime-separation margins greater than `1e-8` for normalized CCM minimum current and
DCM idle fraction. A numerically indistinguishable near-BCM request is rejected
before any evidence destination is created.

Output-grid independence and selected solver settings are covered by regression
smokes. PEARSim additionally implements a bounded producer for the complete declared
canonical-buck numerical profile and a native Linux/Darwin/Windows aggregate. This
does not make every revision a pass: the status is carried only by a concrete set of
source-bound retained artifacts. The sweep contract is documented in
[`m2-agent-evidence.md`](m2-agent-evidence.md), and the full numerical/platform
contract in [`m2-numerical-evidence.md`](m2-numerical-evidence.md).

This evidence does not complete VAL-003. It does not yet cover the standalone
analytical diode turn-off case or a general reduction for arbitrary floating-inductor,
multi-diode, and unrecognized converter networks. The boost and inverting-buck-boost
slices also lack the canonical buck's full numerical/native-platform matrix. None of this evidence
establishes compatibility with any third-party simulator or physical validation against
hardware; no independent-engine comparison exists for these slices.

### VAL-004 — single- and three-phase rectifiers

Purpose: verify periodic sources, diode selection, commutation overlap, pulse count,
average output, ripple, and harmonic extraction.

Cases cover a single-phase full bridge and a three-phase six-pulse bridge with both
zero and finite per-phase source inductance. Under the standard continuous-current
assumptions:

```text
Ls = 0:      Vdc = 3 * V_LL_peak / pi
finite Ls:   Vdc = (3/pi) * (V_LL_peak - omega * Ls * I_dc_avg)
```

Blocking gates:

| Quantity | Gate |
|---|---:|
| Mean DC output voltage | relative error `<= 0.1%` |
| RMS ripple | relative error `<= 0.5%` |
| Commutation start/end | `<= 0.1%` of electrical period |
| Dominant harmonic amplitudes | relative error `<= 0.5%` |
| Pulse count in each steady electrical period | exact |

All FFT comparisons use identical integer-period windows, sample treatment, harmonic
bandwidth, and DC-removal policy.

### VAL-005 — two-level voltage-source inverter

Purpose: verify SPWM/SVPWM scheduling, complementary switching, dead time, balanced
three-phase behavior, fundamental amplitude, and harmonics.

Use balanced R and RL loads, a modulation-index sweep within the linear range, and a
dead-time sweep including zero and a deliberately exaggerated value. An independently
generated switching schedule supplies transition times and expected pole states.
Fourier coefficients are calculated over an integer number of fundamental periods.

Blocking gates:

| Quantity | Gate |
|---|---:|
| Gate and pole transition time | `<= 0.05%` of `T_switch` |
| Fundamental voltage amplitude | relative error `<= 0.1%` |
| Line-voltage and phase-current THD | absolute difference `<= 0.01` |
| Balanced-case phase imbalance | `<= 1e-5` of base amplitude |
| Steady-period energy-balance residual | `<= 0.01%` |

The harmonic bandwidth must be reported with each THD value. A result using a different
bandwidth is incomparable, not merely less accurate.

Current status: **partial verification evidence only**. The complementary-PWM contract supplies an independently
checked complementary schedule, atomic zero-dead-time swaps, intrinsic interlock, and
a synchronous-buck dead-time fixture. The SPWM contract adds a bounded synchronous
offline-midpoint-sampled, center-aligned SPWM source with explicit `abc`/`acb`
sequence, nonzero dead time, commanded-cell and realized-event artifacts, and
event-aligned dense-output Fourier/THD observations. Its zero-dead-time 400 V,
`m=0.8`, carrier-ratio-42 fixture meets the analytical line-fundamental gate above;
synthetic signals independently check the Fourier implementation. The executable
fixture is still limited to one balanced RL case with the load neutral tied to the
DC midpoint (the circuit reference node is `dc_minus`). It does not
implement SVPWM, the modulation-index/load sweep, the retained inverter dead-time
sweep, an independently sourced THD golden, or the complete balanced R/RL load matrix
above. It must not be presented as validated three-phase inverter support.

### VAL-006 — buck small-signal response

Purpose: verify perturbation injection, frequency-response extraction, phase unwrap,
and low-frequency agreement between switching and averaged models.

Test control-to-output and input-to-output transfer functions of an ideal CCM buck.
The primary oracle is an independently evaluated averaged state-space transfer
function. Sweep from two decades below the LC pole through the control bandwidth.
Switching-versus-average comparison is limited to `f <= 0.1 * f_switch`; near the
switching frequency the averaged model is no longer an appropriate oracle.

Gates:

| Comparison | Magnitude | Phase |
|---|---:|---:|
| PEARSim averaged model vs analytical model | `<= 0.1 dB` | `<= 0.5 deg` |
| PEARSim switching model vs analytical average, below `0.1*f_switch` | `<= 0.5 dB` | `<= 2 deg` |

The analytical branch is blocking.

### VAL-007 — semiconductor conduction and DPT switching losses

Purpose: verify multidimensional lookup/interpolation and assess Eon, Eoff, and Erec
under matched double-pulse-test conditions.

First, synthetic surfaces with exact knots verify the table engine independently of
device data. Then curated points from the Infineon
[FS380R12A6T4B datasheet](https://www.infineon.com/assets/row/public/documents/10/49/infineon-fs380r12a6t4b-datasheet-en.pdf)
exercise temperature/current/gate-resistance interpolation. The ROHM
[SiC application note](https://fscdn.rohm.com/en/products/databook/applinote/discrete/sic/common/sic_appli-e.pdf)
provides a second DPT target for BSM120D12P2C005 at `Vds = 600 V`,
`Vgs = 18/0 V`, `Rg = 3.9 ohm`, module parasitic approximately `25 nH`, and
circuit parasitic approximately `15 nH`. Energy definitions and integration windows
must follow the referenced
[ROHM switching-loss note](https://fscdn.rohm.com/en/products/databook/applinote/common/pd_calc_power_dissipation_switching_cir_an-e.pdf).

Provisional gates:

| Quantity | Gate |
|---|---:|
| Exact synthetic LUT knots | relative error `<= 1e-12` |
| Conduction curve at matched current and temperature | `<= 5%` |
| Eon, Eoff, Erec at fully matched DPT conditions | `<= 10%` |

The physical gates are deliberately labeled provisional. Datasheet plots normally show
typical behavior, not raw traceable measurements or guaranteed production limits.
Changing any of `Vdc`, device current, junction temperature, gate voltages, gate
resistance, commutation device, power-loop/gate parasitics, or integration window makes
a DPT result a different experiment. The relevant IEC landing pages are
[IEC 60747-9 for IGBTs](https://webstore.iec.ch/en/publication/32214) and
[IEC 60747-8 for FETs](https://webstore.iec.ch/en/publication/69348); their paywalled
texts are not redistributed by this project.

### VAL-008 — Foster thermal impedance

Purpose: verify thermal state propagation, step response, asymptote, superposition,
and convolution for arbitrary power histories.

The Infineon FS380 datasheet publishes these Foster coefficients:

```text
IGBT:  r   = [0.007, 0.038, 0.050, 0.020] K/W
Diode: r   = [0.015, 0.070, 0.055, 0.020] K/W
Both:  tau = [0.001, 0.030, 0.250, 1.500] s

Zth(t) = sum_i r_i * (1 - exp(-t/tau_i))
```

Independent IGBT checkpoints are:

| `t` (s) | `Zth` (K/W) |
|---:|---:|
| 0.001 | 0.005883561516160484 |
| 0.01 | 0.019864910315970822 |
| 0.1 | 0.061418246250390086 |
| 1 | 0.10381587567491132 |
| 10 | 0.1149745473239732 |

Blocking gates:

| Quantity | Gate |
|---|---:|
| `Zth` at declared times | relative error `<= 0.1%` |
| Asymptotic `Rth = sum(r_i)` | relative error `<= 0.1%` |
| Arbitrary power-profile temperature response | relative error `<= 0.2%` |

The [Infineon thermal-model note](https://www.infineon.com/dgdl/Infineon-Thermal_equivalent_circuit_models-ApplicationNotes-v01_01-EN.pdf?fileId=db3a30431a5c32f2011aa65358394dd2)
is essential for interpretation: Foster internal nodes do not represent physical
package layers, and terminal-equivalent Foster and Cauer networks need not have the
same internal temperatures.

### VAL-009 — end-to-end electrothermal converter

Purpose: exercise switching, semiconductor loss tables, thermal propagation, control,
initialization, energy accounting, and reporting in one integrated case.

#### Held-out experimental validation

The candidate physical dataset is the CC BY 4.0
[electrothermal DC/DC dataset on Mendeley Data](https://data.mendeley.com/datasets/fh9x8459h8/1),
with experimental context in the
[associated Applied Energy article](https://doi.org/10.1016/j.apenergy.2019.02.071).
It remains link-only in this plan and was not downloaded.

Before activation, reviewers must freeze calibration and held-out subsets, record
artifact hashes and license terms, reproduce the experimental boundary conditions, and
document sensor accuracy and uncertainty. Initial gates are an absolute efficiency
error no greater than one percentage point, steady temperature error `<= 5 K`, and
temperature-transient NRMSE `<= 5%`. These values are placeholders until measurement
uncertainty supports a more defensible acceptance rule.

## 4. CI and release gates

| Tier | Cases | External network | Expected purpose |
|---|---|---|---|
| Pull request | VAL-001–005, VAL-008 | prohibited | fast deterministic regression against analytical oracles |
| Nightly | pull-request set plus VAL-006, VAL-007 | prohibited | frequency sweeps and approved local reference artifacts |
| Manual | VAL-009 | prohibited during run | reviewed physical-validation campaign |

Pull-request jobs must be deterministic and self-contained. Nightly jobs may read an
approved local artifact store, but they may not silently update it. A missing artifact
is `not-run`; it is never converted to a zero metric or a pass.

Initially, VAL-001–005 and VAL-008 are blocking. The analytical branch of VAL-006
becomes blocking when implemented. VAL-007 physical gates become blocking
only after their digitized/reference artifacts and extraction procedures have been
reviewed. VAL-009 cannot become a release claim until the experimental protocol,
held-out split, and uncertainty treatment are frozen.

Recommended promotion sequence:

1. land exact/analytical cases and make the pull-request tier mandatory;
2. add convergence studies by repeating representative cases at tighter solver and
   event tolerances;
3. curate manufacturer and third-party simulator artifacts under a separate provenance
   registry;
4. run cross-simulator diagnostics without replacing analytical oracles;
5. activate the held-out experimental campaign and publish its uncertainty budget;
6. freeze a release evidence bundle with hashes for code, manifest, inputs, references,
   and results.

## 5. Reference-data governance

No source linked in this plan is copied into the repository by the documentation task.
When a later implementation acquires an artifact, its registry record must contain:

- canonical source URL, publisher, title, revision, and retrieval date;
- license or access terms and whether redistribution is permitted;
- byte-level SHA-256 of the original and every derived table;
- page/figure/table and the digitization method, operator, and tool version;
- units, coordinate transformation, interpolation rule, and estimated digitization
  uncertainty;
- full electrical, gate-drive, parasitic, thermal, and measurement conditions;
- whether the artifact was used for calibration, verification, compatibility, or
  held-out validation.

Reference updates require review. A newer datasheet or a re-digitized curve is a new
artifact version and must not silently overwrite historical release evidence.

## 6. Known limitations and future extensions

- Ideal-switch verification does not establish numerical robustness for stiff,
  continuous semiconductor models or detailed reverse recovery.
- Manufacturer curves are sparse and usually typical. Their uncertainty can dominate
  a result even when the solver error is negligible.
- Published third-party simulation scalar outputs do not reveal all solver settings or
  proprietary model details; they support compatibility, not a unique implementation.
- Electrothermal agreement can be non-identifiable: several combinations of electrical
  loss and thermal impedance may yield a similar junction temperature. Electrical and
  thermal submodels therefore need their own tests as well as VAL-009.
- Long-duration thermal and controller-in-the-loop behavior need performance and
  real-time benchmarks separate from numerical accuracy.
- Future validation may add the CEC inverter matrix of six load levels
  (`10, 20, 30, 50, 75, 100%`) at three DC voltages, as summarized by
  [Sandia PVPMC](https://pvpmc.sandia.gov/modeling-guide/dc-to-ac-conversion/cec-inverter-test-protocol/).
- Convergence is evidence, not an oracle. At least one representative case per model
  family should demonstrate stable metrics as time-step and event tolerances tighten.

## 7. Review checklist

Before calling a release “validated,” answer all of the following:

- Are verification, compatibility, and physical-validation results labeled separately?
- Did every blocking case run, or is any missing artifact incorrectly shown as pass?
- Are all comparisons made at identical circuit, drive, parasitic, thermal, and
  measurement conditions?
- Are waveform event times checked independently from event-aligned NRMSE?
- Are the FFT window, bandwidth, phase convention, and initialization recorded?
- Are calibration and held-out datasets disjoint and immutable for this revision?
- Are source and derived artifacts hash-pinned with documented uncertainty?
- Are solver convergence and energy balance satisfactory in addition to output fit?
- Is the final claim no broader than the topologies, regimes, devices, and boundary
  conditions actually tested?

If any answer is no, the result is useful development evidence but not a complete
release-validation claim.
