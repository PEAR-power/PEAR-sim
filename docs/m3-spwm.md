# Three-phase SPWM and harmonic observation

Status: experimental in PEARSim 0.4.0.dev0. This is a narrow, reviewable vertical
slice for a two-level inverter. It is not the complete VAL-005 matrix, a
third-party-simulator compatibility claim, physical gate-driver timing, or hardware
sign-off.

## Source contract

One logical source expands to six explicit gate IDs:

```yaml
controls:
  bridge:
    type: three_phase_spwm
    carrier_frequency: 20 kHz
    fundamental_frequency: 500 Hz
    modulation_index: 80 %
    carrier_phase: 0 s
    reference_phase: 0 s
    sequence: abc
    dead_time: 500 ns
```

The generated outputs are `bridge.a.high`, `bridge.a.low`, `bridge.b.high`,
`bridge.b.low`, `bridge.c.high`, and `bridge.c.low`. They are part of the resolved
physics contract; an `ideal_switch.control` must name one explicitly. The SPWM source
must be the model's only logical control in this contract.

The carrier ratio `N = carrier_frequency / fundamental_frequency` must be an integer
from 2 through 8333. The modulation index is limited to `0 <= m < 1`; overmodulation and
clamping are rejected. `sequence` is explicitly `abc` or `acb`. The dead time must
be non-negative and strictly shorter than the smallest possible command pulse,
`0.5 * (1 - m) / carrier_frequency`.

For carrier cell `k`, references are mathematically sampled once at the cell center.
This is an offline midpoint-sampled schedule: because the centered pulse begins before
that midpoint, it does not claim causal digital-controller timing. For `abc`:

```text
r_a = sin(theta_k)
r_b = sin(theta_k - 2*pi/3)
r_c = sin(theta_k + 2*pi/3)
D_x = 0.5 * (1 + m*r_x)
```

`acb` swaps the B and C references. Each high-side command pulse is centered in its
carrier cell. The outgoing gate turns off at the command edge and the incoming gate
turns on after the declared dead time. All changes at one exact timestamp are applied
atomically, with off-before-on serialization. Mathematically coincident balanced-phase
edges that differ only within the scheduler's ULP envelope are canonicalized inside
the same transition role; command, dead-time, opposite-pulse, and repeated-gate edges
are never merged by that rule. The initial state uses the same right-continuous
absolute-cell construction as the future schedule.

## Explicit harmonic request

Harmonic analysis is opt-in and bound to named outputs and engineering semantics:

```yaml
analysis:
  harmonics:
    start: 4 ms
    stop: 8 ms
    fundamental_frequency: 500 Hz
    max_harmonic: 40
    signals:
      - {signal: v(vab), semantic: line_voltage_ab}
      - {signal: i(la), semantic: phase_current_a}
```

The half-open window `[start, stop)` must lie inside the transient and contain an
integer number of fundamental periods. Harmonic order is bounded to `2..256`; each
signal and semantic may occur once and the signal must also be requested in `outputs`.
Voltage/current expression families must match their semantics. Portable panel and
order-by-signal work budgets are checked before simulation and again while integrating.
These settings are part of the request hash. Models without the block retain the
legacy request semantics and hashes.

The producer integrates complex Fourier coefficients from solver dense output on
event-aligned topology intervals. Eighth-order Gauss-Legendre panels are subdivided
to bound the phase advance at the highest requested harmonic. Isolated switching
boundary values therefore have zero measure, and the result is independent of the
CSV output grid. `harmonics.json` records the exact window, endpoint convention,
fundamental, maximum order, bandwidth, window function, DC/THD policy, coefficient
convention, amplitudes, phases, and all coefficients. DC is retained as coefficient
zero and excluded from THD. THD is a bandwidth-limited observation, not an
independent-reference acceptance verdict.

## Human and machine evidence

An SPWM run keeps commanded and realized switching evidence separate:

- `modulation-cycles.jsonl` is the authoritative per-carrier-cell command record;
- `events.jsonl` is the authoritative realized gate and natural-commutation log;
- `three-phase-modulation.json` binds the six gate IDs to devices and phases and
  provides bounded command/fundamental and worst-pulse projections for review;
- `harmonics.json` contains producer-computed Fourier observations;
- the schematic and offline report display the explicit mapping, sequence, carrier
  ratio, modulation index, dead time, harmonic window, bandwidth, and source files.

The HTML report validates and copies these artifacts. It does not infer a device
mapping from names, recompute Fourier values, or turn an observation into a validation
claim. All artifacts are bounded, schema-checked, hash-bound by the manifest, escaped,
and rendered offline without scripts or network resources.

## Executable fixture and evidence boundary

`examples/inverter_three_phase_spwm.yaml` is a split 400 V DC bus, six ideal switches,
six explicit antiparallel diodes, and a balanced `10 ohm / 5 mH` RL load whose neutral
is tied to the DC midpoint; the circuit reference node is `dc_minus`. Its
21 kHz / 500 Hz carrier ratio is 42, divisible by three for an exactly cyclic
sampled phase set. It exercises nonzero dead time, explicit antiparallel-diode active
sets, line and phase signals, energy balance, `abc` sequence, and a 42nd-order harmonic
observation whose bandwidth includes the carrier but excludes upper switching
sidebands above 21 kHz. It is deliberately reported as bandwidth-limited THD.

The regression suite includes an independent offline-midpoint schedule oracle, an
analytic zero-dead-time line-voltage fundamental check in the linear range, synthetic
Fourier signals with known harmonics, phase-sequence checks, output-grid independence,
intrinsic no-overlap checks, artifact tamper rejection, and all earlier converter
tests. This is partial VAL-005 evidence only. SVPWM, overmodulation, arbitrary mixed
controls, a floating-neutral load, R-only and broader RL matrices, retained modulation
and dead-time sweeps, an independently sourced THD golden, closed-loop control,
semiconductor losses, and physical validation remain outside this contract.

## Dead-time acceptance matrix (midpoint-tied neutral)

`tests/test_spwm_dead_time_golden.py` runs the shipped fixture over 24 points —
plain SPWM and min-max injection, dead time 0 / 0.5 / 1 us, a damped
`10 ohm / 5 mH` load and a ripple-dominated `40 ohm / 1 mH` load, m = 0.5 and 0.9 — and
compares each run with `pear_sim.benchmarks.spwm_dead_time`, an independent oracle
that never imports the solver. With the star on the DC midpoint the three legs decouple,
so each one is a first-order RL circuit with a piecewise-constant pole voltage, solved
interval by interval in closed form.

The oracle does not use the textbook dead-time formula. In each gap the pole is pinned
by the diode that the current sign selects. When the current reaches zero inside the gap
(at `s = tau*ln(1 + |i0|*R/Vh)`), that diode is starved and the current stays at
zero with the pole at the load voltage for the rest of the gap. The zero-crossing
region, where a sign-only model of the error breaks down, is therefore covered instead
of cut out. A tripwire requires the matrix to clamp leg a inside the harmonic window
(9 of the 16 nonzero-dead-time points do).

The comparison covers the three phase currents on the whole output grid, the complex
Fourier coefficients 0..42 of `v_ab` and `v_an` (`v_an` also carries the triplen
content that cancels in the line voltage), and the `v_ab` THD under the producer's
bandwidth and DC policy. The THD reference is integrated analytically from the oracle
waveform, so it is an independent reference value, not a second estimate from the same
samples. Budgets: 1e-9 for the spectra and THD, and 100 x the declared `rtol` for the
currents, because they are integrated (measured 2026-11-16: 4.3e-12 and 2.0e-9). Three
wrong oracles each miss the budget by at least four orders of magnitude: no dead time,
the pole on the wrong rail, and no zero-current clamp.

On the same oracle, the textbook loss `|dV1| = (4/pi)*Td*fc*Vdc` holds within 2 % on
the damped load. On the ripple-dominated load the real loss is 0.47-0.90 of the formula,
because parts of the gaps sit at 0 V instead of on a rail.

The dead time admitted by the scheduler is the tighter of the two documented bounds.
The low switch must turn on inside the same carrier cell, so `dead_time` must stay below
`0.25*(1 - m*reference_peak)/carrier_frequency`. A larger value passes the loader and
is refused when the run starts (tracked separately).

This is verification against an independent solution of the same ideal circuit. The
floating-neutral star, the sector bridge driven with dead time, unbalanced loads,
overmodulation and hardware remain outside it.
