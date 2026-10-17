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
