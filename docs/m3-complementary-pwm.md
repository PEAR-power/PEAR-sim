# Complementary PWM and dead time

Status: experimental in PEARSim 0.3.0.dev0. This is a control-schedule contract,
not a claim that a converter, inverter, or hardware design has been validated.

## Source contract

A complementary block has one commanded duty ratio and two deterministic gate IDs:

```yaml
controls:
  leg:
    type: complementary_pwm
    frequency: 20 kHz
    duty: 40 %
    phase: 0 s
    dead_time: 500 ns
```

The generated outputs are `leg.high` and `leg.low`. An `ideal_switch.control` refers
to one of those IDs. The names are part of the resolved physics contract and cannot
be remapped implicitly.

Let `T` be the period, `D` the commanded high-side duty, `phi` the phase, and
`delta` the dead time. For `0 < D < 1` and integer cycle `k`, the right-continuous
gate states are:

```text
high = on  for [phi + k*T + delta, phi + k*T + D*T)
low  = on  for [phi + k*T + D*T + delta, phi + (k+1)*T)
```

The outgoing output turns off at the command boundary. The incoming output turns on
after `delta`. Therefore both outputs are off during two blanking intervals per
cycle and they are never both on. At zero dead time, the two updates at a command
boundary are one atomic event group: serialization is off-before-on, while the
circuit topology is compiled only after the complete group is applied.

Duty endpoints are constant (`D=0`: low only; `D=1`: high only). For a switching
pair, dead time must be non-negative and strictly smaller than both commanded pulse
widths. Unknown fields, unit errors, generated-ID collisions, deleted pulses, and a
dead time that cannot be represented at the requested time scale fail before the
simulation.

## Human and machine evidence

Every run records raw semantic transitions in `events.jsonl`. A run containing gate
controls additionally records `gate-drive.json`, including:

- control-to-output and output-to-device mappings;
- declared command parameters and intrinsic interlock policy;
- initial and final gate states;
- exact-time atomic transition groups and resulting states;
- per-control retained/full-run transaction counts, selection and truncation metadata,
  with `events.jsonl` explicitly retained as the authoritative full-run record;
- observed interlock violations.

The offline report copies this producer artifact; it does not infer gate mappings or
recompute an acceptance verdict. The schematic labels a controlled switch with its
gate ID, not the ignored source `initial` drawing state.

## Evidence boundary

The independent schedule oracle checks exact binary-representable transition times,
phase wrapping, zero dead time, endpoint duties, invalid pulse widths, solver/output
grid independence, and the no-overlap invariant. The synchronous-buck example is an
executable integration fixture. This contract by itself does not implement SPWM, SVPWM, THD,
semiconductor switching loss, temperature, EMI, or the full VAL-005 inverter matrix;
the later narrow SPWM/harmonic slice is documented separately in
[three-phase SPWM](m3-spwm.md).
