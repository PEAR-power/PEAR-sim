# PE-Sim model format `pear-sim/v0.1`

Status: early-alpha contract
Encoding: UTF-8
Input: YAML 1.2 or JSON
Normative schema: [`../schema/model.schema.json`](../schema/model.schema.json)

This document describes the actual model contract used by the reference Python kernel. Extensions planned in the architecture — SPICE resources, separately declared probes and multiple analyses per file — are not part of `v0.1` and require a new schema version. (Hierarchy and table-based loss/thermal data, once on that list, are now part of `v0.1`: see §6a and §7a.)

## 1. Minimal document

```yaml
schema: pear-sim/v0.1
name: rc_step

circuit:
  ground: gnd
  components:
    - {id: src, type: voltage_source, p: in, n: gnd, value: 12 V}
    - {id: r1, type: resistor, p: in, n: out, value: 10 ohm}
    - id: c1
      type: capacitor
      p: out
      n: gnd
      value: 100 uF
      initial_voltage: 0 V

analysis:
  transient:
    start: 0 s
    stop: 10 ms
    output_step: 10 us
    solver: dopri5
    rtol: 1.0e-8
    atol: 1.0e-10

outputs:
  - v(out)
  - i(c1)
```

## 2. Complete buck example

```yaml
schema: pear-sim/v0.1
name: buck_ccm

parameters:
  vin: 100 V
  fsw: 20 kHz
  duty: 40 %
  load: 10 ohm

circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: "${vin}"}
    - {id: q1, type: ideal_switch, p: in, n: sw, control: pwm, initial: on}
    - {id: d1, type: diode, p: gnd, n: sw, initial: off}
    - {id: l1, type: inductor, p: sw, n: out, value: 1 mH, initial_current: 4 A}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, initial_voltage: 40 V}
    - {id: load, type: resistor, p: out, n: gnd, value: "${load}"}

controls:
  pwm:
    type: pwm
    frequency: "${fsw}"
    duty: "${duty}"
    phase: 0 s

analysis:
  transient:
    start: 0 s
    stop: 2 ms
    output_step: 1 us
    solver: dopri5
    rtol: 1.0e-8
    atol: 1.0e-10

outputs:
  - v(out)
  - i(l1)
  - v(q1)
  - i(q1)
  - i(d1)
  - i(vin)
  - i(load)
```

The version of this example used by the repository is in [`../examples/buck.yaml`](../examples/buck.yaml).

## 3. Top-level fields

| Field | Type | Required | Meaning |
|---|---|---:|---|
| `schema` | string | yes | Must be exactly `pear-sim/v0.1`. |
| `name` | string | no | Model name; defaults to the file name without extension. |
| `parameters` | object | no | Named compile-time values (scalar or envelope form, §5.3). |
| `circuit` | object | yes | Ground reference, component list and optional `couplings`. |
| `subcircuits` | object | no | Named reusable subcircuit definitions; see §6a. |
| `controls` | object | no | Named gate sources: PWM, complementary PWM, three-phase SPWM, NPC SPWM, closed-loop PI / dq controllers and signal-driven modulators (§7, §9.3, §9.6). |
| `machines` | object | no | Named electrical machines (currently `type: pmsm`, at most 4), expanded into audited windings; see §9.1, §9.2, §9.4, §9.8. |
| `signals` | object | no | Declared discrete control blocks executed by one periodic task; see §9.3. |
| `device_cards` | object | no | Device loss/thermal cards; see §7a. |
| `cooling` | object | no | TIM, heatsink, fan and cold-plate data; see §7a. |
| `loss_analysis` | object | no | Post-processing loss and junction-temperature estimate; see §7a. |
| `perturbations` | object | no | Run-level overrides (`forced_gate_windows`). Normally generated from a perturbation request file rather than hand-written; a hand-written one (e.g. a stuck gate) is legal and validated. |
| `analysis` | object | yes | Transient configuration and an optional explicit harmonic window. |
| `outputs` | array[string] | yes | List of requested voltages, currents and other signals. |

Unknown fields are forbidden by the JSON Schema. The loader **enforces** the normative schema on every load (Draft 2020-12 structural validation before semantic validation); a copy of the schema ships inside the package (`pear_sim/spec/model.schema.json`) and a test guards its byte-for-byte equality with `schema/model.schema.json`. Rules the schema cannot express (units, parameter references, control expansion, topology) remain in the loader.

## 4. Identifiers and nodes

Identifiers of parameters used by `${...}` match:

```text
[A-Za-z_][A-Za-z0-9_]*
```

The model name, parameter names, component and control identifiers, node names
and `outputs` expressions are limited to 256 characters. `outputs` must be a non-empty
list of unique expressions; a duplicate is an error because it would produce ambiguous
samples and integrals. Output expressions accept identifiers that start with a letter
or `_`, followed by letters, digits, `_`, `.`, `:`, `-`. The model source file is
limited to 2 MiB.

In `v0.1` nodes are not declared separately. They are derived from the `p` and `n` fields. The node named by `circuit.ground` is at `0 V` and is not an MNA (Modified Nodal Analysis) unknown.

Every component:

- must have a non-empty, unique `id`;
- must have non-empty `p` and `n`;
- must not have `p == n`;
- defines positive voltage as `v = v(p) - v(n)`;
- defines positive current as flowing from `p` to `n`.

## 5. Values and units

A value can be:

- a unitless number, interpreted in SI units of the expected dimension;
- a string `number + unit`, e.g. `100 uF`;
- an exact parameter reference, e.g. `"${vin}"`.

`v0.1` has no general expression language. Constructs such as `${1/fsw}`, Python code and mathematical functions are errors. A referenced value must have the dimension expected by the target field.

### 5.1. Supported units

The parser is case-insensitive except for the special distinction between `MOhm` and `mOhm`; the recommended spelling is shown below.

| Dimension | Input units | SI |
|---|---|---|
| dimensionless | none, `1`, `%` | `1` |
| time | `s`, `ms`, `us`, `ns` | `s` |
| frequency | `Hz`, `kHz`, `MHz` | `Hz` |
| voltage | `V`, `mV`, `kV` | `V` |
| current | `A`, `mA`, `kA` | `A` |
| resistance | `ohm`, `kohm`, `MOhm`, `mOhm`, `Ω` | `ohm` |
| inductance | `H`, `mH`, `uH`, `nH` | `H` |
| capacitance | `F`, `mF`, `uF`, `nF`, `pF` | `F` |
| power | `W`, `kW` | `W` |
| energy | `J`, `mJ` | `J` |

The characters `µ` and `μ` are aliases of `u`.

### 5.2. Parameters and overrides

```yaml
parameters:
  vin: 100 V
  duty: 40 %
```

A parameter can only be used as the whole string:

```yaml
value: "${vin}"
```

The CLI can override an existing parameter:

```text
pear-sim simulate buck.yaml --set vin="120 V" --set duty="45 %" --out run/
```

`--set` cannot add a new parameter. An override must keep the dimension of the original value and is applied before components and analysis are validated.

### 5.3. Parameter drift envelope (object form)

Instead of a scalar, a parameter may declare an **envelope** — the same nominal value
plus an explicit statement of how far it may drift and where that number comes from:

```yaml
parameters:
  vin: 100 V                       # scalar form — unchanged
  esr:
    nominal: 20 mohm
    source: assumption             # datasheet | test_report | assumption | standard
    note: "Placeholder value; does not describe any real component."
    corners: {eol_lo: 18 mohm, eol_hi: 60 mohm}
    distribution: {type: normal, sigma_percent: 10, seed: 20260727}
    life: {type: geometric, end_of_life_ratio: 3.0}
```

Rules, all fail-closed:

- **No envelope = no drift.** The loader never guesses a tolerance; absence
  means "not stated", never "assume ±x%".
- **`source` and `note` are mandatory** when you declare an envelope. Drift taken
  from a qualification report and guessed drift count the same — and must not
  look the same in a report.
- An envelope must declare **at least one** of `corners` / `distribution` /
  `life`; provenance alone without drift is rejected (it is a scalar in disguise).
- Corner names: `[A-Za-z_][A-Za-z0-9_]*`, at most 16; `nom` and `nominal` are
  reserved — the nominal point always exists and is called `nom`.
- `distribution` **requires an explicit `seed`** (types: `uniform` with `min`/`max`,
  `normal` with `sigma` or `sigma_percent`, `triangular` with `min`/`max`/`mode`).
  The normal distribution is truncated at ±3σ so that a sample cannot produce a
  negative resistance; the truncation is recorded in the artifact, not hidden.
- `life` describes `p = f(p_bol, life_fraction)` on `life_fraction ∈ [0, 1]`:
  `linear` with `end_of_life_percent` or `geometric` with `end_of_life_ratio`
  (the latter suits quantities that grow by decades, e.g. leakage current).

The envelope is a **declaration about the parameter, not a change to it**.
`Model.parameters` still holds the scalar, the solver still sees one number,
and `model_sha256` (computed over `physics_model_dict`) does not see the
envelope — a model without an envelope hashes exactly as it did before this
section existed. Envelopes are written to `model.resolved.json` **only when declared**.

## 6. Circuit

```yaml
circuit:
  ground: gnd
  components:
    - {id: r1, type: resistor, p: in, n: out, value: 10 ohm}
```

`ground` is optional and defaults to `gnd`. `components` is a non-empty list.

### 6.1. v0.1 component catalog

| `type` | Required fields besides `id`, `p`, `n` | Optional fields | Equation / semantics |
|---|---|---|---|
| `resistor` | `value` | — | `v = R i`, `R > 0`. |
| `capacitor` | `value` | `initial_voltage`, `esr` | `i = C dv/dt`, `C > 0`; the state is `v(p)-v(n)` (with `esr`: the voltage on the ideal capacitance, see §6.1.0a). |
| `inductor` | `value` | `initial_current`, `dcr` | `v = L di/dt`, `L > 0`; the state is positive from `p` to `n`. |
| `sat_inductor` | `value`, `l_sat`, `i_sat` | `initial_current`, `dcr` | Saturating inductor: incremental inductance `value` for `|i| <= i_sat`, `l_sat` above it (`0 < l_sat < value`); continuous flux, two-segment piecewise-linear characteristic. A segment crossing is an exact restart event (`saturation_boundary`). Stored energy is computed per segment (not `0.5*L*i^2`). v0.1 limit: one `sat_inductor` per model; no zero-current (DCM) reduction for this element. |
| `voltage_source` | `value` XOR `waveform` | — | Enforces `v(p)-v(n) = value` or `u(t)` from the waveform. |
| `current_source` | `value` XOR `waveform` | — | Enforces the current `value` (or `u(t)`) from `p` to `n`. |
| `ideal_switch` | `control` | `initial`, `on_resistance` | Closed: `v = R_on·i` (`v=0` when ideal); open: `i=0`. |
| `diode` | — | `initial`, `forward_voltage`, `on_resistance` | Conducting: `v = V_f + R_on·i, i>=0`; blocking: `i=0, v<=V_f`. |
| `thyristor` | — | `control`, `initial` | Latch. It closes ONLY when the gate is asserted and the element is forward-biased; it opens ONLY at zero current. Removing the gate while conducting does not turn the element off — that is the whole difference from `ideal_switch`. Without a gate (`control` omitted) it never fires: a model of a permanently blocking element. A gate held high gives exactly `ideal_switch` + `diode` in series (asserted in `tests/test_thyristor.py`). Ideal: no forward drop, no holding/latching current, no `dv/dt` firing, no turn-off recovery time (`t_q`) — so it **cannot answer whether a forced-commutation circuit gives the device enough time**. |
| `voltage_probe` | — | — | Does not load the circuit; `v(id)` returns `v(p)-v(n)`. |

`initial_voltage` defaults to `0 V`, and `initial_current` to `0 A`.

`initial` for switch, diode and thyristor accepts a boolean or one of the aliases:

- closed: `on`, `true`, `closed`, `conducting`;
- open: `off`, `false`, `open`, `blocking`.

The default `initial` is `off`.

### 6.1.0. Conduction drops (`on_resistance`, `forward_voltage`)

A closed `ideal_switch` or a conducting `diode` can carry a piecewise-linear
conduction model instead of a perfect short:

```yaml
- {id: q1, type: ideal_switch, p: in, n: sw, control: pwm, on_resistance: 8 milliohm}
- {id: d1, type: diode, p: gnd, n: sw, forward_voltage: 0.7 V, on_resistance: 20 milliohm}
```

- `on_resistance` (resistance, `>= 0`, default `0`): `ideal_switch` and `diode`.
- `forward_voltage` (voltage, `>= 0`, default `0`): `diode` only.

The drop is part of the circuit equations: it changes the waveforms, the operating
point and the damping, and the run's energy audit includes the dissipation. A blocking
diode turns on when its voltage reaches `forward_voltage`. A MOSFET is an
`ideal_switch` with `on_resistance` plus an antiparallel `diode` (its body diode) with
`forward_voltage`; above `V_f / R_on` of reverse current the body diode shares it, as
in hardware.

Limits (also in the `model.piecewise_linear_conduction` capability): the drop is
constant (no temperature or current dependence beyond `R_on`), and switching
transitions, reverse recovery and EMI are not modelled. A component with a drop cannot
also be bound to a loss card in `loss_analysis` — both would describe the same
conduction loss. Units are case-sensitive: `mohm` is milliohm, `Mohm` megaohm;
`milliohm` is the unambiguous spelling. Declaring `0` is the ideal device and keeps
the model hash unchanged.

### 6.1.0a. Series parasitics (`esr`, `dcr`)

A capacitor can declare its equivalent series resistance and an inductor its winding
resistance:

```yaml
- {id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, esr: 20 mohm}
- {id: c2, type: capacitor, p: out, n: gnd, value: 10 uF, esr: 3 mohm}
- {id: l1, type: inductor, p: sw, n: out, value: 1 mH, dcr: 45 mohm}
```

- `esr` (resistance, `> 0`): `capacitor` only.
- `dcr` (resistance, `> 0`): `inductor` and `sat_inductor`.

Each is shorthand for a resistor in series, added before validation:
`p --[c1.esr]-- c1.core --| c1 |-- n` and `p --( l1 )-- l1.core --[l1.dcr]-- n`. The
result is bit-identical to writing that resistor by hand. Consequences:

- terminal voltages such as `v(out)` include the drop across the ESR, as a probe on the
  board would; `v(c1.core)` is the voltage on the ideal capacitance;
- `initial_voltage` sets the ideal capacitance (the state), `initial_current` the
  inductor current;
- the dissipation appears in the energy audit as the resistors `c1.esr` / `l1.dcr`;
- the value may be a `${parameter}`, so envelopes, corners, `--set`, sweeps and
  sensitivity apply to it;
- the ESR also removes the capacitor loops of §6.2: two capacitors in parallel, or a
  capacitor across a voltage source, run once one capacitor in the loop has an `esr`.

`0` is rejected: omit the field for an ideal component. The value is constant (state
it at the frequency that matters, usually the switching frequency); there is no
frequency or temperature dependence, no leakage resistance and no ESL. A node named
`<id>.core` next to a component with a parasitic is rejected.

### 6.1.1. Time-varying sources (`waveform`)

`voltage_source` and `current_source` may declare a `waveform` object instead of a
constant `value` (exactly one of the two — never both). Level values have the
dimension of the source (V or A), times are in seconds, frequencies in Hz. Five types:

```yaml
waveform: {type: sine, amplitude: 5 V, frequency: 1 kHz}            # + optional offset, phase_deg
waveform: {type: step, initial: 0 V, final: 5 V, at: 2 ms}
waveform: {type: pulse, low: 0 V, high: 5 V, frequency: 1 kHz, duty: 25 %}  # + optional phase
waveform: {type: pwl, points: [[0 s, 0 V], [1 ms, 5 V], [3 ms, 5 V]]}
waveform: {type: double_exponential, amplitude: 1 A, tau_rise: 1 ns, tau_fall: 100 ns, at: 1 ms}
```

Semantics:

- `sine`: `offset + amplitude * sin(2*pi*frequency*t + phase_deg)`.
- `step`: right-continuous — at time `at` the value is already `final`; `at >= 0`.
- `pulse`: periodic rectangle, `high` for the first `duty` fraction of the period
  (`0 < duty < 1`, strictly); `phase` shifts the start of the period.
- `pwl`: piecewise-linear interpolation over strictly increasing times; outside
  the range of points the end value is held (limit 10,000 points).
- `double_exponential`: a single pulse shaped as
  `exp(-t/tau_fall) - exp(-t/tau_rise)` measured from `at`, scaled so that the
  extremum is exactly `amplitude`; before `at` the value is zero.
  `tau_fall > tau_rise` is required **strictly** — equal time constants make the
  shape vanish and reversed ones mirror it, and neither form is regularized
  into something that looks sensible. The sign of `amplitude` sets the
  polarity. **This is a declared input of the experimenter, not a quantity this
  simulator derives** — which amplitude and width are physically right for a
  given circuit follows from test data, not from simulation.

Waveform discontinuities and kinks (the `step` jump, `pulse` edges, `pwl` points)
are integration restart boundaries for the solver — an integration step never
smears a source discontinuity. DC models remain byte-identical in the contracts:
the `waveform` key appears in the canonical JSON only when it was declared.

Validity limits (fail closed): the periodic steady-state and small-signal
analyses (`pear-sim analyze stability`) require DC sources — a model with a
waveform ends with an explicit error, not a silent approximation. Canonical
topology matchers (buck/boost/…) require a constant `value` and report
`not_applicable` for time-varying sources.

### 6.1.2. Coupled inductors (`circuit.couplings`)

Two ordinary inductors can be coupled magnetically (two-winding transformer):

```yaml
circuit:
  components:
    - {id: lp, type: inductor, p: p1, n: gnd, value: 1 mH}
    - {id: ls, type: inductor, p: s1, n: gnd, value: 4 mH}
  couplings:
    - {inductors: [lp, ls], coupling: 0.95}   # k, or: mutual: 1.9 mH
```

Declare exactly one of `coupling` (dimensionless k, strictly `0 < k < 1`)
or `mutual` (M in henries, strictly `M² < L1·L2`). Ideal coupling (k=1)
is rejected as singular — leakage is declared explicitly; the kernel does not add
hidden parasitics. Stored energy includes the mutual term
`E = ½L1·i1² + ½L2·i2² + M·i1·i2` and balances in the physics audit.
v0.1 restrictions (fail closed): an inductor may belong to at most one coupling;
a `sat_inductor` cannot be coupled; a coupled winding needs a DC path in every
switch configuration (an open secondary sees a reflected voltage that the
zero-current reduction cannot express — add a leakage resistor).

### 6.1.3. Floating inductor nodes (three-wire star)

A node where only ordinary inductors meet (e.g. the star point of a three-wire
RL inverter load) has no algebraic support in MNA: its KCL row is `Σ ±i_k = 0` —
an exact linear **invariant** of the states, not an equation for the node voltage.
The kernel replaces the degenerate row with the derivative of the invariant,
`Σ ±(v_pk − v_nk)/L_k = 0`, which determines the star-point voltage; the invariant
itself is validated at start-up (declared `initial_current` values must sum to
zero — otherwise an explicit error, never a silent projection) and is preserved
EXACTLY by the integrator (linear conservation law of RK methods). This also
applies to the node between two series inductors.

Dead time on a floating star: a leg whose current is zero during the dead-time
window has no conduction path — the active set freezes that inductor (zero-current
reduction; candidate subsets are tried smallest first — with up to four
zero-current inductors every subset is enumerated, beyond four only single
inductors and the full set are offered), and the star reduces over the remaining
live legs. A frozen branch does not count as algebraic support of its node.

**Coupled windings.** A star of magnetically coupled windings reduces the same
way, except that the derivative of the invariant must carry the MUTUAL terms:
`Σ ±di_k/dt = 0`, where `di_k/dt` is the row of `inv(L_AA)` applied to the
terminal voltages of ALL live windings of the group (an ordinary inductor is the
1×1 case, i.e. `1/L`). A frozen coupled winding does NOT become a 0 V short —
an open transformer secondary shows its induced voltage, so its branch carries
`v_f = L_fA inv(L_AA) v_A`. Which windings to freeze follows from STRUCTURE (a
branch with no live companion at a terminal cannot carry current), computed as a
fixed point rather than guessed from state values — so a nine-winding transformer
does not consume the subset-enumeration budget meant for guessing. Example:
`examples/rectifier_6pulse_transformer.yaml` (floating star point, zero shunts).
Still open: a 12-pulse stack where freezing the DC filter choke meets six frozen
windings.

v0.1 limits (fail closed): only inductors in the star (no `sat_inductor`
and no current sources); the star must have ≥2 live
(non-frozen) members — degeneration to a single live member (e.g. two legs
in dead time at once with three zero currents) ends with an explicit error;
the commutation livelock guard remains for shapes outside the contract.
Example: `examples/inverter_three_phase_floating_neutral.yaml` (dead time
500 ns; common-mode voltage (CMV) levels of ±Vdc/6·2 and ±Vdc/2, as theory predicts).

### 6.2. Ideal topology restrictions

Ideal shorts and sources can create a singular or contradictory circuit. The kernel does not add a hidden `gmin` or parasitics. An invalid topology ends with a `TopologyError` describing the closed elements. A loop of only capacitors and voltage sources (two capacitors in parallel, a capacitor directly across a source) is such a topology; the error then names the capacitors and suggests an `esr` (§6.1.0a).

The reference kernel has one narrow exception to the assumption that every L remains an
active differential state in every mode. When an inductor current is exactly zero
within an explicit state-constraint tolerance (in this version `1e-9 A`), the
ordinary mode is singular, and the reduced `0 V` branch has a current forced by KCL
to be identically zero as an affine map of all active states, the compiler may
temporarily mark that inductor as a `constrained_inductor`. The global state identity
is preserved, its derivative and value in the idle mode are exactly zero, and
re-energizing activates the state again. The reduction is a fallback after the
ordinary mode fails to compile; it is exercised by the canonical DCM converters, by
floating inductor stars and by coupled-winding groups (§6.1.3). It is not a general
descriptor/DAE solver nor permission to freeze a non-zero current; an inconsistent
branch or a more general cutset still ends with a `TopologyError`.

In `v0.1` the active-set search for diode states supports at most 64 ideal diodes
(thyristors count towards this limit), and each stable-mode query examines at most
65,536 candidate configurations, ordered by Hamming distance from the preferred
configuration. Models with up to 16 diodes still have their full state space
enumerated. These are limits of the reference kernel, not of the target format.

### 6.3. `model_meta` declaration (optional)

Every component may declare the provenance and validity envelope of its device
model:

```yaml
- id: q1
  type: ideal_switch
  p: in
  n: sw
  control: pwm
  model_meta:
    version: "vendor-part-model/1.2"
    validity:
      frequency_max_hz: 250000
      temperature_min_c: -40
      temperature_max_c: 125
      dvdt_max_v_per_s: 5.0e+9
```

Rules:

- `version` is required; all `validity` fields are optional.
- Limits are plain finite numbers (SI / degrees Celsius according to the field
  name) — no unit strings and no `${parameter}` references, because they describe
  the device model, not the resolved circuit.
- The declaration is **a record, not an evaluation**: the reference kernel simulates
  ideal elements and does not compare operating conditions with the envelope. The
  resolved model echoes the declaration with the field `"evaluated": false`, and a
  run with declared metadata receives the warning `MODEL_META_DECLARED_NOT_EVALUATED`
  in `warnings.json`.
- A model without `model_meta` has the same hashes as before this field was
  introduced (purely additive change).

Example: `examples/buck_model_meta.yaml`.

## 6a. Subcircuits — reusable building blocks

A subcircuit is a named, self-contained unit: explicit `ports:`, an explicit list of
`parameters:` and a body of ordinary components (plus optional `couplings:`).
An instance is a component with `type: subcircuit`, a `ref:`, a `ports:` map and —
when the definition declares parameters — a `params:` map.

```yaml
subcircuits:
  three_phase_diode_bridge:
    ports: [a, b, c, dcp, dcn]
    components:
      - {id: d1, type: diode, p: a, n: dcp, initial: off}
      # ... d2..d6

circuit:
  components:
    - id: y
      type: subcircuit
      ref: three_phase_diode_bridge
      ports: {a: ya, b: yb, c: yc, dcp: ytop, dcn: dcmid}
```

Rules (all enforced fail-closed, naming the offending definition/instance):

- **Expansion is deterministic and happens at load time**, source-to-source,
  before semantic validation (as with machines). Every derived component goes
  through the same rules as a hand-written one.
- **Namespacing**: derived identifiers and internal nodes are
  `<instance>.<internal>`. The separator is `.`, and a segment (definition, port,
  parameter or instance name, internal id/node) must not contain `.` or `/` —
  so the convention maps 1:1 onto a KiCad hierarchical sheet path (separator `/`).
  This is the one interoperability decision that is cheap now and expensive later.
- **Hierarchy = presentation, never physics**: the canonical IR and `model_sha256`
  are functions of the flattened model only. A flat and a hierarchical description
  of the same circuit (same ids after flattening) have an identical physics hash.
- **Self-containment**: a `${ref}` in a definition body must name a declared
  subcircuit parameter — never a global one. A declared but unused parameter is
  an error. A binding in `params:` can be a literal with a unit or
  `${global_parameter}`; the dimension is checked after expansion, in the context
  of use.
- **Ports**: the `ports:` map must bind exactly the declared set (a missing port or
  an extra port = error). An internal node named like the ground (`circuit.ground`)
  is an error — a subcircuit reaches outside only through its ports; ground is
  bound through a port.
- **Addressing**: `v(instance.port)` resolves at load time to
  `v(bound_node)`; `i(instance.port)` is an error (a port is a net, not a branch).
  Internals are addressed directly: `v(y.d1)`, `i(y.d1)`.
- **Nesting** is allowed (depth limit 8); recursive references are an error that
  prints the cycle. Id collisions after expansion are errors.
- **`couplings:` in a definition** refer to local inductors and, after expansion,
  go (with prefix) into `circuit.couplings` — a transformer as a subcircuit is an
  intended use case.
- **Render**: `pear-sim render` draws an instance as a box with named ports;
  `--expand-subcircuits` draws the flattened components. There is no alternative
  connectivity path (the clean-room rule is unchanged).

Equivalence regression: `examples/rectifier_12pulse_yyd.yaml` (two bridges as
one definition, instances `y`/`z`) produces waveforms **bit-for-bit** identical
to the flat description — tested in `tests/test_subcircuits.py`.

## 7. Controls

`controls` is a map `control_id -> definition`:

```yaml
controls:
  pwm:
    type: pwm
    frequency: 20 kHz
    duty: 40 %
    phase: 0 s
```

`v0.1` supports `pwm`, the experimental `complementary_pwm`, a narrow
`three_phase_spwm`, the multilevel `three_phase_npc_spwm` (§7.3), the closed loops
`sampled_pi_pwm` (§7.1) and `sampled_dq_current_pwm` (§7.2), and the signal-layer
modulators `signal_pwm` (§9.3) and `signal_svpwm` (§9.6).

Fields of `pwm` / `complementary_pwm`:

| Field | Required | Rule |
|---|---:|---|
| `type` | yes | `pwm` or `complementary_pwm` |
| `frequency` | yes | positive frequency |
| `duty` | yes | dimensionless `[0, 1]`; a percentage is allowed |
| `phase` | no | time; defaults to `0 s` |
| `dead_time` | for `complementary_pwm` | non-negative time, shorter than both pulses |

`complementary_pwm` expands the block name into two outputs `<id>.high` and `<id>.low`.
The portable contract limits a model to 256 logical controls and 256 gate outputs
after expansion; this way every accepted model fits within the limits of the
`gate-drive.json` producer and its report consumer.
An output leaving the active state turns off immediately, and its partner turns on
after `dead_time`; the interlock is built in and cannot be disabled. With zero dead
time both changes are one atomic transaction, serialized as off-before-on. The exact
contract for intervals, constraints and the observation artifact is described in
[Complementary PWM and dead time](m3-complementary-pwm.md).

### 7.1. Closed loop: `sampled_pi_pwm`

A sampled PI controller driving the duty of one gate output (voltage- or
current-mode, depending on `sense`):

```yaml
controls:
  ctrl:
    type: sampled_pi_pwm
    frequency: 50 kHz     # cycle frequency = sampling frequency
    sense: v(out)         # MUST be listed in outputs (auditable signal)
    reference: 12 V       # dimension matching sense (V for v(...), A for i(...))
    kp: 0.002             # duty per unit of error
    ki: 30                # duty per (unit of error * second)
    duty_min: 5 %         # strictly 0 < duty_min < duty_max < 1
    duty_max: 95 %
```

Deterministic control law (exact order of events):

1. At every cycle boundary `t_k = start + k/frequency` the controller samples
   `sense` from the **state and topology before the edge** — sampling happens
   strictly before any gate edge with the same timestamp.
2. `e_k = reference − y_k`; integral candidate `I' = I + e_k·T_s`;
   `duty_k = clamp(kp·e_k + ki·I', duty_min, duty_max)`.
3. Anti-windup by conditional integration: the integral freezes only when the
   output is saturated **and** the error pushes deeper into saturation; an error
   that drives out of saturation always integrates (otherwise the loop locks up
   at the limit).
4. The gate is on for the first `duty_k` fraction of the cycle; the OFF edge is
   injected into the schedule at the moment of computation.

Strictly sharp duty limits guarantee that gate edges never coincide with a cycle
boundary. The full controller trace (time, sample, error, integral, duty command,
saturation flag) is written per cycle to the `control-trace.jsonl` artifact;
`gate-drive.json` marks such an output with `command_source: closed_loop_sampled_pi`.
Example model (buck with a load step):
`examples/buck_closed_loop.yaml`. Gain selection is supported by the stability
analysis (`pear-sim analyze stability` with a declared compensator `[kp, ki]/[1, 0]`).

### 7.2. Three-phase closed loop: `sampled_dq_current_pwm` (grid-following)

A dq current controller with a PLL in the loop — one logical block expanded into six
gate outputs `<id>.<a|b|c>.<high|low>`:

```yaml
controls:
  ctrl:
    type: sampled_dq_current_pwm
    frequency: 10 kHz            # carrier = sampling frequency
    dead_time: 1 us
    sense_currents: [i(lfa), i(lfb), i(lfc)]   # abc, MUST be in outputs
    sense_voltages: [v(pa), v(pb), v(pc)]      # PCC voltages for the PLL
    id_ref: 5 A
    iq_ref: 0 A
    current_kp: 12               # V per A
    current_ki: 3000             # V per A*s
    voltage_scale: 400 V         # half DC bus: duty = 0.5*(1 + v_cmd/scale)
    pll_nominal_frequency: 50 Hz
    pll_bandwidth: 20 Hz         # + optional pll_damping, theta0_deg
    injection: min_max           # optional, as in SPWM
```

Deterministic law at every cycle boundary `t_k` (exact order):
sample 3 currents and 3 voltages FROM THE STATE BEFORE the edge → SRF-PLL step
(control in this cycle uses the angle before the update) → Park transform of the
currents → one PI step per axis with anti-windup by conditional integration against
the ±`voltage_scale` clamp → inverse Park (+ optional min-max injection) → three
center-aligned pulse pairs with dead time (turn-off immediately, turn-on after the
dead time). A full per-cycle trace (theta, frequency, i_d/i_q, v_d/v_q commands,
duties, saturation) is in `control-trace.jsonl`; `gate-drive.json` reports the outputs
as `command_source: closed_loop_sampled_dq` and verifies the absence of overlap.

Restrictions: `duty_min/duty_max` strictly in (0,1) with margins for dead time;
PLL bandwidth < 1/10 of the sampling frequency; the grid island must be referenced
to ground (see the example `grid_following_converter.yaml` — a star-point grounding
resistor; NOTE: such a resistor creates a fast zero-sequence mode and the model
should declare the implicit `radau` solver).
dVOC (grid-forming) is deliberately a future capability.

`three_phase_spwm` is a single logical source that must be the only entry in
`controls` (this applies to every three-phase modulator, including
`three_phase_npc_spwm`). It expands into six outputs `<id>.<a|b|c>.<high|low>`.
Source fields:

| Field | Required | Rule |
|---|---:|---|
| `type` | yes | exactly `three_phase_spwm` |
| `carrier_frequency` | yes | positive, greater than `fundamental_frequency` |
| `fundamental_frequency` | yes | positive frequency |
| `modulation_index` | yes | dimensionless `[0, 1)`; overmodulation is rejected |
| `injection` | no | `none` (default) or `min_max` — reference centering `-(max+min)/2` (SVPWM equivalent); pure common-mode component: line-to-line voltages unchanged, linear range extends to `2/sqrt(3) ≈ 1.1547` |
| `carrier_phase` | no | time; defaults to `0 s` |
| `reference_phase` | no | time shift of the sinusoidal reference; defaults to `0 s` |
| `sequence` | no | explicit `abc` or `acb`; defaults to `abc` |
| `dead_time` | yes | non-negative and shorter than `0.5*(1-m)/carrier_frequency` |

The carrier/fundamental ratio must be an integer, at least two, and must not exceed
the portable limit of 8333. The source mathematically samples the references at the
middle of each cell offline and centers the pulses in the same cell; it does not
claim to be a causal digital modulator. Commands and realized events are recorded
separately. The full contract, formulas and claim boundaries are described in
[Three-phase SPWM and harmonic observation](m3-spwm.md).

### 7.3. Multilevel NPC modulator: `three_phase_npc_spwm`

`three_phase_npc_spwm` is a single logical source (it must be the only entry in
`controls`) expanded into `6*(levels-1)` gate outputs. For three levels
(default) these are **twelve** outputs
`<id>.<a|b|c>.<outer_high|inner_high|inner_low|outer_low>` — the two complementary
pairs of standard NPC gating per leg: `outer_high ↔ inner_low`
(upper pair) and `inner_high ↔ outer_low` (lower pair). For five levels
(`levels: 5`, diode-clamped bridge) — **twenty-four** outputs
`<id>.<a|b|c>.<h1..h4|l1..l4>`: upper arm S1..S4 = `h1..h4`, lower arm
S5..S8 = `l1..l4`, complementary pairs `(h_k ↔ l_k)`; pair `k` pulses in band
`levels-1-k` of the carrier stack. The runtime additionally enforces the **staircase
invariant** of diode-clamped gating (an arm device is never on without its neighbor
closer to the pole: `h1⇒h2⇒…`, mirrored in the lower arm).

**Cold start of deep bridges:** at start-up most internal nodes of a leg float
(both adjacent switches off), and the consistent configuration of clamping diodes
lies outside the reach of the distance-ordered enumeration. The active-set search
then falls back to a **deterministic anchor repair**: a floating node is anchored
first through a non-antiparallel diode (structural classification by terminal pair,
never by name), and dead ends backtrack; models that fit within the enumeration
budget never take this path (behavior bit-for-bit unchanged). Repair counter:
`Circuit.active_set_search_stats()["anchor_repairs"]`.

Modulation law (level-shifted, in-phase-disposition (PD) carriers): the reference
range `[-1, 1]` is divided into `levels-1` carrier bands of height `2/(levels-1)`; the
pair of a band realizes a cell-centered pulse with dead time, pairs outside their band
are saturated (for 3 levels: `duty_upper = clip(r, 0, 1)`,
`duty_lower = clip(1+r, 0, 1)`). Saturating a pair for whole half-periods is how NPC
works, not an edge case. At a band transition, conduction is handed over at the cell
boundary through one dead time.

| Field | Required | Rule |
|---|---:|---|
| `type` | yes | exactly `three_phase_npc_spwm` |
| `levels` | no | `3` (default) or `5`; other depths fail closed (the band law is generic, but they lack a validated topology and a reference fixture) |
| `carrier_frequency` | yes | positive, greater than `fundamental_frequency`; integer ratio, `>= 2`, `<= 100000/(18*(levels-1))` (2777 for 3L, 1388 for 5L) |
| `fundamental_frequency` | yes | positive frequency |
| `modulation_index` | yes | dimensionless `[0, 1)` (with `min_max` up to `2/sqrt(3)`) |
| `injection` | no | `none` (default) or `min_max` |
| `carrier_phase` / `reference_phase` | no | times; default `0 s` |
| `sequence` | no | `abc` (default) or `acb` |
| `dead_time` | yes | non-negative and shorter than a quarter of the carrier period |

**Minimum-pulse suppression (declared behavior):** commands near band edges can be
arbitrarily short; a commanded pulse no wider than the dead time is suppressed —
exactly like the minimum-pulse interlock of a hardware gate driver — and the
suppression is recorded per cell in `modulation-cycles.jsonl` and counted in the
`NPC_MINIMUM_PULSE_SUPPRESSED` warning. The runtime additionally enforces the NPC
gating invariant (an outer switch never on without the adjacent inner one, never both
outer switches together) as a hard interlock.

Artifacts: the authoritative per-carrier-cell command record in
`modulation-cycles.jsonl` (schema
`pear-sim/modulation-cycle/three-phase-npc-spwm/v0.1`) and the compact observation
`npc-modulation.json`, whose content is recomputed entirely and independently
from the modulation law (bit-exact contract, as for `three_phase_spwm`). It does
not claim to be a causal digital modulator.

`ideal_switch.control` must point to an existing control output. Events in the
transient window follow the `(start, stop]` convention, and the state at `start` is
right-continuous. A model should not rely on YAML map ordering.

## 8. Transient analysis

`v0.1` always requires exactly one transient analysis and may additionally contain an
explicit harmonic analysis:

```yaml
analysis:
  transient:
    start: 0 s
    stop: 2 ms
    output_step: 1 us
    solver: dopri5
    rtol: 1.0e-8
    atol: 1.0e-10
    max_step: 200 ns
```

| Field | Required | Default | Rule |
|---|---:|---|---|
| `start` | no | `0 s` | start time |
| `stop` | yes | — | time greater than `start` |
| `output_step` | yes | — | positive, not greater than `stop-start`; at most 2 000 001 grid points per run |
| `solver` | no | `dopri5` | `dopri5`, `dop853`, `radau`, `bdf` |
| `rtol` | no | `1e-7` | finite float `>= 2.220446049250313e-14`; a smaller value would be silently raised by SciPy and is therefore rejected |
| `atol` | no | `1e-9` | finite positive float |
| `max_step` | no | none | positive time |

`output_step` defines a common, uniform result grid. Internal adaptive steps and events may occur between grid points. A PWM breakpoint splits integration, so the solver never uses history from the wrong topology.

The optional `analysis.harmonics` has the fields `start`, `stop`,
`fundamental_frequency`, `max_harmonic` and a non-empty list `signals`. The window is
half-open `[start, stop)`, must lie within the transient and span an integer number of
fundamental periods. `max_harmonic` is an integer in the range `2..256`.
The minimum cost `4 * period_count * max_harmonic` must not exceed 100000
panels. A separate budget of 100000000 units also accounts for the harmonic order,
the eight-point quadrature and the number of signals; the runtime enforces both hard
limits after splitting into steps and topologies.
Each entry of `signals` maps exactly one existing output to exactly one semantic
from a closed set:

```text
line_voltage_ab | line_voltage_bc | line_voltage_ca
phase_voltage_a | phase_voltage_b | phase_voltage_c
phase_current_a | phase_current_b | phase_current_c
generic
```

The same signal or semantic must not occur twice (except `generic`). Family
consistency applies: `v(...)` may only have a voltage semantic, and `i(...)` only
`phase_current_*`. `generic` accepts any recorded signal — a DC-side ripple, a
single-phase quantity — with no family checks and no sequence components. THD is not
computed for it (`not_computed_generic_signal`): the harmonic amplitudes are the result.
Every result record states its unit `V`/`A`. The Fourier integral is computed from the
dense output over topology intervals, not from the `waveforms.csv` grid. The method,
bandwidth, coefficient convention, DC policy and THD policy are recorded in
`harmonics.json`; `status: observed` on its own is not a validation result.

## 9. Outputs

`outputs` is a non-empty list of expressions:

```yaml
outputs:
  - v(out)
  - v(q1)
  - i(l1)
```

Semantics:

- `v(node)` — node voltage with respect to `circuit.ground`;
- `v(component)` — `v(p)-v(n)` of the referenced component;
- `i(component)` — component current from `p` to `n`.

The target must exist. `i(node)` is not allowed. A duplicate output is rejected
by the loader and by the public JSON Schema.

Machine signals (§9.1) and declared signal-layer values (`s(...)`, §9.3) are
additional output families.

### 9.1 Machine signals

A declared machine exposes its mechanical state as outputs:

```yaml
outputs:
  - torque(motor)        # N*m — electromagnetic torque, INSTANTANEOUS value
  - speed_mech(motor)    # rad/s — mechanical speed
  - speed_elec(motor)    # rad/s — electrical speed = pole_pairs * mechanical
  - angle_elec(motor)    # rad — electrical rotor angle
  - angle_mech(motor)    # rad — mechanical angle = electrical / pole_pairs
```

**The target is a machine id from the `machines` section, never a node or a component.**
A machine expands into branches with identifiers such as `motor.a.l`;
`speed_mech(motor.a.l)` is a load error, not a branch reading.

**Why there is no `angle(m)` or `speed(m)`.** The mechanical state carries the
MECHANICAL speed and the ELECTRICAL angle — they differ by the number of pole pairs,
and both are plain radians, so a mix-up shows neither in the unit nor in a test.
The model must say which one it means.

**Mechanical angle convention.** `angle_mech` = `angle_elec / pole_pairs`,
which places the origin of the mechanical axis at `initial_angle_rad / pole_pairs`.
The state carries only the electrical angle, so this is **a format definition,
not a measurement**.

**Sampling instant.** Speed is HELD within one integration segment, while the angle
advances exactly within it. A sample carries the values of the segment it lies in —
also when it is written after the mechanical state update at the end of that segment.
Torque is computed from the currents at the sample instant, so it is an instantaneous
value; per-segment torque integrals (Gauss quadrature) are in `machine-trace.jsonl`.

**What `metrics.json` does NOT provide for these signals.** A machine signal is not
an affine map of the electrical state, so the continuous quadrature from which
`mean`, `rms` and `integral` are produced is not defined for it. The entry has
`time_integrated_status: not_computed` with a stated reason, and **the window keys are
ABSENT, not zeroed** — indexing them raises an exception instead of returning a
plausible-looking number. Output-grid statistics (`sample_*`) are true and remain.

Harmonic analysis (`analysis.harmonics`) accepts **only** `v(...)`
and `i(...)`: it computes the Fourier series by mapping the expression onto the
compiled circuit, where a machine signal has no representation.

### 9.2 Imposed shaft speed — `machines.<id>.driven_speed`

The dynamometer case: the shaft is driven from outside, and **the response is the torque**.

```yaml
machines:
  motor:
    type: pmsm
    # ... electrical parameters ...
    driven_speed:
      initial_rad_s: 0.0
      final_rad_s: 157.0796
      ramp_start_s: 0.0
      ramp_stop_s: 0.03
```

Constant speed = `initial_rad_s` alone. A ramp requires **both** `final_rad_s`
and `ramp_stop_s` — half a ramp has no reading that would not be a guess at the
author's intent. The profile is clamped at both ends (before `ramp_start_s`
and after `ramp_stop_s` the speed is constant).

**Fields forbidden in this mode:** `inertia_kg_m2`, `viscous_friction_nm_s`,
`load_torque_nm`, `initial_speed_rad_s`. The mechanical loop is not integrated,
so they would have no effect; silently accepting them would make the reader
believe that the declared inertia does something. The loader refuses, listing
the offending fields.

**What it is for.** Two reasons, both substantive:

1. a synchronous machine fed from a fixed frequency **will not start from
   standstill**, so a run with free mechanics cannot even ask the start-up question;
2. our mechanical loop has a **measured, still-open discrepancy** against an
   independent reference machine model. Imposing the speed removes that loop from
   the answer, so a torque test stays meaningful while the discrepancy is open.

**The hold scheme and its two consequences.** Speed is **read at the segment
boundary and held** within the segment — the same as integrated speed. Sampling the
ramp continuously inside a segment would turn the back-EMF from a clean sinusoid into
a chirp, i.e. trade a bounded, shrinking splitting error for an unbounded modeling
error. Hence:

- speed **lags** the ramp by at most the profile drift within one segment
  (measured: 5.24e-2 rad/s for a ramp of 157.08 rad/s / 30 ms
  and a 10 µs segment — matching to the digit);
- the angle is a **left Riemann sum** of the ramp, so it underestimates the exact
  integral by half a drift per step (measured: 15.7050 rad versus 15.7080 rad
  analytically).

**Independence from torque holds up to the segment grid.** Torque does not enter
the speed, but segment boundaries include natural commutations, whose instants depend
on the currents. A change of torque therefore shifts **WHEN** the profile is read,
never **WHAT** it returns; the residual is bounded by the same drift within one
segment (measured: a 27 N·m change of torque moved the speed by
1.96e-2 rad/s, i.e. 1.25e-4 relative).

### 9.3 Signal layer — the `signals` section

An executable control layer: **blocks as data**, not yet another ready-made type
in Python.

```yaml
signals:
  task:
    period: 20 us          # clock is DECLARED, never defaulted
  blocks:
    vout:  {type: probe, source: v(out), unit: V}
    vref:  {type: constant, value: 12.0, unit: V}
    err:   {type: sum, inputs: [vref, vout], signs: [1, -1], unit: V}
    prev:  {type: unit_delay, input: integral, initial: 0.0, unit: V}
    integral: {type: sum, inputs: [prev, err_dt], unit: V}
    duty:  {type: saturate, input: u, min: 0.05, max: 0.95, unit: V}

controls:
  ctrl:
    type: signal_pwm       # BRIDGE: the duty comes from a signal
    frequency: 50 kHz
    duty_signal: s(duty)
```

(This is an excerpt: the blocks `err_dt` and `u` are omitted. A complete model is
`examples/buck_closed_loop_signals.yaml`.)

**Block kinds:** `constant`, `probe`, `time_source` (step/ramp), `gain`,
`sum` (with optional signs), `product`, `saturate`, `unit_delay`,
`switch` (model flag), `lookup_1d`, `compare`, `logic`, `select`, plus the
measurement-chain blocks `noise`, `quantize`, `delay` (§9.5) and the rotating-frame
blocks `park`, `inverse_park`, `lookup_2d`, `hypot`, `divide` (§9.7), and the
controller shorthand `pi` below.

**`pi` — discrete PI controller.** One block instead of six:

```yaml
duty:
  type: pi
  input: err          # the error signal
  kp: 0.002           # proportional gain
  ki: 30.0            # integral gain in 1/s (multiplied by task.period internally)
  min: 0.05           # optional output limits, give both or neither
  max: 0.95
  kaw: 1.0            # back-calculation anti-windup gain; default 1 with limits, 0 without
  initial: 0.0        # integrator start value
  unit: V
```

It means exactly: `int[k] = int[k-1] + ki·T·e[k] + kaw·(u[k-1] − raw[k-1])`,
`raw[k] = kp·e[k] + int[k]`, `u[k] = clamp(raw[k], min, max)`. The saturation gap
re-enters the integrator one sample late, as in a real implementation. The block is
expanded at load time into `gain`, `sum`, `saturate` and `unit_delay` blocks named
`<id>__p`, `<id>__int`, …; the output keeps the block's own id, and the internals can
be recorded like any other signal (e.g. `s(duty__int)`).

**The unit is DECLARED in every block.** The control layer carries volts, amperes,
newton-metres and dimensionless quantities side by side, so no syntax can infer it.
A `probe` **cannot rename** what it reads — its `unit` must match the unit of the
source. Block values are **bare numbers**: the unit is already in `unit`, and a second
record of the same fact is a second thing that can disagree with the first. Times
(`task.period`, `time_source.start/stop`) carry units as everywhere else.

**Execution order is a function of the graph, not of the order in the file**
(topological sort, ties broken by id) and is recorded in the bundle.

**An algebraic loop is an error; a loop through `unit_delay` is a program.**
A cycle through combinational blocks is rejected with the **cycle named** — its
value would depend on the order in which the author happened to write the blocks.
A cycle through `unit_delay` is not a loop but a **difference equation**, because the
block responds with the value of its input **from the previous tick**. Both
back-calculation anti-windup and angle prediction need this.

**Probes must be in `outputs`** — the same rule that applies to sampled
controllers: whatever the controller samples must be a recorded signal, otherwise
the run cannot be checked against what the controller actually saw.

**The `signal_pwm` bridge**: one tick = one carrier cycle, and the loader refuses any
other ratio. A duty updated at a rate that the carrier does not divide belongs to no
single cycle, and silently picking one is exactly the timing assumption this format
does not make on the author's behalf. Controller and modulator are kept **separate**.

**The graph enters the physics hash.** Two models with different control laws have
different waveforms, so they cannot share a hash.

**Metrics:** a declared signal is a discrete value held between ticks, so — like a
machine signal — it gets `time_integrated_status: not_computed` **without window
keys**. Per-tick values are in `signal-trace.jsonl`.

#### Comparisons, logic and signal-controlled branching

```yaml
    u_prev:   {type: unit_delay, input: u, initial: 0.5, unit: "1"}
    hi:       {type: constant, value: 0.95, unit: "1"}
    at_upper: {type: compare, left: u_prev, right: hi, op: ge, unit: bool}
    limited:  {type: logic, op: or, inputs: [at_upper, at_lower], unit: bool}
    integral: {type: select, condition: limited,
               if_true: int_prev, if_false: int_free, unit: V*s}
```

`compare` takes `left`, `right` and `op` ∈ `lt` / `le` / `gt` / `ge`.
`logic` takes `op` ∈ `and` / `or` / `not` and `inputs` — exactly one input for
`not`, at least two for the others. `select` takes `condition`, `if_true` and
`if_false`.

**There is no equality, and that is a decision.** Two floating-point numbers that
reached the same value by different paths are equal by accident — `eq` would be a
condition that works in a test and fails in a run. The physical question
("did the limiter engage?") is asked with a threshold via `ge` / `le`.

**A logical value has its own unit `bool` and must not be renamed.**
Dimensionless is not the same claim as logical: duty, a per-unit voltage and a
modulation index are all `"1"`, and any of them wired into a condition input would
be read as "true when non-zero", i.e. as a control law nobody wrote.

**Unit checking works in one direction.** Nothing except a block with unit `bool`
may enter `logic.inputs` or `select.condition`. The other way round is fine: a
logical value wired into arithmetic is **correct and useful**, because multiplying
a term by a 0/1 gate is the standard way of switching a component off.

**`switch` and `select` are not the same.** `switch` chooses by a **model flag**:
which control law this run executes (decoupling on/off) — a property of the
experiment, fixed before the first tick. `select` chooses by a **signal**: which
branch this tick takes — a property of the state, decided every tick. Merging them
would equate the questions "what did we run?" and "what did it do?", and only the
first belongs to a comparison of two runs.

**Conditional integration (anti-windup) is what these blocks are for** —
`examples/buck_closed_loop_antiwindup.yaml`. Beware of the trap: the saturation test
must read the command **from the previous tick**, because `integral → u → limited →
integral` closes within one tick and the loader would reject that as an algebraic
loop. One tick of delay in the anti-windup path is what every digital controller
does — here it is declared rather than hidden inside a block.

**Out of scope for v0.1, named:** multiple rates (one task only) and continuous
blocks (a block that needs its own integration belongs in the kernel).

### 9.4 Salient-pole machine — `d_axis_inductance_h` / `q_axis_inductance_h`

Declaring **different** axis inductances makes the machine a **salient-pole**
machine.

```yaml
machines:
  motor:
    type: pmsm
    d_axis_inductance_h: 1 mH
    q_axis_inductance_h: 3 mH
    flux_linkage_wb: 0.0        # PURELY RELUCTANCE machine — zero is legal
```

**Zero magnet flux is a machine, not an error.** A synchronous reluctance machine
has no magnets at all — all its torque comes from saliency. Only the combination of
**no magnets and no saliency** is rejected: such a machine has no torque mechanism.

**What changes inside.** The windings stop being three independent inductors and
become **one coupled group** whose matrix depends on the rotor position:

```
L_aa = L_A + L_B*cos(2θ),   L_ab = -L_A/2 + L_B*cos(2θ - 2π/3)
L_A = (Ld + Lq)/3,          L_B = (Ld - Lq)/3
```

The matrix is **held within a segment**, exactly like the speed, so the electrical
side stays strictly linear inside a segment, and the splitting error is of the same
kind. **Cost measured before implementation:** mode recompilation 0.102 ms × 7537
segments of a 40 ms run = **0.77 s on a 9.3 s run (8 % overhead)** — not a
prohibitive cost.

**The phase-to-phase couplings are NEGATIVE**, and that is physics: the flux of one
phase opposes the others. The admissibility condition is **positive definiteness**
of the group matrix, not the sign of a single coupling.

**The zero-sequence component is singular and set explicitly.** The vector [1,1,1]
of this matrix has, by definition, eigenvalue zero — the machine presents no
inductance to zero-sequence current. With a floating star this current has no path,
so the value is **unobservable**, but the kernel inverts the matrix and cannot invert
a singular one. The entry along [1,1,1] is therefore set to `L_A`. **That the result
does not depend on it is a TEST, not a comment:** a tenfold change of this number
moves current and torque by 1e-15 relative.

**Torque** has two terms: `T = 3/2·p·[ψ_f·i_q + (Ld − Lq)·i_d·i_q]`.
The second is the only one left to a machine without magnets.

**Backward compatibility measured:** a non-salient machine declares no couplings
and computes **bit-for-bit** as before (waveforms, metrics and
`machine-trace.jsonl` identical; only `wall_time_s` differs).

**What is still missing here:** saturation. A real reluctance machine operates
deep in saturation, so constant Ld/Lq is a linear model — good for checking that
reluctance torque exists, not for designing a machine. Saturation is covered by
flux maps (§9.8).

### 9.5 Measurement chain — `noise`, `quantize`, `delay`

What the controller SEES stops being what IS. Three signal-layer blocks.

```yaml
signals:
  blocks:
    i_true:  {type: probe, source: i(l1), unit: A}
    i_noisy: {type: sum, inputs: [i_true, adc_noise], unit: A}
    adc_noise: {type: noise, stddev: 0.02, seed: 12345, unit: A}
    i_adc:   {type: quantize, input: i_noisy, step: 0.01, mode: round, unit: A}
    i_seen:  {type: delay, input: i_adc, samples: 2, initial: 0.0, unit: A}
```

**`noise` — THE SEED IS MANDATORY.** The promise of this project is
`seed → bit`; a generator whose seed the model does not state cannot keep it, and
"it was random" is not a reproducible run. A sample is a **pure function of
(seed, tick number)** — not of internal state — so evaluating the graph twice for
the same tick gives the same number. `stddev: 0` gives exactly zero.

**`quantize`** — converter step, `mode: round | truncate`. The step must be
positive.

**`delay` — a delay of N TICKS, not one.** A controller's computational delay is
usually one or two samples, and being off by one is a phase error that no aggregate
metric shows. For the first N ticks the block returns the declared initial value: a
delay of N samples means that at the start nothing that old exists yet. Returning the
oldest available sample would silently shorten the delay exactly where the transient
is. `delay` **breaks a loop** just like `unit_delay`.

### 9.6 Vector-commanded bridge — `signal_svpwm`

The modulator a field-oriented controller actually needs.

```yaml
controls:
  bridge:
    type: signal_svpwm
    frequency: 10 kHz
    alpha_signal: s(u_alpha)     # COMMANDED VECTOR, not a duty
    beta_signal: s(u_beta)
    dc_link_voltage: 600 V
    dead_time: 1 us
```

**The difference from `signal_pwm` is fundamental.** That one takes a **duty**,
i.e. the result of modulation — so the control law has to modulate by itself. This
one takes the **voltage vector in the stationary frame**, exactly what the inverse
Park transform produces, and decides the sector and the base-vector times itself.
As a result, swapping the modulator no longer means rewriting the controller.

The block expands into **six gates** with an internal interlock and dead time; the
pulse is **centered** in the carrier period. One task tick = one carrier cycle (the
loader refuses any other ratio). Overmodulation is **rejected**, not clipped. Sector
and vector times go to `signal-trace.jsonl`.

**Checkable with one number:** a commanded vector `alpha = 200 V` gives a mean phase
voltage on the load of **200.0000 V** (error 5.7e-14, continuous quadrature). Note
for your own measurements: a mean over the **output grid** will show something else
if the grid is sparse relative to the carrier — a centered rectangle sampled 10 times
per cycle gives 100 V instead of 150 V. That is grid aliasing, not a modulator error;
`metrics.json` computes the mean by continuous quadrature and is reliable here.

### 9.7 Rotating-frame blocks and 2D maps

The last missing links needed to write a field-oriented controller as data:

| block | what it does |
|---|---|
| `park` | abc → dq, field `axis: d\|q`, inputs `a`,`b`,`c`,`angle` |
| `inverse_park` | dq → αβ, field `axis: alpha\|beta`, inputs `d`,`q`,`angle` |
| `lookup_2d` | two-dimensional map (the shape of flux maps ψ(id,iq)) |
| `hypot` | vector magnitude — for **circular** voltage limiting |
| `divide` | division; **division by zero is an error naming the block**, not a silent infinity |

**A block yields ONE value**, so the axis is part of the declaration rather than the
block having two outputs: a controller declares `park` twice, once per axis, and the
pair reads like the transform it is.

🔴 **The convention is the same as in the machine, and this is TESTED.**
Phase offsets and amplitude-invariant scaling match `pear_sim.spec.machines` to the bit.
A controller transforming in a different convention than the machine it drives
produces a **plausible and wrong** torque that no aggregate metric reveals — and this
class of mistake has occurred three times in this repository's history.

**Circle limit, not rectangle.** The inverter limit is a circle; clipping each axis
separately would let the magnitude reach √2 times the real limit. Hence `hypot` +
`divide` + `select`, not two `saturate` blocks.

### 9.8 Machine flux maps — `flux_map`

Saturation instead of constant Ld/Lq. A real reluctance or IPM machine operates
deep in saturation, where axis inductances drop with current and the axes influence
each other (cross-saturation) — two constants express neither.

```yaml
machines:
  motor:
    type: pmsm
    pole_pairs: 4
    stator_resistance_ohm: 0.05
    flux_map:
      id_axis: [-40, -20, 0, 20, 40]
      iq_axis: [-40, -20, 0, 20, 40]
      psi_d: [[...], ...]     # one row per id_axis value
      psi_q: [[...], ...]
```

**The kernel reads SLOPES, not chords.** In the equation `v = R·i + dψ/dt` it is
`dψ/di` that multiplies `di/dt`, so the inductance matrix is formed from local
derivatives of the map at the segment's operating point. The chord `ψ/i` would give
**the right flux and the wrong dynamics** — an error that looks correct in steady state.

**Torque** from the general form `T = 1.5·p·(ψd·iq − ψq·id)`, valid both in
saturation and outside it.

**One fact, one place.** The map already contains the magnet flux (`ψd(0,0)`) and
the axis slopes, so the loader **refuses** a simultaneous declaration of
`flux_linkage_wb`, `d_axis_inductance_h`, `q_axis_inductance_h`
and `stator_inductance_h`.

**Physical gate:** flux must increase with its own axis current. A negative slope is
not saturation but a broken table — and would make the inductance matrix indefinite.
Checked at load time.

**Internal oracle:** a **linear** map must reproduce the machine with constant
Ld/Lq — because it is the same physics written differently. Measured: matrix to
6.3e-18 H, a full drive run to 8.9e-13 (0.0000 %).

**The machine map is separate from the controller map** — a shared one would make
the model measure the agreement of a table with itself and would remove the
estimation error.

**What this does not do:** maps cannot currently be derived from geometry; they must
come from identification (field simulation or measurement). The example in the
repository (`examples/pmsm_saturating_flux_map.yaml`) is explicitly **synthetic**
(tanh) and described as such.

## 7a. Device cards and losses/thermal (sections `device_cards` / `cooling` / `loss_analysis`)

The kernel stays ideal; losses and junction temperature are a **post-processing
estimate** on the recorded waveforms (the usual loss-modeling paradigm of
system-level switched simulators), at an EXPLICITLY declared evaluation temperature —
with no hidden electrothermal iteration. All three sections are optional
and additive (models without them are byte-identical in the contracts).

- `device_cards.<id>`: `kind: switch|diode`; `provenance.source` (mandatory,
  per card) + per-table `provenance {source, page, figure}`; validity envelope
  `validity {tj_min_c, tj_max_c, current_max_a, notes}`; `conduction`
  (table `v_on(i, Tj)` — strictly increasing axes, non-negative values);
  optional `switching` (E_on/E_off for a switch, E_rec for a diode, at declared
  double-pulse-test (DPT) conditions `test_voltage_v`/`test_temperature_c`;
  `voltage_scaling: linear` = the only law in v0.1: E·V_comm/V_test);
  `thermal.foster_stages` (R, τ — up to 16 stages).
- `cooling`: `tims` (static Rth), `heatsinks` (Rth and pressure drop vs
  flow), `fans` (PQ curve, non-increasing). Operating point = intersection of the fan
  PQ curve with the heatsink impedance (fails closed when the curves do not intersect).
- `loss_analysis`: `ambient_c`, `tj_evaluation_c` (must lie within the envelope of
  EVERY card), `devices.<component_id> {card, commutation_voltage, cooling}`.
  Required in `outputs`: `i(<device>)` and the commutation voltage — the estimate
  works only on recorded, auditable signals.

The `losses-thermal.json` artifact in the bundle contains, per device: conduction
energy and mean power (trapezoid over the output grid), switching-energy sums with
event counters (hard diode commutation via `switch_manager` and soft via
`natural_commutation` counted the same way), per-table extrapolation COUNTERS, static
case temperature, a Tj waveform from the exact Foster recursion (energy-balance audit
with independent recomputation; the junction-to-case stages start at their steady
rise for the run-mean power, the same assumption the static case temperature makes,
recorded as `thermal.initial_state`), validity-envelope violation flags and a rainflow
count (ASTM E1049) over ΔTj — the count is a FACT about the waveform; a lifetime
model is deliberately NOT part of the artifact. Declared omissions: no TIM/heatsink
dynamics, no coupling of devices on a shared heatsink, no
spreading/orientation/altitude, no temperature dependence of switching energies.
Example: `examples/buck_with_loss_cards.yaml`.


### 7a.1. Liquid cooling (`cooling.cold_plates`)

Cold-plate card: `flow_axis_l_per_min` + `rth_k_per_w` (non-increasing with
flow) + **`coolant_inlet_c`**. The device binding declares
`cold_plate` and `coolant_flow_l_per_min` — the pump and piping curve is NOT
modeled (in the air path the fan PQ curve and the heatsink pressure drop genuinely
intersect; here the operating point is declared), and the omission is recorded in
the artifact.

Most importantly: a liquid-cooled device has a **different reference temperature**
— it sits at the coolant inlet, not at the cabinet ambient. Using `ambient_c` for a
liquid path would overstate the thermal margin, so the artifact records
`reference_temperature_c` and `reference_temperature_source`. The air and liquid
paths are **mutually exclusive** per device.

Example: `examples/traction_inverter_liquid_cooled.yaml`.

## 10. Format-related CLI

Commands directly related to the model format:

```text
pear-sim catalog [--json]
pear-sim lint MODEL [--set NAME=VALUE]... [--json]
pear-sim explain MODEL [--set NAME=VALUE]... [--json]
pear-sim simulate MODEL [--set NAME=VALUE]... --out DIR [--json]
```

The full CLI contract (including `render`, `report`, `evaluate`, `analyze`,
`compare`, `sweep`, `verify`, `review`, `capabilities`) is described in `README.md`
and `pear-sim --help`.

- `catalog` lists components, controls, analyses and solvers available in this version;
- `lint` parses the model, checks parameters, the initial topology and outputs;
- `explain` shows the resolved model, hashes, assumptions and limitations;
- `simulate` runs the transient and writes the result;
- with `--json`, success stdout contains JSON, and an error goes to stderr as JSON
  with the fields `ok: false`, `error` (class name), `code` (stable machine-readable
  code, §13) and `message`;
- the exit code is `0` on success and `2` for all current user/model/backend errors;
  callers should branch on the JSON `code`, not on the exit code.

Example:

```text
pear-sim lint examples/buck.yaml --json
pear-sim simulate examples/buck.yaml --set duty="45 %" --out run/buck --json
```

## 11. Result artifacts

`simulate --out DIR` creates:

```text
DIR/
  run.json
  model.resolved.json
  waveforms.csv
  events.jsonl
  metrics.json
  analytic-checks.json
  physics-audit.json
  mode-coverage.json
  resonance.json
  confidence-card.json
  gate-drive.json
  modulation-cycles.jsonl
  three-phase-modulation.json
  npc-modulation.json
  harmonics.json
  control-trace.jsonl
  signal-trace.jsonl
  machine-trace.jsonl
  losses-thermal.json
  warnings.json
  summary.json
  artifact-manifest.json
```

`gate-drive.json`, the three-phase / NPC modulation artifacts, `harmonics.json`, the
trace files (`control-trace.jsonl`, `signal-trace.jsonl`, `machine-trace.jsonl`) and
`losses-thermal.json` are present only when the corresponding control, section or
request applies; the `run.json.files` manifest is the authoritative map of the files
of a specific run. `resonance.json` holds the modal (resonance) audit of the recorded
run, and `confidence-card.json` maps the features a model uses onto the versioned
validation matrix and reports the weakest link.

### 11.1. `run.json`

Minimal fields of the current manifest:

```json
{
  "schema": "pear-sim/run/v0.1",
  "pear_sim_version": "0.4.0.dev0",
  "model": "buck_ccm",
  "model_file": "buck.yaml",
  "model_sha256": "...",
  "solver": "dopri5",
  "rtol": 1e-8,
  "atol": 1e-10,
  "stats": {},
  "files": {
    "waveforms": "waveforms.csv",
    "events": "events.jsonl",
    "metrics": "metrics.json",
    "analytic_checks": "analytic-checks.json",
    "physics_audit": "physics-audit.json",
    "mode_coverage": "mode-coverage.json"
  }
}
```

`source_sha256` is the hash of the source file bytes. `model_sha256` is the hash of
the canonical, resolved physical semantics, and `request_sha256` covers the analysis,
solver, tolerances, output grid and list of outputs. `run_sha256` binds the model and
request to the backend and the PEARSim version. The hashes are a portable alpha
contract, but their exact canonicalization may change before a stable schema.

### 11.2. `waveforms.csv`

The first column is `time` in seconds; the following columns are named exactly as in `outputs`. Numbers are written with enough precision for a `float64` round-trip.

```csv
time,v(out),i(l1)
0,40,4
1e-06,40.01,4.03
```

The CSV contains the regular `output_step` grid. Times and discrete state changes are recorded in `events.jsonl`, but the electrical `pre/post` values are not yet separate samples. A dedicated two-sided representation of events in the waveform is a planned extension.

### 11.3. `events.jsonl`

Each line is a separate event JSON. The event log is used to diagnose PWM, diode commutation and mode changes. A consumer should tolerate new fields being added within the same run schema version.

The current record contains `time`, `kind`, `target`, `previous` and `current`. `kind` currently takes `scheduled_gate`, `natural_commutation` or `switch_manager`.

### 11.4. `metrics.json`

`mean`, `rms`, `mean_last_half` and `rms_last_half` are continuous-time metrics.
The kernel integrates each actual dense-output interval in the correct topology mode
with 8-point Gauss-Legendre quadrature. Commutation points have zero measure, and the
`last_half` boundary is exactly `(start + stop) / 2`, even when it falls inside an
internal solver step. The `aggregation_method` field records the method.

Statistics computed only from the requested CSV grid have the prefix `sample_*`.
The retained aliases `min`, `max`, `final` and `peak_to_peak` also describe samples,
which is made explicit in `extrema_method`. `ripple`/`ripple_last_half` appears only
for outputs recognized as structurally continuous in all visited modes. For switched
signals, such as an ideal switch voltage or a diode current, `peak_to_peak` is
reported, not "ripple". The unit is `V` for `v(...)` and `A` for `i(...)`.

Metrics do not replace the waveform or a separate event analysis. The `last_half`
window is a general alpha aggregate, not automatic proof that steady state has been
reached.

### 11.5. `mode-coverage.json`

`mode-coverage.json` has the schema `pear-sim/mode-coverage/v0.1` and the kind
`hybrid-mode-coverage-observation`. It is a narrow, versioned record of the modes
visited by the accepted hybrid trajectory; it is not a reconstruction from
`events.jsonl` alone. The field `claim.validation_claim` is `false`, and the status
`observed` does not mean `pass`. The current producer records the window of the last
half of the transient, as given by `window.start_s/end_s`; it does not claim that this
is automatically a periodic steady state.

The `v0.1` matcher recognizes only the exact canonical single-switch buck, boost and
inverting buck-boost topologies. It explicitly records the roles of source, control,
switch, diode, inductor, capacitor and load; these must not be derived from component
names. For any other model the status is `not_applicable`,
`recognition.subjects` is `null`, and the observation fields are `null`; a consumer
must not infer a mode or topology from that.

For status `observed` the artifact contains:

- the window, PWM period and cycle start;
- explicit classification rules for `on`, `freewheel`, `idle` and `unclassified`;
- complete intervals `[start_s, end_s)` with the exact gate/diode state, cycle index
  and references to the boundary events;
- producer-recorded duration/fraction per mode, which a report need not recompute;
- `natural_commutation.exercised`, the number of events and the number of natural
  diode turn-offs;
- diode conduction time and fraction;
- the inductor current minimum, its time, and
  `maximum_forbidden_reverse_current_a = max(0, -minimum_a)` together with the extrema method;
- the explicit `state_constraint_tolerance_a` used by this kernel version;
- an event-timing table: absolute time, transition, `cycle_index`, offset from the
  cycle start and phase fraction.

`natural_commutation.exercised: false` means only "coverage not exercised"; it is
neither an automatic solver error nor proof that the mechanism passed a test. The HTML
report hashes and renders the artifact without re-classifying modes, aggregating
intervals, computing constraints or assigning a validation status. Current evidence
covers three exact canonical slices, which is why VAL-003 in the
[validation plan](validation-plan.md) remains `partial`; this is neither a general
classifier nor proof of arbitrary DCM/BCM. The writer of the current version adds the
file to every new bundle; the report reader stays compatible with older runs without a
`files.mode_coverage` role and simply skips that section.

### 11.6. Self-check, energy balance and initialization

`analytic-checks.json` contains a `verification-self-check` only when the conservative
matcher recognizes an exactly supported canonical topology. In `v0.1` this is the ideal
buck in CCM. The relations for `Vout`, inductor current ripple, output voltage ripple
and mean diode current have explicit expected/measured values, tolerance, delta, status
and checked preconditions. For another topology the result is `not_applicable`, and
missing data or a too-short window gives `not_run`; the tool does not guess. Passing is
a verification check of the ideal model's mathematics, not a physical validation.

`physics-audit.json` integrates `v*i` of every component with the same method and
records the balance for the full and the last half window. The passive convention is
positive absorbed power for positive current `p -> n`; energy delivered by sources is
reported separately as a positive value. The artifact contains the L/C energy change,
mean source and resistor power, the residual and an explicit tolerance.

`model.resolved.json.initial_conditions` distinguishes explicitly given initial
conditions (IC) from the default zero, gives every value in SI, and marks `warm_start`
when any L/C state is non-zero. PEARSim does not infer an "analytic", "periodic
steady state" or "previous run" origin from the value alone: if the source does not
declare it in a supported contract, provenance says `not_claimed` instead of creating
a presumption.

### 11.7. Switching and harmonic artifacts

Fixed `pwm`/`complementary_pwm` use `gate-drive.json` v0.1. A
`three_phase_spwm` run instead writes `modulation-cycles.jsonl` as the authoritative
set of command cells and `three-phase-modulation.json` as a bounded projection of the
phase -> gate -> device map, the fundamental waveform and a worst-pulse zoom. The full
source of realized changes remains `events.jsonl`; the report does not equate the
command with its realization.

If the request contains `analysis.harmonics`, the producer writes `harmonics.json` with
the full coefficient series from 0 to `max_harmonic`, peak/RMS amplitudes, phases,
THD, bandwidth and — for a complete phase voltage/current triple — Fortescue symmetric
components. DC is kept as `C0` but excluded from THD. The report only validates,
hashes and presents this result; it does not recompute it.

## 12. Determinism and canonicalization

In `v0.1` semantics do not depend on map key order. Components keep their declaration order for state mapping, but physical results should not change with it. Switch and diode topology is canonicalized by sorting identifiers.

`model_sha256` is computed over a canonical representation of the resolved physics:

1. parameters and units are resolved to SI;
2. default values are applied;
3. the result is serialized as canonical JSON (sorted keys, NaN/Infinity rejected);
4. SHA-256 is taken over those bytes.

A YAML document and its JSON equivalent therefore produce the same `model_sha256`
(checked, e.g., for `examples/buck.yaml`). As stated in §11.1, the exact
canonicalization may still change before a stable schema.

## 13. Validation and errors

Model errors should point to the field path or the component identifier. The current error classes and their stable machine-readable codes are:

- `ModelError` (`model.invalid`) — structure, references, outputs or parameter range;
- `UnitError` (`model.units`) — unit parsing or dimension mismatch;
- `TopologyError` (`model.topology`) — a singular/contradictory mode or no stable diode state;
- `SimulationError` (`simulation.failed`) — integration or event-processing failure.

Errors outside these classes are reported as `io.error` (file system), `request.invalid` (invalid request values) or `internal.error`. The codes are part of the public contract: changing one is a breaking change. JSON Pointer locations are planned before v1 is frozen. The early-alpha message text is not yet a stable API.

## 14. Deliberately unsupported in v0.1

- separate declaration of nodes and typed physical domains;
- general parameter expressions;
- controlled voltage/current sources;
- multi-rate control (the signal layer runs exactly one periodic task, §9.3) and
  continuous-time control blocks;
- coupled electrothermal simulation (losses and junction temperature are a
  post-processing estimate at a declared temperature, §7a);
- resources and vendor netlists;
- selecting a SPICE backend in the file;
- DC operating-point and AC analyses declared in the model file — `analysis` accepts
  only `transient` and `harmonics` (periodic steady state and loop gain are available
  separately and experimentally through `pear-sim analyze stability`; parameter sweeps
  through `pear-sim sweep`);
- native import/export of commercial simulator model formats.

Features on this list must not be silently ignored. Adding them requires a schema extension or a new `pear-sim/v0.x` version, plus migration tests.

Time-varying sources (§6.1.1), the closed-loop and signal-driven modulators (§7, §9.3,
§9.6), declared control blocks (§9.3) and table-based loss/thermal data (§7a) are
supported in `v0.1` and are therefore no longer on this list; their maturity
(`supported` / `experimental`) is reported by `pear-sim capabilities --json`.

## 15. Security

- YAML is loaded with a safe loader without custom object constructors.
  A custom tag (e.g. `!mycomp`) is an error.
- **Reusing YAML nodes is forbidden**: an anchor `&name`, an alias `*name` and the
  merge key `<<` end with a `ModelError` pointing to the line, column and remedy.
  The reason is not aesthetic: the parser expands them **before** JSON Schema
  validation and before unknown-field rejection, so `<<: *leg` reaches the loader as
  an ordinary map and no later gate has a chance to object. In addition, the JSON
  surface cannot express reuse — allowing it in YAML would split the two syntaxes into
  two different languages. Reusing a component template is the job of declared
  subcircuits (§6a), where expansion is explicit, namespaced and part of the
  canonical hash.
- JSON/YAML cannot run code or commands.
- `v0.1` does not fetch network resources.
- all numbers must be finite;
- a simulation should be given an external timeout and step/event limits in the orchestration interface; these fields are not yet part of the `v0.1` schema.

## 16. Format evolution

`pear-sim/v0.1` is early-alpha, but every change must update, together:

1. `schema/model.schema.json`;
2. the loader and model types;
3. this document;
4. examples;
5. the YAML/JSON round-trip test;
6. release notes and a migrator, if saved user models exist.

A breaking change after a published release requires a new schema identifier. An optional SPICE backend remains outside `v0.1`; see [architecture.md](architecture.md) and [ADR-0001](adr/0001-system-level-kernel.md).
