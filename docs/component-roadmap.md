# Component roadmap

This roadmap defines the intended product boundary for PEARSim. It is a scope document, not a claim that a listed item is implemented. A component becomes supported only when its schema, equations, diagnostics, tests, and user documentation all meet the acceptance gate below.

## Priority labels

| Label | Meaning |
|---|---|
| MVP | Required for the first useful headless electrical simulator |
| Next | Planned after the MVP contract is stable |
| Later | Valuable extension with a known place in the architecture |
| Deferred | Intentionally excluded until demand, legal review, or a separate backend justifies it |

Implementation status should be reported by release notes and passing tests. The labels here describe priority only.

## Product invariants

Every phase must preserve these properties:

- a model is a versioned, declarative YAML or JSON document;
- validation happens before an expensive simulation starts;
- component equations and sign conventions are public and testable;
- solver choices, tolerances, event policy, and output times are explicit;
- results are machine-readable and include reproducibility metadata;
- a model has the same physical meaning on Linux, macOS, and Windows;
- optional backends never change semantics silently;
- proprietary tools are never runtime dependencies of the reference backend.

## Foundation

| Capability | Priority | Minimum contract |
|---|---:|---|
| Versioned model schema | MVP | YAML and JSON map to one normalized internal model; unknown keys and versions receive actionable errors |
| Semantic validation | MVP | Unique IDs, valid terminals, connected reference node, finite parameters, legal initial conditions, requested-output validation |
| SI numeric convention | MVP | Base values have documented SI meaning; no implicit engineering-unit guessing |
| Network compiler | MVP | Build a deterministic equation/topology representation from the normalized model |
| Reference transient kernel | MVP | Portable Python implementation optimized for clarity and conformance rather than peak speed |
| Event queue | MVP | Stable ordering for coincident scheduled, PWM, and topology events; explicit tie-breaking policy |
| Output contract | MVP | Requested voltages/currents on an explicit time grid plus solver and model metadata |
| Diagnostics | MVP | Stable error/warning codes for invalid topology, singular systems, inconsistent states, and convergence failure |
| Reusable subcircuits | Next | Parameterized hierarchy with explicit ports and deterministic expansion |
| Schema migration | Next | Lossless migration between published schema versions where practical |
| Accelerated kernel | Later | Backend conformance suite before a C++20 or other native kernel can be selected |

## Electrical domain

### Passive and reference elements

| Component | Priority | Required behavior |
|---|---:|---|
| Electrical ground/reference | MVP | Exactly defined zero-potential node; diagnostics for missing or contradictory references |
| Resistor | MVP | \(v=Ri\), positive finite resistance in the first schema |
| Capacitor | MVP | \(i=C\,dv/dt\), explicit or default initial voltage, stored-energy checks |
| Inductor | MVP | \(v=L\,di/dt\), explicit or default initial current, stored-energy checks |
| Series/parallel composites | Next | Convenience expansion into primitives, not separate hidden physics |
| Mutual inductance | Later | Consistent dot convention, coupling bounds, energy-positive parameter validation |
| Frequency-dependent impedance | Later | Explicit rational/state-space representation with passivity diagnostics where applicable |

### Sources and measurements

| Component | Priority | Required behavior |
|---|---:|---|
| Independent DC voltage source | MVP | Constant voltage with a documented terminal polarity |
| Independent DC current source | MVP | Constant current with a documented positive-flow direction |
| Requested node voltage | MVP | Difference between named nodes; ground shorthand is only syntactic sugar |
| Requested branch current | MVP | Direction defined by the referenced component terminals |
| Step and pulse sources | MVP | Exact scheduled discontinuities, initial value, edge ordering, and boundary-sample convention |
| Sine source | Next | Frequency and deterministic amplitude/phase convention |
| Piecewise-linear/table source | Next | Explicit interpolation and extrapolation policy |
| Controlled voltage/current sources | Next | VCVS, VCCS, CCVS, CCCS with algebraic-loop diagnostics |
| Three-phase source helpers | Later | Expansion to independent primitives with explicit sequence and phase convention |

### Switching devices

| Component | Priority | Required behavior |
|---|---:|---|
| Ideal controlled switch | MVP | Closed: zero voltage; open: zero current; deterministic event at the requested instant |
| Ideal diode | MVP | Natural commutation using a documented complementarity/sign convention and event localization |
| PWM gate/event source | MVP | Frequency/period, duty ratio, phase, initial state, and coincident-edge ordering are explicit |
| Complementary PWM with dead time | Next | No overlapping commands; deterministic behavior at zero or excessive dead time |
| Ideal MOSFET/IGBT/thyristor facades | Next | Named convenience components built from documented switch and commutation semantics |
| Breaker/relay | Next | Scheduled or controlled opening/closing with the same ideal-switch core |
| On-resistance/forward-drop switch | Next | Piecewise-affine device with explicit reverse-current behavior |
| Reverse recovery | Later | Independently specified charge-based approximation with dedicated validation |
| Detailed semiconductor compact model | Deferred | Belongs in the optional SPICE backend, not the reference ideal-switch kernel |

### Converter helpers

Converter blocks are conveniences, never opaque solver primitives. Each must expand to documented components and expose its generated topology.

| Helper | Priority |
|---|---:|
| Half bridge and synchronous buck leg | Next |
| Single-phase full bridge | Next |
| Three-phase two-level bridge | Next |
| Diode bridge, single- and three-phase | Next |
| Buck, boost, and buck-boost templates | Next |
| Flyback/forward/resonant templates | Later |
| Multilevel and modular converters | Later |

The inverting-buck-boost boundary producer recognizes one exact ideal source,
switch, inductor, diode, capacitor, and load topology for verification. It is not an
implemented reusable buck-boost helper and does not promote arbitrary buck-boost or
general DCM/DAE support.

## Control and signal domain

The MVP needs only enough control behavior to drive deterministic switching. A broad block-diagram simulator is not a prerequisite for useful power-stage simulation.

| Component/capability | Priority | Notes |
|---|---:|---|
| Constant and external parameter | MVP | Typed scalar values available to PWM and source parameters |
| Scheduled Boolean gate | MVP | Explicit timestamp/state pairs |
| PWM generator | MVP | Event-producing block, not a numerically sampled approximation |
| Gain, sum, product, limiter | Next | Algebraic signal operations with loop detection |
| Comparator with hysteresis | Next | Explicit threshold equality and event policy |
| Zero-order hold and unit delay | Next | Sample time and initial output required |
| Continuous integrator | Next | Initial condition and solver coupling defined |
| Continuous/discrete transfer function | Later | Coefficient ordering, normalization, and state initialization documented |
| PI/PID helpers | Later | Expand to primitive blocks; anti-windup behavior explicit |
| Multi-rate scheduler | Later | Rational-rate checks and deterministic task ordering |
| Hierarchical state machine | Deferred | Separate feature after core control semantics stabilize |
| Arbitrary embedded C/Python block | Deferred | Security, determinism, and portability require a separate design |

## Analysis tools

| Analysis | Priority | Acceptance target |
|---|---:|---|
| Transient | MVP | Explicit output grid, initial state, stop time, diagnostics, and convergence tests |
| Parameter sweep | Next | Isolated runs, deterministic ordering, structured aggregate output |
| Operating-point/DC | Next | Linear networks first; clear unsupported diagnostic for switched periodic state |
| Periodic steady state | Later | Converter benchmarks with convergence and failure reporting |
| Small-signal/AC | Later | Linear circuits first, then an explicitly defined switched-system method |
| Fourier/RMS/THD post-processing | Later | PEARSim implements a bounded opt-in event-aligned Fourier/THD observation with stored window, bandwidth, fundamental, and normalization; general post-processing remains Later |
| Sensitivity/optimization | Deferred | External tools should consume the stable CLI before native optimization is considered |

PEARSim implements a bounded subset of the parameter-sweep target: one existing model
parameter, one to 64 request-ordered points, retained per-point runs and failures,
exact metric projections, and optional expectations evaluation. It is not a
multidimensional batch system, optimizer, ranking engine, or inferred parameter
envelope. Without expectations, execution may pass while acceptance remains
explicitly `not_run`.

## Thermal and loss modeling

Thermal functionality starts only after electrical switching waveforms and event values are trustworthy.

| Component/capability | Priority | Notes |
|---|---:|---|
| Static conduction-loss curve/table | Later | Voltage-drop or power map with units and interpolation policy |
| Switching-energy lookup table | Later | Separate turn-on/turn-off energy versus documented axes such as current, voltage, and temperature |
| Double-pulse-test evaluator | Later | Compare lookup/interpolation behavior against independently licensed data |
| Thermal resistance/capacitance | Later | Lumped thermal network with energy balance |
| Ambient and ideal thermal reference | Later | Explicit absolute-temperature convention |
| Foster thermal impedance | Later | Step-response benchmark and parameter validation |
| Cauer thermal network | Later | Network conversion only when numerically and physically justified |
| Electrothermal iteration | Later | Electrical loss to junction temperature with documented coupling cadence |
| Manufacturer model import | Deferred | Per-format and per-dataset license review required |

The validation plan already reserves cases for conduction and double-pulse losses, Foster impedance, and end-to-end electrothermal behavior. Their presence in the manifest does not mean the components are implemented.

## Magnetic and mechanical domains

These domains are useful for complete drive systems but are outside the initial product claim.

| Domain | Priority | First possible slice |
|---|---:|---|
| Ideal transformer and coupled winding refinements | Later | Electrical-domain primitives with energy tests |
| Lumped magnetic circuit | Deferred | Winding, permeance, air gap, saturation, hysteresis, and core loss require a dedicated formulation |
| Rotational mechanics | Deferred | Inertia, torque source, damper, spring, position/speed sensors |
| Translational mechanics | Deferred | Mass, force source, damper, spring, position/speed sensors |
| Clutch, backlash, friction | Deferred | Hybrid events and collision semantics require a dedicated validation suite |
| FEM/field simulation | Deferred | Not planned as a native PEARSim capability |

## Interoperability and user surfaces

| Capability | Priority | Policy |
|---|---:|---|
| YAML/JSON input | MVP | Canonical schema owned by PEARSim |
| JSON/CSV results | MVP | Stable machine-readable result and table exports |
| Python library API | MVP | Thin programmatic surface using the same normalized model and result objects as the CLI |
| Deterministic schematic renderer | MVP | Clean-room SVG generated from the canonical model; stable component/net IDs and no alternate connectivity |
| Offline human report | MVP | Versioned `report.json` rendered to standalone HTML with KPI, waveforms, events, diagnostics, provenance, and validation coverage |
| Hash-bound review packet | MVP | Pre-run topology decision and post-run result decision stored separately from immutable model/run artifacts |
| Local review UI | Next | Optional localhost-only UI for annotations; CLI/export path remains fully functional without a browser server |
| Agent/MCP adapter | Next | Thin wrapper around the stable CLI/library; no alternate physics path |
| SPICE netlist import/backend | Later | Separate optional dependency and explicit dialect/support report |
| FMI import/export | Later | Standards-based boundary; supported FMI version and mode declared in each artifact |
| Full graphical schematic editor | Deferred | Optional drag-and-drop front end consuming the same model schema; never the only source of model semantics |

## Component acceptance gate

A component is not supported merely because a class or parser entry exists. It needs all of the following:

1. public equation, state, terminal, polarity, and unit documentation;
2. schema validation for legal ranges and dimensions;
3. initialization and discontinuity semantics;
4. at least one analytic or independently computable test;
5. energy, passivity, or conservation checks where physically meaningful;
6. convergence behavior over multiple solver settings;
7. event-boundary tests for switched or sampled components;
8. invalid-topology and invalid-parameter diagnostics;
9. serialization round-trip and deterministic ordering tests;
10. cross-platform CI evidence within documented numerical tolerances.

Composite components additionally need an inspectable expansion into accepted primitives. Optional backends need differential tests against the reference kernel on the intersection of supported behavior.

## Roadmap exit criteria

### MVP exit

- at least one analytic RLC case passes on every supported platform;
- ideal-switch and diode events pass CCM, DCM/BCM, and rectifier-commutation cases within declared tolerances;
- malformed models fail before simulation with stable diagnostics;
- the CLI emits reproducible metadata and requested voltage/current outputs;
- no commercial simulator installation, proprietary native model, or proprietary dataset is required.

### Electrical alpha exit

- buck, boost, and buck-boost cases cover steady ripple and operating-mode boundaries;
- single- and three-phase rectifiers cover natural commutation;
- two-level PWM inverter covers dead time and THD post-processing;
- reference and optional backends publish an explicit compatibility matrix.

### Electrothermal alpha exit

- loss lookup interpolation passes synthetic and independently licensed data cases;
- Foster/Cauer networks pass analytic thermal step responses;
- end-to-end electrical loss and junction-temperature energy balance is documented;
- every external dataset has provenance and redistribution terms recorded.
