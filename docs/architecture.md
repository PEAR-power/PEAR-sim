# PEARSim architecture

Status: baseline design
Date: 2026-07-18
Scope: a headless power-electronics simulator intended primarily for coding agents and CI automation

## 1. Summary of decisions

PEARSim is built as its own system-level simulation kernel, not as a SPICE fork or a thin wrapper around a single existing program.

Key decisions:

- the public contract consists of a versioned model format, the CLI and the Python API;
- the first kernel implementation is a reference implementation written in Python;
- the target production kernel will be written in C++20 without changing the model and result contract;
- the primary computational path uses ideal switches, exact events and switched state-space equations;
- semiconductor losses are computed from conduction and switching-energy tables, without simulating every transistor transient;
- SPICE is an optional backend for detailed models and independent validation, not the core of the product;
- macOS is the first development platform, but the formats and API must work natively on Windows without Docker from the start.

The system-level kernel decision is described in [ADR-0001](adr/0001-system-level-kernel.md), and the cross-platform rules in [ADR-0002](adr/0002-cross-platform.md).

## 2. Goals and boundaries

### 2.1. Goals

PEARSim should enable:

1. creating and modifying models by a human or a coding agent;
2. deterministic execution of simulations without a GUI;
3. fast study of circuits with a large number of switching events;
4. analysis of waveforms, energy balance, losses and temperatures;
5. comparing results between kernel versions and against external solvers;
6. diagnosing an incorrect topology in terms of the model's elements and nodes;
7. presenting to a human an unambiguous schematic, assumptions, KPIs and level of evidence;
8. binding feedback and acceptance to a specific revision of the model and of the result;
9. preserving a single contract during the Python -> C++20 migration.

### 2.2. Scope of the first usable release

- YAML/JSON format and model validation;
- R, L, C elements, reference, basic sources and measurements;
- ideal controlled switch and ideal diode;
- constant and PWM/pulse signals;
- transient analysis with exact event instants;
- recording only the requested waveforms;
- CLI, Python API, run manifest and machine-readable errors;
- deterministic SVG schematic, offline HTML report and a hash-bound review packet;
- analytical tests plus buck, boost and diode-rectifier models.

### 2.3. Outside the first scope

- a full graphical schematic editor;
- a full catalog of SPICE transistor models;
- EMI simulation and nanosecond edges as the default path;
- machines, the full magnetic domain, FMI and real-time code generation;
- executing arbitrary Python or C from a model file.

## 3. Build vs wrap

| Area | Decision | Rationale |
|---|---|---|
| Model format and canonical IR | build | This is the core product contract and must remain stable regardless of the solver. |
| Ideal switching semantics | build | SPICE approximates a switch with a large/small resistance and does not provide the required event semantics. |
| Event queue and discrete-block scheduler | build | Determinism and consistent switching of many elements require kernel control. |
| Sparse linear algebra | wrap | We use proven libraries; we do not implement sparse LU from scratch. |
| ODE/DAE integrators | build/wrap | DOPRI and the event manager belong to the kernel; a DAE/stiff solver may use SUNDIALS. |
| Transistor models and vendor netlists | wrap | Run through an optional ngspice/Xyce backend, preferably as a separate process. |
| Renderer and report | build | The SVG schematic and HTML are a deterministic projection of the canonical model/run, without a separate physics path. |
| Full GUI editor | not now | The renderer, report and review are sufficient for the first agent + human workflow. |

This split makes it possible to deliver a useful API early without making the product's semantics depend on the license or the limitations of a single external program.

## 4. System layers

```text
Model YAML/JSON
      |
      v
Parser -> schema validation -> units/expressions -> semantic validation
      |
      v
Canonical Model / IR
      |
      +--------------------------+
      |                          |
      v                          v
Native system-level backend   Optional SPICE backend
      |                          |
      v                          v
MNA/descriptor compiler       netlist adapter/process
      |
      v
Mode operator cache + event manager + ODE/DAE integrator
      |
      v
RunResult -> manifest + waveforms + event log + metrics + mode coverage
      |
      +--> CLI
      +--> Python API
      +--> MCP adapter
      +--> report model -> schematic SVG + offline HTML + diff
                         -> human review sidecar
```

### 4.1. Public contract

The public logical types are independent of the implementation language:

```text
ModelDocument -> CompiledModel -> SimulationRequest -> RunResult
```

- `ModelDocument` represents a validated user document.
- `CompiledModel` contains the flattened hierarchy, resolved parameters and the block schedule.
- `SimulationRequest` selects the analysis, backend, parameter overrides and resource limits.
- `RunResult` points to the artifacts and reports status, metrics and diagnostics.

The document format is described in [model-format.md](model-format.md).

### 4.2. Parser and validation

The validation order is part of the contract:

1. safe parse of YAML or JSON;
2. validation of structure and schema version;
3. expansion of safe parameter expressions;
4. unit conversion to SI and dimension checks;
5. validation of identifiers, ports, nodes and signals;
6. flattening of the hierarchy;
7. detection of direct-feedthrough loops and topology problems;
8. creation of the canonical IR and the input hash.

Unknown fields are an error, except in the explicitly reserved `extensions`. This way an agent's typo does not become a silent model change.

### 4.3. Canonical IR

The IR stores:

- typed component instances and their resolved SI parameters;
- separate power connections and directed signal connections;
- component equations/stamps;
- indices of continuous and discrete states;
- guard/zero-crossing functions;
- scheduled sample hits;
- definitions of probes and analyses;
- a map from IR elements to locations in the source file.

The IR must not depend on Python classes, C++ pointers or YAML key order. Eventually it may be serialized as versioned JSON or a binary format, but canonical JSON remains the source of the compatibility hash.

## 5. Physics kernel

### 5.1. MNA and reduction to state equations

Components emit sparse stamps into a descriptor system:

\[
E_\sigma \dot w = K_\sigma w + L_\sigma u
\]

where `sigma` is the switch configuration. For a well-posed linear index-1 system the compiler separates the dynamic and algebraic variables and then builds a logically equivalent operator:

\[
\dot x = A_\sigma x + B_\sigma u, \qquad
y = C_\sigma x + D_\sigma u
\]

Small models may materialize the matrices. Large models should keep a sparse factorization of the algebraic block and apply the operator through a sparse solve, to avoid densifying the Schur complement.

Each configuration `sigma` is compiled only once it is visited. The cache stores:

- the symbolic structure;
- the numerical factorization;
- the map of states and probes;
- the guard functions active in the given mode.

The cache is LRU-bounded. The kernel never enumerates all `2^N` configurations up front.

The current Python reference does not yet implement the full descriptor/DAE described above.
The kernel adds only the conservative state-reduction case needed by the canonical buck
DCM: after `i_L=0` is reached, the singular switch-off/diode-off mode may replace the L state
with an algebraic `0 V` branch, but only when KCL independently forces its current to zero.
The identity of the global state is preserved, the value is projected to exactly
zero within an explicit state-constraint tolerance (currently `1e-9 A`) and
reactivated at the next ON. The reduction is a
fallback after a singularity of the ordinary mode; it does not freeze a mode that was already well
conditioned, nor does it accept a non-zero current. General L cutsets, C loops,
multiple constraints and arbitrary DCM topologies still require a proper
descriptor/DAE and may end with `TopologyError`.

### 5.2. Energy-storage states

Capacitor voltages and inductor currents or fluxes are physical continuous states. They remain continuous across an event unless the model explicitly allows an impulse.

After a topology change the kernel:

1. keeps the vector of physical states from the left side of the event;
2. builds the operator of the new mode;
3. solves for the new algebraic variables;
4. checks constraint consistency;
5. reports `state_discontinuity` if the new topology forces a state jump.

Interrupting an inductive current or shorting a charged capacitor is not automatically masked by `gmin`. Optional parasitics must be explicit in the model.

In the approved narrow case of the canonical buck DCM, the natural turn-off of the diode locates
the current zero, stabilization selects the idle mode with a constrained inductor, and the next PWM edge
reactivates the state. This is evidence for one topology and one transition mechanism, not a
general claim of DCM/BCM support.

### 5.3. Switch Manager

Events have two sources:

- scheduled: PWM, sample time, PWL breakpoint;
- detected: a guard function crossing zero.

Event algorithm:

1. integration stops exactly at the earliest event;
2. the kernel records the `pre` values needed, among others, by the loss model;
3. all explicitly controlled switches for that timestamp change state atomically;
4. the equations and conditions of naturally commutated elements are recomputed;
5. diodes and other elements may change state in zero time;
6. steps 4-5 repeat until stabilization;
7. a repeated configuration indicates a cycle and ends the step with a controlled error;
8. the `post` values are recorded, after which integration resumes.

Tie-breaking order is deterministic: event type, then the lexicographic element identifier. The implementation must not depend on Python dictionary order or object addresses.

### 5.4. Integrators

Planned hierarchy:

- a reference DOPRI5 with step adaptation and dense output for non-stiff systems;
- 5th-order Radau IIA for stiff systems;
- DAE/BDF via SUNDIALS IDA as a fallback for models that cannot be safely reduced;
- fixed-step as a later, separate path intended for real-time and code generation.

Breakpoints known in advance are hard step boundaries. State-dependent events are located with bracketed bisection/Brent. After a model change a multistep DAE solver must be reinitialized with consistent initial conditions.

### 5.5. Control blocks

Direct-feedthrough blocks are sorted topologically. A strongly connected component indicates an algebraic loop and is solved with Newton's method with residual scaling.

Sample times are scheduled on a common integer time base, not by repeatedly adding `float` values. This keeps PWM and controllers synchronized over a long simulation.

## 6. Thermal model and losses

The system-level semiconductor model keeps ideal electrical switching and computes losses independently:

- conduction: `Pcond = f(i, Tj)` or `v_on(i, Tj) * i`;
- turn-on: `Eon = f(i_pre, v_block, Tj)`;
- turn-off: `Eoff = f(i_pre, v_block, Tj)`;
- diode recovery: `Erec = f(i_pre, v_block, Tj)`.

Tables are interpolated multilinearly with an explicit extrapolation policy. The event energy goes into a thermal accumulator, and a thermal R/C network integrates case and junction temperature. Every loss report states the table source, the input range and the number of extrapolations.

The thermal model must not affect the electrical side unless a component declares temperature coupling. Coupling is evaluated at controlled points so as not to introduce a hidden algebraic loop.

## 7. Optional SPICE

SPICE runs as a process adapter:

1. the supported IR subset is exported to a netlist;
2. the backend runs the configured executable without a system shell;
3. the log, exit code and raw result are kept as artifacts;
4. waveforms are normalized to the `RunResult` contract;
5. unsupported ideal semantics are reported, not silently approximated.

Ngspice and Xyce are primarily:

- an oracle for RLC, nonlinear models and circuits with explicit parasitics;
- a backend for vendor netlists;
- a differential-testing tool.

Embedded ngspice linking is not part of the initial plan because of heterogeneous licenses and tighter ABI coupling.

## 8. User interfaces

### 8.1. CLI

The CLI is the automation contract and should work without an active server. The current `v0.1` contract covers:

```text
pear-sim catalog --json
pear-sim lint model.yaml --json
pear-sim simulate model.yaml --out run/ --json
```

Once the transient is stabilized, separate commands are planned:

```text
pear-sim compile model.yaml --out compiled/
pear-sim measure run/ --metric mean:vout --window 4ms:5ms
pear-sim compare run-a/ run-b/ --profile tolerances.yaml
```

In `--json` mode stdout contains a single machine-readable document. Logs go to stderr. The interface does not open windows or a browser.

### 8.2. Python

The target public Python API is a thin layer over the same contract. The ergonomic facade below is a planned contract; the early alpha currently exposes only the lower-level `load_model` and `HybridSimulator`:

```python
model = pear_sim.load("buck.yaml")
result = pear_sim.simulate(model, analysis="tran", overrides={"duty": 0.45})
summary = result.measure("mean", probe="vout", window=(4e-3, 5e-3))
```

After the kernel migration, the call may switch from the Python implementation to a C++ binding without changing the model or the logical result types.

### 8.3. MCP

MCP is an optional layer over the CLI/API. It returns summaries and artifact paths, not thousands of samples in the model's context. The current adapter (`pear-sim mcp`) and its tool set are described in [agent-interface.md](agent-interface.md). The agent is not given `approve`, `reject` or `revoke` tools.

### 8.4. Presentation and review pipeline

The presentation pipeline reads the model snapshot and a completed `RunResult`, but never
runs the solver or modifies physics artifacts:

```text
model snapshot + run artifacts
        |
        v
PresentationLoader -> pear-sim/report/v0.1
        |                    |             |
        v                    v             v
schematic.svg             KPI records   diff.json
        \____________________|_____________/
                             v
                     standalone report.html
                             |
                             v
                 human review sidecar/ledger
```

`report.json` is the only source of the HTML content. KPIs are computed on the full data before
rendering; waveform decimation affects only the plot and keeps each display bucket's
minimum and maximum, so a switching waveform is never drawn as a staircase that is not in
the data. Such an overview of a long switched run is a filled band, so under it the report
draws one shared window of 200 sample steps undecimated, placed in the second half of the
run where the first plotted signal varies most (on a PWM node: duty near one half). The
schematic layout is a sidecar and does not enter the physics hash.

Bridge converters (a DC link of series sources, two or more phase legs, optional NPC clamp
columns, one output per leg) are drawn in the textbook layout of `render_bridge.py`: link
column on the left, rails along the top and bottom, one column per leg, loads hanging below,
a hop at every crossing and voltage probes listed with the quantity they measure. Recognition
is strict and the result is checked geometrically; any other circuit keeps the generic
potential-ordered layout. The embedded metadata says which layout drew the figure
(`"layout": "bridge"`), and `tests/test_render.py` rebuilds the connectivity from the drawn
geometry and compares it with the netlist. A review is bound to the model/request digest
before the simulation and to the run digest after the simulation.

`mode-coverage.json` is a separate producer artifact, not a KPI computed by the
PresentationLoader. For an exactly recognized canonical buck it records, within an explicit window
covering the last half of the transient, the accepted hybrid intervals, the ON/freewheel/idle states, natural
diode turn-off, conduction fraction, minimum `i_L` and event timing relative to the PWM cycle.
For other topologies it has
`not_applicable` and `null` observation fields; the report hashes and shows the data without
reconstructing it from the event log or recomputing statuses. The artifact's claim is
`verification-coverage-observation`, `validation_claim=false`.

The validation gates are described in [validation-plan.md](validation-plan.md).

## 9. Artifacts and reproducibility

A current `v0.1` run produces:

- `run.json`: the input file hash, version, solver settings and statistics;
- `model.resolved.json`: a portable SI snapshot, conventions and IC provenance;
- `waveforms.csv`: the requested voltages and currents on an explicit output grid;
- `events.jsonl`: events and mode changes;
- `metrics.json`: continuous-time mean/RMS and explicit output-grid statistics;
- `analytic-checks.json`: conservative self-checks of recognized topologies;
- `physics-audit.json`: component energy integrals, storage and the balance residual;
- `mode-coverage.json`: observation of ON/freewheel/idle modes, natural commutation,
  conduction fraction, the L current constraint and event timing, only for exactly
  recognized canonical buck/boost/inverting buck-boost;
- `warnings.json`, `summary.json` and the artifact hash manifest.

A columnar `waveforms.arrow` and an external backend's `stderr.log` are planned artifact extensions.

The target manifest will also record the canonical model hash, platform, backend, seed and resource hashes. A content-addressed cache is planned after the canonicalization described in the format contract is implemented.

## 10. Validation

We do not use a single program as the absolute truth. Validation has four layers:

1. **Analytical:** RC, RL, RLC, LC resonance, KCL/KVL, energy balance and exact PWL/PWM instants.
2. **Topological:** continuity of `v_C` and `i_L`, singularities, Switch Manager cycles, simultaneous events.
3. **Differential:** ngspice/Xyce/Siconos/OpenModelica.
4. **Application-level:** buck CCM/DCM, boost, rectifier, half/full bridge, three-phase inverter and DAB.

We do not compare raw sample indices between solvers. We compare:

- event times and `pre/post` states;
- waveforms after interpolation onto a common grid;
- mean, RMS, ripple, peak, energy and harmonics;
- KCL/DAE residual and energy-balance error;
- convergence as tolerances are tightened.

The release report publishes a `component x analysis x solver x validation source` matrix. The term "validated" refers only to cells of that matrix.

The current evidence does not close this matrix: the independent oracle and the
canonical buck DCM regressions cover zero-current reduction, natural turn-off and the absence of
observed reverse current in that case, so VAL-003 is `partial`. A full pass
still requires a sweep across CCM/BCM/DCM, boost, a separate diode case, all
event-time/conduction/energy gates and convergence studies. This implies no claim
of compatibility with any third-party tool, nor physical validation.

## 11. Python -> C++20 migration

### Stage A: Python reference

- format validation and canonicalization;
- readable stamps and small matrices;
- a deterministic event queue;
- analytical tests and golden files;
- profiling of real models.

The priority is correctness and observability, not maximum performance.

### Stage B: stabilizing the boundary

- freezing `ModelDocument`, `SimulationRequest`, `RunResult` v1;
- a corpus of models and reference results;
- a CLI/API compatibility test on macOS and Windows;
- an explicit budget for numerical differences.

### Stage C: C++20 kernel

- port of the IR, stamping, mode cache, integrators and event manager;
- Eigen in an MPL-2.0-compatible mode or an equivalent sparse library;
- SUNDIALS as an optional DAE dependency;
- a Python binding and the same contract tests;
- a single reference mode for deterministic regressions.

Python remains the orchestration, analysis and ergonomics layer. C++ takes over the integration loop, stamping and factorizations.

## 12. Cross-platform

- no core feature requires Docker;
- paths are handled by platform libraries, not concatenated with `/`;
- external processes are launched with an argument list, without `sh -c`/`cmd.exe`;
- text formats are UTF-8 with line-ending normalization;
- simulation times do not depend on the wall clock;
- artifacts use names portable across APFS, NTFS and typical Linux file systems;
- an optional backend is detected explicitly, and a missing executable yields a `backend_unavailable` error.

Packaging and CI details are in [ADR-0002](adr/0002-cross-platform.md).

## 13. Security and limits

- no arbitrary code in the model;
- relative resources must not escape the project root;
- limits on time, number of steps, events, cached modes and output size;
- a YAML parser without custom tags or object constructors;
- an external backend runs in a separate working directory;
- all errors contain a code, the model location and the component context;
- future custom components should use a WASM sandbox, not unconditionally loaded user libraries.

## 14. Roadmap

The stages below are a technical summary.

### R0 — contract and harness

- schema v0.1, CLI/API skeleton, run manifest;
- analytical models and an adapter for one oracle;
- exit criterion: the same YAML model yields the same canonical hash on macOS and Windows.

### R1 — linear kernel

- R/L/C, sources, MNA, transient without switching;
- exit criterion: analytical tolerances and KCL residual.

### R2 — ideal switching

- switch, diode, PWM, root finding, mode cache, diagnostics;
- exit criterion: buck/boost/rectifier and state continuity.

### R3 — control and thermal

- controller blocks, discrete scheduler, loss LUTs, thermal RC network;
- exit criterion: closed loop and loss/temperature balance.

### R4 — production C++20

- port of the hot path, sparse backend, Python wheels and a native CLI;
- exit criterion: corpus compatibility and a significant speed-up without metric regressions.

### R5 — extensions

- SPICE hybrid, periodic steady state, AC analysis, machines, magnetics and FMI.

## 15. Main risks

| Risk | Consequence | Mitigation |
|---|---|---|
| singular or variable-index topology | no solution or a state jump | rank checks, a readable minimal cycle/cutset, explicit parasitics |
| chatter/Zeno | a very large number of events | detection, a micro-iteration limit, optional hysteresis/dwell time |
| mode explosion | memory and compile time | lazy LRU cache, no full enumeration |
| control-plant algebraic loops | no convergence | SCC detection, scaled Newton, initial guesses |
| stiffness and poor scaling | small steps or spurious events | nominal scales, Radau/IDA, residual report |
| different solver results | brittle golden tests | physical metrics, common interpolation and a tolerance budget |
| model and solver licenses | inability to distribute | separate processes, SBOM/SPDX, permissive core, audit before vendoring |
| scope too broad | no usable release | acceptance gates on specific topologies |
| large waveforms | I/O cost and agent context cost | record-on-request, Arrow, decimation and tool-side metrics |

## 16. Technical sources

- SUNDIALS IDA solves `F(t,y,y')=0` with BDF and provides sparse/Krylov solvers and root finding: [LLNL IDA](https://computing.llnl.gov/projects/sundials/ida). SUNDIALS is licensed under BSD-3-Clause.
- Siconos supports nonsmooth systems as well as ideal and piecewise-linear switched circuits: [Siconos documentation](https://nonsmooth.gricad-pages.univ-grenoble-alpes.fr/siconos/), Apache-2.0.
- Eigen provides SparseLU/SparseQR and is licensed under MPL-2.0: [Eigen](https://libeigen.gitlab.io/).
- Ngspice has a heterogeneous set of licenses and therefore requires a separate audit: [ngspice COPYING](https://raw.githubusercontent.com/ngspice/ngspice/master/COPYING).
