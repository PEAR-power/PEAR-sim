# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) (during 0.x, minor versions may
break the model format; every break carries a migration note).

## [Unreleased]

## [0.1.0] - 2026-11-09

First public release of PEARSim, published by PEAR Power.

### Simulator

- Event-driven piecewise-LTI kernel (modified nodal analysis with an ideal-diode
  active set): scheduled gate edges and natural commutations are located as
  exact events; capacitor voltages and inductor currents stay continuous.
- DC-DC (buck, boost, inverting buck-boost in CCM/BCM/DCM), DAB, half-bridge
  LLC, half-wave / 6-pulse / 12-pulse and phase-controlled rectifiers,
  two-level three-phase inverters (six-step, SPWM, SVPWM, floating neutral,
  weak grid, grid-following with PLL), three- and five-level NPC inverters,
  PMSM, synchronous reluctance and DC machines, coupled windings and a
  saturable inductor.
- Control: fixed and complementary PWM with dead time, three-phase and NPC
  modulators, sampled PI and dq current loops, and a signal-block layer
  (gain, sum, product, saturation, unit delay, lookup tables, Park/inverse
  Park, logic) for writing controllers as data.
- Post-processed losses and thermal: device cards, Foster networks, air and
  liquid cooling, efficiency map, rainflow counting.

- Piecewise-linear conduction: `on_resistance` on `ideal_switch` and `diode`,
  `forward_voltage` on `diode` (v = V_f + R_on·i in the circuit equations; a blocking
  diode turns on at V_f). Checked against closed forms; ideal devices keep their hashes.
- `pi` signal block: a discrete PI controller with optional limits and back-calculation
  anti-windup, expanded at load time into primitive blocks.
- Clearer errors: an unknown component type lists the valid ones (and how to model a
  MOSFET); a node left without a conduction path next to a switch suggests the missing
  antiparallel diode.

### Analyses

- Periodic steady state by Newton shooting (cycle averages exact for every state);
  small-signal loop gain with phase/gain margins for a declared compensator,
  including the ZOH and computation-delay term, for one PWM leg with or without
  dead time; a gain margin is reported unbounded when the sweep covers the
  validity band and the phase never reaches −180°; modal resonance audit; event-aligned Fourier/THD;
  bounded parameter and tolerance-corner sweeps; run comparison.

### Fixed before release (found by running 20 user stories through the MCP server)

- Loop gain: margins were taken at the FIRST 0 dB / −180° crossing; a loop crossing
  0 dB twice could report ~150° instead of the binding ~53°. Margins are now the worst
  over every crossing, and all crossings are recorded.
- A parameter sweep failed on any model with a thyristor (the schematic renderer did
  not know the symbol).
- The buck analytic self-check failed correct circuits whose run had not settled; it
  now requires periodic steady state and otherwise reports `not_run` with the reason.
- EMI and compliance are declared unsupported in the capability registry and the
  server instructions.
- MCP: tolerance corners pass on warnings and flag corners that had not settled;
  `simulate` reports whether the run reached steady state.
- A run whose energy balance misses its numerical budget now carries a warning that
  names the likely cause (a stiff circuit on an explicit solver) and the remedy.
- The flyback library entry explains the bleeder-resistor condition under which a
  user model runs, instead of claiming it cannot run.

### Fixed before release (second round of 20 user stories)

- `losses-thermal.json` publishes each device's mean switching power, its total mean
  power and a min/max envelope of its junction temperature, so the report and the MCP
  server draw them instead of dividing energies themselves.
- Report waveforms were thinned by a fixed stride: on a 20 kHz PWM recorded at 1 us
  and drawn at 800 points it picked about one sample per carrier period and drew a
  clean multilevel staircase that is not in the data. The display now keeps each
  bucket's minimum and maximum, so every pulse stays visible. Because that overview of
  a long switched run is a filled band, each main waveform now also gets a zoom: 200
  sample steps drawn undecimated, shaded on the overview, shared by all signals and
  placed where the first signal switches with a duty near one half.
- Junction temperature no longer starts the junction-to-case Foster stages cold under a
  case already held at its steady temperature. A slow stage (100 ms against a 40 ms run)
  read as headroom: on the 250 kW five-level example the hottest switch was reported at
  92.5 °C instead of 103.1 °C. The stages now start at the steady rise for the run-mean
  power (`thermal.initial_state`).
- A heatsink bound without a fan must declare a 0 m³/h row: the estimate reads that
  natural-convection point, and a table starting at forced air was read as one.
- The report labels analytic-check values read from the run "Simulated", not
  "Measured"; the NPC modulator warning no longer says "three-level" on a five-level
  run; the ideal-device note says where device-card losses come from.
- A transient run is limited to 2 000 001 output samples, checked at load. Before,
  2 s at a 10 ns output step allocated a 1.6 GB time grid and kept growing until the
  machine ran out of memory, long before any time limit stopped it.
- MCP `simulate` and `steady_state` return the confidence level together with
  what it means and which features have no independent check, so `not_run` is no
  longer read as a failed run.
- An override of a parameter that no value references as `${name}` is refused with
  the reason; it used to be accepted and change nothing. The FOC, flux-map and
  signal-block buck examples no longer declare parameters their signal blocks do
  not use.
- Energy audit with a salient machine or flux map: the reluctance power
  1/2 i^T (dL/dt) i is not represented, so the balance is now `not_assessed` with
  that reason instead of `outside_numerical_tolerance` with a stiff-solver hint.
- Machine back-EMF branches are no longer counted as supplies; the audit and the
  MCP power summary report the machines' electromagnetic power separately, and
  `efficiency_output` accepts a machine id (drive efficiency).
- Harmonics: a `generic` semantic analyses any recorded signal, e.g. a rectifier's
  DC-side ripple (amplitudes only, no THD).
- MCP: `get_artifact` reads line artifacts (events, control and signal traces) by
  line range; a relative `model_path` on a server without a project directory
  explains how to fix it.

### Design exploration

- `pear-sim batch`, `api.evaluate_designs` and the MCP tool `evaluate_designs`: screen
  up to 2 000 designs of one model (Latin hypercube, grid or an explicit list), rank
  feasible designs by one objective under any number of constraints (signal metrics,
  device losses, junction temperature), and re-run the top finalists with full
  evidence. A finalist must reproduce its screening values exactly or the batch fails.
  Screening runs are lean and keep only the files the ranking reads. The result is the
  best design of the sample, not an optimum. See `docs/batch.md`.
- `pear-sim sensitivity`, `api.sensitivity` and the MCP tool `sensitivity`: global
  sensitivity over declared parameter ranges. Morris screening (`mu_star`, `sigma`) or
  Sobol indices (first-order `S1`, total `ST`, Saltelli design), every index with a
  bootstrap 95 % interval. The estimators are checked against closed forms (a linear
  function, the Ishigami function) and the ideal buck law. See `docs/sensitivity.md`.

### Performance

- Recorded output samples solve the circuit's algebraic unknowns once per sample
  instead of once per output, and the mode feasibility check reuses the vector it
  already holds for every diode guard. Same arithmetic, fewer evaluations: every
  example bundle is identical; buck ~20 % and NPC ~8 % less CPU time.
- Diode mode selection replays its search for a repeated switch configuration: which
  candidates it visits and what each compiles to depend only on the gates and the
  diode guess, so only the state-dependent feasibility check runs again. The
  topology key is also built once per search instead of once per candidate. Five-
  level NPC over three line periods: mode selection 2.9 → 1.3 s, whole run 7.0 →
  5.1 s CPU. Every example's bundle is unchanged except wall time
  (`tests/test_mode_plans.py` holds replayed and fresh searches to exact equality).
- Optional compiled kernel `rust/pear-sim-core` (Rust, PyO3) and `simulate --backend
  rust`. Inside each segment the kernel integrates (a step-for-step port of SciPy's
  RK45 with terminal events), samples every output, integrates the metrics and the
  energy audit, and finds the state extrema. Every covered example agrees with the
  reference engine within 1e-12 on metrics, samples, energy audit and extrema, with
  identical event sequences (CI job `rust-core`). Simulation phase 1.2–25× faster
  (buck 7×, NPC 2–2.6×). Covers `dopri5` models with DC sources; anything else is
  refused with the reason. A run records its backend, which is part of `run_sha256`.
  See `docs/rust-core.md`.
- Long-lived point workers for `batch`, `sensitivity`, `sweep parameter` and
  `tolerance-corners`: each worker imports Python and PEARSim once and then runs many
  points, instead of starting a fresh interpreter per point. Each point is still a
  killable child process with its own captured stderr, warnings, working directory and
  timeout; a worker that overruns or dies is replaced. Retained bundles are identical
  to one-shot workers apart from measured wall times (`tests/test_worker_pool.py`).
  Small models: 950 → 199 ms per point; the 40-design loss-card batch example
  41 → 24 s on 7 jobs. Workers are recycled after 200 points.
- `batch` and `sensitivity` take `"backend": "rust"` in the request: every run uses the
  compiled kernel, checked against the model before anything runs. Runs are accepted
  only from the exact requested engine id; finalists re-run on the same engine. The
  loss-card batch example ranks identically to the reference engine (values within
  2e-12 relative) in 15 s instead of 24 s.
- The diode active-set search skips candidates already known to leave a node floating
  instead of compiling them (multilevel bridges: about 7× fewer mode compilations,
  five-level NPC 1.7× faster), and dense-state extrema use raw coefficient arrays.
  Both are bit-identical: every example bundle matches the unoptimized code file for file.

### Evidence

- Closed-form oracles written independently of the solver, an ngspice
  differential adapter over the linear subset, per-run physics invariants, and
  a fail-closed capability registry (`pear-sim capabilities --json`).
- Continuous integration on Linux (x86 and ARM), macOS and Windows: the ngspice
  differential tests on every platform; the built wheel and sdist of `pear-sim`
  and `pear-sim-mcp` installed with `uvx` outside the repository and used (CLI
  run, MCP stdio session); the canonical-buck numerical evidence matrix every
  night with its bundles kept as artifacts; and a CPU-time benchmark of each
  pull request against its base, reported in shadow mode (it never blocks;
  `scripts/perf_cpu.py`).

### Interfaces

- CLI `pear-sim` and a typed Python API `pear_sim.api`.
- Deterministic SVG schematics. Bridge converters (a DC link of series sources, two or
  more phase legs, NPC clamp columns, one output per leg) are drawn in the textbook
  layout: link column, rails along the top and bottom, one column per leg, loads below,
  a hop at every crossing, voltage probes listed with what they measure. Other circuits
  use a potential-ordered layout. A test rebuilds connectivity from the drawn geometry
  and compares it with the netlist.
- Offline HTML report with waveforms, analytic checks, energy audit and, for models
  with device cards, a losses and junction-temperature section (loss per device split
  into conduction and switching, junction-temperature envelopes of the hottest devices,
  full table), all copied from the run's artifacts. The report has a table of
  contents, puts the schematic, waveforms and losses first, shows warnings and model
  notes as text with a severity mark, collapses long material (device currents, every
  metric, per-component energies, checks that do not apply to the model) and reads
  on a phone: tables scroll inside their frame or become cards instead of breaking
  identifiers mid-word.
- MCP server (`pear-sim mcp`, optional extra `pear-sim[mcp]`, official MCP Python SDK):
  19 tools (examples, format reference, validation, schematic, simulation, steady state,
  loop gain, sweeps, design batches, sensitivity, selective result reads), resources and prompts. Models inline or
  from the project; reads confined to the project and workspace, writes to the
  workspace; computations isolated in a child process with a time limit; actionable
  error hints. Runs are bit-identical to the CLI. The `pear-sim-mcp` launcher package
  (`uvx pear-sim-mcp`) installs the server with its extra; `server.json` describes it
  for the MCP registry.
