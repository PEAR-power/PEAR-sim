# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) (during 0.x, minor versions may
break the model format; every break carries a migration note).

## [Unreleased]

### Fixed

- `gate-drive.json` now follows `forced_gate` events as well as `scheduled_gate` ones.
  Before, a `perturbations.forced_gate_windows` entry that changed a gate (e.g. a forced
  turn-off in the middle of a PWM pulse) left the observer tracking the wrong state, and
  the run failed in `write_result` with `analysis.gate_drive_invalid` (a continuity
  mismatch or a false complementary overlap). Each timeline change now carries
  `source_event_kind`; the summary adds `forced_gate_event_count` and
  `effective_gate_event_count`, and `scheduled_gate_event_count` counts scheduled events
  only. `source_binding.timeline_source` is now
  `accepted_initial_mode_and_scheduled_and_forced_gate_events`. Additive within v0.1: no
  run with a forced gate change could produce the artifact before.

### Added

- `impulsive_events: conserve` now also covers inductor cuts: a switch opening that
  leaves a node with only plain inductors (series pair, star, lone inductor) takes a
  flux-conserving jump, `(L1·i1 + L2·i2)/(L1+L2)`, with the loss booked as
  `commutation_loss_j` (`kind: inductor_cut` in `impulsive_commutations`). Every cut is an
  `IMPULSIVE_INDUCTOR_CUT` warning (an overvoltage across the switch in hardware). Checked
  against flux conservation on paper (1e-12) and against a resistor of 1k/10k/100k ohm
  across the switch on the default engine, whose dissipation reaches the same loss for
  every R (1e-6). `state_discontinuity` findings gain `preferred_configuration` (the
  finding belongs to the diodes' actual configuration, not another search candidate).

- `pear-sim analyze loop-gain` and MCP `measure_loop_gain`: the loop gain of a
  closed-loop model measured by injection on the switching circuit (Middlebrook,
  started on the closed-loop orbit, phasors fitted on the controller's samples),
  with refined phase and gain margins. Verified against Floquet: scaling the
  controller by the measured gain margin puts a multiplier on the unit circle at the
  measured frequency (1.2e-7). Unstable orbits, engaged limiters and injections that
  change the switching mode sequence are refused.

- `analysis.transient.impulsive_events: reject | conserve` (default `reject`, unchanged).
  Under `conserve` a capacitor loop closed by an ideal switch takes a charge-conserving
  jump; the dissipated energy is booked as `commutation_loss_j` in the physics audit
  (every jump listed under `impulsive_commutations`) and the balance closes with it. After
  the jump the loop is reduced (one capacitor per loop carries the loop law's derivative,
  dual of the inductor-star reduction). Checked against charge conservation on paper (two
  capacitors with C1 != C2, onto a DC source, a chain of three: 1e-12) and against the
  ordinary engine with switch on-resistance 1/0.1/0.01 ohm, whose dissipation reaches the
  same loss for every R (1e-6). Loops through diodes or with waveform sources are refused,
  as are steady-state, closed-loop and gradient analyses under the opt-in. New capability
  `electrical.impulsive_commutation_conservation` (experimental); the request hash
  changes only for `conserve`.

- `state_discontinuity` diagnostic (`StateDiscontinuityError`, code
  `model.state_discontinuity`, subclass of `TopologyError`): a switch or diode closing a
  loop of capacitors/voltage sources at unequal voltages, or opening the last path of an
  inductor current, is refused with the loop or cut, the jump (V or A), the instant and the
  triggering event; the CLI `--json` error carries them under `details`. Promised by
  architecture.md §5.2 and ADR-0001, absent until now. The run is still refused (no
  impulsive regularization); a capacitor loop at equal voltages keeps `model.topology`.

- `loss_analysis.tj_mode: self_consistent` — conduction losses iterated to a
  self-consistent junction temperature trajectory on the recorded run, with the
  electrothermal loop gain measured and reported; loop gain ≥ 1 returns
  `thermal_runaway` and no loss number (so does a fixed point that only exists
  because v_on is clamped at the card's temperature-axis edge). Checked against the closed form
  Tj* = (Ta + Rth·I²·R0·(1 − α·T0)) / (1 − Rth·I²·R0·α) at loop gains 0.3, 0.9
  and −0.9 (worst 9e-13 K), runaway at 1.1. On `buck_with_loss_cards.yaml` the
  devices settle at 53–66 °C instead of the declared 125 °C and the total loss
  estimate moves 27.45 → 26.79 W. Default `declared` is unchanged bit for bit.

- Dead-time acceptance matrix for the three-phase SPWM inverter (VAL-005): 24 runs
  (SPWM and min-max x dead time 0/0.5/1 us x damped and ripple-dominated RL x m 0.5/0.9)
  against a new independent oracle, `pear_sim.benchmarks.spwm_dead_time`, which solves
  the midpoint-neutral inverter in closed form and resolves current reversals inside
  dead-time gaps exactly. Currents, v_ab/v_an Fourier coefficients and v_ab THD agree to
  4.3e-12 (currents 2.0e-9, integrator-bound); three wrong oracles miss by at least four
  orders. New capability `converter.inverter_three_phase.dead_time_acceptance_midpoint_neutral`
  (experimental); the composite `spwm_svpwm_dead_time` stays unsupported.
- Closed-loop Floquet in DCM is analytic: the guard saltation
  `S = I + (f+ - f-) gᵀ/(g·f-)` at each natural commutation, so a DCM orbit no longer
  falls back to finite differences alone. New closed-form DCM oracle
  (`tests/reference/buck_closed_loop_floquet_oracle.py`, derived without saltation
  matrices) agrees to 2.7e-13 on the multipliers; dropping the saltation is caught by
  the built-in cross-check. A natural commutation coincident with a gate edge, or a
  grazing one, still uses finite differences and says why.
- Signal block values may reference unit-less parameters (`gain: "${kp}"` with
  `kp: 0.002`), so `--set`, MCP `overrides` and parameter sweeps reach controller gains,
  limits and initial values. Bit-identical to the literal; a parameter with a unit in a
  bare-number field is refused by name. `examples/buck_closed_loop_floquet.yaml` now
  declares `kp`/`ki` as parameters (`--set kp=0.08 --set ki=150` is its unstable design).
- The topology error of a model whose `dcr` hides an inductor-only node (a floating
  star, inductors in series) now names those inductors and the node and says how to
  model the winding resistance instead. Measured on the shipped examples with `dcr` on
  every inductor: the hint appears on all 8 that fail and on none of the 19 that run.
- MCP tool `floquet`: the closed-loop Floquet analysis for agents — verdict, gates,
  largest multipliers with the oscillation frequency, Lyapunov certificate, orbit state,
  violated assumptions and the engine's notes (e.g. "pass `initial_state`" when Newton
  stops at a limiter). `pear-sim analyze floquet` gains `--set NAME=VALUE`, and the
  evidence records the overrides. `steady_state` on a closed-loop model now points to
  `floquet` in its error.
- motulator (Aalto Electric Drives, MIT) as an independent engine for the machine
  models (`pear_sim.oracle.motulator`): a PMSM fed by a balanced sine supply at an
  imposed speed, compared on the same time grid. motulator runs in its own interpreter
  (`PEAR_SIM_MOTULATOR_PYTHON`), never imported by PEARSim; a new CI job installs it.
  Non-salient at constant speed agrees to ~1e-12; salient rotor, speed ramps and an
  unsaturated flux map agree within 2 % at a 10 us step with first-order convergence.
  Example `examples/pmsm_sine_fed_driven.yaml`.
- `core_losses` section: a post-processing core-loss estimate for linear inductors by
  the improved Generalized Steinmetz Equation (iGSE) over the last declared period of
  the recorded current (`B = L·i/(N·A_e)`), with material cards carrying Steinmetz
  ranges, the `ct(T)` temperature factor, mandatory provenance and a validity envelope.
  New artifact `core-losses.json`, summary entry `core_losses`, capability
  `electrothermal.core_loss_igse` (experimental). Validated against measured MagNet N87
  core loss (`tests/test_core_losses_magnet.py`, data subset in `tests/data/`). New
  optional section only: existing models are unchanged.
- Lyapunov certificate for a `stable` closed-loop Floquet verdict: `floquet.json` stores `P`
  with `P` and `P - Mᵀ P M` positive definite (checked by Cholesky, eigenvalue-free, valid
  for both computed monodromies), and `analyze floquet` has a blocking
  `lyapunov_certificate` gate. The certificate must guarantee a per-cycle decrease of `V` of at least 1e-8
  (unit-free), so round-off cannot certify a map on the unit circle. Anyone can re-check it from the bundle with two
  factorisations.

- Closed-loop periodic steady state and Floquet multipliers
  (`pear_sim.analysis.closed_loop.solve_closed_loop_steady_state`, experimental, Python
  API only): Newton shooting on the production cycle map with the state augmented by every
  `unit_delay` register of the signal layer, and the multipliers of the closed loop on the
  switching model — `stable`, `unstable`, `not_resolved` (inside the step-size spread) or
  `not_assessed` (limiter, duty clamp or mode sequence changes). Local stability of the
  orbit only; see `docs/stability-analysis.md`. Capability
  `analysis.steady_state.closed_loop_floquet`.
- `pear-sim analyze floquet MODEL --out DIR [--initial-state NAME=VALUE]`: a retained,
  hash-manifested bundle (`floquet.json`, `floquet-evidence.json` with its blocking
  gates) for the closed-loop analysis; exit 0 locally stable, 1 evidence retained but not
  stable or not attestable, 2 refused. The monodromy is analytic (the orbit's mode matrix
  exponentials plus the controlled switching-instant term) and cross-checked against
  finite differences on every run; a disagreement withholds the verdict. New example
  `examples/buck_closed_loop_floquet.yaml`.

- `esr` on `capacitor` and `dcr` on `inductor` / `sat_inductor`: a constant series
  resistance, expanded into an ordinary resistor (`<id>.esr`, `<id>.dcr`, internal node
  `<id>.core`) and bit-identical to writing that resistor by hand. Capacitor banks and a
  capacitor across a voltage source now run when one capacitor in the loop has an `esr`;
  the topology error for such a loop names the capacitors and suggests it. New optional
  fields only: existing models are unchanged.
- Energy audit of declared parasitics: `parasitic_loss_by_component_w` and
  `parasitic_absorbed_energy_j` per window, and `capacitor_rms_current_a` for every
  capacitor with an `esr`. Optional `ripple_current_rating` on a capacitor (requires
  `esr`): warning `CAPACITOR_RIPPLE_CURRENT_EXCEEDED` above it, and
  `CAPACITOR_RIPPLE_CURRENT_NOT_ASSESSED` on a `--lean` run. `model.resolved.json` lists
  the expansions (`passive_parasitics`). Efficiency maps count declared ESR/DCR losses
  (`declared_parasitic_loss_w`) and refuse to count one twice.

### Changed

- Failures that are state jumps now report `model.state_discontinuity` instead of
  `model.topology` (capacitor loop closed by a switch/diode, single inductor cut).
  Migration: clients branching on `model.topology` should also accept the new code;
  Python callers catching `TopologyError` are unaffected.

- Faster evidence: the dense-output quadrature and the state-extremum search run as
  array operations (per-mode gather plan; one batched root search per segment) instead
  of Python loops over every output, component and accepted step. Results are
  **bit-identical** — pinned against the former loops, kept verbatim in
  `tests/reference/dense_metrics_reference.py`, on every integrator and on whole runs.
  `examples/grid_following_converter.yaml` (60 ms): full evidence 147.6 s -> 48.7 s,
  lean 74.8 s -> 43.1 s; integration is now 57-64 % of the run.

### Fixed

- A switch opening between two inductors with different currents ran SILENTLY with a
  phantom current into the node: the floating-star reduction preserves the KCL residual
  instead of restoring it, and the only trace was an energy-balance miss attributed to
  stiffness. It is now a `state_discontinuity`. The check's bound is the run's own
  atol + rtol*|i| (largest legitimate drift on the 41 examples: 4.6e-9 relative, on
  coupled windings at rtol 1e-7).

- Three-phase SPWM: the dead-time bound the loader (and `lint`) accepts is now the one
  the scheduler can place, `0.25*(1-m*peak)/carrier_frequency`, not the full minimum
  pulse `0.5*(...)`. A model between the two passed `lint` and failed at run start
  ("not representable at the requested absolute time origin"). **Migration:** such
  models never ran; they are now rejected at load time with the bound in the message.

- A saturating `flux_map` no longer diverges in saturation. The machine driver carried
  `L_old*i -> L_new*i` across segment boundaries with the map's DIFFERENTIAL inductances,
  which equals the flux only while the map is linear; in saturation the current ran off
  the map (i_d to -67 A and a refusal) at every step size. The boundary now carries the
  map's true flux and inverts the map for the new currents (Newton on the interpolant's
  exact Jacobian; a flux the map cannot produce is refused), and the back-EMF inside a
  segment comes from the part of the map flux the held matrix does not carry. Against
  motulator the saturated case now converges first order (1.53 % / 0.77 % / 0.39 % of the
  peak at 10 / 5 / 2.5 us). Results of unsaturated maps change by ~1e-10; constant-Ld/Lq
  machines are untouched. The test docstring's 0.38 % for the unsaturated map was a stale
  figure — it measures 0.67 % at 10 us with both the old and the new kernel.

- Periodic steady state (shooting) now refuses closed-loop models by name. Shooting solves
  for circuit state only, so a sampled controller's memory (integrator, unit delays) was
  outside it: a closed-loop model was refused with the misleading "requires at least one
  PWM control", and a model mixing a fixed PWM leg with a sampled loop was shot with the
  controller restarted from its declared memory every cycle — a fixed point of the wrong
  map. Both are now `SteadyStateError` naming the sampled controls.

- Switched AC sweep (`analysis/ac_sweep.correlate`): the correlation window now starts
  exactly at `window_start_s` (endpoints interpolated) and the phase is referenced to
  absolute time. Before, when the window start fell between output samples, the phase
  was measured from the first sample (an error of up to 360·f·`output_step` degrees) and
  the late start dropped a whole perturbation period, letting switching ripple leak into
  the result. Gains and phases can change slightly for sweeps whose window start was off
  the output grid; on the canonical buck at 3 kHz the phase spread across window starts
  falls from 0.15° to 0.001°.

## [0.1.1] - 2026-11-13

Documentation and provenance release; no change to the simulator or its results.

### Changed

- README: install section now points at the published PyPI packages and the MCP Registry
  entry (it still said the first release was pending); PyPI and registry badges.
- Every Python and Rust source file now carries an `SPDX-License-Identifier: Apache-2.0`
  and copyright header, checked by a test; README and AGENTS.md spell out the attribution
  that copying the code requires.
- Product source files under `src/` and `rust/` carry a provenance-mark GUID in their
  header (`pear_sim._provenance`), checked by a test.

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
