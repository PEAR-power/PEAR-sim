# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) (during 0.x, minor versions may
break the model format; every break carries a migration note).

## [Unreleased]

### Changed

- **Sampled extrema no longer pass a requirement the output grid could not see.**
  `min`, `max`, `peak_to_peak` and `ripple` (and their `_last_half` / `sample_` forms)
  are statistics of the requested output grid. On `examples/buck.yaml` (20 kHz, inductor
  current peaking at 4.60 A) recorded once per PWM period (`output_step: 50 us`), the
  grid reads 3.40 A, and `i(l1) max_last_half <= 4.1 A` was accepted by `evaluate`, by
  `sweep perturbation` (margin +0.70 A) and by `batch` (one feasible design, ranked
  first). Every signal record in `metrics.json` now carries `extrema_grid`
  (`output_step_s`, `switching_control_count`, `shortest_switching_period_s`,
  `samples_per_shortest_switching_period`), and one rule
  (`pear_sim.evidence.metric_fitness`) is applied on all three paths: with fewer than 10
  output samples per shortest declared switching period, a sampled extremum decides a
  bound only when its one-sided error cannot change the verdict (a sampled maximum is a
  lower bound on the true one, so a value already above an upper limit still fails).
  Otherwise `evaluate` returns `not_comparable`, the campaign point gets
  `not_comparable` with no margin (run `acceptance_status: not_comparable`, campaign
  not `pass`, `boundary_search` verdict `not_comparable_at_bracket_low` when the
  bracket start cannot be judged), and the batch design is listed under
  `not_comparable`, is not feasible and is not ranked (`not_comparable_count` in the
  summary, a `not_comparable` column in `designs.csv`). At `output_step: 1 us` all
  three paths report the violation. Models without a switching control are unaffected.
  The floor rules out aliasing against the carrier; it does not make a sampled extremum
  exact. `sensitivity` applies the same rule to its outputs: if any run cannot supply
  a sampled extremum, that output is `not_estimated` (`not_comparable_runs`, summary
  `incomplete`) instead of being estimated on aliased values or on the subset of runs
  the grid happened to resolve; other outputs of the request are estimated as before.

  Samples per period do not see a pulse narrower than the grid step. With 4 % duty at
  20 kHz (the switch conducting from 1 µs to 3 µs) and `output_step: 5 us` there are
  exactly 10 samples per period and none inside the pulse: `i(q1)` sampled as 0 A and
  `max i(q1) <= 0.1 A` passed, against 3.46 A at `0.25 us`. `extrema_grid.event_edge`
  now records each electrical output's `min`/`max` (full window and last half) over the
  end points of the solver's integration intervals, on both sides of every event, and
  the rule compares the limit with the sampled value tightened by them. Both grids now
  fail with 3.48 A, and the buck above fails `<= 4.1 A` at `50 us` with the same 4.60 A
  as at `1 us`. Verdicts report `value` (the bound used) and `sampled_value` when they
  differ; the metric fields stay grid statistics. An extremum strictly inside an
  interval between two events is still seen only by the samples.

  A peak-to-peak or ripple is tightened to the span of all observations: the highest of
  samples and event edges minus the lowest of both. With the maximum seen at an edge
  (+1.000 V) and the minimum in a sample (−0.9921 V) the waveform spans at least
  1.9921 V, and a 1.989 V limit is violated although the sampled span is 1.9842 V.
  An event exactly at the start of the last-half window enters that window with its
  state after the switching, like the sample recorded there: a switch that turns off at
  the window start no longer contributes its pre-turn-off current to
  `event_edge.max_last_half`.

  Three more cases the observations got wrong:

  - An interval end point reads its sources on that interval's own side of the instant.
    With a supply stepping from 0 to 100 V at the instant a series switch to a 10 Ω load
    opens, the closing interval paired the old topology with the new source value and
    recorded 10 A that never flowed: `max <= 5 A` failed and `max >= 5 A` passed (campaign
    margin +5 A, `batch` winner). The edge now reads 0 A. The side is the source's exact
    left or right limit, not a read a time tolerance away: a source that is continuous at
    the instant is read exactly there, so a PWL apex declared at a node (100 V on a
    100 V/ns edge) is recorded as 100 V whatever the run length.
  - A periodic source declares a period the grid must resolve, like a switching control.
    A 10 V, 1 kHz sine recorded every 1 ms samples one phase and reads 0 V; with no PWM
    control the grid counted as resolved and `max <= 5 V` passed. `extrema_grid` adds
    `periodic_source_count`, `shortest_source_period_s` and
    `samples_per_shortest_source_period` (sine and pulse sources), held to the same
    10-sample floor: `not_comparable` at 1 ms, `fail` with 10 V at 25 µs.
  - The start of the last-half window is observed when it falls inside a solver
    interval. A capacitor discharging from 10 V with a 1 ms time constant has its
    1–2 ms maximum, 3.679 V, exactly at 1 ms; on a 600 µs grid the first sample in the
    window read 3.012 V and `max_last_half <= 3.3 V` passed. Both grids now fail.

  `batch` reads every sampled extremum as that tightened value: `values` in the
  summary, the ranking, `designs.csv` and the finalist check all use it, and
  `sampled_values` keeps the raw grid statistic where the two differ. Minimizing
  `max i(q1)` over `vin` = 120 V and 80 V with the pulse between samples used to tie at
  0 A and pick 120 V by id; it now ranks 80 V (3.44 A) ahead of 120 V (3.52 A).

  `boundary_search` reports a number only between a comparable `pass` and a comparable
  `fail`. It used to count a probe it could not judge as a failure: searching `fsw`
  from 20 to 200 kHz against a 1000 A limit, where no probe can fail, returned
  `bisected` at 98.75 kHz with a +393.75 % margin, which is where the grid stops
  resolving the period. New verdicts `not_comparable_at_bracket_high` and
  `not_comparable_inside_bracket` leave `max_passing_value_si` and
  `max_passing_drift_percent` null; `largest_passing_probe_si`,
  `smallest_failing_probe_si` and `probe_status_counts` state what was observed.
  A probe that did not run (model refused, worker failed or timed out) has status
  `error`, not `fail`: searching `duty` from 40 % to 120 % against a limit no run can
  reach returned `bisected` with a +150 % margin because the probes above 100 % were
  refused by the PWM validator. The verdict is now `execution_error_at_bracket_high`
  (likewise `_at_bracket_low`, `_inside_bracket`, and `not_run_...` for an absent
  metric), with `undecided_probe` and `undecided_reason`. A campaign point whose only
  shortfall is an absent metric has `acceptance_status: not_run` instead of `fail`.
  The HTML report follows the JSON: it no longer says that no value in the bracket met
  the criteria when the search merely stopped at an undecided probe, and it states the
  probes actually run and the bisection steps actually taken.

  **Migration.** Bundles written by 0.1.x have no `extrema_grid`. `report` reads them
  as before and `evaluate` still decides means, RMS and `final`, but a sampled-extremum
  criterion that the old value does not already violate is `not_comparable`:
  re-simulate the model to judge it. A model whose `analysis.transient.output_step` gives
  fewer than 10 samples per switching period needs a finer step for such criteria.

- **Campaign criteria take their dimension from the measured signal**
  (`pear-sim sweep perturbation`). `unit` was a display label beside a bare number, so
  `signal: v(out), field: mean_last_half, op: max, limit: 50, unit: A` passed with
  margin +10 against a 40 V output. The unit now follows from `signal` and `field`
  (`fsw_measured_hz` is hertz). `limit` and `reference` are a number in that SI unit or
  a unit-bearing string of the same dimension (`"50 mV"`); `limit: "50 mV"` and
  `limit: 0.05, unit: V` give the same criterion and the same margin. The summary
  always records the SI unit.

  **Migration.** These requests are now refused before any run: a `unit` of another
  dimension (`v(out)` in `A`), a scaled `unit` beside a bare number (`limit: 50,
  unit: mV` — write `limit: "50 mV"`), a `field` other than `min`, `max`,
  `mean_last_half`, `min_last_half`, `max_last_half` or `fsw_measured_hz` (it used to
  give `not_run` at every point), and a `signal` that is not a signal expression.
  A repeated `criteria[].id` is refused as well: verdicts are keyed by id, so the later
  criterion replaced the earlier one and `v(out) <= 1 V` plus `v(out) <= 100 V` under one
  id passed or failed a 40 V output depending on their order.
  Requests with a matching or absent `unit` are unchanged.

### Fixed

- Core loss (iGSE): with `1/period` in a **gap between two Steinmetz ranges** of a
  material card, the last range was used whatever its distance, and the flag reported
  the distance to that far range. With ranges 25–99 kHz and 200–1000 kHz and a 100 kHz
  waveform on `examples/buck_core_losses_n87.yaml`, the 200–1000 kHz coefficients were
  used with `signed_distance_hz = -900000`. The range whose edge is nearest is now used
  (the lower one on an exact tie), here 25–99 kHz with `signed_distance_hz = +1000`, so
  the estimate equals the one from a card whose lower range covers 100 kHz.
  `signed_distance_hz` is always measured to an edge of the range that was used.
  Estimates with `1/period` inside a range, or outside all of them, are unchanged.

- MCP `simulate`: the `losses` block now carries `electrothermal_mode` and
  `electrothermal_status`, and each device `electrothermal_status` and
  `electrothermal_loop_gain`. When a self-consistent junction-temperature loop has no
  fixed point (`thermal_runaway`, `beyond_card_temperature_axis`, `not_converged`) the
  kernel withholds every loss and temperature number; the reply showed
  `total_mean_power_w: null` and `tj_max_c: null` with the reason only in
  `losses-thermal.json`. The kernel's behaviour is unchanged.


- `fsw_measured_hz` in perturbation-campaign summaries (`pear-sim sweep perturbation`)
  is now the switching frequency, not the pooled rate of gate rises. Before, it counted
  rising edges of **all** gates together and counted `forced_gate` rises and the
  release rise at a forced window's end as switching edges. On
  `examples/synchronous_buck_complementary.yaml` (20 kHz) with a forced-off window on
  `leg.high` in the settled half it reported 45.5 kHz for 1 µs and 10 µs windows,
  40.9 kHz for 60 µs and 31.8 kHz for 120 µs; it now reports 20 kHz for all four.
  The estimate is taken per gate from `scheduled_gate` rises only, excluding every
  rise inside a declared window on that gate (`from <= t <= to`, so the release edge
  is excluded) and every rise-to-rise interval that spans one (the window may have
  masked command edges). New metric keys: `<gate>::fsw_measured_hz` per gate (usable
  in criteria as `signal: <gate>`, `field: fsw_measured_hz`), and
  `fsw_measured_lo_hz` / `fsw_measured_hi_hz`, the interval that must hold the true
  frequency if every gate rises once per carrier period (`(n ± c)/span` for `n` clean
  intervals in `c` clean runs). `fsw_measured_hz` is the median of the per-gate
  estimates and is reported only when every gate with events is measurable, all
  per-gate intervals intersect and the median lies strictly inside that shared
  (open) interval; otherwise
  (or when nothing is measurable) it is absent and `fsw_measured_status` says why.
  Checked on six examples (SPWM, SVPWM, DAB, LLC, H-bridge, closed-loop buck): none
  is rejected, and SPWM gives 21000.0003 Hz for a 21 kHz carrier where the per-gate
  spread is ±0.05 %. Skipped or extra pulses are not detected.

- `three-phase-modulation.json` now reconciles realized SPWM gate edges against the
  command record **through** the declared `perturbations.forced_gate_windows`. Before,
  it compared `scheduled_gate` events only with the nominal schedule, so a window that
  forced an SPWM output (e.g. `bridge.a.high` off for 1 µs mid-pulse) let the solver
  finish and then failed `write_result` with a count mismatch (`got 53, expected 52`
  on a 200 µs run):
  the forced edge was invisible and the window's release edge, logged as
  `scheduled_gate`, had no command behind it. Each expected edge now carries its kind
  (`forced_gate` inside a window, `scheduled_gate` outside) and is checked for time,
  state, kind and continuity. The replay uses the solver's own rules: the unforced
  initial level comes from the schedule at `transient.start` (so a window already open
  there is reconciled, and the recorded initial state must equal the masked level), and
  windows are re-evaluated at `transient.stop` exactly when some gate has a scheduled
  edge there (so a release landing on `stop` is expected). The summaries add
  `effective_gate_edge_count` (scheduled + forced — the report's "Realized gate edges"
  card now shows it), `forced_gate_edge_count`,
  `forced_gate_edges_by_output` (per block), `window_release_edge_count` and
  `suppressed_command_edge_count` (command edges a window masked);
  `scheduled_gate_edge_count` still counts `scheduled_gate` events, release edges
  included. The artifact lists the windows on its outputs under `forced_gate_windows`
  and names the method in `source_binding.realized_edge_reconciliation`. The command record
  (`modulation-cycles.jsonl`) is unchanged — a window changes the realization, never
  the command. Additive within v0.1: no run with a forced SPWM edge could produce the
  artifact before.

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

- Loss analysis (`losses-thermal.json`) now counts `forced_gate` edges as switching
  events. Before, a forced turn-off mid-pulse booked no `E_off` while the window's
  release edge still booked an `E_on`, so the switching-loss total came out silently low.

- Loss analysis read the current of an edge that falls between output samples from the
  wrong side: the post-edge sample was anchored a quarter step early, so a turn-on at
  30.15 us read the 30 us sample (device still off) and booked `E_on = 0` on a 1 us grid
  but 0.29 mJ on a 50 ns grid. The side of an edge a sample lies on is now the solver's
  own rule (`core.solver.run_time_tolerance`, `max(1e-15 s, 1e-12 · run span)`), so an
  edge 0.5 ps after a sample no longer reads that pre-edge sample either.
  An edge whose pulse is shorter than the output grid — no sample shows the device
  conducting — is counted in the new `switching.events_unresolved_by_output_grid`, its
  energy is not booked and a warning marks the totals as a lower bound. The MCP loss
  summary now carries that counter and the artifact's warnings.

- Mode-sequence extraction (gradient analysis, topology comparison) refuses a cycle that
  overlaps a `perturbations.forced_gate_windows` entry, naming the window (half-open on
  both sides: a window opening exactly at the cycle's end or closing exactly at its start
  does not touch it). Before, a
  forced edge inside the cycle failed with an unclear "cannot classify" error, and a
  window opened in the previous cycle had its release edge (logged as `scheduled_gate`)
  read as a PWM edge whose instant depends on the duty — a plausible but wrong
  sensitivity. Pick a `cycle_index` clear of every window.

### Added

- Optional model section `switching_transitions` (docs/model-format.md §7c): a
  post-processed estimate of the dead-time Coss swing of declared half-bridge legs (or
  full-bridge pairs) from a datasheet Coss(V)/Qoss(V) card with mandatory provenance —
  ZVS yes/no, node voltage at turn-on, minimum/maximum dead time, capacitive turn-on
  energy — and of the turn-off overshoot from a user-declared loop inductance. New
  artifact `switching-transitions.json`, a summary in `summary.json` and in the MCP
  `simulate` reply, warning `SWITCHING_TRANSITIONS_POSTPROCESSED`, capabilities
  `analysis.switching_transitions.coss_dead_time` and `.loop_overshoot` (experimental).
  The circuit equations are unchanged. Verified against closed forms and ngspice;
  validated only at the cited published operating points
  (`validation/switching_transitions/`). Examples `synchronous_buck_gan_transition`,
  `dab_phase_shift_zvs`, `llc_resonant_zvs`. Additive: models without the section are
  unchanged.

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
