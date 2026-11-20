# Switching-transition post-processing: dead-time Coss transition (ZVS) and commutation-loop overshoot

Status: draft (2026-11-20). Source brief: kokpit backlog item "Model przejścia łącznika" (2026-11-20),
literature note `~/kokpit/plany/arxiv-llm-agenci-energoelektronika-2026-11-20.md`.

## 📝 TLDR

PEARSim switches are ideal: a dead time is an instant node jump and a turn-off has no overshoot.
For GaN/SiC design the two questions an agent most needs answered are (1) does this leg reach
zero-voltage switching (ZVS) in the declared dead time, and if not at what voltage does the
incoming device turn on and what does that cost, and (2) how far does the commutation-loop
inductance Lσ push the turn-off peak voltage above the bus. This spec adds an optional,
post-processed `switching_transitions` section: a datasheet Coss(V)/Qoss(V) table on the device
card, per-leg bindings with an explicit Lσ, and a `switching-transitions.json` artifact with
per-edge results. The circuit equations do not change.

## Resolved assumptions (autonomous defaults)

| # | Question | Chosen answer | Rationale |
|---|---|---|---|
| Q1 | Does the transition enter the circuit equations or is it post-processing? | **Post-processing** on dead-time windows of the recorded ideal run. | The transition charge (2·Qoss ≈ tens of nC) is small against the inductor charge per period in every target example; post-processing keeps the kernel, the steady-state/Floquet maps and the gradient engine untouched and is fully reversible (an optional section). The size of what it ignores is COMPUTED per edge (`transition_fraction_of_dead_time`, `volt_second_deficit_v_s`, see API Contracts) and published; no validity threshold is enforced yet — there is no history to set one (rule: shadow mode first). ⚠ NEEDS HUMAN CONFIRMATION before any claim of waveform fidelity inside the dead time. |
| Q2 | One spec or two (ZVS vs overshoot)? | One spec, two phases. | Both share the Coss card, the leg binding and the artifact; each phase ships alone. |
| Q3 | Where does Coss live? | Own `coss_cards` inside the `switching_transitions` section (mandatory `provenance`), the same pattern as `core_losses.materials`. | `device_cards` are all-or-nothing with `loss_analysis` (conduction + thermal tables required) and refuse components with `forward_voltage` — which is how GaN reverse conduction is described. Putting Coss there would block the main GaN use. |
| Q4 | How is the commutating current driven during the dead time? | Two declared regimes: constant current (no `commutation_inductance`) and lumped LC (declared `commutation_inductance` + `far_voltage`). | Constant current is the textbook Qoss/I estimate and exact in the stiff-inductor limit (buck); DAB/LLC need the energy-limited resonant case. Extracting L and the far-terminal voltage from the netlist automatically is deferred (fragile through coupled windings). |
| Q5 | di/dt for the overshoot: declared or derived? | Both: declared `current_fall_time` gives the conventional V + Lσ·I/t_f AND a lossless Lσ–Coss ramp solution; without it, the instant-cut energy limit ½·Lσ·I² is reported. | Matches the brief ("podany albo wyprowadzony"); the derived number is an envelope, labelled as such. |
| Q7 | Full bridges switch two legs in the same dead time (DAB). | Optional `bridge_partner`: the two legs are integrated jointly on one commutation current; overlapping windows of unpaired declared legs are `not_evaluated`. | A one-leg model with a frozen far voltage is wrong when the far node swings too (review finding 8). |
| Q8 | Split overshoot into its own spec? | No: one spec, two phases, two PRs. | Both phases share the card, the leg binding and the artifact; the brief asks for both; each phase ships alone. |
| Q6 | New MCP tool or fields in an existing result? | Fields: new artifact readable with the existing `get_artifact`, plus a compact `switching_transitions` summary in the run summary. | Least new surface. |

## 📝 Problem Statement

- `docs/model-format.md` §6.1.0 and the `PIECEWISE_LINEAR_CONDUCTION` warning state: switching
  transitions are not modelled. Switching losses are table energies at DPT conditions (§7a).
- ABB "Power Circuit AI" (Hetrich, Sahu et al.) names the missing physical feedback between
  schematic and PCB: commutation-loop inductance and Vpeak = V_DC + Lσ·di/dt. Agentic design flows in
  the literature (PHIA, arXiv 2411.14214) learn these effects from oscilloscope data; none offers a
  deterministic datasheet-driven answer an agent can call.
- For GaN the design-critical choices are dead time and ZVS margin, and they are governed by the
  strongly nonlinear Coss(V) (10–100× from 0 V to rated voltage). A linear Coss guess is wrong by
  the factor Qoss·V / 2·Eoss (≈1.3–2 for GaN/superjunction).

## 📝 Proposed Solution

Post-process each dead-time window of a declared half-bridge **leg** (high switch, low switch, optional
antiparallel diodes) on the recorded ideal waveforms:

1. Take the commutating current I at the outgoing switch's turn-off (last pre-edge sample, the same
   convention as §7a turn-off energies) and the bus voltage V at the edge.
2. Integrate the transition of the incoming device's voltage u from V towards 0 through the
   nonlinear node capacitance C_node(u) = Coss_in(u) + Coss_out(V − u) + C_extra:
   - constant current: du/dt = −I / C_node(u);
   - lumped LC: du/dt = −i / C_node(u), L_c·di/dt = u − u_x, with u_x the declared far-terminal
     voltage referred to the incoming device's far rail, held at its pre-edge value.
   Reverse conduction clamps u at 0 (the GaN reverse drop is ignored here, declared); if the current
   reaches zero before the clamp the node turns back (resonant case) and is clamped at V.
   Far voltage per direction: incoming low (node falling) u_x = v_x; incoming high (node rising)
   u_x = V − v_x, with v_x the far-terminal potential measured from the leg's negative rail.
   With `bridge_partner` (full bridge, both legs swing in one window): state (u_a, u_b, i),
   L_c·di/dt = u_a + u_b − (V ± v_t), with v_t the declared voltage of the rest of the path (+ when
   leg a falls, − when it rises).
3. Report per edge: `zvs` (u ≤ `zvs_threshold`·V, default 1 %), `voltage_at_turn_on_v`,
   `minimum_dead_time_s` (all nodes clamped; `null` when not reached within the horizon),
   `maximum_dead_time_s` (lumped LC only: after the clamped current reverses, the first time a node
   leaves the threshold), `capacitive_turn_on_energy_j` = energy dissipated when the incoming
   channel discharges the residual: E = Eoss_in(u_r) + ∫_{V−u_r}^{V}(V − w)·Coss_out(w) dw + ½·C_extra·u_r²
   (= Qoss·V for identical devices at u_r = V), plus `incoming_eoss_j` = Eoss_in(u_r) alone. This is
   NOT added to §7a switching energies: a DPT E_on usually already contains the Coss energy, and
   the V·I overlap loss at u_r is not part of it. The two are reported side by side, never summed.
4. Overshoot (phase 2) for every turn-off that hard-commutates the current into the opposite device
   (the outgoing device ends up blocking V): `vpeak_conventional_v` = V + Lσ·|I|/t_f when
   `current_fall_time` is declared; `vpeak_ramp_v` from the lossless ramp ODE
   (Lσ·di_L/dt = −x, Coss_out(V + x)·dx/dt = i_L − i_dev(t), i_dev a linear fall over t_f);
   `vpeak_instant_cut_v` from ½·Lσ·I² = ∫_V^{V+ΔV}(w − V)·Coss_out(w) dw. Turn-offs whose current goes
   into the outgoing device's own diode are `not_evaluated` (their stress is the opposite device's
   reverse recovery / hard turn-on: out of scope). The ramp model assumes the clamped-inductive
   order (voltage first, then current); when t_f < Q_node(V)/I the channel current is gone before the
   node moves, so `vpeak_ramp_v` is `null` (`ramp_applicable: false`) and only the instant-cut
   estimate is given. All three are hard-commutation ESTIMATES, not envelopes: the instant cut
   ignores the incoming device's Coss in the loop.

Alternatives rejected: (a) nonlinear capacitors in the MNA kernel — breaks the "ideal switch, exact
event" contract, makes every dead time a stiff ODE, and touches every evidence family; deferred until
post-processing proves insufficient by its own error metric. (b) learned surrogate from scope data
(PHIA style) — contradicts the deterministic, datasheet-only decision; oscilloscope calibration stays a
separate backlog item.

## 📝 Architecture

- `spec/switching_transitions.py` (new): parse/validate the `switching_transitions` section with its
  own `coss_cards` (`voltage_axis_v` starting at 0, exactly one of `coss_f` (piecewise-linear C) or
  `qoss_c` (piecewise-linear Q, Q(0)=0, strictly increasing → piecewise-constant C), mandatory
  `provenance`); independent of `loss_analysis`.
- `analysis/switching_transitions.py` (new): the producer. Uses `result.events` (`scheduled_gate`
  edges of the switches' controls) to find windows: the outgoing switch's off edge and the incoming
  switch's next on edge of the same leg.
- `run/io.py`: writes `switching-transitions.json`, adds a summary to `summary.json`, adds a
  `SWITCHING_TRANSITIONS_POSTPROCESSED` info warning carrying the claim boundary.
- Capabilities: `analysis.switching_transitions.coss_dead_time` and
  `analysis.switching_transitions.loop_overshoot`, both `experimental`, evidence = verification tests;
  validation status stated separately.
- Layering: the producer imports nothing from evidence/benchmarks; the verification oracles
  (closed forms) live in tests and `validation/`, and the ngspice cross-check goes through a small
  local netlist writer in tests (the existing `oracle/` adapter covers the linear subset only).

## 📝 Data Model

```yaml
switching_transitions:           # NEW, optional
  zvs_threshold: 1 %             # optional, fraction of V (default 1 %)
  coss_cards:
    gan650:
      voltage_axis_v: [0, 10, 50, 100, 200, 400]
      coss_f: [...]              # OR qoss_c: [...]
      provenance: {source: "...", page: 4, figure: "Fig. 7"}
  legs:
    a:
      high: qpah                 # ideal_switch, p = bus+, n = node
      low: qpal                  # ideal_switch, p = node, n = bus-
      high_diode: dpah           # required, antiparallel to high
      low_diode: dpal            # required
      card: gan650               # both devices
      bus_voltage: v(dcp)        # recorded output, or {v(vp): 1, v(vn): 1}
      extra_node_capacitance: 0 F            # optional (transformer winding, snubber)
      commutation_inductance: 195 uH         # optional → lumped LC regime
      far_voltage: {v(lsec): 0.95}           # required iff commutation_inductance
      bridge_partner: b                      # optional, full bridge
      loop_inductance: 5 nH                  # optional → overshoot (phase 2)
      current_fall_time: 5 ns                # optional
```

Both antiparallel diodes are REQUIRED: without a reverse-conduction path the ideal run cuts the
inductor current in the dead time and the recorded waveform is not a usable basis.
Required outputs: `i(high)`, `i(low)`, `i(<diodes>)`, the bus voltage and every
`far_voltage` term. Fail closed on any missing one, on wrong orientation (high.p must equal
low.n), on a card without provenance, on non-positive values.

## 📝 API Contracts

`switching-transitions.json` (schema `pear-sim/switching-transitions/v0.1`):
`{schema, status, claim_boundary, legs: [{leg, regime, card, provenance, edges: [{time_s,
outgoing, incoming, bus_voltage_v, commutating_current_a, dead_time_s, zvs, voltage_at_turn_on_v,
minimum_dead_time_s, maximum_dead_time_s, hard_turn_on_energy_j, incoming_eoss_j,
transition_fraction_of_dead_time, volt_second_deficit_v_s, current_at_end_a, overshoot: {...}|null}], summary: {edges, zvs_edges, worst_voltage_at_turn_on_v,
max_minimum_dead_time_s, hard_turn_on_energy_total_j, extrapolated_lookups}}]}`.
Shadow metrics (published, not gated — no history yet to set a threshold):
`transition_fraction_of_dead_time` = t_min / dead time, and `volt_second_deficit_v_s` = ∫ u dt over the
dead time: the volt-seconds the ideal run applied to the load path that the real node did not.
Large values mean the recorded waveform after this edge is itself off (DAB duty loss, light-load or
TCM buck), so later edges inherit an error the post-processing cannot see.

## 📝 Edge Cases & Failure Scenarios

- Zero declared dead time: windows of length 0 → `zvs: false` with `minimum_dead_time_s` still
  reported (the useful answer: "you would need ≥ X ns").
- Commutating current ≈ 0 or in the wrong direction: node does not move; `voltage_at_turn_on_v = V`.
- Bus voltage above the Coss axis: lookups hold the last value and increment
  `extrapolated_lookups` (same as §7a).
- Overshoot above the Coss axis: same; the instant-cut root search fails closed if no root ≤ 10·V.
- Windows straddling the end of the run are dropped and counted.

## 📝 Risks & Impact Review

Additive and optional: models without the section are byte-identical in every contract. Further declared limits: datasheet Coss(V) is small-signal; large-signal Qoss and Coss hysteresis
(GaN, superjunction) make the real charge and the soft-switching loss larger; one static
`commutation_inductance` cannot follow an LLC whose effective inductance changes with rectifier
conduction (Lr while the secondary conducts, Lr + Lm while it idles). Risk is
mis-reading: an agent presenting post-processed ZVS as measured. Mitigation: claim boundary in the
artifact, warning, MCP `INSTRUCTIONS`, and the word "verified, NOT validated" unless a published
measurement agrees (see Validation). Rollback = remove the section.

## 📋 Verification and validation (separate, by design)

- **Verification (two independent methods, not PEARSim vs PEARSim):**
  (a) closed forms evaluated by exact piecewise integration of the card table: t = Q_node(V)/I;
  lumped LC ZVS iff ½·L·I² ≥ ∫_0^V (u_x − u)·C_node(u) du (for u_x within the swing); Eoss, Qoss·V;
  linear-C overshoot V + (Lσ I/t_f)·max(1 − cos(min(θ, π)), 2·|sin(θ/2)|), θ = ω·t_f (post-ramp) and I·√(Lσ/C).
  (b) ngspice transient with a behavioural nonlinear capacitor `I = pwl(V)·ddt(V)` on the same case.
  Gates must stay silent on correct code and fire on mutations (Coss × 1.2, inverted current sign).
- **Validation:** comparison with published bench measurements (app notes / papers, cited with page
  and figure). If none can be read reliably, the artifact and docs say "verified, NOT validated".
  `MEASURED` is reserved for those third-party measurements.
- Oracles in tests use their own literal tables, never the producer's parser.
- **Sensitivity sweep:** Coss scaled over a declared uncertainty band (source stated next to it) and Lσ over a
  declared band; report which conclusions (ZVS yes/no per operating point) are robust and which are
  only ranges.

## 📋 Phasing

Delivered (2026-11-20): phases 1–3 on one implementation branch, in separate commits;
the spec ships as its own PR first.

- Phase 1 — Coss card + leg binding + dead-time transition (ZVS, turn-on voltage, min/max dead time,
  hard-turn-on energy) + artifact + docs + MCP boundary + verification tests + examples
  (synchronous buck, DAB, LLC with dead time).
- Phase 2 — commutation-loop overshoot (Lσ, t_f) + its verification.
- Phase 3 — validation note and sensitivity sweep (`validation/switching_transitions/`).

## 📋 Implementation Plan

1. Card: `coss_cards` parse + validation + JSON Schema + unit tests (positive/invalid).
2. Section: `switching_transitions` parse + model wiring + schema + tests.
3. Producer: window detection from events, constant-current and lumped-LC integration, energies;
   artifact + summary + warning; tests on the synchronous buck.
4. Examples `synchronous_buck_gan_transition.yaml`, `dab_phase_shift_zvs.yaml`,
   `llc_resonant_zvs.yaml`; docs §7c; MCP `INSTRUCTIONS`; capabilities; CHANGELOG.
5. Verification tests: closed form + ngspice (skipped with a reason when ngspice is absent) +
   mutation tests.
6. Overshoot: fields, closed-form + ngspice tests.
7. Validation note + sensitivity sweep script and results table.
