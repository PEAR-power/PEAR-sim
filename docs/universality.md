# Universality: how PEARSim becomes a "model anything" tool

Date: 2026-07-21. Question: general-purpose system-level simulators model anything;
PEARSim currently ships device-type-by-device-type. How do we get to a universal tool
without losing the evidence DNA? This note records the architectural direction.

## Where we actually are (verified in code, not from memory)

The kernel is **already universal in shape, narrow in reach**. It does *not*
hard-code three topologies: `core/circuit.py` compiles **any** circuit graph
over the supported component set via MNA, then searches diode/switch
configurations (active-set, ≤16 diodes) and hybrid-integrates with event
localization. The three canonical topologies (buck/boost/inverting) are
**validation oracles**, not kernel boundaries.

Current reach limits:

- **Component set** (`spec/model.py`, checked 2026-11-14): resistor, capacitor
  (optional `esr`), inductor and saturating inductor (optional `dcr`), voltage/current
  source (DC or sine/step/pulse/PWL/double-exponential waveform), ideal switch (optional
  `on_resistance`), diode (optional `forward_voltage`, `on_resistance`), thyristor,
  voltage probe — 10 types; plus coupled inductors (`circuit.couplings`), electrical
  machines and subcircuits, which expand into these. Of the Wall 2 list below, sources,
  coupled inductors, the saturable inductor and the thyristor exist; controlled sources
  do not.
- **Singularity handling** (`core/circuit.py`): at most **one** constrained
  inductor per mode; capacitor loops and general inductor cutsets are rejected
  with `TopologyError`. This is the honest version of what general-purpose
  simulators solve with full index reduction + consistent initialization.

## The three walls to widen (in priority order)

### Wall 1 — general descriptor/DAE kernel (*the* wall)

Full index reduction for arbitrary capacitor loops / inductor cutsets, and
consistent initialization, replacing the one-constrained-inductor special case.
This single item converts "compiles most circuits" into "compiles any circuit
you can draw". Already named in the kernel scope; without it, topology
recognition does not scale. Everything else in this note gets cheaper after
this lands.

### Wall 2 — component library as *data* (stamps), never as code paths

A general-purpose simulator's component library is one MNA machine plus
per-component "stamps" and event surfaces. Rule for PEARSim: **a new component =
declared stamp + event surfaces + claim boundary; never a new oracle, never a new
code path through the solver.** Unlock order by value:

1. **Sources: sine/step/pulse/PWL** (already in kernel scope) — unlocks rectifiers,
   load steps, closed loop.
2. **Transformer / coupled inductors** — unlocks roughly half of the typical
   power-electronics demo catalog (flyback, forward, LLC, DAB, phase-shift).
   Without it PEARSim is confined to non-isolated DC/DC forever.
   Magnetizing-inductance model first (stays in state space); ideal-transformer
   algebraic constraint may need Wall 1.
3. **Controlled sources**; saturable inductor; PWL resistor.
4. Thyristor/triac — only when line-commutated rectifiers enter scope.

### Wall 3 — controls and analyses

Working with a system-level simulator is 90% *analyses*, not waveform staring:
periodic steady state, loop gain/Bode with margins, sweeps. Control blocks are
tracked separately.

## What we deliberately do NOT copy

- **The canvas.** General-purpose simulators are universal through a GUI
  schematic editor. PEARSim is agent-first: the canvas is the YAML model + the
  agent. Universality of the *model space*, not of the UI.
- **Proof-free universality.** A simulator that simulates anything but certifies
  nothing. PEARSim's differentiator stays: a universal kernel where `validated`
  status is granted only by the validation ladder (the per-run confidence card). Universal
  tool, graded trust — that combination is the product.

## Sequence (one line)

DAE kernel → transformer + AC/PWL sources → stamps library grows cheaply
→ analyses. After the first two steps the comparison stops being "narrow vs
universal" and becomes "what the user needs, with evidence — vs everything,
without".
