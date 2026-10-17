# ADR-0001: Own system-level kernel with ideal switching

- Status: Accepted
- Date: 2026-07-18
- Deciders: the PEARSim project

## Context

The goal of the project is a usable, headless power-electronics simulator that can serve as a tool for coding agents. The main use cases cover converters, controllers, PWM, semiconductor losses and the thermal model.

Directions considered:

1. an ngspice or Xyce wrapper;
2. an OpenModelica wrapper;
3. a wrapper of a commercial system-level simulator via RPC;
4. a full transistor-level solver from scratch;
5. an own system-level kernel with ideal switching and optional SPICE.

Classic SPICE is optimized for nonlinear device models and approximates switches with very-low- or very-high-resistance states. Power-electronics systems often need a different representation: an ideal switch changes the topology exactly at an event, and between events the circuit is linear or piecewise-linear.

This is the approach described in the public documentation of commercial system-level power-electronics simulators: switched state-space equations, a configuration cache and a Switch Manager iterating over naturally commutated elements. That behavior matters more for the project's goal than compatibility with any particular tool's file format or GUI.

## Decision

We build our own system-level kernel with the following semantics:

- the circuit is compiled via MNA/descriptor form into state-equation operators;
- ideal switches/diodes change a discrete mode, not just the value of an extreme resistor;
- modes are compiled lazily and cached;
- the event queue handles exact PWM/sample hits and state-dependent zero crossings;
- the Switch Manager stabilizes natural commutation in zero time;
- capacitor voltages and inductor currents/fluxes are preserved across a mode change;
- a contradictory topology yields a `state_discontinuity` or `singular_topology` diagnostic;
- conduction and switching losses are computed from LUTs based on `pre/post` values;
- detailed SPICE models are an optional backend, not part of the ideal kernel's semantics.

The first kernel will be a Python reference implementation. The model, request and result contract will be stabilized before the hot path is ported to C++20.

## Why not just a SPICE wrapper

- it does not give exactly the same semantics of ideal shorts and opens;
- a large `Roff/Ron` ratio worsens numerical conditioning;
- event ordering and natural commutation remain outside PEARSim's control;
- thermal LUTs and system-level control would require an additional kernel despite the wrapper;
- licenses and the availability of vendor models are uneven across platforms.

SPICE remains valuable for:

- validating RLC circuits and nonlinear models;
- studying explicit parasitics and transistor edges;
- running existing manufacturer netlists;
- differential testing of an independent implementation.

## Why not OpenModelica as the core

OpenModelica offers multiphysics and a mature compiler, but:

- distribution and execution are much heavier than the target tool;
- the project's format and semantics would be tied to Modelica;
- OSMC-PL/AGPL complicates distributing a product under a different license model;
- it does not remove the need for an agent-friendly DSL, diagnostics and result normalization.

OpenModelica and the Modelica Standard Library may serve as an oracle for selected models.

## Why not a full SPICE from scratch

Implementing compact models, their manufacturer variants and a stable nonlinear solver is a separate, multi-year undertaking. It is not needed to reach the first value: fast system-level simulation of converters and control.

## Positive consequences

- the semantics are tailored to power electronics;
- event results, losses and diagnostics are under full control;
- the model format remains independent of the solver;
- both a fast ideal path and a more accurate SPICE path are possible;
- an agent can work through a small, deterministic CLI/API;
- Python enables fast testing, and C++ a later speed-up.

## Costs and negative consequences

- the project is responsible for hard topological cases and event handling;
- rank checks, consistent initialization and good error messages are needed;
- simultaneous switching and chatter/Zeno require an explicit policy;
- initially the component catalog will be smaller than in established commercial tools or OpenModelica;
- implementing Radau/DAE and an efficient sparse cache increases the scope;
- validation must draw on multiple independent sources.

## Criteria for the decision being correct

We consider the decision confirmed when the kernel:

1. passes analytical RLC and energy-balance tests;
2. correctly simulates buck CCM/DCM, boost and a rectifier;
3. preserves `v_C` and `i_L` through valid events;
4. finds commutation instants within a specified tolerance;
5. yields stable metrics as tolerances are tightened;
6. is faster than the SPICE backend on an event-heavy model of equivalent level of detail after the C++ port.

## Signals for reconsideration

The ADR should be reopened if:

- most real cases require transistor models and EMI rather than ideal switching;
- the state-space reducer cannot handle an important class of topologies despite the DAE fallback;
- the cost of maintaining the event manager exceeds its value compared with an existing permissively licensed kernel;
- a requirement for compatibility with a specific external format becomes more important than our own contract.

## References

- [SUNDIALS IDA](https://computing.llnl.gov/projects/sundials/ida)
- [Siconos nonsmooth simulation](https://nonsmooth.gricad-pages.univ-grenoble-alpes.fr/siconos/)
