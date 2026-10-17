# Documentation

Start here:

- [Model format](model-format.md) — the YAML contract: components, controls, analyses, outputs.
- [Agent interface](agent-interface.md) — the MCP server, its tools and safety rules; CLI and Python API.
- [Python API](api.md) — the typed `pear_sim.api` surface.
- [Stability analysis](stability-analysis.md) — steady state, averaged small-signal models, loop gain and margins.

How it works and why:

- [Architecture](architecture.md) — layers, data flow and the main design choices.
- [Architecture decision records](adr/) — kernel approach, cross-platform policy, naming.
- [Universality](universality.md) — what the kernel can represent and where it stops.

Evidence and validation:

- [Validation plan](validation-plan.md) — the test design and the limits of the evidence.
- [Benchmark harness](benchmark-harness.md), [numerical evidence](m2-numerical-evidence.md),
  [agent evidence](m2-agent-evidence.md), [boost boundary evidence](m2-boost-boundary-evidence.md),
  [three-phase SPWM](m3-spwm.md), [complementary PWM](m3-complementary-pwm.md).

Design-loop layer (experimental):

- [Design specification](design-spec.md), [topology library](topology-library.md),
  [topology comparison](topology-comparison.md), [component roadmap](component-roadmap.md).
