# PEARSim — project memory

Read and follow [`AGENTS.md`](AGENTS.md) first: the CLI contract, model schema,
validation gates, clean-room rules, and the current claim boundary there are
binding requirements. This file adds orientation, architecture, and current
priorities; when they conflict, AGENTS.md wins.

## What this is

A clean-room, headless, deterministic power-electronics simulator (Python,
`numpy`/`scipy`/`PyYAML`/`jsonschema` only) built to be driven programmatically — by shell,
CI, and AI agents over MCP. The CLI is the product surface; every
result is a hash-manifested, portable artifact bundle. Early alpha: do not use
for design sign-off. Honest claim boundaries are a core feature — never widen a
claim to make output look better.

## Architecture

```
src/pear_sim/
  core/        circuit (MNA + diode active set), solver (hybrid solve_ivp loop),
               scheduling (PWM/complementary/SPWM edges), metrics (quadrature, extrema)
  spec/        model (loader/validator), controls (gate expansion + PwmControl),
               units, canonical_topologies (structural recognition)
  run/         io (bundle writer), contracts (canonical JSON + hashing)
  analysis/    harmonics, analytic_checks, mode_coverage, gate_drive,
               three_phase_modulation
  evidence/    expectations (KPI eval), sweep + benchmark_harness (retained runs),
               numerical/platform evidence, run_compare, parameter_sweep(+worker)
  benchmarks/  independent closed-form oracles (buck/boost/inverting) + _shared
  review/      hash-chained human-review packets
  reporting/   offline HTML/SVG projections (never recompute producer data)
  api/         cli (entry point), capabilities (fail-closed catalog)
  mcp/         MCP server (stdio), tool definitions, sandboxed workspace + worker
  design/      typed design specification (hard vs soft requirements)
  topology/    declarative topology library + comparison
  oracle/      second-engine adapter (ngspice, linear subset)
  errors.py, limits.py, _jsonutil.py at the root

rust/pear-sim-core/  optional compiled kernel (Rust + PyO3): RK45 segment integration
               with terminal events, mirroring solve_ivp; reached through
               core/rust_backend.py with `backend="rust"`; Python stays the reference
```

Old flat import paths (`pear_sim.solver`, `pear_sim.model`, …) still work via
`sys.modules` alias forwarders; new code should import the canonical paths.
Layering is a DAG: benchmarks never import the solver (oracle independence);
reports never recompute; core never imports the evidence/benchmark layer.

## Commands

```bash
uv sync --locked --extra dev
uv run pytest                 # full suite, must stay green per commit
uv run ruff check .
uv run pear-sim lint|explain|render|simulate|report|review ...
uv run pear-sim sweep buck-boundary examples/buck_dcm.yaml --out OUT --json
```

Evidence-equivalence check for refactors touching producers: regenerate the
three boundary sweeps and compare against a pre-change baseline ignoring only
volatile fields (`*sha256*`, `wall_time`, `bytes`, `runtime_s`; skip `.html`).
Everything else — statuses, gate values, metrics — must match exactly.

## Conventions

- Conventional commits; one concern per commit; never mix refactor + feature.
- Every solver/model bugfix starts with a regression test; numerical tests
  assert documented tolerances, never exact floats.
- Fail closed: unknown fields, non-finite numbers, and unknown capabilities are
  errors, not defaults. Never silently regularize an invalid circuit.
- Schema changes update JSON Schema + loader + `docs/model-format.md` +
  examples + tests in the same change, and get a `CHANGELOG.md` entry
  (breaking changes need a migration note).
- Tests may keep importing old flat paths — that is the forwarder regression
  suite. Don't "clean up" those imports.
