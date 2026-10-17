# Programmatic API (`pear_sim.api`)

Status: early alpha, additive-first. This is the agent-facing typed surface;
the CLI is a client of the same library calls. Import once:

```python
import pear_sim.api as api
```

## Verbs

| Call | Returns | Notes |
|---|---|---|
| `api.load(path, overrides=None)` | `Model` | Full validation: JSON Schema, then semantics. |
| `api.lint(model_or_path)` | `LintReport` | Static + initial-topology check; no simulation. |
| `api.explain(model_or_path)` | `dict` | Resolved model, hashes, conventions, limitations. |
| `api.hashes(model_or_path)` | `dict[str, str]` | `source/model/request` SHA-256. |
| `api.render(model_or_path)` | `RenderedSchematic` | Deterministic SVG + layout metadata; caller persists. |
| `api.simulate(model_or_path, out_dir)` | `SimulationRunBundle` | Runs transient, writes the portable bundle; `bundle.metrics()` for time-weighted metrics. |
| `api.report(run_dir, out_dir)` | `ReportArtifacts` | Offline HTML/JSON projection; never recomputes. |
| `api.evaluate(run_dir, expectations, out_dir)` | `EvidenceArtifacts` | Fail-closed KPI evaluation of a verified bundle. |
| `api.compare(ref_run, cand_run, profile, out_dir)` | `RunComparisonArtifacts` | Same-physics verified-run comparison. |
| `api.sweep_buck_boundary(model, out, load_ratios=…)` | `BuckBoundarySweepArtifacts` | Likewise `sweep_boost_boundary`, `sweep_inverting_buck_boost_boundary`, `sweep_parameter`. |
| `api.steady_state(model_or_path)` | `PeriodicSteadyState` | Newton shooting on the one-period cycle map; convergence status retained. |
| `api.stability(model, out_dir, request=…)` | `StabilityArtifacts` | Retained bundle: steady state + declared-compensator loop-gain margins as blocking gates. |
| `api.list_capabilities()` / `api.get_capability(id)` | `dict` | Fail-closed capability catalog; unknown ids come back `unsupported`. |

Functions accepting `model_or_path` take a file path (optionally with
`overrides={"vin": "120 V"}`) or an already-loaded `Model` (no overrides).

## Errors

Every failure raised by the API carries a stable machine-readable code:

```python
try:
    run = api.simulate("buck.yaml", "runs/buck")
except Exception as exc:
    code = api.error_code(exc)   # e.g. "model.invalid", "simulation.failed"
```

Codes are public contract (changing one is a breaking change); message text
is not. Current codes: `model.invalid`, `model.units`, `model.topology`,
`simulation.failed`, `evidence.expectations_invalid`, `sweep.parameter_invalid`,
`compare.invalid`, `evidence.platform_invalid`, `report.invalid`,
`report.sweep_invalid`, `report.numerical_invalid`, `report.svg_invalid`,
`analysis.gate_drive_invalid`, `analysis.modulation_invalid`,
`benchmark.{buck,boost,inverting_buck_boost}_boundary_invalid`,
`review.{error,invalid,integrity,stale,decision_denied}`, plus fallbacks
`io.error`, `request.invalid`, `internal.error`. The CLI's `--json` error
payload carries the same `code` field.

## Out of scope (deliberate)

- Review-gated simulation (packet + decision binding) stays a CLI workflow
  composed from `pear_sim.review` — it exists to put a human in the loop.
- The numerical/platform evidence producers (`verify numerics|platforms`)
  remain importable from `pear_sim.evidence` but are not facade verbs yet.
