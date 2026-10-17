# Stability analysis: periodic steady state + loop-gain margins

Status: experimental (early alpha). Two capabilities, one retained bundle:

1. **Periodic steady state by shooting** (`pear_sim.analysis.steady_state`).
   Newton iteration on the one-period cycle map of the *production* hybrid
   solver: `F(x*) = x*` with a forward-difference Jacobian. Works for any
   model the transient solver handles — including natural-commutation (DCM)
   cycles — provided all controls share one switching frequency. Convergence
   status, iteration count, and residual are part of the result; a
   non-converged outcome is retained evidence, not an exception.
2. **Cycle-averaged small-signal loop gain** (`pear_sim.analysis.small_signal`).
   State-space averaging at the shooting operating point over every topology
   the cycle visits: `A = Σ d_k·A_k`. A duty perturbation lengthens the interval
   with the main gate on and shortens the one with the complementary gate on, so
   `B_d = (A_on−A_off)·x̄ + (c_on−c_off)`; dead-time intervals of a complementary
   leg keep their duration and only enter the average. Matching output feedthrough.
   Control-to-output Bode plus loop gain `L = C·G·H` for a **caller-declared**
   compensator `C(s)` (polynomial coefficients) and sensor gain `H`.

## Claim boundary (read before trusting a margin)

- The kernel has no closed-loop control blocks yet. Margins attest a
  *declared* design: "with this compensator and sensor gain, the averaged
  model of this exact converter has PM/GM of at least X". They are not a
  closed-loop simulation, not a guarantee outside the averaged model's
  validity, and meaningless above `f_sw/10` (recorded as `validity_max_hz`).
- A digitally sampled controller (`sampled_pi_pwm` updates once per period)
  adds ZOH + computation-delay phase lag the continuous analysis cannot see —
  up to `180 deg * f/f_update` from the ZOH alone. Declare it via the
  `sampling` request field to fold `(1 - e^(-sT))/(sT) * e^(-s*tau)` into
  `L(s)`; without a declaration the result is continuous and the bundle says
  so explicitly (`"sampling": "not_modelled"` in `loop-gain.json`). For the
  shipped `buck_closed_loop.yaml` gains this is the difference between
  "PM ~2 deg" (continuous, overstated) and a negative margin that predicts
  the kp=0.08/ki=150 limit cycle the switching kernel actually reproduces.
- Averaging supports one PWM leg (a `pwm` gate, or a complementary `high`/`low`
  pair with dead time) with zero natural events (CCM). DCM averaging and several
  legs are follow-ups; the gate `ccm_single_pwm_leg` fails closed.
- A loop may cross 0 dB (or −180°) more than once. Every crossing is recorded
  (`gain_crossovers_hz`, `phase_crossovers_hz` in `loop-gain.json`) and the reported
  margins are the worst over them, at the frequency where that worst margin occurs.
- A declared gain-margin minimum with no phase crossover passes only when the sweep
  covers the validity band (up to `f_sw/10`) and the phase stays above −180° across
  it ("unbounded within the validity band"); a sweep that stops earlier fails
  closed and says to extend `f_max_hz`.
- The steady-state step has no such restriction and is verified against the
  independent closed-form periodic oracles for buck/boost in CCM *and* DCM.

## CLI

```console
uv run pear-sim analyze stability MODEL --out OUT [--request REQUEST.json] [--json]
```

Without `--request`: steady state only. Exit `0` = all blocking gates pass,
`1` = retained evidence with a blocking `fail`, `2` = invalid request/input.

Request document (`schema/stability-request.schema.json`):

```json
{
  "schema": "pear-sim/stability-request/v0.1",
  "output": "v(out)",
  "compensator": {"numerator": [0.02, 40.0], "denominator": [1.0, 0.0]},
  "sensor_gain": 0.025,
  "f_min_hz": 10.0,
  "f_max_hz": 5000.0,
  "points": 200,
  "margins": {"min_phase_margin_deg": 45.0, "min_gain_margin_db": 6.0},
  "sampling": {"update_frequency_hz": 20000.0, "computation_delay_s": 0.0}
}
```

Compensator coefficients are descending powers of `s` (the example is the PI
`0.02 + 40/s`). Declared margins become **blocking gates**; a declared
minimum with no crossover on the grid is a `fail`, never a silent pass.

`sampling` declares the digital update model: `update_frequency_hz` defaults
to the switching frequency (the `sampled_pi_pwm` contract — one update per
period), `computation_delay_s` to zero. An empty object `"sampling": {}` is
therefore "ZOH at f_sw, no extra delay". The grid must stay below the update
frequency (fail closed). Omit the field entirely to keep the continuous
analysis; the omission is recorded, never silent.

## Bundle

`steady-state.json` (fixed point, convergence record, cycle averages),
`loop-gain.json` (grids, plant and loop Bode, margins, averaged model),
`stability-evidence.json` (request, gates, overall status, claim boundary),
`artifact-manifest.json` (SHA-256 of everything).

## Python API

```python
import pear_sim.api as api

pss = api.steady_state("examples/buck.yaml")          # PeriodicSteadyState
artifacts = api.stability("examples/buck.yaml", "out/stab",
                          request={"output": "v(out)", "sensor_gain": 0.025,
                                   "margins": {"min_phase_margin_deg": 45}})
```

## Verification

- Shooting fixed points match the independent periodic oracles (buck CCM
  cold-start, boost CCM, buck DCM) within documented tolerances.
- The averaged transfer function matches the textbook ideal-buck closed form
  to ~1e-9 relative; margins match an independent `brentq` computation on the
  closed form.
