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

## Closed-loop orbit and Floquet multipliers (experimental)

The margins above attest a **declared** compensator on an **averaged** model.
`solve_closed_loop_steady_state` asks the question directly on the switching
model with the controller inside it: does a small perturbation of the
closed-loop periodic orbit decay or grow from one cycle to the next?

```console
uv run pear-sim analyze floquet examples/buck_closed_loop_floquet.yaml --out runs/floquet
uv run pear-sim analyze floquet my_loop.yaml --out runs/f2 \
    --initial-state l1=0.92 --initial-state signal:int_prev=0.0167
```

The bundle holds `floquet.json` (orbit, multipliers, monodromy, assumptions),
`floquet-evidence.json` (blocking gates `closed_loop_orbit_converged`,
`multipliers_assessed`, `locally_stable` and, for a stable verdict,
`lyapunov_certificate` — plus the claim boundary) and
`artifact-manifest.json`. Exit code 0: locally stable; 1: evidence retained but
unstable, not resolved or not assessed; 2: model outside scope or invalid input,
nothing written. `--set NAME=VALUE` overrides model parameters, as for
`simulate`, and reaches the controller wherever a block value is written as
`"${kp}"` with a unit-less parameter (`kp: 0.002`), as in the shipped example:
`--set kp=0.08 --set ki=150` is the unstable design.

Agents get the same analysis through the MCP tool `floquet` (same worker and
code path, so the same numbers): it returns the verdict, the gates, the largest
multipliers, the orbit state, any violated assumption and the engine's notes —
including, when Newton stops at a limiter, the advice to pass `initial_state`.

```python
from pear_sim.spec.model import load_model
from pear_sim.analysis.closed_loop import solve_closed_loop_steady_state

result = solve_closed_loop_steady_state(
    load_model("my_closed_loop.yaml"),
    initial_state={"l1": 0.92, "c1": 12.0, "signal:int_prev": 0.0167},
)
result.stability        # "stable" | "unstable" | "not_resolved" | "not_assessed"
result.multipliers      # |mu|, oscillation frequency, uncertainty, largest first
result.as_dict()        # JSON-ready, including the assumptions checked
```

How it works. The state is augmented with the controller's memory —
`z = [circuit states ; every unit_delay register]` — and the one-cycle map is
one period of the production simulator started from `z` (registers enter as
each `unit_delay`'s `initial`). Newton finds `z* = F(z*)`; the monodromy
`dF/dz` at `z*` comes from central differences at two step sizes, and its
eigenvalues are the Floquet multipliers. All `|mu| < 1` means local stability;
the angle of a multiplier gives the frequency a perturbation oscillates at.

Two independent monodromies. Where the orbit has no natural commutation, the
primary one is **analytic**: the product of `expm(a_k tau_k)` over the orbit's
own compiled modes, plus the switching-instant term
`Phi_after (f_before - f_after) · T dd/dz` for each controlled turn-off, with
`dd/dz` from forward differentiation of the declared block equations — no
probing of the simulator. Central finite differences of the production cycle
map are computed every time as a cross-check; if the two disagree by more than
1e-6 the verdict is withheld (`not_assessed`) rather than one of them chosen.
With a natural commutation (DCM) the analytic rule does not apply and the
result says so and uses finite differences alone (`monodromy_method`).

Lyapunov certificate. A `stable` verdict also carries an eigenvalue-free
witness in `floquet.json` (`lyapunov_certificate`): a matrix `P` such that `P`
and `P - Mᵀ P M` are both positive definite, so `V(z) = zᵀ P z` decreases every
cycle. Re-checking it needs two Cholesky factorisations on the stored numbers
and nothing else. The same `P` is required to certify the second monodromy too
(finite differences), and `contraction_bound` — the guaranteed per-cycle decay
of `V` — must be at least `max |mu|²`. On the shipped example it equals it
(0.97940130). `P` is taken from the eigenvector metric when that passes: the
textbook choice `Mᵀ P M - P = -I` is valid here but nearly empty
(contraction 0.99999998), because the state mixes amperes, volts and
volt-seconds. A stable verdict without a certificate fails the
`lyapunov_certificate` gate.

Claim boundary:

- **Local** stability of the computed orbit only — not global stability, not
  the absence of other orbits or limit cycles, not robustness to parameters.
- The verdict is withheld (`not_resolved`) when a multiplier's distance from
  the unit circle is inside the spread between the two step sizes.
- No multipliers (`not_assessed`) when the map is not differentiable at the
  orbit: a limiter or the duty clamp engaged, or a probe run switched through a
  different mode sequence. The worst margin over every derivative run is
  reported as an assumption.
- Newton is local. Start near the orbit (`initial_state`), e.g. from the last
  cycle of a settled transient; a start that clamps the duty is reported as
  non-converged with the limiter named.
- v1 scope, everything else refused by name: control laws in the signal layer
  driving `signal_pwm` (the built-in `sampled_pi_pwm` keeps its integrator
  inside the kernel, where it cannot be set); blocks constant, probe, gain,
  sum, product, divide, saturate, unit_delay and model-level switch; DC
  sources; one switching frequency with the signal task at offset 0; no
  machines, saturable inductors, inductor-star invariants or forced gate
  windows.

Verification (`tests/test_closed_loop_floquet.py`):

- Analytic monodromy vs the closed-form oracle below: multipliers 1.6e-15 /
  4.8e-14, monodromy entries 3.4e-11 / 1.7e-10; dropping the switching-instant
  term moves the multipliers by 1.4e-2 / 2.1e-1 and is caught at runtime by the
  finite-difference cross-check. In DCM one multiplier is exactly 0 (the
  inductor current returns to zero every cycle, erasing its perturbation).
- **Closed-form oracle** (`tests/reference/buck_closed_loop_floquet_oracle.py`,
  imports nothing from PEARSim): for the ideal CCM buck under a sampled PI both
  topologies share the state matrix, so the monodromy is two 2×2 matrix
  exponentials plus one rank-one switching-instant term, and the orbit is one
  bracketed root. Engine vs oracle: multipliers 1.2e-11, monodromy entries
  5.6e-8, valley current 1.9e-9 (gate 1e-7 / 1e-6 / 1e-7). The oracle is itself
  checked against physics outside its algebra: the textbook RLC natural
  frequencies (1e-12), and the slow-integrator decay rate `ki·g`, whose error
  shrinks in proportion to `ki`. `g` is the DC gain **as sampled** at the
  cycle start (24.085 V on this plant, not Vin = 24 V: the tick reads the
  ripple) — with Vin the gap stays at 0.36 %, which is what an averaged model
  that ignores the sampling point misses.
- The multipliers match a monodromy identified by least squares from ONE long
  run started off the orbit (no finite differences, controller memory carried
  by the kernel): max difference 1.2e-8 healthy, 3.5e-6 unstable design;
  carrying a register's old value instead of its input gives 6.9e-3.
- On the 24 V → 12 V buck with a PI in signal blocks, kp=0.002/ki=30 is
  `stable` and kp=0.08/ki=150 is `unstable` with the critical pair at ~3.9 kHz,
  where the switching kernel shows a ~4 kHz limit cycle; the discrete-aware
  averaged margins agree on which design is unstable and on the crossover
  within 15 %. That agreement is between two methods of this tool —
  verification, not validation against hardware.
