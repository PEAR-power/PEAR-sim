# Compiled kernel (Rust, optional)

`rust/pear-sim-core` is an optional compiled kernel. It takes over the work inside each
segment between switching events:

- integration: Dormand-Prince 5(4) on the segment's affine system `x' = A x + c`, with
  the topology events located inside each step;
- what is read from it: every recorded output at the sample times, the Gauss-Legendre
  quadrature of every output (means, RMS) and, in a full run, of every component's
  absorbed power (the energy audit), and the state extrema.

Everything else (loading, mode compilation, the diode active set, evidence, reports)
stays in Python, and the Python engine stays the reference.

```bash
uvx maturin build --release -m rust/pear-sim-core/Cargo.toml -o rust-dist
uv pip install rust-dist/*.whl
pear-sim simulate examples/buck.yaml --out runs/buck-rust --backend rust
```

## Same algorithm, step for step

The kernel mirrors SciPy's `solve_ivp(method="RK45")` operation for operation: the same
Dormand-Prince tableau (identical bit for bit), the same initial-step choice, error
norm and step control, the same dense output, and the same terminal-event handling
with a port of SciPy's Brent root finder. It returns the same result shape, so every
consumer (metrics, extrema, sampling, events) runs unchanged.

Agreement is in a tight tolerance (1e-12 relative), not bit for bit: BLAS may sum a matrix-vector product in a
different order than the kernel's loops, and the event guards are read as affine rows
instead of through the full node solution.

## Checked

| Check | Result |
|---|---|
| Tableau vs SciPy | identical, bit for bit |
| Brent roots vs SciPy | 1 539 of 1 541 identical, the rest within one ulp |
| Error against the **exact** solution (matrix exponential), 200 random stable systems | same accuracy class as SciPy: median 0.27 vs 0.31 of the tolerance, worst 12.6 for both |
| All 17 supported examples, both engines, lean and full | every metric, every recorded sample, every number of the energy audit and time integrals, and every state extremum within 1e-12 of the reference (worst seen 2.2e-13); identical event sequences, event times within Brent's own bracket (8 eps) |
| Extremum search vs NumPy on the same steps | within 1e-13 of the state's range on 40 random systems |
| The gate itself: step-control safety factor 0.9 → 0.89 in the kernel | caught on 15 of 17 examples (deviations 5e-12 to 2.4e-10); the other two take steps fixed by `max_step`, so step control cannot move them |
| The gate itself: extremum search without derivative roots | caught on 9 of 17 examples (1e-7 to 4e-6) |

Two allowances in the gate, both derived rather than chosen. Event times are compared
within 8 eps (relative), because `solve_ivp` locates roots with Brent's method at
xtol = rtol = 4 eps: two correct engines can stop anywhere in that bracket. An extremum
of one mode interval may differ by as much as that interval's end states do, because it
often sits on the boundary that such an event time defines. Extremum values are judged
against the state's run-wide magnitude, not the swing inside a 1e-13 s interval.

`tests/test_rust_core.py` pins these; `scripts/rust_parity.py` runs the example gate,
and the CI job `rust-core` runs both on every change.

## What it covers, and refuses

Supported: the `dopri5` solver, DC sources, ideal and piecewise-linear switches and
diodes, thyristors, saturable inductors, coupled windings, sampled controllers and
signal blocks (they act between segments). Refused, with the reason: the `dop853`,
`radau` and `bdf` solvers, time-varying waveform sources, electrical machines. Nothing
falls back to the Python engine silently.

A run from the kernel records `backend: rust-core-<version>` in `run.json`, and the
backend is part of `run_sha256`: it is a different run identity. Verifiers that accept
only reference-engine runs (evidence evaluation, sweeps, run comparison) refuse it.

## Speed, measured

Simulation phase alone, CPU time, median of three, both engines alternating:

| Example | lean | full |
|---|---|---|
| buck | 7.4× | 6.8× |
| buck, discontinuous conduction | 16× | 25× |
| buck with loss cards | 6.1× | 4.9× |
| liquid-cooled traction inverter | 2.9× | 2.9× |
| three-level NPC | 2.0× | 2.6× |
| five-level NPC | 1.2× | 1.2× |

What remains is Python work around the segments: choosing the next mode after a
switching edge (the diode active-set search) and building the event descriptions. On the
NPC inverters that is most of what is left. For a small model run from the command
line, starting Python and writing the result bundle now cost more than the simulation.
