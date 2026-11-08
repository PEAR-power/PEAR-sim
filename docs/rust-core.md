# Compiled kernel (Rust, optional)

`rust/pear-sim-core` is an optional compiled kernel. It takes over the integration of
each segment between switching events: Dormand-Prince 5(4) on the segment's affine
system `x' = A x + c`, with the topology events located inside each step. Everything
else (loading, mode compilation, the diode active set, metrics, evidence, reports)
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

Agreement is in tolerances, not bit for bit: BLAS may sum a matrix-vector product in a
different order than the kernel's loops, and the event guards are read as affine rows
instead of through the full node solution.

## Checked

| Check | Result |
|---|---|
| Tableau vs SciPy | identical, bit for bit |
| Brent roots vs SciPy | 1 539 of 1 541 identical, the rest within one ulp |
| Error against the **exact** solution (matrix exponential), 200 random stable systems | same accuracy class as SciPy: median 0.27 vs 0.31 of the tolerance, worst 12.6 for both |
| All 17 supported examples, both engines | every float metric within 1e-9 relative (worst seen 3.8e-14), identical event sequences |

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

On the simulation phase alone (CPU time, median of three, both engines alternating):
1.05–1.6× on the covered examples. That is what the profile predicted: integration is
only part of a run's cost. The rest is reading the dense solution between steps
(samples, metric quadrature, extrema) and solving the node voltages for each recorded
output, still in Python. Moving those into the kernel is the next step; the
integration interface and the parity gate built here are what it will plug into.
