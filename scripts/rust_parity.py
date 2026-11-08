"""Run every example the compiled kernel supports on both engines and compare.

Gate for the optional Rust kernel (pear-sim-core): every float metric must
agree with the reference engine within a relative tolerance (scaled by the
signal's own range for values near zero), and the topology event sequence
must be identical. Prints one line per example and exits non-zero on any
disagreement. Run: uv run python scripts/rust_parity.py [--tolerance 1e-12]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import SimulationError
from pear_sim.spec.model import load_model


def _worst(reference: dict, compiled: dict) -> tuple[float, str]:
    worst, where = 0.0, ""
    for signal, fields in reference.items():
        span = max(abs(fields.get("max", 0.0) or 0.0), abs(fields.get("min", 0.0) or 0.0))
        for key, value in fields.items():
            other = compiled[signal].get(key)
            if not isinstance(value, float) or not isinstance(other, float):
                continue
            scale = max(abs(value), abs(other), span, 1e-300)
            difference = abs(value - other) / scale
            if difference > worst:
                worst, where = difference, f"{signal}.{key}"
    return worst, where


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=1e-12)
    parser.add_argument("--mode", choices=("lean", "full"), default="lean")
    args = parser.parse_args()
    failures = 0
    covered = 0
    for path in sorted(glob.glob("examples/*.yaml")):
        name = os.path.basename(path)[:-5]
        try:
            HybridSimulator(load_model(path), backend="rust")
        except SimulationError:
            print(f"{name:36s} not covered")
            continue
        covered += 1
        timings = {}
        results = {}
        for backend in ("python", "rust"):
            started = time.process_time()
            results[backend] = HybridSimulator(
                load_model(path), evidence_mode=args.mode, backend=backend
            ).simulate()
            timings[backend] = time.process_time() - started
        worst, where = _worst(results["python"].metrics(), results["rust"].metrics())
        reference_events, compiled_events = results["python"].events, results["rust"].events
        same_events = [
            (e.kind, e.target, e.previous, e.current) for e in reference_events
        ] == [(e.kind, e.target, e.previous, e.current) for e in compiled_events] and all(
            abs(a.time - b.time) <= args.tolerance * max(1.0, abs(a.time)) + 1e-15
            for a, b in zip(reference_events, compiled_events)
        )
        ok = worst <= args.tolerance and same_events
        failures += not ok
        print(
            f"{name:36s} {'ok  ' if ok else 'FAIL'} worst {worst:8.1e} {where[:34]:34s} "
            f"events {'same' if same_events else 'DIFFER'}  "
            f"cpu {timings['python']:6.2f}s -> {timings['rust']:6.2f}s",
            flush=True,
        )
    print(f"{covered} examples covered, {failures} disagree")
    return 1 if failures or not covered else 0


if __name__ == "__main__":
    sys.exit(main())
