"""Run every example the compiled kernel supports on both engines and compare.

Gate for the optional Rust kernel (pear-sim-core): every float metric, every
recorded sample, every number of the physics audit (component energies) and
of the time integrals must agree with the reference engine within a relative
tolerance (scaled by the signal's or record's own range for values near
zero), and the topology event sequence must be identical. Prints one line per example and exits non-zero on any
disagreement. Run: uv run python scripts/rust_parity.py [--tolerance 1e-12]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

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


def _leaves(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _leaves(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _leaves(item, f"{path}[{index}]")
    elif isinstance(value, float):
        yield path, value


def _worst_tree(reference, compiled, label: str) -> tuple[float, str]:
    # Every float in a nested record (physics audit, time integrals). The scale
    # is the largest magnitude in the record, so near-zero entries (a balance
    # residual) are judged against the energies they balance.
    reference_leaves = dict(_leaves(reference))
    compiled_leaves = dict(_leaves(compiled))
    if reference_leaves.keys() != compiled_leaves.keys():
        return float("inf"), f"{label}: different fields"
    scale = max((abs(v) for v in reference_leaves.values()), default=1.0) or 1.0
    worst, where = 0.0, ""
    for key, value in reference_leaves.items():
        difference = abs(value - compiled_leaves[key]) / scale
        if difference > worst:
            worst, where = difference, f"{label}.{key}"
    return worst, where


def _state_scales(result) -> dict[str, float]:
    # Largest magnitude each state reaches anywhere in the run. Interval
    # extrema are judged against it: a 4e-13 s interval at a current zero
    # crossing has a swing of 1e-8 A, and judging rounding (1e-13 A) against
    # that swing reads as a 1e-5 disagreement that is not there.
    scales: dict[str, float] = {}
    records = [result.state_extrema.get("states", {})] + [
        interval.get("state_extrema") or {} for interval in result.mode_intervals
    ]
    for record in records:
        for state, entry in record.items():
            scales[state] = max(
                scales.get(state, 0.0), abs(entry["minimum"]), abs(entry["maximum"])
            )
    return scales


def _worst_extrema(
    reference,
    compiled,
    label: str,
    scales: dict[str, float],
    endpoint_shift: dict[str, float] | None = None,
) -> tuple[float, str]:
    # Extremum VALUES gate, judged against the state's run-wide magnitude.
    # Extremum TIMES do not gate: where a flat curve peaks is ill-conditioned
    # and moves with rounding without the value moving.
    if reference is None or compiled is None:
        return (0.0, "") if reference == compiled else (float("inf"), f"{label}: missing")
    reference = reference.get("states", reference)
    compiled = compiled.get("states", compiled)
    if reference.keys() != compiled.keys():
        return float("inf"), f"{label}: different states"
    worst, where = 0.0, ""
    for state, a in reference.items():
        b = compiled[state]
        swing = max(scales.get(state, 0.0), 1e-300)
        # An interval's extremum often sits on its boundary, and the boundary
        # is an event time. Event roots are only defined to Brent's tolerance
        # (4 eps absolute), so the boundary state may move by slope x 1e-15 s;
        # an extremum may differ by as much as its interval's end states do.
        allowance = (endpoint_shift or {}).get(state, 0.0)
        for key in ("minimum", "maximum"):
            difference = max(0.0, abs(a[key] - b[key]) - allowance) / swing
            if difference > worst:
                worst, where = difference, f"{label}.{state}.{key}"
    return worst, where


def _endpoint_shift(a: dict, b: dict) -> dict[str, float]:
    shift: dict[str, float] = {}
    for key in ("state_start", "state_end"):
        for state, value in (a.get(key) or {}).items():
            other = (b.get(key) or {}).get(state)
            if isinstance(value, float) and isinstance(other, float):
                shift[state] = max(shift.get(state, 0.0), abs(value - other))
    return shift


def _worst_signals(reference, compiled) -> tuple[float, str]:
    worst, where = 0.0, ""
    if not np.array_equal(reference.time, compiled.time):
        return float("inf"), "sample times differ"
    for name, values in reference.signals.items():
        span = float(np.max(np.abs(values))) or 1.0
        difference = float(np.max(np.abs(values - compiled.signals[name]))) / span
        if difference > worst:
            worst, where = difference, f"samples.{name}"
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
        reference, compiled = results["python"], results["rust"]
        scales = _state_scales(reference)
        worst, where = max(
            _worst(reference.metrics(), compiled.metrics()),
            _worst_signals(reference, compiled),
            _worst_tree(reference.physics_audit, compiled.physics_audit, "physics_audit"),
            _worst_tree(reference.time_integrals, compiled.time_integrals, "time_integrals"),
            _worst_extrema(reference.state_extrema, compiled.state_extrema, "state_extrema", scales),
            *(
                _worst_extrema(
                    a.get("state_extrema"),
                    b.get("state_extrema"),
                    f"interval[{i}]",
                    scales,
                    _endpoint_shift(a, b),
                )
                for i, (a, b) in enumerate(
                    zip(reference.mode_intervals, compiled.mode_intervals, strict=True)
                )
            ),
        )
        reference_events, compiled_events = results["python"].events, results["rust"].events
        # Event roots are located by Brent's method with xtol = rtol = 4 eps
        # (solve_ivp), so two correct engines may place one up to about that
        # far apart; allow twice the bracket.
        root_tolerance = 8 * np.finfo(float).eps
        same_events = [
            (e.kind, e.target, e.previous, e.current) for e in reference_events
        ] == [(e.kind, e.target, e.previous, e.current) for e in compiled_events] and all(
            abs(a.time - b.time) <= root_tolerance * (1.0 + abs(a.time))
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
