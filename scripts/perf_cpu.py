"""CPU-time benchmark of the solver, for comparing two checkouts on one machine.

Two subcommands:

``measure --label NAME --round N --out FILE``
    Simulate a fixed set of shipped examples once each and append one JSON line
    per example to FILE with the CPU time (``time.process_time``) it took.
    CPU time, not wall time: a shared CI runner steals wall time from us at
    random, CPU time is what our own code spent. Run it with the Python of the
    checkout you want to measure; it only uses the stable public API
    (``pear_sim.load_model`` + ``HybridSimulator``), so the same file measures
    an older checkout too.

``compare FILE --base NAME --head NAME [--threshold F] [--summary PATH]``
    Read the lines, take the MINIMUM over rounds per example and side (the
    least-disturbed run is the best estimate of the code's own cost), and
    report head/base per example. It also reports the A/A ratio — the same
    side split into odd and even rounds — which is the noise floor measured in
    the very same job. A head/base ratio inside the A/A spread is not
    evidence of anything.

The verdict is advisory (shadow mode): ``compare`` always exits 0 unless the
input is broken. Whether a regression *would* have been flagged is printed and
written to the summary, so the threshold can be set later from recorded data.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

# Chosen to cover different solver paths at a few seconds of CPU in total:
# plain diode converters, a transformer rectifier, MOSFET with on-resistance,
# an H-bridge drive, three-phase SVPWM and a dual active bridge.
# Short examples repeat inside one measurement so that every sample is about a
# second of CPU: a 0.06 s sample is dominated by timer and cache noise.
REPEATS = {
    "buck": 10,
    "boost": 10,
    "rectifier_6pulse_transformer": 2,
    "sync_buck_mosfet": 1,
    "dc_drive_hbridge": 1,
    "inverter_three_phase_svpwm": 1,
    "dab_phase_shift": 1,
}
WORKLOADS = tuple(REPEATS)


def measure(args: argparse.Namespace) -> int:
    import pear_sim
    from pear_sim import HybridSimulator, load_model

    examples = Path(args.examples)
    with open(args.out, "a", encoding="utf-8") as sink:
        for name in WORKLOADS:
            model = load_model(examples / f"{name}.yaml")
            start = time.process_time()
            for _ in range(REPEATS[name]):
                HybridSimulator(model).simulate()
            cpu_s = time.process_time() - start
            sink.write(
                json.dumps(
                    {
                        "label": args.label,
                        "round": args.round,
                        "workload": name,
                        "cpu_s": cpu_s,
                        "repeats": REPEATS[name],
                        "pear_sim": str(Path(pear_sim.__file__).parent),
                        "python": platform.python_version(),
                    }
                )
                + "\n"
            )
    return 0


def _best(rows: list[dict[str, Any]], label: str, rounds: set[int] | None = None) -> dict[str, float]:
    best: dict[str, float] = {}
    for row in rows:
        if row["label"] != label or (rounds is not None and row["round"] not in rounds):
            continue
        best[row["workload"]] = min(best.get(row["workload"], math.inf), row["cpu_s"])
    return best


def _ratios(top: dict[str, float], bottom: dict[str, float]) -> dict[str, float]:
    missing = set(WORKLOADS) - (set(top) & set(bottom))
    if missing:
        raise SystemExit(f"workloads missing from the measurements: {sorted(missing)}")
    return {name: top[name] / bottom[name] for name in WORKLOADS}


def _geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def compare(args: argparse.Namespace) -> int:
    rows = [json.loads(line) for line in Path(args.file).read_text(encoding="utf-8").splitlines() if line]
    rounds = sorted({row["round"] for row in rows if row["label"] == args.base})
    if len(rounds) < 2:
        raise SystemExit("need at least two rounds per side to measure the A/A noise floor")
    odd = {r for i, r in enumerate(rounds) if i % 2 == 0}
    even = set(rounds) - odd

    base, head = _best(rows, args.base), _best(rows, args.head)
    ab = _ratios(head, base)
    aa_base = _ratios(_best(rows, args.base, odd), _best(rows, args.base, even))
    aa_head = _ratios(_best(rows, args.head, odd), _best(rows, args.head, even))

    # Noise floor: the largest A/A departure from 1 seen in this job, either side.
    noise = max(abs(math.log(v)) for v in [*aa_base.values(), *aa_head.values()])
    geo = _geomean(list(ab.values()))
    limit = 1.0 + args.threshold
    over = [name for name, ratio in ab.items() if ratio > limit]
    would_fire = geo > limit or bool(over)

    result = {
        "mode": "shadow",
        "threshold": args.threshold,
        "rounds_per_side": len(rounds),
        "geomean_head_over_base": geo,
        "aa_noise_floor": math.exp(noise) - 1.0,
        "would_fire": would_fire,
        "workloads": {
            name: {
                "base_cpu_s": base[name],
                "head_cpu_s": head[name],
                "head_over_base": ab[name],
                "aa_base": aa_base[name],
                "aa_head": aa_head[name],
            }
            for name in WORKLOADS
        },
    }
    print(json.dumps(result, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "## CPU-time benchmark (shadow mode — never blocks)",
        "",
        f"Geometric mean head/base: **{geo:.3f}** · A/A noise floor in this job: "
        f"±{result['aa_noise_floor'] * 100:.1f} % · provisional threshold +{args.threshold * 100:.0f} % · "
        f"would have flagged: **{'YES' if would_fire else 'no'}**"
        + (f" ({', '.join(over)})" if over else ""),
        "",
        "| workload | base CPU s | head CPU s | head/base | A/A base | A/A head |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in WORKLOADS:
        w = result["workloads"][name]
        lines.append(
            f"| {name} | {w['base_cpu_s']:.3f} | {w['head_cpu_s']:.3f} | {w['head_over_base']:.3f} "
            f"| {w['aa_base']:.3f} | {w['aa_head']:.3f} |"
        )
    lines += [
        "",
        f"Minimum CPU time (`time.process_time`) over {len(rounds)} interleaved rounds per side on one runner. "
        "A/A = the same checkout, odd rounds over even rounds: a head/base ratio inside that spread "
        "is noise. Shadow mode: this records what the gate *would* do; it does not fail the job.",
    ]
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as sink:
            sink.write("\n".join(lines) + "\n")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--label", required=True)
    m.add_argument("--round", type=int, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--examples", default="examples")
    c = sub.add_parser("compare")
    c.add_argument("file")
    c.add_argument("--base", required=True)
    c.add_argument("--head", required=True)
    c.add_argument("--threshold", type=float, default=0.10)
    c.add_argument("--summary")
    c.add_argument("--json-out")
    args = parser.parse_args(argv)
    return measure(args) if args.command == "measure" else compare(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
