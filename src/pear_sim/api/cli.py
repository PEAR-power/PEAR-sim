from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from pear_sim import __version__
from pear_sim.benchmarks import (
    BOOST_DEFAULT_LOAD_RATIOS,
    DEFAULT_LOAD_RATIOS,
    INVERTING_BUCK_BOOST_DEFAULT_LOAD_RATIOS,
)
from pear_sim.api.capabilities import build_capabilities, get_capability
from pear_sim.api.facade import error_code
from pear_sim.run.contracts import explain_model, explanation_text
from pear_sim.evidence.evidence_io import evaluate_run
from pear_sim.errors import PearSimError
from pear_sim.run.io import write_result
from pear_sim.spec.model import load_model
from pear_sim.evidence.numerical_evidence import FULL_PROFILE_ID, run_buck_numerical_evidence
from pear_sim.evidence.parameter_sweep import run_parameter_sweep
from pear_sim.evidence.tolerance_corners import run_tolerance_corners
from pear_sim.evidence.platform_evidence import REQUIRED_SYSTEMS, aggregate_platform_evidence
from pear_sim.reporting.render import render_schematic
from pear_sim.reporting.report import generate_report
from pear_sim.review.packets import (
    append_review_annotation,
    check_review_decision,
    check_review_packet,
    compute_topology_subject_digest,
    prepare_result_review,
    prepare_topology_review,
)
from pear_sim.evidence.run_compare import compare_runs
from pear_sim.evidence.stability import run_stability_analysis
from pear_sim.core.solver import HybridSimulator
from pear_sim.evidence.sweep import (
    run_boost_boundary_benchmark,
    run_buck_boundary_benchmark,
    run_inverting_buck_boost_boundary_benchmark,
)


def _overrides(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(f"Override must have NAME=VALUE form: {value!r}")
        name, raw = value.split("=", 1)
        if not name or not raw:
            raise argparse.ArgumentTypeError(f"Override must have NAME=VALUE form: {value!r}")
        result[name] = raw
    return result


def _load_ratios(
    value: str | None,
    default: tuple[float, ...] = DEFAULT_LOAD_RATIOS,
) -> tuple[float, ...]:
    if value is None:
        return default
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError("--ratios must be a comma-separated list of finite positive numbers")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise ValueError(
            "--ratios must be a comma-separated list of finite positive numbers"
        ) from exc


def _required_systems(value: str) -> tuple[str, ...]:
    systems = tuple(part.strip() for part in value.split(","))
    if any(not system for system in systems) or len(systems) != len(set(systems)):
        raise ValueError("--require must contain unique, comma-separated platform.system names")
    if set(systems) != set(REQUIRED_SYSTEMS):
        raise ValueError(
            "Cross-platform evidence requires exactly " + ",".join(REQUIRED_SYSTEMS)
        )
    return systems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pear-sim", description="Headless power-electronics simulator"
    )
    parser.add_argument("--version", action="version", version=f"pear-sim {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    catalog = subparsers.add_parser("catalog", help="List implemented components and analyses")
    catalog.add_argument("--json", action="store_true")

    capabilities = subparsers.add_parser(
        "capabilities",
        help="Describe fail-closed agent capabilities and claim boundaries",
    )
    capabilities.add_argument("--id", help="Return one capability; unlisted IDs are unsupported")
    capabilities.add_argument("--json", action="store_true")

    mcp = subparsers.add_parser(
        "mcp",
        help='Serve the MCP server over stdio (needs: pip install "pear-sim[mcp]")',
    )
    mcp.add_argument(
        "--workspace",
        type=Path,
        help="Directory for inline models and run bundles "
        "(default: $PEAR_SIM_WORKSPACE, else <project>/.pear-sim, else ~/.pear-sim)",
    )
    mcp.add_argument(
        "--project-root",
        type=Path,
        help="Directory model_path arguments may read from (default: current directory, "
        "unless it is / or the home directory)",
    )
    topology = subparsers.add_parser(
        "topology", help="Read and audit the declarative topology library"
    )
    topology_actions = topology.add_subparsers(dest="topology_action", required=True)
    topology_list = topology_actions.add_parser(
        "list", help="List validated library entries"
    )
    topology_list.add_argument("--dir", type=Path, required=True)
    topology_list.add_argument("--json", action="store_true")
    topology_verify = topology_actions.add_parser(
        "verify",
        help=(
            "Re-extract every stored mode sequence from a fresh run of its "
            "source model and compare"
        ),
    )
    topology_verify.add_argument("--dir", type=Path, required=True)
    topology_verify.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory source_model paths are resolved against",
    )
    topology_verify.add_argument("--json", action="store_true")
    topology_compare = topology_actions.add_parser(
        "compare",
        help=(
            "Optimise each finalist to its OWN optimum and place them on a "
            "Pareto front judged by the oracle"
        ),
    )
    topology_compare.add_argument("--dir", type=Path, required=True)
    topology_compare.add_argument(
        "--request", type=Path, required=True, help="Comparison request JSON"
    )
    topology_compare.add_argument(
        "--out", type=Path, help="Write the comparison record as JSON"
    )
    topology_compare.add_argument("--json", action="store_true")
    doctor = subparsers.add_parser(
        "doctor", help="Diagnose runtime, backends and platform capabilities"
    )
    doctor.add_argument("--json", action="store_true")
    lint = subparsers.add_parser("lint", help="Parse and validate a model")
    lint.add_argument("model", type=Path)
    lint.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    lint.add_argument("--json", action="store_true")

    spec = subparsers.add_parser(
        "spec",
        help="Validate a design specification; report open questions instead of defaults",
    )
    spec.add_argument("spec_file", type=Path)
    spec.add_argument("--json", action="store_true")

    explain = subparsers.add_parser("explain", help="Describe resolved topology and assumptions")
    explain.add_argument("model", type=Path)
    explain.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    explain.add_argument("--json", action="store_true")

    render = subparsers.add_parser("render", help="Render a deterministic SVG schematic")
    render.add_argument("model", type=Path)
    render.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    render.add_argument("--out", type=Path, required=True)
    render.add_argument("--json", action="store_true")
    render.add_argument(
        "--expand-subcircuits",
        action="store_true",
        help=(
            "Draw the flattened components of every subcircuit instance "
            "instead of the default box-with-ports view"
        ),
    )

    simulate = subparsers.add_parser("simulate", help="Run a transient simulation")
    simulate.add_argument("model", type=Path)
    simulate.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    simulate.add_argument("--out", type=Path, required=True)
    simulate.add_argument("--review", type=Path, help="Topology review packet directory")
    simulate.add_argument("--decision", type=Path, help="Human-authored review decision JSON")
    simulate.add_argument("--require-review", action="store_true")
    simulate.add_argument(
        "--lean",
        action="store_true",
        help=(
            "Lean run: skip the per-component energy balance and the dense state "
            "extrema outside the reported window. Everything still reported is "
            "bit-for-bit identical to a full run; the omissions are declared, never "
            "approximated. Omit this flag for the full evidence bundle."
        ),
    )
    simulate.add_argument("--json", action="store_true")

    report = subparsers.add_parser("report", help="Generate an offline human report")
    report.add_argument("run", type=Path)
    report.add_argument("--out", type=Path)
    report.add_argument("--schematic", type=Path)
    report.add_argument("--json", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="Evaluate a verified run against a versioned expectations file",
    )
    evaluate.add_argument("run", type=Path)
    evaluate.add_argument("--expectations", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--json", action="store_true")

    analyze = subparsers.add_parser(
        "analyze",
        help="Run retained analyses (periodic steady state, loop-gain margins)",
    )
    analyze_actions = analyze.add_subparsers(dest="analysis_kind", required=True)
    stability = analyze_actions.add_parser(
        "stability",
        help="Shooting steady state plus declared-compensator loop-gain gates",
    )
    stability.add_argument("model", type=Path)
    stability.add_argument("--request", type=Path, help="Stability request JSON")
    stability.add_argument("--out", type=Path, required=True)
    stability.add_argument("--json", action="store_true")

    compare = subparsers.add_parser(
        "compare",
        help="Compare two manifest-verified runs with a versioned numerical profile",
    )
    compare.add_argument("reference", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--profile", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--json", action="store_true")

    batch = subparsers.add_parser(
        "batch",
        help="Screen many designs (sampled or listed), rank them, re-run the finalists",
    )
    batch.add_argument("model", type=Path)
    batch.add_argument("--request", type=Path, required=True)
    batch.add_argument("--out", type=Path, required=True)
    batch.add_argument(
        "--jobs",
        type=int,
        help="Designs simulated at the same time (default: all cores but one)",
    )
    batch.add_argument("--json", action="store_true")

    sensitivity = subparsers.add_parser(
        "sensitivity",
        help="Global sensitivity (Morris screening or Sobol indices) over parameter ranges",
    )
    sensitivity.add_argument("model", type=Path)
    sensitivity.add_argument("--request", type=Path, required=True)
    sensitivity.add_argument("--out", type=Path, required=True)
    sensitivity.add_argument(
        "--jobs",
        type=int,
        help="Runs simulated at the same time (default: all cores but one)",
    )
    sensitivity.add_argument("--json", action="store_true")

    sweep = subparsers.add_parser(
        "sweep",
        help="Run a bounded, artifact-retaining parameter sweep",
    )
    sweep_actions = sweep.add_subparsers(dest="sweep_kind", required=True)
    parameter = sweep_actions.add_parser(
        "parameter",
        help="Run a bounded one-dimensional model-parameter experiment",
    )
    parameter.add_argument("model", type=Path)
    parameter.add_argument("--request", type=Path, required=True)
    parameter.add_argument("--expectations", type=Path)
    parameter.add_argument("--out", type=Path, required=True)
    parameter.add_argument("--json", action="store_true")
    parameter.add_argument(
        "--jobs",
        type=int,
        help=(
            "Point simulations run at the same time (default: all cores but one). "
            "Results do not depend on it."
        ),
    )
    tolerance_corners = sweep_actions.add_parser(
        "tolerance-corners",
        help="Worst-case-of-the-batch sweep over declared component tolerances",
    )
    tolerance_corners.add_argument("model", type=Path)
    tolerance_corners.add_argument("--request", type=Path, required=True)
    tolerance_corners.add_argument("--expectations", type=Path)
    tolerance_corners.add_argument("--out", type=Path, required=True)
    tolerance_corners.add_argument("--json", action="store_true")
    tolerance_corners.add_argument(
        "--jobs",
        type=int,
        help=(
            "Point simulations run at the same time (default: all cores but one). "
            "Results do not depend on it."
        ),
    )
    perturbation = sweep_actions.add_parser(
        "perturbation",
        help="Run a parametric-corner / event-injection campaign with criteria",
    )
    perturbation.add_argument("model", type=Path)
    perturbation.add_argument("--request", type=Path, required=True)
    perturbation.add_argument("--out", type=Path, required=True)
    perturbation.add_argument("--json", action="store_true")
    perturbation.add_argument(
        "--jobs",
        type=int,
        help=(
            "Point simulations run at the same time (default: all cores but one). "
            "Results do not depend on it."
        ),
    )
    buck_boundary = sweep_actions.add_parser(
        "buck-boundary",
        help="Verify a canonical ideal buck across CCM, BCM, and DCM",
    )
    buck_boundary.add_argument("model", type=Path)
    buck_boundary.add_argument(
        "--ratios",
        metavar="R1,R2,...",
        help="Strictly increasing R/R_BCM values containing exact 1.0",
    )
    buck_boundary.add_argument("--out", type=Path, required=True)
    buck_boundary.add_argument("--json", action="store_true")
    boost_boundary = sweep_actions.add_parser(
        "boost-boundary",
        help="Verify a canonical ideal boost across CCM, BCM, and DCM",
    )
    boost_boundary.add_argument("model", type=Path)
    boost_boundary.add_argument(
        "--ratios",
        metavar="R1,R2,...",
        help="Strictly increasing R/R_BCM values containing exact 1.0",
    )
    boost_boundary.add_argument("--out", type=Path, required=True)
    boost_boundary.add_argument("--json", action="store_true")
    inverting_buck_boost_boundary = sweep_actions.add_parser(
        "inverting-buck-boost-boundary",
        help="Verify a canonical ideal inverting buck-boost across CCM, BCM, and DCM",
    )
    inverting_buck_boost_boundary.add_argument("model", type=Path)
    inverting_buck_boost_boundary.add_argument(
        "--ratios",
        metavar="R1,R2,...",
        help="Strictly increasing R/R_BCM values containing exact 1.0",
    )
    inverting_buck_boost_boundary.add_argument("--out", type=Path, required=True)
    inverting_buck_boost_boundary.add_argument("--json", action="store_true")

    verify = subparsers.add_parser(
        "verify",
        help="Produce retained numerical and cross-platform verification evidence",
    )
    verify_actions = verify.add_subparsers(dest="verify_action", required=True)
    numerics = verify_actions.add_parser(
        "numerics",
        help="Run a versioned numerical evidence matrix",
    )
    numerical_topologies = numerics.add_subparsers(
        dest="numerical_topology",
        required=True,
    )
    numerical_buck = numerical_topologies.add_parser(
        "buck-boundary",
        help="Run the canonical buck numerical tolerance/grid/solver matrix",
    )
    numerical_buck.add_argument("model", type=Path)
    numerical_buck.add_argument("--profile", choices=(FULL_PROFILE_ID,), default=FULL_PROFILE_ID)
    numerical_buck.add_argument("--source-revision")
    numerical_buck.add_argument("--out", type=Path, required=True)
    numerical_buck.add_argument("--json", action="store_true")

    platforms = verify_actions.add_parser(
        "platforms",
        help="Aggregate manifest-verified Linux, macOS, and Windows evidence",
    )
    platforms.add_argument("evidence", nargs="+", type=Path)
    platforms.add_argument(
        "--require",
        default=",".join(REQUIRED_SYSTEMS),
        help="Required platform.system values; the default requires Darwin,Linux,Windows",
    )
    platforms.add_argument("--out", type=Path, required=True)
    platforms.add_argument("--json", action="store_true")

    review = subparsers.add_parser("review", help="Prepare or validate human review packets")
    review_actions = review.add_subparsers(dest="review_action", required=True)

    prepare = review_actions.add_parser("prepare", help="Create an immutable pending packet")
    prepare_phases = prepare.add_subparsers(dest="review_phase", required=True)
    prepare_topology = prepare_phases.add_parser("topology", help="Prepare model review")
    prepare_topology.add_argument("model", type=Path)
    prepare_topology.add_argument("--request", type=Path)
    prepare_topology.add_argument("--expectations", type=Path)
    prepare_topology.add_argument("--out", type=Path, required=True)
    prepare_topology.add_argument("--json", action="store_true")
    prepare_result = prepare_phases.add_parser("result", help="Prepare run/report review")
    prepare_result.add_argument("run", type=Path)
    prepare_result.add_argument("--report", type=Path)
    prepare_result.add_argument("--out", type=Path, required=True)
    prepare_result.add_argument("--json", action="store_true")

    check = review_actions.add_parser("check", help="Verify packet integrity and a decision")
    check.add_argument("packet", type=Path)
    check.add_argument("--decision", type=Path)
    check.add_argument("--subject-digest")
    check.add_argument("--allow-non-approval", action="store_true")
    check.add_argument("--json", action="store_true")

    annotate = review_actions.add_parser("annotate", help="Append a hash-chained annotation")
    annotate.add_argument("packet", type=Path)
    annotate.add_argument("--file", type=Path, required=True)
    annotate.add_argument("--json", action="store_true")

    return parser


def _print_batch(artifacts) -> None:
    summary = json.loads(artifacts.summary.read_text(encoding="utf-8"))
    print(
        f"Batch {artifacts.status}: {artifacts.design_count} designs, "
        f"{artifacts.feasible_count} feasible, {artifacts.failed_count} failed"
    )
    (sense, selector), = summary["objective"].items()
    for entry in summary["ranking"][:5]:
        values = ", ".join(f"{key}={value:.6g}" for key, value in entry["values"].items())
        parameters = ", ".join(
            f"{key}={value:.4g}" for key, value in entry["parameters_si"].items()
        )
        print(f"  #{entry['rank']} {entry['design_id']}  {parameters}  ->  {values}")
    for finalist in summary["finalists"]:
        if finalist["status"] != "pass":
            print(f"  finalist {finalist['design_id']}: {finalist['status']}")
    target = " ".join(str(value) for value in selector.values())
    print(f"  ({sense} {target}; best of the sample, not an optimum)")
    print(f"  table: {artifacts.designs_csv}")


def _print_sensitivity(artifacts) -> None:
    summary = json.loads(artifacts.summary.read_text(encoding="utf-8"))
    print(
        f"Sensitivity ({artifacts.method}) {artifacts.status}: {artifacts.run_count} runs, "
        f"{artifacts.failed_count} failed"
    )
    for output in summary["outputs"]:
        print(f"  {output['output']}")
        if output["status"] != "pass":
            print(f"    not estimated: {output['reason']}")
            continue
        if output.get("note"):
            print(f"    {output['note']}")
        by_name = {item["name"]: item for item in output["parameters"]}
        for name in output["ranking"]:
            item = by_name[name]
            if artifacts.method == "morris":
                low, high = item["mu_star_ci95"]
                print(
                    f"    {name:<16} mu*={item['mu_star']:.4g} [{low:.3g}, {high:.3g}]  "
                    f"sigma={item['sigma']:.3g}"
                )
            else:
                s1_low, s1_high = item["S1_ci95"]
                st_low, st_high = item["ST_ci95"]
                print(
                    f"    {name:<16} ST={item['ST']:.3f} [{st_low:.3f}, {st_high:.3f}]  "
                    f"S1={item['S1']:.3f} [{s1_low:.3f}, {s1_high:.3f}]"
                )
    print("  (over the declared ranges only; see the claim in sensitivity-summary.json)")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "mcp":
            try:
                from pear_sim.mcp.server import serve
            except ImportError:  # pragma: no cover - only without the extra
                print('the MCP server needs the optional SDK: pip install "pear-sim[mcp]"',
                      file=sys.stderr)
                return 2
            return serve(workspace=args.workspace, project_root=args.project_root)
        if args.command == "doctor":
            from pear_sim.api.doctor import run_doctor

            payload = run_doctor()
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"pear-sim {payload['pear_sim_version']} — {payload['status']}")
                print(f"python {payload['python']['version']} on "
                      f"{payload['platform']['system']}/{payload['platform']['machine']}")
                for name, version in sorted(payload["dependencies"].items()):
                    print(f"  dep {name}: {version}")
                for name, value in sorted(payload["package_integrity"].items()):
                    print(f"  integrity {name}: {value}")
                backends = payload["backends"]
                print(f"  ngspice: {backends['ngspice_differential']}")
                probe = payload["float_determinism_probe"]
                print(f"  float probe digest: {probe['digest']}")
                print(f"  limits: {len(payload['limits'])} enforced (see --json)")
            return 0 if payload["status"] == "ok" else 1
        if args.command == "topology":
            from pear_sim.topology import load_topology_library, verify_entry_sequence

            entries = load_topology_library(args.dir)
            if args.topology_action == "list":
                payload = {
                    "entries": [
                        {
                            "id": entry.id,
                            "name": entry.name,
                            "isolation": entry.prerequisites.isolation,
                            "direction": entry.prerequisites.direction,
                            "mode_sequences": [
                                {
                                    "label": item.label,
                                    "source_model": item.source_model,
                                    "steps": len(item.sequence.steps),
                                    "source_run_sha256": item.sequence.source_run_sha256,
                                }
                                for item in entry.mode_sequences
                            ],
                            "known_limitations": len(entry.known_limitations),
                        }
                        for entry in entries
                    ]
                }
                if args.json:
                    print(json.dumps(payload, indent=2))
                else:
                    for entry in entries:
                        print(f"{entry.id:<16} {entry.name}")
                return 0
            if args.topology_action == "compare":
                from pear_sim._jsonutil import write_json
                from pear_sim.topology.comparison import run_comparison_request

                request = json.loads(Path(args.request).read_text(encoding="utf-8"))
                record = run_comparison_request(request, entries)
                if args.out is not None:
                    write_json(Path(args.out), record)
                if args.json:
                    print(json.dumps(record, indent=2))
                else:
                    summary = record["summary"]
                    print(
                        f"{summary['ranked']} ranked / {summary['not_ranked']} "
                        f"not ranked; {summary['non_dominated']} on the front"
                    )
                    for point in record["points"]:
                        mark = "FRONT" if not point["dominated"] else "     "
                        flags = (
                            f"  [{', '.join(point['flags'])}]" if point["flags"] else ""
                        )
                        print(f"  {mark} {point['topology_id']}{flags}")
                        for metric, value in sorted(point["objectives"].items()):
                            print(f"          {metric} = {value:.6g}")
                    for missing in record["not_ranked"]:
                        print(
                            f"  ---   {missing['topology_id']}: "
                            f"{missing['status']} — {missing['reason']}"
                        )
                    if summary["mixed_loss_instrumentation"]:
                        print(
                            "  note: the field is UNEVENLY instrumented for "
                            "losses; candidates without cards are marked "
                            "OPTIMISTIC because a model with no cards reports "
                            "no dissipation at all"
                        )
                # Nothing rankable is a blocking outcome, not a success: the
                # question "which of these is better?" went unanswered.
                return 0 if record["summary"]["ranked"] else 1
            verdicts = []
            for entry in entries:
                for item in entry.mode_sequences:
                    agreed, reasons = verify_entry_sequence(item, root=args.root)
                    verdicts.append(
                        {
                            "id": entry.id,
                            "label": item.label,
                            "agreed": agreed,
                            "reasons": list(reasons),
                        }
                    )
            all_agreed = all(verdict["agreed"] for verdict in verdicts)
            if args.json:
                print(json.dumps({"ok": all_agreed, "sequences": verdicts}, indent=2))
            else:
                for verdict in verdicts:
                    status = "ok" if verdict["agreed"] else "DISAGREES"
                    print(f"{verdict['id']}.{verdict['label']}: {status}")
                    for reason in verdict["reasons"]:
                        print(f"  {reason}")
            return 0 if all_agreed else 1
        if args.command == "catalog":
            payload = {
                "components": [
                    "resistor",
                    "capacitor",
                    "inductor",
                    "sat_inductor",
                    "voltage_source",
                    "current_source",
                    "ideal_switch",
                    "diode",
                    "voltage_probe",
                ],
                "controls": ["pwm", "complementary_pwm", "three_phase_spwm"],
                "analyses": ["transient"],
                "solvers": ["dopri5", "dop853", "radau", "bdf"],
            }
            print(json.dumps(payload, indent=2) if args.json else "\n".join(payload["components"]))
            return 0
        if args.command == "capabilities":
            payload = get_capability(args.id) if args.id is not None else build_capabilities()
            if args.json:
                print(json.dumps(payload, indent=2))
            elif args.id is not None:
                print(
                    f"{payload['status']}: {payload['id']}\n"
                    f"Scope: {payload['scope']}\n"
                    f"{payload['statement']}"
                )
            else:
                for record in payload["capabilities"]:
                    print(f"{record['status']:<12} {record['id']}")
            return 0
        if args.command == "analyze":
            artifacts = run_stability_analysis(
                args.model, args.out, request=args.request
            )
            payload = {
                "ok": True,
                "overall_status": artifacts.overall_status,
                "steady_state_json": str(artifacts.steady_state_json),
                "loop_gain_json": (
                    None if artifacts.loop_gain_json is None else str(artifacts.loop_gain_json)
                ),
                "evidence_json": str(artifacts.evidence_json),
                "artifact_manifest": str(artifacts.artifact_manifest),
            }
            print(
                json.dumps(payload, indent=2)
                if args.json
                else (
                    f"Stability evidence: {artifacts.evidence_json} "
                    f"(overall_status={artifacts.overall_status})"
                )
            )
            return 0 if artifacts.overall_status == "pass" else 1
        if args.command == "evaluate":
            artifacts = evaluate_run(args.run, args.expectations, args.out)
            accepted = artifacts.blocking_status == "pass"
            payload = {
                "ok": True,
                "accepted": accepted,
                "blocking_status": artifacts.blocking_status,
                "evidence_json": str(artifacts.evidence_json),
                "confidence_json": str(artifacts.confidence_json),
                "artifact_manifest": str(artifacts.artifact_manifest),
            }
            print(
                json.dumps(payload, indent=2)
                if args.json
                else (
                    f"Evidence: {artifacts.evidence_json} "
                    f"(blocking_status={artifacts.blocking_status})"
                )
            )
            return 0 if accepted else 1
        if args.command == "compare":
            artifacts = compare_runs(
                args.reference,
                args.candidate,
                args.profile,
                args.out,
            )
            accepted = artifacts.status == "pass"
            payload = {
                "ok": True,
                "accepted": accepted,
                "status": artifacts.status,
                "output": str(artifacts.output_dir),
                "resolved_profile": str(artifacts.resolved_profile),
                "comparison_json": str(artifacts.comparison_json),
                "report_html": str(artifacts.report_html),
                "artifact_manifest": str(artifacts.artifact_manifest),
            }
            print(
                json.dumps(payload, indent=2)
                if args.json
                else (
                    f"Comparison: {artifacts.comparison_json} "
                    f"(status={artifacts.status})"
                )
            )
            return 0 if accepted else 1
        if args.command == "verify":
            if args.verify_action == "numerics":
                artifacts = run_buck_numerical_evidence(
                    args.model,
                    args.out,
                    source_revision=args.source_revision,
                )
                accepted = artifacts.status == "pass"
                payload = {
                    "ok": True,
                    "accepted": accepted,
                    "status": artifacts.status,
                    "profile": args.profile,
                    "output": str(artifacts.output_dir),
                    "platform_evidence": str(artifacts.platform_evidence),
                    "report_html": str(artifacts.report_html),
                    "artifact_manifest": str(artifacts.artifact_manifest),
                }
                print(
                    json.dumps(payload, indent=2)
                    if args.json
                    else (
                        f"Numerical evidence: {artifacts.platform_evidence} "
                        f"(status={artifacts.status})"
                    )
                )
                return 0 if accepted else 1

            _required_systems(args.require)
            artifacts = aggregate_platform_evidence(args.evidence, args.out)
            accepted = artifacts.overall_status == "pass"
            payload = {
                "ok": True,
                "accepted": accepted,
                "status": artifacts.overall_status,
                "output": str(args.out.resolve()),
                "cross_platform_evidence": str(artifacts.evidence_json),
                "report_html": str(artifacts.report_html),
                "artifact_manifest": str(artifacts.artifact_manifest),
            }
            print(
                json.dumps(payload, indent=2)
                if args.json
                else (
                    f"Cross-platform evidence: {artifacts.evidence_json} "
                    f"(status={artifacts.overall_status})"
                )
            )
            return 0 if accepted else 1
        if args.command == "batch":
            from pear_sim.evidence.batch import run_batch

            artifacts = run_batch(args.model, args.request, args.out, jobs=args.jobs)
            if args.json:
                print(
                    json.dumps(
                        {
                            "ok": True,
                            "status": artifacts.status,
                            "design_count": artifacts.design_count,
                            "feasible_count": artifacts.feasible_count,
                            "failed_count": artifacts.failed_count,
                            "best_design": artifacts.best_design,
                            "output": str(artifacts.output_dir),
                            "summary": str(artifacts.summary),
                            "designs_csv": str(artifacts.designs_csv),
                        },
                        indent=2,
                    )
                )
            else:
                _print_batch(artifacts)
            return 0 if artifacts.status == "pass" else 1
        if args.command == "sensitivity":
            from pear_sim.evidence.sensitivity import run_sensitivity

            artifacts = run_sensitivity(args.model, args.request, args.out, jobs=args.jobs)
            if args.json:
                print(
                    json.dumps(
                        {
                            "ok": True,
                            "status": artifacts.status,
                            "method": artifacts.method,
                            "run_count": artifacts.run_count,
                            "failed_count": artifacts.failed_count,
                            "output": str(artifacts.output_dir),
                            "summary": str(artifacts.summary),
                            "runs_csv": str(artifacts.runs_csv),
                        },
                        indent=2,
                    )
                )
            else:
                _print_sensitivity(artifacts)
            return 0 if artifacts.status in {"pass", "pass_with_failed_runs"} else 1
        if args.command == "sweep":
            if args.sweep_kind == "perturbation":
                from pear_sim.evidence.perturbation_campaign import run_campaign

                artifacts = run_campaign(
                    args.model, args.request, args.out, jobs=args.jobs
                )
                payload = {
                    "ok": True,
                    "run_count": artifacts.run_count,
                    "execution_status": artifacts.execution_status,
                    "acceptance_status": artifacts.acceptance_status,
                    "overall_status": artifacts.overall_status,
                    "output": str(artifacts.output_dir),
                    "summary": str(artifacts.summary),
                    "report": str(artifacts.report_html),
                }
                print(
                    json.dumps(payload, indent=2)
                    if args.json
                    else artifacts.overall_status
                )
                return 0 if artifacts.overall_status == "pass" else 1
            if args.sweep_kind == "tolerance-corners":
                artifacts = run_tolerance_corners(
                    args.model,
                    args.request,
                    args.out,
                    expectations_file=args.expectations,
                    jobs=args.jobs,
                )
                payload = {
                    "ok": True,
                    "execution_status": artifacts.execution_status,
                    "acceptance_status": artifacts.acceptance_status,
                    "overall_status": artifacts.overall_status,
                    "output": str(artifacts.output_dir),
                    "summary": str(artifacts.summary),
                }
                print(
                    json.dumps(payload, indent=2)
                    if args.json
                    else artifacts.overall_status
                )
                return 0 if artifacts.overall_status == "pass" else 1
            if args.sweep_kind == "parameter":
                artifacts = run_parameter_sweep(
                    args.model,
                    args.request,
                    args.out,
                    expectations_file=args.expectations,
                    jobs=args.jobs,
                )
                successful = (
                    artifacts.execution_status == "pass"
                    and artifacts.overall_status == "pass"
                )
                payload = {
                    "ok": True,
                    "accepted": artifacts.accepted,
                    "execution_status": artifacts.execution_status,
                    "acceptance_status": artifacts.acceptance_status,
                    "overall_status": artifacts.overall_status,
                    "output": str(artifacts.output_dir),
                    "source_request": str(artifacts.source_request),
                    "normalized_request": str(artifacts.normalized_request),
                    "sweep_summary": str(artifacts.sweep_summary),
                    "projections": str(artifacts.projections),
                    "failures": str(artifacts.failures),
                    "confidence_json": str(artifacts.confidence_json),
                    "report_html": str(artifacts.report_html),
                    "artifact_manifest": str(artifacts.artifact_manifest),
                }
                print(
                    json.dumps(payload, indent=2)
                    if args.json
                    else (
                        f"Parameter sweep: {artifacts.sweep_summary} "
                        f"(overall_status={artifacts.overall_status})"
                    )
                )
                return 0 if successful else 1
            if args.sweep_kind == "boost-boundary":
                runner = run_boost_boundary_benchmark
                default_ratios = BOOST_DEFAULT_LOAD_RATIOS
            elif args.sweep_kind == "inverting-buck-boost-boundary":
                runner = run_inverting_buck_boost_boundary_benchmark
                default_ratios = INVERTING_BUCK_BOOST_DEFAULT_LOAD_RATIOS
            else:
                runner = run_buck_boundary_benchmark
                default_ratios = DEFAULT_LOAD_RATIOS
            artifacts = runner(
                args.model,
                args.out,
                load_ratios=_load_ratios(args.ratios, default_ratios),
            )
            accepted = artifacts.overall_status == "pass"
            payload = {
                "ok": True,
                "accepted": accepted,
                "overall_status": artifacts.overall_status,
                "output": str(artifacts.output_dir),
                "sweep_summary": str(artifacts.sweep_summary),
                "benchmark_result": str(artifacts.benchmark_result),
                "confidence_json": str(artifacts.confidence_json),
                "report_html": str(artifacts.report_html),
                "artifact_manifest": str(artifacts.artifact_manifest),
            }
            if args.sweep_kind in {
                "boost-boundary",
                "inverting-buck-boost-boundary",
            }:
                payload["schematic_svg"] = (
                    None if artifacts.schematic_svg is None else str(artifacts.schematic_svg)
                )
            print(
                json.dumps(payload, indent=2)
                if args.json
                else (
                    f"Sweep: {artifacts.sweep_summary} (overall_status={artifacts.overall_status})"
                )
            )
            return 0 if accepted else 1
        if args.command == "report":
            artifacts = generate_report(args.run, args.out, args.schematic)
            payload = {
                "ok": True,
                "report_json": str(artifacts.report_json),
                "report_html": str(artifacts.report_html),
                "artifact_manifest": str(artifacts.artifact_manifest),
                "schematic_svg": (
                    None if artifacts.schematic_svg is None else str(artifacts.schematic_svg)
                ),
            }
            print(
                json.dumps(payload, indent=2) if args.json else f"Report: {artifacts.report_html}"
            )
            return 0
        if args.command == "review":
            if args.review_action == "prepare":
                if args.review_phase == "topology":
                    packet = prepare_topology_review(
                        args.model,
                        request=args.request,
                        expectations=args.expectations,
                        out_dir=args.out,
                    )
                else:
                    packet = prepare_result_review(
                        args.run,
                        report_dir=args.report,
                        out_dir=args.out,
                    )
                payload = {
                    "ok": True,
                    "packet": str((args.out / "packet.json").resolve()),
                    "review_id": packet["review_id"],
                    "phase": packet["phase"],
                    "status": packet["status"],
                    "subject_digest": packet["subject_digest"],
                }
                print(
                    json.dumps(payload, indent=2)
                    if args.json
                    else f"Review packet: {payload['packet']}"
                )
                return 0
            if args.review_action == "annotate":
                annotation = json.loads(args.file.read_text(encoding="utf-8"))
                if not isinstance(annotation, dict):
                    raise ValueError("Annotation file must contain a JSON object")
                record = append_review_annotation(args.packet, annotation)
                payload = {
                    "ok": True,
                    "review_id": record["review_id"],
                    "annotation_id": record["id"],
                    "sequence": record["sequence"],
                    "annotation_digest": record["annotation_digest"],
                }
                print(
                    json.dumps(payload, indent=2)
                    if args.json
                    else f"Annotation appended: {record['id']}"
                )
                return 0
            packet = check_review_packet(args.packet)
            decision = None
            if args.decision is not None:
                decision = check_review_decision(
                    args.decision,
                    args.packet,
                    current_subject_digest=args.subject_digest,
                    require_approval=not args.allow_non_approval,
                )
            payload = {
                "ok": True,
                "review_id": packet["review_id"],
                "phase": packet["phase"],
                "status": packet["status"],
                "subject_digest": packet["subject_digest"],
                "decision": None if decision is None else decision["decision"],
            }
            print(
                json.dumps(payload, indent=2)
                if args.json
                else f"Review packet valid: {packet['review_id']}"
            )
            return 0
        if args.command == "spec":
            from pear_sim.design import DesignSpec

            design = DesignSpec.load(args.spec_file)
            questions = design.questions()
            payload = design.resolved()
            payload["open_questions"] = [question.as_dict() for question in questions]
            if design.complete:
                payload["spec_sha256"] = design.spec_sha256()
            if args.json:
                print(json.dumps(payload, indent=2, sort_keys=True))
            elif design.complete:
                print(f"COMPLETE: {design.name} ({payload['spec_sha256'][:12]})")
            else:
                print(f"INCOMPLETE: {design.name} — {len(questions)} open question(s)")
                for question in questions:
                    print(f"  ? {question.field}: {question.why}")
            return 0 if design.complete else 1
        overrides = _overrides(args.set)
        model = load_model(args.model, overrides)
        if args.command == "lint":
            simulator = HybridSimulator(model)
            topology = simulator.validate_initial_topology()
            explanation = explain_model(model)
            payload = {
                "ok": True,
                "model": model.name,
                "components": len(model.components),
                **topology,
                "outputs": list(model.outputs),
                "hashes": explanation["hashes"],
                "limitations": explanation["limitations"],
            }
            print(json.dumps(payload, indent=2) if args.json else f"OK: {model.name}")
            return 0
        if args.command == "explain":
            payload = explain_model(model)
            print(json.dumps(payload, indent=2) if args.json else explanation_text(payload))
            return 0
        if args.command == "render":
            result = render_schematic(
                model, expand_subcircuits=args.expand_subcircuits
            )
            if args.out.suffix.lower() == ".svg":
                svg_path = args.out.resolve()
                svg_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_path = svg_path.with_suffix(".json")
            else:
                output_dir = args.out.resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                svg_path = output_dir / "schematic.svg"
                metadata_path = output_dir / "schematic.json"
            svg_path.write_text(result.svg, encoding="utf-8")
            metadata = {
                "schema": "pear-sim/schematic/v0.1",
                **asdict(result.metadata),
                "svg": svg_path.name,
                # Additive: present only when the collapsed view drew boxes,
                # so subcircuit-free schematics keep their exact bytes.
                **(
                    {
                        "subcircuits": [
                            asdict(box) for box in result.subcircuit_boxes
                        ]
                    }
                    if result.subcircuit_boxes
                    else {}
                ),
            }
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            payload = {
                "ok": True,
                "svg": str(svg_path),
                "metadata": str(metadata_path),
                "warnings": list(result.metadata.warnings),
            }
            print(json.dumps(payload, indent=2) if args.json else f"Schematic: {svg_path}")
            return 0
        topology_review = None
        if args.require_review and args.review is None:
            raise ValueError("--require-review needs --review and --decision")
        if (args.review is None) != (args.decision is None):
            raise ValueError("--review and --decision must be provided together")
        if args.review is not None:
            if args.set:
                raise ValueError(
                    "Reviewed simulation does not allow --set; prepare a review for the resolved model"
                )
            packet = check_review_packet(args.review)
            if packet["phase"] != "topology":
                raise ValueError("Simulation requires a topology review packet")
            review_path = args.review.resolve()
            review_base = review_path if review_path.is_dir() else review_path.parent
            expectation_record = next(
                (item for item in packet["artifacts"] if item["role"] == "expectations"),
                None,
            )
            expectations_path = (
                None if expectation_record is None else review_base / expectation_record["path"]
            )
            # Recompute the model and the request that will actually be simulated,
            # while retaining the immutable expectations already bound into the
            # reviewed packet. A custom request that the simulator would not consume
            # therefore fails stale instead of silently authorizing different work.
            subject_digest = compute_topology_subject_digest(
                args.model,
                expectations=expectations_path,
            )
            decision = check_review_decision(
                args.decision,
                args.review,
                current_subject_digest=subject_digest,
            )
            topology_review = {
                "review_id": packet["review_id"],
                "subject_digest": packet["subject_digest"],
                "packet_digest": packet["packet_digest"],
                "decision": decision["decision"],
                "reviewer_id": decision["reviewer"]["id"],
                "decided_at": decision["decided_at"],
                "scope": decision["scope"],
            }
        evidence_mode = "lean" if getattr(args, "lean", False) else "full"
        result = HybridSimulator(model, evidence_mode=evidence_mode).simulate()
        destination = write_result(result, args.out, topology_review=topology_review)
        payload = {
            "ok": True,
            "output": str(destination),
            "evidence_mode": evidence_mode,
            "stats": result.stats,
            "metrics": result.metrics(),
        }
        if args.json:
            print(json.dumps(payload, indent=2))
        elif evidence_mode == "lean":
            # The mode is never inferred from silence: a lean run says so on the
            # console, in warnings.json, in summary.json and in run.json.
            print(f"Simulation complete (LEAN run, not an evidence bundle): {destination}")
            print(
                "  not computed: per-component energy balance (physics-audit.json "
                "status=not_computed), dense state extrema outside the reported window"
            )
            print("  re-run without --lean for the full evidence bundle")
        else:
            print(f"Simulation complete: {destination}")
        return 0
    except (PearSimError, OSError, ValueError) as exc:
        payload = {
            "ok": False,
            "error": type(exc).__name__,
            "code": error_code(exc),
            "message": str(exc),
        }
        if getattr(args, "json", False):
            print(json.dumps(payload), file=sys.stderr)
        else:
            print(f"pear-sim: {payload['error']}: {payload['message']}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
