# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Bounded benchmark runners that retain every requested operating point."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from pathlib import Path
from typing import Any, Callable, Sequence

from pear_sim import __version__
from pear_sim.evidence.benchmark_harness import (
    BoundaryHarnessContract,
    aggregate_boundary_points,
    build_boundary_confidence,
    build_boundary_runner_result,
    build_boundary_spec,
    build_boundary_sweep_summary,
    prepare_destination as _prepare_destination,
    regime_coverage,
    root_manifest as _root_manifest,
    run_retained_boundary_points,
    write_json as _write_json,
)
from pear_sim.reporting.benchmark_report import SweepReportArtifacts, generate_sweep_report
from pear_sim.benchmarks import (
    BOOST_DEFAULT_LOAD_RATIOS,
    BUCK_BOUNDARY_ORACLE_SCHEMA,
    BUCK_BOUNDARY_RESULT_SCHEMA,
    DEFAULT_LOAD_RATIOS,
    INVERTING_BUCK_BOOST_BOUNDARY_ORACLE_SCHEMA,
    INVERTING_BUCK_BOOST_BOUNDARY_RESULT_SCHEMA,
    INVERTING_BUCK_BOOST_DEFAULT_LOAD_RATIOS,
    INVERTING_BUCK_BOOST_MAX_DIMENSIONLESS_CAPACITOR_DECAY,
    INVERTING_BUCK_BOOST_MAX_DIMENSIONLESS_RESONANT_FREQUENCY,
    INVERTING_BUCK_BOOST_MAX_LOAD_RATIO,
    INVERTING_BUCK_BOOST_MIN_LOAD_RATIO,
    INVERTING_BUCK_BOOST_MIN_REGIME_SEPARATION_NORMALIZED,
    BoostBoundaryObservation,
    BoostBoundaryPoint,
    BuckBoundaryObservation,
    BuckBoundaryPoint,
    InvertingBuckBoostBoundaryObservation,
    InvertingBuckBoostBoundaryPoint,
    compare_boost_boundary_point,
    compare_buck_boundary_point,
    compare_inverting_buck_boost_boundary_point,
    extract_boost_boundary_parameters,
    extract_buck_boundary_parameters,
    extract_inverting_buck_boost_boundary_parameters,
    observe_boost_simulation_result,
    observe_inverting_buck_boost_simulation_result,
    observe_simulation_result,
    run_boost_boundary_sweep,
    run_buck_boundary_sweep,
    run_inverting_buck_boost_boundary_sweep,
)
from pear_sim.run.contracts import model_hashes, sha256_file
from pear_sim.run.io import write_result
from pear_sim.spec.model import Component, Model, load_model
from pear_sim.reporting.render import render_schematic
from pear_sim.core.solver import HybridSimulator, SimulationResult
from pear_sim.spec.units import Quantity


BUCK_BOUNDARY_SPEC_SCHEMA = "pear-sim/benchmark-spec/ideal-buck-boundary/v0.1"
BUCK_BOUNDARY_SWEEP_SCHEMA = "pear-sim/sweep-summary/ideal-buck-boundary/v0.1"
BUCK_BOUNDARY_POINT_SCHEMA = "pear-sim/benchmark-point/ideal-buck-boundary/v0.1"
BUCK_BOUNDARY_RUNNER_RESULT_SCHEMA = "pear-sim/benchmark-runner-result/v0.1"
SWEEP_CONFIDENCE_SCHEMA = "pear-sim/confidence/buck-boundary-sweep/v0.1"
BOOST_BOUNDARY_ORACLE_SCHEMA = "pear-sim/benchmark-oracle/ideal-boost-boundary/v0.1"
BOOST_BOUNDARY_RESULT_SCHEMA = "pear-sim/benchmark-result/ideal-boost-boundary/v0.1"
BOOST_BOUNDARY_SPEC_SCHEMA = "pear-sim/benchmark-spec/ideal-boost-boundary/v0.1"
BOOST_BOUNDARY_SWEEP_SCHEMA = "pear-sim/sweep-summary/ideal-boost-boundary/v0.1"
BOOST_BOUNDARY_POINT_SCHEMA = "pear-sim/benchmark-point/ideal-boost-boundary/v0.1"
BOOST_BOUNDARY_RUNNER_RESULT_SCHEMA = (
    "pear-sim/benchmark-runner-result/ideal-boost-boundary/v0.1"
)
BOOST_SWEEP_CONFIDENCE_SCHEMA = "pear-sim/confidence/boost-boundary-sweep/v0.1"
INVERTING_BUCK_BOOST_BOUNDARY_SPEC_SCHEMA = (
    "pear-sim/benchmark-spec/ideal-inverting-buck-boost-boundary/v0.1"
)
INVERTING_BUCK_BOOST_BOUNDARY_SWEEP_SCHEMA = (
    "pear-sim/sweep-summary/ideal-inverting-buck-boost-boundary/v0.1"
)
INVERTING_BUCK_BOOST_BOUNDARY_POINT_SCHEMA = (
    "pear-sim/benchmark-point/ideal-inverting-buck-boost-boundary/v0.1"
)
INVERTING_BUCK_BOOST_BOUNDARY_RUNNER_RESULT_SCHEMA = (
    "pear-sim/benchmark-runner-result/ideal-inverting-buck-boost-boundary/v0.1"
)
INVERTING_BUCK_BOOST_SWEEP_CONFIDENCE_SCHEMA = (
    "pear-sim/confidence/inverting-buck-boost-boundary-sweep/v0.1"
)
MAX_SWEEP_POINTS = 64
MAX_INVERTING_BUCK_BOOST_SOURCE_BYTES = 2 * 1024 * 1024
MAX_INVERTING_BUCK_BOOST_COMPONENTS = 38
MAX_INVERTING_BUCK_BOOST_VOLTAGE_PROBES = 32
MAX_INVERTING_BUCK_BOOST_SOURCE_OUTPUTS = 60
MAX_INVERTING_BUCK_BOOST_POINT_OUTPUTS = 64

_BUCK_HARNESS = BoundaryHarnessContract(
    point_schema=BUCK_BOUNDARY_POINT_SCHEMA,
    comparison_schema=BUCK_BOUNDARY_RESULT_SCHEMA,
    gate_ids=(
        "regime",
        "boundary_state_nonzero",
        "boundary_state_zero",
        "natural_event_time",
        "reverse_current",
        "cycle_mean_output",
        "mode_fractions",
        "energy_balance",
        "cycle_periodicity",
        "unclassified_fraction",
        "mode_fraction_sum",
    ),
    failure_includes_comparison_schema=False,
)
_BOOST_HARNESS = BoundaryHarnessContract(
    point_schema=BOOST_BOUNDARY_POINT_SCHEMA,
    comparison_schema=BOOST_BOUNDARY_RESULT_SCHEMA,
    gate_ids=(
        "regime",
        "boundary_state_nonzero",
        "boundary_state_zero",
        "natural_event_time",
        "reverse_current",
        "cycle_mean_output",
        "state_ripple",
        "mode_fractions",
        "event_semantics",
        "energy_balance",
        "cycle_periodicity",
        "unclassified_fraction",
        "mode_fraction_sum",
    ),
)
_INVERTING_BUCK_BOOST_HARNESS = BoundaryHarnessContract(
    point_schema=INVERTING_BUCK_BOOST_BOUNDARY_POINT_SCHEMA,
    comparison_schema=INVERTING_BUCK_BOOST_BOUNDARY_RESULT_SCHEMA,
    gate_ids=(
        "regime",
        "boundary_state_nonzero",
        "boundary_state_zero",
        "natural_event_time",
        "reverse_current",
        "signed_cycle_mean_output",
        "signed_output_polarity",
        "state_ripple",
        "mode_fractions",
        "event_semantics",
        "energy_balance",
        "cycle_periodicity",
        "unclassified_fraction",
        "mode_fraction_sum",
    ),
    bounded_error_text=True,
)


@dataclass(frozen=True)
class BuckBoundarySweepArtifacts:
    output_dir: Path
    benchmark_spec: Path
    oracle_json: Path
    benchmark_result: Path
    sweep_summary: Path
    confidence_json: Path
    artifact_manifest: Path
    report_json: Path
    report_html: Path
    overall_status: str


@dataclass(frozen=True)
class BoostBoundarySweepArtifacts:
    output_dir: Path
    benchmark_spec: Path
    oracle_json: Path
    benchmark_result: Path
    sweep_summary: Path
    confidence_json: Path
    artifact_manifest: Path
    report_json: Path
    report_html: Path
    schematic_svg: Path | None
    overall_status: str


@dataclass(frozen=True)
class InvertingBuckBoostBoundarySweepArtifacts:
    output_dir: Path
    benchmark_spec: Path
    oracle_json: Path
    benchmark_result: Path
    sweep_summary: Path
    confidence_json: Path
    artifact_manifest: Path
    report_json: Path
    report_html: Path
    schematic_svg: Path | None
    overall_status: str


def _canonical_fixture(model: Model) -> tuple[Component, Component, Component, Component]:
    by_type = {component.type: component for component in model.components}
    return (
        by_type["resistor"],
        by_type["inductor"],
        by_type["capacitor"],
        by_type["diode"],
    )


def periodic_point_model(
    source: Model,
    reference: BuckBoundaryPoint,
) -> Model:
    load, inductor, capacitor, diode = _canonical_fixture(source)
    components: list[Component] = []
    for component in source.components:
        if component.id == load.id:
            component = replace(component, value=reference.load_ohm)
        elif component.id == inductor.id:
            component = replace(
                component,
                initial_state=reference.start_state.inductor_current_a,
                # The seed is generated by the benchmark oracle, not declared in
                # the source model.  Keep the generic run provenance conservative;
                # the exact origin is recorded in benchmark-point.json.
                initial_state_explicit=False,
            )
        elif component.id == capacitor.id:
            component = replace(
                component,
                initial_state=reference.start_state.output_voltage_v,
                initial_state_explicit=False,
            )
        components.append(component)

    parameters = dict(source.parameters)
    for name, quantity in source.parameters.items():
        if quantity.dimension == "resistance" and math.isclose(
            quantity.value,
            float(load.value),
            rel_tol=0.0,
            abs_tol=max(1e-15, abs(float(load.value)) * 1e-15),
        ):
            parameters[name] = Quantity(reference.load_ohm, "resistance")

    period = next(iter(source.controls.values())).period
    output_node = inductor.n
    required_outputs = (
        f"v({output_node})",
        f"i({inductor.id})",
        f"i({diode.id})",
        f"i({load.id})",
    )
    outputs = tuple(dict.fromkeys((*source.outputs, *required_outputs)))
    return replace(
        source,
        name=f"{source.name}__{reference.point_id}",
        parameters=parameters,
        components=tuple(components),
        transient=replace(
            source.transient,
            start=0.0,
            stop=period,
            output_step=period / 200.0,
            solver="dopri5",
            rtol=1e-10,
            atol=1e-12,
            max_step=period / 100.0,
        ),
        outputs=outputs,
    )


# Deprecated alias kept for one release; use ``periodic_point_model``.
_periodic_point_model = periodic_point_model


def _boost_periodic_point_model(
    source: Model,
    reference: BoostBoundaryPoint,
) -> Model:
    """Build one oracle-seeded period without broadening the source topology."""

    load, inductor, capacitor, diode = _canonical_fixture(source)
    components: list[Component] = []
    for component in source.components:
        if component.id == load.id:
            component = replace(component, value=reference.load_ohm)
        elif component.id == inductor.id:
            component = replace(
                component,
                initial_state=reference.start_state.inductor_current_a,
                initial_state_explicit=False,
            )
        elif component.id == capacitor.id:
            component = replace(
                component,
                initial_state=reference.start_state.output_voltage_v,
                initial_state_explicit=False,
            )
        components.append(component)

    parameters = dict(source.parameters)
    for name, quantity in source.parameters.items():
        if quantity.dimension == "resistance" and math.isclose(
            quantity.value,
            float(load.value),
            rel_tol=0.0,
            abs_tol=max(1e-15, abs(float(load.value)) * 1e-15),
        ):
            parameters[name] = Quantity(reference.load_ohm, "resistance")

    period = next(iter(source.controls.values())).period
    output_node = capacitor.p
    required_outputs = (
        f"v({output_node})",
        f"i({inductor.id})",
        f"i({diode.id})",
        f"i({load.id})",
    )
    outputs = tuple(dict.fromkeys((*source.outputs, *required_outputs)))
    return replace(
        source,
        name=f"{source.name}__{reference.point_id}",
        parameters=parameters,
        components=tuple(components),
        transient=replace(
            source.transient,
            start=0.0,
            stop=period,
            output_step=period / 200.0,
            solver="dopri5",
            rtol=1e-10,
            atol=1e-12,
            max_step=period / 100.0,
        ),
        outputs=outputs,
    )


def _inverting_buck_boost_periodic_point_model(
    source: Model,
    reference: InvertingBuckBoostBoundaryPoint,
) -> Model:
    """Build one oracle-seeded period for the exact inverting buck-boost slice."""

    load, inductor, capacitor, _diode = _canonical_fixture(source)
    components: list[Component] = []
    for component in source.components:
        if component.id == load.id:
            component = replace(component, value=reference.load_ohm)
        elif component.id == inductor.id:
            component = replace(
                component,
                initial_state=reference.start_state.inductor_current_a,
                initial_state_explicit=False,
            )
        elif component.id == capacitor.id:
            component = replace(
                component,
                # C is oriented gnd->out, so its positive state is -v(out).
                initial_state=reference.start_state.output_magnitude_voltage_v,
                initial_state_explicit=False,
            )
        components.append(component)

    parameters = dict(source.parameters)
    for name, quantity in source.parameters.items():
        if quantity.dimension == "resistance" and math.isclose(
            quantity.value,
            float(load.value),
            rel_tol=0.0,
            abs_tol=max(1e-15, abs(float(load.value)) * 1e-15),
        ):
            parameters[name] = Quantity(reference.load_ohm, "resistance")

    period = next(iter(source.controls.values())).period
    output_node = capacitor.n
    required_outputs = (
        f"v({output_node})",
        f"i({inductor.id})",
        f"i({_diode.id})",
        f"i({load.id})",
    )
    outputs = tuple(dict.fromkeys((*source.outputs, *required_outputs)))
    if len(outputs) > MAX_INVERTING_BUCK_BOOST_POINT_OUTPUTS:
        raise ValueError(
            "Resolved point exceeds the bounded output limit of "
            f"{MAX_INVERTING_BUCK_BOOST_POINT_OUTPUTS}"
        )
    return replace(
        source,
        name=f"{source.name}__{reference.point_id}",
        parameters=parameters,
        components=tuple(components),
        transient=replace(
            source.transient,
            start=0.0,
            stop=period,
            output_step=period / 200.0,
            solver="dopri5",
            rtol=1e-10,
            atol=1e-12,
            max_step=period / 100.0,
        ),
        outputs=outputs,
    )


def _noninverting_summary_metrics(
    observation: BuckBoundaryObservation | BoostBoundaryObservation | None,
) -> dict[str, Any]:
    if observation is None:
        return {
            "mean_output_voltage_v": None,
            "minimum_inductor_current_a": None,
            "conduction_fraction": None,
            "fraction_by_mode": None,
            "energy_balance_relative": None,
        }
    return {
        "mean_output_voltage_v": observation.mean_output_voltage_v,
        "minimum_inductor_current_a": observation.minimum_inductor_current_a,
        "conduction_fraction": observation.conduction_fraction,
        "fraction_by_mode": dict(observation.fraction_by_mode),
        "energy_balance_relative": observation.energy_balance_relative,
    }


def _inverting_summary_metrics(
    observation: InvertingBuckBoostBoundaryObservation | None,
) -> dict[str, Any]:
    if observation is None:
        return {
            "signed_mean_output_voltage_v": None,
            "maximum_signed_output_voltage_v": None,
            "minimum_inductor_current_a": None,
            "conduction_fraction": None,
            "fraction_by_mode": None,
            "energy_balance_relative": None,
        }
    return {
        "signed_mean_output_voltage_v": observation.signed_mean_output_voltage_v,
        "maximum_signed_output_voltage_v": observation.maximum_signed_output_voltage_v,
        "minimum_inductor_current_a": observation.minimum_inductor_current_a,
        "conduction_fraction": observation.conduction_fraction,
        "fraction_by_mode": dict(observation.fraction_by_mode),
        "energy_balance_relative": observation.energy_balance_relative,
    }


@dataclass(frozen=True)
class _BoundaryFamily:
    """Per-topology configuration of the shared retained boundary-sweep runner.

    Topology physics stays in the oracle modules; this record wires one
    oracle's extractor, sweep, observation, and gate comparison into the
    topology-neutral retention trunk below.
    """

    benchmark_id: str
    topology_label: str
    subject_kind: str
    spec_schema: str
    oracle_schema: str
    comparison_schema: str
    runner_result_schema: str
    sweep_schema: str
    confidence_schema: str
    harness_contract: BoundaryHarnessContract
    load_source: Callable[[str | Path], tuple[Model, dict[str, Any] | None]]
    extract_parameters: Callable[[Model], Any]
    run_oracle_sweep: Callable[[Any, tuple[float, ...]], Any]
    point_model: Callable[[Model, Any], Model]
    observe: Callable[..., Any]
    compare: Callable[[Any, Any, Any], dict[str, Any]]
    initial_conditions: Callable[[Any], dict[str, Any]]
    summary_metrics: Callable[[Any], dict[str, Any]]
    spec_limits: dict[str, Any]
    spec_extra_fields: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]] | None
    confidence_envelope_extra: (
        Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]] | None
    )
    schematic_caption: str | None
    summary_schematic_pointer: bool
    build_artifacts: Callable[..., Any]


def _run_boundary_benchmark(
    family: _BoundaryFamily,
    model_file: str | Path,
    output_dir: str | Path,
    load_ratios: Sequence[float],
) -> Any:
    ratios = tuple(float(value) for value in load_ratios)
    if len(ratios) > MAX_SWEEP_POINTS:
        raise ValueError(f"Sweep exceeds the {MAX_SWEEP_POINTS}-point contract limit")
    source_model, source_work = family.load_source(model_file)
    parameters = family.extract_parameters(source_model)
    oracle = family.run_oracle_sweep(parameters, ratios)
    oracle_payload = oracle.as_dict()
    source_schematic_svg = (
        render_schematic(source_model).svg if family.schematic_caption is not None else None
    )
    destination = _prepare_destination(output_dir)

    spec_path = destination / "benchmark-spec.json"
    oracle_path = destination / "oracle.json"
    benchmark_result_path = destination / "benchmark-result.json"
    summary_path = destination / "sweep-summary.json"
    confidence_path = destination / "confidence.json"
    spec = build_boundary_spec(
        schema=family.spec_schema,
        benchmark_id=family.benchmark_id,
        claim=oracle_payload["claim"],
        source_model={
            "file": source_model.source_path.name,
            "hashes": model_hashes(source_model),
        },
        parameters_si=parameters.as_dict(),
        load_ratios=ratios,
        limits=family.spec_limits,
        oracle_schema=family.oracle_schema,
        comparison_schema=family.comparison_schema,
        **(
            {"extra_fields": family.spec_extra_fields(oracle_payload, source_work)}
            if family.spec_extra_fields is not None
            else {}
        ),
    )
    _write_json(spec_path, spec)
    _write_json(oracle_path, oracle_payload)

    def simulate_point(reference: Any) -> SimulationResult:
        return HybridSimulator(family.point_model(source_model, reference)).simulate()

    def observe_point(result: SimulationResult, reference: Any) -> Any:
        return family.observe(
            result,
            reference,
            parameters=parameters,
            bcm_load_ohm=oracle.bcm_load_ohm,
        )

    def compare_point(reference: Any, observation: Any) -> dict[str, Any]:
        return family.compare(parameters, reference, observation)

    retained = run_retained_boundary_points(
        destination,
        oracle.points,
        contract=family.harness_contract,
        simulate=simulate_point,
        observe=observe_point,
        compare=compare_point,
        retain_run=write_result,
        initial_conditions=family.initial_conditions,
        observation_payload=asdict,
        summary_metrics=family.summary_metrics,
    )
    aggregate = aggregate_boundary_points(retained)
    benchmark_result = build_boundary_runner_result(
        schema=family.runner_result_schema,
        benchmark_id=family.benchmark_id,
        oracle_schema=family.oracle_schema,
        comparison_schema=family.comparison_schema,
        claim=oracle_payload["claim"],
        parameters_si=parameters.as_dict(),
        boundary_load_ohm=oracle.bcm_load_ohm,
        retained=retained,
        aggregate=aggregate,
    )
    _write_json(benchmark_result_path, benchmark_result)

    artifacts_index: dict[str, Any] = {
        "benchmark_spec": {"path": spec_path.name, "sha256": sha256_file(spec_path)},
        "oracle": {"path": oracle_path.name, "sha256": sha256_file(oracle_path)},
        "benchmark_result": {
            "path": benchmark_result_path.name,
            "sha256": sha256_file(benchmark_result_path),
        },
        "confidence": {"path": confidence_path.name},
        "human_report": {"path": "report/sweep-report.html"},
    }
    if family.summary_schematic_pointer:
        artifacts_index["schematic"] = {"path": "report/schematic.svg"}
    artifacts_index["artifact_manifest"] = {"path": "artifact-manifest.json"}
    summary = build_boundary_sweep_summary(
        schema=family.sweep_schema,
        benchmark_id=family.benchmark_id,
        claim=oracle_payload["claim"],
        boundary_load_ohm=oracle.bcm_load_ohm,
        retained=retained,
        aggregate=aggregate,
        artifacts=artifacts_index,
        provenance={
            "pear_sim_version": __version__,
            "source_model_hashes": model_hashes(source_model),
            "spec_sha256": sha256_file(spec_path),
            "oracle_sha256": sha256_file(oracle_path),
            "benchmark_result_sha256": sha256_file(benchmark_result_path),
            "network_access": False,
        },
    )
    _write_json(summary_path, summary)
    requested_regimes, completed_regimes = regime_coverage(oracle.points, retained.summaries)
    confidence = build_boundary_confidence(
        schema=family.confidence_schema,
        status=aggregate.status,
        sweep_summary_sha256=sha256_file(summary_path),
        benchmark_result_sha256=sha256_file(benchmark_result_path),
        topology=family.topology_label,
        requested_regimes=requested_regimes,
        completed_regimes=completed_regimes,
        load_ratios=ratios,
        parameters_si=parameters.as_dict(),
        **(
            {"envelope_extra": family.confidence_envelope_extra(oracle_payload, source_work)}
            if family.confidence_envelope_extra is not None
            else {}
        ),
    )
    _write_json(confidence_path, confidence)
    report_kwargs: dict[str, Any] = {}
    if family.schematic_caption is not None:
        report_kwargs["schematic_svg"] = source_schematic_svg
        report_kwargs["schematic_caption"] = family.schematic_caption
    report: SweepReportArtifacts = generate_sweep_report(
        summary,
        destination / "report",
        source_artifact_sha256=sha256_file(summary_path),
        **report_kwargs,
    )
    manifest_path = _root_manifest(
        destination,
        summary_path,
        subject_kind=family.subject_kind,
    )
    return family.build_artifacts(
        destination=destination,
        spec_path=spec_path,
        oracle_path=oracle_path,
        benchmark_result_path=benchmark_result_path,
        summary_path=summary_path,
        confidence_path=confidence_path,
        manifest_path=manifest_path,
        report=report,
        status=aggregate.status,
    )


def _observe_buck_point(result: SimulationResult, reference: Any, **kwargs: Any) -> Any:
    # Late binding: resolve the module-level name at call time so tests can
    # monkeypatch observe_simulation_result on this module.
    return observe_simulation_result(result, reference, **kwargs)


def _observe_boost_point(result: SimulationResult, reference: Any, **kwargs: Any) -> Any:
    return observe_boost_simulation_result(result, reference, **kwargs)


def _observe_inverting_buck_boost_point(
    result: SimulationResult, reference: Any, **kwargs: Any
) -> Any:
    return observe_inverting_buck_boost_simulation_result(result, reference, **kwargs)


def _plain_source(model_file: str | Path) -> tuple[Model, None]:
    return load_model(model_file), None


def _bounded_inverting_buck_boost_source(
    model_file: str | Path,
) -> tuple[Model, dict[str, Any]]:
    source_path = Path(model_file).resolve()
    try:
        source_bytes = source_path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Cannot inspect source model {source_path}: {exc}") from exc
    if source_bytes > MAX_INVERTING_BUCK_BOOST_SOURCE_BYTES:
        raise ValueError(
            "Source model exceeds the bounded inverting buck-boost input limit of "
            f"{MAX_INVERTING_BUCK_BOOST_SOURCE_BYTES} bytes"
        )
    source_model = load_model(source_path)
    component_count = len(source_model.components)
    voltage_probe_count = sum(
        component.type == "voltage_probe" for component in source_model.components
    )
    output_count = len(source_model.outputs)
    if component_count > MAX_INVERTING_BUCK_BOOST_COMPONENTS:
        raise ValueError(
            f"Source model has {component_count} components; bounded runner limit is "
            f"{MAX_INVERTING_BUCK_BOOST_COMPONENTS}"
        )
    if voltage_probe_count > MAX_INVERTING_BUCK_BOOST_VOLTAGE_PROBES:
        raise ValueError(
            f"Source model has {voltage_probe_count} voltage probes; bounded runner limit is "
            f"{MAX_INVERTING_BUCK_BOOST_VOLTAGE_PROBES}"
        )
    if output_count > MAX_INVERTING_BUCK_BOOST_SOURCE_OUTPUTS:
        raise ValueError(
            f"Source model has {output_count} outputs; bounded source-model limit is "
            f"{MAX_INVERTING_BUCK_BOOST_SOURCE_OUTPUTS}"
        )
    source_work = {
        "source_model_bytes": source_bytes,
        "component_count": component_count,
        "voltage_probe_count": voltage_probe_count,
        "output_count": output_count,
    }
    return source_model, source_work


def _noninverting_initial_conditions(reference: Any) -> dict[str, Any]:
    return {
        "source": "independent periodic affine oracle",
        "inductor_current_a": reference.start_state.inductor_current_a,
        "output_voltage_v": reference.start_state.output_voltage_v,
    }


def _inverting_initial_conditions(reference: Any) -> dict[str, Any]:
    return {
        "source": "independent periodic affine oracle",
        "inductor_current_a": reference.start_state.inductor_current_a,
        "output_magnitude_voltage_v": reference.start_state.output_magnitude_voltage_v,
        "signed_output_voltage_v": reference.start_state.signed_output_voltage_v,
    }


def _inverting_spec_extra_fields(
    oracle_payload: dict[str, Any], source_work: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "coordinate_convention": {
            "oracle_state": "x=[iL,u], u=v(gnd)-v(out)=-v(out)",
            "reported_output": "v(out) is negative in the supported operating envelope",
        },
        "source_work": source_work,
        "solver_work": oracle_payload["solver_work_contract"],
        "regime_separation": oracle_payload["regime_separation_contract"],
    }


def _inverting_confidence_envelope_extra(
    oracle_payload: dict[str, Any], source_work: dict[str, Any] | None
) -> dict[str, Any]:
    return {
        "supported_load_ratio_to_bcm": [
            INVERTING_BUCK_BOOST_MIN_LOAD_RATIO,
            INVERTING_BUCK_BOOST_MAX_LOAD_RATIO,
        ],
        "output_polarity": "v(out) < 0",
        "source_work": source_work,
        "solver_work": oracle_payload["solver_work_contract"],
        "regime_separation": oracle_payload["regime_separation_contract"],
    }


def _build_buck_artifacts(*, destination, spec_path, oracle_path, benchmark_result_path,
                          summary_path, confidence_path, manifest_path, report, status):
    return BuckBoundarySweepArtifacts(
        output_dir=destination,
        benchmark_spec=spec_path,
        oracle_json=oracle_path,
        benchmark_result=benchmark_result_path,
        sweep_summary=summary_path,
        confidence_json=confidence_path,
        artifact_manifest=manifest_path,
        report_json=report.report_json,
        report_html=report.report_html,
        overall_status=status,
    )


def _build_boost_artifacts(*, destination, spec_path, oracle_path, benchmark_result_path,
                           summary_path, confidence_path, manifest_path, report, status):
    return BoostBoundarySweepArtifacts(
        output_dir=destination,
        benchmark_spec=spec_path,
        oracle_json=oracle_path,
        benchmark_result=benchmark_result_path,
        sweep_summary=summary_path,
        confidence_json=confidence_path,
        artifact_manifest=manifest_path,
        report_json=report.report_json,
        report_html=report.report_html,
        schematic_svg=report.schematic_svg,
        overall_status=status,
    )


def _build_inverting_buck_boost_artifacts(*, destination, spec_path, oracle_path,
                                          benchmark_result_path, summary_path, confidence_path,
                                          manifest_path, report, status):
    return InvertingBuckBoostBoundarySweepArtifacts(
        output_dir=destination,
        benchmark_spec=spec_path,
        oracle_json=oracle_path,
        benchmark_result=benchmark_result_path,
        sweep_summary=summary_path,
        confidence_json=confidence_path,
        artifact_manifest=manifest_path,
        report_json=report.report_json,
        report_html=report.report_html,
        schematic_svg=report.schematic_svg,
        overall_status=status,
    )


_BUCK_FAMILY = _BoundaryFamily(
    benchmark_id="VAL-003:canonical-buck-boundary-sweep",
    topology_label="canonical single-switch ideal buck",
    subject_kind="buck-boundary-sweep",
    spec_schema=BUCK_BOUNDARY_SPEC_SCHEMA,
    oracle_schema=BUCK_BOUNDARY_ORACLE_SCHEMA,
    comparison_schema=BUCK_BOUNDARY_RESULT_SCHEMA,
    runner_result_schema=BUCK_BOUNDARY_RUNNER_RESULT_SCHEMA,
    sweep_schema=BUCK_BOUNDARY_SWEEP_SCHEMA,
    confidence_schema=SWEEP_CONFIDENCE_SCHEMA,
    harness_contract=_BUCK_HARNESS,
    load_source=_plain_source,
    extract_parameters=extract_buck_boundary_parameters,
    run_oracle_sweep=run_buck_boundary_sweep,
    point_model=periodic_point_model,
    observe=_observe_buck_point,
    compare=compare_buck_boundary_point,
    initial_conditions=_noninverting_initial_conditions,
    summary_metrics=_noninverting_summary_metrics,
    spec_limits={
        "maximum_points": MAX_SWEEP_POINTS,
        "retain_failed_points": True,
        "network_access": False,
    },
    spec_extra_fields=None,
    confidence_envelope_extra=None,
    schematic_caption=None,
    summary_schematic_pointer=False,
    build_artifacts=_build_buck_artifacts,
)


_BOOST_FAMILY = _BoundaryFamily(
    benchmark_id="VAL-003:canonical-boost-boundary-sweep",
    topology_label="canonical single-switch ideal boost",
    subject_kind="boost-boundary-sweep",
    spec_schema=BOOST_BOUNDARY_SPEC_SCHEMA,
    oracle_schema=BOOST_BOUNDARY_ORACLE_SCHEMA,
    comparison_schema=BOOST_BOUNDARY_RESULT_SCHEMA,
    runner_result_schema=BOOST_BOUNDARY_RUNNER_RESULT_SCHEMA,
    sweep_schema=BOOST_BOUNDARY_SWEEP_SCHEMA,
    confidence_schema=BOOST_SWEEP_CONFIDENCE_SCHEMA,
    harness_contract=_BOOST_HARNESS,
    load_source=_plain_source,
    extract_parameters=extract_boost_boundary_parameters,
    run_oracle_sweep=run_boost_boundary_sweep,
    point_model=_boost_periodic_point_model,
    observe=_observe_boost_point,
    compare=compare_boost_boundary_point,
    initial_conditions=_noninverting_initial_conditions,
    summary_metrics=_noninverting_summary_metrics,
    spec_limits={
        "maximum_points": MAX_SWEEP_POINTS,
        "retain_failed_points": True,
        "network_access": False,
    },
    spec_extra_fields=None,
    confidence_envelope_extra=None,
    schematic_caption=(
        "Nominal source-model topology template. The retained point runs replace "
        "the load resistance with each recorded R/R_BCM value."
    ),
    summary_schematic_pointer=False,
    build_artifacts=_build_boost_artifacts,
)


_INVERTING_BUCK_BOOST_FAMILY = _BoundaryFamily(
    benchmark_id="VAL-003:canonical-inverting-buck-boost-boundary-sweep",
    topology_label="canonical single-switch ideal inverting buck-boost",
    subject_kind="inverting-buck-boost-boundary-sweep",
    spec_schema=INVERTING_BUCK_BOOST_BOUNDARY_SPEC_SCHEMA,
    oracle_schema=INVERTING_BUCK_BOOST_BOUNDARY_ORACLE_SCHEMA,
    comparison_schema=INVERTING_BUCK_BOOST_BOUNDARY_RESULT_SCHEMA,
    runner_result_schema=INVERTING_BUCK_BOOST_BOUNDARY_RUNNER_RESULT_SCHEMA,
    sweep_schema=INVERTING_BUCK_BOOST_BOUNDARY_SWEEP_SCHEMA,
    confidence_schema=INVERTING_BUCK_BOOST_SWEEP_CONFIDENCE_SCHEMA,
    harness_contract=_INVERTING_BUCK_BOOST_HARNESS,
    load_source=_bounded_inverting_buck_boost_source,
    extract_parameters=extract_inverting_buck_boost_boundary_parameters,
    run_oracle_sweep=run_inverting_buck_boost_boundary_sweep,
    point_model=_inverting_buck_boost_periodic_point_model,
    observe=_observe_inverting_buck_boost_point,
    compare=compare_inverting_buck_boost_boundary_point,
    initial_conditions=_inverting_initial_conditions,
    summary_metrics=_inverting_summary_metrics,
    spec_limits={
        "maximum_points": MAX_SWEEP_POINTS,
        "minimum_load_ratio_to_bcm": INVERTING_BUCK_BOOST_MIN_LOAD_RATIO,
        "maximum_load_ratio_to_bcm": INVERTING_BUCK_BOOST_MAX_LOAD_RATIO,
        "maximum_dimensionless_capacitor_decay": (
            INVERTING_BUCK_BOOST_MAX_DIMENSIONLESS_CAPACITOR_DECAY
        ),
        "maximum_dimensionless_resonant_frequency": (
            INVERTING_BUCK_BOOST_MAX_DIMENSIONLESS_RESONANT_FREQUENCY
        ),
        "minimum_regime_separation_normalized_exclusive": (
            INVERTING_BUCK_BOOST_MIN_REGIME_SEPARATION_NORMALIZED
        ),
        "maximum_source_model_bytes": MAX_INVERTING_BUCK_BOOST_SOURCE_BYTES,
        "maximum_components": MAX_INVERTING_BUCK_BOOST_COMPONENTS,
        "maximum_voltage_probes": MAX_INVERTING_BUCK_BOOST_VOLTAGE_PROBES,
        "maximum_source_outputs": MAX_INVERTING_BUCK_BOOST_SOURCE_OUTPUTS,
        "maximum_point_outputs": MAX_INVERTING_BUCK_BOOST_POINT_OUTPUTS,
        "retain_failed_points": True,
        "network_access": False,
    },
    spec_extra_fields=_inverting_spec_extra_fields,
    confidence_envelope_extra=_inverting_confidence_envelope_extra,
    schematic_caption=(
        "Nominal inverting buck-boost topology. The capacitor and load are oriented "
        "gnd-to-out, so the plotted v(out) is negative; each retained point replaces "
        "the load resistance with its recorded R/R_BCM value."
    ),
    summary_schematic_pointer=True,
    build_artifacts=_build_inverting_buck_boost_artifacts,
)


def run_buck_boundary_benchmark(
    model_file: str | Path,
    output_dir: str | Path,
    *,
    load_ratios: Sequence[float] = DEFAULT_LOAD_RATIOS,
) -> BuckBoundarySweepArtifacts:
    """Run and retain an independently referenced canonical-buck boundary sweep."""

    return _run_boundary_benchmark(_BUCK_FAMILY, model_file, output_dir, load_ratios)


def run_boost_boundary_benchmark(
    model_file: str | Path,
    output_dir: str | Path,
    *,
    load_ratios: Sequence[float] = BOOST_DEFAULT_LOAD_RATIOS,
) -> BoostBoundarySweepArtifacts:
    """Run and retain an independently referenced canonical-boost boundary sweep."""

    return _run_boundary_benchmark(_BOOST_FAMILY, model_file, output_dir, load_ratios)


def run_inverting_buck_boost_boundary_benchmark(
    model_file: str | Path,
    output_dir: str | Path,
    *,
    load_ratios: Sequence[float] = INVERTING_BUCK_BOOST_DEFAULT_LOAD_RATIOS,
) -> InvertingBuckBoostBoundarySweepArtifacts:
    """Retain an independently referenced canonical inverting buck-boost sweep."""

    return _run_boundary_benchmark(
        _INVERTING_BUCK_BOOST_FAMILY, model_file, output_dir, load_ratios
    )


__all__ = [
    "BOOST_BOUNDARY_POINT_SCHEMA",
    "BOOST_BOUNDARY_RUNNER_RESULT_SCHEMA",
    "BOOST_BOUNDARY_SPEC_SCHEMA",
    "BOOST_BOUNDARY_SWEEP_SCHEMA",
    "BOOST_SWEEP_CONFIDENCE_SCHEMA",
    "BUCK_BOUNDARY_POINT_SCHEMA",
    "BUCK_BOUNDARY_RUNNER_RESULT_SCHEMA",
    "BUCK_BOUNDARY_SPEC_SCHEMA",
    "BUCK_BOUNDARY_SWEEP_SCHEMA",
    "INVERTING_BUCK_BOOST_BOUNDARY_POINT_SCHEMA",
    "INVERTING_BUCK_BOOST_BOUNDARY_RUNNER_RESULT_SCHEMA",
    "INVERTING_BUCK_BOOST_BOUNDARY_SPEC_SCHEMA",
    "INVERTING_BUCK_BOOST_BOUNDARY_SWEEP_SCHEMA",
    "INVERTING_BUCK_BOOST_SWEEP_CONFIDENCE_SCHEMA",
    "BoostBoundarySweepArtifacts",
    "BuckBoundarySweepArtifacts",
    "InvertingBuckBoostBoundarySweepArtifacts",
    "MAX_SWEEP_POINTS",
    "run_boost_boundary_benchmark",
    "run_buck_boundary_benchmark",
    "run_inverting_buck_boost_boundary_benchmark",
]
