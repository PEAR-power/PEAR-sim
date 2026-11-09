# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Programmatic agent-facing API.

One import point for driving PEARSim without shelling out to the CLI:

    import pear_sim.api as api

    report = api.lint("buck.yaml")
    run = api.simulate("buck.yaml", "runs/buck")
    evidence = api.evaluate(run.output_dir, "expectations.json", "evidence/buck")

Contract:

- Every function returns a typed result (dataclasses or the documented dict
  contracts) — never printed text.
- Every raised failure carries a stable machine-readable ``code`` attribute
  (see :func:`error_code`); codes are public contract, message text is not.
- Functions delegate to the same library calls the CLI uses; there is no
  API-only behavior. Review-gated simulation (packet + decision binding)
  remains a CLI workflow built from :mod:`pear_sim.review`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pear_sim.api.capabilities import build_capabilities, get_capability
from pear_sim.core.solver import HybridSimulator, SimulationResult
from pear_sim.errors import PearSimError
from pear_sim.evidence.evidence_io import EvidenceArtifacts, evaluate_run
from pear_sim.evidence.parameter_sweep import ParameterSweepArtifacts, run_parameter_sweep
from pear_sim.evidence.run_compare import RunComparisonArtifacts, compare_runs
from pear_sim.evidence.stability import StabilityArtifacts, run_stability_analysis
from pear_sim.evidence.sweep import (
    BoostBoundarySweepArtifacts,
    BuckBoundarySweepArtifacts,
    InvertingBuckBoostBoundarySweepArtifacts,
    run_boost_boundary_benchmark,
    run_buck_boundary_benchmark,
    run_inverting_buck_boost_boundary_benchmark,
)
from pear_sim.analysis.confidence import build_confidence_card
from pear_sim.evidence.tolerance_corners import run_tolerance_corners
from pear_sim.evidence.batch import BatchArtifacts, run_batch
from pear_sim.evidence.sensitivity import SensitivityArtifacts, run_sensitivity
from pear_sim.analysis.losses_thermal import (
    build_losses_thermal_observation,
    rainflow_count,
)
from pear_sim.analysis.pll import PllResult, srf_pll
from pear_sim.analysis.reference_frames import (
    ThreePhaseFrames,
    clarke,
    instantaneous_power,
    park,
    three_phase_frames,
)
from pear_sim.analysis.small_signal import (
    AveragedSmallSignal,
    LoopGainResult,
    SamplingModel,
    averaged_small_signal,
    loop_gain,
)
from pear_sim.analysis.steady_state import (
    PeriodicSteadyState,
    solve_periodic_steady_state,
)
from pear_sim.reporting.render import render_schematic
from pear_sim.reporting.report import ReportArtifacts, generate_report
from pear_sim.run.contracts import explain_model, model_hashes
from pear_sim.run.io import write_result
from pear_sim.spec.model import Model, load_model

__all__ = [
    "BoostBoundarySweepArtifacts",
    "BuckBoundarySweepArtifacts",
    "EvidenceArtifacts",
    "InvertingBuckBoostBoundarySweepArtifacts",
    "LintReport",
    "ParameterSweepArtifacts",
    "ReportArtifacts",
    "RunComparisonArtifacts",
    "SimulationRunBundle",
    "StabilityArtifacts",
    "AveragedSmallSignal",
    "LoopGainResult",
    "SamplingModel",
    "PeriodicSteadyState",
    "get_capability",
    "list_capabilities",
    "compare",
    "error_code",
    "evaluate",
    "explain",
    "hashes",
    "lint",
    "load",
    "render",
    "report",
    "simulate",
    "stability",
    "steady_state",
    "small_signal",
    "loop_gain",
    "clarke",
    "park",
    "three_phase_frames",
    "instantaneous_power",
    "ThreePhaseFrames",
    "srf_pll",
    "PllResult",
    "losses_thermal",
    "rainflow_count",
    "confidence_card",
    "tolerance_corners",
    "sweep_boost_boundary",
    "sweep_buck_boundary",
    "sweep_inverting_buck_boost_boundary",
    "sweep_parameter",
    "evaluate_designs",
    "BatchArtifacts",
    "sensitivity",
    "SensitivityArtifacts",
]


@dataclass(frozen=True)
class LintReport:
    """Static validation summary of one model document."""

    model_name: str
    components: int
    states: int
    nodes: int
    diodes_conducting: int
    outputs: tuple[str, ...]
    hashes: dict[str, str]
    limitations: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class SimulationRunBundle:
    """A completed simulation and its persisted, hash-manifested bundle."""

    output_dir: Path
    result: SimulationResult

    @property
    def run_manifest(self) -> Path:
        return self.output_dir / "run.json"

    @property
    def summary(self) -> Path:
        return self.output_dir / "summary.json"

    def metrics(self) -> dict[str, dict[str, Any]]:
        return self.result.metrics()


@dataclass(frozen=True)
class RenderedSchematic:
    """Deterministic schematic SVG plus machine-readable layout metadata."""

    svg: str
    metadata: Any


def error_code(exc: BaseException) -> str:
    """Stable machine-readable code for any failure raised by this API."""

    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    if isinstance(exc, PearSimError):
        return PearSimError.code
    if isinstance(exc, OSError):
        return "io.error"
    if isinstance(exc, ValueError):
        return "request.invalid"
    return "internal.error"


def _resolve_model(model: str | Path | Model, overrides: dict[str, str] | None) -> Model:
    if isinstance(model, Model):
        if overrides:
            raise ValueError("overrides require a model file path, not a loaded Model")
        return model
    return load_model(model, overrides)


def load(model: str | Path, overrides: dict[str, str] | None = None) -> Model:
    """Load and fully validate a model document (schema + semantics)."""

    return load_model(model, overrides)


def lint(model: str | Path | Model, overrides: dict[str, str] | None = None) -> LintReport:
    """Validate a model and its initial topology without simulating."""

    resolved = _resolve_model(model, overrides)
    topology = HybridSimulator(resolved).validate_initial_topology()
    explanation = explain_model(resolved)
    return LintReport(
        model_name=resolved.name,
        components=len(resolved.components),
        states=topology["states"],
        nodes=topology["nodes"],
        diodes_conducting=topology["diodes_conducting"],
        outputs=tuple(resolved.outputs),
        hashes=dict(explanation["hashes"]),
        limitations=tuple(explanation["limitations"]),
    )


def explain(model: str | Path | Model, overrides: dict[str, str] | None = None) -> dict[str, Any]:
    """Resolved-model explanation: topology, hashes, conventions, limitations."""

    return explain_model(_resolve_model(model, overrides))


def simulate(
    model: str | Path | Model,
    out_dir: str | Path,
    overrides: dict[str, str] | None = None,
) -> SimulationRunBundle:
    """Run a transient simulation and persist the portable run bundle."""

    resolved = _resolve_model(model, overrides)
    result = HybridSimulator(resolved).simulate()
    destination = write_result(result, out_dir)
    return SimulationRunBundle(output_dir=destination, result=result)


def render(model: str | Path | Model, overrides: dict[str, str] | None = None) -> RenderedSchematic:
    """Deterministic schematic for a resolved model (caller decides persistence)."""

    rendered = render_schematic(_resolve_model(model, overrides))
    return RenderedSchematic(svg=rendered.svg, metadata=rendered.metadata)


def report(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    schematic_svg: str | Path | None = None,
) -> ReportArtifacts:
    """Offline human report for a verified run bundle (never recomputes)."""

    return generate_report(run_dir, out_dir, schematic_svg=schematic_svg)


def hashes(model: str | Path | Model, overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Source/model/request hashes of a resolved model."""

    return model_hashes(_resolve_model(model, overrides))


def steady_state(model: str | Path | Model, overrides: dict[str, str] | None = None) -> PeriodicSteadyState:
    """Periodic steady state of the switching cycle via Newton shooting."""

    return solve_periodic_steady_state(_resolve_model(model, overrides))


stability = run_stability_analysis
small_signal = averaged_small_signal
evaluate = evaluate_run
compare = compare_runs
sweep_buck_boundary = run_buck_boundary_benchmark
sweep_boost_boundary = run_boost_boundary_benchmark
sweep_inverting_buck_boost_boundary = run_inverting_buck_boost_boundary_benchmark
sweep_parameter = run_parameter_sweep
list_capabilities = build_capabilities
losses_thermal = build_losses_thermal_observation
confidence_card = build_confidence_card
tolerance_corners = run_tolerance_corners
evaluate_designs = run_batch
sensitivity = run_sensitivity
