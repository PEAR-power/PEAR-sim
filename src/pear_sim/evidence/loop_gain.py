# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Retained loop-gain-by-injection bundles.

``run_loop_gain_analysis`` measures the loop gain of a closed-loop model on
its switching circuit (see :mod:`pear_sim.analysis.loop_gain`) and retains a
non-overwriting, hash-manifested bundle: ``loop_gain.json`` (points,
crossovers, margins, the orbit's largest multiplier, settle and amplitude)
and ``loop-gain-evidence.json`` (gate and claim boundary).

Everything that would make the numbers meaningless is refused BEFORE a
bundle is written: an orbit that is not locally stable, a limiter engaging
during an injection, an injection that changes the switching mode sequence.

Blocking gate:

- ``gain_crossover_in_sweep`` — |L| crosses 1 inside the requested band, so
  a phase margin exists. A sweep that misses the crossover fails rather than
  reporting a margin it never saw.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from pear_sim import __version__
from pear_sim.analysis.loop_gain import measure_loop_gain
from pear_sim.evidence.benchmark_harness import prepare_destination, root_manifest, write_json
from pear_sim.run.contracts import model_hashes
from pear_sim.spec.model import load_model

__all__ = ["LoopGainArtifacts", "run_loop_gain_analysis"]

EVIDENCE_SCHEMA = "pear-sim/loop-gain-evidence/v0.1"


@dataclass(frozen=True)
class LoopGainArtifacts:
    output_dir: Path
    loop_gain_json: Path
    evidence_json: Path
    artifact_manifest: Path
    phase_margin_deg: float | None
    gain_margin: float | None
    overall_status: str


def run_loop_gain_analysis(
    model_file: str | Path,
    output_dir: str | Path,
    frequencies_hz: Iterable[float],
    *,
    inject_at: str | None = None,
    amplitude: float | None = None,
    initial_state: Mapping[str, float] | None = None,
    overrides: Mapping[str, str] | None = None,
) -> LoopGainArtifacts:
    model = load_model(model_file, None if not overrides else dict(overrides))
    result = measure_loop_gain(
        model,
        list(frequencies_hz),
        inject_at=inject_at,
        amplitude=amplitude,
        initial_state=None if initial_state is None else dict(initial_state),
    )
    destination = prepare_destination(output_dir)
    data_path = destination / "loop_gain.json"
    write_json(data_path, result.as_dict())

    low = result.points[0].frequency_hz
    high = result.points[-1].frequency_hz
    found = bool(result.gain_crossovers)
    gates = [
        {
            "id": "gain_crossover_in_sweep",
            "status": "pass" if found else "fail",
            "blocking": True,
            "detail": (
                f"|L| = 1 at {result.gain_crossovers[0].frequency_hz!r} Hz"
                if found
                else f"|L| stays on one side of 1 over {low!r}-{high!r} Hz; widen the band"
            ),
        }
    ]
    overall = "pass" if all(gate["status"] == "pass" for gate in gates) else "fail"
    evidence_path = destination / "loop-gain-evidence.json"
    write_json(
        evidence_path,
        {
            "schema": EVIDENCE_SCHEMA,
            "kind": "closed-loop-gain-injection",
            "pear_sim_version": __version__,
            "model": str(Path(model_file)),
            "model_hashes": model_hashes(model),
            "injection_block": result.injection_block,
            "initial_state": None if initial_state is None else dict(initial_state),
            "overrides": None if not overrides else dict(overrides),
            "gates": gates,
            "phase_margin_deg": result.phase_margin_deg,
            "gain_margin": result.gain_margin,
            "overall_status": overall,
            "claim_boundary": (
                "small-signal loop gain of the computed closed-loop periodic orbit "
                "of the exact resolved model at the injection point, on the "
                "reference solver; not large-signal behaviour, not robustness to "
                "parameter changes; verified against Floquet multipliers, not "
                "validated against hardware"
            ),
        },
    )
    manifest_path = root_manifest(
        destination, evidence_path, subject_kind="closed-loop-gain-injection"
    )
    return LoopGainArtifacts(
        output_dir=destination,
        loop_gain_json=data_path,
        evidence_json=evidence_path,
        artifact_manifest=manifest_path,
        phase_margin_deg=result.phase_margin_deg,
        gain_margin=result.gain_margin,
        overall_status=overall,
    )
