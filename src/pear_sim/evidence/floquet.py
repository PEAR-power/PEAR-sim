# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Retained closed-loop Floquet bundles.

``run_floquet_analysis`` solves the closed-loop periodic orbit of a model whose
control law lives in the signal layer (see
:mod:`pear_sim.analysis.closed_loop`) and retains a non-overwriting,
hash-manifested bundle: ``floquet.json`` (orbit, multipliers, monodromy,
assumptions) and ``floquet-evidence.json`` (gates and claim boundary).

Blocking gates, all fail closed:

- ``closed_loop_orbit_converged`` — Newton found the orbit.
- ``multipliers_assessed`` — the map is differentiable at the orbit (no
  limiter or clamp engaged, mode sequence unchanged across the probes) and,
  where the analytic monodromy applies, it agrees with finite differences.
- ``locally_stable`` — every multiplier is inside the unit circle by more than
  its uncertainty. ``unstable`` and ``not_resolved`` both fail: a number that
  cannot be told from 1 does not attest stability.

A passing bundle says "this orbit of this exact model is locally stable" —
not global stability, not robustness to parameter changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from pear_sim import __version__
from pear_sim.analysis.closed_loop import solve_closed_loop_steady_state
from pear_sim.evidence.benchmark_harness import prepare_destination, root_manifest, write_json
from pear_sim.run.contracts import model_hashes
from pear_sim.spec.model import load_model

__all__ = ["FloquetArtifacts", "run_floquet_analysis"]

EVIDENCE_SCHEMA = "pear-sim/floquet-evidence/v0.1"


@dataclass(frozen=True)
class FloquetArtifacts:
    output_dir: Path
    floquet_json: Path
    evidence_json: Path
    artifact_manifest: Path
    stability: str
    overall_status: str


def _gate(gate_id: str, status: str, detail: str) -> dict[str, object]:
    return {"id": gate_id, "status": status, "blocking": True, "detail": detail}


def run_floquet_analysis(
    model_file: str | Path,
    output_dir: str | Path,
    *,
    initial_state: Mapping[str, float] | None = None,
) -> FloquetArtifacts:
    model = load_model(model_file)
    # Scope errors are raised before anything is written: a refused model
    # leaves no half bundle behind.
    result = solve_closed_loop_steady_state(
        model, initial_state=None if initial_state is None else dict(initial_state)
    )
    destination = prepare_destination(output_dir)
    floquet_path = destination / "floquet.json"
    write_json(floquet_path, result.as_dict())

    gates = [
        _gate(
            "closed_loop_orbit_converged",
            "pass" if result.converged else "fail",
            f"residual {result.residual_inf!r} after {result.iterations} Newton iteration(s)",
        )
    ]
    if result.converged:
        assessed = result.stability != "not_assessed"
        gates.append(
            _gate(
                "multipliers_assessed",
                "pass" if assessed else "fail",
                (
                    f"monodromy by {result.monodromy_method}"
                    if assessed
                    else "; ".join(result.notes) or "no multipliers"
                ),
            )
        )
        if assessed:
            largest = result.multipliers[0]
            gates.append(
                _gate(
                    "locally_stable",
                    "pass" if result.stability == "stable" else "fail",
                    f"{result.stability}: largest |mu| {largest.magnitude!r} "
                    f"+- {largest.uncertainty!r} at {largest.frequency_hz!r} Hz",
                )
            )
    overall = "pass" if all(gate["status"] == "pass" for gate in gates) else "fail"

    evidence_path = destination / "floquet-evidence.json"
    write_json(
        evidence_path,
        {
            "schema": EVIDENCE_SCHEMA,
            "kind": "closed-loop-floquet",
            "pear_sim_version": __version__,
            "model": str(Path(model_file)),
            "model_hashes": model_hashes(model),
            "initial_state": None if initial_state is None else dict(initial_state),
            "gates": gates,
            "stability": result.stability,
            "overall_status": overall,
            "claim_boundary": (
                "local stability of the computed closed-loop periodic orbit of the "
                "exact resolved model, on the reference solver; not global "
                "stability, not the absence of other orbits, not robustness to "
                "parameter changes; verification against closed forms, not "
                "validation against hardware"
            ),
        },
    )
    manifest_path = root_manifest(
        destination, evidence_path, subject_kind="closed-loop-floquet"
    )
    return FloquetArtifacts(
        output_dir=destination,
        floquet_json=floquet_path,
        evidence_json=evidence_path,
        artifact_manifest=manifest_path,
        stability=result.stability,
        overall_status=overall,
    )
