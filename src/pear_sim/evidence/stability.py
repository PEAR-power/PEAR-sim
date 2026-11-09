# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Retained stability-analysis bundles: steady state + loop-gain gates.

``run_stability_analysis`` combines the shooting steady state with the
cycle-averaged loop-gain sweep and retains a non-overwriting, hash-manifested
bundle. Margins become blocking evidence gates when the request declares
minimums — an auditable-stability artifact: the compensator and sensor gain
are part of the request, so a passing bundle states "this declared design has
at least these margins on this model", nothing more.

Blocking gates:

- ``steady_state_converged`` — always blocking.
- ``ccm_single_pwm_leg`` — blocking whenever a loop-gain sweep was requested.
- ``phase_margin`` / ``gain_margin`` — blocking iff the request declares
  ``min_phase_margin_deg`` / ``min_gain_margin_db``; a declared minimum with
  no crossover on the grid is a ``fail`` (fail closed), with one exception for
  the gain margin: when the sweep covers the model's validity band and the
  phase never reaches -180 deg there, the margin is unbounded within that band
  and the gate passes, saying so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from pear_sim import __version__
from pear_sim.analysis.small_signal import (
    SamplingModel,
    SmallSignalError,
    averaged_small_signal,
    loop_gain,
)
from pear_sim.analysis.steady_state import solve_periodic_steady_state
from pear_sim.evidence.benchmark_harness import prepare_destination, root_manifest, write_json
from pear_sim.run.contracts import model_hashes
from pear_sim.spec.model import load_model

__all__ = ["StabilityArtifacts", "StabilityRequestError", "run_stability_analysis"]

REQUEST_SCHEMA = "pear-sim/stability-request/v0.1"
EVIDENCE_SCHEMA = "pear-sim/stability-evidence/v0.1"
MAX_REQUEST_BYTES = 64 * 1024


class StabilityRequestError(ValueError):
    """The stability request document is invalid."""

    code = "analysis.stability_request_invalid"


@dataclass(frozen=True)
class StabilityArtifacts:
    output_dir: Path
    steady_state_json: Path
    loop_gain_json: Path | None
    evidence_json: Path
    artifact_manifest: Path
    overall_status: str


def _load_request(request: str | Path | Mapping[str, Any] | None) -> dict[str, Any]:
    if request is None:
        return {"schema": REQUEST_SCHEMA}
    if isinstance(request, Mapping):
        payload: Any = dict(request)
    else:
        path = Path(request)
        if path.stat().st_size > MAX_REQUEST_BYTES:
            raise StabilityRequestError(
                f"stability request exceeds the {MAX_REQUEST_BYTES}-byte limit"
            )
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StabilityRequestError("stability request must be a JSON object")
    allowed = {
        "schema",
        "output",
        "compensator",
        "sensor_gain",
        "f_min_hz",
        "f_max_hz",
        "points",
        "margins",
        "sampling",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise StabilityRequestError(
            "stability request contains unknown field(s): " + ", ".join(unknown)
        )
    if payload.get("schema", REQUEST_SCHEMA) != REQUEST_SCHEMA:
        raise StabilityRequestError(
            f"stability request schema must be {REQUEST_SCHEMA!r}"
        )
    compensator = payload.get("compensator", {})
    if not isinstance(compensator, dict) or set(compensator) - {"numerator", "denominator"}:
        raise StabilityRequestError(
            "compensator must be an object with numerator/denominator arrays"
        )
    margins = payload.get("margins", {})
    if not isinstance(margins, dict) or set(margins) - {
        "min_phase_margin_deg",
        "min_gain_margin_db",
    }:
        raise StabilityRequestError(
            "margins must declare only min_phase_margin_deg / min_gain_margin_db"
        )
    sampling = payload.get("sampling")
    if sampling is not None:
        if not isinstance(sampling, dict) or set(sampling) - {
            "update_frequency_hz",
            "computation_delay_s",
        }:
            raise StabilityRequestError(
                "sampling must be an object with only update_frequency_hz / "
                "computation_delay_s"
            )
    return payload


def _gate(gate_id: str, status: str, blocking: bool, detail: str) -> dict[str, Any]:
    return {"id": gate_id, "status": status, "blocking": blocking, "detail": detail}


def run_stability_analysis(
    model_file: str | Path,
    output_dir: str | Path,
    *,
    request: str | Path | Mapping[str, Any] | None = None,
) -> StabilityArtifacts:
    payload = _load_request(request)
    model = load_model(model_file)
    destination = prepare_destination(output_dir)

    steady_state = solve_periodic_steady_state(model)
    steady_path = destination / "steady-state.json"
    write_json(steady_path, steady_state.as_dict())

    gates: list[dict[str, Any]] = [
        _gate(
            "steady_state_converged",
            "pass" if steady_state.converged else "fail",
            True,
            f"residual_inf={steady_state.residual_inf!r} after "
            f"{steady_state.iterations} iteration(s), tolerance "
            f"{steady_state.tolerance!r}",
        )
    ]

    loop_requested = any(
        key in payload
        for key in ("output", "compensator", "sensor_gain", "margins", "sampling")
    )
    loop_path: Path | None = None
    loop_error: str | None = None
    if loop_requested:
        output = payload.get("output")
        if not isinstance(output, str) or not output:
            raise StabilityRequestError(
                "a loop-gain request must name the plant output, e.g. \"v(out)\""
            )
        if output not in model.outputs:
            raise StabilityRequestError(
                f"requested output {output!r} is not among the model outputs "
                f"{list(model.outputs)}"
            )
        compensator = payload.get("compensator", {})
        margins = payload.get("margins", {})
        declared_sampling = payload.get("sampling")
        try:
            small_signal = averaged_small_signal(model, steady_state, output)
            sampling = None
            if declared_sampling is not None:
                # ZOH at f_sw unless the request declares a slower update.
                sampling = SamplingModel(
                    update_frequency_hz=float(
                        declared_sampling.get(
                            "update_frequency_hz",
                            small_signal.switching_frequency_hz,
                        )
                    ),
                    computation_delay_s=float(
                        declared_sampling.get("computation_delay_s", 0.0)
                    ),
                )
            result = loop_gain(
                small_signal,
                compensator_numerator=tuple(
                    float(value) for value in compensator.get("numerator", (1.0,))
                ),
                compensator_denominator=tuple(
                    float(value) for value in compensator.get("denominator", (1.0,))
                ),
                sensor_gain=float(payload.get("sensor_gain", 1.0)),
                f_min_hz=payload.get("f_min_hz"),
                f_max_hz=payload.get("f_max_hz"),
                points=int(payload.get("points", 200)),
                sampling=sampling,
            )
        except SmallSignalError as exc:
            loop_error = str(exc)
            gates.append(_gate("ccm_single_pwm_leg", "fail", True, loop_error))
        else:
            gates.append(
                _gate(
                    "ccm_single_pwm_leg",
                    "pass",
                    True,
                    "CCM cycle of one PWM leg (dead-time intervals averaged in) with zero natural events",
                )
            )
            loop_path = destination / "loop-gain.json"
            body = result.as_dict()
            body["small_signal"] = small_signal.as_dict()
            write_json(loop_path, body)

            minimum_pm = margins.get("min_phase_margin_deg")
            if minimum_pm is not None:
                if result.phase_margin_deg is None:
                    gates.append(
                        _gate(
                            "phase_margin",
                            "fail",
                            True,
                            "no gain crossover on the requested grid; declared "
                            f"minimum {float(minimum_pm)!r} deg cannot be attested",
                        )
                    )
                else:
                    ok = result.phase_margin_deg >= float(minimum_pm)
                    gates.append(
                        _gate(
                            "phase_margin",
                            "pass" if ok else "fail",
                            True,
                            f"phase margin {result.phase_margin_deg!r} deg vs "
                            f"declared minimum {float(minimum_pm)!r} deg",
                        )
                    )
            minimum_gm = margins.get("min_gain_margin_db")
            if minimum_gm is not None:
                covers_validity = result.frequencies_hz[-1] >= result.validity_max_hz
                stays_above = min(result.loop_phase_deg) > -180.0
                if result.gain_margin_db is None and covers_validity and stays_above:
                    # The whole band where the averaged model is valid was swept
                    # and the phase never reaches -180 deg: within the model's
                    # validity the gain can rise without bound before the loop
                    # loses stability through a phase crossover.
                    gates.append(
                        _gate(
                            "gain_margin",
                            "pass",
                            True,
                            "phase stays above -180 deg over the whole sweep, which "
                            "covers the model's validity band (up to "
                            f"{result.validity_max_hz!r} Hz): gain margin unbounded "
                            "within the validity band",
                        )
                    )
                elif result.gain_margin_db is None:
                    reason = (
                        "the sweep stops below the model's validity limit "
                        f"{result.validity_max_hz!r} Hz — extend f_max_hz to reach it"
                        if not covers_validity
                        else "the phase starts below -180 deg"
                    )
                    gates.append(
                        _gate(
                            "gain_margin",
                            "fail",
                            True,
                            f"no phase crossover on the requested grid and {reason}; "
                            f"declared minimum {float(minimum_gm)!r} dB cannot be attested",
                        )
                    )
                else:
                    ok = result.gain_margin_db >= float(minimum_gm)
                    gates.append(
                        _gate(
                            "gain_margin",
                            "pass" if ok else "fail",
                            True,
                            f"gain margin {result.gain_margin_db!r} dB vs "
                            f"declared minimum {float(minimum_gm)!r} dB",
                        )
                    )

    overall = "pass"
    if any(gate["status"] == "fail" and gate["blocking"] for gate in gates):
        overall = "fail"

    evidence_path = destination / "stability-evidence.json"
    write_json(
        evidence_path,
        {
            "schema": EVIDENCE_SCHEMA,
            "kind": "stability-analysis",
            "pear_sim_version": __version__,
            "model": str(Path(model_file)),
            "model_hashes": model_hashes(model),
            "request": payload,
            "gates": gates,
            "overall_status": overall,
            "loop_gain_error": loop_error,
            "claim_boundary": (
                "shooting steady state on the reference solver plus two-topology "
                "state-space averaging; margins apply to the declared compensator, "
                "sensor gain, and sampling model (continuous when not declared) "
                "only, below switching_frequency/10, for the exact resolved model"
            ),
        },
    )
    manifest_path = root_manifest(
        destination, evidence_path, subject_kind="stability-analysis"
    )
    return StabilityArtifacts(
        output_dir=destination,
        steady_state_json=steady_path,
        loop_gain_json=loop_path,
        evidence_json=evidence_path,
        artifact_manifest=manifest_path,
        overall_status=overall,
    )
