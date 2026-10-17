from __future__ import annotations

from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import scipy

from pear_sim import __version__
from pear_sim.analysis.analytic_checks import run_analytic_checks
from pear_sim.run.contracts import model_hashes, resolved_model_dict, sha256_file, sha256_json
from pear_sim.spec.signals import signal_unit
from pear_sim.analysis.gate_drive import build_gate_drive_observation
from pear_sim.analysis.confidence import build_confidence_card
from pear_sim.analysis.losses_thermal import build_losses_thermal_observation
from pear_sim.analysis.mode_coverage import build_mode_coverage
from pear_sim.spec.model import Model
from pear_sim.core.solver import SimulationResult
from pear_sim.analysis.resonance import build_resonance_observation
from pear_sim.analysis.npc_modulation import (
    build_npc_modulation_observation,
    has_three_phase_npc_spwm,
    normalized_npc_modulation_cycles,
)
from pear_sim.analysis.three_phase_modulation import (
    build_three_phase_modulation_observation,
    has_three_phase_spwm,
    normalized_modulation_cycles,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _bound_harmonic_analysis(
    payload: Any,
    *,
    model: Model,
    hashes: dict[str, str],
    run_sha256: str,
) -> dict[str, Any] | None:
    settings = model.harmonics
    if payload is None:
        if settings is not None:
            raise ValueError(
                "Model contains an analysis.harmonics request, but "
                "SimulationResult.harmonic_analysis is missing"
            )
        return None
    if settings is None:
        raise ValueError(
            "SimulationResult.harmonic_analysis is present, but the model contains no "
            "analysis.harmonics request"
        )
    if not isinstance(payload, dict):
        raise ValueError("SimulationResult.harmonic_analysis must be an object or None")
    artifact = deepcopy(payload)
    if artifact.get("schema") != "pear-sim/harmonic-analysis/v0.1":
        raise ValueError("SimulationResult.harmonic_analysis must use pear-sim/harmonic-analysis/v0.1")
    _require_harmonic_request_match(artifact, model)
    binding = artifact.get("source_binding")
    if binding is None:
        binding = {}
        artifact["source_binding"] = binding
    if not isinstance(binding, dict):
        raise ValueError("harmonic_analysis.source_binding must be an object")
    expected = {
        "model_sha256": hashes["model_sha256"],
        "request_sha256": hashes["request_sha256"],
        "run_sha256": run_sha256,
        "recorded_waveform_artifact": "waveforms.csv",
    }
    for name, value in expected.items():
        recorded = binding.get(name)
        if recorded is not None and recorded != value:
            raise ValueError(f"harmonic_analysis.source_binding.{name} disagrees with this run")
        binding[name] = value
    return artifact


def _require_harmonic_request_match(artifact: dict[str, Any], model: Model) -> None:
    settings = model.harmonics
    if settings is None:  # Kept local so this helper remains fail-closed if called directly.
        raise ValueError("Cannot bind harmonic_analysis without an analysis.harmonics request")

    window = _require_object(artifact.get("window"), "harmonic_analysis.window")
    _require_exact_float(
        window,
        "start_s",
        settings.start,
        "harmonic_analysis.window.start_s",
    )
    _require_exact_float(
        window,
        "stop_s",
        settings.stop,
        "harmonic_analysis.window.stop_s",
    )
    _require_exact_float(
        window,
        "duration_s",
        settings.stop - settings.start,
        "harmonic_analysis.window.duration_s",
    )
    _require_exact_int(
        window,
        "fundamental_period_count",
        settings.period_count,
        "harmonic_analysis.window.fundamental_period_count",
    )

    frequency = _require_object(
        artifact.get("frequency"),
        "harmonic_analysis.frequency",
    )
    _require_exact_float(
        frequency,
        "fundamental_hz",
        settings.fundamental_frequency,
        "harmonic_analysis.frequency.fundamental_hz",
    )
    _require_exact_int(
        frequency,
        "max_harmonic",
        settings.max_harmonic,
        "harmonic_analysis.frequency.max_harmonic",
    )
    _require_exact_float(
        frequency,
        "bandwidth_hz",
        settings.fundamental_frequency * settings.max_harmonic,
        "harmonic_analysis.frequency.bandwidth_hz",
    )

    signals = artifact.get("signals")
    if not isinstance(signals, list):
        raise ValueError("harmonic_analysis.signals must be an array")
    if len(signals) != len(settings.signals):
        raise ValueError(
            "harmonic_analysis.signals must exactly match the ordered harmonic request"
        )
    for index, (record, requested) in enumerate(zip(signals, settings.signals, strict=True)):
        prefix = f"harmonic_analysis.signals[{index}]"
        signal_record = _require_object(record, prefix)
        expected_unit = signal_unit(requested.signal)
        expected_identity = {
            "signal": requested.signal,
            "semantic": requested.semantic,
            "unit": expected_unit,
        }
        for name, expected in expected_identity.items():
            if signal_record.get(name) != expected:
                raise ValueError(f"{prefix}.{name} disagrees with the ordered harmonic request")

        coefficients = signal_record.get("coefficients")
        expected_count = settings.max_harmonic + 1
        if not isinstance(coefficients, list) or len(coefficients) != expected_count:
            raise ValueError(
                f"{prefix}.coefficients must contain exactly harmonic orders "
                f"0..{settings.max_harmonic}"
            )
        for harmonic, raw_row in enumerate(coefficients):
            row_prefix = f"{prefix}.coefficients[{harmonic}]"
            row = _require_object(raw_row, row_prefix)
            _require_exact_int(row, "harmonic", harmonic, f"{row_prefix}.harmonic")
            _require_exact_float(
                row,
                "frequency_hz",
                harmonic * settings.fundamental_frequency,
                f"{row_prefix}.frequency_hz",
            )


def _require_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_exact_float(
    container: dict[str, Any],
    field: str,
    expected: float,
    name: str,
) -> None:
    value = container.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) != expected
    ):
        raise ValueError(f"{name} disagrees with the harmonic request")


def _require_exact_int(
    container: dict[str, Any],
    field: str,
    expected: int,
    name: str,
) -> None:
    value = container.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{name} disagrees with the harmonic request")


def write_result(
    result: SimulationResult,
    output_dir: str | Path,
    *,
    topology_review: dict[str, Any] | None = None,
) -> Path:
    destination = Path(output_dir).resolve()
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Run output is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError(
                f"Run output directory is not empty: {destination}. "
                "Choose a new directory so an existing run is never overwritten."
            )
    else:
        destination.mkdir(parents=True)
    hashes = model_hashes(result.model)
    run_sha256 = sha256_json(
        {
            "model_sha256": hashes["model_sha256"],
            "request_sha256": hashes["request_sha256"],
            "backend": "reference-python",
            "pear_sim_version": __version__,
        }
    )
    metrics = result.metrics()
    analytic_checks = run_analytic_checks(
        result.model, metrics, time=result.time, waveforms=result.signals
    )
    mode_coverage = build_mode_coverage(result)
    is_three_phase_modulation = has_three_phase_spwm(result.model)
    is_npc_modulation = has_three_phase_npc_spwm(result.model)
    gate_drive = None
    modulation_cycles: list[dict[str, Any]] | None = None
    three_phase_modulation = None
    npc_modulation = None
    if is_three_phase_modulation:
        modulation_cycles = normalized_modulation_cycles(result)
        three_phase_modulation = build_three_phase_modulation_observation(
            result,
            normalized_cycles=modulation_cycles,
        )
        three_phase_modulation["source_binding"]["run_sha256"] = run_sha256
    elif is_npc_modulation:
        modulation_cycles = normalized_npc_modulation_cycles(result)
        npc_modulation = build_npc_modulation_observation(
            result,
            normalized_cycles=modulation_cycles,
        )
        npc_modulation["source_binding"]["run_sha256"] = run_sha256
    else:
        gate_drive = build_gate_drive_observation(result)
        gate_drive["source_binding"]["run_sha256"] = run_sha256
    losses_thermal = None
    if getattr(result.model, "loss_analysis", None) is not None:
        losses_thermal = build_losses_thermal_observation(result)
    resonance = build_resonance_observation(result)
    harmonic_analysis = _bound_harmonic_analysis(
        getattr(result, "harmonic_analysis", None),
        model=result.model,
        hashes=hashes,
        run_sha256=run_sha256,
    )
    mode_coverage_summary = {
        "artifact": "mode-coverage.json",
        **{
            key: mode_coverage.get(key)
            for key in (
                "status",
                "recognition",
                "window",
                "summary",
                "diode_conduction",
                "inductor_current",
            )
        },
    }
    evidence_mode = getattr(result, "evidence_mode", "full")
    warnings = [
        {
            "code": "MATURITY_EXPERIMENTAL",
            "severity": "warning",
            "scope": "run",
            "message": (
                "PEARSim is an early alpha. This result is exploratory and must not be used "
                "for hardware sign-off."
            ),
        },
        {
            "code": "IDEAL_DEVICE_MODEL",
            "severity": "info",
            "scope": "model",
            "message": (
                "Ideal switches/diodes: the electrical solution has no switching loss, "
                "temperature or EMI. Device-card losses, when declared, are estimated "
                "afterwards from this ideal run (losses-thermal.json)."
            ),
        },
    ]
    audit_full = (result.physics_audit or {}).get("full") or {}
    if audit_full.get("balance_status") == "not_assessed":
        warnings.append(
            {
                "code": "ENERGY_AUDIT_NOT_ASSESSED",
                "severity": "info",
                "scope": "run",
                "message": (
                    "The per-run energy balance was not assessed: "
                    + str(audit_full.get("balance_not_assessed_reason", "unsupported model"))
                    + ". Simulated waveforms are unaffected; only this cross-check is missing."
                ),
            }
        )
    if audit_full.get("balance_status") == "outside_numerical_tolerance":
        warnings.append(
            {
                "code": "ENERGY_BALANCE_OUTSIDE_TOLERANCE",
                "severity": "warning",
                "scope": "run",
                "message": (
                    "The run's energy balance misses its numerical budget (relative residual "
                    f"{audit_full.get('balance_residual_relative', float('nan')):.3g} vs "
                    f"{audit_full.get('balance_tolerance_relative', float('nan')):.3g}): the "
                    "integration error is larger than declared. This is usually a stiff "
                    "circuit (time constants far shorter than the run) on an explicit solver: "
                    "use solver: radau or bdf, or tighten rtol/atol, and re-run."
                ),
            }
        )
    lossy_devices = sorted(
        component.id
        for component in result.model.components
        if component.type in {"ideal_switch", "diode"} and not component.is_ideal_conductor
    )
    if lossy_devices:
        warnings.append(
            {
                "code": "PIECEWISE_LINEAR_CONDUCTION",
                "severity": "info",
                "scope": "model",
                "message": (
                    "Conduction is modelled as v = V_f + R_on * i for "
                    + ", ".join(lossy_devices)
                    + "; the drop is constant (no temperature or current dependence "
                    "beyond R_on) and switching transitions, reverse recovery and EMI "
                    "are still not modelled."
                ),
            }
        )
    if evidence_mode == "lean":
        warnings.append(
            {
                "code": "EVIDENCE_MODE_LEAN",
                "severity": "warning",
                "scope": "run",
                "message": (
                    "This run was produced in the lean evidence mode: it is not an "
                    "evidence bundle. Every value it does report was "
                    "computed by the identical code path as a full run and is bit-for-bit "
                    "identical to one. What is missing is missing entirely, never "
                    "approximated: the per-component energy balance (physics-audit.json "
                    "carries status=not_computed) and the dense state extrema of intervals "
                    "outside the reported last-half window. Re-run without --lean before "
                    "using this result as evidence."
                ),
            }
        )
    declared_meta_ids = sorted(
        component.id
        for component in result.model.components
        if component.model_meta is not None
    )
    if declared_meta_ids:
        warnings.append(
            {
                "code": "MODEL_META_DECLARED_NOT_EVALUATED",
                "severity": "info",
                "scope": "component-model-metadata",
                "message": (
                    "Component model metadata (version/validity envelope) is declared for "
                    + ", ".join(declared_meta_ids)
                    + " and recorded in model.resolved.json, but this release does not "
                    "evaluate operating conditions against the declared envelope."
                ),
            }
        )
    if any(
        component.type == "diode" for component in result.model.components
    ) and not result.stats.get("natural_events", 0):
        warnings.append(
            {
                "code": "NATURAL_COMMUTATION_NOT_EXERCISED",
                "severity": "info",
                "scope": "verification-coverage",
                "message": (
                    "This run recorded no natural diode commutation event. Its result does not "
                    "exercise or verify zero-crossing event detection; scheduled gate transitions "
                    "and switch-manager stabilization are separate mechanisms."
                ),
            }
        )
    flagged_resonances = [
        entry for entry in resonance.get("resonances", ()) if entry.get("flagged")
    ]
    if flagged_resonances:
        nearest = ", ".join(
            f"{entry['frequency_hz']:.4g} Hz (zeta={entry['damping_ratio']:.3g}, "
            f"nearest line {entry['nearest_excitation_hz']:.4g} Hz)"
            for entry in flagged_resonances[:4]
        )
        warnings.append(
            {
                "code": "RESONANCE_NEAR_EXCITATION",
                "severity": "warning",
                "scope": "network-modal-audit",
                "message": (
                    "A structural, lightly damped resonance sits close to a "
                    f"modulation excitation line: {nearest}. Damping here comes "
                    "only from modelled resistances and is therefore a lower "
                    "bound; see resonance.json for the thresholds and the "
                    "declared limits of the excitation list."
                ),
            }
        )
    if analytic_checks.get("overall_status") == "fail":
        warnings.append(
            {
                "code": "ANALYTIC_SELF_CHECK_FAILED",
                "severity": "warning",
                "scope": "verification-self-check",
                "message": (
                    "At least one applicable analytic self-check failed. Inspect "
                    "analytic-checks.json before using this run as evidence."
                ),
            }
        )
    if any(control.kind == "complementary_pwm" for control in result.model.controls.values()):
        warnings.append(
            {
                "code": "COMPLEMENTARY_PWM_EXPERIMENTAL",
                "severity": "info",
                "scope": "control",
                "message": (
                    "Complementary PWM scheduling and intrinsic interlock are experimental; "
                    "this run is not evidence of physical gate-driver timing or hardware safety."
                ),
            }
        )
    if is_three_phase_modulation:
        warnings.append(
            {
                "code": "THREE_PHASE_SPWM_EXPERIMENTAL",
                "severity": "info",
                "scope": "control",
                "message": (
                    "Three-phase offline midpoint-sampled SPWM scheduling and its "
                    "human-facing command projection are experimental; this run is not "
                    "inverter validation or evidence of causal digital-controller or "
                    "physical gate-driver timing."
                ),
            }
        )
    if npc_modulation is not None:
        warnings.append(
            {
                "code": "NPC_MODULATOR_EXPERIMENTAL",
                "severity": "info",
                "scope": "control",
                "message": (
                    "The level-shifted NPC modulator is experimental; "
                    "this run is not inverter validation or evidence of physical "
                    "gate-driver timing."
                ),
            }
        )
        suppressed_cells = sum(
            count
            for by_pair in npc_modulation["minimum_pulse_suppressions"].values()
            for count in by_pair.values()
        )
        if suppressed_cells:
            warnings.append(
                {
                    "code": "NPC_MINIMUM_PULSE_SUPPRESSED",
                    "severity": "info",
                    "scope": "control",
                    "message": (
                        f"{suppressed_cells} commanded band-edge pulse(s) were "
                        "narrower than the dead time and were suppressed, the "
                        "same behavior as a hardware gate driver's "
                        "minimum-pulse interlock. The suppressions are recorded "
                        "per cell in modulation-cycles.jsonl."
                    ),
                }
            )

    with (destination / "waveforms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        names = list(result.signals)
        writer.writerow(["time", *names])
        for index, time in enumerate(result.time):
            writer.writerow(
                [
                    f"{time:.17g}",
                    *(f"{result.signals[name][index]:.17g}" for name in names),
                ]
            )
    with (destination / "events.jsonl").open("w", encoding="utf-8") as handle:
        for event in result.events:
            handle.write(json.dumps(event.as_dict(), sort_keys=True) + "\n")
    if modulation_cycles is not None:
        with (destination / "modulation-cycles.jsonl").open("w", encoding="utf-8") as handle:
            for cycle in modulation_cycles:
                handle.write(
                    json.dumps(cycle, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n"
                )
    if getattr(result, "signal_trace", None):
        with (destination / "signal-trace.jsonl").open("w", encoding="utf-8") as handle:
            for entry in result.signal_trace:
                handle.write(
                    json.dumps(entry, allow_nan=False, ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
    if getattr(result, "machine_trace", None):
        with (destination / "machine-trace.jsonl").open("w", encoding="utf-8") as handle:
            for entry in result.machine_trace:
                handle.write(
                    json.dumps(entry, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n"
                )
    if result.control_trace:
        with (destination / "control-trace.jsonl").open("w", encoding="utf-8") as handle:
            for entry in result.control_trace:
                handle.write(
                    json.dumps(entry, allow_nan=False, ensure_ascii=False, sort_keys=True) + "\n"
                )
    _write_json(destination / "metrics.json", metrics)
    _write_json(destination / "analytic-checks.json", analytic_checks)
    _write_json(destination / "physics-audit.json", result.physics_audit)
    _write_json(destination / "mode-coverage.json", mode_coverage)
    if gate_drive is not None:
        _write_json(destination / "gate-drive.json", gate_drive)
    if three_phase_modulation is not None:
        _write_json(
            destination / "three-phase-modulation.json",
            three_phase_modulation,
        )
    if npc_modulation is not None:
        _write_json(destination / "npc-modulation.json", npc_modulation)
    _write_json(destination / "resonance.json", resonance)
    if harmonic_analysis is not None:
        _write_json(destination / "harmonics.json", harmonic_analysis)
    if losses_thermal is not None:
        _write_json(destination / "losses-thermal.json", losses_thermal)
    confidence_card = build_confidence_card(result.model)
    _write_json(destination / "confidence-card.json", confidence_card)
    _write_json(destination / "model.resolved.json", resolved_model_dict(result.model))
    _write_json(destination / "warnings.json", warnings)
    summary_payload: dict[str, Any] = {
        "schema": "pear-sim/run-summary/v0.1",
        "ok": True,
        "evidence_mode": evidence_mode,
        "model": result.model.name,
        "run_sha256": run_sha256,
        "stats": result.stats,
        "metrics": metrics,
        "analytic_self_checks": analytic_checks,
        "physics_audit": result.physics_audit,
        "mode_coverage": mode_coverage_summary,
        "warnings": warnings,
        "topology_review": topology_review,
    }
    if gate_drive is not None:
        summary_payload["gate_drive"] = {
            "artifact": "gate-drive.json",
            "schema": gate_drive.get("schema"),
            "summary": gate_drive.get("summary"),
        }
    if three_phase_modulation is not None:
        summary_payload["three_phase_modulation"] = {
            "artifact": "three-phase-modulation.json",
            "command_artifact": "modulation-cycles.jsonl",
            "schema": three_phase_modulation.get("schema"),
            "summary": three_phase_modulation.get("summary"),
        }
    if npc_modulation is not None:
        summary_payload["npc_modulation"] = {
            "artifact": "npc-modulation.json",
            "command_artifact": "modulation-cycles.jsonl",
            "schema": npc_modulation.get("schema"),
            "cycles_recorded": npc_modulation.get("cycles_recorded"),
            "band_transition_handovers": npc_modulation.get(
                "band_transition_handovers"
            ),
        }
    if harmonic_analysis is not None:
        summary_payload["harmonics"] = {
            "artifact": "harmonics.json",
            "schema": harmonic_analysis.get("schema"),
            "summary": harmonic_analysis.get("summary"),
        }
    summary_payload["resonance"] = {
        "artifact": "resonance.json",
        "schema": resonance.get("schema"),
        "status": resonance.get("status"),
        "resonances_reported": len(resonance.get("resonances", [])),
    }
    summary_payload["confidence_card"] = {
        "artifact": "confidence-card.json",
        "overall": confidence_card["overall"],
    }
    if losses_thermal is not None:
        summary_payload["losses_thermal"] = {
            "artifact": "losses-thermal.json",
            "schema": losses_thermal.get("schema"),
            "claim": losses_thermal.get("claim"),
            "total_mean_power_w": losses_thermal.get("total_mean_power_w"),
            "warnings": losses_thermal.get("warnings"),
        }
    _write_json(destination / "summary.json", summary_payload)

    run_files: dict[str, str] = {
        "resolved_model": "model.resolved.json",
        "waveforms": "waveforms.csv",
        "events": "events.jsonl",
        "metrics": "metrics.json",
        "analytic_checks": "analytic-checks.json",
        "physics_audit": "physics-audit.json",
        "mode_coverage": "mode-coverage.json",
        "warnings": "warnings.json",
        "summary": "summary.json",
    }
    if gate_drive is not None:
        run_files["gate_drive"] = "gate-drive.json"
    if getattr(result, "machine_trace", None):
        run_files["machine_trace"] = "machine-trace.jsonl"
    if getattr(result, "signal_trace", None):
        run_files["signal_trace"] = "signal-trace.jsonl"
    if result.control_trace:
        run_files["control_trace"] = "control-trace.jsonl"
    if three_phase_modulation is not None:
        run_files["modulation_cycles"] = "modulation-cycles.jsonl"
        run_files["three_phase_modulation"] = "three-phase-modulation.json"
    if npc_modulation is not None:
        run_files["modulation_cycles"] = "modulation-cycles.jsonl"
        run_files["npc_modulation"] = "npc-modulation.json"
    if harmonic_analysis is not None:
        run_files["harmonics"] = "harmonics.json"
    if losses_thermal is not None:
        run_files["losses_thermal"] = "losses-thermal.json"
    run_files["resonance"] = "resonance.json"
    run_files["confidence_card"] = "confidence-card.json"

    manifest: dict[str, Any] = {
        "schema": "pear-sim/run/v0.1",
        "pear_sim_version": __version__,
        "maturity": "experimental",
        # Always present, in both modes: a reader must never have to infer the
        # evidence scope from a missing field.
        "evidence_mode": evidence_mode,
        "backend": "reference-python",
        "model_schema": result.model.schema,
        "model": result.model.name,
        "model_file": result.model.source_path.name,
        **hashes,
        "run_sha256": run_sha256,
        "solver": result.model.transient.solver,
        "rtol": result.model.transient.rtol,
        "atol": result.model.transient.atol,
        "parameters": {
            name: {"value_si": quantity.value, "dimension": quantity.dimension}
            for name, quantity in sorted(result.model.parameters.items())
        },
        "outputs": list(result.model.outputs),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "stats": result.stats,
        "warning_count": len(warnings),
        "topology_review": topology_review,
        "files": run_files,
    }
    _write_json(destination / "run.json", manifest)
    artifact_names = ["run.json", *manifest["files"].values()]
    _write_json(
        destination / "artifact-manifest.json",
        {
            "schema": "pear-sim/artifact-manifest/v0.1",
            "subject": {"kind": "run", "sha256": run_sha256},
            "artifacts": [
                {
                    "path": name,
                    "sha256": sha256_file(destination / name),
                    "bytes": (destination / name).stat().st_size,
                }
                for name in sorted(artifact_names)
            ],
        },
    )
    return destination
