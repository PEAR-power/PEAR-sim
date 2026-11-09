# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

import pear_sim.report as report_module
from pear_sim.io import write_result
from pear_sim.model import load_model
from pear_sim.report import MAX_EVENT_LINE_BYTES, REPORT_SCHEMA, ReportError, generate_report
from pear_sim.render import render_schematic
from pear_sim.report_svg import waveform_svg
from pear_sim.solver import HybridSimulator


EXAMPLE = Path(__file__).parents[1] / "examples" / "buck.yaml"


def _write_source_artifact_manifest(run_dir: Path, manifest: dict[str, object]) -> None:
    artifact_names = ["run.json", *manifest["files"].values()]
    artifact_records = []
    for name in sorted(artifact_names):
        data = (run_dir / name).read_bytes()
        artifact_records.append(
            {"path": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        )
    (run_dir / "artifact-manifest.json").write_text(
        json.dumps(
            {
                "schema": "pear-sim/artifact-manifest/v0.1",
                "subject": {"kind": "run", "sha256": manifest["run_sha256"]},
                "artifacts": artifact_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_run(run_dir: Path) -> dict[str, object]:
    run_dir.mkdir()
    manifest: dict[str, object] = {
        "schema": "pear-sim/run/v0.1",
        "pear_sim_version": "0.1.test",
        "model_schema": "pear-sim/v0.1",
        "model": "buck <script>alert('model')</script> & fixture",
        "model_path": "/example/<unsafe>/buck.yaml",
        "model_sha256": "a" * 64,
        "run_sha256": "b" * 64,
        "solver": "dopri5",
        "rtol": 1e-8,
        "atol": 1e-10,
        "parameters": {"vin": {"value_si": 12.0, "dimension": "voltage"}},
        "outputs": ["v(out)<unsafe>"],
        "platform": {"system": "TestOS", "machine": "test64", "python": "3.11"},
        "stats": {"samples": 4, "scheduled_events": 1},
        "files": {
            "resolved_model": "model.resolved.json",
            "waveforms": "waveforms.csv",
            "events": "events.jsonl",
            "metrics": "metrics.json",
            "warnings": "warnings.json",
            "summary": "summary.json",
        },
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "v(out)<unsafe>": {
            "min": 0.0,
            "max": 3.0,
            "mean": 1.5,
            "note": "<b>recorded, not markup</b>",
        }
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "waveforms.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time", "v(out)<unsafe>"])
        writer.writerows([(0.0, 0.0), (1e-6, 1.0), (2e-6, 2.0), (3e-6, 3.0)])
    (run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "time": 2e-6,
                "kind": "scheduled_gate",
                "target": "q1<script>unsafe</script>",
                "previous": False,
                "current": True,
                "note": "<img src=x onerror=alert(1)>",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    warnings = [
        {
            "code": "FIXTURE_WARNING",
            "severity": "warning",
            "message": "Voltage < expected & must be reviewed",
        }
    ]
    (run_dir / "warnings.json").write_text(json.dumps(warnings, indent=2) + "\n", encoding="utf-8")
    (run_dir / "model.resolved.json").write_text(
        json.dumps(
            {
                "schema": "pear-sim/resolved-model/v0.1",
                "name": "buck <resolved>",
                "hashes": {"model_sha256": "a" * 64},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "schema": "pear-sim/run-summary/v0.1",
                "ok": True,
                "model": manifest["model"],
                "stats": manifest["stats"],
                "metrics": metrics,
                "warnings": warnings,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_source_artifact_manifest(run_dir, manifest)
    return manifest


def _attach_analysis_artifacts(
    run_dir: Path, manifest: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    analytic_checks: dict[str, object] = {
        "schema": "pear-sim/analytic-self-checks/v0.1",
        "kind": "verification-self-check",
        "claim": {
            "classification": "verification",
            "validation_claim": False,
            "statement": (
                "Idealized self-check only; <strong>hardware validation is not claimed</strong>."
            ),
        },
        "overall_status": "pass",
        "recognition": {
            "topology": "ideal_buck_ccm",
            "status": "matched",
            "reason": "Exact fixture topology <matched>.",
        },
        "preconditions": [
            {
                "id": "buck.measurement_window",
                "description": "Last-half window spans enough cycles <unsafe>.",
                "measured": {"value": 10.0, "unit": "cycle"},
                "tolerance": {"minimum_inclusive": 5.0, "unit": "cycle"},
                "status": "pass",
                "reason": None,
            },
            {
                "id": "buck.ccm_regime",
                "description": "Minimum current remains above the CCM floor.",
                "measured": {
                    "value": 1.2,
                    "unit": "A",
                    "signal": "i(l1)<unsafe>",
                    "metric": "min_last_half",
                },
                "tolerance": {"minimum_exclusive": 1e-6, "unit": "A"},
                "status": "pass",
                "reason": None,
            },
        ],
        "checks": [
            {
                "id": "buck.output_voltage",
                "relation": "Vout = D * Vin <script>alert('relation')</script>",
                "expected": {"value": 6.0, "unit": "V"},
                "measured": {
                    "value": 6.03,
                    "unit": "V",
                    "signal": "v(out)<unsafe>",
                    "metric": "mean_last_half",
                },
                "delta": {
                    "signed": 0.03,
                    "absolute": 0.03,
                    "relative": 0.005,
                    "unit": "V",
                },
                "tolerance": {
                    "absolute": 0.05,
                    "relative": 0.01,
                    "effective_absolute": 0.06,
                    "unit": "V",
                    "comparison": "abs(measured - expected) <= effective_absolute",
                },
                "status": "pass",
                "reason": "<img src=x onerror=alert('reason')>",
            }
        ],
    }
    physics_audit: dict[str, object] = {
        "schema": "pear-sim/physics-audit/v0.1",
        "method": "trapezoidal_fixture",
        "power_sign_convention": "positive_absorbed_from_p_to_n",
        "full": {
            "window_start_s": 0.0,
            "window_end_s": 0.001,
            "duration_s": 0.001,
            "quadrature_intervals": 17,
            "component_absorbed_energy_j": {"vin": -0.012, "load<script>": 0.0105},
            "storage_energy_start_j": 0.0,
            "storage_energy_end_j": 0.001,
            "storage_energy_change_j": {"l1": 0.001},
            "delta_storage_energy_j": 0.001,
            "source_delivered_energy_j": 123.456,
            "source_delivered_mean_power_w": 123456.0,
            "resistor_absorbed_energy_j": 7.89,
            "resistor_absorbed_mean_power_w": 7890.0,
            "balance_residual_j": -1.2e-6,
            "balance_tolerance_j": 2e-6,
            "balance_within_tolerance": True,
            "balance_status": "within_numerical_tolerance",
        },
        "last_half": {
            "window_start_s": 0.0005,
            "window_end_s": 0.001,
            "duration_s": 0.0005,
            "quadrature_intervals": 9,
            "component_absorbed_energy_j": {"vin": -0.006, "load": 0.0055},
            "source_delivered_energy_j": 54.321,
            "load_absorbed_energy_j": 4.56,
            "balance_residual_j": 3.4e-8,
            "balance_within_tolerance": False,
            "balance_status": "outside_numerical_tolerance",
        },
    }
    files = dict(manifest["files"])
    files.update(
        {
            "analytic_checks": "analytic-checks.json",
            "physics_audit": "physics-audit.json",
        }
    )
    manifest["files"] = files
    manifest["stats"] = {
        "samples": 4,
        "scheduled_events": 1,
        "wall_time_s": 0.42,
        "integration_steps": 99,
        "natural_events": 0,
    }
    (run_dir / "analytic-checks.json").write_text(
        json.dumps(analytic_checks, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "physics-audit.json").write_text(
        json.dumps(physics_audit, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "model.resolved.json").write_text(
        json.dumps(
            {
                "schema": "pear-sim/resolved-model/v0.1",
                "name": "buck <resolved>",
                "hashes": {"model_sha256": "a" * 64},
                "conventions": {
                    "component_voltage": {
                        "definition": "v(component)=V(p)-V(n)",
                        "positive_terminal": "p",
                        "negative_terminal": "n",
                    },
                    "component_current": {
                        "definition": "i(component)>0 means current flows p->n",
                        "positive_direction": "p->n",
                    },
                },
                "initial_conditions": {
                    "strategy": "explicit_seeded_values",
                    "origin_policy": "user declared or default zero <policy>",
                    "analytic_initialization_claimed": False,
                    "states": [
                        {
                            "component_id": "l1<script>",
                            "component_type": "inductor",
                            "state_expression": "i(l1)",
                            "source_field": "initial_current",
                            "value_si": 1.25,
                            "unit": "A",
                            "value_class": "nonzero",
                            "origin": "user_declared",
                            "analytic_derivation": "not_claimed",
                        },
                        {
                            "component_id": "c1",
                            "component_type": "capacitor",
                            "state_expression": "v(c1)",
                            "source_field": "initial_voltage",
                            "value_si": 0.0,
                            "unit": "V",
                            "value_class": "zero",
                            "origin": "default_zero",
                            "analytic_derivation": "not_claimed",
                        },
                    ],
                    "summary": {
                        "state_count": 2,
                        "zero_count": 1,
                        "nonzero_count": 1,
                        "user_declared_count": 1,
                        "default_zero_count": 1,
                        "unknown_origin_count": 0,
                        "warm_start": True,
                        "warm_start_components": ["l1<script>"],
                    },
                    "warm_start": {
                        "enabled": True,
                        "definition": "one or more L/C seeds are nonzero",
                        "provenance_claim": "seed values only; no analytic origin inferred",
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_source_artifact_manifest(run_dir, manifest)
    return analytic_checks, physics_audit


def _mode_coverage_fixture() -> dict[str, object]:
    return {
        "schema": "pear-sim/mode-coverage/v0.1",
        "kind": "hybrid-mode-coverage-observation",
        "claim": {
            "classification": "verification-coverage-observation",
            "validation_claim": False,
            "statement": (
                "Observed hybrid modes only; <strong>DCM validation is not claimed</strong>."
            ),
        },
        "status": "observed",
        "recognition": {
            "topology": "canonical_single_switch_buck",
            "status": "matched",
            "reason": "Exact canonical topology <matched>.",
            "subjects": {
                "control_id": "pwm",
                "switch_id": "q1",
                "freewheel_diode_id": "d1<script>",
                "inductor_id": "l1",
            },
        },
        "window": {
            "start_s": 0.0,
            "end_s": 0.001,
            "duration_s": 0.001,
            "switching_period_s": 0.00005,
            "cycle_origin_s": 0.0,
        },
        "classification": {
            "method": "accepted_hybrid_segments_adjacent_equal_modes_merged",
            "boundary_convention": (
                "[start_s,end_s), with the simulation endpoint owned by the final interval"
            ),
            "rules": {
                "on": "gate_on=true and diode_conducting=false",
                "freewheel": "gate_on=false and diode_conducting=true",
                "idle": "gate_on=false and diode_conducting=false",
                "unclassified": "all other combinations <unsafe>",
            },
        },
        "summary": {
            "interval_count": 3,
            "covered_duration_s": 0.001,
            # Deliberately not derivable from the fixture intervals: the renderer must copy.
            "duration_by_mode_s": {
                "on": 123.456,
                "freewheel": 0.0003,
                "idle": 0.0005,
                "unclassified": 0.0,
            },
            "fraction_by_mode": {
                "on": 0.111,
                "freewheel": 0.222,
                "idle": 0.667,
                "unclassified": 0.0,
            },
            "natural_commutation": {
                "exercised": True,
                "event_count": 7,
                "diode_turn_off_event_count": 6,
            },
        },
        "diode_conduction": {
            "component_id": "d1",
            "conducting_duration_s": 0.0003,
            "observed_duration_s": 0.001,
            "fraction": 0.314159,
            "method": "sum accepted mode intervals with diode_conducting=true",
        },
        "inductor_current": {
            "component_id": "l1",
            "sign_convention": "positive_p_to_n",
            "minimum_a": -2e-12,
            "minimum_time_s": 0.000037,
            "maximum_forbidden_reverse_current_a": 9.876,
            "definition": "max(0, -minimum_a)",
            "state_constraint_tolerance_a": 1e-9,
            "method": "event_aligned_dense_trajectory_extrema",
        },
        "event_timing": {
            "time_reference": "absolute_simulation_time_s",
            "cycle_reference": "recognized PWM origin and period",
            "event_merge_tolerance_s": 1e-15,
            "events": [
                {
                    "index": 0,
                    "time_s": 0.000037,
                    "kind": "natural_commutation",
                    "target": "d1<img src=x onerror=alert(1)>",
                    "previous": True,
                    "current": False,
                    "cycle_index": 0,
                    "offset_from_cycle_start_s": 0.000037,
                    "phase_fraction": 0.74,
                }
            ],
        },
        "intervals": [
            {
                "index": 0,
                "start_s": 0.0,
                "end_s": 0.00002,
                "duration_s": 0.00002,
                "mode": "on",
                "gate_on": True,
                "diode_conducting": False,
                "cycle_index": 0,
                "start_event_indices": [],
                "end_event_indices": [0],
            },
            {
                "index": 1,
                "start_s": 0.00002,
                "end_s": 0.000037,
                "duration_s": 0.000017,
                "mode": "freewheel",
                "gate_on": False,
                "diode_conducting": True,
                "cycle_index": 0,
                "start_event_indices": [0],
                "end_event_indices": [1],
            },
            {
                "index": 2,
                "start_s": 0.000037,
                "end_s": 0.00005,
                "duration_s": 0.000013,
                "mode": "idle<script>",
                "gate_on": False,
                "diode_conducting": False,
                "cycle_index": 0,
                "start_event_indices": [1],
                "end_event_indices": [2],
            },
        ],
    }


def _attach_mode_coverage(
    run_dir: Path,
    manifest: dict[str, object],
    mode_coverage: dict[str, object] | None = None,
) -> dict[str, object]:
    payload = _mode_coverage_fixture() if mode_coverage is None else mode_coverage
    files = dict(manifest["files"])
    files["mode_coverage"] = "mode-coverage.json"
    manifest["files"] = files
    (run_dir / "mode-coverage.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_source_artifact_manifest(run_dir, manifest)
    return payload


def test_report_is_deterministic_escaped_and_offline(tmp_path: Path):
    run_dir = tmp_path / "run"
    manifest = _write_run(run_dir)
    schematic = """
<svg viewBox="0 0 120 40" onload="alert('root')">
  <rect x="2" y="2" width="30" height="20" fill="#fff" onclick="alert('rect')" />
  <script>alert('schematic')</script>
  <image href="https://example.invalid/tracker.svg" width="10" height="10" />
  <text x="40" y="18">safe &amp; sound</text>
</svg>
""".strip()

    report_dir = tmp_path / "rendered-report"
    artifacts = generate_report(run_dir, report_dir, schematic)
    first_json = artifacts.report_json.read_bytes()
    first_html = artifacts.report_html.read_bytes()
    second = generate_report(run_dir, report_dir, schematic)

    assert second.report_json.read_bytes() == first_json
    assert second.report_html.read_bytes() == first_html
    report = json.loads(first_json)
    assert report["schema"] == REPORT_SCHEMA
    assert report["run"] == manifest
    assert report["warnings"] == [
        {
            "code": "FIXTURE_WARNING",
            "severity": "warning",
            "message": "Voltage < expected & must be reviewed",
        }
    ]
    assert report["waveforms"]["sample_count"] == 4
    assert report["waveforms"]["signals"] == ["v(out)<unsafe>"]
    assert report["events"]["count"] == 1
    assert report["events"]["by_kind"] == {"scheduled_gate": 1}
    assert report["events"]["time"] == {
        "numeric_count": 1,
        "start_s": 2e-6,
        "stop_s": 2e-6,
        "monotonic": True,
    }
    assert report["events"]["items"][0]["target"] == "q1<script>unsafe</script>"
    assert set(report["provenance"]["input_artifacts"]) == {
        "events",
        "metrics",
        "resolved_model",
        "run",
        "summary",
        "waveforms",
        "warnings",
        "artifact_manifest",
    }
    assert report["provenance"]["source_artifact_manifest"]["verified"] is True
    assert report["source_summary"]["schema"] == "pear-sim/run-summary/v0.1"
    assert report["resolved_model"]["schema"] == "pear-sim/resolved-model/v0.1"
    assert report["analytic_checks"] is None
    assert report["physics_audit"] is None
    assert report["mode_coverage"] is None
    assert report["provenance"]["schematic"]["present"] is True
    assert artifacts.schematic_svg == report_dir / "schematic.svg"
    assert artifacts.artifact_manifest == report_dir / "artifact-manifest.json"
    assert artifacts.schematic_svg.is_file()

    rendered = first_html.decode("utf-8")
    lowered = rendered.lower()
    assert "<script" not in lowered
    assert "onload" not in lowered
    assert "onclick" not in lowered
    assert "<image" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "fetch(" not in lowered
    assert "<link" not in lowered
    # Only in-page anchors (the table of contents) may link; nothing leaves the page.
    import re as _re

    assert not _re.search(r' href="(?!#[a-z0-9-]+")', lowered)
    assert "example.invalid" not in lowered
    assert "cdn" not in lowered
    assert "buck &lt;script&gt;alert(&#x27;model&#x27;)&lt;/script&gt; &amp; fixture" in rendered
    assert "&lt;b&gt;recorded, not markup&lt;/b&gt;" in rendered
    assert "Voltage &lt; expected &amp; must be reviewed" in rendered
    assert "safe &amp; sound" in rendered
    assert "<h2>Events</h2>" in rendered
    assert "<h2>Measurement &amp; initialization</h2>" in rendered
    assert "Measurement conventions were not recorded." in rendered
    assert "<h2>Analytic verification self-check</h2>" not in rendered
    assert "<h2>Physics audit</h2>" not in rendered
    assert "<h2>Hybrid mode coverage</h2>" not in rendered
    assert "<small>Wall time</small><strong>—</strong>" in rendered
    assert "<small>Integration steps</small><strong>—</strong>" in rendered
    assert "<small>Natural commutations</small><strong>—</strong>" in rendered
    assert "not recorded" in rendered
    assert "Show recorded event table (1 row(s))" in rendered
    assert "last-half mean" in rendered
    assert "q1&lt;script&gt;unsafe&lt;/script&gt;" in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert rendered.count("<svg") == 2
    assert "<polyline" in rendered
    assert "time [µs]" in rendered


def test_report_embeds_and_renders_optional_verification_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    manifest = _write_run(run_dir)
    analytic_checks, physics_audit = _attach_analysis_artifacts(run_dir, manifest)
    report_dir = tmp_path / "report"

    artifacts = generate_report(run_dir, report_dir)
    first_json = artifacts.report_json.read_bytes()
    first_html = artifacts.report_html.read_bytes()
    second = generate_report(run_dir, report_dir)

    assert second.report_json.read_bytes() == first_json
    assert second.report_html.read_bytes() == first_html
    report = json.loads(first_json)
    assert report["analytic_checks"] == analytic_checks
    assert report["physics_audit"] == physics_audit
    input_artifacts = report["provenance"]["input_artifacts"]
    assert {"analytic_checks", "physics_audit"} <= set(input_artifacts)
    for role, filename in (
        ("analytic_checks", "analytic-checks.json"),
        ("physics_audit", "physics-audit.json"),
    ):
        source = (run_dir / filename).read_bytes()
        assert input_artifacts[role] == {
            "file": filename,
            "bytes": len(source),
            "sha256": hashlib.sha256(source).hexdigest(),
        }

    rendered = first_html.decode("utf-8")
    lowered = rendered.lower()
    assert "<h2>Measurement &amp; initialization</h2>" in rendered
    assert "v(component)=V(p)-V(n)" in rendered
    assert "i(component)&gt;0 means current flows p-&gt;n" in rendered
    assert "explicit_seeded_values" in rendered
    assert "l1&lt;script&gt;" in rendered
    assert "1.25 A" in rendered
    assert "user_declared" in rendered
    assert "not_claimed" in rendered
    assert "<h2>Analytic verification self-check</h2>" in rendered
    assert "VERIFICATION SELF-CHECK — NOT A VALIDATION CLAIM." in rendered
    assert "validation_claim=false" in rendered
    assert "<small>Overall status</small><strong>pass</strong>" in rendered
    assert "buck.measurement_window" in rendered
    assert "buck.ccm_regime" in rendered
    assert "<th>Expected</th><th>Simulated</th><th>Δ abs / Δ %</th>" in rendered
    assert "0.03 V" in rendered
    assert "0.5%" in rendered
    assert "relative=1%" in rendered
    assert "Vout = D * Vin &lt;script&gt;alert(&#x27;relation&#x27;)&lt;/script&gt;" in rendered
    assert "&lt;img src=x onerror=alert(&#x27;reason&#x27;)&gt;" in rendered
    assert "<h2>Physics audit</h2>" in rendered
    assert "positive_absorbed_from_p_to_n" in rendered
    assert "Full interval (<code>full</code>)" in rendered
    assert "Last half (<code>last_half</code>)" in rendered
    assert "123.456" in rendered
    assert "artifact field source_delivered_energy_j" in rendered
    assert "7.89" in rendered
    assert "artifact field resistor_absorbed_energy_j" in rendered
    assert "54.321" in rendered
    assert "artifact field load_absorbed_energy_j" in rendered
    assert "load&lt;script&gt;" in rendered
    assert "the report does not recompute balances, negate source energy" in rendered
    assert "<small>Wall time</small><strong>0.42 s</strong>" in rendered
    assert "<small>Integration steps</small><strong>99</strong>" in rendered
    assert "<small>Natural commutations</small><strong>0</strong>" in rendered
    assert "coverage not exercised" in rendered
    assert "<script" not in lowered
    assert "<img" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered


def test_report_embeds_and_renders_mode_coverage_without_recomputing(tmp_path: Path):
    run_dir = tmp_path / "run"
    manifest = _write_run(run_dir)
    mode_coverage = _attach_mode_coverage(run_dir, manifest)
    report_dir = tmp_path / "report"

    artifacts = generate_report(run_dir, report_dir)
    first_json = artifacts.report_json.read_bytes()
    first_html = artifacts.report_html.read_bytes()
    second = generate_report(run_dir, report_dir)

    assert second.report_json.read_bytes() == first_json
    assert second.report_html.read_bytes() == first_html
    report = json.loads(first_json)
    assert report["mode_coverage"] == mode_coverage
    source = (run_dir / "mode-coverage.json").read_bytes()
    assert report["provenance"]["input_artifacts"]["mode_coverage"] == {
        "file": "mode-coverage.json",
        "bytes": len(source),
        "sha256": hashlib.sha256(source).hexdigest(),
    }

    rendered = first_html.decode("utf-8")
    lowered = rendered.lower()
    assert "<h2>Hybrid mode coverage</h2>" in rendered
    assert "COVERAGE OBSERVATION — NOT A VALIDATION CLAIM." in rendered
    assert "validation_claim=false" in rendered
    assert "canonical_single_switch_buck" in rendered
    assert "d1&lt;script&gt;" in rendered
    assert "accepted_hybrid_segments_adjacent_equal_modes_merged" in rendered
    assert "<h3>Mode fractions</h3>" in rendered
    assert "123.456" in rendered
    assert "0.111" in rendered
    assert "0.222" in rendered
    assert "0.667" in rendered
    assert "does not derive them from the interval list" in rendered
    assert "<small>Natural commutation exercised</small><strong>true</strong>" in rendered
    assert "<small>Natural event count</small><strong>7</strong>" in rendered
    assert "diode turn-off=6" in rendered
    assert "0.314159" in rendered
    assert "9.876" in rendered
    assert "state-constraint tolerance=1e-09 A" in rendered
    assert "does not calculate a reverse-current violation or a pass/fail gate" in rendered
    assert "<h3>Event timing</h3>" in rendered
    assert "<details><summary>Recorded event timing table (1 row)</summary>" in rendered
    assert 'class="cards coverage-cards"' in rendered
    assert 'class="card method-card"' in rendered
    assert "3.7e-05" in rendered
    assert "0.74" in rendered
    assert "d1&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert "<h3>Mode interval timeline</h3>" in rendered
    assert "Equal-width interval tiles are deliberately not to scale" in rendered
    assert 'data-mode="on"' in rendered
    assert 'data-mode="freewheel"' in rendered
    assert 'data-mode="unclassified"' in rendered
    assert "idle&lt;script&gt;" in rendered
    assert "Recorded interval table" in rendered
    assert "<script" not in lowered
    assert "<img" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered


def test_mode_coverage_false_means_coverage_not_exercised(tmp_path: Path):
    run_dir = tmp_path / "run"
    manifest = _write_run(run_dir)
    mode_coverage = _mode_coverage_fixture()
    summary = dict(mode_coverage["summary"])
    summary["natural_commutation"] = {
        "exercised": False,
        "event_count": 0,
        "diode_turn_off_event_count": 0,
    }
    mode_coverage["summary"] = summary
    _attach_mode_coverage(run_dir, manifest, mode_coverage)

    artifacts = generate_report(run_dir, tmp_path / "report")
    rendered = artifacts.report_html.read_text(encoding="utf-8")

    assert "<small>Natural commutation exercised</small><strong>false</strong>" in rendered
    assert "coverage not exercised; this is not an engine failure" in rendered
    assert "<small>Natural event count</small><strong>0</strong>" in rendered


def test_report_renders_not_applicable_mode_coverage_without_inference(tmp_path: Path):
    run_dir = tmp_path / "run"
    manifest = _write_run(run_dir)
    mode_coverage = {
        "schema": "pear-sim/mode-coverage/v0.1",
        "kind": "hybrid-mode-coverage-observation",
        "claim": {
            "classification": "verification-coverage-observation",
            "validation_claim": False,
            "statement": "No observation outside the canonical matcher.",
        },
        "status": "not_applicable",
        "recognition": {
            "topology": "canonical_single_switch_buck",
            "status": "not_applicable",
            "reason": "Topology does not match; roles are not guessed.",
            "subjects": None,
        },
        "window": None,
        "classification": None,
        "summary": None,
        "diode_conduction": None,
        "inductor_current": None,
        "event_timing": None,
        "intervals": None,
    }
    _attach_mode_coverage(run_dir, manifest, mode_coverage)

    artifacts = generate_report(run_dir, tmp_path / "report")
    report = json.loads(artifacts.report_json.read_text(encoding="utf-8"))
    rendered = artifacts.report_html.read_text(encoding="utf-8")

    assert report["mode_coverage"] == mode_coverage
    assert "<small>Artifact status</small><strong>not_applicable</strong>" in rendered
    assert "Topology does not match; roles are not guessed." in rendered
    assert "No mode coverage is inferred" in rendered
    assert "No recognized subject mapping was recorded." in rendered
    assert "No mode intervals were recorded." in rendered


@pytest.mark.parametrize(
    ("time", "expected_axis"),
    [
        ([0.0, 1e-6, 2e-6], "time [µs]"),
        ([0.0, 1e-3, 2e-3], "time [ms]"),
        ([0.0, 1.0, 2.0], "time [s]"),
    ],
)
def test_waveform_time_axis_uses_deterministic_si_units(
    time: list[float],
    expected_axis: str,
) -> None:
    values = [0.0, 1.0, 0.0]
    first = waveform_svg(time, values, "v(out)")
    second = waveform_svg(time, values, "v(out)")

    assert first == second
    assert expected_axis in first


@pytest.mark.parametrize(
    ("values", "minimum_relation", "maximum_relation"),
    [
        ([0.0, 1.0, 0.0], "zero", "positive"),
        ([0.001, 1.0, 0.5], "zero", "positive"),
        ([-1e-9, 1.0, 0.5], "negative", "positive"),
        ([-1.0, -0.001, -0.5], "negative", "zero"),
    ],
)
def test_waveform_padding_never_invents_or_hides_a_zero_crossing(
    values: list[float],
    minimum_relation: str,
    maximum_relation: str,
) -> None:
    rendered = waveform_svg([0.0, 1.0, 2.0], values, "i(l1)")
    root = ET.fromstring(rendered)
    y_ticks = [
        float(element.text)
        for element in root.findall("text")
        if element.attrib.get("x") == "65.000"
    ]

    assert len(y_ticks) == 5
    if minimum_relation == "zero":
        assert min(y_ticks) == 0.0
    else:
        assert min(y_ticks) < 0.0
    if maximum_relation == "zero":
        assert max(y_ticks) == 0.0
    else:
        assert max(y_ticks) > 0.0


def test_report_accepts_schematic_path_and_optional_absence(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    schematic_path = tmp_path / "schematic.svg"
    schematic_path.write_text(
        '<svg viewBox="0 0 10 10"><line x1="0" y1="0" x2="10" y2="10" /></svg>',
        encoding="utf-8",
    )

    with_schematic = generate_report(run_dir, tmp_path / "report-with", schematic_path)
    report = json.loads(with_schematic.report_json.read_text(encoding="utf-8"))
    assert report["provenance"]["schematic"]["source_kind"] == "file"
    assert "Circuit schematic" in with_schematic.report_html.read_text(encoding="utf-8")

    without_schematic = generate_report(run_dir)
    report = json.loads(without_schematic.report_json.read_text(encoding="utf-8"))
    assert report["provenance"]["schematic"] == {"present": False}
    assert "<h2>Schematic</h2>" not in without_schematic.report_html.read_text(encoding="utf-8")
    assert without_schematic.report_json.parent == tmp_path / "run-report"
    assert without_schematic.schematic_svg is None
    assert without_schematic.artifact_manifest.is_file()
    assert not (run_dir / "report.json").exists()
    assert not (run_dir / "report.html").exists()


def test_report_rejects_artifact_path_escape(tmp_path: Path):
    run_dir = tmp_path / "run"
    manifest = _write_run(run_dir)
    files = dict(manifest["files"])
    files["metrics"] = "../metrics.json"
    manifest["files"] = files
    (run_dir / "run.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    (tmp_path / "metrics.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ReportError, match="escapes the run directory"):
        generate_report(run_dir, tmp_path / "report")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema", "pear-sim/artifact-manifest/v999", "field 'schema'"),
        ("subject.kind", "report", "subject.kind must be 'run'"),
        ("subject.sha256", "0" * 64, "does not match run.json run_sha256"),
    ],
)
def test_report_rejects_unbound_source_artifact_manifest(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    manifest_path = run_dir / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if field == "schema":
        manifest["schema"] = value
    else:
        subject = dict(manifest["subject"])
        subject[field.removeprefix("subject.")] = value
        manifest["subject"] = subject
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ReportError, match=message):
        generate_report(run_dir, tmp_path / "report")


@pytest.mark.parametrize("coverage_change", ["missing", "unexpected"])
def test_report_requires_exact_source_artifact_manifest_coverage(
    tmp_path: Path,
    coverage_change: str,
):
    run_dir = tmp_path / "run"
    run_manifest = _write_run(run_dir)
    _attach_mode_coverage(run_dir, run_manifest)
    manifest_path = run_dir / "artifact-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if coverage_change == "missing":
        manifest["artifacts"] = [
            record
            for record in manifest["artifacts"]
            if record["path"] != "mode-coverage.json"
        ]
        expected_message = "missing 'mode-coverage.json'"
    else:
        extra_path = run_dir / "unbound.json"
        extra_path.write_text("{}\n", encoding="utf-8")
        source = extra_path.read_bytes()
        manifest["artifacts"].append(
            {
                "path": extra_path.name,
                "sha256": hashlib.sha256(source).hexdigest(),
                "bytes": len(source),
            }
        )
        expected_message = "unexpected 'unbound.json'"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ReportError, match=expected_message):
        generate_report(run_dir, tmp_path / "report")


def test_report_rejects_run_root_as_output(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)

    with pytest.raises(ReportError, match="outside the immutable run tree"):
        generate_report(run_dir, run_dir)
    with pytest.raises(ReportError, match="outside the immutable run tree"):
        generate_report(run_dir, run_dir / "nested-report")


def test_report_refuses_to_mix_with_unowned_output_files(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_run(run_dir)
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "notes.txt").write_text("belongs to the user\n", encoding="utf-8")

    with pytest.raises(ReportError, match="not owned by PEARSim"):
        generate_report(run_dir, report_dir)

    assert (report_dir / "notes.txt").read_text(encoding="utf-8") == "belongs to the user\n"


def test_report_integrates_with_current_run_bundle_without_mutating_it(tmp_path: Path):
    result = HybridSimulator(load_model(EXAMPLE)).simulate()
    run_dir = write_result(result, tmp_path / "buck-run")
    before = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}

    artifacts = generate_report(run_dir)

    after = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}
    assert after == before
    assert artifacts.report_json.parent == tmp_path / "buck-run-report"
    report = json.loads(artifacts.report_json.read_text(encoding="utf-8"))
    source_run = json.loads(before["run.json"])
    assert report["run"]["run_sha256"] == source_run["run_sha256"]
    assert report["provenance"]["source_artifact_manifest"]["verified"] is True
    assert report["resolved_model"]["schema"] == "pear-sim/resolved-model/v0.1"
    assert report["analytic_checks"]["schema"] == "pear-sim/analytic-self-checks/v0.1"
    assert report["analytic_checks"]["claim"]["validation_claim"] is False
    assert report["physics_audit"] == result.physics_audit
    assert report["physics_audit"]["power_sign_convention"] == ("positive_absorbed_from_p_to_n")
    assert report["events"]["count"] == len(result.events)
    rendered = artifacts.report_html.read_text(encoding="utf-8")
    assert "<h2>Measurement &amp; initialization</h2>" in rendered
    assert "<h2>Analytic verification self-check</h2>" in rendered
    assert "<h2>Physics audit</h2>" in rendered
    if result.stats["natural_events"] == 0:
        assert "coverage not exercised" in rendered


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not-json\n", "invalid JSON"),
        ("[]\n", "must contain a JSON object"),
        ("\n", "line 1 is empty"),
        ('{"time": NaN}\n', "invalid JSON"),
        ('{"time": 1e999}\n', "non-finite number"),
        ('{"time": 1, "time": 2}\n', "duplicate JSON object key"),
    ],
)
def test_report_rejects_invalid_or_non_object_event_rows(
    tmp_path: Path, content: str, message: str
):
    run_dir = tmp_path / "run"
    manifest = _write_run(run_dir)
    (run_dir / "events.jsonl").write_text(content, encoding="utf-8")
    _write_source_artifact_manifest(run_dir, manifest)

    with pytest.raises(ReportError, match=message):
        generate_report(run_dir, tmp_path / "rendered")


def test_report_enforces_event_line_row_and_file_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "run"
    manifest = _write_run(run_dir)
    (run_dir / "events.jsonl").write_text(
        json.dumps({"payload": "x" * MAX_EVENT_LINE_BYTES}) + "\n",
        encoding="utf-8",
    )
    _write_source_artifact_manifest(run_dir, manifest)
    with pytest.raises(ReportError, match="line 1.*limit"):
        generate_report(run_dir, tmp_path / "line-report")

    monkeypatch.setattr(report_module, "MAX_EVENT_ROWS", 2)
    (run_dir / "events.jsonl").write_text("{}\n{}\n{}\n", encoding="utf-8")
    _write_source_artifact_manifest(run_dir, manifest)
    with pytest.raises(ReportError, match="2 event-row limit"):
        generate_report(run_dir, tmp_path / "row-report")

    monkeypatch.setattr(report_module, "MAX_EVENT_FILE_BYTES", 5)
    (run_dir / "events.jsonl").write_text("{}\n{}\n", encoding="utf-8")
    _write_source_artifact_manifest(run_dir, manifest)
    with pytest.raises(ReportError, match="limit is 5 bytes"):
        generate_report(run_dir, tmp_path / "file-report")


def test_report_rejects_non_finite_numbers_in_json_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    manifest = _write_run(run_dir)
    (run_dir / "metrics.json").write_text(
        '{"v(out)":{"mean":1e999}}\n',
        encoding="utf-8",
    )
    _write_source_artifact_manifest(run_dir, manifest)

    with pytest.raises(ReportError, match="non-finite number"):
        generate_report(run_dir, tmp_path / "rendered")


@pytest.mark.parametrize(
    ("role", "filename", "content", "message"),
    [
        ("analytic_checks", "analytic-checks.json", "[]\n", "must contain a JSON object"),
        (
            "physics_audit",
            "physics-audit.json",
            '{"full": NaN}\n',
            "non-standard JSON constant",
        ),
        ("mode_coverage", "mode-coverage.json", "[]\n", "must contain a JSON object"),
    ],
)
def test_report_rejects_invalid_optional_verification_artifacts(
    tmp_path: Path,
    role: str,
    filename: str,
    content: str,
    message: str,
) -> None:
    run_dir = tmp_path / "run"
    manifest = _write_run(run_dir)
    files = dict(manifest["files"])
    files[role] = filename
    manifest["files"] = files
    (run_dir / filename).write_text(content, encoding="utf-8")
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    _write_source_artifact_manifest(run_dir, manifest)

    with pytest.raises(ReportError, match=message):
        generate_report(run_dir, tmp_path / "rendered")


def test_report_bounds_mode_coverage_file_and_nested_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    file_run = tmp_path / "file-run"
    file_manifest = _write_run(file_run)
    _attach_mode_coverage(file_run, file_manifest)
    monkeypatch.setattr(report_module, "MAX_MODE_COVERAGE_FILE_BYTES", 10)
    with pytest.raises(ReportError, match="mode-coverage limit is 10 bytes"):
        generate_report(file_run, tmp_path / "file-report")

    monkeypatch.setattr(report_module, "MAX_MODE_COVERAGE_FILE_BYTES", 64 * 1024 * 1024)
    row_run = tmp_path / "row-run"
    row_manifest = _write_run(row_run)
    payload = _mode_coverage_fixture()
    event_timing = dict(payload["event_timing"])
    event_timing["events"] = [dict(event_timing["events"][0]) for _ in range(3)]
    payload["event_timing"] = event_timing
    _attach_mode_coverage(row_run, row_manifest, payload)
    monkeypatch.setattr(report_module, "MAX_MODE_COVERAGE_EVENT_ROWS", 2)
    with pytest.raises(ReportError, match="2 mode-coverage event-row limit"):
        generate_report(row_run, tmp_path / "row-report")

    monkeypatch.setattr(report_module, "MAX_MODE_COVERAGE_EVENT_ROWS", 100_000)
    interval_run = tmp_path / "interval-run"
    interval_manifest = _write_run(interval_run)
    interval_payload = _mode_coverage_fixture()
    interval_payload["intervals"] = [dict(interval_payload["intervals"][0]) for _ in range(3)]
    _attach_mode_coverage(interval_run, interval_manifest, interval_payload)
    monkeypatch.setattr(report_module, "MAX_MODE_COVERAGE_INTERVAL_ROWS", 2)
    with pytest.raises(ReportError, match="2 mode-coverage interval-row limit"):
        generate_report(interval_run, tmp_path / "interval-report")


def test_report_truncates_mode_coverage_events_in_html_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    run_dir = tmp_path / "run"
    manifest = _write_run(run_dir)
    payload = _mode_coverage_fixture()
    event_timing = dict(payload["event_timing"])
    base_event = dict(event_timing["events"][0])
    event_timing["events"] = [
        {**base_event, "index": index, "target": f"coverage-event-{index}"}
        for index in range(3)
    ]
    payload["event_timing"] = event_timing
    _attach_mode_coverage(run_dir, manifest, payload)
    monkeypatch.setattr(report_module, "MAX_MODE_COVERAGE_HTML_EVENTS", 2)

    artifacts = generate_report(run_dir, tmp_path / "report")
    rendered = artifacts.report_html.read_text(encoding="utf-8")
    report = json.loads(artifacts.report_json.read_text(encoding="utf-8"))

    assert "coverage-event-0" in rendered
    assert "coverage-event-1" in rendered
    assert "coverage-event-2" not in rendered
    assert "first 2 of 3 recorded mode-coverage events" in rendered
    assert "Recorded event timing table (2 of 3 rows)" in rendered
    assert report["mode_coverage"]["event_timing"]["events"][2]["target"] == (
        "coverage-event-2"
    )


def test_report_preserves_renderer_semantics_and_uses_report_owned_styles(tmp_path: Path):
    model = load_model(EXAMPLE)
    result = HybridSimulator(model).simulate()
    run_dir = write_result(result, tmp_path / "buck-run")
    source_svg = render_schematic(model).svg

    artifacts = generate_report(run_dir, tmp_path / "buck-report", source_svg)

    assert artifacts.schematic_svg is not None
    sanitized = artifacts.schematic_svg.read_text(encoding="utf-8")
    rendered = artifacts.report_html.read_text(encoding="utf-8")
    assert "<style" not in sanitized
    assert 'class="wire"' in sanitized
    assert 'class="component-label"' in sanitized
    assert 'id="component-q1"' in sanitized
    assert 'data-component-id="q1"' in sanitized
    assert "javascript:" not in sanitized.lower()
    assert ".schematic-svg .wire" in rendered
    assert ".schematic-svg .component-label" in rendered
    assert "buck_ccm" in rendered


def test_waveform_decimation_keeps_pwm_pulses_instead_of_aliasing_them() -> None:
    """40 001 samples of a 20 kHz PWM at 1 us drawn at 800 points: a stride pick lands
    once per carrier period and the chart shows a clean staircase that is not in the
    data. The display must keep every bucket's extremes; a smooth signal stays smooth."""

    import re as _re

    from pear_sim.reporting.report_svg import phase_series_svg, waveform_svg

    time = [index * 1e-6 for index in range(40_001)]
    # 50-sample carrier period; the pulse (10 samples high) never sits on index % 50 == 0.
    pwm = [300.0 if 20 <= index % 50 < 30 else 0.0 for index in range(40_001)]

    def windows_missing_a_level(svg: str) -> int:
        """Windows of ten carrier periods whose plotted points show only one level."""
        points = [
            tuple(float(value) for value in point.split(","))
            for point in _re.search(r'points="([^"]+)"', svg).group(1).split()
        ]
        xs = [x for x, _ in points]
        x0, x1 = min(xs), max(xs)
        windows: dict[int, set[float]] = {}
        for x, y in points:
            windows.setdefault(min(int((x - x0) / (x1 - x0) * 80), 79), set()).add(round(y, 3))
        return sum(1 for levels in windows.values() if len(levels) < 2)

    assert windows_missing_a_level(waveform_svg(time, pwm, "v(pa)")) == 0
    traces = phase_series_svg(time, {"a": pwm, "b": pwm, "c": pwm}, "v", y_label="V")
    assert windows_missing_a_level(traces) == 0

    # Discrimination: a signal without sub-bucket detail is drawn with the same budget.
    ramp = [float(index) for index in range(40_001)]
    points = _re.search(r'points="([^"]+)"', waveform_svg(time, ramp, "ramp")).group(1).split()
    assert len(points) <= 800


def test_losses_section_draws_the_producer_numbers_and_only_when_present(tmp_path: Path) -> None:
    from pear_sim.core.solver import HybridSimulator
    from pear_sim.reporting.report import generate_report as build

    examples = Path(__file__).parents[1] / "examples"
    with_cards = write_result(
        HybridSimulator(load_model(examples / "buck_with_loss_cards.yaml")).simulate(),
        tmp_path / "with",
    )
    html = (build(with_cards, tmp_path / "with-report").report_html).read_text(encoding="utf-8")
    losses = json.loads((with_cards / "losses-thermal.json").read_text(encoding="utf-8"))
    assert "Losses and junction temperatures" in html
    assert f"{losses['total_mean_power_w']:.4g} W" in html
    for device in losses["devices"]:
        assert f"<code>{device['component_id']}</code>" in html
    assert 'aria-label="Mean loss per device, conduction and switching"' in html
    assert 'aria-label="Junction temperature envelopes of the hottest devices"' in html

    without = write_result(
        HybridSimulator(load_model(examples / "buck.yaml")).simulate(), tmp_path / "without"
    )
    plain = (build(without, tmp_path / "without-report").report_html).read_text(encoding="utf-8")
    assert "Losses and junction temperatures" not in plain


def test_waveform_zoom_window_lands_where_pulses_are_readable() -> None:
    """The overview of a long switched run is a filled band; the zoom under it must sit
    where the PWM node switches with a duty near one half, not at the peak, where the
    pulses are narrowest. A run short enough to draw whole gets no zoom."""

    import math

    from pear_sim.reporting.report import _PLOT_POINTS, _ZOOM_SAMPLES, _zoom_window

    period = 50
    length = 40_001
    time = [index * 1e-6 for index in range(length)]
    # Cosine-modulated two-level PWM: the second half starts and ends at duty 0.98.
    duty = [0.5 + 0.48 * math.cos(2 * math.pi * 50 * t) for t in time]
    pwm = [600.0 if (index % period) < duty[index] * period else 0.0 for index in range(length)]
    current = [math.sin(2 * math.pi * 50 * t) for t in time]

    first, last = _zoom_window(time, {"v(pa)": pwm, "i(la)": current})
    assert last - first == _ZOOM_SAMPLES
    assert length // 2 <= first and last < length
    window_duty = sum(1 for value in pwm[first : last + 1] if value > 0) / (last - first + 1)
    assert 0.3 < window_duty < 0.7
    # The rejected rules — centre on the largest |value|, or take the end of the run —
    # land where the duty saturates and the pulses are slivers.
    peak = max(range(length // 2, length), key=lambda index: abs(pwm[index]))
    peak_window = pwm[peak - _ZOOM_SAMPLES // 2 : peak + _ZOOM_SAMPLES // 2]
    assert not 0.3 < sum(1 for value in peak_window if value > 0) / len(peak_window) < 0.7
    end_window = pwm[-_ZOOM_SAMPLES:]
    assert not 0.3 < sum(1 for value in end_window if value > 0) / len(end_window) < 0.7

    assert _zoom_window(time[:_PLOT_POINTS], {"v(pa)": pwm[:_PLOT_POINTS]}) is None
