from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import pear_sim.report as report_module
from pear_sim.io import write_result
from pear_sim.model import load_model
from pear_sim.report import ReportError, generate_report
from pear_sim.solver import HybridSimulator
from pear_sim.limits import MAX_HARMONIC_ORDER


ROOT = Path(__file__).parents[1]


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _harmonic_payload(signal: str = "v(out)") -> dict[str, object]:
    return {
        "schema": "pear-sim/harmonic-analysis/v0.1",
        "kind": "harmonic-analysis-observation",
        "status": "observed",
        "claim": {
            "validation_claim": False,
            "hardware_sign_off_allowed": False,
            "statement": "Producer values only; no validation verdict.",
        },
        "source_binding": {"run_sha256": "test-binding"},
        "window": {
            "start_s": 0.004,
            "stop_s": 0.008,
            "duration_s": 0.004,
            "endpoint_convention": "half_open_[start,stop)",
            "fundamental_period_count": 2,
            "source": "solver_dense_output_on_event_aligned_topology_intervals",
        },
        "frequency": {
            "fundamental_hz": 500.0,
            "max_harmonic": 2,
            "bandwidth_hz": 1000.0,
        },
        "method": {
            "name": "event_aligned_dense_output_gauss_legendre_fourier",
            "window_function": "rectangular",
            "dc_policy": "retained_as_coefficient_0_excluded_from_thd",
            "amplitude_convention": "producer-defined-test-convention",
            "report_recomputes_values": False,
        },
        "signals": [
            {
                "signal": signal,
                "semantic": "line_voltage_ab",
                "unit": "V",
                "dc": 1.25,
                "fundamental": {
                    "frequency_hz": 500.0,
                    "peak_amplitude": 174.594,
                    "rms_amplitude": 123.456,
                    "phase_deg": -30.0,
                },
                "thd": {
                    "status": "observed",
                    "value_fraction": 0.987,
                    "harmonic_rms": 121.851,
                    "fundamental_rms": 123.456,
                    "included_harmonics": "2..2",
                },
                "coefficients": [
                    {
                        "harmonic": 0,
                        "frequency_hz": 0.0,
                        "peak_amplitude": 1.25,
                        "rms_amplitude": 1.25,
                        "phase_deg": 0.0,
                    },
                    {
                        "harmonic": 1,
                        "frequency_hz": 500.0,
                        "peak_amplitude": 174.594,
                        "rms_amplitude": 123.456,
                        "phase_deg": -30.0,
                    },
                    {
                        "harmonic": 2,
                        "frequency_hz": 1000.0,
                        "peak_amplitude": 172.324,
                        "rms_amplitude": 121.851,
                        "phase_deg": 15.0,
                    },
                ],
            }
        ],
        "sequence_components": [],
        "summary": {
            "signal_count": 1,
            "undefined_thd_count": 0,
            "acceptance_status": "not_evaluated",
        },
    }


def _add_harmonics(run_dir: Path, payload: dict[str, object]) -> None:
    run_path = run_dir / "run.json"
    run = _read_json(run_path)
    payload["source_binding"] = {
        "model_sha256": run["model_sha256"],
        "request_sha256": run["request_sha256"],
        "run_sha256": run["run_sha256"],
        "recorded_waveform_artifact": "waveforms.csv",
    }
    harmonic_path = run_dir / "harmonics.json"
    harmonic_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run["files"]["harmonics"] = harmonic_path.name
    run_path.write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest_path = run_dir / "artifact-manifest.json"
    manifest = _read_json(manifest_path)
    records = [record for record in manifest["artifacts"] if record["path"] != harmonic_path.name]
    by_path = {record["path"]: record for record in records}
    for path in (run_path, harmonic_path):
        data = path.read_bytes()
        by_path[path.name] = {
            "path": path.name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    manifest["artifacts"] = [by_path[name] for name in sorted(by_path)]
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_report_copies_and_renders_harmonic_producer_values_without_recomputing(
    tmp_path: Path,
) -> None:
    run_dir = write_result(
        HybridSimulator(load_model(ROOT / "examples" / "buck.yaml")).simulate(),
        tmp_path / "run",
    )
    hostile = "v(out)<script>alert('harmonic')</script>&"
    payload = _harmonic_payload(hostile)
    _add_harmonics(run_dir, payload)

    first = generate_report(run_dir, tmp_path / "report")
    first_json = first.report_json.read_bytes()
    first_html = first.report_html.read_bytes()
    second = generate_report(run_dir, tmp_path / "report")
    assert second.report_json.read_bytes() == first_json
    assert second.report_html.read_bytes() == first_html

    report = json.loads(first_json)
    assert report["harmonics"] == payload
    rendered = first_html.decode("utf-8")
    lowered = rendered.lower()
    assert "Harmonic analysis" in rendered
    assert "THE REPORT DOES NOT RECOMPUTE FOURIER KPI" in rendered
    assert "event_aligned_dense_output_gauss_legendre_fourier" in rendered
    assert "half_open_[start,stop)" in rendered
    assert "123.456" in rendered
    assert "0.987" in rendered
    assert "Fundamental RMS [V]" in rendered
    assert "RMS [V]" in rendered
    assert "Producer harmonic coefficients" in rendered
    assert "harmonics.json remains authoritative" in rendered
    assert "v(out)&lt;script&gt;alert(&#x27;harmonic&#x27;)&lt;/script&gt;&amp;" in rendered
    assert "<script" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "fetch(" not in lowered


def test_write_result_rejects_harmonic_artifact_without_model_request(tmp_path: Path) -> None:
    result = HybridSimulator(load_model(ROOT / "examples" / "buck.yaml")).simulate()
    payload = _harmonic_payload()
    payload["source_binding"] = {}
    result.harmonic_analysis = payload
    with pytest.raises(ValueError, match="contains no analysis.harmonics request"):
        write_result(result, tmp_path / "run")


def test_report_rejects_harmonic_tampering_and_bounds_nested_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = write_result(
        HybridSimulator(load_model(ROOT / "examples" / "buck.yaml")).simulate(),
        tmp_path / "run",
    )
    payload = _harmonic_payload()
    _add_harmonics(run_dir, payload)
    harmonic_path = run_dir / "harmonics.json"
    harmonic_path.write_bytes(harmonic_path.read_bytes() + b" \n")
    with pytest.raises(
        ReportError,
        match="SHA-256 mismatch for source artifact 'harmonics.json'",
    ):
        generate_report(run_dir, tmp_path / "tampered-report")

    # Restore a valid hash-bound artifact, then prove the nested reader cap is active.
    _add_harmonics(run_dir, payload)
    monkeypatch.setattr(report_module, "MAX_HARMONICS_PER_SIGNAL", 1)
    with pytest.raises(ReportError, match="coefficients must contain at most 1 rows"):
        generate_report(run_dir, tmp_path / "bounded-report")


@pytest.mark.parametrize("unit", [None, "", "W", "<script>"])
def test_report_rejects_missing_or_unsupported_harmonic_units(
    tmp_path: Path,
    unit: object,
) -> None:
    run_dir = write_result(
        HybridSimulator(load_model(ROOT / "examples" / "buck.yaml")).simulate(),
        tmp_path / "run",
    )
    payload = _harmonic_payload()
    payload["signals"][0]["unit"] = unit
    _add_harmonics(run_dir, payload)

    # Asserted against the DERIVATION, not against a hard-coded list: the set
    # of legal units grows as new signal kinds are declared, and a test pinned
    # to today's list would fail on growth while the rule still held.
    with pytest.raises(ReportError, match="unit must be one of"):
        generate_report(run_dir, tmp_path / "report")


def test_maximum_legal_harmonic_order_survives_run_and_report_pipeline(
    tmp_path: Path,
) -> None:
    model = load_model(ROOT / "examples" / "inverter_three_phase_spwm.yaml")
    assert model.harmonics is not None
    model = replace(
        model,
        harmonics=replace(model.harmonics, max_harmonic=MAX_HARMONIC_ORDER),
    )
    result = HybridSimulator(model).simulate()
    run_dir = write_result(result, tmp_path / "run")
    artifacts = generate_report(run_dir, tmp_path / "report")

    source = _read_json(run_dir / "harmonics.json")
    report = _read_json(artifacts.report_json)
    expected_rows = MAX_HARMONIC_ORDER + 1
    assert source["frequency"]["max_harmonic"] == MAX_HARMONIC_ORDER
    assert all(len(signal["coefficients"]) == expected_rows for signal in source["signals"])
    assert report["harmonics"] == source
    assert f"max harmonic={MAX_HARMONIC_ORDER}" in artifacts.report_html.read_text(encoding="utf-8")
