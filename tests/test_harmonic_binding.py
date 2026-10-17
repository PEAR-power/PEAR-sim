from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import pytest

from pear_sim.io import write_result
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator, SimulationResult


ROOT = Path(__file__).parents[1]
SPWM_EXAMPLE = ROOT / "examples" / "inverter_three_phase_spwm.yaml"


@pytest.fixture(scope="module")
def spwm_result() -> SimulationResult:
    result = HybridSimulator(load_model(SPWM_EXAMPLE)).simulate()
    assert result.harmonic_analysis is not None
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mutate_payload(payload: dict[str, Any], case: str) -> None:
    if case == "window_start":
        payload["window"]["start_s"] = 0.003
    elif case == "window_stop":
        payload["window"]["stop_s"] = 0.007
    elif case == "window_duration":
        payload["window"]["duration_s"] = 0.003
    elif case == "period_count":
        payload["window"]["fundamental_period_count"] = 1
    elif case == "fundamental_frequency":
        payload["frequency"]["fundamental_hz"] = 501.0
    elif case == "max_harmonic":
        payload["frequency"]["max_harmonic"] = 39
    elif case == "bandwidth":
        payload["frequency"]["bandwidth_hz"] = 19_999.0
    elif case == "signal":
        payload["signals"][0]["signal"] = "v(vbc)"
    elif case == "semantic":
        payload["signals"][0]["semantic"] = "line_voltage_bc"
    elif case == "unit":
        payload["signals"][0]["unit"] = "A"
    elif case == "signal_order":
        payload["signals"][0], payload["signals"][1] = (
            payload["signals"][1],
            payload["signals"][0],
        )
    elif case == "coefficient_count":
        payload["signals"][0]["coefficients"].pop()
    elif case == "coefficient_order":
        payload["signals"][0]["coefficients"][2]["harmonic"] = 3
    elif case == "coefficient_frequency":
        payload["signals"][0]["coefficients"][2]["frequency_hz"] = 999.0
    else:  # pragma: no cover - test-table programming error
        raise AssertionError(f"Unknown mutation case {case!r}")


def test_real_spwm_harmonics_are_request_and_hash_bound(
    tmp_path: Path,
    spwm_result: SimulationResult,
) -> None:
    run_dir = write_result(spwm_result, tmp_path / "run")
    run = _read_json(run_dir / "run.json")
    artifact = _read_json(run_dir / "harmonics.json")

    assert artifact["window"] == spwm_result.harmonic_analysis["window"]
    assert artifact["frequency"] == spwm_result.harmonic_analysis["frequency"]
    assert artifact["signals"] == spwm_result.harmonic_analysis["signals"]
    assert artifact["source_binding"] == {
        "model_sha256": run["model_sha256"],
        "request_sha256": run["request_sha256"],
        "run_sha256": run["run_sha256"],
        "recorded_waveform_artifact": "waveforms.csv",
    }
    assert spwm_result.harmonic_analysis["source_binding"] == {}


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("window_start", r"window\.start_s disagrees"),
        ("window_stop", r"window\.stop_s disagrees"),
        ("window_duration", r"window\.duration_s disagrees"),
        ("period_count", r"window\.fundamental_period_count disagrees"),
        ("fundamental_frequency", r"frequency\.fundamental_hz disagrees"),
        ("max_harmonic", r"frequency\.max_harmonic disagrees"),
        ("bandwidth", r"frequency\.bandwidth_hz disagrees"),
        ("signal", r"signals\[0\]\.signal disagrees"),
        ("semantic", r"signals\[0\]\.semantic disagrees"),
        ("unit", r"signals\[0\]\.unit disagrees"),
        ("signal_order", r"signals\[0\]\.signal disagrees"),
        ("coefficient_count", r"coefficients must contain exactly harmonic orders"),
        ("coefficient_order", r"coefficients\[2\]\.harmonic disagrees"),
        ("coefficient_frequency", r"coefficients\[2\]\.frequency_hz disagrees"),
    ],
)
def test_harmonic_binding_rejects_request_and_coefficient_mutations(
    tmp_path: Path,
    spwm_result: SimulationResult,
    case: str,
    message: str,
) -> None:
    payload = deepcopy(spwm_result.harmonic_analysis)
    _mutate_payload(payload, case)

    with pytest.raises(ValueError, match=message):
        write_result(
            replace(spwm_result, harmonic_analysis=payload),
            tmp_path / "run",
        )


def test_harmonic_payload_is_rejected_without_an_explicit_model_request(
    tmp_path: Path,
    spwm_result: SimulationResult,
) -> None:
    buck_result = HybridSimulator(load_model(ROOT / "examples" / "buck.yaml")).simulate()
    injected = replace(
        buck_result,
        harmonic_analysis=deepcopy(spwm_result.harmonic_analysis),
    )

    with pytest.raises(ValueError, match="model contains no analysis.harmonics request"):
        write_result(injected, tmp_path / "run")


def test_explicit_harmonic_request_rejects_missing_producer_payload(
    tmp_path: Path,
    spwm_result: SimulationResult,
) -> None:
    missing = replace(spwm_result, harmonic_analysis=None)

    with pytest.raises(ValueError, match="harmonic_analysis is missing"):
        write_result(missing, tmp_path / "run")
