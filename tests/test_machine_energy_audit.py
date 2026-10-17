"""Energy audit and power accounting for models with machines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pear_sim.core.solver import HybridSimulator
from pear_sim.run.io import write_result
from pear_sim.spec.model import load_model

EXAMPLES = Path(__file__).parents[1] / "examples"


def _run(name: str, tmp_path: Path):
    result = HybridSimulator(load_model(EXAMPLES / f"{name}.yaml")).simulate()
    write_result(result, tmp_path / name)
    warnings = json.loads((tmp_path / name / "warnings.json").read_text())
    return result.physics_audit["full"], {w["code"] for w in warnings}


def test_position_dependent_inductance_is_not_assessed_rather_than_failed(tmp_path) -> None:
    """A salient rotor exchanges energy through dL/dtheta (reluctance torque);
    the audit stores 1/2 L i^2 with constant L and cannot see that term, so it
    must not call the run numerically wrong — nor blame a stiff solver."""

    audit, codes = _run("synrm_torque_test", tmp_path)
    assert audit["balance_status"] == "not_assessed"
    assert "position-dependent" in audit["balance_not_assessed_reason"]
    assert "ENERGY_BALANCE_OUTSIDE_TOLERANCE" not in codes
    assert "ENERGY_AUDIT_NOT_ASSESSED" in codes


def test_a_non_salient_machine_keeps_a_closed_energy_balance(tmp_path) -> None:
    audit, codes = _run("pmsm_inverter_drive", tmp_path)
    assert audit["balance_status"] == "within_numerical_tolerance"
    assert "ENERGY_AUDIT_NOT_ASSESSED" not in codes


def test_machine_back_emf_is_not_counted_as_a_supply(tmp_path) -> None:
    """The back-EMF branches of a machine absorb the shaft power while motoring;
    counting them with the DC link as "sources" netted the supply down to the
    copper losses. Supplies and machine electromagnetic power are reported apart."""

    audit, _ = _run("pmsm_inverter_drive", tmp_path)
    energies = audit["component_absorbed_energy_j"]
    supplies = -sum(v for k, v in energies.items() if k in ("vp", "vn", "vdc", "vin"))
    machine = sum(v for k, v in energies.items() if k.startswith("motor.") and k.endswith(".e"))
    assert audit["source_delivered_energy_j"] == pytest.approx(supplies, rel=1e-9)
    assert audit["machine_electromagnetic_energy_j"] == pytest.approx(machine, rel=1e-9)


def test_generic_harmonics_resolve_the_six_pulse_dc_ripple(tmp_path) -> None:
    """A six-pulse bridge ripples at six times the grid frequency. With the
    `generic` semantic the DC-side current can be analysed directly: its
    dominant harmonic is the 6th (300 Hz), and THD is undefined because a DC
    signal has no fundamental to normalise by."""

    text = (EXAMPLES / "rectifier_6pulse_transformer.yaml").read_text().replace(
        "    atol: 1.0e-9\n\noutputs:",
        "    atol: 1.0e-9\n  harmonics:\n    start: 0 s\n    stop: 20 ms\n"
        "    fundamental_frequency: \"${fgrid}\"\n    max_harmonic: 24\n"
        "    signals:\n      - {signal: i(rl), semantic: generic}\n"
        "      - {signal: v(ytop), semantic: generic}\n\noutputs:",
    )
    assert "semantic: generic" in text
    path = tmp_path / "six_pulse.yaml"
    path.write_text(text)
    result = HybridSimulator(load_model(path)).simulate()
    record = next(s for s in result.harmonic_analysis["signals"] if s["signal"] == "i(rl)")
    ac = [c for c in record["coefficients"] if c["harmonic"] >= 1]
    dominant = max(ac, key=lambda c: c["peak_amplitude"])
    assert dominant["harmonic"] == 6
    assert record["thd"]["status"] != "observed"
