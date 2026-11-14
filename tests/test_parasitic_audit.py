# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Declared ESR/DCR in the energy audit: losses per component, capacitor RMS
current, and the ripple-current rating check.

The expanded resistors were already inside the energy balance; these tests pin
that the audit REGROUPS them under the user's component without changing the
balance, that the capacitor RMS current derived from the ESR energy agrees
with the same current integrated by a different route (the output-signal
quadrature), and that the rating check is silent below the rating, rings
above it, and refuses to pass silently when nothing was measured.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pear_sim.core import rust_backend
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.evidence.efficiency_map import EfficiencyMapError, build_efficiency_map
from pear_sim.evidence.parameter_sweep import run_parameter_sweep
from pear_sim.run.contracts import model_hashes, resolved_model_dict
from pear_sim.run.io import _ripple_current_warnings
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]

BUCK = """
schema: pear-sim/v0.1
name: buck_parasitic_audit

circuit:
  ground: gnd
  components:
    - {{id: vin, type: voltage_source, p: in, n: gnd, value: 100 V}}
    - {{id: q1, type: ideal_switch, p: in, n: sw, control: pwm, initial: on}}
    - {{id: d1, type: diode, p: gnd, n: sw, initial: off}}
    - {{id: l1, type: inductor, p: sw, n: out, value: 1 mH, {dcr}initial_current: 3.4 A}}
    - {{id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, {esr}initial_voltage: 40 V}}
    - {{id: load, type: resistor, p: out, n: gnd, value: 10 ohm}}

controls:
  pwm: {{type: pwm, frequency: 20 kHz, duty: 40 %, phase: 0 s}}

analysis:
  transient:
    start: 0 s
    stop: 1 ms
    output_step: 1 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11

outputs:
  - v(out)
  - i(l1)
  - i(c1)
"""


def _model(tmp_path: Path, name: str, *, esr: str = "esr: 50 mohm, ", dcr: str = "dcr: 45 mohm, "):
    path = tmp_path / f"{name}.yaml"
    path.write_text(BUCK.format(esr=esr, dcr=dcr), encoding="utf-8")
    return load_model(path)


def _audit(tmp_path: Path, name: str = "m", backend: str = "python", **kw):
    model = _model(tmp_path, name, **kw)
    result = HybridSimulator(model, backend=backend).simulate()
    return model, result


# ---------------------------------------------------------------------------
# Grouping: a regrouping of resistor energy, never a new term in the balance
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("window", ["full", "last_half"])
def test_losses_are_grouped_under_the_user_component(tmp_path: Path, window: str) -> None:
    _, result = _audit(tmp_path)
    audit = result.physics_audit[window]
    energy = audit["component_absorbed_energy_j"]
    duration = audit["duration_s"]
    assert audit["parasitic_loss_by_component_w"] == pytest.approx(
        {"c1": energy["c1.esr"] / duration, "l1": energy["l1.dcr"] / duration}, rel=1e-15
    )
    assert audit["parasitic_absorbed_energy_j"] == pytest.approx(
        energy["c1.esr"] + energy["l1.dcr"], rel=1e-15
    )
    # Still inside the resistor total, and the balance still closes.
    assert audit["parasitic_absorbed_energy_j"] < audit["resistor_absorbed_energy_j"]
    assert audit["balance_status"] == "within_numerical_tolerance"


def test_capacitor_rms_from_esr_energy_matches_the_signal_quadrature(tmp_path: Path) -> None:
    """Two routes to one number: I_rms = sqrt(E_esr / (ESR*T)) from the
    per-component power quadrature, and sqrt(integral of i(c1)^2 / T) from the
    output-signal quadrature. They share the dense solution, not the formula."""

    _, result = _audit(tmp_path)
    for window in ("full", "last_half"):
        from_esr = result.physics_audit[window]["capacitor_rms_current_a"]["c1"]
        signal = result.time_integrals[window]
        from_signal = math.sqrt(
            signal["signals"]["i(c1)"]["square_integral"] / signal["duration_s"]
        )
        assert from_esr == pytest.approx(from_signal, rel=1e-9), window
    # An inductor has no capacitor RMS entry: its DCR is grouped, not rated.
    assert set(result.physics_audit["full"]["capacitor_rms_current_a"]) == {"c1"}


@pytest.mark.skipif(not rust_backend.available(), reason="optional Rust kernel not built")
def test_the_rust_kernel_reports_the_same_parasitic_audit(tmp_path: Path) -> None:
    _, py = _audit(tmp_path, "py", "python")
    _, rs = _audit(tmp_path, "rs", "rust")
    for window in ("full", "last_half"):
        a, b = py.physics_audit[window], rs.physics_audit[window]
        assert b["parasitic_loss_by_component_w"] == pytest.approx(
            a["parasitic_loss_by_component_w"], rel=1e-9
        )
        assert b["capacitor_rms_current_a"] == pytest.approx(
            a["capacitor_rms_current_a"], rel=1e-9
        )


def test_a_model_without_parasitics_carries_no_new_fields(tmp_path: Path) -> None:
    model, result = _audit(tmp_path, esr="", dcr="")
    for window in ("full", "last_half"):
        audit = result.physics_audit[window]
        assert "parasitic_loss_by_component_w" not in audit
        assert "capacitor_rms_current_a" not in audit
    assert "passive_parasitics" not in resolved_model_dict(model)
    assert _ripple_current_warnings(result) == []


# ---------------------------------------------------------------------------
# Ripple-current rating: silent below, rings above, never silently unchecked
# ---------------------------------------------------------------------------


def _rated(tmp_path: Path, rating: str, name: str, evidence_mode: str = "full"):
    model = _model(tmp_path, name, esr=f"esr: 50 mohm, ripple_current_rating: {rating}, ")
    result = HybridSimulator(model, evidence_mode=evidence_mode).simulate()
    return result


def test_the_rating_check_is_silent_below_and_rings_above(tmp_path: Path) -> None:
    measured = _audit(tmp_path)[1].physics_audit["last_half"]["capacitor_rms_current_a"]["c1"]
    assert 0.1 < measured < 10.0  # the bracketing below must straddle a real current

    below = _rated(tmp_path, f"{measured * 1.05} A", "ok")
    assert _ripple_current_warnings(below) == []

    above = _rated(tmp_path, f"{measured * 0.95} A", "hot")
    (warning,) = _ripple_current_warnings(above)
    assert warning["code"] == "CAPACITOR_RIPPLE_CURRENT_EXCEEDED"
    assert warning["component"] == "c1"
    assert warning["rms_current_a"] == pytest.approx(measured, rel=1e-12)
    assert warning["window"] == "last_half"


def test_a_lean_run_says_the_rating_was_not_checked(tmp_path: Path) -> None:
    result = _rated(tmp_path, "1 mA", "lean", evidence_mode="lean")
    (warning,) = _ripple_current_warnings(result)
    assert warning["code"] == "CAPACITOR_RIPPLE_CURRENT_NOT_ASSESSED"
    assert warning["components"] == ["c1"]


def test_the_warning_reaches_the_written_bundle(tmp_path: Path) -> None:
    from pear_sim.run.io import write_result

    result = _rated(tmp_path, "1 mA", "bundle")
    out = tmp_path / "run"
    write_result(result, out)
    codes = {item["code"] for item in json.loads((out / "warnings.json").read_text())}
    assert "CAPACITOR_RIPPLE_CURRENT_EXCEEDED" in codes
    resolved = json.loads((out / "model.resolved.json").read_text())
    assert resolved["passive_parasitics"] == [
        {"component": "c1", "kind": "esr", "resistor": "c1.esr", "ripple_current_rating_a": 0.001},
        {"component": "l1", "kind": "dcr", "resistor": "l1.dcr"},
    ]


def test_the_rating_is_a_check_not_physics(tmp_path: Path) -> None:
    """A rating does not change the simulated circuit, so it must not change
    the physics hash; the ESR it rides on does."""

    plain = model_hashes(_model(tmp_path, "a"))
    rated = model_hashes(_model(tmp_path, "b", esr="esr: 50 mohm, ripple_current_rating: 2 A, "))
    other_esr = model_hashes(_model(tmp_path, "c", esr="esr: 60 mohm, "))
    assert rated["model_sha256"] == plain["model_sha256"]
    assert other_esr["model_sha256"] != plain["model_sha256"]


@pytest.mark.parametrize(
    ("esr", "dcr", "message"),
    [
        ("ripple_current_rating: 2 A, ", "", r"c1\.ripple_current_rating: needs an `esr`"),
        ("esr: 50 mohm, ripple_current_rating: 0 A, ", "", r"positive, finite current"),
        ("esr: 50 mohm, ripple_current_rating: 2 V, ", "", r"c1\.ripple_current_rating"),
        ("", "dcr: 45 mohm, ripple_current_rating: 2 A, ", r"l1\.ripple_current_rating: only valid for capacitor"),
    ],
)
def test_bad_ratings_are_rejected(tmp_path: Path, esr: str, dcr: str, message: str) -> None:
    with pytest.raises(ModelError, match=message):
        _model(tmp_path, "bad", esr=esr, dcr=dcr)


# ---------------------------------------------------------------------------
# Efficiency map: declared parasitics are losses without being asked
# ---------------------------------------------------------------------------

LOSS_BUCK = ROOT / "examples" / "buck_with_loss_cards.yaml"


def _sweep(tmp_path: Path, model_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps({
            "schema": "pear-sim/parameter-sweep-request/v0.1",
            "parameter": "load",
            "values": ["4 ohm"],
        }),
        encoding="utf-8",
    )
    return run_parameter_sweep(model_path, request, tmp_path / "sweep").output_dir


def test_the_efficiency_map_counts_declared_parasitics_and_refuses_double_counting(
    tmp_path: Path,
) -> None:
    text = LOSS_BUCK.read_text(encoding="utf-8")
    inductor = "value: 1 mH, initial_current"
    capacitor = "value: 100 uF, initial_voltage"
    assert text.count(inductor) == 1 and text.count(capacitor) == 1
    lossy_path = tmp_path / "buck_parasitics.yaml"
    lossy_path.write_text(
        text.replace(inductor, "value: 1 mH, dcr: 45 mohm, initial_current").replace(
            capacitor, "value: 100 uF, esr: 50 mohm, initial_voltage"
        ),
        encoding="utf-8",
    )
    plain = build_efficiency_map(_sweep(tmp_path / "plain", LOSS_BUCK))
    lossy_dir = _sweep(tmp_path / "lossy", lossy_path)
    lossy = build_efficiency_map(lossy_dir)
    point = lossy["points"][0]
    assert point["declared_parasitic_loss_w"] > 0.0
    assert point["parasitic_loss_w"] == point["declared_parasitic_loss_w"]
    assert "declared_parasitic_loss_w" not in plain["points"][0]
    assert point["efficiency"] < plain["points"][0]["efficiency"]
    with pytest.raises(EfficiencyMapError, match="already counted"):
        build_efficiency_map(lossy_dir, parasitic_components=("l1.dcr",))
