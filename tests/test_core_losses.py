# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Core-loss post-processing: iGSE identities, loader refusals, end-to-end run.

The identity checks solve each quantity by a DIFFERENT method than the
producer (numerical quadrature for the |cos|^alpha integral, the closed-form
piecewise-linear iGSE for triangles), so a shared wrong assumption cannot
pass on both sides.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.integrate import quad

from pear_sim.analysis.core_losses import (
    CoreLossError,
    igse_ki,
    igse_volumetric_loss,
)
from pear_sim.errors import ModelError
from pear_sim.core.solver import HybridSimulator
from pear_sim.run.io import write_result
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "buck_core_losses_n87.yaml"

K, ALPHA, BETA = 3.0335883, 1.5224303, 2.8878710


def test_ki_matches_numerical_quadrature():
    for alpha, beta in [(1.1, 2.5), (1.5224303, 2.887871), (2.3, 2.6)]:
        integral, _ = quad(lambda x, a=alpha: abs(math.cos(x)) ** a, 0.0, 2.0 * math.pi, limit=200)
        expected = 1.0 / ((2.0 * math.pi) ** (alpha - 1.0) * 2.0 ** (beta - alpha) * integral)
        assert igse_ki(1.0, alpha, beta) == pytest.approx(expected, rel=1e-10)


def test_igse_reduces_to_steinmetz_for_a_sinusoid():
    frequency, amplitude = 100e3, 0.1
    time = np.linspace(0.0, 1.0 / frequency, 20001)
    flux = amplitude * np.sin(2.0 * math.pi * frequency * time)
    result = igse_volumetric_loss(time, flux, k=K, alpha=ALPHA, beta=BETA)
    steinmetz = K * frequency**ALPHA * amplitude**BETA
    # Grid error only (sampled sine); the identity itself is exact.
    assert result["volumetric_loss_w_per_m3"] == pytest.approx(steinmetz, rel=1e-6)
    assert result["direction_changes"] == 2


@pytest.mark.parametrize("duty", [0.1, 0.25, 0.5, 0.8])
def test_igse_triangle_matches_closed_form(duty):
    frequency, swing = 50e3, 0.2
    period = 1.0 / frequency
    time = np.array([0.0, duty * period, period])
    flux = np.array([-swing / 2.0, swing / 2.0, -swing / 2.0])
    result = igse_volumetric_loss(time, flux, k=K, alpha=ALPHA, beta=BETA)
    closed = (
        igse_ki(K, ALPHA, BETA)
        * swing**BETA
        * frequency**ALPHA
        * (duty ** (1.0 - ALPHA) + (1.0 - duty) ** (1.0 - ALPHA))
    )
    assert result["volumetric_loss_w_per_m3"] == pytest.approx(closed, rel=1e-12)


def test_flat_segments_are_not_minor_loops():
    time = np.array([0.0, 0.3, 0.5, 0.8, 1.0]) * 1e-5
    flux = np.array([-0.1, 0.1, 0.1, -0.1, -0.1])
    result = igse_volumetric_loss(time, flux, k=K, alpha=ALPHA, beta=BETA)
    assert result["direction_changes"] == 2
    assert result["volumetric_loss_w_per_m3"] > 0.0


def test_minor_loops_are_refused_not_approximated():
    time = np.linspace(0.0, 1e-5, 401)
    flux = 0.1 * np.sin(2 * math.pi * 1e5 * time) + 0.03 * np.sin(2 * math.pi * 5e5 * time)
    with pytest.raises(CoreLossError, match="minor"):
        igse_volumetric_loss(time, flux, k=K, alpha=ALPHA, beta=BETA)


# ---------------------------------------------------------------- loader


def _example_document() -> dict:
    return yaml.safe_load(EXAMPLE.read_text())


def _load(tmp_path: Path, document: dict):
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return load_model(path)


def test_example_loads_with_core_losses(tmp_path):
    model = _load(tmp_path, _example_document())
    assert model.core_losses is not None
    assert model.core_losses.period_s == pytest.approx(10e-6)
    assert model.core_losses.bindings[0].inductor_id == "l1"


def _mutate(document: dict, path: list, value) -> dict:
    target = document
    for key in path[:-1]:
        target = target[key]
    if value is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return document


_DELETE = object()
_MATERIAL = ["core_losses", "materials", "n87"]
_RANGE0 = _MATERIAL + ["steinmetz_ranges", 0]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (["core_losses", "unknown"], 1, "unknown"),
        (_MATERIAL + ["provenance"], _DELETE, "provenance"),
        (_MATERIAL + ["validity"], _DELETE, "validity"),
        (_RANGE0 + ["beta"], 1.2, "exponents"),
        (_RANGE0 + ["ct2"], -1.0, "ct\\(T\\)"),
        (_RANGE0 + ["frequency_max_hz"], 200000, "overlap"),
        (["core_losses", "evaluation_temperature_c"], 150, "validity envelope"),
        (["core_losses", "inductors", "l1", "material"], "n97", "not a declared"),
        (["core_losses", "inductors", "l1", "turns"], 0, "positive"),
        (["core_losses", "period"], "1 s", "longer than the transient"),
        (["outputs"], ["v(out)"], "i\\(l1\\)"),
        (["analysis", "transient", "output_step"], "1 us", "64 recorded samples"),
    ],
)
def test_loader_refuses(tmp_path, path, value, message):
    document = _mutate(_example_document(), path, value)
    with pytest.raises(ModelError, match=message):
        _load(tmp_path, document)


def test_loader_refuses_a_non_inductor_binding(tmp_path):
    document = _example_document()
    document["core_losses"]["inductors"] = {
        "c1": dict(document["core_losses"]["inductors"]["l1"])
    }
    document["outputs"].append("i(c1)")
    with pytest.raises(ModelError, match="linear inductor"):
        _load(tmp_path, document)


def test_loader_refuses_an_unbound_material(tmp_path):
    document = _example_document()
    materials = document["core_losses"]["materials"]
    materials["spare"] = json.loads(json.dumps(materials["n87"]))
    with pytest.raises(ModelError, match="never bound"):
        _load(tmp_path, document)


# ---------------------------------------------------------------- end to end


@pytest.fixture(scope="module")
def example_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("core_losses_run")
    write_result(HybridSimulator(load_model(EXAMPLE)).simulate(), out)
    return out


def test_run_writes_core_losses_artifact(example_run):
    manifest = json.loads((example_run / "run.json").read_text())
    assert manifest["files"]["core_losses"] == "core-losses.json"
    artifact = json.loads((example_run / "core-losses.json").read_text())
    assert artifact["schema"] == "pear-sim/core-losses/v0.1"
    (record,) = artifact["inductors"]
    assert record["status"] == "estimated"
    assert record["flags"] == []
    assert record["flux_density"]["periodic_closure_relative"] < 1e-4
    summary = json.loads((example_run / "summary.json").read_text())
    assert summary["core_losses"]["total_power_w"] == pytest.approx(record["power_w"])


def test_run_loss_matches_closed_form_triangle(example_run):
    """Ideal buck at D = 0.5: B(t) is a symmetric triangle with
    dB = L*dI/(N*Ae), dI = (Vin - Vout)*D/(f*L). Corners are not on the
    output grid, so this bounds the grid error of the recorded-sample iGSE."""

    artifact = json.loads((example_run / "core-losses.json").read_text())
    (record,) = artifact["inductors"]
    vin, vout, duty, frequency = 48.0, 24.0, 0.5, 100e3
    inductance, turns, area, volume = 47e-6, 12.0, 7.6e-5, 5.47e-6
    swing = inductance * ((vin - vout) * duty / (frequency * inductance)) / (turns * area)
    selected = record["steinmetz_range"]
    k, alpha, beta = selected["k"], selected["alpha"], selected["beta"]
    closed = (
        igse_ki(k, alpha, beta)
        * swing**beta
        * frequency**alpha
        * (duty ** (1.0 - alpha) + (1.0 - duty) ** (1.0 - alpha))
        * record["temperature_factor"]
    )
    # The capacitor ripple makes the real slopes differ slightly from the
    # ideal constant-voltage triangle; 1 % covers that and the grid.
    assert record["volumetric_loss_w_per_m3"] == pytest.approx(closed, rel=1e-2)
    assert record["power_w"] == pytest.approx(closed * volume, rel=1e-2)


def test_frequency_outside_ranges_is_flagged(tmp_path):
    document = _example_document()
    document["core_losses"]["materials"]["n87"]["steinmetz_ranges"] = [
        document["core_losses"]["materials"]["n87"]["steinmetz_ranges"][1]
    ]
    model = _load(tmp_path, document)
    out = tmp_path / "run"
    write_result(HybridSimulator(model).simulate(), out)
    artifact = json.loads((out / "core-losses.json").read_text())
    codes = [flag["code"] for flag in artifact["inductors"][0]["flags"]]
    assert codes == ["FREQUENCY_OUTSIDE_RANGES"]
    assert artifact["inductors"][0]["flags"][0]["signed_distance_hz"] == pytest.approx(-50e3)
