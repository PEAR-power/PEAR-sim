# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""motulator as an independent engine for PEARSim's machine models.

Two layers:

* adapter tests that need no motulator: the subset is refused outside its
  bounds, the supply mapping is right, the grid check refuses to interpolate,
  and the worker's flux-map interpolation is PEARSim's own;
* engine comparisons, run only when ``PEAR_SIM_MOTULATOR_PYTHON`` names an
  interpreter with motulator (CI installs one). They are VERIFICATION
  between two simulators, never validation against a machine.

The gates were set from measurement (2026-11-15, motulator 0.24.1):

* non-salient machine at constant speed: the engines agree to ~1e-12 of the
  peak — PEARSim integrates that case exactly, so anything above 1e-9 is a
  defect, not tolerance;
* salient rotor, speed ramp and a flux map in its unsaturated region:
  PEARSim HOLDS the speed and the inductance matrix per output step, a
  first-order splitting error. Measured worst error at 10 us: salient 0.99 %,
  ramp 0.041 %, salient + ramp 0.66 %, flux map 0.38 %; each halves at 5 us
  (ratio 0.50). The gate asks for <= 2 % at 10 us AND a ratio in
  [0.45, 0.55]: a modelling error (wrong phase sequence, swapped axes) does
  not shrink with the step and fails it;
* a flux map driven INTO saturation: the exact solution peaks at 15.7 A,
  well inside the +-40 A map, but PEARSim diverges (i_d -> -67 A and a
  refusal). Recorded as a strict xfail — see the test's docstring.
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.oracle.motulator import (
    MotulatorOracleError,
    MotulatorSimulation,
    compare_with_result,
    emit_request,
    find_motulator_python,
    simulate_model,
)
from pear_sim.oracle.motulator_worker import _bilinear, _current_map
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "pmsm_sine_fed_driven.yaml"
FLUX_MAP_EXAMPLE = ROOT / "examples" / "pmsm_saturating_flux_map.yaml"
PYTHON = find_motulator_python()
if os.environ.get("PEAR_SIM_REQUIRE_MOTULATOR") == "1" and PYTHON is None:
    # The CI job that installs motulator sets this, so a broken install fails
    # loudly instead of turning every engine comparison into a silent skip.
    pytest.fail("PEAR_SIM_REQUIRE_MOTULATOR=1 but no motulator interpreter is configured", pytrace=False)
needs_motulator = pytest.mark.skipif(
    PYTHON is None, reason="PEAR_SIM_MOTULATOR_PYTHON is not set to a motulator interpreter"
)


def _document() -> dict:
    return yaml.safe_load(EXAMPLE.read_text())


def _load(tmp_path: Path, document: dict, name: str = "model.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    return load_model(path)


# ---------------------------------------------------------------- adapter


def test_request_maps_the_supply_and_the_machine(tmp_path):
    request = emit_request(_load(tmp_path, _document()))
    # PEARSim phase a = A*sin(w t + 200 deg) = A*cos(w t + 110 deg)
    assert request["source"]["phase_rad"] == pytest.approx(math.radians(110.0))
    assert request["source"]["amplitude_v"] == pytest.approx(70.0)
    assert request["source"]["angular_frequency_rad_s"] == pytest.approx(math.tau * 100.0)
    machine = request["machine"]
    assert machine["magnetics"] == "linear"
    assert machine["d_axis_inductance_h"] == machine["q_axis_inductance_h"] == 0.002
    # One extra step so motulator's end-point-free grid equals PEARSim's.
    assert request["n_eval"] == 4001
    assert request["t_stop_s"] == pytest.approx(0.04 + 1e-5)


def _without_driven_speed(document):
    document["machines"]["motor"].pop("driven_speed")
    document["machines"]["motor"]["inertia_kg_m2"] = 0.01
    return document


def _unbalanced(document):
    document["circuit"]["components"][1]["waveform"]["amplitude"] = "71 V"
    return document


def _negative_sequence(document):
    components = document["circuit"]["components"]
    components[1]["waveform"]["phase_deg"], components[2]["waveform"]["phase_deg"] = 320, 80
    return document


def _extra_component(document):
    document["circuit"]["components"].append(
        {"id": "rx", "type": "resistor", "p": "ta", "n": "gnd", "value": "1 kohm"}
    )
    return document


def _tied_star(document):
    document["circuit"]["components"].append(
        {"id": "rn", "type": "resistor", "p": "nstar", "n": "gnd", "value": "1 kohm"}
    )
    return document


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_without_driven_speed, "driven_speed"),
        (_unbalanced, "amplitudes differ"),
        (_negative_sequence, "120 degrees"),
        (_extra_component, "outside the comparable subset"),
        (_tied_star, "outside the comparable subset|isolated"),
    ],
)
def test_request_refuses_outside_the_subset(tmp_path, mutate, message):
    model = _load(tmp_path, mutate(_document()))
    with pytest.raises(MotulatorOracleError, match=message):
        emit_request(model)


def test_comparison_refuses_a_different_grid(tmp_path):
    model = _load(tmp_path, _document())
    result = HybridSimulator(model).simulate()
    time = tuple(float(t) * 1.0001 for t in result.time)
    zeros = tuple(0.0 for _ in time)
    fake = MotulatorSimulation("python", {}, {}, time, (zeros, zeros, zeros), zeros)
    with pytest.raises(MotulatorOracleError, match="same time grid"):
        compare_with_result(result, fake)


def test_worker_interpolates_the_flux_map_exactly_as_pearsim(tmp_path):
    spec = load_model(FLUX_MAP_EXAMPLE).machines[0]
    flux_map = spec.flux_map
    rng = np.random.default_rng(7)
    for i_d, i_q in rng.uniform(-40.0, 40.0, size=(200, 2)):
        expected = flux_map.flux(float(i_d), float(i_q))
        assert _bilinear(flux_map.id_axis, flux_map.iq_axis, flux_map.psi_d, i_d, i_q) == expected[0]
        assert _bilinear(flux_map.id_axis, flux_map.iq_axis, flux_map.psi_q, i_d, i_q) == expected[1]


def test_worker_current_map_inverts_the_flux_map():
    spec = load_model(FLUX_MAP_EXAMPLE).machines[0]
    flux_map = spec.flux_map
    current = _current_map(
        {
            "id_axis": list(flux_map.id_axis),
            "iq_axis": list(flux_map.iq_axis),
            "psi_d": [list(row) for row in flux_map.psi_d],
            "psi_q": [list(row) for row in flux_map.psi_q],
        }
    )
    for i_d, i_q in [(0.0, 0.0), (-15.0, 12.0), (8.0, -30.0), (-35.0, 35.0)]:
        psi_d, psi_q = flux_map.flux(i_d, i_q)
        recovered = current(complex(psi_d, psi_q))
        assert recovered == pytest.approx(complex(i_d, i_q), abs=1e-9)


# ---------------------------------------------------------------- engines


def _salient(document):
    document["machines"]["motor"].update(d_axis_inductance_h=0.0015, q_axis_inductance_h=0.003)
    return document


def _ramp(document):
    document["machines"]["motor"]["driven_speed"] = {
        "initial_rad_s": 150.0,
        "final_rad_s": 165.0,
        "ramp_start_s": 0.005,
        "ramp_stop_s": 0.035,
    }
    return document


def _flux_map(lead_deg: float):
    def mutate(document):
        machine = document["machines"]["motor"]
        machine.pop("stator_inductance_h")
        machine.pop("flux_linkage_wb")
        machine["flux_map"] = next(
            iter(yaml.safe_load(FLUX_MAP_EXAMPLE.read_text())["machines"].values())
        )["flux_map"]
        machine["driven_speed"] = {"initial_rad_s": 100.0}
        # 100 rad/s * 4 pole pairs = 400 rad/s electrical, back-EMF 40 V.
        document["parameters"]["amplitude"] = "40 V"
        document["parameters"]["frequency"] = f"{400.0 / math.tau!r} Hz"
        for index, component in enumerate(document["circuit"]["components"]):
            component["waveform"]["phase_deg"] = 180.0 + lead_deg - 120.0 * index
        return document

    return mutate


def _worst(tmp_path, document, step: str) -> float:
    document = copy.deepcopy(document)
    document["analysis"]["transient"]["output_step"] = step
    model = _load(tmp_path, document, name=f"m_{step.replace(' ', '')}.yaml")
    result = HybridSimulator(model).simulate()
    record = compare_with_result(result, simulate_model(model, python=PYTHON))
    assert record["versions"]["motulator"]
    return record["worst_relative_to_peak"]


@needs_motulator
def test_non_salient_constant_speed_agrees_to_round_off(tmp_path):
    assert _worst(tmp_path, _document(), "10 us") <= 1e-9


@needs_motulator
@pytest.mark.parametrize(
    "mutations",
    [
        pytest.param((_salient,), id="salient"),
        pytest.param((_ramp,), id="speed-ramp"),
        pytest.param((_salient, _ramp), id="salient-and-ramp"),
        pytest.param((_flux_map(6.0),), id="flux-map-unsaturated"),
    ],
)
def test_held_segment_error_is_first_order(tmp_path, mutations):
    document = _document()
    for mutate in mutations:
        document = mutate(document)
    coarse = _worst(tmp_path, document, "10 us")
    fine = _worst(tmp_path, document, "5 us")
    assert coarse <= 0.02
    assert 0.45 <= fine / coarse <= 0.55, (coarse, fine)


@needs_motulator
@pytest.mark.xfail(strict=True, raises=(AssertionError, ModelError), reason=(
    "KNOWN DEFECT: a saturating flux map diverges in saturation. The kernel "
    "puts the DIFFERENTIAL inductances in the winding matrix and carries "
    "L_old*i -> L_new*i across segment boundaries as if it were the flux "
    "(core/solver.py, _MachineDriver.begin_segment); in saturation L*i is not "
    "psi(i), the transformed current grows as L falls, and the run runs away."
))
def test_flux_map_in_saturation(tmp_path):
    """Exact solution (motulator): peak 15.7 A, inside the +-40 A map.
    PEARSim: i_d -> -67 A and a non-positive-inductance refusal, at 10, 5 and
    2.5 us alike — it does not shrink with the step, so it is a modelling
    defect, not splitting error. When the kernel is fixed this test XPASSes
    and the strict marker fails the suite until the xfail is removed."""

    document = _flux_map(10.0)(_document())
    assert _worst(tmp_path, document, "10 us") <= 0.02
