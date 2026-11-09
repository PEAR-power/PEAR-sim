# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Losses/thermal: synthetic-exact LUT tests, analytic goldens, end to end.

VAL-007 flavour: conduction and switching-energy lookups against closed
forms on runs where the device current is exactly known. VAL-008 flavour:
Foster step response against the analytic sum-of-exponentials (the PWC
recursion is exact at grid points). Rainflow against the canonical
ASTM E1049 example. Chain energy balance audited per device.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from pear_sim.analysis.losses_thermal import (
    LossThermalError,
    _fan_heatsink_operating_point,
    _foster_response,
    build_losses_thermal_observation,
    rainflow_count,
)
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.device_cards import FanCard, FosterStage, HeatsinkCard, CardProvenance
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "buck_with_loss_cards.yaml"


# ---------------------------------------------------------------------------
# Loader / spec validation
# ---------------------------------------------------------------------------


def _mutated(tmp_path: Path, mutate) -> Path:
    document = yaml.safe_load(EXAMPLE.read_text())
    mutate(document)
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_loader_accepts_the_example(tmp_path: Path) -> None:
    model = load_model(EXAMPLE)
    assert model.loss_analysis is not None
    assert {b.component_id for b in model.loss_analysis.devices} == {"q1", "d1"}


def test_provenance_is_required_per_table(tmp_path: Path) -> None:
    def mutate(document):
        del document["device_cards"]["igbt_syn600"]["conduction"]["provenance"]

    with pytest.raises(ModelError, match="provenance"):
        load_model(_mutated(tmp_path, mutate))


def test_tj_evaluation_must_sit_inside_every_validity_envelope(tmp_path: Path) -> None:
    def mutate(document):
        document["loss_analysis"]["tj_evaluation_c"] = 200

    with pytest.raises(ModelError, match="validity envelope"):
        load_model(_mutated(tmp_path, mutate))


def test_device_current_must_be_recorded(tmp_path: Path) -> None:
    def mutate(document):
        document["outputs"] = [o for o in document["outputs"] if o != "i(q1)"]

    with pytest.raises(ModelError, match=r"i\(q1\)"):
        load_model(_mutated(tmp_path, mutate))


def test_fan_requires_a_heatsink(tmp_path: Path) -> None:
    def mutate(document):
        document["loss_analysis"]["devices"]["q1"]["cooling"] = {"fan": "fan_axial"}

    with pytest.raises(ModelError, match="fan requires a heatsink"):
        load_model(_mutated(tmp_path, mutate))


def test_switch_card_requires_commutation_voltage(tmp_path: Path) -> None:
    def mutate(document):
        del document["loss_analysis"]["devices"]["q1"]["commutation_voltage"]

    with pytest.raises(ModelError, match="commutation_voltage"):
        load_model(_mutated(tmp_path, mutate))


# ---------------------------------------------------------------------------
# Analytic goldens
# ---------------------------------------------------------------------------


def test_foster_step_response_matches_the_analytic_sum(tmp_path: Path) -> None:
    stages = (
        FosterStage(resistance_k_per_w=0.3, time_constant_s=0.01),
        FosterStage(resistance_k_per_w=0.7, time_constant_s=0.15),
    )
    step = 1e-4
    count = 4001
    power = np.full(count, 25.0)
    reference_c = 40.0
    temperatures, stored = _foster_response(power, step, stages, reference_c)
    t = np.arange(count) * step
    analytic = reference_c + 25.0 * sum(
        stage.resistance_k_per_w * (1.0 - np.exp(-t / stage.time_constant_s))
        for stage in stages
    )
    # The PWC recursion is exact for constant power at grid points.
    assert np.max(np.abs(temperatures - analytic)) < 1e-9
    # Stored energy = sum C_k * dT_k with C = tau/R.
    expected_stored = sum(
        (stage.time_constant_s / stage.resistance_k_per_w)
        * 25.0
        * stage.resistance_k_per_w
        * (1.0 - math.exp(-(t[-1]) / stage.time_constant_s))
        for stage in stages
    )
    assert stored == pytest.approx(expected_stored, rel=1e-6)


def test_fan_heatsink_operating_point_matches_hand_calculation() -> None:
    provenance = CardProvenance(source="test")
    fan = FanCard(
        id="fan",
        flow_axis_m3_per_h=(0.0, 60.0),
        pressure_pa=(90.0, 0.0),
        provenance=provenance,
    )
    heatsink = HeatsinkCard(
        id="hs",
        airflow_axis_m3_per_h=(0.0, 60.0),
        rth_k_per_w=(2.0, 0.5),
        pressure_drop_pa=(0.0, 90.0),
        provenance=provenance,
    )
    airflow, rth = _fan_heatsink_operating_point(fan, heatsink)
    # Symmetric straight lines cross exactly in the middle.
    assert airflow == pytest.approx(30.0, abs=1e-12)
    assert rth == pytest.approx(1.25, abs=1e-12)


def test_fan_that_never_meets_the_heatsink_fails_closed() -> None:
    provenance = CardProvenance(source="test")
    fan = FanCard(
        id="weak",
        flow_axis_m3_per_h=(0.0, 60.0),
        pressure_pa=(5.0, 0.0),
        provenance=provenance,
    )
    heatsink = HeatsinkCard(
        id="hs",
        airflow_axis_m3_per_h=(0.0, 60.0),
        rth_k_per_w=(2.0, 0.5),
        pressure_drop_pa=(10.0, 300.0),
        provenance=provenance,
    )
    with pytest.raises(LossThermalError, match="pressure deficit"):
        _fan_heatsink_operating_point(fan, heatsink)


def test_rainflow_matches_the_astm_e1049_example() -> None:
    # ASTM E1049-85, section 5.4.4 example history.
    series = np.array([-2.0, 1.0, -3.0, 5.0, -1.0, 3.0, -4.0, 4.0, -2.0])
    cycles = rainflow_count(series)
    counted: dict[float, float] = {}
    for cycle in cycles:
        counted[cycle["range"]] = counted.get(cycle["range"], 0.0) + cycle["count"]
    assert counted == {3.0: 0.5, 4.0: 1.5, 6.0: 0.5, 8.0: 1.0, 9.0: 0.5}


def test_conduction_matches_the_closed_form_for_constant_current(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    path.write_text(
        """
schema: pear-sim/v0.1
name: diode_constant_current

circuit:
  ground: gnd
  components:
    - {id: isrc, type: current_source, p: gnd, n: mid, value: 20 A}
    - {id: d1, type: diode, p: mid, n: out, initial: on}
    - {id: rload, type: resistor, p: out, n: gnd, value: 1 ohm}
    - {id: cpar, type: capacitor, p: out, n: gnd, value: 1 uF, initial_voltage: 20 V}

device_cards:
  diode_card:
    kind: diode
    provenance: {source: "synthetic"}
    validity: {tj_min_c: 25, tj_max_c: 150, current_max_a: 60}
    conduction:
      current_axis_a: [0, 10, 30]
      temperature_axis_c: [25, 125]
      voltage_table_v:
        - [0.5, 1.0, 1.8]
        - [0.4, 1.2, 2.0]
      provenance: {source: "synthetic", figure: "F1"}
    thermal:
      foster_stages:
        - {resistance_k_per_w: 0.5, time_constant_s: 0.01}
      provenance: {source: "synthetic", figure: "F2"}

loss_analysis:
  ambient_c: 40
  tj_evaluation_c: 75
  devices:
    d1: {card: diode_card}

analysis:
  transient: {start: 0 s, stop: 10 ms, output_step: 10 us, solver: dopri5, rtol: 1e-9, atol: 1e-11}

outputs: [i(d1), v(out)]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result = HybridSimulator(load_model(path)).simulate()
    observation = build_losses_thermal_observation(result)
    (device,) = observation["devices"]
    # v_on(20 A, 75 C): bilinear between rows [1.0, 1.8] @25C and [1.2, 2.0]
    # @125C at i=20 (midpoint of 10..30) and Tj=75 (midpoint of 25..125).
    v_25 = 1.0 + (1.8 - 1.0) * 0.5
    v_125 = 1.2 + (2.0 - 1.2) * 0.5
    v_on = 0.5 * (v_25 + v_125)
    expected_power = v_on * 20.0
    assert device["conduction"]["mean_power_w"] == pytest.approx(
        expected_power, rel=1e-9
    )
    assert device["conduction"]["extrapolated_lookups"] == 0
    # Constant power: the case is held at its steady rise for the mean power, so
    # the junction-to-case stages start there too and Tj sits at case + R_jc * P
    # for the whole run. Started cold, the 10 ms stage would still be 37 % short
    # at the end of this 10 ms run and the margin would look larger than it is.
    thermal = device["thermal"]
    assert thermal["balance_within_tolerance"]
    assert thermal["initial_state"] == "steady_state_at_run_mean_power"
    expected_steady = thermal["case_c_static"] + 0.5 * expected_power
    assert thermal["tj_final_c"] == pytest.approx(expected_steady, rel=1e-6)
    assert thermal["tj_min_c"] == pytest.approx(expected_steady, rel=1e-6)


def test_switching_energy_scales_with_the_commutation_voltage() -> None:
    result = HybridSimulator(load_model(EXAMPLE)).simulate()
    observation = build_losses_thermal_observation(result)
    devices = {d["component_id"]: d for d in observation["devices"]}
    q1 = devices["q1"]
    counted = q1["switching"]["events_counted"]
    assert counted["turn_on"] == counted["turn_off"] == 400  # 20 kHz * 20 ms
    # CCM: current at every event sits near 10 A (small ripple); the table
    # value at 10 A is 0.8/1.1 mJ, scaled by V/V_test = 100/300.
    scale = 100.0 / 300.0
    expected_on = 400 * 0.8e-3 * scale
    expected_off = 400 * 1.1e-3 * scale
    assert q1["switching"]["totals_j"]["e_on_j"] == pytest.approx(expected_on, rel=0.1)
    assert q1["switching"]["totals_j"]["e_off_j"] == pytest.approx(expected_off, rel=0.1)
    d1 = devices["d1"]
    assert d1["switching"]["events_counted"]["reverse_recovery"] == 400
    assert d1["switching"]["totals_j"]["e_rec_j"] == pytest.approx(
        400 * 0.4e-3 * scale, rel=0.1
    )


def test_extrapolated_lookups_are_counted_and_warned(tmp_path: Path) -> None:
    def mutate(document):
        # Shrink the IGBT current axis so the ~10 A current extrapolates.
        card = document["device_cards"]["igbt_syn600"]
        card["conduction"]["current_axis_a"] = [0, 2, 4]
        card["conduction"]["voltage_table_v"] = [[0.8, 1.0, 1.1], [0.7, 1.05, 1.2]]
        card["switching"]["current_axis_a"] = [0, 2, 4]
        card["switching"]["e_on_j"] = [0.0, 0.2e-3, 0.4e-3]
        card["switching"]["e_off_j"] = [0.0, 0.3e-3, 0.5e-3]
        document["analysis"]["transient"]["stop"] = "2 ms"

    result = HybridSimulator(load_model(_mutated(tmp_path, mutate))).simulate()
    observation = build_losses_thermal_observation(result)
    q1 = next(d for d in observation["devices"] if d["component_id"] == "q1")
    assert q1["conduction"]["extrapolated_lookups"] > 0
    assert q1["switching"]["extrapolated_lookups"]["e_on_j"] > 0


# ---------------------------------------------------------------------------
# End to end bundle
# ---------------------------------------------------------------------------


def test_bundle_contains_the_losses_thermal_artifact(tmp_path: Path) -> None:
    from pear_sim.run.io import write_result

    def mutate(document):
        document["analysis"]["transient"]["stop"] = "5 ms"

    result = HybridSimulator(load_model(_mutated(tmp_path, mutate))).simulate()
    destination = write_result(result, tmp_path / "bundle")
    payload = json.loads((destination / "losses-thermal.json").read_text())
    assert payload["schema"] == "pear-sim/losses-thermal/v0.1"
    assert "post-processing estimate" in payload["claim"]
    assert len(payload["devices"]) == 2
    for device in payload["devices"]:
        assert device["thermal"]["balance_within_tolerance"]
        assert device["conduction"]["provenance"]["source"]
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["losses_thermal"]["artifact"] == "losses-thermal.json"


def test_per_device_mean_powers_and_tj_envelope_agree_with_the_totals(tmp_path: Path) -> None:
    """Consumers (report, MCP) draw these fields and must never divide or sum
    themselves, so the producer publishes them and they must agree with the
    energies and extremes it already published."""

    from pear_sim.run.io import write_result

    def mutate(document):
        document["analysis"]["transient"]["stop"] = "5 ms"

    result = HybridSimulator(load_model(_mutated(tmp_path, mutate))).simulate()
    payload = json.loads(
        (write_result(result, tmp_path / "bundle") / "losses-thermal.json").read_text()
    )
    duration = float(result.time[-1] - result.time[0])
    assert sum(device["mean_power_w"] for device in payload["devices"]) == pytest.approx(
        payload["total_mean_power_w"], rel=1e-9
    )
    for device in payload["devices"]:
        switching = device["switching"]
        assert switching["mean_power_w"] * duration == pytest.approx(
            sum(switching["totals_j"].values()), rel=1e-12, abs=1e-15
        )
        assert device["mean_power_w"] == pytest.approx(
            device["conduction"]["mean_power_w"] + switching["mean_power_w"], rel=1e-6
        )
        envelope = device["tj_envelope"]
        assert 1 < len(envelope["time_s"]) <= 200
        assert min(envelope["min_c"]) == device["thermal"]["tj_min_c"]
        assert max(envelope["max_c"]) == device["thermal"]["tj_max_c"]


def test_foster_response_can_start_from_a_steady_state() -> None:
    stages = (
        FosterStage(resistance_k_per_w=0.3, time_constant_s=0.01),
        FosterStage(resistance_k_per_w=0.7, time_constant_s=0.15),
    )
    power = np.full(101, 25.0)
    cold, _ = _foster_response(power, 1e-4, stages, 40.0)
    steady, stored = _foster_response(power, 1e-4, stages, 40.0, initial_power_w=25.0)
    assert cold[0] == pytest.approx(40.0)
    assert np.max(np.abs(steady - (40.0 + 25.0 * 1.0))) < 1e-9
    assert stored == pytest.approx(sum(s.time_constant_s * 25.0 for s in stages), rel=1e-12)
