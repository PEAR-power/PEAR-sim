# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Periodic steady state (shooting) + averaged loop-gain margins."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from jsonschema import Draft202012Validator

from pear_sim.analysis.small_signal import (
    SamplingModel,
    SmallSignalError,
    averaged_small_signal,
    frequency_response,
    loop_gain,
)
from pear_sim.analysis.steady_state import solve_periodic_steady_state
from pear_sim.evidence.stability import StabilityRequestError, run_stability_analysis
from pear_sim.model import load_model

ROOT = Path(__file__).parents[1]

# Documented tolerances: shooting must land on the independent closed-form
# periodic points; the averaged transfer function must match the textbook
# ideal-buck expression essentially exactly (same math, independent path).
SHOOTING_VS_ORACLE_A = 1e-6
SHOOTING_VS_ORACLE_V = 1e-5
AVERAGED_VS_CLOSED_FORM_REL = 1e-9


def test_buck_ccm_shooting_matches_the_independent_oracle(tmp_path: Path) -> None:
    from pear_sim.benchmarks.buck_boundary import (
        extract_buck_boundary_parameters,
        find_bcm_load,
        solve_periodic_buck,
    )

    # Cold start: zero initial conditions, so the Newton iteration has to work.
    source = (ROOT / "examples" / "buck.yaml").read_text(encoding="utf-8")
    cold = source.replace("initial_current: 3.39970055 A", "initial_current: 0 A")
    cold = cold.replace("initial_voltage: 39.98969319 V", "initial_voltage: 0 V")
    path = tmp_path / "buck_cold.yaml"
    path.write_text(cold, encoding="utf-8")
    model = load_model(path)

    steady = solve_periodic_steady_state(model)
    assert steady.converged
    assert steady.iterations > 1, "cold start must actually iterate"
    assert steady.natural_events_per_cycle == 0
    assert steady.distinct_topologies == 2

    parameters = extract_buck_boundary_parameters(model)
    point = solve_periodic_buck(
        parameters, 10.0, bcm_load_ohm=find_bcm_load(parameters)
    )
    state = steady.state_by_name()
    assert abs(state["l1"] - point.start_state.inductor_current_a) < SHOOTING_VS_ORACLE_A
    assert abs(state["c1"] - point.start_state.output_voltage_v) < SHOOTING_VS_ORACLE_V


def test_boost_ccm_shooting_matches_the_independent_oracle() -> None:
    from pear_sim.benchmarks.boost_boundary import (
        extract_boost_boundary_parameters,
        find_boost_bcm_load,
        solve_periodic_boost,
    )

    model = load_model(ROOT / "examples" / "boost.yaml")
    steady = solve_periodic_steady_state(model)
    assert steady.converged
    parameters = extract_boost_boundary_parameters(model)
    point = solve_periodic_boost(
        parameters, 24.0, bcm_load_ohm=find_boost_bcm_load(parameters)
    )
    state = steady.state_by_name()
    assert abs(state["l1"] - point.start_state.inductor_current_a) < SHOOTING_VS_ORACLE_A
    assert abs(state["c1"] - point.start_state.output_voltage_v) < SHOOTING_VS_ORACLE_V


def test_buck_dcm_shooting_handles_natural_commutation() -> None:
    from pear_sim.benchmarks.buck_boundary import (
        extract_buck_boundary_parameters,
        find_bcm_load,
        solve_periodic_buck,
    )

    model = load_model(ROOT / "examples" / "buck_dcm.yaml")
    steady = solve_periodic_steady_state(model)
    assert steady.converged
    assert steady.natural_events_per_cycle == 1, "the DCM cycle has one diode turn-off"
    assert steady.distinct_topologies == 3

    parameters = extract_buck_boundary_parameters(model)
    point = solve_periodic_buck(
        parameters, 1000.0, bcm_load_ohm=find_bcm_load(parameters)
    )
    assert point.regime == "DCM"
    state = steady.state_by_name()
    assert abs(state["l1"] - point.start_state.inductor_current_a) < SHOOTING_VS_ORACLE_A
    assert abs(state["c1"] - point.start_state.output_voltage_v) < SHOOTING_VS_ORACLE_V


def test_shooting_solves_a_series_inductor_star_on_the_invariant_subspace() -> None:
    """Regression: the LLC's Lr+Lm node is a kernel star invariant with a
    structural unit eigenvalue. Before invariant-aware shooting the FD probes violated the star
    KCL and the solve refused to run at all; now probes and Newton steps stay
    inside the invariant kernel, the level is pinned to the DECLARED initial
    state (never projected), and the pinning is recorded in the result."""

    steady = solve_periodic_steady_state(
        load_model(ROOT / "examples" / "llc_resonant.yaml")
    )
    assert steady.converged
    assert steady.iterations <= 10  # measured 5 from the cold declared state
    (invariant,) = steady.invariants
    assert invariant["node"] == "pri"
    assert invariant["terms"] == {"lres": 1.0, "lprim": -1.0}
    assert invariant["pinned_value_a"] == pytest.approx(0.0, abs=1e-12)
    assert invariant["source"] == "declared initial state"
    # The fixed point actually sits on the pinned level set.
    state = steady.state_by_name()
    assert state["lres"] - state["lprim"] == pytest.approx(0.0, abs=1e-9)
    # Above resonance the rectifier commutates naturally inside the cycle.
    assert steady.natural_events_per_cycle >= 2
    # The record is additive and conditional: the artifact carries the
    # invariants...
    assert steady.as_dict()["invariants"] == [dict(invariant)]
    # ...while a model WITHOUT star nodes keeps its invariant-free artifact shape
    # byte-for-byte (no key at all, not an empty list).
    plain = solve_periodic_steady_state(load_model(ROOT / "examples" / "buck.yaml"))
    assert plain.invariants == ()
    assert "invariants" not in plain.as_dict()


def test_averaged_transfer_function_matches_the_closed_form() -> None:
    model = load_model(ROOT / "examples" / "buck.yaml")
    steady = solve_periodic_steady_state(model)
    small_signal = averaged_small_signal(model, steady, "v(out)")
    assert small_signal.duty == pytest.approx(0.4, abs=1e-12)

    frequencies = np.logspace(1.0, 4.0, 61)
    response = frequency_response(small_signal, frequencies)
    vin, inductance, capacitance, load = 100.0, 1e-3, 100e-6, 10.0
    s = 2j * np.pi * frequencies
    closed_form = vin / (inductance * capacitance * s**2 + (inductance / load) * s + 1.0)
    relative = np.abs(response - closed_form) / np.abs(closed_form)
    assert np.max(relative) < AVERAGED_VS_CLOSED_FORM_REL


def test_loop_gain_margins_match_an_independent_computation() -> None:
    from scipy.optimize import brentq

    model = load_model(ROOT / "examples" / "buck.yaml")
    steady = solve_periodic_steady_state(model)
    small_signal = averaged_small_signal(model, steady, "v(out)")
    sensor = 0.025
    result = loop_gain(small_signal, sensor_gain=sensor, points=400)

    vin, inductance, capacitance, load = 100.0, 1e-3, 100e-6, 10.0

    def loop_mag_db(frequency: float) -> float:
        s = 2j * np.pi * frequency
        gain = vin / (inductance * capacitance * s**2 + (inductance / load) * s + 1.0)
        return 20.0 * np.log10(abs(gain * sensor))

    crossover = brentq(loop_mag_db, 10.0, 9e3, xtol=1e-6)
    s = 2j * np.pi * crossover
    phase = np.degrees(
        np.angle(vin / (inductance * capacitance * s**2 + (inductance / load) * s + 1.0))
    )
    expected_pm = 180.0 + phase
    assert result.gain_crossover_hz == pytest.approx(crossover, rel=1e-3)
    assert result.phase_margin_deg == pytest.approx(expected_pm, abs=0.1)
    # A second-order plant never reaches -180 deg: fail-closed None, not a number.
    assert result.gain_margin_db is None


def test_zoh_sampling_folds_the_analytic_factor_into_the_loop() -> None:
    """Sampling model: L(s) with sampling == continuous L(s) * (1-e^(-sT))/(sT) * e^(-s*tau)."""

    model = load_model(ROOT / "examples" / "buck.yaml")
    steady = solve_periodic_steady_state(model)
    small_signal = averaged_small_signal(model, steady, "v(out)")
    sampling = SamplingModel(
        update_frequency_hz=small_signal.switching_frequency_hz,
        computation_delay_s=2e-6,
    )
    continuous = loop_gain(small_signal, sensor_gain=0.025, points=300)
    sampled = loop_gain(small_signal, sensor_gain=0.025, points=300, sampling=sampling)

    frequencies = np.asarray(continuous.frequencies_hz)
    s = 2j * np.pi * frequencies
    period = 1.0 / sampling.update_frequency_hz
    factor = (1.0 - np.exp(-s * period)) / (s * period) * np.exp(
        -s * sampling.computation_delay_s
    )
    expected_db = np.asarray(continuous.loop_magnitude_db) + 20.0 * np.log10(
        np.abs(factor)
    )
    expected_phase = np.asarray(continuous.loop_phase_deg) + np.degrees(
        np.unwrap(np.angle(factor))
    )
    assert np.max(np.abs(np.asarray(sampled.loop_magnitude_db) - expected_db)) < 1e-9
    assert np.max(np.abs(np.asarray(sampled.loop_phase_deg) - expected_phase)) < 1e-6
    # The plant Bode is untouched — sampling lives in the loop, not in G_yd.
    assert sampled.plant_magnitude_db == continuous.plant_magnitude_db
    assert sampled.plant_phase_deg == continuous.plant_phase_deg


def test_sampling_declaration_fails_closed(tmp_path: Path) -> None:
    model = load_model(ROOT / "examples" / "buck.yaml")
    steady = solve_periodic_steady_state(model)
    small_signal = averaged_small_signal(model, steady, "v(out)")

    with pytest.raises(SmallSignalError, match="update_frequency_hz"):
        loop_gain(small_signal, sampling=SamplingModel(update_frequency_hz=0.0))
    with pytest.raises(SmallSignalError, match="computation_delay_s"):
        loop_gain(
            small_signal,
            sampling=SamplingModel(
                update_frequency_hz=20_000.0, computation_delay_s=-1e-6
            ),
        )
    # The grid must stay below the update frequency: a ZOH null inside the
    # sweep would put -inf dB into the retained artifact.
    with pytest.raises(SmallSignalError, match="below the sampling update"):
        loop_gain(
            small_signal,
            f_max_hz=6_000.0,
            sampling=SamplingModel(update_frequency_hz=5_000.0),
        )
    with pytest.raises(StabilityRequestError, match="sampling"):
        run_stability_analysis(
            ROOT / "examples" / "buck.yaml",
            tmp_path / "bad",
            request={"output": "v(out)", "sampling": {"surprise": 1}},
        )
    with pytest.raises(StabilityRequestError, match="sampling"):
        run_stability_analysis(
            ROOT / "examples" / "buck.yaml",
            tmp_path / "bad2",
            request={"output": "v(out)", "sampling": 5},
        )


def test_loop_gain_json_records_the_sampling_declaration(tmp_path: Path) -> None:
    """Sampling model: declared model recorded; omission explicit, never silent."""

    declared = run_stability_analysis(
        ROOT / "examples" / "buck.yaml",
        tmp_path / "declared",
        request={
            "output": "v(out)",
            "sensor_gain": 0.025,
            "sampling": {"computation_delay_s": 1e-6},
        },
    )
    assert declared.overall_status == "pass"
    assert declared.loop_gain_json is not None
    body = json.loads(declared.loop_gain_json.read_text(encoding="utf-8"))
    assert body["schema"] == "pear-sim/loop-gain/v0.2"
    # update_frequency_hz defaults to f_sw (the sampled_pi_pwm contract).
    assert body["sampling"]["kind"] == "zoh_with_computation_delay"
    assert body["sampling"]["update_frequency_hz"] == pytest.approx(20_000.0)
    assert body["sampling"]["computation_delay_s"] == pytest.approx(1e-6)
    evidence = json.loads(declared.evidence_json.read_text(encoding="utf-8"))
    assert evidence["request"]["sampling"] == {"computation_delay_s": 1e-6}
    schema = json.loads(
        (ROOT / "schema" / "stability-evidence.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator(schema).validate(evidence)

    continuous = run_stability_analysis(
        ROOT / "examples" / "buck.yaml",
        tmp_path / "continuous",
        request={"output": "v(out)", "sensor_gain": 0.025},
    )
    assert continuous.loop_gain_json is not None
    body = json.loads(continuous.loop_gain_json.read_text(encoding="utf-8"))
    assert body["sampling"] == "not_modelled"


def test_stability_bundle_gates_pass_and_fail(tmp_path: Path) -> None:
    passing = run_stability_analysis(
        ROOT / "examples" / "buck.yaml",
        tmp_path / "pass",
        request={
            "output": "v(out)",
            "sensor_gain": 0.025,
            "margins": {"min_phase_margin_deg": 10.0},
        },
    )
    assert passing.overall_status == "pass"
    evidence = json.loads(passing.evidence_json.read_text(encoding="utf-8"))
    schema = json.loads(
        (ROOT / "schema" / "stability-evidence.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(evidence)
    assert [gate["id"] for gate in evidence["gates"]] == [
        "steady_state_converged",
        "ccm_single_pwm_leg",
        "phase_margin",
    ]

    failing = run_stability_analysis(
        ROOT / "examples" / "buck.yaml",
        tmp_path / "fail",
        request={
            "output": "v(out)",
            "sensor_gain": 0.025,
            "margins": {"min_phase_margin_deg": 80.0},
        },
    )
    assert failing.overall_status == "fail"

    # A plant whose phase only approaches -180 deg, swept over the whole
    # validity band: the gain margin is unbounded there, which is a pass.
    unbounded = run_stability_analysis(
        ROOT / "examples" / "buck.yaml",
        tmp_path / "gm",
        request={
            "output": "v(out)",
            "sensor_gain": 0.025,
            "margins": {"min_gain_margin_db": 6.0},
        },
    )
    gm_evidence = json.loads(unbounded.evidence_json.read_text(encoding="utf-8"))
    gm_gate = next(g for g in gm_evidence["gates"] if g["id"] == "gain_margin")
    assert gm_gate["status"] == "pass" and "unbounded" in gm_gate["detail"]

    # The same plant swept only up to 100 Hz cannot attest anything beyond
    # it: fail closed, and say how to fix the request.
    truncated = run_stability_analysis(
        ROOT / "examples" / "buck.yaml",
        tmp_path / "gm-short",
        request={
            "output": "v(out)",
            "sensor_gain": 0.025,
            "f_max_hz": 100.0,
            "margins": {"min_gain_margin_db": 6.0},
        },
    )
    assert truncated.overall_status == "fail"
    short_evidence = json.loads(truncated.evidence_json.read_text(encoding="utf-8"))
    short_gate = next(g for g in short_evidence["gates"] if g["id"] == "gain_margin")
    assert "cannot be attested" in short_gate["detail"] and "f_max_hz" in short_gate["detail"]


def test_dcm_model_fails_closed_for_loop_gain_but_not_steady_state(tmp_path: Path) -> None:
    artifacts = run_stability_analysis(
        ROOT / "examples" / "buck_dcm.yaml",
        tmp_path / "dcm",
        request={"output": "v(out)"},
    )
    assert artifacts.overall_status == "fail"
    evidence = json.loads(artifacts.evidence_json.read_text(encoding="utf-8"))
    by_id = {gate["id"]: gate for gate in evidence["gates"]}
    assert by_id["steady_state_converged"]["status"] == "pass"
    assert by_id["ccm_single_pwm_leg"]["status"] == "fail"
    assert "CCM only" in evidence["loop_gain_error"]

    steady_only = run_stability_analysis(
        ROOT / "examples" / "buck_dcm.yaml", tmp_path / "dcm_only"
    )
    assert steady_only.overall_status == "pass"
    assert steady_only.loop_gain_json is None


def test_request_validation_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(StabilityRequestError, match="unknown field"):
        run_stability_analysis(
            ROOT / "examples" / "buck.yaml", tmp_path / "bad", request={"surprise": 1}
        )
    with pytest.raises(StabilityRequestError, match="not among the model outputs"):
        run_stability_analysis(
            ROOT / "examples" / "buck.yaml",
            tmp_path / "bad2",
            request={"output": "v(nope)"},
        )
    schema = json.loads(
        (ROOT / "schema" / "stability-request.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)


def test_cli_analyze_stability_exit_codes(tmp_path: Path, capsys) -> None:
    from pear_sim.api.cli import main

    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "output": "v(out)",
                "sensor_gain": 0.025,
                "margins": {"min_phase_margin_deg": 10.0},
            }
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "analyze",
            "stability",
            str(ROOT / "examples" / "buck.yaml"),
            "--request",
            str(request),
            "--out",
            str(tmp_path / "ok"),
            "--json",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["overall_status"] == "pass"

    strict = tmp_path / "strict.json"
    strict.write_text(
        json.dumps(
            {
                "output": "v(out)",
                "sensor_gain": 0.025,
                "margins": {"min_phase_margin_deg": 80.0},
            }
        ),
        encoding="utf-8",
    )
    code = main(
        [
            "analyze",
            "stability",
            str(ROOT / "examples" / "buck.yaml"),
            "--request",
            str(strict),
            "--out",
            str(tmp_path / "strict-out"),
            "--json",
        ]
    )
    assert code == 1


def test_cycle_average_of_an_unrecorded_capacitor_is_exact() -> None:
    """The cycle average must not depend on which signals the model records.

    A capacitor voltage is parabolic inside a mode interval (its current ramps),
    so reconstructing it linearly between mode boundaries biases the average —
    by 0.08 % on a 200 kHz synchronous buck, enough to put the averaged-model
    operating point off the volt-second balance. Volt-second balance of the
    ideal buck fixes the exact answer: mean v(c1) = Vin * (D - t_dead * fsw).
    """

    source = (ROOT / "examples" / "synchronous_buck_complementary.yaml").read_text()
    model_text = (
        source.replace("vin: 100 V", "vin: 48 V")
        .replace("fsw: 20 kHz", "fsw: 200 kHz")
        .replace("duty: 40 %", "duty: 25 %")
        .replace("dead_time: 500 ns", "dead_time: 50 ns")
        .replace("load: 10 ohm", "load: 1.2 ohm")
        .replace("value: 1 mH, initial_current: 4 A", "value: 10 uH, initial_current: 10 A")
        .replace("initial_voltage: 40 V", "initial_voltage: 12 V")
    )
    # Record only v(out): c1's own voltage is NOT among the outputs.
    head, _ = model_text.split("outputs:")
    model_text = head + "outputs:\n  - v(out)\n  - i(l1)\n"
    path = ROOT / "tests" / "_unrecorded_cap.yaml"
    try:
        path.write_text(model_text, encoding="utf-8")
        steady = solve_periodic_steady_state(load_model(path))
    finally:
        path.unlink(missing_ok=True)
    averages = dict(zip(steady.state_names, steady.cycle_average_state))
    expected = 48.0 * (0.25 - 50e-9 * 200e3)
    assert averages["c1"] == pytest.approx(expected, rel=1e-6)
    assert averages["l1"] == pytest.approx(expected / 1.2, rel=1e-6)


def test_dead_time_cycle_averages_to_the_ideal_buck_transfer_function(tmp_path: Path) -> None:
    """A complementary leg with dead time visits four topologies per cycle.

    In CCM with positive inductor current the dead-time intervals are diode
    freewheeling — electrically the low-side state — so the duty-to-output
    transfer function of the ideal buck is unchanged by them:
    G_vd(s) = Vin / (L C s^2 + (L / R) s + 1).
    """

    model = load_model(ROOT / "examples" / "synchronous_buck_complementary.yaml")
    steady = solve_periodic_steady_state(model)
    assert steady.distinct_topologies == 3  # high-on, low-on, diode freewheel
    averaged = averaged_small_signal(model, steady, "v(out)")
    frequencies = np.array([10.0, 100.0, 300.0, 1000.0])
    response = frequency_response(averaged, frequencies)
    s = 2j * np.pi * frequencies
    expected = 100.0 / (1e-3 * 100e-6 * s**2 + (1e-3 / 10.0) * s + 1.0)
    assert np.max(np.abs(response - expected) / np.abs(expected)) < 1e-9
    # The reported duty is the high-side on fraction: commanded duty minus one dead time.
    assert averaged.duty == pytest.approx(0.4 - 500e-9 * 20e3, rel=1e-9)


def test_the_phase_margin_is_the_worst_over_every_gain_crossover() -> None:
    """A loop can cross 0 dB more than once.

    The canonical buck's LC resonance (503 Hz, ~+10 dB peak) lifts a plain
    gain K = 0.005 (loop at -6 dB at DC) above 0 dB and back down: two gain
    crossovers. The first, below the resonance, has ~150 deg of phase to
    spare; the second, above it, ~53 deg. Stability is decided by the worst
    one, so that is the margin to report, with every crossover listed.
    """

    model = load_model(ROOT / "examples" / "buck.yaml")
    small_signal = averaged_small_signal(model, solve_periodic_steady_state(model), "v(out)")
    result = loop_gain(small_signal, compensator_numerator=(0.005,), compensator_denominator=(1.0,))
    assert len(result.gain_crossovers_hz) == 2
    low, high = result.gain_crossovers_hz
    assert low < 503.0 < high
    assert result.gain_crossover_hz == pytest.approx(high)
    assert 45.0 < result.phase_margin_deg < 60.0
    phase_at_high = float(
        np.interp(np.log10(high), np.log10(result.frequencies_hz), result.loop_phase_deg)
    )
    assert result.phase_margin_deg == pytest.approx(180.0 + phase_at_high)
