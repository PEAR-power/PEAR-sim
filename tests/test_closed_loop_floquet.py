# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Closed-loop periodic steady state and Floquet multipliers.

The plant and both gain pairs are the ones the averaged loop-gain tests in
``test_closed_loop_control.py`` use: a 24 V -> 12 V buck at 50 kHz with a
5.45 ohm load (2.2 A, CCM), a PI written in signal blocks. kp=0.002/ki=30 is
the healthy design; kp=0.08/ki=150 produced a ~4 kHz limit cycle in the
switching kernel.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pear_sim.analysis import closed_loop
from pear_sim.analysis.closed_loop import (
    ClosedLoopSteadyStateError,
    solve_closed_loop_steady_state,
)
from pear_sim.core.solver import HybridSimulator
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]
PERIOD_S = 20e-6

BODY = """
schema: pear-sim/v0.1
name: buck_closed_loop_floquet
parameters: {{vin: 24 V, fsw: 50 kHz}}
circuit:
  ground: gnd
  components:
    - {{id: vin, type: voltage_source, p: in, n: gnd, value: "${{vin}}"}}
    - {{id: q1, type: ideal_switch, p: in, n: sw, control: ctrl, initial: on}}
    - {{id: d1, type: diode, p: gnd, n: sw, initial: off}}
    - {{id: l1, type: inductor, p: sw, n: out, value: 47 uH, initial_current: 0.92 A}}
    - {{id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, initial_voltage: 12 V}}
    - {{id: load, type: resistor, p: out, n: gnd, value: 5.45 ohm}}
signals:
  task: {{period: 20 us}}
  blocks:
    vout: {{type: probe, source: v(out), unit: V}}
    vref: {{type: constant, value: 12.0, unit: V}}
    err: {{type: sum, inputs: [vref, vout], signs: [1, -1], unit: V}}
    err_dt: {{type: gain, input: err, gain: 2.0e-5, unit: V}}
    int_prev: {{type: unit_delay, input: integral, initial: {integral0}, unit: V}}
    integral: {{type: sum, inputs: [int_prev, err_dt], unit: V}}
    kp_term: {{type: gain, input: err, gain: {kp}, unit: V}}
    ki_term: {{type: gain, input: integral, gain: {ki}, unit: V}}
    u: {{type: sum, inputs: [kp_term, ki_term], unit: V}}
    duty: {{type: saturate, input: u, min: 0.05, max: 0.95, unit: V}}
controls:
  ctrl: {{type: signal_pwm, frequency: "${{fsw}}", duty_signal: s(duty), duty_min: 5 %, duty_max: 95 %}}
analysis:
  transient: {{start: 0 s, stop: 1 ms, output_step: 2 us, solver: dopri5, rtol: 1.0e-9, atol: 1.0e-11}}
outputs: [v(out), i(l1)]
"""

HEALTHY = (0.002, 30.0)
HOT = (0.08, 150.0)


def _model(tmp_path: Path, gains: tuple[float, float], *, integral0: float | None = None):
    kp, ki = gains
    # Near the orbit: the integrator holds duty / ki at zero error.
    start = 0.5 / ki if integral0 is None else integral0
    path = tmp_path / f"loop_{kp}_{ki}.yaml"
    path.write_text(BODY.format(kp=kp, ki=ki, integral0=start), encoding="utf-8")
    return load_model(path)


@pytest.fixture(scope="module")
def solved(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("floquet")
    return {
        name: (model, solve_closed_loop_steady_state(model))
        for name, model in (
            ("healthy", _model(tmp_path, HEALTHY)),
            ("hot", _model(tmp_path, HOT)),
        )
    }


def test_orbit_regulates_and_is_differentiable(solved) -> None:
    for name in ("healthy", "hot"):
        _model_, result = solved[name]
        assert result.converged, name
        state = dict(zip(result.state_names, result.state, strict=True))
        # Zero average error: the integrator settles the output at 12 V.
        assert state["c1"] == pytest.approx(12.0, abs=0.05)
        assert all(item.held for item in result.assumptions), result.assumptions
        assert len(result.multipliers) == 3  # l1, c1, one register


def test_floquet_verdict_discriminates_the_two_designs(solved) -> None:
    """The gate must stay quiet on the healthy design and ring on the hot one,
    and the hot design must oscillate where the kernel's limit cycle did."""

    healthy = solved["healthy"][1]
    hot = solved["hot"][1]
    assert healthy.stability == "stable"
    assert healthy.max_multiplier_magnitude < 1.0
    assert hot.stability == "unstable"
    critical = hot.multipliers[0]
    assert critical.magnitude > 1.0
    assert critical.frequency_hz == pytest.approx(4_000.0, rel=0.15)


def test_floquet_agrees_with_the_discrete_aware_averaged_margins(solved) -> None:
    """A different method (state-space averaging + ZOH, a declared
    compensator) must agree on WHICH design is unstable and roughly where."""

    from pear_sim.analysis.small_signal import (
        SamplingModel,
        averaged_small_signal,
        loop_gain,
    )
    from pear_sim.analysis.steady_state import solve_periodic_steady_state
    from tests.test_closed_loop_control import OPEN_LOOP_PLANT_BODY

    path = Path(solved["healthy"][0].source_path).with_name("open_loop.yaml")
    path.write_text(OPEN_LOOP_PLANT_BODY, encoding="utf-8")
    plant = load_model(path)
    signal = averaged_small_signal(plant, solve_periodic_steady_state(plant), "v(out)")
    sampling = SamplingModel(update_frequency_hz=signal.switching_frequency_hz)
    for name, (kp, ki) in (("healthy", HEALTHY), ("hot", HOT)):
        margins = loop_gain(
            signal,
            compensator_numerator=(kp, ki),
            compensator_denominator=(1.0, 0.0),
            sensor_gain=1.0,
            f_min_hz=5.0,
            f_max_hz=5_000.0,
            points=600,
            sampling=sampling,
        )
        averaged_unstable = margins.phase_margin_deg < 0.0
        assert averaged_unstable == (solved[name][1].stability == "unstable"), name
    hot_crossover = loop_gain(
        signal,
        compensator_numerator=HOT,
        compensator_denominator=(1.0, 0.0),
        sensor_gain=1.0,
        f_min_hz=5.0,
        f_max_hz=5_000.0,
        points=600,
        sampling=sampling,
    ).gain_crossover_hz
    assert solved["hot"][1].multipliers[0].frequency_hz == pytest.approx(
        hot_crossover, rel=0.15
    )


def _trajectory_multipliers(model, result, cycles: int) -> np.ndarray:
    """Identify the monodromy from ONE long run started off the orbit.

    Independent of the probe arithmetic: no finite differences, no per-cycle
    restarts — the controller memory is carried by the kernel tick to tick,
    and the matrix is the least-squares fit of d_{k+1} = M d_k over the run.
    """

    plan = closed_loop._plan(model)
    orbit = np.asarray(result.state)
    rng = np.random.default_rng(1)
    offset = 1e-4 * np.maximum(np.abs(orbit), 1e-3) * rng.standard_normal(len(orbit))
    start = closed_loop._cycle_model(plan, orbit + offset)
    run_model = replace(
        start,
        transient=replace(start.transient, stop=cycles * PERIOD_S, output_step=PERIOD_S),
        outputs=tuple(dict.fromkeys((*start.outputs, "i(l1)", "v(c1)"))),
    )
    run = HybridSimulator(run_model).simulate()
    ticks = {entry["tick_index"]: entry["values"] for entry in run.signal_trace}
    rows = []
    for cycle in range(cycles):
        index = int(np.argmin(np.abs(run.time - cycle * PERIOD_S)))
        rows.append(
            [run.signals["i(l1)"][index], run.signals["v(c1)"][index], ticks[cycle]["int_prev"]]
        )
    deviations = np.asarray(rows) - orbit
    fitted = deviations[1:].T @ np.linalg.pinv(deviations[:-1].T)
    return np.sort_complex(np.linalg.eigvals(fitted))


@pytest.mark.parametrize("name", ["healthy", "hot"])
def test_multipliers_match_a_monodromy_identified_from_one_long_run(solved, name) -> None:
    """Measured 2026-11-15: max |difference| 1.2e-8 (healthy) and 3.5e-6 (hot,
    whose perturbation grows over the 60 cycles). Carrying the register's OLD
    value instead of its input — the mapping error this guards — gives 6.9e-3.
    The 1e-4 budget sits between the two by two orders of magnitude each way."""

    model, result = solved[name]
    expected = np.sort_complex(
        np.asarray([complex(item.real, item.imag) for item in result.multipliers])
    )
    identified = _trajectory_multipliers(model, result, cycles=60)
    assert np.max(np.abs(identified - expected)) < 1e-4


def test_a_limiter_engaged_during_newton_is_reported_not_raised(tmp_path: Path) -> None:
    """Starting the integrator at 0 clamps the duty at 5 %: the map is flat in
    the controller direction. That is a non-converged RESULT naming the
    limiter, not a 'singular matrix' crash or a fabricated orbit."""

    result = solve_closed_loop_steady_state(_model(tmp_path, HEALTHY, integral0=0.0))
    assert not result.converged
    assert result.stability == "not_assessed"
    assert not result.multipliers
    assert "limiter" in result.notes[0] and "duty" in result.notes[0]


def test_result_payload_states_its_claim(solved) -> None:
    payload = solved["hot"][1].as_dict()
    assert payload["claim"] == "local stability of the computed periodic orbit only"
    assert payload["stability"] == "unstable"
    assert set(payload["state"]) == {"l1", "c1", "signal:int_prev"}


@pytest.mark.parametrize(
    ("example", "fragment"),
    [
        # Built-in PI: its integrator lives in the kernel and cannot be set.
        ("buck_closed_loop.yaml", "not signal_pwm"),
        # Anti-windup with compare/select: a threshold is a kink.
        ("buck_closed_loop_antiwindup.yaml", "non-smooth"),
        # The shipped signal-block example has a load-step waveform source.
        ("buck_closed_loop_signals.yaml", "DC sources only"),
        # Open loop belongs to the ordinary shooting.
        ("buck.yaml", "signal layer"),
    ],
)
def test_out_of_scope_models_are_refused_by_name(example: str, fragment: str) -> None:
    with pytest.raises(ClosedLoopSteadyStateError, match=fragment):
        solve_closed_loop_steady_state(load_model(ROOT / "examples" / example))


# --- Closed-form oracle (tests/reference/buck_closed_loop_floquet_oracle.py) ---
# Derived on paper, solved by a different method (van Loan blocks + one
# bracketed root), importing nothing from PEARSim. The oracle itself is first
# checked against physics that does not go through its matrix algebra.

from tests.reference.buck_closed_loop_floquet_oracle import (  # noqa: E402
    BuckLoop,
    rlc_multipliers,
    solve_dcm,
)
from tests.reference.buck_closed_loop_floquet_oracle import (  # noqa: E402
    solve as oracle_solve,
)

_PLANT = dict(
    vin_v=24.0,
    inductance_h=47e-6,
    capacitance_f=100e-6,
    load_ohm=5.45,
    switching_frequency_hz=50e3,
    vref_v=12.0,
)


def test_oracle_plant_is_the_textbook_rlc_network() -> None:
    """Both topologies of the CCM buck share A; its period map must carry the
    series-L / parallel-RC natural frequencies, exp(lambda T)."""

    from scipy.linalg import expm

    loop = BuckLoop(**_PLANT, kp=0.0, ki=1.0)
    plant = np.sort_complex(np.linalg.eigvals(expm(loop.a * loop.period_s)))
    assert np.max(np.abs(plant - np.sort_complex(rlc_multipliers(loop)))) < 1e-12


def test_oracle_integrator_mode_converges_to_the_sampled_gain_formula() -> None:
    """Slow-integrator physics: with kp=0 and ki small against the LC
    resonance, the integrator decays at ki * g per second, g = dv_C(0)/dd the
    DC gain AS SAMPLED (the tick reads v_C at the cycle start, on the ripple —
    measured 2026-11-15: g = 24.085 V, not Vin = 24 V; using Vin leaves a
    0.36 % gap that does not shrink). The relative error must shrink in
    proportion to ki — an approximation converging, not a defect."""

    from scipy.linalg import expm

    from tests.reference.buck_closed_loop_floquet_oracle import _forced_response

    probe = BuckLoop(**_PLANT, kp=0.0, ki=1.0)
    period = probe.period_s
    inverse = np.linalg.inv(np.eye(2) - expm(probe.a * period))

    def sampled_output(duty: float) -> float:
        return float((inverse @ _forced_response(probe, duty * period))[1])

    gain = (sampled_output(0.5 + 1e-6) - sampled_output(0.5 - 1e-6)) / 2e-6
    errors = {}
    for ki in (5.0, 1.0):
        orbit = oracle_solve(BuckLoop(**_PLANT, kp=0.0, ki=ki))
        real = [value.real for value in orbit.multipliers if abs(value.imag) < 1e-12]
        rate = -np.log(max(real)) / period
        errors[ki] = abs(rate - ki * gain) / (ki * gain)
    assert errors[1.0] < 3e-4  # measured 2.0e-4
    assert 4.0 < errors[5.0] / errors[1.0] < 6.0  # first order in ki: 4.7


@pytest.mark.parametrize(("name", "gains"), [("healthy", HEALTHY), ("hot", HOT)])
def test_engine_matches_the_closed_form_oracle(solved, name, gains) -> None:
    """The truth gate: a different derivation, a different solution method.

    Measured 2026-11-15: multipliers 1.2e-11, monodromy entries 5.6e-8,
    valley current 1.9e-9. Carrying a register's OLD value instead of its
    input gives multipliers off by 1.4e-2 (healthy) / 2.4e-2 (hot)."""

    kp, ki = gains
    oracle = oracle_solve(BuckLoop(**_PLANT, kp=kp, ki=ki))
    result = solved[name][1]
    state = dict(zip(result.state_names, result.state, strict=True))
    assert state["l1"] == pytest.approx(oracle.inductor_current_a, abs=1e-7)
    assert state["c1"] == pytest.approx(oracle.capacitor_voltage_v, abs=1e-7)
    assert state["signal:int_prev"] == pytest.approx(oracle.register, abs=1e-9)
    engine = np.sort_complex(
        np.asarray([complex(item.real, item.imag) for item in result.multipliers])
    )
    assert np.max(np.abs(engine - np.sort_complex(oracle.multipliers))) < 1e-7
    assert np.max(np.abs(np.asarray(result.monodromy) - oracle.monodromy)) < 1e-6


# --- K3: analytic monodromy from the orbit's compiled modes ---------------


@pytest.mark.parametrize(("name", "gains"), [("healthy", HEALTHY), ("hot", HOT)])
def test_analytic_monodromy_matches_the_oracle_to_machine_precision(
    solved, name, gains
) -> None:
    """Measured 2026-11-15: multipliers 1.6e-15 / 4.8e-14, monodromy entries
    3.4e-11 / 1.7e-10 against the closed form; the finite-difference
    cross-check agrees to 1.2e-11."""

    kp, ki = gains
    result = solved[name][1]
    assert result.monodromy_method == "analytic_saltation"
    (agreement,) = [
        item for item in result.assumptions if item.name == "analytic_and_finite_difference_agree"
    ]
    assert agreement.held
    oracle = oracle_solve(BuckLoop(**_PLANT, kp=kp, ki=ki))
    engine = np.sort_complex(
        np.asarray([complex(item.real, item.imag) for item in result.multipliers])
    )
    assert np.max(np.abs(engine - np.sort_complex(oracle.multipliers))) < 1e-10
    assert np.max(np.abs(np.asarray(result.monodromy) - oracle.monodromy)) < 1e-8


def test_a_missing_saltation_term_is_caught_at_runtime(tmp_path: Path, monkeypatch) -> None:
    """Mutation: drop the controlled-edge term (the controller then does not
    act on the plant in the linearisation). The built-in cross-check against
    finite differences must withhold the verdict rather than publish it.
    Measured: the multipliers move by 1.4e-2 (healthy) / 2.1e-1 (hot)."""

    monkeypatch.setattr(
        closed_loop,
        "_saltation_term",
        lambda after_edge, jump, period, duty_row: np.zeros((len(jump), len(duty_row))),
    )
    for gains in (HEALTHY, HOT):
        result = solve_closed_loop_steady_state(_model(tmp_path, gains))
        assert result.stability == "not_assessed"
        assert not result.multipliers
        (agreement,) = [
            item
            for item in result.assumptions
            if item.name == "analytic_and_finite_difference_agree"
        ]
        assert not agreement.held
        assert agreement.margin < -1e-3


def _dcm_model(tmp_path: Path):
    path = tmp_path / "dcm.yaml"
    body = BODY.format(kp=0.002, ki=30.0, integral0=0.0066).replace(
        "value: 5.45 ohm", "value: 60 ohm"
    )
    path.write_text(body.replace("initial_current: 0.92 A", "initial_current: 0 A"), encoding="utf-8")
    return load_model(path)


DCM_LOAD_OHM = 60.0


def test_dcm_analytic_monodromy_matches_the_dcm_oracle(tmp_path: Path) -> None:
    """In DCM the diode turns off at a state-dependent instant; the analytic
    path carries it by the guard saltation and must agree with a closed form
    derived differently (three exact intervals, brentq on the fall time,
    implicit-function derivatives — no saltation matrix).

    Measured 2026-11-15 at Newton tolerance 1e-12: multipliers 2.7e-13,
    monodromy 1.9e-11 relative, duty 6e-11. At the default 1e-9 Newton stops
    one iteration earlier and the orbit (not the saltation) is off by ~1e-7:
    hence the tighter tolerance here."""

    result = solve_closed_loop_steady_state(_dcm_model(tmp_path), tolerance=1e-12)
    assert result.converged
    assert result.monodromy_method == "analytic_saltation"
    (agreement,) = [
        item for item in result.assumptions if item.name == "analytic_and_finite_difference_agree"
    ]
    assert agreement.held
    oracle = solve_dcm(BuckLoop(**{**_PLANT, "load_ohm": DCM_LOAD_OHM}, kp=0.002, ki=30.0))
    engine = np.sort_complex(
        np.asarray([complex(item.real, item.imag) for item in result.multipliers])
    )
    assert np.max(np.abs(engine - np.sort_complex(oracle.multipliers))) < 1e-11
    scale = np.max(np.abs(oracle.monodromy))
    assert np.max(np.abs(np.asarray(result.monodromy) - oracle.monodromy)) < 1e-9 * scale
    state = dict(zip(result.state_names, result.state, strict=True))
    assert 30.0 * state["signal:int_prev"] == pytest.approx(oracle.duty, rel=1e-9)
    # Physics: the current returns to zero every cycle, so a perturbation of it
    # is erased — one multiplier is exactly 0, and the inductor column is zero.
    assert min(item.magnitude for item in result.multipliers) < 1e-12
    assert result.stability == "stable"


def test_the_dcm_oracle_reaches_the_textbook_duty_as_ripple_vanishes() -> None:
    """The oracle checked by physics, not by the engine: the textbook DCM buck
    duty D = sqrt(2 L fsw / R * M^2 / (1 - M)) assumes a ripple-free output.
    The oracle's gap to it must shrink as 1/C (measured 1.04e-3, 1.04e-4,
    1.04e-5 for C = 0.1, 1, 10 mF)."""

    plant = {**_PLANT, "load_ohm": DCM_LOAD_OHM}
    m = plant["vref_v"] / plant["vin_v"]
    textbook = np.sqrt(
        2.0 * plant["inductance_h"] * plant["switching_frequency_hz"] / plant["load_ohm"]
        * m * m / (1.0 - m)
    )
    gaps = []
    for capacitance in (100e-6, 1e-3, 1e-2):
        orbit = solve_dcm(
            BuckLoop(**{**plant, "capacitance_f": capacitance}, kp=0.002, ki=30.0)
        )
        gaps.append(orbit.duty / textbook - 1.0)
    assert 0.0 < gaps[2] < gaps[1] < gaps[0] < 2e-3
    assert gaps[0] / gaps[1] == pytest.approx(10.0, rel=0.02)
    assert gaps[1] / gaps[2] == pytest.approx(10.0, rel=0.02)


def test_a_missing_guard_saltation_is_caught_at_runtime(tmp_path: Path, monkeypatch) -> None:
    """Mutation: drop the natural-commutation saltation (S = I). The inductor
    perturbation then survives the zero-current interval; the built-in
    finite-difference cross-check must withhold the verdict."""

    monkeypatch.setattr(
        closed_loop,
        "_guard_saltation",
        lambda f_minus, f_plus, guard, slope: np.eye(len(f_minus)),
    )
    result = solve_closed_loop_steady_state(_dcm_model(tmp_path))
    assert result.stability == "not_assessed"
    (agreement,) = [
        item for item in result.assumptions if item.name == "analytic_and_finite_difference_agree"
    ]
    assert not agreement.held
    assert agreement.margin < -1e-3


# --- K5: retained CLI bundle -------------------------------------------------


def test_cli_retains_a_manifested_bundle_and_exit_codes_carry_the_verdict(
    tmp_path: Path,
) -> None:
    import json

    from pear_sim.api.cli import main

    stable_out = tmp_path / "stable"
    code = main(
        ["analyze", "floquet", str(ROOT / "examples" / "buck_closed_loop_floquet.yaml"),
         "--out", str(stable_out), "--json"]
    )
    assert code == 0
    evidence = json.loads((stable_out / "floquet-evidence.json").read_text())
    assert evidence["schema"] == "pear-sim/floquet-evidence/v0.1"
    assert evidence["stability"] == "stable"
    assert [gate["id"] for gate in evidence["gates"]] == [
        "closed_loop_orbit_converged",
        "multipliers_assessed",
        "locally_stable",
        "lyapunov_certificate",
    ]
    assert all(gate["status"] == "pass" for gate in evidence["gates"])
    assert "not validation against hardware" in evidence["claim_boundary"]
    manifest = json.loads((stable_out / "artifact-manifest.json").read_text())
    listed = {entry["path"] for entry in manifest["artifacts"]}
    assert {"floquet.json", "floquet-evidence.json"} <= listed
    floquet = json.loads((stable_out / "floquet.json").read_text())
    assert floquet["monodromy_method"] == "analytic_saltation"

    hot = _model(tmp_path, HOT)
    hot_out = tmp_path / "hot"
    assert main(["analyze", "floquet", str(hot.source_path), "--out", str(hot_out)]) == 1
    hot_evidence = json.loads((hot_out / "floquet-evidence.json").read_text())
    assert hot_evidence["overall_status"] == "fail"
    assert hot_evidence["gates"][-1]["status"] == "fail"

    refused_out = tmp_path / "refused"
    assert (
        main(["analyze", "floquet", str(ROOT / "examples" / "buck_closed_loop.yaml"),
              "--out", str(refused_out)])
        == 2
    )
    assert not refused_out.exists()

    bad = main(["analyze", "floquet", str(hot.source_path), "--out",
                str(tmp_path / "bad"), "--initial-state", "l1"])
    assert bad == 2


# --- Lyapunov certificate: an eigenvalue-free witness of the verdict ---------

from pear_sim.analysis.closed_loop import lyapunov_certificate  # noqa: E402


def _cholesky_ok(matrix: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(0.5 * (matrix + matrix.T))
    except np.linalg.LinAlgError:
        return False
    return True


@pytest.mark.parametrize(
    ("matrix", "found", "rho_sq"),
    [
        # diagonal: the certificate is exact, contraction = max mu^2
        (np.diag([0.5, 0.9]), True, 0.81),
        (np.diag([0.5, 1.01]), False, None),
        # a pure rotation scaled by r: |mu| = r for both
        (0.95 * np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]]), True, 0.9025),
        # on the unit circle: lossless, never asymptotically stable
        (np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]]), False, None),
        # inside the circle by less than the margin: round-off cannot tell it
        # from the lossless case, so no certificate either
        (
            (1.0 - 1e-12)
            * np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]]),
            False,
            None,
        ),
        # strongly non-normal but stable: transient growth, still certifiable
        (np.array([[0.9, 100.0], [0.0, 0.8]]), True, 0.81),
    ],
)
def test_certificate_on_closed_form_cases(matrix, found, rho_sq) -> None:
    """The unit-circle case regressed on ARM (2026-11-15): P - M^T P M is zero
    up to round-off there, and Cholesky accepted +1.3e-17 as positive."""

    certificate = lyapunov_certificate(matrix)
    assert certificate.found is found
    if found:
        assert certificate.contraction_bound == pytest.approx(rho_sq, rel=1e-9)
        p = np.asarray(certificate.p_matrix)
        assert _cholesky_ok(p) and _cholesky_ok(p - matrix.T @ p @ matrix)


def test_certificate_discriminates_the_two_designs_and_is_tight(solved) -> None:
    healthy = solved["healthy"][1]
    certificate = healthy.certificate
    assert certificate.found
    assert certificate.certifies_other_monodromy is True
    # measured 2026-11-15: 0.97940130 = |mu_max|^2 to 1e-15
    assert certificate.contraction_bound == pytest.approx(
        healthy.max_multiplier_magnitude**2, rel=1e-9
    )
    assert not solved["hot"][1].certificate.found


def test_certificate_rechecks_from_the_bundle_alone(tmp_path: Path) -> None:
    """The point of a certificate: two Cholesky factorisations on numbers in
    floquet.json, no PEARSim needed. And it must not certify a map it was not
    made for — the hot design's monodromy fails the healthy P."""

    import json

    from pear_sim.api.cli import main

    out = tmp_path / "bundle"
    assert main(["analyze", "floquet", str(ROOT / "examples" / "buck_closed_loop_floquet.yaml"),
                 "--out", str(out)]) == 0
    payload = json.loads((out / "floquet.json").read_text())
    m = np.asarray(payload["monodromy"])
    p = np.asarray(payload["lyapunov_certificate"]["p_matrix"])
    assert _cholesky_ok(p)
    assert _cholesky_ok(p - m.T @ p @ m)

    hot_dir = tmp_path / "hot"
    hot_dir.mkdir()
    hot = solve_closed_loop_steady_state(_model(hot_dir, HOT))
    hot_m = np.asarray(hot.monodromy)
    assert not _cholesky_ok(p - hot_m.T @ p @ hot_m)


def test_certificate_refuses_a_decrease_that_round_off_could_fake(monkeypatch) -> None:
    """Simulate the ARM outcome on any machine: force Cholesky to accept the
    zero-decrease matrix. The margin, not the factorisation, must refuse."""

    rotation = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
    monkeypatch.setattr(closed_loop, "_positive_definite", lambda matrix: True)
    assert not lyapunov_certificate(rotation).found
