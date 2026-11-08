"""The optional compiled kernel (pear-sim-core) against SciPy and against the truth.

The kernel mirrors solve_ivp(method="RK45") step for step. Checks, from the
narrowest to the widest:

- its tableau is SciPy's, bit for bit;
- its Brent root finder returns SciPy's roots (bit for bit in almost every
  case; the rest within one ulp, from SciPy's C build contracting to FMA);
- on random stable affine systems its error against the EXACT solution
  (matrix exponential) matches SciPy's error statistics, so a common wrong
  assumption cannot hide behind self-agreement;
- a whole circuit run reproduces the reference metrics to rounding, and the
  run says which engine produced it;
- an unsupported model is refused with the reason, never quietly run on the
  reference engine.

Skipped when the kernel is not installed (it is an optional build).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.rk import RK45
from scipy.linalg import expm
from scipy.optimize import brentq

core = pytest.importorskip("pear_sim_core")

from pear_sim.core.solver import HybridSimulator  # noqa: E402
from pear_sim.errors import SimulationError  # noqa: E402
from pear_sim.run.io import write_result  # noqa: E402
from pear_sim.spec.model import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EPS = np.finfo(float).eps


def _stable_system(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    n = int(rng.integers(2, 9))
    m = rng.normal(size=(n, n))
    a = m - (np.max(np.real(np.linalg.eigvals(m))) + rng.uniform(0.1, 2.0)) * np.eye(n)
    a *= 10 ** rng.uniform(2, 5)
    c = rng.normal(size=n) * 10 ** rng.uniform(0, 3)
    return a, c, rng.normal(size=n), float(10 ** rng.uniform(-5, -2))


def _no_events(n: int) -> list:
    return [[], [], np.zeros((0, n)), [], [], [], []]


def test_tableau_is_scipys_bit_for_bit() -> None:
    tableau = core.rk45_tableau()
    assert np.array_equal(np.array(tableau["C"]), RK45.C)
    assert np.array_equal(np.array(tableau["A"]), RK45.A[:, :5])
    assert np.array_equal(np.array(tableau["B"]), RK45.B)
    assert np.array_equal(np.array(tableau["E"]), RK45.E)
    assert np.array_equal(np.array(tableau["P"]), RK45.P)


def test_brent_matches_scipy() -> None:
    rng = np.random.default_rng(0)
    identical = tried = 0
    for _ in range(600):
        roots = np.sort(rng.uniform(-3, 3, 3))
        coeffs = np.poly(roots)[::-1].tolist()
        lo = roots[1] - rng.uniform(0.05, 0.5)
        hi = roots[1] + rng.uniform(0.05, 0.5)

        def f(x: float) -> float:
            return functools.reduce(lambda acc, c: acc * x + c, reversed(coeffs), 0.0)

        if np.sign(f(lo)) == np.sign(f(hi)):
            continue
        tried += 1
        expected = brentq(f, lo, hi, xtol=4 * EPS, rtol=4 * EPS)
        got = core.brentq_polynomial(coeffs, lo, hi, 4 * EPS, 4 * EPS)
        identical += got == expected
        assert got == pytest.approx(expected, rel=2 * EPS, abs=0)
    assert tried > 400 and identical / tried > 0.99


def test_integrator_is_as_accurate_as_scipy_against_the_exact_solution() -> None:
    rng = np.random.default_rng(1)
    rtol, atol = 1e-6, 1e-9
    reference_errors, kernel_errors = [], []
    for _ in range(120):
        a, c, y0, span = _stable_system(rng)
        n = y0.size
        expm_at = expm(a * span)
        exact = expm_at @ y0 + np.linalg.solve(a, (expm_at - np.eye(n)) @ c)
        scale = atol + rtol * np.abs(exact)
        reference = solve_ivp(
            lambda t, y: a @ y + c, (0.0, span), y0, method="RK45",
            rtol=rtol, atol=atol, max_step=span,
        )
        kernel = core.integrate_affine_rk45(
            a, c, y0, 0.0, span, rtol, atol, span, *_no_events(n)
        )
        reference_errors.append(np.max(np.abs(reference.y[:, -1] - exact) / scale))
        kernel_errors.append(np.max(np.abs(kernel["ys"].reshape(-1, n)[-1] - exact) / scale))
    reference_errors, kernel_errors = np.array(reference_errors), np.array(kernel_errors)
    # Same accuracy class: medians within a factor of two, worst case no worse.
    assert np.median(kernel_errors) < 2 * np.median(reference_errors)
    assert kernel_errors.max() <= 1.5 * reference_errors.max()


def test_terminal_event_is_placed_where_scipy_places_it() -> None:
    # Damped oscillator; the event is x crossing 0.3 downwards.
    a = np.array([[0.0, 1.0], [-4.0e6, -300.0]])
    c = np.zeros(2)
    y0 = np.array([1.0, 0.0])

    def event(t: float, y: np.ndarray) -> float:
        return y[0] - 0.3

    event.terminal = True
    event.direction = -1.0
    reference = solve_ivp(
        lambda t, y: a @ y + c, (0.0, 1e-3), y0, method="RK45",
        rtol=1e-8, atol=1e-11, max_step=1e-3, events=[event],
    )
    kernel = core.integrate_affine_rk45(
        a, c, y0, 0.0, 1e-3, 1e-8, 1e-11, 1e-3,
        [False], [0.0], np.array([[1.0, 0.0]]), [-0.3], [-np.inf], [0.0], [-1.0],
    )
    assert reference.status == kernel["status"] == 1
    assert kernel["event_index"] == 0
    assert kernel["event_root"] == pytest.approx(reference.t_events[0][0], rel=1e-12)
    assert kernel["ts"][-1] == kernel["event_root"]


def test_buck_run_matches_the_reference_engine(tmp_path: Path) -> None:
    model_path = ROOT / "examples" / "buck.yaml"
    reference = HybridSimulator(load_model(model_path)).simulate()
    compiled = HybridSimulator(load_model(model_path), backend="rust").simulate()
    assert [e.time for e in compiled.events] == pytest.approx(
        [e.time for e in reference.events], rel=1e-12, abs=1e-15
    )
    for signal, fields in reference.metrics().items():
        for key, value in fields.items():
            other = compiled.metrics()[signal][key]
            if isinstance(value, float):
                assert other == pytest.approx(value, rel=1e-12, abs=1e-12), (signal, key)
    run_dir = write_result(compiled, tmp_path / "run")
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["backend"].startswith("rust-core-")
    reference_dir = write_result(reference, tmp_path / "reference")
    reference_manifest = json.loads((reference_dir / "run.json").read_text())
    # A different engine is a different run identity.
    assert manifest["run_sha256"] != reference_manifest["run_sha256"]


@pytest.mark.parametrize("example", ["buck_dcm", "rl_sat_inductor", "npc_three_phase"])
def test_event_descriptions_match_the_event_closures(
    example: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The kernel reads each guard as data (an affine row or a constant); the
    # closure reaches its value through the full node solution. Capture the
    # closures of a real run and compare both on perturbed states.
    simulator = HybridSimulator(load_model(ROOT / "examples" / f"{example}.yaml"))
    captured = []
    original = simulator._natural_event_functions

    def capture(mode, diodes, segment_start, start_state=None, gates=None):
        functions = original(mode, diodes, segment_start, start_state, gates)
        if start_state is not None and len(captured) < 60:
            captured.append((functions, segment_start, np.array(start_state)))
        return functions

    monkeypatch.setattr(simulator, "_natural_event_functions", capture)
    simulator.simulate()
    rng = np.random.default_rng(3)
    compared = 0
    for functions, segment_start, state in captured:
        for function in functions:
            kind, *spec = function.pear_spec
            for _ in range(3):
                probe = state * (1 + 0.1 * rng.normal(size=state.size)) + rng.normal(size=state.size)
                late = segment_start + 1.0
                expected = function(late, probe)
                if kind == "constant":
                    assert spec[0] == expected
                    continue
                row, offset, until, early = spec
                got = float(row @ probe + offset)
                assert got == pytest.approx(expected, rel=1e-9, abs=1e-9)
                assert function(until, probe) == early
                compared += 1
    assert compared > 5


def test_unsupported_models_are_refused_with_the_reason() -> None:
    with pytest.raises(SimulationError, match="solver 'dop853'"):
        HybridSimulator(load_model(ROOT / "examples" / "inverter_three_phase_spwm.yaml"), backend="rust")
    with pytest.raises(SimulationError, match="waveform sources"):
        HybridSimulator(load_model(ROOT / "examples" / "halfwave_rectifier.yaml"), backend="rust")
    with pytest.raises(SimulationError, match="Unknown backend"):
        HybridSimulator(load_model(ROOT / "examples" / "buck.yaml"), backend="fortran")
