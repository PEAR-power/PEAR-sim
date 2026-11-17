# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Three-phase SPWM with dead time against an independent oracle (VAL-005).

The acceptance matrix behind ``converter.inverter_three_phase.spwm_svpwm_dead_time``:
modulation law (plain SPWM, min-max/SVPWM-equivalent) x dead time (0, 0.5, 1 us)
x load (a well-damped RL with small ripple, and a ripple-dominated RL whose
current reverses INSIDE dead-time gaps) x modulation index (0.5, 0.9) — 24
runs of the shipped inverter example, each compared with
``pear_sim.benchmarks.spwm_dead_time``, a closed-form piecewise-exponential
solution that shares nothing with the kernel but the circuit definition.

Compared per run: the three phase currents on the whole output grid, the
complex Fourier coefficients 0..42 of the line voltage v_ab and of the phase
voltage v_an (which also carries the triplen dead-time and injection content
that cancels in the line voltage), and the bandwidth-limited THD of v_ab.

Budgets. Voltage spectra and THD: 1e-9 (relative to the fundamental;
absolute for THD) — the pole voltage is piecewise constant, so the kernel's
Fourier integral is exact up to round-off and the switching instants. Phase
currents: 100 x the declared rtol (1e-9) of the example, relative to the peak —
the current IS integrated, and with the 25 us time constant of the
ripple-dominated load the integrator error is what shows (measured 1.8e-9).
Measured on 2026-11-16: spectra and THD worst 4.3e-12, currents 2.0e-9.

The comparison has to be able to ring: three wrong oracles (no dead time, the
dead-time pole on the wrong rail, no zero-current clamp) are run against the
same kernel results and must each miss the budget by orders of magnitude.
The clamp is only visible where a current reverses inside a gap in the
harmonic window, so the matrix is required to contain such runs.
"""

from __future__ import annotations

import math
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import yaml

from pear_sim.benchmarks.spwm_dead_time import (
    SpwmDeadTimeCase,
    leading_order_fundamental_error,
    solve,
    thd_fraction,
)
from pear_sim.core.solver import HybridSimulator
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "inverter_three_phase_spwm.yaml"

HALF_BUS = 200.0
FUNDAMENTAL = 500.0
RATIO = 42
STOP = 6e-3
WINDOW = (2e-3, 6e-3)
ORDERS = 42
OMEGA = 2.0 * math.pi * FUNDAMENTAL

BUDGET = 1e-9
# The example declares dop853 at rtol 1e-9; the current inherits that error.
CURRENT_BUDGET = 100 * 1e-9
# A wrong oracle must miss by at least this factor over the budget.
RING_FACTOR = 1e3

LOADS = {
    "damped": (10.0, 5e-3),  # tau 0.5 ms, ripple << current
    "ripple": (40.0, 1e-3),  # tau 25 us, ripple ~ current: reversals in gaps
}
MATRIX = list(
    product(("none", "min_max"), (0, 500, 1000), tuple(LOADS), (0.5, 0.9))
)


def _ident(point) -> str:
    injection, dead_ns, load, m = point
    return f"{injection}-{dead_ns}ns-{load}-m{m}"


def _case(point) -> SpwmDeadTimeCase:
    injection, dead_ns, load, m = point
    r, inductance = LOADS[load]
    return SpwmDeadTimeCase(
        half_bus_v=HALF_BUS,
        fundamental_hz=FUNDAMENTAL,
        carrier_ratio=RATIO,
        modulation_index=m,
        dead_time_s=dead_ns * 1e-9,
        phase_r_ohm=r,
        phase_l_h=inductance,
        stop_s=STOP,
        injection=injection,
    )


@lru_cache(maxsize=None)
def _kernel(point, directory: str):
    injection, dead_ns, load, m = point
    r, inductance = LOADS[load]
    document = yaml.safe_load(EXAMPLE.read_text(encoding="utf-8"))
    parameters = document["parameters"]
    parameters["dead_time"] = f"{dead_ns} ns"
    parameters["modulation_index"] = f"{m * 100} %"
    parameters["phase_r"] = f"{r} ohm"
    parameters["phase_l"] = f"{inductance * 1e3} mH"
    if injection != "none":
        document["controls"]["bridge"]["injection"] = injection
    document["analysis"]["transient"]["stop"] = f"{STOP * 1e3} ms"
    harmonics = document["analysis"]["harmonics"]
    harmonics["start"] = f"{WINDOW[0] * 1e3} ms"
    harmonics["stop"] = f"{WINDOW[1] * 1e3} ms"
    harmonics["signals"].append({"signal": "v(van)", "semantic": "phase_voltage_a"})
    path = Path(directory) / f"{_ident(point)}.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = HybridSimulator(load_model(path)).simulate()
    records = {item["semantic"]: item for item in result.harmonic_analysis["signals"]}

    def coefficients(semantic: str) -> np.ndarray:
        return np.array(
            [c["coefficient_real"] + 1j * c["coefficient_imag"] for c in records[semantic]["coefficients"]]
        )

    return {
        "time": np.asarray(result.time, dtype=float),
        "currents": [np.asarray(result.signals[f"i(l{x})"], dtype=float) for x in "abc"],
        "vab": coefficients("line_voltage_ab"),
        "van": coefficients("phase_voltage_a"),
        "thd": records["line_voltage_ab"]["thd"]["value_fraction"],
    }


@pytest.fixture(scope="module")
def workdir(tmp_path_factory) -> str:
    return str(tmp_path_factory.mktemp("spwm_dead_time"))


def _deviations(kernel, phases) -> dict[str, float]:
    peak = max(float(np.max(np.abs(current))) for current in kernel["currents"])
    current = max(
        float(np.max(np.abs(observed - phase.current(kernel["time"]))))
        for observed, phase in zip(kernel["currents"], phases)
    ) / peak
    a = phases[0].pole_coefficients(*WINDOW, OMEGA, ORDERS)
    b = phases[1].pole_coefficients(*WINDOW, OMEGA, ORDERS)
    vab, van = a - b, a
    return {
        "current": current,
        "vab": float(np.max(np.abs(kernel["vab"] - vab)) / abs(vab[1])),
        "van": float(np.max(np.abs(kernel["van"] - van)) / abs(van[1])),
        "thd": abs(kernel["thd"] - thd_fraction(vab)),
    }


def _clamped_in_window(phase) -> float:
    """Seconds of the harmonic window this leg spends with its diode starved
    (current held at zero, pole at the load voltage)."""

    a = np.maximum(phase.start, WINDOW[0])
    b = np.minimum(phase.stop, WINDOW[1])
    keep = (b > a) & (phase.pole == 0.0) & (phase.initial == 0.0)
    return float(np.sum(b[keep] - a[keep]))


@pytest.mark.parametrize("point", MATRIX, ids=_ident)
def test_the_kernel_matches_the_independent_dead_time_oracle(point, workdir) -> None:
    deviations = _deviations(_kernel(point, workdir), solve(_case(point)))
    for name, value in deviations.items():
        assert value < (CURRENT_BUDGET if name == "current" else BUDGET), (name, value)


@pytest.mark.parametrize(
    "point", [p for p in MATRIX if p[1] > 0], ids=_ident
)
def test_a_wrong_dead_time_model_misses_by_orders_of_magnitude(point, workdir) -> None:
    kernel = _kernel(point, workdir)
    for mutation in ("no_dead_time", "flipped_dead_time_pole"):
        deviations = _deviations(kernel, solve(_case(point), _mutation=mutation))
        assert deviations["current"] > 10 * CURRENT_BUDGET, mutation
        assert deviations["vab"] > RING_FACTOR * BUDGET, mutation
    phases = solve(_case(point))
    unclamped = _deviations(kernel, solve(_case(point), _mutation="no_zero_clamp"))
    if any(_clamped_in_window(phase) > 0.0 for phase in phases):
        # Without the clamp the starved diode keeps its rail: the current moves.
        assert unclamped["current"] > 10 * CURRENT_BUDGET
    if _clamped_in_window(phases[0]) > 0.0:
        # ...and so does the spectrum of the leg that clamps (v_an, v_ab see leg a).
        assert unclamped["van"] > RING_FACTOR * BUDGET
        assert unclamped["vab"] > RING_FACTOR * BUDGET


def test_the_matrix_reverses_current_inside_dead_time_gaps() -> None:
    """Tripwire: without clamped gaps on leg a inside the harmonic window the
    spectral clamp check above would be vacuous. Measured 2026-11-16: 9 of 16
    nonzero-dead-time points clamp leg a inside the window."""

    clamped = {
        _ident(p): _clamped_in_window(solve(_case(p))[0]) for p in MATRIX if p[1] > 0
    }
    assert sum(1 for value in clamped.values() if value > 0.0) >= 6, clamped
    assert any(value > 10e-6 for value in clamped.values()), clamped


@pytest.mark.parametrize("m", (0.5, 0.9))
def test_the_textbook_dead_time_loss_holds_only_while_the_ripple_is_small(m) -> None:
    """The oracle against the closed form every textbook gives:
    |dV1| = (4/pi)*Td*fc*Vdc, in phase with the current.

    With small ripple they agree within a few percent. With a ripple-dominated
    load the current reverses inside the gaps, part of every gap is clamped at
    0 V instead of a rail, and the real loss is well BELOW the formula
    (measured 0.47..0.90 of it). The formula overstates the dead-time loss
    exactly where the ripple is large.
    """

    for load, low, high in (("damped", 0.95, 1.05), ("ripple", 0.4, 0.95)):
        point = ("none", 2000, load, m)
        phases = solve(_case(point))
        reference = solve(_case(("none", 0, load, m)))
        measured = (
            phases[0].pole_coefficients(*WINDOW, OMEGA, 1)[1]
            - reference[0].pole_coefficients(*WINDOW, OMEGA, 1)[1]
        )
        current = phases[0].current_coefficients(*WINDOW, OMEGA, 1)[1]
        formula = leading_order_fundamental_error(_case(point), float(np.angle(current)) + math.pi / 2.0)
        ratio = abs(measured) / abs(formula)
        assert low < ratio < high, (load, m, ratio)
        assert abs(math.degrees(np.angle(measured / formula))) < 5.0, (load, m)
