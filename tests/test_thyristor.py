"""The thyristor is a LATCH, and each half of that has to be asserted.

Without gate-edge turn-on it is a diode. Without zero-current turn-off it is a
switch. Neither half can be reproduced by a gate signal alone, which is why the
latch lives in the active-set search rather than in the control layer — and why
the tests below are written as contrasts against the two devices it is NOT.

The physics reference is the closed form of a half-wave phase-controlled
rectifier on an R-L load:

    i(theta) = (Vm/Z) * [sin(theta - phi) - sin(alpha - phi) * exp((alpha-theta)/(wL/R))]

with the extinction angle beta the first root above alpha. It is derived here
independently of the kernel.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from pear_sim.core.circuit import Circuit
from pear_sim.errors import ModelError
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator

ROOT = Path(__file__).parents[1]
MODEL = ROOT / "examples" / "thyristor_phase_controlled.yaml"

VAMP = 325.0
FGRID = 50.0
RLOAD = 10.0
LLOAD = 20e-3
ALPHA_DEG = 60.0
PULSE_S = 0.02 / FGRID  # the 2 % gate pulse: 400 us

OMEGA = 2.0 * math.pi * FGRID
IMPEDANCE = math.hypot(RLOAD, OMEGA * LLOAD)
PHI = math.atan2(OMEGA * LLOAD, RLOAD)
WL_OVER_R = OMEGA * LLOAD / RLOAD
ALPHA = math.radians(ALPHA_DEG)

# Declared tolerances, measured then rounded up: the trajectory agrees with the
# closed form to 4.7e-3 A on a 25.4 A peak (0.018 %), and the extinction
# instant is resolved to better than one 20 us output sample (0.36 degrees).
TRAJECTORY_TOLERANCE_A = 0.02
ANGLE_TOLERANCE_DEG = 0.5


def _closed_form(theta: float) -> float:
    return (VAMP / IMPEDANCE) * (
        math.sin(theta - PHI)
        - math.sin(ALPHA - PHI) * math.exp((ALPHA - theta) / WL_OVER_R)
    )


def _extinction_angle() -> float:
    return brentq(_closed_form, ALPHA + 0.1, ALPHA + 1.5 * math.pi)


@pytest.fixture(scope="module")
def run():
    model = load_model(MODEL)
    return model, HybridSimulator(model).simulate()


def _conduction_window(result, period_start: float):
    times = np.asarray(result.time)
    current = np.asarray(result.signals["i(ll)"])
    mask = (times >= period_start) & (times < period_start + 1.0 / FGRID)
    window_times, window_current = times[mask], current[mask]
    conducting = window_current > 1e-9
    return window_times, window_current, conducting


def test_the_device_latches_after_its_gate_pulse_ends(run) -> None:
    """The defining behaviour: conduction OUTLIVES the gate.

    A switch would open with the pulse. Here the pulse is 400 us and the
    conduction lasts about 8.4 ms, so the current is carried by the latch for
    twenty times longer than the gate is asserted.
    """

    _, result = run
    times, current, conducting = _conduction_window(result, 40e-3)
    assert conducting.any()
    fired_at = float(times[conducting][0])
    extinguished_at = float(times[conducting][-1])

    gate_ends_at = 40e-3 + ALPHA / OMEGA + PULSE_S
    assert fired_at < gate_ends_at, "did not fire within its own gate pulse"
    assert extinguished_at > gate_ends_at + 5e-3, (
        "conduction stopped with the gate: this is a switch, not a latch"
    )
    # And it carried the current past the supply's zero crossing, which only an
    # inductive load plus a latch can do.
    assert extinguished_at > 40e-3 + 10e-3


def test_the_firing_and_extinction_angles_match_the_closed_form(run) -> None:
    _, result = run
    times, current, conducting = _conduction_window(result, 40e-3)
    fired_at = float(times[conducting][0])
    extinguished_at = float(times[conducting][-1])

    fired_deg = (fired_at - 40e-3) * OMEGA * 180.0 / math.pi
    assert fired_deg == pytest.approx(ALPHA_DEG, abs=ANGLE_TOLERANCE_DEG)

    beta_deg = math.degrees(_extinction_angle())
    extinguished_deg = (extinguished_at - 40e-3) * OMEGA * 180.0 / math.pi
    assert extinguished_deg == pytest.approx(beta_deg, abs=ANGLE_TOLERANCE_DEG)


def test_the_conducting_trajectory_matches_the_closed_form(run) -> None:
    _, result = run
    times, current, conducting = _conduction_window(result, 40e-3)
    angles = (times[conducting] - 40e-3) * OMEGA
    reference = np.array([max(_closed_form(angle), 0.0) for angle in angles])
    deviation = float(np.max(np.abs(reference - current[conducting])))
    assert deviation < TRAJECTORY_TOLERANCE_A, deviation
    # Peak current, the number a designer actually sizes on.
    analytic_peak = max(
        _closed_form(angle)
        for angle in np.linspace(ALPHA, _extinction_angle(), 4000)
    )
    assert float(current.max()) == pytest.approx(analytic_peak, rel=1e-4)


def test_an_ungated_thyristor_never_conducts_under_forward_bias(
    tmp_path: Path,
) -> None:
    """The contrast with a diode: forward bias is not enough.

    Removing the gate leaves a device that blocks all half-cycles. A diode in
    the same position would rectify.
    """

    body = MODEL.read_text(encoding="utf-8").replace(
        "control: fire, initial: off", "initial: off"
    )
    body = "\n".join(
        line
        for line in body.splitlines()
        if not line.startswith("controls:")
        and not line.strip().startswith(("fire:", "type: pwm", "frequency:", "duty:", "phase:"))
    )
    path = tmp_path / "ungated.yaml"
    path.write_text(body + "\n", encoding="utf-8")
    result = HybridSimulator(load_model(path)).simulate()
    assert float(np.max(np.abs(np.asarray(result.signals["i(ll)"])))) < 1e-12

    # The same circuit with a diode DOES conduct — proving the model is
    # otherwise capable of rectifying and the difference is the device.
    diode_body = body.replace("type: thyristor", "type: diode")
    diode_path = tmp_path / "diode.yaml"
    diode_path.write_text(diode_body, encoding="utf-8")
    diode_result = HybridSimulator(load_model(diode_path)).simulate()
    assert float(np.max(np.asarray(diode_result.signals["i(ll)"]))) > 20.0


def test_a_held_gate_reproduces_a_switch_in_series_with_a_diode(
    tmp_path: Path,
) -> None:
    """The equivalence that made the thyristor look like a cheap win.

    With the gate held on, a thyristor must close the moment it becomes forward
    biased and open at zero current — which is exactly an ideal switch in series
    with a diode. Asserting it both documents WHY a held-gate rectifier needed
    no new component and pins the boundary of that claim.
    """

    held = MODEL.read_text(encoding="utf-8").replace("  pulse: 2 %", "  pulse: 100 %")
    thyristor_path = tmp_path / "held.yaml"
    thyristor_path.write_text(held, encoding="utf-8")
    thyristor = HybridSimulator(load_model(thyristor_path)).simulate()

    equivalent = held.replace(
        "    - {id: t1, type: thyristor, p: ac, n: la, control: fire, initial: off}",
        "    - {id: t1, type: ideal_switch, p: ac, n: sw, control: fire}\n"
        "    - {id: d1, type: diode, p: sw, n: la, initial: off}",
    )
    equivalent_path = tmp_path / "switch_diode.yaml"
    equivalent_path.write_text(equivalent, encoding="utf-8")
    reference = HybridSimulator(load_model(equivalent_path)).simulate()

    thyristor_current = np.asarray(thyristor.signals["i(ll)"])
    reference_current = np.asarray(reference.signals["i(ll)"])
    assert thyristor_current.shape == reference_current.shape
    assert float(np.max(np.abs(thyristor_current - reference_current))) < 1e-9
    assert float(thyristor_current.max()) > 20.0, "the equivalence must not be trivial"


def test_a_later_firing_angle_delivers_less(tmp_path: Path) -> None:
    """Phase control, which is the whole reason the device is in the model."""

    means = []
    for alpha_ms, _label in ((1.111, "20 deg"), (3.333, "60 deg"), (5.556, "100 deg")):
        body = MODEL.read_text(encoding="utf-8").replace(
            "  alpha: 3.333 ms", f"  alpha: {alpha_ms} ms"
        )
        path = tmp_path / f"alpha_{alpha_ms}.yaml"
        path.write_text(body, encoding="utf-8")
        result = HybridSimulator(load_model(path)).simulate()
        times = np.asarray(result.time)
        current = np.asarray(result.signals["i(ll)"])
        tail = times >= times[-1] - 1.0 / FGRID
        means.append(float(current[tail].mean()))
    assert means[0] > means[1] > means[2] > 0.0, means


def test_a_model_without_thyristors_declares_no_latch() -> None:
    """Bit-identical behaviour for every existing model rests on this.

    The latch is a restriction on the active-set search; if the set of latching
    devices is empty the search is the one that was there before.
    """

    circuit = Circuit(load_model(ROOT / "examples" / "buck.yaml"))
    assert circuit.latching == frozenset()

    thyristor_circuit = Circuit(load_model(MODEL))
    assert thyristor_circuit.latching == {"t1"}
    # And the latching device shares the diode machinery rather than duplicating it.
    assert "t1" in {component.id for component in thyristor_circuit.diodes}


def test_a_gate_on_a_passive_component_is_still_refused(tmp_path: Path) -> None:
    body = MODEL.read_text(encoding="utf-8").replace(
        '    - {id: rl, type: resistor, p: la, n: lb, value: "${rload}"}',
        '    - {id: rl, type: resistor, p: la, n: lb, value: "${rload}", control: fire}',
    )
    path = tmp_path / "bad_control.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ModelError, match="control is only valid"):
        load_model(path)
