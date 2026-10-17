"""A field-oriented current controller for a PMSM, written as data.

The structure is the standard textbook one — Park transform on the rotor
angle, dq PI regulators with BACK-CALCULATION anti-windup, a decoupling
feed-forward behind a flag, a CIRCULAR voltage limit that feeds the
anti-windup back, angle prediction, and inverse Park into a space-vector
modulator — expressed entirely in `signals`, with no line in `src/`.

These tests pin the properties that make it a controller rather than an
arrangement of blocks: that it REGULATES, that its integral term closes the
error, and that it shares the machine's frame convention.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.machines import PHASE_OFFSETS, salient_torque, parse_machines
from pear_sim.spec.model import load_model
from pear_sim.spec.signal_graph import evaluate_block, parse_signal_graph

ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "examples" / "pmsm_foc_current_control.yaml"
OFFSETS = [offset for _, offset in PHASE_OFFSETS]

ID_REF = -6.0
IQ_REF = 12.0


@pytest.fixture(scope="module")
def controlled():
    return HybridSimulator(load_model(CONTROLLER)).simulate()


def _tail(values):
    return values[int(len(values) * 0.7):]


def test_the_controller_reaches_its_current_references(controlled) -> None:
    """The headline: it REGULATES, not merely runs.

    Starting from zero current, the dq components must land on their
    references. The residual is the finite-gain steady-state error and is
    bounded here rather than assumed away.
    """

    id_mean = float(np.mean(_tail(controlled.signals["s(id_meas)"])))
    iq_mean = float(np.mean(_tail(controlled.signals["s(iq_meas)"])))
    assert id_mean == pytest.approx(ID_REF, abs=0.05)
    assert iq_mean == pytest.approx(IQ_REF, abs=0.05)
    # And it genuinely started somewhere else.
    assert abs(float(controlled.signals["s(id_meas)"][0])) < 1e-6


def test_the_integral_term_is_what_closes_the_error() -> None:
    """Raising ki must shrink the steady-state error, as theory says.

    Measured while building this: ki 0.02 leaves -0.70 A on the d axis,
    0.10 leaves -0.14 A, 0.30 leaves -0.0014 A. A controller whose error did
    not respond to its integral gain would be one whose loop is not actually
    closed — which is exactly the failure the signal layer's first
    accumulator had.
    """

    source = yaml.safe_load(CONTROLLER.read_text(encoding="utf-8"))
    errors = []
    for gain in (0.02, 0.3):
        variant = yaml.safe_load(yaml.safe_dump(source))
        variant["signals"]["blocks"]["ki_d"]["gain"] = gain
        variant["signals"]["blocks"]["ki_q"]["gain"] = gain
        path = ROOT / "tests" / "_torque_variant.yaml"
        try:
            path.write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
            result = HybridSimulator(load_model(path)).simulate()
        finally:
            path.unlink(missing_ok=True)
        errors.append(abs(float(np.mean(_tail(result.signals["s(id_meas)"]))) - ID_REF))
    assert errors[1] < errors[0] / 10.0


def test_torque_matches_the_closed_form_for_the_achieved_currents(controlled) -> None:
    """Controller and machine must share ONE frame convention.

    A controller transforming on a different convention than the machine it
    drives produces a plausible, wrong torque that no aggregate metric
    reveals. This repository has been bitten by exactly that three times, so
    the check is explicit: recompute the torque from the dq currents the
    CONTROLLER measured and compare with what the MACHINE reported.
    """

    spec = load_model(CONTROLLER).machines[0]
    id_mean = float(np.mean(_tail(controlled.signals["s(id_meas)"])))
    iq_mean = float(np.mean(_tail(controlled.signals["s(iq_meas)"])))
    expected = 1.5 * spec.pole_pairs * (
        spec.flux_linkage_wb * iq_mean
        + (spec.d_axis_inductance_h - spec.q_axis_inductance_h) * id_mean * iq_mean
    )
    measured = float(np.mean(_tail(controlled.signals["torque(motor)"])))
    assert measured == pytest.approx(expected, rel=2e-3)


def test_the_park_block_uses_the_machine_convention() -> None:
    """Pinned directly, not only through the drive.

    The block's transform must equal the one `salient_torque` performs
    internally — same offsets, same amplitude-invariant scaling.
    """

    graph = parse_signal_graph(
        {
            "task": {"period": 1e-4},
            "blocks": {
                "ia": {"type": "constant", "value": 10.0, "unit": "A"},
                "ib": {"type": "constant", "value": -4.0, "unit": "A"},
                "ic": {"type": "constant", "value": -6.0, "unit": "A"},
                "th": {"type": "constant", "value": 0.7, "unit": "rad"},
                "id": {"type": "park", "axis": "d", "a": "ia", "b": "ib",
                       "c": "ic", "angle": "th", "unit": "A"},
                "iq": {"type": "park", "axis": "q", "a": "ia", "b": "ib",
                       "c": "ic", "angle": "th", "unit": "A"},
            },
        },
        machine_ids=frozenset(),
        parameters={},
    )
    values: dict[str, float] = {}
    for block_id in graph.order:
        values[block_id] = evaluate_block(
            graph.by_id[block_id], values, {}, time=0.0, probe_reader=lambda e: 0.0
        )
    currents = (10.0, -4.0, -6.0)
    angle = 0.7
    reference_d = (2.0 / 3.0) * sum(
        c * math.cos(angle - o) for c, o in zip(currents, OFFSETS, strict=True)
    )
    reference_q = -(2.0 / 3.0) * sum(
        c * math.sin(angle - o) for c, o in zip(currents, OFFSETS, strict=True)
    )
    assert values["id"] == pytest.approx(reference_d, abs=1e-15)
    assert values["iq"] == pytest.approx(reference_q, abs=1e-15)

    spec = parse_machines(
        {
            "m": {
                "type": "pmsm", "terminals": ["a", "b", "c"], "neutral": "n",
                "pole_pairs": 3, "stator_resistance_ohm": 0.1,
                "stator_inductance_h": 2e-3, "flux_linkage_wb": 0.08,
                "inertia_kg_m2": 0.01, "d_axis_inductance_h": 1e-3,
                "q_axis_inductance_h": 3e-3,
            }
        }
    )[0]
    from_blocks = 1.5 * 3 * (
        0.08 * values["iq"] + (1e-3 - 3e-3) * values["id"] * values["iq"]
    )
    assert from_blocks == pytest.approx(salient_torque(spec, currents, angle), abs=1e-12)


def test_inverse_park_round_trips() -> None:
    graph = parse_signal_graph(
        {
            "task": {"period": 1e-4},
            "blocks": {
                "d": {"type": "constant", "value": 30.0, "unit": "V"},
                "q": {"type": "constant", "value": -40.0, "unit": "V"},
                "th": {"type": "constant", "value": 1.3, "unit": "rad"},
                "al": {"type": "inverse_park", "axis": "alpha", "d": "d", "q": "q",
                       "angle": "th", "unit": "V"},
                "be": {"type": "inverse_park", "axis": "beta", "d": "d", "q": "q",
                       "angle": "th", "unit": "V"},
            },
        },
        machine_ids=frozenset(),
        parameters={},
    )
    values: dict[str, float] = {}
    for block_id in graph.order:
        values[block_id] = evaluate_block(
            graph.by_id[block_id], values, {}, time=0.0, probe_reader=lambda e: 0.0
        )
    # Magnitude is preserved by a rotation.
    assert math.hypot(values["al"], values["be"]) == pytest.approx(50.0, abs=1e-12)


def test_the_anti_windup_path_must_go_through_a_delay(tmp_path) -> None:
    """Removing the delay must be REFUSED, with the cycle named.

    Back-calculation reads the limiter, which depends on the regulator
    output, which depends on the integral, which the anti-windup feeds — an
    algebraic loop. In a real controller the limiter is evaluated after the
    regulator, so its correction lands one sample later; the delay is the
    physics, not a workaround for the checker.
    """

    source = yaml.safe_load(CONTROLLER.read_text(encoding="utf-8"))
    source["signals"]["blocks"]["aw_d"]["input"] = "sat_gap_d"
    path = tmp_path / "looped.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    with pytest.raises(ModelError, match="algebraic loop through"):
        load_model(path)


def test_the_voltage_limit_is_a_circle_not_a_box(controlled) -> None:
    """A per-axis clamp would admit |u| up to sqrt(2) times the real bound."""

    limit = 329.09
    d_axis = controlled.signals["s(u_d_lim)"]
    q_axis = controlled.signals["s(u_q_lim)"]
    magnitude = np.hypot(d_axis, q_axis)
    assert float(np.max(magnitude)) <= limit * (1.0 + 1e-9)


def test_a_flag_selects_which_law_the_run_exercises(tmp_path) -> None:
    """Decoupling on and off must give measurably different runs.

    The controller carries these flags so the contribution of each term can be
    shown separately; a flag that changed nothing would mean the term was
    never wired in.
    """

    source = yaml.safe_load(CONTROLLER.read_text(encoding="utf-8"))
    source["signals"]["blocks"]["dist_d"]["flag"] = False
    source["signals"]["blocks"]["dist_q"]["flag"] = False
    path = tmp_path / "no_decoupling.yaml"
    path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
    without = HybridSimulator(load_model(path)).simulate()
    with_flag = HybridSimulator(load_model(CONTROLLER)).simulate()
    difference = float(
        np.max(np.abs(without.signals["s(u_q_lim)"] - with_flag.signals["s(u_q_lim)"]))
    )
    assert difference > 1.0
