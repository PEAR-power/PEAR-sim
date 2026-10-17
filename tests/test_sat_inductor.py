"""Saturable inductor: loader, segment events, energy, closed forms.

The piecewise-linear two-segment flux model: incremental inductance ``value``
below ``i_sat`` and ``l_sat`` beyond it. Segment crossings are exact
integration-restart events; the region is re-derived from the state, with the
derivative sign resolving an exactly-on-boundary state.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.model import load_model

RL_BODY = """
schema: pear-sim/v0.1
name: rl_sat

circuit:
  ground: gnd
  components:
    - id: vin
      type: voltage_source
      p: in
      n: gnd
      {source}
    - {{id: r1, type: resistor, p: in, n: mid, value: 1 ohm}}
    - id: lsat
      type: sat_inductor
      p: mid
      n: gnd
      value: 1 mH
      l_sat: 0.2 mH
      i_sat: 5 A
      initial_current: {i0}

analysis:
  transient:
    start: 0 s
    stop: {stop}
    output_step: 2 us
    solver: dopri5
    rtol: 1.0e-10
    atol: 1.0e-12

outputs:
  - i(lsat)
  - v(mid)
"""


def _model(tmp_path: Path, *, source: str = "value: 10 V", i0: str = "0 A", stop: str = "5 ms"):
    path = tmp_path / "model.yaml"
    path.write_text(RL_BODY.format(source=source, i0=i0, stop=stop), encoding="utf-8")
    return load_model(path)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_loader_requires_both_saturation_fields(tmp_path: Path) -> None:
    body = RL_BODY.format(source="value: 10 V", i0="0 A", stop="1 ms")
    body = body.replace("      l_sat: 0.2 mH\n", "")
    path = tmp_path / "model.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ModelError, match="l_sat"):
        load_model(path)


def test_loader_rejects_sat_inductance_not_below_value(tmp_path: Path) -> None:
    body = RL_BODY.format(source="value: 10 V", i0="0 A", stop="1 ms")
    body = body.replace("l_sat: 0.2 mH", "l_sat: 2 mH")
    path = tmp_path / "model.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ModelError, match="smaller than"):
        load_model(path)


def test_loader_rejects_saturation_fields_on_plain_inductor(tmp_path: Path) -> None:
    body = RL_BODY.format(source="value: 10 V", i0="0 A", stop="1 ms")
    body = body.replace("type: sat_inductor", "type: inductor")
    path = tmp_path / "model.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ModelError, match="only valid for sat_inductor"):
        load_model(path)


def test_two_sat_inductors_fail_closed(tmp_path: Path) -> None:
    body = RL_BODY.format(source="value: 10 V", i0="0 A", stop="1 ms")
    body = body.replace(
        "    - {id: r1, type: resistor, p: in, n: mid, value: 1 ohm}",
        """    - {id: r1, type: resistor, p: in, n: mid, value: 1 ohm}
    - id: lsat2
      type: sat_inductor
      p: in
      n: mid
      value: 1 mH
      l_sat: 0.2 mH
      i_sat: 5 A""",
    )
    path = tmp_path / "model.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ModelError, match="at most one sat_inductor"):
        HybridSimulator(load_model(path))


# ---------------------------------------------------------------------------
# Kernel vs closed forms
# ---------------------------------------------------------------------------


def test_rl_step_saturates_with_exact_segment_event(tmp_path: Path) -> None:
    result = HybridSimulator(_model(tmp_path)).simulate()
    t, i = result.time, result.signals["i(lsat)"]
    tau_main, tau_sat = 1e-3, 0.2e-3
    t1 = tau_main * math.log(2.0)  # current reaches i_sat = V/(2R)
    reference = np.where(
        t < t1,
        10.0 * (1.0 - np.exp(-t / tau_main)),
        10.0 - 5.0 * np.exp(-np.clip(t - t1, 0.0, None) / tau_sat),
    )
    assert np.max(np.abs(i - reference)) < 5e-8
    events = [e for e in result.events if e.kind == "saturation_boundary"]
    assert len(events) == 1
    assert events[0].time == pytest.approx(t1, abs=1e-9)
    assert (events[0].previous, events[0].current) == (False, True)
    audit = result.physics_audit["full"]
    assert audit["balance_status"] == "within_numerical_tolerance"


def test_step_down_desaturates_through_the_same_boundary(tmp_path: Path) -> None:
    model = _model(
        tmp_path,
        source="waveform: {type: step, initial: 10 V, final: 0 V, at: 3 ms}",
        stop="6 ms",
    )
    result = HybridSimulator(model).simulate()
    t, i = result.time, result.signals["i(lsat)"]
    tau_main, tau_sat = 1e-3, 0.2e-3
    t1 = tau_main * math.log(2.0)
    # Decay from ~10 A at 3 ms: saturated segment until i = 5 A, then main.
    i_at_step = 10.0 - 5.0 * math.exp(-(3e-3 - t1) / tau_sat)
    t2 = 3e-3 + tau_sat * math.log(i_at_step / 5.0)
    decay = np.where(
        t < t2,
        i_at_step * np.exp(-np.clip(t - 3e-3, 0.0, None) / tau_sat),
        5.0 * np.exp(-np.clip(t - t2, 0.0, None) / tau_main),
    )
    reference = np.where(
        t < t1,
        10.0 * (1.0 - np.exp(-t / tau_main)),
        np.where(t < 3e-3, 10.0 - 5.0 * np.exp(-np.clip(t - t1, 0, None) / tau_sat), decay),
    )
    assert np.max(np.abs(i - reference)) < 5e-7
    events = [e for e in result.events if e.kind == "saturation_boundary"]
    assert [(e.previous, e.current) for e in events] == [(False, True), (True, False)]
    assert events[1].time == pytest.approx(t2, abs=1e-8)
    assert result.physics_audit["full"]["balance_status"] == "within_numerical_tolerance"


def test_warm_start_inside_saturation_selects_saturated_region(tmp_path: Path) -> None:
    result = HybridSimulator(_model(tmp_path, i0="8 A", stop="2 ms")).simulate()
    assert result.mode_intervals[0]["mode_signature"].endswith(
        "saturated_inductors[lsat]"
    )
    # 8 A start, drives to 10 A entirely inside the saturated segment.
    t, i = result.time, result.signals["i(lsat)"]
    reference = 10.0 - 2.0 * np.exp(-t / 0.2e-3)
    assert np.max(np.abs(i - reference)) < 5e-8
    assert not [e for e in result.events if e.kind == "saturation_boundary"]


def test_sat_inductor_simulation_is_bit_deterministic(tmp_path: Path) -> None:
    first = HybridSimulator(_model(tmp_path)).simulate()
    second = HybridSimulator(_model(tmp_path)).simulate()
    assert np.array_equal(first.signals["i(lsat)"], second.signals["i(lsat)"])


def test_stored_energy_is_piecewise_not_half_l_i_squared(tmp_path: Path) -> None:
    result = HybridSimulator(_model(tmp_path)).simulate()
    audit = result.physics_audit["full"]
    final_current = 10.0
    lm, lsat, isat = 1e-3, 0.2e-3, 5.0
    expected = 0.5 * lm * isat**2 + 0.5 * lsat * (final_current**2 - isat**2)
    assert audit["storage_energy_end_j"] == pytest.approx(expected, rel=1e-6)
    naive = 0.5 * lm * final_current**2
    assert abs(audit["storage_energy_end_j"] - naive) > 0.02  # clearly not naive
