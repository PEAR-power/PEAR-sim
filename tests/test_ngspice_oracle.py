# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""ngspice differential adapter: second independent evidence class.

Netlist-emission checks always run; live comparisons need a local ngspice
binary and self-skip otherwise. The declared error budget is RELATIVE:
1e-3 of each signal's peak-to-peak range. It covers ngspice solver
tolerance AND the linear resampling of its ``linearize`` command on the
shared output grid — the dominant term for smooth waveforms scales with
amplitude (~(w*dt)^2/8 * amplitude), so an absolute budget would be either
loose for small signals or false for ringing ones.
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from pear_sim.core.solver import HybridSimulator
from pear_sim.oracle.ngspice import (
    NgspiceOracleError,
    emit_netlist,
    find_ngspice,
    simulate_model,
)
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]
RECORDS_DIR = ROOT / "validation" / "oracle-records"
RELATIVE_BUDGET = 1e-3  # of each signal's peak-to-peak range

_live = pytest.mark.skipif(
    find_ngspice() is None, reason="no ngspice binary on this machine"
)


def _model(tmp_path: Path, body: str):
    path = tmp_path / "model.yaml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    return load_model(path)


RC_SINE = """
schema: pear-sim/v0.1
name: rc_sine_diff
circuit:
  ground: gnd
  components:
    - id: vin
      type: voltage_source
      p: in
      n: gnd
      waveform: {type: sine, amplitude: 5 V, frequency: 1 kHz}
    - {id: r1, type: resistor, p: in, n: out, value: 1 kohm}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 nF, initial_voltage: 0 V}
analysis:
  transient: {start: 0 s, stop: 5 ms, output_step: 5 us, solver: dopri5, rtol: 1e-9, atol: 1e-11}
outputs: [v(out)]
"""

RLC_STEP = """
schema: pear-sim/v0.1
name: rlc_step_diff
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 10 V}
    - {id: r1, type: resistor, p: in, n: mid, value: 20 ohm}
    - {id: l1, type: inductor, p: mid, n: out, value: 10 mH, initial_current: 0 A}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 1 uF, initial_voltage: 0 V}
analysis:
  transient: {start: 0 s, stop: 5 ms, output_step: 5 us, solver: dopri5, rtol: 1e-9, atol: 1e-11}
outputs: [v(out), v(mid)]
"""

PWL_RAMP = """
schema: pear-sim/v0.1
name: pwl_ramp_diff
circuit:
  ground: gnd
  components:
    - id: vin
      type: voltage_source
      p: in
      n: gnd
      waveform: {type: pwl, points: [[0 s, 0 V], [1 ms, 0 V], [3 ms, 5 V], [5 ms, 5 V]]}
    - {id: r1, type: resistor, p: in, n: out, value: 1 kohm}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 nF, initial_voltage: 0 V}
analysis:
  transient: {start: 0 s, stop: 5 ms, output_step: 5 us, solver: dopri5, rtol: 1e-9, atol: 1e-11}
outputs: [v(out)]
"""


# ---------------------------------------------------------------------------
# Netlist emission (always run)
# ---------------------------------------------------------------------------


def test_netlist_maps_the_linear_subset(tmp_path: Path) -> None:
    netlist, compared = emit_netlist(_model(tmp_path, RLC_STEP))
    assert compared == ["out", "mid"]
    assert "R1 in mid 20" in netlist
    # This model declares explicit zero ICs; they are emitted and honoured.
    assert "ic=0" in netlist
    assert "uic" in netlist
    assert ".options reltol=1e-6" in netlist


def test_netlist_fails_closed_outside_the_subset(tmp_path: Path) -> None:
    body = RLC_STEP.replace(
        "    - {id: r1, type: resistor, p: in, n: mid, value: 20 ohm}",
        """    - {id: r1, type: resistor, p: in, n: mid, value: 20 ohm}
    - {id: d1, type: diode, p: mid, n: gnd, initial: off}""",
    )
    with pytest.raises(NgspiceOracleError, match="physical models"):
        emit_netlist(_model(tmp_path, body))


def test_netlist_requires_a_comparable_output(tmp_path: Path) -> None:
    body = RC_SINE.replace("outputs: [v(out)]", "outputs: [i(r1)]")
    with pytest.raises(NgspiceOracleError, match="v\\(node\\)"):
        emit_netlist(_model(tmp_path, body))


# ---------------------------------------------------------------------------
# Live differential comparisons
# ---------------------------------------------------------------------------


def _persist(case_id, simulation, deviations, *, derived=None, notes=None) -> None:
    """Retain the comparison record.

    ``derived`` carries signals that follow ALGEBRAICALLY from a compared node
    voltage (a resistor current, say). They are recorded because a report may
    cite them, and labelled as derived because they add no independent
    evidence: both engines compute them from the same compared quantity.
    """

    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": "pear-sim/oracle-comparison/v0.1",
        "case_id": case_id,
        "oracle_product": "ngspice",
        "oracle_version_declared": simulation.version,
        "pear_sim_version": __import__("pear_sim").__version__,
        "captured_at_utc": os.environ.get("PEAR_SIM_CAPTURE_UTC", "unrecorded"),
        "fixture": "netlist emitted from the model itself (differential adapter)",
        "grid_points": len(simulation.time),
        "signals": {
            name: {
                "max_abs_deviation": entry["deviation"],
                "tolerance": entry["budget"],
                "unit": "V",
            }
            for name, entry in deviations.items()
        },
        "verdict": (
            "pass"
            if all(e["deviation"] < e["budget"] for e in deviations.values())
            else "fail"
        ),
        "notes": notes
        or (
            "linear-subset differential run; RELATIVE budget 1e-3 of each "
            "signal's peak-to-peak covers ngspice solver tolerance plus the "
            "linearize resampling term on the shared grid"
        ),
    }
    if derived:
        record["derived_signals"] = derived
    (RECORDS_DIR / f"{case_id}.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _compare(model):
    result = HybridSimulator(model).simulate()
    simulation = simulate_model(model)
    deviations = {}
    ngspice_time = np.asarray(simulation.time)
    for node, values in simulation.voltages.items():
        signal = result.signals[f"v({node})"]
        resampled = np.interp(result.time, ngspice_time, np.asarray(values))
        peak_to_peak = float(np.max(signal) - np.min(signal))
        deviations[f"v({node})"] = {
            "deviation": float(np.max(np.abs(resampled - signal))),
            "budget": RELATIVE_BUDGET * max(peak_to_peak, 1e-9),
        }
    return simulation, deviations


@_live
def test_rc_sine_agrees_within_budget(tmp_path: Path) -> None:
    simulation, deviations = _compare(_model(tmp_path, RC_SINE))
    entry = deviations["v(out)"]
    assert entry["deviation"] < entry["budget"]
    _persist("ngspice-rc-sine", simulation, deviations)


@_live
def test_rlc_step_agrees_within_budget(tmp_path: Path) -> None:
    simulation, deviations = _compare(_model(tmp_path, RLC_STEP))
    for entry in deviations.values():
        assert entry["deviation"] < entry["budget"]
    _persist("ngspice-rlc-step", simulation, deviations)


@_live
def test_pwl_ramp_agrees_within_budget(tmp_path: Path) -> None:
    simulation, deviations = _compare(_model(tmp_path, PWL_RAMP))
    entry = deviations["v(out)"]
    assert entry["deviation"] < entry["budget"]
    _persist("ngspice-pwl-ramp", simulation, deviations)


@_live
def test_step_waveform_edge_semantics_stay_inside_the_budget(tmp_path: Path) -> None:
    body = RC_SINE.replace(
        "      waveform: {type: sine, amplitude: 5 V, frequency: 1 kHz}",
        "      waveform: {type: step, initial: 0 V, final: 5 V, at: 2 ms}",
    ).replace("name: rc_sine_diff", "name: rc_step_diff")
    simulation, deviations = _compare(_model(tmp_path, body))
    # The declared 1 ns PWL edge vs pear_sim's exact discontinuity: the transient
    # difference dies with tau = 100 us, far below the budget on this grid.
    entry = deviations["v(out)"]
    assert entry["deviation"] < entry["budget"]
    _persist("ngspice-rc-step", simulation, deviations)


def test_a_switched_model_is_refused_by_the_adapter() -> None:
    """Ideal switches are outside the comparable linear subset.

    A stated boundary that nothing tests is just a sentence, so the refusal
    is asserted against a shipped switching model.
    """

    model = load_model(ROOT / "examples" / "buck.yaml")
    with pytest.raises(NgspiceOracleError):
        emit_netlist(model)
