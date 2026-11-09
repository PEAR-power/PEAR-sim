# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Zero-current reduction for COUPLED windings.

The kernel could freeze a plain inductor at zero current by replacing it with a
0 V branch. A winding is not a wire: while it carries no current its neighbours'
flux still induces a voltage on its open terminals, so 0 V is a modelling error
rather than a reduction. Refusing the topology was the honest response, and it
cost every transformer-fed bridge rectifier.

Three claims are asserted here:

1. the induced voltage is RIGHT, checked against the turns ratio on an open
   secondary where the closed form is unambiguous;
2. the bridge fed through that transformer runs with a floating secondary
   neutral and nothing propping the windings up;
3. it agrees with the declared workaround it replaces — the 50 kohm shunts the
   twelve-pulse example still carries — which is how "negligible" stops being a
   claim and becomes a measurement.

What is still refused is stated too, and it is not what I first assumed: the
delta connection turned out to be solvable, while the twelve-pulse stack still
needs its declared shunts. The last two tests pin both facts.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pear_sim.core.circuit import Circuit
from pear_sim.errors import TopologyError
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator

ROOT = Path(__file__).parents[1]
MODEL = ROOT / "examples" / "rectifier_6pulse_transformer.yaml"
GRID_HZ = 50.0

SHUNTS = """
    - {id: rma, type: resistor, p: ya, n: yn, value: 50 kohm}
    - {id: rmb, type: resistor, p: yb, n: yn, value: 50 kohm}
    - {id: rmc, type: resistor, p: yc, n: yn, value: 50 kohm}
"""


@pytest.fixture(scope="module")
def run():
    model = load_model(MODEL)
    return model, HybridSimulator(model).simulate()


def _tail(times: np.ndarray) -> np.ndarray:
    return times >= times[-1] - 10e-3


def test_an_open_secondary_shows_the_induced_voltage_not_zero(tmp_path: Path) -> None:
    """The physics the old 0 V branch got wrong, in its simplest form.

    One primary driven by a sine, one secondary open. The secondary's terminal
    voltage must be k*sqrt(Ls/Lp) times the primary's — 0.495 here — and the
    old reduction would have reported zero.
    """

    body = """
schema: pear-sim/v0.1
name: open_secondary
circuit:
  ground: gnd
  components:
    - id: vin
      type: voltage_source
      p: pri
      n: gnd
      waveform: {type: sine, amplitude: 100 V, frequency: 50 Hz}
    - {id: lp, type: inductor, p: pri, n: gnd, value: 50 mH, initial_current: 0 A}
    - {id: ls, type: inductor, p: seca, n: gnd, value: 12.5 mH, initial_current: 0 A}
    - {id: dblock, type: diode, p: seca, n: cap, initial: off}
    - {id: rprobe, type: resistor, p: cap, n: gnd, value: 1 megaohm}
  couplings:
    - {inductors: [lp, ls], coupling: 0.99}
analysis:
  transient: {start: 0 s, stop: 40 ms, output_step: 20 us, solver: radau, rtol: 1e-8, atol: 1e-10}
outputs: [v(pri), v(seca), i(ls)]
"""
    path = tmp_path / "open_secondary.yaml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    result = HybridSimulator(load_model(path)).simulate()

    times = np.asarray(result.time)
    tail = _tail(times)
    primary = np.asarray(result.signals["v(pri)"])[tail]
    secondary = np.asarray(result.signals["v(seca)"])[tail]
    current = np.asarray(result.signals["i(ls)"])[tail]

    # The winding really is open: no current at all.
    assert float(np.max(np.abs(current))) < 1e-9

    expected_ratio = 0.99 * math.sqrt(12.5 / 50.0)
    primary_rms = float(np.sqrt(np.mean(primary**2)))
    secondary_rms = float(np.sqrt(np.mean(secondary**2)))
    assert primary_rms > 1.0
    assert secondary_rms / primary_rms == pytest.approx(expected_ratio, rel=1e-3)


def test_the_bridge_runs_with_a_floating_secondary_and_no_winding_shunts(run) -> None:
    model, result = run
    circuit = Circuit(model)
    # Nothing but windings meets at the secondary star point.
    star_company = [
        component.id
        for component in model.components
        if "yn" in (component.p, component.n)
    ]
    assert sorted(star_company) == ["lya", "lyb", "lyc"]

    times = np.asarray(result.time)
    tail = _tail(times)
    load = np.asarray(result.signals["i(rl)"])[tail]
    assert float(load.mean()) > 15.0
    assert circuit.coupling_groups == (
        ("lpa", "lya"),
        ("lpb", "lyb"),
        ("lpc", "lyc"),
    )


def test_the_ripple_is_six_pulse(run) -> None:
    _, result = run
    times = np.asarray(result.time)
    tail = _tail(times)
    load = np.asarray(result.signals["i(rl)"])[tail]
    window = load - load.mean()
    frequencies = np.fft.rfftfreq(len(window), float(times[1] - times[0]))
    amplitudes = np.abs(np.fft.rfft(window)) / len(window) * 2.0

    def at(hertz: float) -> float:
        return float(amplitudes[int(np.argmin(np.abs(frequencies - hertz)))])

    # Six pulses per cycle: the fundamental ripple is at 6*f_grid and dominates
    # the grid frequency itself, which balanced operation removes.
    assert at(6 * GRID_HZ) > 1.0
    assert at(6 * GRID_HZ) > 3.0 * at(12 * GRID_HZ)
    assert at(GRID_HZ) < 0.1 * at(6 * GRID_HZ)


def test_it_agrees_with_the_declared_shunt_workaround(tmp_path: Path, run) -> None:
    """The workaround called itself negligible; this measures it."""

    _, reference = run
    body = MODEL.read_text(encoding="utf-8").replace(
        "    # Six-pulse bridge.", SHUNTS.strip("\n") + "\n    # Six-pulse bridge."
    )
    path = tmp_path / "with_shunts.yaml"
    path.write_text(body, encoding="utf-8")
    shunted = HybridSimulator(load_model(path)).simulate()

    times = np.asarray(reference.time)
    tail = _tail(times)
    plain_load = np.asarray(reference.signals["i(rl)"])[tail]
    shunted_load = np.asarray(shunted.signals["i(rl)"])[tail]
    assert float(shunted_load.mean()) == pytest.approx(
        float(plain_load.mean()), rel=1e-5
    )
    # And not just the mean: the ripple matches too.
    assert float(shunted_load.std()) == pytest.approx(
        float(plain_load.std()), rel=1e-3
    )


def test_the_twelve_pulse_example_still_needs_its_declared_shunts(
    tmp_path: Path,
) -> None:
    """The remaining boundary, pinned where it actually is.

    My first guess was the delta connection, and it was wrong: a delta of
    coupled windings feeding a diode runs fine, because each corner is a
    two-winding series node the star reduction already handles. What still
    refuses is the twelve-pulse stack — removing EITHER set of winding shunts
    fails at the DC rail, where the filter inductor's own zero-current freeze
    has to coexist with six frozen windings, and the search finds no consistent
    mode. So the example keeps its declared shunts and this test records the
    real remainder instead of the one I assumed.
    """

    twelve_pulse = ROOT / "examples" / "rectifier_12pulse_yyd.yaml"
    body = twelve_pulse.read_text(encoding="utf-8")
    for group in (("rma", "rmb", "rmc"), ("rna", "rnb", "rnc")):
        stripped = "\n".join(
            line
            for line in body.splitlines()
            if not any(f"id: {name}," in line for name in group)
            and not any(f"- i({name})" in line for name in group)
        )
        path = tmp_path / f"without_{group[0]}.yaml"
        path.write_text(stripped + "\n", encoding="utf-8")
        with pytest.raises(TopologyError, match="no algebraic support"):
            HybridSimulator(load_model(path)).simulate()


def test_a_delta_of_coupled_windings_is_solvable(tmp_path: Path) -> None:
    """The guess that turned out to be wrong, kept as the record.

    Three windings in a delta share their currents through KCL at every corner.
    Each corner is a node where exactly two windings meet in series, which the
    generalized star reduction pins like any other — so the delta itself was
    never the blocker.
    """

    body = """
schema: pear-sim/v0.1
name: delta_secondary
circuit:
  ground: gnd
  components:
    - id: vin
      type: voltage_source
      p: pri
      n: gnd
      waveform: {type: sine, amplitude: 100 V, frequency: 50 Hz}
    - {id: lp, type: inductor, p: pri, n: gnd, value: 50 mH, initial_current: 0 A}
    - {id: lda, type: inductor, p: da, n: db, value: 37.5 mH, initial_current: 0 A}
    - {id: ldb, type: inductor, p: db, n: dc, value: 37.5 mH, initial_current: 0 A}
    - {id: ldc, type: inductor, p: dc, n: da, value: 37.5 mH, initial_current: 0 A}
    - {id: d1, type: diode, p: da, n: top, initial: off}
    - {id: rload, type: resistor, p: top, n: gnd, value: 20 ohm}
  couplings:
    - {inductors: [lp, lda], coupling: 0.5}
analysis:
  transient: {start: 0 s, stop: 5 ms, output_step: 20 us, solver: radau, rtol: 1e-7, atol: 1e-9}
outputs: [i(rload), i(lda)]
"""
    path = tmp_path / "delta.yaml"
    path.write_text(body.strip() + "\n", encoding="utf-8")
    result = HybridSimulator(load_model(path)).simulate()
    # The delta's circulating current is a genuine state and stays finite.
    assert np.all(np.isfinite(np.asarray(result.signals["i(lda)"])))
