# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Capacitor ESR and inductor DCR: expansion, equivalence, fail-closed loading.

A declared ``esr``/``dcr`` is shorthand for a series resistor. These tests pin
that it is EXACTLY that shorthand (the same waveforms as the hand-written
resistor, bit for bit), that it unlocks the capacitor banks the ideal kernel
rejects, and that the topology error for such a bank says how to fix it.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

from pear_sim.core import rust_backend
from pear_sim.core.circuit import capacitor_loop_hint, dcr_star_hint
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError, TopologyError
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]

BUCK = """
schema: pear-sim/v0.1
name: buck_parasitics

parameters:
  esr_out: 20 mohm

circuit:
  ground: gnd
  components:
    - {{id: vin, type: voltage_source, p: in, n: gnd, value: 100 V}}
    - {{id: q1, type: ideal_switch, p: in, n: sw, control: pwm, initial: on}}
    - {{id: d1, type: diode, p: gnd, n: sw, initial: off}}
{inductor}
{capacitors}
    - {{id: load, type: resistor, p: out, n: gnd, value: 10 ohm}}

controls:
  pwm: {{type: pwm, frequency: 20 kHz, duty: 40 %, phase: 0 s}}

analysis:
  transient:
    start: 0 s
    stop: 0.5 ms
    output_step: 1 us
    solver: dopri5
    rtol: 1.0e-8
    atol: 1.0e-10

outputs:
  - v(out)
  - i(l1)
"""

L_IDEAL = "    - {id: l1, type: inductor, p: sw, n: out, value: 1 mH, initial_current: 3.4 A}"
C_IDEAL = "    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, initial_voltage: 40 V}"


def _write(tmp_path: Path, name: str, *, inductor: str = L_IDEAL, capacitors: str = C_IDEAL) -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(BUCK.format(inductor=inductor, capacitors=capacitors), encoding="utf-8")
    return path


def _run(path: Path, backend: str = "python") -> dict[str, np.ndarray]:
    result = HybridSimulator(load_model(path), backend=backend).simulate()
    return {name: np.asarray(values) for name, values in result.signals.items()}


# ---------------------------------------------------------------------------
# Verification: the shorthand IS the series resistor (code against code)
# ---------------------------------------------------------------------------

BACKENDS = ["python"] + (["rust"] if rust_backend.available() else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_esr_is_bit_identical_to_an_explicit_series_resistor(tmp_path: Path, backend: str) -> None:
    inline = _write(
        tmp_path,
        "inline",
        capacitors="    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, "
        'esr: "${esr_out}", initial_voltage: 40 V}',
    )
    explicit = _write(
        tmp_path,
        "explicit",
        capacitors="    - {id: c1, type: capacitor, p: mid, n: gnd, value: 100 uF, initial_voltage: 40 V}\n"
        '    - {id: resr, type: resistor, p: out, n: mid, value: "${esr_out}"}',
    )
    a, b = _run(inline, backend), _run(explicit, backend)
    for name in ("v(out)", "i(l1)"):
        assert np.array_equal(a[name], b[name]), name


@pytest.mark.parametrize("backend", BACKENDS)
def test_dcr_is_bit_identical_to_an_explicit_series_resistor(tmp_path: Path, backend: str) -> None:
    inline = _write(
        tmp_path,
        "inline",
        inductor="    - {id: l1, type: inductor, p: sw, n: out, value: 1 mH, dcr: 45 mohm, "
        "initial_current: 3.4 A}",
    )
    explicit = _write(
        tmp_path,
        "explicit",
        inductor="    - {id: l1, type: inductor, p: sw, n: lmid, value: 1 mH, initial_current: 3.4 A}\n"
        "    - {id: rdcr, type: resistor, p: lmid, n: out, value: 45 mohm}",
    )
    a, b = _run(inline, backend), _run(explicit, backend)
    for name in ("v(out)", "i(l1)"):
        assert np.array_equal(a[name], b[name]), name


def test_the_parasitic_changes_the_answer(tmp_path: Path) -> None:
    """Guard against an expansion that silently drops the field."""

    ideal = _run(_write(tmp_path, "ideal"))
    with_esr = _run(
        _write(
            tmp_path,
            "esr",
            capacitors="    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, "
            "esr: 200 mohm, initial_voltage: 40 V}",
        )
    )
    ripple_ideal = np.ptp(ideal["v(out)"][len(ideal["v(out)"]) // 2 :])
    ripple_esr = np.ptp(with_esr["v(out)"][len(with_esr["v(out)"]) // 2 :])
    assert ripple_esr > 3.0 * ripple_ideal


def test_expansion_is_visible_and_names_are_namespaced(tmp_path: Path) -> None:
    model = load_model(
        _write(
            tmp_path,
            "names",
            inductor="    - {id: l1, type: inductor, p: sw, n: out, value: 1 mH, dcr: 45 mohm, "
            "initial_current: 3.4 A}",
            capacitors="    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, "
            "esr: 20 mohm, initial_voltage: 40 V}",
        )
    )
    by_id = {component.id: component for component in model.components}
    assert (by_id["c1.esr"].type, by_id["c1.esr"].p, by_id["c1.esr"].n) == ("resistor", "out", "c1.core")
    assert (by_id["c1"].p, by_id["c1"].n, by_id["c1"].initial_state) == ("c1.core", "gnd", 40.0)
    assert (by_id["l1"].p, by_id["l1"].n) == ("sw", "l1.core")
    assert (by_id["l1.dcr"].p, by_id["l1.dcr"].n) == ("l1.core", "out")
    assert by_id["l1.dcr"].value == pytest.approx(0.045)


def test_a_model_without_parasitics_is_untouched(tmp_path: Path) -> None:
    model = load_model(_write(tmp_path, "plain"))
    assert [component.id for component in model.components] == ["vin", "q1", "d1", "l1", "c1", "load"]


# ---------------------------------------------------------------------------
# Capacitor banks: rejected without ESR (with a hint), accepted with it
# ---------------------------------------------------------------------------

BANK = (
    "    - {{id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, {e1}initial_voltage: 40 V}}\n"
    "    - {{id: c2, type: capacitor, p: out, n: gnd, value: 10 uF, {e2}initial_voltage: 40 V}}"
)


def test_a_bank_without_esr_fails_and_the_error_says_how_to_fix_it(tmp_path: Path) -> None:
    path = _write(tmp_path, "bank", capacitors=BANK.format(e1="", e2=""))
    with pytest.raises(TopologyError, match=r"Capacitors c1, c2 close a loop.*`esr`"):
        HybridSimulator(load_model(path)).simulate()


def test_a_bank_with_esr_runs(tmp_path: Path) -> None:
    path = _write(tmp_path, "bank", capacitors=BANK.format(e1="esr: 50 mohm, ", e2="esr: 3 mohm, "))
    signals = _run(path)
    assert np.all(np.isfinite(signals["v(out)"]))


def test_the_loop_hint_is_silent_on_every_shipped_example() -> None:
    examples = sorted(glob.glob(str(ROOT / "examples" / "*.yaml")))
    assert len(examples) > 30
    for example in examples:
        model = load_model(example)
        assert capacitor_loop_hint(model.components, model.ground) == "", example


def test_the_loop_hint_names_a_capacitor_across_a_voltage_source(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "cin",
        capacitors=C_IDEAL
        + "\n    - {id: cin, type: capacitor, p: in, n: gnd, value: 10 uF, initial_voltage: 100 V}",
    )
    model = load_model(path)
    assert "Capacitor cin closes" in capacitor_loop_hint(model.components, model.ground)


# Measured 2026-11-15: `dcr` on EVERY inductor of the shipped examples breaks
# exactly these (an inductor-only node: floating star, or inductors in series).
DCR_STAR_BREAKS = {
    "grid_following_converter", "inverter_signal_svpwm",
    "inverter_three_phase_floating_neutral", "inverter_three_phase_svpwm", "llc_resonant",
    # 2026-11-20: the dead-time copy of the LLC has the same tank star.
    "llc_resonant_zvs",
    "npc_three_phase_midpoint", "rectifier_6pulse_transformer",
    "traction_inverter_liquid_cooled",
}


def _with_dcr_everywhere(example: Path, tmp_path: Path) -> Path | None:
    import yaml

    from pear_sim.spec.model import _ModelSafeLoader

    document = yaml.load(example.read_text(encoding="utf-8"), Loader=_ModelSafeLoader)
    inductors = [
        c for c in document.get("circuit", {}).get("components", [])
        if isinstance(c, dict) and c.get("type") == "inductor"
    ]
    if not inductors:
        return None
    for inductor in inductors:
        inductor["dcr"] = "10 mohm"
    path = tmp_path / example.name
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_the_dcr_star_hint_names_exactly_the_models_dcr_breaks(tmp_path: Path) -> None:
    """Discriminates both ways: rings on every example that dcr breaks, silent
    on every example with inductors that still runs with dcr (and on the
    internal ``l.core`` node every dcr creates)."""

    named, silent = set(), set()
    for example in sorted((ROOT / "examples").glob("*.yaml")):
        path = _with_dcr_everywhere(example, tmp_path)
        if path is None:
            continue
        model = load_model(path)
        hint = dcr_star_hint(model.components, model.ground, model.passive_parasitics)
        (named if hint else silent).add(example.stem)
    assert named == DCR_STAR_BREAKS
    assert len(silent) >= 15
    # Without any dcr the hint has nothing to say, even on a floating star.
    model = load_model(ROOT / "examples" / "inverter_three_phase_floating_neutral.yaml")
    assert dcr_star_hint(model.components, model.ground, model.passive_parasitics) == ""


def test_the_topology_error_of_a_dcr_star_says_how_to_fix_it(tmp_path: Path) -> None:
    path = _with_dcr_everywhere(
        ROOT / "examples" / "inverter_three_phase_floating_neutral.yaml", tmp_path
    )
    with pytest.raises(
        TopologyError, match=r"la, lb, lc at node 'star'.*explicit\s+`resistor`"
    ):
        HybridSimulator(load_model(path)).simulate()


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("capacitor", "message"),
    [
        ("esr: 0 ohm", r"c1\.esr must be a positive, finite resistance; omit the field"),
        ("esr: -1 mohm", r"c1\.esr must be a positive"),
        ("esr: 5 uF", r"c1\.esr"),
        ('esr: "${missing}"', r"missing"),
    ],
)
def test_bad_esr_values_are_rejected(tmp_path: Path, capacitor: str, message: str) -> None:
    path = _write(
        tmp_path,
        "bad",
        capacitors="    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, "
        f"{capacitor}, initial_voltage: 40 V}}",
    )
    with pytest.raises(ModelError, match=message):
        load_model(path)


def test_esr_on_a_resistor_and_dcr_on_a_capacitor_are_rejected(tmp_path: Path) -> None:
    bad_esr = _write(
        tmp_path,
        "esr_on_r",
        capacitors=C_IDEAL + "\n    - {id: r9, type: resistor, p: out, n: gnd, value: 1 kohm, esr: 1 mohm}",
    )
    with pytest.raises(ModelError, match=r"r9\.esr: only valid for capacitor"):
        load_model(bad_esr)
    bad_dcr = _write(
        tmp_path,
        "dcr_on_c",
        capacitors="    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, dcr: 1 mohm, "
        "initial_voltage: 40 V}",
    )
    with pytest.raises(ModelError, match=r"c1\.dcr: only valid for inductor, sat_inductor"):
        load_model(bad_dcr)


def test_a_user_node_with_the_reserved_name_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "clash",
        capacitors="    - {id: c1, type: capacitor, p: out, n: gnd, value: 100 uF, esr: 1 mohm, "
        "initial_voltage: 40 V}\n"
        "    - {id: r9, type: resistor, p: out, n: c1.core, value: 1 kohm}",
    )
    with pytest.raises(ModelError, match=r"node name 'c1\.core' is reserved"):
        load_model(path)
