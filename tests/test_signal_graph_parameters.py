# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Unit-less parameters in signal block values.

A block value written as ``"${kp}"`` must mean EXACTLY the number the
parameter holds (bit-identical run), must follow overrides, and must refuse a
parameter that carries a unit instead of silently dropping it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pear_sim.analysis.closed_loop import solve_closed_loop_steady_state
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.model import load_model

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "buck_closed_loop_floquet.yaml"


def _literal(tmp_path: Path, kp: str = "0.002", ki: str = "30.0") -> Path:
    source = EXAMPLE.read_text(encoding="utf-8")
    assert '"${kp}"' in source and '"${ki}"' in source
    path = tmp_path / f"literal_{kp}_{ki}.yaml"
    path.write_text(
        source.replace('"${kp}"', kp).replace('"${ki}"', ki), encoding="utf-8"
    )
    return path


def test_a_parameter_runs_bit_identical_to_the_literal(tmp_path: Path) -> None:
    by_parameter = HybridSimulator(load_model(EXAMPLE)).simulate()
    by_literal = HybridSimulator(load_model(_literal(tmp_path))).simulate()
    for signal in ("v(out)", "i(l1)"):
        assert np.array_equal(by_parameter.signals[signal], by_literal.signals[signal]), signal


def test_overrides_reach_the_controller(tmp_path: Path) -> None:
    """kp=0.08/ki=150 is the unstable design: the override must give the
    same multipliers as writing those numbers into the blocks."""

    start = {"signal:int_prev": 0.5 / 150.0}
    overridden = solve_closed_loop_steady_state(
        load_model(EXAMPLE, {"kp": "0.08", "ki": "150"}), initial_state=start
    )
    written = solve_closed_loop_steady_state(
        load_model(_literal(tmp_path, "0.08", "150.0")), initial_state=start
    )
    assert overridden.stability == written.stability == "unstable"
    assert overridden.max_multiplier_magnitude == written.max_multiplier_magnitude
    nominal = solve_closed_loop_steady_state(load_model(EXAMPLE), initial_state=None)
    assert nominal.stability == "stable"


def test_a_parameter_with_a_unit_is_refused_by_name(tmp_path: Path) -> None:
    source = EXAMPLE.read_text(encoding="utf-8").replace("  kp: 0.002 ", "  kp: 2 mV ")
    path = tmp_path / "unit.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ModelError, match=r"kp_term\.gain: \$\{kp\} is not a plain number"):
        load_model(path)


def test_an_unknown_parameter_is_refused_by_name(tmp_path: Path) -> None:
    source = EXAMPLE.read_text(encoding="utf-8").replace('"${kp}"', '"${kq}"')
    path = tmp_path / "unknown.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ModelError, match=r"unknown parameter 'kq'"):
        load_model(path)


def test_time_fields_still_take_parameters_with_units(tmp_path: Path) -> None:
    """Substitution leaves references with a unit alone, so a time_source
    start written as ${t_step} (a time parameter) still resolves."""

    source = EXAMPLE.read_text(encoding="utf-8").replace(
        "  ki: 30.0 ", "  ki: 30.0 \n  t_step: 1 ms\n"
    ).replace(
        "    vref: {type: constant, value: 12.0, unit: V}",
        "    vref: {type: time_source, shape: step, initial: 12.0, final: 12.5, "
        'start: "${t_step}", unit: V}',
    )
    path = tmp_path / "time.yaml"
    path.write_text(source, encoding="utf-8")
    graph = load_model(path).signals
    step = graph.by_id["vref"]
    assert step.params["start"] == pytest.approx(1e-3)
