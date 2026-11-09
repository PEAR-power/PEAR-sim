# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from pear_sim.benchmarks.affine import (
    affine_map,
    propagate,
    propagate_with_coordinate_integral,
)


def test_affine_map_matches_scalar_closed_form() -> None:
    decay = 2.0
    forcing = 6.0
    duration = 0.4
    transition, affine = affine_map(
        np.asarray([[-decay]]),
        np.asarray([forcing]),
        duration,
    )
    expected_transition = math.exp(-decay * duration)
    expected_affine = forcing / decay * (1.0 - expected_transition)
    assert transition == pytest.approx(np.asarray([[expected_transition]]), rel=1e-14)
    assert affine == pytest.approx(np.asarray([expected_affine]), rel=1e-14)


def test_propagation_and_coordinate_integral_match_constant_acceleration() -> None:
    matrix = np.zeros((2, 2), dtype=float)
    offset = np.asarray([2.0, -3.0])
    state = np.asarray([5.0, 7.0])
    duration = 0.25
    expected = state + offset * duration

    assert propagate(matrix, offset, state, duration) == pytest.approx(expected, abs=1e-15)
    propagated, integral = propagate_with_coordinate_integral(
        matrix,
        offset,
        state,
        duration,
        coordinate_index=1,
    )
    assert propagated == pytest.approx(expected, abs=1e-15)
    assert integral == pytest.approx(
        state[1] * duration + 0.5 * offset[1] * duration * duration,
        abs=1e-15,
    )


def test_affine_primitives_reject_incompatible_shapes_and_coordinates() -> None:
    with pytest.raises(ValueError, match="square"):
        affine_map(np.zeros((2, 3)), np.zeros(2), 1.0)
    with pytest.raises(ValueError, match="offset"):
        affine_map(np.zeros((2, 2)), np.zeros(3), 1.0)
    with pytest.raises(ValueError, match="state"):
        propagate(np.zeros((2, 2)), np.zeros(2), np.zeros(3), 1.0)
    with pytest.raises(ValueError, match="coordinate"):
        propagate_with_coordinate_integral(
            np.zeros((2, 2)),
            np.zeros(2),
            np.zeros(2),
            1.0,
            coordinate_index=2,
        )


def test_affine_oracle_module_has_no_production_kernel_import() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "pear_sim" / "benchmarks" / "affine.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(name.startswith("pear_sim.circuit") for name in imported)
    assert not any(name.startswith("pear_sim.solver") for name in imported)
    assert not any(name.startswith("pear_sim.mode_coverage") for name in imported)
