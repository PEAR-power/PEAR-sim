# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""The committed validation results are current, and the claims made from them hold.

validation/switching_transitions/README.md and the capability registry state what
the comparison with published measurements supports. This test recomputes the
validation part (fast; the sensitivity sweep is not re-run here) and fails when
the committed results.json is stale or when a stated claim stops being true —
the check sits where a wrong sentence would do harm, not in the prose.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).parents[1] / "validation" / "switching_transitions"


def _load_script():
    spec = importlib.util.spec_from_file_location("st_validate", HERE / "validate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def fresh():
    module = _load_script()
    return module._jsonable(module.compute(include_sweep=False))["validation"]


@pytest.fixture(scope="module")
def committed():
    return json.loads((HERE / "results.json").read_text(encoding="utf-8"))


def _assert_close(a, b, path="validation"):
    if isinstance(a, dict):
        assert set(a) == set(b), f"{path}: keys differ"
        for key in a:
            _assert_close(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: lengths differ"
        for index, (x, y) in enumerate(zip(a, b)):
            _assert_close(x, y, f"{path}[{index}]")
    elif isinstance(a, float) and isinstance(b, (int, float)) and not isinstance(b, bool):
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-15), f"{path}: {a} != {b}"
    else:
        assert a == b, f"{path}: {a!r} != {b!r}"


def test_committed_results_are_current(fresh, committed):
    _assert_close(fresh, committed["validation"])


def test_stated_zvs_boundary_is_reproduced(fresh):
    assert all(fresh["zvs"]["zvs_boundary_agrees"].values())


def test_measured_fall_times_lie_in_the_datasheet_band(fresh):
    for row in fresh["zvs"]["rows"]:
        measured = row.get("measured_fall_time_s")
        if measured is None:
            continue
        low = row["typ_plot_window"]["minimum_dead_time_s"]
        high = row["max_plot_window"]["minimum_dead_time_s"]
        precision = 5e-9  # plot reading
        assert low - precision <= measured <= high + precision, row


def test_measured_overshoot_lies_between_the_two_estimates(fresh):
    assert all(row["measured_within_bracket"] for row in fresh["overshoot"]["rows"])

