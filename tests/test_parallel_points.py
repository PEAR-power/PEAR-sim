# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Concurrent point execution returns outcomes in declaration order.

The batch producers rely on two properties: outcomes come back in task order
whatever order the tasks finish in, and a failing task becomes that point's
outcome instead of stopping the batch. The sweep-level test then checks that
a whole parameter sweep retains the same files with one job and with many.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from pear_sim.evidence.parallel_points import (
    MAX_JOBS,
    default_jobs,
    resolve_jobs,
    run_ordered,
)
from pear_sim.evidence.parameter_sweep import ParameterSweepError, run_parameter_sweep

ROOT = Path(__file__).resolve().parents[1]


def _sleeper(value: int, delay: float):
    def task() -> int:
        time.sleep(delay)
        return value

    return task


def test_outcomes_follow_task_order_not_finish_order() -> None:
    # The first task finishes last; order must still be 0, 1, 2, 3.
    tasks = [_sleeper(index, delay) for index, delay in enumerate((0.3, 0.0, 0.1, 0.0))]
    values = [outcome.unwrap() for outcome in run_ordered(tasks, jobs=4)]
    assert values == [0, 1, 2, 3]


def test_tasks_overlap_when_jobs_allow_it() -> None:
    running = 0
    peak = 0
    lock = threading.Lock()

    def task() -> None:
        nonlocal running, peak
        with lock:
            running += 1
            peak = max(peak, running)
        time.sleep(0.05)
        with lock:
            running -= 1

    list(run_ordered([task] * 6, jobs=3))
    assert peak == 3
    peak = 0
    list(run_ordered([task] * 6, jobs=1))
    assert peak == 1


def test_a_failing_task_is_its_own_outcome() -> None:
    def boom() -> int:
        raise RuntimeError("point 1 failed")

    outcomes = list(run_ordered([_sleeper(0, 0.0), boom, _sleeper(2, 0.0)], jobs=3))
    assert outcomes[0].unwrap() == 0
    assert isinstance(outcomes[1].error, RuntimeError)
    with pytest.raises(RuntimeError, match="point 1 failed"):
        outcomes[1].unwrap()
    assert outcomes[2].unwrap() == 2


@pytest.mark.parametrize("bad", [0, -1, MAX_JOBS + 1, 2.0, True, "4"])
def test_invalid_job_counts_are_refused(bad) -> None:
    with pytest.raises(ValueError, match="jobs must be an integer"):
        resolve_jobs(bad)


def test_default_jobs_leaves_one_core_free() -> None:
    assert resolve_jobs(None) == default_jobs()
    assert 1 <= default_jobs() <= MAX_JOBS


def _retained(root: Path) -> dict[str, object]:
    # Hashes and wall times differ between any two runs; everything else
    # (statuses, values, file set) must be the same.
    def strip(value):
        if isinstance(value, dict):
            return {
                key: strip(item)
                for key, item in value.items()
                if "sha256" not in key and "wall_time" not in key and key != "bytes"
            }
        if isinstance(value, list):
            return [strip(item) for item in value]
        return value

    files: dict[str, object] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix == ".html":
            continue
        name = path.relative_to(root).as_posix()
        files[name] = strip(json.loads(path.read_text())) if path.suffix == ".json" else None
    return files


def test_parameter_sweep_retains_the_same_files_with_one_job_or_many(tmp_path: Path) -> None:
    model = ROOT / "examples" / "buck.yaml"
    request = ROOT / "examples" / "buck_parameter_sweep.json"
    serial = run_parameter_sweep(model, request, tmp_path / "serial", jobs=1)
    parallel = run_parameter_sweep(model, request, tmp_path / "parallel", jobs=3)
    assert serial.execution_status == parallel.execution_status == "pass"
    assert _retained(tmp_path / "serial") == _retained(tmp_path / "parallel")


def test_parameter_sweep_refuses_an_invalid_job_count(tmp_path: Path) -> None:
    with pytest.raises(ParameterSweepError, match="jobs must be an integer"):
        run_parameter_sweep(
            ROOT / "examples" / "buck.yaml",
            ROOT / "examples" / "buck_parameter_sweep.json",
            tmp_path / "out",
            jobs=0,
        )
    assert not (tmp_path / "out").exists()
