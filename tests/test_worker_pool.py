"""Long-lived point workers must be indistinguishable from one-shot workers.

The pool exists only to skip interpreter start-up per point. Everything the
parent checks or retains has to come out the same: the run bundle bytes, the
failure reply, captured stderr (per point, never leaking into the next), the
timeout, and a worker that dies mid-point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pear_sim.evidence.parameter_sweep import ParameterSweepError, _run_point_worker
from pear_sim.evidence.worker_pool import PointWorkerPool, WorkerTimeout

ROOT = Path(__file__).resolve().parents[1]
BUCK = ROOT / "examples" / "buck.yaml"
VOLATILE = {"wall_time_s"}


def _strip_volatile(value):
    if isinstance(value, dict):
        return {k: _strip_volatile(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


def _bundle(run_dir: Path) -> dict[str, object]:
    """Every retained file: JSON with volatile timings removed, others as bytes."""

    files: dict[str, object] = {}
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        raw = path.read_bytes()
        if path.suffix == ".json":
            files[relative] = _strip_volatile(json.loads(raw))
        else:
            files[relative] = raw
    return files


def _manifest_free(bundle: dict[str, object]) -> dict[str, object]:
    # The manifest and run.json carry hashes of files that hold wall times;
    # their non-hash content is compared through the stripped JSON above.
    return {
        name: content
        for name, content in bundle.items()
        if name not in {"artifact-manifest.json"}
    }


@pytest.mark.parametrize("lean", [True, False])
def test_pooled_bundles_match_one_shot_bundles(tmp_path: Path, lean: bool) -> None:
    designs = [{"duty": 0.3}, {"duty": 0.45}, {"duty": 0.6}]
    one_shot = []
    for index, design in enumerate(designs):
        point_dir = tmp_path / "fresh" / str(index)
        point_dir.mkdir(parents=True)
        one_shot.append(
            _run_point_worker(BUCK, design, point_dir, timeout_s=60.0, lean=lean)
        )
    pooled = []
    with PointWorkerPool() as pool:
        for index, design in enumerate(designs):
            point_dir = tmp_path / "pooled" / str(index)
            point_dir.mkdir(parents=True)
            pooled.append(
                _run_point_worker(
                    BUCK, design, point_dir, timeout_s=60.0, lean=lean, pool=pool
                )
            )
        assert pool.started == 1  # one interpreter served all three points
    for fresh_dir, pooled_dir in zip(one_shot, pooled, strict=True):
        fresh = _bundle(fresh_dir)
        reused = _bundle(pooled_dir)
        assert sorted(fresh) == sorted(reused)
        assert _manifest_free(fresh) == _manifest_free(reused)
        assert (fresh_dir / "metrics.json").read_bytes() == (
            pooled_dir / "metrics.json"
        ).read_bytes()
        assert not (pooled_dir.parent / ".worker-stderr").exists()


def test_failed_point_keeps_the_worker_and_reports_like_one_shot(tmp_path: Path) -> None:
    messages = []
    for pool in (None, PointWorkerPool()):
        point_dir = tmp_path / ("pooled" if pool else "fresh")
        point_dir.mkdir()
        with pytest.raises(ParameterSweepError) as caught:
            _run_point_worker(
                BUCK, {"no_such_parameter": 1.0}, point_dir, timeout_s=60.0, pool=pool
            )
        messages.append(str(caught.value))
        assert not (point_dir / ".run-worker").exists()
        if pool is not None:
            # The same worker answers the next point.
            ok_dir = tmp_path / "after"
            ok_dir.mkdir()
            _run_point_worker(BUCK, {"duty": 0.4}, ok_dir, timeout_s=60.0, pool=pool)
            assert pool.started == 1
            pool.close()
    assert messages[0] == messages[1]


_FAKE_WORKER = r"""
import os, sys, warnings
import pear_sim.evidence.parameter_sweep_worker as worker

def fake(model, parameters, values, out, lean):
    mode = parameters[0]
    if mode == "warn":
        warnings.warn("same place every time", RuntimeWarning)
    elif mode == "print":
        print("stray stdout")
    elif mode == "die":
        os._exit(7)
    elif mode == "hang":
        import time
        time.sleep(60)
    return 0, {"status": "pass"}

worker._simulate_point = fake
raise SystemExit(worker._serve())
"""


def _fake_pool(tmp_path: Path, **options) -> PointWorkerPool:
    script = tmp_path / "fake_worker.py"
    script.write_text(_FAKE_WORKER)
    return PointWorkerPool(command=[sys.executable, str(script)], **options)


def _point(tmp_path: Path, name: str) -> Path:
    point_dir = tmp_path / name
    point_dir.mkdir()
    return point_dir


def test_each_point_gets_its_own_stderr_and_warnings_repeat(tmp_path: Path) -> None:
    with _fake_pool(tmp_path) as pool:
        replies = [
            pool.run(BUCK, {mode: 1.0}, tmp_path / f"out{index}", cwd=_point(tmp_path, f"p{index}"),
                     timeout_s=30.0, lean=True)
            for index, mode in enumerate(["warn", "quiet", "warn", "print", "quiet"])
        ]
        assert pool.started == 1
    stderr = [reply.stderr.decode() for reply in replies]
    # A one-shot worker shows the warning every time; so must a reused one.
    assert "same place every time" in stderr[0]
    assert "same place every time" in stderr[2]
    # Nothing carries over into a quiet point.
    assert stderr[1] == "" and stderr[4] == ""
    # Stray stdout is retained as that point's output, not read as a reply.
    assert "stray stdout" in stderr[3]
    assert all(json.loads(reply.stdout) == {"status": "pass"} for reply in replies)


def test_a_dying_worker_fails_only_its_point(tmp_path: Path) -> None:
    with _fake_pool(tmp_path) as pool:
        died = pool.run(BUCK, {"die": 1.0}, tmp_path / "o1", cwd=_point(tmp_path, "p1"),
                        timeout_s=30.0, lean=True)
        after = pool.run(BUCK, {"quiet": 1.0}, tmp_path / "o2", cwd=_point(tmp_path, "p2"),
                         timeout_s=30.0, lean=True)
        assert pool.started == 2
    assert died.returncode == 7
    assert after.returncode == 0


def test_timeout_kills_the_worker_and_the_next_point_gets_a_fresh_one(tmp_path: Path) -> None:
    with _fake_pool(tmp_path) as pool:
        with pytest.raises(WorkerTimeout):
            pool.run(BUCK, {"hang": 1.0}, tmp_path / "o1", cwd=_point(tmp_path, "p1"),
                     timeout_s=0.5, lean=True)
        after = pool.run(BUCK, {"quiet": 1.0}, tmp_path / "o2", cwd=_point(tmp_path, "p2"),
                         timeout_s=30.0, lean=True)
        assert pool.started == 2
    assert after.returncode == 0


def test_timeout_message_matches_one_shot(tmp_path: Path) -> None:
    with _fake_pool(tmp_path) as pool:
        with pytest.raises(ParameterSweepError, match="worker budget"):
            _run_point_worker(BUCK, {"hang": 1.0}, _point(tmp_path, "p"), timeout_s=0.5, pool=pool)


def test_workers_are_recycled(tmp_path: Path) -> None:
    with _fake_pool(tmp_path, recycle_after=2) as pool:
        for index in range(5):
            pool.run(BUCK, {"quiet": 1.0}, tmp_path / f"o{index}", cwd=_point(tmp_path, f"p{index}"),
                     timeout_s=30.0, lean=True)
        assert pool.started == 3


def test_closed_pool_refuses_work(tmp_path: Path) -> None:
    pool = _fake_pool(tmp_path)
    pool.close()
    with pytest.raises(RuntimeError, match="closed"):
        pool.run(BUCK, {"quiet": 1.0}, tmp_path / "o", cwd=_point(tmp_path, "p"),
                 timeout_s=30.0, lean=True)


def test_a_briefly_locked_stderr_file_does_not_mask_the_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Windows keeps a killed worker's stderr file locked for a moment; the
    # point must still fail with the budget message, not PermissionError.
    real_unlink = Path.unlink
    refusals = {"left": 3}

    def locked_unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name == ".worker-stderr" and refusals["left"]:
            refusals["left"] -= 1
            raise PermissionError(32, "being used by another process")
        real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", locked_unlink)
    point_dir = _point(tmp_path, "p")
    with _fake_pool(tmp_path) as pool:
        with pytest.raises(ParameterSweepError, match="worker budget"):
            _run_point_worker(BUCK, {"hang": 1.0}, point_dir, timeout_s=0.5, pool=pool)
    assert refusals["left"] == 0
    assert not (point_dir / ".worker-stderr").exists()
