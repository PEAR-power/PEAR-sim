"""Long-lived point workers, so a batch imports Python and PEARSim once per worker.

A point used to cost a fresh interpreter: start Python, import NumPy, SciPy and
PEARSim, simulate, exit. For small models the start-up is most of the work.
A pool keeps up to ``jobs`` worker processes alive and hands each one point at
a time over a line-based JSON protocol (see ``parameter_sweep_worker._serve``).

What does not change:

- Each point still runs in a child process the parent can kill. A point that
  overruns its budget kills its worker; the next point gets a fresh one.
- The parent receives the same reply, exit code and captured stderr it got from
  a one-shot worker, and checks them the same way (identity, empty stderr,
  bundle present). ``tests/test_worker_pool.py`` holds the retained bundles to
  byte equality with the one-shot path.
- A worker that dies takes only the point it was running with it.

Workers are recycled after ``recycle_after`` points to bound slow growth of a
long-lived process (memory, open files).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Mapping, Sequence, TypeVar

from pear_sim.evidence.parallel_points import Outcome, run_ordered

WORKER_MODULE = "pear_sim.evidence.parameter_sweep_worker"
START_TIMEOUT_S = 60.0
RECYCLE_AFTER = 200
# Windows can hold a killed or exited worker's files open for a moment after
# the process is gone (handle teardown, antivirus scans).
UNLINK_RETRY_S = 5.0

T = TypeVar("T")


class WorkerTimeout(Exception):
    """The point did not answer within its budget; its worker was killed."""


@dataclass(frozen=True)
class WorkerReply:
    """What a one-shot worker run returns: exit code, stdout, stderr."""

    returncode: int
    stdout: bytes
    stderr: bytes


def _remove(path: Path) -> None:
    """Delete ``path``, waiting out a Windows sharing violation.

    Called right after a worker was killed or exited: on Windows the file can
    still be locked for a moment, and ``unlink`` would raise PermissionError
    instead of the point's own failure.
    """

    deadline = time.monotonic() + UNLINK_RETRY_S
    while True:
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.pop("PYTHONHOME", None)
    environment.pop("PYTHONPATH", None)
    return environment


@dataclass(eq=False)  # identity, so a worker can live in a set
class _Worker:
    process: subprocess.Popen[bytes]
    lines: "queue.Queue[bytes | None]" = field(default_factory=queue.Queue)
    served: int = 0

    def start_reader(self) -> None:
        def pump() -> None:
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.lines.put(line)
            self.lines.put(None)

        threading.Thread(target=pump, name="pear-sim-worker-reader", daemon=True).start()

    def next_line(self, timeout_s: float) -> bytes | None:
        return self.lines.get(timeout=max(0.001, timeout_s))

    def kill(self) -> None:
        self.process.kill()
        self.process.wait()

    def close(self) -> None:
        try:
            assert self.process.stdin is not None
            self.process.stdin.close()
            self.process.wait(timeout=5.0)
        except (OSError, subprocess.TimeoutExpired):
            self.kill()


class PointWorkerPool:
    """Thread-safe pool of long-lived point workers; use as a context manager."""

    def __init__(
        self,
        *,
        recycle_after: int = RECYCLE_AFTER,
        command: Sequence[str] | None = None,
    ) -> None:
        if recycle_after < 1:
            raise ValueError("recycle_after must be at least 1")
        self._recycle_after = recycle_after
        # ``command`` replaces the worker command line (tests only).
        self._command = list(command) if command is not None else [
            sys.executable, "-I", "-m", WORKER_MODULE, "--serve"
        ]
        self._idle: list[_Worker] = []
        self._live: set[_Worker] = set()
        self._lock = threading.Lock()
        self._closed = False
        self.started = 0  # workers spawned, for tests and diagnostics

    def __enter__(self) -> "PointWorkerPool":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            workers = list(self._live)
            self._idle.clear()
            self._live.clear()
        for worker in workers:
            worker.close()

    def _spawn(self) -> _Worker:
        process = subprocess.Popen(
            self._command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=tempfile.gettempdir(),
            env=worker_environment(),
        )
        worker = _Worker(process)
        worker.start_reader()
        try:
            line = worker.next_line(START_TIMEOUT_S)
        except queue.Empty:
            worker.kill()
            raise RuntimeError(
                f"point worker did not start within {START_TIMEOUT_S:.0f} seconds"
            ) from None
        if line is None or json.loads(line) != {"status": "ready"}:
            worker.kill()
            raise RuntimeError(
                f"point worker failed to start (exit status {process.poll()})"
            )
        with self._lock:
            self.started += 1
            self._live.add(worker)
        return worker

    def _acquire(self) -> _Worker:
        with self._lock:
            if self._closed:
                raise RuntimeError("the point worker pool is closed")
            if self._idle:
                return self._idle.pop()
        return self._spawn()

    def _release(self, worker: _Worker) -> None:
        with self._lock:
            keep = (
                not self._closed
                and worker.served < self._recycle_after
                and worker.process.poll() is None
            )
            if keep:
                self._idle.append(worker)
                return
            self._live.discard(worker)
        worker.close()

    def _discard(self, worker: _Worker) -> None:
        with self._lock:
            self._live.discard(worker)
        worker.kill()

    def run(
        self,
        model_path: Path,
        overrides: Mapping[str, float],
        out: Path,
        *,
        cwd: Path,
        timeout_s: float,
        lean: bool,
    ) -> WorkerReply:
        """Run one point; raise :class:`WorkerTimeout` past ``timeout_s``."""

        worker = self._acquire()
        stderr_path = cwd / ".worker-stderr"
        request = {
            "model": str(model_path),
            "parameters": list(overrides),
            "values_si": [float(value) for value in overrides.values()],
            "out": str(out),
            "lean": lean,
            "cwd": str(cwd),
            "stderr_path": str(stderr_path),
        }
        try:
            assert worker.process.stdin is not None
            worker.process.stdin.write(json.dumps(request).encode("ascii") + b"\n")
            worker.process.stdin.flush()
            line = worker.next_line(timeout_s)
        except queue.Empty:
            self._discard(worker)
            _remove(stderr_path)
            raise WorkerTimeout from None
        except OSError:  # the worker died before reading the request
            line = None
        try:
            captured = stderr_path.read_bytes()
        except FileNotFoundError:
            captured = b""
        _remove(stderr_path)
        if line is None:
            self._discard(worker)
            return WorkerReply(
                returncode=worker.process.returncode or 1,
                stdout=b"",
                stderr=captured or b"point worker exited before answering",
            )
        answer = json.loads(line)
        worker.served += 1
        self._release(worker)
        return WorkerReply(
            returncode=int(answer["exit"]),
            stdout=answer["reply"].encode("utf-8") + b"\n",
            stderr=captured,
        )


def run_pooled(
    make_tasks: Callable[[PointWorkerPool], Sequence[Callable[[], T]]],
    *,
    jobs: int,
) -> Iterator[Outcome[T]]:
    """:func:`run_ordered` with a pool that lives exactly as long as the run.

    ``make_tasks`` receives the pool and returns the point tasks. The pool
    closes when the outcomes are exhausted or the iterator is closed.
    """

    with PointWorkerPool() as pool:
        yield from run_ordered(make_tasks(pool), jobs=jobs)
