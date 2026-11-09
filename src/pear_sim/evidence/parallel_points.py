# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Run independent point workers concurrently and hand results back in order.

Every batch producer (parameter sweep, tolerance corners, perturbation grid)
runs each point as its own killable child process. Those children share
nothing, so several can run at once. This module only overlaps the waiting:

- Each task is a callable that launches one child and blocks on it. Threads
  are enough, because a thread waiting on a child releases the GIL.
- Results come back in declaration order, whatever order the children finish
  in. The consumer then reads, verifies and records each point exactly as the
  serial loop did, so every retained artifact is identical to a serial run.
- ``jobs=1`` calls the tasks inline, one after another: the serial path.

A task's exception is returned as that point's outcome, never raised here, so
one failed point cannot stop the others (the producers already retain
point-local failures and continue).
"""

from __future__ import annotations

import os
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Generic, Iterator, Sequence, TypeVar

T = TypeVar("T")

MAX_JOBS = 64


@dataclass(frozen=True)
class Outcome(Generic[T]):
    """One task's result, or the exception it raised."""

    value: T | None = None
    error: Exception | None = None

    def unwrap(self) -> T:
        if self.error is not None:
            raise self.error
        return self.value  # type: ignore[return-value]


def default_jobs() -> int:
    """All cores but one, so the machine stays responsive."""

    return max(1, min(MAX_JOBS, (os.cpu_count() or 1) - 1))


def resolve_jobs(jobs: int | None) -> int:
    """Validate a requested worker count; ``None`` means :func:`default_jobs`."""

    if jobs is None:
        return default_jobs()
    if isinstance(jobs, bool) or not isinstance(jobs, int) or not 1 <= jobs <= MAX_JOBS:
        raise ValueError(f"jobs must be an integer in 1..{MAX_JOBS}, got {jobs!r}")
    return jobs


def _call(task: Callable[[], T]) -> Outcome[T]:
    try:
        return Outcome(value=task())
    except Exception as exc:  # retained as this point's outcome
        return Outcome(error=exc)


def run_ordered(
    tasks: Sequence[Callable[[], T]],
    *,
    jobs: int,
) -> Iterator[Outcome[T]]:
    """Yield one :class:`Outcome` per task, in task order.

    Up to ``jobs`` tasks run at the same time. The consumer may process an
    early outcome while later tasks are still running. Closing the iterator
    early cancels tasks that have not started yet.
    """

    if jobs == 1 or len(tasks) <= 1:
        for task in tasks:
            yield _call(task)
        return
    pool = ThreadPoolExecutor(
        max_workers=min(jobs, len(tasks)),
        thread_name_prefix="pear-sim-point",
    )
    futures: list[Future[Outcome[T]]] = []
    try:
        futures = [pool.submit(_call, task) for task in tasks]
        for future in futures:
            yield future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
