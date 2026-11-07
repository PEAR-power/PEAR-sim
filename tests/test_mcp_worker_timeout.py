"""A timed-out MCP computation stops its whole process tree.

Sweep tools run their points in grandchild processes, several at once. If a
timeout killed only the worker, those points would keep running as orphans.
The call must also return promptly: the orphans still hold the worker's output
pipes, so a version that kills the worker and then reads its output to the end
would block until they finished.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pytest

from pear_sim.mcp import tools

pytestmark = pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX")

_TREE = """
import subprocess, sys, time
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
open(sys.argv[1], "w").write(str(child.pid))
time.sleep(60)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_timeout_stops_the_worker_and_its_grandchildren(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pid_file = tmp_path / "grandchild.pid"
    monkeypatch.setattr(
        tools, "_worker_command", lambda: [sys.executable, "-c", _TREE, str(pid_file)]
    )
    started = time.monotonic()
    with pytest.raises(tools.ToolInputError, match="exceeded timeout_s"):
        tools._run_worker({}, timeout_s=2.0)
    assert time.monotonic() - started < 15.0
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 5.0
    while _alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _alive(grandchild)
