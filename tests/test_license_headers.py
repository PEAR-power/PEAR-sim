# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Every tracked Python and Rust source file carries the SPDX license header."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPDX = "SPDX-License-Identifier: Apache-2.0"
COPYRIGHT = "Copyright 2026 PEAR Power and the PEARSim contributors."


def _has_header(text: str) -> bool:
    head = text.splitlines()[:3]
    return any(SPDX in line for line in head) and any(COPYRIGHT in line for line in head)


def _tracked_sources() -> list[Path]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "*.py", "*.rs"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / name for name in out.split()]


def test_every_source_file_has_the_spdx_header() -> None:
    files = _tracked_sources()
    assert files, "git ls-files found no sources"
    missing = [
        str(path.relative_to(ROOT))
        for path in files
        if not _has_header(path.read_text(encoding="utf-8"))
    ]
    assert not missing, f"missing SPDX header: {missing}"


def test_the_check_rejects_a_file_without_the_header() -> None:
    assert not _has_header('"""Docstring only."""\n')
    assert not _has_header(f"# {SPDX}\n")
    assert _has_header(f"#!/usr/bin/env python3\n# {SPDX}\n# {COPYRIGHT}\n")
