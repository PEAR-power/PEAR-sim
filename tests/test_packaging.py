# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""The launcher package must always install exactly this version of pear-sim."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _project(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))["project"]


def test_launcher_version_and_pin_follow_the_core_package() -> None:
    core = _project(ROOT / "pyproject.toml")
    launcher = _project(ROOT / "packaging" / "pear-sim-mcp" / "pyproject.toml")
    assert launcher["version"] == core["version"]
    assert launcher["dependencies"] == [f"pear-sim[mcp]=={core['version']}"]
    module, _, function = launcher["scripts"]["pear-sim-mcp"].partition(":")
    assert module == "pear_sim.mcp.server" and function == "main"


def test_registry_entry_names_the_launcher_and_the_readme_claims_it() -> None:
    entry = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    package = entry["packages"][0]
    assert package["registryType"] == "pypi" and package["identifier"] == "pear-sim-mcp"
    readme = (ROOT / "packaging" / "pear-sim-mcp" / "README.md").read_text(encoding="utf-8")
    assert f"mcp-name: {entry['name']}" in readme


def test_registry_entry_fits_the_registry_limits() -> None:
    # The MCP registry rejects a description over 100 characters (HTTP 422 at
    # publish time, after PyPI is already out). Catch it here instead.
    entry = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    assert len(entry["description"]) <= 100
    assert entry["version"] == _project(ROOT / "pyproject.toml")["version"]
