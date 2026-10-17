"""``pear-sim doctor``: runtime, backend and platform diagnosis.

Reports facts an agent or a human needs before trusting results on this
machine: interpreter and dependency versions, package integrity (schemas and
the validation ledger present and loadable), oracle backends (ngspice
binary), a floating-point determinism probe, and the
portable limits the kernel enforces. Read-only: the doctor never mutates
state and never phones home.
"""

from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

from pear_sim import __version__

__all__ = ["run_doctor"]

DOCTOR_SCHEMA = "pear-sim/doctor/v0.1"


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for module_name in ("numpy", "scipy", "yaml", "jsonschema"):
        try:
            module = importlib.import_module(module_name)
            versions[module_name] = str(getattr(module, "__version__", "unknown"))
        except Exception:  # noqa: BLE001 - a missing dep is a finding, not a crash
            versions[module_name] = "MISSING"
    return versions


def _package_integrity() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    findings: dict[str, Any] = {"package_root": str(package_root)}
    schema_path = package_root / "spec" / "model.schema.json"
    ledger_path = package_root / "validation_status.json"
    for label, path in (("model_schema", schema_path), ("validation_ledger", ledger_path)):
        try:
            json.loads(path.read_text(encoding="utf-8"))
            findings[label] = "ok"
        except Exception as exc:  # noqa: BLE001 - report, do not crash
            findings[label] = f"BROKEN: {exc}"
    return findings


def _float_determinism_probe() -> dict[str, Any]:
    """A tiny fixed computation whose digest must match across sessions."""

    import hashlib

    import numpy as np

    values = np.linspace(0.0, 1.0, 1001)
    result = np.sin(values * 3.75) @ np.cos(values * 1.25)
    digest = hashlib.sha256(f"{float(result):.17g}".encode()).hexdigest()[:16]
    return {
        "value": f"{float(result):.17g}",
        "digest": digest,
        "note": (
            "compare this digest across machines/sessions; a mismatch means "
            "platform floating-point differences that platform-evidence "
            "bundles must declare"
        ),
    }


def _ngspice_backend() -> dict[str, Any]:
    from pear_sim.oracle.ngspice import BINARY_ENV, find_ngspice

    binary = find_ngspice()
    if binary is None:
        return {
            "available": False,
            "note": f"install ngspice or set {BINARY_ENV}",
        }
    return {"available": True, "binary": binary}


def _limits() -> dict[str, Any]:
    import pear_sim.limits as limits_module
    from pear_sim.evidence.parameter_sweep import (
        MAX_PARAMETER_SWEEP_POINT_WALL_TIME_S,
        MAX_PARAMETER_SWEEP_TOTAL_WALL_TIME_S,
    )
    from pear_sim.spec.model import MAX_MODEL_SOURCE_BYTES

    values = {
        name: getattr(limits_module, name)
        for name in dir(limits_module)
        if name.startswith("MAX_")
    }
    values["MAX_MODEL_SOURCE_BYTES"] = MAX_MODEL_SOURCE_BYTES
    values["MAX_PARAMETER_SWEEP_POINT_WALL_TIME_S"] = (
        MAX_PARAMETER_SWEEP_POINT_WALL_TIME_S
    )
    values["MAX_PARAMETER_SWEEP_TOTAL_WALL_TIME_S"] = (
        MAX_PARAMETER_SWEEP_TOTAL_WALL_TIME_S
    )
    return values


def run_doctor() -> dict[str, Any]:
    """Collect the diagnosis; pure inspection, no side effects."""

    payload = {
        "schema": DOCTOR_SCHEMA,
        "pear_sim_version": __version__,
        "python": {
            "version": sys.version.split()[0],
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "dependencies": _dependency_versions(),
        "package_integrity": _package_integrity(),
        "float_determinism_probe": _float_determinism_probe(),
        "backends": {
            "ngspice_differential": _ngspice_backend(),
        },
        "limits": _limits(),
    }
    problems = []
    if any(v == "MISSING" for v in payload["dependencies"].values()):
        problems.append("missing dependency")
    integrity = payload["package_integrity"]
    if any(str(value).startswith("BROKEN") for value in integrity.values()):
        problems.append("package integrity")
    payload["status"] = "ok" if not problems else "problems: " + ", ".join(problems)
    return payload
