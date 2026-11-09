# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Gate/error primitives shared verbatim by the boundary oracles.

Only helpers whose per-oracle copies were byte-identical live here.  Topology
equations, gate contracts, tolerances, and observation semantics remain local
to each oracle module — sharing those would weaken the documented oracle
independence boundary (AGENTS.md), which permits shared bookkeeping but not
shared physics.
"""

from __future__ import annotations

import math
from typing import Any

__all__ = [
    "append_state_error",
    "comparison_gate",
    "exact_gate",
    "oracle_gate",
    "relative_or_scaled_error",
]


def comparison_gate(gate_id: str, actual: float, limit: float) -> dict[str, Any]:
    finite = math.isfinite(actual)
    return {
        "id": gate_id,
        "actual": actual if finite else None,
        "operator": "<=",
        "limit": limit,
        "status": "pass" if finite and 0.0 <= actual <= limit else "fail",
    }


def append_state_error(
    expected: float,
    actual: float,
    scale: float,
    relative_errors: list[float],
    zero_errors: list[float],
) -> None:
    if abs(expected) <= scale * 1e-10:
        zero_errors.append(abs(actual - expected) / scale)
    else:
        relative_errors.append(abs(actual - expected) / abs(expected))


def exact_gate(gate_id: str, actual: Any, reference: Any) -> dict[str, Any]:
    return {
        "id": gate_id,
        "metric": "exact_match",
        "actual": actual,
        "reference": reference,
        "limit": True,
        "status": "pass" if actual == reference else "fail",
    }


def oracle_gate(gate_id: str, actual: float, limit: float) -> dict[str, Any]:
    finite = math.isfinite(actual) and actual >= 0.0
    return {
        "id": gate_id,
        "metric": "normalized_absolute_error",
        "actual": actual,
        "operator": "<=",
        "limit": limit,
        "status": "pass" if finite and actual <= limit else "fail",
    }


def relative_or_scaled_error(expected: float, actual: float, scale: float) -> float:
    if abs(expected) <= scale * 1e-12:
        return abs(actual - expected) / scale
    return abs(actual - expected) / abs(expected)
