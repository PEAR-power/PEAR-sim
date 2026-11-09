# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Shared strict-JSON primitives for evidence producers and reports.

Only helpers whose previous per-module copies were behaviorally identical
live here.  Producer-specific policies — destination guards, error-type
mapping, artifact-manifest subjects, and bounded readers — intentionally
stay local to their modules; unifying them is a semantic change, not a
deduplication.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

__all__ = [
    "reject_json_constant",
    "require_finite",
    "unique_json_object",
    "write_json",
]


def reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def require_finite(value: Any, path: str = "$") -> None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"number outside the finite float contract at {path}") from exc
        if not math.isfinite(number):
            raise ValueError(f"non-finite number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, f"{path}[{index}]")


def write_json(path: Path, payload: Any) -> None:
    """Atomically write deterministic, finite JSON (sorted keys, no NaN)."""

    text = (
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
