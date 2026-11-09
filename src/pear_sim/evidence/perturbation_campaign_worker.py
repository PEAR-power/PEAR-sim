# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Isolated point worker for a perturbation campaign.

Mirrors ``parameter_sweep_worker`` but takes a model that is already complete:
the campaign materializes every override and every event into the source
before the worker sees it, so there is nothing left to pass on the command
line. The separate process is not ceremony — a badly conditioned point really
can fail to converge and must be killable on a timeout rather than hanging the
campaign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from pear_sim.core.solver import HybridSimulator
from pear_sim.run.io import write_result
from pear_sim.spec.model import load_model


_MAX_WORKER_ERROR_CHARS = 1024


def _error(exc: Exception) -> dict[str, str]:
    message = str(exc)
    if len(message) > _MAX_WORKER_ERROR_CHARS:
        message = message[: _MAX_WORKER_ERROR_CHARS - 1] + "…"
    return {"type": type(exc).__name__[:256], "message": message}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = HybridSimulator(load_model(args.model)).simulate()
        write_result(result, args.out)
    except Exception as exc:  # noqa: BLE001 - reported verbatim to the parent
        print(
            json.dumps(
                {"status": "fail", "error": _error(exc)},
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps({"status": "pass"}, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
