"""Isolated point worker for bounded parameter-sweep simulations.

This module is an implementation detail.  The parent process owns request
validation, timeouts, artifact verification, aggregation, and human reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import pear_sim
from pear_sim.run.io import write_result
from pear_sim.spec.model import load_model
from pear_sim.core.solver import HybridSimulator


_MAX_WORKER_ERROR_CHARS = 1024


def _error(exc: Exception) -> dict[str, str]:
    message = str(exc)
    if len(message) > _MAX_WORKER_ERROR_CHARS:
        message = message[: _MAX_WORKER_ERROR_CHARS - 1] + "…"
    return {"type": type(exc).__name__[:256], "message": message}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("model", type=Path)
    # Repeatable pairs: one pair is the 1-D sweep, several pairs are one
    # tolerance corner. Pair order is positional and must match.
    parser.add_argument("--parameter", required=True, action="append")
    parser.add_argument("--value-si", type=float, required=True, action="append")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if len(args.parameter) != len(args.value_si):
            raise ValueError("--parameter and --value-si pairs must match")
        if len(set(args.parameter)) != len(args.parameter):
            raise ValueError("duplicate --parameter overrides")
        model = load_model(
            args.model,
            overrides=dict(zip(args.parameter, args.value_si)),
        )
        result = HybridSimulator(model).simulate()
        write_result(result, args.out)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "fail", "error": _error(exc)},
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    package_init = Path(pear_sim.__file__).resolve()
    worker_module = Path(__file__).resolve()
    print(
        json.dumps(
            {
                "status": "pass",
                "identity": {
                    "package_init": str(Path(pear_sim.__file__).resolve()),
                    "package_init_sha256": hashlib.sha256(
                        package_init.read_bytes()
                    ).hexdigest(),
                    "worker_module": str(worker_module),
                    "worker_module_sha256": hashlib.sha256(
                        worker_module.read_bytes()
                    ).hexdigest(),
                    "version": pear_sim.__version__,
                },
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the parent process
    raise SystemExit(main())
