"""Isolated point worker for bounded parameter-sweep simulations.

This module is an implementation detail.  The parent process owns request
validation, timeouts, artifact verification, aggregation, and human reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import warnings
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
    parser.add_argument("model", type=Path, nargs="?")
    # Repeatable pairs: one pair is the 1-D sweep, several pairs are one
    # tolerance corner. Pair order is positional and must match.
    parser.add_argument("--parameter", action="append")
    parser.add_argument("--value-si", type=float, action="append")
    parser.add_argument("--out", type=Path)
    # Lean runs compute the same metrics with less evidence (batch screening).
    parser.add_argument("--lean", action="store_true")
    # Stay alive and answer many points (see _serve).
    parser.add_argument("--serve", action="store_true")
    return parser


def _identity() -> dict[str, str]:
    package_init = Path(pear_sim.__file__).resolve()
    worker_module = Path(__file__).resolve()
    return {
        "package_init": str(package_init),
        "package_init_sha256": hashlib.sha256(package_init.read_bytes()).hexdigest(),
        "worker_module": str(worker_module),
        "worker_module_sha256": hashlib.sha256(worker_module.read_bytes()).hexdigest(),
        "version": pear_sim.__version__,
    }


def _simulate_point(
    model_path: Path,
    parameters: Sequence[str],
    values_si: Sequence[float],
    out: Path,
    lean: bool,
) -> tuple[int, dict[str, object]]:
    """Run one point; return the exit code and the reply the parent checks."""

    try:
        if len(parameters) != len(values_si):
            raise ValueError("--parameter and --value-si pairs must match")
        if len(set(parameters)) != len(parameters):
            raise ValueError("duplicate --parameter overrides")
        model = load_model(model_path, overrides=dict(zip(parameters, values_si)))
        result = HybridSimulator(
            model, evidence_mode="lean" if lean else "full"
        ).simulate()
        write_result(result, out)
    except Exception as exc:
        return 1, {"status": "fail", "error": _error(exc)}
    return 0, {"status": "pass", "identity": _identity()}


def _reply_text(code: int, payload: dict[str, object]) -> str:
    # A failure may quote the user's text; a pass is pure ASCII paths and hashes.
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=code == 0,
        sort_keys=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.serve:
        return _serve()
    if args.model is None or args.out is None or not args.parameter:
        print(_reply_text(1, {"status": "fail", "error": _error(
            ValueError("model, --parameter, --value-si and --out are required")
        )}))
        return 1
    code, payload = _simulate_point(
        args.model, args.parameter, args.value_si or [], args.out, args.lean
    )
    print(_reply_text(code, payload))
    return code


def _serve() -> int:
    """Answer point requests, one JSON line in and one JSON line out, until EOF.

    The parent keeps this process alive across points so Python and PEARSim
    are imported once, not once per point. Each point must behave exactly as
    if it had its own process:

    - stdout and stderr (file descriptors 1 and 2) point at a file the parent
      names for that point while it runs, so anything it prints is retained
      for that point alone and the parent rejects it as before;
    - the working directory is the point's directory;
    - warning filters and the "already shown" registries are restored after
      the point, so a warning shown for one point is shown again for the next.

    Replies travel on a private copy of the original stdout, which no point
    code can write to.
    """

    protocol = os.fdopen(os.dup(1), "w", encoding="ascii", buffering=1)
    os.dup2(2, 1)
    protocol.write(json.dumps({"status": "ready"}) + "\n")
    for line in sys.stdin:
        request = json.loads(line)
        os.chdir(request["cwd"])
        sys.stdout.flush()
        sys.stderr.flush()
        saved = (os.dup(1), os.dup(2))
        capture = os.open(
            request["stderr_path"],
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0),
            0o600,
        )
        os.dup2(capture, 1)
        os.dup2(capture, 2)
        os.close(capture)
        try:
            with warnings.catch_warnings():
                code, payload = _simulate_point(
                    Path(request["model"]),
                    request["parameters"],
                    request["values_si"],
                    Path(request["out"]),
                    bool(request["lean"]),
                )
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])
        protocol.write(
            json.dumps({"exit": code, "reply": _reply_text(code, payload)}) + "\n"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the parent process
    raise SystemExit(main())
