"""Bridge to the optional compiled kernel ``pear-sim-core`` (Rust).

The kernel integrates one segment of x' = A x + c with Dormand-Prince 5(4)
and terminal events, mirroring ``solve_ivp(method="RK45")`` step for step.
This module turns a compiled mode and its event closures into plain arrays,
calls the kernel, and wraps the answer in the same ``OdeResult`` shape SciPy
returns, so every consumer downstream (metrics, extrema, sampling, events)
runs unchanged.

Fail closed: a model or segment the kernel does not cover raises with the
reason; nothing falls back to the reference engine silently.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
from scipy.integrate._ivp.common import OdeSolution
from scipy.integrate._ivp.ivp import OdeResult
from scipy.integrate._ivp.rk import RkDenseOutput

from pear_sim.errors import SimulationError

try:  # optional extra
    import pear_sim_core as _core
except ImportError:  # pragma: no cover - depends on the environment
    _core = None

_MESSAGES = {
    0: "The solver successfully reached the end of the integration interval.",
    1: "A termination event occurred.",
}


def available() -> bool:
    return _core is not None


def kernel_id() -> str:
    if _core is None:
        raise SimulationError("the compiled kernel pear-sim-core is not installed")
    return f"rust-core-{_core.__version__}"


def require_supported(model: Any, circuit: Any) -> str:
    """Return the backend id, or raise naming every unsupported feature."""

    if _core is None:
        raise SimulationError(
            "backend 'rust' needs the optional compiled kernel: install pear-sim-core "
            "(build it from rust/pear-sim-core with maturin)"
        )
    problems = []
    if model.transient.solver != "dopri5":
        problems.append(f"solver {model.transient.solver!r} (only dopri5 is ported)")
    if getattr(circuit, "_source_waveforms", ()):
        problems.append("time-varying waveform sources")
    if getattr(model, "machines", ()):
        problems.append("electrical machines")
    if problems:
        raise SimulationError(
            "backend 'rust' does not support this model yet: " + "; ".join(problems)
        )
    return kernel_id()


def _event_rows(
    functions: Sequence[Callable[..., float]], n: int
) -> tuple[list[bool], list[float], np.ndarray, list[float], list[float], list[float], list[float]]:
    is_constant: list[bool] = []
    constant: list[float] = []
    coef = np.zeros((len(functions), n))
    offset: list[float] = []
    early_until: list[float] = []
    early_value: list[float] = []
    direction: list[float] = []
    for index, function in enumerate(functions):
        spec = getattr(function, "pear_spec", None)
        if spec is None:
            raise SimulationError(
                "backend 'rust': an event function has no data description "
                f"({getattr(function, '__name__', 'event')})"
            )
        if getattr(function, "terminal", False) is not True:
            raise SimulationError("backend 'rust' supports terminal events only")
        if spec[0] == "constant":
            is_constant.append(True)
            constant.append(float(spec[1]))
            offset.append(0.0)
            early_until.append(-np.inf)
            early_value.append(0.0)
        elif spec[0] == "affine":
            _, row, row_offset, until, value = spec
            is_constant.append(False)
            constant.append(0.0)
            coef[index, :] = row
            offset.append(float(row_offset))
            early_until.append(float(until))
            early_value.append(float(value))
        else:  # pragma: no cover - programming error
            raise SimulationError(f"unknown event description {spec[0]!r}")
        direction.append(float(getattr(function, "direction", 0.0)))
    return is_constant, constant, coef, offset, early_until, early_value, direction


def solve_segment(
    mode: Any,
    t0: float,
    t1: float,
    state: np.ndarray,
    *,
    rtol: float,
    atol: float,
    max_step: float,
    events: Sequence[Callable[..., float]] | None,
) -> OdeResult:
    """Integrate one segment in the kernel; same result shape as ``solve_ivp``."""

    if getattr(mode, "source_count", 0):
        raise SimulationError("backend 'rust': a segment with waveform sources reached the kernel")
    a = np.ascontiguousarray(mode.a, dtype=float)
    c = np.ascontiguousarray(mode.c, dtype=float)
    y0 = np.ascontiguousarray(state, dtype=float)
    n = y0.size
    functions = list(events or ())
    rows = _event_rows(functions, n)
    out = _core.integrate_affine_rk45(
        a, c, y0, float(t0), float(t1), float(rtol), float(atol), float(max_step), *rows
    )
    ts = out["ts"]
    ys = out["ys"].reshape(-1, n).T
    count = out["dense_t"].size
    y_old = out["dense_y_old"].reshape(count, n)
    q = out["dense_q"].reshape(count, n, 4)
    interpolants = [
        RkDenseOutput(out["dense_t_old"][i], out["dense_t"][i], y_old[i], q[i])
        for i in range(count)
    ]
    status = int(out["status"])
    if functions:
        fired = int(out["event_index"])
        t_events = [
            np.array([out["event_root"]]) if index == fired else np.array([])
            for index in range(len(functions))
        ]
        y_events = [
            np.array([ys[:, -1]]) if index == fired else np.array([])
            for index in range(len(functions))
        ]
    else:
        t_events = None
        y_events = None
    return OdeResult(
        t=ts,
        y=ys,
        sol=OdeSolution(ts, interpolants) if interpolants else None,
        t_events=t_events,
        y_events=y_events,
        nfev=int(out["nfev"]),
        njev=0,
        nlu=0,
        status=status,
        message=_MESSAGES.get(status, out["message"]),
        success=status >= 0,
    )
