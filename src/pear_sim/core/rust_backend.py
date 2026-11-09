# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
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

from pear_sim.core.metrics import _QUADRATURE_NODES, _QUADRATURE_WEIGHTS
from pear_sim.errors import SimulationError

try:  # optional extra
    import pear_sim_core as _core
except ImportError:  # pragma: no cover - depends on the environment
    _core = None

_NODES = [float(x) for x in _QUADRATURE_NODES]
_WEIGHTS = [float(x) for x in _QUADRATURE_WEIGHTS]

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
    result = OdeResult(
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
    # The raw dense arrays, so observe_segment does not rebuild them.
    result.pear_raw = out
    return result


def _rows(maps: Sequence[tuple[np.ndarray, float, np.ndarray]], n: int) -> tuple[np.ndarray, list[float]]:
    coef = np.zeros((len(maps), n))
    offset: list[float] = []
    for index, (row, constant, sources) in enumerate(maps):
        if np.any(np.asarray(sources) != 0.0):
            raise SimulationError("backend 'rust': an output depends on a waveform source")
        coef[index, :] = row
        offset.append(float(constant))
    return coef, offset


def _observer_rows(mode: Any, outputs: Sequence[str], energy: bool) -> tuple:
    cache = mode.__dict__.setdefault("_pear_observer_rows", {})
    key = (tuple(outputs), energy)
    if key not in cache:
        n = len(mode.circuit.state_components)
        output_rows = _rows([mode.output_affine(name) for name in outputs], n)
        if energy:
            components = mode.circuit.components
            voltage_rows = _rows([mode.component_voltage_affine(c) for c in components], n)
            current_rows = _rows(
                [
                    (np.zeros(n), 0.0, np.zeros(0))
                    if c.type == "voltage_probe"
                    else mode.component_current_affine(c)
                    for c in components
                ],
                n,
            )
        else:
            voltage_rows = current_rows = (np.zeros((0, n)), [])
        cache[key] = (output_rows, voltage_rows, current_rows)
    return cache[key]


def _window_state(window: Any, outputs: Sequence[str], components: Sequence[str]) -> tuple:
    energy = window.component_absorbed_energy
    return (
        float(window.integrated_interval_s),
        int(window.quadrature_intervals),
        [float(window.signal_integrals[name]) for name in outputs],
        [float(window.signal_square_integrals[name]) for name in outputs],
        [float(energy[cid]) for cid in components] if energy is not None else [],
    )


def _store_window(window: Any, observed: dict, outputs: Sequence[str], components: Sequence[str]) -> None:
    window.integrated_interval_s = observed["duration"]
    window.quadrature_intervals = observed["intervals"]
    for name, value in zip(outputs, observed["integrals"], strict=True):
        window.signal_integrals[name] = value
    for name, value in zip(outputs, observed["squares"], strict=True):
        window.signal_square_integrals[name] = value
    if window.component_absorbed_energy is not None:
        for cid, value in zip(components, observed["energy"], strict=True):
            window.component_absorbed_energy[cid] = value


def observe_segment(
    solution: OdeResult,
    mode: Any,
    accumulator: Any,
    outputs: Sequence[str],
    sample_times: Sequence[float],
) -> np.ndarray | None:
    """Add the segment's quadrature to ``accumulator`` and sample ``outputs``.

    Replaces ``accumulator.add_solution`` and the per-sample output reads for
    a kernel segment; returns one row of output values per sample time, or
    None for a segment without a step (nothing to integrate or sample).
    """

    if not accumulator.begin_segment(solution, mode):
        return None
    raw = solution.pear_raw
    energy = accumulator._component_energy_requested
    (out_coef, out_off), (v_coef, v_off), (i_coef, i_off) = _observer_rows(mode, outputs, energy)
    components = [c.id for c in mode.circuit.components] if energy else []
    observed = _core.observe_segment(
        [float(t) for t in solution.t],
        raw["dense_t_old"].tolist(),
        raw["dense_t"].tolist(),
        raw["dense_y_old"],
        raw["dense_q"],
        _NODES,
        _WEIGHTS,
        float(accumulator.midpoint),
        out_coef, out_off, v_coef, v_off, i_coef, i_off,
        [float(t) for t in sample_times],
        _window_state(accumulator.full, outputs, components),
        _window_state(accumulator.last_half, outputs, components),
    )
    _store_window(accumulator.full, observed["full"], outputs, components)
    _store_window(accumulator.last_half, observed["last_half"], outputs, components)
    return np.asarray(observed["samples"]).reshape(len(sample_times), len(outputs))


def state_extrema(
    solution: OdeResult, mode: Any, window_start: float, window_end: float
) -> dict[str, dict[str, float]]:
    """Kernel version of metrics._dense_state_extrema for a kernel segment."""

    if solution.sol is None or window_end <= window_start:
        return {}
    raw = solution.pear_raw
    circuit = mode.circuit
    components = sorted(circuit.state_components, key=lambda component: component.id)
    order = [circuit.state_index[component.id] for component in components]
    found = _core.state_extrema(
        [float(t) for t in solution.t],
        raw["dense_t_old"].tolist(),
        raw["dense_t"].tolist(),
        raw["dense_y_old"],
        raw["dense_q"],
        float(window_start),
        float(window_end),
        order,
    )
    extrema: dict[str, dict[str, float]] = {}
    for component, index in zip(components, order, strict=True):
        present, minimum, minimum_time, maximum, maximum_time = found[index]
        if present:
            extrema[component.id] = {
                "minimum": minimum,
                "minimum_time_s": minimum_time,
                "maximum": maximum,
                "maximum_time_s": maximum_time,
            }
    return extrema
