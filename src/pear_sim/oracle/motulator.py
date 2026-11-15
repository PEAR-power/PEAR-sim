# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""motulator differential adapter: an independent engine for MACHINES.

ngspice cannot judge PEARSim's machine models — SPICE has no synchronous
machine — so until now every machine capability carried "no independent-engine
comparison". motulator (Aalto Electric Drives, MIT licence) is an open,
independently written drive simulator that integrates the stator flux
linkage in rotor coordinates; PEARSim integrates abc winding currents with
the speed and the (salient or saturated) inductance matrix HELD per segment.
Same physics, different formulation, so a shared wrong assumption is not
the expected outcome — the definition of a useful second engine.

The comparable subset (fail closed outside it): ONE ``pmsm`` machine with a
``driven_speed`` (constant or ramp), its three terminals fed by a balanced
positive-sequence set of sinusoidal voltage sources referred to ground, an
isolated star point, no controls. Converter and PWM semantics differ
between the engines (motulator models its own modulator and computational
delay), so they are deliberately outside this adapter.

motulator runs in a SEPARATE interpreter named by ``PEAR_SIM_MOTULATOR_PYTHON``
through :mod:`pear_sim.oracle.motulator_worker`; PEARSim never imports it. The
adapter self-skips when no such interpreter is configured, and every record
carries the exact motulator, numpy and Python versions.

Agreement here is VERIFICATION between two simulators, not validation against
a machine on a test bench.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pear_sim.errors import ModelError
from pear_sim.spec.machines import PHASE_OFFSETS
from pear_sim.spec.model import Model
from pear_sim.spec.waveforms import SineWaveform

__all__ = [
    "MotulatorOracleError",
    "MotulatorOracleUnavailable",
    "MotulatorSimulation",
    "PYTHON_ENV",
    "compare_with_result",
    "emit_request",
    "find_motulator_python",
    "simulate_model",
]

PYTHON_ENV = "PEAR_SIM_MOTULATOR_PYTHON"
WORKER = Path(__file__).with_name("motulator_worker.py")
_TIMEOUT_S = 600.0
# The balanced-set check: amplitudes and frequencies equal to round-off, and
# phases 120 degrees apart to well below anything a model file states.
_BALANCE_RTOL = 1e-12
_PHASE_TOL_RAD = 1e-9
# The reference engine runs FAR tighter than motulator's defaults (rtol 1e-3)
# so the comparison measures PEARSim, not motulator's tolerance.
SOLVER = {"method": "DOP853", "rtol": 1e-10, "atol": 1e-12}


class MotulatorOracleError(ModelError):
    """The adapter cannot faithfully represent this model."""

    code = "oracle.motulator_invalid"


class MotulatorOracleUnavailable(RuntimeError):
    """No interpreter with motulator is configured on this machine."""


@dataclass(frozen=True)
class MotulatorSimulation:
    python: str
    versions: dict[str, str]
    request: dict[str, Any]
    time: tuple[float, ...]
    phase_currents_a: tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]
    torque_nm: tuple[float, ...]


def find_motulator_python(environ: dict[str, str] | None = None) -> str | None:
    import os

    env = environ if environ is not None else dict(os.environ)
    override = env.get(PYTHON_ENV)
    if override and Path(override).exists():
        return override
    return None


def _wrap(angle: float) -> float:
    return (angle + math.pi) % math.tau - math.pi


def emit_request(model: Model) -> dict[str, Any]:
    """The worker request for the comparable subset; fail closed outside it."""

    if len(model.machines) != 1:
        raise MotulatorOracleError("the adapter compares exactly one machine")
    (spec,) = model.machines
    if spec.kind != "pmsm":
        raise MotulatorOracleError(f"machine kind {spec.kind!r} has no motulator mapping")
    if spec.driven_speed is None:
        raise MotulatorOracleError(
            "the machine must have a driven_speed: with free mechanics the "
            "comparison would mix the electromagnetic and mechanical models"
        )
    if model.controls or model.sampled_controls or model.signals is not None:
        raise MotulatorOracleError(
            "controls are outside the comparable subset (the engines model PWM and "
            "computational delay differently)"
        )
    prefix = f"{spec.id}."
    windings = set(spec.winding_ids)
    for coupling in model.couplings:
        if coupling.first not in windings or coupling.second not in windings:
            raise MotulatorOracleError("couplings outside the machine are not compared")
    external = [c for c in model.components if not c.id.startswith(prefix)]
    sources: dict[str, Any] = {}
    for component in external:
        if component.type != "voltage_source" or not isinstance(component.waveform, SineWaveform):
            raise MotulatorOracleError(
                f"component {component.id!r} is outside the comparable subset: only "
                "three sinusoidal voltage sources feeding the machine terminals"
            )
        if component.n != model.ground or component.p not in spec.terminals:
            raise MotulatorOracleError(
                f"source {component.id!r} must run from a machine terminal (p) to "
                "ground (n)"
            )
        if component.p in sources:
            raise MotulatorOracleError(f"terminal {component.p!r} is fed twice")
        sources[component.p] = component.waveform
    if set(sources) != set(spec.terminals):
        raise MotulatorOracleError("every machine terminal needs exactly one source")
    neutral_users = [
        c.id
        for c in model.components
        if spec.neutral in (c.p, c.n) and not c.id.startswith(prefix)
    ]
    if neutral_users or spec.neutral == model.ground:
        raise MotulatorOracleError("the star point must be isolated")

    waves = [sources[terminal] for terminal in spec.terminals]
    reference = waves[0]
    for wave, (label, offset) in zip(waves, PHASE_OFFSETS, strict=True):
        if wave.offset != 0.0:
            raise MotulatorOracleError(f"phase {label}: a DC offset is not compared")
        if not math.isclose(wave.amplitude, reference.amplitude, rel_tol=_BALANCE_RTOL):
            raise MotulatorOracleError("the supply is not balanced (amplitudes differ)")
        if not math.isclose(wave.frequency_hz, reference.frequency_hz, rel_tol=_BALANCE_RTOL):
            raise MotulatorOracleError("the supply is not balanced (frequencies differ)")
        if abs(_wrap(wave.phase_rad - (reference.phase_rad - offset))) > _PHASE_TOL_RAD:
            raise MotulatorOracleError(
                f"phase {label} is not 120 degrees from its neighbours in the "
                "positive (abc) sequence"
            )

    # PEARSim: v_k = A*sin(w*t + phase_a - offset_k)
    #        = A*cos(w*t + phase_a - pi/2 - offset_k)
    # motulator: v_k = Re(e_g * exp(j*(w*t + phi)) * exp(-j*offset_k)), so
    # phi = phase_a - pi/2.
    machine: dict[str, Any] = {
        "pole_pairs": spec.pole_pairs,
        "stator_resistance_ohm": spec.stator_resistance_ohm,
        "initial_angle_rad": spec.initial_angle_rad,
    }
    if spec.flux_map is not None:
        flux_map = spec.flux_map
        machine.update(
            magnetics="flux_map",
            flux_map={
                "id_axis": list(flux_map.id_axis),
                "iq_axis": list(flux_map.iq_axis),
                "psi_d": [list(row) for row in flux_map.psi_d],
                "psi_q": [list(row) for row in flux_map.psi_q],
            },
        )
    else:
        machine.update(
            magnetics="linear",
            d_axis_inductance_h=spec.d_axis_inductance_h,
            q_axis_inductance_h=spec.q_axis_inductance_h,
            flux_linkage_wb=spec.flux_linkage_wb,
        )
    transient = model.transient
    if transient.start != 0.0:
        raise MotulatorOracleError("the comparison window must start at t = 0")
    steps = round((transient.stop - transient.start) / transient.output_step)
    driven = spec.driven_speed
    return {
        "machine": machine,
        "source": {
            "amplitude_v": reference.amplitude,
            "angular_frequency_rad_s": math.tau * reference.frequency_hz,
            "phase_rad": reference.phase_rad - math.pi / 2.0,
        },
        "speed": {
            "initial_rad_s": driven.initial_rad_s,
            "final_rad_s": driven.final_rad_s if driven.final_rad_s is not None else driven.initial_rad_s,
            "ramp_start_s": driven.ramp_start_s,
            "ramp_stop_s": driven.ramp_stop_s,
        },
        "phase_offsets_rad": [offset for _label, offset in PHASE_OFFSETS],
        # motulator evaluates linspace(0, t_stop, N_eval + 1) WITHOUT its end
        # point, so asking for one extra step returns exactly PEARSim's grid
        # 0, dt, ..., stop. The comparison then refuses any other grid rather
        # than interpolating between them.
        "t_stop_s": transient.stop + transient.output_step,
        "n_eval": steps + 1,
        "solver": {**SOLVER, "max_step": transient.output_step},
    }


def simulate_model(model: Model, *, python: str | None = None) -> MotulatorSimulation:
    interpreter = python or find_motulator_python()
    if interpreter is None:
        raise MotulatorOracleUnavailable(
            f"set {PYTHON_ENV} to an interpreter that has motulator installed"
        )
    request = emit_request(model)
    with tempfile.TemporaryDirectory(prefix="pear-sim-motulator-") as folder:
        request_path = Path(folder) / "request.json"
        output_path = Path(folder) / "result.json"
        request_path.write_text(json.dumps(request), encoding="utf-8")
        completed = subprocess.run(
            [interpreter, "-I", str(WORKER), str(request_path), str(output_path)],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_S,
            check=False,
        )
        if completed.returncode != 0:
            raise MotulatorOracleError(
                f"motulator worker failed (exit {completed.returncode}): "
                f"{completed.stderr.strip()[-2000:]}"
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    for key in ("time", "phase_currents_a", "torque_nm", "motulator_version", "success"):
        if key not in payload:
            raise MotulatorOracleError(f"motulator worker output lacks {key!r}")
    if not payload["success"]:
        raise MotulatorOracleError("motulator reported a numerical failure")
    currents = payload["phase_currents_a"]
    return MotulatorSimulation(
        python=interpreter,
        versions={
            "motulator": payload["motulator_version"],
            "numpy": payload["numpy_version"],
            "python": payload["python"],
        },
        request=request,
        time=tuple(payload["time"]),
        phase_currents_a=(tuple(currents[0]), tuple(currents[1]), tuple(currents[2])),
        torque_nm=tuple(payload["torque_nm"]),
    )


def compare_with_result(result: Any, reference: MotulatorSimulation) -> dict[str, Any]:
    """Peak-normalised maximum differences on PEARSim's own output grid.

    Requires the three winding currents and the torque among the outputs, so
    the comparison reads recorded, auditable signals only.
    """

    (spec,) = result.model.machines
    names = [f"i({winding})" for winding in spec.winding_ids] + [f"torque({spec.id})"]
    for name in names:
        if name not in result.signals:
            raise MotulatorOracleError(f"the comparison needs {name} among the outputs")
    time = np.asarray(result.time, dtype=float)
    reference_time = np.asarray(reference.time, dtype=float)
    step = float(result.model.transient.output_step)
    if time.shape != reference_time.shape or np.max(np.abs(time - reference_time)) > 1e-9 * step:
        raise MotulatorOracleError(
            "the two engines did not report on the same time grid; the comparison "
            "refuses to interpolate between grids"
        )
    series = list(reference.phase_currents_a) + [reference.torque_nm]
    record: dict[str, Any] = {
        "engine": "motulator",
        "versions": reference.versions,
        "claim": (
            "VERIFICATION against an independent open simulator on the declared "
            "comparable subset; not validation against hardware"
        ),
        "signals": {},
    }
    for name, values in zip(names, series, strict=True):
        ours = np.asarray(result.signals[name], dtype=float)
        theirs = np.asarray(values, dtype=float)
        scale = float(max(np.max(np.abs(ours)), np.max(np.abs(theirs))))
        difference = float(np.max(np.abs(ours - theirs)))
        record["signals"][name] = {
            "peak": scale,
            "max_abs_difference": difference,
            "max_relative_to_peak": difference / scale if scale > 0.0 else 0.0,
        }
    record["worst_relative_to_peak"] = max(
        item["max_relative_to_peak"] for item in record["signals"].values()
    )
    return record
