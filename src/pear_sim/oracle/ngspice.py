# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""ngspice differential adapter: a second, independent evidence class.

Where semantics are genuinely comparable — the LINEAR kernel: R/L/C networks
driven by DC or waveform sources — a pear_sim model is emitted as a SPICE
netlist and simulated by a local ngspice binary in batch mode. Anything
outside that subset fails closed: ngspice diodes and switches are physical
device models, not ideal ones, and pretending those semantics match would
manufacture false confidence.

Comparison conventions (declared, not hidden):

- ngspice output is resampled by its own ``linearize`` command onto the
  model's output grid, so both engines report on the same explicit times;
- sources with a ``step`` waveform are emitted as PWL with a declared finite
  edge (1 ns) — an explicit semantic difference covered by the error budget;
- only node voltages are compared in v0.1 (declared subset).

The adapter self-skips when no binary is present; there is no licensing
gate (ngspice is free software), but records still carry the exact version.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pear_sim.errors import ModelError
from pear_sim.spec.model import Model
from pear_sim.spec.waveforms import (
    PulseWaveform,
    PwlWaveform,
    SineWaveform,
    StepWaveform,
)

__all__ = [
    "NgspiceOracleError",
    "NgspiceOracleUnavailable",
    "NgspiceSimulation",
    "emit_netlist",
    "find_ngspice",
    "simulate_model",
]

BINARY_ENV = "PEAR_SIM_NGSPICE_BIN"
_STEP_EDGE_S = 1e-9
_TIMEOUT_S = 120.0


class NgspiceOracleError(ModelError):
    """The adapter cannot faithfully represent or run this model."""

    code = "oracle.ngspice_invalid"


class NgspiceOracleUnavailable(RuntimeError):
    """No usable ngspice binary on this machine."""


@dataclass(frozen=True)
class NgspiceSimulation:
    binary: str
    version: str
    time: tuple[float, ...]
    voltages: dict[str, tuple[float, ...]]
    netlist: str


def find_ngspice(environ: dict[str, str] | None = None) -> str | None:
    import os

    env = environ if environ is not None else dict(os.environ)
    override = env.get(BINARY_ENV)
    if override:
        return override if Path(override).exists() else None
    return shutil.which("ngspice")


def _node(name: str, ground: str) -> str:
    return "0" if name == ground else name


def _source_expression(component: Any) -> str:
    waveform = component.waveform
    if waveform is None:
        return f"DC {float(component.value):.17g}"
    if isinstance(waveform, SineWaveform):
        import math

        # SPICE SIN(VO VA FREQ TD THETA PHASE): phase is in degrees.
        return (
            f"SIN({waveform.offset:.17g} {waveform.amplitude:.17g} "
            f"{waveform.frequency_hz:.17g} 0 0 "
            f"{math.degrees(waveform.phase_rad):.17g})"
        )
    if isinstance(waveform, StepWaveform):
        # Declared finite edge: an explicit semantic difference vs pear_sim's
        # exact discontinuity, covered by the comparison budget.
        return (
            f"PWL(0 {waveform.initial:.17g} "
            f"{waveform.at_s:.17g} {waveform.initial:.17g} "
            f"{waveform.at_s + _STEP_EDGE_S:.17g} {waveform.final:.17g})"
        )
    if isinstance(waveform, PulseWaveform):
        period = 1.0 / waveform.frequency_hz
        width = waveform.duty * period
        return (
            f"PULSE({waveform.low:.17g} {waveform.high:.17g} "
            f"{waveform.phase_s:.17g} {_STEP_EDGE_S:.9g} {_STEP_EDGE_S:.9g} "
            f"{max(width - _STEP_EDGE_S, _STEP_EDGE_S):.17g} {period:.17g})"
        )
    if isinstance(waveform, PwlWaveform):
        pairs = " ".join(
            f"{time:.17g} {value:.17g}" for time, value in waveform.points
        )
        return f"PWL({pairs})"
    raise NgspiceOracleError(
        f"waveform kind {getattr(waveform, 'kind', '?')!r} has no SPICE mapping"
    )


def emit_netlist(model: Model) -> tuple[str, list[str]]:
    """Emit a SPICE netlist for the comparable linear subset; fail closed.

    Returns the netlist text and the compared node names (model outputs of
    the ``v(node)`` form).
    """

    supported = {"resistor", "capacitor", "inductor", "voltage_source", "current_source", "voltage_probe"}
    for component in model.components:
        if component.type not in supported:
            raise NgspiceOracleError(
                f"component {component.id!r} ({component.type}) is outside the "
                "comparable linear subset — ngspice devices for switches and "
                "diodes are physical models, not ideal ones"
            )
    if model.controls or model.sampled_controls:
        raise NgspiceOracleError("controls are outside the comparable subset")
    if model.couplings:
        raise NgspiceOracleError(
            "coupled inductors are not emitted in v0.1 of the adapter"
        )

    ground = model.ground
    node_names = {ground}
    lines = [
        f"* pear_sim differential netlist: {model.name}",
        ".options reltol=1e-6 abstol=1e-12 vntol=1e-9",
    ]
    needs_uic = False
    for index, component in enumerate(model.components):
        p = _node(component.p, ground)
        n = _node(component.n, ground)
        node_names.update((component.p, component.n))
        if component.type == "resistor":
            lines.append(f"R{index} {p} {n} {float(component.value):.17g}")
        elif component.type == "capacitor":
            suffix = ""
            if component.initial_state_explicit or component.initial_state != 0.0:
                suffix = f" ic={component.initial_state:.17g}"
                needs_uic = True
            lines.append(
                f"C{index} {p} {n} {float(component.value):.17g}{suffix}"
            )
        elif component.type == "inductor":
            suffix = ""
            if component.initial_state_explicit or component.initial_state != 0.0:
                suffix = f" ic={component.initial_state:.17g}"
                needs_uic = True
            lines.append(
                f"L{index} {p} {n} {float(component.value):.17g}{suffix}"
            )
        elif component.type == "voltage_source":
            lines.append(f"V{index} {p} {n} {_source_expression(component)}")
        elif component.type == "current_source":
            lines.append(f"I{index} {p} {n} {_source_expression(component)}")
        # voltage_probe adds no element.

    compared: list[str] = []
    for output in model.outputs:
        if output.startswith("v(") and output[2:-1] in node_names and output[2:-1] != ground:
            compared.append(output[2:-1])
    if not compared:
        raise NgspiceOracleError(
            "no comparable outputs: the adapter compares v(node) outputs only"
        )

    transient = model.transient
    wrdata_targets = " ".join(f"v({node})" for node in compared)
    uic = " uic" if needs_uic else ""
    lines.extend(
        [
            ".control",
            "set filetype=ascii",
            f"tran {transient.output_step:.17g} {transient.stop:.17g}{uic}",
            f"linearize {wrdata_targets}",
            f"wrdata OUTPUT_FILE {wrdata_targets}",
            "quit",
            ".endc",
            ".end",
            "",
        ]
    )
    return "\n".join(lines), compared


def simulate_model(model: Model, *, binary: str | None = None) -> NgspiceSimulation:
    """Run the comparable subset through ngspice batch mode."""

    resolved = binary or find_ngspice()
    if resolved is None:
        raise NgspiceOracleUnavailable(
            "no ngspice binary found (install ngspice or set "
            f"{BINARY_ENV})"
        )
    netlist, compared = emit_netlist(model)
    with tempfile.TemporaryDirectory(prefix="pear_sim-ngspice-") as workdir:
        directory = Path(workdir)
        data_path = directory / "waves.data"
        netlist_text = netlist.replace("OUTPUT_FILE", str(data_path))
        netlist_path = directory / "circuit.cir"
        netlist_path.write_text(netlist_text, encoding="utf-8")
        try:
            completed = subprocess.run(
                [resolved, "-b", str(netlist_path)],
                capture_output=True,
                timeout=_TIMEOUT_S,
                cwd=directory,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise NgspiceOracleError("ngspice exceeded the time budget") from exc
        if completed.returncode != 0 or not data_path.exists():
            stderr = completed.stderr.decode("utf-8", errors="replace")[-2000:]
            raise NgspiceOracleError(f"ngspice failed: {stderr}")
        version_line = "unknown"
        for line in completed.stdout.decode("utf-8", errors="replace").splitlines():
            if "ngspice" in line.lower() and any(ch.isdigit() for ch in line):
                version_line = line.strip()
                break
        raw = np.loadtxt(data_path)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
    # wrdata layout: time value time value ... per compared vector.
    if raw.shape[1] != 2 * len(compared):
        raise NgspiceOracleError(
            f"unexpected wrdata layout: {raw.shape[1]} columns for "
            f"{len(compared)} vectors"
        )
    time = tuple(float(value) for value in raw[:, 0])
    voltages = {
        node: tuple(float(value) for value in raw[:, 2 * index + 1])
        for index, node in enumerate(compared)
    }
    return NgspiceSimulation(
        binary=resolved,
        version=version_line,
        time=time,
        voltages=voltages,
        netlist=netlist_text,
    )
