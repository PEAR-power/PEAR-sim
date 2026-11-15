# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Standalone worker run by the motulator oracle in a SEPARATE interpreter.

This file is executed as ``python -I motulator_worker.py request.json out.json``
by an interpreter whose environment has motulator (MIT, Aalto Electric
Drives) installed. It deliberately imports nothing from pear_sim, so
PEARSim's own runtime never depends on motulator or on its dependencies.

The request describes one synchronous machine fed directly by a balanced
three-phase voltage source with an externally imposed rotor speed — the
subset where both engines integrate the SAME physics through different
formulations (motulator: stator flux linkage in rotor coordinates as the
state; PEARSim: abc winding currents with the speed and the inductance
matrix held per segment).
"""

from __future__ import annotations

import json
import platform
import sys


def _bilinear(axis_d, axis_q, table, i_d, i_q):
    """PEARSim's FluxMap interpolation (bilinear), reimplemented.

    Rows follow the d axis, columns the q axis, exactly as the model file
    states them, so the inverse below inverts the SAME map PEARSim uses
    rather than a re-gridded copy of it.
    """

    def bracket(axis, value):
        # Inside the grid this is exactly PEARSim's bracket. Outside it the
        # edge cell is EXTRAPOLATED linearly instead of clamped: a clamped map
        # has zero slope there and cannot be inverted, and the solver's trial
        # stages may step outside briefly. PEARSim itself refuses to run in
        # that region (non-positive differential inductance), so the
        # comparison never relies on the extrapolated values.
        if value <= axis[0]:
            return 0, (value - axis[0]) / (axis[1] - axis[0])
        if value >= axis[-1]:
            return len(axis) - 2, (value - axis[-2]) / (axis[-1] - axis[-2])
        for index in range(1, len(axis)):
            if value <= axis[index]:
                return index - 1, (value - axis[index - 1]) / (axis[index] - axis[index - 1])
        return len(axis) - 2, 1.0

    row, fx = bracket(axis_d, i_d)
    col, fy = bracket(axis_q, i_q)
    lower = table[row][col] + fy * (table[row][col + 1] - table[row][col])
    upper = table[row + 1][col] + fy * (table[row + 1][col + 1] - table[row + 1][col])
    return lower + fx * (upper - lower)


def _current_map(flux_map):
    """psi -> i, the exact inverse of the bilinear flux map, by Newton."""

    import numpy as np

    axis_d = flux_map["id_axis"]
    axis_q = flux_map["iq_axis"]
    psi_d_table = flux_map["psi_d"]
    psi_q_table = flux_map["psi_q"]
    step_d = 1e-6 * (axis_d[-1] - axis_d[0])
    step_q = 1e-6 * (axis_q[-1] - axis_q[0])

    def flux(i_d, i_q):
        return (
            _bilinear(axis_d, axis_q, psi_d_table, i_d, i_q),
            _bilinear(axis_d, axis_q, psi_q_table, i_d, i_q),
        )

    def invert_one(psi):
        target_d, target_q = psi.real, psi.imag
        i_d = i_q = 0.0
        for _ in range(60):
            f_d, f_q = flux(i_d, i_q)
            r_d, r_q = f_d - target_d, f_q - target_q
            if abs(r_d) + abs(r_q) < 1e-13:
                break
            a = (flux(i_d + step_d, i_q)[0] - flux(i_d - step_d, i_q)[0]) / (2 * step_d)
            b = (flux(i_d, i_q + step_q)[0] - flux(i_d, i_q - step_q)[0]) / (2 * step_q)
            c = (flux(i_d + step_d, i_q)[1] - flux(i_d - step_d, i_q)[1]) / (2 * step_d)
            d = (flux(i_d, i_q + step_q)[1] - flux(i_d, i_q - step_q)[1]) / (2 * step_q)
            det = a * d - b * c
            i_d -= (d * r_d - b * r_q) / det
            i_q -= (-c * r_d + a * r_q) / det
        else:
            raise RuntimeError(f"flux-map inversion did not converge at psi={psi}")
        return complex(i_d, i_q)

    def current(psi):
        if np.ndim(psi) == 0:
            return invert_one(complex(psi))
        values = np.asarray(psi, dtype=complex)
        return np.vectorize(invert_one, otypes=[complex])(values)

    return current


def main(request_path: str, output_path: str) -> int:
    import importlib.metadata

    import numpy as np
    from motulator.common.model import Simulation, SolverCfg, ThreePhaseSource
    from motulator.drive import model as drive_model

    with open(request_path, encoding="utf-8") as handle:
        request = json.load(handle)
    machine = request["machine"]
    if machine["magnetics"] == "linear":
        par = drive_model.SynchronousMachinePars(
            n_p=machine["pole_pairs"],
            R_s=machine["stator_resistance_ohm"],
            L_d=machine["d_axis_inductance_h"],
            L_q=machine["q_axis_inductance_h"],
            psi_f=machine["flux_linkage_wb"],
        )
    elif machine["magnetics"] == "flux_map":
        par = drive_model.SaturatedSynchronousMachinePars(
            n_p=machine["pole_pairs"],
            R_s=machine["stator_resistance_ohm"],
            i_s_dq_fcn=_current_map(machine["flux_map"]),
        )
    else:
        raise ValueError(f"unknown magnetics {machine['magnetics']!r}")
    sync = drive_model.SynchronousMachine(par)
    sync.state.theta_m = complex(machine["initial_angle_rad"])

    speed = request["speed"]

    def speed_at(t: float) -> float:
        start, stop = speed["ramp_start_s"], speed["ramp_stop_s"]
        if stop is None or t <= start:
            return speed["initial_rad_s"]
        if t >= stop:
            return speed["final_rad_s"]
        fraction = (t - start) / (stop - start)
        return speed["initial_rad_s"] + fraction * (speed["final_rad_s"] - speed["initial_rad_s"])

    mechanics = drive_model.ExternalRotorSpeed()
    mechanics.set_external_rotor_speed(speed_at)
    source_spec = request["source"]
    source = ThreePhaseSource(
        w_g=source_spec["angular_frequency_rad_s"],
        e_g=source_spec["amplitude_v"],
        phi=source_spec["phase_rad"],
    )
    mdl = drive_model.GridConnectedMachine(sync, mechanics, source)
    solver = request["solver"]
    cfg = SolverCfg(
        method=solver["method"],
        rtol=solver["rtol"],
        atol=solver["atol"],
        max_step=solver["max_step"],
    )
    sim = Simulation(mdl, show_progress=False, cfg=cfg)
    res = sim.simulate(t_stop=request["t_stop_s"], N_eval=request["n_eval"])

    time = np.asarray(res.mdl.t, dtype=float)
    i_s_ab = np.asarray(res.mdl.machine.i_s_ab, dtype=complex)
    offsets = request["phase_offsets_rad"]
    payload = {
        "engine": "motulator",
        "motulator_version": importlib.metadata.version("motulator"),
        "numpy_version": np.__version__,
        "python": platform.python_version(),
        "success": bool(res.success),
        "time": time.tolist(),
        # Phase currents from the peak-valued space vector; the isolated
        # neutral carries no zero-sequence current, so this is exact.
        "phase_currents_a": [
            np.real(i_s_ab * np.exp(-1j * offset)).tolist() for offset in offsets
        ],
        "torque_nm": np.asarray(res.mdl.machine.tau_M, dtype=float).tolist(),
        "electrical_angle_rad": np.asarray(res.mdl.machine.theta_m, dtype=float).tolist(),
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: motulator_worker.py REQUEST.json OUTPUT.json", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
