# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Rotating-machine declarations and their expansion into circuit branches.

A surface-magnet (non-salient) synchronous machine is, seen from its
terminals, three identical branches — stator resistance, the three-wire
equivalent stator inductance, and a motional back-EMF — plus one mechanical
state. The expansion is explicit: the loader turns the declaration into
ordinary components a reader can audit, exactly as control blocks expand
into gates. Nothing about the machine is hidden inside the kernel.

WHY THE THREE-WIRE EQUIVALENT. In a symmetric three-phase winding with an
isolated neutral the phase currents sum to zero, so the mutual coupling
folds into the self term and each phase behaves as a single inductance
``L_eq = L_s - M``. That is what ``stator_inductance_h`` declares, and it is
why no coupled-inductor machinery is needed here.

DECLARED SCOPE (v0.1). Non-salient only: ``d_axis_inductance_h`` and
``q_axis_inductance_h`` may be declared, but the loader fails closed unless
they are equal, because a salient machine has a rotor-position-dependent
inductance matrix that would have to be recompiled every integration
segment. Sinusoidal back-EMF only — no trapezoidal drive, no cogging, no
slot harmonics. No magnetic saturation, no iron loss, and no temperature
dependence of the flux linkage or the resistance.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from pear_sim.errors import ModelError

MACHINE_KINDS = ("pmsm",)
MAX_MACHINES = 4

# Phase label -> electrical angle offset of that winding's magnet axis, used
# as sin(theta_e - offset). Phase b LAGS a by 120 degrees and c leads it, the
# positive (abc) sequence. The sign here is not cosmetic: swapping b and c
# reverses the sequence, which leaves the internal power identity intact
# (both the EMF and the torque use the same offsets) while making the machine
# generate where it should motor. Only a sequence-sensitive test catches it,
# and one is in the suite.
PHASE_OFFSETS: tuple[tuple[str, float], ...] = (
    ("a", 0.0),
    ("b", 2.0 * math.pi / 3.0),
    ("c", -2.0 * math.pi / 3.0),
)


@dataclass(frozen=True)
class DrivenSpeed:
    """A mechanical speed imposed from outside instead of integrated.

    The dynamometer case: the shaft is turned by something that does not
    care what the machine does, and the QUESTION is the torque produced —
    not the speed reached. It is also the only way to ask a torque question
    of a machine whose own mechanical loop is not yet independently verified:
    imposing the speed removes that loop from the answer.

    ``final`` and ``ramp_stop`` express a linear ramp; a constant speed is
    the degenerate case where they are absent.
    """

    initial_rad_s: float
    final_rad_s: float | None = None
    ramp_start_s: float = 0.0
    ramp_stop_s: float | None = None

    def speed_at(self, time: float) -> float:
        if self.final_rad_s is None or self.ramp_stop_s is None:
            return self.initial_rad_s
        if time <= self.ramp_start_s:
            return self.initial_rad_s
        if time >= self.ramp_stop_s:
            return self.final_rad_s
        span = self.ramp_stop_s - self.ramp_start_s
        fraction = (time - self.ramp_start_s) / span
        return self.initial_rad_s + fraction * (self.final_rad_s - self.initial_rad_s)


@dataclass(frozen=True)
class FluxMap:
    """Saturating flux linkage psi_d(i_d, i_q), psi_q(i_d, i_q).

    WHY A MAP AND NOT TWO INDUCTANCES. A real reluctance or interior-magnet
    machine works deep in saturation: its axis inductances fall with current
    and each axis loads the other (cross-saturation). Constant Ld/Lq cannot
    express either. The map states the flux linkage directly, which is the
    quantity the machine equations actually need.

    THE MAP BELONGS TO THE MACHINE, and the controller declares its own.
    Sharing one would make the model measure a table against itself
    and delete the estimation error a drive engineer is looking for.

    THE MAGNET FLUX LIVES IN THE MAP. psi_d(0, 0) is what a magnet machine
    has with no current at all, so a map makes `flux_linkage_wb` redundant
    and the loader refuses declaring both — one fact, one place.
    """

    id_axis: tuple[float, ...]
    iq_axis: tuple[float, ...]
    psi_d: tuple[tuple[float, ...], ...]
    psi_q: tuple[tuple[float, ...], ...]

    def _bracket(self, axis: tuple[float, ...], value: float) -> tuple[int, float]:
        if value <= axis[0]:
            return 0, 0.0
        if value >= axis[-1]:
            return len(axis) - 2, 1.0
        for index in range(1, len(axis)):
            if value <= axis[index]:
                span = axis[index] - axis[index - 1]
                return index - 1, (value - axis[index - 1]) / span
        return len(axis) - 2, 1.0

    def _interpolate(
        self, grid: tuple[tuple[float, ...], ...], d_current: float, q_current: float
    ) -> float:
        row, fx = self._bracket(self.id_axis, d_current)
        col, fy = self._bracket(self.iq_axis, q_current)
        lower = grid[row][col] + fy * (grid[row][col + 1] - grid[row][col])
        upper = grid[row + 1][col] + fy * (grid[row + 1][col + 1] - grid[row + 1][col])
        return float(lower + fx * (upper - lower))

    def flux(self, d_current: float, q_current: float) -> tuple[float, float]:
        """Flux linkage of both axes at one operating point."""

        return (
            self._interpolate(self.psi_d, d_current, q_current),
            self._interpolate(self.psi_q, d_current, q_current),
        )

    def differential_inductances(
        self, d_current: float, q_current: float
    ) -> tuple[float, float]:
        """Local d/dq slopes d(psi)/d(i), which is what the kernel integrates.

        The SLOPE, not the chord psi/i: the circuit equation is
        v = R*i + d(psi)/dt, so what multiplies di/dt is the differential
        inductance. Using the chord instead would give the right flux and the
        wrong dynamics — a mistake that looks fine in steady state.

        Evaluated by central differences on a step tied to the local grid
        spacing, so the derivative inherits the map's own resolution rather
        than a constant nobody chose.
        """

        d_step = max(1e-9, 0.01 * (self.id_axis[-1] - self.id_axis[0]))
        q_step = max(1e-9, 0.01 * (self.iq_axis[-1] - self.iq_axis[0]))
        d_plus, _ = self.flux(d_current + d_step, q_current)
        d_minus, _ = self.flux(d_current - d_step, q_current)
        _, q_plus = self.flux(d_current, q_current + q_step)
        _, q_minus = self.flux(d_current, q_current - q_step)
        l_d = (d_plus - d_minus) / (2.0 * d_step)
        l_q = (q_plus - q_minus) / (2.0 * q_step)
        # A non-positive slope would mean flux falling with its own current:
        # not saturation but a broken table, and it would make the inductance
        # matrix indefinite.
        if l_d <= 0.0 or l_q <= 0.0:
            raise ModelError(
                f"flux map has a non-positive differential inductance at "
                f"(i_d={d_current:.6g}, i_q={q_current:.6g}): Ld={l_d:.6g}, "
                f"Lq={l_q:.6g}; flux must increase with its own axis current"
            )
        return l_d, l_q


def _parse_flux_map(raw: Any, *, machine_id: str) -> "FluxMap | None":
    if raw is None:
        return None
    name = f"machine {machine_id}.flux_map"
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    allowed = {"id_axis", "iq_axis", "psi_d", "psi_q"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ModelError(f"{name}: unknown field(s): {', '.join(unknown)}")
    for field in sorted(allowed):
        if field not in raw:
            raise ModelError(f"{name} requires {field}")
    axes = {}
    for field in ("id_axis", "iq_axis"):
        values = raw[field]
        if not isinstance(values, list) or len(values) < 2:
            raise ModelError(f"{name}.{field} must list at least two currents")
        parsed = tuple(
            _number(item, name=f"{name}.{field}[{index}]")
            for index, item in enumerate(values)
        )
        for index in range(1, len(parsed)):
            if parsed[index] <= parsed[index - 1]:
                raise ModelError(f"{name}.{field} must be strictly increasing")
        axes[field] = parsed
    grids = {}
    for field in ("psi_d", "psi_q"):
        rows = raw[field]
        if not isinstance(rows, list) or len(rows) != len(axes["id_axis"]):
            raise ModelError(
                f"{name}.{field} must have one row per id_axis entry "
                f"({len(axes['id_axis'])} rows)"
            )
        parsed_rows = []
        for row_index, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != len(axes["iq_axis"]):
                raise ModelError(
                    f"{name}.{field}[{row_index}] must have one entry per "
                    f"iq_axis value ({len(axes['iq_axis'])} entries)"
                )
            parsed_rows.append(
                tuple(
                    _number(item, name=f"{name}.{field}[{row_index}][{col}]")
                    for col, item in enumerate(row)
                )
            )
        grids[field] = tuple(parsed_rows)
    return FluxMap(
        id_axis=axes["id_axis"], iq_axis=axes["iq_axis"],
        psi_d=grids["psi_d"], psi_q=grids["psi_q"],
    )


@dataclass(frozen=True)
class MachineSpec:
    """One declared machine and the branch ids its expansion created."""

    id: str
    kind: str
    terminals: tuple[str, str, str]
    neutral: str
    pole_pairs: int
    stator_resistance_ohm: float
    stator_inductance_h: float
    #: Axis inductances. For a non-salient machine both equal the
    #: three-wire equivalent `stator_inductance_h` and nothing downstream
    #: changes; when they differ the machine expands into a COUPLED group
    #: whose matrix is re-evaluated every integration segment.
    d_axis_inductance_h: float
    q_axis_inductance_h: float
    flux_linkage_wb: float
    inertia_kg_m2: float
    viscous_friction_nm_s: float
    load_torque_nm: float
    initial_speed_rad_s: float
    initial_angle_rad: float
    emf_ids: tuple[str, str, str]
    current_expressions: tuple[str, str, str]
    #: When present the mechanical loop is NOT integrated: the speed is
    #: imposed and the torque is the measurement.
    driven_speed: "DrivenSpeed | None" = None
    #: When present, saturation replaces the constant axis inductances: the
    #: flux linkage comes from the map and the inductance matrix is rebuilt
    #: from its LOCAL SLOPES at the operating point of each segment.
    flux_map: "FluxMap | None" = None

    @property
    def phase_offsets(self) -> tuple[float, float, float]:
        return tuple(offset for _, offset in PHASE_OFFSETS)

    @property
    def is_salient(self) -> bool:
        # A mapped machine is salient by construction: its axes differ
        # wherever the map says they do, at every operating point.
        return (
            self.flux_map is not None
            or self.d_axis_inductance_h != self.q_axis_inductance_h
        )

    @property
    def is_saturating(self) -> bool:
        return self.flux_map is not None

    @property
    def winding_ids(self) -> tuple[str, str, str]:
        return tuple(f"{self.id}.{label}.l" for label, _ in PHASE_OFFSETS)


def _number(raw: Any, *, name: str, positive: bool = False, allow_zero: bool = True) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ModelError(f"{name} must be a number")
    value = float(raw)
    if not math.isfinite(value):
        raise ModelError(f"{name} must be finite")
    if positive and value <= 0.0:
        raise ModelError(f"{name} must be positive")
    if not allow_zero and value == 0.0:
        raise ModelError(f"{name} must be non-zero")
    if not positive and value < 0.0 and name.endswith(("_ohm", "_h", "_kg_m2", "_nm_s")):
        raise ModelError(f"{name} must be non-negative")
    return value


def _text(raw: Any, *, name: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ModelError(f"{name} must be a non-empty string")
    return raw


def _parse_driven_speed(raw: Any, *, machine_id: str) -> "DrivenSpeed | None":
    """Parse the optional imposed-speed profile, fail-closed."""

    if raw is None:
        return None
    name = f"machine {machine_id}.driven_speed"
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    allowed = {"initial_rad_s", "final_rad_s", "ramp_start_s", "ramp_stop_s"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ModelError(f"{name}: unknown field(s): {', '.join(unknown)}")
    if "initial_rad_s" not in raw:
        raise ModelError(f"{name} requires initial_rad_s")
    initial = _number(raw["initial_rad_s"], name=f"{name}.initial_rad_s")
    has_final = "final_rad_s" in raw
    has_stop = "ramp_stop_s" in raw
    if has_final != has_stop:
        # Half a ramp is not a ramp. Declaring an end speed without saying
        # when it is reached (or the reverse) has no reading that is not a
        # guess about the author's intent.
        raise ModelError(
            f"{name}: declare both final_rad_s and ramp_stop_s for a ramp, or "
            "neither for a constant speed"
        )
    if not has_final:
        if "ramp_start_s" in raw:
            raise ModelError(
                f"{name}: ramp_start_s is meaningless without a ramp"
            )
        return DrivenSpeed(initial_rad_s=initial)
    final = _number(raw["final_rad_s"], name=f"{name}.final_rad_s")
    start = _number(raw.get("ramp_start_s", 0.0), name=f"{name}.ramp_start_s")
    stop = _number(raw["ramp_stop_s"], name=f"{name}.ramp_stop_s")
    if stop <= start:
        raise ModelError(f"{name}.ramp_stop_s must be later than ramp_start_s")
    return DrivenSpeed(
        initial_rad_s=initial,
        final_rad_s=final,
        ramp_start_s=start,
        ramp_stop_s=stop,
    )


def parse_machines(raw_machines: Any) -> tuple[MachineSpec, ...]:
    """Validate the ``machines`` section; expansion happens in the loader."""

    if raw_machines is None:
        return ()
    if not isinstance(raw_machines, dict):
        raise ModelError("machines must be an object")
    if len(raw_machines) > MAX_MACHINES:
        raise ModelError(
            f"machines declares {len(raw_machines)} machines; the portable model "
            f"limit is {MAX_MACHINES}"
        )
    specs: list[MachineSpec] = []
    for machine_id in sorted(raw_machines):
        raw = raw_machines[machine_id]
        machine_id = _text(machine_id, name="machine id")
        if not isinstance(raw, dict):
            raise ModelError(f"machine {machine_id} must be an object")
        allowed = {
            "type",
            "terminals",
            "neutral",
            "pole_pairs",
            "stator_resistance_ohm",
            "stator_inductance_h",
            "d_axis_inductance_h",
            "q_axis_inductance_h",
            "flux_linkage_wb",
            "inertia_kg_m2",
            "viscous_friction_nm_s",
            "load_torque_nm",
            "initial_speed_rad_s",
            "initial_angle_rad",
            "driven_speed",
            "flux_map",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ModelError(
                f"machine {machine_id}: unknown field(s): {', '.join(unknown)}"
            )
        kind = raw.get("type")
        if kind not in MACHINE_KINDS:
            raise ModelError(
                f"machine {machine_id}.type must be one of {MACHINE_KINDS}"
            )
        driven = _parse_driven_speed(raw.get("driven_speed"), machine_id=machine_id)
        flux_map = _parse_flux_map(raw.get("flux_map"), machine_id=machine_id)
        required = [
            "terminals",
            "neutral",
            "pole_pairs",
            "stator_resistance_ohm",
        ]
        if flux_map is None:
            required += ["stator_inductance_h", "flux_linkage_wb"]
        else:
            # ONE FACT, ONE PLACE. psi_d(0,0) already IS the magnet flux, and
            # declared axis inductances are the map's slopes; accepting both
            # would let two statements of the same quantity disagree.
            duplicated = sorted(
                field
                for field in ("flux_linkage_wb", "d_axis_inductance_h",
                              "q_axis_inductance_h")
                if field in raw
            )
            if "stator_inductance_h" in raw:
                duplicated = duplicated + ["stator_inductance_h"]
            if duplicated:
                raise ModelError(
                    f"machine {machine_id}: flux_map already states the flux "
                    f"linkage and its slopes, so {', '.join(duplicated)} would "
                    "be a second, possibly disagreeing statement of the same "
                    "quantity; remove them or remove flux_map"
                )
        if driven is None:
            required.append("inertia_kg_m2")
        else:
            # A driven shaft does not integrate its own mechanical loop, so
            # inertia, friction, load torque and the initial speed have no
            # effect on anything. Accepting them silently would let a reader
            # believe a declared inertia was doing work. Fail closed instead.
            ignored = sorted(
                field
                for field in (
                    "inertia_kg_m2",
                    "viscous_friction_nm_s",
                    "load_torque_nm",
                    "initial_speed_rad_s",
                )
                if field in raw
            )
            if ignored:
                raise ModelError(
                    f"machine {machine_id}: driven_speed imposes the speed, so "
                    f"{', '.join(ignored)} would have no effect; remove "
                    "them or remove driven_speed"
                )
        for field in required:
            if field not in raw:
                raise ModelError(f"machine {machine_id}: missing required field {field}")

        terminals = raw.get("terminals")
        if (
            not isinstance(terminals, list)
            or len(terminals) != 3
            or any(not isinstance(item, str) or not item for item in terminals)
        ):
            raise ModelError(
                f"machine {machine_id}.terminals must list exactly three node names"
            )
        if len(set(terminals)) != 3:
            raise ModelError(f"machine {machine_id}.terminals must be distinct")
        neutral = _text(raw.get("neutral"), name=f"machine {machine_id}.neutral")
        if neutral in terminals:
            raise ModelError(
                f"machine {machine_id}.neutral must differ from the phase terminals"
            )

        pole_pairs = raw.get("pole_pairs")
        if isinstance(pole_pairs, bool) or not isinstance(pole_pairs, int) or pole_pairs < 1:
            raise ModelError(f"machine {machine_id}.pole_pairs must be an integer >= 1")

        inductance = (
            0.0
            if flux_map is not None
            else _number(
                raw.get("stator_inductance_h"),
                name=f"machine {machine_id}.stator_inductance_h",
                positive=True,
            )
        )
        # Saliency is accepted as a DECLARATION and refused as a capability:
        # a position-dependent inductance matrix would have to be recompiled
        # every integration segment, which the mode cache cannot express.
        d_axis = raw.get("d_axis_inductance_h")
        q_axis = raw.get("q_axis_inductance_h")
        if (d_axis is None) != (q_axis is None):
            raise ModelError(
                f"machine {machine_id}: declare both d_axis_inductance_h and "
                "q_axis_inductance_h or neither"
            )
        if flux_map is not None:
            # Nominal axis values recorded from the map's slopes AT ZERO
            # CURRENT: they are what the machine looks like unsaturated, used
            # for reporting and for the initial matrix. Every segment then
            # re-reads the slopes at its own operating point.
            d_value, q_value = flux_map.differential_inductances(0.0, 0.0)
            inductance = 0.5 * (d_value + q_value)
            flux_linkage = flux_map.flux(0.0, 0.0)[0]
        elif d_axis is not None:
            d_value = _number(d_axis, name=f"machine {machine_id}.d_axis_inductance_h", positive=True)
            q_value = _number(q_axis, name=f"machine {machine_id}.q_axis_inductance_h", positive=True)
            if d_value == q_value and d_value != inductance:
                raise ModelError(
                    f"machine {machine_id}: stator_inductance_h must equal the "
                    "declared axis inductances when they are equal"
                )
            if d_value != q_value:
                # SALIENT. The winding set becomes one magnetically coupled
                # group whose matrix depends on rotor position, so the
                # three-wire scalar equivalent no longer describes it and
                # declaring one alongside would be a second, disagreeing
                # statement of the same fact.
                if "stator_inductance_h" in raw and raw.get(
                    "stator_inductance_h"
                ) is not None and d_value != inductance and q_value != inductance:
                    pass  # stator_inductance_h is required by schema; kept as nominal
        else:
            d_value = inductance
            q_value = inductance

        # ZERO MAGNET FLUX IS A MACHINE, NOT AN ERROR. A synchronous
        # reluctance machine has no magnets at all: its entire torque comes
        # from the saliency. Refusing psi_f = 0 would put that machine out
        # of reach in the same way refusing Ld != Lq did. What must NOT be
        # accepted is a machine with no torque mechanism whatsoever, so the
        # gate is the physical one — magnets or saliency, at least one.
        if flux_map is None:
            flux_linkage = _number(
                raw.get("flux_linkage_wb"),
                name=f"machine {machine_id}.flux_linkage_wb",
            )
        if flux_map is None and flux_linkage < 0.0:
            raise ModelError(
                f"machine {machine_id}.flux_linkage_wb must be non-negative"
            )
        if flux_map is None and flux_linkage == 0.0 and d_value == q_value:
            raise ModelError(
                f"machine {machine_id}: no magnets (flux_linkage_wb = 0) and no "
                "saliency (d_axis_inductance_h == q_axis_inductance_h) leaves the "
                "machine with no torque mechanism at all; declare one or the other"
            )
        specs.append(
            MachineSpec(
                id=machine_id,
                kind=kind,
                terminals=(terminals[0], terminals[1], terminals[2]),
                neutral=neutral,
                pole_pairs=pole_pairs,
                stator_resistance_ohm=_number(
                    raw.get("stator_resistance_ohm"),
                    name=f"machine {machine_id}.stator_resistance_ohm",
                    positive=True,
                ),
                stator_inductance_h=inductance,
                d_axis_inductance_h=d_value,
                q_axis_inductance_h=q_value,
                flux_linkage_wb=flux_linkage,
                inertia_kg_m2=(
                    # Unused when the speed is imposed; carried as a finite
                    # placeholder so nothing downstream divides by None.
                    1.0
                    if driven is not None
                    else _number(
                        raw.get("inertia_kg_m2"),
                        name=f"machine {machine_id}.inertia_kg_m2",
                        positive=True,
                    )
                ),
                viscous_friction_nm_s=_number(
                    raw.get("viscous_friction_nm_s", 0.0),
                    name=f"machine {machine_id}.viscous_friction_nm_s",
                ),
                load_torque_nm=_number(
                    raw.get("load_torque_nm", 0.0),
                    name=f"machine {machine_id}.load_torque_nm",
                ),
                initial_speed_rad_s=(
                    driven.speed_at(0.0)
                    if driven is not None
                    else _number(
                        raw.get("initial_speed_rad_s", 0.0),
                        name=f"machine {machine_id}.initial_speed_rad_s",
                    )
                ),
                initial_angle_rad=_number(
                    raw.get("initial_angle_rad", 0.0),
                    name=f"machine {machine_id}.initial_angle_rad",
                ),
                emf_ids=tuple(f"{machine_id}.{label}.e" for label, _ in PHASE_OFFSETS),
                current_expressions=tuple(
                    f"i({machine_id}.{label}.l)" for label, _ in PHASE_OFFSETS
                ),
                driven_speed=driven,
                flux_map=flux_map,
            )
        )
    return tuple(specs)


def machine_branch_definitions(spec: MachineSpec) -> list[dict[str, Any]]:
    """Raw component records the machine expands into, in electrical order.

    terminal --R_s-- internal --e(t)-- internal --L_eq-- neutral

    Series elements commute, and the ORDER IS CHOSEN so the windings — not
    the EMF sources — meet at the star point. A three-wire machine leaves
    that point floating, and a floating star of plain inductors is exactly
    what the floating-star index reduction recognizes; a star of voltage sources would
    present the same state invariant as an algebraic row and the topology
    would be singular.
    """

    records: list[dict[str, Any]] = []
    # A salient machine's self inductances are the diagonal of its
    # position-dependent matrix, evaluated at the initial angle; a
    # non-salient one keeps the scalar three-wire equivalent so its shipped
    # models stay byte-identical.
    if spec.is_salient:
        diagonal = salient_inductance_matrix(spec, spec.initial_angle_rad).diagonal()
    else:
        diagonal = None
    for index, (terminal, (label, _offset)) in enumerate(
        zip(spec.terminals, PHASE_OFFSETS, strict=True)
    ):
        resistance_node = f"{spec.id}.{label}.n1"
        emf_node = f"{spec.id}.{label}.n2"
        records.append(
            {
                "id": f"{spec.id}.{label}.r",
                "type": "resistor",
                "p": terminal,
                "n": resistance_node,
                "value": f"{spec.stator_resistance_ohm!r} ohm",
            }
        )
        records.append(
            {
                "id": f"{spec.id}.{label}.e",
                "type": "voltage_source",
                "p": resistance_node,
                "n": emf_node,
                # A benign, valid placeholder: the driver rewrites amplitude,
                # frequency and phase from the mechanical state at every
                # integration segment, so this value is never integrated.
                "waveform": {
                    "type": "sine",
                    "amplitude": "0 V",
                    "frequency": "1 Hz",
                },
            }
        )
        records.append(
            {
                "id": f"{spec.id}.{label}.l",
                "type": "inductor",
                "p": emf_node,
                "n": spec.neutral,
                "value": (
                    f"{float(diagonal[index])!r} H"
                    if diagonal is not None
                    else f"{spec.stator_inductance_h!r} H"
                ),
                "initial_current": "0 A",
            }
        )
    return records


def machine_coupling_definitions(spec: MachineSpec) -> list[dict[str, Any]]:
    """Raw coupling records for a SALIENT machine's winding group.

    A non-salient machine returns none: its three-wire scalar equivalent
    already folds the mutual term into the self term, and introducing an
    explicit group there would change a shipped model's physics hash for no
    physical reason.

    The values here are the matrix AT THE INITIAL ROTOR ANGLE. They are a
    valid starting point, not a constant: the driver replaces the whole
    matrix at every integration segment. They still have to be physically
    admissible on their own, because the loader's positive-definiteness gate
    runs on exactly these numbers.
    """

    if not spec.is_salient:
        return []
    matrix = salient_inductance_matrix(spec, spec.initial_angle_rad)
    windings = spec.winding_ids
    pairs = ((0, 1), (1, 2), (0, 2))
    return [
        {
            "inductors": [windings[row], windings[column]],
            "mutual": f"{float(matrix[row, column])!r} H",
        }
        for row, column in pairs
    ]


def back_emf_waveform(
    spec: MachineSpec,
    *,
    electrical_speed: float,
    electrical_angle: float,
    segment_start: float,
    phase_offset: float,
) -> tuple[float, float, float]:
    """(amplitude, frequency_hz, phase_rad) of one phase's motional EMF.

    The phase flux linkage of a surface-magnet machine is
    ``psi_f * cos(theta_e - offset)``, so its motional voltage is
    ``-omega_e * psi_f * sin(theta_e - offset)``. With the speed held over a
    segment the angle still advances exactly, ``theta_e(t) = theta_0 +
    omega_e * (t - t0)``, which is a pure sinusoid the kernel integrates
    without approximation:

        e(t) = A * sin(2*pi*f*t + phi),
        A = -omega_e * psi_f,  f = omega_e / (2*pi),
        phi = theta_0 - offset - omega_e * t0
    """

    amplitude = -electrical_speed * spec.flux_linkage_wb
    frequency = electrical_speed / math.tau
    phase = electrical_angle - phase_offset - electrical_speed * segment_start
    return amplitude, frequency, phase


def park_currents(
    currents: tuple[float, float, float], electrical_angle: float
) -> tuple[float, float]:
    """(i_d, i_q) of a phase-current triple, amplitude-invariant convention."""

    i_d = (2.0 / 3.0) * sum(
        current * math.cos(electrical_angle - offset)
        for current, (_label, offset) in zip(currents, PHASE_OFFSETS, strict=True)
    )
    i_q = -(2.0 / 3.0) * sum(
        current * math.sin(electrical_angle - offset)
        for current, (_label, offset) in zip(currents, PHASE_OFFSETS, strict=True)
    )
    return i_d, i_q


def saturated_axis_inductances(
    spec: "MachineSpec",
    currents: tuple[float, float, float],
    electrical_angle: float,
) -> tuple[float, float]:
    """Local (Ld, Lq) of a mapped machine at this operating point."""

    if spec.flux_map is None:
        return spec.d_axis_inductance_h, spec.q_axis_inductance_h
    i_d, i_q = park_currents(currents, electrical_angle)
    return spec.flux_map.differential_inductances(i_d, i_q)


def mapped_torque(
    spec: "MachineSpec",
    currents: tuple[float, float, float],
    electrical_angle: float,
) -> float:
    """Torque straight from the flux map: T = 1.5*p*(psi_d*i_q - psi_q*i_d).

    The general form, valid saturated or not. The familiar
    ``psi_f*i_q + (Ld-Lq)*i_d*i_q`` is what this collapses to when the flux
    is linear in current — which is the equivalence the tests assert rather
    than assume.
    """

    if spec.flux_map is None:
        return salient_torque(spec, currents, electrical_angle)
    i_d, i_q = park_currents(currents, electrical_angle)
    psi_d, psi_q = spec.flux_map.flux(i_d, i_q)
    return 1.5 * spec.pole_pairs * (psi_d * i_q - psi_q * i_d)


def salient_inductance_matrix(
    spec: "MachineSpec",
    electrical_angle: float,
    currents: "tuple[float, float, float] | None" = None,
) -> Any:
    """abc inductance matrix of a salient machine at one rotor position.

    A salient rotor presents a different reluctance to the d and q axes, so
    the phase inductances breathe at TWICE the electrical angle:

        L_aa = L_A + L_B*cos(2*theta)
        L_ab = -L_A/2 + L_B*cos(2*theta - 2*pi/3)      (and cyclic)

    with the axis inductances recovered as

        L_d = (3/2)*(L_A + L_B),  L_q = (3/2)*(L_A - L_B)

    so ``L_A = (L_d + L_q)/3`` and ``L_B = (L_d - L_q)/3``. Setting
    ``L_d = L_q`` collapses L_B to zero and the matrix back to the constant
    one the non-salient machine uses, which is the continuity the tests
    check rather than assume.

    THE MATRIX IS EVALUATED, NOT INTEGRATED. The kernel holds it across one
    segment exactly as it holds the speed, so the splitting error is of the
    same kind and shrinks the same way — measurable by tightening the
    segment, not merely declared.
    """

    import numpy as np

    if spec.flux_map is not None and currents is not None:
        # SATURATION ENTERS AS THE LOCAL SLOPE. The matrix is rebuilt from
        # d(psi)/d(i) at this segment's operating point, so a machine driven
        # deeper into saturation genuinely stiffens — which constant Ld/Lq
        # cannot express at all.
        axis_d, axis_q = saturated_axis_inductances(spec, currents, electrical_angle)
    else:
        axis_d, axis_q = spec.d_axis_inductance_h, spec.q_axis_inductance_h
    l_a = (axis_d + axis_q) / 3.0
    l_b = (axis_d - axis_q) / 3.0
    two_theta = 2.0 * electrical_angle
    shift = 2.0 * math.pi / 3.0
    self_terms = (
        l_a + l_b * math.cos(two_theta),
        l_a + l_b * math.cos(two_theta + shift),
        l_a + l_b * math.cos(two_theta - shift),
    )
    # Mutual between a-b, b-c, c-a. The -L_A/2 term is the ordinary
    # three-phase coupling (negative: one phase's flux opposes the others);
    # the L_B term is the saliency riding on top of it.
    mutual_ab = -0.5 * l_a + l_b * math.cos(two_theta - shift)
    mutual_bc = -0.5 * l_a + l_b * math.cos(two_theta)
    mutual_ca = -0.5 * l_a + l_b * math.cos(two_theta + shift)
    matrix = np.array(
        [
            [self_terms[0], mutual_ab, mutual_ca],
            [mutual_ab, self_terms[1], mutual_bc],
            [mutual_ca, mutual_bc, self_terms[2]],
        ],
        dtype=float,
    )
    # THE ZERO SEQUENCE IS SINGULAR, AND THAT IS PHYSICS, NOT AN ERROR.
    # The all-ones eigenvector of this matrix carries L_A + 2*(-L_A/2) = 0:
    # a machine described this way offers NO inductance to a zero-sequence
    # current. In a three-wire machine that current has nowhere to flow, so
    # the value is unobservable — but the kernel inverts the matrix, and a
    # singular one has no inverse.
    #
    # The zero-sequence entry is therefore set to an explicit positive value.
    # It changes only the eigenvalue along [1,1,1], leaving the d and q axes
    # (which are orthogonal to it) exactly as declared. `L_A` is used because
    # it keeps the matrix well conditioned at every rotor angle. The claim
    # that the result does not depend on this number is a TEST, not a
    # comment: with a floating star, sweeping it changes no current.
    #
    # A machine whose neutral were tied would make this observable, and the
    # loader keeps the star floating for exactly that reason.
    ones = np.ones((3, 3), dtype=float) / 3.0
    return matrix + l_a * ones


def salient_torque(
    spec: "MachineSpec",
    currents: tuple[float, float, float],
    electrical_angle: float,
) -> float:
    """Torque of a salient machine: magnet term plus reluctance term.

    ``T = (3/2)*p*[psi_f*i_q + (L_d - L_q)*i_d*i_q]``

    The second term is the whole point. A machine with no magnets at all
    (``psi_f = 0``) produces torque ONLY through it, which is why a model
    that forces L_d = L_q reports exactly zero for such a machine.
    """

    shift = 2.0 * math.pi / 3.0
    # Amplitude-invariant Park, same convention as the analysis layer.
    i_d = (2.0 / 3.0) * sum(
        current * math.cos(electrical_angle - offset)
        for current, offset in zip(currents, (0.0, shift, -shift), strict=True)
    )
    i_q = -(2.0 / 3.0) * sum(
        current * math.sin(electrical_angle - offset)
        for current, offset in zip(currents, (0.0, shift, -shift), strict=True)
    )
    magnet = spec.flux_linkage_wb * i_q
    reluctance = (spec.d_axis_inductance_h - spec.q_axis_inductance_h) * i_d * i_q
    return 1.5 * spec.pole_pairs * (magnet + reluctance)


def electromagnetic_torque(
    spec: MachineSpec,
    currents: tuple[float, float, float],
    electrical_angle: float,
) -> float:
    """Torque of a non-salient machine from the instantaneous phase currents.

    With the amplitude-invariant Park convention used across the analysis
    layer, ``i_q = -(2/3) * sum(i_x * sin(theta_e - offset_x))`` and
    ``T_e = (3/2) * p * psi_f * i_q``, which reduces to the sum below. The
    identity ``sum(e_x * i_x) == T_e * omega_mechanical`` holds exactly for
    the EMF above, and the test suite asserts it.
    """

    projection = sum(
        current * math.sin(electrical_angle - offset)
        for current, (_label, offset) in zip(currents, PHASE_OFFSETS, strict=True)
    )
    return -spec.pole_pairs * spec.flux_linkage_wb * projection
