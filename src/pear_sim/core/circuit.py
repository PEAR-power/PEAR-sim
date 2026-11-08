from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import math
import re
from typing import Any, Iterable, Iterator

import numpy as np

from pear_sim.errors import ModelError, TopologyError
from pear_sim.spec.signals import is_declared_signal, is_machine_signal
from pear_sim.limits import MAX_ACTIVE_SET_CANDIDATES, MAX_IDEAL_DIODES
from pear_sim.spec.model import Component, Model

# Distinct switch configurations whose active-set search is kept for replay.
MAX_MODE_PLANS = 4096

# A thyristor is a latching branch: it closes ONLY on a gate edge while
# forward biased, and reopens ONLY when its own current reaches zero. Both
# halves matter — without the first it is a diode, without the second it is a
# switch — and neither is expressible by the gate signal alone, which is why
# the latch is kernel state and not a control-layer trick.
LATCHING_TYPES = frozenset({"thyristor"})
LATCHING_OR_DIODE = frozenset({"diode"}) | LATCHING_TYPES
SWITCHING_BRANCH_TYPES = frozenset({"ideal_switch"}) | LATCHING_OR_DIODE



_OUTPUT_RE = re.compile(r"^(v|i)\(([A-Za-z_][A-Za-z0-9_.:-]*)\)$")
STATE_CONSTRAINT_TOLERANCE = 1e-9  # A; deliberately independent of ODE atol.


@dataclass(frozen=True)
class Topology:
    gates: tuple[tuple[str, bool], ...]
    diodes: tuple[tuple[str, bool], ...]
    constrained_inductors: tuple[str, ...] = ()
    saturated_inductors: tuple[str, ...] = ()

    def gate_dict(self) -> dict[str, bool]:
        return dict(self.gates)

    def diode_dict(self) -> dict[str, bool]:
        return dict(self.diodes)


class CompiledMode:
    def __init__(self, circuit: "Circuit", topology: Topology):
        self.circuit = circuit
        self.topology = topology
        gates = topology.gate_dict()
        diode_states = topology.diode_dict()
        constrained_inductors = frozenset(topology.constrained_inductors)
        # The bound guards the ENUMERATED search: freezing a plain inductor is a
        # guess the active-set loop makes and must pay for. A coupled winding is
        # different — it is frozen because its own series path is open, which is
        # read off the topology rather than searched for, so a transformer with
        # nine windings does not blow a budget meant for four guesses.
        enumerated = constrained_inductors - {
            component.id
            for component in circuit.inductors
            if component.id in circuit.coupling_by_inductor
        }
        if len(enumerated) > 4:
            raise TopologyError(
                "State reduction supports at most four constrained plain "
                "inductors per mode (portable bound on the active-set search)"
            )
        unknown_constraints = constrained_inductors - {
            component.id for component in circuit.inductors
        }
        if unknown_constraints:
            raise TopologyError(
                "State constraints reference non-inductors: "
                + ", ".join(sorted(unknown_constraints))
            )
        self.constrained_inductors = tuple(sorted(constrained_inductors))
        self.constrained_state_indices = tuple(
            circuit.state_index[component_id] for component_id in self.constrained_inductors
        )
        self.active_state_indices = tuple(
            index
            for index in range(len(circuit.state_components))
            if index not in self.constrained_state_indices
        )
        gates_signature = ",".join(
            f"{name}={'on' if value else 'off'}" for name, value in topology.gates
        )
        diodes_signature = ",".join(
            f"{name}={'on' if value else 'off'}" for name, value in topology.diodes
        )
        constraints_signature = ",".join(self.constrained_inductors) or "none"
        self.mode_signature = (
            f"gates[{gates_signature}];diodes[{diodes_signature}];"
            f"constrained_inductors[{constraints_signature}]"
        )
        saturated = frozenset(topology.saturated_inductors)
        unknown_saturated = saturated - {
            component.id for component in circuit.sat_inductors
        }
        if unknown_saturated:
            raise TopologyError(
                "Saturation flags reference non-saturable inductors: "
                + ", ".join(sorted(unknown_saturated))
            )
        self.saturated_inductors = tuple(sorted(saturated))
        if self.saturated_inductors:
            self.mode_signature += (
                f";saturated_inductors[{','.join(self.saturated_inductors)}]"
            )

        branch_components: list[Component] = []
        for component in circuit.components:
            if component.type in {"voltage_source", "capacitor"}:
                branch_components.append(component)
            elif component.type == "ideal_switch" and gates[component.control or ""]:
                branch_components.append(component)
            elif component.type in LATCHING_OR_DIODE and diode_states[component.id]:
                branch_components.append(component)
            elif (
                component.type == "inductor"
                and component.id in constrained_inductors
            ):
                # A zero-current inductor cutset is index-reduced to a 0 V algebraic
                # branch. KCL must independently force this branch current to zero;
                # stable_mode validates that condition before accepting the mode.
                branch_components.append(component)

        self.branch_components = tuple(branch_components)
        self.branch_index = {
            component.id: len(circuit.nodes) + index for index, component in enumerate(branch_components)
        }
        size = len(circuit.nodes) + len(branch_components)
        if size == 0:
            raise TopologyError("Circuit has no algebraic variables")
        self.source_count = len(circuit.waveform_sources)
        matrix = np.zeros((size, size), dtype=float)
        state_rhs = np.zeros((size, len(circuit.state_components)), dtype=float)
        constant_rhs = np.zeros(size, dtype=float)
        source_rhs = np.zeros((size, self.source_count), dtype=float)

        for component in circuit.components:
            p = circuit.node_index.get(component.p)
            n = circuit.node_index.get(component.n)
            if component.type == "resistor":
                conductance = 1.0 / float(component.value)
                _stamp_conductance(matrix, p, n, conductance)
            elif component.type == "current_source":
                if component.waveform is not None:
                    source_index = circuit.waveform_index[component.id]
                    _stamp_known_current(source_rhs[:, source_index], p, n, 1.0)
                else:
                    _stamp_known_current(constant_rhs, p, n, float(component.value))
            elif component.type == "inductor":
                if component.id not in constrained_inductors:
                    state_index = circuit.state_index[component.id]
                    _stamp_known_current(state_rhs[:, state_index], p, n, 1.0)
            elif component.type == "sat_inductor":
                state_index = circuit.state_index[component.id]
                _stamp_known_current(state_rhs[:, state_index], p, n, 1.0)

        induced_rows = _induced_voltage_rows(circuit, constrained_inductors)
        gamma_rows = _active_gamma_rows(circuit, constrained_inductors)
        for branch_offset, component in enumerate(branch_components):
            branch = len(circuit.nodes) + branch_offset
            row = branch
            p = circuit.node_index.get(component.p)
            n = circuit.node_index.get(component.n)
            _stamp_branch_incidence(matrix, p, n, branch)
            if p is not None:
                matrix[row, p] += 1.0
            if n is not None:
                matrix[row, n] -= 1.0
            if component.type == "voltage_source":
                if component.waveform is not None:
                    source_rhs[row, circuit.waveform_index[component.id]] = 1.0
                else:
                    constant_rhs[row] = float(component.value)
            elif component.type == "capacitor":
                state_rhs[row, circuit.state_index[component.id]] = 1.0
            elif component.type in SWITCHING_BRANCH_TYPES:
                # Piecewise-linear conduction: v_p - v_n - R_on * i = V_f, with
                # i the branch current p -> n (anode -> cathode for a diode).
                # Both terms are zero for an ideal device, leaving the 0 V short.
                if component.on_resistance:
                    matrix[row, branch] -= float(component.on_resistance)
                if component.forward_voltage:
                    constant_rhs[row] += float(component.forward_voltage)
            elif component.id in induced_rows:
                # A frozen COUPLED winding is not a short. Its current is zero,
                # but the flux its neighbours carry still induces a terminal
                # voltage — that is what an open transformer secondary shows,
                # and pinning it at 0 V would be a modelling error dressed up
                # as a reduction. With di/dt of the frozen winding zero, the
                # active members obey v_A = L_AA di_A/dt, so
                #     v_f = L_fA inv(L_AA) v_A,
                # a plain linear relation between terminal voltages that stamps
                # like a voltage-controlled source.
                for active_component, coefficient in induced_rows[component.id]:
                    ap = circuit.node_index.get(active_component.p)
                    an = circuit.node_index.get(active_component.n)
                    if ap is not None:
                        matrix[row, ap] -= coefficient
                    if an is not None:
                        matrix[row, an] += coefficient
            # A constrained PLAIN inductor intentionally keeps the default 0 V RHS.

        # --- structurally shorted node classes ------------------------------
        # Branches whose mode equation pins v_p - v_n = 0 EXACTLY: closed
        # ideal switches, conducting diodes/latches, and frozen PLAIN
        # inductors (a frozen COUPLED winding pins an induced voltage, not
        # zero). A blocking diode whose terminals land in one class has an
        # identically-zero voltage guard BY STRUCTURE. Deciding that from the
        # solved affine row instead makes the classification hostage to
        # platform-dependent elimination dust — the constant-guard livelock knife, which
        # cut differently on ubuntu (dust in the constant), macOS (constant
        # dust on one diode, exact zeros elsewhere) and windows (dust in the
        # state coefficients too).
        shorted_parent: dict[str, str] = {}

        def shorted_find(node: str) -> str:
            while shorted_parent.get(node, node) != node:
                node = shorted_parent[node]
            return node

        for component in circuit.components:
            # A conducting branch with on-resistance or a forward voltage does
            # NOT pin v_p == v_n, so it never joins a shorted class.
            pins_zero_volts = (
                (
                    component.type == "ideal_switch"
                    and gates[component.control or ""]
                    and component.is_ideal_conductor
                )
                or (
                    component.type in LATCHING_OR_DIODE
                    and diode_states[component.id]
                    and component.is_ideal_conductor
                )
                or (
                    component.type == "inductor"
                    and component.id in constrained_inductors
                    and component.id not in circuit.coupling_by_inductor
                )
            )
            if pins_zero_volts:
                shorted_parent[shorted_find(component.p)] = shorted_find(
                    component.n
                )
        self._shorted_root = {
            node: shorted_find(node) for node in shorted_parent
        }

        # --- floating inductor-star nodes (three-wire loads) ----------------
        # A node whose every incident component is an inductor has a KCL row
        # with no algebraic unknown: sum(+/- i_k) = 0 is an exact linear
        # INVARIANT of the states, not an equation for the node voltage. The
        # degenerate row is replaced by the invariant's time derivative,
        # sum(+/- (v_pk - v_nk)/L_k) = 0, which pins the star-point voltage.
        # The invariant itself is validated against the initial state by the
        # simulator and is preserved exactly by Runge-Kutta integration
        # (linear conservation law of the replaced vector field).
        # A frozen branch — plain or coupled — offers no support of its own
        # here. A plain one is a 0 V short and a coupled one carries an induced
        # voltage, but both pin only the DIFFERENCE across their terminals, so
        # a node whose only company is a frozen branch is still unpinned unless
        # the branch's other end is. Counting them as support was tried and it
        # floats an isolated secondary star: v(ya) and v(yn) then move together
        # with nothing to object.
        constrained_branch_columns = [
            self.branch_index[component_id]
            for component_id in self.constrained_inductors
        ]
        star_constraints: list[tuple[str, tuple[tuple[int, float], ...]]] = []
        for node_row in range(len(circuit.nodes)):
            row_support = matrix[node_row, :].copy()
            # A frozen (zero-current constrained) inductor branch offers no real
            # algebraic support to its node: its current is forced to zero
            # through its own terminal chain. Ignore those columns when
            # deciding whether the node is a reducible inductor star.
            for column in constrained_branch_columns:
                row_support[column] = 0.0
            if np.any(row_support):
                continue
            node_name = circuit.nodes[node_row]
            star_inductors: list[Component] = []
            offending: Component | None = None
            for component in circuit.components:
                if node_name not in (component.p, component.n):
                    continue
                if component.type == "voltage_probe":
                    continue
                if (
                    component.type == "ideal_switch"
                    and not gates[component.control or ""]
                ):
                    continue
                if (
                    component.type in LATCHING_OR_DIODE
                    and not diode_states[component.id]
                ):
                    continue
                if (
                    component.type == "inductor"
                    and component.id in constrained_inductors
                ):
                    # Frozen member: contributes exactly zero current here.
                    continue
                if component.type == "inductor":
                    # Coupled windings belong here too: a star of transformer
                    # secondaries is a floating star like any other, it just
                    # needs the mutual terms in the invariant's derivative.
                    star_inductors.append(component)
                    continue
                offending = component
            if offending is not None or len(star_inductors) < 2:
                if np.any(matrix[node_row, :]):
                    # Supported by frozen zero-current branches: the plain KCL
                    # row is exactly what forces those branch currents to zero
                    # (the DCM idle mechanism) — leave it untouched.
                    continue
                if offending is not None:
                    error = TopologyError(
                        f"Node {node_name!r} has no algebraic support in this "
                        "topology; only a floating star point of plain, "
                        "uncoupled inductors is reduced in v0.1 "
                        f"(offending component: {offending.id!r})"
                    )
                    error.floating_node = node_name
                    raise error
                touches_switch = any(
                    component.type == "ideal_switch" and node_name in (component.p, component.n)
                    for component in circuit.components
                )
                error = TopologyError(
                    f"Node {node_name!r} has no algebraic support in this "
                    "topology and is not a reducible inductor star point"
                    + (
                        "; the node has no conduction path while the switches touching "
                        "it are open — an ideal_switch has no body diode, so add an "
                        "antiparallel diode across it (e.g. for dead time)"
                        if touches_switch
                        else ""
                    )
                )
                error.floating_node = node_name
                raise error
            if np.any(matrix[node_row, :]):
                # Star of >=2 live inductors plus frozen branches: replace the
                # overdetermined KCL row with the invariant derivative; each
                # frozen branch's zero current is still forced through its own
                # terminal chain on the other side.
                matrix[node_row, :] = 0.0
            state_rhs[node_row, :] = 0.0
            constant_rhs[node_row] = 0.0
            source_rhs[node_row, :] = 0.0
            constraint_terms: list[tuple[int, float]] = []
            for component in star_inductors:
                sign = 1.0 if component.n == node_name else -1.0
                # sum(+/- di_k/dt) = 0. For a plain inductor di/dt is v_k/L_k,
                # a single term; for a coupled winding it is a row of the
                # group's inv(L) applied to every active winding's terminal
                # voltage. The plain case is the one-by-one special case.
                if component.id in gamma_rows:
                    terms = gamma_rows[component.id]
                else:
                    terms = ((component, 1.0 / float(component.value)),)
                for source_component, coefficient in terms:
                    p_index = circuit.node_index.get(source_component.p)
                    n_index = circuit.node_index.get(source_component.n)
                    if p_index is not None:
                        matrix[node_row, p_index] += sign * coefficient
                    if n_index is not None:
                        matrix[node_row, n_index] -= sign * coefficient
                constraint_terms.append(
                    (circuit.state_index[component.id], sign)
                )
            star_constraints.append((node_name, tuple(constraint_terms)))
        self.star_constraints = tuple(star_constraints)

        try:
            condition = float(np.linalg.cond(matrix))
            if not math.isfinite(condition) or condition > 1e14:
                raise np.linalg.LinAlgError(f"condition number {condition:.3e}")
            self.solution_from_state = np.linalg.solve(matrix, state_rhs)
            self.solution_constant = np.linalg.solve(matrix, constant_rhs)
            self.solution_from_sources = np.linalg.solve(matrix, source_rhs)
        except np.linalg.LinAlgError as exc:
            closed_switches = [
                component.id
                for component in circuit.components
                if (component.type == "ideal_switch" and gates[component.control or ""])
                or (
                    component.type in LATCHING_OR_DIODE
                    and diode_states[component.id]
                )
            ]
            raise TopologyError(
                "Singular or inconsistent ideal topology; closed devices: "
                + (", ".join(closed_switches) or "none")
            ) from exc

        state_count = len(circuit.state_components)
        self.a = np.zeros((state_count, state_count), dtype=float)
        self.c = np.zeros(state_count, dtype=float)
        self.b = np.zeros((state_count, self.source_count), dtype=float)
        for component in circuit.state_components:
            row = circuit.state_index[component.id]
            if component.type == "capacitor":
                branch = self.branch_index[component.id]
                self.a[row, :] = self.solution_from_state[branch, :] / float(component.value)
                self.c[row] = self.solution_constant[branch] / float(component.value)
                self.b[row, :] = self.solution_from_sources[branch, :] / float(component.value)
            else:
                if component.id in constrained_inductors:
                    # The global state identity is preserved while this mode removes
                    # the inductor from the active differential state set.
                    continue
                p = circuit.node_index.get(component.p)
                n = circuit.node_index.get(component.n)
                voltage_from_state = np.zeros(state_count)
                voltage_constant = 0.0
                voltage_from_sources = np.zeros(self.source_count)
                if p is not None:
                    voltage_from_state += self.solution_from_state[p, :]
                    voltage_constant += self.solution_constant[p]
                    voltage_from_sources += self.solution_from_sources[p, :]
                if n is not None:
                    voltage_from_state -= self.solution_from_state[n, :]
                    voltage_constant -= self.solution_constant[n]
                    voltage_from_sources -= self.solution_from_sources[n, :]
                inductance = float(component.value)
                if (
                    component.type == "sat_inductor"
                    and component.id in saturated
                ):
                    inductance = float(component.sat_inductance)
                self.a[row, :] = voltage_from_state / inductance
                self.c[row] = voltage_constant / inductance
                self.b[row, :] = voltage_from_sources / inductance

        # Magnetically coupled windings are solved as GROUPS, not pairs: a
        # three-winding transformer is declared as three pairwise mutuals, and
        # every winding reachable through a mutual shares one inductance
        # matrix whose inverse maps the branch voltages to di/dt. Two windings
        # is the special case, not the contract.
        for group in circuit.coupling_groups:
            all_members = [circuit.component_by_id[member_id] for member_id in group]
            members = [
                item for item in all_members if item.id not in constrained_inductors
            ]
            if not members:
                # Every winding of this group is frozen: nothing carries flux,
                # so there is no induced voltage either and the 0 V branches
                # left by the stamping loop are the right answer.
                continue
            rows = tuple(circuit.state_index[item.id] for item in members)
            size = len(members)
            voltage_state = np.zeros((size, state_count))
            voltage_const = np.zeros(size)
            voltage_sources = np.zeros((size, self.source_count))
            for slot, component in enumerate(members):
                p = circuit.node_index.get(component.p)
                n = circuit.node_index.get(component.n)
                if p is not None:
                    voltage_state[slot] += self.solution_from_state[p, :]
                    voltage_const[slot] += self.solution_constant[p]
                    voltage_sources[slot] += self.solution_from_sources[p, :]
                if n is not None:
                    voltage_state[slot] -= self.solution_from_state[n, :]
                    voltage_const[slot] -= self.solution_constant[n]
                    voltage_sources[slot] -= self.solution_from_sources[n, :]
            inductance_matrix = circuit.coupling_matrix(group)
            if size != len(all_members):
                # Restricted to the windings that still carry current. With
                # di/dt of the frozen ones pinned at zero their columns leave
                # the equation entirely, so the reduced system is exactly
                # v_A = L_AA di_A/dt — and L_AA is a principal submatrix of a
                # positive definite matrix, hence positive definite itself.
                slots = [
                    index
                    for index, item in enumerate(all_members)
                    if item.id not in constrained_inductors
                ]
                inductance_matrix = inductance_matrix[np.ix_(slots, slots)]
            # di/dt = inv(L) @ v. The loader proves L is symmetric positive
            # definite (Cholesky), so it is invertible and the stored energy
            # 0.5 i^T L i is positive for every nonzero current set.
            gamma = np.linalg.inv(inductance_matrix)
            block_a = gamma @ voltage_state
            block_c = gamma @ voltage_const
            block_b = gamma @ voltage_sources
            for slot, row in enumerate(rows):
                self.a[row, :] = block_a[slot, :]
                self.c[row] = block_c[slot]
                self.b[row, :] = block_b[slot, :]

    def _source_vector(self, time: float | None) -> np.ndarray:
        if time is None:
            # Fail closed: silently evaluating waveform sources at an implicit
            # time would produce plausible but wrong algebra.
            raise ModelError(
                "Evaluation time is required for a circuit with waveform sources"
            )
        return self.circuit.source_values(time)

    def algebraic(self, state: np.ndarray, time: float | None = None) -> np.ndarray:
        values = self.solution_from_state @ state + self.solution_constant
        if self.source_count:
            values = values + self.solution_from_sources @ self._source_vector(time)
        return values

    def derivative(self, time: float, state: np.ndarray) -> np.ndarray:
        if self.source_count:
            return self.a @ state + self.c + self.b @ self.circuit.source_values(time)
        return self.a @ state + self.c

    def _algebraic_row_affine(
        self, index: int | None
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """(state coefficients, constant, source coefficients) of one solution row."""

        if index is None:
            return (
                np.zeros(len(self.circuit.state_components)),
                0.0,
                np.zeros(self.source_count),
            )
        return (
            self.solution_from_state[index, :].copy(),
            float(self.solution_constant[index]),
            self.solution_from_sources[index, :].copy(),
        )

    def component_voltage_affine(
        self, component: Component
    ) -> tuple[np.ndarray, float, np.ndarray]:
        p_coeff, p_const, p_src = self._algebraic_row_affine(
            self.circuit.node_index.get(component.p)
        )
        n_coeff, n_const, n_src = self._algebraic_row_affine(
            self.circuit.node_index.get(component.n)
        )
        return p_coeff - n_coeff, p_const - n_const, p_src - n_src

    def diode_guard_affine(
        self, diode: Component, conducting: bool
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """The guard's exact affine map: value = coeffs.state + const + src.u(t)."""

        if conducting:
            return self._algebraic_row_affine(self.branch_index[diode.id])
        # A blocking diode turns on when its voltage reaches the forward drop.
        coefficients, constant, sources = self.component_voltage_affine(diode)
        return coefficients, constant - float(diode.forward_voltage), sources

    def nodes_structurally_shorted(self, p: str, n: str) -> bool:
        """True when this mode's zero-voltage branches force v(p) == v(n).

        Exact graph reasoning over closed switches, conducting diodes and
        frozen plain inductors — no solved matrix involved, so the answer is
        the same on every platform's arithmetic.
        """

        return self._shorted_root.get(p, p) == self._shorted_root.get(n, n)

    def component_current_affine(
        self, component: Component
    ) -> tuple[np.ndarray, float, np.ndarray]:
        state_count = len(self.circuit.state_components)
        if component.type == "resistor":
            coefficients, constant, sources = self.component_voltage_affine(component)
            value = float(component.value)
            return coefficients / value, constant / value, sources / value
        if component.type in {"inductor", "sat_inductor"}:
            coefficients = np.zeros(state_count)
            coefficients[self.circuit.state_index[component.id]] = 1.0
            return coefficients, 0.0, np.zeros(self.source_count)
        if component.type == "current_source":
            sources = np.zeros(self.source_count)
            if component.waveform is not None:
                sources[self.circuit.waveform_index[component.id]] = 1.0
                return np.zeros(state_count), 0.0, sources
            return np.zeros(state_count), float(component.value), sources
        if component.id in self.branch_index:
            return self._algebraic_row_affine(self.branch_index[component.id])
        if component.type in SWITCHING_BRANCH_TYPES:
            return np.zeros(state_count), 0.0, np.zeros(self.source_count)
        raise ModelError(f"Current output is not implemented for {component.type}")

    def output_affine(
        self, expression: str
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """Exact affine map of a recorded output over (state, 1, u(t))."""

        kind, target = self.circuit.output_targets[expression]
        if kind == "node_voltage":
            if target == self.circuit.ground:
                return self._algebraic_row_affine(None)
            return self._algebraic_row_affine(self.circuit.node_index[target])
        component = self.circuit.component_by_id[target]
        if kind == "component_voltage":
            return self.component_voltage_affine(component)
        return self.component_current_affine(component)

    def constrained_branch_current(
        self,
        component_id: str,
        state: np.ndarray,
        algebraic: np.ndarray | None = None,
    ) -> float:
        """Return the hidden 0 V branch current of a constrained inductor."""

        if component_id not in self.constrained_inductors:
            raise ModelError(f"Inductor {component_id!r} is not constrained in this mode")
        values = self.algebraic(state) if algebraic is None else algebraic
        return float(values[self.branch_index[component_id]])

    def constrained_branch_current_affine(
        self,
        component_id: str,
    ) -> tuple[np.ndarray, float]:
        """Return the hidden branch-current affine map over the global state."""

        if component_id not in self.constrained_inductors:
            raise ModelError(f"Inductor {component_id!r} is not constrained in this mode")
        branch = self.branch_index[component_id]
        return self.solution_from_state[branch, :].copy(), float(
            self.solution_constant[branch]
        )

    def node_voltage(
        self,
        node: str,
        state: np.ndarray,
        algebraic: np.ndarray | None = None,
        time: float | None = None,
    ) -> float:
        if node == self.circuit.ground:
            return 0.0
        values = self.algebraic(state, time) if algebraic is None else algebraic
        return float(values[self.circuit.node_index[node]])

    def component_voltage(
        self,
        component: Component,
        state: np.ndarray,
        algebraic: np.ndarray | None = None,
        time: float | None = None,
    ) -> float:
        values = self.algebraic(state, time) if algebraic is None else algebraic
        return self.node_voltage(component.p, state, values) - self.node_voltage(component.n, state, values)

    def component_current(
        self,
        component: Component,
        state: np.ndarray,
        algebraic: np.ndarray | None = None,
        time: float | None = None,
    ) -> float:
        values = self.algebraic(state, time) if algebraic is None else algebraic
        if component.type == "resistor":
            return self.component_voltage(component, state, values) / float(component.value)
        if component.type in {"inductor", "sat_inductor"}:
            return float(state[self.circuit.state_index[component.id]])
        if component.type == "current_source":
            if component.waveform is not None:
                if time is None:
                    raise ModelError(
                        "Evaluation time is required for a waveform current source"
                    )
                return float(component.waveform.value(time))
            return float(component.value)
        if component.id in self.branch_index:
            return float(values[self.branch_index[component.id]])
        if component.type in SWITCHING_BRANCH_TYPES:
            return 0.0
        raise ModelError(f"Current output is not implemented for {component.type}")

    def diode_guard(
        self,
        diode: Component,
        state: np.ndarray,
        conducting: bool,
        time: float | None = None,
    ) -> float:
        values = self.algebraic(state, time)
        if conducting:
            return self.component_current(diode, state, values, time)
        return self.component_voltage(diode, state, values, time) - float(
            diode.forward_voltage
        )

    def output(
        self, expression: str, state: np.ndarray, time: float | None = None
    ) -> float:
        kind, target = self.circuit.output_targets[expression]
        values = self.algebraic(state, time)
        if kind == "node_voltage":
            return self.node_voltage(target, state, values, time)
        component = self.circuit.component_by_id[target]
        if kind == "component_voltage":
            return self.component_voltage(component, state, values, time)
        return self.component_current(component, state, values, time)

    def power_balance_residual(
        self, state: np.ndarray, time: float | None = None
    ) -> float:
        """Return dE/dt plus absorbed power in all non-storage elements."""
        algebraic = self.algebraic(state, time)
        derivative = self.derivative(time if time is not None else 0.0, state)
        storage_rate = 0.0
        for component in self.circuit.state_components:
            index = self.circuit.state_index[component.id]
            incremental = float(component.value)
            if (
                component.type == "sat_inductor"
                and abs(float(state[index])) > float(component.sat_current)
            ):
                incremental = float(component.sat_inductance)
            storage_rate += incremental * state[index] * derivative[index]
        for coupling in self.circuit.couplings:
            first_index = self.circuit.state_index[coupling.first]
            second_index = self.circuit.state_index[coupling.second]
            # Cross term of E = 0.5 i^T L i: d/dt(M i1 i2).
            storage_rate += float(coupling.mutual) * (
                state[first_index] * derivative[second_index]
                + state[second_index] * derivative[first_index]
            )
        absorbed = 0.0
        for component in self.circuit.components:
            if component.type in {"capacitor", "inductor", "sat_inductor", "voltage_probe"}:
                continue
            voltage = self.component_voltage(component, state, algebraic, time)
            current = self.component_current(component, state, algebraic, time)
            absorbed += voltage * current
        return float(storage_rate + absorbed)


class Circuit:
    def __init__(self, model: Model):
        self.model = model
        self.ground = model.ground
        self.components = model.components
        self.component_by_id = {component.id: component for component in self.components}
        node_names = sorted(
            {node for component in self.components for node in (component.p, component.n) if node != self.ground}
        )
        self.nodes = tuple(node_names)
        self.node_index = {name: index for index, name in enumerate(self.nodes)}
        self.state_components = tuple(
            component
            for component in self.components
            if component.type in {"capacitor", "inductor", "sat_inductor"}
        )
        self.inductors = tuple(
            component for component in self.components if component.type == "inductor"
        )
        self.sat_inductors = tuple(
            component for component in self.components if component.type == "sat_inductor"
        )
        self.couplings = tuple(model.couplings)
        self.coupling_by_inductor = {}
        for coupling in self.couplings:
            self.coupling_by_inductor[coupling.first] = coupling
            self.coupling_by_inductor[coupling.second] = coupling
        self.mutual_by_pair = {
            (min(c.first, c.second), max(c.first, c.second)): float(c.mutual)
            for c in self.couplings
        }
        self.coupling_groups = _coupling_groups(self.couplings)
        if len(self.sat_inductors) > 1:
            raise ModelError(
                "v0.1 supports at most one sat_inductor per model; region "
                "resolution for coupled saturable inductors is a follow-up"
            )
        self.state_index = {component.id: index for index, component in enumerate(self.state_components)}
        self.initial_state = np.array(
            [component.initial_state for component in self.state_components], dtype=float
        )
        # Diodes and thyristors are both "ideal branches with an inequality":
        # they share the whole active-set machinery, the compiled-mode cache and
        # the guard-rooting in the solver. What differs is only WHEN they may
        # close and whether blocking a forward voltage is admissible, which is
        # handled in the two places that decide those questions.
        self.diodes = tuple(
            component
            for component in self.components
            if component.type in LATCHING_OR_DIODE
        )
        self.latching = frozenset(
            component.id
            for component in self.components
            if component.type in LATCHING_TYPES
        )
        # Diodes that share a node with a magnetically coupled winding. Only
        # there does a guard sitting exactly on its boundary fail to decide:
        # a transformer-fed bridge starts with every winding current at zero,
        # so several configurations satisfy every inequality and the search
        # takes whichever comes first. Elsewhere the boundary is a boundary and
        # the established behaviour stands — narrowing this was not caution but
        # measurement: applied everywhere it silently dropped a commutation
        # from the plain half-wave rectifier.
        coupled_nodes = {
            node
            for component in self.components
            if component.id in self.coupling_by_inductor
            for node in (component.p, component.n)
        }
        self.coupled_adjacent_diodes = frozenset(
            component.id
            for component in self.diodes
            if component.p in coupled_nodes or component.n in coupled_nodes
        )
        self.switches = tuple(component for component in self.components if component.type == "ideal_switch")
        self.waveform_sources = tuple(
            component
            for component in self.components
            if component.type in {"voltage_source", "current_source"}
            and component.waveform is not None
        )
        self.waveform_index = {
            component.id: index for index, component in enumerate(self.waveform_sources)
        }
        # Live waveform table. A declared machine rewrites its back-EMF at every
        # integration segment from the mechanical state; the compiled modes are
        # untouched because WHICH components are sources never changes, only
        # the u(t) they contribute.
        self._source_waveforms = [
            component.waveform for component in self.waveform_sources
        ]
        self.initial_diodes = {component.id: component.initial for component in self.diodes}
        self._cache: dict[Topology, CompiledMode | TopologyError] = {}
        # Active-set search plans, see _ModePlan. Cleared with the cache.
        self._plans: dict[tuple[Any, ...], _ModePlan] = {}
        # Floating-node memo for the active-set search. "Node X has no
        # algebraic support" is decided by node X's own KCL row, and that row
        # is stamped ONLY by the components touching X. So once a compile
        # fails at X, any later topology that puts every component touching X
        # in the same state fails too — it is skipped without being compiled.
        # This changes cost, never outcome: the skipped compile would have
        # raised. See _predicted_floating_node.
        self._incident: dict[str, tuple[Component, ...]] = {
            node: tuple(
                component
                for component in self.components
                if node in (component.p, component.n)
            )
            for node in self.nodes
        }
        self._floating_signatures: dict[str, set[tuple[Any, ...]]] = {}
        self._active_set_stats = {
            "calls": 0,
            "candidates_examined": 0,
            "max_candidates_in_call": 0,
            "anchor_repairs": 0,
        }
        self.output_targets = self._validate_outputs(model.outputs)

    def active_set_search_stats(self) -> dict[str, int]:
        """Counters of the diode active-set search performed so far.

        ``candidates_examined`` counts every diode configuration whose modes
        were compiled or fetched from cache and feasibility-checked;
        ``max_candidates_in_call`` is the worst single ``stable_mode`` query.
        Diagnostic view for scale benchmarks; not part of any evidence hash.
        """

        return dict(self._active_set_stats)

    def set_source_waveform(self, component_id: str, waveform: Any) -> None:
        """Replace one source's waveform without recompiling any mode."""

        index = self.waveform_index.get(component_id)
        if index is None:
            raise ModelError(f"{component_id!r} is not a waveform source")
        self._source_waveforms[index] = waveform

    def set_group_inductance_matrix(
        self, group: tuple[str, ...], matrix: np.ndarray
    ) -> None:
        """Replace one coupled group's inductance matrix, invalidating modes.

        Unlike a source waveform, an inductance is baked into every compiled
        mode's state-space matrices, so changing one INVALIDATES THE MODE
        CACHE. That cost is the price of a rotor-position-dependent machine
        and it was measured before this was built: recompiling every segment
        of the shipped 40 ms drive run costs 0.77 s against a 9.3 s run — an
        8 % overhead, not the wall the original design note assumed.

        Only the entries of THIS group are touched; a model whose machine is
        non-salient never calls this and keeps its cache intact.
        """

        if len(group) != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
            raise ModelError(
                f"inductance matrix shape {matrix.shape} does not match the "
                f"{len(group)}-winding group"
            )
        if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-15):
            raise ModelError("an inductance matrix must be symmetric")
        for row, member in enumerate(group):
            component = self.component_by_id[member]
            object.__setattr__(component, "value", float(matrix[row, row]))
            for column in range(row + 1, len(group)):
                key = (min(member, group[column]), max(member, group[column]))
                self.mutual_by_pair[key] = float(matrix[row, column])
        # Every cached mode carries the old matrix in its compiled form.
        self._cache.clear()
        self._plans.clear()
        # Floating is structural, not value-dependent, but the memo follows the
        # cache so that "what was learned" never outlives "what it was learned on".
        self._floating_signatures.clear()

    def coupling_matrix(self, group: tuple[str, ...]) -> np.ndarray:
        """Symmetric inductance matrix of one magnetically coupled group.

        Self inductances on the diagonal, declared mutuals off it, and zero
        where no mutual was declared — an undeclared pair inside a group is
        genuinely uncoupled, not assumed.
        """

        size = len(group)
        matrix = np.zeros((size, size), dtype=float)
        for row, member in enumerate(group):
            matrix[row, row] = float(self.component_by_id[member].value)
            for column in range(row + 1, size):
                key = (min(member, group[column]), max(member, group[column]))
                mutual = self.mutual_by_pair.get(key, 0.0)
                matrix[row, column] = mutual
                matrix[column, row] = mutual
        return matrix

    def source_values(self, time: float) -> np.ndarray:
        """Waveform-source values u(t) in ``waveform_index`` order."""

        return np.array(
            [waveform.value(time) for waveform in self._source_waveforms],
            dtype=float,
        )

    def source_slopes(self, time: float) -> np.ndarray:
        """Waveform-source time derivatives, valid within a smooth segment."""

        return np.array(
            [waveform.slope(time) for waveform in self._source_waveforms],
            dtype=float,
        )

    def source_breakpoints(self, start: float, stop: float) -> tuple[float, ...]:
        """Sorted waveform discontinuity/kink times strictly inside (start, stop)."""

        times: set[float] = set()
        for waveform in self._source_waveforms:
            times.update(waveform.breakpoints(start, stop))
        return tuple(sorted(times))

    def _validate_outputs(self, outputs: Iterable[str]) -> dict[str, tuple[str, str]]:
        targets: dict[str, tuple[str, str]] = {}
        all_nodes = set(self.nodes) | {self.ground}
        for expression in outputs:
            # Machine signals address the mechanical state, not the network.
            # The loader has already bound them to a declared machine; the
            # circuit deliberately has no opinion on them, and must not
            # resolve them against the expanded branch ids they look like.
            if is_machine_signal(expression) or is_declared_signal(expression):
                continue
            match = _OUTPUT_RE.match(expression.strip())
            if not match:
                raise ModelError(f"Invalid output expression {expression!r}; use v(node), v(component), or i(component)")
            kind, target = match.groups()
            if kind == "v" and target in all_nodes:
                targets[expression] = ("node_voltage", target)
            elif target in self.component_by_id:
                targets[expression] = (
                    "component_voltage" if kind == "v" else "component_current",
                    target,
                )
            else:
                raise ModelError(f"Output {expression!r} references unknown node/component {target!r}")
        return targets

    def compiled_topology_stats(self) -> dict[str, int]:
        """Counts of successfully compiled topologies in the compilation cache.

        ``reduced_topologies`` counts compiled modes that carry at least one
        zero-current constrained inductor.  Failed compilations cached as
        :class:`TopologyError` are excluded from both counts.
        """

        compiled = [value for value in self._cache.values() if isinstance(value, CompiledMode)]
        return {
            "compiled_topologies": len(compiled),
            "reduced_topologies": sum(
                bool(mode.constrained_inductors) for mode in compiled
            ),
        }

    def topology(
        self,
        gates: dict[str, bool],
        diodes: dict[str, bool],
        constrained_inductors: Iterable[str] = (),
        saturated_inductors: Iterable[str] = (),
    ) -> Topology:
        return Topology(
            tuple(sorted(gates.items())),
            tuple(sorted(diodes.items())),
            tuple(sorted(set(constrained_inductors))),
            tuple(sorted(set(saturated_inductors))),
        )

    def compile(
        self,
        gates: dict[str, bool],
        diodes: dict[str, bool],
        constrained_inductors: Iterable[str] = (),
        saturated_inductors: Iterable[str] = (),
        *,
        skip_known_floating: bool = False,
    ) -> CompiledMode:
        topology = self.topology(
            gates, diodes, constrained_inductors, saturated_inductors
        )
        return self._compile_topology(topology, skip_known_floating=skip_known_floating)

    def _compile_topology(
        self, topology: Topology, *, skip_known_floating: bool = False
    ) -> CompiledMode:
        cached = self._cache.get(topology)
        if isinstance(cached, TopologyError):
            raise cached
        if cached is not None:
            return cached
        if skip_known_floating:
            node = self._predicted_floating_node(topology)
            if node is not None:
                raise _SkippedFloatingTopology(self, topology, node)
        try:
            compiled = CompiledMode(self, topology)
        except TopologyError as exc:
            self._cache[topology] = exc
            node = getattr(exc, "floating_node", None)
            if node is not None:
                self._floating_signatures.setdefault(node, set()).add(
                    self._node_signature(node, topology)
                )
            raise
        self._cache[topology] = compiled
        return compiled

    def _node_signature(self, node: str, topology: Topology) -> tuple[Any, ...]:
        """State of every component touching ``node`` — all that its KCL row sees."""
        return self._signature_from(node, *_topology_lookup(topology))

    def _signature_from(
        self,
        node: str,
        gates: dict[str, bool],
        diodes: dict[str, bool],
        constrained: frozenset[str],
        saturated: frozenset[str],
    ) -> tuple[Any, ...]:
        return tuple(
            gates[component.control or ""]
            if component.type == "ideal_switch"
            else diodes[component.id]
            if component.type in LATCHING_OR_DIODE
            else component.id in constrained
            if component.type == "inductor"
            else component.id in saturated
            if component.type == "sat_inductor"
            else None
            for component in self._incident[node]
        )

    def _predicted_floating_node(self, topology: Topology) -> str | None:
        """A node known to float in ``topology``, or None.

        Sound by construction: a node's KCL row, the frozen-branch columns that
        are ignored in it and the star-point test all read only the components
        touching that node, so an identical local signature reproduces the
        failure exactly. The converse is not claimed — an unknown signature is
        simply compiled.
        """
        if not self._floating_signatures:
            return None
        lookup = _topology_lookup(topology)
        for node, signatures in self._floating_signatures.items():
            if self._signature_from(node, *lookup) in signatures:
                return node
        return None

    def _structurally_open_windings(
        self,
        gates: dict[str, bool],
        diode_states: dict[str, bool],
    ) -> tuple[str, ...]:
        """Coupled windings whose own series path is open in this configuration.

        A winding carrying zero current is not automatically frozen — it may be
        about to start conducting. It must be frozen when its current CANNOT
        change because the branch it sits in is interrupted: a terminal node
        with no other live component leaves the winding nowhere to send current,
        which is exactly the state of a transformer secondary between two
        blocking diodes.

        Read off the topology rather than searched for, because a nine-winding
        transformer offers more subsets than an enumeration should ever visit.
        The fixpoint matters: freezing one winding can strip the last support
        from the node its neighbour hangs on.
        """

        if not self.coupling_groups:
            return ()
        # Every coupled winding is a candidate, whatever its present current.
        # Whether the branch is OPEN is a fact about the topology; whether the
        # current is actually zero is a physical precondition, and the
        # feasibility check already refuses a frozen winding whose state is
        # not zero. Filtering here on the state instead would miss the very
        # instant that matters — a commutation, where the current reaches zero
        # and the branch opens in the same event.
        candidates = {
            component.id
            for component in self.inductors
            if component.id in self.coupling_by_inductor
        }
        if not candidates:
            return ()

        def carries_current(component: Component, frozen: set[str]) -> bool:
            if component.type == "voltage_probe":
                return False
            if component.type == "ideal_switch":
                return bool(gates.get(component.control or "", False))
            if component.type in LATCHING_OR_DIODE:
                return bool(diode_states.get(component.id, False))
            if component.type in {"inductor", "sat_inductor"}:
                return component.id not in frozen
            return True

        frozen: set[str] = set()
        changed = True
        while changed:
            changed = False
            for winding_id in sorted(candidates - frozen):
                winding = self.component_by_id[winding_id]
                for node in (winding.p, winding.n):
                    if node == self.ground:
                        continue
                    supported = any(
                        other.id != winding_id
                        and node in (other.p, other.n)
                        and carries_current(other, frozen)
                        for other in self.components
                    )
                    if not supported:
                        frozen.add(winding_id)
                        changed = True
                        break
        return tuple(sorted(frozen))

    def stable_mode(
        self,
        gates: dict[str, bool],
        preferred_diodes: dict[str, bool],
        state: np.ndarray,
        tolerance: float,
        time: float | None = None,
    ) -> tuple[dict[str, bool], CompiledMode]:
        if len(self.diodes) > MAX_IDEAL_DIODES:
            raise TopologyError(
                f"Active-set search supports at most {MAX_IDEAL_DIODES} ideal "
                "diodes (portable enumeration bound)"
            )
        diode_ids = [component.id for component in self.diodes]
        preferred_bits = tuple(preferred_diodes[diode_id] for diode_id in diode_ids)
        # The latch, expressed as a restriction on the search space: a
        # thyristor may appear CLOSED in a candidate only if it is already
        # conducting (it latched earlier and its current has not reached zero)
        # or its gate is high right now. Turn-off needs no rule of its own —
        # the feasibility check rejects conduction the moment the current would
        # reverse, which is natural commutation.
        latch_locked = tuple(
            index
            for index, diode_id in enumerate(diode_ids)
            if diode_id in self.latching
            and not preferred_diodes[diode_id]
            and not self._gate_permits_latch(diode_id, gates)
        )
        zero_current_inductors = tuple(
            sorted(
                component.id
                for component in self.inductors
                if component.id not in self.coupling_by_inductor
                and abs(float(state[self.state_index[component.id]]))
                <= STATE_CONSTRAINT_TOLERANCE
            )
        )
        saturation_candidates = self._saturation_candidates(state)
        failures: list[TopologyError] = []
        preferred_open_windings = self._structurally_open_windings(
            gates, preferred_diodes
        )
        examined = 0
        any_mode_compiled = False
        beyond_enumeration = 2 ** len(diode_ids) > MAX_ACTIVE_SET_CANDIDATES
        # All configurations within Hamming distance two of the preferred one.
        near_field = 1 + len(diode_ids) + len(diode_ids) * (len(diode_ids) - 1) // 2
        self._active_set_stats["calls"] += 1
        plan = self._mode_plan(
            gates,
            preferred_bits,
            latch_locked,
            zero_current_inductors,
            saturation_candidates,
            preferred_open_windings,
        )
        for entry in plan.entries():
            if (
                beyond_enumeration
                and not any_mode_compiled
                and examined >= near_field
            ):
                # Nothing even compiled anywhere near the preferred
                # configuration: the failure is structural (missing anchor
                # diodes for floating nodes), which distance-ordered
                # enumeration cannot reach in bounded time. Models within the
                # enumeration budget never take this path.
                repaired = self._anchor_repair(
                    gates,
                    dict(preferred_diodes),
                    state,
                    tolerance,
                    time,
                    zero_current_inductors,
                    saturation_candidates,
                )
                if repaired is not None:
                    self._active_set_stats["candidates_examined"] += examined
                    self._active_set_stats["anchor_repairs"] += 1
                    return repaired
            if entry is None:
                # A latch-locked configuration. Never counted as examined:
                # it is not physically reachable, so it is not evidence of
                # search effort either.
                continue
            examined += 1
            candidate, modes, entry_failures = entry
            candidate = dict(candidate)
            failures.extend(entry_failures)
            if modes:
                any_mode_compiled = True
            for mode in modes:
                if self._mode_is_feasible(
                    mode,
                    candidate,
                    state,
                    tolerance,
                    time,
                    gates,
                ):
                    self._active_set_stats["candidates_examined"] += examined
                    self._active_set_stats["max_candidates_in_call"] = max(
                        self._active_set_stats["max_candidates_in_call"], examined
                    )
                    return candidate, mode
        self._active_set_stats["candidates_examined"] += examined
        self._active_set_stats["max_candidates_in_call"] = max(
            self._active_set_stats["max_candidates_in_call"], examined
        )
        if examined >= MAX_ACTIVE_SET_CANDIDATES and 2 ** len(diode_ids) > examined:
            repaired = self._anchor_repair(
                gates,
                dict(preferred_diodes),
                state,
                tolerance,
                time,
                zero_current_inductors,
                saturation_candidates,
            )
            if repaired is not None:
                self._active_set_stats["anchor_repairs"] += 1
                return repaired
            first_failure = (
                f"; first failure: {_failure_text(failures[0])}" if failures else ""
            )
            raise TopologyError(
                "No stable ideal-switch configuration found within the "
                f"active-set search budget of {MAX_ACTIVE_SET_CANDIDATES} "
                f"candidates ({len(diode_ids)} diodes; the search covered every "
                "configuration up to the reached Hamming distance from the "
                "preferred one) and the deterministic anchor repair could not "
                "resolve a consistent mode either. This bounds the search "
                "instead of a hard diode-count limit; a physically consistent "
                "mode this far from the preferred configuration indicates an "
                "inconsistent initial diode guess or a modeling error"
                f"{first_failure}"
            )
        detail = (
            _failure_text(failures[0])
            if failures
            else "all diode inequalities are inconsistent"
        )
        raise TopologyError(f"No stable ideal-switch configuration found: {detail}")

    def _mode_plan(
        self,
        gates: dict[str, bool],
        preferred_bits: tuple[bool, ...],
        latch_locked: tuple[int, ...],
        zero_current_inductors: tuple[str, ...],
        saturation_candidates: list[tuple[str, ...]],
        preferred_open_windings: tuple[str, ...],
    ) -> "_ModePlan":
        """The active-set search for one configuration, built once, replayed after.

        Which candidates the search visits, in which order, and which modes
        each one compiles to (or fails with) depend only on the key below —
        never on the state values or the time. Only the feasibility check
        does, and stable_mode still runs it on every call. So a repeated
        configuration (every PWM period of a converter) replays the list
        instead of rebuilding topologies and re-probing the compile cache.

        The plan is extended lazily: a replay that needs more candidates
        than the first search visited continues the same enumeration.

        WARNING: anything added to the candidate loop that reads ``state``,
        ``time`` or ``tolerance`` must go into the key, or replays go wrong.
        """

        key = (
            tuple(sorted(gates.items())),
            preferred_bits,
            latch_locked,
            zero_current_inductors,
            tuple(saturation_candidates),
        )
        plan = self._plans.get(key)
        if plan is None or plan.broken:
            if len(self._plans) >= MAX_MODE_PLANS:
                self._plans.clear()  # bounds memory; costs time, never changes a result
            plan = _ModePlan(
                self._plan_source(
                    dict(gates),
                    preferred_bits,
                    latch_locked,
                    zero_current_inductors,
                    list(saturation_candidates),
                    preferred_open_windings,
                )
            )
            self._plans[key] = plan
        return plan

    def _plan_source(
        self,
        gates: dict[str, bool],
        preferred_bits: tuple[bool, ...],
        latch_locked: tuple[int, ...],
        zero_current_inductors: tuple[str, ...],
        saturation_candidates: list[tuple[str, ...]],
        preferred_open_windings: tuple[str, ...],
    ) -> Iterator[Any]:
        diode_ids = [component.id for component in self.diodes]
        # The Topology key, assembled the way self.topology() does it but with
        # the gate part sorted once per search instead of once per candidate.
        gate_items = tuple(sorted(gates.items()))
        by_id = sorted(range(len(diode_ids)), key=diode_ids.__getitem__)
        ids_sorted = [diode_ids[index] for index in by_id]
        for bits in _hamming_ordered_candidates(
            preferred_bits, MAX_ACTIVE_SET_CANDIDATES
        ):
            if any(bits[index] for index in latch_locked):
                yield None
                continue
            candidate = dict(zip(diode_ids, bits))
            diode_items = tuple(zip(ids_sorted, map(bits.__getitem__, by_id)))
            open_windings = (
                preferred_open_windings
                if bits == preferred_bits
                else self._structurally_open_windings(gates, candidate)
            )
            modes: list[CompiledMode] = []
            failures: list[TopologyError] = []
            for saturated in saturation_candidates:
                try:
                    modes.append(
                        self._compile_topology(
                            Topology(
                                gate_items,
                                diode_items,
                                tuple(sorted(set(open_windings))),
                                tuple(sorted(set(saturated))),
                            ),
                            skip_known_floating=True,
                        )
                    )
                except TopologyError as exc:
                    failures.append(exc)
                    # Never freeze a well-posed active inductor merely to satisfy
                    # a guard. Reduction is a fallback only for a singular
                    # current-constrained mode. Subsets are tried smallest-first
                    # so the least-frozen feasible mode always wins.
                    for constrained_subset in _reduction_subsets(
                        zero_current_inductors
                    ):
                        try:
                            modes.append(
                                self._compile_topology(
                                    Topology(
                                        gate_items,
                                        diode_items,
                                        tuple(
                                            sorted(
                                                set(constrained_subset)
                                                | set(open_windings)
                                            )
                                        ),
                                        tuple(sorted(set(saturated))),
                                    ),
                                    skip_known_floating=True,
                                )
                            )
                        except TopologyError as reduced_exc:
                            failures.append(reduced_exc)
            yield candidate, tuple(modes), tuple(failures)

    def _anchor_repair(
        self,
        gates: dict[str, bool],
        candidate: dict[str, bool],
        state: np.ndarray,
        tolerance: float,
        time: float | None,
        zero_current_inductors: tuple[str, ...],
        saturation_candidates: list[tuple[str, ...]],
    ) -> tuple[dict[str, bool], CompiledMode] | None:
        """Deterministic greedy repair beyond the enumeration budget.

        Many-level bridges can require flipping MANY zero-current anchor
        diodes at once (one per floating internal node of every leg — a cold
        start of a three-phase five-level NPC needs ~nine), which no
        distance-ordered enumeration can reach. The repair walks the compile
        diagnostics instead: a floating node turns on its first off diode, an
        infeasible mode flips its first violated guard. Every step is
        deterministic (sorted component order) and the walk is bounded, so
        the result is reproducible; models within the enumeration budget
        never reach this path, keeping their historical behavior
        bit-identical.
        """

        switch_terminal_pairs = {
            frozenset((component.p, component.n))
            for component in self.switches
        }
        antiparallel = {
            diode.id
            for diode in self.diodes
            if frozenset((diode.p, diode.n)) in switch_terminal_pairs
        }
        diodes_by_node: dict[str, list[str]] = {}
        for diode in self.diodes:
            diodes_by_node.setdefault(diode.p, []).append(diode.id)
            diodes_by_node.setdefault(diode.n, []).append(diode.id)
        # A floating internal node is anchored preferentially through a diode
        # that is NOT the body diode of an off-commanded switch: clamp diodes
        # exist precisely to define internal-node potentials, while anchoring
        # across an off device usually ties the node to a wrong rail and only
        # surfaces as a violated guard several branches later.
        for node_name, incident in diodes_by_node.items():
            diodes_by_node[node_name] = sorted(
                incident, key=lambda diode_id: (diode_id in antiparallel, diode_id)
            )
        diode_order = [diode.id for diode in self.diodes]
        visited: set[tuple[bool, ...]] = set()
        stack: list[dict[str, bool]] = [dict(candidate)]
        compile_budget = 4096
        while stack and compile_budget > 0:
            current = stack.pop()
            signature = tuple(current[diode_id] for diode_id in diode_order)
            if signature in visited:
                continue
            visited.add(signature)
            # A floating node offers several possible anchors (clamp vs
            # antiparallel); an anchor that merely ties the node to a wrong
            # potential surfaces later as a singular topology or a violated
            # guard, so the walk branches over the alternatives and dead
            # ends simply backtrack.
            flip_candidates: list[str] = []
            for saturated in saturation_candidates:
                modes: list[CompiledMode] = []
                try:
                    compile_budget -= 1
                    modes.append(self.compile(gates, current, (), saturated))
                except TopologyError as exc:
                    floating_node = getattr(exc, "floating_node", None)
                    if floating_node is not None:
                        flip_candidates.extend(
                            diode_id
                            for diode_id in diodes_by_node.get(floating_node, [])
                            if not current[diode_id]
                        )
                    for constrained_subset in _reduction_subsets(
                        zero_current_inductors
                    ):
                        try:
                            compile_budget -= 1
                            modes.append(
                                self.compile(
                                    gates, current, constrained_subset, saturated
                                )
                            )
                        except TopologyError as reduced_exc:
                            reduced_node = getattr(
                                reduced_exc, "floating_node", None
                            )
                            if reduced_node is not None:
                                flip_candidates.extend(
                                    diode_id
                                    for diode_id in diodes_by_node.get(
                                        reduced_node, []
                                    )
                                    if not current[diode_id]
                                )
                for mode in modes:
                    if self._mode_is_feasible(
                        mode, current, state, tolerance, time, gates
                    ):
                        return current, mode
                    for diode in self.diodes:
                        conducting = current[diode.id]
                        guard = mode.diode_guard(diode, state, conducting, time)
                        if (
                            not conducting
                            and diode.id in self.latching
                            and not self._gate_permits_latch(diode.id, gates)
                        ):
                            # An ungated blocking thyristor is never the
                            # inconsistency: standing off a forward voltage is
                            # legal, and the repair must not "fix" it by
                            # closing the device.
                            continue
                        if (conducting and guard < -tolerance) or (
                            not conducting and guard > tolerance
                        ):
                            flip_candidates.append(diode.id)
                            break
            deduplicated: list[str] = []
            for diode_id in flip_candidates:
                if diode_id not in deduplicated:
                    deduplicated.append(diode_id)
            # Reversed push order makes the stack explore candidates in the
            # deterministic listed order.
            for diode_id in reversed(deduplicated):
                branched = dict(current)
                branched[diode_id] = not branched[diode_id]
                stack.append(branched)
        return None

    def _saturation_candidates(self, state: np.ndarray) -> list[tuple[str, ...]]:
        """Deterministic saturation-region candidates for the current state.

        Away from the |i| = i_sat boundary the region is a pure function of
        the state. Exactly on the boundary both regions are offered (the
        unsaturated one first) and the consistency check in
        ``_mode_is_feasible`` picks the one whose own derivative moves the
        current into it. The inductor voltage is independent of the segment
        inductance, so the derivative sign agrees across candidates and the
        choice is unambiguous.
        """

        if not self.sat_inductors:
            return [()]
        component = self.sat_inductors[0]
        current = float(state[self.state_index[component.id]])
        boundary = float(component.sat_current)
        tolerance = max(STATE_CONSTRAINT_TOLERANCE, 1e-12 * boundary)
        if abs(current) > boundary + tolerance:
            return [(component.id,)]
        if abs(current) < boundary - tolerance:
            return [()]
        return [(), (component.id,)]

    def _gate_permits_latch(self, component_id: str, gates: dict[str, bool]) -> bool:
        """Is this thyristor's gate asserted right now?

        A thyristor with no declared gate can never close, which is a useful
        thing to be able to model (a device held permanently blocking) and is
        refused at load time only if it would make the model meaningless.
        """

        component = self.component_by_id[component_id]
        control = component.control
        if control is None:
            return False
        return bool(gates.get(control, False))

    def _mode_is_feasible(
        self,
        mode: CompiledMode,
        diode_states: dict[str, bool],
        state: np.ndarray,
        diode_tolerance: float,
        time: float | None = None,
        gates: dict[str, bool] | None = None,
    ) -> bool:
        for component in self.sat_inductors:
            index = self.state_index[component.id]
            current = float(state[index])
            boundary = float(component.sat_current)
            tolerance = max(STATE_CONSTRAINT_TOLERANCE, 1e-12 * boundary)
            flag = component.id in mode.saturated_inductors
            if abs(current) > boundary + tolerance and not flag:
                return False
            if abs(current) < boundary - tolerance and flag:
                return False
            if boundary - tolerance <= abs(current) <= boundary + tolerance:
                derivative = mode.derivative(
                    time if time is not None else 0.0, state
                )[index]
                outward = current * float(derivative)
                if outward > 0.0 and not flag:
                    return False
                if outward < 0.0 and flag:
                    return False
        algebraic = mode.algebraic(state, time)
        for component_id in mode.constrained_inductors:
            state_value = float(state[self.state_index[component_id]])
            if abs(state_value) > STATE_CONSTRAINT_TOLERANCE:
                return False
            coefficients, constant = mode.constrained_branch_current_affine(
                component_id
            )
            active_coefficients = coefficients[list(mode.active_state_indices)]
            source_coefficients = mode.solution_from_sources[
                mode.branch_index[component_id], :
            ]
            # Entry-time cancellation is insufficient: the hidden branch current must
            # be the zero affine map — including waveform-source terms — for every
            # active-state trajectory in this mode.
            if (
                constant != 0.0
                or np.any(active_coefficients != 0.0)
                or np.any(source_coefficients != 0.0)
            ):
                return False
            branch_current = mode.constrained_branch_current(
                component_id,
                state,
                algebraic,
            )
            if abs(branch_current) > STATE_CONSTRAINT_TOLERANCE:
                return False

        for diode in self.diodes:
            conducting = diode_states[diode.id]
            guard = mode.diode_guard(diode, state, conducting, time)
            if conducting and guard < -diode_tolerance:
                return False
            if not conducting and guard > diode_tolerance:
                # An UNGATED thyristor is allowed to stand off a forward
                # voltage — that is the entire point of the device. With its
                # gate asserted it is obliged to close under one, exactly like
                # a diode, which is what makes a held gate equivalent to an
                # ideal switch in series with a diode.
                if diode.id not in self.latching or self._gate_permits_latch(
                    diode.id, gates or {}
                ):
                    return False
        return True


class _ModePlan:
    """The visited candidates of one active-set search, extended on demand.

    Each entry is (diodes, compiled modes, compile failures), or None for a
    latch-locked configuration the search steps over.
    """

    def __init__(self, source: Iterator[Any]) -> None:
        self._source = source
        self._visited: list[Any] = []
        self._finished = False
        # An unexpected error inside the search ends the generator. A plan
        # that kept going would then look complete while missing candidates,
        # so it is marked broken and the next call searches afresh.
        self.broken = False

    def entries(self) -> Iterator[Any]:
        index = 0
        while True:
            if index < len(self._visited):
                yield self._visited[index]
            elif self._finished:
                return
            else:
                try:
                    entry = next(self._source)
                except StopIteration:
                    self._finished = True
                    return
                except BaseException:
                    self.broken = True
                    raise
                self._visited.append(entry)
                yield entry
            index += 1


class _SkippedFloatingTopology(TopologyError):
    """A candidate rejected by the floating-node memo instead of compiled.

    Its text is produced on demand by compiling the topology for real, so a
    message that reaches the user is word for word the one the full compile
    gives (which may name an earlier floating node than the memo did).
    """

    def __init__(self, circuit: "Circuit", topology: Topology, node: str) -> None:
        super().__init__(f"Node {node!r} has no algebraic support in this topology")
        self.floating_node = node
        self._circuit = circuit
        self._topology = topology

    def exact_message(self) -> str:
        try:
            self._circuit.compile(
                self._topology.gate_dict(),
                self._topology.diode_dict(),
                self._topology.constrained_inductors,
                self._topology.saturated_inductors,
            )
        except TopologyError as exc:
            return str(exc)
        raise AssertionError(  # pragma: no cover - the memo is sound by construction
            "floating-node memo predicted a failure the full compile did not reproduce"
        )


def _topology_lookup(
    topology: Topology,
) -> tuple[dict[str, bool], dict[str, bool], frozenset[str], frozenset[str]]:
    return (
        topology.gate_dict(),
        topology.diode_dict(),
        frozenset(topology.constrained_inductors),
        frozenset(topology.saturated_inductors),
    )


def _failure_text(failure: TopologyError) -> str:
    if isinstance(failure, _SkippedFloatingTopology):
        return failure.exact_message()
    return str(failure)


def _active_gamma_rows(
    circuit: "Circuit", constrained: frozenset[str]
) -> dict[str, tuple[tuple[Component, float], ...]]:
    """Rows of inv(L_AA): di_k/dt as a linear map over terminal voltages.

    For a plain inductor this would be the single term 1/L. A coupled winding's
    di/dt depends on EVERY active winding of its group, which is what makes a
    star point of transformer secondaries different from a star of plain
    inductors: the invariant derivative that pins the star voltage has to carry
    the mutual terms, not just the self inductances.
    """

    rows: dict[str, tuple[tuple[Component, float], ...]] = {}
    for group in circuit.coupling_groups:
        members = [circuit.component_by_id[member_id] for member_id in group]
        active = [item for item in members if item.id not in constrained]
        if not active:
            continue
        matrix = circuit.coupling_matrix(group)
        slot = {item.id: index for index, item in enumerate(members)}
        active_slots = [slot[item.id] for item in active]
        gamma = np.linalg.inv(matrix[np.ix_(active_slots, active_slots)])
        for index, winding in enumerate(active):
            rows[winding.id] = tuple(
                (active[column], float(gamma[index, column]))
                for column in range(len(active))
            )
    return rows


def _induced_voltage_rows(
    circuit: "Circuit", constrained: frozenset[str]
) -> dict[str, tuple[tuple[Component, float], ...]]:
    """Coefficients of the induced voltage on each frozen coupled winding.

    For a group whose frozen members F carry no current, the active members A
    obey v_A = L_AA di_A/dt, so every frozen winding shows

        v_f = sum_j L_fj di_j/dt = L_fA inv(L_AA) v_A

    on its open terminals. Returned as (active winding, coefficient) pairs so
    the caller can stamp the relation directly onto node voltages.
    """

    rows: dict[str, tuple[tuple[Component, float], ...]] = {}
    for group in circuit.coupling_groups:
        members = [circuit.component_by_id[member_id] for member_id in group]
        frozen = [item for item in members if item.id in constrained]
        if not frozen:
            continue
        active = [item for item in members if item.id not in constrained]
        if not active:
            # Nothing carries flux, so there is nothing to induce.
            continue
        matrix = circuit.coupling_matrix(group)
        slot = {item.id: index for index, item in enumerate(members)}
        active_slots = [slot[item.id] for item in active]
        active_inverse = np.linalg.inv(matrix[np.ix_(active_slots, active_slots)])
        for winding in frozen:
            coupling_row = matrix[slot[winding.id], active_slots]
            coefficients = coupling_row @ active_inverse
            rows[winding.id] = tuple(
                (active[index], float(coefficients[index]))
                for index in range(len(active))
            )
    return rows


def _coupling_groups(couplings: Iterable[Any]) -> tuple[tuple[str, ...], ...]:
    """Windings joined by any declared mutual, as deterministic sorted groups."""

    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    for coupling in couplings:
        left, right = find(coupling.first), find(coupling.second)
        if left != right:
            parent[max(left, right)] = min(left, right)
    groups: dict[str, list[str]] = {}
    for node in sorted(parent):
        groups.setdefault(find(node), []).append(node)
    return tuple(tuple(sorted(members)) for _root, members in sorted(groups.items()))


def _hamming_ordered_candidates(
    preferred_bits: tuple[bool, ...],
    budget: int,
) -> Iterator[tuple[bool, ...]]:
    """Diode configurations by (Hamming distance from preferred, lexicographic).

    Reproduces exactly the order of the former eager enumeration —
    ``sorted(product((False, True), repeat=n), key=(distance, bits))`` — but
    lazily, one distance level at a time, so large diode counts never
    materialize the full 2**n space. At most ``budget`` configurations are
    yielded; when the budget truncates a level, its lexicographically
    smallest members are kept, so the visited prefix is identical to the
    eager ordering. For ``2**n <= budget`` the sequence is the complete
    space, bit-identical to the former behavior.
    """

    n = len(preferred_bits)
    positions = range(n)
    yielded = 0
    for distance in range(n + 1):
        remaining = budget - yielded
        if remaining <= 0:
            return
        flipped_level = (
            _flipped(preferred_bits, combination)
            for combination in itertools.combinations(positions, distance)
        )
        if math.comb(n, distance) <= remaining:
            level = sorted(flipped_level)
        else:
            level = heapq.nsmallest(remaining, flipped_level)
        for bits in level:
            yield bits
            yielded += 1


def _flipped(bits: tuple[bool, ...], positions: tuple[int, ...]) -> tuple[bool, ...]:
    flipped = list(bits)
    for index in positions:
        flipped[index] = not flipped[index]
    return tuple(flipped)


def _reduction_subsets(
    zero_current_inductors: tuple[str, ...],
) -> list[tuple[str, ...]]:
    """Deterministic candidate sets of inductors to freeze, smallest first.

    Multi-leg dead-time windows can require freezing several zero-current
    inductors at once. Full subset enumeration is bounded:
    beyond four zero-current inductors only singles and the full set are
    offered, which keeps the active-set search portable.
    """

    members = tuple(zero_current_inductors)
    if len(members) > 4:
        singles: list[tuple[str, ...]] = [(name,) for name in members]
        return singles + [members]
    subsets: list[tuple[str, ...]] = []
    for size in range(1, len(members) + 1):
        subsets.extend(itertools.combinations(members, size))
    return subsets


def _stamp_conductance(matrix: np.ndarray, p: int | None, n: int | None, value: float) -> None:
    if p is not None:
        matrix[p, p] += value
    if n is not None:
        matrix[n, n] += value
    if p is not None and n is not None:
        matrix[p, n] -= value
        matrix[n, p] -= value


def _stamp_known_current(rhs: np.ndarray, p: int | None, n: int | None, value: float) -> None:
    if p is not None:
        rhs[p] -= value
    if n is not None:
        rhs[n] += value


def _stamp_branch_incidence(
    matrix: np.ndarray, p: int | None, n: int | None, branch: int
) -> None:
    if p is not None:
        matrix[p, branch] += 1.0
    if n is not None:
        matrix[n, branch] -= 1.0
