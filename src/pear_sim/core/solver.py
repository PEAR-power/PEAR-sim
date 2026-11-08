from __future__ import annotations

import bisect
from dataclasses import dataclass, field
import math
from time import perf_counter
from typing import Any, Callable

import numpy as np
from scipy.integrate import solve_ivp

from pear_sim.core.circuit import Circuit, CompiledMode, STATE_CONSTRAINT_TOLERANCE
from pear_sim.errors import SimulationError
from pear_sim.core import rust_backend
from pear_sim.analysis.harmonics import HarmonicAccumulator
from pear_sim.limits import MAX_RUN_EVENT_ROWS
from pear_sim.spec.signals import (
    declared_signal_target,
    is_declared_signal,
    is_machine_signal,
    parse_signal_expression,
    signal_unit,
    unit_for,
)
from pear_sim.core.svpwm import svpwm_duties
from pear_sim.spec.signal_graph import evaluate_block
from pear_sim.core.scheduling import MAX_SCHEDULED_GATE_EDGES
from pear_sim.core.metrics import (
    _QUADRATURE_NODES,
    _QUADRATURE_WEIGHTS,
    EVIDENCE_MODES,
    _MetricAccumulator,
    _STATE_EXTREMA_METHOD,
    _STATE_EXTREMA_NOT_COMPUTED,
    _TIME_METRIC_METHOD,
    _dense_state_extrema,
    _merge_state_extrema,
    _state_values_by_component,
)
from pear_sim.spec.machines import (
    PHASE_OFFSETS,
    back_emf_waveform,
    electromagnetic_torque,
    mapped_torque,
    salient_inductance_matrix,
    salient_torque,
)
from pear_sim.spec.model import Model

from pear_sim.spec.waveforms import SineWaveform
from pear_sim.core.scheduling import (
    _build_gate_schedule,
    _initial_gate_values,
)

# Compatibility re-exports: the scheduler moved to pear_sim.core.scheduling in the
# stage-2 restructuring, but pear_sim.core.solver remains its historical import path.
from pear_sim.core.scheduling import _GateSchedule as _GateSchedule  # noqa: F401
from pear_sim.core.scheduling import _legacy_scheduled_edges as _legacy_scheduled_edges  # noqa: F401
from pear_sim.core.scheduling import _pwm_value as _pwm_value  # noqa: F401
from pear_sim.core.scheduling import _scheduled_edges as _scheduled_edges  # noqa: F401
from pear_sim.core.scheduling import _ulp_close as _ulp_close  # noqa: F401
from pear_sim.core.scheduling import scheduled_edges as scheduled_edges  # noqa: F401

#: Engines that can integrate a segment. "rust" is the optional compiled
#: kernel (pear-sim-core); "python" is the reference.
BACKENDS = ("python", "rust")


@dataclass(frozen=True)
class SimulationEvent:
    time: float
    kind: str
    target: str
    previous: bool
    current: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "time": self.time,
            "kind": self.kind,
            "target": self.target,
            "previous": self.previous,
            "current": self.current,
        }


@dataclass
class SimulationResult:
    model: Model
    time: np.ndarray
    signals: dict[str, np.ndarray]
    events: list[SimulationEvent]
    stats: dict[str, Any]
    time_integrals: dict[str, Any]
    continuous_outputs: frozenset[str]
    physics_audit: dict[str, Any]
    mode_intervals: list[dict[str, Any]]
    state_extrema: dict[str, Any]
    modulation_cycles: list[dict[str, Any]]
    harmonic_analysis: dict[str, Any] | None = None
    control_trace: list[dict[str, Any]] = field(default_factory=list)
    machine_trace: list[dict[str, Any]] = field(default_factory=list)
    signal_trace: list[dict[str, Any]] = field(default_factory=list)
    #: Evidence scope this run was produced under. Defaults to ``"full"`` so
    #: every existing producer and every hand-built result keeps its meaning.
    evidence_mode: str = "full"
    #: Engine that integrated the segments. It is part of the run identity
    #: (run_sha256), so a run from another engine is never mistaken for one
    #: from the reference implementation.
    backend: str = "reference-python"

    def metrics(self) -> dict[str, dict[str, Any]]:
        """Return time-weighted metrics and explicitly named output-grid statistics.

        ``mean`` and ``rms`` are derived from quadrature over the integrator's actual
        dense-output intervals. Values at isolated topology events have zero measure.
        The ``sample_*`` fields retain statistics of the requested output grid.
        """

        result: dict[str, dict[str, Any]] = {}
        full_window = self.time_integrals["full"]
        tail_window = self.time_integrals["last_half"]
        tail_start = float(tail_window["start_s"])
        tail_index = int(np.searchsorted(self.time, tail_start, side="left"))
        graph = getattr(self.model, "signals", None)
        declared_units = (
            {block.id: block.unit for block in graph.blocks} if graph else {}
        )
        for name, values in self.signals.items():
            tail = values[tail_index:]
            if not len(tail):
                tail = values[-1:]
            if is_machine_signal(name) or is_declared_signal(name):
                # A machine signal is not an affine map over the electrical
                # state, so the segment quadrature that produces `mean` and
                # `rms` cannot be applied to it. Reporting zeros, or the
                # output-grid average under the continuous-time name, would
                # be a quieter lie than saying so. The window keys are
                # OMITTED, not zeroed: indexing them raises instead of
                # returning a plausible number.
                result[name] = _non_affine_metrics(
                    name, values, tail, tail_start, declared_units
                )
                continue
            full_moments = full_window["signals"][name]
            tail_moments = tail_window["signals"][name]
            full_duration = float(full_window["duration_s"])
            tail_duration = float(tail_window["duration_s"])
            sample_min = float(np.min(values))
            sample_max = float(np.max(values))
            sample_tail_min = float(np.min(tail))
            sample_tail_max = float(np.max(tail))
            payload: dict[str, Any] = {
                "unit": signal_unit(name),
                "aggregation_method": _TIME_METRIC_METHOD,
                "mean": float(full_moments["integral"] / full_duration),
                "rms": float(math.sqrt(max(0.0, full_moments["square_integral"] / full_duration))),
                "integral": float(full_moments["integral"]),
                "square_integral": float(full_moments["square_integral"]),
                "mean_last_half": float(tail_moments["integral"] / tail_duration),
                "rms_last_half": float(
                    math.sqrt(max(0.0, tail_moments["square_integral"] / tail_duration))
                ),
                "integral_last_half": float(tail_moments["integral"]),
                "square_integral_last_half": float(tail_moments["square_integral"]),
                "full_window_start_s": float(full_window["start_s"]),
                "full_window_end_s": float(full_window["end_s"]),
                "full_window_duration_s": full_duration,
                "window_start_s": tail_start,
                "window_end_s": float(tail_window["end_s"]),
                "window_duration_s": tail_duration,
                "sample_count": int(len(values)),
                "sample_count_last_half": int(len(tail)),
                "sample_min": sample_min,
                "sample_max": sample_max,
                "sample_mean": float(np.mean(values)),
                "sample_rms": float(np.sqrt(np.mean(values * values))),
                "sample_final": float(values[-1]),
                "sample_peak_to_peak": sample_max - sample_min,
                "sample_min_last_half": sample_tail_min,
                "sample_max_last_half": sample_tail_max,
                "sample_mean_last_half": float(np.mean(tail)),
                "sample_rms_last_half": float(np.sqrt(np.mean(tail * tail))),
                "sample_peak_to_peak_last_half": sample_tail_max - sample_tail_min,
                # Compatibility aliases remain explicitly described as sampled extrema.
                "extrema_method": "requested_output_grid_samples",
                "min": sample_min,
                "max": sample_max,
                "final": float(values[-1]),
                "peak_to_peak": sample_max - sample_min,
                "min_last_half": sample_tail_min,
                "max_last_half": sample_tail_max,
                "peak_to_peak_last_half": sample_tail_max - sample_tail_min,
            }
            if name in self.continuous_outputs:
                payload.update(
                    {
                        "continuity": "structurally_continuous_across_visited_modes",
                        "ripple": sample_max - sample_min,
                        "ripple_last_half": sample_tail_max - sample_tail_min,
                        "ripple_method": "requested_output_grid_peak_to_peak",
                    }
                )
            else:
                payload["continuity"] = "topology_dependent_or_unproven"
            result[name] = payload
        return result


_MACHINE_METRIC_NOT_COMPUTED = (
    "this signal is not an affine map over the electrical state — a machine "
    "signal is sampled from the mechanical state, and a declared signal block "
    "is a discrete value held between task instants — so continuous-time "
    "quadrature is not defined for it. Instantaneous values are in "
    "waveforms.csv, per-segment torque integrals in machine-trace.jsonl, and "
    "per-tick block values in signal-trace.jsonl."
)


def _non_affine_metrics(
    name: str,
    values: np.ndarray,
    tail: np.ndarray,
    tail_start: float,
    declared_units: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Output-grid statistics for a machine signal, with no invented integrals.

    Everything reported here is a statistic OF THE REQUESTED OUTPUT GRID and
    is named `sample_*` accordingly — the same convention the electrical
    outputs use for their grid statistics. The continuous-time keys are
    absent, and their absence is stated rather than filled.
    """

    sample_min = float(np.min(values))
    sample_max = float(np.max(values))
    sample_tail_min = float(np.min(tail))
    sample_tail_max = float(np.max(tail))
    return {
        "unit": unit_for(name, declared_units),
        "aggregation_method": _STATE_EXTREMA_NOT_COMPUTED,
        "time_integrated_status": "not_computed",
        "time_integrated_reason": _MACHINE_METRIC_NOT_COMPUTED,
        "continuity": "topology_dependent_or_unproven",
        "extrema_method": "requested_output_grid_samples",
        "window_start_s": tail_start,
        "sample_count": int(len(values)),
        "sample_count_last_half": int(len(tail)),
        "sample_min": sample_min,
        "sample_max": sample_max,
        "sample_mean": float(np.mean(values)),
        "sample_rms": float(np.sqrt(np.mean(values * values))),
        "sample_final": float(values[-1]),
        "sample_peak_to_peak": sample_max - sample_min,
        "sample_min_last_half": sample_tail_min,
        "sample_max_last_half": sample_tail_max,
        "sample_mean_last_half": float(np.mean(tail)),
        "sample_rms_last_half": float(np.sqrt(np.mean(tail * tail))),
        "sample_peak_to_peak_last_half": sample_tail_max - sample_tail_min,
    }


def _initial_sampled_gate_values(model: Model) -> dict[str, bool]:
    """Initial gate states for sampled controls.

    A single-gate sampled PI cycle starts in its on phase. A dq block's
    center-aligned pulses start with every high gate off and every low gate
    on (the pole rests at the negative rail until this cycle's pulse).
    """

    gates: dict[str, bool] = {}
    for control_id, control in model.sampled_controls.items():
        if getattr(control, "kind", "sampled_pi_pwm") in {
            "sampled_dq_current_pwm",
            "signal_svpwm",
        }:
            gates[control_id] = control.role == "low"
        else:
            gates[control_id] = True
    return gates


class HybridSimulator:
    def __init__(
        self, model: Model, *, evidence_mode: str = "full", backend: str = "python"
    ):
        if evidence_mode not in EVIDENCE_MODES:
            raise SimulationError(
                f"Unknown evidence mode {evidence_mode!r}; expected one of "
                + ", ".join(EVIDENCE_MODES)
            )
        if backend not in BACKENDS:
            raise SimulationError(
                f"Unknown backend {backend!r}; expected one of " + ", ".join(BACKENDS)
            )
        self.evidence_mode = evidence_mode
        self.model = model
        self.circuit = Circuit(model)
        self.backend = backend
        if backend == "rust":
            # Fail closed: a model the compiled kernel does not cover is
            # refused with the reason, never quietly run on the reference.
            self.backend_id = rust_backend.require_supported(model, self.circuit)
        else:
            self.backend_id = "reference-python"
        span = model.transient.stop - model.transient.start
        self.time_tolerance = max(1e-15, span * 1e-12)
        self.guard_tolerance = 1e-10
        self.constraint_tolerance = STATE_CONSTRAINT_TOLERANCE
        # Shadow of the unforced gate schedule, refreshed at the start of each
        # run. Initialized here too so callers that drive _apply_scheduled_edges
        # directly (the scheduler tests do) work without a full simulate().
        self._natural_gates: dict[str, bool] = {}

    def initial_mode(self) -> tuple[dict[str, bool], CompiledMode]:
        """Resolve and return the stable t=start mode (diode states + mode).

        Public seam: the steady-state layer needs the initial mode's star
        constraints to keep its probes on the invariant subspace; this
        is the exact resolution `validate_initial_topology` performs.
        """

        settings = self.model.transient
        machine_driver = _MachineDriver(self.circuit, self.model)
        if machine_driver.active:
            # The back-EMF must exist before the first mode is selected.
            machine_driver.begin_segment(settings.start)
        gates = _initial_gate_values(self.model.controls, settings.start)
        gates.update(_initial_sampled_gate_values(self.model))
        # Shadow of the unforced schedule; a window overrides the effective
        # value without erasing what the schedule would have said.
        self._natural_gates = dict(gates)
        if self.model.forced_gate_windows:
            gates = self._force_gate_windows(gates, settings.start)
        self._assert_complementary_interlocks(gates, time=settings.start)
        diodes, mode = self.circuit.stable_mode(
            gates,
            dict(self.circuit.initial_diodes),
            self.circuit.initial_state,
            self.guard_tolerance,
            time=settings.start,
        )
        self._assert_star_invariants(mode, self.circuit.initial_state)
        return diodes, mode

    def validate_initial_topology(self) -> dict[str, int]:
        diodes, _mode = self.initial_mode()
        return {
            "states": len(self.circuit.state_components),
            "nodes": len(self.circuit.nodes),
            "diodes_conducting": sum(diodes.values()),
        }

    def simulate(self) -> SimulationResult:
        wall_time_start = perf_counter()
        settings = self.model.transient
        state = self.circuit.initial_state.copy()
        if len(state) == 0:
            raise SimulationError(
                "v0.1 transient simulation requires at least one capacitor or inductor"
            )
        machine_driver = _MachineDriver(self.circuit, self.model)
        if machine_driver.active:
            # The back-EMF must exist before the first mode is selected.
            machine_driver.begin_segment(settings.start)
        gates = _initial_gate_values(self.model.controls, settings.start)
        gates.update(_initial_sampled_gate_values(self.model))
        # Shadow of the unforced schedule; a window overrides the effective
        # value without erasing what the schedule would have said.
        self._natural_gates = dict(gates)
        if self.model.forced_gate_windows:
            gates = self._force_gate_windows(gates, settings.start)
        self._assert_complementary_interlocks(gates, time=settings.start)
        diodes = dict(self.circuit.initial_diodes)
        diodes, mode = self.circuit.stable_mode(
            gates,
            diodes,
            state,
            self.guard_tolerance,
            time=settings.start,
        )
        self._assert_star_invariants(mode, state)
        state, constrained_state_projections = self._project_constrained_state(state, mode)
        state_constraint_events = 0
        metric_accumulator = _MetricAccumulator(
            self.model,
            self.circuit,
            state,
            evidence_mode=self.evidence_mode,
        )
        metric_accumulator._record_mode(mode)
        harmonic_accumulator = HarmonicAccumulator(self.model)

        gate_schedule = _build_gate_schedule(
            self.model.controls,
            settings.start,
            settings.stop,
        )
        scheduled = list(gate_schedule.edges)
        edge_key = lambda edge: (edge[0], edge[2], edge[1])  # noqa: E731
        dq_blocks: dict[str, Any] = {}
        for control in self.model.sampled_controls.values():
            if getattr(control, "kind", "sampled_pi_pwm") == "sampled_dq_current_pwm":
                dq_blocks.setdefault(control.block_id, control)
        for control in self.model.sampled_controls.values():
            # Types whose modulator emits BOTH edges of every pulse itself.
            # This loop seeds a turn-ON at each cycle boundary, which is right
            # for a single-gate controller (it turns on, then the controller
            # schedules its own turn-off) and CATASTROPHIC for a bridge block:
            # six gates would be commanded on at once, which is a rail-to-rail
            # short. That regression was diagnosed to this line rather than to
            # the insertion-order suspicion first recorded with it.
            if getattr(control, "kind", "sampled_pi_pwm") in {
                "sampled_dq_current_pwm",
                "signal_svpwm",
            }:
                continue
            period = control.period
            cycle_count = (settings.stop - settings.start) / period
            if 2.0 * cycle_count > MAX_SCHEDULED_GATE_EDGES - len(scheduled):
                raise SimulationError(
                    f"Scheduled gate edges exceed the global limit of "
                    f"{MAX_SCHEDULED_GATE_EDGES}"
                )
            k = 1
            while True:
                edge_time = settings.start + k * period
                if edge_time >= settings.stop - self.time_tolerance:
                    break
                bisect.insort(scheduled, (edge_time, control.id, True), key=edge_key)
                k += 1
        for block_id, representative in dq_blocks.items():
            period = representative.period
            cycle_count = (settings.stop - settings.start) / period
            # 1 no-op tick + up to 12 runtime edges per carrier cycle.
            if 13.0 * cycle_count > MAX_SCHEDULED_GATE_EDGES - len(scheduled):
                raise SimulationError(
                    f"Scheduled gate edges exceed the global limit of "
                    f"{MAX_SCHEDULED_GATE_EDGES}"
                )
            tick_id = f"{block_id}.a.high"
            k = 1
            while True:
                edge_time = settings.start + k * period
                if edge_time >= settings.stop - self.time_tolerance:
                    break
                # Value False == the gate's state at every cycle boundary: the
                # tick is a no-op transition that only forces the solver to
                # stop exactly at the sampling instant.
                bisect.insort(scheduled, (edge_time, tick_id, False), key=edge_key)
                k += 1
        sampled_states: dict[str, dict[str, float]] = {
            control_id: {"integrator": 0.0, "last_cycle": -1.0}
            for control_id, control in self.model.sampled_controls.items()
            if getattr(control, "kind", "sampled_pi_pwm")
            not in {"sampled_dq_current_pwm", "signal_pwm", "signal_svpwm"}
        }
        for block_id, representative in dq_blocks.items():
            sampled_states[block_id] = {
                "last_cycle": -1.0,
                "pll_theta": float(representative.theta0_rad),
                "pll_integrator": 0.0,
                "d_integrator": 0.0,
                "q_integrator": 0.0,
            }
        control_trace: list[dict[str, Any]] = []

        def fire_sampled_controllers(
            time: float, sample_state: np.ndarray, sample_mode: CompiledMode
        ) -> None:
            """Sample, update PI, and inject this cycle's turn-off edge.

            Runs strictly before any gate edge is applied at ``time``: the
            sample comes from the pre-edge state and pre-edge topology, which
            makes the event order at a shared timestamp deterministic.
            """

            for block_id, block in dq_blocks.items():
                period = block.period
                cycle = round((time - settings.start) / period)
                boundary = settings.start + cycle * period
                if abs(time - boundary) > self.time_tolerance:
                    continue
                state_entry = sampled_states[block_id]
                if cycle <= state_entry["last_cycle"]:
                    continue
                state_entry["last_cycle"] = float(cycle)

                currents = [
                    sample_mode.output(expr, sample_state, time)
                    for expr in block.sense_currents
                ]
                voltages = [
                    sample_mode.output(expr, sample_state, time)
                    for expr in block.sense_voltages
                ]
                # Amplitude-invariant Clarke of the PCC voltages for the PLL.
                v_alpha = (2.0 * voltages[0] - voltages[1] - voltages[2]) / 3.0
                v_beta = (voltages[1] - voltages[2]) / math.sqrt(3.0)
                theta = state_entry["pll_theta"]
                cos_t = math.cos(theta)
                sin_t = math.sin(theta)
                v_d = v_alpha * cos_t + v_beta * sin_t
                v_q = -v_alpha * sin_t + v_beta * cos_t
                magnitude = math.hypot(v_d, v_q)
                normalized_q = v_q / max(magnitude, 1e-12)
                omega_n = 2.0 * math.pi * block.pll_bandwidth_hz
                pll_kp = 2.0 * block.pll_damping * omega_n
                pll_ki = omega_n * omega_n
                omega = (
                    2.0 * math.pi * block.pll_nominal_frequency_hz
                    + pll_kp * normalized_q
                    + state_entry["pll_integrator"]
                )
                # This cycle's control uses the PRE-update theta; the PLL then
                # advances by one sample (deterministic, documented order).
                state_entry["pll_integrator"] += pll_ki * normalized_q * period
                state_entry["pll_theta"] = math.fmod(
                    theta + omega * period, 2.0 * math.pi
                )

                i_alpha = (2.0 * currents[0] - currents[1] - currents[2]) / 3.0
                i_beta = (currents[1] - currents[2]) / math.sqrt(3.0)
                i_d = i_alpha * cos_t + i_beta * sin_t
                i_q = -i_alpha * sin_t + i_beta * cos_t

                commands: dict[str, float] = {}
                any_saturated = False
                for axis, measured, reference, key in (
                    ("d", i_d, block.id_ref, "d_integrator"),
                    ("q", i_q, block.iq_ref, "q_integrator"),
                ):
                    error = reference - measured
                    candidate = state_entry[key] + error * period
                    unsaturated = block.current_kp * error + block.current_ki * candidate
                    command = min(
                        max(unsaturated, -block.voltage_scale), block.voltage_scale
                    )
                    saturated = unsaturated != command
                    drives_deeper = (unsaturated > command and error > 0.0) or (
                        unsaturated < command and error < 0.0
                    )
                    if not (saturated and drives_deeper):
                        state_entry[key] = candidate
                    any_saturated = any_saturated or saturated
                    commands[axis] = command

                ref_alpha = commands["d"] * cos_t - commands["q"] * sin_t
                ref_beta = commands["d"] * sin_t + commands["q"] * cos_t
                references = (
                    ref_alpha,
                    -0.5 * ref_alpha + 0.5 * math.sqrt(3.0) * ref_beta,
                    -0.5 * ref_alpha - 0.5 * math.sqrt(3.0) * ref_beta,
                )
                normalized = [value / block.voltage_scale for value in references]
                if block.injection == "min_max":
                    offset = -0.5 * (max(normalized) + min(normalized))
                    normalized = [value + offset for value in normalized]
                duties = {}
                for phase_label, value in zip(("a", "b", "c"), normalized):
                    duty = 0.5 * (1.0 + value)
                    clipped = min(max(duty, block.duty_min), block.duty_max)
                    any_saturated = any_saturated or clipped != duty
                    duties[phase_label] = clipped

                dead_time = block.dead_time
                for phase_label, duty in duties.items():
                    high_id = f"{block_id}.{phase_label}.high"
                    low_id = f"{block_id}.{phase_label}.low"
                    command_on = time + 0.5 * (1.0 - duty) * period
                    command_off = time + 0.5 * (1.0 + duty) * period
                    for edge_time, gate_id, value in (
                        (command_on, low_id, False),
                        (command_on + dead_time, high_id, True),
                        (command_off, high_id, False),
                        (command_off + dead_time, low_id, True),
                    ):
                        if edge_time < settings.stop - self.time_tolerance:
                            bisect.insort(
                                scheduled, (edge_time, gate_id, value), key=edge_key
                            )
                if len(control_trace) >= MAX_RUN_EVENT_ROWS:
                    raise SimulationError(
                        "Sampled-control trace exceeds the portable artifact "
                        f"limit of {MAX_RUN_EVENT_ROWS} rows"
                    )
                control_trace.append(
                    {
                        "time_s": float(time),
                        "control_id": block_id,
                        "cycle_index": int(cycle),
                        "theta_rad": float(theta),
                        "frequency_hz": float(omega / (2.0 * math.pi)),
                        "i_d": float(i_d),
                        "i_q": float(i_q),
                        "v_d_command": float(commands["d"]),
                        "v_q_command": float(commands["q"]),
                        "duties": {
                            phase: float(value) for phase, value in duties.items()
                        },
                        "saturated": bool(any_saturated),
                    }
                )

            for control in self.model.sampled_controls.values():
                kind = getattr(control, "kind", "sampled_pi_pwm")
                if kind == "sampled_dq_current_pwm":
                    continue
                if kind in {"signal_pwm", "signal_svpwm"}:
                    # Driven by the signal task, not by this loop: the duty
                    # is computed in fire_signal_task and the edge injected
                    # there, so there is nothing to sample here.
                    continue
                period = control.period
                cycle = round((time - settings.start) / period)
                boundary = settings.start + cycle * period
                if abs(time - boundary) > self.time_tolerance:
                    continue
                state_entry = sampled_states[control.id]
                if cycle <= state_entry["last_cycle"]:
                    continue
                state_entry["last_cycle"] = float(cycle)
                sense_value = sample_mode.output(control.sense, sample_state, time)
                error = control.reference - sense_value
                candidate = state_entry["integrator"] + error * period
                unsaturated = control.kp * error + control.ki * candidate
                duty = min(max(unsaturated, control.duty_min), control.duty_max)
                saturated = unsaturated != duty
                # Conditional integration: hold the integrator only while the
                # error drives further INTO the active limit; an error pulling
                # out of saturation must integrate or the loop deadlocks at the
                # limit (classic anti-windup pitfall).
                drives_deeper = (unsaturated > duty and error > 0.0) or (
                    unsaturated < duty and error < 0.0
                )
                if not (saturated and drives_deeper):
                    state_entry["integrator"] = candidate
                off_time = time + duty * period
                if off_time < settings.stop - self.time_tolerance:
                    bisect.insort(
                        scheduled, (off_time, control.id, False), key=edge_key
                    )
                if len(control_trace) >= MAX_RUN_EVENT_ROWS:
                    raise SimulationError(
                        "Sampled-control trace exceeds the portable artifact "
                        f"limit of {MAX_RUN_EVENT_ROWS} rows"
                    )
                control_trace.append(
                    {
                        "time_s": float(time),
                        "control_id": control.id,
                        "cycle_index": int(cycle),
                        "sense_value": float(sense_value),
                        "error": float(error),
                        "integrator": float(state_entry["integrator"]),
                        "duty_command": float(duty),
                        "saturated": bool(saturated),
                    }
                )

        edge_index = 0
        source_breakpoints = self.circuit.source_breakpoints(
            settings.start, settings.stop
        )
        # A forced-gate window edge is the same kind of object as a waveform
        # discontinuity: a pure integration-restart boundary. Reusing that path
        # means the override never lands mid-step.
        if self.model.forced_gate_windows:
            boundaries = {
                boundary
                for window in self.model.forced_gate_windows
                for boundary in (window.from_s, window.to_s)
                if settings.start < boundary < settings.stop
            }
            source_breakpoints = sorted(set(source_breakpoints) | boundaries)
        signal_graph = getattr(self.model, "signals", None)
        if signal_graph is not None:
            # The signal task must be able to run at ITS instants, so those
            # instants become integration boundaries. Reusing the waveform
            # breakpoint machinery rather than injecting a no-op gate edge:
            # a sampling instant is a restart boundary, not a gate event, and
            # a model may declare a signal layer with no gates at all.
            source_breakpoints = sorted(
                set(source_breakpoints)
                | {
                    instant
                    for instant in signal_graph.task.instants(
                        settings.start, settings.stop
                    )
                    if settings.start < instant < settings.stop
                }
            )
        breakpoint_index = 0
        sample_times = _sample_times(settings.start, settings.stop, settings.output_step)
        sample_index = 0
        recorded_time: list[float] = []
        recorded_signals = {output: [] for output in self.model.outputs}
        events: list[SimulationEvent] = []
        integration_steps = 0
        natural_events = 0
        recent_natural_times: list[float] = []
        livelock_window = 1000.0 * self.time_tolerance
        mode_intervals: list[dict[str, Any]] = []
        last_half_start = 0.5 * (settings.start + settings.stop)
        last_half_state_extrema: dict[str, dict[str, float]] = {}
        current_time = settings.start

        def retain_events(new_events: list[SimulationEvent]) -> None:
            if len(events) + len(new_events) > MAX_RUN_EVENT_ROWS:
                raise SimulationError(
                    "Run event log exceeds the portable artifact limit of "
                    f"{MAX_RUN_EVENT_ROWS} rows"
                )
            events.extend(new_events)

        machine_outputs = frozenset(
            output for output in self.model.outputs if is_machine_signal(output)
        )
        declared_signal_outputs = {
            output: declared_signal_target(output)
            for output in self.model.outputs
            if is_declared_signal(output)
        }
        signal_values: dict[str, float] = {}
        signal_previous: dict[str, float] = {}
        signal_task_state = {"last_tick": -1.0}
        signal_trace: list[dict[str, Any]] = []
        signal_trace_duty: dict[str, float] = {}
        signal_buffers: dict[str, list[float]] = {}
        signal_trace_sector: dict[str, dict[str, float]] = {}

        def fire_signal_task(
            time: float, sample_state: np.ndarray, sample_mode: CompiledMode
        ) -> None:
            """Run the declared signal graph once, at a task instant.

            Ordering matches the sampled controllers: the graph reads the
            PRE-EDGE state and pre-edge topology, so what the controller sees
            at a shared timestamp is what the kernel had before any gate moved
            there. The previous sample is frozen BEFORE evaluation, which is
            what makes `unit_delay` mean the previous sample rather than
            "whatever this pass has written so far".
            """

            if signal_graph is None:
                return
            task = signal_graph.task
            tick = round((time - settings.start - task.offset) / task.period)
            boundary = settings.start + task.offset + tick * task.period
            if abs(time - boundary) > self.time_tolerance:
                return
            if tick <= signal_task_state["last_tick"]:
                return
            signal_task_state["last_tick"] = float(tick)

            def read_probe(expression: str) -> float:
                if is_machine_signal(expression):
                    return machine_driver.sample(expression, sample_state, time)
                return sample_mode.output(expression, sample_state, time)

            previous = dict(signal_values)
            resolved: dict[str, float] = {}
            blocks = signal_graph.by_id
            for block_id in signal_graph.order:
                resolved[block_id] = evaluate_block(
                    blocks[block_id],
                    resolved,
                    previous,
                    time=time,
                    probe_reader=read_probe,
                    tick=int(tick),
                    buffers=signal_buffers,
                )
            # Shift every multi-tick delay AFTER the pass, so a delay reads
            # the sample from N ticks ago rather than one written during this
            # pass — the same reason unit_delay reads `previous`.
            for block in signal_graph.blocks:
                if block.kind != "delay":
                    continue
                queue = signal_buffers.setdefault(block.id, [])
                queue.append(resolved[block.inputs[0]])
                depth = int(block.params["samples"])
                if len(queue) > depth:
                    del queue[: len(queue) - depth]
            signal_previous.clear()
            signal_previous.update(previous)
            signal_values.clear()
            signal_values.update(resolved)

            # Bridge OUT: one tick commands the cycle that starts here. Same
            # law as the built-in sampled PI — gate on for the first `duty`
            # fraction — so the only difference between the two is WHERE the
            # duty came from.
            for control in self.model.sampled_controls.values():
                if getattr(control, "kind", "") != "signal_pwm":
                    continue
                block_id = declared_signal_target(control.duty_signal)
                raw_duty = resolved[block_id]
                duty = min(max(raw_duty, control.duty_min), control.duty_max)
                off_time = time + duty * control.period
                if off_time < settings.stop - self.time_tolerance:
                    bisect.insort(
                        scheduled, (off_time, control.id, False), key=edge_key
                    )
                signal_trace_duty[control.id] = float(duty)

            svpwm_seen: set[str] = set()
            for control in self.model.sampled_controls.values():
                if getattr(control, "kind", "") != "signal_svpwm":
                    continue
                if control.block_id in svpwm_seen:
                    continue
                svpwm_seen.add(control.block_id)
                alpha = resolved[declared_signal_target(control.alpha_signal)]
                beta = resolved[declared_signal_target(control.beta_signal)]
                duties, observation = svpwm_duties(
                    alpha, beta, control.dc_link_voltage
                )
                period = control.period
                for phase_label, duty in zip(("a", "b", "c"), duties, strict=True):
                    high_id = f"{control.block_id}.{phase_label}.high"
                    low_id = f"{control.block_id}.{phase_label}.low"
                    on_span = duty * period
                    start = time + 0.5 * (period - on_span)
                    stop_edge = start + on_span
                    for edge_time, gate_id, value in (
                        (start, low_id, False),
                        (start + control.dead_time, high_id, True),
                        (stop_edge, high_id, False),
                        (stop_edge + control.dead_time, low_id, True),
                    ):
                        if edge_time < settings.stop - self.time_tolerance:
                            bisect.insort(
                                scheduled, (edge_time, gate_id, value), key=edge_key
                            )
                    signal_trace_duty[f"{control.block_id}.{phase_label}"] = float(duty)
                signal_trace_sector[control.block_id] = observation
            if len(signal_trace) < MAX_RUN_EVENT_ROWS:
                signal_trace.append(
                    {
                        "time_s": float(time),
                        "tick_index": int(tick),
                        "values": {
                            block_id: float(value)
                            for block_id, value in sorted(resolved.items())
                        },
                        "commanded_duty": {
                            control_id: value
                            for control_id, value in sorted(
                                signal_trace_duty.items()
                            )
                        },
                    }
                )


        def record(time: float, sample_state: np.ndarray, sample_mode: CompiledMode) -> None:
            recorded_time.append(float(time))
            for output in self.model.outputs:
                if output in machine_outputs:
                    recorded_signals[output].append(
                        machine_driver.sample(output, sample_state, time)
                    )
                    continue
                if output in declared_signal_outputs:
                    # Zero-order hold between task instants: the value a
                    # discrete block holds until its next update IS its value.
                    recorded_signals[output].append(
                        signal_values.get(declared_signal_outputs[output], 0.0)
                    )
                    continue
                recorded_signals[output].append(
                    sample_mode.output(output, sample_state, time)
                )

        fire_sampled_controllers(current_time, state, mode)
        fire_signal_task(current_time, state, mode)
        record(current_time, state, mode)
        sample_index = 1

        while current_time < settings.stop:
            next_edge_time = (
                scheduled[edge_index][0] if edge_index < len(scheduled) else settings.stop
            )
            while (
                breakpoint_index < len(source_breakpoints)
                and source_breakpoints[breakpoint_index] <= current_time
            ):
                breakpoint_index += 1
            next_breakpoint_time = (
                source_breakpoints[breakpoint_index]
                if breakpoint_index < len(source_breakpoints)
                else settings.stop
            )
            # A waveform discontinuity is a pure integration-restart boundary:
            # _apply_scheduled_edges consumes no gate edges there and only
            # re-selects the diode active set against the post-boundary u(t).
            target_time = min(next_edge_time, next_breakpoint_time, settings.stop)
            if machine_driver.active:
                target_time = min(
                    target_time, current_time + machine_driver.update_step
                )
            if target_time <= current_time:
                fire_sampled_controllers(target_time, state, mode)
                fire_signal_task(target_time, state, mode)
                previous_constraints = frozenset(mode.constrained_inductors)
                state, gates, diodes, mode, new_events, edge_index = self._apply_scheduled_edges(
                    target_time, state, gates, diodes, scheduled, edge_index
                )
                state, projection_count = self._project_constrained_state(state, mode)
                constrained_state_projections += projection_count
                if frozenset(mode.constrained_inductors) - previous_constraints:
                    state_constraint_events += 1
                metric_accumulator._record_mode(mode)
                retain_events(new_events)
                current_time = target_time
                sample_delta = (
                    float(sample_times[sample_index]) - current_time
                    if sample_index < len(sample_times)
                    else math.inf
                )
                if -self.time_tolerance <= sample_delta <= 0.0:
                    record(float(sample_times[sample_index]), state, mode)
                    sample_index += 1
                continue

            if machine_driver.active:
                machine_driver.begin_segment(current_time, state)
            event_functions = self._natural_event_functions(
                mode, diodes, current_time, state, gates
            )
            method = {
                "dopri5": "RK45",
                "dop853": "DOP853",
                "radau": "Radau",
                "bdf": "BDF",
            }[settings.solver]
            max_step = settings.max_step or (target_time - current_time)
            if self.backend == "rust":
                solution = rust_backend.solve_segment(
                    mode,
                    current_time,
                    target_time,
                    state,
                    rtol=settings.rtol,
                    atol=settings.atol,
                    max_step=max_step,
                    events=event_functions or None,
                )
            else:
                solution = solve_ivp(
                    mode.derivative,
                    (current_time, target_time),
                    state,
                    method=method,
                    rtol=settings.rtol,
                    atol=settings.atol,
                    max_step=max_step,
                    dense_output=True,
                    events=event_functions or None,
                )
            integration_steps += max(0, len(solution.t) - 1)
            if not solution.success or solution.sol is None:
                raise SimulationError(
                    f"Integrator failed at t={current_time:.12g}: {solution.message}"
                )
            if float(solution.t[-1]) > float(solution.t[0]):
                interval_start = float(solution.t[0])
                interval_end = float(solution.t[-1])
                # The published `state_extrema` observation covers the last-half
                # window only; per-interval extrema outside it exist solely for
                # the `mode_intervals` audit trail. A lean run therefore searches
                # roots only where the reported window needs them — which leaves
                # that reported observation bit-for-bit identical while skipping
                # roughly half of the most expensive work in the run.
                report_extrema = self.evidence_mode == "full"
                if report_extrema:
                    interval_state_extrema = _dense_state_extrema(
                        solution,
                        mode,
                        interval_start,
                        interval_end,
                    )
                else:
                    interval_state_extrema = None
                if interval_end > last_half_start:
                    if interval_start >= last_half_start and interval_state_extrema is not None:
                        tail_extrema = interval_state_extrema
                    else:
                        tail_extrema = _dense_state_extrema(
                            solution,
                            mode,
                            max(interval_start, last_half_start),
                            interval_end,
                        )
                    _merge_state_extrema(last_half_state_extrema, tail_extrema)
                mode_intervals.append(
                    {
                        "start_s": interval_start,
                        "end_s": interval_end,
                        "mode_signature": mode.mode_signature,
                        "gates": dict(mode.topology.gates),
                        "diodes": dict(mode.topology.diodes),
                        "constrained_inductors": list(mode.constrained_inductors),
                        "state_start": _state_values_by_component(
                            self.circuit,
                            np.asarray(solution.y[:, 0], dtype=float),
                        ),
                        "state_end": _state_values_by_component(
                            self.circuit,
                            np.asarray(solution.y[:, -1], dtype=float),
                        ),
                        "state_extrema_method": (
                            _STATE_EXTREMA_METHOD
                            if interval_state_extrema is not None
                            else _STATE_EXTREMA_NOT_COMPUTED
                        ),
                        "state_extrema": interval_state_extrema,
                    }
                )
            metric_accumulator.add_solution(solution, mode)
            harmonic_accumulator.add_solution(solution, mode)
            if machine_driver.active:
                machine_driver.end_segment(
                    solution, float(solution.t[0]), float(solution.t[-1])
                )

            actual_end = float(solution.t[-1])
            while (
                sample_index < len(sample_times)
                and sample_times[sample_index] < actual_end - self.time_tolerance
            ):
                sample_time = float(sample_times[sample_index])
                record(sample_time, np.asarray(solution.sol(sample_time), dtype=float), mode)
                sample_index += 1

            state = np.asarray(solution.y[:, -1], dtype=float)
            at_target = _same_integration_endpoint(actual_end, target_time)
            roots = _terminal_roots(solution.t_events, actual_end, self.time_tolerance)
            scheduled_at_target = (
                edge_index < len(scheduled) and scheduled[edge_index][0] == target_time
            )
            if at_target and scheduled_at_target:
                # solve_ivp reports a terminal event only after a sign crossing.  At
                # exact BCM the conducting-diode current can instead approach zero
                # from above and land on the scheduled edge within roundoff, so the
                # root has no sample on the other side.  Resolve that continuous
                # transition before applying the discontinuous gate update.
                roots = sorted(
                    set(
                        roots
                        + self._coincident_natural_roots(
                            mode,
                            diodes,
                            actual_end,
                            state,
                        )
                    )
                )
            previous_time = current_time
            current_time = actual_end
            if roots:
                natural_events += len(roots)
                if current_time <= previous_time + self.time_tolerance:
                    raise SimulationError(
                        f"Non-advancing diode event at t={current_time:.12g}; possible chattering"
                    )
                recent_natural_times.extend([current_time] * len(roots))
                if len(recent_natural_times) > 20:
                    del recent_natural_times[:-20]
                if (
                    len(recent_natural_times) == 20
                    and recent_natural_times[-1] - recent_natural_times[0]
                    <= livelock_window
                ):
                    hint = (
                        " Dead-time freewheeling with a near-zero phase current "
                        "at a floating star point is outside v0.1's contract: "
                        "use zero dead time or tie the load neutral."
                        if getattr(mode, "star_constraints", ())
                        else ""
                    )
                    raise SimulationError(
                        "Natural-commutation livelock: 20 events within "
                        f"{livelock_window:.3e} s at t={current_time:.12g}."
                        + hint
                    )
                diode_count = len(self.circuit.diodes)
                for root_index in roots:
                    if root_index >= diode_count:
                        # Saturation-boundary root: no discrete state to toggle;
                        # stable_mode below re-derives the region from the state.
                        component = self.circuit.sat_inductors[
                            (root_index - diode_count) // 2
                        ]
                        previously_saturated = (
                            component.id in mode.saturated_inductors
                        )
                        retain_events(
                            [
                                SimulationEvent(
                                    current_time,
                                    "saturation_boundary",
                                    component.id,
                                    previously_saturated,
                                    not previously_saturated,
                                )
                            ]
                        )
                        continue
                    diode = self.circuit.diodes[root_index]
                    previous = diodes[diode.id]
                    diodes[diode.id] = not previous
                    retain_events(
                        [
                            SimulationEvent(
                                current_time,
                                "natural_commutation",
                                diode.id,
                                previous,
                                not previous,
                            )
                        ]
                    )
                previous_constraints = frozenset(mode.constrained_inductors)
                stabilized, mode = self.circuit.stable_mode(
                    gates,
                    diodes,
                    state,
                    self.guard_tolerance,
                    time=current_time,
                )
                state, projection_count = self._project_constrained_state(state, mode)
                constrained_state_projections += projection_count
                if frozenset(mode.constrained_inductors) - previous_constraints:
                    state_constraint_events += 1
                metric_accumulator._record_mode(mode)
                for diode_id, current in stabilized.items():
                    previous = diodes[diode_id]
                    if current != previous:
                        retain_events(
                            [
                                SimulationEvent(
                                    current_time,
                                    "switch_manager",
                                    diode_id,
                                    previous,
                                    current,
                                )
                            ]
                        )
                diodes = stabilized
                if not at_target:
                    sample_delta = (
                        float(sample_times[sample_index]) - current_time
                        if sample_index < len(sample_times)
                        else math.inf
                    )
                    if -self.time_tolerance <= sample_delta <= 0.0:
                        record(float(sample_times[sample_index]), state, mode)
                        sample_index += 1
                    continue

            if abs(current_time - target_time) > self.time_tolerance:
                raise SimulationError("Integrator stopped before the next scheduled event")
            if target_time < settings.stop or scheduled_at_target:
                fire_sampled_controllers(target_time, state, mode)
                fire_signal_task(target_time, state, mode)
                previous_constraints = frozenset(mode.constrained_inductors)
                state, gates, diodes, mode, new_events, edge_index = self._apply_scheduled_edges(
                    target_time, state, gates, diodes, scheduled, edge_index
                )
                state, projection_count = self._project_constrained_state(state, mode)
                constrained_state_projections += projection_count
                if frozenset(mode.constrained_inductors) - previous_constraints:
                    state_constraint_events += 1
                metric_accumulator._record_mode(mode)
                retain_events(new_events)
            while (
                sample_index < len(sample_times)
                and sample_times[sample_index] <= current_time + self.time_tolerance
            ):
                record(float(sample_times[sample_index]), state, mode)
                sample_index += 1

        if sample_index < len(sample_times):
            record(float(sample_times[-1]), state, mode)

        time_integrals, continuous_outputs, physics_audit = metric_accumulator.finalize(
            state,
            self.time_tolerance,
        )
        harmonic_analysis = harmonic_accumulator.finalize()
        missing_extrema = {component.id for component in self.circuit.state_components} - set(
            last_half_state_extrema
        )
        if missing_extrema:
            raise SimulationError(
                "Dense state-extrema observation did not cover the last-half window for: "
                + ", ".join(sorted(missing_extrema))
            )
        state_extrema = {
            "method": _STATE_EXTREMA_METHOD,
            "window": "last_half",
            "start_s": last_half_start,
            "end_s": settings.stop,
            "states": {
                component.id: {
                    "unit": "A" if component.type in {"inductor", "sat_inductor"} else "V",
                    **last_half_state_extrema[component.id],
                }
                for component in sorted(
                    self.circuit.state_components,
                    key=lambda item: item.id,
                )
            },
        }
        wall_time_s = perf_counter() - wall_time_start

        return SimulationResult(
            model=self.model,
            time=np.asarray(recorded_time),
            signals={name: np.asarray(values) for name, values in recorded_signals.items()},
            events=events,
            stats={
                "integration_steps": integration_steps,
                "natural_events": natural_events,
                "scheduled_events": sum(event.kind == "scheduled_gate" for event in events),
                **self.circuit.compiled_topology_stats(),
                "state_constraint_events": state_constraint_events,
                "constrained_state_projections": constrained_state_projections,
                "mode_intervals": len(mode_intervals),
                "samples": len(recorded_time),
                "integrated_interval_s": time_integrals["full"]["duration_s"],
                "metric_topology_segments": metric_accumulator.topology_segments,
                "metric_quadrature_intervals": time_integrals["full"]["quadrature_intervals"],
                # Additive-only: models without sampled controls keep their
                # exact pre-feature stats payload (hash-neutral contracts).
                **(
                    {"sampled_control_updates": len(control_trace)}
                    if self.model.sampled_controls
                    else {}
                ),
                "wall_time_s": wall_time_s,
            },
            time_integrals=time_integrals,
            continuous_outputs=continuous_outputs,
            physics_audit=physics_audit,
            mode_intervals=mode_intervals,
            state_extrema=state_extrema,
            modulation_cycles=gate_schedule.modulation_cycles,
            harmonic_analysis=harmonic_analysis,
            control_trace=control_trace,
            machine_trace=machine_driver.trace,
            signal_trace=signal_trace,
            evidence_mode=self.evidence_mode,
            backend=self.backend_id,
        )

    def _assert_star_invariants(
        self, mode: CompiledMode, state: np.ndarray
    ) -> None:
        """Fail closed when the initial state violates KCL at a floating star.

        The reduced mode preserves the invariant exactly; only the declared
        initial currents can break it, and silently projecting them would
        fabricate a state the user never declared.
        """

        for node_name, terms in getattr(mode, "star_constraints", ()):
            residual = sum(sign * float(state[index]) for index, sign in terms)
            if abs(residual) > STATE_CONSTRAINT_TOLERANCE:
                raise SimulationError(
                    f"Initial inductor currents violate KCL at the floating "
                    f"star point {node_name!r}: residual {residual:.12g} A. "
                    "Declare initial_current values that sum to zero at the "
                    "star point."
                )

    def _project_constrained_state(
        self,
        state: np.ndarray,
        mode: CompiledMode,
    ) -> tuple[np.ndarray, int]:
        projected = state
        corrections = 0
        for component_id in mode.constrained_inductors:
            index = self.circuit.state_index[component_id]
            value = float(state[index])
            if abs(value) > self.constraint_tolerance:
                raise SimulationError(
                    f"Mode {mode.mode_signature} cannot constrain {component_id}="
                    f"{value:.12g} A to zero"
                )
            if value != 0.0:
                if projected is state:
                    projected = state.copy()
                projected[index] = 0.0
                corrections += 1
        return projected, corrections

    def _natural_event_functions(
        self,
        mode: CompiledMode,
        diodes: dict[str, bool],
        segment_start: float,
        start_state: np.ndarray | None = None,
        gates: dict[str, bool] | None = None,
    ) -> list[Callable[[float, np.ndarray], float]]:
        functions: list[Callable[[float, np.ndarray], float]] = []
        delay = self.time_tolerance * 10.0
        # Directions the mode's flow cannot change: star-KCL invariant vectors
        # and frozen (zero-current constrained) states. A guard whose state
        # coefficients lie in their span is CONSTANT along every trajectory of
        # this mode — float noise on the invariant (~1e-13) must not be read
        # as a crossing (dead-time chattering at a floating star).
        invariant_directions: list[np.ndarray] = []
        state_count = len(self.circuit.state_components)
        for _node, terms in getattr(mode, "star_constraints", ()):
            vector = np.zeros(state_count)
            for state_index, sign in terms:
                vector[state_index] = sign
            invariant_directions.append(vector)
        for state_index in mode.constrained_state_indices:
            vector = np.zeros(state_count)
            vector[state_index] = 1.0
            invariant_directions.append(vector)
        invariant_basis = (
            np.column_stack(invariant_directions) if invariant_directions else None
        )

        def constant_along_flow(coefficients: np.ndarray) -> bool:
            if not np.any(coefficients != 0.0):
                # No state dependence at all: the guard is the bare affine
                # constant, which elimination dust can leave at ~1e-13 instead
                # of an exact 0.0 whenever the declared parameters are not
                # binary-representable. It cannot cross inside the
                # segment regardless of any invariant subspace.
                return True
            if invariant_basis is None:
                return False
            solution, _res, _rank, _sv = np.linalg.lstsq(
                invariant_basis, coefficients, rcond=None
            )
            residual = coefficients - invariant_basis @ solution
            scale = max(1.0, float(np.max(np.abs(coefficients))))
            return bool(np.max(np.abs(residual)) <= 1e-9 * scale)

        for diode in self.circuit.diodes:
            conducting = diodes[diode.id]
            if (
                not conducting
                and diode.id in self.circuit.latching
                and not self.circuit._gate_permits_latch(diode.id, gates or {})
            ):
                # An UNGATED blocking thyristor has no turn-on guard: forward
                # voltage is not a crossing for it, only a gate edge is, and
                # gate edges arrive as scheduled events. With the gate held
                # high the device must fire the moment it becomes forward
                # biased, so its voltage guard stays live in that case. The
                # event list keeps its diode-index mapping either way, so this
                # slot is a constant that never roots.
                def blocked_latch(time: float, state: np.ndarray) -> float:
                    return -self.guard_tolerance

                blocked_latch.terminal = True  # type: ignore[attr-defined]
                blocked_latch.direction = 1.0  # type: ignore[attr-defined]
                blocked_latch.pear_spec = ("constant", -self.guard_tolerance)  # type: ignore[attr-defined]
                functions.append(blocked_latch)
                continue
            guard_coefficients, guard_constant, guard_sources = mode.diode_guard_affine(
                diode, conducting
            )
            # An open diode parallel to a closed ideal switch has an identically
            # zero voltage guard in this topology.  Zero is a feasible boundary,
            # not a crossing: treating it as an event creates a candidate toggle
            # that the active-set manager immediately undoes.  Preserve the event
            # list's diode-index mapping, but make this guard a directed constant
            # until the next topology change.  A guard coupled to any waveform
            # source is time-varying and therefore never identically zero.
            # The classification is STRUCTURAL for a blocking diode whose
            # terminals the mode's zero-voltage branches short together: the
            # solved affine row carries platform-dependent elimination dust
            # (~1e-13 in the constant AND the coefficients), and reading that
            # dust as a crossing is the constant-guard livelock. The numeric all-zero
            # test stays for rows that do cancel exactly.
            guard_is_identically_zero = (
                guard_constant == 0.0
                and not np.any(guard_coefficients != 0.0)
                and not np.any(guard_sources != 0.0)
            ) or (
                not conducting
                and mode.nodes_structurally_shorted(diode.p, diode.n)
            )
            if (
                not guard_is_identically_zero
                and not np.any(guard_sources != 0.0)
                and constant_along_flow(guard_coefficients)
            ):
                # Constant along the flow: its exact value never changes inside
                # this segment, so it cannot cross zero. Hold it at a directed
                # non-crossing constant (its entry value if clearly nonzero).
                if start_state is not None:
                    entry_value = float(
                        guard_coefficients @ start_state + guard_constant
                    )
                else:
                    entry_value = 0.0
                if abs(entry_value) > self.guard_tolerance:
                    guard_hold: float | None = entry_value
                else:
                    guard_hold = (
                        self.guard_tolerance
                        if conducting
                        else -self.guard_tolerance
                    )
            else:
                guard_hold = None

            def event(
                time: float,
                state: np.ndarray,
                *,
                component=diode,
                is_conducting=conducting,
                identically_zero=guard_is_identically_zero,
                hold=guard_hold,
            ) -> float:
                if identically_zero:
                    return self.guard_tolerance if is_conducting else -self.guard_tolerance
                if hold is not None:
                    return hold
                if time <= segment_start + delay:
                    return self.guard_tolerance if is_conducting else -self.guard_tolerance
                return mode.diode_guard(component, state, is_conducting, time)

            event.terminal = True  # type: ignore[attr-defined]
            event.direction = -1.0 if conducting else 1.0  # type: ignore[attr-defined]
            # The same guard as data, for the compiled kernel. It reads the
            # affine row directly; this closure reaches the same value through
            # the full node solution, so the two agree to rounding.
            restart_value = self.guard_tolerance if conducting else -self.guard_tolerance
            if guard_is_identically_zero:
                event.pear_spec = ("constant", restart_value)  # type: ignore[attr-defined]
            elif guard_hold is not None:
                event.pear_spec = ("constant", guard_hold)  # type: ignore[attr-defined]
            elif not np.any(guard_sources != 0.0):
                event.pear_spec = (  # type: ignore[attr-defined]
                    "affine",
                    np.asarray(guard_coefficients, dtype=float),
                    float(guard_constant),
                    segment_start + delay,
                    restart_value,
                )
            functions.append(event)

        for component in self.circuit.sat_inductors:
            state_index = self.circuit.state_index[component.id]
            boundary = float(component.sat_current)
            saturated = component.id in mode.saturated_inductors
            for sign in (1.0, -1.0):
                level = sign * boundary
                # Hold sign during the restart delay: the side the start state
                # actually sits on; exactly on the boundary the accepted region
                # tells which side this segment will move into.
                if start_state is not None:
                    start_guard = float(start_state[state_index]) - level
                else:
                    start_guard = 0.0
                if abs(start_guard) > self.guard_tolerance:
                    hold = math.copysign(self.guard_tolerance, start_guard)
                elif saturated:
                    hold = self.guard_tolerance if sign > 0 else -self.guard_tolerance
                else:
                    hold = -self.guard_tolerance if sign > 0 else self.guard_tolerance

                # Guard i -/+ i_sat: hitting either signed boundary is a pure
                # integration-restart point; the region itself is re-derived
                # from the state by stable_mode, so a missed direction can
                # never latch a wrong segment.
                def saturation_event(
                    time: float,
                    state: np.ndarray,
                    *,
                    index=state_index,
                    guard_level=level,
                    hold_value=hold,
                ) -> float:
                    if time <= segment_start + delay:
                        return hold_value
                    return float(state[index]) - guard_level

                saturation_event.terminal = True  # type: ignore[attr-defined]
                saturation_event.direction = 0.0  # type: ignore[attr-defined]
                unit_row = np.zeros(state_count)
                unit_row[state_index] = 1.0
                saturation_event.pear_spec = (  # type: ignore[attr-defined]
                    "affine", unit_row, -level, segment_start + delay, hold
                )
                functions.append(saturation_event)
        return functions

    def _coincident_natural_roots(
        self,
        mode: CompiledMode,
        diodes: dict[str, bool],
        time: float,
        state: np.ndarray,
    ) -> list[int]:
        """Return directed zero guards reached exactly on a scheduled boundary.

        This is deliberately narrower than general near-zero stabilization: the
        caller invokes it only after the integrator has reached a scheduled edge.
        Requiring both a near-zero guard and the natural crossing direction avoids
        manufacturing an event for an idle diode whose guard is identically zero.
        A guard-scale check alone is insufficient because a small guard with a slow
        derivative can still be far from its physical root in time.  The linearized
        time-to-zero check below restricts the merge to roots coincident with the
        scheduled edge at the simulator's time resolution.
        """

        state_derivative = mode.derivative(time, state)
        source_slopes = (
            self.circuit.source_slopes(time) if mode.source_count else None
        )
        roots: list[int] = []
        for index, diode in enumerate(self.circuit.diodes):
            conducting = diodes[diode.id]
            if not conducting and mode.nodes_structurally_shorted(
                diode.p, diode.n
            ):
                # Structurally zero guard: a feasible boundary, never a
                # crossing — elimination dust in its slope must not
                # manufacture a root on the scheduled edge.
                continue
            guard = mode.diode_guard(diode, state, conducting, time)
            if abs(guard) > self.guard_tolerance:
                continue

            coefficients, _constant, source_coefficients = mode.diode_guard_affine(
                diode, conducting
            )
            slope = float(coefficients @ state_derivative)
            if source_slopes is not None:
                slope += float(source_coefficients @ source_slopes)
            directed = (conducting and slope < 0.0) or (not conducting and slope > 0.0)
            if not directed:
                continue
            time_to_zero = abs(guard / slope)
            coincidence_tolerance = max(
                self.time_tolerance,
                8.0 * math.ulp(time),
            )
            if time_to_zero <= coincidence_tolerance:
                roots.append(index)
        return roots

    def _apply_scheduled_edges(
        self,
        time: float,
        state: np.ndarray,
        gates: dict[str, bool],
        diodes: dict[str, bool],
        scheduled: list[tuple[float, str, bool]],
        edge_index: int,
    ) -> tuple[
        np.ndarray,
        dict[str, bool],
        dict[str, bool],
        CompiledMode,
        list[SimulationEvent],
        int,
    ]:
        new_gates = dict(gates)
        new_events: list[SimulationEvent] = []
        if self.model.forced_gate_windows:
            # Two states, not one. `self._natural_gates` is what the schedule
            # says — including a duty a closed-loop controller computed during
            # the run — and `new_gates` is what the circuit actually sees after
            # the override mask. Seeding the natural shadow from the effective
            # gates would bake a forced level into it and the gate would never
            # come back when the window ended.
            #
            # Exactly one event per control per timestamp, carrying the true
            # effective prior value: downstream observers verify that a gate
            # event's `previous` matches the state they are tracking, so a
            # silent override would fail their continuity check.
            while edge_index < len(scheduled) and scheduled[edge_index][0] == time:
                _, control_id, value = scheduled[edge_index]
                self._natural_gates[control_id] = value
                edge_index += 1
            new_gates = self._force_gate_windows(dict(self._natural_gates), time)
            for control_id, value in new_gates.items():
                previous = gates[control_id]
                if previous == value:
                    continue
                overridden = any(
                    window.control_id == control_id and window.contains(time)
                    for window in self.model.forced_gate_windows
                )
                new_events.append(
                    SimulationEvent(
                        time,
                        "forced_gate" if overridden else "scheduled_gate",
                        control_id,
                        previous,
                        value,
                    )
                )
        else:
            while edge_index < len(scheduled) and scheduled[edge_index][0] == time:
                _, control_id, value = scheduled[edge_index]
                previous = new_gates[control_id]
                new_gates[control_id] = value
                if previous != value:
                    new_events.append(
                        SimulationEvent(
                            time, "scheduled_gate", control_id, previous, value
                        )
                    )
                edge_index += 1
        self._assert_complementary_interlocks(new_gates, time=time)
        stabilized, mode = self.circuit.stable_mode(
            new_gates,
            diodes,
            state,
            self.guard_tolerance,
            time=time,
        )
        for diode_id, current in stabilized.items():
            previous = diodes[diode_id]
            if current != previous:
                new_events.append(
                    SimulationEvent(time, "switch_manager", diode_id, previous, current)
                )
        return state, new_gates, stabilized, mode, new_events, edge_index

    def _force_gate_windows(
        self, gates: dict[str, bool], time: float
    ) -> dict[str, bool]:
        """Override gate levels held by an active forced window at ``time``."""

        for window in self.model.forced_gate_windows:
            if window.contains(time):
                gates[window.control_id] = window.level
        return gates

    def _assert_complementary_interlocks(
        self,
        gates: dict[str, bool],
        *,
        time: float,
    ) -> None:
        """Fail closed if a complementary block ever commands both gates on."""

        checked_pairs: set[tuple[str, str]] = set()
        for control in getattr(self.model, "sampled_controls", {}).values():
            kind = getattr(control, "kind", "sampled_pi_pwm")
            if kind not in {"sampled_dq_current_pwm", "signal_svpwm"}:
                continue
            pair = tuple(sorted((control.id, control.partner_id)))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            if gates.get(pair[0], False) and gates.get(pair[1], False):
                # Naming the violation matters more than catching it. Without
                # this check the same mistake surfaces as "singular or
                # inconsistent ideal topology", which points at the circuit
                # instead of at the block that commanded a shoot-through.
                raise SimulationError(
                    f"Complementary interlock violation for {kind} block "
                    f"{control.block_id!r}.{control.phase_label} at t={time:.17g}: "
                    f"gates {pair[0]!r} and {pair[1]!r} are both on"
                )
        npc_legs: dict[tuple[str, str], dict[str, bool]] = {}
        npc_levels: dict[str, int] = {}
        for control in self.model.controls.values():
            if control.kind not in {
                "complementary_pwm",
                "three_phase_spwm",
                "three_phase_npc_spwm",
            }:
                continue
            block_id = control.block_id
            partner_id = control.partner_id
            if block_id is None or partner_id is None or partner_id not in gates:
                raise SimulationError(
                    f"Complementary gate metadata is incomplete for gate {control.id!r}"
                )
            if control.kind == "three_phase_npc_spwm":
                phase_label = getattr(control, "phase_label", None) or ""
                suffix = control.id.rsplit(".", 1)[-1]
                npc_legs.setdefault((block_id, phase_label), {})[suffix] = gates.get(
                    control.id, False
                )
                npc_levels[block_id] = getattr(control, "levels", 3)
            pair = tuple(sorted((control.id, partner_id)))
            if pair in checked_pairs:
                continue
            checked_pairs.add(pair)
            if gates.get(control.id, False) and gates[partner_id]:
                phase_label = getattr(control, "phase_label", None)
                logical_pair = (
                    f"{block_id}.{phase_label}"
                    if isinstance(phase_label, str) and phase_label
                    else block_id
                )
                raise SimulationError(
                    f"Complementary PWM interlock violation for block {logical_pair!r} "
                    f"at t={time:.17g}: gates {pair[0]!r} and {pair[1]!r} "
                    "are both on"
                )
        for (block_id, phase_label), leg in npc_legs.items():
            # Diode-clamped gating staircase: a top-arm device never conducts
            # without its pole-adjacent neighbor (S_k => S_(k+1)), and the
            # bottom arm mirrors it (S_(2(N-1)) => ... => S_N). Any violation
            # would connect a DC-link tap across part of the link through the
            # clamp path. The three-level outer/inner rule is the N=3 case.
            levels = npc_levels.get(block_id, 3)
            if levels == 3:
                top_chain = ("outer_high", "inner_high")
                bottom_chain = ("outer_low", "inner_low")
            else:
                arm = levels - 1
                top_chain = tuple(f"h{k}" for k in range(1, arm + 1))
                bottom_chain = tuple(f"l{k}" for k in range(arm, 0, -1))
            violation: str | None = None
            for chain in (top_chain, bottom_chain):
                for outer_suffix, inner_suffix in zip(chain, chain[1:]):
                    if leg.get(outer_suffix, False) and not leg.get(
                        inner_suffix, False
                    ):
                        violation = (
                            f"{outer_suffix} on without {inner_suffix}"
                        )
                        break
                if violation is not None:
                    break
            if violation is None and leg.get(top_chain[0], False) and leg.get(
                bottom_chain[0], False
            ):
                violation = (
                    f"{top_chain[0]} and {bottom_chain[0]} both on"
                )
            if violation is not None:
                raise SimulationError(
                    f"NPC gating interlock violation for leg "
                    f"{block_id!r}.{phase_label}: {violation} at t={time:.17g}"
                )


class _MachineDriver:
    """Couples declared machines to the piecewise-LTI electrical segments.

    Within one integration segment the mechanical speed is HELD while the
    rotor angle still advances exactly, so every back-EMF is a pure sinusoid
    the kernel integrates without approximation and the electrical side stays
    strictly linear. The mechanical state is advanced once per segment from
    the quadrature integral of the torque over that segment. The resulting
    splitting error is bounded by the speed change inside one segment,
    (T_e - T_load - B*omega)/J * dt, and shrinks with the segment length —
    it is therefore measurable by tightening, not merely asserted.
    """

    def __init__(self, circuit: Any, model: Model) -> None:
        self.circuit = circuit
        self.machines = tuple(getattr(model, "machines", ()) or ())
        # Holding the speed is only legitimate over a SHORT segment. A model
        # without switching would otherwise integrate its whole window as one
        # segment and update the mechanics in a single leap, so the update is
        # capped at the output resolution the model itself asked for. In a
        # switched converter the gate edges are already far shorter than this
        # and the cap never binds.
        self.update_step = float(model.transient.output_step)
        self.state: dict[str, dict[str, float]] = {}
        self.current_indices: dict[str, tuple[int, int, int]] = {}
        # Start of the segment the held speed belongs to. Sampling a machine
        # signal needs it, because within a segment the angle advances from
        # THIS origin at the held speed.
        self.segment_start: float | None = None
        self._salient_matrices: dict[str, Any] = {}
        # Snapshot of the mechanical state as it entered the current segment.
        # Sampling MUST read this and never `self.state`, because the solver
        # calls `end_segment` — which advances `self.state` — BEFORE it records
        # the samples that fall inside the segment just integrated. Reading the
        # live state there would add the segment's advance a second time.
        self.segment_state: dict[str, dict[str, float]] = {}
        self._spec_by_id = {spec.id: spec for spec in self.machines}
        for spec in self.machines:
            self.state[spec.id] = {
                "speed_rad_s": spec.initial_speed_rad_s,
                "angle_rad": spec.initial_angle_rad,
                "torque_nm": 0.0,
            }
            self.current_indices[spec.id] = tuple(
                circuit.state_index[f"{spec.id}.{label}.l"]
                for label, _ in PHASE_OFFSETS
            )
        self.trace: list[dict[str, Any]] = []

    @property
    def active(self) -> bool:
        return bool(self.machines)

    def begin_segment(self, segment_start: float, state: "np.ndarray | None" = None) -> None:
        """Refresh EMFs and the salient matrices for a new segment.

        ``state`` is the live solver state vector; when given, the winding
        currents are transformed to keep the flux linkage continuous across
        the matrix swap. ``None`` (the t=start call) means there is no prior
        matrix to be continuous WITH, so no transform happens.
        """

        self.segment_start = float(segment_start)
        for spec in self.machines:
            if not spec.is_salient:
                continue
            # A salient rotor changes the winding inductance matrix as it
            # turns. It is HELD across the segment exactly as the speed is,
            # so the electrical side stays strictly linear inside the
            # segment and the splitting error is of the same measurable kind.
            segment_currents = None
            if spec.is_saturating and state is not None:
                segment_currents = tuple(
                    float(state[index]) for index in self.current_indices[spec.id]
                )
            new_matrix = salient_inductance_matrix(
                spec, self.state[spec.id]["angle_rad"], segment_currents
            )
            previous = self._salient_matrices.get(spec.id)
            self.circuit.set_group_inductance_matrix(spec.winding_ids, new_matrix)
            self._salient_matrices[spec.id] = new_matrix
            if previous is not None and state is not None:
                # FLUX IS THE CONSERVED QUANTITY, NOT CURRENT. Between two
                # held-matrix segments the rotor has turned and L(theta) has
                # changed; physically psi = L(theta(t))*i evolves continuously
                # (its derivative is the winding voltage) and the reluctance
                # voltage omega*dL/dtheta*i lives exactly in that continuity.
                # Carrying the CURRENT across the boundary instead makes psi
                # jump by (L_new - L_old)*i every segment — thousands of small
                # jumps per run that add up to a 40-50 % RMS divergence on a
                # rotating salient machine, while a locked rotor (no
                # matrix swaps) shows nothing wrong at all.
                # Solving L_new * i_new = L_old * i_old restores continuity;
                # the explicit zero-sequence entry keeps L_new invertible and
                # maps a zero-sum current set to a zero-sum current set, so
                # the floating star invariant survives the transform exactly.
                indices = [
                    self.circuit.state_index[w] for w in spec.winding_ids
                ]
                currents = np.array([state[i] for i in indices], dtype=float)
                preserved = np.linalg.solve(new_matrix, previous @ currents)
                for index, value in zip(indices, preserved, strict=True):
                    state[index] = float(value)
        for spec in self.machines:
            if spec.driven_speed is not None:
                # An imposed speed is READ at the segment boundary and then
                # HELD across the segment, exactly like an integrated one.
                # Sampling the ramp continuously inside the segment would
                # make the back-EMF a chirp rather than the pure sinusoid
                # the kernel integrates exactly, trading a bounded, shrinking
                # splitting error for an unbounded modelling one.
                self.state[spec.id]["speed_rad_s"] = spec.driven_speed.speed_at(
                    float(segment_start)
                )
        self.segment_state = {
            machine_id: dict(values) for machine_id, values in self.state.items()
        }
        for spec in self.machines:
            mechanical = self.state[spec.id]
            electrical_speed = spec.pole_pairs * mechanical["speed_rad_s"]
            for (label, offset), source_id in zip(
                PHASE_OFFSETS, spec.emf_ids, strict=True
            ):
                amplitude, frequency, phase = back_emf_waveform(
                    spec,
                    electrical_speed=electrical_speed,
                    electrical_angle=mechanical["angle_rad"],
                    segment_start=segment_start,
                    phase_offset=offset,
                )
                self.circuit.set_source_waveform(
                    source_id,
                    SineWaveform(
                        offset=0.0,
                        amplitude=amplitude,
                        frequency_hz=frequency,
                        phase_rad=phase,
                    ),
                )

    def sample(self, expression: str, state: np.ndarray, time: float) -> float:
        """Instantaneous value of a machine signal at ``time``.

        The angle is advanced EXACTLY from the segment origin at the held
        electrical speed — the same law the back-EMF is built from, so the
        recorded angle and the voltage the kernel integrated cannot disagree.
        The torque is evaluated from the instantaneous phase currents at
        ``time``, so it is a point value, not the segment mean that
        ``machine-trace.jsonl`` reports.
        """

        kind, machine_id = parse_signal_expression(expression)
        spec = self._spec_by_id[machine_id]
        mechanical = self.segment_state.get(machine_id)
        if mechanical is None:
            # Before the first segment (the t=start sample) the entry state IS
            # the segment state.
            mechanical = self.state[machine_id]
        speed_mech = mechanical["speed_rad_s"]
        speed_elec = spec.pole_pairs * speed_mech
        if kind == "speed_mech":
            return float(speed_mech)
        if kind == "speed_elec":
            return float(speed_elec)
        # The very first sample is taken at t=start, before any segment has
        # begun; there the entry state IS the value, so the advance is zero.
        origin = self.segment_start if self.segment_start is not None else time
        angle_elec = mechanical["angle_rad"] + speed_elec * (time - origin)
        if kind == "angle_elec":
            return float(angle_elec)
        if kind == "angle_mech":
            # Declared convention: the mechanical angle is the electrical
            # angle divided by the pole-pair count, which fixes the
            # mechanical origin at initial_angle_rad / pole_pairs. The state
            # carries only the electrical angle, so this is a definition the
            # model format states, not a measurement.
            return float(angle_elec / spec.pole_pairs)
        if kind == "torque":
            currents = tuple(
                float(state[index]) for index in self.current_indices[machine_id]
            )
            if spec.is_saturating:
                return float(mapped_torque(spec, currents, angle_elec))
            if spec.is_salient:
                return float(salient_torque(spec, currents, angle_elec))
            return float(electromagnetic_torque(spec, currents, angle_elec))
        raise SimulationError(f"unsupported machine signal kind {kind!r}")

    def end_segment(self, solution: Any, start: float, end: float) -> None:
        """Advance the mechanical state over an integrated segment."""

        duration = end - start
        if duration <= 0.0:
            return
        half_width = 0.5 * duration
        midpoint = 0.5 * (start + end)
        nodes = midpoint + half_width * _QUADRATURE_NODES
        weights = half_width * _QUADRATURE_WEIGHTS
        states = np.asarray(solution.sol(nodes), dtype=float)
        if states.ndim == 1:
            states = states[:, np.newaxis]
        for spec in self.machines:
            mechanical = self.state[spec.id]
            speed = mechanical["speed_rad_s"]
            angle = mechanical["angle_rad"]
            electrical_speed = spec.pole_pairs * speed
            indices = self.current_indices[spec.id]
            torque_integral = 0.0
            mean_torque = 0.0
            for position, weight in enumerate(weights):
                sample = states[:, position]
                currents = tuple(float(sample[index]) for index in indices)
                # The angle advances exactly at the held speed.
                sample_angle = angle + electrical_speed * (
                    float(nodes[position]) - start
                )
                if spec.is_saturating:
                    torque = mapped_torque(spec, currents, sample_angle)
                elif spec.is_salient:
                    torque = salient_torque(spec, currents, sample_angle)
                else:
                    torque = electromagnetic_torque(spec, currents, sample_angle)
                torque_integral += weight * torque
                mean_torque += weight * torque
            mean_torque /= duration
            if spec.driven_speed is None:
                resisting = spec.load_torque_nm + spec.viscous_friction_nm_s * speed
                mechanical["speed_rad_s"] = speed + (
                    torque_integral - resisting * duration
                ) / spec.inertia_kg_m2
            else:
                # The shaft is turned from outside: the torque is the ANSWER,
                # not an input to a mechanical loop. The next segment's speed
                # comes from the profile, read in begin_segment.
                mechanical["speed_rad_s"] = spec.driven_speed.speed_at(end)
            mechanical["angle_rad"] = angle + electrical_speed * duration
            mechanical["torque_nm"] = mean_torque
            if len(self.trace) < MAX_RUN_EVENT_ROWS:
                self.trace.append(
                    {
                        "machine_id": spec.id,
                        "segment_start_s": start,
                        "segment_end_s": end,
                        "electrical_angle_rad": angle,
                        "speed_rad_s": speed,
                        "speed_end_rad_s": mechanical["speed_rad_s"],
                        "mean_torque_nm": mean_torque,
                    }
                )


def _sample_times(start: float, stop: float, step: float) -> np.ndarray:
    count = int(math.floor((stop - start) / step + 1e-12))
    times = start + np.arange(count + 1, dtype=float) * step
    if times[-1] < stop - max(1e-15, (stop - start) * 1e-12):
        times = np.append(times, stop)
    else:
        times[-1] = stop
    return times


def _terminal_roots(
    event_times: list[np.ndarray] | None, actual_end: float, tolerance: float
) -> list[int]:
    roots: list[int] = []
    if event_times is None:
        return roots
    for index, times in enumerate(event_times):
        if len(times) and abs(float(times[-1]) - actual_end) <= tolerance:
            roots.append(index)
    return roots


def _same_integration_endpoint(actual: float, target: float) -> bool:
    """Distinguish an attained endpoint from a nearby terminal root.

    Event merge tolerances are intentionally much wider than floating-point
    endpoint roundoff.  Using an ulp-sized comparison preserves a real, however
    short, interval when a natural event occurs just before a scheduled edge.
    """

    tolerance = 8.0 * max(math.ulp(actual), math.ulp(target))
    return abs(actual - target) <= tolerance
