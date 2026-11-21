# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Loss + thermal post-processing over ideal waveforms.

The kernel stays ideal; this producer layers datasheet lookup tables on the
RECORDED run: conduction power from v_on(i, Tj_eval)*i on the output grid
(trapezoid), per-event switching energies from the energy tables with the
declared linear voltage scaling, a Foster junction-to-case chain solved with
the EXACT piecewise-constant-power recursion, a static TIM+heatsink path
(dynamics deliberately omitted and declared), the fan-PQ/heatsink-impedance
operating point, and an ASTM-E1049 rainflow count over the junction
temperature swing. Every out-of-axis lookup is clamped AND counted; every
validity-envelope violation is flagged; nothing is silently regularized.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from functools import partial
from typing import Any

import numpy as np

from pear_sim.errors import ModelError
from pear_sim.spec.device_cards import (
    DeviceCard,
    FanCard,
    HeatsinkCard,
    LossAnalysisSpec,
)

__all__ = ["LossThermalError", "build_losses_thermal_observation", "rainflow_count"]

LOSSES_THERMAL_SCHEMA = "pear-sim/losses-thermal/v0.1"


class LossThermalError(ModelError):
    """The run cannot support the requested loss/thermal estimate."""

    code = "analysis.losses_thermal_invalid"


class _Clamped1D:
    """Piecewise-linear 1-D lookup with clamped extrapolation counters."""

    def __init__(self, axis: tuple[float, ...], values: tuple[float, ...]):
        self.axis = axis
        self.values = values
        self.below = 0
        self.above = 0

    def __call__(self, position: float) -> float:
        axis = self.axis
        if position <= axis[0]:
            if position < axis[0]:
                self.below += 1
            return self.values[0]
        if position >= axis[-1]:
            if position > axis[-1]:
                self.above += 1
            return self.values[-1]
        index = bisect_left(axis, position)
        left, right = axis[index - 1], axis[index]
        fraction = (position - left) / (right - left)
        return self.values[index - 1] * (1.0 - fraction) + self.values[index] * fraction


def _conduction_voltage_lookup(card: DeviceCard, tj_evaluation_c: float) -> _Clamped1D:
    """Reduce the (i, Tj) table to a 1-D v_on(i) row at the declared Tj."""

    table = card.conduction
    temperatures = table.temperature_axis_c
    if len(temperatures) == 1:
        row = table.voltage_table_v[0]
    else:
        clamped = min(max(tj_evaluation_c, temperatures[0]), temperatures[-1])
        index = bisect_left(temperatures, clamped)
        if index == 0:
            row = table.voltage_table_v[0]
        elif clamped == temperatures[index] if index < len(temperatures) else False:
            row = table.voltage_table_v[index]
        else:
            left_t, right_t = temperatures[index - 1], temperatures[index]
            fraction = (clamped - left_t) / (right_t - left_t)
            row = tuple(
                low * (1.0 - fraction) + high * fraction
                for low, high in zip(
                    table.voltage_table_v[index - 1], table.voltage_table_v[index]
                )
            )
    return _Clamped1D(table.current_axis_a, tuple(row))


def _fan_heatsink_operating_point(
    fan: FanCard, heatsink: HeatsinkCard
) -> tuple[float, float]:
    """Airflow where the fan PQ curve meets the heatsink pressure drop.

    Both curves are piecewise linear: the difference (fan - heatsink) is
    non-increasing, so the crossing is found on a merged breakpoint grid.
    Fail closed when the curves do not cross inside the shared flow range.
    """

    fan_curve = _Clamped1D(fan.flow_axis_m3_per_h, fan.pressure_pa)
    sink_curve = _Clamped1D(heatsink.airflow_axis_m3_per_h, heatsink.pressure_drop_pa)
    low = max(fan.flow_axis_m3_per_h[0], heatsink.airflow_axis_m3_per_h[0])
    high = min(fan.flow_axis_m3_per_h[-1], heatsink.airflow_axis_m3_per_h[-1])
    if high <= low:
        raise LossThermalError(
            f"fan {fan.id!r} and heatsink {heatsink.id!r} flow ranges do not overlap"
        )
    breakpoints = sorted(
        {low, high}
        | {value for value in fan.flow_axis_m3_per_h if low <= value <= high}
        | {value for value in heatsink.airflow_axis_m3_per_h if low <= value <= high}
    )
    previous_flow = breakpoints[0]
    previous_diff = fan_curve(previous_flow) - sink_curve(previous_flow)
    if previous_diff < 0.0:
        raise LossThermalError(
            f"fan {fan.id!r} cannot open the heatsink {heatsink.id!r} flow path "
            "(pressure deficit at minimum flow)"
        )
    for flow in breakpoints[1:]:
        diff = fan_curve(flow) - sink_curve(flow)
        if diff <= 0.0:
            if diff == previous_diff:
                crossing = flow
            else:
                fraction = previous_diff / (previous_diff - diff)
                crossing = previous_flow + fraction * (flow - previous_flow)
            rth = _Clamped1D(heatsink.airflow_axis_m3_per_h, heatsink.rth_k_per_w)
            return crossing, rth(crossing)
        previous_flow, previous_diff = flow, diff
    raise LossThermalError(
        f"fan {fan.id!r} PQ curve never meets heatsink {heatsink.id!r} "
        "pressure drop inside the shared flow range (no operating point)"
    )


def rainflow_count(values: np.ndarray) -> list[dict[str, float]]:
    """ASTM E1049-85 rainflow cycle counting (four-point stack method).

    Returns cycles as ``{"range", "mean", "count"}`` rows with count 1.0 for
    closed cycles and 0.5 for the residue — counting is a FACT about the
    series; any lifetime model belongs elsewhere, with its own provenance.
    """

    series = np.asarray(values, dtype=float)
    if series.ndim != 1 or len(series) < 2:
        return []
    # Reduce to turning points (strict sign changes of the difference).
    deltas = np.diff(series)
    keep = [0]
    for index in range(1, len(series) - 1):
        if deltas[index - 1] == 0.0:
            continue
        if deltas[index] == 0.0 or (deltas[index - 1] > 0) != (deltas[index] > 0):
            keep.append(index)
    keep.append(len(series) - 1)
    points = [float(series[index]) for index in keep]

    cycles: list[dict[str, float]] = []
    stack: list[float] = []
    for point in points:
        stack.append(point)
        while len(stack) >= 3:
            if len(stack) == 3:
                x = abs(stack[1] - stack[0])
                y = abs(stack[2] - stack[1])
                if y < x:
                    break
                # Half cycle from the starting residue side.
                cycles.append(
                    {
                        "range": x,
                        "mean": 0.5 * (stack[0] + stack[1]),
                        "count": 0.5,
                    }
                )
                stack.pop(0)
            else:
                x = abs(stack[-2] - stack[-3])
                y = abs(stack[-1] - stack[-2])
                if y < x:
                    break
                cycles.append(
                    {
                        "range": x,
                        "mean": 0.5 * (stack[-2] + stack[-3]),
                        "count": 1.0,
                    }
                )
                del stack[-3:-1]
    for left, right in zip(stack, stack[1:]):
        cycles.append(
            {"range": abs(right - left), "mean": 0.5 * (left + right), "count": 0.5}
        )
    return [cycle for cycle in cycles if cycle["range"] > 0.0]


def _foster_response(
    power_w: np.ndarray,
    step_s: float,
    stages: tuple,
    ambient_reference_c: float,
    initial_power_w: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Exact junction rise for piecewise-constant power on a uniform grid.

    Returns the temperature series and the final stored energy of the Foster
    states (for the chain energy-balance audit): each stage is R, tau with
    C = tau/R; T_k' = (R_k*P - T_k)/tau_k has the exact PWC-step solution
    T_k[n+1] = T_k[n]*exp(-dt/tau) + R_k*P[n]*(1 - exp(-dt/tau)).
    Each stage starts at its steady rise for ``initial_power_w`` (0 = cold).
    """

    count = len(power_w)
    temperatures = np.full(count, ambient_reference_c, dtype=float)
    stage_states = np.array([stage.resistance_k_per_w * initial_power_w for stage in stages])
    decays = np.array([math.exp(-step_s / stage.time_constant_s) for stage in stages])
    gains = np.array(
        [stage.resistance_k_per_w * (1.0 - decay) for stage, decay in zip(stages, decays)]
    )
    for index in range(count):
        temperatures[index] += float(np.sum(stage_states))
        if index < count - 1:
            stage_states = stage_states * decays + gains * power_w[index]
    stored_j = float(
        sum(
            (stage.time_constant_s / stage.resistance_k_per_w) * state
            for stage, state in zip(stages, stage_states)
        )
    )
    return temperatures, stored_j


def build_losses_thermal_observation(result: Any) -> dict[str, Any]:
    """Produce the losses-thermal artifact for a finished run."""

    model = result.model
    spec: LossAnalysisSpec | None = getattr(model, "loss_analysis", None)
    if spec is None:
        raise LossThermalError("model declares no loss_analysis section")

    time = np.asarray(result.time, dtype=float)
    if len(time) < 3:
        raise LossThermalError("loss analysis needs at least three recorded samples")
    step = model.transient.output_step
    component_by_id = {component.id: component for component in model.components}

    devices_payload: list[dict[str, Any]] = []
    warnings: list[str] = []
    total_power = np.zeros_like(time)

    for binding in spec.devices:
        component = component_by_id.get(binding.component_id)
        if component is None or component.type not in {"ideal_switch", "diode"}:
            raise LossThermalError(
                f"loss_analysis.devices.{binding.component_id} must reference an "
                "ideal_switch or diode component"
            )
        card = spec.cards[binding.card_id]
        if card.kind == "switch" and component.type != "ideal_switch":
            raise LossThermalError(
                f"card {card.id!r} (switch) bound to non-switch "
                f"{binding.component_id!r}"
            )
        if card.kind == "diode" and component.type != "diode":
            raise LossThermalError(
                f"card {card.id!r} (diode) bound to non-diode "
                f"{binding.component_id!r}"
            )
        current_name = f"i({binding.component_id})"
        if current_name not in result.signals:
            raise LossThermalError(
                f"loss analysis requires {current_name} among the recorded outputs"
            )
        current = np.abs(np.asarray(result.signals[current_name], dtype=float))

        # --- switching -----------------------------------------------------
        switching_payload: dict[str, Any] | None = None
        switching_power = np.zeros_like(time)
        if card.switching is not None:
            table = card.switching
            voltage_name = binding.commutation_voltage
            if voltage_name not in result.signals:
                raise LossThermalError(
                    f"loss analysis requires {voltage_name} among the recorded "
                    f"outputs (commutation voltage of {binding.component_id})"
                )
            commutation = np.asarray(result.signals[voltage_name], dtype=float)
            lookups = {
                "e_on_j": _Clamped1D(table.current_axis_a, table.e_on_j)
                if table.e_on_j is not None
                else None,
                "e_off_j": _Clamped1D(table.current_axis_a, table.e_off_j)
                if table.e_off_j is not None
                else None,
                "e_rec_j": _Clamped1D(table.current_axis_a, table.e_rec_j)
                if table.e_rec_j is not None
                else None,
            }
            totals = {"e_on_j": 0.0, "e_off_j": 0.0, "e_rec_j": 0.0}
            counted = {"turn_on": 0, "turn_off": 0, "reverse_recovery": 0}
            unresolved = {"turn_on": 0, "turn_off": 0, "reverse_recovery": 0}
            control_id = component.control
            if component.type == "ideal_switch":
                # Every EFFECTIVE gate change switches the device: the solver
                # records a change inside a forced_gate_windows entry as
                # ``forced_gate``, not ``scheduled_gate``. Skipping it books
                # the window's release edge but not its forced edge.
                edge_kinds = {"scheduled_gate", "forced_gate"}
                edge_target = control_id
            else:
                # A freewheel diode is hard-commutated by the opposite switch
                # (switch_manager) or soft-commutated at zero current
                # (natural_commutation, where E_rec(~0) ~ 0).
                edge_kinds = {"natural_commutation", "switch_manager"}
                edge_target = component.id
            edges = [
                event
                for event in result.events
                if event.kind in edge_kinds
                and event.target == edge_target
                and bool(event.previous) != bool(event.current)
            ]
            # Float-noise allowance for "this event sits exactly on a grid
            # sample". It must stay far below the grid step: a quarter-step
            # anchor once read the sample 0.15 us BEFORE an off-grid turn-on as
            # its post-edge sample and booked E_on = 0 for a conducting pulse.
            on_grid = 1e-6 * step
            for position, event in enumerate(edges):
                if component.type == "ideal_switch":
                    turning_on = bool(event.current)
                    field = "e_on_j" if turning_on else "e_off_j"
                    label = "turn_on" if turning_on else "turn_off"
                else:
                    if not bool(event.previous):
                        continue
                    field, label = "e_rec_j", "reverse_recovery"
                lookup = lookups[field]
                if lookup is None:
                    continue
                # Grid samples exactly AT a boundary are recorded post-edge.
                # ``base`` is the first grid sample at/after the event.
                # Turn-on energy uses that post-edge sample (current taken
                # over); turn-off/recovery uses the last strictly pre-event
                # sample (the interrupted current).
                base = int(np.searchsorted(time, event.time - on_grid))
                post_side = field == "e_on_j"
                sample = base if post_side else base - 1
                # The sample must see the state THIS edge left or ended: a
                # neighbouring edge of the same device between the event and
                # its sample (a pulse shorter than the output grid) would make
                # it read the other state and book a silent ~0 J. Such an edge
                # is counted as unresolved instead, never as a number.
                if post_side:
                    following = edges[position + 1].time if position + 1 < len(edges) else None
                    resolved = sample < len(time) and (
                        following is None or time[sample] < following - on_grid
                    )
                else:
                    preceding = edges[position - 1].time if position > 0 else None
                    resolved = sample >= 0 and (
                        preceding is None or time[sample] >= preceding - on_grid
                    )
                if not resolved:
                    unresolved[label] += 1
                    continue
                post = min(base, len(time) - 1)
                pre = max(base - 1, 0)
                event_current = float(current[sample])
                event_voltage = abs(float(commutation[post]))
                energy = lookup(event_current) * event_voltage / table.test_voltage_v
                totals[field] += energy
                counted[label] += 1
                # Assign the event energy to its enclosing output-grid interval
                # for the thermal power series (declared smearing convention).
                interval = min(pre, len(time) - 2)
                switching_power[interval] += energy / step
            if any(unresolved.values()):
                warnings.append(
                    f"{binding.component_id}: {sum(unresolved.values())} switching "
                    "edge(s) fall in a pulse shorter than the output grid, so no "
                    "recorded sample shows the device current at the edge; their "
                    "energy is NOT booked (switching totals are a lower bound). "
                    "Reduce analysis.transient.output_step to resolve them."
                )
            switching_payload = {
                "test_voltage_v": table.test_voltage_v,
                "test_temperature_c": table.test_temperature_c,
                "voltage_scaling": table.voltage_scaling,
                "totals_j": {
                    key: value for key, value in totals.items() if counted_any
                }
                if (counted_any := any(counted.values()))
                else {key: 0.0 for key in totals},
                "events_counted": counted,
                "events_unresolved_by_output_grid": unresolved,
                "extrapolated_lookups": {
                    key: lookup.above
                    for key, lookup in lookups.items()
                    if lookup is not None
                },
                "provenance": table.provenance.as_dict(),
            }
            if spec.tj_mode == "self_consistent":
                warnings.append(
                    f"{binding.component_id}: switching energies measured at "
                    f"{table.test_temperature_c:g} C are used unchanged; the "
                    "self-consistent junction temperature is applied to conduction "
                    "only (dependency not provided by the card)"
                )
            elif table.test_temperature_c != spec.tj_evaluation_c:
                warnings.append(
                    f"{binding.component_id}: switching energies measured at "
                    f"{table.test_temperature_c:g} C are applied at the declared "
                    f"tj_evaluation_c={spec.tj_evaluation_c:g} C without a "
                    "temperature correction (dependency not provided by the card)"
                )

        # --- cooling path ---------------------------------------------------
        path_rth = 0.0
        cooling_payload: dict[str, Any] = {"omissions": [
            "TIM and heatsink modelled as static resistances (no dynamics)",
            "no shared-heatsink coupling between devices",
            "no spreading, orientation, or altitude corrections",
        ]}
        if binding.cooling.tim is not None:
            tim = spec.tims[binding.cooling.tim]
            path_rth += tim.rth_k_per_w
            cooling_payload["tim"] = {
                "id": tim.id,
                "rth_k_per_w": tim.rth_k_per_w,
                "provenance": tim.provenance.as_dict(),
            }
        if binding.cooling.heatsink is not None:
            heatsink = spec.heatsinks[binding.cooling.heatsink]
            if binding.cooling.fan is not None:
                fan = spec.fans[binding.cooling.fan]
                airflow, rth_sa = _fan_heatsink_operating_point(fan, heatsink)
                cooling_payload["fan_operating_point"] = {
                    "fan": fan.id,
                    "airflow_m3_per_h": airflow,
                    "rth_k_per_w": rth_sa,
                }
            else:
                rth_sa = heatsink.rth_k_per_w[0]
                cooling_payload["natural_convection_assumption"] = (
                    "zero-airflow table point used"
                )
            path_rth += rth_sa
            cooling_payload["heatsink"] = {
                "id": heatsink.id,
                "rth_k_per_w": rth_sa,
                "provenance": heatsink.provenance.as_dict(),
            }
        reference_c = spec.ambient_c
        if binding.cooling.cold_plate is not None:
            plate = spec.cold_plates[binding.cooling.cold_plate]
            flow = float(binding.cooling.coolant_flow_l_per_min)
            plate_curve = _Clamped1D(plate.flow_axis_l_per_min, plate.rth_k_per_w)
            rth_plate = plate_curve(flow)
            path_rth += rth_plate
            # A liquid-cooled device sits at the coolant inlet, not at cabinet
            # ambient: using ambient here would silently overstate the margin.
            reference_c = plate.coolant_inlet_c
            cooling_payload["cold_plate"] = {
                "id": plate.id,
                "coolant_flow_l_per_min": flow,
                "rth_k_per_w": rth_plate,
                "coolant_inlet_c": plate.coolant_inlet_c,
                "flow_below_axis": plate_curve.below,
                "flow_above_axis": plate_curve.above,
                "provenance": plate.provenance.as_dict(),
            }
            cooling_payload["omissions"] = [
                *cooling_payload["omissions"],
                "pump and plumbing curve not modelled; the operating coolant "
                "flow is declared, not solved",
                "coolant temperature rise along the loop not modelled",
            ]
            cooling_payload["reference_temperature_c"] = reference_c
            cooling_payload["reference_temperature_source"] = "coolant_inlet"
        else:
            cooling_payload["reference_temperature_c"] = reference_c
            cooling_payload["reference_temperature_source"] = "ambient"

        # --- conduction + thermal chain --------------------------------------
        chain = partial(
            _thermal_chain,
            time=time,
            step=step,
            stages=card.foster_stages,
            reference_c=reference_c,
            path_rth=path_rth,
        )

        electrothermal: dict[str, Any] | None = None
        if spec.tj_mode == "self_consistent":
            loop = _self_consistent_junction(
                card, current, switching_power, chain, spec.tj_evaluation_c
            )
            electrothermal = loop["record"]
            if electrothermal["status"] != "converged":
                # No loss or temperature number survives a loop without a fixed
                # point: reporting the last iterate would read as a result.
                devices_payload.append(
                    {
                        "component_id": binding.component_id,
                        "card": card.id,
                        "card_source": card.source,
                        "kind": card.kind,
                        "electrothermal": electrothermal,
                    }
                )
                warnings.append(
                    f"{binding.component_id}: {electrothermal['status']} — "
                    f"electrothermal loop gain {electrothermal['loop_gain']:.3g}; "
                    "no loss or junction temperature is reported for this device"
                )
                continue
            power = loop["conduction_power"]
            conduction_extrapolated = loop["current_extrapolated_lookups"]
            thermal_state = loop["chain"]
        else:
            v_on = _conduction_voltage_lookup(card, spec.tj_evaluation_c)
            power = np.array([v_on(value) * value for value in current])
            conduction_extrapolated = v_on.above
            thermal_state = chain(power + switching_power)
        conduction_energy = float(np.trapezoid(power, time))
        device_power = power + switching_power
        total_power += device_power
        mean_power = thermal_state["mean_power"]
        case_c = thermal_state["case_c"]
        junction = thermal_state["junction"]
        stored_j = thermal_state["stored_j"]
        balance_error = thermal_state["balance_error"]
        balance_tolerance = thermal_state["balance_tolerance"]

        tj_max = float(np.max(junction))
        tj_min = float(np.min(junction))
        validity_violated = tj_max > card.validity_tj_max_c
        if validity_violated:
            warnings.append(
                f"{binding.component_id}: computed Tj max {tj_max:.1f} C exceeds "
                f"the card validity limit {card.validity_tj_max_c:g} C"
            )
        current_peak = float(np.max(current))
        if current_peak > card.validity_current_max_a:
            warnings.append(
                f"{binding.component_id}: peak current {current_peak:.2f} A "
                f"exceeds the card validity limit {card.validity_current_max_a:g} A"
            )
        tj_mismatch = (
            abs(spec.tj_evaluation_c - tj_max)
            if electrothermal is None
            else electrothermal["final_residual_c"]
        )

        cycles = rainflow_count(junction)
        cycles.sort(key=lambda item: (-item["range"], item["mean"]))

        device_record = {
                "component_id": binding.component_id,
                "card": card.id,
                "card_source": card.source,
                "kind": card.kind,
                "conduction": {
                    "energy_j": conduction_energy,
                    "mean_power_w": conduction_energy / (time[-1] - time[0]),
                    "tj_evaluation_c": spec.tj_evaluation_c
                    if electrothermal is None
                    else "self_consistent_trajectory",
                    "extrapolated_lookups": conduction_extrapolated,
                    "method": "trapezoid_of_v_on(|i|)*|i|_on_recorded_output_grid",
                    "provenance": card.conduction.provenance.as_dict(),
                },
                "switching": _with_mean_power(switching_payload, time),
                "mean_power_w": mean_power,
                "cooling": cooling_payload,
                "thermal": {
                    "network": "foster_junction_to_case_plus_static_path",
                    "initial_state": "steady_state_at_run_mean_power",
                    "stages": [
                        {
                            "resistance_k_per_w": stage.resistance_k_per_w,
                            "time_constant_s": stage.time_constant_s,
                        }
                        for stage in card.foster_stages
                    ],
                    "case_c_static": case_c,
                    "tj_max_c": tj_max,
                    "tj_min_c": tj_min,
                    "tj_final_c": float(junction[-1]),
                    "stored_energy_final_j": stored_j,
                    "balance_error_j": balance_error,
                    "balance_within_tolerance": balance_error <= balance_tolerance,
                    "validity_tj_max_c": card.validity_tj_max_c,
                    "validity_violated": validity_violated,
                    "tj_evaluation_gap_c": tj_mismatch,
                    "provenance": card.thermal_provenance.as_dict(),
                },
                "tj_envelope": _envelope(time, junction),
                "rainflow_delta_tj": {
                    "method": "astm_e1049_four_point",
                    "cycle_count": len(cycles),
                    "cycles": cycles[:64],
                    "truncated": len(cycles) > 64,
                },
            }
        if electrothermal is not None:
            device_record["electrothermal"] = electrothermal
        devices_payload.append(device_record)

    if spec.tj_mode == "self_consistent":
        return _self_consistent_artifact(spec, time, total_power, devices_payload, warnings)
    return {
        "schema": LOSSES_THERMAL_SCHEMA,
        "kind": "losses-thermal-estimate",
        "claim": (
            "post-processing estimate over IDEAL recorded waveforms at a "
            "declared evaluation temperature; no electrothermal iteration; "
            "not a measurement"
        ),
        "ambient_c": spec.ambient_c,
        "tj_evaluation_c": spec.tj_evaluation_c,
        "total_mean_power_w": float(
            np.trapezoid(total_power, time) / (time[-1] - time[0])
        ),
        "devices": devices_payload,
        "warnings": warnings,
    }


# Self-consistent junction temperature (loss_analysis.tj_mode: self_consistent)
# -----------------------------------------------------------------------------
# The loop closed here is losses <-> Tj on ONE recorded run. It is not
# Tj <-> waveforms: the kernel stays ideal, so dissipation never reaches the
# circuit, and iterating the circuit would change nothing.
SELF_CONSISTENT_TOLERANCE_C = 0.01
SELF_CONSISTENT_MAX_EVALUATIONS = 50


def _thermal_chain(
    device_power: np.ndarray,
    *,
    time: np.ndarray,
    step: float,
    stages: tuple,
    reference_c: float,
    path_rth: float,
    audit: bool = True,
) -> dict[str, Any]:
    """Junction trajectory for one device power series (static path + Foster)."""

    mean_power = float(np.trapezoid(device_power, time) / (time[-1] - time[0]))
    # The case is held at the steady rise for the run-mean power, so the
    # junction-to-case stages start at that same steady state. Starting them
    # cold under a hot case would read a slow stage (tau of 100 ms against a
    # 40 ms run) as headroom that is not there.
    case_c = reference_c + path_rth * mean_power
    junction, stored_j = _foster_response(
        device_power, step, stages, case_c, initial_power_w=mean_power
    )
    state: dict[str, Any] = {
        "mean_power": mean_power,
        "case_c": case_c,
        "junction": junction,
        "stored_j": stored_j,
    }
    if not audit:
        return state
    input_energy = float(np.trapezoid(device_power, time))
    # Foster balance: input = dissipated-through-chain + stored; the
    # dissipated part equals input - stored by construction, so audit the
    # STORED term against an independent recomputation from temperatures.
    independent_stored = float(
        sum(
            (stage.time_constant_s / stage.resistance_k_per_w) * rise
            for stage, rise in zip(
                stages,
                _final_stage_rises(device_power, step, stages, initial_power_w=mean_power),
            )
        )
    )
    state["balance_error"] = abs(stored_j - independent_stored)
    state["balance_tolerance"] = max(1e-12, 1e-9 * max(abs(input_energy), 1.0))
    return state


def _conduction_voltage_trajectory(
    card: DeviceCard, current: np.ndarray, junction_c: np.ndarray
) -> tuple[np.ndarray, int, int]:
    """v_on(|i|[n], Tj[n]) per sample, bilinear in the (i, Tj) table.

    The same interpolation as the declared path (temperature first, then
    current), evaluated per sample instead of at one temperature. Returns the
    voltages and how many samples were clamped above the current axis and
    outside the temperature axis.
    """

    table = card.conduction
    grid = np.asarray(table.voltage_table_v, dtype=float)
    temperatures = np.asarray(table.temperature_axis_c, dtype=float)
    count = len(current)
    if len(temperatures) == 1:
        rows = np.broadcast_to(grid[0], (count, grid.shape[1]))
        temperature_clamped = 0
    else:
        clamped_t = np.clip(junction_c, temperatures[0], temperatures[-1])
        temperature_clamped = int(np.count_nonzero(clamped_t != junction_c))
        upper = np.clip(
            np.searchsorted(temperatures, clamped_t, side="right"), 1, len(temperatures) - 1
        )
        fraction = (clamped_t - temperatures[upper - 1]) / (
            temperatures[upper] - temperatures[upper - 1]
        )
        rows = (
            grid[upper - 1] * (1.0 - fraction)[:, np.newaxis]
            + grid[upper] * fraction[:, np.newaxis]
        )
    axis = np.asarray(table.current_axis_a, dtype=float)
    current_above = int(np.count_nonzero(current > axis[-1]))
    if len(axis) == 1:
        return np.asarray(rows[:, 0], dtype=float), current_above, temperature_clamped
    clamped_i = np.clip(current, axis[0], axis[-1])
    upper_i = np.clip(np.searchsorted(axis, clamped_i, side="right"), 1, len(axis) - 1)
    fraction_i = (clamped_i - axis[upper_i - 1]) / (axis[upper_i] - axis[upper_i - 1])
    samples = np.arange(count)
    voltage = (
        rows[samples, upper_i - 1] * (1.0 - fraction_i) + rows[samples, upper_i] * fraction_i
    )
    return voltage, current_above, temperature_clamped


def _self_consistent_junction(
    card: DeviceCard,
    current: np.ndarray,
    switching_power: np.ndarray,
    chain: Any,
    start_c: float,
) -> dict[str, Any]:
    """Fixed point Tj = F(Tj) of conduction losses and the thermal chain.

    F maps a junction trajectory to the trajectory its conduction losses
    produce. Each pass evaluates F twice; the ratio of the two increments is
    the electrothermal loop gain g (dTj_out/dTj_in). g >= 1 is thermal
    runaway: the losses grow faster with temperature than the cooling path
    removes them, so there is no stable operating point and no number is
    returned. Otherwise the pass ends with an Aitken step, T1 + d2/(1 - g),
    which is exact when F is affine; convergence is declared when the
    estimated distance of the evaluated trajectory from the fixed point,
    max|d2|/(1 - g), is below SELF_CONSISTENT_TOLERANCE_C.
    """

    def evaluate(junction_c: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
        v_on, above, clamped_t = _conduction_voltage_trajectory(card, current, junction_c)
        conduction = v_on * current
        result = chain(conduction + switching_power, audit=False)
        return result["junction"], conduction, above, clamped_t

    trajectory = np.full(len(current), float(start_c))
    evaluations = 0
    # The gain is measured on increments well above round-off; a pass that
    # starts ON the fixed point (as after an exact Aitken step) keeps the
    # gain of the pass that found it instead of reading 0/0 noise.
    gain = 0.0
    history: list[float] = []
    while evaluations + 2 <= SELF_CONSISTENT_MAX_EVALUATIONS:
        first, _, _, _ = evaluate(trajectory)
        second, conduction, above, clamped_t = evaluate(first)
        evaluations += 2
        d1 = first - trajectory
        d2 = second - first
        noise = 1e-9 * max(1.0, float(np.max(np.abs(trajectory))))
        if float(np.max(np.abs(d1))) > noise:
            gain = float(np.dot(d1, d2)) / float(np.dot(d1, d1))
        if gain >= 1.0:
            return {
                "record": {
                    "mode": "self_consistent",
                    "status": "thermal_runaway",
                    "loop_gain": gain,
                    "evaluations": evaluations,
                    "last_tj_max_c": float(np.max(second)),
                    "tolerance_c": SELF_CONSISTENT_TOLERANCE_C,
                }
            }
        distance = float(np.max(np.abs(d2))) / (1.0 - gain)
        history.append(distance)
        if distance < SELF_CONSISTENT_TOLERANCE_C and clamped_t:
            # A fixed point found with v_on held at the table edge is a
            # property of the clamp, not of the device: a runaway cut off by a
            # short temperature axis lands here and must not read as a result.
            return {
                "record": {
                    "mode": "self_consistent",
                    "status": "beyond_card_temperature_axis",
                    "loop_gain": gain,
                    "evaluations": evaluations,
                    "last_tj_max_c": float(np.max(second)),
                    "temperature_clamped_lookups": clamped_t,
                    "temperature_axis_c": [
                        card.conduction.temperature_axis_c[0],
                        card.conduction.temperature_axis_c[-1],
                    ],
                    "tolerance_c": SELF_CONSISTENT_TOLERANCE_C,
                }
            }
        if distance < SELF_CONSISTENT_TOLERANCE_C:
            return {
                "record": {
                    "mode": "self_consistent",
                    "status": "converged",
                    "loop_gain": gain,
                    "evaluations": evaluations,
                    "final_residual_c": float(np.max(np.abs(d2))),
                    "estimated_error_c": distance,
                    "tolerance_c": SELF_CONSISTENT_TOLERANCE_C,
                    "loop": "conduction_losses_and_junction_temperature_only",
                },
                "conduction_power": conduction,
                "current_extrapolated_lookups": above,
                # Losses were evaluated on ``first``; ``chain`` of them is the
                # junction trajectory that goes with them (= ``second``).
                "chain": chain(conduction + switching_power),
            }
        trajectory = first + d2 / (1.0 - gain)
    return {
        "record": {
            "mode": "self_consistent",
            "status": "not_converged",
            "loop_gain": gain,
            "evaluations": evaluations,
            "last_tj_max_c": float(np.max(trajectory)),
            "estimated_error_history_c": history,
            "tolerance_c": SELF_CONSISTENT_TOLERANCE_C,
        }
    }


def _self_consistent_artifact(
    spec: LossAnalysisSpec,
    time: np.ndarray,
    total_power: np.ndarray,
    devices: list[dict[str, Any]],
    warnings: list[str],
) -> dict[str, Any]:
    statuses = [device["electrothermal"]["status"] for device in devices]
    status = next((item for item in statuses if item != "converged"), "converged")
    artifact: dict[str, Any] = {
        "schema": LOSSES_THERMAL_SCHEMA,
        "kind": "losses-thermal-estimate",
        "claim": (
            "post-processing estimate over IDEAL recorded waveforms; conduction "
            "losses iterated to a self-consistent junction temperature on this "
            "one run (losses do not feed back into the waveforms; switching "
            "energies stay at their card test temperature); not a measurement"
        ),
        "ambient_c": spec.ambient_c,
        "tj_evaluation_c": spec.tj_evaluation_c,
        "electrothermal": {
            "mode": "self_consistent",
            "status": status,
            "iteration_start_c": spec.tj_evaluation_c,
            "tolerance_c": SELF_CONSISTENT_TOLERANCE_C,
            "max_evaluations": SELF_CONSISTENT_MAX_EVALUATIONS,
        },
        "devices": devices,
        "warnings": warnings,
    }
    if status == "converged":
        artifact["total_mean_power_w"] = float(
            np.trapezoid(total_power, time) / (time[-1] - time[0])
        )
    return artifact


TJ_ENVELOPE_BUCKETS = 200


def _with_mean_power(switching: Any, time: np.ndarray) -> Any:
    """Add the mean switching power over the run, so consumers never divide."""

    if not isinstance(switching, dict) or "totals_j" not in switching:
        return switching
    return {
        **switching,
        "mean_power_w": float(sum(switching["totals_j"].values()) / (time[-1] - time[0])),
    }


def _envelope(time: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    """Junction temperature reduced to per-bucket minimum and maximum.

    The full trace has one point per output sample and dozens of devices; the
    envelope keeps every excursion (a bucket's extremes) at a fixed size, so a
    report can draw it without recomputing anything.
    """

    edges = np.linspace(0, len(time), min(TJ_ENVELOPE_BUCKETS, len(time)) + 1).astype(int)
    pairs = [(start, stop) for start, stop in zip(edges[:-1], edges[1:]) if stop > start]
    return {
        "method": f"per-bucket min/max over {len(pairs)} equal sample buckets",
        "time_s": [float(time[start]) for start, _ in pairs],
        "time_end_s": [float(time[stop - 1]) for _, stop in pairs],
        "min_c": [float(np.min(values[start:stop])) for start, stop in pairs],
        "max_c": [float(np.max(values[start:stop])) for start, stop in pairs],
    }


def _final_stage_rises(
    power_w: np.ndarray, step_s: float, stages: tuple, initial_power_w: float = 0.0
) -> list[float]:
    """Independent per-stage recomputation used by the balance audit."""

    rises = []
    for stage in stages:
        decay = math.exp(-step_s / stage.time_constant_s)
        gain = stage.resistance_k_per_w * (1.0 - decay)
        state = stage.resistance_k_per_w * initial_power_w
        for value in power_w[:-1]:
            state = state * decay + gain * value
        rises.append(state)
    return rises
