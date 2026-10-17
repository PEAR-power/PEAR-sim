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

        # --- conduction ----------------------------------------------------
        v_on = _conduction_voltage_lookup(card, spec.tj_evaluation_c)
        power = np.array([v_on(value) * value for value in current])
        conduction_energy = float(np.trapezoid(power, time))
        conduction_extrapolated = v_on.above

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
            control_id = component.control
            for event in result.events:
                if component.type == "ideal_switch":
                    if event.kind != "scheduled_gate" or event.target != control_id:
                        continue
                    turning_on = bool(event.current) and not bool(event.previous)
                    field = "e_on_j" if turning_on else "e_off_j"
                    label = "turn_on" if turning_on else "turn_off"
                else:
                    # A freewheel diode is hard-commutated by the opposite
                    # switch (switch_manager) or soft-commutated at zero
                    # current (natural_commutation, where E_rec(~0) ~ 0).
                    if (
                        event.kind not in {"natural_commutation", "switch_manager"}
                        or event.target != component.id
                        or not (bool(event.previous) and not bool(event.current))
                    ):
                        continue
                    field, label = "e_rec_j", "reverse_recovery"
                lookup = lookups[field]
                if lookup is None:
                    continue
                # Grid samples exactly AT a boundary are recorded post-edge.
                # Anchor the index a quarter-step early so float noise cannot
                # flip it: ``base`` is the first grid sample at/after the
                # event. Turn-on energy uses that post-edge sample (current
                # taken over); turn-off/recovery uses the last strictly
                # pre-event sample (the interrupted current).
                base = int(np.searchsorted(time, event.time - 0.25 * step))
                post = min(base, len(time) - 1)
                pre = max(base - 1, 0)
                sample = post if field == "e_on_j" else pre
                event_current = float(current[sample])
                event_voltage = abs(float(commutation[post]))
                energy = lookup(event_current) * event_voltage / table.test_voltage_v
                totals[field] += energy
                counted[label] += 1
                # Assign the event energy to its enclosing output-grid interval
                # for the thermal power series (declared smearing convention).
                interval = min(pre, len(time) - 2)
                switching_power[interval] += energy / step
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
                "extrapolated_lookups": {
                    key: lookup.above
                    for key, lookup in lookups.items()
                    if lookup is not None
                },
                "provenance": table.provenance.as_dict(),
            }
            if table.test_temperature_c != spec.tj_evaluation_c:
                warnings.append(
                    f"{binding.component_id}: switching energies measured at "
                    f"{table.test_temperature_c:g} C are applied at the declared "
                    f"tj_evaluation_c={spec.tj_evaluation_c:g} C without a "
                    "temperature correction (dependency not provided by the card)"
                )

        device_power = power + switching_power
        total_power += device_power

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

        # --- thermal chain ---------------------------------------------------
        mean_power = float(np.trapezoid(device_power, time) / (time[-1] - time[0]))
        # The case is held at the steady rise for the run-mean power, so the
        # junction-to-case stages start at that same steady state. Starting them
        # cold under a hot case would read a slow stage (tau of 100 ms against a
        # 40 ms run) as headroom that is not there.
        case_c = reference_c + path_rth * mean_power
        junction, stored_j = _foster_response(
            device_power, step, card.foster_stages, case_c, initial_power_w=mean_power
        )
        input_energy = float(np.trapezoid(device_power, time))
        # Foster balance: input = dissipated-through-chain + stored; the
        # dissipated part equals input - stored by construction, so audit the
        # STORED term against an independent recomputation from temperatures.
        independent_stored = float(
            sum(
                (stage.time_constant_s / stage.resistance_k_per_w) * rise
                for stage, rise in zip(
                    card.foster_stages,
                    _final_stage_rises(
                        device_power, step, card.foster_stages, initial_power_w=mean_power
                    ),
                )
            )
        )
        balance_error = abs(stored_j - independent_stored)
        balance_tolerance = max(1e-12, 1e-9 * max(abs(input_energy), 1.0))

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
        tj_mismatch = abs(spec.tj_evaluation_c - tj_max)

        cycles = rainflow_count(junction)
        cycles.sort(key=lambda item: (-item["range"], item["mean"]))

        devices_payload.append(
            {
                "component_id": binding.component_id,
                "card": card.id,
                "card_source": card.source,
                "kind": card.kind,
                "conduction": {
                    "energy_j": conduction_energy,
                    "mean_power_w": conduction_energy / (time[-1] - time[0]),
                    "tj_evaluation_c": spec.tj_evaluation_c,
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
        )

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
