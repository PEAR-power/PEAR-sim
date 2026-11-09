# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import math
from typing import Any

from pear_sim.spec.model import Component, Model, PwmControl


ANALYTIC_CHECKS_SCHEMA = "pear-sim/analytic-self-checks/v0.1"
SELF_CHECK_KIND = "verification-self-check"


@dataclass(frozen=True)
class BuckCcmTolerances:
    """Explicit engineering tolerances for the alpha ideal-buck self-check."""

    output_voltage_relative: float = 0.001
    output_voltage_absolute_v: float = 1e-6
    inductor_ripple_relative: float = 0.01
    inductor_ripple_absolute_a: float = 1e-9
    output_ripple_relative: float = 0.02
    output_ripple_absolute_v: float = 1e-9
    diode_average_relative: float = 0.02
    diode_average_absolute_a: float = 1e-9
    ccm_floor_relative_to_expected_output_current: float = 1e-6
    ccm_floor_absolute_a: float = 1e-9
    minimum_measurement_cycles: float = 5.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class _BuckTopology:
    source: Component
    switch: Component
    diode: Component
    inductor: Component
    capacitor: Component
    load: Component
    pwm: PwmControl
    input_node: str
    switch_node: str
    output_node: str

    @property
    def component_ids(self) -> dict[str, str]:
        return {
            "source": self.source.id,
            "switch": self.switch.id,
            "diode": self.diode.id,
            "inductor": self.inductor.id,
            "capacitor": self.capacitor.id,
            "load": self.load.id,
            "pwm": self.pwm.id,
        }


@dataclass(frozen=True)
class _Measurement:
    value: float
    signal: str
    metric: str
    unit: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "unit": self.unit,
            "signal": self.signal,
            "metric": self.metric,
        }


def run_analytic_checks(
    model: Model,
    metrics: Mapping[str, Any],
    tolerances: BuckCcmTolerances | None = None,
    *,
    time: Any = None,
    waveforms: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run conservative analytic self-checks for an exact ideal buck CCM topology.

    This function consumes already-calculated run metrics. It does not simulate, tune,
    infer missing signals, or make a physical-validation claim.
    """

    selected_tolerances = tolerances or BuckCcmTolerances()
    tolerance_record = asdict(selected_tolerances)
    base: dict[str, Any] = {
        "schema": ANALYTIC_CHECKS_SCHEMA,
        "kind": SELF_CHECK_KIND,
        "claim": {
            "classification": "verification",
            "validation_claim": False,
            "statement": (
                "Analytic self-check of an idealized model; passing does not establish "
                "cross-tool compatibility or agreement with physical hardware."
            ),
        },
        "model": {
            "name": model.name,
            "schema": model.schema,
            "source_sha256": model.source_sha256,
        },
        "tolerances": tolerance_record,
    }
    topology, reason = _match_ideal_buck(model)
    if topology is None:
        status = "not_applicable"
        base.update(
            {
                "recognition": {
                    "topology": "ideal_buck_ccm",
                    "status": status,
                    "reason": reason,
                    "components": None,
                    "parameters_si": None,
                },
                "preconditions": [],
                "checks": _inactive_checks(status, reason),
                "overall_status": status,
            }
        )
        return base

    parameters = _buck_parameters(topology)
    base["recognition"] = {
        "topology": "ideal_buck_ccm",
        "status": "matched",
        "reason": "Exact canonical ideal buck topology matched.",
        "components": topology.component_ids,
        "nodes": {
            "ground": model.ground,
            "input": topology.input_node,
            "switch": topology.switch_node,
            "output": topology.output_node,
        },
        "parameters_si": parameters,
    }

    minimum_cycles = selected_tolerances.minimum_measurement_cycles
    measurement_cycles = (
        (model.transient.stop - model.transient.start) * 0.5 * topology.pwm.frequency
    )
    window_status = "pass" if measurement_cycles >= minimum_cycles else "not_run"
    window_precondition = {
        "id": "buck.measurement_window",
        "description": "The last-half metric window must span enough switching cycles.",
        "measured": {"value": measurement_cycles, "unit": "cycle"},
        "tolerance": {"minimum_inclusive": minimum_cycles, "unit": "cycle"},
        "status": window_status,
        "reason": (
            None
            if window_status == "pass"
            else "The metric window is too short for the configured analytic self-check."
        ),
    }

    expected_output_current = parameters["duty"] * parameters["vin_v"] / parameters["load_ohm"]
    ccm_floor = max(
        selected_tolerances.ccm_floor_absolute_a,
        selected_tolerances.ccm_floor_relative_to_expected_output_current
        * abs(expected_output_current),
    )
    inductor_signal = f"i({topology.inductor.id})"
    minimum_current, current_reason = _find_measurement(
        metrics,
        (inductor_signal,),
        ("min_last_half",),
        "A",
    )
    if minimum_current is None:
        ccm_status = "not_run"
        ccm_reason = current_reason
    elif minimum_current.value <= ccm_floor:
        ccm_status = "not_applicable"
        ccm_reason = (
            "Measured minimum inductor current does not establish strict CCM: "
            f"{minimum_current.value:.12g} A <= {ccm_floor:.12g} A."
        )
    else:
        ccm_status = "pass"
        ccm_reason = None
    ccm_precondition = {
        "id": "buck.ccm_regime",
        "description": "Measured minimum inductor current must be strictly above the CCM floor.",
        "measured": None if minimum_current is None else minimum_current.as_dict(),
        "tolerance": {
            "minimum_exclusive": ccm_floor,
            "unit": "A",
            "absolute_floor_a": selected_tolerances.ccm_floor_absolute_a,
            "relative_floor_to_expected_output_current": (
                selected_tolerances.ccm_floor_relative_to_expected_output_current
            ),
        },
        "status": ccm_status,
        "reason": ccm_reason,
    }
    base["preconditions"] = [window_precondition, ccm_precondition]

    if time is not None and waveforms is not None and window_status == "pass":
        steady = _periodic_steady_state_precondition(
            topology, parameters, time, waveforms, selected_tolerances
        )
        base["preconditions"].append(steady)
        if steady["status"] != "pass":
            base["checks"] = _inactive_checks("not_run", str(steady["reason"]))
            base["overall_status"] = "not_run"
            return base

    if window_status != "pass":
        reason = window_precondition["reason"]
        base["checks"] = _inactive_checks("not_run", str(reason))
        base["overall_status"] = "not_run"
        return base
    if ccm_status != "pass":
        base["checks"] = _inactive_checks(ccm_status, str(ccm_reason))
        base["overall_status"] = ccm_status
        return base

    vin = parameters["vin_v"]
    duty = parameters["duty"]
    frequency = parameters["switching_frequency_hz"]
    inductance = parameters["inductance_h"]
    capacitance = parameters["capacitance_f"]
    resistance = parameters["load_ohm"]
    expected_vout = duty * vin
    expected_inductor_ripple = expected_vout * (1.0 - duty) / (inductance * frequency)
    expected_output_ripple = expected_inductor_ripple / (8.0 * capacitance * frequency)
    expected_load_current = expected_vout / resistance
    expected_diode_average = (1.0 - duty) * expected_load_current

    voltage_signals = (
        f"v({topology.output_node})",
        f"v({topology.capacitor.id})",
        f"v({topology.load.id})",
    )
    output_voltage, output_voltage_reason = _find_measurement(
        metrics, voltage_signals, ("mean_last_half",), "V"
    )
    inductor_ripple, inductor_ripple_reason = _find_measurement(
        metrics, (inductor_signal,), ("ripple_last_half",), "A"
    )
    output_ripple, output_ripple_reason = _find_measurement(
        metrics, voltage_signals, ("ripple_last_half",), "V"
    )
    diode_average, diode_average_reason = _find_measurement(
        metrics,
        (f"i({topology.diode.id})",),
        ("mean_last_half",),
        "A",
    )

    checks = [
        _comparison(
            check_id="buck.output_voltage",
            relation="Vout = D * Vin",
            expected=expected_vout,
            measured=output_voltage,
            unit="V",
            relative_tolerance=selected_tolerances.output_voltage_relative,
            absolute_tolerance=selected_tolerances.output_voltage_absolute_v,
            missing_reason=output_voltage_reason,
        ),
        _comparison(
            check_id="buck.inductor_current_ripple",
            relation="delta_IL = Vout * (1 - D) / (L * fsw)",
            expected=expected_inductor_ripple,
            measured=inductor_ripple,
            unit="A",
            relative_tolerance=selected_tolerances.inductor_ripple_relative,
            absolute_tolerance=selected_tolerances.inductor_ripple_absolute_a,
            missing_reason=inductor_ripple_reason,
        ),
        _comparison(
            check_id="buck.output_voltage_ripple",
            relation="delta_Vout = delta_IL / (8 * C * fsw)",
            expected=expected_output_ripple,
            measured=output_ripple,
            unit="V",
            relative_tolerance=selected_tolerances.output_ripple_relative,
            absolute_tolerance=selected_tolerances.output_ripple_absolute_v,
            missing_reason=output_ripple_reason,
        ),
        _comparison(
            check_id="buck.diode_average_current",
            relation="I_diode_avg = (1 - D) * Io; Io = Vout / R",
            expected=expected_diode_average,
            measured=diode_average,
            unit="A",
            relative_tolerance=selected_tolerances.diode_average_relative,
            absolute_tolerance=selected_tolerances.diode_average_absolute_a,
            missing_reason=diode_average_reason,
        ),
    ]
    base["checks"] = checks
    statuses = {check["status"] for check in checks}
    if "fail" in statuses:
        overall = "fail"
    elif "not_run" in statuses:
        overall = "not_run"
    else:
        overall = "pass"
    base["overall_status"] = overall
    return base


def _periodic_steady_state_precondition(
    topology: _BuckTopology,
    parameters: dict[str, float],
    time: Any,
    waveforms: Mapping[str, Any],
    tolerances: BuckCcmTolerances,
) -> dict[str, Any]:
    """Is the whole last-half window on the periodic orbit?

    The ripple relations hold on the periodic orbit. A run that still carries
    its start-up transient anywhere in the metric window measures that
    transient, and comparing it with the switching-ripple formula fails a
    correct circuit — even when the run has settled by its final period. The
    waveform must repeat between the window's first and last periods to within
    half of the matching ripple tolerance.
    """

    import numpy as np

    period = 1.0 / parameters["switching_frequency_hz"]
    vout = parameters["duty"] * parameters["vin_v"]
    expected_il_ripple = vout * (1.0 - parameters["duty"]) / (
        parameters["inductance_h"] * parameters["switching_frequency_hz"]
    )
    expected_v_ripple = expected_il_ripple / (
        8.0 * parameters["capacitance_f"] * parameters["switching_frequency_hz"]
    )
    from pear_sim.analysis.settling import window_drift

    times = np.asarray(time, dtype=float)
    end = float(times[-1])
    window_start = end - 0.5 * (end - float(times[0]))
    drifts: dict[str, Any] = {}
    limits = {
        f"i({topology.inductor.id})": 0.5 * tolerances.inductor_ripple_relative * expected_il_ripple,
        f"v({topology.output_node})": 0.5 * tolerances.output_ripple_relative * expected_v_ripple,
    }
    status = "pass"
    for signal, limit in limits.items():
        values = waveforms.get(signal)
        if values is None or end - period < float(times[0]):
            return {
                "id": "buck.periodic_steady_state",
                "description": "The state must repeat over the last switching period.",
                "measured": None,
                "tolerance": None,
                "status": "not_run",
                "reason": f"{signal} is not recorded or the run is shorter than one period",
            }
        drift = window_drift(times, values, period, window_start)
        if drift is None:
            return {
                "id": "buck.periodic_steady_state",
                "description": "The waveform must repeat across the last-half window.",
                "measured": None,
                "tolerance": None,
                "status": "not_run",
                "reason": "the last-half window holds fewer than two switching periods",
            }
        drifts[signal] = {"drift_across_window": drift, "limit": limit}
        if drift > limit:
            status = "not_run"
    return {
        "id": "buck.periodic_steady_state",
        "description": "The waveform must repeat across the last-half window.",
        "measured": drifts,
        "tolerance": {"rule": "drift <= 0.5 * ripple tolerance of the matching check"},
        "status": status,
        "reason": (
            None
            if status == "pass"
            else "The run has not reached periodic steady state: the state still drifts "
            "over a switching period, so the last-half window measures the start-up "
            "transient. Simulate longer or start from the steady-state solution."
        ),
    }


def check_ideal_buck_ccm(
    model: Model,
    metrics: Mapping[str, Any],
    tolerances: BuckCcmTolerances | None = None,
) -> dict[str, Any]:
    """Named alias for the currently supported analytic-check family."""

    return run_analytic_checks(model, metrics, tolerances)


def match_ideal_buck(model: Model) -> tuple[_BuckTopology | None, str]:
    """Recognize the exact canonical buck shared by verification observers.

    Recognition is intentionally structural and conservative.  The returned topology
    carries resolved component/control objects; callers must not infer a match from
    names alone.
    """

    return _match_ideal_buck(model)


def buck_parameters(topology: _BuckTopology) -> dict[str, float]:
    """Return the canonical buck's resolved SI parameters."""

    return _buck_parameters(topology)


def _match_ideal_buck(model: Model) -> tuple[_BuckTopology | None, str]:
    functional = [component for component in model.components if component.type != "voltage_probe"]
    expected_counts = {
        "voltage_source": 1,
        "ideal_switch": 1,
        "diode": 1,
        "inductor": 1,
        "capacitor": 1,
        "resistor": 1,
    }
    counts = Counter(component.type for component in functional)
    if counts != Counter(expected_counts):
        return None, (
            "Canonical buck matching requires exactly one voltage source, ideal switch, "
            "diode, inductor, capacitor, and load resistor, with no other power components; "
            f"found {dict(sorted(counts.items()))}."
        )
    if len(model.controls) != 1:
        return None, f"Canonical buck matching requires exactly one PWM control; found {len(model.controls)}."

    by_type = {component.type: component for component in functional}
    source = by_type["voltage_source"]
    switch = by_type["ideal_switch"]
    diode = by_type["diode"]
    inductor = by_type["inductor"]
    capacitor = by_type["capacitor"]
    load = by_type["resistor"]
    pwm = next(iter(model.controls.values()))
    ground = model.ground

    if source.n != ground or source.value is None or source.value <= 0.0:
        return None, "The sole DC voltage source must be positive and oriented from input to ground."
    if switch.control != pwm.id:
        return None, "The sole ideal switch must be driven by the sole PWM control."
    input_node = source.p
    switch_node = switch.n
    output_node = inductor.n
    if switch.p != input_node:
        return None, "The ideal switch must connect the positive source node to the switch node."
    if diode.p != ground or diode.n != switch_node:
        return None, "The freewheel diode must be oriented from ground to the switch node."
    if inductor.p != switch_node:
        return None, "The inductor must be oriented from the switch node to the output node."
    if capacitor.p != output_node or capacitor.n != ground:
        return None, "The output capacitor must be oriented from output to ground."
    if load.p != output_node or load.n != ground:
        return None, "The sole load resistor must be oriented from output to ground."
    if len({ground, input_node, switch_node, output_node}) != 4:
        return None, "Ground, input, switch, and output must be four distinct nodes."
    canonical_nodes = {ground, input_node, switch_node, output_node}
    for probe in (component for component in model.components if component.type == "voltage_probe"):
        if probe.p not in canonical_nodes or probe.n not in canonical_nodes:
            return None, f"Voltage probe {probe.id!r} introduces a non-canonical node."
    if not 0.0 < pwm.duty < 1.0:
        return None, "PWM duty must be strictly between zero and one for the CCM buck relations."
    return (
        _BuckTopology(
            source=source,
            switch=switch,
            diode=diode,
            inductor=inductor,
            capacitor=capacitor,
            load=load,
            pwm=pwm,
            input_node=input_node,
            switch_node=switch_node,
            output_node=output_node,
        ),
        "",
    )


def _buck_parameters(topology: _BuckTopology) -> dict[str, float]:
    return {
        "vin_v": float(topology.source.value),
        "duty": topology.pwm.duty,
        "switching_frequency_hz": topology.pwm.frequency,
        "inductance_h": float(topology.inductor.value),
        "capacitance_f": float(topology.capacitor.value),
        "load_ohm": float(topology.load.value),
    }


def _find_measurement(
    metrics: Mapping[str, Any],
    signal_candidates: Sequence[str],
    metric_candidates: Sequence[str],
    expected_unit: str,
) -> tuple[_Measurement | None, str | None]:
    normalized: dict[str, list[tuple[str, Any]]] = {}
    for raw_signal, record in metrics.items():
        if isinstance(raw_signal, str):
            normalized.setdefault(raw_signal.strip(), []).append((raw_signal, record))
    for candidate in signal_candidates:
        matches = normalized.get(candidate, [])
        if len(matches) > 1:
            return None, f"Metric signal {candidate!r} is ambiguous after whitespace normalization."
        if not matches:
            continue
        raw_signal, record = matches[0]
        if not isinstance(record, Mapping):
            return None, f"Metric record for {raw_signal!r} is not an object."
        unit = record.get("unit")
        if unit is not None and unit != expected_unit:
            return None, (
                f"Metric record for {raw_signal!r} has unit {unit!r}; expected {expected_unit!r}."
            )
        for metric_name in metric_candidates:
            value = record.get(metric_name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric = float(value)
                if math.isfinite(numeric):
                    return _Measurement(numeric, raw_signal, metric_name, expected_unit), None
                return None, f"Metric {raw_signal}.{metric_name} is non-finite."
        joined = ", ".join(metric_candidates)
        return None, f"Metric record for {raw_signal!r} lacks a finite value in: {joined}."
    joined_signals = ", ".join(signal_candidates)
    return None, f"Required metric signal is absent; accepted expressions: {joined_signals}."


def _comparison(
    *,
    check_id: str,
    relation: str,
    expected: float,
    measured: _Measurement | None,
    unit: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    missing_reason: str | None,
) -> dict[str, Any]:
    effective = max(absolute_tolerance, relative_tolerance * abs(expected))
    tolerance = {
        "absolute": absolute_tolerance,
        "relative": relative_tolerance,
        "effective_absolute": effective,
        "unit": unit,
        "comparison": "abs(measured - expected) <= effective_absolute",
    }
    if measured is None:
        return {
            "id": check_id,
            "relation": relation,
            "expected": {"value": expected, "unit": unit},
            "measured": None,
            "tolerance": tolerance,
            "delta": None,
            "status": "not_run",
            "reason": missing_reason,
        }
    signed_delta = measured.value - expected
    absolute_delta = abs(signed_delta)
    relative_delta = absolute_delta / abs(expected) if expected != 0.0 else None
    status = "pass" if absolute_delta <= effective else "fail"
    return {
        "id": check_id,
        "relation": relation,
        "expected": {"value": expected, "unit": unit},
        "measured": measured.as_dict(),
        "tolerance": tolerance,
        "delta": {
            "signed": signed_delta,
            "absolute": absolute_delta,
            "relative": relative_delta,
            "unit": unit,
        },
        "status": status,
        "reason": None if status == "pass" else "Measured value is outside tolerance.",
    }


def _inactive_checks(status: str, reason: str) -> list[dict[str, Any]]:
    definitions = (
        ("buck.output_voltage", "Vout = D * Vin"),
        ("buck.inductor_current_ripple", "delta_IL = Vout * (1 - D) / (L * fsw)"),
        ("buck.output_voltage_ripple", "delta_Vout = delta_IL / (8 * C * fsw)"),
        ("buck.diode_average_current", "I_diode_avg = (1 - D) * Io"),
    )
    return [
        {
            "id": check_id,
            "relation": relation,
            "expected": None,
            "measured": None,
            "tolerance": None,
            "delta": None,
            "status": status,
            "reason": reason,
        }
        for check_id, relation in definitions
    ]
