"""Deterministic, portable views of models and simulation requests."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from pear_sim.spec.model import Component, Model, PwmControl


_VALUE_DIMENSIONS = {
    "resistor": "resistance",
    "capacitor": "capacitance",
    "inductor": "inductance",
    "sat_inductor": "inductance",
    "voltage_source": "voltage",
    "current_source": "current",
}


def _measurement_conventions() -> dict[str, Any]:
    return {
        "component_voltage": {
            "definition": "v(component)=V(p)-V(n)",
            "positive_terminal": "p",
            "negative_terminal": "n",
        },
        "component_current": {
            "definition": "i(component)>0 means current flows p->n",
            "positive_direction": "p->n",
        },
    }


def _initial_conditions_dict(model: Model) -> dict[str, Any]:
    states: list[dict[str, Any]] = []
    for component in model.components:
        if component.type == "capacitor":
            state_kind = "voltage"
            state_expression = f"v({component.id})"
            source_field = "initial_voltage"
            unit = "V"
        elif component.type in {"inductor", "sat_inductor"}:
            state_kind = "current"
            state_expression = f"i({component.id})"
            source_field = "initial_current"
            unit = "A"
        else:
            continue
        is_zero = component.initial_state == 0.0
        if component.initial_state_explicit:
            origin = "user_declared"
        elif is_zero:
            origin = "default_zero"
        else:
            # This is reachable only for a Component constructed outside the v0.1 loader.
            origin = "unknown"
        states.append(
            {
                "component_id": component.id,
                "component_type": component.type,
                "state_kind": state_kind,
                "state_expression": state_expression,
                "source_field": source_field,
                "value_si": component.initial_state,
                "unit": unit,
                "value_class": "zero" if is_zero else "nonzero",
                "seed_strategy": "explicit_numeric_seed",
                "origin": origin,
                "analytic_derivation": "not_claimed",
            }
        )

    zero_count = sum(state["value_class"] == "zero" for state in states)
    user_declared_count = sum(state["origin"] == "user_declared" for state in states)
    default_zero_count = sum(state["origin"] == "default_zero" for state in states)
    unknown_origin_count = sum(state["origin"] == "unknown" for state in states)
    warm_start_components = [
        state["component_id"] for state in states if state["value_class"] == "nonzero"
    ]
    warm_start = bool(warm_start_components)
    return {
        "strategy": "explicit_seeded_values",
        "origin_policy": (
            "user_declared when the IC field is present; default_zero when omitted; "
            "otherwise unknown"
        ),
        "analytic_initialization_claimed": False,
        "states": states,
        "summary": {
            "state_count": len(states),
            "zero_count": zero_count,
            "nonzero_count": len(states) - zero_count,
            "user_declared_count": user_declared_count,
            "default_zero_count": default_zero_count,
            "unknown_origin_count": unknown_origin_count,
            "warm_start": warm_start,
            "warm_start_components": warm_start_components,
        },
        "warm_start": {
            "enabled": warm_start,
            "definition": "one or more L/C initial-state seeds are nonzero",
            "provenance_claim": (
                "seed values only; no analytic, periodic-steady-state, or previous-run "
                "origin is inferred"
            ),
        },
    }


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON-compatible object for hashing across supported platforms."""

    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _component_dict(component: Component) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": component.id,
        "type": component.type,
        "p": component.p,
        "n": component.n,
    }
    if component.value is not None:
        payload["value_si"] = component.value
        payload["dimension"] = _VALUE_DIMENSIONS[component.type]
    if component.waveform is not None:
        # Additive-only: DC models omit the key, so their hashes are unchanged.
        payload["waveform"] = component.waveform.as_dict()
        payload["dimension"] = _VALUE_DIMENSIONS[component.type]
    if component.control is not None:
        payload["control"] = component.control
    if component.type in {"ideal_switch", "diode"}:
        payload["initial"] = component.initial
    # Additive-only: ideal devices omit both keys, so their hashes are unchanged.
    if component.on_resistance:
        payload["on_resistance_si"] = component.on_resistance
    if component.forward_voltage:
        payload["forward_voltage_si"] = component.forward_voltage
    if component.type == "capacitor":
        payload["initial_voltage_si"] = component.initial_state
    elif component.type in {"inductor", "sat_inductor"}:
        payload["initial_current_si"] = component.initial_state
    if component.type == "sat_inductor":
        payload["l_sat_si"] = component.sat_inductance
        payload["i_sat_si"] = component.sat_current
        payload["saturation_model"] = "piecewise_linear_two_segment_flux"
    if component.model_meta is not None:
        meta = component.model_meta
        validity = {
            field: value
            for field, value in (
                ("frequency_max_hz", meta.frequency_max_hz),
                ("temperature_min_c", meta.temperature_min_c),
                ("temperature_max_c", meta.temperature_max_c),
                ("dvdt_max_v_per_s", meta.dvdt_max_v_per_s),
            )
            if value is not None
        }
        payload["model_meta"] = {
            "version": meta.version,
            "validity": validity,
            "evaluated": False,
        }
    return payload


def physics_model_dict(model: Model) -> dict[str, Any]:
    """Return resolved physical semantics, excluding source layout and run settings."""

    controls: list[dict[str, Any]] = []
    complementary: dict[str, PwmControl] = {}
    three_phase_spwm: dict[str, dict[str, PwmControl]] = {}
    for control in sorted(model.controls.values(), key=lambda item: item.id):
        if control.kind == "pwm":
            controls.append(
                {
                    "id": control.id,
                    "type": "pwm",
                    "frequency_hz": control.frequency,
                    "duty": control.duty,
                    "phase_s": control.phase,
                }
            )
        elif (
            control.kind == "complementary_pwm"
            and control.role == "high"
            and control.block_id is not None
        ):
            complementary[control.block_id] = control
        elif (
            control.kind == "three_phase_spwm"
            and control.role == "high"
            and control.block_id is not None
            and control.phase_label is not None
        ):
            three_phase_spwm.setdefault(control.block_id, {})[
                control.phase_label
            ] = control
    for block_id, control in sorted(complementary.items()):
        controls.append(
            {
                "id": block_id,
                "type": "complementary_pwm",
                "frequency_hz": control.frequency,
                "duty": control.command_duty,
                "phase_s": control.command_phase,
                "dead_time_s": control.dead_time,
                "outputs": {
                    "high": control.id,
                    "low": control.partner_id,
                },
                "edge_policy": "turn_off_immediate_turn_on_after_dead_time",
                "interlock": "intrinsic_non_disableable",
            }
        )
    for block_id, phases in sorted(three_phase_spwm.items()):
        representative = phases["a"]
        controls.append(
            {
                "id": block_id,
                "type": "three_phase_spwm",
                "carrier_frequency_hz": representative.frequency,
                "fundamental_frequency_hz": representative.fundamental_frequency,
                "carrier_ratio": representative.carrier_ratio,
                "modulation_index": representative.modulation_index,
                "carrier_phase_s": representative.phase,
                "reference_phase_s": representative.reference_phase,
                "sequence": representative.sequence,
                "dead_time_s": representative.dead_time,
                "sampling": "offline_midpoint_sampled_center_aligned",
                **(
                    {"reference_injection": representative.injection}
                    if getattr(representative, "injection", None)
                    not in (None, "none")
                    else {}
                ),
                "outputs": {
                    phase_label: {
                        "high": phases[phase_label].id,
                        "low": phases[phase_label].partner_id,
                    }
                    for phase_label in ("a", "b", "c")
                },
                "edge_policy": "turn_off_immediate_turn_on_after_dead_time",
                "interlock": "intrinsic_non_disableable",
            }
        )
    dq_blocks: dict[str, Any] = {}
    svpwm_blocks: dict[str, Any] = {}
    for control in sorted(model.sampled_controls.values(), key=lambda item: item.id):
        if getattr(control, "kind", "sampled_pi_pwm") == "sampled_dq_current_pwm":
            dq_blocks.setdefault(control.block_id, control)
            continue
        if getattr(control, "kind", "") == "signal_svpwm":
            svpwm_blocks.setdefault(control.block_id, control)
            continue
        if getattr(control, "kind", "") == "signal_pwm":
            controls.append(
                {
                    "id": control.id,
                    "type": "signal_pwm",
                    "frequency_hz": control.frequency,
                    "duty_signal": control.duty_signal,
                    "duty_min": control.duty_min,
                    "duty_max": control.duty_max,
                    "sampling": "signal_task_tick_pre_edge_state",
                }
            )
            continue
        controls.append(
            {
                "id": control.id,
                "type": "sampled_pi_pwm",
                "frequency_hz": control.frequency,
                "sense": control.sense,
                "reference_si": control.reference,
                "kp": control.kp,
                "ki": control.ki,
                "duty_min": control.duty_min,
                "duty_max": control.duty_max,
                "sampling": "cycle_boundary_pre_edge_state",
                "anti_windup": "conditional_integration",
            }
        )
    for block_id, block in sorted(svpwm_blocks.items()):
        controls.append(
            {
                "id": block_id,
                "type": "signal_svpwm",
                "frequency_hz": block.frequency,
                "alpha_signal": block.alpha_signal,
                "beta_signal": block.beta_signal,
                "dc_link_voltage_v": block.dc_link_voltage,
                "dead_time_s": block.dead_time,
                "modulator": "sector_space_vector_symmetric_zero_split",
                "sampling": "signal_task_tick_pre_edge_state",
            }
        )
    for block_id, block in sorted(dq_blocks.items()):
        controls.append(
            {
                "id": block_id,
                "type": "sampled_dq_current_pwm",
                "frequency_hz": block.frequency,
                "dead_time_s": block.dead_time,
                "sense_currents": list(block.sense_currents),
                "sense_voltages": list(block.sense_voltages),
                "id_ref_si": block.id_ref,
                "iq_ref_si": block.iq_ref,
                "current_kp": block.current_kp,
                "current_ki": block.current_ki,
                "voltage_scale_si": block.voltage_scale,
                "pll": {
                    "nominal_frequency_hz": block.pll_nominal_frequency_hz,
                    "bandwidth_hz": block.pll_bandwidth_hz,
                    "damping": block.pll_damping,
                    "theta0_rad": block.theta0_rad,
                },
                "reference_injection": block.injection,
                "duty_min": block.duty_min,
                "duty_max": block.duty_max,
                "sampling": "cycle_boundary_pre_edge_state_pre_update_theta",
                "edge_policy": "center_aligned_turn_off_immediate_turn_on_after_dead_time",
                "anti_windup": "conditional_integration_per_axis",
                "outputs": {
                    phase: {
                        "high": f"{block_id}.{phase}.high",
                        "low": f"{block_id}.{phase}.low",
                    }
                    for phase in ("a", "b", "c")
                },
            }
        )
    controls.sort(key=lambda item: item["id"])

    payload = {
        "schema": "pear-sim/physics-model/v0.1",
        "ground": model.ground,
        "components": [
            _component_dict(component) for component in sorted(model.components, key=lambda item: item.id)
        ],
        "controls": controls,
    }
    graph = getattr(model, "signals", None)
    if graph is not None:
        # The control law determines the gate edges, so it determines the
        # trajectory: two models whose signal graphs differ are not the same
        # physics and must not share a hash. Additive-only — a model with no
        # signal layer keeps its payload byte-for-byte.
        payload["signals"] = {
            "task": {"period_s": graph.task.period, "offset_s": graph.task.offset},
            "execution_order": list(graph.order),
            "blocks": [
                {
                    "id": block.id,
                    "type": block.kind,
                    "unit": block.unit,
                    "inputs": list(block.inputs),
                    "params": {
                        key: (list(value) if isinstance(value, tuple) else value)
                        for key, value in sorted(block.params.items())
                    },
                }
                for block in sorted(graph.blocks, key=lambda item: item.id)
            ],
        }
    if model.couplings:
        # Additive-only: models without couplings keep their exact payload.
        payload["couplings"] = [
            {
                "inductors": sorted((coupling.first, coupling.second)),
                "mutual_si": coupling.mutual,
                "model": "linear_mutual_inductance",
            }
            for coupling in sorted(
                model.couplings, key=lambda item: (item.first, item.second)
            )
        ]
    if model.forced_gate_windows:
        # Additive-only, and deliberately INSIDE the physics hash: a forced
        # gate changes what was simulated, so a perturbed run must never carry
        # the same `model_sha256` as its baseline.
        payload["forced_gate_windows"] = [
            window.as_dict() for window in model.forced_gate_windows
        ]
    return payload


def simulation_request_dict(model: Model) -> dict[str, Any]:
    settings = model.transient
    request = {
        "schema": "pear-sim/simulation-request/v0.1",
        "analysis": "transient",
        "start_s": settings.start,
        "stop_s": settings.stop,
        "output_step_s": settings.output_step,
        "solver": settings.solver,
        "rtol": settings.rtol,
        "atol": settings.atol,
        "max_step_s": settings.max_step,
        "outputs": list(model.outputs),
    }
    if model.harmonics is not None:
        request["harmonics"] = {
            "start_s": model.harmonics.start,
            "stop_s": model.harmonics.stop,
            "window_boundary": "half_open_[start,stop)",
            "fundamental_frequency_hz": model.harmonics.fundamental_frequency,
            "period_count": model.harmonics.period_count,
            "max_harmonic": model.harmonics.max_harmonic,
            "signals": [
                {"signal": signal.signal, "semantic": signal.semantic}
                for signal in model.harmonics.signals
            ],
        }
    return request


def model_hashes(model: Model) -> dict[str, str]:
    return {
        "source_sha256": model.source_sha256,
        "model_sha256": sha256_json(physics_model_dict(model)),
        "request_sha256": sha256_json(simulation_request_dict(model)),
    }


def resolved_model_dict(model: Model) -> dict[str, Any]:
    """Return a portable snapshot sufficient for report and review generation."""

    return {
        "schema": "pear-sim/resolved-model/v0.1",
        "source_schema": model.schema,
        "name": model.name,
        "source_file": model.source_path.name,
        "hashes": model_hashes(model),
        "parameters": {
            name: {"value_si": quantity.value, "dimension": quantity.dimension}
            for name, quantity in sorted(model.parameters.items())
        },
        "conventions": _measurement_conventions(),
        "initial_conditions": _initial_conditions_dict(model),
        "physics": physics_model_dict(model),
        "request": simulation_request_dict(model),
        # Emitted only when declared. `model_sha256` is taken over
        # `physics_model_dict`, which envelopes deliberately do not enter — a
        # declaration about how far a value may drift is not the physics of
        # this run. Absence here keeps every pre-existing bundle byte-identical.
        **(
            {
                "parameter_envelopes": {
                    name: envelope.as_dict()
                    for name, envelope in sorted(model.parameter_envelopes.items())
                }
            }
            if model.parameter_envelopes
            else {}
        ),
        # Emitted only when declared, and likewise OUTSIDE `model_sha256`:
        # a subcircuit instance is presentation — which flattened components
        # came from which declared unit — never physics.
        **(
            {
                "subcircuits": [
                    instance.as_dict()
                    for instance in sorted(
                        model.subcircuit_instances, key=lambda item: item.path
                    )
                ]
            }
            if model.subcircuit_instances
            else {}
        ),
    }


def explain_model(model: Model) -> dict[str, Any]:
    """Build a compact, machine-readable pre-simulation explanation."""

    nodes = sorted(
        {node for component in model.components for node in (component.p, component.n)}
    )
    component_counts = Counter(component.type for component in model.components)
    state_components = [
        component.id
        for component in model.components
        if component.type in {"capacitor", "inductor", "sat_inductor"}
    ]
    limitations = [
        "ideal switches and diodes; no switching-edge or EMI model",
        "no semiconductor loss or temperature calculation",
        "dependent capacitor loops and inductor cutsets are not yet supported",
        "result is exploratory and not suitable for hardware sign-off",
    ]
    initial_conditions = _initial_conditions_dict(model)
    logical_controls = physics_model_dict(model)["controls"]
    return {
        "schema": "pear-sim/explanation/v0.1",
        "name": model.name,
        "hashes": model_hashes(model),
        "summary": {
            "components": len(model.components),
            "component_types": dict(sorted(component_counts.items())),
            "nodes": len(nodes),
            "state_count": len(state_components),
            "state_components": state_components,
            "controls": len(logical_controls),
            "outputs": len(model.outputs),
            "warm_start": initial_conditions["summary"]["warm_start"],
        },
        "nodes": nodes,
        "ground": model.ground,
        "conventions": _measurement_conventions(),
        "initial_conditions": initial_conditions,
        "parameters": resolved_model_dict(model)["parameters"],
        "components": [_component_dict(component) for component in model.components],
        "controls": logical_controls,
        "request": simulation_request_dict(model),
        "limitations": limitations,
    }


def explanation_text(explanation: dict[str, Any]) -> str:
    summary = explanation["summary"]
    request = explanation["request"]
    conventions = explanation["conventions"]
    initial_conditions = explanation["initial_conditions"]
    initial_summary = initial_conditions["summary"]
    lines = [
        f"Model: {explanation['name']}",
        f"Physics hash: {explanation['hashes']['model_sha256']}",
        (
            "Topology: "
            f"{summary['components']} components, {summary['nodes']} nodes, "
            f"{summary['state_count']} energy states"
        ),
        (
            "Transient: "
            f"{request['start_s']:.9g}..{request['stop_s']:.9g} s, "
            f"solver={request['solver']}, output_step={request['output_step_s']:.9g} s"
        ),
        (
            "Conventions: "
            f"{conventions['component_voltage']['definition']}; "
            f"{conventions['component_current']['definition']}"
        ),
        (
            "Initial conditions: "
            f"strategy={initial_conditions['strategy']}, "
            f"zero={initial_summary['zero_count']}, "
            f"nonzero={initial_summary['nonzero_count']}, "
            f"warm_start={'yes' if initial_summary['warm_start'] else 'no'}, "
            "analytic_origin=not_claimed"
        ),
        "Outputs: " + ", ".join(request["outputs"]),
        "Limitations:",
        *(f"- {item}" for item in explanation["limitations"]),
    ]
    return "\n".join(lines)
