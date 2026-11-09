# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import yaml

from pear_sim.errors import ModelError
from pear_sim.limits import (
    MAX_RUN_OUTPUT_SAMPLES,
    HARMONIC_QUADRATURE_ORDER,
    MAX_HARMONIC_ORDER,
    MAX_HARMONIC_QUADRATURE_PANELS,
    MAX_HARMONIC_QUADRATURE_WORK_UNITS,
    MAX_HARMONIC_SIGNALS,
)
from pear_sim.spec.controls import (
    PwmControl,
    SampledDqCurrentPwmControl,
    SampledPiPwmControl,
    SignalPwmControl,
    SignalSvpwmControl,
    _reject_unknown,
    _require_model_text,
    _resolve,
    parse_controls,
    parse_sampled_dq_control,
    parse_sampled_pi_control,
    parse_signal_pwm_control,
    parse_signal_svpwm_control,
)
from pear_sim.spec.device_cards import LossAnalysisSpec, parse_loss_sections
from pear_sim.spec.perturbation import (
    ForcedGateWindow,
    parse_forced_gate_windows,
)
from pear_sim.spec.parameter_envelope import (
    ParameterEnvelope,
    parse_parameter_envelope,
)
from pear_sim.spec.signals import (
    SIGNAL_UNITS,
    declared_signal_target,
    is_signal_expression,
    machine_signal_target,
)
from pear_sim.spec.signal_graph import SignalGraph, parse_signal_graph
from pear_sim.spec.machines import (
    MachineSpec,
    machine_branch_definitions,
    machine_coupling_definitions,
    parse_machines,
)
from pear_sim.spec.subcircuits import (
    SubcircuitInstance,
    expand_subcircuits,
)
from pear_sim.spec.units import Quantity, parse_quantity
from pear_sim.spec.waveforms import (
    MAX_PWL_POINTS,
    DoubleExponentialWaveform,
    PulseWaveform,
    PwlWaveform,
    SineWaveform,
    StepWaveform,
    Waveform,
)


_OUTPUT_RE = re.compile(r"^[vi]\([A-Za-z_][A-Za-z0-9_.:-]*\)$")
_PARAMETER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_YAML_EXPONENT_FLOAT_RE = re.compile(
    r"^[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)$"
)
_VALUE_DIMENSIONS = {
    "resistor": "resistance",
    "capacitor": "capacitance",
    "inductor": "inductance",
    "sat_inductor": "inductance",
    "voltage_source": "voltage",
    "current_source": "current",
}
_SUPPORTED_COMPONENTS = frozenset(
    {
        "resistor",
        "capacitor",
        "inductor",
        "sat_inductor",
        "voltage_source",
        "current_source",
        "ideal_switch",
        "diode",
        "thyristor",
        "voltage_probe",
    }
)
MAX_MODEL_SOURCE_BYTES = 2 * 1024 * 1024


def _reject_unknown_component_types(data: Any) -> None:
    """Name an unknown component ``type`` before the schema does.

    The schema's component entry is a choice between a primitive and a
    subcircuit instance, so an unknown type surfaces as "'ref' is a required
    property" — true, and useless to whoever wrote ``type: mosfet``.
    """

    circuit = data.get("circuit") if isinstance(data, dict) else None
    components = circuit.get("components") if isinstance(circuit, dict) else None
    if not isinstance(components, list):
        return
    for index, component in enumerate(components):
        if not isinstance(component, dict) or "ref" in component:
            continue
        kind = component.get("type")
        if isinstance(kind, str) and kind not in _SUPPORTED_COMPONENTS:
            suggestion = (
                " A MOSFET is an ideal_switch with on_resistance plus an antiparallel "
                "diode (its body diode) with forward_voltage."
                if kind.lower() in {"mosfet", "igbt", "transistor", "switch", "gan", "sic"}
                else ""
            )
            raise ModelError(
                f"circuit.components[{index}] ({component.get('id', '?')}): unknown "
                f"component type {kind!r}; supported types: "
                f"{', '.join(sorted(_SUPPORTED_COMPONENTS))} (a subcircuit instance uses "
                f"`ref:` instead of `type:`).{suggestion}"
            )
MIN_RELATIVE_TOLERANCE = 100.0 * sys.float_info.epsilon
_HARMONIC_SEMANTICS = frozenset(
    {
        "line_voltage_ab",
        "line_voltage_bc",
        "line_voltage_ca",
        "phase_voltage_a",
        "phase_voltage_b",
        "phase_voltage_c",
        "phase_current_a",
        "phase_current_b",
        "phase_current_c",
        # Any recorded signal (a DC-side ripple, a single-phase quantity): no
        # family checks, no sequence components; repeatable.
        "generic",
    }
)


def _yaml_position(mark: Any) -> str:
    if mark is None:
        return "unknown position"
    return f"line {mark.line + 1}, column {mark.column + 1}"


class _ModelSafeLoader(yaml.SafeLoader):
    """Safe YAML loader with JSON-style exponent-only floats, no node reuse.

    PyYAML otherwise treats an unquoted scalar such as ``1e-10`` as text even
    though JSON and engineering notation treat it as a number.  Quoted values
    remain strings, which lets the runtime preserve the public schema's numeric
    rtol/atol contract.

    Anchors, aliases and merge keys are rejected.  The parser expands them
    before either the JSON Schema or ``_reject_unknown`` sees the document, so
    a reused node is invisible to every later check: ``<<: *leg`` arrives as an
    ordinary mapping and nothing can object to it.  Two consequences make that
    unacceptable for this format.  The JSON surface cannot express node reuse
    at all, so the two surfaces would stop being the same language — and
    reusing a component template is exactly the job of a declared subcircuit,
    where expansion is explicit, namespaced and hashed.  A
    silent YAML-level copy would be a second, unaudited hierarchy mechanism.
    """

    def compose_node(self, parent: Any, index: Any) -> Any:
        event = self.peek_event()
        # A well-formed document always declares the anchor first, so the
        # anchor branch below is what a reused node actually trips. The alias
        # branch is for an alias with no anchor at all: invalid YAML, where
        # naming the alias beats PyYAML's "found undefined alias".
        if isinstance(event, yaml.events.AliasEvent):
            raise ModelError(
                f"YAML alias '*{event.anchor}' at {_yaml_position(event.start_mark)}: "
                "node reuse is not part of pear-sim/v0.1 — write the value out, or "
                "use a declared subcircuit (`subcircuits:`)"
            )
        anchor = getattr(event, "anchor", None)
        if anchor is not None:
            raise ModelError(
                f"YAML anchor '&{anchor}' at {_yaml_position(event.start_mark)}: "
                "node reuse is not part of pear-sim/v0.1 — write the value out, or "
                "use a declared subcircuit (`subcircuits:`)"
            )
        return super().compose_node(parent, index)

    def flatten_mapping(self, node: Any) -> Any:
        # Reached for `<<: {inline: map}`, which needs no alias and would
        # otherwise slip past the alias check above.
        for key_node, _ in node.value:
            if key_node.tag == "tag:yaml.org,2002:merge":
                raise ModelError(
                    f"YAML merge key '<<' at {_yaml_position(key_node.start_mark)}: "
                    "node reuse is not part of pear-sim/v0.1 — write the fields out, "
                    "or use a declared subcircuit (`subcircuits:`)"
                )
        return super().flatten_mapping(node)


_ModelSafeLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    _YAML_EXPONENT_FLOAT_RE,
    list("-+0123456789"),
)


@dataclass(frozen=True)
class ComponentModelMeta:
    """Declared device-model provenance and validity envelope.

    This is a recorded declaration only: the reference kernel simulates ideal
    elements and does not evaluate operation against the envelope.  Runs whose
    model declares metadata receive an explicit not-evaluated warning.
    """

    version: str
    frequency_max_hz: float | None = None
    temperature_min_c: float | None = None
    temperature_max_c: float | None = None
    dvdt_max_v_per_s: float | None = None


@dataclass(frozen=True)
class Component:
    id: str
    type: str
    p: str
    n: str
    value: float | None = None
    control: str | None = None
    initial: bool = False
    initial_state: float = 0.0
    initial_state_explicit: bool = False
    model_meta: ComponentModelMeta | None = None
    waveform: "Waveform | None" = None
    sat_inductance: float | None = None
    sat_current: float | None = None
    #: Piecewise-linear conduction: a closed ideal_switch or a conducting diode
    #: obeys v = forward_voltage + on_resistance * i instead of v = 0. Both are
    #: zero for an ideal device; forward_voltage is diode-only.
    on_resistance: float = 0.0
    forward_voltage: float = 0.0

    @property
    def is_ideal_conductor(self) -> bool:
        return self.on_resistance == 0.0 and self.forward_voltage == 0.0


@dataclass(frozen=True)
class InductorCoupling:
    """Mutual coupling between two plain inductors (a two-winding transformer).

    ``mutual`` is the mutual inductance M in henries; the loader enforces
    M^2 < L1*L2 strictly (perfect coupling makes the inductance matrix
    singular and the ideal model ill-posed — fail closed, no hidden leakage).
    """

    first: str
    second: str
    mutual: float


@dataclass(frozen=True)
class TransientSettings:
    start: float
    stop: float
    output_step: float
    solver: str
    rtol: float
    atol: float
    max_step: float | None


@dataclass(frozen=True)
class HarmonicSignal:
    signal: str
    semantic: str


@dataclass(frozen=True)
class HarmonicSettings:
    start: float
    stop: float
    fundamental_frequency: float
    period_count: int
    max_harmonic: int
    signals: tuple[HarmonicSignal, ...]


@dataclass(frozen=True)
class Model:
    schema: str
    name: str
    source_path: Path
    source_sha256: str
    parameters: dict[str, Quantity]
    ground: str
    components: tuple[Component, ...]
    controls: dict[str, PwmControl]
    transient: TransientSettings
    outputs: tuple[str, ...]
    harmonics: HarmonicSettings | None = None
    sampled_controls: dict[str, SampledPiPwmControl] = dataclass_field(
        default_factory=dict
    )
    couplings: tuple[InductorCoupling, ...] = ()
    loss_analysis: LossAnalysisSpec | None = None
    machines: tuple[MachineSpec, ...] = ()
    signals: "SignalGraph | None" = None
    # Declared drift envelopes, keyed by parameter name. Empty for every model
    # that states plain scalars — which is every model written before this
    # existed, and is why their hashes do not move.
    parameter_envelopes: dict[str, ParameterEnvelope] = dataclass_field(
        default_factory=dict
    )
    # Gate overrides materialized from a perturbation request. Physics of THIS
    # run, so unlike envelopes these do enter `model_sha256` when present.
    forced_gate_windows: tuple[ForcedGateWindow, ...] = ()
    # Presentation records of expanded subcircuit instances. Hierarchy
    # is presentation, never physics: this field is deliberately absent from
    # `physics_model_dict`, so the canonical IR hash is a function of the
    # flattened component list only.
    subcircuit_instances: tuple[SubcircuitInstance, ...] = ()


def _parse_waveform(
    raw: Any,
    parameters: dict[str, Quantity],
    dimension: str,
    *,
    name: str,
) -> Waveform:
    """Parse a source waveform object; levels use the source dimension's units."""

    if not isinstance(raw, dict):
        raise ModelError(f"{name}: expected an object")
    kind = raw.get("type")
    if kind == "sine":
        _reject_unknown(
            raw,
            {"type", "offset", "amplitude", "frequency", "phase_deg"},
            name=name,
        )
        if "amplitude" not in raw or "frequency" not in raw:
            raise ModelError(f"{name}: sine requires amplitude and frequency")
        amplitude = _resolve(
            raw["amplitude"], parameters, dimension, name=f"{name}.amplitude"
        )
        frequency = _resolve(
            raw["frequency"], parameters, "frequency", name=f"{name}.frequency"
        )
        if frequency <= 0.0:
            raise ModelError(f"{name}.frequency must be positive")
        offset = _resolve(
            raw.get("offset", 0.0), parameters, dimension, name=f"{name}.offset"
        )
        phase_deg = _resolve(
            raw.get("phase_deg", 0.0),
            parameters,
            "dimensionless",
            name=f"{name}.phase_deg",
        )
        return SineWaveform(
            offset=offset,
            amplitude=amplitude,
            frequency_hz=frequency,
            phase_rad=math.radians(phase_deg),
        )
    if kind == "step":
        _reject_unknown(raw, {"type", "initial", "final", "at"}, name=name)
        for field in ("initial", "final", "at"):
            if field not in raw:
                raise ModelError(f"{name}: step requires initial, final and at")
        initial = _resolve(
            raw["initial"], parameters, dimension, name=f"{name}.initial"
        )
        final = _resolve(raw["final"], parameters, dimension, name=f"{name}.final")
        at_s = _resolve(raw["at"], parameters, "time", name=f"{name}.at")
        if not math.isfinite(at_s) or at_s < 0.0:
            raise ModelError(f"{name}.at must be a finite non-negative time")
        return StepWaveform(initial=initial, final=final, at_s=at_s)
    if kind == "pulse":
        _reject_unknown(
            raw,
            {"type", "low", "high", "frequency", "duty", "phase"},
            name=name,
        )
        for field in ("low", "high", "frequency", "duty"):
            if field not in raw:
                raise ModelError(
                    f"{name}: pulse requires low, high, frequency and duty"
                )
        low = _resolve(raw["low"], parameters, dimension, name=f"{name}.low")
        high = _resolve(raw["high"], parameters, dimension, name=f"{name}.high")
        frequency = _resolve(
            raw["frequency"], parameters, "frequency", name=f"{name}.frequency"
        )
        if frequency <= 0.0:
            raise ModelError(f"{name}.frequency must be positive")
        duty = _resolve(
            raw["duty"], parameters, "dimensionless", name=f"{name}.duty"
        )
        if not 0.0 < duty < 1.0:
            raise ModelError(f"{name}.duty must lie strictly between 0 and 1")
        phase_s = _resolve(
            raw.get("phase", 0.0), parameters, "time", name=f"{name}.phase"
        )
        return PulseWaveform(
            low=low, high=high, frequency_hz=frequency, duty=duty, phase_s=phase_s
        )
    if kind == "pwl":
        _reject_unknown(raw, {"type", "points"}, name=name)
        raw_points = raw.get("points")
        if not isinstance(raw_points, list) or len(raw_points) < 2:
            raise ModelError(f"{name}.points must list at least two [time, value] pairs")
        if len(raw_points) > MAX_PWL_POINTS:
            raise ModelError(
                f"{name}.points exceeds the {MAX_PWL_POINTS}-point limit"
            )
        points: list[tuple[float, float]] = []
        for index, raw_point in enumerate(raw_points):
            if not isinstance(raw_point, list) or len(raw_point) != 2:
                raise ModelError(
                    f"{name}.points[{index}] must be a [time, value] pair"
                )
            point_time = _resolve(
                raw_point[0], parameters, "time", name=f"{name}.points[{index}][0]"
            )
            point_value = _resolve(
                raw_point[1], parameters, dimension, name=f"{name}.points[{index}][1]"
            )
            if not math.isfinite(point_time):
                raise ModelError(f"{name}.points[{index}][0] must be finite")
            points.append((point_time, point_value))
        for (t0, _), (t1, _) in zip(points, points[1:]):
            if t1 <= t0:
                raise ModelError(
                    f"{name}.points times must be strictly increasing"
                )
        return PwlWaveform(points=tuple(points))
    if kind == "double_exponential":
        _reject_unknown(
            raw, {"type", "amplitude", "tau_rise", "tau_fall", "at"}, name=name
        )
        for field in ("amplitude", "tau_rise", "tau_fall", "at"):
            if field not in raw:
                raise ModelError(
                    f"{name}: double_exponential requires amplitude, tau_rise, "
                    "tau_fall and at"
                )
        amplitude = _resolve(
            raw["amplitude"], parameters, dimension, name=f"{name}.amplitude"
        )
        tau_rise = _resolve(
            raw["tau_rise"], parameters, "time", name=f"{name}.tau_rise"
        )
        tau_fall = _resolve(
            raw["tau_fall"], parameters, "time", name=f"{name}.tau_fall"
        )
        at_s = _resolve(raw["at"], parameters, "time", name=f"{name}.at")
        if not math.isfinite(tau_rise) or tau_rise <= 0.0:
            raise ModelError(f"{name}.tau_rise must be finite and positive")
        if not math.isfinite(tau_fall) or tau_fall <= 0.0:
            raise ModelError(f"{name}.tau_fall must be finite and positive")
        if tau_fall <= tau_rise:
            # Equal time constants cancel the shape identically; a fall shorter
            # than the rise inverts it. Neither is regularized into something
            # plausible-looking.
            raise ModelError(
                f"{name}: tau_fall must strictly exceed tau_rise "
                f"(got {tau_fall!r} <= {tau_rise!r})"
            )
        if not math.isfinite(at_s) or at_s < 0.0:
            raise ModelError(f"{name}.at must be a finite non-negative time")
        if not math.isfinite(amplitude):
            raise ModelError(f"{name}.amplitude must be finite")
        return DoubleExponentialWaveform(
            amplitude=amplitude,
            tau_rise_s=tau_rise,
            tau_fall_s=tau_fall,
            start_s=at_s,
        )
    raise ModelError(
        f"{name}.type must be one of sine, step, pulse, pwl, "
        f"double_exponential (got {kind!r})"
    )


def _read_document(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ModelError(f"Cannot inspect {path}: {exc}") from exc
    if size > MAX_MODEL_SOURCE_BYTES:
        raise ModelError(
            f"Model source exceeds the {MAX_MODEL_SOURCE_BYTES}-byte limit: {path}"
        )
    raw_bytes = path.read_bytes()
    if len(raw_bytes) > MAX_MODEL_SOURCE_BYTES:
        raise ModelError(
            f"Model source exceeds the {MAX_MODEL_SOURCE_BYTES}-byte limit: {path}"
        )
    try:
        if path.suffix.lower() == ".json":
            data = json.loads(raw_bytes)
        else:
            data = yaml.load(raw_bytes, Loader=_ModelSafeLoader)
    except (json.JSONDecodeError, yaml.YAMLError, UnicodeDecodeError) as exc:
        raise ModelError(f"Cannot parse {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ModelError("Model root must be an object")
    _validate_against_schema(data)
    return data, raw_bytes


_SCHEMA_VALIDATOR: Any = None


def _schema_validator() -> Any:
    """Build (once) the Draft 2020-12 validator for the packaged model schema."""

    global _SCHEMA_VALIDATOR
    if _SCHEMA_VALIDATOR is None:
        import jsonschema

        schema_path = Path(__file__).with_name("model.schema.json")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator_cls = jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)
        _SCHEMA_VALIDATOR = validator_cls(schema)
    return _SCHEMA_VALIDATOR


def _schema_error_message(error: Any) -> str:
    """Map a jsonschema error to the loader's diagnostic phrasing."""

    pointer = "/".join(str(part) for part in error.absolute_path)
    location = f"/{pointer}" if pointer else "model root"
    if error.validator == "additionalProperties":
        allowed = set((error.schema or {}).get("properties", {}))
        present = set(error.instance) if isinstance(error.instance, dict) else set()
        unknown = sorted(present - allowed)
        return f"{location}: unknown field(s): {', '.join(unknown)}"
    if error.validator == "required":
        return f"{location}: {error.message}"
    return f"{location}: {error.message}"


def _subtree_has_type_mismatch(error: Any) -> bool:
    """True when a branch's failure includes disagreeing with a declared type.

    Used to skip oneOf branches selected for a different control ``type`` than
    the document declares, so diagnostics come from the intended branch.
    """

    if error.validator in {"const", "enum"} and list(error.absolute_path)[-1:] == ["type"]:
        return True
    return any(_subtree_has_type_mismatch(child) for child in error.context or [])


def _validate_against_schema(data: dict[str, Any]) -> None:
    """Reject documents that do not conform to the normative JSON Schema.

    Structural validation runs before semantic validation so the published
    schema is enforced, not merely documented.  Semantic rules the schema
    cannot express (units, parameter references, control expansion, topology)
    remain in the loader.
    """

    def pick(errors: list[Any]) -> Any:
        # Branches that merely disagree with the declared control ``type``
        # discriminator are skipped when an intended-branch error exists —
        # otherwise a missing SPWM field would be reported as the pwm
        # branch's complaint. Among the rest, prefer the deepest, most
        # specific diagnostic.
        candidates = [
            error for error in errors if not _subtree_has_type_mismatch(error)
        ] or errors
        return max(
            candidates,
            key=lambda error: (
                len(list(error.absolute_path)),
                error.validator not in {"oneOf", "anyOf"},
            ),
        )

    _reject_unknown_component_types(data)
    errors = list(_schema_validator().iter_errors(data))
    if errors:
        error = pick(errors)
        while error.context:
            error = pick(list(error.context))
        raise ModelError(
            "Model does not conform to schema pear-sim/v0.1 at "
            + _schema_error_message(error)
        )


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"on", "true", "closed", "conducting"}:
        return True
    if isinstance(value, str) and value.lower() in {"off", "false", "open", "blocking"}:
        return False
    raise ModelError(f"{name}: expected on/off or boolean")


def _parse_component_model_meta(raw: Any, *, name: str) -> ComponentModelMeta | None:
    """Parse a declared device-model provenance/validity record.

    Bounds are plain finite SI/degree-Celsius numbers by contract — no unit
    strings and no ``${parameter}`` references — because they describe the
    device model, not the resolved circuit.
    """

    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    _reject_unknown(raw, {"version", "validity"}, name=name)
    version = _require_model_text(raw.get("version"), name=f"{name}.version")
    validity = raw.get("validity", {})
    if not isinstance(validity, dict):
        raise ModelError(f"{name}.validity must be an object")
    allowed = {
        "frequency_max_hz",
        "temperature_min_c",
        "temperature_max_c",
        "dvdt_max_v_per_s",
    }
    _reject_unknown(validity, allowed, name=f"{name}.validity")
    bounds: dict[str, float | None] = {}
    for field in sorted(allowed):
        raw_bound = validity.get(field)
        if raw_bound is None:
            bounds[field] = None
            continue
        if isinstance(raw_bound, bool) or not isinstance(raw_bound, (int, float)):
            raise ModelError(f"{name}.validity.{field} must be a plain finite number")
        bound = float(raw_bound)
        if not math.isfinite(bound):
            raise ModelError(f"{name}.validity.{field} must be a plain finite number")
        bounds[field] = bound
    frequency = bounds["frequency_max_hz"]
    if frequency is not None and frequency <= 0.0:
        raise ModelError(f"{name}.validity.frequency_max_hz must be positive")
    dvdt = bounds["dvdt_max_v_per_s"]
    if dvdt is not None and dvdt <= 0.0:
        raise ModelError(f"{name}.validity.dvdt_max_v_per_s must be positive")
    minimum = bounds["temperature_min_c"]
    maximum = bounds["temperature_max_c"]
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ModelError(
            f"{name}.validity.temperature_min_c must not exceed temperature_max_c"
        )
    return ComponentModelMeta(
        version=version,
        frequency_max_hz=frequency,
        temperature_min_c=minimum,
        temperature_max_c=maximum,
        dvdt_max_v_per_s=dvdt,
    )


def _parse_parameters(
    raw: Any, overrides: dict[str, str] | None
) -> tuple[dict[str, Quantity], dict[str, ParameterEnvelope]]:
    """Parse ``parameters:`` into nominal scalars plus any declared envelopes.

    Both forms live in the same section. A plain scalar (``vin: 100 V``) is
    unchanged. The object form additionally declares corners, a distribution
    or a life model together with its provenance; the *nominal* value it
    carries is what lands in ``Model.parameters``, so the resolved physics of
    a nominal run is identical either way.

    An override replaces the nominal value only. It deliberately does not
    rewrite the envelope: the campaign producer generates override values
    *from* the envelope, and letting an override silently redefine the
    envelope it came from would make a corner unable to name its own origin.
    """

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ModelError("parameters must be an object")
    parsed: dict[str, Quantity] = {}
    envelopes: dict[str, ParameterEnvelope] = {}
    for raw_name, value in raw.items():
        name = _require_model_text(raw_name, name="parameter name")
        if _PARAMETER_NAME_RE.fullmatch(name) is None:
            raise ModelError(
                f"parameter name must match {_PARAMETER_NAME_RE.pattern!r}"
            )
        if isinstance(value, dict):
            nominal, envelope = parse_parameter_envelope(value, parameter=name)
            parsed[name] = nominal
            envelopes[name] = envelope
        else:
            parsed[name] = parse_quantity(value, name=f"parameter {name}")
    for name, value in (overrides or {}).items():
        if name not in parsed:
            raise ModelError(f"Cannot override unknown parameter {name!r}")
        expected = parsed[name].dimension
        parsed[name] = parse_quantity(value, expected, name=f"override {name}")
    return parsed, envelopes


def _reject_unreferenced_overrides(
    data: dict[str, Any], overrides: dict[str, Any] | None
) -> None:
    """Refuse an override of a parameter that no value references.

    Such an override is a silent no-op: the caller believes the design changed
    and every result is identical. References are ``${name}`` strings anywhere
    outside the ``parameters`` block itself.
    """

    if not overrides:
        return
    referenced: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, str):
            referenced.update(_PARAMETER_REFERENCE_RE.findall(node))
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk({key: value for key, value in data.items() if key != "parameters"})
    for name in overrides:
        if name not in referenced:
            raise ModelError(
                f"Cannot override parameter {name!r}: it is declared but no value "
                f"references it as '${{{name}}}', so the override would change nothing"
            )


_PARAMETER_REFERENCE_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _parse_harmonics(
    raw: Any,
    parameters: dict[str, Quantity],
    transient: TransientSettings,
    outputs: tuple[str, ...],
) -> HarmonicSettings | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ModelError("analysis.harmonics must be an object")
    _reject_unknown(
        raw,
        {
            "start",
            "stop",
            "fundamental_frequency",
            "max_harmonic",
            "signals",
        },
        name="analysis.harmonics",
    )
    missing = sorted(
        {
            "start",
            "stop",
            "fundamental_frequency",
            "max_harmonic",
            "signals",
        }
        - set(raw)
    )
    if missing:
        raise ModelError(
            "analysis.harmonics: missing required field(s): " + ", ".join(missing)
        )

    start = _resolve(
        raw["start"],
        parameters,
        "time",
        name="analysis.harmonics.start",
    )
    stop = _resolve(
        raw["stop"],
        parameters,
        "time",
        name="analysis.harmonics.stop",
    )
    fundamental_frequency = _resolve(
        raw["fundamental_frequency"],
        parameters,
        "frequency",
        name="analysis.harmonics.fundamental_frequency",
    )
    if not transient.start <= start < stop <= transient.stop:
        raise ModelError(
            "analysis.harmonics window [start, stop) must lie within the transient window"
        )
    if fundamental_frequency <= 0.0:
        raise ModelError("analysis.harmonics.fundamental_frequency must be positive")

    period_count_value = (stop - start) * fundamental_frequency
    if not math.isfinite(period_count_value):
        raise ModelError(
            "analysis.harmonics period count must fit the finite portable contract"
        )
    nearest_period_count = round(period_count_value)
    integer_tolerance = min(0.25, 8.0 * math.ulp(period_count_value))
    if nearest_period_count < 1 or not math.isclose(
        period_count_value,
        float(nearest_period_count),
        rel_tol=0.0,
        abs_tol=integer_tolerance,
    ):
        raise ModelError(
            "analysis.harmonics window must contain an integer number of "
            "fundamental periods"
        )

    max_harmonic = raw["max_harmonic"]
    if isinstance(max_harmonic, bool) or not isinstance(max_harmonic, int):
        raise ModelError("analysis.harmonics.max_harmonic must be an integer")
    if not 2 <= max_harmonic <= MAX_HARMONIC_ORDER:
        raise ModelError(
            "analysis.harmonics.max_harmonic must be in "
            f"[2, {MAX_HARMONIC_ORDER}]"
        )
    minimum_quadrature_panels = 4 * int(nearest_period_count) * max_harmonic
    if minimum_quadrature_panels > MAX_HARMONIC_QUADRATURE_PANELS:
        raise ModelError(
            "analysis.harmonics request requires at least "
            f"{minimum_quadrature_panels} quadrature panels, exceeding the portable "
            f"limit of {MAX_HARMONIC_QUADRATURE_PANELS}"
        )

    raw_signals = raw["signals"]
    if not isinstance(raw_signals, list) or not raw_signals:
        raise ModelError("analysis.harmonics.signals must be a non-empty array")
    if len(raw_signals) > MAX_HARMONIC_SIGNALS:
        raise ModelError(
            f"analysis.harmonics.signals exceeds the {MAX_HARMONIC_SIGNALS}-signal limit"
        )
    parsed_signals: list[HarmonicSignal] = []
    seen_signals: set[str] = set()
    seen_semantics: set[str] = set()
    for index, raw_signal in enumerate(raw_signals):
        prefix = f"analysis.harmonics.signals[{index}]"
        if not isinstance(raw_signal, dict):
            raise ModelError(f"{prefix} must be an object")
        _reject_unknown(raw_signal, {"signal", "semantic"}, name=prefix)
        if set(raw_signal) != {"signal", "semantic"}:
            raise ModelError(f"{prefix} requires signal and semantic")
        signal = _require_model_text(raw_signal["signal"], name=f"{prefix}.signal")
        if _OUTPUT_RE.fullmatch(signal) is None:
            raise ModelError(
                f"{prefix}.signal must be a canonical v(target) or i(target) expression"
            )
        if signal not in outputs:
            raise ModelError(f"{prefix}.signal {signal!r} is not present in outputs")
        semantic = _require_model_text(
            raw_signal["semantic"],
            name=f"{prefix}.semantic",
        )
        if semantic not in _HARMONIC_SEMANTICS:
            raise ModelError(f"{prefix}.semantic {semantic!r} is unsupported")
        if semantic != "generic" and signal.startswith("v(") and not (
            semantic.startswith("line_voltage_")
            or semantic.startswith("phase_voltage_")
        ):
            raise ModelError(
                f"{prefix}.semantic {semantic!r} is incompatible with voltage signal "
                f"{signal!r}"
            )
        if semantic != "generic" and signal.startswith("i(") and not semantic.startswith(
            "phase_current_"
        ):
            raise ModelError(
                f"{prefix}.semantic {semantic!r} is incompatible with current signal "
                f"{signal!r}"
            )
        if signal in seen_signals:
            raise ModelError(
                "analysis.harmonics.signals must map every signal at most once"
            )
        if semantic != "generic" and semantic in seen_semantics:
            raise ModelError(
                "analysis.harmonics.signals must map every semantic at most once"
            )
        seen_signals.add(signal)
        seen_semantics.add(semantic)
        parsed_signals.append(HarmonicSignal(signal=signal, semantic=semantic))

    minimum_work_units = (
        minimum_quadrature_panels
        * HARMONIC_QUADRATURE_ORDER
        * (max_harmonic + 1)
        * (len(parsed_signals) + 1)
    )
    if minimum_work_units > MAX_HARMONIC_QUADRATURE_WORK_UNITS:
        raise ModelError(
            "analysis.harmonics minimum quadrature work estimate is "
            f"{minimum_work_units} units, exceeding the portable limit of "
            f"{MAX_HARMONIC_QUADRATURE_WORK_UNITS}"
        )

    return HarmonicSettings(
        start=start,
        stop=stop,
        fundamental_frequency=fundamental_frequency,
        period_count=int(nearest_period_count),
        max_harmonic=max_harmonic,
        signals=tuple(parsed_signals),
    )


def load_model(path: str | Path, overrides: dict[str, str] | None = None) -> Model:
    source_path = Path(path).resolve()
    data, raw_bytes = _read_document(source_path)
    _reject_unknown(
        data,
        {
            "schema",
            "name",
            "parameters",
            "circuit",
            "controls",
            "analysis",
            "outputs",
            "device_cards",
            "cooling",
            "loss_analysis",
            "machines",
            "signals",
            "subcircuits",
            # Materialized by the perturbation layer, not normally hand-written.
            "perturbations",
        },
        name="model",
    )
    schema = data.get("schema")
    if schema != "pear-sim/v0.1":
        raise ModelError(f"Unsupported schema {schema!r}; expected 'pear-sim/v0.1'")

    # Declared subcircuits EXPAND source-to-source before any semantic
    # validation (the machine precedent): every derived component is checked
    # by the same rules, under its namespaced `<instance>.<inner>` id.
    data, subcircuit_instances = expand_subcircuits(data)
    name = data.get("name", source_path.stem)
    name = _require_model_text(name, name="name")
    parameters, parameter_envelopes = _parse_parameters(
        data.get("parameters"), overrides
    )
    _reject_unreferenced_overrides(data, overrides)

    circuit = data.get("circuit")
    if not isinstance(circuit, dict):
        raise ModelError("circuit must be an object")
    _reject_unknown(circuit, {"ground", "components", "couplings"}, name="circuit")
    ground = circuit.get("ground", "gnd")
    ground = _require_model_text(ground, name="circuit.ground")
    raw_components = circuit.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise ModelError("circuit.components must be a non-empty list")

    # A declared machine EXPANDS into ordinary branches before validation, so
    # every generated resistor, inductor and back-EMF source is checked by the
    # same rules and is visible to a reader of the resolved model.
    machines = parse_machines(data.get("machines"))
    signal_graph = parse_signal_graph(
        data.get("signals"),
        machine_ids=frozenset(m.id for m in machines),
        parameters=parameters,
    )
    raw_components = list(raw_components)
    machine_couplings: list[dict[str, Any]] = []
    for machine in machines:
        raw_components.extend(machine_branch_definitions(machine))
        machine_couplings.extend(machine_coupling_definitions(machine))

    components: list[Component] = []
    ids: set[str] = set()
    for index, raw_component in enumerate(raw_components):
        prefix = f"circuit.components[{index}]"
        if not isinstance(raw_component, dict):
            raise ModelError(f"{prefix}: expected an object")
        _reject_unknown(
            raw_component,
            {
                "id",
                "type",
                "p",
                "n",
                "value",
                "waveform",
                "l_sat",
                "i_sat",
                "control",
                "initial",
                "initial_voltage",
                "initial_current",
                "model_meta",
                "on_resistance",
                "forward_voltage",
            },
            name=prefix,
        )
        component_id = raw_component.get("id")
        component_type = raw_component.get("type")
        p = raw_component.get("p")
        n = raw_component.get("n")
        component_id = _require_model_text(component_id, name=f"{prefix}.id")
        if component_id in ids:
            raise ModelError(f"Duplicate component id {component_id!r}")
        ids.add(component_id)
        if component_type not in _SUPPORTED_COMPONENTS:
            raise ModelError(f"{prefix}.type: unsupported component {component_type!r}")
        p = _require_model_text(p, name=f"{prefix}.p")
        n = _require_model_text(n, name=f"{prefix}.n")
        if p == n:
            raise ModelError(f"{component_id}: p and n cannot be the same node")

        value: float | None = None
        waveform: Waveform | None = None
        is_source = component_type in {"voltage_source", "current_source"}
        if "waveform" in raw_component and not is_source:
            raise ModelError(
                f"{component_id}: waveform is only valid for voltage_source and "
                "current_source"
            )
        if component_type in _VALUE_DIMENSIONS:
            has_value = "value" in raw_component
            has_waveform = is_source and "waveform" in raw_component
            if has_value and has_waveform:
                raise ModelError(
                    f"{component_id}: value and waveform are mutually exclusive"
                )
            if not has_value and not has_waveform:
                raise ModelError(f"{component_id}: missing value")
            if has_value:
                value = _resolve(
                    raw_component["value"],
                    parameters,
                    _VALUE_DIMENSIONS[component_type],
                    name=f"{component_id}.value",
                )
                if (
                    component_type in {"resistor", "capacitor", "inductor", "sat_inductor"}
                    and value <= 0
                ):
                    raise ModelError(f"{component_id}.value must be positive")
            else:
                waveform = _parse_waveform(
                    raw_component["waveform"],
                    parameters,
                    _VALUE_DIMENSIONS[component_type],
                    name=f"{component_id}.waveform",
                )

        sat_inductance: float | None = None
        sat_current: float | None = None
        if component_type == "sat_inductor":
            for field_name in ("l_sat", "i_sat"):
                if field_name not in raw_component:
                    raise ModelError(f"{component_id}: sat_inductor requires {field_name}")
            sat_inductance = _resolve(
                raw_component["l_sat"],
                parameters,
                "inductance",
                name=f"{component_id}.l_sat",
            )
            if sat_inductance <= 0:
                raise ModelError(f"{component_id}.l_sat must be positive")
            if value is not None and sat_inductance >= value:
                raise ModelError(
                    f"{component_id}.l_sat must be smaller than the unsaturated value"
                )
            sat_current = _resolve(
                raw_component["i_sat"],
                parameters,
                "current",
                name=f"{component_id}.i_sat",
            )
            if sat_current <= 0:
                raise ModelError(f"{component_id}.i_sat must be positive")
        elif "l_sat" in raw_component or "i_sat" in raw_component:
            raise ModelError(
                f"{component_id}: l_sat/i_sat are only valid for sat_inductor"
            )

        control = raw_component.get("control")
        if component_type == "ideal_switch":
            control = _require_model_text(
                control,
                name=f"{component_id}.control",
            )
        elif component_type == "thyristor":
            # A gate is optional: a thyristor with none can never fire, which
            # models a permanently blocking device and is a legitimate thing to
            # declare rather than an error to guess about.
            if control is not None:
                control = _require_model_text(
                    control,
                    name=f"{component_id}.control",
                )
        elif control is not None:
            raise ModelError(
                f"{component_id}: control is only valid for ideal_switch and thyristor"
            )

        on_resistance = 0.0
        forward_voltage = 0.0
        if "on_resistance" in raw_component:
            if component_type not in {"ideal_switch", "diode"}:
                raise ModelError(
                    f"{component_id}: on_resistance is only valid for ideal_switch and diode"
                )
            on_resistance = _resolve(
                raw_component["on_resistance"],
                parameters,
                "resistance",
                name=f"{component_id}.on_resistance",
            )
            if not on_resistance >= 0.0:
                raise ModelError(f"{component_id}.on_resistance must be zero or positive")
        if "forward_voltage" in raw_component:
            if component_type != "diode":
                raise ModelError(f"{component_id}: forward_voltage is only valid for diode")
            forward_voltage = _resolve(
                raw_component["forward_voltage"],
                parameters,
                "voltage",
                name=f"{component_id}.forward_voltage",
            )
            if not forward_voltage >= 0.0:
                raise ModelError(f"{component_id}.forward_voltage must be zero or positive")

        initial = False
        if component_type in {"ideal_switch", "diode", "thyristor"}:
            initial = _as_bool(raw_component.get("initial", False), name=f"{component_id}.initial")
        initial_state = 0.0
        initial_state_explicit = False
        if component_type == "capacitor":
            initial_state_explicit = "initial_voltage" in raw_component
            initial_state = _resolve(
                raw_component.get("initial_voltage", 0.0),
                parameters,
                "voltage",
                name=f"{component_id}.initial_voltage",
            )
        elif component_type in {"inductor", "sat_inductor"}:
            initial_state_explicit = "initial_current" in raw_component
            initial_state = _resolve(
                raw_component.get("initial_current", 0.0),
                parameters,
                "current",
                name=f"{component_id}.initial_current",
            )
        model_meta = _parse_component_model_meta(
            raw_component.get("model_meta"),
            name=f"{component_id}.model_meta",
        )
        components.append(
            Component(
                component_id,
                component_type,
                p,
                n,
                value,
                control,
                initial,
                initial_state,
                initial_state_explicit,
                model_meta,
                waveform=waveform,
                sat_inductance=sat_inductance,
                sat_current=sat_current,
                on_resistance=on_resistance,
                forward_voltage=forward_voltage,
            )
        )

    couplings: list[InductorCoupling] = []
    declared_pairs: set[tuple[str, str]] = set()
    raw_couplings = circuit.get("couplings", [])
    if isinstance(raw_couplings, list) and machine_couplings:
        # A salient machine's windings ARE a coupled group. Appended after the
        # hand-written ones so a model may still couple its own inductors.
        raw_couplings = list(raw_couplings) + machine_couplings
    if not isinstance(raw_couplings, list):
        raise ModelError("circuit.couplings must be a list")
    components_by_id = {component.id: component for component in components}
    coupled_ids: set[str] = set()
    for index, raw_coupling in enumerate(raw_couplings):
        prefix = f"circuit.couplings[{index}]"
        if not isinstance(raw_coupling, dict):
            raise ModelError(f"{prefix}: expected an object")
        _reject_unknown(
            raw_coupling, {"inductors", "mutual", "coupling"}, name=prefix
        )
        raw_pair = raw_coupling.get("inductors")
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            raise ModelError(f"{prefix}.inductors must list exactly two inductor ids")
        pair: list[Component] = []
        for raw_id in raw_pair:
            member_id = _require_model_text(raw_id, name=f"{prefix}.inductors")
            member = components_by_id.get(member_id)
            if member is None or member.type != "inductor":
                raise ModelError(
                    f"{prefix}: {member_id!r} is not a plain inductor "
                    "(sat_inductor cannot be coupled in v0.1)"
                )
            coupled_ids.add(member_id)
            pair.append(member)
        if pair[0].id == pair[1].id:
            raise ModelError(f"{prefix}: cannot couple an inductor to itself")
        pair_key = (min(pair[0].id, pair[1].id), max(pair[0].id, pair[1].id))
        if pair_key in declared_pairs:
            raise ModelError(
                f"{prefix}: the pair {pair_key[0]!r}/{pair_key[1]!r} is already "
                "coupled; declare each mutual once"
            )
        declared_pairs.add(pair_key)
        has_mutual = "mutual" in raw_coupling
        has_k = "coupling" in raw_coupling
        if has_mutual == has_k:
            raise ModelError(
                f"{prefix}: declare exactly one of mutual (henries) or "
                "coupling (dimensionless k)"
            )
        l1 = float(pair[0].value)
        l2 = float(pair[1].value)
        if has_mutual:
            mutual = _resolve(
                raw_coupling["mutual"], parameters, "inductance", name=f"{prefix}.mutual"
            )
        else:
            k = _resolve(
                raw_coupling["coupling"],
                parameters,
                "dimensionless",
                name=f"{prefix}.coupling",
            )
            # A NEGATIVE k is physics, not an error. The sign of a mutual
            # inductance is a statement about winding orientation, and in a
            # three-phase star the phase-to-phase couplings are negative:
            # flux from one phase opposes the other two. That case cannot be
            # re-expressed by flipping an inductor's orientation, because all
            # three windings are symmetric — flipping one to fix a pair
            # breaks the other pair. Zero stays refused: an undeclared pair
            # already means uncoupled, so declaring zero says nothing.
            if not 0.0 < abs(k) < 1.0:
                raise ModelError(
                    f"{prefix}.coupling magnitude must lie strictly between "
                    "0 and 1"
                )
            mutual = k * math.sqrt(l1 * l2)
        if mutual == 0.0:
            raise ModelError(
                f"{prefix}: a zero mutual says nothing — an undeclared pair is "
                "already uncoupled; remove the entry"
            )
        # The decisive physical gate is NOT the sign but positive definiteness
        # of the assembled group matrix, which is checked below on the whole
        # group and rejects any set of windings that would store negative
        # energy. M^2 >= L1*L2 stays here because it is the pairwise case of
        # exactly that condition and gives the better error message.
        if mutual * mutual >= l1 * l2:
            raise ModelError(
                f"{prefix}: perfect or over-unity coupling (M^2 >= L1*L2) makes "
                "the inductance matrix singular; declare leakage explicitly"
            )
        couplings.append(
            InductorCoupling(first=pair[0].id, second=pair[1].id, mutual=mutual)
        )

    if couplings:
        # Pairwise k < 1 is necessary but NOT sufficient once three or more
        # windings share a core: the group's inductance matrix must be
        # positive definite, otherwise some current set would carry negative
        # stored energy and the derivative map would be unphysical. Cholesky
        # is the decisive test and it runs on the assembled group.
        import numpy as _np
        from pear_sim.core.circuit import _coupling_groups as _groups

        inductances = {
            component.id: float(component.value)
            for component in components
            if component.type == "inductor"
        }
        mutuals = {
            (min(c.first, c.second), max(c.first, c.second)): c.mutual
            for c in couplings
        }
        for group in _groups(couplings):
            size = len(group)
            matrix = _np.zeros((size, size), dtype=float)
            for row, member in enumerate(group):
                matrix[row, row] = inductances[member]
                for column in range(row + 1, size):
                    key = (min(member, group[column]), max(member, group[column]))
                    value = mutuals.get(key, 0.0)
                    matrix[row, column] = value
                    matrix[column, row] = value
            try:
                _np.linalg.cholesky(matrix)
            except _np.linalg.LinAlgError as exc:
                raise ModelError(
                    "circuit.couplings: the inductance matrix of the coupled "
                    f"group ({', '.join(group)}) is not positive definite, so "
                    "some winding-current set would store negative energy; "
                    "reduce the declared mutual couplings"
                ) from exc

    raw_controls = data.get("controls", {})
    if not isinstance(raw_controls, dict):
        raise ModelError("controls must be an object")
    raw_sampled = {
        control_id: raw
        for control_id, raw in raw_controls.items()
        if isinstance(raw, dict)
        and raw.get("type")
        in {"sampled_pi_pwm", "sampled_dq_current_pwm", "signal_pwm", "signal_svpwm"}
    }
    raw_open_loop = {
        control_id: raw
        for control_id, raw in raw_controls.items()
        if control_id not in raw_sampled
    }
    controls = parse_controls(raw_open_loop, parameters)
    sampled_controls: dict[
        str, SampledPiPwmControl | SampledDqCurrentPwmControl | SignalPwmControl
        | SignalSvpwmControl,
    ] = {}
    for control_id, raw in raw_sampled.items():
        if raw.get("type") == "signal_svpwm":
            expansion = parse_signal_svpwm_control(control_id, raw, parameters)
            representative = next(iter(expansion.values()))
            if signal_graph is None:
                raise ModelError(
                    f"control {control_id}: signal_svpwm requires a signals section"
                )
            block_ids = {block.id for block in signal_graph.blocks}
            for field, expression in (
                ("alpha_signal", representative.alpha_signal),
                ("beta_signal", representative.beta_signal),
            ):
                target = declared_signal_target(expression)
                if target is None:
                    raise ModelError(
                        f"control {control_id}.{field} must be s(block), got "
                        f"{expression!r}"
                    )
                if target not in block_ids:
                    raise ModelError(
                        f"control {control_id}.{field} reads {target!r}, which is "
                        "not a declared signal block"
                    )
            expected = 1.0 / signal_graph.task.period
            if abs(representative.frequency - expected) > 1e-9 * expected:
                raise ModelError(
                    f"control {control_id}.frequency "
                    f"({representative.frequency:.17g} Hz) must equal the signal "
                    f"task rate ({expected:.17g} Hz); declare one tick per "
                    "carrier cycle"
                )
            overlap = set(expansion) & set(sampled_controls)
            if overlap:
                raise ModelError(
                    "sampled control ids collide: " + ", ".join(sorted(overlap))
                )
            sampled_controls.update(expansion)
            continue
        if raw.get("type") == "signal_pwm":
            control = parse_signal_pwm_control(control_id, raw, parameters)
            target = declared_signal_target(control.duty_signal)
            if target is None:
                raise ModelError(
                    f"control {control_id}.duty_signal must be s(block), got "
                    f"{control.duty_signal!r}"
                )
            if signal_graph is None or target not in {
                block.id for block in signal_graph.blocks
            }:
                raise ModelError(
                    f"control {control_id}.duty_signal reads {target!r}, which is "
                    "not a declared signal block"
                )
            # One tick, one carrier cycle. A duty updated at a rate the
            # carrier does not share belongs to no single cycle, and choosing
            # one silently is exactly the timing assumption this format
            # refuses to make on the author's behalf.
            expected = 1.0 / signal_graph.task.period
            if abs(control.frequency - expected) > 1e-9 * expected:
                raise ModelError(
                    f"control {control_id}.frequency ({control.frequency:.17g} Hz) "
                    f"must equal the signal task rate ({expected:.17g} Hz); "
                    "declare one tick per carrier cycle"
                )
            sampled_controls[control_id] = control
            continue
        if raw.get("type") == "sampled_pi_pwm":
            sampled_controls[control_id] = parse_sampled_pi_control(
                control_id, raw, parameters
            )
            continue
        expansion = parse_sampled_dq_control(control_id, raw, parameters)
        overlap = set(expansion) & set(sampled_controls)
        if overlap:
            raise ModelError(
                "sampled control ids collide: " + ", ".join(sorted(overlap))
            )
        sampled_controls.update(expansion)
    duplicate_ids = set(controls) & set(sampled_controls)
    if duplicate_ids:
        raise ModelError(
            "control ids collide across open-loop and sampled controls: "
            + ", ".join(sorted(duplicate_ids))
        )

    for component in components:
        if component.type == "ideal_switch" and (
            component.control not in controls
            and component.control not in sampled_controls
        ):
            raise ModelError(f"{component.id}: unknown control {component.control!r}")

    analysis = data.get("analysis")
    if isinstance(analysis, dict):
        _reject_unknown(analysis, {"transient", "harmonics"}, name="analysis")
    transient_raw = analysis.get("transient") if isinstance(analysis, dict) else None
    if not isinstance(transient_raw, dict):
        raise ModelError("analysis.transient must be an object")
    _reject_unknown(
        transient_raw,
        {"start", "stop", "output_step", "solver", "rtol", "atol", "max_step"},
        name="analysis.transient",
    )
    start = _resolve(transient_raw.get("start", 0.0), parameters, "time", name="transient.start")
    stop = _resolve(transient_raw.get("stop"), parameters, "time", name="transient.stop")
    output_step = _resolve(
        transient_raw.get("output_step"), parameters, "time", name="transient.output_step"
    )
    if stop <= start:
        raise ModelError("transient.stop must be greater than start")
    if output_step <= 0 or output_step > stop - start:
        raise ModelError("transient.output_step must be positive and no larger than the time span")
    ratio = (stop - start) / output_step
    if not math.isfinite(ratio) or ratio + 1 > MAX_RUN_OUTPUT_SAMPLES:
        samples = f"{ratio:.3g}" if math.isfinite(ratio) else "unboundedly many"
        if math.isfinite(ratio) and ratio < 1e15:
            samples = str(math.floor(ratio + 1e-12) + 1)
        raise ModelError(
            f"transient requests {samples} output samples; the limit is "
            f"{MAX_RUN_OUTPUT_SAMPLES} per run. Raise output_step or shorten the run "
            "(a steady-state question is better answered by steady_state)"
        )
    solver = str(transient_raw.get("solver", "dopri5")).lower()
    if solver not in {"dopri5", "dop853", "radau", "bdf"}:
        raise ModelError("transient.solver must be dopri5, dop853, radau, or bdf")
    rtol_raw = transient_raw.get("rtol", 1e-7)
    atol_raw = transient_raw.get("atol", 1e-9)
    if (
        isinstance(rtol_raw, bool)
        or not isinstance(rtol_raw, (int, float))
        or isinstance(atol_raw, bool)
        or not isinstance(atol_raw, (int, float))
    ):
        raise ModelError("rtol and atol must be JSON numbers, not strings or booleans")
    try:
        rtol = float(rtol_raw)
        atol = float(atol_raw)
    except (OverflowError, ValueError) as exc:
        raise ModelError("rtol and atol must fit the finite float contract") from exc
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol <= 0 or atol <= 0:
        raise ModelError("rtol and atol must be finite and positive")
    if rtol < MIN_RELATIVE_TOLERANCE:
        raise ModelError(
            "rtol must be at least "
            f"{MIN_RELATIVE_TOLERANCE:.17g}; smaller values are silently raised by SciPy"
        )
    max_step_raw = transient_raw.get("max_step")
    max_step = (
        None
        if max_step_raw is None
        else _resolve(max_step_raw, parameters, "time", name="transient.max_step")
    )
    if max_step is not None and max_step <= 0:
        raise ModelError("transient.max_step must be positive")
    transient = TransientSettings(start, stop, output_step, solver, rtol, atol, max_step)

    raw_outputs = data.get("outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ModelError("outputs must be a non-empty list of signal expressions")
    outputs = tuple(
        _require_model_text(output, name=f"outputs[{index}]")
        for index, output in enumerate(raw_outputs)
    )
    machine_ids = {machine.id for machine in machines}
    signal_block_ids = (
        {block.id for block in signal_graph.blocks} if signal_graph else set()
    )
    for index, output in enumerate(outputs):
        block = declared_signal_target(output)
        if block is not None:
            # A declared signal block, recorded like any other output. Its
            # unit comes from the block, not from the expression.
            if block not in signal_block_ids:
                known = (
                    ", ".join(sorted(signal_block_ids))
                    if signal_block_ids
                    else "none declared"
                )
                raise ModelError(
                    f"outputs[{index}] reads signal block {block!r}, which is not "
                    f"declared (declared: {known})"
                )
            continue
        if not is_signal_expression(output):
            raise ModelError(
                f"outputs[{index}] must be a canonical signal expression: one of "
                f"{', '.join(sorted(SIGNAL_UNITS))} applied to a target, or "
                "s(block) for a declared signal block"
            )
        target = machine_signal_target(output)
        if target is None:
            continue
        # A machine signal names a declared machine, never a node or a
        # component. Resolving it against the circuit would find the expanded
        # branch ids (`motor.a.l`) and read something real but wrong.
        if target not in machine_ids:
            known = ", ".join(sorted(machine_ids)) if machine_ids else "none declared"
            raise ModelError(
                f"outputs[{index}] reads machine {target!r}, which is not a "
                f"declared machine (declared: {known})"
            )
    if len(set(outputs)) != len(outputs):
        raise ModelError("outputs must not contain duplicate signal expressions")
    harmonics = _parse_harmonics(
        analysis.get("harmonics") if isinstance(analysis, dict) else None,
        parameters,
        transient,
        outputs,
    )

    forced_gate_windows = parse_forced_gate_windows(
        data.get("perturbations"),
        set(controls) | set(sampled_controls),
    )
    for window in forced_gate_windows:
        if not transient.start <= window.from_s < transient.stop:
            raise ModelError(
                f"perturbations: forced window on {window.control_id!r} starts "
                "outside the analysis window"
            )

    loss_analysis = parse_loss_sections(
        data.get("device_cards"),
        data.get("cooling"),
        data.get("loss_analysis"),
    )
    if loss_analysis is not None:
        for binding in loss_analysis.devices:
            target = components_by_id.get(binding.component_id)
            if target is None or target.type not in {
                "ideal_switch",
                "diode",
                "thyristor",
            }:
                raise ModelError(
                    f"loss_analysis.devices.{binding.component_id} must reference "
                    "an ideal_switch, diode or thyristor component"
                )
            if not target.is_ideal_conductor:
                # The card's conduction table and the in-circuit drop describe
                # the SAME loss; keeping both would count it twice.
                raise ModelError(
                    f"loss_analysis.devices.{binding.component_id}: the component "
                    "declares on_resistance/forward_voltage, which already models "
                    "its conduction loss in the circuit; a loss card would count it "
                    "twice — use one or the other"
                )
            card = loss_analysis.cards[binding.card_id]
            if (card.kind == "switch") != (target.type == "ideal_switch"):
                raise ModelError(
                    f"loss_analysis.devices.{binding.component_id}: card kind "
                    f"{card.kind!r} does not match component type {target.type!r}"
                )
            required = [f"i({binding.component_id})"]
            if binding.commutation_voltage is not None:
                required.append(binding.commutation_voltage)
            for expression in required:
                if expression not in outputs:
                    raise ModelError(
                        f"loss_analysis.devices.{binding.component_id}: "
                        f"{expression!r} must be listed in outputs so the loss "
                        "estimate works from recorded, auditable signals"
                    )

    for control in sampled_controls.values():
        if control.kind == "sampled_pi_pwm":
            sense_expressions: tuple[str, ...] = (control.sense,)
        elif control.kind in {"signal_pwm", "signal_svpwm"}:
            # This control senses nothing itself; its duty comes from the
            # signal layer, whose probes are held to the same rule below.
            sense_expressions = ()
        else:
            sense_expressions = (*control.sense_currents, *control.sense_voltages)
        for expression in sense_expressions:
            if expression not in outputs:
                raise ModelError(
                    f"control {control.id}: sense {expression!r} must be listed "
                    "in outputs so the sampled value is a recorded, auditable "
                    "signal"
                )

    if signal_graph is not None:
        # Same rule the sampled controls obey: what a controller SAMPLES must
        # be a recorded signal, or the run cannot be audited against what the
        # controller actually saw.
        for expression in signal_graph.probe_expressions():
            if expression not in outputs:
                raise ModelError(
                    f"signals: probe reads {expression!r}, which must be listed "
                    "in outputs so the sampled value is a recorded, auditable "
                    "signal"
                )

    return Model(
        schema=schema,
        name=name,
        source_path=source_path,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        parameters=parameters,
        ground=ground,
        components=tuple(components),
        controls=controls,
        transient=transient,
        outputs=outputs,
        harmonics=harmonics,
        sampled_controls=sampled_controls,
        couplings=tuple(couplings),
        loss_analysis=loss_analysis,
        machines=machines,
        signals=signal_graph,
        parameter_envelopes=parameter_envelopes,
        forced_gate_windows=forced_gate_windows,
        subcircuit_instances=subcircuit_instances,
    )
