# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Declared subcircuits and their compile-time expansion.

A subcircuit is a named, self-contained unit: explicit ``ports``, an explicit
``parameters`` list, and a body of ordinary components plus optional
couplings.  Expansion is deterministic, source-to-source, and happens BEFORE
semantic validation, so every derived component is checked by exactly the
same rules as a hand-written one — the machine-expansion precedent.
Hierarchy is presentation, never physics: the canonical IR and every hash
are functions of the flattened model only.

NAMESPACING.  Derived ids and internal nodes are ``<instance>.<inner>``.
The separator is ``.`` and no segment (definition name, port, parameter,
instance id, inner component id, inner node name) may contain ``.`` or
``/`` — which keeps the convention 1:1 mappable onto a KiCad hierarchical
sheet path (``/``-separated) — an interop property that is cheap to keep now
and expensive to retrofit later.

SELF-CONTAINMENT.  A ``${ref}`` inside a definition body must name a
declared subcircuit parameter, never a global model parameter: a definition
that silently depends on its surroundings is not a reusable unit.  Checked
at definition parse time, fail closed, naming the definition.  Bindings at
the instance MAY be ``${global}`` references — they are substituted
verbatim and resolved later against the model's parameters, with the usual
dimension checks, under the derived component's namespaced id.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from pear_sim.errors import ModelError
from pear_sim.spec.controls import _reject_unknown, _require_model_text

SEPARATOR = "."
MAX_SUBCIRCUIT_DEPTH = 8
MAX_EXPANDED_COMPONENTS = 4096

_PARAMETER_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_REFERENCE_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
_OUTPUT_TARGET_RE = re.compile(r"^([vi])\(([A-Za-z_][A-Za-z0-9_.:-]*)\)$")


@dataclass(frozen=True)
class SubcircuitDefinition:
    """One parsed ``subcircuits:`` entry, body kept as raw records."""

    name: str
    ports: tuple[str, ...]
    parameters: tuple[str, ...]
    components: tuple[dict[str, Any], ...]
    couplings: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SubcircuitInstance:
    """Presentation record of one expanded instance.

    ``path`` is the flattened instance path (``y``, or ``psu.leg1`` for a
    nested instance).  ``members`` lists every flattened component id the
    instance produced, including those of nested instances — exactly the
    set a renderer must hide to draw the instance as a box.  This record
    never enters ``physics_model_dict``: hierarchy is presentation.
    """

    path: str
    ref: str
    ports: tuple[tuple[str, str], ...]
    members: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ref": self.ref,
            "ports": {port: net for port, net in self.ports},
            "members": list(self.members),
        }


def _require_segment(value: Any, *, name: str) -> str:
    text = _require_model_text(value, name=name)
    if SEPARATOR in text or "/" in text:
        raise ModelError(
            f"{name}: {text!r} may not contain {SEPARATOR!r} or '/' — segment "
            "names must stay 1:1 mappable onto a KiCad hierarchical sheet path"
        )
    return text


def _collect_references(value: Any, refs: set[str]) -> None:
    if isinstance(value, str):
        match = _REFERENCE_RE.match(value.strip())
        if match:
            refs.add(match.group(1))
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_references(item, refs)
        return
    if isinstance(value, list):
        for item in value:
            _collect_references(item, refs)


def parse_subcircuits(raw: Any) -> dict[str, SubcircuitDefinition]:
    """Parse the top-level ``subcircuits:`` section, fail closed.

    Body components are validated only structurally here (they must be
    objects with a local id); every semantic rule runs after expansion, on
    the flattened records, under their namespaced ids.
    """

    if raw is None:
        return {}
    if not isinstance(raw, dict) or not raw:
        raise ModelError("subcircuits must be a non-empty object")
    definitions: dict[str, SubcircuitDefinition] = {}
    for raw_name, raw_definition in raw.items():
        name = _require_segment(raw_name, name="subcircuits key")
        prefix = f"subcircuits.{name}"
        if not isinstance(raw_definition, dict):
            raise ModelError(f"{prefix} must be an object")
        _reject_unknown(
            raw_definition,
            {"ports", "parameters", "components", "couplings"},
            name=prefix,
        )

        raw_ports = raw_definition.get("ports")
        if not isinstance(raw_ports, list) or not raw_ports:
            raise ModelError(f"{prefix}.ports must be a non-empty list")
        ports: list[str] = []
        for index, raw_port in enumerate(raw_ports):
            port = _require_segment(raw_port, name=f"{prefix}.ports[{index}]")
            if port in ports:
                raise ModelError(f"{prefix}.ports: duplicate port {port!r}")
            ports.append(port)

        raw_parameters = raw_definition.get("parameters", [])
        if not isinstance(raw_parameters, list):
            raise ModelError(f"{prefix}.parameters must be a list of names")
        parameters: list[str] = []
        for index, raw_parameter in enumerate(raw_parameters):
            parameter = _require_model_text(
                raw_parameter, name=f"{prefix}.parameters[{index}]"
            )
            if _PARAMETER_NAME_RE.fullmatch(parameter) is None:
                raise ModelError(
                    f"{prefix}.parameters[{index}] must match "
                    f"{_PARAMETER_NAME_RE.pattern!r}"
                )
            if parameter in parameters:
                raise ModelError(
                    f"{prefix}.parameters: duplicate parameter {parameter!r}"
                )
            parameters.append(parameter)

        raw_components = raw_definition.get("components")
        if not isinstance(raw_components, list) or not raw_components:
            raise ModelError(f"{prefix}.components must be a non-empty list")
        component_ids: list[str] = []
        for index, raw_component in enumerate(raw_components):
            if not isinstance(raw_component, dict):
                raise ModelError(f"{prefix}.components[{index}] must be an object")
            inner_id = _require_segment(
                raw_component.get("id"), name=f"{prefix}.components[{index}].id"
            )
            if inner_id in component_ids:
                raise ModelError(
                    f"{prefix}.components: duplicate id {inner_id!r}"
                )
            if inner_id in ports:
                raise ModelError(
                    f"{prefix}: component id {inner_id!r} collides with a port "
                    "name — the namespaced reference would be ambiguous"
                )
            component_ids.append(inner_id)

        raw_couplings = raw_definition.get("couplings", [])
        if not isinstance(raw_couplings, list):
            raise ModelError(f"{prefix}.couplings must be a list")
        for index, raw_coupling in enumerate(raw_couplings):
            if not isinstance(raw_coupling, dict):
                raise ModelError(f"{prefix}.couplings[{index}] must be an object")

        # Self-containment: every ${ref} in the body names a declared
        # subcircuit parameter.  A declared parameter nothing references is
        # equally refused — an unused knob invites silent drift between what
        # a definition promises and what it does.
        referenced: set[str] = set()
        for raw_component in raw_components:
            _collect_references(raw_component, referenced)
        for raw_coupling in raw_couplings:
            _collect_references(raw_coupling, referenced)
        undeclared = sorted(referenced - set(parameters))
        if undeclared:
            raise ModelError(
                f"{prefix} references undeclared parameter(s): "
                + ", ".join(undeclared)
                + " — a subcircuit body may only use its own declared parameters"
            )
        unused = sorted(set(parameters) - referenced)
        if unused:
            raise ModelError(
                f"{prefix} declares parameter(s) never referenced by its body: "
                + ", ".join(unused)
            )

        definitions[name] = SubcircuitDefinition(
            name=name,
            ports=tuple(ports),
            parameters=tuple(parameters),
            components=tuple(raw_components),
            couplings=tuple(raw_couplings),
        )
    return definitions


def _substitute(value: Any, bindings: dict[str, Any]) -> Any:
    if isinstance(value, str):
        match = _REFERENCE_RE.match(value.strip())
        if match and match.group(1) in bindings:
            return bindings[match.group(1)]
        return value
    if isinstance(value, dict):
        return {key: _substitute(item, bindings) for key, item in value.items()}
    if isinstance(value, list):
        return [_substitute(item, bindings) for item in value]
    return value


class _Expansion:
    """Mutable state of one document's expansion pass."""

    def __init__(
        self, definitions: dict[str, SubcircuitDefinition], ground: str
    ) -> None:
        self.definitions = definitions
        self.ground = ground
        self.components: list[dict[str, Any]] = []
        self.couplings: list[dict[str, Any]] = []
        self.instances: list[SubcircuitInstance] = []
        self.port_bindings: dict[str, str] = {}

    def expand_instance(
        self,
        record: dict[str, Any],
        *,
        prefix: str,
        stack: tuple[str, ...],
    ) -> tuple[str, ...]:
        _reject_unknown(
            record,
            {"id", "type", "ref", "ports", "params"},
            name=f"subcircuit instance {prefix or '(top level)'}",
        )
        instance_id = _require_segment(
            record.get("id"), name="subcircuit instance id"
        )
        path = f"{prefix}{instance_id}"
        ref = _require_model_text(record.get("ref"), name=f"{path}.ref")
        definition = self.definitions.get(ref)
        if definition is None:
            raise ModelError(
                f"instance {path!r}: unknown subcircuit {ref!r} — declare it "
                "under top-level subcircuits"
            )
        if ref in stack:
            cycle = " -> ".join((*stack, ref))
            raise ModelError(
                f"instance {path!r}: recursive subcircuit reference ({cycle})"
            )
        if len(stack) >= MAX_SUBCIRCUIT_DEPTH:
            raise ModelError(
                f"instance {path!r} exceeds the nesting depth limit of "
                f"{MAX_SUBCIRCUIT_DEPTH}"
            )

        raw_ports = record.get("ports")
        if not isinstance(raw_ports, dict):
            raise ModelError(f"instance {path!r}: ports must be an object")
        declared_ports = set(definition.ports)
        bound_ports = set(raw_ports)
        unbound = sorted(declared_ports - bound_ports)
        if unbound:
            raise ModelError(
                f"instance {path!r}: unbound port(s): " + ", ".join(unbound)
            )
        extra = sorted(bound_ports - declared_ports)
        if extra:
            raise ModelError(
                f"instance {path!r}: unknown port(s): " + ", ".join(extra)
            )
        port_map = {
            port: _require_model_text(
                raw_ports[port], name=f"instance {path!r} port {port!r}"
            )
            for port in definition.ports
        }

        raw_params = record.get("params")
        declared_parameters = set(definition.parameters)
        if raw_params is None:
            raw_params = {}
        if not isinstance(raw_params, dict):
            raise ModelError(f"instance {path!r}: params must be an object")
        missing = sorted(declared_parameters - set(raw_params))
        if missing:
            raise ModelError(
                f"instance {path!r}: unbound parameter(s): " + ", ".join(missing)
            )
        unknown = sorted(set(raw_params) - declared_parameters)
        if unknown:
            raise ModelError(
                f"instance {path!r}: unknown parameter(s): " + ", ".join(unknown)
            )
        bindings: dict[str, Any] = {}
        for parameter in definition.parameters:
            value = raw_params[parameter]
            if isinstance(value, (dict, list)):
                raise ModelError(
                    f"instance {path!r}: parameter {parameter!r} must be a "
                    "scalar (a quantity, a number, or a ${reference})"
                )
            bindings[parameter] = value

        def map_node(raw_node: Any, *, context: str) -> str:
            node = _require_model_text(raw_node, name=context)
            if node in declared_ports:
                return port_map[node]
            if SEPARATOR in node or "/" in node:
                raise ModelError(
                    f"{context}: internal node {node!r} may not contain "
                    f"{SEPARATOR!r} or '/'"
                )
            if node == self.ground:
                raise ModelError(
                    f"instance {path!r}: internal node {node!r} shadows "
                    "circuit.ground — a subcircuit reaches the outside only "
                    "through its ports; bind a port to the ground net instead"
                )
            return f"{path}{SEPARATOR}{node}"

        members: list[str] = []
        for raw_component in definition.components:
            expanded = _substitute(dict(raw_component), bindings)
            if expanded.get("type") == "subcircuit":
                nested_ports = expanded.get("ports")
                if not isinstance(nested_ports, dict):
                    raise ModelError(
                        f"instance {path}.{expanded.get('id')!r}: ports must "
                        "be an object"
                    )
                expanded["ports"] = {
                    port: map_node(
                        net,
                        context=(
                            f"instance {path!r} nested instance "
                            f"{expanded.get('id')!r} port {port!r}"
                        ),
                    )
                    for port, net in nested_ports.items()
                }
                members.extend(
                    self.expand_instance(
                        expanded,
                        prefix=f"{path}{SEPARATOR}",
                        stack=(*stack, ref),
                    )
                )
                continue
            inner_id = expanded.get("id")
            derived_id = f"{path}{SEPARATOR}{inner_id}"
            expanded["id"] = derived_id
            expanded["p"] = map_node(
                expanded.get("p"), context=f"{derived_id}.p"
            )
            expanded["n"] = map_node(
                expanded.get("n"), context=f"{derived_id}.n"
            )
            if len(self.components) >= MAX_EXPANDED_COMPONENTS:
                raise ModelError(
                    f"instance {path!r}: expansion exceeds the "
                    f"{MAX_EXPANDED_COMPONENTS}-component limit"
                )
            self.components.append(expanded)
            members.append(derived_id)

        for index, raw_coupling in enumerate(definition.couplings):
            expanded = _substitute(dict(raw_coupling), bindings)
            raw_pair = expanded.get("inductors")
            if not isinstance(raw_pair, list) or len(raw_pair) != 2:
                raise ModelError(
                    f"instance {path!r} coupling [{index}]: inductors must "
                    "list exactly two local inductor ids"
                )
            expanded["inductors"] = [
                f"{path}{SEPARATOR}"
                + _require_segment(
                    raw_id, name=f"instance {path!r} coupling [{index}]"
                )
                for raw_id in raw_pair
            ]
            self.couplings.append(expanded)

        for port in definition.ports:
            self.port_bindings[f"{path}{SEPARATOR}{port}"] = port_map[port]
        self.instances.append(
            SubcircuitInstance(
                path=path,
                ref=ref,
                ports=tuple((port, port_map[port]) for port in definition.ports),
                members=tuple(members),
            )
        )
        return tuple(members)


def _rewrite_outputs(
    raw_outputs: Any, port_bindings: dict[str, str]
) -> Any:
    """Resolve ``v(<instance>.<port>)`` to the bound net.

    ``i(<instance>.<port>)`` is refused: a port is a net, not a branch, so a
    port current does not exist — name the member branch instead.
    """

    if not isinstance(raw_outputs, list):
        return raw_outputs
    rewritten: list[Any] = []
    for raw_output in raw_outputs:
        if not isinstance(raw_output, str):
            rewritten.append(raw_output)
            continue
        match = _OUTPUT_TARGET_RE.match(raw_output.strip())
        if match is None:
            rewritten.append(raw_output)
            continue
        kind, target = match.groups()
        bound = port_bindings.get(target)
        if bound is None:
            rewritten.append(raw_output)
            continue
        if kind == "i":
            raise ModelError(
                f"output {raw_output!r}: {target!r} is a subcircuit port — a "
                "port is a net, not a branch; request i() of a member "
                "component instead"
            )
        rewritten.append(f"v({bound})")
    return rewritten


def expand_subcircuits(
    data: dict[str, Any],
) -> tuple[dict[str, Any], tuple[SubcircuitInstance, ...]]:
    """Expand every declared subcircuit instance, source-to-source.

    Returns the expanded document (unchanged object when nothing is
    declared) plus the presentation records of every instance, outermost
    first.  All structural refusals here name the offending definition or
    instance; everything semantic (dimensions, topology, controls) is left
    to the ordinary validation of the flattened records.
    """

    definitions = parse_subcircuits(data.get("subcircuits"))
    circuit = data.get("circuit")
    raw_components = (
        circuit.get("components") if isinstance(circuit, dict) else None
    )
    has_instances = isinstance(raw_components, list) and any(
        isinstance(record, dict) and record.get("type") == "subcircuit"
        for record in raw_components
    )
    if not definitions and not has_instances:
        return data, ()
    if not has_instances:
        raise ModelError(
            "subcircuits are declared but never instantiated — remove the "
            "section or add a type: subcircuit component"
        )
    if not isinstance(circuit, dict) or not isinstance(raw_components, list):
        raise ModelError("circuit.components must be a non-empty list")

    ground = circuit.get("ground", "gnd")
    if not isinstance(ground, str):
        ground = "gnd"
    expansion = _Expansion(definitions, ground)
    for record in raw_components:
        if isinstance(record, dict) and record.get("type") == "subcircuit":
            expansion.expand_instance(record, prefix="", stack=())
        else:
            expansion.components.append(record)

    new_circuit = dict(circuit)
    new_circuit["components"] = expansion.components
    if expansion.couplings:
        raw_couplings = circuit.get("couplings", [])
        if not isinstance(raw_couplings, list):
            raise ModelError("circuit.couplings must be a list")
        new_circuit["couplings"] = [*raw_couplings, *expansion.couplings]

    new_data = dict(data)
    new_data.pop("subcircuits", None)
    new_data["circuit"] = new_circuit
    new_data["outputs"] = _rewrite_outputs(
        data.get("outputs"), expansion.port_bindings
    )
    return new_data, tuple(expansion.instances)
