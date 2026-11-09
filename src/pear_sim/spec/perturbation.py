# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Hook B: declared events injected into a run, without editing the netlist.

An event is a property of the *experiment*, not of the circuit, so it is
declared in a request file and never by hand-editing a model. Applying one is a
**source-to-source transform**: the request plus the original model document
produce a new, complete, readable model document that is then simulated by the
ordinary unmodified pipeline. Nothing about the run is special-cased downstream,
the materialized source is a retained artifact a human can read, and its
``source_sha256`` honestly hashes what was actually simulated.

Three event types, each with a deliberately narrow contract:

``analog_transient``
    A double-exponential current injected at a node. Materializes as a
    ``current_source`` between the target node and ground. No core support
    needed — the waveform machinery already exists.

``step``
    A step current onset at a node: the shape a sudden leakage or latch-up
    current takes. Materializes as a ``current_source`` with a step waveform.

``pulse``
    A forced level on a gate for a window: the shape a strike on a comparator
    or gate driver takes. This one **cannot** be expressed as a source, so it
    materializes into a ``perturbations.forced_gate_windows`` section that the
    solver honours as an override mask.

What is deliberately NOT supported, and why it fails closed rather than
approximating: a ``step`` on an arbitrary model *parameter*. Element values
(R, L, C) are stamped into the MNA matrices once per compiled mode; only
sources are functions of time. A mid-run change of a resistance is therefore
not a thing this kernel can represent, and pretending otherwise by, say,
restarting with a new value would silently change the meaning of every metric
that integrates across the discontinuity.

Nothing here estimates whether an amplitude, width or drift is physically
correct for any device. Those numbers are inputs, taken from test data.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from typing import TYPE_CHECKING, Any, Mapping

from pear_sim.errors import ModelError
from pear_sim.spec.units import format_si, parse_quantity

if TYPE_CHECKING:  # the loader imports this module; keep the edge one-way
    from pear_sim.spec.model import Model


__all__ = [
    "REQUEST_SCHEMA",
    "ForcedGateWindow",
    "PerturbationError",
    "PerturbationEvent",
    "PerturbationSpec",
    "apply_perturbation_to_document",
    "parse_forced_gate_windows",
    "parse_perturbation_request",
]

REQUEST_SCHEMA = "pear-sim/perturbation-request/v0.1"

_EVENT_KINDS = ("pulse", "analog_transient", "step")
_MAX_EVENTS = 64
_INJECTED_PREFIX = "_pert_"


class PerturbationError(ModelError):
    """Raised for an invalid or unsupported perturbation declaration."""


@dataclass(frozen=True)
class ForcedGateWindow:
    """A gate held at ``level`` over ``[from_s, to_s)``, overriding its schedule.

    The window is an override *mask*, not an extra pair of edges: it wins over
    whatever the schedule says, including a duty that a closed-loop controller
    computes dynamically during the run. Outside the window the gate returns to
    its natural value with no memory of the override.
    """

    control_id: str
    from_s: float
    to_s: float
    level: bool

    def contains(self, time: float) -> bool:
        return self.from_s <= time < self.to_s

    def as_dict(self) -> dict[str, Any]:
        return {
            "control": self.control_id,
            "from_s": self.from_s,
            "to_s": self.to_s,
            "level": "on" if self.level else "off",
        }


@dataclass(frozen=True)
class PerturbationEvent:
    """One declared event. ``at_s`` and ``phase`` are mutually exclusive."""

    id: str
    kind: str
    target_kind: str  # "node" | "control"
    target: str
    at_s: float | None = None
    phase: float | None = None
    phase_reference: str | None = None
    phase_anchor_s: float | None = None
    # analog_transient / step
    amplitude_a: float | None = None
    tau_rise_s: float | None = None
    tau_fall_s: float | None = None
    # pulse
    width_s: float | None = None
    level: bool | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "type": self.kind,
            "target": f"{self.target_kind}:{self.target}",
        }
        for name in (
            "at_s",
            "phase",
            "phase_reference",
            "phase_anchor_s",
            "amplitude_a",
            "tau_rise_s",
            "tau_fall_s",
            "width_s",
        ):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        if self.level is not None:
            payload["level"] = "on" if self.level else "off"
        return payload


@dataclass(frozen=True)
class PerturbationSpec:
    """A named set of events plus the parameter overrides they run alongside."""

    label: str
    events: tuple[PerturbationEvent, ...] = ()
    parameter_overrides_si: tuple[tuple[str, float], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"label": self.label}
        if self.events:
            payload["events"] = [event.as_dict() for event in self.events]
        if self.parameter_overrides_si:
            payload["parameter_overrides_si"] = dict(self.parameter_overrides_si)
        return payload


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], *, name: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise PerturbationError(
            f"{name}: unknown field(s) " + ", ".join(sorted(unknown))
        )


def _quantity(raw: Any, dimension: str, *, name: str) -> float:
    return float(parse_quantity(raw, dimension, name=name).value)


def _positive_time(raw: Any, *, name: str) -> float:
    value = _quantity(raw, "time", name=name)
    if not math.isfinite(value) or value <= 0.0:
        raise PerturbationError(f"{name} must be a finite positive time")
    return value


def _parse_target(raw: Any, *, name: str) -> tuple[str, str]:
    if not isinstance(raw, str) or ":" not in raw:
        raise PerturbationError(
            f"{name}.target must read 'node:<name>' or 'control:<id>'"
        )
    kind, _, value = raw.partition(":")
    if kind == "parameter":
        raise PerturbationError(
            f"{name}.target: a perturbation of a model parameter is not "
            "supported. Element values are stamped into the compiled mode once "
            "and only sources vary with time, so a mid-run change of R/L/C "
            "cannot be represented. Model a parameter change as a separate run "
            "with the changed value instead."
        )
    if kind not in ("node", "control") or not value:
        raise PerturbationError(
            f"{name}.target must read 'node:<name>' or 'control:<id>' (got {raw!r})"
        )
    return kind, value


def _parse_timing(
    raw: Mapping[str, Any], *, name: str
) -> tuple[float | None, float | None, str | None, float | None]:
    has_at = "at" in raw
    has_phase = "phase" in raw
    if has_at == has_phase:
        raise PerturbationError(
            f"{name}: declare exactly one of 'at' (absolute time) and 'phase' "
            "(fraction of the switching period)"
        )
    if has_at:
        at_s = _quantity(raw["at"], "time", name=f"{name}.at")
        if not math.isfinite(at_s) or at_s < 0.0:
            raise PerturbationError(f"{name}.at must be finite and non-negative")
        return at_s, None, None, None
    phase = raw["phase"]
    if isinstance(phase, bool) or not isinstance(phase, (int, float)):
        raise PerturbationError(f"{name}.phase must be a number in [0, 1)")
    phase = float(phase)
    if not 0.0 <= phase < 1.0:
        raise PerturbationError(f"{name}.phase must lie in [0, 1)")
    reference = raw.get("phase_reference")
    if reference is not None and not isinstance(reference, str):
        raise PerturbationError(f"{name}.phase_reference must be a control id")
    if "phase_anchor" not in raw:
        # Required, not defaulted. A phase alone says WHERE IN THE CYCLE, never
        # WHICH CYCLE, and the two sensible guesses are both wrong: the first
        # cycle lands in start-up, the last leaves no window to observe the
        # response in. The caller states the settled instant to anchor on.
        raise PerturbationError(
            f"{name}: 'phase' also needs 'phase_anchor' — the settled time to "
            "anchor the cycle on. Phase says where in a cycle, not which cycle."
        )
    anchor = _quantity(raw["phase_anchor"], "time", name=f"{name}.phase_anchor")
    if not math.isfinite(anchor) or anchor < 0.0:
        raise PerturbationError(f"{name}.phase_anchor must be finite and non-negative")
    return None, phase, reference, anchor


def _parse_event(raw: Any, index: int) -> PerturbationEvent:
    name = f"events[{index}]"
    if not isinstance(raw, dict):
        raise PerturbationError(f"{name} must be an object")
    kind = raw.get("type")
    if kind not in _EVENT_KINDS:
        raise PerturbationError(
            f"{name}.type must be one of " + ", ".join(_EVENT_KINDS) + f" (got {kind!r})"
        )
    event_id = raw.get("id")
    if not isinstance(event_id, str) or not event_id.replace("_", "").isalnum():
        raise PerturbationError(f"{name}.id must be an alphanumeric/underscore string")

    common = {"id", "type", "target", "at", "phase", "phase_reference", "phase_anchor"}
    if kind == "analog_transient":
        _reject_unknown(
            raw, common | {"amplitude", "tau_rise", "tau_fall"}, name=name
        )
    elif kind == "step":
        _reject_unknown(raw, common | {"amplitude"}, name=name)
    else:
        _reject_unknown(raw, common | {"width", "level"}, name=name)

    target_kind, target = _parse_target(raw.get("target"), name=name)
    at_s, phase, phase_reference, phase_anchor = _parse_timing(raw, name=name)

    if kind in ("analog_transient", "step"):
        if target_kind != "node":
            raise PerturbationError(
                f"{name}: {kind} injects current and must target a node"
            )
        amplitude = _quantity(raw.get("amplitude"), "current", name=f"{name}.amplitude")
        if not math.isfinite(amplitude) or amplitude == 0.0:
            raise PerturbationError(f"{name}.amplitude must be finite and non-zero")
        if kind == "step":
            return PerturbationEvent(
                id=event_id, kind=kind, target_kind=target_kind, target=target,
                at_s=at_s, phase=phase, phase_reference=phase_reference,
                phase_anchor_s=phase_anchor,
                amplitude_a=amplitude,
            )
        tau_rise = _positive_time(raw.get("tau_rise"), name=f"{name}.tau_rise")
        tau_fall = _positive_time(raw.get("tau_fall"), name=f"{name}.tau_fall")
        if tau_fall <= tau_rise:
            raise PerturbationError(
                f"{name}: tau_fall must strictly exceed tau_rise"
            )
        return PerturbationEvent(
            id=event_id, kind=kind, target_kind=target_kind, target=target,
            at_s=at_s, phase=phase, phase_reference=phase_reference,
                phase_anchor_s=phase_anchor,
            amplitude_a=amplitude, tau_rise_s=tau_rise, tau_fall_s=tau_fall,
        )

    if target_kind != "control":
        raise PerturbationError(f"{name}: pulse forces a gate and must target a control")
    width_s = _positive_time(raw.get("width"), name=f"{name}.width")
    level = raw.get("level")
    if level not in ("on", "off"):
        raise PerturbationError(f"{name}.level must be 'on' or 'off'")
    return PerturbationEvent(
        id=event_id, kind=kind, target_kind=target_kind, target=target,
        at_s=at_s, phase=phase, phase_reference=phase_reference,
                phase_anchor_s=phase_anchor,
        width_s=width_s, level=level == "on",
    )


def parse_perturbation_request(payload: Any) -> tuple[PerturbationEvent, ...]:
    """Validate the ``events`` block of a perturbation request."""

    if not isinstance(payload, dict):
        raise PerturbationError("perturbation request must be an object")
    if payload.get("schema") != REQUEST_SCHEMA:
        raise PerturbationError(
            f"perturbation request schema must be {REQUEST_SCHEMA!r}"
        )
    raw_events = payload.get("events")
    if not isinstance(raw_events, list) or not 1 <= len(raw_events) <= _MAX_EVENTS:
        raise PerturbationError(f"events must list 1..{_MAX_EVENTS} entries")
    events = tuple(_parse_event(raw, index) for index, raw in enumerate(raw_events))
    seen: set[str] = set()
    for event in events:
        if event.id in seen:
            raise PerturbationError(f"duplicate event id {event.id!r}")
        seen.add(event.id)
    return events


def parse_forced_gate_windows(
    raw: Any, control_ids: set[str], *, name: str = "perturbations"
) -> tuple[ForcedGateWindow, ...]:
    """Parse the materialized ``perturbations.forced_gate_windows`` section."""

    if raw is None:
        return ()
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    _reject_unknown(raw, {"forced_gate_windows"}, name=name)
    entries = raw.get("forced_gate_windows")
    if entries is None:
        return ()
    if not isinstance(entries, list) or not 1 <= len(entries) <= _MAX_EVENTS:
        raise ModelError(f"{name}.forced_gate_windows must list 1..{_MAX_EVENTS} entries")
    windows: list[ForcedGateWindow] = []
    for index, entry in enumerate(entries):
        label = f"{name}.forced_gate_windows[{index}]"
        if not isinstance(entry, dict):
            raise ModelError(f"{label} must be an object")
        _reject_unknown(entry, {"control", "from", "to", "level"}, name=label)
        control_id = entry.get("control")
        if control_id not in control_ids:
            raise ModelError(
                f"{label}.control {control_id!r} is not a declared control"
            )
        from_s = _quantity(entry.get("from"), "time", name=f"{label}.from")
        to_s = _quantity(entry.get("to"), "time", name=f"{label}.to")
        if not math.isfinite(from_s) or from_s < 0.0:
            raise ModelError(f"{label}.from must be finite and non-negative")
        if not to_s > from_s:
            raise ModelError(f"{label}.to must strictly exceed {label}.from")
        level = entry.get("level")
        if level not in ("on", "off"):
            raise ModelError(f"{label}.level must be 'on' or 'off'")
        windows.append(
            ForcedGateWindow(
                control_id=str(control_id), from_s=from_s, to_s=to_s,
                level=level == "on",
            )
        )
    ordered = tuple(sorted(windows, key=lambda w: (w.from_s, w.control_id)))
    for first, second in zip(ordered, ordered[1:]):
        if first.control_id == second.control_id and second.from_s < first.to_s:
            raise ModelError(
                f"{name}.forced_gate_windows: overlapping windows on control "
                f"{first.control_id!r} — a gate has one forced level at a time"
            )
    return ordered


# --------------------------------------------------------------------------
# source-to-source materialization
# --------------------------------------------------------------------------


def _resolve_time(event: PerturbationEvent, model: Model) -> float:
    """Absolute onset time, resolving a phase against the reference control."""

    if event.at_s is not None:
        return event.at_s
    reference = event.phase_reference or (
        event.target if event.target_kind == "control" else None
    )
    if reference is None:
        raise PerturbationError(
            f"event {event.id!r}: 'phase' needs a switching period — declare "
            "phase_reference, or target a control"
        )
    period: float | None = None
    control = model.controls.get(reference)
    if control is not None:
        period = control.period
    else:
        sampled = model.sampled_controls.get(reference)
        if sampled is not None:
            period = sampled.period
    if period is None:
        raise PerturbationError(
            f"event {event.id!r}: phase_reference {reference!r} is not a control"
        )
    # The cycle is the one that begins at or after the DECLARED anchor; the
    # phase then places the event inside that cycle.
    stop = model.transient.stop
    start = model.transient.start
    anchor_s = float(event.phase_anchor_s or 0.0)
    if not start <= anchor_s < stop:
        raise PerturbationError(
            f"event {event.id!r}: phase_anchor lies outside the analysis window"
        )
    cycle = math.ceil((anchor_s - start) / period - 1e-12)
    onset = start + cycle * period + float(event.phase) * period
    if onset >= stop:
        raise PerturbationError(
            f"event {event.id!r}: the anchored cycle ends after the analysis "
            "window; move phase_anchor earlier or extend the window"
        )
    return onset


def _node_names(document: Mapping[str, Any]) -> set[str]:
    components = document.get("circuit", {}).get("components", [])
    names: set[str] = set()
    for component in components:
        if isinstance(component, dict):
            for key in ("p", "n"):
                value = component.get(key)
                if isinstance(value, str):
                    names.add(value)
    return names


def apply_perturbation_to_document(
    document: Mapping[str, Any],
    spec: PerturbationSpec,
    model: Model,
) -> dict[str, Any]:
    """Return a new model document with ``spec`` materialized into it.

    The result is a complete, ordinary model: current sources for the injected
    node events, a ``perturbations`` section for the forced gate windows, and
    parameter overrides written back into ``parameters``. It is simulated by
    the unmodified pipeline.
    """

    result = copy.deepcopy(dict(document))
    ground = result.get("circuit", {}).get("ground", "gnd")
    nodes = _node_names(result)
    components = result.setdefault("circuit", {}).setdefault("components", [])
    windows: list[dict[str, Any]] = []

    for parameter, value_si in spec.parameter_overrides_si:
        parameters = result.setdefault("parameters", {})
        if parameter not in parameters:
            raise PerturbationError(
                f"override targets unknown parameter {parameter!r}"
            )
        existing = parameters[parameter]
        quantity = model.parameters.get(parameter)
        if quantity is None:
            raise PerturbationError(
                f"override targets unknown parameter {parameter!r}"
            )
        # Written back WITH its unit. A bare float would resolve as
        # dimensionless and the component that consumes it would reject the
        # generated source — the dimension has to survive the round trip.
        rendered = format_si(float(value_si), quantity.dimension)
        # An envelope keeps its declaration; only the nominal moves, so the
        # materialized source still names where the drift figure came from.
        if isinstance(existing, dict):
            existing["nominal"] = rendered
        else:
            parameters[parameter] = rendered

    for event in spec.events:
        onset = _resolve_time(event, model)
        if event.kind == "pulse":
            if event.target not in model.controls and event.target not in model.sampled_controls:
                raise PerturbationError(
                    f"event {event.id!r}: control {event.target!r} is not declared"
                )
            windows.append(
                {
                    "control": event.target,
                    "from": float(onset),
                    "to": float(onset + float(event.width_s or 0.0)),
                    "level": "on" if event.level else "off",
                }
            )
            continue
        if event.target not in nodes:
            raise PerturbationError(
                f"event {event.id!r}: node {event.target!r} does not exist in the circuit"
            )
        if event.target == ground:
            raise PerturbationError(
                f"event {event.id!r}: injecting into the ground node is a no-op"
            )
        component: dict[str, Any] = {
            "id": f"{_INJECTED_PREFIX}{event.id}",
            "type": "current_source",
            "p": event.target,
            "n": ground,
        }
        if event.kind == "analog_transient":
            component["waveform"] = {
                "type": "double_exponential",
                "amplitude": float(event.amplitude_a or 0.0),
                "tau_rise": float(event.tau_rise_s or 0.0),
                "tau_fall": float(event.tau_fall_s or 0.0),
                "at": float(onset),
            }
        else:
            component["waveform"] = {
                "type": "step",
                "initial": 0.0,
                "final": float(event.amplitude_a or 0.0),
                "at": float(onset),
            }
        components.append(component)

    if windows:
        result["perturbations"] = {"forced_gate_windows": windows}
    return result
