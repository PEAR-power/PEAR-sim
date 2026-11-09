# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Control-block parsing and gate expansion for the v0.1 model schema.

Extracted verbatim from ``pear_sim.spec.model.load_model`` (restructuring stage 2,
move 9).  ``parse_controls`` validates the raw ``controls`` mapping and
expands ``pwm``, ``complementary_pwm``, and ``three_phase_spwm`` sources
into the concrete per-gate :class:`PwmControl` records the scheduler
executes.  The shared model-text/parameter primitives live here so both
this module and the loader can use them without an import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from pear_sim.errors import ModelError
from pear_sim.limits import (
    MAX_MODEL_GATE_OUTPUTS,
    MAX_MODEL_LOGICAL_CONTROLS,
    MAX_SCHEDULED_GATE_EDGES,
    MAX_SPWM_CARRIER_RATIO,
)
from pear_sim.spec.units import Quantity, parse_quantity, require_dimension


_REF_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")
MAX_MODEL_TEXT_LENGTH = 256


def _reject_unknown(mapping: dict[str, Any], allowed: set[str], *, name: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ModelError(f"{name}: unknown field(s): {', '.join(unknown)}")


def _require_model_text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ModelError(f"{name} must be a non-empty string")
    if len(value) > MAX_MODEL_TEXT_LENGTH:
        raise ModelError(
            f"{name} exceeds the {MAX_MODEL_TEXT_LENGTH}-character model limit"
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ModelError(f"{name} must contain valid UTF-8 Unicode scalar text") from exc
    return value


@dataclass(frozen=True)
class PwmControl:
    id: str
    frequency: float
    duty: float
    phase: float = 0.0
    kind: str = "pwm"
    block_id: str | None = None
    role: str | None = None
    partner_id: str | None = None
    dead_time: float = 0.0
    command_duty: float | None = None
    command_phase: float | None = None
    normalized_phase: float | None = None
    phase_label: str | None = None
    sequence: str | None = None
    fundamental_frequency: float | None = None
    modulation_index: float | None = None
    reference_phase: float | None = None
    carrier_ratio: int | None = None
    injection: str | None = None
    pair_label: str | None = None
    levels: int | None = None

    @property
    def period(self) -> float:
        return 1.0 / self.frequency


@dataclass(frozen=True)
class SampledPiPwmControl:
    """Cycle-synchronous sampled PI voltage/current-mode PWM (closed loop).

    Deterministic law, documented in docs/model-format.md: at every cycle
    boundary t_k = start + k/frequency the controller samples ``sense`` from
    the pre-edge state and pre-edge topology, updates the discrete PI with
    conditional-integration anti-windup, and commands the duty of the cycle
    that begins at t_k. The gate is on for the first ``duty_k`` fraction of
    the cycle. Sampling always happens strictly before the gate edge at the
    same timestamp.
    """

    id: str
    frequency: float
    sense: str
    reference: float
    kp: float
    ki: float
    duty_min: float
    duty_max: float
    kind: str = "sampled_pi_pwm"

    @property
    def period(self) -> float:
        return 1.0 / self.frequency


@dataclass(frozen=True)
class SampledDqCurrentPwmControl:
    """One gate output of a closed-loop dq current controller (grid tie).

    Six of these (a/b/c x high/low) share one logical block. Deterministic
    per-cycle law, documented in docs/model-format.md: at every carrier
    boundary the controller samples three currents and three PCC voltages
    from the pre-edge state, advances an SRF-PLL (theta used for THIS cycle
    is the pre-update angle), runs one PI step per dq axis with conditional-
    integration anti-windup against the +/-voltage_scale clamp, inverse-Park
    transforms the voltage commands, optionally applies min-max injection,
    and realizes three center-aligned complementary pulse pairs with dead
    time (turn-off immediate, turn-on after dead time).
    """

    id: str
    block_id: str
    phase_label: str
    role: str
    partner_id: str
    frequency: float
    dead_time: float
    sense_currents: tuple[str, str, str]
    sense_voltages: tuple[str, str, str]
    id_ref: float
    iq_ref: float
    current_kp: float
    current_ki: float
    voltage_scale: float
    pll_nominal_frequency_hz: float
    pll_bandwidth_hz: float
    pll_damping: float
    theta0_rad: float
    injection: str
    duty_min: float
    duty_max: float
    kind: str = "sampled_dq_current_pwm"

    @property
    def period(self) -> float:
        return 1.0 / self.frequency


def parse_sampled_dq_control(
    control_id: str,
    raw: dict[str, Any],
    parameters: dict[str, Quantity],
) -> dict[str, SampledDqCurrentPwmControl]:
    name = f"control {control_id}"
    _reject_unknown(
        raw,
        {
            "type",
            "frequency",
            "dead_time",
            "sense_currents",
            "sense_voltages",
            "id_ref",
            "iq_ref",
            "current_kp",
            "current_ki",
            "voltage_scale",
            "pll_nominal_frequency",
            "pll_bandwidth",
            "pll_damping",
            "theta0_deg",
            "injection",
            "duty_min",
            "duty_max",
        },
        name=name,
    )
    required = (
        "frequency",
        "sense_currents",
        "sense_voltages",
        "id_ref",
        "iq_ref",
        "current_kp",
        "current_ki",
        "voltage_scale",
        "pll_nominal_frequency",
    )
    for field in required:
        if field not in raw:
            raise ModelError(f"{name}: sampled_dq_current_pwm requires {field}")
    frequency = _resolve(raw["frequency"], parameters, "frequency", name=f"{name}.frequency")
    if frequency <= 0.0:
        raise ModelError(f"{name}.frequency must be positive")
    dead_time = _resolve(
        raw.get("dead_time", 0.0), parameters, "time", name=f"{name}.dead_time"
    )
    if dead_time < 0.0:
        raise ModelError(f"{name}.dead_time must be non-negative")

    def _sense_triple(field: str, expected_prefix: str) -> tuple[str, str, str]:
        raw_list = raw.get(field)
        if not isinstance(raw_list, list) or len(raw_list) != 3:
            raise ModelError(
                f"{name}.{field} must list exactly three output expressions "
                "in abc order"
            )
        values = tuple(
            _require_model_text(item, name=f"{name}.{field}") for item in raw_list
        )
        for expression in values:
            if not expression.startswith(expected_prefix):
                raise ModelError(
                    f"{name}.{field} entries must be {expected_prefix}...) "
                    "output expressions"
                )
        if len(set(values)) != 3:
            raise ModelError(f"{name}.{field} entries must be distinct")
        return values  # type: ignore[return-value]

    sense_currents = _sense_triple("sense_currents", "i(")
    sense_voltages = _sense_triple("sense_voltages", "v(")
    id_ref = _resolve(raw["id_ref"], parameters, "current", name=f"{name}.id_ref")
    iq_ref = _resolve(raw["iq_ref"], parameters, "current", name=f"{name}.iq_ref")
    current_kp = _resolve(
        raw["current_kp"], parameters, "dimensionless", name=f"{name}.current_kp"
    )
    current_ki = _resolve(
        raw["current_ki"], parameters, "dimensionless", name=f"{name}.current_ki"
    )
    voltage_scale = _resolve(
        raw["voltage_scale"], parameters, "voltage", name=f"{name}.voltage_scale"
    )
    if voltage_scale <= 0.0:
        raise ModelError(f"{name}.voltage_scale must be positive")
    pll_nominal = _resolve(
        raw["pll_nominal_frequency"],
        parameters,
        "frequency",
        name=f"{name}.pll_nominal_frequency",
    )
    if pll_nominal <= 0.0:
        raise ModelError(f"{name}.pll_nominal_frequency must be positive")
    pll_bandwidth = _resolve(
        raw.get("pll_bandwidth", 20.0),
        parameters,
        "frequency",
        name=f"{name}.pll_bandwidth",
    )
    if not 0.0 < pll_bandwidth < 0.1 * frequency:
        raise ModelError(
            f"{name}.pll_bandwidth must be positive and below a tenth of the "
            "sampling frequency"
        )
    pll_damping = _resolve(
        raw.get("pll_damping", 0.7071067811865476),
        parameters,
        "dimensionless",
        name=f"{name}.pll_damping",
    )
    if not 0.0 < pll_damping < 10.0:
        raise ModelError(f"{name}.pll_damping must be positive and sane")
    theta0_deg = _resolve(
        raw.get("theta0_deg", 0.0),
        parameters,
        "dimensionless",
        name=f"{name}.theta0_deg",
    )
    injection = raw.get("injection", "none")
    if injection not in {"none", "min_max"}:
        raise ModelError(f"{name}.injection must be none or min_max")
    duty_min = _resolve(
        raw.get("duty_min", 0.02), parameters, "dimensionless", name=f"{name}.duty_min"
    )
    duty_max = _resolve(
        raw.get("duty_max", 0.98), parameters, "dimensionless", name=f"{name}.duty_max"
    )
    if not 0.0 < duty_min < duty_max < 1.0:
        raise ModelError(
            f"{name}: duty limits must satisfy 0 < duty_min < duty_max < 1 strictly"
        )
    period = 1.0 / frequency
    if dead_time >= duty_min * period:
        raise ModelError(
            f"{name}.dead_time must be smaller than the minimum commanded "
            "pulse duty_min/frequency"
        )
    if dead_time >= 0.5 * (1.0 - duty_max) * period:
        raise ModelError(
            f"{name}.dead_time must be smaller than the end-of-cycle margin "
            "0.5*(1-duty_max)/frequency"
        )

    expansion: dict[str, SampledDqCurrentPwmControl] = {}
    common = dict(
        block_id=control_id,
        frequency=frequency,
        dead_time=dead_time,
        sense_currents=sense_currents,
        sense_voltages=sense_voltages,
        id_ref=id_ref,
        iq_ref=iq_ref,
        current_kp=current_kp,
        current_ki=current_ki,
        voltage_scale=voltage_scale,
        pll_nominal_frequency_hz=pll_nominal,
        pll_bandwidth_hz=pll_bandwidth,
        pll_damping=pll_damping,
        theta0_rad=math.radians(theta0_deg),
        injection=injection,
        duty_min=duty_min,
        duty_max=duty_max,
    )
    for phase_label in ("a", "b", "c"):
        high_id = f"{control_id}.{phase_label}.high"
        low_id = f"{control_id}.{phase_label}.low"
        expansion[high_id] = SampledDqCurrentPwmControl(
            id=high_id,
            phase_label=phase_label,
            role="high",
            partner_id=low_id,
            **common,
        )
        expansion[low_id] = SampledDqCurrentPwmControl(
            id=low_id,
            phase_label=phase_label,
            role="low",
            partner_id=high_id,
            **common,
        )
    return expansion


@dataclass(frozen=True)
class SignalPwmControl:
    """PWM whose duty comes from the declared signal layer.

    The bridge that makes the signal layer STEER rather than merely observe.
    It is deliberately thin: this block owns the gate timing and nothing
    else, while the control law lives entirely in `signals`. That split is
    deliberate — a controller and a modulator are two things, and fusing them is
    why the older types cannot have their modulator swapped without being
    rewritten.

    The carrier is locked to the signal task (one tick, one cycle) and the
    loader refuses any other ratio: a duty updated at a rate the carrier does
    not share has no single well-defined cycle to belong to, and picking one
    silently is the class of timing assumption this format refuses.
    """

    id: str
    frequency: float
    duty_signal: str
    duty_min: float
    duty_max: float
    kind: str = "signal_pwm"

    @property
    def period(self) -> float:
        return 1.0 / self.frequency


def parse_signal_pwm_control(
    control_id: str,
    raw: dict[str, Any],
    parameters: dict[str, Quantity],
) -> SignalPwmControl:
    name = f"control {control_id}"
    _reject_unknown(
        raw,
        {"type", "frequency", "duty_signal", "duty_min", "duty_max"},
        name=name,
    )
    for field in ("frequency", "duty_signal"):
        if field not in raw:
            raise ModelError(f"{name}: signal_pwm requires {field}")
    frequency = _resolve(raw["frequency"], parameters, "frequency", name=f"{name}.frequency")
    if frequency <= 0.0:
        raise ModelError(f"{name}.frequency must be positive")
    duty_signal = _require_model_text(raw["duty_signal"], name=f"{name}.duty_signal")
    duty_min = _resolve(
        raw.get("duty_min", 0.0), parameters, "dimensionless", name=f"{name}.duty_min"
    )
    duty_max = _resolve(
        raw.get("duty_max", 1.0), parameters, "dimensionless", name=f"{name}.duty_max"
    )
    if not 0.0 <= duty_min < duty_max <= 1.0:
        raise ModelError(
            f"{name}: duty limits must satisfy 0 <= duty_min < duty_max <= 1"
        )
    return SignalPwmControl(
        id=control_id,
        frequency=frequency,
        duty_signal=duty_signal,
        duty_min=duty_min,
        duty_max=duty_max,
    )


@dataclass(frozen=True)
class SignalSvpwmControl:
    """Three-phase bridge driven by a VOLTAGE VECTOR from the signal layer.

    The modulator a field-oriented controller actually needs: `signal_pwm`
    takes a duty, which is already a modulator's output, so a control law
    using it has to modulate itself. This one takes the commanded vector in
    the stationary frame — exactly what an inverse Park produces — and owns
    the sector decision and the dwell times.
    """

    id: str
    block_id: str
    phase_label: str
    role: str
    partner_id: str
    frequency: float
    alpha_signal: str
    beta_signal: str
    dc_link_voltage: float
    dead_time: float
    kind: str = "signal_svpwm"

    @property
    def period(self) -> float:
        return 1.0 / self.frequency


def parse_signal_svpwm_control(
    control_id: str,
    raw: dict[str, Any],
    parameters: dict[str, Quantity],
) -> dict[str, SignalSvpwmControl]:
    name = f"control {control_id}"
    _reject_unknown(
        raw,
        {"type", "frequency", "alpha_signal", "beta_signal", "dc_link_voltage",
         "dead_time"},
        name=name,
    )
    for field in ("frequency", "alpha_signal", "beta_signal", "dc_link_voltage"):
        if field not in raw:
            raise ModelError(f"{name}: signal_svpwm requires {field}")
    frequency = _resolve(
        raw["frequency"], parameters, "frequency", name=f"{name}.frequency"
    )
    if frequency <= 0.0:
        raise ModelError(f"{name}.frequency must be positive")
    dc_link = _resolve(
        raw["dc_link_voltage"], parameters, "voltage", name=f"{name}.dc_link_voltage"
    )
    if dc_link <= 0.0:
        raise ModelError(f"{name}.dc_link_voltage must be positive")
    dead_time = _resolve(
        raw.get("dead_time", 0.0), parameters, "time", name=f"{name}.dead_time"
    )
    if dead_time < 0.0:
        raise ModelError(f"{name}.dead_time must be non-negative")
    if dead_time >= 0.5 / frequency:
        raise ModelError(
            f"{name}.dead_time must be shorter than half the carrier period"
        )
    alpha = _require_model_text(raw["alpha_signal"], name=f"{name}.alpha_signal")
    beta = _require_model_text(raw["beta_signal"], name=f"{name}.beta_signal")
    if alpha == beta:
        raise ModelError(
            f"{name}: alpha_signal and beta_signal must be different blocks"
        )
    expansion: dict[str, SignalSvpwmControl] = {}
    common = dict(
        block_id=control_id,
        frequency=frequency,
        alpha_signal=alpha,
        beta_signal=beta,
        dc_link_voltage=dc_link,
        dead_time=dead_time,
    )
    for phase_label in ("a", "b", "c"):
        high_id = f"{control_id}.{phase_label}.high"
        low_id = f"{control_id}.{phase_label}.low"
        expansion[high_id] = SignalSvpwmControl(
            id=high_id, phase_label=phase_label, role="high",
            partner_id=low_id, **common,
        )
        expansion[low_id] = SignalSvpwmControl(
            id=low_id, phase_label=phase_label, role="low",
            partner_id=high_id, **common,
        )
    return expansion


def parse_sampled_pi_control(
    control_id: str,
    raw: dict[str, Any],
    parameters: dict[str, Quantity],
) -> SampledPiPwmControl:
    name = f"control {control_id}"
    _reject_unknown(
        raw,
        {
            "type",
            "frequency",
            "sense",
            "reference",
            "kp",
            "ki",
            "duty_min",
            "duty_max",
        },
        name=name,
    )
    for field in ("frequency", "sense", "reference", "kp", "ki", "duty_min", "duty_max"):
        if field not in raw:
            raise ModelError(f"{name}: sampled_pi_pwm requires {field}")
    frequency = _resolve(raw["frequency"], parameters, "frequency", name=f"{name}.frequency")
    if frequency <= 0.0:
        raise ModelError(f"{name}.frequency must be positive")
    sense = _require_model_text(raw["sense"], name=f"{name}.sense")
    if sense.startswith("v("):
        sense_dimension = "voltage"
    elif sense.startswith("i("):
        sense_dimension = "current"
    else:
        raise ModelError(
            f"{name}.sense must be a v(...) or i(...) output expression"
        )
    reference = _resolve(
        raw["reference"], parameters, sense_dimension, name=f"{name}.reference"
    )
    kp = _resolve(raw["kp"], parameters, "dimensionless", name=f"{name}.kp")
    ki = _resolve(raw["ki"], parameters, "dimensionless", name=f"{name}.ki")
    duty_min = _resolve(
        raw["duty_min"], parameters, "dimensionless", name=f"{name}.duty_min"
    )
    duty_max = _resolve(
        raw["duty_max"], parameters, "dimensionless", name=f"{name}.duty_max"
    )
    if not 0.0 < duty_min < duty_max < 1.0:
        raise ModelError(
            f"{name}: duty limits must satisfy 0 < duty_min < duty_max < 1 "
            "strictly (gate edges must never coincide with cycle boundaries)"
        )
    return SampledPiPwmControl(
        id=control_id,
        frequency=frequency,
        sense=sense,
        reference=reference,
        kp=kp,
        ki=ki,
        duty_min=duty_min,
        duty_max=duty_max,
    )


def _resolve(
    raw: Any,
    parameters: dict[str, Quantity],
    expected: str,
    *,
    name: str,
) -> float:
    if isinstance(raw, str):
        match = _REF_RE.match(raw.strip())
        if match:
            parameter_name = match.group(1)
            if parameter_name not in parameters:
                raise ModelError(f"{name}: unknown parameter {parameter_name!r}")
            return require_dimension(parameters[parameter_name], expected, name=name)
    return parse_quantity(raw, expected, name=name).value


NPC_PAIRS = (
    # (pair_label, high-side gate suffix, low-side gate suffix). "high" of the
    # upper pair is the outer-high device; its complementary partner is the
    # inner-low device (standard three-level NPC gating: Q1<->Q3, Q2<->Q4).
    ("upper", "outer_high", "inner_low"),
    ("lower", "inner_high", "outer_low"),
)

NPC_SUPPORTED_LEVELS = (3, 5)


def npc_pair_specs(levels: int) -> tuple[tuple[str, str, str], ...]:
    """Complementary-pair naming for an N-level diode-clamped leg, top first.

    Pair ``k`` couples top-arm device ``S_k`` with bottom-arm device
    ``S_(k+N-1)``; its high-side duty is band ``N-1-k`` of the bottom-first
    :func:`npc_pair_duties` decomposition. Three-level legs keep the shipped
    ``outer/inner`` names; five-level legs use the systematic ``h1..h4`` /
    ``l1..l4`` scheme (h1 = S1 at the positive rail, l4 = S8 at the negative
    rail).
    """

    if levels == 3:
        return NPC_PAIRS
    if levels == 5:
        return tuple(
            (f"p{k}", f"h{k}", f"l{k}") for k in range(1, 5)
        )
    raise ModelError(
        f"NPC pair naming is defined for levels in {NPC_SUPPORTED_LEVELS}, "
        f"got {levels}"
    )


def npc_pair_duties(reference: float, levels: int) -> tuple[float, ...]:
    """Per-pair duties of a level-shifted (PD) carrier stack, bottom pair first.

    The [-1, 1] reference range is split into ``levels - 1`` stacked carrier
    bands of height 2/(levels-1); pair ``j`` saturates at 0 below its band,
    at 1 above it, and interpolates linearly inside. Written generically so a
    five-level ladder is a parameter change; v0.1 validates ``levels == 3``.
    """

    band_height = 2.0 / (levels - 1)
    duties = []
    for band in range(levels - 1):
        lower_edge = -1.0 + band * band_height
        duty = (reference - lower_edge) / band_height
        duties.append(min(1.0, max(0.0, duty)))
    return tuple(duties)


def _parse_npc_spwm(
    control_id: str,
    raw_control: dict[str, Any],
    parameters: dict[str, Quantity],
) -> tuple[PwmControl, ...]:
    """Expand a three-level NPC level-shifted modulator into 12 gate outputs."""

    required_fields = {
        "type",
        "carrier_frequency",
        "fundamental_frequency",
        "modulation_index",
        "dead_time",
    }
    _reject_unknown(
        raw_control,
        required_fields
        | {"carrier_phase", "reference_phase", "sequence", "injection", "levels"},
        name=f"control {control_id}",
    )
    missing_fields = sorted(required_fields - set(raw_control))
    if missing_fields:
        raise ModelError(
            f"control {control_id}: missing required field(s): "
            + ", ".join(missing_fields)
        )
    levels = raw_control.get("levels", 3)
    if (
        isinstance(levels, bool)
        or not isinstance(levels, int)
        or levels not in NPC_SUPPORTED_LEVELS
    ):
        raise ModelError(
            f"control {control_id}.levels must be one of "
            f"{NPC_SUPPORTED_LEVELS}; other ladder depths await a validated "
            "topology and oracle fixture (the carrier-band law itself is "
            "level-generic)"
        )
    carrier_frequency = _resolve(
        raw_control.get("carrier_frequency"),
        parameters,
        "frequency",
        name=f"control {control_id}.carrier_frequency",
    )
    fundamental_frequency = _resolve(
        raw_control.get("fundamental_frequency"),
        parameters,
        "frequency",
        name=f"control {control_id}.fundamental_frequency",
    )
    modulation_index = _resolve(
        raw_control.get("modulation_index"),
        parameters,
        "dimensionless",
        name=f"control {control_id}.modulation_index",
    )
    dead_time = _resolve(
        raw_control.get("dead_time"),
        parameters,
        "time",
        name=f"control {control_id}.dead_time",
    )
    carrier_phase = _resolve(
        raw_control.get("carrier_phase", 0.0),
        parameters,
        "time",
        name=f"control {control_id}.carrier_phase",
    )
    reference_phase = _resolve(
        raw_control.get("reference_phase", 0.0),
        parameters,
        "time",
        name=f"control {control_id}.reference_phase",
    )
    sequence = _require_model_text(
        raw_control.get("sequence", "abc"),
        name=f"control {control_id}.sequence",
    )
    if sequence not in {"abc", "acb"}:
        raise ModelError(f"control {control_id}.sequence must be 'abc' or 'acb'")
    if carrier_frequency <= 0.0:
        raise ModelError(f"control {control_id}.carrier_frequency must be positive")
    if fundamental_frequency <= 0.0:
        raise ModelError(
            f"control {control_id}.fundamental_frequency must be positive"
        )
    if carrier_frequency <= fundamental_frequency:
        raise ModelError(
            f"control {control_id}.carrier_frequency must be greater than "
            "fundamental_frequency"
        )
    carrier_ratio_value = carrier_frequency / fundamental_frequency
    if not math.isfinite(carrier_ratio_value):
        raise ModelError(
            f"control {control_id}: carrier ratio must fit the finite portable "
            "integer contract"
        )
    nearest_carrier_ratio = round(carrier_ratio_value)
    if not math.isclose(
        carrier_ratio_value,
        float(nearest_carrier_ratio),
        rel_tol=0.0,
        abs_tol=min(0.25, 8.0 * math.ulp(carrier_ratio_value)),
    ):
        raise ModelError(
            f"control {control_id}: carrier_frequency/fundamental_frequency "
            "must be an integer carrier ratio"
        )
    carrier_ratio = int(nearest_carrier_ratio)
    if carrier_ratio < 2:
        raise ModelError(f"control {control_id}.carrier_ratio must be at least 2")
    # One carrier cell is bounded by six edges per pair: 18*(levels-1) edges
    # across three phases. levels=3 reproduces MAX_NPC_CARRIER_RATIO.
    max_carrier_ratio = MAX_SCHEDULED_GATE_EDGES // (18 * (levels - 1))
    if carrier_ratio > max_carrier_ratio:
        raise ModelError(
            f"control {control_id}.carrier_ratio exceeds the portable limit "
            f"of {max_carrier_ratio} for {levels} levels"
        )
    injection = raw_control.get("injection", "none")
    if injection not in {"none", "min_max"}:
        raise ModelError(f"control {control_id}.injection must be none or min_max")
    reference_peak = math.sqrt(3.0) / 2.0 if injection == "min_max" else 1.0
    if not 0.0 <= modulation_index * reference_peak < 1.0:
        limit = "2/sqrt(3)" if injection == "min_max" else "1"
        raise ModelError(
            f"control {control_id}.modulation_index must be in [0, {limit}) "
            f"for injection={injection}"
        )
    if dead_time < 0.0:
        raise ModelError(f"control {control_id}.dead_time must be non-negative")
    # Unlike two-level SPWM, band-edge cells legitimately command pulses
    # shorter than the dead time (the scheduler suppresses them and records
    # the suppression — hardware minimum-pulse behavior). The loader bound
    # only guarantees the carrier cell itself can host a dead-time fix-up.
    if dead_time >= 0.25 / carrier_frequency:
        raise ModelError(
            f"control {control_id}.dead_time must be smaller than a quarter "
            "of the carrier period"
        )

    carrier_period = 1.0 / carrier_frequency
    common = {
        "frequency": carrier_frequency,
        "duty": 0.5,
        "phase": carrier_phase,
        "kind": "three_phase_npc_spwm",
        "block_id": control_id,
        "dead_time": dead_time,
        "normalized_phase": carrier_phase % carrier_period,
        "sequence": sequence,
        "fundamental_frequency": fundamental_frequency,
        "modulation_index": modulation_index,
        "reference_phase": reference_phase,
        "carrier_ratio": carrier_ratio,
        "injection": injection,
        "levels": levels,
    }
    expansion: list[PwmControl] = []
    for phase_label in ("a", "b", "c"):
        for pair_label, high_suffix, low_suffix in npc_pair_specs(levels):
            high_id = f"{control_id}.{phase_label}.{high_suffix}"
            low_id = f"{control_id}.{phase_label}.{low_suffix}"
            for derived_id in (high_id, low_id):
                if len(derived_id) > MAX_MODEL_TEXT_LENGTH:
                    raise ModelError(
                        f"control {control_id}: derived gate id {derived_id!r} "
                        f"exceeds the {MAX_MODEL_TEXT_LENGTH}-character model limit"
                    )
            expansion.extend(
                (
                    PwmControl(
                        id=high_id,
                        role="high",
                        partner_id=low_id,
                        phase_label=phase_label,
                        pair_label=pair_label,
                        **common,
                    ),
                    PwmControl(
                        id=low_id,
                        role="low",
                        partner_id=high_id,
                        phase_label=phase_label,
                        pair_label=pair_label,
                        **common,
                    ),
                )
            )
    return tuple(expansion)


def parse_controls(
    raw_controls: Any,
    parameters: dict[str, Quantity],
) -> dict[str, PwmControl]:
    if not isinstance(raw_controls, dict):
        raise ModelError("controls must be an object")
    if len(raw_controls) > MAX_MODEL_LOGICAL_CONTROLS:
        raise ModelError(
            f"controls contains {len(raw_controls)} logical controls; the portable "
            f"model limit is {MAX_MODEL_LOGICAL_CONTROLS}"
        )
    three_phase_sources = [
        control_id
        for control_id, raw_control in raw_controls.items()
        if isinstance(raw_control, dict)
        and raw_control.get("type") in {"three_phase_spwm", "three_phase_npc_spwm"}
    ]
    if three_phase_sources and (
        len(three_phase_sources) != 1 or len(raw_controls) != 1
    ):
        raise ModelError(
            "a three-phase modulator (three_phase_spwm or three_phase_npc_spwm) "
            "must be the only source in controls and may occur exactly once"
        )
    controls: dict[str, PwmControl] = {}
    expanded_controls: list[tuple[str, tuple[PwmControl, ...]]] = []
    for control_id, raw_control in raw_controls.items():
        if not isinstance(raw_control, dict):
            raise ModelError("Each control must be a named object")
        control_id = _require_model_text(control_id, name="control id")
        control_type = raw_control.get("type")
        if control_type not in {
            "pwm",
            "complementary_pwm",
            "three_phase_spwm",
            "three_phase_npc_spwm",
        }:
            raise ModelError(
                f"control {control_id}: type must be pwm, complementary_pwm, "
                "three_phase_spwm, or three_phase_npc_spwm"
            )
        if control_type == "three_phase_npc_spwm":
            expanded_controls.append(
                (
                    control_id,
                    _parse_npc_spwm(control_id, raw_control, parameters),
                )
            )
            continue
        if control_type == "three_phase_spwm":
            required_fields = {
                "type",
                "carrier_frequency",
                "fundamental_frequency",
                "modulation_index",
                "dead_time",
            }
            _reject_unknown(
                raw_control,
                {
                    "type",
                    "carrier_frequency",
                    "fundamental_frequency",
                    "modulation_index",
                    "dead_time",
                    "carrier_phase",
                    "reference_phase",
                    "sequence",
                    "injection",
                },
                name=f"control {control_id}",
            )
            missing_fields = sorted(required_fields - set(raw_control))
            if missing_fields:
                raise ModelError(
                    f"control {control_id}: missing required field(s): "
                    + ", ".join(missing_fields)
                )
            carrier_frequency = _resolve(
                raw_control.get("carrier_frequency"),
                parameters,
                "frequency",
                name=f"control {control_id}.carrier_frequency",
            )
            fundamental_frequency = _resolve(
                raw_control.get("fundamental_frequency"),
                parameters,
                "frequency",
                name=f"control {control_id}.fundamental_frequency",
            )
            modulation_index = _resolve(
                raw_control.get("modulation_index"),
                parameters,
                "dimensionless",
                name=f"control {control_id}.modulation_index",
            )
            dead_time = _resolve(
                raw_control.get("dead_time"),
                parameters,
                "time",
                name=f"control {control_id}.dead_time",
            )
            carrier_phase = _resolve(
                raw_control.get("carrier_phase", 0.0),
                parameters,
                "time",
                name=f"control {control_id}.carrier_phase",
            )
            reference_phase = _resolve(
                raw_control.get("reference_phase", 0.0),
                parameters,
                "time",
                name=f"control {control_id}.reference_phase",
            )
            sequence = raw_control.get("sequence", "abc")
            sequence = _require_model_text(
                sequence,
                name=f"control {control_id}.sequence",
            )
            if sequence not in {"abc", "acb"}:
                raise ModelError(
                    f"control {control_id}.sequence must be 'abc' or 'acb'"
                )
            if carrier_frequency <= 0.0:
                raise ModelError(
                    f"control {control_id}.carrier_frequency must be positive"
                )
            if fundamental_frequency <= 0.0:
                raise ModelError(
                    f"control {control_id}.fundamental_frequency must be positive"
                )
            if carrier_frequency <= fundamental_frequency:
                raise ModelError(
                    f"control {control_id}.carrier_frequency must be greater than "
                    "fundamental_frequency"
                )
            carrier_ratio_value = carrier_frequency / fundamental_frequency
            if not math.isfinite(carrier_ratio_value):
                raise ModelError(
                    f"control {control_id}: carrier ratio must fit the finite "
                    "portable integer contract"
                )
            nearest_carrier_ratio = round(carrier_ratio_value)
            carrier_ratio_tolerance = min(
                0.25,
                8.0 * math.ulp(carrier_ratio_value),
            )
            if not math.isclose(
                carrier_ratio_value,
                float(nearest_carrier_ratio),
                rel_tol=0.0,
                abs_tol=carrier_ratio_tolerance,
            ):
                raise ModelError(
                    f"control {control_id}: carrier_frequency/fundamental_frequency "
                    "must be an integer carrier ratio"
                )
            carrier_ratio = int(nearest_carrier_ratio)
            if carrier_ratio < 2:
                raise ModelError(
                    f"control {control_id}.carrier_ratio must be at least 2"
                )
            if carrier_ratio > MAX_SPWM_CARRIER_RATIO:
                raise ModelError(
                    f"control {control_id}.carrier_ratio exceeds the portable limit "
                    f"of {MAX_SPWM_CARRIER_RATIO}"
                )
            injection = raw_control.get("injection", "none")
            if injection not in {"none", "min_max"}:
                raise ModelError(
                    f"control {control_id}.injection must be none or min_max"
                )
            # Peak of the (possibly injected) per-phase reference for a unit
            # sinusoidal set: 1 without injection, sqrt(3)/2 with min-max
            # (SVPWM-equivalent) centering — linear range extends to 2/sqrt(3).
            reference_peak = (
                math.sqrt(3.0) / 2.0 if injection == "min_max" else 1.0
            )
            if not 0.0 <= modulation_index * reference_peak < 1.0:
                limit = "2/sqrt(3)" if injection == "min_max" else "1"
                raise ModelError(
                    f"control {control_id}.modulation_index must be in "
                    f"[0, {limit}) for injection={injection}"
                )
            if dead_time < 0.0:
                raise ModelError(
                    f"control {control_id}.dead_time must be non-negative"
                )
            maximum_dead_time = (
                0.5 * (1.0 - modulation_index * reference_peak) / carrier_frequency
            )
            if dead_time >= maximum_dead_time:
                raise ModelError(
                    f"control {control_id}.dead_time must be smaller than the "
                    "minimum commanded pulse, "
                    "0.5*(1-modulation_index*reference_peak)/carrier_frequency"
                )

            carrier_period = 1.0 / carrier_frequency
            normalized_carrier_phase = carrier_phase % carrier_period
            expansion: list[PwmControl] = []
            common = {
                "frequency": carrier_frequency,
                "duty": 0.5,
                "phase": carrier_phase,
                "kind": "three_phase_spwm",
                "block_id": control_id,
                "dead_time": dead_time,
                "normalized_phase": normalized_carrier_phase,
                "sequence": sequence,
                "fundamental_frequency": fundamental_frequency,
                "modulation_index": modulation_index,
                "reference_phase": reference_phase,
                "carrier_ratio": carrier_ratio,
                "injection": injection,
            }
            for phase_label in ("a", "b", "c"):
                high_id = f"{control_id}.{phase_label}.high"
                low_id = f"{control_id}.{phase_label}.low"
                for derived_id in (high_id, low_id):
                    if len(derived_id) > MAX_MODEL_TEXT_LENGTH:
                        raise ModelError(
                            f"control {control_id}: derived gate id {derived_id!r} "
                            f"exceeds the {MAX_MODEL_TEXT_LENGTH}-character model limit"
                        )
                expansion.extend(
                    (
                        PwmControl(
                            id=high_id,
                            role="high",
                            partner_id=low_id,
                            phase_label=phase_label,
                            **common,
                        ),
                        PwmControl(
                            id=low_id,
                            role="low",
                            partner_id=high_id,
                            phase_label=phase_label,
                            **common,
                        ),
                    )
                )
            expanded_controls.append((control_id, tuple(expansion)))
            continue

        allowed_fields = {"type", "frequency", "duty", "phase"}
        if control_type == "complementary_pwm":
            allowed_fields.add("dead_time")
        _reject_unknown(raw_control, allowed_fields, name=f"control {control_id}")
        frequency = _resolve(
            raw_control.get("frequency"), parameters, "frequency", name=f"control {control_id}.frequency"
        )
        duty = _resolve(
            raw_control.get("duty"), parameters, "dimensionless", name=f"control {control_id}.duty"
        )
        phase = _resolve(
            raw_control.get("phase", 0.0), parameters, "time", name=f"control {control_id}.phase"
        )
        if frequency <= 0:
            raise ModelError(f"control {control_id}.frequency must be positive")
        if not 0.0 <= duty <= 1.0:
            raise ModelError(f"control {control_id}.duty must be in [0, 1]")
        if control_type == "pwm":
            expanded_controls.append(
                (control_id, (PwmControl(control_id, frequency, duty, phase),))
            )
            continue

        dead_time = _resolve(
            raw_control.get("dead_time"),
            parameters,
            "time",
            name=f"control {control_id}.dead_time",
        )
        if dead_time < 0.0:
            raise ModelError(f"control {control_id}.dead_time must be non-negative")
        period = 1.0 / frequency
        if duty in {0.0, 1.0}:
            if dead_time != 0.0:
                raise ModelError(
                    f"control {control_id}.dead_time must be zero when duty is 0 or 1"
                )
        else:
            shortest_command = min(duty * period, (1.0 - duty) * period)
            if dead_time >= shortest_command:
                raise ModelError(
                    f"control {control_id}.dead_time must be smaller than both commanded pulses"
                )

        high_id = f"{control_id}.high"
        low_id = f"{control_id}.low"
        for derived_id in (high_id, low_id):
            if len(derived_id) > MAX_MODEL_TEXT_LENGTH:
                raise ModelError(
                    f"control {control_id}: derived gate id {derived_id!r} exceeds the "
                    f"{MAX_MODEL_TEXT_LENGTH}-character model limit"
                )

        if duty == 0.0:
            high_duty, high_phase = 0.0, phase
            low_duty, low_phase = 1.0, phase
        elif duty == 1.0:
            high_duty, high_phase = 1.0, phase
            low_duty, low_phase = 0.0, phase
        else:
            canonical_phase = phase % period
            high_phase = canonical_phase + dead_time
            low_phase = canonical_phase + duty * period + dead_time
            high_duty = duty - dead_time / period
            low_duty = (1.0 - duty) - dead_time / period
            high_off = canonical_phase + duty * period
            cycle_end = canonical_phase + period
            next_high_on = cycle_end + dead_time
            if not 0.0 < high_duty < 1.0 or not 0.0 < low_duty < 1.0:
                raise ModelError(
                    f"control {control_id}.dead_time deletes a pulse at float precision"
                )
            if not high_phase < high_off or not low_phase < cycle_end:
                raise ModelError(
                    f"control {control_id}.dead_time deletes a pulse at float precision"
                )
            if dead_time > 0.0 and not (
                high_phase > canonical_phase
                and low_phase > high_off
                and next_high_on > cycle_end
            ):
                raise ModelError(
                    f"control {control_id}.dead_time is not representable at this time scale"
                )

        common = {
            "kind": "complementary_pwm",
            "block_id": control_id,
            "dead_time": dead_time,
            "command_duty": duty,
            "command_phase": phase,
            "normalized_phase": (
                canonical_phase if duty not in {0.0, 1.0} else phase % period
            ),
        }
        expanded_controls.append(
            (
                control_id,
                (
                    PwmControl(
                        high_id,
                        frequency,
                        high_duty,
                        high_phase,
                        role="high",
                        partner_id=low_id,
                        **common,
                    ),
                    PwmControl(
                        low_id,
                        frequency,
                        low_duty,
                        low_phase,
                        role="low",
                        partner_id=high_id,
                        **common,
                    ),
                ),
            )
        )

    owners: dict[str, str] = {}
    source_ids = set(raw_controls)
    for source_id, expansion in expanded_controls:
        for control in expansion:
            if control.id in source_ids and control.id != source_id:
                raise ModelError(
                    f"control {source_id}: generated gate id {control.id!r} collides with "
                    "another source control id"
                )
            previous_owner = owners.get(control.id)
            if previous_owner is not None:
                raise ModelError(
                    f"control {source_id}: generated gate id {control.id!r} collides with "
                    f"control {previous_owner!r}"
                )
            owners[control.id] = source_id
            controls[control.id] = control

    if len(controls) > MAX_MODEL_GATE_OUTPUTS:
        raise ModelError(
            f"controls expand to {len(controls)} gate outputs; the portable model "
            f"limit is {MAX_MODEL_GATE_OUTPUTS}"
        )

    return controls
