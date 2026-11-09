# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Device and cooling-path cards: the SHAPE of loss modelling.

A card bundles datasheet-derived lookup tables with per-figure provenance and
an explicit validity envelope. The reference kernel stays ideal — losses are
a post-processing estimate layered on ideal waveforms (the usual paradigm of
ideal-switch loss modelling), and every extrapolated lookup is counted, never
silently trusted.

Deliberately NOT here (declared omissions, see docs/model-format.md):
3-D geometry, heat spreading, shared-heatsink coupling between devices,
convection orientation/altitude corrections, and closed-loop electrothermal
iteration (tables are evaluated at a DECLARED junction temperature).
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any

from pear_sim.errors import ModelError
from pear_sim.spec.controls import _reject_unknown, _require_model_text

__all__ = [
    "CardProvenance",
    "ConductionTable",
    "CoolingPath",
    "DeviceCard",
    "DeviceLossBinding",
    "FanCard",
    "FosterStage",
    "HeatsinkCard",
    "LossAnalysisSpec",
    "SwitchingTable",
    "TimCard",
    "parse_loss_sections",
]

_MAX_AXIS_POINTS = 64
_MAX_FOSTER_STAGES = 16


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ModelError(f"{name} must be a finite number")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise ModelError(f"{name} must be a finite number")
    return result


def _number_list(value: Any, *, name: str, minimum_len: int = 2) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) < minimum_len:
        raise ModelError(f"{name} must list at least {minimum_len} numbers")
    if len(value) > _MAX_AXIS_POINTS:
        raise ModelError(f"{name} exceeds the {_MAX_AXIS_POINTS}-point axis limit")
    return tuple(_finite_number(item, name=f"{name}[{index}]") for index, item in enumerate(value))


def _strictly_increasing(values: tuple[float, ...], *, name: str) -> None:
    for left, right in zip(values, values[1:]):
        if right <= left:
            raise ModelError(f"{name} must be strictly increasing")


@dataclass(frozen=True)
class CardProvenance:
    """Per-table provenance: enough to point a reviewer at the exact figure."""

    source: str
    page: int | None = None
    figure: str | None = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"source": self.source}
        if self.page is not None:
            payload["page"] = self.page
        if self.figure is not None:
            payload["figure"] = self.figure
        return payload


def _parse_provenance(raw: Any, *, name: str) -> CardProvenance:
    if not isinstance(raw, dict):
        raise ModelError(f"{name}.provenance is required and must be an object")
    _reject_unknown(raw, {"source", "page", "figure"}, name=f"{name}.provenance")
    source = _require_model_text(raw.get("source"), name=f"{name}.provenance.source")
    page = raw.get("page")
    if page is not None and (isinstance(page, bool) or not isinstance(page, int) or page < 1):
        raise ModelError(f"{name}.provenance.page must be a positive integer")
    figure = raw.get("figure")
    if figure is not None:
        figure = _require_model_text(figure, name=f"{name}.provenance.figure")
    return CardProvenance(source=source, page=page, figure=figure)


@dataclass(frozen=True)
class ConductionTable:
    """On-state voltage v_on(i, Tj) as a piecewise-bilinear table."""

    current_axis_a: tuple[float, ...]
    temperature_axis_c: tuple[float, ...]
    voltage_table_v: tuple[tuple[float, ...], ...]  # rows follow temperature axis
    provenance: CardProvenance


@dataclass(frozen=True)
class SwitchingTable:
    """Per-event energies at declared DPT test conditions.

    ``voltage_scaling: linear`` is the only supported law in v0.1 and is a
    DECLARED assumption: E(event) = E_table(i) * V_commutation / test_voltage.
    """

    test_voltage_v: float
    test_temperature_c: float
    current_axis_a: tuple[float, ...]
    e_on_j: tuple[float, ...] | None
    e_off_j: tuple[float, ...] | None
    e_rec_j: tuple[float, ...] | None
    voltage_scaling: str
    provenance: CardProvenance


@dataclass(frozen=True)
class FosterStage:
    resistance_k_per_w: float
    time_constant_s: float


@dataclass(frozen=True)
class DeviceCard:
    id: str
    kind: str  # "switch" | "diode"
    source: str
    validity_tj_min_c: float
    validity_tj_max_c: float
    validity_current_max_a: float
    conduction: ConductionTable
    switching: SwitchingTable | None
    foster_stages: tuple[FosterStage, ...]
    thermal_provenance: CardProvenance


@dataclass(frozen=True)
class HeatsinkCard:
    id: str
    airflow_axis_m3_per_h: tuple[float, ...]
    rth_k_per_w: tuple[float, ...]
    pressure_drop_pa: tuple[float, ...]
    provenance: CardProvenance


@dataclass(frozen=True)
class FanCard:
    id: str
    flow_axis_m3_per_h: tuple[float, ...]
    pressure_pa: tuple[float, ...]
    provenance: CardProvenance


@dataclass(frozen=True)
class ColdPlateCard:
    """Liquid cold plate: thermal resistance against coolant flow.

    A liquid loop's flow is set by its pump and plumbing, which this card
    does NOT model — the operating flow is declared per device binding and
    the omission is recorded, unlike the air path where the fan PQ curve and
    the heatsink pressure drop genuinely intersect at one point. The plate
    also carries its own reference temperature: a liquid-cooled device sits
    at the coolant inlet, not at cabinet ambient.
    """

    id: str
    flow_axis_l_per_min: tuple[float, ...]
    rth_k_per_w: tuple[float, ...]
    coolant_inlet_c: float
    provenance: CardProvenance


@dataclass(frozen=True)
class TimCard:
    id: str
    rth_k_per_w: float
    provenance: CardProvenance


@dataclass(frozen=True)
class CoolingPath:
    tim: str | None
    heatsink: str | None
    fan: str | None
    cold_plate: str | None = None
    coolant_flow_l_per_min: float | None = None


@dataclass(frozen=True)
class DeviceLossBinding:
    component_id: str
    card_id: str
    commutation_voltage: str | None
    cooling: CoolingPath


@dataclass(frozen=True)
class LossAnalysisSpec:
    ambient_c: float
    tj_evaluation_c: float
    devices: tuple[DeviceLossBinding, ...]
    cards: dict[str, DeviceCard]
    heatsinks: dict[str, HeatsinkCard]
    fans: dict[str, FanCard]
    tims: dict[str, TimCard]
    cold_plates: dict[str, ColdPlateCard] = dataclass_field(default_factory=dict)


def _parse_conduction(raw: Any, *, name: str) -> ConductionTable:
    if not isinstance(raw, dict):
        raise ModelError(f"{name}.conduction is required and must be an object")
    _reject_unknown(
        raw,
        {"current_axis_a", "temperature_axis_c", "voltage_table_v", "provenance"},
        name=f"{name}.conduction",
    )
    current_axis = _number_list(
        raw.get("current_axis_a"), name=f"{name}.conduction.current_axis_a"
    )
    _strictly_increasing(current_axis, name=f"{name}.conduction.current_axis_a")
    if current_axis[0] < 0.0:
        raise ModelError(f"{name}.conduction.current_axis_a must be non-negative")
    temperature_axis = _number_list(
        raw.get("temperature_axis_c"),
        name=f"{name}.conduction.temperature_axis_c",
        minimum_len=1,
    )
    if len(temperature_axis) > 1:
        _strictly_increasing(
            temperature_axis, name=f"{name}.conduction.temperature_axis_c"
        )
    rows = raw.get("voltage_table_v")
    if not isinstance(rows, list) or len(rows) != len(temperature_axis):
        raise ModelError(
            f"{name}.conduction.voltage_table_v must have one row per "
            "temperature-axis point"
        )
    table = tuple(
        _number_list(row, name=f"{name}.conduction.voltage_table_v[{index}]", minimum_len=len(current_axis))
        for index, row in enumerate(rows)
    )
    for index, row in enumerate(table):
        if len(row) != len(current_axis):
            raise ModelError(
                f"{name}.conduction.voltage_table_v[{index}] must match the "
                "current axis length"
            )
        if any(value < 0.0 for value in row):
            raise ModelError(
                f"{name}.conduction.voltage_table_v[{index}] must be non-negative"
            )
    return ConductionTable(
        current_axis_a=current_axis,
        temperature_axis_c=temperature_axis,
        voltage_table_v=table,
        provenance=_parse_provenance(raw.get("provenance"), name=f"{name}.conduction"),
    )


def _parse_switching(raw: Any, *, name: str, kind: str) -> SwitchingTable | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ModelError(f"{name}.switching must be an object")
    _reject_unknown(
        raw,
        {
            "test_voltage_v",
            "test_temperature_c",
            "current_axis_a",
            "e_on_j",
            "e_off_j",
            "e_rec_j",
            "voltage_scaling",
            "provenance",
        },
        name=f"{name}.switching",
    )
    test_voltage = _finite_number(
        raw.get("test_voltage_v"), name=f"{name}.switching.test_voltage_v"
    )
    if test_voltage <= 0.0:
        raise ModelError(f"{name}.switching.test_voltage_v must be positive")
    test_temperature = _finite_number(
        raw.get("test_temperature_c"), name=f"{name}.switching.test_temperature_c"
    )
    current_axis = _number_list(
        raw.get("current_axis_a"), name=f"{name}.switching.current_axis_a"
    )
    _strictly_increasing(current_axis, name=f"{name}.switching.current_axis_a")
    if current_axis[0] < 0.0:
        raise ModelError(f"{name}.switching.current_axis_a must be non-negative")

    def _energy(field: str) -> tuple[float, ...] | None:
        value = raw.get(field)
        if value is None:
            return None
        energies = _number_list(
            value, name=f"{name}.switching.{field}", minimum_len=len(current_axis)
        )
        if len(energies) != len(current_axis):
            raise ModelError(
                f"{name}.switching.{field} must match the current axis length"
            )
        if any(item < 0.0 for item in energies):
            raise ModelError(f"{name}.switching.{field} must be non-negative")
        return energies

    e_on = _energy("e_on_j")
    e_off = _energy("e_off_j")
    e_rec = _energy("e_rec_j")
    if kind == "switch" and (e_on is None or e_off is None):
        raise ModelError(
            f"{name}.switching for a switch card requires e_on_j and e_off_j"
        )
    if kind == "diode" and e_rec is None:
        raise ModelError(f"{name}.switching for a diode card requires e_rec_j")
    scaling = raw.get("voltage_scaling", "linear")
    if scaling != "linear":
        raise ModelError(
            f"{name}.switching.voltage_scaling supports only 'linear' in v0.1"
        )
    return SwitchingTable(
        test_voltage_v=test_voltage,
        test_temperature_c=test_temperature,
        current_axis_a=current_axis,
        e_on_j=e_on,
        e_off_j=e_off,
        e_rec_j=e_rec,
        voltage_scaling=scaling,
        provenance=_parse_provenance(raw.get("provenance"), name=f"{name}.switching"),
    )


def _parse_foster(raw: Any, *, name: str) -> tuple[FosterStage, ...]:
    if not isinstance(raw, dict):
        raise ModelError(f"{name}.thermal is required and must be an object")
    _reject_unknown(raw, {"foster_stages", "provenance"}, name=f"{name}.thermal")
    stages_raw = raw.get("foster_stages")
    if not isinstance(stages_raw, list) or not 1 <= len(stages_raw) <= _MAX_FOSTER_STAGES:
        raise ModelError(
            f"{name}.thermal.foster_stages must list 1..{_MAX_FOSTER_STAGES} stages"
        )
    stages = []
    for index, stage in enumerate(stages_raw):
        if not isinstance(stage, dict):
            raise ModelError(f"{name}.thermal.foster_stages[{index}] must be an object")
        _reject_unknown(
            stage,
            {"resistance_k_per_w", "time_constant_s"},
            name=f"{name}.thermal.foster_stages[{index}]",
        )
        resistance = _finite_number(
            stage.get("resistance_k_per_w"),
            name=f"{name}.thermal.foster_stages[{index}].resistance_k_per_w",
        )
        time_constant = _finite_number(
            stage.get("time_constant_s"),
            name=f"{name}.thermal.foster_stages[{index}].time_constant_s",
        )
        if resistance <= 0.0 or time_constant <= 0.0:
            raise ModelError(
                f"{name}.thermal.foster_stages[{index}] must have positive "
                "resistance and time constant"
            )
        stages.append(
            FosterStage(resistance_k_per_w=resistance, time_constant_s=time_constant)
        )
    return tuple(stages)


def _parse_device_card(card_id: str, raw: Any) -> DeviceCard:
    name = f"device_cards.{card_id}"
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    _reject_unknown(
        raw,
        {"kind", "provenance", "validity", "conduction", "switching", "thermal"},
        name=name,
    )
    kind = raw.get("kind")
    if kind not in {"switch", "diode"}:
        raise ModelError(f"{name}.kind must be switch or diode")
    provenance = _parse_provenance(raw.get("provenance"), name=name)
    validity = raw.get("validity")
    if not isinstance(validity, dict):
        raise ModelError(f"{name}.validity is required and must be an object")
    _reject_unknown(
        validity,
        {"tj_min_c", "tj_max_c", "current_max_a", "notes"},
        name=f"{name}.validity",
    )
    tj_min = _finite_number(validity.get("tj_min_c"), name=f"{name}.validity.tj_min_c")
    tj_max = _finite_number(validity.get("tj_max_c"), name=f"{name}.validity.tj_max_c")
    current_max = _finite_number(
        validity.get("current_max_a"), name=f"{name}.validity.current_max_a"
    )
    if tj_max <= tj_min or current_max <= 0.0:
        raise ModelError(f"{name}.validity envelope is inconsistent")
    conduction = _parse_conduction(raw.get("conduction"), name=name)
    switching = _parse_switching(raw.get("switching"), name=name, kind=kind)
    thermal_raw = raw.get("thermal")
    foster = _parse_foster(thermal_raw, name=name)
    thermal_provenance = _parse_provenance(
        thermal_raw.get("provenance"), name=f"{name}.thermal"
    )
    return DeviceCard(
        id=card_id,
        kind=kind,
        source=provenance.source,
        validity_tj_min_c=tj_min,
        validity_tj_max_c=tj_max,
        validity_current_max_a=current_max,
        conduction=conduction,
        switching=switching,
        foster_stages=foster,
        thermal_provenance=thermal_provenance,
    )


def _parse_heatsink(card_id: str, raw: Any) -> HeatsinkCard:
    name = f"cooling.heatsinks.{card_id}"
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    _reject_unknown(
        raw,
        {"airflow_axis_m3_per_h", "rth_k_per_w", "pressure_drop_pa", "provenance"},
        name=name,
    )
    airflow = _number_list(raw.get("airflow_axis_m3_per_h"), name=f"{name}.airflow_axis_m3_per_h")
    _strictly_increasing(airflow, name=f"{name}.airflow_axis_m3_per_h")
    if airflow[0] < 0.0:
        raise ModelError(f"{name}.airflow_axis_m3_per_h must be non-negative")
    rth = _number_list(raw.get("rth_k_per_w"), name=f"{name}.rth_k_per_w")
    pressure = _number_list(raw.get("pressure_drop_pa"), name=f"{name}.pressure_drop_pa")
    if len(rth) != len(airflow) or len(pressure) != len(airflow):
        raise ModelError(f"{name} tables must share the airflow axis length")
    if any(value <= 0.0 for value in rth):
        raise ModelError(f"{name}.rth_k_per_w must be positive")
    if any(value < 0.0 for value in pressure):
        raise ModelError(f"{name}.pressure_drop_pa must be non-negative")
    for left, right in zip(pressure, pressure[1:]):
        if right < left:
            raise ModelError(f"{name}.pressure_drop_pa must be non-decreasing")
    return HeatsinkCard(
        id=card_id,
        airflow_axis_m3_per_h=airflow,
        rth_k_per_w=rth,
        pressure_drop_pa=pressure,
        provenance=_parse_provenance(raw.get("provenance"), name=name),
    )


def _parse_fan(card_id: str, raw: Any) -> FanCard:
    name = f"cooling.fans.{card_id}"
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    _reject_unknown(
        raw, {"flow_axis_m3_per_h", "pressure_pa", "provenance"}, name=name
    )
    flow = _number_list(raw.get("flow_axis_m3_per_h"), name=f"{name}.flow_axis_m3_per_h")
    _strictly_increasing(flow, name=f"{name}.flow_axis_m3_per_h")
    if flow[0] < 0.0:
        raise ModelError(f"{name}.flow_axis_m3_per_h must be non-negative")
    pressure = _number_list(raw.get("pressure_pa"), name=f"{name}.pressure_pa")
    if len(pressure) != len(flow):
        raise ModelError(f"{name} tables must share the flow axis length")
    if any(value < 0.0 for value in pressure):
        raise ModelError(f"{name}.pressure_pa must be non-negative")
    for left, right in zip(pressure, pressure[1:]):
        if right > left:
            raise ModelError(f"{name}.pressure_pa must be non-increasing (fan PQ)")
    return FanCard(
        id=card_id,
        flow_axis_m3_per_h=flow,
        pressure_pa=pressure,
        provenance=_parse_provenance(raw.get("provenance"), name=name),
    )


def _parse_cold_plate(card_id: str, raw: Any) -> ColdPlateCard:
    name = f"cooling.cold_plates.{card_id}"
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    _reject_unknown(
        raw,
        {"flow_axis_l_per_min", "rth_k_per_w", "coolant_inlet_c", "provenance"},
        name=name,
    )
    flow = _number_list(raw.get("flow_axis_l_per_min"), name=f"{name}.flow_axis_l_per_min")
    _strictly_increasing(flow, name=f"{name}.flow_axis_l_per_min")
    if flow[0] < 0.0:
        raise ModelError(f"{name}.flow_axis_l_per_min must be non-negative")
    rth = _number_list(raw.get("rth_k_per_w"), name=f"{name}.rth_k_per_w")
    if len(rth) != len(flow):
        raise ModelError(f"{name} tables must share the flow axis length")
    if any(value <= 0.0 for value in rth):
        raise ModelError(f"{name}.rth_k_per_w must be positive")
    for left, right in zip(rth, rth[1:]):
        if right > left:
            raise ModelError(
                f"{name}.rth_k_per_w must be non-increasing with flow"
            )
    inlet = raw.get("coolant_inlet_c")
    if isinstance(inlet, bool) or not isinstance(inlet, (int, float)):
        raise ModelError(f"{name}.coolant_inlet_c is required and must be a number")
    return ColdPlateCard(
        id=card_id,
        flow_axis_l_per_min=flow,
        rth_k_per_w=rth,
        coolant_inlet_c=float(inlet),
        provenance=_parse_provenance(raw.get("provenance"), name=name),
    )


def _parse_tim(card_id: str, raw: Any) -> TimCard:
    name = f"cooling.tims.{card_id}"
    if not isinstance(raw, dict):
        raise ModelError(f"{name} must be an object")
    _reject_unknown(raw, {"rth_k_per_w", "provenance"}, name=name)
    rth = _finite_number(raw.get("rth_k_per_w"), name=f"{name}.rth_k_per_w")
    if rth <= 0.0:
        raise ModelError(f"{name}.rth_k_per_w must be positive")
    return TimCard(
        id=card_id,
        rth_k_per_w=rth,
        provenance=_parse_provenance(raw.get("provenance"), name=name),
    )


def parse_loss_sections(
    raw_cards: Any,
    raw_cooling: Any,
    raw_analysis: Any,
) -> LossAnalysisSpec | None:
    """Parse the three optional loss/thermal sections; all-or-nothing, fail closed."""

    if raw_analysis is None:
        if raw_cards or raw_cooling:
            raise ModelError(
                "device_cards/cooling sections require a loss_analysis section "
                "that binds them to devices"
            )
        return None
    if not isinstance(raw_analysis, dict):
        raise ModelError("loss_analysis must be an object")
    if not isinstance(raw_cards, dict) or not raw_cards:
        raise ModelError("loss_analysis requires a non-empty device_cards section")

    cards = {
        _require_model_text(card_id, name="device_cards key"): _parse_device_card(
            card_id, raw
        )
        for card_id, raw in raw_cards.items()
    }

    heatsinks: dict[str, HeatsinkCard] = {}
    fans: dict[str, FanCard] = {}
    tims: dict[str, TimCard] = {}
    cold_plates: dict[str, ColdPlateCard] = {}
    if raw_cooling is not None:
        if not isinstance(raw_cooling, dict):
            raise ModelError("cooling must be an object")
        _reject_unknown(
            raw_cooling,
            {"heatsinks", "fans", "tims", "cold_plates"},
            name="cooling",
        )
        for card_id, raw in (raw_cooling.get("heatsinks") or {}).items():
            heatsinks[card_id] = _parse_heatsink(card_id, raw)
        for card_id, raw in (raw_cooling.get("fans") or {}).items():
            fans[card_id] = _parse_fan(card_id, raw)
        for card_id, raw in (raw_cooling.get("tims") or {}).items():
            tims[card_id] = _parse_tim(card_id, raw)
        for card_id, raw in (raw_cooling.get("cold_plates") or {}).items():
            cold_plates[card_id] = _parse_cold_plate(card_id, raw)

    _reject_unknown(
        raw_analysis,
        {"ambient_c", "tj_evaluation_c", "devices"},
        name="loss_analysis",
    )
    ambient = _finite_number(raw_analysis.get("ambient_c"), name="loss_analysis.ambient_c")
    tj_evaluation = _finite_number(
        raw_analysis.get("tj_evaluation_c"), name="loss_analysis.tj_evaluation_c"
    )
    raw_devices = raw_analysis.get("devices")
    if not isinstance(raw_devices, dict) or not raw_devices:
        raise ModelError("loss_analysis.devices must be a non-empty object")
    bindings = []
    for component_id, raw in raw_devices.items():
        name = f"loss_analysis.devices.{component_id}"
        if not isinstance(raw, dict):
            raise ModelError(f"{name} must be an object")
        _reject_unknown(
            raw, {"card", "commutation_voltage", "cooling"}, name=name
        )
        card_id = _require_model_text(raw.get("card"), name=f"{name}.card")
        if card_id not in cards:
            raise ModelError(f"{name}.card references unknown card {card_id!r}")
        commutation = raw.get("commutation_voltage")
        if commutation is not None:
            commutation = _require_model_text(
                commutation, name=f"{name}.commutation_voltage"
            )
        if cards[card_id].switching is not None and commutation is None:
            raise ModelError(
                f"{name}: a card with switching tables requires "
                "commutation_voltage"
            )
        cooling_raw = raw.get("cooling") or {}
        if not isinstance(cooling_raw, dict):
            raise ModelError(f"{name}.cooling must be an object")
        _reject_unknown(
            cooling_raw,
            {"tim", "heatsink", "fan", "cold_plate", "coolant_flow_l_per_min"},
            name=f"{name}.cooling",
        )
        tim_id = cooling_raw.get("tim")
        heatsink_id = cooling_raw.get("heatsink")
        fan_id = cooling_raw.get("fan")
        plate_id = cooling_raw.get("cold_plate")
        coolant_flow = cooling_raw.get("coolant_flow_l_per_min")
        if tim_id is not None and tim_id not in tims:
            raise ModelError(f"{name}.cooling.tim references unknown TIM {tim_id!r}")
        if heatsink_id is not None and heatsink_id not in heatsinks:
            raise ModelError(
                f"{name}.cooling.heatsink references unknown heatsink {heatsink_id!r}"
            )
        if heatsink_id is not None and fan_id is None:
            # Without a fan the estimate uses the heatsink's zero-airflow point;
            # a table starting at forced air would pass forced-air Rth off as
            # natural convection.
            first_airflow = heatsinks[heatsink_id].airflow_axis_m3_per_h[0]
            if first_airflow != 0.0:
                raise ModelError(
                    f"{name}.cooling.heatsink {heatsink_id!r} has no fan, so it needs a "
                    "natural-convection point: its airflow_axis_m3_per_h starts at "
                    f"{first_airflow:g}, not 0. Add the 0 m3/h row or bind a fan"
                )
        if fan_id is not None:
            if heatsink_id is None:
                raise ModelError(
                    f"{name}.cooling.fan requires a heatsink card (the operating "
                    "point is the fan-PQ/heatsink-impedance intersection)"
                )
            if fan_id not in fans:
                raise ModelError(f"{name}.cooling.fan references unknown fan {fan_id!r}")
        if plate_id is not None:
            # A device is air-cooled or liquid-cooled, not both: allowing both
            # would silently add two spreader resistances in series.
            if heatsink_id is not None or fan_id is not None:
                raise ModelError(
                    f"{name}.cooling: cold_plate excludes heatsink/fan — declare "
                    "one cooling path per device"
                )
            if plate_id not in cold_plates:
                raise ModelError(
                    f"{name}.cooling.cold_plate references unknown cold plate "
                    f"{plate_id!r}"
                )
            if (
                isinstance(coolant_flow, bool)
                or not isinstance(coolant_flow, (int, float))
                or float(coolant_flow) <= 0.0
            ):
                raise ModelError(
                    f"{name}.cooling.coolant_flow_l_per_min is required with a "
                    "cold plate and must be positive (the pump curve is NOT "
                    "modelled, so the operating flow is declared)"
                )
            coolant_flow = float(coolant_flow)
        elif coolant_flow is not None:
            raise ModelError(
                f"{name}.cooling.coolant_flow_l_per_min requires a cold_plate"
            )
        bindings.append(
            DeviceLossBinding(
                component_id=component_id,
                card_id=card_id,
                commutation_voltage=commutation,
                cooling=CoolingPath(
                    tim=tim_id,
                    heatsink=heatsink_id,
                    fan=fan_id,
                    cold_plate=plate_id,
                    coolant_flow_l_per_min=coolant_flow,
                ),
            )
        )

    for card in cards.values():
        if not (
            card.validity_tj_min_c <= tj_evaluation <= card.validity_tj_max_c
        ):
            raise ModelError(
                f"loss_analysis.tj_evaluation_c={tj_evaluation:g} is outside the "
                f"validity envelope of card {card.id!r} "
                f"[{card.validity_tj_min_c:g}, {card.validity_tj_max_c:g}]"
            )

    return LossAnalysisSpec(
        ambient_c=ambient,
        tj_evaluation_c=tj_evaluation,
        devices=tuple(bindings),
        cards=cards,
        heatsinks=heatsinks,
        fans=fans,
        tims=tims,
        cold_plates=cold_plates,
    )
