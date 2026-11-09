# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Efficiency across an operating range, from declared loss models only.

An efficiency figure is a claim about losses, so this producer refuses to
compute one for a model that declares none: an ideal-switch kernel would
report 100 % and be arithmetically right and engineering-useless. A model
must carry `loss_analysis` (device cards with provenance) before its
efficiency can be reported at all.

The map is a projection over runs someone else already retained: the
bounded parameter sweep produces one verified bundle per operating point,
and this producer reads each point's `losses-thermal.json` and
`physics-audit.json` and reports

    P_loss  = semiconductor losses from the declared cards
              + power absorbed by explicitly modelled resistances that the
                request names as parasitic (none by default)
    P_out   = P_delivered_by_sources - P_loss
    eta     = P_out / P_delivered_by_sources

Every point carries the counters that say how far the numbers leaned on the
cards' validity: extrapolated lookups, junction temperatures beyond the card
envelope, and the gap between the evaluation temperature and the computed
junction. A point with extrapolated lookups is reported, never silently
smoothed — and never quietly dropped from the map.

WHAT THIS IS NOT. A single efficiency number per point, not a band: the
band comes from running this map at tolerance corners, which is a separate
declared step. Losses absent from the cards (magnetics, gate drive, control
supply, snubbers, busbars) are absent from the result, and the artifact says
so rather than implying completeness.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EFFICIENCY_MAP_SCHEMA = "pear-sim/efficiency-map/v0.1"
EFFICIENCY_MAP_KIND = "efficiency-map-projection"

MAX_POINTS = 256


class EfficiencyMapError(ValueError):
    """Raised when a retained sweep cannot support an efficiency projection."""

    code = "evidence.efficiency_map_invalid"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise EfficiencyMapError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EfficiencyMapError(f"cannot read {label} {path.name}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EfficiencyMapError(f"{label} {path.name} must contain an object")
    return payload


def _counter_total(value: Any) -> int:
    """Sum a lookup counter that may be a scalar or a per-table mapping."""

    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, dict):
        return sum(_counter_total(item) for item in value.values())
    raise EfficiencyMapError(f"unexpected extrapolation counter shape: {value!r}")


def _uninstrumented_devices(
    resolved: dict[str, Any], losses: dict[str, Any]
) -> list[str]:
    """Switching devices in the model that carry no loss card.

    Their losses are simply absent from the total, so the efficiency of a
    partly instrumented converter is OPTIMISTIC. Reporting which devices were
    left out is the difference between a bounded claim and a flattering one.
    """

    instrumented = {
        str(device.get("component_id"))
        for device in losses.get("devices", ())
        if isinstance(device, dict)
    }
    components = (resolved.get("physics") or {}).get("components")
    if components is None:
        raise EfficiencyMapError(
            "resolved model carries no physics.components, so the instrumented "
            "share of the switching devices cannot be established"
        )
    missing: list[str] = []
    for component in components or ():
        if not isinstance(component, dict):
            continue
        if component.get("type") not in {"ideal_switch", "diode"}:
            continue
        identifier = str(component.get("id"))
        if identifier not in instrumented:
            missing.append(identifier)
    return sorted(missing)


def _device_loss_w(losses: dict[str, Any], window_s: float) -> tuple[float, dict[str, Any]]:
    """Total semiconductor loss and the counters that qualify it."""

    total = losses.get("total_mean_power_w")
    if isinstance(total, bool) or not isinstance(total, (int, float)):
        raise EfficiencyMapError("losses-thermal.json lacks total_mean_power_w")
    extrapolated = 0
    beyond_envelope: list[str] = []
    hottest = None
    for device in losses.get("devices", ()):
        if not isinstance(device, dict):
            continue
        conduction = device.get("conduction") or {}
        switching = device.get("switching") or {}
        # The conduction counter is a scalar, the switching one is per energy
        # table; both are summed rather than reduced to a boolean, so the
        # artifact says HOW MUCH a point leaned outside its cards.
        extrapolated += _counter_total(conduction.get("extrapolated_lookups"))
        extrapolated += _counter_total(switching.get("extrapolated_lookups"))
        thermal = device.get("thermal") or {}
        if thermal.get("validity_violated"):
            beyond_envelope.append(str(device.get("component_id")))
        junction = thermal.get("tj_max_c")
        if isinstance(junction, (int, float)) and not isinstance(junction, bool):
            hottest = junction if hottest is None else max(hottest, float(junction))
    return float(total), {
        "extrapolated_card_lookups": extrapolated,
        "devices_beyond_validity": sorted(beyond_envelope),
        "hottest_junction_c": hottest,
        "window_s": window_s,
    }


def build_efficiency_map(
    sweep_dir: str | Path,
    *,
    parasitic_components: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Project a retained parameter sweep into an efficiency map.

    ``sweep_dir`` is the output directory of :func:`run_parameter_sweep`;
    every retained point must carry a losses artifact, which is exactly the
    condition that makes an efficiency claim admissible.
    """

    directory = Path(sweep_dir).resolve()
    summary = _read_json(directory / "sweep-summary.json", "sweep summary")
    points = summary.get("points")
    if not isinstance(points, list) or not points:
        raise EfficiencyMapError("sweep summary carries no points")
    if len(points) > MAX_POINTS:
        raise EfficiencyMapError(
            f"sweep has {len(points)} points; the portable projection limit is "
            f"{MAX_POINTS}"
        )

    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            raise EfficiencyMapError(f"point {index} must be an object")
        if point.get("execution_status") != "pass":
            skipped.append(
                {
                    "index": index,
                    "value_si": point.get("parameter_value_si"),
                    "reason": f"execution_status={point.get('execution_status')!r}",
                }
            )
            continue
        artifacts = point.get("artifacts") or {}
        run_record = artifacts.get("run") or {}
        run_path = run_record.get("path")
        if not isinstance(run_path, str):
            raise EfficiencyMapError(f"point {index} lacks a retained run path")
        run_dir = (directory / run_path).resolve().parent

        losses_path = run_dir / "losses-thermal.json"
        if not losses_path.is_file():
            raise EfficiencyMapError(
                f"point {index} has no losses-thermal.json: efficiency is a claim "
                "about losses, so a model without a declared loss model (device "
                "cards) cannot report one"
            )
        losses = _read_json(losses_path, "losses artifact")
        audit = _read_json(run_dir / "physics-audit.json", "physics audit")
        resolved = _read_json(run_dir / "model.resolved.json", "resolved model")
        uninstrumented = _uninstrumented_devices(resolved, losses)
        window = audit.get("full") or {}
        duration = window.get("duration_s")
        delivered = window.get("source_delivered_energy_j")
        if not isinstance(duration, (int, float)) or not duration:
            raise EfficiencyMapError(f"point {index} has no usable audit window")
        if not isinstance(delivered, (int, float)):
            raise EfficiencyMapError(
                f"point {index} audit lacks source_delivered_energy_j"
            )
        input_w = float(delivered) / float(duration)
        semiconductor_w, qualifiers = _device_loss_w(losses, float(duration))
        qualifiers["uninstrumented_switching_devices"] = uninstrumented

        parasitic_w = 0.0
        component_energy = window.get("component_absorbed_energy_j") or {}
        for component_id in parasitic_components:
            if component_id not in component_energy:
                raise EfficiencyMapError(
                    f"point {index}: parasitic component {component_id!r} is not in "
                    "the audit's component map"
                )
            parasitic_w += float(component_energy[component_id]) / float(duration)

        loss_w = semiconductor_w + parasitic_w
        if input_w <= 0.0:
            skipped.append(
                {
                    "index": index,
                    "value_si": point.get("parameter_value_si"),
                    "reason": "sources delivered no net power over the window",
                }
            )
            continue
        output_w = input_w - loss_w
        rows.append(
            {
                "index": index,
                "parameter": summary.get("parameter"),
                "value_si": point.get("parameter_value_si"),
                "input_power_w": input_w,
                "loss_power_w": loss_w,
                "semiconductor_loss_w": semiconductor_w,
                "parasitic_loss_w": parasitic_w,
                "output_power_w": output_w,
                "efficiency": output_w / input_w,
                "qualifiers": qualifiers,
            }
        )

    if not rows:
        raise EfficiencyMapError("no retained point could support an efficiency figure")

    best = max(rows, key=lambda row: row["efficiency"])
    worst = min(rows, key=lambda row: row["efficiency"])
    leaning = [row["index"] for row in rows if row["qualifiers"]["extrapolated_card_lookups"]]
    uninstrumented = sorted(
        {
            device
            for row in rows
            for device in row["qualifiers"]["uninstrumented_switching_devices"]
        }
    )
    beyond = [row["index"] for row in rows if row["qualifiers"]["devices_beyond_validity"]]
    return {
        "schema": EFFICIENCY_MAP_SCHEMA,
        "kind": EFFICIENCY_MAP_KIND,
        "source_sweep": {
            "summary": "sweep-summary.json",
            "parameter": summary.get("parameter"),
            "points_retained": len(points),
        },
        "definition": (
            "eta = (P_delivered_by_sources - P_loss) / P_delivered_by_sources, "
            "with P_loss the semiconductor losses from the declared device "
            "cards plus any resistances the request names as parasitic"
        ),
        "points": rows,
        "skipped_points": skipped,
        "best": {"index": best["index"], "efficiency": best["efficiency"]},
        "worst": {"index": worst["index"], "efficiency": worst["efficiency"]},
        "points_with_extrapolated_lookups": leaning,
        "uninstrumented_switching_devices": uninstrumented,
        "efficiency_completeness": (
            "every switching device carries a loss card"
            if not uninstrumented
            else f"OPTIMISTIC: {len(uninstrumented)} switching device(s) carry no "
            "loss card, so their losses are absent from the total"
        ),
        "points_beyond_card_validity": beyond,
        "claim_boundary": (
            "One figure per operating point, NOT a band — the band comes from "
            "running this map at tolerance corners, which is a separate step. "
            "Losses the cards do not carry are absent from the result: "
            "magnetics, gate drive, control supply, snubbers and busbars. A "
            "point whose card lookups were extrapolated is listed rather than "
            "smoothed, and no point is dropped silently. A converter whose "
            "switching devices are only partly instrumented yields an "
            "OPTIMISTIC efficiency, and the devices left out are named."
        ),
    }
