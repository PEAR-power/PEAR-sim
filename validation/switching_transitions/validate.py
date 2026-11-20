# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Switching transitions: validation against published measurements + sensitivity sweep.

Run:  uv run python validation/switching_transitions/validate.py
Writes results.json and RESULTS.md next to this file (both GENERATED: edit the
script, never the outputs). tests/test_switching_transitions_validation.py
re-runs the fast part and fails when the committed files are stale.

Every MEASURED number below is a third-party bench measurement, cited to page
and figure/table; nothing PEARSim computes is called measured. Numbers read off
a plot carry the reading precision next to them.

Parts:
  A. ZVS boundary and transition time — Barba et al., Energies 2024 (EPC2065).
  B. Turn-off overshoot vs loop inductance — onsemi AND90238/D Table 7.
  C. Sensitivity of the shipped examples to the Coss band (datasheet typ -> max)
     and to the loop inductance, re-post-processing ONE recorded run per example
     (exact: the transition never feeds back into the circuit).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

from pear_sim.analysis import switching_transitions as st  # noqa: E402
from pear_sim.analysis.switching_transitions import (  # noqa: E402
    CossCurve,
    build_switching_transitions_observation,
    ramp_overshoot,
    simulate_transition,
)
from pear_sim.spec.device_cards import CardProvenance  # noqa: E402
from pear_sim.spec.switching_transitions import CossCard  # noqa: E402

# --- sources ---------------------------------------------------------------------------

EPC2065_CARD = CossCard(
    id="epc2065",
    voltage_axis_v=(0.0, 10.0, 20.0, 40.0, 60.0, 80.0),
    coss_f=(1.34e-9, 1.23e-9, 0.64e-9, 0.535e-9, 0.49e-9, 0.445e-9),
    qoss_c=None,
    provenance=CardProvenance(
        "EPC2065 datasheet (EPC, 2021-06-11), Coss typical read from plot (±20 pF)", 3, "Fig. 5a"
    ),
)
# Datasheet TABLE (p. 1): QOSS at 40 V = 33 nC typ, 50 nC max. The band of the
# sweep is this ratio — a source-stated spread, not a guess.
EPC2065_QOSS_40V_TYP_C = 33e-9
EPC2065_QOSS_40V_MAX_C = 50e-9

BARBA = {
    "citation": (
        "V. Barba, S. Musumeci, F. Stella, F. Mandrile, M. Palma, 'Investigation of Dead "
        "Time Losses in Inverter Switching Leg Operation: GaN FET vs. MOSFET Comparison', "
        "Energies 17(15) 3855, 2024, doi:10.3390/en17153855 (CC-BY)"
    ),
    "device": "EPC2065 half-bridge, 48 V, high-side turn-off with Ia > 0, tdt = 150 ns",
    # Stated in the text below Fig. 4 (PDF p. 5): ZVS for the GaN FET "between Ia = 0.5 A
    # and Ia = 1 A"; partial hard switching (PHS) at 0.5 A.
    "stated_zvs_boundary_a": (0.5, 1.0),
    "dead_time_s": 150e-9,
    # Read off Fig. 4(a) (10 V/div, 50 ns/div; about ±5 ns, plus a fall-onset
    # ambiguity of 10-20 ns): every trace starts falling at ~26 ns, Q_LS turns on
    # at ~169 ns, so the node has ~143 ns to swing.
    "effective_window_s": 143e-9,
    "plot_fall_time_s": {2.0: 37e-9, 1.5: 62e-9, 1.0: 100e-9},
    "plot_fall_precision_s": 5e-9,
    "plot_residual_at_0p5a_v": 14.0,
    "plot_residual_precision_v": 1.5,
    "bus_v": 48.0,
}

ONSEMI = {
    "citation": (
        "onsemi, 'Impact of Stray Inductance on EliteSiC Power and VE-Trac IGBT Module's "
        "Switching Characteristics', AND90238/D Rev. 1, 2025-03, Table 7 (p. 6), test "
        "conditions p. 5, loop setups Table 1 (p. 2)"
    ),
    "device": "NVXR17S90M2SPC SiC half-bridge module, low-side DPT, 400 V, 600 A, RG(off) = 1.8 ohm",
    "bus_v": 400.0,
    "current_a": 600.0,
    # Table 7: loop inductance (nH) -> (-di/dt 80-20 % in A/ns, VDS,peak in V)
    "table7": {23.0: (10.4, 647.0), 30.0: (8.9, 717.0), 37.0: (8.5, 774.0)},
    # Datasheet nvxr17s90m2spc-d Rev. 6: Coss = 3.3 nF typ at 400 V (table). Above
    # 400 V the curve is not published: held (and flagged) — an over-estimate of C.
    "coss_at_bus_f": 3.3e-9,
}


# --- A. ZVS ------------------------------------------------------------------------------


def _node(curve: CossCurve, bus: float):
    return st._node_capacitance(curve, bus, 0.0)


def part_a() -> dict:
    bus = BARBA["bus_v"]
    threshold = 0.01 * bus
    scales = {
        "typ": 1.0,
        "max": EPC2065_QOSS_40V_MAX_C / EPC2065_QOSS_40V_TYP_C,
    }
    rows = []
    for current in (0.5, 1.0, 1.5, 2.0):
        row = {"current_a": current}
        for label, scale in scales.items():
            card = replace(EPC2065_CARD, coss_f=tuple(scale * c for c in EPC2065_CARD.coss_f))
            curve = CossCurve(card)
            for window_label, window in (
                ("stated_dead_time", BARBA["dead_time_s"]),
                ("plot_window", BARBA["effective_window_s"]),
            ):
                outcome = simulate_transition([_node(curve, bus)], bus, current, window, threshold)
                row[f"{label}_{window_label}"] = {
                    "zvs": outcome.voltage_at_turn_on_v[0] <= threshold * (1 + 1e-9),
                    "voltage_at_turn_on_v": outcome.voltage_at_turn_on_v[0],
                    "minimum_dead_time_s": outcome.minimum_dead_time_s,
                }
        measured = BARBA["plot_fall_time_s"].get(current)
        if measured is not None:
            row["measured_fall_time_s"] = measured
        rows.append(row)
    low, high = BARBA["stated_zvs_boundary_a"]

    def boundary_agrees(label: str, window_label: str) -> bool:
        by_current = {row["current_a"]: row[f"{label}_{window_label}"]["zvs"] for row in rows}
        return (not by_current[low]) and by_current[high]

    agreement = {
        f"{label}_{window}": boundary_agrees(label, window)
        for label in scales
        for window in ("stated_dead_time", "plot_window")
    }
    return {
        "source": BARBA["citation"],
        "device": BARBA["device"],
        "card_provenance": EPC2065_CARD.provenance.as_dict(),
        "coss_band_scales": scales,
        "coss_band_source": "EPC2065 datasheet table p. 1: QOSS(40 V) 33 nC typ, 50 nC max",
        "qoss_40v_from_card_c": CossCurve(EPC2065_CARD).charge(40.0),
        "rows": rows,
        "zvs_boundary_agrees": agreement,
        "plot_residual_at_0p5a_v": BARBA["plot_residual_at_0p5a_v"],
        "plot_reading": {
            "time_precision_s": BARBA["plot_fall_precision_s"],
            "onset_ambiguity_s": [10e-9, 20e-9],
            "residual_precision_v": BARBA["plot_residual_precision_v"],
        },
    }


# --- B. overshoot ------------------------------------------------------------------------


def part_b() -> dict:
    bus, current = ONSEMI["bus_v"], ONSEMI["current_a"]
    card = CossCard(
        "nvxr17s90m2spc",
        (0.0, bus),
        (ONSEMI["coss_at_bus_f"], ONSEMI["coss_at_bus_f"]),
        None,
        CardProvenance("NVXR17S90M2SPC datasheet Rev. 6: Coss 3.3 nF typ at 400 V (table)"),
    )
    curve = CossCurve(card)
    rows = []
    for loop_nh, (di_dt_a_per_ns, vpeak_measured) in ONSEMI["table7"].items():
        loop = loop_nh * 1e-9
        fall = current / (di_dt_a_per_ns * 1e9)
        conventional = bus + loop * current / fall
        ramp = bus + ramp_overshoot(curve, bus, current, loop, fall)
        rows.append(
            {
                "loop_inductance_nh": loop_nh,
                "measured_di_dt_a_per_ns": di_dt_a_per_ns,
                "measured_vpeak_v": vpeak_measured,
                "vpeak_conventional_v": conventional,
                "conventional_error": (conventional - vpeak_measured) / vpeak_measured,
                "vpeak_lossless_ramp_v": ramp,
                "lossless_ramp_error": (ramp - vpeak_measured) / vpeak_measured,
                "measured_within_bracket": conventional <= vpeak_measured <= ramp,
            }
        )
    return {
        "source": ONSEMI["citation"],
        "device": ONSEMI["device"],
        "note": (
            "di/dt is the table's 80-20 % average; t_f = I/(di/dt) turns it into the "
            "declared linear fall. Coss above 400 V is not published and is held at the "
            "400 V table value (extrapolated lookup). The lossless ramp keeps the "
            "Lsigma-Coss ringing undamped."
        ),
        "rows": rows,
    }


# --- C. sweep --------------------------------------------------------------------------------


EXAMPLES = {
    "synchronous_buck_gan_transition": {"loop_nh": [0.4, 1.0, 1.6]},
    "dab_phase_shift_zvs": {"loop_nh": [2.0, 5.0, 10.0]},
    "llc_resonant_zvs": {"loop_nh": []},
}
# Loop band of the buck: 0.4 nH (EPC "optimal" layout) to 1.6 nH (EPC WP010 p. 2,
# conventional layout), both from EPC's layout study; DAB and LLC cards are
# SYNTHETIC, their loop band is illustrative and says so.


def _scaled_spec(spec, scale: float, loop_h: float | None):
    cards = {
        key: replace(card, coss_f=None if card.coss_f is None else tuple(scale * c for c in card.coss_f),
                     qoss_c=None if card.qoss_c is None else tuple(scale * q for q in card.qoss_c))
        for key, card in spec.cards.items()
    }
    legs = tuple(
        replace(leg, loop_inductance_h=loop_h) if (loop_h is not None and leg.loop_inductance_h is not None) else leg
        for leg in spec.legs
    )
    return replace(spec, cards=cards, legs=legs)


def _digest(observation: dict) -> dict:
    out = {}
    for leg in observation["legs"]:
        summary = leg["summary"]
        out[leg["leg"]] = {
            "by_direction": {
                direction: {
                    "zvs_edges": block["zvs_edges"],
                    "edges": block["edges"],
                    "worst_voltage_at_turn_on_v": block["worst_voltage_at_turn_on_v"],
                    "max_minimum_dead_time_s": block["max_minimum_dead_time_s"],
                }
                for direction, block in summary.get("by_direction", {}).items()
            },
            "capacitive_turn_on_power_w": summary.get("capacitive_turn_on_power_w"),
            "max_overshoot": summary.get("max_overshoot"),
        }
    return out


def part_c() -> dict:
    from pear_sim.core.solver import HybridSimulator
    from pear_sim.spec.model import load_model

    scale_max = EPC2065_QOSS_40V_MAX_C / EPC2065_QOSS_40V_TYP_C
    scales = [1.0, 1.25, scale_max]
    results = {}
    for name, options in EXAMPLES.items():
        model = load_model(ROOT / "examples" / f"{name}.yaml")
        result = HybridSimulator(model).simulate()
        base_spec = model.switching_transitions
        cases = []
        loops = options["loop_nh"] or [None]
        for scale in scales:
            for loop_nh in loops:
                spec = _scaled_spec(base_spec, scale, None if loop_nh is None else loop_nh * 1e-9)
                result.model = replace(model, switching_transitions=spec)
                cases.append(
                    {
                        "coss_scale": scale,
                        "loop_inductance_nh": loop_nh,
                        "digest": _digest(build_switching_transitions_observation(result)),
                    }
                )
        result.model = model
        results[name] = {"cases": cases, "robustness": _robustness(cases)}
    return {
        "coss_band": {"scales": scales, "source": "EPC2065 QOSS max/typ at 40 V (datasheet p. 1) = 50/33"},
        "examples": results,
    }


def _robustness(cases: list[dict]) -> dict:
    """Per leg and direction: is the ZVS verdict the same across the whole sweep?"""

    verdicts: dict[str, dict[str, set]] = {}
    ranges: dict[str, dict[str, list]] = {}
    for case in cases:
        for leg, block in case["digest"].items():
            for direction, values in block["by_direction"].items():
                fraction = values["zvs_edges"] / values["edges"] if values["edges"] else None
                verdicts.setdefault(leg, {}).setdefault(direction, set()).add(fraction)
                ranges.setdefault(leg, {}).setdefault(direction, []).append(values["worst_voltage_at_turn_on_v"])
    return {
        leg: {
            direction: {
                "zvs_fraction_values": sorted(v for v in fractions if v is not None),
                "robust": len(fractions) == 1,
                "worst_voltage_at_turn_on_range_v": [min(ranges[leg][direction]), max(ranges[leg][direction])],
            }
            for direction, fractions in by_direction.items()
        }
        for leg, by_direction in verdicts.items()
    }


# --- rendering -------------------------------------------------------------------------------


def _ns(value):
    return "—" if value is None else f"{value * 1e9:.1f}"


def render(results: dict) -> str:
    a, b = results["validation"]["zvs"], results["validation"]["overshoot"]
    lines = [
        "# Switching transitions — validation and sensitivity results",
        "",
        "GENERATED by `validation/switching_transitions/validate.py`; do not edit by hand.",
        "MEASURED = a third-party bench value, cited. Everything else is computed by PEARSim.",
        "",
        "## A. ZVS boundary and transition time (EPC2065, 48 V)",
        "",
        f"Source: {a['source']}. {a['device']}.",
        f"Card: {a['card_provenance']['source']}, p. {a['card_provenance']['page']}, "
        f"{a['card_provenance']['figure']}; Qoss(40 V) of the card = "
        f"{a['qoss_40v_from_card_c'] * 1e9:.1f} nC (datasheet table: 33 nC typ, 50 nC max).",
        "",
        "| Ia (A) | MEASURED fall time (plot, ±5 ns) | PEARSim t_min, Coss typ (ns) | PEARSim t_min, Coss max (ns) | PEARSim ZVS in 143 ns window, typ / max |",
        "|---|---|---|---|---|",
    ]
    for row in a["rows"]:
        measured = row.get("measured_fall_time_s")
        typ, top = row["typ_plot_window"], row["max_plot_window"]
        lines.append(
            f"| {row['current_a']:g} | {_ns(measured) if measured else 'PHS (partial)'} | "
            f"{_ns(typ['minimum_dead_time_s'])} | {_ns(top['minimum_dead_time_s'])} | "
            f"{'yes' if typ['zvs'] else 'no ({:.1f} V)'.format(typ['voltage_at_turn_on_v'])} / "
            f"{'yes' if top['zvs'] else 'no ({:.1f} V)'.format(top['voltage_at_turn_on_v'])} |"
        )
    agree = a["zvs_boundary_agrees"]
    lines += [
        "",
        "Stated (text, p. 5): ZVS between Ia = 0.5 A and 1 A at tdt = 150 ns. PEARSim boundary "
        "agrees: " + ", ".join(f"{key} = {'yes' if value else 'NO'}" for key, value in agree.items()) + ".",
        f"MEASURED residual at 0.5 A (plot): ~{a['plot_residual_at_0p5a_v']:g} V ± "
        f"{a['plot_reading']['residual_precision_v']:g} V.",
        "",
        "## B. Turn-off overshoot vs loop inductance (SiC module, 400 V / 600 A)",
        "",
        f"Source: {b['source']}. {b['device']}. {b['note']}",
        "",
        "| Lσ (nH) | MEASURED −di/dt (A/ns) | MEASURED V_peak (V) | V + Lσ·di/dt (V) | error | lossless ramp (V) | error | measured inside bracket |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in b["rows"]:
        lines.append(
            f"| {row['loop_inductance_nh']:g} | {row['measured_di_dt_a_per_ns']:g} | "
            f"{row['measured_vpeak_v']:g} | {row['vpeak_conventional_v']:.0f} | "
            f"{100 * row['conventional_error']:+.1f} % | {row['vpeak_lossless_ramp_v']:.0f} | "
            f"{100 * row['lossless_ramp_error']:+.1f} % | {'yes' if row['measured_within_bracket'] else 'no'} |"
        )
    sweep = results.get("sensitivity")
    if sweep:
        lines += [
            "",
            "## C. Sensitivity of the shipped examples",
            "",
            f"Coss scaled by {', '.join(f'{s:.2f}' for s in sweep['coss_band']['scales'])} "
            f"({sweep['coss_band']['source']}); loop inductance swept where declared.",
            "",
            "| example | leg | direction | ZVS fraction across sweep | robust | worst turn-on voltage range (V) |",
            "|---|---|---|---|---|---|",
        ]
        for name, data in sweep["examples"].items():
            for leg, by_direction in data["robustness"].items():
                for direction, value in by_direction.items():
                    fractions = ", ".join(f"{f:.2f}" for f in value["zvs_fraction_values"])
                    low, high = value["worst_voltage_at_turn_on_range_v"]
                    lines.append(
                        f"| {name} | {leg} | {direction} | {fractions} | "
                        f"{'yes' if value['robust'] else 'NO — range only'} | {low:.2f} – {high:.2f} |"
                    )
        lines += ["", "Overshoot ranges (max over edges):", ""]
        for name, data in sweep["examples"].items():
            for case in data["cases"]:
                for leg, block in case["digest"].items():
                    if block.get("max_overshoot"):
                        peaks = ", ".join(f"{k} = {v:.1f} V" for k, v in block["max_overshoot"].items())
                        lines.append(
                            f"- {name} / {leg}: Coss ×{case['coss_scale']:.2f}, "
                            f"Lσ = {case['loop_inductance_nh']} nH → {peaks}"
                        )
    return "\n".join(lines) + "\n"


def compute(include_sweep: bool = True) -> dict:
    results = {
        "schema": "pear-sim/switching-transitions-validation/v0.1",
        "validation": {"zvs": part_a(), "overshoot": part_b()},
    }
    if include_sweep:
        results["sensitivity"] = part_c()
    return results


def _jsonable(value):
    return json.loads(json.dumps(value, default=lambda o: sorted(o) if isinstance(o, set) else str(o)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-sweep", action="store_true", help="validation only (fast)")
    args = parser.parse_args(argv)
    results = _jsonable(compute(include_sweep=not args.no_sweep))
    if args.no_sweep:
        print(json.dumps(results["validation"], indent=2))
        return 0
    (HERE / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "RESULTS.md").write_text(render(results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
