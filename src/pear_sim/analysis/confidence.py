"""Per-run confidence card computed from validation coverage.

The card maps the FEATURES a model actually uses onto the versioned
validation matrix (packaged copy of validation/status.json) and reports the
weakest link. Rules:

- a missing oracle yields ``not_run`` — never a positive result;
- ``partial`` always carries the exact claim boundary and the list of
  implemented evidence cases, so every claim points at concrete cases;
- canonical-topology coverage does NOT transfer to non-canonical switching
  models: those are reported ``not_run`` with the reason;
- the paradigm trap is stated on every card: agreement with another
  ideal-switch simulator proves nothing where the paradigm itself is
  blind (EMI, silicon physics) — those claims require goldens/measurements.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pear_sim import __version__
from pear_sim.spec.canonical_topologies import (
    extract_boost_boundary_parameters,
    extract_buck_boundary_parameters,
    extract_inverting_buck_boost_boundary_parameters,
)
from pear_sim.spec.model import Model

__all__ = ["build_confidence_card"]

CONFIDENCE_CARD_SCHEMA = "pear-sim/confidence-card/v0.1"

_PARADIGM_TRAP = (
    "Agreement with another ideal-switch engine is differential evidence between two "
    "paradigms; it proves nothing where the paradigm is blind (EMI, "
    "switching-edge physics, silicon) — such claims require analytic "
    "goldens or measurements"
)

_STATUS_ORDER = {"not_run": 0, "partial": 1, "validated": 2}


def _load_status() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "validation_status.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_match(model: Model) -> str | None:
    for label, extractor in (
        ("canonical_buck", extract_buck_boundary_parameters),
        ("canonical_boost", extract_boost_boundary_parameters),
        (
            "canonical_inverting_buck_boost",
            extract_inverting_buck_boost_boundary_parameters,
        ),
    ):
        try:
            extractor(model)
        except Exception:  # noqa: BLE001 - matchers raise their own error types
            continue
        return label
    return None


def _row(
    feature: str,
    val_ref: str | None,
    status_payload: dict[str, Any] | None,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    if val_ref is None or status_payload is None:
        return {
            "feature": feature,
            "validation_ref": val_ref,
            "status": "not_run",
            "reason": reason or "no validation area covers this feature",
        }
    raw_status = status_payload.get("status", "not_run")
    status = raw_status if raw_status in _STATUS_ORDER else "not_run"
    row: dict[str, Any] = {
        "feature": feature,
        "validation_ref": val_ref,
        "status": status,
    }
    if status == "not_run" and reason is None:
        reason = "validation area not started — no oracle, so never a positive"
    if reason is not None:
        row["reason"] = reason
    if status != "not_run":
        claim_boundary = status_payload.get("claim_boundary")
        if claim_boundary is None:
            # Older ledger entries carry the boundary implicitly: exactly the
            # implemented list, nothing more.
            claim_boundary = (
                "exactly the implemented cases listed under evidence_cases; "
                "nothing beyond them"
            )
        row["claim_boundary"] = claim_boundary
        row["evidence_cases"] = list(status_payload.get("implemented", []))
        row["known_gaps"] = list(status_payload.get("missing", []))
    return row


def build_confidence_card(model: Model) -> dict[str, Any]:
    """Coverage-derived confidence for one model; weakest link wins."""

    status = _load_status()
    tests: dict[str, Any] = status.get("tests", {})
    component_types = {component.type for component in model.components}
    control_kinds = {control.kind for control in model.controls.values()}
    sampled_kinds = {
        getattr(control, "kind", "sampled_pi_pwm")
        for control in model.sampled_controls.values()
    }
    has_waveform_sources = any(
        component.waveform is not None for component in model.components
    )

    rows: list[dict[str, Any]] = []
    rows.append(
        _row(
            "linear kernel (R/L/C, sources, transient integration)",
            "VAL-001",
            tests.get("VAL-001"),
        )
    )

    if "ideal_switch" in component_types or "diode" in component_types:
        canonical = _canonical_match(model)
        if canonical is not None:
            rows.append(
                _row(
                    f"switching topology ({canonical})",
                    "VAL-002",
                    tests.get("VAL-002"),
                )
            )
            rows.append(
                _row(
                    f"DCM/BCM boundaries ({canonical})",
                    "VAL-003",
                    tests.get("VAL-003"),
                )
            )
        elif (
            (control_kinds or sampled_kinds)
            and "three_phase_spwm" not in control_kinds
            and "three_phase_npc_spwm" not in control_kinds
            and "sampled_dq_current_pwm" not in sampled_kinds
        ):
            # A three-phase bridge belongs to VAL-005 below; only switching
            # topologies with NO covering area land here.
            rows.append(
                _row(
                    "non-canonical switching topology",
                    "VAL-002",
                    None,
                    reason=(
                        "VAL-002/VAL-003 cover exactly the canonical "
                        "buck/boost/inverting slices; coverage does not "
                        "transfer to this topology"
                    ),
                )
            )
        if (
            "diode" in component_types
            and has_waveform_sources
            and not control_kinds
            and not sampled_kinds
        ):
            rows.append(
                _row(
                    "rectifier / natural commutation",
                    "VAL-004",
                    tests.get("VAL-004"),
                )
            )

    if "three_phase_spwm" in control_kinds or "sampled_dq_current_pwm" in sampled_kinds:
        rows.append(
            _row(
                "three-phase inverter / modulation",
                "VAL-005",
                tests.get("VAL-005"),
            )
        )

    if "three_phase_npc_spwm" in control_kinds:
        rows.append(
            _row(
                "three-level NPC bridge / level-shifted modulation",
                "VAL-010",
                tests.get("VAL-010"),
            )
        )

    if "sat_inductor" in component_types:
        rows.append(
            _row(
                "saturable inductor (two-segment flux)",
                "VAL-002",
                None,
                reason=(
                    "validated against its own analytic two-segment step "
                    "response; no dedicated VAL area yet"
                ),
            )
        )

    if getattr(model, "loss_analysis", None) is not None:
        rows.append(
            _row(
                "semiconductor loss estimate (device cards)",
                "VAL-007",
                tests.get("VAL-007"),
            )
        )
        rows.append(
            _row(
                "thermal chain (Foster + cooling path)",
                "VAL-008",
                tests.get("VAL-008"),
            )
        )

    weakest = min(
        (row["status"] for row in rows),
        key=lambda value: _STATUS_ORDER[value],
        default="not_run",
    )
    return {
        "schema": CONFIDENCE_CARD_SCHEMA,
        "kind": "coverage-derived-confidence",
        "pear_sim_version": __version__,
        "validation_status_date": status.get("date"),
        "overall": weakest,
        "policy": (
            "weakest-link over the features this model uses; computed from "
            "coverage, never from a marketing label"
        ),
        "features": rows,
        "paradigm_trap": _PARADIGM_TRAP,
    }
