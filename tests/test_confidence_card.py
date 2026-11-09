# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Coverage-derived confidence card: weakest link, no free positives."""

import json
from pathlib import Path

from pear_sim.analysis.confidence import build_confidence_card
from pear_sim.core.solver import HybridSimulator
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]


def test_packaged_status_matches_the_repo_ledger() -> None:
    packaged = json.loads(
        (ROOT / "src" / "pear_sim" / "validation_status.json").read_text()
    )
    canonical = json.loads((ROOT / "validation" / "status.json").read_text())
    assert packaged == canonical, (
        "src/pear_sim/validation_status.json must be a byte-for-byte copy of "
        "validation/status.json (cp after every ledger update)"
    )


def test_canonical_buck_reports_partial_with_exact_evidence() -> None:
    card = build_confidence_card(load_model(ROOT / "examples" / "buck_dcm.yaml"))
    assert card["overall"] == "partial"
    by_ref = {row["validation_ref"]: row for row in card["features"]}
    assert by_ref["VAL-002"]["status"] == "partial"
    assert by_ref["VAL-002"]["claim_boundary"]
    assert by_ref["VAL-002"]["evidence_cases"], "partial must cite exact cases"
    assert by_ref["VAL-003"]["feature"].endswith("(canonical_buck)")
    assert "paradigm" in card["paradigm_trap"]


def test_rectifier_maps_to_not_started_val004() -> None:
    card = build_confidence_card(
        load_model(ROOT / "examples" / "halfwave_rectifier.yaml")
    )
    by_ref = {row["validation_ref"]: row for row in card["features"]}
    assert by_ref["VAL-004"]["status"] == "not_run"
    assert "never a positive" in by_ref["VAL-004"]["reason"]
    assert card["overall"] == "not_run"


def test_non_canonical_switching_never_inherits_coverage() -> None:
    card = build_confidence_card(
        load_model(ROOT / "examples" / "buck_closed_loop.yaml")
    )
    rows = {row["feature"]: row for row in card["features"]}
    non_canonical = rows["non-canonical switching topology"]
    assert non_canonical["status"] == "not_run"
    assert "does not transfer" in non_canonical["reason"]
    assert card["overall"] == "not_run"


def test_inverter_and_loss_model_report_their_areas() -> None:
    inverter = build_confidence_card(
        load_model(ROOT / "examples" / "inverter_three_phase_floating_neutral.yaml")
    )
    assert inverter["overall"] == "partial"
    refs = {row["validation_ref"] for row in inverter["features"]}
    assert "VAL-005" in refs

    losses = build_confidence_card(
        load_model(ROOT / "examples" / "buck_with_loss_cards.yaml")
    )
    refs = {row["validation_ref"] for row in losses["features"]}
    assert {"VAL-007", "VAL-008"} <= refs
    assert losses["overall"] == "partial"


def test_every_bundle_carries_the_card(tmp_path: Path) -> None:
    from dataclasses import replace

    from pear_sim.run.io import write_result

    model = load_model(ROOT / "examples" / "buck_dcm.yaml")
    model = replace(model, transient=replace(model.transient, stop=1e-3))
    result = HybridSimulator(model).simulate()
    destination = write_result(result, tmp_path / "bundle")
    card = json.loads((destination / "confidence-card.json").read_text())
    assert card["schema"] == "pear-sim/confidence-card/v0.1"
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["confidence_card"]["overall"] == card["overall"]
    run_manifest = json.loads((destination / "run.json").read_text())
    assert "confidence_card" in json.dumps(run_manifest)
