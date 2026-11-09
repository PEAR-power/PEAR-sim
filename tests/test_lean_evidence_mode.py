# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""The lean evidence mode drops claims, never accuracy.

Every assertion here is deliberately an exact equality: the whole point of the
mode is that what a lean run still reports came out of the identical code path
as a full run's, so "close enough" would silently permit a real regression.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pear_sim.core.metrics import (
    _STATE_EXTREMA_METHOD,
    _STATE_EXTREMA_NOT_COMPUTED,
    _IntervalIntegral,
    _MetricAccumulator,
    _WindowAccumulator,
)
from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import SimulationError
from pear_sim.run.io import write_result
from pear_sim.spec.model import load_model


EXAMPLES = Path(__file__).parents[1] / "examples"
BUCK = EXAMPLES / "buck.yaml"


@pytest.fixture(scope="module")
def buck_runs():
    model = load_model(BUCK)
    full = HybridSimulator(model).simulate()
    lean = HybridSimulator(model, evidence_mode="lean").simulate()
    return full, lean


def test_lean_reproduces_every_reported_number_bit_for_bit(buck_runs):
    full, lean = buck_runs
    assert np.array_equal(full.time, lean.time)
    assert set(full.signals) == set(lean.signals)
    for name, values in full.signals.items():
        assert np.array_equal(values, lean.signals[name]), name
    assert full.metrics() == lean.metrics()
    assert full.time_integrals == lean.time_integrals
    assert full.state_extrema == lean.state_extrema
    assert full.continuous_outputs == lean.continuous_outputs


def test_lean_declares_the_energy_balance_absent_instead_of_zeroed(buck_runs):
    full, lean = buck_runs
    assert "status" not in full.physics_audit
    assert full.physics_audit["full"]["balance_within_tolerance"] is True
    assert lean.physics_audit["status"] == "not_computed"
    assert lean.physics_audit["evidence_mode"] == "lean"
    # A reader indexing a window must fail, not read a balance that looks
    # conserved because nothing was ever integrated into it.
    assert "full" not in lean.physics_audit
    assert "last_half" not in lean.physics_audit
    with pytest.raises(KeyError):
        lean.physics_audit["full"]


def test_lean_never_labels_uncomputed_extrema_with_the_search_method(buck_runs):
    full, lean = buck_runs
    assert len(full.mode_intervals) == len(lean.mode_intervals)
    last_half_start = 0.5 * (full.model.transient.start + full.model.transient.stop)
    skipped = 0
    for full_interval, lean_interval in zip(
        full.mode_intervals, lean.mode_intervals, strict=True
    ):
        assert full_interval["state_extrema_method"] == _STATE_EXTREMA_METHOD
        assert lean_interval["state_extrema"] is None
        assert lean_interval["state_extrema_method"] == _STATE_EXTREMA_NOT_COMPUTED
        if float(lean_interval["end_s"]) <= last_half_start:
            skipped += 1
        # The endpoints stay exact in both modes: they are read off the
        # integrator, not searched for.
        assert full_interval["state_start"] == lean_interval["state_start"]
        assert full_interval["state_end"] == lean_interval["state_end"]
    assert skipped > 0, "the fixture must exercise intervals outside the reported window"


def test_lean_does_less_work(buck_runs):
    full, lean = buck_runs
    # Identical trajectory: the saving comes from observation, not integration.
    assert full.stats["integration_steps"] == lean.stats["integration_steps"]
    assert full.stats["natural_events"] == lean.stats["natural_events"]
    assert full.stats["mode_intervals"] == lean.stats["mode_intervals"]
    assert full.stats["metric_quadrature_intervals"] == lean.stats["metric_quadrature_intervals"]


def test_unknown_evidence_mode_fails_closed():
    model = load_model(BUCK)
    with pytest.raises(SimulationError, match="Unknown evidence mode"):
        HybridSimulator(model, evidence_mode="cheap")
    with pytest.raises(SimulationError, match="Unknown evidence mode"):
        _MetricAccumulator(model, HybridSimulator(model).circuit, np.zeros(2), evidence_mode="")


def test_partial_energy_balance_cannot_be_accumulated():
    model = load_model(BUCK)
    window = _WindowAccumulator.create(
        model,
        model.transient.start,
        model.transient.stop,
        component_energy_requested=False,
    )
    with pytest.raises(SimulationError, match="partial energy balance"):
        window.add(
            _IntervalIntegral(
                duration=1.0,
                signal_integrals={name: 0.0 for name in model.outputs},
                signal_square_integrals={name: 0.0 for name in model.outputs},
                component_absorbed_energy={component.id: 0.0 for component in model.components},
            )
        )


def test_full_run_result_defaults_to_the_full_evidence_mode(buck_runs):
    full, lean = buck_runs
    assert full.evidence_mode == "full"
    assert lean.evidence_mode == "lean"


def _written(result, destination):
    write_result(result, destination)
    return {
        name: json.loads((destination / name).read_text())
        for name in ("run.json", "summary.json", "warnings.json")
    }


def test_bundle_declares_the_mode_in_every_place_a_reader_looks(buck_runs, tmp_path):
    full, lean = buck_runs
    full_bundle = _written(full, tmp_path / "full")
    lean_bundle = _written(lean, tmp_path / "lean")

    assert full_bundle["run.json"]["evidence_mode"] == "full"
    assert full_bundle["summary.json"]["evidence_mode"] == "full"
    assert lean_bundle["run.json"]["evidence_mode"] == "lean"
    assert lean_bundle["summary.json"]["evidence_mode"] == "lean"

    full_codes = {warning["code"] for warning in full_bundle["warnings.json"]}
    lean_codes = {warning["code"] for warning in lean_bundle["warnings.json"]}
    assert "EVIDENCE_MODE_LEAN" not in full_codes
    assert "EVIDENCE_MODE_LEAN" in lean_codes
    lean_warning = next(
        warning
        for warning in lean_bundle["warnings.json"]
        if warning["code"] == "EVIDENCE_MODE_LEAN"
    )
    assert lean_warning["severity"] == "warning"
    assert "not an evidence bundle" in lean_warning["message"]

    audit = json.loads((tmp_path / "lean" / "physics-audit.json").read_text())
    assert audit["status"] == "not_computed"


def test_lean_bundle_differs_from_full_only_in_the_mode_declaration(buck_runs, tmp_path):
    """The evidence-equivalence gate for this feature.

    Anything outside this list changing between the two modes means lean stopped
    being a claim reduction and became a different simulation.
    """

    full, lean = buck_runs
    full_dir, lean_dir = tmp_path / "full", tmp_path / "lean"
    write_result(full, full_dir)
    write_result(lean, lean_dir)
    may_differ = {
        "physics-audit.json",
        "warnings.json",
        "summary.json",
        "run.json",
        "artifact-manifest.json",
    }
    differing = {
        path.name
        for path in sorted(full_dir.iterdir())
        if path.read_bytes() != (lean_dir / path.name).read_bytes()
    }
    assert differing <= may_differ, f"lean changed artifacts it must not touch: {differing - may_differ}"
    assert "metrics.json" not in differing
    assert "waveforms.csv" not in differing
    assert "mode-coverage.json" not in differing
