# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Regression witness for the spike that adds parameter envelopes and events.

The parametric-corner hook (A) and the event-injection hook (B) both extend the
model surface. Neither may move an existing result. This module pins the
unperturbed ``buck_ccm`` reference: the hash chain of the resolved model and
request, and the declared band metrics of the two state signals.

The hash assertions are the sharp ones. ``model_sha256`` is taken over the
resolved model dictionary, so a new field emitted unconditionally — even an
empty one — would change it. Corner declarations and events must therefore stay
absent from the resolved model of a model that declares none.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pear_sim.core.solver import HybridSimulator
from pear_sim.run.io import write_result
from pear_sim.spec.model import load_model


EXAMPLES_DIR = Path(__file__).parents[1] / "examples"


# Recorded 2026-07-27 from origin/main at 953b836, before the spike.
BASELINE_MODEL_SHA256 = "1423749ba5cce42c67b00a05d50d346fd0c146c5815bf10c6cced19fde30f852"
BASELINE_REQUEST_SHA256 = "829eb76c92639d2b8ba5d338736c2e9b05700ee276c837c3d3ccfffdf15c4f26"

# Steady-state band of the reference buck. Tolerances are absolute and far
# tighter than any physically meaningful change, but loose enough that a
# platform-level float difference does not fail the suite.
BASELINE_BANDS = {
    "v(out)": {
        "mean_last_half": (40.000000009863236, 1e-6),
        "min_last_half": (39.95997780186545, 1e-6),
        "max_last_half": (40.03502140755006, 1e-6),
    },
    "i(l1)": {
        "mean_last_half": (4.000000001683603, 1e-7),
        "min_last_half": (3.3997005534480254, 1e-7),
        "max_last_half": (4.600300892512303, 1e-7),
    },
}


@pytest.fixture(scope="module")
def buck_run(tmp_path_factory):
    model = load_model(EXAMPLES_DIR / "buck.yaml")
    result = HybridSimulator(model).simulate()
    destination = tmp_path_factory.mktemp("buck-regression") / "run"
    write_result(result, destination)
    return destination


def test_reference_buck_hash_chain_is_unchanged(buck_run):
    """A model that declares no envelope and no event hashes as it always did."""

    run = json.loads((buck_run / "run.json").read_text())
    assert run["model_sha256"] == BASELINE_MODEL_SHA256
    assert run["request_sha256"] == BASELINE_REQUEST_SHA256


def test_reference_buck_resolved_model_carries_no_spike_fields(buck_run):
    """Absence, not emptiness: no `parameter_envelopes`, no `perturbation`."""

    resolved = json.loads((buck_run / "model.resolved.json").read_text())
    assert "parameter_envelopes" not in resolved
    assert "perturbation" not in resolved


def test_reference_buck_bands_are_unchanged(buck_run):
    metrics = json.loads((buck_run / "metrics.json").read_text())
    for signal, fields in BASELINE_BANDS.items():
        for field, (expected, tolerance) in fields.items():
            assert metrics[signal][field] == pytest.approx(expected, abs=tolerance), (
                f"{signal}.{field} moved"
            )
