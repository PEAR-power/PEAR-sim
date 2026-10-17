"""DC-link midpoint balance as a measured, gated quantity.

The capacitive-link NPC example replaces the stiff +-600 V sources with
real half-link capacitors and a floating three-wire load star, so ALL
midpoint current flows through the clamp path — the physical NPC balance
problem. The midpoint envelope is gated through the standard expectations
machinery (worst-case blocking), with the declared claim boundary that no
active balancing exists yet.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pear_sim.core.solver import HybridSimulator
from pear_sim.evidence.evidence_io import evaluate_run
from pear_sim.run.io import write_result
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "npc_three_phase_midpoint.yaml"
EXPECTATIONS = ROOT / "examples" / "npc_midpoint_expectations.json"


@pytest.fixture(scope="module")
def midpoint_bundle(tmp_path_factory) -> Path:
    model = load_model(EXAMPLE)
    result = HybridSimulator(model).simulate()
    destination = tmp_path_factory.mktemp("npc-midpoint") / "bundle"
    write_result(result, destination)
    return destination


def test_midpoint_dynamics_are_physical(midpoint_bundle: Path) -> None:
    metrics = json.loads(
        (midpoint_bundle / "metrics.json").read_text(encoding="utf-8")
    )
    midpoint = metrics["v(mid)"]
    # The midpoint is a live state: third-harmonic clamp-current ripple on
    # 2 mF at ~300 A rms load current gives a substantial envelope. Bounds
    # are deliberately loose (platform-portable), but must prove the
    # midpoint MOVES — a stiff-source model would show exactly zero.
    assert 50.0 < midpoint["peak_to_peak_last_half"] < 150.0
    assert 540.0 < midpoint["mean_last_half"] < 620.0
    # Energy sanity: both half-link voltages stay inside the rail span.
    top = metrics["v(cdc_top)"]
    bottom = metrics["v(cdc_bot)"]
    assert top["min"] > 400.0 and top["max"] < 800.0
    assert bottom["min"] > 400.0 and bottom["max"] < 800.0


def test_clamp_path_carries_the_midpoint_current(midpoint_bundle: Path) -> None:
    metrics = json.loads(
        (midpoint_bundle / "metrics.json").read_text(encoding="utf-8")
    )
    # With a floating load star there is no 4-wire return: nonzero clamp
    # currents are the only midpoint mechanism, so they must be substantial.
    assert metrics["i(d5a)"]["max"] > 100.0
    assert metrics["i(d6a)"]["min"] < -100.0 or metrics["i(d6a)"]["max"] > 100.0


def test_declared_envelope_gate_passes(midpoint_bundle: Path, tmp_path: Path) -> None:
    artifacts = evaluate_run(
        midpoint_bundle, EXPECTATIONS, tmp_path / "evidence-pass"
    )
    assert artifacts.blocking_status == "pass"


def test_tightened_envelope_gate_blocks(midpoint_bundle: Path, tmp_path: Path) -> None:
    expectations = json.loads(EXPECTATIONS.read_text(encoding="utf-8"))
    for criterion in expectations["criteria"]:
        if criterion["id"] == "midpoint_ripple_envelope":
            criterion["maximum"] = 10.0  # far tighter than the physics allows
    tightened = tmp_path / "tight.json"
    tightened.write_text(json.dumps(expectations, indent=2), encoding="utf-8")
    artifacts = evaluate_run(midpoint_bundle, tightened, tmp_path / "evidence-fail")
    assert artifacts.blocking_status == "fail"
    evidence = json.loads(
        (tmp_path / "evidence-fail" / "evidence.json").read_text(encoding="utf-8")
    )
    failing = [
        row["id"]
        for row in evidence["criteria"]
        if row.get("status") == "fail" and row.get("blocking")
    ]
    assert failing == ["midpoint_ripple_envelope"]


def test_midpoint_metric_is_deterministic(midpoint_bundle: Path) -> None:
    # A second simulation of the same model must reproduce the midpoint
    # metrics bit-identically (determinism is what makes the gate evidence).
    model = load_model(EXAMPLE)
    result = HybridSimulator(model).simulate()
    metrics = json.loads(
        (midpoint_bundle / "metrics.json").read_text(encoding="utf-8")
    )
    fresh = result.metrics()
    assert fresh["v(mid)"]["peak_to_peak_last_half"] == pytest.approx(
        metrics["v(mid)"]["peak_to_peak_last_half"], abs=0.0
    )
    assert np.isfinite(fresh["v(mid)"]["mean_last_half"])
