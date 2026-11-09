# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pear_sim.numerical_report import NumericalReportError, generate_numerical_report


def _evidence() -> dict[str, object]:
    return {
        "schema": "pear-sim/platform-evidence/canonical-buck-numerical/v0.1",
        "kind": "canonical-buck-numerical-evidence",
        "status": "pass",
        "profile": {"id": "m2.2-test-v1", "cell_count": 1},
        "source": {"revision": "abc123", "pear_sim_version": "0.2.0.dev2"},
        "platform": {"system": "Darwin", "machine": "arm64", "python": "3.13.5"},
        "configurations": [
            {
                "id": "reference",
                "studies": ["solver-comparison"],
                "solver": "dopri5",
                "rtol": 1e-10,
                "atol": 1e-12,
                "max_step_over_period": 0.01,
                "output_step_over_period": 0.005,
            }
        ],
        "cells": [
            {
                "id": "reference:ratio-1",
                "status": "pass",
                "expected_regime": "BCM",
                "gates": [{"id": "regime", "status": "pass"}],
            }
        ],
        "studies": {
            "solver-comparison": {
                "status": "pass",
                "cell_count": 1,
                "cell_status_counts": {"pass": 1},
            }
        },
        "summary": {"status": "pass", "cell_count": 1, "gate_count": 1},
    }


def test_numerical_report_is_offline_hash_bound_and_presentation_only(tmp_path: Path) -> None:
    evidence = _evidence()
    before = json.dumps(evidence, sort_keys=True)
    source_text = json.dumps(evidence, sort_keys=True) + "\n"
    digest = hashlib.sha256(source_text.encode()).hexdigest()

    artifacts = generate_numerical_report(
        evidence,
        tmp_path / "report",
        source_artifact_sha256=digest,
    )

    assert json.dumps(evidence, sort_keys=True) == before
    report = json.loads(artifacts.report_json.read_text(encoding="utf-8"))
    assert report["evidence"] == evidence
    assert report["source"]["artifact_sha256"] == digest
    assert report["claim"]["hardware_sign_off_allowed"] is False
    page = artifacts.report_html.read_text(encoding="utf-8")
    assert "script-src &#x27;none&#x27;" not in page
    assert "script-src 'none'" in page
    assert "<script" not in page.lower()
    assert "https://" not in page and "http://" not in page
    assert "m2.2-test-v1" in page
    manifest = json.loads(artifacts.artifact_manifest.read_text(encoding="utf-8"))
    assert [item["path"] for item in manifest["artifacts"]] == [
        "numerical-report.html",
        "numerical-report.json",
    ]


def test_numerical_report_rejects_nonfinite_input_and_unowned_destination(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    evidence["bad"] = float("nan")
    with pytest.raises(NumericalReportError, match="finite JSON"):
        generate_numerical_report(evidence, tmp_path / "nonfinite", source_artifact_sha256="a" * 64)

    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "user.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(NumericalReportError, match="not owned"):
        generate_numerical_report(_evidence(), destination, source_artifact_sha256="b" * 64)
    assert (destination / "user.txt").read_text(encoding="utf-8") == "keep"
