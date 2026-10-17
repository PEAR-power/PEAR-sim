from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import pear_sim.benchmark_report as benchmark_report
from pear_sim.benchmark_report import BenchmarkReportError, generate_sweep_report


def _sweep_fixture() -> dict[str, object]:
    return {
        "schema": "pear-sim/sweep-summary/v0.1",
        "benchmark_id": "VAL-003 buck <script>alert('id')</script>",
        "overall_status": "fail",
        "bcm": {
            "r_bcm_ohm": 5.0,
            "method": "independent analytic boundary",
        },
        "points": [
            {
                "index": 0,
                "id": "r-low",
                "r_ohm": 2.5,
                "r_over_r_bcm": 0.5,
                "vout_mean_v": 39.99,
                "minimum_inductor_current_a": 1.2,
                "regime": "ccm",
                "status": "pass",
                "runtime_s": 0.21,
                "artifacts": {"run": "runs/r-low/run.json"},
            },
            {
                "index": 1,
                "id": "r-bcm",
                "r_ohm": 5.0,
                "r_over_r_bcm": 1.0,
                "vout_mean_v": 40.01,
                "minimum_inductor_current_a": 0.0,
                "regime": "bcm",
                "status": "fail",
                "runtime_s": 0.22,
                "artifacts": {
                    "report": {
                        "path": "reports/<unsafe>/report.html",
                        "sha256": "a" * 64,
                    }
                },
            },
            {
                "index": 2,
                "id": "r-high",
                "r_ohm": 10.0,
                "r_over_r_bcm": 2.0,
                "vout_mean_v": 40.02,
                "minimum_inductor_current_a": -2e-12,
                "regime": "dcm",
                "status": "not-run",
                "runtime_s": None,
            },
            {
                "index": 3,
                "id": "outside-envelope",
                "r_ohm": 20.0,
                "r_over_r_bcm": 4.0,
                "vout_mean_v": None,
                "minimum_inductor_current_a": None,
                "regime": "unclassified<img src=x onerror=alert(1)>",
                "status": "not-applicable",
            },
        ],
        # Deliberately inconsistent with the failing point. The renderer must copy,
        # not recompute, the producer's aggregate gate record.
        "worst_gates": [
            {
                "gate": "reverse-current",
                "status": "pass",
                "value": 9.876e-9,
                "limit": 1e-8,
                "unit": "I_base",
                "point_id": "r-bcm",
                "message": "recorded <strong>producer result</strong>",
            },
            {
                "gate": "optional-oracle",
                "status": "not-applicable",
                "value": None,
                "limit": None,
                "unit": None,
                "point_id": None,
                "message": "oracle not configured",
            },
        ],
        "failures": [
            {
                "id": "failure-1",
                "status": "fail",
                "point_id": "r-bcm",
                "gate": "event-time",
                "message": "delta exceeded <img src=x onerror=alert(1)>",
                "artifacts": {"diagnostic": "failures/failure-1.json"},
            }
        ],
        "artifacts": {
            "raw_table": {
                "path": "data/sweep.csv",
                "sha256": "b" * 64,
            },
            "unsafe_role<script>": "artifacts/<unsafe>.json",
        },
        "provenance": {
            "solver": "dopri5",
            "platform": "test-platform",
        },
    }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_sweep_report_is_deterministic_offline_escaped_and_human_facing(
    tmp_path: Path,
) -> None:
    summary = _sweep_fixture()
    destination = tmp_path / "report"

    first = generate_sweep_report(summary, destination)
    first_json = first.report_json.read_bytes()
    first_html = first.report_html.read_bytes()
    first_manifest = first.artifact_manifest.read_bytes() if first.artifact_manifest else b""
    second = generate_sweep_report(summary, destination)

    assert second.report_json.read_bytes() == first_json
    assert second.report_html.read_bytes() == first_html
    assert second.artifact_manifest is not None
    assert second.artifact_manifest.read_bytes() == first_manifest

    report = json.loads(first_json)
    assert report["schema"] == "pear-sim/sweep-report/v0.1"
    assert report["claim"] == {
        "level": "exploratory",
        "statement": (
            "This report presents recorded sweep evidence for exploration; "
            "it is not a validation or hardware sign-off claim."
        ),
        "validation_claim": False,
    }
    assert report["sweep_summary"] == summary
    assert (
        report["source"]["sweep_summary_semantic_sha256"]
        == hashlib.sha256(_canonical_bytes(summary)).hexdigest()
    )
    assert report["source"]["sweep_summary_artifact_sha256"] is None

    rendered = first_html.decode("utf-8")
    lowered = rendered.lower()
    assert "EXPLORATORY — NOT VALIDATION" in rendered
    assert "does not recompute gates" in rendered
    assert "script-src &#x27;none&#x27;" not in rendered
    assert "script-src 'none'" in rendered
    assert "connect-src 'none'" in rendered
    assert "<script" not in lowered
    assert "<img" not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered
    assert "VAL-003 buck &lt;script&gt;alert(&#x27;id&#x27;)&lt;/script&gt;" in rendered
    assert "unclassified&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert "delta exceeded &lt;img src=x onerror=alert(1)&gt;" in rendered
    assert "unsafe_role&lt;script&gt;" in rendered
    assert "reports/&lt;unsafe&gt;/report.html" in rendered

    assert 'class="status status-pass">pass</strong>' in rendered
    assert 'class="status status-fail">fail</strong>' in rendered
    assert 'class="status status-not-run">not-run</strong>' in rendered
    assert 'class="status status-not-applicable">not-applicable</strong>' in rendered
    assert "BCM · R/R_BCM = 1" in rendered
    assert "Mean Vout [V]" in rendered
    assert "Minimum iL [A]" in rendered
    assert "R / R_BCM (log scale)" in rendered
    assert "CCM" in rendered and "BCM" in rendered and "DCM" in rendered
    assert "9.876e-09" in rendered
    assert "recorded &lt;strong&gt;producer result&lt;/strong&gt;" in rendered
    assert "failures/failure-1.json" in rendered
    assert "data/sweep.csv" in rendered
    assert "Recorded sweep point table (4 of 4 row(s))" in rendered

    bcm_label = re.search(r'data-chart-role="bcm-label"[^>]+y="([0-9.]+)"', rendered)
    panel_titles = re.findall(
        r'data-chart-role="panel-title"[^>]+y="([0-9.]+)"[^>]*>([^<]+)</text>',
        rendered,
    )
    assert bcm_label is not None
    minimum_title_y = next(float(y) for y, title in panel_titles if title == "Minimum iL [A]")
    assert minimum_title_y - float(bcm_label.group(1)) >= 20.0


def test_sweep_report_manifest_hashes_exact_generated_artifacts(tmp_path: Path) -> None:
    summary = _sweep_fixture()
    source_artifact_sha256 = "c" * 64
    artifacts = generate_sweep_report(
        summary,
        tmp_path / "report",
        source_artifact_sha256=source_artifact_sha256,
    )

    assert artifacts.artifact_manifest is not None
    manifest = json.loads(artifacts.artifact_manifest.read_text(encoding="utf-8"))
    records = manifest["artifacts"]
    assert [record["path"] for record in records] == [
        "sweep-report.html",
        "sweep-report.json",
    ]
    for record in records:
        path = artifacts.artifact_manifest.parent / record["path"]
        assert record["bytes"] == path.stat().st_size
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest["subject"] == {
        "kind": "sweep-report",
        "sha256": hashlib.sha256(_canonical_bytes(records)).hexdigest(),
        "sweep_summary_artifact_sha256": source_artifact_sha256,
        "sweep_summary_semantic_sha256": hashlib.sha256(_canonical_bytes(summary)).hexdigest(),
    }


def test_sweep_report_embeds_and_hashes_a_sanitized_nominal_schematic(
    tmp_path: Path,
) -> None:
    source = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10" '
        'role="img" aria-label="Nominal topology">'
        '<title>Nominal topology</title><rect x="1" y="1" width="8" height="8" />'
        "</svg>"
    )
    artifacts = generate_sweep_report(
        _sweep_fixture(),
        tmp_path / "report",
        schematic_svg=source,
        schematic_caption="Nominal template <not a point-specific model>",
    )

    assert artifacts.schematic_svg is not None
    assert artifacts.schematic_svg.name == "schematic.svg"
    sanitized = artifacts.schematic_svg.read_text(encoding="utf-8")
    rendered = artifacts.report_html.read_text(encoding="utf-8")
    report = json.loads(artifacts.report_json.read_text(encoding="utf-8"))
    manifest = json.loads(artifacts.artifact_manifest.read_text(encoding="utf-8"))

    assert "<script" not in sanitized.lower()
    assert "Source topology" in rendered
    assert sanitized in rendered
    assert "Nominal template &lt;not a point-specific model&gt;" in rendered
    assert report["files"]["schematic"] == "schematic.svg"
    assert report["presentation"]["schematic"]["role"] == (
        "nominal_source_topology_template"
    )
    assert report["source"]["schematic"] == {
        "bytes": artifacts.schematic_svg.stat().st_size,
        "path": "schematic.svg",
        "sha256": hashlib.sha256(artifacts.schematic_svg.read_bytes()).hexdigest(),
    }
    assert [record["path"] for record in manifest["artifacts"]] == [
        "schematic.svg",
        "sweep-report.html",
        "sweep-report.json",
    ]


def test_sweep_report_strips_unsafe_schematic_content_before_writing_output(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "unsafe-schematic"

    artifacts = generate_sweep_report(
        _sweep_fixture(),
        destination,
        schematic_svg=(
            '<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
        ),
    )

    assert artifacts.schematic_svg is not None
    sanitized = artifacts.schematic_svg.read_text(encoding="utf-8").lower()
    rendered = artifacts.report_html.read_text(encoding="utf-8").lower()
    assert "<script" not in sanitized
    assert "<script" not in rendered
    assert "alert(1)" not in sanitized


def test_sweep_report_can_omit_manifest_and_remove_owned_stale_manifest(tmp_path: Path) -> None:
    destination = tmp_path / "report"
    first = generate_sweep_report(_sweep_fixture(), destination)
    assert first.artifact_manifest is not None

    second = generate_sweep_report(_sweep_fixture(), destination, write_manifest=False)
    report = json.loads(second.report_json.read_text(encoding="utf-8"))

    assert second.artifact_manifest is None
    assert not (destination / "artifact-manifest.json").exists()
    assert report["files"]["manifest"] is None


def test_sweep_report_gracefully_renders_nonexecuted_and_missing_fields(tmp_path: Path) -> None:
    summary = {
        "schema": "pear-sim/sweep-summary/v0.1",
        "benchmark_id": "not-run-sweep",
        "overall_status": "not-run",
        "points": [
            {
                "index": 0,
                "r_over_r_bcm": "not numeric",
                "status": "not-applicable",
            }
        ],
        "worst_gates": [
            {
                "gate": "oracle-comparison",
                "status": "not-run",
                "message": "oracle unavailable",
            }
        ],
        "failures": None,
        "artifacts": None,
    }

    artifacts = generate_sweep_report(summary, tmp_path / "report")
    rendered = artifacts.report_html.read_text(encoding="utf-8")

    assert 'class="status status-not-run">not-run</strong>' in rendered
    assert 'class="status status-not-applicable">not-applicable</strong>' in rendered
    assert "No points contain finite recorded" in rendered
    assert "No failure records were provided. This is not itself a pass claim." in rendered
    assert "No sweep-level artifact pointers were recorded." in rendered
    assert "oracle unavailable" in rendered


def test_sweep_report_accepts_nested_recorded_benchmark_point_fields(tmp_path: Path) -> None:
    summary = {
        "schema": "pear-sim/benchmark-oracle/ideal-buck-boundary/v0.1",
        "kind": "independent-periodic-affine-buck-boundary-verification",
        "boundary": {"load_ohm": 12.5},
        "summary": {"status": "pass"},
        "points": [
            {
                "point_id": "load-ratio-0p5",
                "load_ratio_to_bcm": 0.5,
                "load_ohm": 6.25,
                "regime": "CCM",
                "status": "pass",
                "metrics": {
                    "mean_output_voltage_v": 40.125,
                    "minimum_inductor_current_a": 1.75,
                },
            }
        ],
    }

    artifacts = generate_sweep_report(summary, tmp_path / "report")
    rendered = artifacts.report_html.read_text(encoding="utf-8")

    assert "independent-periodic-affine-buck-boundary-verification" in rendered
    assert '<strong class="status status-pass">pass</strong>' in rendered
    assert "12.5" in rendered
    assert "load-ratio-0p5" in rendered
    assert "6.25" in rendered
    assert "40.125" in rendered
    assert "1.75" in rendered
    assert "No points contain finite recorded" not in rendered


def test_sweep_report_bounds_human_rows_but_retains_full_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = _sweep_fixture()
    points = [
        {
            "index": index,
            "id": f"point-{index}",
            "r_over_r_bcm": 0.5 + index,
            "vout_mean_v": 40.0 + index,
            "minimum_inductor_current_a": float(index),
            "regime": "ccm",
            "status": "pass",
        }
        for index in range(5)
    ]
    summary["points"] = points
    monkeypatch.setattr(benchmark_report, "MAX_HTML_POINT_ROWS", 2)

    artifacts = generate_sweep_report(summary, tmp_path / "report")
    report = json.loads(artifacts.report_json.read_text(encoding="utf-8"))
    rendered = artifacts.report_html.read_text(encoding="utf-8")

    assert report["sweep_summary"]["points"] == points
    assert report["presentation"]["point_rows"]["html_limit"] == 2
    assert "HTML shows 2 deterministically sampled rows from 5 recorded points" in rendered
    assert "Recorded sweep point table (2 of 5 row(s))" in rendered


def test_sweep_chart_axis_never_hides_a_recorded_negative_value() -> None:
    positive_lower, _ = benchmark_report._axis_bounds(
        [39.99, 40.01],
        preserve_unipolar_zero=True,
    )
    zero_lower, _ = benchmark_report._axis_bounds(
        [0.0, 1.0],
        preserve_unipolar_zero=True,
    )
    negative_lower, _ = benchmark_report._axis_bounds(
        [-1e-12, 1.0],
        preserve_unipolar_zero=True,
    )
    zero_only_bounds = benchmark_report._axis_bounds(
        [0.0],
        preserve_unipolar_zero=True,
    )

    assert positive_lower > 0.0
    assert zero_lower == 0.0
    assert negative_lower < 0.0
    assert zero_only_bounds == (0.0, 1.0)


def test_sweep_report_renders_a_single_zero_current_point(tmp_path: Path) -> None:
    artifacts = generate_sweep_report(
        {
            "schema": "pear-sim/sweep-summary/v0.1",
            "points": [
                {
                    "point_id": "boundary",
                    "load_ratio_to_bcm": 1.0,
                    "mean_output_voltage_v": 48.0,
                    "minimum_inductor_current_a": 0.0,
                    "regime": "BCM",
                    "status": "pass",
                }
            ],
        },
        tmp_path / "single-point-report",
    )

    rendered = artifacts.report_html.read_text(encoding="utf-8")
    assert "boundary" in rendered
    assert "Minimum iL [A]" in rendered
    assert "BCM · R/R_BCM = 1" in rendered


def test_sweep_chart_breaks_lines_and_marks_a_retained_missing_point(tmp_path: Path) -> None:
    points = [
        {
            "load_ratio_to_bcm": 0.5,
            "mean_output_voltage_v": 20.0,
            "minimum_inductor_current_a": 2.0,
            "regime": "CCM",
            "status": "pass",
        },
        {
            "load_ratio_to_bcm": 1.0,
            "mean_output_voltage_v": None,
            "minimum_inductor_current_a": None,
            "regime": "BCM",
            "status": "not-run",
        },
        {
            "load_ratio_to_bcm": 2.0,
            "mean_output_voltage_v": 28.0,
            "minimum_inductor_current_a": 0.0,
            "regime": "DCM",
            "status": "pass",
        },
    ]

    artifacts = generate_sweep_report(
        {"schema": "pear-sim/sweep-summary/v0.1", "points": points},
        tmp_path / "report",
    )
    rendered = artifacts.report_html.read_text(encoding="utf-8")

    assert rendered.count('data-chart-role="missing-point"') == 2
    assert "status=not-run" in rendered
    assert "<polyline" not in rendered


def test_bcm_label_anchors_inward_when_boundary_is_at_right_edge(tmp_path: Path) -> None:
    points = [
        {
            "load_ratio_to_bcm": 0.5,
            "mean_output_voltage_v": 20.0,
            "minimum_inductor_current_a": 2.0,
            "regime": "CCM",
            "status": "pass",
        },
        {
            "load_ratio_to_bcm": 1.0,
            "mean_output_voltage_v": 20.0,
            "minimum_inductor_current_a": 0.0,
            "regime": "BCM",
            "status": "pass",
        },
    ]

    artifacts = generate_sweep_report(
        {"schema": "pear-sim/sweep-summary/v0.1", "points": points},
        tmp_path / "report",
    )
    rendered = artifacts.report_html.read_text(encoding="utf-8")

    assert re.search(
        r'data-chart-role="bcm-label"[^>]+text-anchor="end"',
        rendered,
    )


def test_sweep_report_rejects_nonfinite_oversized_and_unowned_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(BenchmarkReportError, match="must be a mapping"):
        generate_sweep_report([], tmp_path / "not-a-mapping")  # type: ignore[arg-type]
    with pytest.raises(BenchmarkReportError, match="finite JSON data"):
        generate_sweep_report({"value": float("nan")}, tmp_path / "nonfinite")
    with pytest.raises(BenchmarkReportError, match="source_artifact_sha256"):
        generate_sweep_report({}, tmp_path / "bad-digest", source_artifact_sha256="BAD")

    monkeypatch.setattr(benchmark_report, "MAX_SWEEP_SUMMARY_BYTES", 10)
    with pytest.raises(BenchmarkReportError, match="limit is 10 bytes"):
        generate_sweep_report({"payload": "larger than ten bytes"}, tmp_path / "large")

    destination = tmp_path / "owned"
    destination.mkdir()
    (destination / "user-notes.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(BenchmarkReportError, match="files not owned by PEARSim"):
        generate_sweep_report({}, destination)
