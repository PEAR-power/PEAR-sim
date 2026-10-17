"""Bounded, deterministic human report for parameter-sweep summaries.

This module is intentionally presentation-only.  It accepts an already produced
sweep-summary mapping, preserves it verbatim in ``sweep-report.json``, and never
recomputes benchmark gates or validation status.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any

from pear_sim import __version__
from pear_sim.reporting.report_svg import SvgError, sanitize_schematic_svg


SWEEP_REPORT_SCHEMA = "pear-sim/sweep-report/v0.1"
MAX_SWEEP_SUMMARY_BYTES = 16 * 1024 * 1024
MAX_HTML_POINT_ROWS = 500
MAX_HTML_GATE_ROWS = 100
MAX_HTML_FAILURE_ROWS = 100
MAX_HTML_ARTIFACT_ROWS = 500
_OWNED_OUTPUT_FILES = {
    "artifact-manifest.json",
    "schematic.svg",
    "sweep-report.html",
    "sweep-report.json",
}
_KNOWN_STATUSES = {"pass", "fail", "not-run", "not-applicable"}
_REGIME_ORDER = {"ccm": 0, "bcm": 1, "dcm": 2}
_REGIME_COLORS = {
    "ccm": "#2563eb",
    "bcm": "#d97706",
    "dcm": "#0f766e",
    "unclassified": "#64748b",
}


class BenchmarkReportError(ValueError):
    """Raised when a sweep summary cannot produce a trustworthy bounded report."""

    code = "report.sweep_invalid"


@dataclass(frozen=True)
class SweepReportArtifacts:
    """Paths written by :func:`generate_sweep_report`."""

    report_json: Path
    report_html: Path
    artifact_manifest: Path | None
    schematic_svg: Path | None


def generate_sweep_report(
    sweep_summary: Mapping[str, Any],
    output_dir: str | Path,
    *,
    write_manifest: bool = True,
    source_artifact_sha256: str | None = None,
    schematic_svg: str | Path | None = None,
    schematic_caption: str | None = None,
) -> SweepReportArtifacts:
    """Write a deterministic offline report for a recorded sweep summary.

    The renderer recognizes the following optional point fields for the compact
    table and chart: ``r_ohm``/``load_ohm``,
    ``r_over_r_bcm``/``load_ratio_to_bcm``, ``vout_mean_v`` or recorded
    ``metrics.mean_output_voltage_v`` (or explicit signed equivalent),
    ``minimum_inductor_current_a``, ``regime``,
    ``status``, and ``artifacts``. Missing or non-numeric chart fields are displayed
    as unavailable rather than inferred. Top-level ``worst_gates``, ``failures``,
    and ``artifacts`` are likewise copied for human inspection without recomputing
    their meaning.
    """

    if not isinstance(sweep_summary, Mapping):
        raise BenchmarkReportError("sweep_summary must be a mapping")
    summary = _json_copy(dict(sweep_summary))
    summary_bytes = _canonical_json_bytes(summary)
    if len(summary_bytes) > MAX_SWEEP_SUMMARY_BYTES:
        raise BenchmarkReportError(
            f"Sweep summary is {len(summary_bytes)} bytes; limit is {MAX_SWEEP_SUMMARY_BYTES} bytes"
        )
    if source_artifact_sha256 is not None and not _is_sha256(source_artifact_sha256):
        raise BenchmarkReportError(
            "source_artifact_sha256 must be a lowercase 64-character SHA-256 digest"
        )
    sanitized_schematic = _load_schematic(schematic_svg)
    if schematic_caption is not None and not isinstance(schematic_caption, str):
        raise BenchmarkReportError("schematic_caption must be a string or None")
    caption = (
        schematic_caption
        if schematic_caption is not None
        else (
            "Nominal source topology used as the sweep template; point-specific "
            "parameter values remain authoritative in the recorded table."
        )
    )

    destination = Path(output_dir).resolve()
    _prepare_destination(destination)
    report_json_path = destination / "sweep-report.json"
    report_html_path = destination / "sweep-report.html"
    manifest_path = destination / "artifact-manifest.json"
    schematic_path = destination / "schematic.svg"
    if sanitized_schematic is None:
        schematic_path.unlink(missing_ok=True)
        schematic_record = None
    else:
        _write_text(schematic_path, sanitized_schematic)
        schematic_record = _artifact_record(schematic_path)

    summary_semantic_sha256 = hashlib.sha256(summary_bytes).hexdigest()
    report: dict[str, Any] = {
        "schema": SWEEP_REPORT_SCHEMA,
        "generator": {"name": "pear_sim", "version": __version__},
        "claim": {
            "level": "exploratory",
            "validation_claim": False,
            "statement": (
                "This report presents recorded sweep evidence for exploration; "
                "it is not a validation or hardware sign-off claim."
            ),
        },
        "source": {
            "schema": summary.get("schema"),
            "sweep_summary_semantic_sha256": summary_semantic_sha256,
            "sweep_summary_artifact_sha256": source_artifact_sha256,
        },
        "sweep_summary": summary,
        "presentation": {
            "point_rows": {
                "recorded": len(_list_value(summary.get("points"))),
                "html_limit": MAX_HTML_POINT_ROWS,
                "selection": "deterministic_even_index_sampling_with_endpoints",
            },
            "gate_rows_html_limit": MAX_HTML_GATE_ROWS,
            "failure_rows_html_limit": MAX_HTML_FAILURE_ROWS,
            "artifact_rows_html_limit": MAX_HTML_ARTIFACT_ROWS,
        },
        "files": {
            "html": "sweep-report.html",
            "manifest": "artifact-manifest.json" if write_manifest else None,
        },
    }
    if schematic_record is not None:
        report["source"]["schematic"] = schematic_record
        report["presentation"]["schematic"] = {
            "caption": caption,
            "role": "nominal_source_topology_template",
        }
        report["files"]["schematic"] = "schematic.svg"
    report_json_text = _pretty_json(report)
    report_html_text = _render_html(report, sanitized_schematic)
    _write_text(report_json_path, report_json_text)
    _write_text(report_html_path, report_html_text)

    artifact_manifest: Path | None = None
    if write_manifest:
        records = [
            _artifact_record(report_html_path),
            _artifact_record(report_json_path),
        ]
        if schematic_record is not None:
            records.append(schematic_record)
        records.sort(key=lambda record: str(record["path"]))
        bundle_sha256 = hashlib.sha256(_canonical_json_bytes(records)).hexdigest()
        manifest = {
            "schema": "pear-sim/artifact-manifest/v0.1",
            "subject": {
                "kind": "sweep-report",
                "sha256": bundle_sha256,
                "sweep_summary_semantic_sha256": summary_semantic_sha256,
            },
            "artifacts": records,
        }
        if source_artifact_sha256 is not None:
            manifest["subject"]["sweep_summary_artifact_sha256"] = source_artifact_sha256
        _write_text(manifest_path, _pretty_json(manifest))
        artifact_manifest = manifest_path
    else:
        manifest_path.unlink(missing_ok=True)

    return SweepReportArtifacts(
        report_json=report_json_path,
        report_html=report_html_path,
        artifact_manifest=artifact_manifest,
        schematic_svg=schematic_path if schematic_record is not None else None,
    )


def _load_schematic(source: str | Path | None) -> str | None:
    if source is None:
        return None
    try:
        if isinstance(source, Path):
            if source.is_symlink() or not source.is_file():
                raise BenchmarkReportError(
                    f"schematic_svg is not a regular file: {source}"
                )
            raw = source.read_text(encoding="utf-8")
        elif isinstance(source, str):
            raw = source
        else:
            raise BenchmarkReportError("schematic_svg must be an SVG string, Path, or None")
        return sanitize_schematic_svg(raw)
    except (OSError, UnicodeError, SvgError) as exc:
        raise BenchmarkReportError(f"Unable to use schematic SVG: {exc}") from exc


def _prepare_destination(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise BenchmarkReportError(f"Sweep report output is not a directory: {destination}")
        unexpected = sorted(
            path.name
            for path in destination.iterdir()
            if path.name not in _OWNED_OUTPUT_FILES or path.is_symlink() or not path.is_file()
        )
        if unexpected:
            raise BenchmarkReportError(
                "Sweep report output contains files not owned by PEARSim: " + ", ".join(unexpected)
            )
    else:
        destination.mkdir(parents=True)


def _json_copy(value: Any) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BenchmarkReportError(f"sweep_summary must be finite JSON data: {exc}") from exc
    if not isinstance(decoded, dict):
        raise BenchmarkReportError("sweep_summary must serialize to a JSON object")
    return decoded


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _pretty_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def _write_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _render_html(report: Mapping[str, Any], schematic_svg: str | None = None) -> str:
    summary_value = report.get("sweep_summary")
    summary = summary_value if isinstance(summary_value, dict) else {}
    points = _list_value(summary.get("points"))
    rendered_points = _bounded_rows(points, MAX_HTML_POINT_ROWS)
    benchmark_id = _first_recorded(
        summary,
        "benchmark_id",
        "id",
        "name",
        "kind",
        default="unnamed sweep",
    )
    recorded_summary_value = summary.get("summary")
    recorded_summary = recorded_summary_value if isinstance(recorded_summary_value, dict) else {}
    status = _first_recorded(summary, "overall_status", "status", default=None)
    if status is None:
        status = _first_recorded(recorded_summary, "status", default="not-recorded")
    bcm_value = summary.get("bcm")
    bcm = bcm_value if isinstance(bcm_value, dict) else {}
    boundary_value = summary.get("boundary")
    boundary = boundary_value if isinstance(boundary_value, dict) else {}
    bcm_resistance = _first_recorded(
        bcm,
        "r_bcm_ohm",
        "resistance_ohm",
        default=None,
    )
    if bcm_resistance is None:
        bcm_resistance = _first_recorded(summary, "boundary_load_ohm", default=None)
    if bcm_resistance is None:
        bcm_resistance = _first_recorded(boundary, "load_ohm", default=None)

    chart_html = _sweep_chart(rendered_points)
    point_table = _point_table(points, rendered_points)
    gates_html = _gate_section(summary.get("worst_gates"))
    failures_html = _failure_section(summary.get("failures"))
    artifacts_html = _artifact_section(summary.get("artifacts"))
    schematic_html = _schematic_section(report, schematic_svg)
    metadata_json = _escape(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    source_value = report.get("source")
    source = source_value if isinstance(source_value, dict) else {}
    artifact_digest = source.get("sweep_summary_artifact_sha256")
    if artifact_digest is None:
        source_digest_label = "Canonical summary digest (SHA-256)"
        source_digest = source.get("sweep_summary_semantic_sha256")
    else:
        source_digest_label = "Source summary artifact SHA-256"
        source_digest = artifact_digest

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; script-src 'none'; connect-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'">
  <title>PEARSim exploratory sweep report — {_escape(benchmark_id)}</title>
  <style>
    :root {{ color-scheme: light; font-family: Inter, ui-sans-serif, system-ui, sans-serif; color: #102a43; background: #f5f7fa; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: #f5f7fa; line-height: 1.45; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 22px 64px; }}
    header, section {{ background: #fff; border: 1px solid #d9e2ec; border-radius: 12px; padding: 22px; margin-bottom: 18px; box-shadow: 0 2px 8px rgba(16, 42, 67, .05); }}
    h1, h2, h3 {{ margin-top: 0; }}
    h1 {{ margin-bottom: 6px; font-size: 1.75rem; }}
    h2 {{ font-size: 1.25rem; border-bottom: 1px solid #e6edf3; padding-bottom: 9px; }}
    .subtitle, .empty, figcaption {{ color: #52606d; }}
    .claim {{ display: inline-block; margin-top: 12px; padding: 5px 9px; border-radius: 999px; background: #fff3c4; color: #7c4d00; font-weight: 700; font-size: .78rem; letter-spacing: .04em; }}
    .notice {{ border-left: 4px solid #f0b429; padding: 10px 13px; background: #fffbea; margin-top: 16px; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 18px; }}
    .card {{ border: 1px solid #d9e2ec; border-radius: 9px; padding: 12px; min-width: 0; }}
    .card small {{ display: block; color: #627d98; text-transform: uppercase; letter-spacing: .04em; }}
    .card strong, .card code {{ overflow-wrap: anywhere; }}
    .status {{ display: inline-block; border-radius: 999px; padding: 2px 8px; font-weight: 700; white-space: nowrap; }}
    .status-pass {{ color: #065f46; background: #d1fae5; }}
    .status-fail {{ color: #991b1b; background: #fee2e2; }}
    .status-not-run {{ color: #374151; background: #e5e7eb; }}
    .status-not-applicable {{ color: #5b21b6; background: #ede9fe; }}
    .status-unknown {{ color: #334155; background: #e2e8f0; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
    th, td {{ border-bottom: 1px solid #e6edf3; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ color: #334e68; background: #f5f7fa; }}
    td.number {{ text-align: right; font-variant-numeric: tabular-nums; }}
    code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
    code {{ overflow-wrap: anywhere; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; background: #102a43; color: #f0f4f8; padding: 14px; border-radius: 8px; font-size: .79rem; }}
    .scroll {{ overflow-x: auto; }}
    .chart svg {{ display: block; width: 100%; height: auto; background: #fff; }}
    .schematic svg {{ display: block; width: 100%; height: auto; max-height: 680px; background: #fff; }}
    details summary {{ cursor: pointer; font-weight: 700; }}
    footer {{ color: #627d98; text-align: center; font-size: .8rem; padding-top: 12px; }}
    @media (max-width: 520px) {{ main {{ padding: 14px 10px 40px; }} header, section {{ padding: 14px; }} .cards {{ grid-template-columns: 1fr; }} }}
    @media print {{ body {{ background: #fff; }} header, section {{ box-shadow: none; }} }}
  </style>
</head>
<body>
<main>
  <header>
    <h1>PEARSim sweep report</h1>
    <p class="subtitle">{_escape(benchmark_id)}</p>
    <span class="claim">EXPLORATORY — NOT VALIDATION</span>
    <p class="notice">Recorded sweep evidence only. The renderer does not recompute gates, infer missing regimes, or authorize hardware sign-off.</p>
    <div class="cards">
      <div class="card"><small>Overall recorded status</small>{_status_badge(status)}</div>
      <div class="card"><small>Recorded sweep points</small><strong>{len(points)}</strong></div>
      <div class="card"><small>R<sub>BCM</sub> [Ω]</small><strong>{_escape(_display(bcm_resistance))}</strong></div>
      <div class="card"><small>{_escape(source_digest_label)}</small><code>{_escape(_display(source_digest))}</code></div>
    </div>
  </header>
  {schematic_html}
  <section>
    <h2>Operating-regime sweep</h2>
    <p class="subtitle">Recorded R/R<sub>BCM</sub>, mean Vout, minimum inductor current, and regime. The vertical line marks R/R<sub>BCM</sub> = 1.</p>
    <div class="chart">{chart_html}</div>
    {point_table}
  </section>
  {gates_html}
  {failures_html}
  {artifacts_html}
  <section>
    <h2>Recorded source summary</h2>
    <details><summary>Complete sweep-summary JSON</summary><pre>{metadata_json}</pre></details>
  </section>
  <footer>Generated deterministically by PEARSim {_escape(__version__)} · schema {_escape(SWEEP_REPORT_SCHEMA)} · no scripts or network resources</footer>
</main>
</body>
</html>
"""


def _schematic_section(report: Mapping[str, Any], schematic_svg: str | None) -> str:
    if schematic_svg is None:
        return ""
    presentation_value = report.get("presentation")
    presentation = presentation_value if isinstance(presentation_value, dict) else {}
    schematic_value = presentation.get("schematic")
    schematic = schematic_value if isinstance(schematic_value, dict) else {}
    caption = _display(
        schematic.get(
            "caption",
            "Nominal source topology used as the sweep template.",
        )
    )
    return f"""
  <section>
    <h2>Source topology</h2>
    <figure class="schematic">
      {schematic_svg}
      <figcaption>{_escape(caption)}</figcaption>
    </figure>
  </section>"""


def _point_table(points: list[Any], rendered_points: list[Any]) -> str:
    if not points:
        return '<p class="empty">No sweep points were recorded.</p>'
    rows: list[str] = []
    for position, raw_point in enumerate(rendered_points):
        if not isinstance(raw_point, dict):
            rows.append(
                '<tr><td colspan="9"><code>' + _escape(_display(raw_point)) + "</code></td></tr>"
            )
            continue
        artifacts = _compact_artifact_pointers(raw_point.get("artifacts"))
        point_id = _first_recorded(raw_point, "index", "point_id", default=position)
        rows.append(
            "<tr>"
            f'<td class="number">{_escape(_display(point_id))}</td>'
            f'<td class="number">{_escape(_display(_point_load(raw_point)))}</td>'
            f'<td class="number">{_escape(_display(_point_ratio(raw_point)))}</td>'
            f'<td class="number">{_escape(_display(_point_vout(raw_point)))}</td>'
            f'<td class="number">{_escape(_display(_point_minimum_current(raw_point)))}</td>'
            f"<td><code>{_escape(_display(_point_regime(raw_point)))}</code></td>"
            f"<td>{_status_badge(raw_point.get('status'))}</td>"
            f'<td class="number">{_escape(_display(raw_point.get("runtime_s")))}</td>'
            f"<td>{artifacts}</td>"
            "</tr>"
        )
    truncation = ""
    if len(rendered_points) < len(points):
        truncation = (
            f'<p class="notice">HTML shows {len(rendered_points)} deterministically '
            f"sampled rows from {len(points)} recorded points, including both endpoints. "
            "sweep-report.json retains the complete summary.</p>"
        )
    return (
        truncation + f"<details><summary>Recorded sweep point table ({len(rendered_points)} of "
        f"{len(points)} row(s))</summary>"
        '<div class="scroll"><table><thead><tr><th>#</th><th>R [Ω]</th>'
        "<th>R/R<sub>BCM</sub></th><th>Mean Vout [V]</th><th>Minimum iL [A]</th>"
        "<th>Regime</th><th>Status</th><th>Runtime [s]</th><th>Artifacts</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div></details>"
    )


def _gate_section(raw_gates: Any) -> str:
    gates = _list_value(raw_gates)
    if not gates:
        return """
  <section>
    <h2>Worst recorded gates</h2>
    <p class="empty">No worst-gate records were provided; no gate result is inferred.</p>
  </section>"""
    rendered = gates[:MAX_HTML_GATE_ROWS]
    rows: list[str] = []
    for raw_gate in rendered:
        if not isinstance(raw_gate, dict):
            rows.append(f'<tr><td colspan="7"><code>{_escape(_display(raw_gate))}</code></td></tr>')
            continue
        rows.append(
            "<tr>"
            f"<td><code>{_escape(_display(raw_gate.get('gate')))}</code></td>"
            f"<td>{_status_badge(raw_gate.get('status'))}</td>"
            f'<td class="number">{_escape(_display(raw_gate.get("value")))}</td>'
            f'<td class="number">{_escape(_display(raw_gate.get("limit")))}</td>'
            f"<td>{_escape(_display(raw_gate.get('unit')))}</td>"
            f"<td><code>{_escape(_display(raw_gate.get('point_id')))}</code></td>"
            f"<td>{_escape(_display(raw_gate.get('message')))}</td>"
            "</tr>"
        )
    truncation = _bounded_notice("worst-gate", len(rendered), len(gates))
    return f"""
  <section>
    <h2>Worst recorded gates</h2>
    <p class="subtitle">Producer ordering and status are preserved; the report does not rank or recompute gates.</p>
    {truncation}
    <div class="scroll"><table><thead><tr><th>Gate</th><th>Status</th><th>Value</th><th>Limit</th><th>Unit</th><th>Point</th><th>Recorded detail</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
  </section>"""


def _failure_section(raw_failures: Any) -> str:
    failures = _list_value(raw_failures)
    if not failures:
        return """
  <section>
    <h2>Recorded failures</h2>
    <p class="empty">No failure records were provided. This is not itself a pass claim.</p>
  </section>"""
    rendered = failures[:MAX_HTML_FAILURE_ROWS]
    rows: list[str] = []
    for raw_failure in rendered:
        if not isinstance(raw_failure, dict):
            rows.append(
                f'<tr><td colspan="6"><code>{_escape(_display(raw_failure))}</code></td></tr>'
            )
            continue
        rows.append(
            "<tr>"
            f"<td><code>{_escape(_display(raw_failure.get('id')))}</code></td>"
            f"<td>{_status_badge(raw_failure.get('status'))}</td>"
            f"<td><code>{_escape(_display(raw_failure.get('point_id')))}</code></td>"
            f"<td><code>{_escape(_display(raw_failure.get('gate')))}</code></td>"
            f"<td>{_escape(_display(raw_failure.get('message')))}</td>"
            f"<td>{_compact_artifact_pointers(raw_failure.get('artifacts'))}</td>"
            "</tr>"
        )
    truncation = _bounded_notice("failure", len(rendered), len(failures))
    return f"""
  <section>
    <h2>Recorded failures</h2>
    {truncation}
    <div class="scroll"><table><thead><tr><th>ID</th><th>Status</th><th>Point</th><th>Gate</th><th>Recorded message</th><th>Artifacts</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div>
  </section>"""


def _artifact_section(raw_artifacts: Any) -> str:
    records = _artifact_rows(raw_artifacts)
    if not records:
        return """
  <section>
    <h2>Artifact pointers</h2>
    <p class="empty">No sweep-level artifact pointers were recorded.</p>
  </section>"""
    rendered = records[:MAX_HTML_ARTIFACT_ROWS]
    rows = "".join(
        "<tr>"
        f"<td><code>{_escape(role)}</code></td>"
        f"<td><code>{_escape(pointer)}</code></td>"
        f"<td><code>{_escape(digest)}</code></td>"
        "</tr>"
        for role, pointer, digest in rendered
    )
    truncation = _bounded_notice("artifact", len(rendered), len(records))
    return f"""
  <section>
    <h2>Artifact pointers</h2>
    <p class="subtitle">Pointers are displayed as recorded text; the offline report does not resolve or fetch them.</p>
    {truncation}
    <div class="scroll"><table><thead><tr><th>Role</th><th>Recorded pointer</th><th>SHA-256</th></tr></thead><tbody>{rows}</tbody></table></div>
  </section>"""


def _sweep_chart(points: list[Any]) -> str:
    numeric_points: list[dict[str, Any]] = []
    for position, raw_point in enumerate(points):
        if not isinstance(raw_point, dict):
            continue
        x = _finite_float(_point_ratio(raw_point))
        if x is None:
            continue
        vout = _finite_float(_point_vout(raw_point))
        minimum_current = _finite_float(_point_minimum_current(raw_point))
        numeric_points.append(
            {
                "position": position,
                "x": x,
                "vout": vout,
                "minimum_current": minimum_current,
                "regime": _regime(_point_regime(raw_point)),
                "status": _status(raw_point.get("status"))[0],
            }
        )
    if not numeric_points or not any(
        point["vout"] is not None or point["minimum_current"] is not None
        for point in numeric_points
    ):
        return (
            '<p class="empty">No points contain finite recorded '
            "R/R<sub>BCM</sub> and output/current values.</p>"
        )
    numeric_points.sort(key=lambda point: (point["x"], point["position"]))
    x_values = [point["x"] for point in numeric_points]
    logarithmic_x = all(value > 0.0 for value in x_values)
    transformed_x_values = [math.log10(value) for value in x_values] if logarithmic_x else x_values
    boundary_x = 0.0 if logarithmic_x else 1.0
    x_min, x_max = _axis_bounds(
        [*transformed_x_values, boundary_x],
        preserve_unipolar_zero=not logarithmic_x,
    )
    vout_values = [point["vout"] for point in numeric_points if point["vout"] is not None]
    current_values = [
        point["minimum_current"] for point in numeric_points if point["minimum_current"] is not None
    ]

    width = 920.0
    left = 82.0
    right = 28.0
    panel_height = 142.0
    top_one = 42.0
    top_two = 260.0
    plot_width = width - left - right

    def x_position(value: float) -> float:
        transformed = math.log10(value) if logarithmic_x else value
        return left + (transformed - x_min) / (x_max - x_min) * plot_width

    parts = [
        '<svg viewBox="0 0 920 480" role="img" '
        'aria-label="Sweep of output voltage and minimum inductor current versus R over R BCM">',
        "<title>Operating-regime sweep</title>",
        "<desc>Two aligned panels show recorded mean output voltage and minimum inductor "
        "current. Point shape and label identify CCM, BCM, DCM, or unclassified regime. "
        + (
            "The shared R over R BCM axis uses a logarithmic scale. "
            if logarithmic_x
            else "The shared R over R BCM axis uses a linear scale. "
        )
        + "A vertical line marks R over R BCM equal to one. Crosses mark retained "
        "points without a plottable metric and break the connecting line.</desc>",
    ]
    for top, values, label, key in (
        (top_one, vout_values, "Mean Vout [V]", "vout"),
        (top_two, current_values, "Minimum iL [A]", "minimum_current"),
    ):
        if values:
            y_min, y_max = _axis_bounds(values, preserve_unipolar_zero=True)
        else:
            y_min, y_max = 0.0, 1.0

        def y_position(value: float, *, low: float = y_min, high: float = y_max) -> float:
            return top + (high - value) / (high - low) * panel_height

        for tick in range(5):
            fraction = tick / 4.0
            y = top + fraction * panel_height
            value = y_max - fraction * (y_max - y_min)
            parts.append(
                f'<line x1="{left:.3f}" y1="{y:.3f}" x2="{left + plot_width:.3f}" '
                f'y2="{y:.3f}" stroke="#d9e2ec" stroke-width="1" />'
            )
            parts.append(
                f'<text x="{left - 10:.3f}" y="{y + 4:.3f}" text-anchor="end" '
                f'font-size="11" fill="#52606d">{_escape(_format_number(value))}</text>'
            )
        parts.append(
            f'<text data-chart-role="panel-title" x="{left:.3f}" y="{top - 14:.3f}" font-size="13" '
            f'font-weight="700" fill="#102a43">{_escape(label)}</text>'
        )
        parts.append(
            f'<line x1="{left:.3f}" y1="{top + panel_height:.3f}" '
            f'x2="{left + plot_width:.3f}" y2="{top + panel_height:.3f}" '
            'stroke="#334e68" stroke-width="1.4" />'
        )
        parts.append(
            f'<line x1="{left:.3f}" y1="{top:.3f}" x2="{left:.3f}" '
            f'y2="{top + panel_height:.3f}" stroke="#334e68" stroke-width="1.4" />'
        )
        bcm_x = x_position(1.0)
        parts.append(
            f'<line x1="{bcm_x:.3f}" y1="{top:.3f}" x2="{bcm_x:.3f}" '
            f'y2="{top + panel_height:.3f}" stroke="#7c3aed" stroke-width="1.5" '
            'stroke-dasharray="5 4" />'
        )
        plottable = [point for point in numeric_points if point[key] is not None]
        if plottable:
            contiguous_groups: list[list[dict[str, Any]]] = []
            current_group: list[dict[str, Any]] = []
            for point in numeric_points:
                if point[key] is None:
                    if current_group:
                        contiguous_groups.append(current_group)
                        current_group = []
                else:
                    current_group.append(point)
            if current_group:
                contiguous_groups.append(current_group)
            for group in contiguous_groups:
                if len(group) < 2:
                    continue
                line_points = " ".join(
                    f"{x_position(point['x']):.3f},{y_position(point[key]):.3f}" for point in group
                )
                parts.append(
                    f'<polyline points="{line_points}" fill="none" stroke="#94a3b8" '
                    'stroke-width="1.4" stroke-linejoin="round" />'
                )
            for point in plottable:
                x = x_position(point["x"])
                y = y_position(point[key])
                parts.append(
                    _chart_marker(
                        x,
                        y,
                        point["regime"],
                        (
                            f"R/R_BCM={_format_number(point['x'])}; "
                            f"{label}={_format_number(point[key])}; "
                            f"regime={point['regime']}; status={point['status']}"
                        ),
                    )
                )
        if not plottable:
            parts.append(
                f'<text x="{left + plot_width / 2:.3f}" '
                f'y="{top + panel_height / 2:.3f}" text-anchor="middle" '
                'font-size="12" fill="#52606d">not recorded</text>'
            )
        for point in numeric_points:
            if point[key] is None:
                parts.append(
                    _chart_missing_marker(
                        x_position(point["x"]),
                        top + panel_height / 2.0,
                        (
                            f"R/R_BCM={_format_number(point['x'])}; {label}=not recorded; "
                            f"regime={point['regime']}; status={point['status']}"
                        ),
                    )
                )

    for tick in range(5):
        fraction = tick / 4.0
        x = left + fraction * plot_width
        transformed_value = x_min + fraction * (x_max - x_min)
        value = 10.0**transformed_value if logarithmic_x else transformed_value
        parts.append(
            f'<text x="{x:.3f}" y="425" text-anchor="middle" font-size="11" '
            f'fill="#52606d">{_escape(_format_number(value))}</text>'
        )
    bcm_x = x_position(1.0)
    if bcm_x > left + plot_width / 2.0:
        bcm_label_x = bcm_x - 5.0
        bcm_label_anchor = "end"
    else:
        bcm_label_x = bcm_x + 5.0
        bcm_label_anchor = "start"
    parts.append(
        f'<text data-chart-role="bcm-label" x="{bcm_label_x:.3f}" y="218" '
        f'text-anchor="{bcm_label_anchor}" font-size="11" '
        'fill="#5b21b6">BCM · R/R_BCM = 1</text>'
    )
    x_axis_label = "R / R_BCM (log scale)" if logarithmic_x else "R / R_BCM"
    parts.append(
        f'<text x="892" y="474" text-anchor="end" font-size="11" '
        f'fill="#52606d">{x_axis_label}</text>'
    )
    legend_x = left
    for regime in ("ccm", "bcm", "dcm", "unclassified"):
        parts.append(_chart_marker(legend_x + 7, 454.0, regime, regime.upper()))
        parts.append(
            f'<text x="{legend_x + 18:.3f}" y="458" font-size="11" '
            f'fill="#52606d">{_escape(regime.upper())}</text>'
        )
        legend_x += 92.0
    parts.append("</svg>")
    return "".join(parts)


def _chart_marker(x: float, y: float, regime: str, title: str) -> str:
    color = _REGIME_COLORS[regime]
    title_html = f"<title>{_escape(title)}</title>"
    if regime == "bcm":
        points = f"{x:.3f},{y - 5:.3f} {x + 5:.3f},{y:.3f} {x:.3f},{y + 5:.3f} {x - 5:.3f},{y:.3f}"
        return f'<polygon points="{points}" fill="{color}" stroke="#fff">{title_html}</polygon>'
    if regime == "dcm":
        points = f"{x:.3f},{y - 5:.3f} {x + 5:.3f},{y + 4:.3f} {x - 5:.3f},{y + 4:.3f}"
        return f'<polygon points="{points}" fill="{color}" stroke="#fff">{title_html}</polygon>'
    if regime == "unclassified":
        return (
            f'<rect x="{x - 4:.3f}" y="{y - 4:.3f}" width="8" height="8" '
            f'fill="{color}" stroke="#fff">{title_html}</rect>'
        )
    return f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4.5" fill="{color}" stroke="#fff">{title_html}</circle>'


def _chart_missing_marker(x: float, y: float, title: str) -> str:
    return (
        f'<g data-chart-role="missing-point" stroke="#b91c1c" stroke-width="2">'
        f"<title>{_escape(title)}</title>"
        f'<line x1="{x - 5:.3f}" y1="{y - 5:.3f}" '
        f'x2="{x + 5:.3f}" y2="{y + 5:.3f}" />'
        f'<line x1="{x - 5:.3f}" y1="{y + 5:.3f}" '
        f'x2="{x + 5:.3f}" y2="{y - 5:.3f}" />'
        "</g>"
    )


def _axis_bounds(
    values: Sequence[float],
    *,
    preserve_unipolar_zero: bool,
) -> tuple[float, float]:
    lower = min(values)
    upper = max(values)
    if lower == upper:
        if preserve_unipolar_zero and lower == 0.0:
            return 0.0, 1.0
        padding = max(abs(lower) * 0.05, 1.0)
        padded_lower = lower - padding
        padded_upper = upper + padding
        if preserve_unipolar_zero and lower >= 0.0 and padded_lower < 0.0:
            padded_lower = 0.0
        if preserve_unipolar_zero and upper <= 0.0 and padded_upper > 0.0:
            padded_upper = 0.0
        return padded_lower, padded_upper
    padding = (upper - lower) * 0.05
    padded_lower = lower - padding
    padded_upper = upper + padding
    if preserve_unipolar_zero and lower >= 0.0 and padded_lower < 0.0:
        padded_lower = 0.0
    if preserve_unipolar_zero and upper <= 0.0 and padded_upper > 0.0:
        padded_upper = 0.0
    return padded_lower, padded_upper


def _bounded_rows(values: list[Any], maximum: int) -> list[Any]:
    if len(values) <= maximum:
        return values.copy()
    indices = sorted({round(index * (len(values) - 1) / (maximum - 1)) for index in range(maximum)})
    return [values[index] for index in indices]


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _first_recorded(
    values: Mapping[str, Any],
    *names: str,
    default: Any,
) -> Any:
    for name in names:
        if name in values:
            return values[name]
    return default


def _point_load(point: Mapping[str, Any]) -> Any:
    return _first_recorded(point, "r_ohm", "load_ohm", default=None)


def _point_ratio(point: Mapping[str, Any]) -> Any:
    return _first_recorded(
        point,
        "r_over_r_bcm",
        "load_ratio_to_bcm",
        default=None,
    )


def _point_vout(point: Mapping[str, Any]) -> Any:
    direct = _first_recorded(
        point,
        "vout_mean_v",
        "mean_output_voltage_v",
        "signed_mean_output_voltage_v",
        default=None,
    )
    if direct is not None:
        return direct
    metrics_value = point.get("metrics")
    metrics = metrics_value if isinstance(metrics_value, dict) else {}
    return _first_recorded(
        metrics,
        "mean_output_voltage_v",
        "signed_mean_output_voltage_v",
        default=None,
    )


def _point_minimum_current(point: Mapping[str, Any]) -> Any:
    direct = _first_recorded(point, "minimum_inductor_current_a", default=None)
    if direct is not None:
        return direct
    metrics_value = point.get("metrics")
    metrics = metrics_value if isinstance(metrics_value, dict) else {}
    return _first_recorded(metrics, "minimum_inductor_current_a", default=None)


def _point_regime(point: Mapping[str, Any]) -> Any:
    return _first_recorded(
        point,
        "regime",
        "observed_regime",
        "expected_regime",
        default=None,
    )


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _regime(value: Any) -> str:
    candidate = str(value).strip().lower() if value is not None else ""
    return candidate if candidate in _REGIME_ORDER else "unclassified"


def _status(value: Any) -> tuple[str, str]:
    display = _display(value) if value is not None else "not-recorded"
    normalized = display.strip().lower().replace("_", "-")
    css_status = normalized if normalized in _KNOWN_STATUSES else "unknown"
    return display, css_status


def _status_badge(value: Any) -> str:
    display, css_status = _status(value)
    return f'<strong class="status status-{css_status}">{_escape(display)}</strong>'


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.9g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return str(value)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _compact_artifact_pointers(value: Any) -> str:
    records = _artifact_rows(value)
    if not records:
        return "—"
    shown = records[:5]
    text = "; ".join(
        f"{role}={pointer}" + (f" [{digest}]" if digest != "—" else "")
        for role, pointer, digest in shown
    )
    if len(records) > len(shown):
        text += f"; … +{len(records) - len(shown)}"
    return f"<code>{_escape(text)}</code>"


def _artifact_rows(value: Any) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    if isinstance(value, dict):
        for role, raw_record in sorted(value.items(), key=lambda item: str(item[0])):
            if isinstance(raw_record, dict):
                pointer = _first_recorded(raw_record, "path", "file", "uri", default=raw_record)
                digest = raw_record.get("sha256")
            else:
                pointer = raw_record
                digest = None
            rows.append((str(role), _display(pointer), _display(digest)))
    elif isinstance(value, list):
        for index, raw_record in enumerate(value):
            if isinstance(raw_record, dict):
                role = _first_recorded(raw_record, "role", "kind", default=index)
                pointer = _first_recorded(raw_record, "path", "file", "uri", default=raw_record)
                digest = raw_record.get("sha256")
            else:
                role, pointer, digest = index, raw_record, None
            rows.append((_display(role), _display(pointer), _display(digest)))
    return rows


def _bounded_notice(label: str, rendered: int, recorded: int) -> str:
    if rendered >= recorded:
        return ""
    return (
        f'<p class="notice">HTML shows the first {rendered} of {recorded} recorded '
        f"{_escape(label)} rows. sweep-report.json retains the complete summary.</p>"
    )


def _format_number(value: float) -> str:
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 10_000 or magnitude < 0.001:
        return f"{value:.3e}"
    return f"{value:.5g}"


__all__ = [
    "BenchmarkReportError",
    "SWEEP_REPORT_SCHEMA",
    "SweepReportArtifacts",
    "generate_sweep_report",
]
