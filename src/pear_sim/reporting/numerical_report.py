"""Offline human projection of recorded canonical-buck numerical evidence.

The renderer is deliberately presentation-only: it copies finite JSON evidence,
selects bounded rows for display, and never recalculates solver or acceptance gates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
from typing import Any

from pear_sim import __version__


NUMERICAL_REPORT_SCHEMA = "pear-sim/numerical-evidence-report/v0.1"
MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
MAX_CONFIGURATION_ROWS = 100
MAX_NON_PASS_ROWS = 250
_OWNED_FILES = {
    "artifact-manifest.json",
    "numerical-report.html",
    "numerical-report.json",
}


class NumericalReportError(ValueError):
    """Raised when recorded evidence cannot be rendered safely."""

    code = "report.numerical_invalid"


@dataclass(frozen=True)
class NumericalReportArtifacts:
    report_json: Path
    report_html: Path
    artifact_manifest: Path


def generate_numerical_report(
    evidence: Mapping[str, Any],
    output_dir: str | Path,
    *,
    source_artifact_sha256: str,
) -> NumericalReportArtifacts:
    """Write bounded JSON and static HTML from an existing evidence record."""

    if not isinstance(evidence, Mapping):
        raise NumericalReportError("evidence must be a mapping")
    if not _is_sha256(source_artifact_sha256):
        raise NumericalReportError(
            "source_artifact_sha256 must be a lowercase 64-character SHA-256 digest"
        )
    copied = _finite_json_copy(dict(evidence))
    encoded = _canonical_json_bytes(copied)
    if len(encoded) > MAX_EVIDENCE_BYTES:
        raise NumericalReportError(
            f"Evidence is {len(encoded)} bytes; limit is {MAX_EVIDENCE_BYTES} bytes"
        )
    schema = copied.get("schema")
    if not isinstance(schema, str) or not schema.startswith("pear-sim/"):
        raise NumericalReportError("evidence must declare a versioned PEARSim schema")

    destination = Path(output_dir).resolve()
    _prepare_destination(destination)
    report_json_path = destination / "numerical-report.json"
    report_html_path = destination / "numerical-report.html"
    manifest_path = destination / "artifact-manifest.json"

    semantic_sha256 = hashlib.sha256(encoded).hexdigest()
    report = {
        "schema": NUMERICAL_REPORT_SCHEMA,
        "generator": {"name": "pear-sim", "version": __version__},
        "claim": {
            "level": "EXPLORATORY",
            "validation_claim": False,
            "compatibility_claim": False,
            "hardware_sign_off_allowed": False,
            "statement": (
                "This is numerical verification evidence for recorded equations and the "
                "declared canonical topology, not physical validation or hardware sign-off."
            ),
        },
        "source": {
            "schema": schema,
            "artifact_sha256": source_artifact_sha256,
            "semantic_sha256": semantic_sha256,
        },
        "presentation": {
            "configuration_rows_limit": MAX_CONFIGURATION_ROWS,
            "non_pass_rows_limit": MAX_NON_PASS_ROWS,
            "selection": "recorded-order prefix; raw evidence remains authoritative",
        },
        "evidence": copied,
        "files": {
            "html": report_html_path.name,
            "manifest": manifest_path.name,
        },
    }
    _write_text(report_json_path, _pretty_json(report))
    _write_text(report_html_path, _render_html(report))

    artifacts = [_artifact_record(report_html_path), _artifact_record(report_json_path)]
    artifacts.sort(key=lambda record: str(record["path"]))
    bundle_sha256 = hashlib.sha256(_canonical_json_bytes(artifacts)).hexdigest()
    manifest = {
        "schema": "pear-sim/artifact-manifest/v0.1",
        "subject": {
            "kind": "numerical-evidence-report",
            "sha256": bundle_sha256,
            "source_artifact_sha256": source_artifact_sha256,
            "source_semantic_sha256": semantic_sha256,
        },
        "artifacts": artifacts,
    }
    _write_text(manifest_path, _pretty_json(manifest))
    return NumericalReportArtifacts(report_json_path, report_html_path, manifest_path)


def _prepare_destination(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise NumericalReportError(f"Report output is not a directory: {destination}")
        unexpected = sorted(
            path.name
            for path in destination.iterdir()
            if path.name not in _OWNED_FILES or path.is_symlink() or not path.is_file()
        )
        if unexpected:
            raise NumericalReportError(
                "Report output contains files not owned by PEARSim: " + ", ".join(unexpected)
            )
    else:
        destination.mkdir(parents=True)


def _finite_json_copy(value: Any) -> dict[str, Any]:
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
        raise NumericalReportError(f"evidence must be finite JSON data: {exc}") from exc
    if not isinstance(decoded, dict):
        raise NumericalReportError("evidence must serialize to an object")
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
    return json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


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
    return {"path": path.name, "sha256": _sha256_file(path), "bytes": path.stat().st_size}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _records(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [record for record in value if isinstance(record, Mapping)]


def _display(value: Any) -> str:
    if value is None:
        return "not recorded"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.8g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _escape(value: Any) -> str:
    return html.escape(_display(value), quote=True)


def _status_class(value: Any) -> str:
    normalized = str(value).lower().replace("_", "-")
    return normalized if normalized in {"pass", "fail", "not-run", "warning"} else "unknown"


def _render_html(report: Mapping[str, Any]) -> str:
    evidence = _mapping(report.get("evidence"))
    source = _mapping(report.get("source"))
    profile = _mapping(evidence.get("profile"))
    summary = _mapping(evidence.get("summary"))
    platform = _mapping(evidence.get("platform"))
    status = evidence.get("status", summary.get("status", "not-recorded"))
    recorded_source = _mapping(evidence.get("source"))
    source_revision = recorded_source.get("revision", evidence.get("source_revision"))
    gate_count = summary.get("gate_count", summary.get("comparison_count"))
    configurations = _records(evidence.get("configurations"))
    cells = _records(evidence.get("cells"))
    non_pass = [cell for cell in cells if cell.get("status") != "pass"]
    failures = _records(evidence.get("failures"))

    study_rows: list[str] = []
    studies = _mapping(evidence.get("studies"))
    for name, raw in list(studies.items())[:100]:
        record = _mapping(raw)
        counts = _mapping(record.get("cell_status_counts"))
        study_rows.append(
            "<tr>"
            f"<th scope='row'>{_escape(name)}</th>"
            f"<td><span class='badge {_status_class(record.get('status'))}'>"
            f"{_escape(record.get('status'))}</span></td>"
            f"<td>{_escape(record.get('cell_count'))}</td>"
            f"<td>{_escape(counts)}</td>"
            "</tr>"
        )

    configuration_rows: list[str] = []
    for record in configurations[:MAX_CONFIGURATION_ROWS]:
        configuration_rows.append(
            "<tr>"
            f"<th scope='row'><code>{_escape(record.get('id'))}</code></th>"
            f"<td>{_escape(record.get('studies'))}</td>"
            f"<td>{_escape(record.get('solver'))}</td>"
            f"<td>{_escape(record.get('rtol'))}</td>"
            f"<td>{_escape(record.get('atol'))}</td>"
            f"<td>{_escape(record.get('max_step_over_period'))}</td>"
            f"<td>{_escape(record.get('output_step_over_period'))}</td>"
            "</tr>"
        )

    non_pass_rows: list[str] = []
    for cell in non_pass[:MAX_NON_PASS_ROWS]:
        gates = _records(cell.get("gates"))
        failing_gate_ids = [
            str(gate.get("id")) for gate in gates if gate.get("status") != "pass"
        ]
        non_pass_rows.append(
            "<tr>"
            f"<th scope='row'><code>{_escape(cell.get('id'))}</code></th>"
            f"<td><span class='badge {_status_class(cell.get('status'))}'>"
            f"{_escape(cell.get('status'))}</span></td>"
            f"<td>{_escape(cell.get('expected_regime'))}</td>"
            f"<td>{_escape(failing_gate_ids)}</td>"
            f"<td>{_escape(cell.get('error'))}</td>"
            "</tr>"
        )

    platform_records = _records(evidence.get("platforms"))
    platform_rows = "".join(
        "<tr>"
        f"<th scope='row'>{_escape(item.get('system') or item.get('id'))}</th>"
        f"<td>{_escape(item.get('machine'))}</td>"
        f"<td>{_escape(item.get('python'))}</td>"
        f"<td><span class='badge {_status_class(item.get('status'))}'>"
        f"{_escape(item.get('status'))}</span></td>"
        "</tr>"
        for item in platform_records
    )
    failure_rows = "".join(
        "<tr>"
        f"<th scope='row'>{_escape(item.get('kind'))}</th>"
        f"<td>{_escape(item.get('system') or item.get('cell_id'))}</td>"
        f"<td>{_escape(item.get('gate_id'))}</td>"
        f"<td>{_escape(item.get('status') or item.get('comparison_status'))}</td>"
        f"<td>{_escape(item)}</td>"
        "</tr>"
        for item in failures[:MAX_NON_PASS_ROWS]
    )

    style = """
    :root{color-scheme:light;font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#172033;background:#f5f7fb}
    *{box-sizing:border-box}body{margin:0}main{max-width:1180px;margin:0 auto;padding:28px 20px 56px}
    h1{font-size:2rem;margin:.25rem 0}.eyebrow{color:#475569;font-size:.82rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase}
    .banner{border:1px solid #f1c16b;background:#fff8e8;border-radius:12px;padding:14px 16px;margin:18px 0}.banner strong{display:block}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:16px 0}.card,section{background:#fff;border:1px solid #dbe2ea;border-radius:12px;padding:16px}
    .card .label{display:block;color:#64748b;font-size:.78rem;text-transform:uppercase;letter-spacing:.06em}.card .value{display:block;font-weight:750;margin-top:6px;overflow-wrap:anywhere}
    section{margin-top:16px;overflow:auto}h2{font-size:1.18rem;margin:0 0 12px}table{width:100%;border-collapse:collapse;font-size:.86rem}th,td{text-align:left;border-bottom:1px solid #e8edf3;padding:9px;vertical-align:top}thead th{color:#475569;background:#f8fafc;position:sticky;top:0}
    code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.82em}.badge{display:inline-block;border-radius:999px;padding:3px 8px;font-weight:750}.pass{color:#0f6848;background:#dff6ec}.fail{color:#9b1c1c;background:#fde7e7}.not-run,.warning{color:#8a4b08;background:#fff0d5}.unknown{color:#475569;background:#e9eef5}.muted{color:#64748b}.empty{padding:10px 0;color:#64748b}
    """
    csp = (
        "default-src 'none'; base-uri 'none'; connect-src 'none'; font-src 'none'; "
        "form-action 'none'; frame-src 'none'; img-src data:; media-src 'none'; "
        "object-src 'none'; script-src 'none'; style-src 'unsafe-inline'"
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{csp}"><title>PEARSim numerical evidence</title><style>{style}</style></head>
<body><main data-evidence-status="{_escape(status)}"><div class="eyebrow">PEARSim · canonical-buck numerical evidence</div>
<h1>Numerical evidence matrix</h1><div class="banner"><strong>EXPLORATORY VERIFICATION — NOT PHYSICAL VALIDATION</strong>
This page presents recorded acceptance gates for the exact canonical topology. It does not authorize hardware sign-off.</div>
<div class="grid">
<div class="card"><span class="label">Overall status</span><span class="value"><span class="badge {_status_class(status)}">{_escape(status)}</span></span></div>
<div class="card"><span class="label">Profile</span><span class="value">{_escape(profile.get('id'))}</span></div>
<div class="card"><span class="label">Source revision</span><span class="value"><code>{_escape(source_revision)}</code></span></div>
<div class="card"><span class="label">Platform</span><span class="value">{_escape(platform.get('system') or (str(len(platform_records)) + ' native systems' if platform_records else None))} · {_escape(platform.get('machine'))}</span></div>
<div class="card"><span class="label">Cells</span><span class="value">{_escape(summary.get('cell_count'))}</span></div>
<div class="card"><span class="label">Gates / comparisons</span><span class="value">{_escape(gate_count)}</span></div>
</div>
<section><h2>Recorded study results</h2>{('<table><thead><tr><th>Study</th><th>Status</th><th>Cells</th><th>Cell status counts</th></tr></thead><tbody>'+''.join(study_rows)+'</tbody></table>') if study_rows else '<div class="empty">No study records were supplied.</div>'}</section>
<section><h2>Platform aggregation</h2>{('<table><thead><tr><th>System</th><th>Machine</th><th>Python</th><th>Status</th></tr></thead><tbody>'+platform_rows+'</tbody></table>') if platform_rows else '<div class="empty">This artifact records one platform; cross-platform rows are not present.</div>'}</section>
<section><h2>Numerical configurations</h2>{('<table><thead><tr><th>ID</th><th>Studies</th><th>Solver</th><th>rtol</th><th>atol</th><th>max step / T</th><th>output step / T</th></tr></thead><tbody>'+''.join(configuration_rows)+'</tbody></table>') if configuration_rows else '<div class="empty">Configurations are recorded in the authoritative JSON only.</div>'}</section>
<section><h2>Recorded non-pass cells</h2>{('<table><thead><tr><th>Cell</th><th>Status</th><th>Expected regime</th><th>Non-pass gates</th><th>Error</th></tr></thead><tbody>'+''.join(non_pass_rows)+'</tbody></table>') if non_pass_rows else '<div class="empty">No non-pass cell is recorded.</div>'}
<p class="muted">Showing {min(len(non_pass), MAX_NON_PASS_ROWS)} of {len(non_pass)} recorded non-pass cells. Raw evidence remains authoritative.</p></section>
<section><h2>Recorded aggregate failures</h2>{('<table><thead><tr><th>Kind</th><th>System / cell</th><th>Gate</th><th>Status</th><th>Recorded detail</th></tr></thead><tbody>'+failure_rows+'</tbody></table>') if failure_rows else '<div class="empty">No aggregate failure is recorded.</div>'}</section>
<section><h2>Provenance binding</h2><table><tbody><tr><th scope="row">Source schema</th><td><code>{_escape(source.get('schema'))}</code></td></tr>
<tr><th scope="row">Source artifact SHA-256</th><td><code>{_escape(source.get('artifact_sha256'))}</code></td></tr>
<tr><th scope="row">Source semantic SHA-256</th><td><code>{_escape(source.get('semantic_sha256'))}</code></td></tr></tbody></table></section>
</main></body></html>"""


__all__ = [
    "NUMERICAL_REPORT_SCHEMA",
    "NumericalReportArtifacts",
    "NumericalReportError",
    "generate_numerical_report",
]
