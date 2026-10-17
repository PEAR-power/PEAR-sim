"""Offline HTML projection of a perturbation campaign.

Like every other reporter here it is a **projection**: it reads the summary the
producer already wrote and never recomputes a number. Waveform overlays are
read from the retained run bundles and downsampled for drawing only; the
metrics table quotes the producer's values verbatim.

The disclaimer is not decoration. This page is the thing a reader will forward,
so the boundary of what was and was not computed has to travel on the page
itself, above the numbers rather than in a footnote.
"""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


__all__ = ["render_campaign_report"]

_MAX_OVERLAY_RUNS = 12
_OVERLAY_POINTS = 400
_PLOT_W = 900
_PLOT_H = 260

_DISCLAIMER = """
This page reports the response of a <em>declared</em> model to <em>declared</em>
inputs. It does not compute — and must not be cited as — a cross section, a
threshold, an event rate, a dose-dependent parameter shift, or any device's
response to any environment. Every drift figure here is an input that came from
somewhere else; the provenance table below names where. Figures marked
<code>assumption</code> are placeholders and describe no real component.
The question answered is the inverse one: <strong>how far can a parameter drift,
or how wide can an injected transient be, before this architecture stops meeting
its declared criteria.</strong> The answer is a requirement for whoever selects
parts, not a prediction.
"""

_CSS = """
:root { color-scheme: light dark; }
body { font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, Segoe UI, sans-serif;
       margin: 0; padding: 2rem; max-width: 1200px; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 .5rem; border-bottom: 1px solid #8884;
     padding-bottom: .25rem; }
.sub { opacity: .7; margin: 0 0 1rem; }
.disclaimer { border-left: 4px solid #b8860b; background: #b8860b18;
              padding: .75rem 1rem; margin: 1rem 0 1.5rem; }
.claim { border-left: 4px solid #6a7; background: #6a71a; padding: .6rem 1rem;
         margin: .75rem 0; font-size: .93em; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { border: 1px solid #8884; padding: .3rem .5rem; text-align: left; }
th { background: #8881; position: sticky; top: 0; }
td.num { text-align: right; }
.scroll { overflow-x: auto; max-height: 32rem; overflow-y: auto; }
.pass { color: #2a7; font-weight: 600; }
.fail { color: #d33; font-weight: 700; }
.notrun { opacity: .6; font-style: italic; }
tr.violation { background: #d3333a; }
code { background: #8881; padding: 0 .25rem; border-radius: 3px; }
figure { margin: 1rem 0; }
figcaption { opacity: .75; font-size: .9em; margin-top: .35rem; }
svg { max-width: 100%; height: auto; display: block; }
.legend span { margin-right: 1rem; font-size: .85em; }
.swatch { display: inline-block; width: .8rem; height: .8rem; vertical-align: -1px;
          margin-right: .25rem; border-radius: 2px; }
"""

_PALETTE = (
    "#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377",
    "#bbbbbb", "#ee8866", "#77aadd", "#99ddff", "#44bb99", "#ddcc77",
)


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _num(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _esc(value)
    magnitude = abs(float(value))
    if magnitude and (magnitude < 1e-3 or magnitude >= 1e6):
        return f"{value:.4g}"
    return f"{value:.6g}"


def _status_cell(status: str) -> str:
    css = {"pass": "pass", "fail": "fail"}.get(status, "notrun")
    return f'<td class="{css}">{_esc(status)}</td>'


# ---------------------------------------------------------------------------
# waveform overlay
# ---------------------------------------------------------------------------


def _load_series(run_dir: Path, signal: str) -> tuple[list[float], list[float]] | None:
    path = run_dir / "waveforms.csv"
    if not path.exists():
        return None
    times: list[float] = []
    values: list[float] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        if signal not in (reader.fieldnames or []):
            return None
        for row in reader:
            times.append(float(row["time"]))
            values.append(float(row[signal]))
    if not times:
        return None
    stride = max(1, len(times) // _OVERLAY_POINTS)
    return times[::stride], values[::stride]


def _overlay(summary: Mapping[str, Any], base: Path, signal: str) -> str:
    """Baseline plus perturbed runs on one axis; drawing-only downsample."""

    runs = [row for row in summary["runs"] if row.get("run")]
    if not runs:
        return "<p class='notrun'>No retained run to draw.</p>"
    dropped = max(0, len(runs) - _MAX_OVERLAY_RUNS)
    selected = runs[:_MAX_OVERLAY_RUNS]

    series: list[tuple[str, list[float], list[float]]] = []
    for row in selected:
        loaded = _load_series(base / row["run"], signal)
        if loaded is not None:
            series.append((row["label"], loaded[0], loaded[1]))
    if not series:
        return f"<p class='notrun'>Signal <code>{_esc(signal)}</code> is not in these bundles.</p>"

    x_min = min(min(times) for _, times, _ in series)
    x_max = max(max(times) for _, times, _ in series)
    y_min = min(min(values) for _, _, values in series)
    y_max = max(max(values) for _, _, values in series)
    if x_max <= x_min:
        x_max = x_min + 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    pad = 0.06 * (y_max - y_min)
    y_min, y_max = y_min - pad, y_max + pad

    def sx(value: float) -> float:
        return 60 + (value - x_min) / (x_max - x_min) * (_PLOT_W - 80)

    def sy(value: float) -> float:
        return _PLOT_H - 30 - (value - y_min) / (y_max - y_min) * (_PLOT_H - 50)

    paths = []
    legend = []
    for index, (label, times, values) in enumerate(series):
        colour = _PALETTE[index % len(_PALETTE)]
        points = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(times, values))
        paths.append(
            f'<polyline fill="none" stroke="{colour}" stroke-width="1.2" '
            f'points="{points}"/>'
        )
        legend.append(
            f'<span><i class="swatch" style="background:{colour}"></i>'
            f"{_esc(label)}</span>"
        )
    axes = (
        f'<line x1="60" y1="{_PLOT_H - 30}" x2="{_PLOT_W - 20}" y2="{_PLOT_H - 30}" '
        f'stroke="#888"/>'
        f'<line x1="60" y1="20" x2="60" y2="{_PLOT_H - 30}" stroke="#888"/>'
        f'<text x="60" y="{_PLOT_H - 12}" font-size="11" fill="#888">{_num(x_min)} s</text>'
        f'<text x="{_PLOT_W - 20}" y="{_PLOT_H - 12}" font-size="11" fill="#888" '
        f'text-anchor="end">{_num(x_max)} s</text>'
        f'<text x="8" y="24" font-size="11" fill="#888">{_num(y_max)}</text>'
        f'<text x="8" y="{_PLOT_H - 34}" font-size="11" fill="#888">{_num(y_min)}</text>'
    )
    note = (
        f"<figcaption>{_esc(signal)} across {len(series)} run(s)"
        + (
            f"; <strong>{dropped} further run(s) are not drawn</strong> "
            f"(overlay caps at {_MAX_OVERLAY_RUNS}) — the metrics table below "
            "covers every run."
            if dropped
            else "."
        )
        + f" Downsampled to ~{_OVERLAY_POINTS} points for drawing only.</figcaption>"
    )
    return (
        f'<figure><svg viewBox="0 0 {_PLOT_W} {_PLOT_H}" role="img">'
        f"{axes}{''.join(paths)}</svg>"
        f'<div class="legend">{"".join(legend)}</div>{note}</figure>'
    )


def _boundary_plot(boundary: Mapping[str, Any]) -> str:
    probes = boundary.get("probes") or []
    if not probes:
        return ""
    values = [float(probe["value_si"]) for probe in probes]
    low, high = min(values), max(values)
    if high <= low:
        high = low + 1.0
    width, height = _PLOT_W, 130

    def sx(value: float) -> float:
        return 60 + (value - low) / (high - low) * (width - 80)

    marks = []
    for index, probe in enumerate(probes):
        x = sx(float(probe["value_si"]))
        passed = probe.get("status") == "pass"
        colour = "#2a7" if passed else "#d33"
        y = 40 + (index % 3) * 14
        marks.append(
            f'<circle cx="{x:.1f}" cy="{y}" r="4.5" fill="{colour}"/>'
            f'<text x="{x:.1f}" y="{y - 8}" font-size="9" fill="#888" '
            f'text-anchor="middle">{_esc(probe.get("tag", ""))}</text>'
        )
    edge = boundary.get("max_passing_value_si")
    if isinstance(edge, (int, float)):
        marks.append(
            f'<line x1="{sx(float(edge)):.1f}" y1="20" x2="{sx(float(edge)):.1f}" '
            f'y2="{height - 30}" stroke="#b8860b" stroke-width="2" '
            f'stroke-dasharray="4 3"/>'
        )
    axis = (
        f'<line x1="60" y1="{height - 28}" x2="{width - 20}" y2="{height - 28}" stroke="#888"/>'
        f'<text x="60" y="{height - 12}" font-size="11" fill="#888">{_num(low)}</text>'
        f'<text x="{width - 20}" y="{height - 12}" font-size="11" fill="#888" '
        f'text-anchor="end">{_num(high)}</text>'
    )
    return (
        f'<figure><svg viewBox="0 0 {width} {height}" role="img">{axis}'
        f'{"".join(marks)}</svg><figcaption>Probes on '
        f'<code>{_esc(boundary["parameter"])}</code> (SI). Green passed every '
        f"criterion, red failed at least one; the dashed line is the largest "
        f"probed value that still passed.</figcaption></figure>"
    )


# ---------------------------------------------------------------------------
# tables
# ---------------------------------------------------------------------------


def _metric_columns(runs: Sequence[Mapping[str, Any]]) -> list[str]:
    preferred = [
        "vout_peak_deviation_v",
        "recovery_time_s",
        "vout_settled_error_v",
        "i(l1)::max",
        "v(q1)::max",
        "fsw_measured_hz",
    ]
    present = {key for run in runs for key in (run.get("metrics") or {})}
    columns = [name for name in preferred if name in present]
    columns += sorted(
        key
        for key in present
        if key not in columns and key not in ("protection", "recovery_status")
    )[:6]
    return columns


def _runs_table(summary: Mapping[str, Any]) -> str:
    runs = summary["runs"]
    criteria = [criterion["id"] for criterion in summary["criteria"]]
    columns = _metric_columns(runs)
    header = (
        "<tr><th>point</th><th>label</th><th>exec</th><th>accept</th>"
        + "".join(f"<th>{_esc(name)}</th>" for name in columns)
        + "".join(f"<th>{_esc(name)}</th>" for name in criteria)
        + "</tr>"
    )
    body = []
    for run in runs:
        violated = run.get("acceptance_status") != "pass"
        cells = [
            f'<td><code>{_esc(run["point_id"])}</code></td>',
            f'<td>{_esc(run["label"])}</td>',
            _status_cell(run.get("execution_status", "not_run")),
            _status_cell(run.get("acceptance_status", "not_run")),
        ]
        metrics = run.get("metrics") or {}
        cells += [f'<td class="num">{_num(metrics.get(name))}</td>' for name in columns]
        verdicts = run.get("criteria") or {}
        for name in criteria:
            verdict = verdicts.get(name) or {}
            status = verdict.get("status", "not_run")
            css = {"pass": "pass", "fail": "fail"}.get(status, "notrun")
            margin = verdict.get("margin")
            label = status if margin is None else f"{status} ({_num(margin)})"
            cells.append(f'<td class="{css}">{_esc(label)}</td>')
        body.append(
            f'<tr class="{"violation" if violated else ""}">{"".join(cells)}</tr>'
        )
    return f'<div class="scroll"><table>{header}{"".join(body)}</table></div>'


def _provenance_table(summary: Mapping[str, Any]) -> str:
    rows = []
    for axis in summary["axes"]:
        rows.append(
            f'<tr><td>axis <code>{_esc(axis["id"])}</code></td>'
            f'<td>{_esc(axis.get("parameter") or axis.get("event") or "—")}</td>'
            f'<td>{_esc(axis["kind"])}</td>'
            f'<td><code>{_esc(axis.get("source", "—"))}</code></td>'
            f'<td>{_esc(axis.get("note", "—"))}</td></tr>'
        )
    for criterion in summary["criteria"]:
        rows.append(
            f'<tr><td>criterion <code>{_esc(criterion["id"])}</code></td>'
            f'<td>{_esc(criterion["signal"])}.{_esc(criterion["field"])}</td>'
            f'<td>{_esc(criterion["op"])} {_num(criterion["limit"])} '
            f'{_esc(criterion.get("unit", ""))}</td>'
            f'<td><code>{_esc(criterion.get("source", "—"))}</code></td>'
            f'<td>{_esc(criterion.get("note", "—"))}</td></tr>'
        )
    boundary = summary.get("boundary_search")
    if boundary:
        rows.append(
            f'<tr><td>boundary bracket</td><td>{_esc(boundary["parameter"])}</td>'
            f'<td>{_num(boundary["bracket_si"][0])} … {_num(boundary["bracket_si"][1])}</td>'
            f'<td><code>{_esc(boundary.get("source", "—"))}</code></td>'
            f'<td>{_esc(boundary.get("note", "—"))}</td></tr>'
        )
    return (
        "<table><tr><th>what</th><th>target</th><th>how</th><th>source</th>"
        f'<th>note</th></tr>{"".join(rows)}</table>'
    )


def _boundary_section(summary: Mapping[str, Any]) -> str:
    boundary = summary.get("boundary_search")
    if not boundary:
        return ""
    percent = boundary.get("max_passing_drift_percent")
    headline = (
        "no value in the bracket met every criterion"
        if boundary.get("max_passing_value_si") is None
        else (
            f'<strong>{_esc(boundary["parameter"])} tolerates up to '
            f'{_num(boundary["max_passing_value_si"])} '
            f'({_num(percent)} % from nominal)</strong> before a declared '
            "criterion fails"
        )
    )
    return (
        "<h2>Tolerable-drift boundary</h2>"
        f"<p>{headline}. Verdict: <code>{_esc(boundary['verdict'])}</code>, "
        f"{_esc(boundary['iterations'])} bisection steps.</p>"
        f'<div class="claim"><strong>Assumption:</strong> '
        f"{_esc(boundary['assumption'])}</div>"
        f"{_boundary_plot(boundary)}"
    )


def render_campaign_report(summary: Mapping[str, Any], base_dir: Path | None = None) -> str:
    """Render the campaign summary as a self-contained offline HTML page."""

    base = Path(base_dir) if base_dir is not None else Path(".")
    status = summary["overall_status"]
    status_css = "pass" if status == "pass" else "fail"
    axes_line = ", ".join(
        f'{axis["id"]} ({axis["kind"]}, {axis["point_count"]} pts)'
        for axis in summary["axes"]
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PEARSim perturbation campaign — {_esc(summary['label'])}</title>
<style>{_CSS}</style></head><body>
<h1>Perturbation campaign — {_esc(summary['label'])}</h1>
<p class="sub">Model <code>{_esc(summary['model'])}</code> ·
{_esc(summary['run_count'])} runs · axes: {_esc(axes_line)} ·
overall <span class="{status_css}">{_esc(status)}</span></p>

<div class="disclaimer"><strong>What this is, and is not.</strong>{_DISCLAIMER}</div>
<div class="claim"><strong>Claim boundary.</strong> {_esc(summary['claim'])}</div>
<div class="claim"><strong>Protection.</strong> {_esc(summary['protection_modelling'])}</div>

<h2>Runs and criteria</h2>
<p>Rows highlighted in red violate at least one declared criterion. Each
criterion cell shows its margin in SI units; a negative margin is the amount by
which the limit was exceeded.</p>
{_runs_table(summary)}

<h2>Waveform overlay</h2>
{_overlay(summary, base, "v(out)")}
{_overlay(summary, base, "i(l1)")}

{_boundary_section(summary)}

<h2>Provenance of every declared number</h2>
<p>A value marked <code>assumption</code> was chosen to exercise the machinery
and describes no real component. Nothing on this page becomes evidence about a
device until its row here reads <code>datasheet</code>, <code>test_report</code>
or <code>standard</code>.</p>
{_provenance_table(summary)}

<h2>Machine-readable</h2>
<p>Every number above is quoted verbatim from <code>campaign-summary.json</code>;
this page recomputes nothing. Per-run bundles, including the exact materialized
model source that was simulated, are retained under <code>points/</code>.</p>
</body></html>
"""


def render_campaign_report_file(summary_path: str | Path) -> str:
    payload = json.loads(Path(summary_path).read_text())
    return render_campaign_report(payload, Path(summary_path).parent)
