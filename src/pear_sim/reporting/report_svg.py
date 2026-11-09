# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
from __future__ import annotations

from collections.abc import Sequence
import html
import math
import re
import xml.etree.ElementTree as ET

import numpy as np


_MAX_SCHEMATIC_BYTES = 2_000_000
_MAX_SCHEMATIC_NODES = 5_000
_ALLOWED_SVG_TAGS = {
    "circle",
    "desc",
    "ellipse",
    "g",
    "line",
    "path",
    "polygon",
    "polyline",
    "rect",
    "svg",
    "text",
    "title",
    "tspan",
}
_ALLOWED_SVG_ATTRIBUTES = {
    "aria-label",
    "aria-labelledby",
    "class",
    "cx",
    "cy",
    "data-component-id",
    "data-component-type",
    "data-n",
    "data-net-name",
    "data-p",
    "d",
    "fill",
    "fill-opacity",
    "fill-rule",
    "font-family",
    "font-size",
    "font-weight",
    "height",
    "id",
    "opacity",
    "points",
    "preserveAspectRatio",
    "r",
    "role",
    "rx",
    "ry",
    "stroke",
    "stroke-dasharray",
    "stroke-linecap",
    "stroke-linejoin",
    "stroke-opacity",
    "stroke-width",
    "text-anchor",
    "transform",
    "vector-effect",
    "viewBox",
    "width",
    "x",
    "x1",
    "x2",
    "y",
    "y1",
    "y2",
}
_ALLOWED_SVG_CLASSES = {
    "canvas",
    "component",
    "component-label",
    "component-symbol",
    "component-value",
    "diode-fill",
    "ground-junction",
    "ground-symbol",
    "junction",
    "layout-warning",
    "net",
    "net-label",
    "probe-letter",
    "subtitle",
    "symbol-stroke",
    "terminal",
    "title",
    "wire",
}
_UNSAFE_SVG_VALUE = re.compile(
    r"(?:javascript:|vbscript:|data:|https?:|file:|url\s*\(|@import)", re.IGNORECASE
)


class SvgError(ValueError):
    """Raised when an SVG cannot be rendered safely."""

    code = "report.svg_invalid"


def waveform_svg(
    time: Sequence[float],
    values: Sequence[float],
    signal: str,
    *,
    max_points: int = 800,
    title: str | None = None,
    highlight: tuple[float, float] | None = None,
    height: float = 260.0,
    time_label: str = "time",
) -> str:
    """Return a deterministic, static inline SVG for one waveform.

    ``title`` replaces the signal name in the heading; ``highlight`` shades a time
    window (the zoom shown under an overview) without changing any drawn sample.
    """

    if len(time) != len(values) or not time:
        raise SvgError("Waveform time and value columns must have the same non-zero length")
    if max_points < 2:
        raise SvgError("max_points must be at least 2")
    if any(not math.isfinite(value) for value in time) or any(
        not math.isfinite(value) for value in values
    ):
        raise SvgError("Waveform contains a non-finite value")

    indices = _sample_indices(len(time), max_points, (values,))
    sampled_time = [float(time[index]) for index in indices]
    sampled_values = [float(values[index]) for index in indices]

    width = 840.0
    left = 74.0
    right = 22.0
    top = 36.0
    bottom = 48.0
    plot_width = width - left - right
    plot_height = height - top - bottom

    time_min = min(time)
    time_max = max(time)
    value_min = min(values)
    value_max = max(values)
    time_span = time_max - time_min
    value_span = value_max - value_min
    if time_span == 0.0:
        time_span = max(abs(time_min), 1.0)
    time_multiplier, time_unit = _time_axis_scale(time_min, time_max)
    if value_span == 0.0:
        padding = max(abs(value_min) * 0.05, 1.0)
        value_min -= padding
        value_max += padding
        value_span = value_max - value_min
    else:
        padding = value_span * 0.04
        padded_min = value_min - padding
        padded_max = value_max + padding
        # Preserve zero as the boundary when padding alone would cross it for a
        # unilateral signal. This never uses a tolerance: a genuinely negative or
        # positive observation remains visible and receives symmetric padding.
        if value_min >= 0.0 and padded_min < 0.0:
            value_max += padding
            value_min = 0.0
        elif value_max <= 0.0 and padded_max > 0.0:
            value_min -= padding
            value_max = 0.0
        else:
            value_min = padded_min
            value_max = padded_max
        value_span = value_max - value_min

    def x_position(value: float) -> float:
        return left + (value - time_min) / time_span * plot_width

    def y_position(value: float) -> float:
        return top + (value_max - value) / value_span * plot_height

    points = " ".join(
        f"{x_position(t):.3f},{y_position(y):.3f}"
        for t, y in zip(sampled_time, sampled_values, strict=True)
    )
    escaped_signal = html.escape(signal, quote=True)
    escaped_title = html.escape(title if title is not None else signal, quote=True)
    highlight_svg = ""
    if highlight is not None:
        start = min(max(highlight[0], time_min), time_max)
        stop = min(max(highlight[1], time_min), time_max)
        x0 = x_position(start)
        # A window narrower than a pixel still gets a visible marker.
        span = max(x_position(stop) - x0, 3.0)
        highlight_svg = (
            f'<rect x="{x0 - (span - (x_position(stop) - x0)) / 2:.3f}" y="{top:.3f}" '
            f'width="{span:.3f}" height="{plot_height:.3f}" fill="#eb6834" '
            'fill-opacity="0.18" stroke="#eb6834" stroke-width="1" />'
        )
    grid_lines: list[str] = []
    tick_labels: list[str] = []
    for index in range(5):
        fraction = index / 4
        x = left + fraction * plot_width
        y = top + fraction * plot_height
        time_value = time_min + fraction * (time_max - time_min)
        amplitude = value_max - fraction * value_span
        grid_lines.append(
            f'<line x1="{x:.3f}" y1="{top:.3f}" x2="{x:.3f}" '
            f'y2="{top + plot_height:.3f}" stroke="#d9e2ec" stroke-width="1" />'
        )
        grid_lines.append(
            f'<line x1="{left:.3f}" y1="{y:.3f}" x2="{left + plot_width:.3f}" '
            f'y2="{y:.3f}" stroke="#d9e2ec" stroke-width="1" />'
        )
        tick_labels.append(
            f'<text x="{x:.3f}" y="{height - 23:.3f}" text-anchor="middle" '
            f'font-size="11" fill="#52606d">'
            f"{html.escape(_format_number(time_value * time_multiplier))}</text>"
        )
        tick_labels.append(
            f'<text x="{left - 9:.3f}" y="{y + 4:.3f}" text-anchor="end" '
            f'font-size="11" fill="#52606d">{html.escape(_format_number(amplitude))}</text>'
        )

    return "".join(
        [
            f'<svg viewBox="0 0 840 {height:g}" role="img" ',
            f'aria-label="Waveform {escaped_signal}" preserveAspectRatio="xMidYMid meet">',
            f"<title>{escaped_signal} waveform</title>",
            f'<rect x="0" y="0" width="840" height="{height:g}" rx="8" fill="#ffffff" />',
            *grid_lines,
            highlight_svg,
            f'<line x1="{left:.3f}" y1="{top + plot_height:.3f}" '
            f'x2="{left + plot_width:.3f}" y2="{top + plot_height:.3f}" '
            'stroke="#334e68" stroke-width="1.4" />',
            f'<line x1="{left:.3f}" y1="{top:.3f}" x2="{left:.3f}" '
            f'y2="{top + plot_height:.3f}" stroke="#334e68" stroke-width="1.4" />',
            *tick_labels,
            f'<polyline points="{points}" fill="none" stroke="#1565c0" '
            'stroke-width="1.2" stroke-linejoin="round" stroke-linecap="round" />',
            f'<text x="{left:.3f}" y="22" font-size="14" font-weight="700" '
            f'fill="#102a43">{escaped_title}</text>',
            f'<text x="818" y="{height - 5:g}" text-anchor="end" font-size="11" fill="#52606d">'
            f"{html.escape(time_label)} [{time_unit}]</text>",
            "</svg>",
        ]
    )


def phase_series_svg(
    time: Sequence[float],
    series: dict[str, Sequence[float]],
    title: str,
    *,
    y_label: str,
    max_points: int = 800,
) -> str:
    """Return aligned deterministic a/b/c traces in one static SVG."""

    if set(series) != {"a", "b", "c"}:
        raise SvgError("Phase-series plot requires exactly a, b, and c")
    if not time or any(len(values) != len(time) for values in series.values()):
        raise SvgError("Phase-series time and value columns must have equal non-zero length")
    if max_points < 2:
        raise SvgError("max_points must be at least 2")
    if any(not math.isfinite(value) for value in time) or any(
        not math.isfinite(value) for values in series.values() for value in values
    ):
        raise SvgError("Phase-series plot contains a non-finite value")

    indices = _sample_indices(
        len(time), max_points, tuple(series[phase] for phase in ("a", "b", "c"))
    )
    sampled_time = [float(time[index]) for index in indices]
    sampled = {
        phase: [float(series[phase][index]) for index in indices] for phase in ("a", "b", "c")
    }
    width = 840.0
    height = 280.0
    left, right, top, bottom = 74.0, 22.0, 42.0, 52.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    time_min, time_max = min(time), max(time)
    time_span = time_max - time_min
    if time_span == 0.0:
        time_span = max(abs(time_min), 1.0)
    all_values = [value for values in series.values() for value in values]
    value_min, value_max = min(all_values), max(all_values)
    if value_min == value_max:
        padding = max(abs(value_min) * 0.05, 1.0)
    else:
        padding = 0.05 * (value_max - value_min)
    value_min -= padding
    value_max += padding
    value_span = value_max - value_min
    time_multiplier, time_unit = _time_axis_scale(time_min, time_max)

    def x_position(value: float) -> float:
        return left + (value - time_min) / time_span * plot_width

    def y_position(value: float) -> float:
        return top + (value_max - value) / value_span * plot_height

    grid: list[str] = []
    labels: list[str] = []
    for index in range(5):
        fraction = index / 4
        x = left + fraction * plot_width
        y = top + fraction * plot_height
        grid.append(
            f'<line x1="{x:.3f}" y1="{top:.3f}" x2="{x:.3f}" '
            f'y2="{top + plot_height:.3f}" stroke="#d9e2ec" stroke-width="1" />'
        )
        grid.append(
            f'<line x1="{left:.3f}" y1="{y:.3f}" x2="{left + plot_width:.3f}" '
            f'y2="{y:.3f}" stroke="#d9e2ec" stroke-width="1" />'
        )
        time_value = time_min + fraction * (time_max - time_min)
        amplitude = value_max - fraction * value_span
        labels.append(
            f'<text x="{x:.3f}" y="{height - 24:.3f}" text-anchor="middle" '
            f'font-size="11" fill="#52606d">'
            f"{html.escape(_format_number(time_value * time_multiplier))}</text>"
        )
        labels.append(
            f'<text x="{left - 9:.3f}" y="{y + 4:.3f}" text-anchor="end" '
            f'font-size="11" fill="#52606d">{html.escape(_format_number(amplitude))}</text>'
        )

    styles = {
        "a": ("#1565c0", ""),
        "b": ("#d97706", "8 4"),
        "c": ("#7c3aed", "2 4"),
    }
    traces: list[str] = []
    legends: list[str] = []
    for phase_index, phase in enumerate(("a", "b", "c")):
        color, dash = styles[phase]
        points = " ".join(
            f"{x_position(t):.3f},{y_position(value):.3f}"
            for t, value in zip(sampled_time, sampled[phase], strict=True)
        )
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        traces.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            f'stroke-width="1.2" stroke-linejoin="round" '
            f'stroke-linecap="round"{dash_attribute} />'
        )
        legend_x = left + phase_index * 80
        legends.append(
            f'<line x1="{legend_x:.3f}" y1="26" x2="{legend_x + 24:.3f}" y2="26" '
            f'stroke="{color}" stroke-width="2"{dash_attribute} />'
            f'<text x="{legend_x + 30:.3f}" y="30" font-size="11" '
            f'fill="#334e68">phase {phase.upper()}</text>'
        )
    escaped_title = html.escape(title, quote=True)
    escaped_y_label = html.escape(y_label, quote=True)
    return "".join(
        [
            '<svg viewBox="0 0 840 280" role="img" ',
            f'aria-label="{escaped_title}" preserveAspectRatio="xMidYMid meet">',
            f"<title>{escaped_title}</title>",
            '<rect x="0" y="0" width="840" height="280" rx="8" fill="#ffffff" />',
            *grid,
            *labels,
            *traces,
            *legends,
            f'<text x="818" y="267" text-anchor="end" font-size="11" fill="#52606d">time [{time_unit}]</text>',
            f'<text x="12" y="150" transform="rotate(-90 12 150)" text-anchor="middle" font-size="11" fill="#52606d">{escaped_y_label}</text>',
            "</svg>",
        ]
    )


def harmonic_spectrum_svg(
    coefficients: Sequence[dict[str, object]],
    signal: str,
    *,
    max_bars: int = 64,
) -> str:
    """Render a bounded static RMS harmonic spectrum from producer values."""

    if max_bars < 1:
        raise SvgError("max_bars must be positive")
    rows: list[tuple[int, float]] = []
    for index, coefficient in enumerate(coefficients):
        order = coefficient.get("harmonic", coefficient.get("order"))
        rms = coefficient.get("rms_amplitude", coefficient.get("rms"))
        if (
            isinstance(order, bool)
            or not isinstance(order, int)
            or order < 0
            or isinstance(rms, bool)
            or not isinstance(rms, (int, float))
            or not math.isfinite(float(rms))
            or float(rms) < 0.0
        ):
            raise SvgError(f"Invalid harmonic coefficient at row {index}")
        rows.append((order, float(rms)))
    if not rows:
        raise SvgError("Harmonic spectrum requires at least one coefficient")
    rows = rows[:max_bars]
    maximum = max(value for _, value in rows)
    if maximum == 0.0:
        maximum = 1.0
    width, height = 840.0, 260.0
    left, right, top, bottom = 74.0, 22.0, 38.0, 48.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    slot = plot_width / len(rows)
    bars: list[str] = []
    for index, (order, value) in enumerate(rows):
        bar_height = value / maximum * plot_height
        x = left + index * slot + 0.15 * slot
        y = top + plot_height - bar_height
        bars.append(
            f'<rect x="{x:.3f}" y="{y:.3f}" width="{max(1.0, 0.7 * slot):.3f}" '
            f'height="{bar_height:.3f}" fill="#1565c0"><title>h{order}: '
            f"{html.escape(_format_number(value))} RMS</title></rect>"
        )
        if len(rows) <= 24 or index in {0, len(rows) - 1} or order % 10 == 0:
            bars.append(
                f'<text x="{x + 0.35 * slot:.3f}" y="{height - 24:.3f}" '
                f'text-anchor="middle" font-size="10" fill="#52606d">h{order}</text>'
            )
    escaped_signal = html.escape(signal, quote=True)
    return "".join(
        [
            '<svg viewBox="0 0 840 260" role="img" ',
            f'aria-label="Harmonic spectrum {escaped_signal}" preserveAspectRatio="xMidYMid meet">',
            f"<title>{escaped_signal} harmonic spectrum</title>",
            '<rect x="0" y="0" width="840" height="260" rx="8" fill="#ffffff" />',
            f'<line x1="{left:.3f}" y1="{top + plot_height:.3f}" x2="{left + plot_width:.3f}" y2="{top + plot_height:.3f}" stroke="#334e68" stroke-width="1.4" />',
            f'<line x1="{left:.3f}" y1="{top:.3f}" x2="{left:.3f}" y2="{top + plot_height:.3f}" stroke="#334e68" stroke-width="1.4" />',
            *bars,
            f'<text x="{left:.3f}" y="22" font-size="14" font-weight="700" fill="#102a43">{escaped_signal}</text>',
            '<text x="12" y="140" transform="rotate(-90 12 140)" text-anchor="middle" font-size="11" fill="#52606d">RMS magnitude</text>',
            "</svg>",
        ]
    )


def sanitize_schematic_svg(source: str) -> str:
    """Return a conservative inline-SVG subset with active/external content removed."""

    encoded = source.encode("utf-8")
    if len(encoded) > _MAX_SCHEMATIC_BYTES:
        raise SvgError("Schematic SVG exceeds the 2 MB alpha report limit")
    lowered = source.lower()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise SvgError("DOCTYPE and ENTITY declarations are not allowed in schematic SVG")
    try:
        root = ET.fromstring(source)
    except ET.ParseError as exc:
        raise SvgError(f"Invalid schematic SVG: {exc}") from exc
    if _local_name(root.tag) != "svg":
        raise SvgError("Schematic document root must be <svg>")
    if sum(1 for _ in root.iter()) > _MAX_SCHEMATIC_NODES:
        raise SvgError("Schematic SVG exceeds the alpha report node limit")

    sanitized = _sanitize_element(root)
    if sanitized is None:
        raise SvgError("Schematic SVG contains no supported root")
    sanitized.set("class", "schematic-svg")
    sanitized.set("role", "img")
    sanitized.set("aria-label", "Circuit schematic")
    return ET.tostring(sanitized, encoding="unicode", short_empty_elements=True)


def _sanitize_element(element: ET.Element) -> ET.Element | None:
    tag = _local_name(element.tag)
    if tag not in _ALLOWED_SVG_TAGS:
        return None
    clean = ET.Element(tag)
    for raw_name, raw_value in sorted(element.attrib.items()):
        name = _local_name(raw_name)
        if name not in _ALLOWED_SVG_ATTRIBUTES or name.lower().startswith("on"):
            continue
        if _UNSAFE_SVG_VALUE.search(raw_value):
            continue
        if name == "class":
            raw_value = " ".join(
                token for token in raw_value.split() if token in _ALLOWED_SVG_CLASSES
            )
            if not raw_value:
                continue
        clean.set(name, raw_value)
    clean.text = element.text
    for child in element:
        clean_child = _sanitize_element(child)
        if clean_child is not None:
            clean.append(clean_child)
            clean_child.tail = child.tail
    return clean


def _local_name(name: str) -> str:
    return name.rsplit("}", 1)[-1]


def _sample_indices(
    length: int, maximum: int, columns: Sequence[Sequence[float]] = ()
) -> list[int]:
    """Indices to draw, at most ``maximum``: the first, the last, and in every equal
    bucket of the run the samples holding each column's minimum and maximum.

    A plain stride is not used: on switched waveforms it lands at a fixed phase of
    the carrier (20 kHz drawn at 800 points from a 1 us grid picks one sample per
    period) and draws a clean staircase that is not in the data. Keeping each
    bucket's extremes means every pulse the data holds stays visible.
    """

    if length <= maximum:
        return list(range(length))
    if not columns:
        return sorted({round(index * (length - 1) / (maximum - 1)) for index in range(maximum)})
    buckets = max(1, (maximum - 2) // (2 * len(columns)))
    edges = [bucket * length // buckets for bucket in range(buckets + 1)]
    chosen = {0, length - 1}
    for column in columns:
        data = np.asarray(column, dtype=float)
        for start, stop in zip(edges, edges[1:]):
            if stop > start:
                window = data[start:stop]
                chosen.add(start + int(np.argmin(window)))
                chosen.add(start + int(np.argmax(window)))
    return sorted(chosen)


def _time_axis_scale(time_min: float, time_max: float) -> tuple[float, str]:
    reference = max(abs(time_min), abs(time_max))
    if 0.0 < reference < 1e-3:
        return 1e6, "µs"
    if 0.0 < reference < 1.0:
        return 1e3, "ms"
    return 1.0, "s"


def _format_number(value: float) -> str:
    if value == 0.0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 10_000 or magnitude < 0.001:
        return f"{value:.3e}"
    return f"{value:.5g}"


_CONDUCTION = "#2a78d6"
_SWITCHING = "#eb6834"
_ENVELOPES = ("#2a78d6", "#eb6834", "#1baf7a")


def loss_bars_svg(rows: Sequence[tuple[str, float, float]]) -> str:
    """Horizontal stacked bars: mean conduction and switching loss per device.

    ``rows`` are (device, conduction_w, switching_w) as published by the
    losses producer; the report only draws them.
    """

    if not rows:
        raise SvgError("Loss chart needs at least one device")
    for _, conduction, switching in rows:
        if not (math.isfinite(conduction) and math.isfinite(switching)):
            raise SvgError("Loss chart contains a non-finite value")
    row_h, gap, left, right, top = 16.0, 4.0, 150.0, 70.0, 44.0
    width = 840.0
    height = top + len(rows) * (row_h + gap) + 36.0
    plot_w = width - left - right
    step = _nice_step((max((c + s for _, c, s in rows), default=0.0) or 1.0) / 4.0)
    peak = step * math.ceil((max((c + s for _, c, s in rows), default=0.0) or 1.0) / step)
    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        'aria-label="Mean loss per device, conduction and switching" '
        'preserveAspectRatio="xMidYMid meet">',
        "<title>Mean loss per device</title>",
        f'<rect x="{left}" y="14" width="10" height="10" rx="2" fill="{_CONDUCTION}" />'
        f'<text x="{left + 16}" y="23" font-size="12" fill="#334e68">conduction</text>'
        f'<rect x="{left + 110}" y="14" width="10" height="10" rx="2" fill="{_SWITCHING}" />'
        f'<text x="{left + 126}" y="23" font-size="12" fill="#334e68">switching (incl. recovery)</text>',
    ]
    for tick in range(int(round(peak / step)) + 1):
        value = step * tick
        x = left + plot_w * value / peak
        parts.append(
            f'<line x1="{x:.2f}" y1="{top - 6}" x2="{x:.2f}" y2="{height - 30:.2f}" '
            'stroke="#e4e9f0" stroke-width="1" />'
            f'<text x="{x:.2f}" y="{height - 14:.2f}" font-size="11" fill="#52606d" '
            f'text-anchor="middle">{_format_number(value)} W</text>'
        )
    for index, (device, conduction, switching) in enumerate(rows):
        y = top + index * (row_h + gap)
        w_c = plot_w * conduction / peak
        w_s = plot_w * switching / peak
        total = conduction + switching
        parts.append(
            f'<g><title>{html.escape(device)}: conduction {_format_number(conduction)} W, '
            f'switching {_format_number(switching)} W, total {_format_number(total)} W</title>'
            f'<text x="{left - 8}" y="{y + 12:.2f}" font-size="11" fill="#172033" '
            f'text-anchor="end">{html.escape(device)}</text>'
            f'<rect x="{left:.2f}" y="{y:.2f}" width="{max(w_c, 0.0):.2f}" height="{row_h}" '
            f'fill="{_CONDUCTION}" />'
            f'<rect x="{left + w_c + (2.0 if w_s > 0 else 0.0):.2f}" y="{y:.2f}" '
            f'width="{max(w_s - (2.0 if w_s > 2 else 0.0), 0.0):.2f}" height="{row_h}" '
            f'rx="2" fill="{_SWITCHING}" />'
            f'<text x="{left + w_c + w_s + 6:.2f}" y="{y + 12:.2f}" font-size="11" '
            f'fill="#52606d">{_format_number(total)} W</text></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def tj_envelope_svg(series: Sequence[tuple[str, dict]]) -> str:
    """Junction temperature over time as min/max envelopes, one per device.

    Each envelope is the producer's per-bucket minimum and maximum; the band
    shows the ripple inside a bucket and the upper edge is drawn as the line.
    """

    if not series or len(series) > len(_ENVELOPES):
        raise SvgError(f"Envelope chart takes 1..{len(_ENVELOPES)} devices")
    width, height = 840.0, 280.0
    left, right, top, bottom = 74.0, 22.0, 48.0, 48.0
    plot_w, plot_h = width - left - right, height - top - bottom
    times = [t for _, env in series for t in (*env["time_s"], *env["time_end_s"])]
    temps = [v for _, env in series for v in (*env["min_c"], *env["max_c"])]
    if not times or any(not math.isfinite(v) for v in (*times, *temps)):
        raise SvgError("Envelope chart contains a non-finite value")
    t0, t1 = min(times), max(times)
    v0, v1 = min(temps), max(temps)
    pad = max((v1 - v0) * 0.08, 0.5)
    v0, v1 = v0 - pad, v1 + pad
    scale, unit = _time_axis_scale(t0, t1)

    def x_of(t: float) -> float:
        return left + plot_w * ((t - t0) / (t1 - t0) if t1 > t0 else 0.5)

    def y_of(v: float) -> float:
        return top + plot_h * (1.0 - (v - v0) / (v1 - v0))

    parts = [
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        'aria-label="Junction temperature envelopes of the hottest devices" '
        'preserveAspectRatio="xMidYMid meet">',
        "<title>Junction temperature, hottest devices</title>",
    ]
    step = _nice_step((v1 - v0) / 4.0)
    first = math.ceil(v0 / step) * step
    ticks = [first + k * step for k in range(int((v1 - first) / step) + 1)]
    for v in ticks:
        y = y_of(v)
        parts.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            'stroke="#e4e9f0" stroke-width="1" />'
            f'<text x="{left - 8}" y="{y + 4:.2f}" font-size="11" fill="#52606d" '
            f'text-anchor="end">{_format_number(v)} °C</text>'
        )
    for tick in range(5):
        t = t0 + (t1 - t0) * tick / 4.0
        parts.append(
            f'<text x="{x_of(t):.2f}" y="{height - 26}" font-size="11" fill="#52606d" '
            f'text-anchor="middle">{_format_number(t * scale)}</text>'
        )
    parts.append(
        f'<text x="{left + plot_w}" y="{height - 8}" font-size="11" fill="#52606d" '
        f'text-anchor="end">time [{unit}]</text>'
    )
    for index, (device, env) in enumerate(series):
        color = _ENVELOPES[index]
        upper = [(x_of(t), y_of(v)) for t, v in zip(env["time_s"], env["max_c"])]
        lower = [(x_of(t), y_of(v)) for t, v in zip(env["time_s"], env["min_c"])]
        band = " ".join(f"{x:.2f},{y:.2f}" for x, y in (*upper, *reversed(lower)))
        line = " ".join(f"{x:.2f},{y:.2f}" for x, y in upper)
        peak = max(env["max_c"])
        parts.append(
            f'<g><title>{html.escape(device)}: Tj max {peak:.2f} °C</title>'
            f'<polygon points="{band}" fill="{color}" fill-opacity="0.18" stroke="none" />'
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="1.5" />'
            f'<rect x="{left + index * 220:.2f}" y="14" width="10" height="10" rx="2" fill="{color}" />'
            f'<text x="{left + index * 220 + 16:.2f}" y="23" font-size="12" fill="#172033">'
            f'{html.escape(device)} · max {peak:.1f} °C</text></g>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _nice_step(raw: float) -> float:
    """1, 2 or 5 times a power of ten, at least ``raw``."""

    if raw <= 0 or not math.isfinite(raw):
        return 1.0
    power = 10.0 ** math.floor(math.log10(raw))
    for factor in (1.0, 2.0, 5.0, 10.0):
        if factor * power >= raw:
            return factor * power
    return 10.0 * power
