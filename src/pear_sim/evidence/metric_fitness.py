# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Is a sampled extremum fit to decide a declared bound?

``max``, ``min``, ``peak_to_peak`` and ``ripple`` in ``metrics.json`` are
statistics of the requested output grid (``extrema_method:
requested_output_grid_samples``), not of the continuous waveform. A grid can
only miss a peak, never invent one, so the error has a known sign:

* a sampled maximum (or peak-to-peak, or ripple) is a LOWER bound on the true one;
* a sampled minimum is an UPPER bound on the true one.

A bound is therefore decided by a sampled extremum in exactly two cases: the
grid resolves the switching period (``MIN_SAMPLES_PER_SWITCHING_PERIOD``), or
the known-sign error can only push the value further in the direction the
verdict already points (a sampled maximum above an upper limit is a violation
on any grid). Everything else is ``not_comparable``: a coarse grid that reads
3.4 A where the waveform peaks at 4.6 A must not pass ``max <= 4.1 A``.

Samples per period alone do not see a pulse narrower than the grid step: ten
samples per period can all fall outside a 4 % duty conduction interval and
read 0 A. The producer therefore also records each signal's extremes over the
END POINTS of the solver's integration intervals (``extrema_grid.event_edge``,
both sides of every event). Those points are on the waveform, so they tighten
the sampled bound from the same side, and it is the tightened value that is
compared with the limit. What neither the samples nor the event edges see is
an extremum strictly inside an interval between events (a resonant peak).

This is the one place that rule lives. ``evaluate`` (expectations), the
perturbation campaign and the batch ranking all call it, so the same run and
the same requirement cannot get different verdicts on different paths.

The floor rules out aliasing against the declared carrier. It is not an
accuracy statement: a resolved sampled extremum is still a grid statistic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any


#: Fewest output samples per shortest declared switching period for a sampled
#: extremum to count as resolved.
MIN_SAMPLES_PER_SWITCHING_PERIOD = 10
# 10 us on a 100 kHz carrier is 10 samples, whatever 1e-5 / 1e-5 rounds to.
_RATIO_SLACK = 1.0e-9

EXTREMA_GRID_FIELD = "extrema_grid"

#: Sampled value <= true value.
_UNDERESTIMATED = frozenset(
    {
        "max",
        "max_last_half",
        "peak_to_peak",
        "peak_to_peak_last_half",
        "ripple",
        "ripple_last_half",
        "sample_max",
        "sample_max_last_half",
        "sample_peak_to_peak",
        "sample_peak_to_peak_last_half",
    }
)
#: Sampled value >= true value.
_OVERESTIMATED = frozenset(
    {"min", "min_last_half", "sample_min", "sample_min_last_half"}
)
SAMPLED_EXTREMUM_METRICS = _UNDERESTIMATED | _OVERESTIMATED


def sampled_extremum_bias(metric: str) -> str | None:
    """``"under"``, ``"over"`` or ``None`` when ``metric`` is no sampled extremum."""

    if metric in _UNDERESTIMATED:
        return "under"
    if metric in _OVERESTIMATED:
        return "over"
    return None


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def grid_unresolved_reason(grid: Any) -> str | None:
    """Why this ``extrema_grid`` record does not resolve the waveform, or ``None``.

    Fail closed: a record that is absent or malformed is unresolved. A model
    with no switching control has no carrier to alias against and is resolved.
    """

    if not isinstance(grid, Mapping):
        return (
            "the run records no output-grid metadata (extrema_grid); it was "
            "produced before sampled extrema carried it and must be re-simulated"
        )
    controls = grid.get("switching_control_count")
    if isinstance(controls, bool) or not isinstance(controls, int) or controls < 0:
        return "extrema_grid.switching_control_count is not a non-negative integer"
    if controls == 0:
        return None
    samples = _number(grid.get("samples_per_shortest_switching_period"))
    if samples is None or samples <= 0.0:
        return "extrema_grid does not state the samples per switching period"
    if samples < MIN_SAMPLES_PER_SWITCHING_PERIOD * (1.0 - _RATIO_SLACK):
        return (
            f"the output grid has {samples:.6g} sample(s) per shortest switching "
            f"period; at least {MIN_SAMPLES_PER_SWITCHING_PERIOD} are required"
        )
    return None


@dataclass(frozen=True)
class SampledBound:
    """How a sampled extremum stands against ``lower <= metric <= upper``.

    ``value`` is the number to compare with the bounds: the sampled value
    tightened by the signal's event-edge extremes (still a one-sided bound on
    the true extremum, never looser than the sampled one). ``unfit_reason`` is
    ``None`` exactly when comparing ``value`` with the bounds is a sound verdict.
    """

    value: float
    sampled: float
    unfit_reason: str | None

    @property
    def tightened(self) -> bool:
        return self.value != self.sampled


def fitness_context(record: Any) -> dict[str, Any] | None:
    """What the rule needs from one signal record of ``metrics.json``.

    The record's ``extrema_grid`` plus, under ``sampled``, the grid's own
    extremes, so that a peak-to-peak can be rebuilt from the joint extremes of
    samples and event edges. ``None`` (unresolved) without ``extrema_grid``.
    """

    grid = record.get(EXTREMA_GRID_FIELD) if isinstance(record, Mapping) else None
    if not isinstance(grid, Mapping):
        return None
    sampled = {
        key: record.get(key, record.get(f"sample_{key}"))
        for key in ("min", "max", "min_last_half", "max_last_half")
    }
    return {**grid, "sampled": sampled}


def _event_edge_bound(metric: str, grid: Any, value: float, bias: str) -> float:
    """``value`` tightened by the signal's values at the solver's interval end points.

    The output grid can step over a whole conduction interval (a 2 us switch
    pulse between two samples 5 us apart reads 0 A). The end points of every
    integration interval are part of the waveform, so their extremes bound
    the true extremum from the same side the samples do.

    A peak-to-peak (or ripple) is the span of ALL observations: the highest
    of samples and edges minus the lowest of samples and edges. The larger of
    the two separate spans is not enough when the maximum was seen at an edge
    and the minimum in a sample.
    """

    edge = grid.get("event_edge") if isinstance(grid, Mapping) else None
    if not isinstance(edge, Mapping):
        return value
    suffix = "_last_half" if metric.endswith("_last_half") else ""
    low, high = _number(edge.get(f"min{suffix}")), _number(edge.get(f"max{suffix}"))
    if "peak_to_peak" in metric or "ripple" in metric:
        if low is None or high is None:
            return value
        sampled = grid.get("sampled")
        sampled = sampled if isinstance(sampled, Mapping) else {}
        sample_low = _number(sampled.get(f"min{suffix}"))
        sample_high = _number(sampled.get(f"max{suffix}"))
        if sample_low is not None:
            low = min(low, sample_low)
        if sample_high is not None:
            high = max(high, sample_high)
        return max(value, high - low)
    if bias == "under":
        return value if high is None else max(value, high)
    return value if low is None else min(value, low)


def judge_sampled_bound(
    metric: str,
    grid: Any,
    value: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> SampledBound:
    """Judge ``lower <= metric <= upper`` from the sampled ``value``.

    ``grid`` is ``fitness_context(record)`` for the signal (a bare
    ``extrema_grid`` also works, without the joint peak-to-peak). With neither
    bound the question is whether the number itself is usable (a ranking
    objective), which needs a resolved grid.
    """

    bias = sampled_extremum_bias(metric)
    if bias is None:
        return SampledBound(value, value, None)
    bound = _event_edge_bound(metric, grid, value, bias)
    unresolved = grid_unresolved_reason(grid)
    if unresolved is None:
        return SampledBound(bound, value, None)
    if bias == "under":
        # true >= bound
        if upper is not None and bound > upper:
            return SampledBound(bound, value, None)  # above the limit; so is the truth
        if upper is None and lower is not None and bound >= lower:
            return SampledBound(bound, value, None)  # above the floor; so is the truth
        direction = "a lower bound on"
    else:
        # true <= bound
        if lower is not None and bound < lower:
            return SampledBound(bound, value, None)
        if lower is None and upper is not None and bound <= upper:
            return SampledBound(bound, value, None)
        direction = "an upper bound on"
    return SampledBound(
        bound,
        value,
        f"sampled {metric} is only {direction} the true value and {unresolved}, "
        "so it cannot decide this requirement",
    )


def sampled_bound_unfit_reason(
    metric: str,
    grid: Any,
    value: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> str | None:
    """``judge_sampled_bound(...).unfit_reason``."""

    return judge_sampled_bound(metric, grid, value, lower=lower, upper=upper).unfit_reason


__all__ = [
    "EXTREMA_GRID_FIELD",
    "MIN_SAMPLES_PER_SWITCHING_PERIOD",
    "SAMPLED_EXTREMUM_METRICS",
    "SampledBound",
    "fitness_context",
    "judge_sampled_bound",
    "grid_unresolved_reason",
    "sampled_bound_unfit_reason",
    "sampled_extremum_bias",
]
