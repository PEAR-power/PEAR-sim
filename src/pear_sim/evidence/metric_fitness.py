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

This is the one place that rule lives. ``evaluate`` (expectations), the
perturbation campaign and the batch ranking all call it, so the same run and
the same requirement cannot get different verdicts on different paths.

The floor rules out aliasing against the declared carrier. It is not an
accuracy statement: a resolved sampled extremum is still a grid statistic.
"""

from __future__ import annotations

from collections.abc import Mapping
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


def sampled_bound_unfit_reason(
    metric: str,
    grid: Any,
    value: float,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> str | None:
    """Why ``lower <= metric <= upper`` cannot be decided from ``value``, or ``None``.

    ``None`` means the plain comparison of ``value`` with the bounds is a
    sound verdict. ``grid`` is the signal record's ``extrema_grid``. With
    neither bound the question is whether the number itself is usable (a
    ranking objective), which needs a resolved grid.
    """

    bias = sampled_extremum_bias(metric)
    if bias is None:
        return None
    unresolved = grid_unresolved_reason(grid)
    if unresolved is None:
        return None
    if bias == "under":
        # true >= value
        if upper is not None and value > upper:
            return None  # already above the limit; the true value is too
        if upper is None and lower is not None and value >= lower:
            return None  # already above the floor; the true value is too
        direction = "a lower bound on"
    else:
        # true <= value
        if lower is not None and value < lower:
            return None
        if lower is None and upper is not None and value <= upper:
            return None
        direction = "an upper bound on"
    return (
        f"sampled {metric} is only {direction} the true value and {unresolved}, "
        "so it cannot decide this requirement"
    )


__all__ = [
    "EXTREMA_GRID_FIELD",
    "MIN_SAMPLES_PER_SWITCHING_PERIOD",
    "SAMPLED_EXTREMUM_METRICS",
    "grid_unresolved_reason",
    "sampled_bound_unfit_reason",
    "sampled_extremum_bias",
]
