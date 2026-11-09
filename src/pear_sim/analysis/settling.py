# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Is a statistics window on the periodic orbit?

Last-half statistics describe steady state only if the waveform repeats across
the whole window, not just at its end: a run can be settled by its final period
while the window still opens on start-up ringing. The test compares the last
switching period of the window with the same-phase period at the window's start
(a whole number of periods earlier) and reports the largest difference.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["window_drift"]


def window_drift(time: Any, values: Any, period: float, window_start: float) -> float | None:
    """max |x(t) - x(t - n*T)| over the last period, n*T spanning the window.

    Returns None when the window holds fewer than two whole periods.
    """

    times = np.asarray(time, dtype=float)
    data = np.asarray(values, dtype=float)
    end = float(times[-1])
    cycles = int(np.floor((end - window_start) / period + 1e-9)) - 1
    if cycles < 1:
        return None
    shift = cycles * period
    mask = times >= end - period
    if not np.any(mask):
        return None
    late = data[mask]
    early = np.interp(times[mask] - shift, times, data)
    return float(np.max(np.abs(late - early)))
