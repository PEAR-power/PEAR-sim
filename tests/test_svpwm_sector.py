"""Sector space-vector modulation, judged against an independent oracle.

The capability catalogue has carried `spwm_svpwm_dead_time` as `unsupported`
because no sector matrix existed. This is that matrix.

The whole test strategy rests on a theorem: with the zero-vector time split
evenly between the two zero states, sector SVPWM and sinusoidal modulation
with min-max injection produce ALGEBRAICALLY IDENTICAL duties. That gives a
free, fully independent oracle — one built from sines and a min/max, the
other from sector geometry and dwell times — and any disagreement means the
sector implementation is wrong.

It earned its keep immediately: the first version used a hand-written
high/mid/low table, agreed with the oracle in sector 1, and disagreed by up
to 0.85 in the sectors where the switching pattern reverses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pear_sim.core.svpwm import (
    LINEAR_RANGE_LIMIT,
    min_max_injection_duties,
    sector_of,
    svpwm_duties,
)
from pear_sim.errors import ModelError

DC_LINK = 600.0


def test_sector_modulation_equals_min_max_injection_everywhere() -> None:
    """The headline gate: two independent routes, one answer."""

    worst = 0.0
    for modulation in np.linspace(0.005, LINEAR_RANGE_LIMIT * 0.9999, 40):
        for angle in np.linspace(0.0, 2.0 * math.pi, 361, endpoint=False):
            alpha = modulation * DC_LINK * math.cos(angle)
            beta = modulation * DC_LINK * math.sin(angle)
            sector_duties, _ = svpwm_duties(alpha, beta, DC_LINK)
            oracle = min_max_injection_duties(alpha, beta, DC_LINK)
            worst = max(
                worst,
                max(abs(a - b) for a, b in zip(sector_duties, oracle, strict=True)),
            )
    # Measured 8.9e-16 over 14 440 points; the gate is set at machine
    # precision because these are the same numbers, not similar ones.
    assert worst < 1e-12, f"sector modulator drifted from min-max by {worst:.3e}"


@pytest.mark.parametrize(
    ("angle_deg", "expected"),
    [(0.0, 1), (30.0, 1), (59.9, 1), (60.0, 2), (150.0, 3), (200.0, 4), (280.0, 5), (330.0, 6)],
)
def test_sector_boundaries(angle_deg: float, expected: int) -> None:
    angle = math.radians(angle_deg)
    assert sector_of(math.cos(angle), math.sin(angle)) == expected


def test_duties_stay_inside_the_unit_interval() -> None:
    for modulation in np.linspace(0.0, LINEAR_RANGE_LIMIT, 30):
        for angle in np.linspace(0.0, 2.0 * math.pi, 73, endpoint=False):
            duties, _ = svpwm_duties(
                modulation * DC_LINK * math.cos(angle),
                modulation * DC_LINK * math.sin(angle),
                DC_LINK,
            )
            assert all(-1e-12 <= duty <= 1.0 + 1e-12 for duty in duties)


def test_dwell_times_fill_the_period() -> None:
    _, observation = svpwm_duties(200.0, 100.0, DC_LINK)
    total = (
        observation["dwell_first"]
        + observation["dwell_second"]
        + observation["dwell_zero"]
    )
    assert total == pytest.approx(1.0, abs=1e-12)


def test_a_zero_vector_gives_every_phase_half_the_period() -> None:
    duties, observation = svpwm_duties(0.0, 0.0, DC_LINK)
    assert all(duty == pytest.approx(0.5) for duty in duties)
    assert observation["dwell_zero"] == pytest.approx(1.0)


def test_overmodulation_is_refused_not_clipped() -> None:
    """Silently saturating would misreport what the bridge did.

    Beyond 1/sqrt(3) the commanded vector leaves the circle inscribed in the
    hexagon and can no longer be synthesised on average. Clipping there is a
    DIFFERENT modulator with its own harmonic contract, so it is refused.
    """

    magnitude = LINEAR_RANGE_LIMIT * DC_LINK * 1.01
    with pytest.raises(ModelError, match="linear-range limit"):
        svpwm_duties(magnitude, 0.0, DC_LINK)


def test_the_linear_range_boundary_itself_is_accepted() -> None:
    """Saturation is reached at 30 degrees, not at 0 — and that is geometry.

    The linear range is the circle INSCRIBED in the vector hexagon, so it
    touches the hexagon at the midpoints of its sides (30, 90, ... degrees)
    and stays strictly inside at the vertices (0, 60, ...). A duty of exactly
    1.0 therefore appears at 30 degrees, while at 0 degrees the same
    magnitude still leaves headroom (0.933). The first version of this test
    asserted saturation at 0 degrees and was simply wrong about the shape.
    """

    magnitude = LINEAR_RANGE_LIMIT * DC_LINK
    at_vertex, _ = svpwm_duties(magnitude, 0.0, DC_LINK)
    assert max(at_vertex) == pytest.approx(0.9330127, abs=1e-6)
    at_side, _ = svpwm_duties(
        magnitude * math.cos(math.pi / 6.0),
        magnitude * math.sin(math.pi / 6.0),
        DC_LINK,
    )
    assert max(at_side) == pytest.approx(1.0, abs=1e-4)


def test_a_non_positive_dc_link_is_refused() -> None:
    with pytest.raises(ModelError, match="positive dc_link_voltage"):
        svpwm_duties(10.0, 0.0, 0.0)
