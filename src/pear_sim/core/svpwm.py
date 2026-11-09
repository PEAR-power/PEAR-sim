# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Sector-based space-vector modulation.

WHAT A SECTOR MODULATOR IS. A three-phase bridge can produce exactly eight
switching states: six active vectors 60 degrees apart and two zero vectors.
To synthesise a commanded voltage vector over one carrier period the
modulator picks the two active vectors bracketing it, spends a computed
share of the period on each, and fills the rest with the zero vectors. That
"which two, and for how long" is the sector decision, and having it EXPLICIT
is what this module adds — the capability catalogue has carried
`spwm_svpwm_dead_time` as `unsupported` precisely because no sector matrix
existed.

WHY THE RESULT LOOKS FAMILIAR. With the remaining zero-vector time split
EVENLY between the two zero states — the symmetric, centre-aligned choice —
the resulting duties are algebraically identical to sinusoidal modulation
with min-max (third-harmonic) injection. That is not a coincidence to be
explained away: it is a theorem, and it hands this module a free and
completely independent oracle. The test suite drives both paths over the
whole modulation range and the whole angle range and asserts they agree to
machine precision. A sector modulator that disagrees with min-max injection
is wrong, and now cannot ship.

WHAT IT ADDS BEYOND min_max. The commanded vector arrives as (alpha, beta)
rather than as an amplitude and a phase, which is what a field-oriented
controller actually produces, and the sector and dwell times become
observable quantities. Both are prerequisites for the things min-max
injection cannot express: overmodulation, discontinuous (DPWM) zero-vector
placement, and a modulator swapped under an unchanged control law.
Neither of those is implemented here, and saying so is part of the contract.
"""

from __future__ import annotations

import math
from typing import Final

from pear_sim.errors import ModelError

#: Largest vector magnitude, relative to the DC link, that the linear range
#: can synthesise: the radius of the circle inscribed in the vector hexagon.
LINEAR_RANGE_LIMIT: Final[float] = 1.0 / math.sqrt(3.0)

_SQRT3: Final[float] = math.sqrt(3.0)


def sector_of(alpha: float, beta: float) -> int:
    """Sector 1..6 containing the commanded vector.

    Sector 1 spans [0, 60) degrees measured from the alpha axis. A zero
    vector has no direction; it is reported as sector 1 so the caller gets a
    total function, and its dwell times are then both zero anyway.
    """

    angle = math.atan2(beta, alpha)
    if angle < 0.0:
        angle += 2.0 * math.pi
    return int(angle // (math.pi / 3.0)) + 1


def svpwm_duties(
    alpha: float,
    beta: float,
    dc_link_voltage: float,
) -> tuple[tuple[float, float, float], dict[str, float]]:
    """Phase duties for a commanded vector, plus the sector observation.

    ``alpha``/``beta`` are the commanded phase voltages in the stationary
    frame (volts, amplitude-invariant convention); ``dc_link_voltage`` is the
    full DC link. Returns duties in [0, 1] referred to the negative rail and
    a record of the sector, the two dwell fractions and the zero-vector time.

    Fails closed outside the linear range rather than clipping quietly:
    overmodulation is a different modulator with its own harmonic contract,
    and silently saturating into it would misreport what the bridge did.
    """

    if dc_link_voltage <= 0.0:
        raise ModelError("svpwm requires a positive dc_link_voltage")
    magnitude = math.hypot(alpha, beta)
    modulation = magnitude / dc_link_voltage
    if modulation > LINEAR_RANGE_LIMIT * (1.0 + 1e-12):
        raise ModelError(
            f"commanded vector {magnitude:.6g} V is {modulation:.6g} of the DC "
            f"link, beyond the linear-range limit {LINEAR_RANGE_LIMIT:.6g} "
            "(1/sqrt(3)); overmodulation is not implemented and is refused "
            "rather than clipped"
        )

    sector = sector_of(alpha, beta)
    # Angle measured from the start of the sector, in [0, 60).
    angle = math.atan2(beta, alpha)
    if angle < 0.0:
        angle += 2.0 * math.pi
    local = angle - (sector - 1) * (math.pi / 3.0)

    # Dwell fractions of the two bracketing active vectors.
    scale = _SQRT3 * magnitude / dc_link_voltage
    t_first = scale * math.sin(math.pi / 3.0 - local)
    t_second = scale * math.sin(local)
    t_zero = max(0.0, 1.0 - t_first - t_second)

    # Symmetric placement: the zero time is split evenly between the two zero
    # vectors, which centres every pulse and is what makes this equivalent to
    # min-max injection. A different split is a different modulator (DPWM),
    # not a tuning knob, and is out of scope here.
    half_zero = 0.5 * t_zero

    # THE SECTOR MATRIX, WRITTEN AS THE SWITCHING STATES THEMSELVES.
    # The six active vectors are the bridge's own outputs, each a triple of
    # "is this phase tied to the positive rail":
    #     V1 (1,0,0)  V2 (1,1,0)  V3 (0,1,0)
    #     V4 (0,1,1)  V5 (0,0,1)  V6 (1,0,1)
    # and sector n is bracketed by (V_n, V_{n+1}) cyclically. A phase's duty
    # is then simply the time it spends high: half the zero time (both zero
    # vectors are shared by every phase equally at the pattern's centre) plus
    # whichever active dwell it is high for.
    #
    # A first version of this used a hand-written high/mid/low table instead.
    # It agreed with the min-max oracle in sector 1 and disagreed by up to
    # 0.85 elsewhere, because it assigned the dwell times to the wrong phases
    # in the sectors where the pattern reverses. Deriving the duty from the
    # states removes the chance to get that wrong.
    active_vectors = {
        1: ((1, 0, 0), (1, 1, 0)),
        2: ((1, 1, 0), (0, 1, 0)),
        3: ((0, 1, 0), (0, 1, 1)),
        4: ((0, 1, 1), (0, 0, 1)),
        5: ((0, 0, 1), (1, 0, 1)),
        6: ((1, 0, 1), (1, 0, 0)),
    }
    first_state, second_state = active_vectors[sector]
    duties = tuple(
        half_zero + t_first * first_state[phase] + t_second * second_state[phase]
        for phase in range(3)
    )
    observation = {
        "sector": float(sector),
        "dwell_first": t_first,
        "dwell_second": t_second,
        "dwell_zero": t_zero,
        "modulation": modulation,
    }
    return duties, observation


def min_max_injection_duties(
    alpha: float,
    beta: float,
    dc_link_voltage: float,
) -> tuple[float, float, float]:
    """The same duties by the OTHER route: sine references plus min-max.

    Kept here beside the sector modulator on purpose. It is the independent
    oracle the sector implementation is judged against, and reading the two
    side by side is what makes the equivalence checkable by a person as well
    as by a test.
    """

    references = [
        alpha,
        -0.5 * alpha + 0.5 * _SQRT3 * beta,
        -0.5 * alpha - 0.5 * _SQRT3 * beta,
    ]
    normalized = [value / (0.5 * dc_link_voltage) for value in references]
    offset = -0.5 * (max(normalized) + min(normalized))
    return tuple(0.5 * (value + offset + 1.0) for value in normalized)
