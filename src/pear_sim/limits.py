# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Shared resource limits for portable run producers and consumers."""

MAX_RUN_EVENT_ROWS = 100_000
# Output samples per transient run: (stop - start) / output_step + 1. The grid is
# allocated up front and every sample is kept in memory and written to the bundle.
# Measured on the shipped buck (M1 Pro): 2 M samples = 0.9 GB peak memory, 209 MB
# on disk, 46 s. Without a bound, one mistyped output_step (2 s at 10 ns) fills
# the machine's memory long before any time limit stops it. The largest shipped
# example uses 50 001 samples.
MAX_RUN_OUTPUT_SAMPLES = 2_000_001
MAX_SCHEDULED_GATE_EDGES = 100_000
# Active-set search budget: candidate diode configurations examined per
# stable-mode query, in (Hamming distance from the preferred configuration,
# lexicographic) order. 2**16 keeps every model that the former 16-diode
# limit accepted bit-identical (the full space is still enumerated), while
# larger diode counts (multilevel bridges) get a measured, portable bound
# instead of a hard component-count wall.
MAX_ACTIVE_SET_CANDIDATES = 65_536
MAX_IDEAL_DIODES = 64
MAX_MODEL_LOGICAL_CONTROLS = 256
MAX_MODEL_GATE_OUTPUTS = 256
MAX_HARMONIC_SIGNALS = 32
MAX_HARMONIC_ORDER = 256
HARMONIC_QUADRATURE_ORDER = 8
MAX_HARMONIC_QUADRATURE_PANELS = 100_000
MAX_HARMONIC_QUADRATURE_WORK_UNITS = 100_000_000
# One three-phase carrier cell emits at most twelve scheduled gate edges. Keeping
# one synchronous reference table within the global event budget makes every model
# that passes semantic validation schedulable under the same portable bound.
MAX_SPWM_CARRIER_RATIO = MAX_SCHEDULED_GATE_EDGES // 12
# A three-level NPC carrier cell has two complementary pairs per phase; each
# pair emits at most six edges per cell (a boundary band-transition fix-up
# plus the center-aligned pulse), so one cell is bounded by 36 edges.
MAX_NPC_CARRIER_RATIO = MAX_SCHEDULED_GATE_EDGES // 36


__all__ = [
    "HARMONIC_QUADRATURE_ORDER",
    "MAX_ACTIVE_SET_CANDIDATES",
    "MAX_HARMONIC_ORDER",
    "MAX_HARMONIC_QUADRATURE_PANELS",
    "MAX_HARMONIC_QUADRATURE_WORK_UNITS",
    "MAX_HARMONIC_SIGNALS",
    "MAX_IDEAL_DIODES",
    "MAX_MODEL_GATE_OUTPUTS",
    "MAX_MODEL_LOGICAL_CONTROLS",
    "MAX_NPC_CARRIER_RATIO",
    "MAX_RUN_EVENT_ROWS",
    "MAX_RUN_OUTPUT_SAMPLES",
    "MAX_SCHEDULED_GATE_EDGES",
    "MAX_SPWM_CARRIER_RATIO",
]
