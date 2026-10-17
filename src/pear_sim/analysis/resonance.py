"""Modal (resonance) audit of the recorded run.

An LC output filter, a DC link, a cable or a parasitic loop all put lightly
damped poles into the network. Whether that matters is not a property of the
filter alone: it depends on where the poles sit RELATIVE TO WHAT EXCITES
THEM. This producer answers exactly that question, and does it structurally
rather than by recognizing named topologies.

For every hybrid mode the solver actually accepted, the eigenvalues of that
mode's state matrix give its natural frequencies and damping ratios. Poles
that recur across modes covering most of the run are structural (a filter);
poles appearing in one brief mode are not. The excitation lines are taken
from the declared modulation — the carrier, its harmonics and their
sidebands, plus the fundamental — and every structural resonance is reported
with its nearest line and the separation between them.

DECLARED LIMITS. Eigenvalues come from the per-mode LINEARIZED state matrix
of an ideal-switch model: damping is whatever the explicitly modelled
resistances provide, so copper, core and semiconductor losses that a real
filter also damps with are absent and the reported damping is a LOWER bound.
The excitation line list covers the carrier band and the fundamental; the
baseband harmonics that dead time and device nonlinearity inject are real
but not enumerated here, so a resonance reported as clear of excitation may
still be driven by them. This is an audit that flags a design question, not
a stability proof.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pear_sim.core.circuit import Circuit

RESONANCE_SCHEMA = "pear-sim/resonance-observation/v0.1"
RESONANCE_KIND = "modal-resonance-observation"

# Declared thresholds. They select what gets FLAGGED for a human; they never
# change a number and never gate a run on their own.
LIGHTLY_DAMPED_RATIO = 0.2
PROXIMITY_FRACTION = 0.15
# A pole confined to a sliver of the run (a dead-time window) is not a design
# feature; one that exists for a fifth of it or more is. Discontinuous
# conduction legitimately parks a filter pole for part of every cycle, so a
# half-the-run threshold would hide exactly the case worth flagging.
STRUCTURAL_COVERAGE = 0.2
REPORTED_COVERAGE_FLOOR = 0.01
CLUSTER_TOLERANCE = 0.05
MAX_ANALYSED_MODES = 256
MAX_REPORTED_RESONANCES = 32
# Carrier harmonics and sideband orders enumerated around each of them.
CARRIER_HARMONICS = 3
SIDEBAND_ORDERS = 6


class ResonanceError(ValueError):
    """Raised when a run cannot support an exact modal observation."""

    code = "analysis.resonance_invalid"


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _mode_durations(result: Any) -> tuple[dict[str, dict[str, Any]], float]:
    """Total accepted time per distinct mode signature."""

    intervals = getattr(result, "mode_intervals", None)
    if not isinstance(intervals, list) or not intervals:
        raise ResonanceError("result carries no accepted mode intervals")
    modes: dict[str, dict[str, Any]] = {}
    total = 0.0
    for interval in intervals:
        start = _finite(interval.get("start_s"))
        end = _finite(interval.get("end_s"))
        signature = interval.get("mode_signature")
        if start is None or end is None or not isinstance(signature, str):
            raise ResonanceError("mode interval is missing timing or signature")
        duration = max(0.0, end - start)
        total += duration
        entry = modes.setdefault(
            signature,
            {
                "duration_s": 0.0,
                "gates": dict(interval.get("gates", {})),
                "diodes": dict(interval.get("diodes", {})),
                "constrained_inductors": list(
                    interval.get("constrained_inductors", ())
                ),
            },
        )
        entry["duration_s"] += duration
    if total <= 0.0:
        raise ResonanceError("accepted mode intervals have zero total duration")
    return modes, total


def _modal_pairs(matrix: np.ndarray) -> list[tuple[float, float]]:
    """(frequency [Hz], damping ratio) of every oscillatory eigenvalue pair."""

    if matrix.size == 0:
        return []
    eigenvalues = np.linalg.eigvals(matrix)
    pairs: list[tuple[float, float]] = []
    for value in eigenvalues:
        imaginary = float(np.imag(value))
        if imaginary <= 0.0:
            continue  # conjugate pairs are reported once
        real = float(np.real(value))
        magnitude = math.hypot(real, imaginary)
        if magnitude <= 0.0 or not math.isfinite(magnitude):
            continue
        pairs.append((imaginary / math.tau, -real / magnitude))
    return pairs


def _excitation_lines(model: Any) -> dict[str, Any]:
    """Frequencies the modulation actually injects, from declared metadata."""

    carriers: set[float] = set()
    fundamentals: set[float] = set()
    for control in getattr(model, "controls", {}).values():
        frequency = _finite(getattr(control, "frequency", None))
        if frequency and frequency > 0.0:
            carriers.add(frequency)
        fundamental = _finite(getattr(control, "fundamental_frequency", None))
        if fundamental and fundamental > 0.0:
            fundamentals.add(fundamental)
    for control in getattr(model, "sampled_controls", {}).values():
        frequency = _finite(getattr(control, "frequency", None))
        if frequency and frequency > 0.0:
            carriers.add(frequency)
        nominal = _finite(getattr(control, "pll_nominal_frequency_hz", None))
        if nominal and nominal > 0.0:
            fundamentals.add(nominal)
    for component in getattr(model, "components", ()):
        waveform = getattr(component, "waveform", None)
        frequency = _finite(getattr(waveform, "frequency_hz", None))
        if frequency and frequency > 0.0:
            fundamentals.add(frequency)

    lines: set[float] = set()
    for fundamental in fundamentals:
        lines.add(fundamental)
    for carrier in carriers:
        for harmonic in range(1, CARRIER_HARMONICS + 1):
            centre = harmonic * carrier
            lines.add(centre)
            for fundamental in fundamentals:
                for order in range(1, SIDEBAND_ORDERS + 1):
                    lines.add(centre + order * fundamental)
                    lower = centre - order * fundamental
                    if lower > 0.0:
                        lines.add(lower)
    return {
        "carrier_frequencies_hz": sorted(carriers),
        "fundamental_frequencies_hz": sorted(fundamentals),
        "lines_hz": sorted(lines),
        "line_model": (
            f"fundamentals plus carrier harmonics up to {CARRIER_HARMONICS} with "
            f"sidebands up to order {SIDEBAND_ORDERS}; baseband harmonics from "
            "dead time and device nonlinearity are NOT enumerated"
        ),
    }


def build_resonance_observation(result: Any) -> dict[str, Any]:
    """Cluster the accepted modes' oscillatory poles and place them against
    what the modulation excites."""

    modes, total_duration = _mode_durations(result)
    circuit = Circuit(result.model)
    ordered = sorted(
        modes.items(), key=lambda item: (-item[1]["duration_s"], item[0])
    )
    analysed = ordered[:MAX_ANALYSED_MODES]
    truncated = len(ordered) - len(analysed)

    # (frequency, damping, coverage) samples across the analysed modes.
    samples: list[tuple[float, float, float, str]] = []
    unanalysable: list[str] = []
    for signature, entry in analysed:
        coverage = entry["duration_s"] / total_duration
        try:
            mode = circuit.compile(
                entry["gates"],
                entry["diodes"],
                entry["constrained_inductors"],
                (),
            )
        except Exception:  # noqa: BLE001 - a mode we cannot rebuild is declared
            unanalysable.append(signature)
            continue
        for frequency, damping in _modal_pairs(np.asarray(mode.a, dtype=float)):
            samples.append((frequency, damping, coverage, signature))

    # Cluster by relative frequency: the same physical resonance appears in
    # every mode that leaves its branch intact.
    clusters: list[dict[str, Any]] = []
    for frequency, damping, coverage, signature in sorted(samples):
        target = None
        for cluster in clusters:
            reference = cluster["frequency_hz"]
            if abs(frequency - reference) <= CLUSTER_TOLERANCE * max(
                reference, 1e-12
            ):
                target = cluster
                break
        if target is None:
            clusters.append(
                {
                    "frequency_hz": frequency,
                    "damping_ratio": damping,
                    "coverage": coverage,
                    "modes": {signature},
                    "min_damping_ratio": damping,
                }
            )
            continue
        target["coverage"] += coverage
        target["modes"].add(signature)
        target["min_damping_ratio"] = min(target["min_damping_ratio"], damping)

    excitation = _excitation_lines(result.model)
    lines = excitation["lines_hz"]
    resonances: list[dict[str, Any]] = []
    for cluster in sorted(clusters, key=lambda item: -item["coverage"]):
        coverage = min(1.0, cluster["coverage"])
        if coverage < REPORTED_COVERAGE_FLOOR:
            continue
        frequency = cluster["frequency_hz"]
        damping = cluster["min_damping_ratio"]
        nearest = None
        separation = None
        if lines and frequency > 0.0:
            nearest = min(lines, key=lambda line: abs(line - frequency))
            separation = abs(nearest - frequency) / frequency
        structural = coverage >= STRUCTURAL_COVERAGE
        lightly_damped = damping < LIGHTLY_DAMPED_RATIO
        near_excitation = separation is not None and separation <= PROXIMITY_FRACTION
        resonances.append(
            {
                "frequency_hz": frequency,
                "damping_ratio": damping,
                "mode_coverage_fraction": coverage,
                "structural": structural,
                "lightly_damped": lightly_damped,
                "nearest_excitation_hz": nearest,
                "separation_fraction": separation,
                "near_excitation": near_excitation,
                "flagged": structural and lightly_damped and near_excitation,
                "present_in_modes": len(cluster["modes"]),
            }
        )
        if len(resonances) >= MAX_REPORTED_RESONANCES:
            break

    flagged = [entry for entry in resonances if entry["flagged"]]
    return {
        "schema": RESONANCE_SCHEMA,
        "kind": RESONANCE_KIND,
        "status": "flagged" if flagged else "ok",
        "thresholds": {
            "lightly_damped_below_ratio": LIGHTLY_DAMPED_RATIO,
            "near_excitation_within_fraction": PROXIMITY_FRACTION,
            "structural_coverage_fraction": STRUCTURAL_COVERAGE,
            "cluster_relative_tolerance": CLUSTER_TOLERANCE,
        },
        "excitation": excitation,
        "analysed_modes": len(analysed),
        "modes_not_rebuilt": sorted(unanalysable),
        "modes_truncated": max(0, truncated),
        "resonances": resonances,
        "claim": (
            "Eigenvalues of each accepted mode's linearized state matrix. "
            "Damping comes only from explicitly modelled resistances, so it is "
            "a LOWER bound on the physical damping; the excitation list covers "
            "the carrier band and the fundamentals, not the baseband harmonics "
            "injected by dead time and device nonlinearity. A flag marks a "
            "design question to answer, not a proven instability."
        ),
    }
