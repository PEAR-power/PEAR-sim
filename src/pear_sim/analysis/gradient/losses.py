# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Gradient backend extension: device losses inside the Jacobian.

Loss power becomes a metric the engine can differentiate: conduction energy is
integrated over the closed-form steady trajectory (Gauss-Legendre per
segment), switching energies are looked up at the exact segment-boundary
currents and commutation voltages, and both reuse the ORACLE'S OWN table
semantics — the clamped piecewise-linear lookup is imported from
``analysis.losses_thermal``, not re-implemented, so the engine cannot quietly
diverge from what the artifact pipeline computes.

Three decisions govern the honesty of the result:

Smoothing — table smoothing (a monotone PCHIP spline over the same nodes) exists in
     THIS module only, never in the oracle, is off by default, and when
     enabled the result carries a declaration flag that also reports the
     difference against the unsmoothed evaluation.
Clamping — a lookup clamped outside its axis is a FLAG, not a silent plateau: the
     per-device flags carry the signed distance from the operating currents
     to the table edge, so an optimizer sees the wall before leaning on a
     zero gradient that the physics does not have.
Thermal — the electrothermal loop stays OPEN: every card is evaluated at the
     declared ``tj_evaluation_c``, and the result is labelled "no d/dT_j
     term" instead of pretending the Jacobian carries one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pear_sim.analysis.losses_thermal import _Clamped1D, _conduction_voltage_lookup
from pear_sim.errors import PearSimError

__all__ = [
    "LossEvaluation",
    "LossGradientError",
    "NO_THERMAL_TERM_LABEL",
]

NO_THERMAL_TERM_LABEL = (
    "loss Jacobian carries no d/dT_j term: every card is evaluated at the "
    "declared tj_evaluation_c (open-loop thermal)"
)


class LossGradientError(PearSimError):
    """The model cannot support loss metrics in the gradient engine."""

    code = "gradient.losses_invalid"


@dataclass
class _TableMargin:
    """Worst-case proximity of the evaluated currents to a table's axis end.

    ``margin`` is signed in the axis' own units (amperes): positive means the
    closest evaluated current stayed that far INSIDE the axis, negative means
    an evaluation was clamped that far BEYOND the edge — the flag consumers
    never have to guess which."""

    margin: float = float("inf")
    clipped: int = 0

    def observe(self, position: float, axis: tuple[float, ...]) -> None:
        low, high = axis[0], axis[-1]
        if position < low:
            self.margin = min(self.margin, position - low)
            self.clipped += 1
        elif position > high:
            self.margin = min(self.margin, high - position)
            self.clipped += 1
        else:
            self.margin = min(self.margin, position - low, high - position)


def _smoothed(lookup: _Clamped1D) -> object:
    """A monotone cubic over the same nodes — engine-only, never in the oracle.

    PCHIP preserves monotone runs and never overshoots the table values, so
    smoothing changes DIFFERENTIABILITY at the nodes, not the claim about
    what the table says. Clamping behaviour outside the axis is unchanged.
    """

    from scipy.interpolate import PchipInterpolator

    axis = np.asarray(lookup.axis, dtype=float)
    values = np.asarray(lookup.values, dtype=float)
    if len(axis) < 3:
        return lookup  # a two-point table has nothing to smooth
    interpolator = PchipInterpolator(axis, values, extrapolate=False)

    def evaluate(position: float) -> float:
        if position <= axis[0]:
            return float(values[0])
        if position >= axis[-1]:
            return float(values[-1])
        return float(interpolator(position))

    return evaluate


@dataclass(frozen=True)
class LossEvaluation:
    """Per-cycle loss powers plus the honesty payload for the flags."""

    total_power_w: float
    per_device_power_w: dict[str, float]
    table_margins: dict[str, tuple[float, int]]
    """device id -> (signed worst margin in A, clipped lookup count)."""

    smoothed: bool
    unsmoothed_total_power_w: float | None
    """Only set when smoothing is on: the same evaluation without it."""


def evaluate_losses(
    model,
    circuit,
    modes,
    operators,
    boundary_states,
    durations,
    period_s: float,
    gauss_nodes: np.ndarray,
    gauss_weights: np.ndarray,
    segment_operator,
    *,
    smooth_tables: bool = False,
) -> LossEvaluation:
    """Losses of the declared steady cycle, from the cards, analytically.

    ``segment_operator`` is the engine's `_segment_operators` — passed in so
    this module stays free of a circular import while reusing the exact same
    trajectory evaluation the metrics use.
    """

    spec = getattr(model, "loss_analysis", None)
    if spec is None:
        raise LossGradientError(
            "loss metrics require a loss_analysis section: an ideal-switch "
            "model has no losses to differentiate, and inventing them would "
            "be worse than refusing"
        )

    component_by_id = {component.id: component for component in model.components}
    per_device: dict[str, float] = {}
    margins: dict[str, _TableMargin] = {}
    unsmoothed_total: float | None = 0.0 if smooth_tables else None

    for binding in spec.devices:
        component = component_by_id[binding.component_id]
        card = spec.cards[binding.card_id]
        margin = _TableMargin()
        margins[binding.component_id] = margin

        v_on_table = _conduction_voltage_lookup(card, spec.tj_evaluation_c)
        v_on = _smoothed(v_on_table) if smooth_tables else v_on_table
        current_expression = f"i({binding.component_id})"

        # --- conduction: quadrature over the conducting segments ----------
        conduction_j = 0.0
        conduction_unsmoothed_j = 0.0
        for k, (mode, duration) in enumerate(zip(modes, durations)):
            row, const, _ = mode.output_affine(current_expression)
            row = np.asarray(row, dtype=float)
            if not np.any(row) and const == 0.0:
                continue  # the device does not conduct in this mode
            x0 = np.asarray(boundary_states[k], dtype=float)
            half = 0.5 * duration
            for node, weight in zip(gauss_nodes, gauss_weights):
                t = half * (node + 1.0)
                phi_t, psi_t, _, _ = segment_operator(mode.a, mode.c, t)
                magnitude = abs(float(row @ (phi_t @ x0 + psi_t)) + const)
                margin.observe(magnitude, v_on_table.axis)
                conduction_j += half * weight * v_on(magnitude) * magnitude
                if smooth_tables:
                    conduction_unsmoothed_j += (
                        half * weight * v_on_table(magnitude) * magnitude
                    )

        # --- switching: exact boundary currents and voltages ---------------
        switching_j = 0.0
        switching_unsmoothed_j = 0.0
        if card.switching is not None:
            table = card.switching
            lookups = {}
            for field in ("e_on_j", "e_off_j", "e_rec_j"):
                values = getattr(table, field)
                if values is not None:
                    base = _Clamped1D(table.current_axis_a, values)
                    lookups[field] = (
                        base,
                        _smoothed(base) if smooth_tables else base,
                    )

            def edge_energy(field: str, current_a: float, voltage_v: float):
                nonlocal switching_j, switching_unsmoothed_j
                if field not in lookups:
                    return
                base, active = lookups[field]
                margin.observe(current_a, table.current_axis_a)
                scale = abs(voltage_v) / table.test_voltage_v
                switching_j += active(current_a) * scale
                if smooth_tables:
                    switching_unsmoothed_j += base(current_a) * scale

            boundary_count = len(modes)
            for k in range(boundary_count):
                # The edge between step k and step (k+1) mod N happens at the
                # boundary state the steady map already carries; the wrap
                # closes the period so every edge is counted exactly once.
                before = modes[k]
                after = modes[(k + 1) % boundary_count]
                # boundary_states has N+1 entries; index k+1 is the state AT
                # this edge, and for the last step it equals the steady cycle
                # start, which closes the period without double counting.
                state = np.asarray(boundary_states[k + 1], dtype=float)
                row_b, const_b, _ = before.output_affine(current_expression)
                row_a, const_a, _ = after.output_affine(current_expression)
                i_before = abs(float(np.asarray(row_b) @ state) + const_b)
                i_after = abs(float(np.asarray(row_a) @ state) + const_a)
                conducts_before = bool(np.any(np.asarray(row_b))) or const_b != 0.0
                conducts_after = bool(np.any(np.asarray(row_a))) or const_a != 0.0

                voltage_row, voltage_const, _ = after.output_affine(
                    binding.commutation_voltage
                )
                commutation = float(np.asarray(voltage_row) @ state) + voltage_const

                if component.type == "ideal_switch":
                    if not conducts_before and conducts_after:
                        edge_energy("e_on_j", i_after, commutation)
                    elif conducts_before and not conducts_after:
                        edge_energy("e_off_j", i_before, commutation)
                else:  # diode: hard commutation off -> reverse recovery
                    if conducts_before and not conducts_after and i_before > 0.0:
                        edge_energy("e_rec_j", i_before, commutation)

        device_power = (conduction_j + switching_j) / period_s
        per_device[binding.component_id] = device_power
        if smooth_tables:
            unsmoothed_total += (
                conduction_unsmoothed_j + switching_unsmoothed_j
            ) / period_s

    return LossEvaluation(
        total_power_w=float(sum(per_device.values())),
        per_device_power_w=per_device,
        table_margins={
            device: (entry.margin, entry.clipped)
            for device, entry in margins.items()
        },
        smoothed=smooth_tables,
        unsmoothed_total_power_w=unsmoothed_total,
    )
