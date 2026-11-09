# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Dense-state polynomials as raw arrays are bit-identical to the Polynomial class.

State extrema are found from every accepted step's dense-output polynomial.
The kernel builds those polynomials as plain coefficient arrays with the same
numpy routines the Polynomial class delegates to; the class only added object
overhead. The reference below is the former class-based construction, kept
verbatim, so any drift in a single bit — coefficients, derivative roots or
values — fails here, for every solve_ivp dense-output kind the kernel accepts.
"""

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from scipy.integrate import solve_ivp

from pear_sim.core import metrics


def _reference_polynomials(interpolant, state_count):  # former implementation
    kind = type(interpolant).__name__
    if kind in {"RkDenseOutput", "RadauDenseOutput"}:
        factor = float(interpolant.h) if kind == "RkDenseOutput" else 1.0
        return tuple(
            Polynomial(
                np.concatenate(
                    (
                        np.asarray([interpolant.y_old[index]], dtype=float),
                        factor * np.asarray(interpolant.Q[index, :], dtype=float),
                    )
                )
            )
            for index in range(state_count)
        )
    if kind == "Dop853DenseOutput":
        x = Polynomial([0.0, 1.0])
        one_minus_x = Polynomial([1.0, -1.0])
        polynomials = []
        for state_index in range(state_count):
            polynomial = Polynomial([0.0])
            for index, coefficient in enumerate(reversed(interpolant.F[:, state_index])):
                polynomial += float(coefficient)
                polynomial *= x if index % 2 == 0 else one_minus_x
            polynomial += float(interpolant.y_old[state_index])
            polynomials.append(polynomial)
        return tuple(polynomials)
    if kind == "BdfDenseOutput":
        step = float(interpolant.t - interpolant.t_old)
        polynomials = []
        for state_index in range(state_count):
            polynomial = Polynomial([float(interpolant.D[0, state_index])])
            product = Polynomial([1.0])
            for index in range(interpolant.order):
                product *= Polynomial(
                    [
                        float(
                            (interpolant.t_old - interpolant.t_shift[index])
                            / interpolant.denom[index]
                        ),
                        float(step / interpolant.denom[index]),
                    ]
                )
                polynomial += float(interpolant.D[index + 1, state_index]) * product
            polynomials.append(polynomial)
        return tuple(polynomials)
    raise AssertionError(kind)


def _damped_oscillator(t, y):
    # Three states with real and complex modes plus a forcing term, so the
    # derivative polynomials have real roots inside the step and complex ones.
    return np.array(
        [
            y[1],
            -4.0e6 * y[0] - 3.0e2 * y[1] + 5.0 * np.sin(1.3e3 * t),
            -1.7e3 * y[2] + y[0],
        ]
    )


@pytest.mark.parametrize("method", ["RK45", "DOP853", "Radau", "BDF"])
def test_raw_array_polynomials_match_the_class_bit_for_bit(method: str) -> None:
    solution = solve_ivp(
        _damped_oscillator,
        (0.0, 4.0e-3),
        [1.0, 0.0, 0.2],
        method=method,
        dense_output=True,
        rtol=1e-9,
        atol=1e-12,
    )
    interpolants = solution.sol.interpolants
    assert len(interpolants) > 20
    probe = np.linspace(0.0, 1.0, 9)
    for interpolant in interpolants:
        new = metrics._dense_step_state_polynomials(interpolant, 3)
        old = _reference_polynomials(interpolant, 3)
        for raw, reference in zip(new, old, strict=True):
            assert raw.dtype == reference.coef.dtype
            assert np.array_equal(raw, reference.coef)
            assert np.array_equal(
                metrics._poly_derivative(raw), reference.deriv().coef
            )
            np.testing.assert_array_equal(
                metrics._poly_roots(metrics._poly_derivative(raw)),
                reference.deriv().roots(),
            )
            np.testing.assert_array_equal(
                metrics._poly_value(raw, probe), reference(probe)
            )
