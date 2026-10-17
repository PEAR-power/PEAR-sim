"""Small linear-algebra primitives shared by independent affine benchmark oracles.

Topology equations remain in each oracle module.  This file only evaluates an
already-derived affine system with a matrix exponential; it has no dependency on the
PEARSim circuit compiler, solver, event manager, or mode classifier.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def _validated_system(
    matrix: np.ndarray,
    offset: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    resolved_matrix = np.asarray(matrix, dtype=float)
    resolved_offset = np.asarray(offset, dtype=float)
    if resolved_matrix.ndim != 2 or resolved_matrix.shape[0] != resolved_matrix.shape[1]:
        raise ValueError("Affine oracle matrix must be square")
    if resolved_offset.shape != (resolved_matrix.shape[0],):
        raise ValueError("Affine oracle offset must match the matrix dimension")
    if not np.all(np.isfinite(resolved_matrix)) or not np.all(np.isfinite(resolved_offset)):
        raise ValueError("Affine oracle coefficients must be finite")
    return resolved_matrix, resolved_offset


def affine_map(
    matrix: np.ndarray,
    offset: np.ndarray,
    duration_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``x(t+h) = transition @ x(t) + affine`` for constant A and b."""

    resolved_matrix, resolved_offset = _validated_system(matrix, offset)
    dimension = resolved_matrix.shape[0]
    augmented = np.zeros((dimension + 1, dimension + 1), dtype=float)
    augmented[:dimension, :dimension] = resolved_matrix
    augmented[:dimension, dimension] = resolved_offset
    transition = expm(augmented * duration_s)
    return (
        np.asarray(transition[:dimension, :dimension]),
        np.asarray(transition[:dimension, dimension]),
    )


def propagate(
    matrix: np.ndarray,
    offset: np.ndarray,
    state: np.ndarray,
    duration_s: float,
) -> np.ndarray:
    transition, affine = affine_map(matrix, offset, duration_s)
    resolved_state = np.asarray(state, dtype=float)
    if resolved_state.shape != affine.shape:
        raise ValueError("Affine oracle state must match the matrix dimension")
    return transition @ resolved_state + affine


def propagate_with_coordinate_integral(
    matrix: np.ndarray,
    offset: np.ndarray,
    state: np.ndarray,
    duration_s: float,
    coordinate_index: int = 1,
) -> tuple[np.ndarray, float]:
    """Propagate a state and integrate one state coordinate over the interval."""

    resolved_matrix, resolved_offset = _validated_system(matrix, offset)
    resolved_state = np.asarray(state, dtype=float)
    dimension = resolved_matrix.shape[0]
    if resolved_state.shape != (dimension,):
        raise ValueError("Affine oracle state must match the matrix dimension")
    if not 0 <= coordinate_index < dimension:
        raise ValueError("Affine oracle integral coordinate is outside the state")
    augmented = np.zeros((dimension + 2, dimension + 2), dtype=float)
    augmented[:dimension, :dimension] = resolved_matrix
    augmented[:dimension, dimension + 1] = resolved_offset
    augmented[dimension, coordinate_index] = 1.0
    initial = np.concatenate((resolved_state, np.asarray([0.0, 1.0])))
    propagated = expm(augmented * duration_s) @ initial
    return np.asarray(propagated[:dimension]), float(propagated[dimension])


__all__ = ["affine_map", "propagate", "propagate_with_coordinate_integral"]
