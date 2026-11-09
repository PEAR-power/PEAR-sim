# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Gradient backend building blocks.

The explicit mode sequence with provenance and the typed design vector. The
engine itself (monodromy + shooting + sensitivities) builds on these and is
guarded by the T1+T2 gate tests (steady-state agreement with the oracle and
Jacobian agreement with finite differences).
"""

from pear_sim.analysis.gradient.design_vector import DesignVector, DesignVectorError
from pear_sim.analysis.gradient.losses import (
    LossEvaluation,
    LossGradientError,
    NO_THERMAL_TERM_LABEL,
)
from pear_sim.analysis.gradient.engine import (
    AssumptionFlag,
    GradientEngineError,
    GradientResult,
    MetricJacobian,
    MetricSpec,
    SteadyStatePoint,
    solve,
)
from pear_sim.analysis.gradient.sequence import (
    EXACTLY_TIMED_EXITS,
    MODE_SEQUENCE_SCHEMA,
    ExitCondition,
    ExitKind,
    ModeSequence,
    ModeSequenceError,
    ModeStep,
    extract_mode_sequence,
    manual_mode_sequence,
    mode_sequence_as_dict,
    mode_sequence_from_dict,
    sequences_agree,
)

__all__ = [
    "AssumptionFlag",
    "DesignVector",
    "DesignVectorError",
    "GradientEngineError",
    "GradientResult",
    "LossEvaluation",
    "LossGradientError",
    "MetricJacobian",
    "NO_THERMAL_TERM_LABEL",
    "MetricSpec",
    "SteadyStatePoint",
    "solve",
    "EXACTLY_TIMED_EXITS",
    "ExitCondition",
    "ExitKind",
    "ModeSequence",
    "ModeSequenceError",
    "ModeStep",
    "MODE_SEQUENCE_SCHEMA",
    "extract_mode_sequence",
    "manual_mode_sequence",
    "mode_sequence_as_dict",
    "mode_sequence_from_dict",
    "sequences_agree",
]
