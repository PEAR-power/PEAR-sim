# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pear_sim.benchmarks import (
    extract_buck_boundary_parameters,
    run_buck_boundary_sweep,
)
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator
from pear_sim.sweep import _periodic_point_model


ROOT = Path(__file__).parents[1]


def test_dop853_records_bcm_natural_turnoff_before_coincident_rising_edge() -> None:
    source = load_model(ROOT / "examples" / "buck_dcm.yaml")
    parameters = extract_buck_boundary_parameters(source)
    boundary = run_buck_boundary_sweep(parameters, [1.0])
    reference = boundary.points[0]
    period = parameters.period_s
    model = _periodic_point_model(source, reference)
    model = replace(
        model,
        transient=replace(
            model.transient,
            solver="dop853",
            rtol=1e-10,
            atol=1e-12,
            max_step=period / 100.0,
            output_step=period / 200.0,
        ),
    )

    result = HybridSimulator(model).simulate()
    coincident = [
        event for event in result.events if event.time == pytest.approx(period, rel=0.0, abs=1e-18)
    ]

    assert [(event.kind, event.target, event.previous, event.current) for event in coincident] == [
        ("natural_commutation", "d1", True, False),
        ("scheduled_gate", "pwm", False, True),
    ]
    assert result.stats["natural_events"] == 1
    assert result.stats["scheduled_events"] == 2
    assert result.stats["state_constraint_events"] == 1
    assert result.signals["i(l1)"][-1] == 0.0
    assert result.time[-1] == period
    assert result.time_integrals["full"]["duration_s"] == pytest.approx(
        period,
        rel=0.0,
        abs=1e-18,
    )


def test_near_zero_guard_is_not_coincident_when_root_is_far_away_in_time() -> None:
    simulator = object.__new__(HybridSimulator)
    simulator.circuit = SimpleNamespace(diodes=[SimpleNamespace(id="d1")])
    simulator.guard_tolerance = 1e-10
    simulator.time_tolerance = 1e-12

    class SlowMode:
        source_count = 0

        @staticmethod
        def diode_guard_affine(
            _diode: object, _conducting: bool
        ) -> tuple[np.ndarray, float, np.ndarray]:
            return np.array([1.0]), 0.0, np.zeros(0)

        @staticmethod
        def derivative(_time: float, _state: np.ndarray) -> np.ndarray:
            return np.array([-1e-6])

        @staticmethod
        def diode_guard(
            _diode: object,
            state: np.ndarray,
            _conducting: bool,
            _time: float | None = None,
        ) -> float:
            return float(state[0])

    roots = simulator._coincident_natural_roots(
        SlowMode(),
        {"d1": True},
        1.0,
        np.array([5e-11]),
    )

    assert roots == []


def test_near_zero_guard_is_coincident_when_linearized_root_is_within_time_tolerance() -> None:
    simulator = object.__new__(HybridSimulator)
    simulator.circuit = SimpleNamespace(diodes=[SimpleNamespace(id="d1")])
    simulator.guard_tolerance = 1e-10
    simulator.time_tolerance = 1e-12

    class FastMode:
        source_count = 0

        @staticmethod
        def diode_guard_affine(
            _diode: object, _conducting: bool
        ) -> tuple[np.ndarray, float, np.ndarray]:
            return np.array([1.0]), 0.0, np.zeros(0)

        @staticmethod
        def derivative(_time: float, _state: np.ndarray) -> np.ndarray:
            return np.array([-100.0])

        @staticmethod
        def diode_guard(
            _diode: object,
            state: np.ndarray,
            _conducting: bool,
            _time: float | None = None,
        ) -> float:
            return float(state[0])

    roots = simulator._coincident_natural_roots(
        FastMode(),
        {"d1": True},
        1.0,
        np.array([5e-11]),
    )

    assert roots == [0]
