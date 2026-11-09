# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Retained numerical-evidence matrices for the canonical ideal buck.

The runner in this module is intentionally headless.  It evaluates every load and
numerical configuration declared by a versioned profile, retains every successful
run, and records failures as structured ``fail`` or ``not-run`` evidence.  It does
not turn partial execution into a passing result and does not make a validation or
hardware-correlation claim.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
from typing import Any, Literal, Sequence

import numpy as np
import scipy

from pear_sim import __version__
from pear_sim.benchmarks import (
    BCM_COINCIDENCE_FRACTION_TOLERANCE,
    BUCK_BOUNDARY_ORACLE_SCHEMA,
    DEFAULT_LOAD_RATIOS,
    BuckBoundaryObservation,
    BuckBoundaryPoint,
    compare_buck_boundary_point,
    extract_buck_boundary_parameters,
    observe_simulation_result,
    run_buck_boundary_sweep,
)
from pear_sim.run.contracts import model_hashes, sha256_file, sha256_json
from pear_sim.run.io import write_result
from pear_sim.spec.model import Model, load_model
from pear_sim.reporting.numerical_report import generate_numerical_report
from pear_sim.core.solver import HybridSimulator
from pear_sim._jsonutil import write_json as _write_json
from pear_sim.evidence.sweep import periodic_point_model


NUMERICAL_EVIDENCE_SCHEMA = "pear-sim/platform-evidence/canonical-buck-numerical/v0.1"
NUMERICAL_PROFILE_SCHEMA = "pear-sim/numerical-evidence-profile/v0.1"
NUMERICAL_CELL_SCHEMA = "pear-sim/numerical-evidence-cell/v0.1"
CROSS_PLATFORM_POLICY_SCHEMA = "pear-sim/cross-platform-numerical-policy/v0.1"
SEMANTIC_SIGNATURE_SCHEMA = "pear-sim/numerical-semantic-signature/canonical-buck/v0.1"
FULL_PROFILE_ID = "full-v1"
MAX_NUMERICAL_CELLS = 256

Status = Literal["pass", "fail", "not-run"]


@dataclass(frozen=True)
class NumericalConfiguration:
    """One fully resolved transient configuration in a numerical study."""

    id: str
    studies: tuple[str, ...]
    solver: str
    rtol: float
    atol: float
    max_step_over_period: float
    output_step_over_period: float

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", self.id):
            raise ValueError(f"invalid numerical configuration id {self.id!r}")
        if not self.studies or len(set(self.studies)) != len(self.studies):
            raise ValueError(f"configuration {self.id!r} must name unique studies")
        if self.solver not in {"dopri5", "dop853", "radau", "bdf"}:
            raise ValueError(f"unsupported numerical-evidence solver {self.solver!r}")
        values = {
            "rtol": self.rtol,
            "atol": self.atol,
            "max_step_over_period": self.max_step_over_period,
            "output_step_over_period": self.output_step_over_period,
        }
        for name, value in values.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"configuration {self.id!r} has invalid {name}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "studies": list(self.studies),
            "solver": self.solver,
            "rtol": self.rtol,
            "atol": self.atol,
            "max_step_over_period": self.max_step_over_period,
            "output_step_over_period": self.output_step_over_period,
        }


@dataclass(frozen=True)
class NumericalEvidenceProfile:
    """A bounded, immutable definition of a numerical evidence matrix."""

    id: str
    load_ratios: tuple[float, ...]
    configurations: tuple[NumericalConfiguration, ...]
    study_configuration_ids: dict[str, tuple[str, ...]]
    cross_platform_policy: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("numerical evidence profile id must not be empty")
        if not self.load_ratios:
            raise ValueError("numerical evidence profile must contain load ratios")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.load_ratios):
            raise ValueError("numerical evidence load ratios must be finite and positive")
        if any(
            right <= left
            for left, right in zip(self.load_ratios[:-1], self.load_ratios[1:], strict=True)
        ):
            raise ValueError("numerical evidence load ratios must be strictly increasing")
        ids = [configuration.id for configuration in self.configurations]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("numerical evidence configuration ids must be unique")
        expected_studies = {
            study for configuration in self.configurations for study in configuration.studies
        }
        if set(self.study_configuration_ids) != expected_studies:
            raise ValueError("study index does not match configuration study membership")
        known_ids = set(ids)
        for study, indexed_ids in self.study_configuration_ids.items():
            if not indexed_ids or len(indexed_ids) != len(set(indexed_ids)):
                raise ValueError(f"study {study!r} must index unique configurations")
            if not set(indexed_ids) <= known_ids:
                raise ValueError(f"study {study!r} references an unknown configuration")
            actual_ids = {
                configuration.id
                for configuration in self.configurations
                if study in configuration.studies
            }
            if set(indexed_ids) != actual_ids:
                raise ValueError(f"study {study!r} index differs from membership")
        if len(self.load_ratios) * len(self.configurations) > MAX_NUMERICAL_CELLS:
            raise ValueError(
                f"profile exceeds the {MAX_NUMERICAL_CELLS}-cell numerical evidence limit"
            )

    @property
    def cell_count(self) -> int:
        return len(self.load_ratios) * len(self.configurations)

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": NUMERICAL_PROFILE_SCHEMA,
            "id": self.id,
            "load_ratios": list(self.load_ratios),
            "point_count": len(self.load_ratios),
            "configuration_count": len(self.configurations),
            "cell_count": self.cell_count,
            "studies": {
                study: list(configuration_ids)
                for study, configuration_ids in sorted(self.study_configuration_ids.items())
            },
            "configurations": [configuration.as_dict() for configuration in self.configurations],
            "cross_platform_policy": self.cross_platform_policy,
        }
        return {**payload, "sha256": sha256_json(payload)}


@dataclass(frozen=True)
class NumericalEvidenceArtifacts:
    output_dir: Path
    profile_json: Path
    oracle_json: Path
    platform_evidence: Path
    artifact_manifest: Path
    report_json: Path
    report_html: Path
    status: Status


def _cross_platform_policy() -> dict[str, Any]:
    """Return the profile-bound policy for signed semantic field comparison."""

    field_policy_ids = {
        "regime": "exact",
        "state.cycle_start.inductor_current_a": "boundary_state_current",
        "state.cycle_start.output_voltage_v": "boundary_state_voltage",
        "state.switch_off.inductor_current_a": "boundary_state_current",
        "state.switch_off.output_voltage_v": "boundary_state_voltage",
        "state.cycle_end.inductor_current_a": "boundary_state_current",
        "state.cycle_end.output_voltage_v": "boundary_state_voltage",
        "events.diode_zero.present": "exact",
        "events.diode_zero.time_s": "event_time",
        "metrics.mean_output_voltage_v": "cycle_mean_output",
        "metrics.minimum_inductor_current_a": "minimum_current",
        "modes.fraction.on": "mode_fraction",
        "modes.fraction.freewheel": "mode_fraction",
        "modes.fraction.idle": "mode_fraction",
        "modes.fraction.unclassified": "mode_fraction",
        "physics.balance_residual_relative_signed": "energy_balance",
        "cycle.delta_inductor_current_a": "cycle_delta_current",
        "cycle.delta_output_voltage_v": "cycle_delta_voltage",
        "events.semantic_sequence_sha256": "exact",
        "events.count": "exact",
        "modes.topology_sequence_sha256": "exact",
        "modes.interval_count": "exact",
    }
    return {
        "schema": CROSS_PLATFORM_POLICY_SCHEMA,
        "signature_schema": SEMANTIC_SIGNATURE_SCHEMA,
        "comparison": (
            "Match cell and flat semantic field ids; require exact categorical equality, "
            "otherwise accept absolute OR relative agreement of signed physical values. "
            "Every source platform must first pass its independent oracle gates."
        ),
        "field_tolerances": "embedded-and-profile-bound-per-semantic-field",
        "numeric_formula": "max(value)-min(value) <= max(absolute, relative*max(abs(value)))",
        "exact_formula": "canonical JSON scalar values are identical",
        "semantic_signature": {
            "schema": SEMANTIC_SIGNATURE_SCHEMA,
            "required_field_count": len(field_policy_ids),
            "field_policy_ids": dict(sorted(field_policy_ids.items())),
            "tolerance_policies": {
                "exact": {"kind": "exact"},
                "boundary_state_current": {
                    "kind": "numeric",
                    "absolute_rule": "2e-7*abs(oracle) when nonzero else 2e-10*I_base",
                    "relative": 1e-6,
                },
                "boundary_state_voltage": {
                    "kind": "numeric",
                    "absolute_rule": "2e-7*abs(oracle) when nonzero else 2e-10*Vin",
                    "relative": 1e-6,
                },
                "event_time": {
                    "kind": "numeric",
                    "absolute_rule": "1e-5*T_switch",
                    "relative": 1e-6,
                },
                "cycle_mean_output": {
                    "kind": "numeric",
                    "absolute_rule": "2e-5*abs(oracle_cycle_mean_output)",
                    "relative": 1e-6,
                },
                "minimum_current": {
                    "kind": "numeric",
                    "absolute_rule": "2e-10*I_base",
                    "relative": 1e-6,
                },
                "mode_fraction": {
                    "kind": "numeric",
                    "absolute": 2e-5,
                    "relative": 1e-6,
                },
                "energy_balance": {
                    "kind": "numeric",
                    "absolute": 2e-6,
                    "relative": 1e-6,
                },
                "cycle_delta_current": {
                    "kind": "numeric",
                    "absolute_rule": "2e-10*I_base",
                    "relative": 1e-6,
                },
                "cycle_delta_voltage": {
                    "kind": "numeric",
                    "absolute_rule": "2e-10*Vin",
                    "relative": 1e-6,
                },
            },
        },
    }


def full_v1_profile() -> NumericalEvidenceProfile:
    """Return the complete 9-by-16 canonical-buck numerical evidence profile.

    Each study changes one numerical dimension while holding the others fixed.  The
    shared DOPRI5 configuration belongs to both tolerance and output-grid studies,
    so the declared 17 study memberships deduplicate to 16 executed configurations.
    """

    nominal = NumericalConfiguration(
        id="reference-tolerance-output-grid",
        studies=("tolerance", "output_grid"),
        solver="dopri5",
        rtol=1e-10,
        atol=1e-12,
        max_step_over_period=1.0 / 80.0,
        output_step_over_period=0.005,
    )
    configurations = (
        NumericalConfiguration(
            "tolerance-1e-8",
            ("tolerance",),
            "dopri5",
            1e-8,
            1e-10,
            1.0 / 80.0,
            0.005,
        ),
        NumericalConfiguration(
            "tolerance-1e-9",
            ("tolerance",),
            "dopri5",
            1e-9,
            1e-11,
            1.0 / 80.0,
            0.005,
        ),
        nominal,
        NumericalConfiguration(
            "tolerance-1e-11",
            ("tolerance",),
            "dopri5",
            1e-11,
            1e-13,
            1.0 / 80.0,
            0.005,
        ),
        NumericalConfiguration(
            "tolerance-1e-12",
            ("tolerance",),
            "dopri5",
            1e-12,
            1e-14,
            1.0 / 80.0,
            0.005,
        ),
        NumericalConfiguration(
            "max-step-t-over-40",
            ("max_step",),
            "dopri5",
            1e-10,
            1e-12,
            1.0 / 40.0,
            1.0 / 37.0,
        ),
        NumericalConfiguration(
            "max-step-t-over-60",
            ("max_step",),
            "dopri5",
            1e-10,
            1e-12,
            1.0 / 60.0,
            1.0 / 37.0,
        ),
        NumericalConfiguration(
            "max-step-t-over-80",
            ("max_step",),
            "dopri5",
            1e-10,
            1e-12,
            1.0 / 80.0,
            1.0 / 37.0,
        ),
        NumericalConfiguration(
            "max-step-t-over-120",
            ("max_step",),
            "dopri5",
            1e-10,
            1e-12,
            1.0 / 120.0,
            1.0 / 37.0,
        ),
        NumericalConfiguration(
            "max-step-t-over-160",
            ("max_step",),
            "dopri5",
            1e-10,
            1e-12,
            1.0 / 160.0,
            1.0 / 37.0,
        ),
        NumericalConfiguration(
            "output-grid-t-over-13",
            ("output_grid",),
            "dopri5",
            1e-10,
            1e-12,
            1.0 / 80.0,
            1.0 / 13.0,
        ),
        NumericalConfiguration(
            "output-grid-t-over-37",
            ("output_grid",),
            "dopri5",
            1e-10,
            1e-12,
            1.0 / 80.0,
            1.0 / 37.0,
        ),
        NumericalConfiguration(
            "solver-dopri5",
            ("solver",),
            "dopri5",
            1e-10,
            1e-12,
            0.01,
            0.005,
        ),
        NumericalConfiguration(
            "solver-dop853",
            ("solver",),
            "dop853",
            1e-10,
            1e-12,
            0.01,
            0.005,
        ),
        NumericalConfiguration(
            "solver-radau",
            ("solver",),
            "radau",
            1e-10,
            1e-12,
            0.01,
            0.005,
        ),
        NumericalConfiguration(
            "solver-bdf",
            ("solver",),
            "bdf",
            1e-10,
            1e-12,
            0.01,
            0.005,
        ),
    )
    studies = {
        "tolerance": (
            "tolerance-1e-8",
            "tolerance-1e-9",
            "reference-tolerance-output-grid",
            "tolerance-1e-11",
            "tolerance-1e-12",
        ),
        "max_step": (
            "max-step-t-over-40",
            "max-step-t-over-60",
            "max-step-t-over-80",
            "max-step-t-over-120",
            "max-step-t-over-160",
        ),
        "output_grid": (
            "output-grid-t-over-13",
            "output-grid-t-over-37",
            "reference-tolerance-output-grid",
        ),
        "solver": (
            "solver-dopri5",
            "solver-dop853",
            "solver-radau",
            "solver-bdf",
        ),
    }
    return NumericalEvidenceProfile(
        id=FULL_PROFILE_ID,
        load_ratios=tuple(DEFAULT_LOAD_RATIOS),
        configurations=configurations,
        study_configuration_ids=studies,
        cross_platform_policy=_cross_platform_policy(),
    )


def _prepare_destination(output_dir: str | Path) -> Path:
    destination = Path(output_dir).resolve()
    if destination.exists():
        if not destination.is_dir():
            raise ValueError(f"Numerical evidence output is not a directory: {destination}")
        if any(destination.iterdir()):
            raise ValueError(
                f"Numerical evidence output directory is not empty: {destination}. "
                "Choose a new directory so prior evidence is never overwritten."
            )
    else:
        destination.mkdir(parents=True)
    return destination


def _status(statuses: Sequence[str]) -> Status:
    if not statuses:
        return "not-run"
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status != "pass" for status in statuses):
        return "not-run"
    return "pass"


def _not_run_gates(reference: BuckBoundaryPoint, reason: str) -> list[dict[str, Any]]:
    gate_ids = (
        "regime",
        "boundary_state_nonzero",
        "boundary_state_zero",
        "natural_event_time",
        "reverse_current",
        "cycle_mean_output",
        "mode_fractions",
        "energy_balance",
        "cycle_periodicity",
        "unclassified_fraction",
        "mode_fraction_sum",
    )
    return [
        {
            "id": gate_id,
            "actual": None,
            "reference": None,
            "limit": None,
            "status": "not-run",
            "reason": reason,
            "expected_regime": reference.regime,
        }
        for gate_id in gate_ids
    ]


def _numeric_signature_field(
    value: float,
    *,
    quantity: str,
    unit: str,
    policy_id: str,
    absolute_tolerance: float,
    relative_tolerance: float = 1e-6,
) -> dict[str, Any]:
    values = (value, absolute_tolerance, relative_tolerance)
    if any(not math.isfinite(item) for item in values):
        raise ValueError(f"semantic signature field {quantity!r} must be finite")
    if absolute_tolerance < 0.0 or relative_tolerance < 0.0:
        raise ValueError(f"semantic signature field {quantity!r} has a negative tolerance")
    return {
        "quantity": quantity,
        "unit": unit,
        "policy_id": policy_id,
        "value": float(value),
        "cross_platform_tolerance": {
            "kind": "numeric",
            "policy_id": policy_id,
            "absolute": float(absolute_tolerance),
            "relative": float(relative_tolerance),
        },
    }


def _exact_signature_field(
    value: str | int | bool,
    *,
    quantity: str,
    unit: str,
    policy_id: str = "exact",
) -> dict[str, Any]:
    return {
        "quantity": quantity,
        "unit": unit,
        "policy_id": policy_id,
        "value": value,
        "cross_platform_tolerance": {"kind": "exact", "policy_id": policy_id},
    }


def _state_absolute_tolerance(reference: float, scale: float) -> float:
    if abs(reference) <= scale * 1e-10:
        return max(scale * 2e-10, 1e-15)
    return max(abs(reference) * 2e-7, scale * 1e-15)


def _semantic_signature(
    *,
    parameters: Any,
    reference: BuckBoundaryPoint,
    observation: BuckBoundaryObservation,
    result: Any,
) -> dict[str, Any]:
    """Build a flat, typed signature of signed observed physical quantities."""

    fields: dict[str, dict[str, Any]] = {}
    fields["regime"] = _exact_signature_field(
        observation.regime,
        quantity="canonical_buck_operating_regime",
        unit="category",
    )
    state_records = (
        ("cycle_start", observation.start_state, reference.start_state),
        ("switch_off", observation.switch_off_state, reference.switch_off_state),
        ("cycle_end", observation.end_state, reference.end_state),
    )
    for state_name, observed_state, reference_state in state_records:
        fields[f"state.{state_name}.inductor_current_a"] = _numeric_signature_field(
            observed_state.inductor_current_a,
            quantity="inductor_current",
            unit="A",
            policy_id="boundary_state_current",
            absolute_tolerance=_state_absolute_tolerance(
                reference_state.inductor_current_a,
                parameters.current_base_a,
            ),
        )
        fields[f"state.{state_name}.output_voltage_v"] = _numeric_signature_field(
            observed_state.output_voltage_v,
            quantity="output_voltage",
            unit="V",
            policy_id="boundary_state_voltage",
            absolute_tolerance=_state_absolute_tolerance(
                reference_state.output_voltage_v,
                parameters.vin_v,
            ),
        )

    event_present = observation.diode_zero_s is not None
    fields["events.diode_zero.present"] = _exact_signature_field(
        event_present,
        quantity="natural_diode_zero_event_presence",
        unit="boolean",
    )
    fields["events.diode_zero.time_s"] = _numeric_signature_field(
        0.0 if observation.diode_zero_s is None else observation.diode_zero_s,
        quantity="natural_diode_zero_event_time",
        unit="s",
        policy_id="event_time",
        absolute_tolerance=max(parameters.period_s * 1e-5, 1e-15),
    )
    fields["metrics.mean_output_voltage_v"] = _numeric_signature_field(
        observation.mean_output_voltage_v,
        quantity="cycle_mean_output_voltage",
        unit="V",
        policy_id="cycle_mean_output",
        absolute_tolerance=max(
            abs(reference.mean_output_voltage_v) * 2e-5,
            parameters.vin_v * 2e-17,
        ),
    )
    fields["metrics.minimum_inductor_current_a"] = _numeric_signature_field(
        observation.minimum_inductor_current_a,
        quantity="minimum_inductor_current",
        unit="A",
        policy_id="minimum_current",
        absolute_tolerance=max(parameters.current_base_a * 2e-10, 1e-15),
    )
    for mode in ("on", "freewheel", "idle", "unclassified"):
        fields[f"modes.fraction.{mode}"] = _numeric_signature_field(
            observation.fraction_by_mode[mode],
            quantity=f"canonical_mode_fraction_{mode}",
            unit="1",
            policy_id="mode_fraction",
            absolute_tolerance=2e-5,
        )

    audit = getattr(result, "physics_audit", {}).get("full")
    if not isinstance(audit, dict) or "balance_residual_j" not in audit:
        raise ValueError("semantic signature requires the signed full-cycle balance residual")
    signed_balance_relative = math.copysign(
        observation.energy_balance_relative,
        float(audit["balance_residual_j"]),
    )
    if float(audit["balance_residual_j"]) == 0.0:
        signed_balance_relative = 0.0
    fields["physics.balance_residual_relative_signed"] = _numeric_signature_field(
        signed_balance_relative,
        quantity="signed_full_cycle_energy_balance_residual_relative",
        unit="1",
        policy_id="energy_balance",
        absolute_tolerance=2e-6,
    )
    fields["cycle.delta_inductor_current_a"] = _numeric_signature_field(
        observation.end_state.inductor_current_a - observation.start_state.inductor_current_a,
        quantity="cycle_end_minus_start_inductor_current",
        unit="A",
        policy_id="cycle_delta_current",
        absolute_tolerance=max(parameters.current_base_a * 2e-10, 1e-15),
    )
    fields["cycle.delta_output_voltage_v"] = _numeric_signature_field(
        observation.end_state.output_voltage_v - observation.start_state.output_voltage_v,
        quantity="cycle_end_minus_start_output_voltage",
        unit="V",
        policy_id="cycle_delta_voltage",
        absolute_tolerance=max(parameters.vin_v * 2e-10, 1e-15),
    )

    events = [
        {
            "kind": event.kind,
            "target": event.target,
            "previous": event.previous,
            "current": event.current,
        }
        for event in result.events
    ]
    fields["events.semantic_sequence_sha256"] = _exact_signature_field(
        sha256_json(events),
        quantity="semantic_event_sequence_digest",
        unit="sha256",
    )
    fields["events.count"] = _exact_signature_field(
        len(events),
        quantity="semantic_event_count",
        unit="count",
    )
    modes = []
    for interval in result.mode_intervals:
        duration_fraction = (
            float(interval["end_s"]) - float(interval["start_s"])
        ) / parameters.period_s
        if duration_fraction <= BCM_COINCIDENCE_FRACTION_TOLERANCE:
            continue
        modes.append(
            {
                "gates": dict(sorted(interval.get("gates", {}).items())),
                "diodes": dict(sorted(interval.get("diodes", {}).items())),
                "constrained_inductors": sorted(interval.get("constrained_inductors", ())),
            }
        )
    fields["modes.topology_sequence_sha256"] = _exact_signature_field(
        sha256_json(modes),
        quantity="accepted_mode_topology_sequence_digest",
        unit="sha256",
    )
    fields["modes.interval_count"] = _exact_signature_field(
        len(modes),
        quantity="accepted_mode_interval_count",
        unit="count",
    )
    return {
        "schema": SEMANTIC_SIGNATURE_SCHEMA,
        "status": "available",
        "fields": dict(sorted(fields.items())),
    }


def _not_run_semantic_signature(reason: str) -> dict[str, Any]:
    return {
        "schema": SEMANTIC_SIGNATURE_SCHEMA,
        "status": "not-run",
        "fields": {},
        "reason": reason,
    }


def _configured_model(
    source_model: Model,
    reference: BuckBoundaryPoint,
    configuration: NumericalConfiguration,
) -> Model:
    periodic = periodic_point_model(source_model, reference)
    period = next(iter(periodic.controls.values())).period
    return replace(
        periodic,
        name=f"{periodic.name}__{configuration.id}",
        transient=replace(
            periodic.transient,
            solver=configuration.solver,
            rtol=configuration.rtol,
            atol=configuration.atol,
            max_step=period * configuration.max_step_over_period,
            output_step=period * configuration.output_step_over_period,
        ),
    )


def _platform_record() -> dict[str, Any]:
    system = platform.system() or "unknown"
    machine = platform.machine() or "unknown"
    implementation = platform.python_implementation() or "unknown"
    python_version = platform.python_version()
    platform_id = "-".join(
        re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"
        for value in (system, machine, implementation, python_version)
    )
    record: dict[str, Any] = {
        "id": platform_id,
        "system": system,
        "release": platform.release(),
        "machine": machine,
        "python_implementation": implementation,
        "python": python_version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    uv_version = os.environ.get("PEAR_SIM_UV_VERSION", "").strip()
    if uv_version:
        record["toolchain"] = {"uv": uv_version}
    ci_names = (
        "GITHUB_ACTIONS",
        "RUNNER_OS",
        "RUNNER_ARCH",
        "RUNNER_ENVIRONMENT",
        "ImageOS",
        "ImageVersion",
        "GITHUB_WORKFLOW_REF",
        "GITHUB_SHA",
    )
    ci_environment = {
        name: value for name in ci_names if (value := os.environ.get(name, "").strip())
    }
    if ci_environment:
        record["ci_environment"] = ci_environment
    return record


def _source_revision(explicit: str | None, source_path: Path) -> tuple[str, str]:
    if explicit is not None:
        value = explicit.strip()
        if not value or len(value) > 256:
            raise ValueError("source revision must contain 1 to 256 characters")
        return value, "explicit"
    for variable in ("PEAR_SIM_SOURCE_REVISION", "GITHUB_SHA"):
        value = os.environ.get(variable, "").strip()
        if value:
            return value[:256], f"environment:{variable}"
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=source_path.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown", "unavailable"
    value = completed.stdout.strip()
    return (value, "git") if value else ("unknown", "unavailable")


def _cell_summary(
    *,
    reference: BuckBoundaryPoint,
    configuration: NumericalConfiguration,
    comparison: dict[str, Any],
    observation: BuckBoundaryObservation | None,
    runtime_s: float | None,
    run_path: Path | None,
    cell_path: Path,
    destination: Path,
    error: dict[str, str] | None,
    numerics: dict[str, Any] | None,
    semantic_signature: dict[str, Any],
) -> dict[str, Any]:
    run_record: dict[str, Any] | None = None
    if run_path is not None:
        run_manifest_path = run_path / "run.json"
        run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        run_record = {
            "path": run_manifest_path.relative_to(destination).as_posix(),
            "run_sha256": run_manifest["run_sha256"],
            "manifest_sha256": sha256_file(run_manifest_path),
        }
    metrics = (
        None
        if observation is None
        else {
            "mean_output_voltage_v": observation.mean_output_voltage_v,
            "minimum_inductor_current_a": observation.minimum_inductor_current_a,
            "fraction_by_mode": dict(observation.fraction_by_mode),
            "energy_balance_relative": observation.energy_balance_relative,
        }
    )
    return {
        "id": f"{reference.point_id}--{configuration.id}",
        "point_id": reference.point_id,
        "configuration_id": configuration.id,
        "load_ratio_to_bcm": reference.load_ratio_to_bcm,
        "expected_regime": reference.regime,
        "observed_regime": None if observation is None else observation.regime,
        "status": comparison["status"],
        "runtime_s": runtime_s,
        "metrics": metrics,
        "gates": comparison["gates"],
        "semantic_signature": semantic_signature,
        "numerics": numerics,
        "run": run_record,
        "cell_artifact": {
            "path": cell_path.relative_to(destination).as_posix(),
            "sha256": sha256_file(cell_path),
        },
        "error": error,
    }


def _numeric_gate_budget_fractions(cell: dict[str, Any]) -> dict[str, float] | None:
    """Return one normalized value for every numeric oracle gate in a cell.

    Convergence must follow a fixed physical gate across configurations.  Reducing a
    cell to its maximum first would allow the identity of the dominating gate to
    change during refinement and could manufacture a trend that no individual gate
    exhibits.
    """

    gates = cell.get("gates")
    if not isinstance(gates, list):
        return None
    fractions: dict[str, float] = {}
    for gate in gates:
        if not isinstance(gate, dict):
            return None
        limit = gate.get("limit")
        if isinstance(limit, bool) or not isinstance(limit, (int, float)):
            # Categorical gates, such as exact regime matching, do not define a
            # numeric error budget and are covered by the source-cell status.
            continue
        actual = gate.get("actual")
        gate_id = gate.get("id")
        if (
            not isinstance(gate_id, str)
            or not gate_id
            or gate_id in fractions
            or not math.isfinite(float(limit))
            or float(limit) <= 0.0
            or isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
        ):
            return None
        fractions[gate_id] = abs(float(actual)) / float(limit)
    return fractions or None


def _convergence_series_record(
    normalized_values: Sequence[float | None],
    control_values: Sequence[float],
    *,
    slope_limit: float,
    plateau_limit: float,
) -> dict[str, Any]:
    complete = (
        len(normalized_values) >= 2
        and len(normalized_values) == len(control_values)
        and all(
            value is not None and math.isfinite(value) and value >= 0.0
            for value in normalized_values
        )
        and all(math.isfinite(value) and value > 0.0 for value in control_values)
    )
    if complete:
        numeric_values = [float(value) for value in normalized_values if value is not None]
        fitted_slope = float(
            np.polyfit(
                np.log(np.asarray(control_values, dtype=float)),
                np.log(np.maximum(np.asarray(numeric_values, dtype=float), 1e-18)),
                1,
            )[0]
        )
        final_three_max = max(numeric_values[-3:]) if len(numeric_values) >= 3 else None
    else:
        fitted_slope = None
        final_three_max = None
    positive_trend = fitted_slope is not None and fitted_slope >= slope_limit
    final_three_plateau = final_three_max is not None and final_three_max <= plateau_limit
    if not complete:
        status: Status = "not-run"
        reason = "the fixed oracle gate lacks a finite normalized value in one or more cells"
    elif positive_trend or final_three_plateau:
        status = "pass"
        reason = None
    else:
        status = "fail"
        reason = "neither positive convergence trend nor final-three <1% plateau was observed"
    return {
        "status": status,
        "normalized_budget_fractions": list(normalized_values),
        "positive_trend": positive_trend,
        "fitted_log_log_slope": fitted_slope,
        "slope_limit": slope_limit,
        "final_three_plateau": final_three_plateau,
        "actual": final_three_max,
        "limit": plateau_limit,
        "reason": reason,
    }


def _signature_fields(cell: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    signature = cell.get("semantic_signature")
    if (
        not isinstance(signature, dict)
        or signature.get("schema") != SEMANTIC_SIGNATURE_SCHEMA
        or signature.get("status") != "available"
        or not isinstance(signature.get("fields"), dict)
    ):
        return None
    fields = signature["fields"]
    if not all(
        isinstance(field_id, str) and isinstance(record, dict)
        for field_id, record in fields.items()
    ):
        return None
    return fields


def _exact_signature_stability(
    cells: Sequence[dict[str, Any]],
    field_ids: Sequence[str],
) -> tuple[bool, dict[str, list[Any]]]:
    values_by_field: dict[str, list[Any]] = {}
    for field_id in field_ids:
        values: list[Any] = []
        for cell in cells:
            fields = _signature_fields(cell)
            if fields is None or field_id not in fields:
                values.append(None)
                continue
            record = fields[field_id]
            tolerance = record.get("cross_platform_tolerance")
            if not isinstance(tolerance, dict) or tolerance.get("kind") != "exact":
                values.append(None)
                continue
            values.append(record.get("value"))
        values_by_field[field_id] = values
    stable = all(
        bool(values)
        and values[0] is not None
        and all(sha256_json(value) == sha256_json(values[0]) for value in values[1:])
        for values in values_by_field.values()
    )
    return stable, values_by_field


def _convergence_gate(
    study: str,
    point_id: str,
    ordered_cells: Sequence[dict[str, Any]],
    configurations: dict[str, NumericalConfiguration],
) -> dict[str, Any]:
    statuses = [cell["status"] for cell in ordered_cells]
    if any(status == "fail" for status in statuses):
        source_status: Status = "fail"
        source_reason = "one or more source cells failed an independent oracle gate"
    elif any(status != "pass" for status in statuses):
        source_status = "not-run"
        source_reason = "one or more source cells did not complete"
    else:
        source_status = "pass"
        source_reason = None

    plateau_limit = 0.01
    if study == "tolerance":
        control_parameter = "rtol"
        control_values = [
            configurations[str(cell["configuration_id"])].rtol for cell in ordered_cells
        ]
    else:
        control_parameter = "max_step_over_period"
        control_values = [
            configurations[str(cell["configuration_id"])].max_step_over_period
            for cell in ordered_cells
        ]
    slope_limit = 0.05
    fractions_by_cell = [_numeric_gate_budget_fractions(cell) for cell in ordered_cells]
    numeric_gate_ids = sorted(
        {
            gate_id
            for fractions in fractions_by_cell
            if fractions is not None
            for gate_id in fractions
        }
    )
    fixed_gate_set = bool(numeric_gate_ids) and all(
        fractions is not None and set(fractions) == set(numeric_gate_ids)
        for fractions in fractions_by_cell
    )
    oracle_gate_series: dict[str, dict[str, Any]] = {}
    for gate_id in numeric_gate_ids:
        normalized_values = [
            None if fractions is None else fractions.get(gate_id) for fractions in fractions_by_cell
        ]
        oracle_gate_series[gate_id] = _convergence_series_record(
            normalized_values,
            control_values,
            slope_limit=slope_limit,
            plateau_limit=plateau_limit,
        )
    series_status_counts = Counter(series["status"] for series in oracle_gate_series.values())

    if fixed_gate_set:
        worst_scores = [
            max(fractions.values()) for fractions in fractions_by_cell if fractions is not None
        ]
        worst_envelope = _convergence_series_record(
            worst_scores,
            control_values,
            slope_limit=slope_limit,
            plateau_limit=plateau_limit,
        )
    else:
        worst_scores = []
        worst_envelope = _convergence_series_record(
            [],
            control_values,
            slope_limit=slope_limit,
            plateau_limit=plateau_limit,
        )
    worst_envelope["acceptance_role"] = "diagnostic-only"

    event_sequence_stable, event_signature_values = _exact_signature_stability(
        ordered_cells,
        ("events.semantic_sequence_sha256", "events.count"),
    )
    mode_sequence_stable, mode_signature_values = _exact_signature_stability(
        ordered_cells,
        ("modes.topology_sequence_sha256", "modes.interval_count"),
    )
    if source_status != "pass":
        status = source_status
        reason = source_reason
    elif not fixed_gate_set:
        status = "not-run"
        reason = "source cells do not expose one complete, fixed set of numeric oracle gates"
    else:
        failed_series = sorted(
            gate_id for gate_id, series in oracle_gate_series.items() if series["status"] == "fail"
        )
        incomplete_series = sorted(
            gate_id
            for gate_id, series in oracle_gate_series.items()
            if series["status"] != "pass" and gate_id not in failed_series
        )
        if failed_series:
            status = "fail"
            reason = "fixed oracle gate series failed convergence: " + ", ".join(failed_series)
        elif incomplete_series:
            status = "not-run"
            reason = "fixed oracle gate series did not complete: " + ", ".join(incomplete_series)
        elif not (event_sequence_stable and mode_sequence_stable):
            status = "fail"
            reason = "semantic event or accepted-mode sequence changed during refinement"
        else:
            status = "pass"
            reason = None
    return {
        "id": f"{study}:{point_id}:convergence",
        "point_id": point_id,
        "kind": "per-oracle-gate-positive-trend-or-final-three-plateau",
        "status": status,
        "configuration_ids": [cell["configuration_id"] for cell in ordered_cells],
        "control_parameter": control_parameter,
        "control_values": control_values,
        "numeric_oracle_gate_ids": numeric_gate_ids,
        "fixed_numeric_gate_set": fixed_gate_set,
        "oracle_gate_series_status_counts": dict(sorted(series_status_counts.items())),
        "oracle_gate_series": oracle_gate_series,
        "worst_envelope_summary": worst_envelope,
        "stability": {
            "event_sequence": {
                "status": "pass" if event_sequence_stable else "fail",
                "values": event_signature_values,
            },
            "mode_sequence": {
                "status": "pass" if mode_sequence_stable else "fail",
                "values": mode_signature_values,
            },
        },
        "reason": reason,
    }


def _semantic_sequences_equal(cells: Sequence[dict[str, Any]]) -> bool:
    sequences = [
        cell.get("numerics", {}).get("semantic_event_sequence")
        if isinstance(cell.get("numerics"), dict)
        else None
        for cell in cells
    ]
    return (
        bool(sequences)
        and sequences[0] is not None
        and all(sequence == sequences[0] for sequence in sequences[1:])
    )


def _value_range(values: Sequence[float]) -> float:
    return max(values) - min(values)


def _output_grid_gate(
    point_id: str,
    ordered_cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    statuses = [cell["status"] for cell in ordered_cells]
    if any(status == "fail" for status in statuses):
        status: Status = "fail"
        reason = "one or more source cells failed an independent oracle gate"
    elif any(status != "pass" for status in statuses):
        status = "not-run"
        reason = "one or more source cells did not complete"
    else:
        status = "pass"
        reason = None

    complete = all(
        isinstance(cell.get("numerics"), dict) and isinstance(cell.get("metrics"), dict)
        for cell in ordered_cells
    )
    sample_counts = (
        [int(cell["numerics"]["sample_count"]) for cell in ordered_cells] if complete else []
    )
    distinct_sample_counts = len(sample_counts) == len(set(sample_counts)) == len(ordered_cells)
    event_sequence_equal = _semantic_sequences_equal(ordered_cells) if complete else False
    mode_sequence_equal, mode_signature_values = _exact_signature_stability(
        ordered_cells,
        ("modes.topology_sequence_sha256", "modes.interval_count"),
    )
    if complete:
        means = [float(cell["metrics"]["mean_output_voltage_v"]) for cell in ordered_cells]
        mean_scale = max(max(abs(value) for value in means), 1e-300)
        mean_relative_range = _value_range(means) / mean_scale
        mode_fraction_range = max(
            _value_range(
                [float(cell["metrics"]["fraction_by_mode"][mode]) for cell in ordered_cells]
            )
            for mode in ("on", "freewheel", "idle", "unclassified")
        )
        energy_range = _value_range(
            [float(cell["metrics"]["energy_balance_relative"]) for cell in ordered_cells]
        )
    else:
        mean_relative_range = None
        mode_fraction_range = None
        energy_range = None
        if status == "pass":
            status = "not-run"
            reason = "source cells lack numerical or time-integral records"

    checks = {
        "distinct_sample_counts": {
            "actual": len(set(sample_counts)),
            "expected": len(ordered_cells),
            "status": "pass" if distinct_sample_counts else "fail",
        },
        "semantic_event_sequence": {
            "actual": event_sequence_equal,
            "expected": True,
            "status": "pass" if event_sequence_equal else "fail",
        },
        "accepted_mode_sequence": {
            "actual": mode_sequence_equal,
            "expected": True,
            "values": mode_signature_values,
            "status": "pass" if mode_sequence_equal else "fail",
        },
        "mean_output_relative_range": {
            "actual": mean_relative_range,
            "limit": 2e-5,
            "status": (
                "not-run"
                if mean_relative_range is None
                else "pass"
                if mean_relative_range <= 2e-5
                else "fail"
            ),
        },
        "mode_fraction_range": {
            "actual": mode_fraction_range,
            "limit": 2e-5,
            "status": (
                "not-run"
                if mode_fraction_range is None
                else "pass"
                if mode_fraction_range <= 2e-5
                else "fail"
            ),
        },
        "energy_balance_range": {
            "actual": energy_range,
            "limit": 2e-6,
            "status": (
                "not-run" if energy_range is None else "pass" if energy_range <= 2e-6 else "fail"
            ),
        },
    }
    if status == "pass":
        status = _status([str(check["status"]) for check in checks.values()])
        if status != "pass":
            reason = "output-grid independence checks did not all pass"
    return {
        "id": f"output_grid:{point_id}:independence",
        "point_id": point_id,
        "kind": "output-grid-independence",
        "status": status,
        "configuration_ids": [cell["configuration_id"] for cell in ordered_cells],
        "sample_counts": sample_counts,
        "checks": checks,
        "reason": reason,
    }


def _solver_gate(
    point_id: str,
    ordered_cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    statuses = [cell["status"] for cell in ordered_cells]
    if any(status == "fail" for status in statuses):
        status: Status = "fail"
        reason = "one or more source cells failed an independent oracle gate"
    elif any(status != "pass" for status in statuses):
        status = "not-run"
        reason = "one or more source cells did not complete"
    else:
        status = "pass"
        reason = None
    signatures = [_signature_fields(cell) for cell in ordered_cells]
    field_ranges: dict[str, Any] = {}
    if any(fields is None for fields in signatures):
        if status == "pass":
            status = "not-run"
            reason = "one or more source cells lacks an available semantic signature"
        field_ids: tuple[str, ...] = ()
    else:
        assert all(fields is not None for fields in signatures)
        field_sets = {frozenset(fields) for fields in signatures if fields is not None}
        if len(field_sets) != 1:
            if status == "pass":
                status = "not-run"
                reason = "semantic signature field IDs differ between solver cells"
            field_ids = ()
        else:
            field_ids = tuple(sorted(next(iter(field_sets))))

    for field_id in field_ids:
        records = [fields[field_id] for fields in signatures if fields is not None]
        quantities = {record.get("quantity") for record in records}
        units = {record.get("unit") for record in records}
        policy_ids = {record.get("policy_id") for record in records}
        tolerances = [record.get("cross_platform_tolerance") for record in records]
        if (
            len(quantities) != 1
            or len(units) != 1
            or len(policy_ids) != 1
            or any(not isinstance(tolerance, dict) for tolerance in tolerances)
            or any(tolerance != tolerances[0] for tolerance in tolerances[1:])
        ):
            field_ranges[field_id] = {
                "status": "not-run",
                "reason": "field type, unit, policy, or tolerance differs between solvers",
            }
            continue
        tolerance = tolerances[0]
        assert isinstance(tolerance, dict)
        values = [record.get("value") for record in records]
        if tolerance.get("kind") == "exact":
            passed = (
                bool(values)
                and values[0] is not None
                and all(sha256_json(value) == sha256_json(values[0]) for value in values[1:])
            )
            field_ranges[field_id] = {
                "kind": "exact",
                "quantity": next(iter(quantities)),
                "unit": next(iter(units)),
                "values": values,
                "status": "pass" if passed else "fail",
            }
            continue
        if tolerance.get("kind") != "numeric" or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in values
        ):
            field_ranges[field_id] = {
                "kind": tolerance.get("kind"),
                "values": values,
                "status": "not-run",
            }
            continue
        absolute = tolerance.get("absolute")
        relative = tolerance.get("relative")
        if (
            not isinstance(absolute, (int, float))
            or isinstance(absolute, bool)
            or not math.isfinite(float(absolute))
            or float(absolute) < 0.0
            or not isinstance(relative, (int, float))
            or isinstance(relative, bool)
            or not math.isfinite(float(relative))
            or float(relative) < 0.0
        ):
            field_ranges[field_id] = {
                "kind": "numeric",
                "values": values,
                "status": "not-run",
                "reason": "invalid embedded numeric tolerance",
            }
            continue
        numeric_values = [float(value) for value in values]
        observed_range = _value_range(numeric_values)
        allowed = max(
            float(absolute) * 2.5,
            float(relative) * max(max(abs(value) for value in numeric_values), 1e-300),
        )
        field_ranges[field_id] = {
            "kind": "numeric",
            "quantity": next(iter(quantities)),
            "unit": next(iter(units)),
            "policy_id": next(iter(policy_ids)),
            "values": numeric_values,
            "actual_range": observed_range,
            "limit": allowed,
            "cross_solver_multiplier_over_cross_platform_absolute": 2.5,
            "status": "pass" if observed_range <= allowed else "fail",
        }
    event_sequence_equal, event_signature_values = _exact_signature_stability(
        ordered_cells,
        ("events.semantic_sequence_sha256", "events.count"),
    )
    mode_sequence_equal, mode_signature_values = _exact_signature_stability(
        ordered_cells,
        ("modes.topology_sequence_sha256", "modes.interval_count"),
    )
    checks = {
        "semantic_event_sequence": {
            "actual": event_sequence_equal,
            "expected": True,
            "values": event_signature_values,
            "status": "pass" if event_sequence_equal else "fail",
        },
        "accepted_mode_sequence": {
            "actual": mode_sequence_equal,
            "expected": True,
            "values": mode_signature_values,
            "status": "pass" if mode_sequence_equal else "fail",
        },
        "recorded_semantic_signature": {
            "status": _status([str(record["status"]) for record in field_ranges.values()])
        },
    }
    if status == "pass":
        status = _status([str(check["status"]) for check in checks.values()])
        if status != "pass":
            reason = "solver event/mode sequence or signed semantic-value consistency failed"
    return {
        "id": f"solver:{point_id}:consistency",
        "point_id": point_id,
        "kind": "solver-consistency",
        "policy": {
            "kind": "cross-solver-oracle-budget-fraction",
            "oracle_budget_fraction": 0.05,
            "distinct_from_cross_platform_policy": True,
            "statement": (
                "Different integration methods may differ by 2.5 times each signed field's "
                "2%-of-oracle-budget cross-platform absolute tolerance, equivalent to the "
                "separately declared 5% cross-solver envelope."
            ),
        },
        "status": status,
        "configuration_ids": [cell["configuration_id"] for cell in ordered_cells],
        "checks": checks,
        "semantic_signature_schema": SEMANTIC_SIGNATURE_SCHEMA,
        "signature_field_ranges": field_ranges,
        "reason": reason,
    }


def _study_summaries(
    profile: NumericalEvidenceProfile,
    cells: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summaries: dict[str, Any] = {}
    point_ids = list(dict.fromkeys(str(cell["point_id"]) for cell in cells))
    configurations = {configuration.id: configuration for configuration in profile.configurations}
    for study, configuration_ids in sorted(profile.study_configuration_ids.items()):
        identifiers = set(configuration_ids)
        selected = [cell for cell in cells if cell["configuration_id"] in identifiers]
        by_key = {(str(cell["point_id"]), str(cell["configuration_id"])): cell for cell in selected}
        family_gates: list[dict[str, Any]] = []
        for point_id in point_ids:
            ordered_cells = [
                by_key[(point_id, configuration_id)]
                for configuration_id in configuration_ids
                if (point_id, configuration_id) in by_key
            ]
            if len(ordered_cells) != len(configuration_ids):
                family_gates.append(
                    {
                        "id": f"{study}:{point_id}:matrix-completeness",
                        "point_id": point_id,
                        "kind": "matrix-completeness",
                        "status": "not-run",
                        "reason": "one or more declared study cells are missing",
                    }
                )
            elif study in {"tolerance", "max_step"}:
                family_gates.append(
                    _convergence_gate(
                        study,
                        point_id,
                        ordered_cells,
                        configurations,
                    )
                )
            elif study == "output_grid":
                family_gates.append(_output_grid_gate(point_id, ordered_cells))
            elif study == "solver":
                family_gates.append(
                    _solver_gate(
                        point_id,
                        ordered_cells,
                    )
                )
            else:
                family_gates.append(
                    {
                        "id": f"{study}:{point_id}:unsupported",
                        "point_id": point_id,
                        "kind": "unsupported-study",
                        "status": "not-run",
                        "reason": f"no family gate evaluator for {study!r}",
                    }
                )

        cell_statuses = Counter(cell["status"] for cell in selected)
        oracle_gate_statuses = Counter(
            gate["status"] for cell in selected for gate in cell["gates"]
        )
        family_gate_statuses = Counter(gate["status"] for gate in family_gates)
        study_status = _status(
            [cell["status"] for cell in selected] + [gate["status"] for gate in family_gates]
        )
        summaries[study] = {
            "status": study_status,
            "configuration_ids": list(configuration_ids),
            "configuration_count": len(configuration_ids),
            "cell_count": len(selected),
            "cell_status_counts": dict(sorted(cell_statuses.items())),
            "oracle_gate_status_counts": dict(sorted(oracle_gate_statuses.items())),
            "family_gate_status_counts": dict(sorted(family_gate_statuses.items())),
            "gates": family_gates,
        }
    return summaries


def _root_manifest(destination: Path, evidence_path: Path) -> Path:
    manifest_path = destination / "artifact-manifest.json"
    artifacts = []
    for path in sorted(destination.rglob("*")):
        if (
            not path.is_file()
            or path == manifest_path
            or path.is_symlink()
            or path.name.startswith(".")
        ):
            continue
        artifacts.append(
            {
                "path": path.relative_to(destination).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    _write_json(
        manifest_path,
        {
            "schema": "pear-sim/artifact-manifest/v0.1",
            "subject": {
                "kind": "canonical-buck-numerical-evidence",
                "sha256": sha256_file(evidence_path),
            },
            "artifacts": artifacts,
        },
    )
    return manifest_path


def run_buck_numerical_evidence(
    model_file: str | Path,
    output_dir: str | Path,
    *,
    profile: NumericalEvidenceProfile | None = None,
    source_revision: str | None = None,
) -> NumericalEvidenceArtifacts:
    """Execute and retain a complete canonical-buck numerical evidence profile."""

    selected_profile = full_v1_profile() if profile is None else profile
    if not isinstance(selected_profile, NumericalEvidenceProfile):
        raise TypeError("profile must be a NumericalEvidenceProfile")
    source_model = load_model(model_file)
    parameters = extract_buck_boundary_parameters(source_model)
    oracle = run_buck_boundary_sweep(parameters, selected_profile.load_ratios)
    revision, revision_method = _source_revision(source_revision, source_model.source_path)
    destination = _prepare_destination(output_dir)
    cells_root = destination / "cells"
    cells_root.mkdir()

    profile_payload = selected_profile.as_dict()
    profile_path = destination / "numerical-profile.json"
    oracle_path = destination / "oracle.json"
    evidence_path = destination / "platform-evidence.json"
    _write_json(profile_path, profile_payload)
    _write_json(oracle_path, oracle.as_dict())

    cells: list[dict[str, Any]] = []
    for configuration_index, configuration in enumerate(selected_profile.configurations):
        configuration_dir = cells_root / f"{configuration_index:02d}-{configuration.id}"
        configuration_dir.mkdir()
        for point_index, reference in enumerate(oracle.points):
            cell_dir = configuration_dir / f"{point_index:03d}-{reference.point_id}"
            cell_dir.mkdir()
            observation: BuckBoundaryObservation | None = None
            runtime_s: float | None = None
            run_path: Path | None = None
            error: dict[str, str] | None = None
            numerics: dict[str, Any] | None = None
            semantic_signature = _not_run_semantic_signature("cell did not complete")
            try:
                point_model = _configured_model(source_model, reference, configuration)
                result = HybridSimulator(point_model).simulate()
                runtime_s = float(result.stats["wall_time_s"])
                period = parameters.period_s
                numerics = {
                    "sample_count": len(result.time),
                    "integration_steps": int(result.stats["integration_steps"]),
                    "semantic_event_sequence": [
                        {
                            "kind": event.kind,
                            "target": event.target,
                            "previous": event.previous,
                            "current": event.current,
                        }
                        for event in result.events
                    ],
                    "event_times_over_period": [
                        (float(event.time) - point_model.transient.start) / period
                        for event in result.events
                    ],
                }
                observation = observe_simulation_result(
                    result,
                    reference,
                    parameters=parameters,
                    bcm_load_ohm=oracle.bcm_load_ohm,
                )
                comparison = compare_buck_boundary_point(parameters, reference, observation)
                semantic_signature = _semantic_signature(
                    parameters=parameters,
                    reference=reference,
                    observation=observation,
                    result=result,
                )
                run_path = write_result(result, cell_dir / "run")
            except Exception as exc:  # every failed cell remains visible and cannot pass
                error = {"type": type(exc).__name__, "message": str(exc)}
                reason = f"{error['type']}: {error['message']}"
                comparison = {
                    "point_id": reference.point_id,
                    "load_ratio_to_bcm": reference.load_ratio_to_bcm,
                    "load_ohm": reference.load_ohm,
                    "expected_regime": reference.regime,
                    "status": "not-run",
                    "gates": _not_run_gates(reference, reason),
                }
                semantic_signature = _not_run_semantic_signature(reason)

            cell_path = cell_dir / "cell.json"
            _write_json(
                cell_path,
                {
                    "schema": NUMERICAL_CELL_SCHEMA,
                    "id": f"{reference.point_id}--{configuration.id}",
                    "status": comparison["status"],
                    "configuration": configuration.as_dict(),
                    "reference": reference.as_dict(),
                    "observation": None if observation is None else asdict(observation),
                    "comparison": comparison,
                    "semantic_signature": semantic_signature,
                    "numerics": numerics,
                    "run": (
                        None
                        if run_path is None
                        else {
                            "path": (run_path / "run.json").relative_to(destination).as_posix(),
                        }
                    ),
                    "error": error,
                },
            )
            cells.append(
                _cell_summary(
                    reference=reference,
                    configuration=configuration,
                    comparison=comparison,
                    observation=observation,
                    runtime_s=runtime_s,
                    run_path=run_path,
                    cell_path=cell_path,
                    destination=destination,
                    error=error,
                    numerics=numerics,
                    semantic_signature=semantic_signature,
                )
            )

    studies = _study_summaries(selected_profile, cells)
    status = _status(
        [cell["status"] for cell in cells] + [study["status"] for study in studies.values()]
    )
    cell_statuses = Counter(cell["status"] for cell in cells)
    gate_statuses = Counter(gate["status"] for cell in cells for gate in cell["gates"])
    family_gate_statuses = Counter(
        gate["status"] for study in studies.values() for gate in study["gates"]
    )
    platform_record = _platform_record()
    evidence = {
        "schema": NUMERICAL_EVIDENCE_SCHEMA,
        "kind": "canonical-buck-full-numerical-evidence-matrix",
        "status": status,
        "claim": {
            "classification": "verification",
            "validation_claim": False,
            "hardware_sign_off_allowed": False,
            "scope": (
                "canonical single-switch ideal buck periodic CCM/BCM/DCM numerical "
                "configuration matrix"
            ),
        },
        "profile": {
            key: profile_payload[key]
            for key in (
                "id",
                "sha256",
                "load_ratios",
                "point_count",
                "configuration_count",
                "cell_count",
                "studies",
            )
        },
        "source": {
            "revision": revision,
            "revision_method": revision_method,
            "pear_sim_version": __version__,
            "model_file": source_model.source_path.name,
            "model_hashes": model_hashes(source_model),
        },
        "platform": platform_record,
        "oracle": {
            "schema": BUCK_BOUNDARY_ORACLE_SCHEMA,
            "sha256": sha256_file(oracle_path),
            "boundary_load_ohm": oracle.bcm_load_ohm,
        },
        "cross_platform_policy": selected_profile.cross_platform_policy,
        "configurations": [
            configuration.as_dict() for configuration in selected_profile.configurations
        ],
        "cells": cells,
        "studies": studies,
        "summary": {
            "status": status,
            "point_count": len(oracle.points),
            "configuration_count": len(selected_profile.configurations),
            "cell_count": len(cells),
            "gate_count": sum(gate_statuses.values()),
            "family_gate_count": sum(family_gate_statuses.values()),
            "cell_status_counts": dict(sorted(cell_statuses.items())),
            "gate_status_counts": dict(sorted(gate_statuses.items())),
            "family_gate_status_counts": dict(sorted(family_gate_statuses.items())),
        },
        "artifacts": {
            "profile": {"path": profile_path.name, "sha256": sha256_file(profile_path)},
            "oracle": {"path": oracle_path.name, "sha256": sha256_file(oracle_path)},
            "human_report": {"path": "report/numerical-report.html"},
            "artifact_manifest": {"path": "artifact-manifest.json"},
        },
    }
    _write_json(evidence_path, evidence)
    report = generate_numerical_report(
        evidence,
        destination / "report",
        source_artifact_sha256=sha256_file(evidence_path),
    )
    manifest_path = _root_manifest(destination, evidence_path)
    return NumericalEvidenceArtifacts(
        output_dir=destination,
        profile_json=profile_path,
        oracle_json=oracle_path,
        platform_evidence=evidence_path,
        artifact_manifest=manifest_path,
        report_json=report.report_json,
        report_html=report.report_html,
        status=status,
    )


__all__ = [
    "CROSS_PLATFORM_POLICY_SCHEMA",
    "FULL_PROFILE_ID",
    "MAX_NUMERICAL_CELLS",
    "NUMERICAL_CELL_SCHEMA",
    "NUMERICAL_EVIDENCE_SCHEMA",
    "NUMERICAL_PROFILE_SCHEMA",
    "SEMANTIC_SIGNATURE_SCHEMA",
    "NumericalConfiguration",
    "NumericalEvidenceArtifacts",
    "NumericalEvidenceProfile",
    "full_v1_profile",
    "run_buck_numerical_evidence",
]
