from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Callable

from jsonschema import Draft202012Validator
import pytest

from pear_sim.contracts import sha256_file, sha256_json
from pear_sim.numerical_evidence import full_v1_profile
from pear_sim.platform_evidence import (
    CROSS_PLATFORM_EVIDENCE_SCHEMA,
    PlatformEvidenceError,
    aggregate_platform_evidence,
)


ROOT = Path(__file__).parents[1]
SYSTEMS = ("Linux", "Darwin", "Windows")
LOCAL_GATE_IDS = (
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
SIGNATURE_FIELD_IDS = (
    "regime",
    "state.cycle_start.inductor_current_a",
    "state.cycle_start.output_voltage_v",
    "state.switch_off.inductor_current_a",
    "state.switch_off.output_voltage_v",
    "state.cycle_end.inductor_current_a",
    "state.cycle_end.output_voltage_v",
    "events.diode_zero.present",
    "events.diode_zero.time_s",
    "metrics.mean_output_voltage_v",
    "metrics.minimum_inductor_current_a",
    "modes.fraction.on",
    "modes.fraction.freewheel",
    "modes.fraction.idle",
    "modes.fraction.unclassified",
    "physics.balance_residual_relative_signed",
    "cycle.delta_inductor_current_a",
    "cycle.delta_output_voltage_v",
    "events.semantic_sequence_sha256",
    "events.count",
    "modes.topology_sequence_sha256",
    "modes.interval_count",
)
EXACT_VALUES: dict[str, Any] = {
    "events.semantic_sequence_sha256": "1" * 64,
    "events.count": 4,
    "modes.topology_sequence_sha256": "2" * 64,
    "modes.interval_count": 3,
}
STUDY_SUFFIX = {
    "tolerance": "convergence",
    "max_step": "convergence",
    "output_grid": "independence",
    "solver": "consistency",
}


def _combined_status(statuses: list[str]) -> str:
    if "fail" in statuses:
        return "fail"
    if any(status != "pass" for status in statuses):
        return "not-run"
    return "pass"


def _counts(statuses: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(statuses).items()))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _oracle(profile: dict[str, Any], *, boundary_load_ohm: float = 12.5) -> dict[str, Any]:
    points = []
    for ratio in profile["load_ratios"]:
        point_id = f"load-ratio-{format(float(ratio), '.12g').replace('.', 'p')}"
        regime = "CCM" if ratio < 1.0 else "BCM" if ratio == 1.0 else "DCM"
        points.append(
            {
                "point_id": point_id,
                "load_ratio_to_bcm": ratio,
                "load_ohm": boundary_load_ohm * ratio,
                "regime": regime,
                "states": {
                    "cycle_start": {
                        "inductor_current_a": 5.0,
                        "output_voltage_v": 5.0,
                    },
                    "switch_off": {
                        "inductor_current_a": 5.0,
                        "output_voltage_v": 5.0,
                    },
                    "cycle_end": {
                        "inductor_current_a": 5.0,
                        "output_voltage_v": 5.0,
                    },
                },
                "metrics": {"mean_output_voltage_v": 0.05},
            }
        )
    return {
        "schema": "pear-sim/benchmark-oracle/ideal-buck-boundary/v0.1",
        "kind": "independent-periodic-affine-buck-boundary-verification",
        "claim": {
            "classification": "verification",
            "validation_claim": False,
            "scope": "canonical buck test oracle",
        },
        "method": {
            "periodic_ccm": "independent affine fixed point",
            "bcm_boundary": "independent scalar root",
            "pear_sim_circuit_or_solver_imported": False,
        },
        "parameters_si": {
            "vin_v": 100.0,
            "inductance_h": 0.001,
            "capacitance_f": 0.0001,
            "switching_frequency_hz": 20_000.0,
            "period_s": 0.00005,
            "duty": 0.4,
        },
        "boundary": {
            "load_ohm": boundary_load_ohm,
            "definition": "periodic CCM cycle-start inductor current equals zero",
            "coincidence_fraction_tolerance": 1e-8,
        },
        "load_ratios": profile["load_ratios"],
        "comparison_gate_contract": [{"id": gate_id} for gate_id in LOCAL_GATE_IDS],
        "oracle_health": {"status": "pass", "point_count": 9, "failed_gate_count": 0},
        "points": points,
    }


def _signature(
    profile: dict[str, Any],
    *,
    oracle_point: dict[str, Any],
    oracle_parameters: dict[str, Any],
    mean_output_voltage_v: float,
) -> dict[str, Any]:
    policy = profile["cross_platform_policy"]["semantic_signature"]
    regime = oracle_point["regime"]
    vin_v = float(oracle_parameters["vin_v"])
    current_base_a = (
        vin_v
        * float(oracle_parameters["duty"])
        * float(oracle_parameters["period_s"])
        / float(oracle_parameters["inductance_h"])
    )

    def state_tolerance(reference: float, scale: float) -> float:
        if abs(reference) <= scale * 1e-10:
            return max(scale * 2e-10, 1e-15)
        return max(abs(reference) * 2e-7, scale * 1e-15)

    def dynamic_absolute(field_id: str) -> float:
        if field_id.startswith("state."):
            _, state_name, quantity = field_id.split(".")
            reference = float(oracle_point["states"][state_name][quantity])
            scale = current_base_a if quantity == "inductor_current_a" else vin_v
            return state_tolerance(reference, scale)
        if field_id == "events.diode_zero.time_s":
            return max(float(oracle_parameters["period_s"]) * 1e-5, 1e-15)
        if field_id == "metrics.mean_output_voltage_v":
            reference = float(oracle_point["metrics"]["mean_output_voltage_v"])
            return max(abs(reference) * 2e-5, vin_v * 2e-17)
        if field_id in {
            "metrics.minimum_inductor_current_a",
            "cycle.delta_inductor_current_a",
        }:
            return max(current_base_a * 2e-10, 1e-15)
        if field_id == "cycle.delta_output_voltage_v":
            return max(vin_v * 2e-10, 1e-15)
        raise AssertionError(f"unexpected dynamic tolerance field {field_id}")

    fields: dict[str, Any] = {}
    for field_index, field_id in enumerate(SIGNATURE_FIELD_IDS):
        policy_id = policy["field_policy_ids"][field_id]
        tolerance_policy = policy["tolerance_policies"][policy_id]
        if field_id == "regime":
            value: Any = regime
        elif field_id == "events.diode_zero.present":
            value = regime != "CCM"
        elif field_id in EXACT_VALUES:
            value = EXACT_VALUES[field_id]
        elif field_id == "metrics.mean_output_voltage_v":
            value = mean_output_voltage_v
        else:
            value = float(field_index) / 100.0
        if tolerance_policy["kind"] == "exact":
            tolerance = {"kind": "exact", "policy_id": policy_id}
        else:
            tolerance = {
                "kind": "numeric",
                "policy_id": policy_id,
                "absolute": float(
                    tolerance_policy["absolute"]
                    if "absolute" in tolerance_policy
                    else dynamic_absolute(field_id)
                ),
                "relative": float(tolerance_policy["relative"]),
            }
        fields[field_id] = {
            "quantity": field_id.replace(".", "_"),
            "unit": "1",
            "policy_id": policy_id,
            "value": value,
            "cross_platform_tolerance": tolerance,
        }
    return {
        "schema": "pear-sim/numerical-semantic-signature/canonical-buck/v0.1",
        "status": "available",
        "fields": fields,
    }


def _payload(
    system: str,
    *,
    mean_output_voltage_v: float = 20.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    profile = full_v1_profile().as_dict()
    oracle = _oracle(profile)
    cells = []
    for configuration in profile["configurations"]:
        for point in oracle["points"]:
            cells.append(
                {
                    "id": f"{point['point_id']}--{configuration['id']}",
                    "point_id": point["point_id"],
                    "configuration_id": configuration["id"],
                    "load_ratio_to_bcm": point["load_ratio_to_bcm"],
                    "expected_regime": point["regime"],
                    "observed_regime": point["regime"],
                    "status": "pass",
                    "gates": [{"id": gate_id, "status": "pass"} for gate_id in LOCAL_GATE_IDS],
                    "semantic_signature": _signature(
                        profile,
                        oracle_point=point,
                        oracle_parameters=oracle["parameters_si"],
                        mean_output_voltage_v=mean_output_voltage_v,
                    ),
                }
            )
    studies: dict[str, Any] = {}
    for study_id, configuration_ids in profile["studies"].items():
        selected = [cell for cell in cells if cell["configuration_id"] in configuration_ids]
        family_gates = [
            {
                "id": f"{study_id}:{point['point_id']}:{STUDY_SUFFIX[study_id]}",
                "point_id": point["point_id"],
                "configuration_ids": list(configuration_ids),
                "status": "pass",
            }
            for point in oracle["points"]
        ]
        studies[study_id] = {
            "status": "pass",
            "configuration_ids": list(configuration_ids),
            "configuration_count": len(configuration_ids),
            "cell_count": len(selected),
            "cell_status_counts": {"pass": len(selected)},
            "oracle_gate_status_counts": {"pass": len(selected) * len(LOCAL_GATE_IDS)},
            "family_gate_status_counts": {"pass": len(family_gates)},
            "gates": family_gates,
        }
    source_hashes = {
        "source_sha256": "1" * 64,
        "model_sha256": "2" * 64,
        "request_sha256": "3" * 64,
    }
    payload = {
        "schema": "pear-sim/platform-evidence/canonical-buck-numerical/v0.1",
        "kind": "canonical-buck-full-numerical-evidence-matrix",
        "status": "pass",
        "claim": {
            "classification": "verification",
            "validation_claim": False,
            "hardware_sign_off_allowed": False,
            "scope": "canonical buck full-v1 test matrix",
        },
        "profile": {
            key: profile[key]
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
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "revision_method": "git",
            "pear_sim_version": "0.2.0.dev2",
            "model_file": "buck_dcm.yaml",
            "model_hashes": source_hashes,
        },
        "platform": {
            "id": system.lower(),
            "system": system,
            "python": "3.12.8",
            "numpy": "2.2.0",
            "scipy": "1.15.0",
        },
        "cross_platform_policy": profile["cross_platform_policy"],
        "configurations": profile["configurations"],
        "cells": cells,
        "studies": studies,
        "summary": {
            "status": "pass",
            "point_count": 9,
            "configuration_count": 16,
            "cell_count": 144,
            "gate_count": 144 * len(LOCAL_GATE_IDS),
            "family_gate_count": 36,
            "cell_status_counts": {"pass": 144},
            "gate_status_counts": {"pass": 144 * len(LOCAL_GATE_IDS)},
            "family_gate_status_counts": {"pass": 36},
        },
    }
    return payload, profile, oracle


def _recompute_statuses(payload: dict[str, Any]) -> None:
    for study_id, study in payload["studies"].items():
        configuration_ids = study["configuration_ids"]
        selected = [
            cell for cell in payload["cells"] if cell["configuration_id"] in configuration_ids
        ]
        cell_statuses = [cell["status"] for cell in selected]
        oracle_statuses = [gate["status"] for cell in selected for gate in cell["gates"]]
        family_statuses = [gate["status"] for gate in study["gates"]]
        study["cell_count"] = len(selected)
        study["cell_status_counts"] = _counts(cell_statuses)
        study["oracle_gate_status_counts"] = _counts(oracle_statuses)
        study["family_gate_status_counts"] = _counts(family_statuses)
        study["status"] = _combined_status([*cell_statuses, *family_statuses])
    cell_statuses = [cell["status"] for cell in payload["cells"]]
    gate_statuses = [gate["status"] for cell in payload["cells"] for gate in cell["gates"]]
    family_statuses = [
        gate["status"] for study in payload["studies"].values() for gate in study["gates"]
    ]
    status = _combined_status(
        [*cell_statuses, *(study["status"] for study in payload["studies"].values())]
    )
    payload["status"] = status
    payload["summary"] = {
        "status": status,
        "point_count": 9,
        "configuration_count": 16,
        "cell_count": 144,
        "gate_count": 144 * len(LOCAL_GATE_IDS),
        "family_gate_count": 36,
        "cell_status_counts": _counts(cell_statuses),
        "gate_status_counts": _counts(gate_statuses),
        "family_gate_status_counts": _counts(family_statuses),
    }


def _write_bundle(
    root: Path,
    payload: dict[str, Any],
    profile: dict[str, Any],
    oracle: dict[str, Any],
) -> Path:
    root.mkdir()
    profile_path = root / "numerical-profile.json"
    oracle_path = root / "oracle.json"
    _write_json(profile_path, profile)
    _write_json(oracle_path, oracle)
    payload["oracle"] = {
        "schema": oracle["schema"],
        "sha256": sha256_file(oracle_path),
        "boundary_load_ohm": oracle["boundary"]["load_ohm"],
    }
    payload["artifacts"] = {
        "profile": {"path": profile_path.name, "sha256": sha256_file(profile_path)},
        "oracle": {"path": oracle_path.name, "sha256": sha256_file(oracle_path)},
        "human_report": {"path": "report/numerical-report.html"},
        "artifact_manifest": {"path": "artifact-manifest.json"},
    }
    evidence_path = root / "platform-evidence.json"
    _write_json(evidence_path, payload)
    artifacts = [
        {
            "path": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted((oracle_path, profile_path, evidence_path))
    ]
    _write_json(
        root / "artifact-manifest.json",
        {
            "schema": "pear-sim/artifact-manifest/v0.1",
            "subject": {
                "kind": "canonical-buck-numerical-evidence",
                "sha256": sha256_file(evidence_path),
            },
            "artifacts": artifacts,
        },
    )
    return root


def _bundles(
    tmp_path: Path,
    payloads: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] | None = None,
) -> list[Path]:
    selected = payloads or [_payload(system) for system in SYSTEMS]
    return [
        _write_bundle(
            tmp_path / f"input-{index}",
            deepcopy(payload),
            deepcopy(profile),
            deepcopy(oracle),
        )
        for index, (payload, profile, oracle) in enumerate(selected)
    ]


def _cell(payload: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return payload["cells"][index]


def _signature_field(
    payload: dict[str, Any],
    field_id: str,
    *,
    cell_index: int = 0,
) -> dict[str, Any]:
    return _cell(payload, cell_index)["semantic_signature"]["fields"][field_id]


def test_aggregate_full_v1_passes_and_binds_all_provenance(tmp_path: Path) -> None:
    inputs = _bundles(tmp_path)

    artifacts = aggregate_platform_evidence(
        [inputs[2], inputs[0], inputs[1]], tmp_path / "cross-platform"
    )

    result = json.loads(artifacts.evidence_json.read_text(encoding="utf-8"))
    manifest = json.loads(artifacts.artifact_manifest.read_text(encoding="utf-8"))
    profile = full_v1_profile().as_dict()
    schema = json.loads(
        (ROOT / "schema" / "cross-platform-evidence.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator(schema).validate(result)
    assert artifacts.overall_status == "pass"
    assert result["schema"] == CROSS_PLATFORM_EVIDENCE_SCHEMA
    assert result["status"] == "pass"
    assert result["profile_semantic_sha256"] == profile["sha256"]
    assert result["profile_semantic_sha256"] != sha256_json(result["profile"])
    assert result["summary"]["cell_count"] == 144
    assert result["summary"]["comparison_count"] == 3168
    assert len(result["comparisons"]) == 3168
    assert result["comparison_policy"]["oracle_gate_values_compared"] is False
    assert result["comparison_policy"]["source_binding"]["model_hashes"] == {
        "model_sha256": "2" * 64,
        "request_sha256": "3" * 64,
        "source_sha256": "1" * 64,
    }
    oracle_binding = result["comparison_policy"]["oracle_binding"]
    assert oracle_binding["schema"] == (
        "pear-sim/benchmark-oracle/ideal-buck-boundary/v0.1"
    )
    assert len(oracle_binding["oracle_identity_sha256"]) == 64
    assert oracle_binding["boundary_load_ohm"]["status"] == "pass"
    assert set(oracle_binding["local_artifact_sha256"]) == {
        "Darwin",
        "Linux",
        "Windows",
    }
    assert result["studies"]["solver"]["platform_family_evidence"]["Windows"][
        "family_gate_status_counts"
    ] == {"pass": 9}
    comparison = next(
        row
        for row in result["comparisons"]
        if row["gate_id"] == "metrics.mean_output_voltage_v"
    )
    assert comparison["quantity"] == "metrics_mean_output_voltage_v"
    assert comparison["policy_id"] == "cycle_mean_output"
    assert comparison["status"] == "pass"
    assert manifest["subject"] == {
        "kind": "cross-platform-evidence",
        "sha256": sha256_file(artifacts.evidence_json),
        "semantic_sha256": sha256_json(result),
    }
    assert artifacts.report_html.is_file()
    assert artifacts.report_json.is_file()


def test_aggregate_turns_physical_signature_drift_into_failure(tmp_path: Path) -> None:
    payloads = [_payload(system) for system in SYSTEMS]
    _signature_field(payloads[1][0], "metrics.mean_output_voltage_v")["value"] = 20.01

    artifacts = aggregate_platform_evidence(
        _bundles(tmp_path, payloads), tmp_path / "cross-platform"
    )

    result = json.loads(artifacts.evidence_json.read_text(encoding="utf-8"))
    comparison = next(
        row
        for row in result["comparisons"]
        if row["cell_id"] == result["cells"][0]["id"]
        and row["gate_id"] == "metrics.mean_output_voltage_v"
    )
    assert artifacts.overall_status == "fail"
    assert comparison["status"] == "fail"
    assert comparison["actual_range"] > comparison["limit"]


def test_aggregate_preserves_valid_fail_and_not_run_source_evidence(tmp_path: Path) -> None:
    payloads = [_payload(system) for system in SYSTEMS]
    failed = _cell(payloads[0][0], 0)
    failed["gates"][0]["status"] = "fail"
    failed["status"] = "fail"
    _recompute_statuses(payloads[0][0])
    not_run = _cell(payloads[1][0], 1)
    for gate in not_run["gates"]:
        gate["status"] = "not-run"
    not_run["status"] = "not-run"
    not_run["semantic_signature"] = {
        "schema": "pear-sim/numerical-semantic-signature/canonical-buck/v0.1",
        "status": "not-run",
        "fields": {},
        "reason": "synthetic unavailable solver cell",
    }
    _recompute_statuses(payloads[1][0])

    artifacts = aggregate_platform_evidence(
        _bundles(tmp_path, payloads), tmp_path / "cross-platform"
    )

    result = json.loads(artifacts.evidence_json.read_text(encoding="utf-8"))
    by_id = {cell["id"]: cell for cell in result["cells"]}
    assert artifacts.overall_status == "fail"
    assert by_id[failed["id"]]["source_status"] == "fail"
    assert by_id[not_run["id"]]["source_status"] == "not-run"
    not_run_comparison = next(
        row
        for row in result["comparisons"]
        if row["cell_id"] == not_run["id"] and row["gate_id"] == "regime"
    )
    assert not_run_comparison["status"] == "not-run"
    assert not_run_comparison["values"]["Darwin"] is None
    assert len(result["comparisons"]) == 3168


def test_aggregate_rejects_tampered_platform_artifact(tmp_path: Path) -> None:
    inputs = _bundles(tmp_path)
    evidence_path = inputs[0] / "platform-evidence.json"
    tampered = json.loads(evidence_path.read_text(encoding="utf-8"))
    tampered["source"]["pear_sim_version"] = "tampered"
    _write_json(evidence_path, tampered)

    with pytest.raises(PlatformEvidenceError, match="hash mismatch"):
        aggregate_platform_evidence(inputs, tmp_path / "cross-platform")


def test_aggregate_requires_exact_native_platform_set(tmp_path: Path) -> None:
    payloads = [_payload("Linux"), _payload("Darwin"), _payload("FreeBSD")]

    with pytest.raises(PlatformEvidenceError, match="must contain exactly"):
        aggregate_platform_evidence(
            _bundles(tmp_path, payloads), tmp_path / "cross-platform"
        )


BindingMutation = Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None]


def _mutate_model_hash(
    payload: dict[str, Any], profile: dict[str, Any], oracle: dict[str, Any]
) -> None:
    del profile, oracle
    payload["source"]["model_hashes"]["model_sha256"] = "9" * 64


def _mutate_version(
    payload: dict[str, Any], profile: dict[str, Any], oracle: dict[str, Any]
) -> None:
    del profile, oracle
    payload["source"]["pear_sim_version"] = "0.2.0.mutated"


def _mutate_oracle_contract(
    payload: dict[str, Any], profile: dict[str, Any], oracle: dict[str, Any]
) -> None:
    del payload, profile
    oracle["method"]["periodic_ccm"] = "mutated oracle method"


def _mutate_unknown_revision(
    payload: dict[str, Any], profile: dict[str, Any], oracle: dict[str, Any]
) -> None:
    del profile, oracle
    payload["source"]["revision"] = "unknown"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (_mutate_model_hash, "source/model/version binding differs"),
        (_mutate_version, "source/model/version binding differs"),
        (_mutate_oracle_contract, "oracle identity binding differs"),
        (_mutate_unknown_revision, "unknown source revision"),
    ],
)
def test_aggregate_rejects_provenance_binding_mutations(
    tmp_path: Path,
    mutation: BindingMutation,
    match: str,
) -> None:
    payloads = [_payload(system) for system in SYSTEMS]
    mutation(*payloads[2])

    with pytest.raises(PlatformEvidenceError, match=match):
        aggregate_platform_evidence(
            _bundles(tmp_path, payloads), tmp_path / "cross-platform"
        )


def _set_oracle_boundary(oracle: dict[str, Any], value: float) -> None:
    oracle["boundary"]["load_ohm"] = value
    for point in oracle["points"]:
        point["load_ohm"] = value * point["load_ratio_to_bcm"]


def test_aggregate_tolerates_one_ulp_local_oracle_boundary_difference(
    tmp_path: Path,
) -> None:
    payloads = [_payload(system) for system in SYSTEMS]
    shifted = math.nextafter(12.5, math.inf)
    _set_oracle_boundary(payloads[2][2], shifted)

    artifacts = aggregate_platform_evidence(
        _bundles(tmp_path, payloads), tmp_path / "cross-platform"
    )

    result = json.loads(artifacts.evidence_json.read_text(encoding="utf-8"))
    binding = result["comparison_policy"]["oracle_binding"]
    comparison = binding["boundary_load_ohm"]
    assert artifacts.overall_status == "pass"
    assert comparison["status"] == "pass"
    assert comparison["values"]["Windows"] == shifted
    assert comparison["actual_range"] > 0.0
    assert comparison["actual_range"] <= comparison["limit"]
    assert len(set(binding["local_artifact_sha256"].values())) == 2
    assert {
        record["system"]: record["oracle"]["local_artifact_sha256"]
        for record in result["inputs"]
    } == binding["local_artifact_sha256"]
    assert len(
        {
            record["oracle"]["oracle_identity_sha256"]
            for record in result["inputs"]
        }
    ) == 1


def test_aggregate_turns_material_oracle_boundary_drift_into_failure(
    tmp_path: Path,
) -> None:
    payloads = [_payload(system) for system in SYSTEMS]
    _set_oracle_boundary(payloads[1][2], 12.75)

    artifacts = aggregate_platform_evidence(
        _bundles(tmp_path, payloads), tmp_path / "cross-platform"
    )

    result = json.loads(artifacts.evidence_json.read_text(encoding="utf-8"))
    comparison = result["comparison_policy"]["oracle_binding"]["boundary_load_ohm"]
    assert artifacts.overall_status == "fail"
    assert comparison["status"] == "fail"
    assert comparison["actual_range"] > comparison["limit"]
    assert {cell["status"] for cell in result["cells"]} == {"pass"}
    assert any(
        failure["kind"] == "oracle-boundary-comparison"
        for failure in result["failures"]
    )


def _mutate_missing_cell(
    payload: dict[str, Any], profile: dict[str, Any], oracle: dict[str, Any]
) -> None:
    del profile, oracle
    payload["cells"].pop()


def _mutate_cell_id(
    payload: dict[str, Any], profile: dict[str, Any], oracle: dict[str, Any]
) -> None:
    del profile, oracle
    payload["cells"][0]["id"] = "wrong-cell-id"


def _mutate_local_gate_id(
    payload: dict[str, Any], profile: dict[str, Any], oracle: dict[str, Any]
) -> None:
    del profile, oracle
    payload["cells"][0]["gates"][0]["id"] = "wrong-gate"


def _mutate_signature_field(
    payload: dict[str, Any], profile: dict[str, Any], oracle: dict[str, Any]
) -> None:
    del profile, oracle
    del payload["cells"][0]["semantic_signature"]["fields"]["regime"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (_mutate_missing_cell, "exactly 144"),
        (_mutate_cell_id, "canonical point/config pair"),
        (_mutate_local_gate_id, "eleven canonical oracle gate IDs"),
        (_mutate_signature_field, "exactly 22 canonical fields"),
    ],
)
def test_aggregate_rejects_matrix_and_signature_mutations(
    tmp_path: Path,
    mutation: BindingMutation,
    match: str,
) -> None:
    payloads = [_payload(system) for system in SYSTEMS]
    mutation(*payloads[0])

    with pytest.raises(PlatformEvidenceError, match=match):
        aggregate_platform_evidence(
            _bundles(tmp_path, payloads), tmp_path / "cross-platform"
        )


def _mutate_family_configuration_ids(
    payload: dict[str, Any], profile: dict[str, Any], oracle: dict[str, Any]
) -> None:
    del profile, oracle
    payload["studies"]["solver"]["configuration_ids"].pop()


def _mutate_family_count(
    payload: dict[str, Any], profile: dict[str, Any], oracle: dict[str, Any]
) -> None:
    del profile, oracle
    payload["studies"]["solver"]["family_gate_status_counts"] = {"pass": 8}


def _mutate_family_status(
    payload: dict[str, Any], profile: dict[str, Any], oracle: dict[str, Any]
) -> None:
    del profile, oracle
    payload["studies"]["solver"]["status"] = "fail"


def _mutate_family_gate_id(
    payload: dict[str, Any], profile: dict[str, Any], oracle: dict[str, Any]
) -> None:
    del profile, oracle
    payload["studies"]["solver"]["gates"][0]["id"] = "solver:wrong:consistency"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (_mutate_family_configuration_ids, "configuration_ids differ"),
        (_mutate_family_count, "family_gate_status_counts are inconsistent"),
        (_mutate_family_status, "status differs from its cells and family gates"),
        (_mutate_family_gate_id, "does not match its point"),
    ],
)
def test_aggregate_rejects_family_study_mutations(
    tmp_path: Path,
    mutation: BindingMutation,
    match: str,
) -> None:
    payloads = [_payload(system) for system in SYSTEMS]
    mutation(*payloads[1])

    with pytest.raises(PlatformEvidenceError, match=match):
        aggregate_platform_evidence(
            _bundles(tmp_path, payloads), tmp_path / "cross-platform"
        )


def test_aggregate_rejects_dynamic_tolerance_inflation_even_when_all_platforms_agree(
    tmp_path: Path,
) -> None:
    payloads = [_payload(system) for system in SYSTEMS]
    for payload, _profile_record, _oracle_record in payloads:
        tolerance = _signature_field(
            payload, "state.cycle_start.inductor_current_a"
        )["cross_platform_tolerance"]
        tolerance["absolute"] *= 2.0

    with pytest.raises(PlatformEvidenceError, match="differs from its local oracle rule"):
        aggregate_platform_evidence(
            _bundles(tmp_path, payloads), tmp_path / "cross-platform"
        )


def test_aggregate_accepts_large_ulp_locally_verified_dynamic_drift_and_uses_safe_limit(
    tmp_path: Path,
) -> None:
    payloads = [_payload(system) for system in SYSTEMS]
    windows_payload, _windows_profile, windows_oracle = payloads[2]
    point = windows_oracle["points"][0]
    point_id = point["point_id"]
    shifted_reference = float(point["states"]["cycle_start"]["inductor_current_a"])
    for _ in range(1000):
        shifted_reference = math.nextafter(shifted_reference, math.inf)
    point["states"]["cycle_start"]["inductor_current_a"] = shifted_reference
    shifted_absolute = shifted_reference * 2e-7
    for cell in windows_payload["cells"]:
        if cell["point_id"] == point_id:
            cell["semantic_signature"]["fields"][
                "state.cycle_start.inductor_current_a"
            ]["cross_platform_tolerance"]["absolute"] = shifted_absolute

    artifacts = aggregate_platform_evidence(
        _bundles(tmp_path, payloads), tmp_path / "cross-platform"
    )

    result = json.loads(artifacts.evidence_json.read_text(encoding="utf-8"))
    cell = next(item for item in result["cells"] if item["point_id"] == point_id)
    gate = next(
        item
        for item in cell["gates"]
        if item["id"] == "state.cycle_start.inductor_current_a"
    )
    local_limits = gate["comparison"]["platform_absolute_tolerances"]
    assert artifacts.overall_status == "pass"
    assert local_limits["Windows"] == shifted_absolute
    assert local_limits["Linux"] < local_limits["Windows"]
    assert gate["comparison"]["dynamic_absolute_tolerance_ulp_span"] > 640
    assert gate["comparison"]["dynamic_absolute_tolerance_cross_platform_ceiling"] is None
    assert gate["comparison"]["dynamic_absolute_common_limit_rule"] == (
        "minimum-locally-verified-platform-tolerance"
    )
    assert gate["comparison"]["absolute_tolerance"] == min(local_limits.values())
    contract = result["comparison_policy"]["dynamic_absolute_tolerance"]
    assert contract == {
        "local_verification": "within one ULP of the local oracle rule",
        "cross_platform_ceiling": None,
        "common_limit": "minimum locally verified platform tolerance",
        "ulp_span": "recorded as a diagnostic only",
    }
