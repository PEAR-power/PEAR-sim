# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from pear_sim.capabilities import build_capabilities, get_capability
from pear_sim.contracts import canonical_json_bytes, sha256_json


def test_capability_catalog_is_versioned_deterministic_and_fail_closed() -> None:
    first = build_capabilities()
    second = build_capabilities()

    assert first == second
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert sha256_json(first) == sha256_json(second)
    assert first["schema"] == "pear-sim/capabilities/v0.1"
    assert first["maturity"] == "early_alpha"
    assert first["default_policy"]["unlisted_capability"] == "unsupported"
    assert first["default_policy"]["silent_fallback"] is False
    assert first["claim_boundary"]["validation_claim"] is False
    assert first["claim_boundary"]["compatibility_claim"] is False

    records = first["capabilities"]
    identifiers = [record["id"] for record in records]
    assert identifiers == sorted(identifiers)
    assert len(identifiers) == len(set(identifiers))
    assert {record["status"] for record in records} == {
        "supported",
        "experimental",
        "unsupported",
    }


def test_capability_payload_is_a_fresh_copy_and_known_lookup_is_isolated() -> None:
    payload = build_capabilities()
    original_first_status = payload["capabilities"][0]["status"]
    payload["capabilities"][0]["status"] = "unsupported"
    payload["claim_boundary"]["prohibited_uses"].clear()

    fresh = build_capabilities()
    assert fresh["capabilities"][0]["status"] == original_first_status
    assert fresh["claim_boundary"]["prohibited_uses"]

    known = get_capability("converter.buck.dcm.canonical_ideal")
    assert known["status"] == "experimental"
    known["evidence"].clear()
    assert get_capability("converter.buck.dcm.canonical_ideal")["evidence"]

    numerical_matrix = get_capability("verification.canonical_buck.numerical_matrix")
    assert numerical_matrix["status"] == "supported"
    assert "exact recognized canonical ideal buck" in numerical_matrix["scope"]


def test_canonical_boost_capabilities_are_narrow_and_producer_is_supported() -> None:
    for regime in ("ccm", "bcm", "dcm"):
        capability = get_capability(f"converter.boost.{regime}.canonical_ideal")
        assert capability["status"] == "experimental"
        assert "canonical" in capability["scope"]

    producer = get_capability("verification.canonical_boost.boundary_sweep")
    assert producer["status"] == "supported"
    assert "exact recognized canonical ideal boost" in producer["scope"]
    assert "verification evidence only" in producer["statement"]

    assert get_capability("converter.boost")["status"] == "unsupported"
    assert get_capability("converter.dc_dc.bcm_general")["status"] == "unsupported"
    assert get_capability("converter.dc_dc.dcm_general")["status"] == "unsupported"


def test_m24_agent_experiment_capabilities_are_narrow_and_fail_closed() -> None:
    for regime in ("ccm", "bcm", "dcm"):
        capability = get_capability(
            f"converter.buck_boost.inverting.{regime}.canonical_ideal"
        )
        assert capability["status"] == "experimental"
        assert "exact canonical" in capability["scope"]

    producer = get_capability(
        "verification.canonical_inverting_buck_boost.boundary_sweep"
    )
    assert producer["status"] == "supported"
    assert "exact recognized canonical ideal" in producer["scope"]
    assert "verification evidence only" in producer["statement"]

    comparison = get_capability("artifacts.run_comparison")
    assert comparison["status"] == "supported"
    assert "sign correction" in comparison["statement"]
    assert get_capability("verification.agent.run_compare")["status"] == "supported"
    assert get_capability("experiments.parameter_sweep.one_dimension")["status"] == "supported"

    # The exact slice does not silently broaden the general converter/DAE claims.
    assert get_capability("converter.buck_boost")["status"] == "unsupported"
    assert get_capability("converter.dc_dc.bcm_general")["status"] == "unsupported"
    assert get_capability("converter.dc_dc.dcm_general")["status"] == "unsupported"
    assert get_capability("electrical.general_descriptor_dae")["status"] == "unsupported"


def test_m31_complementary_pwm_is_narrowly_experimental() -> None:
    capability = get_capability("control.complementary_pwm_dead_time")

    assert capability["status"] == "experimental"
    assert "intrinsic interlock" in capability["scope"]
    assert "atomic" in capability["statement"]
    gate_drive = get_capability("artifacts.gate_drive_observation")
    assert gate_drive["status"] == "supported"
    assert "without claiming physical" in gate_drive["statement"]
    assert get_capability("converter.inverter_three_phase.spwm_svpwm_dead_time")[
        "status"
    ] == "unsupported"


def test_m32_spwm_and_harmonic_capabilities_are_narrow_and_fail_closed() -> None:
    modulation = get_capability("control.three_phase_spwm_offline_midpoint_sampled")
    assert modulation["status"] == "experimental"
    assert "offline-midpoint-sampled" in modulation["scope"]
    assert "six interlocked gate outputs" in modulation["statement"]

    harmonics = get_capability("analysis.harmonics.event_aligned_fourier")
    assert harmonics["status"] == "experimental"
    assert "dense output" in harmonics["statement"]
    assert "no independent reference" in harmonics["statement"]

    example = get_capability("converter.inverter_three_phase.spwm_midpoint_neutral_rl")
    assert example["status"] == "experimental"
    assert "neutral" in example["scope"] and "DC midpoint" in example["scope"]

    observation = get_capability("artifacts.three_phase_modulation_observation")
    assert observation["status"] == "supported"
    assert "without recomputing" in observation["statement"]

    # The implemented slice does not silently promote the complete VAL-005 surface.
    complete = get_capability("converter.inverter_three_phase.spwm_svpwm_dead_time")
    assert complete["status"] == "unsupported"
    assert "SVPWM" in complete["statement"]


def test_unknown_capability_is_explicitly_unsupported() -> None:
    result = get_capability("converter.future.magic")

    assert result == {
        "id": "converter.future.magic",
        "status": "unsupported",
        "scope": "unlisted capability",
        "statement": "This capability is absent from the versioned catalog and is unsupported.",
        "evidence": [],
        "reason": "fail_closed_unlisted_capability",
    }


def test_capability_lookup_rejects_invalid_or_unbounded_identifier() -> None:
    for invalid in ("", "   ", None, 17):
        try:
            get_capability(invalid)  # type: ignore[arg-type]
        except ValueError:
            pass
        else:  # pragma: no cover - assertion gives a clearer failure than pytest parametrization
            raise AssertionError(f"Expected ValueError for {invalid!r}")

    try:
        get_capability("x" * 257)
    except ValueError as exc:
        assert "256-character" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for an unbounded capability ID")
