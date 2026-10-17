from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from pear_sim.contracts import physics_model_dict, simulation_request_dict
from pear_sim.errors import ModelError
from pear_sim.limits import (
    MAX_HARMONIC_QUADRATURE_PANELS,
    MAX_HARMONIC_QUADRATURE_WORK_UNITS,
    MAX_SPWM_CARRIER_RATIO,
)
from pear_sim.model import load_model


def _payload() -> dict[str, object]:
    return {
        "schema": "pear-sim/v0.1",
        "name": "three_phase_spwm_contract",
        "circuit": {
            "ground": "gnd",
            "components": [
                {
                    "id": "c1",
                    "type": "capacitor",
                    "p": "out",
                    "n": "gnd",
                    "value": "1 uF",
                }
            ],
        },
        "controls": {
            "bridge": {
                "type": "three_phase_spwm",
                "carrier_frequency": "20 kHz",
                "fundamental_frequency": "1 kHz",
                "modulation_index": "80 %",
                "dead_time": "500 ns",
                "carrier_phase": "2 us",
                "reference_phase": "100 us",
                "sequence": "abc",
            }
        },
        "analysis": {
            "transient": {
                "start": "0 s",
                "stop": "10 ms",
                "output_step": "2 us",
                "solver": "dop853",
                "rtol": 1e-9,
                "atol": 1e-11,
            },
            "harmonics": {
                "start": "5 ms",
                "stop": "10 ms",
                "fundamental_frequency": "1 kHz",
                "max_harmonic": 100,
                "signals": [
                    {"signal": "v(c1)", "semantic": "phase_voltage_a"}
                ],
            },
        },
        "outputs": ["v(c1)"],
    }


def _load(tmp_path: Path, payload: dict[str, object], name: str = "model"):
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return load_model(path)


def test_three_phase_spwm_expands_to_six_explicit_gate_outputs(tmp_path: Path) -> None:
    model = _load(tmp_path, _payload())

    assert tuple(model.controls) == (
        "bridge.a.high",
        "bridge.a.low",
        "bridge.b.high",
        "bridge.b.low",
        "bridge.c.high",
        "bridge.c.low",
    )
    for phase in ("a", "b", "c"):
        high = model.controls[f"bridge.{phase}.high"]
        low = model.controls[f"bridge.{phase}.low"]
        assert high.kind == low.kind == "three_phase_spwm"
        assert high.block_id == low.block_id == "bridge"
        assert high.phase_label == low.phase_label == phase
        assert high.role == "high"
        assert low.role == "low"
        assert high.partner_id == low.id
        assert low.partner_id == high.id
        assert high.frequency == low.frequency == 20_000.0
        assert high.duty == low.duty == 0.5
        assert high.phase == low.phase == 2e-6
        assert high.normalized_phase == low.normalized_phase == 2e-6
        assert high.dead_time == low.dead_time == pytest.approx(500e-9)
        assert high.sequence == low.sequence == "abc"
        assert high.fundamental_frequency == low.fundamental_frequency == 1_000.0
        assert high.modulation_index == low.modulation_index == 0.8
        assert high.reference_phase == low.reference_phase == pytest.approx(100e-6)
        assert high.carrier_ratio == low.carrier_ratio == 20


def test_three_phase_spwm_defaults_and_grouped_physics_contract(tmp_path: Path) -> None:
    payload = _payload()
    source = payload["controls"]["bridge"]  # type: ignore[index]
    del source["carrier_phase"]
    del source["reference_phase"]
    del source["sequence"]
    model = _load(tmp_path, payload)

    assert model.controls["bridge.a.high"].phase == 0.0
    assert model.controls["bridge.a.high"].reference_phase == 0.0
    assert model.controls["bridge.a.high"].sequence == "abc"
    resolved_dead_time = model.controls["bridge.a.high"].dead_time
    assert physics_model_dict(model)["controls"] == [
        {
            "id": "bridge",
            "type": "three_phase_spwm",
            "carrier_frequency_hz": 20_000.0,
            "fundamental_frequency_hz": 1_000.0,
            "carrier_ratio": 20,
            "modulation_index": 0.8,
            "carrier_phase_s": 0.0,
            "reference_phase_s": 0.0,
            "sequence": "abc",
            "dead_time_s": resolved_dead_time,
            "sampling": "offline_midpoint_sampled_center_aligned",
            "outputs": {
                "a": {"high": "bridge.a.high", "low": "bridge.a.low"},
                "b": {"high": "bridge.b.high", "low": "bridge.b.low"},
                "c": {"high": "bridge.c.high", "low": "bridge.c.low"},
            },
            "edge_policy": "turn_off_immediate_turn_on_after_dead_time",
            "interlock": "intrinsic_non_disableable",
        }
    ]


def test_harmonic_request_is_explicit_and_part_of_request_semantics(tmp_path: Path) -> None:
    model = _load(tmp_path, _payload())

    assert model.harmonics is not None
    assert model.harmonics.start == 5e-3
    assert model.harmonics.stop == 10e-3
    assert model.harmonics.fundamental_frequency == 1_000.0
    assert model.harmonics.period_count == 5
    assert model.harmonics.max_harmonic == 100
    assert [
        (signal.signal, signal.semantic) for signal in model.harmonics.signals
    ] == [("v(c1)", "phase_voltage_a")]
    assert simulation_request_dict(model)["harmonics"] == {
        "start_s": 5e-3,
        "stop_s": 10e-3,
        "window_boundary": "half_open_[start,stop)",
        "fundamental_frequency_hz": 1_000.0,
        "period_count": 5,
        "max_harmonic": 100,
        "signals": [{"signal": "v(c1)", "semantic": "phase_voltage_a"}],
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("carrier_frequency", "1 kHz", "greater than fundamental"),
        ("carrier_frequency", "20.5 kHz", "integer carrier ratio"),
        ("fundamental_frequency", "0 Hz", "must be positive"),
        ("modulation_index", -0.01, r"must be in \[0, 1\)"),
        ("modulation_index", 1.0, r"must be in \[0, 1\)"),
        ("dead_time", "-1 ns", "must be non-negative"),
        ("dead_time", "5 us", "minimum commanded pulse"),
        ("sequence", "cab", r"is not one of \['abc', 'acb'\]"),
    ],
)
def test_invalid_spwm_parameters_fail_closed(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _payload()
    payload["controls"]["bridge"][field] = value  # type: ignore[index]

    with pytest.raises(ModelError, match=message):
        _load(tmp_path, payload)


def test_three_phase_spwm_cannot_mix_or_repeat_source_controls(tmp_path: Path) -> None:
    mixed = _payload()
    mixed["controls"]["legacy"] = {  # type: ignore[index]
        "type": "pwm",
        "frequency": "20 kHz",
        "duty": "50 %",
    }
    with pytest.raises(ModelError, match="controls/legacy"):
        _load(tmp_path, mixed, "mixed")

    repeated = _payload()
    repeated["controls"]["other"] = deepcopy(  # type: ignore[index]
        repeated["controls"]["bridge"]  # type: ignore[index]
    )
    with pytest.raises(ModelError, match="has too many properties"):
        _load(tmp_path, repeated, "repeated")


def test_three_phase_spwm_requires_every_source_parameter(tmp_path: Path) -> None:
    payload = _payload()
    del payload["controls"]["bridge"]["dead_time"]  # type: ignore[index]

    with pytest.raises(ModelError, match="'dead_time' is a required property"):
        _load(tmp_path, payload)


def test_three_phase_spwm_rejects_unportable_derived_gate_ids(tmp_path: Path) -> None:
    payload = _payload()
    source = payload["controls"].pop("bridge")  # type: ignore[union-attr]
    payload["controls"]["x" * 250] = source  # type: ignore[index]

    with pytest.raises(ModelError, match="derived gate id.*exceeds"):
        _load(tmp_path, payload)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("outside_start", r"window \[start, stop\).*within"),
        ("outside_stop", r"window \[start, stop\).*within"),
        ("noninteger_periods", "integer number of fundamental periods"),
        ("order_low", "less than the minimum of 2"),
        ("order_high", "greater than the maximum of 256"),
        ("order_bool", "is not of type 'integer'"),
        ("missing_output", "is not present in outputs"),
        ("bad_semantic", "semantic: 'voltage' is not one of"),
        ("semantic_family_mismatch", "semantic: 'phase_current_a' is not one of"),
        ("duplicate_signal", "map every signal at most once"),
        ("duplicate_semantic", "map every semantic at most once"),
        ("too_many_signals", "is not one of|is too long"),
    ],
)
def test_invalid_harmonic_requests_fail_closed(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    payload = _payload()
    harmonic = payload["analysis"]["harmonics"]  # type: ignore[index]
    if case == "outside_start":
        harmonic["start"] = "-1 ms"
    elif case == "outside_stop":
        harmonic["stop"] = "11 ms"
    elif case == "noninteger_periods":
        harmonic["stop"] = "9.5 ms"
    elif case == "order_low":
        harmonic["max_harmonic"] = 1
    elif case == "order_high":
        harmonic["max_harmonic"] = 257
    elif case == "order_bool":
        harmonic["max_harmonic"] = True
    elif case == "missing_output":
        harmonic["signals"] = [
            {"signal": "v(missing)", "semantic": "phase_voltage_a"}
        ]
    elif case == "bad_semantic":
        harmonic["signals"] = [{"signal": "v(c1)", "semantic": "voltage"}]
    elif case == "semantic_family_mismatch":
        harmonic["signals"] = [
            {"signal": "v(c1)", "semantic": "phase_current_a"}
        ]
    elif case == "duplicate_signal":
        harmonic["signals"] = [
            {"signal": "v(c1)", "semantic": "phase_voltage_a"},
            {"signal": "v(c1)", "semantic": "phase_voltage_b"},
        ]
    elif case == "duplicate_semantic":
        payload["circuit"]["components"].append(  # type: ignore[index]
            {
                "id": "probe",
                "type": "voltage_probe",
                "p": "out",
                "n": "gnd",
            }
        )
        payload["outputs"].append("v(probe)")  # type: ignore[union-attr]
        harmonic["signals"] = [
            {"signal": "v(c1)", "semantic": "phase_voltage_a"},
            {"signal": "v(probe)", "semantic": "phase_voltage_a"},
        ]
    elif case == "too_many_signals":
        harmonic["signals"] = [
            {"signal": "v(c1)", "semantic": semantic}
            for semantic in (
                "line_voltage_ab",
                "line_voltage_bc",
                "line_voltage_ca",
                "phase_voltage_a",
                "phase_voltage_b",
                "phase_voltage_c",
                "phase_current_a",
                "phase_current_b",
                "phase_current_c",
            )
            for _ in range(4)
        ]
    else:  # pragma: no cover - parametrization exhausts this branch
        raise AssertionError(case)

    with pytest.raises(ModelError, match=message):
        _load(tmp_path, payload, case)


def test_acb_sequence_is_explicitly_supported(tmp_path: Path) -> None:
    payload = _payload()
    payload["controls"]["bridge"]["sequence"] = "acb"  # type: ignore[index]

    model = _load(tmp_path, payload)
    assert {control.sequence for control in model.controls.values()} == {"acb"}
    assert physics_model_dict(model)["controls"][0]["sequence"] == "acb"


def test_carrier_ratio_accepts_only_ulp_level_float_normalization(tmp_path: Path) -> None:
    payload = _payload()
    payload["controls"]["bridge"]["carrier_frequency"] = (  # type: ignore[index]
        "20000.000000000004 Hz"
    )

    model = _load(tmp_path, payload)
    assert {control.carrier_ratio for control in model.controls.values()} == {20}


def test_carrier_ratio_portable_limit_fails_during_model_loading(tmp_path: Path) -> None:
    payload = _payload()
    payload["controls"]["bridge"]["carrier_frequency"] = (  # type: ignore[index]
        f"{MAX_SPWM_CARRIER_RATIO + 1} kHz"
    )

    with pytest.raises(ModelError, match="carrier_ratio exceeds the portable limit"):
        _load(tmp_path, payload)


def test_harmonic_minimum_work_budget_fails_during_model_loading(tmp_path: Path) -> None:
    payload = _payload()
    max_harmonic = 256
    periods = MAX_HARMONIC_QUADRATURE_PANELS // (4 * max_harmonic) + 1
    stop_ms = periods
    payload["analysis"]["transient"]["stop"] = f"{stop_ms} ms"  # type: ignore[index]
    harmonic = payload["analysis"]["harmonics"]  # type: ignore[index]
    harmonic["start"] = "0 ms"
    harmonic["stop"] = f"{stop_ms} ms"
    harmonic["max_harmonic"] = max_harmonic

    with pytest.raises(ModelError, match="quadrature panels.*portable limit"):
        _load(tmp_path, payload)


def test_harmonic_order_signal_work_budget_fails_during_model_loading(
    tmp_path: Path,
) -> None:
    payload = _payload()
    max_harmonic = 256
    per_period_work = 4 * max_harmonic * 8 * (max_harmonic + 1) * 2
    periods = MAX_HARMONIC_QUADRATURE_WORK_UNITS // per_period_work + 1
    stop_ms = periods
    payload["analysis"]["transient"]["stop"] = f"{stop_ms} ms"  # type: ignore[index]
    harmonic = payload["analysis"]["harmonics"]  # type: ignore[index]
    harmonic["start"] = "0 ms"
    harmonic["stop"] = f"{stop_ms} ms"
    harmonic["max_harmonic"] = max_harmonic

    assert 4 * periods * max_harmonic <= MAX_HARMONIC_QUADRATURE_PANELS
    with pytest.raises(ModelError, match="minimum quadrature work estimate"):
        _load(tmp_path, payload)
