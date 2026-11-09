# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pytest

from pear_sim.errors import ModelError
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator


# Every timing value in the golden fixture is an integer multiple of this binary-
# exact quantum.  The oracle therefore exercises event construction without hiding
# platform drift behind an approximate decimal-time comparison.
Q = 2.0**-19
PERIOD = 64.0 * Q
PHASE = 8.0 * Q
STOP = 100.0 * Q


def _seconds(value: float) -> str:
    return f"{value:.17g} s"


def _write_complementary_model(
    tmp_path: Path,
    *,
    duty: float = 3.0 / 8.0,
    dead_time: str | None = None,
    name: str = "complementary_pwm_golden",
) -> Path:
    if dead_time is None:
        dead_time = _seconds(Q)
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"""
schema: pear-sim/v0.1
name: {name}
circuit:
  ground: gnd
  components:
    - {{id: vin, type: voltage_source, p: dc, n: gnd, value: 100 V}}
    - {{id: qh, type: ideal_switch, p: dc, n: sw, control: leg.high}}
    - {{id: ql, type: ideal_switch, p: sw, n: gnd, control: leg.low}}
    - {{id: r1, type: resistor, p: sw, n: out, value: 10 ohm}}
    - {{id: c1, type: capacitor, p: out, n: gnd, value: 1 mF, initial_voltage: 0 V}}
controls:
  leg:
    type: complementary_pwm
    frequency: 8192 Hz
    duty: {duty:.17g}
    phase: {_seconds(PHASE)}
    dead_time: {dead_time}
analysis:
  transient:
    start: 0 s
    stop: {_seconds(STOP)}
    output_step: {_seconds(Q)}
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11
outputs: [v(out), v(qh), v(ql)]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_legacy_pwm_model(tmp_path: Path) -> Path:
    path = tmp_path / "legacy-pwm.yaml"
    path.write_text(
        f"""
schema: pear-sim/v0.1
name: legacy_pwm_golden
circuit:
  ground: gnd
  components:
    - {{id: vin, type: voltage_source, p: dc, n: gnd, value: 100 V}}
    - {{id: q1, type: ideal_switch, p: dc, n: sw, control: gate}}
    - {{id: r1, type: resistor, p: sw, n: out, value: 10 ohm}}
    - {{id: c1, type: capacitor, p: out, n: gnd, value: 1 mF, initial_voltage: 0 V}}
controls:
  gate:
    type: pwm
    frequency: 8192 Hz
    duty: 0.375
    phase: {_seconds(PHASE)}
analysis:
  transient:
    start: 0 s
    stop: {_seconds(STOP)}
    output_step: {_seconds(Q)}
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11
outputs: [v(out), v(q1)]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _scheduled_events(result):
    return [event for event in result.events if event.kind == "scheduled_gate"]


def _assert_mode_intervals(result, expected) -> None:
    observed = []
    for interval in result.mode_intervals:
        gates = interval["gates"]
        high = gates["leg.high"]
        low = gates["leg.low"]
        assert not (high and low)
        observed.append(
            (
                interval["start_s"] / Q,
                interval["end_s"] / Q,
                high,
                low,
            )
        )
    assert observed == expected


def test_complementary_pwm_binary_exact_golden_sequence(tmp_path: Path) -> None:
    model = load_model(_write_complementary_model(tmp_path))
    assert tuple(model.controls) == ("leg.high", "leg.low")
    high = model.controls["leg.high"]
    low = model.controls["leg.low"]
    assert (
        high.kind,
        high.block_id,
        high.role,
        high.partner_id,
        high.period,
        high.duty,
        high.phase,
        high.dead_time,
        high.command_duty,
        high.command_phase,
    ) == (
        "complementary_pwm",
        "leg",
        "high",
        "leg.low",
        PERIOD,
        23.0 / 64.0,
        9.0 * Q,
        Q,
        3.0 / 8.0,
        PHASE,
    )
    assert (low.role, low.partner_id, low.duty, low.phase) == (
        "low",
        "leg.high",
        39.0 / 64.0,
        33.0 * Q,
    )

    result = HybridSimulator(model).simulate()

    observed = [
        (event.time / Q, event.target, event.previous, event.current)
        for event in _scheduled_events(result)
    ]
    assert observed == [
        (8.0, "leg.low", True, False),
        (9.0, "leg.high", False, True),
        (32.0, "leg.high", True, False),
        (33.0, "leg.low", False, True),
        (72.0, "leg.low", True, False),
        (73.0, "leg.high", False, True),
        (96.0, "leg.high", True, False),
        (97.0, "leg.low", False, True),
    ]
    assert result.stats["scheduled_events"] == 8
    _assert_mode_intervals(
        result,
        [
            (0.0, 8.0, False, True),
            (8.0, 9.0, False, False),
            (9.0, 32.0, True, False),
            (32.0, 33.0, False, False),
            (33.0, 72.0, False, True),
            (72.0, 73.0, False, False),
            (73.0, 96.0, True, False),
            (96.0, 97.0, False, False),
            (97.0, 100.0, False, True),
        ],
    )


def test_zero_dead_time_swaps_both_gates_atomically_without_overlap(
    tmp_path: Path,
) -> None:
    result = HybridSimulator(
        load_model(_write_complementary_model(tmp_path, dead_time="0 s", name="zero_dead_time"))
    ).simulate()

    groups = defaultdict(list)
    for event in _scheduled_events(result):
        groups[event.time / Q].append((event.target, event.previous, event.current))

    # Records in one timestamp use the documented off-before-on order. The
    # absence of an intermediate mode interval is what proves the swap was atomic;
    # consumers must not interpret prefixes of this serialized group as topology.
    assert dict(groups) == {
        8.0: [
            ("leg.low", True, False),
            ("leg.high", False, True),
        ],
        32.0: [
            ("leg.high", True, False),
            ("leg.low", False, True),
        ],
        72.0: [
            ("leg.low", True, False),
            ("leg.high", False, True),
        ],
        96.0: [
            ("leg.high", True, False),
            ("leg.low", False, True),
        ],
    }
    _assert_mode_intervals(
        result,
        [
            (0.0, 8.0, False, True),
            (8.0, 32.0, True, False),
            (32.0, 72.0, False, True),
            (72.0, 96.0, True, False),
            (96.0, 100.0, False, True),
        ],
    )


@pytest.mark.parametrize(
    ("duty", "expected_high", "expected_low"),
    [(0.0, False, True), (1.0, True, False)],
)
def test_endpoint_duties_are_constant_and_schedule_no_edges(
    tmp_path: Path,
    duty: float,
    expected_high: bool,
    expected_low: bool,
) -> None:
    result = HybridSimulator(
        load_model(
            _write_complementary_model(
                tmp_path,
                duty=duty,
                dead_time="0 s",
                name=f"endpoint_duty_{int(duty)}",
            )
        )
    ).simulate()

    assert _scheduled_events(result) == []
    assert result.stats["scheduled_events"] == 0
    _assert_mode_intervals(
        result,
        [(0.0, 100.0, expected_high, expected_low)],
    )


@pytest.mark.parametrize(
    "dead_time",
    [
        "-1 s",
        # D*T is 24Q. Equality would create a zero-width high-side pulse and is
        # deliberately rejected instead of depending on same-timestamp ordering.
        _seconds(24.0 * Q),
        _seconds(25.0 * Q),
        "1 V",
    ],
)
def test_invalid_dead_time_is_rejected_by_the_public_model_loader(
    tmp_path: Path,
    dead_time: str,
) -> None:
    path = _write_complementary_model(
        tmp_path,
        dead_time=dead_time,
        name="invalid_dead_time",
    )

    with pytest.raises(ModelError, match="dead.time|expected time"):
        load_model(path)


@pytest.mark.parametrize("duty", [0.0, 1.0])
def test_endpoint_duty_rejects_a_nonzero_dead_time(
    tmp_path: Path,
    duty: float,
) -> None:
    path = _write_complementary_model(
        tmp_path,
        duty=duty,
        dead_time=_seconds(Q),
        name=f"invalid_endpoint_dead_time_{int(duty)}",
    )

    with pytest.raises(ModelError, match="dead.time must be zero"):
        load_model(path)


def test_legacy_pwm_event_semantics_do_not_regress(tmp_path: Path) -> None:
    result = HybridSimulator(load_model(_write_legacy_pwm_model(tmp_path))).simulate()

    assert [
        (event.time / Q, event.target, event.previous, event.current)
        for event in _scheduled_events(result)
    ] == [
        (8.0, "gate", False, True),
        (32.0, "gate", True, False),
        (72.0, "gate", False, True),
        (96.0, "gate", True, False),
    ]
    assert result.stats["scheduled_events"] == 4


def test_expanded_gate_outputs_respect_the_portable_report_limit(
    tmp_path: Path,
) -> None:
    controls = "\n".join(
        f"  leg_{index}:\n"
        "    type: complementary_pwm\n"
        "    frequency: 10 kHz\n"
        "    duty: 0\n"
        "    dead_time: 0 s"
        for index in range(129)
    )
    path = tmp_path / "too-many-expanded-gates.yaml"
    path.write_text(
        f"""
schema: pear-sim/v0.1
name: too_many_expanded_gates
circuit:
  ground: gnd
  components:
    - {{id: r1, type: resistor, p: out, n: gnd, value: 10 ohm}}
controls:
{controls}
analysis:
  transient:
    start: 0 s
    stop: 1 us
    output_step: 100 ns
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11
outputs: [v(r1)]
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ModelError,
        match=r"expand to 258 gate outputs.*limit is 256",
    ):
        load_model(path)
