# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""The executable signal layer.

The headline test is the signal layer's acceptance criterion: a control law written
ENTIRELY AS DATA must reproduce the one compiled into Python. Everything else
here pins the properties that make such a law trustworthy — a deterministic
order, a refused algebraic loop, a delay that means the previous sample, and
a declared unit rather than an inferred one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.model import load_model
from pear_sim.spec.signal_graph import SignalTask, parse_signal_graph

ROOT = Path(__file__).resolve().parents[1]
BUILTIN = ROOT / "examples" / "buck_closed_loop.yaml"
AS_DATA = ROOT / "examples" / "buck_closed_loop_signals.yaml"

NO_MACHINES: frozenset[str] = frozenset()


def _graph(blocks: dict, period: float = 1e-5):
    return parse_signal_graph(
        {"task": {"period": period}, "blocks": blocks},
        machine_ids=NO_MACHINES,
        parameters={},
    )


# --------------------------------------------------------------------------
# The exit criterion
# --------------------------------------------------------------------------


def test_a_control_law_written_as_data_reproduces_the_compiled_one() -> None:
    """Acceptance: the controller must be expressible with no line in src/.

    `buck_closed_loop.yaml` runs the `sampled_pi_pwm` type, whose law lives in
    Python. `buck_closed_loop_signals.yaml` is the same circuit, the same
    operating point and the same gains, with the controller assembled from
    declared blocks — probe, constant, sum, gain, unit_delay, saturate — and a
    `signal_pwm` bridge. Nothing else differs.

    Agreement is asserted against the SOLVER's own tolerance (rtol 1e-9), not
    a loose engineering band: these are meant to be the same trajectory, and
    a difference larger than the integrator's noise floor would mean the two
    laws are not the same law.
    """

    builtin = HybridSimulator(load_model(BUILTIN)).simulate()
    as_data = HybridSimulator(load_model(AS_DATA)).simulate()

    assert np.array_equal(builtin.time, as_data.time)
    for signal in ("v(out)", "i(l1)"):
        reference = builtin.signals[signal]
        candidate = as_data.signals[signal]
        scale = float(np.max(np.abs(reference)))
        deviation = float(np.max(np.abs(reference - candidate)))
        assert deviation / scale < 1e-6, f"{signal} drifted by {deviation:.3e}"

    # The load step at 25 ms is where a controller difference would show up
    # first, so the run must actually contain it.
    assert float(np.min(as_data.signals["s(duty)"])) < 0.2
    assert float(np.max(as_data.signals["s(duty)"])) > 0.4


def test_the_signal_layer_actually_steered() -> None:
    # Guards against the test above passing because both runs were open loop.
    result = HybridSimulator(load_model(AS_DATA)).simulate()
    duty = result.signals["s(duty)"]
    assert len(set(duty.tolist())) > 100, "duty never moved: the loop is not closed"
    integral = result.signals["s(integral)"]
    assert float(np.max(integral)) > 1e-3, "the accumulator never accumulated"


# --------------------------------------------------------------------------
# Order, loops, and the delay
# --------------------------------------------------------------------------


def test_algebraic_loop_is_refused_and_names_the_cycle() -> None:
    with pytest.raises(ModelError, match="algebraic loop through"):
        _graph(
            {
                "a": {"type": "gain", "input": "b", "gain": 1.0, "unit": "1"},
                "b": {"type": "gain", "input": "a", "gain": 1.0, "unit": "1"},
            }
        )


def test_a_loop_through_unit_delay_is_a_difference_equation_not_an_error() -> None:
    graph = _graph(
        {
            "acc": {"type": "sum", "inputs": ["prev", "one"], "unit": "1"},
            "prev": {"type": "unit_delay", "input": "acc", "unit": "1"},
            "one": {"type": "constant", "value": 1.0, "unit": "1"},
        }
    )
    assert graph is not None
    # The delay must be schedulable before its own input.
    assert graph.order.index("prev") < graph.order.index("acc")


def test_unit_delay_returns_its_INPUT_from_the_previous_tick() -> None:
    """The bug this test exists for.

    A delay that returns its OWN previous value instead of its input's still
    produces numbers and still updates every tick — and an accumulator built
    from it simply never accumulates. It was caught end-to-end (the duty sat
    on its lower limit for a whole run), never by inspection.
    """

    from pear_sim.spec.signal_graph import evaluate_block

    graph = _graph(
        {
            "acc": {"type": "sum", "inputs": ["prev", "one"], "unit": "1"},
            "prev": {"type": "unit_delay", "input": "acc", "unit": "1"},
            "one": {"type": "constant", "value": 1.0, "unit": "1"},
        }
    )
    assert graph is not None
    blocks = graph.by_id
    values: dict[str, float] = {}
    for tick in range(5):
        previous = dict(values)
        resolved: dict[str, float] = {}
        for block_id in graph.order:
            resolved[block_id] = evaluate_block(
                blocks[block_id],
                resolved,
                previous,
                time=tick * 1e-5,
                probe_reader=lambda expression: 0.0,
            )
        values = resolved
    # Counting 1,2,3,4,5 — not stuck at 1.
    assert values["acc"] == pytest.approx(5.0)


def test_execution_order_is_a_function_of_the_graph_not_of_the_file() -> None:
    forward = {
        "src": {"type": "constant", "value": 2.0, "unit": "1"},
        "mid": {"type": "gain", "input": "src", "gain": 3.0, "unit": "1"},
        "out": {"type": "gain", "input": "mid", "gain": 5.0, "unit": "1"},
    }
    reversed_file = dict(reversed(list(forward.items())))
    assert _graph(forward).order == _graph(reversed_file).order


def test_a_block_may_not_read_itself_without_a_delay() -> None:
    with pytest.raises(ModelError, match="reads itself without a delay"):
        _graph({"a": {"type": "gain", "input": "a", "gain": 1.0, "unit": "1"}})


def test_unknown_input_is_named() -> None:
    with pytest.raises(ModelError, match="not a declared block"):
        _graph({"a": {"type": "gain", "input": "ghost", "gain": 1.0, "unit": "1"}})


# --------------------------------------------------------------------------
# Units, probes, and the extrapolation policy
# --------------------------------------------------------------------------


def test_a_probe_may_not_relabel_what_it_reads() -> None:
    with pytest.raises(ModelError, match="may not relabel"):
        _graph({"p": {"type": "probe", "source": "v(out)", "unit": "A"}})


def test_lookup_must_declare_its_extrapolation_policy() -> None:
    base = {
        "x": {"type": "constant", "value": 0.5, "unit": "1"},
        "t": {
            "type": "lookup_1d",
            "input": "x",
            "axis": [0.0, 1.0],
            "values": [10.0, 20.0],
            "unit": "1",
        },
    }
    with pytest.raises(ModelError, match="extrapolation must be declared"):
        _graph(base)


def test_lookup_interpolates_and_honours_its_policy() -> None:
    from pear_sim.spec.signal_graph import evaluate_block

    for policy, argument, expectation in (
        ("hold", 5.0, 20.0),
        ("hold", -1.0, 10.0),
    ):
        graph = _graph(
            {
                "x": {"type": "constant", "value": argument, "unit": "1"},
                "t": {
                    "type": "lookup_1d",
                    "input": "x",
                    "axis": [0.0, 1.0],
                    "values": [10.0, 20.0],
                    "extrapolation": policy,
                    "unit": "1",
                },
            }
        )
        value = evaluate_block(
            graph.by_id["t"], {"x": argument}, {}, time=0.0, probe_reader=lambda e: 0.0
        )
        assert value == pytest.approx(expectation)

    strict = _graph(
        {
            "x": {"type": "constant", "value": 5.0, "unit": "1"},
            "t": {
                "type": "lookup_1d",
                "input": "x",
                "axis": [0.0, 1.0],
                "values": [10.0, 20.0],
                "extrapolation": "error",
                "unit": "1",
            },
        }
    )
    with pytest.raises(ModelError, match="outside the declared axis"):
        evaluate_block(
            strict.by_id["t"], {"x": 5.0}, {}, time=0.0, probe_reader=lambda e: 0.0
        )


def test_axis_must_be_strictly_increasing() -> None:
    with pytest.raises(ModelError, match="strictly increasing"):
        _graph(
            {
                "x": {"type": "constant", "value": 0.0, "unit": "1"},
                "t": {
                    "type": "lookup_1d",
                    "input": "x",
                    "axis": [1.0, 1.0],
                    "values": [1.0, 2.0],
                    "extrapolation": "hold",
                    "unit": "1",
                },
            }
        )


# --------------------------------------------------------------------------
# Clock, bridge, and provenance
# --------------------------------------------------------------------------


def test_task_instants_are_exact_multiples() -> None:
    task = SignalTask(period=1e-5, offset=0.0)
    instants = task.instants(0.0, 5e-5)
    assert len(instants) == 6
    assert instants[3] == pytest.approx(3e-5)


def test_bridge_refuses_a_carrier_that_is_not_the_task_rate(tmp_path: Path) -> None:
    data = yaml.safe_load(AS_DATA.read_text(encoding="utf-8"))
    data["controls"]["ctrl"]["frequency"] = "37 kHz"
    variant = tmp_path / "mismatched.yaml"
    variant.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ModelError, match="must equal the signal task rate"):
        load_model(variant)


def test_probes_must_be_recorded_outputs(tmp_path: Path) -> None:
    data = yaml.safe_load(AS_DATA.read_text(encoding="utf-8"))
    data["outputs"] = [name for name in data["outputs"] if name != "v(out)"]
    variant = tmp_path / "unrecorded.yaml"
    variant.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    with pytest.raises(ModelError, match="must be listed\\s+in outputs"):
        load_model(variant)


def test_the_signal_graph_enters_the_physics_hash(tmp_path: Path) -> None:
    """Two different control laws must not share a hash."""

    from pear_sim.run.contracts import model_hashes

    baseline = model_hashes(load_model(AS_DATA))["model_sha256"]
    data = yaml.safe_load(AS_DATA.read_text(encoding="utf-8"))
    data["signals"]["blocks"]["kp_term"]["gain"] = 0.004
    variant = tmp_path / "other-gains.yaml"
    variant.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    assert model_hashes(load_model(variant))["model_sha256"] != baseline


def test_declared_signals_report_units_and_refuse_invented_integrals() -> None:
    result = HybridSimulator(load_model(AS_DATA)).simulate()
    payload = result.metrics()["s(duty)"]
    assert payload["unit"] == "V"
    assert payload["time_integrated_status"] == "not_computed"
    for key in ("mean", "rms", "integral"):
        with pytest.raises(KeyError):
            payload[key]
    assert payload["sample_count"] > 0


def test_signal_trace_records_every_tick() -> None:
    result = HybridSimulator(load_model(AS_DATA)).simulate()
    trace = result.signal_trace
    assert trace, "no signal trace was retained"
    assert trace[0]["tick_index"] == 0
    assert set(trace[0]["values"]) == {block.id for block in result.model.signals.blocks}
    assert "ctrl" in trace[0]["commanded_duty"]


# --------------------------------------------------------------------------
# Comparison, logic, and the signal-driven branch
#
# The capability v0.1 did not have. `buck_closed_loop_signals.yaml` records
# its own gap in its header: conditional anti-windup was not expressible
# because there was no way to ask a question about a signal. These tests pin
# the answer and, above all, the one type rule that keeps it honest.
# --------------------------------------------------------------------------


def _run_ticks(graph, count: int, probe: float = 0.0) -> dict[str, float]:
    """Evaluate ``graph`` for ``count`` ticks and return the last sample."""

    from pear_sim.spec.signal_graph import evaluate_block

    blocks = graph.by_id
    values: dict[str, float] = {}
    for tick in range(count):
        previous = dict(values)
        resolved: dict[str, float] = {}
        for block_id in graph.order:
            resolved[block_id] = evaluate_block(
                blocks[block_id],
                resolved,
                previous,
                time=tick * 1e-5,
                probe_reader=lambda expression: probe,
            )
        values = resolved
    return values


def test_conditional_anti_windup_is_expressible_and_measurably_holds(
    tmp_path: Path,
) -> None:
    """The headline: the control law v0.1 could not write, and what it buys.

    `buck_closed_loop_antiwindup.yaml` drives a 24 V buck at a 30 V reference
    for 10 ms — physically unreachable, so the limiter pins and the error
    stays positive. The protected loop holds its integrator while the limiter
    is engaged; the unprotected one, built here by replacing exactly one block
    with a plain sum, accumulates a command it can never spend.

    The difference is ASSERTED AS A MEASUREMENT, not narrated: the unprotected
    integrator reaches roughly three times the protected one, and only the
    protected loop returns to its reference band before the run ends.
    """

    guarded_path = ROOT / "examples" / "buck_closed_loop_antiwindup.yaml"

    # One block apart: `integral` integrates unconditionally, so the
    # comparison isolates the guard rather than two different controllers.
    plain = yaml.safe_load(guarded_path.read_text(encoding="utf-8"))
    plain["name"] = "buck_closed_loop_windup"
    plain["signals"]["blocks"]["integral"] = {
        "type": "sum",
        "inputs": ["int_prev", "err_dt"],
        "unit": "V*s",
    }
    for orphan in (
        "int_free",
        "at_upper",
        "at_lower",
        "limited",
        "u_prev",
        "hi_limit",
        "lo_limit",
    ):
        del plain["signals"]["blocks"][orphan]
    plain["outputs"] = [name for name in plain["outputs"] if name != "s(limited)"]
    plain_path = tmp_path / "windup.yaml"
    plain_path.write_text(yaml.safe_dump(plain), encoding="utf-8")

    guarded = HybridSimulator(load_model(guarded_path)).simulate()
    unguarded = HybridSimulator(load_model(plain_path)).simulate()

    guarded_peak = float(np.max(guarded.signals["s(integral)"]))
    plain_peak = float(np.max(unguarded.signals["s(integral)"]))
    assert plain_peak > 2.0 * guarded_peak, (
        f"the guard did not bite: {guarded_peak:.4g} vs {plain_peak:.4g} V*s"
    )

    # The guard must actually have engaged, or the comparison above is between
    # two runs that never saturated and the test proves nothing.
    assert 0.1 < float(np.mean(guarded.signals["s(limited)"])) < 0.9

    # Recovery after the reference returns to 12 V at 15 ms.
    def unsettled_until(result) -> float:
        window = result.time >= 0.016
        times = result.time[window]
        error = np.abs(result.signals["v(out)"][window] - 12.0)
        outside = np.flatnonzero(error > 0.24)
        return float(times[outside[-1]]) if outside.size else float(times[0])

    assert unsettled_until(guarded) < 0.023, "the protected loop never recovered"
    assert unsettled_until(unguarded) >= float(unguarded.time[-1]) - 1e-9, (
        "the unprotected loop recovered too, so this scenario does not "
        "discriminate and the test would pass for the wrong reason"
    )


def test_asking_about_this_tick_command_is_an_algebraic_loop() -> None:
    """Why the example reads the command one tick late, spelled out.

    The naive anti-windup asks whether THIS sample's command is limited, which
    closes integral -> u -> limited -> integral inside one tick. The loader
    must refuse that rather than pick an order, because the answer would
    depend on the order the author happened to write the blocks in.
    """

    with pytest.raises(ModelError, match="algebraic loop through"):
        _graph(
            {
                "err": {"type": "constant", "value": 1.0, "unit": "V"},
                "limit": {"type": "constant", "value": 0.95, "unit": "1"},
                "int_prev": {"type": "unit_delay", "input": "integral", "unit": "V*s"},
                "int_free": {
                    "type": "sum",
                    "inputs": ["int_prev", "err"],
                    "unit": "V*s",
                },
                "integral": {
                    "type": "select",
                    "condition": "limited",
                    "if_true": "int_prev",
                    "if_false": "int_free",
                    "unit": "V*s",
                },
                "u": {"type": "gain", "input": "integral", "gain": 30.0, "unit": "1"},
                "limited": {
                    "type": "compare",
                    "left": "u",
                    "right": "limit",
                    "op": "ge",
                    "unit": "bool",
                },
            }
        )


def test_a_quantity_may_not_be_used_as_a_condition() -> None:
    """The rule that stops a duty ratio from becoming a control decision.

    Without it, any dimensionless signal reaching a condition would be read as
    "true whenever non-zero" — a law nobody wrote, and one that no number
    downstream would look wrong for.
    """

    with pytest.raises(ModelError, match="expects a condition"):
        _graph(
            {
                "duty": {"type": "constant", "value": 0.4, "unit": "1"},
                "a": {"type": "constant", "value": 1.0, "unit": "V"},
                "b": {"type": "constant", "value": 2.0, "unit": "V"},
                "pick": {
                    "type": "select",
                    "condition": "duty",
                    "if_true": "a",
                    "if_false": "b",
                    "unit": "V",
                },
            }
        )


def test_a_truth_value_may_feed_arithmetic() -> None:
    """The check runs one way only, and that is deliberate.

    Multiplying a term by a 0/1 gate is how a contribution gets switched off.
    Refusing it would ban a standard technique to enforce a rule that only
    protects condition inputs.
    """

    graph = _graph(
        {
            "x": {"type": "constant", "value": 3.0, "unit": "V"},
            "zero": {"type": "constant", "value": 0.0, "unit": "V"},
            "gate": {
                "type": "compare",
                "left": "x",
                "right": "zero",
                "op": "gt",
                "unit": "bool",
            },
            "gated": {"type": "product", "inputs": ["x", "gate"], "unit": "V"},
        }
    )
    assert _run_ticks(graph, 1)["gated"] == pytest.approx(3.0)


def test_comparison_and_logic_carry_bool_and_may_not_be_relabelled() -> None:
    with pytest.raises(ModelError, match="not a physical quantity"):
        _graph(
            {
                "a": {"type": "constant", "value": 1.0, "unit": "V"},
                "b": {"type": "constant", "value": 2.0, "unit": "V"},
                "c": {
                    "type": "compare",
                    "left": "a",
                    "right": "b",
                    "op": "lt",
                    "unit": "V",
                },
            }
        )


def test_equality_is_not_offered_and_the_refusal_says_why() -> None:
    """Two floats reached by different arithmetic are equal only by luck."""

    with pytest.raises(ModelError, match="equality is not offered"):
        _graph(
            {
                "a": {"type": "constant", "value": 1.0, "unit": "V"},
                "b": {"type": "constant", "value": 1.0, "unit": "V"},
                "c": {
                    "type": "compare",
                    "left": "a",
                    "right": "b",
                    "op": "eq",
                    "unit": "bool",
                },
            }
        )


def test_logic_arity_is_enforced_per_operator() -> None:
    def graph_for(op: str, inputs: list[str]):
        blocks: dict = {
            "x": {"type": "constant", "value": 1.0, "unit": "V"},
            "zero": {"type": "constant", "value": 0.0, "unit": "V"},
            "p": {
                "type": "compare",
                "left": "x",
                "right": "zero",
                "op": "gt",
                "unit": "bool",
            },
            "q": {
                "type": "compare",
                "left": "x",
                "right": "zero",
                "op": "lt",
                "unit": "bool",
            },
        }
        blocks["r"] = {"type": "logic", "op": op, "inputs": inputs, "unit": "bool"}
        return _graph(blocks)

    with pytest.raises(ModelError, match="exactly 1 block"):
        graph_for("not", ["p", "q"])
    with pytest.raises(ModelError, match="between 2 and 16 block"):
        graph_for("or", ["p"])
    assert graph_for("not", ["p"]) is not None
    assert graph_for("and", ["p", "q"]) is not None


def test_comparison_logic_and_select_evaluate_as_declared() -> None:
    graph = _graph(
        {
            "two": {"type": "constant", "value": 2.0, "unit": "V"},
            "three": {"type": "constant", "value": 3.0, "unit": "V"},
            "lt": {
                "type": "compare",
                "left": "two",
                "right": "three",
                "op": "lt",
                "unit": "bool",
            },
            "ge": {
                "type": "compare",
                "left": "two",
                "right": "three",
                "op": "ge",
                "unit": "bool",
            },
            "both": {"type": "logic", "op": "and", "inputs": ["lt", "ge"], "unit": "bool"},
            "either": {"type": "logic", "op": "or", "inputs": ["lt", "ge"], "unit": "bool"},
            "neither": {"type": "logic", "op": "not", "inputs": ["either"], "unit": "bool"},
            "picked": {
                "type": "select",
                "condition": "lt",
                "if_true": "two",
                "if_false": "three",
                "unit": "V",
            },
        }
    )
    values = _run_ticks(graph, 1)
    assert (values["lt"], values["ge"]) == (1.0, 0.0)
    assert (values["both"], values["either"], values["neither"]) == (0.0, 1.0, 0.0)
    assert values["picked"] == pytest.approx(2.0)


def test_select_follows_the_state_where_switch_follows_the_experiment() -> None:
    """The two branches are not interchangeable, and this is the difference.

    `switch` reads a model-level flag fixed before the first tick: which law
    this run exercises. `select` reads a signal: which branch this sample
    takes. Only the first belongs in a comparison between two runs.
    """

    common: dict = {
        "count": {"type": "sum", "inputs": ["prev", "one"], "unit": "1"},
        "prev": {"type": "unit_delay", "input": "count", "unit": "1"},
        "one": {"type": "constant", "value": 1.0, "unit": "1"},
        "threshold": {"type": "constant", "value": 2.5, "unit": "1"},
        "low": {"type": "constant", "value": 10.0, "unit": "V"},
        "high": {"type": "constant", "value": 20.0, "unit": "V"},
        "past": {
            "type": "compare",
            "left": "prev",
            "right": "threshold",
            "op": "gt",
            "unit": "bool",
        },
        "by_state": {
            "type": "select",
            "condition": "past",
            "if_true": "high",
            "if_false": "low",
            "unit": "V",
        },
        "by_experiment": {
            "type": "switch",
            "flag": False,
            "if_true": "high",
            "if_false": "low",
            "unit": "V",
        },
    }
    graph = _graph(common)
    assert _run_ticks(graph, 2)["by_state"] == pytest.approx(10.0)
    assert _run_ticks(graph, 6)["by_state"] == pytest.approx(20.0)
    # The flag never moves, however long the run.
    assert _run_ticks(graph, 2)["by_experiment"] == pytest.approx(10.0)
    assert _run_ticks(graph, 6)["by_experiment"] == pytest.approx(10.0)
