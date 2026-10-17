"""Measurement chain: noise, quantisation and computational delay.

What a controller SEES stops being what IS. These three blocks are the rest
of the measurement chain — they were deliberately deferred until the signal
layer existed, because before that there was nowhere to put them.

The properties that matter here are not "does it add noise" but: is the run
still reproducible, is the delay really N ticks rather than one, and does a
chain that declares nothing behave exactly as if it were absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pear_sim.errors import ModelError
from pear_sim.spec.signal_graph import evaluate_block, parse_signal_graph

ROOT = Path(__file__).resolve().parents[1]
NO_MACHINES: frozenset[str] = frozenset()


def _graph(blocks: dict, period: float = 1e-5):
    return parse_signal_graph(
        {"task": {"period": period}, "blocks": blocks},
        machine_ids=NO_MACHINES,
        parameters={},
    )


def _run(graph, ticks: int, probe=lambda expression: 0.0):
    """Evaluate a graph for N ticks, keeping delay buffers like the solver."""

    values: dict[str, float] = {}
    buffers: dict[str, list[float]] = {}
    history = []
    for tick in range(ticks):
        previous = dict(values)
        resolved: dict[str, float] = {}
        for block_id in graph.order:
            resolved[block_id] = evaluate_block(
                graph.by_id[block_id],
                resolved,
                previous,
                time=tick * 1e-5,
                probe_reader=probe,
                tick=tick,
                buffers=buffers,
            )
        for block in graph.blocks:
            if block.kind != "delay":
                continue
            queue = buffers.setdefault(block.id, [])
            queue.append(resolved[block.inputs[0]])
            depth = int(block.params["samples"])
            if len(queue) > depth:
                del queue[: len(queue) - depth]
        values = resolved
        history.append(resolved)
    return history


# --------------------------------------------------------------------------
# Noise: reproducible or absent
# --------------------------------------------------------------------------


def test_a_noise_block_must_declare_its_seed() -> None:
    """The promise is seed -> bit, and a generator with no stated seed
    cannot keep it. 'It was random' is not a reproducible run."""

    with pytest.raises(ModelError, match="seed must be declared"):
        _graph({"n": {"type": "noise", "stddev": 1.0, "unit": "A"}})


def test_the_same_seed_gives_the_same_samples_and_a_different_one_does_not() -> None:
    def series(seed: int):
        graph = _graph(
            {"n": {"type": "noise", "stddev": 2.0, "seed": seed, "unit": "A"}}
        )
        return [row["n"] for row in _run(graph, 20)]

    assert series(7) == series(7)
    assert series(7) != series(8)


def test_a_noise_sample_depends_on_the_tick_not_on_the_call_count() -> None:
    """Evaluating the graph twice must not advance the noise.

    A generator carrying internal state would return different numbers on a
    second pass over the same tick, which would make a run's result depend
    on how many times something happened to be recomputed.
    """

    graph = _graph({"n": {"type": "noise", "stddev": 1.0, "seed": 3, "unit": "A"}})
    block = graph.by_id["n"]
    first = evaluate_block(block, {}, {}, time=0.0, probe_reader=lambda e: 0.0, tick=5)
    second = evaluate_block(block, {}, {}, time=0.0, probe_reader=lambda e: 0.0, tick=5)
    assert first == second
    other = evaluate_block(block, {}, {}, time=0.0, probe_reader=lambda e: 0.0, tick=6)
    assert other != first


def test_zero_stddev_is_exactly_zero() -> None:
    """A declared-but-silent noise source must not perturb anything."""

    graph = _graph({"n": {"type": "noise", "stddev": 0.0, "seed": 1, "unit": "A"}})
    assert all(row["n"] == 0.0 for row in _run(graph, 10))


def test_noise_has_roughly_the_declared_spread() -> None:
    import statistics

    graph = _graph({"n": {"type": "noise", "stddev": 3.0, "seed": 11, "unit": "A"}})
    samples = [row["n"] for row in _run(graph, 4000)]
    assert statistics.fmean(samples) == pytest.approx(0.0, abs=0.15)
    assert statistics.pstdev(samples) == pytest.approx(3.0, rel=0.05)


# --------------------------------------------------------------------------
# Quantisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "value", "expected"),
    [
        ("round", 1.24, 1.25),
        ("round", 1.26, 1.25),
        ("round", -1.24, -1.25),
        ("truncate", 1.49, 1.25),
        ("truncate", -1.01, -1.25),
    ],
)
def test_quantise_snaps_to_the_declared_step(mode, value, expected) -> None:
    graph = _graph(
        {
            "x": {"type": "constant", "value": value, "unit": "A"},
            "q": {"type": "quantize", "input": "x", "step": 0.25, "mode": mode,
                  "unit": "A"},
        }
    )
    assert _run(graph, 1)[0]["q"] == pytest.approx(expected)


def test_quantise_needs_a_positive_step() -> None:
    with pytest.raises(ModelError, match="step must be positive"):
        _graph(
            {
                "x": {"type": "constant", "value": 1.0, "unit": "A"},
                "q": {"type": "quantize", "input": "x", "step": 0.0, "unit": "A"},
            }
        )


# --------------------------------------------------------------------------
# Computational delay
# --------------------------------------------------------------------------


def test_delay_is_n_ticks_not_one() -> None:
    """The bug this guards: a multi-tick delay collapsing to unit_delay.

    A controller's computational delay is usually one or two SAMPLES, and
    getting it wrong by one sample is a phase error no aggregate metric
    reveals.
    """

    graph = _graph(
        {
            "t": {"type": "time_source", "shape": "ramp", "initial": 0.0,
                  "final": 100.0, "start": 0.0, "stop": 1e-3, "unit": "1"},
            "d": {"type": "delay", "input": "t", "samples": 3, "unit": "1"},
        }
    )
    rows = _run(graph, 8)
    for index in range(3, 8):
        assert rows[index]["d"] == pytest.approx(rows[index - 3]["t"])


def test_delay_reports_its_initial_value_until_the_buffer_fills() -> None:
    graph = _graph(
        {
            "one": {"type": "constant", "value": 5.0, "unit": "1"},
            "d": {"type": "delay", "input": "one", "samples": 4, "initial": -2.0,
                  "unit": "1"},
        }
    )
    rows = _run(graph, 6)
    assert [row["d"] for row in rows[:4]] == [-2.0, -2.0, -2.0, -2.0]
    assert rows[4]["d"] == pytest.approx(5.0)


def test_a_delay_may_close_a_loop_like_unit_delay() -> None:
    graph = _graph(
        {
            "acc": {"type": "sum", "inputs": ["prev", "one"], "unit": "1"},
            "prev": {"type": "delay", "input": "acc", "samples": 2, "unit": "1"},
            "one": {"type": "constant", "value": 1.0, "unit": "1"},
        }
    )
    assert graph is not None
    assert graph.order.index("prev") < graph.order.index("acc")


def test_delay_samples_must_be_a_positive_integer() -> None:
    for samples in (0, -1, 1.5):
        with pytest.raises(ModelError, match="samples must be an integer"):
            _graph(
                {
                    "x": {"type": "constant", "value": 1.0, "unit": "1"},
                    "d": {"type": "delay", "input": "x", "samples": samples,
                          "unit": "1"},
                }
            )
