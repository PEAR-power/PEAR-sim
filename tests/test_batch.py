# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Batch evaluation: sampling, request validation, ranking and the finalist check."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pear_sim.evidence import batch
from pear_sim.evidence.batch import BatchError, latin_hypercube, run_batch

ROOT = Path(__file__).resolve().parents[1]
LOSS_BUCK = ROOT / "examples" / "buck_with_loss_cards.yaml"
PLAIN_BUCK = ROOT / "examples" / "buck.yaml"


def _request(tmp_path: Path, payload: dict, name: str = "request.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def _base(**overrides) -> dict:
    payload = {
        "schema": "pear-sim/batch-request/v0.1",
        "designs": {
            "method": "list",
            "rows": [
                {"fsw": "10 kHz", "duty": "40 %"},
                {"fsw": "20 kHz", "duty": "40 %"},
                {"fsw": "10 kHz", "duty": "150 %"},
            ],
        },
        "objective": {"minimize": {"losses": "total_mean_power_w"}},
        "constraints": [{"losses": "tj_max_c", "max": 125}],
        "finalists": 2,
    }
    payload.update(overrides)
    return payload


# --- sampling -------------------------------------------------------------


def test_latin_hypercube_fills_every_stratum_exactly_once() -> None:
    count = 37
    sample = latin_hypercube(count, 4, seed=3)
    assert sample.shape == (count, 4)
    for column in sample.T:
        strata = np.floor(column * count).astype(int)
        assert sorted(strata) == list(range(count))


def test_latin_hypercube_is_reproducible_by_seed() -> None:
    assert np.array_equal(latin_hypercube(20, 3, 7), latin_hypercube(20, 3, 7))
    assert not np.array_equal(latin_hypercube(20, 3, 7), latin_hypercube(20, 3, 8))


def test_log_scale_samples_stay_inside_the_range() -> None:
    model = batch.load_model(LOSS_BUCK)
    _method, _names, designs, _spec = batch._expand_designs(
        {
            "method": "lhs",
            "count": 50,
            "seed": 1,
            "parameters": [{"name": "fsw", "from": "5 kHz", "to": "50 kHz", "scale": "log"}],
        },
        model,
    )
    values = [design["fsw"] for design in designs]
    assert 5e3 <= min(values) and max(values) <= 50e3
    # Log stratification: as many designs below 15.8 kHz (geometric middle) as above.
    middle = (5e3 * 50e3) ** 0.5
    assert sum(value < middle for value in values) == 25


def test_grid_is_the_full_product() -> None:
    model = batch.load_model(LOSS_BUCK)
    _method, names, designs, _spec = batch._expand_designs(
        {
            "method": "grid",
            "parameters": [
                {"name": "fsw", "from": "10 kHz", "to": "30 kHz", "points": 3},
                {"name": "duty", "from": "30 %", "to": "40 %", "points": 2},
            ],
        },
        model,
    )
    assert names == ("fsw", "duty")
    assert len(designs) == 6
    assert designs[0] == {"fsw": 10e3, "duty": 0.3}
    assert designs[-1] == {"fsw": 30e3, "duty": 0.4}


# --- request validation ---------------------------------------------------


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"extra": 1}, "unknown field"),
        ({"designs": {"method": "lhs", "count": 2001, "parameters": []}}, "designs.count"),
        (
            {
                "designs": {
                    "method": "grid",
                    "parameters": [
                        {"name": "fsw", "from": "1 kHz", "to": "9 kHz", "points": 50},
                        {"name": "duty", "from": "10 %", "to": "90 %", "points": 50},
                    ],
                }
            },
            "above the 2000-run limit",
        ),
        (
            {
                "designs": {
                    "method": "lhs",
                    "count": 4,
                    "parameters": [{"name": "nope", "from": "1 kHz", "to": "2 kHz"}],
                }
            },
            "not a model parameter",
        ),
        (
            {
                "designs": {
                    "method": "lhs",
                    "count": 4,
                    "parameters": [{"name": "fsw", "from": "2 kHz", "to": "1 kHz"}],
                }
            },
            "'from' must be below 'to'",
        ),
        (
            {"objective": {"minimize": {"losses": "total_mean_power_w", "max": 3}}},
            "takes no bounds",
        ),
        ({"objective": {"lowest": {"losses": "total_mean_power_w"}}}, "objective must be"),
        ({"constraints": [{"signal": "v(out)", "metric": "mean"}]}, "needs 'min', 'max'"),
        ({"constraints": [{"signal": "v(nowhere)", "metric": "mean", "max": 1}]}, "not a model output"),
        ({"constraints": [{"losses": "mean_power_w", "max": 1}]}, "needs a device"),
        ({"finalists": 11}, "finalists must be"),
    ],
)
def test_invalid_requests_are_refused_before_anything_runs(
    tmp_path: Path, change: dict, message: str
) -> None:
    with pytest.raises(BatchError, match=message):
        run_batch(LOSS_BUCK, _request(tmp_path, _base(**change)), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_losses_need_device_cards(tmp_path: Path) -> None:
    payload = _base(
        designs={"method": "list", "rows": [{"load": "5 ohm"}]},
        constraints=[],
    )
    with pytest.raises(BatchError, match="no device cards"):
        run_batch(PLAIN_BUCK, _request(tmp_path, payload), tmp_path / "out")


# --- end to end -----------------------------------------------------------


def test_batch_ranks_feasible_designs_and_reruns_finalists_in_full(tmp_path: Path) -> None:
    artifacts = run_batch(LOSS_BUCK, _request(tmp_path, _base()), tmp_path / "out", jobs=3)
    assert artifacts.status == "pass"
    assert (artifacts.design_count, artifacts.feasible_count, artifacts.failed_count) == (3, 2, 1)
    summary = json.loads(artifacts.summary.read_text())

    # Lower switching frequency, lower switching loss: d0001 wins.
    assert [entry["design_id"] for entry in summary["ranking"]] == ["d0001", "d0002"]
    losses = [entry["values"]["losses.total_mean_power_w"] for entry in summary["ranking"]]
    assert losses == sorted(losses)
    assert summary["failures"][0]["design_id"] == "d0003"

    # Screening keeps only the ranking inputs; finalists keep the full bundle.
    screening = {path.name for path in (tmp_path / "out/designs/d0001/run").iterdir()}
    assert "waveforms.csv" not in screening and "metrics.json" in screening
    assert "waveforms.csv" in {path.name for path in (tmp_path / "out/finalists/d0001/run").iterdir()}
    assert [record["status"] for record in summary["finalists"]] == ["pass", "pass"]

    with artifacts.designs_csv.open() as handle:
        table = list(csv.DictReader(handle))
    assert [row["design_id"] for row in table] == ["d0001", "d0002", "d0003"]
    assert table[0]["rank"] == "1" and table[2]["status"] == "fail"


def test_a_finalist_that_disagrees_with_screening_fails_the_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Simulate a lean shortcut that changed a ranked number in the last bit.
    real_read = batch._read_run

    def skewed(run_dir, *args, **kwargs):
        sha, values = real_read(run_dir, *args, **kwargs)
        if "designs" in run_dir.parts:
            values = {key: np.nextafter(value, np.inf) for key, value in values.items()}
        return sha, values

    monkeypatch.setattr(batch, "_read_run", skewed)
    artifacts = run_batch(
        LOSS_BUCK,
        _request(tmp_path, _base(finalists=1)),
        tmp_path / "out",
        jobs=2,
    )
    assert artifacts.status == "finalist_check_failed"
    finalist = json.loads(artifacts.summary.read_text())["finalists"][0]
    assert finalist["status"] == "screening_mismatch"
    assert finalist["mismatches"]


def test_no_feasible_design_is_its_own_status(tmp_path: Path) -> None:
    payload = _base(
        designs={"method": "list", "rows": [{"fsw": "10 kHz"}]},
        constraints=[{"losses": "tj_max_c", "max": 30}],
    )
    artifacts = run_batch(LOSS_BUCK, _request(tmp_path, payload), tmp_path / "out", jobs=1)
    assert artifacts.status == "no_feasible_design"
    assert artifacts.best_design is None


# --- engine choice --------------------------------------------------------


def test_unknown_backend_is_refused(tmp_path: Path) -> None:
    with pytest.raises(BatchError, match="backend must be"):
        run_batch(LOSS_BUCK, _request(tmp_path, _base(backend="gpu")), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_kernel_is_refused_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pear_sim.core import rust_backend

    monkeypatch.setattr(rust_backend, "_core", None)
    with pytest.raises(BatchError, match="compiled kernel"):
        run_batch(LOSS_BUCK, _request(tmp_path, _base(backend="rust")), tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_run_from_another_engine_is_refused(tmp_path: Path) -> None:
    """A run is accepted only from the exact engine the batch asked for."""

    from pear_sim.evidence.parameter_sweep import ParameterSweepError, _read_verified_metrics
    from pear_sim.spec.model import load_model

    run_batch(LOSS_BUCK, _request(tmp_path, _base(finalists=1)), tmp_path / "out")
    run_dir = tmp_path / "out/finalists/d0001/run"
    model = load_model(tmp_path / "out/source-model/model.yaml", overrides={"fsw": 1e4, "duty": 0.4})
    _read_verified_metrics(run_dir, expected_model=model)  # the engine it ran on
    with pytest.raises(ParameterSweepError, match="not the requested 'rust-core-"):
        _read_verified_metrics(run_dir, expected_model=model, expected_backend="rust-core-0.1.0")


def test_rust_batch_ranks_like_the_reference_engine(tmp_path: Path) -> None:
    pytest.importorskip("pear_sim_core")
    from pear_sim.core import rust_backend

    reference = run_batch(LOSS_BUCK, _request(tmp_path, _base(), "py.json"), tmp_path / "py")
    kernel = run_batch(
        LOSS_BUCK, _request(tmp_path, _base(backend="rust"), "rs.json"), tmp_path / "rs"
    )
    assert kernel.status == "pass"
    py = json.loads(reference.summary.read_text())
    rs = json.loads(kernel.summary.read_text())
    assert py["backend"] == "reference-python"
    assert rs["backend"] == rust_backend.kernel_id()
    # Finalists re-run on the same engine, so they must reproduce screening exactly.
    assert [record["status"] for record in rs["finalists"]] == ["pass", "pass"]
    run = json.loads((tmp_path / "rs/designs/d0001/run/run.json").read_text())
    assert run["backend"] == rust_backend.kernel_id()
    assert [e["design_id"] for e in rs["ranking"]] == [e["design_id"] for e in py["ranking"]]
    for left, right in zip(py["ranking"], rs["ranking"], strict=True):
        for label, value in left["values"].items():
            assert right["values"][label] == pytest.approx(value, rel=1e-10, abs=1e-12)
