from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

import jsonschema
import pytest

import pear_sim.parameter_sweep as parameter_sweep_module
from pear_sim.parameter_sweep import ParameterSweepError, run_parameter_sweep
from pear_sim.evidence.parallel_points import run_ordered
from pear_sim.evidence.worker_pool import PointWorkerPool, WorkerReply, WorkerTimeout


ROOT = Path(__file__).parents[1]


def _write_model(path: Path) -> Path:
    path.write_text(
        """schema: pear-sim/v0.1
name: rc_parameter_fixture
parameters:
  load: 10 ohm
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 12 V}
    - {id: r1, type: resistor, p: in, n: out, value: "${load}"}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 1 uF, initial_voltage: 0 V}
controls: {}
analysis:
  transient:
    start: 0 s
    stop: 50 us
    output_step: 1 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11
outputs:
  - v(out)
  - i(r1)
""",
        encoding="utf-8",
    )
    return path


def _write_request(
    path: Path,
    *,
    values: list[object] | None = None,
    projections: list[dict[str, str]] | None = None,
    parameter: str = "load",
) -> Path:
    payload = {
        "schema": "pear-sim/parameter-sweep-request/v0.1",
        "parameter": parameter,
        "values": values if values is not None else ["10 ohm", "100 ohm"],
        "projections": (
            projections
            if projections is not None
            else [
                {
                    "id": "vout_final",
                    "signal": "v(out)",
                    "metric": "sample_final",
                    "unit": "V",
                }
            ]
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def completed_sweep(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("parameter-sweep")
    return run_parameter_sweep(
        _write_model(root / "model.yaml"),
        _write_request(root / "request.json"),
        root / "output",
    )


def test_sweep_retains_all_runs_and_separates_execution_from_acceptance(
    completed_sweep,
) -> None:
    assert completed_sweep.execution_status == "pass"
    assert completed_sweep.acceptance_status == "not_run"
    assert completed_sweep.accepted is None
    assert completed_sweep.overall_status == "pass"

    summary = json.loads(completed_sweep.sweep_summary.read_text(encoding="utf-8"))
    normalized = json.loads(completed_sweep.normalized_request.read_text(encoding="utf-8"))
    projections = json.loads(completed_sweep.projections.read_text(encoding="utf-8"))
    failures = json.loads(completed_sweep.failures.read_text(encoding="utf-8"))

    assert summary["schema"] == "pear-sim/parameter-sweep-summary/v0.1"
    assert summary["summary"] == {
        "acceptance_status_counts": {"not_run": 2},
        "execution_status_counts": {"pass": 2},
        "failure_count": 0,
        "overall_status_counts": {"pass": 2},
        "point_count": 2,
        "projection_record_status_counts": {"recorded": 2},
        "projection_selector_count": 1,
    }
    assert [point["parameter_value_si"] for point in summary["points"]] == [10.0, 100.0]
    assert [point["index"] for point in summary["points"]] == [0, 1]
    assert all(point["accepted"] is None for point in summary["points"])
    assert len(list((completed_sweep.output_dir / "points").glob("*/run/run.json"))) == 2
    assert normalized["parameter"] == {
        "base_value_si": 10.0,
        "dimension": "resistance",
        "name": "load",
    }
    assert normalized["values"] == [
        {"index": 0, "value_si": 10.0},
        {"index": 1, "value_si": 100.0},
    ]
    values = projections["projections"][0]["points"]
    assert [item["status"] for item in values] == ["recorded", "recorded"]
    assert values[0]["value"] > values[1]["value"]
    assert failures["failures"] == []

    first_run = next((completed_sweep.output_dir / "points").glob("*/run"))
    metrics = json.loads((first_run / "metrics.json").read_text(encoding="utf-8"))
    waveform_rows = len((first_run / "waveforms.csv").read_text(encoding="utf-8").splitlines()) - 1
    assert metrics["v(out)"]["sample_count"] == waveform_rows
    assert metrics["i(r1)"]["sample_count"] == waveform_rows


def test_source_request_is_retained_and_semantic_hash_ignores_json_formatting(
    tmp_path: Path,
) -> None:
    model = _write_model(tmp_path / "model.yaml")
    first_request = _write_request(
        tmp_path / "first-request.json",
        values=["10 ohm"],
    )
    payload = json.loads(first_request.read_text(encoding="utf-8"))
    second_request = tmp_path / "second-request.json"
    second_request.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")

    first = run_parameter_sweep(model, first_request, tmp_path / "first")
    second = run_parameter_sweep(model, second_request, tmp_path / "second")
    first_summary = json.loads(first.sweep_summary.read_text(encoding="utf-8"))
    second_summary = json.loads(second.sweep_summary.read_text(encoding="utf-8"))

    assert (first.output_dir / "request.source.json").read_bytes() == first_request.read_bytes()
    assert (second.output_dir / "request.source.json").read_bytes() == second_request.read_bytes()
    assert first_summary["provenance"]["request_source_sha256"] != (
        second_summary["provenance"]["request_source_sha256"]
    )
    assert first_summary["provenance"]["request_semantic_sha256"] == (
        second_summary["provenance"]["request_semantic_sha256"]
    )


def test_root_manifest_exactly_hashes_every_nested_regular_artifact(completed_sweep) -> None:
    manifest = json.loads(completed_sweep.artifact_manifest.read_text(encoding="utf-8"))
    paths = [record["path"] for record in manifest["artifacts"]]
    actual = sorted(
        path.relative_to(completed_sweep.output_dir).as_posix()
        for path in completed_sweep.output_dir.rglob("*")
        if path.is_file() and path != completed_sweep.artifact_manifest
    )

    assert manifest["subject"] == {
        "kind": "parameter-sweep",
        "sha256": hashlib.sha256(completed_sweep.sweep_summary.read_bytes()).hexdigest(),
    }
    assert paths == sorted(paths) == actual
    assert "request.normalized.json" in paths
    assert "request.source.json" in paths
    assert "source-model/model.yaml" in paths
    assert "points/point-000-" in "\n".join(paths)
    assert "report/parameter-sweep-report.html" in paths
    assert "report/schematic.svg" in paths
    for record in manifest["artifacts"]:
        path = completed_sweep.output_dir / record["path"]
        assert record["bytes"] == path.stat().st_size
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_request_and_summary_match_published_json_schemas(completed_sweep) -> None:
    request_schema = json.loads(
        (ROOT / "schema" / "parameter-sweep-request.schema.json").read_text(encoding="utf-8")
    )
    summary_schema = json.loads(
        (ROOT / "schema" / "parameter-sweep-summary.schema.json").read_text(encoding="utf-8")
    )
    request = {
        "schema": "pear-sim/parameter-sweep-request/v0.1",
        "parameter": "load",
        "values": ["10 ohm", 100.0],
        "projections": [
            {
                "id": "vout_final",
                "signal": "v(out)",
                "metric": "sample_final",
                "unit": "V",
            }
        ],
    }
    summary = json.loads(completed_sweep.sweep_summary.read_text(encoding="utf-8"))

    jsonschema.Draft202012Validator.check_schema(request_schema)
    jsonschema.Draft202012Validator.check_schema(summary_schema)
    jsonschema.validate(request, request_schema)
    jsonschema.validate(summary, summary_schema)

    whitespace_id = json.loads(json.dumps(request))
    whitespace_id["projections"][0]["id"] = " vout_final "
    whitespace_signal = json.loads(json.dumps(request))
    whitespace_signal["projections"][0]["signal"] = " v(out) "
    embedded_newline_id = json.loads(json.dumps(request))
    embedded_newline_id["projections"][0]["id"] = "vout\nfinal"
    trailing_newline_id = json.loads(json.dumps(request))
    trailing_newline_id["projections"][0]["id"] = "vout_final\n"
    trailing_newline_parameter = json.loads(json.dumps(request))
    trailing_newline_parameter["parameter"] = "load\n"
    trailing_newline_signal = json.loads(json.dumps(request))
    trailing_newline_signal["projections"][0]["signal"] = "v(out)\n"
    assert not jsonschema.Draft202012Validator(request_schema).is_valid(whitespace_id)
    assert not jsonschema.Draft202012Validator(request_schema).is_valid(
        whitespace_signal
    )
    assert not jsonschema.Draft202012Validator(request_schema).is_valid(
        embedded_newline_id
    )
    validator = jsonschema.Draft202012Validator(request_schema)
    assert not validator.is_valid(trailing_newline_id)
    assert not validator.is_valid(trailing_newline_parameter)
    assert not validator.is_valid(trailing_newline_signal)


def test_report_is_offline_csp_bound_and_copies_authoritative_documents(
    completed_sweep,
) -> None:
    report = json.loads(completed_sweep.report_json.read_text(encoding="utf-8"))
    html_text = completed_sweep.report_html.read_text(encoding="utf-8")
    summary = json.loads(completed_sweep.sweep_summary.read_text(encoding="utf-8"))
    projections = json.loads(completed_sweep.projections.read_text(encoding="utf-8"))
    failures = json.loads(completed_sweep.failures.read_text(encoding="utf-8"))

    assert report["recorded"] == {
        "failures": failures,
        "projections": projections,
        "sweep_summary": summary,
    }
    assert report["claim"]["validation_claim"] is False
    assert "does not recompute" in report["claim"]["statement"]
    assert "EXPLORATORY — NOT VALIDATION" in html_text
    assert "script-src 'none'" in html_text
    assert "connect-src 'none'" in html_text
    assert "no scripts or network resources" in html_text
    assert "<script" not in html_text.lower()


def test_optional_expectations_evaluate_each_verified_run(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "model.yaml")
    request = _write_request(tmp_path / "request.json")
    expectations = tmp_path / "expectations.json"
    expectations.write_text(
        json.dumps(
            {
                "schema": "pear-sim/expectations/v0.1",
                "criteria": [
                    {
                        "id": "minimum_final_output",
                        "signal": "v(out)",
                        "metric": "sample_final",
                        "unit": "V",
                        "minimum": 8.0,
                        "blocking": True,
                    }
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    artifacts = run_parameter_sweep(
        model,
        request,
        tmp_path / "output",
        expectations_file=expectations,
    )
    summary = json.loads(artifacts.sweep_summary.read_text(encoding="utf-8"))

    assert artifacts.execution_status == "pass"
    assert artifacts.acceptance_status == "fail"
    assert artifacts.accepted is False
    assert artifacts.overall_status == "fail"
    assert [point["acceptance_status"] for point in summary["points"]] == ["pass", "fail"]
    assert [point["accepted"] for point in summary["points"]] == [True, False]
    assert len(list((artifacts.output_dir / "points").glob("*/evidence/evidence.json"))) == 2
    assert summary["expectations"] is not None
    assert (
        artifacts.output_dir / "expectations.source.json"
    ).read_bytes() == expectations.read_bytes()


def test_point_override_failure_is_retained_and_later_points_continue(tmp_path: Path) -> None:
    artifacts = run_parameter_sweep(
        _write_model(tmp_path / "model.yaml"),
        _write_request(
            tmp_path / "request.json",
            values=["10 ohm", "-1 ohm", "20 ohm"],
        ),
        tmp_path / "output",
    )
    summary = json.loads(artifacts.sweep_summary.read_text(encoding="utf-8"))
    failures = json.loads(artifacts.failures.read_text(encoding="utf-8"))

    assert artifacts.execution_status == "fail"
    assert artifacts.acceptance_status == "not_run"
    assert artifacts.accepted is None
    assert artifacts.overall_status == "fail"
    assert [point["execution_status"] for point in summary["points"]] == [
        "pass",
        "fail",
        "pass",
    ]
    failed = summary["points"][1]
    assert failed["artifacts"]["run"] is None
    assert failed["projections"][0]["status"] == "not_run"
    assert len(list((artifacts.output_dir / "points").glob("*/run/run.json"))) == 2
    assert failures["summary"] == {"count": 1, "kind_counts": {"execution": 1}}
    assert failures["failures"][0]["stage"] == "model_override"
    assert "must be positive" in failures["failures"][0]["message"]


def test_nested_run_must_verify_before_projection_or_execution_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = parameter_sweep_module._run_point_worker

    def tampering_point_worker(*args, **kwargs):
        destination = original(*args, **kwargs)
        metrics_path = destination / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["v(out)"]["sample_final"] = 999.0
        metrics_path.write_text(json.dumps(metrics) + "\n", encoding="utf-8")
        return destination

    monkeypatch.setattr(
        parameter_sweep_module,
        "_run_point_worker",
        tampering_point_worker,
    )
    artifacts = run_parameter_sweep(
        _write_model(tmp_path / "model.yaml"),
        _write_request(tmp_path / "request.json", values=["10 ohm"]),
        tmp_path / "output",
    )
    summary = json.loads(artifacts.sweep_summary.read_text(encoding="utf-8"))
    failures = json.loads(artifacts.failures.read_text(encoding="utf-8"))

    assert artifacts.execution_status == "fail"
    assert artifacts.overall_status == "fail"
    assert summary["points"][0]["execution_status"] == "fail"
    assert summary["points"][0]["projections"][0]["status"] == "not_run"
    assert "mismatch" in failures["failures"][0]["message"]


def test_requested_projection_failure_blocks_overall_pass(tmp_path: Path) -> None:
    request = _write_request(
        tmp_path / "request.json",
        values=["10 ohm"],
        projections=[
            {
                "id": "invalid_switch_ripple",
                "signal": "v(q1)",
                "metric": "ripple",
                "unit": "V",
            }
        ],
    )
    artifacts = run_parameter_sweep(
        ROOT / "examples" / "buck.yaml",
        request,
        tmp_path / "output",
    )
    summary = json.loads(artifacts.sweep_summary.read_text(encoding="utf-8"))

    assert artifacts.execution_status == "fail"
    assert artifacts.overall_status == "fail"
    assert summary["points"][0]["projections"][0]["status"] == "not_run"
    assert summary["points"][0]["overall_status"] == "fail"


def test_unused_parameter_and_unbounded_work_fail_before_output(tmp_path: Path) -> None:
    unused_model = _write_model(tmp_path / "unused.yaml")
    unused_model.write_text(
        unused_model.read_text(encoding="utf-8").replace(
            "  load: 10 ohm\n",
            "  load: 10 ohm\n  unused: 5 ohm\n",
        ),
        encoding="utf-8",
    )
    unused_request = _write_request(
        tmp_path / "unused-request.json",
        parameter="unused",
        values=["5 ohm"],
        projections=[],
    )
    with pytest.raises(ParameterSweepError, match="does not affect resolved physics"):
        run_parameter_sweep(unused_model, unused_request, tmp_path / "unused-output")
    assert not (tmp_path / "unused-output").exists()

    unbounded_model = _write_model(tmp_path / "unbounded.yaml")
    unbounded_model.write_text(
        unbounded_model.read_text(encoding="utf-8")
        # 1.5 M samples: inside the per-run limit, beyond the sweep's per-point one.
        .replace("stop: 50 us", "stop: 1.5 ms")
        .replace("output_step: 1 us", "output_step: 1 ns"),
        encoding="utf-8",
    )
    bounded_request = _write_request(
        tmp_path / "bounded-request.json",
        values=["10 ohm"],
        projections=[],
    )
    with pytest.raises(ParameterSweepError, match="waveform samples"):
        run_parameter_sweep(unbounded_model, bounded_request, tmp_path / "unbounded-output")
    assert not (tmp_path / "unbounded-output").exists()

    overflow_grid_model = _write_model(tmp_path / "overflow-grid.yaml")
    overflow_grid_model.write_text(
        overflow_grid_model.read_text(encoding="utf-8")
        .replace("stop: 50 us", "stop: 1e300 s")
        .replace("output_step: 1 us", "output_step: 1e-300 s"),
        encoding="utf-8",
    )
    # The loader's per-run bound now refuses an overflowing grid before the sweep sees it.
    with pytest.raises(ParameterSweepError, match="unboundedly many output samples"):
        run_parameter_sweep(
            overflow_grid_model,
            bounded_request,
            tmp_path / "overflow-grid-output",
        )
    assert not (tmp_path / "overflow-grid-output").exists()

    overflow_edges_model = tmp_path / "overflow-edges.yaml"
    overflow_edges_model.write_text(
        (ROOT / "examples" / "buck.yaml")
        .read_text(encoding="utf-8")
        .replace("fsw: 20 kHz", "fsw: 1e300 Hz")
        .replace("stop: 2 ms", "stop: 1e300 s")
        .replace("output_step: 1 us", "output_step: 1e300 s"),
        encoding="utf-8",
    )
    with pytest.raises(ParameterSweepError, match="PWM edges"):
        run_parameter_sweep(
            overflow_edges_model,
            bounded_request,
            tmp_path / "overflow-edges-output",
        )
    assert not (tmp_path / "overflow-edges-output").exists()

    tiny_max_step_model = _write_model(tmp_path / "tiny-max-step.yaml")
    tiny_max_step_model.write_text(
        tiny_max_step_model.read_text(encoding="utf-8").replace(
            "    atol: 1.0e-11\n",
            "    atol: 1.0e-11\n    max_step: 1e-300 s\n",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ParameterSweepError, match="max-step intervals"):
        run_parameter_sweep(
            tiny_max_step_model,
            bounded_request,
            tmp_path / "tiny-max-step-output",
        )
    assert not (tmp_path / "tiny-max-step-output").exists()

    tiny_rtol_model = _write_model(tmp_path / "tiny-rtol.yaml")
    tiny_rtol_model.write_text(
        tiny_rtol_model.read_text(encoding="utf-8").replace(
            "rtol: 1.0e-9",
            "rtol: 1.0e-30",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ParameterSweepError, match="less than the minimum"):
        run_parameter_sweep(
            tiny_rtol_model,
            bounded_request,
            tmp_path / "tiny-rtol-output",
        )
    assert not (tmp_path / "tiny-rtol-output").exists()

    quoted_rtol_model = _write_model(tmp_path / "quoted-rtol.yaml")
    quoted_rtol_model.write_text(
        quoted_rtol_model.read_text(encoding="utf-8").replace(
            "rtol: 1.0e-9",
            'rtol: "1.0e-9"',
        ),
        encoding="utf-8",
    )
    quoted_rtol_output = tmp_path / "quoted-rtol-output"
    with pytest.raises(ParameterSweepError, match="is not of type 'number'"):
        run_parameter_sweep(
            quoted_rtol_model,
            bounded_request,
            quoted_rtol_output,
        )
    assert not quoted_rtol_output.exists()

    surrogate_parameter_model = _write_model(tmp_path / "surrogate-parameter.yaml")
    surrogate_parameter_model.write_text(
        surrogate_parameter_model.read_text(encoding="utf-8").replace(
            "parameters:\n",
            'parameters:\n  "\\uD800": 1 ohm\n',
        ),
        encoding="utf-8",
    )
    surrogate_parameter_output = tmp_path / "surrogate-parameter-output"
    with pytest.raises(ParameterSweepError, match="does not match|valid UTF-8 Unicode scalar"):
        run_parameter_sweep(
            surrogate_parameter_model,
            bounded_request,
            surrogate_parameter_output,
        )
    assert not surrogate_parameter_output.exists()


def test_duplicate_outputs_and_oversized_model_text_fail_before_output(
    tmp_path: Path,
) -> None:
    duplicate_outputs = _write_model(tmp_path / "duplicate-outputs.yaml")
    duplicate_outputs.write_text(
        duplicate_outputs.read_text(encoding="utf-8") + "  - v(out)\n",
        encoding="utf-8",
    )
    request = _write_request(
        tmp_path / "request.json",
        values=["10 ohm"],
    )
    duplicate_destination = tmp_path / "duplicate-output"
    with pytest.raises(ValueError, match="has non-unique elements"):
        run_parameter_sweep(
            duplicate_outputs,
            request,
            duplicate_destination,
        )
    assert not duplicate_destination.exists()

    whitespace_output = _write_model(tmp_path / "whitespace-output.yaml")
    whitespace_output.write_text(
        whitespace_output.read_text(encoding="utf-8").replace(
            "  - v(out)\n",
            "  - ' v(out) '\n",
        ),
        encoding="utf-8",
    )
    whitespace_destination = tmp_path / "whitespace-output"
    with pytest.raises(ValueError, match="does not match"):
        run_parameter_sweep(
            whitespace_output,
            request,
            whitespace_destination,
        )
    assert not whitespace_destination.exists()

    oversized_name = _write_model(tmp_path / "oversized-name.yaml")
    oversized_name.write_text(
        oversized_name.read_text(encoding="utf-8").replace(
            "name: rc_parameter_fixture",
            "name: " + "x" * 257,
        ),
        encoding="utf-8",
    )
    oversized_destination = tmp_path / "oversized-name-output"
    with pytest.raises(ValueError, match="is too long"):
        run_parameter_sweep(
            oversized_name,
            request,
            oversized_destination,
        )
    assert not oversized_destination.exists()

    overflow_parameter = _write_model(tmp_path / "overflow-parameter.yaml")
    overflow_parameter.write_text(
        overflow_parameter.read_text(encoding="utf-8").replace(
            "  load: 10 ohm\n",
            "  load: 10 ohm\n  unused_overflow: 1e308 kV\n",
        ),
        encoding="utf-8",
    )
    overflow_destination = tmp_path / "overflow-parameter-output"
    with pytest.raises(ParameterSweepError, match="SI-normalized value must be finite"):
        run_parameter_sweep(
            overflow_parameter,
            request,
            overflow_destination,
        )
    assert not overflow_destination.exists()


def _use_worker_path(monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    """Route sweep points through one-shot workers or the long-lived pool."""

    if path == "one_shot":
        monkeypatch.setattr(
            parameter_sweep_module,
            "run_pooled",
            lambda make_tasks, *, jobs: run_ordered(make_tasks(None), jobs=jobs),
        )


@pytest.mark.parametrize("path", ["one_shot", "pooled"])
def test_point_worker_timeout_is_retained_and_cannot_leave_partial_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    def time_out(*args, **kwargs):
        raise parameter_sweep_module.subprocess.TimeoutExpired(args[0], 0.01)

    def pool_time_out(*args, **kwargs):
        raise WorkerTimeout

    _use_worker_path(monkeypatch, path)
    monkeypatch.setattr(parameter_sweep_module.subprocess, "run", time_out)
    monkeypatch.setattr(PointWorkerPool, "run", pool_time_out)
    artifacts = run_parameter_sweep(
        _write_model(tmp_path / "model.yaml"),
        _write_request(tmp_path / "request.json", values=["10 ohm"]),
        tmp_path / "output",
    )
    summary = json.loads(artifacts.sweep_summary.read_text(encoding="utf-8"))
    failures = json.loads(artifacts.failures.read_text(encoding="utf-8"))

    assert artifacts.execution_status == "fail"
    assert artifacts.overall_status == "fail"
    assert summary["points"][0]["execution_status"] == "fail"
    assert failures["failures"][0]["stage"] == "simulation"
    assert "worker budget" in failures["failures"][0]["message"]
    assert not list(artifacts.output_dir.rglob(".run-worker"))


@pytest.mark.parametrize("path", ["one_shot", "pooled"])
def test_successful_worker_with_unretained_stderr_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: str,
) -> None:
    def pass_payload() -> dict:
        module_path = Path(parameter_sweep_module.__file__).resolve()
        payload = {
            "status": "pass",
            "identity": {
                "package_init": str(module_path.with_name("__init__.py")),
                "worker_module": str(
                    module_path.with_name("parameter_sweep_worker.py")
                ),
                "version": parameter_sweep_module.__version__,
            },
        }
        return payload

    def warn_on_stderr(command, **_kwargs):
        Path(command[command.index("--out") + 1]).mkdir()
        return parameter_sweep_module.subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(pass_payload()).encode("utf-8"),
            stderr=b"solver adjusted a requested setting",
        )

    def pool_warn_on_stderr(self, model_path, overrides, out, **_kwargs):
        out.mkdir()
        return WorkerReply(
            returncode=0,
            stdout=json.dumps(pass_payload()).encode("utf-8"),
            stderr=b"solver adjusted a requested setting",
        )

    _use_worker_path(monkeypatch, path)
    monkeypatch.setattr(parameter_sweep_module.subprocess, "run", warn_on_stderr)
    monkeypatch.setattr(PointWorkerPool, "run", pool_warn_on_stderr)
    artifacts = run_parameter_sweep(
        _write_model(tmp_path / "model.yaml"),
        _write_request(tmp_path / "request.json", values=["10 ohm"]),
        tmp_path / "output",
    )
    failures = json.loads(artifacts.failures.read_text(encoding="utf-8"))

    assert artifacts.execution_status == "fail"
    assert "unretained stderr" in failures["failures"][0]["message"]
    assert not list(artifacts.output_dir.rglob(".run-worker"))


def test_nested_run_is_bound_to_the_exact_expected_point_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _write_model(tmp_path / "model.yaml")
    seed = run_parameter_sweep(
        model,
        _write_request(tmp_path / "seed-request.json", values=["10 ohm"]),
        tmp_path / "seed",
    )
    seed_run = next((seed.output_dir / "points").glob("*/run"))

    def reuse_stale_run(_model_path, _overrides, point_dir, **_kwargs):
        destination = point_dir / "run"
        shutil.copytree(seed_run, destination)
        return destination

    monkeypatch.setattr(
        parameter_sweep_module,
        "_run_point_worker",
        reuse_stale_run,
    )
    artifacts = run_parameter_sweep(
        model,
        _write_request(tmp_path / "target-request.json", values=["20 ohm"]),
        tmp_path / "target",
    )
    failures = json.loads(artifacts.failures.read_text(encoding="utf-8"))
    summary = json.loads(artifacts.sweep_summary.read_text(encoding="utf-8"))

    assert artifacts.execution_status == "fail"
    assert artifacts.overall_status == "fail"
    assert summary["points"][0]["projections"][0]["status"] == "not_run"
    assert failures["failures"][0]["stage"] == "run_bundle"
    assert "expected sweep point" in failures["failures"][0]["message"]


def test_worker_import_is_not_shadowable_from_caller_working_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _write_model(tmp_path / "model.yaml")
    request = _write_request(
        tmp_path / "request.json",
        values=["20 ohm"],
    )
    shadow_root = tmp_path / "shadow"
    shadow_package = shadow_root / "pear_sim"
    shadow_package.mkdir(parents=True)
    marker = tmp_path / "shadow-worker-loaded.txt"
    (shadow_package / "__init__.py").write_text("\n", encoding="utf-8")
    (shadow_package / "parameter_sweep_worker.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('loaded', encoding='utf-8')\n"
        "raise SystemExit(99)\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(shadow_root)

    artifacts = run_parameter_sweep(
        model,
        request,
        tmp_path / "output",
    )
    run_manifest_path = next((artifacts.output_dir / "points").glob("*/run/run.json"))
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    assert artifacts.execution_status == "pass"
    assert not marker.exists()
    assert run_manifest["parameters"]["load"]["value_si"] == 20.0


@pytest.mark.parametrize(
    ("request_update", "match"),
    [
        ({"parameter": "missing"}, "unknown model parameter"),
        ({"values": []}, "1..64"),
        ({"values": ["1 A"]}, "expected resistance"),
        ({"values": ["1" * 257]}, "256-character"),
        ({"values": ["1000 mOhm", "1 ohm"]}, "duplicates"),
        (
            {
                "projections": [
                    {
                        "id": "wrong",
                        "signal": "v(missing)",
                        "metric": "mean",
                        "unit": "V",
                    }
                ]
            },
            "not an exact source-model output",
        ),
        (
            {
                "projections": [
                    {
                        "id": " vout ",
                        "signal": "v(out)",
                        "metric": "mean",
                        "unit": "V",
                    }
                ]
            },
            "leading or trailing whitespace",
        ),
        (
            {
                "projections": [
                    {
                        "id": "vout\nfinal",
                        "signal": "v(out)",
                        "metric": "mean",
                        "unit": "V",
                    }
                ]
            },
            "must match",
        ),
        (
            {
                "projections": [
                    {
                        "id": "vout",
                        "signal": " v(out) ",
                        "metric": "mean",
                        "unit": "V",
                    }
                ]
            },
            "leading or trailing whitespace",
        ),
    ],
)
def test_invalid_request_fails_before_creating_output(
    tmp_path: Path,
    request_update: dict[str, object],
    match: str,
) -> None:
    model = _write_model(tmp_path / "model.yaml")
    request = {
        "schema": "pear-sim/parameter-sweep-request/v0.1",
        "parameter": "load",
        "values": ["10 ohm"],
    }
    request.update(request_update)
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    output = tmp_path / "output"

    with pytest.raises(ParameterSweepError, match=match):
        run_parameter_sweep(model, request_path, output)

    assert not output.exists()


def test_duplicate_json_key_and_invalid_expectations_fail_before_output(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "model.yaml")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema":"pear-sim/parameter-sweep-request/v0.1",'
        '"parameter":"load","parameter":"load","values":[10]}',
        encoding="utf-8",
    )
    output = tmp_path / "duplicate-output"
    with pytest.raises(ParameterSweepError, match="duplicate JSON object key"):
        run_parameter_sweep(model, duplicate, output)
    assert not output.exists()

    request = _write_request(tmp_path / "request.json", values=["10 ohm"])
    invalid_expectations = tmp_path / "invalid-expectations.json"
    invalid_expectations.write_text('{"schema":[],"criteria":[]}', encoding="utf-8")
    output = tmp_path / "expectations-output"
    with pytest.raises(ParameterSweepError, match="Invalid expectations"):
        run_parameter_sweep(
            model,
            request,
            output,
            expectations_file=invalid_expectations,
        )
    assert not output.exists()


def test_sweep_never_overwrites_prior_artifact(tmp_path: Path) -> None:
    destination = tmp_path / "output"
    destination.mkdir()
    marker = destination / "human-note.txt"
    marker.write_text("keep\n", encoding="utf-8")

    with pytest.raises(ParameterSweepError, match="not empty"):
        run_parameter_sweep(
            _write_model(tmp_path / "model.yaml"),
            _write_request(tmp_path / "request.json", values=["10 ohm"]),
            destination,
        )

    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_high_level_artifacts_are_deterministic(tmp_path: Path) -> None:
    model = _write_model(tmp_path / "model.yaml")
    request = _write_request(tmp_path / "request.json", values=["10 ohm", "20 ohm"])

    first = run_parameter_sweep(model, request, tmp_path / "first")
    second = run_parameter_sweep(model, request, tmp_path / "second")

    for first_path, second_path in (
        (first.normalized_request, second.normalized_request),
        (first.source_model, second.source_model),
        (first.sweep_summary, second.sweep_summary),
        (first.projections, second.projections),
        (first.failures, second.failures),
        (first.confidence_json, second.confidence_json),
        (first.report_json, second.report_json),
        (first.report_html, second.report_html),
    ):
        assert first_path.read_bytes() == second_path.read_bytes()
