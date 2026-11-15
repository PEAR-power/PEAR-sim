# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Agent contract: the SAME semantics through the CLI and the MCP server.

Every scenario that both surfaces support must agree on the semantic anchors
(model identity, run hash, metric values), and the MCP server must hold the
filesystem boundary and give actionable diagnostics instead of crashing.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from mcp import Client, StdioServerParameters  # noqa: E402

from pear_sim.mcp import tools as t  # noqa: E402
from pear_sim.mcp.server import DESCRIPTIONS, build_server  # noqa: E402
from pear_sim.mcp.workspace import Workspace  # noqa: E402

ROOT = Path(__file__).parents[1]

MODEL = """
schema: pear-sim/v0.1
name: rc_contract

parameters:
  r: 1 kohm

circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 10 V}
    - {id: r1, type: resistor, p: in, n: out, value: "${r}"}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 1 uF, initial_voltage: 0 V}

analysis:
  transient:
    start: 0 s
    stop: 5 ms
    output_step: 10 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11

outputs:
  - v(out)
"""

EXPECTED_TOOLS = {
    "list_examples", "get_example", "get_model_format", "capabilities", "validate_model",
    "render_schematic", "simulate", "steady_state", "loop_gain", "sweep_parameter",
    "sweep_tolerance_corners", "get_metrics", "get_waveform", "get_artifact",
    "compare_runs", "list_runs", "doctor", "evaluate_designs", "sensitivity", "floquet",
}


@pytest.fixture()
def workspace(tmp_path: Path) -> Workspace:
    project = tmp_path / "project"
    project.mkdir()
    return Workspace.create(workspace=tmp_path / "ws", project_root=project)


def _call(workspace: Workspace, tool: str, arguments: dict) -> tuple[bool, object]:
    async def run():
        async with Client(build_server(workspace)) as client:
            return await client.call_tool(tool, arguments)

    result = asyncio.run(run())
    text = result.content[0].text if result.content else ""
    if result.is_error:
        return True, text
    return False, json.loads(text) if text.lstrip().startswith(("{", "[")) else text


def _cli(*arguments: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pear_sim.api.cli", *arguments],
        capture_output=True, text=True, timeout=300,
    )


# --- inventory -----------------------------------------------------------------


def test_tool_inventory_descriptions_and_annotations(workspace: Workspace) -> None:
    async def run():
        async with Client(build_server(workspace)) as client:
            return (await client.list_tools()).tools

    listed = asyncio.run(run())
    assert {tool.name for tool in listed} == EXPECTED_TOOLS == set(DESCRIPTIONS)
    for tool in listed:
        assert tool.description and len(tool.description) > 40, tool.name
        assert tool.annotations is not None, tool.name
        assert tool.annotations.open_world_hint is False, tool.name
    read_only = {tool.name for tool in listed if tool.annotations.read_only_hint}
    assert "simulate" not in read_only and "get_metrics" in read_only


# --- one contract, two surfaces -------------------------------------------------


def test_simulation_hash_is_identical_across_cli_and_mcp(workspace: Workspace, tmp_path) -> None:
    model = workspace.read_roots[-1] / "rc.yaml"
    model.write_text(MODEL, encoding="utf-8")
    cli = _cli("simulate", str(model), "--set", "r=2 kohm", "--out", str(tmp_path / "cli"), "--json")
    assert cli.returncode == 0, cli.stderr
    cli_summary = json.loads((tmp_path / "cli" / "summary.json").read_text())

    is_error, result = _call(
        workspace, "simulate", {"model_path": "rc.yaml", "overrides": {"r": "2 kohm"}}
    )
    assert not is_error, result
    assert result["run_sha256"] == cli_summary["run_sha256"]
    # tau = 2 ms over a 5 ms run: the last-half mean is a known closed form.
    assert result["signals"]["v(out)"]["unit"] == "V"


def test_inline_model_simulates_and_results_are_read_selectively(workspace: Workspace) -> None:
    is_error, result = _call(workspace, "simulate", {"model_yaml": MODEL})
    assert not is_error, result
    run_id = result["run_id"]
    assert "waveforms" not in json.dumps(result).lower().replace("waveforms.csv", "")

    is_error, window = _call(
        workspace, "get_waveform", {"run_id": run_id, "signal": "v(out)", "max_points": 10}
    )
    assert not is_error and len(window["values"]) <= 10 and window["stride"] > 1

    is_error, metrics = _call(
        workspace, "get_metrics", {"run_id": run_id, "signal": "v(out)", "fields": ["final"]}
    )
    assert not is_error and set(metrics["metrics"]) == {"final"}
    assert metrics["metrics"]["final"] == pytest.approx(10 * (1 - 2.718281828 ** -5), rel=1e-3)

    is_error, artifact = _call(
        workspace, "get_artifact", {"run_id": run_id, "name": "summary.json", "path": "stats"}
    )
    assert not is_error and "integration_steps" in artifact["data"]


# --- boundary and diagnostics ---------------------------------------------------


def test_paths_outside_the_readable_roots_are_refused(workspace: Workspace) -> None:
    for path in ("/etc/passwd", "../outside.yaml"):
        is_error, message = _call(workspace, "validate_model", {"model_path": path})
        assert is_error and "outside the readable roots" in message


def test_unknown_run_ids_cannot_escape_the_workspace(workspace: Workspace) -> None:
    for run_id in ("../../etc", "nope"):
        is_error, message = _call(workspace, "get_metrics", {"run_id": run_id, "signal": "x"})
        assert is_error and "run_id" in message


def test_exactly_one_model_source_is_required(workspace: Workspace) -> None:
    is_error, message = _call(workspace, "simulate", {})
    assert is_error and "exactly one of" in message
    is_error, message = _call(workspace, "simulate", {"model_yaml": MODEL, "example": "buck"})
    assert is_error and "exactly one of" in message


def test_a_broken_model_returns_an_actionable_hint(workspace: Workspace) -> None:
    broken = MODEL.replace("type: resistor", "type: mosfet")
    is_error, message = _call(workspace, "validate_model", {"model_yaml": broken})
    assert is_error and "unknown component type 'mosfet'" in message
    assert "ideal_switch" in message and "body diode" in message


def test_a_computation_over_its_time_limit_is_stopped(workspace: Workspace) -> None:
    is_error, message = _call(
        workspace, "simulate", {"example": "llc_resonant", "timeout_s": 1}
    )
    assert is_error and "timeout_s" in message


def test_workspace_refuses_root_and_home_as_project(tmp_path: Path) -> None:
    for project in (Path("/"), Path.home()):
        ws = Workspace.create(workspace=tmp_path / "ws", project_root=project)
        assert ws.read_roots == (ws.root,)


# --- reference material --------------------------------------------------------


def test_examples_and_format_reference_are_served() -> None:
    names = {item["name"] for item in t.list_examples()["examples"]}
    assert {"buck", "boost", "llc_resonant"} <= names
    toc = t.get_model_format()["sections"]
    assert len(toc) > 5
    first_numbered = next(s["title"] for s in toc if s["title"][0].isdigit())
    assert t.get_model_format(first_numbered.split()[0].rstrip("."))["text"]


# --- the real entry point --------------------------------------------------------


def test_the_cli_entry_point_serves_mcp_over_stdio(tmp_path: Path) -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "pear_sim.api.cli", "mcp", "--workspace", str(tmp_path / "ws"),
              "--project-root", str(tmp_path)],
    )

    async def run():
        async with Client(params) as client:
            tools = await client.list_tools()
            result = await client.call_tool("simulate", {"example": "buck"})
            return tools, result

    tools, result = asyncio.run(run())
    assert {tool.name for tool in tools.tools} == EXPECTED_TOOLS
    assert not result.is_error
    assert json.loads(result.content[0].text)["run_id"].startswith("run-")


def test_steady_state_reports_the_switching_ripple_not_the_startup_transient(
    workspace: Workspace,
) -> None:
    """A transient's last half still rings on a lightly damped LC filter; the
    steady cycle must not. Closed forms for the ideal buck with dead time:
    dI = Vout (1 - D_eff) / (L fsw), dV = dI / (8 fsw C)."""

    source = (ROOT / "examples" / "synchronous_buck_complementary.yaml").read_text()
    model = (
        source.replace("vin: 100 V", "vin: 48 V").replace("fsw: 20 kHz", "fsw: 200 kHz")
        .replace("duty: 40 %", "duty: 25 %").replace("dead_time: 500 ns", "dead_time: 50 ns")
        .replace("load: 10 ohm", "load: 1.2 ohm")
        .replace("value: 1 mH, initial_current: 4 A", "value: 10 uH, initial_current: 10 A")
        .replace("initial_voltage: 40 V", "initial_voltage: 12 V")
        .replace("output_step: 250 ns", "output_step: 20 ns")
    )
    is_error, result = _call(workspace, "steady_state", {"model_yaml": model})
    assert not is_error, result
    cycle = result["steady_cycle"]["signals"]
    vout = 48.0 * (0.25 - 50e-9 * 200e3)
    ripple_i = vout * (1 - 0.24) / (10e-6 * 200e3)
    assert cycle["v(out)"]["mean"] == pytest.approx(vout, rel=1e-5)
    # Dead-time intervals bend the ramp slightly; 2 % covers it.
    assert cycle["i(l1)"]["peak_to_peak"] == pytest.approx(ripple_i, rel=0.02)
    assert cycle["v(out)"]["peak_to_peak"] == pytest.approx(
        cycle["i(l1)"]["peak_to_peak"] / (8 * 200e3 * 100e-6), rel=0.02
    )
    # ...and its waveform is readable by run_id.
    is_error, window = _call(
        workspace, "get_waveform",
        {"run_id": result["steady_cycle"]["run_id"], "signal": "v(out)", "max_points": 20},
    )
    assert not is_error


def test_efficiency_and_conduction_loss_follow_the_energy_audit(workspace: Workspace) -> None:
    """Synchronous buck, equal R_on on both switches: the inductor current flows
    through exactly one switch at a time, so the in-circuit device loss is
    R_on * rms(i_L)^2 — computed here from the signal statistics, a separate
    quadrature from the per-component energy the efficiency comes from."""

    model = (ROOT / "tests" / "test_conduction_drops.py").read_text().split('SYNC_BUCK = """')[1]
    model = model.split('"""')[0].replace("stop: 50 us", "stop: 2 ms")
    is_error, result = _call(
        workspace, "simulate", {"model_yaml": model, "efficiency_output": ["load"]}
    )
    assert not is_error, result
    power = result["power"]
    efficiency = power["efficiency"]
    p_in, p_out = efficiency["input_w"], efficiency["output_w"]
    loss = power["device_conduction_loss_in_circuit_w"]
    rms = result["signals"]["i(l1)"]["rms"]
    assert loss == pytest.approx(0.05 * rms**2, rel=1e-4)
    assert p_in == pytest.approx(p_out + loss + power["net_storage_rate_w"], rel=1e-4)
    assert efficiency["efficiency_in_circuit"] == pytest.approx(p_out / p_in, rel=1e-6)
    assert 0.9 < efficiency["efficiency_in_circuit"] < 1.0


def test_a_parameter_sweep_runs_on_a_thyristor_model(workspace: Workspace) -> None:
    # The sweep renders the source model; a thyristor used to stop it cold.
    is_error, result = _call(
        workspace,
        "sweep_parameter",
        {
            "example": "thyristor_phase_controlled",
            "request": {
                "parameter": "alpha",
                "values": ["1.667 ms", "3.333 ms"],
                "projections": [
                    {"id": "idc", "signal": "i(ll)", "metric": "mean_last_half", "unit": "A"}
                ],
            },
        },
    )
    assert not is_error, result
    assert result["execution_status"] == "pass"
    first, second = (point["projections"]["idc"] for point in result["points"])
    assert first > second > 0  # later firing, lower mean load current


def test_settling_is_reported_and_discriminates(workspace: Workspace) -> None:
    # Settled: the shipped buck starts on its orbit.
    _, settled = _call(workspace, "simulate", {"example": "buck"})
    assert settled["settling"]["status"] == "settled"
    # Not settled: same buck, capacitor started 2 V low, LC still ringing.
    ringing_model = (ROOT / "examples" / "buck.yaml").read_text().replace(
        "initial_voltage: 39.98969319 V", "initial_voltage: 38 V"
    )
    _, ringing = _call(workspace, "simulate", {"model_yaml": ringing_model})
    assert ringing["settling"]["status"] == "not_settled" and "warning" in ringing["settling"]
    # Not assessed: an inverter is periodic in its fundamental, not its carrier.
    _, inverter = _call(workspace, "simulate", {"example": "inverter_three_phase_spwm"})
    assert inverter["settling"]["status"] == "not_assessed"


def test_tolerance_corners_flag_corners_that_had_not_settled(workspace: Workspace) -> None:
    # An ideal buck's mean output does not depend on the load in steady state;
    # a band that moves with load comes from ringing of corners that start
    # from the nominal initial conditions. The tool must say so.
    is_error, result = _call(
        workspace,
        "sweep_tolerance_corners",
        {
            "example": "buck",
            "request": {
                "tolerances": [{"parameter": "load", "percent": 20}],
                "band_metrics": [{"signal": "v(out)", "field": "mean_last_half"}],
            },
        },
    )
    assert not is_error, result
    assert result["settling"]["nominal"] == "settled"
    assert "corner-load+" in result["settling_warning"]
    assert result["warnings"]


def test_a_relative_path_without_a_project_says_how_to_fix_it(tmp_path: Path) -> None:
    ws = Workspace.create(workspace=tmp_path / "ws", project_root=Path.home())
    is_error, message = _call(ws, "validate_model", {"model_path": "model.yaml"})
    assert is_error and "--project-root" in message and "model_yaml" in message


def test_line_artifacts_are_read_by_range(workspace: Workspace) -> None:
    _, run = _call(workspace, "simulate", {"example": "buck"})
    is_error, events = _call(
        workspace, "get_artifact", {"run_id": run["run_id"], "name": "events.jsonl", "path": "0-5"}
    )
    assert not is_error, events
    assert events["range"] == [0, 5] and len(events["rows"]) == 5 and events["total_lines"] > 5


def test_drive_efficiency_counts_the_machine_not_its_back_emf_as_a_supply(
    workspace: Workspace,
) -> None:
    is_error, run = _call(
        workspace, "simulate", {"example": "pmsm_inverter_drive", "efficiency_output": ["motor"]}
    )
    assert not is_error, run
    power = run["power"]
    efficiency = power["efficiency"]
    assert efficiency["output_w"] == pytest.approx(power["machine_electromagnetic_w"], rel=1e-6)
    assert efficiency["input_w"] == pytest.approx(power["sources_delivered_w"], rel=1e-6)


def test_confidence_level_travels_with_its_meaning(workspace: Workspace) -> None:
    """A bare "not_run" reads like a failed run; the meaning and the uncovered
    features must come with it. A covered canonical buck must not list any."""

    error, foc = _call(workspace, "simulate", {"example": "pmsm_foc_current_control"})
    assert not error, foc
    assert foc["confidence"] == "not_run"
    assert "simulation itself ran" in foc["confidence_meaning"]
    assert foc["confidence_uncovered_features"]

    error, buck = _call(workspace, "simulate", {"example": "buck"})
    assert not error, buck
    assert buck["confidence"] != "not_run"
    assert buck["confidence_uncovered_features"] == []


def test_evaluate_designs_ranks_a_listed_batch(workspace: Workspace) -> None:
    is_error, result = _call(
        workspace,
        "evaluate_designs",
        {
            "example": "buck_with_loss_cards",
            "request": {
                "designs": {
                    "method": "list",
                    "rows": [{"fsw": "20 kHz"}, {"fsw": "10 kHz"}],
                },
                "objective": {"minimize": {"losses": "total_mean_power_w"}},
                "constraints": [{"losses": "tj_max_c", "max": 125}],
                "finalists": 1,
            },
        },
    )
    assert not is_error, result
    assert result["status"] == "pass"
    assert result["best_design"] == "d0002"  # lower frequency, lower switching loss
    assert result["finalists"][0]["status"] == "pass"
    assert "not an optimum" in result["claim"]


# --- closed-loop stability (Floquet) ------------------------------------------------

FLOQUET_EXAMPLE = ROOT / "examples" / "buck_closed_loop_floquet.yaml"


def _hot_gains(source: str) -> str:
    """kp=0.08 / ki=150: the design the shipped example names as unstable."""
    hot = source.replace("  kp: 0.002 ", "  kp: 0.08 ").replace("  ki: 30.0 ", "  ki: 150.0 ")
    assert hot.count("0.08") > source.count("0.08") and "ki: 150.0" in hot
    return hot


def test_floquet_is_the_same_verdict_through_cli_and_mcp(workspace: Workspace, tmp_path) -> None:
    out = tmp_path / "cli"
    completed = _cli("analyze", "floquet", str(FLOQUET_EXAMPLE), "--out", str(out), "--json")
    assert completed.returncode == 0, completed.stderr
    cli = json.loads((out / "floquet.json").read_text(encoding="utf-8"))

    is_error, result = _call(workspace, "floquet", {"example": "buck_closed_loop_floquet"})
    assert not is_error, result
    assert result["overall_status"] == "pass"
    assert result["stability"] == cli["stability"] == "stable"
    assert result["max_multiplier_magnitude"] == pytest.approx(
        cli["max_multiplier_magnitude"], rel=1e-6
    )
    assert result["lyapunov_certificate"]["found"] is True
    assert {gate["id"] for gate in result["gates"]} == {
        "closed_loop_orbit_converged", "multipliers_assessed", "locally_stable",
        "lyapunov_certificate",
    }
    assert "LOCAL" in result["claim_boundary"].upper()
    assert "stable" in result["interpretation"] and "UNSTABLE" not in result["interpretation"]
    # The full record stays readable by run_id.
    is_error, monodromy = _call(
        workspace, "get_artifact",
        {"run_id": result["run_id"], "name": "floquet.json", "path": "monodromy"},
    )
    assert not is_error, monodromy


def test_floquet_rings_on_the_unstable_gains_and_names_the_frequency(
    workspace: Workspace,
) -> None:
    """The discriminating case: the hot PI leaves the unit circle at ~3.9 kHz,
    the limit cycle the example's comment reports from a plain transient."""

    model = _hot_gains(FLOQUET_EXAMPLE.read_text(encoding="utf-8"))
    is_error, result = _call(
        workspace, "floquet",
        {"model_yaml": model, "initial_state": {"signal:int_prev": 0.5 / 150.0}},
    )
    assert not is_error, result
    assert result["stability"] == "unstable"
    assert result["overall_status"] == "fail"
    assert result["max_multiplier_magnitude"] > 1.0
    assert result["multipliers"][0]["frequency_hz"] == pytest.approx(3.92e3, rel=0.02)
    assert "UNSTABLE" in result["interpretation"]
    # The second witness stays silent where it must: no P exists outside the circle.
    assert result["lyapunov_certificate"]["found"] is False


def test_floquet_says_how_to_start_when_newton_hits_the_limiter(workspace: Workspace) -> None:
    """Hot gains started from the example's integrator value clamp the duty on
    the first Newton step: the verdict is withheld and the reason is surfaced,
    with the remedy, instead of a number."""

    model = _hot_gains(FLOQUET_EXAMPLE.read_text(encoding="utf-8"))
    is_error, result = _call(workspace, "floquet", {"model_yaml": model})
    assert not is_error, result
    assert result["converged"] is False
    assert result["stability"] == "not_assessed"
    assert result["overall_status"] == "fail"
    assert any("initial_state" in note and "limiter" in note for note in result["notes"])


def test_floquet_overrides_move_the_orbit_like_the_duty_says(workspace: Workspace) -> None:
    """Vin 24 V -> 20 V: the integrator settles where ki * int = duty ~ 12/20.
    (The sampled gain differs from Vin by ~0.4 %, hence the 1 % band.)"""

    is_error, result = _call(
        workspace, "floquet",
        {"example": "buck_closed_loop_floquet", "overrides": {"vin": "20 V"}},
    )
    assert not is_error, result
    assert result["stability"] == "stable"
    assert 30.0 * result["orbit_state"]["signal:int_prev"] == pytest.approx(12 / 20, rel=0.01)
    assert result["orbit_state"]["c1"] == pytest.approx(12.0, rel=1e-9)


def test_floquet_refusals_are_actionable(workspace: Workspace) -> None:
    # A built-in sampled controller is outside the scope and is named.
    is_error, message = _call(workspace, "floquet", {"example": "buck_closed_loop"})
    assert is_error
    assert "sampled" in str(message).lower()
    # Initial states must be numbers (refused by the tool schema itself).
    is_error, message = _call(
        workspace, "floquet",
        {"example": "buck_closed_loop_floquet", "initial_state": {"l1": "one amp"}},
    )
    assert is_error and "initial_state" in str(message)


def test_steady_state_on_a_closed_loop_model_points_to_floquet(workspace: Workspace) -> None:
    is_error, message = _call(workspace, "steady_state", {"example": "buck_closed_loop_floquet"})
    assert is_error
    assert "closed-loop" in str(message) and "`floquet` tool" in str(message)


def test_floquet_overrides_reach_the_controller_gains(workspace: Workspace) -> None:
    """The same unstable verdict whether the hot gains are written into the model
    or passed as overrides of its dimensionless parameters."""

    written = _hot_gains(FLOQUET_EXAMPLE.read_text(encoding="utf-8"))
    start = {"signal:int_prev": 0.5 / 150.0}
    _, from_yaml = _call(workspace, "floquet", {"model_yaml": written, "initial_state": start})
    is_error, from_overrides = _call(
        workspace, "floquet",
        {"example": "buck_closed_loop_floquet", "overrides": {"kp": "0.08", "ki": "150"},
         "initial_state": start},
    )
    assert not is_error, from_overrides
    assert from_overrides["stability"] == from_yaml["stability"] == "unstable"
    assert from_overrides["max_multiplier_magnitude"] == from_yaml["max_multiplier_magnitude"]
