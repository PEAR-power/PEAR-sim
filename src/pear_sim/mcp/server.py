"""MCP server for PEARSim, built on the official MCP Python SDK.

Install with ``pip install "pear-sim[mcp]"`` and start with ``pear-sim mcp``.
The core package stays numpy/scipy/PyYAML/jsonschema-only; the SDK is an optional extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pear_sim import __version__
from pear_sim.mcp import tools as t
from pear_sim.mcp.workspace import Workspace

__all__ = ["INSTRUCTIONS", "build_server", "main", "serve"]

INSTRUCTIONS = """\
PEARSim is a deterministic switching simulator for power electronics. Models are YAML.

Workflow that works:
1. Start from a shipped example close to the user's circuit (list_examples, get_example);
   read the relevant part of the format with get_model_format when you need a detail.
2. validate_model before simulating; errors come with a hint on how to fix them.
3. simulate returns compact statistics over the last half of the run plus a run_id.
   Make the simulated span long enough to reach steady state (or use steady_state).
4. Read more selectively: get_metrics, get_waveform (downsampled), get_artifact (JSON
   artifacts such as losses-thermal.json or harmonics.json), compare_runs.

Claim boundaries you must pass on to the user:
- Switches and diodes are IDEAL unless the model declares on_resistance /
  forward_voltage (then conduction is v = V_f + R_on*i, still without recovery or
  switching transitions). Model capacitor ESR and inductor DCR as series resistors.
  Loss cards are post-processed and synthetic in the examples. Do not present
  efficiency or temperatures as validated.
- Every result carries warnings and a confidence level; report them, don't drop them.
- Loop-gain margins attest the DECLARED compensator on an averaged model; they are not a
  closed-loop proof.
- Out of scope, say so instead of estimating: EMI and compliance, real part catalogues
  and switching losses of named parts, reverse recovery, FEM, PCB parasitics.
- For ripple and efficiency of a converter, prefer steady_state: a transient's last half
  can still carry start-up ringing.
- Capability status (supported / experimental / unsupported) is in the capabilities tool.
"""

_MODEL_INPUT = (
    "Give exactly one of: model_yaml (full model text), model_path (a file inside the "
    "project directory) or example (a shipped example name)."
)


def _doc(text: str, *, model_input: bool = False) -> str:
    flat = " ".join(text.split())
    return f"{flat} {_MODEL_INPUT}" if model_input else flat


DESCRIPTIONS = {
    "list_examples": _doc(
        """List shipped example models (buck, boost, LLC, DAB, rectifiers, inverters, NPC,
        machines, closed loop, losses/thermal) with a one-line summary each."""
    ),
    "get_example": _doc(
        """Return the full YAML of a shipped example. Adapting an example is the fastest
        correct way to write a new model."""
    ),
    "get_model_format": _doc(
        """The model-format reference. Without `section` returns the table of contents;
        with `section` (a title or its leading number, e.g. "5") returns that section."""
    ),
    "capabilities": _doc(
        """Fail-closed capability registry: what is supported, experimental or
        unsupported. Pass capability_id for the full statement and the evidence."""
    ),
    "validate_model": _doc(
        """Parse and fully validate a model and explain the resolved circuit (parameters,
        topology, assumptions, limitations). `overrides` maps parameter names to values
        with units, e.g. {"duty": "45 %"}.""",
        model_input=True,
    ),
    "render_schematic": _doc(
        """Render a deterministic SVG schematic of the model, returned as SVG text.""",
        model_input=True,
    ),
    "simulate": _doc(
        """Run a transient simulation. Returns a run_id, per-signal mean/rms/min/max/
        peak-to-peak over the LAST HALF of the run, loss and THD summaries when the model
        requests them, mean powers from the run's energy audit, warnings and a confidence
        level — never full waveforms. `overrides` maps parameter names to values with
        units. `efficiency_output` (component ids, e.g. ["load"]) adds efficiency =
        P(those components) / P(sources).""",
        model_input=True,
    ),
    "steady_state": _doc(
        """Periodic steady state of a PWM-driven converter by Newton shooting: the state
        (inductor currents, capacitor voltages) at the start of a switching period and its
        cycle averages, plus a fresh run started ON the orbit that reports steady-state
        ripple and powers. `efficiency_output` (component ids, e.g. ["load"]) adds
        steady-state efficiency. Needs at least one PWM control.""",
        model_input=True,
    ),
    "loop_gain": _doc(
        """Small-signal loop gain and phase/gain margins of an OPEN-LOOP PWM plant with a
        DECLARED compensator (averaged model around the periodic steady state, including
        the ZOH and computation delay of digital control). `request` fields: output
        (signal, e.g. "v(out)"), compensator {numerator, denominator} as polynomial
        coefficients in s, sensor_gain, f_min_hz, f_max_hz, points, margins
        {min_phase_margin_deg, min_gain_margin_db}, sampling {update_frequency_hz,
        computation_delay_s}.""",
        model_input=True,
    ),
    "sweep_parameter": _doc(
        """Bounded 1-D sweep of one model parameter; every point is a retained run.
        `request`: {"parameter": "load", "values": ["5 ohm", "10 ohm"], "projections":
        [{"id": "vout", "signal": "v(out)", "metric": "mean_last_half", "unit": "V"}]}.""",
        model_input=True,
    ),
    "sweep_tolerance_corners": _doc(
        """Worst case over parameter tolerance corners (nominal point plus every
        corner); names the binding corner. `request`: {"tolerances": [{"parameter":
        "load", "percent": 10}], "band_metrics": [{"signal": "v(out)", "field":
        "mean_last_half"}]}.""",
        model_input=True,
    ),
    "evaluate_designs": _doc(
        """Screen many designs of one model and rank them: best design OF THE SAMPLE,
        not an optimum. Designs are sampled (Latin hypercube or grid) or listed; up to
        2000 per call, run in parallel. Feasible designs are ranked by the objective;
        the top `finalists` are re-run in full and must reproduce their screening
        values. `request`: {"designs": {"method": "lhs", "count": 40, "seed": 1,
        "parameters": [{"name": "fsw", "from": "5 kHz", "to": "50 kHz", "scale":
        "log"}]}, "objective": {"minimize": {"losses": "total_mean_power_w"}},
        "constraints": [{"signal": "v(out)", "metric": "peak_to_peak_last_half",
        "max": 0.5}, {"losses": "tj_max_c", "max": 125}], "finalists": 3}. Losses
        need device cards in the model. "backend": "rust" runs on the optional compiled
        kernel (refused with the reason if the model is not covered). Large batches
        need a larger timeout_s.""",
        model_input=True,
    ),
    "sensitivity": _doc(
        """Global sensitivity of outputs to parameters over declared ranges. "morris"
        (r*(k+1) runs): mu_star = typical output change across a parameter's whole
        range, sigma = interaction/non-linearity. "sobol" (N*(k+2) runs, N a power of
        two): S1 = share of output variance a parameter explains alone, ST = including
        interactions. Every index has a bootstrap 95 % interval. Parameters are
        independent and uniform over their ranges, so indices describe the model in
        THIS box only. `request`: {"method": "sobol", "parameters": [{"name": "duty",
        "from": "35 %", "to": "45 %"}, {"name": "fsw", "from": "10 kHz", "to": "40
        kHz", "scale": "log"}], "outputs": [{"signal": "v(out)", "metric":
        "peak_to_peak_last_half"}, {"losses": "total_mean_power_w"}], "base_samples":
        64, "seed": 1}. Optional "backend": "rust" as in evaluate_designs. At most 2000
        runs; large studies need a larger timeout_s.""",
        model_input=True,
    ),
    "get_metrics": _doc(
        """All (or selected) metrics of one recorded signal: full-window and last-half
        mean, rms, min, max, peak-to-peak, integrals."""
    ),
    "get_waveform": _doc(
        """A bounded, downsampled window of one recorded signal (max 2048 points). Zoom
        with start_s/stop_s to see switching ripple."""
    ),
    "get_artifact": _doc(
        """Read a JSON artifact of a run (e.g. losses-thermal.json, harmonics.json,
        physics-audit.json, confidence-card.json), optionally narrowed with a
        slash-separated path such as "devices/0/thermal". Line artifacts (.jsonl:
        events, control-trace, signal-trace, machine-trace) are read by line range,
        path="0-100"."""
    ),
    "compare_runs": _doc(
        """Side-by-side metrics of 2-12 runs (default: mean, rms and peak-to-peak over the
        last half), e.g. to compare two design variants."""
    ),
    "list_runs": _doc("Run ids retained in this server's workspace, newest first."),
    "doctor": _doc(
        """Versions, package integrity, available oracle backends, enforced limits and a
        float-determinism probe for this installation."""
    ),
}


def build_server(workspace: Workspace):
    try:
        from mcp.server.mcpserver import MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        from mcp_types import ToolAnnotations
    except ImportError as exc:  # pragma: no cover - exercised by the CLI guard
        raise RuntimeError(
            'the MCP server needs the optional SDK: pip install "pear-sim[mcp]"'
        ) from exc

    server = MCPServer(
        name="pear-sim",
        title="PEARSim power-electronics simulator",
        instructions=INSTRUCTIONS,
        website_url="https://github.com/PEAR-power/pear-sim",
        version=__version__,
    )
    read_only = ToolAnnotations(readOnlyHint=True, openWorldHint=False)
    computes = ToolAnnotations(
        readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False
    )

    def guarded(fn, *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except t.ToolInputError as exc:
            raise ToolError(str(exc)) from exc

    def tool(name: str, annotations):
        return server.tool(name=name, description=DESCRIPTIONS[name], annotations=annotations)

    # -- learning the format ---------------------------------------------------
    @tool("list_examples", read_only)
    def list_examples() -> dict[str, Any]:
        return guarded(t.list_examples)

    @tool("get_example", read_only)
    def get_example(name: str) -> dict[str, Any]:
        return guarded(t.get_example, name)

    @tool("get_model_format", read_only)
    def get_model_format(section: str | None = None) -> dict[str, Any]:
        return guarded(t.get_model_format, section)

    @tool("capabilities", read_only)
    def capabilities(capability_id: str | None = None) -> dict[str, Any]:
        return guarded(t.capabilities, capability_id)

    # -- models ------------------------------------------------------------------
    @tool("validate_model", read_only)
    def validate_model(
        model_yaml: str | None = None,
        model_path: str | None = None,
        example: str | None = None,
        overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return guarded(
            t.validate_model, workspace, model_yaml=model_yaml, model_path=model_path,
            example=example, overrides=overrides,
        )

    @tool("render_schematic", read_only)
    def render_schematic(
        model_yaml: str | None = None,
        model_path: str | None = None,
        example: str | None = None,
    ) -> str:
        return guarded(
            t.render_schematic, workspace, model_yaml=model_yaml, model_path=model_path,
            example=example,
        )

    # -- computing -----------------------------------------------------------------
    @tool("simulate", computes)
    def simulate(
        model_yaml: str | None = None,
        model_path: str | None = None,
        example: str | None = None,
        overrides: dict[str, str] | None = None,
        timeout_s: float | None = None,
        efficiency_output: list[str] | None = None,
    ) -> dict[str, Any]:
        return guarded(
            t.simulate, workspace, model_yaml=model_yaml, model_path=model_path,
            example=example, overrides=overrides, timeout_s=timeout_s,
            efficiency_output=efficiency_output,
        )

    @tool("steady_state", computes)
    def steady_state(
        model_yaml: str | None = None,
        model_path: str | None = None,
        example: str | None = None,
        overrides: dict[str, str] | None = None,
        timeout_s: float | None = None,
        efficiency_output: list[str] | None = None,
    ) -> dict[str, Any]:
        return guarded(
            t.steady_state, workspace, model_yaml=model_yaml, model_path=model_path,
            example=example, overrides=overrides, timeout_s=timeout_s,
            efficiency_output=efficiency_output,
        )

    @tool("loop_gain", computes)
    def loop_gain(
        request: dict[str, Any],
        model_yaml: str | None = None,
        model_path: str | None = None,
        example: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return guarded(
            t.loop_gain, workspace, request=request, model_yaml=model_yaml,
            model_path=model_path, example=example, timeout_s=timeout_s,
        )

    @tool("sweep_parameter", computes)
    def sweep_parameter(
        request: dict[str, Any],
        model_yaml: str | None = None,
        model_path: str | None = None,
        example: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return guarded(
            t.sweep_parameter, workspace, request=request, model_yaml=model_yaml,
            model_path=model_path, example=example, timeout_s=timeout_s,
        )

    @tool("sweep_tolerance_corners", computes)
    def sweep_tolerance_corners(
        request: dict[str, Any],
        model_yaml: str | None = None,
        model_path: str | None = None,
        example: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return guarded(
            t.sweep_tolerance_corners, workspace, request=request, model_yaml=model_yaml,
            model_path=model_path, example=example, timeout_s=timeout_s,
        )

    @tool("evaluate_designs", computes)
    def evaluate_designs(
        request: dict[str, Any],
        model_yaml: str | None = None,
        model_path: str | None = None,
        example: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return guarded(
            t.evaluate_designs, workspace, request=request, model_yaml=model_yaml,
            model_path=model_path, example=example, timeout_s=timeout_s,
        )

    @tool("sensitivity", computes)
    def sensitivity(
        request: dict[str, Any],
        model_yaml: str | None = None,
        model_path: str | None = None,
        example: str | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return guarded(
            t.sensitivity, workspace, request=request, model_yaml=model_yaml,
            model_path=model_path, example=example, timeout_s=timeout_s,
        )

    # -- reading results -------------------------------------------------------------
    @tool("get_metrics", read_only)
    def get_metrics(run_id: str, signal: str, fields: list[str] | None = None) -> dict[str, Any]:
        return guarded(t.get_metrics, workspace, run_id=run_id, signal=signal, fields=fields)

    @tool("get_waveform", read_only)
    def get_waveform(
        run_id: str,
        signal: str,
        start_s: float | None = None,
        stop_s: float | None = None,
        max_points: int = 400,
    ) -> dict[str, Any]:
        return guarded(
            t.get_waveform, workspace, run_id=run_id, signal=signal, start_s=start_s,
            stop_s=stop_s, max_points=max_points,
        )

    @tool("get_artifact", read_only)
    def get_artifact(run_id: str, name: str, path: str | None = None) -> dict[str, Any]:
        return guarded(t.get_artifact, workspace, run_id=run_id, name=name, path=path)

    @tool("compare_runs", read_only)
    def compare_runs(
        run_ids: list[str], signals: list[str] | None = None, metrics: list[str] | None = None
    ) -> dict[str, Any]:
        return guarded(
            t.compare_runs, workspace, run_ids=run_ids, signals=signals, metrics=metrics
        )

    @tool("list_runs", read_only)
    def list_runs() -> dict[str, Any]:
        return guarded(t.list_runs, workspace)

    @tool("doctor", read_only)
    def doctor() -> dict[str, Any]:
        return guarded(t.doctor)

    # -- resources -------------------------------------------------------------------
    @server.resource("pear-sim://docs/model-format", mime_type="text/markdown")
    def model_format_doc() -> str:
        """The complete model-format reference."""
        return t.data_file("docs/model-format.md").read_text(encoding="utf-8")

    @server.resource("pear-sim://schema/model", mime_type="application/schema+json")
    def model_schema() -> str:
        """JSON Schema of the model document."""
        from importlib import resources

        return resources.files("pear_sim").joinpath("spec", "model.schema.json").read_text(
            encoding="utf-8"
        )

    @server.resource("pear-sim://examples/{name}", mime_type="application/yaml")
    def example_resource(name: str) -> str:
        """A shipped example model."""
        return guarded(t.get_example, name)["model_yaml"]

    # -- prompts ---------------------------------------------------------------------
    @server.prompt()
    def design_converter(requirements: str) -> str:
        """Design and verify a converter from plain-language requirements."""
        return (
            "Design a power converter for these requirements:\n\n"
            f"{requirements}\n\n"
            "Use the pear-sim tools. Pick a topology, then start from the closest shipped "
            "example (list_examples/get_example) and adapt it. Validate, simulate long enough "
            "to reach steady state, and check every requirement against the simulated "
            "numbers. Report a table of requirement vs. simulated value, the warnings and "
            "confidence of the runs, and state clearly which effects the ideal-switch model "
            "does not capture."
        )

    @server.prompt()
    def review_model(model_path: str) -> str:
        """Review an existing model file for mistakes and missing effects."""
        return (
            f"Review the PEARSim model at {model_path}. Validate it, explain the resolved "
            "circuit, simulate it, and look for: wrong units or polarity, a run too short to "
            "reach steady state, missing damping or ESR, missing antiparallel diodes, and "
            "claims the ideal-device model cannot support. Report findings with evidence "
            "from the tool results."
        )

    return server


def serve(workspace: Path | None = None, project_root: Path | None = None) -> int:
    server = build_server(Workspace.create(workspace=workspace, project_root=project_root))
    server.run("stdio")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point of the ``pear-sim-mcp`` command (same options as ``pear-sim mcp``)."""

    import argparse

    parser = argparse.ArgumentParser(prog="pear-sim-mcp", description="PEARSim MCP server (stdio)")
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--project-root", type=Path)
    args = parser.parse_args(argv)
    return serve(workspace=args.workspace, project_root=args.project_root)
