# Agent interface: MCP server, CLI and Python API

One contract, three surfaces. The CLI (`pear-sim`), the Python API (`pear_sim.api`)
and the MCP server (`pear-sim mcp`) run the same code path: the same model through
any of them produces a bit-identical `run_sha256` (`tests/test_agent_contract.py`
checks this, together with every MCP rule below).

## Installing and connecting the MCP server

The server lives in the optional extra `pear-sim[mcp]`, so the core package stays
`numpy`/`scipy`/`PyYAML`/`jsonschema` only. The `pear-sim-mcp` launcher package installs it with the
extra; with [uv](https://docs.astral.sh/uv/) nothing has to be installed up front:

```bash
uvx pear-sim-mcp            # same server as `pear-sim mcp` from an install with [mcp]
```

The first `uvx` start downloads numpy, scipy and the MCP SDK, which can take a minute or
more on a slow connection — longer than some clients wait for a server to start. Run
`uvx pear-sim-mcp --help` once before adding it to a client; later starts take a few
seconds.

From a source checkout (before a release, or to try local changes), point uv at the
checkout without leaving your project directory:

```bash
claude mcp add pear-sim -- uv run --project /path/to/pear-sim pear-sim mcp
```

`--project` keeps the current directory, so the server reads models from your project;
`uv run --directory` would make the checkout the project root instead.

**Claude Code**

```bash
claude mcp add pear-sim -- uvx pear-sim-mcp
```

**Claude Desktop, Cursor and other JSON-configured clients**
(`claude_desktop_config.json`, `.cursor/mcp.json`, …):

```json
{
  "mcpServers": {
    "pear-sim": {
      "command": "uvx",
      "args": ["pear-sim-mcp"]
    }
  }
}
```

**Codex CLI** (`~/.codex/config.toml`):

```toml
[mcp_servers.pear-sim]
command = "uvx"
args = ["pear-sim-mcp"]
```

Options: `--workspace DIR` (where inline models and run bundles go; default
`$PEAR_SIM_WORKSPACE`, else `<project>/.pear-sim`, else `~/.pear-sim`) and
`--project-root DIR` (the directory `model_path` may read from; default the current
directory). The per-call time limit defaults to 120 s (`PEAR_SIM_MCP_TIMEOUT_S`) and
can be raised per call with `timeout_s`, up to 900 s.

## Design rules

1. **Small JSON plus a `run_id`.** No tool returns full waveforms. `simulate` returns
   per-signal statistics over the last half of the run, loss and THD summaries when the
   model asks for them, warnings and a confidence level. Everything else stays in the
   run bundle and is read selectively.
2. **Models inline or from the project.** Every model-taking tool accepts exactly one
   of `model_yaml` (the text), `model_path` (a file under the project root) or `example`
   (a shipped example). Clients without filesystem access work with `model_yaml`.
3. **A closed filesystem boundary.** Reads are confined to the project root and the
   workspace; writes to the workspace. The server refuses `/` and the home directory as
   a project root, so a desktop client started there cannot read the disk. Runs are
   addressed by `run_id`, never by path.
4. **Isolation and a hard time limit.** Simulations and analyses run in a child process;
   one that exceeds its limit is stopped and reported, and the server keeps serving.
5. **Actionable errors.** A failed call returns an error result naming the cause and,
   for common mistakes (missing antiparallel diodes, an unknown component type, a
   capacitor loop), a hint on how to fix it.
6. **Claim boundaries travel with the results.** The server instructions tell the agent
   what the ideal-device model does not capture and to pass warnings on to the user.
7. **Bounded memory.** A run is limited to 2 000 001 output samples (about 0.9 GB of memory
   and 200 MB on disk for a small converter), checked when the model is loaded, so a
   mistyped `output_step` fails at once instead of filling the machine's memory.
8. **Models from others are untrusted data.** The server never executes anything from a
   model, but names and comments in a model reach the agent as text. A model received from
   someone else can carry instructions aimed at the agent; read it as data, not as a request.

## Tools

| Tool | Purpose |
|---|---|
| `list_examples`, `get_example` | shipped example models, the fastest correct starting point |
| `get_model_format` | the format reference, by section |
| `capabilities` | supported / experimental / unsupported, with evidence |
| `validate_model` | parse, validate and explain the resolved circuit |
| `render_schematic` | deterministic SVG schematic |
| `simulate` | transient run, compact summary and a `run_id` |
| `steady_state` | periodic steady state by Newton shooting |
| `loop_gain` | loop gain and margins for a declared compensator |
| `sweep_parameter`, `sweep_tolerance_corners` | bounded sweeps, every point retained |
| `evaluate_designs` | screen up to 2 000 designs, rank against an objective and constraints, re-run finalists in full |
| `get_metrics`, `get_waveform`, `get_artifact` | selective reads from a run |
| `compare_runs` | side-by-side metrics of several runs |
| `list_runs`, `doctor` | workspace listing and installation diagnosis |

Resources: `pear-sim://docs/model-format`, `pear-sim://schema/model`,
`pear-sim://examples/{name}`. Prompts: `design_converter`, `review_model`.

## CLI

`pear-sim --help` lists every command; `--json` gives stable machine-readable output.
Breaking changes to a `--json` payload need a CHANGELOG entry with a migration note.

## `pear-sim doctor`

Read-only diagnosis: interpreter and dependency versions, package integrity (schemas and
validation ledger), the ngspice backend, a float-determinism probe (compare its digest
across machines) and every enforced limit. `--json` for agents; non-zero exit on
problems.
