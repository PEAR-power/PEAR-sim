"""Run the MCP agent evaluation against a real agent CLI.

    uv run python evals/mcp/run.py --agent claude [--task ID ...]
    uv run python evals/mcp/run.py --agent cursor --model sonnet-4

Every task gets a fresh server workspace. The agent talks to
``pear-sim mcp`` from this checkout, its final answer is graded by
``checks.py`` against the runs it produced, and results are written to
``evals/mcp/results/``. This spends the agent account's usage.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from checks import CHECKS  # noqa: E402

SUFFIX = (
    "\n\nUse the pear-sim MCP tools for every simulation and analysis; do not run shell "
    "commands or write files yourself. End with a short answer that states the numbers."
)


def _server(workspace: Path, project: Path) -> dict:
    return {
        "command": shutil.which("uv") or "uv",
        "args": [
            "run", "--project", str(ROOT), "--extra", "mcp", "pear-sim", "mcp",
            "--workspace", str(workspace), "--project-root", str(project),
        ],
    }


def _run_claude(prompt: str, workspace: Path, project: Path, model: str | None, timeout: float) -> str:
    config = project / "mcp.json"
    config.write_text(json.dumps({"mcpServers": {"pear-sim": _server(workspace, project)}}))
    command = [
        "claude", "-p", prompt + SUFFIX, "--mcp-config", str(config), "--strict-mcp-config",
        "--allowedTools", "mcp__pear-sim", "--output-format", "json",
    ]
    if model:
        command += ["--model", model]
    completed = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=timeout)
    try:
        return json.loads(completed.stdout).get("result", "")
    except json.JSONDecodeError:
        return completed.stdout + completed.stderr


def _run_cursor(prompt: str, workspace: Path, project: Path, model: str | None, timeout: float) -> str:
    (project / ".cursor").mkdir(exist_ok=True)
    (project / ".cursor" / "mcp.json").write_text(
        json.dumps({"mcpServers": {"pear-sim": _server(workspace, project)}})
    )
    agent = shutil.which("cursor-agent") or str(Path.home() / ".local/bin/cursor-agent")
    command = [agent, "-p", "--approve-mcps", "--output-format", "text", "--workspace", str(project)]
    if model:
        command += ["--model", model]
    command.append(prompt + SUFFIX)
    completed = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=timeout)
    return completed.stdout + (completed.stderr if completed.returncode else "")


def _run_api(prompt: str, workspace: Path, project: Path, model: str | None, timeout: float) -> str:
    """A plain tool-use loop on the Messages API, wired to the real MCP server.

    The model sees the server's own tool list and instructions — what any MCP
    client would show it — and every tool call goes over stdio to
    ``pear-sim mcp``. Usage is appended to ``api-usage.jsonl`` next to the
    results so the cost of an evaluation is on record.
    """

    import asyncio

    import anthropic
    from mcp import Client, StdioServerParameters

    from pear_sim.mcp.server import INSTRUCTIONS

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "pear_sim.api.cli", "mcp", "--workspace", str(workspace),
              "--project-root", str(project)],
    )
    chosen = model or "claude-opus-5-5"
    client = anthropic.Anthropic()
    system = (
        "You are an engineering assistant with access to the PEARSim power-electronics "
        "simulator through MCP tools.\n\nServer instructions:\n" + INSTRUCTIONS
    )

    async def loop() -> str:
        async with Client(params) as mcp_client:
            listed = (await mcp_client.list_tools()).tools
            tools = [
                {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                for t in listed
            ]
            messages: list = [{"role": "user", "content": prompt + SUFFIX}]
            usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "turns": 0}
            final = ""
            for _ in range(40):
                response = await asyncio.to_thread(
                    client.messages.create,
                    model=chosen,
                    max_tokens=16000,
                    system=system,
                    tools=tools,
                    messages=messages,
                    cache_control={"type": "ephemeral"},
                )
                usage["turns"] += 1
                usage["input"] += response.usage.input_tokens
                usage["output"] += response.usage.output_tokens
                usage["cache_read"] += response.usage.cache_read_input_tokens or 0
                usage["cache_write"] += response.usage.cache_creation_input_tokens or 0
                messages.append({"role": "assistant", "content": response.content})
                text = "\n".join(b.text for b in response.content if b.type == "text")
                if text:
                    final = text
                if response.stop_reason != "tool_use":
                    if response.stop_reason == "refusal":
                        final = f"[refusal] {final}"
                    break
                results = []
                for block in response.content:
                    if block.type != "tool_use":
                        continue
                    outcome = await mcp_client.call_tool(block.name, block.input)
                    body = "\n".join(
                        getattr(item, "text", "") for item in (outcome.content or [])
                    ) or "(no content)"
                    results.append(
                        {"type": "tool_result", "tool_use_id": block.id,
                         "content": body[:60000], "is_error": bool(outcome.is_error)}
                    )
                messages.append({"role": "user", "content": results})
            (HERE / "results").mkdir(exist_ok=True)
            with (HERE / "results").joinpath("api-usage.jsonl").open("a") as log:
                log.write(json.dumps({"model": chosen, "workspace": str(workspace), **usage}) + "\n")
            return final

    return asyncio.run(asyncio.wait_for(loop(), timeout))


RUNNERS = {"claude": _run_claude, "cursor": _run_cursor, "api": _run_api}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=sorted(RUNNERS), required=True)
    parser.add_argument("--model")
    parser.add_argument("--task", action="append", help="run only these task ids")
    parser.add_argument("--timeout", type=float, default=900.0, help="per task, seconds")
    parser.add_argument(
        "--tasks-file", default="tasks.yaml",
        help="task list in this directory; tasks without a grader are recorded for review",
    )
    args = parser.parse_args()

    tasks = yaml.safe_load((HERE / args.tasks_file).read_text())
    if args.task:
        tasks = [task for task in tasks if task["id"] in set(args.task)]
    results = []
    for task in tasks:
        base = Path(tempfile.mkdtemp(prefix=f"pear-sim-eval-{task['id']}-"))
        workspace, project = base / "ws", base / "project"
        project.mkdir()
        for name, content in (task.get("files") or {}).items():
            (project / name).write_text(content, encoding="utf-8")
        started = time.monotonic()
        try:
            answer = RUNNERS[args.agent](task["prompt"], workspace, project, args.model, args.timeout)
            grader = CHECKS.get(task["id"])
            passed, detail = grader(answer, workspace) if grader else (None, "for manual review")
        except (subprocess.TimeoutExpired, TimeoutError):
            answer, passed, detail = "", False, f"agent exceeded {args.timeout:g} s"
        elapsed = time.monotonic() - started
        runs = sorted(p.name for p in (workspace / "runs").glob("*")) if (workspace / "runs").is_dir() else []
        results.append({
            "id": task["id"], "passed": passed, "detail": detail, "seconds": round(elapsed, 1),
            "runs": runs, "answer": answer,
        })
        label = "REVIEW" if passed is None else ("PASS" if passed else "FAIL")
        print(f"{label:<6} {task['id']:<28} {elapsed:6.1f}s  {detail}", flush=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ")
    stem = Path(args.tasks_file).stem
    out = HERE / "results" / f"{stem}-{args.agent}-{args.model or 'default'}-{stamp}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"agent": args.agent, "model": args.model, "results": results}, indent=2))
    graded = [item for item in results if item["passed"] is not None]
    passed = sum(bool(item["passed"]) for item in graded)
    print(f"\n{passed}/{len(graded)} graded passed, {len(results) - len(graded)} for review — {out.relative_to(ROOT)}")
    return 0 if passed == len(graded) else 1


if __name__ == "__main__":
    raise SystemExit(main())
