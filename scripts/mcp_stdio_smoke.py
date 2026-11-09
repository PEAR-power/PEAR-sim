# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Talk to an installed ``pear-sim-mcp`` over stdio and check it really works.

Standard library only: CI copies this file to a directory OUTSIDE the
repository and runs it against a server installed from the built wheels, so
nothing from the source tree can stand in for the installed package.

Usage: python mcp_stdio_smoke.py -- <command that starts the server...>

Checks, each failing with a message that names what was missing:
1. the server answers ``initialize`` and reports its name;
2. ``tools/list`` has the tools an agent needs first;
3. ``list_examples`` finds the examples shipped inside the wheel (catches a
   wheel that forgot its packaged data);
4. ``simulate`` runs the shipped ``buck`` example end to end (worker process
   included) and returns metrics.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from typing import Any

REQUIRED_TOOLS = {"list_examples", "validate_model", "simulate", "doctor"}
TIMEOUT_S = 300.0


class Session:
    def __init__(self, command: list[str]) -> None:
        self.proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            encoding="utf-8",
        )
        self._next_id = 0

    def notify(self, method: str, params: dict[str, Any] | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._next_id += 1
        wanted = self._next_id
        self._send({"jsonrpc": "2.0", "id": wanted, "method": method, "params": params or {}})
        while True:
            message = self._read()
            if message.get("id") == wanted:
                if "error" in message:
                    raise SystemExit(f"{method} returned an error: {message['error']}")
                return message["result"]

    def _send(self, message: dict[str, Any]) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def _read(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        box: list[str] = []
        reader = threading.Thread(target=lambda: box.append(self.proc.stdout.readline()))
        reader.daemon = True
        reader.start()
        reader.join(TIMEOUT_S)
        if not box:
            raise SystemExit(f"no answer from the server within {TIMEOUT_S:.0f} s")
        if not box[0]:
            raise SystemExit(f"the server closed stdout (exit code {self.proc.poll()})")
        return json.loads(box[0])

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def tool_payload(result: dict[str, Any], tool: str) -> dict[str, Any]:
    if result.get("isError"):
        raise SystemExit(f"tool {tool} failed: {result.get('content')}")
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        # FastMCP wraps non-object returns as {"result": ...}; ours are objects.
        return structured.get("result", structured) if set(structured) == {"result"} else structured
    for block in result.get("content", []):
        if block.get("type") == "text":
            return json.loads(block["text"])
    raise SystemExit(f"tool {tool} returned no readable content: {result}")


def main(argv: list[str]) -> int:
    if "--" not in argv:
        raise SystemExit("usage: mcp_stdio_smoke.py -- <server command...>")
    command = argv[argv.index("--") + 1 :]
    session = Session(command)
    try:
        init = session.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "pear-sim-ci-smoke", "version": "0"},
            },
        )
        name = init.get("serverInfo", {}).get("name")
        if name != "pear-sim":
            raise SystemExit(f"serverInfo.name is {name!r}, expected 'pear-sim'")
        session.notify("notifications/initialized")

        tools = {tool["name"] for tool in session.request("tools/list")["tools"]}
        missing = REQUIRED_TOOLS - tools
        if missing:
            raise SystemExit(f"tools missing from tools/list: {sorted(missing)}")

        examples = tool_payload(
            session.request("tools/call", {"name": "list_examples", "arguments": {}}),
            "list_examples",
        )
        names = json.dumps(examples)
        if '"buck"' not in names:
            raise SystemExit(f"list_examples does not list 'buck' (packaged data missing?): {names[:500]}")

        run = tool_payload(
            session.request(
                "tools/call", {"name": "simulate", "arguments": {"example": "buck"}}
            ),
            "simulate",
        )
        if "metrics" not in json.dumps(run):
            raise SystemExit(f"simulate returned no metrics: {json.dumps(run)[:500]}")
    finally:
        session.close()
    print(f"MCP smoke OK: server {name} {init['serverInfo'].get('version')}, {len(tools)} tools")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
