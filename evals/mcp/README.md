# MCP agent evaluation

Eleven engineering requests (`tasks.yaml`) sent to a real agent connected to
`pear-sim mcp`. Each answer is graded by `checks.py` against the runs the agent
actually produced in its server workspace or against closed forms: numbers quoted
in the answer must match the simulator, and claim boundaries must be passed on.

```bash
uv run python evals/mcp/run.py --agent claude            # Claude Code CLI
uv run python evals/mcp/run.py --agent cursor --model X  # Cursor CLI
```

Results land in `evals/mcp/results/`. Running the evaluation spends the agent
account's usage.

The graders are themselves tested for discrimination: a scripted correct answer
(built only through the MCP tools) must pass and a wrong one must fail, for every
task.

## Results

| Date | Agent | Model | Tasks | Runs | Passed | Cost |
|---|---|---|---|---|---|---|
| 2026-10-16 | `run.py --agent api` (Messages API loop over the real MCP server) | Claude Opus 5.5 | 11 | 3 | 33 / 33 | ≈ 4.2 USD |

What this does and does not show: an agent with no prior knowledge of the format
can go from a plain-language engineering request to correct simulated numbers
through this server, repeatedly, and passes on the tool's warnings. It does not
show that the numbers match hardware — the simulator's own claim boundaries
(ideal or piecewise-linear devices, synthetic loss cards) still apply — and one
model on eleven tasks is a small sample.

## User stories (ungraded, reviewed by hand)

`stories.yaml` and `stories2.yaml` hold 40 open-ended requests: longer workflows,
project files, configuration mistakes, requests outside the tool's scope and vague
ones. They have no automatic grader; each answer was read against the runs it
produced and against closed forms. Every defect they exposed was fixed before the
first release and is listed in `CHANGELOG.md` under "Fixed before release".

```bash
uv run python evals/mcp/run.py --agent api --tasks-file evals/mcp/stories2.yaml
```
