# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""MCP tool implementations, independent of the MCP SDK.

Each public function takes a :class:`Workspace` plus plain arguments and
returns a JSON-serialisable dict, or raises :class:`ToolInputError` with a
message written for the model that called it: what went wrong and what to do
next. The SDK layer (``server.py``) only registers these functions.

Contract (unchanged from the CLI): compact JSON, never waveform dumps; large
artifacts stay on disk and are read selectively by ``run_id``.
"""

from __future__ import annotations

import csv
import json
import os
import re
import signal as _signal
import subprocess
import sys
from importlib import resources
from pathlib import Path
from typing import Any

from pear_sim import __version__
from pear_sim.mcp.workspace import Workspace, WorkspaceError

__all__ = ["ToolInputError", "DEFAULT_TIMEOUT_S", "MAX_TIMEOUT_S"]

DEFAULT_TIMEOUT_S = float(os.environ.get("PEAR_SIM_MCP_TIMEOUT_S", "120"))
MAX_TIMEOUT_S = 900.0
MAX_WINDOW_POINTS = 2048
MAX_ARTIFACT_BYTES = 200_000
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _sig(value: Any, digits: int = 7) -> Any:
    """Round floats for compact summaries; exact values stay in the artifacts."""

    if isinstance(value, float):
        return float(f"{value:.{digits}g}")
    if isinstance(value, dict):
        return {key: _sig(item, digits) for key, item in value.items()}
    if isinstance(value, list):
        return [_sig(item, digits) for item in value]
    return value


class ToolInputError(ValueError):
    """An error the calling agent can act on; the message says how."""


# ---------------------------------------------------------------------------
# Packaged reference data (docs and examples)
# ---------------------------------------------------------------------------


def data_file(relative: str) -> Path:
    """A reference file shipped in the wheel, or in the source checkout."""

    packaged = resources.files("pear_sim").joinpath("_data", relative)
    if packaged.is_file():
        return Path(str(packaged))
    checkout = _REPO_ROOT / relative
    if checkout.is_file():
        return checkout
    raise ToolInputError(f"reference file {relative!r} is not available in this installation")


def _examples_dir() -> Path:
    packaged = resources.files("pear_sim").joinpath("_data", "examples")
    if packaged.is_dir():
        return Path(str(packaged))
    return _REPO_ROOT / "examples"


def _example_summary(path: Path) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines()[1:40]:
        if line.startswith("#"):
            text = line.lstrip("#").strip()
            if text:
                lines.append(text)
            elif lines:
                break
        elif lines:
            break
    return " ".join(lines)[:300]


def list_examples() -> dict[str, Any]:
    examples = [
        {"name": path.stem, "summary": _example_summary(path)}
        for path in sorted(_examples_dir().glob("*.yaml"))
    ]
    return {
        "examples": examples,
        "hint": "get_example(name) returns the full YAML; adapt it rather than writing from scratch",
    }


def get_example(name: str) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ToolInputError(f"invalid example name {name!r}")
    path = _examples_dir() / f"{name}.yaml"
    if not path.is_file():
        known = ", ".join(p.stem for p in sorted(_examples_dir().glob("*.yaml")))
        raise ToolInputError(f"no example {name!r}; known examples: {known}")
    return {"name": name, "model_yaml": path.read_text(encoding="utf-8")}


_HEADING = re.compile(r"^(#{2,3}) +(.+?)\s*$")


def _format_sections() -> list[tuple[str, str]]:
    text = data_file("docs/model-format.md").read_text(encoding="utf-8")
    sections: list[tuple[str, list[str]]] = [("Introduction", [])]
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match and match.group(1) == "##":
            sections.append((match.group(2), [line]))
        else:
            sections[-1][1].append(line)
    return [(title, "\n".join(body).strip()) for title, body in sections]


def get_model_format(section: str | None = None) -> dict[str, Any]:
    sections = _format_sections()
    if section is None:
        return {
            "sections": [
                {"title": title, "characters": len(body)} for title, body in sections
            ],
            "hint": "call again with section=<title or its leading number> to read one section",
        }
    wanted = section.strip().lower()
    for title, body in sections:
        lowered = title.lower()
        if lowered == wanted or lowered.startswith(wanted + " ") or lowered.startswith(wanted + "."):
            return {"title": title, "text": body}
    for title, body in sections:
        if wanted in title.lower():
            return {"title": title, "text": body}
    hits = [title for title, body in sections if wanted in body.lower()]
    if hits:
        return {
            "query": section,
            "matching_sections": hits,
            "hint": "no section has that title; these sections mention it — call again "
            "with one of these titles",
        }
    raise ToolInputError(
        f"no section matching {section!r}; titles: " + "; ".join(t for t, _ in sections)
    )


# ---------------------------------------------------------------------------
# Model input
# ---------------------------------------------------------------------------


def resolve_model(
    workspace: Workspace,
    model_yaml: str | None = None,
    model_path: str | None = None,
    example: str | None = None,
) -> Path:
    given = [name for name, value in (("model_yaml", model_yaml), ("model_path", model_path), ("example", example)) if value]
    if len(given) != 1:
        raise ToolInputError(
            "give exactly one of model_yaml (the model text), model_path (a file in the "
            "project) or example (a shipped example name)"
            + (f"; got {', '.join(given)}" if given else "")
        )
    try:
        if model_yaml:
            return workspace.store_inline_model(model_yaml)
        if model_path:
            return workspace.readable(model_path)
    except WorkspaceError as exc:
        raise ToolInputError(str(exc)) from exc
    get_example(example or "")  # validates the name with a helpful error
    return _examples_dir() / f"{example}.yaml"


def _overrides(overrides: dict[str, Any] | None) -> dict[str, str] | None:
    if not overrides:
        return None
    result = {}
    for name, value in overrides.items():
        if not isinstance(name, str) or not name:
            raise ToolInputError("override names must be non-empty parameter names")
        result[name] = str(value)
    return result


def _load(path: Path, overrides: dict[str, str] | None):
    from pear_sim.api.facade import error_code, load

    try:
        return load(path, overrides)
    except Exception as exc:  # noqa: BLE001 - becomes an actionable tool error
        raise ToolInputError(_explain_error(exc, error_code(exc))) from exc


_HINTS = (
    (
        re.compile(r"no algebraic support", re.I),
        "a node has no conduction path in some switch state. Ideal switches are "
        "unidirectional and have no body diode: add an antiparallel `diode` across "
        "each `ideal_switch` (cathode to the switch's p terminal), as the shipped "
        "synchronous_buck_complementary example does.",
    ),
    (
        re.compile(r"'ref' is a required property|is not valid under any of the given schemas", re.I),
        "a component entry did not match any known type. Valid `type` values are listed "
        "by the capabilities tool and in get_model_format('5'); subcircuit instances need "
        "`ref:`.",
    ),
    (
        re.compile(r"capacitor loop|inductor cutset|dependent states", re.I),
        "the circuit has a capacitor loop or an inductor cutset, which this kernel "
        "refuses rather than regularizing. Add the physical series resistance (ESR, "
        "source resistance) that breaks the loop.",
    ),
)


def _explain_error(exc: BaseException, code: str) -> str:
    message = f"{type(exc).__name__} [{code}]: {exc}"
    for pattern, hint in _HINTS:
        if pattern.search(message):
            return f"{message}\nHint: {hint}"
    return message


# ---------------------------------------------------------------------------
# Worker subprocess
# ---------------------------------------------------------------------------


def _timeout(timeout_s: float | None) -> float:
    value = DEFAULT_TIMEOUT_S if timeout_s is None else float(timeout_s)
    if not 1.0 <= value <= MAX_TIMEOUT_S:
        raise ToolInputError(f"timeout_s must lie in [1, {MAX_TIMEOUT_S:g}] seconds")
    return value


def _worker_command() -> list[str]:
    return [sys.executable, "-m", "pear_sim.mcp.worker"]


def _stop_worker_tree(process: subprocess.Popen[str]) -> None:
    # A sweep worker runs its points in grandchild processes, several at once.
    # Killing only the worker would leave them running as orphans, so on POSIX
    # the worker leads its own process group and the whole group is stopped.
    if os.name == "posix":
        try:
            os.killpg(process.pid, _signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:  # pragma: no cover - Windows has no process groups to signal
        process.kill()
    process.communicate()


def _run_worker(request: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    process = subprocess.Popen(
        _worker_command(),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(json.dumps(request), timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        _stop_worker_tree(process)
        raise ToolInputError(
            f"the computation exceeded timeout_s={timeout_s:g} s and was stopped. Shorten "
            "the simulated time span, raise output_step, or pass a larger timeout_s."
        ) from exc
    except BaseException:
        _stop_worker_tree(process)
        raise
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ToolInputError(
            f"the worker exited with code {process.returncode} and no result: "
            f"{stderr.strip()[-800:]}"
        )
    payload = json.loads(lines[-1])
    if not payload["ok"]:
        message = f"[{payload['code']}] {payload['error']}"
        for pattern, hint in _HINTS:
            if pattern.search(message):
                message = f"{message}\nHint: {hint}"
                break
        raise ToolInputError(message)
    return payload["result"]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def validate_model(
    workspace: Workspace,
    *,
    model_yaml: str | None = None,
    model_path: str | None = None,
    example: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from pear_sim.api.facade import explain

    path = resolve_model(workspace, model_yaml, model_path, example)
    resolved_overrides = _overrides(overrides)
    model = _load(path, resolved_overrides)
    try:
        explanation = explain(model)
    except Exception as exc:  # noqa: BLE001
        from pear_sim.api.facade import error_code

        raise ToolInputError(_explain_error(exc, error_code(exc))) from exc
    return {
        "valid": True,
        "name": model.name,
        "explanation": explanation,
    }


def _signal_unit(signal: str) -> str | None:
    if signal.startswith("v("):
        return "V"
    if signal.startswith("i("):
        return "A"
    return None


def _signal_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    summary = {}
    for signal, values in metrics.items():
        if not isinstance(values, dict):
            continue
        entry = {
            key: values.get(f"{key}_last_half")
            for key in ("mean", "rms", "min", "max", "peak_to_peak")
        }
        unit = _signal_unit(signal)
        if unit:
            entry["unit"] = unit
        summary[signal] = entry
    return summary


_CONFIDENCE_MEANING = {
    "validated": "every feature this model uses is covered by independent evidence cases",
    "partial": "the features this model uses are covered only within stated claim boundaries",
    "not_run": "at least one feature this model uses has no independent check yet; the "
    "simulation itself ran, but nothing outside the engine confirms that feature",
}


def _confidence(bundle: Path) -> dict[str, Any]:
    """The run's coverage confidence, with what the level means and what is uncovered.

    A bare level is easy to misread ("not_run" sounds like the run failed), so
    the meaning and the uncovered features travel with it.
    """

    path = bundle / "confidence-card.json"
    card = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    overall = card.get("overall")
    uncovered = [
        row.get("feature") for row in card.get("features", []) if row.get("status") == "not_run"
    ]
    return {
        "confidence": overall,
        "confidence_meaning": _CONFIDENCE_MEANING.get(overall),
        "confidence_uncovered_features": uncovered,
    }


def _warnings(bundle: Path) -> list[str]:
    path = bundle / "warnings.json"
    if not path.is_file():
        return []
    items = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(items, dict):
        items = items.get("warnings", [])
    seen, result = set(), []
    for item in items:
        code = item.get("code")
        if code in seen:
            continue
        seen.add(code)
        result.append(f"{item.get('severity', 'info')} {code}: {item.get('message', '')}")
    return result


def _losses_summary(bundle: Path) -> dict[str, Any] | None:
    path = bundle / "losses-thermal.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    devices = []
    for device in data.get("devices", []):
        conduction = device.get("conduction") or {}
        switching = device.get("switching") or {}
        thermal = device.get("thermal") or {}
        devices.append(
            {
                "component": device.get("component_id"),
                "kind": device.get("kind"),
                "card": device.get("card"),
                "card_source": device.get("card_source"),
                "conduction_mean_power_w": conduction.get("mean_power_w"),
                "switching_energy_totals_j": switching.get("totals_j"),
                "switching_events": switching.get("events_counted"),
                "tj_max_c": thermal.get("tj_max_c"),
                "tj_validity_violated": thermal.get("validity_violated"),
            }
        )
    return {
        "total_mean_power_w": data.get("total_mean_power_w"),
        "devices": devices,
        "claim": data.get("claim"),
    }


def _settling(bundle: Path, model: Any) -> dict[str, Any]:
    """Has the run reached periodic steady state by the end?

    Assessed only where periodicity is unambiguous: fixed-frequency `pwm` /
    `complementary_pwm` gates sharing one frequency (DC-DC). Every recorded
    signal is compared between the first and last switching periods of the
    last-half window; a change above 2 % of its peak-to-peak there means the last-half
    statistics still contain the start-up transient. The comparison spans the
    whole last-half window (first vs last period), because a run can settle by
    its end while the window still opens on ringing. Inverters (periodic in
    the fundamental, not the carrier) and closed loops are "not_assessed" —
    never a guess either way.
    """

    controls = list(model.controls.values())
    kinds = {getattr(control, "kind", None) for control in controls}
    frequencies = {getattr(control, "frequency", None) for control in controls}
    if not controls or not kinds <= {"pwm", "complementary_pwm"} or len(frequencies) != 1:
        return {"status": "not_assessed", "reason": "no single fixed-frequency PWM to define a period"}
    period = 1.0 / float(next(iter(frequencies)))
    path = bundle / "waveforms.csv"
    metrics_path = bundle / "metrics.json"
    if not path.is_file() or not metrics_path.is_file():
        return {"status": "not_assessed", "reason": "waveforms not retained"}
    import numpy as np

    data = np.genfromtxt(path, delimiter=",", names=True, deletechars="")
    names = data.dtype.names[1:]
    times = np.asarray(data[data.dtype.names[0]], dtype=float)
    end = float(times[-1])
    if end - period < float(times[0]):
        return {"status": "not_assessed", "reason": "run shorter than one switching period"}
    from pear_sim.analysis.settling import window_drift

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    window_start = end - 0.5 * (end - float(times[0]))
    worst: tuple[str, float] | None = None
    for name in names:
        values = np.asarray(data[name], dtype=float)
        swing = (metrics.get(name) or {}).get("peak_to_peak_last_half")
        drift = window_drift(times, values, period, window_start)
        if not swing or drift is None:
            continue
        relative = drift / swing
        if worst is None or relative > worst[1]:
            worst = (name, relative)
    if worst is None:
        return {"status": "not_assessed", "reason": "no varying signal recorded"}
    settled = worst[1] <= 0.02
    return {
        "status": "settled" if settled else "not_settled",
        "worst_signal": worst[0],
        "change_across_window_vs_last_half_swing": worst[1],
        **(
            {}
            if settled
            else {
                "warning": (
                    "the run has not reached steady state: last-half statistics include the "
                    "start-up transient (ringing, drift), so ripple and averages may be wrong — "
                    "use steady_state, simulate longer, or start nearer the operating point"
                )
            }
        ),
    }


def _power_summary(
    bundle: Path, model: Any, efficiency_output: list[str] | None
) -> dict[str, Any] | None:
    """Mean powers over the last half, read from the run's own energy audit."""

    path = bundle / "physics-audit.json"
    if not path.is_file():
        return None
    window = json.loads(path.read_text(encoding="utf-8")).get("last_half") or {}
    energies = window.get("component_absorbed_energy_j")
    duration = window.get("duration_s")
    if not energies or not duration:
        return None
    kinds = {component.id: component.type for component in model.components}
    absorbed = {cid: energy / duration for cid, energy in energies.items()}
    delivered = window.get("source_delivered_mean_power_w")
    storage_rate = (window.get("delta_storage_energy_j") or 0.0) / duration
    in_circuit_device_loss = sum(
        power for cid, power in absorbed.items()
        if kinds.get(cid) in {"ideal_switch", "diode", "thyristor"}
    )
    summary: dict[str, Any] = {
        "window": "last half of the run (same window as the signal statistics)",
        "sign_convention": "absorbed power is positive; a source delivering power is negative",
        "sources_delivered_w": delivered,
        "absorbed_w": {
            cid: power for cid, power in sorted(absorbed.items())
            if kinds.get(cid) not in {"voltage_source", "current_source", "voltage_probe"}
        },
        "device_conduction_loss_in_circuit_w": in_circuit_device_loss,
        **(
            {
                "machine_electromagnetic_w": window.get("machine_electromagnetic_mean_power_w"),
                "machine_note": "power converted by the machines (shaft side while motoring); "
                "not counted as a supply",
            }
            if model.machines
            else {}
        ),
        "net_storage_rate_w": storage_rate,
        "energy_balance": window.get("balance_status"),
    }
    if delivered and abs(storage_rate) > 0.01 * abs(delivered):
        summary["steady_state_warning"] = (
            f"stored energy is still changing at {storage_rate:.4g} W "
            f"({abs(storage_rate) / abs(delivered):.1%} of the input): the run has not "
            "reached steady state, so efficiency from this window is unreliable — "
            "simulate longer or use steady_state"
        )
    if efficiency_output:
        machine_ids = {machine.id for machine in model.machines}

        def output_power(cid: str) -> float:
            if cid in machine_ids:  # electromagnetic power of the whole machine
                return sum(
                    power for key, power in absorbed.items()
                    if key.startswith(f"{cid}.") and kinds.get(key) == "voltage_source"
                )
            return absorbed[cid]

        unknown = [cid for cid in efficiency_output if cid not in absorbed and cid not in machine_ids]
        if unknown:
            raise ToolInputError(
                f"efficiency_output names unknown components: {', '.join(unknown)}"
            )
        output = sum(output_power(cid) for cid in efficiency_output)
        result: dict[str, Any] = {
            "output_components": list(efficiency_output),
            "output_w": output,
            "input_w": delivered,
            "efficiency_in_circuit": (output / delivered) if delivered else None,
            "note": (
                "in-circuit efficiency counts only losses that are part of the circuit "
                "equations (resistors, on_resistance, forward_voltage)"
            ),
        }
        cards = bundle / "losses-thermal.json"
        if cards.is_file() and delivered:
            card_total = json.loads(cards.read_text(encoding="utf-8")).get("total_mean_power_w")
            if isinstance(card_total, (int, float)):
                result["post_processed_card_loss_w"] = card_total
                result["efficiency_with_card_losses_estimate"] = output / (delivered + card_total)
                result["note"] += (
                    "; the loss-card estimate is computed afterwards from ideal waveforms "
                    "over the whole run and is not drawn from the source in the simulation, "
                    "so efficiency_with_card_losses_estimate = P_out / (P_in + P_cards) is an "
                    "estimate, not a simulated quantity"
                )
        summary["efficiency"] = result
    return summary


def _harmonics_summary(bundle: Path) -> dict[str, Any] | None:
    path = bundle / "harmonics.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        signal["signal"]: {
            "fundamental_hz": signal["fundamental"]["frequency_hz"],
            "fundamental_rms": signal["fundamental"]["rms_amplitude"],
            "thd_fraction": signal["thd"].get("value_fraction"),
            "thd_harmonics": signal["thd"].get("included_harmonics"),
            "unit": signal.get("unit"),
        }
        for signal in data.get("signals", [])
    }


def simulate(
    workspace: Workspace,
    *,
    model_yaml: str | None = None,
    model_path: str | None = None,
    example: str | None = None,
    overrides: dict[str, Any] | None = None,
    timeout_s: float | None = None,
    efficiency_output: list[str] | None = None,
) -> dict[str, Any]:
    path = resolve_model(workspace, model_yaml, model_path, example)
    resolved_overrides = _overrides(overrides)
    model = _load(path, resolved_overrides)  # fail fast, in-process, with hints
    run_id, out = workspace.new_run_dir(model.name)
    _run_worker(
        {"op": "simulate", "model": str(path), "overrides": resolved_overrides, "out": str(out)},
        _timeout(timeout_s),
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    result: dict[str, Any] = {
        "run_id": run_id,
        "model": model.name,
        "run_sha256": summary.get("run_sha256"),
        "window": "statistics below are over the LAST HALF of the simulated time span",
        "signals": _sig(_signal_summary(metrics)),
        "stats": {
            key: summary.get("stats", {}).get(key)
            for key in ("integration_steps", "natural_events", "scheduled_events", "wall_time_s")
        },
        **_confidence(out),
        "warnings": _warnings(out),
        "artifacts": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    losses = _losses_summary(out)
    if losses is not None:
        result["losses"] = _sig(losses)
    harmonics = _harmonics_summary(out)
    if harmonics is not None:
        result["harmonics"] = _sig(harmonics)
    power = _power_summary(out, model, efficiency_output)
    if power is not None:
        result["power"] = _sig(power)
    result["settling"] = _sig(_settling(out, model))
    return result


def get_metrics(
    workspace: Workspace, *, run_id: str, signal: str, fields: list[str] | None = None
) -> dict[str, Any]:
    bundle = _run(workspace, run_id)
    metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
    if signal not in metrics:
        raise ToolInputError(f"signal {signal!r} not in run; available: {', '.join(sorted(metrics))}")
    payload = metrics[signal]
    if fields:
        missing = [field for field in fields if field not in payload]
        if missing:
            raise ToolInputError(
                f"unknown field(s) {', '.join(missing)}; available: {', '.join(sorted(payload))}"
            )
        payload = {field: payload[field] for field in fields}
    return {"run_id": run_id, "signal": signal, "metrics": payload}


def get_waveform(
    workspace: Workspace,
    *,
    run_id: str,
    signal: str,
    start_s: float | None = None,
    stop_s: float | None = None,
    max_points: int = 400,
) -> dict[str, Any]:
    if not 2 <= max_points <= MAX_WINDOW_POINTS:
        raise ToolInputError(f"max_points must lie in [2, {MAX_WINDOW_POINTS}]")
    bundle = _run(workspace, run_id)
    lower = float("-inf") if start_s is None else start_s
    upper = float("inf") if stop_s is None else stop_s
    times: list[float] = []
    values: list[float] = []
    with (bundle / "waveforms.csv").open(encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if signal not in header:
            raise ToolInputError(f"signal {signal!r} not recorded; available: {', '.join(header[1:])}")
        column = header.index(signal)
        for row in reader:
            time = float(row[0])
            if lower <= time <= upper:
                times.append(time)
                values.append(float(row[column]))
    if not times:
        raise ToolInputError("the window contains no samples")
    stride = max(1, -(-len(times) // max_points))
    return {
        "run_id": run_id,
        "signal": signal,
        "unit": _signal_unit(signal),
        "window_s": [times[0], times[-1]],
        "samples_in_window": len(times),
        "stride": stride,
        "downsampling": "plain stride; extremes between kept points are dropped — use get_metrics for exact min/max",
        "time_s": times[::stride],
        "values": values[::stride],
    }


def get_artifact(
    workspace: Workspace, *, run_id: str, name: str, path: str | None = None
) -> dict[str, Any]:
    bundle = _run(workspace, run_id)
    if "/" in name or "\\" in name or not name.endswith((".json", ".jsonl")):
        raise ToolInputError(
            "name must be a .json or .jsonl artifact file name from the run's artifact list"
        )
    if name.endswith(".jsonl"):
        return _jsonl_artifact(bundle, run_id, name, path)
    target = bundle / name
    if not target.is_file():
        available = ", ".join(sorted(p.name for p in bundle.glob("*.json")))
        raise ToolInputError(f"no artifact {name!r}; JSON artifacts: {available}")
    data: Any = json.loads(target.read_text(encoding="utf-8"))
    if path:
        for part in [p for p in path.split("/") if p]:
            if isinstance(data, list) and part.isdigit():
                data = data[int(part)]
            elif isinstance(data, dict) and part in data:
                data = data[part]
            else:
                keys = list(data)[:50] if isinstance(data, dict) else f"list of {len(data)}"
                raise ToolInputError(f"path segment {part!r} not found; available: {keys}")
    encoded = json.dumps(data)
    if len(encoded) > MAX_ARTIFACT_BYTES:
        keys = list(data)[:80] if isinstance(data, dict) else f"list of {len(data)} items"
        raise ToolInputError(
            f"selection is {len(encoded)} bytes (limit {MAX_ARTIFACT_BYTES}); narrow it with "
            f"path=... — top-level entries: {keys}"
        )
    return {"run_id": run_id, "name": name, "path": path or "", "data": data}


def _jsonl_artifact(bundle: Path, run_id: str, name: str, path: str | None) -> dict[str, Any]:
    """Line-oriented artifacts (events, control/signal traces): read a line range.

    ``path`` is "start-stop" (0-based, stop exclusive) or a single start index;
    at most 200 lines per call. Without it the first 50 lines and the count.
    """

    target = bundle / name
    if not target.is_file():
        available = ", ".join(sorted(p.name for p in bundle.glob("*.jsonl")))
        raise ToolInputError(f"no artifact {name!r}; line artifacts: {available}")
    lines = target.read_text(encoding="utf-8").splitlines()
    start, stop = 0, 50
    if path:
        try:
            if "-" in path:
                first, last = path.split("-", 1)
                start, stop = int(first), int(last)
            else:
                start = int(path)
                stop = start + 50
        except ValueError as exc:
            raise ToolInputError("for a .jsonl artifact, path is a line range such as 0-100") from exc
    if not 0 <= start < stop or stop - start > 200:
        raise ToolInputError("line range must satisfy 0 <= start < stop and cover at most 200 lines")
    rows = [json.loads(line) for line in lines[start:stop]]
    return {
        "run_id": run_id,
        "name": name,
        "total_lines": len(lines),
        "range": [start, min(stop, len(lines))],
        "rows": rows,
        "note": "use path='start-stop' to read further (max 200 lines per call)",
    }


def compare_runs(
    workspace: Workspace,
    *,
    run_ids: list[str],
    signals: list[str] | None = None,
    metrics: list[str] | None = None,
) -> dict[str, Any]:
    if not 2 <= len(run_ids) <= 12:
        raise ToolInputError("give between 2 and 12 run_ids")
    wanted_metrics = metrics or ["mean_last_half", "rms_last_half", "peak_to_peak_last_half"]
    table: dict[str, dict[str, Any]] = {}
    for run_id in run_ids:
        data = json.loads((_run(workspace, run_id) / "metrics.json").read_text(encoding="utf-8"))
        chosen = signals or sorted(data)
        for signal in chosen:
            row = table.setdefault(signal, {})
            entry = data.get(signal)
            row[run_id] = None if entry is None else {m: entry.get(m) for m in wanted_metrics}
    return {
        "run_ids": run_ids,
        "metrics": wanted_metrics,
        "table": table,
        "note": "side-by-side metrics only; a missing signal in a run shows as null",
    }


def list_runs(workspace: Workspace) -> dict[str, Any]:
    return {"runs": workspace.list_runs()[:200], "workspace": str(workspace.root)}


def steady_state(
    workspace: Workspace,
    *,
    model_yaml: str | None = None,
    model_path: str | None = None,
    example: str | None = None,
    overrides: dict[str, Any] | None = None,
    timeout_s: float | None = None,
    efficiency_output: list[str] | None = None,
) -> dict[str, Any]:
    path = resolve_model(workspace, model_yaml, model_path, example)
    resolved_overrides = _overrides(overrides)
    model = _load(path, resolved_overrides)
    run_id, out = workspace.new_run_dir(model.name, kind="steady")
    result = _run_worker(
        {
            "op": "steady_state",
            "model": str(path),
            "overrides": resolved_overrides,
            "cycle_out": str(out),
        },
        _timeout(timeout_s),
    )
    result.pop("mode_intervals", None)
    cycle = result.pop("steady_cycle", None)
    if cycle and cycle.get("status") == "run":
        bundle = Path(cycle["bundle"])
        metrics = json.loads((bundle / "metrics.json").read_text(encoding="utf-8"))
        signals = {}
        for signal, values in metrics.items():
            if isinstance(values, dict):
                signals[signal] = {
                    key: values.get(key) for key in ("mean", "rms", "min", "max", "peak_to_peak")
                }
                unit = _signal_unit(signal)
                if unit:
                    signals[signal]["unit"] = unit
        result["steady_cycle"] = {
            "run_id": run_id,
            "periods": cycle["periods"],
            "note": "a fresh run started ON the periodic orbit: these are steady-state "
            "values (ripple = peak_to_peak) with no start-up transient. Waveforms: "
            "get_waveform(run_id=..., ...) reads this cycle.",
            "signals": _sig(signals),
        }
        result.update(_confidence(bundle))
        result["warnings"] = _warnings(bundle)
        # The cycle run spans two periods on the orbit, so its "last half" is
        # exactly one steady-state period: the cleanest power measurement there is.
        power = _power_summary(bundle, model, efficiency_output)
        if power is not None:
            power["window"] = "one switching period on the periodic orbit (steady state)"
            result["steady_cycle"]["power"] = _sig(power)
        # Let the generic run readers find the cycle run under its run_id.
        for item in bundle.iterdir():
            item.rename(out / item.name)
        bundle.rmdir()
    elif cycle:
        result["steady_cycle"] = cycle
    return _sig(result)


def loop_gain(
    workspace: Workspace,
    *,
    request: dict[str, Any],
    model_yaml: str | None = None,
    model_path: str | None = None,
    example: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    path = resolve_model(workspace, model_yaml, model_path, example)
    model = _load(path, None)
    payload = dict(request)
    payload.setdefault("schema", "pear-sim/stability-request/v0.1")
    run_id, out = workspace.new_run_dir(model.name, kind="stability")
    _run_worker(
        {"op": "stability", "model": str(path), "out": str(out), "request": payload},
        _timeout(timeout_s),
    )
    evidence = json.loads((out / "stability-evidence.json").read_text(encoding="utf-8"))
    result = {
        "run_id": run_id,
        "overall_status": evidence.get("overall_status"),
        "gates": evidence.get("gates"),
        "claim_boundary": evidence.get("claim_boundary"),
        "loop_gain_error": evidence.get("loop_gain_error"),
        "artifacts": sorted(p.name for p in out.iterdir() if p.is_file()),
    }
    loop = out / "loop-gain.json"
    if loop.is_file():
        data = json.loads(loop.read_text(encoding="utf-8"))
        result["margins"] = _sig(
            {
                key: data.get(key)
                for key in (
                    "phase_margin_deg",
                    "gain_crossover_hz",
                    "gain_margin_db",
                    "phase_crossover_hz",
                    "gain_crossovers_hz",
                    "phase_crossovers_hz",
                    "validity_max_hz",
                    "validity_note",
                )
            }
        )
        result["sampling"] = data.get("sampling")
        if len(data.get("gain_crossovers_hz") or []) > 1:
            result["multiple_crossovers"] = (
                f"the loop crosses 0 dB {len(data['gain_crossovers_hz'])} times; the phase "
                "margin reported is the WORST of them (at gain_crossover_hz)"
            )
        result["note"] = (
            "Bode data (frequencies_hz, loop_magnitude_db, loop_phase_deg) is in "
            "loop-gain.json: get_artifact(run_id, 'loop-gain.json', path=...)"
        )
    gate_ids = {gate.get("id") for gate in result.get("gates") or []}
    if not any("margin" in str(gate_id) for gate_id in gate_ids):
        result["warning"] = (
            "overall_status covers ONLY the gates listed (steady state, operating mode); "
            "no margin requirement was declared, so stability was NOT judged. Read "
            "margins yourself, or pass request.margins {min_phase_margin_deg, "
            "min_gain_margin_db} to make them a gate."
        )
    return result


def _request_file(workspace: Workspace, request: dict[str, Any], out: Path, schema: str) -> Path:
    payload = dict(request)
    payload.setdefault("schema", schema)
    out.parent.mkdir(parents=True, exist_ok=True)
    path = out.parent / f"{out.name}.request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def sweep_parameter(
    workspace: Workspace,
    *,
    request: dict[str, Any],
    model_yaml: str | None = None,
    model_path: str | None = None,
    example: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    path = resolve_model(workspace, model_yaml, model_path, example)
    model = _load(path, None)
    run_id, out = workspace.new_run_dir(model.name, kind="sweep")
    request_path = _request_file(workspace, request, out, "pear-sim/parameter-sweep-request/v0.1")
    result = _run_worker(
        {"op": "sweep_parameter", "model": str(path), "request_path": str(request_path), "out": str(out)},
        _timeout(timeout_s),
    )
    summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
    points = []
    for point in summary.get("points") or []:
        points.append(
            {
                "value_si": point.get("parameter_value_si"),
                "status": point.get("execution_status"),
                "projections": {
                    item["id"]: item.get("value") if item.get("status") == "recorded"
                    else f"{item.get('status')}: {item.get('reason')}"
                    for item in point.get("projections", [])
                },
            }
        )
    return {
        "run_id": run_id,
        "parameter": summary.get("parameter"),
        "execution_status": result["execution_status"],
        "points": _sig(points),
        "note": "each point is a retained run inside this sweep's bundle",
    }


def evaluate_designs(
    workspace: Workspace,
    *,
    request: dict[str, Any],
    model_yaml: str | None = None,
    model_path: str | None = None,
    example: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    path = resolve_model(workspace, model_yaml, model_path, example)
    model = _load(path, None)
    run_id, out = workspace.new_run_dir(model.name, kind="batch")
    request_path = _request_file(workspace, request, out, "pear-sim/batch-request/v0.1")
    result = _run_worker(
        {"op": "evaluate_designs", "model": str(path), "request_path": str(request_path), "out": str(out)},
        _timeout(timeout_s),
    )
    summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "status": summary["status"],
        "claim": summary["claim"],
        "design_count": summary["design_count"],
        "feasible_count": summary["feasible_count"],
        "failed_count": summary["failed_count"],
        "best_design": summary["best_design"],
        "ranking": _sig(summary["ranking"][:10]),
        "finalists": summary["finalists"],
        "failures": summary["failures"][:10],
        "note": (
            "designs.csv in this bundle lists every design; finalists are full runs "
            "under finalists/<design_id>/run"
        ),
    }


def sensitivity(
    workspace: Workspace,
    *,
    request: dict[str, Any],
    model_yaml: str | None = None,
    model_path: str | None = None,
    example: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    path = resolve_model(workspace, model_yaml, model_path, example)
    model = _load(path, None)
    run_id, out = workspace.new_run_dir(model.name, kind="sensitivity")
    request_path = _request_file(workspace, request, out, "pear-sim/sensitivity-request/v0.1")
    result = _run_worker(
        {"op": "sensitivity", "model": str(path), "request_path": str(request_path), "out": str(out)},
        _timeout(timeout_s),
    )
    summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
    return {
        "run_id": run_id,
        "method": summary["method"],
        "status": summary["status"],
        "claim": summary["claim"],
        "run_count": summary["run_count"],
        "failed_count": summary["failed_count"],
        "outputs": _sig(summary["outputs"]),
        "note": "runs.csv in this bundle lists every run's inputs and outputs",
    }


def sweep_tolerance_corners(
    workspace: Workspace,
    *,
    request: dict[str, Any],
    model_yaml: str | None = None,
    model_path: str | None = None,
    example: str | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    path = resolve_model(workspace, model_yaml, model_path, example)
    model = _load(path, None)
    run_id, out = workspace.new_run_dir(model.name, kind="corners")
    request_path = _request_file(workspace, request, out, "pear-sim/tolerance-corners-request/v0.1")
    result = _run_worker(
        {"op": "tolerance_corners", "model": str(path), "request_path": str(request_path), "out": str(out)},
        _timeout(timeout_s),
    )
    summary = json.loads(Path(result["summary"]).read_text(encoding="utf-8"))
    corners_dir = Path(result["bundle"]) / "corners"
    settling = {}
    warnings: list[str] = []
    for corner in sorted(corners_dir.iterdir()) if corners_dir.is_dir() else []:
        run_dir = corner / "run"
        if run_dir.is_dir():
            settling[corner.name] = _settling(run_dir, model)["status"]
            if not warnings:
                warnings = _warnings(run_dir)
    unsettled = sorted(name for name, status in settling.items() if status == "not_settled")
    payload: dict[str, Any] = {
        "run_id": run_id,
        "execution_status": result["execution_status"],
        "binding_corner": summary.get("binding_corner"),
        "bands": _sig(summary.get("bands")),
        "settling": settling,
        "warnings": warnings,
    }
    if unsettled:
        payload["settling_warning"] = (
            f"corners {', '.join(unsettled)} had not reached steady state: every corner starts "
            "from the nominal model's initial conditions, so their last-half values include "
            "the start-up transient and the bands may show it instead of the tolerance effect. "
            "Lengthen the run (analysis.transient.stop) and repeat."
        )
    return payload


def render_schematic(
    workspace: Workspace,
    *,
    model_yaml: str | None = None,
    model_path: str | None = None,
    example: str | None = None,
) -> str:
    from pear_sim.api.facade import render

    path = resolve_model(workspace, model_yaml, model_path, example)
    model = _load(path, None)
    return render(model).svg


def capabilities(capability_id: str | None = None) -> dict[str, Any]:
    from pear_sim.api.capabilities import build_capabilities

    catalog = build_capabilities()
    entries = catalog.get("capabilities", [])
    if capability_id:
        for entry in entries:
            if entry.get("id") == capability_id:
                return entry
        raise ToolInputError(f"unknown capability id {capability_id!r}")
    return {
        "version": __version__,
        "capabilities": [
            {"id": e.get("id"), "status": e.get("status"), "scope": e.get("scope")}
            for e in entries
        ],
        "hint": "call with capability_id=<id> for the full statement and evidence",
    }


def doctor() -> dict[str, Any]:
    from pear_sim.api.doctor import run_doctor

    return run_doctor()


def _run(workspace: Workspace, run_id: str) -> Path:
    try:
        return workspace.run_dir(run_id)
    except WorkspaceError as exc:
        raise ToolInputError(str(exc)) from exc
