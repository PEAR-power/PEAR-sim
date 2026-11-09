# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Graders for the MCP agent evaluation.

Every grader receives the agent's final answer and the server workspace it
used. Reference values come either from closed forms or from running the same
shipped example through the Python API here — never from the agent's own
claims — and numbers quoted in the answer must match them.
"""

from __future__ import annotations

import json
import re
import tempfile
from functools import cache
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"

_SCALE = {
    "m": 1e-3, "u": 1e-6, "µ": 1e-6, "n": 1e-9, "k": 1e3, "M": 1e6, "": 1.0,
}
_NUMBER = re.compile(
    r"(?<![\w.])(-?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?)\s*"
    r"(%|[munµkM]?(?:V|A|W|Hz|s|Ω|ohm|ohms)|°C|deg|degrees)?",
)


def numbers(text: str) -> list[tuple[float, str]]:
    """Every number in ``text`` with its unit, scaled to SI (``%`` kept as %)."""

    found = []
    for raw, unit in _NUMBER.findall(text.replace("−", "-")):
        value = float(raw.replace(",", "."))
        unit = unit or ""
        base = unit
        if unit and unit not in ("%", "°C", "deg", "degrees") and unit[0] in _SCALE and len(unit) > 1:
            if unit[1:] in ("V", "A", "W", "Hz", "s", "Ω", "ohm", "ohms"):
                value *= _SCALE[unit[0]]
                base = unit[1:]
        found.append((value, base))
    return found


def has_number(text: str, target: float, rel: float, units: tuple[str, ...] | None = None) -> bool:
    for value, unit in numbers(text):
        if units is not None and unit not in units and unit != "":
            continue
        if target == 0:
            if abs(value) <= rel:
                return True
        elif abs(abs(value) - abs(target)) <= rel * abs(target):
            return True
    return False


def runs(workspace: Path) -> list[Path]:
    root = workspace / "runs"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def metrics(run: Path) -> dict:
    path = run / "metrics.json"
    return json.loads(path.read_text()) if path.is_file() else {}


@cache
def _simulate_example(name: str) -> Path:
    from pear_sim.api.facade import simulate

    out = Path(tempfile.mkdtemp(prefix=f"eval-ref-{name}-")) / "run"
    simulate(EXAMPLES / f"{name}.yaml", out)
    return out


Result = tuple[bool, str]


def shipped_buck_mean(answer: str, workspace: Path) -> Result:
    reference = metrics(_simulate_example("buck"))["v(out)"]["mean_last_half"]
    ok = has_number(answer, reference, 0.01, ("V",))
    return ok, f"reference mean v(out) = {reference:.4g} V"


def sync_buck_48_12(answer: str, workspace: Path) -> Result:
    candidates = []
    for run in runs(workspace):
        data = metrics(run)
        vout, il = data.get("v(out)"), data.get("i(l1)")
        if not vout or not il:
            continue
        key = "" if run.name.startswith("steady-") else "_last_half"
        mean_v = vout.get(f"mean{key}")
        if mean_v and 11.3 <= mean_v <= 12.7:
            candidates.append((run.name.startswith("steady-"), run, vout, il, key))
    if not candidates:
        return False, "no run with a ~12 V output and an i(l1) signal in the workspace"
    candidates.sort(key=lambda item: (item[0], item[1].name))
    _, run, vout, il, key = candidates[-1]
    ripple_v = vout[f"peak_to_peak{key}"]
    ripple_i = il[f"peak_to_peak{key}"]
    met = ripple_v < 0.05
    quoted = has_number(answer, ripple_v, 0.10, ("V",)) and has_number(answer, ripple_i, 0.10, ("A",))
    detail = (f"run {run.name}: ripple v(out) {ripple_v * 1e3:.3g} mV, i(l1) {ripple_i:.3g} A; "
              f"requirement met={met}, quoted={quoted}")
    return met and quoted, detail


def boost_12_24(answer: str, workspace: Path) -> Result:
    for run in reversed(runs(workspace)):
        data = metrics(run)
        vout = data.get("v(out)")
        inductor = next((v for k, v in data.items() if k.startswith("i(l")), None)
        if not vout or not inductor:
            continue
        key = "" if run.name.startswith("steady-") else "_last_half"
        mean_v = vout.get(f"mean{key}")
        if mean_v and abs(mean_v - 24) <= 0.04 * 24:
            ok = has_number(answer, mean_v, 0.02, ("V",)) and has_number(
                answer, inductor[f"peak_to_peak{key}"], 0.10, ("A",)
            )
            return ok, f"run {run.name}: mean v(out) {mean_v:.4g} V, ripple {inductor[f'peak_to_peak{key}']:.3g} A"
    return False, "no run with a ~24 V output and an inductor current"


def _thd(name: str) -> float:
    data = json.loads((_simulate_example(name) / "harmonics.json").read_text())
    signal = next(s for s in data["signals"] if s["signal"] == "v(vab)")
    return signal["thd"]["value_fraction"]


def spwm_vs_svpwm_thd(answer: str, workspace: Path) -> Result:
    spwm, svpwm = _thd("inverter_three_phase_spwm"), _thd("inverter_three_phase_svpwm")

    def quoted(fraction: float) -> bool:
        return has_number(answer, fraction * 100, 0.05, ("%",)) or has_number(answer, fraction, 0.05)

    lower = "svpwm" if svpwm < spwm else "spwm"
    text = answer.lower()
    named_lower = re.search(rf"\b{lower}\b[^.\n]{{0,80}}\b(lower|smaller|better|less)\b", text) or re.search(
        rf"\b(lower|smaller|less)\b[^.\n]{{0,60}}\b{lower}\b", text
    )
    ok = quoted(spwm) and quoted(svpwm) and bool(named_lower)
    return ok, f"reference THD SPWM {spwm:.4%}, SVPWM {svpwm:.4%}; lower: {lower}"


def buck_dcm_boundary(answer: str, workspace: Path) -> Result:
    # Ideal buck: R_crit = 2 L fsw / (1 - D), shipped example L = 1 mH, 20 kHz, D = 40 %.
    reference = 2 * 1e-3 * 20e3 / (1 - 0.4)
    return has_number(answer, reference, 0.10, ("Ω", "ohm", "ohms")), f"closed form R_crit = {reference:.4g} ohm"


def loop_gain_margin(answer: str, workspace: Path) -> Result:
    from pear_sim.mcp import tools
    from pear_sim.mcp.workspace import Workspace

    scratch = Workspace.create(workspace=Path(tempfile.mkdtemp(prefix="eval-lg-")), project_root=ROOT)
    result = tools.loop_gain(
        scratch,
        example="buck",
        request={
            "output": "v(out)",
            "compensator": {"numerator": [0.01, 100], "denominator": [1, 0]},
            "sensor_gain": 1.0,
            "f_min_hz": 10,
            "f_max_hz": 10000,
        },
    )
    margins = result.get("margins") or {}
    pm = margins.get("phase_margin_deg")
    fc = margins.get("gain_crossover_hz")
    if pm is None:
        return False, f"reference has no phase margin: {margins}"
    ok = has_number(answer, pm, 0.03, ("deg", "degrees", "")) or any(
        abs(v - pm) <= 1.0 for v, _ in numbers(answer)
    )
    if fc:
        ok = ok and has_number(answer, fc, 0.05, ("Hz",))
    return ok, f"reference PM {pm:.3g} deg, crossover {fc}"


def fix_broken_model(answer: str, workspace: Path) -> Result:
    fixed = [
        run for run in runs(workspace)
        if run.name.endswith("half_bridge_rl") and (run / "summary.json").is_file()
    ]
    mentions_diode = "diode" in answer.lower()
    return bool(fixed) and mentions_diode, f"successful half_bridge_rl runs: {len(fixed)}; diode named: {mentions_diode}"


def loss_breakdown(answer: str, workspace: Path) -> Result:
    data = json.loads((_simulate_example("buck_with_loss_cards") / "losses-thermal.json").read_text())
    devices = data["devices"]
    top = max(devices, key=lambda d: d["conduction"]["mean_power_w"])
    tj = max(d["thermal"]["tj_max_c"] for d in devices)
    kind_words = {"diode": ("diode", "d1"), "switch": ("igbt", "switch", "q1", "transistor")}[top["kind"]]
    named = any(word in answer.lower() for word in kind_words)
    ok = named and has_number(answer, top["conduction"]["mean_power_w"], 0.02, ("W",)) and any(
        abs(v - tj) <= 1.0 for v, _ in numbers(answer)
    )
    return ok, (f"reference: {top['component_id']} ({top['kind']}) "
                f"{top['conduction']['mean_power_w']:.3g} W, Tj max {tj:.3g} C")


def dead_time_effect(answer: str, workspace: Path) -> Result:
    # Each dead-time interval hands the switch node to a diode for t_dead per period:
    # d<Vout>/d t_dead = -Vin * fsw = -100 V * 20 kHz = -2 V/us on the shipped model.
    return has_number(answer, 2.0, 0.10), "closed form slope = -Vin*fsw = -2 V/us"


def claim_boundary(answer: str, workspace: Path) -> Result:
    text = answer.lower()
    negative = bool(re.search(r"\b(not|no|shouldn't|should not|cannot|can't|don't)\b", text))
    reasons = [w for w in ("ideal", "synthetic", "experimental", "alpha", "post-process") if w in text]
    return negative and len(reasons) >= 2, f"negative={negative}, reasons={reasons}"


def mosfet_buck_efficiency(answer: str, workspace: Path) -> Result:
    from pear_sim.mcp import tools
    from pear_sim.mcp.workspace import Workspace

    scratch = Workspace.create(workspace=Path(tempfile.mkdtemp(prefix="eval-eff-")), project_root=ROOT)
    reference = tools.steady_state(scratch, example="sync_buck_mosfet", efficiency_output=["load"])
    power = reference["steady_cycle"]["power"]
    eta = power["efficiency"]["efficiency_in_circuit"]
    loss = power["device_conduction_loss_in_circuit_w"]
    ok = (has_number(answer, eta * 100, 0.002, ("%",)) or has_number(answer, eta, 0.002)) and has_number(
        answer, loss, 0.05, ("W",)
    )
    return ok, f"reference efficiency {eta:.4%}, device conduction loss {loss:.3g} W"


CHECKS: dict[str, Callable[[str, Path], Result]] = {
    name: function
    for name, function in globals().items()
    if callable(function) and name in {
        "shipped_buck_mean", "sync_buck_48_12", "boost_12_24", "spwm_vs_svpwm_thd",
        "buck_dcm_boundary", "loop_gain_margin", "fix_broken_model", "loss_breakdown",
        "dead_time_effect", "claim_boundary", "mosfet_buck_efficiency",
    }
}
