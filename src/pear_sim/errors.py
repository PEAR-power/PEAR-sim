# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
class PearSimError(Exception):
    """Base class for actionable user-facing errors.

    Every subclass declares a stable machine-readable ``code`` so agent
    clients can branch on failures without parsing message text.  Codes are
    part of the public contract: changing one is a breaking change.
    """

    code = "pear_sim.error"


class ModelError(PearSimError):
    """The input model is invalid."""

    code = "model.invalid"


class UnitError(ModelError):
    """A value has an invalid or incompatible unit."""

    code = "model.units"


class TopologyError(PearSimError):
    """A circuit topology cannot be solved consistently."""

    code = "model.topology"


class SimulationError(PearSimError):
    """The hybrid simulation failed."""

    code = "simulation.failed"


class StateDiscontinuityError(TopologyError):
    """A mode change would force a capacitor voltage or inductor current to jump.

    Raised when a switch or diode closes a loop of capacitors (and voltage
    sources) whose voltages disagree, or opens the last path of inductor
    current. An ideal kernel would have to move charge or flux in zero time,
    which it does not do silently: every finding names the loop or cut, the
    jump (volts or amperes), the instant and the event that caused it.
    """

    code = "model.state_discontinuity"

    def __init__(self, findings: list[dict], *, time: float | None, event: str | None = None):
        self.findings = list(findings)
        self.time = time
        self.event = event
        super().__init__()

    def details(self) -> dict:
        return {
            "diagnostic": "state_discontinuity",
            "time_s": self.time,
            "event": self.event,
            "findings": self.findings,
        }

    def __str__(self) -> str:
        when = "" if self.time is None else f" at t={self.time:.12g} s"
        cause = f" after {self.event}" if self.event else ""
        parts = []
        for finding in self.findings:
            members = " - ".join(finding["members"])
            if finding["kind"] == "capacitor_loop":
                parts.append(
                    f"{finding['closed_by']} closes the loop {members} whose voltages "
                    f"disagree by {finding['jump']:.6g} V"
                )
            else:
                text = (
                    f"opening the last current path at node {finding['node']!r} cuts "
                    f"inductors {members} that still carry a net {finding['jump']:.6g} A"
                )
                if finding.get("open_switches"):
                    text += (
                        f" (an ideal_switch has no body diode: add an antiparallel diode "
                        f"across {', '.join(finding['open_switches'])}, e.g. for dead time)"
                    )
                parts.append(text)
        return (
            f"state_discontinuity{when}{cause}: " + "; ".join(parts) + ". The ideal "
            "kernel would have to move charge or flux in zero time and refuses instead "
            "of doing it silently. Model the physical path that limits it: an `esr` or "
            "a switch `on_resistance` in a capacitor loop, a freewheeling diode, snubber "
            "or clamp across an opened inductor current"
        )
