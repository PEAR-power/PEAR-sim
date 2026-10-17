"""Output-signal expressions and the unit each one carries.

WHY THIS MODULE EXISTS. The physical unit of an output was, until now,
re-derived at every site that needed it, always by the same two-branch
guess: ``"V" if name.startswith("v(") else "A"``. Six copies of one rule
(solver metrics, harmonic records, bundle validation, run comparison,
parameter sweeps, reports) agreed only because the rule was trivially
small. It stops being trivially small the moment a signal is neither a
voltage nor a current — a rotor angle, a speed, a torque — and then the
LAST branch of that guess quietly relabels newtons as amperes.

The unit is therefore derived in ONE place, from the expression's leading
kind, and every consumer asks. A kind that is not declared here has no
unit, and asking for one is an error rather than a fallback: unit guessing
is exactly the silent assumption this repository refuses.
"""

from __future__ import annotations

import re
from typing import Final

from pear_sim.errors import ModelError

# Expression kinds and the unit each one carries. The key is the leading
# identifier of the expression, the value is the unit string that appears in
# artifacts. Units are written in the repository's flat ASCII form ("N*m",
# not "N·m") because they travel through JSON, filenames, and comparison
# records where a non-ASCII byte is a portability question nobody needs.
SIGNAL_UNITS: Final[dict[str, str]] = {
    "v": "V",
    "i": "A",
    "torque": "N*m",
    "speed_mech": "rad/s",
    "speed_elec": "rad/s",
    "angle_mech": "rad",
    "angle_elec": "rad",
}

# Kinds whose target is a declared machine rather than a circuit node or
# component. They are sampled from the mechanical state, not from the MNA
# solution, which is why they cannot be an affine map over the electrical
# state and why the metric layer treats them separately.
MACHINE_SIGNAL_KINDS: Final[frozenset[str]] = frozenset(
    {"torque", "speed_mech", "speed_elec", "angle_mech", "angle_elec"}
)

# WHY BOTH MECHANICAL AND ELECTRICAL ARE NAMED, AND NEITHER IS "angle".
# The mechanical state holds a MECHANICAL speed and an ELECTRICAL angle —
# they differ by the pole-pair count, and the mismatch is not visible in the
# field names. A single `angle(m)` or `speed(m)` would therefore be right
# half the time and silently wrong the other half, in a way no test catches
# because both are plain radians. Every drive convention bug starts exactly
# here, so the model has to say which one it means.

_TARGET = r"[A-Za-z_][A-Za-z0-9_.:-]*"
_EXPRESSION_RE: Final[re.Pattern[str]] = re.compile(
    rf"^({'|'.join(sorted(SIGNAL_UNITS, key=len, reverse=True))})\(({_TARGET})\)$"
)


def parse_signal_expression(expression: str) -> tuple[str, str]:
    """Split ``kind(target)`` into its kind and target, or fail closed."""

    match = _EXPRESSION_RE.fullmatch(expression.strip())
    if match is None:
        raise ModelError(
            f"{expression!r} is not a canonical signal expression; expected one "
            f"of {', '.join(sorted(SIGNAL_UNITS))} applied to a target name"
        )
    return match.group(1), match.group(2)


def signal_unit(expression: str) -> str:
    """Unit carried by ``expression``.

    Raises rather than defaulting: an expression whose kind is unknown has an
    unknown unit, and labelling it with the last branch of an if-statement is
    how a torque becomes an ampere.
    """

    kind, _ = parse_signal_expression(expression)
    return SIGNAL_UNITS[kind]


def is_signal_expression(expression: str) -> bool:
    """Whether ``expression`` is a canonical signal expression."""

    return _EXPRESSION_RE.fullmatch(expression.strip()) is not None


#: A declared signal-layer block, read as an output. Deliberately NOT in
#: SIGNAL_UNITS: its unit comes from the block's own declaration, because a
#: control layer carries volts, amperes, newton-metres and dimensionless
#: quantities side by side and no syntax can tell them apart. Asking
#: `signal_unit` for one therefore raises, which forces every consumer to go
#: through `unit_for` and take the declared value instead of guessing.
_DECLARED_SIGNAL_RE: Final[re.Pattern[str]] = re.compile(rf"^s\(({_TARGET})\)$")


def is_declared_signal(expression: str) -> bool:
    """Whether ``expression`` reads a declared signal-layer block."""

    return _DECLARED_SIGNAL_RE.fullmatch(expression.strip()) is not None


def declared_signal_target(expression: str) -> str | None:
    """Block id addressed by ``expression``, or ``None``."""

    match = _DECLARED_SIGNAL_RE.fullmatch(expression.strip())
    return match.group(1) if match else None


def unit_for(expression: str, declared_units: dict[str, str] | None = None) -> str:
    """Unit of any recordable expression, declared values included."""

    target = declared_signal_target(expression)
    if target is None:
        return signal_unit(expression)
    if not declared_units or target not in declared_units:
        raise ModelError(
            f"{expression!r} reads a signal block whose unit was not declared"
        )
    return declared_units[target]


def is_machine_signal(expression: str) -> bool:
    """Whether ``expression`` reads a machine's mechanical state."""

    match = _EXPRESSION_RE.fullmatch(expression.strip())
    return match is not None and match.group(1) in MACHINE_SIGNAL_KINDS


def machine_signal_target(expression: str) -> str | None:
    """Machine id addressed by ``expression``, or ``None`` if it is electrical."""

    match = _EXPRESSION_RE.fullmatch(expression.strip())
    if match is None or match.group(1) not in MACHINE_SIGNAL_KINDS:
        return None
    return match.group(2)
