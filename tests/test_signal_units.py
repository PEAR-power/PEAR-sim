# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""One source of truth for the unit an output signal carries.

The rule used to live in six copies of ``"V" if name.startswith("v(") else
"A"``. Those copies agreed only because the rule was small; the failure mode
they share is that the LAST branch is a fallback, so any expression that is
neither a voltage nor a current silently becomes an ampere. These tests pin
the single derivation and, above all, pin that an unknown kind RAISES.
"""

from __future__ import annotations

import pytest

from pear_sim.errors import ModelError
from pear_sim.spec.signals import (
    SIGNAL_UNITS,
    is_signal_expression,
    parse_signal_expression,
    signal_unit,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("v(out)", "V"),
        ("v(node.with.dots)", "V"),
        ("i(l1)", "A"),
        ("i(motor.a.l)", "A"),
    ],
)
def test_known_kinds_carry_their_unit(expression: str, expected: str) -> None:
    assert signal_unit(expression) == expected


def test_unknown_kind_raises_instead_of_falling_back() -> None:
    # The whole point: no fallback branch. An unknown kind has an unknown
    # unit, and inventing one is how a torque would become an ampere.
    # (`torque` is a declared kind now; `power` stands in as the undeclared
    # one, and this test keeps working as the table grows.)
    with pytest.raises(ModelError, match="canonical signal expression"):
        signal_unit("power(motor)")


@pytest.mark.parametrize(
    "expression",
    ["", "v()", "v(1bad)", "vout", "v out", "V(out)", "i(out) ", " v(out)"],
)
def test_malformed_expressions_are_rejected_or_stripped(expression: str) -> None:
    stripped = expression.strip()
    if stripped in {"v(out)", "i(out)"}:
        assert is_signal_expression(expression)
        return
    assert not is_signal_expression(expression)
    with pytest.raises(ModelError):
        signal_unit(expression)


def test_parse_returns_kind_and_target() -> None:
    assert parse_signal_expression("i(motor.a.l)") == ("i", "motor.a.l")


def test_every_declared_kind_is_derivable() -> None:
    # Guards the table against a kind added without a unit, and against a
    # regex that stops matching a declared kind.
    for kind, unit in SIGNAL_UNITS.items():
        assert signal_unit(f"{kind}(target)") == unit
