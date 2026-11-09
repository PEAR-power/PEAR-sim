# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Errors for the design-specification layer."""

from __future__ import annotations

from pear_sim.errors import PearSimError


class DesignSpecError(PearSimError):
    """The design specification is invalid (wrong shape, mixed sections,
    unknown fields, or an assumption without a named human)."""

    code = "design_spec.invalid"


class IncompleteDesignSpecError(DesignSpecError):
    """The specification still has open questions; it cannot be consumed
    by selection or hashed. The open questions ARE the payload — the caller
    is expected to surface them to an engineer, not to fill defaults."""

    code = "design_spec.incomplete"
