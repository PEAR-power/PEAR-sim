# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Agent-facing surfaces: CLI, capability declarations, and the typed facade."""

from pear_sim.api.facade import *  # noqa: F401,F403
from pear_sim.api.facade import __all__ as _facade_all

__all__ = list(_facade_all)
