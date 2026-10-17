"""Agent-facing surfaces: CLI, capability declarations, and the typed facade."""

from pear_sim.api.facade import *  # noqa: F401,F403
from pear_sim.api.facade import __all__ as _facade_all

__all__ = list(_facade_all)
