"""Compatibility alias; this module moved to pear_sim.api.capabilities (stage-3 restructuring)."""

import sys

import pear_sim.api.capabilities as _moved

sys.modules[__name__] = _moved
