"""Compatibility alias; this module moved to pear_sim.reporting.render (stage-3 restructuring)."""

import sys

import pear_sim.reporting.render as _moved

sys.modules[__name__] = _moved
