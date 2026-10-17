"""Compatibility alias; this module moved to pear_sim.spec.units (stage-3 restructuring)."""

import sys

import pear_sim.spec.units as _moved

sys.modules[__name__] = _moved
