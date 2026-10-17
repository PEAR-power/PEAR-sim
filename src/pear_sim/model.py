"""Compatibility alias; this module moved to pear_sim.spec.model (stage-3 restructuring)."""

import sys

import pear_sim.spec.model as _moved

sys.modules[__name__] = _moved
