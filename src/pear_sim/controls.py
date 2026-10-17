"""Compatibility alias; this module moved to pear_sim.spec.controls (stage-3 restructuring)."""

import sys

import pear_sim.spec.controls as _moved

sys.modules[__name__] = _moved
