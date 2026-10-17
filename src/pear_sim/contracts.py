"""Compatibility alias; this module moved to pear_sim.run.contracts (stage-3 restructuring)."""

import sys

import pear_sim.run.contracts as _moved

sys.modules[__name__] = _moved
