"""Compatibility alias; this module moved to pear_sim.evidence.run_compare (stage-3 restructuring)."""

import sys

import pear_sim.evidence.run_compare as _moved

sys.modules[__name__] = _moved
