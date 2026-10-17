"""Compatibility alias; this module moved to pear_sim.analysis.three_phase_modulation (stage-3 restructuring)."""

import sys

import pear_sim.analysis.three_phase_modulation as _moved

sys.modules[__name__] = _moved
