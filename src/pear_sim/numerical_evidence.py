"""Compatibility alias; this module moved to pear_sim.evidence.numerical_evidence (stage-3 restructuring)."""

import sys

import pear_sim.evidence.numerical_evidence as _moved

sys.modules[__name__] = _moved
