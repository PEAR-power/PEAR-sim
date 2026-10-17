"""Compatibility alias; this module moved to pear_sim.evidence.platform_evidence (stage-3 restructuring)."""

import sys

import pear_sim.evidence.platform_evidence as _moved

sys.modules[__name__] = _moved
