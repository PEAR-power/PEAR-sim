"""Compatibility alias; this module moved to pear_sim.spec.canonical_topologies (stage-3 restructuring)."""

import sys

import pear_sim.spec.canonical_topologies as _moved

sys.modules[__name__] = _moved
