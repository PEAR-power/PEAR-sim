"""Compatibility alias; this module moved to pear_sim.evidence.benchmark_harness (stage-3 restructuring)."""

import sys

import pear_sim.evidence.benchmark_harness as _moved

sys.modules[__name__] = _moved
