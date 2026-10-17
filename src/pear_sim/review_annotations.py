"""Compatibility alias; this module moved to pear_sim.review.review_annotations (stage-3 restructuring)."""

import sys

import pear_sim.review.review_annotations as _moved

sys.modules[__name__] = _moved
