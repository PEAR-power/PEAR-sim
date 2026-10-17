"""Compatibility alias; this module moved to pear_sim.review.review_errors (stage-3 restructuring)."""

import sys

import pear_sim.review.review_errors as _moved

sys.modules[__name__] = _moved
