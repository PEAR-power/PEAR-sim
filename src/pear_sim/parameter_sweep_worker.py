# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Compatibility alias; this module moved to pear_sim.evidence.parameter_sweep_worker (stage-3 restructuring)."""

import sys

import pear_sim.evidence.parameter_sweep_worker as _moved

sys.modules[__name__] = _moved
