# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Compatibility alias; this module moved to pear_sim.reporting.render (stage-3 restructuring)."""

import sys

import pear_sim.reporting.render as _moved

sys.modules[__name__] = _moved
