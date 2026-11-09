# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
class PearSimError(Exception):
    """Base class for actionable user-facing errors.

    Every subclass declares a stable machine-readable ``code`` so agent
    clients can branch on failures without parsing message text.  Codes are
    part of the public contract: changing one is a breaking change.
    """

    code = "pear_sim.error"


class ModelError(PearSimError):
    """The input model is invalid."""

    code = "model.invalid"


class UnitError(ModelError):
    """A value has an invalid or incompatible unit."""

    code = "model.units"


class TopologyError(PearSimError):
    """A circuit topology cannot be solved consistently."""

    code = "model.topology"


class SimulationError(PearSimError):
    """The hybrid simulation failed."""

    code = "simulation.failed"
