# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Topology library — declarative per-topology data for selection."""

from pear_sim.topology.library import (
    TOPOLOGY_ENTRY_SCHEMA,
    EntryModeSequence,
    KernelSupport,
    PowerRange,
    Prerequisites,
    TopologyEntry,
    TopologyLibraryError,
    TopologyTemplate,
    VoltageRange,
    load_topology_entry,
    load_topology_library,
    verify_entry_sequence,
)

__all__ = [
    "TOPOLOGY_ENTRY_SCHEMA",
    "EntryModeSequence",
    "KernelSupport",
    "PowerRange",
    "Prerequisites",
    "TopologyEntry",
    "TopologyLibraryError",
    "TopologyTemplate",
    "VoltageRange",
    "load_topology_entry",
    "load_topology_library",
    "verify_entry_sequence",
]
