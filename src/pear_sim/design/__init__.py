"""Design-specification layer: the requirements side of a job.

`DesignSpec` is the typed input for topology screening and honest
finalist comparison: hard constraints reject, soft criteria rank, and
every value carries a named human in its provenance.
"""

from pear_sim.design.errors import DesignSpecError, IncompleteDesignSpecError
from pear_sim.design.spec import (
    DESIGN_SPEC_SCHEMA,
    DesignSpec,
    FieldSpec,
    FIELDS,
    OpenQuestion,
    ProvenancedValue,
)

__all__ = [
    "DESIGN_SPEC_SCHEMA",
    "DesignSpec",
    "DesignSpecError",
    "FIELDS",
    "FieldSpec",
    "IncompleteDesignSpecError",
    "OpenQuestion",
    "ProvenancedValue",
]
