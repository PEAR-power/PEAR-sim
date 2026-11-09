# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Coupled inductors / two-winding transformer: loader, kernel, energy.

The reference is an independent linear solution (matrix exponential of the
exact 2x2 system), not the kernel's own integrator.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import expm

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.model import load_model

BODY = """
schema: pear-sim/v0.1
name: transformer_rl

circuit:
  ground: gnd
  components:
    - {{id: vin, type: voltage_source, p: in, n: gnd, value: 10 V}}
    - {{id: r1, type: resistor, p: in, n: p1, value: 1 ohm}}
    - {{id: lp, type: inductor, p: p1, n: gnd, value: 1 mH, initial_current: 0 A}}
    - {{id: ls, type: inductor, p: s1, n: gnd, value: 4 mH, initial_current: 0 A}}
    - {{id: rload, type: resistor, p: s1, n: gnd, value: 10 ohm}}
  couplings:
    - {coupling_line}

analysis:
  transient:
    start: 0 s
    stop: 10 ms
    output_step: 5 us
    solver: dopri5
    rtol: 1.0e-10
    atol: 1.0e-12

outputs:
  - i(lp)
  - i(ls)
  - v(s1)
"""


def _load(tmp_path: Path, coupling_line: str = "{inductors: [lp, ls], coupling: 0.95}"):
    path = tmp_path / "model.yaml"
    path.write_text(BODY.format(coupling_line=coupling_line), encoding="utf-8")
    return load_model(path)


def test_loader_parses_coupling_k_to_mutual(tmp_path: Path) -> None:
    model = _load(tmp_path)
    (coupling,) = model.couplings
    assert (coupling.first, coupling.second) == ("lp", "ls")
    assert coupling.mutual == pytest.approx(0.95 * np.sqrt(1e-3 * 4e-3))


def test_loader_rejects_perfect_coupling(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="strictly between 0 and 1"):
        _load(tmp_path, "{inductors: [lp, ls], coupling: 1.0}")
    with pytest.raises(ModelError, match="singular"):
        _load(tmp_path, "{inductors: [lp, ls], mutual: 2 mH}")


def test_loader_rejects_sat_inductor_in_coupling(tmp_path: Path) -> None:
    path = tmp_path / "model.yaml"
    body = BODY.format(coupling_line="{inductors: [lp, ls], coupling: 0.95}")
    body = body.replace(
        "    - {id: ls, type: inductor, p: s1, n: gnd, value: 4 mH, initial_current: 0 A}",
        """    - id: ls
      type: sat_inductor
      p: s1
      n: gnd
      value: 4 mH
      l_sat: 1 mH
      i_sat: 2 A""",
    )
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ModelError, match="not a plain inductor"):
        load_model(path)


def test_loader_rejects_the_same_pair_declared_twice(tmp_path: Path) -> None:
    """A winding may join several couplings — that is how a three-winding
    transformer is declared — but one mutual per pair, in one place."""

    path = tmp_path / "model.yaml"
    body = BODY.format(
        coupling_line="{inductors: [lp, ls], coupling: 0.95}\n"
        "    - {inductors: [ls, lp], coupling: 0.5}"
    )
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ModelError, match="already"):
        load_model(path)


def _reference(t: np.ndarray) -> np.ndarray:
    l_p, l_s, k = 1e-3, 4e-3, 0.95
    mutual = k * np.sqrt(l_p * l_s)
    inductance = np.array([[l_p, mutual], [mutual, l_s]])
    voltage_map = np.array([[-1.0, 0.0], [0.0, -10.0]])
    voltage_offset = np.array([10.0, 0.0])
    gamma = np.linalg.inv(inductance)
    a = gamma @ voltage_map
    b = gamma @ voltage_offset
    x_inf = -np.linalg.solve(a, b)
    return np.array([x_inf + expm(a * tt) @ (-x_inf) for tt in t])


def test_transformer_matches_matrix_exponential_reference(tmp_path: Path) -> None:
    result = HybridSimulator(_load(tmp_path)).simulate()
    t = result.time
    reference = _reference(t)
    assert np.max(np.abs(result.signals["i(lp)"] - reference[:, 0])) < 1e-7
    assert np.max(np.abs(result.signals["i(ls)"] - reference[:, 1])) < 1e-8
    # Turns-ratio kick on the secondary at t=0+ (n = sqrt(L2/L1) = 2, k = 0.95).
    assert float(np.max(np.abs(result.signals["v(s1)"]))) > 10.0
    audit = result.physics_audit["full"]
    assert audit["balance_status"] == "within_numerical_tolerance"


def test_coupled_energy_includes_the_mutual_term(tmp_path: Path) -> None:
    result = HybridSimulator(_load(tmp_path)).simulate()
    audit = result.physics_audit["full"]
    i_p = float(result.signals["i(lp)"][-1])
    i_s = float(result.signals["i(ls)"][-1])
    l_p, l_s = 1e-3, 4e-3
    mutual = 0.95 * np.sqrt(l_p * l_s)
    expected = 0.5 * l_p * i_p**2 + 0.5 * l_s * i_s**2 + mutual * i_p * i_s
    assert audit["storage_energy_end_j"] == pytest.approx(expected, rel=1e-9)


def test_transformer_with_diode_rectifier_commutates(tmp_path: Path) -> None:
    """Coupled block composes with the diode active set (sine in, diode out)."""

    path = tmp_path / "model.yaml"
    path.write_text(
        """
schema: pear-sim/v0.1
name: transformer_rectifier

circuit:
  ground: gnd
  components:
    - id: vin
      type: voltage_source
      p: in
      n: gnd
      waveform: {type: sine, amplitude: 10 V, frequency: 50 Hz}
    - {id: r1, type: resistor, p: in, n: p1, value: 1 ohm}
    - {id: lp, type: inductor, p: p1, n: gnd, value: 100 mH, initial_current: 0 A}
    - {id: ls, type: inductor, p: s1, n: gnd, value: 400 mH, initial_current: 0 A}
    - {id: d1, type: diode, p: s1, n: rect, initial: off}
    - {id: rload, type: resistor, p: rect, n: gnd, value: 100 ohm}
    # v0.1 limitation: a coupled winding needs a DC path in every active-set
    # configuration (an open secondary sees the reflected voltage, which the
    # zero-current reduction cannot express). The bleeder models real leakage.
    - {id: rbleed, type: resistor, p: s1, n: gnd, value: 10 kohm}
  couplings:
    - {inductors: [lp, ls], coupling: 0.98}

analysis:
  transient:
    start: 0 s
    stop: 60 ms
    output_step: 20 us
    solver: dopri5
    rtol: 1.0e-9
    atol: 1.0e-11

outputs:
  - v(rect)
  - i(d1)
""".strip()
        + "\n",
        encoding="utf-8",
    )
    result = HybridSimulator(load_model(path)).simulate()
    commutations = [e for e in result.events if e.kind == "natural_commutation"]
    assert len(commutations) >= 4  # one on/off pair per mains cycle at least
    assert float(np.min(result.signals["i(d1)"])) > -1e-9
    assert result.physics_audit["full"]["balance_status"] == "within_numerical_tolerance"


# ---------------------------------------------------------------------------
# Negative mutuals
# ---------------------------------------------------------------------------


def _star_model(tmp_path, mutual: str, inductance: str = "2 mH"):
    """Three coupled windings meeting at a floating star point."""

    import yaml

    data = {
        "schema": "pear-sim/v0.1",
        "name": "coupled_star",
        "circuit": {
            "ground": "gnd",
            "components": [
                {"id": "src", "type": "voltage_source", "p": "ta", "n": "gnd",
                 "value": "10 V"},
                {"id": "rb", "type": "resistor", "p": "tb", "n": "gnd",
                 "value": "1 ohm"},
                {"id": "rc", "type": "resistor", "p": "tc", "n": "gnd",
                 "value": "1 ohm"},
                {"id": "la", "type": "inductor", "p": "ta", "n": "star",
                 "value": inductance},
                {"id": "lb", "type": "inductor", "p": "tb", "n": "star",
                 "value": inductance},
                {"id": "lc", "type": "inductor", "p": "tc", "n": "star",
                 "value": inductance},
            ],
            "couplings": [
                {"inductors": ["la", "lb"], "mutual": mutual},
                {"inductors": ["la", "lc"], "mutual": mutual},
                {"inductors": ["lb", "lc"], "mutual": mutual},
            ],
        },
        "analysis": {
            "transient": {"start": "0 s", "stop": "1 ms", "output_step": "10 us",
                          "solver": "dopri5", "rtol": 1.0e-9, "atol": 1.0e-11}
        },
        "outputs": ["i(la)", "i(lb)", "i(lc)"],
    }
    target = tmp_path / "star.yaml"
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def test_a_negative_mutual_is_physics_not_an_error(tmp_path):
    """Three-phase windings couple NEGATIVELY and cannot be re-oriented.

    The sign of a mutual states winding orientation. In a symmetric star the
    phase-to-phase couplings are negative — flux from one phase opposes the
    other two — and flipping one inductor to fix a pair breaks the other
    pair. Refusing the sign would put every rotating machine out of reach.
    """

    from pear_sim.spec.model import load_model

    model = load_model(_star_model(tmp_path, "-0.5 mH"))
    assert [coupling.mutual for coupling in model.couplings] == [-5e-4] * 3


def test_negative_mutuals_still_face_the_positive_definite_gate(tmp_path):
    """The sign is free; the energy is not.

    With three windings at M = -0.9*L the assembled matrix stops being
    positive definite (its all-ones eigenvector carries L + 2M < 0), so some
    current set would store negative energy. That is what must be refused —
    not the sign.
    """

    import pytest

    from pear_sim.errors import ModelError
    from pear_sim.spec.model import load_model

    with pytest.raises(ModelError, match="not positive definite"):
        load_model(_star_model(tmp_path, "-1.8 mH"))


def test_a_zero_mutual_is_refused_as_saying_nothing(tmp_path):
    import pytest

    from pear_sim.errors import ModelError
    from pear_sim.spec.model import load_model

    with pytest.raises(ModelError, match="zero mutual says nothing"):
        load_model(_star_model(tmp_path, "0 mH"))


def test_negative_coupling_coefficient_is_accepted_by_magnitude(tmp_path):
    import yaml

    from pear_sim.spec.model import load_model

    path = _star_model(tmp_path, "-0.5 mH")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    for entry in data["circuit"]["couplings"]:
        entry.pop("mutual")
        entry["coupling"] = -0.25
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    model = load_model(path)
    assert all(coupling.mutual < 0.0 for coupling in model.couplings)


def test_star_invariant_holds_with_negative_mutuals(tmp_path):
    """The floating star still sums to zero — the reduction is unaffected."""

    import numpy as np

    from pear_sim.core.solver import HybridSimulator
    from pear_sim.spec.model import load_model

    result = HybridSimulator(load_model(_star_model(tmp_path, "-0.5 mH"))).simulate()
    total = (
        result.signals["i(la)"] + result.signals["i(lb)"] + result.signals["i(lc)"]
    )
    assert float(np.max(np.abs(total))) < 1e-9
