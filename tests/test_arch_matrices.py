from pathlib import Path

import numpy as np

from pear_sim.circuit import Circuit
from pear_sim.model import load_model


ROOT = Path(__file__).parents[1]


def test_buck_modes_match_published_hybrid_equations():
    """Check the circuit compiler against the ARCH buck state equations."""
    circuit = Circuit(load_model(ROOT / "examples" / "buck.yaml"))
    inductance = 1e-3
    capacitance = 100e-6
    resistance = 10.0
    vin = 100.0
    expected_a = np.array(
        [[0.0, -1.0 / inductance], [1.0 / capacitance, -1.0 / (resistance * capacitance)]]
    )
    on = circuit.compile({"pwm": True}, {"d1": False})
    off = circuit.compile({"pwm": False}, {"d1": True})
    np.testing.assert_allclose(on.a, expected_a, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(off.a, expected_a, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(on.c, [vin / inductance, 0.0], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(off.c, [0.0, 0.0], rtol=0.0, atol=1e-11)


def test_boost_modes_match_published_hybrid_equations():
    """Check the circuit compiler against the ARCH boost state equations."""
    circuit = Circuit(load_model(ROOT / "examples" / "boost.yaml"))
    inductance = 500e-6
    capacitance = 100e-6
    resistance = 24.0
    vin = 24.0
    on = circuit.compile({"pwm": True}, {"d1": False})
    off = circuit.compile({"pwm": False}, {"d1": True})
    expected_on = np.array([[0.0, 0.0], [0.0, -1.0 / (resistance * capacitance)]])
    expected_off = np.array(
        [[0.0, -1.0 / inductance], [1.0 / capacitance, -1.0 / (resistance * capacitance)]]
    )
    np.testing.assert_allclose(on.a, expected_on, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(off.a, expected_off, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(on.c, [vin / inductance, 0.0], rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(off.c, [vin / inductance, 0.0], rtol=0.0, atol=1e-9)
