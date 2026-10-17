from pathlib import Path

import numpy as np

from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator


EXAMPLE = Path(__file__).parents[1] / "examples" / "boost.yaml"


def test_boost_tracks_ideal_conversion_ratio_and_commutates():
    result = HybridSimulator(load_model(EXAMPLE)).simulate()
    tail = result.signals["v(out)"][len(result.time) // 2 :]
    inductor = result.signals["i(l1)"][len(result.time) // 2 :]
    diode = result.signals["i(d1)"]
    assert abs(float(np.mean(tail)) - 48.0) < 0.08
    assert float(np.min(inductor)) > 0.0
    assert float(np.min(diode)) > -1e-8
    assert result.stats["scheduled_events"] >= 70
    assert result.stats["compiled_topologies"] == 2
