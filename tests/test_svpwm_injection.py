"""Min-max (SVPWM-equivalent) reference injection for three_phase_spwm.

The injection is a pure common-mode shift: line-to-line quantities are
unchanged at the same modulation index, while the linear range extends to
2/sqrt(3). The three-phase modulation contract recomputes the injected law
independently and bit-exactly (exercised via the CLI bundle writer).
"""

import math
from pathlib import Path

import pytest
import yaml

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.spec.model import load_model

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "examples" / "inverter_three_phase_floating_neutral.yaml"


def _variant(tmp_path: Path, *, injection: str | None, m_percent: float, stop_ms: float = 2.0) -> Path:
    document = yaml.safe_load(EXAMPLE.read_text())
    document["name"] = "svpwm_variant"
    document["parameters"]["modulation_index"] = f"{m_percent} %"
    if injection is not None:
        document["controls"]["bridge"]["injection"] = injection
    document["analysis"]["transient"]["stop"] = f"{stop_ms} ms"
    document["analysis"].pop("harmonics", None)
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_injection_extends_the_linear_range(tmp_path: Path) -> None:
    with pytest.raises(ModelError, match="modulation_index"):
        load_model(_variant(tmp_path, injection=None, m_percent=105.0))
    load_model(_variant(tmp_path, injection="min_max", m_percent=105.0))
    with pytest.raises(ModelError, match="modulation_index"):
        load_model(
            _variant(tmp_path, injection="min_max", m_percent=100.0 * 2.0 / math.sqrt(3.0) + 0.5)
        )
    with pytest.raises(ModelError, match="injection"):
        load_model(_variant(tmp_path, injection="third_harmonic", m_percent=80.0))


def test_injected_duties_follow_the_min_max_law(tmp_path: Path) -> None:
    model = load_model(_variant(tmp_path, injection="min_max", m_percent=105.0, stop_ms=1.0))
    result = HybridSimulator(model).simulate()
    assert result.modulation_cycles, "SPWM run must retain modulation cycles"
    m = 1.05
    for cycle in result.modulation_cycles[:63]:
        angle = float(cycle["electrical_angle_rad"])
        raw = {
            "a": math.sin(angle),
            "b": math.sin(angle - 2.0 * math.pi / 3.0),
            "c": math.sin(angle + 2.0 * math.pi / 3.0),
        }
        offset = -0.5 * (max(raw.values()) + min(raw.values()))
        for phase in ("a", "b", "c"):
            expected_duty = 0.5 * (1.0 + m * (raw[phase] + offset))
            assert cycle["duties"][phase] == pytest.approx(expected_duty, abs=1e-15)
            assert 0.0 < cycle["duties"][phase] < 1.0


def test_line_line_fundamental_matches_theory_at_m_1_05(tmp_path: Path) -> None:
    document = yaml.safe_load(EXAMPLE.read_text())
    document["name"] = "svpwm_fundamental"
    document["parameters"]["modulation_index"] = "105 %"
    # Zero dead time: this test pins the MODULATION LAW alone. With the
    # example's 500 ns dead time the fundamental drops by ~1.4% — the classic
    # dead-time voltage loss, which is real physics, not a law error.
    document["parameters"]["dead_time"] = "0 s"
    document["controls"]["bridge"]["injection"] = "min_max"
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    result = HybridSimulator(load_model(path)).simulate()
    payload = result.harmonic_analysis
    record = next(
        signal
        for signal in payload["signals"]
        if signal["semantic"] == "line_voltage_ab"
    )
    theory = 1.05 * 200.0 * math.sqrt(3.0)  # m * (Vdc/2) * sqrt(3)
    assert record["fundamental"]["peak_amplitude"] == pytest.approx(theory, rel=5e-3)


def test_common_mode_injection_leaves_line_line_unchanged(tmp_path: Path) -> None:
    """Same m, with and without injection: v_ab fundamental must agree."""


    results = {}
    for injection in (None, "min_max"):
        document = yaml.safe_load(EXAMPLE.read_text())
        document["name"] = "svpwm_cmv"
        document["parameters"]["modulation_index"] = "80 %"
        if injection:
            document["controls"]["bridge"]["injection"] = injection
        path = tmp_path / f"model_{injection}.yaml"
        path.write_text(yaml.safe_dump(document), encoding="utf-8")
        result = HybridSimulator(load_model(path)).simulate()
        record = next(
            signal
            for signal in result.harmonic_analysis["signals"]
            if signal["semantic"] == "line_voltage_ab"
        )
        results[injection] = record["fundamental"]["peak_amplitude"]
    assert results["min_max"] == pytest.approx(results[None], rel=2e-3)


def test_bundle_writer_validates_injected_law_end_to_end(tmp_path: Path) -> None:
    """write_result runs the three-phase modulation contract, which recomputes the injected law."""

    from pear_sim.run.io import write_result

    model = load_model(_variant(tmp_path, injection="min_max", m_percent=105.0, stop_ms=1.0))
    result = HybridSimulator(model).simulate()
    destination = write_result(result, tmp_path / "bundle")
    import json

    observation = json.loads(
        (destination / "three-phase-modulation.json").read_text()
    )
    config = observation["blocks"][0]["config"]
    assert config["reference_injection"] == "min_max"
    assert config["linear_region_limit"] == pytest.approx(2.0 / math.sqrt(3.0))
