import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from pear_sim.errors import UnitError
from pear_sim.errors import ModelError
from pear_sim.model import load_model
from pear_sim.units import parse_quantity


ROOT = Path(__file__).parents[1]


def test_si_prefixes_and_unicode_units():
    assert math.isclose(parse_quantity("470 µH", "inductance").value, 470e-6)
    assert math.isclose(parse_quantity("2.5 kΩ", "resistance").value, 2500.0)
    assert math.isclose(parse_quantity("2 MΩ", "resistance").value, 2e6)
    assert math.isclose(parse_quantity("2 mΩ", "resistance").value, 2e-3)
    assert math.isclose(parse_quantity("40 %", "dimensionless").value, 0.4)


def test_dimension_mismatch_is_rejected():
    with pytest.raises(UnitError):
        parse_quantity("1 A", "voltage")


@pytest.mark.parametrize("value", ["1e999 V", "1e308 kV", "-1e308 kV"])
def test_nonfinite_or_prefix_overflow_is_rejected(value: str):
    with pytest.raises(UnitError, match="finite"):
        parse_quantity(value, "voltage")


def test_public_model_schema_matches_runtime_text_and_output_uniqueness_limits():
    schema = json.loads(
        (ROOT / "schema" / "model.schema.json").read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)
    payload = {
        "schema": "pear-sim/v0.1",
        "name": "bounded",
        "circuit": {
            "ground": "gnd",
            "components": [
                {
                    "id": "c1",
                    "type": "capacitor",
                    "p": "out",
                    "n": "gnd",
                    "value": "1 uF",
                }
            ],
        },
        "analysis": {"transient": {}},
        "outputs": ["v(out)"],
    }
    assert validator.is_valid(payload)

    duplicate_outputs = {**payload, "outputs": ["v(out)", "v(out)"]}
    whitespace_output = {**payload, "outputs": [" v(out) "]}
    oversized_name = {**payload, "name": "x" * 257}
    trailing_newline_parameter = {
        **payload,
        "parameters": {"load\n": "10 ohm"},
    }
    tiny_rtol = {
        **payload,
        "analysis": {"transient": {"rtol": 1e-30}},
    }
    quoted_rtol = {
        **payload,
        "analysis": {"transient": {"rtol": "1.0e-8"}},
    }
    complementary = {
        **payload,
        "controls": {
            "leg": {
                "type": "complementary_pwm",
                "frequency": "20 kHz",
                "duty": "40 %",
                "dead_time": "500 ns",
            }
        },
    }
    complementary_without_dead_time = {
        **complementary,
        "controls": {
            "leg": {
                "type": "complementary_pwm",
                "frequency": "20 kHz",
                "duty": "40 %",
            }
        },
    }
    pwm_with_dead_time = {
        **payload,
        "controls": {
            "gate": {
                "type": "pwm",
                "frequency": "20 kHz",
                "duty": "40 %",
                "dead_time": "500 ns",
            }
        },
    }
    three_phase_spwm = {
        **payload,
        "controls": {
            "bridge": {
                "type": "three_phase_spwm",
                "carrier_frequency": "20 kHz",
                "fundamental_frequency": "1 kHz",
                "modulation_index": "80 %",
                "dead_time": "500 ns",
            }
        },
        "analysis": {
            "transient": {},
            "harmonics": {
                "start": "0 s",
                "stop": "5 ms",
                "fundamental_frequency": "1 kHz",
                "max_harmonic": 25,
                "signals": [
                    {"signal": "v(out)", "semantic": "phase_voltage_a"}
                ],
            },
        },
    }
    mixed_spwm = {
        **three_phase_spwm,
        "controls": {
            **three_phase_spwm["controls"],
            "gate": {"type": "pwm", "frequency": "20 kHz", "duty": "50 %"},
        },
    }
    invalid_harmonic_semantic = {
        **three_phase_spwm,
        "analysis": {
            **three_phase_spwm["analysis"],
            "harmonics": {
                **three_phase_spwm["analysis"]["harmonics"],
                "signals": [{"signal": "v(out)", "semantic": "voltage"}],
            },
        },
    }
    invalid_harmonic_family = {
        **three_phase_spwm,
        "analysis": {
            **three_phase_spwm["analysis"],
            "harmonics": {
                **three_phase_spwm["analysis"]["harmonics"],
                "signals": [
                    {"signal": "v(out)", "semantic": "phase_current_a"}
                ],
            },
        },
    }
    assert validator.is_valid(complementary)
    assert validator.is_valid(three_phase_spwm)
    assert not validator.is_valid(duplicate_outputs)
    assert not validator.is_valid(whitespace_output)
    assert not validator.is_valid(oversized_name)
    assert not validator.is_valid(trailing_newline_parameter)
    assert not validator.is_valid(tiny_rtol)
    assert not validator.is_valid(quoted_rtol)
    assert not validator.is_valid(complementary_without_dead_time)
    assert not validator.is_valid(pwm_with_dead_time)
    assert not validator.is_valid(mixed_spwm)
    assert not validator.is_valid(invalid_harmonic_semantic)
    assert not validator.is_valid(invalid_harmonic_family)


def test_huge_numeric_tolerance_and_quantity_fail_as_model_errors(tmp_path):
    huge_integer = "9" * 4000
    tolerance_path = tmp_path / "huge-tolerance.yaml"
    tolerance_path.write_text(
        """
schema: pear-sim/v0.1
name: huge_tolerance
circuit:
  components: [{id: c1, type: capacitor, p: out, n: gnd, value: 1 uF}]
analysis:
  transient: {stop: 1 ms, output_step: 1 us, rtol: HUGE}
outputs: [v(out)]
""".replace("HUGE", huge_integer),
        encoding="utf-8",
    )
    with pytest.raises(ModelError, match="finite float contract"):
        load_model(tolerance_path)

    quantity_path = tmp_path / "huge-quantity.yaml"
    quantity_path.write_text(
        tolerance_path.read_text(encoding="utf-8")
        .replace("rtol: " + huge_integer, "rtol: 1.0e-8")
        .replace("value: 1 uF", "value: " + huge_integer),
        encoding="utf-8",
    )
    with pytest.raises(UnitError, match="finite float contract"):
        load_model(quantity_path)


def test_unknown_model_fields_are_rejected(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
schema: pear-sim/v0.1
name: typo
circut: {}
circuit:
  components: [{id: c1, type: capacitor, p: out, n: gnd, value: 1 uF}]
analysis:
  transient: {stop: 1 ms, output_step: 1 us}
outputs: [v(out)]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ModelError, match="circut"):
        load_model(path)


def test_celsius_normalizes_with_the_offset_not_only_a_factor():
    """The one dimension where a scale factor alone is a silent 273.15 error.

    A plausibility check cannot catch it: 25 K and 25 degC are both numbers
    that look like temperatures, and every downstream formula would keep
    working while reporting nonsense.
    """

    assert parse_quantity("25 degC", "temperature").value == pytest.approx(298.15)
    assert parse_quantity("0 degC", "temperature").value == pytest.approx(273.15)
    assert parse_quantity("298.15 K", "temperature").value == pytest.approx(298.15)
    # Round trip: the same physical temperature in either unit.
    assert parse_quantity("125 degC", "temperature").value == pytest.approx(
        parse_quantity("398.15 K", "temperature").value
    )


def test_temperature_below_absolute_zero_is_refused():
    with pytest.raises(UnitError, match="absolute zero"):
        parse_quantity("-300 degC", "temperature")
    with pytest.raises(UnitError, match="absolute zero"):
        parse_quantity("-1 K", "temperature")
