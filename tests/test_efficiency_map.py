# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Efficiency across an operating range — and what makes the figure honest.

Efficiency is a claim about losses, so a model that declares none cannot
report one: the ideal-switch kernel would say 100 %. And a converter whose
devices are only PARTLY instrumented reports a flattering figure, which this
suite measures rather than trusting: the same sweep with one card removed
comes out visibly better, and the artifact has to say why.
"""

import json
from pathlib import Path

import pytest

from pear_sim.evidence.efficiency_map import EfficiencyMapError, build_efficiency_map
from pear_sim.evidence.parameter_sweep import run_parameter_sweep

ROOT = Path(__file__).parents[1]
EXAMPLES = ROOT / "examples"
TRACTION = EXAMPLES / "traction_inverter_liquid_cooled.yaml"


def _request(tmp_path: Path, parameter: str, values: list[str]) -> Path:
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "schema": "pear-sim/parameter-sweep-request/v0.1",
                "parameter": parameter,
                "values": values,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def traction_map(tmp_path_factory):
    directory = tmp_path_factory.mktemp("efficiency")
    request = _request(directory, "rload", ["0.5 ohm", "1.0 ohm"])
    artifacts = run_parameter_sweep(TRACTION, request, directory / "sweep")
    return build_efficiency_map(artifacts.output_dir)


def test_efficiency_is_reported_per_point_with_its_qualifiers(traction_map) -> None:
    assert traction_map["schema"] == "pear-sim/efficiency-map/v0.1"
    assert len(traction_map["points"]) == 2
    for row in traction_map["points"]:
        # A SiC leg of this class lands in the high nineties; anything at or
        # above unity would mean the loss model was not actually applied.
        assert 0.95 < row["efficiency"] < 1.0
        assert row["loss_power_w"] > 0.0
        assert row["output_power_w"] == pytest.approx(
            row["input_power_w"] - row["loss_power_w"], rel=1e-12
        )
        assert row["qualifiers"]["hottest_junction_c"] > 0.0
    # Light load is more efficient here: conduction loss falls faster than the
    # switching loss it trades against.
    light = max(traction_map["points"], key=lambda row: row["value_si"])
    heavy = min(traction_map["points"], key=lambda row: row["value_si"])
    assert light["efficiency"] > heavy["efficiency"]


def test_a_fully_instrumented_converter_says_so(traction_map) -> None:
    assert traction_map["uninstrumented_switching_devices"] == []
    assert (
        traction_map["efficiency_completeness"]
        == "every switching device carries a loss card"
    )


def test_partial_instrumentation_is_flagged_and_flatters_the_result(
    tmp_path: Path, traction_map
) -> None:
    """Removing one device's card must raise efficiency AND be declared."""

    body = TRACTION.read_text(encoding="utf-8")
    stripped = body.replace(
        """    dch:
      card: sic_diode_1200v_400a
      commutation_voltage: v(vdcp)
      cooling: {tim: tim_grease, cold_plate: plate_wg50, coolant_flow_l_per_min: 8}
""",
        "",
    )
    assert stripped != body, "the example no longer instruments dch"
    model = tmp_path / "partial.yaml"
    model.write_text(stripped, encoding="utf-8")

    request = _request(tmp_path, "rload", ["0.5 ohm", "1.0 ohm"])
    artifacts = run_parameter_sweep(model, request, tmp_path / "sweep")
    partial = build_efficiency_map(artifacts.output_dir)

    assert partial["uninstrumented_switching_devices"] == ["dch"]
    assert partial["efficiency_completeness"].startswith("OPTIMISTIC")

    # The omission is not cosmetic: the number itself improves.
    complete_worst = traction_map["worst"]["efficiency"]
    assert partial["worst"]["efficiency"] > complete_worst


def test_a_model_without_a_loss_model_cannot_report_efficiency(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path, "load", ["10 ohm", "20 ohm"])
    artifacts = run_parameter_sweep(
        EXAMPLES / "buck.yaml", request, tmp_path / "sweep"
    )
    with pytest.raises(EfficiencyMapError, match="claim about losses"):
        build_efficiency_map(artifacts.output_dir)


def test_declared_parasitics_are_added_to_the_loss_total(traction_map, tmp_path: Path) -> None:
    request = _request(tmp_path, "rload", ["1.0 ohm"])
    artifacts = run_parameter_sweep(TRACTION, request, tmp_path / "sweep")
    plain = build_efficiency_map(artifacts.output_dir)
    # The load resistors are the OUTPUT here, so naming one as parasitic must
    # move power out of the output and into the loss column.
    with_parasitic = build_efficiency_map(
        artifacts.output_dir, parasitic_components=("ra",)
    )
    assert with_parasitic["points"][0]["parasitic_loss_w"] > 0.0
    assert with_parasitic["points"][0]["efficiency"] < plain["points"][0]["efficiency"]


def test_unknown_parasitic_component_fails_closed(tmp_path: Path) -> None:
    request = _request(tmp_path, "rload", ["1.0 ohm"])
    artifacts = run_parameter_sweep(TRACTION, request, tmp_path / "sweep")
    with pytest.raises(EfficiencyMapError, match="not in"):
        build_efficiency_map(
            artifacts.output_dir, parasitic_components=("no_such_component",)
        )
