from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from pear_sim.analytic_checks import (
    ANALYTIC_CHECKS_SCHEMA,
    SELF_CHECK_KIND,
    BuckCcmTolerances,
    check_ideal_buck_ccm,
    run_analytic_checks,
)
from pear_sim.model import Model, load_model
from pear_sim.solver import HybridSimulator


ROOT = Path(__file__).parents[1]
BUCK = ROOT / "examples" / "buck.yaml"
BOOST = ROOT / "examples" / "boost.yaml"


@pytest.fixture(scope="module")
def buck_case() -> tuple[Model, dict[str, dict[str, float | str]]]:
    model = load_model(BUCK)
    metrics = HybridSimulator(model).simulate().metrics()
    return model, metrics


def _checks_by_id(result: dict[str, object]) -> dict[str, dict[str, object]]:
    return {check["id"]: check for check in result["checks"]}


def test_buck_ccm_self_check_is_versioned_deterministic_and_not_validation(
    buck_case: tuple[Model, dict[str, dict[str, float | str]]],
):
    model, metrics = buck_case

    first = run_analytic_checks(model, metrics)
    second = check_ideal_buck_ccm(model, metrics)

    assert first == second
    assert first["schema"] == ANALYTIC_CHECKS_SCHEMA
    assert first["kind"] == SELF_CHECK_KIND == "verification-self-check"
    assert first["claim"]["classification"] == "verification"
    assert first["claim"]["validation_claim"] is False
    assert first["recognition"]["status"] == "matched"
    assert first["overall_status"] == "pass"
    assert [check["status"] for check in first["checks"]] == ["pass"] * 4
    assert json.dumps(first, allow_nan=False, sort_keys=True) == json.dumps(
        second, allow_nan=False, sort_keys=True
    )


def test_buck_expected_relations_and_structured_deltas(
    buck_case: tuple[Model, dict[str, dict[str, float | str]]],
):
    model, metrics = buck_case
    result = run_analytic_checks(model, metrics)
    checks = _checks_by_id(result)

    expected = {
        "buck.output_voltage": 40.0,
        "buck.inductor_current_ripple": 1.2,
        "buck.output_voltage_ripple": 0.075,
        "buck.diode_average_current": 2.4,
    }
    for check_id, expected_value in expected.items():
        check = checks[check_id]
        assert math.isclose(check["expected"]["value"], expected_value, rel_tol=0.0, abs_tol=1e-12)
        assert check["measured"]["value"] is not None
        assert check["tolerance"]["relative"] >= 0.0
        assert check["tolerance"]["absolute"] >= 0.0
        assert check["tolerance"]["effective_absolute"] >= check["tolerance"]["absolute"]
        assert check["delta"]["absolute"] == abs(check["delta"]["signed"])
        assert check["status"] == "pass"

    assert result["recognition"]["components"] == {
        "source": "vin",
        "switch": "q1",
        "diode": "d1",
        "inductor": "l1",
        "capacitor": "c1",
        "load": "load",
        "pwm": "pwm",
    }
    assert result["preconditions"][1]["id"] == "buck.ccm_regime"
    assert result["preconditions"][1]["status"] == "pass"


def test_non_buck_topology_is_not_applicable_and_is_not_guessed():
    model = load_model(BOOST)
    metrics = HybridSimulator(model).simulate().metrics()

    result = run_analytic_checks(model, metrics)

    assert result["recognition"]["status"] == "not_applicable"
    assert result["overall_status"] == "not_applicable"
    assert all(check["status"] == "not_applicable" for check in result["checks"])
    assert all(check["expected"] is None for check in result["checks"])


def test_extra_power_component_makes_buck_match_ambiguous(tmp_path: Path):
    model_path = tmp_path / "ambiguous-buck.yaml"
    source = BUCK.read_text(encoding="utf-8")
    source = source.replace(
        "\ncontrols:",
        "\n    - {id: extra_load, type: resistor, p: out, n: gnd, value: 100 ohm}\n\ncontrols:",
    )
    model_path.write_text(source, encoding="utf-8")
    model = load_model(model_path)

    result = run_analytic_checks(model, {})

    assert result["overall_status"] == "not_applicable"
    assert "exactly one" in result["recognition"]["reason"]


def test_missing_ccm_metric_is_not_run(
    buck_case: tuple[Model, dict[str, dict[str, float | str]]],
):
    model, metrics = buck_case
    incomplete = deepcopy(metrics)
    del incomplete["i(l1)"]

    result = run_analytic_checks(model, incomplete)

    assert result["preconditions"][1]["status"] == "not_run"
    assert result["overall_status"] == "not_run"
    assert all(check["status"] == "not_run" for check in result["checks"])


def test_non_ccm_measurement_is_not_applicable(
    buck_case: tuple[Model, dict[str, dict[str, float | str]]],
):
    model, metrics = buck_case
    dcm_metrics = deepcopy(metrics)
    dcm_metrics["i(l1)"]["min_last_half"] = 0.0

    result = run_analytic_checks(model, dcm_metrics)

    assert result["preconditions"][1]["status"] == "not_applicable"
    assert result["overall_status"] == "not_applicable"
    assert all(check["status"] == "not_applicable" for check in result["checks"])


def test_missing_individual_signal_keeps_other_checks_but_overall_is_not_run(
    buck_case: tuple[Model, dict[str, dict[str, float | str]]],
):
    model, metrics = buck_case
    incomplete = deepcopy(metrics)
    del incomplete["i(d1)"]

    result = run_analytic_checks(model, incomplete)
    checks = _checks_by_id(result)

    assert result["overall_status"] == "not_run"
    assert checks["buck.diode_average_current"]["status"] == "not_run"
    assert checks["buck.diode_average_current"]["measured"] is None
    assert checks["buck.diode_average_current"]["expected"]["value"] == 2.4
    assert checks["buck.output_voltage"]["status"] == "pass"


def test_out_of_tolerance_measurement_fails_without_changing_tolerance(
    buck_case: tuple[Model, dict[str, dict[str, float | str]]],
):
    model, metrics = buck_case
    bad_metrics = deepcopy(metrics)
    bad_metrics["v(out)"]["mean_last_half"] = 30.0

    result = run_analytic_checks(model, bad_metrics)
    check = _checks_by_id(result)["buck.output_voltage"]

    assert result["overall_status"] == "fail"
    assert check["status"] == "fail"
    assert check["expected"]["value"] == 40.0
    assert check["measured"]["value"] == 30.0
    assert check["tolerance"]["relative"] == BuckCcmTolerances().output_voltage_relative
    assert check["delta"]["signed"] == -10.0


def test_short_measurement_window_is_not_run(
    buck_case: tuple[Model, dict[str, dict[str, float | str]]],
):
    model, metrics = buck_case
    tolerances = BuckCcmTolerances(minimum_measurement_cycles=100.0)

    result = run_analytic_checks(model, metrics, tolerances)

    assert result["preconditions"][0]["status"] == "not_run"
    assert result["overall_status"] == "not_run"
    assert all(check["status"] == "not_run" for check in result["checks"])


def test_invalid_tolerances_are_rejected():
    with pytest.raises(ValueError, match="finite and non-negative"):
        BuckCcmTolerances(output_voltage_relative=-1.0)


def _write_buck(tmp_path: Path, replacements: dict[str, str]) -> Path:
    text = (Path(__file__).parents[1] / "examples" / "buck.yaml").read_text(encoding="utf-8")
    for old, new in replacements.items():
        assert old in text, old
        text = text.replace(old, new)
    path = tmp_path / "buck_variant.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_run_still_ringing_is_not_compared_instead_of_failed(tmp_path: Path) -> None:
    """A self-check that fires on a correct circuit teaches people to ignore it.

    Started off its orbit, the canonical buck's LC filter (Q ~ 3) still rings in
    the last half of a 2 ms run, so its peak-to-peak "ripple" there is start-up
    oscillation, not switching ripple. The physics is right; the measurement is
    not comparable. The check must say so, and still pass the settled run.
    """

    from pear_sim.run.io import write_result

    # Inductor current on its orbit, capacitor 2 V low: the filter rings with
    # tau = 2RC = 2 ms (still ~1.2 V in the last half) while the current never
    # leaves CCM, so the ripple comparison is reached — and must not run.
    cold = _write_buck(tmp_path, {"initial_voltage: 39.98969319 V": "initial_voltage: 38 V"})
    result = HybridSimulator(load_model(cold)).simulate()
    write_result(result, tmp_path / "cold")
    checks = json.loads((tmp_path / "cold" / "analytic-checks.json").read_text())
    assert checks["overall_status"] == "not_run"
    settled = next(p for p in checks["preconditions"] if p["id"] == "buck.periodic_steady_state")
    assert settled["status"] == "not_run" and "steady" in settled["reason"]

    warm = HybridSimulator(load_model(Path(__file__).parents[1] / "examples" / "buck.yaml")).simulate()
    write_result(warm, tmp_path / "warm")
    warm_checks = json.loads((tmp_path / "warm" / "analytic-checks.json").read_text())
    assert warm_checks["overall_status"] == "pass"


def test_a_run_settled_only_by_its_end_is_still_not_compared(tmp_path: Path) -> None:
    """Settled at the end is not settled across the window.

    From a cold start the canonical buck rings with tau = 2RC = 2 ms. At 20 ms
    the final period repeats to well under a millivolt, but the last-half window
    opens at 10 ms with ~0.27 V of ringing left — far above the 75 mV ripple the
    formula predicts. Comparing the endpoints alone called this run settled.
    """

    from pear_sim.run.io import write_result

    cold = _write_buck(tmp_path, {
        "initial_current: 3.39970055 A": "initial_current: 0 A",
        "initial_voltage: 39.98969319 V": "initial_voltage: 0 V",
        "stop: 2 ms": "stop: 20 ms",
    })
    write_result(HybridSimulator(load_model(cold)).simulate(), tmp_path / "late")
    checks = json.loads((tmp_path / "late" / "analytic-checks.json").read_text())
    ccm = next(p for p in checks["preconditions"] if p["id"] == "buck.ccm_regime")
    assert ccm["status"] == "pass"  # the comparison would otherwise be reached
    assert checks["overall_status"] == "not_run"
