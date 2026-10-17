from dataclasses import replace
import math
from pathlib import Path

from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator, SimulationResult


ROOT = Path(__file__).parents[1]


def _rc_result(tmp_path: Path) -> SimulationResult:
    model_path = tmp_path / "rc_time_metrics.yaml"
    model_path.write_text(
        """
schema: pear-sim/v0.1
name: rc_time_metrics
circuit:
  ground: gnd
  components:
    - {id: vin, type: voltage_source, p: in, n: gnd, value: 10 V}
    - {id: r1, type: resistor, p: in, n: out, value: 1 kohm}
    - {id: c1, type: capacitor, p: out, n: gnd, value: 1 uF, initial_voltage: 0 V}
analysis:
  transient:
    start: 0 s
    stop: 2.3 ms
    output_step: 0.9 ms
    solver: dopri5
    rtol: 1.0e-11
    atol: 1.0e-13
outputs: [v(out)]
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return HybridSimulator(load_model(model_path)).simulate()


def _rc_integral(time: float, voltage: float, tau: float) -> float:
    return voltage * (time + tau * math.exp(-time / tau))


def _rc_square_integral(time: float, voltage: float, tau: float) -> float:
    return (
        voltage
        * voltage
        * (time + 2.0 * tau * math.exp(-time / tau) - 0.5 * tau * math.exp(-2.0 * time / tau))
    )


def test_rc_time_weighted_mean_and_rms_match_analytic_integrals(tmp_path: Path) -> None:
    result = _rc_result(tmp_path)
    metric = result.metrics()["v(out)"]
    voltage = 10.0
    tau = 1e-3
    stop = 2.3e-3
    half = stop / 2.0

    expected_integral = _rc_integral(stop, voltage, tau) - _rc_integral(0.0, voltage, tau)
    expected_square = _rc_square_integral(stop, voltage, tau) - _rc_square_integral(
        0.0, voltage, tau
    )
    expected_tail_integral = _rc_integral(stop, voltage, tau) - _rc_integral(half, voltage, tau)
    expected_tail_square = _rc_square_integral(stop, voltage, tau) - _rc_square_integral(
        half, voltage, tau
    )

    assert math.isclose(metric["mean"], expected_integral / stop, rel_tol=0, abs_tol=2e-9)
    assert math.isclose(metric["rms"], math.sqrt(expected_square / stop), rel_tol=0, abs_tol=2e-9)
    assert math.isclose(
        metric["mean_last_half"],
        expected_tail_integral / (stop - half),
        rel_tol=0,
        abs_tol=2e-9,
    )
    assert math.isclose(
        metric["rms_last_half"],
        math.sqrt(expected_tail_square / (stop - half)),
        rel_tol=0,
        abs_tol=2e-9,
    )
    assert metric["window_start_s"] == half
    expected_sample_times = [0.0, 0.0009, 0.0018, 0.0023]
    assert len(result.time) == len(expected_sample_times)
    assert all(
        math.isclose(observed, expected, abs_tol=1e-18)
        for observed, expected in zip(result.time, expected_sample_times, strict=True)
    )
    assert abs(metric["sample_mean"] - metric["mean"]) > 0.2
    assert abs(metric["sample_mean_last_half"] - metric["mean_last_half"]) > 0.2


def test_rc_physics_audit_matches_analytic_storage_change(tmp_path: Path) -> None:
    result = _rc_result(tmp_path)
    audit = result.physics_audit["full"]
    stop = 2.3e-3
    tau = 1e-3
    final_voltage = 10.0 * (1.0 - math.exp(-stop / tau))
    expected_storage_change = 0.5 * 1e-6 * final_voltage * final_voltage

    assert math.isclose(
        audit["delta_storage_energy_j"],
        expected_storage_change,
        rel_tol=0,
        abs_tol=2e-14,
    )
    assert math.isclose(
        audit["component_absorbed_energy_j"]["c1"],
        expected_storage_change,
        rel_tol=0,
        abs_tol=2e-13,
    )
    assert audit["component_absorbed_energy_j"]["r1"] > 0.0
    assert audit["component_absorbed_energy_j"]["vin"] < 0.0
    assert audit["source_delivered_energy_j"] == -audit["source_absorbed_energy_j"]
    assert audit["source_delivered_energy_j"] > 0.0
    assert audit["resistor_absorbed_energy_j"] == audit["component_absorbed_energy_j"]["r1"]
    assert math.isclose(
        audit["source_delivered_mean_power_w"],
        audit["source_delivered_energy_j"] / stop,
        rel_tol=0,
        abs_tol=1e-14,
    )
    assert math.isclose(
        audit["resistor_absorbed_mean_power_w"],
        audit["resistor_absorbed_energy_j"] / stop,
        rel_tol=0,
        abs_tol=1e-14,
    )
    assert abs(audit["network_power_residual_j"]) < 1e-14
    assert audit["balance_residual_relative"] < 1e-8
    assert audit["balance_within_tolerance"] is True
    assert audit["balance_status"] == "within_numerical_tolerance"
    assert math.isclose(result.stats["integrated_interval_s"], stop, abs_tol=1e-15)
    assert result.stats["metric_topology_segments"] == 1
    assert result.stats["metric_quadrature_intervals"] > 0
    assert result.stats["wall_time_s"] > 0.0


def test_buck_switching_fingerprint_uses_time_not_output_grid() -> None:
    model = load_model(ROOT / "examples" / "buck.yaml")
    result = HybridSimulator(model).simulate()
    metrics = result.metrics()

    assert math.isclose(metrics["v(q1)"]["mean_last_half"], 60.0, abs_tol=1e-10)
    assert math.isclose(metrics["i(d1)"]["mean_last_half"], 2.4, abs_tol=1e-6)
    assert abs(metrics["v(q1)"]["sample_mean_last_half"] - 60.0) > 0.01
    assert abs(metrics["i(d1)"]["sample_mean_last_half"] - 2.4) > 0.005

    assert metrics["v(q1)"]["peak_to_peak_last_half"] == 100.0
    assert "ripple_last_half" not in metrics["v(q1)"]
    assert "ripple_last_half" not in metrics["i(q1)"]
    assert "ripple_last_half" not in metrics["i(d1)"]
    assert "ripple_last_half" not in metrics["i(vin)"]
    assert metrics["v(out)"]["ripple_last_half"] > 0.0
    assert metrics["i(l1)"]["ripple_last_half"] > 0.0
    assert metrics["i(load)"]["ripple_last_half"] > 0.0
    assert result.continuous_outputs == frozenset({"v(out)", "i(l1)", "i(load)"})


def test_time_metrics_are_independent_of_requested_output_grid() -> None:
    model = load_model(ROOT / "examples" / "buck.yaml")
    coarse_model = replace(
        model,
        transient=replace(model.transient, output_step=17e-6),
    )
    fine = HybridSimulator(model).simulate().metrics()
    coarse = HybridSimulator(coarse_model).simulate().metrics()

    for signal in model.outputs:
        for metric in ("mean", "rms", "mean_last_half", "rms_last_half"):
            assert math.isclose(fine[signal][metric], coarse[signal][metric], abs_tol=1e-12)
    assert fine["v(q1)"]["sample_mean"] != coarse["v(q1)"]["sample_mean"]


def test_dcm_integrals_and_natural_events_are_independent_of_output_grid() -> None:
    source = load_model(ROOT / "examples" / "buck_dcm.yaml")
    short = replace(
        source,
        transient=replace(source.transient, stop=500e-6),
    )
    coarse_model = replace(
        short,
        transient=replace(short.transient, output_step=5e-6),
    )

    fine_result = HybridSimulator(short).simulate()
    coarse_result = HybridSimulator(coarse_model).simulate()
    fine = fine_result.metrics()
    coarse = coarse_result.metrics()

    for signal in short.outputs:
        for metric in ("mean", "rms", "mean_last_half", "rms_last_half"):
            assert math.isclose(fine[signal][metric], coarse[signal][metric], abs_tol=1e-12)
    fine_events = [
        event.time
        for event in fine_result.events
        if event.kind == "natural_commutation" and event.target == "d1"
    ]
    coarse_events = [
        event.time
        for event in coarse_result.events
        if event.kind == "natural_commutation" and event.target == "d1"
    ]
    assert fine_events == coarse_events
    assert fine["v(out)"]["sample_count"] > coarse["v(out)"]["sample_count"]
