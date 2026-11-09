# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from pear_sim.benchmarks import (
    compare_buck_boundary_point,
    extract_buck_boundary_parameters,
    observe_simulation_result,
    run_buck_boundary_sweep,
)
from pear_sim.model import load_model
from pear_sim.solver import HybridSimulator
from pear_sim.sweep import _periodic_point_model


ROOT = Path(__file__).parents[1]


def _comparison(model, reference, sweep, *, rtol, atol, max_step, output_step):
    configured = replace(
        model,
        transient=replace(
            model.transient,
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            output_step=output_step,
        ),
    )
    result = HybridSimulator(configured).simulate()
    observation = observe_simulation_result(
        result,
        reference,
        parameters=sweep.parameters,
        bcm_load_ohm=sweep.bcm_load_ohm,
    )
    return compare_buck_boundary_point(sweep.parameters, reference, observation)


def test_representative_ccm_bcm_dcm_errors_contract_with_tighter_settings() -> None:
    source = load_model(ROOT / "examples" / "buck_dcm.yaml")
    parameters = extract_buck_boundary_parameters(source)
    sweep = run_buck_boundary_sweep(parameters, [0.5, 1.0, 2.0])
    period = parameters.period_s
    loose_statuses: list[str] = []

    for reference in sweep.points:
        model = _periodic_point_model(source, reference)
        loose = _comparison(
            model,
            reference,
            sweep,
            rtol=1e-6,
            atol=1e-8,
            max_step=period / 20.0,
            output_step=period / 40.0,
        )
        tight = _comparison(
            model,
            reference,
            sweep,
            rtol=1e-9,
            atol=1e-11,
            max_step=period / 80.0,
            output_step=period / 200.0,
        )
        loose_statuses.append(loose["status"])

        assert tight["status"] == "pass"
        loose_gates = {gate["id"]: gate for gate in loose["gates"]}
        tight_gates = {gate["id"]: gate for gate in tight["gates"]}
        for gate_id in (
            "boundary_state_nonzero",
            "cycle_mean_output",
            "energy_balance",
        ):
            assert tight_gates[gate_id]["actual"] < loose_gates[gate_id]["actual"]

    # The exact boundary is deliberately sensitive: coarse settings classify its
    # tiny numerical idle sliver as DCM. The coarse DCM point also exceeds the
    # strengthened periodicity gate. Tight settings recover every blocking gate;
    # this is why a convergence check is evidence, not an oracle.
    assert loose_statuses == ["pass", "fail", "fail"]
