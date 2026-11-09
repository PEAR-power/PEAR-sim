# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Declared subcircuits: expansion semantics, refusals, equivalence.

The decisive test is bit-identity: the twelve-pulse rectifier rewritten
around a three-phase diode-bridge subcircuit must produce EXACTLY the same
waveforms as the flat original — hierarchy that moves a single digit is a
bug, not a refactor.
"""

from pathlib import Path

import numpy as np
import pytest

from pear_sim.core.solver import HybridSimulator
from pear_sim.errors import ModelError
from pear_sim.render import render_schematic
from pear_sim.run.contracts import model_hashes, resolved_model_dict
from pear_sim.spec.model import load_model

EXAMPLES = Path(__file__).parents[1] / "examples"


BUCK_LEG = """
schema: pear-sim/v0.1
name: {name}

subcircuits:
  rc_load:
    ports: [top, bottom]
    parameters: [r, c]
    components:
      - {{id: rload, type: resistor, p: top, n: bottom, value: "${{r}}"}}
      - {{id: cout, type: capacitor, p: top, n: bottom, value: "${{c}}", initial_voltage: 0 V}}

parameters:
  vin: 24 V
  rval: 6 ohm

circuit:
  ground: gnd
  components:
    - {{id: vsrc, type: voltage_source, p: mid, n: gnd, value: "${{vin}}"}}
    - {{id: load, type: subcircuit, ref: rc_load,
       ports: {{top: mid, bottom: gnd}},
       params: {{r: "${{rval}}", c: 10 uF}}}}

analysis:
  transient: {{start: 0 s, stop: 1 ms, output_step: 10 us}}

outputs:
  - v(mid)
  - i(load.rload)
"""


def _write(tmp_path: Path, text: str, *, name: str = "sub_model") -> Path:
    path = tmp_path / f"{name}.yaml"
    path.write_text(text.format(name=name), encoding="utf-8")
    return path


def test_an_instance_expands_into_namespaced_validated_components(tmp_path):
    model = load_model(_write(tmp_path, BUCK_LEG))
    by_id = {component.id: component for component in model.components}
    assert set(by_id) == {"vsrc", "load.rload", "load.cout"}
    # Ports resolve to the bound nets; parameter bindings resolve through the
    # ordinary ${...} machinery with dimension checks.
    assert (by_id["load.rload"].p, by_id["load.rload"].n) == ("mid", "gnd")
    assert by_id["load.rload"].value == pytest.approx(6.0)
    assert by_id["load.cout"].value == pytest.approx(10e-6)
    (instance,) = model.subcircuit_instances
    assert instance.path == "load"
    assert instance.ref == "rc_load"
    assert dict(instance.ports) == {"top": "mid", "bottom": "gnd"}
    assert instance.members == ("load.rload", "load.cout")


def test_internal_nodes_are_namespaced_and_ground_is_not_captured(tmp_path):
    text = """
schema: pear-sim/v0.1
name: {name}
subcircuits:
  divider:
    ports: [top, bottom]
    components:
      - {{id: r1, type: resistor, p: top, n: mid, value: 1 ohm}}
      - {{id: r2, type: resistor, p: mid, n: bottom, value: 1 ohm}}
      - {{id: c1, type: capacitor, p: mid, n: bottom, value: 1 uF, initial_voltage: 0 V}}
circuit:
  components:
    - {{id: vsrc, type: voltage_source, p: vin, n: gnd, value: 10 V}}
    - {{id: div, type: subcircuit, ref: divider, ports: {{top: vin, bottom: gnd}}}}
analysis:
  transient: {{start: 0 s, stop: 1 ms, output_step: 100 us}}
outputs:
  - v(div.mid)
"""
    model = load_model(_write(tmp_path, text))
    by_id = {component.id: component for component in model.components}
    assert by_id["div.r1"].n == "div.mid"
    assert by_id["div.r2"].p == "div.mid"
    result = HybridSimulator(model).simulate()
    assert result.signals["v(div.mid)"][-1] == pytest.approx(5.0, rel=1e-6)


def test_addressing_a_port_from_outside_resolves_to_the_bound_net(tmp_path):
    text = BUCK_LEG.replace("- v(mid)", "- v(load.top)")
    model = load_model(_write(tmp_path, text))
    # The rewrite happens at load time: the canonical output names the net.
    assert model.outputs == ("v(mid)", "i(load.rload)")


def test_port_current_is_refused_because_a_port_is_a_net(tmp_path):
    text = BUCK_LEG.replace("- i(load.rload)", "- i(load.top)")
    with pytest.raises(ModelError, match="port is a net, not a branch"):
        load_model(_write(tmp_path, text))


@pytest.mark.parametrize(
    ("needle", "replacement", "message"),
    [
        # Unbound port: drop `bottom` from the instance binding.
        (
            "ports: {{top: mid, bottom: gnd}}",
            "ports: {{top: mid}}",
            "unbound port",
        ),
        # Unknown port: bind one the definition does not declare.
        (
            "ports: {{top: mid, bottom: gnd}}",
            "ports: {{top: mid, bottom: gnd, shield: gnd}}",
            "unknown port",
        ),
        # Unbound parameter: drop `c`.
        (
            'params: {{r: "${{rval}}", c: 10 uF}}',
            'params: {{r: "${{rval}}"}}',
            "unbound parameter",
        ),
        # Unknown parameter: bind one the definition does not declare.
        (
            'params: {{r: "${{rval}}", c: 10 uF}}',
            'params: {{r: "${{rval}}", c: 10 uF, esr: 1 ohm}}',
            "unknown parameter",
        ),
        # Unknown definition.
        ("ref: rc_load,", "ref: lc_load,", "unknown subcircuit"),
    ],
)
def test_binding_mistakes_fail_closed_naming_the_instance(
    tmp_path, needle, replacement, message
):
    text = BUCK_LEG.replace(needle, replacement)
    assert text != BUCK_LEG
    with pytest.raises(ModelError) as excinfo:
        load_model(_write(tmp_path, text))
    assert message in str(excinfo.value)
    assert "load" in str(excinfo.value)


def test_a_body_reference_to_an_undeclared_parameter_is_refused(tmp_path):
    text = BUCK_LEG.replace('value: "${{c}}"', 'value: "${{esr}}"')
    with pytest.raises(ModelError, match="undeclared parameter"):
        load_model(_write(tmp_path, text))


def test_a_declared_parameter_nothing_references_is_refused(tmp_path):
    text = BUCK_LEG.replace('value: "${{c}}"', "value: 10 uF")
    with pytest.raises(ModelError, match="never referenced"):
        load_model(_write(tmp_path, text))


def test_recursive_references_are_refused_with_the_cycle(tmp_path):
    text = """
schema: pear-sim/v0.1
name: {name}
subcircuits:
  a:
    ports: [p, q]
    components:
      - {{id: inner, type: subcircuit, ref: b, ports: {{p: p, q: q}}}}
  b:
    ports: [p, q]
    components:
      - {{id: inner, type: subcircuit, ref: a, ports: {{p: p, q: q}}}}
circuit:
  components:
    - {{id: vsrc, type: voltage_source, p: vin, n: gnd, value: 10 V}}
    - {{id: top, type: subcircuit, ref: a, ports: {{p: vin, q: gnd}}}}
analysis:
  transient: {{start: 0 s, stop: 1 ms, output_step: 100 us}}
outputs:
  - v(vin)
"""
    with pytest.raises(ModelError, match="recursive subcircuit reference"):
        load_model(_write(tmp_path, text))


def test_separator_characters_in_segment_names_are_refused(tmp_path):
    # The `.` separator must stay 1:1 mappable to a KiCad `/` sheet path, so
    # neither character may appear inside a segment.
    text = BUCK_LEG.replace("id: load,", "id: load.a,")
    with pytest.raises(ModelError, match="may not contain"):
        load_model(_write(tmp_path, text))


def test_an_id_collision_with_an_expanded_component_is_refused(tmp_path):
    text = BUCK_LEG.replace(
        'value: "${{vin}}"',
        'value: "${{vin}}"}}\n    - {{id: load.rload, type: resistor, p: mid, n: gnd, value: 1 ohm',
    )
    with pytest.raises(ModelError, match="Duplicate component id"):
        load_model(_write(tmp_path, text))


def test_an_internal_node_shadowing_ground_is_refused(tmp_path):
    text = """
schema: pear-sim/v0.1
name: {name}
subcircuits:
  bad:
    ports: [top]
    components:
      - {{id: r1, type: resistor, p: top, n: gnd, value: 1 ohm}}
circuit:
  components:
    - {{id: vsrc, type: voltage_source, p: vin, n: gnd, value: 10 V}}
    - {{id: u1, type: subcircuit, ref: bad, ports: {{top: vin}}}}
analysis:
  transient: {{start: 0 s, stop: 1 ms, output_step: 100 us}}
outputs:
  - v(vin)
"""
    with pytest.raises(ModelError, match="shadows circuit.ground"):
        load_model(_write(tmp_path, text))


def test_declared_but_never_instantiated_subcircuits_are_refused(tmp_path):
    text = """
schema: pear-sim/v0.1
name: {name}
subcircuits:
  unused:
    ports: [p, q]
    components:
      - {{id: r1, type: resistor, p: p, n: q, value: 1 ohm}}
circuit:
  components:
    - {{id: vsrc, type: voltage_source, p: vin, n: gnd, value: 10 V}}
    - {{id: rl, type: resistor, p: vin, n: gnd, value: 1 ohm}}
analysis:
  transient: {{start: 0 s, stop: 1 ms, output_step: 100 us}}
outputs:
  - v(vin)
"""
    with pytest.raises(ModelError, match="never instantiated"):
        load_model(_write(tmp_path, text))


def test_couplings_inside_a_definition_are_namespaced(tmp_path):
    text = """
schema: pear-sim/v0.1
name: {name}
subcircuits:
  transformer:
    ports: [ap, an, bp, bn]
    parameters: [lp, ls]
    components:
      - {{id: lprim, type: inductor, p: ap, n: an, value: "${{lp}}", initial_current: 0 A}}
      - {{id: lsec, type: inductor, p: bp, n: bn, value: "${{ls}}", initial_current: 0 A}}
    couplings:
      - {{inductors: [lprim, lsec], coupling: 0.98}}
circuit:
  components:
    - {{id: vsrc, type: voltage_source, p: vin, n: gnd,
       waveform: {{type: sine, amplitude: 10 V, frequency: 1 kHz}}}}
    - {{id: t1, type: subcircuit, ref: transformer,
       ports: {{ap: vin, an: gnd, bp: sec, bn: gnd}},
       params: {{lp: 10 mH, ls: 10 mH}}}}
    - {{id: rl, type: resistor, p: sec, n: gnd, value: 100 ohm}}
analysis:
  transient: {{start: 0 s, stop: 1 ms, output_step: 10 us}}
outputs:
  - v(sec)
"""
    model = load_model(_write(tmp_path, text))
    (coupling,) = model.couplings
    assert {coupling.first, coupling.second} == {"t1.lprim", "t1.lsec"}


def test_nesting_flattens_through_every_level(tmp_path):
    text = """
schema: pear-sim/v0.1
name: {name}
subcircuits:
  cell:
    ports: [a, b]
    components:
      - {{id: r1, type: resistor, p: a, n: b, value: 2 ohm}}
  pair:
    ports: [top, bottom]
    components:
      - {{id: u1, type: subcircuit, ref: cell, ports: {{a: top, b: joint}}}}
      - {{id: u2, type: subcircuit, ref: cell, ports: {{a: joint, b: bottom}}}}
      - {{id: cj, type: capacitor, p: joint, n: bottom, value: 1 uF, initial_voltage: 0 V}}
circuit:
  components:
    - {{id: vsrc, type: voltage_source, p: vin, n: gnd, value: 8 V}}
    - {{id: chain, type: subcircuit, ref: pair, ports: {{top: vin, bottom: gnd}}}}
analysis:
  transient: {{start: 0 s, stop: 1 ms, output_step: 100 us}}
outputs:
  - v(chain.joint)
"""
    model = load_model(_write(tmp_path, text))
    ids = {component.id for component in model.components}
    assert {"chain.u1.r1", "chain.u2.r1"} <= ids
    # The mid node belongs to the OUTER definition, so it namespaces once.
    assert model.outputs == ("v(chain.joint)",)
    result = HybridSimulator(model).simulate()
    assert result.signals["v(chain.joint)"][-1] == pytest.approx(4.0, rel=1e-6)
    paths = {instance.path for instance in model.subcircuit_instances}
    assert paths == {"chain", "chain.u1", "chain.u2"}
    top = next(i for i in model.subcircuit_instances if i.path == "chain")
    assert set(top.members) == {"chain.u1.r1", "chain.u2.r1", "chain.cj"}


def test_hierarchy_is_presentation_never_physics(tmp_path):
    """Flat and hierarchical spellings of one circuit share `model_sha256`."""

    flat = """
schema: pear-sim/v0.1
name: {name}
circuit:
  components:
    - {{id: vsrc, type: voltage_source, p: mid, n: gnd, value: 24 V}}
    - {{id: load.rload, type: resistor, p: mid, n: gnd, value: 6 ohm}}
    - {{id: load.cout, type: capacitor, p: mid, n: gnd, value: 10 uF, initial_voltage: 0 V}}
analysis:
  transient: {{start: 0 s, stop: 1 ms, output_step: 10 us}}
outputs:
  - v(mid)
  - i(load.rload)
"""
    hier_model = load_model(_write(tmp_path, BUCK_LEG, name="hier"))
    flat_model = load_model(_write(tmp_path, flat, name="flat"))
    assert (
        model_hashes(hier_model)["model_sha256"]
        == model_hashes(flat_model)["model_sha256"]
    )
    # The resolved snapshot records the hierarchy additively — and only for
    # the model that declared it.
    assert "subcircuits" in resolved_model_dict(hier_model)
    assert "subcircuits" not in resolved_model_dict(flat_model)


def test_the_rewritten_twelve_pulse_example_is_bit_identical_to_the_flat_form(
    tmp_path,
):
    """The subcircuit acceptance regression, in-suite form.

    The shipped example now spells its two bridges as subcircuit instances.
    This test reconstructs the pre-subcircuit flat spelling (diodes y1..y6/z1..z6
    written out by hand) and asserts the waveforms agree BIT FOR BIT over a
    twenty-millisecond run.
    """

    source = (EXAMPLES / "rectifier_12pulse_yyd.yaml").read_text(encoding="utf-8")
    assert "type: subcircuit" in source
    short = source.replace("stop: 80 ms", "stop: 20 ms")

    bridge_y = """    - {id: y, type: subcircuit, ref: three_phase_diode_bridge,
       ports: {a: ya, b: yb, c: yc, dcp: ytop, dcn: dcmid}}"""
    flat_y = """    - {id: y1, type: diode, p: ya, n: ytop, initial: off}
    - {id: y2, type: diode, p: yb, n: ytop, initial: off}
    - {id: y3, type: diode, p: yc, n: ytop, initial: off}
    - {id: y4, type: diode, p: dcmid, n: ya, initial: off}
    - {id: y5, type: diode, p: dcmid, n: yb, initial: off}
    - {id: y6, type: diode, p: dcmid, n: yc, initial: off}"""
    bridge_z = """    - {id: z, type: subcircuit, ref: three_phase_diode_bridge,
       ports: {a: da, b: db, c: dc, dcp: dcmid, dcn: dbot}}"""
    flat_z = """    - {id: z1, type: diode, p: da, n: dcmid, initial: off}
    - {id: z2, type: diode, p: db, n: dcmid, initial: off}
    - {id: z3, type: diode, p: dc, n: dcmid, initial: off}
    - {id: z4, type: diode, p: dbot, n: da, initial: off}
    - {id: z5, type: diode, p: dbot, n: db, initial: off}
    - {id: z6, type: diode, p: dbot, n: dc, initial: off}"""
    assert bridge_y in short and bridge_z in short
    flat = short.replace(bridge_y, flat_y).replace(bridge_z, flat_z)
    # Strip the now-unused definition so the flat file is genuinely flat.
    start = flat.index("subcircuits:")
    end = flat.index("parameters:")
    flat = flat[:start] + flat[end:]

    hier_path = tmp_path / "hier12.yaml"
    hier_path.write_text(short, encoding="utf-8")
    flat_path = tmp_path / "flat12.yaml"
    flat_path.write_text(flat, encoding="utf-8")

    hier_result = HybridSimulator(load_model(hier_path)).simulate()
    flat_result = HybridSimulator(load_model(flat_path)).simulate()
    assert np.array_equal(hier_result.time, flat_result.time)
    for signal in flat_result.signals:
        assert np.array_equal(
            hier_result.signals[signal], flat_result.signals[signal]
        ), f"{signal} drifted: hierarchy moved a digit"


def test_render_draws_an_instance_as_a_box_and_can_expand_it():
    model = load_model(EXAMPLES / "rectifier_12pulse_yyd.yaml")
    collapsed = render_schematic(model)
    assert [
        (box.instance, box.ref) for box in collapsed.subcircuit_boxes
    ] == [
        ("y", "three_phase_diode_bridge"),
        ("z", "three_phase_diode_bridge"),
    ]
    assert 'data-subcircuit-instance="y"' in collapsed.svg
    port_names = {port.name for port in collapsed.subcircuit_boxes[0].ports}
    assert port_names == {"a", "b", "c", "dcp", "dcn"}
    collapsed_ids = {item.component_id for item in collapsed.metadata.components}
    assert "y.d1" not in collapsed_ids

    expanded = render_schematic(model, expand_subcircuits=True)
    assert expanded.subcircuit_boxes == ()
    expanded_ids = {item.component_id for item in expanded.metadata.components}
    assert "y.d1" in expanded_ids and "z.d6" in expanded_ids
    # Determinism, both views.
    assert render_schematic(model).svg == collapsed.svg
    assert (
        render_schematic(model, expand_subcircuits=True).svg == expanded.svg
    )
