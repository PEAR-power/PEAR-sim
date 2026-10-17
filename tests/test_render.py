from dataclasses import replace
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from pear_sim.model import load_model
from pear_sim.render import RenderOptions, render_schematic, render_svg


EXAMPLES = Path(__file__).parents[1] / "examples"
SVG_NAMESPACE = {"svg": "http://www.w3.org/2000/svg"}


@pytest.mark.parametrize(
    "filename",
    [
        "buck.yaml",
        "buck_dcm.yaml",
        "boost.yaml",
        "inverter_three_phase_six_step.yaml",
        "inverter_three_phase_spwm.yaml",
        "thyristor_phase_controlled.yaml",
    ],
)
def test_render_is_deterministic_and_contains_every_component_and_net(filename: str):
    model = load_model(EXAMPLES / filename)

    first = render_schematic(model)
    second = render_schematic(model)
    reordered = render_svg(model=replace(model, components=tuple(reversed(model.components))))

    assert first.svg == second.svg == reordered
    root = ET.fromstring(first.svg)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"

    all_ids = [element.attrib["id"] for element in root.iter() if "id" in element.attrib]
    assert len(all_ids) == len(set(all_ids))

    component_elements = root.findall(".//svg:g[@data-component-id]", SVG_NAMESPACE)
    net_elements = root.findall(".//svg:g[@data-net-name]", SVG_NAMESPACE)
    expected_components = {component.id for component in model.components}
    expected_nets = {
        model.ground,
        *(node for component in model.components for node in (component.p, component.n)),
    }

    assert {element.attrib["data-component-id"] for element in component_elements} == (
        expected_components
    )
    assert {element.attrib["data-net-name"] for element in net_elements} == expected_nets
    assert {placement.component_id for placement in first.metadata.components} == expected_components
    assert {placement.name for placement in first.metadata.nets} == expected_nets
    assert first.metadata.width == int(root.attrib["width"])
    assert first.metadata.height == int(root.attrib["height"])

    placements = {item.component_id: item for item in first.metadata.components}
    if filename == "buck.yaml":
        # d1.p is ground below the switch node, so its visible p terminal must be lower.
        assert placements["d1"].p_pin.y > placements["d1"].n_pin.y
        assert placements["d1"].rotation == -90

    for component in model.components:
        assert root.find(f".//svg:g[@id='component-{component.id}']", SVG_NAMESPACE) is not None
    for net in expected_nets:
        assert root.find(f".//svg:g[@id='net-{net}']", SVG_NAMESPACE) is not None


def test_layout_follows_the_potential_order_of_a_buck() -> None:
    """The vertical axis is the rail order, so a chain reads top to bottom."""

    result = render_schematic(load_model(EXAMPLES / "buck_dcm.yaml"))
    nets = {item.name: item for item in result.metadata.nets}

    assert result.metadata.renderer == "pear_sim-svg/v0.7"
    assert (
        nets["in"].point.y
        < nets["sw"].point.y
        < nets["out"].point.y
        < nets["gnd"].point.y
    ), "input rail must sit above the switching node, output and ground"
    # A single chain occupies one column; the source bridging it gets a lane
    # of its own instead of covering the devices in between.
    assert nets["in"].point.x == pytest.approx(nets["gnd"].point.x)
    components = {item.component_id: item for item in result.metadata.components}
    assert components["vin"].center.x > nets["in"].point.x
    assert not result.metadata.warnings


def test_converter_leg_reads_as_one_descending_column() -> None:
    """Every NPC leg must draw as a column, rails top and bottom."""

    result = render_schematic(load_model(EXAMPLES / "npc_three_phase.yaml"))
    nets = {item.name: item for item in result.metadata.nets}
    chain = ["dcp", "n1a", "pa", "n2a", "dcn"]
    heights = [nets[name].point.y for name in chain]
    assert heights == sorted(heights), f"leg A is not monotone top-to-bottom: {heights}"
    assert nets["dcp"].point.y < nets["mid"].point.y < nets["dcn"].point.y

    # Each phase forms its own column: the three pole nodes share no abscissa.
    poles = {nets[name].point.x for name in ("pa", "pb", "pc")}
    assert len(poles) == 3

    # A switch and its antiparallel diode share a node pair and must stay
    # side by side rather than being scattered by overlap avoidance.
    components = {item.component_id: item for item in result.metadata.components}
    for switch, diode in (("q1a", "d1a"), ("q2a", "d2a"), ("q3a", "d3a")):
        assert components[switch].center.y == pytest.approx(
            components[diode].center.y, abs=1.0
        )
        assert abs(components[switch].center.x - components[diode].center.x) <= 200.0
    assert not result.metadata.warnings


def test_render_exposes_values_controls_and_component_types():
    model = load_model(EXAMPLES / "buck.yaml")
    result = render_schematic(model)
    root = ET.fromstring(result.svg)
    visible_text = " ".join(text.text or "" for text in root.findall(".//svg:text", SVG_NAMESPACE))

    assert "100 V" in visible_text
    assert "1 mH" in visible_text
    assert "100 µF" in visible_text
    assert "10 Ω" in visible_text
    # Gate bindings moved off the symbols into the legend: repeating them
    # under every device is what made a dense bridge unreadable.
    assert "gate: pwm" not in visible_text
    assert "q1 \u2190 pwm" in visible_text
    assert "closed" not in visible_text

    components = {
        element.attrib["data-component-id"]: element
        for element in root.findall(".//svg:g[@data-component-id]", SVG_NAMESPACE)
    }
    assert components["q1"].attrib["data-component-type"] == "ideal_switch"
    assert components["d1"].attrib["data-component-type"] == "diode"
    assert components["l1"].attrib["data-p"] == "sw"
    assert components["l1"].attrib["data-n"] == "out"


def test_render_can_hide_visible_values_without_changing_connectivity():
    model = load_model(EXAMPLES / "boost.yaml")
    result = render_schematic(model, RenderOptions(show_values=False))
    root = ET.fromstring(result.svg)

    assert not root.findall(".//svg:text[@class='component-value']", SVG_NAMESPACE)
    assert len(root.findall(".//svg:g[@data-component-id]", SVG_NAMESPACE)) == len(
        model.components
    )
    assert len(root.findall(".//svg:g[@data-net-name]", SVG_NAMESPACE)) == len(
        {model.ground, *(node for item in model.components for node in (item.p, item.n))}
    )


def test_antiparallel_pairs_draw_as_one_composite_device() -> None:
    """A switch and its body diode are one device in every published figure."""

    model = load_model(EXAMPLES / "npc_three_phase.yaml")
    result = render_schematic(model)
    components = {item.component_id: item for item in result.metadata.components}

    # Shared centre and shared pins: the pair reads as a single symbol.
    assert components["q1a"].center == components["d1a"].center
    assert components["q1a"].p_pin == components["d1a"].n_pin
    assert components["q1a"].n_pin == components["d1a"].p_pin

    root = ET.fromstring(result.svg)
    labels = [
        text.text or ""
        for text in root.findall(".//svg:text[@class='component-label']", SVG_NAMESPACE)
    ]
    assert "q1a/d1a" in labels, "the composite carries one label naming both devices"
    assert "d1a" not in labels, "the diode must not be labelled a second time"
    # One label per composite, so the drawing carries fewer labels than parts.
    assert len(labels) < len(model.components)

    # Both members stay individually addressable for review tooling.
    for component_id in ("q1a", "d1a"):
        assert (
            root.find(f".//svg:g[@id='component-{component_id}']", SVG_NAMESPACE)
            is not None
        )


def test_gate_legend_lists_every_switch() -> None:
    model = load_model(EXAMPLES / "npc_three_phase.yaml")
    root = ET.fromstring(render_svg(model))
    legend = root.find(".//svg:g[@id='schematic-legend']", SVG_NAMESPACE)
    assert legend is not None
    rows = " ".join(text.text or "" for text in legend.iter())
    for component in model.components:
        if component.type == "ideal_switch":
            assert f"{component.id} \u2190 {component.control}" in rows



BRIDGE_EXAMPLES = [
    "npc_three_phase.yaml",
    "npc5_three_phase.yaml",
    "npc_three_phase_lc_filter.yaml",
    "inverter_three_phase_spwm.yaml",
    "inverter_three_phase_floating_neutral.yaml",
    "pmsm_inverter_drive.yaml",
    "dc_drive_hbridge.yaml",
]


def _layout_kind(svg: str) -> str:
    import json as _json

    root = ET.fromstring(svg)
    metadata = root.find("svg:metadata", SVG_NAMESPACE)
    return _json.loads(metadata.text).get("layout", "generic")


def _drawn_points(d: str) -> list[tuple[float, float]]:
    """Vertices of a wire path; a hop arc keeps the wire on its own line."""

    tokens = d.replace(",", " ").split()
    points, index = [], 0
    while index < len(tokens):
        command = tokens[index]
        if command in ("M", "L"):
            points.append((float(tokens[index + 1]), float(tokens[index + 2])))
            index += 3
        elif command == "A":
            points.append((float(tokens[index + 6]), float(tokens[index + 7])))
            index += 8
        else:
            raise AssertionError(f"unexpected path command {command!r}")
    return points


def _on(point, a, b) -> bool:
    (x, y), (ax, ay), (bx, by) = point, a, b
    if abs(ax - bx) < 1e-6 and abs(x - ax) < 1e-6:
        return min(ay, by) - 1e-6 <= y <= max(ay, by) + 1e-6
    if abs(ay - by) < 1e-6 and abs(y - ay) < 1e-6:
        return min(ax, bx) - 1e-6 <= x <= max(ax, bx) + 1e-6
    return False


@pytest.mark.parametrize("filename", BRIDGE_EXAMPLES)
def test_bridge_drawing_connects_exactly_the_netlist(filename: str) -> None:
    """Rebuild connectivity from the drawn geometry alone and compare it with the
    netlist: every net is one connected drawing that reaches each of its pins,
    and no wire touches a pin of another net. This is the drawing checked
    against the model by a different method than the one that drew it."""

    model = load_model(EXAMPLES / filename)
    result = render_schematic(model)
    assert _layout_kind(result.svg) == "bridge"
    root = ET.fromstring(result.svg)
    segments: dict[str, list] = {}
    for group in root.findall(".//svg:g[@data-net-name]", SVG_NAMESPACE):
        net = group.attrib["data-net-name"]
        for path in group.findall("svg:path[@class='wire']", SVG_NAMESPACE):
            points = _drawn_points(path.attrib["d"])
            # A hopped horizontal wire is one straight conductor end to end.
            segments.setdefault(net, []).append((points[0], points[-1]))
    pins: dict[str, list] = {}
    probes = {c.id for c in model.components if c.type == "voltage_probe"}
    for placement in result.metadata.components:
        if placement.component_id in probes:
            continue
        component = next(c for c in model.components if c.id == placement.component_id)
        pins.setdefault(component.p, []).append((placement.p_pin.x, placement.p_pin.y))
        pins.setdefault(component.n, []).append((placement.n_pin.x, placement.n_pin.y))

    for net, net_pins in pins.items():
        wires = segments.get(net, [])
        assert wires, f"net {net} has no drawn wire"
        # Union-find over wires: two wires join when an end of one lies on the other.
        parent = list(range(len(wires)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, (a, b) in enumerate(wires):
            for j, (c, d) in enumerate(wires):
                if i < j and (_on(a, c, d) or _on(b, c, d) or _on(c, a, b) or _on(d, a, b)):
                    parent[find(i)] = find(j)
        assert len({find(i) for i in range(len(wires))}) == 1, f"net {net} is drawn in pieces"
        for pin in net_pins:
            assert any(_on(pin, a, b) for a, b in wires), f"pin of net {net} at {pin} is not wired"
        for other, other_wires in segments.items():
            if other == net:
                continue
            for pin in net_pins:
                assert not any(_on(pin, a, b) for a, b in other_wires), (
                    f"a wire of net {other} touches a pin of net {net}"
                )


def test_bridge_layout_is_deterministic_and_hops_every_crossing() -> None:
    model = load_model(EXAMPLES / "npc5_three_phase.yaml")
    first = render_svg(model)
    assert first == render_svg(replace(model, components=tuple(reversed(model.components))))
    root = ET.fromstring(first)
    hopped = [
        path
        for path in root.findall(".//svg:path[@class='wire']", SVG_NAMESPACE)
        if " A " in path.attrib["d"]
    ]
    assert hopped, "a three-phase five-level bridge cannot be drawn without crossings"
    labels = " ".join(t.text or "" for t in root.findall(".//svg:text", SVG_NAMESPACE))
    assert "s_h1a/d_h1a" in labels


def test_bridge_layout_lists_probes_with_what_they_measure() -> None:
    root = ET.fromstring(render_svg(load_model(EXAMPLES / "inverter_three_phase_spwm.yaml")))
    labels = [t.text or "" for t in root.findall(".//svg:text[@class='component-label']", SVG_NAMESPACE)]
    assert "vab = v(a) − v(b)" in labels


@pytest.mark.parametrize("filename", ["buck.yaml", "boost.yaml", "npc_three_phase_midpoint.yaml"])
def test_circuits_outside_the_bridge_pattern_keep_the_generic_layout(filename: str) -> None:
    assert _layout_kind(render_svg(load_model(EXAMPLES / filename))) == "generic"


def test_a_leg_with_a_load_on_an_inner_node_falls_back_instead_of_misdrawing(tmp_path) -> None:
    """The bridge pattern allows one output per leg. A second tap on a leg is
    outside it, so the drawing must fall back rather than drop the connection."""

    text = (EXAMPLES / "npc_three_phase.yaml").read_text(encoding="utf-8")
    text = text.replace(
        "    - {id: la,",
        "    - {id: rx, type: resistor, p: n1a, n: mid, value: 1 kohm}\n    - {id: la,",
        1,
    )
    path = tmp_path / "extra.yaml"
    path.write_text(text, encoding="utf-8")
    assert _layout_kind(render_svg(load_model(path))) == "generic"
