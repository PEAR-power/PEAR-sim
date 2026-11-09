# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
# PEARSim provenance mark: fac45c9d-2490-41e0-8e53-bbd84e280889
"""Textbook layout for bridge converters: DC link, phase legs, clamps, loads.

The generic renderer places every net as one point and wires each pin to it in
a star. That stays legible for a buck, but a three-phase multilevel bridge turns
into a tangle: legs zigzag, loads sit inside legs and every wire crosses every
other. Published figures of these converters all follow one convention, and this
module draws it when the circuit has that structure:

* the DC link is a column of sources on the left, its positive and negative
  ends continue as horizontal rails along the top and the bottom;
* every phase leg is one vertical column of devices between the rails, an
  antiparallel switch/diode pair drawn as one device;
* clamp devices (NPC) form a column beside their leg, joined to their link tap
  by a horizontal bus;
* each leg's output drops down beside the leg and its load hangs below the
  bridge; load returns run along tracks under the loads;
* a wire crossing another net is drawn with a hop, so a crossing can never be
  read as a connection;
* voltage probes are instruments, not circuit elements: they are listed with
  the quantity they measure instead of being wired across the drawing.

Recognition is strict and the result is checked geometrically (no wire through
a symbol, no overlapping wires of different nets, no wire end touching another
net). Anything outside the pattern returns ``None`` and the caller falls back to
the generic layout, so this module can only ever replace a drawing with a
cleaner one of the same netlist.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from pear_sim.reporting.render import Point, _composite_partners, _PlacedComponent
from pear_sim.spec.model import Component, Model

ROW = 150.0  # pitch between node rows of a leg
PIN = 36.0  # half length of a symbol between its pins
CLAMP_PITCH = 110.0
LEG_AFTER_CLAMPS = 110.0
DROP_DX = 160.0
BRANCH_PITCH = 120.0
BLOCK_GAP = 140.0
FIRST_BLOCK = 150.0
TRACK_OFFSETS = (22.0, 31.0)
LOAD_TOP_GAP = 70.0
LOAD_PITCH = 140.0
RISER_DX = 90.0
RISER_PITCH = 30.0
HOP = 5.0

_SEMIS = frozenset({"ideal_switch", "diode", "thyristor"})

Segment = tuple[Point, Point]


@dataclass(frozen=True)
class BridgeDrawing:
    components: tuple[_PlacedComponent, ...]
    wires: dict[str, tuple[Segment, ...]]
    anchors: dict[str, Point]
    # Nets whose label sits left of the anchor (the link column).
    left_labels: frozenset[str]
    probes: tuple[_PlacedComponent, ...]


@dataclass(frozen=True)
class _Unit:
    ids: tuple[str, ...]
    nets: frozenset[str]


class _NotABridge(Exception):
    """The circuit is outside the pattern; the generic layout draws it."""


def bridge_drawing(model: Model) -> BridgeDrawing | None:
    try:
        return _Builder(model).build()
    except _NotABridge:
        return None


class _Builder:
    def __init__(self, model: Model) -> None:
        self.model = model
        self.by_id = {component.id: component for component in model.components}
        self.placed: list[_PlacedComponent] = []
        self.wires: dict[str, list[Segment]] = defaultdict(list)
        self.anchors: dict[str, Point] = {}

    # --- recognition -----------------------------------------------------

    def build(self) -> BridgeDrawing:
        model = self.model
        probes = [c for c in model.components if c.type == "voltage_probe"]
        circuit = [c for c in model.components if c.type != "voltage_probe"]
        link, link_y_order = self._link(circuit)
        top, bottom = link_y_order[0], link_y_order[-1]
        taps = set(link_y_order[1:-1])
        link_nets = set(link_y_order)
        rest = [c for c in circuit if c.id not in {item.id for item in link}]

        units = self._units([c for c in rest if c.type in _SEMIS])
        chain_units = [u for u in units if not (u.nets & taps)]
        clamp_units = [u for u in units if u.nets & taps]
        legs = self._legs(chain_units, top, bottom)
        if len(legs) < 2 or len({len(nodes) for nodes, _ in legs}) != 1:
            raise _NotABridge
        leg_of_node = {
            node: index for index, (nodes, _) in enumerate(legs) for node in nodes[1:-1]
        }

        clamps: dict[int, dict[str, list[tuple[int, _Unit]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for unit in clamp_units:
            tap_side = unit.nets & taps
            other = unit.nets - tap_side
            if len(tap_side) != 1 or len(other) != 1:
                raise _NotABridge
            (tap,) = tap_side
            (node,) = other
            if node not in leg_of_node:
                raise _NotABridge
            leg = leg_of_node[node]
            clamps[leg][tap].append((legs[leg][0].index(node), unit))

        loads = [c for c in rest if c.type not in _SEMIS]
        load_degree: dict[str, int] = defaultdict(int)
        for component in loads:
            load_degree[component.p] += 1
            load_degree[component.n] += 1
        outputs: list[str] = []
        for nodes, _ in legs:
            touched = [node for node in nodes[1:-1] if load_degree.get(node)]
            if len(touched) != 1:
                raise _NotABridge
            outputs.append(touched[0])
        for component in loads:
            for net in (component.p, component.n):
                if net in leg_of_node and net not in outputs:
                    raise _NotABridge

        order = sorted(range(len(legs)), key=lambda index: outputs[index])
        legs = [legs[index] for index in order]
        outputs = [outputs[index] for index in order]
        clamps = {position: clamps.get(index, {}) for position, index in enumerate(order)}

        chains = self._chains(loads, set(outputs) | link_nets, load_degree)
        trees = self._trees(chains, outputs, link_nets, load_degree)

        self._layout(link, link_y_order, legs, clamps, outputs, trees, taps)
        self._place_probes(probes)
        segments = {net: tuple(items) for net, items in self.wires.items()}
        self._check(segments)
        return BridgeDrawing(
            components=tuple(self.placed),
            wires=segments,
            anchors=dict(self.anchors),
            left_labels=frozenset(link_nets),
            probes=tuple(self._probe_placements),
        )

    def _link(self, circuit: list[Component]) -> tuple[list[Component], list[str]]:
        sources = [
            c
            for c in circuit
            if c.type == "voltage_source" and c.waveform is None and c.value is not None
        ]
        if not sources:
            raise _NotABridge
        adjacency: dict[str, list[Component]] = defaultdict(list)
        for source in sources:
            adjacency[source.p].append(source)
            adjacency[source.n].append(source)
        if any(len(items) > 2 for items in adjacency.values()):
            raise _NotABridge
        ends = sorted(net for net, items in adjacency.items() if len(items) == 1)
        if len(ends) != 2:
            raise _NotABridge
        potential = {ends[0]: 0.0}
        walk = [ends[0]]
        used: set[str] = set()
        while True:
            here = walk[-1]
            step = [s for s in adjacency[here] if s.id not in used]
            if not step:
                break
            source = step[0]
            used.add(source.id)
            there = source.n if source.p == here else source.p
            delta = source.value if source.p == here else -source.value
            potential[there] = potential[here] - delta
            walk.append(there)
        if len(used) != len(sources):
            raise _NotABridge
        if all(a > b for a, b in zip((potential[n] for n in walk), (potential[n] for n in walk[1:]))):
            return sources, walk
        if all(a < b for a, b in zip((potential[n] for n in walk), (potential[n] for n in walk[1:]))):
            return sources, list(reversed(walk))
        raise _NotABridge

    def _units(self, semis: list[Component]) -> list[_Unit]:
        partners = _composite_partners(self.model)
        units: list[_Unit] = []
        seen: set[str] = set()
        for component in sorted(semis, key=lambda item: item.id):
            if component.id in seen:
                continue
            partner = partners.get(component.id)
            ids = (component.id,) if partner is None else tuple(sorted((component.id, partner)))
            seen.update(ids)
            units.append(_Unit(ids, frozenset((component.p, component.n))))
        return units

    def _legs(
        self, units: list[_Unit], top: str, bottom: str
    ) -> list[tuple[list[str], list[_Unit]]]:
        at: dict[str, list[_Unit]] = defaultdict(list)
        for unit in units:
            if len(unit.nets) != 2:
                raise _NotABridge
            for net in unit.nets:
                at[net].append(unit)
        legs = []
        used: set[tuple[str, ...]] = set()
        for start in sorted(at[top], key=lambda unit: unit.ids):
            nodes, chain = [top], []
            unit, here = start, top
            while True:
                if unit.ids in used:
                    raise _NotABridge
                used.add(unit.ids)
                chain.append(unit)
                (here,) = unit.nets - {here}
                nodes.append(here)
                if here == bottom:
                    break
                if here == top or len(at[here]) != 2:
                    raise _NotABridge
                (unit,) = [item for item in at[here] if item is not unit]
            legs.append((nodes, chain))
        if len(used) != len(units):
            raise _NotABridge
        return legs

    def _chains(
        self, loads: list[Component], fixed: set[str], degree: dict[str, int]
    ) -> list[tuple[str, list[Component], str]]:
        anchors = {net for net, count in degree.items() if count != 2} | fixed
        at: dict[str, list[Component]] = defaultdict(list)
        for component in loads:
            at[component.p].append(component)
            at[component.n].append(component)
        used: set[str] = set()
        chains = []
        for start in sorted(anchors):
            for first in sorted(at.get(start, []), key=lambda item: item.id):
                if first.id in used:
                    continue
                here, component, members = start, first, []
                while True:
                    used.add(component.id)
                    members.append(component)
                    here = component.n if component.p == here else component.p
                    if here in anchors:
                        break
                    (component,) = [c for c in at[here] if c.id != component.id]
                chains.append((start, members, here))
        if len(used) != len(loads):
            raise _NotABridge
        return chains

    def _trees(self, chains, outputs, link_nets, degree):
        """For each output: its own chain, an optional branch point, its terminals."""

        output_set = set(outputs)
        by_end: dict[str, list[int]] = defaultdict(list)
        for index, (start, _, end) in enumerate(chains):
            by_end[start].append(index)
            by_end[end].append(index)
        owned: set[int] = set()
        trees: dict[str, tuple[list, list]] = {}
        for output in outputs:
            mine = [index for index in by_end[output] if index not in owned]
            if not mine:
                trees[output] = ([], [])
                continue
            if len(mine) != 1:
                raise _NotABridge
            (index,) = mine
            owned.add(index)
            nets, members = self._oriented(chains[index], output)
            end = nets[-1]
            branches = []
            # A star point joins the loads of several outputs; a local branch
            # point (LC filter node, a machine's parallel R//C) belongs to one.
            other_owners = [
                other
                for other in by_end[end]
                if other != index
                and any(
                    net in output_set and degree.get(net) == 1
                    for net in (chains[other][0], chains[other][2])
                    if net != end
                )
            ]
            local = end not in output_set and end not in link_nets and not other_owners
            if local:
                for other in sorted(by_end[end]):
                    if other == index:
                        continue
                    if other in owned:
                        raise _NotABridge
                    owned.add(other)
                    branch_nets, branch_members = self._oriented(chains[other], end)
                    terminal = branch_nets[-1]
                    if terminal not in link_nets and terminal not in output_set and (
                        degree.get(terminal, 0) <= 2
                    ):
                        raise _NotABridge
                    branches.append((branch_nets, branch_members))
            trees[output] = ([(nets, members)], branches)
        if len(owned) != len(chains):
            raise _NotABridge
        return trees

    @staticmethod
    def _oriented(chain, start):
        first, members, last = chain
        if first != start:
            members = list(reversed(members))
            first, last = last, first
        nets = [first]
        for component in members:
            nets.append(component.n if component.p == nets[-1] else component.p)
        return nets, members

    # --- geometry --------------------------------------------------------

    def _put(self, component: Component, center: Point, upper: str, *, lateral=0.0,
             labelled=True, orientation="vertical", label_side="right") -> tuple[Point, Point]:
        reversed_direction = component.p != upper
        placed = _PlacedComponent(
            component, center, orientation, reversed_direction,
            lateral=lateral, labelled=labelled, label_side=label_side,
        )
        self.placed.append(placed)
        if orientation == "vertical":
            return Point(center.x, center.y - PIN), Point(center.x, center.y + PIN)
        return Point(center.x - PIN, center.y), Point(center.x + PIN, center.y)

    def _put_unit(self, unit: _Unit, center: Point, upper: str) -> tuple[Point, Point]:
        pins = None
        for component_id in unit.ids:
            component = self.by_id[component_id]
            composite = len(unit.ids) == 2
            pins = self._put(
                component,
                center,
                upper,
                lateral=-15.0 if composite else 0.0,
                labelled=not composite or component.type == "ideal_switch",
            )
        return pins

    def _wire(self, net: str, *points: Point) -> None:
        for a, b in zip(points, points[1:]):
            if a != b:
                if a.x != b.x and a.y != b.y:
                    raise _NotABridge
                self.wires[net].append((a, b))

    def _layout(self, link, link_order, legs, clamps, outputs, trees, taps) -> None:
        rows = len(legs[0][0]) - 1
        bottom_y = rows * ROW
        top, bottom = link_order[0], link_order[-1]

        # Tap buses: one track in the gap under the row their clamp pairs centre on.
        tap_rows: dict[str, tuple[int, int]] = {}
        for leg_clamps in clamps.values():
            for tap, entries in leg_clamps.items():
                found = tuple(sorted(row for row, _ in entries))
                if len(found) != 2 or found[1] - found[0] < 2:
                    raise _NotABridge
                if tap_rows.setdefault(tap, found) != found:
                    raise _NotABridge
        track_y: dict[str, float] = {}
        per_gap: dict[int, list[str]] = defaultdict(list)
        for tap in sorted(tap_rows):
            per_gap[(tap_rows[tap][0] + tap_rows[tap][1]) // 2].append(tap)
        for gap, names in per_gap.items():
            if len(names) > len(TRACK_OFFSETS):
                raise _NotABridge
            for offset, tap in zip(TRACK_OFFSETS, names):
                track_y[tap] = gap * ROW + offset

        link_y = {top: 0.0, bottom: bottom_y, **track_y}
        known = [index for index, net in enumerate(link_order) if net in link_y]
        for left, right in zip(known, known[1:]):
            for index in range(left + 1, right):
                fraction = (index - left) / (right - left)
                link_y[link_order[index]] = link_y[link_order[left]] + fraction * (
                    link_y[link_order[right]] - link_y[link_order[left]]
                )
        ys = [link_y[net] for net in link_order]
        if any(b - a < 2 * PIN + 20.0 for a, b in zip(ys, ys[1:])):
            raise _NotABridge

        x_link = 0.0
        sources = {frozenset((s.p, s.n)): s for s in link}
        for upper, lower in zip(link_order, link_order[1:]):
            source = sources[frozenset((upper, lower))]
            center = Point(x_link, (link_y[upper] + link_y[lower]) / 2.0)
            pin_top, pin_bottom = self._put(source, center, upper)
            self._wire(upper, Point(x_link, link_y[upper]), pin_top)
            self._wire(lower, pin_bottom, Point(x_link, link_y[lower]))
        for net in link_order:
            self.anchors[net] = Point(x_link, link_y[net])

        cursor = x_link + FIRST_BLOCK
        drops: dict[str, float] = {}
        output_rows: dict[str, int] = {}
        leg_xs = []
        tap_reach: dict[str, float] = {}
        branch_room = {
            output: max(0, len(trees[output][1]) - 1) for output in outputs
        }
        for position, (nodes, units) in enumerate(legs):
            leg_clamps = clamps.get(position, {})
            ordered_taps = sorted(leg_clamps, key=lambda tap: (track_y[tap], tap))
            columns = {tap: cursor + index * CLAMP_PITCH for index, tap in enumerate(ordered_taps)}
            x_leg = cursor + (len(ordered_taps) - 1) * CLAMP_PITCH + LEG_AFTER_CLAMPS if ordered_taps else cursor
            leg_xs.append(x_leg)
            for row, net in enumerate(nodes):
                if 0 < row < rows:
                    self.anchors[net] = Point(x_leg, row * ROW)
            for row, unit in enumerate(units):
                center = Point(x_leg, (row + 0.5) * ROW)
                pin_top, pin_bottom = self._put_unit(unit, center, nodes[row])
                self._wire(nodes[row], Point(x_leg, row * ROW), pin_top)
                self._wire(nodes[row + 1], pin_bottom, Point(x_leg, (row + 1) * ROW))
            for tap, x_clamp in columns.items():
                tap_reach[tap] = max(tap_reach.get(tap, x_clamp), x_clamp)
                (upper_row, upper_unit), (lower_row, lower_unit) = sorted(
                    leg_clamps[tap], key=lambda entry: entry[0]
                )
                upper_net, lower_net = nodes[upper_row], nodes[lower_row]
                junction = Point(x_clamp, track_y[tap])
                pin_top, pin_bottom = self._put_unit(
                    upper_unit, Point(x_clamp, (upper_row + 0.5) * ROW), upper_net
                )
                self._wire(upper_net, Point(x_leg, upper_row * ROW), Point(x_clamp, upper_row * ROW), pin_top)
                self._wire(tap, pin_bottom, junction)
                pin_top, pin_bottom = self._put_unit(
                    lower_unit, Point(x_clamp, (lower_row - 0.5) * ROW), tap
                )
                self._wire(tap, junction, pin_top)
                self._wire(lower_net, pin_bottom, Point(x_clamp, lower_row * ROW), Point(x_leg, lower_row * ROW))
            output = outputs[position]
            output_rows[output] = nodes.index(output)
            drops[output] = x_leg + DROP_DX
            cursor = drops[output] + branch_room[output] * BRANCH_PITCH + BLOCK_GAP

        self._wire(top, *[Point(x, 0.0) for x in [x_link, *leg_xs]])
        self._wire(bottom, *[Point(x, bottom_y) for x in [x_link, *leg_xs]])
        for tap, reach in tap_reach.items():
            self._wire(tap, Point(x_link, track_y[tap]), Point(reach, track_y[tap]))

        # Loads hang below the bridge, one column per output.
        load_top = bottom_y + LOAD_TOP_GAP
        terminals: dict[str, list[Point]] = defaultdict(list)
        lowest = load_top
        for output in outputs:
            x = drops[output]
            row_y = output_rows[output] * ROW
            own, branches = trees[output]
            if not own:
                # Reached by another output's load: its drop joins that load's track.
                terminals[output].append(Point(x, row_y))
                continue
            nets, members = own[0]
            self._wire(output, Point(x - DROP_DX, row_y), Point(x, row_y))
            bottom_pin, last_point = self._hang(x, load_top, nets, members, Point(x, row_y))
            lowest = max(lowest, bottom_pin.y)
            if branches:
                junction_net = nets[-1]
                self._wire(junction_net, bottom_pin, last_point)
                self.anchors[junction_net] = last_point
                for index, (branch_nets, branch_members) in enumerate(branches):
                    bx = x + index * BRANCH_PITCH
                    if index:
                        self._wire(junction_net, last_point, Point(bx, last_point.y))
                    start = Point(bx, last_point.y)
                    pin, _ = self._hang(bx, last_point.y + 20.0, branch_nets, branch_members, start)
                    lowest = max(lowest, pin.y)
                    terminals[branch_nets[-1]].append(pin)
            else:
                terminals[nets[-1]].append(bottom_pin)

        track_names = sorted(
            net for net in terminals if any(p.y > bottom_y for p in terminals[net]) or net in outputs
        )
        risers = [net for net in track_names if net in self.anchors and net in set(link_order)]
        for index, net in enumerate(track_names):
            y = lowest + 40.0 + index * RISER_PITCH
            xs = []
            for point in terminals[net]:
                if point.y <= bottom_y:  # an output reached by another output's load
                    self._wire(net, Point(point.x - DROP_DX, point.y), point, Point(point.x, y))
                else:
                    self._wire(net, point, Point(point.x, y))
                xs.append(point.x)
            if net in risers:
                x_riser = x_link - RISER_DX - risers.index(net) * RISER_PITCH
                xs.append(x_riser)
                self._wire(net, Point(x_riser, y), Point(x_riser, link_y[net]), Point(x_link, link_y[net]))
            xs = sorted(set(xs))
            if len(xs) > 1:
                self._wire(net, *[Point(value, y) for value in xs])
            if net not in self.anchors:
                self.anchors[net] = Point(xs[0], y)
        for output in outputs:
            self.anchors.setdefault(output, Point(drops[output] - DROP_DX, output_rows[output] * ROW))
        self._lowest = lowest + 40.0 + len(track_names) * RISER_PITCH

    def _hang(self, x, y_top, nets, members, start: Point):
        """Stack ``members`` downward at ``x``; return the last pin and end point."""

        previous = start
        for index, component in enumerate(members):
            center = Point(x, y_top + 56.0 + index * LOAD_PITCH)
            pin_top, pin_bottom = self._put(component, center, nets[index])
            self._wire(nets[index], previous, Point(x, previous.y), pin_top)
            node = Point(x, center.y + PIN + 34.0)
            if index + 1 < len(members):
                self._wire(nets[index + 1], pin_bottom, node)
                self.anchors.setdefault(nets[index + 1], node)
                previous = node
            else:
                previous = pin_bottom
        return previous, Point(x, previous.y + 34.0) if members else start

    def _place_probes(self, probes: list[Component]) -> None:
        self._probe_placements = []
        y = getattr(self, "_lowest", 0.0) + 60.0
        for index, probe in enumerate(sorted(probes, key=lambda item: item.id)):
            column, line = divmod(index, 6)
            center = Point(column * 260.0, y + line * 70.0)
            self._probe_placements.append(
                _PlacedComponent(probe, center, "horizontal", False, label_side="probe")
            )

    # --- verification ----------------------------------------------------

    def _check(self, wires: dict[str, tuple[Segment, ...]]) -> None:
        boxes = []
        for placed in self.placed:
            c = placed.center
            half_w = 22.0 + abs(placed.lateral)
            if placed.orientation == "vertical":
                boxes.append((placed.component, (c.x - half_w, c.y - PIN + 2, c.x + half_w, c.y + PIN - 2)))
            else:
                boxes.append((placed.component, (c.x - PIN + 2, c.y - half_w, c.x + PIN - 2, c.y + half_w)))
        flat = [(net, a, b) for net, items in wires.items() for a, b in items]
        for _net, a, b in flat:
            for _component, box in boxes:
                if _crosses_box(a, b, box):
                    raise _NotABridge
        for index, (net, a, b) in enumerate(flat):
            for other, c, d in flat[index + 1:]:
                if other == net:
                    continue
                if _collinear_overlap(a, b, c, d):
                    raise _NotABridge
                for point in (a, b):
                    if _on_segment(point, c, d):
                        raise _NotABridge
                for point in (c, d):
                    if _on_segment(point, a, b):
                        raise _NotABridge


def _crosses_box(a: Point, b: Point, box) -> bool:
    x0, y0, x1, y1 = box
    lo_x, hi_x = sorted((a.x, b.x))
    lo_y, hi_y = sorted((a.y, b.y))
    return lo_x < x1 and hi_x > x0 and lo_y < y1 and hi_y > y0


def _on_segment(point: Point, a: Point, b: Point) -> bool:
    if a.x == b.x == point.x:
        return min(a.y, b.y) <= point.y <= max(a.y, b.y)
    if a.y == b.y == point.y:
        return min(a.x, b.x) <= point.x <= max(a.x, b.x)
    return False


def _collinear_overlap(a: Point, b: Point, c: Point, d: Point) -> bool:
    if a.x == b.x == c.x == d.x:
        return min(max(a.y, b.y), max(c.y, d.y)) > max(min(a.y, b.y), min(c.y, d.y))
    if a.y == b.y == c.y == d.y:
        return min(max(a.x, b.x), max(c.x, d.x)) > max(min(a.x, b.x), min(c.x, d.x))
    return False


def hops(wires: dict[str, tuple[Segment, ...]]) -> dict[tuple[str, int], list[float]]:
    """x positions where a horizontal wire jumps over a vertical wire of another net."""

    vertical = [
        (net, a, b) for net, items in wires.items() for a, b in items if a.x == b.x and a.y != b.y
    ]
    result: dict[tuple[str, int], list[float]] = {}
    for net, items in wires.items():
        for index, (a, b) in enumerate(items):
            if a.y != b.y:
                continue
            lo, hi = sorted((a.x, b.x))
            xs = sorted(
                c.x
                for other, c, d in vertical
                if other != net
                and lo + HOP < c.x < hi - HOP
                and min(c.y, d.y) < a.y < max(c.y, d.y)
            )
            if xs:
                result[(net, index)] = xs
    return result


def junctions(segments: tuple[Segment, ...], pins: list[Point]) -> list[Point]:
    """Points where three or more wire ends (or wire ends and pins) meet."""

    degree: dict[tuple[float, float], int] = defaultdict(int)
    points = {(p.x, p.y) for a, b in segments for p in (a, b)}
    for a, b in segments:
        for x, y in points:
            point = Point(x, y)
            if point in (a, b):
                degree[(x, y)] += 1
            elif _on_segment(point, a, b):
                degree[(x, y)] += 2
    return [Point(x, y) for (x, y), count in sorted(degree.items()) if count >= 3]


__all__ = ["BridgeDrawing", "bridge_drawing", "hops", "junctions"]
