# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 PEAR Power and the PEARSim contributors.
"""Deterministic, dependency-free SVG rendering for PEARSim models.

The renderer is deliberately a presentation-only layer.  It consumes an
already validated :class:`pear_sim.spec.model.Model`, never invokes the simulator,
and never changes circuit connectivity to improve the drawing.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, replace
import hashlib
import heapq
import json
import math
import re
from typing import Iterable
from xml.sax.saxutils import escape, quoteattr

from pear_sim.run.contracts import model_hashes
from pear_sim.spec.model import Component, Model
from pear_sim.spec.subcircuits import SEPARATOR, SubcircuitInstance


RENDERER_VERSION = "pear_sim-svg/v0.7"

# Lateral half-spacing of the two bodies inside a composite device symbol.
_COMPOSITE_LATERAL = 15.0


@dataclass(frozen=True)
class RenderOptions:
    """Geometry and visibility settings for the deterministic renderer."""

    margin: float = 56.0
    node_gap: float = 170.0
    layer_gap: float = 170.0
    # A component body plus its two label rows is about 130 units wide, so
    # parallel members of one node pair (a switch and its body diode) need at
    # least that much lateral room; anything tighter forced the overlap
    # fallback to scatter them across the drawing.
    component_lane_gap: float = 150.0
    show_values: bool = True

    def __post_init__(self) -> None:
        for name in ("margin", "node_gap", "layer_gap", "component_lane_gap"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive finite number")


@dataclass(frozen=True)
class Point:
    x: float
    y: float


@dataclass(frozen=True)
class ComponentPlacement:
    component_id: str
    svg_id: str
    component_type: str
    center: Point
    orientation: str
    rotation: int
    p_pin: Point
    n_pin: Point


@dataclass(frozen=True)
class NetPlacement:
    name: str
    svg_id: str
    point: Point
    degree: int
    is_ground: bool


@dataclass(frozen=True)
class RenderMetadata:
    """Machine-readable geometry corresponding exactly to the returned SVG."""

    renderer: str
    width: int
    height: int
    model_name: str
    model_sha256: str
    components: tuple[ComponentPlacement, ...]
    nets: tuple[NetPlacement, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SubcircuitPortPin:
    """One named port of a collapsed subcircuit box and its wire target."""

    name: str
    net: str
    pin: Point


@dataclass(frozen=True)
class SubcircuitBox:
    """Geometry of one subcircuit instance drawn as a box.

    Presentation only: the box replaces its member components in the
    drawing, never in the electrical model — there is no alternate
    connectivity path (clean-room rule unchanged).
    """

    instance: str
    ref: str
    svg_id: str
    x: float
    y: float
    width: float
    height: float
    ports: tuple[SubcircuitPortPin, ...]


@dataclass(frozen=True)
class RenderResult:
    svg: str
    metadata: RenderMetadata
    # Non-empty only when the model declares subcircuits and the caller did
    # not ask for the expanded view; additive so existing callers and
    # artifacts are untouched.
    subcircuit_boxes: tuple[SubcircuitBox, ...] = ()


@dataclass(frozen=True)
class _PlacedComponent:
    component: Component
    center: Point
    orientation: str
    reversed: bool
    # Lateral shift of the body inside the component's own rotated frame.
    # Two antiparallel devices share one centre and one pin pair; because
    # their rotations differ by 180 degrees, the same negative shift puts
    # them on opposite sides and they read as one composite device symbol.
    lateral: float = 0.0
    # A composite is labelled once, by its switch.
    labelled: bool = True
    # "below" (generic layout), "right" (bridge layout: labels beside a column
    # of devices, off the wires) or "probe" (instrument list, no wires).
    label_side: str = "below"


_SAFE_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_SOURCES = frozenset({"voltage_source", "current_source"})
# Which member of a parallel group decides the drawing direction of its node
# pair (lower wins). See :func:`_collapsed_edges`.
_DIRECTION_PRIORITY = {
    "voltage_source": 0,
    "current_source": 0,
    "ideal_switch": 1,
    "inductor": 2,
    "sat_inductor": 2,
    "capacitor": 2,
    "resistor": 2,
    "diode": 3,
    "thyristor": 3,
    "voltage_probe": 4,
}
_UNIT_BY_TYPE = {
    "resistor": "Ω",
    "capacitor": "F",
    "inductor": "H",
    "sat_inductor": "H",
    "voltage_source": "V",
    "current_source": "A",
}
_PREFIXES = {
    -12: "p",
    -9: "n",
    -6: "µ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
}


def _svg_id(kind: str, raw: str) -> str:
    if _SAFE_ID_RE.fullmatch(raw):
        return f"{kind}-{raw}"
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", raw).strip("-") or "item"
    if not re.match(r"[A-Za-z_]", slug):
        slug = f"item-{slug}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{kind}-{slug}-{digest}"


def _coord(value: float) -> str:
    if abs(value) < 0.0005:
        value = 0.0
    return f"{value:.3f}".rstrip("0").rstrip(".")


def _attrs(**values: object) -> str:
    rendered: list[str] = []
    for key in sorted(values):
        value = values[key]
        if value is None:
            continue
        name = key[:-1] if key.endswith("_") else key.replace("_", "-")
        rendered.append(f"{name}={quoteattr(str(value))}")
    return " ".join(rendered)


def _tag(name: str, content: str = "", **attrs: object) -> str:
    attributes = _attrs(**attrs)
    opening = f"<{name}{' ' if attributes else ''}{attributes}>"
    return f"{opening}{content}</{name}>"


def _empty_tag(name: str, **attrs: object) -> str:
    attributes = _attrs(**attrs)
    return f"<{name}{' ' if attributes else ''}{attributes}/>"


def _path(points: Iterable[Point]) -> str:
    points = tuple(points)
    if not points:
        return ""
    commands = [f"M {_coord(points[0].x)} {_coord(points[0].y)}"]
    commands.extend(f"L {_coord(point.x)} {_coord(point.y)}" for point in points[1:])
    return " ".join(commands)


def _all_nodes(model: Model) -> tuple[str, ...]:
    return tuple(
        sorted({model.ground, *(node for component in model.components for node in (component.p, component.n))})
    )


def _adjacency(model: Model) -> dict[str, tuple[tuple[str, str], ...]]:
    neighbours: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for component in sorted(model.components, key=lambda item: item.id):
        neighbours[component.p].append((component.id, component.n))
        neighbours[component.n].append((component.id, component.p))
    return {
        node: tuple(sorted(entries, key=lambda entry: (entry[0], entry[1])))
        for node, entries in neighbours.items()
    }


def _collapsed_edges(model: Model) -> tuple[tuple[str, str], ...]:
    """One directed edge per distinct node pair, oriented deterministically.

    Antiparallel devices (a switch and its body diode) share a node pair and
    point opposite ways, so a plain majority ties and the drawing direction
    would be decided by a name. Instead the pair's direction comes from its
    most defining member: sources first, then controlled switches, then
    passives, and only then diodes — a freewheeling diode is a return path,
    not the direction of the main chain. A rectifier, whose groups contain
    only diodes, is still oriented by its diodes. This is a drawing decision;
    the electrical graph is never modified.
    """

    members: dict[tuple[str, str], list[Component]] = defaultdict(list)
    for component in sorted(model.components, key=lambda item: item.id):
        if component.p == component.n:
            continue
        key = (min(component.p, component.n), max(component.p, component.n))
        members[key].append(component)

    edges: list[tuple[str, str, int]] = []
    for key in sorted(members):
        group = members[key]
        leading = min(_DIRECTION_PRIORITY.get(item.type, 5) for item in group)
        deciding = [item for item in group if _DIRECTION_PRIORITY.get(item.type, 5) == leading]
        vote = sum(1 if item.p == key[0] else -1 for item in deciding)
        if vote > 0:
            edges.append((*key, leading))
        elif vote < 0:
            edges.append((key[1], key[0], leading))
        else:
            edges.append((deciding[0].p, deciding[0].n, leading))
    return tuple(edges)


def _acyclic_edges(
    nodes: tuple[str, ...], edges: tuple[tuple[str, str, int], ...]
) -> tuple[tuple[str, str], ...]:
    """Keep the most meaningful orientations and drop what closes a cycle.

    Converter legs plus their return paths are genuinely cyclic (a leg
    descends while a clamp or a freewheeling diode climbs back), so some edge
    must stop constraining the vertical order. WHICH one decides whether the
    drawing reads correctly: cutting the source branch of a buck would float
    ground above the switching node. Edges are therefore admitted in order of
    how much their direction means — sources, then controlled switches, then
    passives, then diodes — and an edge is dropped only when the orientations
    already admitted make it a cycle. The dropped edges are still drawn as
    ordinary wires; only their endpoints stop constraining the layering.
    """

    kept: list[tuple[str, str]] = []
    reachable: dict[str, set[str]] = {node: {node} for node in nodes}
    for source, target, _priority in sorted(
        edges, key=lambda edge: (edge[2], edge[0], edge[1])
    ):
        if source in reachable.get(target, ()):
            continue  # admitting this edge would close a cycle
        kept.append((source, target))
        # Transitive closure update: everything that reaches ``source`` now
        # also reaches everything ``target`` reaches.
        target_closure = reachable.get(target, {target})
        for node, seen in reachable.items():
            if source in seen:
                seen.update(target_closure)
    return tuple(kept)


def _potential_ranks(
    nodes: tuple[str, ...], acyclic: tuple[tuple[str, str], ...]
) -> dict[str, int]:
    """Longest-path layering: every edge descends at least one layer.

    The vertical axis of a power schematic is the rail order, not the hop
    distance from ground: a series chain (rail, devices, pole, devices, rail)
    must read as one column. Longest-path layering over the ``p -> n``
    direction produces exactly that, for any topology, without recognizing
    named converter families.
    """

    successors: dict[str, list[str]] = defaultdict(list)
    indegree = dict.fromkeys(nodes, 0)
    for source, target in acyclic:
        successors[source].append(target)
        indegree[target] = indegree.get(target, 0) + 1
    rank = dict.fromkeys(nodes, 0)
    ready = [node for node in nodes if indegree.get(node, 0) == 0]
    heapq.heapify(ready)
    while ready:
        node = heapq.heappop(ready)
        for target in sorted(successors.get(node, ())):
            rank[target] = max(rank[target], rank[node] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    return rank


def _balanced_ranks(
    nodes: tuple[str, ...],
    acyclic: tuple[tuple[str, str], ...],
    rank: dict[str, int],
) -> dict[str, int]:
    """Spread layering slack so parallel chains span the same height.

    Longest-path layering places every node as high as its longest incoming
    path allows, which squeezes a short chain (a four-tap DC link) against the
    top while a long one (a leg with eight devices) uses the full height. Each
    node with slack is pulled toward the mean rank of its neighbours, clamped
    to the interval its own neighbours currently allow, so every edge keeps
    descending at all times and the result stays a valid layering.
    """

    successors: dict[str, list[str]] = defaultdict(list)
    predecessors: dict[str, list[str]] = defaultdict(list)
    for source, target in acyclic:
        successors[source].append(target)
        predecessors[target].append(source)
    bottom = max(rank.values(), default=0)

    balanced = dict(rank)
    for _ in range(8):
        moved = False
        for node in sorted(nodes):
            incoming = predecessors.get(node, ())
            outgoing = successors.get(node, ())
            if not incoming and not outgoing:
                continue
            lower = max((balanced[item] for item in incoming), default=-1) + 1
            upper = min((balanced[item] for item in outgoing), default=bottom + 1) - 1
            if upper <= lower:
                continue
            related = [balanced[item] for item in (*incoming, *outgoing)]
            target = int(round(sum(related) / len(related)))
            placed = min(max(target, lower), upper)
            if placed != balanced[node]:
                balanced[node] = placed
                moved = True
        if not moved:
            break
    return balanced


def _ordered_columns(
    rank: dict[str, int],
    edges: tuple[tuple[str, str], ...],
    group: frozenset[str],
) -> dict[str, list[str]]:
    """Barycenter ordering inside every layer (crossing reduction)."""

    layers: dict[int, list[str]] = defaultdict(list)
    for node in sorted(group):
        layers[rank[node]].append(node)

    above: dict[str, list[str]] = defaultdict(list)
    below: dict[str, list[str]] = defaultdict(list)
    for source, target in edges:
        if source not in group or target not in group:
            continue
        if rank[source] < rank[target]:
            above[target].append(source)
            below[source].append(target)
        elif rank[target] < rank[source]:
            above[source].append(target)
            below[target].append(source)

    ordinals = {
        node: index for layer in layers.values() for index, node in enumerate(layer)
    }
    ranks = sorted(layers)
    for _ in range(4):
        for sweep, neighbours in ((ranks[1:], above), (ranks[-2::-1], below)):
            for layer_rank in sweep:
                layer = layers[layer_rank]

                def barycenter(node: str, *, source=neighbours) -> float:
                    related = [
                        ordinals[item] for item in source.get(node, ()) if item in ordinals
                    ]
                    return (
                        sum(related) / len(related) if related else float(ordinals[node])
                    )

                layer.sort(key=lambda node: (barycenter(node), ordinals[node], node))
                for index, node in enumerate(layer):
                    ordinals[node] = index
    return {layer_rank: layers[layer_rank] for layer_rank in ranks}


def _connected_groups(
    nodes: tuple[str, ...], adjacency: dict[str, tuple[tuple[str, str], ...]], ground: str
) -> tuple[tuple[str, ...], ...]:
    unseen = set(nodes)
    groups: list[tuple[str, ...]] = []
    while unseen:
        seed = ground if ground in unseen else min(unseen)
        queue = deque([seed])
        unseen.remove(seed)
        group: list[str] = []
        while queue:
            node = queue.popleft()
            group.append(node)
            for _, neighbour in adjacency.get(node, ()):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
        groups.append(tuple(sorted(group)))
    groups.sort(key=lambda group: (0 if ground in group else 1, group))
    return tuple(groups)


def _distances(
    group: tuple[str, ...],
    adjacency: dict[str, tuple[tuple[str, str], ...]],
    ground: str,
) -> dict[str, int]:
    root = ground if ground in group else group[0]
    distance = {root: 0}
    queue = deque([root])
    while queue:
        node = queue.popleft()
        for _, neighbour in adjacency.get(node, ()):
            if neighbour in group and neighbour not in distance:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)
    return distance


def _ordered_layer(
    layer: set[str],
    distance: dict[str, int],
    adjacency: dict[str, tuple[tuple[str, str], ...]],
    components: tuple[Component, ...],
) -> tuple[str, ...]:
    source_seeds = [
        component.p
        for component in sorted(components, key=lambda item: item.id)
        if component.type in _SOURCES and component.p in layer
    ]
    seeds = tuple(dict.fromkeys([*source_seeds, *sorted(layer)]))
    unseen = set(layer)
    ordered: list[str] = []
    for seed in seeds:
        if seed not in unseen:
            continue
        queue = deque([seed])
        unseen.remove(seed)
        while queue:
            node = queue.popleft()
            ordered.append(node)
            same_layer = [
                (component_id, neighbour)
                for component_id, neighbour in adjacency.get(node, ())
                if neighbour in unseen and distance[neighbour] == distance[node]
            ]
            for _, neighbour in sorted(same_layer):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
    return tuple(ordered)


def _assign_columns(
    ordered_layers: dict[int, list[str]],
    edges: tuple[tuple[str, str], ...],
    rank: dict[str, int],
    group: frozenset[str],
    gap: float,
) -> dict[str, float]:
    """Place nodes horizontally so a series chain becomes one straight column.

    Ordering alone still lets a chain zig-zag when neighbouring layers have
    different widths. Alternating sweeps pull every node toward the mean
    abscissa of its neighbours in the adjacent layer and then re-establish the
    minimum gap in the fixed left-to-right order, which straightens chains
    without ever reordering (and therefore without creating crossings).
    """

    neighbours: dict[str, list[str]] = defaultdict(list)
    for source, target in edges:
        if source in group and target in group and rank[source] != rank[target]:
            neighbours[source].append(target)
            neighbours[target].append(source)

    x = {
        node: float(index) * gap
        for layer in ordered_layers.values()
        for index, node in enumerate(layer)
    }
    ranks = sorted(ordered_layers)
    for _ in range(6):
        for sweep, delta in ((ranks[1:], -1), (ranks[-2::-1], 1)):
            for layer_rank in sweep:
                layer = ordered_layers[layer_rank]
                desired: list[float] = []
                for node in layer:
                    related = [
                        x[item]
                        for item in neighbours.get(node, ())
                        if rank[item] == layer_rank + delta
                    ]
                    desired.append(
                        sum(related) / len(related) if related else x[node]
                    )
                cursor = -math.inf
                for node, target in zip(layer, desired):
                    cursor = max(target, cursor + gap)
                    x[node] = cursor
    return x


def _net_positions(model: Model, options: RenderOptions) -> tuple[dict[str, Point], list[str]]:
    nodes = _all_nodes(model)
    adjacency = _adjacency(model)
    groups = _connected_groups(nodes, adjacency, model.ground)
    warnings: list[str] = []
    if len(groups) > 1:
        warnings.append(f"Circuit contains {len(groups)} disconnected net groups")

    weighted_edges = _collapsed_edges(model)
    edges = tuple((source, target) for source, target, _ in weighted_edges)
    acyclic = _acyclic_edges(nodes, weighted_edges)
    rank = _balanced_ranks(nodes, acyclic, _potential_ranks(nodes, acyclic))
    global_max_rank = max(rank.values(), default=0)

    # Columns must be far enough apart to host the widest parallel group that
    # sits between them, otherwise neighbouring branches collide and the
    # fallback scatters components out of their column.
    composites = _composite_partners(model)
    parallel_counts: dict[tuple[str, str], int] = defaultdict(int)
    for component in model.components:
        if component.id in composites:
            continue  # a composite occupies a single lane
        parallel_counts[(min(component.p, component.n), max(component.p, component.n))] += 1
    for component in model.components:
        if component.id in composites:
            key = (min(component.p, component.n), max(component.p, component.n))
            parallel_counts[key] = max(parallel_counts[key], 1)
    widest_group = max(parallel_counts.values(), default=1)
    column_gap = max(
        options.node_gap, widest_group * options.component_lane_gap + 40.0
    )

    positions: dict[str, Point] = {}
    cursor_x = 0.0
    for group in groups:
        members = frozenset(group)
        ordered_layers = _ordered_columns(rank, edges, members)
        columns = _assign_columns(
            ordered_layers, edges, rank, members, column_gap
        )
        origin = min(columns.values(), default=0.0)
        for layer_rank, layer in ordered_layers.items():
            # Rank 0 is the highest potential in the drawing, so it is drawn at
            # the top; the ground rail settles at the bottom by construction.
            y = float(layer_rank) * options.layer_gap
            for node in layer:
                positions[node] = Point(cursor_x + columns[node] - origin, y)
        group_width = max(columns.values(), default=0.0) - origin
        cursor_x += group_width + 2.0 * column_gap

    grid = 10.0
    positions = {
        name: Point(round(point.x / grid) * grid, round(point.y / grid) * grid)
        for name, point in positions.items()
    }

    if global_max_rank == 0 and len(nodes) > 1:
        warnings.append("Layered layout collapsed to a single rank")
    return positions, warnings


def _composite_partners(model: Model) -> dict[str, str]:
    """Map each member of an antiparallel switch/diode pair to its partner.

    A controlled switch and the diode across the same two terminals in the
    opposite sense are one physical device in every datasheet and every
    published figure. Drawing them as two separate symbols with two labels
    doubles the visual element count for no information, so they are placed
    as one composite and labelled once.
    """

    groups: dict[tuple[str, str], list[Component]] = defaultdict(list)
    for component in sorted(model.components, key=lambda item: item.id):
        groups[(min(component.p, component.n), max(component.p, component.n))].append(
            component
        )
    partners: dict[str, str] = {}
    for members in groups.values():
        if len(members) != 2:
            continue
        first, second = members
        if {first.type, second.type} != {"ideal_switch", "diode"}:
            continue
        if first.p != second.n or first.n != second.p:
            continue  # parallel, not antiparallel: keep them side by side
        partners[first.id] = second.id
        partners[second.id] = first.id
    return partners


def _parallel_offsets(model: Model, options: RenderOptions) -> dict[str, float]:
    composites = _composite_partners(model)
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for component in model.components:
        groups[tuple(sorted((component.p, component.n)))].append(component.id)
    offsets: dict[str, float] = {}
    for component_ids in groups.values():
        ordered = sorted(component_ids)
        if len(ordered) == 2 and ordered[0] in composites:
            # One composite device: a shared centre, no lateral lane.
            offsets.update({component_id: 0.0 for component_id in ordered})
            continue
        center = (len(ordered) - 1) / 2.0
        for index, component_id in enumerate(ordered):
            offsets[component_id] = (index - center) * options.component_lane_gap
    return offsets


def _boxes_overlap(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> bool:
    padding = 5.0
    return not (
        left[2] + padding <= right[0]
        or right[2] + padding <= left[0]
        or left[3] + padding <= right[1]
        or right[3] + padding <= left[1]
    )


def _body_box(center: Point, orientation: str) -> tuple[float, float, float, float]:
    # Include the two text rows as well as the electrical symbol.  This keeps a
    # valid but dense topology from producing labels that visually merge into a
    # different component during human review.
    if orientation == "horizontal":
        return center.x - 65.0, center.y - 55.0, center.x + 65.0, center.y + 24.0
    return center.x - 65.0, center.y - 40.0, center.x + 65.0, center.y + 76.0


def _place_components(
    model: Model,
    nets: dict[str, Point],
    options: RenderOptions,
    warnings: list[str],
) -> tuple[_PlacedComponent, ...]:
    offsets = _parallel_offsets(model, options)
    composites = _composite_partners(model)
    composite_centers: dict[str, Point] = {}
    placed: list[_PlacedComponent] = []
    occupied: list[tuple[float, float, float, float]] = []

    # A component bridging non-adjacent layers (the source closing a converter
    # chain, a return path) cannot sit on the column it spans without covering
    # the devices in between, so it gets its own lane beside the drawing —
    # which is how such a branch is drawn by hand anyway.
    # Two layers still belong to the main chain (a leg device whose column
    # happens to have a link tap between its endpoints); only a branch
    # reaching further is drawn beside the circuit.
    bypass_span = 2.5 * options.layer_gap
    right_edge = max((point.x for point in nets.values()), default=0.0)
    bypass_x: dict[str, float] = {}
    lane_bottoms: list[float] = []

    def claim_lane(top: float, bottom: float) -> float:
        # Lanes are shared by components whose vertical extents do not
        # overlap, so a tall circuit does not grow one column per branch.
        lane = next(
            (index for index, used in enumerate(lane_bottoms) if used < top - 24.0),
            len(lane_bottoms),
        )
        if lane == len(lane_bottoms):
            lane_bottoms.append(bottom)
        else:
            lane_bottoms[lane] = bottom
        return right_edge + (lane + 1) * options.component_lane_gap

    spanning = [
        (min(nets[item.p].y, nets[item.n].y), max(nets[item.p].y, nets[item.n].y), item.id)
        for item in model.components
        if abs(nets[item.p].y - nets[item.n].y) > bypass_span
    ]
    for top, bottom, component_id in sorted(spanning):
        bypass_x[component_id] = claim_lane(top, bottom)

    for component in sorted(model.components, key=lambda item: item.id):
        p = nets[component.p]
        n = nets[component.n]
        orientation = "horizontal" if abs(p.y - n.y) < 0.001 else "vertical"
        reversed_direction = p.x > n.x if orientation == "horizontal" else p.y > n.y
        center = Point((p.x + n.x) / 2.0, (p.y + n.y) / 2.0)
        if orientation == "vertical" and (component.p == model.ground) != (
            component.n == model.ground
        ):
            # Ground branches read most naturally directly below their live net.
            # Midpoint-x routing creates long buses that cross unrelated vertical
            # branches when the ground node is centered under a wide upper layer.
            live_node = component.n if component.p == model.ground else component.p
            center = Point(nets[live_node].x, center.y)
        if component.id in bypass_x:
            center = Point(bypass_x[component.id], center.y)
        parallel_offset = offsets[component.id]
        if orientation == "horizontal":
            center = Point(center.x, center.y + parallel_offset)
            candidates = [(0.0, 0.0), (-48.0, 0.0), (48.0, 0.0)]
            candidates.extend((0.0, direction * step) for step in (48.0, 96.0) for direction in (-1, 1))
        else:
            center = Point(center.x + parallel_offset, center.y)
            candidates = [(0.0, 0.0), (0.0, -48.0), (0.0, 48.0)]
            candidates.extend((direction * step, 0.0) for step in (48.0, 96.0) for direction in (-1, 1))

        partner = composites.get(component.id)
        lateral = -_COMPOSITE_LATERAL if partner is not None else 0.0
        if partner is not None and partner in composite_centers:
            # The partner already claimed the space: share its centre so the
            # two bodies form one composite device instead of competing for
            # room and being pushed apart by overlap avoidance.
            placed.append(
                _PlacedComponent(
                    component,
                    composite_centers[partner],
                    orientation,
                    reversed_direction,
                    lateral=lateral,
                    labelled=component.type == "ideal_switch",
                )
            )
            continue

        chosen: Point | None = None
        for dx, dy in candidates:
            candidate = Point(center.x + dx, center.y + dy)
            box = _body_box(candidate, orientation)
            if not any(_boxes_overlap(box, other) for other in occupied):
                chosen = candidate
                occupied.append(box)
                break
        if chosen is None:
            # No room on the column: this branch joins the same packed side
            # lanes the long branches use. That is a layout decision, not a
            # failure, so it is only reported when even a fresh lane clashes.
            lane_x = claim_lane(
                min(p.y, n.y) if component.id not in bypass_x else center.y - 1.0,
                max(p.y, n.y),
            )
            candidate = Point(lane_x, center.y)
            box = _body_box(candidate, orientation)
            if not any(_boxes_overlap(box, other) for other in occupied):
                chosen = candidate
                occupied.append(box)
            else:
                step = 1
                while chosen is None:
                    candidate = Point(
                        lane_x + step * options.component_lane_gap, center.y
                    )
                    box = _body_box(candidate, orientation)
                    if not any(_boxes_overlap(box, other) for other in occupied):
                        chosen = candidate
                        occupied.append(box)
                    step += 1
                warnings.append(
                    f"Component {component.id!r} required an extended layout lane"
                )
        if partner is not None:
            composite_centers[component.id] = chosen
        placed.append(
            _PlacedComponent(
                component,
                chosen,
                orientation,
                reversed_direction,
                lateral=lateral,
                labelled=partner is None or component.type == "ideal_switch",
            )
        )
    return tuple(placed)


def _pins(placed: _PlacedComponent) -> tuple[Point, Point]:
    center = placed.center
    if placed.orientation == "horizontal":
        first = Point(center.x - 36.0, center.y)
        second = Point(center.x + 36.0, center.y)
    else:
        first = Point(center.x, center.y - 36.0)
        second = Point(center.x, center.y + 36.0)
    return (second, first) if placed.reversed else (first, second)


def _rotation(placed: _PlacedComponent) -> int:
    if placed.orientation == "horizontal":
        return 180 if placed.reversed else 0
    return -90 if placed.reversed else 90


def _shift_point(point: Point, dx: float, dy: float) -> Point:
    return Point(point.x + dx, point.y + dy)


def _translated_layout(
    nets: dict[str, Point],
    components: tuple[_PlacedComponent, ...],
    options: RenderOptions,
) -> tuple[dict[str, Point], tuple[_PlacedComponent, ...], int, int]:
    all_x = [point.x for point in nets.values()]
    all_y = [point.y for point in nets.values()]
    for placed in components:
        all_x.extend((placed.center.x - 80.0, placed.center.x + 80.0))
        all_y.extend((placed.center.y - 55.0, placed.center.y + 72.0))
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    title_height = 44.0
    dx = options.margin - min_x
    dy = options.margin + title_height - min_y
    translated_nets = {name: _shift_point(point, dx, dy) for name, point in nets.items()}
    translated_components = tuple(
        _PlacedComponent(
            placed.component,
            _shift_point(placed.center, dx, dy),
            placed.orientation,
            placed.reversed,
            lateral=placed.lateral,
            labelled=placed.labelled,
            label_side=placed.label_side,
        )
        for placed in components
    )
    width = int(math.ceil(max_x + dx + options.margin))
    height = int(math.ceil(max_y + dy + options.margin))
    return translated_nets, translated_components, width, height


def _engineering_value(component: Component) -> str:
    if component.value is None:
        if component.waveform is not None:
            return f"{component.waveform.kind} waveform"
        if component.type == "ideal_switch":
            return f"gate: {component.control}"
        if component.type == "diode":
            return "conducting" if component.initial else "blocking"
        if component.type == "thyristor":
            return f"gate: {component.control}" if component.control else "no gate"
        if component.type == "voltage_probe":
            return "voltage probe"
        return ""
    value = component.value
    if value == 0.0:
        exponent = 0
    else:
        exponent = int(math.floor(math.log10(abs(value)) / 3.0) * 3)
        exponent = min(9, max(-12, exponent))
    scaled = value / (10.0**exponent)
    number = f"{scaled:.6g}"
    unit = _UNIT_BY_TYPE[component.type]
    return f"{number} {_PREFIXES[exponent]}{unit}"


def _wire_route(
    node: Point,
    pin: Point,
    orientation: str,
    lane_index: int,
) -> tuple[Point, ...]:
    if node == pin:
        return (node,)
    lane_offset = 20.0 + lane_index * 7.0
    if orientation == "vertical":
        if abs(node.x - pin.x) < 0.001:
            return node, pin
        direction = 1.0 if pin.y > node.y else -1.0
        maximum = max(7.0, abs(pin.y - node.y) - 7.0)
        lane_y = node.y + direction * min(lane_offset, maximum)
        return node, Point(node.x, lane_y), Point(pin.x, lane_y), pin
    if abs(node.y - pin.y) < 0.001:
        if abs(node.x - pin.x) <= 80.0:
            return node, pin
        lane_y = node.y + lane_offset
        return node, Point(node.x, lane_y), Point(pin.x, lane_y), pin
    direction = 1.0 if pin.x > node.x else -1.0
    maximum = max(7.0, abs(pin.x - node.x) - 7.0)
    lane_x = node.x + direction * min(lane_offset, maximum)
    return node, Point(lane_x, node.y), Point(lane_x, pin.y), pin


def _routing_lanes(nets: dict[str, Point]) -> dict[str, int]:
    layers: dict[float, list[str]] = defaultdict(list)
    for name, point in nets.items():
        layers[round(point.y, 6)].append(name)
    return {
        name: index
        for names in layers.values()
        for index, name in enumerate(sorted(names))
    }


def _symbol_body(component: Component) -> str:
    common = _attrs(class_="symbol-stroke", vector_effect="non-scaling-stroke")
    if component.type == "resistor":
        return (
            f'<path {common} d="M -36 0 L -24 0 L -19 -9 L -9 9 L 1 -9 '
            'L 11 9 L 21 -9 L 26 0 L 36 0"/>'
        )
    if component.type == "capacitor":
        return (
            f'<path {common} d="M -36 0 L -7 0 M -7 -16 L -7 16 '
            'M 7 -16 L 7 16 M 7 0 L 36 0"/>'
        )
    if component.type in {"inductor", "sat_inductor"}:
        winding = (
            f'<path {common} d="M -36 0 L -24 0 '
            'C -24 -13 -12 -13 -12 0 C -12 -13 0 -13 0 0 '
            'C 0 -13 12 -13 12 0 C 12 -13 24 -13 24 0 L 36 0"/>'
        )
        if component.type == "sat_inductor":
            # Core bar marks the saturable magnetic path.
            winding += f'<path {common} d="M -24 -16 L 24 -16"/>'
        return winding
    if component.type == "voltage_source":
        return (
            f'<path {common} d="M -36 0 L -18 0 M 18 0 L 36 0"/>'
            f'<circle {common} cx="0" cy="0" r="18"/>'
            f'<path {common} d="M -9 -5 L -9 5 M -14 0 L -4 0 M 5 0 L 14 0"/>'
        )
    if component.type == "current_source":
        return (
            f'<path {common} d="M -36 0 L -18 0 M 18 0 L 36 0"/>'
            f'<circle {common} cx="0" cy="0" r="18"/>'
            f'<path {common} d="M -10 0 L 10 0 M 10 0 L 3 -5 M 10 0 L 3 5"/>'
        )
    if component.type == "ideal_switch":
        # This is a controlled-switch symbol, not a claim about gate state at
        # simulation start. Gate state is determined by the scheduler.
        blade = "M -18 0 L 15 -13"
        return (
            f'<path {common} d="M -36 0 L -18 0 M 18 0 L 36 0 {blade}"/>'
            f'<circle class="terminal" cx="-18" cy="0" r="2.8"/>'
            f'<circle class="terminal" cx="18" cy="0" r="2.8"/>'
        )
    if component.type == "diode":
        return (
            f'<path {common} d="M -36 0 L -15 0 M 12 0 L 36 0 M 12 -15 L 12 15"/>'
            '<path class="diode-fill" vector-effect="non-scaling-stroke" d="M -15 -14 L 10 0 L -15 14 Z"/>'
        )
    if component.type == "thyristor":
        # A diode with a gate lead off the cathode side.
        return (
            f'<path {common} d="M -36 0 L -15 0 M 12 0 L 36 0 M 12 -15 L 12 15 '
            'M 12 8 L 22 20"/>'
            '<path class="diode-fill" vector-effect="non-scaling-stroke" d="M -15 -14 L 10 0 L -15 14 Z"/>'
        )
    if component.type == "voltage_probe":
        return (
            f'<path {common} d="M -36 0 L -18 0 M 18 0 L 36 0"/>'
            f'<circle {common} cx="0" cy="0" r="18"/>'
            '<text class="probe-letter" text-anchor="middle" x="0" y="5">V</text>'
        )
    raise ValueError(f"Cannot render unsupported component type {component.type!r}")


def _render_component(
    placed: _PlacedComponent,
    options: RenderOptions,
    composite_label: str | None = None,
) -> str:
    component = placed.component
    svg_id = _svg_id("component", component.id)
    rotation_degrees = _rotation(placed)
    rotation = f" rotate({rotation_degrees})" if rotation_degrees else ""
    body = _symbol_body(component)
    if placed.lateral:
        stroke = _attrs(class_="symbol-stroke", vector_effect="non-scaling-stroke")
        lateral = _coord(placed.lateral)
        body = (
            _tag("g", body, transform=f"translate(0 {lateral})")
            # Short jogs from the shared pins onto the offset body: the two
            # members of a composite meet at the same two terminals.
            + f'<path {stroke} d="M -36 0 L -36 {lateral} M 36 0 L 36 {lateral}"/>'
        )
    symbol = _tag(
        "g",
        body,
        class_="component-symbol",
        transform=f"translate({_coord(placed.center.x)} {_coord(placed.center.y)}){rotation}",
    )
    label_y = (
        placed.center.y + 49.0
        if placed.orientation == "vertical"
        else placed.center.y - 36.0
    )
    label_x = placed.center.x
    anchor = "middle"
    if placed.label_side == "right":
        label_x = placed.center.x + 28.0 + abs(placed.lateral)
        label_y = placed.center.y + 4.0
        anchor = "start"
    elif placed.label_side == "probe":
        label_x = placed.center.x + 48.0
        label_y = placed.center.y + 4.0
        anchor = "start"
    label = ""
    value = ""
    if placed.labelled:
        label = _tag(
            "text",
            escape(composite_label or component.id),
            class_="component-label",
            id=f"{svg_id}-label",
            text_anchor=anchor,
            x=_coord(label_x),
            y=_coord(label_y),
        )
        # Switching devices carry their gate binding and initial state in the
        # legend and the tooltip; repeating them under every symbol is what
        # turned a dense bridge into a wall of text.
        if options.show_values and component.type not in {"ideal_switch", "diode", "thyristor"}:
            value = _tag(
                "text",
                escape(_engineering_value(component)),
                class_="component-value",
                text_anchor=anchor,
                x=_coord(label_x),
                y=_coord(label_y + 15.0),
            )
    title = _tag(
        "title",
        escape(
            f"{component.id}: {component.type}; p={component.p}; n={component.n}; "
            f"{_engineering_value(component)}"
        ),
    )
    identification = (
        {"aria_labelledby": f"{svg_id}-label"}
        if placed.labelled
        else {"aria_label": f"{component.id} ({component.type})"}
    )
    return _tag(
        "g",
        title + symbol + label + value,
        class_="component",
        data_component_id=component.id,
        data_component_type=component.type,
        data_n=component.n,
        data_p=component.p,
        id=svg_id,
        **identification,
    )


def _legend(model: Model, options: RenderOptions, origin: Point) -> str:
    """Device bindings, off the drawing: gate map first, then diode states."""

    if not options.show_values:
        return ""
    rows: list[str] = []
    switches = [
        item for item in model.components
        if item.type == "ideal_switch" or (item.type == "thyristor" and item.control)
    ]
    for component in sorted(switches, key=lambda item: item.id):
        rows.append(f"{component.id} ← {component.control}")
    if not rows:
        return ""
    diodes = sum(1 for item in model.components if item.type == "diode")
    lines = [
        _tag(
            "text",
            escape("Gate bindings"),
            class_="legend-title",
            x=_coord(origin.x),
            y=_coord(origin.y),
        )
    ]
    columns = 1 + (len(rows) - 1) // 12
    per_column = math.ceil(len(rows) / columns)
    for index, row in enumerate(rows):
        column, offset = divmod(index, per_column)
        lines.append(
            _tag(
                "text",
                escape(row),
                class_="legend-row",
                x=_coord(origin.x + column * 210.0),
                y=_coord(origin.y + 20.0 + offset * 15.0),
            )
        )
    if diodes:
        lines.append(
            _tag(
                "text",
                escape(
                    f"{diodes} diode(s): initial state and terminals in the "
                    "element tooltip; a device drawn beside a switch is its "
                    "antiparallel diode."
                ),
                class_="legend-row",
                x=_coord(origin.x),
                y=_coord(origin.y + 28.0 + per_column * 15.0),
            )
        )
    return _tag("g", "".join(lines), class_="legend", id="schematic-legend")


def _render_net(
    name: str,
    point: Point,
    model: Model,
    components: dict[str, _PlacedComponent],
    lane_index: int,
) -> str:
    svg_id = _svg_id("net", name)
    branches: list[str] = []
    for component in sorted(model.components, key=lambda item: item.id):
        terminal: str | None = None
        if component.p == name:
            terminal = "p"
        elif component.n == name:
            terminal = "n"
        if terminal is None:
            continue
        placed = components[component.id]
        p_pin, n_pin = _pins(placed)
        pin = p_pin if terminal == "p" else n_pin
        route = _wire_route(point, pin, placed.orientation, lane_index)
        branch_id = f"{svg_id}-to-{_svg_id('component', component.id)}-{terminal}"
        branches.append(
            _empty_tag(
                "path",
                class_="wire",
                d=_path(route),
                id=branch_id,
                vector_effect="non-scaling-stroke",
            )
        )

    node_marker = _empty_tag(
        "circle",
        class_="junction ground-junction" if name == model.ground else "junction",
        cx=_coord(point.x),
        cy=_coord(point.y),
        r="4",
    )
    if name == model.ground:
        ground = (
            _empty_tag(
                "path",
                class_="ground-symbol",
                d=(
                    f"M {_coord(point.x)} {_coord(point.y + 4)} "
                    f"L {_coord(point.x)} {_coord(point.y + 13)} "
                    f"M {_coord(point.x - 13)} {_coord(point.y + 13)} "
                    f"L {_coord(point.x + 13)} {_coord(point.y + 13)} "
                    f"M {_coord(point.x - 8)} {_coord(point.y + 18)} "
                    f"L {_coord(point.x + 8)} {_coord(point.y + 18)} "
                    f"M {_coord(point.x - 3)} {_coord(point.y + 23)} "
                    f"L {_coord(point.x + 3)} {_coord(point.y + 23)}"
                ),
                vector_effect="non-scaling-stroke",
            )
            + _tag(
                "text",
                escape(name),
                class_="net-label",
                x=_coord(point.x + 18),
                y=_coord(point.y + 24),
            )
        )
        node_content = node_marker + ground
    else:
        label = _tag(
            "text",
            escape(name),
            class_="net-label",
            x=_coord(point.x + 8),
            y=_coord(point.y - 10),
        )
        node_content = node_marker + label
    title = _tag("title", escape(f"Net: {name}"))
    return _tag(
        "g",
        title + "".join(branches) + node_content,
        class_="net",
        data_net_name=name,
        id=svg_id,
    )


_STYLE = """
:root { color-scheme: light; }
.canvas { fill: #fbfcfe; }
.title { fill: #162033; font: 600 18px 'DejaVu Sans', sans-serif; }
.subtitle { fill: #58667a; font: 12px 'DejaVu Sans', sans-serif; }
.layout-warning { fill: #9a3412; font: 600 11px 'DejaVu Sans', sans-serif; }
.wire, .ground-symbol { fill: none; stroke: #334155; stroke-linecap: round;
  stroke-linejoin: round; stroke-width: 1.25; }
.junction { fill: #334155; stroke: #fbfcfe; stroke-width: 1; }
.ground-junction { fill: #0f766e; }
.net-label { fill: #0f766e; font: 600 12px 'DejaVu Sans', sans-serif;
  paint-order: stroke; stroke: #fbfcfe; stroke-width: 4px; stroke-linejoin: round; }
.component-symbol { fill: #fbfcfe; }
.symbol-stroke { fill: none; stroke: #0f172a; stroke-linecap: round;
  stroke-linejoin: round; stroke-width: 1.25; }
.terminal { fill: #0f172a; }
.diode-fill { fill: #fbfcfe; stroke: #0f172a; stroke-linejoin: round; stroke-width: 1.25; }
.probe-letter { fill: #0f172a; font: 700 13px 'DejaVu Sans', sans-serif; }
.component-label { fill: #172033; font: 700 12px 'DejaVu Sans', sans-serif;
  paint-order: stroke; stroke: #fbfcfe; stroke-width: 4px; stroke-linejoin: round; }
.component-value { fill: #58667a; font: 11px 'DejaVu Sans', sans-serif;
  paint-order: stroke; stroke: #fbfcfe; stroke-width: 4px; stroke-linejoin: round; }
.component:focus .symbol-stroke, .component:hover .symbol-stroke { stroke: #2563eb; }
.legend-title { fill: #162033; font: 700 12px 'DejaVu Sans', sans-serif; }
.legend-row { fill: #58667a; font: 11px 'DejaVu Sans', sans-serif; }
.subcircuit-box { fill: #eef2f7; stroke: #0f172a; stroke-width: 2; }
.subcircuit-ref { fill: #58667a; font: italic 10px 'DejaVu Sans', sans-serif; }
.subcircuit-port { fill: #0f172a; }
.subcircuit-port-label { fill: #334155; font: 600 10px 'DejaVu Sans', sans-serif; }
""".strip()


def _subcircuit_link_components(
    instances: tuple[SubcircuitInstance, ...],
) -> tuple[Component, ...]:
    """Invisible two-pin chains that keep a box's bound nets connected.

    The layered layout only sees two-pin components; without these links a
    net reached exclusively through a collapsed instance would fall into a
    separate group and drift away from the box.  The links join the bound
    nets in declared port order, participate in layout only, and are never
    placed or drawn.
    """

    links: list[Component] = []
    for instance in sorted(instances, key=lambda item: item.path):
        bound: list[str] = []
        for _port, net in instance.ports:
            if net not in bound:
                bound.append(net)
        for index, (first, second) in enumerate(zip(bound, bound[1:])):
            links.append(
                Component(
                    id=f"{instance.path}{SEPARATOR}__layout_link_{index}",
                    type="voltage_probe",
                    p=first,
                    n=second,
                )
            )
    return tuple(links)


def _place_subcircuit_boxes(
    instances: tuple[SubcircuitInstance, ...],
    nets: dict[str, Point],
) -> tuple[SubcircuitBox, ...]:
    """Deterministic box geometry: centred on the bound nets, grid-snapped.

    Ports face the side their net lies on; each side is ordered by the
    net's vertical position, then the port name.  Later boxes shift down
    until they clear earlier ones — a naive but deterministic rule.
    """

    grid = 10.0
    boxes: list[SubcircuitBox] = []
    for instance in sorted(instances, key=lambda item: item.path):
        points = [nets[net] for _port, net in instance.ports]
        centre_x = sum(point.x for point in points) / len(points)
        centre_y = sum(point.y for point in points) / len(points)
        left: list[tuple[str, str]] = []
        right: list[tuple[str, str]] = []
        for port, net in instance.ports:
            (left if nets[net].x <= centre_x else right).append((port, net))
        left.sort(key=lambda item: (nets[item[1]].y, item[0]))
        right.sort(key=lambda item: (nets[item[1]].y, item[0]))
        rows = max(len(left), len(right), 1)
        height = 34.0 + rows * 20.0
        # Wide enough for the ref label (~6 px/char at 10 px italic) plus a
        # port-label column on each side.
        width = max(150.0, 24.0 + 6.0 * len(instance.ref))
        x = max(8.0, round((centre_x - width / 2.0) / grid) * grid)
        y = round((centre_y - height / 2.0) / grid) * grid
        while any(
            x < other.x + other.width
            and other.x < x + width
            and y < other.y + other.height
            and other.y < y + height
            for other in boxes
        ):
            y += grid
        pins: list[SubcircuitPortPin] = []
        for row, (port, net) in enumerate(left):
            pins.append(
                SubcircuitPortPin(
                    name=port, net=net, pin=Point(x, y + 28.0 + row * 20.0)
                )
            )
        for row, (port, net) in enumerate(right):
            pins.append(
                SubcircuitPortPin(
                    name=port,
                    net=net,
                    pin=Point(x + width, y + 28.0 + row * 20.0),
                )
            )
        boxes.append(
            SubcircuitBox(
                instance=instance.path,
                ref=instance.ref,
                svg_id=_svg_id("subcircuit", instance.path),
                x=x,
                y=y,
                width=width,
                height=height,
                ports=tuple(pins),
            )
        )
    return tuple(boxes)


def _render_subcircuit_box(box: SubcircuitBox, nets: dict[str, Point]) -> str:
    parts: list[str] = []
    for port in box.ports:
        target = nets[port.net]
        parts.append(
            _empty_tag(
                "line",
                class_="wire",
                x1=_coord(port.pin.x),
                y1=_coord(port.pin.y),
                x2=_coord(target.x),
                y2=_coord(target.y),
            )
        )
    parts.append(
        _empty_tag(
            "rect",
            class_="subcircuit-box",
            height=_coord(box.height),
            rx="6",
            width=_coord(box.width),
            x=_coord(box.x),
            y=_coord(box.y),
        )
    )
    parts.append(
        _tag(
            "text",
            escape(box.instance),
            class_="component-label",
            text_anchor="middle",
            x=_coord(box.x + box.width / 2.0),
            y=_coord(box.y + 14.0),
        )
    )
    parts.append(
        _tag(
            "text",
            escape(box.ref),
            class_="subcircuit-ref",
            text_anchor="middle",
            x=_coord(box.x + box.width / 2.0),
            y=_coord(box.y + box.height - 6.0),
        )
    )
    for port in box.ports:
        is_left = port.pin.x <= box.x + box.width / 2.0
        parts.append(
            _empty_tag(
                "circle",
                class_="subcircuit-port",
                cx=_coord(port.pin.x),
                cy=_coord(port.pin.y),
                r="3",
            )
        )
        parts.append(
            _tag(
                "text",
                escape(port.name),
                class_="subcircuit-port-label",
                text_anchor="start" if is_left else "end",
                x=_coord(port.pin.x + (7.0 if is_left else -7.0)),
                y=_coord(port.pin.y + 3.5),
            )
        )
    return _tag(
        "g",
        "\n".join(parts),
        class_="subcircuit",
        data_subcircuit_instance=box.instance,
        data_subcircuit_ref=box.ref,
        id=box.svg_id,
    )


def _translated_bridge(bridge, options: RenderOptions):
    """Shift the bridge drawing onto the canvas, below the title block."""

    placed_all = (*bridge.components, *bridge.probes)
    xs = [point.x for point in bridge.anchors.values()]
    ys = [point.y for point in bridge.anchors.values()]
    for items in bridge.wires.values():
        for a, b in items:
            xs.extend((a.x, b.x))
            ys.extend((a.y, b.y))
    for placed in placed_all:
        reach = 260.0 if placed.label_side == "probe" else 150.0
        xs.extend((placed.center.x - 60.0, placed.center.x + reach))
        ys.extend((placed.center.y - 50.0, placed.center.y + 60.0))
    # Link-net labels sit left of the link column.
    xs.append(min(point.x for point in bridge.anchors.values()) - 90.0)
    dx = options.margin - min(xs)
    dy = options.margin + 64.0 - min(ys)
    nets = {name: _shift_point(point, dx, dy) for name, point in bridge.anchors.items()}
    components = tuple(
        replace(placed, center=_shift_point(placed.center, dx, dy)) for placed in placed_all
    )
    wires = {
        name: tuple((_shift_point(a, dx, dy), _shift_point(b, dx, dy)) for a, b in items)
        for name, items in bridge.wires.items()
    }
    width = int(math.ceil(max(xs) + dx + options.margin))
    height = int(math.ceil(max(ys) + dy + options.margin))
    return nets, components, wires, width, height


def _hopped_path(a: Point, b: Point, jumps: list[float]) -> str:
    if not jumps:
        return _path((a, b))
    left, right = (a, b) if a.x < b.x else (b, a)
    y = _coord(left.y)
    parts = [f"M {_coord(left.x)} {y}"]
    for x in jumps:
        parts.append(f"L {_coord(x - 5.0)} {y}")
        parts.append(f"A 5 5 0 0 1 {_coord(x + 5.0)} {y}")
    parts.append(f"L {_coord(right.x)} {y}")
    return " ".join(parts)


def _render_wired_net(
    name: str,
    point: Point,
    segments: tuple[tuple[Point, Point], ...],
    jumps: dict[int, list[float]],
    dots: list[Point],
    *,
    is_ground: bool,
    label_left: bool,
) -> str:
    """One net of the bridge layout: explicit orthogonal wires, hops, junction dots."""

    svg_id = _svg_id("net", name)
    parts = [
        _empty_tag(
            "path",
            class_="wire",
            d=_hopped_path(a, b, jumps.get(index, [])),
            id=f"{svg_id}-w{index}",
            vector_effect="non-scaling-stroke",
        )
        for index, (a, b) in enumerate(segments)
    ]
    parts.extend(
        _empty_tag("circle", class_="junction", cx=_coord(dot.x), cy=_coord(dot.y), r="4")
        for dot in dots
    )
    if is_ground:
        gx, gy = point.x - 40.0, point.y
        parts.append(
            _empty_tag(
                "path",
                class_="ground-symbol",
                d=(
                    f"M {_coord(point.x)} {_coord(gy)} L {_coord(gx)} {_coord(gy)} "
                    f"L {_coord(gx)} {_coord(gy + 13)} "
                    f"M {_coord(gx - 13)} {_coord(gy + 13)} L {_coord(gx + 13)} {_coord(gy + 13)} "
                    f"M {_coord(gx - 8)} {_coord(gy + 18)} L {_coord(gx + 8)} {_coord(gy + 18)} "
                    f"M {_coord(gx - 3)} {_coord(gy + 23)} L {_coord(gx + 3)} {_coord(gy + 23)}"
                ),
                vector_effect="non-scaling-stroke",
            )
        )
    parts.append(
        _tag(
            "text",
            escape(name),
            class_="net-label",
            text_anchor="end" if label_left else "start",
            x=_coord(point.x - 8 if label_left else point.x + 8),
            y=_coord(point.y - 8),
        )
    )
    title = _tag("title", escape(f"Net: {name}"))
    return _tag(
        "g", title + "".join(parts), class_="net", data_net_name=name, id=svg_id
    )


def render_schematic(
    model: Model,
    options: RenderOptions | None = None,
    *,
    expand_subcircuits: bool = False,
) -> RenderResult:
    """Render ``model`` into deterministic standalone SVG plus layout metadata.

    A model with declared subcircuits draws each top-level instance as a box
    with named ports by default; ``expand_subcircuits=True`` draws the
    flattened components instead.  Either way the electrical model is the
    flattened one — the box view only hides members, it never rewires.
    """

    options = options or RenderOptions()
    source_model = model
    top_instances = (
        tuple(
            instance
            for instance in model.subcircuit_instances
            if SEPARATOR not in instance.path
        )
        if not expand_subcircuits
        else ()
    )
    if top_instances:
        hidden = {
            member for instance in top_instances for member in instance.members
        }
        visible = tuple(
            component
            for component in model.components
            if component.id not in hidden
        )
        layout_model = replace(
            model,
            components=(*visible, *_subcircuit_link_components(top_instances)),
        )
        model = replace(model, components=visible)
    else:
        layout_model = model
    bridge = None
    if not top_instances:
        from pear_sim.reporting.render_bridge import bridge_drawing

        bridge = bridge_drawing(model)
    wires: dict[str, tuple[tuple[Point, Point], ...]] = {}
    if bridge is not None:
        warnings: list[str] = []
        nets, components, wires, width, height = _translated_bridge(bridge, options)
    else:
        raw_nets, warnings = _net_positions(layout_model, options)
        raw_components = _place_components(model, raw_nets, options, warnings)
        nets, components, width, height = _translated_layout(
            raw_nets, raw_components, options
        )
    subcircuit_boxes = (
        _place_subcircuit_boxes(top_instances, nets) if top_instances else ()
    )
    for box in subcircuit_boxes:
        width = max(width, int(math.ceil(box.x + box.width + options.margin)))
        height = max(height, int(math.ceil(box.y + box.height + options.margin)))
    drawing_height = height
    switch_count = sum(
        1 for component in model.components if component.type == "ideal_switch"
    )
    if options.show_values and switch_count:
        # Reserve the strip the legend occupies below the drawing.
        legend_columns = 1 + (switch_count - 1) // 12
        legend_rows = math.ceil(switch_count / legend_columns)
        height += int(math.ceil(20.0 + legend_rows * 15.0 + 40.0))
        width = max(width, int(math.ceil(options.margin * 2 + legend_columns * 210.0)))
    components_by_id = {placed.component.id: placed for placed in components}
    routing_lanes = _routing_lanes(nets)

    degree: dict[str, int] = defaultdict(int)
    for component in model.components:
        degree[component.p] += 1
        degree[component.n] += 1

    component_metadata = tuple(
        ComponentPlacement(
            component_id=placed.component.id,
            svg_id=_svg_id("component", placed.component.id),
            component_type=placed.component.type,
            center=placed.center,
            orientation=placed.orientation,
            rotation=_rotation(placed),
            p_pin=_pins(placed)[0],
            n_pin=_pins(placed)[1],
        )
        for placed in components
    )
    net_metadata = tuple(
        NetPlacement(
            name=name,
            svg_id=_svg_id("net", name),
            point=nets[name],
            degree=degree[name],
            is_ground=name == model.ground,
        )
        for name in sorted(nets)
    )
    metadata = RenderMetadata(
        renderer=RENDERER_VERSION,
        width=width,
        height=height,
        model_name=source_model.name,
        model_sha256=model_hashes(source_model)["model_sha256"],
        components=component_metadata,
        nets=net_metadata,
        warnings=tuple(warnings),
    )

    embedded_metadata = json.dumps(
        {
            "component_count": len(component_metadata),
            "model_name": source_model.name,
            "net_count": len(net_metadata),
            "renderer": RENDERER_VERSION,
            "schema": model.schema,
            "warnings": warnings,
            # Present only for the bridge layout, so every other drawing keeps
            # its exact embedded-metadata bytes.
            **({"layout": "bridge"} if bridge is not None else {}),
            # Present only in the collapsed view, so every subcircuit-free
            # model keeps its exact embedded-metadata bytes.
            **(
                {"subcircuit_count": len(subcircuit_boxes)}
                if subcircuit_boxes
                else {}
            ),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    title_id = "schematic-title"
    description_id = "schematic-description"
    body = [
        _tag("title", escape(model.name), id=title_id),
        _tag(
            "desc",
            escape(
                f"PEARSim electrical schematic with {len(component_metadata)} components "
                f"and {len(net_metadata)} nets."
            ),
            id=description_id,
        ),
        _tag("metadata", escape(embedded_metadata), id="pear_sim-render-metadata"),
        _tag("style", escape(_STYLE)),
        _empty_tag("rect", class_="canvas", height="100%", width="100%", x="0", y="0"),
        _tag(
            "text",
            escape(model.name),
            class_="title",
            x=_coord(options.margin),
            y="28",
        ),
        _tag(
            "text",
            escape(f"{len(component_metadata)} components · {len(net_metadata)} nets"),
            class_="subtitle",
            x=_coord(options.margin),
            y="46",
        ),
    ]
    if warnings:
        body.append(
            _tag(
                "text",
                escape(
                    f"Auto-layout warning: {len(warnings)} extended lanes; verify component/net IDs."
                ),
                class_="layout-warning",
                x=_coord(options.margin),
                y="64",
            )
        )
    if bridge is not None:
        from pear_sim.reporting.render_bridge import hops, junctions

        jumps = hops(wires)
        pins_by_net: dict[str, list[Point]] = defaultdict(list)
        for placed in components:
            if placed.label_side == "probe":
                continue
            p_pin, n_pin = _pins(placed)
            pins_by_net[placed.component.p].append(p_pin)
            pins_by_net[placed.component.n].append(n_pin)
        body.extend(
            _render_wired_net(
                name,
                nets[name],
                wires.get(name, ()),
                {index: xs for (net, index), xs in jumps.items() if net == name},
                junctions(wires.get(name, ()), pins_by_net[name]),
                is_ground=name == model.ground,
                label_left=name in bridge.left_labels,
            )
            for name in sorted(nets)
        )
    else:
        body.extend(
            _render_net(
                name,
                nets[name],
                model,
                components_by_id,
                routing_lanes[name],
            )
            for name in sorted(nets)
        )
    composites = _composite_partners(model)
    composite_labels = {
        placed.component.id: f"{placed.component.id}/{composites[placed.component.id]}"
        for placed in components
        if placed.labelled and placed.component.id in composites
    }
    composite_labels.update(
        {
            placed.component.id: (
                f"{placed.component.id} = v({placed.component.p}) − v({placed.component.n})"
            )
            for placed in components
            if placed.label_side == "probe"
        }
    )
    body.extend(
        _render_component(placed, options, composite_labels.get(placed.component.id))
        for placed in components
    )
    body.extend(
        _render_subcircuit_box(box, nets) for box in subcircuit_boxes
    )
    probe_places = [placed for placed in components if placed.label_side == "probe"]
    if probe_places:
        first = min(probe_places, key=lambda placed: (placed.center.y, placed.center.x))
        body.append(
            _tag(
                "text",
                escape("Voltage probes (measuring instruments, not wired in the drawing)"),
                class_="legend-title",
                x=_coord(first.center.x - 40.0),
                y=_coord(first.center.y - 34.0),
            )
        )
    body.append(_legend(model, options, Point(options.margin, drawing_height - 16.0)))

    svg_attrs = _attrs(
        aria_labelledby=f"{title_id} {description_id}",
        height=height,
        role="img",
        version="1.1",
        viewBox=f"0 0 {width} {height}",
        width=width,
        xmlns="http://www.w3.org/2000/svg",
    )
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<svg {svg_attrs}>\n"
        + "\n".join(body)
        + "\n</svg>\n"
    )
    return RenderResult(
        svg=svg, metadata=metadata, subcircuit_boxes=subcircuit_boxes
    )


def render_svg(model: Model, options: RenderOptions | None = None) -> str:
    """Return only the standalone SVG representation of ``model``."""

    return render_schematic(model, options).svg


# The explicit alias reads naturally for callers discovering the renderer next
# to ``load_model`` while preserving the short ``render_svg`` convenience API.
render_model_svg = render_svg


__all__ = [
    "ComponentPlacement",
    "NetPlacement",
    "Point",
    "RENDERER_VERSION",
    "RenderMetadata",
    "RenderOptions",
    "RenderResult",
    "SubcircuitBox",
    "SubcircuitPortPin",
    "render_model_svg",
    "render_schematic",
    "render_svg",
]
