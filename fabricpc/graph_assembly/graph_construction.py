"""Graph builder that assembles nodes and edges into a GraphStructure."""

import types
from dataclasses import replace
from typing import List, Dict, Optional, Tuple
from fabricpc.core.types import GraphStructure, NodeInfo, EdgeInfo, SlotInfo
from fabricpc.core.inference import InferenceBase
from fabricpc.core.mupc import MuPCConfig, compute_mupc_scalings
from fabricpc.core.topology import Edge
from fabricpc.nodes.base import NodeBase
from fabricpc.graph_initialization.state_initializer import (
    StateInitBase,
    FeedforwardStateInit,
)


class TaskMap:
    """
    Maps task names (x, y, etc.) to nodes.
    Accepts node objects or node name strings.
    """

    def __init__(self, **kwargs):
        mapping: Dict[str, str] = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                mapping[key] = value
            else:
                # NodeBase instance
                mapping[key] = value.name
        self._map = types.MappingProxyType(mapping)  # Immutable dictionary

    def to_dict(self) -> Dict[str, str]:
        return dict(self._map)


def _build_slots(node: NodeBase, in_edges: Dict[str, EdgeInfo]) -> Dict[str, SlotInfo]:
    """Build SlotInfo objects from node's slot specs and incoming edges."""
    slot_specs = type(node).get_slots()
    slots = {}

    for slot_name, slot_spec in slot_specs.items():
        # Find source nodes connecting to this slot
        in_neighbors = [e.source for e in in_edges.values() if e.slot == slot_name]

        # Validate single-input constraint
        if not slot_spec.is_multi_input and len(in_neighbors) > 1:
            raise ValueError(
                f"Slot '{slot_name}' in node '{node.name}' is single-input "
                f"but has {len(in_neighbors)} connections"
            )

        slots[slot_name] = SlotInfo(
            name=slot_name,
            parent_node=node.name,
            is_multi_input=slot_spec.is_multi_input,
            is_variance_scalable=slot_spec.is_variance_scalable,
            is_skip_connection=slot_spec.is_skip_connection,
            in_neighbors=tuple(in_neighbors),
        )

    return slots


class GraphCycleError(ValueError):
    """Raised when a graph contains cycles and no unroll degree was given."""


def first_occurrence_order(schedule: Tuple[str, ...]) -> Tuple[str, ...]:
    """Deduplicate a visit schedule, keeping first occurrences (the unique node order)."""
    seen = set()
    order = []
    for name in schedule:
        if name not in seen:
            seen.add(name)
            order.append(name)
    return tuple(order)


def _tarjan_sccs(
    nodes: Dict[str, NodeBase], edges: Dict[str, EdgeInfo]
) -> List[List[str]]:
    """
    Iterative Tarjan strongly-connected-components decomposition.

    Roots iterate in dict order and successors in out_edges order, so the
    result is deterministic for a given node/edge insertion order.
    """
    index_counter = 0
    index: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    on_stack: Dict[str, bool] = {}
    stack: List[str] = []
    sccs: List[List[str]] = []

    for start in nodes:
        if start in index:
            continue
        work = [(start, 0)]  # (node, next-successor pointer)
        while work:
            v, pointer = work[-1]
            if pointer == 0:
                index[v] = index_counter
                lowlink[v] = index_counter
                index_counter += 1
                stack.append(v)
                on_stack[v] = True
            successors = [edges[k].target for k in nodes[v].node_info.out_edges]
            descended = False
            for i in range(pointer, len(successors)):
                w = successors[i]
                if w not in index:
                    work[-1] = (v, i + 1)
                    work.append((w, 0))
                    descended = True
                    break
                if on_stack.get(w, False):
                    lowlink[v] = min(lowlink[v], index[w])
            if descended:
                continue
            if lowlink[v] == index[v]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    scc.append(w)
                    if w == v:
                        break
                sccs.append(scc)
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[v])

    return sccs


def _topological_sort(
    nodes: Dict[str, NodeBase],
    edges: Dict[str, EdgeInfo],
    unroll: Optional[int] = None,
) -> Tuple[str, ...]:
    """
    Node visit schedule for forward traversal (initialization and ePC
    state derivation).

    On a DAG this is the BFS/Kahn topological order, one visit per node,
    whatever ``unroll`` is. On a cyclic graph an explicit ``unroll=U ≥ 1``
    is required: strongly connected components collapse to a condensation
    DAG, Kahn orders the condensation with the same seeding/successor
    rules, and each nontrivial component's members are emitted U times in
    BFS order from the component's entry nodes (members with an in-edge
    from outside the component; the first member in dict order if none).

    Args:
        nodes: Dictionary of NodeBase instances (access in_degree/out_edges via node.node_info)
        edges: Dictionary of EdgeInfo instances
        unroll: Number of traversals of each cycle. None (default) means the
            graph must be acyclic.

    Returns:
        Tuple of node names; cycle members may repeat; every node appears
        at least once.

    Raises:
        GraphCycleError: The graph contains cycles and ``unroll`` is None.
    """
    # Count in-degrees from node.node_info
    in_degree = {name: node.node_info.in_degree for name, node in nodes.items()}

    # Queue of nodes, begin with nodes having no incoming edges
    queue = [name for name, deg in in_degree.items() if deg == 0]
    result = []

    while queue:
        node_name = queue.pop(0)
        result.append(node_name)

        # Reduce in-degree of neighbors
        for out_edge_key in nodes[node_name].node_info.out_edges:
            edge_info = edges[out_edge_key]
            target_name = edge_info.target
            in_degree[target_name] -= 1

            if in_degree[target_name] == 0:
                # Dependencies have been processed, now add next node to the queue
                queue.append(target_name)

    if len(result) == len(nodes):
        # Acyclic: the plain Kahn order, one visit per node.
        return tuple(result)

    ordered = set(result)
    unordered = [name for name in nodes if name not in ordered]
    if unroll is None:
        raise GraphCycleError(
            f"Graph contains cycles; nodes {unordered} cannot be "
            f"topologically ordered. Pass graph(..., unroll=U) with U >= 1 "
            f"to unroll each cycle into U traversals (U=1 visits each cycle "
            f"member once)."
        )

    # Cyclic with an explicit unroll degree: order the condensation DAG.
    sccs = _tarjan_sccs(nodes, edges)
    # Identify each SCC by its first member in dict order; process SCCs in
    # that order for deterministic seeding.
    node_position = {name: i for i, name in enumerate(nodes)}
    for scc in sccs:
        scc.sort(key=lambda n: node_position[n])
    sccs.sort(key=lambda scc: node_position[scc[0]])
    scc_of = {name: i for i, scc in enumerate(sccs) for name in scc}

    # Condensation in-degrees and successor lists (edge multiplicity kept:
    # Kahn decrements once per edge, mirroring the node-level loop).
    cond_in_degree = [0] * len(sccs)
    cond_successors: List[List[int]] = [[] for _ in sccs]
    for scc_id, scc in enumerate(sccs):
        for member in scc:
            for out_edge_key in nodes[member].node_info.out_edges:
                target_scc = scc_of[edges[out_edge_key].target]
                if target_scc != scc_id:
                    cond_successors[scc_id].append(target_scc)
                    cond_in_degree[target_scc] += 1

    def scc_visit_order(scc_id: int) -> List[str]:
        members = sccs[scc_id]
        if len(members) == 1:
            return members
        member_set = set(members)
        entries = [
            name
            for name in members
            if any(
                edges[k].source not in member_set
                for k in nodes[name].node_info.in_edges
            )
        ]
        bfs_queue = entries or [members[0]]
        visited = set(bfs_queue)
        order = []
        while bfs_queue:
            name = bfs_queue.pop(0)
            order.append(name)
            for out_edge_key in nodes[name].node_info.out_edges:
                target = edges[out_edge_key].target
                if target in member_set and target not in visited:
                    visited.add(target)
                    bfs_queue.append(target)
        return order

    scc_queue = [i for i, deg in enumerate(cond_in_degree) if deg == 0]
    schedule: List[str] = []
    while scc_queue:
        scc_id = scc_queue.pop(0)
        repeats = unroll if len(sccs[scc_id]) > 1 else 1
        schedule.extend(scc_visit_order(scc_id) * repeats)
        for target_scc in cond_successors[scc_id]:
            cond_in_degree[target_scc] -= 1
            if cond_in_degree[target_scc] == 0:
                scc_queue.append(target_scc)

    return tuple(schedule)


def graph(
    nodes: List[NodeBase],
    edges: List[Edge],
    task_map: TaskMap,
    inference: InferenceBase,
    graph_state_initializer: Optional[StateInitBase] = None,
    scaling=None,
    unroll: Optional[int] = None,
) -> GraphStructure:
    """
    Build a GraphStructure from node objects, edge objects, and a task map.

    This is the primary entry point for constructing predictive coding graphs.
    Uses copy-on-finalize: original node objects are not modified.

    Args:
        nodes: List of NodeBase instances
        edges: List of Edge instances
        task_map: TaskMap instance or dict mapping task names to node names
        inference: InferenceBase instance for inference algorithm
        graph_state_initializer: Optional StateInitBase instance
            (default: FeedforwardStateInit())
        scaling: Optional MuPCConfig instance for muPC parameterization.
            When provided, per-node scaling factors are computed from graph
            topology and attached to each NodeInfo.scaling_config.
        unroll: Required for cyclic graphs: the number of traversals of each
            cycle in the visit schedule (``structure.schedule``), used by
            feedforward initialization and ePC state derivation. ``unroll=1``
            visits each cycle member once. Ignored on acyclic graphs.

    Returns:
        GraphStructure with finalized nodes, edges, and topology

    Raises:
        GraphCycleError: The graph contains cycles and ``unroll`` was not given.
    """
    if unroll is not None and (not isinstance(unroll, int) or unroll < 1):
        raise ValueError(f"unroll must be an int >= 1, got {unroll!r}")
    # 1. Build EdgeInfo objects from Edge objects
    edge_infos = {}
    for edge in edges:
        source_name = edge.source.name
        target_name = edge.target_node.name
        target_slot = edge.target_slot
        key = f"{source_name}->{target_name}:{target_slot}"
        edge_infos[key] = EdgeInfo(
            key=key, source=source_name, target=target_name, slot=target_slot
        )

    # 2. Build node names set for validation
    node_names = {node.name for node in nodes}

    # 3. Validate edge endpoints
    for edge_key, edge_info in edge_infos.items():
        if edge_info.source == edge_info.target:
            raise ValueError(f"Self-edge not allowed: '{edge_key}'")
        if edge_info.source not in node_names:
            raise ValueError(f"Edge source node '{edge_info.source}' does not exist")
        if edge_info.target not in node_names:
            raise ValueError(f"Edge target node '{edge_info.target}' does not exist")

    # 4. For each node: resolve defaults, build slots, build NodeInfo, copy-on-finalize
    finalized_nodes = {}
    for node in nodes:
        name = node.name

        # Validate unique names
        if name in finalized_nodes:
            raise ValueError(f"Duplicate node name '{name}'")

        # Find edges for this node
        in_edges = {k: e for k, e in edge_infos.items() if e.target == name}
        out_edges = {k: e for k, e in edge_infos.items() if e.source == name}

        # Build slots
        slots = _build_slots(node, in_edges)

        # Validate incoming edges connect to valid slots
        for edge_key, edge in in_edges.items():
            if edge.slot not in slots:
                raise ValueError(
                    f"Edge '{edge_key}' connects to non-existent slot '{edge.slot}' "
                    f"in node '{name}'. Available slots: {list(slots.keys())}"
                )

        # Validate slots the node declares as mandatory. A node whose whole
        # purpose is a slot (SkipConnection's residual stream) degrades
        # silently if that slot is left empty: muPC would scale the remaining
        # edges as an ordinary sum and stop counting the node toward the
        # residual depth L.
        for slot_name, slot_spec in type(node).get_slots().items():
            if slot_spec.require_connected and not slots[slot_name].in_neighbors:
                raise ValueError(
                    f"Node '{name}' ({type(node).__name__}) requires at least one "
                    f"edge into slot '{slot_name}', which received none. "
                    f"Connected slots: "
                    f"{[s for s, i in slots.items() if i.in_neighbors] or 'none'}."
                )

        node_info = NodeInfo(
            name=name,
            shape=node.shape,
            node_type=type(node).__name__,
            node_class=type(node),
            node_config=node._extra_config,
            activation=node._activation,
            energy=node._energy,
            latent_init=node._latent_init,
            weight_init=node._weight_init,
            slots=slots,
            in_degree=len(in_edges),
            out_degree=len(out_edges),
            in_edges=tuple(in_edges.keys()),
            out_edges=tuple(out_edges.keys()),
        )
        finalized_nodes[name] = node._with_graph_info(node_info)

    # 5. Topological visit schedule and unique node order
    schedule = _topological_sort(finalized_nodes, edge_infos, unroll)
    node_order = first_occurrence_order(schedule)
    if set(node_order) != set(finalized_nodes):
        raise ValueError(
            f"Internal error: schedule omits nodes "
            f"{sorted(set(finalized_nodes) - set(node_order))}"
        )

    # 5b. Compute and attach muPC scalings if requested
    if scaling is not None:
        if not isinstance(scaling, MuPCConfig):
            raise TypeError(
                f"scaling must be a MuPCConfig instance, got {type(scaling)}"
            )

        mupc_scalings = compute_mupc_scalings(
            finalized_nodes, edge_infos, scaling, node_order
        )

        # Attach scaling_config to each NodeInfo via copy-on-finalize
        updated_nodes = {}
        for name, node in finalized_nodes.items():
            node_scaling = mupc_scalings.get(name)
            if node_scaling is not None:
                new_info = replace(node.node_info, scaling_config=node_scaling)
                updated_nodes[name] = node._with_graph_info(new_info)
            else:
                updated_nodes[name] = node
        finalized_nodes = updated_nodes

    # 6. Resolve task map
    if isinstance(task_map, TaskMap):
        task_map_dict = task_map.to_dict()
    elif isinstance(task_map, dict):
        task_map_dict = task_map
    else:
        raise TypeError(f"task_map must be TaskMap or dict, got {type(task_map)}")

    # 7. Build GraphStructure
    gs_config = {
        "graph_state_initializer": graph_state_initializer or FeedforwardStateInit(),
        "inference": inference,
        "unroll": unroll,
    }

    return GraphStructure(
        nodes=finalized_nodes,
        edges=edge_infos,
        task_map=task_map_dict,
        node_order=node_order,
        schedule=schedule,
        config=gs_config,
    )
