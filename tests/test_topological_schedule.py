"""
Tests for the generalized topological visit schedule (graph unrolling).

``_topological_sort(nodes, edges, unroll)`` returns the full node visit
schedule: on DAGs the plain Kahn/BFS order regardless of ``unroll``; on
cyclic graphs it requires an explicit ``unroll=U >= 1`` and emits each
nontrivial strongly connected component's members U times, or raises
``GraphCycleError`` without one. ``graph()`` exposes the schedule as
``structure.schedule`` with ``structure.node_order`` its first-occurrence
deduplication.
"""

import itertools

import pytest

from fabricpc.core.activations import IdentityActivation, TanhActivation
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import (
    GraphCycleError,
    TaskMap,
    first_occurrence_order,
    graph,
)


def _make_nodes():
    w_init = NormalInitializer(std=0.1)
    from fabricpc.nodes import Linear
    from fabricpc.nodes.identity import IdentityNode

    x = IdentityNode(shape=(6,), name="x")
    a = Linear(shape=(8,), name="a", activation=TanhActivation(), weight_init=w_init)
    b = Linear(shape=(8,), name="b", activation=TanhActivation(), weight_init=w_init)
    y = Linear(
        shape=(4,), name="y", activation=IdentityActivation(), weight_init=w_init
    )
    return x, a, b, y


def _build_dag(insertion_order, unroll=None):
    """Diamond DAG x -> {a, b} -> y with caller-controlled insertion order."""
    x, a, b, y = _make_nodes()
    by_name = {"x": x, "a": a, "b": b, "y": y}
    return graph(
        nodes=[by_name[n] for n in insertion_order],
        edges=[
            Edge(source=x, target=a.slot("in")),
            Edge(source=x, target=b.slot("in")),
            Edge(source=a, target=y.slot("in")),
            Edge(source=b, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=1),
        unroll=unroll,
    )


def _build_cycle(insertion_order=("x", "a", "b", "y"), unroll=None):
    """x -> a <-> b -> y with a 2-node cycle in the middle."""
    x, a, b, y = _make_nodes()
    by_name = {"x": x, "a": a, "b": b, "y": y}
    return graph(
        nodes=[by_name[n] for n in insertion_order],
        edges=[
            Edge(source=x, target=a.slot("in")),
            Edge(source=a, target=b.slot("in")),
            Edge(source=b, target=a.slot("in")),
            Edge(source=b, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=1),
        unroll=unroll,
    )


class TestDAGSchedule:
    def test_dag_schedule_equals_node_order(self):
        """On a DAG the schedule is the Kahn order, one visit per node."""
        structure = _build_dag(("x", "a", "b", "y"))
        assert structure.schedule == structure.node_order
        assert structure.schedule == ("x", "a", "b", "y")

    @pytest.mark.parametrize("unroll", [None, 1, 3])
    def test_dag_order_insensitive_to_unroll(self, unroll):
        """unroll does not change a DAG's schedule."""
        structure = _build_dag(("x", "a", "b", "y"), unroll=unroll)
        assert structure.schedule == ("x", "a", "b", "y")
        assert structure.node_order == ("x", "a", "b", "y")

    def test_dag_order_across_insertion_permutations(self):
        """Every insertion permutation yields the same Kahn order (queue
        seeded from dict order, successors in edge-declaration order) with
        and without unroll — the legacy behavior, bit-identical."""
        for perm in itertools.permutations(("x", "a", "b", "y")):
            structure = _build_dag(perm)
            structure_u = _build_dag(perm, unroll=2)
            # x is the only seed; successors follow edge order (a before b).
            assert structure.schedule == ("x", "a", "b", "y")
            assert structure_u.schedule == structure.schedule


class TestCyclicSchedule:
    def test_cycle_without_unroll_raises(self):
        """Cyclic graph with no unroll raises, naming the unordered nodes."""
        with pytest.raises(GraphCycleError) as excinfo:
            _build_cycle()
        message = str(excinfo.value)
        for name in ("a", "b", "y"):
            assert f"'{name}'" in message
        assert "'x'" not in message
        assert "unroll" in message

    def test_cycle_unroll_2(self):
        structure = _build_cycle(unroll=2)
        assert structure.schedule == ("x", "a", "b", "a", "b", "y")
        assert structure.node_order == ("x", "a", "b", "y")

    def test_cycle_unroll_1_single_visit(self):
        structure = _build_cycle(unroll=1)
        assert structure.schedule == ("x", "a", "b", "y")
        assert structure.node_order == ("x", "a", "b", "y")

    def test_node_order_is_first_occurrence_of_schedule(self):
        for unroll in (1, 2, 5):
            structure = _build_cycle(unroll=unroll)
            assert structure.node_order == first_occurrence_order(structure.schedule)
            assert set(structure.node_order) == set(structure.nodes)

    def test_cycle_schedule_deterministic(self):
        first = _build_cycle(unroll=3).schedule
        for _ in range(3):
            assert _build_cycle(unroll=3).schedule == first

    def test_cycle_insertion_order_determinism(self):
        """The intra-cycle order follows BFS from the entry node, so the
        cycle emits (a, b) from a's in-edge from x regardless of whether a
        or b was inserted first."""
        for perm in itertools.permutations(("x", "a", "b", "y")):
            structure = _build_cycle(insertion_order=perm, unroll=2)
            assert structure.schedule.count("a") == 2
            assert structure.schedule.count("b") == 2
            assert structure.schedule[0] == "x"
            assert structure.schedule[-1] == "y"
            # entry node a is visited before b within each traversal
            first_a = structure.schedule.index("a")
            first_b = structure.schedule.index("b")
            assert first_a < first_b


class TestGraphUnrollValidation:
    @pytest.mark.parametrize("bad_unroll", [0, -1, 1.5, "2", True, False])
    def test_graph_rejects_bad_unroll(self, bad_unroll):
        """Bools are rejected too: isinstance(True, int) holds, so without
        the explicit bool check graph(..., unroll=True) would silently build
        a degree-1 schedule."""
        with pytest.raises(ValueError, match="unroll"):
            _build_dag(("x", "a", "b", "y"), unroll=bad_unroll)

    def test_unroll_recorded_in_config(self):
        structure = _build_cycle(unroll=2)
        assert structure.config["unroll"] == 2
        dag = _build_dag(("x", "a", "b", "y"))
        assert dag.config["unroll"] is None


def _identity_nodes(names, dim=4):
    from fabricpc.nodes.identity import IdentityNode

    return {name: IdentityNode(shape=(dim,), name=name) for name in names}


def _graph_of(nodes_by_name, edge_pairs, task_map, unroll):
    """Build a graph from IdentityNodes and (source, target) name pairs."""
    return graph(
        nodes=list(nodes_by_name.values()),
        edges=[
            Edge(source=nodes_by_name[s], target=nodes_by_name[t].slot("in"))
            for s, t in edge_pairs
        ],
        task_map=task_map,
        inference=InferenceSGD(eta_infer=0.1, infer_steps=1),
        unroll=unroll,
    )


class TestComplexTopologies:
    def test_three_node_scc(self):
        """x -> a -> b -> c -> a, c -> y: BFS from the entry a walks the
        cycle in edge order, repeated U times."""
        n = _identity_nodes(("x", "a", "b", "c", "y"))
        structure = _graph_of(
            n,
            [("x", "a"), ("a", "b"), ("b", "c"), ("c", "a"), ("c", "y")],
            TaskMap(x=n["x"], y=n["y"]),
            unroll=2,
        )
        assert structure.schedule == ("x", "a", "b", "c", "a", "b", "c", "y")
        assert structure.node_order == ("x", "a", "b", "c", "y")

    def test_two_disjoint_cycles(self):
        """Two independent SCCs unroll independently, ordered by the
        condensation Kahn (seeded from dict order, successors in edge
        order)."""
        n = _identity_nodes(("x", "a", "b", "c", "d", "y"))
        structure = _graph_of(
            n,
            [
                ("x", "a"),
                ("a", "b"),
                ("b", "a"),
                ("x", "c"),
                ("c", "d"),
                ("d", "c"),
                ("b", "y"),
                ("d", "y"),
            ],
            TaskMap(x=n["x"], y=n["y"]),
            unroll=2,
        )
        assert structure.schedule == ("x", "a", "b", "a", "b", "c", "d", "c", "d", "y")
        assert structure.node_order == ("x", "a", "b", "c", "d", "y")

    def test_overlapping_cycles_share_one_scc(self):
        """a <-> b and b <-> c overlap in b, so Tarjan merges them into one
        SCC {a, b, c}: the whole component repeats U times as a unit."""
        n = _identity_nodes(("x", "a", "b", "c", "y"))
        structure = _graph_of(
            n,
            [("x", "a"), ("a", "b"), ("b", "a"), ("b", "c"), ("c", "b"), ("c", "y")],
            TaskMap(x=n["x"], y=n["y"]),
            unroll=2,
        )
        assert structure.schedule == ("x", "a", "b", "c", "a", "b", "c", "y")

    def test_multi_entry_scc(self):
        """Both a and b have external in-edges, so both seed the intra-SCC
        BFS (in member dict order)."""
        n = _identity_nodes(("x1", "x2", "a", "b", "y"))
        structure = _graph_of(
            n,
            [("x1", "a"), ("x2", "b"), ("a", "b"), ("b", "a"), ("b", "y")],
            TaskMap(x=n["x1"], y=n["y"]),
            unroll=2,
        )
        assert structure.schedule == ("x1", "x2", "a", "b", "a", "b", "y")

    def test_entryless_scc_falls_back_to_first_member(self):
        """A cycle with no external in-edge (no source feeds it) seeds its
        BFS from the first member in dict order."""
        n = _identity_nodes(("a", "b", "y"))
        structure = _graph_of(
            n,
            [("a", "b"), ("b", "a"), ("b", "y")],
            TaskMap(y=n["y"]),
            unroll=2,
        )
        assert structure.schedule == ("a", "b", "a", "b", "y")
        assert structure.node_order == ("a", "b", "y")


class TestFirstOccurrenceOrder:
    def test_dedup_keeps_first_occurrences(self):
        assert first_occurrence_order(("x", "a", "b", "a", "b", "y")) == (
            "x",
            "a",
            "b",
            "y",
        )

    def test_identity_on_unique(self):
        assert first_occurrence_order(("x", "a", "y")) == ("x", "a", "y")
