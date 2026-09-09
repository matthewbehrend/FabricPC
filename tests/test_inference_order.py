"""
Tests for inference iteration order independence and self-grad scaling.

These tests pin two invariants of the Phase-2 inference loop
(``InferenceBase.forward_value_and_grad``):

1. **Insertion-order independence.** The per-node ``latent_grad`` produced
   by one forward+grad pass must not depend on the order in which nodes
   were inserted into ``structure.nodes``. Each test builds the same
   graph twice with different ``nodes=[...]`` orderings, runs one pass
   from identical initial state, and asserts ``latent_grad`` (plus
   ``z_mu``, ``error``, ``energy``) match per node.

2. **Self-grad accumulation, not replacement.** ``scale_self_grad`` must
   apply only to the new ``dE/dz_latent`` contribution from the current
   node — never to the cumulative ``latent_grad`` value, which can
   already contain contributions added by earlier-visited successors.
   The sentinel test seeds a known prior value into a node's
   ``latent_grad`` and checks it survives a forward+grad pass as a pure
   addition.

To make the scaling logic observable, tests inject a non-unity
``self_grad_scale`` (muPC's default is 1.0, which would mask any
arithmetic on ``latent_grad`` that happens to multiply by it). Coverage
spans:

- DAG chain ``x → h → y`` under topological, reverse, and arbitrary
  insertion orders, with and without muPC scaling.
- Cyclic graph ``x → a ⇄ b → y`` where no valid topological order
  exists, with and without muPC scaling.
- Sentinel injection on the chain to verify pre-existing ``latent_grad``
  is preserved as a pure addend.
"""

import dataclasses

import jax
import jax.numpy as jnp

from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core.activations import IdentityActivation, TanhActivation
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.mupc import MuPCConfig, MuPCScalingFactors
from fabricpc.core.types import GraphParams, GraphState
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.state_ops import set_latents_to_clamps
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode

_NONUNITY_SELF_GRAD_SCALE = 2.5


def _force_self_grad_scale(structure, scale=_NONUNITY_SELF_GRAD_SCALE):
    """Return a new GraphStructure where every non-input node has a
    MuPCScalingFactors with ``self_grad_scale = scale`` and unit forward /
    topdown / weight scaling. This isolates the order-dependent bug from
    muPC's hard-coded ``self_grad_scale = 1.0`` (which masks it).
    """
    new_nodes = {}
    for name, node in structure.nodes.items():
        info = node.node_info
        if info.in_degree == 0:
            new_nodes[name] = node
            continue
        forward_scale = {ek: 1.0 for ek in info.in_edges}
        topdown_grad_scale = {ek: 1.0 for ek in info.in_edges}
        weight_grad_scale = {ek: 1.0 for ek in info.in_edges}
        new_sc = MuPCScalingFactors(
            forward_scale=forward_scale,
            self_grad_scale=scale,
            topdown_grad_scale=topdown_grad_scale,
            weight_grad_scale=weight_grad_scale,
        )
        new_info = dataclasses.replace(info, scaling_config=new_sc)
        new_nodes[name] = node._with_graph_info(new_info)
    return structure._replace(nodes=new_nodes)


def _build_chain(insertion_order, scaling=None):
    """Build x->h->y chain with caller-controlled node insertion order."""
    w_init = NormalInitializer(std=0.1)
    x = IdentityNode(shape=(6,), name="x")
    h = Linear(shape=(8,), name="h", activation=TanhActivation(), weight_init=w_init)
    y = Linear(
        shape=(4,), name="y", activation=IdentityActivation(), weight_init=w_init
    )
    by_name = {"x": x, "h": h, "y": y}
    return graph(
        nodes=[by_name[n] for n in insertion_order],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=1),
        scaling=scaling,
    )


def _build_cycle(insertion_order, scaling=None, unroll=1):
    """Build x -> a <-> b -> y with a 2-node cycle in the middle."""
    w_init = NormalInitializer(std=0.1)
    x = IdentityNode(shape=(6,), name="x")
    a = Linear(shape=(8,), name="a", activation=TanhActivation(), weight_init=w_init)
    b = Linear(shape=(8,), name="b", activation=TanhActivation(), weight_init=w_init)
    y = Linear(
        shape=(4,), name="y", activation=IdentityActivation(), weight_init=w_init
    )
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
        scaling=scaling,
        unroll=unroll,
    )


def _matching_state(structure, ref_state):
    """Wrap an existing per-node state dict into a fresh GraphState for `structure`.

    Both structures must contain the same node names; ref_state's NodeState
    values are reused verbatim so the two runs start from identical state
    regardless of the structures' insertion orders.
    """
    return GraphState(
        nodes={n: ref_state.nodes[n] for n in structure.nodes},
        batch_size=ref_state.batch_size,
    )


def _matching_params(structure, ref_params):
    return GraphParams(nodes={n: ref_params.nodes[n] for n in structure.nodes})


def _one_grad_pass(structure, params, state, clamps):
    """Run zero_grads + forward_value_and_grad once, return new state."""
    cls = type(structure.config["inference"])
    state = cls.zero_grads(params, state, clamps, structure)
    state = cls.forward_value_and_grad(params, state, clamps, structure)
    return state


def _make_data(rng_key, batch_size=4):
    k_x, k_y = jax.random.split(rng_key, 2)
    x_data = jax.random.normal(k_x, (batch_size, 6))
    y_data = jax.random.normal(k_y, (batch_size, 4))
    return {"x": x_data, "y": y_data}


def _assert_states_match(state_a, state_b, names, atol=1e-6):
    for n in names:
        for field in ("latent_grad", "z_mu", "error", "energy"):
            va = getattr(state_a.nodes[n], field)
            vb = getattr(state_b.nodes[n], field)
            assert jnp.allclose(va, vb, atol=atol), (
                f"{field} for node '{n}' differs across insertion orders: "
                f"max |a-b| = {float(jnp.max(jnp.abs(va - vb)))}"
            )


class TestInsertionOrderIndependenceDAG:
    """latent_grad must not depend on user node-list ordering on a DAG."""

    def test_chain_with_mupc_topo_vs_reverse(self, rng_key):
        s_topo = _force_self_grad_scale(
            _build_chain(["x", "h", "y"], scaling=MuPCConfig())
        )
        s_rev = _force_self_grad_scale(
            _build_chain(["y", "h", "x"], scaling=MuPCConfig())
        )

        params = initialize_params(s_topo, rng_key)
        params_rev = _matching_params(s_rev, params)

        clamps = _make_data(rng_key)
        state_topo = initialize_graph_state(
            s_topo, batch_size=4, rng_key=rng_key, clamps=clamps, params=params
        )
        state_topo = set_latents_to_clamps(state_topo, clamps)
        state_rev = _matching_state(s_rev, state_topo)

        out_topo = _one_grad_pass(s_topo, params, state_topo, clamps)
        out_rev = _one_grad_pass(s_rev, params_rev, state_rev, clamps)

        _assert_states_match(out_topo, out_rev, ["x", "h", "y"])

    def test_chain_with_mupc_arbitrary_order(self, rng_key):
        """Insertion order [h, y, x] also must not perturb gradients."""
        s_ref = _force_self_grad_scale(
            _build_chain(["x", "h", "y"], scaling=MuPCConfig())
        )
        s_alt = _force_self_grad_scale(
            _build_chain(["h", "y", "x"], scaling=MuPCConfig())
        )

        params = initialize_params(s_ref, rng_key)
        params_alt = _matching_params(s_alt, params)

        clamps = _make_data(rng_key)
        state_ref = initialize_graph_state(
            s_ref, batch_size=4, rng_key=rng_key, clamps=clamps, params=params
        )
        state_ref = set_latents_to_clamps(state_ref, clamps)
        state_alt = _matching_state(s_alt, state_ref)

        out_ref = _one_grad_pass(s_ref, params, state_ref, clamps)
        out_alt = _one_grad_pass(s_alt, params_alt, state_alt, clamps)

        _assert_states_match(out_ref, out_alt, ["x", "h", "y"])

    def test_chain_no_mupc(self, rng_key):
        """Without muPC scaling the bug doesn't apply, but the test should
        pass on both old and new code as a sanity check."""
        s_topo = _build_chain(["x", "h", "y"], scaling=None)
        s_rev = _build_chain(["y", "h", "x"], scaling=None)

        params = initialize_params(s_topo, rng_key)
        params_rev = _matching_params(s_rev, params)

        clamps = _make_data(rng_key)
        state_topo = initialize_graph_state(
            s_topo, batch_size=4, rng_key=rng_key, clamps=clamps, params=params
        )
        state_topo = set_latents_to_clamps(state_topo, clamps)
        state_rev = _matching_state(s_rev, state_topo)

        out_topo = _one_grad_pass(s_topo, params, state_topo, clamps)
        out_rev = _one_grad_pass(s_rev, params_rev, state_rev, clamps)

        _assert_states_match(out_topo, out_rev, ["x", "h", "y"])


class TestInsertionOrderIndependenceCyclic:
    """Cyclic graphs have no valid topological order — order-independence
    must still hold."""

    def test_cycle_with_mupc(self, rng_key):
        s_fwd = _force_self_grad_scale(
            _build_cycle(["x", "a", "b", "y"], scaling=MuPCConfig())
        )
        s_bwd = _force_self_grad_scale(
            _build_cycle(["y", "b", "a", "x"], scaling=MuPCConfig())
        )

        params = initialize_params(s_fwd, rng_key)
        params_bwd = _matching_params(s_bwd, params)

        clamps = _make_data(rng_key)
        state_fwd = initialize_graph_state(
            s_fwd, batch_size=4, rng_key=rng_key, clamps=clamps, params=params
        )
        state_fwd = set_latents_to_clamps(state_fwd, clamps)
        state_bwd = _matching_state(s_bwd, state_fwd)

        out_fwd = _one_grad_pass(s_fwd, params, state_fwd, clamps)
        out_bwd = _one_grad_pass(s_bwd, params_bwd, state_bwd, clamps)

        _assert_states_match(out_fwd, out_bwd, ["x", "a", "b", "y"])

    def test_cycle_no_mupc(self, rng_key):
        s_fwd = _build_cycle(["x", "a", "b", "y"], scaling=None)
        s_bwd = _build_cycle(["y", "b", "a", "x"], scaling=None)

        params = initialize_params(s_fwd, rng_key)
        params_bwd = _matching_params(s_bwd, params)

        clamps = _make_data(rng_key)
        state_fwd = initialize_graph_state(
            s_fwd, batch_size=4, rng_key=rng_key, clamps=clamps, params=params
        )
        state_fwd = set_latents_to_clamps(state_fwd, clamps)
        state_bwd = _matching_state(s_bwd, state_fwd)

        out_fwd = _one_grad_pass(s_fwd, params, state_fwd, clamps)
        out_bwd = _one_grad_pass(s_bwd, params_bwd, state_bwd, clamps)

        _assert_states_match(out_fwd, out_bwd, ["x", "a", "b", "y"])


class TestSelfGradAccumulation:
    """The self-grad contribution must be added to (not replace)
    pre-existing latent_grad. Verify by injecting a known prior latent_grad
    value into a node before forward_value_and_grad and checking it is
    preserved."""

    def test_pre_existing_latent_grad_preserved_with_mupc(self, rng_key):
        """Bypass zero_grads, seed h.latent_grad with a sentinel, then run
        forward_value_and_grad. The sentinel must survive into the output
        unscaled by self_grad_scale (only added to, not multiplied)."""
        s = _force_self_grad_scale(_build_chain(["x", "h", "y"], scaling=MuPCConfig()))
        params = initialize_params(s, rng_key)
        clamps = _make_data(rng_key)

        state = initialize_graph_state(
            s, batch_size=4, rng_key=rng_key, clamps=clamps, params=params
        )
        state = set_latents_to_clamps(state, clamps)

        # Reference run: zero grads, then forward+grad
        cls = type(s.config["inference"])
        ref_state = cls.zero_grads(params, state, clamps, s)
        ref_state = cls.forward_value_and_grad(params, ref_state, clamps, s)

        # Sentinel run: same but inject a known prior into h.latent_grad
        sentinel = jnp.ones_like(state.nodes["h"].latent_grad) * 0.7
        zeroed = cls.zero_grads(params, state, clamps, s)
        seeded = zeroed._replace(
            nodes={
                **zeroed.nodes,
                "h": zeroed.nodes["h"]._replace(latent_grad=sentinel),
            }
        )
        out_state = cls.forward_value_and_grad(params, seeded, clamps, s)

        # Expected: out.h.latent_grad == ref.h.latent_grad + sentinel
        # (self_grad and topdown contributions are scaled identically in both
        # runs; the sentinel is pure pre-existing accumulation that should
        # only be added to, never multiplied).
        diff = out_state.nodes["h"].latent_grad - ref_state.nodes["h"].latent_grad
        assert jnp.allclose(diff, sentinel, atol=1e-6), (
            "Pre-existing latent_grad on 'h' was not preserved as a pure "
            "addition: max |diff - sentinel| = "
            f"{float(jnp.max(jnp.abs(diff - sentinel)))}"
        )
