"""
Tests for the linear-Gaussian oracle and for both solvers against it.

The oracle (``fabricpc.utils.linear_pc_oracle``) assembles the energy of a
linear-Gaussian DAG as a quadratic from params, edges, muPC forward scales,
biases, and precisions, and solves it exactly. ``TestOracleSelfChecks``
pins the oracle against hand-computed numbers, Innocenti et al. 2024
Theorem 1, and explicit Hessian forms. The solver tests then require
``EPCInference`` and ``InferenceSGD`` to reach the oracle's equilibrium,
pin the exact stability bound of each (which fixes the scale of the
gradient implementations, not only their direction), and check the
error-coordinate Hessian-vector product and power iteration against the
oracle's H_ε.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fabricpc.core import EPCInference
from fabricpc.core.activations import IdentityActivation, TanhActivation
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.initializers import MuPCInitializer, NormalInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.topology import Edge
from fabricpc.core.types import GraphParams, NodeParams
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear, StorkeyHopfield
from fabricpc.nodes.identity import IdentityNode
from fabricpc.utils import linear_pc_oracle as oracle

PLACEHOLDER = EPCInference(eta_infer=1e-3, infer_steps=1)


# =============================================================================
# Builders
# =============================================================================


def _linear(shape, name, std=0.3, precision=1.0, use_bias=True, weight_init=None):
    return Linear(
        shape=(shape,),
        name=name,
        activation=IdentityActivation(),
        energy=GaussianEnergy(precision=precision),
        use_bias=use_bias,
        weight_init=weight_init or NormalInitializer(std=std),
    )


def _chain(
    hidden,
    dims=(5, 4, 3),
    std=0.3,
    use_bias=True,
    precisions=None,
    scaling=None,
    weight_init=None,
):
    """x(dims[0]) -> h1..h_hidden (dims[1]) -> y(dims[2]), identity activations.

    ``precisions`` maps node names to GaussianEnergy precisions.
    """
    precisions = precisions or {}
    x = IdentityNode(shape=(dims[0],), name="x")
    hs = [
        _linear(
            dims[1],
            f"h{i + 1}",
            std=std,
            precision=precisions.get(f"h{i + 1}", 1.0),
            use_bias=use_bias,
            weight_init=weight_init,
        )
        for i in range(hidden)
    ]
    y = _linear(
        dims[2],
        "y",
        std=std,
        precision=precisions.get("y", 1.0),
        use_bias=use_bias,
        weight_init=weight_init,
    )
    nodes = [x, *hs, y]
    edges = [Edge(source=a, target=b.slot("in")) for a, b in zip(nodes[:-1], nodes[1:])]
    return graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=x, y=y),
        inference=PLACEHOLDER,
        scaling=scaling,
    )


def _fork_merge():
    """x -> {a, b} -> y: y sums two weighted edges."""
    x = IdentityNode(shape=(5,), name="x")
    a = _linear(4, "a")
    b = _linear(4, "b")
    y = _linear(3, "y")
    return graph(
        nodes=[x, a, b, y],
        edges=[
            Edge(source=x, target=a.slot("in")),
            Edge(source=x, target=b.slot("in")),
            Edge(source=a, target=y.slot("in")),
            Edge(source=b, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=PLACEHOLDER,
    )


def _prior_source():
    """The convex DAG of test_inference_epc.TestSPCEquivalence: an unclamped
    top-down prior feeding h beside the clamped input."""
    w_init = NormalInitializer(std=0.8)
    x = IdentityNode(shape=(5,), name="x")
    prior = _linear(3, "prior", weight_init=w_init)
    h = _linear(4, "h", weight_init=w_init)
    y = _linear(6, "y", weight_init=w_init)
    return graph(
        nodes=[x, prior, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=prior, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=PLACEHOLDER,
    )


def _identity_cycle(unroll):
    """x -> a <-> b -> y with identity activations, for the DAG check."""
    x = IdentityNode(shape=(4,), name="x")
    a = _linear(4, "a")
    b = _linear(4, "b")
    y = _linear(2, "y")
    return graph(
        nodes=[x, a, b, y],
        edges=[
            Edge(source=x, target=a.slot("in")),
            Edge(source=a, target=b.slot("in")),
            Edge(source=b, target=a.slot("in")),
            Edge(source=b, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=PLACEHOLDER,
        unroll=unroll,
    )


def _inject_biases(params, key, std=0.5):
    """Linear.initialize_params zero-fills biases (nodes/linear.py:191), so
    use_bias=True alone exercises nothing. Draw them here."""
    nodes = {}
    for i, (name, node_params) in enumerate(params.nodes.items()):
        biases = dict(node_params.biases)
        if "b" in biases and biases["b"].size > 0:
            biases["b"] = std * jax.random.normal(
                jax.random.fold_in(key, i), biases["b"].shape
            )
        nodes[name] = NodeParams(weights=node_params.weights, biases=biases)
    return GraphParams(nodes=nodes)


BATCH = 3


def _clamps(structure, key, clamp_output=True, extra=()):
    """Gaussian clamps for x, optionally y, and any extra node names."""
    keys = jax.random.split(key, 2 + len(extra))
    names = {n: structure.nodes[n].node_info.shape[0] for n in structure.nodes}
    clamps = {"x": jax.random.normal(keys[0], (BATCH, names["x"]))}
    if clamp_output:
        clamps["y"] = jax.random.normal(keys[1], (BATCH, names["y"]))
    for k, name in zip(keys[2:], extra):
        clamps[name] = jax.random.normal(k, (BATCH, names[name]))
    return clamps


class Bunch:
    """A fixture: how to build the structure, params, and clamps."""

    def __init__(self, build, clamp_output=True, extra_clamps=(), biases=False):
        self.build = build
        self.clamp_output = clamp_output
        self.extra_clamps = extra_clamps
        self.biases = biases

    def make(self, key):
        structure = self.build()
        params = initialize_params(structure, key)
        if self.biases:
            params = _inject_biases(params, jax.random.fold_in(key, 7))
        clamps = _clamps(
            structure, jax.random.fold_in(key, 1), self.clamp_output, self.extra_clamps
        )
        return structure, params, clamps


BUNCH = {
    "chain-h1": Bunch(lambda: _chain(1)),
    "chain-h2": Bunch(lambda: _chain(2)),
    "chain-h3": Bunch(lambda: _chain(3)),
    "chain-h4": Bunch(lambda: _chain(4)),
    "chain-h2-bias": Bunch(lambda: _chain(2), biases=True),
    "chain-h2-precision": Bunch(lambda: _chain(2, precisions={"h1": 2.0, "y": 0.5})),
    "chain-h3-std0.8": Bunch(lambda: _chain(3, std=0.8)),
    "fork-merge": Bunch(_fork_merge),
    "clamped-internal": Bunch(lambda: _chain(2), extra_clamps=("h2",)),
    "prior-source": Bunch(_prior_source),
    "unclamped-readout": Bunch(lambda: _chain(2), clamp_output=False),
    "mupc-chain-h3": Bunch(
        lambda: _chain(3, scaling=MuPCConfig(), weight_init=MuPCInitializer())
    ),
}
CHAINS = [k for k in BUNCH if k.startswith("chain-h") and "precision" not in k]


def _source_means(structure, state):
    """z_mu of every unclamped source as the state carries it (constant
    through an ePC segment once begin_segment has assigned it)."""
    return {
        name: np.asarray(state.nodes[name].z_mu)
        for name in structure.nodes
        if structure.nodes[name].node_info.in_degree == 0
    }


def _epc_initial_state(structure, params, clamps, key):
    """Feedforward init followed by begin_segment: ε = 0 and every source's
    z_mu fixed at its initial latent."""
    state = initialize_graph_state(structure, BATCH, key, clamps, params=params)
    return EPCInference.begin_segment(params, state, clamps, structure)


# =============================================================================
# Oracle self-checks
# =============================================================================


class TestOracleSelfChecks:
    def test_scalar_chain_hand_numbers(self):
        """x=1, W1=2, W2=3, y=1: E(z) = ½(z − 2)² + ½(1 − 3z)² has its minimum
        at z = 0.5 with E* = 1.25; errors −1.5 (h) and −0.5 (y); S = 1 + 9,
        r = 1 − 6; both Hessians are the scalar 10."""
        structure = _chain(1, dims=(1, 1, 1), use_bias=False)
        params = GraphParams(
            nodes={
                "x": NodeParams(weights={}, biases={}),
                "h1": NodeParams(weights={"x->h1:in": jnp.array([[2.0]])}, biases={}),
                "y": NodeParams(weights={"h1->y:in": jnp.array([[3.0]])}, biases={}),
            }
        )
        clamps = {"x": jnp.array([[1.0]]), "y": jnp.array([[1.0]])}
        eq = oracle.linear_equilibrium(params, structure, clamps)
        np.testing.assert_allclose(eq.z_star["h1"], [[0.5]])
        np.testing.assert_allclose(eq.total_energy, [1.25])
        np.testing.assert_allclose(eq.error_star["h1"], [[-1.5]])
        np.testing.assert_allclose(eq.error_star["y"], [[-0.5]])
        E_star, S, r = oracle.theorem1_energy(params, structure, clamps)
        np.testing.assert_allclose(E_star, [1.25])
        np.testing.assert_allclose(S, [[10.0]])
        np.testing.assert_allclose(r, [[-5.0]])
        np.testing.assert_allclose(oracle.latent_hessian(eq.quad), [[10.0]])
        np.testing.assert_allclose(oracle.epsilon_hessian(eq.quad), [[10.0]])
        assert oracle.stability_bound(oracle.epsilon_hessian(eq.quad)) == 0.2

    @pytest.mark.parametrize(
        "bunch", CHAINS + ["chain-h2-bias", "chain-h2-precision", "mupc-chain-h3"]
    )
    def test_theorem1_matches_least_squares(self, rng_key, bunch):
        structure, params, clamps = BUNCH[bunch].make(rng_key)
        eq = oracle.linear_equilibrium(params, structure, clamps)
        E_star, _, _ = oracle.theorem1_energy(params, structure, clamps)
        np.testing.assert_allclose(E_star, eq.total_energy, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("bunch", CHAINS + ["chain-h2-precision"])
    def test_error_pullback_is_precision_weighted(self, rng_key, bunch):
        """p_l·ε_l* = p_y·ε_y*·P_lᵀ on a chain (ε_l* = ε_y*·P_lᵀ at unit
        precision): the equilibrium hidden errors are the output error
        pulled back through the downstream maps."""
        structure, params, clamps = BUNCH[bunch].make(rng_key)
        eq = oracle.linear_equilibrium(params, structure, clamps)
        w_eff = oracle.effective_edge_matrices(params, structure)
        order = structure.node_order
        p = {
            n: structure.nodes[n].node_info.energy.config.get("precision", 1.0)
            for n in order[1:]
        }
        for i, h in enumerate(order[1:-1]):
            P = np.eye(structure.nodes[h].node_info.shape[0])
            for name in order[i + 2 :]:
                P = P @ w_eff[structure.nodes[name].node_info.in_edges[0]]
            np.testing.assert_allclose(
                p[h] * eq.error_star[h],
                p["y"] * eq.error_star["y"] @ P.T,
                atol=1e-10,
            )

    def test_unclamped_readout_has_zero_energy(self, rng_key):
        structure, params, clamps = BUNCH["unclamped-readout"].make(rng_key)
        eq = oracle.linear_equilibrium(params, structure, clamps)
        np.testing.assert_allclose(eq.total_energy, 0.0, atol=1e-12)
        for name in structure.nodes:
            np.testing.assert_allclose(eq.z_star[name], eq.quad.z_ff[name], atol=1e-12)

    @pytest.mark.parametrize(
        "bunch", ["fork-merge", "prior-source", "clamped-internal"]
    )
    def test_normal_equations_hold(self, rng_key, bunch):
        structure, params, clamps = BUNCH[bunch].make(rng_key)
        eq = oracle.linear_equilibrium(params, structure, clamps)
        quad = eq.quad
        z = oracle.flatten_free(quad, eq.z_star)
        residual = quad.A.T @ (quad.A @ z - quad.c)
        assert np.linalg.norm(residual) <= 1e-10 * max(1.0, np.linalg.norm(quad.c))

    @pytest.mark.parametrize(
        "bunch", ["fork-merge", "prior-source", "clamped-internal"]
    )
    def test_epsilon_hessian_explicit_form(self, rng_key, bunch):
        """H_ε = diag(p over free in-degree > 0 nodes) + Σ_{clamped t} p_t J_tᵀJ_t,
        J_t = ∂μ_t/∂ε: an unclamped source contributes no diagonal block, so
        the floor comes only from the nodes that own an energy term."""
        structure, params, clamps = BUNCH[bunch].make(rng_key)
        eq = oracle.linear_equilibrium(params, structure, clamps)
        quad = eq.quad
        D = quad.A.shape[1]
        w_eff = oracle.effective_edge_matrices(params, structure)
        H = np.zeros((D, D))
        for t in quad.rows:
            info = structure.nodes[t].node_info
            p = quad.precision[t]
            if t not in clamps:
                cols = quad.col_offsets[t]
                H[cols, cols] += p * np.eye(info.shape[0])
                continue
            G = np.zeros((info.shape[0], D))
            for key in info.in_edges:
                s = structure.edges[key].source
                if s not in clamps:
                    G[:, quad.col_offsets[s]] += w_eff[key].T
            J = G @ quad.M
            H += p * J.T @ J
        np.testing.assert_allclose(oracle.epsilon_hessian(quad), H, atol=1e-10)
        assert np.all(np.triu(quad.B_lower) == 0)
        np.testing.assert_allclose(np.linalg.det(quad.M), 1.0, rtol=1e-10)

    def test_eigenvalue_floor(self, rng_key):
        for bunch in CHAINS + ["chain-h2-precision"]:
            structure, params, clamps = BUNCH[bunch].make(rng_key)
            eq = oracle.linear_equilibrium(params, structure, clamps)
            lam_min = np.linalg.eigvalsh(oracle.epsilon_hessian(eq.quad))[0]
            assert lam_min >= min(eq.quad.precision.values()) - 1e-10, bunch
        structure, params, clamps = BUNCH["prior-source"].make(rng_key)
        eq = oracle.linear_equilibrium(params, structure, clamps)
        assert np.linalg.eigvalsh(oracle.epsilon_hessian(eq.quad))[0] < 1.0

    def test_validate_rejects_non_linear_gaussian(self):
        x = IdentityNode(shape=(3,), name="x")

        def build(y):
            return graph(
                nodes=[x, y],
                edges=[Edge(source=x, target=y.slot("in"))],
                task_map=TaskMap(x=x, y=y),
                inference=PLACEHOLDER,
            )

        with pytest.raises(ValueError, match="activation"):
            oracle.validate_linear_gaussian(
                build(Linear(shape=(2,), name="y", activation=TanhActivation()))
            )
        with pytest.raises(ValueError, match="energy"):
            oracle.validate_linear_gaussian(
                build(Linear(shape=(2,), name="y", energy=CrossEntropyEnergy()))
            )
        with pytest.raises(ValueError, match="flatten_input"):
            oracle.validate_linear_gaussian(
                build(Linear(shape=(2,), name="y", flatten_input=True))
            )
        with pytest.raises(ValueError):
            oracle.validate_linear_gaussian(
                build(
                    StorkeyHopfield(
                        shape=(3,), name="y", hopfield_strength=2.0, use_bias=False
                    )
                )
            )

    @pytest.mark.parametrize("unroll", [1, 2])
    def test_validate_rejects_cycles_at_any_unroll(self, unroll):
        """At unroll=1 a cycle's members are visited once, so the schedule
        length equals the node count; the back-edge check still fires."""
        structure = _identity_cycle(unroll)
        assert len(structure.schedule) == len(structure.node_order) or unroll > 1
        with pytest.raises(ValueError, match="cycle"):
            oracle.validate_linear_gaussian(structure)

    def test_relaxed_fraction_and_steps(self):
        eigs = np.array([1.0, 10.0])
        np.testing.assert_allclose(
            oracle.relaxed_fraction(0.05, 5, eigs), [1 - 0.95**5, 1 - 0.5**5]
        )
        T = oracle.steps_to_contract(0.1, eigs, 1e-6)
        assert 0.9**T <= 1e-6 < 0.9 ** (T - 1)
        with pytest.raises(ValueError):
            oracle.steps_to_contract(0.3, eigs, 1e-6)
