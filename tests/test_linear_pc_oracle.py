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
error-coordinate Hessian-vector product and the Lanczos spectrum estimator
(``fabricpc.core.epsilon_spectrum``) against the oracle's H_ε.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import inject_biases, with_inference
from fabricpc.core import EPCInference, InferenceSGD
from fabricpc.core.activations import IdentityActivation, TanhActivation
from fabricpc.core.epsilon_spectrum import epsilon_spectrum, weighted_relaxed_fraction
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
from fabricpc.utils.dashboarding.inference_tracking import run_inference_with_history

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
            params = inject_biases(params, jax.random.fold_in(key, 7))
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

    def test_stability_bound_needs_positive_curvature(self):
        """An indefinite or negative quadratic has no stable descent rate;
        2/λ_max would be a negative number presented as a rate."""
        with pytest.raises(ValueError, match="no positive curvature"):
            oracle.stability_bound(np.diag([-3.0, -1.0]))
        with pytest.raises(ValueError, match="no positive curvature"):
            oracle.stability_bound(np.zeros((2, 2)))
        assert oracle.stability_bound(np.diag([-3.0, 4.0])) == 0.5

    def test_gradient_weights_and_weighted_fraction(self):
        """Weights are squared overlaps summed over samples; the weighted
        fraction averages f over the positive modes only."""
        H = np.diag([-2.0, 1.0, 4.0])
        g0 = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])  # (D, batch)
        eigs, w = oracle.gradient_weights(H, g0)
        np.testing.assert_allclose(eigs, [-2.0, 1.0, 4.0])
        np.testing.assert_allclose(w, np.array([1.0, 4.0, 2.0]) / 7.0)
        f = oracle.relaxed_fraction(0.1, 3, np.array([1.0, 4.0]))
        expected = (4.0 * f[0] + 2.0 * f[1]) / 6.0
        assert oracle.weighted_relaxed_fraction(eigs, w, 0.1, 3) == pytest.approx(
            expected
        )
        assert np.isnan(oracle.weighted_relaxed_fraction([-1.0], [1.0], 0.1, 3))

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


# =============================================================================
# Solvers against the oracle
# =============================================================================


def _oracle_case(bunch, key):
    """Structure, params, clamps, the feedforward-initialized state, and the
    oracle equilibrium with every unclamped source's z_mu read from that
    state (the constant ePC holds it at)."""
    structure, params, clamps = BUNCH[bunch].make(key)
    state = initialize_graph_state(structure, BATCH, key, clamps, params=params)
    eq = oracle.linear_equilibrium(
        params, structure, clamps, source_means=_source_means(structure, state)
    )
    return structure, params, clamps, state, eq


def _assert_matches_oracle(structure, final, eq, clamps, *, source_errors=True):
    """z_latent on every node; energy and error on every in_degree > 0 node;
    error on unclamped sources when ``source_errors`` (the state-based
    solvers re-sync a source's z_mu, so their source error is 0 by
    construction); the in_degree > 0 total against the oracle total."""
    tol = dict(rtol=1e-4, atol=1e-4)
    total = np.zeros(BATCH)
    for name in structure.nodes:
        info = structure.nodes[name].node_info
        node = final.nodes[name]
        np.testing.assert_allclose(
            np.asarray(node.z_latent),
            eq.z_star[name],
            err_msg=f"{name}: z_latent",
            **tol,
        )
        if info.in_degree > 0:
            np.testing.assert_allclose(
                np.asarray(node.energy),
                eq.node_energy[name],
                err_msg=f"{name}: energy",
                **tol,
            )
            np.testing.assert_allclose(
                np.asarray(node.error),
                eq.error_star[name],
                err_msg=f"{name}: error",
                **tol,
            )
            total = total + np.asarray(node.energy)
        elif source_errors and name not in clamps:
            np.testing.assert_allclose(
                np.asarray(node.error),
                eq.error_star[name],
                err_msg=f"{name}: source error",
                **tol,
            )
    np.testing.assert_allclose(total, eq.total_energy, err_msg="total energy", **tol)


def _epc_schedule(eq, eta_scale=1.0, atol=1e-4):
    """(eta, T) for ePC: eta = eta_scale / λ_max(H_ε); T contracts the excited
    modes (the floor modes are never excited from ε = 0) until the latent
    error is below atol / 10, using ‖M‖₂ to convert ε error to z error."""
    quad = eq.quad
    H = oracle.epsilon_hessian(quad)
    eta = eta_scale / float(np.linalg.eigvalsh(H)[-1])
    dist = np.linalg.norm(oracle.flatten_free(quad, eq.error_star))
    if dist == 0.0:
        return eta, 1
    ratio = atol / (10.0 * np.linalg.norm(quad.M, 2) * dist)
    excited = oracle.excited_eigenvalues(H, oracle.epsilon_gradient_at_zero(quad))
    return eta, max(1, oracle.steps_to_contract(eta, excited, ratio))


def _spc_schedule(eq, eta_scale=1.0, atol=1e-4):
    """(eta, T) for the state-based solver: eta = eta_scale / λ_max(H_z); T
    from the full spectrum (every mode can be excited from the feedforward
    point)."""
    quad = eq.quad
    H = oracle.latent_hessian(quad)
    eta = eta_scale / float(np.linalg.eigvalsh(H)[-1])
    dist = np.linalg.norm(
        oracle.flatten_free(quad, quad.z_ff) - oracle.flatten_free(quad, eq.z_star)
    )
    if dist == 0.0:
        return eta, 1
    ratio = atol / (10.0 * dist)
    return eta, max(1, oracle.steps_to_contract(eta, np.linalg.eigvalsh(H), ratio))


def _run(structure, inference, params, state, clamps):
    structure = with_inference(structure, inference=inference)
    return structure, inference.run_inference(params, state, clamps, structure)


class TestEPCReachesOracle:
    @pytest.mark.parametrize("bunch", list(BUNCH))
    def test_equilibrium(self, rng_key, bunch):
        structure, params, clamps, state, eq = _oracle_case(bunch, rng_key)
        eta, steps = _epc_schedule(eq)
        structure, final = _run(
            structure,
            EPCInference(eta_infer=eta, infer_steps=steps),
            params,
            state,
            clamps,
        )
        _assert_matches_oracle(structure, final, eq, clamps)


class TestSPCReachesOracle:
    @pytest.mark.parametrize("bunch", list(BUNCH))
    def test_equilibrium(self, rng_key, bunch):
        """Includes the muPC chain: with identity activations the top-down
        scale a·jacobian_gain is the exact chain-rule factor (jacobian_gain
        = 1) and self_grad_scale = 1, so sPC+muPC is plain gradient descent
        on the input-scaled energy the oracle assembles."""
        structure, params, clamps, state, eq = _oracle_case(bunch, rng_key)
        eta, steps = _spc_schedule(eq)
        structure, final = _run(
            structure,
            InferenceSGD(eta_infer=eta, infer_steps=steps),
            params,
            state,
            clamps,
        )
        _assert_matches_oracle(structure, final, eq, clamps, source_errors=False)


class TestStabilityBracket:
    """The exact bound 2/λ_max pins the scale of each solver's gradient, not
    only its direction: 0.95× converges to the oracle, 1.05× diverges.

    On ``mupc-chain-h3`` the identity activations make muPC's top-down scale
    the exact chain-rule factor (jacobian_gain = 1, self_grad_scale = 1), so
    the bracket pins sPC+muPC's gradient scale as well; the equilibrium test
    alone cannot, because a diagonal preconditioner shares the fixed point.
    """

    @pytest.mark.parametrize("bunch", ["chain-h3", "mupc-chain-h3"])
    @pytest.mark.parametrize("solver", ["epc", "spc"])
    def test_bracket(self, rng_key, solver, bunch):
        structure, params, clamps, state, eq = _oracle_case(bunch, rng_key)
        if solver == "epc":
            H, schedule, make = (
                oracle.epsilon_hessian(eq.quad),
                _epc_schedule,
                EPCInference,
            )
        else:
            H, schedule, make = (
                oracle.latent_hessian(eq.quad),
                _spc_schedule,
                InferenceSGD,
            )
        bound = oracle.stability_bound(H)

        eta, steps = schedule(eq, eta_scale=0.95 * 2.0)
        assert abs(eta - 0.95 * bound) < 1e-12
        structure_c, final = _run(
            structure, make(eta_infer=eta, infer_steps=steps), params, state, clamps
        )
        _assert_matches_oracle(structure_c, final, eq, clamps, source_errors=False)

        structure_d = with_inference(
            structure, inference=make(eta_infer=1.05 * bound, infer_steps=150)
        )
        _, history = run_inference_with_history(params, state, clamps, structure_d)
        total = sum(
            np.asarray(history[name]["energy"])
            for name in structure.nodes
            if structure.nodes[name].node_info.in_degree > 0
        )
        assert np.all(np.isfinite(total))
        # Once the divergent mode dominates, the energy grows every step.
        assert np.all(np.diff(total[-50:]) > 0), total[-50:]


class TestEpsilonHVPMatchesOracle:
    @pytest.mark.parametrize("bunch", ["fork-merge", "prior-source"])
    def test_hvp(self, rng_key, bunch):
        """The Hessian-vector product through EPCInference.error_energy equals
        H_ε v column-wise per sample (float32 against float64)."""
        structure, params, clamps, state, eq = _oracle_case(bunch, rng_key)
        synced = EPCInference.begin_segment(params, state, clamps, structure)
        energy_of, errors = EPCInference.error_energy(params, synced, clamps, structure)
        v = {
            name: jax.random.normal(jax.random.fold_in(rng_key, i), e.shape)
            for i, (name, e) in enumerate(errors.items())
        }
        hv = jax.jvp(jax.grad(lambda e: energy_of(e)[0]), (errors,), (v,))[1]
        expected = oracle.unflatten_free(
            eq.quad, oracle.epsilon_hessian(eq.quad) @ oracle.flatten_free(eq.quad, v)
        )
        assert set(hv) == set(eq.quad.free)
        for name in hv:
            np.testing.assert_allclose(np.asarray(hv[name]), expected[name], atol=1e-4)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    @pytest.mark.parametrize(
        "bunch", ["fork-merge", "prior-source", "chain-h3-std0.8", "chain-h3"]
    )
    def test_lanczos_matches_excited_extremes(self, rng_key, bunch, dtype):
        """Lanczos from g0 returns the extremes of the excited spectrum, not
        of the full one, and its gradient-weighted relaxed fraction equals
        the oracle's. On the chains the Krylov space has dimension d_y = 3,
        so the relative breakdown guard must stop the recurrence before it
        wanders onto the unexcited unit-precision floor; ``chain-h3`` in
        float32 is the near-breakdown case (β_3 about 1e-6·|α| there)."""
        structure, params, clamps, state, eq = _oracle_case(bunch, rng_key)
        H = oracle.epsilon_hessian(eq.quad)
        g0 = oracle.epsilon_gradient_at_zero(eq.quad)
        eigs, weights = oracle.gradient_weights(H, g0)
        excited = oracle.excited_eigenvalues(H, g0)
        eta, steps = 0.05, 5
        expected_fbar = oracle.weighted_relaxed_fraction(eigs, weights, eta, steps)

        jax.config.update("jax_enable_x64", dtype == "float64")
        try:
            cast = lambda t: jax.tree_util.tree_map(  # noqa: E731
                lambda x: jnp.asarray(x, getattr(jnp, dtype)), t
            )
            params_c, clamps_c = cast(params), cast(clamps)
            state_c = initialize_graph_state(
                structure, BATCH, rng_key, clamps_c, params=params_c
            )
            spectrum = epsilon_spectrum(
                params_c, state_c, clamps_c, structure, iters=30, key=rng_key
            )
        finally:
            jax.config.update("jax_enable_x64", False)

        np.testing.assert_allclose(spectrum.lambda_max, eigs[-1], rtol=1e-3)
        np.testing.assert_allclose(spectrum.lambda_min, excited.min(), rtol=1e-3)
        fbar = weighted_relaxed_fraction(spectrum, eta, steps)
        np.testing.assert_allclose(fbar, expected_fbar, atol=1e-3)
        assert not spectrum.random_start
        if bunch != "prior-source":
            # d_y = 3 excited modes: the guard freezes the recurrence early.
            assert spectrum.k < spectrum.iters, (bunch, dtype, spectrum.k)
