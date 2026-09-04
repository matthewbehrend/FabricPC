"""
Tests for EPCInference — error-parameterized predictive coding.

ePC relaxes the prediction errors ε and derives the latents by a forward
pass along ``structure.schedule`` (z_latent = z_mu + ε), taking one global
``jax.value_and_grad`` of the total in_degree > 0 energy with respect to the
ε pytree per step. These tests pin: the ε = 0 <-> feedforward-init
correspondence, gradient correctness against the closed form and a
hand-rolled ``jax.grad``, energy descent, the sPC equilibrium equivalence
(including an unclamped top-down prior — the case that distinguishes
ε-relaxed sources from frozen ones), the ``forward_from_error`` branch
coverage (clamped/unclamped x source/internal), the ``begin_segment``
resync (ε := z_latent - z_mu at the carried latents, so ePC continues
exactly from any incoming state), cyclic warm-start semantics, muPC input
scaling (and the recorded divergence of sPC+muPC's preconditioned fixed
point from the true energy minimum ePC reaches), insertion-order
independence, and the z_latent = z_mu + ε invariant of the finalized state.
"""

import jax
import jax.numpy as jnp
import pytest

from conftest import inject_biases, total_energy, with_inference
from fabricpc.core import EPCInference, InferenceSGD, run_inference
from fabricpc.core.activations import (
    IdentityActivation,
    SoftmaxActivation,
    TanhActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy, graph_energy
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.state_ops import update_node_in_state
from fabricpc.core.topology import Edge
from fabricpc.core.types import GraphState
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear, StorkeyHopfield
from fabricpc.nodes.identity import IdentityNode

W_INIT = NormalInitializer(std=0.3)


def _chain(inference=None, scaling=None):
    """x (Identity source) -> h (Linear) -> y (Linear), identity activations."""
    x = IdentityNode(shape=(5,), name="x")
    h = Linear(
        shape=(4,), name="h", activation=IdentityActivation(), weight_init=W_INIT
    )
    y = Linear(
        shape=(3,), name="y", activation=IdentityActivation(), weight_init=W_INIT
    )
    return graph(
        nodes=[x, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=inference or EPCInference(eta_infer=0.05, infer_steps=5),
        scaling=scaling,
    )


def _epsilon_grad_sq_norm(params, state, clamps, structure):
    """Squared norm of the true energy gradient in ε coordinates at the
    state's latents. begin_segment resyncs ε := z_latent - z_mu, so the
    measurement is taken exactly at the incoming latents; the ε ↔ z_latent
    Jacobian is unit-triangular, so this is zero iff the state is a
    stationary point of the energy in z coordinates too."""
    synced = EPCInference.begin_segment(params, state, clamps, structure)
    synced = EPCInference.zero_grads(params, synced, clamps, structure)
    with_grads = EPCInference.forward_value_and_grad(params, synced, clamps, structure)
    return sum(
        float(jnp.sum(with_grads.nodes[name].latent_grad ** 2))
        for name in EPCInference._relaxed_errors(structure, clamps)
    )


class TestZeroErrorIsFeedforward:
    @pytest.mark.parametrize("clamp_output", [False, True])
    def test_derive_states_preserves_feedforward_init(self, rng_key, clamp_output):
        """At ε = 0 the derived states equal FeedforwardStateInit's output:
        initialization is the derived forward at zero error."""
        structure = _chain()
        params = initialize_params(structure, rng_key)
        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 5))
        clamps = {"x": x}
        if clamp_output:
            clamps["y"] = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 3))

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps, params=params
        )
        derived = EPCInference.derive_states(params, state, clamps, structure)

        for name in structure.nodes:
            assert jnp.allclose(
                derived.nodes[name].z_latent, state.nodes[name].z_latent, atol=1e-6
            ), f"{name}: derive_states at ε=0 moved z_latent off the feedforward init"
        if not clamp_output:
            assert jnp.allclose(
                derived.nodes["y"].z_latent, derived.nodes["y"].z_mu, atol=1e-6
            )


class TestGradientCorrectness:
    def _setup(self, rng_key):
        structure = _chain()
        params = initialize_params(structure, rng_key)
        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 5))
        y = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 3))
        clamps = {"x": x, "y": y}
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps, params=params
        )
        # Seed a nonzero relaxed error on the single unclamped node.
        eps = jax.random.normal(jax.random.PRNGKey(2), (batch_size, 4))
        state = update_node_in_state(state, "h", error=eps)
        return structure, params, clamps, state, x, y, eps

    def test_matches_closed_form(self, rng_key):
        """2-layer linear chain: ∇_ε E = ε_h - residual @ W_yᵀ, where the
        residual is y - z_mu_y at the derived latent z_h = μ_h + ε_h."""
        structure, params, clamps, state, x, y, eps = self._setup(rng_key)

        new_state = EPCInference.forward_value_and_grad(
            params, state, clamps, structure
        )

        W_h = params.nodes["h"].weights["x->h:in"]
        b_h = params.nodes["h"].biases["b"]
        W_y = params.nodes["y"].weights["h->y:in"]
        b_y = params.nodes["y"].biases["b"]
        mu_h = x @ W_h + b_h
        z_h = mu_h + eps
        residual = y - (z_h @ W_y + b_y)
        expected = eps - residual @ W_y.T

        assert jnp.allclose(new_state.nodes["h"].latent_grad, expected, atol=1e-5)

    def test_matches_hand_rolled_jax_grad(self, rng_key):
        structure, params, clamps, state, x, y, eps = self._setup(rng_key)

        W_h = params.nodes["h"].weights["x->h:in"]
        b_h = params.nodes["h"].biases["b"]
        W_y = params.nodes["y"].weights["h->y:in"]
        b_y = params.nodes["y"].biases["b"]

        def energy(eps_h):
            z_h = (x @ W_h + b_h) + eps_h
            e_h = 0.5 * jnp.sum(eps_h**2)
            e_y = 0.5 * jnp.sum((y - (z_h @ W_y + b_y)) ** 2)
            return e_h + e_y

        expected = jax.grad(energy)(eps)
        new_state = EPCInference.forward_value_and_grad(
            params, state, clamps, structure
        )
        assert jnp.allclose(new_state.nodes["h"].latent_grad, expected, atol=1e-5)

    def test_grads_accumulate_into_latent_grad(self, rng_key):
        """∇_ε E is added to latent_grad, never replaces it."""
        structure, params, clamps, state, *_ = self._setup(rng_key)
        sentinel = jnp.full((4, 4), 1.75)
        state = update_node_in_state(state, "h", latent_grad=sentinel)

        with_sentinel = EPCInference.forward_value_and_grad(
            params, state, clamps, structure
        )
        without = EPCInference.forward_value_and_grad(
            params,
            update_node_in_state(state, "h", latent_grad=jnp.zeros((4, 4))),
            clamps,
            structure,
        )
        assert jnp.allclose(
            with_sentinel.nodes["h"].latent_grad,
            without.nodes["h"].latent_grad + sentinel,
            atol=1e-6,
        )


class TestEnergyDescent:
    def test_energy_decreases_over_steps(self, rng_key):
        structure = _chain(inference=EPCInference(eta_infer=0.05, infer_steps=1))
        params = initialize_params(structure, rng_key)
        batch_size = 4
        clamps = {
            "x": jax.random.normal(rng_key, (batch_size, 5)),
            "y": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 3)),
        }
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps, params=params
        )

        energies = []
        inference = structure.config["inference"]
        for _ in range(6):
            state = inference.run_inference(params, state, clamps, structure)
            energies.append(float(total_energy(state, structure)))

        assert all(
            b < a for a, b in zip(energies, energies[1:])
        ), f"energy did not decrease monotonically: {energies}"


class TestSPCEquivalence:
    def _convex_graph(self, inference):
        """Strictly convex DAG with an unclamped top-down prior: shapes chosen
        so the stacked residual Jacobian has full column rank (W_y injective
        on z_h, W_p injective on z_prior), giving one global minimizer. The
        larger weight std keeps the prior direction (curvature ~
        sigma_min(W_y W_p)^2) well conditioned so both solvers converge
        within the step budget."""
        w_init = NormalInitializer(std=0.8)
        x = IdentityNode(shape=(5,), name="x")
        prior = Linear(shape=(3,), name="prior", weight_init=w_init)
        h = Linear(
            shape=(4,), name="h", activation=IdentityActivation(), weight_init=w_init
        )
        y = Linear(
            shape=(6,), name="y", activation=IdentityActivation(), weight_init=w_init
        )
        return graph(
            nodes=[x, prior, h, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=prior, target=h.slot("in")),
                Edge(source=h, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=inference,
        )

    @pytest.mark.slow
    def test_shared_equilibrium_and_weight_grads(self, rng_key):
        batch_size = 3
        x = jax.random.normal(rng_key, (batch_size, 5))
        y = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 6))
        clamps = {"x": x, "y": y}

        finals = {}
        # ePC's rate sits below sPC's local rate: the ε-space curvature is
        # amplified by the triangular reparameterization (eta 0.1 diverges
        # here), the constructor's tuning guidance in action.
        for key, inference in (
            ("spc", InferenceSGD(eta_infer=0.1, infer_steps=20000)),
            ("epc", EPCInference(eta_infer=0.05, infer_steps=20000)),
        ):
            structure = self._convex_graph(inference)
            params = initialize_params(structure, rng_key)
            state = initialize_graph_state(
                structure, batch_size, rng_key, clamps, params=params
            )
            final = inference.run_inference(params, state, clamps, structure)
            finals[key] = (structure, params, final)

        spc_structure, spc_params, spc = finals["spc"]
        _, epc_params, epc = finals["epc"]

        # Both finals are stationary points of the shared energy, not merely
        # near each other: the true ε-coordinate gradient vanishes at both.
        assert _epsilon_grad_sq_norm(spc_params, spc, clamps, spc_structure) < 1e-6
        assert _epsilon_grad_sq_norm(epc_params, epc, clamps, spc_structure) < 1e-6

        for name in spc_structure.nodes:
            assert jnp.allclose(
                spc.nodes[name].z_latent, epc.nodes[name].z_latent, atol=1e-4
            ), f"{name}: z_latent equilibria differ"
            # z_mu and energy agree for in_degree > 0 nodes (the energy's
            # domain). A source's z_mu is bookkeeping outside E: sPC re-syncs
            # it to the relaxed latent, ePC holds it at the init constant.
            if spc_structure.nodes[name].node_info.in_degree > 0:
                assert jnp.allclose(
                    spc.nodes[name].z_mu, epc.nodes[name].z_mu, atol=1e-4
                ), f"{name}: z_mu equilibria differ"
                assert jnp.allclose(
                    spc.nodes[name].energy, epc.nodes[name].energy, atol=1e-4
                ), f"{name}: energies differ"

        grads_spc = compute_local_weight_gradients(spc_params, spc, spc_structure)
        grads_epc = compute_local_weight_gradients(epc_params, epc, spc_structure)
        for name in grads_spc.nodes:
            for edge_key, g in grads_spc.nodes[name].weights.items():
                assert jnp.allclose(
                    g, grads_epc.nodes[name].weights[edge_key], atol=1e-4
                ), f"weight grad differs at {name}/{edge_key}"
            for bias_key, g in grads_spc.nodes[name].biases.items():
                assert jnp.allclose(
                    g, grads_epc.nodes[name].biases[bias_key], atol=1e-4
                ), f"bias grad differs at {name}/{bias_key}"


class TestBeginSegmentResync:
    def test_resync_preserves_incoming_latents_on_dag(self, rng_key):
        """begin_segment then derive_states reproduces arbitrary incoming
        latents exactly on a DAG: the resync computes ε := z_latent - z_mu at
        the carried latents, and the derive inverts it node by node in
        schedule order. This is what makes a distribution initializer's
        random internal latents (or a previous solver segment's state)
        survive the handoff instead of being overwritten."""
        structure = _chain()
        params = initialize_params(structure, rng_key)
        batch_size = 4
        clamps = {"x": jax.random.normal(rng_key, (batch_size, 5))}
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps, params=params
        )
        # Perturb every unclamped latent off the feedforward point — the
        # state a distribution initializer would hand over.
        for name, shape in (("h", (batch_size, 4)), ("y", (batch_size, 3))):
            perturbed = state.nodes[name].z_latent + jax.random.normal(
                jax.random.PRNGKey(hash(name) % 2**31), shape
            )
            state = update_node_in_state(state, name, z_latent=perturbed)

        synced = EPCInference.begin_segment(params, state, clamps, structure)
        derived = EPCInference.derive_states(params, synced, clamps, structure)
        for name in structure.nodes:
            assert jnp.allclose(
                derived.nodes[name].z_latent,
                state.nodes[name].z_latent,
                atol=1e-6,
            ), f"{name}: resync + derive moved the incoming latent"

    def test_spc_to_epc_handoff_preserves_latents(self, rng_key):
        """A zero-step ePC segment after sPC (begin_segment + finalize only)
        returns sPC's latents unchanged: nothing of the sPC refinement —
        including its final latent update — is lost at the boundary."""
        spc = InferenceSGD(eta_infer=0.05, infer_steps=7)
        epc = EPCInference(eta_infer=0.01, infer_steps=0)
        structure = _chain(inference=spc)
        params = initialize_params(structure, rng_key)
        batch_size = 4
        clamps = {
            "x": jax.random.normal(rng_key, (batch_size, 5)),
            "y": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 3)),
        }
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps, params=params
        )
        after_spc = spc.run_inference(params, state, clamps, structure)
        after_epc = epc.run_inference(params, after_spc, clamps, structure)
        for name in structure.nodes:
            assert jnp.allclose(
                after_epc.nodes[name].z_latent,
                after_spc.nodes[name].z_latent,
                atol=1e-6,
            ), f"{name}: the ePC boundary moved sPC's latents"


class TestNonlinearEquivalence:
    @pytest.mark.slow
    def test_tanh_cross_entropy_equilibrium(self, rng_key):
        """sPC/ePC equivalence off the linear-Gaussian case: tanh hidden
        layers and a CrossEntropy output energy. Both solvers start from the
        same feedforward init; the shared stationarity check (true ε-gradient
        ~ 0 at both finals) plus matching latents pins one shared minimum,
        and the local weight gradients agree there."""

        def build(inference):
            x = IdentityNode(shape=(4,), name="x")
            h1 = Linear(
                shape=(3,), name="h1", activation=TanhActivation(), weight_init=W_INIT
            )
            h2 = Linear(
                shape=(3,), name="h2", activation=TanhActivation(), weight_init=W_INIT
            )
            y = Linear(
                shape=(2,),
                name="y",
                activation=SoftmaxActivation(),
                energy=CrossEntropyEnergy(),
                weight_init=W_INIT,
            )
            return graph(
                nodes=[x, h1, h2, y],
                edges=[
                    Edge(source=x, target=h1.slot("in")),
                    Edge(source=h1, target=h2.slot("in")),
                    Edge(source=h2, target=y.slot("in")),
                ],
                task_map=TaskMap(x=x, y=y),
                inference=inference,
            )

        batch_size = 3
        clamps = {
            "x": jax.random.normal(rng_key, (batch_size, 4)),
            "y": jax.nn.one_hot(jnp.array([0, 1, 0]), 2),
        }

        finals = {}
        for key, inference in (
            ("spc", InferenceSGD(eta_infer=0.1, infer_steps=8000)),
            ("epc", EPCInference(eta_infer=0.05, infer_steps=8000)),
        ):
            structure = build(inference)
            params = initialize_params(structure, rng_key)
            state = initialize_graph_state(
                structure, batch_size, rng_key, clamps, params=params
            )
            final = inference.run_inference(params, state, clamps, structure)
            finals[key] = (structure, params, final)

        structure, params, spc = finals["spc"]
        _, _, epc = finals["epc"]

        assert _epsilon_grad_sq_norm(params, spc, clamps, structure) < 1e-6
        assert _epsilon_grad_sq_norm(params, epc, clamps, structure) < 1e-6
        for name in structure.nodes:
            assert jnp.allclose(
                spc.nodes[name].z_latent, epc.nodes[name].z_latent, atol=1e-3
            ), f"{name}: z_latent equilibria differ"

        grads_spc = compute_local_weight_gradients(params, spc, structure)
        grads_epc = compute_local_weight_gradients(params, epc, structure)
        for name in grads_spc.nodes:
            for edge_key, g in grads_spc.nodes[name].weights.items():
                assert jnp.allclose(
                    g, grads_epc.nodes[name].weights[edge_key], atol=1e-3
                ), f"weight grad differs at {name}/{edge_key}"


class TestForwardFromErrorBranches:
    def test_cross_entropy_clamped_output(self, rng_key):
        """Clamped internal node: z_latent stays the clamp, error and energy
        are derived, and the output loss's gradient reaches upstream ε."""
        x = IdentityNode(shape=(5,), name="x")
        h = Linear(
            shape=(4,), name="h", activation=TanhActivation(), weight_init=W_INIT
        )
        y = Linear(
            shape=(3,),
            name="y",
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            weight_init=W_INIT,
        )
        structure = graph(
            nodes=[x, h, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=h, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=EPCInference(eta_infer=0.01, infer_steps=3),
        )
        params = initialize_params(structure, rng_key)
        batch_size = 4
        y_onehot = jax.nn.one_hot(jnp.array([0, 1, 2, 1]), 3)
        clamps = {"x": jax.random.normal(rng_key, (batch_size, 5)), "y": y_onehot}
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps, params=params
        )

        new_state = EPCInference.forward_value_and_grad(
            params, state, clamps, structure
        )
        y_state = new_state.nodes["y"]
        assert jnp.array_equal(y_state.z_latent, y_onehot)
        assert jnp.allclose(y_state.error, y_onehot - y_state.z_mu, atol=1e-6)
        energy_obj = structure.nodes["y"].node_info.energy
        expected = type(energy_obj).energy(y_onehot, y_state.z_mu, energy_obj.config)
        assert jnp.allclose(y_state.energy, expected, atol=1e-6)
        # The output loss's gradient reached the upstream relaxed error.
        assert not jnp.allclose(new_state.nodes["h"].latent_grad, 0.0)

    def test_gaussian_readout_stays_at_zero_error(self, rng_key):
        """An unclamped pure-Gaussian readout at ε = 0 has ∇_ε E =
        precision * ε = 0, so it stays put while upstream ε moves."""
        structure = _chain(inference=EPCInference(eta_infer=0.05, infer_steps=4))
        params = initialize_params(structure, rng_key)
        batch_size = 4
        clamps = {"x": jax.random.normal(rng_key, (batch_size, 5))}
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps, params=params
        )
        final = run_inference(params, state, clamps, structure)
        assert jnp.allclose(final.nodes["y"].error, 0.0, atol=1e-7)
        assert jnp.allclose(final.nodes["y"].z_latent, final.nodes["y"].z_mu, atol=1e-6)

    def test_storkey_hopfield_readout_gets_attractor_gradient(self, rng_key):
        """A Hopfield readout retains its nonzero attractor energy and
        receives the attractor gradient even at ε = 0."""
        probe = IdentityNode(shape=(6,), name="probe")
        hop = StorkeyHopfield(
            shape=(6,), name="hop", hopfield_strength=2.0, use_bias=False
        )
        structure = graph(
            nodes=[probe, hop],
            edges=[Edge(source=probe, target=hop.slot("in"))],
            task_map=TaskMap(x=probe, y=hop),
            inference=EPCInference(eta_infer=0.01, infer_steps=3),
        )
        params = initialize_params(structure, rng_key)
        batch_size = 4
        clamps = {"probe": jax.random.normal(rng_key, (batch_size, 6))}
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps, params=params
        )

        new_state = EPCInference.forward_value_and_grad(
            params, state, clamps, structure
        )
        hop_state = new_state.nodes["hop"]
        assert not jnp.allclose(hop_state.energy, 0.0)
        assert not jnp.allclose(hop_state.latent_grad, 0.0)

        final = run_inference(params, state, clamps, structure)
        assert not jnp.any(jnp.isnan(final.nodes["hop"].z_latent))

    def test_unclamped_prior_source_relaxes(self, rng_key):
        """A top-down prior's z_mu stays the init constant while its ε
        receives gradient through downstream z_mu."""
        prior = Linear(shape=(3,), name="prior", weight_init=W_INIT)
        h = Linear(
            shape=(4,), name="h", activation=IdentityActivation(), weight_init=W_INIT
        )
        structure = graph(
            nodes=[prior, h],
            edges=[Edge(source=prior, target=h.slot("in"))],
            task_map=TaskMap(y=h),
            inference=EPCInference(eta_infer=0.05, infer_steps=3),
        )
        params = initialize_params(structure, rng_key)
        batch_size = 4
        clamps = {"h": jax.random.normal(rng_key, (batch_size, 4))}
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps, params=params
        )
        init_z_mu = state.nodes["prior"].z_mu

        new_state = EPCInference.forward_value_and_grad(
            params, state, clamps, structure
        )
        assert not jnp.allclose(new_state.nodes["prior"].latent_grad, 0.0)

        final = run_inference(params, state, clamps, structure)
        prior_final = final.nodes["prior"]
        assert jnp.array_equal(prior_final.z_mu, init_z_mu)
        assert jnp.allclose(
            prior_final.z_latent, prior_final.z_mu + prior_final.error, atol=1e-6
        )
        # ε moved, so the latent moved off the init.
        assert not jnp.allclose(prior_final.z_latent, state.nodes["prior"].z_latent)

    def test_int_token_embedding_graph(self, rng_key):
        """Int-dtype token clamps never enter the AD pytree; the clamped
        source's state passes through forward_from_error untouched."""
        from fabricpc.nodes.transformer_v2 import EmbeddingNode

        tokens = IdentityNode(shape=(4,), name="tokens")
        emb = EmbeddingNode(shape=(4, 8), name="emb", vocab_size=11, embed_dim=8)
        out = Linear(
            shape=(4, 8),
            name="out",
            activation=IdentityActivation(),
            weight_init=W_INIT,
        )
        structure = graph(
            nodes=[tokens, emb, out],
            edges=[
                Edge(source=tokens, target=emb.slot("in")),
                Edge(source=emb, target=out.slot("in")),
            ],
            task_map=TaskMap(x=tokens, y=out),
            inference=EPCInference(eta_infer=0.01, infer_steps=2),
        )
        params = initialize_params(structure, rng_key)
        batch_size = 3
        token_ids = jnp.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 0, 1]], dtype=jnp.int32
        )
        clamps = {"tokens": token_ids}
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps, params=params
        )

        assert "tokens" not in EPCInference._relaxed_errors(structure, clamps)
        final = run_inference(params, state, clamps, structure)
        assert final.nodes["tokens"].z_latent.dtype == jnp.int32
        assert jnp.array_equal(final.nodes["tokens"].z_latent, token_ids)
        expected_mu = params.nodes["emb"].weights["embeddings"][token_ids]
        assert jnp.allclose(final.nodes["emb"].z_mu, expected_mu, atol=1e-6)


class TestCyclicSchedule:
    def _cycle(self, unroll, infer_steps):
        x = IdentityNode(shape=(5,), name="x")
        a = Linear(
            shape=(4,), name="a", activation=TanhActivation(), weight_init=W_INIT
        )
        b = Linear(
            shape=(4,), name="b", activation=TanhActivation(), weight_init=W_INIT
        )
        y = Linear(
            shape=(3,), name="y", activation=IdentityActivation(), weight_init=W_INIT
        )
        return graph(
            nodes=[x, a, b, y],
            edges=[
                Edge(source=x, target=a.slot("in")),
                Edge(source=a, target=b.slot("in")),
                Edge(source=b, target=a.slot("in")),
                Edge(source=b, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=EPCInference(eta_infer=0.02, infer_steps=infer_steps),
            unroll=unroll,
        )

    def test_cyclic_smoke_jit_and_energy(self, rng_key):
        structure = self._cycle(unroll=2, infer_steps=1)
        params = initialize_params(structure, rng_key)
        batch_size = 3
        clamps = {
            "x": jax.random.normal(rng_key, (batch_size, 5)),
            "y": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 3)),
        }
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps, params=params
        )
        inference = structure.config["inference"]

        @jax.jit
        def step(p, s):
            return inference.run_inference(p, s, clamps, structure)

        energies = []
        for _ in range(5):
            state = step(params, state)
            energies.append(float(total_energy(state, structure)))
        assert all(jnp.isfinite(jnp.array(energies)))
        assert energies[-1] < energies[0]

    def test_clamped_cycle_member_keeps_clamp(self, rng_key):
        """A clamp on a cycle member holds through repeated schedule visits:
        derive_states never relaxes it, and the run stays finite."""
        structure = self._cycle(unroll=2, infer_steps=2)
        params = initialize_params(structure, rng_key)
        batch_size = 3
        a_clamp = jax.random.normal(jax.random.PRNGKey(3), (batch_size, 4))
        clamps = {
            "x": jax.random.normal(rng_key, (batch_size, 5)),
            "a": a_clamp,
        }
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps, params=params
        )
        final = structure.config["inference"].run_inference(
            params, state, clamps, structure
        )
        assert jnp.array_equal(final.nodes["a"].z_latent, a_clamp)
        assert jnp.isfinite(float(total_energy(final, structure)))
        # b relaxed against the clamped a.
        assert not jnp.allclose(final.nodes["b"].error, 0.0)

    def test_warm_start_two_steps_u1_differs_from_one_step_u2(self, rng_key):
        """Each step starts from the carried state (truncated warm start), so
        two steps at U=1 is a different computation from one step at U=2."""
        batch_size = 3
        x = jax.random.normal(rng_key, (batch_size, 5))
        y = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 3))
        clamps = {"x": x, "y": y}

        results = {}
        for unroll, infer_steps in ((1, 2), (2, 1)):
            structure = self._cycle(unroll=unroll, infer_steps=infer_steps)
            params = initialize_params(structure, rng_key)
            state = initialize_graph_state(
                structure, batch_size, rng_key, clamps, params=params
            )
            final = structure.config["inference"].run_inference(
                params, state, clamps, structure
            )
            results[(unroll, infer_steps)] = final

        assert not jnp.allclose(
            results[(1, 2)].nodes["a"].z_latent,
            results[(2, 1)].nodes["a"].z_latent,
            atol=1e-6,
        )


class TestMuPCScaling:
    def test_derived_z_mu_matches_manual_scaling(self, rng_key):
        from fabricpc.core.initializers import MuPCInitializer

        x = IdentityNode(shape=(5,), name="x")
        h = Linear(
            shape=(4,),
            name="h",
            activation=IdentityActivation(),
            weight_init=MuPCInitializer(),
        )
        y = Linear(
            shape=(3,),
            name="y",
            activation=IdentityActivation(),
            weight_init=MuPCInitializer(),
        )
        structure = graph(
            nodes=[x, h, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=h, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=EPCInference(eta_infer=0.01, infer_steps=2),
            scaling=MuPCConfig(),
        )
        params = initialize_params(structure, rng_key)
        batch_size = 4
        x_data = jax.random.normal(rng_key, (batch_size, 5))
        clamps = {"x": x_data}
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps, params=params
        )

        derived = EPCInference.derive_states(params, state, clamps, structure)
        scale = structure.nodes["h"].node_info.scaling_config.forward_scale["x->h:in"]
        expected_mu = (scale * x_data) @ params.nodes["h"].weights["x->h:in"] + (
            params.nodes["h"].biases["b"]
        )
        assert jnp.allclose(derived.nodes["h"].z_mu, expected_mu, atol=1e-6)


class TestMuPCDivergence:
    @pytest.mark.slow
    def test_spc_mupc_fixed_point_differs_from_true_minimum(self, rng_key):
        """Under muPC scaling, sPC and ePC have different fixed points, and
        the divergence is directional: ePC descends the true gradient of the
        input-scaled energy and stops at its stationary point, while sPC's
        per-hop updates carry ``topdown_grad_scale = a * jacobian_gain``
        (tanh's jacobian_gain ~ 1.26), so its fixed point zeroes the
        preconditioned sum, not the true gradient. The gain is taken from
        the edge's TARGET activation, so the shifted fixed point needs an
        unclamped node (h1) feeding a tanh node (h2). Measured with one
        shared criterion — the true ε-gradient norm at each solver's final
        state — ePC's vanishes and sPC's does not."""
        w_init = NormalInitializer(std=0.3)

        def build(inference):
            x = IdentityNode(shape=(5,), name="x")
            h1 = Linear(
                shape=(4,), name="h1", activation=TanhActivation(), weight_init=w_init
            )
            h2 = Linear(
                shape=(4,), name="h2", activation=TanhActivation(), weight_init=w_init
            )
            y = Linear(
                shape=(3,),
                name="y",
                activation=IdentityActivation(),
                weight_init=w_init,
            )
            return graph(
                nodes=[x, h1, h2, y],
                edges=[
                    Edge(source=x, target=h1.slot("in")),
                    Edge(source=h1, target=h2.slot("in")),
                    Edge(source=h2, target=y.slot("in")),
                ],
                task_map=TaskMap(x=x, y=y),
                inference=inference,
                scaling=MuPCConfig(),
            )

        batch_size = 3
        clamps = {
            "x": jax.random.normal(rng_key, (batch_size, 5)),
            "y": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 3)),
        }

        finals = {}
        for key, inference in (
            ("spc", InferenceSGD(eta_infer=0.1, infer_steps=5000)),
            ("epc", EPCInference(eta_infer=0.05, infer_steps=5000)),
        ):
            structure = build(inference)
            params = initialize_params(structure, rng_key)
            state = initialize_graph_state(
                structure, batch_size, rng_key, clamps, params=params
            )
            finals[key] = (
                structure,
                params,
                inference.run_inference(params, state, clamps, structure),
            )

        structure, params, epc = finals["epc"]
        _, _, spc = finals["spc"]
        epc_grad = _epsilon_grad_sq_norm(params, epc, clamps, structure)
        spc_grad = _epsilon_grad_sq_norm(params, spc, clamps, structure)
        assert epc_grad < 1e-8, "ePC did not reach the true energy's minimum"
        assert spc_grad > 100 * max(epc_grad, 1e-10), (
            "sPC+muPC's preconditioned fixed point unexpectedly coincides "
            "with the true energy minimum"
        )
        assert not jnp.allclose(
            spc.nodes["h1"].z_latent, epc.nodes["h1"].z_latent, atol=1e-4
        )


class TestComputeNewError:
    def test_decay_and_gradient_step(self):
        """ε update formula: ε * (1 - eta * decay) - eta * grad."""
        from fabricpc.core.types import NodeState

        error = jnp.array([[2.0, -4.0]])
        grad = jnp.array([[0.5, 1.0]])
        node_state = NodeState(
            z_latent=jnp.zeros((1, 2)),
            z_mu=jnp.zeros((1, 2)),
            error=error,
            energy=jnp.zeros((1,)),
            latent_grad=grad,
        )
        config = {"eta_infer": 0.1, "latent_decay": 0.5}
        new_error = EPCInference.compute_new_error("n", node_state, config)
        expected = error * (1.0 - 0.1 * 0.5) - 0.1 * grad
        assert jnp.allclose(new_error, expected)


class TestOrderIndependence:
    def test_one_step_grads_insertion_order_independent(self, rng_key):
        """The global ε gradient is a property of the graph, not of node
        insertion order: the derived forward walks the topological schedule."""

        def build(order):
            x = IdentityNode(shape=(5,), name="x")
            a = Linear(
                shape=(4,), name="a", activation=TanhActivation(), weight_init=W_INIT
            )
            b = Linear(
                shape=(4,), name="b", activation=TanhActivation(), weight_init=W_INIT
            )
            y = Linear(
                shape=(3,),
                name="y",
                activation=IdentityActivation(),
                weight_init=W_INIT,
            )
            by_name = {"x": x, "a": a, "b": b, "y": y}
            return graph(
                nodes=[by_name[n] for n in order],
                edges=[
                    Edge(source=x, target=a.slot("in")),
                    Edge(source=x, target=b.slot("in")),
                    Edge(source=a, target=y.slot("in")),
                    Edge(source=b, target=y.slot("in")),
                ],
                task_map=TaskMap(x=x, y=y),
                inference=EPCInference(eta_infer=0.05, infer_steps=1),
            )

        structure_a = build(("x", "a", "b", "y"))
        structure_b = build(("y", "b", "a", "x"))
        params = initialize_params(structure_a, rng_key)
        batch_size = 3
        clamps = {
            "x": jax.random.normal(rng_key, (batch_size, 5)),
            "y": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 3)),
        }
        state_a = initialize_graph_state(
            structure_a, batch_size, rng_key, clamps, params=params
        )
        # Same per-node states wrapped for the permuted structure.
        state_b = GraphState(
            nodes={name: state_a.nodes[name] for name in structure_b.nodes},
            batch_size=batch_size,
        )

        grads_a = EPCInference.forward_value_and_grad(
            params, state_a, clamps, structure_a
        )
        grads_b = EPCInference.forward_value_and_grad(
            params, state_b, clamps, structure_b
        )
        for name in structure_a.nodes:
            assert jnp.allclose(
                grads_a.nodes[name].latent_grad,
                grads_b.nodes[name].latent_grad,
                atol=1e-6,
            ), f"{name}: one-step ε grads depend on insertion order"


class TestFinalStateInvariant:
    def test_z_latent_equals_z_mu_plus_error_after_run(self, rng_key):
        structure = _chain(inference=EPCInference(eta_infer=0.05, infer_steps=5))
        params = initialize_params(structure, rng_key)
        batch_size = 4
        clamps = {
            "x": jax.random.normal(rng_key, (batch_size, 5)),
            "y": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 3)),
        }
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps, params=params
        )
        final = run_inference(params, state, clamps, structure)

        for name in structure.nodes:
            node_state = final.nodes[name]
            if name in clamps:
                assert jnp.allclose(node_state.z_latent, clamps[name], atol=1e-6)
            assert jnp.allclose(
                node_state.z_latent.astype(node_state.z_mu.dtype),
                node_state.z_mu + node_state.error,
                atol=1e-6,
            ), f"{name}: z_latent != z_mu + error after run_inference"


class TestBackpropCorrespondence:
    """1-step ePC against backprop (Goemaere et al., Theorem C.9, Case 1).

    x(4) -> h1(3, tanh) -> h2(3, tanh) -> y(2), biases drawn, Gaussian or
    softmax + cross-entropy output. In FabricPC the activation sits in the
    receiving node's prediction (μ_h2 = tanh(z_h1 W + b)), so the tanh'
    factor attaches to the downstream node's pre-activation in the hand
    recursion below.
    """

    @staticmethod
    def _mlp(output, eta, steps):
        x = IdentityNode(shape=(4,), name="x")
        h1 = Linear(
            shape=(3,), name="h1", activation=TanhActivation(), weight_init=W_INIT
        )
        h2 = Linear(
            shape=(3,), name="h2", activation=TanhActivation(), weight_init=W_INIT
        )
        if output == "gaussian":
            y = Linear(
                shape=(2,),
                name="y",
                activation=IdentityActivation(),
                weight_init=W_INIT,
            )
        else:
            y = Linear(
                shape=(2,),
                name="y",
                activation=SoftmaxActivation(),
                energy=CrossEntropyEnergy(),
                weight_init=W_INIT,
            )
        return graph(
            nodes=[x, h1, h2, y],
            edges=[
                Edge(source=x, target=h1.slot("in")),
                Edge(source=h1, target=h2.slot("in")),
                Edge(source=h2, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=EPCInference(eta_infer=eta, infer_steps=steps),
        )

    def _setup(self, rng_key, output, eta=0.05, steps=1, batch=3):
        structure = self._mlp(output, eta, steps)
        params = inject_biases(
            initialize_params(structure, rng_key), jax.random.fold_in(rng_key, 7)
        )
        x = jax.random.normal(rng_key, (batch, 4))
        if output == "gaussian":
            y = jax.random.normal(jax.random.PRNGKey(1), (batch, 2))
        else:
            labels = jax.random.randint(jax.random.PRNGKey(1), (batch,), 0, 2)
            y = jax.nn.one_hot(labels, 2)
        clamps = {"x": x, "y": y}
        state = initialize_graph_state(structure, batch, rng_key, clamps, params=params)
        return structure, params, clamps, state

    @staticmethod
    def _backprop_activation_grads(params, x, y, output):
        """dL/dz_h1, dL/dz_h2 at the feedforward point, and a1 = μ_h1."""
        W1, b1 = params.nodes["h1"].weights["x->h1:in"], params.nodes["h1"].biases["b"]
        W2, b2 = params.nodes["h2"].weights["h1->h2:in"], params.nodes["h2"].biases["b"]
        Wy, by = params.nodes["y"].weights["h2->y:in"], params.nodes["y"].biases["b"]
        a1 = jnp.tanh(x @ W1 + b1)
        a2 = jnp.tanh(a1 @ W2 + b2)
        pre_y = a2 @ Wy + by
        if output == "gaussian":
            delta_y = pre_y - y  # precision 1: d(½‖y − μ‖²)/dμ
        else:
            delta_y = jax.nn.softmax(pre_y, axis=-1) - y
        g_h2 = delta_y @ Wy.T
        g_h1 = (g_h2 * (1.0 - a2**2)) @ W2.T
        return {"h1": g_h1, "h2": g_h2}, a1

    @pytest.mark.parametrize("output", ["gaussian", "ce"])
    def test_epsilon_grad_at_zero_is_backprop_activation_grad(self, rng_key, output):
        """∇_ε E at ε = 0 is the backprop activation gradient, exactly."""
        structure, params, clamps, state = self._setup(rng_key, output)
        state = EPCInference.begin_segment(params, state, clamps, structure)
        state = EPCInference.zero_grads(params, state, clamps, structure)
        with_grads = EPCInference.forward_value_and_grad(
            params, state, clamps, structure
        )
        g, _ = self._backprop_activation_grads(params, clamps["x"], clamps["y"], output)
        for name in ("h1", "h2"):
            assert jnp.allclose(
                with_grads.nodes[name].latent_grad, g[name], atol=1e-6
            ), name

    @pytest.mark.parametrize("output", ["gaussian", "ce"])
    def test_one_step_error_is_minus_eta_backprop_grad(self, rng_key, output):
        """After one step, ε = −η·g on every hidden node and z_h1 = a1 − η·g_h1
        (h2's latent is re-derived at the perturbed h1, so only h1's is the
        literal backprop step)."""
        structure, params, clamps, state = self._setup(
            rng_key, output, eta=0.05, steps=1
        )
        final = structure.config["inference"].run_inference(
            params, state, clamps, structure
        )
        g, a1 = self._backprop_activation_grads(
            params, clamps["x"], clamps["y"], output
        )
        for name in ("h1", "h2"):
            assert jnp.allclose(
                final.nodes[name].error, -0.05 * g[name], atol=1e-6
            ), name
        assert jnp.allclose(final.nodes["h1"].z_latent, a1 - 0.05 * g["h1"], atol=1e-6)

    @pytest.mark.parametrize("output", ["gaussian", "ce"])
    def test_one_step_weight_grads_are_eta_backprop_first_order(self, rng_key, output):
        """Local weight gradients after one ePC step equal η × backprop's on
        the hidden layers and backprop's on the output layer, to first order
        in η. The O(η²) remainder comes from the node's input latent being
        re-derived at the perturbed upstream state, so a layer whose input is
        the clamp (h1) matches exactly, up to float32, at every η, while h2
        and y show a relative deviation d(η) below 1e-2 at η = 1e-3 that
        grows linearly in η (within a factor of 3 of 10× per decade). Batch
        1, so batch-summed and batch-mean gradients coincide and the
        reference needs no normalization."""
        structure, params, clamps, _ = self._setup(rng_key, output, batch=1)

        def loss(p):
            state = initialize_graph_state(structure, 1, rng_key, clamps, params=p)
            return graph_energy(state, structure, node_names=["y"])

        g_bp = jax.grad(loss)(params)

        def deviation(eta):
            s = with_inference(
                structure, inference=EPCInference(eta_infer=eta, infer_steps=1)
            )
            state = initialize_graph_state(s, 1, rng_key, clamps, params=params)
            final = s.config["inference"].run_inference(params, state, clamps, s)
            g_pc = compute_local_weight_gradients(params, final, s)
            out = {}
            for name in ("h1", "h2", "y"):
                scale = 1.0 if name == "y" else eta
                for kind in ("weights", "biases"):
                    for key, ref in getattr(g_bp.nodes[name], kind).items():
                        got = getattr(g_pc.nodes[name], kind)[key] / scale
                        out[(name, key)] = float(
                            jnp.linalg.norm(got - ref) / jnp.linalg.norm(ref)
                        )
            return out

        d = {eta: deviation(eta) for eta in (1e-3, 1e-2, 1e-1)}
        for key in d[1e-3]:
            if key[0] == "h1":
                for eta in d:
                    assert d[eta][key] < 1e-3, (key, eta, d[eta][key])
                continue
            assert d[1e-3][key] < 1e-2, (key, d[1e-3][key])
            r1 = d[1e-2][key] / d[1e-3][key]
            r2 = d[1e-1][key] / d[1e-2][key]
            assert 3.0 <= r1 <= 30.0 and 3.0 <= r2 <= 30.0, (key, d[1e-3][key], r1, r2)


class TestRegimeLabel:
    def test_bands(self):
        default = EPCInference()
        assert "backprop-like" in default.regime_label(1.0)
        assert "partially relaxed" in default.regime_label(75.0)
        assert "near PC equilibrium" in EPCInference(
            eta_infer=0.05, infer_steps=100
        ).regime_label(10.0)
        assert default.regime_label(3000.0).startswith("unstable")
        assert "overshooting" in EPCInference(
            eta_infer=0.15, infer_steps=1
        ).regime_label(10.0)
