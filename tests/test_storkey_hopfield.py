"""Tests for StorkeyHopfield associative memory node."""

import pytest
import jax
import jax.numpy as jnp

from fabricpc.nodes import Linear, IdentityNode, StorkeyHopfield
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.activations import SigmoidActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.initializers import XavierInitializer, NormalInitializer
from fabricpc.training import make_train_step
from conftest import with_inference
import optax

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


def _build_hopfield_graph_1d(D=32, infer_steps=10, eta_infer=0.1):
    """Build a simple Linear -> StorkeyHopfield -> Linear graph with 1D shapes."""
    input_node = IdentityNode(shape=(D,), name="input")
    hopfield = StorkeyHopfield(shape=(D,), name="hopfield")
    output = Linear(shape=(5,), activation=SigmoidActivation(), name="output")

    structure = graph(
        nodes=[input_node, hopfield, output],
        edges=[
            Edge(source=input_node, target=hopfield.slot("in")),
            Edge(source=hopfield, target=output.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
    )
    return structure


# =========================================================================
# Shape tests
# =========================================================================


class TestShapes:
    """Verify parameter and state shapes."""

    def test_init_params_1d(self, rng_key):
        """1D shape: W is (D, D), no bias by default, learnable hopfield_strength."""
        D = 32
        structure = _build_hopfield_graph_1d(D=D)
        params = initialize_params(structure, rng_key)

        hop_params = params.nodes["hopfield"]
        edge_keys = list(hop_params.weights.keys())
        assert len(edge_keys) == 1
        assert hop_params.weights[edge_keys[0]].shape == (D, D)
        assert "b" not in hop_params.biases
        assert "hopfield_strength" in hop_params.biases
        assert hop_params.biases["hopfield_strength"].shape == ()

    def test_forward_1d(self, rng_key):
        """Forward pass produces correctly shaped state."""
        D = 32
        structure = _build_hopfield_graph_1d(D=D)
        params = initialize_params(structure, rng_key)
        batch_size = 4

        rng_key, state_key = jax.random.split(rng_key)
        x = jax.random.normal(state_key, (batch_size, D))
        y = jax.random.normal(state_key, (batch_size, 5))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size=batch_size,
            rng_key=rng_key,
            clamps=clamps,
            params=params,
        )

        hop_state = state.nodes["hopfield"]
        assert hop_state.z_latent.shape == (batch_size, D)
        assert hop_state.z_mu.shape == (batch_size, D)


# =========================================================================
# Energy tests
# =========================================================================


class TestEnergy:
    """Verify energy behavior during inference."""

    def test_attractor_energy_formula(self, rng_key):
        """Value-pin the attractor term: energy() minus the base PC term
        equals s * (1/2D) z^T (W^2 - W) z computed independently, where W is
        the prepared (symmetrized) weight matrix and s the effective
        (softplus'd) strength."""
        D = 8
        structure = _build_hopfield_graph_1d(D=D)
        params = initialize_params(structure, rng_key)
        info = structure.nodes["hopfield"].node_info
        hop_params = params.nodes["hopfield"]

        batch_size = 3
        probe = jax.random.normal(jax.random.PRNGKey(2), (batch_size, D))
        inputs = {info.in_edges[0]: probe}
        from fabricpc.core.types import NodeState

        z = jax.random.normal(jax.random.PRNGKey(3), (batch_size, D))
        state = NodeState(
            z_latent=z,
            z_mu=jnp.zeros((batch_size, D)),
            error=jnp.zeros((batch_size, D)),
            energy=jnp.zeros((batch_size,)),
            latent_grad=jnp.zeros((batch_size, D)),
        )
        new_state, _ = StorkeyHopfield.forward_with_aux(hop_params, inputs, state, info)

        energy_obj = info.energy
        pc_term = type(energy_obj).energy(
            new_state.z_latent, new_state.z_mu, energy_obj.config
        )
        edge_key = info.in_edges[0]
        W = StorkeyHopfield._prepare_W(hop_params.weights[edge_key], info.node_config)
        s = jax.nn.softplus(hop_params.biases["hopfield_strength"])
        # (1/2D) z^T (W^2 - W) z, written as its quadratic forms.
        expected_attractor = (0.5 / D) * (
            jnp.einsum("bi,ij,jk,bk->b", z, W, W, z)
            - jnp.einsum("bi,ij,bj->b", z, W, z)
        )
        assert jnp.allclose(
            new_state.energy - pc_term, s * expected_attractor, atol=1e-5
        )

    def test_energy_decreases_during_inference(self, rng_key):
        """Energy should decrease (or not increase) over more inference steps."""
        D = 16
        structure = _build_hopfield_graph_1d(D=D, infer_steps=50, eta_infer=0.05)
        params = initialize_params(structure, rng_key)
        batch_size = 4

        rng_key, x_key = jax.random.split(rng_key)
        x = jax.random.normal(x_key, (batch_size, D))
        y = jax.random.normal(x_key, (batch_size, 5))
        clamps = {"input": x, "output": y}

        # Few steps
        struct_few = with_inference(structure, infer_steps=5, eta_infer=0.05)
        state_init = initialize_graph_state(
            struct_few,
            batch_size=batch_size,
            rng_key=rng_key,
            clamps=clamps,
            params=params,
        )
        state_few = run_inference(params, state_init, clamps, struct_few)
        energy_few = sum(
            float(jnp.sum(state_few.nodes[n].energy))
            for n in structure.nodes
            if structure.nodes[n].node_info.in_degree > 0
        )

        # More steps
        struct_many = with_inference(structure, infer_steps=50, eta_infer=0.05)
        state_init = initialize_graph_state(
            struct_many,
            batch_size=batch_size,
            rng_key=rng_key,
            clamps=clamps,
            params=params,
        )
        state_many = run_inference(params, state_init, clamps, struct_many)
        energy_many = sum(
            float(jnp.sum(state_many.nodes[n].energy))
            for n in structure.nodes
            if structure.nodes[n].node_info.in_degree > 0
        )

        assert (
            energy_many <= energy_few + 1e-3
        ), f"Energy did not decrease: {energy_few} -> {energy_many}"


# =========================================================================
# Gradient tests
# =========================================================================


class TestGradients:
    """Verify autodiff weight gradients through combined energy."""

    def test_gradient_shapes_and_nonzero(self, rng_key):
        """Weight gradients match param shapes and Hopfield W gradient is nonzero."""
        D = 16
        structure = _build_hopfield_graph_1d(D=D, infer_steps=5)
        params = initialize_params(structure, rng_key)
        batch_size = 4

        rng_key, x_key = jax.random.split(rng_key)
        x = jax.random.normal(x_key, (batch_size, D))
        y = jax.random.normal(x_key, (batch_size, 5))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size=batch_size,
            rng_key=rng_key,
            clamps=clamps,
            params=params,
        )
        final_state = run_inference(params, state, clamps, structure)
        gradients = compute_local_weight_gradients(params, final_state, structure)

        hop_grads = gradients.nodes["hopfield"]
        hop_params = params.nodes["hopfield"]

        # Shapes match
        for key in hop_params.weights:
            assert key in hop_grads.weights
            assert hop_grads.weights[key].shape == hop_params.weights[key].shape

        for key in hop_params.biases:
            assert key in hop_grads.biases
            assert hop_grads.biases[key].shape == hop_params.biases[key].shape

        # W gradient is nonzero
        w_key = list(hop_grads.weights.keys())[0]
        assert jnp.any(
            hop_grads.weights[w_key] != 0.0
        ), "Hopfield W gradient is all zeros"

    def test_hopfield_strength_gradient(self, rng_key):
        """Learnable hopfield_strength receives a finite gradient."""
        D = 16
        structure = _build_hopfield_graph_1d(D=D, infer_steps=5)
        params = initialize_params(structure, rng_key)
        batch_size = 4

        rng_key, x_key = jax.random.split(rng_key)
        x = jax.random.normal(x_key, (batch_size, D))
        y = jax.random.normal(x_key, (batch_size, 5))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size=batch_size,
            rng_key=rng_key,
            clamps=clamps,
            params=params,
        )
        final_state = run_inference(params, state, clamps, structure)
        gradients = compute_local_weight_gradients(params, final_state, structure)

        strength_grad = gradients.nodes["hopfield"].biases["hopfield_strength"]
        assert jnp.isfinite(strength_grad), "hopfield_strength gradient is not finite"


# =========================================================================
# Symmetry and Config tests
# =========================================================================


class TestSymmetryAndConfig:
    """Verify W constraints and constructor config options."""

    def test_prepare_w_constraints(self, rng_key):
        """_prepare_W enforces symmetry and zero diagonal as configured."""
        D = 16
        W = jax.random.normal(rng_key, (D, D))

        # Symmetry
        W_sym = StorkeyHopfield._prepare_W(
            W, {"enforce_symmetry": True, "zero_diagonal": False}
        )
        assert jnp.allclose(W_sym, W_sym.T, atol=1e-6)

        # Zero diagonal
        W_zd = StorkeyHopfield._prepare_W(
            W, {"enforce_symmetry": False, "zero_diagonal": True}
        )
        assert jnp.allclose(jnp.diag(W_zd), 0.0, atol=1e-6)

    def test_config_options(self, rng_key):
        """Fixed hopfield_strength excluded from biases; use_bias=False omits bias."""
        D = 16
        input_node = IdentityNode(shape=(D,), name="input")
        hopfield = StorkeyHopfield(
            shape=(D,), name="hopfield", hopfield_strength=1.0, use_bias=False
        )
        output = Linear(shape=(5,), activation=SigmoidActivation(), name="output")

        structure = graph(
            nodes=[input_node, hopfield, output],
            edges=[
                Edge(source=input_node, target=hopfield.slot("in")),
                Edge(source=hopfield, target=output.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)

        assert "hopfield_strength" not in params.nodes["hopfield"].biases
        assert "b" not in params.nodes["hopfield"].biases


# =========================================================================
# Integration tests
# =========================================================================


class TestIntegration:
    """End-to-end test with full PC graph."""

    def test_four_node_graph(self, rng_key):
        """4-node graph: input -> Linear -> Hopfield -> Linear (classifier)."""
        D_hidden = 16
        D_hop = 16
        n_classes = 5

        input_node = IdentityNode(shape=(32,), name="input")
        hidden = Linear(
            shape=(D_hidden,),
            activation=SigmoidActivation(),
            name="hidden",
            weight_init=XavierInitializer(),
        )
        hopfield = StorkeyHopfield(
            shape=(D_hop,),
            name="hopfield",
            weight_init=NormalInitializer(mean=0.0, std=0.01),
        )
        output = Linear(
            shape=(n_classes,),
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            name="class",
            weight_init=XavierInitializer(),
        )

        structure = graph(
            nodes=[input_node, hidden, hopfield, output],
            edges=[
                Edge(source=input_node, target=hidden.slot("in")),
                Edge(source=hidden, target=hopfield.slot("in")),
                Edge(source=hopfield, target=output.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output),
            inference=InferenceSGD(eta_infer=0.05, infer_steps=10),
        )

        params = initialize_params(structure, rng_key)
        batch_size = 8

        rng_key, x_key, train_key = jax.random.split(rng_key, 3)
        x = jax.random.normal(x_key, (batch_size, 32))
        y = jax.nn.one_hot(
            jax.random.randint(x_key, (batch_size,), 0, n_classes), n_classes
        )
        # Batch keys are TASK keys (x/y), not node names: a batch keyed by
        # node names matches nothing in the task_map, produces zero clamps,
        # and now raises instead of silently training unclamped.
        batch = {"x": x, "y": y}

        optimizer = optax.adam(1e-3)
        opt_state = optimizer.init(params)

        step = make_train_step(structure, optimizer)
        new_params, opt_state, metrics, _ = step(params, opt_state, batch, train_key)
        assert jnp.isfinite(metrics["energy"])
        # Params should have changed
        edge_key = list(params.nodes["hopfield"].weights.keys())[0]
        assert not jnp.allclose(
            params.nodes["hopfield"].weights[edge_key],
            new_params.nodes["hopfield"].weights[edge_key],
            atol=1e-8,
        ), "Hopfield W did not update"
