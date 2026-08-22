#!/usr/bin/env python3
"""
Test suite for N-Dimensional Tensor Support.

Verifies that multi-dimensional node shapes (2D images, 3D NHWC tensors)
work correctly through graph construction, inference, and training.
"""

import jax
import jax.numpy as jnp

from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.inference import InferenceSGD
import optax
from fabricpc.training import make_train_step
from fabricpc.core.activations import ReLUActivation, TanhActivation, SigmoidActivation
from conftest import with_inference


class TestNDimShapes:
    """Test suite for n-dimensional tensor shapes."""

    def test_2d_shape(self, rng_key):
        """Test node with 2D shape: shape=(28, 28) - image without channels."""
        node_image = Linear(shape=(28, 28), name="image")
        node_hidden = Linear(
            shape=(128,), activation=ReLUActivation(), flatten_input=True, name="hidden"
        )
        node_output = Linear(shape=(10,), name="output")

        structure = graph(
            nodes=[node_image, node_hidden, node_output],
            edges=[
                Edge(source=node_image, target=node_hidden.slot("in")),
                Edge(source=node_hidden, target=node_output.slot("in")),
            ],
            task_map=TaskMap(x=node_image, y=node_output),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)

        assert structure.nodes["image"].node_info.shape == (28, 28)
        assert structure.nodes["hidden"].node_info.shape == (128,)

        hidden_weights = params.nodes["hidden"].weights["image->hidden:in"]
        assert hidden_weights.shape == (784, 128)

        hidden_bias = params.nodes["hidden"].biases["b"]
        assert hidden_bias.shape == (1, 128)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 28, 28))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"image": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        struct_mod = with_inference(structure, eta_infer=0.1, infer_steps=5)
        final_state = struct_mod.config["inference"].run_inference(
            params, state, clamps, struct_mod
        )

        assert final_state.nodes["image"].z_latent.shape == (batch_size, 28, 28)
        assert final_state.nodes["hidden"].z_latent.shape == (batch_size, 128)
        assert final_state.nodes["output"].z_latent.shape == (batch_size, 10)

    def test_3d_shape(self, rng_key):
        """Test node with 3D shape: shape=(28, 28, 1) - image with channels (NHWC)."""
        node_image = Linear(shape=(28, 28, 1), name="image")
        node_hidden = Linear(
            shape=(64,), activation=TanhActivation(), flatten_input=True, name="hidden"
        )
        node_output = Linear(shape=(10,), name="output")

        structure = graph(
            nodes=[node_image, node_hidden, node_output],
            edges=[
                Edge(source=node_image, target=node_hidden.slot("in")),
                Edge(source=node_hidden, target=node_output.slot("in")),
            ],
            task_map=TaskMap(x=node_image, y=node_output),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)

        assert structure.nodes["image"].node_info.shape == (28, 28, 1)

        hidden_weights = params.nodes["hidden"].weights["image->hidden:in"]
        assert hidden_weights.shape == (784, 64)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 28, 28, 1))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"image": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        struct_mod = with_inference(structure, eta_infer=0.1, infer_steps=5)
        final_state = struct_mod.config["inference"].run_inference(
            params, state, clamps, struct_mod
        )

        assert final_state.nodes["image"].z_latent.shape == (batch_size, 28, 28, 1)
        assert final_state.nodes["hidden"].z_latent.shape == (batch_size, 64)

    def test_mixed_shapes_in_graph(self, rng_key):
        """Test mixed shapes: 2D input -> 1D hidden -> 1D output."""
        node_image = Linear(shape=(28, 28), name="image")
        node_hidden1 = Linear(
            shape=(256,),
            activation=ReLUActivation(),
            flatten_input=True,
            name="hidden1",
        )
        node_hidden2 = Linear(shape=(128,), activation=ReLUActivation(), name="hidden2")
        node_output = Linear(shape=(10,), name="output")

        structure = graph(
            nodes=[node_image, node_hidden1, node_hidden2, node_output],
            edges=[
                Edge(source=node_image, target=node_hidden1.slot("in")),
                Edge(source=node_hidden1, target=node_hidden2.slot("in")),
                Edge(source=node_hidden2, target=node_output.slot("in")),
            ],
            task_map=TaskMap(x=node_image, y=node_output),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)

        assert params.nodes["hidden1"].weights["image->hidden1:in"].shape == (784, 256)
        assert params.nodes["hidden2"].weights["hidden1->hidden2:in"].shape == (
            256,
            128,
        )
        assert params.nodes["output"].weights["hidden2->output:in"].shape == (128, 10)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 28, 28))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"image": x, "output": y}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        struct_mod = with_inference(structure, eta_infer=0.1, infer_steps=5)
        final_state = struct_mod.config["inference"].run_inference(
            params, state, clamps, struct_mod
        )

        assert final_state.nodes["image"].z_latent.shape == (batch_size, 28, 28)
        assert final_state.nodes["hidden1"].z_latent.shape == (batch_size, 256)
        assert final_state.nodes["hidden2"].z_latent.shape == (batch_size, 128)
        assert final_state.nodes["output"].z_latent.shape == (batch_size, 10)


class TestNDimTraining:
    """Test that training works with n-dimensional shapes."""

    def test_training_with_2d_input(self, rng_key):
        """Test complete training step with 2D image input."""
        node_image = Linear(shape=(28, 28), name="image")
        node_hidden = Linear(
            shape=(64,),
            activation=SigmoidActivation(),
            flatten_input=True,
            name="hidden",
        )
        node_output = Linear(shape=(10,), name="output")

        structure = graph(
            nodes=[node_image, node_hidden, node_output],
            edges=[
                Edge(source=node_image, target=node_hidden.slot("in")),
                Edge(source=node_hidden, target=node_output.slot("in")),
            ],
            task_map=TaskMap(x=node_image, y=node_output),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)

        optimizer = optax.adam(0.01)
        opt_state = optimizer.init(params)

        batch_size = 8
        batch = {
            "x": jax.random.normal(rng_key, (batch_size, 28, 28)),
            "y": jax.random.normal(rng_key, (batch_size, 10)),
        }

        step = make_train_step(structure, optimizer)
        new_params, new_opt_state, metrics, final_state = step(
            params, opt_state, batch, rng_key
        )

        energy = metrics["energy"]
        assert not jnp.isnan(energy)
        assert energy > 0

        old_w = params.nodes["hidden"].weights["image->hidden:in"]
        new_w = new_params.nodes["hidden"].weights["image->hidden:in"]
        assert not jnp.allclose(old_w, new_w)
