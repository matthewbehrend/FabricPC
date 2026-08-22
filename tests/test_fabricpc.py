#!/usr/bin/env python3
"""
Core test suite for FabricPC-JAX.

Covers graph construction, validation, inference, training, forward methods,
complex graph topologies, and identity node behavior.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from fabricpc.core.types import NodeState, NodeParams, GraphState
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.inference import InferenceSGD
import optax
from fabricpc.training import make_train_step
from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    TanhActivation,
    SigmoidActivation,
)
from fabricpc.core.initializers import XavierInitializer
from fabricpc.core.energy import GaussianEnergy
from conftest import with_inference


@pytest.fixture
def sample_graph_structure(rng_key):
    """Fixture providing a sample graph structure with initialized parameters."""
    input_node = Linear(shape=(10,), name="input")
    hidden1 = Linear(
        shape=(20,),
        activation=ReLUActivation(),
        weight_init=XavierInitializer(),
        name="hidden1",
    )
    hidden2 = Linear(shape=(15,), activation=TanhActivation(), name="hidden2")
    output_node = Linear(shape=(5,), activation=SigmoidActivation(), name="output")

    structure = graph(
        nodes=[input_node, hidden1, hidden2, output_node],
        edges=[
            Edge(source=input_node, target=hidden1.slot("in")),
            Edge(source=hidden1, target=hidden2.slot("in")),
            Edge(source=hidden2, target=output_node.slot("in")),
            Edge(source=hidden1, target=output_node.slot("in")),  # skip connection
        ],
        task_map=TaskMap(x=input_node, y=output_node),
        inference=InferenceSGD(),
    )

    params = initialize_params(structure, rng_key)
    return params, structure


class TestGraphConstruction:
    """Test suite for graph construction and validation."""

    def test_graph_construction_with_slots(self, sample_graph_structure):
        """Test building a graph with slot validation and node classes."""
        params, structure = sample_graph_structure

        assert len(structure.nodes) == 4
        assert len(structure.edges) == 4

        hidden1_node = structure.nodes["hidden1"]
        assert "in" in hidden1_node.node_info.slots
        assert hidden1_node.node_info.slots["in"].is_multi_input

        output_node = structure.nodes["output"]
        assert output_node.node_info.in_degree == 2

        assert "hidden1" in params.nodes
        node_params = params.nodes["hidden1"]
        assert isinstance(node_params, NodeParams)
        assert "weights" in node_params._fields
        assert "biases" in node_params._fields
        assert "input->hidden1:in" in node_params.weights

    def test_invalid_slot_rejection(self):
        """Test that invalid slot connections are rejected."""
        b_node = Linear(shape=(5,), name="b")

        with pytest.raises(KeyError, match="no slot"):
            b_node.slot("invalid_slot")


class TestValidation:
    """Test suite for validation and error handling."""

    def test_duplicate_node_names_raise(self):
        """Test that duplicate node names raise an error."""
        x1 = Linear(shape=(2,), activation=SigmoidActivation(), name="x")
        x2 = Linear(shape=(2,), activation=SigmoidActivation(), name="x")

        with pytest.raises(ValueError, match="Duplicate node name"):
            graph(
                nodes=[x1, x2],
                edges=[],
                task_map=TaskMap(),
                inference=InferenceSGD(),
            )

    def test_self_edge_disallowed(self):
        """Test that self-edges are not allowed."""
        n1 = Linear(shape=(2,), activation=SigmoidActivation(), name="n1")

        with pytest.raises(ValueError, match="Self-edge"):
            graph(
                nodes=[n1],
                edges=[Edge(source=n1, target=n1.slot("in"))],
                task_map=TaskMap(),
                inference=InferenceSGD(),
            )

    def test_nonexistent_node_in_edge(self):
        """Test that edges referencing non-existent nodes raise an error."""
        a = Linear(shape=(2,), name="a")
        nonexistent = Linear(shape=(2,), name="nonexistent")

        with pytest.raises(ValueError, match="not found|does not exist"):
            graph(
                nodes=[a],
                edges=[Edge(source=a, target=nonexistent.slot("in"))],
                task_map=TaskMap(),
                inference=InferenceSGD(),
            )


class TestInference:
    """Test suite for inference and gradient computation."""

    @pytest.fixture
    def inference_data(self, sample_graph_structure, rng_key):
        """Prepare data for inference tests."""
        params, structure = sample_graph_structure
        batch_size = 32
        input_shape = structure.nodes["input"].node_info.shape
        output_shape = structure.nodes["output"].node_info.shape

        rng_key, x_key, y_key, state_key = jax.random.split(rng_key, 4)

        x_data = jax.random.normal(x_key, (batch_size, *input_shape))
        y_data = jax.random.normal(y_key, (batch_size, *output_shape))

        clamps = {"input": x_data, "output": y_data}

        return {
            "params": params,
            "structure": structure,
            "batch_size": batch_size,
            "clamps": clamps,
            "state_key": state_key,
        }

    def test_inference_with_local_gradients(self, inference_data):
        """Test inference loop with local Jacobian-based gradients."""
        params = inference_data["params"]
        structure = inference_data["structure"]
        batch_size = inference_data["batch_size"]
        clamps = inference_data["clamps"]
        state_key = inference_data["state_key"]

        initial_state = initialize_graph_state(
            structure,
            batch_size,
            state_key,
            clamps=clamps,
            params=params,
        )

        assert "nodes" in initial_state._fields
        assert isinstance(initial_state.nodes, dict)

        struct_1step = with_inference(structure, eta_infer=0.1, infer_steps=1)
        state_after_1_step = struct_1step.config["inference"].run_inference(
            params, initial_state, clamps, struct_1step
        )

        struct_20step = with_inference(structure, eta_infer=0.1, infer_steps=20)
        final_state = struct_20step.config["inference"].run_inference(
            params, initial_state, clamps, struct_20step
        )

        assert "hidden1" in final_state.nodes
        assert "latent_grad" in final_state.nodes["hidden1"]._fields
        assert final_state.nodes["hidden1"].latent_grad.shape == (batch_size, 20)

        initial_energy = sum(
            jnp.sum(state_after_1_step.nodes[name].energy)
            for name in structure.nodes
            if structure.nodes[name].node_info.in_degree > 0
        )
        final_energy = sum(
            jnp.sum(final_state.nodes[name].energy)
            for name in structure.nodes
            if structure.nodes[name].node_info.in_degree > 0
        )

        assert final_energy < initial_energy

    def test_local_weight_updates(self, inference_data):
        """Test that weight updates use local gradients."""
        params = inference_data["params"]
        structure = inference_data["structure"]
        batch_size = inference_data["batch_size"]
        clamps = inference_data["clamps"]
        state_key = inference_data["state_key"]

        initial_state = initialize_graph_state(
            structure,
            batch_size,
            state_key,
            clamps=clamps,
            params=params,
        )
        struct_10step = with_inference(structure, eta_infer=0.1, infer_steps=10)
        final_state = struct_10step.config["inference"].run_inference(
            params, initial_state, clamps, struct_10step
        )

        grads = compute_local_weight_gradients(params, final_state, structure)

        assert grads.nodes.keys() == params.nodes.keys()

        for node_name, node in structure.nodes.items():
            if node.node_info.in_degree > 0:
                node_grads = grads.nodes[node_name]
                assert isinstance(node_grads, NodeParams)

                for edge_key in params.nodes[node_name].weights.keys():
                    assert edge_key in node_grads.weights.keys()
                    weight_grad = node_grads.weights[edge_key]
                    w = params.nodes[node_name].weights[edge_key]
                    assert weight_grad.shape == w.shape


class TestTraining:
    """Test suite for training functionality."""

    def test_complete_training_step(self, sample_graph_structure, rng_key):
        """Test a complete training step with local learning."""
        params, structure = sample_graph_structure
        batch_size = 8
        input_shape = structure.nodes["input"].node_info.shape
        output_shape = structure.nodes["output"].node_info.shape

        rng_key, x_key, y_key = jax.random.split(rng_key, 3)

        batch = {
            "x": jax.random.normal(x_key, (batch_size, *input_shape)),
            "y": jax.random.normal(y_key, (batch_size, *output_shape)),
        }

        optimizer = optax.adam(0.01)
        opt_state = optimizer.init(params)

        step = make_train_step(structure, optimizer)
        new_params, new_opt_state, metrics, final_state = step(
            params, opt_state, batch, rng_key
        )
        energy = metrics["energy"]

        for node_name in ["hidden1", "hidden2", "output"]:
            edge_key = next(iter(structure.nodes[node_name].node_info.in_edges))
            w_old = params.nodes[node_name].weights[edge_key]
            w_new = new_params.nodes[node_name].weights[edge_key]
            diff = jnp.max(jnp.abs(w_new - w_old))
            assert diff > 0, f"Weights not updated for {node_name}"

        assert not jnp.isnan(energy)
        assert energy > 0


class TestForwardMethods:
    """Test suite for node forward methods."""

    @pytest.fixture
    def forward_setup(self, sample_graph_structure, rng_key):
        """Setup for forward method tests."""
        params, structure = sample_graph_structure
        batch_size = 4

        node_names = list(structure.nodes.keys())
        rng_keys = jax.random.split(rng_key, len(node_names))

        nodes = {}
        for i, (node_name, node) in enumerate(structure.nodes.items()):
            node_info = node.node_info
            full_shape = (batch_size, *node_info.shape)
            nodes[node_name] = NodeState(
                z_latent=jax.random.normal(rng_keys[i], full_shape),
                latent_grad=jnp.zeros(full_shape),
                z_mu=jnp.zeros(full_shape),
                error=jnp.zeros(full_shape),
                energy=jnp.zeros((batch_size,)),
            )

        state = GraphState(nodes=nodes, batch_size=batch_size)
        return params, structure, state

    def test_forward_and_latent_grads_shapes(self, forward_setup):
        """Test that forward_and_latent_grads returns correct shapes."""
        params, structure, state = forward_setup

        for node_name, node in structure.nodes.items():
            node_info = node.node_info
            if node_info.in_degree > 0:
                node_class = node_info.node_class
                node_state = state.nodes[node_name]

                edge_inputs = {}
                for edge_key in node_info.in_edges:
                    in_edge_info = structure.edges[edge_key]
                    edge_inputs[edge_key] = state.nodes[in_edge_info.source].z_latent

                new_state, input_grads, self_grad = node_class.forward_and_latent_grads(
                    params.nodes[node_name],
                    edge_inputs,
                    node_state,
                    node_info,
                    is_clamped=False,
                )

                assert new_state.z_mu.shape == node_state.z_latent.shape
                assert new_state.error.shape == node_state.z_latent.shape

                # self_grad must match z_latent shape and carry finite values.
                # forward_and_latent_grads must NOT mutate state.latent_grad — that's
                # the callsite's responsibility — so the returned NodeState's
                # latent_grad must equal the input's exactly.
                assert self_grad.shape == node_state.z_latent.shape
                assert jnp.all(jnp.isfinite(self_grad))
                assert jnp.array_equal(new_state.latent_grad, node_state.latent_grad)

                for edge_key in node_info.in_edges:
                    edge_info = structure.edges[edge_key]
                    source_shape = structure.nodes[edge_info.source].node_info.shape
                    expected_shape = (state.batch_size, *source_shape)
                    assert input_grads[edge_key].shape == expected_shape

    def test_forward_and_weight_grads_shapes(self, forward_setup):
        """Test that forward_and_weight_grads returns correct shapes."""
        params, structure, state = forward_setup

        for node_name, node in structure.nodes.items():
            node_info = node.node_info
            if node_info.in_degree > 0:
                node_class = node_info.node_class
                node_state = state.nodes[node_name]
                node_params = params.nodes[node_name]

                edge_inputs = {}
                for edge_key in node_info.in_edges:
                    in_edge_info = structure.edges[edge_key]
                    edge_inputs[edge_key] = state.nodes[in_edge_info.source].z_latent

                new_state, param_grads = node_class.forward_and_weight_grads(
                    node_params, edge_inputs, node_state, node_info
                )

                for edge_key in node_params.weights:
                    assert (
                        param_grads.weights[edge_key].shape
                        == node_params.weights[edge_key].shape
                    )

                for bias_key in node_params.biases:
                    assert (
                        param_grads.biases[bias_key].shape
                        == node_params.biases[bias_key].shape
                    )


class TestComplexGraphs:
    """Test more complex graph structures."""

    def test_skip_connection_graph(self, rng_key):
        """Test graph with skip connections."""
        input_node = Linear(shape=(10,), name="input")
        h1 = Linear(shape=(20,), activation=ReLUActivation(), name="h1")
        h2 = Linear(shape=(15,), activation=ReLUActivation(), name="h2")
        output_node = Linear(shape=(5,), name="output")

        structure = graph(
            nodes=[input_node, h1, h2, output_node],
            edges=[
                Edge(source=input_node, target=h1.slot("in")),
                Edge(source=h1, target=h2.slot("in")),
                Edge(source=h2, target=output_node.slot("in")),
                Edge(source=input_node, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        output_edges = structure.nodes["output"].node_info.in_edges
        assert len(output_edges) == 2

        batch_size = 8
        x = jax.random.normal(rng_key, (batch_size, 10))
        y = jax.random.normal(rng_key, (batch_size, 5))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps=clamps,
            params=params,
        )

        struct_mod = with_inference(structure, eta_infer=0.1, infer_steps=10)
        final_state = struct_mod.config["inference"].run_inference(
            params, state, clamps, struct_mod
        )

        assert not jnp.any(jnp.isnan(final_state.nodes["output"].z_mu))

    def test_multi_input_node(self, rng_key):
        """Test node with multiple inputs from different sources."""
        a = Linear(shape=(5,), name="a")
        b = Linear(shape=(4,), name="b")
        c = Linear(shape=(3,), name="c")
        merger = Linear(shape=(6,), name="merger")

        structure = graph(
            nodes=[a, b, c, merger],
            edges=[
                Edge(source=a, target=merger.slot("in")),
                Edge(source=b, target=merger.slot("in")),
                Edge(source=c, target=merger.slot("in")),
            ],
            task_map=TaskMap(x=a, y=merger),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        assert structure.nodes["merger"].node_info.in_degree == 3
        assert len(structure.nodes["merger"].node_info.in_edges) == 3
        assert len(params.nodes["merger"].weights) == 3


class TestIdentityNode:
    """Test IdentityNode behavior."""

    @pytest.mark.parametrize("num_inputs", [1, 3])
    def test_output_equals_sum_of_inputs(self, rng_key, num_inputs):
        """z_mu should equal sum of inputs."""
        from fabricpc.core.types import NodeInfo
        from fabricpc.core.initializers import NormalInitializer

        batch_size, node_shape = 4, (10,)
        full_shape = (batch_size,) + node_shape

        keys = jax.random.split(rng_key, num_inputs + 1)
        edge_keys = [f"src{i}->node:in" for i in range(num_inputs)]
        inputs = {
            k: jax.random.normal(keys[i], full_shape) for i, k in enumerate(edge_keys)
        }

        state = NodeState(
            z_latent=jax.random.normal(keys[-1], full_shape),
            z_mu=jnp.zeros(full_shape),
            error=jnp.zeros(full_shape),
            energy=jnp.zeros((batch_size,)),
            latent_grad=jnp.zeros(full_shape),
        )
        node_info = NodeInfo(
            name="node",
            shape=node_shape,
            node_type="identity",
            node_class=IdentityNode,
            node_config={
                "name": "node",
                "shape": node_shape,
                "type": "identity",
                "scale": 1.0,
            },
            activation=IdentityActivation(),
            energy=GaussianEnergy(),
            latent_init=NormalInitializer(),
            weight_init=None,
            slots={"in": None},
            in_degree=len(edge_keys),
            out_degree=0,
            in_edges=tuple(edge_keys),
            out_edges=(),
        )
        params = NodeParams(weights={}, biases={})

        new_state = IdentityNode.forward(params, inputs, state, node_info)

        expected = sum(inputs.values())
        np.testing.assert_allclose(new_state.z_mu, expected, rtol=1e-5)


class TestFractionalEpochs:
    """Test fractional epoch support in PC training."""

    def test_fractional_epoch_runs(self, rng_key):
        """train with num_epochs=0.5 runs fewer batches than a full epoch."""
        from fabricpc.training import train

        x = IdentityNode(shape=(4,), name="x")
        h = Linear(shape=(8,), activation=TanhActivation(), name="h")
        y = Linear(shape=(2,), name="y")

        structure = graph(
            nodes=[x, h, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=h, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=InferenceSGD(eta_infer=0.1, infer_steps=3),
        )
        params = initialize_params(structure, rng_key)
        optimizer = optax.adam(1e-3)

        x_data = jax.random.normal(rng_key, (16, 4))
        y_data = jax.random.normal(rng_key, (16, 2))
        loader = [(x_data[:8], y_data[:8]), (x_data[8:], y_data[8:])]

        iters_half = []
        train(
            params,
            structure,
            loader,
            optimizer,
            {"num_epochs": 0.5},
            rng_key,
            verbose=False,
            iter_callback=lambda ctx: iters_half.append(1) or ctx.metrics,
        )

        iters_full = []
        train(
            params,
            structure,
            loader,
            optimizer,
            {"num_epochs": 1},
            rng_key,
            verbose=False,
            iter_callback=lambda ctx: iters_full.append(1) or ctx.metrics,
        )

        assert len(iters_half) < len(iters_full)
        assert len(iters_half) == 1  # 0.5 * 2 batches = 1 batch
