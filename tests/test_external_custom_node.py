#!/usr/bin/env python3
"""
Test that custom nodes defined outside the fabricpc package can be used
in FabricPC graphs.

This test verifies the external custom-node extension contract: an external
package can subclass NodeBase, implement get_slots/initialize_params/predict
(plus an optional energy override), and integrate with FabricPC's graph
construction, parameter initialization, state initialization, default
autodiff gradient paths, and the ``run_inference`` inference loop.

The custom node defined here (``ScaledSumNode``) is intentionally minimal:
each incoming edge contributes a learned scalar gain on the source's
z_latent; gains are summed elementwise; a learned bias is added; an
activation is applied. Inputs must match the node's output shape (the
contract is strict — no silent reshape). The node's ``energy`` override
weights the functional default by an ``energy_weight`` value from
``node_info.node_config``, demonstrating both the custom-energy-term
surface and the ``**extra_config`` channel that external nodes use for
trace-time-static per-node configuration.

Scope note: this file uses the same import pattern an external package
would use, but the tests live alongside the fabricpc package. The test
verifies the API contract, not import isolation.
"""

import dataclasses
from typing import Dict, Any, Tuple

import pytest
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation, ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import InitializerBase, NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.core.inference import InferenceSGD, run_inference, gather_inputs
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear


class ScaledSumNode(NodeBase):
    """
    Minimal custom node defined outside fabricpc to exercise the extension
    contract.

    Predict: z_mu = activation( sum_e(scale_e * x_e) + bias )
    Energy: GaussianEnergy(z_latent, z_mu) multiplied by node_config["energy_weight"]
    via an ``energy()`` override.

    Per-edge weights are learned scalars stored in ``params.weights[edge_key]``.
    Bias is a learned vector stored at ``params.biases["b"]``. Input shapes
    must match the node's output shape exactly; mismatched shapes raise at
    ``initialize_params`` time.

    The energy multiplier is read from ``node_info.node_config["energy_weight"]``
    in ``energy()``. This is the supported ``**extra_config`` surface for
    trace-time-static per-node configuration.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(std=0.01),
        weight_init=NormalInitializer(std=0.01),
        energy_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            energy_weight=energy_weight,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Single multi-input slot that accepts an arbitrary number of edges."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: InitializerBase,
        config: Dict[str, Any],
    ) -> NodeParams:
        """
        Per-edge learnable scalar gain plus a learned bias vector.

        Raises ValueError if any incoming edge's source shape does not match
        ``node_shape``. The strict contract avoids silent reshape and makes
        shape mismatches surface at graph-construction time.
        """
        for edge_key, in_shape in input_shapes.items():
            if tuple(in_shape) != tuple(node_shape):
                raise ValueError(
                    f"ScaledSumNode requires input shape to match node "
                    f"shape. Edge {edge_key!r}: input {tuple(in_shape)} "
                    f"vs. node {tuple(node_shape)}."
                )

        keys = jax.random.split(key, len(input_shapes) + 1)
        weights = {}
        for i, edge_key in enumerate(input_shapes):
            weights[edge_key] = jax.random.normal(keys[i], ()) * 0.1 + 1.0

        bias = jax.random.normal(keys[-1], node_shape) * 0.01
        return NodeParams(weights=weights, biases={"b": bias})

    @staticmethod
    def predict(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jnp.ndarray, None]:
        """Prediction per the NodeBase.predict contract."""
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape

        pre_activation = jnp.zeros((batch_size,) + out_shape)
        for edge_key, x in inputs.items():
            scale = params.weights[edge_key]
            pre_activation = pre_activation + scale * x
        pre_activation = pre_activation + params.biases["b"]

        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)
        return z_mu, None

    @staticmethod
    def energy(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        aux: None,
        node_info: NodeInfo,
    ) -> jnp.ndarray:
        """Custom energy term: the functional default weighted by config."""
        energy = NodeBase.energy(params, inputs, state, aux, node_info)
        return energy * node_info.node_config.get("energy_weight", 1.0)


def _build_simple_graph(shape=(8,)):
    """Build a 3-node graph: Linear input -> ScaledSumNode -> Linear output."""
    input_node = Linear(shape=shape, name="input")
    custom_node = ScaledSumNode(shape=shape, name="custom")
    output_node = Linear(shape=(4,), name="output")
    structure = graph(
        nodes=[input_node, custom_node, output_node],
        edges=[
            Edge(source=input_node, target=custom_node.slot("in")),
            Edge(source=custom_node, target=output_node.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output_node),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=4),
    )
    return structure


@pytest.fixture
def simple_custom_graph(rng_key):
    """Structure + initialized params for the standard 3-node test graph."""
    structure = _build_simple_graph()
    params = initialize_params(structure, rng_key)
    return structure, params


@pytest.fixture
def simple_custom_state(simple_custom_graph, rng_key):
    """Structure, params, clamps, and initialized GraphState."""
    structure, params = simple_custom_graph
    batch_size = 4
    k_x, k_y, k_state = jax.random.split(rng_key, 3)
    x_data = jax.random.normal(k_x, (batch_size, 8))
    y_data = jax.random.normal(k_y, (batch_size, 4))
    clamps = {"input": x_data, "output": y_data}
    state = initialize_graph_state(
        structure=structure,
        batch_size=batch_size,
        rng_key=k_state,
        clamps=clamps,
        params=params,
    )
    return structure, params, clamps, state


class TestExternalCustomNode:
    """Test suite for the external custom-node extension contract."""

    # -------------------------------------------------------------------
    # Basic construction and shape contract
    # -------------------------------------------------------------------

    def test_custom_node_instantiation(self):
        node = ScaledSumNode(shape=(10,), name="custom")
        assert node.name == "custom"
        assert node.shape == (10,)

    def test_custom_node_slots(self):
        slots = ScaledSumNode.get_slots()
        assert "in" in slots
        assert slots["in"].is_multi_input is True

    def test_initialize_params_rejects_shape_mismatch(self, rng_key):
        """Strict-shape contract: input shape must match node shape."""
        input_node = Linear(shape=(16,), name="input")
        custom_node = ScaledSumNode(shape=(8,), name="custom")
        structure = graph(
            nodes=[input_node, custom_node],
            edges=[Edge(source=input_node, target=custom_node.slot("in"))],
            task_map=TaskMap(x=input_node),
            inference=InferenceSGD(),
        )
        with pytest.raises(ValueError, match="input shape to match node"):
            initialize_params(structure, rng_key)

    # -------------------------------------------------------------------
    # Graph construction, parameter and state initialization
    # -------------------------------------------------------------------

    def test_custom_node_in_graph(self, simple_custom_graph):
        structure, _ = simple_custom_graph
        assert len(structure.nodes) == 3
        assert "custom" in structure.nodes
        assert structure.nodes["custom"].node_info.in_degree == 1
        assert structure.nodes["custom"].node_info.out_degree == 1

    def test_custom_node_params_initialize(self, simple_custom_graph):
        structure, params = simple_custom_graph
        assert "custom" in params.nodes
        custom_params = params.nodes["custom"]
        in_edge_keys = structure.nodes["custom"].node_info.in_edges
        assert len(in_edge_keys) == 1
        for edge_key in in_edge_keys:
            assert edge_key in custom_params.weights
            assert custom_params.weights[edge_key].shape == ()
        assert custom_params.biases["b"].shape == (8,)

    def test_custom_node_state_initialize(self, simple_custom_state):
        structure, params, clamps, state = simple_custom_state
        assert "custom" in state.nodes
        assert state.nodes["custom"].z_latent.shape == (4, 8)
        assert state.nodes["custom"].z_mu.shape == (4, 8)

    def test_custom_node_multiple_inputs(self, rng_key):
        input1 = Linear(shape=(8,), name="input1")
        input2 = Linear(shape=(8,), name="input2")
        custom_node = ScaledSumNode(shape=(8,), name="custom")
        structure = graph(
            nodes=[input1, input2, custom_node],
            edges=[
                Edge(source=input1, target=custom_node.slot("in")),
                Edge(source=input2, target=custom_node.slot("in")),
            ],
            task_map=TaskMap(x=input1),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)
        assert structure.nodes["custom"].node_info.in_degree == 2
        in_edge_keys = structure.nodes["custom"].node_info.in_edges
        assert len(in_edge_keys) == 2
        for edge_key in in_edge_keys:
            assert edge_key in params.nodes["custom"].weights
            assert params.nodes["custom"].weights[edge_key].shape == ()

    def test_custom_node_with_activation(self, rng_key):
        input_node = Linear(shape=(8,), name="input")
        custom_node = ScaledSumNode(
            shape=(8,), name="custom", activation=ReLUActivation()
        )
        structure = graph(
            nodes=[input_node, custom_node],
            edges=[Edge(source=input_node, target=custom_node.slot("in"))],
            task_map=TaskMap(x=input_node),
            inference=InferenceSGD(),
        )
        initialize_params(structure, rng_key)
        assert isinstance(
            structure.nodes["custom"].node_info.activation, ReLUActivation
        )

    # -------------------------------------------------------------------
    # ``**extra_config`` -> ``node_info.node_config`` surface
    # -------------------------------------------------------------------

    def test_extra_config_reaches_node_info(self, rng_key):
        """Kwargs to NodeBase.__init__ land in node_info.node_config."""
        input_node = Linear(shape=(8,), name="input")
        custom_node = ScaledSumNode(
            shape=(8,),
            name="custom",
            energy_weight=2.5,
            shell=1,
            route_prior=0.7,
        )
        structure = graph(
            nodes=[input_node, custom_node],
            edges=[Edge(source=input_node, target=custom_node.slot("in"))],
            task_map=TaskMap(x=input_node),
            inference=InferenceSGD(),
        )
        cfg = structure.nodes["custom"].node_info.node_config
        assert cfg["energy_weight"] == 2.5
        assert cfg["shell"] == 1
        assert cfg["route_prior"] == 0.7

    def test_extra_config_consumed_in_forward(self, simple_custom_state):
        """Two energy_weight values produce energies in the documented ratio."""
        structure, params, clamps, state = simple_custom_state

        custom_node_info = structure.nodes["custom"].node_info
        inputs = gather_inputs(custom_node_info, structure, state)
        custom_state = state.nodes["custom"]
        custom_params = params.nodes["custom"]

        info_a = dataclasses.replace(
            custom_node_info,
            node_config={**dict(custom_node_info.node_config), "energy_weight": 1.0},
        )
        info_b = dataclasses.replace(
            custom_node_info,
            node_config={**dict(custom_node_info.node_config), "energy_weight": 3.0},
        )

        state_a = ScaledSumNode.forward(custom_params, inputs, custom_state, info_a)
        state_b = ScaledSumNode.forward(custom_params, inputs, custom_state, info_b)
        energy_a = jnp.sum(state_a.energy)
        energy_b = jnp.sum(state_b.energy)

        assert jnp.isfinite(energy_a)
        assert jnp.isfinite(energy_b)
        assert jnp.allclose(energy_b, 3.0 * energy_a, rtol=1e-5)

    # -------------------------------------------------------------------
    # Default autodiff gradient paths (what the inference loop calls)
    # -------------------------------------------------------------------

    def test_forward_and_latent_grads(self, simple_custom_state):
        """NodeBase.forward_and_latent_grads -- inference-loop entry point."""
        structure, params, _, state = simple_custom_state
        custom_node_info = structure.nodes["custom"].node_info
        inputs = gather_inputs(custom_node_info, structure, state)
        custom_state = state.nodes["custom"]
        custom_params = params.nodes["custom"]

        new_state, input_grads, self_grad = ScaledSumNode.forward_and_latent_grads(
            custom_params, inputs, custom_state, custom_node_info, is_clamped=False
        )

        assert new_state.z_mu.shape == custom_state.z_latent.shape
        assert new_state.error.shape == custom_state.z_latent.shape
        assert self_grad.shape == custom_state.z_latent.shape
        assert jnp.isfinite(self_grad).all()

        assert set(input_grads.keys()) == set(inputs.keys())
        for edge_key, grad in input_grads.items():
            assert grad.shape == inputs[edge_key].shape
            assert jnp.isfinite(grad).all()

    def test_forward_and_weight_grads(self, simple_custom_state):
        """NodeBase.forward_and_weight_grads -- learning-loop entry point."""
        structure, params, _, state = simple_custom_state
        custom_node_info = structure.nodes["custom"].node_info
        inputs = gather_inputs(custom_node_info, structure, state)
        custom_state = state.nodes["custom"]
        custom_params = params.nodes["custom"]

        new_state, params_grad = ScaledSumNode.forward_and_weight_grads(
            custom_params, inputs, custom_state, custom_node_info
        )

        assert new_state.z_mu.shape == custom_state.z_latent.shape
        assert set(params_grad.weights.keys()) == set(custom_params.weights.keys())
        for edge_key in custom_params.weights:
            assert (
                params_grad.weights[edge_key].shape
                == custom_params.weights[edge_key].shape
            )
            assert jnp.isfinite(params_grad.weights[edge_key]).all()
        assert params_grad.biases["b"].shape == custom_params.biases["b"].shape
        assert jnp.isfinite(params_grad.biases["b"]).all()

    # -------------------------------------------------------------------
    # End-to-end inference integration
    # -------------------------------------------------------------------

    def test_run_inference_with_custom_node(self, simple_custom_state):
        """run_inference drives a multi-step loop over the custom node."""
        structure, params, clamps, state = simple_custom_state

        z_before = state.nodes["custom"].z_latent

        final_state = run_inference(params, state, clamps, structure)

        z_after = final_state.nodes["custom"].z_latent
        assert z_after.shape == z_before.shape
        assert jnp.isfinite(z_after).all()
        assert not jnp.allclose(z_after, z_before)

        assert jnp.allclose(final_state.nodes["input"].z_latent, clamps["input"])
        assert jnp.allclose(final_state.nodes["output"].z_latent, clamps["output"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
