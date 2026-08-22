"""
Linear node implementation for JAX predictive coding networks.

This implements a linear transformation node with configurable activation functions.
The node has a single multi-input slot that accepts multiple incoming connections.

By default, linear nodes apply matrix multiplication on the **last axis only**, which is
standard for embeddings, projections, and transformer layers. This means:
- Input shape: (batch, ..., in_features)
- Weight shape: (in_features, out_features)
- Output shape: (batch, ..., out_features)

For fully-connected (dense) behavior that flattens all dimensions, set `flatten_input=True`.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple
import numpy as np
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import (
    NodeBase,
    SlotSpec,
    FlattenInputMixin,
)
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer, KaimingInitializer, initialize

from fabricpc.core.activations import ActivationBase
from fabricpc.core.energy import EnergyFunctional
from fabricpc.core.initializers import InitializerBase


class Linear(FlattenInputMixin, NodeBase):
    """
    Linear transformation node: y = activation(W @ x + b)

    This node type:
    - Has a single multi-input slot named "in"
    - Concatenates all inputs and applies a linear transformation
    - Supports various activation functions
    - Implements local Hebbian learning

    Uses FlattenInputMixin for flatten/reshape operations.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: ActivationBase = IdentityActivation(),
        energy: EnergyFunctional = GaussianEnergy(),
        use_bias: bool = True,
        flatten_input: bool = False,
        weight_init: InitializerBase = KaimingInitializer(),
        latent_init: InitializerBase = NormalInitializer(),
    ):
        """
        Args:
            shape: Output shape tuple (excluding batch dimension)
            name: Node name
            activation: ActivationBase instance (default: IdentityActivation)
            energy: EnergyFunctional instance (default: GaussianEnergy)
            use_bias: Whether to use bias (default: True)
            flatten_input: If True, flatten all input dims for dense behavior (default: False)
            weight_init: InitializerBase instance for weights (default: KaimingInitializer)
            latent_init: InitializerBase instance for latent states
        """
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
            flatten_input=flatten_input,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """Linear nodes have a single multi-input slot."""
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def predict(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Linear transformation with activation.

        When flatten_input=False (default): applies matmul on last axis only.
        When flatten_input=True: flattens all dimensions for dense behavior.

        Returns:
            Tuple of (z_mu, pre_activation). The pre_activation aux feeds
            LinearExplicitGrad's analytic f'(z) gain-modulated error.
        """
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        flatten_input = node_info.node_config.get("flatten_input", False)

        if flatten_input:
            pre_activation = FlattenInputMixin.compute_linear(
                inputs, params.weights, batch_size, out_shape
            )
        else:
            pre_activation = jnp.zeros((batch_size,) + out_shape)
            for edge_key, x in inputs.items():
                pre_activation = pre_activation + jnp.matmul(
                    x, params.weights[edge_key]
                )
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)
        return z_mu, pre_activation

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: InitializerBase,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialize weight matrix and bias vector.

        Weight shape depends on `flatten_input` config:
        - flatten_input=False (default): weights have shape (in_features, out_features)
        - flatten_input=True: weights have shape (in_numel, out_numel) for dense layers.

        Args:
            key: JAX random key
            node_shape: Output shape of this node (excluding batch dimension)
            input_shapes: Dictionary with EdgeInfo.key -> source shape for that edge
            weight_init: InitializerBase instance for weight initialization,
                supplied by the graph builder from the constructor's weight_init
            config: Node configuration with weight_init settings

        Returns:
            NodeParams with initialized W and b
        """
        if config is None:
            config = {}

        flatten_input = config.get("flatten_input", False)

        # Split key for weights and biases
        key_w, key_b = jax.random.split(key)

        # Initialize weight matrix for each incoming edge
        weights_dict = {}
        rand_key_w = dict(
            zip(input_shapes.keys(), jax.random.split(key_w, len(input_shapes)))
        )

        for edge_key, in_shape in input_shapes.items():
            if ":in" not in edge_key:
                raise ValueError(
                    f"linear node requires 'in' slot dimension. got edge key {edge_key}"
                )

            if flatten_input:
                in_numel = int(np.prod(in_shape))
                out_numel = int(np.prod(node_shape))
                weight_shape = (in_numel, out_numel)
            else:
                in_features = in_shape[-1]
                out_features = node_shape[-1]
                weight_shape = (in_features, out_features)

            weights_dict[edge_key] = initialize(
                rand_key_w[edge_key], weight_shape, weight_init
            )

        # Initialize bias (usually zeros)
        # Bias shape for proper broadcasting, prepending batch dimension: (1, ..., 1, out_features)
        use_bias = config.get("use_bias", True)
        if use_bias:
            bias_shape = (1,) * len(node_shape) + (node_shape[-1],)
            b = jnp.zeros(
                bias_shape
            )  # TODO allow configurable bias initialization across all nodes that use bias; default to zeros init; use the initializer object and seed key_b

        return NodeParams(weights=weights_dict, biases={"b": b} if use_bias else {})
