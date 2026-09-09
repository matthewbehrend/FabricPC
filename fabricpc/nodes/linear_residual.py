"""
Linear residual node for deep residual predictive coding networks.

LinearResidual combines a linear transformation with a residual (skip)
connection in a single PC node:

    z_mu = activation(W @ x_in + b) + x_skip

Two input slots:
  - "in"   (is_variance_scalable=True):  receives the transform path input,
            has a weight matrix, and is scaled by muPC.
  - "skip" (is_variance_scalable=False): receives the identity skip path,
            no weight matrix, passes through at scale 1.0.

This halves the graph depth compared to the Linear + SkipConnection pattern
(one PC node per residual block instead of two), while muPC automatically
applies the correct per-edge scaling via the slot's is_variance_scalable flag.

Example — a residual chain::

    prev = stem
    for i in range(num_blocks):
        res = LinearResidual(shape=(W,), activation=TanhActivation(),
                             weight_init=MuPCInitializer(), name=f"res{i}")
        edges += [
            Edge(source=prev, target=res.slot("in")),    # transform path
            Edge(source=prev, target=res.slot("skip")),   # identity skip
        ]
        prev = res
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
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

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


class LinearResidual(FlattenInputMixin, NodeBase):
    """
    Linear residual node: z_mu = activation(W @ x_in + b) + x_skip

    Combines a linear transformation (on the "in" slot) with an identity
    residual connection (on the "skip" slot) in one PC node. muPC scales
    the "in" edges normally and leaves "skip" edges at scale 1.0.
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
        return {
            "in": SlotSpec(name="in", is_multi_input=True, is_variance_scalable=True),
            "skip": SlotSpec(
                name="skip",
                is_multi_input=True,
                is_variance_scalable=False,
                is_skip_connection=True,
            ),
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: InitializerBase,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialize weight matrices for "in" slot edges only.
        Skip slot edges are identity (no parameters).
        """
        if config is None:
            config = {}

        flatten_input = config.get("flatten_input", False)

        # weight_init is defaulted once, in the __init__ signature, and flows
        # in via node_info.weight_init. It is not re-defaulted here — the
        # single source of truth is the constructor.

        key_w, key_b = jax.random.split(key)

        # Weight matrices only for "in" slot edges
        in_slot_shapes = {k: v for k, v in input_shapes.items() if ":in" in k}

        weights_dict = {}
        rand_key_w = dict(
            zip(in_slot_shapes.keys(), jax.random.split(key_w, len(in_slot_shapes)))
        )

        for edge_key, in_shape in in_slot_shapes.items():
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

        # Bias
        use_bias = config.get("use_bias", True)
        if use_bias:
            bias_shape = (1,) * len(node_shape) + (node_shape[-1],)
            b = jnp.zeros(bias_shape)

        return NodeParams(weights=weights_dict, biases={"b": b} if use_bias else {})

    @staticmethod
    def predict(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jnp.ndarray, None]:
        """
        Prediction: z_mu = activation(W @ x_in + b) + x_skip
        """
        # Separate inputs by slot
        in_inputs = {k: v for k, v in inputs.items() if ":in" in k}
        skip_inputs = {k: v for k, v in inputs.items() if ":skip" in k}

        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape
        flatten_input = node_info.node_config.get("flatten_input", False)

        # Linear transform on "in" slot inputs
        if flatten_input:
            pre_activation = FlattenInputMixin.compute_linear(
                in_inputs, params.weights, batch_size, out_shape
            )
        else:
            pre_activation = jnp.zeros((batch_size,) + out_shape)
            for edge_key, x in in_inputs.items():
                pre_activation = pre_activation + jnp.matmul(
                    x, params.weights[edge_key]
                )

        # Add bias
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        # Apply activation to the transform path
        activation = node_info.activation
        transformed = type(activation).forward(pre_activation, activation.config)

        # Sum skip inputs (identity, no transform)
        skip_sum = None
        for x in skip_inputs.values():
            skip_sum = x if skip_sum is None else skip_sum + x

        # Residual sum
        z_mu = transformed + skip_sum if skip_sum is not None else transformed

        return z_mu, None
