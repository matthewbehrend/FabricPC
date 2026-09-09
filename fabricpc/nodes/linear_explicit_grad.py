from typing import Dict, Tuple

import jax.numpy as jnp
from fabricpc.nodes.linear import Linear
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer
from fabricpc.nodes.base import FlattenInputMixin, NodeBase
from fabricpc.core.activations import ActivationBase
from fabricpc.core.energy import EnergyFunctional
from fabricpc.core.initializers import InitializerBase


class LinearExplicitGrad(Linear):
    """
    Linear node with explicit (non-autodiff) gradient computation.
    Demonstrates overriding NodeBase's autodiff-based gradient computation.

    Useful for:
    - Verifying correctness of manual gradient implementations
    - Prototyping optimized gradients
    - Debugging gradient computation issues
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: ActivationBase = IdentityActivation(),
        energy: EnergyFunctional = GaussianEnergy(),
        use_bias: bool = True,
        flatten_input: bool = False,
        weight_init: InitializerBase = NormalInitializer(),
        latent_init: InitializerBase = NormalInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            use_bias=use_bias,
            flatten_input=flatten_input,
            weight_init=weight_init,
            latent_init=latent_init,
        )

    @staticmethod
    def forward_and_latent_grads(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
        is_clamped: bool,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray], jnp.ndarray]:
        """Forward pass with explicit (non-autodiff) gradient computation.

        Demonstrates the override pattern: computes input gradients and
        self-latent gradient analytically using energy.grad_latent() and
        activation.derivative(). muPC scaling and accumulation into
        ``state.latent_grad`` are handled by the callsite.
        """
        node_class = node_info.node_class

        # Source semantics and the zero-grad short-circuit are base-owned;
        # the analytic path below needs the pre_activation aux, which only
        # exists when predict() runs (in_degree > 0).
        if node_info.in_degree == 0:
            return NodeBase.forward_and_latent_grads(
                params, inputs, state, node_info, is_clamped
            )

        # Single forward computing energy, state, and pre_activation together
        # (Linear.predict returns pre_activation as its aux).
        # pre_activation feeds the explicit gain-modulated error below.
        state, pre_activation = node_class.forward_with_aux(
            params, inputs, state, node_info
        )

        # Explicit self-latent gradient
        energy_obj = node_info.energy
        self_grad = type(energy_obj).grad_latent(
            state.z_latent, state.z_mu, energy_obj.config
        )

        # Gain-modulated error for input gradients
        gain_mod_error = node_class.compute_gain_mod_error(
            pre_activation, state.error, node_info
        )

        energy_type = type(energy_obj).__name__
        flatten_input = node_info.node_config.get("flatten_input", False)
        input_grads = {}

        for edge_key, z in inputs.items():
            source_shape = z.shape[1:]

            if energy_type == "GaussianEnergy":
                if flatten_input:
                    gain_mod_error_flat = FlattenInputMixin.flatten_input(
                        gain_mod_error
                    )
                    grad_flat = -jnp.matmul(
                        gain_mod_error_flat, params.weights[edge_key].T
                    )
                    grad_contribution = FlattenInputMixin.reshape_output(
                        grad_flat, source_shape
                    )
                else:
                    grad_contribution = -jnp.matmul(
                        gain_mod_error,
                        params.weights[edge_key].T,
                    )
            else:
                raise NotImplementedError(
                    f"energy functional '{energy_type}' not implemented in LinearExplicitGrad."
                )

            input_grads[edge_key] = grad_contribution

        return state, input_grads, self_grad

    @staticmethod
    def forward_and_weight_grads(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[NodeState, NodeParams]:
        """Forward pass with explicit weight gradient computation.

        muPC scaling is applied by the callsite.
        """
        node_class = node_info.node_class

        # Single forward computing energy, state, and pre_activation together
        # (Linear.predict returns pre_activation as its aux).
        state, pre_activation = node_class.forward_with_aux(
            params, inputs, state, node_info
        )
        flatten_input = node_info.node_config.get("flatten_input", False)

        # Gain-modulated error computation
        gain_mod_error = node_class.compute_gain_mod_error(
            pre_activation, state.error, node_info
        )
        weight_grads = {}
        bias_grads = {}

        # Weight gradient
        for edge_key, in_tensor in inputs.items():
            if flatten_input:
                in_flat = FlattenInputMixin.flatten_input(in_tensor)
                gain_mod_error_flat = FlattenInputMixin.flatten_input(gain_mod_error)
                grad_w = -jnp.matmul(in_flat.T, gain_mod_error_flat)
            else:
                in_shape = in_tensor.shape
                err_shape = gain_mod_error.shape
                in_flat = in_tensor.reshape(-1, in_shape[-1])
                err_flat = gain_mod_error.reshape(-1, err_shape[-1])
                grad_w = -jnp.matmul(in_flat.T, err_flat)
            weight_grads[edge_key] = grad_w

        # Bias gradient
        if "b" in params.biases:
            grad_b = -jnp.sum(gain_mod_error, axis=0, keepdims=True)
            bias_grads["b"] = grad_b

        return state, NodeParams(weights=weight_grads, biases=bias_grads)

    @staticmethod
    def compute_gain_mod_error(
        pre_activation: jnp.ndarray,
        error: jnp.ndarray,
        node_info: NodeInfo,
    ) -> jnp.ndarray:
        """Compute gain-modulated error: error * f'(pre_activation).

        Returns:
            gain_mod_error array (error * activation derivative)
        """
        activation = node_info.activation
        f_prime = type(activation).derivative(pre_activation, activation.config)
        return error * f_prime
