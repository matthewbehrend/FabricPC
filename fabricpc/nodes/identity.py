"""
Identity node implementation for JAX predictive coding networks.

The IdentityNode passes input through unchanged with no transformation, no
activation and no learnable parameters. This is useful for input nodes or
auxiliary nodes that serve as conduits for data without learning.

When multiple inputs are connected, they are summed together.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


class IdentityNode(NodeBase):
    """
    Identity node: passes input through unchanged.

    This node type:
    - Has a single multi-input slot named "in"
    - Sums all inputs (if multiple) and passes through as z_mu
    - Has no learnable parameters (no weights, no biases)
    - Useful for input nodes or passthrough connections
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: ActivationBase = IdentityActivation(),
        energy: EnergyFunctional = GaussianEnergy(),
        latent_init: InitializerBase = NormalInitializer(),
        scale: float = 1.0,  # Optional fixed scaling factor of the node output (default 1.0, no scaling)
    ):
        """
        Args:
            shape: Output shape tuple (excluding batch dimension)
            name: Node name
            activation: ActivationBase instance (default: IdentityActivation)
            energy: EnergyFunctional instance (default: GaussianEnergy)
            latent_init: InitializerBase instance for latent states
        """
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            scale=scale,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """
        Identity nodes have a single multi-input slot.

        Returns:
            Dictionary with one slot "in" that accepts multiple inputs
        """
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_variance_factor(
        source_shape: Tuple[int, ...],
        config: Dict[str, Any],
        weight_init: Optional["InitializerBase"],
    ) -> float:
        """Return the muPC variance factor.

        IdentityNode has no weight matrix and no reduction — inputs are summed
        and passed through, so the transform leaves variance unchanged.
        Returning 1.0 reduces a = gain/sqrt(v * K_slot) to a = gain/sqrt(K_slot),
        compensating only for multi-edge summation variance amplification.
        """
        return 1.0

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialize parameters for identity node (none needed).

        Args:
            key: JAX random key (unused)
            node_shape: Output shape of this node (unused)
            input_shapes: Dictionary with edge keys to source shapes (unused)
            weight_init: InitializerBase instance (unused, identity has no weights)
            config: Node configuration dictionary

        Returns:
            NodeParams with empty weights and biases
        """
        if config is None:
            config = {}
        return NodeParams(weights={}, biases={})

    @staticmethod
    def predict(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jnp.ndarray, None]:
        """
        Identity prediction: sum inputs and pass through.

        Never called on a terminal source node (in_degree=0) — the base
        templates own source semantics (z_mu mirrors z_latent).

        Args:
            params: Node parameters (empty for identity node)
            inputs: Dictionary mapping edge keys to input tensors
            state: Current node state
            node_info: NodeInfo object

        Returns:
            Tuple of (z_mu, None).
        """
        # Sum all inputs
        z_mu = None
        for edge_key, x in inputs.items():
            if z_mu is None:
                z_mu = x
            else:
                z_mu = z_mu + x

        z_mu = (
            z_mu * node_info.node_config["scale"]
        )  # Apply fixed scaling factor (default is 1.0)

        return z_mu, None
