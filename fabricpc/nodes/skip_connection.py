"""
Skip connection node for residual architectures.

SkipConnection sums inputs from two slots and passes the sum through:

  - "in"   (``is_variance_scalable=True``): receives computed branch
    contributions. The node is weightless (fan_in=1), so muPC scales each
    edge by ``gain/sqrt(K_slot * L)`` — the once-per-branch depth damping,
    applied where the branch joins the residual stream.
  - "skip" (``is_skip_connection=True``): receives the residual stream.
    Edges pass through at scale 1.0, preserving the identity mapping that
    carries signal through deep networks. Connecting this slot makes the
    node a merge node and counts it toward the residual depth L.

This is LinearResidual's slot layout without the weights. Use
SkipConnection to merge a computed branch into the residual stream. Use
IdentityNode for summation points where all inputs are independent and
should be variance-scaled without depth damping.

Example — a residual block in graph form::

    linear = Linear(shape=(128,), name="h1")
    skip   = SkipConnection(shape=(128,), name="res1")

    edges = [
        Edge(source=prev, target=linear.slot("in")),    # branch (transform)
        Edge(source=prev, target=skip.slot("skip")),    # stream (unscaled)
        Edge(source=linear, target=skip.slot("in")),    # branch joins stream
    ]
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


class SkipConnection(NodeBase):
    """
    Skip connection node: merges computed branches into the residual stream.

    Sums all inputs from both slots, no learnable parameters. Edges into
    "in" carry the branch's muPC depth damping ``gain/sqrt(K_slot * L)``
    (weightless, fan_in=1); edges into "skip" pass through at scale 1.0,
    preserving the identity stream through deep residual networks. Without
    the unscaled stream slot, muPC's in-degree formula would scale the
    stream by 1/sqrt(K), causing exponential signal decay (0.707^L for
    K=2).
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: ActivationBase = IdentityActivation(),
        energy: EnergyFunctional = GaussianEnergy(),
        latent_init: InitializerBase = NormalInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {
            "in": SlotSpec(
                name="in",
                is_multi_input=True,
                is_variance_scalable=True,
            ),
            "skip": SlotSpec(
                name="skip",
                is_multi_input=True,
                is_variance_scalable=False,
                is_skip_connection=True,
                # A SkipConnection with no stream edge is an IdentityNode with
                # extra steps: it stops counting toward L, and edges meant for
                # the stream get summed and scaled like ordinary branch inputs,
                # reintroducing the 1/sqrt(K)^L decay this node exists to
                # prevent. Fail at construction rather than train quietly wrong.
                require_connected=True,
            ),
        }

    @staticmethod
    def get_variance_factor(
        source_shape: Tuple[int, ...],
        config: Dict[str, Any],
        weight_init: Optional[InitializerBase],
    ) -> float:
        """No weight matrix and no reduction — the transform is a sum, so 1.0."""
        return 1.0

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional[InitializerBase] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        return NodeParams(weights={}, biases={})

    @staticmethod
    def predict(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jnp.ndarray, None]:
        """Sum all inputs from both slots and pass through (no transformation)."""
        z_mu = None
        for edge_key, x in inputs.items():
            if z_mu is None:
                z_mu = x
            else:
                z_mu = z_mu + x
        # no activation function applied: z_mu is the plain sum
        return z_mu, None
