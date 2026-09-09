"""
Unified convolutional node (1D, 2D, 3D) for predictive coding graphs.

Spatial rank is inferred from ``len(shape) - 1`` at construction time.
All gradient computation, energy accumulation, and the activation convention
are handled by NodeBase; only ``get_slots``, ``get_variance_factor``,
``initialize_params``, and ``forward`` are implemented here.

Design notes
------------
The user-supplied ``weight_init`` flows through verbatim as
``node_info.weight_init`` and is invoked directly on the conv kernel shape;
the shape-aware ``KaimingInitializer`` / ``XavierInitializer`` derive
fan-in / fan-out from that shape on their own. The conv node does not
inspect the activation or rewrite the initializer — pairing an
appropriate initializer config (e.g. ``nonlinearity="leaky_relu"``) with
the chosen activation is the caller's responsibility.

Shape validation (rank, kernel/stride length, declared-vs-computed output)
fires in ``initialize_params`` via the shared ``validate_windowed_output``
helper — not at construction — because the input shapes it checks against are
only known once the graph is wired. Dilation is not supported (the underlying
``lax.conv_general_dilated`` is always called with the default dilation of 1).
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Union, Sequence, TYPE_CHECKING
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec, validate_windowed_output
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import (
    KaimingInitializer,
    ZerosInitializer,
    NormalInitializer,
    initialize,
)

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


class ConvNode(NodeBase):
    """
    Unified convolutional node (1D, 2D, 3D) for predictive coding graphs.

    Spatial rank is inferred from the output shape:
        len(shape) == 2  →  1D conv  (L_out, C_out)
        len(shape) == 3  →  2D conv  (H_out, W_out, C_out)
        len(shape) == 4  →  3D conv  (D_out, H_out, W_out, C_out)

    Args:
        shape:       Output shape excluding batch dimension.
        name:        Node name.
        kernel_size: Spatial kernel size per axis. **Required — no default.**
        stride:      Stride per spatial axis. Defaults to all-ones.
        padding:     "SAME" or "VALID", or sequence of (low, high) pad pairs.
                     Defaults to "SAME" (conv usually preserves spatial size);
                     note the pooling nodes default to "VALID" instead.
        activation:  Default: ReLUActivation().
        energy:      Default: GaussianEnergy().
        use_bias:    Default: True.
        weight_init: Default: KaimingInitializer().
        bias_init:   Default: ZerosInitializer().
        latent_init: Default: NormalInitializer().
    """

    _DIM_NUMBERS = {
        1: ("NLC", "LIO", "NLC"),
        2: ("NHWC", "HWIO", "NHWC"),
        3: ("NDHWC", "DHWIO", "NDHWC"),
    }

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        kernel_size: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]] = None,
        padding: Union[str, Sequence[Tuple[int, int]]] = "SAME",
        activation: "ActivationBase" = ReLUActivation(),
        energy: "EnergyFunctional" = GaussianEnergy(),
        use_bias: bool = True,
        weight_init: "InitializerBase" = KaimingInitializer(),
        bias_init: "InitializerBase" = ZerosInitializer(),
        latent_init: "InitializerBase" = NormalInitializer(),
    ):
        """
        Args:
            shape: Output shape excluding batch dimension.
            name: Node name.
            kernel_size: Spatial kernel size per axis.
            stride: Stride per spatial axis. Defaults to all-ones.
            padding: "SAME" or "VALID", or sequence of (low, high) pad pairs.
            activation: ActivationBase instance. Default: ReLUActivation.
            energy: EnergyFunctional instance. Default: GaussianEnergy.
            use_bias: Whether to add a learnable bias. Default: True.
            weight_init: Initializer for kernels. Default: KaimingInitializer.
            bias_init: Initializer for bias. Default: ZerosInitializer.
            latent_init: Initializer for latent states. Default: NormalInitializer.
        """
        # Stride default depends on spatial_rank, so it stays in the body.
        # Other defaults moved to signature: framework guarantees init class
        # configs are MappingProxyType (immutable), so sharing the singleton
        # default across nodes is safe. Structural validation (rank,
        # kernel/stride length, declared-vs-computed output) is deferred to
        # initialize_params via validate_windowed_output — see module docstring.
        if stride is None:
            stride = (1,) * (len(shape) - 1)

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            use_bias=use_bias,
            bias_init=bias_init,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def get_variance_factor(
        source_shape: Tuple[int, ...],
        config: Dict[str, Any],
        weight_init: Optional["InitializerBase"],
    ) -> float:
        """Kaiming fan_in = C_in × ∏(kernel_size): each output unit sums that
        many products. Overrides NodeBase (which returns only C_in)."""
        kernel_size = config.get("kernel_size")
        return float(source_shape[-1] * int(np.prod(kernel_size)))

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: "InitializerBase",
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """
        Initialise kernels (*kernel_size, C_in, C_out) and bias (1,…,1, C_out).
        """
        if config is None:
            config = {}

        kernel_size = config.get("kernel_size")
        out_channels = node_shape[-1]
        use_bias = config.get("use_bias", True)
        # weight_init and bias_init are defaulted once, in the __init__
        # signature, and flow in via node_info.weight_init / config. They are
        # not re-defaulted here — the single source of truth is the constructor.
        bias_init = config.get("bias_init")
        # Fail fast rather than passing None to initialize() below. Normal graph
        # construction never hits this (the constructor defaults bias_init to
        # ZerosInitializer); it only guards a hand-built config or an explicit
        # bias_init=None paired with use_bias=True.
        if use_bias and bias_init is None:
            raise ValueError(
                "ConvNode: use_bias=True but bias_init is None. Pass a bias "
                "initializer (e.g. ZerosInitializer()) or set use_bias=False."
            )

        # One key per weight edge, plus one for the bias only when needed.
        keys = jax.random.split(key, len(input_shapes) + (1 if use_bias else 0))

        # Fail fast (here, not late inside JIT) if rank, kernel/stride length, or
        # the declared output shape disagree with what the conv will produce.
        # channels_preserved=False: a conv sets its own C_out via the kernel.
        validate_windowed_output(
            node_shape,
            input_shapes,
            window=kernel_size,
            stride=config.get("stride"),
            padding=config.get("padding"),
            op_name="ConvNode",
            channels_preserved=False,
        )

        weights_dict = {}
        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_channels = in_shape[-1]
            kernel_shape = (*kernel_size, in_channels, out_channels)
            weights_dict[edge_key] = initialize(keys[i], kernel_shape, weight_init)

        if use_bias:
            bias_shape = (1,) * len(node_shape) + (out_channels,)
            return NodeParams(
                weights=weights_dict,
                biases={"b": initialize(keys[-1], bias_shape, bias_init)},
            )
        return NodeParams(weights=weights_dict, biases={})

    @staticmethod
    def predict(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jnp.ndarray, None]:
        """
        Standard NodeBase predict contract: conv sum → bias → activation.
        """
        config = node_info.node_config
        dim_numbers = ConvNode._DIM_NUMBERS[len(node_info.shape) - 1]

        pre_activation = jnp.zeros((state.z_latent.shape[0], *node_info.shape))
        for edge_key, x in inputs.items():
            pre_activation = pre_activation + lax.conv_general_dilated(
                lhs=x,
                rhs=params.weights[edge_key],
                window_strides=config.get("stride"),
                padding=config.get("padding"),
                dimension_numbers=dim_numbers,
            )

        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)
        return z_mu, None
