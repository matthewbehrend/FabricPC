"""
Pooling operations for predictive coding graphs.

Provides ``MaxPool`` (windowed max pooling) and ``AvgPool`` (windowed or
global average pooling), both built on a shared ``_PoolBase``. They are
parameter-free graph elements: you wire them in like any other operation, but
the dropped ``Node`` suffix signals that they carry no learnable weights and
exist only to reduce spatial dimensions — not to learn a representation.

Windowed shape validation (rank, window/stride length, channel count,
declared-vs-computed output) fires in ``initialize_params`` via the shared
``validate_windowed_output`` helper — not at construction — because the input
shapes it checks against are only known once the graph is wired. Global
``AvgPool``'s rank-1 shape check is not windowed and stays at construction.

muPC scaling note
-----------------
Pooling is a weightless *transformation*, not an identity skip-connection
path, so its slot stays variance-scalable (``is_variance_scalable=False`` is
strictly for identity-mapping skip paths — see SkipConnection and the
transformer mask/residual edges). Pooling declares no skip slot, so a pool is
never a merge node and its edges carry no depth factor L. What each pool
reports through ``get_variance_factor`` is how its reduction changes input
variance; muPC then applies ``a = gain / sqrt(v * K_slot)``.

The two reductions differ, and the difference is not cosmetic:

  * ``AvgPool`` divides a sum of ``n`` cells by ``n``. For uncorrelated inputs
    that multiplies variance by ``1/n`` exactly, independent of the input
    distribution, because the mean is a linear operator. It returns
    ``v = 1/n``, so muPC amplifies by ``sqrt(n)``. This is the same convention
    muPC already applies to a ``K_slot``-way edge sum (``1/sqrt(K_slot)``, not
    ``1/K_slot``): a spatial mean and an edge sum are both sums, and both are
    normalized to preserve variance rather than magnitude.

    Assumption: the pooled cells are uncorrelated. Real conv feature maps are
    spatially correlated, so the true reduction lies between ``1/n``
    (independent) and ``1`` (perfectly correlated), and ``sqrt(n)``
    over-corrects in proportion. This is the same independence assumption
    Kaiming/muPC already make across a weight matrix's fan_in.

  * ``MaxPool`` has no distribution-free variance factor, so it returns
    ``v = 1.0``. Measured variance of ``max`` over ``n`` iid cells, relative to
    the input variance:

        n     Gaussian N(0,1)     ReLU(N(0,1))
        4     0.49                1.32
        9     0.36                1.05
        16    0.29                0.87
        25    0.26                0.76

    A max pool in a conv graph follows a ReLU, where the ratio at the usual
    2x2 window is 1.32 — a Gaussian-derived correction (``1/sqrt(0.49)`` =
    1.43) would be wrong by a factor of 2.7 for the inputs it actually sees.
    No single constant is right for both columns, so no correction is applied.

    A limitation no edge multiplier can fix: max pooling shifts the mean
    (measured mean 1.05 at n=4 for ReLU inputs, against a standard deviation
    of 0.67), and a scalar scale multiplies an offset rather than removing it.
    In an un-normalized PC network that offset propagates downstream. Prefer
    ``AvgPool`` where variance behavior through depth matters.

Returning the upstream channel count (the ``NodeBase`` default) would be wrong
for both: pooling has no weight matrix, so it would attenuate activations and
gradients through every pool by ``1/sqrt(C_in)``.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, Sequence, Union, TYPE_CHECKING
import numpy as np
import jax
import jax.lax as lax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec, validate_windowed_output
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


class _PoolBase(NodeBase):
    """
    Shared structure for parameter-free pooling operations.

    Subclasses implement the reduction in ``_pool(x_sum, node_info)``. The common
    parts — slots, fan_in=1, empty params, and the forward template (sum inputs →
    pool → activation → error → energy) — live here. Not exported; use
    ``MaxPool`` or ``AvgPool``.
    """

    # -- shared helpers -----------------------------------------------------

    @staticmethod
    def _format_pool_padding(
        padding: Union[str, Sequence[Tuple[int, int]]],
    ) -> Union[str, Tuple[Tuple[int, int], ...]]:
        """Format padding for ``lax.reduce_window``.

        Strings ("SAME"/"VALID") pass through. A sequence of spatial ``(low, high)``
        pairs is wrapped with ``(0, 0)`` for the batch and channel dims, since
        ``reduce_window`` requires a pair for every dimension (unlike
        ``conv_general_dilated``, which takes spatial-only padding).
        """
        if isinstance(padding, str):
            return padding
        spatial = tuple((int(lo), int(hi)) for lo, hi in padding)
        return ((0, 0),) + spatial + ((0, 0),)

    # -- NodeBase contract --------------------------------------------------

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        return {"in": SlotSpec(name="in", is_multi_input=True)}

    @staticmethod
    def _pool_cell_count(source_shape: Tuple[int, ...], config: Dict[str, Any]) -> int:
        """Number of input cells reduced into one output cell.

        Global mode collapses every spatial axis of the source (``source_shape``
        excludes batch and ends in channels). Windowed mode reduces one window,
        whose volume is ``prod(window_shape)`` — matching the divisor
        ``count_include_pad=True`` (the default) uses. With
        ``count_include_pad=False`` the divisor is the count of real cells,
        which varies per window; the full window volume is used instead, exact
        under the default ``"VALID"`` padding, which adds no padded cells.
        """
        if config.get("global_pool", False):
            return int(np.prod(source_shape[:-1]))
        return int(np.prod(config["window_shape"]))

    @staticmethod
    def get_variance_factor(
        source_shape: Tuple[int, ...],
        config: Dict[str, Any],
        weight_init: Optional["InitializerBase"],
    ) -> float:
        """Subclass-specific: how the reduction changes input variance. Override."""
        raise NotImplementedError

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: Optional["InitializerBase"] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> NodeParams:
        """Pooling has no learnable parameters.

        Windowed mode validates here (the only place input shapes are known)
        that rank, window/stride length, channel count, and the declared output
        spatial shape all agree with what the window/stride/padding will
        produce — failing fast instead of late in forward. Global mode has no
        spatial output, so it is skipped (its rank-1 shape is checked at
        construction).
        """
        config = config or {}
        if not config.get("global_pool", False):
            # channels_preserved=True: pooling cannot change the channel count.
            validate_windowed_output(
                node_shape,
                input_shapes,
                window=config.get("window_shape"),
                stride=config.get("stride"),
                padding=config.get("padding"),
                op_name=config.get("pool_op_name", "Pooling"),
                channels_preserved=True,
                reject_pad_ge_window=config.get("reject_pad_ge_window", False),
            )
        return NodeParams(weights={}, biases={})

    @staticmethod
    def _pool(x_sum: jnp.ndarray, node_info: NodeInfo) -> jnp.ndarray:
        """Subclass-specific reduction over the summed inputs. Override."""
        raise NotImplementedError

    @staticmethod
    def predict(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jnp.ndarray, None]:
        """
        Sum incoming edges, apply the subclass reduction, apply activation.
        Dispatches the reduction through ``node_info.node_class`` so the
        concrete subclass's ``_pool`` runs.
        """
        x_sum = sum(inputs.values())

        pre_activation = node_info.node_class._pool(x_sum, node_info)

        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)
        return z_mu, None


class MaxPool(_PoolBase):
    """
    Windowed max pooling for spatial downsampling.

    Reduces spatial dimensions by taking the maximum value over a sliding window.
    Channels-last format: (Batch, Spatial..., Channels). No learnable params.

    Args:
        shape:        Output shape excluding batch dimension.
        name:         Node name.
        window_shape: Spatial window size per axis.
        stride:       Stride per spatial axis. Defaults to ``window_shape``.
        padding:      "SAME" / "VALID", or a sequence of spatial (low, high) pairs.
                      Defaults to "VALID" (pooling usually downsamples); note
                      that ConvNode defaults to "SAME" instead.
        activation:   Default: IdentityActivation().
        energy:       Default: GaussianEnergy().
        latent_init:  Default: NormalInitializer().
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        window_shape: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]] = None,
        padding: Union[str, Sequence[Tuple[int, int]]] = "VALID",
        activation: "ActivationBase" = IdentityActivation(),
        energy: "EnergyFunctional" = GaussianEnergy(),
        latent_init: "InitializerBase" = NormalInitializer(),
    ):
        # Stride defaults to the window (non-overlapping pooling). Structural
        # validation is deferred to initialize_params (see module docstring).
        if stride is None:
            stride = window_shape

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=None,  # No weights
            use_bias=False,  # No bias
            window_shape=window_shape,
            stride=stride,
            padding=padding,
            pool_op_name="MaxPool",
            # Max pooling fills with -inf, so a window fully covered by explicit
            # padding would output -inf; validate_windowed_output rejects it.
            reject_pad_ge_window=True,
        )

    @staticmethod
    def get_variance_factor(
        source_shape: Tuple[int, ...],
        config: Dict[str, Any],
        weight_init: Optional["InitializerBase"],
    ) -> float:
        """No correction: a max has no distribution-free variance factor.

        The variance of ``max`` over ``n`` cells depends on the input
        distribution, and the two distributions that matter here disagree — at
        n=4 the ratio is 0.49 for Gaussian inputs but 1.32 for the ReLU outputs
        a max pool actually receives. See the module docstring for the measured
        table and for the mean shift, which no edge scale can remove.
        """
        return 1.0

    @staticmethod
    def _pool(x_sum: jnp.ndarray, node_info: NodeInfo) -> jnp.ndarray:
        config = node_info.node_config
        window_shape = config["window_shape"]
        stride = config["stride"]
        padding = _PoolBase._format_pool_padding(config["padding"])

        # Batch and channel are not pooled, so their window/stride is 1.
        full_window = (1,) + tuple(window_shape) + (1,)
        full_strides = (1,) + tuple(stride) + (1,)

        return lax.reduce_window(
            x_sum, -jnp.inf, lax.max, full_window, full_strides, padding
        )


class AvgPool(_PoolBase):
    """
    Average pooling — windowed or global.

    * Windowed (``global_pool=False``): averages over a sliding window, keeping
      spatial structure like MaxPool: (B, Spatial..., C) -> (B, Spatial'..., C).
      ``shape`` carries the reduced spatial rank, e.g. (14, 14, 32).
    * Global (``global_pool=True``): averages over *all* spatial dimensions,
      collapsing them: (B, Spatial..., C) -> (B, C). ``shape`` is just (C,) and
      ``window_shape`` is ignored.

    No learnable parameters.

    Args:
        shape:        Output shape excluding batch dimension.
        name:         Node name.
        window_shape: Spatial window size per axis (windowed mode only).
        stride:       Stride per spatial axis. Defaults to ``window_shape``.
        padding:      "SAME" / "VALID", or a sequence of spatial (low, high) pairs.
                      Defaults to "VALID" (pooling usually downsamples); note
                      that ConvNode defaults to "SAME" instead.
        global_pool:  If True, average over all spatial dims -> (B, C).
        count_include_pad: Windowed mode only. If True (default, matches
                      PyTorch), divide each window by the full window volume
                      (padded cells count as zeros). If False, divide by the
                      number of real (non-padded) elements. No effect in
                      global mode.
        activation:   Default: IdentityActivation().
        energy:       Default: GaussianEnergy().
        latent_init:  Default: NormalInitializer().
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        window_shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
        padding: Union[str, Sequence[Tuple[int, int]]] = "VALID",
        global_pool: bool = False,
        count_include_pad: bool = True,
        activation: "ActivationBase" = IdentityActivation(),
        energy: "EnergyFunctional" = GaussianEnergy(),
        latent_init: "InitializerBase" = NormalInitializer(),
    ):
        if global_pool:
            # Global mode collapses all spatial dims -> (B, C), so the declared
            # output shape (batch excluded) must be rank-1: (C,). This is not a
            # windowed check, so it stays at construction. Fail fast here rather
            # than at the first forward with an opaque shape mismatch.
            if len(shape) != 1:
                raise ValueError(
                    f"AvgPool global_pool=True requires a rank-1 shape (C,), "
                    f"got shape={shape}."
                )
            # Global mode collapses all spatial dims; window/stride unused.
            window_shape = window_shape or ()
            stride = stride or ()
        else:
            if window_shape is None:
                raise ValueError(
                    "AvgPool: window_shape is required when global_pool=False."
                )
            # Stride defaults to the window (non-overlapping pooling). Windowed
            # validation is deferred to initialize_params (module docstring).
            if stride is None:
                stride = window_shape

        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=None,  # No weights
            use_bias=False,  # No bias
            window_shape=window_shape,
            stride=stride,
            padding=padding,
            global_pool=global_pool,
            count_include_pad=count_include_pad,
            pool_op_name="AvgPool",
        )

    @staticmethod
    def get_variance_factor(
        source_shape: Tuple[int, ...],
        config: Dict[str, Any],
        weight_init: Optional["InitializerBase"],
    ) -> float:
        """``1/n`` for ``n`` pooled cells: the mean divides a sum of n terms by n.

        For uncorrelated cells this is exact and distribution-free, so muPC
        amplifies the in-edge by ``sqrt(n)`` and the pool preserves variance
        instead of attenuating by ``1/sqrt(n)``. See the module docstring for
        the independence assumption and how correlated feature maps weaken it.
        """
        return 1.0 / _PoolBase._pool_cell_count(source_shape, config)

    @staticmethod
    def _pool(x_sum: jnp.ndarray, node_info: NodeInfo) -> jnp.ndarray:
        config = node_info.node_config

        if config.get("global_pool", False):
            # Average over every spatial axis: (B, Spatial..., C) -> (B, C).
            spatial_axes = tuple(range(1, x_sum.ndim - 1))
            return jnp.mean(x_sum, axis=spatial_axes)

        window_shape = config["window_shape"]
        stride = config["stride"]
        padding = _PoolBase._format_pool_padding(config["padding"])

        full_window = (1,) + tuple(window_shape) + (1,)
        full_strides = (1,) + tuple(stride) + (1,)

        summed = lax.reduce_window(
            x_sum, 0.0, lax.add, full_window, full_strides, padding
        )

        if config.get("count_include_pad", True):
            # Divide by the full window volume; padded cells count as zeros.
            # Exact for the default VALID padding (no padding is added).
            return summed / float(np.prod(window_shape))

        # count_include_pad=False: divide each window by its number of *real*
        # (non-padded) elements. Summing an all-ones array through the identical
        # window yields that count exactly (padding contributes 0). The safe
        # divisor + jnp.where guard the degenerate fully-padded window
        # (count == 0), which would otherwise emit NaN that propagates through
        # the network; matching PyTorch, such windows yield 0.
        counts = lax.reduce_window(
            jnp.ones_like(x_sum), 0.0, lax.add, full_window, full_strides, padding
        )
        safe_counts = jnp.where(counts > 0, counts, 1.0)
        return jnp.where(counts > 0, summed / safe_counts, 0.0)
