"""
Base node classes for JAX predictive coding networks.

This module provides the abstract base class for all node types, defining the
interface for custom transfer functions, multiple input slots, and local gradient computation.
All node methods are pure functions (no side effects) for JAX compatibility.

User Extensibility
------------------
Users can create custom nodes by extending NodeBase:

    class MyNode(NodeBase):
        def __init__(self, shape, name,
                     activation=IdentityActivation(),
                     energy=GaussianEnergy(),
                     latent_init=NormalInitializer(),
                     **kwargs):
            super().__init__(shape=shape, name=name, activation=activation,
                             energy=energy, latent_init=latent_init, **kwargs)

        @staticmethod
        def get_slots():
            return {"in": SlotSpec(name="in", is_multi_input=True)}

        @staticmethod
        def initialize_params(key, node_shape, input_shapes, weight_init, config):
            ...

        @staticmethod
        def predict(params, inputs, state, node_info):
            ...

Place the default activation/energy/initializer objects directly in the
``__init__`` signature, as above. Those objects are immutable — their base
classes block mutation after construction — so the single default instance is
safe to share across every node that does not override it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import copy
import types
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.topology import SlotRef, _get_current_namespace
from fabricpc.core.activations import ActivationBase, IdentityActivation
from fabricpc.core.energy import EnergyFunctional, GaussianEnergy
from fabricpc.core.initializers import InitializerBase, NormalInitializer


@dataclass(frozen=True)
class SlotSpec:
    """Specification for an input slot to a node."""

    name: str
    is_multi_input: bool  # True = multiple inputs allowed, False = single input only
    is_variance_scalable: bool = (
        True  # False = muPC leaves edges to this slot unscaled (scale 1.0)
    )
    is_skip_connection: bool = (
        False  # True = identity bypass path that counts toward muPC depth L
    )
    require_connected: bool = (
        False  # True = graph construction errors if this slot receives no edge
    )

    def __post_init__(self):
        if self.is_skip_connection and self.is_variance_scalable:
            raise ValueError(
                "is_skip_connection and is_variance_scalable were both set to True, but skip connection slots should NOT be subject to muPC variance scaling"
            )


@dataclass(frozen=True)
class Slot:
    """Runtime slot information with connected edges."""

    spec: SlotSpec
    in_neighbors: Dict[str, str]  # edge_key -> source_node_name mapping


class FlattenInputMixin:
    """
    Mixin providing flatten/reshape utilities for dense (fully-connected) nodes.

    Use this mixin when your node needs to:
    - Flatten arbitrary-shaped inputs to 2D for matrix multiplication
    - Reshape flat outputs back to a target shape
    """

    @staticmethod
    def flatten_input(x: jnp.ndarray) -> jnp.ndarray:
        """
        Flatten input tensor to 2D: (batch, *shape) -> (batch, numel).

        Args:
            x: Input tensor with batch dimension first

        Returns:
            Flattened tensor of shape (batch, numel)
        """
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)

    @staticmethod
    def reshape_output(x_flat: jnp.ndarray, out_shape: Tuple[int, ...]) -> jnp.ndarray:
        """
        Reshape flat tensor to target shape: (batch, numel) -> (batch, *out_shape).

        Args:
            x_flat: Flat tensor of shape (batch, numel)
            out_shape: Target shape (excluding batch dimension)

        Returns:
            Reshaped tensor of shape (batch, *out_shape)
        """
        batch_size = x_flat.shape[0]
        return x_flat.reshape(batch_size, *out_shape)

    @staticmethod
    def compute_linear(
        inputs: Dict[str, jnp.ndarray],
        weights: Dict[str, jnp.ndarray],
        batch_size: int,
        out_shape: Tuple[int, ...],
    ) -> jnp.ndarray:
        """
        Compute linear transformation: sum of (flattened_input @ weight) for each edge.

        Args:
            inputs: Dictionary mapping edge keys to input tensors
            weights: Dictionary mapping edge keys to weight matrices (in_numel, out_numel)
            batch_size: Batch size for output initialization
            out_shape: Target output shape (excluding batch)

        Returns:
            Pre-activation tensor of shape (batch, *out_shape)
        """
        out_numel = int(np.prod(out_shape))
        pre_activation_flat = jnp.zeros((batch_size, out_numel))

        for edge_key, x in inputs.items():
            x_flat = FlattenInputMixin.flatten_input(x)
            pre_activation_flat = pre_activation_flat + jnp.matmul(
                x_flat, weights[edge_key]
            )

        return FlattenInputMixin.reshape_output(pre_activation_flat, out_shape)


class NodeBase(ABC):
    """
    Abstract base class for all predictive coding nodes.

    All computation methods are pure functions (static, no side effects) for JAX
    compatibility. Nodes can have multiple input slots and custom transfer functions.

    Nodes are instantiated with their configuration, then finalized by the
    graph() builder which attaches topology info via copy-on-finalize.

    Subclasses implement three static methods: ``get_slots`` (the input slots),
    ``initialize_params`` (the weights and biases), and ``predict`` (the
    prediction z_mu plus any aux intermediates). Nodes with extra energy terms
    additionally override ``energy``. The error pair (error = z_latent - z_mu
    and its inverse z_latent = z_mu + error) and the assembly templates
    (``forward``, ``forward_with_aux``, ``forward_from_error``) are base-owned
    and are not override points.

    Subclasses set concrete default instances for activation, energy, latent_init,
    and weight_init directly in their ``__init__`` parameter defaults. Those
    default objects are immutable, so a single shared default instance cannot
    leak state across nodes.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: ActivationBase = IdentityActivation(),
        energy: EnergyFunctional = GaussianEnergy(),
        latent_init: InitializerBase = NormalInitializer(),
        weight_init: Optional[InitializerBase] = None,
        **extra_config,
    ):
        """
        Initialize a node descriptor.

        Args:
            shape: Output shape tuple (excluding batch dimension)
            name: Node name. Automatically prefixed with current GraphNamespace.
            activation: ActivationBase instance (default: IdentityActivation)
            energy: EnergyFunctional instance (default: GaussianEnergy)
            latent_init: InitializerBase instance (default: NormalInitializer)
            weight_init: InitializerBase instance, or None for a node with no
                weights (e.g. pooling)
            **extra_config: Node-specific config (use_bias, flatten_input, etc.)

        Raises:
            TypeError: If activation, energy, or latent_init is not an instance
                of its base class, or weight_init is neither an InitializerBase
                instance nor None. None is not a "use the default" sentinel;
                the defaults live in the signature.
        """
        for param, value, base in (
            ("activation", activation, ActivationBase),
            ("energy", energy, EnergyFunctional),
            ("latent_init", latent_init, InitializerBase),
        ):
            if not isinstance(value, base):
                raise TypeError(
                    f"Node '{name}': {param} must be an {base.__name__} "
                    f"instance; got {type(value).__name__}"
                )
        if weight_init is not None and not isinstance(weight_init, InitializerBase):
            raise TypeError(
                f"Node '{name}': weight_init must be an InitializerBase "
                f"instance, or None for a weight-free node; "
                f"got {type(weight_init).__name__}"
            )
        ns = _get_current_namespace()
        self._name = f"{ns}/{name}" if ns else name
        self._shape = tuple(shape)
        self._activation = activation
        self._energy = energy
        self._latent_init = latent_init
        self._weight_init = weight_init
        self._extra_config = types.MappingProxyType(
            extra_config
        )  # Immutable dictionary
        self._node_info = None  # Set by graph builder (copy-on-finalize)

    @property
    def name(self) -> str:
        """Node name, including namespace prefix if any."""
        return self._name

    @property
    def shape(self) -> Tuple[int, ...]:
        """Output shape excluding batch dimension."""
        return self._shape

    @property
    def node_info(self) -> NodeInfo:
        """NodeInfo with topology info. None until graph() is called."""
        return self._node_info

    def slot(self, slot_name: str):
        """
        Create a SlotRef for connecting edges to a specific slot.

        Args:
            slot_name: Name of the slot (e.g., "in", "mask")

        Returns:
            SlotRef pointing to this node's slot

        Raises:
            KeyError: If slot_name is not defined for this node type
        """
        slot_specs = type(self).get_slots()
        if slot_name not in slot_specs:
            raise KeyError(
                f"Node '{self._name}' has no slot '{slot_name}'. "
                f"Available: {list(slot_specs.keys())}"
            )
        return SlotRef(node=self, slot=slot_name)

    def _with_graph_info(self, node_info: NodeInfo) -> "NodeBase":
        """
        Copy-on-finalize: return a shallow copy with graph topology info attached.

        The original node object is not modified.
        """
        new = copy.copy(self)
        new._node_info = node_info
        return new

    # =========================================================================
    # Abstract methods - subclasses must implement
    # =========================================================================

    @staticmethod
    @abstractmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """
        Define the input slots for this node type.
        Create as many named input slots as needed, and specify whether each slot allows multiple inputs.
        Don't set is_multi_input=True unless you intend to aggregate an arbitrary number of inputs to a single named slot and create appropriate parameters and forward logic to handle that.

        Returns:
            Dictionary mapping slot names to SlotSpec objects

        Example:
            return {
                "in": SlotSpec(name="in", is_multi_input=True),
                "gate": SlotSpec(name="gate", is_multi_input=False)
            }
        """
        pass

    @staticmethod
    @abstractmethod
    def initialize_params(
        key: jax.Array,  # from jax.random.PRNGKey
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],  # edge_key -> source shape
        weight_init: Optional[InitializerBase],
        config: Dict[str, Any],
    ) -> NodeParams:
        """
        Define and initialize the parameters required for the node.

        Args:
            key: JAX random key
            node_shape: Output shape of this node (excluding batch dimension)
            input_shapes: Dictionary mapping edge keys to source node shapes
            weight_init: InitializerBase instance for weight initialization, or
                None for a weight-free node
            config: Node configuration (may contain initialization settings)

        Returns:
            NodeParams with initialized weights and biases
        """
        pass

    @staticmethod
    @abstractmethod
    def predict(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],  # keyed on EdgeInfo.key -> inputs data
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jnp.ndarray, Any]:
        """
        All parameterized computation: predict z_mu from the in-edge inputs.

        ``predict`` is a pure function: it has no side effects and expresses
        its dependence on ``params``, ``inputs``, and ``state`` entirely
        through JAX operations. The base templates differentiate through it
        (w.r.t. inputs and z_latent in ``forward_and_latent_grads``, w.r.t.
        params in ``forward_and_weight_grads``, and w.r.t. the relaxed errors
        in ePC's global energy gradient).

        How the prediction is produced is intentionally unconstrained: Linear
        does a matmul, IdentityNode sums its inputs, TransformerBlock runs an
        attention pipeline, StorkeyHopfield blends a probe with a learned
        weight matrix. The pair (error = z_latent - z_mu) and the energy call
        are applied by the base templates; do not compute them here.

        ``predict`` may read the node's own ``state``, but under
        ``EPCInference.derive_states`` the state carries the previous step's
        values: z_mu is always evaluated at the carried state, never at the
        latent being derived.

        muPC scaling is NOT applied here; the inference/learning callsite
        applies it. Do not scale inputs or gradients inside this method.

        See ``Linear`` (linear.py), ``IdentityNode`` (identity.py), and
        ``StorkeyHopfield`` (storkey_hopfield.py) for worked examples, and
        docs/user_guides/06_custom_nodes.md for the full guide.

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
            state: NodeState for this node
            node_info: NodeInfo object (contains activation, energy, etc.)

        Returns:
            Tuple of (z_mu, aux):
                - z_mu: prediction of this node's latent, shape
                  ``(batch,) + node_info.shape``.
                - aux: arbitrary pytree of intermediates for ``energy()``
                  (None if unused). aux must depend only on params and inputs
                  (e.g. Linear's pre_activation, StorkeyHopfield's
                  (W, strength)). aux is snapshotted when ``predict`` runs —
                  under ePC, before z_latent is derived — so an aux entry
                  computed from ``state.z_latent`` would freeze the carried
                  latent into an energy otherwise evaluated at the derived
                  latent, making sPC and ePC minimize different energies. An
                  energy term that needs the node's own latent reads
                  ``state.z_latent`` inside ``energy()`` instead.
        """
        pass

    @staticmethod
    def energy(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        aux: Any,
        node_info: NodeInfo,
    ) -> jnp.ndarray:
        """
        Per-sample energy (shape (batch,)) at (state.z_latent, state.z_mu).

        The default scores the node's energy functional. Override to add
        terms (StorkeyHopfield's attractor; ScaledSumNode's weighting in the
        external-node tests). A z_latent-dependent term must read
        ``state.z_latent`` here — never an aux entry computed from it: aux is
        snapshotted at ``predict`` time, which under ePC is before z_latent
        is derived, so such an entry would evaluate the energy at two
        different latents (see ``predict``).

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
            state: NodeState with z_latent and the freshly assigned z_mu/error
            aux: The intermediates ``predict`` returned (None on source nodes)
            node_info: NodeInfo object (contains the energy functional)

        Returns:
            Per-sample energy, shape (batch,).
        """
        energy_obj = node_info.energy
        return type(energy_obj).energy(state.z_latent, state.z_mu, energy_obj.config)

    # =========================================================================
    # Base-owned pair and assembly templates — NOT node override points.
    # The ePC <-> sPC equivalence requires one volume-preserving bijection
    # between error and z_latent shared by every node; tests audit that no
    # registered node overrides these.
    # =========================================================================

    @staticmethod
    def pair_error(z_latent: jnp.ndarray, z_mu: jnp.ndarray) -> jnp.ndarray:
        """sPC direction of the pair: error = z_latent - z_mu."""
        return z_latent - z_mu

    @staticmethod
    def pair_latent(z_mu: jnp.ndarray, error: jnp.ndarray) -> jnp.ndarray:
        """ePC direction of the pair — the inverse: z_latent = z_mu + error."""
        return z_mu + error

    @staticmethod
    def forward_with_aux(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[NodeState, Any]:
        """
        One predict -> pair -> energy pass, also surfacing predict's aux.

        Owns source semantics: an ``in_degree == 0`` node has no in-edges to
        project a z_mu, so z_mu mirrors z_latent (cast — a source z_latent
        may carry an int clamp dtype) with zero error, and ``predict`` is
        never called.
        """
        node_class = node_info.node_class
        if node_info.in_degree == 0:
            new_state = state._replace(
                z_mu=state.z_latent.astype(state.z_mu.dtype),
                error=jnp.zeros_like(state.error),
            )
            return (
                new_state._replace(
                    energy=node_class.energy(params, inputs, new_state, None, node_info)
                ),
                None,
            )
        z_mu, aux = node_class.predict(params, inputs, state, node_info)
        new_state = state._replace(
            z_mu=z_mu, error=node_class.pair_error(state.z_latent, z_mu)
        )
        return (
            new_state._replace(
                energy=node_class.energy(params, inputs, new_state, aux, node_info)
            ),
            aux,
        )

    @staticmethod
    def forward(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> NodeState:
        """
        Predict this node's latent state and report the resulting energy.

        sPC-direction template: z_mu from ``predict``, error =
        ``pair_error(z_latent, z_mu)``, energy from ``energy()``. Signature
        and semantics match the pre-split per-node ``forward`` bodies.
        """
        new_state, _ = node_info.node_class.forward_with_aux(
            params, inputs, state, node_info
        )
        return new_state

    @staticmethod
    def forward_from_error(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
        is_clamped: bool,
    ) -> NodeState:
        """
        ePC state derivation: ``state.error`` is the relaxed variable ε and
        z_latent is derived as ``pair_latent(z_mu, ε)`` — one ``predict`` per
        visit. Runs inside EPCInference's global ``jax.grad`` and is
        differentiable w.r.t. inputs and ``state.error``. Never writes
        ``latent_grad``.

        The clamp decides which side of the pair is free:
        - Unclamped: ε is relaxed; z_latent := z_mu + ε. This holds for every
          degree — top-down priors (in_degree == 0, whose z_mu is the constant
          assigned at initialization) and readouts (out_degree == 0) included.
        - Clamped: z_latent stays the clamp and ε is derived — with in-edges
          the sPC-direction template recomputes error = pair_error(clamp, z_mu)
          and energy(clamp, z_mu), the output loss; a clamped source keeps its
          init state (z_mu = clamp, error = 0) with nothing to recompute.
        """
        node_class = node_info.node_class
        if node_info.in_degree == 0:
            if is_clamped:
                return state  # clamp fixed at init; nothing to recompute
            # top-down prior: z_mu is the constant assigned at initialization
            return state._replace(
                z_latent=node_class.pair_latent(state.z_mu, state.error)
            )
        if is_clamped:
            new_state, _ = node_class.forward_with_aux(params, inputs, state, node_info)
            return new_state
        z_mu, aux = node_class.predict(params, inputs, state, node_info)
        eps = state.error  # the relaxed variable, written back bit-exact
        new_state = state._replace(
            z_latent=node_class.pair_latent(z_mu, eps), z_mu=z_mu, error=eps
        )
        return new_state._replace(
            energy=node_class.energy(params, inputs, new_state, aux, node_info)
        )

    # =========================================================================
    # muPC variance factor for scaling — override per node type
    # =========================================================================

    @staticmethod
    def get_variance_factor(
        source_shape: Tuple[int, ...],
        config: Dict[str, Any],
        weight_init: Optional[InitializerBase],
    ) -> float:
        """
        Return the factor by which this node's input transform multiplies input
        variance, before the activation. muPC scales each in-edge by

            a = gain / sqrt(v * K_slot)

        so v is what the scale undoes.

        For a matmul against unit-variance weights, v is the Kaiming fan_in:
        each output unit sums fan_in independent products, so variance scales
        by fan_in. The default below covers that case; override for
        node-specific transforms (ConvNode uses C_in * prod(kernel_size)).

        v is a float, not a dimension count, and may be below 1. A transform
        that *reduces* variance returns v < 1, which amplifies the edge scale:
        average pooling over n cells returns 1/n so that muPC applies sqrt(n).
        Weightless summation nodes return 1.0, leaving only the K_slot term.

        - flatten_input=True (dense): all dims flattened → prod(source_shape)
        - flatten_input=False (per-position): last-axis features only

        Args:
            source_shape: Shape of the source (presynaptic) node, excluding batch.
            config: Node configuration dictionary (e.g., kernel_size, flatten_input).
            weight_init: The node's weight initializer, or None for a weight-free
                node. Needed only by nodes that build weight matrices internally
                rather than one per in-edge (StorkeyHopfield's W); nodes whose
                per-edge weights muPC already accounts for ignore it.

        Returns:
            Variance factor v > 0 for the transform connecting source to this node.
        """
        if config.get("flatten_input", False):
            return float(np.prod(source_shape))
        # Typically nodes operate on the last (feature dimension)
        return float(source_shape[-1])

    # =========================================================================
    # Default implementations - can be overridden for explicit gradients
    # =========================================================================

    @staticmethod
    def forward_and_latent_grads(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
        is_clamped: bool,
    ) -> Tuple[NodeState, Dict[str, jnp.ndarray], jnp.ndarray]:
        """
        Forward pass with autodiff: computes updated state, gradients w.r.t.
        inputs (for updating upstream latents), and the self-latent gradient
        (dE/dz_latent).
        Called in the inference phase of predictive coding.

        Contract:
        1. In-degree-0 nodes short-circuit the gradient computation: the
           state comes from the template ``forward()`` (whose source guard
           mirrors z_mu <- z_latent with zero error) and all gradients are
           zero — the source's latent still moves via the contributions its
           downstream successors accumulate into its ``latent_grad``.
        2. Every node with in-degree > 0 — unclamped readouts included —
           takes the autodiff path: error = z_latent - z_mu, energy as
           ``forward()`` assigns, z_latent relaxed like any other node.
        3. The per-sample ``state.energy`` (shape (batch,)) is summed over
           the batch dimension to a scalar.
        4. ``jax.value_and_grad`` differentiates that scalar w.r.t. the
           input tensors and z_latent.

        Override this method to implement explicit (non-autodiff) gradient
        computation. When overriding, use ``energy.grad_latent()`` and
        ``activation.derivative()`` (or ``activation.jacobian()``) for
        analytical gradients.

        muPC scaling is NOT applied here — it is handled by the callsite
        (inference loop). Node methods are pure autodiff. The returned
        ``self_grad`` is the contribution from this node only, so the
        callsite scales it independently and adds it to ``state.latent_grad``
        without re-scaling pre-existing accumulated contributions from
        downstream successors.

        Args:
            params: Node parameters (weights, biases)
            inputs: Dictionary mapping edge keys to input tensors
                (already muPC-scaled by the callsite when scaling is active)
            state: NodeState for this node
            node_info: NodeInfo object
            is_clamped: Whether this node is clamped to data. The base body
                no longer branches on it (kept for overrides, e.g.
                EmbeddingNode, and for solver-side gating).

        Returns:
            Tuple of (NodeState, input_grads, self_grad):
                - NodeState: updated state (z_mu, error, energy).
                  ``latent_grad`` is *not* modified here.
                - input_grads: dict of gradients w.r.t. each input edge
                  (dE/d_input per edge), unscaled.
                - self_grad: dE/dz_latent contribution from this node,
                  unscaled, same shape as ``state.z_latent``.
        """
        node_class = node_info.node_class

        # Terminal source nodes: gradient short-circuit. The state comes from
        # the template forward(), whose in_degree == 0 guard owns source
        # semantics (z_mu <- z_latent cast to z_mu's dtype, zero error).
        # A source has no inputs and contributes no self-gradient; its latent
        # still moves via contributions accumulated by downstream successors.
        if node_info.in_degree == 0:
            new_state = node_class.forward(params, inputs, state, node_info)
            input_grads = {
                edge_key: jnp.zeros_like(inputs[edge_key]) for edge_key in inputs
            }
            self_grad = jnp.zeros_like(state.latent_grad)

        else:
            # Every node with in-edges — unclamped readouts included:
            # autodiff for input AND self-latent gradients.
            # Extract z_latent as a separate differentiable argument via closure.
            def energy_fn(input_args, z_latent):
                s = state._replace(z_latent=z_latent)
                new_s = node_class.forward(params, input_args, s, node_info)
                # Sum the per-sample energy over the batch dimension to the
                # scalar differentiated by autodiff
                total_energy = jnp.sum(new_s.energy)
                return total_energy, new_s

            (total_energy, new_state), (input_grads, self_grad) = jax.value_and_grad(
                energy_fn, argnums=(0, 1), has_aux=True
            )(inputs, state.z_latent)

        return new_state, input_grads, self_grad

    @staticmethod
    def forward_and_weight_grads(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[NodeState, NodeParams]:
        """
        Forward pass with autodiff: computes the node's local energy gradient w.r.t. weights.
        Called in the learning phase of predictive coding.

        Sums the per-sample ``state.energy`` (shape (batch,)) over the batch
        dimension to the scalar differentiated by ``jax.value_and_grad``
        w.r.t. params.

        Override this method to implement explicit weight gradient computation
        or apply node-specific post-processing (e.g., LayerNorm compensation).

        muPC scaling is NOT applied here — it is handled by the callsite
        (learning loop). Node methods are pure autodiff.

        Args:
            params: Current node parameters
            inputs: Dictionary with edge_key -> input tensor
                (already muPC-scaled by the callsite when scaling is active)
            state: NodeState for this node
            node_info: NodeInfo object

        Returns:
            Tuple of (NodeState, params_grad):
                - NodeState: updated node state
                - params_grad: NodeParams containing weight and bias gradients
        """
        node_class = node_info.node_class

        def energy_fn(p):
            new_s = node_class.forward(p, inputs, state, node_info)
            # Sum the per-sample energy over the batch dimension to the
            # scalar differentiated by autodiff
            total_energy = jnp.sum(new_s.energy)
            return total_energy, new_s

        (total_energy, new_state), params_grad = jax.value_and_grad(
            energy_fn, has_aux=True
        )(params)

        return new_state, params_grad


def compute_windowed_output_shape(
    in_spatial: Tuple[int, ...],
    kernel: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Any,
) -> Tuple[int, ...]:
    """Expected spatial output dims for a windowed op (conv or pooling).

    Mirrors ``lax.conv_general_dilated`` / ``lax.reduce_window`` (dilation=1),
    per spatial axis:

      * ``"SAME"``        -> ceil(in / stride)
      * ``"VALID"``       -> floor((in - k) / stride) + 1
      * ``(low, high)``   -> floor((in + low + high - k) / stride) + 1

    Validated empirically against both lax ops across stride/padding cases.
    Used by ConvNode and the pooling nodes to fail fast in ``initialize_params``
    when the declared node shape disagrees with what the op will produce
    (otherwise the mismatch only surfaces as an opaque error inside the JITted
    forward pass).
    """
    out = []
    for axis, (n, k, s) in enumerate(zip(in_spatial, kernel, stride)):
        if isinstance(padding, str):
            p = padding.upper()
            if p == "SAME":
                o = -(-n // s)  # ceil division
            elif p == "VALID":
                o = (n - k) // s + 1
            else:
                raise ValueError(
                    f"Unknown padding string {padding!r}; expected 'SAME' or 'VALID'."
                )
        else:
            lo, hi = padding[axis]
            o = (n + lo + hi - k) // s + 1
        if o <= 0:
            raise ValueError(
                f"Windowed op produces a non-positive output size ({o}) on "
                f"spatial axis {axis}: input={n}, kernel/window={k}, stride={s}, "
                f"padding={padding!r}. The kernel/window is larger than the "
                f"(padded) input along this axis."
            )
        out.append(o)
    return tuple(out)


def validate_windowed_output(
    node_shape: Tuple[int, ...],
    input_shapes: Dict[str, Tuple[int, ...]],
    window: Optional[Tuple[int, ...]],
    stride: Optional[Tuple[int, ...]],
    padding: Any,
    *,
    op_name: str,
    channels_preserved: bool,
    reject_pad_ge_window: bool = False,
) -> None:
    """Validate a windowed op's declared output shape against its inputs.

    Single source of truth for ConvNode and the pooling nodes — called from
    ``initialize_params`` (the only place input shapes are known), so windowed
    validation fires there rather than at construction. Raises ``ValueError``
    with a message naming both shapes and the offending edge on any
    disagreement. Checks, in order:

      * spatial rank ``len(node_shape) - 1`` is 1/2/3;
      * ``window`` and ``stride`` are both present — a missing key raises rather
        than silently skipping validation — and their lengths equal the rank;
      * with ``reject_pad_ge_window`` (max pooling), explicit ``(low, high)``
        padding stays below the window on every axis, since a fully-padded
        window fills with ``-inf``;
      * channels match the input when ``channels_preserved`` (pooling keeps the
        channel count; conv sets it from the kernel, so it passes False);
      * the computed spatial output equals the declared spatial shape.
    """
    spatial_rank = len(node_shape) - 1
    if spatial_rank not in (1, 2, 3):
        raise ValueError(
            f"{op_name} shape must have 2-4 elements (spatial_rank 1/2/3). "
            f"Got shape={node_shape}."
        )
    if window is None or stride is None:
        raise ValueError(
            f"{op_name}: both window/kernel and stride are required for output "
            f"shape validation; got window={window!r}, stride={stride!r}."
        )
    if len(window) != spatial_rank:
        raise ValueError(
            f"{op_name}: window/kernel length {len(window)} must equal "
            f"spatial_rank {spatial_rank} inferred from shape={node_shape}."
        )
    if len(stride) != spatial_rank:
        raise ValueError(
            f"{op_name}: stride length {len(stride)} must equal "
            f"spatial_rank {spatial_rank} inferred from shape={node_shape}."
        )
    if reject_pad_ge_window and not isinstance(padding, str):
        for axis, ((lo, hi), k) in enumerate(zip(padding, window)):
            if lo >= k or hi >= k:
                raise ValueError(
                    f"{op_name}: explicit padding {(lo, hi)} on spatial axis "
                    f"{axis} is >= the window size {k}. A fully-padded max-pool "
                    f"window would output -inf; reduce the padding below the "
                    f"window size."
                )

    declared_spatial = tuple(node_shape[:-1])
    declared_channels = node_shape[-1]
    for edge_key, in_shape in input_shapes.items():
        if channels_preserved and in_shape[-1] != declared_channels:
            raise ValueError(
                f"{op_name} preserves channels but declares {declared_channels} "
                f"while edge '{edge_key}' supplies {in_shape[-1]} (input shape "
                f"{tuple(in_shape)}). Pooling cannot change the channel count; "
                f"set the node's channel dimension to match its input."
            )
        expected_spatial = compute_windowed_output_shape(
            in_shape[:-1], window, stride, padding
        )
        if expected_spatial != declared_spatial:
            raise ValueError(
                f"{op_name} declared output spatial shape {declared_spatial} but "
                f"edge '{edge_key}' (input spatial {tuple(in_shape[:-1])}, window "
                f"{tuple(window)}, stride {tuple(stride)}, padding {padding!r}) "
                f"produces {expected_spatial}."
            )
