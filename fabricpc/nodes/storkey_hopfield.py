"""
Storkey Hopfield associative memory node for predictive coding networks.
Acts as a filter selective for stored patterns in the latent space, with
energy-based learning of attractors.

Implements a Hopfield memory layer that combines standard PC prediction-error
energy with a Hopfield attractor energy term. The Hopfield energy pulls the
node's latent state z toward stored patterns (attractors in z-space), while
the PC energy pulls z toward the upstream prediction mu. The equilibrium z*
is the PC-optimal compromise between top-down expectation and internal memory.

Energy formulation:
    E_total = E_pc + hopfield_strength * E_hop

    E_pc = 0.5 ||z - mu||^2              (Gaussian, or user-specified)
    E_hop = (1/2D) z^T (W^2 - W) z       (Hopfield attractor energy)

    where D = dimension of z (last axis) for scale-invariance.

The standard PC energy path (via energy_functional()) is called normally.
The Hopfield energy is added afterward via accumulate_hopfield_energy(),
which augments state.energy. The gradient dE_hop/dz is computed
automatically via autodiff in forward_and_latent_grads().

Attractor dynamics arise naturally from the Hopfield energy gradient
(strength/D)(W^2 - W)z, which is accumulated to latent_grad and applied
during PC inference (z -= eta * latent_grad). Since PC inference IS gradient
descent on E_total, the inference loop itself iterates z toward the
attractors encoded in W.

The input from the upstream node serves as a "probe pattern". The prediction
z_mu is a strength-weighted blend of the probe (residual) and its projection
through W:

    z_mu = activation(probe / (1+s) + (probe @ W) * s / (1+s) + bias)

where s = hopfield_strength. This ensures:
    - At s=0: z_mu = activation(probe) — identity pass-through, W is unused.
    - At s→large: z_mu = activation(probe @ W) — learned projection dominates.
    - At moderate s: smooth interpolation; the residual provides the
      information pathway while W specializes in attractor corrections.

The residual is critical: without it, W must simultaneously serve as the sole
information pathway (prediction) AND define the attractor landscape, and the
prediction role dominates. The residual decouples these roles, allowing W to
develop attractor structure that provides a denoising benefit at moderate
strength (inverted-U response to strength).

Use moderate values for hopfield_strength (s) because it scales the hopfield energy term without normalization.

muPC scaling:
    The blend shrinks variance, and the "in" slot is scalable so that muPC
    undoes exactly that shrinkage. The coefficients 1/(1+s) and s/(1+s) sum
    to 1, but that preserves scale only for perfectly correlated terms. The
    probe and probe @ W are close to independent, so their variances add in
    quadrature:

        v(s) = Var(blend)/Var(probe) = (1 + s^2 * r) / (1 + s)^2

    with r = Var(probe @ W)/Var(probe) = sum_i E[W_ij^2], the variance gain
    of one pass through W. v(s) <= 1 for every s >= 0, reaching its minimum
    r/(1+r) at s = 1/r.

    Under the node's own initialization — Xavier on (D, D), so E[W_ij^2] =
    1/D, followed by _prepare_W's symmetrization W <- (W + W^T)/2, which
    halves off-diagonal variance and leaves the diagonal alone:

        r = (D-1)/(2D) + 1/D = (D+1)/(2D)   ->  1/2 for large D

    So the worst case is s = 2, v = 1/3, a per-node standard-deviation ratio
    of 0.577; at the default learnable strength (softplus(raw) = 1.0 at init)
    v = 0.375, ratio 0.612. Left uncorrected that compounds — a chain of 32
    such nodes attenuates by 0.612^32 ~ 1e-7. get_variance_factor returns
    v(s), so muPC applies a = gain/sqrt(v(s) * K_slot) and the node holds
    O(1) variance at any s and any depth.

    The blend is linear in the probe, so the edge scale factors straight out:
    activation(a * [blend] + b). At s = 0, v = 1 and a = gain, the same
    treatment every other muPC node's identity-like path receives. Nothing is
    scaled twice: muPC scales the edge, W's own scale is set by its
    initializer, and v accounts for both.

    Two caveats, both inherent to a scale computed once from topology:
      - r depends on weight_init, which is why get_variance_factor receives
        it and derives r from InitializerBase.element_variance rather than
        assuming Xavier.
      - With hopfield_strength=None (the default) s is learnable, so v is
        computed at graph construction from s = softplus(raw_init) = 1.0 and
        does not track training. This is muPC's standard contract: scales are
        static and correct at initialization. The drift is bounded — v ranges
        only over [r/(1+r), 1], i.e. [1/3, 1] at r = 1/2, so the scale is
        never off by more than sqrt(3) ~ 1.73.

    The slot is not a skip connection (is_skip_connection=False): the
    activation wraps the identity path, so no raw identity stream passes
    through the node. It is not a residual-stream merge and must not count
    toward the residual depth L — counting it would raise L and over-damp
    the graph's true merges.

Notation:
    xi^mu = stored Hopfield patterns (absolute states in z-space)
    epsilon = z - mu = PC prediction error (context-dependent residual)
    These are completely different objects. The Hopfield attractor operates
    on z (absolute state), not on epsilon (prediction error).

Architecture (internal to forward()):

    probe ──┬──► probe / (1+s) ──────────┐
            │                            ├──► + bias ──► activation ──► z_mu
            └──► (probe @ W) * s/(1+s) ──┘

Recurrency comes from the Hopfield energy gradient during inference steps.

    error = z - z_mu ──► PC energy (E_pc) ──► z_latent ──► Hopfield energy (E_hop)
                                                 ▲                    │
                                                 └────────────────────┘
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import TanhActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import (
    ZerosInitializer,
    NormalInitializer,
    XavierInitializer,
    element_variance,
    initialize,
)

if TYPE_CHECKING:
    from fabricpc.core.activations import ActivationBase
    from fabricpc.core.energy import EnergyFunctional
    from fabricpc.core.initializers import InitializerBase


def inverse_softplus(x: jax.Array) -> jax.Array:
    """Inverse of jax.nn.softplus.

    Computes log(exp(x) - 1) in a numerically stable way using expm1.
    Useful for initializing a raw parameter `raw` such that
    `jax.nn.softplus(raw)` equals a desired positive target value.
    """
    return jnp.log(jnp.expm1(x))


class StorkeyHopfield(NodeBase):
    """
    Hopfield associative memory node with energy-based learning.

    Combines standard PC prediction-error energy with a Hopfield attractor
    energy term. The two energies compete on z_latent:

    - E_pc pulls z toward the upstream prediction mu
    - E_hop pulls z toward stored patterns (attractors learned via W)

    The forward pass computes z_mu as a strength-weighted blend:
        z_mu = activation(probe/(1+s) + (probe @ W)*s/(1+s) + bias)
    where s = hopfield_strength. At s=0 this is a pure pass-through; at
    large s the learned projection probe @ W dominates. Attractor dynamics
    come from the Hopfield energy gradient accumulated to latent_grad during
    forward(), which the PC inference loop applies via z -= eta * latent_grad.
    W is a (D, D) matrix on the last axis of z_latent.

    Args:
        shape: Output shape (excluding batch). Last dim is the Hopfield dimension D.
        name: Node name.
        activation: Activation function (default: TanhActivation).
        energy: Energy functional for PC term (default: GaussianEnergy).
        hopfield_strength: Scaling of E_hop relative to E_pc.
            If None (default), a learnable scalar initialized to 1.0.
            If a float, fixed at that value.
        use_bias: Whether to include bias (default: False).
        enforce_symmetry: Symmetrize W via 0.5*(W+W.T) in forward (default: True).
        zero_diagonal: Zero W diagonal in forward (default: False).
        weight_init: Initializer for weights (default: XavierInitializer()).
        bias_init: Initializer for biases (default: ZerosInitializer()).
        latent_init: Initializer for latent states (default: NormalInitializer()).
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        name: str,
        activation: ActivationBase = TanhActivation(),
        energy: EnergyFunctional = GaussianEnergy(),
        hopfield_strength: Optional[float] = None,
        use_bias: bool = False,
        enforce_symmetry: bool = True,
        zero_diagonal: bool = False,
        latent_init: InitializerBase = NormalInitializer(),
        weight_init: InitializerBase = XavierInitializer(),
        bias_init: InitializerBase = ZerosInitializer(),
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            bias_init=bias_init,
            hopfield_strength=hopfield_strength,
            use_bias=use_bias,
            enforce_symmetry=enforce_symmetry,
            zero_diagonal=zero_diagonal,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        """One single-input slot. Input shape must match node output shape (D).

        Variance-scalable: the blend shrinks variance by v(s), and
        get_variance_factor reports v(s) so muPC's edge scale undoes it
        exactly. Not a skip connection: the activation wraps the identity
        path, so the node is not a residual-stream merge and must not count
        toward the residual depth L. See the module docstring, "muPC scaling".
        """
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def get_variance_factor(
        source_shape: Tuple[int, ...],
        config: Dict[str, Any],
        weight_init: Optional[InitializerBase],
    ) -> float:
        """Variance factor of the strength-weighted blend:

            v(s) = (1 + s^2 * r) / (1 + s)^2

        where s is the Hopfield strength and r = sum_i E[W_ij^2] is the
        variance gain of one pass through W. r is derived from the
        initializer's element variance and from _prepare_W's constraints,
        which change W's second moment: symmetrization W <- (W + W^T)/2
        halves the off-diagonal variance and leaves the diagonal alone, and
        zero_diagonal drops the diagonal term.

        With a learnable strength (hopfield_strength=None) s is evaluated at
        its initial value, softplus(raw_init) = 1.0. See the module
        docstring, "muPC scaling", for the derivation and the bounds.
        """
        D = source_shape[-1]
        sigma_sq = element_variance((D, D), weight_init)

        off_diag = sigma_sq / 2.0 if config.get("enforce_symmetry", True) else sigma_sq
        diag = 0.0 if config.get("zero_diagonal", False) else sigma_sq
        r = (D - 1) * off_diag + diag

        s = config.get("hopfield_strength")
        if s is None:
            # Learnable: raw is initialized so that softplus(raw) = 1.0.
            s = 1.0

        return (1.0 + s**2 * r) / (1.0 + s) ** 2

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: InitializerBase,
        config: Optional[Dict[str, Any]],
    ) -> NodeParams:
        """
        Initialize Hopfield W matrix and biases.

        Weights:
            - Hopfield W: (D, D) stored under the input edge key

        Biases:
            - "b": bias vector if use_bias
            - "hopfield_strength": learnable scalar if hopfield_strength is None
        """

        # weight_init is defaulted once, in the __init__ signature, and flows
        # in via node_info.weight_init. It is not re-defaulted here — the
        # single source of truth is the constructor.

        # Weights
        weights_dict = {}
        D = node_shape[-1]
        key_hop, key_b = jax.random.split(key)

        # Hopfield W: (D, D)
        W = initialize(key_hop, (D, D), weight_init)
        # Apply symmetry/diagonal constraints at init time too
        W = StorkeyHopfield._prepare_W(W, config)

        # This node takes a single input edge.
        edge_key, in_shape = next(iter(input_shapes.items()))
        if in_shape[-1] != D:
            raise ValueError(
                f"Input last dimension {in_shape[-1]} must match node output last dimension {D} for Hopfield recurrence."
            )
        weights_dict[edge_key] = (
            W  # Store W under the input edge key for gradient flow to presynaptic node.
        )

        # Bias shape broadcasts over the batch dim: (1, ..., 1, out_features).
        biases = {}
        if config["use_bias"]:
            bias_shape = (1,) * len(node_shape) + (node_shape[-1],)
            biases["b"] = initialize(key_b, bias_shape, config["bias_init"])

        # Learnable hopfield_strength: stored as an unconstrained raw parameter
        # Softplus is applied in forward() to guarantee effective strength >= 0
        # Init raw so that softplus(raw) = 1.0
        hopfield_strength = config["hopfield_strength"]
        if hopfield_strength is None:
            biases["hopfield_strength"] = inverse_softplus(jnp.array(1.0))

        return NodeParams(weights=weights_dict, biases=biases)

    @staticmethod
    def _prepare_W(W: jnp.ndarray, config: Dict[str, Any]) -> jnp.ndarray:
        """Symmetrize and/or zero diagonal of W based on config.

        Both operations are differentiable, safe under JAX tracing.
        """
        D = W.shape[0]
        if config.get("enforce_symmetry", True):
            W = 0.5 * (W + W.T)
        if config.get("zero_diagonal", False):
            W = W * (1.0 - jnp.eye(D))
        return W

    @staticmethod
    def predict(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jax.Array]]:
        """
        Prediction: compute z_mu from the input probe.

        z_mu = activation(probe/(1+s) + (probe @ W)*s/(1+s) + bias)

        where s = hopfield_strength. At s=0 z_mu = activation(probe)
        (identity pass-through); at large s z_mu ≈ activation(probe @ W).

        Returns:
            Tuple of (z_mu, (W, strength)): the prepared weight matrix and
            effective strength feed the Hopfield attractor term in
            ``energy()``. Both depend only on params and config, so the aux
            snapshot is valid under both solver directions.
        """
        config = node_info.node_config

        edge_key, input_probe_state = next(iter(inputs.items()))

        # Symmetrize the weights
        W = StorkeyHopfield._prepare_W(params.weights[edge_key], config)

        # Learnable strength is stored raw; softplus keeps the effective value >= 0.
        if "hopfield_strength" in params.biases:
            strength = jax.nn.softplus(params.biases["hopfield_strength"])
        else:
            strength = config["hopfield_strength"]

        # Strength-weighted blend: probe/(1+s) + (probe @ W)*s/(1+s).
        # The probe residual carries the information pathway while
        # W specializes in attractor corrections. At s=0 it is a pure
        # pass-through; the learned projection dominates as s grows.
        blend = 1.0 / (1.0 + strength)
        pre_activation = input_probe_state * blend + (input_probe_state @ W) * (
            1.0 - blend
        )

        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        activation = node_info.activation
        z_mu = type(activation).forward(pre_activation, activation.config)
        return z_mu, (W, strength)

    @staticmethod
    def energy(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        aux: Tuple[jnp.ndarray, jax.Array],
        node_info: NodeInfo,
    ) -> jnp.ndarray:
        """Combined energy: E_pc(z, z_mu) + s * E_hop(W, z).

        The Hopfield attractor term E_hop = (1/2D) z^T (W^2 - W) z pulls z
        toward stored patterns (attractors in z-space). Combined with the PC
        energy (which pulls z toward the upstream prediction mu), the
        equilibrium z* is the PC-optimal compromise between top-down
        expectation and internal memory prior. It reads ``state.z_latent``
        here — the z_latent-dependent-term rule — so both solver directions
        evaluate it at the same latent as the PC term. The gradient
        dE_hop/dz = (s/D)(W^2 - W)z arrives via autodiff.
        """
        W, strength = aux
        energy = NodeBase.energy(params, inputs, state, aux, node_info)

        z = state.z_latent  # (batch, ..., D)
        wz = z @ W  # (batch, ..., D)
        D = z.shape[-1]
        E_hopfield = (0.5 / D) * jnp.sum(wz * (wz - z), axis=-1)  # (1/2D) z^T(W^2-W)z
        return energy + strength * E_hopfield
