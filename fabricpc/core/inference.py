"""
Core inference dynamics for JAX predictive coding networks.

This module provides:
- InferenceBase: Abstract base class for inference algorithms
- InferenceSGD: Default SGD-based inference (z -= eta * grad)
- run_inference: Convenience function wrapping the class-based API

Inference algorithms control how latent states are updated during the
inference loop. The primary extension point is `compute_new_latent()`.

Usage:
    from fabricpc.core.inference import InferenceSGD

    structure = graph(
        nodes=[...], edges=[...], task_map=...,
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import types

import jax
import jax.numpy as jnp

from fabricpc.core.scaling import (
    scale_input_grads,
    scale_inputs,
    scale_self_grad,
)
from fabricpc.core.types import (
    GraphParams,
    GraphState,
    GraphStructure,
    NodeInfo,
    NodeState,
)
from fabricpc.core.state_ops import update_node_in_state

# =============================================================================
# Utility Functions
# =============================================================================


def gather_inputs(
    node_info: NodeInfo,
    structure: GraphStructure,
    state: GraphState,
) -> Dict[str, jax.Array]:
    """
    Gather inputs for a node from the graph structure.
    """
    in_edges_data = {}
    for edge_key in node_info.in_edges:
        edge_info = structure.edges[edge_key]  # get the edge object
        node = edge_info.source
        in_edges_data[edge_key] = state.nodes[
            node
        ].z_latent  # get the data sent along this edge

    return in_edges_data


# =============================================================================
# Inference Base Class
# =============================================================================


class InferenceBase(ABC):
    """
    Abstract base class for inference algorithms.

    Inference algorithms control how latent states are updated during the
    inference loop. The primary extension point is `compute_new_latent()`.

    Custom inference algorithms extend this class.

    Computation methods are classmethods dispatching on ``cls`` (pure
    functions of their arguments, no instance state); ``run_inference`` is an
    instance method because it reads ``self.config``.
    """

    def __init__(self, **config):
        self.config = types.MappingProxyType(config)  # Immutable dictionary

    @classmethod
    def inference_step(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """
        Single inference step: forward phase -> latent update.

        Override for algorithms that need a different phase structure
        (e.g., momentum that accumulates across steps).
        """
        # Phase 1: Zero the latent gradients
        state = cls.zero_grads(params, state, clamps, structure)

        # Phase 2: Forward pass to predict expected state (z_mu) & accumulate latent grads
        state = cls.forward_value_and_grad(params, state, clamps, structure)

        # Phase 3: Update the latent (z_state) with gradient
        state = cls.update_latents(params, state, clamps, structure, config)

        return state

    @staticmethod
    def zero_grads(
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Phase 1: Zero the latent gradients
        """
        # Phase 1: Zero latent gradients
        # Use latent_grad (not z_latent) as the dtype/shape source: z_latent may
        # carry an integer clamp dtype on source nodes, but gradients are float.
        for node_name in structure.nodes:
            node_state = state.nodes[node_name]
            zero_grad = jnp.zeros_like(node_state.latent_grad)
            state = update_node_in_state(state, node_name, latent_grad=zero_grad)

        return state

    @classmethod
    def forward_value_and_grad(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Phase 2: Forward pass value & grad; accumulate latent grads.

        This is the universal predictive coding mechanics shared by all inference algorithms.
        Phase 3 (latent update) is handled by the inference algorithm's latent_update() method.

        muPC scaling is applied here (pre-scale inputs, post-scale gradients),
        keeping node methods (forward_and_latent_grads) scaling-unaware. The
        self-grad contribution is scaled and *added* to ``latent_grad``
        without re-scaling pre-existing accumulations from previously
        iterated downstream successors — those were already scaled by
        their own ``topdown_grad_scale``.
        """
        for node_name in structure.nodes:
            # Get node and its info
            node = structure.nodes[node_name]
            node_info = node.node_info
            node_class = node_info.node_class
            node_state = state.nodes[node_name]
            node_params = params.nodes[node_name]

            # Gather inputs for each slot
            in_edges_data = gather_inputs(node_info, structure, state)

            # Pre-scale inputs by muPC forward scaling factors
            sc = node_info.scaling_config
            scaled_inputs = scale_inputs(in_edges_data, sc)

            # Compute predictions, error, input gradients, and the self-latent
            # gradient contribution (dE/dz_latent for this node only).
            node_state, inedge_grads, self_grad = node_class.forward_and_latent_grads(
                node_params,
                scaled_inputs,
                node_state,
                node_info,
                is_clamped=(node_name in clamps),
            )

            # Scale the new self-grad and add it to the accumulator. Any
            # contributions added earlier in this iteration by downstream
            # successors are preserved unchanged.
            inedge_grads = scale_input_grads(inedge_grads, sc)
            self_grad = scale_self_grad(self_grad, sc)
            node_state = node_state._replace(
                latent_grad=node_state.latent_grad + self_grad
            )

            # Update the graph state with node state containing errors and energy
            state = state._replace(nodes={**state.nodes, node_name: node_state})

            # Accumulate gradient contributions to pre-synaptic nodes (local backward pass to in-neighbors)
            for edge_key, grad in inedge_grads.items():
                source_name = structure.edges[edge_key].source
                latent_grad = state.nodes[source_name].latent_grad + grad
                state = update_node_in_state(
                    state, source_name, latent_grad=latent_grad
                )

        return state

    @classmethod
    def update_latents(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """
        Update latent states for each node based on the accumulated latent gradients.
        """
        for node_name in structure.nodes:
            node_state = state.nodes[node_name]

            # Use the inference algorithm's compute_new_latent() method to get the updated z_latent
            if node_name not in clamps:
                new_z_latent = cls.compute_new_latent(
                    node_name,
                    node_state,
                    config,
                )
                state = update_node_in_state(state, node_name, z_latent=new_z_latent)

        return state

    @staticmethod
    @abstractmethod
    def compute_new_latent(
        node_name: str,
        node_state: NodeState,
        config: Dict[str, Any],
    ) -> jnp.ndarray:
        """
        Compute the updated z_latent for a single node.
        This is the primary extension point for custom inference algorithms.

        Returns:
            Updated z_latent array
        """
        pass

    @classmethod
    def begin_segment(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Segment-boundary hook run before this solver's first inference step.

        The default is identity: the incoming z_latent/z_mu/error are taken
        exactly as the previous segment (or the initializer) left them.
        """
        return state

    @classmethod
    def finalize_state(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Segment-boundary hook run after this solver's last inference step.

        The default is identity. Solvers whose per-step state is not the
        consumable final state override this (e.g. a rebuild so downstream
        weight updates and dashboards read a self-consistent state).
        """
        return state

    def segments(self):
        """
        The (solver, steps) segments this inference object executes.

        A plain solver is a single segment of its own ``infer_steps``.
        Schedule objects override this to flatten their component solvers,
        so per-step consumers (e.g. tracking) can iterate segments instead
        of assuming one global step count.
        """
        return ((self, int(self.config["infer_steps"])),)

    def run_inference(
        self,
        params: GraphParams,
        initial_state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Outer inference loop using lax.fori_loop.

        Override for scan-based tracking, adaptive stopping, etc.
        infer_steps is read from self.config['infer_steps'].
        """
        cls = type(self)
        config = self.config
        infer_steps = config["infer_steps"]

        state = cls.begin_segment(params, initial_state, clamps, structure)

        def body_fn(t, state):
            return cls.inference_step(params, state, clamps, structure, config)

        # Use lax.fori_loop for efficiency
        state = jax.lax.fori_loop(0, infer_steps, body_fn, state)
        return cls.finalize_state(params, state, clamps, structure)


# =============================================================================
# Built-in Inference Algorithms
# =============================================================================


class InferenceSGD(InferenceBase):
    """
    Standard SGD inference: z -= eta * grad.

    This is the default inference algorithm for predictive coding networks.

    Args:
        eta_infer: Inference rate (default: 0.1)
        infer_steps: Number of inference iterations (default: 20)
    """

    def __init__(self, eta_infer=0.1, infer_steps=20, latent_decay=0.0):
        super().__init__(
            eta_infer=eta_infer, infer_steps=infer_steps, latent_decay=latent_decay
        )

    @staticmethod
    def compute_new_latent(node_name, node_state, config):
        # Precondition: node_state.z_latent is float here.
        # update_latents() gates calls with `node_name not in clamps`, so this
        # only runs on non-clamped nodes — and non-clamped z_latent is always
        # float (state initializers only propagate the clamp dtype to nodes
        # that are clamped). A clamped node's int z_latent (e.g. token
        # indices) never reaches this update path.
        eta_infer = config["eta_infer"]
        latent_decay = config["latent_decay"]

        new_latent = (
            node_state.z_latent * (1.0 - eta_infer * latent_decay)
            - eta_infer * node_state.latent_grad
        )
        return new_latent


class InferenceSGDNormClip(InferenceBase):
    """
    SGD inference with per-node gradient norm clipping: z -= eta * clip(grad).

    Clips the L2 norm of each node's latent gradient independently per sample.
    If ||grad|| > max_norm, scales grad down to max_norm.
    Uses safe division with epsilon to handle zero gradients.

    Args:
        eta_infer: Inference rate (default: 0.1)
        infer_steps: Number of inference iterations (default: 20)
        max_norm: Maximum gradient norm per node (default: 1.0)
        eps: Small constant for numerical stability (default: 1e-8)
    """

    def __init__(
        self, eta_infer=0.1, infer_steps=20, latent_decay=0.0, max_norm=1.0, eps=1e-8
    ):
        super().__init__(
            eta_infer=eta_infer,
            latent_decay=latent_decay,
            infer_steps=infer_steps,
            max_norm=max_norm,
            eps=eps,
        )

    @staticmethod
    def compute_new_latent(node_name, node_state, config):
        eta_infer = config["eta_infer"]
        latent_decay = config["latent_decay"]
        max_norm = config["max_norm"]
        eps = config["eps"]

        grad = node_state.latent_grad
        # Per-sample L2 norm (sum over all non-batch dims)
        grad_norm = jnp.sqrt(
            jnp.sum(grad.conj() * grad, axis=tuple(range(1, grad.ndim)), keepdims=True)
        )
        clip_factor = jnp.minimum(1.0, max_norm / (grad_norm + eps))
        clipped_grad = grad * clip_factor

        return (
            node_state.z_latent * (1.0 - eta_infer * latent_decay)
            - eta_infer * clipped_grad
        )


class InferenceSchedule(InferenceBase):
    """
    Composable inference schedule: solvers run as segments per weight update.

    Usage:
        inference = InferenceSchedule(
            EPCInference(eta_infer=1e-2, infer_steps=5),
            InferenceSGD(eta_infer=0.05, infer_steps=20),
        )

    The example composes a few cheap global ePC steps to near-equilibrium
    with sPC refinement on the true arbitrary-graph energy (back edges
    included), warm-started from ePC's solution.

    Chained execution contract:
    1. Node states are initialized once, by the graph's configured
       initializer, before the first segment; no segment re-initializes.
    2. Each solver receives z_latent, z_mu, and error exactly as the
       previous segment (or the initializer) left them — no resync, no
       re-derivation at the boundary. Each solver applies its own
       ``begin_segment``/``finalize_state``.
    3. The next solver continues from the resulting state (after e.g. ePC's
       ``finalize_state`` rebuild).

    A schedule has no single per-step rule, so ``inference_step`` and
    ``compute_new_latent`` raise; per-step consumers (tracking) iterate
    ``segments()`` instead, which flattens nested schedules.
    """

    def __init__(self, *solvers):
        if not solvers:
            raise ValueError("InferenceSchedule requires at least one solver")
        for solver in solvers:
            if not isinstance(solver, InferenceBase):
                raise TypeError(
                    f"InferenceSchedule accepts InferenceBase instances; "
                    f"got {type(solver).__name__}"
                )
        super().__init__(solvers=tuple(solvers))

    def run_inference(
        self,
        params: GraphParams,
        initial_state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """Fold the state through each solver's run_inference in order."""
        state = initial_state
        for solver in self.config["solvers"]:
            state = solver.run_inference(params, state, clamps, structure)
        return state

    def segments(self):
        """Flatten component segments — nested schedules compose."""
        flattened = []
        for solver in self.config["solvers"]:
            flattened.extend(solver.segments())
        return tuple(flattened)

    @classmethod
    def inference_step(cls, params, state, clamps, structure, config):
        raise NotImplementedError(
            "InferenceSchedule has no single per-step rule; iterate "
            "segments() and step each segment's own solver."
        )

    @staticmethod
    def compute_new_latent(node_name, node_state, config):
        raise NotImplementedError(
            "InferenceSchedule has no single per-step rule; iterate "
            "segments() and step each segment's own solver."
        )


# =============================================================================
# Convenience Function
# =============================================================================


def run_inference(
    params: GraphParams,
    initial_state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
) -> GraphState:
    """
    Run inference using the algorithm object stored in the graph structure.

    Convenience wrapper that extracts the inference object from
    ``structure.config["inference"]`` and delegates to its
    ``run_inference`` method.

    Args:
        params: Graph parameters.
        initial_state: Initial graph state.
        clamps: Clamped node values.
        structure: Graph structure containing the inference object.

    Returns:
        Converged graph state after inference.
    """
    inference_object = structure.config["inference"]
    return inference_object.run_inference(params, initial_state, clamps, structure)
