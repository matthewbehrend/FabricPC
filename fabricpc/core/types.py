"""
Core JAX types for predictive coding networks.

All types are immutable and registered as JAX pytrees for automatic differentiation.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, Tuple, NamedTuple, TYPE_CHECKING
import jax.numpy as jnp
from jax import tree_util
from dataclasses import dataclass

if TYPE_CHECKING:
    from fabricpc.nodes.base import NodeBase
    from fabricpc.core.mupc import MuPCScalingFactors


@dataclass(frozen=True)
class SlotInfo:
    """Metadata for an input slot to a node."""

    name: str  # Slot name (e.g., "in")
    parent_node: str  # Name of the parent node
    is_multi_input: bool  # True if slot accepts multiple edges, False for single edge
    is_variance_scalable: (
        bool  # muPC scales edges to this slot (True) or leaves at 1.0 (False)
    )
    is_skip_connection: bool  # Identity bypass path that counts toward muPC depth L
    in_neighbors: Tuple[str, ...]  # Tuple of node names connecting to this slot


@dataclass(frozen=True)
class NodeInfo:
    """Metadata for a single node in the graph.

    The shape field represents the output dimensions of the node, excluding
    the batch dimension. Batch is always the first dimension in arrays
    and is stored in GraphState.batch_size at runtime.

    Examples:
        - Linear layer: shape=(128,) for 128-dimensional output
        - Image: shape=(28, 28, 1) for 28x28 grayscale image (NHWC format)
        - Sequence: shape=(100, 64) for 100 timesteps, 64 features
    """

    name: str
    shape: Tuple[int, ...]  # Output shape excluding batch dimension
    node_type: str  # "linear", "transformer", etc. (kept for debugging/display)
    node_class: NodeBase  # The node class (Linear, TransformerBlock, etc.)
    node_config: Dict[str, Any]  # Extra config (use_bias, flatten_input, etc.)
    activation: Any  # ActivationBase instance
    energy: Any  # EnergyFunctional instance
    latent_init: Any  # InitializerBase instance
    weight_init: Any  # InitializerBase instance, or None for a weight-free node
    slots: Dict[str, SlotInfo]  # {"in": SlotInfo, ...}
    in_degree: int  # Number of incoming edges
    out_degree: int  # Number of outgoing edges
    in_edges: Tuple[str, ...]  # Tuple of edge keys
    out_edges: Tuple[str, ...]  # Tuple of edge keys
    scaling_config: Optional["MuPCScalingFactors"] = (
        None  # MuPCScalingFactors instance for muPC parameterization, or None
    )


@dataclass(frozen=True)
class EdgeInfo:
    """Metadata for a single edge in the graph."""

    key: str  # "source->target:slot"
    source: str
    target: str
    slot: str


class NodeParams(NamedTuple):
    """Parameters for a single node (weights, biases, etc.)."""

    weights: Dict[
        str, jnp.ndarray
    ]  # Named weight matrices, where name identifies the substructure of the node for the parameters
    biases: Dict[str, jnp.ndarray]  # Named bias vectors


class GraphParams(NamedTuple):
    """
    Learnable parameters of the predictive coding network.

    Now organized by node rather than edge, supporting complex nodes with
    multiple internal parameters (e.g., transformer blocks).

    Attributes:
        nodes: Dictionary mapping node names to their parameters
            Each node has a dict of weights and a dict of biases
            Complex nodes may have multiple named weight/bias matrices
    """

    nodes: Dict[str, NodeParams]  # {node_name: NodeParams}

    def __repr__(self) -> str:
        n_nodes = len(self.nodes)
        total_params = 0
        for node_params in self.nodes.values():
            if "weights" in node_params._fields:
                total_params += sum(w.size for w in node_params.weights.values())
            if "biases" in node_params._fields:
                total_params += sum(b.size for b in node_params.biases.values())
        return f"GraphParams(nodes={n_nodes}, total_params={total_params})"


class NodeState(NamedTuple):
    """
    Dynamic state of the Node during inference.

    Attributes:
        z_latent: Latent states (what the network infers)
        z_mu: Predicted expectations (what the network predicts)
        error: Prediction errors (z_latent - z_mu); under ``EPCInference``
            the first-class relaxed variable ε, with z_latent derived as
            z_mu + ε
        energy: Energy
        latent_grad: Gradient accumulator for inference updates — under
            state-based solvers the one-hop accumulated dE/dz_latent; under
            ``EPCInference`` the gradient of the total energy w.r.t. ε
            through the full derived forward
    """

    z_latent: jnp.ndarray
    z_mu: jnp.ndarray
    error: jnp.ndarray
    energy: jnp.ndarray  # per-sample energy, shape (batch_size,)
    latent_grad: jnp.ndarray  # accumulator: sPC dE/dz_latent, ePC dE/depsilon


class GraphState(NamedTuple):
    """
    Dynamic state of the network during inference.

    All states are dictionaries mapping node names to arrays.

    Attributes:
        nodes: Dictionary mapping node names to NodeState
        batch_size: Current batch size
    """

    nodes: Dict[str, NodeState]  # {node_name: NodeState}
    batch_size: int

    def __repr__(self) -> str:
        n_nodes = len(self.nodes)
        return f"GraphState(nodes={n_nodes}, batch_size={self.batch_size})"


class GraphStructure(NamedTuple):
    """
    Static graph topology (compile-time constant).

    This structure is immutable and marked as static during JIT compilation.

    Attributes:
        nodes: Dictionary mapping node names to NodeBase instances (with node_info attribute)
        edges: Dictionary mapping edge keys to EdgeInfo
        task_map: Dictionary mapping task names to node names
        node_order: Unique topological node order — first_occurrence_order(schedule)
        schedule: Full node visit schedule; on cyclic graphs built with
            graph(..., unroll=U), cycle members repeat U times
        config: Graph configuration
    """

    nodes: Dict[str, Any]  # Dict[str, NodeBase] - node instances with node_info
    edges: Dict[str, EdgeInfo]
    task_map: Dict[str, str]
    node_order: Tuple[str, ...]  # Unique node order (one entry per node)
    schedule: Tuple[str, ...]  # Visit schedule (cycle members may repeat)
    config: Dict[str, Any]  # Graph configuration

    def __repr__(self) -> str:
        n_nodes = len(self.nodes)
        n_edges = len(self.edges)
        return f"GraphStructure(nodes={n_nodes}, edges={n_edges})"


# Register as pytrees for JAX transformations
tree_util.register_pytree_node(
    GraphParams,
    lambda gp: ((gp.nodes,), None),
    lambda aux, children: GraphParams(*children),
)

tree_util.register_pytree_node(
    NodeParams,
    lambda np: ((np.weights, np.biases), None),
    lambda aux, children: NodeParams(*children),
)

tree_util.register_pytree_node(
    NodeState,
    lambda ns: (
        (
            ns.z_latent,
            ns.z_mu,
            ns.error,
            ns.energy,
            ns.latent_grad,
        ),
        None,
    ),
    lambda aux, children: NodeState(*children),
)

tree_util.register_pytree_node(
    GraphState,
    lambda gs: (
        (gs.nodes,),  # Dynamic children (differentiable)
        (gs.batch_size,),  # Static auxiliary data (metadata)
    ),
    lambda aux, children: GraphState(children[0], aux[0]),
)

# GraphStructure is static, so we register it as having no dynamic components
tree_util.register_pytree_node(
    GraphStructure,
    # Aux tuple order must match the NamedTuple field order: the unflatten
    # below rebuilds positionally.
    lambda gs: (
        (),
        (gs.nodes, gs.edges, gs.task_map, gs.node_order, gs.schedule, gs.config),
    ),
    lambda aux, _: GraphStructure(*aux),  # Reconstruct from aux data
)
