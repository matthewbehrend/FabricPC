"""
muPC (Maximal Update Parameterization for Predictive Coding) scaling computation.

This module computes per-node scaling factors that maintain O(1) variance of
activations, prediction errors, and gradients across networks of arbitrary
width and topology. Based on:

  - Depth-muP (Yang et al.) — width+depth scaling for deep networks
  - muPC (Innocenti et al., arXiv:2505.13124) — adaptation to predictive coding

Scaling is computed per in-edge, based on the target slot's properties:

  - Edges arriving at a slot with is_variance_scalable=True get

        a = gain / sqrt(v * K_slot)          (non-merge nodes)
        a = gain / sqrt(v * K_slot * L)      (merge nodes)

    where v is the target node's input-transform variance factor (from
    get_variance_factor()) — the Kaiming fan_in for a weight matrix, 1.0
    for a weightless sum, 1/n for an average pool over n cells —
    K_slot is the in-degree of the *specific slot* (not the whole node),
    gain = activation.variance_gain() is the Kaiming-style gain, and
    L is the residual depth. A merge node is a node with at least one
    connected is_skip_connection slot — the point where a computed branch
    joins the residual stream. The 1/sqrt(L) damps each branch
    contribution once, at the merge; edges elsewhere (stems,
    branch-interior layers, stream projections, post-stream layers)
    carry no depth factor.

  - Edges arriving at a slot with is_variance_scalable=False pass through
    at scale 1.0. This preserves the identity mapping that carries signal
    through deep residual networks.

  - Edges into output nodes (out_degree=0, with include_output=True) get

        a = gain / (v * sqrt(K_slot))

    with no depth factor: the readout is applied once to the O(1) stream,
    not summed L times. This matches a_L = 1/N in Table 1 of the muPC
    paper (arXiv:2505.13124), where N is the readout fan_in.

Residual depth L is the number of nodes along the longest path that have
at least one connected slot with is_skip_connection=True (a slot with at
least one incoming edge). These are the variance-accumulating merge
points where skip and compute paths sum. For pure sequential chains
(no skip connections), L=1. For residual networks with D blocks, L=D.
Only is_skip_connection slots contribute to L; is_variance_scalable
controls edge scaling, never L. Non-scalable non-skip slots (e.g.,
metadata like attention masks) and declared-but-unconnected skip slots
(e.g., a LinearResidual used without a skip edge) do not count.

  input → h1 → skip1(+) → h2 → skip2(+) → h3 → skip3(+) → output
           │      ↑         │      ↑         │      ↑
           └──────┘         └──────┘         └──────┘
              L=1              L=2              L=3

Top-down gradient scaling combines chain rule correction with Jacobian
compensation for deep gradient propagation:

    c_td = a * jacobian_gain

Chain rule correction (a): Because scale_inputs() (in scaling.py)
pre-scales inputs (x -> a*x) before the value_and_grad closure,
autodiff yields dE/d(a*x). Multiplying by a restores dE/dx.

Jacobian compensation (jacobian_gain): The per-hop Jacobian
diag(act'(z)) @ (a*W) has RMS singular value ~ gain * rms(act'(z)),
which is < 1 for saturating activations (e.g. tanh: 0.79). Over L
hops, gradients shrink as (gain*rms(act'))^L. The jacobian_gain
factor = 1/(gain*rms(act'(z))) normalizes this to ~1.0 per hop.
For identity/ReLU/LeakyReLU, jacobian_gain = 1.0 (exact preservation).

Additional scaling factors:
  - Self-gradient scaling (c_self per node): 1.0 (self-gradient is
    already O(1) when forward scaling maintains O(1) activations)
  - Weight gradient scaling (c_w per edge): 1.0 (optimizer handles
    magnitude; placeholder for future exploration)

Usage:
    from fabricpc.core.mupc import MuPCConfig

    structure = graph(
        nodes=[...], edges=[...], task_map=...,
        inference=InferenceSGD(...),
        scaling=MuPCConfig(include_output=True),
    )
"""

import math
import warnings
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass(frozen=True)
class MuPCScalingFactors:
    """
    Per-node scaling factors for muPC parameterization.

    All dictionaries are keyed by edge_key (e.g., "h1->h2:in").

    Attributes:
        forward_scale: Per-edge forward scaling factor (a_l).
            Applied to inputs before the node's forward() call.
        self_grad_scale: Scalar scaling for the node's self-gradient (dE/dz_self).
            Applied by the inference loop to the autodiff self-gradient.
        topdown_grad_scale: Per-edge scaling for the top-down gradient to
            presynaptic nodes. Equals a * jacobian_gain, combining chain rule
            correction (a) with Jacobian compensation (jacobian_gain) for
            deep gradient propagation. Applied after autodiff in
            forward_and_latent_grads().
        weight_grad_scale: Per-edge scaling for weight gradients,
            populated only for edges whose target slot is variance-scalable.
            Non-scalable slots (mask, skip, residual) are absent. Applied
            in scale_weight_grads() to weights whose key matches an
            edge_key; weights with non-edge keys (e.g., transformer
            internal projections) pass through at 1.0 and are expected to
            handle scaling internally.
    """

    forward_scale: Dict[str, float]
    self_grad_scale: float
    topdown_grad_scale: Dict[str, float]
    weight_grad_scale: Dict[str, float]

    # Edges arriving at non-variance-scalable slots are absent from all three
    # per-edge dicts (forward_scale, topdown_grad_scale, weight_grad_scale).
    # Callsites treat missing keys as identity pass-through (no multiplication),
    # which preserves input dtype — required for integer clamps that flow from
    # terminal source nodes into a downstream forward (e.g. token indices into
    # EmbeddingNode), where x * 1.0 would silently promote int → float.


@dataclass(frozen=True)
class MuPCConfig:
    """
    Configuration for muPC scaling computation.

    Pass this as the `scaling` argument to the graph() builder.

    Args:
        include_output: Whether to include output nodes (out_degree=0) in muPC
            scaling. When True, output nodes get a = gain/(fan_in * sqrt(K))
            (matching jpc reference a_L = 1/N for K=1 and gain=1; jpc carries
            no activation-gain factor). When False (default),
            output nodes are excluded and should use standard initialization
            (e.g., Xavier). Use True with MSE/Gaussian energy; False with
            softmax+CE.
        terminal_input_variance: Assumed variance of terminal input node
            outputs. Default 1.0. Not currently used in the simplified formula
            (all scaled nodes produce O(1) output), but reserved for future
            extensions with non-unit input variance.
    """

    include_output: bool = False
    terminal_input_variance: float = 1.0
    # Deprecated: kept for backward compatibility
    depth_metric: Optional[Any] = None
    min_depth: Optional[int] = None

    def __post_init__(self):
        if self.depth_metric is not None:
            warnings.warn(
                "MuPCConfig.depth_metric is deprecated and ignored. "
                "muPC scaling now uses a=gain/sqrt(fan_in*K*L) based on graph "
                "topology (in-degree K, depth L) instead of depth metrics.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.min_depth is not None:
            warnings.warn(
                "MuPCConfig.min_depth is deprecated and ignored. "
                "muPC scaling now uses a=gain/sqrt(fan_in*K*L) based on graph "
                "topology (in-degree K, depth L) instead of depth metrics.",
                DeprecationWarning,
                stacklevel=2,
            )


def _is_merge_node(node_info: Any) -> bool:
    """
    True if this node is a residual-stream merge: it has at least one
    is_skip_connection slot that receives at least one edge — the point
    where a computed branch joins the stream.

    Slots that are merely non-scalable (e.g., metadata like attention
    masks) do not qualify, and neither do declared-but-unconnected skip
    slots (e.g., a LinearResidual used without a skip edge).

    This single predicate drives both halves of the depth rule: which
    nodes count toward L (_count_skip_connections_depth) and which nodes'
    scalable edges carry the 1/sqrt(L) factor (compute_mupc_scalings).
    They must agree, or the damping lands on edges the depth count did
    not account for.
    """
    return any(
        s.is_skip_connection and len(s.in_neighbors) > 0
        for s in node_info.slots.values()
    )


def _count_skip_connections_depth(
    nodes: Dict[str, Any],
    edges: Dict[str, Any],
    node_order: List[str],
) -> int:
    """
    Count the number of merge nodes (see _is_merge_node) along the longest
    path in the graph. This is the "residual depth" — the number of
    variance-accumulating points where skip and compute paths sum.

    In a pure sequential chain (no skip connections), returns 0.
    In a ResNet with D residual blocks, returns D.

    The caller uses max(skip_depth, 1) as L in the scaling formula so that
    pure chains degenerate to a = gain/sqrt(fan_in * K) (no depth factor).

    node_order must be the unique node order (structure.node_order), not an
    unrolled visit schedule: depth models one merge-sum energy term per
    merge node regardless of visit count, and back edges contribute depth 0
    naturally (a later-ordered source reads skip_counts.get(source, 0)).
    """
    if len(set(node_order)) != len(node_order):
        raise ValueError(
            "node_order contains duplicate entries; pass the unique node "
            "order (structure.node_order), not the unrolled visit schedule."
        )
    skip_counts: Dict[str, int] = {}

    for node_name in node_order:
        node = nodes[node_name]
        node_info = node.node_info

        if node_info.in_degree == 0:
            skip_counts[node_name] = 0
            continue

        # Find max skip count among predecessors
        max_pred_skips = 0
        for edge_key in node_info.in_edges:
            edge_info = edges[edge_key]
            pred_skips = skip_counts.get(edge_info.source, 0)
            max_pred_skips = max(max_pred_skips, pred_skips)

        if _is_merge_node(node_info):
            skip_counts[node_name] = max_pred_skips + 1
        else:
            skip_counts[node_name] = max_pred_skips

    return max(skip_counts.values()) if skip_counts else 0


def compute_mupc_scalings(
    nodes: Dict[str, Any],
    edges: Dict[str, Any],
    config: MuPCConfig,
    node_order: Optional[List[str]] = None,
) -> Dict[str, MuPCScalingFactors]:
    """
    Compute per-node MuPCScalingFactors from graph topology.

    Uses edge-based scaling: each in-edge is scaled according to the
    target slot's is_variance_scalable flag. Non-scalable slots (e.g.,
    skip/residual connections) pass through at scale 1.0. Scalable slots
    get:

        a = gain / sqrt(v * K_slot)          (non-merge nodes)
        a = gain / sqrt(v * K_slot * L)      (merge nodes)

    where v is the target node's input-transform variance factor (from
    get_variance_factor()), K_slot is the in-degree of the *specific slot*
    (not the whole node), L is the residual depth, and
    gain = activation.variance_gain(). A merge node has at least one
    connected is_skip_connection slot; the 1/sqrt(L) damps each branch
    contribution once, where it joins the residual stream.

    For output nodes (include_output=True):

        a = gain / (v * sqrt(K_slot))

    with no depth factor: the readout is applied once to the O(1) stream,
    not summed L times. This matches a_L = 1/N in Table 1 of the muPC
    paper (arXiv:2505.13124), where N is the readout fan_in.

    Args:
        nodes: Dictionary mapping node names to finalized NodeBase instances
               (each has .node_info with shape, in_edges, out_edges, etc.)
        edges: Dictionary mapping edge keys to EdgeInfo objects
        config: MuPCConfig with scaling parameters.
        node_order: Optional topological ordering of node names. If provided,
                    nodes are processed in this order for determinism.

    Returns:
        Dictionary mapping node names to MuPCScalingFactors instances.
        Terminal input nodes (in_degree=0) get None.
        Terminal output nodes (out_degree=0) get None unless include_output=True.
    """
    # Detect output nodes (out_degree=0).
    output_nodes = {
        name for name, node in nodes.items() if node.node_info.out_degree == 0
    }

    # Use topological order if provided, otherwise iterate dict order
    iteration_order = node_order if node_order is not None else list(nodes.keys())
    if len(set(iteration_order)) != len(iteration_order):
        raise ValueError(
            "node_order contains duplicate entries; pass the unique node "
            "order (structure.node_order), not the unrolled visit schedule."
        )

    # Compute residual depth: number of nodes with connected
    # is_skip_connection slots along the longest path. For pure chains
    # this is 0, for ResNets it's the number of residual blocks. Using
    # max(depth, 1) ensures pure chains degenerate to the original
    # a = gain/sqrt(fan_in * K).
    skip_depth = _count_skip_connections_depth(nodes, edges, iteration_order)
    L = max(skip_depth, 1)

    scalings = {}

    for node_name in iteration_order:
        node = nodes[node_name]
        node_info = node.node_info
        node_class = node_info.node_class

        # Terminal input nodes (in_degree=0): no scaling needed.
        if node_info.in_degree == 0:
            scalings[node_name] = None
            continue

        # Output nodes: skip unless include_output is set.
        # With softmax+CE, excluding output avoids over-compressing logits.
        # With MSE/Gaussian energy, including output matches jpc reference.
        is_output = node_name in output_nodes
        if is_output and not config.include_output:
            scalings[node_name] = None
            continue

        node_config = node_info.node_config

        # Only edges into a merge node's scalable slots carry the depth
        # factor L — the same predicate that counted this node toward L.
        is_merge = _is_merge_node(node_info)

        # Activation-aware gain for variance preservation (Kaiming convention).
        activation = node_info.activation
        if activation is not None:
            gain = type(activation).variance_gain(activation.config)
            jac_gain = type(activation).jacobian_gain(activation.config)
        else:
            gain = 1.0
            jac_gain = 1.0

        forward_scale = {}
        topdown_grad_scale = {}
        weight_grad_scale = {}

        for edge_key in node_info.in_edges:
            edge_info = edges[edge_key]
            slot_name = edge_info.slot
            slot_info = node_info.slots[slot_name]

            if not slot_info.is_variance_scalable:
                # Non-scalable slot: identity pass-through. Edge is omitted
                # from all three per-edge dicts; scaling callsites treat a
                # missing key as a no-op (no multiplication) instead of
                # multiplying by 1.0. This preserves input dtype — necessary
                # for integer clamps flowing from terminal source nodes into
                # a downstream forward (e.g. token indices into EmbeddingNode),
                # where x * 1.0 would silently promote int → float.
                continue

            # Scalable slot: full muPC formula
            # K_slot = in-degree of this specific slot
            K_slot = len(slot_info.in_neighbors)

            source_node = nodes[edge_info.source]
            source_shape = source_node.node_info.shape

            # How this node's input transform scales variance. Weighted nodes
            # return their Kaiming fan_in; weightless sums (IdentityNode,
            # SkipConnection) return 1.0; reducing transforms return less than
            # 1.0 (AvgPool over n cells returns 1/n, so a amplifies by sqrt(n)).
            v = node_class.get_variance_factor(
                source_shape, node_config, node_info.weight_init
            )

            # Forward scaling formula with activation gain:
            #   Non-merge: a = gain/sqrt(v * K_slot)
            #       — K_slot handles multi-input variance amplification per slot
            #   Merge:     a = gain/sqrt(v * K_slot * L)
            #       — damps each branch contribution once, where it joins
            #         the residual stream; with unit stream variance the
            #         stream then grows as (1+1/L)^L ~ e over L merges
            #   Output:    a = gain/(v * sqrt(K_slot))
            #       — no depth factor: the readout is applied once to the
            #         O(1) stream, not summed L times (a_L = 1/N in Table 1
            #         of the muPC paper, arXiv:2505.13124)
            if is_output:
                a = gain / (v * math.sqrt(K_slot))
            elif is_merge:
                a = gain / math.sqrt(v * K_slot * L)
            else:
                a = gain / math.sqrt(v * K_slot)

            forward_scale[edge_key] = a

            # Top-down gradient scaling: c_td = a * jacobian_gain.
            #
            # Two components:
            # 1. Chain rule correction (a): scale_inputs() pre-scales
            #    inputs (x -> a*x) outside the value_and_grad closure. Autodiff
            #    yields dE/d(a*x); multiplying by a restores dE/dx.
            # 2. Jacobian compensation (jacobian_gain): the per-hop Jacobian
            #    diag(act'(z)) @ (a*W) has RMS singular value ~ gain*rms(act'(z)),
            #    which is < 1 for saturating activations. Multiplying by
            #    jacobian_gain = 1/(gain*rms(act'(z))) normalizes the expected
            #    per-hop gradient propagation factor to ~1.0.
            topdown_grad_scale[edge_key] = a * jac_gain

            # Weight gradient scaling: 1.0 (optimizer handles magnitude).
            weight_grad_scale[edge_key] = 1.0

        # Self-gradient scaling: 1.0
        # The self-gradient (dE/dz from the node's energy) is already O(1)
        # when the forward scaling maintains O(1) activations.
        self_grad_scale = 1.0

        scalings[node_name] = MuPCScalingFactors(
            forward_scale=forward_scale,
            self_grad_scale=self_grad_scale,
            topdown_grad_scale=topdown_grad_scale,
            weight_grad_scale=weight_grad_scale,
        )

    return scalings
