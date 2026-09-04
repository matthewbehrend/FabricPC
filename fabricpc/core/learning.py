from fabricpc.core.types import GraphParams, GraphState, GraphStructure, NodeParams
from fabricpc.core.inference import gather_inputs
from fabricpc.core.scaling import scale_inputs, scale_weight_grads


def compute_local_weight_gradients(
    params: GraphParams,
    final_state: GraphState,
    structure: GraphStructure,
) -> GraphParams:
    """
    Compute local weight gradients for each node using its own error signal.

    This implements the local Hebbian learning rule for predictive coding.
    muPC scaling is applied here (pre-scale inputs, post-scale weight
    gradients), keeping node methods (forward_and_weight_grads) scaling-unaware.

    The returned gradients are batch-summed: each node's
    ``forward_and_weight_grads`` differentiates the sum of its per-sample
    energies over the batch. Sums are associative, so they survive data
    sharding and gradient accumulation unchanged. The trainer's
    ``fabricpc.training.pc_weight_gradients`` divides them once by the
    prediction count ``fabricpc.training.grad_denominator``, whose docstring
    carries the rule for microbatching and padded positions. Optimizer-facing
    callers use that function; this one serves inspection and composition.

    Args:
        params: Current model parameters
        final_state: Converged state after inference
        structure: Graph structure

    Returns:
        GraphParams containing batch-summed gradients for the parameters
    """
    gradients = {}

    for node_name, node in structure.nodes.items():
        node_info = node.node_info
        # Source nodes have no weights, but need empty gradient dict for consistency
        if node_info.in_degree == 0:
            gradients[node_name] = NodeParams(weights={}, biases={})
            continue

        in_edges_data = gather_inputs(node_info, structure, final_state)

        node_class = node_info.node_class
        sc = node_info.scaling_config

        # Pre-scale inputs by muPC forward scaling factors
        scaled_inputs = scale_inputs(in_edges_data, sc)

        # Compute local gradients using node's method (pure autodiff)
        node_state, grad_params = node_class.forward_and_weight_grads(
            params.nodes[node_name],
            scaled_inputs,
            final_state.nodes[node_name],
            node_info,
        )

        # Post-scale weight gradients by muPC factors
        grad_params = scale_weight_grads(grad_params, sc)

        # Store gradients
        gradients[node_name] = grad_params

    # convert to GraphParams
    params_gradients = GraphParams(nodes=gradients)

    return params_gradients
