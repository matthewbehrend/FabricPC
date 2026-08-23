"""Pure functions for extracting metrics from FabricPC data structures.

These functions extract metrics from GraphState and GraphParams without
side effects. They are compatible with JAX's functional paradigm and can
be used safely with JIT-compiled code by calling them on the outputs.
"""

from typing import Dict, List, Optional
import jax.numpy as jnp
import numpy as np

from fabricpc.core.energy import graph_energy
from fabricpc.core.types import GraphState, GraphParams, GraphStructure


def extract_node_energies(state: GraphState) -> Dict[str, np.ndarray]:
    """Extract per-node energy values from GraphState.

    Args:
        state: GraphState after inference.

    Returns:
        Dictionary mapping node names to energy arrays (batch_size,).
    """
    return {
        node_name: np.asarray(node_state.energy)
        for node_name, node_state in state.nodes.items()
    }


def extract_total_energy(
    state: GraphState,
    structure: GraphStructure,
) -> float:
    """Extract total energy (sum over nodes with in_degree>0).

    Args:
        state: GraphState after inference.
        structure: GraphStructure.

    Returns:
        Total energy as float.
    """
    return float(graph_energy(state, structure))


def extract_latent_statistics(
    state: GraphState,
    nodes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Extract latent state statistics (mean, std, min, max) per node.

    Args:
        state: GraphState.
        nodes: Optional list of node names (default: all nodes).

    Returns:
        Dictionary: {node_name: {"mean": ..., "std": ..., "min": ..., "max": ...}}
    """
    nodes = nodes or list(state.nodes.keys())
    stats = {}
    for node_name in nodes:
        z = state.nodes[node_name].z_latent
        stats[node_name] = {
            "mean": float(jnp.mean(z)),
            "std": float(jnp.std(z)),
            "min": float(jnp.min(z)),
            "max": float(jnp.max(z)),
        }
    return stats


def extract_activation_statistics(
    state: GraphState,
    nodes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Extract activation (z_mu) statistics per node.

    Args:
        state: GraphState.
        nodes: Optional list of node names (default: all nodes).

    Returns:
        Dictionary: {node_name: {"mean": ..., "std": ..., "min": ..., "max": ...}}
    """
    nodes = nodes or list(state.nodes.keys())
    stats = {}
    for node_name in nodes:
        z_mu = state.nodes[node_name].z_mu
        stats[node_name] = {
            "mean": float(jnp.mean(z_mu)),
            "std": float(jnp.std(z_mu)),
            "min": float(jnp.min(z_mu)),
            "max": float(jnp.max(z_mu)),
        }
    return stats


def extract_weight_statistics(
    params: GraphParams,
    nodes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Extract weight matrix statistics per node per edge.

    Args:
        params: GraphParams.
        nodes: Optional list of node names (default: all nodes).

    Returns:
        Dictionary: {node_name: {edge_key: {"mean": ..., "std": ..., ...}}}
    """
    nodes = nodes or list(params.nodes.keys())
    stats = {}
    for node_name in nodes:
        node_params = params.nodes[node_name]
        edge_stats = {}
        for edge_key, weight in node_params.weights.items():
            edge_stats[edge_key] = {
                "mean": float(jnp.mean(weight)),
                "std": float(jnp.std(weight)),
                "min": float(jnp.min(weight)),
                "max": float(jnp.max(weight)),
                "norm": float(jnp.linalg.norm(weight)),
            }
        stats[node_name] = edge_stats
    return stats


def extract_bias_statistics(
    params: GraphParams,
    nodes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Extract bias statistics per node.

    Args:
        params: GraphParams.
        nodes: Optional list of node names (default: all nodes).

    Returns:
        Dictionary: {node_name: {bias_key: {"mean": ..., "std": ..., ...}}}
    """
    nodes = nodes or list(params.nodes.keys())
    stats = {}
    for node_name in nodes:
        node_params = params.nodes[node_name]
        bias_stats = {}
        for bias_key, bias in node_params.biases.items():
            bias_stats[bias_key] = {
                "mean": float(jnp.mean(bias)),
                "std": float(jnp.std(bias)),
                "min": float(jnp.min(bias)),
                "max": float(jnp.max(bias)),
                "norm": float(jnp.linalg.norm(bias)),
            }
        stats[node_name] = bias_stats
    return stats


def extract_error_statistics(
    state: GraphState,
    nodes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Extract prediction error statistics per node.

    Args:
        state: GraphState.
        nodes: Optional list of node names (default: all nodes).

    Returns:
        Dictionary: {node_name: {"mean_abs": ..., "std": ..., "max_abs": ...}}
    """
    nodes = nodes or list(state.nodes.keys())
    stats = {}
    for node_name in nodes:
        error = state.nodes[node_name].error
        stats[node_name] = {
            "mean_abs": float(jnp.mean(jnp.abs(error))),
            "std": float(jnp.std(error)),
            "max_abs": float(jnp.max(jnp.abs(error))),
        }
    return stats


def extract_latent_grad_statistics(
    state: GraphState,
    nodes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Extract latent gradient statistics per node.

    This is useful for understanding inference convergence.

    Args:
        state: GraphState.
        nodes: Optional list of node names (default: all nodes).

    Returns:
        Dictionary: {node_name: {"mean_abs": ..., "norm": ..., "max_abs": ...}}
    """
    nodes = nodes or list(state.nodes.keys())
    stats = {}
    for node_name in nodes:
        grad = state.nodes[node_name].latent_grad
        stats[node_name] = {
            "mean_abs": float(jnp.mean(jnp.abs(grad))),
            "norm": float(jnp.linalg.norm(grad)),
            "max_abs": float(jnp.max(jnp.abs(grad))),
        }
    return stats


def flatten_for_distribution(arr: jnp.ndarray) -> np.ndarray:
    """Flatten array to 1D float64 numpy with non-finite values removed,
    for Aim Distribution tracking.

    Aim's Distribution histograms the samples with np.histogram, which
    raises ValueError when the data contains inf/NaN or when max - min
    overflows the sample dtype (float32 weights in a diverging run reach
    ~1e38, so their range exceeds float32 max). A tracker must record
    pathological training states, not crash the run on them: cast to
    float64 so any finite float32 range is binnable, and drop non-finite
    entries. Callers skip distribution tracking when nothing finite
    remains.

    Args:
        arr: JAX array of any shape.

    Returns:
        1D float64 numpy array of the finite elements (may be empty).
    """
    flat = np.asarray(arr, dtype=np.float64).ravel()
    return flat[np.isfinite(flat)]


def extract_all_distributions(
    state: GraphState,
    params: GraphParams,
    nodes: Optional[List[str]] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Extract all distributions for tracking.

    Args:
        state: GraphState after inference.
        params: GraphParams.
        nodes: Optional list of node names.

    Returns:
        Dictionary organized by category:
        {
            "z_latent": {node_name: flattened_array, ...},
            "z_mu": {...},
            "error": {...},
            "weights": {f"{node}_{edge}": flattened_array, ...},
            "biases": {...},
        }
    """
    nodes = nodes or list(state.nodes.keys())

    distributions: Dict[str, Dict[str, np.ndarray]] = {
        "z_latent": {},
        "z_mu": {},
        "error": {},
        "weights": {},
        "biases": {},
    }

    for node_name in nodes:
        node_state = state.nodes[node_name]
        distributions["z_latent"][node_name] = flatten_for_distribution(
            node_state.z_latent
        )
        distributions["z_mu"][node_name] = flatten_for_distribution(node_state.z_mu)
        distributions["error"][node_name] = flatten_for_distribution(node_state.error)

    for node_name in nodes:
        if node_name in params.nodes:
            node_params = params.nodes[node_name]
            for edge_key, weight in node_params.weights.items():
                key = f"{node_name}/{edge_key}"
                distributions["weights"][key] = flatten_for_distribution(weight)
            for bias_key, bias in node_params.biases.items():
                key = f"{node_name}/{bias_key}"
                distributions["biases"][key] = flatten_for_distribution(bias)

    return distributions
