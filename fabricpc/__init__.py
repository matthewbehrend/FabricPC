"""
FabricPC-JAX: Predictive Coding Networks in JAX
================================================

A functional, high-performance implementation of predictive coding networks
using JAX for automatic differentiation, JIT compilation, and multi-device parallelism.

Key Features:
- Functional programming paradigm (immutable data structures)
- JIT-compiled inference and training loops
- Multi-GPU/TPU data parallelism with jit + NamedSharding meshes
- XLA optimization for maximum performance

Example:
    >>> from fabricpc.nodes import Linear
    >>> from fabricpc.core.topology import Edge
    >>> from fabricpc.graph_assembly import TaskMap, graph
    >>> from fabricpc.graph_initialization import initialize_params
    >>> from fabricpc import train, evaluate
    >>>
    >>> # Define nodes
    >>> input_node = Linear(shape=(784,), name="input")
    >>> hidden = Linear(shape=(128,), name="hidden")
    >>> output = Linear(shape=(10,), name="output")
    >>>
    >>> # Build graph
    >>> structure = graph(
    ...     nodes=[input_node, hidden, output],
    ...     edges=[
    ...         Edge(source=input_node, target=hidden.slot("in")),
    ...         Edge(source=hidden, target=output.slot("in")),
    ...     ],
    ...     task_map=TaskMap(x=input_node, y=output),
    ...     inference=InferenceSGD(eta_infer=0.05, infer_steps=10),
    ... )
    >>> params = initialize_params(structure, rng_key)
    >>> result = train(params, structure, train_loader, optimizer, config, rng_key)
    >>> metrics = evaluate(result.params, structure, test_loader, config, rng_key)
"""

from importlib.metadata import version

__version__ = version("fabricpc")

# Submodules (for advanced use)
from fabricpc import (
    core,
    graph_initialization,
    nodes,
    training,
    utils,
    graph_assembly,
    models,
    experiments,
)

# Core API - what most users need
from fabricpc.graph_initialization import initialize_params
from fabricpc.training import train, evaluate
from fabricpc.jax_config import setup_jax

# Types - for type hints
from fabricpc.core.types import GraphParams, GraphState, GraphStructure

__all__ = [
    # Core API (common use)
    "initialize_params",
    "train",
    "evaluate",
    "setup_jax",
    # Types (for type hints)
    "GraphParams",
    "GraphState",
    "GraphStructure",
    # Submodules (advanced use)
    "core",
    "graph_assembly",
    "graph_initialization",
    "models",
    "nodes",
    "training",
    "utils",
    "experiments",
]
