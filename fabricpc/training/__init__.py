"""Training utilities for JAX predictive coding networks.

One energy-framed API trains FabricPC graphs with either learning rule,
selected by ``algorithm`` (``"pc"`` or ``"backprop"``). Backprop operates on
the same graph models, ensuring no divergence of model code; its objective is
the energy of the clamped target nodes, so the output node's energy
functional selects the loss. If there are cycles in the graph, don't expect
backprop to learn meaningful weights in those recurrency paths.
"""

from fabricpc.training import metrics
from fabricpc.training.generation import generate
from fabricpc.training.metrics import EvalMetric
from fabricpc.training.trainer import (
    EpochContext,
    TrainResult,
    batch_size_of,
    build_clamps,
    convert_batch,
    create_causal_mask,
    evaluate,
    grad_denominator,
    make_train_step,
    pc_weight_gradients,
    train,
)

__all__ = [
    "train",
    "evaluate",
    "make_train_step",
    "generate",
    "batch_size_of",
    "build_clamps",
    "convert_batch",
    "create_causal_mask",
    "grad_denominator",
    "pc_weight_gradients",
    "TrainResult",
    "EpochContext",
    "EvalMetric",
    "metrics",
]
