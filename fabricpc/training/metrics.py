"""Pluggable evaluation metrics for :func:`fabricpc.training.evaluate`.

A metric separates three concerns:

- **Per-batch computation** — ``EvalMetric.fn(state, batch, structure) ->
  (value, weight)``, two per-sample arrays of shape ``(batch,)``, run inside
  the jitted eval step on the settled state. The weight is the metric's own
  denominator (e.g. predictions per sample), so per-sample and per-token
  metrics aggregate correctly across ragged batches.
- **Aggregation** — owned by ``evaluate``: padded samples get zero weight,
  Σvalue and Σweight accumulate across batches and devices, and the result
  is ``finalize(Σvalue / Σweight)``.
- **Post-aggregation transform** — ``EvalMetric.finalize``, applied once to
  the global mean. ``perplexity`` is ``exp`` of the aggregated mean
  cross-entropy, not a mean of per-batch ``exp``\\ s.

``default_metrics(structure, algorithm)`` derives the default metric set
from the graph: the target node's own energy functional selects the loss and
therefore the default eval metrics.
"""

from typing import Any, Callable, Dict, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.types import GraphState, GraphStructure


def _identity(x):
    return x


class EvalMetric(NamedTuple):
    """One evaluation metric: a per-batch function plus a final transform.

    ``fn(state, batch, structure)`` returns ``(value, weight)``, both
    per-sample arrays of shape ``(batch,)``. ``finalize`` is applied once to
    the globally aggregated mean ``Σvalue / Σweight``.
    """

    fn: Callable[[GraphState, Dict[str, jnp.ndarray], GraphStructure], Tuple]
    finalize: Callable = _identity


def as_eval_metric(metric: Any) -> EvalMetric:
    """Wrap a bare callable ``(state, batch, structure) -> (value, weight)``
    into an :class:`EvalMetric` with the identity finalize."""
    if isinstance(metric, EvalMetric):
        return metric
    if callable(metric):
        return EvalMetric(fn=metric)
    raise TypeError(
        f"Metrics must be EvalMetric instances or callables; got {type(metric)!r}"
    )


# ---------------------------------------------------------------------------
# Target selection helpers
# ---------------------------------------------------------------------------


def _target_items(
    structure: GraphStructure, batch: Dict[str, jnp.ndarray]
) -> List[Tuple[str, str]]:
    """``(task_key, node_name)`` pairs for batch keys mapped to target nodes.

    A task key is a target iff its mapped node has ``in_degree > 0`` (source
    nodes carry inputs, not predictions).
    """
    items = [
        (key, structure.task_map[key])
        for key in batch
        if key in structure.task_map
        and structure.nodes[structure.task_map[key]].node_info.in_degree > 0
    ]
    if not items:
        raise ValueError(
            "No target task key: no batch key maps to an in_degree>0 node, so "
            "target-based metrics (target_energy, accuracy, cross_entropy, "
            "perplexity) are undefined. Pass an explicit metrics dict to "
            "evaluate(...) for graphs without a target."
        )
    return items


def _as_one_hot(y: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    """One-hot a non-float target (int or bool class indices); float targets
    pass through unchanged (already one-hot or continuous)."""
    y = jnp.asarray(y)
    if jnp.issubdtype(y.dtype, jnp.floating):
        return y
    return jax.nn.one_hot(y.astype(jnp.int32), num_classes)


def _predictions_per_sample(shaped: jnp.ndarray) -> float:
    """Prediction positions per sample: the product of the non-batch,
    non-class axes (seq_len for sequences, 1 for classification)."""
    n = 1
    for d in shaped.shape[1:-1]:
        n *= d
    return float(n)


def _sum_per_sample(values: jnp.ndarray) -> jnp.ndarray:
    """Sum all non-batch axes down to a per-sample ``(batch,)`` array."""
    axes = tuple(range(1, values.ndim))
    return jnp.sum(values, axis=axes) if axes else values


# ---------------------------------------------------------------------------
# Built-in metric functions
# ---------------------------------------------------------------------------


def _target_energy_fn(state, batch, structure):
    """Per-sample ``E(y, z_mu)`` under each target node's own energy
    functional; weight = prediction positions per sample."""
    value = None
    weight = None
    for key, node_name in _target_items(structure, batch):
        node_info = structure.nodes[node_name].node_info
        y = _as_one_hot(batch[key], node_info.shape[-1])
        z_mu = state.nodes[node_name].z_mu
        energy_obj = node_info.energy
        e = type(energy_obj).energy(y, z_mu, energy_obj.config)
        w = jnp.full(e.shape, _predictions_per_sample(y))
        value = e if value is None else value + e
        weight = w if weight is None else weight + w
    return value, weight


def _cross_entropy_fn(state, batch, structure):
    """Per-sample categorical cross-entropy of ``z_mu`` against the one-hot
    target; weight = prediction positions per sample.

    Uses the target node's own ``CrossEntropyEnergy`` when it has one (so its
    eps matches and this is identical to ``target_energy``), else a default
    ``CrossEntropyEnergy()``.
    """
    value = None
    weight = None
    for key, node_name in _target_items(structure, batch):
        node_info = structure.nodes[node_name].node_info
        y = _as_one_hot(batch[key], node_info.shape[-1])
        z_mu = state.nodes[node_name].z_mu
        energy_obj = node_info.energy
        if not isinstance(energy_obj, CrossEntropyEnergy):
            energy_obj = _DEFAULT_CE
        e = type(energy_obj).energy(y, z_mu, energy_obj.config)
        w = jnp.full(e.shape, _predictions_per_sample(y))
        value = e if value is None else value + e
        weight = w if weight is None else weight + w
    return value, weight


_DEFAULT_CE = CrossEntropyEnergy()


def _accuracy_fn(state, batch, structure):
    """Per-sample correct predictions via ``argmax(z_mu, -1)``; weight =
    prediction positions per sample. Targets with the same rank as ``z_mu``
    are argmaxed (one-hot); lower-rank targets are compared as class indices.
    Argmax-based: meaningful for class-like targets, not continuous ones."""
    value = None
    weight = None
    for key, node_name in _target_items(structure, batch):
        z_mu = state.nodes[node_name].z_mu
        y = jnp.asarray(batch[key])
        pred_labels = jnp.argmax(z_mu, axis=-1)
        if y.ndim == z_mu.ndim:
            true_labels = jnp.argmax(y, axis=-1)
        else:
            true_labels = y
        correct = _sum_per_sample((pred_labels == true_labels).astype(jnp.float32))
        w = jnp.full(correct.shape, float(pred_labels.size // pred_labels.shape[0]))
        value = correct if value is None else value + correct
        weight = w if weight is None else weight + w
    return value, weight


def _internal_energy_fn(state, batch, structure):
    """Per-sample energy summed over internal (``in_degree > 0``) nodes — the
    same node set as the PC training objective; weight = 1 per sample.

    Summation order matches :func:`fabricpc.core.energy.graph_energy`:
    ``structure.node_order`` first, then nodes the topological sort omitted
    (cycle members) in ``structure.nodes`` insertion order.
    """
    internal = {
        name for name, node in structure.nodes.items() if node.node_info.in_degree > 0
    }
    ordered = [n for n in structure.node_order if n in internal]
    ordered += [
        n for n in structure.nodes if n in internal and n not in structure.node_order
    ]
    value = jnp.zeros((state.batch_size,))
    for name in ordered:
        value = value + state.nodes[name].energy
    return value, jnp.ones_like(value)


# ---------------------------------------------------------------------------
# Built-in metrics and graph-derived defaults
# ---------------------------------------------------------------------------

target_energy = EvalMetric(fn=_target_energy_fn)
cross_entropy = EvalMetric(fn=_cross_entropy_fn)
perplexity = EvalMetric(fn=_cross_entropy_fn, finalize=jnp.exp)
accuracy = EvalMetric(fn=_accuracy_fn)
internal_energy = EvalMetric(fn=_internal_energy_fn)


def default_metrics(structure: GraphStructure, algorithm: str) -> Dict[str, EvalMetric]:
    """Graph-derived default metric set for :func:`fabricpc.training.evaluate`.

    ``target_energy`` and ``accuracy`` always; ``cross_entropy`` and
    ``perplexity`` when a target node's energy functional is
    ``CrossEntropyEnergy``; ``energy`` (per-sample internal energy) for PC.
    Raises ``ValueError`` on a graph with no target task key — pass an
    explicit metrics dict to evaluate such a graph.
    """
    target_nodes = [
        node_name
        for node_name in structure.task_map.values()
        if structure.nodes[node_name].node_info.in_degree > 0
    ]
    if not target_nodes:
        raise ValueError(
            "default_metrics requires a target task key (a task_map entry "
            "mapping to an in_degree>0 node); this graph has none. Pass an "
            "explicit metrics dict to evaluate(...)."
        )
    metrics: Dict[str, EvalMetric] = {
        "target_energy": target_energy,
        "accuracy": accuracy,
    }
    if any(
        isinstance(structure.nodes[n].node_info.energy, CrossEntropyEnergy)
        for n in target_nodes
    ):
        metrics["cross_entropy"] = cross_entropy
        metrics["perplexity"] = perplexity
    if algorithm == "pc":
        metrics["energy"] = internal_energy
    return metrics
