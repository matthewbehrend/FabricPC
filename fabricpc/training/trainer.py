"""Unified energy-framed trainer for FabricPC — PC and backprop.

One ``train``/``evaluate``/``make_train_step`` API serves both learning
algorithms, selected by ``algorithm`` (``"pc"`` or ``"backprop"``). Both
algorithms clamp identically during training (input AND target); they differ
in exactly three places inside the step:

    sub-step          PC                                   backprop
    ----------------  -----------------------------------  --------------------------------
    1 batch->dict     convert_batch                        (shared)
    2 build clamps    build_clamps(clamp_target=True)      (shared, identical)
    3 produce state   initialize_graph_state +             initialize_graph_state with
                      run_inference (settle)               FeedforwardStateInit (single
                                                           forward pass, no inference)
    4 objective       graph_energy over all in_degree>0    graph_energy over target nodes
                      nodes / batch                        / batch (same function,
                                                           different node subset)
    5 gradients       compute_local_weight_gradients       jax.value_and_grad over steps
                      (local, per node)                    3-4 (global)
    6 apply update    optax update + apply_updates         (shared)

The backprop objective is the energy of the clamped target nodes — the
negative log probability the output node's energy functional assigns to the
clamped target given the feedforward prediction. There is no ``loss_type``:
**the output node's energy functional in the graph definition selects the
loss** (``CrossEntropyEnergy`` -> cross-entropy, ``GaussianEnergy`` ->
``0.5 * precision * SSE``).

There is no ``autoregressive`` flag: the causal mask is derived from the
graph (a ``"causal_mask"`` entry in the ``TaskMap``, v1 transformer graphs)
or applied inside the node (``MhaResidualNode(is_causal=True)``, v2 graphs),
and one-hot conversion of integer/bool targets is derived from the target
dtype.

Multi-device data parallelism runs on jit + ``NamedSharding`` over an
optional ``mesh`` with axis ``"data"`` (``"model"`` is reserved for future
model parallelism): ``jax.make_mesh((jax.device_count(),), ("data",))``.

RNG contract: the training key affects only latent initialization
(``initialize_graph_state``); inference is deterministic and the package has
no dropout or noise. Keys derive as ``fold_in(rng_key, epoch_idx)`` ->
``fold_in(epoch_key, batch_idx)``, a pure function of
``(rng_key, epoch_idx, batch_idx)`` independent of loader length — so
``train(..., opt_state=ckpt, start_epoch=k)`` reproduces the uninterrupted
run's stream exactly.
"""

from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, cast

import math
import warnings

import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from tqdm.auto import tqdm as _tqdm_cls

from fabricpc.core.energy import graph_energy
from fabricpc.core.inference import run_inference
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.graph_initialization.state_initializer import (
    FeedforwardStateInit,
    initialize_graph_state,
)
from fabricpc.training.metrics import as_eval_metric, default_metrics

ALGORITHMS = ("pc", "backprop")

Algorithm = Literal["pc", "backprop"]


class TrainResult(NamedTuple):
    """Return value of :func:`train`.

    Attributes:
        params: Trained parameters.
        opt_state: Final optimizer state — pass it back via
            ``train(..., opt_state=...)`` to resume without resetting
            optimizer moments or an optax schedule's count.
        step: Optimizer updates applied in this call (0-based, monotonic).
        iter_results: 2D list ``[epoch][batch]`` of per-batch metric dicts
            (floats), or the ``iter_callback`` replacement values.
        epoch_results: List of per-epoch mean metric dicts (floats), or the
            ``epoch_callback`` replacement values.
    """

    params: GraphParams
    opt_state: optax.OptState
    step: int
    iter_results: list
    epoch_results: list


class EpochContext(NamedTuple):
    """Context passed to ``epoch_callback`` at the end of each epoch.

    Grows by field addition, never by positional breakage — read fields by
    name. ``metrics`` holds the epoch means of the per-batch training
    metrics. ``rng_key`` is the base training key; derive per-epoch keys via
    ``jax.random.fold_in(rng_key, epoch_idx)``.

    Note: the internal training step donates the params/opt_state buffers,
    so ``params``/``opt_state`` are valid during the callback but must be
    copied (``jax.tree_util.tree_map(jnp.copy, ...)``) if retained past it —
    the next training step invalidates them.
    """

    epoch_idx: int
    step: int
    params: GraphParams
    opt_state: optax.OptState
    structure: GraphStructure
    config: dict
    rng_key: jax.Array
    metrics: Dict[str, float]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def convert_batch(batch_data) -> Dict[str, jnp.ndarray]:
    """Normalize a loader batch (dict, or ``(x, y)`` tuple/list) to a dict of
    JAX arrays."""
    if isinstance(batch_data, (list, tuple)):
        return {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
    if isinstance(batch_data, dict):
        return {k: jnp.array(v) for k, v in batch_data.items()}
    raise ValueError(f"Unsupported batch format: {type(batch_data)}")


def create_causal_mask(seq_len: int) -> jnp.ndarray:
    """Lower-triangular causal mask of shape ``(seq_len, seq_len)``:
    ``1`` where ``j <= i``, so position ``i`` attends only to ``0..i``."""
    return jnp.tril(jnp.ones((seq_len, seq_len)))


def build_clamps(
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    *,
    clamp_target: bool,
) -> Dict[str, jnp.ndarray]:
    """Assemble the node clamps for one batch via ``structure.task_map``.

    1. Each batch key present in the task map clamps its mapped node.
       ``clamp_target=True`` (training, both algorithms) clamps every key;
       ``clamp_target=False`` (evaluation) clamps only non-target keys — a
       key is a target iff its mapped node has ``in_degree > 0``.
    2. A clamped target with non-floating dtype (int or bool class indices)
       is one-hot encoded with ``num_classes`` from the target node's
       ``shape[-1]`` — token loaders yield int32 targets to keep the
       host->device transfer small.
    3. If the task map declares a ``"causal_mask"`` node (v1 transformer
       graphs), a lower-triangular mask of shape ``(batch, 1, seq, seq)`` is
       injected, with ``seq`` from ``batch["x"].shape[1]``. Graphs without
       the entry (v2 masks internally via ``is_causal``) are untouched.
    """
    clamps: Dict[str, jnp.ndarray] = {}
    for task_name, task_value in batch.items():
        if task_name not in structure.task_map:
            continue
        node_name = structure.task_map[task_name]
        node_info = structure.nodes[node_name].node_info
        is_target = node_info.in_degree > 0
        if is_target and not clamp_target:
            continue
        value = jnp.asarray(task_value)
        if is_target and not jnp.issubdtype(value.dtype, jnp.floating):
            value = jax.nn.one_hot(value.astype(jnp.int32), node_info.shape[-1])
        clamps[node_name] = value
    if "causal_mask" in structure.task_map:
        batch_size, seq_len = batch["x"].shape[0], batch["x"].shape[1]
        mask = create_causal_mask(seq_len)[None, None, :, :]
        clamps[structure.task_map["causal_mask"]] = jnp.broadcast_to(
            mask, (batch_size, 1, seq_len, seq_len)
        )
    return clamps


def _validate_config(config: dict) -> None:
    """Fail fast on retired config keys with one-line migration text."""
    if "loss_type" in config:
        raise ValueError(
            "config['loss_type'] is retired: the output node's energy "
            "functional selects the loss — give the output node "
            "CrossEntropyEnergy() for cross-entropy or GaussianEnergy() for "
            "squared error."
        )
    if "use_causal_mask" in config:
        raise ValueError(
            "config['use_causal_mask'] is retired: the causal mask is derived "
            "from the graph — declare a 'causal_mask' node in the TaskMap "
            "(v1 transformer graphs) or use MhaResidualNode(is_causal=True) "
            "(v2 graphs)."
        )


def _validate_algorithm(algorithm: str, structure: GraphStructure) -> None:
    """Raise if ``algorithm`` is unknown or its graph prerequisite is missing:
    PC needs an inference algorithm, backprop a feedforward state
    initializer. Runs once at build time, not per batch."""
    if algorithm == "pc":
        if structure.config.get("inference") is None:
            raise ValueError(
                "algorithm='pc' requires an inference algorithm: build the "
                "graph with graph(..., inference=...)."
            )
    elif algorithm == "backprop":
        init = structure.config["graph_state_initializer"]
        if not isinstance(init, FeedforwardStateInit):
            raise ValueError(
                "algorithm='backprop' requires FeedforwardStateInit as the "
                f"graph_state_initializer, got {type(init).__name__}."
            )
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r}; choose from {ALGORITHMS}")


# ---------------------------------------------------------------------------
# The unified step
# ---------------------------------------------------------------------------


def _target_node_names(structure, clamps):
    """Clamped nodes with ``in_degree > 0`` — the backprop objective's node
    set and the ``target_energy`` metric's node set. Static at trace time."""
    return tuple(
        name
        for name in clamps
        if name in structure.nodes and structure.nodes[name].node_info.in_degree > 0
    )


def _batch_grads(params, batch, structure, rng_key, *, algorithm):
    """Gradients and metrics for one batch — the only algorithm branch."""
    batch_size = next(iter(batch.values())).shape[0]
    clamps = build_clamps(batch, structure, clamp_target=True)
    target_nodes = _target_node_names(structure, clamps)

    if algorithm == "pc":
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        state = run_inference(params, state, clamps, structure)
        energy = graph_energy(state, structure) / batch_size
        grads = compute_local_weight_gradients(params, state, structure)
    else:  # backprop
        if not target_nodes:
            raise ValueError(
                "algorithm='backprop' requires a clamped target: no batch key "
                "maps to an in_degree>0 node, so the objective is empty."
            )

        def objective(p):
            state = initialize_graph_state(
                structure, batch_size, rng_key, clamps=clamps, params=p
            )
            return (
                graph_energy(state, structure, node_names=target_nodes) / batch_size,
                state,
            )

        (energy, state), grads = jax.value_and_grad(objective, has_aux=True)(params)

    # Total prediction positions across target clamps: batch*seq for
    # sequences, batch for classification. Shapes are static at trace time.
    n_predictions = sum(math.prod(clamps[name].shape[:-1]) for name in target_nodes)
    if target_nodes:
        target_e = (
            graph_energy(state, structure, node_names=target_nodes) / n_predictions
        )
    else:
        target_e = jnp.zeros(())
    # Note the two keys use different normalizations: "energy" is per-sample
    # (/ batch), "target_energy" per-prediction (/ batch*seq for sequences).
    # "energy" is algorithm-dependent (all internal nodes for PC, target
    # nodes only for backprop) — cross-algorithm comparison of it is invalid.
    metrics = {"energy": energy, "target_energy": target_e}
    return grads, metrics, state


def _make_step(structure, optimizer, *, algorithm, with_state, donate):
    """Build the jitted per-batch step.

    ``with_state=True`` returns the final GraphState (public escape hatch);
    ``donate=True`` donates the params/opt_state buffers
    (``donate_argnums=(0, 1)``) — used by the internal loop, where
    ``final_state`` is dropped so donation removes a full extra
    params+opt_state copy from peak memory.
    """

    def step(params, opt_state, batch, rng_key):
        grads, metrics, state = _batch_grads(
            params, batch, structure, rng_key, algorithm=algorithm
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = cast(GraphParams, optax.apply_updates(params, updates))
        if with_state:
            return params, opt_state, metrics, state
        return params, opt_state, metrics

    return jax.jit(step, donate_argnums=(0, 1) if donate else ())


def make_train_step(
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    *,
    algorithm: Algorithm = "pc",
    mesh: Optional[Mesh] = None,
):
    """Build a jitted training step for custom loops.

    Returns ``step(params, opt_state, batch, rng_key) -> (params, opt_state,
    metrics, final_state)``. ``metrics`` is a dict of device scalars
    (``"energy"``: the per-sample objective; ``"target_energy"``: target-node
    energy per prediction). ``final_state`` is the settled (PC) or
    feedforward (backprop) GraphState — the escape hatch for dashboards.
    Inputs are NOT donated: callers may reuse the initial params.

    With ``mesh``, params/opt_state are placed replicated and each batch is
    sharded on its leading axis over the ``"data"`` mesh axis.
    """
    _validate_algorithm(algorithm, structure)
    jitted = _make_step(
        structure, optimizer, algorithm=algorithm, with_state=True, donate=False
    )
    if mesh is None:
        return jitted

    batch_sharding = NamedSharding(mesh, P("data"))
    replicated = NamedSharding(mesh, P())

    def step(params, opt_state, batch, rng_key):
        params = jax.device_put(params, replicated)
        opt_state = jax.device_put(opt_state, replicated)
        batch = {k: jax.device_put(v, batch_sharding) for k, v in batch.items()}
        return jitted(params, opt_state, batch, rng_key)

    return step


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    params: GraphParams,
    structure: GraphStructure,
    train_loader: Any,
    optimizer: optax.GradientTransformation,
    config: dict,
    rng_key: jax.Array,
    *,
    algorithm: Algorithm = "pc",
    opt_state: Optional[optax.OptState] = None,
    start_epoch: int = 0,
    mesh: Optional[Mesh] = None,
    verbose: bool = True,
    epoch_callback: Optional[Callable] = None,
    iter_callback: Optional[Callable] = None,
) -> TrainResult:
    """Train a FabricPC graph with PC or backprop.

    Args:
        params: Initial (or checkpointed) parameters.
        structure: Graph structure. Settling parameters (``infer_steps``,
            ``eta_infer``) live in the inference object inside
            ``structure.config``, not in ``config``.
        train_loader: Iterable of batches supporting ``len()``.
        optimizer: Optax optimizer.
        config: Reads only ``num_epochs`` (fractional supported: a partial
            epoch's ``epoch_results`` entry is the mean over the batches
            actually run). Otherwise an opaque pass-through to callbacks and
            experiment harnesses. Retired keys (``loss_type``,
            ``use_causal_mask``) raise ``ValueError``.
        rng_key: Base training key. It affects only latent initialization;
            inference is deterministic. Per-epoch keys are
            ``fold_in(rng_key, epoch_idx)``, per-batch keys
            ``fold_in(epoch_key, batch_idx)`` — independent of loader length.
        algorithm: ``"pc"`` (default) or ``"backprop"``.
        opt_state: Restored optimizer state for resume; ``None`` creates a
            fresh one via ``optimizer.init(params)``.
        start_epoch: Offset added to the epoch index, so a resumed run
            reproduces the uninterrupted run's RNG stream exactly.
        mesh: Optional ``jax.sharding.Mesh`` with axis ``"data"`` for
            data-parallel training. A batch whose size is not divisible by
            the data-axis size is skipped with a one-time warning.
        verbose: Show tqdm progress bars and epoch summaries. The tqdm
            postfix forces a per-batch device sync; ``verbose=False`` keeps
            metrics on device until each epoch boundary.
        epoch_callback: ``(ctx: EpochContext) -> Any``; a non-None return
            replaces that epoch's ``epoch_results`` entry. Exceptions
            propagate (tuner pruning depends on this).
        iter_callback: ``(epoch_idx, batch_idx, metrics: Dict[str, float])
            -> Any``; a non-None return replaces that batch's
            ``iter_results`` entry. Supplying it forces a per-batch device
            sync. Exceptions propagate.

    Returns:
        :class:`TrainResult` — pass ``result.opt_state`` and
        ``start_epoch=k`` back in to resume.
    """
    _validate_config(config)
    _validate_algorithm(algorithm, structure)

    if opt_state is None:
        opt_state = optimizer.init(params)

    # The internal step donates its params/opt_state buffers; copy once so
    # the caller's arrays stay valid after this call.
    params = jax.tree_util.tree_map(jnp.copy, params)
    opt_state = jax.tree_util.tree_map(jnp.copy, opt_state)

    data_axis_size = None
    batch_sharding = None
    if mesh is not None:
        batch_sharding = NamedSharding(mesh, P("data"))
        replicated = NamedSharding(mesh, P())
        params = jax.device_put(params, replicated)
        opt_state = jax.device_put(opt_state, replicated)
        data_axis_size = mesh.shape["data"]

    step_fn = _make_step(
        structure, optimizer, algorithm=algorithm, with_state=False, donate=True
    )

    num_epochs = config.get("num_epochs", 10)
    total_epochs = math.ceil(num_epochs)
    frac = num_epochs - math.floor(num_epochs)

    num_batches = len(train_loader)
    total_batches = sum(
        (
            round(frac * num_batches)
            if (e == total_epochs - 1 and frac > 0)
            else num_batches
        )
        for e in range(total_epochs)
    )
    progress = _tqdm_cls(total=total_batches, disable=not verbose, leave=True)
    sync_per_batch = verbose or iter_callback is not None
    shard_warned = False

    step = 0
    iter_results: List[Any] = []
    epoch_results: List[Any] = []
    for epoch_offset in range(total_epochs):
        epoch_idx = start_epoch + epoch_offset
        is_last_epoch = epoch_offset == total_epochs - 1
        max_batches = (
            round(frac * num_batches) if (is_last_epoch and frac > 0) else num_batches
        )
        progress.set_description(f"Epoch {epoch_offset + 1}/{total_epochs}")

        # Keys are a pure function of (rng_key, epoch_idx, batch_idx),
        # independent of loader length, so start_epoch=k reproduces the
        # uninterrupted run's stream exactly.
        epoch_key = jax.random.fold_in(rng_key, epoch_idx)

        batch_metrics: List[Any] = []
        epoch_sums: Optional[Dict[str, jnp.ndarray]] = None
        batches_run = 0
        for batch_idx, batch_data in enumerate(train_loader):
            if batch_idx >= max_batches:
                break
            batch = convert_batch(batch_data)
            if mesh is not None:
                bsz = next(iter(batch.values())).shape[0]
                if bsz % data_axis_size != 0:
                    if not shard_warned:
                        warnings.warn(
                            f"Skipping batch: size {bsz} is not divisible by "
                            f"the 'data' mesh axis size {data_axis_size}."
                        )
                        shard_warned = True
                    progress.update(1)
                    continue
                batch = {k: jax.device_put(v, batch_sharding) for k, v in batch.items()}
            batch_key = jax.random.fold_in(epoch_key, batch_idx)
            params, opt_state, metrics = step_fn(params, opt_state, batch, batch_key)
            step += 1
            batches_run += 1
            epoch_sums = (
                dict(metrics)
                if epoch_sums is None
                else {k: epoch_sums[k] + metrics[k] for k in epoch_sums}
            )

            if sync_per_batch:
                float_metrics = {k: float(v) for k, v in metrics.items()}
                if verbose:
                    progress.set_postfix(
                        energy=f"{float_metrics['energy']:.4f}",
                        epoch=f"{epoch_offset + 1}/{total_epochs}",
                    )
                stored: Any = float_metrics
                if iter_callback is not None:
                    replaced = iter_callback(epoch_idx, batch_idx, float_metrics)
                    if replaced is not None:
                        stored = replaced
                batch_metrics.append(stored)
            else:
                batch_metrics.append(metrics)
            progress.update(1)

        # Epoch boundary: materialize device scalars to floats.
        if not sync_per_batch:
            batch_metrics = [{k: float(v) for k, v in m.items()} for m in batch_metrics]
        iter_results.append(batch_metrics)

        epoch_means = (
            {k: float(v) / batches_run for k, v in epoch_sums.items()}
            if epoch_sums is not None
            else {}
        )
        entry: Any = epoch_means
        if epoch_callback is not None:
            ctx = EpochContext(
                epoch_idx=epoch_idx,
                step=step,
                params=params,
                opt_state=opt_state,
                structure=structure,
                config=config,
                rng_key=rng_key,
                metrics=epoch_means,
            )
            replaced = epoch_callback(ctx)
            if replaced is not None:
                entry = replaced
        epoch_results.append(entry)

        if verbose:
            summary = ", ".join(f"{k}: {v:.4f}" for k, v in epoch_means.items())
            _tqdm_cls.write(f"Epoch {epoch_offset + 1}/{total_epochs} — {summary}")

    progress.close()
    return TrainResult(
        params=params,
        opt_state=opt_state,
        step=step,
        iter_results=iter_results,
        epoch_results=epoch_results,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    params: GraphParams,
    structure: GraphStructure,
    test_loader: Any,
    config: dict,
    rng_key: jax.Array,
    *,
    algorithm: Algorithm = "pc",
    mesh: Optional[Mesh] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Evaluate a FabricPC graph: inputs clamped, targets free.

    PC settles the latents via ``run_inference``; backprop takes the single
    feedforward pass. Metrics are pluggable: ``metrics`` is a dict of named
    :class:`fabricpc.training.metrics.EvalMetric` instances (or bare
    callables ``(state, batch, structure) -> (value, weight)``, wrapped with
    the identity finalize). ``metrics=None`` selects the graph-derived
    defaults from :func:`fabricpc.training.metrics.default_metrics`:
    ``target_energy`` and ``accuracy`` always, ``cross_entropy``/
    ``perplexity`` when the target functional is ``CrossEntropyEnergy``,
    ``energy`` for PC. The defaults raise ``ValueError`` on a graph with no
    target task key; a caller-supplied dict carries no such requirement.

    Aggregation is weighted: each metric's per-sample ``(value, weight)``
    pairs accumulate across batches (and devices, under ``mesh``) and the
    result is ``finalize(Σvalue / Σweight)`` — so a ragged final batch never
    skews a mean, and ``perplexity`` is ``exp`` of the aggregated mean, not a
    mean of per-batch ``exp`` s. With ``mesh``, a ragged batch is zero-padded
    to the data-axis size and the padded samples get zero weight.
    """
    _validate_config(config)
    _validate_algorithm(algorithm, structure)

    if metrics is None:
        metric_map = default_metrics(structure, algorithm)
    else:
        metric_map = {name: as_eval_metric(m) for name, m in metrics.items()}
    metric_names = tuple(metric_map)

    def eval_step(p, batch, key, sample_mask):
        batch_size = next(iter(batch.values())).shape[0]
        clamps = build_clamps(batch, structure, clamp_target=False)
        state = initialize_graph_state(
            structure, batch_size, key, clamps=clamps, params=p
        )
        if algorithm == "pc":
            state = run_inference(p, state, clamps, structure)
        out = {}
        for name in metric_names:
            value, weight = metric_map[name].fn(state, batch, structure)
            out[name] = (
                jnp.sum(value * sample_mask),
                jnp.sum(weight * sample_mask),
            )
        return out

    jit_eval = jax.jit(eval_step)

    data_axis_size = None
    batch_sharding = None
    mask_sharding = None
    if mesh is not None:
        batch_sharding = NamedSharding(mesh, P("data"))
        mask_sharding = NamedSharding(mesh, P("data"))
        params = jax.device_put(params, NamedSharding(mesh, P()))
        data_axis_size = mesh.shape["data"]

    totals = {name: (jnp.zeros(()), jnp.zeros(())) for name in metric_names}
    for batch_idx, batch_data in enumerate(test_loader):
        batch = convert_batch(batch_data)
        bsz = next(iter(batch.values())).shape[0]
        sample_mask = jnp.ones((bsz,))
        if mesh is not None:
            if bsz % data_axis_size != 0:
                pad = data_axis_size - (bsz % data_axis_size)
                batch = {
                    k: jnp.concatenate(
                        [v, jnp.zeros((pad,) + v.shape[1:], dtype=v.dtype)]
                    )
                    for k, v in batch.items()
                }
                sample_mask = jnp.concatenate([sample_mask, jnp.zeros((pad,))])
            batch = {k: jax.device_put(v, batch_sharding) for k, v in batch.items()}
            sample_mask = jax.device_put(sample_mask, mask_sharding)
        key = jax.random.fold_in(rng_key, batch_idx)
        out = jit_eval(params, batch, key, sample_mask)
        totals = {
            name: (totals[name][0] + out[name][0], totals[name][1] + out[name][1])
            for name in metric_names
        }

    results: Dict[str, float] = {}
    for name in metric_names:
        total_value = float(totals[name][0])
        total_weight = float(totals[name][1])
        mean = total_value / total_weight if total_weight > 0 else float("nan")
        results[name] = float(metric_map[name].finalize(mean))
    return results
