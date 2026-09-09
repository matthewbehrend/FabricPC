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
                      nodes / N                            / N (same function,
                                                           different node subset)
    5 gradients       pc_weight_gradients (local, per      jax.value_and_grad over steps
                      node, summed gradients / N)          3-4 (global)
    6 apply update    optax update + apply_updates         (shared)

N is the prediction count from :func:`grad_denominator`: the total number of
clamped-target prediction positions in the batch (``batch`` for
classification, ``batch * seq`` for token targets). Both algorithms hand
optax mean gradients per prediction, so one learning rate, clipping
threshold, or Adam epsilon means the same under either algorithm and across
batch sizes and sequence lengths.

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
from fabricpc.core.types import GraphParams, GraphState, GraphStructure
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
        step: Count of optimizer updates applied in this call. Not offset
            by ``start_epoch``: a resumed run counts from 0 again, so a
            caller logging a global step adds its own checkpoint offset.
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

    Grows by field addition at the end, never by positional breakage — read
    fields by name. ``metrics`` holds the epoch means of the per-batch
    training metrics. ``rng_key`` is the base training key and ``epoch_key``
    is ``fold_in(rng_key, epoch_idx)``, the key this epoch's batch keys
    derive from. ``algorithm`` is the ``algorithm`` passed to :func:`train`.

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
    algorithm: Algorithm
    epoch_key: jax.Array


class IterContext(NamedTuple):
    """Context passed to ``iter_callback`` after each batch's update.

    A superset of :class:`EpochContext`: its fields first, in its order, then
    the per-batch ones. Grows by field addition at the end, never by
    positional breakage — read fields by name.

    ``step`` counts optimizer updates applied in this :func:`train` call,
    this batch included. ``state`` is the GraphState the step produced for
    this batch: the settled latents under PC, the feedforward pass under
    backprop. ``batch`` is the converted batch dict fed to the step
    (task-mapped keys; under ``mesh`` its arrays carry the ``P("data")``
    sharding) and ``batch_key`` (``fold_in(epoch_key, batch_idx)``) is the
    key the step used for latent initialization. ``metrics`` holds this
    batch's float metrics.

    Buffer lifetimes: ``params``/``opt_state`` are donated by the next step,
    so copy them if retained past the callback (the ``EpochContext`` caveat).
    ``state`` and ``batch`` are not donated; the trainer drops its own
    reference to ``state`` as soon as the callback returns, so retaining it
    keeps exactly that one GraphState alive and not retaining it frees it.
    """

    epoch_idx: int
    step: int
    params: GraphParams
    opt_state: optax.OptState
    structure: GraphStructure
    config: dict
    rng_key: jax.Array
    metrics: Dict[str, float]
    algorithm: Algorithm
    epoch_key: jax.Array
    batch_idx: int
    state: GraphState
    batch_key: jax.Array
    batch: Dict[str, jnp.ndarray]


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


def batch_size_of(batch: Dict[str, jnp.ndarray], structure: GraphStructure) -> int:
    """Leading-axis size of the first batch key the ``task_map`` names.

    Reading an arbitrary batch value would trust extra keys whose leading
    axis is not the batch; only task-mapped keys are guaranteed to be
    ``(batch, ...)`` arrays.
    """
    for key, value in batch.items():
        if key in structure.task_map:
            return jnp.asarray(value).shape[0]
    raise ValueError(
        f"No batch key maps to a task: batch keys {sorted(batch)}, "
        f"task keys {sorted(structure.task_map)}."
    )


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
       host->device transfer small. Target clamps must then match the node
       shape (``(batch, *node.shape)``); a mismatch raises ``ValueError``
       here instead of an opaque XLA broadcast error inside the step.
    3. If the task map declares a ``"causal_mask"`` node (v1 transformer
       graphs), a lower-triangular mask of shape ``(batch, 1, seq, seq)`` is
       injected, with ``seq`` from the mask node's declared shape. Graphs
       without the entry (v2 masks internally via ``is_causal``) are
       untouched.
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
        if is_target:
            if not jnp.issubdtype(value.dtype, jnp.floating):
                value = jax.nn.one_hot(value.astype(jnp.int32), node_info.shape[-1])
            if value.shape[1:] != tuple(node_info.shape):
                raise ValueError(
                    f"target '{task_name}' has shape {value.shape} (after "
                    f"one-hot), but node '{node_name}' expects "
                    f"(batch, {', '.join(str(d) for d in node_info.shape)}). "
                    f"Integer/bool targets are one-hot encoded to the node's "
                    f"trailing class axis; float targets must already match "
                    f"the node shape."
                )
        clamps[node_name] = value
    if "causal_mask" in structure.task_map:
        mask_node = structure.task_map["causal_mask"]
        # The mask node declares (..., seq, seq); read seq from the graph, not
        # from a hard-coded batch key.
        seq_len = structure.nodes[mask_node].node_info.shape[-1]
        batch_size = batch_size_of(batch, structure)
        mask = create_causal_mask(seq_len)[None, None, :, :]
        clamps[mask_node] = jnp.broadcast_to(mask, (batch_size, 1, seq_len, seq_len))
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


def _data_axis_size(mesh: Mesh) -> int:
    """Size of the mesh's ``"data"`` axis, with an actionable error when the
    mesh names no such axis (``mesh.shape["data"]`` alone raises a bare
    ``KeyError``)."""
    if "data" not in mesh.shape:
        raise ValueError(
            f"mesh must name a 'data' axis for data parallelism, got axes "
            f"{tuple(mesh.shape)}; build it with "
            f"jax.make_mesh((jax.device_count(),), ('data',))."
        )
    return mesh.shape["data"]


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


def grad_denominator(structure: GraphStructure, clamps: Dict[str, jnp.ndarray]) -> int:
    """Prediction count N: the single denominator that turns batch-summed
    energies and weight gradients into means per prediction.

    N is the total number of clamped-target prediction positions in the
    batch, ``sum(prod(clamps[name].shape[:-1]))`` over the target nodes (the
    clamped nodes with ``in_degree > 0``). The trailing axis of a target
    clamp is the class axis: ``build_clamps`` validates every target clamp
    against ``(batch, *node.shape)``, so rank >= 2 holds. A rank-2
    classification target ``(B, C)`` gives N = B; a rank-3 token target
    ``(B, S, V)`` gives N = B * S. With several target heads N is the sum of
    their positions (two same-shape heads give N = 2 * B), so adding a head
    halves the step every shared parameter takes at a fixed learning rate.
    With no clamped target (associative-memory graphs) N is the batch size,
    read from the leading axis of the first clamp. Empty ``clamps`` raises
    ``ValueError``: nothing is clamped, so there is no objective.

    N is the per-batch total of the per-sample weight that
    ``metrics._internal_energy_fn`` assigns in ``evaluate``, so the train and
    eval ``energy`` share one scale. A clamped input node with feedback edges
    has ``in_degree > 0`` and counts as a target; no graph in the repository
    does this.

    Gradients arrive batch-summed from ``compute_local_weight_gradients``
    and are divided once by this count in ``pc_weight_gradients``.
    Accumulation over microbatches or a padding mask must keep that shape:
    sum first, divide once by the window's total count.
    """
    if not clamps:
        raise ValueError(
            "grad_denominator: clamps is empty, so nothing is clamped and there "
            "is no objective to normalize."
        )
    target_nodes = _target_node_names(structure, clamps)
    if target_nodes:
        return sum(math.prod(clamps[name].shape[:-1]) for name in target_nodes)
    return next(iter(clamps.values())).shape[0]


def pc_weight_gradients(
    params: GraphParams,
    state: GraphState,
    structure: GraphStructure,
    clamps: Dict[str, jnp.ndarray],
) -> GraphParams:
    """Local PC weight gradients as means per prediction: the batch-summed
    gradients from ``compute_local_weight_gradients`` divided once by
    ``grad_denominator(structure, clamps)``.

    This is the PC path's optimizer-facing gradient. Custom loops call it in
    place of ``compute_local_weight_gradients`` so that their learning rate,
    clipping threshold, and Adam epsilon mean the same as in :func:`train`.
    """
    denom = grad_denominator(structure, clamps)
    grads = compute_local_weight_gradients(params, state, structure)
    return jax.tree_util.tree_map(lambda g: g / denom, grads)


def _batch_grads(params, batch, structure, rng_key, *, algorithm):
    """Gradients and metrics for one batch — the only algorithm branch."""
    batch_size = batch_size_of(batch, structure)
    clamps = build_clamps(batch, structure, clamp_target=True)
    target_nodes = _target_node_names(structure, clamps)
    # Static at trace time; global under jit + NamedSharding.
    denom = grad_denominator(structure, clamps)

    if algorithm == "pc":
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        state = run_inference(params, state, clamps, structure)
        energy = graph_energy(state, structure) / denom
        grads = pc_weight_gradients(params, state, structure, clamps)
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
                graph_energy(state, structure, node_names=target_nodes) / denom,
                state,
            )

        (energy, state), grads = jax.value_and_grad(objective, has_aux=True)(params)

    if target_nodes:
        target_e = graph_energy(state, structure, node_names=target_nodes) / denom
    else:
        target_e = jnp.zeros(())
    # Both keys are means per prediction over the same denominator. "energy"
    # is the optimized objective; its node set is algorithm-dependent (all
    # internal nodes for PC, target nodes only for backprop), so comparing it
    # across algorithms compares different node sets. "target_energy" is the
    # same quantity under both algorithms.
    metrics = {"energy": energy, "target_energy": target_e}
    return grads, metrics, state


def _make_step(structure, optimizer, *, algorithm, with_state, donate):
    """Build the jitted per-batch step.

    ``with_state=True`` returns the final GraphState: :func:`make_train_step`
    always does, and :func:`train` does when an ``iter_callback`` is supplied
    so the callback's :class:`IterContext` can carry it. ``donate=True``
    donates the params/opt_state buffers (``donate_argnums=(0, 1)``) — used
    by the internal loop, where donation removes a full extra
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
    (``"energy"``: the objective per prediction; ``"target_energy"``:
    target-node energy per prediction). ``final_state`` is the settled (PC) or
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

    _data_axis_size(mesh)
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
    epoch_callback: Optional[Callable[[EpochContext], Any]] = None,
    iter_callback: Optional[Callable[[IterContext], Any]] = None,
) -> TrainResult:
    """Train a FabricPC graph with PC or backprop.

    Args:
        params: Initial (or checkpointed) parameters.
        structure: Graph structure. Settling parameters (``infer_steps``,
            ``eta_infer``) live in the inference object inside
            ``structure.config``, not in ``config``.
        train_loader: Iterable of batches supporting ``len()``.
        optimizer: Optax optimizer.
        config: Must contain ``num_epochs`` (fractional supported: a partial
            epoch's ``epoch_results`` entry is the mean over the batches
            actually run; a fractional tail that rounds to zero batches is
            dropped). A missing ``num_epochs`` raises ``ValueError`` — there
            is no silent default. Otherwise an opaque pass-through to
            callbacks and experiment harnesses. Retired keys (``loss_type``,
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
        iter_callback: ``(ctx: IterContext) -> Any``; a non-None return
            replaces that batch's ``iter_results`` entry. Supplying it
            forces a per-batch device sync and makes the internal step
            return the batch's GraphState for ``ctx.state``. Exceptions
            propagate.

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
        data_axis_size = _data_axis_size(mesh)
        batch_sharding = NamedSharding(mesh, P("data"))
        replicated = NamedSharding(mesh, P())
        params = jax.device_put(params, replicated)
        opt_state = jax.device_put(opt_state, replicated)

    # The step returns the batch's GraphState only when a callback will read
    # it; without one the state is dropped inside the step.
    with_state = iter_callback is not None
    step_fn = _make_step(
        structure, optimizer, algorithm=algorithm, with_state=with_state, donate=True
    )

    if "num_epochs" not in config:
        raise ValueError(
            "config['num_epochs'] is required (fractional values are "
            "supported); there is no default epoch count."
        )
    num_epochs = config["num_epochs"]

    try:
        num_batches = len(train_loader)
    except TypeError as exc:
        raise TypeError(
            f"train_loader must support len() — the epoch schedule and "
            f"progress total need the batch count; got "
            f"{type(train_loader).__name__}. Wrap a generator in a list or a "
            f"loader class with __len__."
        ) from exc

    full_epochs = math.floor(num_epochs)
    frac = num_epochs - full_epochs
    # A fractional tail that rounds to zero batches is dropped entirely: no
    # empty epoch entry, no callback invoked on empty metrics.
    partial_batches = round(frac * num_batches) if frac > 0 else 0
    total_epochs = full_epochs + (1 if partial_batches > 0 else 0)
    total_batches = full_epochs * num_batches + partial_batches
    progress = _tqdm_cls(total=total_batches, disable=not verbose, leave=True)
    sync_per_batch = verbose or iter_callback is not None
    shard_warned = False

    step = 0
    iter_results: List[Any] = []
    epoch_results: List[Any] = []
    for epoch_offset in range(total_epochs):
        epoch_idx = start_epoch + epoch_offset
        max_batches = num_batches if epoch_offset < full_epochs else partial_batches
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
                bsz = batch_size_of(batch, structure)
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
            if with_state:
                params, opt_state, metrics, state = step_fn(
                    params, opt_state, batch, batch_key
                )
            else:
                params, opt_state, metrics = step_fn(
                    params, opt_state, batch, batch_key
                )
                state = None
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
                    ctx = IterContext(
                        epoch_idx=epoch_idx,
                        batch_idx=batch_idx,
                        step=step,
                        params=params,
                        opt_state=opt_state,
                        state=state,
                        structure=structure,
                        config=config,
                        algorithm=algorithm,
                        rng_key=rng_key,
                        epoch_key=epoch_key,
                        batch_key=batch_key,
                        batch=batch,
                        metrics=float_metrics,
                    )
                    replaced = iter_callback(ctx)
                    if replaced is not None:
                        stored = replaced
                    del ctx
                batch_metrics.append(stored)
            else:
                batch_metrics.append(metrics)
            # The context and this name were the trainer's only references to
            # the batch's GraphState; dropping both frees its device buffers
            # unless the callback retained them.
            del state
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
            epoch_ctx = EpochContext(
                epoch_idx=epoch_idx,
                step=step,
                params=params,
                opt_state=opt_state,
                structure=structure,
                config=config,
                algorithm=algorithm,
                rng_key=rng_key,
                epoch_key=epoch_key,
                metrics=epoch_means,
            )
            replaced = epoch_callback(epoch_ctx)
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
        batch_size = batch_size_of(batch, structure)
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
        data_axis_size = _data_axis_size(mesh)
        batch_sharding = NamedSharding(mesh, P("data"))
        mask_sharding = NamedSharding(mesh, P("data"))
        params = jax.device_put(params, NamedSharding(mesh, P()))

    totals = {name: (jnp.zeros(()), jnp.zeros(())) for name in metric_names}
    for batch_idx, batch_data in enumerate(test_loader):
        batch = convert_batch(batch_data)
        bsz = batch_size_of(batch, structure)
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
