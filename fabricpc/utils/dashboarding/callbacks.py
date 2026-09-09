"""Callback factories for ``train`` that log to an ``AimExperimentTracker``.

``create_iter_callback``/``create_epoch_callback``/``create_tracking_callbacks``
produce callbacks for train's ``iter_callback``/``epoch_callback`` parameters.
The iteration callback reads everything it logs from its ``IterContext``
(metrics, parameters, the batch's GraphState, the batch and its key);
``TrackingConfig`` decides which of the tracker's batch-level methods log
anything, so one factory serves energy-only runs and full state tracking.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training.trainer import (
    EpochContext,
    IterContext,
    batch_size_of,
    build_clamps,
)
from fabricpc.utils.dashboarding.inference_tracking import make_inference_history
from fabricpc.utils.dashboarding.trackers import AimExperimentTracker, TrackingConfig


def make_tracked_settle(
    structure: GraphStructure, *, every: int
) -> Callable[[GraphParams, jax.Array, Dict[str, jnp.ndarray]], GraphState]:
    """Build the jitted re-settle the iteration callback runs on a tracked
    PC batch: clamps and latent initialization from the batch and key, then
    :func:`make_inference_history` sampled every ``every`` steps.

    Returns ``settle(params, key, batch) -> states``, a GraphState pytree
    with a leading axis of length ``infer_steps // every + 1`` (index ``i`` is
    the state after ``i * every`` inference steps). One compiled program per
    structure and batch shape.
    """
    history = make_inference_history(structure, every=every)

    def settle(params, key, batch):
        clamps = build_clamps(batch, structure, clamp_target=True)
        init_state = initialize_graph_state(
            structure,
            batch_size_of(batch, structure),
            key,
            clamps=clamps,
            params=params,
        )
        _, states = history(params, init_state, clamps)
        return states

    return jax.jit(settle)


def create_iter_callback(
    tracker: AimExperimentTracker,
) -> Callable[[IterContext], Dict[str, float]]:
    """Create an iter_callback for train; ``tracker.config`` decides what it logs.

    Per batch, in order:

    1. batch energy from ``ctx.metrics`` (``track_energy``);
    2. per-node energy of the training settle, ``ctx.state``
       (``nodes_to_track``);
    3. weight and bias distributions from ``ctx.params`` every
       ``tracking_every_n_batches``, for ``distribution_nodes``
       (``track_weight_distributions``);
    4. on those same batches, state tracking for ``distribution_nodes`` when
       ``track_state`` or ``track_state_distributions`` is set. Under PC the
       callback settles ``ctx.batch`` again under the post-update
       ``ctx.params``, initialized with ``ctx.batch_key``, in one jitted
       program (:func:`make_tracked_settle`), and logs the state after
       ``0, k, 2k, ...`` inference steps up to ``infer_steps`` with
       ``k = state_tracking_every_n_infer_steps``. This is a fresh settle,
       not the training settle: under a parameter-dependent initializer
       (``FeedforwardStateInit``) the initial latents also come from the
       updated parameters. Cost: one jitted settle per tracked batch. Under
       backprop there is no settling to record, so it logs the feedforward
       ``ctx.state`` once at ``infer_step=0``.

    An empty ``distribution_nodes`` logs no distributions (items 3 and 4), as
    an empty ``nodes_to_track`` logs no per-node energy (item 2).

    Args:
        tracker: AimExperimentTracker instance.

    Returns:
        Callback ``(ctx: IterContext) -> ctx.metrics``, so train stores the
        float metrics as usual.
    """
    # The re-settle compiles once per structure; train has one structure, so
    # the cache is a single entry checked by identity.
    settle_fn: Optional[Callable] = None
    settle_structure: Optional[GraphStructure] = None

    def iter_callback(ctx: IterContext) -> Dict[str, float]:
        nonlocal settle_fn, settle_structure
        epoch, batch = ctx.epoch_idx, ctx.batch_idx
        config = tracker.config
        # metrics["energy"] is the training objective per prediction.
        tracker.track_batch_energy(ctx.metrics["energy"], epoch=epoch, batch=batch)
        tracker.track_batch_energy_per_node(
            ctx.state, ctx.structure, epoch=epoch, batch=batch
        )
        tracker.track_weight_distributions(
            ctx.params, ctx.structure, epoch=epoch, batch=batch
        )

        if config.tracks_state and batch % config.tracking_every_n_batches == 0:
            if ctx.algorithm == "pc":
                every = config.state_tracking_every_n_infer_steps
                if settle_fn is None or settle_structure is not ctx.structure:
                    settle_fn = make_tracked_settle(ctx.structure, every=every)
                    settle_structure = ctx.structure
                states = settle_fn(ctx.params, ctx.batch_key, ctx.batch)
                n_sampled = jax.tree_util.tree_leaves(states)[0].shape[0]
                for i in range(n_sampled):
                    tracker.track_state(
                        jax.tree_util.tree_map(lambda a, i=i: a[i], states),
                        epoch=epoch,
                        batch=batch,
                        infer_step=i * every,
                    )
            else:
                tracker.track_state(ctx.state, epoch=epoch, batch=batch, infer_step=0)
        return ctx.metrics

    return iter_callback


def create_epoch_callback(
    tracker: AimExperimentTracker,
    structure: GraphStructure,
    eval_fn: Optional[Callable] = None,
    eval_loader: Any = None,
    eval_config: Optional[dict] = None,
) -> Callable[[EpochContext], Optional[dict]]:
    """Create an epoch_callback for train that runs and tracks evaluation.

    The returned callback runs an optional evaluation and returns its metrics
    dict; train stores a non-None return as that epoch's ``epoch_results``
    entry, so dashboards get mid-training eval results in the history.
    Weight distributions are logged by the iteration callback at the
    ``tracking_every_n_batches`` cadence, not here.

    Args:
        tracker: AimExperimentTracker instance.
        structure: GraphStructure.
        eval_fn: Optional evaluation function (e.g., fabricpc.training.evaluate).
        eval_loader: Optional evaluation data loader.
        eval_config: Optional evaluation config.

    Returns:
        Callback function taking an EpochContext.
    """

    def epoch_callback(ctx: EpochContext) -> Optional[dict]:
        eval_metrics = None
        if eval_fn is not None and eval_loader is not None:
            eval_metrics = eval_fn(
                ctx.params,
                ctx.structure,
                eval_loader,
                eval_config or ctx.config,
                ctx.rng_key,
            )
            tracker.track_epoch_metrics(eval_metrics, epoch=ctx.epoch_idx, subset="val")

        return eval_metrics

    return epoch_callback


def create_tracking_callbacks(
    config: Optional[TrackingConfig] = None,
    structure: Optional[GraphStructure] = None,
    eval_fn: Optional[Callable] = None,
    eval_loader: Any = None,
    eval_config: Optional[dict] = None,
    hparams: Optional[dict] = None,
    repo: Optional[str] = None,
) -> Tuple[AimExperimentTracker, Callable, Optional[Callable]]:
    """Create both iter_callback and epoch_callback with a shared tracker.

    This is the recommended way to set up tracking for train.

    Args:
        config: TrackingConfig (optional, uses defaults if not provided).
        structure: GraphStructure (required for epoch callback).
        eval_fn: Optional evaluation function.
        eval_loader: Optional evaluation data loader.
        eval_config: Optional evaluation config.
        hparams: Optional hyperparameters to log.
        repo: Optional path to Aim repository.

    Returns:
        Tuple of (tracker, iter_callback, epoch_callback).

    Usage example: docs/user_guides/09_experiment_tracking.md (the canonical,
    contract-tested copy).
    """
    tracker = AimExperimentTracker(config or TrackingConfig(), repo=repo)

    if hparams:
        tracker.log_hyperparams(hparams)

    if structure:
        tracker.log_graph_structure(structure)

    iter_callback = create_iter_callback(tracker)
    epoch_callback = (
        create_epoch_callback(tracker, structure, eval_fn, eval_loader, eval_config)
        if structure
        else None
    )

    return tracker, iter_callback, epoch_callback
