"""Callback factories for integration with fabricpc.training.train.

``create_iter_callback``/``create_epoch_callback``/``create_tracking_callbacks``
produce callbacks for train's ``iter_callback``/``epoch_callback`` parameters.
The iteration callback reads everything it logs from its ``IterContext``
(metrics, parameters, the batch's GraphState, the batch and its key);
``TrackingConfig`` decides which of the tracker's batch-level methods it
calls, so one factory serves energy-only runs and full state tracking.
"""

from typing import Any, Callable, Dict, Optional, Tuple

from fabricpc.core.types import GraphStructure
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training.trainer import (
    EpochContext,
    IterContext,
    batch_size_of,
    build_clamps,
)
from fabricpc.utils.dashboarding.inference_tracking import (
    run_inference_with_full_history,
)
from fabricpc.utils.dashboarding.trackers import AimExperimentTracker, TrackingConfig


def create_iter_callback(
    tracker: AimExperimentTracker,
) -> Callable[[IterContext], Dict[str, float]]:
    """Create an iter_callback for train; ``tracker.config`` decides what it logs.

    Per batch, in order:

    1. batch energy from ``ctx.metrics`` (``track_energy``);
    2. per-node energy from ``ctx.state`` (``nodes_to_track``);
    3. weight distributions from ``ctx.params`` every
       ``tracking_every_n_batches`` (``track_weight_distributions``);
    4. on those same batches, state tracking when ``track_state`` or
       ``track_state_distributions`` is set. Under PC the callback re-runs
       inference on ``ctx.batch`` from ``ctx.batch_key`` under the
       post-update ``ctx.params`` and logs every
       ``state_tracking_every_n_infer_steps``-th step: one extra inference
       pass per tracked batch. Under backprop there is no settling to
       record, so it logs the feedforward ``ctx.state`` once at
       ``infer_step=0``.

    A non-empty ``nodes_to_track`` scopes items 3 and 4 to those nodes; empty
    tracks weights and state for every node.

    Args:
        tracker: AimExperimentTracker instance.

    Returns:
        Callback ``(ctx: IterContext) -> ctx.metrics``, so train stores the
        float metrics as usual.
    """

    def iter_callback(ctx: IterContext) -> Dict[str, float]:
        epoch, batch = ctx.epoch_idx, ctx.batch_idx
        config = tracker.config
        # A non-empty nodes_to_track scopes weights and state; None means all.
        nodes = config.nodes_to_track or None
        # metrics["energy"] is the training objective per prediction.
        tracker.track_batch_energy(ctx.metrics["energy"], epoch=epoch, batch=batch)
        tracker.track_batch_energy_per_node(
            ctx.state, ctx.structure, epoch=epoch, batch=batch
        )
        tracker.track_weight_distributions(
            ctx.params, ctx.structure, epoch=epoch, batch=batch, nodes=nodes
        )

        if config.tracks_state and batch % config.tracking_every_n_batches == 0:
            if ctx.algorithm == "pc":
                clamps = build_clamps(ctx.batch, ctx.structure, clamp_target=True)
                init_state = initialize_graph_state(
                    ctx.structure,
                    batch_size_of(ctx.batch, ctx.structure),
                    ctx.batch_key,
                    clamps=clamps,
                    params=ctx.params,
                )
                _, history = run_inference_with_full_history(
                    ctx.params, init_state, clamps, ctx.structure
                )
                for infer_step, step_state in enumerate(history):
                    tracker.track_state(
                        step_state,
                        epoch=epoch,
                        batch=batch,
                        infer_step=infer_step,
                        nodes=nodes,
                    )
            else:
                tracker.track_state(
                    ctx.state, epoch=epoch, batch=batch, infer_step=0, nodes=nodes
                )
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
