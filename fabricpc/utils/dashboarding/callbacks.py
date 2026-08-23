"""Callback factories for integration with fabricpc.training.train.

``create_iter_callback``/``create_epoch_callback``/``create_tracking_callbacks``
produce callbacks for train's ``iter_callback``/``epoch_callback`` parameters.
``create_detailed_iter_callback`` is the exception: it consumes the final
``GraphState``, which only ``make_train_step``'s step returns, so it plugs
into a custom loop, not into ``train``.
"""

from typing import Any, Callable, Dict, Optional, Tuple

from fabricpc.core.types import GraphState, GraphStructure
from fabricpc.training.trainer import EpochContext
from fabricpc.utils.dashboarding.trackers import AimExperimentTracker, TrackingConfig


def create_iter_callback(
    tracker: AimExperimentTracker,
) -> Callable[[int, int, Dict[str, float]], Dict[str, float]]:
    """Create an iter_callback for train that tracks batch energy.

    Args:
        tracker: AimExperimentTracker instance.

    Returns:
        Callback function: (epoch_idx, batch_idx, metrics) -> metrics
    """

    def iter_callback(
        epoch_idx: int, batch_idx: int, metrics: Dict[str, float]
    ) -> Dict[str, float]:
        # metrics["energy"] is the per-sample training objective.
        tracker.track_batch_energy(metrics["energy"], epoch=epoch_idx, batch=batch_idx)
        return metrics

    return iter_callback


def create_epoch_callback(
    tracker: AimExperimentTracker,
    structure: GraphStructure,
    eval_fn: Optional[Callable] = None,
    eval_loader: Any = None,
    eval_config: Optional[dict] = None,
) -> Callable[[EpochContext], Optional[dict]]:
    """Create an epoch_callback for train that tracks epoch metrics and
    distributions.

    The returned callback runs an optional evaluation and returns its metrics
    dict; train stores a non-None return as that epoch's ``epoch_results``
    entry, so dashboards get mid-training eval results in the history.

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
        # Track weight distributions
        tracker.track_weight_distributions(
            ctx.params, ctx.structure, epoch=ctx.epoch_idx, batch=0
        )

        # Optionally run evaluation
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


def create_detailed_iter_callback(
    tracker: AimExperimentTracker,
    structure: GraphStructure,
) -> Callable[[int, int, Dict[str, float], "GraphState"], Dict[str, float]]:
    """Create a per-batch callback that also tracks state distributions.

    This callback consumes the final GraphState, which only
    ``make_train_step``'s step returns, so it plugs into a custom loop —
    not into ``train(iter_callback=...)``, whose callbacks receive
    ``(epoch_idx, batch_idx, metrics)`` without the state. Custom-loop
    example: docs/user_guides/09_experiment_tracking.md.

    Args:
        tracker: AimExperimentTracker instance.
        structure: GraphStructure.

    Returns:
        Callback function:
        (epoch_idx, batch_idx, metrics, final_state) -> metrics,
        where ``metrics`` is the step's metric dict
        (``{"energy", "target_energy"}``).
    """

    def detailed_iter_callback(
        epoch_idx: int,
        batch_idx: int,
        metrics: Dict[str, float],
        final_state: GraphState,
    ) -> Dict[str, float]:
        # metrics["energy"] is the per-sample training objective.
        tracker.track_batch_energy(
            float(metrics["energy"]), epoch=epoch_idx, batch=batch_idx
        )

        # Track per-node energy
        tracker.track_batch_energy_per_node(
            final_state, structure, epoch=epoch_idx, batch=batch_idx
        )

        # Track state stats/distributions (if at right batch + infer_step frequency)
        if batch_idx % tracker.config.tracking_every_n_batches == 0:
            tracker.track_state(
                final_state, epoch=epoch_idx, batch=batch_idx, infer_step=0
            )

        return metrics

    return detailed_iter_callback
