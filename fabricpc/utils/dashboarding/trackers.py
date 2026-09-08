"""Core tracker classes for Aim integration.

This module provides the main tracking interface for experiment monitoring
with Aim. All tracking is designed to be optional and lazy-loaded.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from fabricpc.core.types import GraphParams, GraphState, GraphStructure
from fabricpc.utils.dashboarding._aim_available import get_aim, is_aim_available
from fabricpc.utils.dashboarding.extractors import (
    extract_node_energies,
    flatten_for_distribution,
)


@dataclass
class TrackingConfig:
    """Configuration for what to track and when.

    Attributes:
        track_energy: Track energy (batch and epoch level).
        track_accuracy: Track accuracy (epoch level).
        track_weight_distributions: Track weight distribution histograms.
        track_state: Track state summary stats (mean, std, L2 norm of
            z_latent, z_mu, and energy per node) on every
            ``tracking_every_n_batches``-th batch.
        track_state_distributions: Also track full distribution histograms
            for z_mu, z_latent, and energy; implies ``track_state``.
        track_error: Track prediction error statistics.
        nodes_to_track: When non-empty, scopes weight/bias and state
            tracking to these nodes and enables per-node energy and
            inference-dynamics breakdowns. Empty tracks weights and state
            for every node and logs no per-node breakdowns.
        tracking_every_n_batches: How often (in batches) to log weight
            distributions, state stats/distributions, and inference dynamics.
        tracking_every_n_epochs: How often (in epochs) to log epoch-level
            metrics such as weight distributions.
        state_tracking_every_n_infer_steps: Within a tracked batch, how often
            (in inference steps) to log state. Checked inside track_state().
        experiment_name: Name of the experiment in Aim.
        run_name: Name of this specific run.
    """

    # What to track
    track_energy: bool = True
    track_accuracy: bool = True
    track_error: bool = False
    track_weight_distributions: bool = True
    track_state: bool = False
    track_state_distributions: bool = False
    # Node-level filtering (empty = no per-node breakdown)
    nodes_to_track: List[str] = field(default_factory=list)
    # Frequency controls
    tracking_every_n_batches: int = 50
    tracking_every_n_epochs: int = 1
    state_tracking_every_n_infer_steps: int = 5
    # Naming
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None

    @property
    def tracks_state(self) -> bool:
        """True when state tracking fires: ``track_state`` or
        ``track_state_distributions`` (histograms need the same states)."""
        return self.track_state or self.track_state_distributions


class AimExperimentTracker:
    """Main tracker class for experiment tracking with Aim.

    Wraps Aim's Run object and provides FabricPC-specific tracking methods.
    All tracking methods are no-ops if Aim is not available.

    Example:
        tracker = AimExperimentTracker(config=TrackingConfig(
            track_weight_distributions=True,
            experiment_name="mnist_pcn"
        ))
        tracker.log_hyperparams(train_config)

        # In training loop:
        tracker.track_batch_energy(energy, epoch=epoch, batch=batch_idx)
        tracker.track_epoch_metrics({"accuracy": acc}, epoch=epoch)
        tracker.track_weight_distributions(params, structure, epoch=epoch)

        tracker.close()
    """

    def __init__(
        self,
        config: Optional[TrackingConfig] = None,
        aim_run: Any = None,
        repo: Optional[str] = None,
    ):
        """Initialize tracker.

        Args:
            config: TrackingConfig for what/when to track.
            aim_run: Optional existing Aim Run object.
            repo: Optional path to Aim repo (passed to Run if aim_run not provided).
        """
        self.config = config or TrackingConfig()
        self._run = aim_run
        self._repo = repo
        self._initialized = False
        self._global_step = 0

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of Aim Run.

        Returns:
            True if Aim is available and initialized, False otherwise.

        Raises:
            Any exception from Aim initialization that is not a connection
            or import error (those are warned and suppressed).
        """
        if self._initialized:
            return self._run is not None

        self._initialized = True
        if not is_aim_available():
            self._run = None
            print(
                "WARNING: Aim is not installed. Tracking disabled. Install with: pip install aim"
            )
            return False

        try:
            aim = get_aim()
            if self._run is None:
                self._run = aim.Run(
                    repo=self._repo,
                    experiment=self.config.experiment_name,
                )
                if self.config.run_name:
                    self._run.name = self.config.run_name
            return True
        except (ConnectionError, OSError, TimeoutError) as e:
            self._run = None
            print(f"WARNING: Failed to connect to Aim server: {e}. Tracking disabled.")
            return False

    @property
    def run(self) -> Any:
        """Access the underlying Aim Run (may be None if not available)."""
        self._ensure_initialized()
        return self._run

    def log_hyperparams(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters to Aim.

        Args:
            hparams: Dictionary of hyperparameters.
        """
        if not self._ensure_initialized():
            return
        self._run["hparams"] = hparams

    def log_graph_structure(self, structure: GraphStructure) -> None:
        """Log graph structure information.

        Args:
            structure: GraphStructure object.
        """
        if not self._ensure_initialized():
            return
        self._run["graph"] = {
            "num_nodes": len(structure.nodes),
            "num_edges": len(structure.edges),
            "node_order": list(structure.node_order),
            "task_map": structure.task_map,
            "nodes": {
                name: {
                    "shape": list(node.node_info.shape),
                    "type": node.node_info.node_type,
                    "in_degree": node.node_info.in_degree,
                    "out_degree": node.node_info.out_degree,
                }
                for name, node in structure.nodes.items()
            },
        }

    def track_batch_energy(
        self,
        energy: float,
        epoch: int,
        batch: int,
        context: Optional[Dict[str, str]] = None,
    ) -> None:
        """Track batch-level energy.

        Args:
            energy: energy value.
            epoch: Current epoch.
            batch: Current batch index.
            context: Optional additional context.
        """
        if not self.config.track_energy:
            return
        if not self._ensure_initialized():
            return

        ctx = {"subset": "train", "level": "batch"}
        if context:
            ctx.update(context)

        self._run.track(
            energy,
            name="energy",
            step=self._global_step,
            epoch=epoch,
            context=ctx,
        )
        self._global_step += 1

    def track_batch_energy_per_node(
        self,
        state: GraphState,
        structure: GraphStructure,
        epoch: int,
        batch: int,
    ) -> None:
        """Track per-node energy at batch level.

        Args:
            state: Final GraphState after inference.
            structure: GraphStructure.
            epoch: Current epoch.
            batch: Current batch index.
        """
        if not self.config.nodes_to_track:
            return
        if not self._ensure_initialized():
            return

        energies = extract_node_energies(state)
        for node_name in self.config.nodes_to_track:
            if node_name not in energies:
                continue
            energy = energies[node_name]
            if structure.nodes[node_name].node_info.in_degree > 0:
                mean_energy = float(np.mean(energy))
                self._run.track(
                    mean_energy,
                    name="node_energy",
                    step=self._global_step,
                    epoch=epoch,
                    context={"node": node_name, "subset": "train"},
                )

    def track_epoch_metrics(
        self,
        metrics: Dict[str, float],
        epoch: int,
        subset: str = "train",
    ) -> None:
        """Track epoch-level metrics.

        Args:
            metrics: Dictionary of metric name -> value.
            epoch: Current epoch.
            subset: "train" or "val".
        """
        if not self._ensure_initialized():
            return

        for name, value in metrics.items():
            should_track = (
                (name == "energy" and self.config.track_energy)
                or (name == "accuracy" and self.config.track_accuracy)
                or (name not in ["energy", "accuracy"])  # Track other metrics always
            )
            if should_track:
                self._run.track(
                    value,
                    name=name,
                    epoch=epoch,
                    context={"subset": subset, "level": "epoch"},
                )

    def track_weight_distributions(
        self,
        params: GraphParams,
        structure: GraphStructure,
        epoch: int,
        batch: int,
        nodes: Optional[List[str]] = None,
    ) -> None:
        """Track weight distributions using Aim Distribution.

        Args:
            params: Current GraphParams.
            structure: GraphStructure.
            epoch: Current epoch.
            batch: Current batch index.
            nodes: Optional list of nodes to track (default: all).
        """
        if not self.config.track_weight_distributions:
            return
        if batch % self.config.tracking_every_n_batches != 0:
            return
        if not self._ensure_initialized():
            return

        aim = get_aim()
        nodes = nodes or list(params.nodes.keys())

        for node_name in nodes:
            node_params = params.nodes[node_name]
            for edge_key, weight in node_params.weights.items():
                samples = flatten_for_distribution(weight)
                if samples.size == 0:
                    continue  # no finite values (diverged run) — nothing to bin
                self._run.track(
                    aim.Distribution(samples),
                    name="weights",
                    step=self._global_step,
                    epoch=epoch,
                    context={"node": node_name, "edge": edge_key},
                )

            for bias_key, bias in node_params.biases.items():
                samples = flatten_for_distribution(bias)
                if samples.size == 0:
                    continue
                self._run.track(
                    aim.Distribution(samples),
                    name="biases",
                    step=self._global_step,
                    epoch=epoch,
                    context={"node": node_name, "key": bias_key},
                )

    def track_state(
        self,
        state: GraphState,
        epoch: int,
        batch: int,
        infer_step: int,
        nodes: Optional[List[str]] = None,
    ) -> None:
        """Track node state summary stats and optional distributions.

        A no-op unless ``config.tracks_state`` (``track_state`` or
        ``track_state_distributions``). Logs mean, std, and L2 norm for
        z_latent, z_mu, and energy; adds full distribution histograms when
        ``config.track_state_distributions`` is True.

        Batch-level gating (``tracking_every_n_batches``) is the
        responsibility of the caller.  This method only checks the
        inference-step interval.

        Args:
            state: GraphState after (partial) inference.
            epoch: Current epoch.
            batch: Current batch index.
            infer_step: Current inference step within this batch.
            nodes: Optional list of nodes to track (default: all).
        """
        if not self.config.tracks_state:
            return
        if infer_step % self.config.state_tracking_every_n_infer_steps != 0:
            return
        if not self._ensure_initialized():
            return

        aim = get_aim()
        nodes = nodes or list(state.nodes.keys())

        state_vars = ("z_latent", "z_mu", "energy")

        for node_name in nodes:
            node_state = state.nodes[node_name]
            ctx = {"node": node_name, "infer_step": infer_step}

            for var_name in state_vars:
                data = np.asarray(getattr(node_state, var_name))

                # Always track summary statistics
                self._run.track(
                    float(np.mean(data)),
                    name=f"{var_name}_mean",
                    step=self._global_step,
                    epoch=epoch,
                    context=ctx,
                )
                self._run.track(
                    float(np.std(data)),
                    name=f"{var_name}_std",
                    step=self._global_step,
                    epoch=epoch,
                    context=ctx,
                )
                self._run.track(
                    float(np.linalg.norm(data)),
                    name=f"{var_name}_norm",
                    step=self._global_step,
                    epoch=epoch,
                    context=ctx,
                )

                # Optionally track full distributions
                if self.config.track_state_distributions:
                    samples = flatten_for_distribution(data)
                    if samples.size == 0:
                        continue
                    self._run.track(
                        aim.Distribution(samples),
                        name=var_name,
                        step=self._global_step,
                        epoch=epoch,
                        context=ctx,
                    )

    def track_inference_dynamics(
        self,
        state_history: List[GraphState],
        epoch: int,
        batch: int,
        nodes: Optional[List[str]] = None,
    ) -> None:
        """Track how states evolve over inference steps.

        Args:
            state_history: List of GraphState, one per inference step.
            epoch: Current epoch.
            batch: Current batch.
            nodes: Optional list of nodes to track.
        """
        nodes = nodes or self.config.nodes_to_track
        if not nodes:
            return
        if not self._ensure_initialized():
            return

        for infer_step, state in enumerate(state_history):
            for node_name in nodes:
                if node_name not in state.nodes:
                    continue
                node_state = state.nodes[node_name]

                # Track energy evolution
                mean_energy = float(np.mean(np.asarray(node_state.energy)))
                self._run.track(
                    mean_energy,
                    name="inference_energy",
                    step=infer_step,
                    context={
                        "node": node_name,
                        "epoch": epoch,
                        "batch": batch,
                    },
                )

                # Track latent gradient norm (convergence indicator)
                grad_norm = float(np.linalg.norm(np.asarray(node_state.latent_grad)))
                self._run.track(
                    grad_norm,
                    name="inference_grad_norm",
                    step=infer_step,
                    context={
                        "node": node_name,
                        "epoch": epoch,
                        "batch": batch,
                    },
                )

                # Track error norm
                error_norm = float(np.linalg.norm(np.asarray(node_state.error)))
                self._run.track(
                    error_norm,
                    name="inference_error_norm",
                    step=infer_step,
                    context={
                        "node": node_name,
                        "epoch": epoch,
                        "batch": batch,
                    },
                )

    def close(self) -> None:
        """Close the Aim run."""
        if self._run is not None:
            self._run.close()
            self._run = None


class StateHistoryCollector:
    """Collects GraphState history during inference for later analysis.

    This is a lightweight container that can be used with the modified
    inference loop to capture per-step dynamics without side effects
    during JIT compilation.

    Example:
        collector = StateHistoryCollector()
        final_state, history = run_inference_with_history(
            params, init_state, clamps, structure
        )
        collector.add_history(history, {"epoch": epoch, "batch": batch})

        # Later, outside JIT:
        tracker.track_inference_dynamics(collector.latest, epoch, batch)
    """

    def __init__(self, max_histories: int = 10):
        """Initialize collector.

        Args:
            max_histories: Maximum number of histories to keep.
        """
        self.max_histories = max_histories
        self._histories: List[List[GraphState]] = []
        self._metadata: List[Dict[str, Any]] = []

    def add_history(
        self,
        history: List[GraphState],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a state history from one training step.

        Args:
            history: List of GraphState from inference loop.
            metadata: Optional metadata (epoch, batch, etc.).
        """
        self._histories.append(history)
        self._metadata.append(metadata or {})

        # Keep only recent histories
        if len(self._histories) > self.max_histories:
            self._histories.pop(0)
            self._metadata.pop(0)

    @property
    def history(self) -> List[List[GraphState]]:
        """Get all collected histories."""
        return self._histories

    @property
    def latest(self) -> Optional[List[GraphState]]:
        """Get the most recent history."""
        return self._histories[-1] if self._histories else None

    @property
    def latest_metadata(self) -> Optional[Dict[str, Any]]:
        """Get metadata for the most recent history."""
        return self._metadata[-1] if self._metadata else None

    def clear(self) -> None:
        """Clear all collected histories."""
        self._histories.clear()
        self._metadata.clear()
