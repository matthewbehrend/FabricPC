# Experiment Tracking with Aim

FabricPC integrates with [Aim](https://aimstack.io/) for comprehensive experiment tracking and visualization. This enables detailed monitoring of training quality, batch-level debugging, and hyperparameter tuning for predictive coding networks.

## Installation

> Aim supports Python ≤3.12. On Python 3.13 it is skipped automatically by `[viz]`/`[all]`,
> so experiment tracking is unavailable there — use a Python ≤3.12 environment for Aim.

```bash
pip install "fabricpc[viz]"
```

Or install Aim directly:

```bash
pip install aim
```

## Quick Start

```python
from fabricpc.utils.dashboarding import (
    AimExperimentTracker,
    TrackingConfig,
    create_tracking_callbacks,
)
from fabricpc.training import train, evaluate
import optax

# Create tracking configuration
tracking_config = TrackingConfig(
    experiment_name="my_experiment",
    track_energy=True,
    track_weight_distributions=True,
)

# Create callbacks for train
tracker, iter_cb, epoch_cb = create_tracking_callbacks(
    config=tracking_config,
    structure=structure,
    eval_fn=evaluate,
    eval_loader=test_loader,
    hparams=train_config,
)

# Train with tracking
optimizer = optax.adamw(1e-3)
result = train(
    params, structure, train_loader, optimizer, train_config, rng_key,
    iter_callback=iter_cb,
    epoch_callback=epoch_cb,
)
trained_params = result.params

# Close the tracker
tracker.close()
```

Then launch the Aim UI:

```bash
aim up
```

Click on the link returned in the console to explore the dashboard.

Be sure to run the python script and start aim from the same working directory to ensure the tracking data in folder .aim/ is correctly linked.

## TrackingConfig Options

```python
@dataclass
class TrackingConfig:
    # What to track
    track_energy: bool = True
    track_accuracy: bool = True
    track_error: bool = False
    track_weight_distributions: bool = True
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
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `track_energy` | `bool` | `True` | Track energy at both batch and epoch level. |
| `track_accuracy` | `bool` | `True` | Track accuracy at epoch level. |
| `track_error` | `bool` | `False` | Track prediction error statistics. |
| `track_weight_distributions` | `bool` | `True` | Track weight and bias distribution histograms. |
| `track_state_distributions` | `bool` | `False` | Track full distribution histograms for `z_latent`, `z_mu`, and `energy`. Summary stats (mean, std, norm) are always collected when state tracking fires. |
| `nodes_to_track` | `List[str]` | `[]` | Nodes for per-node tracking. Empty list disables per-node breakdowns (energy, state, inference dynamics). |
| `tracking_every_n_batches` | `int` | `50` | How often (in batches) to log weight distributions, state stats/distributions, and inference dynamics. |
| `tracking_every_n_epochs` | `int` | `1` | How often (in epochs) to log epoch-level metrics such as weight distributions. |
| `state_tracking_every_n_infer_steps` | `int` | `5` | Within a tracked batch, how often (in inference steps) to log state. |
| `experiment_name` | `Optional[str]` | `None` | Name of the experiment in Aim. |
| `run_name` | `Optional[str]` | `None` | Name of this specific run. |

## Tracking Predictive Coding Metrics

### Weight Distributions

Track how weights and biases evolve during training:

```python
config = TrackingConfig(
    track_weight_distributions=True,
    tracking_every_n_batches=100,  # Log every 100 batches
)
```

### State Distributions

Track `z_latent`, `z_mu`, and `energy` distributions per node. Summary statistics (mean, std, norm) are always collected when state tracking fires; set `track_state_distributions=True` to also log full distribution histograms:

```python
config = TrackingConfig(
    track_state_distributions=True,
    tracking_every_n_batches=50,
    state_tracking_every_n_infer_steps=5,
)
```

### Per-Node Energy and Inference Dynamics

Use `nodes_to_track` to enable per-node energy breakdowns and inference dynamics tracking for specific nodes:

```python
config = TrackingConfig(
    nodes_to_track=["h1", "h2", "h3"],  # Specific nodes
)
```

## Advanced Usage: Custom Training Loop

For detailed tracking including inference dynamics, use a custom training loop with `train_step_with_history`:

```python
from fabricpc.utils.dashboarding import (
    AimExperimentTracker,
    TrackingConfig,
    train_step_with_history,
    unstack_inference_history,
    summarize_inference_convergence,
)

tracking_config = TrackingConfig(
    experiment_name="detailed_tracking",
    track_state_distributions=True,
    nodes_to_track=["h1", "h2", "h3"],
    tracking_every_n_batches=50,
)

tracker = AimExperimentTracker(config=tracking_config)

# JIT compile with history collection
collect_every = 5  # Collect every 5th inference step
jit_train_step = jax.jit(
    lambda p, o, b, k: train_step_with_history(
        p, o, b, structure, optimizer, k,
        collect_every=collect_every,
    )
)

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        params, opt_state, energy, final_state, stacked_history = jit_train_step(
            params, opt_state, batch, rng_key
        )

        # Unstack inference history outside of JIT
        inference_history = unstack_inference_history(
            stacked_history, collect_every=collect_every
        )

        # Track batch metrics
        tracker.track_batch_energy(energy / batch_size, epoch, batch_idx)
        tracker.track_batch_energy_per_node(final_state, structure, epoch, batch_idx)

        # Track state stats/distributions at configured frequency
        if batch_idx % tracker.config.tracking_every_n_batches == 0:
            tracker.track_state(
                final_state, epoch=epoch, batch=batch_idx, infer_step=0
            )

        # Analyze inference convergence
        convergence = summarize_inference_convergence(inference_history)
        print(f"h1 final energy: {convergence['h1']['final_energy']:.4f}")

tracker.close()
```

## Metric Extractors

Use extractors to get specific metrics from `GraphState` and `GraphParams`:

```python
from fabricpc.utils.dashboarding import (
    extract_node_energies,
    extract_total_energy,
    extract_weight_statistics,
    extract_bias_statistics,
    extract_latent_statistics,
    extract_activation_statistics,
    extract_error_statistics,
    extract_latent_grad_statistics,
    extract_all_distributions,
)

# After training step
energies = extract_node_energies(final_state)
# {'pixels': array([...]), 'h1': array([...]), ...}

total_energy = extract_total_energy(final_state, structure)
# float

weight_stats = extract_weight_statistics(params)
# {'h1': {'pixels->h1:in': {'mean': 0.01, 'std': 0.05, ...}}}

latent_stats = extract_latent_statistics(final_state)
# {'h1': {'mean': 0.5, 'std': 0.2, 'min': 0.0, 'max': 1.0}}
```

## Graceful Degradation

The dashboarding module works even when Aim is not installed:

```python
from fabricpc.utils.dashboarding import is_aim_available

if is_aim_available():
    tracker = AimExperimentTracker(config)
    # Full tracking
else:
    tracker = None
    # Training continues without tracking
```

## Best Practices for PC Debugging

1. **Track weight distributions** to detect exploding/vanishing gradients in the Hebbian learning updates.

2. **Track per-node energy** (`nodes_to_track`) to identify which layers are contributing most to the total energy.

3. **Track inference dynamics** to verify that the inference loop converges (energy should decrease, gradient norms should approach zero). Use `train_step_with_history` and `summarize_inference_convergence` for detailed analysis.

4. **Monitor state distributions** (`track_state_distributions`) to ensure `z_latent` and `z_mu` values are in the expected range (e.g., [0, 1] for sigmoid).

5. **Tune tracking frequency** using `tracking_every_n_batches` and `state_tracking_every_n_infer_steps` to balance detail vs. overhead. Use frequent tracking for debugging and sparser tracking for production runs.

## Example Output

See `examples/mnist_aim_tracking.py` for a complete example that tracks:

- Batch-level system energy and per-node energy
- Epoch-level accuracy and weight/bias distributions
- State distributions (`z_latent`, `z_mu`, `energy`) with summary stats per node
- Inference dynamics (energy and gradient norm convergence per step)

## Launching the Dashboard

After training, view your experiments:

```bash
aim up
```

This opens a web interface at `http://localhost:43800` where you can:

- Compare runs with different hyperparameters
- Visualize weight distribution evolution
- Explore per-node energy contributions
- Analyze inference convergence patterns
