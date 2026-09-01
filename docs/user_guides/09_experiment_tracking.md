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
    track_state: bool = False
    track_state_distributions: bool = False

    # Node-level filtering (empty = no per-node breakdown / no distributions)
    nodes_to_track: List[str] = field(default_factory=list)
    distribution_nodes: List[str] = field(default_factory=list)

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
| `track_weight_distributions` | `bool` | `True` | Track weight and bias distribution histograms for `distribution_nodes` every `tracking_every_n_batches`. |
| `track_state` | `bool` | `False` | Track per-node state summary stats (mean, std, L2 norm of `z_latent`, `z_mu`, `energy`) for `distribution_nodes` on every `tracking_every_n_batches`-th batch. |
| `track_state_distributions` | `bool` | `False` | Also track full distribution histograms for `z_latent`, `z_mu`, and `energy`; implies `track_state`. |
| `nodes_to_track` | `List[str]` | `[]` | Nodes for per-node breakdowns: batch-level per-node energy and inference dynamics. Empty logs no per-node breakdowns. |
| `distribution_nodes` | `List[str]` | `[]` | Nodes whose weight/bias and state distributions are logged. Empty logs no distributions, as an empty `nodes_to_track` logs no per-node breakdowns. |
| `tracking_every_n_batches` | `int` | `50` | How often (in batches) the iteration callback logs weight distributions and state stats/distributions, and custom loops log inference dynamics. |
| `tracking_every_n_epochs` | `int` | `1` | Reserved; read by nothing. |
| `state_tracking_every_n_infer_steps` | `int` | `5` | Within a tracked batch, log the state after every this-many inference steps (`0, k, 2k, ...` up to `infer_steps`). |
| `experiment_name` | `Optional[str]` | `None` | Name of the experiment in Aim. |
| `run_name` | `Optional[str]` | `None` | Name of this specific run. |

## Tracking Predictive Coding Metrics

### Weight Distributions

Track how weights and biases evolve during training. The iteration callback
logs them for the nodes in `distribution_nodes` every
`tracking_every_n_batches`; an empty `distribution_nodes` logs none:

```python
config = TrackingConfig(
    track_weight_distributions=True,
    distribution_nodes=["h1", "h2"],
    tracking_every_n_batches=100,  # Log every 100 batches
)
```

### State Distributions

Set `track_state=True` for per-node summary statistics (mean, std, L2 norm)
of `z_latent`, `z_mu`, and `energy`, or `track_state_distributions=True` to
also log their histograms, for the nodes in `distribution_nodes`. On every
`tracking_every_n_batches`-th batch the iteration callback from
`create_tracking_callbacks` settles that batch again under the updated
parameters, initialized from the step's batch key, in one jitted program
(`make_tracked_settle`), and logs the state after `0, k, 2k, ...` inference
steps up to `infer_steps`, with `k = state_tracking_every_n_infer_steps`.
Step 0 is the initialization and, when `infer_steps` is a multiple of `k`,
the last record is the settled state. This is a fresh settle, not the
training settle: under `FeedforwardStateInit` the initial latents also come
from the updated parameters, and the per-node energy logged for the same
batch (`nodes_to_track`) comes from the training settle. Cost: one jitted
settle per tracked batch. Under `algorithm="backprop"` there is no settling
to record; the feedforward state is logged once per tracked batch at
`infer_step=0`.

```python
config = TrackingConfig(
    track_state_distributions=True,
    distribution_nodes=["h1", "h2"],
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

To record the training settle's own inference history on every batch, without the second inference pass the iteration callback makes on tracked batches, use a custom training loop with `train_step_with_history`, which collects the history inside the jitted step:

```python
import jax
from fabricpc.training import convert_batch
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
    epoch_key = jax.random.fold_in(rng_key, epoch)
    for batch_idx, batch_data in enumerate(train_loader):
        # convert_batch turns a (x, y) tuple batch into the {"x", "y"} dict
        # the step's clamp assembly iterates.
        batch = convert_batch(batch_data)
        batch_key = jax.random.fold_in(epoch_key, batch_idx)
        params, opt_state, energy, final_state, stacked_history = jit_train_step(
            params, opt_state, batch, batch_key
        )

        # Unstack inference history outside of JIT
        inference_history = unstack_inference_history(
            stacked_history, collect_every=collect_every
        )

        # energy is the objective per prediction (graph_energy over
        # internal nodes / prediction count) — no further normalization.
        tracker.track_batch_energy(float(energy), epoch, batch_idx)
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

### Standalone inference probes with `make_tracked_probe`
#### TODO: migrate this to new callback API
To record per-step energies on a fixed batch without training,
`make_tracked_probe` returns a jitted `params -> (final_state,
stacked_metrics)` callable that compiles `initialize_graph_state` and
`run_inference_with_history` into one XLA program. Keep them together:
initializing eagerly and tracking under `jax.jit` runs the same convolutions
in two separately compiled programs, which on GPU at default precision can
select different cuDNN algorithms (TF32 vs FP32, per conv shape) — unclamped
nodes then record the squared difference between the two paths (up to ~1e-3)
as their step-0 energy instead of 0.

```python
from fabricpc.utils.dashboarding import make_tracked_probe

probe = make_tracked_probe(structure, clamps, rng_key, batch_size)
final_state, stacked_metrics = probe(params)  # reuse across param sets
```

`clamps` and `rng_key` are fixed at creation; calling the probe with new
params (e.g. training checkpoints) reuses the compiled program, so recorded
histories differ only in params.

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

1. **Track weight distributions** (`distribution_nodes`) to detect exploding/vanishing gradients in the Hebbian learning updates.

2. **Track per-node energy** (`nodes_to_track`) to identify which layers are contributing most to the total energy.

3. **Track inference dynamics** to verify that the inference loop converges (energy should decrease, gradient norms should approach zero). Use `train_step_with_history` and `summarize_inference_convergence` for detailed analysis.

4. **Monitor state distributions** (`track_state_distributions` with `distribution_nodes`) to ensure `z_latent` and `z_mu` values are in the expected range (e.g., [0, 1] for sigmoid).

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
