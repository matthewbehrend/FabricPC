"""
MNIST with Aim Experiment Tracking

Tracks batch/epoch energy, weight/latent distributions, per-node energy,
and inference dynamics using Aim.

Architecture::

    pixels(784) ──→ h1(256) ──→ h2(64) ──→ h3(64) ──→ class(10)
     Identity       Sigmoid     Sigmoid     Sigmoid    Softmax+CE

After running, launch the Aim UI with:  aim up

Requirements: pip install fabricpc[viz]
"""

import time

import jax
import jax.numpy as jnp
from fabricpc.utils.data.dataloader import MnistLoader

from fabricpc.nodes import Linear, IdentityNode
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params, FeedforwardStateInit
from fabricpc.core.activations import (
    SigmoidActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import GaussianEnergy, CrossEntropyEnergy
from fabricpc.core.initializers import KaimingInitializer
from fabricpc.core.inference import InferenceSGD
import optax
from fabricpc.training import evaluate

# Import dashboarding utilities
from fabricpc.utils.dashboarding import (
    AimExperimentTracker,
    TrackingConfig,
    is_aim_available,
    train_step_with_history,
    unstack_inference_history,
    summarize_inference_convergence,
)
from fabricpc import setup_jax

setup_jax()

# Check if Aim is available
if not is_aim_available():
    print("WARNING: Aim is not installed. Install with: pip install aim")
    print("Tracking will be disabled. Continuing with training only...")
    TRACKING_ENABLED = False
else:
    TRACKING_ENABLED = True
    print("Aim is available. Experiment tracking enabled.")

jax.config.update("jax_default_prng_impl", "threefry2x32")
master_rng_key = jax.random.PRNGKey(42)
graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

# --- Network ---

pixels = IdentityNode(shape=(784,), name="pixels")
h1 = Linear(
    shape=(256,),
    activation=SigmoidActivation(),
    energy=GaussianEnergy(precision=1.0),
    weight_init=KaimingInitializer(),
    name="h1",
)
h2 = Linear(
    shape=(64,),
    activation=SigmoidActivation(),
    energy=GaussianEnergy(precision=1.0),
    weight_init=KaimingInitializer(),
    name="h2",
)
h3 = Linear(
    shape=(64,),
    activation=SigmoidActivation(),
    energy=GaussianEnergy(precision=1.0),
    weight_init=KaimingInitializer(),
    name="h3",
)
class_node = Linear(
    shape=(10,),
    activation=SoftmaxActivation(),
    energy=CrossEntropyEnergy(),
    weight_init=KaimingInitializer(),
    name="class",
)

structure = graph(
    nodes=[pixels, h1, h2, h3, class_node],
    edges=[
        Edge(source=pixels, target=h1.slot("in")),
        Edge(source=h1, target=h2.slot("in")),
        Edge(source=h2, target=h3.slot("in")),
        Edge(source=h3, target=class_node.slot("in")),
    ],
    task_map=TaskMap(x=pixels, y=class_node),
    graph_state_initializer=FeedforwardStateInit(),
    inference=InferenceSGD(eta_infer=0.20, infer_steps=20),
)

optimizer = optax.adamw(0.001, weight_decay=0.001)
train_config = {}
batch_size = 200
num_epochs = 1
INFERENCE_COLLECT_EVERY = 5  # Inference history collection interval

# --- Create Model ---

params = initialize_params(structure, graph_key)
num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

print(f"{len(structure.nodes)} nodes, {num_params:,} parameters")

# --- Data ---

train_loader = MnistLoader(
    "train", batch_size=batch_size, tensor_format="flat", shuffle=True, seed=42
)
test_loader = MnistLoader(
    "test", batch_size=batch_size, tensor_format="flat", shuffle=False
)

# --- Aim Tracking Setup ---

if TRACKING_ENABLED:
    tracking_config = TrackingConfig(
        experiment_name="mnist_pcn_tracking",
        run_name=f"5layer_lr0.001_infer{structure.config['inference'].config['infer_steps']}",
        track_energy=True,
        track_accuracy=True,
        track_weight_distributions=True,
        track_state_distributions=True,
        nodes_to_track=["h1", "h2", "h3", "class"],
        tracking_every_n_batches=50,
        state_tracking_every_n_infer_steps=5,
    )

    tracker = AimExperimentTracker(config=tracking_config)

    tracker.log_hyperparams(
        {
            "model_config": {
                "num_layers": len(structure.nodes),
                "layer_sizes": [
                    structure.nodes[n].node_info.shape for n in structure.node_order
                ],
                "activation": "sigmoid/softmax",
                "energy_type": "gaussian/cross_entropy",
            },
            "train_config": train_config,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
        }
    )

    tracker.log_graph_structure(structure)
else:
    tracker = None

# --- Training ---

opt_state = optimizer.init(params)

jit_train_step = jax.jit(
    lambda p, o, b, k: train_step_with_history(
        p,
        o,
        b,
        structure,
        optimizer,
        k,
        collect_every=INFERENCE_COLLECT_EVERY,
    )
)

print(f"Training for {num_epochs} epochs (JIT compilation on first batch)...\n")

best_accuracy = 0.0
training_history = []
global_step = 0

num_batches = len(train_loader)
all_rng_keys = jax.random.split(train_key, num_epochs * num_batches)
all_rng_keys = all_rng_keys.reshape((num_epochs, num_batches, 2))

for epoch in range(num_epochs):
    epoch_start = time.time()
    epoch_energies = []

    for batch_idx, (x, y) in enumerate(train_loader):
        batch = {"x": jnp.array(x), "y": y}

        energy = 0
        params, opt_state, energy, final_state, stacked_history = jit_train_step(
            params, opt_state, batch, all_rng_keys[epoch, batch_idx]
        )

        inference_history = unstack_inference_history(
            stacked_history, collect_every=INFERENCE_COLLECT_EVERY
        )

        # train_step_with_history returns the per-sample internal energy
        # (graph_energy over in_degree>0 nodes / batch_size) — no further
        # normalization needed.
        normalized_energy = float(energy)
        epoch_energies.append(normalized_energy)

        if tracker is not None:
            tracker.track_batch_energy(normalized_energy, epoch=epoch, batch=batch_idx)
            tracker.track_batch_energy_per_node(
                final_state, structure, epoch=epoch, batch=batch_idx
            )

            if batch_idx % tracker.config.tracking_every_n_batches == 0:
                tracker.track_state(
                    final_state, epoch=epoch, batch=batch_idx, infer_step=0
                )

            if batch_idx % tracker.config.tracking_every_n_batches == 0:
                for step_idx, step_metrics in enumerate(inference_history):
                    for node_name, metrics in step_metrics.items():
                        if node_name in tracker.config.nodes_to_track:
                            tracker._run.track(
                                metrics["energy"],
                                name="inference_energy",
                                step=step_idx * INFERENCE_COLLECT_EVERY,
                                context={
                                    "node": node_name,
                                    "epoch": epoch,
                                    "batch": batch_idx,
                                },
                            )
                            tracker._run.track(
                                metrics["latent_grad_norm"],
                                name="inference_grad_norm",
                                step=step_idx * INFERENCE_COLLECT_EVERY,
                                context={
                                    "node": node_name,
                                    "epoch": epoch,
                                    "batch": batch_idx,
                                },
                            )

        global_step += 1

        n_batch_update = 100
        if (batch_idx + 1) % n_batch_update == 0:
            avg_energy = sum(epoch_energies[-n_batch_update:]) / len(
                epoch_energies[-n_batch_update:]
            )
            convergence = summarize_inference_convergence(inference_history)
            h1_final = convergence.get("h1", {}).get("final_energy", 0)
            print(
                f"  Epoch {epoch+1}/{num_epochs}, "
                f"Batch {batch_idx+1}/{len(train_loader)}, "
                f"energy: {avg_energy:.4f}, "
                f"h1 Energy: {h1_final:.4f}"
            )

    epoch_time = time.time() - epoch_start
    avg_energy = sum(epoch_energies) / len(epoch_energies)

    if tracker is not None:
        tracker.track_weight_distributions(params, structure, epoch=epoch, batch=0)

    epoch_eval_key, eval_key = jax.random.split(eval_key)
    metrics = evaluate(params, structure, test_loader, train_config, epoch_eval_key)
    accuracy = metrics["accuracy"] * 100

    if tracker is not None:
        tracker.track_epoch_metrics(
            {"energy": avg_energy, "accuracy": accuracy / 100},
            epoch=epoch,
            subset="val",
        )

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = params
        print(f"  * New best accuracy: {accuracy:.2f}%")

    print(
        f"  Epoch {epoch+1}/{num_epochs} - "
        f"energy: {avg_energy:.4f}, "
        f"Accuracy: {accuracy:.2f}%, "
        f"Time: {epoch_time:.1f}s"
    )

    training_history.append(
        {
            "epoch": epoch + 1,
            "energy": avg_energy,
            "accuracy": accuracy,
            "time": epoch_time,
        }
    )

# --- Results ---

print(f"\nBest accuracy:  {best_accuracy:.2f}%")
print(f"Final accuracy: {training_history[-1]['accuracy']:.2f}%")
print(f"Total time:     {sum(h['time'] for h in training_history):.1f}s")

if tracker is not None:
    tracker.track_epoch_metrics(
        {"final_best_accuracy": best_accuracy / 100},
        epoch=num_epochs - 1,
        subset="final",
    )
    tracker.close()
    print("\nRun 'aim up' to view the dashboard.")
