"""
Predictive Coding Network — Advanced MNIST
===========================================

Custom training loop with optimizer selection and progress monitoring.

Architecture::

    pixels(784) ──→ h1(256) ──→ h2(128) ──→ h3(64) ──→ class(10)
     Identity       Sigmoid     Sigmoid     Sigmoid     Sigmoid

Usage:
    PYTHONPATH=. python examples/mnist_advanced.py --optimizer adamw
    PYTHONPATH=. python examples/mnist_advanced.py --optimizer sgd --num_epochs 2
    FABRICPC_OPTIMIZER=ngd_diag PYTHONPATH=. python examples/mnist_advanced.py
"""

import os
import argparse
import jax
import jax.numpy as jnp
import time

from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params, FeedforwardStateInit
from fabricpc.core.activations import IdentityActivation, SigmoidActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.inference import InferenceSGD
import optax
from fabricpc.training import evaluate, make_train_step
from fabricpc.training.natural_gradients import (
    scale_by_natural_gradient_diag,
    scale_by_natural_gradient_layerwise,
)
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc import setup_jax

setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")

# --- Network ---

pixels = Linear(shape=(784,), activation=IdentityActivation(), name="pixels")
h1 = Linear(
    shape=(256,),
    activation=SigmoidActivation(),
    energy=GaussianEnergy(precision=1.0),
    weight_init=NormalInitializer(mean=0.0, std=0.05),
    name="h1",
)
h2 = Linear(
    shape=(128,),
    activation=SigmoidActivation(),
    energy=GaussianEnergy(precision=1.0),
    weight_init=NormalInitializer(mean=0.0, std=0.05),
    name="h2",
)
h3 = Linear(
    shape=(64,),
    activation=SigmoidActivation(),
    energy=GaussianEnergy(precision=1.0),
    weight_init=NormalInitializer(mean=0.0, std=0.05),
    name="h3",
)
class_node = Linear(
    shape=(10,),
    activation=SigmoidActivation(),
    energy=GaussianEnergy(precision=1.0),
    weight_init=NormalInitializer(mean=0.0, std=0.05),
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
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)

# --- Hyperparameters ---

train_config = {}
batch_size = 200
num_epochs = 10

# --- Optimizer Presets ---

OPTIMIZER_PRESETS = {
    "adam": lambda: optax.chain(optax.adam(0.001)),
    "adamw": lambda: optax.adamw(0.001, weight_decay=0.1),
    # The trainer hands optax mean gradients per prediction (summed gradients
    # / N, N = batch_size = 200 here). Coupled L2 with SGD applies
    # lr * (g + wd * theta); the former summed-gradient constants lr = 0.01,
    # wd = 0.1 rescale exactly to lr * N = 2.0 and wd / N = 5e-4, and the
    # momentum trace is linear, so the trajectory is unchanged.
    "sgd": lambda: optax.chain(
        optax.add_decayed_weights(5e-4), optax.sgd(2.0, momentum=0.9)
    ),
    # Natural-gradient presets. The transforms' Fisher is the squared mean
    # gradient (see fabricpc.training.natural_gradients): where the damping
    # dominates it the update is SGD with rate scale / damping, elsewhere about
    # 1 / g. At the default damping (1e-8) 95% of the Fisher entries sit below
    # it at step 1 and 99.7% after one epoch, so these presets run as SGD with
    # rate 60 (diag) and 200 (layerwise) on almost every weight. Swept on this
    # graph over damping 1e-8..1e-4 and scale / damping 20..2000 at 10 epochs:
    # ngd_diag 16.27%, ngd_layerwise 10.28% (chance) (adamw 97%);
    # no setting did better. Issue 68 tracks the estimator fix.
    "ngd_diag": lambda: optax.chain(
        optax.add_decayed_weights(5e-4),
        scale_by_natural_gradient_diag(),
        optax.scale(-6e-7),
    ),
    "ngd_layerwise": lambda: optax.chain(
        optax.add_decayed_weights(5e-4),
        scale_by_natural_gradient_layerwise(),
        optax.scale(-2e-6),
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse CLI args. Env fallback: FABRICPC_OPTIMIZER."""
    default_optimizer = os.environ.get("FABRICPC_OPTIMIZER", "adamw")
    parser = argparse.ArgumentParser(description="FabricPC MNIST demo (advanced)")
    parser.add_argument(
        "--optimizer",
        default=default_optimizer,
        help=(
            "Optimizer preset to use. "
            f"Choices: {', '.join(OPTIMIZER_PRESETS.keys())}. "
            "CLI flag overrides FABRICPC_OPTIMIZER."
        ),
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=num_epochs,
        help=f"Training epochs (default: {num_epochs})",
    )
    args = parser.parse_args()
    if args.optimizer.lower() not in OPTIMIZER_PRESETS:
        valid = ", ".join(OPTIMIZER_PRESETS.keys())
        parser.error(f"unknown optimizer '{args.optimizer}'. valid choices: {valid}")
    return args


def get_optimizer(name: str) -> optax.GradientTransformation:
    """Return optimizer preset by name."""
    key = name.lower()
    if key not in OPTIMIZER_PRESETS:
        valid = ", ".join(OPTIMIZER_PRESETS.keys())
        raise ValueError(f"unknown optimizer '{name}'. valid choices: {valid}")
    return OPTIMIZER_PRESETS[key]()


# --- Train & Evaluate ---

if __name__ == "__main__":
    args = parse_args()
    optimizer = get_optimizer(args.optimizer)
    num_epochs = args.num_epochs

    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    params = initialize_params(structure, graph_key)
    num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))

    print(
        f"\n{len(structure.nodes)} nodes, {len(structure.edges)} edges, "
        f"{num_params:,} parameters"
    )
    print(f"Optimizer: {args.optimizer}")

    train_loader = MnistLoader(
        "train", batch_size=batch_size, tensor_format="flat", shuffle=True, seed=42
    )
    test_loader = MnistLoader(
        "test", batch_size=batch_size, tensor_format="flat", shuffle=False
    )

    opt_state = optimizer.init(params)

    step = make_train_step(structure, optimizer)

    print(f"\nTraining for {num_epochs} epochs (JIT compilation on first batch)...\n")

    best_accuracy = 0.0
    training_history = []

    num_batches = len(train_loader)
    all_rng_keys = jax.random.split(train_key, num_epochs * num_batches)
    all_rng_keys = all_rng_keys.reshape((num_epochs, num_batches, 2))

    for epoch in range(num_epochs):
        epoch_start = time.time()
        epoch_energies = []

        for batch_idx, (x, y) in enumerate(train_loader):
            batch = {"x": jnp.array(x), "y": y}

            params, opt_state, train_metrics, _ = step(
                params, opt_state, batch, all_rng_keys[epoch, batch_idx]
            )
            epoch_energies.append(float(train_metrics["energy"]))

            if (batch_idx + 1) % 100 == 0:
                avg_energy = sum(epoch_energies[-100:]) / len(epoch_energies[-100:])
                print(
                    f"  Epoch {epoch+1}/{num_epochs}, "
                    f"Batch {batch_idx+1}/{num_batches}, "
                    f"energy: {avg_energy:.4f}"
                )

        epoch_time = time.time() - epoch_start
        avg_energy = sum(epoch_energies) / len(epoch_energies)

        epoch_eval_key, eval_key = jax.random.split(eval_key)
        metrics = evaluate(params, structure, test_loader, train_config, epoch_eval_key)
        accuracy = metrics["accuracy"] * 100

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params

        print(
            f"  Epoch {epoch+1}/{num_epochs} - "
            f"energy: {avg_energy:.4f}, Accuracy: {accuracy:.2f}%, "
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

    print(f"\n{'Epoch':>5} | {'Energy':>7} | {'Accuracy':>8} | {'Time':>5}")
    print(f"{'-----':>5}-+-{'-------':>7}-+-{'--------':>8}-+-{'-----':>5}")
    for h in training_history:
        print(
            f"{h['epoch']:5d} | {h['energy']:7.4f} | {h['accuracy']:7.2f}% | {h['time']:4.1f}s"
        )
