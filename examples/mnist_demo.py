"""
Predictive Coding Network — MNIST Demo
=======================================

Train a small predictive coding network on MNIST using the object API.

Architecture::

    pixels(784) ──→ hidden1(256) ──→ hidden2(64) ──→ class(10)
     Identity        Sigmoid          Sigmoid        Softmax+CE

For optimizer selection and advanced controls, see mnist_advanced.py.

Results:
Avg training time: 1.37s per epoch
20 epochs
Test Accuracy: 98.14%
4 nodes, 3 edges, 218,058 parameters
"""

import jax
from fabricpc.nodes import Linear, IdentityNode
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import SigmoidActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import XavierInitializer
import optax
from fabricpc.training import train, evaluate
from fabricpc.utils.data.dataloader import MnistLoader
import time
from fabricpc import setup_jax

setup_jax()  # options: "cpu", "cuda", "tpu"
jax.config.update("jax_default_prng_impl", "threefry2x32")

# --- Network ---

pixels = IdentityNode(shape=(784,), name="pixels")
hidden1 = Linear(
    shape=(256,),
    activation=SigmoidActivation(),
    name="hidden1",
    weight_init=XavierInitializer(),
)
hidden2 = Linear(
    shape=(64,),
    activation=SigmoidActivation(),
    name="hidden2",
    weight_init=XavierInitializer(),
)
output = Linear(
    shape=(10,),
    activation=SoftmaxActivation(),
    energy=CrossEntropyEnergy(),
    name="class",
    weight_init=XavierInitializer(),
)

# x= and y= tell the trainer which nodes are inputs and targets
structure = graph(
    nodes=[pixels, hidden1, hidden2, output],
    edges=[
        Edge(source=pixels, target=hidden1.slot("in")),
        Edge(source=hidden1, target=hidden2.slot("in")),
        Edge(source=hidden2, target=output.slot("in")),
    ],
    task_map=TaskMap(x=pixels, y=output),
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)

# --- Hyperparameters ---

train_config = {"num_epochs": 20}
batch_size = 200
optimizer = optax.adamw(0.001, weight_decay=0.1)

# --- Train & Evaluate ---

if __name__ == "__main__":
    master_rng_key = jax.random.PRNGKey(0)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    params = initialize_params(structure, graph_key)

    train_loader = MnistLoader(
        "train", batch_size=batch_size, tensor_format="flat", shuffle=True, seed=42
    )
    test_loader = MnistLoader(
        "test", batch_size=batch_size, tensor_format="flat", shuffle=False
    )

    print("\nTraining (JIT compilation on first batch)...")
    start_time = time.time()
    result = train(
        params,
        structure,
        train_loader,
        optimizer,
        train_config,
        train_key,
        verbose=True,
    )
    trained_params = result.params
    elapsed = time.time() - start_time
    print(f"Avg training time: {elapsed / train_config['num_epochs']:.2f}s per epoch")

    print("\nEvaluating...")
    metrics = evaluate(trained_params, structure, test_loader, train_config, eval_key)
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")

    print(
        f"\n{len(structure.nodes)} nodes, {len(structure.edges)} edges, "
        f"{sum(p.size for p in jax.tree_util.tree_leaves(params)):,} parameters"
    )
