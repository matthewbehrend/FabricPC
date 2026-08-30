"""
Predictive Coding Network — Convolutional MNIST Demo
====================================================

Trains a predictive coding Convolutional Neural Network on MNIST using
the unified ConvNode and MaxPool.

Architecture::

    pixels (28x28x1)
      │
    ConvNode  (3x3, stride=1, SAME, LeakyReLU)  ──→ conv1  (28x28x32)
      │
    MaxPool (2x2, stride=2)                  ──→ pool1  (14x14x32)
      │
    ConvNode  (3x3, stride=1, SAME, LeakyReLU)  ──→ conv2  (14x14x64)
      │
    MaxPool (2x2, stride=2)                  ──→ pool2  (7x7x64)
      │
    ConvNode  (3x3, stride=1, SAME, LeakyReLU)  ──→ conv3  (7x7x128)
      │
    Linear (flatten_input=True, Softmax+CE)      ──→ class  (10)
"""

import time
import jax
import optax

from fabricpc.nodes import ConvNode, MaxPool, Linear, IdentityNode
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import LeakyReLUActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import KaimingInitializer, XavierInitializer
from fabricpc.training import train, evaluate
from fabricpc.utils.data.dataloader import MnistLoader

jax.config.update("jax_default_prng_impl", "threefry2x32")

# --- Network Definition ---

pixels = IdentityNode(shape=(28, 28, 1), name="pixels")

conv1 = ConvNode(
    shape=(28, 28, 32),
    kernel_size=(3, 3),
    stride=(1, 1),
    padding="SAME",
    activation=LeakyReLUActivation(alpha=0.01),
    energy=GaussianEnergy(),
    weight_init=KaimingInitializer(nonlinearity="leaky_relu", a=0.01),
    name="conv1",
)

pool1 = MaxPool(
    shape=(14, 14, 32),
    window_shape=(2, 2),
    stride=(2, 2),
    padding="VALID",
    name="pool1",
)

conv2 = ConvNode(
    shape=(14, 14, 64),
    kernel_size=(3, 3),
    stride=(1, 1),
    padding="SAME",
    activation=LeakyReLUActivation(alpha=0.01),
    energy=GaussianEnergy(),
    weight_init=KaimingInitializer(nonlinearity="leaky_relu", a=0.01),
    name="conv2",
)

pool2 = MaxPool(
    shape=(7, 7, 64),
    window_shape=(2, 2),
    stride=(2, 2),
    padding="VALID",
    name="pool2",
)

conv3 = ConvNode(
    shape=(7, 7, 128),
    kernel_size=(3, 3),
    stride=(1, 1),
    padding="SAME",
    activation=LeakyReLUActivation(alpha=0.01),
    energy=GaussianEnergy(),
    weight_init=KaimingInitializer(nonlinearity="leaky_relu", a=0.01),
    name="conv3",
)

output = Linear(
    shape=(10,),
    activation=SoftmaxActivation(),
    energy=CrossEntropyEnergy(),
    flatten_input=True,
    weight_init=XavierInitializer(),
    name="class",
)

structure = graph(
    nodes=[pixels, conv1, pool1, conv2, pool2, conv3, output],
    edges=[
        Edge(source=pixels, target=conv1.slot("in")),
        Edge(source=conv1, target=pool1.slot("in")),
        Edge(source=pool1, target=conv2.slot("in")),
        Edge(source=conv2, target=pool2.slot("in")),
        Edge(source=pool2, target=conv3.slot("in")),
        Edge(source=conv3, target=output.slot("in")),
    ],
    task_map=TaskMap(x=pixels, y=output),
    inference=InferenceSGD(eta_infer=0.01, infer_steps=100),
)

# --- Hyperparameters ---

train_config = {"num_epochs": 20}
batch_size = 200
optimizer = optax.adamw(0.001, weight_decay=0.01)

# --- Train & Evaluate ---

if __name__ == "__main__":
    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    params = initialize_params(structure, graph_key)

    train_loader = MnistLoader(
        "train", batch_size=batch_size, tensor_format="nhwc", shuffle=True, seed=42
    )
    test_loader = MnistLoader(
        "test", batch_size=batch_size, tensor_format="nhwc", shuffle=False
    )

    print(
        f"\n{len(structure.nodes)} nodes, {len(structure.edges)} edges, "
        f"{sum(p.size for p in jax.tree_util.tree_leaves(params)):,} parameters"
    )

    print("\nTraining (JIT compilation on first batch)...")
    start_time = time.time()
    result = train(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        verbose=True,
    )
    trained_params = result.params
    elapsed = time.time() - start_time
    print(f"Avg training time: {elapsed / train_config['num_epochs']:.2f}s per epoch")

    print("\nEvaluating...")
    metrics = evaluate(trained_params, structure, test_loader, train_config, eval_key)
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
