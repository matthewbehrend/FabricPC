"""
Predictive Coding Network — Multi-GPU MNIST
============================================

Data-parallel training across multiple GPUs using jit with NamedSharding:
a one-axis ("data",) mesh spans all devices, each batch is sharded along
that axis, and parameters are replicated on every device.
Works with 1 GPU (a size-1 mesh) but benefits from 2+.

Architecture (parameters replicated across N GPUs)::

    pixels(784) ──→ hidden1(256) ──→ hidden2(64) ──→ class(10)
     Identity        Sigmoid          Sigmoid        Softmax+CE
"""

import jax
import time

from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    SigmoidActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
import optax
from fabricpc.training import train, evaluate
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc import setup_jax

setup_jax()

# --- Network ---

pixels = Linear(shape=(784,), activation=IdentityActivation(), name="pixels")
hidden1 = Linear(
    shape=(256,),
    activation=SigmoidActivation(),
    name="hidden1",
)
hidden2 = Linear(
    shape=(64,),
    activation=SigmoidActivation(),
    name="hidden2",
)
class_node = Linear(
    shape=(10,),
    activation=SoftmaxActivation(),
    energy=CrossEntropyEnergy(),
    name="class",
)

structure = graph(
    nodes=[pixels, hidden1, hidden2, class_node],
    edges=[
        Edge(source=pixels, target=hidden1.slot("in")),
        Edge(source=hidden1, target=hidden2.slot("in")),
        Edge(source=hidden2, target=class_node.slot("in")),
    ],
    task_map=TaskMap(x=pixels, y=class_node),
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)

# --- Hyperparameters ---

# Weight gradients are the global batch sum, and the batch below scales with
# the device count, so the gradient magnitude grows with N devices. adamw
# normalizes the update by the gradient's second moment, which cancels that
# scale (up to eps and the decoupled weight decay); a scale-sensitive
# optimizer (SGD) would need lr / N here. The 0.4 pmap trainer applied a
# device mean instead — see the 0.5.0 CHANGELOG behavior notes.
optimizer = optax.adamw(0.001, weight_decay=0.001)
train_config = {"num_epochs": 20}

# --- Train & Evaluate ---

master_rng_key = jax.random.PRNGKey(0)
graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

n_devices = jax.device_count()
print(f"Devices: {n_devices} ({[d.device_kind for d in jax.devices()]})")

mesh = jax.make_mesh((jax.device_count(),), ("data",))

params = initialize_params(structure, graph_key)
num_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
print(
    f"{len(structure.nodes)} nodes, {len(structure.edges)} edges, {num_params:,} parameters"
)

# Batch size scales with device count
batch_size = 200 * n_devices
print(f"Batch size: {batch_size} ({batch_size // n_devices} per device)")

train_loader = MnistLoader(
    "train", batch_size=batch_size, tensor_format="flat", shuffle=True, seed=42
)
test_loader = MnistLoader(
    "test", batch_size=batch_size, tensor_format="flat", shuffle=False
)

print(f"\nTraining on {n_devices} device(s) (JIT compilation on first batch)...\n")

start_time = time.time()
result = train(
    params=params,
    structure=structure,
    train_loader=train_loader,
    optimizer=optimizer,
    config=train_config,
    rng_key=train_key,
    mesh=mesh,
    verbose=True,
)
trained_params = result.params
training_time = time.time() - start_time

throughput = train_loader.num_examples * train_config["num_epochs"] / training_time
print(
    f"\nTotal: {training_time:.1f}s, {training_time / train_config['num_epochs']:.1f}s/epoch, "
    f"{throughput:.0f} samples/sec"
)

metrics = evaluate(
    trained_params, structure, test_loader, train_config, eval_key, mesh=mesh
)
print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
