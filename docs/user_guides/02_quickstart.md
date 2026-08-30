# Quickstart

Train a predictive coding network on MNIST in under 5 minutes.

## The 5-Step Pattern

Every FabricPC workflow follows this pattern:

1. **Define Nodes** — Create nodes with shapes, activations, and energy functionals
2. **Build Graph** — Connect nodes with edges and specify the task mapping
3. **Initialize Parameters** — Create weight and bias parameters
4. **Train** — Run inference and local weight updates
5. **Evaluate** — Test the trained network

## Complete MNIST Example

```python
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
from fabricpc import setup_jax

setup_jax()  # auto-detects the backend; pass platform="cuda" to require GPU
jax.config.update("jax_default_prng_impl", "threefry2x32")

# --- Step 1: Define Nodes ---
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
    energy=CrossEntropyEnergy(),  # Use cross-entropy for classification
    name="class",
    weight_init=XavierInitializer(),
)

# --- Step 2: Build Graph ---
structure = graph(
    nodes=[pixels, hidden1, hidden2, output],
    edges=[
        Edge(source=pixels, target=hidden1.slot("in")),
        Edge(source=hidden1, target=hidden2.slot("in")),
        Edge(source=hidden2, target=output.slot("in")),
    ],
    task_map=TaskMap(x=pixels, y=output),  # x=input, y=target
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)

# --- Step 3: Initialize Parameters ---
master_rng_key = jax.random.PRNGKey(0)
graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)
params = initialize_params(structure, graph_key)

# --- Step 4: Train ---
train_config = {"num_epochs": 20}
batch_size = 200
optimizer = optax.adamw(0.001, weight_decay=0.1)

train_loader = MnistLoader(
    "train", batch_size=batch_size, tensor_format="flat", shuffle=True, seed=42
)
test_loader = MnistLoader(
    "test", batch_size=batch_size, tensor_format="flat", shuffle=False
)

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

# --- Step 5: Evaluate ---
metrics = evaluate(trained_params, structure, test_loader, train_config, eval_key)
print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")
```

## What Just Happened?

### Inner Loop: Inference

For each training batch, the network runs **inference** to minimize energy:

1. Initialize latent states `z_latent` at all nodes (clamped to data at input/output)
2. Compute predictions `z_mu` from incoming connections
3. Compute prediction errors: `error = z_latent - z_mu`
4. Update latent states to reduce energy: `z_latent -= eta_infer * dE/dz`
5. Repeat steps 2-4 for `infer_steps` iterations

This is implemented by `InferenceSGD` with 20 inference steps per batch.

### Outer Loop: Learning

After inference converges, the network updates weights using **local learning rules**:

1. Compute local weight gradients at each node from converged states
2. Apply optimizer updates (AdamW in this example)
3. Move to next batch

This is the predictive coding alternative to backpropagation: gradients are computed locally at each node from its prediction error, not via a global backward pass.

### Energy Functional

The default energy is Gaussian: `E = 0.5 * ||z_latent - z_mu||^2`. The output node uses `CrossEntropyEnergy` instead, which is appropriate for classification targets.

## Next Steps

- [How Predictive Coding Works](03_how_predictive_coding_works.md) — Understand the theory behind the code
- [Building Models](04_building_models.md) — Learn about all node types and graph topologies
- [Initialization and Scaling](05_initialization_and_scaling.md) — Weight init strategies and muPC scaling for deep networks
- [Training and Evaluation](08_training_and_evaluation.md) — Callbacks, multi-GPU, and advanced training patterns
