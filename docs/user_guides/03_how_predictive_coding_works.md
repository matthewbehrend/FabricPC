# How Predictive Coding Works

This guide maps predictive coding theory to FabricPC code. If you're new to predictive coding, start here to understand the core concepts.

## The Brain's Prediction Engine

The brain is constantly generating predictions about incoming sensory data. Rather than passively receiving information, the cortex actively predicts what it expects to see, hear, or feel. When predictions don't match reality, prediction errors propagate through the hierarchy, updating the brain's internal model of the world.

Predictive coding formalizes this idea as a principle for neural computation. Each layer in a hierarchical network maintains a hypothesis about the state of the layer below. When the hypothesis doesn't match the actual state, the mismatch (prediction error) drives both inference (updating the hypothesis) and learning (updating the connections). This framework provides a biologically plausible account of perception and learning, grounded in principles of Bayesian inference and energy minimization.

FabricPC implements this framework as an iterative optimization process over a graph of nodes. Unlike traditional neural networks that compute outputs in a single forward pass, predictive coding networks iteratively refine their internal states until they reach a minimum-energy configuration that best explains the data.

## Bilevel Optimization

Predictive coding operates as a **bilevel optimization** process with two nested loops:

```
 ┌─ Outer Loop: Learning ─────────────────────────────────────┐
 │                                                            │
 │   ┌─ Inner Loop: Inference ──────────────────────┐         │
 │   │                                              │         │
 │   │   clamp x, y                                 │         │
 │   │       ↓                                      │         │
 │   │                                              │         │
 │   │   z ← z - η_infer · ∂E/∂z   (repeat N)       │         │
 │   │                                              │         │
 │   │       ↓                                      │         │
 │   │   converged z*                               │         │
 │   │                                              │         │
 │   └──────────────────────────────────────────────┘         │
 │       ↓                                                    │
 │                                                            │
 │   ∂E/∂W computed locally at each node from z*              │
 │       ↓                                                    │
 │                                                            │ 
 │   W ← W - η_learn · optimizer(∂E/∂W)                       │
 │                                                            │
 └────────────────────────────────────────────────────────────┘
```

### Inner Loop: Inference

The **inference loop** minimizes energy by updating latent states `z_latent` while keeping weights fixed. This is gradient descent on the energy landscape:

```
z_latent -= eta_infer * dE/dz
```

where `eta_infer` is the inference rate (distinct from the weight learning rate). The network runs this update for a fixed number of steps (e.g., 20) until the states converge to a low-energy configuration.

In FabricPC, inference is controlled by an inference algorithm like `InferenceSGD`:

```python
from fabricpc.core.inference import InferenceSGD

inference = InferenceSGD(eta_infer=0.05, infer_steps=20)
```

During training, the input node is clamped to `x` (the batch data) and the output node is clamped to `y` (the target labels). These clamped nodes provide boundary conditions, and the inference loop adjusts all unclamped latent states to minimize the total network energy given these constraints.

The state-based update above moves the error signal one hop per step, so on deep graphs it needs many steps for the output clamp to reach early layers. `EPCInference` minimizes the same energy in error coordinates: the prediction errors are the relaxed variables, the latents are derived by one forward pass (`z_latent = z_mu + error`), and one global reverse pass per step delivers the loss signal to every layer at once — a few steps replace hundreds on deep DAGs, at backprop-scale memory per step. When `eta_infer × infer_steps × λ_max` is small, λ_max the top eigenvalue of the energy's Hessian in error coordinates, the errors stay at the backprop activation gradient and ePC's weight gradients are backprop's with rescaled magnitude (Goemaere et al., Theorem C.9); the inference guide's backprop-regime paragraph gives the rule and the measured resnet18 numbers. The two parameterizations share the same equilibria on DAGs, and `InferenceSchedule` composes them per weight update (e.g. a few ePC steps, then sPC refinement). On cyclic graphs — built with `graph(..., unroll=U)` — the distinction is what each solver minimizes: the state-based solvers ignore `structure.schedule` and relax the exact graph energy, while ePC minimizes its unrolled approximation at degree U. See the [Inference Algorithms API](12_api_inference.md).

### Outer Loop: Learning

The **learning loop** updates weights using gradients computed from the converged states. After inference reaches equilibrium, each node computes local weight gradients based on its prediction error and the activity of its inputs. This is a Hebbian-like learning rule:

```
dE/dW ~ error * input_activity
```

In FabricPC, local gradients are computed by `compute_local_weight_gradients` (batch-summed), divided once by the batch's prediction count in the trainer's `pc_weight_gradients`, and then an Optax optimizer (e.g., Adam, SGD) applies the updates:

```python
import optax
from fabricpc.training import train

optimizer = optax.adamw(0.001, weight_decay=0.1)
result = train(
    params=params,
    structure=structure,
    train_loader=train_loader,
    optimizer=optimizer,
    config={"num_epochs": 20},
    rng_key=train_key,
)
trained_params = result.params
```

The key difference from backpropagation: weight gradients are computed **locally** at each node from local information (its own error and inputs), not via a global backward pass through the network.

## Energy Minimization

Energy is a scalar value that measures how well the network's internal model explains the data. Each node contributes energy based on its **prediction error**:

```
error = z_latent - z_mu
```

where:
- `z_latent` is the node's current hypothesis about its state
- `z_mu` is the prediction from upstream nodes

The default energy functional is **Gaussian**:

```
E = 0.5 * ||z_latent - z_mu||^2
```

This penalizes squared prediction error. The network seeks to minimize total energy across all nodes by adjusting latent states during inference and adjusting weights during learning.

### Clamped Nodes

During training, input and output nodes are **clamped** to observed data:
- Input node: `z_latent = x` (fixed to batch data)
- Output node: `z_latent = y` (fixed to target labels)

Clamped nodes don't update their latent states during inference. They serve as boundary conditions that constrain the energy landscape. The network must find a configuration of unclamped (hidden) latent states that reconciles these boundary conditions while minimizing energy.

During evaluation, only the input is clamped. The output node is free to settle to whatever prediction minimizes energy given the input.

### Custom Energy Functionals

For classification tasks, the output node typically uses **cross-entropy energy** instead of Gaussian:

```python
from fabricpc.core.energy import CrossEntropyEnergy

output = Linear(
    shape=(10,),
    activation=SoftmaxActivation(),
    energy=CrossEntropyEnergy(),  # Use cross-entropy instead of Gaussian
    name="output",
)
```

This treats the output as a probability distribution and measures the KL divergence between the target and prediction, which is more appropriate for categorical targets than squared error.

## Mapping to FabricPC

Here's how predictive coding concepts map to FabricPC types:

| PC Concept | FabricPC Type | Description |
|------------|---------------|-------------|
| Latent state `z` | `NodeState.z_latent` | The node's current hypothesis about its state |
| Prediction `mu` | `NodeState.z_mu` | The prediction from incoming connections |
| Prediction error | `NodeState.error` | `z_latent - z_mu` |
| Energy | `NodeState.energy` | Computed by the node's `EnergyFunctional` |
| Latent gradient | `NodeState.latent_grad` | Gradient accumulator `dE/dz_latent` consumed by the inference update |
| Inference update | `InferenceBase` | `z -= eta * dE/dz` (updates latent states) |
| Weight gradient | `compute_local_weight_gradients` | Local Hebbian gradient `dE/dW` |
| Clamped node | `clamps` dict | Node whose `z_latent` is fixed to observed data |
| Inference steps | `InferenceSGD.infer_steps` | Number of inner-loop iterations |
| Inference rate | `InferenceSGD.eta_infer` | Step size for latent state updates |
| Weight learning rate | `optax.adam(lr)` | Step size for weight updates |

### Example: Accessing States

After running inference, you can inspect the converged states:

```python
from fabricpc.core.inference import run_inference
from fabricpc.graph_initialization.state_initializer import initialize_graph_state

# Initialize states
state = initialize_graph_state(structure, batch_size=32, rng_key=key, clamps=clamps, params=params)

# Run inference to convergence
final_state = run_inference(params, state, clamps, structure)

# Access node states
hidden_latent = final_state.nodes["hidden1"].z_latent  # Shape: (32, 256)
hidden_mu = final_state.nodes["hidden1"].z_mu
hidden_error = final_state.nodes["hidden1"].error
hidden_energy = final_state.nodes["hidden1"].energy  # Shape: (32,) - per-sample energy
```

## Comparison with Backpropagation

| Aspect | Predictive Coding | Backpropagation |
|--------|-------------------|-----------------|
| **Gradient computation** | Local (each node computes its own gradients) | Global (backward pass through entire network) |
| **Information flow** | Bidirectional (predictions flow down, errors flow up) | Unidirectional (forward then backward) |
| **Computation** | Iterative inference (multiple steps to convergence) | Single forward pass |
| **Learning rule** | Local Hebbian (based on prediction error and input activity) | Credit assignment via chain rule |
| **Network topology** | Arbitrary graphs (cycles, skip connections) | Typically acyclic (DAGs) |
| **Biological plausibility** | High (local learning, iterative dynamics) | Low (global error signals, weight transport problem) |

### When They Agree

Under certain conditions, predictive coding converges to the same solution as backpropagation:
- Feedforward topology (no recurrent connections)
- Linear nodes or small learning rates
- Inference fully converged before weight updates

FabricPC provides both modes on the same graph structure through `train`'s `algorithm` argument:
- `train(..., algorithm="pc")` — Predictive coding with local learning rules (the default)
- `train(..., algorithm="backprop")` — Standard backpropagation for comparison

This allows direct A/B testing of the two approaches on identical architectures.

## Why Predictive Coding?

### Advantages

- **Neuromorphic hardware compatibility** — Local learning rules map naturally to distributed hardware without global synchronization
- **Arbitrary topologies** — Handles recurrent connections, skip connections, and cycles without modification
- **Associative memory** — Nodes like `StorkeyHopfield` implement Hopfield networks for pattern completion and few-shot learning
- **Biological grounding** — Offers insights into cortical computation and learning
- **Novel plasticity rules** — Framework for exploring alternatives to backprop (e.g., attention-modulated learning, homeostatic plasticity)

### Trade-offs

- **Computational cost** — Iterative inference adds overhead compared to single forward pass
- **Hyperparameter complexity** — Requires tuning both inference parameters (eta_infer, infer_steps) and weight optimizer parameters
- **Convergence** — Inference may not fully converge in the allotted steps, especially in deep or recurrent networks

### When to Use Predictive Coding

Consider predictive coding if you're interested in:
- Biologically plausible learning algorithms
- Deploying to neuromorphic or distributed hardware
- Networks with recurrent or bidirectional connections
- Associative memory and pattern completion
- Research on alternatives to backpropagation

For standard supervised learning on feedforward architectures, backpropagation remains more efficient. FabricPC supports both paradigms, enabling exploration and comparison.

## Next Steps

- [Building Models](04_building_models.md) — Explore different node types and graph architectures
- [Initialization and Scaling](05_initialization_and_scaling.md) — Learn about muPC scaling for training deep predictive coding networks
- [Custom Nodes](06_custom_nodes.md) — Implement your own node types with custom energy functionals
- [Training and Evaluation](08_training_and_evaluation.md) — Advanced training techniques and callbacks
