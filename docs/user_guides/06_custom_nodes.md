# Writing Custom Nodes

FabricPC's built-in nodes cover many common use cases, but you may need custom nodes for specialized computations. This guide walks through creating your own node types.

## When to Create a Custom Node

Built-in nodes include:
- `Linear`: Fully-connected layers
- `IdentityNode`: Passthrough nodes
- `SkipConnection`, `LinearResidual`: Residual paths and residual blocks
- `ConvNode`: 1D/2D/3D convolution
- `MaxPool`, `AvgPool`: Windowed and global pooling
- `StorkeyHopfield`: Associative memory
- `TransformerBlock`: Multi-head attention and feedforward
- Transformer v2 (`EmbeddingNode`, `MhaResidualNode`, `LnMlp1Node`, `Mlp2ResidualNode`, `VocabProjectionNode`): decomposed transformer stages

Create a custom node when you need:
- Gating mechanisms (LSTM, GRU)
- Custom transfer functions
- Specialized domain-specific projections
- Any computation not covered by built-in nodes

## The NodeBase Contract

All nodes inherit from `NodeBase` and must implement three static methods:

1. **`get_slots()`**: Define input slots (the connection interface)
2. **`initialize_params()`**: Allocate and initialize weights and biases
3. **`predict()`**: Compute the prediction `z_mu` (plus optional `aux` intermediates)

Nodes with extra energy terms additionally override a fourth:

4. **`energy()`** (optional): Per-sample energy at `(state.z_latent, state.z_mu)`. The default scores the node's energy functional; override it to add terms (see [Custom Energy Terms](#custom-energy-terms)).

These methods are static because FabricPC uses a functional JAX-based design. Node instances hold configuration; the static methods define pure functional transformations.

`predict()` is where the node does its real work, and its body is intentionally unconstrained: how you turn inputs into a prediction is up to you — a matmul in `Linear`, an attention pipeline in `TransformerBlock`, an embedding lookup in `EmbeddingNode`. Everything around the prediction is **base-owned and not a node override point**:

- The error pair — `error = z_latent - z_mu` (`pair_error`) and its inverse `z_latent = z_mu + error` (`pair_latent`) — lives on `NodeBase`. State-based inference relaxes `z_latent` and derives `error`; the error-parameterized solver (`EPCInference`) relaxes `error` and derives `z_latent`. Both directions share one bijection, which is why nodes cannot override it.
- The assembly templates `forward()` (predict → pair → energy), `forward_with_aux()` (the same, surfacing `aux`), and `forward_from_error()` (the ePC derive direction) live on `NodeBase`. Source semantics (`in_degree == 0`) also live there — `predict()` is never called on a source node: in the sPC direction (`forward`/`forward_with_aux`) a source's `z_mu` mirrors its `z_latent` with zero error, while in the ePC direction (`forward_from_error`) an unclamped source derives `z_latent = z_mu + error` with `z_mu` held fixed.

## Step-by-Step: Conv2D Node

Let's build a 2D convolutional node from scratch, to illustrate the node contract. (FabricPC now ships a production `ConvNode` in `fabricpc.nodes`; this from-scratch version is for teaching, not for use.)

### Step 1: Define the Class and Constructor

```python
import jax.numpy as jnp
import numpy as np
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.activations import ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.initializers import NormalInitializer, initialize

class Conv2DNode(NodeBase):
    def __init__(
        self,
        shape,
        name,
        kernel_size,
        stride=(1, 1),
        padding="SAME",
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(),
        **kwargs
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            **kwargs
        )
```

Key points:
- Accept standard node parameters (`shape`, `name`, `activation`, etc.)
- Accept custom parameters (`kernel_size`, `stride`, `padding`)
- Pass everything to `super().__init__()` via `**kwargs`
- Custom parameters end up in `node_info.node_config` and are accessible in static methods
- Object-valued defaults (`activation`, `energy`, `weight_init`, `latent_init`) go **in the signature**, as above. These objects are immutable — construction stores only a read-only `config` mapping of validated immutable values (scalars and tuples; a list, dict, or array raises `TypeError`) and blocks further attribute writes — so the module-level singleton a signature default creates cannot leak state between nodes.
- `None` is not a "use the default" sentinel: `NodeBase.__init__` raises `TypeError` if `activation`, `energy`, or `latent_init` is not an instance of its base class. `weight_init=None` stays legal and means a weight-free node (pooling).
- Fill a default **in the body** (`param=None` + `if param is None:`) only when it is computed from another argument (e.g. a `stride` derived from the kernel's spatial rank).
- Never default a parameter to a `dict`/`list`/`set` literal; that shared object is mutated across calls. `ruff` B006 rejects it in CI.

### Step 2: Define Input Slots

```python
@staticmethod
def get_slots():
    return {"in": SlotSpec(name="in", is_multi_input=True)}
```

This defines a single input slot named `"in"` that accepts multiple incoming edges. The `is_multi_input=True` flag means contributions from different source nodes will be summed.

For nodes with multiple distinct inputs:

```python
@staticmethod
def get_slots():
    return {
        "in": SlotSpec(name="in", is_multi_input=True),
        "mask": SlotSpec(name="mask", is_multi_input=False),
    }
```

### Step 3: Report the Variance Factor (Optional)

For muPC scaling, override `get_variance_factor()` to report how your node's input transform changes input variance, before the activation. muPC scales each in-edge by `a = gain / sqrt(v * K_slot)`, so `v` is what the scale undoes.

```python
@staticmethod
def get_variance_factor(source_shape, config, weight_init):
    kernel_size = config.get("kernel_size", (1, 1))
    C_in = source_shape[-1]  # channels-last format
    return float(C_in * int(np.prod(kernel_size)))
```

For a matmul against unit-variance weights, `v` is the Kaiming fan_in — each output unit sums that many independent products. A 3x3 kernel with 16 input channels gives `v = 16 * 3 * 3 = 144`.

`v` is a float, not a dimension count, and may be **below 1**. A transform that *reduces* variance returns `v < 1`, and muPC amplifies instead of attenuating: `AvgPool` over `n` cells returns `1/n` so the edge scale becomes `sqrt(n)`. A weightless node that only sums its inputs returns `1.0`, leaving just the `K_slot` term.

The `weight_init` argument is the node's own initializer. Ignore it unless your node builds a weight matrix internally rather than one per in-edge — `StorkeyHopfield` uses it with `InitializerBase.element_variance` to derive the variance gain of its internally-initialized Hopfield matrix.

If you don't override this method, the default uses the flattened or last-axis source shape, which works for fully-connected layers but not for convolutions or reducing transforms.

### Step 4: Initialize Parameters

```python
@staticmethod
def initialize_params(key, node_shape, input_shapes, weight_init, config):
    """
    Initialize convolutional kernels and biases.

    Args:
        key: JAX random key
        node_shape: Output shape of this node (H, W, C_out)
        input_shapes: Dict mapping edge_key -> source_shape
        weight_init: Weight initializer
        config: Node configuration dict

    Returns:
        NodeParams(weights=weights_dict, biases=biases_dict)
    """
    kernel_size = config.get("kernel_size", (1, 1))
    C_out = node_shape[-1]  # output channels

    weights = {}
    biases = {}

    for edge_key, source_shape in input_shapes.items():
        C_in = source_shape[-1]  # input channels

        # Kernel shape: (kH, kW, C_in, C_out)
        kernel_shape = (*kernel_size, C_in, C_out)

        # Initialize kernel weights
        subkey, key = jax.random.split(key)
        fan_in = C_in * int(np.prod(kernel_size))
        fan_out = C_out * int(np.prod(kernel_size))

        weights[edge_key] = initialize(
            subkey,
            kernel_shape,
            weight_init,
            fan_in=fan_in,
            fan_out=fan_out,
        )

        # Bias shape: (1, 1, 1, C_out) for broadcasting
        biases[edge_key] = jnp.zeros((1, 1, 1, C_out))

    return NodeParams(weights=weights, biases=biases)
```

Key points:
- `input_shapes` is a dict keyed by edge identifiers (e.g., `"conv1->conv2:in"`)
- Each edge gets its own weight matrix and bias vector
- Use the `initialize()` helper function with the provided `weight_init` initializer
- Return `NodeParams` with dicts of weights and biases

### Step 5: Implement the Prediction

```python
@staticmethod
def predict(params, inputs, state, node_info):
    """
    Prediction: compute convolution, bias, and activation.

    Args:
        params: NodeParams with weights and biases dicts
        inputs: Dict mapping edge_key -> input_array
        state: Current NodeState
        node_info: NodeInfo with configuration

    Returns:
        Tuple of (z_mu, aux). aux is None here; see the aux pattern below.
    """
    # Extract config
    kernel_size = node_info.node_config.get("kernel_size", (1, 1))
    stride = node_info.node_config.get("stride", (1, 1))
    padding = node_info.node_config.get("padding", "SAME")
    activation = node_info.activation

    # Accumulate convolution outputs from all input edges
    pre_activation = None

    for edge_key, input_array in inputs.items():
        kernel = params.weights[edge_key]
        bias = params.biases[edge_key]

        # Perform convolution
        # JAX uses (batch, H, W, C) format
        conv_out = jax.lax.conv_general_dilated(
            lhs=input_array,
            rhs=kernel,
            window_strides=stride,
            padding=padding,
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )

        conv_out = conv_out + bias

        if pre_activation is None:
            pre_activation = conv_out
        else:
            pre_activation = pre_activation + conv_out

    # Apply activation function
    z_mu = type(activation).forward(pre_activation, activation.config)

    return z_mu, None
```

`predict()` is a **pure function**. It must have no side effects and must express its dependence on `params`, `inputs`, and `state` entirely through JAX operations, because the framework differentiates through it — `forward_and_latent_grads` with respect to inputs and `z_latent`, `forward_and_weight_grads` with respect to `params`, and `EPCInference`'s global energy gradient with respect to the relaxed errors. Side effects or Python-level control flow on traced values produce wrong gradients during inference and learning.

The contract:

1. **Return `z_mu`**: the node's prediction of its own latent, with shape `(batch,) + node_info.shape`. *How* is up to the node — a convolution here, a matmul in `Linear`, an attention pipeline in `TransformerBlock`.
2. **Return `aux`**: an arbitrary pytree of intermediates for `energy()`, or `None` if unused. See [the aux pattern](#the-aux-pattern) — aux must depend only on `params` and `inputs`.
3. **Do not compute the error or the energy.** The base templates apply `error = z_latent - z_mu` and call `energy()`; a node body that repeats them breaks the error-parameterized solver, which derives `z_latent` from the error instead.

`predict()` must not read `state.z_latent` values (shape/dtype reads like `state.z_latent.shape[0]` are fine). The state-based solvers differentiate through such a read — `forward_and_latent_grads` re-binds `z_latent` and differentiates the whole forward — while `EPCInference` evaluates `z_mu` at the carried latent, so a `z_latent`-dependent prediction makes the two solver families minimize different energies. An energy term that needs the node's own latent belongs in `energy()`.

> **muPC scaling is not applied inside `predict()`.** The inference and learning callsites scale inputs and gradients; doing so again here double-scales them. See the [Initialization and Scaling guide](05_initialization_and_scaling.md).

### Step 6: Use the Custom Node

```python
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import graph, TaskMap

# Create nodes
input_node = IdentityNode(shape=(28, 28, 1), name="input")

conv1 = Conv2DNode(
   shape=(26, 26, 16),  # VALID padding: 28-3+1 = 26
   kernel_size=(3, 3),
   stride=(1, 1),
   padding="VALID",
   name="conv1",
)

conv2 = Conv2DNode(
   shape=(24, 24, 32),
   kernel_size=(3, 3),
   stride=(1, 1),
   padding="VALID",
   name="conv2",
)

output_node = Linear(
   shape=(10,),
   flatten_input=True,
   name="output",
)

# Build graph
structure = graph(
   nodes=[input_node, conv1, conv2, output_node],
   edges=[
      Edge(source=input_node, target=conv1.slot("in")),
      Edge(source=conv1, target=conv2.slot("in")),
      Edge(source=conv2, target=output_node.slot("in")),
   ],
   task_map=TaskMap(x=input_node, y=output_node),
   inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
   scaling=MuPCConfig(),
)
```

## Custom Energy Terms

The default `energy()` scores the node's energy functional at `(state.z_latent, state.z_mu)`. Override it to add terms on top. `StorkeyHopfield` is the worked example — its attractor term reads `state.z_latent` inside `energy()` while its aux carries only `(W, strength)`, which come from params:

```python
@staticmethod
def energy(params, inputs, state, aux, node_info):
    """Combined energy: the functional default plus the attractor term."""
    if aux is None:                        # source node: predict() never ran and
        return NodeBase.energy(params, inputs, state, aux, node_info)  # no W exists

    W, strength = aux                      # params-derived, snapshotted at predict time
    energy = NodeBase.energy(params, inputs, state, aux, node_info)

    z = state.z_latent                     # read the latent HERE, not via aux
    wz = z @ W
    D = z.shape[-1]
    E_hopfield = (0.5 / D) * jnp.sum(wz * (wz - z), axis=-1)
    return energy + strength * E_hopfield
```

The returned energy stays per-sample (shape `(batch,)`); summation over the batch dimension is owned by the gradient templates, which need the resulting scalar for autodiff.

An override must tolerate `aux=None`: on the `in_degree == 0` path `predict()` never runs (and the param initializer gives sources empty params), so an extra term whose inputs a source cannot have falls back to the base energy, as above.

### The aux pattern

`aux` carries intermediates that `predict()` already computed and `energy()` wants to reuse — `Linear` returns its `pre_activation` (consumed by `LinearExplicitGrad`'s analytic gradients); `StorkeyHopfield` returns its prepared `(W, strength)`. The rule: **aux entries may depend only on `params` and `inputs`**, which don't change between `predict` and `energy`.

**The anti-pattern**: an aux entry computed from `state.z_latent` in `predict`. aux is snapshotted when `predict` runs — under `EPCInference`, *before* `z_latent` is derived as `z_mu + error`. Under state-based inference that's harmless (z_latent doesn't change within the call), but under ePC anything in aux computed from `state.z_latent` is frozen at the carried latent while the rest of the energy uses the derived one. Concretely: a `predict` that stashes `gate = sigmoid(state.z_latent)` in aux and an `energy` returning `gate * ||z_latent - z_mu||**2` gives sPC the one-point energy `sigmoid(z)*||z - mu||**2` but gives ePC `sigmoid(z_old)*||z - mu||**2` — a function of two different latents. The two solvers then minimize different energies and no longer share equilibria. An energy term that needs the node's own latent reads `state.z_latent` inside `energy()`, as the Hopfield attractor term above does.

## Useful Patterns

### FlattenInputMixin

For nodes that need dense (fully-connected) behavior with flattened inputs, use the `FlattenInputMixin`:

```python
from fabricpc.nodes.base import FlattenInputMixin

class MyDenseNode(FlattenInputMixin, NodeBase):
    @staticmethod
    def predict(params, inputs, state, node_info):
        batch_size = state.z_latent.shape[0]
        out_shape = node_info.shape

        # Sum (flattened_input @ weight) over all input edges -> (batch, *out_shape)
        pre_activation = MyDenseNode.compute_linear(
            inputs, params.weights, batch_size, out_shape
        )

        # Add bias, if the node has one
        if "b" in params.biases and params.biases["b"].size > 0:
            pre_activation = pre_activation + params.biases["b"]

        # Apply activation
        z_mu = type(node_info.activation).forward(
            pre_activation, node_info.activation.config
        )

        return z_mu, None
```

The mixin provides:
- `flatten_input(x)`: Flattens one input tensor from `(batch, *shape)` to `(batch, numel)`
- `reshape_output(x_flat, out_shape)`: Reshapes `(batch, numel)` back to `(batch, *out_shape)`
- `compute_linear(inputs, weights, batch_size, out_shape)`: Sums `flattened_input @ weight` across all input edges and reshapes to `(batch, *out_shape)`. It does not add a bias — add it yourself, as shown above.

### The `forward_and_latent_grads()` contract

`forward_and_latent_grads(params, inputs, state, node_info, is_clamped)` drives the inference phase. The base implementation's responsibilities:

1. **In-degree-0 nodes short-circuit the gradient computation**: the state comes from the template `forward()` (whose source guard mirrors `z_mu <- z_latent`, cast to `z_mu`'s dtype, with zero error) and all gradients are zero — the source's latent still moves via the contributions its downstream successors accumulate into its `latent_grad`.
2. **Every node with in-degree > 0 — unclamped readouts included — takes the autodiff path**: error = `z_latent - z_mu`, energy as `forward()` assigns, `z_latent` relaxed like any other node. A Hopfield readout therefore settles onto its attractor.
3. **The per-sample `state.energy` (shape `(batch,)`) is summed over the batch dimension** to a scalar. The resulting weight gradients are batch sums; `fabricpc.training.pc_weight_gradients` divides them once by the prediction count so the optimizer sees means per prediction.
4. **`jax.value_and_grad` differentiates that scalar** w.r.t. the input tensors and `z_latent`.

It returns `(NodeState, input_grads, self_grad)`: the updated state, gradients w.r.t. each input edge (dE/d_input, unscaled), and this node's dE/dz_latent contribution (unscaled). muPC scaling and accumulation into `state.latent_grad` are handled by the callsite (the inference loop).

### Explicit Gradients

By default, FabricPC computes gradients with JAX autodiff: it differentiates through your `predict()` to obtain both the latent gradients (inference) and the weight gradients (learning). For hand-coded gradients (e.g. for efficiency or control), override `forward_and_latent_grads()` and `forward_and_weight_grads()`. These return gradients alongside the updated state:

```python
class MyNode(NodeBase):
    @staticmethod
    def forward_and_latent_grads(params, inputs, state, node_info, is_clamped):
        # Run the template forward() for the updated state, then compute
        # gradients analytically. forward_with_aux() also surfaces predict's
        # aux (LinearExplicitGrad reads pre_activation this way).
        node_class = node_info.node_class
        state = node_class.forward(params, inputs, state, node_info)

        # Self-latent gradient dE/dz_latent via the energy functional
        energy_obj = node_info.energy
        self_grad = type(energy_obj).grad_latent(
            state.z_latent, state.z_mu, energy_obj.config
        )

        # Per-edge input gradients dE/d_input (uses activation.derivative())
        input_grads = {...}

        # Returns (updated_state, input_grads, self_grad)
        return state, input_grads, self_grad

    @staticmethod
    def forward_and_weight_grads(params, inputs, state, node_info):
        node_class = node_info.node_class
        state = node_class.forward(params, inputs, state, node_info)

        # Compute weight/bias gradients analytically ...
        # Returns (updated_state, NodeParams(weights=..., biases=...))
        return state, NodeParams(weights=weight_grads, biases=bias_grads)
```

Note that muPC scaling and accumulation into `state.latent_grad` are handled by the callsite, not inside these overrides. See `LinearExplicitGrad` (`fabricpc/nodes/linear_explicit_grad.py`) for a complete example, including its `compute_gain_mod_error()` helper that combines `state.error` with `activation.derivative()`.

### Multiple Input Slots

For nodes with distinct input types (e.g., data and mask):

```python
@staticmethod
def get_slots():
    return {
        "in": SlotSpec(name="in", is_multi_input=True),
        "mask": SlotSpec(name="mask", is_multi_input=False),
    }

@staticmethod
def predict(params, inputs, state, node_info):
    # Access inputs by slot
    data_inputs = {k: v for k, v in inputs.items() if k.endswith(":in")}
    mask_inputs = {k: v for k, v in inputs.items() if k.endswith(":mask")}

    # Process separately
    # ...
```

### Working Within the Fixed NodeState Schema

`NodeState` is a fixed-schema `NamedTuple` with exactly these fields:

```python
class NodeState(NamedTuple):
    z_latent: jnp.ndarray       # latent states (what the network infers)
    z_mu: jnp.ndarray           # predictions (what the network predicts)
    error: jnp.ndarray          # prediction errors (z_latent - z_mu)
    energy: jnp.ndarray         # per-sample energy, shape (batch,)
    latent_grad: jnp.ndarray    # gradient accumulator for inference updates
```

You cannot add custom fields to it. `state._replace(...)` only updates these existing fields, so there is no place to stash arbitrary per-step memory (such as an RNN hidden vector) on the `NodeState`. `aux` does not help here either: it is created and consumed within one forward trace, never stored.

If your node needs additional state, route it through what already exists:

- **Latent memory**: anything the node should "remember" and infer over belongs in `z_latent`. The inference loop already carries `z_latent` across steps and updates it from the energy gradient.
- **Inputs**: state supplied by other nodes arrives through `inputs`, keyed by edge. Add an input slot (see [Multiple Input Slots](#multiple-input-slots)) to receive it.
- **Parameters**: fixed-per-graph quantities belong in `params.weights` / `params.biases`, allocated in `initialize_params()`.

Adding a genuinely new piece of dynamic state (a field on `NodeState`) is a framework-level change, not something a single node can do on its own.

## Testing Your Node

Write tests to verify:

1. **Shape correctness**: Output shapes match the node's declared shape
2. **Energy decreases**: Inference reduces free energy across steps
3. **Gradient flow**: Learning updates weights in the expected direction

Example test:

```python
import jax
import jax.numpy as jnp
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import graph, TaskMap
from fabricpc.core.inference import InferenceSGD
from fabricpc.graph_initialization import initialize_params


def test_conv2d_energy_decreases():
   """Test that inference reduces energy for Conv2D node."""
   # Build graph
   input_node = IdentityNode(shape=(28, 28, 1), name="input")
   conv = Conv2DNode(shape=(28, 28, 16), kernel_size=(3, 3), name="conv")
   output = IdentityNode(shape=(28, 28, 16), name="output")

   structure = graph(
      nodes=[input_node, conv, output],
      edges=[
         Edge(source=input_node, target=conv.slot("in")),
         Edge(source=conv, target=output.slot("in")),
      ],
      task_map=TaskMap(x=input_node, y=output),
      inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
   )

   # Initialize parameters
   rng_key = jax.random.PRNGKey(0)
   params = initialize_params(structure, rng_key)

   # Create dummy data
   batch_size = 4
   x = jax.random.normal(rng_key, (batch_size, 28, 28, 1))
   y = jax.random.normal(rng_key, (batch_size, 28, 28, 16))

   # Run inference and track energy
   # ... (see existing test examples in tests/)

   # Assert energy decreases
   assert final_energy < initial_energy
```

## Summary

Creating custom nodes involves:

1. **Subclass `NodeBase`**: Define your node class
2. **Implement `get_slots()`**: Specify input slots
3. **Implement `initialize_params()`**: Allocate and initialize weights/biases
4. **Implement `predict()`**: Compute the prediction `z_mu`, returning `(z_mu, aux)`
5. **Optional overrides**:
   - `energy()`: For extra energy terms (read `state.z_latent` there, never via aux)
   - `get_variance_factor()`: For correct muPC scaling
   - `forward_and_latent_grads()` / `forward_and_weight_grads()`: For explicit gradients
6. **Test**: Verify shapes, energy convergence, and gradient flow

The base templates (`forward`, `forward_with_aux`, `forward_from_error`) and the error pair (`pair_error`/`pair_latent`) are not override points. With the methods above in place, your custom node integrates with the rest of FabricPC's infrastructure: graph building, both inference parameterizations, learning, and scaling.
