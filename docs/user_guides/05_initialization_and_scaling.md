# Initialization and Scaling

Proper initialization is critical for training deep predictive coding networks. This guide covers weight initialization strategies, the muPC scaling framework, and state initialization.

## Why Initialization Matters in PC

In predictive coding, the quality of initialization directly affects:

1. **Inference convergence**: If activations or errors are poorly scaled, the inference process (optimizing latent states to minimize energy) may fail to converge or get stuck.

2. **Gradient flow**: During learning, weight gradients are computed from prediction errors. If errors vanish or explode, learning stalls or diverges.

3. **Depth scaling**: In deep networks, these problems compound across layers. Without careful initialization, networks deeper than a few layers become untrainable.

Poor initialization causes:
- Vanishing or exploding activations
- Inference failing to converge (energy stays flat or oscillates)
- Learning stalling (zero or exploding weight gradients)
- Instability across network depths

## Weight Initializers

FabricPC provides several weight initialization strategies.

### XavierInitializer

Glorot initialization, designed to maintain variance across layers.

```python
from fabricpc.core.initializers import XavierInitializer

weight_init = XavierInitializer(gain=1.0)
```

Weights are drawn from:
```
std = gain * sqrt(2 / (fan_in + fan_out))
W ~ N(0, std^2)
```

Best for sigmoid and tanh activations in shallow to medium-depth networks.

### KaimingInitializer

He initialization, optimized for ReLU activations.

```python
from fabricpc.core.initializers import KaimingInitializer

weight_init = KaimingInitializer(gain=1.0)
```

Weights are drawn from:
```
std = gain * sqrt(2 / fan_in)
W ~ N(0, std^2)
```

Best for ReLU and other rectified activations. Does not account for fan_out, focusing on maintaining variance in the forward pass.

### MuPCInitializer

Unit variance initialization for use with muPC scaling.

```python
from fabricpc.core.initializers import MuPCInitializer

weight_init = MuPCInitializer(gain=1.0)
```

Weights are drawn from:
```
W ~ N(0, gain^2)
```

The width and depth scaling is handled separately by the muPC framework during the forward pass. This decoupling is key to muPC's effectiveness.

### NormalInitializer

Direct control over mean and standard deviation.

```python
from fabricpc.core.initializers import NormalInitializer

weight_init = NormalInitializer(mean=0.0, std=0.01)
```

Weights are drawn from:
```
W ~ N(mean, std^2)
```

Useful for custom initialization schemes or debugging.

### UniformInitializer

Uniform distribution over a specified range.

```python
from fabricpc.core.initializers import UniformInitializer

weight_init = UniformInitializer(min_val=-0.1, max_val=0.1)
```

Weights are drawn from:
```
W ~ U(min_val, max_val)
```

## muPC: The Recommended Default

**muPC (Maximal Update Parameterization for Predictive Coding)** is the recommended initialization and scaling strategy for networks with more than a few layers. It maintains O(1) activations, errors, and gradients across arbitrary graph topologies, from 2 layers to 100+ layers.

muPC achieves this through four complementary scaling mechanisms:

1. **Variance-factor scaling** — normalizes for the node's input transform (Kaiming fan_in for a weight matrix, `1/n` for an average pool)
2. **Per-slot in-degree scaling (K_slot)** — normalizes for multiple inputs summing into a slot
3. **Residual depth scaling (L)** — damps each branch contribution once, where it joins the residual stream at a merge node
4. **Saturative activation compensation** — normalizes gradients for activations like tanh and GELU whose derivatives are < 1

These are combined in a single per-edge scaling factor computed automatically from the graph topology, enabling stable training of networks with 100+ layers.

### Key Idea

muPC decouples weight initialization from forward pass scaling:

1. **Initialization**: Weights are drawn from a simple distribution (e.g., `N(0, 1)`) independent of network width or depth.

2. **Forward scaling**: Width and depth scaling is applied during the forward pass via per-edge scaling factors computed automatically from the graph topology. Each edge is scaled based on the properties of its target slot, not the whole node.

This separation allows the same initialization to work across different network depths and widths without retuning hyperparameters.

### How to Enable muPC

Use `MuPCInitializer` for node weights and `MuPCConfig` in the graph builder:

```python
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.initializers import MuPCInitializer
from fabricpc.nodes import Linear
from fabricpc.core.activations import TanhActivation

hidden = Linear(
    shape=(256,),
    activation=TanhActivation(),
    weight_init=MuPCInitializer(),
    name="h1",
)

structure = graph(
    nodes=[input_node, hidden, output_node],
    edges=[...],
    task_map=TaskMap(x=input_node, y=output_node),
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    scaling=MuPCConfig(),
)
```

### The Scaling Formula

muPC computes a scaling factor `a` for each incoming edge based on the target node and slot.

```
                        Per-Edge Scaling Overview

  source ──────── edge ──────── target:slot
                    │
                    ↓
          ┌───────────────────────────┐
          │ is_variance_scalable?     │
          │                           │
          │  Yes, non-merge target:   │   No (skip/mask):
          │  a = gain                 │   a = 1.0
          │     / sqrt(v*K_slot)      │   (pass through)
          │                           │
          │  Yes, merge target:       │
          │  a = gain                 │
          │     / sqrt(v*K_slot       │
          │            * L)           │
          └───────────────────────────┘
```

#### Hidden Nodes

For edges into hidden nodes with `is_variance_scalable=True`:

```
a = gain / sqrt(v * K_slot)          (non-merge nodes)
a = gain / sqrt(v * K_slot * L)      (merge nodes)
```

where:
- `gain`: The activation function's variance gain (e.g., `sqrt(5/3)` for tanh, `sqrt(2)` for ReLU)
- `v`: The node's input-transform variance factor (from `get_variance_factor()` — see Variance Factor below)
- `K_slot`: The in-degree of the **specific slot** receiving this edge (not the whole node's in-degree)
- `L`: The residual depth — number of merge nodes along the longest path (see Residual Depth below)

A **merge node** is a node with at least one connected `is_skip_connection` slot — the point where a computed branch joins the residual stream. The `1/sqrt(L)` damps each branch contribution once, at the merge; edges elsewhere — stems, branch-interior layers, stream projections, post-stream layers — carry no depth factor.

This scaling ensures:
- **Width invariance**: O(1) activations as `v` increases
- **Multi-input invariance**: O(1) activations when multiple edges sum into a slot
- **Depth invariance**: O(1) activations as the number of residual blocks increases
- **Activation invariance**: Correct variance preservation for any activation function

For edges into slots with `is_variance_scalable=False` (e.g., skip connections, attention masks), the scaling factor is 1.0 — the input passes through unmodified.

#### Output Nodes

For edges into output nodes (when `include_output=True`):

```
a = gain / (v * sqrt(K_slot))
```

The stronger `1/v` scaling (instead of `1/sqrt(v)`) is a feature-learning requirement: at initialization the readout sums `fan_in` uncorrelated terms (an O(sqrt(fan_in)) sum, giving O(1/sqrt(fan_in)) outputs), but after training the weight updates correlate with the incoming features and the sum grows as O(fan_in), so `1/fan_in` keeps trained outputs O(1). This scaling is optimal for regression tasks with identity activation and Gaussian energy (MSE loss).

The depth factor `L` is absent by design: the hidden `1/sqrt(L)` damps the L branch contributions summed into the residual stream, while the readout is applied once to the already-O(1) stream. This matches the muPC reference output scaling `a_L = 1/N` (Table 1 of arXiv:2505.13124), which carries no depth factor.

### Variance Factor

`v` is the factor by which a node's input transform multiplies input variance, before the activation. Each node class reports it through `get_variance_factor(source_shape, config, weight_init)`, and the edge scale `a = gain / sqrt(v * K_slot)` undoes it.

For a matmul against unit-variance weights, `v` is the Kaiming fan_in: each output unit sums `fan_in` independent products. But `v` is a float, not a dimension count, and it can be **below 1** — a transform that *reduces* variance reports `v < 1`, and muPC amplifies rather than attenuates.

| Node Type | `v` | Notes |
|-----------|-----|-------|
| Linear (`flatten_input=False`) | `source_shape[-1]` | Kaiming fan_in, last-axis features |
| Linear (`flatten_input=True`) | `prod(source_shape)` | Kaiming fan_in, all dims flattened |
| LinearResidual | Same as Linear | Only "in" slot has weights |
| ConvNode | `C_in * prod(kernel_size)` | Kaiming fan_in, input values per output unit |
| TransformerBlock | `embed_dim` | Last axis of input shape |
| Transformer v2 weighted nodes | `source_shape[-1]` | Base-class default (`embed_dim` or `ff_dim`) |
| IdentityNode / SkipConnection | 1 | Weightless sum, no reduction |
| AvgPool | `1/n` | Mean over `n` cells, so `a` amplifies by `sqrt(n)` |
| MaxPool | 1 | No distribution-free factor exists — see below |
| StorkeyHopfield | `(1 + s²r)/(1 + s)²` | Strength-weighted blend; `s` = hopfield_strength |

**AvgPool.** A mean over `n` cells divides a sum of `n` terms by `n`, so for uncorrelated cells it multiplies variance by `1/n` exactly, independent of the input distribution. `n` is the window volume, or every spatial dimension under `global_pool=True`. This is the same convention muPC applies to a `K_slot`-way edge sum (`1/sqrt(K_slot)`, not `1/K_slot`): both are sums, and both are normalized to preserve variance rather than magnitude. Real conv feature maps are spatially correlated, so the realized reduction lies between `1/n` and `1` and `sqrt(n)` over-corrects in proportion — the same independence assumption Kaiming already makes across a weight matrix's fan_in.

**MaxPool.** The variance of a max depends on the input distribution, and the two that matter disagree. Measured ratio of `Var(max over n)` to the input variance:

| n | Gaussian N(0,1) input | ReLU(N(0,1)) input |
|---|---|---|
| 4 | 0.49 | 1.32 |
| 9 | 0.36 | 1.05 |
| 16 | 0.29 | 0.87 |

A max pool in a conv graph follows a ReLU, where the ratio at the usual 2×2 window is 1.32 — a Gaussian-derived correction would be wrong by a factor of 2.7 for the inputs it actually sees. No single constant is right, so `v = 1`. Max pooling also shifts the mean, which no edge multiplier can remove; prefer `AvgPool` where variance behavior through depth matters.

**StorkeyHopfield.** The node blends an identity path and a learned path internally, and the blend shrinks variance: the coefficients `1/(1+s)` and `s/(1+s)` sum to 1, but the two terms are near-independent so their variances add in quadrature. With `r = Var(probe @ W)/Var(probe)`, `v(s) = (1 + s²r)/(1 + s)²`, which is at most 1 and bottoms out at `r/(1+r)` when `s = 1/r`. `r` is derived from `weight_init` (via `InitializerBase.element_variance`) and the node's symmetry constraints; under the defaults it is `(D+1)/(2D) ≈ 1/2`, so the worst case is `v = 1/3`. Left uncorrected that compounds through a chain. With a learnable strength the factor is evaluated at the initial `s = 1.0` and held fixed, muPC's standard static-scale contract; the drift is bounded by `sqrt(3)` since `v` never leaves `[1/3, 1]`.

### Per-Slot In-Degree Scaling (K_slot)

Unlike simple per-node in-degree scaling, muPC computes K (in-degree) independently for each **slot** on a node. This matters for nodes with multiple input slots like `LinearResidual`:

```python
# LinearResidual has two slots:
#   "in"   — receives transform path (is_variance_scalable=True)
#   "skip" — receives identity path  (is_variance_scalable=False)

res = LinearResidual(shape=(128,), activation=TanhActivation(),
                     weight_init=MuPCInitializer(), name="res1")

edges = [
    Edge(source=prev, target=res.slot("in")),    # K_slot = 1 for "in"
    Edge(source=prev, target=res.slot("skip")),   # not scaled (skip slot)
]
```

Here, the "in" slot has `K_slot=1` (one incoming edge) and gets the full muPC formula. The "skip" slot has `is_variance_scalable=False`, so its edge passes through at scale 1.0 regardless of how many edges connect to it.

If multiple edges connect to the same scalable slot, each is scaled by `1/sqrt(K_slot)`:

```python
# Two edges into the same "in" slot: K_slot = 2
Edge(source=a, target=node.slot("in"))
Edge(source=b, target=node.slot("in"))
# Each edge scaled by gain / sqrt(v * 2), times 1/sqrt(L) if the
# target is a merge node
```

### Residual Depth Scaling (L)

In residual networks, the identity (skip) path preserves signal at scale 1.0 while each residual block adds variance from the compute path. Without depth scaling, total variance grows linearly with the number of blocks.

**L** is the number of merge nodes — nodes with at least one **connected** `is_skip_connection=True` slot — along the longest path in the graph. It counts the variance-accumulating points where a computed branch sums into the residual stream.

```
How L is counted (ResNet with 3 blocks):

input ──→ h1 ──→ skip1(+) ──→ h2 ──→ skip2(+) ──→ h3 ──→ skip3(+) ──→ output
           │        ↑           │        ↑           │        ↑
           └────────┘           └────────┘           └────────┘
                L=1                 L=2                 L=3

Only nodes with connected is_skip_connection slots count toward L.
Nodes like h1, h2, h3 (no skip slots) do not increment L.
```

The `1/sqrt(L)` factor applies only to scalable edges **into merge nodes** — the branch contributions being summed into the stream. All other scalable edges are L-free: the stem (which produces the initial stream), layers inside a branch, weighted projections on the stream path (e.g., a downsample 1×1 conv), and layers after the last merge. Each branch is therefore damped exactly once, at the point where it joins the stream.

| Topology | L | Effect |
|----------|---|--------|
| Pure sequential chain | 1 | No depth factor: `a = gain / sqrt(v * K_slot)` |
| ResNet with D blocks | D | Each branch scaled by `1/sqrt(L)` at its merge |
| Decomposed transformer, depth d | 2d | Two merge nodes per block |
| Mixed architecture | max skip depth | Computed from the longest merge-node path |

In the decomposed transformer (see [Nodes API](10_api_nodes.md)), each block contributes two merge nodes — `MhaResidualNode` via `"skip"` and `Mlp2ResidualNode` via `"residual"` — so a depth-`d` model has residual depth `L = 2d`. Because these nodes are merges, their scalable `"in"` edges carry the L factor.

With the L-free stem producing a unit-variance stream, each merge adds `O(1/L)` variance and the total grows as `(1 + 1/L)^L`, which is bounded by approximately **e** (~2.72) — stable regardless of depth.

**What counts toward L**: Only connected slots with `is_skip_connection=True`. `is_variance_scalable` controls edge scaling, never L: slots that are merely non-scalable (like attention mask slots with `is_variance_scalable=False, is_skip_connection=False`) do **not** contribute. Neither does a declared-but-unconnected skip slot (e.g., a `LinearResidual` used without a skip edge) — it neither raises L nor makes its node a merge.

### Activation Gain and Jacobian Compensation

muPC uses two activation-dependent factors to maintain proper signal and gradient flow.

#### Variance Gain

The `variance_gain()` method returns the Kaiming-style gain for each activation — the factor needed to preserve unit variance through the activation function:

| Activation | `variance_gain()` | Rationale |
|------------|-------------------|-----------|
| Identity | 1.0 | Linear passthrough |
| ReLU | sqrt(2) | Zeroes negative half |
| LeakyReLU(α) | sqrt(2 / (1 + α²)) | Attenuates negative half |
| GELU | sqrt(2) | Similar to ReLU |
| Tanh | sqrt(5/3) | Saturating compression |
| HardTanh | sqrt(5/3) | Saturating compression |
| Sigmoid | 1.0 | — |
| Softmax | 1.0 | — |

This gain appears in the forward scaling formula, ensuring that `Var(activation(a * W @ x))` remains O(1).

#### Jacobian Gain

The `jacobian_gain()` method compensates for saturating activations that cause per-hop gradient decay. The per-hop Jacobian `diag(act'(z)) @ (a*W)` has RMS singular value proportional to `gain * rms(act'(z))`. For saturating activations, `rms(act'(z)) < 1`, causing gradients to shrink as `(gain * rms(act'))^L` over L hops.

The Jacobian gain normalizes this to ~1.0 per hop:

```
jacobian_gain = 1 / (gain * rms(act'(z)))
```

| Activation | `jacobian_gain()` | `rms(act'(z))` |
|------------|-------------------|-----------------|
| Identity | 1.0 | 1.0 |
| ReLU | 1.0 | 1/sqrt(2) |
| LeakyReLU(α) | 1.0 | sqrt((1+α²)/2) |
| GELU | 1.168 | ~0.606 |
| Tanh | 1.261 | ~0.614 |
| HardTanh | 1.261 | ~0.614 |

For non-saturating activations (Identity, ReLU, LeakyReLU), `jacobian_gain = 1.0` — gradients propagate without decay. For saturating activations (Tanh, GELU, HardTanh), the Jacobian gain is > 1, compensating for the activation's slope being < 1 on average.

### Gradient Scaling

muPC scales three types of gradients to maintain O(1) magnitude across depth.

#### Top-Down (Latent) Gradient Scaling

During inference, gradients flow from downstream nodes back to upstream nodes to update latent states. muPC scales these gradients per edge:

```
c_td = a * jacobian_gain
```

This combines two corrections:

1. **Chain rule correction (a)**: The forward scaling pre-multiplies inputs (`x → a*x`) before the `value_and_grad` closure. Autodiff yields `dE/d(a*x)`; multiplying by `a` restores the correct gradient `dE/dx`.

2. **Jacobian compensation (jacobian_gain)**: Normalizes the per-hop gradient propagation factor to ~1.0, preventing exponential gradient vanishing in deep networks with saturating activations.

For edges into non-scalable slots (`is_variance_scalable=False`), the top-down gradient scale is 1.0.

#### Self-Gradient Scaling

The self-gradient `dE/dz` from a node's energy functional is already O(1) when forward scaling maintains O(1) activations. The self-gradient scale is always **1.0**.

#### Weight Gradient Scaling

Weight gradient scaling is currently **1.0** per edge — the optimizer's learning rate handles gradient magnitude. This is a placeholder for future exploration of per-edge learning rate adaptation.

### Skip Connections and Residual Networks

Deep residual networks require special handling to prevent muPC from attenuating the identity (skip) path. FabricPC uses the `SlotSpec` attributes `is_variance_scalable` and `is_skip_connection` to control this.

#### The Problem with Naive In-Degree Scaling

Consider a residual block where both the skip and transform paths feed into a node with in-degree K=2. Without special handling, both paths would be scaled by `1/sqrt(2)` ≈ 0.707. Over L blocks, the skip path's signal decays as `0.707^L` — destroying the identity mapping that makes residual networks trainable.

```
Skip signal amplitude through 10 residual blocks:

  With naive scaling (both paths × 0.707):    Without skip scaling (muPC):
  Block 1:  ████████████████████  1.00         ████████████████████  1.00
  Block 2:  ██████████████        0.71         ████████████████████  1.00
  Block 3:  ██████████            0.50         ████████████████████  1.00
  Block 4:  ███████               0.35         ████████████████████  1.00
       ...     ...                 ...              ...               ...
  Block 10: █                     0.03         ████████████████████  1.00
```

#### SlotSpec Attributes

Each input slot on a node has two muPC-relevant flags:

- **`is_variance_scalable`**: When `True` (default), muPC applies the full scaling formula. When `False`, the edge passes through at scale 1.0.

- **`is_skip_connection`**: When `True`, the slot is a variance-accumulating merge point and contributes to the residual depth L. This flag implies `is_variance_scalable=False` — setting both to `True` raises a `ValueError`.

#### SkipConnection Node

`SkipConnection` is a weightless merge node that sums inputs from two slots and passes the sum through — `LinearResidual`'s slot layout without the weights:

- **"in"** (`is_variance_scalable=True`): receives computed branch contributions; each edge is scaled by `gain / sqrt(K_slot * L)` (`v`=1), the once-per-branch depth damping applied where the branch joins the stream.
- **"skip"** (`is_skip_connection=True`): receives the residual stream; edges pass through at scale 1.0. Connecting this slot counts the node toward L.

```python
from fabricpc.nodes import Linear, SkipConnection

linear = Linear(shape=(128,), activation=TanhActivation(),
                weight_init=MuPCInitializer(), name="h1")
skip = SkipConnection(shape=(128,), name="res1")

edges = [
    Edge(source=prev, target=linear.slot("in")),    # branch (scaled)
    Edge(source=prev, target=skip.slot("skip")),    # stream (unscaled)
    Edge(source=linear, target=skip.slot("in")),    # branch joins stream
]
```

#### LinearResidual Node

`LinearResidual` combines a linear transform and a residual sum in a single PC node, using two slots with different scaling behavior:

- **"in"** slot (`is_variance_scalable=True`): Receives the transform path. Has a weight matrix. Scaled by muPC.
- **"skip"** slot (`is_skip_connection=True`): Receives the identity path. No weight matrix. Passes through at scale 1.0.

```python
from fabricpc.nodes import LinearResidual

prev = stem
for i in range(num_blocks):
    res = LinearResidual(shape=(W,), activation=TanhActivation(),
                         weight_init=MuPCInitializer(), name=f"res{i}")
    edges += [
        Edge(source=prev, target=res.slot("in")),    # transform path (scaled)
        Edge(source=prev, target=res.slot("skip")),   # identity skip (unscaled)
    ]
    prev = res
```

#### When to Use Which

- **SkipConnection** (2 nodes per block): More PC inference flexibility — each block has two latent variables to optimize. Better for architectures where skip and transform paths need independent inference dynamics.

- **LinearResidual** (1 node per block): More efficient — halves the number of PC nodes and inference steps. Better when you want a compact residual network.

### Arbitrary Graph Support

muPC works on any DAG topology, not just sequential chains or ResNets. Per-edge scaling based on slot properties handles:

- **Multi-input nodes**: K_slot normalizes for multiple sources into one slot
- **Skip connections**: Non-scalable slots preserve identity mappings
- **Branching paths**: Each branch endpoint is independently scaled at its target
- **Lateral connections**: Edges between parallel paths are scaled normally

The residual depth L is a global property computed once from the full graph. The per-slot in-degree K_slot is a local property of each slot. Together, they provide correct scaling for any topology — feedforward chains, ResNets, U-Nets, multi-path architectures with lateral connections, and more.

### The `include_output` Flag

The `MuPCConfig` constructor accepts an `include_output` parameter:

```python
# Exclude output nodes from muPC scaling (default)
scaling = MuPCConfig(include_output=False)

# Include output nodes in muPC scaling
scaling = MuPCConfig(include_output=True)
```

**When to use `include_output=False` (default)**:
- Classification tasks with softmax activation and cross-entropy energy
- Networks trained with standard backprop-style output layers
- When the output layer should use traditional initialization (Xavier/Kaiming)

**When to use `include_output=True`**:
- Regression tasks with identity activation and Gaussian energy (MSE)
- When you want consistent muPC scaling throughout the entire network
- When output dimensionality is large and variance scaling matters

### Result

With muPC, you can train networks from 2 to 100+ layers without tuning the learning rate, weight initialization, or inference parameters based on depth. The same hyperparameters work across different architectures.

The combination of depth L scaling, Jacobian compensation for saturating activations, and identity-preserving skip connections ensures that activations, prediction errors, and gradients all remain O(1) regardless of network depth. For example, a 128-block ResNet with muPC trains stably on MNIST with the same hyperparameters as an 8-block version.

## State Initialization Strategies

After building the graph and initializing parameters, you need to set initial latent states (`z_latent`) for inference. FabricPC provides several strategies:

### FeedforwardStateInit (Default)

Runs a forward pass through the network and sets `z_latent = z_mu` for each node.

```python
from fabricpc.graph_initialization.state_initializer import FeedforwardStateInit

structure = graph(
    nodes=[...],
    edges=[...],
    task_map=...,
    inference=...,
    graph_state_initializer=FeedforwardStateInit(),
)
```

**Advantages**:
- Fast convergence for feedforward DAGs
- Latent states start close to the forward predictions
- Fewer inference steps needed to reach equilibrium

**Best for**: Feedforward and skip-connection architectures.

### GlobalStateInit

Initializes all nodes with the same distribution.

```python
from fabricpc.graph_initialization.state_initializer import GlobalStateInit
from fabricpc.core.initializers import NormalInitializer

structure = graph(
    nodes=[...],
    edges=[...],
    task_map=...,
    inference=...,
    graph_state_initializer=GlobalStateInit(initializer=NormalInitializer(mean=0.0, std=0.1)),
)
```

**Advantages**:
- Simple and consistent
- No forward pass needed

**Disadvantages**:
- May require more inference steps to converge
- Latent states start far from equilibrium

**Best for**: Debugging and baselines. (Cyclic graphs no longer need it: `FeedforwardStateInit` walks `structure.schedule`, so a graph built with `graph(..., unroll=U)` gets feedforward initialization through its cycles at degree U.)

### NodeDistributionStateInit

Each node uses its own `latent_init` initializer specified during node construction.

```python
from fabricpc.graph_initialization.state_initializer import NodeDistributionStateInit

hidden = Linear(
    shape=(256,),
    activation=TanhActivation(),
    latent_init=NormalInitializer(mean=0.0, std=0.1),
    name="hidden",
)

structure = graph(
    nodes=[...],
    edges=[...],
    task_map=...,
    inference=...,
    graph_state_initializer=NodeDistributionStateInit(),
)
```

**Advantages**:
- Fine-grained control per node
- Can tailor initialization to specific node types

**Best for**: Custom architectures where different nodes need different initialization strategies.

## Choosing Your Initialization

### Recommended Default

For most use cases, especially deep networks:

```python
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.initializers import MuPCInitializer
from fabricpc.graph_initialization.state_initializer import FeedforwardStateInit

# Node definition
hidden = Linear(
    shape=(256,),
    activation=TanhActivation(),
    weight_init=MuPCInitializer(),
    name="hidden",
)

# Graph construction
structure = graph(
    nodes=[...],
    edges=[...],
    task_map=...,
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    scaling=MuPCConfig(),
    graph_state_initializer=FeedforwardStateInit(),
)
```

### Alternative: Traditional Initialization

For shallow networks (2-3 layers) or when matching non-muPC baselines:

```python
from fabricpc.core.initializers import XavierInitializer

# Use Xavier for tanh/sigmoid
hidden = Linear(
    shape=(256,),
    activation=TanhActivation(),
    weight_init=XavierInitializer(),
    name="hidden",
)

# Or Kaiming for ReLU
hidden = Linear(
    shape=(256,),
    activation=ReLUActivation(),
    weight_init=KaimingInitializer(),
    name="hidden",
)

# Build graph without muPC scaling
structure = graph(
    nodes=[...],
    edges=[...],
    task_map=...,
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    # No scaling parameter
)
```

### Summary

| Network Depth | Recommended Initialization | Scaling | State Init |
|---------------|---------------------------|---------|------------|
| 2-3 layers | Xavier/Kaiming | None | FeedforwardStateInit |
| 4-10 layers | MuPCInitializer | MuPCConfig | FeedforwardStateInit |
| 10+ layers | MuPCInitializer | MuPCConfig | FeedforwardStateInit |
| Regression (MSE) | MuPCInitializer | MuPCConfig(include_output=True) | FeedforwardStateInit |
| Classification (CE) | MuPCInitializer | MuPCConfig(include_output=False) | FeedforwardStateInit |

When in doubt, use muPC. It's designed to work out-of-the-box for arbitrary depths and topologies.
