# Nodes API Reference

All node types extend `NodeBase` from `fabricpc.nodes.base`.

## The node contract

A node implements two computation methods:

- **`predict(params, inputs, state, node_info) -> (z_mu, aux)`** — all parameterized computation: the prediction `z_mu` (shape `(batch,) + node_info.shape`) plus `aux`, an arbitrary pytree of intermediates for `energy()` (`None` if unused). `predict` may read the node's own `state`, but under `EPCInference` the state carries the previous step's values, so `z_mu` is always evaluated at the carried state.
- **`energy(params, inputs, state, aux, node_info) -> (batch,)`** — per-sample energy at `(state.z_latent, state.z_mu)`. The default scores the node's energy functional; override to add terms (`StorkeyHopfield`'s attractor term).

Everything else is base-owned and **not an override point**: the error pair (`pair_error`: `error = z_latent - z_mu`; `pair_latent`: `z_latent = z_mu + error` — one volume-preserving bijection shared by both inference parameterizations) and the assembly templates `forward` (predict → pair → energy), `forward_with_aux` (also surfacing `aux`), and `forward_from_error` (the ePC derive direction). Source nodes (`in_degree == 0`) never reach `predict`: the templates mirror `z_mu <- z_latent` (cast to `z_mu`'s float dtype) with zero error.

**The aux pattern.** aux carries intermediates that depend only on `params` and `inputs`: `Linear` returns its `pre_activation` (consumed by `LinearExplicitGrad`'s analytic gradients); `StorkeyHopfield` returns its prepared `(W, strength)`. An energy term that needs the node's own `z_latent` reads `state.z_latent` inside `energy()` — the `StorkeyHopfield` attractor term is the worked example. **The anti-pattern**: an aux entry computed from `state.z_latent` in `predict`. aux is snapshotted at `predict` time — under ePC, before `z_latent` is derived — so such an entry freezes the carried latent into an energy otherwise evaluated at the derived latent, and the two solvers minimize different energies. See the [Writing Custom Nodes guide](06_custom_nodes.md#the-aux-pattern).

## Linear

`fabricpc.nodes.Linear`

Weighted projection node: `z_mu = activation(W @ x + b)`

```python
from fabricpc.nodes import Linear

node = Linear(
    shape=(256,),
    name="hidden1",
    activation=SigmoidActivation(),
    energy=GaussianEnergy(),
    use_bias=True,
    flatten_input=False,
    weight_init=KaimingInitializer(),
    latent_init=NormalInitializer(),
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, ...]` | required | Output shape excluding batch dimension |
| `name` | `str` | required | Node name (auto-prefixed with current namespace) |
| `activation` | `ActivationBase` | `IdentityActivation()` | Activation function |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |
| `use_bias` | `bool` | `True` | Whether to include a bias term |
| `flatten_input` | `bool` | `False` | If True, flatten all input dims for dense behavior |
| `weight_init` | `InitializerBase` | `KaimingInitializer()` | Weight initializer |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |

**Slots:** `"in"` (multi-input)

**Weight shape:**
- `flatten_input=False`: `(in_features, out_features)` — matmul on last axis
- `flatten_input=True`: `(in_numel, out_numel)` — fully-connected dense

**muPC variance factor:** the Kaiming fan_in.
- `flatten_input=False`: `source_shape[-1]` (last axis features)
- `flatten_input=True`: `prod(source_shape)` (all dims flattened)

---

## IdentityNode

`fabricpc.nodes.IdentityNode`

Passthrough node with no learnable parameters. Sums all inputs when multiple edges connect.

```python
from fabricpc.nodes import IdentityNode

pixels = IdentityNode(shape=(784,), name="pixels")
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, ...]` | required | Output shape excluding batch |
| `name` | `str` | required | Node name |
| `activation` | `ActivationBase` | `IdentityActivation()` | Activation function |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |
| `scale` | `float` | `1.0` | Fixed scaling factor applied to output |

**Slots:** `"in"` (multi-input)

**muPC variance factor:** Always returns `1` (weightless node).

---

## StorkeyHopfield

`fabricpc.nodes.StorkeyHopfield`

Associative memory node combining PC prediction-error energy with Hopfield attractor energy.

```python
from fabricpc.nodes import StorkeyHopfield

hopfield = StorkeyHopfield(
    shape=(128,),
    name="memory",
    hopfield_strength=1.0,
    activation=TanhActivation(),
)
# Connect with: Edge(source, hopfield.slot("in"))
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, ...]` | required | Output shape excluding batch |
| `name` | `str` | required | Node name |
| `hopfield_strength` | `float` | `None` | Initial blending strength (learnable if None, fixed if float) |
| `activation` | `ActivationBase` | `TanhActivation()` | Activation function |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | PC energy functional |
| `use_bias` | `bool` | `False` | Whether to include bias term |
| `enforce_symmetry` | `bool` | `True` | Symmetrize W via 0.5*(W+W.T) |
| `zero_diagonal` | `bool` | `False` | Zero W diagonal in forward pass |
| `weight_init` | `InitializerBase` | `XavierInitializer()` | Weight initializer for W matrix |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |

**Slots:** `"in"` (single-input, variance-scalable).

**muPC variance factor:** `v(s) = (1 + s²r)/(1 + s)²`, where `r = Var(probe @ W)/Var(probe)`. The blend coefficients `1/(1+s)` and `s/(1+s)` sum to 1, but the two terms are near-independent so their variances add in quadrature — the blend shrinks variance rather than preserving it. `v` is at most 1 and bottoms out at `r/(1+r)` when `s = 1/r`; under the default Xavier initialization plus symmetrization, `r = (D+1)/(2D) ≈ 1/2`, so the worst case is `v = 1/3` and the default `s = 1` gives `v = 0.375`. Uncorrected that compounds through a chain of these nodes. `get_variance_factor` reports `v(s)`, so muPC applies `a = gain/sqrt(v(s) · K_slot)` and the node holds O(1) variance at any strength and depth. With a learnable strength the factor is evaluated at the initial `s = 1.0` and held fixed (muPC's static-scale contract); the drift is bounded by `sqrt(3)`. The slot is not a skip connection, so the node does not count toward the residual depth L.

**Energy formulation:**
```
E_total = E_pc + hopfield_strength * E_hop
E_pc   = 0.5 ||z - mu||^2  (or user-specified energy)
E_hop  = (1/2D) z^T (W^2 - W) z
```

**Prediction:**
```
z_mu = activation(probe/(1+s) + (probe @ W) * s/(1+s) + bias)
```
where `s = softplus(raw_strength)` if `hopfield_strength=None` (learnable), otherwise `s = hopfield_strength` (fixed).

**Learnable parameters:** `W` (D x D matrix), `bias` (if `use_bias=True`), `raw_hopfield_strength` (if `hopfield_strength=None`)

---

## SkipConnection

`fabricpc.nodes.SkipConnection`

Weightless merge node for residual architectures. Sums inputs from two slots and passes the sum through: computed branch contributions enter `"in"` (scaled by muPC's once-per-branch depth damping `gain/sqrt(K_slot * L)`), the residual stream enters `"skip"` (unscaled, counts the node toward the residual depth L). This is `LinearResidual`'s slot layout without the weights.

```python
from fabricpc.nodes import SkipConnection

skip = SkipConnection(shape=(128,), name="res1")
# Connect with:
# Edge(branch, skip.slot("in"))     — branch joins the stream (scaled)
# Edge(stream, skip.slot("skip"))   — residual stream (unscaled)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, ...]` | required | Output shape excluding batch |
| `name` | `str` | required | Node name |
| `activation` | `ActivationBase` | `IdentityActivation()` | Activation function |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |

**Slots:**
- `"in"` (multi-input, `is_variance_scalable=True`) — branch contributions, scaled `gain/sqrt(K_slot * L)`
- `"skip"` (multi-input, `is_variance_scalable=False`, `is_skip_connection=True`) — residual stream, unscaled

**muPC variance factor:** Always returns `1` (weightless node).

**Difference from IdentityNode:** IdentityNode has a single scalable slot and no skip slot, so it does not count toward the residual depth L and its edges carry no depth factor. SkipConnection's unscaled `"skip"` slot preserves the identity mapping through deep residual networks, and its `"in"` slot applies the once-per-branch `1/sqrt(L)` damping.

---

## LinearResidual

`fabricpc.nodes.LinearResidual`

Linear residual node: `z_mu = activation(W @ x_in + b) + x_skip`. Combines a linear transformation (on the `"in"` slot) with an identity residual connection (on the `"skip"` slot) in one PC node.

```python
from fabricpc.nodes import LinearResidual

res = LinearResidual(
    shape=(128,),
    name="res1",
    activation=TanhActivation(),
    weight_init=MuPCInitializer(),
)
# Connect with:
# Edge(source, res.slot("in"))    — transform path (scaled)
# Edge(source, res.slot("skip"))  — identity skip (unscaled)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, ...]` | required | Output shape excluding batch |
| `name` | `str` | required | Node name |
| `activation` | `ActivationBase` | `IdentityActivation()` | Activation function |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |
| `use_bias` | `bool` | `True` | Whether to include a bias term |
| `flatten_input` | `bool` | `False` | If True, flatten all input dims for dense behavior |
| `weight_init` | `InitializerBase` | `KaimingInitializer()` | Weight initializer |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |

**Slots:**
- `"in"` (multi-input, `is_variance_scalable=True`): Transform path with weight matrix, scaled by muPC.
- `"skip"` (multi-input, `is_variance_scalable=False`, `is_skip_connection=True`): Identity skip path, no weight matrix, passes through at scale 1.0.

```
              ┌─────────────────────────────────────┐
              │         LinearResidual node         │
              │                                     │
prev ─────────┤  slot("in")  → W @ x + b → act()    │
              │                             ↓       │
              │                            (+) ──→ z_mu
              │                             ↑       │
prev ─────────┤  slot("skip") ──────────────┘       │
              │   (identity, unscaled)              │
              └─────────────────────────────────────┘
```

**Weight shape:** Same as Linear — `(in_features, out_features)` or `(in_numel, out_numel)` if `flatten_input=True`. Only `"in"` slot edges get weight matrices.

**muPC variance factor:** Same as Linear — `source_shape[-1]` or `prod(source_shape)` if `flatten_input=True`.

---

## ConvNode

`fabricpc.nodes.ConvNode`

Convolution node: `z_mu = activation(conv(x, kernel) + b)`, where `x` is the input feature map arriving on the `"in"` slot, `kernel` the node's convolution kernel, and `b` the per-channel bias. Each output position is predicted from a local window of the input feature map. One class covers 1D, 2D, and 3D convolution; the spatial rank is inferred from `len(shape) - 1`. Layout is channels-last: `(batch, spatial..., channels)`.

```python
from fabricpc.nodes import ConvNode

conv1 = ConvNode(
    shape=(28, 28, 32),      # output (H, W, C_out); padding "SAME" preserves 28x28
    name="conv1",
    kernel_size=(3, 3),
    stride=(1, 1),
    padding="SAME",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, ...]` | required | Output shape excluding batch: `(spatial..., C_out)` |
| `name` | `str` | required | Node name |
| `kernel_size` | `Tuple[int, ...]` | required | Window extent per spatial axis |
| `stride` | `Tuple[int, ...]` | all ones | Step per spatial axis |
| `padding` | `str` or pairs | `"SAME"` | `"SAME"`, `"VALID"`, or explicit `(low, high)` pairs per spatial axis |
| `activation` | `ActivationBase` | `ReLUActivation()` | Activation function |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |
| `use_bias` | `bool` | `True` | Whether to include a per-channel bias |
| `weight_init` | `InitializerBase` | `KaimingInitializer()` | Kernel initializer |
| `bias_init` | `InitializerBase` | `ZerosInitializer()` | Bias initializer |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |

**Slots:** `"in"` (multi-input)

**Weight shape:** `(*kernel_size, C_in, C_out)` — one kernel per incoming edge; contributions from multiple edges are summed. Dilation is not supported.

**Output shape validation:** the declared `shape` is checked against kernel, stride, and padding at `initialize_params`; a mismatch raises `ValueError` naming the node and the expected shape. Per spatial axis with input extent `n`, kernel extent `k`, and stride `s`: `"SAME"` gives `ceil(n / s)`; `"VALID"` gives `floor((n - k) / s) + 1`.

**muPC variance factor:** `C_in * prod(kernel_size)` — the number of input values contributing to each output unit.

---

## MaxPool

`fabricpc.nodes.MaxPool`

Max-pooling node: `z_mu = activation(windowed_max(x))`. Reduces spatial extent by taking the maximum over each window; no learnable parameters.

```python
from fabricpc.nodes import MaxPool

pool1 = MaxPool(
    shape=(14, 14, 32),      # output after 2x2 non-overlapping windows on 28x28
    name="pool1",
    window_shape=(2, 2),
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, ...]` | required | Output shape excluding batch: `(spatial..., C)` |
| `name` | `str` | required | Node name |
| `window_shape` | `Tuple[int, ...]` | required | Window extent per spatial axis |
| `stride` | `Tuple[int, ...]` | `window_shape` | Step per spatial axis; the default gives non-overlapping windows |
| `padding` | `str` or pairs | `"VALID"` | `"SAME"`, `"VALID"`, or explicit `(low, high)` pairs. Note: default differs from ConvNode's `"SAME"` |
| `activation` | `ActivationBase` | `IdentityActivation()` | Activation function |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |

**Slots:** `"in"` (multi-input)

Explicit padding that covers a full window on any axis is rejected at `initialize_params`: max pooling pads with negative infinity, and a window containing only padding would output negative infinity.

**muPC variance factor:** `1` — no correction. The variance of a max depends on the input distribution: at a 2×2 window the ratio is 0.49 for Gaussian inputs but 1.32 for the ReLU outputs a max pool actually receives, so no single constant is right. Max pooling also shifts the mean, which no edge scale can remove. Prefer `AvgPool` where variance behavior through depth matters.

---

## AvgPool

`fabricpc.nodes.AvgPool`

Average-pooling node: `z_mu = activation(windowed_mean(x))`. Two modes: windowed (like MaxPool, with the mean instead of the maximum) and global (`global_pool=True`), which averages over all spatial axes at once, `(batch, spatial..., C) -> (batch, C)`.

```python
from fabricpc.nodes import AvgPool

# Global average pooling: collapse the spatial grid to one vector per channel.
# Standard bridge from a convolutional stack to a classifier head.
avgpool = AvgPool(shape=(256,), name="avgpool", global_pool=True)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, ...]` | required | Output shape excluding batch. Rank-1 `(C,)` required when `global_pool=True`; construction raises `ValueError` otherwise |
| `name` | `str` | required | Node name |
| `window_shape` | `Tuple[int, ...]` | `None` | Window extent per spatial axis (windowed mode) |
| `stride` | `Tuple[int, ...]` | `window_shape` | Step per spatial axis |
| `padding` | `str` or pairs | `"VALID"` | `"SAME"`, `"VALID"`, or explicit `(low, high)` pairs |
| `global_pool` | `bool` | `False` | Average over all spatial axes instead of windows |
| `count_include_pad` | `bool` | `True` | Divide by the full window volume; `False` divides by the count of real (non-padding) elements |
| `activation` | `ActivationBase` | `IdentityActivation()` | Activation function |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |

**Slots:** `"in"` (multi-input)

**muPC variance factor:** `1/n`, where `n` is the window volume, or every spatial dimension under `global_pool=True`. A mean over `n` cells multiplies variance by `1/n` for uncorrelated cells, so muPC amplifies the in-edge by `sqrt(n)` instead of letting the pool attenuate by `1/sqrt(n)`.

---

## TransformerBlock

`fabricpc.nodes.TransformerBlock`

Multi-head self-attention + feedforward MLP in a single node. Uses Rotary Position Embeddings (RoPE).

```python
from fabricpc.nodes import TransformerBlock

block = TransformerBlock(
    shape=(256, 128),        # (seq_len, embed_dim)
    num_heads=8,
    ff_dim=512,
    name="transformer",
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, ...]` | required | Output shape `(seq_len, embed_dim)` |
| `name` | `str` | required | Node name |
| `activation` | `ActivationBase` | `IdentityActivation()` | Output activation function |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |
| `internal_activation` | `ActivationBase` | `GeluActivation()` | FFN internal activation |
| `num_heads` | `int` | `8` | Number of attention heads |
| `ff_dim` | `int` | `4 * embed_dim` | Feedforward hidden dimension |
| `dropout_rate` | `float` | `0.0` | Dropout rate (currently unused) |
| `pre_norm` | `bool` | `True` | Use pre-norm architecture |
| `use_rope` | `bool` | `True` | Use Rotary Position Embeddings |
| `rope_theta` | `float` | `10000.0` | Base frequency for RoPE |
| `weight_init` | `InitializerBase` | `KaimingInitializer()` | Weight initializer |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |

**Slots:** `"in"` (token embeddings), `"mask"` (optional attention mask)

**Architecture:**
```
x → LayerNorm → MHA → + → LayerNorm → FFN → +
└─────────────────────┘ └────────────────────┘
        (skip)                 (skip)
```

---

## Decomposed Transformer (v2)

Fine-grained transformer components, exported from `fabricpc.nodes`; the graph builder `create_deep_transformer` lives in `fabricpc.models`. Each sub-block stage is a separate PC node, so inference assigns a latent state and a prediction error to the attention and MLP stages individually instead of to a whole block.

- **`EmbeddingNode`** — Token embedding lookup
- **`MhaResidualNode`** — Multi-head attention with the block's first residual added inside the node
- **`LnMlp1Node`** — LayerNorm + first MLP projection
- **`Mlp2ResidualNode`** — Second MLP projection with the block's second residual added inside the node
- **`VocabProjectionNode`** — Projection to vocabulary logits

```
                         ┌──────────────────────── one transformer block ─────────────────┐
                         │                                                                │
tokens → EmbeddingNode ──┼──→ MhaResidualNode ──────→ LnMlp1Node ──→ Mlp2ResidualNode ────┼──→ VocabProjectionNode → logits
                    │    │    ("in", scaled) ↑   │                   ("in", scaled) ↑     │
                    │    │                   │   │                                  │     │
                    └────┼───→ ("skip") ─────┘   └──────→ ("residual") ─────────────┘     │
                         └────────────────────────────────────────────────────────────────┘
```

Each block wires two unscaled bypass edges: the previous block's output feeds both `MhaResidualNode` slots (`"in"` and `"skip"`), and the attention output feeds both the MLP path (`LnMlp1Node`) and `Mlp2ResidualNode`'s `"residual"` slot. Causal masking happens inside `MhaResidualNode` via `is_causal`; no mask node or mask edge exists in the graph.

Symbols used in the node formulas below:

| Symbol | Meaning |
|--------|---------|
| `z_mu` | The node's prediction for its latent state, computed by its forward pass |
| `x_in` | Input arriving on the node's `"in"` slot (the muPC-scaled path) |
| `x_skip`, `x_residual` | Inputs on the unscaled bypass slots of `MhaResidualNode` / `Mlp2ResidualNode` |
| `E` | The `(vocab_size, embed_dim)` embedding table |
| `W_q`, `W_k`, `W_v`, `W_o` | Attention projection weights inside `MhaResidualNode` |
| `W_ff1`, `b_ff1` | First MLP projection weight and bias (`LnMlp1Node`) |
| `W_ff2`, `b_ff2` | Second MLP projection weight and bias (`Mlp2ResidualNode`) |
| `W_out`, `b_out` | Vocabulary projection weight and bias (`VocabProjectionNode`) |
| `d` | Number of transformer blocks (the builder's `depth`) |

### create_deep_transformer

`fabricpc.models.create_deep_transformer`

Builds the complete depth-`d` language-model graph: embedding, `d` blocks of the three block nodes, and the vocabulary projection.

```python
from fabricpc.models import create_deep_transformer
from fabricpc.core.inference import InferenceSGD

structure = create_deep_transformer(
    depth=4,
    embed_dim=128,
    num_heads=8,
    mlp_dim=512,
    seq_len=256,
    vocab_size=11711,
    inference=InferenceSGD(eta_infer=0.1, infer_steps=20),
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `depth` | `int` | required | Number of transformer blocks |
| `embed_dim` | `int` | required | Embedding and residual-stream width |
| `num_heads` | `int` | required | Attention heads per block |
| `mlp_dim` | `int` | required | Hidden width of the block MLP |
| `seq_len` | `int` | required | Sequence length |
| `vocab_size` | `int` | required | Vocabulary size |
| `inference` | `InferenceBase` | required | Inference algorithm for latent updates |
| `weight_init` | `dict` | `None` | Block-weight initializer spec, e.g. `{"type": "normal", "std": 0.05}` or `{"type": "xavier"}`; `None` gives `NormalInitializer(std=0.02)` |

Returns a `GraphStructure` with `TaskMap(x=input_ids, y=logits)` — clamp `x` with int32 token ids `(batch, seq_len)`, read logits from `y` — plus `MuPCConfig(include_output=False)` and `FeedforwardStateInit()`.

**Initialization:** `weight_init` covers the block weights only. The builder overrides two nodes regardless: the embedding uses `NormalInitializer(std=1.0)` because muPC scaling is disabled on the discrete lookup edge and unit-normal keeps each token's embedding at order-1 variance, and the vocabulary projection uses `Normal(std=sqrt(1/embed_dim))` because each logit is a dot product over `embed_dim` features and this keeps initial logit variance at order 1. Nodes constructed directly get the class defaults in the tables below, not these overrides.

### EmbeddingNode

Token-embedding lookup: `z_mu = E[token_ids]`, where `E` is the `(vocab_size, embed_dim)` embedding table. Input is int32 token ids `(batch, seq_len)`; output is `(batch, seq_len, embed_dim)`. Token ids are discrete, so no gradient flows back through the input edge.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, int]` | required | `(seq_len, embed_dim)` |
| `name` | `str` | required | Node name |
| `vocab_size` | `int` | required | Rows of the embedding table |
| `embed_dim` | `int` | required | Columns of the embedding table |
| `weight_init` | `InitializerBase` | `NormalInitializer(std=0.02)` | Embedding-table initializer |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |

**Slots:** `"in"` (single-input, `is_variance_scalable=False` — muPC leaves the token-id edge unscaled)

### MhaResidualNode

Pre-norm multi-head self-attention with the residual added inside the node: `z_mu = x_skip + W_o @ MHA(LayerNorm(x_in))`, where `x_in` is the `"in"` slot input and `x_skip` the `"skip"` slot input. With `is_causal=True` a lower-triangular mask is applied to the attention scores inside the node. With `use_rope=True` rotary position embeddings are applied to queries and keys.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, int]` | required | `(seq_len, embed_dim)` |
| `name` | `str` | required | Node name |
| `embed_dim` | `int` | required | Residual-stream width |
| `num_heads` | `int` | required | Attention heads; head dimension is `embed_dim / num_heads` |
| `use_rope` | `bool` | `True` | Apply rotary position embeddings to Q and K |
| `rope_theta` | `float` | `10000.0` | Base frequency for RoPE |
| `is_causal` | `bool` | `True` | Apply the lower-triangular mask inside attention |
| `weight_init` | `InitializerBase` | `XavierInitializer()` | Initializer for `W_q`, `W_k`, `W_v`, `W_o` |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |

**Slots:**
- `"in"` (single-input, `is_variance_scalable=True`): attention-branch input, muPC-scaled.
- `"skip"` (single-input, `is_variance_scalable=False`, `is_skip_connection=True`): residual bypass, unscaled.

**muPC variance factor:** `source_shape[-1]` (base-class default) — `embed_dim` for the `"in"` edge.

### LnMlp1Node

LayerNorm followed by the first MLP projection: `z_mu = activation(W_ff1 @ LayerNorm(x_in) + b_ff1)`, where `x_in` is the `"in"` slot input — the block's `MhaResidualNode` output.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, int]` | required | `(seq_len, ff_dim)` |
| `name` | `str` | required | Node name |
| `embed_dim` | `int` | required | Input width (LayerNorm and `W_ff1` rows) |
| `ff_dim` | `int` | required | Hidden width (`W_ff1` columns) |
| `activation` | `ActivationBase` | `GeluActivation()` | Activation after the projection |
| `weight_init` | `InitializerBase` | `KaimingInitializer()` | Initializer for `W_ff1` |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |

**Slots:** `"in"` (single-input)

**muPC variance factor:** `source_shape[-1]` (base-class default) — `embed_dim`.

### Mlp2ResidualNode

Second MLP projection with the block residual added inside the node: `z_mu = x_residual + W_ff2 @ x_in + b_ff2`, where `x_in` is the `LnMlp1Node` output and `x_residual` the attention output.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, int]` | required | `(seq_len, embed_dim)` |
| `name` | `str` | required | Node name |
| `embed_dim` | `int` | required | Output width (`W_ff2` columns) |
| `ff_dim` | `int` | required | Input width (`W_ff2` rows) |
| `weight_init` | `InitializerBase` | `XavierInitializer()` | Initializer for `W_ff2` |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |
| `energy` | `EnergyFunctional` | `GaussianEnergy()` | Energy functional |

**Slots:**
- `"in"` (single-input, `is_variance_scalable=True`): MLP path, muPC-scaled.
- `"residual"` (single-input, `is_variance_scalable=False`, `is_skip_connection=True`): residual bypass, unscaled.

**muPC variance factor:** `source_shape[-1]` (base-class default) — `ff_dim` for the `"in"` edge.

### VocabProjectionNode

Projection to vocabulary logits: `z_mu = activation(W_out @ x_in + b_out)`, where `x_in` is the `"in"` slot input — the final block's `Mlp2ResidualNode` output — and the default activation is softmax. The default energy is `CrossEntropyEnergy` — unlike every other built-in node, which defaults to `GaussianEnergy` — so clamping one-hot targets on this node makes its energy the cross-entropy training loss.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `shape` | `Tuple[int, int]` | required | `(seq_len, vocab_size)` |
| `name` | `str` | required | Node name |
| `vocab_size` | `int` | required | Output width (`W_out` columns) |
| `embed_dim` | `int` | required | Input width (`W_out` rows) |
| `activation` | `ActivationBase` | `SoftmaxActivation()` | Output activation |
| `weight_init` | `InitializerBase` | `XavierInitializer()` | Initializer for `W_out` |
| `latent_init` | `InitializerBase` | `NormalInitializer()` | Latent state initializer |
| `energy` | `EnergyFunctional` | `CrossEntropyEnergy()` | Energy functional |

**Slots:** `"in"` (single-input)

**muPC variance factor:** `source_shape[-1]` (base-class default) — `embed_dim`.

**See also:** the muPC residual depth of this architecture (`L = 2d`) is derived in [Initialization and Scaling](05_initialization_and_scaling.md); `examples/transformer_v2_demo.py` (`--mode pc|backprop`, `--tokenizer char|bpe`) for end-to-end training and generation; [Training and Evaluation](08_training_and_evaluation.md) for the autoregressive training API.
