# Optimizers and Chaining

Tutorial-style guide on Optax integration and natural gradient options.

## Optax Basics

FabricPC uses [Optax](https://optax.readthedocs.io/) for gradient-based weight optimization. Any Optax optimizer works:

```python
import optax

optimizer = optax.adam(1e-3)
optimizer = optax.adamw(1e-3, weight_decay=0.1)
optimizer = optax.sgd(0.01, momentum=0.9)
```

Pass the optimizer to `train()`:

```python
from fabricpc.training import train

result = train(
    params=params, structure=structure, train_loader=train_loader,
    optimizer=optimizer, config={"num_epochs": 10}, rng_key=train_key,
)
trained_params = result.params
```

Or manage state manually with a step built by `make_train_step()`:

```python
from fabricpc.training import make_train_step

train_step = make_train_step(structure, optimizer)
opt_state = optimizer.init(params)
params, opt_state, metrics, _ = train_step(params, opt_state, batch, rng_key)
```

## Gradient Scale

The gradients that reach Optax are means per prediction, under both
`algorithm="pc"` and `"backprop"`. The trainer divides the batch-summed
gradients once by the prediction count N, the number of clamped-target
prediction positions in the batch (`batch` for classification, `batch *
seq_len` for token targets). Learning rates, clipping thresholds, Adam
epsilon, and the natural-gradient damping below therefore sit on the same
scale as standard mean-loss training, and they transfer across batch sizes and
sequence lengths. A custom loop that builds its own step gets the same scale
from `fabricpc.training.pc_weight_gradients(params, state, structure, clamps)`,
or from `fabricpc.training.grad_denominator(structure, clamps)` for the count
itself.

## Chaining Transforms

Optax transforms compose via `optax.chain()`:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(1e-3),
)
```

Common patterns:
- Gradient clipping + optimizer
- Learning rate schedule + optimizer
- Weight decay via `optax.adamw()` or explicit `optax.add_decayed_weights()`

## Learning Rate Schedules

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=100, decay_steps=5000,
)
optimizer = optax.adam(schedule)
```

## Natural Gradient Transforms

FabricPC provides two natural-gradient-style transforms in
`fabricpc.training.natural_gradients`. Both divide the gradient by an online
diagonal Fisher estimate F, an exponential moving average (EMA) of the squared
gradient, bias-corrected for the EMA's zero start (`F / (1 - fisher_decay**t)`
after `t` steps, as Adam corrects its second moment). Follow them with
`optax.scale(-lr)` to apply a step size; the presets in
`examples/mnist_advanced.py` carry constants swept on the MNIST demo.

**Diagonal Fisher preconditioning**:

```python
from fabricpc.training.natural_gradients import scale_by_natural_gradient_diag

optimizer = optax.chain(
    scale_by_natural_gradient_diag(fisher_decay=0.95, damping=1e-8),
    optax.scale(-1e-3),
)
```

One Fisher entry per parameter. More expressive but higher memory.

**Layer-wise Fisher preconditioning**:

```python
from fabricpc.training.natural_gradients import scale_by_natural_gradient_layerwise

optimizer = optax.chain(
    scale_by_natural_gradient_layerwise(fisher_decay=0.95, damping=1e-8),
    optax.scale(-1e-3),
)
```

One scalar Fisher estimate per parameter tensor (the EMA of the tensor's mean
squared gradient). Cheaper for large tensors.

Parameters for both:
- `fisher_decay` — EMA decay for the Fisher estimate, in [0, 1). Default: 0.95
- `damping` — positive constant added to the Fisher. Default: 1e-8,
  chosen on the MNIST demo at the per-prediction gradient scale, where
  gradients of about 1e-3 per weight give Fisher entries of about 1e-6.

**What the update is.** With `f` the bias-corrected Fisher entry, the update is
`g / (f + damping)`. Where `damping` dominates `f`, the update is
`g / damping`: SGD with learning rate `scale / damping`. Where `f` dominates, `f` is
about `g²`, because it is built from the squared mean gradient of the batch
rather than from per-sample gradients, so the update is about `1 / g`: the
entries with the largest gradients move least, and the step grows relative to
the gradient as training shrinks it. Neither regime is a natural-gradient step,
and no damping value produces one: a smaller value moves more entries into the
`1 / g` regime, a larger value into SGD. At the default the damping exceeds 95%
of the Fisher entries from the first step of the MNIST demo, so both transforms
act as SGD on almost every parameter; `ngd_diag` reaches 16% and
`ngd_layerwise` stays at chance (10%) at 10 epochs, against 97% for
`optax.adamw`. Treat them as research baselines. The estimator redesign, a
Fisher from per-sample gradients at latents sampled from each node's predictive
distribution, is tracked in [issue 68](https://github.com/trueagi-
io/FabricPC/issues/68).

## Practical Guidance

- **Default**: `optax.adamw(1e-3, weight_decay=0.1)` is a good starting point
- **Weight decay**: 0.001–0.1 depending on model size
- **Learning rate**: 1e-3 for Adam/AdamW. SGD rates depend on the gradient
  scale: on per-prediction PC gradients the MNIST demo's `sgd` preset uses
  `optax.sgd(2.0, momentum=0.9)` with `add_decayed_weights(5e-4)`. A rate
  tuned before the per-prediction normalization (on batch-summed gradients)
  is multiplied by the prediction count N, and a coupled weight decay divided
  by N.
- Natural gradient transforms are experimental; useful for research comparisons
