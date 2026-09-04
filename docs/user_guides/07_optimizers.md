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

FabricPC provides two natural gradient transforms in `fabricpc.training.natural_gradients`.
Both divide the gradient by an online diagonal Fisher estimate F, an
exponential moving average (EMA) of the squared gradient, bias-corrected for
the EMA's zero start (`F / (1 - fisher_decay**t)` after `t` steps, as Adam
corrects its second moment). Follow them with `optax.scale(-lr)` to apply a
step size; the presets in `examples/mnist_advanced.py` show calibrated
constants.

**Diagonal Fisher preconditioning**:

```python
from fabricpc.training.natural_gradients import scale_by_natural_gradient_diag

optimizer = optax.chain(
    scale_by_natural_gradient_diag(fisher_decay=0.95, relative_damping=0.1),
    optax.scale(-1e-3),
)
```

One Fisher entry per parameter. More expressive but higher memory.

**Layer-wise Fisher preconditioning**:

```python
from fabricpc.training.natural_gradients import scale_by_natural_gradient_layerwise

optimizer = optax.chain(
    scale_by_natural_gradient_layerwise(fisher_decay=0.95, relative_damping=0.1),
    optax.scale(-1e-3),
)
```

One scalar Fisher estimate per parameter tensor (the EMA of the tensor's mean
squared gradient). Cheaper and more stable for large tensors.

Parameters for both:
- `fisher_decay` — EMA decay for the Fisher estimate, in [0, 1). Default: 0.95
- `relative_damping` — damping as a fraction of the mean Fisher entry, >= 0.
  Default: 0.1
- `damping` — absolute damping added to the denominator, >= 0. Default: 0
  (off). At least one of the two damping terms must be positive.

**Why the damping is relative.** The update `g / F` is covariant: scaling
every gradient by a constant c scales F by c² and the update by 1/c. The
damping reference is r = trace(F) / dim, the mean bias-corrected Fisher entry
over the whole parameter pytree (for the layer-wise transform each tensor's
scalar counts with weight equal to the tensor's size, so both transforms use
the same r), and the update is `g / (F + relative_damping * r)`. Both
denominator terms scale by c², so the update still scales by exactly 1/c and
the set of parameters where damping dominates the Fisher does not depend on
the gradient scale. An absolute damping constant would pin that regime to one
gradient scale: with gradients N times smaller, F shrinks N² times and the
constant silently takes over, turning the transform into plain SGD. With
`damping = 0` no constant is added to the denominator; if every gradient seen
so far is zero the update is zero.

**Limitations of the present implementations.** F is the squared *mean*
gradient of the batch, not a per-sample Fisher, so `g / F` is about `1 / g`:
the entries with the largest gradients get the smallest steps, and the step
grows relative to the gradient as training shrinks it. On the MNIST demo
(`examples/mnist_advanced.py`, a four-layer sigmoid MLP) neither transform left
chance accuracy in 10 epochs with relative damping alone, at any of 48 tried
combinations of `optax.scale`, `relative_damping`, and a global-norm clip,
while `optax.adamw` reaches 97% on the same graph. The demo's `ngd_diag` and
`ngd_layerwise` presets use `relative_damping=0` with an absolute `damping`
that, measured, lies above 96.5% of the Fisher entries from the first step:
those entries are updated as SGD with rate `scale / damping`, and only the few
large-gradient entries receive the natural-gradient step. Even so `ngd_diag`
reaches 24% at 10 epochs and `ngd_layerwise` stays at chance. Use these
transforms as research baselines, not as tuned optimizers. The measurements
are recorded in `docs/dev_plans_archive/mean_gradient_normalization.md`.

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
