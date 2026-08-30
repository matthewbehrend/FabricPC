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

FabricPC provides two natural gradient transforms in `fabricpc.training.natural_gradients`:

**Diagonal Fisher preconditioning**:

```python
from fabricpc.training.natural_gradients import scale_by_natural_gradient_diag

optimizer = optax.chain(
    scale_by_natural_gradient_diag(fisher_decay=0.95, damping=1e-3),
    optax.adam(1e-3),
)
```

Uses an EMA diagonal Fisher approximation. More expressive but higher memory.

**Layer-wise Fisher preconditioning**:

```python
from fabricpc.training.natural_gradients import scale_by_natural_gradient_layerwise

optimizer = optax.chain(
    scale_by_natural_gradient_layerwise(fisher_decay=0.95, damping=1e-3),
    optax.adam(1e-3),
)
```

One scalar Fisher estimate per parameter tensor. Cheaper and more stable for large tensors.

Parameters for both:
- `fisher_decay` — EMA decay for Fisher estimate, in [0, 1). Default: 0.95
- `damping` — Positive damping added to the Fisher. Default: 1e-3

## Practical Guidance

- **Default**: `optax.adamw(1e-3, weight_decay=0.1)` is a good starting point
- **Weight decay**: 0.001–0.1 depending on model size
- **Learning rate**: 1e-3 for Adam/AdamW, 0.01–0.1 for SGD with momentum
- Natural gradient transforms are experimental; useful for research comparisons
