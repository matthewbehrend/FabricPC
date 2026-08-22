# Inference Algorithms API

All inference algorithms extend `InferenceBase` from `fabricpc.core.inference`.

## Overview

Inference is the inner optimization loop of predictive coding. Given fixed weights and clamped data, it iteratively updates latent states to minimize total network energy.

Each inference step has three phases:
1. **Zero gradients** — Reset accumulated latent gradients
2. **Forward pass** — Compute predictions, errors, and accumulate gradient contributions
3. **Latent update** — Apply the algorithm-specific update rule to z_latent

## InferenceSGD

Standard gradient descent inference: `z -= eta * grad`

```python
from fabricpc.core.inference import InferenceSGD

inference = InferenceSGD(eta_infer=0.05, infer_steps=20, latent_decay=0.0)
structure = graph(..., inference=inference)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eta_infer` | `float` | `0.1` | Inference rate |
| `infer_steps` | `int` | `20` | Number of inference iterations |
| `latent_decay` | `float` | `0.0` | Weight decay on latent states |

**Update rule:**
```
z_new = z * (1 - eta * latent_decay) - eta * latent_grad
```

## InferenceSGDNormClip

SGD inference with per-node gradient norm clipping.

```python
from fabricpc.core.inference import InferenceSGDNormClip

inference = InferenceSGDNormClip(
    eta_infer=0.1, infer_steps=20,
    max_norm=1.0, latent_decay=0.0, eps=1e-8,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eta_infer` | `float` | `0.1` | Inference rate |
| `infer_steps` | `int` | `20` | Number of inference iterations |
| `latent_decay` | `float` | `0.0` | Weight decay on latent states |
| `max_norm` | `float` | `1.0` | Maximum L2 norm per node per sample |
| `eps` | `float` | `1e-8` | Numerical stability constant |

**Update rule:**
```
grad_norm = ||latent_grad||_2  (per sample)
clip_factor = min(1.0, max_norm / (grad_norm + eps))
clipped_grad = latent_grad * clip_factor
z_new = z * (1 - eta * latent_decay) - eta * clipped_grad
```

## EPCInference

Error-parameterized predictive coding (ePC, Goemaere et al., arXiv 2505.20137). The prediction error ε is the first-class relaxed variable; each latent is derived by a forward pass along `structure.schedule` as `z_latent = z_mu + ε`. Because every node's `z_mu` depends on all upstream latents, one `jax.value_and_grad` over the ε pytree per step delivers the output-loss signal to every layer unattenuated — a few steps replace sPC's hundreds on deep DAGs. The ε ↔ z_latent map is a volume-preserving bijection: identical energies, identical equilibria, and the final derived state feeds the local weight-gradient path unchanged.

```python
from fabricpc.core.inference_epc import EPCInference

inference = EPCInference(eta_infer=1e-2, infer_steps=5, latent_decay=0.0)
structure = graph(..., inference=inference)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eta_infer` | `float` | `1e-2` | Inference rate on ε — tune like a weight learning rate (see below) |
| `infer_steps` | `int` | `5` | Number of inference iterations |
| `latent_decay` | `float` | `0.0` | Weight decay on the relaxed errors |

**Update rule (per step):**
```
derive z_latent = z_mu + error along structure.schedule (clamped nodes keep the clamp, derive error)
latent_grad = d(total energy of in_degree > 0 nodes)/d(error)   # one global reverse pass
error_new = error * (1 - eta * latent_decay) - eta * latent_grad
```

The ε gradient is taken through the full network's transfer function — a change in one node's ε moves every downstream derived latent — so `eta_infer` must be tuned like a weight learning rate, not like sPC's local rate: sPC's typical 0.05–0.1 can overshoot the minimum along the global gradient. The default 1e-2 comes from the measured resnet18/CIFAR-10 convergence (`examples/epc_spc_resnet18_compare.py --mode convergence`): reaching sPC-120's final total energy took 105 steps at 1e-3, 12 at 1e-2, and 5 at 3e-2.

On cyclic graphs, ePC minimizes the unrolled approximation of the graph energy fixed by `graph(..., unroll=U)`; state-based solvers minimize the exact graph energy as-is. Memory: each ePC step's single reverse pass stores activations for the whole derived forward (depth × unroll), backprop-scale rather than sPC's per-node closures.

## InferenceSchedule

Composes solvers as segments per weight update — e.g. a few cheap global ePC steps to near-equilibrium, then sPC refinement on the true arbitrary-graph energy, warm-started from ePC's solution:

```python
from fabricpc.core.inference import InferenceSGD, InferenceSchedule
from fabricpc.core.inference_epc import EPCInference

inference = InferenceSchedule(
    EPCInference(eta_infer=1e-2, infer_steps=5),
    InferenceSGD(eta_infer=0.05, infer_steps=20),
)
```

Chained execution contract:
1. Node states are initialized once, by the graph's configured initializer, before the first segment; no segment re-initializes.
2. Each solver receives `z_latent`, `z_mu`, and `error` exactly as the previous segment (or the initializer) left them — no resync at the boundary.
3. The next solver continues from the resulting state (after e.g. ePC's final derive rebuild).

Schedules nest, and `segments()` flattens them for per-step consumers (tracking iterates segments instead of assuming one global step count). A schedule has no single per-step rule, so `inference_step()` and `compute_new_latent()` raise. Under a composed schedule, a tracked `latent_grad_norm` series carries each segment's own gradient semantics — sPC's one-hop dE/dz_latent, ePC's full-forward ε gradient.

## Tuning Guidance

| Parameter | Typical Range | Notes |
|-----------|:------------:|-------|
| `eta_infer` | 0.01–0.2 | Lower for stability, higher for faster convergence |
| `infer_steps` | 10–50 | More steps = better convergence, slower training |
| `latent_decay` | 0.0 | Rarely needed; try 0.001 if latents drift |
| `max_norm` | 0.5–2.0 | For InferenceSGDNormClip; prevents gradient explosions |

For deep networks (>10 layers), consider:
- Increasing `infer_steps` to `max(20, 4 * num_layers)`
- Using `InferenceSGDNormClip` for stability

## Creating Custom Inference Algorithms

Subclass `InferenceBase` and implement `compute_new_latent()`:

```python
from fabricpc.core.inference import InferenceBase
import jax.numpy as jnp

class InferenceMomentum(InferenceBase):
    def __init__(self, eta_infer=0.1, infer_steps=20, momentum=0.9):
        super().__init__(eta_infer=eta_infer, infer_steps=infer_steps, momentum=momentum)

    @staticmethod
    def compute_new_latent(node_name, node_state, config):
        eta = config["eta_infer"]
        momentum = config["momentum"]
        # Your custom update rule here
        # Example: add momentum tracking via node_state auxiliary fields
        return node_state.z_latent - eta * node_state.latent_grad
```

For more radical changes, override `inference_step()`, `forward_value_and_grad()`, or `run_inference()`.

## Convenience Function

```python
from fabricpc.core.inference import run_inference

# Run inference using the algorithm stored in structure.config["inference"]
final_state = run_inference(params, initial_state, clamps, structure)
```

This is a convenience wrapper that extracts the inference object from the graph structure and delegates to its `run_inference()` method.
