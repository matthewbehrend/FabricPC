# Inference Algorithms API

All inference algorithms extend `InferenceBase` from `fabricpc.core.inference`.

## Overview

Inference is the inner optimization loop of predictive coding. Given fixed weights and clamped data, it iteratively updates latent states to minimize total network energy.

Each inference step has three phases:
1. **Zero gradients** — Reset accumulated latent gradients
2. **Forward pass** — Compute predictions, errors, and accumulate gradient contributions
3. **Update** — Apply the algorithm-specific update rule to the relaxed variables (`z_latent` for the state-based solvers, ε for `EPCInference`)

`run_inference` brackets the step loop with two segment hooks, `begin_segment` and `finalize_state` (identity by default) — a solver's entry adaptation and exit rebuild when its per-step state is not the consumable final state.

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

Error-parameterized predictive coding (ePC, Goemaere et al., arXiv 2505.20137). The prediction error ε is the first-class relaxed variable; each latent is derived by a forward pass along `structure.schedule` as `z_latent = z_mu + ε`. Because every node's `z_mu` depends on all upstream latents, one `jax.value_and_grad` over the ε pytree per step delivers the output-loss signal to every layer unattenuated — a few steps replace sPC's hundreds on deep DAGs. The ε ↔ z_latent map is a volume-preserving bijection: identical energies, identical equilibria, and the final derived state feeds the local weight-gradient path unchanged. (The equivalence relies on the node contract's rule that `predict` never reads `state.z_latent` values — see the custom-nodes guide.)

**Backprop regime.** One ePC step from ε = 0 leaves ε_t = −eta_infer·∂L/∂z_t exactly, the backprop activation gradient at the feedforward point, so the local weight gradients match backprop's to first order in eta_infer·λ_max(H_ε): hidden layers scaled by eta_infer, the output layer unscaled (Goemaere et al., Theorem C.9; exact for a layer fed only by clamped nodes). H_ε is the Hessian of the total energy in error coordinates and λ_max its top eigenvalue, measured on any graph by `fabricpc.utils.linear_pc_oracle.top_epsilon_eigenvalue` (power iteration through `EPCInference.error_energy`) or exactly on linear graphs by that module's oracle. After T steps each excited error mode with eigenvalue λ has relaxed toward equilibrium by 1 − (1 − eta_infer·λ)^T, so the regime parameter is eta_infer·T·λ_max: ≪ 1 is backprop-like; ≳ 3/λ_min,excited reaches the PC equilibrium; and eta_infer·λ_max < 2 is required for stability at every T, T = 1 included (one step lands each mode at eta_infer·λ times its equilibrium value, so a mode with eta_infer·λ > 2 ends farther from equilibrium than it started). `EPCInference.regime_label(lambda_max)` names the regime; the demos print it at init. Under Adam the eta_infer scaling of the hidden-layer gradients is normalized away, so 1-step ePC with Adam trains as backprop with Adam.

Measured on the muPC ResNet-18 demo (`examples/resnet18_cifar10_demo.py`): λ_max(H_ε) = 16.4 at init on a 64-sample batch, and a one-eigenvalue fit of the 2-epoch sweep gives λ_eff ≈ 12, so the defaults (eta_infer·T = 0.005) sit at eta_infer·T·λ_max ≈ 0.08, backprop-like at init. λ_max grows with the weights: in the 100-epoch runs the defaults trained as backprop for tens of epochs and collapsed to chance at epoch 20, while `infer_steps` 1 and 2 at the same rate reached 76.7% and 75.8%, and eta_infer = 1e-2 collapsed by epoch 10 at every step count. Tracking λ_max through those runs (`scripts/epc_analysis.py --track_lambda_max`) shows the mechanism: for the defaults λ_max grew from 16 at init to 3500 in epoch 14, crossing 2/eta_infer, and accuracy reached chance in epoch 15; for eta_infer = 1e-2 with one step the crossing came in epoch 6 and chance in epoch 7. Pick eta_infer below 2/λ_max with margin for growth, or track λ_max during training with `scripts/epc_analysis.py --track_lambda_max N`; the same script reproduces the regime analysis on linear graphs.

```python
from fabricpc.core.inference_epc import EPCInference

inference = EPCInference(eta_infer=1e-3, infer_steps=5, latent_decay=0.0)
structure = graph(..., inference=inference)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eta_infer` | `float` | `1e-3`  | Inference rate on ε — tune like a weight learning rate (see below) |
| `infer_steps` | `int` | `5`     | Number of inference iterations |
| `latent_decay` | `float` | `0.0`   | Weight decay on the relaxed errors |

**Update rule (per step):**
```
derive z_latent = z_mu + error along structure.schedule (clamped nodes keep the clamp, derive error)
latent_grad = d(total energy of in_degree > 0 nodes)/d(error)   # one global reverse pass
error_new = error * (1 - eta * latent_decay) - eta * latent_grad
```

A segment starts with `begin_segment` — one forward pass at the carried latents setting ε := z_latent − z_mu, so relaxation continues exactly from the incoming state (the initializer's output or a previous segment's latents) — and ends with `finalize_state`, one detached derive so the returned state satisfies z_latent = z_mu + ε with energies at the final point.

The ε gradient is taken through the full network's transfer function — a change in one node's ε moves every downstream derived latent — so `eta_infer` must be tuned like a weight learning rate, not like sPC's local per-node rate.

On cyclic graphs, ePC minimizes the unrolled approximation of the graph energy fixed by `graph(..., unroll=U)`; state-based solvers minimize the exact graph energy as-is. Memory: each ePC step's single reverse pass stores activations for the whole derived forward (depth × unroll), backprop-scale rather than sPC's per-node closures.

## InferenceSchedule

Composes solvers as segments per weight update — e.g. a few cheap global ePC steps to near-equilibrium, then sPC refinement on the true arbitrary-graph energy, warm-started from ePC's solution:

```python
from fabricpc.core.inference import InferenceSGD, InferenceSchedule
from fabricpc.core.inference_epc import EPCInference

inference = InferenceSchedule(
    EPCInference(eta_infer=1e-3, infer_steps=2),
    InferenceSGD(eta_infer=0.05, infer_steps=20),
)
```

Chained execution contract:
1. Node states are initialized once, by the graph's configured initializer, before the first segment; no segment re-initializes.
2. Each solver receives `z_latent` exactly as the previous segment (or the initializer) left it, and its `begin_segment` adapts the derived fields to its own parameterization without moving the latents — ePC recomputes ε := z_latent − z_mu at the carried latents, so relaxation continues from the incoming latents rather than from stale ε.
3. The next solver continues from the resulting state (after e.g. ePC's final derive rebuild).

Schedules nest, and `segments()` flattens them for per-step consumers (tracking iterates segments instead of assuming one global step count). A schedule has no single per-step rule, so `inference_step()` and `compute_new_latent()` raise. Under a composed schedule, a tracked `latent_grad_norm` series carries each segment's own gradient semantics — sPC's one-hop dE/dz_latent, ePC's full-forward ε gradient.

## Tuning Guidance

| Parameter | Typical Range | Notes |
|-----------|:-------------:|-------|
| `eta_infer` (state-based) |   0.01–0.2    | A per-node rate; lower for stability, higher for faster convergence |
| `eta_infer` (EPCInference) | below 2/λ_max(H_ε) | A global rate through the whole transfer function; the bound is `top_epsilon_eigenvalue` at init and shrinks as the weights grow (1e-2 collapsed the resnet18 demo at every step count) |
| `infer_steps` (state-based) |  ~5 * depth   | More steps = better convergence, slower training |
| `infer_steps` (EPCInference) | set by eta_infer·T·λ_max | ≪ 1 is backprop-like (the defaults on resnet18); ≳ 3/λ_min,excited reaches the PC equilibrium; one reverse pass per step reaches every layer |
| `latent_decay` |      0.0      | Rarely needed; try 0.001 if latents drift |
| `max_norm` |    0.5–2.0    | For InferenceSGDNormClip; prevents gradient explosions |

For deep networks (>10 layers), consider:
- Increasing `infer_steps` to `max(20, 5 * num_layers)`
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

For more radical changes, override `inference_step()`, `forward_value_and_grad()`, or `run_inference()`. Segment-aware solvers additionally override the boundary hooks `begin_segment()` (entry adaptation, run before the first step) and `finalize_state()` (exit rebuild, run after the last step) — see `EPCInference` for a worked example of both.

## Convenience Function

```python
from fabricpc.core.inference import run_inference

# Run inference using the algorithm stored in structure.config["inference"]
final_state = run_inference(params, initial_state, clamps, structure)
```

This is a convenience wrapper that extracts the inference object from the graph structure and delegates to its `run_inference()` method.
