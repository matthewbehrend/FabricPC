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

**Backprop regime.** One ePC step from ε = 0 leaves ε_t = −eta_infer·∂L/∂z_t exactly, the backprop activation gradient at the feedforward point, so the local weight gradients match backprop's to first order in eta_infer·λ_max(H_ε): hidden layers scaled by eta_infer, the output layer unscaled (Goemaere et al., Theorem C.9; exact for a layer fed only by clamped nodes). H_ε is the Hessian of the total energy in error coordinates. Gradient descent on it splits into independent modes along its eigenvectors, and only the modes along which the starting gradient g0 = ∇_ε E has a component ever move: the excited modes. After T steps a mode with eigenvalue λ has relaxed toward equilibrium by f(λ) = 1 − (1 − eta_infer·λ)^T. The regime is read on the modes that carry the gradient, each weighted by the fraction w of ‖g0‖² it carries: the gradient-weighted relaxed fraction f̄ = Σ w·f(λ) over the positive-curvature modes. f̄ ≪ 0.1 is backprop-like; f̄ > 0.9 is the PC equilibrium, which needs eta_infer·T·λ ≳ 3 on those modes, and on a linear graph they are the eig(S) modes of Innocenti et al.'s Theorem 1 (S = I + JJᵀ, J the map from the hidden errors to the output prediction), so the slowest of them sets T. eta_infer·λ_max < 2, λ_max the largest excited eigenvalue, is required for stability at every T. At odd T the output layer is damaged before that bound: its weight gradient follows the output residual after T steps, which along the top mode is (1 − eta_infer·(λ_max − 1))·r at T = 1 and reverses sign once eta_infer·(λ_max − 1) > 1 (for general odd T, once (1 − eta_infer·λ_max)^T < −1/(λ_max − 1)); the hidden errors keep their sign for every eta_infer·λ < 2, and at even T the residual never reverses.

`fabricpc.core.epsilon_spectrum.epsilon_spectrum(params, state, clamps, structure)` measures the excited spectrum on any graph (Lanczos on Hessian-vector products through `EPCInference.error_energy`, 30 steps, three vectors of ε size in memory): `lambda_max`, `lambda_min`, the Ritz values with their gradient weights, `negative_weight` (the gradient weight on negative-curvature modes, nonzero on nonlinear graphs once the weights are large), and the Ritz residuals. `EPCInference.regime(spectrum)` returns a `Regime` with `unstable` (eta_infer·λ_max > 2), `output_gradient_reverses`, `f_weighted` (f̄), `f_max`, `band` ("backprop-like", "partially relaxed", "near PC equilibrium"), `negative_weight`, and `growth_min` (the T-step growth of the most negative mode); `str(regime)` is the one-line label, and the demos print it at init. The linear oracle in `fabricpc.utils.linear_pc_oracle` gives the same spectrum exactly on linear-Gaussian graphs.

Two optimizer caveats. Under Adam the eta_infer scaling of the hidden-layer gradients is normalized away while eta_infer·|g| ≫ Adam's ε (1e-8), so 1-step ePC with Adam trains as backprop with Adam; below that the ε term damps the hidden layers. Without Adam the hidden layers learn eta_infer times slower than the output layer, a 1000× disparity at the default.

λ_max is a weight-scale quantity: on a chain λ_max = 1 + σ_max(J)² grows with the product of the downstream weights, so it grows during training and a fixed eta_infer can cross either threshold late in a run. Normalization layers and weight decay bound it. Goemaere et al. trained ResNet-18 at eta_infer = 1e-3, T = 5 for 50 epochs without instability, with batch normalization after every convolution, ReLU, weight decay ≤ 1e-3, and a standard parameterization. The muPC ResNet-18 demo (`examples/resnet18_cifar10_demo.py`) has no normalization layers, gelu, weight decay 1e-2, muPC scaling, and a 100-epoch schedule.

Measured on that demo at init on a 64-sample test batch (`scripts/epc_analysis.py --resnet18`): λ_max = 16.4; λ_min = −0.42, so the error Hessian is indefinite already at init, with 1.2% of the gradient weight on negative curvature (growth 1.002 over the defaults' five steps); f̄ = 0.010 at the defaults against f_max = 0.080, so most of the gradient weight sits on modes near the precision floor, far below λ_max. A one-eigenvalue fit of the 2-epoch sweep gives λ_eff ≈ 12, near λ_max: the accuracy transition from the backprop value to the PC plateau follows the relaxation of the top modes, not the gradient-weighted bulk, so on this graph f_max tracks the sweep and f̄ from the init spectrum lags it (cells whose accuracy has left the backprop value still read f̄ < 0.1). The defaults (eta_infer·T = 0.005) read backprop-like at init on both measures. In the 100-epoch runs the defaults trained as backprop for tens of epochs and collapsed to chance by epoch 20, while `infer_steps` 1 and 2 at the same rate reached 76.7% and 75.8%, and eta_infer = 1e-2 collapsed by epoch 10 at every step count; λ_max tracked through the defaults' collapse grew from 16 at init past 2/eta_infer = 2000 in the epoch before accuracy reached chance. The sweep and the 100-epoch numbers were measured with batch-summed weight gradients, before release 0.5.1 divided the gradients by the prediction count; the control runs recorded by `--track_regime` do not predate it.

Pick eta_infer below 1/(λ_max − 1) at odd T and below 2/λ_max at every T, with margin for growth, and track the spectrum during training with `fabricpc.training.RegimeProbe`: a `train` callback that records the spectrum, the regime flags, and the Frobenius norm of every weight every N updates on a fixed probe batch (or the training batch), plus the test accuracy per epoch, to a CSV that `scripts/epc_analysis.py --plot_track` renders. Supplying `iter_callback` forces a device sync on every batch, probed or not. The same script reproduces the regime analysis on linear graphs.

```python
import jax
from fabricpc.core import EPCInference, epsilon_spectrum
from fabricpc.graph_assembly import graph
from fabricpc.training import RegimeProbe, train

inference = EPCInference(eta_infer=1e-3, infer_steps=5, latent_decay=0.0)
structure = graph(..., inference=inference)

spectrum = epsilon_spectrum(params, state, clamps, structure)
print(inference.regime(spectrum))

probe = RegimeProbe(
    structure, probe_clamps, every=50, key=jax.random.PRNGKey(1),
    csv_path="epc_regime_track.csv",
)
result = train(
    params, structure, train_loader, optimizer, {"num_epochs": 30}, rng_key,
    iter_callback=probe.on_iter,
    epoch_callback=lambda ctx: probe.on_epoch(ctx, accuracy=None),
)
print(probe.summary(chance=0.1))
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
| `eta_infer` (EPCInference) | below 1/(λ_max − 1) at odd T, below 2/λ_max at every T | A global rate through the whole transfer function; λ_max is the largest excited eigenvalue of the error Hessian, measured with `epsilon_spectrum` at init and tracked with `RegimeProbe` because it grows with the weights (1e-2 collapsed the resnet18 demo at every step count) |
| `infer_steps` (state-based) |  ~5 * depth   | More steps = better convergence, slower training |
| `infer_steps` (EPCInference) | set by f̄ from `EPCInference.regime` | f̄ ≪ 0.1 is backprop-like (the defaults on resnet18); f̄ > 0.9 is the PC equilibrium, eta_infer·T·λ ≳ 3 on the modes that carry the gradient; one reverse pass per step reaches every layer |
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
