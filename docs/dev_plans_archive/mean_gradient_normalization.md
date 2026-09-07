# Mean gradient normalization with a global prediction-count denominator

## Context

The PC training path handed the optimizer batch-summed weight gradients
(`compute_local_weight_gradients` sums each node's per-sample energy over the
batch), while the backprop path divided its objective by `batch_size`. The
same learning rate, Adam ε, or clipping threshold therefore meant different
things across `algorithm="pc"` and `"backprop"`, and gradient scale changed
with batch size and sequence length. This change normalizes both paths to
mean gradients over one global denominator that is correct for image
minibatches, timeseries targets, and language-model token objectives, is safe
under data-parallel sharding, and keeps the discipline that avoids the
gradient-accumulation defect (per-microbatch means averaged over microbatches
with unequal token counts overweight short microbatches): gradients stay sums
everywhere; division by a global count happens exactly once, in one function.

Because the change moves every gradient to a new scale, it also owns the
optimizer settings in this repository whose behavior depends on that scale:
the coupled-L2 SGD preset, the clipping threshold in the transformer demo, and
the `damping` default of the two natural-gradient transforms, whose Fisher
estimate scales with the square of the gradient. Re-choosing that default on
the new scale exposed a quality gap in the existing transforms: their Fisher
is the squared mean gradient of the batch, so no damping value gives a
natural-gradient step. This change documents that gap and ships a
better-informed default rather than a new estimator; the estimator redesign
and its requirements are recorded in GitHub issue 68.

## Symbols

| Symbol | Meaning |
|---|---|
| B | Batch size, the leading axis of every clamp |
| S | Prediction positions per sample: the product of a target clamp's non-batch, non-class axes (sequence length for token targets, 1 for classification) |
| N | Prediction count, the single denominator: total clamped-target prediction positions in the batch, Σ over target heads of B·S |
| η | `eta_infer`, the latent step size in `InferenceSGD` |
| g | One weight-gradient leaf as handed to optax, a mean per prediction |
| f | One Fisher leaf in a natural-gradient transform: the exponential moving average (EMA) of g² per entry (`scale_by_natural_gradient_diag`) or of mean(g²) per leaf (`scale_by_natural_gradient_layerwise`) |
| t | Optimizer step count held in the transform state (`count`), for the EMA bias correction f̂ = f / (1 − fisher_decay^t) |

## Design

### Denominator rule

N is the total number of clamped-target prediction positions,
`sum(prod(clamps[name].shape[:-1]) for name in target_nodes)`, where a target
node is a clamped node with `in_degree > 0` (`_target_node_names`). The
trailing axis of a target clamp is the class axis: `build_clamps` validates
every target clamp against `(batch, *node.shape)`, so rank ≥ 2 holds. A rank-2
image target `(B, C)` gives N = B; a rank-3 token target `(B, S, V)` gives
N = B·S. With several target heads N is the sum of their positions (two
same-shape heads give N = 2B), so adding a head halves the step every shared
parameter takes at a fixed learning rate. With no clamped target
(associative-memory graphs) N = B, read from the leading axis of the first
clamp. Read per sample, N is the weight `metrics._internal_energy_fn` assigns
in `evaluate()`: the sum of `_predictions_per_sample` over the batch's target
keys. Dividing the whole gradient pytree by one scalar
leaves relative layer scaling untouched; it redefines the objective as total
energy per prediction.

Definition caveat, recorded in the helper docstring: a clamped input node that
receives feedback edges has `in_degree > 0` and counts as a target. No graph in
the repository does this (the cyclic and lateral MNIST demos leave `pixels`
edge-free on the input side).

### Placement: sum mechanics, one mean function

`compute_local_weight_gradients` keeps returning batch-summed gradients. Sums
are associative, so they survive sharding and any future microbatch
accumulation unchanged. Two public functions in trainer.py own the mean:

- `grad_denominator(structure, clamps) -> int`: the rule above. Empty
  `clamps` raises `ValueError` (nothing is clamped, so there is no objective).
- `pc_weight_gradients(params, state, structure, clamps) -> GraphParams`:
  `compute_local_weight_gradients(...)` with every leaf divided by
  `grad_denominator(structure, clamps)`.

`_batch_grads` calls `pc_weight_gradients` on the PC path and divides the
backprop objective by the same `grad_denominator`. `train_step_with_history`
calls `pc_weight_gradients`. These are the only two optimizer-feeding sites in
the package; no example rolls its own step. Both functions, and
`batch_size_of(batch, structure)` (the leading-axis size of the first
task-mapped batch key, which the dashboarding step shares with the trainer),
are exported from `fabricpc.training` so custom loops have a normalized entry
point. The trainer, not learning.py, knows the clamps, which is why the
division lives here.

### Metrics: `energy` is the objective

`energy` reports the quantity the optimizer descends, for both algorithms:
PC reports `graph_energy` over all `in_degree > 0` nodes divided by N;
backprop reports `graph_energy` over the target nodes divided by N, which
equals `target_energy`. `target_energy` is unchanged (target-node energy over
N). For rank-2 targets every number equals the previous release's. For
sequence targets `energy` becomes per token, on the same scale as
`target_energy` and `perplexity`. In `evaluate()`, the default `energy`
metric (`_internal_energy_fn`) weights each sample by its target prediction
count, S summed over the batch's target keys (1 when the batch carries no
target key), so the eval `energy` of a PC model is per prediction as well.

### Sharding safety

Data parallelism is `jax.jit` + `NamedSharding` over a `"data"` mesh axis.
`train` and `make_train_step` place the batch under `NamedSharding` before the
jitted step, so the step traces on global logical shapes: the static shape
product N is the global count and the batch-summed gradients of the
replicated parameters are an XLA all-reduce. No collective is needed. A future
`shard_map`/`pmap` port would see per-shard shapes and must `jax.lax.psum` the
count over the data axis.

### Microbatching (not applicable yet)

The train loop performs one optimizer update per loader batch; no accumulation
exists, and no padding or validity masks enter any energy (the only masks are
attention masks). Static shape counts are exact today. The rule for a future
accumulation path is one sentence in the `grad_denominator` docstring: sum
first, divide once by the window's total count.

### Clipping thresholds

`optax.clip_by_global_norm` thresholds now sit on the per-prediction scale that
standard mean-loss training uses. `examples/transformer_demo.py` keeps
`clip_by_global_norm(0.8)`: before the change the PC summed-gradient norm over
B·S = 16384 positions exceeded 0.8 on essentially every step, so the clip acted
as per-step normalization ahead of Adam; after the change it fires on the
conventional schedule. Outcome (Verification, item 3): backprop is unchanged;
PC at the old `--lr 1e-4` destabilizes after about 3500 steps without that
per-step normalization, and `--lr 3e-5` holds the full epoch at a slightly
better perplexity, so the demo's PC default moves to 3e-5.

### SGD with coupled L2

`optax.chain(add_decayed_weights(wd), sgd(lr))` applies `lr·(g_sum + wd·θ) =
lr·N·(g_mean + (wd/N)·θ)`. The `mnist_advanced.py` `sgd` preset (B = 200) is
rescaled exactly to `lr = 2.0`, `wd = 5e-4`, with a comment stating these are
lr·N and wd/N of the summed-gradient values.

### Natural-gradient transforms: reduced route

Both transforms in natural_gradients.py compute `g / (f + damping)`. Dividing
g by N divides f by N², so a `damping` tuned on summed gradients sits N²
higher relative to f on the new scale: at per-prediction gradients of about
1e-3 per weight, f is about 1e-6 and the parent default `damping = 1e-3`
dominated every entry. The update has two regimes. Where `damping` dominates
f the update is `g / damping`, SGD with rate `scale / damping`. Where f
dominates, f ≈ g² because it is built from the squared *mean* gradient of the
batch rather than from per-sample gradients, so the update is about `1 / g`:
the entries with the largest gradients move least, and the step grows
relative to the gradient as training shrinks it. Neither regime is a
natural-gradient step, and no damping value produces one; choosing `damping`
chooses the regime. This is a defect of the existing estimator, present
before this change; the rescale made it visible.

What ships:

1. Bias-corrected Fisher: f̂ = f / (1 − fisher_decay^t), with t the step
   count held in a new `count` field of both state tuples. Without it the
   first steps see f ≈ (1 − fisher_decay)·g² and an update about 20× too
   large, which the parent's absolute damping largely masked.
2. `damping` default 1e-8 (`DEFAULT_DAMPING`), the best 10-epoch value in a
   sweep of `examples/mnist_advanced.py` on the per-prediction scale
   (Verification, item 2). At that value the damping exceeds 95% of the
   bias-corrected Fisher entries at step 1 and 99.7% after one epoch, so both
   transforms act as SGD on almost every parameter.
3. The two regimes, the estimator defect, the bias correction, and the issue
   68 link are stated in the module docstring, the optimizers guide, the
   CHANGELOG, and the demo preset comment. The `ngd_diag` and `ngd_layerwise`
   presets use the swept constants.
4. Signatures are unchanged (`fisher_decay=0.95, damping=DEFAULT_DAMPING`);
   `_validate_hparams` keeps `damping > 0`. The `count` field means optimizer
   states saved under 0.5.0 with either transform do not restore.

GitHub issue 68 was updated with the learnings from these measurements and
the requirements for the estimator that would make the transforms
natural-gradient steps: a diagonal Fisher from per-sample gradients at latents
drawn from each node's predictive distribution, generic through `NodeBase`
with no per-node closed forms.

## Alternatives considered

- **Divide inside `compute_local_weight_gradients`** (pass N in). Pros: no
  caller can forget. Cons: the denominator is a property of the objective,
  which learning.py does not know; inspection-only callers would supply a
  meaningless constant or an optional default would creep in. Rejected;
  `pc_weight_gradients` in the trainer gives the same guarantee where the
  clamps are known.
- **Divide at each optimizer-feeding call site** via `grad_denominator`
  alone. Pros: smallest surface. Cons: the contract is enforced by docstring;
  `train_step_with_history` was already a hand copy of the PC step that
  diverged on how it read the batch size. Rejected.
- **`jnp.mean` inside node `forward_and_weight_grads`.** Pros: locality.
  Cons: touches every node template and per-sample is the wrong denominator
  for token objectives. Rejected.
- **Per-sample denominator (`batch_size`) everywhere.** Pros: smallest diff.
  Cons: the learning rate does not transfer across sequence lengths. Rejected.
- **Keep PC `energy` per sample.** Pros: no metric change. Cons: the unified
  trainer would report `energy` with a different normalization per algorithm
  for sequence targets, and PC `energy` would no longer be the optimized
  quantity. Rejected.
- **Sum of per-head means for multi-head graphs** (Σ_k L_k / (B·S_k)). Pros:
  the common multi-task convention. Cons: the PC internal energy of a shared
  hidden node has no per-head split, so only one global scalar is
  well-defined. Rejected; the sum-over-heads count is documented instead.
- **Remove or retune the transformer clip.** Pros: fewer moving parts. Cons:
  0.8 to 1.0 on a per-token gradient is the standard LM recipe; removing it
  would itself change dynamics. Rejected in favor of keeping the value and
  measuring.
- **Relative damping for the natural-gradient transforms**, `g / (f̂ + ρ·r)`
  with r = trace(F̂)/dim (the mean of all bias-corrected Fisher entries across
  the parameter pytree) and ρ a fraction of it, pinned by the covariance
  property `update(c·g) = update(g) / c`. Built and measured first. Pros: the
  regime becomes independent of gradient scale, so a later change of batch
  size or sequence length cannot re-break it. Cons: the squared-mean Fisher
  gives `g / f̂ ≈ 1 / g`, and relative damping holds that ratio at every
  scale, so the update grows as the gradient shrinks; 48 MNIST runs stayed at
  chance (Verification, item 2), the review reproduced the non-convergence on
  a 20-dimensional convex quadratic, and every preset had opted out through a
  restored absolute `damping`. Reverted. The design is correct once F is a
  Fisher and is carried in issue 68.
- **Exact rescale of the parent natural-gradient presets**
  (`damping = 1e-3 / N²`, `scale / N`). Pros: reproduces the parent's
  trajectory (measured: `ngd_diag` 23.78% against the parent's 25.25% at 10
  epochs, the residual being the bias correction). Cons: keeps a default that
  acts as SGD on 96.5% of the entries while presenting itself as a
  natural-gradient step, and no user depends on the old values. Rejected; the
  default is re-chosen by a sweep on the new scale instead.
- **Per-sample Monte-Carlo diagonal Fisher** (draw z̃ from each node's energy
  functional at its prediction `z_mu`, vmap `forward_and_weight_grads` at
  batch 1, square and sum). Pros: the true Fisher of each node's likelihood;
  for Gaussian energy with precision π it is π·(∂μ/∂W)², the Gauss-Newton
  diagonal, bounded below and independent of the residual. Cons: a new
  `EnergyFunctional.sample` method, a `fisher=True` trainer flag, and a
  calibration run; too large for this PR. Deferred to issue 68, which holds
  the design and requirements.
- **`g / sqrt(F)`.** Pros: bounded step. Cons: that is `optax.scale_by_rms`
  under another name; the transforms would then be deleted, not kept.
  Rejected.

## Changes

**fabricpc/training/trainer.py**
1. Add `grad_denominator(structure, clamps)` and `pc_weight_gradients(params,
   state, structure, clamps)`; reuse `_target_node_names`. Rename the private
   batch-size helper to `batch_size_of` and export it.
2. `_batch_grads`: `denom = grad_denominator(structure, clamps)` once. PC
   path: `grads = pc_weight_gradients(...)`, `energy = graph_energy(state,
   structure) / denom`. Backprop path: objective `graph_energy(...,
   node_names=target_nodes) / denom`. `target_energy` divides by it.
3. Module docstring: the sub-step 4 rows (`/ N` for both), the sub-step 5 PC
   row (`pc_weight_gradients`, summed gradients / N), and a paragraph under
   the table defining N; the metrics comment (both keys per prediction;
   `energy` remains algorithm-dependent in node set); and `make_train_step`'s
   `"energy"` description.

**fabricpc/training/__init__.py**: import and export `grad_denominator`,
`pc_weight_gradients`, `batch_size_of`.

**fabricpc/training/natural_gradients.py**: both state tuples gain an int32
`count`; `update_fn` divides by the bias-corrected Fisher (`_bias_corrected`,
factor `1 - fisher_decay**count`); `DEFAULT_DAMPING = 1e-8` replaces the 1e-3
default in both signatures; the module docstring states the gradient scale,
the two regimes, the estimator defect, the bias correction, and the issue 68
link. `_validate_hparams` is unchanged.

**fabricpc/training/metrics.py**: split `_target_items` into a non-raising
iterator and the raising wrapper the target metrics use. `_internal_energy_fn`
weight = Σ over target items of `_predictions_per_sample(y)`, or 1 with no
target key. Update the module docstring's description of `energy`.

**fabricpc/utils/dashboarding/inference_tracking.py**:
`train_step_with_history` reads B via `batch_size_of(batch, structure)`,
computes `denom = grad_denominator(structure, clamps)`, reports
`energy = graph_energy(final_state, structure) / denom`, and feeds
`pc_weight_gradients(...)` to `optimizer.update`.

**fabricpc/core/learning.py**: docstring only. State the contract (returns
batch-summed gradients; the trainer's `pc_weight_gradients` divides once by
`grad_denominator`).

**examples/mnist_advanced.py**: `sgd` preset `add_decayed_weights(5e-4)`,
`sgd(2.0, momentum=0.9)` with the rescale comment. `ngd_diag` and
`ngd_layerwise` presets: `add_decayed_weights(5e-4)`, the default-argument
transform, and `optax.scale(-6e-7)` / `optax.scale(-2e-6)` (scale / damping =
60 and 200), with a comment recording the regime fractions, the sweep range,
and the 10-epoch accuracies against `adamw`. A `--num_epochs` argument
(default 10) makes the runs below reproducible.

**examples/transformer_demo.py**: the clip stays at 0.8. The `--lr` default
becomes mode-dependent (`None` resolved to 3e-5 for `pc`, 1e-4 for
`backprop`) after the verification below; the energy comment in the training
loop and the docstring `Results:` block (PC energy now per token) are
updated.

**Docs**
- `docs/user_guides/08_training_and_evaluation.md`: objective rows and the
  Gaussian sentence ("per sample" → per prediction, N); the "Per prediction"
  paragraph defining N with the multi-head sentence; the `energy` bullet and
  the eval-metric table row (the objective per prediction, node set still
  algorithm-dependent); delete the "different normalizations" note.
- `docs/user_guides/09_experiment_tracking.md`: the energy comment becomes
  "energy is the objective per prediction (graph_energy over internal nodes /
  prediction count)".
- `docs/user_guides/06_custom_nodes.md`: add that the trainer divides the
  summed gradients once by the prediction count (`pc_weight_gradients`).
- `docs/user_guides/03_how_predictive_coding_works.md`: the sentence naming
  `compute_local_weight_gradients` as the gradient source gains the division
  step. `examples/mnist_aim_tracking.py` carried the same stale "per-sample /
  batch_size" comment as the tracking guide and is updated with it.
- `docs/user_guides/07_optimizers.md`: new section "Gradient Scale" after
  Optax Basics stating that gradients reaching optax are means per prediction,
  so learning rates, clipping thresholds, Adam epsilon, and the
  natural-gradient damping are on the same scale as standard mean-loss
  training. The "Natural Gradient Transforms" section is rewritten: the bias
  correction, `optax.scale(-lr)` after the transform in both examples (the
  old examples chained `optax.adam`), the `damping` default 1e-8 with the
  scale it was chosen on, a "What the update is" paragraph on the two
  regimes with the measured MNIST accuracies, and the issue 68 link. The
  Practical Guidance learning-rate bullet gives the per-prediction SGD
  constants and the lr·N, wd/N rule.
- `CHANGELOG.md`: an `[Unreleased]` entry with a four-row migration table
  (custom loops, SGD-family rates, the `energy` semantics, and saved
  natural-gradient optimizer states, which gain `count` and do not restore
  from 0.5.0) and a "New" list (the three exported functions, the
  natural-gradient bias correction and damping default, the eval `energy`
  weighting, the dashboarding step's parity, the demo changes).

## Test updates

- `tests/test_trainer.py` `test_pc_parity_hand_rolled_reference`:
  `reference_step` uses `pc_weight_gradients`. Reference-loss comments say
  "/ prediction count". `test_eval_energy_matches_graph_energy` docstring
  says `/ N`.
- Unaffected: direct `compute_local_weight_gradients` calls with shape, sign,
  or finiteness assertions (test_fabricpc.py, test_mupc.py,
  test_storkey_hopfield.py); metric-shape tests; loss-decrease tests;
  Adam-based step tests.
- New tests, `tests/test_trainer.py`:
  1. Backprop objective with a rank-3 target divides by B·S
     (`v1_masked_structure` + `make_v1_token_batch`).
  2. One-step sPC vs backprop on `classification_structure` (single hidden
     layer, `FeedforwardStateInit`, `infer_steps=1`, identical params):
     hidden-node weight gradients equal η × backprop gradients to 1e-5
     relative. Output-node gradients differ by O(η) because
     `compute_local_weight_gradients` re-evaluates the output prediction at
     the moved hidden latent: assert relative deviation below 10·η and that it
     shrinks as η is divided by 10.
  3. Target-free PC graph: N = B; gradients equal the raw sums / B and
     `energy` equals `graph_energy / B`.
  4. `grad_denominator(structure, clamps)` equals B × Σ weights from
     `_internal_energy_fn` for a sequence batch and a two-target batch (pins
     train/eval agreement and the sum-over-heads convention).
  5. PC `energy` on `v1_masked_structure` equals `graph_energy / (B·S)`.
  6. `grad_denominator` raises on empty clamps.
- New test file `tests/test_inference_tracking.py`: `train_step_with_history`
  matches `make_train_step` under `optax.sgd(1.0)` (params within 1e-5,
  energy equal to `graph_energy / N`), so the hand-copied step cannot drift
  from the trainer's normalization again.
- `tests/test_optimizers.py`: `NGD_TRANSFORMS` parametrizes over both
  transforms; `test_natural_gradients_work_in_train_step` runs the
  default-argument transforms through `make_train_step`; the validation test
  adds a negative `damping`. New:
  1. `test_natural_gradient_first_step_is_bias_corrected`: after one step
     from a fresh state f̂ equals g² (diag) or mean(g²) per leaf (layerwise),
     so the update is `g / (f̂ + damping)` and differs from the uncorrected
     `g / (0.05·f̂ + damping)`; `count == 1`.
  2. `test_damping_dominated_update_is_sgd`: gradients of order 1e-3 against
     `damping = 1.0` give `g / damping` on both transforms.
  3. `test_fisher_dominated_diag_update_is_inverse_gradient`: gradients of
     order 1 against `damping = 1e-12` give `1 / g` to 1e-4 relative, pinning
     the documented defect.

## Verification

```
pytest tests/test_trainer.py tests/test_inference_tracking.py tests/test_optimizers.py \
       tests/test_mupc.py tests/test_fabricpc.py tests/test_storkey_hopfield.py \
       tests/test_transformer_nodes.py tests/test_sharding.py
```

**Result** (CPU, `JAX_PLATFORMS=cpu`, after the `batch_size_of` export fix):
the listed files give 142 passed, 5 skipped; the full `pytest tests/` run is
recorded in item 5.

Then:

1. ResNet-18 demo, one default 2-epoch run on the new code, compared with the
   demo's own `Results:` block (train energy 0.4792, test accuracy 33.71%).
   AdamW normalizes a uniform gradient scale (decoupled weight decay; only
   ε-level effects), so the numbers should agree to within the run-to-run
   variation the docstring already states.

   **Result** (`python examples/resnet18_cifar10_demo.py`, RTX 3090 shared
   with another job): test accuracy 33.58%, 432 s per epoch, against the
   docstring's 33.71%. Within the stated variation; the docstring block
   stands.
2. `mnist_advanced.py` `sgd` preset: the exact rescale must reproduce the
   parent's trajectory.

   **Result (RTX 3090, JAX 0.10.1, optax 0.2.8).** Parent commit, `sgd`
   (lr 0.01, wd 0.1): epoch 2 energy 0.4517 / accuracy 10.28%, epoch 10
   0.3245 / 24.78%. New code with lr 2.0, wd 5e-4: epoch 1 energy 0.4811 /
   9.58%, epoch 2 0.4517 / 10.28%, identical to the parent to the printed
   digits. `adamw` reaches 0.0127 / 97.27% at 10 epochs on both.

   Natural-gradient presets, same machine. Parent commit, energy / accuracy:

   | preset | epoch 2 | epoch 10 |
   |---|---|---|
   | `ngd_diag` (damping 1e-3, scale 3e-4) | 0.5017 / 8.92% | 0.1786 / 25.25% |
   | `ngd_layerwise` (damping 1e-3, scale 1e-3) | 0.4513 / 10.28% | 0.4514 / 9.74% |

   Every parent preset, `sgd` included, sits at chance after 2 epochs.

   Relative damping (the reverted design, `g / (f̂ + ρ·r)`): 48 runs, none
   left chance accuracy. `optax.scale` ∈ {1e-8, ..., 1e-3} at ρ = 0.1 for 2
   epochs gave 8.9 to 11.4% (the old constants 3e-4 and 1e-3 diverged, energy
   rising to 1.15 and 3.55); ρ ∈ {0.1, 1, 10} × scale ∈ {1e-6, 1e-5, 1e-4}
   for 10 epochs gave 9.6 to 11.4% with energy plateaued at 0.45 (every
   output near 0.1); the same grid with `clip_by_global_norm(1.0)` before
   `scale` ∈ {0.03, 0.1, 0.3, 1.0} gave the same. Mechanism, logged on the
   diagonal transform (ρ = 1, scale 1e-5): over 250 steps the per-prediction
   gradient norm fell from 1.29 to 0.05 while the update norm stayed between
   0.04 and 0.28, so the step grew relative to the gradient as the fit
   improved.

   Exact rescale of the parent presets (`damping = 1e-3 / N²`, `scale / N`,
   N = 200, with the bias correction), energy / accuracy at epoch 5 and 10:
   `ngd_diag` 0.2761 / 24.12% and 0.1784 / 23.78% against the parent's
   0.2632 / 24.05% and 0.1786 / 25.25%; `ngd_layerwise` identical to the
   parent (0.4514 / 9.58%, 0.4514 / 9.74%). Regime on `ngd_diag`, fraction
   of bias-corrected Fisher entries below the damping and their share of
   trace(F): step 1, 96.5% and 0.01%; step 50, 97.9% and 0.00%; step 300,
   99.7% and 0.00%. The parent presets trained weakly, and `ngd_layerwise`
   not at all, because almost every entry was already updated as SGD.

   Damping sweep on the shipped code, 68 runs, `damping` and `scale / damping`
   as the coordinates. Full grid at 2 epochs, both transforms: `damping` ∈
   {1e-8, 1e-7, 1e-6, 1e-5, 1e-4} × `scale / damping` ∈ {20, 60, 200} (30
   runs, every one at chance, as the parent's presets were at 2 epochs). Then
   38 runs at 10 epochs: `damping` ∈ {1e-7, 1e-6, 1e-5, 1e-4} × `scale /
   damping` ∈ {200, 600, 2000} for both transforms, plus `damping` ∈ {1e-8,
   2.5e-8, 3e-8, 1e-7} at `scale / damping` 60 to 600. Only the diagonal
   transform with `damping` ≤ 2.5e-8 and `scale / damping` ≤ 200 leaves
   chance (12.3 to 16.3%); the layer-wise transform stays at chance in every
   run; training energy falls in many settings without accuracy following.
   Chosen: `damping = 1e-8` for both, `scale = 6e-7` for `ngd_diag` (ratio
   60) and `2e-6` for `ngd_layerwise` (ratio 200). At 10 epochs: `ngd_diag`
   0.2208 / 16.27%, `ngd_layerwise` 0.4509 / 10.28%, against `adamw`'s
   0.0127 / 97.27%. At the default the damping exceeds 95.4% of the 242,762
   bias-corrected Fisher entries at step 1 and 99.7% after one epoch, and the
   entries above it hold essentially all of trace(F). The `ngd_diag` accuracy
   is below the parent preset's 25.25%; the default was chosen for the best
   result available on the new scale, not to reproduce a preset whose
   damping acted as SGD on 96.5% of the entries.
3. `transformer_demo.py` in `--mode pc` and `--mode backprop` at the same
   budget on the parent commit and after; record eval perplexity for both. A
   PC-mode regression is addressed by retuning the demo's `--lr` default, not
   by changing the clip or the normalization.

   **Results (default budget, 1 epoch = 7841 batches of 128 x 128 tokens,
   `--lr 1e-4`, RTX 3090 shared with another job).** The parent PC run
   reproduces the demo docstring exactly.

   | mode | parent: final train energy / test loss / perplexity | new: final train energy / test loss / perplexity |
   |---|---|---|
   | backprop | 1.7140 (per token) / 1.8844 / 6.58 | 1.7136 (per token) / 1.8846 / 6.58 |
   | pc | 352.9587 (per sample = 2.7575 per token) / 2.6988 / 14.86 | 108.0933 (per token) / 4.7353 / 113.90 |

   Backprop is unchanged: its objective moved from `/ B` to `/ (B * S)`, a
   uniform factor that Adam removes, leaving ε-level differences. PC
   regressed: the energy per token rose 40x and the perplexity 7.7x. The
   mechanism is the clip regime named above: on the parent the summed
   gradient's norm exceeded 0.8 on every step, so `clip_by_global_norm(0.8)`
   normalized each step's gradient to a fixed norm before Adam; on the new
   scale the clip is inactive and Adam sees the raw per-token gradients.
   Energy per token along the two full runs (tqdm postfix, parent divided by
   128):

   | step | 100 | 1000 | 1500 | 2500 | 3500 | 4000 | 4500 | 5000 | 6500 | 7800 |
   |---|---|---|---|---|---|---|---|---|---|---|
   | parent | 3.399 | 2.318 | 2.387 | 2.386 | 2.480 | 2.639 | 2.715 | 2.812 | 2.711 | 2.778 |
   | new, lr 1e-4 | 3.399 | 2.340 | 2.422 | 2.566 | 3.111 | 10.29 | 53.47 | 181.0 | 719.8 | 720.8 |

   The runs coincide for about 1000 steps, drift apart slowly, and the new
   run leaves the parent's band after step 3500. At a short budget the two
   agree: `--num_epochs 0.1` (784 steps) gives parent test loss 3.0282 /
   perplexity 20.66 (final train energy 337.57 per sample = 2.637 per token)
   and new 3.0330 / 20.76 (2.636 per token), so the divergence is a
   late-training instability, not a change in the early dynamics.
   Short-budget `--lr` sweep on the new code (`--num_epochs 0.1`, test
   loss / perplexity): 1e-4 gives 3.0330 / 20.76, 3e-5 gives 3.6498 / 38.47,
   1e-5 gives 4.4815 / 88.37; 1e-4 with the clip removed (diagnostic only)
   gives 3.1065 / 22.34, so the 0.8 clip still fires on some steps at the
   per-token scale. Lower rates only slow the early phase; whether they hold
   the late phase is measured at the full budget below.
   Full budget on the new code (`--mode pc`, 1 epoch; test loss /
   perplexity / final train energy per token):

   | `--lr` | test loss | perplexity | final train energy per token | trajectory |
   |---|---|---|---|---|
   | 1e-4 (old default) | 4.7353 | 113.90 | 108.09 | diverges after step 3500 |
   | 3e-5 | 2.6846 | 14.65 | 2.2656 | monotone: 2.72 at step 1000, 2.29 at 5000 |
   | 1e-5 | 2.9998 | 20.08 | 2.5541 | stable, under-trained |
   | parent, 1e-4 | 2.6988 | 14.86 | 2.7575 | bounded, 2.3 to 3.2 |

   Resolution: the demo's `--lr` default becomes mode-dependent, 3e-5 for
   `pc` and 1e-4 for `backprop` (the backprop run at 1e-4 reproduced its
   number), and the docstring `Results:` block records the 3e-5 run. The
   clip stays at 0.8. The shared-GPU timings above are not comparable to the
   docstring's; the perplexities are.
4. Demo learning rates for Adam and AdamW stay as tuned, with one exception
   established by item 3: the transformer demo's PC default moves from 1e-4
   to 3e-5 because the clip no longer normalizes every step. The ResNet
   (AdamW, item 1) and every MNIST Adam/AdamW preset are unchanged.
5. Full test suite and linters on the final tree.

   **Result** (CPU, `JAX_PLATFORMS=cpu pytest tests/`): 408 passed, 5
   skipped. `ruff check` and `black --check` on `fabricpc/training/__init__.py`
   pass.

## Revisions

The first implementation shipped the relative-damping design for the
natural-gradient transforms (the first Alternatives entry on them above). A
review of the branch found the prediction-count normalization correct, placed
where the clamps are known, and pinned by the tests listed above, and found
the relative-damping default unable to converge: the squared-mean Fisher gives
a `1 / g` update, and every preset had opted out through a restored absolute
`damping`. The review also asked for the multi-head convention to be stated
in the training guide, the eval-metric table row to be corrected, the
`grad_denominator` docstring to be trimmed to the rule, the private
batch-size import in the dashboarding step to be replaced by an export, and
the CHANGELOG to record the NGD state field. The revision implemented the
reduced route described under "Natural-gradient transforms: reduced route"
together with those items. GitHub issue 68
(https://github.com/trueagi-io/FabricPC/issues/68) was updated with the
learnings from the NGD tests and the new requirements for the estimator.
