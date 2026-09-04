# Mean gradient normalization with a global prediction-count denominator

## Context

The PC training path hands the optimizer batch-summed weight gradients
(`compute_local_weight_gradients` sums each node's per-sample energy over the
batch), while the backprop path divides its objective by `batch_size`
(trainer.py:309). The same learning rate, Adam ε, clipping threshold, or
natural-gradient damping therefore means different things across
`algorithm="pc"` and `"backprop"`, and gradient scale changes with batch size
and sequence length. This change normalizes both paths to mean gradients over
one global denominator that is correct for image minibatches, timeseries
targets, and language-model token objectives, is safe under data-parallel
sharding, and keeps the discipline that avoids the Hugging Face
gradient-accumulation defect (per-microbatch means averaged over microbatches
with unequal token counts overweight short microbatches): gradients stay sums
everywhere; division by a global count happens exactly once, in one function.

Because the change moves every gradient to a new scale, it also owns the
optimizer components in this repository whose behavior depends on that scale:
the natural-gradient transforms, the coupled-L2 SGD preset, and the clipping
threshold in the transformer demo.

## Symbols

| Symbol | Meaning |
|---|---|
| B | Batch size, the leading axis of every clamp |
| S | Prediction positions per sample: the product of a target clamp's non-batch, non-class axes (sequence length for token targets, 1 for classification) |
| N | Prediction count, the single denominator: total clamped-target prediction positions in the batch, B·S |
| η | `eta_infer`, the latent step size in `InferenceSGD` |
| g | One weight-gradient leaf as handed to optax, a mean per prediction |
| f | The Fisher EMA leaf in a natural-gradient transform, an EMA of g² |
| r | Damping reference, trace(F)/dim: the mean of all Fisher entries across the parameter pytree |
| ρ | `relative_damping`: damping as a fraction of r |

## Design

### Denominator rule

N is the total number of clamped-target prediction positions,
`sum(prod(clamps[name].shape[:-1]) for name in target_nodes)`, where a target
node is a clamped node with `in_degree > 0` (`_target_node_names`). The
trailing axis of a target clamp is the class axis: `build_clamps` validates
every target clamp against `(batch, *node.shape)`, so rank ≥ 2 holds. A rank-2
image target `(B, C)` gives N = B; a rank-3 token target `(B, S, V)` gives
N = B·S. The same rule read per sample is N = B × S with S = 1 when the graph
has no clamped target (associative-memory graphs), which is exactly the weight
`metrics._predictions_per_sample` assigns in `evaluate()`. Dividing the whole
gradient pytree by one scalar leaves relative layer scaling untouched; it
redefines the objective as total energy per prediction.

Definition caveat, recorded in the helper docstring: a clamped input node that
receives feedback edges has `in_degree > 0` and counts as a target. No graph in
the repository does this (the cyclic and lateral MNIST demos leave `pixels`
edge-free on the input side).

### Placement: sum mechanics, one mean function

`compute_local_weight_gradients` keeps returning batch-summed gradients. Sums
are associative, so they survive sharding and any future microbatch
accumulation unchanged. Two public functions in trainer.py own the mean:

- `grad_denominator(structure, clamps) -> int`: the rule above. B is read from
  the leading axis of the first clamp; empty `clamps` raises `ValueError`
  (nothing is clamped, so there is no objective).
- `pc_weight_gradients(params, state, structure, clamps) -> GraphParams`:
  `compute_local_weight_gradients(...)` with every leaf divided by
  `grad_denominator(structure, clamps)`.

`_batch_grads` calls `pc_weight_gradients` on the PC path and divides the
backprop objective by the same `grad_denominator`. `train_step_with_history`
calls `pc_weight_gradients`. These are the only two optimizer-feeding sites in
the package; no example rolls its own step. Both functions are exported from
`fabricpc.training` so custom loops that today call
`compute_local_weight_gradients` directly have a normalized entry point. The
trainer, not learning.py, knows the clamps, which is why the division lives
here.

### Metrics: `energy` is the objective

`energy` reports the quantity the optimizer descends, for both algorithms:
PC reports `graph_energy` over all `in_degree > 0` nodes divided by N;
backprop reports `graph_energy` over the target nodes divided by N, which
equals `target_energy`. `target_energy` is unchanged (target-node energy over
N). For rank-2 targets every number equals today's. For sequence targets
`energy` becomes per token, on the same scale as `target_energy` and
`perplexity`. In `evaluate()`, `internal_energy` weights each sample by its
target prediction count S (1 when the batch carries no target key), so the
eval `energy` of a PC model is per prediction as well.

### Sharding safety

Data parallelism is `jax.jit` + `NamedSharding` over a `"data"` mesh axis
(trainer.py:385, 476). `train` and `make_train_step` place the batch under
`NamedSharding` before the jitted step, so the step traces on global logical
shapes: the static shape product N is the global count and the batch-summed
gradients of the replicated parameters are an XLA all-reduce. No collective is
needed. A future `shard_map`/`pmap` port would see per-shard shapes and must
`jax.lax.psum` the count over the data axis. This is recorded in the
`grad_denominator` docstring.

### Microbatching (not applicable yet)

The train loop performs one optimizer update per loader batch; no accumulation
exists, and no padding or validity masks enter any energy (the only masks are
attention masks). Static shape counts are exact today. The future rule lands
as the `grad_denominator` docstring, referenced from
`compute_local_weight_gradients`:

> Gradients arrive batch-summed and are divided once by this global count. If
> gradient accumulation over microbatches is added, keep that structure:
> accumulate the unnormalized sums across microbatches and divide once by the
> count summed over the whole accumulation window. Dividing per microbatch and
> averaging overweights predictions in small microbatches when token counts
> differ. If padded positions gain a validity mask, replace the static shape
> product with `jnp.sum(mask)`. Under jit + NamedSharding, trace-time shapes
> and reductions are global, so this count is already global; a
> shard_map/pmap port must instead `jax.lax.psum` the per-shard count over the
> data axis.

### Natural-gradient transforms: scale-free damping

Both transforms in natural_gradients.py compute `g / (f + damping)` with f an
EMA of g². Scaling every gradient by 1/N scales f by 1/N², so a fixed
`damping` moves N² closer to dominating. At per-prediction gradient magnitudes
of about 1e-3 per weight, f is about 1e-6 and the current default `damping =
1e-3` dominates every entry: the transform degenerates to SGD with learning
rate `scale / damping`. The natural-gradient update is inherently covariant
(scaling g by c scales `g / f` by 1/c), so no damping rule can make its
magnitude scale-invariant; what a rule can do is make the regime, which term
dominates, independent of scale. The transforms adopt:

1. Bias-corrected Fisher: f̂ = f / (1 − fisher_decay^t), with t the step count
   held in the state. Without this the first steps see f ≈ (1 − decay)·g² and
   over-large updates, which the absolute damping used to mask.
2. Damping reference r = trace(F̂)/dim, the mean of all bias-corrected Fisher
   entries across the pytree. For the layerwise transform each leaf's scalar
   counts with weight equal to the leaf's size, so both transforms use one
   definition.
3. Update `g / (f̂ + ρ·r)`, computed as `g / where(d > 0, d, 1)` with
   d = f̂ + ρ·r. The denominator is zero only when every gradient entry seen
   so far, including the current one, is zero (the EMA includes the current
   g² with weight 1 − fisher_decay > 0), and then g = 0 and the guarded
   division returns the correct update, 0, without forming 0/0. No absolute
   constant enters the denominator. An earlier draft added `floor = 1e-12`;
   measured in float32, that floor breaks the covariance below by about
   5e-4 relative at c = 1e-4, and at per-prediction gradient magnitudes near
   1e-6 (f̂ ≈ 1e-12) it halves the update, so it would have re-introduced an
   absolute scale at the very magnitudes this change produces.

Property pinned by test: `update(c·g) = update(g) / c` exactly (to float32
rounding, about 2e-7) for any c > 0, and the set of entries where damping
dominates is the same for g and c·g.
Signatures become `scale_by_natural_gradient_diag(fisher_decay=0.95,
relative_damping=0.1, damping=0.0)` and
`scale_by_natural_gradient_layerwise(...)` with the same arguments. The state
tuples gain a `count` field.

`damping` was first removed and then restored as an explicit opt-in, default
0, after the calibration below showed that neither transform trains the MNIST
demo with relative damping alone. With `damping > 0` the denominator is
`f̂ + ρ·r + damping`; wherever the constant dominates, the update is
`g / damping`, SGD with rate `scale / damping`, and the covariance above no
longer holds. `_validate_hparams` accepts `relative_damping >= 0` and
`damping >= 0` and rejects both being zero (then `g / f̂ ≈ 1 / g` has no
bound as gradients shrink). The docstrings record this and the limitation
paragraph below as properties of the present implementations.

### Limitation of the present transforms

Both transforms build F from the squared *mean* gradient of the batch, not
from per-sample gradients, so `g / F ≈ 1 / g`: the entries with the largest
gradients receive the smallest steps, and the step grows relative to the
gradient as training reduces it (the covariance property states this: smaller
gradients, larger updates). On `examples/mnist_advanced.py` this leaves both
transforms at chance accuracy with relative damping alone, at every one of 48
settings tried (Verification, item 2). The demo presets train, weakly, only
through an absolute damping that lies above 96.5% of the Fisher entries from
the first step, so those entries are updated as SGD and only the few
large-gradient entries receive the natural-gradient step. This is documented
in the module docstring, the optimizers guide, and the changelog; fixing it
(a per-sample Fisher, or a square-root normalization as in Adam) is outside
this change.

### Clipping thresholds

`optax.clip_by_global_norm` thresholds now sit on the per-prediction scale that
standard mean-loss training uses. `examples/transformer_demo.py` keeps
`clip_by_global_norm(0.8)`: today the PC summed-gradient norm over
B·S = 16384 positions exceeds 0.8 on essentially every step, so the clip acts
as per-step normalization ahead of Adam; after the change it fires on the
conventional schedule. The verification records perplexity in both modes.
Outcome: backprop is unchanged; PC at the old `--lr 1e-4` destabilizes after
about 3500 steps without that per-step normalization, and `--lr 3e-5` holds
the full epoch at a slightly better perplexity, so the demo's PC default moves
to 3e-5 (Verification, item 3).

### SGD with coupled L2

`optax.chain(add_decayed_weights(wd), sgd(lr))` applies `lr·(g_sum + wd·θ) =
lr·N·(g_mean + (wd/N)·θ)`. The `mnist_advanced.py` `sgd` preset (B = 200) is
rescaled exactly to `lr = 2.0`, `wd = 5e-4`, with a comment stating these are
lr·N and wd/N of the summed-gradient values. The two natural-gradient presets
cannot be rescaled exactly because their damping semantics change; they are
retuned by the calibration run in Verification.

## Alternatives considered

- **Divide inside `compute_local_weight_gradients`** (pass N in). Pros: no
  caller can forget. Cons: the denominator is a property of the objective,
  which learning.py does not know; inspection-only callers would supply a
  meaningless constant or an optional default would creep in. Rejected;
  `pc_weight_gradients` in the trainer gives the same guarantee where the
  clamps are known.
- **Divide at each optimizer-feeding call site** via `grad_denominator`
  alone. Pros: smallest surface. Cons: the contract is enforced by docstring;
  `train_step_with_history` is already a hand copy of the PC step that
  diverged on how it reads the batch size. Rejected.
- **`jnp.mean` inside node `forward_and_weight_grads`.** Pros: locality.
  Cons: touches every node template and per-sample is the wrong denominator
  for token objectives. Rejected.
- **Per-sample denominator (`batch_size`) everywhere.** Pros: smallest diff.
  Cons: the learning rate does not transfer across sequence lengths. Rejected.
- **Keep PC `energy` per sample.** Pros: no metric change. Cons: the unified
  trainer would report `energy` with a different normalization per algorithm
  for sequence targets, and PC `energy` would no longer be the optimized
  quantity. Rejected.
- **Absolute retune of natural-gradient damping.** Pros: no API change. Cons:
  the regime stays scale-dependent, so any later change of batch size or
  sequence length re-breaks it silently. Rejected as the default; restored as
  an explicit opt-in (`damping`, default 0) after calibration showed the
  relative-only transforms do not train the MNIST demo, so that the demo
  presets can keep the parent's trajectory and the limitation is documented
  rather than hidden.
- **Remove or retune the transformer clip.** Pros: fewer moving parts. Cons:
  0.8 to 1.0 on a per-token gradient is the standard LM recipe; removing it
  would itself change dynamics. Rejected in favor of keeping the value and
  measuring.

## Changes

**fabricpc/training/trainer.py**
1. Add `grad_denominator(structure, clamps)` and `pc_weight_gradients(params,
   state, structure, clamps)` with the docstrings above; reuse
   `_target_node_names`.
2. `_batch_grads`: `denom = grad_denominator(structure, clamps)` once. PC
   path: `grads = pc_weight_gradients(...)`, `energy = graph_energy(state,
   structure) / denom`. Backprop path: objective `graph_energy(...,
   node_names=target_nodes) / denom`. The `n_predictions` local collapses into
   `denom`; `target_energy` divides by it.
3. Module docstring table rows for sub-step 4 (`/ N` for both), the metrics
   comment at lines 327–330 (both keys per prediction; `energy` remains
   algorithm-dependent in node set), and `make_train_step`'s `"energy"`
   description (line 369: the objective per prediction).

**fabricpc/training/__init__.py**: export `grad_denominator` and
`pc_weight_gradients`.

**fabricpc/training/metrics.py**: split `_target_items` into a non-raising
iterator and the raising wrapper the target metrics use. `_internal_energy_fn`
weight = Σ over target items of `_predictions_per_sample(y)`, or 1 with no
target key. Update the module docstring's description of `energy`.

**fabricpc/utils/dashboarding/inference_tracking.py**:
`train_step_with_history` (lines 161–227) reads B via `_batch_size(batch,
structure)` from trainer, computes `denom = grad_denominator(structure,
clamps)`, reports `energy = graph_energy(final_state, structure) / denom`, and
feeds `pc_weight_gradients(...)` to `optimizer.update`. Docstring: `energy` is
the objective per prediction.

**fabricpc/core/learning.py**: docstring only. State the contract (returns
batch-summed gradients; the trainer's `pc_weight_gradients` divides once by
`grad_denominator`) and point to the helper for the microbatching and padding
rule.

**fabricpc/training/natural_gradients.py**: the design above. Both state
tuples gain `count` (int32, incremented with `optax.safe_int32_increment`
before the bias factor `1 − fisher_decay**count` is formed, so t = 1 on the
first update); `damping` becomes `relative_damping` (default 0.1) with no
alias; the zero guard is the `where` form above; `_validate_hparams` checks
`relative_damping > 0`. The bias correction is written locally rather than
through `optax.tree_utils.tree_bias_correction`, which first appears in
optax 0.2.3 while the declared floor is `optax>=0.1.7` (verified: the 0.1.7
wheel imports and runs `optax.adam` under the venv's JAX 0.10.1). Docstrings
state the covariance property and that the trainer hands over per-prediction
mean gradients. `optimizers.py` re-exports unchanged.

**examples/mnist_advanced.py**: `sgd` preset `add_decayed_weights(5e-4)`,
`sgd(2.0, momentum=0.9)` with the rescale comment. `ngd_diag` and
`ngd_layerwise` presets are the exact per-prediction analog of the parent's
constants: `add_decayed_weights(5e-4)`, `relative_damping=0.0`,
`damping=1e-3 / N**2`, `optax.scale(-scale_old / N)` (F shrinks by N², g by
N). The comment records the calibration and the measured regime. A
`--num_epochs` argument (default 10) makes the calibration command
reproducible.

**examples/transformer_demo.py**: the clip stays at 0.8. The `--lr` default
becomes mode-dependent (`None` resolved to 3e-5 for `pc`, 1e-4 for
`backprop`) after the verification below; the energy comment in the training
loop and the docstring `Results:` block (PC energy now per token) are
updated.

**Docs**
- `docs/user_guides/08_training_and_evaluation.md`: table row at line 28 and
  the Gaussian sentence at line 36 ("per sample" → per prediction, N); the
  `energy` bullet at line 107 (the objective per prediction, node set still
  algorithm-dependent); delete the "different normalizations" note at lines
  115–116.
- `docs/user_guides/09_experiment_tracking.md:193–194`: comment becomes
  "energy is the objective per prediction (graph_energy over internal nodes /
  prediction count)".
- `docs/user_guides/06_custom_nodes.md:264` and `:358`: add that the trainer
  divides the summed gradients once by the prediction count
  (`pc_weight_gradients`).
- `docs/user_guides/03_how_predictive_coding_works.md:69`: the sentence naming
  `compute_local_weight_gradients` as the gradient source gains the division
  step. `examples/mnist_aim_tracking.py:209-211` carried the same stale
  "per-sample / batch_size" comment as the tracking guide and is updated with
  it.
- `CHANGELOG.md`: an `[Unreleased]` entry with a migration table (custom
  loops, SGD-family rates, the `damping` rename, the `energy` semantics).
- `docs/user_guides/07_optimizers.md`: new paragraph "Gradient scale" after
  Optax Basics stating that gradients reaching optax are means per prediction,
  so learning rates, clipping thresholds, and damping are on the same scale as
  standard mean-loss training; rewrite the Natural Gradient section for
  `relative_damping`, the bias-corrected Fisher, and the covariance property.

## Test updates

- `tests/test_trainer.py:161–169` `test_pc_parity_hand_rolled_reference`:
  `reference_step` uses `pc_weight_gradients`. Comments at lines 213 and 231
  say "/ prediction count". `test_eval_energy_matches_graph_energy` docstring
  says `/ N`.
- Unaffected: direct `compute_local_weight_gradients` calls with shape, sign,
  or finiteness assertions (test_fabricpc.py:231, test_mupc.py:1038,
  test_storkey_hopfield.py:181/221); metric-shape tests; loss-decrease tests;
  Adam-based step tests.
- `tests/test_optimizers.py`: migrate `damping=` to `relative_damping=`.
- New tests, `tests/test_trainer.py`:
  1. Backprop objective with a rank-3 target divides by B·S
     (`v1_masked_structure` + `make_token_batches`).
  2. One-step sPC vs backprop on `classification_structure` (single hidden
     layer, `FeedforwardStateInit`, `infer_steps=1`, identical params):
     hidden-node weight gradients equal η × backprop gradients to 1e-6
     relative. Output-node gradients differ by O(η) because
     `compute_local_weight_gradients` re-evaluates the output prediction at
     the moved hidden latent: assert relative deviation below 10·η at
     η = 1e-3 and that it shrinks when η is divided by 10.
  3. Target-free PC graph: N = B; gradients equal the raw sums / B and
     `energy` equals `graph_energy / B`.
  4. `grad_denominator(structure, clamps)` equals B × Σ weights from
     `_internal_energy_fn` for a sequence batch and a two-target batch (pins
     train/eval agreement).
  5. PC `energy` on `v1_masked_structure` equals `graph_energy / (B·S)`.
- New tests, `tests/test_optimizers.py`:
  6. Covariance: `update(c·g) == update(g) / c` for both transforms with
     c = 1e-4 and 1e4, after several steps.
  7. Bias correction: after one step with constant g, f̂ equals g² exactly, so
     the update equals `g / (g² + ρ·mean(g²))`.
  8. All-zero gradient: the update is all zeros and finite.
- New test file `tests/test_inference_tracking.py`: `train_step_with_history`
  matches `make_train_step` under `optax.sgd(1.0)` (params within 1e-5,
  energy equal to `graph_energy / N`), so the hand-copied step cannot drift
  from the trainer's normalization again.

## Verification

```
pytest tests/test_trainer.py tests/test_optimizers.py tests/test_mupc.py \
       tests/test_fabricpc.py tests/test_storkey_hopfield.py \
       tests/test_transformer_nodes.py tests/test_sharding.py
```

Then:

1. ResNet-18 demo. The demo is PC-only and its `--num_epochs` is an `int`,
   so the fractional per-algorithm run first written here cannot execute.
   Substitute: one default 2-epoch run on the new code, compared with the
   demo's own `Results:` block (train energy 0.4792, test accuracy 33.71%).
   AdamW normalizes a uniform gradient scale (decoupled weight decay; only
   ε-level effects), so the numbers should agree to within the run-to-run
   variation the docstring already states. The per-algorithm Adam check is
   item 3, whose demo has `--mode pc|backprop`.

   **Result** (`python examples/resnet18_cifar10_demo.py`, RTX 3090 shared
   with another job): test accuracy 33.58%, 432 s per epoch, against the
   docstring's 33.71%. Within the stated variation; the docstring block
   stands (the current script prints accuracy and time only).
2. `mnist_advanced.py` calibration. On the parent commit, record 2-epoch test
   accuracy for `sgd`, `ngd_diag`, `ngd_layerwise`. After the change: `sgd`
   reproduces its number with the rescaled constants; for each NGD preset
   sweep the `optax.scale` constant over a decade grid and pick the value
   matching the parent's 2-epoch accuracy; if none does within one point,
   adjust `relative_damping` before the constant. Record chosen values and
   accuracies in the preset comments.

   **Results (RTX 3090, JAX 0.10.1, optax 0.2.8; the demo gains
   `--num_epochs` so these commands are reproducible; the parent was run from
   a detached worktree with `num_epochs` edited to 2).**

   Parent commit, `python examples/mnist_advanced.py --optimizer <preset>`:

   | preset | epoch 2 energy / accuracy | epoch 10 energy / accuracy |
   |---|---|---|
   | `adamw` | 0.4432 / 20.95% | 0.0127 / 97.27% (new code; the parent's 2-epoch numbers are identical) |
   | `sgd` (lr 0.01, wd 0.1) | 0.4517 / 10.28% | 0.3245 / 24.78% |
   | `ngd_diag` (damping 1e-3, scale 3e-4) | 0.5017 / 8.92% | 0.1786 / 25.25% |
   | `ngd_layerwise` (damping 1e-3, scale 1e-3) | 0.4513 / 10.28% | 0.4514 / 9.74% |

   All three summed-gradient presets sit at chance after 2 epochs, so the
   2-epoch matching rule above has no target. New code, `sgd` with the exact
   rescale (lr 2.0, wd 5e-4): epoch 1 energy 0.4811 / 9.58%, epoch 2 0.4517
   / 10.28%, identical to the parent to the printed digits.

   `ngd_diag` and `ngd_layerwise` with `relative_damping` did not leave
   chance accuracy at any point of three grids (48 runs), all at 10 epochs
   unless stated:

   - `optax.scale` ∈ {1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3} at ρ = 0.1, 2
     epochs: accuracy 8.9–11.4%; the old constants (3e-4, 1e-3) diverge
     (energy rising to 1.15 and 3.55).
   - ρ ∈ {0.1, 1, 10} × scale ∈ {1e-6, 1e-5, 1e-4}: accuracy 9.6–11.4%;
     energy plateaus at 0.45 (every output near 0.1) except ρ = 10, scale
     1e-4 for the diagonal transform, energy 0.207 with accuracy 9.74%.
   - The same with `optax.clip_by_global_norm(1.0)` inserted before
     `optax.scale`, scale ∈ {0.03, 0.1, 0.3, 1.0}, ρ ∈ {0.1, 1}: accuracy
     9.6–11.4%, energy 0.45.

   Mechanism, confirmed by logging on the diagonal transform (ρ = 1, scale
   1e-5): over 250 steps the per-prediction gradient norm falls from 1.29 to
   0.05 while the update norm stays between 0.04 and 0.28, so the step grows
   relative to the gradient as the fit improves. Both transforms build F from
   the squared mean gradient, so g / F ≈ 1 / g: the entries with the largest
   gradients receive the smallest steps, and the informative directions are
   suppressed relative to the rest. The covariance property this design pins
   makes the 1 / g behavior explicit rather than causing it.

   Resolution (maintainer decision): restore an absolute `damping` opt-in
   and document the limitation. The presets become the exact analog of the
   parent's constants (`damping = 1e-3 / N²`, `scale_old / N`,
   `relative_damping = 0`); with the default `relative_damping = 0.1` added,
   the relative term is about 30× the rescaled absolute term and both
   presets stay at chance. Exact analog, 10 epochs:

   | preset | new code, epoch 5 / epoch 10 | parent, epoch 5 / epoch 10 |
   |---|---|---|
   | `ngd_diag` | 0.2761 / 24.12%, 0.1784 / 23.78% | 0.2632 / 24.05%, 0.1786 / 25.25% |
   | `ngd_layerwise` | 0.4514 / 9.58%, 0.4514 / 9.74% | 0.4514 / 9.58%, 0.4514 / 9.74% |

   The residual difference for `ngd_diag` is the Fisher EMA's bias
   correction in the first steps. Regime, measured on `ngd_diag` over the
   first epoch (fraction of bias-corrected Fisher entries below the absolute
   damping, and their share of trace(F)): step 1, 96.5% and 0.01%; step 50,
   97.9% and 0.00%; step 300, 99.7% and 0.00%. Almost every entry is updated
   as SGD with rate `scale / damping` (60 for `ngd_diag`, 200 for
   `ngd_layerwise`); the few entries holding the Fisher trace receive the
   `1 / g` step. The parent presets therefore trained weakly, and
   `ngd_layerwise` not at all, for the same reason.
3. `transformer_demo.py` in `--mode pc` and `--mode backprop` at the same
   short budget on the parent commit and after; record eval perplexity for
   both. A PC-mode regression is addressed by retuning the demo's `--lr`
   default, not by changing the clip or the normalization.

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
