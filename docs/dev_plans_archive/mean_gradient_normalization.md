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
the coupled-L2 SGD preset and the clipping threshold in the transformer demo.

## Symbols

| Symbol | Meaning |
|---|---|
| B | Batch size, the leading axis of every clamp |
| S | Prediction positions per sample: the product of a target clamp's non-batch, non-class axes (sequence length for token targets, 1 for classification) |
| N | Prediction count, the single denominator: total clamped-target prediction positions in the batch, Σ over target heads of B·S |
| η | `eta_infer`, the latent step size in `InferenceSGD` |
| g | One weight-gradient leaf as handed to optax, a mean per prediction |

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
clamp. Read per sample, N is the weight `metrics._predictions_per_sample`
assigns in `evaluate()`. Dividing the whole gradient pytree by one scalar
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
`target_energy` and `perplexity`. In `evaluate()`, `internal_energy` weights
each sample by its target prediction count S (1 when the batch carries no
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

## Changes

**fabricpc/training/trainer.py**
1. Add `grad_denominator(structure, clamps)` and `pc_weight_gradients(params,
   state, structure, clamps)`; reuse `_target_node_names`. Rename the private
   batch-size helper to `batch_size_of` and export it.
2. `_batch_grads`: `denom = grad_denominator(structure, clamps)` once. PC
   path: `grads = pc_weight_gradients(...)`, `energy = graph_energy(state,
   structure) / denom`. Backprop path: objective `graph_energy(...,
   node_names=target_nodes) / denom`. `target_energy` divides by it.
3. Module docstring table rows for sub-step 4 (`/ N` for both), the metrics
   comment (both keys per prediction; `energy` remains algorithm-dependent in
   node set), and `make_train_step`'s `"energy"` description.

**fabricpc/training/__init__.py**: export `grad_denominator`,
`pc_weight_gradients`, `batch_size_of`.

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
`sgd(2.0, momentum=0.9)` with the rescale comment. A `--num_epochs` argument
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
- `docs/user_guides/07_optimizers.md`: new paragraph "Gradient scale" after
  Optax Basics stating that gradients reaching optax are means per prediction,
  so learning rates and clipping thresholds are on the same scale as standard
  mean-loss training.
- `CHANGELOG.md`: an `[Unreleased]` entry with a migration table (custom
  loops, SGD-family rates, the `energy` semantics).

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

## Verification

```
pytest tests/test_trainer.py tests/test_inference_tracking.py tests/test_mupc.py \
       tests/test_fabricpc.py tests/test_storkey_hopfield.py \
       tests/test_transformer_nodes.py tests/test_sharding.py
```

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
