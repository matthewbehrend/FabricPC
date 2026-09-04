# Mean gradient normalization with a global prediction-count denominator

## Context

The PC training path hands the optimizer batch-summed weight gradients (`compute_local_weight_gradients` sums each node's per-sample energy over the batch), while the backprop path divides its objective by `batch_size` (trainer.py:309). The same learning rate, Adam ε, and any future clipping threshold therefore mean different things across `algorithm="pc"` and `"backprop"`, and gradient scale changes with batch size. This change normalizes both paths to mean gradients over a single global denominator that is correct for image minibatches (B samples), timeseries targets (B·T positions), and language-model token objectives (B·S tokens), and is safe under data-parallel sharding. It also builds in the discipline that avoids the Hugging Face gradient-accumulation defect (per-microbatch means averaged over microbatches with unequal token counts overweight short microbatches): gradients stay sums everywhere; division by a global count happens exactly once, at the optimizer boundary.

Symbols: B = batch size; T/S = time/sequence length of a target clamp; N = the global denominator, the total number of independent prediction terms in the batch.

## Design

**Denominator rule.** N = total clamped-target prediction positions, `sum(prod(clamps[name].shape[:-1]) for name in target_nodes)` — the expression already computed as `n_predictions` at trainer.py:320 and matching `evaluate()`'s per-prediction weighting (`metrics._predictions_per_sample`). A rank-2 image target `(B, C)` gives N = B; a rank-3 sequence target `(B, T, V)` gives N = B·T. Target-free PC training (no clamped `in_degree > 0` node, e.g. associative-memory graphs) falls back to N = `batch_size`. Dividing the whole gradient pytree by one scalar leaves relative layer scaling untouched; it redefines the objective as total energy per prediction.

**Placement: sum mechanics, one mean boundary.** `compute_local_weight_gradients` keeps returning batch-summed gradients — sums are associative, so they survive sharding and any future microbatch accumulation unchanged. Each optimizer-feeding call site divides once by N from a new public helper `grad_denominator(structure, clamps, batch_size)` in trainer.py. There are exactly two such call sites: `_batch_grads` (both algorithm branches) and `train_step_with_history` in inference_tracking.py. The five test files that call `compute_local_weight_gradients` directly only inspect gradients and are unaffected.

**Sharding safety.** Data parallelism is `jax.jit` + `NamedSharding` over a `"data"` mesh axis (trainer.py:385, 476): the jitted step traces on global logical shapes, so the static shape product N is already the global count and the batch-summed gradients are globally reduced by XLA. No collective is needed. A future `shard_map`/`pmap` port would see per-shard shapes and must `jax.lax.psum` the count over the data axis — recorded in the helper's docstring, not in code.

**Microbatching (not applicable yet).** The train loop performs one optimizer update per loader batch; no accumulation exists, and no padding/validity masks enter any energy (the only masks are attention masks). Static shape counts are therefore exact today. Per the request, the future rule lands as docstring guidance on `grad_denominator` (referenced from `compute_local_weight_gradients`):

> Gradients arrive batch-summed and are divided once by this global count. If gradient accumulation over microbatches is added, keep that structure: accumulate the unnormalized sums across microbatches and divide once by the count summed over the whole accumulation window — dividing per microbatch and averaging overweights predictions in small microbatches when token counts differ. If padded positions gain a validity mask, replace the static shape product with `jnp.sum(mask)`. Under jit + NamedSharding, trace-time shapes and reductions are global, so this count is already global; a shard_map/pmap port must instead `jax.lax.psum` the per-shard count over the data axis.

## Alternatives considered

- **Divide inside `compute_local_weight_gradients`** (pass N in). Pros: no caller can forget. Cons: the denominator is a property of the objective (target clamps), which learning.py does not know; inspection-only consumers must supply a meaningless constant or an optional default creeps in — a fallback flag by another name. Rejected.
- **`jnp.mean` inside node `forward_and_weight_grads`.** Pros: locality. Cons: touches every node template, breaks sPC/ePC energy-parity tests, and per-sample is the wrong denominator for token objectives. Rejected.
- **Per-sample denominator (`batch_size`) everywhere.** Pros: smallest diff (backprop path already does it). Cons: learning rate does not transfer across sequence lengths; fails the LM-token requirement. Rejected.

## Changes

**fabricpc/training/trainer.py**
1. Add public `grad_denominator(structure, clamps, batch_size)` with the docstring above; reuse `_target_node_names`.
2. `_batch_grads`: compute `denom = grad_denominator(...)` once. PC path: `grads = jax.tree_util.tree_map(lambda g: g / denom, grads)`. Backprop path: objective becomes `graph_energy(..., node_names=target_nodes) / denom` (replacing `/ batch_size`). The `n_predictions` local for the `target_energy` metric collapses into the same computation.
3. Metrics: PC `"energy"` stays `graph_energy / batch_size` (a per-sample diagnostic, not the gradient scale). Backprop `"energy"` is the objective and moves with it — identical for rank-2 targets, 1/T smaller for sequence targets. Update the normalization comment (lines 327–330), the module-docstring table rows, and `make_train_step`'s `"energy"` description (line 369).

**fabricpc/utils/dashboarding/inference_tracking.py** — `train_step_with_history` (~line 264): apply the same division before `optimizer.update`, importing `grad_denominator` (it already imports `build_clamps` from trainer; no import cycle). Update its docstring.

**fabricpc/core/learning.py** — docstring only: state the contract (returns batch-summed gradients; optimizer-feeding callers divide once by `grad_denominator`) and point to the helper for the microbatching/padding rule.

**fabricpc/training/natural_gradients.py** — docstring note: the update `g / (fisher + damping)` is not scale-invariant; with gradients N× smaller the Fisher EMA shrinks N²×, so an existing `damping` sits N²× closer to the damping-dominated regime and may need retuning.

**Docs** — `docs/user_guides/08_training_and_evaluation.md:28` (objective normalization "per sample" → per prediction); `docs/user_guides/06_custom_nodes.md:322,375` (note the trainer's single global division).

## Test updates

- `tests/test_trainer.py:161-169` `test_pc_parity_hand_rolled_reference`: add the same division to the hand-rolled `reference_step` — the only breaking test (Adam trajectory changes with gradient scale). Update the per-prediction comment near line 213.
- Unaffected: direct `compute_local_weight_gradients` calls with relative/sign assertions (test_fabricpc.py:231, test_mupc.py:1035, test_inference_epc.py:287/422, test_storkey_hopfield.py:223/263); metric-shape tests; loss-decrease tests.
- New tests (tests/test_trainer.py):
  1. Backprop objective with a rank-3 target divides by B·T (small sequence graph).
  2. Cross-algorithm scale parity: 1-step ePC hidden-node weight gradients equal `eta_infer ×` backprop gradients on identical params (the B factor measured in the earlier analysis disappears; pins both the normalization and the ePC/backprop relation).
  3. Target-free PC graph uses the `batch_size` fallback.

## Verification

```
pytest tests/test_trainer.py tests/test_inference_epc.py tests/test_mupc.py \
       tests/test_fabricpc.py tests/test_storkey_hopfield.py tests/test_transformer_nodes.py
```
Then re-run the 1-step ePC vs backprop gradient comparison from the preceding analysis: hidden-layer norm ratio must be exactly `eta_infer` (was B·`eta_infer`) and the output layer exactly 1 (was B). Finally a short ResNet-18 demo run per algorithm (`--num_epochs 0.2`) confirming AdamW training curves are unchanged (Adam normalizes a uniform scale; only ε-level effects expected). Demo learning rates stay as tuned; any future SGD ablation tuned before this change needs lr × N.
