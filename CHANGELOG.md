# Changelog

## [unreleased]
ePC error-parameterized predictive coding (`EPCInference`) arrives as a drop-in solver alongside composable inference schedules and first-class cyclic graphs (`graph(..., unroll=U)`); see `docs/user_guides/12_api_inference.md`.

### Breaking changes
- The custom-node contract splits `forward()` into `predict()` and `energy()`. A node now implements `predict(params, inputs, state, node_info) -> (z_mu, aux)` — the parameterized prediction plus an optional pytree of intermediates — and, only when it adds energy terms, overrides `energy(params, inputs, state, aux, node_info) -> (batch,)`. The error pair (`error = z_latent - z_mu` and its inverse `z_latent = z_mu + error`) and the assembly templates (`forward`, `forward_with_aux`, `forward_from_error`) are base-owned and no longer node code: one prediction pass now serves both the state-based solvers and the error-parameterized `EPCInference`, which derives `z_latent` from the relaxed error and cannot tolerate a node body that recomputes the pair itself. `NodeBase.energy_functional` is deleted; its body is the default `energy()`. Migration: delete the `error`/`_replace`/`energy_functional` tail from each `forward()` body, rename it `predict`, and return `(z_mu, aux)`; move any post-hoc energy patching into an `energy()` override. See `docs/user_guides/06_custom_nodes.md`.
- Unclamped readout nodes (`out_degree == 0`, no clamp) are no longer forced to `error = 0`, `energy = 0` with `z_latent` overwritten by `z_mu` during inference: they take the ordinary relaxation path, so energy terms a readout assigns (a StorkeyHopfield readout's attractor term) survive and the readout settles like any other node. Eval accuracy is unchanged — every eval path reads predictions from `z_mu` (`evaluate_transformer` previously read `z_latent`, which the old forcing made equal to `z_mu`, and is migrated). Reported eval energy now includes readout energy that was previously zeroed.
- Inference-solver methods dispatch on the class: `InferenceBase.run_inference` is an instance method and the computation methods (`inference_step`, `forward_value_and_grad`, `update_latents`) are classmethods, no longer reading `structure.config["inference"]` for dispatch. Call `inference.run_inference(params, state, clamps, structure)`; the old static-call form `type(inference).run_inference(params, ...)` binds `params` as `self` and breaks.
- Cyclic graphs require an explicit `graph(..., unroll=U)`; construction raises `GraphCycleError` instead of printing a warning and silently dropping cycle members (and everything downstream) from the topological order. `GraphStructure` gains a `schedule` field — the full visit schedule at the chosen unroll degree — walked by feedforward initialization and ePC state derivation; `node_order` is its first-occurrence deduplication.

### New
- `EPCInference` — error-parameterized predictive coding (Goemaere et al., arXiv 2505.20137). The prediction errors ε are the relaxed variables; each step takes one global reverse pass through the derived forward (`z_latent = z_mu + ε` along `structure.schedule`), so a few steps replace sPC's hundreds on deep graphs. `begin_segment` resyncs ε := z_latent − z_mu at the carried latents, so ePC continues exactly from any incoming state; `finalize_state` rebuilds a self-consistent state for the unchanged local weight-gradient path. See `docs/user_guides/12_api_inference.md`.
- `InferenceSchedule` — composable per-update solver segments (e.g. a few global ePC steps warm-starting sPC refinement); nested schedules flatten via `segments()`.
- `InferenceBase` segment hooks `begin_segment`/`finalize_state` and `segments()`; the tracking variants (`run_inference_with_history`, `run_inference_with_full_history`) iterate segments, so composed schedules are tracked segment by segment.
- `graph(..., unroll=U)` — first-class cyclic graphs: `GraphStructure.schedule` holds the visit schedule at unroll degree U, walked by feedforward initialization and ePC state derivation (see the breaking-changes entry).
- ePC vs sPC benchmark on ResNet-18/CIFAR-10: `examples/epc_spc_resnet18_compare.py` (trained-accuracy/wall-clock sweep and single-batch convergence modes).

## [0.5.2] - 2026-09-08

Per-batch diagnostics run as `train` callbacks instead of custom training
loops. `iter_callback` now receives one `IterContext` carrying the parameters,
optimizer state, the batch, its RNG key, and the step's `GraphState`.
The dashboarding callbacks run on `train`.

### Migration table

| Changed | Replacement |
|---|---|
| `iter_callback(epoch_idx, batch_idx, metrics)` | `iter_callback(ctx: IterContext)`; read `ctx.epoch_idx`, `ctx.batch_idx`, `ctx.metrics` |
| `create_detailed_iter_callback` in a custom loop | `train(..., iter_callback=create_iter_callback(tracker))` with `TrackingConfig(track_state=True, distribution_nodes=[...])` |
| Weight histograms logged once per epoch for every node | logged every `tracking_every_n_batches` for `distribution_nodes` only. Set it, or no weight or state distributions are logged |
| `run_inference_with_full_history` | `make_inference_history(structure, every=k)`, jitted |
| Hand-built `EpochContext(...)` | add `algorithm` and `epoch_key` |

### New

- `IterContext`: superset of the `EpochContext` fields, then `batch_idx`, `state`,
  `batch_key`, `batch`. `EpochContext` gains `algorithm` and `epoch_key`.
  New fields are appended; existing positions do not move.
- `TrackingConfig.distribution_nodes`: the nodes whose weight and state
  distributions are logged. Empty logs none, as an empty `nodes_to_track`
  logs no per-node energy.
- `TrackingConfig.track_state`: state summary statistics on tracked batches;
  `track_state_distributions` implies it. Under PC the callback re-settles
  the tracked batch under the updated parameters in one jitted program and
  logs the state after `0, k, 2k, ...` inference steps
  (`k = state_tracking_every_n_infer_steps`), the last being the settled
  state. Under backprop it logs the feedforward state once.
- `make_inference_history` and `make_tracked_settle` in
  `fabricpc.utils.dashboarding`: the jitted re-settle, usable from custom
  loops.
- `BayesianTuner` passes its progress callback only when `verbose=True`.

## [0.5.1] - 2026-09-07

Gradients reaching optax are now means per prediction under both algorithms.
The trainer divides the batch-summed PC weight gradients and the backprop
objective once by one global prediction count N: the total number of
clamped-target prediction positions in the batch (`batch` for classification,
`batch * seq_len` for token targets, summed over target heads, `batch` when the
graph has no clamped target). One learning rate, clipping threshold, or Adam
epsilon now means the same under `algorithm="pc"` and `"backprop"` and across
batch sizes and sequence lengths. For rank-2 (classification) targets every
reported number is unchanged; for sequence targets the PC `energy` metric moves
from per sample to per token.

### Migration table

| Changed | Replacement |
|---|---|
| `compute_local_weight_gradients` fed directly to an optimizer in a custom loop (batch-summed gradients) | `pc_weight_gradients(params, state, structure, clamps)` — the same gradients divided by `grad_denominator(structure, clamps)` |
| SGD-family learning rates tuned on summed PC gradients | multiply `lr` by N and divide a coupled `add_decayed_weights` rate by N (Adam/AdamW rates are unchanged) |
| Training `energy` metric read as "per sample" | per prediction, on the same scale as `target_energy`; the node set is still algorithm-dependent |
| `scale_by_natural_gradient_diag` / `scale_by_natural_gradient_layerwise` optimizer states saved under 0.5.0 | do not restore: both states gained a `count` field for the Fisher EMA's bias correction |

### New

- `fabricpc.training.grad_denominator(structure, clamps) -> int`,
  `fabricpc.training.pc_weight_gradients(params, state, structure, clamps)`,
  and `fabricpc.training.batch_size_of(batch, structure)`.
- Natural-gradient transforms: bias-corrected Fisher EMA (the states gain a
  `count` field) and a `damping` default of 1e-8, chosen on the
  MNIST demo at the per-prediction gradient scale. The module docstring
  states the two regimes the transforms have (SGD with rate
  `scale / damping` where the damping dominates, about `1 / g` where the
  Fisher does) and why no damping value yields a natural-gradient step; the
  estimator redesign is tracked in
  https://github.com/trueagi-io/FabricPC/issues/68.
- `evaluate`'s default PC `energy` metric weights each sample by its
  prediction count, so it reports internal energy per prediction and agrees
  with the training `energy`.
- `train_step_with_history` (dashboarding) reads the batch size from the
  task-mapped keys and normalizes like the trainer; a parity test pins it to
  `make_train_step`.
- `examples/mnist_advanced.py` gains `--num_epochs`; its `sgd` preset is
  rescaled exactly (`lr` 0.01 -> 2.0, weight decay 0.1 -> 5e-4 at N = 200)
  and the two natural-gradient presets use constants swept on the
  per-prediction scale.

## [0.5.0] - 2026-08-30
One trainer replaces the four training harnesses. `train`/`evaluate` serve both
learning algorithms, selected by `algorithm="pc"|"backprop"`; backprop is framed
in energy (its objective is the clamped target node's energy, so the output
node's energy functional selects the loss), causal masking and target one-hot
encoding are derived from the graph and target dtype, training is resumable
(`opt_state`/`start_epoch`), and multi-device data parallelism runs on jit +
`NamedSharding` meshes instead of pmap. Clean break: the legacy names are
removed, not deprecated.

### Migration table

| Removed | Replacement |
|---|---|
| `train_pcn` | `train` (returns `TrainResult`; use `result.params`) |
| `evaluate_pcn` | `evaluate` |
| `train_backprop` | `train(..., algorithm="backprop")` |
| `evaluate_backprop` | `evaluate(..., algorithm="backprop")` |
| `train_autoregressive` | `train` (mask graph-derived, one-hot dtype-derived) |
| `evaluate_autoregressive` | `evaluate` |
| `train_backprop_autoregressive` | `train(..., algorithm="backprop")` |
| `evaluate_backprop_autoregressive` | `evaluate(..., algorithm="backprop")` |
| `evaluate_transformer` | `evaluate` |
| `generate_autoregressive` | `generate` (same sampling parameters) |
| `train_step` | `make_train_step(structure, optimizer)` -> `step(params, opt_state, batch, rng_key)` |
| `train_step_backprop`, `train_step_autoregressive`, `train_step_backprop_autoregressive` | `make_train_step(..., algorithm=...)` |
| `train_step_pmap`, `create_pmap_train_step` | `make_train_step(..., mesh=...)` |
| `get_graph_param_gradient` | compose `build_clamps` + `initialize_graph_state` + `run_inference` + `compute_local_weight_gradients` |
| `build_train_clamps` | `build_clamps(batch, structure, clamp_target=True)` |
| `causal_mask_clamps` | `build_clamps` (injected when the `TaskMap` declares `causal_mask`) |
| `compute_loss`, `compute_loss_autoregressive`, `compute_forward_pass` | the output node's energy functional + `graph_energy` |
| `replicate_params`, `replicate_opt_state`, `shard_batch`, `unshard_energies` | not needed: pass `mesh=jax.make_mesh((jax.device_count(),), ("data",))` |
| `train_pcn_multi_gpu`, `evaluate_pcn_multi_gpu`, `evaluate_transformer_multi_gpu`, `fabricpc.training.multi_gpu` | `train`/`evaluate` with `mesh=` |
| `pmap_single_device=`, `use_tqdm=` | removed (`verbose` controls tqdm; test meshes via `XLA_FLAGS=--xla_force_host_platform_device_count=2`) |
| `config["loss_type"]` | removed — raises `ValueError`; set the output node's energy functional |
| `config["use_causal_mask"]` | removed — raises `ValueError`; the mask follows the graph |
| `autoregressive=` (never released) | removed — mask graph-derived, one-hot dtype-derived |
| `iter_callback(epoch_idx, batch_idx, energy: float)` | `iter_callback(epoch_idx, batch_idx, metrics: dict)` — read `metrics["energy"]`; formatting the third argument directly (`f"{energy:.4f}"`) now raises `TypeError` |
| `epoch_callback(epoch_idx, params, structure, config, rng_key)` — five positionals | `epoch_callback(ctx: EpochContext)` — one context argument, fields by name |
| `evaluate_backprop(..., rng_key=None)` (defaulted to `PRNGKey(0)`) | `evaluate` — `rng_key` is a required positional |
| `create_detailed_iter_callback` (dashboarding) `(epoch_idx, batch_idx, energy: float, final_state)` | `(epoch_idx, batch_idx, metrics: dict, final_state)`, for custom loops over `make_train_step` |

### New

- `train(...) -> TrainResult(params, opt_state, step, iter_results, epoch_results)`,
  with `opt_state=` and `start_epoch=` for resume: optimizer moments and optax
  schedule counts survive a save/load boundary, and the fold_in RNG stream makes
  an interrupted run bitwise-equal to the uninterrupted one.
- `epoch_callback(ctx: EpochContext)` — one context argument (`epoch_idx`,
  `step`, `params`, `opt_state`, `structure`, `config`, `rng_key`, `metrics`)
  that grows by field addition. Callback exceptions propagate (tested; tuner
  pruning depends on it); a non-None return replaces the stored history entry.
- Pluggable eval metrics: `evaluate(..., metrics=)` takes named
  `EvalMetric(fn, finalize)` entries with a per-sample `(value, weight)`
  contract and weighted aggregation `finalize(Σvalue/Σweight)`; `None` selects
  graph-derived defaults (`target_energy`, `accuracy`; + `cross_entropy`,
  `perplexity` for `CrossEntropyEnergy` targets; + `energy` for PC).
- `graph_energy(state, structure, node_names=None)` in `fabricpc.core.energy`:
  the one graph-level energy sum (default: all `in_degree>0` nodes, order fixed
  by the structure).
- `make_train_step(structure, optimizer, algorithm=, mesh=)` — the public
  jitted step for custom loops; returns `(params, opt_state, metrics,
  final_state)` and does not donate its inputs.
- Non-float targets: int **and bool** class/token targets are one-hot encoded
  from their dtype (class count from the target node's `shape[-1]`); stock
  int32 token loaders now work with backprop training too.
- Backprop training gains tqdm progress and multi-device data parallelism.
- `generate(..., algorithm=)` with the same validation as `train`/`evaluate`:
  `"pc"` (default) settles via `run_inference`, `"backprop"` samples from the
  feedforward pass — required for graphs built with `inference=None`, which
  previously crashed inside `run_inference` with an opaque `AttributeError`.
- `BayesianTuner(algorithm=)` threads the learning algorithm through every
  trial's `train`/`evaluate` (previously fixed to PC), and raises instead of
  silently scoring `inf` when the trial graph has no `CrossEntropyEnergy`
  target (no `perplexity` key to minimize).
- Fail-fast diagnostics: a wrong-shape target raises an actionable
  `ValueError` from `build_clamps` (was an opaque XLA broadcast error), a
  loader without `len()` and a mesh without a `"data"` axis raise messages
  naming the requirement, and the causal-mask sequence length is read from
  the mask node's declared shape instead of a hard-coded `batch["x"]`.

### Behavior changes

- Multi-device PC weight gradients are now the global batch sum, matching the
  single-device semantics (pinned by the mesh-vs-single-device parity tests).
  The 0.4 pmap path applied a device mean (`pmean`) over per-device shard
  sums, so its gradients were smaller by the device count N for the same
  global batch. To reproduce 0.4 multi-GPU runs with a scale-sensitive
  optimizer (SGD), divide the learning rate by N; Adam-family updates are
  invariant to the gradient scale up to `eps`, so Adam runs shift only
  marginally.
- `config["num_epochs"]` is required by `train`; the legacy silent default of
  10 epochs is removed (a missing key now raises `ValueError`). A fractional
  tail that rounds to zero batches is dropped instead of producing an empty
  epoch entry.
- `evaluate` on an empty loader returns `NaN` for each metric (was `0.0`).
- The default eval metrics raise `ValueError` on a graph with no target task
  key (`evaluate_pcn` silently returned `{"energy": ..., "accuracy": 0.0}`);
  pass an explicit `metrics=` dict to evaluate such a graph.
- Eval result keys: `loss` is renamed `cross_entropy`; `target_energy` is new;
  `cross_entropy`/`perplexity` are reported only for `CrossEntropyEnergy`
  targets (previously a finite-but-meaningless cross-entropy could be reported
  on Gaussian outputs); `num_batches` and `debug=` are dropped. Eval `energy`
  (PC) now sums internal (`in_degree>0`) nodes only, matching the training
  objective (the legacy all-node sum differed only by `E(z,z)` terms on
  terminal nodes, zero under `GaussianEnergy`).
- Accuracy argmaxes on `axis=-1` (the legacy hard-coded `axis=1` mis-reduced
  rank>2 outputs).
- AR-backprop objective is per-sample (sum over sequence positions ÷ batch),
  not the legacy per-token mean: the effective learning rate shifts by
  `×seq_len`; divide legacy learning rates by `seq_len` to reproduce.
- Gaussian-output backprop objective is `0.5·precision·SSE` per sample, not an
  element-mean MSE.
- Cross-entropy numerics: the output functional clips `clip(mu, 1e-7, 1)`
  (the legacy loss used `log(mu + 1e-10)`).
- Transformer evaluation fixes: the legacy eval applied a softmax to
  `z_latent`, which for a free output already holds post-softmax probabilities
  (a double softmax), and added an external squared-error term to energy; the
  unified evaluate reads `z_mu` directly and reports pure internal energy.
  Pre-0.5 transformer eval numbers are not reproducible.
- RNG stream: keys derive as `fold_in(base_key, epoch_idx)` →
  `fold_in(epoch_key, batch_idx)` (loader-length-independent, resumable);
  0.4 training runs are not bitwise reproducible under 0.5.
- `evaluate` clamps all non-target task keys (legacy clamped only `x`);
  affects only multi-input eval batches, which no shipped code uses.
- `train_step_with_history` (dashboarding) reports per-sample internal energy
  (was an unnormalized all-node sum).
- Training metrics are per-batch dicts `{"energy", "target_energy"}` held as
  device scalars and materialized at epoch boundaries; a supplied
  `iter_callback` (or tqdm under `verbose`) forces the per-batch sync.
- The `BayesianTuner` reports train perplexity `exp(target_energy)` to Optuna
  and logs the validation `cross_entropy`; its training-energy diagnostic is
  keyed `train_energy`.

### Packaging

- `flax` removed from the dependencies — nothing imports it (the checkpointing
  follow-up uses Orbax).

## [0.4.0] - 2026-08-19
First release published to PyPI: `pip install fabricpc`. Also a muPC scaling correctness release — deep residual and pooling graphs previously trained with an attenuated signal; activations, losses, and tuned learning rates will shift. See `docs/user_guides/05_initialization_and_scaling.md`.

### Breaking changes
- `from jax_setup import set_jax_flags_before_importing_jax` becomes `from fabricpc import setup_jax`, and the `jax_platforms=` argument is renamed to `platform=`. The helper no longer has to run before `import jax` — call it any time before the first JAX computation. Calling it after the backend has initialized warns (`RuntimeWarning`) and changes nothing; previously the equivalent mistake was silent. A `platform=` argument that conflicts with a `JAX_PLATFORMS` already in the environment also warns; the environment value wins. `FABRICPC_SKIP_XLA_FLAGS=1` makes the helper leave `XLA_FLAGS` untouched, for a jax release that rejects one of the flags it writes.
- Python floor raised to 3.11 (was 3.10 which is reaching end of life)
- `optuna` moved from the core dependencies to the `[experiments]` extra, used by `fabricpc.tuning`.
- `[all]` no longer includes `[dev]`, so `pip install "fabricpc[all]"` stops installing black, ruff, mypy, and pre-commit into user environments. Contributors install `pip install -e ".[all,dev]"`.
- `SkipConnection` gained a `"skip"` slot: route the residual stream there, branch contributions to `"in"`. Construction raises when `"skip"` is unconnected (new `SlotSpec.require_connected`).
- `NodeBase.get_weight_fan_in` is replaced by `get_variance_factor(source_shape, config, weight_init) -> float`. Custom nodes must rename and accept the third argument; weighted nodes keep their existing scaling. Migration: `docs/user_guides/06_custom_nodes.md`.

### Packaging
- Published from GitHub Actions via PyPI Trusted Publishing (`.github/workflows/publish.yml`), triggered by a published GitHub release.
- `.github/workflows/test.yml` runs pytest on Python 3.11 and 3.13 for every push and pull request.
- `jaxlib` dropped from the dependencies — `jax` pins its own matched `jaxlib`. `jax` gains the floor `>=0.7.0`, the oldest release whose own Python floor is 3.11.
- `.github/workflows/test.yml` gains a leg that runs the suite against `jax==0.7.0` on Python 3.11, so the declared floor is tested rather than asserted.
- `[dev]` gains `build` and `twine` for local distribution checks.
- `.github/workflows/publish.yml` restricts the default `GITHUB_TOKEN` to `contents: read`, with the two publish jobs widening to `id-token: write` for the OIDC exchange.

### muPC scaling corrections
- Depth damping `1/sqrt(L)` now applies only to branch edges entering merge nodes. Previously every scalable edge carried it, so stream variance vanished as `e/L` with depth.
- Stems, branch interiors, stream projections, post-stream layers, and output-node readouts are now L-free — each reads a stream already held at O(1).
- `L` counts only connected skip slots, so a declared-but-unconnected slot (`LinearResidual` with no skip edge) no longer inflates the residual depth.
- `AvgPool` reports `v = 1/n` over its `n` pooled cells, so muPC amplifies its in-edge by `sqrt(n)`; previously each pool attenuated by up to `1/sqrt(n)`.
- `MaxPool` is unchanged at `v = 1`: the variance of a max depends on the input distribution, so no distribution-free correction exists.
- `StorkeyHopfield` reports its blend's variance factor rather than `fan_in`. The near-independent blend terms previously shrank variance to `1/3` at default init, compounding across chained nodes.

### New
- `InitializerBase.element_variance(shape, config)` returns the per-element variance an initializer draws, in closed form; implemented for all built-ins. `StorkeyHopfield` derives its factor from it. `StorkeyHopfield` uses it to derive `r` rather than assuming Xavier.

### Fixed
- `[tfds]` installs `tensorflow-cpu` on x86_64 Linux instead of `tensorflow`. The default Linux wheel is a CUDA build that dlopens CUDA libraries by SONAME at import. On machines whose loader search path carries a system CUDA 13 toolkit older than JAX's pip CUDA wheels, importing TF made the system `libcublas.so.13` resident first; glibc deduplicates by SONAME, so JAX's CUDA plugin bound that older copy instead of its own pip copy, failed its version check ("Outdated cuBLAS installation"), and fell back to CPU at the first TFDS data load. `tensorflow-cpu` does no CUDA probing at import, so it cannot preload the stale library. tensorflow-cpu publishes no aarch64 wheels, so aarch64 Linux keeps `tensorflow`.
- Upgrade note: `tensorflow` and `tensorflow-cpu` install the same `tensorflow` package directory, so pip will not cleanly replace one with the other. Existing environments must run `pip uninstall -y tensorflow` before reinstalling the extra.

## [0.3.2] - 2026-07-17
### New features
- Convolutional and pooling nodes: `ConvNode` (unified 1D/2D/3D) and the weight-free `MaxPool`/`AvgPool`, tensors in channels-last order. Declared output shapes are validated at `initialize_params` time, before the JIT-compiled forward pass. Demo: `examples/mnist_conv_demo.py`; see `docs/user_guides/10_api_nodes.md`.
- Autoregressive language modeling with transformer v2: `create_deep_transformer` (new `fabricpc.models` package) builds muPC-scaled graphs with internal causal masking, trained end to end via `train_autoregressive`/`evaluate_autoregressive`/`generate_autoregressive`. Demo: `examples/transformer_v2_demo.py`; see `docs/user_guides/08_training_and_evaluation.md`.
- BPE tokenization: `BpeDataLoader` (HuggingFace `tokenizers`, in the `[tfds]` extra) trains a byte-pair tokenizer on first use and caches the encoded splits. See `docs/user_guides/14_api_data.md`.
- Two-phase Bayesian hyperparameter tuning with Optuna (`fabricpc.tuning.bayesian_tuner`): Phase 1 architecture search with pruning, Phase 2 fine-tuning of continuous hyperparameters; both phases minimize validation perplexity. See `docs/user_guides/15_api_experiments.md`.
- `PlannedMultiContrastExperiment`: N-arm experiment runner with paired arms — every arm sees identical data and batch order per trial seed — and constructor-declared planned contrasts (paired t-test + Cohen's d). `ABExperiment` is now a thin 2-arm wrapper; its API is unchanged. See `docs/user_guides/15_api_experiments.md`.
- Four-arm StorkeyHopfield study in `examples/storkey_hopfield_demo.py`: accuracy gains accumulate with each Linear→StorkeyHopfield substitution under input noise, up to +13.0 pp over the MLP baseline at the noisiest setting; near zero on clean inputs.

### Breaking changes
- `pre_activation` removed from `NodeState`; `forward()` returns only the updated `NodeState` with per-sample energy, and the base gradient methods own the batch summation. Custom-node migration: `docs/user_guides/06_custom_nodes.md`.
- `None` is no longer accepted for `activation`, `energy`, or `latent_init` — `TypeError` at construction. `weight_init=None` still declares a weight-free node.
- Transformer v2 causal masking moved inside `MhaResidualNode` (`is_causal` flag); the external `mask` slot and `causal_mask` node are removed from the v2 builder. v1 graphs keep their external mask node.
- `VocabProjectionNode` default energy is now `CrossEntropyEnergy` (was `KLDivergenceEnergy`).

### Other significant changes
- Kaiming and Xavier initializers compute fan on arbitrary-rank weights; unchanged for 2D `(in, out)` weights, correct for conv kernels.
- Autoregressive trainers migrated from one-hot to integer targets: loaders yield `int32` token ids of shape `(batch, seq_len)`; one-hot encoding happens in the training step. One-hot targets still work.
- `FewShotLoader` now yields the final partial batch; it was previously dropped.
- Activations, energy functionals, and initializers are frozen at construction and validate their config values as immutable; node defaults live once, in the `__init__` signature.
- Added ruff to pre-commit for linting (formatting stays with Black). Run bash `pre-commit install` to enable.

## [0.3.1] - 2026-05-04
Internal infrastructure release: unified autodiff gradient path, muPC scaling lifted to callsites, and a package restructure that resolves circular import.

### Breaking changes — downstream migration guide
**Import path migrations.** The `builder` package is gone; topology primitives live in `core`, the assembly entry point lives in `graph_assembly`, and `graph` is renamed to `graph_initialization`. Mechanical replacements:
- `from fabricpc.builder import Edge` → `from fabricpc.core.topology import Edge`
- `from fabricpc.builder import SlotRef, GraphNamespace` → `from fabricpc.core.topology import SlotRef, GraphNamespace`
- `from fabricpc.builder import graph, TaskMap` → `from fabricpc.graph_assembly import graph, TaskMap`
- `from fabricpc.graph import initialize_params` → `from fabricpc.graph_initialization import initialize_params` (also re-exported from `fabricpc`)
- `from fabricpc.graph.state_initializer import ...` → `from fabricpc.graph_initialization.state_initializer import ...`
- `from fabricpc.graph.graph_net import compute_local_weight_gradients` → `from fabricpc.core.learning import compute_local_weight_gradients`
- `from fabricpc.utils.helpers import update_node_in_state, set_latents_to_clamps` → `from fabricpc.core.state_ops import ...` (`layernorm` stays in `utils.helpers`)

**Node API renames.** Methods on `NodeBase` (and any subclass that overrides them):
- `forward_inference(...)` → `forward_and_latent_grads(...)`. **Return signature changed** from `(NodeState, input_grads)` to `(NodeState, input_grads, self_grad)`. The third value is `dE/dz_latent` for this node only, unscaled; the inference loop scales it and accumulates into `state.latent_grad`. Subclasses that override this method must return the third value.
- `forward_learning(...)` → `forward_and_weight_grads(...)`.

**muPC scaling lifted out of nodes.** `NodeBase._apply_forward_scaling` is removed. Node forward/grad methods are now pure autodiff. Pre-scaling of inputs and post-scaling of input/self/weight grads are applied by the inference and learning loops via `fabricpc.core.scaling.{scale_inputs, scale_input_grads, scale_self_grad, scale_weight_grads}`. Custom nodes with a hand-written `forward_inference`/`forward_learning` override should drop any internal scaling and follow the new contract; see `nodes/linear_explicit_grad.py` (extracted from `linear.py`) for the reference pattern.
**muPC contract for non-variance-scalable slots changed.** Edges arriving at slots with `is_variance_scalable=False` are now **omitted** from `MuPCScalingFactors.{forward_scale, topdown_grad_scale, weight_grad_scale}` rather than populated with 1.0. Callsites treat missing keys as no-op pass-through. This preserves input dtype across the boundary (an `x * 1.0` previously promoted integer token indices to float). Forks that read these dicts directly must use `dict.get(k, 1.0)` or membership checks.
**Integer clamps now flow through to terminal source nodes.** State initializers propagate the clamp dtype onto `z_latent` for clamped nodes; other `NodeState` fields stay float. Callers feeding `EmbeddingNode` should clamp with integer dtype (e.g. `jnp.int32` token indices) — `EmbeddingNode.forward` no longer casts internally, and `train_autoregressive._generation_step` no longer casts indices to float. The `EmbeddingNode` "in" slot is now `is_variance_scalable=False`.
**`StorkeyHopfield`.** `accumulate_hopfield_energy_and_grad(...)` → `accumulate_hopfield_energy(...)`. The Hopfield latent gradient is no longer accumulated manually — autodiff in `forward_and_latent_grads` handles it.
**Removed duplicates / dead code.** `compute_local_weight_gradients_ar` (was a near-duplicate of `compute_local_weight_gradients`), `GraphStructure._topological_sort` (duplicate of the canonical implementation in `graph_assembly`), and the empty `fabricpc/graph_initialization/graph_net.py` shim are gone.
**Other.** `LinearExplicitGrad` moved from `fabricpc/nodes/linear.py` to `fabricpc/nodes/linear_explicit_grad.py` (still re-exported from `fabricpc.nodes`). Forced `float32` dtype removed from state initialization. RNG variable renamed: `node_keys` → `rng_keys`. New `ActivationBase.jacobian()` hook with `SoftmaxActivation.jacobian()` implemented for explicit-gradient overrides.
### Verification
`pytest tests/ -x`: 127 passed. Demos (`mnist_demo.py`, `transformer_v2_demo.py`, `resnet18_cifar10_demo.py`) run clean.

## [0.3.0] - 2026-04-17
- muPC scaling supports arbitrary DAG topologies with correct per-edge scaling, per-slot computation. Scaling formula is `a = gain / sqrt(fan_in * K_slot * L)` where K_slot is the per-slot in-degree and L is the residual depth (number of nodes with skip connection slots along the longest path).
- Stable training demonstrated on networks with 100+ layers with muPC scaling. 
- Associative memory is now a composable network component with `StorkeyHopfield` node: combines PC prediction-error energy with Hopfield attractor energy.
- Consolidated multi-GPU trainer into `train.py`.
- Comprehensive documentation in docs/user_guides folder.
- Added `is_variance_scalable` and `is_skip_connection` attributes to `SlotSpec` for fine-grained control over which edges receive muPC scaling.
- Added `SkipConnection` node: passthrough node with `is_variance_scalable=False` for residual/skip paths. Prevents exponential signal decay in deep residual networks.
- Added `LinearResidual` node: combines linear transform and +skip sum in one PC node with dual slots ("in" scaled, "skip" unscaled). Halves graph depth compared to Linear + SkipConnection pattern.
- Added `jacobian_gain()` to activation functions for gradient compensation in deep networks with saturating activations (tanh, GELU, HardTanh).
- Improved internal variance scaling in TransformerBlock with 1/sqrt(2) residual connections and position-dependent attention variance compensation.

## [0.2.9] - 2026-03-17
- Added transformer_v2 nodes and example decomposing transformer blocks to use PC inference at the attention and feedfordward layers. See examples/transformer_v2_demo.py for details.
- Improved training stability and inference convergence of the v1 transformer block by gradient clipping and residual connections. See examples/transformer_demo.py for details.
- Refactored optimizer integration to use Optax directly. Trainer signature is now train_pcn(..., optimizer=optax.adamw(0.001, weight_decay=0.001))
- Refactored nodes to use weight initializer objects instead of config dicts. New API is node = Linear(shape=(128,), ..., weight_init=XavierInitializer())
- Refactored inference to use algorithm abstraction. New API is structure = graph(nodes=[...], edges=[...], task_map, inference=InferenceSGD(eta_infer=0.05, infer_steps=20))
- Refactored Aim TrackingConfig parameters to improve configurable logging intervals.
- Added ABExperiment class for comparing model variants statistically.
- Added a fixed scaling factor argument to IdentityNode for better control over signal propagation.

## [0.2.8] - 2026-02-25
- Refactored model definition to be object based rather than purely config based. Existing model configs can be easily adapted to new format. See examples folder.
- Nodes now require class constructors instead of config dicts. Activation functions should be called like type(actfn_instance).forward(x, actfn_instance.config);
- Removed registry pattern for nodes, energy functionals, and other components in favor of explicit imports and class constructors. No registration decorators.

## [0.2.7] - 2026-02-18
- Add JAX-compatible MNIST data loader. Removed pytorch dependency from project.
- Enhanced documentation and comments across multiple files for clarity. Refactored inference to ignore energy of nodes that do not have energy (e.g. terminal input nodes).
- Added Aim integration for comprehensive experiment tracking and visualization. docs/user_guides/aim_tensorboard_guide.md provides instructions for setting up Aim and using it with FabricPC.

## [0.2.6] - 2026-01-06
- Fixed multi-GPU training to correctly use graph state initializer from GraphStructure config.
- Aligned gradient computation in multi-GPU training with single-GPU Hebbian learning.

## [0.2.5] - 2025-12-25
- Added v1 TransformerBlock encapsulating multi-head attention, layer normalization, and feedforward networks using Rotary Position Embeddings (RoPE)
- Refactored state initialization: renames "distribution" to "global", adds "node_distribution", and removes fallback configurations.
- Unifies output metric computation across training modules and returns both energy and cross-entropy for autoregressive training.

## [0.2.4] - 2025-12-24
- Added support for custom initializers with registry pattern. Introduced `InitializerBase` and `StateInitializerBase` classes for extensibility.
- Replaced initialize_weights() and initialize_state_values() with fabricpc.core.initializers.initialize() function.
- Added config attribute to GraphStructure class and field "graph_state_initializer".

## [0.2.3] - 2025-12-18
- Change Linear node default behavior to perform matmul on the last tensor dimension. Flattening inputs now requires flag `flatten_input=True`.
- Removed gain_mod_error from NodeState, as it was not used by anything other than explicit grad linear node.
- Added softmax and Gelu activation functions.
- Added KL Divergence energy functional.

## [0.2.2] - 2025-12-05
- Unified config validation and registry pattern across nodes, energy functionals, and activations
- Custom objects now follow a consistent extensibility pattern with `CONFIG_SCHEMA` and `@register_*` decorators
- Node construction delegated to `NodeBase.from_config()` for cleaner separation of concerns
- CONFIG_SCHEMA is now a required class variable for easier access and introspection

## [0.2.1] - 2025-12-04
- Node autograd is the default behavior now; can override by subclassing a node and implementing manual gradients
- N-dimensional tensor support: breaking changes to shape conventions
  - Linear nodes: shape=(features,) e.g., (128,) for 128-dimensional vector
  - 2D Conv nodes: shape=(H, W, C) e.g., (28, 28, 64) for 28x28 image with 64 channels (NHWC)
- Plugin architecture for custom nodes with two choices for registration: decorator or setuptools entry points
