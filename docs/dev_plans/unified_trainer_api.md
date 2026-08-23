# Unified Trainer API — energy-framed PC/backprop trainer (fabricpc 0.5.0)

## Context

The repo carries four training harnesses with duplicated code and divergent contracts:
`fabricpc/training/train.py` (PC), `train_backprop.py`, `train_autoregressive.py` (PC-AR +
backprop-AR), and the in-progress `unified_trainer.py` (branch `feature/unified-trainer`, commit
772795b, purely additive). Duplication is extensive: batch conversion inlined 8×, clamp assembly 9×,
the energy sum open-coded 8×, four structurally identical epoch loops, plus a fifth PC step in
`utils/dashboarding/inference_tracking.py` (whose line-159 TODO names this exact consolidation).
Signatures diverge in argument order, step-return arity (3/4/5-tuples), callback kwargs, config
keys, and eval result keys.

This plan replaces all four with one API. Core semantic positions:

1. **Backprop is framed in energy, not loss.** Both algorithms clamp identically during training
   (input and target). The backprop objective is the energy of the clamped nodes — the negative log
   probability the output node's energy functional assigns to the clamped target given the
   feedforward prediction — differentiated globally with `jax.value_and_grad`. PC differs only in
   that inference settles the latents first, the objective sums every internal node's energy, and
   gradients are local.
2. **The clamping mechanism is shared and graph-derived.** One `build_clamps` builds the same clamp
   dict for both algorithms; one-hot conversion is derived from the target dtype and the causal
   mask from the `TaskMap`, so no caller mode flag exists. There is no `autoregressive` parameter
   anywhere: with the mask graph-derived and the one-hot dtype-derived, the flag has zero
   behavioral effect, so the in-progress `unified_trainer.py`'s parameter is not carried over.
3. **Training is resumable at the top level.** Every legacy harness creates optimizer state
   internally and discards it, so a second `train` call on the returned params silently resets
   Adam moments and any optax schedule's count. `train` accepts and returns optimizer state and a
   step counter; the epoch callback receives them. This is the prerequisite for the Q3
   checkpointing follow-up (`docs/dev_plans_archive/model_checkpointing.md`).
4. **Multi-device runs on jit + `NamedSharding`, not pmap.** FabricPC's stated scope is training
   large transformer models with data and model sharding; pmap is JAX's legacy single-axis API and
   cannot express model parallelism. Mesh axis names are fixed now: `"data"` (used) and `"model"`
   (reserved).

## Decisions

| Decision | Choice |
|---|---|
| Legacy API | Clean break at 0.5.0. Delete all 17 legacy `fabricpc.training` exports + 3 aliases + top-level `train_pcn`/`evaluate_pcn`. Migrate every internal caller in this change. No shims. CHANGELOG migration table for PyPI users. |
| Train return | `TrainResult(params, opt_state, step, iter_results, epoch_results)` NamedTuple. `train` accepts `opt_state=None` (created via `optimizer.init(params)` when None) and `start_epoch=0` for resume. |
| Mode axes | One axis: `algorithm ∈ ("pc", "backprop")`, annotated `Literal`. No `autoregressive` flag — mask from `TaskMap`, one-hot from dtype, `perplexity` from the target functional. |
| AR objective scale | Per-sample energy in all modes (sum over sequence positions ÷ batch size). AR-backprop effective learning rate shifts ×seq_len vs the legacy per-token mean — CHANGELOG flag. |
| Device parallelism | jit + `NamedSharding` over an optional `mesh` (axis `"data"`; `"model"` reserved). Absent mesh = single device, same jitted step. `pmap`, `pmap_single_device`, and the four `device_utils` helpers are deleted, not ported. |
| RNG stream | `fold_in(base_key, epoch_idx)` → `fold_in(epoch_key, batch_idx)`. The key is consumed only by latent initialization; inference is deterministic. Bitwise spot-check against the legacy split-chain stream runs **before** the switch (implementation step 3). |
| Callback contract | `epoch_callback(ctx: EpochContext)` with `EpochContext(epoch_idx, step, params, opt_state, structure, config, rng_key, metrics)`; `iter_callback(epoch_idx, batch_idx, metrics)`. Callback exceptions propagate (guaranteed and tested — tuner pruning depends on it). A non-None return replaces the stored history entry. |
| Eval metrics | Pluggable: `evaluate(..., metrics=None)` takes a dict of named metric functions (per-sample `(value, weight)` contract with a `finalize` transform — design below); `None` selects graph-derived defaults from `default_metrics(structure, algorithm)`: `target_energy` and `accuracy` always, `cross_entropy`/`perplexity` when the target functional is `CrossEntropyEnergy`, `energy` for PC. `argmax` on `axis=-1` (the legacy hard-coded `axis=1`, `train.py:532-533`, mis-reduces rank>2 outputs). No-target graphs raise under the defaults. |
| Metrics materialization | Device scalars inside the loop; converted to floats at epoch boundaries. A supplied `iter_callback` (or tqdm postfix under `verbose`) forces the per-batch sync and is documented as doing so. |
| Tuner pruning signal | Train perplexity `exp(metrics["target_energy"])` — free, since the clamped target node's energy under `CrossEntropyEnergy` is the teacher-forced cross-entropy. Energy is not a performance metric. |
| Dependencies | `flax` dropped from `pyproject.toml` (nothing imports it; the checkpointing follow-up uses Orbax). |
| Parity gate | Bitwise for `mnist_demo` PC (before the RNG-stream switch), then permanent in-test 1e-12 references. Transformer demos are smoke-only; reproduction of pre-unification transformer results is disregarded (this plan deliberately changes those numbers). |

## Design

### Symbol table

- `structure` — the `GraphStructure` (static topology: nodes, edges, `task_map`, `node_order`,
  config). Entirely static pytree aux data, so per-node introspection is free at trace time.
- `params` — the `GraphParams` pytree of per-node weights/biases; the only learnable state.
- `opt_state` — the optax optimizer state for `params`; threaded through `train` for resumption.
- `state` — a `GraphState`: per node, `z_latent` (the latent value), `z_mu` (the prediction from
  upstream latents), `error`, `energy` (per-sample, shape `(batch,)`).
- `clamps` — a dict `{node_name: array}`; a node is clamped iff its name is a key. Consumed by the
  state initializer (sets `z_latent` to the clamp) and the inference loop (skips latent updates).
- **target nodes** — the clamped nodes with `in_degree > 0`, i.e. the nodes `task_map` maps
  non-`x` batch keys to (the output node in every current graph).
- `E(z, μ)` — the node's energy functional applied to its latent `z` and prediction `μ`;
  `CrossEntropyEnergy` gives `-Σ z·log μ` (negative log probability), `GaussianEnergy` gives
  `0.5·precision·Σ(z-μ)²`, summed over all non-batch dimensions.
- `mesh` — an optional `jax.sharding.Mesh` with axis `"data"`; batches are sharded on the leading
  axis with `NamedSharding(mesh, P("data"))`, params replicated with `P()`.
- `step` — the count of optimizer updates applied in this `train` call. Not offset by
  `start_epoch`: a resumed run counts from 0 again, so a caller logging a global step adds its
  own checkpoint offset.

### The unified step (one axis: `algorithm`)

| sub-step | PC | backprop |
|---|---|---|
| 1 batch→dict | `convert_batch` | (shared) |
| 2 build clamps | `build_clamps(clamp_target=True)` — x AND targets | **(shared, identical)** |
| 3 produce state | `initialize_graph_state` + `run_inference` (settle) | `initialize_graph_state` with `FeedforwardStateInit` (single forward pass, no inference) |
| 4 objective | `graph_energy` over all `in_degree>0` nodes ÷ batch | `graph_energy` over **target nodes** ÷ batch (shared function, different node subset) |
| 5 gradients | `compute_local_weight_gradients` (local, per node) | `jax.value_and_grad(objective, has_aux=True)` over steps 3–4 (global) |
| 6 apply update | optax `update` + `apply_updates` | (shared) |

Verified mechanics (read in source, this branch):

- `FeedforwardStateInit.initialize_state` (`graph_initialization/state_initializer.py:294-301`)
  keeps a clamped node's `z_latent` = clamp and stores the freshly computed `z_mu`/`error`/`energy`
  from `node_class.forward`. After a feedforward pass with target `y` clamped, the target node's
  stored energy is exactly `E(y, μ(params))` — cross-entropy under `CrossEntropyEnergy` (modulo
  `clip(μ,1e-7,1)` vs legacy `μ+1e-10`), `0.5·precision·SSE` under `GaussianEnergy`. So `loss_type`
  is retired: **the output node's energy functional in the graph definition selects the loss.**
- Unclamped nodes keep `energy = 0` under feedforward init, and free/source nodes get `E(z,z)` only
  after PC inference — hence the explicit `in_degree>0` / target-node filters.
- No gradient blockers on the differentiated path (no `stop_gradient` in the package; one-hot runs
  outside the closure; clamp-dtype validation is trace-time Python). Differentiating through
  `FeedforwardStateInit` is already proven by the in-progress branch's 1e-12 backprop parity tests.
- Under a sharded batch, XLA inserts the cross-device reduction where sharded per-sample
  quantities contract to replicated weight gradients; no explicit `pmean`/`axis_name` plumbing.
  **Gradient-scale behavior change:** that contraction yields the global batch **sum**, matching
  the single-device path; the legacy pmap path applied `pmean` over per-device shard sums, i.e.
  Σ_global/N on N devices — its gradients disagreed with main's own single-device path by 1/N.
  The new semantics are the correct ones, but multi-GPU learning rates tuned on 0.4 shift by ×N
  for scale-sensitive optimizers — CHANGELOG behavior note required.

### Metrics dict (step output, callbacks, results)

- `metrics["energy"]` — the per-sample objective from step 4 (what the gradients descend).
  **Algorithm-dependent quantity**: all internal nodes for PC, target nodes only for backprop.
  Cross-algorithm comparison of this key is invalid — documented here, in the user guide, and in
  the CHANGELOG.
- `metrics["target_energy"]` — summed target-node energy ÷ number of predictions
  (`prod(target.shape[:-1])`: batch·seq for sequences, batch for classification). Under
  `CrossEntropyEnergy` this is the teacher-forced per-token cross-entropy; `exp` of it is train
  perplexity. Note the two keys use different normalizations (÷batch vs ÷predictions) — stated in
  the docstring.

Both are computed inside the jitted step from the already-materialized state — no extra forward
work. They remain device scalars in the loop; `iter_results[epoch][batch]` and
`epoch_results[epoch]` hold floats converted at epoch boundaries, unless the corresponding
callback returns a replacement value. A supplied `iter_callback` receives floats (forcing the
per-batch device sync — documented cost).

## Public API

```python
# fabricpc/training/trainer.py
ALGORITHMS = ("pc", "backprop")   # algorithm: Literal["pc", "backprop"]

class TrainResult(NamedTuple):
    params: GraphParams
    opt_state: optax.OptState
    step: int                 # optimizer updates applied in this call
    iter_results: list
    epoch_results: list

class EpochContext(NamedTuple):
    epoch_idx: int
    step: int
    params: GraphParams
    opt_state: optax.OptState
    structure: GraphStructure
    config: dict
    rng_key: jax.Array        # the base training key; derive per-epoch keys via fold_in
    metrics: Dict[str, float] # epoch means

def convert_batch(batch_data) -> Dict[str, jnp.ndarray]
def create_causal_mask(seq_len: int) -> jnp.ndarray

def build_clamps(batch, structure, *, clamp_target: bool) -> Dict[str, jnp.ndarray]

def make_train_step(structure, optimizer, *, algorithm="pc", mesh=None)
    # -> step(params, opt_state, batch, rng_key)
    #    -> (params, opt_state, metrics: dict, final_state: GraphState)
    # jitted; validation (algorithm prerequisites) runs once at build time.
    # Public escape hatch: returns final_state, does NOT donate its inputs
    # (external callers legitimately reuse initial params). The internal loop
    # uses a private variant that drops final_state and donates params/opt_state
    # (jax.jit(..., donate_argnums=(0, 1))) — final_state as a jitted output
    # cannot be dead-code-eliminated and keeps every node's latents alive past
    # the update; donation removes a full extra params+opt_state copy from peak
    # memory.

def train(params, structure, train_loader, optimizer, config, rng_key, *,
          algorithm="pc", opt_state=None, start_epoch=0, mesh=None,
          verbose=True, epoch_callback=None, iter_callback=None,
) -> TrainResult

def evaluate(params, structure, test_loader, config, rng_key, *,
             algorithm="pc", mesh=None, metrics=None) -> Dict[str, float]
    # metrics: Dict[str, EvalMetric | callable] | None.
    # None -> default_metrics(structure, algorithm). Given -> exactly those.

# fabricpc/training/metrics.py
class EvalMetric(NamedTuple):
    fn: Callable  # (state: GraphState, batch: dict, structure) -> (value, weight),
                  # both per-sample arrays of shape (batch,)
    finalize: Callable = lambda x: x   # applied once, after global aggregation

accuracy, cross_entropy, perplexity, target_energy, internal_energy  # built-ins
def default_metrics(structure, algorithm) -> Dict[str, EvalMetric]

# fabricpc/training/generation.py
def generate(params, structure, prompt, max_new_tokens, rng_key,
             temperature=1.0, top_k=None, top_p=None,
             algorithm="pc") -> jnp.ndarray
    # Same algorithm split and validation as evaluate: "pc" settles via
    # run_inference each step, "backprop" samples from the feedforward pass.
    # Without it, a backprop-only graph (inference=None, which
    # _validate_algorithm permits) crashes inside run_inference, and a
    # backprop-trained model silently gets PC settling at generation time
    # while its evaluate takes the feedforward pass.

# fabricpc/core/energy.py
def graph_energy(state, structure, *, node_names=None) -> jnp.ndarray
    # Total energy summed over the selected nodes and the batch.
    # node_names=None selects all in_degree>0 nodes. Iterates structure.node_order
    # (float addition is not associative; the 1e-12 parity tests fix the order).
    # Callers normalize explicitly (÷ state.batch_size per-sample, ÷ prediction
    # count per-token). Replaces the 8 open-coded sums and supersedes
    # feature/total-graph-energy's total_graph_energy (whose internal_only flag
    # cannot express the backprop target-only subset).
```

Defaults realize the requirement: `train(...)` with no mode arguments is PC, single-device.
Everything after `rng_key` is keyword-only, so `functools.partial(train, algorithm="backprop")`
satisfies the unchanged `ExperimentArm` positional prefix; the arm's result unpack migrates to
`TrainResult` field access (one site, `ab_experiment.py:344`).

There is no `device_utils.py` and no `use_tqdm`: `verbose=True` shows tqdm bars and epoch
summaries, `verbose=False` is silent.

`config` reads only `num_epochs`, which is **required** — a missing key raises `ValueError`, no
silent default (the same fail-fast stance as the retired keys; `ab_experiment`'s per-epoch time
normalization depends on the value being explicit). Fractional values are supported; a partial
epoch's `epoch_results` entry is the mean over the batches actually run, and a fractional tail
that rounds to zero batches is dropped rather than appending an empty epoch entry and invoking
the callback on empty metrics. The trainer treats `config` otherwise as an opaque
pass-through to callbacks and `ExperimentArm` — in particular, settling parameters (`infer_steps`,
`eta_infer`) live in the inference object inside `structure.config`, not here (docstring states
this). `train`/`evaluate` raise `ValueError` if `config` contains a retired key (`"loss_type"`,
`"use_causal_mask"`) with one-line migration text — fail-fast, not a fallback.

### Resume, RNG, and the step counter

- `opt_state=None` → `optimizer.init(params)`; passing a restored `opt_state` preserves optimizer
  moments and any optax schedule's count.
- Per-epoch key: `fold_in(rng_key, epoch_idx)`; per-batch key: `fold_in(epoch_key, batch_idx)`.
  Keys are independent of loader length. `start_epoch=k` offsets `epoch_idx`, so
  `train(..., opt_state=ckpt, start_epoch=k)` reproduces the uninterrupted run's stream exactly —
  interrupted-equals-uninterrupted is a testable bitwise property.
- The RNG contract, documented in the `train` docstring: the key affects only latent
  initialization (`initialize_graph_state`); `run_inference` is deterministic; there is no dropout
  or noise anywhere in the package.
- `step` counts optimizer updates applied by this call; exposed per epoch in `EpochContext` and
  finally in `TrainResult`. The checkpoint follow-up persists `(epoch_idx, step)` from the context.

### Device parallelism (jit + sharding)

- `mesh=None`: plain `jax.jit`. `mesh` given: params/opt_state placed with
  `NamedSharding(mesh, P())` once at loop start, each batch with `NamedSharding(mesh, P("data"))`;
  the same jitted step serves both cases.
- Ragged batches, one policy per function: training skips a batch not divisible by the `"data"`
  axis size with a one-time warning (legacy behavior); `evaluate` zero-pads, and trims via
  per-sample weights so every sample counts exactly once.
- Multi-device testing needs no API cooperation:
  `XLA_FLAGS=--xla_force_host_platform_device_count=2` plus an explicit mesh.
- Model parallelism is future scope: the `"model"` axis name is reserved now; per-node parameter
  sharding rules arrive later as an additive argument, not a signature break.

### `build_clamps` (absorbs legacy `build_train_clamps`)

1. For each batch key present in `structure.task_map`: clamp `task_map[key]` if `clamp_target`,
   else clamp only non-target keys (evaluation: inputs clamped, targets free).
2. Non-float-target one-hot (training, both algorithms): if a clamped target array has non-floating
   dtype (int **or bool** — `_validate_clamp_dtypes` at `state_initializer.py:333` accepts floating
   only), `jax.nn.one_hot(y, num_classes)` with `num_classes` from the target node's
   `NodeInfo.shape[-1]`. Fixes the in-progress branch's gap where stock int32 token loaders
   (`utils/data/dataloader.py:268` yield; arrays materialized at `:336`/`:466`) hit `TypeError` in
   `_validate_clamp_dtypes`.
3. Causal mask, graph-derived: if `"causal_mask" in structure.task_map`, inject
   `broadcast_to(tril[None,None], (batch, 1, seq, seq))` at `task_map["causal_mask"]`, with `seq`
   read from the mask node's declared shape (`node_info.shape[-1]`) — genuinely graph-derived,
   not from a hard-coded `batch["x"]` whose absence would be a bare `KeyError`; otherwise no-op.
   Target clamps are shape-validated here (`(batch, *node.shape)` after one-hot) so a wrong-shape
   target raises an actionable `ValueError` instead of an opaque XLA broadcast error (the legacy
   AR path had such a check; the shape rule also makes rank-1 target clamps unrepresentable,
   which keeps the trainer's `n_predictions = prod(shape[:-1])` consistent with
   `metrics._predictions_per_sample`). No caller flag: no non-AR graph in the repo declares the key, v1 transformer graphs
   declare the mask node in their `TaskMap`, and v2 graphs mask internally
   (`MhaResidualNode(is_causal=True)`), so the in-progress `_require_causal_mask_node` raise on v2
   graphs is deleted.

### Step pseudocode

```python
def _batch_grads(params, batch, structure, rng_key, *, algorithm):
    # Batch size reads the leading axis of the first task-mapped batch key,
    # never an arbitrary batch value: an extra key whose leading axis is not
    # the batch would yield a wrong size (and, under mesh, a hard sharding
    # failure). A batch with no task-mapped key raises.
    batch_size = <leading axis of the first batch key in structure.task_map>
    clamps = build_clamps(batch, structure, clamp_target=True)           # (2) identical for both
    target_nodes = tuple(n for n in clamps
                         if structure.nodes[n].node_info.in_degree > 0)  # static
    # backprop with no clamped target has an empty objective -> raise at trace time
    if algorithm == "pc":
        state = initialize_graph_state(structure, batch_size, rng_key,
                                       clamps=clamps, params=params)     # (3a)
        state = run_inference(params, state, clamps, structure)          # (3a) settle
        energy = graph_energy(state, structure) / batch_size             # (4) all internal
        grads = compute_local_weight_gradients(params, state, structure) # (5a) local
    else:  # backprop
        def objective(p):
            state = initialize_graph_state(structure, batch_size, rng_key,
                                           clamps=clamps, params=p)      # (3b) feedforward only
            return graph_energy(state, structure,
                                node_names=target_nodes) / batch_size, state  # (4) targets only
        (energy, state), grads = jax.value_and_grad(objective, has_aux=True)(params)  # (5b) global
    n_predictions = <static prod of target shape[:-1]>                   # batch*seq or batch
    metrics = {"energy": energy,
               "target_energy": graph_energy(state, structure,
                                             node_names=target_nodes) / n_predictions}
    return grads, metrics, state

def _make_step(structure, optimizer, *, algorithm, with_state, donate):
    def step(params, opt_state, batch, rng_key):
        grads, metrics, state = _batch_grads(...)
        params, opt_state = _apply_update(params, opt_state, grads, optimizer)  # (6) optax
        return (params, opt_state, metrics, state) if with_state \
          else (params, opt_state, metrics)
    return jax.jit(step, donate_argnums=(0, 1) if donate else ())
# make_train_step: with_state=True, donate=False (public escape hatch)
# internal loop:   with_state=False, donate=True
```

Training loop (`_run_training_loop`, private): tqdm, fractional epochs, ragged-batch
skip-with-warning, fold_in key derivation, device-scalar metric accumulation with epoch-boundary
materialization, `EpochContext` construction, callback dispatch with no exception guard (the
propagation guarantee). Backprop thereby gains tqdm and multi-device data parallelism.

### `evaluate` and the metric system

Shared eval clamping `build_clamps(clamp_target=False)` in both modes; PC settles via
`run_inference`, backprop takes the feedforward pass.

Eval metrics are pluggable because the reported quantity is the user's choice, not the trainer's:
classification wants accuracy, transformers perplexity, some users report cross-entropy for their
own comparisons, and custom metrics must be possible without forking `evaluate`. The design
separates the three concerns a metric has:

- **Per-batch computation** — `EvalMetric.fn(state, batch, structure) -> (value, weight)`, two
  per-sample arrays of shape `(batch,)`, run inside the jitted eval step on the settled state (no
  extra forward work). Examples: accuracy → (correct predictions per sample, predictions per
  sample); per-token cross-entropy → (Σ_token CE per sample, seq_len per sample).
- **Aggregation** — owned by `evaluate`: padded samples in ragged sharded batches get zero weight,
  Σvalue and Σweight accumulate across batches and devices, and the result is
  `finalize(Σvalue / Σweight)`. Because each metric carries its own denominator, per-sample and
  per-token metrics aggregate correctly and a ragged final batch never skews a mean.
- **Post-aggregation transform** — `EvalMetric.finalize`, applied once to the global mean.
  `perplexity = EvalMetric(cross_entropy.fn, finalize=jnp.exp)`: `exp` of the aggregated mean, not
  a mean of per-batch `exp`s.

`metrics=None` selects `default_metrics(structure, algorithm)` (table below); a supplied dict is
used exactly as given, so callers replace or extend at will (a bare callable is wrapped with the
identity finalize):

```python
evaluate(...)                                                        # graph-derived defaults
evaluate(..., metrics={"accuracy": metrics.accuracy, "f1": my_f1})   # exactly these
evaluate(..., metrics={**metrics.default_metrics(structure, "pc"),
                       "f1": my_f1})                                 # defaults + custom
```

The defaults mirror the training-side energy framing — the target node's own functional selects
them. `target_energy` is computed explicitly as `E(y, z_mu)` under the target node's functional
(the stored eval-state energy of a free output is zero), with non-float targets one-hotted on the
fly. The defaults raise `ValueError` on a graph with no target task key (legacy silently returned
`energy: 0.0` through the `total_samples` guard, `train.py:655`); a caller-supplied metrics dict
carries no such requirement.

| default key | when | definition |
|---|---|---|
| `target_energy` | always | Σ `E(y, z_mu)` under the target node's functional ÷ total predictions. Per-token CE for `CrossEntropyEnergy` targets; `0.5·precision·MSE`-scale for `GaussianEnergy`. |
| `accuracy` | always | total `argmax(z_mu, -1) == argmax(y, -1)` ÷ total predictions (int targets compared directly). Documented as argmax-based: meaningful for class-like targets, not for continuous ones. |
| `cross_entropy` | target functional is `CrossEntropyEnergy` | identical to `target_energy` in that case; kept as the conventional name. |
| `perplexity` | target functional is `CrossEntropyEnergy` | `exp(cross_entropy)`. |
| `energy` | `algorithm="pc"` | per-sample `graph_energy` over internal (`in_degree>0`) nodes — the same definition as the training objective. The legacy all-node eval sum differed only by `E(z,z)` terms on terminal nodes (zero under `GaussianEnergy`); CHANGELOG flag. |

`accuracy` stays unconditional: `mnist_conv_demo.py` and `jpc_fc_resnet_compare.py` legitimately
score one-hot class targets under `GaussianEnergy` outputs, so keying `accuracy` on the functional
would break them. `cross_entropy` and `perplexity` stay conditional so no meaningless
cross-entropy is reported on non-CE outputs. The functional check is `isinstance(node_info.energy,
CrossEntropyEnergy)`, free at trace time (`GraphStructure` is static aux; the idiom is already
tested at `test_energy.py:177-178`).

`evaluate_transformer` folds in as `evaluate(...)` on the transformer graph — the key set
`tests/test_transformer_nodes.py:443-466` asserts, minus `num_batches`, plus `target_energy`. Two
deliberate numerical fixes to flag: the legacy transformer eval applied `softmax` to `z_latent`,
which for a free output already holds post-softmax probabilities (a double softmax), and added an
external squared-error term to energy; the unified evaluate reads `z_mu` directly and reports pure
internal energy. `loss` is renamed `cross_entropy`; `num_batches` and the `debug=` kwarg are
dropped.

### Callbacks and downstream consumers

- `epoch_callback(ctx: EpochContext)`: one context argument that grows by field addition, never by
  positional breakage (the legacy AR-only `energy=`/`ce_loss=` kwargs are the precedent for how
  positional contracts rot). `iter_callback(epoch_idx, batch_idx, metrics)` with float metrics.
  Contract guarantees, stated in docstrings and tested: callback exceptions propagate (tuner
  pruning is exception-based); a non-None return replaces the stored history entry (dashboards
  store mid-training eval results this way).
- `ExperimentArm` (`experiments/ab_experiment.py`): arms pass `train` /
  `functools.partial(train, algorithm="backprop")` as `train_fn`, same for `evaluate` as `eval_fn`;
  the single result unpack at `:344` becomes `result.params`.
- `BayesianTuner` (`tuning/bayesian_tuner.py`): `epoch_callback(ctx)` reads `ctx.metrics`; drops
  the `"use_causal_mask"` config injection; reports `exp(metrics["target_energy"])` (train
  perplexity) to Optuna and keeps the divergence guard on `metrics["energy"]`; final score from
  `evaluate(...)["perplexity"]`. **Including `_log`** (`bayesian_tuner.py:230-247`), which reads
  the renamed `loss` key and would otherwise silently log 0.0.
- `utils/dashboarding/callbacks.py`: the four factories take the new contracts
  (`create_epoch_callback`'s eval-and-return behavior is exactly the return-replaces-history rule);
  `create_detailed_iter_callback` remains custom-loop-only on `make_train_step`'s `final_state`
  and **migrates to the step's contract** — `(epoch_idx, batch_idx, metrics: dict, final_state)`,
  reading `metrics["energy"]` — with a custom-loop example in guide 09; leaving it on the pre-0.5
  `(..., energy: float, ...)` shape would strand it with zero valid call sites. The vestigial
  `batch_size` normalization parameters are deleted from all factories (energy is per-sample).
- `utils/dashboarding/extractors.py`: energy extraction points at `graph_energy`.
- `utils/dashboarding/inference_tracking.py:train_step_with_history`: same signature, body rebuilt
  on `build_clamps` → `initialize_graph_state` → `run_inference_with_history` → `graph_energy` →
  `compute_local_weight_gradients` → optax. Its energy becomes internal-only and per-sample
  (was all-node, unnormalized) — CHANGELOG flag. Deletes the fifth duplicate step and the TODO.

## Alternatives considered

- **Trainer class vs module functions.** A `Trainer(structure, optimizer, ...)` object would cache
  jitted steps across calls. Rejected: the codebase is uniformly functional-JAX, the
  `ExperimentArm`/callback ecosystem passes trainers as function values, and `make_train_step`
  already gives callers a persistent jitted step. `TrainResult` supplies the state-bundle benefit
  without the class.
- **Strategy objects / enum dispatch vs string flag.** A layer for exactly two variants whose
  difference is three lines inside one function. String flag with build-time validation kept;
  branches resolve at trace time.
- **Loss-based backprop objective** (separate `_metric_cross_entropy` machinery with
  `metric_fn`/`metric_name`/`loss_type`). Rejected per the energy framing; also removes that
  function's int-target broadcast bug.
- **Per-token AR normalization** — preserves legacy backprop gradient scale but makes the energy
  definition mode-dependent. Rejected: per-sample everywhere.
- **`use_causal_mask` caller flag** — means different things on v1 vs v2 transformer graphs.
  Rejected for graph derivation.
- **Deprecation shims for one release** — rejected: clean break at 0.5.0.
- **`ExperimentArm.algorithm` field** — couples the harness to the trainer signature;
  `functools.partial` keeps it trainer-agnostic.
- **Tuner pruning on validation perplexity** — objective-aligned but costs one eval pass per epoch;
  train perplexity is free from `metrics["target_energy"]`.

- **3-tuple return, opt_state internal (the legacy contract).** Rejected: non-resumable training —
  every legacy loop creates and discards optimizer state (`train.py:344`, `unified_trainer.py:579`),
  and the abandoned `feature/model-checkpointing` branch's `mnist_demo` comment documents the wall
  ("cannot thread real optimizer momentum across the save/load boundary"). The clean break is the
  cheapest moment to fix the 3/4/5-tuple arity drift.
- **`jax.pmap`, public or kept private.** Rejected: since JAX 0.4 the recommended data-parallel
  mechanism is jit over a `Mesh` with `NamedSharding` (the repo floor is jax>=0.7,
  `pyproject.toml:34`); pmap is single-axis and cannot express the stated model-sharding scope; it
  carries `pmap_single_device`, four `device_utils` helpers, per-device replicated params that the
  legacy loop de-replicates before `epoch_callback` (`train.py:455`), and three divergent
  ragged-batch policies (skip-with-warning in training `train.py:412-421`, zero-pad in
  `evaluate_pcn` `:611-621`, silent drop in `evaluate_transformer` `:761-762`).
- **`autoregressive` parameter (the in-progress `unified_trainer.py` signature).** Rejected: zero
  training-side effect once mask and one-hot are graph-/dtype-derived — the only `causal_mask`
  task keys in the repo are `transformer_demo.py:216` and `test_state_initializer.py:253`, both
  v1 AR; it survives only as an eval reporting switch, replaced by the functional check. Keeping
  it as a graph-shape assertion was rejected too — this plan deletes exactly such an assertion
  (`_require_causal_mask_node`, `unified_trainer.py:89-99`) as a defect.
- **Unconditional `cross_entropy` in evaluate (legacy behavior).** Rejected: finite-but-
  meaningless numbers on `GaussianEnergy` outputs — `-Σ y·log(clip(μ,1e-7,1))` on a linear `z_mu`
  clips negative activations to 1e-7 and activations ≥1 to 1 (the `jpc_fc_resnet_compare.py`
  graphs).
- **`accuracy` keyed on the CE functional** (the strict mirror of the conditional CE/perplexity
  rule). Rejected in favor of unconditional default accuracy: `mnist_conv_demo`/
  `jpc_fc_resnet_compare` score one-hot targets under `GaussianEnergy` outputs.
- **Fixed eval key set.** Rejected: classification needs accuracy, transformers perplexity, and
  users report cross-entropy or their own quantities — a closed set forces forking `evaluate`.
  The conditional key table survives as the graph-derived *default*, not the contract.
- **Single `metric_fn`/`metric_name` (the in-progress `unified_trainer.py` extensibility).**
  Rejected: one metric at a time, per-batch-mean aggregation that mis-weights ragged batches and
  per-token metrics, and no post-aggregation transform (a mean of per-batch `exp`s is not
  perplexity). The `EvalMetric` `(value, weight)` + `finalize` contract replaces it.
- **Per-batch float materialization (legacy behavior).** Rejected: each `float()` blocks the
  device stream and forfeits async dispatch; `train_autoregressive.py:337-344` already demonstrated
  the deferred pattern.
- **Legacy `split(epoch_key, max_batches)` stream** (`train.py:401`). Rejected: couples keys to
  loader length, and resume at epoch k requires replaying k splits; `fold_in` makes the stream a
  pure function of `(base_key, epoch_idx, batch_idx)`. The legacy stream is retained only through
  the one-off bitwise spot-check (implementation step 3).
- **Six-positional epoch callback** (`epoch_callback(epoch_idx, params, structure, config,
  rng_key, metrics)`). Rejected: a positional contract cannot grow without breaking every
  implementor — the tuner, the main consumer, uses one of the six (`bayesian_tuner.py:114-121`) —
  and the checkpointing hook needs `opt_state` + `step`, which the positional list lacks.

## Open / deferred

- Per-node model-parallel sharding rules for `GraphParams` — future scope; the `"model"` mesh axis
  name is reserved, the argument will be additive.
- A mesh-construction convenience helper — deferred; the user guide documents
  `jax.make_mesh((jax.device_count(),), ("data",))`.
- An `IterContext` for `iter_callback` — deferred until a consumer needs more than
  `(epoch_idx, batch_idx, metrics)`.
- keep-best-K metric selection — belongs to the checkpointing PR
  (`docs/dev_plans_archive/model_checkpointing.md`).

## File changes

**Created:** `fabricpc/training/trainer.py`, `fabricpc/training/metrics.py`,
`fabricpc/training/generation.py`, `tests/test_trainer.py`, `tests/test_sharding.py`
(replaces `test_multi_gpu.py`).

**Deleted:** `fabricpc/training/{train,train_backprop,train_autoregressive,unified_trainer,multi_gpu}.py`,
`scripts/_parity_check_unified.py`, `scripts/_parity_check_ar.py`,
`tests/test_train_backprop.py`, `tests/test_unified_trainer.py`, `tests/test_multi_gpu.py`.

**Modified:** `fabricpc/core/energy.py` (+`graph_energy`), `fabricpc/training/__init__.py`
(exports: `train, evaluate, make_train_step, generate, build_clamps, convert_batch,
create_causal_mask, TrainResult, EpochContext, EvalMetric`, plus the `metrics` submodule),
`fabricpc/__init__.py` (`train, evaluate` replace
`train_pcn, evaluate_pcn`; docstring example), `pyproject.toml` (0.5.0; **drop `flax`** — nothing
imports it and the checkpointing follow-up uses Orbax), `fabricpc/experiments/ab_experiment.py`
(result unpack), `fabricpc/tuning/bayesian_tuner.py` (callback, keys, `_log`),
`fabricpc/utils/dashboarding/{inference_tracking,callbacks,extractors}.py`,
`.github/workflows/test.yml` (the 2-host-device sharding step — see test plan), plus the
callers/tests/docs below.

## Implementation steps (one PR, ordered)

1. `core/energy.py`: add `graph_energy` (`node_order` iteration; subset selection; `in_degree>0`
   default); unit tests.
2. Create `trainer.py` and `metrics.py` per the design above, **initially with the legacy
   split-chain RNG stream** (reuse the in-progress `run_training_loop`, `_validate_algo`,
   `_accuracy` bodies where they match).
3. **RNG parity gate:** one-off spot-check that new PC `train` matches legacy `train_pcn` bitwise
   on the `mnist_demo` config (identical clamp/init/inference/gradient/RNG stream expected).
4. Switch the stream to `fold_in`; then write the permanent in-test references (PC parity at 1e-12
   against an in-test hand-rolled step under the fold_in stream). Order matters: the references
   must encode the final stream.
5. Create `generation.py`: port `_generation_step`/`generate` from
   `train_autoregressive.py:377-585`, clamps via `build_clamps`.
6. Rebuild `inference_tracking.py:train_step_with_history` on the shared helpers; point
   `extractors.py` at `graph_energy`.
7. Rewire `training/__init__.py` + `fabricpc/__init__.py`; delete the five legacy modules and both
   parity scripts; bump version; drop `flax`.
8. Migrate `ab_experiment.py` (TrainResult unpack), `bayesian_tuner.py` (incl. `_log`), and
   `dashboarding/callbacks.py`.
9. Migrate examples/scripts (below).
10. Rewrite/upgrade tests (below).
11. Docs + README + CHANGELOG.
12. Verification (below).

## Caller migration

- Mechanical `train_pcn→train` + `TrainResult` unpack (`p, _, _ =` → `.params`), `evaluate_pcn→evaluate`:
  `examples/{mnist_demo,mnist_conv_demo,mupc_demo,jpc_fc_resnet_compare,mnist_multi_gpu,
  storkey_hopfield_recall,mnist_cyclic_graph,mnist_lateral_connections,storkey_hopfield_demo}.py`,
  `scripts/storkey_hopfield_diagnostic.py`. `resnet18_cifar10_demo.py` additionally rewrites its
  `epoch_callback` to the `EpochContext` contract. `mnist_multi_gpu.py` constructs an explicit
  `jax.make_mesh((jax.device_count(),), ("data",))` and passes `mesh=`.
- Manual step loops → `step = make_train_step(structure, optimizer[, algorithm="backprop"])`,
  unpack `(params, opt_state, metrics, final_state)`: `examples/mnist_advanced.py`,
  `examples/scaling/mlp_scaling.py` (positional `loss_type` argument disappears),
  `scripts/storkey_hopfield_diagnostic.py` (two jit-lambda sites).
- `examples/PC_backprop_compare.py`: **restructured, not just migrated** (roadmap decision): one
  gelu graph shared by both arms — the repo's only same-graph PC-versus-backprop comparison —
  with `functools.partial(train, algorithm="backprop")` / `partial(evaluate,
  algorithm="backprop")`; arm metric stays `accuracy` (`energy` is not cross-algorithm comparable).
- `examples/transformer_demo.py` (v1): hand-rolled AR loops → `make_train_step(...)`;
  `build_train_clamps` → `build_clamps(..., clamp_target=True)`; evals → `evaluate(...)` dropping
  `debug=`; `generate_autoregressive→generate` (top_k/top_p unchanged); remove `"use_causal_mask"`
  from config (mask now derived from the graph's `TaskMap`).
- `examples/transformer_v2_demo.py`: four trainer/eval calls → `train`/`evaluate`
  (+`algorithm="backprop"` for that mode); drop `"use_causal_mask"`;
  `metrics["loss"]`→`["cross_entropy"]`; `generate`.
- `examples/mnist_aim_tracking.py`: `evaluate_pcn→evaluate`; `train_step_with_history` energy-scale
  label update.
- `examples/transformer_tuning.py`: check base_config for retired keys.
- Recorded demo baselines: the transformer demos' docstring Results blocks were produced by the
  deleted trainers (their print format, the pre-fix eval numerics, the per-token AR-backprop
  objective). The README requires demos to match recorded baselines, so each block is either
  re-measured under 0.5 or explicitly marked pre-0.5 with the reasons the numbers shifted —
  "reproduction disregarded" is a decision about the code, not a license to leave stale numbers
  presented as current.

## Test plan

New `tests/test_trainer.py`:
- PC parity vs an in-test hand-rolled reference step (clamps → init → inference → energy → local
  grads → optax) at 1e-12 under the fold_in stream — permanent parity evidence independent of the
  deleted legacy files.
- **RNG-sensitive legs are mandatory for every test that claims to pin the key stream.** Under
  the default `FeedforwardStateInit`, the second pass overwrites the random `z_latent` of every
  unclamped `in_degree>0` node with the feedforward `z_mu`, the only `in_degree==0` node is
  clamped, and inference is deterministic — so `rng_key` has no effect on such graphs and a
  parity/resume test on them passes under *any* key derivation. The parity and resume tests run
  a second leg on a `GlobalStateInit` graph, plus a negative test that `start_epoch=k` produces
  different params than `start_epoch=0` there (the leg that fails if the stream derives from the
  epoch offset instead of the epoch index).
- Backprop gradient correctness: reference CE loss composed from raw jnp ops on a 2-layer
  softmax graph; `jax.grad` of it vs the step's applied update with `optax.sgd(1.0)`; plus a
  `GaussianEnergy`-output variant asserting the objective equals `0.5·precision·SSE/batch`.
- Resume: `train(num_epochs=N)` bitwise equals
  `train(num_epochs=k)` → `train(opt_state=r.opt_state, start_epoch=k, num_epochs=N-k)` on the
  final params (the property `TrainResult` + fold_in exist to provide).
- 4 smoke trains (pc/backprop × classification/sequence via a tiny `create_deep_transformer`):
  finite metrics, params change, `evaluate` returns exactly the per-mode key set.
- Non-float targets: int32 `(batch, seq)` token targets, `(batch,)` class labels, and bool targets
  train in both algorithms (regression test for the one-hot gap; bool covers the
  `_validate_clamp_dtypes` floating-only check).
- `build_clamps` units: v1 graph yields the `(batch,1,seq,seq)` tril clamp; v2 graph yields no mask
  key; eval mode leaves targets free.
- Default-metric conditionality: a `GaussianEnergy`-target graph returns
  `{target_energy, accuracy, energy}` and no `cross_entropy`/`perplexity`; a CE-target graph
  returns all five; a no-target graph raises under the defaults but evaluates a caller-supplied
  metrics dict.
- Metric system: a custom `EvalMetric` via `metrics=` matches a hand-computed value on a loader
  with uneven batch sizes (weighted aggregation, not per-batch mean); `perplexity` equals `exp` of
  the aggregated cross-entropy, not the mean of per-batch `exp`s; padded samples in a ragged
  sharded batch contribute zero weight.
- Contract guards: unknown algorithm; backprop without `FeedforwardStateInit`; backprop with no
  clamped target; retired config keys raise; PC energy decreases over steps; fractional epochs
  (partial-epoch `epoch_results` mean); `EpochContext` field set; **callback exceptions propagate**;
  iter_callback receives floats; metrics keys.
- `generate`: shape/dtype/prompt-prefix test (first AR pytest coverage; supersedes the deleted
  parity scripts).

New `tests/test_sharding.py` (replaces `test_multi_gpu.py`):
`XLA_FLAGS=--xla_force_host_platform_device_count=2` — mesh-vs-single-device parity for one train
step and for `evaluate` (including a ragged final batch exercising the pad-and-weight path).
Because the multi-device legs skip without that env var, replacing the unconditional
`test_multi_gpu.py` with them silently removes CI coverage unless the workflow changes too:
`.github/workflows/test.yml` gains a step running `pytest tests/test_sharding.py` under the flag.

Updates in place: `test_fabricpc.py`, `test_ndim_shapes.py`, `test_optimizers.py`,
`test_storkey_hopfield.py`, `test_conv_pool_integration.py`, `test_mupc.py`,
`test_transformer_nodes.py` (step sites → `make_train_step`; `:443` → `evaluate(...)` with the
adjusted key-set assertion); `test_bayesian_tuner.py` (fakes drive the `EpochContext` callbacks);
`test_experiments.py` stubs return `TrainResult`.

## Docs / CHANGELOG

- `docs/user_guides/08_training_and_evaluation.md`: full rewrite — `train`/`evaluate`,
  `TrainResult` and resume (`opt_state`, `start_epoch`), `make_train_step` custom loops,
  `generate`, the energy framing (backprop loss = clamped-target energy; the output node's energy
  functional selects the loss **and** the default eval metrics), the `EvalMetric` contract with a
  custom-metric example, the default-key table, graph-derived masking,
  the RNG contract, mesh-based multi-device (`jax.make_mesh`), `EpochContext` callbacks,
  `ExperimentArm` partial pattern.
- Mechanical renames: `02_quickstart.md`, `03_how_predictive_coding_works.md`, `07_optimizers.md`,
  `14_api_data.md`, `15_api_experiments.md`, `README.md`.
  All snippets must bind against the new signatures in the same change
  (`tests/test_doc_snippets.py` enforces this).
- `09_experiment_tracking.md` is **not** a mechanical rename: its custom-loop snippet embeds three
  semantic contracts that changed — `train_step_with_history` now returns per-sample energy (so
  the snippet's `energy / batch_size` double-normalizes), it takes a batch dict (the snippet must
  call `convert_batch` on tuple-yielding loaders), and the per-batch key must advance
  (`fold_in(fold_in(rng_key, epoch), batch_idx)`), not be reused verbatim. The signature-binding
  check in `test_doc_snippets.py` catches none of these; they must be rewritten by hand.
- `CHANGELOG.md` `[0.5.0]`: migration table (every removed name → replacement) + behavior notes:
  AR-backprop per-sample objective (÷ legacy lr by seq_len to reproduce), CE eps `clip(1e-7,1)` vs
  `+1e-10`, retired config keys raise, Gaussian-output backprop objective is `0.5·precision·SSE`
  per sample (not element-mean MSE), eval key normalization + conditional CE/perplexity +
  transformer-eval double-softmax fix + internal-only eval energy, `train_step_with_history`
  energy scale, `TrainResult` return type, RNG stream change (0.4 runs are not bitwise
  reproducible under 0.5), pmap → mesh migration (`pmap_single_device` removed), `flax` dependency
  removed, backprop gains tqdm/multi-device, multi-input eval batches now clamp all non-target
  task keys.
  The migration table also carries every **signature** break a migrating reader hits, not only
  removed names: the `iter_callback` third argument (`energy: float` → metrics dict), the
  `epoch_callback` contract (five positionals → one `EpochContext`), `evaluate`'s now-required
  `rng_key` (legacy `evaluate_backprop` defaulted `None` → `PRNGKey(0)`), and the dashboarding
  `create_detailed_iter_callback` shape. Behavior notes additionally cover: multi-device PC
  gradients are the global batch sum (pmap applied a device mean — ×N scale for the same global
  batch), `num_epochs` is required (silent default of 10 removed), `evaluate` on an empty loader
  returns `NaN` per metric (was `0.0`), and the default metrics raise on a no-target graph
  (`evaluate_pcn` returned `{"energy": …, "accuracy": 0.0}`).

## Verification

```bash
.venv/bin/python -m pytest tests/ -x -q
grep -rn "train_pcn\|evaluate_pcn\|train_backprop\|train_autoregressive\|evaluate_autoregressive\|generate_autoregressive\|evaluate_transformer\|multi_gpu\|unified_trainer\|loss_type\|use_causal_mask\|build_train_clamps\|pmap\|autoregressive=" \
    fabricpc examples scripts docs/user_guides tests README.md           # must be empty
JAX_PLATFORMS=cpu .venv/bin/python examples/mnist_demo.py                # PC smoke
JAX_PLATFORMS=cpu .venv/bin/python examples/transformer_v2_demo.py --num_epochs 0.02          # AR-PC
JAX_PLATFORMS=cpu .venv/bin/python examples/transformer_v2_demo.py --num_epochs 0.02 --mode backprop
JAX_PLATFORMS=cpu .venv/bin/python examples/PC_backprop_compare.py       # same-graph gelu arms
XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
    .venv/bin/python -m pytest tests/test_sharding.py -q                 # real 2-device mesh leg
.venv/bin/python -c "from fabricpc import train, evaluate; from fabricpc.training import make_train_step, generate, TrainResult, EpochContext, EvalMetric, metrics"
```

### Results (2026-08-22)

| Leg | Result |
|---|---|
| `pytest tests/ -x -q` | **373 passed, 6 skipped** in 2:24 (2 optuna `ExperimentalWarning`s only) |
| Retired-symbol grep | 12 hits, **no live call sites**: the `trainer.py` fail-fast messages for the retired `loss_type`/`use_causal_mask` config keys, the user-guide text documenting their retirement, the test parametrization asserting the raise, a `test_sharding.py` docstring naming the pmap tests it replaces, and a `transformer_v2_demo.py` comment pointing at `mnist_multi_gpu.py` |
| `mnist_demo.py` (CPU) | Test accuracy **98.13%**; epoch energy monotone at the tail (0.0026 → 0.0023 → 0.0020); ~3.5 s/epoch |
| `transformer_v2_demo.py --num_epochs 0.02` (GPU, per deviation 6) | Accuracy 26.55%, CE 2.7015, **perplexity 14.90**; generation produces text |
| `transformer_v2_demo.py --num_epochs 0.02 --mode backprop` (GPU) | Accuracy 27.69%, CE 2.6297, **perplexity 13.87** |
| `PC_backprop_compare.py --n_trials 1` (per deviation 6) | Same-graph gelu arms: PC 97.75%, backprop 97.36% (+0.39%); per-epoch time ratio 1.02× |
| `test_sharding.py` (2-device CPU mesh) | **6 passed** in 5.5 s |
| Public-API import check | OK |

Full run of transformer_v2_demo
```bash
python examples/transformer_v2_demo.py```
```
PC mode: batch_size=16
Model parameters: 108,353
Vocab Size: 65
Epoch 1/5 — energy: 275.5329, target_energy: 2.1492                                                                                                                                                                                                                                                                 
Epoch 2/5 — energy: 261.7871, target_energy: 2.0417                                                                                                                                                                                                                                                                 
Epoch 3/5 — energy: 252.7063, target_energy: 1.9710                                                                                                                                                                                                                                                                 
Epoch 4/5 — energy: 249.3224, target_energy: 1.9445                                                                                                                                                                                                                                                                 
Epoch 5/5 — energy: 245.0873, target_energy: 1.9116                                                                                                                                                                                                                                                                 
Epoch 5/5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 313660/313660 [3:16:14<00:00, 26.64it/s, energy=240.8698, epoch=5/5]
Training completed in 11775.3s
Evaluation completed in 82.5s
Test Accuracy:   35.06%
Test CE Loss:    2.2542
Test Perplexity: 9.53
--- Generating ---
ROMEO: my, mes one to ind saconing not dundye
Theroug bears en the to ate.
```

## Risks

- PC bitwise continuity vs legacy `train_pcn` holds only up to the RNG-stream switch; the ordered
  steps (spot-check at step 3, switch at step 4, references after) contain it. The permanent
  guarantee is the in-test reference at 1e-12 under the final stream.
- The sharding rewrite of the multi-device leg is new code, not a port; the 2-CPU-device parity
  test is the gate.
- AR-backprop learning rates need retuning (×seq_len objective scale) — CHANGELOG.
- Transformer eval numbers shift (double-softmax and external-SSE fixes) — correct, but flag;
  reproduction of pre-unification transformer results is explicitly out of scope.
- `evaluate` now clamps all non-target task keys (legacy clamped only `x`); no current caller has
  multi-input eval batches — CHANGELOG line.
- `TrainResult` breaks every `p, _, _ = train(...)` unpack loudly (5 fields) — intended; the
  migration list covers all in-repo sites, and the CHANGELOG table covers PyPI users.

## Deviations from the plan worth knowing

1. **`graph_energy` iteration order:** the plan said iterate `structure.node_order`, but the BFS
   topological sort omits cycle members, which would silently drop their energy on cyclic graphs
   (`mnist_cyclic_graph`). The implementation iterates `node_order` first, then the omitted nodes in
   `structure.nodes` insertion order — deterministic and total.
2. **Parity gate metric tolerance:** parameters are bitwise identical, but the reported per-batch
   energy differs at ~1e-7 relative — the legacy sum folded elements with Python `sum` where the new
   path uses `jnp.sum`. Reduction order only; the gradient path never reads the scalar. The gate ran
   on random MNIST-shaped data with the exact demo graph and optimizer; parity is data-independent.
3. **Donation safety:** `train` copies `params`/`opt_state` once at loop start, so the internal
   step's buffer donation cannot invalidate the caller's arrays. `EpochContext` documents that a
   retained `ctx.params` must be copied.
4. **Tuner key collision:** the tuner's training-energy diagnostic is stored as `train_energy`,
   because `evaluate` now legitimately returns an `energy` key it would otherwise have silently
   overwritten.
5. **`mlp_scaling.py`:** its backprop objective now follows the output node's `GaussianEnergy` (the
   positional `loss_type="cross_entropy"` is gone). The script only measures step time and memory, so
   its result semantics are unchanged.
6. **Verification scale:** `PC_backprop_compare` ran with `--n_trials 1` (the full 10-trial study
   takes hours), and the transformer smoke legs ran on GPU — the CPU legs are dominated by the
   stride-1 test loader (~3,400 eval batches), a cost identical before and after this change.


## Plan gaps found in PR review (2026-08-23)

Defects the review traced to the plan itself, not to implementation slips. Each is now folded
into the relevant section above; this log records what the plan originally missed.

1. **RNG-test vacuity.** The test plan specified parity/resume tests but never required a graph
   on which the key matters. Every planned fixture used the default `FeedforwardStateInit`, whose
   second pass overwrites unclamped latents — so the resume "bitwise property", the permanent
   parity references, and the step-3 bitwise RNG gate were all insensitive to the key stream (the
   gate would have passed the split-chain → fold_in switch even if the switch were wrong). Fixed:
   the test plan now mandates `GlobalStateInit` legs and a negative `start_epoch` test.
2. **CI coverage regression.** The plan deleted `test_multi_gpu.py` (ran unconditionally in CI)
   and replaced it with env-gated sharding tests, but the file-changes list never touched
   `.github/workflows/test.yml` — CI silently stopped executing any multi-device test. Fixed:
   the workflow gains a 2-host-device step; the file-changes and test-plan sections name it.
3. **Multi-device gradient scale unflagged.** The device-parallelism design said "XLA inserts the
   cross-device reduction" without stating the semantic consequence: global batch **sum** where
   pmap applied a device **mean**, a ×N learning-rate-visible change absent from the plan's
   CHANGELOG list. Fixed in the design section and the CHANGELOG list.
4. **`generate` had no algorithm axis.** The plan gave `evaluate` the settle-vs-feedforward split
   and `_validate_algorithm` explicitly permits `inference=None` for backprop, but the planned
   `generate` signature always settled via `run_inference` — crashing on backprop-only graphs and
   silently PC-settling backprop-trained models. Fixed: `generate(..., algorithm=)` with the same
   validation.
5. **`num_epochs` missing-key behavior unspecified.** "`config` reads only `num_epochs`" defined
   the read but not the miss; the implementation inherited a silent default of 10 while
   `ab_experiment` assumed 1 — contradicting the plan's own fail-fast stance on retired keys.
   Fixed: required, raises.
6. **The plan's own pseudocode embedded two fragile derivations** it elsewhere claimed were
   graph-derived: `batch_size = next(iter(batch.values())).shape[0]` (wrong under an extra
   non-batch-leading key; hard sharding failure under mesh) and the causal-mask `seq` from
   `batch["x"]` (a bare `KeyError` when the input task key is not named `x`). Fixed: batch size
   from the first task-mapped key, `seq` from the mask node's declared shape.
7. **CHANGELOG list omitted the signature breaks** a migrating reader actually hits — the
   `iter_callback`/`epoch_callback` contracts, `evaluate`'s required `rng_key`, empty-loader
   `NaN`, and the no-target default-metrics raise. Fixed in the CHANGELOG section.
8. **Doc/example migration misclassified.** Guide 09's custom-loop snippet was listed under
   "mechanical renames" although it embeds three changed semantic contracts (per-sample energy
   scale, batch-dict conversion, per-batch key threading) that the doc-snippet signature check
   cannot catch; and "reproduction of pre-unification transformer results is disregarded" left
   the demos' recorded Results blocks presented as current although the README requires demos to
   match baselines. Fixed: guide 09 called out for a hand rewrite; baseline blocks re-measured or
   marked pre-0.5.
9. **`create_detailed_iter_callback` migration underspecified.** "Remains custom-loop-only on
   `make_train_step`'s `final_state`" named the consumer but not the contract, so the factory
   kept the pre-0.5 `(epoch, batch, energy: float, state)` shape with zero valid call sites.
   Fixed: the step's metrics-dict contract, documented with a custom-loop example.

## Followups TODO
- Support non-deterministic nodes. RNG seed splitting into the inference path for future dropout and other stochastic processing in nodes.
