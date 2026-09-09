# `IterContext` for `train(iter_callback=...)`

Self-contained PR against `main`. The ePC branch (`matthew_cedric/epc`) rebases onto it; nothing in this PR depends on that branch. Revised 2026-09-08 after review (`iter_callback_context_review.md`).

## Context

`train()` calls `iter_callback(epoch_idx, batch_idx, metrics)` after every batch (`fabricpc/training/trainer.py:623` on `main`). The callback cannot see the parameters, the batch, or the settled state, so any per-update diagnostic that needs them re-implements the training loop over `make_train_step`. Three consumers do this today:

- The ePC stability probe (`scripts/epc_analysis.py --track_lambda_max N` on `matthew_cedric/epc`): every N optimizer updates it runs power iteration on the Hessian-vector product of `EPCInference.error_energy` to measure λ_max(H_ε), the top eigenvalue of the error-coordinate Hessian, and logs η·λ_max beside the training energy, where η is the inference step size `eta_infer`. ePC inference is stable only while η·λ_max < 2, and λ_max grows with the downstream weight products as training proceeds, so the probe samples between epoch boundaries. It needs the current parameters after each update; the epoch callback offers them once per epoch.
- `create_detailed_iter_callback` (`fabricpc/utils/dashboarding/callbacks.py:133` on `main`) tracks per-node energy and state statistics from the settled `GraphState`, which `train` discards (`_make_step(with_state=False)`). It plugs into a custom loop, and guide 09 carries a second training loop to host it. Through the recommended path, `create_tracking_callbacks`, the `TrackingConfig` fields `track_state_distributions`, `state_tracking_every_n_infer_steps`, and the state half of `nodes_to_track` have no effect: the iteration callback it builds never sees a state.
- `examples/transformer_demo.py:475-597` on `main`: a custom loop plus a `TrainingProgressBar` class that duplicate `train`'s epoch schedule and tqdm bar, because the demo tracks weight distributions per batch and re-runs inference with history on tracked batches, both of which need the parameters and the batch after each update. The re-run is an unjitted Python loop over `inference_step` (`run_inference_with_full_history`).

`epoch_callback` already receives an `EpochContext` with `params`, `opt_state`, `structure`, `config`, `rng_key`, and `metrics`. This PR gives `iter_callback` one `IterContext` argument carrying every `EpochContext` field plus the per-batch ones (`batch_idx`, `batch`, `batch_key`, `state`), hands both contexts every RNG key the trainer derived at or above their level and the `algorithm` in effect, and then removes the three workarounds above that live in this repository: one dashboarding iteration factory whose behavior the `TrackingConfig` decides, a jitted history collector in place of the Python loop, and the transformer demo on `train`.

## Design

### Trainer contexts

Both contexts are `NamedTuple`s in `fabricpc/training/trainer.py`, exported from `fabricpc.training`. They grow by appending fields, so the positions of existing fields never move; callbacks read fields by name.

`EpochContext` (two fields appended after `metrics`):

| Field | Meaning |
|---|---|
| `epoch_idx` | epoch index including `start_epoch` |
| `step` | optimizer updates applied in this call so far (not offset by `start_epoch`, as `TrainResult.step`) |
| `params`, `opt_state` | parameters and optimizer state after the epoch's last update |
| `structure`, `config` | the graph and the training config passed to `train` |
| `rng_key` | the base training key passed to `train` |
| `metrics` | epoch means of the per-batch float metrics |
| `algorithm` | **new**: the `algorithm` passed to `train`, `"pc"` or `"backprop"` |
| `epoch_key` | **new**: `fold_in(rng_key, epoch_idx)`, the key this epoch's batch keys derive from |

`IterContext` (new): the `EpochContext` fields first, in that order, then:

| Field | Meaning |
|---|---|
| `batch_idx` | batch index within the epoch (loader position; a batch skipped for mesh divisibility still advances it) |
| `state` | the `GraphState` the step produced for this batch: settled latents under PC, the feedforward pass under backprop |
| `batch_key` | `fold_in(epoch_key, batch_idx)`, the key this step used for latent initialization |
| `batch` | the converted batch dict fed to this step (task-mapped keys). Under `mesh`, its arrays carry the `P("data")` sharding |

In `IterContext`, `step` counts this batch's update and `metrics` holds the per-batch float metrics (`energy`, `target_energy`); `params` and `opt_state` are the values after this batch's update.

`iter_callback(ctx: IterContext) -> Any`. A non-None return still replaces that batch's `iter_results` entry; supplying the callback still forces the per-batch device sync. Exceptions propagate.

Buffer lifetimes. The internal step donates the `params` and `opt_state` buffers, so `ctx.params` and `ctx.opt_state` are valid during the callback and must be copied (`tree_map(jnp.copy, ...)`) if retained; this is the existing `EpochContext` caveat. `ctx.state` and `ctx.batch` are not donated. The trainer drops the context and its own `state` name as soon as the callback returns, so a callback that does not retain `ctx.state` adds no device memory beyond its own duration, and a callback that retains it keeps exactly that one `GraphState` alive.

State plumbing. `train` builds the step with `with_state=(iter_callback is not None)`. Without a callback the step returns `(params, opt_state, metrics)` as now, so the no-callback path is unchanged in memory and dispatch. `_make_step`'s docstring names both consumers.

Type annotations on `train` tighten to `Optional[Callable[[EpochContext], Any]]` and `Optional[Callable[[IterContext], Any]]`.

### Jitted history collector

`fabricpc/utils/dashboarding/inference_tracking.py` replaces `run_inference_with_full_history` with `make_inference_history(structure, *, every) -> history(params, initial_state, clamps) -> (final_state, states)`. `history` is one jitted program: a `lax.scan` over `infer_steps // every` blocks, each block a `fori_loop` of `every` inference steps that emits the state at its end, followed by a `fori_loop` over the remaining `infer_steps % every` steps so `final_state` equals `run_inference`'s settle. `states` is a `GraphState` pytree whose leaves carry a leading axis of length `infer_steps // every + 1`: index `i` is the state after `i * every` steps, index 0 is `initial_state`, and when `infer_steps` is a multiple of `every` the last index is `final_state`. Peak device memory is the sampled states, not one per step. The factory compiles on first use; build it once per structure.

Measured on a 784-256-64-64-10 MLP, batch 200, 20 inference steps, `every=5`, CPU: 14.8 ms against 10.1 ms for the plain jitted settle (1.46x); the unjitted loop it replaces took 362 ms (35x).

`callbacks.py` adds `make_tracked_settle(structure, *, every) -> settle(params, key, batch) -> states`: `build_clamps`, `initialize_graph_state`, and `make_inference_history` in one jitted program, so a tracked batch costs one compiled settle. Both factories are exported from `fabricpc.utils.dashboarding`.

### Dashboarding: one iteration factory, the config decides

`fabricpc/utils/dashboarding/callbacks.py` keeps three factories. `create_detailed_iter_callback` is removed; `IterContext` carries `structure`, `state`, `params`, `batch`, and `batch_key`, so nothing distinguishes it from `create_iter_callback` except which tracker methods log, and that is what `TrackingConfig` exists to decide.

`TrackingConfig` gains two fields:

- `distribution_nodes: List[str] = []`: the nodes whose weight/bias and state distributions are logged. Empty logs no distributions, mirroring `nodes_to_track`, whose meaning is unchanged from `main`: per-node energy and inference-dynamics breakdowns, empty logs none. `AimExperimentTracker.track_weight_distributions` and `track_state` default their `nodes` argument to `config.distribution_nodes` and return when it is empty, so direct callers get the same gate as the callback.
- `track_state: bool = False`: log state summary statistics (mean, std, L2 norm of `z_latent`, `z_mu`, `energy`) on tracked batches. `track_state_distributions` keeps its meaning (also log histograms) and implies `track_state`. `TrackingConfig.tracks_state` is `bool(distribution_nodes) and (track_state or track_state_distributions)`; `track_state` returns early unless a state flag is set and there are nodes to log.

`create_iter_callback(tracker) -> Callable[[IterContext], Dict[str, float]]`, per batch, in this order:

1. `tracker.track_batch_energy(ctx.metrics["energy"], epoch=ctx.epoch_idx, batch=ctx.batch_idx)` (gated inside by `track_energy`).
2. `tracker.track_batch_energy_per_node(ctx.state, ctx.structure, ...)` (gated inside by `nodes_to_track`). This is the training settle under the pre-update parameters.
3. `tracker.track_weight_distributions(ctx.params, ctx.structure, epoch=ctx.epoch_idx, batch=ctx.batch_idx)` (gated inside by `track_weight_distributions`, `batch % tracking_every_n_batches`, and `distribution_nodes`). This is the cadence the `TrackingConfig` docstring already promised for weight distributions; `create_epoch_callback` stops making its once-per-epoch `batch=0` call, so there is one owner and no double record at batch 0.
4. State tracking when `tracker.config.tracks_state` and `ctx.batch_idx % tracking_every_n_batches == 0`. Under `ctx.algorithm == "pc"`: build `make_tracked_settle(ctx.structure, every=state_tracking_every_n_infer_steps)` on the first tracked batch (cached in the closure, keyed by structure identity), call it with `(ctx.params, ctx.batch_key, ctx.batch)`, and call `tracker.track_state(tree_map(lambda a: a[i], states), epoch, batch, infer_step=i * every)` for each sampled index. The re-run is a fresh settle of the same batch under the post-update parameters, initialized from the step's batch key; under a parameter-dependent initializer (`FeedforwardStateInit`, the transformer demo's) the initial latents also come from the updated parameters. Under `"backprop"` there is no settling to record: one `track_state(ctx.state, ..., infer_step=0)` call with the feedforward state.
5. Return `ctx.metrics`.

Cost of item 4: one jitted settle per tracked batch, only when `tracks_state`.

`create_epoch_callback(tracker, structure, eval_fn, eval_loader, eval_config)`: evaluation and `track_epoch_metrics` only; the weight-distribution call moves to the iteration callback (item 3). `create_tracking_callbacks` keeps its signature and returns `create_iter_callback(tracker)`; `structure` remains for `log_graph_structure` and the epoch callback.

### `examples/transformer_demo.py` on `train`

Delete `TrainingProgressBar`, the local `create_iter_callback(use_pc_mode)` factory, the five-positional `eval_callback` (the pre-0.5.0 epoch-callback shape called by hand), and the loop. Replace with one `train` call: `verbose=True` (tqdm bar and epoch summary), `iter_callback=create_iter_callback(tracker) if tracker else None`, and an `epoch_callback(ctx)` that runs `evaluate(ctx.params, ctx.structure, test_batches, {}, ctx.epoch_key, algorithm=ctx.algorithm)`, prints via `tqdm.write`, and returns the metrics. The demo's `TrackingConfig` sets `track_state_distributions=True`, `nodes_to_track=TRACKED_NODES`, `distribution_nodes=TRACKED_NODES`, `tracking_every_n_batches=50`, `state_tracking_every_n_infer_steps=5`, so the factory reproduces its energy, weight, per-node, and inference-history tracking with no demo-side tracking code. `energy_history` becomes `result.iter_results`, `eval_results` becomes `result.epoch_results`. Two visible changes, both acceptable for an example and stated in its comments: per-batch keys follow the trainer's `fold_in` stream instead of `jax.random.split`, and the backprop bar shows the energy postfix instead of a perplexity postfix (the epoch summary still prints test perplexity).

### `BayesianTuner`

Passes its progress-printing `iter_callback` only when `verbose=True`. Supplying one forces a per-batch host sync and the step's `GraphState` return; only the print needs either.

## Migrations (same PR, no compatibility shim)

Every `train` caller that passes an `iter_callback`, every constructor of `EpochContext`, every user of the removed factory and the removed history function, every direct caller of the two tracker methods, and every document stating any of them:

- `fabricpc/training/trainer.py`: `IterContext` class; `algorithm` and `epoch_key` appended to `EpochContext`; `with_state` gating; build both contexts at their call sites; `del state` after the iter callback; `train` docstring lines for both callbacks and the `_make_step` docstring.
- `fabricpc/training/__init__.py`: export `IterContext`.
- `fabricpc/tuning/bayesian_tuner.py`: `def iter_callback(ctx)`, reads `ctx.batch_idx`, `ctx.epoch_idx`, `ctx.metrics["energy"]`; passed only under `verbose`. Failure mode if the signature were missed: `train` raises `TypeError`, the tuner's `except Exception` converts it to `optuna.TrialPruned`, and every trial is pruned with "failed during training" while the study reports success.
- `fabricpc/utils/dashboarding/inference_tracking.py`: `make_inference_history` replaces `run_inference_with_full_history`.
- `fabricpc/utils/dashboarding/callbacks.py`: `make_tracked_settle`; the unified `create_iter_callback`; `create_epoch_callback` without the weight call; `create_detailed_iter_callback` deleted; module docstring without the "exception" sentence.
- `fabricpc/utils/dashboarding/__init__.py`: drop `create_detailed_iter_callback` and `run_inference_with_full_history`; export `make_inference_history` and `make_tracked_settle`.
- `fabricpc/utils/dashboarding/trackers.py`: `TrackingConfig.distribution_nodes`, `TrackingConfig.track_state`, `tracks_state`; the gates in `track_weight_distributions` and `track_state`; docstrings, including `tracking_every_n_epochs` ("Reserved; read by nothing") and the class example.
- `examples/transformer_demo.py`: as above.
- `examples/transformer_v2_demo.py`: `def iter_callback(ctx)`, `ctx.batch_idx`, `ctx.epoch_idx`, `ctx.metrics["energy"]`.
- `examples/mnist_aim_tracking.py`: its custom loop over `train_step_with_history` stays (it records the training settle's own history inside the jitted step, which `IterContext` cannot offer without a second pass); its `TrackingConfig` gains `distribution_nodes` so its direct `track_state` and `track_weight_distributions` calls keep logging.
- `examples/resnet18_cifar10_demo.py`: annotate `epoch_callback(ctx: EpochContext)`.
- `tests/test_fabricpc.py`: `lambda ctx: iters_half.append(1) or ctx.metrics`.
- `tests/test_trainer.py`, `tests/test_sharding.py`, `tests/test_bayesian_tuner.py`, `tests/test_dashboarding_callbacks.py`: below.
- `docs/user_guides/08_training_and_evaluation.md`: the iteration-callback fence takes `ctx: IterContext`; the paragraph lists the fields in order and says new fields are appended; the epoch-callback field list adds `algorithm` and `epoch_key`; the contract list gains "`ctx.state` and `ctx.batch` are not donated; the trainer drops its reference to the state after the callback".
- `docs/user_guides/09_experiment_tracking.md`: the dataclass fence and the options table gain `track_state` and `distribution_nodes`; `nodes_to_track` keeps its per-node-breakdown text; `track_state_distributions` says it implies `track_state`; `tracking_every_n_epochs` reads "Reserved; read by nothing"; `state_tracking_every_n_infer_steps` states the `0, k, 2k, ...` sampling. "Weight Distributions" and "State Distributions" set `distribution_nodes` in their fences; "State Distributions" states the jitted re-settle, its sampling, that it is a fresh settle, and its cost. The section "Per-batch state tracking with `make_train_step`" is deleted. "Advanced Usage: Custom Training Loop" stays: `train_step_with_history` collects the history inside the jitted step from the training settle itself, on every batch, without a second pass. `test_doc_snippets.py` checks every fence's imports.
- `CHANGELOG.md`: `## [0.5.2] - 2026-09-08` with the migration table (callback signature, removed factory, weight-distribution owner and `distribution_nodes` default, tracker-method `nodes` default, `run_inference_with_full_history` → `make_inference_history`, `EpochContext` fields appended) and the New list. The package is on PyPI, so the rows are the migration path for external callbacks; the 0.5.0 precedent is a clean break with a table row, followed here.
- `pyproject.toml`: version `0.5.2`.

## Tests

- `tests/test_trainer.py`: `test_iter_context_fields_and_callback_replacement` (both algorithms, `start_epoch=5`, two epochs): `isinstance(ctx, IterContext)`; float metrics; `ctx.step` increments by one per call; `ctx.params` is a `GraphParams`; `ctx.state` is a `GraphState` whose node keys equal `structure.nodes`; `ctx.batch` holds the task-mapped keys; `ctx.algorithm == algorithm`; `ctx.epoch_idx` honours `start_epoch`; `ctx.epoch_key == fold_in(train_key, ctx.epoch_idx)` and `ctx.batch_key == fold_in(ctx.epoch_key, ctx.batch_idx)`; on the first batch, `ctx.params` and `ctx.state` match `make_train_step` replayed on the caller's initial params with `ctx.batch` and `ctx.batch_key` (within `PARITY_TOL`). `test_epoch_context_fields_and_callback_replacement` adds the `algorithm` and `epoch_key` checks. `test_callback_exceptions_propagate` takes `ctx`.
- `tests/test_sharding.py`: under the two-device mesh, the iter callback sees `ctx.batch["x"].sharding` equal to `NamedSharding(mesh, P("data"))` and `ctx.state` is a `GraphState`.
- `tests/test_dashboarding_callbacks.py` (new). A duck-typed stub tracker holding a real `TrackingConfig` records calls to the four batch-level methods; a two-layer PC graph, two batches, real `train`:
  - defaults: energy and weight calls handed to the tracker every batch, no state calls;
  - `track_state=True` with empty `distribution_nodes`: no state calls (no re-settle);
  - `track_state=True, distribution_nodes=["h"], tracking_every_n_batches=1, state_tracking_every_n_infer_steps=1`, `infer_steps=3`, PC: state calls at `infer_step` 0, 1, 2, 3 per batch;
  - `state_tracking_every_n_infer_steps=2`, `infer_steps=4`: `infer_step` 0, 2, 4;
  - the logged states equal `make_inference_history(every=2)` recomputed from the captured `IterContext`; index 0 equals the initialization and the last equals `run_inference`'s settle;
  - `track_state_distributions` implies `tracks_state`;
  - backprop: one state call per tracked batch at `infer_step=0`;
  - `tracking_every_n_batches=2`: state calls at batches 0 and 2 only;
  - `create_epoch_callback` makes no `track_weight_distributions` call;
  - `make_inference_history(every=0)` raises `ValueError`.
  With a fake Aim run and `get_aim` monkeypatched: `track_weight_distributions` and `track_state` log only `distribution_nodes` and log nothing when it is empty; `track_state` logs summary statistics only unless `track_state_distributions`; `track_state` does not touch `_ensure_initialized` when no state flag is set.
- `tests/test_bayesian_tuner.py`: the fake drives both callbacks with keyword-constructed contexts; one test runs `verbose=True` so the print path executes; one asserts the fake receives `iter_callback=None` under `verbose=False`.

## Alternatives considered

- **`IterContext` as a superset of `EpochContext` (chosen).** One rule for both callbacks: each context carries the base key and every key derived at or above its level, the algorithm, and the objects the trainer holds at that point. Future fields append without breakage.
- **Insert the new `EpochContext` fields beside `rng_key`.** The earlier draft. Moves `rng_key` and `metrics` to new positions, which is the positional breakage the docstring rules out. Rejected for appending.
- **Append `params` as a fourth positional argument.** Breaks every caller now and again on each later addition.
- **`rng_key` only, with the `fold_in` chain documented for callers to recompute.** Duplicates the trainer's key derivation in every consumer; a change to the derivation would silently desynchronize them. Rejected for `epoch_key` and `batch_key` fields.
- **Infer the algorithm from `structure.config["inference"]` instead of an `algorithm` field.** A backprop run on a graph that also carries an inference object would re-settle for nothing on every tracked batch. Rejected; `algorithm` is a `train` argument like `config` and rides along the same way.
- **`state` on `EpochContext` too.** The only state available at an epoch boundary is the last batch's, a per-batch quantity sampled at an arbitrary loader position, with no consumer. Delivering it means either holding the previous batch's `GraphState` on device through every step or an `Optional` field that is `None` when the epoch's last batch was skipped for mesh divisibility. Rejected; a callback that wants a state at epoch end samples it from `IterContext` at the batch it chooses.
- **Always return the state from the internal step.** Every `train` call without a callback would keep one `GraphState` alive across the next step. Rejected for `with_state=(iter_callback is not None)` plus `del state` after the callback.
- **Device scalars in `IterContext.metrics`, callback decides when to sync.** Would let a probe that runs every N updates skip the host sync on the other batches. Rejected: `verbose=True` (the default) syncs every batch for the tqdm postfix anyway, the tracking callback wants floats every batch, and the stored `iter_results` entries would need a second materialization path for callback-returned device values.
- **Keep two dashboarding iteration factories, both over `IterContext`.** Preserves a split that `IterContext` makes meaningless and leaves `track_state_distributions`, `state_tracking_every_n_infer_steps`, and state tracking dead through `create_tracking_callbacks`. Rejected for one factory gated by the config, with `track_state` added so the gate has an explicit switch.
- **Factory records `ctx.state` only; inference history stays in custom loops.** Cheaper (no second pass) but leaves `state_tracking_every_n_infer_steps` effective only through `train_step_with_history` loops and keeps the transformer demo's custom callback. Rejected (user decision 2026-09-08) for the re-settle on tracked batches under PC.
- **Re-settle with the unjitted `run_inference_with_full_history`.** The first implementation. 35x a jitted settle on the benchmark above, all `infer_steps` states resident at once, and `history[0]` mislabeled as step 0 (it was the state after one update, and the settled state was never logged when `infer_steps` is a multiple of the sampling interval). Rejected for `make_inference_history`.
- **`nodes_to_track` scopes weight and state distributions too.** The first implementation. Overloads one field with "which nodes get a per-node breakdown" and "which nodes get distributions at all", and silently drops other nodes' weight histograms for an existing `create_tracking_callbacks` user who set it for per-node energy. Rejected for `distribution_nodes`.
- **`distribution_nodes` empty means every node.** Keeps the pre-PR default output (all weight histograms) but gives the two list fields opposite empty semantics. Rejected (user decision 2026-09-08) for empty means none, mirroring `nodes_to_track`; the CHANGELOG states the consequence for the default config.
- **Leave `examples/transformer_demo.py` on its custom loop.** The loop and `TrainingProgressBar` exist only because tracking needed the parameters, batch, and state per batch, which `IterContext` now supplies. Rejected.
- **A separate `probe_every=N, probe=callable` trainer parameter.** A second per-batch mechanism beside `iter_callback` with its own sync and return semantics.
- **Status quo: probes use `make_train_step` in a custom loop.** Every probe re-implements the loop and the callback plumbing; the analysis script, guide 09, and the transformer demo each carry one.

## Out of scope, with reasons

- `TrackingConfig.tracking_every_n_epochs` is read by nothing before or after this PR. Its docstring now says so. Removing it is a `TrackingConfig` API change unrelated to callback visibility; flagged here so it is decided on its own.
- On `matthew_cedric/epc`, the `--track_lambda_max` loop in `scripts/epc_analysis.py` becomes an `iter_callback` that runs the power iteration when `ctx.step % N == 0` on `ctx.params`. That file lives on the other branch and migrates when it rebases.

## Verification

1. `JAX_PLATFORMS=cpu .venv/bin/python -m pytest tests/test_trainer.py tests/test_fabricpc.py tests/test_bayesian_tuner.py tests/test_dashboarding_callbacks.py tests/test_dashboarding_extractors.py tests/test_doc_snippets.py -q` green; `XLA_FLAGS=--xla_force_host_platform_device_count=2 .venv/bin/python -m pytest tests/test_sharding.py -q` green.
2. `python examples/transformer_v2_demo.py --verbose` (one epoch or fewer batches) prints per-batch energy through the migrated callback.
3. `python examples/transformer_demo.py --num_epochs 0.05` for both `--mode` values runs on `train` end to end; with Aim installed, the run contains `energy`, `node_energy`, weight distributions for `embed` and `transformer_0`, and `z_latent_mean` at `infer_step` 0, 5, 10, 15, 20 under PC.
4. `grep -rn "iter_callback\|EpochContext(\|create_detailed_iter_callback\|run_inference_with_full_history" --include=*.py --include=*.md fabricpc tests examples docs/user_guides CHANGELOG.md`: no three-argument `iter_callback` definition remains; the removed names appear only in the CHANGELOG migration table. `docs/dev_plans_archive/` is history and is not edited.
5. `python -c "from fabricpc.training import IterContext, EpochContext; print(IterContext._fields, EpochContext._fields)"` lists the fields in the tables above, `EpochContext`'s as a prefix of `IterContext`'s.
6. `ruff check` and `black --check` clean.

## Sequencing

One PR on a branch off `main`; PR message via a temporary file in the project root. At merge, move this plan and `iter_callback_context_review.md` to `docs/dev_plans_archive/`, the home of every earlier plan. After merge, `matthew_cedric/epc` rebases onto it and its probe consumes `IterContext` directly.
