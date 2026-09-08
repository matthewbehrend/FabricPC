# `IterContext` for `train(iter_callback=...)`

Self-contained PR against `main`. The ePC branch (`matthew_cedric/epc`) rebases onto it; nothing in this PR depends on that branch.

## Context

`train()` calls `iter_callback(epoch_idx, batch_idx, metrics)` after every batch (`fabricpc/training/trainer.py:623`). The callback cannot see the parameters, the batch, or the settled state, so any per-update diagnostic that needs them re-implements the training loop over `make_train_step`. Three consumers do this today:

- The ePC stability probe (`scripts/epc_analysis.py --track_lambda_max N` on `matthew_cedric/epc`): every N optimizer updates it runs power iteration on the Hessian-vector product of `EPCInference.error_energy` to measure λ_max(H_ε), the top eigenvalue of the error-coordinate Hessian, and logs η·λ_max beside the training energy, where η is the inference step size `eta_infer`. ePC inference is stable only while η·λ_max < 2, and λ_max grows with the downstream weight products as training proceeds, so the probe samples between epoch boundaries. It needs the current parameters after each update; the epoch callback offers them once per epoch.
- `create_detailed_iter_callback` (`fabricpc/utils/dashboarding/callbacks.py:133`) tracks per-node energy and state statistics from the settled `GraphState`, which `train` discards (`_make_step(with_state=False)`). It plugs into a custom loop, and guide 09 carries a second training loop to host it. Through the recommended path, `create_tracking_callbacks`, the `TrackingConfig` fields `track_state_distributions`, `state_tracking_every_n_infer_steps`, and the state half of `nodes_to_track` have no effect: the iteration callback it builds never sees a state.
- `examples/transformer_demo.py:475-597`: a custom loop plus a `TrainingProgressBar` class (`:299-340`) that duplicate `train`'s epoch schedule and tqdm bar, because the demo tracks weight distributions per batch and re-runs inference with history on tracked batches, both of which need the parameters and the batch after each update.

`epoch_callback` already receives an `EpochContext` with `params`, `opt_state`, `structure`, `config`, `rng_key`, and `metrics` (`trainer.py:105`). This PR gives `iter_callback` one `IterContext` argument carrying every `EpochContext` field plus the per-batch ones (`batch_idx`, `batch`, `batch_key`, `state`), hands both contexts every RNG key the trainer derived at or above their level and the `algorithm` in effect, and then removes the three workarounds above that live in this repository: one dashboarding iteration factory whose behavior the `TrackingConfig` decides, and the transformer demo on `train`.

## Design

### Trainer contexts

Both contexts are `NamedTuple`s in `fabricpc/training/trainer.py`, exported from `fabricpc.training`. They grow by field addition only; callbacks read fields by name.

`EpochContext` (two fields added):

| Field | Meaning |
|---|---|
| `epoch_idx` | epoch index including `start_epoch` |
| `step` | optimizer updates applied in this call so far (not offset by `start_epoch`, as `TrainResult.step`) |
| `params`, `opt_state` | parameters and optimizer state after the epoch's last update |
| `structure`, `config` | the graph and the training config passed to `train` |
| `algorithm` | **new**: the `algorithm` passed to `train`, `"pc"` or `"backprop"` |
| `rng_key` | the base training key passed to `train` |
| `epoch_key` | **new**: `fold_in(rng_key, epoch_idx)`, the key this epoch's batch keys derive from |
| `metrics` | epoch means of the per-batch float metrics |

`IterContext` (new):

| Field | Meaning |
|---|---|
| `epoch_idx` | epoch index including `start_epoch` |
| `batch_idx` | batch index within the epoch (loader position; a batch skipped for mesh divisibility still advances it) |
| `step` | optimizer updates applied in this call, counting this batch |
| `params`, `opt_state` | parameters and optimizer state after this batch's update |
| `state` | the `GraphState` the step produced for this batch: settled latents under PC, the feedforward pass under backprop |
| `structure`, `config`, `algorithm` | as in `EpochContext` |
| `rng_key`, `epoch_key` | as in `EpochContext` |
| `batch_key` | `fold_in(epoch_key, batch_idx)`, the key this step used for latent initialization |
| `batch` | the converted batch dict fed to this step (task-mapped keys). Under `mesh`, its arrays carry the `P("data")` sharding |
| `metrics` | the per-batch float metrics (`energy`, `target_energy`) |

`iter_callback(ctx: IterContext) -> Any`. A non-None return still replaces that batch's `iter_results` entry; supplying the callback still forces the per-batch device sync. Exceptions propagate.

Buffer lifetimes. The internal step donates the `params` and `opt_state` buffers, so `ctx.params` and `ctx.opt_state` are valid during the callback and must be copied (`tree_map(jnp.copy, ...)`) if retained; this is the existing `EpochContext` caveat. `ctx.state` and `ctx.batch` are not donated. The trainer drops its own reference to the state (`del state`) as soon as the callback returns, so a callback that does not retain `ctx.state` adds no device memory beyond its own duration, and a callback that retains it keeps exactly that one `GraphState` alive.

State plumbing. `train` builds the step with `with_state=(iter_callback is not None)`. Without a callback the step returns `(params, opt_state, metrics)` as now, so the no-callback path is unchanged in memory and dispatch. `_make_step`'s docstring changes from "public escape hatch" to naming both consumers.

Type annotations on `train` tighten to `Optional[Callable[[EpochContext], Any]]` and `Optional[Callable[[IterContext], Any]]`.

### Dashboarding: one iteration factory, the config decides

`fabricpc/utils/dashboarding/callbacks.py` keeps three factories. `create_detailed_iter_callback` is removed; `IterContext` carries `structure`, `state`, `params`, `batch`, and `batch_key`, so nothing distinguishes it from `create_iter_callback` except which tracker methods it calls, and that is what `TrackingConfig` exists to decide.

`create_iter_callback(tracker) -> Callable[[IterContext], Dict[str, float]]`, per batch, in this order:

1. `tracker.track_batch_energy(ctx.metrics["energy"], epoch=ctx.epoch_idx, batch=ctx.batch_idx)` (gated inside by `track_energy`).
2. `tracker.track_batch_energy_per_node(ctx.state, ctx.structure, ...)` (gated inside by `nodes_to_track`).
3. `tracker.track_weight_distributions(ctx.params, ctx.structure, epoch=ctx.epoch_idx, batch=ctx.batch_idx, nodes=nodes_to_track or None)` (gated inside by `track_weight_distributions` and `batch % tracking_every_n_batches`; a non-empty `nodes_to_track` scopes weights and state to those nodes, empty means every node). This is the cadence the `TrackingConfig` docstring already promises for weight distributions; `create_epoch_callback` stops making its once-per-epoch `batch=0` call, so there is one owner and no double record at batch 0.
4. State tracking when `ctx.batch_idx % tracker.config.tracking_every_n_batches == 0` and the config asks for it (`track_state or track_state_distributions`, below). Under `ctx.algorithm == "pc"`: rebuild the clamps with `build_clamps(ctx.batch, ctx.structure, clamp_target=True)`, initialize with `initialize_graph_state(ctx.structure, batch_size_of(ctx.batch, ctx.structure), ctx.batch_key, clamps=clamps, params=ctx.params)`, run `run_inference_with_full_history(ctx.params, init, clamps, ctx.structure)`, and call `tracker.track_state(step_state, epoch, batch, infer_step=i, nodes=tracker.config.nodes_to_track or None)` for each step (the method's own `state_tracking_every_n_infer_steps` gate subsamples). The re-run settles the same batch from the same latent initialization under the post-update parameters, as the transformer demo does today; `ctx.state` is the settle under the pre-update parameters and is what per-node energy (item 2) reports. Under `"backprop"` there is no settling to record: one `track_state(ctx.state, ..., infer_step=0)` call with the feedforward state.
5. Return `ctx.metrics`.

Cost of item 4: one extra inference pass on one batch in every `tracking_every_n_batches` (2% at the default of 50), only when state tracking is enabled.

`TrackingConfig` gains `track_state: bool = False`: log state summary statistics (mean, std, L2 norm of `z_latent`, `z_mu`, `energy`) on tracked batches. `track_state_distributions` keeps its meaning (also log histograms) and implies `track_state`. `AimExperimentTracker.track_state` returns early unless `track_state or track_state_distributions`, so a caller of the method directly gets the same gate. Both flags default `False`, so an existing `create_tracking_callbacks` user sees no new state transfers unless they opt in; they do see weight distributions every `tracking_every_n_batches` instead of once per epoch.

`create_epoch_callback(tracker, structure, eval_fn, eval_loader, eval_config)`: evaluation and `track_epoch_metrics` only; the weight-distribution call moves to the iteration callback (item 3). `create_tracking_callbacks` keeps its signature and returns `create_iter_callback(tracker)`; `structure` remains for `log_graph_structure` and the epoch callback.

### `examples/transformer_demo.py` on `train`

Delete `TrainingProgressBar` (`:299-340`), the local `create_iter_callback(use_pc_mode)` factory (`:475-495`), the five-positional `eval_callback` (`:457-467`, the pre-0.5.0 epoch-callback shape called by hand), and the loop (`:503-597`). Replace with one `train` call: `verbose=True` (tqdm bar and epoch summary), `iter_callback=create_iter_callback(tracker) if tracker else None`, and an `epoch_callback(ctx)` that runs `evaluate(ctx.params, ctx.structure, test_batches, {}, ctx.epoch_key, algorithm=ctx.algorithm)`, prints via `tqdm.write`, and returns the metrics. The demo's `TrackingConfig` already sets `track_state_distributions=True`, `nodes_to_track=TRACKED_NODES`, `tracking_every_n_batches=50`, `state_tracking_every_n_infer_steps=5`, so the factory reproduces its energy, weight, per-node, and inference-history tracking with no demo-side tracking code. `energy_history` becomes `result.iter_results`, `eval_results` becomes `result.epoch_results`; the closing prints read `result.iter_results[-1][-1]["energy"]` and `result.epoch_results[-1]`. Remove the now-unused imports (`build_clamps`, `make_train_step`, `initialize_graph_state`, `run_inference_with_full_history`, `math`, `Optional`/`Any`/`Dict` if unused). Two visible changes, both acceptable for an example and stated in its docstring: per-batch keys follow the trainer's `fold_in` stream instead of `jax.random.split`, and the backprop bar shows the energy postfix instead of a perplexity postfix (the epoch summary still prints test perplexity).

## Migrations (same PR, no compatibility shim)

Every `train` caller that passes an `iter_callback`, every constructor of `EpochContext`, every user of the removed factory, and every document stating either signature:

- `fabricpc/training/trainer.py`: `IterContext` class; `algorithm` and `epoch_key` on `EpochContext`; `with_state` gating; build both contexts at their call sites; `del state` after the iter callback; `train` docstring lines for both callbacks and the `_make_step` docstring.
- `fabricpc/training/__init__.py`: export `IterContext`.
- `fabricpc/tuning/bayesian_tuner.py:111-117`: `def iter_callback(ctx)`, reads `ctx.batch_idx`, `ctx.epoch_idx`, `ctx.metrics["energy"]`. Failure mode if missed: `train` raises `TypeError`, the tuner's `except Exception` at `bayesian_tuner.py:205` converts it to `optuna.TrialPruned`, and every trial is pruned with "failed during training" while the study reports success.
- `fabricpc/utils/dashboarding/callbacks.py`: the unified `create_iter_callback`; `create_epoch_callback` without the weight call; `create_detailed_iter_callback` deleted; module docstring without the "exception" sentence; imports `IterContext`, `build_clamps`, `batch_size_of`, `initialize_graph_state`, `run_inference_with_full_history`.
- `fabricpc/utils/dashboarding/__init__.py`: drop the `create_detailed_iter_callback` import and `__all__` entry.
- `fabricpc/utils/dashboarding/trackers.py`: `TrackingConfig.track_state`; the gate in `track_state`; docstring lines for both.
- `examples/transformer_demo.py`: as above.
- `examples/transformer_v2_demo.py:269`: `def iter_callback(ctx)`, `ctx.batch_idx`, `ctx.epoch_idx`, `ctx.metrics["energy"]`.
- `tests/test_fabricpc.py:546,558`: `lambda ctx: iters_half.append(1) or ctx.metrics`.
- `tests/test_trainer.py:1304,1327`: take `ctx`.
- `tests/test_bayesian_tuner.py:52-90` `_fake_train`: add `algorithm=kwargs.get("algorithm", "pc")` and `epoch_key=jax.random.fold_in(rng, i)` to the `EpochContext` it builds, and drive `iter_callback` with an `IterContext` (`batch_idx=49`, `state=None`, `batch={}` are acceptable in the fake) so the tuner's callback body executes under `verbose=True` in at least one test.
- `docs/user_guides/08_training_and_evaluation.md:131-164`: the iteration-callback fence becomes `def my_iter_callback(ctx): ... ctx.batch_idx ... ctx.metrics['energy']` and the paragraph lists the `IterContext` fields; the epoch-callback field list adds `algorithm` and `epoch_key`; the contract list gains "`ctx.state` and `ctx.batch` are not donated; the trainer drops its reference to the state after the callback".
- `docs/user_guides/09_experiment_tracking.md`: the dataclass fence (`:72-93`) and the options table gain `track_state`; the `track_state_distributions` row says it implies `track_state`; the `tracking_every_n_batches` row gains "weight distributions are logged by the iteration callback at this cadence". The "State Distributions" subsection (`:122-132`) states what the iteration callback does on a tracked batch (PC: re-run with history, one record every `state_tracking_every_n_infer_steps` steps; backprop: the feedforward state once) and its cost. The section "Per-batch state tracking with `make_train_step`" (`:211-247`) is deleted. "Advanced Usage: Custom Training Loop" (`:144-209`) stays: `train_step_with_history` collects the history inside the jitted step from the training settle itself, on every batch, without a second pass; its intro sentence says so. `test_doc_snippets.py` checks every fence.
- `CHANGELOG.md`: new `## [0.5.2] - <merge date>` section (the repo has no Unreleased section; 0.5.0 and 0.5.1 each bumped `pyproject.toml` and added a dated section in the release PR). Migration table rows: `iter_callback(epoch_idx, batch_idx, metrics)` → `iter_callback(ctx: IterContext)`; `create_detailed_iter_callback(tracker, structure)` in a custom loop → `train(..., iter_callback=create_iter_callback(tracker))` with `TrackingConfig(track_state=True)` or `track_state_distributions=True`; `create_epoch_callback` logging weight distributions once per epoch → the iteration callback logs them every `tracking_every_n_batches`; `EpochContext(...)` constructed by hand → add `algorithm` and `epoch_key`. New: the `IterContext` fields, `EpochContext.algorithm` and `.epoch_key`, `TrackingConfig.track_state`. The package is on PyPI, so the rows are the migration path for external callbacks; the 0.5.0 precedent is a clean break with a table row, followed here.
- `pyproject.toml`: version `0.5.2`.

Not migrated: `examples/mnist_aim_tracking.py` runs a custom loop over `train_step_with_history`, which records the training settle's own history inside the jitted step. `IterContext` cannot offer that without a second inference pass, so the loop is a different product, not a workaround for the gap this PR closes.

## Tests

- `tests/test_trainer.py`: `test_iter_callback_receives_floats_and_replaces` becomes `test_iter_context_fields_and_callback_replacement` (mirroring the epoch test) and `test_callback_exceptions_propagate` takes `ctx`. Assertions in the first, run with `start_epoch=5` and both algorithms: `isinstance(ctx, IterContext)`; `ctx.metrics` values are floats; `ctx.step` increments by one per call; `ctx.params` is a `GraphParams`; `ctx.state` is a `GraphState` whose node keys equal `structure.nodes`; `ctx.batch` holds the task-mapped keys; `ctx.algorithm == algorithm`; `ctx.epoch_idx` honours `start_epoch`; `ctx.epoch_key == fold_in(train_key, ctx.epoch_idx)` and `ctx.batch_key == fold_in(ctx.epoch_key, ctx.batch_idx)` (array equality). `test_epoch_context_fields_and_callback_replacement` adds the `algorithm` and `epoch_key` checks.
- `tests/test_sharding.py`: under the two-device mesh, the iter callback sees `ctx.batch["x"].sharding` equal to `NamedSharding(mesh, P("data"))` and `ctx.state` is a `GraphState`.
- `tests/test_dashboarding_callbacks.py` (new; `test_dashboarding_extractors.py` never touches the factories). A duck-typed stub tracker holding a real `TrackingConfig` (a plain dataclass, importable without Aim) and recording calls to `track_batch_energy`, `track_batch_energy_per_node`, `track_weight_distributions`, and `track_state`. A two-layer PC graph as in `tests/test_fabricpc.py:515-529` with `InferenceSGD(infer_steps=3)`, two batches, real `train`:
  - defaults: energy and weight calls at every batch, no state calls;
  - `track_state=True, tracking_every_n_batches=1, state_tracking_every_n_infer_steps=1` under PC: three `track_state` calls per batch with `infer_step` 0, 1, 2, each carrying a `GraphState`, and `nodes=None` when `nodes_to_track` is empty;
  - the same config under backprop (a `FeedforwardStateInit` graph): one `track_state` call per batch with `infer_step=0`;
  - `tracking_every_n_batches=2`: state calls at batch 0 only;
  - `create_epoch_callback` makes no `track_weight_distributions` call.
  Plus a unit test that `AimExperimentTracker.track_state` returns before touching `_ensure_initialized` when both state flags are `False` (construct the tracker with no Aim and a stub `_run`).
- `tests/test_bayesian_tuner.py`: the fake drives both callbacks (above); one test constructs the tuner with `verbose=True` so the print path executes.

## Alternatives considered

- **`IterContext` as a superset of `EpochContext` (chosen).** One rule for both callbacks: each context carries the base key and every key derived at or above its level, the algorithm, and the objects the trainer holds at that point. Future fields add without breakage.
- **Append `params` as a fourth positional argument.** Breaks every caller now and again on each later addition.
- **`rng_key` only, with the `fold_in` chain documented for callers to recompute.** Duplicates the trainer's key derivation in every consumer; a change to the derivation would silently desynchronize them. Rejected for `epoch_key` and `batch_key` fields.
- **Infer the algorithm from `structure.config["inference"]` instead of an `algorithm` field.** A backprop run on a graph that also carries an inference object would re-run PC settling for nothing on every tracked batch. Rejected; `algorithm` is a `train` argument like `config` and rides along the same way.
- **`state` on `EpochContext` too.** The only state available at an epoch boundary is the last batch's, a per-batch quantity sampled at an arbitrary loader position, with no consumer. Delivering it means either holding the previous batch's `GraphState` on device through every step (one full state of extra peak memory on every run that supplies an epoch callback, which the tuner and `create_tracking_callbacks` always do) or an `Optional` field that is `None` when the epoch's last batch was skipped for mesh divisibility. Rejected; a callback that wants a state at epoch end samples it from `IterContext` at the batch it chooses.
- **Always return the state from the internal step.** Every `train` call without a callback would keep one `GraphState` alive across the next step. Rejected for `with_state=(iter_callback is not None)` plus `del state` after the callback.
- **Device scalars in `IterContext.metrics`, callback decides when to sync.** Would let a probe that runs every N updates skip the host sync on the other batches. Rejected: `verbose=True` (the default) syncs every batch for the tqdm postfix anyway, the tracking callback wants floats every batch, and the stored `iter_results` entries would need a second materialization path for callback-returned device values.
- **Keep two dashboarding iteration factories, both over `IterContext`.** The earlier draft. It preserves a split that `IterContext` makes meaningless (the detailed factory's only extra input, `structure`, is now `ctx.structure`) and leaves `track_state_distributions`, `state_tracking_every_n_infer_steps`, and the state half of `nodes_to_track` dead through `create_tracking_callbacks`. Rejected for one factory gated by the config, with `track_state` added so the gate has an explicit switch.
- **Factory records `ctx.state` only; inference history stays in custom loops.** Cheaper (no second pass) but leaves `state_tracking_every_n_infer_steps` effective only through `train_step_with_history` loops and keeps the transformer demo's custom callback. Rejected (user decision 2026-09-08) for the re-run on tracked batches under PC.
- **Leave `examples/transformer_demo.py` on its custom loop.** The loop and `TrainingProgressBar` exist only because tracking needed the parameters, batch, and state per batch, which `IterContext` now supplies. Keeping them would keep a second copy of the epoch schedule and progress bar in the repository with its rationale gone. Rejected.
- **A separate `probe_every=N, probe=callable` trainer parameter.** A second per-batch mechanism beside `iter_callback` with its own sync and return semantics.
- **Status quo: probes use `make_train_step` in a custom loop.** Every probe re-implements the loop and the callback plumbing; the analysis script, guide 09, and the transformer demo each carry one.

## Out of scope, with reasons

- `TrackingConfig.tracking_every_n_epochs` is read by nothing before or after this PR. Removing it is a `TrackingConfig` API change unrelated to callback visibility; flagged here so it is decided on its own.
- On `matthew_cedric/epc`, the `--track_lambda_max` loop in `scripts/epc_analysis.py` becomes an `iter_callback` that runs the power iteration when `ctx.step % N == 0` on `ctx.params`. That file lives on the other branch and migrates when it rebases.

## Verification

1. `python -m pytest tests/test_trainer.py tests/test_fabricpc.py tests/test_sharding.py tests/test_bayesian_tuner.py tests/test_dashboarding_callbacks.py tests/test_dashboarding_extractors.py tests/test_doc_snippets.py -q` green. `test_sharding.py` needs `XLA_FLAGS=--xla_force_host_platform_device_count=2`.
2. `python examples/transformer_v2_demo.py --verbose` (one epoch or fewer batches) prints per-batch energy through the migrated callback.
3. `python examples/transformer_demo.py --num_epochs 0.05` for both `--mode` values runs on `train` end to end; with Aim installed, the run contains `energy`, `node_energy`, weight distributions, and `z_latent_mean` at `infer_step` 0 and 5 under PC.
4. `grep -rn "iter_callback\|EpochContext(\|create_detailed_iter_callback" --include=*.py --include=*.md fabricpc tests examples docs/user_guides CHANGELOG.md`: no three-argument `iter_callback` definition remains; `create_detailed_iter_callback` appears only in the CHANGELOG migration tables. `docs/dev_plans_archive/` is history and is not edited.
5. `python -c "from fabricpc.training import IterContext, EpochContext; print(IterContext._fields, EpochContext._fields)"` lists the fields in the tables above.
6. `ruff check` and `black --check` clean (unused imports in the demo).

## Sequencing

One commit on a branch off `main`; PR message via a temporary file in the project root. After merge, `matthew_cedric/epc` rebases onto it and its probe consumes `IterContext` directly.
