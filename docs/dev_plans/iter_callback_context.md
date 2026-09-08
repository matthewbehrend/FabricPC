# `IterContext` for `train(iter_callback=...)`

Self-contained PR against `main`, to ship before the ePC review-fixes branch (`docs/dev_plans/epc_review_fixes_lanczos_regime_probe.md`), which rebases onto it.

## Context

`train()` calls `iter_callback(epoch_idx, batch_idx, metrics)` after every batch (`fabricpc/training/trainer.py:623`). The callback cannot see the parameters, so any per-update diagnostic that needs them (the ePC λ_max probe, weight-norm logging, gradient-scale checks) has to re-implement the training loop over `make_train_step`, which is what `scripts/epc_analysis.py` did. `epoch_callback` already receives an `EpochContext` with `params`, `opt_state`, `structure`, `config`, `rng_key`, and `metrics` (`trainer.py:105`). This PR gives `iter_callback` the same shape: one `IterContext` argument. The ePC tracking data showed λ_max growing threefold within 50 updates during a collapse, so per-epoch access is not enough for that diagnostic.

## Design

`IterContext` NamedTuple in `fabricpc/training/trainer.py`, beside `EpochContext`, exported from `fabricpc.training`:

| Field | Meaning |
|---|---|
| `epoch_idx` | epoch index including `start_epoch` |
| `batch_idx` | batch index within the epoch |
| `step` | optimizer updates applied in this call, after this batch |
| `params`, `opt_state` | parameters and optimizer state after this batch's update |
| `structure`, `config` | the graph and the training config passed to `train` |
| `rng_key` | the base training key; this batch's key is `fold_in(fold_in(rng_key, epoch_idx), batch_idx)` |
| `batch` | the converted batch dict fed to this step (task-mapped keys), so a probe can measure on the training batch instead of a fixed one |
| `metrics` | the per-batch float metrics (`energy`, `target_energy`) |

`iter_callback(ctx: IterContext) -> Any`; a non-None return still replaces that batch's `iter_results` entry; supplying the callback still forces the per-batch device sync. The `EpochContext` donation caveat applies verbatim: the training step donates the params and opt_state buffers, so `ctx.params` is valid during the callback and must be copied (`tree_map(jnp.copy, ...)`) if retained. Grows by field addition only, read fields by name.

The `verbose` tqdm path and `epoch_callback` are unchanged. `create_detailed_iter_callback` (dashboarding, custom loops over `make_train_step`) keeps its `(epoch_idx, batch_idx, metrics, final_state)` signature: it is not a `train` callback.

## Migrations (same PR, no compatibility shim)

- `fabricpc/training/trainer.py`: build `IterContext` at the call site; `train` docstring line for `iter_callback`.
- `fabricpc/training/__init__.py`: export `IterContext`.
- `fabricpc/utils/dashboarding/callbacks.py:17-37` `create_iter_callback`: `def iter_callback(ctx: IterContext)`, reads `ctx.metrics["energy"]`, `ctx.epoch_idx`, `ctx.batch_idx`; return annotation `Callable[[IterContext], Dict[str, float]]`. `create_tracking_callbacks` forwards it unchanged. Module docstring sentence about the two signatures.
- `examples/transformer_v2_demo.py:269`: `def iter_callback(ctx)`, `ctx.batch_idx`, `ctx.epoch_idx`, `ctx.metrics["energy"]`.
- `tests/test_fabricpc.py:546,558`: `lambda ctx: iters_half.append(1) or ctx.metrics`.
- `tests/test_trainer.py:1304,1327` (`test_iter_callback_receives_floats_and_replaces` and its neighbour): take `ctx`; add assertions that `ctx.metrics` values are floats, `ctx.step` increments by one per call, `ctx.params` is a `GraphParams`, `ctx.batch` holds the task-mapped keys, and `ctx.epoch_idx` honours `start_epoch`.
- `docs/user_guides/08_training_and_evaluation.md:131-138`: the fence becomes `def my_iter_callback(ctx): ... ctx.batch_idx ... ctx.metrics['energy']`, and the paragraph lists the fields as the epoch-callback paragraph does. `09_experiment_tracking.md` uses the factory and needs no edit unless it states the signature (verify).
- `CHANGELOG.md` unreleased, Breaking changes: one bullet, plus a Migration table row `iter_callback(epoch_idx, batch_idx, metrics)` → `iter_callback(ctx: IterContext)`: read `ctx.epoch_idx`, `ctx.batch_idx`, `ctx.metrics`; `ctx.params`, `ctx.opt_state`, `ctx.step`, `ctx.batch` are new.

## Alternatives considered

- **`IterContext` (chosen).** Same contract as `epoch_callback`; one breaking signature with five callers, all in-repo; future fields add without breakage.
- **Append `params` as a fourth positional argument.** Still breaks every caller, and each later addition breaks them again.
- **A separate `probe_every=N, probe=callable` trainer parameter.** No breaking change, but a second per-batch mechanism beside `iter_callback` with its own sync and return semantics; two ways to do one thing.
- **Keep `iter_callback` as is; probes use `make_train_step` in a custom loop.** Status quo; every probe re-implements the loop, evaluation, and callback plumbing (the analysis script did, 120 lines).

## Verification

1. `python -m pytest tests/test_trainer.py tests/test_fabricpc.py tests/test_dashboarding*.py tests/test_doc_snippets.py -q` green.
2. `python examples/transformer_v2_demo.py --verbose` (smoke, one epoch or fewer batches) prints per-batch energy through the migrated callback.
3. `grep -rn "iter_callback" --include=*.py --include=*.md .` shows no three-argument form outside `create_detailed_iter_callback`.

## Sequencing

One commit on a branch off `main`; PR message via a temporary file in the project root. After merge, `matthew_cedric/epc` rebases onto it and the review-fixes plan's probe consumes `IterContext` directly.
