# Model checkpointing on Orbax (fabricpc 0.5.x, follow-up to the unified trainer)

## Context

Q3 roadmap item: save/load of {`GraphParams`, optimizer state, structure metadata, rng state, step
counter}, a save-every-N / keep-best-K hook in the trainer's epoch callback, and resumability
demonstrated — not just supported — in the examples. Checkpointing is a hackathon prerequisite:
participants training for hours need resume, and pre-trained weights let them start from a working
model.

This PR depends on the unified trainer PR (`docs/dev_plans/unified_trainer_api.md`), which supplies
everything checkpointing consumes: `TrainResult` returning `opt_state` and `step`,
`train(..., opt_state=, start_epoch=)` for resume, the `fold_in` RNG stream that makes "rng state"
the pair `(base_key, epoch_idx)`, `EpochContext` exposing `opt_state`/`step` to the hook, and mesh
sharding under jit. The abandoned branch `origin/feature/model-checkpointing` (745a9a8 + 31ea9b1)
is design input only — its own `mnist_demo` comment documents that real resume was inexpressible
against the old trainer ("cannot thread real optimizer momentum across the save/load boundary");
its choices are harvested per the table below, not rebased.

## Decisions

| Decision | Choice |
|---|---|
| Container | Orbax (`orbax-checkpoint`, already a declared dependency). Composite checkpoints: array items via `StandardSave`/`StandardRestore`, the FabricPC-level metadata + structure snapshot via `JsonSave`. Native sharded save/restore under jit + `NamedSharding` (no host gather; restore into a different device topology). |
| Retention | `ocp.CheckpointManager` with `max_to_keep` / `best_fn` / `best_mode` — save-every-N and keep-best-K are Orbax options, not reimplemented logic. |
| Persisted set | `params` (always), `opt_state`, structure snapshot (best-effort), `rng_key` + `epoch` + `step` (the resume triple), optional `GraphState` (persistent-latent / Hopfield runs), user metadata dict. |
| Structure snapshot | `(class_path, config)` per component, preconditioned on the constructor round-trip fix (below). Best-effort on load: a snapshot that fails to rebuild degrades to `structure=None` with a warning and never blocks weight recovery; a snapshot that rebuilds but disagrees with the params on shape raises. Persist `node_order` + muPC factors (avoids re-running the builder's best-effort topological sort on cyclic graphs). |
| Schema version | FabricPC-level `format` magic + `format_version` in the JSON metadata item, reject-newer semantics. Orbax versions its container, not our schema. |
| Precursor fix | Make every `FrozenConfig` component round-trip `cls(**instance.config) == instance`, fixing the three offenders (`UniformInitializer` renames config keys, `ZerosInitializer` drops `gain`, `GlobalStateInit` nests a component), with a contract test over all registered components. This replaces the abandoned branch's `cls.__new__` + private-attribute reconstruction. |
| Resume granularity | Epoch boundaries only. Mid-epoch resume (loader position) is out of scope, documented. |
| Dependencies | `flax` already dropped in the trainer PR. Raise the `orbax-checkpoint` floor to the tested version during implementation. |

## Harvest evaluation of `feature/model-checkpointing`

Each branch choice judged on merit (the branch is abandoned; nothing is rebased):

| Branch choice | Verdict | Reason |
|---|---|---|
| flax-msgpack + JSON directory container | drop | Orbax owns the container: sharded save/restore with no host gather, async save, restore into a different device topology — the properties large-model training needs and msgpack cannot provide. |
| sha256 checksums; staging dir + atomic rename + rollback | drop | Orbax's commit protocol owns atomicity and integrity; reimplementing them above it is dead weight. |
| `format` magic + `FORMAT_VERSION` gate, reject-newer semantics | keep, narrowed | Orbax versions its container, not FabricPC's schema; the FabricPC-level metadata and structure snapshot still need a schema version. |
| Self-describing params: weights load with no model code | keep the requirement | `StandardRestore` without a target returns the saved nesting; rebuilding `GraphParams`/`NodeParams` from it is a thin adapter over the existing pytree registrations in `core/types.py`. |
| opt_state restored into an `optimizer.init(params)` template; wrong optimizer raises | keep the pattern | Orbax-native: restore against an abstract `jax.eval_shape(optimizer.init, params)` target — which also carries shardings, something the flax path cannot express. |
| Structure snapshot rebuilt via `cls.__new__` + private-attribute assignment | replace the mechanism | `__new__` was chosen because three constructors do not round-trip their own `.config` — a patch around defective shared components. Fix the constructors (precursor above); the snapshot reduces to `(class_path, config)` with no coupling to private attribute layouts. |
| Best-effort structure degradation; persisted `node_order` | keep | Weight recovery must survive code evolution; `node_order` persistence keeps cyclic graphs deterministic. |
| `Checkpoint` NamedTuple return; `overwrite=False` default; optional `GraphState` | keep | Container-agnostic API decisions. |
| No rng state, no step counter | gap | Filled here via the unified trainer's `fold_in` stream and `step` counter: the resume triple `(rng_key, epoch, step)` is three scalars. |
| 21 tests | harvest intents selectively | Round-trip exactness, resume parity, wrong-optimizer raise, structure-degrade, shape-mismatch survive; corruption/atomicity/staging tests become Orbax's responsibility — do not re-test the dependency. |

## Public API

```python
# fabricpc/serialization.py
class Checkpoint(NamedTuple):
    params: GraphParams
    opt_state: Optional[Any]          # None unless stored AND optimizer= given
    structure: Optional[GraphStructure]  # best-effort; None never blocks params
    state: Optional[GraphState]       # persistent-latent runs only
    rng_key: Optional[jax.Array]
    epoch: Optional[int]              # next start_epoch
    step: Optional[int]
    metadata: Dict[str, Any]          # user dict from save time

def save_checkpoint(path, params, *,
                    opt_state=None, structure=None, state=None,
                    rng_key=None, epoch=None, step=None,
                    metadata=None, overwrite=False) -> Path

def load_checkpoint(path, *, optimizer=None, strict=True) -> Checkpoint
    # strict=True: raise if the file holds opt_state but no optimizer was given.
    # params always load — self-describing, no model code required.

def create_checkpoint_callback(directory, *,
                               every_n_epochs=1,
                               keep_last=None,           # CheckpointManagerOptions.max_to_keep
                               keep_best=None,           # (metric_key, "min"|"max") -> best_fn
                               ) -> Callable[[EpochContext], None]
    # An epoch_callback saving {params, opt_state, rng_key, epoch, step, metrics}
    # from the EpochContext through an ocp.CheckpointManager. Returns None (never
    # replaces stored history). keep_best has no default metric_key: energy is the
    # optimization objective, not a performance metric, so the caller must name
    # one — e.g. ("target_energy", "min"), or a val metric their own callback
    # merged in.
```

Exports: `save_checkpoint, load_checkpoint, Checkpoint` from `fabricpc`;
`create_checkpoint_callback` from `fabricpc.training`.

The resume idiom the docs teach:

```python
ckpt = load_checkpoint(ckpt_dir, optimizer=optimizer)
result = train(ckpt.params, structure, loader, optimizer, config, ckpt.rng_key,
               opt_state=ckpt.opt_state, start_epoch=ckpt.epoch)
```

which is bitwise-equal to the uninterrupted run (fold_in stream + restored optimizer moments).

## Design notes

- **Composite layout (one Orbax checkpoint):** items `params` (StandardSave), `opt_state`
  (StandardSave), `graph_state` (StandardSave, optional), `meta` (JsonSave: format magic,
  format_version, fabricpc/jax/orbax versions, rng_key bytes, epoch, step, user metadata,
  structure snapshot). Restore order: `meta` first (version gate before any array decode).
- **Params without model code:** `StandardRestore` with no target yields the saved nesting;
  the adapter rebuilds `NodeParams`/`GraphParams` by key. Structure and params decoding stay
  decoupled — a renamed node class costs the structure snapshot (warning), never the weights.
- **opt_state:** restored against `jax.eval_shape(optimizer.init, params)`; a different optimizer
  fails on tree mismatch rather than silently misloading. With a mesh, the abstract target carries
  the target shardings, so restore lands sharded without a host round-trip.
- **Sharded restore across topologies:** the abstract target's shardings may reference a different
  mesh than the one saved under; Orbax reshards on load. Tested with the 2-CPU-device trick.
- **Hook composition:** `train` takes one `epoch_callback`; users composing the checkpoint hook
  with their own wrap both in a lambda (documented). The hook returns None so `epoch_results`
  keeps the metrics dict.

## Alternatives considered

- **Pickle-based checkpointing.** Trivial to write, no schema work. Rejected: silently
  version-fragile across refactors and JAX versions.
- **The branch's flax-msgpack directory format (implemented, 21 tests).** Rejected: Orbax owns
  atomicity, integrity, async save, and sharded restore; keeping msgpack means hand-rolling a
  device gather for sharded params and leaves the declared `orbax-checkpoint` dependency unused.
- **`cls.__new__` + private-attribute structure reconstruction (the branch).** Rejected: couples
  every old checkpoint to internal attribute layouts; the constructor round-trip fix removes the
  reason it existed.
- **Structure required on load.** Rejected: weight recovery must survive code evolution
  (class renames cost the snapshot, not the model).
- **Retention logic inside the hook.** Rejected: `ocp.CheckpointManager` already implements
  `max_to_keep`/`best_fn`; reimplementing invites divergence.

## File changes

**Created:** `fabricpc/serialization.py`, `tests/test_serialization.py`.

**Modified:** `fabricpc/core/initializers.py` (`UniformInitializer`, `ZerosInitializer` config
round-trip), `fabricpc/graph_initialization/state_initializer.py` (`GlobalStateInit` round-trip),
`fabricpc/__init__.py` + `fabricpc/training/__init__.py` (exports), `pyproject.toml`
(orbax floor), `examples/mnist_demo.py` (`--checkpoint` resume demo),
`examples/transformer_v2_demo.py` (`--save`/`--load` + `generate`),
`docs/user_guides/08_training_and_evaluation.md` (+ a checkpointing section or new guide),
`docs/user_guides/16_troubleshooting.md` (replace the roll-your-own Orbax snippet), `CHANGELOG.md`.

## Implementation steps

1. Constructor round-trip fix + `FrozenConfig` contract test parametrized over every registered
   activation/energy/initializer/inference/state-init component (`cls(**c.config)` reproduces `c`).
2. `serialization.py`: structure snapshot codec (`(class_path, config)`, `node_order`, muPC
   factors, schema version) with best-effort restore.
3. `save_checkpoint`/`load_checkpoint` on Orbax composite items; params-without-target adapter;
   opt_state via `eval_shape` template.
4. `create_checkpoint_callback` on `ocp.CheckpointManager`.
5. Demos: `mnist_demo.py --checkpoint <dir>` — train k epochs, save, load, resume, and **assert**
   final params match an uninterrupted run bitwise; `transformer_v2_demo.py` save → fresh-process
   load (no builder call) → `generate`.
6. Tests (below), docs, CHANGELOG.

## Test plan

- Round-trip exactness: params values + dtypes (incl. bf16/fp16/int embeddings), user metadata,
  rng/epoch/step triple.
- Resume parity: interrupted-equals-uninterrupted bitwise through save/load (the demo's assertion,
  as a test on a tiny graph).
- Wrong optimizer raises; `strict=False` skips opt_state; params-only checkpoints load.
- Structure degrade: monkeypatched-unimportable class → warning + `structure=None`, params intact;
  rebuilt-structure/params shape mismatch raises; future `format_version` raises.
- `GraphState` round-trip (Hopfield-style).
- Retention: `every_n_epochs`, `keep_last`, `keep_best` on a fake metrics sequence.
- Sharded: save under a 2-CPU-device `"data"` mesh, restore to single device and back
  (`XLA_FLAGS=--xla_force_host_platform_device_count=2`).
- Not tested here: corruption, atomic-commit crash windows — Orbax's responsibility.

## Verification

```bash
.venv/bin/python -m pytest tests/test_serialization.py -q
JAX_PLATFORMS=cpu .venv/bin/python examples/mnist_demo.py --checkpoint /tmp/mnist_ckpt --epochs 4
JAX_PLATFORMS=cpu .venv/bin/python examples/transformer_v2_demo.py --save /tmp/tf_ckpt --num_epochs 0.02
JAX_PLATFORMS=cpu .venv/bin/python examples/transformer_v2_demo.py --load /tmp/tf_ckpt --generate
XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
    .venv/bin/python -m pytest tests/test_serialization.py -q -k shard
```

## Open

- Whether the hook also snapshots `GraphState` for persistent-latent runs (needs a step that
  returns `final_state`; the epoch loop's step does not — likely a custom-loop-only feature via
  `make_train_step` + `save_checkpoint(state=...)`).
- Orbax floor version — pin to the lowest release whose `CheckpointManager` options and
  `StandardRestore`-without-target behavior the tests pass against.
