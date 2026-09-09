# Training and Evaluation

Tutorial-style guide covering `train`/`evaluate`, resume, callbacks, custom
loops, metrics, autoregressive generation, and mesh-based multi-device
training.

One API trains every FabricPC graph with either learning rule, selected by
`algorithm`:

```python
from fabricpc.training import train, evaluate

result = train(params, structure, train_loader, optimizer,
               {"num_epochs": 20}, train_key)                      # PC (default)
result = train(params, structure, train_loader, optimizer,
               {"num_epochs": 20}, train_key, algorithm="backprop")
```

## The Energy Framing

Both algorithms clamp identically during training: the input **and** the
target are clamped into their task-mapped nodes. They differ in three places
inside the step:

| sub-step | PC | backprop |
|---|---|---|
| produce state | initialize, then settle the latents via `run_inference` | single feedforward pass (`FeedforwardStateInit`), no inference |
| objective | energy summed over all internal (`in_degree > 0`) nodes, per prediction | energy of the clamped target nodes, per prediction |
| gradients | local per-node Hebbian gradients | `jax.value_and_grad` through the forward pass |

The backprop objective is the clamped target node's energy — the negative log
probability the output node's energy functional assigns to the clamped target
given the feedforward prediction. There is no `loss_type` option: **the
output node's energy functional in the graph definition selects the loss.**
`CrossEntropyEnergy()` on the output gives cross-entropy training;
`GaussianEnergy(precision=p)` gives `0.5 * p * ||y - mu||^2` per prediction.

"Per prediction" means divided by N, the prediction count: the total number
of clamped-target prediction positions in the batch (`batch` for
classification, `batch * seq_len` for token targets; `batch` when the graph
has no clamped target). With several clamped target heads N is the sum of
their prediction positions (two same-shape heads give N = 2 * batch), so
adding a head halves the step every shared parameter takes at a fixed
learning rate. Both algorithms divide their objective and their
gradients by the same N, so a learning rate, clipping threshold, or Adam
epsilon means the same under either algorithm and across batch sizes and
sequence lengths. `fabricpc.training.grad_denominator(structure, clamps)`
returns N, and `fabricpc.training.pc_weight_gradients` is the PC gradient
already divided by it, for custom loops.

Integer or bool targets are one-hot encoded automatically from their dtype
(with the class count taken from the target node's last shape axis), so token
loaders can keep yielding compact int32 ids.

## train()

```python
from fabricpc.training import train

result = train(
    params,                    # initial (or checkpointed) parameters
    structure,                 # GraphStructure from graph()
    train_loader,              # iterable of (x, y) tuples or {"x": ..., "y": ...} dicts
    optimizer,                 # any optax optimizer
    {"num_epochs": 20},        # config; fractional epochs supported (e.g. 1.5)
    train_key,                 # base RNG key
    algorithm="pc",            # or "backprop"
    verbose=True,
)
```

`config` reads only `num_epochs`, which is required (a missing key raises
`ValueError` — there is no default epoch count); everything else passes
through untouched to callbacks and experiment harnesses. Settling parameters (`infer_steps`,
`eta_infer`) live in the inference object inside the graph
(`graph(..., inference=InferenceSGD(eta_infer=0.05, infer_steps=20))`), not
in `config`. The retired keys `loss_type` and `use_causal_mask` raise
`ValueError` with migration instructions.

### TrainResult and resume

`train` returns a `TrainResult` named tuple:

- `params` — the trained parameters,
- `opt_state` — the final optimizer state,
- `step` — optimizer updates applied in this call,
- `iter_results` — 2D list `[epoch][batch]` of per-batch metric dicts,
- `epoch_results` — list of per-epoch mean metric dicts.

Because `opt_state` is returned and accepted, training is resumable without
resetting Adam moments or an optax schedule's count:

```python
first = train(params, structure, train_loader, optimizer,
              {"num_epochs": 5}, train_key)

# ... later: continue exactly where the first call stopped
second = train(first.params, structure, train_loader, optimizer,
               {"num_epochs": 5}, train_key,
               opt_state=first.opt_state, start_epoch=5)
```

`start_epoch` offsets the epoch index used for RNG derivation, so the
interrupted run reproduces the uninterrupted run's random stream bitwise.

### The RNG contract

The training key affects only latent initialization
(`initialize_graph_state`); inference is deterministic and the package has no
dropout or noise. Keys derive as `fold_in(rng_key, epoch_idx)` then
`fold_in(epoch_key, batch_idx)` — a pure function of
`(rng_key, epoch_idx, batch_idx)`, independent of loader length.

### Training metrics

Each step produces two device scalars, materialized to floats at epoch
boundaries (or per batch when `verbose=True` or an `iter_callback` is
supplied — both force a per-batch device sync):

- `energy` — the objective the gradients descend, per prediction. Its node
  set is **algorithm-dependent** (all internal nodes for PC, target nodes
  only for backprop), so comparing it across algorithms compares different
  node sets; under backprop it equals `target_energy`.
- `target_energy` — the target-node energy divided by the number of
  predictions (`batch * seq_len` for sequences, `batch` for classification).
  Under `CrossEntropyEnergy` this is the teacher-forced per-token
  cross-entropy, so `exp(target_energy)` is the training perplexity.

## Callbacks

**Iteration callback** — called after each batch with a single `IterContext`
argument (fields, in order: the `EpochContext` fields `epoch_idx`, `step`,
`params`, `opt_state`, `structure`, `config`, `rng_key`, `metrics`,
`algorithm`, `epoch_key`, then `batch_idx`, `state`, `batch_key`, `batch`;
new fields are appended). `metrics` holds this batch's float
metrics; `state` is the batch's `GraphState` (settled under PC, the
feedforward pass under backprop); `batch` is the converted batch dict and
`batch_key` the key the step used for latent initialization; `step` counts
optimizer updates in this `train` call, this batch included:

```python
from fabricpc.training import IterContext

def my_iter_callback(ctx: IterContext):
    if ctx.batch_idx % 100 == 0:
        print(f"  batch {ctx.batch_idx}: energy={ctx.metrics['energy']:.4f}")

result = train(..., iter_callback=my_iter_callback)
```

**Epoch callback** — called after each epoch with a single `EpochContext`
argument (fields: `epoch_idx`, `step`, `params`, `opt_state`, `structure`,
`config`, `rng_key`, `metrics`, `algorithm`, `epoch_key`; new fields are
appended; `epoch_key` is `fold_in(rng_key, epoch_idx)`, the key the epoch's
batch keys derive from):

```python
from fabricpc.training import EpochContext, evaluate

def my_epoch_callback(ctx: EpochContext):
    metrics = evaluate(ctx.params, ctx.structure, test_loader, ctx.config, ctx.rng_key)
    print(f"  Epoch {ctx.epoch_idx}: acc={metrics['accuracy']:.4f}")
    return metrics

result = train(..., epoch_callback=my_epoch_callback)
```

Contract guarantees:

- Callback exceptions propagate — the Bayesian tuner's pruning is
  exception-based and depends on this.
- A non-None return replaces the stored history entry: the callback above
  stores its eval metrics in `result.epoch_results[epoch]`.
- The internal step donates its parameter buffers, so copy `ctx.params`
  (`jax.tree_util.tree_map(jnp.copy, ctx.params)`) if you retain it past the
  callback; the next training step invalidates it.
- `ctx.state` and `ctx.batch` are not donated. The trainer drops its own
  reference to the state when the iteration callback returns, so retain it
  freely; a callback that does not retain it adds no device memory. Without
  an `iter_callback` the step does not return the state at all.

## evaluate()

Evaluation clamps the inputs and leaves the targets free; PC settles the
latents, backprop takes the feedforward pass.

```python
from fabricpc.training import evaluate

metrics = evaluate(result.params, structure, test_loader, {}, eval_key)
print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
```

### Default metrics are graph-derived

With `metrics=None` (the default), the reported keys follow the graph — the
target node's own energy functional selects them, mirroring the training-side
energy framing:

| key | when | definition |
|---|---|---|
| `target_energy` | always | `E(y, z_mu)` under the target node's functional, per prediction |
| `accuracy` | always | `argmax(z_mu, -1)` compared to `argmax(y, -1)` for one-hot targets (same rank as `z_mu`), or directly to integer class labels of lower rank; per prediction (argmax-based: meaningful for class-like targets, not continuous ones) |
| `cross_entropy` | target functional is `CrossEntropyEnergy` | identical to `target_energy` in that case; the conventional name |
| `perplexity` | target functional is `CrossEntropyEnergy` | `exp(cross_entropy)` |
| `energy` | `algorithm="pc"` | internal energy per prediction, the PC training objective's scale |

A graph with no target task key raises under the defaults; pass an explicit
metrics dict to evaluate such a graph.

**Note on eval energy**: for feedforward DAGs with `FeedforwardStateInit`,
evaluation energy is near zero because a free output starts at its
feedforward equilibrium (`z_latent = z_mu`, zero error). Use accuracy or
perplexity to assess task quality.

### Custom metrics: the EvalMetric contract

A metric is an `EvalMetric(fn, finalize)`:

- `fn(state, batch, structure) -> (value, weight)` — two per-sample arrays of
  shape `(batch,)`, computed inside the jitted eval step on the settled
  state. The weight is the metric's own denominator (predictions per sample),
  so per-sample and per-token metrics aggregate correctly.
- Aggregation is owned by `evaluate`: Σvalue and Σweight accumulate across
  batches (and devices), padded samples get zero weight, and the result is
  `finalize(Σvalue / Σweight)`. A ragged final batch never skews a mean.
- `finalize` applies once to the global mean: perplexity is `exp` of the
  aggregated cross-entropy, not a mean of per-batch `exp`s.

```python
from fabricpc.training import EvalMetric, evaluate, metrics
import jax.numpy as jnp

def top2_fn(state, batch, structure):
    z_mu = state.nodes[structure.task_map["y"]].z_mu
    top2 = jnp.argsort(-z_mu, axis=-1)[..., :2]
    labels = jnp.argmax(batch["y"], axis=-1)
    hit = jnp.any(top2 == labels[..., None], axis=-1).astype(jnp.float32)
    return hit, jnp.ones_like(hit)

evaluate(params, structure, test_loader, {}, eval_key,
         metrics={"top2_accuracy": EvalMetric(fn=top2_fn)})          # exactly this
evaluate(params, structure, test_loader, {}, eval_key,
         metrics={**metrics.default_metrics(structure, "pc"),
                  "top2_accuracy": EvalMetric(fn=top2_fn)})          # defaults + custom
```

A bare callable is accepted too (wrapped with the identity finalize). The
built-ins live in `fabricpc.training.metrics`: `accuracy`, `cross_entropy`,
`perplexity`, `target_energy`, `internal_energy`.

## Custom Training Loops: make_train_step()

For per-step control (custom schedules, state inspection, dashboards that
need the settled state), build the jitted step directly:

```python
from fabricpc.training import make_train_step
import jax
import jax.numpy as jnp

step = make_train_step(structure, optimizer)          # or algorithm="backprop"
opt_state = optimizer.init(params)

for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        batch = {"x": jnp.array(x), "y": jnp.array(y)}
        rng_key, step_key = jax.random.split(rng_key)
        params, opt_state, metrics, final_state = step(params, opt_state, batch, step_key)
```

`metrics` holds the `energy`/`target_energy` device scalars; `final_state` is
the settled (PC) or feedforward (backprop) `GraphState`. The step does not
donate its inputs, so the initial `params` stay valid.

## Autoregressive Language Modeling

Next-token models use the same `train`/`evaluate` — there is no
autoregressive flag. Two things make a run autoregressive, both derived from
the graph and the data:

- **Causal masking is graph-derived.** A v1 transformer graph declares a
  `causal_mask` entry in its `TaskMap`; `train`/`evaluate`/`generate` then
  clamp a lower-triangular `(batch, 1, seq, seq)` mask into that node. The
  decomposed v2 transformer masks internally via
  `MhaResidualNode(is_causal=True)`, so its task map has no such entry and
  nothing is clamped. No caller flag exists.
- **One-hot is dtype-derived.** Loaders yield `(batch, seq_len)` int32 token
  ids for both `x` and `y` (`y` is `x` shifted one position left, see
  [Data Loaders](14_api_data.md)); the integer target is one-hot encoded
  inside the step.

Perplexity comes free during training as `exp(metrics["target_energy"])`,
because the clamped target's energy under `CrossEntropyEnergy` is the
teacher-forced per-token cross-entropy. Evaluation reports `cross_entropy`
and `perplexity` whenever the output functional is `CrossEntropyEnergy`.

**Perplexity** is the effective number of tokens the model chooses among at
each step: a model with perplexity 20 is as uncertain as a uniform choice
over 20 tokens. It is `exp(CE)` for the mean per-token cross-entropy `CE`.

### generate()

```python
from fabricpc.training import generate

x_indices, _ = next(iter(loader))
prompt = x_indices[0]        # one sequence: 1-D int32 token ids, shape (seq_len,)

tokens = generate(
    trained_params,
    structure,
    prompt=prompt,
    max_new_tokens=200,
    rng_key=gen_key,
    temperature=0.8,
    top_k=40,
)
print(loader.decode(tokens))  # tokens: shape (seq_len + 200,)
```

The prompt is 1-D `(prompt_len,)` for a single sequence or 2-D
`(batch, prompt_len)` for a batch; the returned tokens match
(`(prompt_len + max_new_tokens,)` or `(batch, prompt_len + max_new_tokens)`).
`loader.decode` accepts one 1-D sequence, so decode a batched result row by
row.

- `temperature` divides the logits before sampling: below 1.0 concentrates
  probability on the most likely tokens, above 1.0 flattens the distribution.
- `top_k` keeps only the k most probable tokens.
- `top_p` keeps the smallest token set whose cumulative probability reaches p.
- `algorithm` selects the state the tokens are sampled from, with the same
  validation as `train`/`evaluate`: `"pc"` (default) settles the graph via
  `run_inference` each step, `"backprop"` samples from the single feedforward
  pass — use it for backprop-trained models and for graphs built with
  `inference=None`.

Generation slides a context window of the model's `seq_len`: once the
sequence exceeds it, the oldest tokens drop out of the context.

End-to-end example: `examples/transformer_v2_demo.py`
(`--mode pc|backprop`, `--tokenizer char|bpe`); hyperparameter search:
[Experiment Framework API](15_api_experiments.md).

## Multi-Device Training (mesh)

Data parallelism runs on jit + `NamedSharding` over an explicit mesh. Build a
mesh over the `data` axis and pass it to `train`/`evaluate`/`make_train_step`
(the `model` axis name is reserved for future model parallelism):

```python
import jax
from fabricpc.training import train, evaluate

mesh = jax.make_mesh((jax.device_count(),), ("data",))

result = train(params, structure, train_loader, optimizer,
               {"num_epochs": 20}, train_key, mesh=mesh)
metrics = evaluate(result.params, structure, test_loader, {}, eval_key, mesh=mesh)
```

Parameters are replicated; each batch is sharded on its leading axis across
the `data` mesh axis. Ragged batches: training skips a batch whose size is
not divisible by the data-axis size (one-time warning); evaluation zero-pads
it and gives the padded samples zero metric weight, so every real sample
counts exactly once. Without `mesh`, the same jitted step runs on a single
device.

To test multi-device code on a CPU-only machine:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=2 python your_script.py
```

## Statistical A/B Experiments

For rigorous comparisons across multiple trials, use the experiment
framework. Arms stay trainer-agnostic: the algorithm is bound with
`functools.partial`, and the arm metric should be `accuracy` or `perplexity`
(the training `energy` is not cross-algorithm comparable):

```python
import functools
from fabricpc.experiments import ExperimentArm, ABExperiment
from fabricpc.training import train, evaluate

arm_pc = ExperimentArm(name="PC", model_factory=create_model, train_fn=train,
                       eval_fn=evaluate, optimizer=optimizer, train_config=config)
arm_bp = ExperimentArm(name="Backprop", model_factory=create_model,
                       train_fn=functools.partial(train, algorithm="backprop"),
                       eval_fn=functools.partial(evaluate, algorithm="backprop"),
                       optimizer=optimizer, train_config=config)

experiment = ABExperiment(arm_a=arm_pc, arm_b=arm_bp, metric="accuracy",
                          data_loader_factory=loader_fn, n_trials=5)
results = experiment.run()
results.print_summary()  # Paired t-test, Cohen's d, effect sizes
```

For more than two arms, `PlannedMultiContrastExperiment` runs N arms with
constructor-declared planned contrasts; `ABExperiment` is its 2-arm wrapper.
See the [Experiment Framework API](15_api_experiments.md) for full details.
