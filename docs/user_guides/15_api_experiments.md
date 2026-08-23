# Experiment Framework API

## Paired Multi-Arm Experiments (incl. A/B Experiments)

`fabricpc.experiments.ab_experiment`

### ExperimentArm

Defines one condition (arm) of an experiment. Shared by both runners below.

```python
from fabricpc.experiments import ExperimentArm
from fabricpc.training import train, evaluate

arm = ExperimentArm(
    name="muPC",
    model_factory=create_model,    # (rng_key) -> (params, structure)
    train_fn=train,
    eval_fn=evaluate,
    optimizer=optax.adamw(1e-3),
    train_config={"num_epochs": 5},
)
```

| Field | Type | Description |
|-------|------|-------------|
| `name` | `str` | Display name for this condition |
| `model_factory` | `Callable` | `(rng_key) -> (params, structure)` |
| `train_fn` | `Callable` | Training function with `train`'s positional prefix, returning a `TrainResult`; select the algorithm with `functools.partial(train, algorithm="backprop")` |
| `eval_fn` | `Callable` | Evaluation function with `evaluate`'s signature, returning a metrics dict |
| `optimizer` | `optax.GradientTransformation` | Optimizer |
| `train_config` | `dict` | Training configuration |

### PlannedMultiContrastExperiment

Runs N arms across paired trials with constructor-declared planned contrasts. Pairing holds by construction: within each trial, every arm receives freshly constructed loaders from `data_loader_factory(trial_seed)` and the same model RNG seed, so all arms see identical data and batch order whenever the factory is deterministic in its seed argument. Per-arm results are therefore independent of arm order and of which arm subset is run. Trial *i* uses seed `seed_offset + i * 1000`.

```python
from fabricpc.experiments import PlannedMultiContrastExperiment

runner = PlannedMultiContrastExperiment(
    arms=[arm_mlp, arm_1hopfield, arm_2hopfield],
    contrasts=[("1hopfield", "MLP"), ("2hopfield", "1hopfield")],
    metric="accuracy",
    data_loader_factory=make_loaders,   # (seed) -> (train_loader, test_loader)
    n_trials=10,
)

results = runner.run()
for c in results.contrast_results():
    print(c.arm_a, c.arm_b, c.mean_diff, c.p_value, c.cohens_d)

total = results.delta("2hopfield", "MLP")   # reported-only delta, no test
print(total.mean, total.se)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `arms` | `List[ExperimentArm]` | Arms to run, each once per trial; names must be unique |
| `contrasts` | `List[Tuple[str, str]]` | `(arm_a_name, arm_b_name)` pairs declaring the planned-contrast family; each name must reference an arm in `arms` |
| `metric` | `str` | Key in each arm's eval-result dict (e.g., `"accuracy"`) |
| `data_loader_factory` | `Callable` | `(seed) -> (train_loader, test_loader)`; must be deterministic in its seed, otherwise the arms are not paired |
| `n_trials` | `int` | Number of paired trials (default 10); the tests require ≥ 2 |
| `seed_offset` | `int` | Base seed offset (default 0) |
| `verbose` | `bool` | Forward `verbose=True` to each arm's `train_fn` |

**`run()` returns `PlannedMultiContrastResults`:**

- `contrast_results()` — one `ContrastResult` per declared contrast, in declaration order: `mean_diff`, `se_diff` (SE of the per-trial paired difference), `t_statistic`, `p_value`, `significant_at_05`, `cohens_d`, `n`. Each contrast is a two-sided paired t-test plus paired Cohen's d on the per-trial difference vector. With `n_trials < 2` the test statistics are NaN and `significant_at_05` is `False`.
- `delta(arm_a, arm_b)` — `DescriptiveDelta` (`mean`, `std`, `se`, `n`) for any arm pair. Carries no test statistics, so a reported-only delta cannot be misread as a planned contrast.
- `per_arm_metrics(name)` / `per_arm_times(name)` — per-trial metric values / training times as `np.ndarray`.

Full runnable version: `examples/storkey_hopfield_demo.py` (four arms, three planned contrasts, one reported-only delta).

### ABExperiment

Two-arm wrapper around `PlannedMultiContrastExperiment` with the single contrast `(arm_a.name, arm_b.name)`; same trial loop, same pairing. Returns `ABResults` with the legacy reporting API.

```python
from fabricpc.experiments import ABExperiment

experiment = ABExperiment(
    arm_a=arm_mupc,
    arm_b=arm_standard,
    metric="accuracy",
    data_loader_factory=lambda seed: (train_loader, test_loader),
    n_trials=5,
    verbose=False,
)

results = experiment.run()
results.print_summary()
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `arm_a` | `ExperimentArm` | First condition |
| `arm_b` | `ExperimentArm` | Second condition |
| `metric` | `str` | Metric key from eval_fn return dict (e.g., `"accuracy"`) |
| `data_loader_factory` | `Callable` | `(seed) -> (train_loader, test_loader)` |
| `n_trials` | `int` | Number of independent trials |
| `verbose` | `bool` | Print per-trial progress |

**Output** from `print_summary()`:
- Descriptive statistics (mean, std, min, max)
- Paired t-test (t-statistic, p-value)
- Cohen's d effect size
- Estimated N for significance

---

## Statistics

`fabricpc.experiments.statistics`

Standalone paired-analysis functions on numpy arrays; the runners above use them internally. Each returns a frozen dataclass.

```python
from fabricpc.experiments.statistics import (
    descriptive_stats,
    paired_ttest,
    cohens_d,
    estimate_required_n,
)

stats = descriptive_stats(a_vals)     # DescriptiveStats: mean, std, se, min, max, n
ttest = paired_ttest(a_vals, b_vals)  # PairedTestResult: t_statistic, p_value,
                                      #   mean_difference, significant_at_05, n
effect = cohens_d(a_vals, b_vals)     # EffectSize: d, magnitude
n_req = estimate_required_n(effect.d) # trials for p<0.05 at 80% power; 999999 when d ~ 0
```

`paired_ttest` raises `ValueError` when the arrays differ in length or hold fewer than 2 samples. `cohens_d` uses the standard deviation of the per-trial differences as the denominator (paired design); `magnitude` is one of `"negligible"`, `"small"`, `"medium"`, `"large"`.

---

## Bayesian Tuning

`fabricpc.tuning.BayesianTuner`

Optuna ships in the `[experiments]` extra, not the core install: `pip install "fabricpc[experiments]"`. Without it, `import fabricpc.tuning` raises `ModuleNotFoundError: optuna`.

Two-phase Optuna search for language-model hyperparameters, both phases minimizing validation perplexity. Phase 1 searches architecture and training parameters together, with Hyperband pruning that allocates training epochs as the trial resource: unpromising trials are stopped after few epochs while strong ones train longer. Phase 2 fixes the Phase 1 winning architecture and refines the continuous training parameters (`lr`, `eta_infer`, `infer_steps`) with a multivariate TPE sampler that models correlations between them. Training energy serves only as a divergence guard: a trial is pruned when its energy becomes non-finite or rises above its best epoch by more than `divergence_rel_tol`.

The perplexity objective requires the trial graph's target node to use `CrossEntropyEnergy` — `evaluate` reports `perplexity` only then, and the tuner raises on a graph without it rather than scoring the trial silently. `algorithm=` selects the learning algorithm for every trial's `train`/`evaluate` call (`"pc"` default, `"backprop"` supported).

```python
from fabricpc.tuning import BayesianTuner

def phase1_search_space(trial):
    return {
        "embed_dim": trial.suggest_categorical("embed_dim", [64, 128]),
        "num_heads": trial.suggest_categorical("num_heads", [4, 8]),
        "depth": trial.suggest_int("depth", 1, 4),
        "lr": trial.suggest_float("lr", 1e-5, 3e-4, log=True),
        "eta_infer": trial.suggest_float("eta_infer", 0.01, 0.15),
    }

def phase2_search_space(trial, best_params):
    lr = best_params["lr"]
    return {
        "lr": trial.suggest_float("lr", lr * 0.5, lr * 2.0, log=True),
        "eta_infer": trial.suggest_float("eta_infer", 0.01, 0.2),
    }

tuner = BayesianTuner(
    train_loader=train_loader,
    val_loader=val_loader,
    trial_model=trial_model,           # (config, rng_key) -> (params, structure)
    base_config={"num_epochs": 5, "seq_len": 128, "vocab_size": vocab_size},
    study_name="transformer_v2_tuning",
)

results = tuner.tune(
    phase1_search_space=phase1_search_space,
    phase2_search_space=phase2_search_space,
    n_trials_phase1=20,
    n_trials_phase2=15,
)
print(results["phase2_best_ppl"], results["final_params"])
```

Full runnable version: `examples/transformer_tuning.py`.

**Constructor:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `train_loader` | iterable | required | Training batches (int32 token-id pairs, see [Data Loaders](14_api_data.md)) |
| `val_loader` | iterable | required | Validation batches for the perplexity objective |
| `trial_model` | `Callable` | required | `(config, rng_key) -> (params, structure)`; may instead return a 4-tuple `(params, structure, train_loader, val_loader)` for trials that rebuild loaders per architecture (e.g. `seq_len` or `batch_size` in the search space) |
| `base_config` | `dict` | required | Fixed config merged under every trial's sampled parameters |
| `study_name` | `str` | `"fabricpc_tuning"` | Optuna study name |
| `storage` | Optuna storage | `None` | Persistent trial storage; `None` keeps trials in memory |
| `log_file` | `str` | `"tuning_results.txt"` | Per-trial results log |
| `divergence_rel_tol` | `float` | `0.5` | Relative energy rise over the trial's best epoch that triggers pruning |
| `verbose` | `bool` | `False` | Print per-epoch trial progress |
| `algorithm` | `str` | `"pc"` | Learning algorithm passed to every trial's `train`/`evaluate` |

**Methods:**

- `tune_phase1(n_trials, search_space)` — `search_space: (trial) -> dict` of sampled parameters. Returns the Optuna `Study`.
- `tune_phase2(n_trials, best_params, search_space)` — `search_space: (trial, best_params) -> dict`, sampling around the Phase 1 winners. Returns the Optuna `Study`.
- `tune(phase1_search_space, phase2_search_space, n_trials_phase1=30, n_trials_phase2=20, save_best_to="tuning/best_hyperparameters.txt")` — runs both phases. Returns `{"phase1_best_ppl", "phase1_best_params", "phase2_best_ppl", "phase2_best_params", "final_params"}`; when Phase 2 completes no trial, the dict carries the Phase 1 keys only.
