"""A/B experiment harness for comparing training methods and architectures.

Provides reusable framework for running paired statistical comparisons
across two or more training configurations (different architectures, different
training algorithms, or both).

Two public runners are provided:

- :class:`PlannedMultiContrastExperiment` — N arms with constructor-declared
  planned contrasts. The single source of truth for the per-trial training
  loop. Within each trial, every arm gets freshly constructed loaders from
  ``data_loader_factory(trial_seed)``, so every arm sees the same minibatch
  stream — paired in data subsample/noise AND in batch order — provided the
  factory is deterministic in its seed argument.
- :class:`ABExperiment` — a thin 2-arm wrapper that delegates entirely to
  ``PlannedMultiContrastExperiment``. Preserves the historical
  ``arm_a``/``arm_b`` / ``ABResults`` API.

Example (multi-arm, planned-contrast)::

    from fabricpc.experiments import (
        ExperimentArm, PlannedMultiContrastExperiment,
    )

    runner = PlannedMultiContrastExperiment(
        arms=[arm_mlp, arm_1hopfield, arm_2hopfield],
        contrasts=[("1hopfield", "MLP"), ("2hopfield", "1hopfield")],
        metric="accuracy",
        data_loader_factory=make_loaders,
        n_trials=10,
    )
    results = runner.run()
    for c in results.contrast_results():
        print(c.arm_a, c.arm_b, c.mean_diff, c.p_value, c.cohens_d)
    # Reported-only delta (descriptive, not in the contrast family):
    total = results.delta("2hopfield", "MLP")
    print(total.mean, total.se)

Example (2-arm, legacy ABExperiment, unchanged from before)::

    from fabricpc.experiments import ExperimentArm, ABExperiment

    experiment = ABExperiment(
        arm_a=arm_lateral,
        arm_b=arm_mlp,
        metric="accuracy",
        data_loader_factory=make_loaders,
        n_trials=10,
    )
    results = experiment.run()
    results.print_summary()
"""

from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any, List
import time

import numpy as np
import jax
import optax

from fabricpc.experiments.statistics import (
    descriptive_stats,
    paired_ttest,
    cohens_d,
    estimate_required_n,
)

# Type aliases
ModelFactory = Callable[[jax.Array], Tuple[Any, Any]]  # rng_key -> (params, structure)
TrainFn = Callable  # (params, structure, loader, optimizer, config, rng_key, verbose=...) -> TrainResult
EvalFn = Callable  # (params, structure, loader, config, rng_key) -> dict
DataLoaderFactory = Callable[
    [int], Tuple[Any, Any]
]  # seed -> (train_loader, test_loader)


@dataclass(frozen=True)
class ExperimentArm:
    """One arm (condition) of an experiment.

    Args:
        name: Human-readable name for this arm (e.g., "PC-sigmoid"). Must be
            unique within a single experiment's arms list.
        model_factory: Callable taking a JAX rng_key and returning
            (GraphParams, GraphStructure). Called fresh each trial.
        train_fn: Training function with fabricpc.training.train's positional
            prefix, returning a TrainResult. Select the algorithm with
            functools.partial(train, algorithm="backprop") — the arm stays
            trainer-agnostic.
        eval_fn: Evaluation function with fabricpc.training.evaluate's
            signature, returning a metrics dict.
        optimizer: Optax optimizer (e.g., optax.adam(1e-3)).
        train_config: Training configuration dict (scalar hyperparams only).
    """

    name: str
    model_factory: ModelFactory
    train_fn: TrainFn
    eval_fn: EvalFn
    optimizer: optax.GradientTransformation
    train_config: dict


@dataclass
class TrialResult:
    """Result of a single trial for one arm."""

    metric_value: float
    train_time: float
    all_metrics: Dict[str, float]


# ---------------------------------------------------------------------------
# Multi-arm, planned-contrast runner (single source of truth)
# ---------------------------------------------------------------------------


@dataclass
class ContrastResult:
    """Statistics for one declared planned contrast (``arm_a - arm_b``).

    The mean and SE are computed on the per-trial difference vector (paired
    design), so the SE is the SE of the paired difference (not the SE of an
    unpaired delta-of-means). The t-test and Cohen's d are likewise paired.
    """

    arm_a: str
    arm_b: str
    mean_diff: float
    se_diff: float
    t_statistic: float
    p_value: float
    significant_at_05: bool
    cohens_d: float
    n: int


@dataclass
class DescriptiveDelta:
    """Descriptive statistics for a reported-only paired delta.

    Returned by :meth:`PlannedMultiContrastResults.delta` for arm pairs that
    are NOT in the declared contrast family. Carries no test (no p-value, no
    significance flag, no Cohen's d) so it cannot be confused with a planned
    contrast at read sites.
    """

    arm_a: str
    arm_b: str
    mean: float
    std: float
    se: float
    n: int


@dataclass
class PlannedMultiContrastResults:
    """Results from a completed :class:`PlannedMultiContrastExperiment`.

    Carries per-arm per-trial metric values plus one :class:`ContrastResult`
    for each contrast declared at construction. Reported-only deltas (any arm
    pair NOT in the declared contrast family) are available via
    :meth:`delta`, which returns descriptive statistics with no test attached.
    """

    arm_names: List[str]
    contrasts: List[Tuple[str, str]]
    metric: str
    n_trials: int
    per_arm_trials: Dict[str, List[TrialResult]]
    seeds: List[int]
    total_time: float
    num_epochs: int

    def per_arm_metrics(self, arm_name: str) -> np.ndarray:
        """Per-trial metric values for one arm (length n_trials)."""
        if arm_name not in self.per_arm_trials:
            raise KeyError(
                f"Unknown arm name {arm_name!r}; "
                f"available: {sorted(self.per_arm_trials.keys())}"
            )
        return np.array([t.metric_value for t in self.per_arm_trials[arm_name]])

    def per_arm_times(self, arm_name: str) -> np.ndarray:
        if arm_name not in self.per_arm_trials:
            raise KeyError(
                f"Unknown arm name {arm_name!r}; "
                f"available: {sorted(self.per_arm_trials.keys())}"
            )
        return np.array([t.train_time for t in self.per_arm_trials[arm_name]])

    def contrast_results(self) -> List[ContrastResult]:
        """One :class:`ContrastResult` per declared contrast, in declaration
        order. Each contrast is a two-sided paired t-test on the per-trial
        difference vector, plus paired Cohen's d. With fewer than 2 trials the
        test statistics are NaN and ``significant_at_05`` is False (a paired
        test needs at least 2 differences), so this method is total in
        ``n_trials``."""
        out: List[ContrastResult] = []
        for a, b in self.contrasts:
            a_vals = self.per_arm_metrics(a)
            b_vals = self.per_arm_metrics(b)
            diff = a_vals - b_vals
            n = len(diff)
            mean_diff = float(np.mean(diff))
            if n > 1:
                ttest = paired_ttest(a_vals, b_vals)
                effect = cohens_d(a_vals, b_vals)
                se_diff = float(np.std(diff, ddof=1) / np.sqrt(n))
                t_statistic = ttest.t_statistic
                p_value = ttest.p_value
                significant = ttest.significant_at_05
                d = effect.d
            else:
                se_diff = 0.0
                t_statistic = float("nan")
                p_value = float("nan")
                significant = False
                d = float("nan")
            out.append(
                ContrastResult(
                    arm_a=a,
                    arm_b=b,
                    mean_diff=mean_diff,
                    se_diff=se_diff,
                    t_statistic=t_statistic,
                    p_value=p_value,
                    significant_at_05=significant,
                    cohens_d=d,
                    n=n,
                )
            )
        return out

    def delta(self, arm_a: str, arm_b: str) -> DescriptiveDelta:
        """Descriptive statistics for ``arm_a - arm_b`` over per-trial
        differences. No t-test, no Cohen's d — use this for reported-only
        deltas that are NOT in the declared planned-contrast family. The
        deliberately narrower return type makes it impossible to misread a
        reported-only delta as a tested contrast."""
        a_vals = self.per_arm_metrics(arm_a)
        b_vals = self.per_arm_metrics(arm_b)
        diff = a_vals - b_vals
        n = len(diff)
        mean = float(np.mean(diff))
        std = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 1 else 0.0
        return DescriptiveDelta(
            arm_a=arm_a, arm_b=arm_b, mean=mean, std=std, se=se, n=n
        )


class PlannedMultiContrastExperiment:
    """Runner for paired N-arm experiments with constructor-declared contrasts.

    Within each trial, every arm gets freshly constructed loaders from
    ``data_loader_factory(trial_seed)``, trains a fresh model, and is
    evaluated. Because the factory receives the same seed for every arm of a
    trial, all arms see identical data and identical batch order whenever the
    factory is deterministic in its seed — the pairing holds by construction,
    with no requirements on the loader objects themselves.

    The shared per-trial seed is also used as the model RNG seed for every
    arm, so the source of model-init randomness is paired across arms (the
    architectures differ, so the resulting weights differ; only the seed is
    shared).

    Args:
        arms: Arms to run, each evaluated once per trial.
        contrasts: List of ``(arm_a_name, arm_b_name)`` pairs declaring the
            planned-contrast family. Each contrast is a two-sided paired
            t-test on the per-trial difference vector. Every name must
            reference an arm in ``arms``.
        metric: Key in each arm's eval-result dict to compare
            (e.g. ``"accuracy"``, ``"perplexity"``).
        data_loader_factory: Callable ``seed -> (train_loader, test_loader)``.
            Called once per arm within each trial, always with the trial's
            seed. Must be deterministic in its seed argument: two calls with
            the same seed must yield loaders producing identical batch
            streams, otherwise the arms are not paired.
        n_trials: Number of independent paired trials. Must be >= 1; the
            paired test statistics require >= 2 (a single trial yields per-arm
            metrics and NaN test statistics).
        seed_offset: Base seed offset. Trial i uses
            ``seed = seed_offset + i * 1000``.
        verbose: If True, forward verbose=True to each arm's train_fn.
    """

    def __init__(
        self,
        arms: List[ExperimentArm],
        contrasts: List[Tuple[str, str]],
        metric: str,
        data_loader_factory: DataLoaderFactory,
        n_trials: int = 10,
        seed_offset: int = 0,
        verbose: bool = False,
    ):
        if len(arms) < 1:
            raise ValueError("arms must contain at least one ExperimentArm.")

        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1; got {n_trials}.")

        arm_names = [a.name for a in arms]
        if len(set(arm_names)) != len(arm_names):
            duplicates = sorted({n for n in arm_names if arm_names.count(n) > 1})
            raise ValueError(
                f"Arm names must be unique within an experiment. Duplicates: {duplicates}"
            )

        name_set = set(arm_names)
        for a, b in contrasts:
            if a not in name_set:
                raise ValueError(
                    f"Contrast references unknown arm {a!r}; "
                    f"available arms: {sorted(name_set)}"
                )
            if b not in name_set:
                raise ValueError(
                    f"Contrast references unknown arm {b!r}; "
                    f"available arms: {sorted(name_set)}"
                )

        self.arms = list(arms)
        self.contrasts = list(contrasts)
        self.metric = metric
        self.data_loader_factory = data_loader_factory
        self.n_trials = n_trials
        self.seed_offset = seed_offset
        self.verbose = verbose

    def _run_arm_trial(
        self,
        arm: ExperimentArm,
        trial_seed: int,
        train_loader: Any,
        test_loader: Any,
    ) -> TrialResult:
        """Run a single trial for one arm on the given loaders."""
        master_key = jax.random.PRNGKey(trial_seed)
        graph_key, train_key, eval_key = jax.random.split(master_key, 3)

        params, structure = arm.model_factory(graph_key)

        t0 = time.time()
        train_result = arm.train_fn(
            params,
            structure,
            train_loader,
            arm.optimizer,
            arm.train_config,
            train_key,
            verbose=self.verbose,
        )
        trained_params = train_result.params
        train_time = time.time() - t0

        metrics = arm.eval_fn(
            trained_params,
            structure,
            test_loader,
            arm.train_config,
            eval_key,
        )

        if self.metric not in metrics:
            available = ", ".join(sorted(metrics.keys()))
            raise KeyError(
                f"Metric '{self.metric}' not found in eval results for arm "
                f"'{arm.name}'. Available: {available}"
            )

        return TrialResult(
            metric_value=float(metrics[self.metric]),
            train_time=train_time,
            all_metrics={k: float(v) for k, v in metrics.items()},
        )

    def run(self) -> PlannedMultiContrastResults:
        """Execute all trials for all arms. Returns
        :class:`PlannedMultiContrastResults`."""
        per_arm_trials: Dict[str, List[TrialResult]] = {a.name: [] for a in self.arms}
        seeds: List[int] = []

        num_epochs = next(
            (
                arm.train_config["num_epochs"]
                for arm in self.arms
                if "num_epochs" in arm.train_config
            ),
            1,
        )
        total_start = time.time()

        for trial_idx in range(self.n_trials):
            trial_seed = self.seed_offset + trial_idx * 1000
            seeds.append(trial_seed)

            print(f"--- Trial {trial_idx + 1}/{self.n_trials} (seed={trial_seed}) ---")

            for arm in self.arms:
                # Fresh loaders per arm from the same trial seed: every arm
                # sees the identical batch stream, so per-arm results are
                # independent of arm order and of arm-subset selection.
                train_loader, test_loader = self.data_loader_factory(trial_seed)
                result = self._run_arm_trial(arm, trial_seed, train_loader, test_loader)
                per_arm_trials[arm.name].append(result)
                print(
                    f"  {arm.name}: {self.metric}={result.metric_value:.4f}  "
                    f"(train: {result.train_time:.1f}s)"
                )

        total_time = time.time() - total_start

        return PlannedMultiContrastResults(
            arm_names=[a.name for a in self.arms],
            contrasts=list(self.contrasts),
            metric=self.metric,
            n_trials=self.n_trials,
            per_arm_trials=per_arm_trials,
            seeds=seeds,
            total_time=total_time,
            num_epochs=num_epochs,
        )


# ---------------------------------------------------------------------------
# 2-arm legacy wrapper
# ---------------------------------------------------------------------------


@dataclass
class ABResults:
    """Results from a completed 2-arm A/B experiment.

    Holds per-trial data for both arms and provides analysis and reporting
    via :meth:`print_summary`. Built by :meth:`ABExperiment.run` (which
    delegates to :class:`PlannedMultiContrastExperiment` internally — this
    class is the legacy view onto the same data).
    """

    arm_a_name: str
    arm_b_name: str
    metric: str
    n_trials: int
    arm_a_trials: List[TrialResult]
    arm_b_trials: List[TrialResult]
    seeds: List[int]
    total_time: float
    num_epochs: int

    @property
    def arm_a_metrics(self) -> np.ndarray:
        return np.array([t.metric_value for t in self.arm_a_trials])

    @property
    def arm_b_metrics(self) -> np.ndarray:
        return np.array([t.metric_value for t in self.arm_b_trials])

    @property
    def arm_a_times(self) -> np.ndarray:
        return np.array([t.train_time for t in self.arm_a_trials])

    @property
    def arm_b_times(self) -> np.ndarray:
        return np.array([t.train_time for t in self.arm_b_trials])

    def print_summary(self) -> None:
        """Print a complete ASCII summary of the experiment results."""
        a_vals = self.arm_a_metrics
        b_vals = self.arm_b_metrics

        # Detect rate metrics (0-1 range) for percentage display
        is_rate = (
            np.all(a_vals >= 0)
            and np.all(a_vals <= 1)
            and np.all(b_vals >= 0)
            and np.all(b_vals <= 1)
        )
        scale = 100.0 if is_rate else 1.0
        pct = "%" if is_rate else ""

        print("=" * 70)
        print(f"A/B Experiment: {self.arm_a_name} vs {self.arm_b_name}")
        print("=" * 70)
        # Training time comparison
        a_epoch_times = self.arm_a_times / self.num_epochs
        b_epoch_times = self.arm_b_times / self.num_epochs
        a_t = descriptive_stats(a_epoch_times)
        b_t = descriptive_stats(b_epoch_times)

        print()
        print("--- Training Time per Epoch ---")
        print(f"{self.arm_a_name}: {a_t.mean:.3f} +/- {a_t.se:.3f}s")
        print(f"{self.arm_b_name}: {b_t.mean:.3f} +/- {b_t.se:.3f}s")
        if b_t.mean > 0:
            print(
                f"Ratio: {self.arm_a_name} is "
                f"{a_t.mean / b_t.mean:.2f}x {self.arm_b_name} time"
            )

        print()
        print(f"Total wall time: {self.total_time:.1f}s")
        print("=" * 70)

        print(f"Metric: {self.metric}")
        print(f"Trials: {self.n_trials}")
        print(f"Epochs per trial: {self.num_epochs}")
        print("Design: Paired (same seed per trial)")
        print()

        # Per-trial table
        col_a = f"{self.arm_a_name}{pct}"
        col_b = f"{self.arm_b_name}{pct}"
        col_d = f"Diff{pct}"
        header = f"{'Trial':<8} {'Seed':<8} {col_a:<20} {col_b:<20} {col_d:<12}"
        print(header)
        print("-" * len(header))
        for i in range(self.n_trials):
            diff = (a_vals[i] - b_vals[i]) * scale
            print(
                f"{i+1:<8} {self.seeds[i]:<8} "
                f"{a_vals[i]*scale:<20.2f} "
                f"{b_vals[i]*scale:<20.2f} "
                f"{diff:<+12.2f}"
            )
        print("-" * len(header))

        # Descriptive stats
        a_stats = descriptive_stats(a_vals * scale)
        b_stats = descriptive_stats(b_vals * scale)
        print()
        print(
            f"{self.arm_a_name}: {a_stats.mean:.2f} +/- {a_stats.se:.2f}{pct}"
            f"  (mean +/- SE, SD={a_stats.std:.2f}{pct})"
        )
        print(
            f"{self.arm_b_name}: {b_stats.mean:.2f} +/- {b_stats.se:.2f}{pct}"
            f"  (mean +/- SE, SD={b_stats.std:.2f}{pct})"
        )

        # Statistical tests (require n >= 2)
        if self.n_trials >= 2:
            ttest = paired_ttest(a_vals, b_vals)
            print()
            print("--- Paired t-test ---")
            print(
                f"Mean difference ({self.arm_a_name} - {self.arm_b_name}): "
                f"{ttest.mean_difference * scale:+.2f}{pct}"
            )
            print(f"t-statistic: {ttest.t_statistic:.4f}")
            print(f"p-value: {ttest.p_value:.4f}, N = {ttest.n}")
            print(
                f"Significant at p<0.05: "
                f"{'YES' if ttest.significant_at_05 else 'NO'}"
            )

            effect = cohens_d(a_vals, b_vals)
            print()
            print("--- Effect Size ---")
            print(f"Cohen's d (paired): {effect.d:.4f}")
            print(f"Interpretation: {effect.magnitude}")

            req_n = estimate_required_n(effect.d)
            print()
            print("--- Power Analysis ---")
            print(f"Estimated trials needed for p<0.05 with 80% power: {req_n}")
            if req_n >= 999999:
                print("  -> Effect size is ~zero; no finite sample can detect it.")
            elif req_n <= self.n_trials:
                print(f"  -> Current n_trials ({self.n_trials}) is sufficient.")
            else:
                print(
                    f"  -> Current n_trials ({self.n_trials}) may be underpowered. "
                    f"Consider increasing to {req_n}."
                )
        else:
            print()
            print("--- Statistical tests require n_trials >= 2 ---")
            diff = float(np.mean(a_vals - b_vals))
            print(
                f"Mean difference ({self.arm_a_name} - {self.arm_b_name}): "
                f"{diff * scale:+.2f}{pct}"
            )


class ABExperiment:
    """Paired 2-arm experiment runner — thin wrapper around
    :class:`PlannedMultiContrastExperiment`.

    Preserved for backward compatibility. The single declared contrast is
    ``(arm_a.name, arm_b.name)`` and the trial loop is the multi-arm
    runner's. ``ABResults`` is reconstructed from the multi-arm results so
    the legacy API surface is identical.

    Args:
        arm_a: First experimental condition.
        arm_b: Second experimental condition.
        metric: Key in eval_fn's return dict to compare.
        data_loader_factory: Callable taking an int seed and returning
            (train_loader, test_loader).
        n_trials: Number of independent paired trials.
        seed_offset: Base seed offset. Trial i uses
            ``seed = seed_offset + i * 1000``.
        verbose: If True, forward verbose=True to each arm's train_fn.
    """

    def __init__(
        self,
        arm_a: ExperimentArm,
        arm_b: ExperimentArm,
        metric: str,
        data_loader_factory: DataLoaderFactory,
        n_trials: int = 10,
        seed_offset: int = 0,
        verbose: bool = False,
    ):
        self.arm_a = arm_a
        self.arm_b = arm_b
        self.metric = metric
        self.data_loader_factory = data_loader_factory
        self.n_trials = n_trials
        self.seed_offset = seed_offset
        self.verbose = verbose

    def run(self) -> ABResults:
        """Run the experiment and return :class:`ABResults`. Internally
        delegates to :class:`PlannedMultiContrastExperiment` so there is one
        trial-loop implementation across both runners."""
        runner = PlannedMultiContrastExperiment(
            arms=[self.arm_a, self.arm_b],
            contrasts=[(self.arm_a.name, self.arm_b.name)],
            metric=self.metric,
            data_loader_factory=self.data_loader_factory,
            n_trials=self.n_trials,
            seed_offset=self.seed_offset,
            verbose=self.verbose,
        )
        multi = runner.run()

        return ABResults(
            arm_a_name=self.arm_a.name,
            arm_b_name=self.arm_b.name,
            metric=self.metric,
            n_trials=self.n_trials,
            arm_a_trials=multi.per_arm_trials[self.arm_a.name],
            arm_b_trials=multi.per_arm_trials[self.arm_b.name],
            seeds=multi.seeds,
            total_time=multi.total_time,
            num_epochs=multi.num_epochs,
        )
