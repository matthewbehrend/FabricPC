"""Tests for the paired experiment runners in fabricpc.experiments.

The pairing guarantee is the core invariant: within a trial, every arm must
consume the identical batch stream. The runner achieves this by calling
data_loader_factory(trial_seed) once per arm, so the tests use a
deliberately STATEFUL stub loader (epoch-dependent shuffle, like
FewShotLoader) — if the runner ever goes back to sharing loader instances
across arms, these tests fail.

No TFDS, no real training: train/eval are stubs that fingerprint the batch
stream they receive. FewShotLoader tests monkeypatch tfds.load and are
gated on the tensorflow/tensorflow_datasets imports.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from fabricpc.experiments import (
    ExperimentArm,
    TrialResult,
    ABExperiment,
    PlannedMultiContrastExperiment,
    PlannedMultiContrastResults,
)
from fabricpc.training import TrainResult

# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class StatefulStubLoader:
    """Loader whose batch order depends on how often it has been iterated.

    Mimics the epoch statefulness of FewShotLoader: each pass shuffles with
    a seed derived from (seed, epoch counter) and advances the counter. Two
    fresh instances with the same seed yield identical streams; one shared
    instance yields a different stream on every pass.
    """

    def __init__(self, seed: int, n_batches: int = 6):
        self.seed = seed
        self.n_batches = n_batches
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        order = rng.permutation(self.n_batches)
        for i in order:
            x = np.full((2, 3), float(i), dtype=np.float32)
            y = np.zeros((2, 10), dtype=np.float32)
            y[:, int(i) % 10] = 1.0
            yield x, y

    def __len__(self):
        return self.n_batches


def stub_factory(seed):
    return StatefulStubLoader(seed), StatefulStubLoader(seed + 1)


def _stream_fingerprint(loader, num_epochs):
    """Order-sensitive fingerprint of the multi-epoch batch stream."""
    values = []
    for _ in range(num_epochs):
        for x, _ in loader:
            values.append(float(x[0, 0]))
    v = np.asarray(values)
    return float(np.dot(v, np.arange(1, len(v) + 1)))


def stub_train(params, structure, train_loader, optimizer, config, key, verbose=False):
    fp = _stream_fingerprint(train_loader, config.get("num_epochs", 1))
    return TrainResult(
        params={"train_fingerprint": fp},
        opt_state=None,
        step=0,
        iter_results=[],
        epoch_results=[],
    )


def stub_eval(trained_params, structure, test_loader, config, key):
    test_fp = _stream_fingerprint(test_loader, 1)
    # Depends on both the training stream and the test stream, so any
    # cross-arm stream divergence shows up in the metric.
    return {"accuracy": trained_params["train_fingerprint"] + test_fp}


def make_arm(name, num_epochs=2, eval_fn=stub_eval):
    return ExperimentArm(
        name=name,
        model_factory=lambda rng_key: ({}, None),
        train_fn=stub_train,
        eval_fn=eval_fn,
        optimizer=None,
        train_config={"num_epochs": num_epochs},
    )


def run_experiment(arm_names, n_trials=3, contrasts=None):
    runner = PlannedMultiContrastExperiment(
        arms=[make_arm(n) for n in arm_names],
        contrasts=contrasts or [],
        metric="accuracy",
        data_loader_factory=stub_factory,
        n_trials=n_trials,
    )
    return runner.run()


# ---------------------------------------------------------------------------
# Pairing invariant
# ---------------------------------------------------------------------------


def test_arms_see_identical_batch_stream():
    """All arms of a trial consume the identical multi-epoch stream, even
    though the loaders are stateful across passes."""
    results = run_experiment(["A", "B", "C"])
    a = results.per_arm_metrics("A")
    b = results.per_arm_metrics("B")
    c = results.per_arm_metrics("C")
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a, c)


def test_trials_differ():
    """Different trials use different seeds, so the streams (and metrics)
    differ across trials — the pairing is within-trial, not global."""
    a = run_experiment(["A", "B"]).per_arm_metrics("A")
    assert len(set(a.tolist())) == len(a)


def test_arm_order_and_subset_independence():
    """An arm's per-trial metrics do not depend on its position in the arms
    list or on which other arms run (regression test for the shared-loader
    pairing bug)."""
    b_first = run_experiment(["B", "A"]).per_arm_metrics("B")
    b_second = run_experiment(["A", "B"]).per_arm_metrics("B")
    b_alone = run_experiment(["B"]).per_arm_metrics("B")
    np.testing.assert_array_equal(b_first, b_second)
    np.testing.assert_array_equal(b_first, b_alone)


# ---------------------------------------------------------------------------
# Contrast statistics
# ---------------------------------------------------------------------------


def _results_from_metrics(per_arm_values, contrasts):
    """Build a results object directly from per-arm metric vectors."""
    per_arm_trials = {
        name: [
            TrialResult(metric_value=v, train_time=0.0, all_metrics={"accuracy": v})
            for v in values
        ]
        for name, values in per_arm_values.items()
    }
    n_trials = len(next(iter(per_arm_values.values())))
    return PlannedMultiContrastResults(
        arm_names=list(per_arm_values.keys()),
        contrasts=contrasts,
        metric="accuracy",
        n_trials=n_trials,
        per_arm_trials=per_arm_trials,
        seeds=list(range(n_trials)),
        total_time=0.0,
        num_epochs=1,
    )


def test_contrast_results_math():
    a = [0.8, 0.9, 0.7, 0.85]
    b = [0.7, 0.75, 0.72, 0.74]
    results = _results_from_metrics({"A": a, "B": b}, contrasts=[("A", "B")])
    (c,) = results.contrast_results()

    diff = np.array(a) - np.array(b)
    assert c.n == 4
    assert c.mean_diff == pytest.approx(float(np.mean(diff)))
    assert c.se_diff == pytest.approx(float(np.std(diff, ddof=1) / np.sqrt(4)))
    assert c.cohens_d == pytest.approx(float(np.mean(diff) / np.std(diff, ddof=1)))
    from scipy import stats

    t_ref, p_ref = stats.ttest_rel(a, b)
    assert c.t_statistic == pytest.approx(float(t_ref))
    assert c.p_value == pytest.approx(float(p_ref))
    assert c.significant_at_05 == (p_ref < 0.05)


def test_descriptive_delta_carries_no_test():
    a = [0.8, 0.9]
    b = [0.7, 0.75]
    results = _results_from_metrics({"A": a, "B": b}, contrasts=[])
    d = results.delta("A", "B")
    diff = np.array(a) - np.array(b)
    assert d.mean == pytest.approx(float(np.mean(diff)))
    assert d.std == pytest.approx(float(np.std(diff, ddof=1)))
    assert d.se == pytest.approx(float(np.std(diff, ddof=1) / np.sqrt(2)))
    assert not hasattr(d, "p_value")
    assert not hasattr(d, "significant_at_05")
    assert not hasattr(d, "cohens_d")


def test_single_trial_yields_nan_statistics():
    """contrast_results() is total in n_trials: one trial gives NaN test
    statistics instead of raising after the training compute is spent."""
    results = _results_from_metrics({"A": [0.8], "B": [0.7]}, contrasts=[("A", "B")])
    (c,) = results.contrast_results()
    assert c.n == 1
    assert c.mean_diff == pytest.approx(0.1)
    assert np.isnan(c.t_statistic)
    assert np.isnan(c.p_value)
    assert np.isnan(c.cohens_d)
    assert c.significant_at_05 is False


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_duplicate_arm_names_rejected():
    with pytest.raises(ValueError, match="unique"):
        PlannedMultiContrastExperiment(
            arms=[make_arm("A"), make_arm("A")],
            contrasts=[],
            metric="accuracy",
            data_loader_factory=stub_factory,
        )


def test_unknown_contrast_arm_rejected():
    with pytest.raises(ValueError, match="unknown arm"):
        PlannedMultiContrastExperiment(
            arms=[make_arm("A"), make_arm("B")],
            contrasts=[("A", "C")],
            metric="accuracy",
            data_loader_factory=stub_factory,
        )


def test_nonpositive_n_trials_rejected():
    with pytest.raises(ValueError, match="n_trials"):
        PlannedMultiContrastExperiment(
            arms=[make_arm("A")],
            contrasts=[],
            metric="accuracy",
            data_loader_factory=stub_factory,
            n_trials=0,
        )


def test_missing_metric_raises_keyerror():
    def bad_eval(trained_params, structure, test_loader, config, key):
        return {"perplexity": 1.0}

    runner = PlannedMultiContrastExperiment(
        arms=[make_arm("A", eval_fn=bad_eval)],
        contrasts=[],
        metric="accuracy",
        data_loader_factory=stub_factory,
        n_trials=2,
    )
    with pytest.raises(KeyError, match="perplexity"):
        runner.run()


# ---------------------------------------------------------------------------
# ABExperiment delegation
# ---------------------------------------------------------------------------


def test_ab_experiment_matches_multi_contrast_runner():
    ab = ABExperiment(
        arm_a=make_arm("A"),
        arm_b=make_arm("B"),
        metric="accuracy",
        data_loader_factory=stub_factory,
        n_trials=3,
    )
    ab_results = ab.run()
    multi = run_experiment(["A", "B"], n_trials=3, contrasts=[("A", "B")])

    np.testing.assert_array_equal(ab_results.arm_a_metrics, multi.per_arm_metrics("A"))
    np.testing.assert_array_equal(ab_results.arm_b_metrics, multi.per_arm_metrics("B"))
    assert ab_results.seeds == multi.seeds
    assert ab_results.num_epochs == multi.num_epochs


# ---------------------------------------------------------------------------
# FewShotLoader: cache, remainder batch, seed determinism (tfds-gated,
# offline via monkeypatched tfds.load)
# ---------------------------------------------------------------------------


class _FakeTensor:
    def __init__(self, value):
        self._value = value

    def numpy(self):
        return self._value


@pytest.fixture
def fewshot_env(monkeypatch):
    pytest.importorskip("tensorflow")
    tfds = pytest.importorskip("tensorflow_datasets")
    from fabricpc.utils.data.dataloader import FewShotLoader

    rng = np.random.default_rng(0)
    n_per_class, num_classes = 20, 3
    records = []
    for c in range(num_classes):
        for _ in range(n_per_class):
            img = rng.integers(0, 256, size=(4, 4, 1), dtype=np.uint8)
            records.append((_FakeTensor(img), _FakeTensor(np.int64(c))))

    load_calls = []

    def fake_load(name, split, as_supervised):
        load_calls.append((name, split))
        return records

    monkeypatch.setattr(tfds, "load", fake_load)
    monkeypatch.setattr(FewShotLoader, "_raw_split_cache", {})  # isolate cache per test
    return FewShotLoader, load_calls, num_classes


def _make_fewshot(cls, num_classes, k=5, batch_size=4, seed=123):
    return cls(
        dataset_name="fake_ds",
        split="train",
        k_per_class=k,
        batch_size=batch_size,
        num_classes=num_classes,
        shuffle=True,
        seed=seed,
        tensor_format="flat",
    )


def test_fewshot_raw_split_loaded_once(fewshot_env):
    cls, load_calls, num_classes = fewshot_env
    first = _make_fewshot(cls, num_classes, seed=1)
    second = _make_fewshot(cls, num_classes, seed=1)
    assert load_calls == [("fake_ds", "train")]
    np.testing.assert_array_equal(first.images, second.images)
    np.testing.assert_array_equal(first.labels, second.labels)


def test_fewshot_yields_remainder_batch(fewshot_env):
    cls, _, num_classes = fewshot_env
    # 3 classes x 5 shots = 15 samples; batch 4 -> 3 full batches + one of 3.
    loader = _make_fewshot(cls, num_classes, k=5, batch_size=4)
    sizes = [x.shape[0] for x, _ in loader]
    assert sizes == [4, 4, 4, 3]
    assert len(loader) == 4
    assert sum(sizes) == loader.num_samples


def test_fewshot_same_seed_instances_pair(fewshot_env):
    """Two fresh same-seed instances yield identical multi-epoch streams —
    the foundation of the runner's factory-per-arm pairing."""
    cls, _, num_classes = fewshot_env
    first = _make_fewshot(cls, num_classes)
    second = _make_fewshot(cls, num_classes)
    for _ in range(3):  # three epochs; shuffle advances per pass
        for (x1, y1), (x2, y2) in zip(first, second):
            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_array_equal(y1, y2)
