"""
Tests for ``fabricpc.training.RegimeProbe`` on a CPU tanh MLP.

The probe is driven two ways. ``TestProbeRows`` feeds it contexts built
from ``make_train_step`` directly, so the row logic, the CSV round trip,
and the backprop mode are pinned without the trainer. ``TestWithTrain``
runs it as ``train(..., iter_callback=probe.on_iter)`` and is skipped
until the trainer ships ``IterContext`` (the per-update context carrying
the parameters), which lands in its own PR.
"""

from typing import Any, NamedTuple, Optional

import jax
import numpy as np
import optax
import pytest

import fabricpc.training as training
from conftest import ListLoader
from fabricpc.core import EPCInference
from fabricpc.core.activations import SoftmaxActivation, TanhActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode
from fabricpc.training import (
    RegimeProbe,
    build_clamps,
    make_train_step,
    read_regime_csv,
    train,
)
from fabricpc.training.regime_probe import (
    DATA_COLUMNS,
    METADATA_COLUMNS,
    WNORM_PREFIX,
    weight_norm_columns,
)

HAS_ITER_CONTEXT = hasattr(training, "IterContext")

BATCH, N_BATCHES, EPOCHS, EVERY = 4, 3, 2, 2


def _mlp(inference=None):
    """x(4) -> h1(3, tanh) -> h2(3, tanh) -> y(2, softmax + cross-entropy)."""
    w_init = NormalInitializer(std=0.5)
    x = IdentityNode(shape=(4,), name="x")
    h1 = Linear(shape=(3,), name="h1", activation=TanhActivation(), weight_init=w_init)
    h2 = Linear(shape=(3,), name="h2", activation=TanhActivation(), weight_init=w_init)
    y = Linear(
        shape=(2,),
        name="y",
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        weight_init=w_init,
    )
    return graph(
        nodes=[x, h1, h2, y],
        edges=[
            Edge(source=x, target=h1.slot("in")),
            Edge(source=h1, target=h2.slot("in")),
            Edge(source=h2, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=inference or EPCInference(eta_infer=1e-3, infer_steps=5),
    )


def _batches(key):
    out = []
    for i in range(N_BATCHES):
        k = jax.random.fold_in(key, i)
        out.append(
            {
                "x": jax.random.normal(k, (BATCH, 4)),
                "y": jax.random.randint(jax.random.fold_in(k, 1), (BATCH,), 0, 2),
            }
        )
    return out


def _probe_clamps(structure, key):
    return build_clamps(
        _batches(jax.random.fold_in(key, 99))[0], structure, clamp_target=True
    )


class _Ctx(NamedTuple):
    """The fields of ``IterContext`` / ``EpochContext`` the probe reads."""

    epoch_idx: int
    step: int
    params: Any
    metrics: dict
    algorithm: str
    batch: Optional[dict] = None


def _drive(probe, structure, params, algorithm, key, accuracy=0.5):
    """Run ``make_train_step`` over the loader and feed the probe the same
    contexts ``train`` will build."""
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    step_fn = make_train_step(structure, optimizer, algorithm=algorithm)
    batches = _batches(key)
    step = 0
    for epoch in range(EPOCHS):
        sums = None
        for batch_idx, batch in enumerate(batches):
            batch_key = jax.random.fold_in(jax.random.fold_in(key, epoch), batch_idx)
            params, opt_state, metrics, _ = step_fn(params, opt_state, batch, batch_key)
            step += 1
            floats = {k: float(v) for k, v in metrics.items()}
            sums = floats if sums is None else {k: sums[k] + floats[k] for k in sums}
            probe.on_iter(
                _Ctx(
                    epoch_idx=epoch,
                    step=step,
                    params=params,
                    metrics=floats,
                    algorithm=algorithm,
                    batch=batch,
                )
            )
        means = {k: v / len(batches) for k, v in sums.items()}
        probe.on_epoch(
            _Ctx(
                epoch_idx=epoch,
                step=step,
                params=params,
                metrics=means,
                algorithm=algorithm,
            ),
            accuracy=accuracy,
        )
    return params


def _assert_rows(probe, structure, params, *, regime):
    n_probes = EPOCHS * N_BATCHES // EVERY
    assert len(probe.rows) == n_probes + EPOCHS
    probe_rows, epoch_rows = probe.probe_rows(), probe.epoch_rows()
    assert [r["update"] for r in probe_rows] == [
        EVERY * (i + 1) for i in range(n_probes)
    ]
    assert [r["epoch"] for r in epoch_rows] == list(range(EPOCHS))
    assert [r["update"] for r in epoch_rows] == [
        N_BATCHES * (e + 1) for e in range(EPOCHS)
    ]
    wnorm = weight_norm_columns(params)
    assert set(wnorm) == {
        f"{WNORM_PREFIX}x->h1:in",
        f"{WNORM_PREFIX}h1->h2:in",
        f"{WNORM_PREFIX}h2->y:in",
    }
    for row in probe_rows:
        assert row["lambda_max"] > 0 and np.isfinite(row["lambda_min"])
        assert 0.0 <= row["negative_weight"] <= 1.0
        assert row["train_energy"] > 0 and row["test_accuracy"] is None
        for column, (node, key) in wnorm.items():
            assert row[column] > 0
        if regime:
            assert isinstance(row["unstable"], bool) and isinstance(
                row["f_weighted"], float
            )
            assert row["eta_lambda_max"] == pytest.approx(1e-3 * row["lambda_max"])
            assert 0.0 < row["f_weighted"] < 0.1
        else:
            assert all(
                row[c] is None
                for c in ("f_weighted", "f_max", "unstable", "eta_lambda_max")
            )
    for row in epoch_rows:
        assert row["test_accuracy"] == 0.5 and row["lambda_max"] is None
    assert probe.first_crossing() is None and probe.first_reversal() is None
    assert probe.first_chance(0.5, margin=0.05) == 0
    assert probe.first_chance(0.25, margin=0.05) is None
    phases = probe.growth_phases()
    assert [e for e, _, _ in phases] == list(range(EPOCHS))
    assert phases[0][2] is None and phases[1][2] == pytest.approx(
        phases[1][1] / phases[0][1]
    )
    text = probe.summary(chance=0.5)
    assert "lambda_max per epoch" in text and "epoch   1" in text
    if regime:
        assert "never crossed" in text and "never flagged" in text
    else:
        assert "no ePC regime" in text


def _assert_round_trip(probe, tmp_path, *, regime):
    path = probe.write_csv(tmp_path / "track.csv")
    header = path.read_text().splitlines()[0].split(",")
    assert header[: len(METADATA_COLUMNS)] == list(METADATA_COLUMNS)
    assert header[
        len(METADATA_COLUMNS) : len(METADATA_COLUMNS) + len(DATA_COLUMNS)
    ] == list(DATA_COLUMNS)
    assert all(
        c.startswith(WNORM_PREFIX)
        for c in header[len(METADATA_COLUMNS) + len(DATA_COLUMNS) :]
    )
    metadata, rows = read_regime_csv(path)
    assert metadata == probe.metadata
    assert metadata["trainer"] == ("pc" if regime else "backprop")
    assert (metadata["eta_infer"], metadata["infer_steps"]) == (
        (1e-3, 5) if regime else (None, None)
    )
    assert len(rows) == len(probe.rows)
    for read, kept in zip(rows, probe.rows):
        for column, value in kept.items():
            assert read[column] == value, column


class TestProbeRows:
    def test_pc_fixed_probe_batch(self, rng_key, tmp_path):
        structure = _mlp()
        params = initialize_params(structure, rng_key)
        probe = RegimeProbe(
            structure,
            _probe_clamps(structure, rng_key),
            every=EVERY,
            iters=12,
            key=rng_key,
        )
        assert probe.inference is structure.config["inference"]
        params = _drive(probe, structure, params, "pc", rng_key)
        assert probe.metadata["probe_batch"] == BATCH
        _assert_rows(probe, structure, params, regime=True)
        _assert_round_trip(probe, tmp_path, regime=True)

    def test_pc_training_batch_and_csv_path(self, rng_key, tmp_path):
        """``probe_clamps=None`` measures on each probed training batch;
        ``csv_path`` is rewritten at every epoch."""
        structure = _mlp()
        params = initialize_params(structure, rng_key)
        target = tmp_path / "live.csv"
        probe = RegimeProbe(
            structure, None, every=EVERY, iters=12, key=rng_key, csv_path=target
        )
        params = _drive(probe, structure, params, "pc", rng_key)
        assert probe.metadata["probe_batch"] == "train"
        _assert_rows(probe, structure, params, regime=True)
        metadata, rows = read_regime_csv(target)
        assert metadata["probe_batch"] == "train" and len(rows) == len(probe.rows)

    def test_backprop_records_spectrum_without_regime(self, rng_key, tmp_path):
        structure = _mlp()
        params = initialize_params(structure, rng_key)
        probe = RegimeProbe(
            structure,
            _probe_clamps(structure, rng_key),
            every=EVERY,
            iters=12,
            key=rng_key,
        )
        params = _drive(probe, structure, params, "backprop", rng_key)
        _assert_rows(probe, structure, params, regime=False)
        _assert_round_trip(probe, tmp_path, regime=False)

    def test_explicit_inference_overrides_structure(self, rng_key):
        structure = _mlp()
        params = initialize_params(structure, rng_key)
        judged = EPCInference(eta_infer=0.5, infer_steps=1)
        probe = RegimeProbe(
            structure,
            _probe_clamps(structure, rng_key),
            every=EVERY,
            inference=judged,
            iters=12,
            key=rng_key,
        )
        _drive(probe, structure, params, "pc", rng_key)
        assert probe.metadata["eta_infer"] == 0.5 and probe.metadata["infer_steps"] == 1
        for row in probe.probe_rows():
            assert row["eta_lambda_max"] == pytest.approx(0.5 * row["lambda_max"])
            assert row["output_gradient_reverses"] == (
                0.5 * (row["lambda_max"] - 1.0) > 1.0
            )

    def test_every_must_be_positive(self, rng_key):
        with pytest.raises(ValueError):
            RegimeProbe(_mlp(), None, every=0, key=rng_key)

    def test_write_csv_without_path_raises(self, rng_key):
        with pytest.raises(ValueError):
            RegimeProbe(_mlp(), None, every=1, key=rng_key).write_csv()


@pytest.mark.skipif(
    not HAS_ITER_CONTEXT,
    reason="train(iter_callback=...) with IterContext not shipped yet",
)
class TestWithTrain:
    @pytest.mark.parametrize("algorithm", ["pc", "backprop"])
    def test_probe_as_train_callbacks(self, rng_key, tmp_path, algorithm):
        structure = _mlp()
        params = initialize_params(structure, rng_key)
        probe = RegimeProbe(
            structure,
            _probe_clamps(structure, rng_key),
            every=EVERY,
            iters=12,
            key=rng_key,
        )
        result = train(
            params,
            structure,
            ListLoader(_batches(rng_key)),
            optax.adam(1e-3),
            {"num_epochs": EPOCHS},
            rng_key,
            algorithm=algorithm,
            verbose=False,
            iter_callback=probe.on_iter,
            epoch_callback=lambda ctx: probe.on_epoch(ctx, 0.5),
        )
        assert result.step == EPOCHS * N_BATCHES
        assert all(isinstance(m, dict) for epoch in result.iter_results for m in epoch)
        _assert_rows(probe, structure, result.params, regime=(algorithm == "pc"))
        _assert_round_trip(probe, tmp_path, regime=(algorithm == "pc"))
