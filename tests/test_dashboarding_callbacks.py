"""Tests for the dashboarding callback factories, driven through real train.

A duck-typed stub stands in for AimExperimentTracker so the tests need no
Aim install: it holds a real TrackingConfig and records every batch-level
call, and the iteration callback's behavior is checked against the config
that is supposed to decide it.
"""

import jax
import optax
import pytest

from conftest import ListLoader, make_classification_structure, with_inference
from fabricpc.core.types import GraphState
from fabricpc.graph_initialization import initialize_params
from fabricpc.training import EpochContext, train
from fabricpc.utils.dashboarding import (
    AimExperimentTracker,
    TrackingConfig,
    create_epoch_callback,
    create_iter_callback,
)


class StubTracker:
    def __init__(self, config):
        self.config = config
        self.calls = []

    def track_batch_energy(self, energy, epoch, batch, context=None):
        self.calls.append(("energy", epoch, batch, energy))

    def track_batch_energy_per_node(self, state, structure, epoch, batch):
        self.calls.append(("node_energy", epoch, batch, state))

    def track_weight_distributions(self, params, structure, epoch, batch, nodes=None):
        self.calls.append(("weights", epoch, batch, params, nodes))

    def track_state(self, state, epoch, batch, infer_step, nodes=None):
        self.calls.append(("state", epoch, batch, infer_step, state, nodes))

    def track_epoch_metrics(self, metrics, epoch, subset="val"):
        self.calls.append(("epoch_metrics", epoch, subset, metrics))

    def of(self, kind):
        return [c for c in self.calls if c[0] == kind]


def make_batches(rng_key, n_batches=2, batch_size=4):
    batches = []
    key = rng_key
    for _ in range(n_batches):
        kx, ky, key = jax.random.split(key, 3)
        x = jax.random.normal(kx, (batch_size, 6))
        y = jax.nn.one_hot(jax.random.randint(ky, (batch_size,), 0, 3), 3)
        batches.append({"x": x, "y": y})
    return ListLoader(batches)


def run(rng_key, config, *, algorithm="pc", infer_steps=3, n_batches=2):
    structure = with_inference(
        make_classification_structure(), eta_infer=0.05, infer_steps=infer_steps
    )
    params = initialize_params(structure, rng_key)
    stub = StubTracker(config)
    train(
        params,
        structure,
        make_batches(rng_key, n_batches),
        optax.adam(1e-2),
        {"num_epochs": 1},
        rng_key,
        algorithm=algorithm,
        verbose=False,
        iter_callback=create_iter_callback(stub),
    )
    return stub


def test_defaults_log_energy_and_weights_only(rng_key):
    stub = run(rng_key, TrackingConfig(tracking_every_n_batches=1))
    assert [(c[1], c[2]) for c in stub.of("energy")] == [(0, 0), (0, 1)]
    assert all(isinstance(c[3], float) for c in stub.of("energy"))
    assert [(c[1], c[2]) for c in stub.of("weights")] == [(0, 0), (0, 1)]
    assert all(c[4] is None for c in stub.of("weights"))  # empty -> every node
    # Per-node energy is handed the batch's GraphState; the tracker's own
    # nodes_to_track gate decides whether it logs anything.
    assert all(isinstance(c[3], GraphState) for c in stub.of("node_energy"))
    assert stub.of("state") == []


def test_track_state_reruns_inference_with_history_under_pc(rng_key):
    config = TrackingConfig(
        track_state=True,
        tracking_every_n_batches=1,
        state_tracking_every_n_infer_steps=1,
    )
    stub = run(rng_key, config, infer_steps=3)
    states = stub.of("state")
    # One record per inference step on every tracked batch.
    assert [(c[1], c[2], c[3]) for c in states] == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
    ]
    assert all(isinstance(c[4], GraphState) for c in states)
    assert all(c[5] is None for c in states)  # empty nodes_to_track -> all nodes


def test_track_state_distributions_implies_track_state(rng_key):
    config = TrackingConfig(
        track_state_distributions=True,
        tracking_every_n_batches=1,
        state_tracking_every_n_infer_steps=1,
    )
    assert config.tracks_state
    stub = run(rng_key, config, infer_steps=2)
    assert len(stub.of("state")) == 4


def test_track_state_logs_feedforward_state_once_under_backprop(rng_key):
    config = TrackingConfig(
        track_state=True, tracking_every_n_batches=1, nodes_to_track=["h"]
    )
    stub = run(rng_key, config, algorithm="backprop")
    states = stub.of("state")
    assert [(c[1], c[2], c[3]) for c in states] == [(0, 0, 0), (0, 1, 0)]
    assert all(isinstance(c[4], GraphState) for c in states)
    # nodes_to_track scopes both state and weights.
    assert all(c[5] == ["h"] for c in states)
    assert all(c[4] == ["h"] for c in stub.of("weights"))


def test_tracking_every_n_batches_gates_state(rng_key):
    config = TrackingConfig(
        track_state=True,
        tracking_every_n_batches=2,
        state_tracking_every_n_infer_steps=1,
    )
    stub = run(rng_key, config, infer_steps=2, n_batches=3)
    assert sorted({c[2] for c in stub.of("state")}) == [0, 2]


def test_epoch_callback_does_not_log_weights(rng_key):
    structure = make_classification_structure()
    stub = StubTracker(TrackingConfig())
    ctx = EpochContext(
        epoch_idx=0,
        step=1,
        params=None,
        opt_state=None,
        structure=structure,
        config={},
        algorithm="pc",
        rng_key=rng_key,
        epoch_key=rng_key,
        metrics={},
    )
    assert create_epoch_callback(stub, structure)(ctx) is None
    assert stub.calls == []


def test_tracker_track_state_is_noop_unless_configured():
    tracker = AimExperimentTracker(config=TrackingConfig())
    tracker._ensure_initialized = lambda: pytest.fail("track_state touched the run")
    tracker.track_state(state=None, epoch=0, batch=0, infer_step=0)
