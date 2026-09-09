"""Tests for the dashboarding callback factories, driven through real train.

A duck-typed stub stands in for AimExperimentTracker so the tests need no
Aim install: it holds a real TrackingConfig and records every batch-level
call. The iteration callback's behavior is checked against the config that
decides it, and the tracker's own node gating is checked with a fake Aim run.
"""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import optax
import pytest

from conftest import ListLoader, make_classification_structure, with_inference
from fabricpc.core.inference import run_inference
from fabricpc.core.types import GraphParams, GraphState
from fabricpc.graph_initialization import initialize_graph_state, initialize_params
from fabricpc.training import EpochContext, batch_size_of, build_clamps, train
from fabricpc.utils.dashboarding import (
    AimExperimentTracker,
    TrackingConfig,
    create_epoch_callback,
    create_iter_callback,
    make_inference_history,
)
import fabricpc.utils.dashboarding.trackers as trackers_mod


class StubTracker:
    def __init__(self, config):
        self.config = config
        self.calls = []

    def track_batch_energy(self, energy, epoch, batch, context=None):
        self.calls.append(("energy", epoch, batch, energy))

    def track_batch_energy_per_node(self, state, structure, epoch, batch):
        self.calls.append(("node_energy", epoch, batch, state))

    def track_weight_distributions(self, params, structure, epoch, batch, nodes=None):
        self.calls.append(("weights", epoch, batch, params))

    def track_state(self, state, epoch, batch, infer_step, nodes=None):
        self.calls.append(("state", epoch, batch, infer_step, state))

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


def run(rng_key, config, *, algorithm="pc", infer_steps=3, n_batches=2, capture=None):
    structure = with_inference(
        make_classification_structure(), eta_infer=0.05, infer_steps=infer_steps
    )
    params = initialize_params(structure, rng_key)
    stub = StubTracker(config)
    callback = inner = create_iter_callback(stub)
    if capture is not None:

        def wrapped(ctx):
            capture.append(ctx)
            return inner(ctx)

        callback = wrapped
    train(
        params,
        structure,
        make_batches(rng_key, n_batches),
        optax.adam(1e-2),
        {"num_epochs": 1},
        rng_key,
        algorithm=algorithm,
        verbose=False,
        iter_callback=callback,
    )
    return stub


def tree_allclose(a, b, atol=1e-5):
    leaves = jax.tree_util.tree_leaves(
        jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y, atol=atol), a, b)
    )
    return bool(jnp.all(jnp.stack(leaves)))


# --- create_iter_callback -------------------------------------------------


def test_defaults_delegate_energy_and_weights_and_skip_state(rng_key):
    stub = run(rng_key, TrackingConfig(tracking_every_n_batches=1))
    assert [(c[1], c[2]) for c in stub.of("energy")] == [(0, 0), (0, 1)]
    assert all(isinstance(c[3], float) for c in stub.of("energy"))
    # Weights and per-node energy are handed to the tracker every batch; the
    # tracker's own config gates (tracking_every_n_batches, distribution_nodes,
    # nodes_to_track) decide whether anything is logged.
    assert [(c[1], c[2]) for c in stub.of("weights")] == [(0, 0), (0, 1)]
    assert all(isinstance(c[3], GraphParams) for c in stub.of("weights"))
    assert all(isinstance(c[3], GraphState) for c in stub.of("node_energy"))
    assert stub.of("state") == []


def test_track_state_without_distribution_nodes_does_not_resettle(rng_key):
    stub = run(
        rng_key,
        TrackingConfig(track_state=True, tracking_every_n_batches=1),
    )
    assert stub.of("state") == []


def test_track_state_logs_resettle_at_every_sampled_step_under_pc(rng_key):
    config = TrackingConfig(
        track_state=True,
        distribution_nodes=["h"],
        tracking_every_n_batches=1,
        state_tracking_every_n_infer_steps=1,
    )
    stub = run(rng_key, config, infer_steps=3)
    states = stub.of("state")
    # Steps 0 (initial state) through infer_steps on every tracked batch.
    assert [(c[1], c[2], c[3]) for c in states] == [
        (0, 0, 0),
        (0, 0, 1),
        (0, 0, 2),
        (0, 0, 3),
        (0, 1, 0),
        (0, 1, 1),
        (0, 1, 2),
        (0, 1, 3),
    ]
    assert all(isinstance(c[4], GraphState) for c in states)


def test_state_tracking_every_n_infer_steps_subsamples_and_keeps_final(rng_key):
    config = TrackingConfig(
        track_state=True,
        distribution_nodes=["h"],
        tracking_every_n_batches=1,
        state_tracking_every_n_infer_steps=2,
    )
    stub = run(rng_key, config, infer_steps=4, n_batches=1)
    assert [c[3] for c in stub.of("state")] == [0, 2, 4]


def test_resettle_states_match_make_inference_history_on_context(rng_key):
    """The logged states are the settle of ctx.batch under ctx.params from
    ctx.batch_key: step 0 is the initialization, the last is run_inference's
    settle."""
    config = TrackingConfig(
        track_state=True,
        distribution_nodes=["h"],
        tracking_every_n_batches=1,
        state_tracking_every_n_infer_steps=2,
    )
    seen = []
    stub = run(rng_key, config, infer_steps=4, n_batches=1, capture=seen)
    (ctx,) = seen
    clamps = build_clamps(ctx.batch, ctx.structure, clamp_target=True)
    init_state = initialize_graph_state(
        ctx.structure,
        batch_size_of(ctx.batch, ctx.structure),
        ctx.batch_key,
        clamps=clamps,
        params=ctx.params,
    )
    final, states = make_inference_history(ctx.structure, every=2)(
        ctx.params, init_state, clamps
    )
    logged = [c[4] for c in stub.of("state")]
    assert len(logged) == 3
    for i, state in enumerate(logged):
        assert tree_allclose(state, jax.tree_util.tree_map(lambda a: a[i], states))
    assert tree_allclose(logged[0], init_state)
    assert tree_allclose(
        logged[-1], run_inference(ctx.params, init_state, clamps, ctx.structure)
    )
    assert tree_allclose(final, logged[-1])


def test_track_state_distributions_implies_track_state(rng_key):
    config = TrackingConfig(
        track_state_distributions=True,
        distribution_nodes=["h"],
        tracking_every_n_batches=1,
        state_tracking_every_n_infer_steps=1,
    )
    assert config.tracks_state
    stub = run(rng_key, config, infer_steps=2)
    assert len(stub.of("state")) == 6  # steps 0, 1, 2 on each of two batches


def test_track_state_logs_feedforward_state_once_under_backprop(rng_key):
    config = TrackingConfig(
        track_state=True, distribution_nodes=["h"], tracking_every_n_batches=1
    )
    stub = run(rng_key, config, algorithm="backprop")
    states = stub.of("state")
    assert [(c[1], c[2], c[3]) for c in states] == [(0, 0, 0), (0, 1, 0)]
    assert all(isinstance(c[4], GraphState) for c in states)


def test_tracking_every_n_batches_gates_state(rng_key):
    config = TrackingConfig(
        track_state=True,
        distribution_nodes=["h"],
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
        rng_key=rng_key,
        metrics={},
        algorithm="pc",
        epoch_key=rng_key,
    )
    assert create_epoch_callback(stub, structure)(ctx) is None
    assert stub.calls == []


# --- make_inference_history -------------------------------------------------


def test_make_inference_history_rejects_every_below_one(rng_key):
    structure = with_inference(make_classification_structure(), infer_steps=2)
    with pytest.raises(ValueError):
        make_inference_history(structure, every=0)


# --- AimExperimentTracker node gating --------------------------------------


class FakeRun:
    def __init__(self):
        self.records = []

    def track(self, value, name, step=None, epoch=None, context=None):
        self.records.append((name, dict(context or {})))


def make_tracker(monkeypatch, config):
    monkeypatch.setattr(
        trackers_mod, "get_aim", lambda: SimpleNamespace(Distribution=lambda s: s)
    )
    tracker = AimExperimentTracker(config=config)
    tracker._run = FakeRun()
    tracker._initialized = True
    return tracker


def params_and_state(rng_key):
    structure = with_inference(make_classification_structure(), infer_steps=1)
    params = initialize_params(structure, rng_key)
    batch = next(iter(make_batches(rng_key, 1)))
    clamps = build_clamps(batch, structure, clamp_target=True)
    state = initialize_graph_state(
        structure,
        batch_size_of(batch, structure),
        rng_key,
        clamps=clamps,
        params=params,
    )
    return structure, params, state


def test_tracker_weights_log_only_distribution_nodes(rng_key, monkeypatch):
    structure, params, _ = params_and_state(rng_key)
    tracker = make_tracker(monkeypatch, TrackingConfig(distribution_nodes=["h"]))
    tracker.track_weight_distributions(params, structure, epoch=0, batch=0)
    nodes = {ctx["node"] for _, ctx in tracker._run.records}
    assert nodes == {"h"}


def test_tracker_weights_log_nothing_without_distribution_nodes(rng_key, monkeypatch):
    structure, params, _ = params_and_state(rng_key)
    tracker = make_tracker(monkeypatch, TrackingConfig())
    tracker.track_weight_distributions(params, structure, epoch=0, batch=0)
    assert tracker._run.records == []


def test_tracker_state_logs_only_distribution_nodes(rng_key, monkeypatch):
    _, _, state = params_and_state(rng_key)
    tracker = make_tracker(
        monkeypatch, TrackingConfig(track_state=True, distribution_nodes=["h"])
    )
    tracker.track_state(state, epoch=0, batch=0, infer_step=0)
    nodes = {ctx["node"] for _, ctx in tracker._run.records}
    assert nodes == {"h"}
    # Summary stats only: no histogram records without track_state_distributions.
    assert {name for name, _ in tracker._run.records} == {
        f"{v}_{s}"
        for v in ("z_latent", "z_mu", "energy")
        for s in ("mean", "std", "norm")
    }


def test_tracker_state_logs_nothing_without_distribution_nodes(rng_key, monkeypatch):
    _, _, state = params_and_state(rng_key)
    tracker = make_tracker(monkeypatch, TrackingConfig(track_state=True))
    tracker.track_state(state, epoch=0, batch=0, infer_step=0)
    assert tracker._run.records == []


def test_tracker_track_state_is_noop_unless_configured():
    tracker = AimExperimentTracker(config=TrackingConfig(distribution_nodes=["h"]))
    tracker._ensure_initialized = lambda: pytest.fail("track_state touched the run")
    tracker.track_state(state=None, epoch=0, batch=0, infer_step=0)
