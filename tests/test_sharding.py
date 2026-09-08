"""Mesh-based data parallelism tests (replaces the deleted pmap tests).

The multi-device legs need two devices; run them with

    XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \\
        pytest tests/test_sharding.py -q

Without the flag those tests skip; the single-device mesh leg always runs.
"""

import warnings

import jax
import jax.numpy as jnp
import optax
import pytest
from jax.sharding import NamedSharding, PartitionSpec as P

from conftest import ListLoader, make_classification_structure, max_param_diff
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.types import GraphState
from fabricpc.training import evaluate, IterContext, make_train_step, train

requires_two_devices = pytest.mark.skipif(
    jax.device_count() < 2,
    reason="needs >=2 devices (XLA_FLAGS=--xla_force_host_platform_device_count=2)",
)

make_structure = make_classification_structure


def make_batch(rng_key, batch_size):
    kx, ky = jax.random.split(rng_key)
    x = jax.random.normal(kx, (batch_size, 6))
    y = jax.nn.one_hot(jax.random.randint(ky, (batch_size,), 0, 3), 3)
    return {"x": x, "y": y}


def test_single_device_mesh_matches_no_mesh(rng_key):
    """A size-1 'data' mesh runs the same jitted step: identical results."""
    structure = make_structure()
    params = initialize_params(structure, rng_key)
    loader = ListLoader([make_batch(rng_key, 4)])
    optimizer = optax.adam(1e-2)
    mesh = jax.make_mesh((1,), ("data",))

    plain = train(
        params, structure, loader, optimizer, {"num_epochs": 1}, rng_key, verbose=False
    )
    meshed = train(
        params,
        structure,
        loader,
        optimizer,
        {"num_epochs": 1},
        rng_key,
        mesh=mesh,
        verbose=False,
    )
    assert max_param_diff(plain.params, meshed.params) < 1e-12

    ev_plain = evaluate(params, structure, loader, {}, rng_key)
    ev_meshed = evaluate(params, structure, loader, {}, rng_key, mesh=mesh)
    for key in ev_plain:
        assert abs(ev_plain[key] - ev_meshed[key]) < 1e-6, key


@requires_two_devices
@pytest.mark.parametrize("algorithm", ["pc", "backprop"])
def test_train_step_mesh_matches_single_device(rng_key, algorithm):
    structure = make_structure()
    params = initialize_params(structure, rng_key)
    batch = make_batch(rng_key, 8)
    optimizer = optax.adam(1e-2)
    mesh = jax.make_mesh((2,), ("data",))

    step_plain = make_train_step(structure, optimizer, algorithm=algorithm)
    step_mesh = make_train_step(structure, optimizer, algorithm=algorithm, mesh=mesh)

    opt_state = optimizer.init(params)
    p1, _, m1, _ = step_plain(params, opt_state, batch, rng_key)
    p2, _, m2, _ = step_mesh(params, opt_state, batch, rng_key)

    # Cross-device reductions reorder float sums; parity is tight, not bitwise.
    assert max_param_diff(p1, p2) < 1e-5
    for key in m1:
        assert abs(float(m1[key]) - float(m2[key])) < 1e-4, key


@requires_two_devices
def test_train_mesh_skips_ragged_batch(rng_key):
    structure = make_structure()
    params = initialize_params(structure, rng_key)
    k1, k2, k3 = jax.random.split(rng_key, 3)
    loader = ListLoader([make_batch(k1, 4), make_batch(k2, 3), make_batch(k3, 4)])
    mesh = jax.make_mesh((2,), ("data",))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = train(
            params,
            structure,
            loader,
            optax.adam(1e-2),
            {"num_epochs": 1},
            rng_key,
            mesh=mesh,
            verbose=False,
        )
    assert result.step == 2  # the size-3 batch was skipped
    assert len(result.iter_results[0]) == 2
    assert any("not divisible" in str(w.message) for w in caught)


@requires_two_devices
def test_evaluate_mesh_matches_single_device_ragged(rng_key):
    """The ragged final batch is zero-padded; per-sample weights make every
    real sample count exactly once, so mesh and single-device eval agree."""
    structure = make_structure()
    params = initialize_params(structure, rng_key)
    k1, k2 = jax.random.split(rng_key)
    loader = ListLoader([make_batch(k1, 4), make_batch(k2, 3)])
    mesh = jax.make_mesh((2,), ("data",))

    plain = evaluate(params, structure, loader, {}, rng_key)
    meshed = evaluate(params, structure, loader, {}, rng_key, mesh=mesh)
    assert set(plain) == set(meshed)
    for key in plain:
        assert abs(plain[key] - meshed[key]) < 1e-4, key


@requires_two_devices
def test_evaluate_padded_samples_zero_weight(rng_key):
    """A custom metric over a ragged sharded batch sees only real samples."""
    structure = make_structure()
    params = initialize_params(structure, rng_key)
    k1, k2 = jax.random.split(rng_key)
    batch_a = make_batch(k1, 4)
    batch_b = make_batch(k2, 3)
    loader = ListLoader([batch_a, batch_b])
    mesh = jax.make_mesh((2,), ("data",))

    def first_feature(state, batch, structure):
        v = batch["x"][:, 0]
        return v, jnp.ones_like(v)

    out = evaluate(
        params,
        structure,
        loader,
        {},
        rng_key,
        mesh=mesh,
        metrics={"first_feature": first_feature},
    )
    expected = float((jnp.sum(batch_a["x"][:, 0]) + jnp.sum(batch_b["x"][:, 0])) / 7.0)
    # If padded zeros leaked in, the denominator would be 8 and this fails.
    assert abs(out["first_feature"] - expected) < 1e-5


@requires_two_devices
def test_iter_context_under_mesh(rng_key):
    """The iteration callback sees the data-sharded batch and the batch's
    GraphState."""
    structure = make_structure()
    params = initialize_params(structure, rng_key)
    k1, k2 = jax.random.split(rng_key)
    loader = ListLoader([make_batch(k1, 4), make_batch(k2, 4)])
    mesh = jax.make_mesh((2,), ("data",))
    seen = []

    def iter_callback(ctx: IterContext):
        assert ctx.batch["x"].sharding == NamedSharding(mesh, P("data"))
        assert isinstance(ctx.state, GraphState)
        seen.append(ctx.batch_idx)

    train(
        params,
        structure,
        loader,
        optax.adam(1e-2),
        {"num_epochs": 1},
        rng_key,
        mesh=mesh,
        verbose=False,
        iter_callback=iter_callback,
    )
    assert seen == [0, 1]
