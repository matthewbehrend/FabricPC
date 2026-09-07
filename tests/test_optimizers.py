#!/usr/bin/env python3
"""
Tests for natural gradient optimizer transforms and their integration with training.
"""

import pytest
import jax
import jax.numpy as jnp
import optax

from fabricpc.training.optimizers import (
    scale_by_natural_gradient_diag,
    scale_by_natural_gradient_layerwise,
)
from fabricpc.training import make_train_step
from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import SigmoidActivation
from fabricpc.core.inference import InferenceSGD

NGD_TRANSFORMS = pytest.mark.parametrize(
    "make_transform",
    [scale_by_natural_gradient_diag, scale_by_natural_gradient_layerwise],
    ids=["diag", "layerwise"],
)


def _gradient_tree(key, scale=1.0):
    """Two leaves of different sizes (35 and 3), so the layerwise transform's
    per-leaf Fisher scalars differ from each other."""
    k_w, k_b = jax.random.split(key)
    return {
        "w": scale * jax.random.normal(k_w, (7, 5)),
        "b": scale * jax.random.normal(k_b, (3,)),
    }


def test_ngd_diag_updates():
    params = {"w": jnp.ones((4, 3)), "b": jnp.ones((3,))}
    grads = jax.tree_util.tree_map(lambda p: jnp.full_like(p, 0.1), params)

    optimizer = optax.chain(
        scale_by_natural_gradient_diag(fisher_decay=0.9, damping=1e-3),
        optax.scale(-1e-2),
    )
    opt_state = optimizer.init(params)
    updates, _ = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    assert jnp.all(jnp.isfinite(new_params["w"]))
    assert not jnp.allclose(new_params["w"], params["w"])


def test_ngd_layerwise_updates():
    params = {"w": jnp.ones((5, 2)), "b": jnp.ones((2,))}
    grads = jax.tree_util.tree_map(lambda p: jnp.full_like(p, 0.05), params)

    optimizer = optax.chain(
        scale_by_natural_gradient_layerwise(fisher_decay=0.9, damping=1e-3),
        optax.scale(-1e-2),
    )
    opt_state = optimizer.init(params)
    updates, _ = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)

    assert jnp.all(jnp.isfinite(new_params["w"]))
    assert not jnp.allclose(new_params["w"], params["w"])


@NGD_TRANSFORMS
def test_natural_gradients_work_in_train_step(rng_key, make_transform):
    """The default-argument transform runs through make_train_step and moves
    the parameters with a finite objective."""
    x_node = Linear(shape=(6,), name="x")
    hidden = Linear(shape=(4,), activation=SigmoidActivation(), name="hidden")
    y_node = Linear(shape=(3,), activation=SigmoidActivation(), name="y")
    structure = graph(
        nodes=[x_node, hidden, y_node],
        edges=[
            Edge(source=x_node, target=hidden.slot("in")),
            Edge(source=hidden, target=y_node.slot("in")),
        ],
        task_map=TaskMap(x=x_node, y=y_node),
        inference=InferenceSGD(),
    )
    params = initialize_params(structure, rng_key)

    optimizer = optax.chain(make_transform(fisher_decay=0.95), optax.scale(-1e-3))
    opt_state = optimizer.init(params)

    batch_size = 8
    key_x, key_y = jax.random.split(rng_key)
    batch = {
        "x": jax.random.normal(key_x, (batch_size, 6)),
        "y": jax.random.normal(key_y, (batch_size, 3)),
    }

    step = make_train_step(structure, optimizer)
    updated_params, _, metrics, _ = step(params, opt_state, batch, rng_key)
    energy = metrics["energy"]

    old_w = params.nodes["hidden"].weights["x->hidden:in"]
    new_w = updated_params.nodes["hidden"].weights["x->hidden:in"]

    assert not jnp.isnan(energy)
    assert energy > 0
    assert not jnp.allclose(old_w, new_w)


def test_natural_gradient_hparams_validation():
    with pytest.raises(ValueError, match="fisher_decay"):
        scale_by_natural_gradient_diag(fisher_decay=1.1)

    with pytest.raises(ValueError, match="damping"):
        scale_by_natural_gradient_layerwise(damping=0.0)

    with pytest.raises(ValueError, match="damping"):
        scale_by_natural_gradient_diag(damping=-1e-3)


@NGD_TRANSFORMS
def test_natural_gradient_first_step_is_bias_corrected(rng_key, make_transform):
    """After one step from a fresh state the bias-corrected Fisher equals g**2
    (diag) or mean(g**2) per leaf (layerwise), so the update is
    g / (F + damping). Without bias correction the EMA would hold only
    (1 - 0.95) * g**2 and the update would be about 20x larger."""
    damping = 1e-3
    grads = _gradient_tree(rng_key)
    params = jax.tree_util.tree_map(jnp.zeros_like, grads)
    transform = make_transform(fisher_decay=0.95, damping=damping)
    updates, state = transform.update(grads, transform.init(params))

    squares = jax.tree_util.tree_map(jnp.square, grads)
    if make_transform is scale_by_natural_gradient_diag:
        fisher_hat = squares
    else:
        fisher_hat = jax.tree_util.tree_map(jnp.mean, squares)
    expected = jax.tree_util.tree_map(lambda g, f: g / (f + damping), grads, fisher_hat)
    uncorrected = jax.tree_util.tree_map(
        lambda g, f: g / (0.05 * f + damping), grads, fisher_hat
    )

    assert int(state.count) == 1
    for name in grads:
        assert jnp.allclose(updates[name], expected[name], rtol=1e-5, atol=0.0)
        assert not jnp.allclose(updates[name], uncorrected[name], rtol=1e-2, atol=0.0)


@NGD_TRANSFORMS
def test_damping_dominated_update_is_sgd(rng_key, make_transform):
    """Where `damping` dominates the Fisher entries the update is g / damping:
    plain SGD with rate scale / damping. This is the regime the demo presets
    train in (see the module docstring)."""
    damping = 1.0
    grads = _gradient_tree(rng_key, scale=1e-3)  # F ~ 1e-6 << damping
    params = jax.tree_util.tree_map(jnp.zeros_like, grads)
    transform = make_transform(fisher_decay=0.95, damping=damping)
    updates, _ = transform.update(grads, transform.init(params))
    for name in grads:
        assert jnp.allclose(updates[name], grads[name] / damping, rtol=1e-5, atol=0.0)


def test_fisher_dominated_diag_update_is_inverse_gradient(rng_key):
    """Where the Fisher dominates `damping` the diagonal update is about 1 / g:
    the Fisher is the squared mean gradient, so g / F inverts the gradient.
    This pins the documented defect of the estimator."""
    damping = 1e-12
    grads = _gradient_tree(rng_key)  # entries of order 1, F ~ 1 >> damping
    params = jax.tree_util.tree_map(jnp.zeros_like, grads)
    transform = scale_by_natural_gradient_diag(fisher_decay=0.95, damping=damping)
    updates, _ = transform.update(grads, transform.init(params))
    for name in grads:
        assert jnp.allclose(updates[name], 1.0 / grads[name], rtol=1e-4, atol=0.0)
