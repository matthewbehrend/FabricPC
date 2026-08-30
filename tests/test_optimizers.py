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


@pytest.mark.parametrize(
    "ngd_transform",
    [
        lambda: scale_by_natural_gradient_diag(fisher_decay=0.95, damping=1e-3),
        lambda: scale_by_natural_gradient_layerwise(fisher_decay=0.95, damping=1e-3),
    ],
)
def test_natural_gradients_work_in_train_step(rng_key, ngd_transform):
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

    optimizer = optax.chain(
        ngd_transform(),
        optax.scale(-1e-3),
    )
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
