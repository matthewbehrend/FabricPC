"""Parity of the dashboarding training step with the trainer's PC step.

``train_step_with_history`` is a hand copy of the PC step with the inference
loop swapped for a history-collecting scan. This pins its gradient
normalization and energy to ``make_train_step``, so the two cannot drift.
"""

import jax
import optax

from conftest import make_classification_structure, max_param_diff
from fabricpc.core.energy import graph_energy
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training import build_clamps, make_train_step
from fabricpc.utils.dashboarding import (
    make_tracked_probe,
    run_inference_with_history,
    train_step_with_history,
)


def test_train_step_with_history_matches_trainer_step(rng_key):
    """Under optax.sgd(1.0) the applied update is the gradient itself, so a
    parameter match pins the per-prediction normalization; the returned
    energy is graph_energy / N with N = B for a rank-2 target."""
    structure = make_classification_structure()
    params = initialize_params(structure, rng_key)
    kx, ky = jax.random.split(rng_key)
    batch_size = 4
    batch = {
        "x": jax.random.normal(kx, (batch_size, 6)),
        "y": jax.nn.one_hot(jax.random.randint(ky, (batch_size,), 0, 3), 3),
    }
    optimizer = optax.sgd(1.0)

    step = make_train_step(structure, optimizer)
    ref_params, _, ref_metrics, _ = step(params, optimizer.init(params), batch, rng_key)

    tracked = jax.jit(
        lambda p, o, b, k: train_step_with_history(p, o, b, structure, optimizer, k)
    )
    new_params, _, energy, final_state, history = tracked(
        params, optimizer.init(params), batch, rng_key
    )

    assert max_param_diff(ref_params, new_params) < 1e-5
    assert abs(float(energy) - float(ref_metrics["energy"])) < 1e-6
    expected = float(graph_energy(final_state, structure)) / batch_size
    assert expected > 0.0
    assert abs(float(energy) - expected) < 1e-6
    # One stacked entry per inference step (the fixture settles for 10 steps).
    assert history["h"]["energy"].shape == (10,)


def test_make_tracked_probe_matches_eager_init_and_history(rng_key):
    """``make_tracked_probe(structure)(params, key, clamps)`` is latent
    initialization plus ``run_inference_with_history`` in one jitted program;
    on CPU it equals the eager composition, and it recompiles for nothing when
    called again with other params."""
    structure = make_classification_structure()
    params = initialize_params(structure, rng_key)
    kx, ky = jax.random.split(rng_key)
    batch = {
        "x": jax.random.normal(kx, (4, 6)),
        "y": jax.nn.one_hot(jax.random.randint(ky, (4,), 0, 3), 3),
    }
    clamps = build_clamps(batch, structure, clamp_target=True)

    probe = make_tracked_probe(structure)
    final, metrics = probe(params, rng_key, clamps)

    init_state = initialize_graph_state(structure, 4, rng_key, clamps, params=params)
    ref_final, ref_metrics = run_inference_with_history(
        params, init_state, clamps, structure
    )
    assert metrics["h"]["energy"].shape == (10,)
    for node, node_metrics in ref_metrics.items():
        for name, ref in node_metrics.items():
            assert jax.numpy.allclose(metrics[node][name], ref, atol=1e-6), (node, name)
    assert jax.numpy.allclose(final.nodes["h"].z_latent, ref_final.nodes["h"].z_latent)

    other = initialize_params(structure, jax.random.fold_in(rng_key, 1))
    _, other_metrics = probe(other, rng_key, clamps)
    assert not jax.numpy.allclose(other_metrics["h"]["energy"], metrics["h"]["energy"])
