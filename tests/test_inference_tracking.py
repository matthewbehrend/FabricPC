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
from fabricpc.training import make_train_step
from fabricpc.utils.dashboarding import train_step_with_history


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
