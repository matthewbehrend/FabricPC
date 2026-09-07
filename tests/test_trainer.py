"""Tests for the unified trainer (fabricpc.training.trainer / metrics / generation).

The permanent parity evidence lives here, independent of the deleted legacy
trainers:

- PC: `train` matches an in-test hand-rolled reference step (clamps -> init
  -> inference -> local grads -> optax) under the fold_in RNG stream.
- Backprop: the applied update equals `jax.grad` of a reference loss composed
  from raw jnp ops, and the objective equals the clamped-target energy.
- Resume: `train(N)` bitwise equals `train(k)` then
  `train(opt_state=..., start_epoch=k, num_epochs=N-k)`.
"""

import math
from typing import List

import jax
import jax.numpy as jnp
import optax
import pytest

from conftest import (
    ListLoader,
    make_classification_structure,
    max_param_diff,
    with_inference,
)
from fabricpc.core.activations import (
    IdentityActivation,
    SigmoidActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy, graph_energy
from fabricpc.core.inference import InferenceSGD, InferenceSGDNormClip, run_inference
from fabricpc.core.learning import compute_local_weight_gradients
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import (
    GlobalStateInit,
    initialize_graph_state,
    initialize_params,
)
from fabricpc.models import create_deep_transformer
from fabricpc.nodes import Linear
from fabricpc.training import (
    EpochContext,
    EvalMetric,
    TrainResult,
    build_clamps,
    evaluate,
    generate,
    grad_denominator,
    make_train_step,
    metrics as metrics_mod,
    pc_weight_gradients,
    train,
)

# Params are float32 and the hand-rolled reference and train() are two
# separately jitted programs, so XLA:CPU may differ in the last bits
# (observed one-ULP diffs up to ~6e-8 on CI). A real composition bug (wrong
# key derivation, extra/missing step, wrong lr at 1e-2) produces diffs >= 1e-3.
PARITY_TOL = 1e-5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_batches(
    rng_key, *, batch_size=4, n_batches=3, in_dim=6, n_classes=3, one_hot=True
):
    batches: List[dict] = []
    key = rng_key
    for _ in range(n_batches):
        kx, ky, key = jax.random.split(key, 3)
        x = jax.random.normal(kx, (batch_size, in_dim))
        labels = jax.random.randint(ky, (batch_size,), 0, n_classes)
        y = jax.nn.one_hot(labels, n_classes) if one_hot else labels
        batches.append({"x": x, "y": y})
    return ListLoader(batches)


classification_structure = make_classification_structure


def rng_sensitive_structure():
    """The classification chain with GlobalStateInit: unclamped latents keep
    their random initialization into inference, so the training key changes
    the result. Under the default FeedforwardStateInit the feedforward pass
    overwrites every unclamped in_degree>0 latent and the key has no effect —
    tests that pin the RNG stream must use this graph."""
    return make_classification_structure(state_initializer=GlobalStateInit())


def sequence_structure(seq_len=6, vocab_size=11):
    """Tiny v2 transformer (masks internally; FeedforwardStateInit built in)."""
    return create_deep_transformer(
        depth=1,
        embed_dim=8,
        num_heads=2,
        mlp_dim=16,
        seq_len=seq_len,
        vocab_size=vocab_size,
        inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=3, max_norm=5.0),
    )


def make_token_batches(rng_key, *, batch_size=4, n_batches=2, seq_len=6, vocab=11):
    batches = []
    key = rng_key
    for _ in range(n_batches):
        kx, ky, key = jax.random.split(key, 3)
        x = jax.random.randint(kx, (batch_size, seq_len), 0, vocab)
        y = jax.random.randint(ky, (batch_size, seq_len), 0, vocab)
        batches.append({"x": x, "y": y})
    return ListLoader(batches)


def v1_masked_structure(seq_len=5, vocab_size=7):
    """Graph declaring an external 'causal_mask' task node (v1 style)."""
    x = Linear(shape=(seq_len, vocab_size), name="inp")
    mask = Linear(shape=(1, seq_len, seq_len), name="mask")
    y = Linear(
        shape=(seq_len, vocab_size),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="out",
    )
    return graph(
        nodes=[x, mask, y],
        edges=[Edge(source=x, target=y.slot("in"))],
        task_map=TaskMap(x=x, y=y, causal_mask=mask),
        inference=InferenceSGD(),
    )


# ---------------------------------------------------------------------------
# PC parity vs a hand-rolled reference step
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("graph_kind", ["feedforward_init", "global_init"])
def test_pc_parity_hand_rolled_reference(rng_key, graph_kind):
    """train(algorithm='pc') matches the composed primitives under the
    fold_in stream: clamps -> init -> inference -> local grads -> optax.

    The global_init leg is the RNG-sensitive one: with GlobalStateInit the
    latent initialization consumes the key, so this leg fails under any
    other key derivation (the feedforward_init leg is key-insensitive and
    pins only the non-RNG mechanics).
    """
    structure = (
        classification_structure()
        if graph_kind == "feedforward_init"
        else rng_sensitive_structure()
    )
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = make_batches(rng_key)
    optimizer = optax.adam(1e-2)

    x_node = structure.task_map["x"]
    y_node = structure.task_map["y"]

    @jax.jit
    def reference_step(p, opt_state, batch, key):
        clamps = {x_node: batch["x"], y_node: batch["y"]}
        state = initialize_graph_state(
            structure, batch["x"].shape[0], key, clamps=clamps, params=p
        )
        state = run_inference(p, state, clamps, structure)
        grads = pc_weight_gradients(p, state, structure, clamps)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state

    ref_params = params
    ref_opt = optimizer.init(params)
    for epoch_idx in range(2):
        epoch_key = jax.random.fold_in(train_key, epoch_idx)
        for batch_idx, batch in enumerate(loader):
            batch_key = jax.random.fold_in(epoch_key, batch_idx)
            ref_params, ref_opt = reference_step(ref_params, ref_opt, batch, batch_key)

    result = train(
        params,
        structure,
        loader,
        optimizer,
        {"num_epochs": 2},
        train_key,
        verbose=False,
    )
    assert max_param_diff(ref_params, result.params) < PARITY_TOL


# ---------------------------------------------------------------------------
# Backprop gradient correctness
# ---------------------------------------------------------------------------


def test_backprop_gradient_matches_reference_ce(rng_key):
    """The applied update under optax.sgd(1.0) equals jax.grad of a reference
    cross-entropy loss composed from raw jnp ops."""
    structure = classification_structure()
    params_key, data_key, step_key = jax.random.split(rng_key, 3)
    params = initialize_params(structure, params_key)
    kx, ky = jax.random.split(data_key)
    x = jax.random.normal(kx, (4, 6))
    y = jax.nn.one_hot(jax.random.randint(ky, (4,), 0, 3), 3)

    def reference_loss(p):
        w1 = p.nodes["h"].weights["x->h:in"]
        b1 = p.nodes["h"].biases["b"]
        w2 = p.nodes["y"].weights["h->y:in"]
        b2 = p.nodes["y"].biases["b"]
        hidden = jax.nn.sigmoid(x @ w1 + b1)
        probs = jax.nn.softmax(hidden @ w2 + b2, axis=-1)
        # CrossEntropyEnergy: -sum y*log(clip(mu, 1e-7, 1)), summed then
        # / prediction count (= batch for a rank-2 target)
        return -jnp.sum(y * jnp.log(jnp.clip(probs, 1e-7, 1.0))) / x.shape[0]

    ref_grads = jax.grad(reference_loss)(params)

    optimizer = optax.sgd(1.0)
    step = make_train_step(structure, optimizer, algorithm="backprop")
    new_params, _, metrics, _ = step(
        params, optimizer.init(params), {"x": x, "y": y}, step_key
    )
    applied_grads = jax.tree_util.tree_map(
        lambda old, new: old - new, params, new_params
    )
    assert max_param_diff(ref_grads, applied_grads) < 1e-5
    assert abs(float(metrics["energy"]) - float(reference_loss(params))) < 1e-5


def test_backprop_gaussian_objective_is_precision_sse(rng_key):
    """A GaussianEnergy output's backprop objective is 0.5*precision*SSE
    / prediction count (= batch for a rank-2 target), not an element-mean
    MSE."""
    precision = 2.0
    structure = classification_structure(
        output_energy=GaussianEnergy(precision=precision),
        output_activation=IdentityActivation(),
    )
    params_key, data_key, step_key = jax.random.split(rng_key, 3)
    params = initialize_params(structure, params_key)
    kx, ky = jax.random.split(data_key)
    x = jax.random.normal(kx, (4, 6))
    y = jax.nn.one_hot(jax.random.randint(ky, (4,), 0, 3), 3)

    w1 = params.nodes["h"].weights["x->h:in"]
    b1 = params.nodes["h"].biases["b"]
    w2 = params.nodes["y"].weights["h->y:in"]
    b2 = params.nodes["y"].biases["b"]
    mu = jax.nn.sigmoid(x @ w1 + b1) @ w2 + b2
    expected = 0.5 * precision * jnp.sum((y - mu) ** 2) / x.shape[0]

    step = make_train_step(structure, optax.sgd(0.1), algorithm="backprop")
    _, _, metrics, _ = step(
        params, optax.sgd(0.1).init(params), {"x": x, "y": y}, step_key
    )
    assert abs(float(metrics["energy"]) - float(expected)) < 1e-5


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("graph_kind", ["feedforward_init", "global_init"])
def test_resume_is_bitwise_identical(rng_key, graph_kind):
    """The global_init leg pins the resume RNG contract: with GlobalStateInit
    the per-epoch keys matter, so this leg fails if start_epoch were ignored
    or the stream derived from the epoch offset instead of the epoch index.
    The feedforward_init leg pins only the opt_state threading."""
    structure = (
        classification_structure()
        if graph_kind == "feedforward_init"
        else rng_sensitive_structure()
    )
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = make_batches(rng_key)
    optimizer = optax.adam(1e-2)

    full = train(
        params,
        structure,
        loader,
        optimizer,
        {"num_epochs": 3},
        train_key,
        verbose=False,
    )
    part1 = train(
        params,
        structure,
        loader,
        optimizer,
        {"num_epochs": 2},
        train_key,
        verbose=False,
    )
    part2 = train(
        part1.params,
        structure,
        loader,
        optimizer,
        {"num_epochs": 1},
        train_key,
        opt_state=part1.opt_state,
        start_epoch=2,
        verbose=False,
    )
    assert max_param_diff(full.params, part2.params) == 0.0
    assert full.step == part1.step + part2.step


def test_rng_key_changes_result_on_global_init(rng_key):
    """Sanity for the RNG-sensitive fixture: two training keys give
    different params, so the parity/resume legs above are not vacuous."""
    structure = rng_sensitive_structure()
    params = initialize_params(structure, rng_key)
    loader = make_batches(rng_key, n_batches=2)
    kwargs = dict(verbose=False)
    r1 = train(
        params,
        structure,
        loader,
        optax.adam(1e-2),
        {"num_epochs": 1},
        jax.random.PRNGKey(0),
        **kwargs,
    )
    r2 = train(
        params,
        structure,
        loader,
        optax.adam(1e-2),
        {"num_epochs": 1},
        jax.random.PRNGKey(1),
        **kwargs,
    )
    assert max_param_diff(r1.params, r2.params) > 0.0


def test_start_epoch_offsets_rng_stream(rng_key):
    """start_epoch=k must shift the per-epoch key to fold_in(rng_key, k):
    on an RNG-sensitive graph the params differ from a start_epoch=0 run.
    Fails if the stream were derived from the epoch offset (always 0 here)
    instead of the absolute epoch index."""
    structure = rng_sensitive_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = make_batches(rng_key, n_batches=2)
    r0 = train(
        params,
        structure,
        loader,
        optax.adam(1e-2),
        {"num_epochs": 1},
        train_key,
        verbose=False,
    )
    rk = train(
        params,
        structure,
        loader,
        optax.adam(1e-2),
        {"num_epochs": 1},
        train_key,
        start_epoch=3,
        verbose=False,
    )
    assert max_param_diff(r0.params, rk.params) > 0.0


def test_caller_params_survive_train(rng_key):
    """The internal step donates buffers; the caller's params must stay valid
    and unmodified."""
    structure = classification_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = make_batches(rng_key, n_batches=1)
    optimizer = optax.adam(1e-2)
    before = jax.tree_util.tree_map(jnp.copy, params)
    r1 = train(
        params,
        structure,
        loader,
        optimizer,
        {"num_epochs": 1},
        train_key,
        verbose=False,
    )
    # Donation must not have modified the caller's arrays in place.
    assert max_param_diff(params, before) == 0.0
    # A second call on the same arrays must not hit deleted buffers, and
    # identical inputs must reproduce the identical result.
    r2 = train(
        params,
        structure,
        loader,
        optimizer,
        {"num_epochs": 1},
        train_key,
        verbose=False,
    )
    assert max_param_diff(r1.params, r2.params) == 0.0


# ---------------------------------------------------------------------------
# Smoke: pc/backprop x classification/sequence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["pc", "backprop"])
@pytest.mark.parametrize("task", ["classification", "sequence"])
def test_train_and_evaluate_smoke(rng_key, algorithm, task):
    if task == "classification":
        structure = classification_structure()
        loader = make_batches(rng_key, n_batches=2)
    else:
        structure = sequence_structure()
        loader = make_token_batches(rng_key)
    params_key, train_key, eval_key = jax.random.split(rng_key, 3)
    params = initialize_params(structure, params_key)
    optimizer = optax.adam(1e-3)

    result = train(
        params,
        structure,
        loader,
        optimizer,
        {"num_epochs": 1},
        train_key,
        algorithm=algorithm,
        verbose=False,
    )
    assert isinstance(result, TrainResult)
    assert result.step == len(loader)
    for batch_metrics in result.iter_results[0]:
        assert set(batch_metrics) == {"energy", "target_energy"}
        for v in batch_metrics.values():
            assert math.isfinite(v)
        # Both keys are energies (CE targets here): non-negative.
        assert batch_metrics["energy"] >= 0.0
        assert batch_metrics["target_energy"] >= 0.0
    assert max_param_diff(params, result.params) > 0.0

    eval_metrics = evaluate(
        result.params, structure, loader, {}, eval_key, algorithm=algorithm
    )
    expected_keys = {"target_energy", "accuracy", "cross_entropy", "perplexity"}
    if algorithm == "pc":
        expected_keys.add("energy")
    assert set(eval_metrics) == expected_keys
    for v in eval_metrics.values():
        assert math.isfinite(v)
    assert 0.0 <= eval_metrics["accuracy"] <= 1.0
    assert eval_metrics["target_energy"] >= 0.0
    assert eval_metrics["perplexity"] >= 1.0
    if algorithm == "pc":
        assert eval_metrics["energy"] >= 0.0


# ---------------------------------------------------------------------------
# Non-float targets (one-hot derived from dtype)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("algorithm", ["pc", "backprop"])
def test_int_class_labels_train(rng_key, algorithm):
    structure = classification_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = make_batches(rng_key, n_batches=2, one_hot=False)  # (batch,) int32
    result = train(
        params,
        structure,
        loader,
        optax.adam(1e-3),
        {"num_epochs": 1},
        train_key,
        algorithm=algorithm,
        verbose=False,
    )
    assert max_param_diff(params, result.params) > 0.0


@pytest.mark.parametrize("algorithm", ["pc", "backprop"])
def test_int_token_targets_train(rng_key, algorithm):
    """(batch, seq) int32 token targets — the stock token-loader format."""
    structure = sequence_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = make_token_batches(rng_key, n_batches=1)
    result = train(
        params,
        structure,
        loader,
        optax.adam(1e-3),
        {"num_epochs": 1},
        train_key,
        algorithm=algorithm,
        verbose=False,
    )
    assert max_param_diff(params, result.params) > 0.0


@pytest.mark.parametrize("algorithm", ["pc", "backprop"])
def test_bool_targets_train(rng_key, algorithm):
    """bool targets one-hot like ints (the legacy dtype validator accepted
    floating only, so bools hit a TypeError before this fix)."""
    x_node = Linear(shape=(6,), name="x")
    h = Linear(shape=(8,), activation=SigmoidActivation(), name="h")
    y_node = Linear(
        shape=(2,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="y",
    )
    structure = graph(
        nodes=[x_node, h, y_node],
        edges=[
            Edge(source=x_node, target=h.slot("in")),
            Edge(source=h, target=y_node.slot("in")),
        ],
        task_map=TaskMap(x=x_node, y=y_node),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )
    params_key, train_key, data_key = jax.random.split(rng_key, 3)
    params = initialize_params(structure, params_key)
    x = jax.random.normal(data_key, (4, 6))
    y = jnp.array([True, False, True, False])
    loader = ListLoader([{"x": x, "y": y}])
    result = train(
        params,
        structure,
        loader,
        optax.adam(1e-3),
        {"num_epochs": 1},
        train_key,
        algorithm=algorithm,
        verbose=False,
    )
    assert max_param_diff(params, result.params) > 0.0


# ---------------------------------------------------------------------------
# build_clamps
# ---------------------------------------------------------------------------


def test_build_clamps_v1_injects_tril_mask(rng_key):
    seq_len, vocab = 5, 7
    structure = v1_masked_structure(seq_len, vocab)
    batch_size = 3
    x = jax.random.normal(rng_key, (batch_size, seq_len, vocab))
    y = jax.random.randint(rng_key, (batch_size, seq_len), 0, vocab)
    clamps = build_clamps({"x": x, "y": y}, structure, clamp_target=True)

    mask = clamps["mask"]
    assert mask.shape == (batch_size, 1, seq_len, seq_len)
    assert jnp.array_equal(mask[0, 0], jnp.tril(jnp.ones((seq_len, seq_len))))
    # int targets one-hot to the node's class axis
    assert clamps["out"].shape == (batch_size, seq_len, vocab)
    assert jnp.issubdtype(clamps["out"].dtype, jnp.floating)


def test_build_clamps_v2_has_no_mask_key(rng_key):
    structure = sequence_structure()
    x = jax.random.randint(rng_key, (2, 6), 0, 11)
    y = jax.random.randint(rng_key, (2, 6), 0, 11)
    clamps = build_clamps({"x": x, "y": y}, structure, clamp_target=True)
    mask_like = [n for n in clamps if "mask" in n]
    assert not mask_like
    assert set(clamps) == {structure.task_map["x"], structure.task_map["y"]}


def test_build_clamps_eval_leaves_targets_free(rng_key):
    structure = classification_structure()
    x = jax.random.normal(rng_key, (4, 6))
    y = jax.nn.one_hot(jnp.zeros(4, dtype=jnp.int32), 3)
    clamps = build_clamps({"x": x, "y": y}, structure, clamp_target=False)
    assert set(clamps) == {structure.task_map["x"]}


# ---------------------------------------------------------------------------
# Default metrics
# ---------------------------------------------------------------------------


def test_default_metrics_gaussian_target(rng_key):
    structure = classification_structure(
        output_energy=GaussianEnergy(), output_activation=IdentityActivation()
    )
    params = initialize_params(structure, rng_key)
    loader = make_batches(rng_key, n_batches=1)
    out = evaluate(params, structure, loader, {}, rng_key)
    assert set(out) == {"target_energy", "accuracy", "energy"}


def test_default_metrics_no_target_graph_raises_and_custom_works(rng_key):
    x_node = Linear(shape=(4,), name="x")
    h = Linear(shape=(5,), activation=SigmoidActivation(), name="h")
    structure = graph(
        nodes=[x_node, h],
        edges=[Edge(source=x_node, target=h.slot("in"))],
        task_map=TaskMap(x=x_node),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=3),
    )
    params = initialize_params(structure, rng_key)
    loader = ListLoader([{"x": jax.random.normal(rng_key, (4, 4))}])

    with pytest.raises(ValueError, match="target"):
        evaluate(params, structure, loader, {}, rng_key)

    def mean_hidden(state, batch, structure):
        v = jnp.mean(state.nodes["h"].z_mu, axis=-1)
        return v, jnp.ones_like(v)

    out = evaluate(
        params, structure, loader, {}, rng_key, metrics={"mean_hidden": mean_hidden}
    )
    assert set(out) == {"mean_hidden"}
    assert math.isfinite(out["mean_hidden"])


# ---------------------------------------------------------------------------
# Metric system: weighted aggregation and finalize
# ---------------------------------------------------------------------------


def test_custom_metric_weighted_aggregation_uneven_batches(rng_key):
    """Sum(value)/Sum(weight) over uneven batch sizes, not a per-batch mean."""
    structure = classification_structure()
    params = initialize_params(structure, rng_key)
    k1, k2 = jax.random.split(rng_key)
    batch_a = {
        "x": jax.random.normal(k1, (6, 6)),
        "y": jax.nn.one_hot(jax.random.randint(k1, (6,), 0, 3), 3),
    }
    batch_b = {
        "x": jax.random.normal(k2, (2, 6)),
        "y": jax.nn.one_hot(jax.random.randint(k2, (2,), 0, 3), 3),
    }
    loader = ListLoader([batch_a, batch_b])

    custom = EvalMetric(fn=metrics_mod.cross_entropy.fn)
    out = evaluate(
        params,
        structure,
        loader,
        {},
        rng_key,
        metrics={"ce": custom, "ppl": metrics_mod.perplexity},
    )

    # Hand-computed: per-sample CE from the feedforward prediction (a free
    # output settles nowhere: the feedforward state is the zero-error fixed
    # point), aggregated over ALL 8 samples.
    def per_sample_ce(batch):
        clamps = build_clamps(batch, structure, clamp_target=False)
        state = initialize_graph_state(
            structure, batch["x"].shape[0], rng_key, clamps=clamps, params=params
        )
        state = run_inference(params, state, clamps, structure)
        mu = state.nodes["y"].z_mu
        return -jnp.sum(batch["y"] * jnp.log(jnp.clip(mu, 1e-7, 1.0)), axis=-1)

    all_ce = jnp.concatenate([per_sample_ce(batch_a), per_sample_ce(batch_b)])
    expected = float(jnp.sum(all_ce) / 8.0)
    naive_per_batch_mean = float(
        (jnp.mean(per_sample_ce(batch_a)) + jnp.mean(per_sample_ce(batch_b))) / 2.0
    )
    assert abs(out["ce"] - expected) < 1e-5
    assert abs(expected - naive_per_batch_mean) > 1e-6  # the distinction is real
    # perplexity = exp of the aggregated mean, not a mean of per-batch exps
    assert abs(out["ppl"] - math.exp(expected)) < 1e-4


def test_eval_energy_matches_graph_energy(rng_key):
    """evaluate's default 'energy' metric must agree with graph_energy / N
    (N = B for a rank-2 target) — metrics._internal_energy_fn re-implements
    graph_energy's node ordering, so a divergence would otherwise be silent.
    GlobalStateInit
    keeps the eval energy nonzero (a free feedforward output sits at its
    zero-error fixed point)."""
    structure = rng_sensitive_structure()
    params = initialize_params(structure, rng_key)
    batch = next(iter(make_batches(rng_key, n_batches=1)))
    loader = ListLoader([batch])
    out = evaluate(params, structure, loader, {}, rng_key)

    clamps = build_clamps(batch, structure, clamp_target=False)
    key = jax.random.fold_in(rng_key, 0)  # evaluate's key for batch 0
    state = initialize_graph_state(
        structure, batch["x"].shape[0], key, clamps=clamps, params=params
    )
    state = run_inference(params, state, clamps, structure)
    expected = float(graph_energy(state, structure)) / batch["x"].shape[0]
    assert expected > 0.0
    assert abs(out["energy"] - expected) < 1e-5


def two_target_structure():
    """x(4) feeding two softmax + CE heads y1(3) and y2(5); task keys
    ``y`` -> y1 and ``y2`` -> y2."""
    x_node = Linear(shape=(4,), name="x")
    y1 = Linear(
        shape=(3,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="y1",
    )
    y2 = Linear(
        shape=(5,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="y2",
    )
    return graph(
        nodes=[x_node, y1, y2],
        edges=[
            Edge(source=x_node, target=y1.slot("in")),
            Edge(source=x_node, target=y2.slot("in")),
        ],
        task_map=TaskMap(x=x_node, y=y1, y2=y2),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )


def test_multi_target_metric_accumulation(rng_key):
    """Two target nodes: each default metric sums values AND weights across
    targets, so the mean is per prediction over both heads — pins the
    accumulation loops in metrics.py, which single-target tests never
    iterate twice."""
    structure = two_target_structure()
    params = initialize_params(structure, rng_key)
    kx, k1, k2 = jax.random.split(rng_key, 3)
    batch = {
        "x": jax.random.normal(kx, (4, 4)),
        "y": jax.random.randint(k1, (4,), 0, 3),
        "y2": jax.random.randint(k2, (4,), 0, 5),
    }
    out = evaluate(params, structure, ListLoader([batch]), {}, rng_key)

    # Hand-compute from the settled eval state.
    clamps = build_clamps(batch, structure, clamp_target=False)
    key = jax.random.fold_in(rng_key, 0)
    state = initialize_graph_state(structure, 4, key, clamps=clamps, params=params)
    state = run_inference(params, state, clamps, structure)
    correct = 0.0
    ce_total = 0.0
    for task_key, node in (("y", "y1"), ("y2", "y2")):
        mu = state.nodes[node].z_mu
        labels = batch[task_key]
        correct += float(jnp.sum(jnp.argmax(mu, axis=-1) == labels))
        onehot = jax.nn.one_hot(labels, mu.shape[-1])
        ce_total += float(-jnp.sum(onehot * jnp.log(jnp.clip(mu, 1e-7, 1.0))))
    # 4 samples x 2 targets = 8 predictions.
    assert abs(out["accuracy"] - correct / 8.0) < 1e-5
    assert abs(out["cross_entropy"] - ce_total / 8.0) < 1e-4
    assert abs(out["perplexity"] - math.exp(ce_total / 8.0)) < 1e-3


# ---------------------------------------------------------------------------
# Gradient normalization: one global prediction count N
# ---------------------------------------------------------------------------


def make_v1_token_batch(rng_key, *, batch_size=3, seq_len=5, vocab=7):
    """Float (B, S, V) input and int (B, S) targets for v1_masked_structure."""
    kx, ky = jax.random.split(rng_key)
    return {
        "x": jax.random.normal(kx, (batch_size, seq_len, vocab)),
        "y": jax.random.randint(ky, (batch_size, seq_len), 0, vocab),
    }


def _relative_deviation(node_grads, reference) -> float:
    """||a - b|| / ||b|| over all leaves of one node's parameters."""
    diff = jax.tree_util.tree_map(
        lambda a, b: jnp.sum((a - b) ** 2), node_grads, reference
    )
    norm = jax.tree_util.tree_map(lambda b: jnp.sum(b**2), reference)
    num = jax.tree_util.tree_reduce(lambda x, y: x + y, diff, jnp.zeros(()))
    den = jax.tree_util.tree_reduce(lambda x, y: x + y, norm, jnp.zeros(()))
    return float(jnp.sqrt(num) / jnp.sqrt(den))


def test_backprop_rank3_objective_divides_by_batch_times_seq(rng_key):
    """A (B, S, V) token target has N = B*S prediction positions: the
    backprop objective, both metrics, and the applied gradient are the target
    energy / (B*S), not / B."""
    seq_len, vocab, batch_size = 5, 7, 3
    structure = v1_masked_structure(seq_len, vocab)
    params = initialize_params(structure, rng_key)
    batch = make_v1_token_batch(rng_key, batch_size=batch_size)
    clamps = build_clamps(batch, structure, clamp_target=True)
    n_predictions = batch_size * seq_len
    assert grad_denominator(structure, clamps) == n_predictions

    optimizer = optax.sgd(1.0)
    step = make_train_step(structure, optimizer, algorithm="backprop")
    new_params, _, metrics, state = step(params, optimizer.init(params), batch, rng_key)
    expected = float(graph_energy(state, structure, node_names=("out",)))
    expected /= n_predictions
    assert expected > 0.0
    assert abs(float(metrics["energy"]) - expected) < 1e-5
    assert abs(float(metrics["target_energy"]) - expected) < 1e-5

    def objective(p):
        st = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=p
        )
        return graph_energy(st, structure, node_names=("out",)) / n_predictions

    ref_grads = jax.grad(objective)(params)
    applied = jax.tree_util.tree_map(lambda o, n: o - n, params, new_params)
    assert max_param_diff(ref_grads, applied) < 1e-5


def test_pc_energy_rank3_target_is_per_prediction(rng_key):
    """PC 'energy' on a (B, S, V) target is graph_energy / (B*S), the same
    scale as 'target_energy'."""
    seq_len, vocab, batch_size = 5, 7, 3
    structure = v1_masked_structure(seq_len, vocab)
    params = initialize_params(structure, rng_key)
    batch = make_v1_token_batch(rng_key, batch_size=batch_size)
    optimizer = optax.sgd(0.1)
    step = make_train_step(structure, optimizer)
    _, _, metrics, state = step(params, optimizer.init(params), batch, rng_key)
    expected = float(graph_energy(state, structure)) / (batch_size * seq_len)
    assert expected > 0.0
    assert abs(float(metrics["energy"]) - expected) < 1e-5


def test_target_free_pc_graph_normalizes_by_batch(rng_key):
    """With no clamped target N = B: the optimizer-facing gradients are the raw
    sums / B and 'energy' is graph_energy / B. The energy comes from the
    internal node h (x -> h -> g): a free terminal node such as g is held at
    its projection with zero energy, and GlobalStateInit keeps h's error
    nonzero (a feedforward-initialized free node sits at its zero-error fixed
    point)."""
    x_node = Linear(shape=(4,), name="x")
    h = Linear(shape=(5,), activation=SigmoidActivation(), name="h")
    g = Linear(shape=(3,), activation=SigmoidActivation(), name="g")
    structure = graph(
        nodes=[x_node, h, g],
        edges=[
            Edge(source=x_node, target=h.slot("in")),
            Edge(source=h, target=g.slot("in")),
        ],
        task_map=TaskMap(x=x_node),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=3),
        graph_state_initializer=GlobalStateInit(),
    )
    params = initialize_params(structure, rng_key)
    batch_size = 6
    batch = {"x": jax.random.normal(rng_key, (batch_size, 4))}
    clamps = build_clamps(batch, structure, clamp_target=True)
    assert grad_denominator(structure, clamps) == batch_size

    state = initialize_graph_state(
        structure, batch_size, rng_key, clamps=clamps, params=params
    )
    state = run_inference(params, state, clamps, structure)
    raw = compute_local_weight_gradients(params, state, structure)
    normalized = pc_weight_gradients(params, state, structure, clamps)
    assert max_param_diff(raw, normalized) > 0.0
    scaled = jax.tree_util.tree_map(lambda g: g / batch_size, raw)
    assert max_param_diff(scaled, normalized) == 0.0

    optimizer = optax.sgd(0.1)
    step = make_train_step(structure, optimizer)
    _, _, metrics, final_state = step(params, optimizer.init(params), batch, rng_key)
    expected = float(graph_energy(final_state, structure)) / batch_size
    assert expected > 0.0
    assert abs(float(metrics["energy"]) - expected) < 1e-5
    assert float(metrics["target_energy"]) == 0.0


def test_grad_denominator_raises_on_empty_clamps():
    structure = classification_structure()
    with pytest.raises(ValueError, match="clamps"):
        grad_denominator(structure, {})


def test_grad_denominator_matches_eval_energy_weights(rng_key):
    """grad_denominator equals the batch total of _internal_energy_fn's
    per-sample weight, for a sequence batch (N = B*S) and a two-target batch
    (N = B * 2): the train and eval 'energy' divide by the same count."""
    cases = [
        (v1_masked_structure(5, 7), make_v1_token_batch(rng_key), 3 * 5),
    ]
    kx, k1, k2 = jax.random.split(rng_key, 3)
    two_target_batch = {
        "x": jax.random.normal(kx, (4, 4)),
        "y": jax.random.randint(k1, (4,), 0, 3),
        "y2": jax.random.randint(k2, (4,), 0, 5),
    }
    cases.append((two_target_structure(), two_target_batch, 4 * 2))
    for structure, batch, expected in cases:
        clamps = build_clamps(batch, structure, clamp_target=True)
        params = initialize_params(structure, rng_key)
        batch_size = batch["x"].shape[0]
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        _, weight = metrics_mod._internal_energy_fn(state, batch, structure)
        assert weight.shape == (batch_size,)
        assert grad_denominator(structure, clamps) == expected
        assert float(jnp.sum(weight)) == expected


def test_one_step_pc_gradients_vs_backprop(rng_key):
    """One InferenceSGD step from the feedforward state moves only the hidden
    latent, by -eta * dE/dz_h (its own error is zero there), so the hidden
    node's local weight gradient is exactly eta times the backprop gradient
    of the target energy, for any eta. The output node's local gradient is
    re-evaluated at the moved hidden latent and differs from backprop by
    O(eta); the deviation shrinks with eta. Both sides divide by N = B.

    The hidden identity is checked at eta = 0.1: the hidden error is formed
    as (mu_h - eta * grad) - mu_h in float32, whose rounding error is
    ulp(mu_h) / (eta * |grad|), about 1e-6 at eta = 0.1 and 1e-3 at
    eta = 1e-4, so smaller eta would measure rounding, not the identity.
    """
    base = classification_structure()
    params = initialize_params(base, rng_key)
    kx, ky = jax.random.split(rng_key)
    batch_size = 4
    batch = {
        "x": jax.random.normal(kx, (batch_size, 6)),
        "y": jax.nn.one_hot(jax.random.randint(ky, (batch_size,), 0, 3), 3),
    }
    clamps = build_clamps(batch, base, clamp_target=True)

    optimizer = optax.sgd(1.0)
    bp_step = make_train_step(base, optimizer, algorithm="backprop")
    new_params, *_ = bp_step(params, optimizer.init(params), batch, rng_key)
    bp_grads = jax.tree_util.tree_map(lambda o, n: o - n, params, new_params)

    def pc_grads(eta):
        structure = with_inference(base, eta_infer=eta, infer_steps=1)
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        state = run_inference(params, state, clamps, structure)
        return pc_weight_gradients(params, state, structure, clamps)

    eta = 0.1
    hidden_ref = jax.tree_util.tree_map(lambda g: eta * g, bp_grads.nodes["h"])
    assert _relative_deviation(pc_grads(eta).nodes["h"], hidden_ref) < 1e-5

    deviations = [
        _relative_deviation(pc_grads(eta).nodes["y"], bp_grads.nodes["y"])
        for eta in (1e-1, 1e-2, 1e-3)
    ]
    for eta, dev in zip((1e-1, 1e-2, 1e-3), deviations):
        assert 0.0 < dev < 10.0 * eta
    assert deviations[0] > deviations[1] > deviations[2]


# ---------------------------------------------------------------------------
# Contract guards
# ---------------------------------------------------------------------------


def test_unknown_algorithm_raises(rng_key):
    structure = classification_structure()
    params = initialize_params(structure, rng_key)
    loader = make_batches(rng_key, n_batches=1)
    with pytest.raises(ValueError, match="algorithm"):
        train(
            params,
            structure,
            loader,
            optax.adam(1e-3),
            {"num_epochs": 1},
            rng_key,
            algorithm="hebbian",
            verbose=False,
        )


def test_backprop_requires_feedforward_init(rng_key):
    x_node = Linear(shape=(6,), name="x")
    y_node = Linear(shape=(3,), activation=SoftmaxActivation(), name="y")
    structure = graph(
        nodes=[x_node, y_node],
        edges=[Edge(source=x_node, target=y_node.slot("in"))],
        task_map=TaskMap(x=x_node, y=y_node),
        inference=InferenceSGD(),
        graph_state_initializer=GlobalStateInit(),
    )
    with pytest.raises(ValueError, match="FeedforwardStateInit"):
        make_train_step(structure, optax.adam(1e-3), algorithm="backprop")


def test_pc_requires_inference(rng_key):
    """algorithm='pc' on a graph built with inference=None fails fast at
    build time, naming the missing prerequisite."""
    x_node = Linear(shape=(6,), name="x")
    y_node = Linear(
        shape=(3,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="y",
    )
    structure = graph(
        nodes=[x_node, y_node],
        edges=[Edge(source=x_node, target=y_node.slot("in"))],
        task_map=TaskMap(x=x_node, y=y_node),
        inference=None,
    )
    with pytest.raises(ValueError, match="inference"):
        make_train_step(structure, optax.adam(1e-3), algorithm="pc")
    params = initialize_params(structure, rng_key)
    loader = make_batches(rng_key, n_batches=1)
    with pytest.raises(ValueError, match="inference"):
        train(
            params,
            structure,
            loader,
            optax.adam(1e-3),
            {"num_epochs": 1},
            rng_key,
            verbose=False,
        )
    # The same graph trains with backprop (feedforward pass only).
    result = train(
        params,
        structure,
        loader,
        optax.adam(1e-3),
        {"num_epochs": 1},
        rng_key,
        algorithm="backprop",
        verbose=False,
    )
    assert max_param_diff(params, result.params) > 0.0


def test_backprop_without_clamped_target_raises(rng_key):
    x_node = Linear(shape=(4,), name="x")
    h = Linear(shape=(5,), activation=SigmoidActivation(), name="h")
    structure = graph(
        nodes=[x_node, h],
        edges=[Edge(source=x_node, target=h.slot("in"))],
        task_map=TaskMap(x=x_node),
        inference=InferenceSGD(),
    )
    params = initialize_params(structure, rng_key)
    step = make_train_step(structure, optax.adam(1e-3), algorithm="backprop")
    with pytest.raises(ValueError, match="target"):
        step(
            params,
            optax.adam(1e-3).init(params),
            {"x": jax.random.normal(rng_key, (4, 4))},
            rng_key,
        )
    # The same guard fires through the train loop, not only the raw step.
    loader = ListLoader([{"x": jax.random.normal(rng_key, (4, 4))}])
    with pytest.raises(ValueError, match="target"):
        train(
            params,
            structure,
            loader,
            optax.adam(1e-3),
            {"num_epochs": 1},
            rng_key,
            algorithm="backprop",
            verbose=False,
        )


def test_wrong_shape_target_raises_actionable_error(rng_key):
    """A float target whose trailing shape mismatches the node raises from
    build_clamps, not as an opaque XLA broadcast error inside the step."""
    structure = classification_structure()  # y expects (batch, 3)
    x = jax.random.normal(rng_key, (4, 6))
    y_bad = jax.nn.one_hot(jnp.zeros(4, dtype=jnp.int32), 4)  # (4, 4)
    with pytest.raises(ValueError, match="target 'y'"):
        build_clamps({"x": x, "y": y_bad}, structure, clamp_target=True)


def test_loader_without_len_raises_actionable_error(rng_key):
    structure = classification_structure()
    params = initialize_params(structure, rng_key)
    batches = (b for b in make_batches(rng_key, n_batches=1))  # no __len__
    with pytest.raises(TypeError, match="len"):
        train(
            params,
            structure,
            batches,
            optax.adam(1e-3),
            {"num_epochs": 1},
            rng_key,
            verbose=False,
        )


def test_mesh_without_data_axis_raises(rng_key):
    structure = classification_structure()
    params = initialize_params(structure, rng_key)
    loader = make_batches(rng_key, n_batches=1)
    mesh = jax.make_mesh((1,), ("batch",))
    with pytest.raises(ValueError, match="'data' axis"):
        make_train_step(structure, optax.adam(1e-3), mesh=mesh)
    with pytest.raises(ValueError, match="'data' axis"):
        train(
            params,
            structure,
            loader,
            optax.adam(1e-3),
            {"num_epochs": 1},
            rng_key,
            mesh=mesh,
            verbose=False,
        )
    with pytest.raises(ValueError, match="'data' axis"):
        evaluate(params, structure, loader, {}, rng_key, mesh=mesh)


def test_verbose_epoch_summary(rng_key, capsys):
    """verbose=True prints the per-epoch summary (tqdm postfix formatting and
    the epoch line are otherwise never executed by the suite)."""
    structure = classification_structure()
    params = initialize_params(structure, rng_key)
    loader = make_batches(rng_key, n_batches=2)
    train(
        params,
        structure,
        loader,
        optax.adam(1e-3),
        {"num_epochs": 1},
        rng_key,
        verbose=True,
    )
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "Epoch 1/1" in text
    assert "energy" in text


@pytest.mark.parametrize("bad_key", ["loss_type", "use_causal_mask"])
def test_retired_config_keys_raise(rng_key, bad_key):
    structure = classification_structure()
    params = initialize_params(structure, rng_key)
    loader = make_batches(rng_key, n_batches=1)
    with pytest.raises(ValueError, match=bad_key):
        train(
            params,
            structure,
            loader,
            optax.adam(1e-3),
            {"num_epochs": 1, bad_key: True},
            rng_key,
            verbose=False,
        )
    with pytest.raises(ValueError, match=bad_key):
        evaluate(params, structure, loader, {bad_key: True}, rng_key)


def test_pc_energy_decreases_over_epochs(rng_key):
    structure = classification_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = make_batches(rng_key, n_batches=2)
    result = train(
        params,
        structure,
        loader,
        optax.adam(1e-2),
        {"num_epochs": 5},
        train_key,
        verbose=False,
    )
    energies = [e["energy"] for e in result.epoch_results]
    assert energies[-1] < energies[0]


def test_train_requires_num_epochs(rng_key):
    """A missing num_epochs raises instead of silently training a default."""
    structure = classification_structure()
    params = initialize_params(structure, rng_key)
    loader = make_batches(rng_key, n_batches=1)
    with pytest.raises(ValueError, match="num_epochs"):
        train(params, structure, loader, optax.adam(1e-3), {}, rng_key, verbose=False)


@pytest.mark.parametrize("algorithm", ["pc", "backprop"])
def test_fractional_epochs(rng_key, algorithm):
    structure = classification_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = make_batches(rng_key, n_batches=4)
    result = train(
        params,
        structure,
        loader,
        optax.adam(1e-3),
        {"num_epochs": 1.5},
        train_key,
        algorithm=algorithm,
        verbose=False,
    )
    assert len(result.iter_results) == 2
    assert len(result.iter_results[0]) == 4
    assert len(result.iter_results[1]) == 2  # round(0.5 * 4)
    assert result.step == 6
    # The partial epoch's entry is the mean over the batches actually run.
    partial = result.iter_results[1]
    expected = sum(m["energy"] for m in partial) / len(partial)
    assert abs(result.epoch_results[1]["energy"] - expected) < 1e-6


def test_fractional_tail_rounding_to_zero_batches_is_dropped(rng_key):
    """num_epochs=1.1 on 4 batches rounds the tail to 0 batches: the tail is
    dropped — no empty epoch entry, no callback invoked on empty metrics."""
    structure = classification_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = make_batches(rng_key, n_batches=4)
    calls = []
    result = train(
        params,
        structure,
        loader,
        optax.adam(1e-3),
        {"num_epochs": 1.1},
        train_key,
        epoch_callback=lambda ctx: calls.append(ctx.epoch_idx),
        verbose=False,
    )
    assert result.step == 4
    assert len(result.iter_results) == 1
    assert len(result.epoch_results) == 1
    assert calls == [0]


@pytest.mark.parametrize("algorithm", ["pc", "backprop"])
def test_epoch_context_fields_and_callback_replacement(rng_key, algorithm):
    structure = classification_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = make_batches(rng_key, n_batches=2)
    seen = []

    def epoch_callback(ctx):
        assert isinstance(ctx, EpochContext)
        assert ctx.structure is structure
        assert set(ctx.metrics) == {"energy", "target_energy"}
        assert isinstance(ctx.metrics["energy"], float)
        seen.append((ctx.epoch_idx, ctx.step))
        return {"replaced": ctx.epoch_idx}

    result = train(
        params,
        structure,
        loader,
        optax.adam(1e-3),
        {"num_epochs": 2},
        train_key,
        algorithm=algorithm,
        start_epoch=5,
        epoch_callback=epoch_callback,
        verbose=False,
    )
    assert seen == [(5, 2), (6, 4)]
    assert result.epoch_results == [{"replaced": 5}, {"replaced": 6}]


def test_callback_exceptions_propagate(rng_key):
    """Tuner pruning is exception-based: no swallowing allowed."""
    structure = classification_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = make_batches(rng_key, n_batches=1)

    class Prune(Exception):
        pass

    def epoch_callback(ctx):
        raise Prune()

    with pytest.raises(Prune):
        train(
            params,
            structure,
            loader,
            optax.adam(1e-3),
            {"num_epochs": 1},
            train_key,
            epoch_callback=epoch_callback,
            verbose=False,
        )

    def iter_callback(epoch_idx, batch_idx, metrics):
        raise Prune()

    with pytest.raises(Prune):
        train(
            params,
            structure,
            loader,
            optax.adam(1e-3),
            {"num_epochs": 1},
            train_key,
            iter_callback=iter_callback,
            verbose=False,
        )


def test_iter_callback_receives_floats_and_replaces(rng_key):
    structure = classification_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = make_batches(rng_key, n_batches=2)
    calls = []

    def iter_callback(epoch_idx, batch_idx, metrics):
        assert isinstance(metrics["energy"], float)
        assert isinstance(metrics["target_energy"], float)
        calls.append((epoch_idx, batch_idx))
        return batch_idx  # replaces the stored entry

    result = train(
        params,
        structure,
        loader,
        optax.adam(1e-3),
        {"num_epochs": 1},
        train_key,
        iter_callback=iter_callback,
        verbose=False,
    )
    assert calls == [(0, 0), (0, 1)]
    assert result.iter_results == [[0, 1]]


def test_step_metrics_keys(rng_key):
    structure = classification_structure()
    params = initialize_params(structure, rng_key)
    loader = make_batches(rng_key, n_batches=1)
    step = make_train_step(structure, optax.adam(1e-3))
    batch = next(iter(loader))
    _, _, metrics, final_state = step(
        params, optax.adam(1e-3).init(params), batch, rng_key
    )
    assert set(metrics) == {"energy", "target_energy"}
    assert final_state.batch_size == batch["x"].shape[0]


# ---------------------------------------------------------------------------
# graph_energy subset selection
# ---------------------------------------------------------------------------


def test_graph_energy_subset_and_default(rng_key):
    structure = classification_structure()
    params = initialize_params(structure, rng_key)
    batch = next(iter(make_batches(rng_key, n_batches=1)))
    clamps = build_clamps(batch, structure, clamp_target=True)
    state = initialize_graph_state(structure, 4, rng_key, clamps=clamps, params=params)
    state = run_inference(params, state, clamps, structure)

    total = float(graph_energy(state, structure))
    by_parts = float(graph_energy(state, structure, node_names=("h",))) + float(
        graph_energy(state, structure, node_names=("y",))
    )
    assert abs(total - by_parts) < 1e-5
    # source node contributes nothing to the default set
    assert float(graph_energy(state, structure, node_names=("x",))) == 0.0
    with pytest.raises(ValueError, match="unknown"):
        graph_energy(state, structure, node_names=("nope",))


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def test_generate_shape_dtype_prefix(rng_key):
    structure = sequence_structure()
    params = initialize_params(structure, rng_key)
    prompt = jnp.array([1, 2, 3], dtype=jnp.int32)
    out = generate(params, structure, prompt, max_new_tokens=4, rng_key=rng_key)
    assert out.shape == (7,)
    assert jnp.issubdtype(out.dtype, jnp.integer)
    assert jnp.array_equal(out[:3], prompt)
    assert bool(jnp.all(out >= 0)) and bool(jnp.all(out < 11))

    batched = jnp.stack([prompt, prompt + 1])
    out2 = generate(
        params, structure, batched, max_new_tokens=2, rng_key=rng_key, top_k=3
    )
    assert out2.shape == (2, 5)
    assert jnp.array_equal(out2[:, :3], batched)


def test_generate_prompt_truncation(rng_key):
    """A prompt longer than the model's seq_len keeps only the trailing
    window as context, and the full prompt survives in the output."""
    structure = sequence_structure()  # seq_len 6
    params = initialize_params(structure, rng_key)
    prompt = (jnp.arange(10) % 11).astype(jnp.int32)
    out = generate(params, structure, prompt, max_new_tokens=2, rng_key=rng_key)
    assert out.shape == (12,)
    assert jnp.array_equal(out[:10], prompt)


def test_generate_top_k_restricts_support(rng_key):
    """top_k=1 leaves a single admissible token per step, so sampling is
    deterministic: two different keys must produce identical tokens."""
    structure = sequence_structure()
    params = initialize_params(structure, rng_key)
    prompt = jnp.array([1, 2, 3], dtype=jnp.int32)
    out_a = generate(
        params,
        structure,
        prompt,
        max_new_tokens=4,
        rng_key=jax.random.PRNGKey(0),
        top_k=1,
    )
    out_b = generate(
        params,
        structure,
        prompt,
        max_new_tokens=4,
        rng_key=jax.random.PRNGKey(1),
        top_k=1,
    )
    assert jnp.array_equal(out_a, out_b)


def test_generate_top_p_and_temperature(rng_key):
    """Exercises the nucleus-filter and temperature paths; a vanishing top_p
    keeps only the most probable token, so sampling turns deterministic."""
    structure = sequence_structure()
    params = initialize_params(structure, rng_key)
    prompt = jnp.array([1, 2, 3], dtype=jnp.int32)
    out = generate(
        params,
        structure,
        prompt,
        max_new_tokens=3,
        rng_key=rng_key,
        temperature=0.7,
        top_p=0.9,
    )
    assert out.shape == (6,)
    assert bool(jnp.all(out >= 0)) and bool(jnp.all(out < 11))

    out_a = generate(
        params,
        structure,
        prompt,
        max_new_tokens=3,
        rng_key=jax.random.PRNGKey(0),
        top_p=1e-6,
    )
    out_b = generate(
        params,
        structure,
        prompt,
        max_new_tokens=3,
        rng_key=jax.random.PRNGKey(1),
        top_p=1e-6,
    )
    assert jnp.array_equal(out_a, out_b)


def test_generate_backprop_algorithm_and_one_hot_input(rng_key):
    """A graph built with inference=None generates via the feedforward pass
    (algorithm='backprop'); its 2D input node exercises the one-hot input
    branch. The default algorithm='pc' fails fast, naming the missing
    inference — previously an opaque AttributeError inside run_inference."""
    seq_len, vocab = 5, 7
    x_node = Linear(shape=(seq_len, vocab), name="inp")
    y_node = Linear(
        shape=(seq_len, vocab),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="out",
    )
    structure = graph(
        nodes=[x_node, y_node],
        edges=[Edge(source=x_node, target=y_node.slot("in"))],
        task_map=TaskMap(x=x_node, y=y_node),
        inference=None,
    )
    params = initialize_params(structure, rng_key)
    prompt = jnp.array([1, 2], dtype=jnp.int32)
    out = generate(
        params,
        structure,
        prompt,
        max_new_tokens=3,
        rng_key=rng_key,
        algorithm="backprop",
    )
    assert out.shape == (5,)
    assert jnp.array_equal(out[:2], prompt)
    assert bool(jnp.all(out >= 0)) and bool(jnp.all(out < vocab))

    with pytest.raises(ValueError, match="inference"):
        generate(params, structure, prompt, max_new_tokens=1, rng_key=rng_key)


def test_generate_requires_x_and_y_task_keys(rng_key):
    x_node = Linear(shape=(4,), name="x")
    h = Linear(shape=(5,), activation=SigmoidActivation(), name="h")
    structure = graph(
        nodes=[x_node, h],
        edges=[Edge(source=x_node, target=h.slot("in"))],
        task_map=TaskMap(x=x_node),
        inference=InferenceSGD(),
    )
    params = initialize_params(structure, rng_key)
    with pytest.raises(ValueError, match="task_map"):
        generate(
            params,
            structure,
            jnp.array([1, 2], dtype=jnp.int32),
            max_new_tokens=1,
            rng_key=rng_key,
        )
