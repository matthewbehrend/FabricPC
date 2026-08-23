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
from typing import Iterator, List

import jax
import jax.numpy as jnp
import optax
import pytest

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
    make_train_step,
    metrics as metrics_mod,
    train,
)

PARITY_TOL = 1e-12


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class ListLoader:
    """Deterministic loader: same batches every time it is iterated."""

    def __init__(self, batches):
        self._batches = batches

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator:
        return iter(self._batches)


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


def classification_structure(output_energy=None, output_activation=None):
    """3-node Linear chain; the default FeedforwardStateInit satisfies both
    algorithms, and InferenceSGD drives PC settling."""
    x = Linear(shape=(6,), name="x")
    h = Linear(shape=(8,), activation=SigmoidActivation(), name="h")
    y = Linear(
        shape=(3,),
        activation=output_activation or SoftmaxActivation(),
        energy=output_energy or CrossEntropyEnergy(),
        name="y",
    )
    return graph(
        nodes=[x, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=10),
    )


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


def max_param_diff(a, b) -> float:
    diffs = jax.tree_util.tree_map(lambda p, q: jnp.max(jnp.abs(p - q)), a, b)
    return float(jax.tree_util.tree_reduce(jnp.maximum, diffs, jnp.array(0.0)))


# ---------------------------------------------------------------------------
# PC parity vs a hand-rolled reference step
# ---------------------------------------------------------------------------


def test_pc_parity_hand_rolled_reference(rng_key):
    """train(algorithm='pc') matches the composed primitives under the
    fold_in stream: clamps -> init -> inference -> local grads -> optax."""
    structure = classification_structure()
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
        grads = compute_local_weight_gradients(p, state, structure)
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
        # CrossEntropyEnergy: -sum y*log(clip(mu, 1e-7, 1)), summed then /batch
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
    """A GaussianEnergy output's backprop objective is 0.5*precision*SSE/batch,
    not an element-mean MSE."""
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


def test_resume_is_bitwise_identical(rng_key):
    structure = classification_structure()
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


def test_caller_params_survive_train(rng_key):
    """The internal step donates buffers; the caller's params must stay valid."""
    structure = classification_structure()
    params_key, train_key = jax.random.split(rng_key)
    params = initialize_params(structure, params_key)
    loader = make_batches(rng_key, n_batches=1)
    optimizer = optax.adam(1e-2)
    train(
        params,
        structure,
        loader,
        optimizer,
        {"num_epochs": 1},
        train_key,
        verbose=False,
    )
    # A second call on the same arrays must not hit deleted buffers.
    train(
        params,
        structure,
        loader,
        optimizer,
        {"num_epochs": 1},
        train_key,
        verbose=False,
    )


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


def test_fractional_epochs(rng_key):
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


def test_epoch_context_fields_and_callback_replacement(rng_key):
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
