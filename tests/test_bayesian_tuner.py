"""Tests for the two-phase BayesianTuner: perplexity objective, the scale-free
energy divergence guard, and Hyperband pruning wiring."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import optuna
import pytest

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.models import create_deep_transformer
from fabricpc.graph_initialization import initialize_params
from fabricpc.training import EpochContext, IterContext, TrainResult
from fabricpc.tuning.bayesian_tuner import BayesianTuner
import fabricpc.tuning.bayesian_tuner as tuner_mod


# tiny fixtures
def _tiny_trial_model(config, rng_key):
    """2-tuple return -> tuner uses the loaders passed to BayesianTuner."""
    structure = create_deep_transformer(
        depth=config.get("depth", 1),
        embed_dim=8,
        num_heads=2,
        mlp_dim=16,
        seq_len=config.get("seq_len", 4),
        vocab_size=config["vocab_size"],
        inference=InferenceSGDNormClip(
            eta_infer=config.get("eta_infer", 0.1),
            infer_steps=config.get("infer_steps", 3),
            max_norm=5.0,
        ),
        weight_init={"type": "normal", "std": config.get("weight_init_std", 0.02)},
    )
    params = initialize_params(structure, rng_key)
    return params, structure


def _tiny_loader(vocab_size=10, n_batches=1, batch=2, seq=4, seed=0):
    rng = np.random.default_rng(seed)
    return [
        {
            "x": rng.integers(0, vocab_size, size=(batch, seq)).astype(np.int32),
            "y": rng.integers(0, vocab_size, size=(batch, seq)).astype(np.int32),
        }
        for _ in range(n_batches)
    ]


def _fake_train(energies, ces):
    """Stand-in for train that drives both callbacks with their contexts: per
    epoch one iteration callback at batch 49 (the tuner prints every 50th
    batch), then the epoch callback."""

    def fake(
        params,
        structure,
        loader,
        optimizer,
        config,
        rng,
        verbose=False,
        iter_callback=None,
        epoch_callback=None,
        **kwargs,
    ):
        opt_state = optimizer.init(params)
        algorithm = kwargs.get("algorithm", "pc")
        for i, (e, ce) in enumerate(zip(energies, ces)):
            epoch_key = jax.random.fold_in(rng, i)
            metrics = {"energy": e, "target_energy": ce}
            if iter_callback is not None:
                iter_callback(
                    IterContext(
                        epoch_idx=i,
                        batch_idx=49,
                        step=i + 1,
                        params=params,
                        opt_state=opt_state,
                        state=None,
                        structure=structure,
                        config=config,
                        algorithm=algorithm,
                        rng_key=rng,
                        epoch_key=epoch_key,
                        batch_key=jax.random.fold_in(epoch_key, 49),
                        batch={},
                        metrics=metrics,
                    )
                )
            if epoch_callback is not None:
                epoch_callback(
                    EpochContext(
                        epoch_idx=i,
                        step=i + 1,
                        params=params,
                        opt_state=opt_state,
                        structure=structure,
                        config=config,
                        algorithm=algorithm,
                        rng_key=rng,
                        epoch_key=epoch_key,
                        metrics=metrics,
                    )
                )
        return TrainResult(
            params=params,
            opt_state=opt_state,
            step=len(energies),
            iter_results=[],
            epoch_results=[],
        )

    return fake


def _make_tuner(tmp_path, trial_model=_tiny_trial_model, **kwargs):
    return BayesianTuner(
        train_loader=_tiny_loader(seed=0),
        val_loader=_tiny_loader(seed=1),
        trial_model=trial_model,
        base_config={"seq_len": 4, "vocab_size": 10, "num_epochs": 1, "use_bpe": False},
        study_name="test_tuner",
        storage=None,
        log_file=str(tmp_path / "log.txt"),
        divergence_rel_tol=0.5,
        **kwargs,
    )


def _run_one(tuner, config):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: tuner._run_trial(t, config, 1)[0], n_trials=1)
    return study.trials[0]


# tests
def test_run_trial_returns_finite_perplexity(tmp_path):
    """End-to-end (real training + eval) on a tiny model returns a perplexity."""
    tuner = _make_tuner(tmp_path)
    config = {
        **tuner.base_config,
        "depth": 1,
        "eta_infer": 0.1,
        "infer_steps": 3,
        "lr": 1e-3,
        "weight_init_std": 0.02,
    }
    t = _run_one(tuner, config)
    assert t.state == optuna.trial.TrialState.COMPLETE
    assert t.value is not None and np.isfinite(t.value) and t.value >= 1.0


def test_divergence_guard_prunes(tmp_path, monkeypatch):
    """Energy that rises above its best epoch by > rel_tol is pruned."""
    tuner = _make_tuner(tmp_path)
    monkeypatch.setattr(tuner_mod, "train", _fake_train([100.0, 500.0], [2.0, 2.0]))
    config = {**tuner.base_config, "depth": 1, "lr": 1e-3}
    t = _run_one(tuner, config)
    assert t.state == optuna.trial.TrialState.PRUNED
    assert "diverged" in t.user_attrs.get("prune_reason", "").lower()


def test_nonfinite_energy_prunes(tmp_path, monkeypatch):
    tuner = _make_tuner(tmp_path)
    monkeypatch.setattr(tuner_mod, "train", _fake_train([float("inf")], [2.0]))
    config = {**tuner.base_config, "depth": 1, "lr": 1e-3}
    t = _run_one(tuner, config)
    assert t.state == optuna.trial.TrialState.PRUNED
    assert "non-finite" in t.user_attrs.get("prune_reason", "").lower()


def test_four_tuple_trial_model_loaders_used(tmp_path, monkeypatch):
    """A trial_model returning (params, structure, train_loader, val_loader)
    has those loaders used for training/eval, not the tuner defaults."""
    trial_train = _tiny_loader(seed=2)
    trial_val = _tiny_loader(seed=3)

    def four_tuple_model(config, rng_key):
        params, structure = _tiny_trial_model(config, rng_key)
        return params, structure, trial_train, trial_val

    seen = {}

    inner_fake = _fake_train([100.0], [2.0])

    def fake_train(params, structure, loader, *args, **kwargs):
        seen["train_loader"] = loader
        return inner_fake(params, structure, loader, *args, **kwargs)

    def fake_eval(params, structure, loader, config, rng, **kwargs):
        seen["val_loader"] = loader
        return {"perplexity": 7.0, "cross_entropy": float(np.log(7.0))}

    monkeypatch.setattr(tuner_mod, "train", fake_train)
    monkeypatch.setattr(tuner_mod, "evaluate", fake_eval)

    tuner = _make_tuner(tmp_path, trial_model=four_tuple_model)
    config = {**tuner.base_config, "depth": 1, "lr": 1e-3}
    t = _run_one(tuner, config)
    assert t.state == optuna.trial.TrialState.COMPLETE
    assert seen["train_loader"] is trial_train
    assert seen["val_loader"] is trial_val
    assert t.value == pytest.approx(7.0)


def test_missing_perplexity_raises(tmp_path, monkeypatch):
    """A trial graph without a CrossEntropyEnergy target yields no
    'perplexity' eval key; the tuner must raise, not score inf silently."""
    monkeypatch.setattr(tuner_mod, "train", _fake_train([100.0], [2.0]))

    def fake_eval(params, structure, loader, config, rng, **kwargs):
        return {"target_energy": 1.0, "accuracy": 0.5, "energy": 1.0}

    monkeypatch.setattr(tuner_mod, "evaluate", fake_eval)
    tuner = _make_tuner(tmp_path)
    config = {**tuner.base_config, "depth": 1, "lr": 1e-3}
    study = optuna.create_study(direction="minimize")
    with pytest.raises(ValueError, match="perplexity"):
        study.optimize(lambda t: tuner._run_trial(t, config, 1)[0], n_trials=1)


def test_algorithm_threads_through_train_and_eval(tmp_path, monkeypatch):
    """BayesianTuner(algorithm=...) reaches every trial's train and evaluate
    call (the tuner previously fixed PC silently)."""
    seen = {}
    inner_fake = _fake_train([100.0], [2.0])

    def fake_train(params, structure, loader, *args, **kwargs):
        seen["train_algorithm"] = kwargs.get("algorithm")
        return inner_fake(params, structure, loader, *args, **kwargs)

    def fake_eval(params, structure, loader, config, rng, **kwargs):
        seen["eval_algorithm"] = kwargs.get("algorithm")
        return {"perplexity": 7.0, "cross_entropy": float(np.log(7.0))}

    monkeypatch.setattr(tuner_mod, "train", fake_train)
    monkeypatch.setattr(tuner_mod, "evaluate", fake_eval)
    tuner = _make_tuner(tmp_path, algorithm="backprop")
    config = {**tuner.base_config, "depth": 1, "lr": 1e-3}
    t = _run_one(tuner, config)
    assert t.state == optuna.trial.TrialState.COMPLETE
    assert seen["train_algorithm"] == "backprop"
    assert seen["eval_algorithm"] == "backprop"


def _p1_space(trial):
    return {
        "depth": trial.suggest_int("depth", 1, 1),
        "eta_infer": trial.suggest_float("eta_infer", 0.05, 0.1),
        "infer_steps": trial.suggest_int("infer_steps", 3, 3),
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "weight_init_std": trial.suggest_float("weight_init_std", 0.01, 0.05, log=True),
    }


def _p2_space(trial, best):
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "eta_infer": trial.suggest_float("eta_infer", 0.05, 0.1),
        "infer_steps": trial.suggest_int("infer_steps", 3, 3),
    }


def test_both_phases_return_perplexity(tmp_path, monkeypatch):
    """tune() returns perplexity-keyed results for both phases (finding 3 rename)."""
    tuner = _make_tuner(tmp_path)
    monkeypatch.setattr(tuner_mod, "train", _fake_train([100.0, 95.0], [2.0, 1.5]))
    results = tuner.tune(
        phase1_search_space=_p1_space,
        phase2_search_space=_p2_space,
        n_trials_phase1=1,
        n_trials_phase2=1,
        save_best_to=str(tmp_path / "best.txt"),
    )
    assert "phase1_best_ppl" in results and np.isfinite(results["phase1_best_ppl"])
    assert "phase2_best_ppl" in results and np.isfinite(results["phase2_best_ppl"])


def test_iter_callback_prints_progress_when_verbose(tmp_path, monkeypatch, capsys):
    """The tuner's iteration callback reads IterContext fields. A signature
    break would not fail loudly: train's TypeError is caught by the tuner and
    every trial is pruned as 'failed during training'."""
    tuner = _make_tuner(tmp_path, verbose=True)
    monkeypatch.setattr(tuner_mod, "train", _fake_train([100.0], [2.0]))
    monkeypatch.setattr(
        tuner_mod,
        "evaluate",
        lambda *a, **k: {"perplexity": 7.0, "cross_entropy": 1.95, "accuracy": 0.1},
    )
    config = {
        **tuner.base_config,
        "depth": 1,
        "eta_infer": 0.1,
        "infer_steps": 3,
        "lr": 1e-3,
        "weight_init_std": 0.02,
    }
    t = _run_one(tuner, config)
    assert t.state == optuna.trial.TrialState.COMPLETE
    assert "Batch 50 | Energy: 100.0000" in capsys.readouterr().out
