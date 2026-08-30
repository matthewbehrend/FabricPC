import jax
import optuna
import os
import random
import numpy as np
import time
from datetime import datetime
import optax as _optax
from pathlib import Path
from typing import Callable, Any, Dict, Tuple, Optional

from fabricpc.training import train, evaluate
from fabricpc.core.types import GraphParams, GraphStructure


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class BayesianTuner:
    """
    Two-phase Bayesian Hyperparameter Tuner using Optuna for FabricPC models.

    Phase 1 — Architecture search: coarse search over architecture + training
        params, minimizing validation perplexity. Energy is used only as a
        scale-free divergence guard to prune unstable trials, never as the
        objective.
    Phase 2 — Continuous fine-tuning: fix the Phase 1 architecture and refine
        lr / eta_infer / infer_steps, also minimizing validation perplexity.

    Both phases score trials by validation perplexity, which ``evaluate``
    reports only when the target node's energy functional is
    ``CrossEntropyEnergy`` — a trial on a graph without one raises instead
    of silently scoring ``inf``. ``algorithm`` selects the learning
    algorithm for every trial's ``train``/``evaluate`` call
    (``"pc"`` default, ``"backprop"`` supported).
    """

    def __init__(
        self,
        train_loader: Any,
        val_loader: Any,
        trial_model: Callable[
            [Dict[str, Any], jax.Array], Tuple[GraphParams, GraphStructure]
        ],
        base_config: Dict[str, Any],
        study_name: str = "fabricpc_tuning",
        storage=None,
        log_file: Optional[str] = "tuning_results.txt",
        divergence_rel_tol: float = 0.5,
        verbose: bool = False,
        algorithm: str = "pc",
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.trial_model = trial_model
        self.base_config = base_config
        self.study_name = study_name
        self.storage = storage
        self.log_file = log_file
        self.divergence_rel_tol = divergence_rel_tol
        self.verbose = verbose
        self.algorithm = algorithm

        if log_file:
            os.makedirs(
                os.path.dirname(log_file) if os.path.dirname(log_file) else ".",
                exist_ok=True,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_trial(
        self,
        trial: optuna.Trial,
        config: Dict[str, Any],
        phase: int,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Run a single train + eval pass and return (perplexity, metrics).

        Both phases optimize the same predictive metric - validation
        perplexity. Energy is not the objective; it is only a scale-free
        divergence guard (a trial is pruned if its training energy is
        non-finite or rises well above its best epoch).
        """
        current_seed = 42 + trial.number
        set_seed(current_seed)
        rng_key = jax.random.PRNGKey(current_seed)
        model_key, train_key, eval_key = jax.random.split(rng_key, 3)

        try:
            result = self.trial_model(config, model_key)
            if len(result) == 4:
                params, structure, train_loader, val_loader = result
            else:
                params, structure = result
                train_loader = self.train_loader
                val_loader = self.val_loader
        except Exception as e:
            print(f"  Trial {trial.number} pruned, model creation failed: {e}")
            raise optuna.TrialPruned()

        optimizer = _optax.adam(config.get("lr", 1e-3))
        train_config = dict(config)

        def iter_callback(epoch_idx, batch_idx, metrics):
            if self.verbose and (batch_idx + 1) % 50 == 0:
                print(
                    f"  [Phase {phase}] Trial {trial.number} | "
                    f"Epoch {epoch_idx + 1} | Batch {batch_idx + 1} | "
                    f"Energy: {metrics['energy']:.4f}"
                )

        # Per-epoch mean energy, accumulated by epoch_callback for the divergence
        # guard so a diverging trial is stopped mid-training.
        trial_energy_means = []

        def epoch_callback(ctx):
            # 1) Scale-free divergence guard, applied each epoch. Energy is not
            #    comparable across architectures, so we prune only on size-free
            #    instability: non-finite energy, or energy risen above its best
            #    epoch by divergence_rel_tol. Running it here stops a blowing-up
            #    trial early instead of wasting the full budget.
            energy = ctx.metrics.get("energy")
            if energy is not None:
                if not np.isfinite(energy):
                    reason = f"Non-finite energy at epoch {ctx.epoch_idx + 1}"
                    trial.set_user_attr("prune_reason", reason)
                    print(f"  Trial {trial.number} pruned — {reason}")
                    raise optuna.TrialPruned()
                trial_energy_means.append(float(energy))
                baseline = min(trial_energy_means)
                if float(energy) > baseline * (1.0 + self.divergence_rel_tol):
                    reason = (
                        f"Energy diverged at epoch {ctx.epoch_idx + 1}: "
                        f"rose to {energy:.4f} from a low of {baseline:.4f}"
                    )
                    trial.set_user_attr("prune_reason", reason)
                    print(f"  Trial {trial.number} pruned — {reason}")
                    raise optuna.TrialPruned()

            # 2) Report training perplexity for the study's pruner (Hyperband in
            #    Phase 1) so stable but underperforming trials are halved early.
            #    Train PPL is free: the clamped target node's energy under
            #    CrossEntropyEnergy is the teacher-forced cross-entropy, so
            #    exp(target_energy) is the training perplexity — and, unlike
            #    energy, it is comparable across architectures.
            target_energy = ctx.metrics.get("target_energy")
            train_ppl = (
                float(np.exp(target_energy))
                if target_energy is not None
                else float("inf")
            )
            if not np.isfinite(train_ppl):
                train_ppl = 1e9  # keep reports finite; diverged trials rank worst
            trial.report(train_ppl, step=ctx.epoch_idx + 1)
            if trial.should_prune():
                # Surface the comparison that justified the prune: this trial's
                # train PPL at this epoch vs the median of completed trials at
                # the same epoch (the bar Hyperband/Median pruning compares to).
                completed = trial.study.get_trials(
                    deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
                )
                peers = [
                    t.intermediate_values[ctx.epoch_idx + 1]
                    for t in completed
                    if (ctx.epoch_idx + 1) in t.intermediate_values
                ]
                bar = (
                    f"above the epoch-{ctx.epoch_idx + 1} median of "
                    f"{float(np.median(peers)):.4f} over {len(peers)} trials"
                    if peers
                    else "below the pruner's rung threshold"
                )
                reason = (
                    f"Pruned by {type(trial.study.pruner).__name__} at epoch "
                    f"{ctx.epoch_idx + 1}: train PPL {train_ppl:.4f} {bar}"
                )
                trial.set_user_attr("prune_reason", reason)
                print(f"  Trial {trial.number} — {reason}")
                raise optuna.TrialPruned()
            return train_ppl

        try:
            result = train(
                params,
                structure,
                train_loader,
                optimizer,
                train_config,
                train_key,
                algorithm=self.algorithm,
                verbose=False,
                iter_callback=iter_callback,
                epoch_callback=epoch_callback,
            )
            trained_params = result.params
        except optuna.TrialPruned:
            raise  # pruned mid-training by the epoch report above
        except Exception as e:
            print(f"  Trial {trial.number} failed during training: {e}")
            raise optuna.TrialPruned()

        # The per-epoch divergence guard in epoch_callback already stops any
        # unstable trial mid-training, so a trial reaching here trained stably.
        # Use its last epoch's mean energy as a diagnostic only.
        avg_energy = trial_energy_means[-1] if trial_energy_means else float("inf")

        try:
            metrics = evaluate(
                trained_params,
                structure,
                val_loader,
                train_config,
                eval_key,
                algorithm=self.algorithm,
            )
        except Exception as e:
            print(f"  Trial {trial.number} failed during eval: {e}")
            raise optuna.TrialPruned()

        # evaluate() already returns an "energy" key (eval internal energy);
        # keep the training diagnostic under its own name.
        metrics["train_energy"] = avg_energy

        # Both phases optimize the same predictive metric: validation perplexity.
        # The energy above is only a stability diagnostic/guard, never the score.
        if "perplexity" not in metrics:
            raise ValueError(
                "BayesianTuner scores trials by validation perplexity, which "
                "evaluate() reports only when the target node's energy "
                "functional is CrossEntropyEnergy; this trial's graph has no "
                "such target. Use a CrossEntropyEnergy output node, or score "
                "trials yourself with a custom Optuna objective."
            )
        return metrics["perplexity"], metrics

    def _log(
        self,
        phase: int,
        trial_number: int,
        duration: float,
        score: float,
        metrics: Dict,
        config: Dict,
    ):
        if not self.log_file:
            return
        line = (
            f"[Phase {phase}] Trial {trial_number:<4} | {duration:<7.1f}s | "
            f"Energy: {metrics.get('train_energy', 0.0):<10.4f} | "
            f"PPL: {metrics.get('perplexity', 0.0):<10.4f} | "
            f"Loss: {metrics.get('cross_entropy', 0.0):<8.4f} | "
            f"LR: {config.get('lr', 'N/A')} | "
            f"Embed: {config.get('embed_dim', 'N/A')} | "
            f"MLP: {config.get('mlp_dim', 'N/A')} | "
            f"Heads: {config.get('num_heads', 'N/A')} | "
            f"Depth: {config.get('depth', 'N/A')} | "
            f"Infer: {config.get('infer_steps', 'N/A')} | "
            f"Eta: {config.get('eta_infer', 'N/A')}\n"
        )
        with open(self.log_file, "a") as f:
            f.write(line)
        print(
            f"  → Score: {score:.4f} | PPL: {metrics.get('perplexity', 0.0):.4f} | Energy: {metrics.get('train_energy', 0.0):.4f}"
        )

    # ------------------------------------------------------------------
    # Phase 1 — architecture search, minimize perplexity
    # ------------------------------------------------------------------

    def tune_phase1(
        self,
        n_trials: int,
        search_space: Callable[[optuna.Trial], Dict[str, Any]],
    ) -> optuna.Study:
        study = optuna.create_study(
            study_name=f"{self.study_name}_phase1_arch",
            storage=self.storage,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1,
                # Resource unit = one training epoch; trials report once per
                # epoch (see _run_trial), so the budget is the epoch count.
                max_resource=int(np.ceil(self.base_config.get("num_epochs", 10))),
                reduction_factor=2,
            ),
        )

        def objective(trial):
            sampled = search_space(trial)
            config = {**self.base_config, **sampled}
            print(f"\n[Phase 1] Trial {trial.number} | {sampled}")
            start = time.time()
            score, metrics = self._run_trial(trial, config, phase=1)
            self._log(1, trial.number, time.time() - start, score, metrics, config)
            return score

        study.optimize(objective, n_trials=n_trials)
        return study

    # ------------------------------------------------------------------
    # Phase 2 — continuous fine-tuning, minimize perplexity
    # ------------------------------------------------------------------

    def tune_phase2(
        self,
        n_trials: int,
        best_params: Dict[str, Any],
        search_space: Callable[[optuna.Trial, Dict], Dict[str, Any]],
    ) -> optuna.Study:
        study = optuna.create_study(
            study_name=f"{self.study_name}_phase2_ppl",
            storage=self.storage,
            direction="minimize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=3,
                multivariate=True,
                group=True,
            ),
        )

        def objective(trial):
            continuous = search_space(trial, best_params)
            config = {**self.base_config, **best_params, **continuous}
            print(f"\n[Phase 2] Trial {trial.number} | fine-tuning: {continuous}")
            start = time.time()
            score, metrics = self._run_trial(trial, config, phase=2)
            self._log(2, trial.number, time.time() - start, score, metrics, config)
            return score

        study.optimize(objective, n_trials=n_trials)
        return study

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def tune(
        self,
        phase1_search_space: Callable[[optuna.Trial], Dict[str, Any]],
        phase2_search_space: Callable[[optuna.Trial, Dict], Dict[str, Any]],
        n_trials_phase1: int = 30,
        n_trials_phase2: int = 20,
        save_best_to: Optional[str] = "tuning/best_hyperparameters.txt",
    ) -> Dict[str, Any]:
        """
        Run the full two-phase tuning pipeline.

        Phase 1: Coarse search over architecture + all params, minimizing
            perplexity; prune diverging trials via a scale-free energy guard.
        Phase 2: Fix architecture, fine-tune lr/eta_infer/infer_steps, minimize perplexity.

        Returns a summary dict with best params from both phases.
        """
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write("=" * 120 + "\n")
                f.write(
                    f"RUN STARTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write("PHASE 1 — Architecture Search (minimize perplexity)\n")
                f.write("=" * 120 + "\n")

        print("\n" + "=" * 60)
        print("PHASE 1: Architecture search — minimizing perplexity")
        print("=" * 60)

        study1 = self.tune_phase1(n_trials_phase1, phase1_search_space)

        # Study.best_trial raises ValueError when no trial completed, so check
        # the trial states directly.
        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study1.trials):
            print("Phase 1 produced no successful trials.")
            return {}

        best_phase1_ppl = study1.best_value
        best_params = study1.best_params
        print(f"\nPhase 1 complete — Best perplexity: {best_phase1_ppl:.4f}")
        print(f"Best architecture: {best_params}")

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write("\n" + "=" * 120 + "\n")
                f.write("PHASE 2 — Continuous Fine-tuning (minimize perplexity)\n")
                f.write(f"Fixed architecture: {best_params}\n")
                f.write("=" * 120 + "\n")

        print("\n" + "=" * 60)
        print("PHASE 2: Fine-tuning — minimizing perplexity")
        print(f"Architecture fixed: {best_params}")
        print("=" * 60)

        study2 = self.tune_phase2(n_trials_phase2, best_params, phase2_search_space)

        if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study2.trials):
            print("Phase 2 produced no successful trials.")
            return {
                "phase1_best_ppl": best_phase1_ppl,
                "phase1_best_params": best_params,
            }

        best_ppl = study2.best_value
        best_continuous = study2.best_params
        final_params = {**best_params, **best_continuous}

        print(f"\nPhase 2 complete — Best perplexity: {best_ppl:.4f}")

        if save_best_to:
            Path(save_best_to).parent.mkdir(parents=True, exist_ok=True)
            with open(save_best_to, "w") as f:
                f.write("=" * 60 + "\n")
                f.write("BEST HYPERPARAMETERS\n")
                f.write("=" * 60 + "\n\n")
                f.write("PHASE 1 — Best for Perplexity\n")
                f.write(f"Best PPL: {best_phase1_ppl:.6f}\n")
                f.write("-" * 40 + "\n")
                for k, v in best_params.items():
                    f.write(f"{k} = {v}\n")
                f.write("\nPHASE 2 — Best for Perplexity\n")
                f.write(f"Best PPL: {best_ppl:.6f}\n")
                f.write("-" * 40 + "\n")
                for k, v in final_params.items():
                    f.write(f"{k} = {v}\n")
            print(f"Best hyperparameters saved to: {save_best_to}")

        return {
            "phase1_best_ppl": best_phase1_ppl,
            "phase1_best_params": best_params,
            "phase2_best_ppl": best_ppl,
            "phase2_best_params": best_continuous,
            "final_params": final_params,
        }
