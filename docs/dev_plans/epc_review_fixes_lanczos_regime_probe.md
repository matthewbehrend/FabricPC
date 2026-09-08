# ePC tuning diagnostics: Lanczos spectrum, `Regime`, and `RegimeProbe`

## Summary

`EPCInference` is FabricPC's error-parameterized predictive-coding solver (ePC, Goemaere et al., arXiv 2505.20137). Its two tuning knobs, the error learning rate `eta_infer` (η) and the step count `infer_steps` (T), have a safe range set by the curvature of the energy the solver descends, and that curvature grows during training. The branch `matthew_cedric/epc` shipped three diagnostics for picking η and T: an eigenvalue estimator (`top_epsilon_eigenvalue`), a regime label (`EPCInference.regime_label`), and a λ_max tracking section in `scripts/epc_analysis.py`. A review of that work (`docs/dev_plans/epc_oracle_and_analysis_review.md`) and a review of the first version of this plan (`docs/dev_plans/epc_review_fixes_plan_review.md`) found the correctness verification sound and the diagnostics defective. This PR replaces them:

- `fabricpc/core/epsilon_spectrum.py`: a Lanczos estimator that returns the largest and smallest curvature the solver's starting gradient excites, and how that gradient is distributed over the spectrum.
- `EPCInference.regime(spectrum) -> Regime`: a structured verdict (stability, output-gradient reversal, relaxation band, indefiniteness) replacing the string label.
- `fabricpc/training/regime_probe.py`: `RegimeProbe`, a `train()` callback that records the spectrum, the regime flags, and per-edge weight norms every N updates on any graph.
- Four GPU control runs on the ResNet-18 demo that separate "curvature grows regardless and a fixed η eventually crosses the bound" from "ePC's relaxation drives the growth", and the resulting corrections to the guide, the CHANGELOG, and the technical report `docs/reports/epc_regime_and_stability_report.md`.

The per-update hook the probe needs (`train(iter_callback=...)` receiving an `IterContext` that carries the parameters) ships first as its own PR (`docs/dev_plans/iter_callback_context.md`); this branch rebases onto it.

## Background

### What ePC relaxes, and what its curvature is

A FabricPC graph has nodes t with latent activity z_t and a prediction μ_t computed from the node's in-edge sources. The energy is E = Σ_t ½ p_t ‖z_t − μ_t‖² over the nodes that receive edges, with p_t the node's Gaussian precision (1.0 unless set); a cross-entropy output replaces its quadratic term. Inference minimizes E over the unclamped latents with the input and the target clamped. State-based PC moves each z_t down its local gradient. ePC relaxes the prediction errors ε_t = z_t − μ_t instead and derives the latents by one forward pass per step, z_t = μ_t + ε_t along the schedule, so one reverse pass yields ∂E/∂ε for every node at once (`EPCInference.error_energy` is the ε-energy closure; `forward_value_and_grad` differentiates it).

Every FabricPC run starts inference at the feedforward state, ε = 0. Near that point E is a quadratic in ε with Hessian H_ε, and gradient descent on a quadratic splits into independent modes along the eigenvectors of H_ε. For a mode with eigenvalue λ, each step multiplies the distance to that mode's minimum by (1 − ηλ):

| ηλ | one step |
|---|---|
| 0 < ηλ < 1 | moves part of the way, same side |
| 1 < ηλ < 2 | overshoots to the other side, closer |
| ηλ > 2 | lands farther away than it started |

After T steps a mode has closed the fraction f(λ) = 1 − (1 − ηλ)^T of its distance. Only modes along which the starting gradient g0 = ∇_ε E at ε = 0 has a component ever move; these are the excited modes. On a linear chain with unit precision, H_ε = I + JᵀJ with J the map from the stacked hidden errors to the output prediction, g0 = Jᵀr with r = y − μ_y the output residual, and the excited modes are the d_y directions of the row space of J, with eigenvalues eig(S), S = I + JJᵀ (Innocenti et al. 2024, Theorem 1). λ_max = 1 + σ_max(J)² grows with the product of the downstream weights, so it grows during training.

Two regimes follow, and one bound. When every excited mode has η·T·λ ≪ 1, the T-step result is ε ≈ −η·T·g0, and the weight gradients computed from those errors are backprop's, hidden layers scaled by η·T and the output layer unscaled (Goemaere et al., Theorem C.9): the backprop-like regime. When every excited mode has relaxed, ε is the PC equilibrium and the output error is r S⁻¹: the PC-equilibrium regime. At every T, η·λ_max < 2 is required for the iteration not to diverge.

### The data the diagnostics must explain

Measured on the muPC ResNet-18 CIFAR-10 demo (`examples/resnet18_cifar10_demo.py`) on one NVIDIA RTX 3090:

- λ_max = 16.4 at init on a 64-sample test batch, by power iteration.
- The 2-epoch sweep (`examples/epc_spc_resnet18_compare.py --mode sweep`, five trials per cell, η ∈ {1e-4 … 1e-1}, T ∈ {1 … 160}): accuracy is 38.8% wherever f(λ_max) ≈ 0 and 31% wherever f(λ_max) ≈ 1, and in between in between. A single eigenvalue fitted to the 45 cells with η ≤ 0.01 is λ_eff = 12.0. The 31% plateau is the PC-equilibrium value at 2 epochs; the 38.8% plateau is the backprop-like value.
- The 100-epoch runs: (1e-3, 1) and (1e-3, 2) reached about 76%; the defaults (1e-3, 5) collapsed to chance; every (1e-2, T) collapsed by epoch 10. Backprop on the same graph reached 77.1%.
- λ_max tracked every 50 updates through the first 30 epochs of that schedule (`epc_lambda_track__eta0.001_T5.csv`, `epc_lambda_track__eta0.01_T1.csv`). For the defaults, λ_max stayed between 15 and 27 through epoch 8, reached 51 at epoch 10, then 84, 131, 471, 3508, and 12539 at epochs 11 to 15, with accuracy at chance from epoch 15. For (1e-2, 1), λ_max reached 101 at update 1100 (epoch 6) and 220 at update 1150, and the run was at chance in epoch 7.

### Symbols

| Symbol | Meaning |
|---|---|
| ε | stacked prediction errors of the unclamped nodes, the variable `EPCInference` relaxes |
| H_ε | Hessian of the total energy in ε coordinates at the feedforward point ε = 0 |
| g0 | ∇_ε E at ε = 0, the backprop activation gradient; the Lanczos start vector |
| excited mode | an eigenvector of H_ε along which g0 has a nonzero component; the only modes gradient descent from ε = 0 moves |
| λ_max, λ_min | the largest and smallest eigenvalues among the excited modes (the two Lanczos Ritz extremes) |
| η, T | `eta_infer`, `infer_steps` |
| f(λ) | relaxed fraction 1 − (1 − ηλ)^T of a mode with eigenvalue λ after T steps |
| f_max | f(λ_max), the fastest mode's relaxed fraction |
| θ_k, w_k | the k-th Ritz value (an eigenvalue of the Lanczos tridiagonal T_k) and the fraction of ‖g0‖² carried by that Ritz mode, the squared first component of T_k's k-th eigenvector; Σ_k w_k = 1 |
| f̄ | gradient-weighted relaxed fraction Σ_{θ_k > 0} w_k f(θ_k) / Σ_{θ_k > 0} w_k |
| α_j, β_j | Lanczos recurrence coefficients, the diagonal and off-diagonal of T_k |
| J, S, r | on a linear chain: the map from the stacked hidden ε to the output prediction, S = I + JJᵀ, and the feedforward output residual y − μ_y |

## Findings: what is wrong with the shipped diagnostics

### F1. The estimator returns the largest-magnitude eigenvalue, not the largest positive one

`top_epsilon_eigenvalue` (`fabricpc/utils/linear_pc_oracle.py:550-614`) is power iteration on Hessian-vector products from a random start; it converges to the eigenvalue of largest magnitude. On a graph with gelu activations and a softmax cross-entropy output, H_ε contains the term Σ_k (∂L/∂μ_k)·∂²μ_k/∂ε², the loss gradient weighting the second derivatives of the network map, and that term makes H_ε indefinite once the weights are large. The (1e-2, 1) tracking CSV records λ_max = −9398.9 at update 1200. Consequences: `regime_label(−9399)` returns "backprop-like" (a negative η·λ makes f_max negative, below the 0.1 band edge); the crossing detector tests η·λ > 2 and never fires; the demo would print η_max = 2/λ as a negative number. Power iteration also has no convergence indicator, and for a positive-definite H its Rayleigh quotient is a lower bound on λ_max, so an unconverged value overstates the safe rate.

### F2. The equilibrium band reads the fastest mode, and the condition is misprinted

`regime_label` (`inference_epc.py:98-129`) names "near PC equilibrium" when f_max > 0.9, which says the fastest excited mode has relaxed 90%. Equilibrium needs the modes that carry the gradient to relax, and the slowest of them sets the pace. The class docstring, the guide paragraph (`docs/user_guides/12_api_inference.md`), and the tuning table all state the condition as "η·T·λ_max … ≳ 3/λ_min,excited", which as written multiplies by λ_max and divides by λ_min, a dimensional error. The label also reports a "slowest" mode from a hard-coded floor eigenvalue of 1, wrong for any precision other than 1. On the ResNet-18 the excited spectrum happened to be compact (λ_eff = 12 against λ_max = 16.4), so the label matched the sweep there; that is not general.

### F3. The T = 1 danger threshold is too lenient: the output gradient reverses at η(λ_max − 1) > 1

On the unit-precision chain, gradient descent from ε = 0 leaves the output residual after T steps at r_T = (r/λ)·[1 + (λ − 1)(1 − ηλ)^T] along a mode with eigenvalue λ; the equilibrium value r/λ per mode is r S⁻¹. At T = 1 this is r_1 = (1 − η(λ − 1))·r. The output layer's local weight gradient is proportional to r_1, so it reverses sign along the top mode at η(λ_max − 1) > 1, before the iteration bound η·λ_max > 2. The hidden errors ε_T = (1 − (1 − ηλ)^T)·ε* keep the sign of their equilibrium value for every ηλ < 2, so the reversal is confined to the output layer. For general T the flip condition is (1 − ηλ)^T < −1/(λ − 1), which can hold only at odd T because (1 − ηλ)^T ≥ 0 at even T. The data agree: in the (1e-2, 1) run λ_max passed 100 at update 1100, where η(λ − 1) ≈ 1, test accuracy fell that epoch from 43.2% to 40.6%, and the network was at chance one epoch later. The shipped label marks η·λ > 1 only as "overshooting" and documents η·λ > 2 as the T = 1 danger.

### F4. The tracking tool cannot run on a user's graph

`section_track_lambda_max` (`scripts/epc_analysis.py:836-956`) re-implements the training loop over `make_train_step` (about 120 lines) and reaches into the demo module for `_create_mupc_model`, `make_optimizer`, and `AugmentedCifar10Loader`. It exists because `train()`'s `iter_callback(epoch_idx, batch_idx, metrics)` cannot see the parameters. `epoch_callback(ctx)` can (`EpochContext.params`), but per-epoch access is too coarse: in the defaults collapse λ_max grew from 584 to 1608 within 50 updates.

### F5. No control runs, and the paper's contrary result is undocumented

Both tracked cells collapsed, so "η·λ_max crossed 2 before the collapse" is an ordering, not a cause. Separating "λ_max grows regardless of the solver and a fixed η eventually crosses the bound" from "ePC's relaxation beyond the backprop-like regime drives the weight growth" needs the cells that did not collapse: (1e-3, 1) and the backprop trainer. Goemaere et al. (Table E.10) trained ResNet-18 at the same error rate 1e-3 and T = 5 for 50 epochs with Adam on the weights and reported no instability. Their network has batch normalization after every convolution, ReLU, weight decay at most 1e-3, and a standard parameterization; the demo uses the muPC parameterization, no normalization layers, gelu, weight decay 1e-2, and a 100-epoch schedule. λ_max = 1 + σ_max(J)² is a weight-scale quantity, so normalization and weight decay are levers the guide should name.

### F6. Smaller defects

- The demo docstring says λ_max "grew about threefold per epoch"; the CSV shows about 1.13× per epoch through epoch 10 and a runaway from epoch 11.
- "Under Adam the η scaling is normalized away" holds only while η·|g| ≫ Adam's ε (1e-8); without Adam, hidden layers learn η times slower than the output layer.
- At equilibrium the output error is r S⁻¹, which damps the learning signal along mode λ_S by 1/λ_S, a matrix rescaling a per-parameter optimizer cannot undo. This linear-chain mechanism is consistent with the 31% plateau trailing the 38.8% plateau, and the analysis script does not show it.
- The sPC+muPC equilibrium test cannot pin the gradient's scale, because a diagonal preconditioner shares the fixed point; only a stability bracket does.
- The oracle docstring does not say that cyclic graphs are outside it, nor that a batch's λ_max is the per-sample maximum (the batch energy is a sum, so H_ε is block-diagonal over samples).
- The sweep fit is a heuristic, and the guide should not call it agreement.

## Design

### 1. Spectrum estimator: Lanczos from g0 with Ritz weights

**Chosen.** `lanczos_extremes(hvp, v0, iters)` runs the three-term Lanczos recurrence from v0 = g0 using only Hessian-vector products (`jax.jvp` of `jax.grad` of the ε-energy). Lanczos builds an orthonormal basis of span{g0, H g0, H² g0, …} and represents H_ε in that basis as a k × k tridiagonal matrix T_k. The eigenvalues of T_k, the Ritz values θ_k, approximate eigenvalues of H_ε and converge fastest at both ends of the spectrum; the squared first component of each eigenvector of T_k, w_k, is the fraction of ‖g0‖² that Ritz mode carries. One run therefore returns the three things the regime needs: λ_max (the stability bound 2/λ_max), λ_min (negative means indefinite), and the distribution {(θ_k, w_k)} of the gradient over the spectrum, from which f̄ = Σ_{θ_k > 0} w_k f(θ_k) / Σ_{θ_k > 0} w_k is the relaxed fraction of the gradient that drives the weight update. Three vectors are carried, so memory is independent of `iters`.

Breakdown guard. When β_j ≤ √eps(dtype)·max(max_i |α_i|, max_i β_i), the Krylov space is exhausted and the recurrence freezes. Frozen steps carry α_j = α_0 (the Rayleigh quotient of g0, inside the excited spectrum by construction) and β_j = 0, so they decouple from T_k as 1 × 1 blocks with zero weight that move neither extreme nor f̄; `k` counts the valid steps. A direction entering with β/|α| below √eps carries weight below eps in g0, the same cutoff the oracle's `excited_eigenvalues` applies through its overlap tolerance. The residual of each extreme Ritz pair is β_k·|s_{k−1}|, zero after a breakdown. Ghost eigenvalues from lost orthogonality duplicate converged extremes and split their weight; they move neither the extremes nor the weighted sums, and the docstring says so.

Why the weights, not only the extremes. On a linear graph g0 = Jᵀr has components only along the row space of J, so every w_k lies on one of the d_y eigenvalues eig(S), a compact band, and the two extremes describe it. On a nonlinear graph the second-derivative term in H_ε (F1) couples g0 to every direction, so nearly every mode is excited and λ_min is the bottom of the full spectrum, far from where the gradient sits. Measured on the analysis script's gelu MLP (x32 → four hidden layers of 64 gelu units → 10-way softmax with cross-entropy, batch 4, 1024 ε entries, full Hessian by `jax.hessian`; Evidence E2): 943 of 1024 modes are excited, the smallest excited eigenvalue is 0.555, below the unit-precision floor, and 80% of the gradient's weight lies between 1.06 and 1.39. A band on λ_min would say the slowest mode is far from relaxed while the modes carrying the gradient are done. The weights fix that: f̄ from 30 Lanczos steps equals f̄ from the exact 1024 × 1024 eigendecomposition to three digits at both weight scales tested. On a linear graph f̄ averages f over eig(S), so it agrees with the slowest-excited-mode condition whenever that band is compact, which is the ResNet-18's case (λ_eff = 12 against λ_max = 16.4).

The module is `fabricpc/core/epsilon_spectrum.py`. It is solver machinery (it calls `EPCInference.begin_segment` and `error_energy`), and no module in `fabricpc/core` imports from `fabricpc/utils` today. `make_epsilon_spectrum` imports `EPCInference` inside the function; `inference_epc.py` annotates `EpsilonSpectrum` under `TYPE_CHECKING`. The linear oracle stays in `fabricpc/utils/linear_pc_oracle.py`, NumPy-only.

**Rejected.**

- *Shifted power iteration* (a second pass on H + |λ_1|·I when the first pass returns λ_1 < 0). Ten lines and a sign fix, but it yields one number: no λ_min, no weights, so the equilibrium band would have to be dropped, and it converges slowly on a compact spectrum.
- *Lanczos extremes only, with the equilibrium band on λ_min.* Exact on linear graphs. On nonlinear graphs λ_min is the bottom of the full spectrum: 0.555 on the gelu MLP, and predicted near or below 1 on the ResNet-18 (to be recorded in `--resnet18`). With λ_min ≈ 1, the band f(λ_min) > 0.9 needs η·T ≳ 3, and most of the 2-epoch sweep cells that sit on the 31% PC-equilibrium plateau would be labelled "partially relaxed": at η = 0.03, T = 5 (31.4%), f(1) = 0.14; at η = 0.1, T = 16 (30.7%), f(1) = 0.82. The label would contradict the data it was built to explain.
- *Exact-zero breakdown guard* (freeze when β_j = 0). On the linear test fixtures the Krylov space has dimension at most d_y = 3, so β_3 is zero in exact arithmetic but about 1e-6·|α| in float32 and 1e-15·|α| in float64. Neither is zero. The recurrence continues on rounding noise, a vector outside the excited subspace, and converges to unexcited floor modes: on a synthetic rank-3 excited spectrum (excited eigenvalues 15.7, 24.2, 33.6; floor 1.0) the minimum Ritz value after 30 steps is 1.000 in both precisions, and the planned test "λ_min against the minimum of `excited_eigenvalues`" fails. λ_max is unaffected, which is why power iteration never exposed this. The relative guard stops at step 3 and returns the three excited eigenvalues to six digits (Evidence E1).
- *Full reorthogonalization.* Keeps k vectors of ε size (on the ResNet-18 probe batch, k copies of every hidden error) to suppress ghosts that move neither the extremes nor the weighted sums.
- *Module in `fabricpc/utils`.* `EPCInference.regime(spectrum: EpsilonSpectrum)` would make core import utils, the first such edge in the package.

### 2. `Regime`: a structured verdict

**Chosen.** `EPCInference.regime(spectrum) -> Regime`, a NamedTuple:

| Field | Meaning |
|---|---|
| `eta_lambda_max`, `eta_T_lambda_max` | η·λ_max and η·T·λ_max |
| `unstable` | η·λ_max > 2: the top mode's distance grows every step |
| `output_gradient_reverses` | (1 − ηλ_max)^T < −1/(λ_max − 1) at unit precision (T = 1: η(λ_max − 1) > 1); always False at even T |
| `f_max`, `f_weighted` | f(λ_max) and f̄ |
| `band` | on f̄: "backprop-like" below 0.1, "near PC equilibrium" above 0.9, "partially relaxed" between |
| `negative_weight`, `growth_min` | the fraction of ‖g0‖² on negative-curvature modes, and (1 + η·max(0, −λ_min))^T, the growth of the most negative mode over T steps |
| `lambda_max`, `lambda_min` | the extremes |

`__str__` is the one-line label with precedence: `unstable`, then `growth_min > 1.1` ("indefinite: negative curvature carrying X% of the gradient grows Y× over T steps"), then the band with f̄, f_max, and the reversal note. The band uses positive modes only.

Indefiniteness by weight and growth, not a boolean. At weight std 2.0 the gelu MLP is indefinite at init, with 16 negative eigenvalues down to −14.8 carrying 22.6% of ‖g0‖² (Evidence E2), yet one ePC step from ε = 0 is still exactly −η·g0. Curvature sign enters at the second step, and over T steps a negative mode grows by (1 + η|λ_min|)^T, which at the defaults (η = 1e-3, T = 5) is 1.08. The 1.1 threshold is the same scale as the 0.1 band edge.

**Rejected.**

- *A string label with an optional `lambda_min_excited` argument.* Keeps substring parsing in the analysis script (`regime_letter`, `epc_analysis.py:304-313`) and cannot carry the flags.
- *A boolean `indefinite` that outranks the band.* Any negative eigenvalue would print "indefinite" over "backprop-like", telling a user whose setting is exact backprop that it is broken.
- *Bands on f_max* (today's label). Names equilibrium from the fastest mode (F2).
- *Bands on f(λ_min).* See the estimator's rejected alternatives.

### 3. `RegimeProbe`: the diagnostic as a `train()` callback

**Chosen.** `RegimeProbe(structure, probe_clamps=None, *, every, inference=None, iters=30, key, csv_path=None)` in `fabricpc/training/regime_probe.py`, exported from `fabricpc.training`. `on_iter(ctx: IterContext)` runs every `every` updates: `initialize_graph_state(structure, batch_size, key, clamps, params=ctx.params)`, the compiled `make_epsilon_spectrum`, per-edge weight Frobenius norms from `ctx.params`, `ctx.metrics["energy"]`, and `inference.regime(spectrum)` when `inference` is an `EPCInference`. `probe_clamps` are fixed clamps built once by the caller (the demo builds them from a 64-sample test batch); `None` means measure on the training batch through `build_clamps(ctx.batch, structure, clamp_target=True)`. `on_epoch(ctx, accuracy=None)` records the epoch row. Readouts: `rows`, `first_reversal()`, `first_crossing()`, `first_chance(chance, margin=0.05)` with chance = 1/num_classes supplied by the caller, `growth_phases()` (the per-epoch λ_max maximum and its ratio to the previous epoch), `write_csv()`, `summary()` (reversal update, crossing update, chance epoch, growth phases, in that order).

CSV: metadata columns first, constant per row (`trainer, eta_infer, infer_steps, every, probe_batch`; η and T empty under backprop), then `update, epoch, lambda_max, lambda_min, f_weighted, negative_weight, growth_min, residual_max, eta_lambda_max, output_gradient_reverses, unstable, train_energy, test_accuracy`, then `wnorm:<edge_key>` per weight. Readers take η, T, and the trainer from the columns; the filename is a label. Works under `algorithm="backprop"` (spectrum and norms; regime columns empty). The docstring states the cost: supplying `iter_callback` forces a device sync on every batch (`trainer.py:570`), not only on probed ones.

The probe consumes the `IterContext` PR: `iter_callback(ctx)` with `params`, `opt_state`, `step`, `batch`, and `metrics`, valid during the callback because the training step donates its buffers. That PR's migration list includes `fabricpc/tuning/bayesian_tuner.py:111`, `fabricpc/utils/dashboarding/callbacks.py`, `examples/transformer_v2_demo.py`, and the trainer and `test_fabricpc.py` tests.

**Rejected.**

- *`epoch_callback` only.* No trainer change, but per-epoch access misses a collapse in which λ_max grows 2.75× within 50 updates (F4).
- *A trainer flag `probe_every=N, probe=callable`.* A second per-batch mechanism beside `iter_callback`, with its own sync and return semantics.
- *Keep the script's custom loop.* The status quo: not reusable on a user's graph (F4).
- *η and T parsed from the CSV filename* (today's `plot_lambda_track`, `epc_analysis.py:1080`). The backprop file has neither, so the reader would need a branch.
- *The name `LambdaMaxProbe`.* It names one column of a probe that records a spectrum, weight norms, and flags.

### 4. Control runs: four cells and a reading rule fixed in advance

**Chosen.** `python examples/resnet18_cifar10_demo.py --num_epochs 30 --schedule_epochs 100 --augment --activation gelu --track_regime 50 --eval_every 1` for four runs: (1e-3, 5), the defaults collapse the report rests on; (1e-3, 1), the 100-epoch survivor; `--trainer backprop`; and (1e-2, 1), the run whose CSV holds the −9399. About 20 minutes each on the 3090. Reading rule: if λ_max and the weight norms grow at a comparable rate in the backprop and (1e-3, 1) runs as in the collapsing cells, growth is a weight-scale effect of this parameterization (no normalization, weight decay 1e-2) and the remedy is a rate that follows λ_max or weight-norm control; if they grow only in the collapsing cells, ePC's relaxation feeds the growth. Record which case was observed, whether λ_max grows in the runs that do not collapse, λ_min and `negative_weight` through the collapses, and for (1e-2, 1) a positive λ_max at update 1200 and the reversal flag at update 1100.

**Rejected.** *Three runs without (1e-3, 5).* Leaves the report's headline series in the old CSV schema from the sign-ambiguous estimator, which the new `--plot_track` reader cannot read without a fallback branch, and without λ_min or weight norms through the collapse.

## Deliverables

### A. `fabricpc/core/epsilon_spectrum.py` (new) and the oracle module

- `lanczos_extremes(hvp, v0, iters) -> LanczosResult(ritz_values, ritz_weights, residual_min, residual_max, breakdown, k)` as in Design 1: `lax.fori_loop` carrying (v_prev, v_cur, β_prev, α[], β[], frozen); T_k → `jnp.linalg.eigh`; w_k = s_{0k}².
- `EpsilonSpectrum(lambda_max, lambda_min, ritz_values, ritz_weights, residual_max, residual_min, negative_weight, gradient_norm, random_start, iters, k)`.
- `weighted_relaxed_fraction(spectrum, eta, steps) -> float`: f̄.
- `make_epsilon_spectrum(structure, iters=30)`: compiled `(params, state, clamps, key) -> EpsilonSpectrum`. `EPCInference.begin_segment`, then `energy_of, errors` from `error_energy`, `hvp` via `jax.jvp(jax.grad(...))`, start vector g0 (random start under `lax.cond` when ‖g0‖ = 0, recorded as `random_start=True`; the extremes then describe the full spectrum, not the excited one).
- `epsilon_spectrum(params, state, clamps, structure, iters=30, key=None)`: one-compile convenience.
- `linear_pc_oracle.py`: delete Part 2b (`make_top_epsilon_eigenvalue`, `top_epsilon_eigenvalue`; the module becomes NumPy-only) and migrate every caller (F, G). The docstring gains the cyclic-graph limit and the per-sample-maximum sentence. `stability_bound(H)` raises `ValueError("no positive curvature ...")` when λ_max ≤ 0.

### B. `fabricpc/core/inference_epc.py`

`Regime` and `regime(spectrum)` as in Design 2; `regime_label` removed. Class docstring: the equilibrium condition "η·T·λ ≳ 3 on the modes that carry the gradient, f̄ > 0.9; on a linear graph these are the eig(S) modes"; the T = 1 sentence replaced by the reversal mechanism (F3); the Adam sentence with the ε caveat and the SGD disparity (F6); λ_max = 1 + σ_max(J)² as a weight-scale quantity controlled by normalization or weight decay; Goemaere et al.'s ResNet-18 setting and the differences (F5).

### C. `fabricpc/training/regime_probe.py` (new)

`RegimeProbe` as in Design 3.

### D. `examples/resnet18_cifar10_demo.py`

- `--track_regime N` (default 0 = off) and `--schedule_epochs` (default `--num_epochs`; 100 reproduces the first epochs of the 100-epoch schedule). Builds the probe on a fixed 64-sample test batch, passes `iter_callback=probe.on_iter`, and the existing `epoch_callback` also calls `probe.on_epoch(ctx, acc)`. CSV `epc_regime_track__{trainer}_eta{eta}_T{T}.csv` (backprop: `epc_regime_track__backprop.csv`). Prints `probe.summary()`.
- `lambda_max_at_init` → `spectrum_at_init(...) -> EpsilonSpectrum`; the settings line prints `str(inference.regime(spectrum))` plus λ_min, f̄, `negative_weight`, and the residual.
- Docstring: growth phases quoted from `summary()` of the (1e-3, 5) re-run with the report's 1-indexed epochs, nothing hardcoded from the old CSV; the T = 1 sentence replaced by the reversal mechanism; the tracking command; a control-run table filled from the runs.

### E. `examples/epc_spc_resnet18_compare.py`

`_demo.spectrum_at_init`; the per-arm regime column shows `band`, f̄, and the reversal flag from `EPCInference(...).regime(spectrum)`.

### F. `scripts/epc_analysis.py`

- Delete `section_track_lambda_max` and its arguments (`--track_lambda_max`, `--track_cells`, `--num_epochs`, `--schedule_epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--augment`); the docstring points to the demo flag.
- `--plot_track`: read η, T, and the trainer from the metadata columns; four panels: λ_max and |λ_min| (log) with the 2/η line only when `eta_infer` is present, f̄ with `negative_weight`, weight norms, test accuracy; vertical lines at the first reversal and the first crossing.
- `--resnet18`: `epsilon_spectrum` → λ_max, λ_min, `negative_weight`, residuals, `str(EPCInference().regime(spectrum))`; record λ_min at init against the prediction (near or below 1). The sweep-regime table prints f̄ per cell from the init spectrum beside the measured accuracy, with the letter from `.band` and `.unstable`; the one-eigenvalue fit stays, labelled heuristic.
- `stability`: on the depth-5 chain, λ_max against `eigvalsh` max, λ_min against the minimum of `excited_eigenvalues`, f̄ against the oracle's weighted fraction Σ_λ (v_λᵀg0)² f(λ) / Σ_λ (v_λᵀg0)² over the excited modes; gelu bracket unchanged.
- `backprop_regime`: `pc_weight_gradients` against `jax.grad` of the output energy divided by `grad_denominator` (the test's normalization; ratios unchanged).
- `equilibrium_profile`: a table from `theorem1_energy`'s S and r: ‖r S⁻¹‖/‖r‖ (batch mean), λ_min(S), λ_max(S) per depth and init, with the F6 paragraph worded "consistent with" and naming it the linear-chain mechanism applied to the nonlinear cross-entropy ResNet.
- Sweep-fit wording: "heuristic" and "consistent with", never "agreement".

### G. Tests

- `tests/test_linear_pc_oracle.py`: `test_power_iteration_matches_oracle` → `test_lanczos_matches_excited_extremes` on `fork-merge`, `prior-source`, `chain-h3-std0.8`, and `chain-h3` in float32 explicitly (the near-breakdown case): λ_max against `eigvalsh` max and λ_min against the minimum of `excited_eigenvalues` (rtol 1e-3), f̄ against the oracle's weighted fraction (atol 1e-3), `breakdown` flagged before `iters`; float32 and float64. `TestStabilityBracket` parametrized over `chain-h3` and `mupc-chain-h3` (identity activations make muPC's top-down scale the exact chain-rule factor, so the muPC bracket pins sPC+muPC's gradient scale, which the equilibrium test alone cannot). `stability_bound` raises when λ_max ≤ 0.
- `tests/test_epsilon_spectrum.py` (new): `lanczos_extremes` on an explicit matrix with eigenvalues {−50, −1, 1, 3, 10} and a start vector of known eigen-components returns 10 and −50, Ritz weights equal to the squared normalized components, `negative_weight` equal to the weight on {−50, −1}; a start vector inside a 2-dimensional invariant subspace → `breakdown` at step 2 in float32, extremes unchanged, zero weight on frozen steps; zero start vector → `random_start`; on the tanh MLP fixture `epsilon_spectrum` extremes and f̄ agree with `jax.hessian` of `energy_of`; on the gelu MLP at weight std 2.0, λ_min < 0 and `negative_weight` agree with `jax.hessian`.
- `tests/test_inference_epc.py`: `TestRegimeLabel` → `TestRegime`: bands from constructed spectra (a compact spectrum; a spread spectrum with the weight at the top versus at the floor, same extremes, different bands); reversal at T = 1 and T = 5; no reversal at T = 2 for any η < 2/λ; unstable; `growth_min` precedence (negative_weight 0.2 with growth 1.08 prints the band, growth 1.5 prints indefinite); `str()`; the parity test reads `epsilon_spectrum(...).lambda_max`.
- `tests/test_regime_probe.py` (new, CPU): tanh MLP, 2 epochs of `train(..., iter_callback=probe.on_iter, epoch_callback=...)` with `every=2`: row count, metadata and regime columns, `first_crossing() is None`, weight-norm columns present, CSV round-trips through `plot_track`'s reader; also under `algorithm="backprop"` (regime columns empty, no 2/η line).
- Dashboarding callback tests and `tests/test_fabricpc.py` migrate in the `IterContext` PR. `tests/test_doc_snippets.py` checks the new guide fence.

### H. Documentation

- `docs/user_guides/12_api_inference.md`: rewrite the backprop-regime paragraph (F2, F3, F5, F6: the f̄ condition with its linear reduction; the reversal threshold and its odd-T scope; the `Regime` fields; the Adam-ε and SGD caveats; the paper's ResNet-18 setting and the differences; λ_max as a weight-scale quantity and its levers; the probe's per-batch sync cost). Tuning-table rows: `eta_infer` below 1/(λ_max − 1) at odd T for the output gradient and below 2/λ_max at every T, measured with `epsilon_spectrum`, tracked with `RegimeProbe`; `infer_steps` by f̄ (≪ 0.1 backprop-like, > 0.9 equilibrium). A python fence showing `RegimeProbe` with `train`.
- `docs/user_guides/16_troubleshooting.md:204` and `03_how_predictive_coding_works.md`: names and condition updated.
- `CHANGELOG.md` unreleased: `regime_label`, `top_epsilon_eigenvalue`, and `--track_lambda_max` are all in the unreleased section, so no breaking-change bullet. Edit the `EPCInference`, `linear_pc_oracle`, `error_energy`, and `epc_analysis` entries to name `epsilon_spectrum`, `Regime` / `regime`, `RegimeProbe`, and `--track_regime`, with the corrected regime sentence.
- `docs/dev_plans/epc_reviewer_response_oracle_and_analysis.md`: a dated revision note under Context pointing to both reviews and this plan; B.2 and D.5 amended.
- `docs/reports/epc_regime_and_stability_report.md`: lines 12–13 (T = 1 mechanism; growth phases from the re-run), 61 and 81 (bands on f̄ and the equilibrium condition), 142 (the muPC bracket now pins the claim), 165 (Adam-ε caveat), 249 ("slowest 0.5%" → f̄ and λ_min at init), 307 (the negative value was the estimator's defect; the re-run's value), 309 and 321 (control-run outcomes; growth phases), 327, 334, 340; regenerate the Section 5.8 sweep letters from `--resnet18`; add a subsection on the paper's ResNet-18 setting and one on the S⁻¹r damping.

### I. GPU control runs

As in Design 4, after A–H are green.

## Evidence

### E1. Plain Lanczos on a rank-3 excited spectrum, with and without the relative guard

H = I + JᵀJ, D = 40, J of rank 3, start vector g0 = Jᵀr, 30 steps, no reorthogonalization. Excited eigenvalues (eig(S)): 15.72, 24.19, 33.62. Floor: 1.0 on the 37 unexcited modes. β_3 was 4.0e-5 in float32 and 8.6e-14 in float64.

| precision | guard | stopped at | min Ritz | max Ritz | Σ w_k on the excited eigenvalues |
|---|---|---|---|---|---|
| float32 | β = 0 | never | 1.000002 | 33.6245 | 1.000000 |
| float64 | β = 0 | never | 1.000000 | 33.6245 | 1.000000 |
| float32 | β ≤ 1e-5·\|α\| | step 3 | 15.7200 | 33.6245 | 1 |
| float64 | β ≤ 1e-5·\|α\| | step 3 | 15.7200 | 33.6245 | 1 |

### E2. Full ε-Hessian of the gelu MLP fixture

x32 → four hidden layers of 64 gelu units → 10-way softmax with cross-entropy, batch 4, 1024 ε entries; `jax.hessian` of `error_energy` at the initialized state, using the analysis script's `build_chain` and `random_clamps` with its keys.

| weight std | λ_min | λ_max | negative modes | excited modes (overlap > 1e-8) | Σ w_k on λ < 0 | median λ by g0 weight | 90% of g0 weight below λ |
|---|---|---|---|---|---|---|---|
| 1.0 | 0.555 | 1.557 | 0 | 943 of 1024 | 0 | 1.14 | 1.39 |
| 2.0 | −14.8 | 11.4 | 16 | 256 of 1024 | 0.226 | 1.23 | 3.08 |

Relaxed fractions at η = 0.03, T = 10:

| weight std | f(λ_max) | f(λ_min) | f̄ exact | f̄ from 30 Lanczos steps | Lanczos-30 Ritz min / max |
|---|---|---|---|---|---|
| 1.0 | 0.380 | 0.155 | 0.298 | 0.298 | 0.5549 / 1.557 |
| 2.0 | 0.985 | −38.5 | 0.458 | 0.458 | −14.81 / 11.43 |

## Verification

1. `python -m pytest tests/` green; `tests/test_doc_snippets.py` and `tests/test_doc_defaults.py` included.
2. `python scripts/epc_analysis.py` on CPU: Lanczos matches the oracle's two extremes and f̄ on the depth-5 chain to 1e-3; the S⁻¹r table prints; the gradient comparison uses the normalized path and reproduces the O(η) rows.
3. `python examples/resnet18_cifar10_demo.py --num_epochs 1 --track_regime 20` prints the regime with λ_min and f̄ and writes a CSV that `scripts/epc_analysis.py --plot_track` renders with four panels; the same with `--trainer backprop` renders without the 2/η line.
4. The four control runs complete; the (1e-2, 1) re-run reports a positive λ_max at update 1200 and the reversal flag before the crossing; λ_min at init is recorded against the prediction; the reading rule is applied and the outcome recorded in the demo docstring, the guide, and the report.

## Sequencing (one commit each, suite green at each gate)

Steps 1 and 2 do not touch the trainer and start now; the `IterContext` merge and rebase gate step 3.

1. A, the oracle edits, and G's oracle and spectrum tests.
2. B and `TestRegime`; E; F's `--resnet18` and `stability` migrations.
3. After the rebase: C and the probe test; D; the rest of F (delete the tracking section, plot panels, S⁻¹r table, normalized gradients).
4. H.
5. I; record the outcomes.

Commit messages via a temporary file in the project root.
