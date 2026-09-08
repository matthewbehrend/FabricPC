# Fix the ePC oracle and analysis review findings

## Context

`docs/dev_plans/epc_oracle_and_analysis_review.md` (commit 4d95c0f) found the ePC correctness work sound and the tuning diagnostics defective in four ways: the eigenvalue estimator returns the largest-magnitude eigenvalue (a real −9399 appears in `epc_lambda_track__eta0.01_T1.csv`), the regime label names an equilibrium band from the fastest mode and misprints the equilibrium condition, the T = 1 danger threshold is stated as η·λ_max > 2 when the output-layer gradient already reverses at η(λ_max − 1) > 1, and λ_max tracking is welded to the ResNet-18 demo. Commit 533a209 (after release 0.5.1's per-prediction gradient normalization, `pc_weight_gradients` / `grad_denominator`) already closed review gap 1.1: the parity test now measures its fixture's λ_max and runs both sides through the normalized path. This plan fixes everything else in the review, including the minor items, and propagates the corrections to the technical report `docs/reports/epc_regime_and_stability_report.md`.

Decisions from the 2026-09-08 interview: Lanczos from the ε-gradient replaces power iteration; `regime_label` becomes a structured `Regime`; tracking moves into the demo through a library probe that runs per update, and GPU control runs are part of the work. The per-update hook (`iter_callback` receiving an `IterContext`) ships first as its own PR, `docs/dev_plans/iter_callback_context.md`; this branch rebases onto it.

Revision 2026-09-08, after `docs/dev_plans/epc_review_fixes_plan_review.md`. That review measured the full ε-Hessian of the analysis script's gelu MLP and ran the plan's Lanczos specification on a synthetic excited spectrum, and changed four decisions. The regime's second number is a gradient-weighted relaxed fraction f̄, not λ_min,exc: on a nonlinear graph nearly every mode is excited (943 of 1024 on the gelu MLP) and the minimum excited eigenvalue is the bottom of the full spectrum (0.555, below the unit floor), so a band on λ_min,exc would relabel the sweep's equilibrium cells "partially relaxed". The Lanczos breakdown guard is relative and dtype-aware: an exact-zero test never fires in floating point, and the recurrence then converges to unexcited floor modes (minimum Ritz value 1.000 against the expected 15.7 on a rank-3 excited spectrum). Indefiniteness is reported as the gradient weight on negative modes and their growth over T steps, not as a boolean that outranks the band (the gelu MLP at weight std 2.0 has 22.6% of its gradient on negative curvature at init and still takes an exact backprop step). The control runs include the defaults collapse (1e-3, 5). The spectrum module lives in `fabricpc/core`, and the probe is `RegimeProbe`.

## Symbols

| Symbol | Meaning |
|---|---|
| ε | stacked prediction errors of the unclamped nodes, the variable `EPCInference` relaxes |
| H_ε | Hessian of the total energy in ε coordinates at the feedforward point (ε = 0) |
| g0 | ∇_ε E at ε = 0, the backprop activation gradient; the Lanczos start vector |
| α_j, β_j | Lanczos recurrence coefficients, the diagonal and off-diagonal of the tridiagonal T_k after k valid steps |
| θ_k, w_k | the k-th Ritz value (eigenvalue of T_k) and the squared first component of its eigenvector; w_k is the fraction of ‖g0‖² on Ritz mode k, Σ_k w_k = 1 |
| λ_max, λ_min | the largest and smallest Ritz values (the extremes of the spectrum g0 excites) |
| f(λ) | relaxed fraction 1 − (1 − ηλ)^T of a mode with eigenvalue λ after T steps |
| f_max | f(λ_max), the fastest mode's relaxed fraction |
| f̄ | gradient-weighted relaxed fraction Σ_{θ_k > 0} w_k f(θ_k) / Σ_{θ_k > 0} w_k |
| S, r | Innocenti's Theorem 1 rescaling I + JJᵀ and the feedforward output residual y − μ_y; the equilibrium output error is r S⁻¹ |
| η, T | `eta_infer`, `infer_steps` |

## Deliverables

### A. `fabricpc/core/epsilon_spectrum.py` (new) and the oracle module

Move the "any graph" estimator out of `linear_pc_oracle.py` (delete its Part 2b, `make_top_epsilon_eigenvalue` / `top_epsilon_eigenvalue`, and migrate every caller listed in G). The oracle module keeps the NumPy linear-only code; its docstring gains two sentences: cyclic graphs are outside the oracle (the unrolled ε-energy is quadratic but warm-started carried latents make it not a pure function of ε), and λ_max of a batch is the per-sample maximum because the batch energy is a sum. `stability_bound(H)` raises `ValueError("no positive curvature ...")` when λ_max ≤ 0.

The module sits in `fabricpc/core` because it is solver machinery (it calls `EPCInference.begin_segment` and `error_energy`), and no module in `fabricpc/core` imports from `fabricpc/utils` today. `make_epsilon_spectrum` imports `EPCInference` inside the function; `inference_epc.py` annotates `EpsilonSpectrum` under `TYPE_CHECKING`.

New module contents:

- `lanczos_extremes(hvp, v0, iters) -> LanczosResult(ritz_values, ritz_weights, residual_min, residual_max, breakdown, k)`: pytree-agnostic three-term recurrence in a `lax.fori_loop` carrying (v_prev, v_cur, β_prev, α[], β[], frozen). Breakdown guard, relative and dtype-aware: β_j ≤ √eps(dtype)·max(max_i |α_i|, max_i β_i) sets `frozen`; a direction entering with β/|α| below √eps carries weight below eps in g0, the cutoff `excited_eigenvalues` applies through its overlap tolerance. Frozen steps carry α_j = α_0 (the Rayleigh quotient of the start vector, inside the excited spectrum by construction) and β_j = 0, so they decouple from T_k as 1 × 1 blocks with zero weight that move neither extreme nor f̄; `k` counts the valid steps. T → `jnp.linalg.eigh`; w_k = s_{0k}²; residual of each extreme Ritz pair = β_k·|s_{k−1}| (zero after a breakdown, an exact invariant subspace). Ghost eigenvalues from lost orthogonality duplicate converged extremes and split their weight without moving the extremes or the weighted sums; the docstring says so.
- `EpsilonSpectrum(lambda_max, lambda_min, ritz_values, ritz_weights, residual_max, residual_min, negative_weight, gradient_norm, random_start, iters, k)`. `negative_weight = Σ w_k over θ_k < 0`.
- `weighted_relaxed_fraction(spectrum, eta, steps) -> float`: f̄ over the positive Ritz modes, renormalized by their weight; the negative modes are reported separately by `negative_weight`.
- `make_epsilon_spectrum(structure, iters=30)`: compiled `(params, state, clamps, key) -> EpsilonSpectrum`. Runs `EPCInference.begin_segment`, gets `energy_of, errors` from `EPCInference.error_energy`, `hvp` via `jax.jvp(jax.grad(...))`, start vector g0 = grad at ε = 0 (random start under `lax.cond` when ‖g0‖ = 0, recorded as `random_start=True`; the extremes then describe the full spectrum, not the excited one).
- `epsilon_spectrum(params, state, clamps, structure, iters=30, key=None)`: one-compile convenience.

Why Lanczos from g0, and why the weights: on a linear graph the Krylov space span{g0, Hg0, …} is the excited subspace and its Ritz values are eig(S), a compact band, so the two extremes are the regime. On a nonlinear graph the Hessian gains the term Σ_k (∂L/∂μ_k)·∂²μ_k/∂ε², the output-loss gradient weighting the second derivatives of the network map, which takes Hg0 out of the row space of J; nearly every mode is excited and λ_min is the bottom of the full spectrum. The weights w_k say where the gradient sits in that spectrum, and f̄ is the relaxed fraction of the gradient that drives the weight update. On the gelu MLP f̄ from 30 Lanczos steps matches the exact eigendecomposition to three digits at both weight scales tested. Three carried vectors, so memory is independent of `iters`.

### B. `fabricpc/core/inference_epc.py`

- `Regime` NamedTuple: `eta_lambda_max`, `eta_T_lambda_max`, `f_max`, `f_weighted` (f̄), `negative_weight`, `growth_min` = (1 + η·max(0, −λ_min))^T, `band` on f̄ ("backprop-like" when f̄ < 0.1; "near PC equilibrium" when f̄ > 0.9; otherwise "partially relaxed"), `output_gradient_reverses` (general form (1 − ηλ_max)^T < −1/(λ_max − 1) at unit precision; T = 1 gives η(λ_max − 1) > 1; always False at even T because (1 − ηλ)^T ≥ 0 there), `unstable` (ηλ_max > 2), `lambda_max`, `lambda_min`. `__str__` produces the one-line label; precedence `unstable`, then `growth_min > 1.1` ("indefinite: negative curvature carrying X% of the gradient grows Y× over T steps"), then the band with f̄, f_max, and the reversal note.
- `regime(self, spectrum: EpsilonSpectrum) -> Regime` replaces `regime_label` (no string overload; callers migrate).
- Class docstring: equilibrium condition written as "η·T·λ ≳ 3 on the modes that carry the gradient, f̄ > 0.9; on a linear graph the gradient sits on eig(S), so this is η·T·λ_min,exc ≳ 3"; the T = 1 sentence replaced by the output-gradient reversal mechanism (r_1 = (1 − η(λ_max − 1)) r along the top mode, output layer only, odd T only; hidden errors keep the sign of ε* for every ηλ < 2); Adam sentence gains the ε caveat (holds while η·|g| ≫ Adam ε) and the SGD disparity (hidden layers η times slower than the output); one sentence that λ_max = 1 + σ_max(J)² is a weight-scale quantity controlled by normalization or weight decay, and that Goemaere et al. (Table E.10) trained ResNet-18 at (1e-3, 5) for 50 epochs without collapse with batch normalization after every convolution, ReLU, weight decay at most 1e-3, and a standard parameterization.

### C. Prerequisite: `IterContext` (separate PR)

Shipped before this branch from `docs/dev_plans/iter_callback_context.md`: `iter_callback(ctx: IterContext)` with `params`, `opt_state`, `step`, `batch`, and `metrics`. Its migration list includes `fabricpc/tuning/bayesian_tuner.py:111` (added 2026-09-08). Nothing in this plan changes the trainer; the probe in D consumes `IterContext` as merged. Supplying `iter_callback` forces a device sync on every batch (`trainer.py:570`), not only on probed ones; D documents that cost.

### D. `fabricpc/training/regime_probe.py` (new)

`RegimeProbe(structure, probe_clamps=None, *, every, inference=None, iters=30, key, csv_path=None)`:
- `probe_clamps`: fixed clamps built once by the caller (the demo builds them from a 64-sample test batch as `lambda_max_at_init` does today, `resnet18_cifar10_demo.py:266-272`); `None` means measure on the training batch, `build_clamps(ctx.batch, structure, clamp_target=True)`.
- `on_iter(ctx: IterContext)`: every `every` steps, `initialize_graph_state(structure, batch_size, key, clamps, params=ctx.params)` → compiled `make_epsilon_spectrum`; per-edge weight Frobenius norms from `ctx.params`; `ctx.metrics["energy"]`; if `inference` is an `EPCInference`, `inference.regime(spectrum)`. Records one row. Docstring states the per-batch device sync that supplying `iter_callback` costs.
- `on_epoch(ctx, accuracy=None)`: records the epoch row.
- `rows`, `first_reversal()`, `first_crossing()`, `first_chance(chance, margin=0.05)` (chance = 1/num_classes, supplied by the caller), `growth_phases()` (per-epoch λ_max maximum and its ratio to the previous epoch), `write_csv()`, `summary()` (the outcome sentence: reversal update, crossing update, chance epoch, growth phases, in order).
- CSV columns: metadata first, constant per row: `trainer, eta_infer, infer_steps, every, probe_batch` (`eta_infer` and `infer_steps` empty under backprop); then `update, epoch, lambda_max, lambda_min, f_weighted, negative_weight, growth_min, residual_max, eta_lambda_max, output_gradient_reverses, unstable, train_energy, test_accuracy`, then `wnorm:<edge_key>` per weight. Readers take η, T, and the trainer from the columns; the filename is a label.
Export from `fabricpc.training`. Works under `algorithm="backprop"` (spectrum and norms only; regime columns empty).

### E. `examples/resnet18_cifar10_demo.py`

- `--track_regime N` (default 0 = off) and `--schedule_epochs` (default `--num_epochs`; 100 reproduces the first epochs of the 100-epoch schedule). Builds the probe on a fixed 64-sample test batch, passes `iter_callback=probe.on_iter`, and the existing `epoch_callback` also calls `probe.on_epoch(ctx, acc)`. CSV `epc_regime_track__{trainer}_eta{eta}_T{T}.csv` (backprop: `epc_regime_track__backprop.csv`). Prints `probe.summary()`.
- `lambda_max_at_init` → `spectrum_at_init(...) -> EpsilonSpectrum`; the settings line prints `str(inference.regime(spectrum))` plus λ_min, f̄, `negative_weight`, and the residual.
- Docstring: replace "threefold per epoch" with the growth phases printed by `summary()` in the J re-run of (1e-3, 5), quoted with the report's 1-indexed epochs (the current CSV gives about 1.13× per epoch through epoch 10 and a runaway from epoch 11; the re-run's numbers replace these, nothing is hardcoded from the old CSV); T = 1 sentence to the reversal mechanism; the tracking command; a control-run table filled from J.

### F. `examples/epc_spc_resnet18_compare.py`

`_demo.spectrum_at_init`; per-arm regime column from `EPCInference(...).regime(spectrum)`: `band`, f̄, and the reversal flag.

### G. `scripts/epc_analysis.py`

- Delete `section_track_lambda_max` and its arguments (`--track_lambda_max`, `--track_cells`, `--num_epochs`, `--schedule_epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--augment`); the docstring points to the demo flag. Keep `--plot_track`: read η, T, and the trainer from the metadata columns (no filename parsing); four panels: λ_max and |λ_min| (log) with the 2/η line only when `eta_infer` is present, f̄ with `negative_weight`, weight norms, test accuracy; mark the first reversal and first crossing as vertical lines.
- `--resnet18`: `epsilon_spectrum` → print λ_max, λ_min, `negative_weight`, residuals, `str(EPCInference().regime(spectrum))`. Record λ_min at init against the review's prediction (near or below 1). The sweep-regime table prints f̄ per cell from the init spectrum beside the measured accuracy, with the letter from `.band` and `.unstable`; the one-eigenvalue fit stays, labelled heuristic.
- `stability`: Lanczos against the oracle on the depth-5 chain: λ_max against `eigvalsh` max, λ_min against the minimum of `excited_eigenvalues`, f̄ against the oracle's weighted fraction Σ_λ (v_λᵀg0)² f(λ) / Σ_λ (v_λᵀg0)² over the excited modes; gelu bracket unchanged.
- `backprop_regime`: `pc_weight_gradients` against `jax.grad` of the output energy divided by `grad_denominator` (same normalization as the test; ratios unchanged).
- `equilibrium_profile`: new table from `theorem1_energy`'s S and r: ‖r S⁻¹‖/‖r‖ (batch mean), λ_min(S), λ_max(S) per depth and init, with the paragraph: at equilibrium the output error is r S⁻¹, damping the learning signal along mode λ_S by 1/λ_S, a matrix rescaling that Adam's per-parameter rescaling cannot undo; this linear-chain mechanism is consistent with the PC-equilibrium cells trailing the backprop-like cells in the 2-epoch sweep on the nonlinear cross-entropy ResNet, and is stated as such.
- Sweep-fit wording: "heuristic" and "consistent with", never "agreement".

### H. Tests

- `tests/test_linear_pc_oracle.py`: `test_power_iteration_matches_oracle` → `test_lanczos_matches_excited_extremes` on `fork-merge`, `prior-source`, `chain-h3-std0.8`, and `chain-h3` in float32 explicitly (the near-breakdown case: Krylov dimension ≤ d_y = 3, β_3 ≈ 1e-6·|α| in float32): λ_max against `eigvalsh` max and λ_min against the minimum of `excited_eigenvalues` (rtol 1e-3), f̄ against the oracle's weighted fraction (atol 1e-3), `breakdown` flagged before `iters`; float32 and float64. `TestStabilityBracket` parametrized over `chain-h3` and `mupc-chain-h3` (identity activations make muPC's top-down scale the exact chain-rule factor, so the muPC bracket pins sPC+muPC's gradient scale, which the equilibrium test alone cannot); `stability_bound` raises when λ_max ≤ 0.
- `tests/test_epsilon_spectrum.py` (new): `lanczos_extremes` on an explicit matrix with eigenvalues {−50, −1, 1, 3, 10} and a start vector of known eigen-components returns 10 and −50, Ritz weights equal to the squared normalized components, `negative_weight` equal to the weight on {−50, −1}; a start vector inside a 2-dimensional invariant subspace → `breakdown` at step 2 in float32 with the extremes unchanged and zero weight on the frozen steps; zero start vector → `random_start`; on the tanh MLP fixture `epsilon_spectrum` extremes and f̄ agree with `jax.hessian` of `energy_of` (small D, exact); on the gelu MLP at weight std 2.0, λ_min < 0 and `negative_weight` agree with `jax.hessian`.
- `tests/test_inference_epc.py`: `TestRegimeLabel` → `TestRegime`: bands from constructed spectra (a compact spectrum; a spread one with the weight at the top versus at the floor, same extremes, different bands); reversal at T = 1 and T = 5; no reversal at T = 2 for any η < 2/λ; unstable; `growth_min` precedence (negative_weight 0.2 with growth 1.08 prints the band, growth 1.5 prints indefinite); `str()`; parity test reads `epsilon_spectrum(...).lambda_max`.
- `tests/test_regime_probe.py` (new, CPU): tanh MLP, 2 epochs of `train(..., iter_callback=probe.on_iter, epoch_callback=...)` with `every=2`: row count, metadata and regime columns, `first_crossing() is None`, weight-norm columns present, CSV round-trips through `plot_track`'s reader; also under `algorithm="backprop"` (regime columns empty, the reader draws no 2/η line).
- Dashboarding callback tests and `tests/test_fabricpc.py` migrate with C. `tests/test_doc_snippets.py` checks the new guide fence.

### I. Documentation touch points

- `docs/user_guides/12_api_inference.md`: rewrite the backprop-regime paragraph (equilibrium condition in the f̄ form with its linear reduction; reversal threshold and its odd-T scope; `regime` fields; Adam-ε and SGD caveats; the paper's ResNet-18 setting and the differences: batch normalization, ReLU, weight decay at most 1e-3, standard parameterization, schedule length; λ_max as a weight-scale quantity and its levers; the probe's per-batch sync cost); tuning-table rows for `eta_infer` (below 1/(λ_max − 1) at odd T for the output gradient, below 2/λ_max at every T, measured with `epsilon_spectrum`, tracked with `RegimeProbe`) and `infer_steps` (f̄ ≪ 0.1 backprop-like; f̄ > 0.9 equilibrium); a python fence showing `RegimeProbe` with `train`.
- `docs/user_guides/16_troubleshooting.md:204` and `03_how_predictive_coding_works.md`: names and condition updated.
- `CHANGELOG.md` unreleased: `regime_label`, `top_epsilon_eigenvalue`, and `--track_lambda_max` are all in the unreleased section, so no breaking-change bullet. Edit the `EPCInference`, `linear_pc_oracle`, `error_energy`, and `epc_analysis` entries to name `epsilon_spectrum`, `Regime` / `regime`, `RegimeProbe`, and the demo's `--track_regime`, with the corrected regime sentence (f̄, the reversal threshold).
- `docs/dev_plans/epc_reviewer_response_oracle_and_analysis.md`: a dated revision note under Context pointing to both reviews and this plan; B.2 and D.5 amended.
- `docs/reports/epc_regime_and_stability_report.md`: correct lines 12–13 (T = 1 mechanism; growth phases from the re-run), 61 and 81 (bands on f̄ and the equilibrium condition), 142 (the muPC bracket now pins the claim), 165 (Adam-ε caveat), 249 ("slowest 0.5%" → f̄ and λ_min at init), 307 (the negative value was the estimator's defect; the re-run's value), 309 and 321 (control-run outcomes; growth phases), 327, 334, 340; regenerate the Section 5.8 sweep letters from `--resnet18`; add a subsection on the paper's ResNet-18 setting and one on the S⁻¹r damping.

### J. GPU control runs (3090, after A–H are green)

Reading rule, fixed before the runs: if λ_max and the weight norms grow at a comparable rate in the backprop and (1e-3, 1) runs as in the collapsing cells, growth is a weight-scale effect of this parameterization (no normalization, weight decay 1e-2) and the remedy is a rate that follows λ_max or weight-norm control; if they grow only in the collapsing cells, ePC's relaxation feeds the growth. Record which case was observed.

`python examples/resnet18_cifar10_demo.py --num_epochs 30 --schedule_epochs 100 --augment --activation gelu --track_regime 50 --eval_every 1` for four runs: `--inference epc --eta_infer 1e-3 --infer_steps 5` (the defaults collapse the report rests on, re-measured with the fixed estimator and the new columns), `--inference epc --eta_infer 1e-3 --infer_steps 1` (the 100-epoch survivor), `--trainer backprop`, and `--inference epc --eta_infer 1e-2 --infer_steps 1` (expected positive λ_max at update 1200 and the reversal flag at update 1100). About 20 minutes each. Every CSV the report cites then comes from one estimator and one schema. Record λ_max growth, λ_min and `negative_weight` through the collapses, weight-norm growth, and the flags in the demo docstring, the guide paragraph, and the report; state the outcome as observed, including whether λ_max grows in the runs that do not collapse.

## Alternatives considered

- **Estimator:** shifted power iteration (10 lines, sign fix only, no second number) versus Lanczos extremes only (both extremes, indefiniteness) versus Lanczos with Ritz weights (chosen: the extremes plus where the gradient sits, at no extra cost; the k × k eigendecomposition supplies the weights). Full reorthogonalization rejected: k vectors of ε size on the ResNet-18 probe batch for no gain at the extremes or in the weights.
- **Regime's second number:** λ_min,exc (exact on linear graphs, the bottom of the full spectrum on nonlinear ones, mislabels the sweep) versus f̄ (chosen; reduces to λ_min,exc on linear graphs where the weight sits on eig(S)).
- **Indefiniteness:** a boolean that outranks the band (misreads backprop-like settings whose one step is exact regardless of curvature) versus `negative_weight` and `growth_min` with a growth threshold (chosen).
- **Breakdown guard:** exact zero (never fires in floating point) versus relative √eps(dtype) (chosen).
- **Regime API:** string with an optional second argument (keeps string parsing in the analysis script) versus `Regime` NamedTuple (chosen). Method name `regime` chosen over keeping `regime_label` because the return is no longer a label.
- **Module location:** `fabricpc/utils/epsilon_spectrum.py` (the first core → utils import, through the `regime` annotation) versus `fabricpc/core/epsilon_spectrum.py` (chosen; the oracle stays in utils and NumPy-only).
- **Per-update probe access:** decided in `docs/dev_plans/iter_callback_context.md` (`IterContext` for `iter_callback`, shipped first); the alternatives and their trade-offs are listed there.
- **Tracking home:** rewrite the script's loop around `train()` (two training entry points remain) versus the demo flag with the library probe (chosen; one loop, script keeps plotting).
- **CSV metadata:** parse η and T from the filename (fails on the backprop file) versus metadata columns (chosen).

## Verification

1. `python -m pytest tests/` green; `tests/test_doc_snippets.py` and `tests/test_doc_defaults.py` included.
2. `python scripts/epc_analysis.py` on CPU: Lanczos matches the oracle's two extremes and f̄ on the depth-5 chain to 1e-3; the S⁻¹r table prints; the gradient comparison uses the normalized path and reproduces the O(η) rows.
3. `python examples/resnet18_cifar10_demo.py --num_epochs 1 --track_regime 20` prints the regime with λ_min and f̄ and writes a CSV that `scripts/epc_analysis.py --plot_track` renders with four panels; the same with `--trainer backprop` renders without the 2/η line.
4. Section J: four runs complete; the (1e-2, 1) re-run reports a positive λ_max at update 1200 and the reversal flag before the crossing; λ_min at init recorded against the prediction; the reading rule applied and the outcome recorded in the three documents.

## Sequencing (one commit each, suite green at each gate)

Steps 1 and 2 do not touch the trainer and start now; the `IterContext` merge and rebase gate step 3.

1. A + oracle edits + H (oracle and spectrum tests).
2. B + inference tests; F; G's regime and spectrum migrations (`--resnet18`, `stability`).
3. After the rebase: D + probe test; E (demo flag, `spectrum_at_init`, docstring); G (delete the tracking section, plot panels, S⁻¹r table, normalized gradients).
4. I (guides, CHANGELOG entries, dev plan note, report corrections).
5. J runs; record outcomes (demo docstring, guide, report).

Commit messages via a temporary file in the project root.
