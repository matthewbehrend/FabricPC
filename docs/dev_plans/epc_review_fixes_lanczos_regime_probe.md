# Fix the ePC oracle and analysis review findings

## Context

`docs/dev_plans/epc_oracle_and_analysis_review.md` (commit 4d95c0f) found the ePC correctness work sound and the tuning diagnostics defective in four ways: the eigenvalue estimator returns the largest-magnitude eigenvalue (a real −9399 appears in `epc_lambda_track__eta0.01_T1.csv`), the regime label names an equilibrium band from the fastest mode and misprints the equilibrium condition, the T = 1 danger threshold is stated as η·λ_max > 2 when the output-layer gradient already reverses at η(λ_max − 1) > 1, and λ_max tracking is welded to the ResNet-18 demo. Commit 533a209 (after release 0.5.1's per-prediction gradient normalization, `pc_weight_gradients` / `grad_denominator`) already closed review gap 1.1: the parity test now measures its fixture's λ_max and runs both sides through the normalized path. This plan fixes everything else in the review, including the minor items, and propagates the corrections to the technical report `docs/reports/epc_regime_and_stability_report.md`.

Decisions from the 2026-09-08 interview: Lanczos from the ε-gradient replaces power iteration; `regime_label` becomes a structured `Regime`; tracking moves into the demo through a library probe that runs per update, and three GPU control runs are part of the work. The per-update hook (`iter_callback` receiving an `IterContext`) ships first as its own PR, `docs/dev_plans/iter_callback_context.md`; this branch rebases onto it.

## Symbols

| Symbol | Meaning |
|---|---|
| ε | stacked prediction errors of the unclamped nodes, the variable `EPCInference` relaxes |
| H_ε | Hessian of the total energy in ε coordinates at the feedforward point (ε = 0) |
| g0 | ∇_ε E at ε = 0, the backprop activation gradient; its Krylov space is the excited subspace |
| λ_max, λ_min,exc | largest eigenvalue of H_ε and smallest eigenvalue among the excited modes (both Ritz extremes of the Lanczos run from g0) |
| S, r | Innocenti's Theorem 1 rescaling I + JJᵀ and the feedforward output residual y − μ_y; the equilibrium output error is r S⁻¹ |
| η, T | `eta_infer`, `infer_steps` |
| f_max, f_min | relaxed fraction 1 − \|1 − ηλ\|^T of the fastest and slowest excited modes |

## Deliverables

### A. `fabricpc/utils/epsilon_spectrum.py` (new) and the oracle module

Move the "any graph" estimator out of `linear_pc_oracle.py` (delete its Part 2b, `make_top_epsilon_eigenvalue` / `top_epsilon_eigenvalue`, and migrate every caller listed in G). The oracle module keeps the NumPy linear-only code; its docstring gains two sentences: cyclic graphs are outside the oracle (the unrolled ε-energy is quadratic but warm-started carried latents make it not a pure function of ε), and λ_max of a batch is the per-sample maximum because the batch energy is a sum. `stability_bound(H)` raises `ValueError("indefinite ...")` when λ_max ≤ 0.

New module contents:

- `lanczos_extremes(hvp, v0, iters) -> LanczosResult(ritz_values, residual_min, residual_max, breakdown)`: pytree-agnostic three-term recurrence in a `lax.fori_loop` carrying (v_prev, v_cur, β_prev, α[], β[]); zero-norm guard (`β_j = 0` marks `breakdown` and freezes the remaining iterations with `jnp.where`); T tridiagonal → `jnp.linalg.eigh`; residual of each extreme Ritz pair = β_k·|s[k−1]|. Ghost eigenvalues from lost orthogonality duplicate converged extremes and do not move them; the docstring says so.
- `EpsilonSpectrum(lambda_max, lambda_min_excited, residual_max, residual_min, indefinite, gradient_norm, iters)`. `indefinite = lambda_min_excited < 0`.
- `make_epsilon_spectrum(structure, iters=30)`: compiled `(params, state, clamps, key) -> EpsilonSpectrum`. Runs `EPCInference.begin_segment`, gets `energy_of, errors` from `EPCInference.error_energy`, `hvp` via `jax.jvp(jax.grad(...))`, start vector g0 = grad at ε = 0 (random start under `lax.cond` when ‖g0‖ = 0, recorded in `gradient_norm`).
- `epsilon_spectrum(params, state, clamps, structure, iters=30, key=None)`: one-compile convenience.

Why Lanczos from g0: the Krylov space span{g0, Hg0, …} is exactly the excited subspace, so the two Ritz extremes are the two numbers the regime needs, and a negative minimum diagnoses indefiniteness instead of corrupting λ_max. Three carried vectors, so memory is independent of `iters`.

### B. `fabricpc/core/inference_epc.py`

- `Regime` NamedTuple: `eta_lambda_max`, `eta_T_lambda_max`, `f_max`, `f_min_excited`, `band` ("backprop-like" when f_max < 0.1; "near PC equilibrium" when f_min_excited > 0.9; otherwise "partially relaxed"), `output_gradient_reverses` (general form (1 − ηλ_max)^T < −1/(λ_max − 1) at unit precision; T = 1 gives η(λ_max − 1) > 1), `unstable` (ηλ_max > 2), `indefinite`, `lambda_max`, `lambda_min_excited`. `__str__` produces the one-line label; `unstable` and `indefinite` take precedence in the text.
- `regime(self, spectrum: EpsilonSpectrum) -> Regime` replaces `regime_label` (no string overload; callers migrate).
- Class docstring: equilibrium condition written as η·T·λ_min,exc ≳ 3; the T = 1 sentence replaced by the output-gradient reversal mechanism (r_1 = (1 − η(λ_max − 1)) r along the top mode); Adam sentence gains the ε caveat (holds while η·|g| ≫ Adam ε) and the SGD disparity (hidden layers η times slower than the output); one sentence that λ_max = 1 + σ_max(J)² is a weight-scale quantity controlled by normalization or weight decay, and that Goemaere et al. (Table E.10) trained ResNet-18 at (1e-3, 5) for 50 epochs without collapse on a standard parameterization.

### C. Prerequisite: `IterContext` (separate PR)

Shipped before this branch from `docs/dev_plans/iter_callback_context.md`: `iter_callback(ctx: IterContext)` with `params`, `opt_state`, `step`, `batch`, and `metrics`. Nothing in this plan changes the trainer; the probe in D consumes `IterContext` as merged.

### D. `fabricpc/training/lambda_max_probe.py` (new)

`LambdaMaxProbe(structure, probe_clamps, *, every, inference=None, iters=30, key, csv_path=None)`:
- `on_iter(ctx: IterContext)`: every `every` steps, `initialize_graph_state(structure, batch, key, probe_clamps, params=ctx.params)` → compiled `make_epsilon_spectrum`; per-edge weight Frobenius norms from `ctx.params`; `ctx.metrics["energy"]`; if `inference` is an `EPCInference`, `inference.regime(spectrum)` flags. Records one row.
- `on_epoch(ctx, accuracy=None)`: records the epoch row.
- `rows`, `first_reversal()`, `first_crossing()`, `first_chance(threshold=0.15)`, `write_csv()`, `summary()` (the outcome sentence: reversal update, crossing update, chance epoch, in order).
- CSV columns: update, epoch, lambda_max, lambda_min_excited, residual_max, indefinite, eta_lambda_max, output_gradient_reverses, unstable, train_energy, test_accuracy, then `wnorm:<edge_key>` per weight.
Export from `fabricpc.training`. Works under `algorithm="backprop"` (spectrum and norms only, no flags).

### E. `examples/resnet18_cifar10_demo.py`

- `--track_lambda_max N` (default 0 = off) and `--schedule_epochs` (default `--num_epochs`; 100 reproduces the first epochs of the 100-epoch schedule). Builds the probe on a fixed 64-sample test batch, passes `iter_callback=probe.on_iter`, and the existing `epoch_callback` also calls `probe.on_epoch(ctx, acc)`. CSV `epc_lambda_track__{trainer}_eta{eta}_T{T}.csv` (backprop: `epc_lambda_track__backprop.csv`). Prints `probe.summary()`.
- `lambda_max_at_init` → `spectrum_at_init(...) -> EpsilonSpectrum`; the settings line prints `str(inference.regime(spectrum))` plus λ_min,exc and the residual.
- Docstring: replace "threefold per epoch" with the measured phases (about 1.25× per epoch through epoch 10, then 51 → 131 → 471 → 3508 → 12539 over epochs 11–14); T = 1 sentence to the reversal mechanism; the tracking command; a control-run table filled from J.

### F. `examples/epc_spc_resnet18_compare.py`

`_demo.spectrum_at_init`; per-arm regime column from `EPCInference(...).regime(spectrum).band` with the reversal flag appended.

### G. `scripts/epc_analysis.py`

- Delete `section_track_lambda_max` and its arguments (`--track_lambda_max`, `--track_cells`, `--num_epochs`, `--schedule_epochs`, `--batch_size`, `--lr`, `--weight_decay`, `--augment`); the docstring points to the demo flag. Keep `--plot_track`: read the new columns, add a weight-norm panel, mark the first reversal and first crossing as vertical lines.
- `--resnet18`: `epsilon_spectrum` → print λ_max, λ_min,exc, residuals, `str(EPCInference().regime(spectrum))`; `regime_letter` reads `.band` and `.unstable`.
- `stability`: Lanczos against the oracle on both extremes (`excited_eigenvalues` min and `eigvalsh` max) on the depth-5 chain; gelu bracket unchanged.
- `backprop_regime`: `pc_weight_gradients` against `jax.grad` of the output energy divided by `grad_denominator` (same normalization as the test; ratios unchanged).
- `equilibrium_profile`: new table from `theorem1_energy`'s S and r: ‖r S⁻¹‖/‖r‖ (batch mean), λ_min(S), λ_max(S) per depth and init, with the paragraph: at equilibrium the output error is r S⁻¹, damping the learning signal along mode λ_S by 1/λ_S, a matrix rescaling Adam cannot undo; this is why the PC-equilibrium cells trail the backprop-like cells in the 2-epoch sweep.
- Sweep-fit wording: "heuristic" and "consistent with", never "agreement".

### H. Tests

- `tests/test_linear_pc_oracle.py`: `test_power_iteration_matches_oracle` → `test_lanczos_matches_excited_extremes` on `fork-merge`, `prior-source`, `chain-h3-std0.8` (λ_max vs `eigvalsh` max, λ_min,exc vs min of `excited_eigenvalues`, rtol 1e-3, float32 vs float64); `TestStabilityBracket` parametrized over `chain-h3` and `mupc-chain-h3` (identity activations make muPC's top-down scale the exact chain-rule factor, so the muPC bracket pins sPC+muPC's gradient scale, which the equilibrium test alone cannot); `stability_bound` raises on an indefinite H.
- `tests/test_epsilon_spectrum.py` (new): `lanczos_extremes` on an explicit matrix with eigenvalues {−50, −1, 1, 3, 10} returns 10 and −50 with `indefinite`; zero start vector → `breakdown`; on the tanh MLP fixture `epsilon_spectrum` agrees with `jax.hessian` of `energy_of` (small D, exact).
- `tests/test_inference_epc.py`: `TestRegimeLabel` → `TestRegime`: bands from (λ_max, λ_min,exc) pairs including a compact spectrum and a spread one; reversal at T = 1 and T = 5; unstable; indefinite; `str()`; parity test reads `epsilon_spectrum(...).lambda_max`.
- `tests/test_fabricpc.py`: the two lambdas take `ctx`; one assertion that `ctx.step` increments and `ctx.params` is a `GraphParams`.
- `tests/test_lambda_max_probe.py` (new, CPU): tanh MLP, 2 epochs of `train(..., iter_callback=probe.on_iter, epoch_callback=...)` with `every=2`: row count, columns, `first_crossing() is None`, weight-norm columns present, CSV round-trips through `plot_track`'s reader; also under `algorithm="backprop"`.
- Dashboarding callback tests migrate with C. `tests/test_doc_snippets.py` checks the new guide fence.

### I. Documentation touch points

- `docs/user_guides/12_api_inference.md`: rewrite the backprop-regime paragraph (correct equilibrium condition; reversal threshold; `regime` fields; Adam-ε and SGD caveats; the paper's ResNet-18 setting and the differences: muPC parameterization, no normalization layers, weight decay 1e-2, schedule length; λ_max as a weight-scale quantity and its levers); tuning-table rows for `eta_infer` (below 1/(λ_max − 1) at T = 1 for the output gradient, below 2/λ_max at every T, measured with `epsilon_spectrum`, tracked with `LambdaMaxProbe`) and `infer_steps` (η·T·λ_max ≪ 1 backprop-like; η·T·λ_min,exc ≳ 3 equilibrium); a python fence showing `LambdaMaxProbe` with `train`.
- `docs/user_guides/16_troubleshooting.md:204` and `03_how_predictive_coding_works.md`: names and condition updated.
- `CHANGELOG.md` unreleased: Breaking (`top_epsilon_eigenvalue` and `regime_label` removed in favor of `epsilon_spectrum` and `regime`), New (`epsilon_spectrum`, `Regime`, `LambdaMaxProbe`, demo `--track_lambda_max`), and the corrected regime sentence in the `EPCInference` entry.
- `docs/dev_plans/epc_reviewer_response_oracle_and_analysis.md`: a dated revision note under Context pointing to the review and this plan; B.2 and D.5 amended.
- `docs/reports/epc_regime_and_stability_report.md`: correct lines 12–13 (T = 1 mechanism; growth phases), 61 and 81 (bands and the equilibrium condition), 142 (the muPC bracket now pins the claim), 165 (Adam-ε caveat), 249 ("slowest 0.5%" → f_min,exc), 307 (the negative value was the estimator's defect; the re-run's value), 309 and 321 (control-run outcomes; growth phases), 327, 334, 340; add a subsection on the paper's ResNet-18 setting and one on the S⁻¹r damping.

### J. GPU control runs (3090, after A–H are green)

`python examples/resnet18_cifar10_demo.py --num_epochs 30 --schedule_epochs 100 --augment --activation gelu --track_lambda_max 50 --eval_every 1` for: `--inference epc --eta_infer 1e-3 --infer_steps 1` (the 100-epoch survivor), `--trainer backprop`, and `--inference epc --eta_infer 1e-2 --infer_steps 1` (re-run with the fixed estimator; expected positive λ_max at update 1200 and the reversal flag at update 1100). About 20 minutes each. Record λ_max growth, weight-norm growth, and the flags in the demo docstring, the guide paragraph, and the report; state the outcome as observed, including whether λ_max grows in the runs that do not collapse.

## Alternatives considered

- **Estimator:** shifted power iteration (10 lines, sign fix only, no λ_min,exc, equilibrium band would have to be dropped) versus Lanczos from g0 (chosen: both extremes, indefiniteness, faster convergence, 3 carried vectors). Full reorthogonalization rejected: k vectors of ε size on the ResNet-18 probe batch for no gain at the extremes.
- **Regime API:** string with an optional `lambda_min_excited` (keeps string parsing in the analysis script) versus `Regime` NamedTuple (chosen). Method name `regime` chosen over keeping `regime_label` because the return is no longer a label.
- **Per-update probe access:** decided in `docs/dev_plans/iter_callback_context.md` (`IterContext` for `iter_callback`, shipped first); the alternatives and their trade-offs are listed there.
- **Tracking home:** rewrite the script's loop around `train()` (two training entry points remain) versus the demo flag with the library probe (chosen; one loop, script keeps plotting).
- **Estimator location:** stay in `linear_pc_oracle.py` (misnamed for nonlinear graphs) versus a new `epsilon_spectrum.py` (chosen; the oracle stays NumPy-only and "obviously right").

## Verification

1. `python -m pytest tests/` green; `tests/test_doc_snippets.py` and `tests/test_doc_defaults.py` included.
2. `python scripts/epc_analysis.py` on CPU: Lanczos matches the oracle's two extremes on the depth-5 chain to 1e-3; the S⁻¹r table prints; the gradient comparison uses the normalized path and reproduces the O(η) rows.
3. `python examples/resnet18_cifar10_demo.py --num_epochs 1 --track_lambda_max 20` prints the regime with λ_min,exc and writes a CSV that `scripts/epc_analysis.py --plot_track` renders with three panels.
4. Section J runs complete; the (1e-2, 1) re-run reports a positive λ_max at update 1200 and the reversal flag before the crossing; results recorded in the three documents.

## Sequencing (one commit each, suite green at each gate)

After the `IterContext` PR merges and this branch rebases onto it:

1. A + oracle edits + H (oracle and spectrum tests).
2. B + inference tests; F; G's regime and spectrum migrations.
3. D + probe test; E (demo flag, `spectrum_at_init`, docstring); G (delete the tracking section, plot panels, S⁻¹r table, normalized gradients).
4. I (guides, CHANGELOG new entries, dev plan note, report corrections).
5. J runs; record outcomes (demo docstring, guide, report).

Commit messages via a temporary file in the project root.
