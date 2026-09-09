# Critical review: ePC oracle, regime label, and analysis script

## Context

Oracle rev 2

Reviewed the implementation of `docs/dev_plans/epc_reviewer_response_oracle_and_analysis.md` (commits 29bb1ac..30aeb83) against Innocenti et al. 2024 (arXiv 2408.11979, Theorem 1) and Goemaere et al. 2025 (arXiv 2505.20137, Appendix C.3, Table E.10). Verified by: re-deriving the oracle math, running `tests/test_linear_pc_oracle.py` and `tests/test_inference_epc.py` (89 passed, 22 s), and reading the two `epc_lambda_track__*.csv` files and `epc_lambda_track.log`.

Two questions: (1) does the work verify FabricPC's ePC code, and (2) does it help a user pick `eta_infer` and `infer_steps` inside a stable regime.

## Symbols

| Symbol | Meaning |
|---|---|
| ε_t | prediction error of node t, the variable `EPCInference` relaxes |
| H_ε | Hessian of the total energy in ε coordinates; λ_max its top eigenvalue |
| J | map from the stacked hidden ε to the output prediction μ_y |
| S | I + JJᵀ on a chain at unit precision; Innocenti's Theorem 1 rescaling. eig(S) = the excited eigenvalues of H_ε from ε = 0 |
| r | y − μ_y at the feedforward point (the backprop residual) |
| η, T | `eta_infer`, `infer_steps` |

## Verdict

Correctness verification: strong. The oracle is independent of node and solver code, its math is right (I re-derived the assembly, the precision-weighted Theorem 1, H_ε = MᵀH_zM, and the excited-spectrum = eig(S) claim), and the solver tests pin both the equilibrium and the gradient scale. Four gaps, none blocking.

Tuning utility: weak in its current form. One measurement defect can silently report a wrong stability bound on nonlinear graphs; the regime label's equilibrium band is misnamed; the T = 1 danger threshold is too lenient; and the only tool that reflects what actually happened during training (λ_max tracking) is welded to the ResNet-18 demo and cannot run on a user's model. The mechanism claim "crossing 2/η preceded collapse" is supported as ordering only; the control runs that would make it causal were not run, and the ePC paper trained ResNet-18 at the same (η, T) without collapse, which the docs do not mention.

---

## Part 1: correctness verification

### Sound, verified

- **Oracle assembly.** Row block for node t is √p_t·(z_t − Σ_s z_s W_eff − b_t) with clamps moved to c; checked for unclamped and clamped t. B strictly lower-triangular, M = (I − B)⁻¹, H_ε = MᵀAᵀAM. `linear_pc_oracle.py:285-303`.
- **Theorem 1 with precisions.** Stationarity gives ε_l* = (p_y/p_l)·ε_y*·P_lᵀ and ε_y* = r S⁻¹ with S = I + Σ_l (p_y/p_l) P_lᵀP_l, so E* = ½ p_y r S⁻¹ rᵀ. Matches `theorem1_energy` and Innocenti's S = I + Σ_{ℓ≥2} W_{L:ℓ}W_{L:ℓ}ᵀ in row convention.
- **Solver tests pin scale, not just direction.** Equilibrium on 12 graph shapes for both solvers, the 0.95×/1.05× stability bracket on `chain-h3`, and the HVP-vs-H_ε check at 1e-4 together fix the ε-gradient's magnitude. Two wrong solvers agreeing is ruled out.
- **1-step identity.** ε_1 = −η·∇_z L exactly; the paper's Case 1 proof (p. 23) does treat ∂ŝ_i/∂θ_i as evaluated at the unperturbed point, so plan fact 5 (weight-gradient parity is first-order only under derive-then-gradient) is correct and the test's h1-exact / h2,y-O(η) split is the right assertion.
- **Cost.** 89 tests in 22 s; no `slow` marks needed.

### Gaps

1. **Parity test never measures λ_max of its fixture** (`test_inference_epc.py`, `test_one_step_weight_grads_are_eta_backprop_first_order`). The docstring claims first order in η·λ_max; the test checks linear scaling in η at one unknown λ. Compute λ_max with `top_epsilon_eigenvalue` in the test and assert d(η) ≤ C·η·λ_max, or parametrize weight std so λ varies and the constant is pinned.
2. **The sPC+muPC equilibrium test does not test what its docstring says** (`test_linear_pc_oracle.py:514-530`). A positive diagonal preconditioner shares the fixed point with plain gradient descent, so reaching the oracle cannot distinguish "exact gradient descent on the scaled energy" from preconditioned descent. Only the plain `chain-h3` bracket pins sPC's scale. If the muPC claim matters, add a `mupc-chain-h3` bracket; otherwise reword the docstring to "shares the equilibrium".
3. **Cyclic graphs have no oracle** (by design; `validate_linear_gaussian` rejects them). The unrolled ε-energy at degree U is still quadratic for linear nodes, but the warm-started carried latents make it not a pure function of ε. State this as a coverage limit in the oracle docstring; ePC's cyclic path is covered only by the existing equivalence tests.
4. **Power iteration has no convergence indicator** (`linear_pc_oracle.py:550-597`). The test needed 300 iterations for rtol 1e-4 on 4-node graphs; the demo and probe use 30. For a positive-definite H the Rayleigh quotient is a lower bound on λ_max, so an unconverged value overstates the safe rate 2/λ_max. Return the relative change over the last iterations (or the residual ‖Hv − λv‖) beside λ.

---

## Part 2: tuning within a stable regime

### Defects

1. **Power iteration returns the largest-magnitude eigenvalue, not the largest positive one.** `epc_lambda_track__eta0.01_T1.csv` row update 1200: λ_max = −9398.86. On a gelu + softmax-CE graph the ε-Hessian is indefinite (second derivatives of the network output enter with the loss gradient as weight), so this is a real negative eigenvalue, not noise. Consequences: `regime_label(−9398)` returns "backprop-like" (x < 0 makes f_max negative, below the 0.1 band); `first_cross` (`epc_analysis.py:903`) tests `eta*lam > 2` and never fires; the demo would print a meaningless η_max = 2/λ. Fix: a second power iteration on H + |λ_1|·I when the first returns λ_1 < 0 (two passes, no new dependency), or a short Lanczos run returning both extremes. Guard `regime_label` and `stability_bound` against λ ≤ 0 with an explicit "indefinite" message.

2. **The "near PC equilibrium" band is a misnomer, and the equilibrium condition is misprinted in three places.** f_max > 0.9 says the fastest excited mode relaxed 90%; equilibrium needs the slowest excited mode relaxed, η·T·λ_min,excited ≳ 3. The class docstring (`inference_epc.py:75`), the guide paragraph, and the tuning table all read "the regime parameter is eta_infer·T·λ_max: … ≳ 3/λ_min,excited reaches the PC equilibrium", which as written says η·T·λ_max ≳ 3/λ_min, a dimensional error. Write η·T ≳ 3/λ_min,excited (equivalently η·T·λ_max ≳ 3κ_excited). On the ResNet-18 the excited spectrum happened to be compact (fitted λ_eff = 12 against λ_max = 16.4), so the labels matched the sweep there; that is not general. Either let `regime_label` take an optional `lambda_min_excited` and only then name the equilibrium band, or rename the top band "fastest mode relaxed > 90%". Also `f_min` hard-codes the floor eigenvalue as 1 (`inference_epc.py:117`), wrong for any precision ≠ 1.

3. **The T = 1 threshold is too lenient.** With ε_1 = −η∇_z L the re-derived output residual is r_1 = (I − η(S − I)) r at unit precision, so along the top mode r_1 = (1 − η(λ_max − 1)) r_λ: the output layer's local weight gradient reverses sign on that mode at η(λ_max − 1) > 1, well before the iteration bound η·λ_max > 2. The data agree: in the (1e-2, 1) run λ_max passed 100 (η·λ = 1.01) at update 1100, test accuracy fell that epoch (43.2% → 40.6%), and the network was at chance one epoch later; the (0.1, 1) sweep cell sat at η·λ_max = 1.6 at init and collapsed at 2 epochs while (0.03, 1) at 0.49 did not. For T > 1 the sign-flip condition is (1 − ηλ)^T < −1/(λ − 1), between 1 and 2 depending on T. Promote the existing "overshooting" flag at x > 1 to the warning users act on, and state the sign-flip mechanism in the guide instead of "lands farther from equilibrium than it started", which describes the ε distance rather than what damages the weight update.

4. **The actionable diagnostic is not reusable.** `regime_label` at init is honest but not predictive: the defaults read backprop-like at init and collapsed. The tool that tracks reality, `--track_lambda_max`, is bound to the demo: `load_demo()`, `demo._create_mupc_model`, `demo.make_optimizer`, `Cifar10Loader`, `demo.AugmentedCifar10Loader` (`epc_analysis.py:836-956`). A user with their own graph cannot run it. The trainer already exposes `epoch_callback(ctx)` with `ctx.params` (`fabricpc/training/trainer.py:415, 583-594`). Move the probe into the library: `fabricpc/utils/lambda_max_probe.py` with a `LambdaMaxProbe(structure, probe_clamps, eta, iters, log=...)` callable usable as `epoch_callback` (and optionally `iter_callback` every N updates) that logs λ_max, η·λ_max, and the flip/bound flags; the script's tracking section becomes a thin caller. Document it in the tuning table as the way to pick η.

5. **Control runs are missing, so "preceded" is not "caused".** Both tracked cells collapsed. To separate "λ_max grows regardless and a fixed η eventually crosses the bound" from "ePC's gradient distortion beyond η·T·λ ≈ 0.2 drives the weight growth", track (1e-3, 1), which survived 100 epochs, and the backprop trainer (`--trainer backprop` exists in the demo). Log per-layer weight norms beside λ_max so growth can be attributed to Π‖W_l‖ versus Jacobian changes. Also correct the demo docstring: "λ_max grew about threefold per epoch once training was under way" misdescribes the CSV, which shows about 1.25× per epoch through epoch 10 (17 → 51) and then a runaway (51 → 131 → 471 → 3508 → 12539 over epochs 11–14).

6. **The paper trained ResNet-18 at the same (η, T) without collapse, and the docs do not say so.** Goemaere et al. Table E.10: e_lr fixed at 1e-3, e_momentum 0, T = 5, Adam on weights, 50 epochs for ResNet-18, no instability reported. FabricPC's identical (1e-3, 5) collapsed at epoch 20. Differences: muPC parameterization with `include_output=False`, no normalization layers (none exist in `fabricpc/nodes`), weight decay 1e-2 versus the paper's ≤ 1e-3, gelu, a 100-epoch schedule. λ_max = 1 + σ_max(J)² is a weight-scale quantity; normalization layers or weight-norm control bound it, and the guide should name those levers rather than only "pick η below 2/λ_max with margin".

7. **Two caveats missing from the Adam sentence.** "Under Adam the η scaling is normalized away" holds only while η·|g| ≫ Adam's ε (1e-8); at η_infer = 1e-4 hidden-layer gradients of order 1e-7 are damped by ε. Without Adam (plain SGD) hidden layers learn η_infer times slower than the output layer, a 1000× disparity at the default. Neither appears in the guide.

8. **Innocenti's rescaling is the unused explanation for the sweep.** At equilibrium the output error is S⁻¹r: the learning signal along mode λ_S is damped by 1/λ_S, up to 16× on this graph, and Adam cannot undo a matrix rescaling. That is the direct answer to why the PC-equilibrium cells (31%) trail the backprop-like cells (38.8%) at 2 epochs. `equilibrium_profile` shows per-layer energies, not this. Add ‖S⁻¹r‖/‖r‖ and eig(S) to that section and one sentence to the sweep interpretation.

9. **The sweep fit is a heuristic.** Mapping accuracy linearly onto relaxed fraction and fitting one eigenvalue is a narrative device; the 1.4 ratio is weak evidence. Fine in the script, but the guide should not cite it as agreement.

### Minor

- The probe batch is 64 unaugmented test samples; training uses 256 augmented samples. λ_max is a per-sample maximum (the batch energy is a sum and decouples per sample), so the training batch's bound is at least as tight. One sentence in the docstring.
- `normalize` in the power iteration divides by zero if Hv = 0 (λ = 0 direction); guard.
- `regime_label` prints "slowest 0.5%" from the hard-coded floor even when no floor mode is excited; drop it or label it "floor λ = 1".

---

## Recommended changes, prioritized

**P1, defects (small, self-contained):**
1. `linear_pc_oracle.make_top_epsilon_eigenvalue`: shifted second pass when λ_1 < 0; return a convergence residual; guard zero norm. `stability_bound` and `regime_label` raise or label "indefinite" for λ ≤ 0. Test: an indefinite quadratic (hand-built H with a dominant negative eigenvalue) returns the positive top eigenvalue.
2. Fix the three "≳ 3/λ_min,excited" statements (class docstring, guide paragraph, tuning table). `regime_label(lambda_max, lambda_min_excited=None)`: equilibrium band only when the second argument is given; otherwise "fastest mode relaxed X%". Use the graph's minimum precision for the floor, or drop f_min.
3. T = 1 sign-flip threshold: label x > 1 as "output-layer gradient reverses on the top mode" and document it in the guide and class docstring; keep x > 2 as "iteration unstable".

**P2, reusable tuning tool:**
4. `fabricpc/utils/lambda_max_probe.py` (or under `training/`): `LambdaMaxProbe` for `train(..., epoch_callback=...)` / `iter_callback`, logging λ_max, η·λ_max, per-layer weight norms, and the P1 flags. Migrate `section_track_lambda_max` to it; add `--track_trainer {pc,backprop}` and run `--track_cells 1e-3:1` plus backprop as controls. Update the demo docstring's growth description from the CSV.

**P3, analysis and docs:**
5. Add S⁻¹r damping (‖S⁻¹r‖/‖r‖, eig(S)) to `equilibrium_profile` and a sentence to the sweep interpretation.
6. Guide: the paper's ResNet-18 setting and the differences (muPC, no normalization, weight decay, schedule); the Adam-ε and SGD caveats; λ_max as a weight-scale quantity and its levers.
7. Parity test measures λ_max and asserts d(η) ≤ C·η·λ_max; muPC stability bracket or reworded docstring.

## Alternatives considered

- **Eigenvalue fix:** Lanczos (fewer iterations, both extremes, more code) versus shifted power iteration (two passes, ~10 lines, reuses the existing HVP). Shifted power iteration recommended; Lanczos if probe cost during training matters.
- **Probe integration:** `epoch_callback` (no trainer change, per-epoch granularity) versus a trainer flag `probe_every=N` (per-update, touches `make_train_step`) versus keeping the standalone loop (status quo, not reusable). Callback recommended; `iter_callback` already gives per-update access if needed.
- **Adaptive rate now** (η = c/λ_max per probe) versus tracking only. Deferred as the plan decided; the controls in P2 decide whether an adaptive rate or weight-norm control is the right follow-up.

## Verification

- `python -m pytest tests/test_linear_pc_oracle.py tests/test_inference_epc.py -q` green, including the new indefinite-Hessian and λ_max-scaled parity tests.
- `python scripts/epc_analysis.py` on CPU prints the S⁻¹r damping row and the convergence residual beside every power-iteration value.
- GPU: `--track_lambda_max 50 --num_epochs 30 --schedule_epochs 100 --augment --track_cells 1e-3:1` and `--track_trainer backprop` complete and report whether λ_max grows in the surviving runs; the (1e-2, 1) re-run reports a positive λ_max at update 1200 and flags the x > 1 crossing at update 1100.
