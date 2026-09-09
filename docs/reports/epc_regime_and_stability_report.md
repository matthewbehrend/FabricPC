# ePC in FabricPC: exact oracle, backprop regime, and the stability bound

Technical report, 2026-09-04. Code at commit `30aeb83` on branch `matthew_cedric/epc`. GPU sections on one NVIDIA RTX 3090 (CUDA 13, JAX 0.10.2); everything else on CPU.

Revision 2026-09-08 (a). The branch was rebased onto release 0.5.1, which shipped the per-prediction gradient normalization (`pc_weight_gradients`, `grad_denominator`); the commits cited here are now `45c18d6..e84f667`. The weight-gradient parity test of Section 5.3 was rewritten for that release: it runs at batch 3 instead of batch 1, compares `pc_weight_gradients` against a backprop reference divided by the same prediction count, measures the fixture's λ_max, and asserts the deviation bound d(η) ≤ 10·η·λ_max at two weight scales. The sweep (Section 5.8) and the 100-epoch runs were not repeated and predate the normalization; the Section 5.8 tracking and its growth phases are superseded by the normalized trainer's control runs (Section 5.9).

Revision 2026-09-08 (b), after the second review round recorded in `docs/dev_plans_archive/epc_inference_solver.md` (Sections 3.4 and 4, and its Record). The power-iteration estimator is replaced by the Lanczos estimator `fabricpc.core.epsilon_spectrum` (both excited extremes, the gradient weight per Ritz mode, Ritz residuals); `regime_label` by `EPCInference.regime(spectrum) -> Regime` (band on the gradient-weighted relaxed fraction f̄, output-gradient reversal flag, indefiniteness by weight and growth); the analysis script's tracking loop by `fabricpc.training.RegimeProbe` inside `train`. Sections 2.3, 2.4, 5.2, 5.7, and 5.8 are amended in place; Sections 5.9 to 5.11 are new. Numbers from the replaced estimator are marked as such where they remain.

## 1. Summary

FabricPC ships two inference solvers for predictive coding (PC). The state-based solver `InferenceSGD` (sPC) relaxes the latent activities by local gradient descent. The error-based solver `EPCInference` (ePC, after Goemaere et al. 2026) relaxes the prediction errors instead and derives the latents by one forward pass per step, so one global reverse pass delivers the output signal to every layer. This report does three things.

1. It builds an exact oracle for linear-Gaussian networks. On such a network the PC energy is a quadratic, its minimizer is a least-squares solution, and Innocenti et al. (2024, Theorem 1) give a closed form on chains that cross-checks the oracle. Both solvers are verified against it, including their exact stability bounds.
2. It shows that ePC's regime is set by the curvature the starting gradient excites, not by η·T alone. Here η is the error learning rate `eta_infer`, T the step count `infer_steps`, λ an eigenvalue of the energy's Hessian in error coordinates, and λ_max the largest one the starting gradient excites. Small η·T·λ on the modes that carry the gradient makes ePC backprop with rescaled gradients (the paper's Theorem C.9); η·λ_max < 2 is required for stability at every T; and at odd T the output layer's weight gradient reverses sign along the top mode once (1 − ηλ_max)^T < −1/(λ_max − 1), at T = 1 once η(λ_max − 1) > 1, before the iteration bound.
3. It measures the excited spectrum on the muPC ResNet-18 demo. At init λ_max = 16.4 and the 2-epoch accuracy sweep is fitted by a single eigenvalue of 12.0; the Hessian is indefinite at init (λ_min = −0.42, 1.2% of the gradient weight on negative curvature), and most of the gradient weight sits near the precision floor (f̄ = 0.010 against f_max = 0.080 at the defaults), so the accuracy transition follows the top modes' relaxation. Four 30-epoch control runs on the normalized trainer (Section 5.9) separate the two candidate causes of that growth: λ_max drifts from 22 to 24 under backprop and to 29 under ePC at (1e-3, 1), both of which reach 68 to 69% accuracy, while at the defaults (1e-3, 5) it goes from 27 at epoch 8 to 31 000 at epoch 15 and the run collapses to chance, and the Frobenius norm of every convolution weight falls on the same weight-decay schedule in all four runs. The growth is driven by ePC's relaxation, not by the weight scale, and the output-gradient reversal flag fires one probe before the stability crossing in both collapsing cells.

| Quantity | Value | Source |
|---|---|---|
| λ_max(H_ε) at init, muPC ResNet-18, 64-sample test batch | 16.45 (Ritz residual 4.6e-4), so η_max = 2/λ_max = 0.122 | `scripts/epc_analysis.py --resnet18` |
| λ_min at init, same batch | −0.42 (indefinite; residual 0.11), 1.2% of the gradient weight on negative curvature | `--resnet18` |
| λ_eff fitted to the 2-epoch sweep (45 cells, η ≤ 0.01) | 12.0, rms residual 0.049 (heuristic) | `--section backprop_regime` |
| Library defaults `EPCInference(eta_infer=1e-3, infer_steps=5)` at init | η·T·λ_max = 0.082, f̄ = 0.010, f_max = 0.080: backprop-like | `EPCInference.regime` |
| Defaults, 100-epoch run | 54.76% at epoch 10, 9.68% at epoch 20 | `sweep_eta0.001_steps5.log` |
| Defaults, tracked (first 30 epochs of the 100-epoch schedule; power-iteration estimator, batch-summed gradients) | η·λ_max first above 2 at update 2700 (epoch 14); chance at epoch 15 | `docs/reports/data/epc_lambda_track__eta0.001_T5.csv` |
| η = 1e-2, T = 1, tracked (same estimator and trainer) | crossing at update 1150 (epoch 6); chance at epoch 7 | `docs/reports/data/epc_lambda_track__eta0.01_T1.csv` |
| Control run (1e-3, 5), normalized trainer, probed every 50 updates | reversal flag at update 2700 (epoch 14), η·λ_max > 2 at 2750 (epoch 15), chance at epoch 15; λ_max 22 → 31 000 | `docs/reports/data/epc_regime_track__pc_eta0.001_T5.csv` |
| Control run (1e-3, 1) | 67.7% at epoch 30; λ_max 22 → 29; no flag | `docs/reports/data/epc_regime_track__pc_eta0.001_T1.csv` |
| Control run, backprop trainer | 69.0% at epoch 30; λ_max 22 → 24 | `docs/reports/data/epc_regime_track__backprop.csv` |
| Control run (1e-2, 1) | reversal at update 1150 (epoch 6), crossing at 1200 (epoch 7), chance at epoch 7; λ_max = 6132 and λ_min = −354 at update 1200 | `docs/reports/data/epc_regime_track__pc_eta0.01_T1.csv` |
| Steps to contract by 1e-3, linear chain, depth 20 | sPC 30 343, ePC 75 | `--section convergence_spectra` |
| Lanczos vs oracle, depth-5 chain: λ_max, excited λ_min, f̄ | relative errors 1.9e-7, 4.3e-8, 2.8e-9 | `--section stability` |
| Test suite at the final commit of this work (2026-09-09) | 654 passed, 6 skipped | `python -m pytest tests/` |

## 2. Background

### 2.1 The energy and the two solvers

A FabricPC graph has nodes t with latent activity z_t (one row per sample) and a prediction μ_t computed from the node's in-edge sources. The training energy sums over every node that receives at least one edge:

    E = Σ_t ½ · p_t · ‖z_t − μ_t‖²,     ε_t = z_t − μ_t,

with p_t the node's Gaussian precision (1.0 unless set). Training clamps the input node to the data and the output node to the target, minimizes E over the remaining latents (inference), then updates each weight from its own node's error and inputs. The two solvers minimize the same E in two coordinate systems. sPC moves z_t down ∂E/∂z_t, and because each node's gradient involves only its neighbors, information travels one edge per step. ePC treats the errors ε_t as the variables: the latents are derived along the topological order as z_t = μ_t + ε_t, so a change in one ε moves every downstream latent, and one reverse pass gives ∂E/∂ε for all nodes at once. The two parameterizations are related by a triangular bijection with unit determinant, so they share minima (Goemaere et al. 2026, Appendix C).

### 2.2 Why a linear network gives an exact answer

When every node is `Linear` or `IdentityNode` with the identity activation and the Gaussian energy, μ_t is an affine function of the upstream latents, so E is a quadratic in the stacked free latents z_free:

    E = ½ ‖A z_free − c‖².

A has one row block per node with in-degree > 0 (that node's residual, scaled by √p_t) and one column block per unclamped node; c collects biases and clamps. The minimizer z* is the least-squares solution, unique when A has full column rank. From z* the oracle reads every node's equilibrium prediction, error, and energy. It uses only the parameters, the edge list, the muPC forward scales, biases, and precisions, never the node or solver code, so it is an independent reference (`fabricpc/utils/linear_pc_oracle.py`).

**Innocenti's Theorem 1, in words.** Take a chain x → h_1 → ⋯ → h_L → y with x and y clamped. Let r = y − μ_y be the output residual at the feedforward point (all hidden errors zero). Each hidden error ε_l moves the output prediction by ε_l P_l, where P_l = W_{l+1} ⋯ W_{L+1} is the product of the weights downstream of layer l. Inference is a trade: absorbing part of r into a hidden error costs ½‖ε_l‖² there but saves output energy. The optimum pulls the output error back through the downstream maps, ε_l* = ε_y*·P_lᵀ, and leaves an output error ε_y* = r·S⁻¹ with

    S = I + Σ_l P_lᵀ P_l,     E* = ½ · r S⁻¹ rᵀ   (per sample).

S is the identity plus the summed leverage of the hidden layers over the output. Large downstream gains mean large S, so the error is absorbed upstream and E* is small. Innocenti et al. (2024) prove this as Theorem 1 (the equilibrated energy of a deep linear network is a rescaled mean-squared error); the oracle extends it to per-node precisions and biases and checks it against the least-squares solution.

*Worked example.* Scalar chain x = 1, W_1 = 2, W_2 = 3, y = 1. The feedforward output is 6, so r = −5 and S = 1 + 3² = 10. Then E* = ½ · 25 / 10 = 1.25, ε_y* = −0.5, ε_h* = ε_y*·3 = −1.5, and z_h* = μ_h + ε_h* = 2 − 1.5 = 0.5. Direct minimization of E(z) = ½(z − 2)² + ½(1 − 3z)² gives the same z = 0.5 and E = 1.25. This example is `TestOracleSelfChecks::test_scalar_chain_hand_numbers`.

### 2.3 Gradient descent on a quadratic: modes, the relaxed fraction, and the 2/λ threshold

Both solvers are gradient descent on a quadratic, so one scalar picture explains their behaviour. For E = ½ λ x², a step x ← x − η·λ·x multiplies the distance to the minimum by (1 − ηλ):

| η·λ | one step does | T steps leave |
|---|---|---|
| 0 < η·λ < 1 | moves part of the way, same side | (1 − ηλ)^T of the distance |
| η·λ = 1 | lands exactly on the minimum | 0 |
| 1 < η·λ < 2 | overshoots to the other side, closer | \|1 − ηλ\|^T, alternating sign; at odd T the output residual along the mode, (r/λ)·[1 + (λ − 1)(1 − ηλ)^T], has reversed sign once (1 − ηλ)^T < −1/(λ − 1), so the output layer's weight gradient points the wrong way there (T = 1: η(λ − 1) > 1) |
| η·λ = 2 | lands the same distance away, opposite side | no progress |
| η·λ > 2 | lands farther away than it started | grows as \|1 − ηλ\|^T |

Every quadratic in many variables decomposes into independent such modes along the eigenvectors of its Hessian H, each with its own eigenvalue λ. The step size must satisfy η < 2/λ_max for the stiffest mode, and that one bound governs the whole system. After T steps mode λ has closed the fraction

    f(η, T, λ) = 1 − (1 − ηλ)^T

of its distance to equilibrium. The slowest mode sets how many steps are needed (κ = λ_max/λ_min steps, roughly, at η = 1/λ_max), the fastest sets the largest usable η.

**Applied to ePC.** The Hessian of E in error coordinates is H_ε = Mᵀ Aᵀ A M, where M = (I − B)⁻¹ maps errors to latents and B is the strictly lower-triangular edge map. On a chain with unit precision, H_ε = I + JᵀJ with J the map from the stacked errors to the output prediction, so λ_max(H_ε) = 1 + σ_max(J)², which grows with the product of the downstream weights. Starting from ε = 0 (the feedforward state every FabricPC run starts from), only the modes with a nonzero initial gradient move: these are the d_y eigen-directions of S, and the floor modes at λ = precision never move.

**The backprop regime.** At ε = 0 the gradient ∂E/∂ε_t is exactly the backprop activation gradient g_t = ∂L/∂z_t, because z depends on ε through the same chain of Jacobians backprop uses. One step therefore leaves ε_t = −η·g_t. If every excited mode has η·T·λ ≪ 1, the T-step result is ε ≈ −η·T·g to first order, and the local weight gradients computed from those errors are backprop's, scaled by η·T on hidden layers and unscaled on the output. Goemaere et al. (2026, Appendix C.3, Theorem C.9) state the two cases, T = 1 and "λ sufficiently small relative to 1/T" (their λ is our η). The second condition hides the network's Jacobian scale: the quantity that must be small is η·T·λ_max, and λ_max = 1 + σ_max(J)² is a property of the weights and depth, not a constant. This is why a rule stated in η·T alone cannot transfer between networks, and why the same (η, T) drifts from the backprop regime into the PC regime and then into instability as the weights grow during training.

### 2.4 The regimes

Only the modes along which the starting gradient g0 = ∇_ε E has a component move, and each carries a fraction w of ‖g0‖². The band reads the gradient-weighted relaxed fraction f̄ = Σ w·f(λ) over the positive-curvature modes (`Regime.f_weighted`); `Regime.f_max` = f(λ_max) is the fastest mode's fraction. On a linear graph g0 lies in the row space of J, every w sits on an eig(S) mode, and f̄ averages f over that band.

| condition | ePC computes | `Regime` |
|---|---|---|
| η·T·λ ≪ 1 on the modes that carry the gradient | backprop's gradients, hidden layers scaled by η·T | `band` "backprop-like" (f̄ < 0.1) |
| some modes relaxed, the bulk not | a mix; accuracy moves from the backprop value toward the PC value | "partially relaxed" (0.1 ≤ f̄ ≤ 0.9) |
| η·T·λ ≳ 3 on the modes that carry the gradient | the PC equilibrium gradients | "near PC equilibrium" (f̄ > 0.9) |
| odd T with (1 − ηλ_max)^T < −1/(λ_max − 1) | the output layer's weight gradient reversed along the top mode | `output_gradient_reverses` |
| negative curvature carrying weight, (1 + η\|λ_min\|)^T > 1.1 | the negative modes grow instead of relaxing | `growth_min`, `negative_weight`; `str()` prints "indefinite" |
| η·λ_max > 2 | the top mode diverges; garbage errors and weight gradients | `unstable` |

Section 5.8 shows that on the ResNet-18 the accuracy transition follows f_max, not f̄, because most of the gradient weight sits on low-curvature modes whose relaxation changes error magnitudes that Adam absorbs.

## 3. Symbols

| Symbol | Meaning |
|---|---|
| z_t, μ_t, ε_t | node t's latent, prediction, and error z_t − μ_t (`NodeState.z_latent`, `z_mu`, `error`) |
| p_t | Gaussian precision of node t (default 1.0) |
| E | total energy over nodes with in-degree > 0 |
| A, c | the quadratic form E = ½‖A z_free − c‖² over the stacked free latents |
| H_z, H_ε | Hessians of E in latent coordinates (AᵀA) and error coordinates (MᵀAᵀAM) |
| λ, λ_max, λ_min | eigenvalues of a Hessian; the top one sets the stability bound 2/λ_max |
| P_l, S | downstream weight product from hidden layer l to the output; S = I + Σ_l P_lᵀP_l |
| r | feedforward output residual y − μ_y |
| J | map from the stacked errors to the output prediction; λ_max(H_ε) = 1 + σ_max(J)² |
| η, T | `eta_infer`, `infer_steps` |
| f(λ), f_max | relaxed fraction 1 − (1 − η·λ)^T of a mode with eigenvalue λ after T steps; f_max = f(λ_max) |
| g0 | ∇_ε E at ε = 0, the starting gradient; equals the backprop activation gradient at every node |
| θ_k, w_k | the k-th Ritz value of the Lanczos tridiagonal matrix (an eigenvalue estimate) and the fraction of ‖g0‖² that mode carries; Σ_k w_k = 1 |
| f̄ | gradient-weighted relaxed fraction Σ_{θ_k > 0} w_k·f(θ_k) / Σ_{θ_k > 0} w_k over the positive Ritz modes |
| α_j, β_j | the Lanczos recurrence coefficients, the diagonal and off-diagonal of the tridiagonal matrix; the Ritz residual of an extreme is β_k·\|s_{k−1}\| with s the Ritz vector's last component |
| λ_eff | one eigenvalue fitted to the 2-epoch sweep through f |

## 4. What was built

| Component | Location | Purpose |
|---|---|---|
| Linear-Gaussian oracle | `fabricpc/utils/linear_pc_oracle.py` | exact z*, ε*, per-node E*; Theorem 1 closed form; H_z, H_ε; stability bound; excited spectrum; relaxed fraction; steps to contract |
| Lanczos spectrum on any graph | `fabricpc.core.epsilon_spectrum` (`epsilon_spectrum`, `make_epsilon_spectrum`) | λ_max, λ_min, Ritz values with gradient weights, `negative_weight`, Ritz residuals from Hessian-vector products through `EPCInference.error_energy`, nonlinear graphs included; replaces the power iteration of revision (a) |
| Solver hook | `EPCInference.error_energy` | the ε-energy closure, one owner for the solver's gradient, the HVP, and the Lanczos estimator |
| Regime verdict | `EPCInference.regime(spectrum) -> Regime` | `unstable`, `output_gradient_reverses`, `f_weighted`, `f_max`, `band`, `negative_weight`, `growth_min`; `str()` is the one-line label the demos print at init; replaces `regime_label` |
| Regime probe | `fabricpc.training.RegimeProbe` | a `train` callback recording the spectrum, the regime flags, and every weight's Frobenius norm every N updates, plus the test accuracy per epoch, to a CSV; `first_reversal`, `first_crossing`, `first_chance`, `growth_phases`, `summary` |
| Tests | `tests/test_linear_pc_oracle.py`, `tests/test_epsilon_spectrum.py`, `tests/test_inference_epc.py`, `tests/test_regime_probe.py` | 89 tests across the first and third at `30aeb83`; the count after revision (b) is in Section 7 |
| Analysis script | `scripts/epc_analysis.py` | four CPU sections (about 20 s), `--resnet18` (GPU), `--plot_track` |
| Demo output | `examples/resnet18_cifar10_demo.py` | prints the regime and the spectrum at init; `--track_regime N` runs the probe; docstring records the 100-epoch outcomes and the control-run table |

## 5. Results

### 5.1 The oracle against itself

| Check | Fixtures | Tolerance | Result |
|---|---|---|---|
| Hand-computed scalar chain (Section 2.2) | 1 | exact | pass |
| Theorem 1 closed form = least squares | chains of depth 1–4, with biases, with precisions (2.0, 0.5), muPC | rtol 1e-10 | pass |
| Precision-weighted pull-back p_l·ε_l* = p_y·ε_y*·P_lᵀ | chains | atol 1e-10 | pass |
| Unclamped readout: E* = 0, z* = feedforward | 1 | atol 1e-12 | pass |
| Normal equations Aᵀ(Az* − c) = 0 | fork-merge, prior source, clamped internal node | 1e-10 relative | pass |
| H_ε = diag(p) + Σ_clamped p_t J_tᵀJ_t, det M = 1, B strictly lower-triangular | 3 graphs | atol 1e-10 | pass |
| Eigenvalue floor: λ_min(H_ε) ≥ min precision on chains; < 1 with an unclamped source | 6 + 1 | — | pass |
| Validator rejects tanh, cross-entropy, flattened input, StorkeyHopfield, cycles at unroll 1 and 2 | 6 | raises | pass |

The cycle check matters: a cycle unrolled once visits each member once, so the schedule length equals the node count, and only a back-edge test against `node_order` detects it (`tests/test_topological_schedule.py:121-124` pins the schedule).

### 5.2 Both solvers reach the oracle

Twelve graphs: chains of depth 1–4 (input 5, hidden 4, output 3, weight std 0.3, batch 3), a chain with drawn biases, a chain with precisions 2.0 and 0.5, a stiff chain (std 0.8, λ_max(S) ≈ 69), a fork-merge, a chain with a clamped internal node, the convex DAG with an unclamped prior source, an unclamped readout, and a muPC chain (muPC initializer and forward scales, Xavier readout). Each solver runs at η = 1/λ_max of its own Hessian for the number of steps the oracle predicts contracts every relevant mode below 1e-5 of the initial distance.

| Test | Assertion | Result |
|---|---|---|
| `TestEPCReachesOracle` (12 graphs) | z_latent, per-node energy, total energy, and error (unclamped sources included) within rtol/atol 1e-4 | 12 pass |
| `TestSPCReachesOracle` (12 graphs) | same, source error skipped (sPC re-syncs source predictions) | 12 pass |
| `TestStabilityBracket[epc]` on the depth-3 chain and the muPC depth-3 chain | η = 0.95·(2/λ_max(H_ε)) reaches the oracle; η = 1.05·bound for 150 steps: finite, energy strictly increasing over the last 50 | pass |
| `TestStabilityBracket[spc]` | same with H_z, both chains | pass |
| `TestEpsilonHVPMatchesOracle` | Hessian-vector product through `error_energy` = H_ε v (atol 1e-4); Lanczos λ_max = max eig(H_ε) and λ_min = min excited eigenvalue (rtol 1e-3), f̄ = the oracle's weighted fraction (atol 1e-3), on four graphs in float32 and float64 | 10 pass |

The muPC chain under sPC confirms a design fact: with identity activations the muPC top-down scale equals the chain-rule factor of the pre-scaled input (`jacobian_gain` = 1) and the self-gradient scale is 1, so sPC with muPC is plain gradient descent on the input-scaled energy. The equilibrium test alone cannot pin that, because a diagonal preconditioner shares the fixed point; the muPC stability bracket does, and since revision (b) it runs. The stability brackets pin the scale of each gradient implementation, not only its direction: a gradient off by a constant factor would pass every equilibrium test and fail the bracket.

### 5.3 One ePC step is backprop

Fixture: x(4) → h1(3, tanh) → h2(3, tanh) → y(2), biases drawn, Gaussian or softmax-plus-cross-entropy output.

| Test | Claim | Result |
|---|---|---|
| ε-gradient at ε = 0 | equals the hand-written backprop activation gradient per hidden node, atol 1e-6 | pass, both outputs |
| One step at η = 0.05 | ε_h = −0.05·g_h on both hidden nodes; z_h1 = a_h1 − 0.05·g_h1 | pass, both outputs |
| One-step local weight gradients (batch 3 through `pc_weight_gradients`; batch 1 before the 2026-09-08 revision) | hidden edges: g_pc/η vs backprop; output edge: g_pc vs backprop; both divided by the prediction count N = 3 | see below |

Relative deviation of the one-step local weight gradient from η × backprop (hidden) and from backprop (output), tanh MLP x16 → 3 × h32 → y10 with softmax + cross-entropy, batch 8 (`--section backprop_regime`):

| η | h1 (input is the clamp) | h2 | h3 | y |
|---|---|---|---|---|
| 1e-4 | 1.0e-3 | 7.3e-4 | 4.8e-4 | 1.2e-4 |
| 1e-3 | 9.5e-5 | 2.8e-4 | 6.9e-4 | 1.2e-3 |
| 1e-2 | 9.9e-6 | 2.8e-3 | 6.9e-3 | 1.2e-2 |
| 1e-1 | 1.1e-6 | 2.8e-2 | 6.9e-2 | 1.2e-1 |

Two mechanisms are visible. For h2, h3, and y the deviation is exactly linear in η (10× per decade): those layers' inputs are latents re-derived at the perturbed upstream state, an O(η) correction that Theorem C.9's proof drops by evaluating the weight Jacobian at the unperturbed point. For h1 the deviation is float32 noise that falls as η grows: h1's input is the clamped x, nothing upstream is perturbed, and its one-step gradient is η × backprop exactly. At η = 1e-4 every entry is noise from dividing a small gradient by a small η. `TestBackpropCorrespondence::test_one_step_weight_grads_are_eta_backprop_first_order` pins both behaviours. Since the 2026-09-08 revision it measures the fixture's λ_max, sets the η grid at η·λ_max ∈ {1e-3, 1e-2, 1e-1}, and asserts d(η) ≤ 10·η·λ_max for h2 and y with the ratio between consecutive η between 3 and 30, at two weight scales (λ_max near 1.2 and near 30); the constant 10 covers the worst measured ratio of 6.7 across weight std 0.3 to 2.5, where λ_max ranges from 1.1 to 229. For h1 it asserts the rounding signature: a deviation below 1e-3 at the largest η that shrinks as η grows.

One Adam update from the ePC gradients has cosine similarity 1.0000 with one Adam update from backprop's on every layer at η = 1e-3 (0.9922 on h3 at η = 1e-2): Adam normalizes the η scaling away, so 1-step ePC trained with Adam is backprop trained with Adam. Two caveats. The normalization holds while η·|g| ≫ Adam's ε (1e-8); at η = 1e-4 hidden-layer gradients of order 1e-7 are damped by ε. Without Adam the hidden layers learn η times slower than the output layer, a 1000× disparity at the default.

### 5.4 The relaxed-fraction formula against the solver

Linear chain x16 → 3 × h16 → y4, batch 8, η stated relative to λ_max(H_ε). Predicted remaining distance ‖ε_T − ε*‖/‖ε*‖ from the eigen-decomposition of H_ε (each mode shrunk by (1 − ηλ)^T) against the solver after T steps:

| η·λ_max | T | predicted | measured |
|---|---|---|---|
| 0.1 | 1 | 0.9516 | 0.9516 |
| 0.1 | 5 | 0.7945 | 0.7945 |
| 0.5 | 3 | 0.5259 | 0.5259 |
| 1.0 | 5 | 0.1552 | 0.1552 |
| 1.5 | 4 | 0.0990 | 0.0990 |

The per-mode picture of Section 2.3 is the solver's exact behaviour on a linear graph.

### 5.5 Why sPC struggles with depth and ePC does not (reviewer bullets 1, 7)

Chains x16 → depth × h16 → y4, weight std 1/√fan_in ("plain") or muPC. Steps = smallest T with max\|1 − ηλ\|^T ≤ 1e-3 at η = 1/λ_max of the relevant Hessian; for ePC only the excited modes count (four here, the output dimension).

| depth | init | λ_max(H_z) | λ_min(H_z) | κ_z | steps sPC | λ_max(H_ε) | λ_min excited | κ_ε | steps ePC |
|---|---|---|---|---|---|---|---|---|---|
| 2 | plain | 5.45 | 0.188 | 29 | 197 | 4.03 | 2.09 | 1.9 | 10 |
| 3 | plain | 6.00 | 0.116 | 52 | 356 | 8.51 | 2.64 | 3.2 | 19 |
| 4 | plain | 5.97 | 0.075 | 80 | 547 | 12.0 | 2.05 | 5.9 | 37 |
| 6 | plain | 7.03 | 0.017 | 403 | 2 784 | 13.9 | 1.95 | 7.1 | 46 |
| 8 | plain | 5.96 | 0.017 | 348 | 2 404 | 14.8 | 3.53 | 4.2 | 26 |
| 12 | plain | 7.54 | 0.0073 | 1 040 | 7 169 | 46.4 | 2.30 | 20 | 136 |
| 16 | plain | 7.56 | 0.0039 | 1 920 | 13 285 | 26.6 | 2.18 | 12 | 81 |
| 20 | plain | 6.45 | 0.0015 | 4 390 | 30 343 | 24.7 | 2.18 | 11 | 75 |
| 20 | muPC | 6.45 | 0.0015 | 4 230 | 29 249 | 38.8 | 2.88 | 14 | 90 |

λ_min(H_z) falls by three decades from depth 2 to 20 while λ_max(H_z) stays near 6, so sPC's condition number and step count grow with depth: the deep latents sit in flat directions of the latent energy. In error coordinates only the d_y excited modes matter, their condition number stays near 10, and ePC's step count stays below 150. This is the quantitative form of "sPC reaches the equilibrium after a huge number of steps" and the reason oracle-based solver tests are kept at five layers or fewer (a depth-12 sPC test at the required 7 000 steps is still cheap; a depth-20 one at 30 000 steps is not). Measured contractions match the predicted step counts (depth 4: 1.8e-4 for sPC at 547 steps, 1.5e-4 for ePC at 37; depth 12: 3.3e-4 at 7 169 and 1.3e-4 at 136; a quarter of the steps leaves 3–7 percent).

### 5.6 What sets the equilibrium energy spacing across layers (bullets 2, 3)

Oracle per-layer equilibrium energies, chains of width 32 (input 32, output 10, one-hot targets), log10 of the batch mean. Spread is the range over hidden layers in decades; slope is the least-squares slope of log10 E_l against layer index.

| depth | init | log10 E, first hidden | middle | last hidden | output | spread (decades) | slope per layer |
|---|---|---|---|---|---|---|---|
| 3 | std 0.5 | −2.46 | −1.84 | −1.20 | −0.47 | 1.25 | +0.63 |
| 3 | std 1.0 | −0.68 | −0.64 | −0.57 | −0.44 | 0.11 | +0.06 |
| 3 | std 1.5 | +0.41 | +0.14 | −0.13 | −0.30 | 0.53 | −0.27 |
| 3 | muPC | −0.61 | −0.57 | −0.50 | −0.53 | 0.11 | +0.06 |
| 10 | std 0.5 | −6.41 | −3.53 | −1.20 | −0.53 | 5.21 | +0.57 |
| 10 | std 1.0 | −1.38 | −1.63 | −1.52 | −1.44 | 0.25 | −0.02 |
| 10 | std 1.5 | +0.32 | −1.08 | −2.08 | −2.48 | 2.40 | −0.25 |
| 10 | muPC | −1.37 | −1.64 | −1.55 | −1.64 | 0.27 | −0.02 |
| 20 | std 0.5 | −12.82 | −6.48 | −1.16 | −0.58 | 11.65 | +0.62 |
| 20 | std 1.0 | −2.03 | −2.21 | −1.72 | −1.65 | 0.50 | +0.01 |
| 20 | std 1.5 | +0.46 | −2.13 | −3.81 | −4.11 | 4.27 | −0.21 |
| 20 | muPC | −2.03 | −2.22 | −1.78 | −1.88 | 0.46 | +0.01 |

The spacing follows from the pull-back ε_l* = ε_y*·P_lᵀ: each layer's equilibrium energy is the output error pushed back through the downstream weights, so log10 E_l changes by about 2·log10 of the per-layer gain per layer. Contracting weights (std 0.5) give deep layers almost no energy (11.7 decades of spread at depth 20); expanding weights (std 1.5) pile energy into the deep layers; unit-gain weights and muPC keep the profile flat to within half a decade. The spacing is therefore a statement about the weights' gain profile, not about the solver.

The solvers approach that profile very differently. sPC on the depth-10, std-1.0 chain at η = 0.1 (log10 batch-mean energy after k updates; the last row is the oracle):

| after k updates | h1 | h3 | h5 | h7 | h10 | y |
|---|---|---|---|---|---|---|
| 10 | −19.87 | −11.44 | −6.75 | −3.21 | −0.04 | +0.15 |
| 50 | −5.35 | −4.01 | −2.53 | −1.33 | −0.41 | −0.34 |
| 200 | −2.35 | −2.14 | −1.70 | −1.21 | −0.87 | −0.79 |
| 1000 | −1.48 | −1.53 | −1.61 | −1.49 | −1.42 | −1.32 |
| 4999 | −1.38 | −1.46 | −1.62 | −1.57 | −1.52 | −1.44 |
| oracle | −1.38 | −1.46 | −1.62 | −1.57 | −1.52 | −1.44 |

After 10 updates the first hidden layer's energy is 20 decades below the output's: the signal has not arrived. It takes thousands of updates for the profile to flatten onto the oracle. ePC at η = 1/λ_max(H_ε) = 0.015 has every layer within 0.3 decades of its final value after one update (h1 −1.65, h10 −2.31, oracle −1.38 and −1.52) and within 0.1 after 19. A global energy curve hides this: the output-adjacent layers dominate the total (the imbalance Pinchetti et al. 2024 report), so sPC can look converged while deep layers have received nothing.

### 5.7 The stability bound and how it moves with the weights (bullets 4, 6)

Oracle λ_max(H_ε) and η_max = 2/λ_max on chains of width 32, output 10:

| depth | std 0.5 | std 1.0 | std 1.5 | std 2.0 | muPC |
|---|---|---|---|---|---|
| 3 | 1.62 (η_max 1.23) | 7.76 (0.26) | 49.7 (0.040) | 235 (0.0085) | 11.3 (0.18) |
| 5 | 1.62 (1.23) | 14.2 (0.14) | 360 (0.0056) | 5 190 (0.00039) | 21.1 (0.095) |
| 10 | 1.66 (1.20) | 26.2 (0.076) | 17 600 (0.00011) | 4.2e6 (4.8e-7) | 39.4 (0.051) |

Because λ_max = 1 + σ_max(J)² and J multiplies the downstream weights together, λ_max is exponential in depth for expanding weights: at std 1.5 it rises from 50 to 17 600 between depth 3 and 10, and the largest stable η falls by three decades. muPC's forward scales hold λ_max near 10–40 at init, growing gently with depth, which is the parameterization's purpose (Innocenti et al. 2025). The same paper notes that the inference landscape of standard PC networks becomes increasingly ill-conditioned with training time; Section 5.8 measures that growth directly on the ResNet-18.

Lanczos through `EPCInference.error_energy` on the depth-5, std-1.0 chain (30 steps, 0.5 s including compile) gives λ_max = 14.1826 against the oracle's 14.1826 (relative error 1.9e-7), the excited λ_min = 1.72621 against 1.72621 (4.3e-8), and f̄ at η = 0.5/λ_max, T = 5 of 0.874904 against 0.874904 (2.8e-9), so the same measurement is available on graphs the oracle cannot assemble. The chain has 160 ε coordinates of which 10 (the output dimension) are excited; the unit-precision floor at λ = 1 is never excited from ε = 0. In float32 the ten excited modes accumulate enough rounding over ten steps that the relative breakdown guard does not fire, the recurrence continues on that noise, and the floor appears as a Ritz value carrying weight 4e-14 against 6e-4 on the smallest excited mode; the estimator takes its extremes over the modes carrying weight above eps(float32) = 1.2e-7, which is what keeps λ_min at 1.726 rather than 1.000 (revision (a)'s power iteration returned λ_max only, relative error 5.7e-8). On a gelu MLP (x32 → 4 × h64 → y10, softmax + cross-entropy, batch 16) Lanczos gives λ_max = 1.557 and λ_min = 0.48 at init with no gradient weight on negative curvature; ePC for 200 steps at 0.9·η_max settles at its minimum energy (2.318 → 1.767), and at 1.1·η_max reaches 1.899 and then rises to 1.906. On a nonlinear energy the linear bound is local, and the outcome is reported as observed.

### 5.8 The muPC ResNet-18: λ_max at init, the sweep, and the collapses

**The spectrum at init.** Lanczos on one 64-sample CIFAR-10 test batch, 30 steps (17 s including compile), on the demo's graph at its first trial seed: λ_max = 16.45 (Ritz residual 4.6e-4), so η_max = 0.122, the same value revision (a)'s power iteration gave. λ_min = −0.42 (residual 0.11, so the bottom of the spectrum has not converged in 30 steps, but its sign has): the error Hessian of the gelu + cross-entropy graph is indefinite already at init, with 1.2% of the gradient weight on negative curvature; at the defaults those modes grow by 1.002 over five steps, no effect. The gradient weight sits low: f̄ = 0.010 at the defaults against f_max = 0.080, so the weight-averaged eigenvalue is about 2, near the precision floor, while λ_max = 16.45. The library defaults read `eta*T*lambda_max = 0.0822 (gradient-weighted relaxed fraction 0.01, fastest mode 0.08): backprop-like`.

**The 2-epoch sweep.** `examples/epc_spc_resnet18_compare.py --mode sweep` recorded mean test accuracy over five trials for η ∈ {1e-4, 1e-3, 1e-2, 3e-2, 1e-1} and T ∈ {1, …, 10, 16, 32, 64, 128, 160}, with an sPC baseline of 120 state-based steps at 34.64% (full tables in Appendix A; charts `epc_step_sweep__epceta_*.html`). Selected cells from `--resnet18`, each as measured accuracy | f̄ from the init spectrum | f_max = f(16.45) | regime letter on f̄ (B backprop-like, P partially relaxed, E near equilibrium; r: the output-layer gradient reverses on the top mode):

| η | T=1 | T=2 | T=5 | T=10 | T=16 | T=32 | T=64 | T=160 |
|---|---|---|---|---|---|---|---|---|
| 1e-1 | 10.2 \| 0.20 \| 1.65 Pr | 10.6 \| 0.30 \| 0.58 P | 14.3 \| 0.53 \| 1.11 Pr | 28.6 \| 0.73 \| 0.99 P | 30.7 \| 0.85 \| 1.00 P | 31.1 \| 0.96 \| 1.00 E | 31.1 \| 0.99 \| 1.00 E | 31.0 \| 1.00 \| 1.00 E |
| 3e-2 | 36.9 \| 0.06 \| 0.49 B | 33.9 \| 0.11 \| 0.74 P | 31.4 \| 0.23 \| 0.97 P | 30.7 \| 0.38 \| 1.00 P | 30.7 \| 0.50 \| 1.00 P | 31.0 \| 0.70 \| 1.00 P | 31.2 \| 0.88 \| 1.00 P | 31.2 \| 0.98 \| 1.00 E |
| 1e-2 | 38.5 \| 0.02 \| 0.16 B | 38.0 \| 0.04 \| 0.30 B | 35.0 \| 0.09 \| 0.59 B | 32.7 \| 0.17 \| 0.83 P | 31.8 \| 0.24 \| 0.94 P | 30.9 \| 0.39 \| 1.00 P | 30.9 \| 0.58 \| 1.00 P | 31.1 \| 0.83 \| 1.00 P |
| 1e-3 | 38.8 \| 0.00 \| 0.02 B | 38.8 \| 0.00 \| 0.03 B | 38.7 \| 0.01 \| 0.08 B | 38.5 \| 0.02 \| 0.15 B | 38.2 \| 0.03 \| 0.23 B | 36.7 \| 0.06 \| 0.41 B | 34.2 \| 0.11 \| 0.65 P | 31.8 \| 0.24 \| 0.93 P |
| 1e-4 | 38.8 \| 0.00 \| 0.00 B | 38.8 \| 0.00 \| 0.00 B | 38.8 \| 0.00 \| 0.01 B | 38.8 \| 0.00 \| 0.02 B | 38.8 \| 0.00 \| 0.03 B | 38.8 \| 0.01 \| 0.05 B | 38.6 \| 0.01 \| 0.10 B | 38.2 \| 0.03 \| 0.23 B |

Accuracy falls monotonically with f_max: 38.8% where f_max is near 0 (ePC's own small-η·T limit; no backprop arm was run at 2 epochs, and the 100-epoch demo holds the only measured backprop number, 77.11%), 31% where f_max is near 1 (the PC equilibrium), and in between where f_max is in between. A least-squares fit of a single eigenvalue to the 45 cells with η ≤ 0.01, mapping accuracy linearly onto f between those two limits, gives λ_eff = 12.0 with rms residual 0.049; the independently measured λ_max at init is 16.45, a factor of 1.4 above. The fit is a heuristic, and its landing near λ_max is consistent with the accuracy following the top of the spectrum. The η = 0.1, T = 1 arm sits at η·λ_max = 1.64 at init: the output residual after its single step is (1 − η(λ_max − 1))·r = −0.55·r along the top mode, so the output layer's weight gradient had the wrong sign there from the first update (the r in the table), and the T = 2 and T = 3 arms at the same η were one weight growth of 22% away from the bound η·λ_max = 2.

**The 100-epoch runs.** Six runs of the demo with `--augment --activation gelu`, evaluated every 10 epochs (`sweep_eta0.001_steps{1,2,5}.log`, `sweep_eta0.01_steps{1,2,5}.log`):

| η | T | η·T | epoch 10 | epoch 20 | final (100) |
|---|---|---|---|---|---|
| 1e-3 | 1 | 0.001 | 55.83% | 63.59% | 76.73% |
| 1e-3 | 2 | 0.002 | 55.72% | 63.17% | 75.76% |
| 1e-3 | 5 (defaults) | 0.005 | 54.76% | 9.68% | 9.75% |
| 1e-2 | 1 | 0.01 | 9.99% | 9.98% | 9.92% |
| 1e-2 | 2 | 0.02 | 9.65% | 9.79% | 9.88% |
| 1e-2 | 5 | 0.05 | 10.15% | 10.18% | 10.13% |

Only η·T ≤ 0.002 survived, while at 2 epochs even η·T = 0.16 trains. The collapse depends on the training horizon, and by the mechanism of Section 5.7 the natural suspect is λ_max growing with the weights until η·λ_max exceeds 2. The backprop trainer on the same graph reached 77.11%.

**Tracking λ_max during training.** `scripts/epc_analysis.py --track_lambda_max 50 --num_epochs 30 --schedule_epochs 100 --augment` trains the demo graph with the demo's optimizer for the first 30 epochs of the same 100-epoch warmup-cosine schedule, at the demo's trial seed 42 and batch order, and runs power iteration on a fixed 64-sample test batch every 50 weight updates. Log: `epc_lambda_track.log`; per-probe data: `docs/reports/data/epc_lambda_track__eta0.001_T5.csv` and `docs/reports/data/epc_lambda_track__eta0.01_T1.csv`; charts with the same stems (`.html`, `.png`).

Defaults, η = 1e-3, T = 5 (2/η = 2 000):

| epoch | test accuracy | λ_max over the epoch | η·λ_max, max | train energy |
|---|---|---|---|---|
| 1 | 20.19% | 15.5 – 22.0 | 0.022 | 2.10 |
| 5 | 44.10% | 16.8 – 22.4 | 0.022 | 1.52 |
| 8 | 50.32% | 23.7 – 27.5 | 0.028 | 1.22 |
| 9 | 53.53% | 26.9 – 39.4 | 0.039 | 1.27 |
| 10 | 54.76% | 37.9 – 51.0 | 0.051 | 1.06 |
| 11 | 55.85% | 46.9 – 84.5 | 0.085 | 1.04 |
| 12 | 56.36% | 80.5 – 131.5 | 0.13 | 0.89 |
| 13 | 53.44% | 139.5 – 470.9 | 0.47 | 0.58 |
| 14 | 41.12% | 583.9 – 3 508 | 3.5 | 1.74 |
| 15 | 9.81% | 1 – 12 539 | 12.5 | 14.3 |
| 16–30 | 9.5 – 9.7% | 1 | 0.001 | 13.7 – 15.7 |

The epoch-10 accuracy equals the 100-epoch log's 54.76%, so this is the run that collapsed, reproduced. Probes across the crossing: update 2 500 (epoch 13) λ_max = 471; 2 550: 584; 2 600: 1 608; 2 650: 1 869; 2 700: 3 508, the first probe with η·λ_max > 2; 2 750 (epoch 15): 9 833; 2 800: 12 539; 2 850 onward: 1. A final λ_max of exactly 1 means the output prediction no longer depends on any error (the Jacobian J is zero), a saturated constant classifier, and the train energy of 14–16 per sample is the cross-entropy of confident wrong predictions. Accuracy started falling at epoch 13, when η·λ_max of 0.2–0.5 with T = 5 had relaxed the fastest mode by 65–95% and moved the run out of the backprop regime toward the PC value, and collapsed once the bound was crossed. From epoch 9 to 14 λ_max grew from 39 to 3 508, about 2.5× per epoch.

η = 1e-2, T = 1 (2/η = 200):

| epoch | test accuracy | λ_max over the epoch | η·λ_max, max |
|---|---|---|---|
| 1 | 20.21% | 15.5 – 22.0 | 0.22 |
| 3 | 37.82% | 18.9 – 20.2 | 0.20 |
| 5 | 43.16% | 29.7 – 39.6 | 0.40 |
| 6 | 40.56% | 60.4 – 219.7 | 2.2 |
| 7 | 9.79% | (diverging) → 1 | — |
| 8–30 | 9.9 – 10.0% | 1 | 0.01 |

Probes: update 1 000 (epoch 6) λ_max = 60; 1 050: 64; 1 100: 101; 1 150: 220, the first probe with η·λ_max > 2; 1 200 (epoch 7): −9 399; then 1. The −9 399 is not a value of λ_max: power iteration converges to the eigenvalue of largest magnitude, and on this indefinite Hessian a negative eigenvalue had overtaken the positive top by update 1200 (Section 5.8's init measurement already shows λ_min < 0). The Lanczos re-run of this cell (Section 5.9) reports both extremes. At T = 1 the mechanism is the output-gradient reversal of Section 2.3: at update 1100, η(λ_max − 1) = 1.00, the output residual after the single step had changed sign along the top mode, test accuracy fell that epoch from 43.2% to 40.6%, and the network was at chance one epoch later.

Both collapses were preceded by η·λ_max crossing 2. The 100-epoch log's evaluation every 10 epochs could not resolve the order of events; the tracking does. Both tracked cells collapsed, so the ordering does not separate "λ_max grows regardless of the solver and a fixed η eventually crosses the bound" from "ePC's relaxation drives the growth"; Section 5.9's control runs do.

**Where the gradient weight sits, and what the accuracy follows.** The sweep table above uses f̄ from the measured init spectrum. Its letters no longer track the accuracy: at η = 0.01, T = 3 to 5 (36.9% to 35.0%, on the way to the plateau) f̄ is 0.06 to 0.09 and reads backprop-like, and at η = 0.03, T = 5 to 64 (31.4% to 31.2%, the plateau) f̄ is 0.23 to 0.88 and reads partially relaxed. f_max = f(16.45) tracks the accuracy in every row, and the single-eigenvalue fit lands near λ_max (λ_eff = 12.0) for the same reason. Reading: the bulk of the gradient weight sits on modes with λ near 1 or 2, whose relaxation moves each hidden error from −η·T·g toward −g/λ, a change of magnitude by about 1/(η·T) with the direction preserved, which Adam's per-parameter normalization absorbs; the top modes couple the hidden errors to the output prediction, and their relaxation re-weights the output error by S⁻¹ (Section 5.10), a matrix rescaling Adam cannot undo. The 2-epoch accuracy therefore moves with the top modes. The `Regime` band is defined on f̄ because f̄ is the equilibrium criterion (the energy is at its minimum only when the modes carrying the gradient have relaxed); f_max is reported beside it (`Regime.f_max`, the second number in `str(regime)`, the third field per cell in `--resnet18`) because on this graph it is the quantity the accuracy follows. The measurement is one batch at one seed.

### 5.9 Control runs

Both tracked cells of Section 5.8 collapsed, so "η·λ_max crossed 2 before the collapse" is an ordering. The design plan fixed four runs and a reading rule in advance, each the first 30 epochs of the 100-epoch schedule at seed 42 with the normalized trainer of release 0.5.1, probed every 50 updates on the same 64-sample test batch by `RegimeProbe` (the spectrum's two extremes, the gradient weight on negative curvature, f̄, the reversal and crossing flags, the Frobenius norm of every weight, the test accuracy per epoch), run on 2026-09-08 on the RTX 3090 (23, 14, 7, and 14 minutes):

```
python examples/resnet18_cifar10_demo.py --num_epochs 30 --schedule_epochs 100 --augment --activation gelu \
    --track_regime 50 --eval_every 1 [--eta_infer 1e-3 --infer_steps 5 | --eta_infer 1e-3 --infer_steps 1 | --trainer backprop | --eta_infer 1e-2 --infer_steps 1]
```

Reading rule, fixed in advance. If λ_max and the weight norms grow at a comparable rate in the backprop and (1e-3, 1) runs as in the collapsing cells, the growth is a weight-scale effect of this parameterization (no normalization layers, weight decay 1e-2) and the remedy is a rate that follows λ_max or weight-norm control; if they grow only in the collapsing cells, ePC's relaxation feeds the growth.

**The defaults, η = 1e-3, T = 5** (`docs/reports/data/epc_regime_track__pc_eta0.001_T5.csv`). Per epoch: test accuracy, the largest λ_max probed in the epoch and its ratio to the previous epoch's, the most negative λ_min, the largest gradient weight on negative curvature, and the total Frobenius norm √Σ‖W‖² over all 21 weights at the epoch's last probe.

| epoch | accuracy | λ_max | ratio | λ_min | negative weight | ‖W‖ total | flags |
|---|---|---|---|---|---|---|---|
| 1 | 20.24% | 22.0 | | −0.37 | 0.001 | 1670 | |
| 5 | 45.07% | 22.9 | 1.22 | −0.29 | 0.001 | 1663 | |
| 8 | 50.21% | 27.3 | 1.14 | −0.83 | 0.004 | 1655 | |
| 9 | 53.20% | 37.6 | 1.38 | −0.82 | 0.007 | 1652 | |
| 10 | 55.15% | 47.7 | 1.27 | −0.91 | 0.010 | 1650 | |
| 11 | 55.35% | 78.2 | 1.64 | −0.98 | 0.006 | 1647 | |
| 12 | 55.82% | 102.8 | 1.32 | −1.70 | 0.012 | 1645 | |
| 13 | 54.23% | 286.8 | 2.79 | −2.46 | 0.036 | 1643 | |
| 14 | 48.06% | 1 632 | 5.69 | −21.9 | 0.059 | 1641 | reversal (update 2700); `growth_min` 1.115, so `str(regime)` reads indefinite |
| 15 | 9.97% | 31 060 | 19.0 | −1 960 | 0.100 | 1656 | η·λ_max > 2 (update 2750); chance |
| 16 to 30 | 9.8 to 9.9% | 1.000 to 1.008 | | 0.99 to 1.00 | 0 | 1654 → 1619 | dead network: J = 0, every excited eigenvalue at the precision floor |

Growth phases from `probe.summary()`: λ_max between 16 and 27 through epoch 8 (ratios 0.74 to 1.22), 1.3 to 1.6× per epoch over epochs 9 to 12, then 2.8×, 5.7×, and 19× over epochs 13 to 15. This replaces the "about 2.5× per epoch" of Section 5.8, which averaged a slow phase and a runaway. The accuracy peak (55.82%, epoch 12) and the chance epoch (15) match the batch-summed run's (56.36%, 15), so the collapse is not an artifact of the gradient normalization. The reversal flag fired one probe before the stability crossing; at that probe (1 − ηλ_max)^5 = (−0.632)^5 = −0.10 against −1/(λ_max − 1) = −0.0006.

**The weight norms.** The total Frobenius norm falls monotonically from 1670 to 1641 through epoch 14, while λ_max grows 74×; it rises only in the collapse epoch (1641 → 1656) and falls again after it. Per weight, every convolution kernel's norm fell by 1.5 to 1.7% between epochs 1 and 14 (for example `s4b1_conv_a->s4b1_conv_b:in` 768 → 754) and only the output layer's grew (4.51 → 6.07, +35%). Weight decay at 1e-2 keeps the weight scale shrinking through the entire growth of λ_max. λ_max = 1 + σ_max(J)² is a spectral quantity, and J is a product of the weights' actions on the gelu derivatives along the network, so it grows through alignment of the weights and the activation pattern along one direction, not through their norms. Frobenius-norm control would not have bounded it.

**The controls.** The survivor (1e-3, 1) and the backprop trainer have the same weight-norm trajectory as the defaults (1670 → 1597 and 1670 → 1601 over 30 epochs, falling every epoch) and no runaway:

| run | accuracy at epochs 8, 12, 15, 30 | λ_max at epochs 1, 8, 12, 15, 30 | λ_min range | negative weight | flags |
|---|---|---|---|---|---|
| (1e-3, 1) | 49.0, 56.4, 59.2, 67.7% | 22.0, 17.0, 19.6, 20.9, 29.2 (max 30.2 at epoch 29) | −0.03 to −0.95 | ≤ 0.005 | none |
| backprop | 50.3, 57.9, 60.9, 69.0% | 21.9, 16.3, 17.8, 19.1, 24.1 (max 24.9 at epoch 29) | −0.04 to −1.15 | ≤ 0.008 | none |
| (1e-3, 5) | 50.2, 55.8, 10.0, 9.9% | 22.0, 27.3, 102.8, 31 060, 1.0 | −0.05 to −1 960 | ≤ 0.10 | reversal 2700, crossing 2750 |

Over epochs 9 to 12, where the defaults' λ_max went 38 → 103, the survivor's went 18 → 20 and backprop's 17 → 18. The generic drift is about 1.01× per epoch in both controls (22 → 24 and 22 → 29 over 30 epochs, with epoch-to-epoch ratios between 0.86 and 1.16). The Hessian is indefinite at init in every run (λ_min = −0.37) and stays mildly indefinite in the controls, with under 1% of the gradient weight on negative curvature.

**η = 1e-2, T = 1** (`docs/reports/data/epc_regime_track__pc_eta0.01_T1.csv`). Accuracy 44.46% at epoch 5, 43.79% at epoch 6, 10.12% at epoch 7. λ_max 22 → 27.8 (epoch 4) → 38.4 (5) → 166.9 (6) → 6 132 (7) → 1.0 from epoch 8. Probes across the collapse:

| update | epoch | λ_max | λ_min | η·λ_max | η(λ_max − 1) | flags |
|---|---|---|---|---|---|---|
| 1 100 | 6 | 85.7 | −0.73 | 0.86 | 0.85 | |
| 1 150 | 6 | 166.9 | −1.45 | 1.67 | 1.66 | reversal |
| 1 200 | 7 | 6 132 | −354 | 61.3 | | reversal, unstable |
| 1 250 | 7 | 1.0 | 1.0 | 0.01 | | dead |

The plan predicted the reversal flag at update 1100; it fired at 1150, one probe later, because η(λ_max − 1) was 0.85 at 1100 on this trainer. Update 1200 reports λ_max = 6 132 with λ_min = −354: the Hessian is strongly indefinite as the network dies, and revision (a)'s power iteration, which converges to the eigenvalue of largest magnitude, returned a negative number here (−9 399 on the batch-summed trainer) where Lanczos reports both extremes. Accuracy fell in the epoch of the reversal and was at chance one epoch later, as in the batch-summed run.

**Reading.** The second case of the rule: λ_max and the weight norms do not grow in the runs that do not collapse (the norms fall in every run, the backprop λ_max drift is 1.01× per epoch), and the runaway occurs only in the two ePC cells with the larger η·T. ePC's relaxation feeds the growth. The remedy the rule assigns is therefore not weight-norm control but keeping the solver in the regime where its weight gradients are backprop's, or a rate that follows λ_max (Section 6.4). Why a run whose regime at init reads backprop-like on both f̄ (0.010) and f_max (0.080) still drives λ_max up is not settled by these runs: over epochs 9 to 12 the defaults' f_max rose from 0.17 to 0.42 (λ_max 38 → 103) while accuracy was still climbing, so the departure from backprop's gradient and the growth of λ_max reinforce each other once λ_max has drifted enough for η·T·λ_max to leave the backprop regime, and the survivor at T = 1 sits five times lower on η·T at the same η.

**What the two relaxed fractions signal during training.** In the defaults run f̄ (last probe of each epoch) went 0.027, 0.052 (epoch 8), 0.065, 0.076, 0.110 (epoch 11), 0.151, 0.271, 0.815 (epoch 14, the reversal probe): it crossed the 0.1 band edge in epoch 11, three epochs before the reversal and one before the accuracy peak. f_max crossed 0.1 in epoch 5 (0.10) and stood at 0.33 in epoch 11, while accuracy climbed for seven more epochs. In the (1e-2, 1) run f̄ went 0.055, 0.066, 0.077, 0.103 (epoch 4), 0.133, 0.585 (epoch 6, the reversal), two epochs of warning, while f_max was already 0.18 in epoch 1 and that run trained to 44% before collapsing. In the survivor f̄ stayed between 0.005 and 0.009 and f_max at 0.02 for 30 epochs. So the two measures answer different questions: f_max at init predicts the 2-epoch accuracy penalty of a fixed (η, T) (Section 5.8), and f̄ rising through 0.1 during training is the early sign that the gradient-carrying bulk has started to relax and λ_max is about to run away. That division is the reason the probe's band stays on f̄ with f_max beside it.

### 5.10 The equilibrium damps the learning signal by S⁻¹

At the PC equilibrium the output error is ε_y* = r S⁻¹ (Section 2.2), so along an eigenmode of S with eigenvalue λ_S the learning signal is 1/λ_S of the backprop residual. This is a matrix rescaling, and a per-parameter optimizer cannot undo it. `--section equilibrium_profile` tabulates ‖r S⁻¹‖/‖r‖ (batch mean of the per-sample ratio) with eig(S) on the linear chains of Section 5.6:

| depth | std 0.5 | std 1.0 | std 1.5 | muPC |
|---|---|---|---|---|
| 3 | 0.802 (eig S 1.06 – 1.55) | 0.308 (1.44 – 6.91) | 0.079 (3.0 – 42.7) | 0.231 (1.67 – 10.0) |
| 5 | 0.776 (1.11 – 1.62) | 0.177 (2.18 – 15.2) | 0.014 (9.14 – 383) | 0.123 (2.79 – 22.7) |
| 10 | 0.764 (1.12 – 1.94) | 0.089 (2.94 – 65.3) | 0.000 (143 – 6.6e4) | 0.058 (3.96 – 99) |
| 20 | 0.721 (1.10 – 1.93) | 0.084 (2.09 – 134) | 0.000 (6.1e3 – 8.5e7) | 0.053 (2.65 – 203) |

At muPC init the equilibrium keeps 5 to 23 percent of the output residual, less with depth. This linear-chain mechanism, applied to the nonlinear cross-entropy ResNet-18, is consistent with the 2-epoch sweep's PC-equilibrium plateau (31%) trailing its backprop-like plateau (38.8%): the cells on the plateau train on an output error damped along the high-leverage modes. It is a consistency argument, not a measurement on the ResNet-18.

### 5.11 The paper's ResNet-18 and this demo

Goemaere et al. (Tables E.9 and E.10) trained ResNet-18 at the same error rate 1e-3 with zero error momentum, T = 5, SGD on the errors and Adam on the weights, for 25 epochs in their sweep and 50 in the final run, and reported no instability. Their network has batch normalization after every convolution, ReLU, weight decay searched in [1e-6, 1e-3], and a standard parameterization. The demo uses the muPC parameterization with `include_output=False`, no normalization layers (none exist in `fabricpc/nodes`), gelu, weight decay 1e-2, and a 100-epoch schedule, and its (1e-3, 5) run collapsed by epoch 20. λ_max = 1 + σ_max(J)² is a weight-scale quantity: normalization layers bound the activations that J is built from, and weight decay bounds the weights, so both are levers on the bound a fixed η has to stay under. Which lever the demo's growth responds to is what Section 5.9's reading rule decides.

## 6. Discussion

### 6.1 Answers to the reviewer's bullets

| Bullet | Question | Finding |
|---|---|---|
| 1 | Why does sPC struggle with deep layers? | λ_min(H_z) falls three decades from depth 2 to 20 while λ_max stays near 6; the deep latents lie in flat directions and need κ ≈ 4 000 steps to relax at depth 20 (Section 5.5). |
| 2, 3 | What sets the equilibrium energy spacing across layers? | The pull-back ε_l* = ε_y*·P_lᵀ: each layer's energy is the output error pushed back through the downstream weights, so the spacing per layer is about 2·log10 of the layer gain; muPC and unit-gain weights give a flat profile (Section 5.6). |
| 4 | ePC stability in deep networks; the largest stable η? | η < 2/λ_max(H_ε) at every T, and η < 1/(λ_max − 1) at odd T for the output-layer gradient; λ_max = 1 + σ_max(J)², exponential in depth for expanding weights, near 10–40 at muPC init; measurable on any graph by Lanczos through `error_energy` (Section 5.7), tracked during training by `RegimeProbe`. |
| 5 | Is 1-step ePC backprop? | Yes: ε = −η·g exactly; weight gradients are backprop's scaled by η on hidden layers (exactly where the input is a clamp, to first order otherwise) and unscaled on the output; Adam removes the scaling while η·\|g\| ≫ its ε (Section 5.3). The condition for T > 1 is η·T·λ ≪ 1 on the excited modes, not η·T ≪ 1. |
| 6 | The training collapses; do they occur under muPC? | Yes, on the muPC ResNet-18: at the defaults λ_max stays between 16 and 27 through epoch 8, grows 1.3 to 1.6× per epoch over epochs 9 to 12, then 2.8×, 5.7×, and 19× over epochs 13 to 15, and both collapses follow the output-gradient reversal and then η·λ_max crossing 2 (Sections 5.8, 5.9). The control runs show the growth is ePC's: under backprop and under ePC at (1e-3, 1) λ_max drifts 1.01× per epoch over the same 30 epochs, and the weight norms fall in every run. muPC controls the scale at init, not its growth. |
| 7 | How many steps does sPC need; why keep oracle checks at ≤ 5 layers? | 197 steps at depth 2, 7 169 at depth 12, 30 343 at depth 20 for a 1e-3 contraction; ePC needs 10–150 (Section 5.5). |

### 6.2 Implications for FabricPC

- The defaults `EPCInference(eta_infer=1e-3, infer_steps=5)` are backprop-like at init on the ResNet-18 (η·T·λ_max = 0.08). The demo's accuracy at these settings is backprop's accuracy, obtained through a slower path: five reverse passes per update to compute what one would.
- A fixed η has no lasting safety margin. λ_max at init predicted η_max = 0.12, and the run at η = 1e-3, more than a hundred times below that, collapsed at epoch 15 because λ_max reached 31 000. The control runs (Section 5.9) rule out the weight scale as the driver: weight decay at 1e-2 shrank every convolution weight's norm through the entire growth, and the same parameterization under backprop or under ePC at T = 1 drifted 1.01× per epoch. The growth is ePC's own, so the remedy is on the solver side: a rate that follows λ_max from `RegimeProbe`, or a step count and rate that keep η·T·λ_max in the backprop regime as λ_max drifts. `RegimeProbe` makes the tracking affordable inside any `train` run (17 s including compile for the first probe on the ResNet-18; 30 Hessian-vector products thereafter) at the cost of a device sync per batch.
- Studying PC dynamics (rather than backprop in disguise) on this graph requires the modes that carry the gradient to relax, f̄ > 0.9, with η·λ_max well below 2 and, at odd T, η(λ_max − 1) below 1. At the init spectrum that needs η·T ≳ 1.5 (for example η = 0.03, T ≥ 50; the top mode alone relaxes by η = 0.03, T = 5). The 2-epoch sweep shows what leaving the backprop regime costs in accuracy at 2 epochs: 31% against 38.8%, and Section 5.8 shows the accuracy reaches that plateau once the top modes have relaxed.
- The warning the reviewer asked for is in place in the form the data supports: `EPCInference.regime(spectrum)` in the solver, the demo's settings block, the compare script's per-arm report, `RegimeProbe` during training, and the inference guide's backprop-regime paragraph. A label from η·T alone was rejected because it reads the slowest mode and misreads the sweep by an order of magnitude; the band is on f̄ with f_max beside it, for the reason given in Section 5.8.

### 6.3 Limits

- The oracle is exact only for linear-Gaussian graphs. On nonlinear graphs the Hessian is evaluated at a point (the feedforward state on one batch) and the bound is local; the gelu bracket at 1.1·η_max rose after its minimum rather than diverging cleanly.
- The spectrum was probed on one fixed 64-sample batch at one seed. A different batch or seed gives a different λ_max at init (a seed-0 run, `epc_lambda_track_seed0_partial.log`, started near 21 rather than 16); the growth pattern, not the initial value, is the finding. The bottom of the spectrum had a Ritz residual of 0.11 after 30 steps, so λ_min = −0.42 is a sign and an order of magnitude, not a converged value.
- The sweep fit maps accuracy linearly onto the relaxed fraction and reads one eigenvalue off it; it is a heuristic. Its landing near λ_max is consistent with the accuracy following the top modes (Section 5.8), not a measurement of the excited band, which the Lanczos weights show is not compact.
- No backprop arm was run at 2 epochs; 38.8% is ePC's own limit. The sweep and the 100-epoch runs predate release 0.5.1's per-prediction gradient normalization; the control runs of Section 5.9 do not.
- The reversal flag is the linear unit-precision chain's formula applied to the local quadratic model at ε = 0. In both collapsing control runs it fired one probe (50 updates) before the stability crossing and in the epoch accuracy started to fall; in the (1e-2, 1) run it fired at update 1150 rather than the predicted 1100, because η(λ_max − 1) was 0.85 at 1100 on the normalized trainer. Its lead over the crossing is one probe interval here, so a controller reading it has 50 updates of warning at this η.
- The control runs are one seed and one probe batch each, 30 epochs of a 100-epoch schedule. They separate the two causes the reading rule named; they do not say what in ePC's relaxation aligns the weights (Section 5.9's last paragraph is an interpretation).

### 6.4 Follow-up

The reading rule selected the solver-side remedy: a stability-aware rate that follows λ_max from `RegimeProbe` (lower η as λ_max grows, or stop with a diagnosis when η(λ_max − 1) approaches 1 at odd T or η·λ_max approaches 2), or a step count chosen so that η·T·λ_max stays in the backprop regime over the run; weight-norm control is not indicated, since the norms fell throughout. The four probe CSVs give the growth curve such a controller has to follow, with the reversal flag one probe ahead of the crossing. Whether the band should read f_max for "backprop-like" (the quantity the 2-epoch accuracy followed at init, Section 5.8) and f̄ for "near PC equilibrium" (the energy criterion) is a design question the measurements raise; the control runs argue for keeping f̄ as the tracked band, since its crossing of 0.1 preceded both collapses by two to three epochs while f_max had crossed it long before with accuracy still improving (Section 5.9). The `Regime` carries both.

## 7. Reproduction

All commands run from the repository root at commit `30aeb83` (`git checkout 30aeb83`), inside the project environment (`.venv/bin/python`).

Tests (CPU, about three and a half minutes for the suite; at the final commit 654 passed and 6 skipped, none of the skips belonging to this work):

```
python -m pytest tests/test_linear_pc_oracle.py tests/test_epsilon_spectrum.py tests/test_inference_epc.py tests/test_regime_probe.py -q
python -m pytest tests/ -q
```

CPU analysis (Sections 5.3 to 5.7, about twenty seconds; `--plot` writes `epc_analysis_sweep_fit.html`, `epc_analysis_equilibrium_profile.html`, `epc_analysis_spectra.html`, and `.png` when kaleido is installed):

```
python scripts/epc_analysis.py
python scripts/epc_analysis.py --section backprop_regime stability --plot
```

The excited spectrum at init on the muPC ResNet-18 (GPU, CIFAR-10 via tensorflow-datasets, about 30 s):

```
python scripts/epc_analysis.py --resnet18
```

The spectrum during training (Section 5.9's control runs, 23, 14, 7, and 14 minutes on the RTX 3090; each writes `epc_regime_track__pc_eta{eta}_T{T}.csv` or `docs/reports/data/epc_regime_track__backprop.csv`, rendered by `--plot_track`):

```
for cell in "--eta_infer 1e-3 --infer_steps 5" "--eta_infer 1e-3 --infer_steps 1" "--trainer backprop" "--eta_infer 1e-2 --infer_steps 1"; do
  python examples/resnet18_cifar10_demo.py --num_epochs 30 --schedule_epochs 100 --augment --activation gelu \
      --track_regime 50 --eval_every 1 $cell 2>&1 | tee "epc_regime_track_${cell// /_}.log"
done
python scripts/epc_analysis.py --plot_track docs/reports/data/epc_regime_track__*.csv
```

The control runs of Section 5.9 wrote `epc_regime_track__pc_eta0.001_T5.{csv,html,png,log}`, `epc_regime_track__pc_eta0.001_T1.*`, `epc_regime_track__backprop.*`, and `epc_regime_track__pc_eta0.01_T1.*` in the project root; the CSVs were moved to `docs/reports/data/`. The revision (a) tracking files `docs/reports/data/epc_lambda_track__eta0.001_T5.csv` and `docs/reports/data/epc_lambda_track__eta0.01_T1.csv` were written by the deleted `--track_lambda_max` section with the power-iteration estimator and the batch-summed trainer; they are kept as data and are not reproducible from the current script.

The 100-epoch runs behind Section 5.8 (about 20 to 45 minutes each on the RTX 3090):

```
for eta in 0.001 0.01; do
for steps in 1 2 5; do
  python examples/resnet18_cifar10_demo.py --num_epochs 100 --eval_every 10 --augment --activation gelu \
    --eta_infer "$eta" --infer_steps "$steps" 2>&1 | tee "sweep_eta${eta}_steps${steps}.log"
done
done
python examples/resnet18_cifar10_demo.py --num_epochs 100 --eval_every 10 --augment --activation gelu --trainer backprop
```

The 2-epoch sweep (about five hours per η on the RTX 3090; tables in Appendix A):

```
python examples/epc_spc_resnet18_compare.py --mode sweep --n_trials 5 --epc_eta 0.001
```

The demo's regime line at init (one epoch, about a minute):

```
python examples/resnet18_cifar10_demo.py --num_epochs 1
```

Data files referenced: `sweep_eta0.001_steps1.log`, `sweep_eta0.001_steps2.log`, `sweep_eta0.001_steps5.log`, `sweep_eta0.01_steps1.log`, `sweep_eta0.01_steps2.log`, `sweep_eta0.01_steps5.log` (100-epoch runs, project root); `epc_lambda_track.log`, `epc_lambda_track__eta0.001_T5.{csv,html,png}`, `epc_lambda_track__eta0.01_T1.{csv,html,png}` (tracking; the CSVs in `docs/reports/data/`, the charts in the project root); `epc_lambda_track_seed0_partial.log` (an aborted seed-0 run, kept for the seed comparison); `epc_step_sweep__epceta_{0.001,0.01,0.03,0.1}.{html,png}` (2-epoch sweep charts); Appendix A (2-epoch sweep tables); `docs/dev_plans_archive/epc_inference_solver.md` (the design record). None of the data files is committed: the six tracking CSVs live in `docs/reports/data/`, the logs and charts in the project root, and every one is regenerated by the commands above.

## 8. References

- Cédric Goemaere, Gaspard Oliviers, Rafal Bogacz, and Thomas Demeester. *ePC: Fast and Deep Predictive Coding in Digital Simulation.* arXiv:2505.20137 (v5, 2026). Appendix C.2 (equivalence of state- and error-based PC), Appendix C.3 and Theorem C.9 (when ePC reduces to backpropagation), Appendix C.4 (PC weight gradients differ from backprop's at equilibrium).
- Francesco Innocenti, El Mehdi Achour, Ryan Singh, and Christopher L. Buckley. *Only Strict Saddles in the Energy Landscape of Predictive Coding Networks?* Advances in Neural Information Processing Systems 37 (NeurIPS 2024), pp. 53649–53683. arXiv:2408.11979. Theorem 1: the equilibrated energy of a deep linear network is a rescaled mean-squared error with S = I + Σ_l P_lᵀP_l.
- Francesco Innocenti, El Mehdi Achour, and Christopher L. Buckley. *μPC: Scaling Predictive Coding to 100+ Layer Networks.* arXiv:2505.13124 (2025). The Depth-μP parameterization FabricPC's `MuPCConfig` implements, and the observation that the inference landscape of standard PC networks becomes increasingly ill-conditioned with model size and training time.
- Luca Pinchetti, Chang Qi, Oleh Lokshyn, Gaspard Oliviers, Cornelius Emde, Mufeng Tang, Amine M'Charrak, Simon Frieder, Bayar Menzat, Rafal Bogacz, Thomas Lukasiewicz, and Tommaso Salvatori. *Benchmarking Predictive Coding Networks — Made Simple.* ICLR 2025; arXiv:2407.01163. The energy imbalance between output-adjacent and deep layers that makes a global energy curve misleading (Section 5.6).

## Appendix A. ResNet-18/CIFAR-10 2-epoch sweep, full tables

The five sweep runs read by Section 5.8, verbatim from the compare script's output. Five trials per arm, two epochs per arm, sPC baseline of 120 state-based steps at η = 0.1; accuracy is mean ± SE over trials; train time is total training wall-clock per arm in seconds. Recorded 2026-09-01 to 2026-09-04 with batch-summed weight gradients, before release 0.5.1.

```
epochs/arm: 2  |  trials: 5  |  sPC: 120 steps @ eta 0.1
sweep epc_eta [0.1, 0.03, 0.01, 0.001, 0.0001]
report accuracy (2 epochs) and train time over trials

python examples/epc_spc_resnet18_compare.py --mode sweep --n_trials 5 --epc_eta 0.1
--- Per-arm results (mean +/- SE over trials) ---
arm          accuracy%          train time (s)    
sPC-120      34.64 +/- 0.65     1008.6
ePC-1        10.22 +/- 0.07     84.8
ePC-2        10.57 +/- 0.15     93.3
ePC-3        10.05 +/- 0.20     99.9
ePC-4        12.12 +/- 0.66     109.3
ePC-5        14.28 +/- 0.85     114.0
ePC-6        15.82 +/- 0.66     121.0
ePC-7        18.57 +/- 0.36     128.1
ePC-8        21.51 +/- 0.60     134.8
ePC-9        25.92 +/- 0.55     142.6
ePC-10       28.63 +/- 0.34     150.1
ePC-16       30.72 +/- 0.59     189.6
ePC-32       31.09 +/- 0.60     301.7
ePC-64       31.05 +/- 0.64     526.5
ePC-128      31.03 +/- 0.65     982.8
ePC-160      31.02 +/- 0.64     1206.8

python examples/epc_spc_resnet18_compare.py --mode sweep --n_trials 5 --epc_eta 0.03
--- Per-arm results (mean +/- SE over trials) ---
arm          accuracy%          train time (s)    
sPC-120      34.64 +/- 0.65     1074.6
ePC-1        36.90 +/- 0.29     85.0
ePC-2        33.87 +/- 0.41     92.7
ePC-3        32.62 +/- 0.48     101.1
ePC-4        31.92 +/- 0.51     110.0
ePC-5        31.45 +/- 0.52     115.9
ePC-6        31.21 +/- 0.55     122.1
ePC-7        31.04 +/- 0.57     127.7
ePC-8        30.92 +/- 0.55     135.2
ePC-9        30.78 +/- 0.54     145.0
ePC-10       30.73 +/- 0.57     151.4
ePC-16       30.73 +/- 0.54     194.5
ePC-32       30.98 +/- 0.53     318.6
ePC-64       31.17 +/- 0.60     565.1
ePC-128      31.17 +/- 0.59     1053.4
ePC-160      31.19 +/- 0.58     1312.9

python examples/epc_spc_resnet18_compare.py --mode sweep --n_trials 5 --epc_eta 0.01
ePC eta: 0.01  |  epochs/arm: 2  |  trials: 5  |  sPC: 120 steps @ eta 0.1
--- Per-arm results (mean +/- SE over trials) ---
arm          accuracy%          train time (s)    
sPC-120      34.64 +/- 0.65     969.9
ePC-1        38.54 +/- 0.33     84.2
ePC-2        37.97 +/- 0.36     92.2
ePC-3        36.91 +/- 0.30     99.2
ePC-4        35.84 +/- 0.28     107.1
ePC-5        34.99 +/- 0.36     113.5
ePC-6        34.35 +/- 0.44     119.2
ePC-7        33.86 +/- 0.42     126.0
ePC-8        33.44 +/- 0.38     132.7
ePC-9        33.02 +/- 0.43     138.9
ePC-10       32.71 +/- 0.49     146.4
ePC-16       31.81 +/- 0.54     185.4
ePC-32       30.93 +/- 0.58     292.0
ePC-64       30.89 +/- 0.57     505.5
ePC-128      31.12 +/- 0.59     933.2
ePC-160      31.15 +/- 0.59     1146.1

python examples/epc_spc_resnet18_compare.py --mode sweep --n_trials 5 --epc_eta 0.001
--- Per-arm results (mean +/- SE over trials) ---
arm          accuracy%          train time (s)    
sPC-120      34.64 +/- 0.65     977.7
ePC-1        38.84 +/- 0.33     84.8
ePC-2        38.83 +/- 0.35     93.3
ePC-3        38.78 +/- 0.34     99.2
ePC-4        38.74 +/- 0.34     107.1
ePC-5        38.70 +/- 0.34     113.7
ePC-6        38.65 +/- 0.34     120.7
ePC-7        38.65 +/- 0.34     126.3
ePC-8        38.60 +/- 0.35     132.9
ePC-9        38.55 +/- 0.35     140.3
ePC-10       38.49 +/- 0.34     147.2
ePC-16       38.18 +/- 0.37     185.3
ePC-32       36.68 +/- 0.29     293.7
ePC-64       34.25 +/- 0.42     510.5
ePC-128      32.29 +/- 0.54     940.8
ePC-160      31.83 +/- 0.52     1154.1

python examples/epc_spc_resnet18_compare.py --mode sweep --n_trials 5 --epc_eta 0.0001
--- Per-arm results (mean +/- SE over trials) ---
arm          accuracy%          train time (s)    
sPC-120      34.64 +/- 0.65     990.2
ePC-1        38.77 +/- 0.34     84.8
ePC-2        38.83 +/- 0.32     93.0
ePC-3        38.84 +/- 0.34     100.5
ePC-4        38.83 +/- 0.34     107.6
ePC-5        38.83 +/- 0.34     114.4
ePC-6        38.84 +/- 0.33     121.0
ePC-7        38.85 +/- 0.33     127.6
ePC-8        38.85 +/- 0.33     133.8
ePC-9        38.82 +/- 0.32     141.9
ePC-10       38.81 +/- 0.33     148.1
ePC-16       38.83 +/- 0.35     188.3
ePC-32       38.78 +/- 0.34     297.8
ePC-64       38.65 +/- 0.35     518.5
ePC-128      38.34 +/- 0.33     960.7
ePC-160      38.16 +/- 0.36     1175.5
```

## Appendix B. Lanczos evidence from the design review (CPU)

The two CPU experiments that set the estimator's second statistic (f̄, not λ_min) and its breakdown guard, recorded during the second review round.

Plain Lanczos on a rank-3 excited spectrum: H = I + JᵀJ, D = 40, J of rank 3, start vector g0 = Jᵀr, 30 steps, no reorthogonalization. Excited eigenvalues eig(S): 15.72, 24.19, 33.62; floor 1.0 on the 37 unexcited modes. β_3 was 4.0e-5 in float32 and 8.6e-14 in float64.

| precision | guard | stopped at | min Ritz | max Ritz |
|---|---|---|---|---|
| float32 | β = 0 | never | 1.000002 | 33.6245 |
| float64 | β = 0 | never | 1.000000 | 33.6245 |
| float32 | β ≤ 1e-5·\|α\| | step 3 | 15.7200 | 33.6245 |
| float64 | β ≤ 1e-5·\|α\| | step 3 | 15.7200 | 33.6245 |

The design's threshold √eps(dtype)·max(max_i |α_i|, max_i β_i) is about 1.2e-2 in float32 and 5e-7 in float64 on this spectrum, above β_3 in both precisions, so it also stops at step 3.

Full ε-Hessian of the gelu MLP fixture (x32 → four hidden layers of 64 gelu units → 10-way softmax with cross-entropy, batch 4, 1024 ε entries, `jax.hessian` of `error_energy`):

| weight std | λ_min | λ_max | negative modes | excited modes | Σ w_k on λ < 0 | 90% of g0 weight below λ |
|---|---|---|---|---|---|---|
| 1.0 | 0.555 | 1.557 | 0 | 943 of 1024 | 0 | 1.39 |
| 2.0 | −14.8 | 11.4 | 16 | 256 of 1024 | 0.226 | 3.08 |

At η = 0.03, T = 10 the gradient-weighted relaxed fraction from 30 Lanczos steps equals the exact value at both weight scales (0.298 at std 1.0; 0.458 at std 2.0, measured as the unnormalized sum Σ_{θ_k > 0} w_k f(θ_k); the normalized f̄ at std 2.0 is 0.458 / 0.774 ≈ 0.59 and has not been re-measured), while f(λ_max) alone reads 0.380 and 0.985.
