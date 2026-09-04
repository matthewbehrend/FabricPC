# ePC reviewer response: linear oracle, backprop-regime documentation, analysis script

## Context

The reviewer replied to the eight observation bullets that accompanied the resnet18 ePC-vs-sPC convergence figure (`examples/epc_spc_resnet18_compare.py --mode convergence`). Their requests split three ways:

- **Test-suite invariants.** A linear oracle that returns the exact equilibrium state and energy for a range of linear networks, with both `EPCInference` and the state-based solvers checked against it, plus a test of the oracle itself (hard-coded numbers on tiny models, or code readable enough to be obviously right via Innocenti et al. 2024, Theorem 1). A warning that 1-step ePC is backprop with gradients scaled by `eta_infer`.
- **Analytical questions.** Why sPC struggles with deep layers (bullet 1); what sets the equilibrium energy spacing across layers (bullets 2, 3); ePC stability in deep networks and how to pick the largest stable `eta_infer` (bullet 4); the training collapses and whether they occur under muPC (bullet 6, confirmed by the user: the resnet18 demo is a muPC model); how many steps sPC needs to reach equilibrium and why oracle checks should stay at 5 layers or fewer (bullet 7).
- **Warning adequacy.** The user asked whether the existing caution against `infer_steps=1` says the right thing and appears where users will see it.

Facts established this session that shape the plan:

1. The ePC paper (Goemaere et al., arXiv 2505.20137, Appendix C.3, Theorem C.9) states two backprop-equivalence conditions: `infer_steps = 1` gives backprop gradients scaled by the error learning rate, and `eta × steps ≪ 1` gives backprop gradients scaled by `eta × steps`. The paper kept its own `eta × steps` above that threshold deliberately.
2. The current default `EPCInference(eta_infer=1e-3, infer_steps=5)` has `eta × steps = 0.005`, inside the backprop regime. The resnet18 sweep tables confirm it: at eta 1e-3 accuracy is flat across `infer_steps ≤ 10` and equals the eta 1e-4 arms, and the 100-epoch demo result at `--infer_steps 1` (76.73%) matches the backprop trainer on the same graph (77.11%).
3. The sweep logs in the project root show every PC-regime resnet18 run collapsing to chance: eta 1e-3 with 5 steps at epoch 20, eta 1e-2 at every step count by epoch 10. The runs that trained were all in the backprop regime.
4. The existing caution ("Use infer_steps > 1") keys on the wrong criterion, and the tuning table in the inference guide recommends `infer_steps` 1 to 5 for ePC two paragraphs below it.
5. A design review of the oracle math verified the quadratic form, the ε-coordinate Hessian, and the exactness of the 1-step gradient identity, and corrected two of my claims: an unclamped source node contributes no curvature floor in ε coordinates, and ePC's stability bound shrinks as downstream weight products grow, so the "ePC is better conditioned" claim must be stated per regime.

User decisions (2026-09-03): keep the defaults and document the regime at every touch point; no `warnings.warn`, the demos print the regime instead; the oracle lives in a package module shared by tests and the analysis script.

## Symbols

| Symbol | Meaning |
|---|---|
| z_t | latent of node t, one row per sample, `NodeState.z_latent` |
| μ_t | node t's prediction of z_t from its in-edge sources, `NodeState.z_mu` |
| ε_t | prediction error z_t − μ_t, `NodeState.error`; the relaxed variable under `EPCInference` |
| W_eff[s→t] | effective edge matrix: muPC forward scale times the Linear weight (or times the IdentityNode scale times I) |
| E | total energy, Σ over in_degree > 0 nodes of ½·precision·‖z_t − μ_t‖², the set the trainer and both solvers sum |
| A, c | the quadratic form of E over the stacked free latents on a linear-Gaussian graph, E = ½‖A z_free − c‖² |
| B, M | B: strictly block-lower-triangular map z_free ← z_free through the edges; M = (I − B)⁻¹, the ε → z map |
| H_z, H_ε | Hessians of E in latent coordinates (AᵀA) and error coordinates (MᵀAᵀAM) |
| S | I + Σ_l P_lᵀP_l on a chain, P_l = W_eff[l+1]⋯W_eff[L] the map from ε_l to the output prediction; Innocenti's Theorem 1 matrix |
| η, T | `eta_infer`, `infer_steps` |

## Deliverables

### A. `fabricpc/utils/linear_pc_oracle.py` (new, pure NumPy float64, ~220 lines)

The definition of the linear-Gaussian energy as a quadratic, solved exactly. Reads only params, edges, `forward_scale`, biases, precision. Never calls node code or solver code.

Conventions, asserted in code: nodes enumerated in `structure.node_order` (topological), so B is strictly block-lower-triangular; samples are columns of c; weights act in row convention (z_mu = z_s @ W), so the block acting on a stacked column is the transpose.

Assembly, for each node t with in_degree > 0 (one residual row block, precision p_t):
- block(t, t) = √p_t·I if t is unclamped; otherwise the clamp moves to c.
- block(t, s) = −√p_t·W_eff[s→t]ᵀ for each unclamped source s; clamped sources move to c.
- c_t = √p_t·(b_t + Σ_{clamped s} W_eff[s→t]ᵀ z_s − [t clamped]·z_t).
- Unclamped sources (in_degree 0) get columns but no row: they have no energy term. Their `z_mu` stays the initializer's constant, which affects ε* but not z* or E*.

API:
- `validate_linear_gaussian(structure)`: DAG (`len(schedule) == len(node_order)`), node class `Linear` or `IdentityNode`, `IdentityActivation`, `GaussianEnergy` (precision from `node_info.energy.config`), `flatten_input=False`, rank-1 node shapes, no `energy()` override. Raises `ValueError` naming the node.
- `effective_edge_matrices(params, structure)`, `assemble_linear_quadratic(params, structure, clamps, *, source_means=None) -> LinearQuadratic` (fields: `free`, `rows`, offsets, `A`, `c`, `B_lower`, `M`, `z_ff`).
- `linear_equilibrium(params, structure, clamps, *, source_means=None) -> LinearEquilibrium`: `z_star`, `z_mu_star`, `error_star`, per-node `node_energy` (batch,), `total_energy` (batch,), `min_singular_value`. Uses `np.linalg.lstsq`; raises if rank < D (non-unique equilibrium, usually an unclamped source whose outgoing maps are not jointly injective).
- `theorem1_energy(params, structure, clamps) -> (E_star, S, r)`: chain-only closed form E* = ½·Σ_n r_n S⁻¹ r_nᵀ, r = y − μ_out at the feedforward point. Biases and muPC scales enter through r and P_l.
- `latent_hessian(quad)`, `epsilon_hessian(quad)`, `stability_bound(H) = 2/λ_max`, `excited_eigenvalues(H, g0)`, `settled_fraction(eta, steps, eigs) = 1 − (1 − eta·eigs)^steps`, `steps_to_contract(eta, eigs, ratio)`, `flatten_free`/`unflatten_free` in `node_order` (never `tree_leaves`, which sorts dict keys).

Module docstring states the regime-dependent conditioning result, not a blanket claim: λ_min(H_z) decays with depth even for benign weights (sPC's slow mode); λ_max(H_ε) = 1 + σ_max(J)² grows with downstream weight products (ePC's shrinking stability bound); from ε = 0 with uniform precision and no unclamped sources, ePC's trajectory lives in the d_y-dimensional row space of J and sees only eig(S).

### B. `fabricpc/core/inference_epc.py`

1. Extract the `energy_of` closure (lines 142–153) verbatim into `@classmethod error_energy(cls, params, state, clamps, structure) -> (energy_of, errors)`; `forward_value_and_grad` calls it. Same ops, same order, bit-identical gradients (pinned by the existing `TestGradientCorrectness`). Purpose: Hessian-vector products `jax.jvp(jax.grad(lambda e: energy_of(e)[0]), (errors,), (v,))` for the analysis script and one new test.
2. `regime_label(self) -> str`: returns e.g. `"eta*steps=0.005 (slowest error mode relaxed <=0.5%): backprop-like regime"`, using the linear-chain lower bound `1 − (1 − eta)^steps`. Bands: below 0.1 "backprop-like", 0.1 to 0.9 "partially relaxed", above 0.9 "near PC equilibrium if stable". Used by the demo and compare script; one owner of the wording.
3. Class docstring: replace the `infer_steps` sentence with the mechanism. One step from ε = 0 leaves ε_t = −η·∂L/∂z_t exactly (the backprop activation gradient), so hidden-layer weight gradients are η times backprop's to first order and the output layer's is backprop's; small η·T after T steps is the same to first order (paper Theorem C.9). On a linear chain the slowest excited error mode is relaxed by at least 1 − (1 − η)^T ≈ η·T after T steps. The defaults (η·T = 0.005) sit in the backprop-like regime deliberately: fastest and most stable training measured on resnet18. To study PC dynamics raise η·T toward 1 or more while keeping η below the stability bound 2/λ_max(H_ε), estimated for any graph by `scripts/epc_analysis.py --section stability`. Under Adam the η scaling of hidden-layer gradients is normalized away, so 1-step ePC with Adam trains as backprop with Adam. Keep the signature defaults unchanged (`tests/test_doc_defaults.py` pins them against the guide table).

### C. Tests

**`tests/test_linear_pc_oracle.py` (new).** Builders: `_chain(hidden, dims, std, use_bias, precisions, clamp_output, scaling)`, `_fork_merge`, `_clamped_internal` (x→h1→h2 clamped→y clamped), `_prior_source` (the `_convex_graph` shapes from `test_inference_epc.py`), and `_inject_biases(params, key)`: `Linear.initialize_params` zero-fills biases (`nodes/linear.py:191`), so `use_bias=True` alone exercises nothing. Weights via `NormalInitializer(std=...)`. Fixture bunch (parametrize ids): `chain-h1..h4` (x5, hidden 4, y3, std 0.3, batch 3), `chain-h2-bias`, `chain-h2-precision` (h1 precision 2.0, y 0.5), `chain-h3-std0.8` (λ_max(S) ≈ 69, stress), `fork-merge`, `clamped-internal`, `prior-source`, `unclamped-readout`, `mupc-chain-h3` (`MuPCConfig()`, `MuPCInitializer`).

- `TestOracleSelfChecks`: scalar chain with hand-built params x=1, W1=2, W2=3, y=1 → z_h* = 0.5, E* = 1.25, error h = −1.5, y = −0.5, S = [[10]], r = [[−5]], eig(H_ε) = eig(H_z) = [10]; `theorem1_energy` equals the lstsq energy on `chain-h1..h4` (rtol 1e-10) and on `chain-h2-bias`, `mupc-chain-h3`; error pull-back ε_l* = ε_y* P_lᵀ on chains; unclamped readout gives E* = 0 and z* = z_ff; normal-equation residual ≤ 1e-10 relative on `fork-merge`, `prior-source`; `epsilon_hessian` equals the explicit form diag(p) + Σ_{clamped t} p_t J_tᵀJ_t (15-line helper) and `triu(B_lower) == 0`, det M = 1; eigenvalue floor: min eig(H_ε) ≥ min precision on chains and < 1 on `prior-source`; `validate_linear_gaussian` rejects tanh, CrossEntropy, `flatten_input`, a cyclic graph with `unroll=1`, and `StorkeyHopfield`.
- `TestEPCReachesOracle::test_equilibrium[bunch]`: η = 1/λ_max(H_ε) as a Python float, T = `steps_to_contract(η, eigvalsh(H_ε), atol/(10·‖M‖₂·‖ε*‖))`; assert `z_latent` vs `z_star`, per-node `energy` vs `node_energy`, `total_energy` vs the oracle total, `error` vs `error_star` (rows; unclamped sources with `source_means` from the state), `assert_allclose(rtol=1e-4, atol=1e-4)`.
- `TestSPCReachesOracle::test_equilibrium[bunch]`: same with `InferenceSGD`, η = 1/λ_max(H_z), including `mupc-chain-h3` (with identity activations `topdown_grad_scale` is the exact chain-rule factor, so sPC+muPC is exact gradient descent on the scaled energy; the tanh-only divergence stays pinned by `TestMuPCDivergence`). Skip `error_star` on unclamped sources (sPC re-syncs source z_mu).
- `TestStabilityBracket[epc|spc]` on `chain-h3`: η = 0.95·bound converges to the oracle at 1e-4; η = 1.05·bound for 150 steps gives finite energy > 1e3 × the feedforward energy. This pins the scale of both gradient implementations, not only their direction.
- `TestEpsilonHVPMatchesOracle`: on `fork-merge` and `prior-source`, the HVP through `EPCInference.error_energy` equals `H_ε @ flatten_free(v)` column-wise per sample (atol 1e-4, float32 vs float64).

Expected cost ≈ 26 compiled loops, 25 to 35 s, no single test above 2 s, so no `slow` marks unless measured otherwise.

**`tests/test_inference_epc.py`, new `TestBackpropCorrespondence`.** Fixture x(4) → h1(3, tanh) → h2(3, tanh) → y(2), biases injected, batch 3, parametrized Gaussian and CrossEntropy outputs.
- `test_epsilon_grad_at_zero_is_backprop_activation_grad`: `forward_value_and_grad` at ε = 0 gives `latent_grad` equal to a hand-written batch-summed backprop recursion (δ_y = precision·(pre − y) or softmax(pre) − y; δ_h = (δ_next @ Wᵀ)·(1 − a²)), atol 1e-6. Exact, no first-order caveat.
- `test_one_step_error_is_minus_eta_backprop_grad`: after `EPCInference(eta_infer=0.05, infer_steps=1).run_inference`, `error == −0.05·g` on both hidden nodes and `z_latent == a − 0.05·g` on h1 (Theorem C.9 case 1 literally).
- `test_regime_label_bands`: the three bands and the default instance's label.
- Weight-gradient parity (η × backprop for hidden layers) is not tested here: `compute_local_weight_gradients` is batch-summed while the backprop trainer divides by batch, and the pending plan `docs/dev_plans/mean_gradient_normalization.md` owns that test. Add a note to that plan: its test #2 must be a first-order assertion (`|g_pc − η·g_bp| ≤ C·η²`), because the equality holds only to O(η²).

### D. `scripts/epc_analysis.py` (new)

CPU by default (`setup_jax(platform="cpu")` as in `scripts/diagnose_deep_mupc.py`), `--section` selects sections, target under two minutes without flags, tables to stdout, optional `--plot` (plotly html, png behind the kaleido guard from the compare script; apply the dataviz skill when writing the chart). Each section names the reviewer bullet it answers.

1. `backprop_regime` (bullet 5). Tanh MLP, batch 8: backprop gradients via `jax.grad` of `graph_energy(initialize_graph_state(...), node_names=target)` batch-summed; 1-step ePC local weight gradients at η ∈ {1e-4, 1e-3, 1e-2, 1e-1}; per-layer relative deviation from η·backprop (hidden) and from backprop (output), showing the O(η) approach. One Adam update from each gradient set on the same params: cosine similarity of the updates (the η scaling cancels under Adam). Regime table for the sweep grid (η ∈ {1e-4, 1e-3, 1e-2, 3e-2, 1e-1} × T ∈ {1, 2, 5, 10, 16, 32, 64, 128, 160}) from `settled_fraction` on a linear chain's eig(S), with measured ‖ε_T − ε*‖/‖ε*‖ for a few cells to validate the formula against the solver, and the label "unstable" where η·λ_max > 2.
2. `equilibrium_profile` (bullets 2, 3). Linear chains, hidden depth {3, 5, 10, 20}, width 32, output 10; weight std {0.5, 1.0, 1.5}/√fan_in and muPC. Oracle per-layer log10 E_l*, spread in decades, slope per layer (≈ 2·log10 of the downstream gain, from ε_l* = ε_y* P_lᵀ). sPC transient at η 0.1 for T ∈ {10, 50, 200, 1000, 5000} via `run_inference_with_history` (per-node `energy` is in the history dict) against the equilibrium profile: top-heavy early, approaching the oracle late. ePC at η = 1/λ_max for T ∈ {1, 5, 20}: all layers move at once.
3. `convergence_spectra` (bullets 1, 7). Depth 2 to 20, width 16: λ_max, λ_min of H_z and of H_ε (excited), steps to 1e-3 contraction at η = 1/λ_max for each solver, plain and muPC. Measured steps for two depths to validate. This quantifies "sPC gets there after a huge number of steps" and the 5-layer advice, and shows where ePC's bound shrinks with expanding weights.
4. `stability` (bullets 4, 6). `top_epsilon_eigenvalue(params, state, clamps, structure, iters)`: power iteration on the HVP through `EPCInference.error_energy`, jitted, pytree axpy via `tree_map`. Validate against the oracle's λ_max(H_ε) on a linear chain (agreement 1e-4), then apply to a gelu MLP: η_max = 2/λ_max; confirm ePC at 0.9·η_max descends and at 1.1·η_max diverges. Collapse experiment: 4-layer gelu MLP on an MNIST subset (`MnistLoader`, ~5k samples, batch 128, AdamW), ePC at η = 0.8·η_max(init) with T = 10 for a few epochs; every 10 updates recompute λ_max on a fixed probe batch and log η·λ_max, train energy, accuracy; report whether a collapse coincides with η·λ_max crossing 2. Report the outcome as observed.
5. `--resnet18` (optional, GPU + CIFAR). Import the demo builder as the compare script does (`importlib` load of `resnet18_cifar10_demo.py`), muPC graph, one CIFAR batch of 64: λ_max(H_ε) at init by power iteration → η_max, and the predicted regime label for each (η, T) cell of the recorded sweep, next to the measured accuracies (eta 0.1 collapsed at T = 1; 0.03 did not).

### E. Documentation touch points (backprop regime)

- `docs/user_guides/12_api_inference.md`: replace the one-line caution with a "Backprop regime" paragraph carrying the η·T rule, the stability bound, the Adam remark, and a pointer to the analysis script; tuning table rows `eta_infer (EPCInference)` and `infer_steps (EPCInference)` state the rule ("η·T ≲ 0.1 is backprop-like; raise toward ≥ 1 for PC equilibrium; keep η < 2/λ_max"). Default column values stay as they are.
- `docs/user_guides/03_how_predictive_coding_works.md` line 61: one sentence that small η·T makes ePC backprop with rescaled gradients.
- `CHANGELOG.md`: the `EPCInference` entry gains the regime sentence; `New` entries for `linear_pc_oracle`, `EPCInference.error_energy`, `regime_label`, and the analysis script.
- `examples/resnet18_cifar10_demo.py`: the settings print line appends `inference.regime_label()`. Docstring: relabel the `--infer_steps 1` result as the backprop-equivalent regime beside the backprop reference (76.73% vs 77.11%); rewrite "ePC achieves higher accuracy because deeper layers learn weights" to state the mechanism (at these settings ePC's weight gradients are backprop's, scaled by η and normalized by AdamW); record that the sweep's collapses were the PC-regime runs, consistent with η exceeding the ε-Hessian bound as weights grow, with the pointer to `scripts/epc_analysis.py --section stability`.
- `examples/epc_spc_resnet18_compare.py`: docstring gains the sweep interpretation (accuracy declines monotonically with η·T from the backprop value ≈ 38.8% toward the PC equilibrium ≈ 31%; sPC-120 at 34.6% sits between because it has not converged) and points to the tables in `docs/dev_plans_archive/epc_inference_solver.md`; the per-arm report adds a regime column from `regime_label()`.
- `docs/dev_plans_archive/epc_inference_solver.md`: one interpretation paragraph under the results tables.

## Alternatives considered

- **Runtime warning at construction.** Keyed on `infer_steps == 1`: fires on the wrong criterion (T = 5 at η 1e-3 is equally backprop). Keyed on η·T: fires on the library's own defaults. Rejected in favor of docs plus printed regime labels (user decision).
- **Change the defaults to a PC regime.** Every PC-regime resnet18 run collapsed over 100 epochs; a default that collapses the flagship demo is worse than a documented backprop-like default. Rejected (user decision); the stability diagnostic is the path to a future adaptive rate.
- **Oracle via `jax.jacfwd` of an affine residual built from the nodes' own `predict`.** Shorter, but not independent of the code under test. Rejected for explicit block assembly from params.
- **Oracle under `tests/` only.** The analysis script needs the same code; duplication or a `sys.path` import. Rejected (user decision).
- **Compare solvers only against each other (existing `TestSPCEquivalence`).** Two wrong solvers can agree; the oracle pins the exact answer and the gradient scale. The existing equivalence tests stay for nonlinear coverage.
- **Stability by trial and error only.** For linear graphs the bound is exact (2/λ_max(H_ε)); for any graph power iteration on the HVP gives the local bound at a few forward-backward passes. Chosen over trial and error.

## Verification

1. `python -m pytest tests/` green; `tests/test_inference_epc.py::TestGradientCorrectness` unchanged after the `error_energy` extraction; `tests/test_doc_defaults.py` green after the guide edits.
2. `python scripts/epc_analysis.py` completes on CPU in under two minutes and prints: O(η) approach of 1-step gradients to η·backprop; a regime table matching the sweep's flat/declining accuracy pattern; per-layer equilibrium spreads; the H_z/H_ε spectra table; HVP λ_max agreeing with the oracle to 1e-4; the 0.9/1.1 bracket on the gelu MLP; the collapse-experiment trace.
3. `python examples/resnet18_cifar10_demo.py --num_epochs 0.2` prints the regime label in the settings line; `python examples/epc_spc_resnet18_compare.py --mode convergence --track_steps 5` still runs (smoke).
4. Optional on the 3090: `python scripts/epc_analysis.py --section stability --resnet18` prints η_max at init for the muPC resnet18 graph and the predicted regime per sweep cell.

## Sequencing (one commit each, suite green at each gate)

1. Oracle module + `TestOracleSelfChecks`.
2. `EPCInference.error_energy` extraction + `regime_label` + solver-vs-oracle, stability-bracket, HVP, and backprop-correspondence tests.
3. `scripts/epc_analysis.py`.
4. Documentation touch points and demo/compare-script regime output; CHANGELOG; note in `mean_gradient_normalization.md`.

Commit messages via a temporary file in the project root.
