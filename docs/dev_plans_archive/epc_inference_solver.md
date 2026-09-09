# ePC inference solver, cyclic-graph unrolling, and the stability and regime probes

## Executive summary

FabricPC's two inference solvers (`InferenceSGD`, `InferenceSGDNormClip`) implement state-based predictive coding (sPC): latent states relax by local gradient descent, so the output-loss signal attenuates by the state learning rate per layer per step and deep graphs need 100s of inference steps (resnet18 demo: 120 steps, 476 s/epoch). The ePC paper (Goemaere et al., arXiv 2505.20137) reparameterizes PC over prediction errors: one reverse-mode auto-diff pass through the whole network delivers the loss signal to every layer unattenuated, reaching the same equilibrium in ~several steps. sPC remains the general solver for arbitrary graphs; ePC is the efficient solver for DAG (or unrolled-cyclic) representations.

This PR delivers four things. **(1) A node contract refactor.** `forward()` splits into `predict()`, which computes the node's prediction z_mu and any intermediates from params and in-edge inputs, and `energy()`, the per-sample score; the additive pair z_latent = z_mu + ε is owned by `NodeBase` and is not an override point. One prediction pass then serves both sPC and ePC parameterizations, and custom energy terms become first-class. **(2) ePC inference.** `EPCInference` relaxes the prediction errors ε and derives the latents by one forward pass per step, so one reverse pass carries the output signal to every layer; it is an `InferenceBase` subclass, which required the base's template methods to dispatch on the class and gain segment hooks. The topological sort is generalized to an unroll degree U, so a cyclic graph becomes ePC's DAG through `graph(..., unroll=U)`. `InferenceSchedule` composes solvers per weight update, a few ePC steps then sPC refinement on the exact energy. An exact linear-Gaussian oracle verifies both solvers' equilibria and stability bounds from params alone. **(3) A stability probe for any graph.** A Lanczos estimate of the error Hessian's excited spectrum gives λ_max, the largest curvature the starting gradient excites, so the rate bound η < 2/λ_max and the output-gradient reversal threshold at odd step counts are measured at init and tracked during training. **(4) A regime probe.** The relaxed fraction of the gradient-carrying modes, f̄, and of the top mode, f_max, place a setting between backprop-like (η·T·λ ≪ 1) and PC equilibrium, and `RegimeProbe` records the spectrum, the regime flags, and every weight's norm every N updates inside `train`.

Breaking changes: custom nodes implement `predict` and `energy` instead of `forward`; solver template methods are classmethods; cyclic graphs need an explicit `unroll`. Two behavior changes without an API change: reported eval energy now includes a free readout's term and is signed under a cross-entropy readout, and cyclic graphs receive true feedforward initialization, so their training curves shift. Two prerequisites the review cycles identified shipped as their own releases: 0.5.1 normalizes the weight gradients reaching the optimizer to means per prediction under both trainers, so one learning rate means the same for PC and backprop and is invariant to batch size and prediction count; 0.5.2 gives `train`'s iteration callback an `IterContext` carrying the parameters, the hook the probes run on. Measurements are in `docs/reports/epc_regime_and_stability_report.md`; tuning guidance in `docs/user_guides/12_api_inference.md`; the API list in `CHANGELOG.md`.

## Design overview

ePC solves the DAG; state-based PC (sPC, today's settling path) refines around cycles; a cycle can instead be unrolled into ePC's own graph. Symbols: `T_s` = settling ticks per update in sPC alone, `T1` = ePC steps, `T2` = sPC refinement steps, `U` = unrolled cycle traversals, `(H)` = a Hopfield (recurrent) node.

**1 — Composable inference schedule.** Solvers are schedule entries composed per weight update, not trainer modes. sPC minimizes the exact graph energy as-is; on a cyclic graph ePC minimizes an approximation whose fidelity is set by the unroll degree.
```
schedule = [ ePC(T1), sPC(T2), WeightUpdate ]

clamp ──► ePC: T1 steps ──► sPC: T2 steps ──────────► weight update ──► next batch
          solves the DAG,   minimizes full-graph
          back-edges        energy, back-edges in;
          excluded or       warm-started from ePC's solution
          unrolled
```

**2 — Where the error lives.** From a feedforward start, sPC moves error one hop per tick with per-hop damping: after `T_s` ticks the error profile decays exponentially from the output clamp and deep layers have seen almost nothing. ePC pushes the output error through the full depth in every step, so ~5 steps leave signal at every layer. What remains is the residual the back-edges introduce; it enters at the cycle, diffuses a few local hops per sPC tick, and the Hopfield node falls onto its fixed-point attractor within a few iterations:

```
error magnitude by depth — deep chain with one Hopfield cycle mid-network

sPC alone (T_s ticks,        ePC (~5 steps):            + sPC refinement (~5 ticks):
feedforward start):          full depth reached         residual local to the cycle

out ████████                 out ██████                 out ·
    ████                         ██████                     ·
    ██                           ██████                     ▪
    █                            ██████                  ┌─►(H)──┐  error hops the cycle;
    ▏                            ██████                  └───▪───┘  (H) settles on its attractor
    ▏                            ██████                     ▪
in  ▏                        in  ██████                 in  ·
```

**3 — The alternative cycle path: unroll into ePC's DAG.** `U` traversals of the cycle become `U` copies of the cyclic subgraph inside one differentiable program — no separate solver, no stopping rule, no handoff; `U` is fixed at graph-build time (`graph(..., unroll=U)`):

```
cyclic graph                     unrolled into ePC's DAG (U = 2)

a ──► b ──► c ──► d              a ──► b₀ ──► c₀ ──► b₁ ──► c₁ ──► d
      ▲     │
      └─────┘                    the back-edge c→b becomes the feedforward edge c₀→b₁
```

The subscripts b₀, b₁ are repeated forward passes through node b, each re-injecting the same ε_b: errors are tied between traversals by construction (mechanism in Section 2.1).

Two questions are experimental, and the benchmark script (Section 2.7) supplies the first measurements: whether an ePC warm start helps a given architecture and at what depth, and whether a cycle is better refined by sPC, unrolled into ePC, or both (an unrolled ePC segment followed by sPC refinement is already expressible with the composed solvers). Cycle depth drives both sPC's signal decay and ePC's unrolling cost.

## Symbols

| Symbol | Meaning |
|---|---|
| z_i, μ_i, ε_i | node i's latent state (`NodeState.z_latent`), its prediction from in-edge sources computed by `predict` (`NodeState.z_mu`), and its prediction error z_i − μ_i (`NodeState.error`), the variable `EPCInference` relaxes |
| p_i | node i's Gaussian precision, 1.0 unless set |
| E | total energy, Σ over nodes with in_degree > 0 of the node's `energy()`: ½·p_i·‖z_i − μ_i‖² by default, a cross-entropy term at a cross-entropy output, plus any custom term |
| aux | a pytree of intermediates `predict` returns for `energy`, computed from params and inputs only |
| η, T | `eta_infer` and `infer_steps` of the `EPCInference` under discussion |
| T_s, T1, T2, U | sPC's own step count; the ePC and sPC segment lengths of an `InferenceSchedule`; the unroll degree of `graph(..., unroll=U)` |
| H_ε, H_z | Hessians of E in error and in latent coordinates at the feedforward point ε = 0 |
| g0 | ∇_ε E at ε = 0, the starting gradient; equal to the backprop activation gradient at every node |
| excited mode | an eigenvector of H_ε along which g0 has a nonzero component, the only modes gradient descent from ε = 0 moves |
| λ, λ_max, λ_min | an eigenvalue of H_ε; the largest and smallest among the excited modes |
| J, S, r, d_y | on a linear chain: J maps the stacked hidden ε to the output prediction μ_y; S = I + JJᵀ (Innocenti et al. 2024, Theorem 1); r = y − μ_y is the feedforward output residual; d_y the output width |
| σ_max(J) | the largest singular value of J, so λ_max = 1 + σ_max(J)² at unit precision |
| f(λ), f_max | relaxed fraction 1 − (1 − ηλ)^T of a mode with eigenvalue λ after T steps; f_max = f(λ_max) |
| θ_k, w_k | the k-th Ritz value (an eigenvalue estimate from the Lanczos tridiagonal matrix) and the fraction of ‖g0‖² that mode carries; Σ_k w_k = 1 |
| f̄ | gradient-weighted relaxed fraction Σ_{θ_k > 0} w_k·f(θ_k) / Σ_{θ_k > 0} w_k over the positive Ritz modes (`Regime.f_weighted`) |
| α_j, β_j | the Lanczos recurrence coefficients, the diagonal and off-diagonal of the tridiagonal matrix |
| λ_eff | one eigenvalue fitted to the 2-epoch sweep accuracies through f(λ) (Section 4.3) |
| ε*, E* | the equilibrium error and the equilibrium energy, the minimizers of E over ε |

## 1. Node contract: predict / energy

Before: `forward()` fused three stages in the sPC dataflow direction. Predict computes z_mu from params and in-edge inputs; pair computes error = z_latent − z_mu, copied into every node body; score applies the energy functional plus any in-forward custom term. sPC consumes z_latent and produces ε. ePC needs the reverse, ε to z_latent, and needs z_mu before z_latent exists. With the fused method the ePC derive pass would call `forward()` twice per node, once for z_mu and once for the energy at the derived latent, and a `predict` that reads the node's own state would return two different z_mu. Three defects rode along: `GaussianEnergy` recomputed the difference and never read `state.error`; custom energy terms were post-hoc `state._replace` patches (StorkeyHopfield, `ScaledSumNode`); and source semantics lived in the solver, so `IdentityNode.forward` crashed if called on an `in_degree == 0` node.

After: node authors implement two staticmethods, and `NodeBase` owns the pair and the assembly.

```python
@staticmethod
def predict(params, inputs, state, node_info) -> Tuple[jnp.ndarray, Any]:
    """z_mu of shape (batch,) + node_info.shape, plus aux (None if unused)."""

@staticmethod
def energy(params, inputs, state, aux, node_info) -> jnp.ndarray:
    """Per-sample energy (batch,) at (state.z_latent, state.z_mu); default is node_info.energy's functional."""
```

Four rules, each with its mechanism:

- `predict` must not read `state.z_latent` values (shape and dtype reads are fine). sPC differentiates z_mu through such a read while ePC evaluates z_mu at the carried latent, so the two solvers would minimize different energies.
- aux carries intermediates that depend only on params and inputs (`Linear`'s pre-activation; StorkeyHopfield's coupling matrix W and attractor weight `strength`). aux is snapshotted when `predict` runs, and under ePC z_latent is replaced after `predict` and before `energy`, so an aux entry computed from z_latent freezes the carried latent into an energy otherwise evaluated at the derived one.
- An energy term that needs the node's own latent reads `state.z_latent` inside `energy()`, as the Hopfield attractor term does. Both solvers then evaluate the term at the same latent as the rest of the energy.
- `energy` overrides tolerate `aux=None`. An `in_degree == 0` node gets no `predict` call, and a source's params are empty, so StorkeyHopfield's `(W, strength)` cannot be recomputed there and the override falls back to the base term.

Templates, all on `NodeBase` and none an override point (`tests/test_node_contract.py` asserts every registered node resolves them to the base): `pair_error(z_latent, z_mu) = z_latent − z_mu` and its inverse `pair_latent(z_mu, error) = z_mu + error`; `forward_with_aux`, the sPC-direction assembly that returns the new state and aux; `forward`, its aux-dropping wrapper with the pre-split signature and bit-identical output; and `forward_from_error`, the ePC-direction assembly that writes z_latent := z_mu + ε with one `predict` per visit. The pair is additive and base-owned because the ε ↔ z_latent equivalence needs a volume-preserving bijection. If a precision-weighted ε is ever wanted, the pair moves to `EnergyFunctional` so both directions and the energy stay consistent.

Source semantics have one owner. An `in_degree == 0` node has no in-edges to project a z_mu, so `forward` mirrors z_mu ← z_latent (cast to z_mu's float dtype, since a token clamp may be int) with error = 0, and `forward_from_error` derives an unclamped source's z_latent from the frozen z_mu.

Readout fix, bundled. sPC's `forward_and_latent_grads` forced an unclamped `out_degree == 0` node to error = 0 and energy = 0 and zeroed its `latent_grad`, contradicting the method's own contract and discarding a Hopfield readout's attractor energy. The branch is deleted. An unclamped readout takes the ordinary path: error = z_latent − z_mu, energy as `forward` assigns, z_latent relaxed like any other node. Predictions read z_mu at every eval site, and `evaluate_transformer` migrated from z_latent to z_mu in the same change (the two coincide only under zero-error init on a DAG). Reported eval energy changes because `eval_step` sums energy over all nodes and the readout's was zero before. With a free cross-entropy readout the eval energy is signed, since −Σ z·log μ is linear in the free z, so the trainer tests assert finiteness and `target_energy ≥ 0` instead of `energy ≥ 0`.

Migration, complete and without fallbacks: the thirteen library `forward()` bodies became `predict()` by deleting the pair and energy tail (`linear.py`, `identity.py`, `skip_connection.py`, `linear_residual.py`, `convolutional.py`, `pooling.py`, `storkey_hopfield.py`, `transformer.py`, and the five `transformer_v2.py` nodes); StorkeyHopfield's attractor term became an `energy()` override; `LinearExplicitGrad`'s analytic overrides call `forward_with_aux`; the custom nodes in `examples/jpc_fc_resnet_compare.py`, `tests/test_external_custom_node.py` (`ScaledSumNode`'s post-hoc weighting became a three-line `energy()` override), and `tests/test_mupc.py` migrated; `energy_functional` is deleted into the default `energy()`. `docs/user_guides/06_custom_nodes.md` is rewritten around the two-method contract and `10_api_nodes.md` states the rules; both are AST-checked by `test_doc_snippets.py`.

## 2. ePC inference

### 2.1 Formulation

Intuition: sPC moves the output error one hop per tick with per-hop damping, so after T_s ticks a deep layer has seen almost nothing (Design overview). ePC makes the prediction errors the relaxed variables and rebuilds the latents from them by a forward pass, so one reverse pass through the whole network carries the output error to every layer in every step.

Mechanism: ε is first-class and z_latent is derived in schedule order, z_i := μ_i + ε_i, where μ_i is recomputed by `predict` from the upstream derived latents. Every node's energy depends on all upstream ε, including the clamped output's `energy(y_clamp, μ_y)`, which is the paper's output loss. One `jax.value_and_grad` over the ε pytree through the whole derived forward gives exact ∇_ε E, and ε steps down it: ε ← ε·(1 − η·decay) − η·∇_ε E, with decay the constructor's `latent_decay`. This is the pattern `train_backprop` already uses, one feedforward pass and one `value_and_grad`, with the differentiated variable swapped from the weights to ε. Because a change in one node's ε moves every downstream latent, η is tuned like a weight learning rate, not like sPC's local rate.

Math: the ε ↔ z_latent map is a bijection with unit-determinant triangular Jacobian (paper Appendix C), so the two parameterizations have identical energies and identical equilibria, and the final derived state feeds the existing local weight-gradient path unchanged. Two exceptions: on a cyclic graph ePC minimizes the unrolled approximation, whose fidelity U sets, while sPC minimizes the exact graph energy; and with muPC attached on a nonlinear graph the two solvers settle to different fixed points (muPC paragraph below). At ε = 0 the derived states equal `FeedforwardStateInit`'s output, so the default initializer is the paper's zero-init at any unroll degree, because both iterate `structure.schedule`.

One derive rule for every node, with the clamp deciding which side is free (computed at trace time from static structure and clamp keys):

- Unclamped: ε is relaxed and the derive pass writes z_latent := z_mu + ε, whatever the node's degree. A top-down prior (`in_degree == 0`) keeps the z_mu assigned at initialization and its ε receives gradient through downstream z_mu; since E excludes sources, that is the same signal sPC accumulates into the source's latent, so the solvers share equilibria on graphs with unclamped priors. An eval readout (`out_degree == 0`) takes the same path: at ε = 0 a pure-Gaussian readout's gradient is zero, so it stays put, while a Hopfield readout receives its attractor gradient.
- Clamped: z_latent stays the clamp and ε is derived as `pair_error(clamp, z_mu)`; a clamped source keeps its init state. Clamped nodes never enter the relaxed pytree, which keeps int-dtype token sources out of the AD pytree.

Sources at initialization: every initializer left `in_degree == 0` nodes with z_mu = 0 and error = 0, violating error = z_latent − z_mu. sPC masked this on its first forward; ePC would read the invalid z_mu directly. `initialize_graph_state` now runs one shared post-pass after the dispatched initializer: z_mu ← z_latent (cast to z_mu's float dtype) and error ← 0 for every source. One implementation point covers `GlobalStateInit`, `NodeDistributionStateInit`, `FeedforwardStateInit`, and any future initializer.

Cyclic graphs: the derive pass iterates `structure.schedule`, in which cycle members repeat U times. Errors are tied between traversals by construction. The relaxed pytree keys ε by node name, `GraphState` carries one `NodeState` per node, and each visit re-injects the same carried ε, recomputes z_mu from the latest source latents, and overwrites the node's single state. Node i's energy enters E once, at its final visit, and that value is the output of a computational graph threaded through every traversal within the step, so the single AD pass differentiates through all U traversals. Across steps the picture differs: each inference step starts from the carried `GraphState`, so a cycle member's first visit reads the previous step's last-visit latent, the effective traversal depth grows as T × U across a segment, and the gradient treats the previous step's latents as constants. StorkeyHopfield needs no special handling: its z_mu comes from the input probe and its self-recurrence enters only through the attractor term on its own z_latent, evaluated at the derived latent.

muPC: `scale_inputs` applies inside the differentiated forward exactly where the sPC loop applies it, and the global AD supplies the chain-rule factors. The per-hop gradient preconditioners (`jacobian_gain`, `self_grad_scale`) condition sPC's local flow and are not replicated in ePC's global backward pass. On a linear graph the gain is constant and both solvers reach the oracle's equilibrium; on a nonlinear graph `jacobian_gain` depends on the edge's target activation and reshapes sPC's flow, so the two solvers settle to different fixed points, and a slow-marked test pins the divergence. The ResNet-18 measurements of Section 4.6 compare the solvers by accuracy, not by a shared equilibrium. `scale_weight_grads` at learning time is untouched.

Memory: each ePC step stores activations for the whole derived forward, full depth times U cycle copies, reverse-mode memory at backprop scale. `train_backprop` already runs the same reverse pass on these demos, so resnet18 at batch 256 fits.

### 2.2 `EPCInference`

`fabricpc/core/inference_epc.py`: `EPCInference(eta_infer=1e-3, infer_steps=5, latent_decay=0.0)`, an `InferenceBase` subclass that inherits `inference_step`, `zero_grads`, `run_inference`, and `segments` and overrides:

- `derive_states(params, state, clamps, structure)`: iterate `structure.schedule`; per visit `gather_inputs`, `scale_inputs`, `forward_from_error(..., is_clamped=(name in clamps))`.
- `forward_value_and_grad`: build the relaxed pytree `{name: state.error}` over the unclamped nodes and take one `jax.value_and_grad` of `error_energy`, which writes the ε leaves into the state, runs `derive_states`, and sums `energy` over `in_degree > 0` nodes, the same set the trainer's energy uses. Grads land in `latent_grad` by accumulation. `error_energy` is public so Hessian-vector products can be taken through the solver's own energy (Section 3.4).
- `update_latents`: every relaxed node's ε steps by `compute_new_error`, the update rule of Section 2.1; `compute_new_latent` has no ePC caller and raises.
- `begin_segment`: resync ε := z_latent − z_mu at the carried latents by one sPC-direction sweep over `node_order`. The first `derive_states` then reproduces the incoming z_latent exactly on a DAG, node by node in schedule order. This is the inverse direction of the bijection, and it is what lets a distribution initializer's random latents or a preceding sPC segment's final update survive the handoff. An identity handoff would overwrite every internal latent with a derive from stale ε (ε = 0 after every built-in initializer; one update stale after an sPC segment). On cyclic graphs cycle members are preserved at their first visit only. With `FeedforwardStateInit` on a DAG the sweep yields ε = 0. Cost: one forward pass per segment entry.
- `finalize_state`: one detached `derive_states` so the final state satisfies z_latent = z_mu + ε with energies at the final point, which is the paper's weight rule; `compute_local_weight_gradients`, the train-loop energy, `eval_step`, and every dashboard reader of `.error` then work unchanged.
- `regime(spectrum) -> Regime` (Section 4.4).

`NodeState.error`'s docstring states the dual role: prediction error z_latent − z_mu, and under `EPCInference` the relaxed variable ε. `latent_grad` is sPC's one-hop dE/dz_latent or ePC's ∇_ε E through the full derived forward.

The default `eta_infer` went 1e-3 → 1e-2 → 1e-3 (Record). At 1e-2 the 100-epoch resnet18 runs collapsed at every step count. At 1e-3 with `infer_steps=5` the solver is backprop-like at init on that graph, and the guide tells users to measure their own λ_max (Section 4.6 on why a default that collapsed at epoch 20 was kept).

### 2.3 `InferenceBase` dispatch and segment hooks

Before: the template methods re-resolved their own class via `type(structure.config["inference"])`, which breaks any composition. After: they dispatch on `cls`, and `run_inference` is an instance method.

| Method | Change |
|---|---|
| `inference_step`, `forward_value_and_grad`, `update_latents` | `@classmethod`; bodies use `cls.zero_grads`, `cls.forward_value_and_grad`, `cls.update_latents`, `cls.compute_new_latent` |
| `zero_grads`, `compute_new_latent` | unchanged static |
| `run_inference` | instance method: `cls.begin_segment`, then `lax.fori_loop` of `cls.inference_step(..., self.config)`, then `cls.finalize_state` |
| `begin_segment`, `finalize_state` | new classmethods, default identity, the segment-boundary hooks ePC overrides |
| `segments()` | new instance method, default `((self, infer_steps),)`; a schedule flattens its components' |

The module-level `run_inference` delegates to the configured solver with its signature unchanged, so the trainers needed no edit; the tests that called the old static form migrated. Both state-based solvers, `InferenceSGD` and `InferenceSGDNormClip`, inherit the hooks with identity defaults, behave as before, and compose in a schedule.

### 2.4 Unrolled topological schedule

Before: on a cyclic graph `_topological_sort` returned a partial order behind a print warning. On `x→a⇄b→y` the order was `("x",)`, so feedforward init left the cycle members and everything downstream at random init, and muPC attached no scaling to them. After: `_topological_sort(nodes, edges, unroll: Optional[int] = None)` returns the full visit schedule.

- On a DAG the existing Kahn loop runs unchanged, whatever `unroll` is, so every existing DAG gets a bit-identical order.
- On a cyclic graph with `unroll=None` it raises `GraphCycleError` naming the unordered nodes; the caller must pass `unroll` explicitly, even for the degenerate single-visit choice.
- With `unroll=U ≥ 1`: iterative Tarjan SCC, Kahn on the condensation DAG, each nontrivial SCC's members emitted U times in BFS order from the entry nodes. `x→a⇄b→y` at U = 2 gives `("x","a","b","a","b","y")`.

`GraphStructure` gains `schedule` after `node_order`, with `node_order == first_occurrence_order(schedule)` enforced at build; `graph(..., unroll=U)` validates U ≥ 1 and rejects bools (`unroll=True` would otherwise build silently at degree 1). Consumers: `FeedforwardStateInit` pass 2 walks `structure.schedule`, so cyclic graphs gain true feedforward init through cycles and initialization is the derived forward at ε = 0 on the same schedule ePC walks; muPC stays on the unique `node_order` (one merge-sum term per merge node regardless of visit count) with a duplicate-entry raise as hardening; `examples/mnist_cyclic_graph.py` and the cyclic test helper pass `unroll` explicitly. U is graph-owned rather than solver-owned because the initializer and the solver must agree on one schedule.

### 2.5 `InferenceSchedule`

```python
inference = InferenceSchedule(
    EPCInference(eta_infer=1e-3, infer_steps=2),   # global steps to near-equilibrium
    InferenceSGD(eta_infer=0.05, infer_steps=20),  # refine on the exact arbitrary-graph energy
)
```

Contract: states are initialized once before the first segment; each solver receives z_latent, z_mu, and error exactly as the previous segment left them, and its own `begin_segment` establishes its parameterization (ePC resyncs ε; sPC's is identity); the next solver continues from the previous one's finalized state. `segments()` flattens nested schedules; `inference_step` and `compute_new_latent` raise, since a schedule has no single per-step rule.

Tracking: `run_inference_with_history` and `make_inference_history` iterate `segments()` inside one jitted program, run `begin_segment`, the segment's steps, and `finalize_state`, and concatenate the per-step metric stacks, so every sampled ePC state is the derived state `run_inference` would return at that step count. `make_tracked_probe(structure)` compiles `initialize_graph_state` and `run_inference_with_history` into one program, the per-step-metrics twin of `make_tracked_settle` in `fabricpc.utils.dashboarding`; it exists because eager init and jitted tracking can select different cuDNN convolution algorithms (TF32 against FP32) and record a phantom step-0 energy. Under a composed schedule the concatenated `latent_grad_norm` series carries each segment's own gradient meaning.

### 2.6 Linear oracle

`fabricpc/utils/linear_pc_oracle.py`, NumPy float64, never calls node or solver code. On a linear-Gaussian DAG the energy is the quadratic E = ½‖A z_free − c‖² over the stacked free latents, with A the precision-weighted edge map and c the clamp-dependent constant, both assembled from params, edges, muPC forward scales, the IdentityNode scale, biases, and precisions. `linear_equilibrium` solves it by least squares; `theorem1_energy` gives the chain closed form E* = ½·r S⁻¹ rᵀ (Innocenti et al. 2024, Theorem 1, extended to precisions and biases); `epsilon_hessian` gives H_ε = MᵀAᵀAM with M = (I − B)⁻¹ the ε → z map and B the strictly lower-triangular edge map; `stability_bound`, `excited_eigenvalues`, `relaxed_fraction`, and `steps_to_contract` are the diagnostics. `validate_linear_gaussian` rejects nonlinear, non-Gaussian, and cyclic graphs; the DAG check is a per-edge back-edge test on `node_order`, since `len(schedule) == len(node_order)` also holds for a cycle at U = 1.

Three design facts the oracle's review settled. An unclamped source contributes no curvature floor in ε coordinates: it has columns in the quadratic form but no residual row, because a source has no energy term. "ePC is better conditioned than sPC" holds per regime: λ_min(H_z) decays with depth even for benign weights (sPC's slow mode), while λ_max(H_ε) grows with the downstream weight products (ePC's shrinking stability bound). The 1-step identity ε_1 = −η·g0 is exact, but the weight-gradient identity "1-step ePC equals η × backprop" holds only to first order in η·λ_max, because `finalize_state` re-derives the latents before the local weight gradient is taken.

What it verifies: both solvers reach the oracle's equilibrium on twelve graph shapes (chains to depth 4, biases, precisions, fork-merge, a clamped internal node, an unclamped prior, an unclamped readout, a muPC chain); stability brackets at 0.95× and 1.05× of 2/λ_max on the plain and muPC depth-3 chains, which pin each solver's gradient scale and not only its direction (a positive diagonal preconditioner shares the fixed point with plain descent, so an equilibrium test alone cannot); the Hessian-vector product against H_ε; and the Lanczos extremes and f̄ against the oracle in float32 and float64. `TestBackpropCorrespondence` pins g0 against a hand-written backprop recursion, `error == −η·g0` after one step, and the local weight gradients through `pc_weight_gradients` against a backprop reference divided by the same prediction count, `grad_denominator` (release 0.5.1), exact at the first hidden layer and with a remainder downstream bounded by a constant times η·λ_max. ePC on cyclic graphs has no oracle; its coverage is the sPC-equivalence tests.

### 2.7 Benchmark script

`examples/epc_spc_resnet18_compare.py` loads the demo's builder, which now takes an `inference: InferenceBase`. `--mode convergence` runs both solvers from identical params and state on one test batch and plots log per-node energy against step, one line per node colored by depth, because the global sum is dominated by output-adjacent nodes (Pinchetti et al., arXiv 2407.01163) and can read as converged while deep nodes have received nothing; sPC's final total energy after its 120 steps is the head-to-head criterion, reported as the number of ePC ε updates needed to reach it, and `--epc_eta` accepts a list. `--mode sweep` trains one arm per T in `--epc_step_sweep` plus an sPC baseline for the same epochs, so each arm is one (wall-clock, accuracy) point and the T grid is the time axis. Accuracy at equal wall-clock and wall-clock to equal accuracy are interpolation and selection statistics, reported as per-trial values and mean ± SE and never tested; the experiment runner separates tested contrasts (`ContrastResult`) from descriptive summaries (`DescriptiveDelta`) for exactly this reason. Outputs `epc_step_sweep.html` and, behind a kaleido guard, `.png`. The recorded sweep is the report's Appendix A.

## 3. Stability probe

### 3.1 The quadratic model at ε = 0

Every FabricPC run starts inference at the feedforward state, ε = 0. Near that point E is quadratic in ε. Its gradient there, g0, equals the backprop activation gradient at every node, because z_i = μ_i + ε_i makes ∂E/∂ε_i the same downstream chain as ∂E/∂z_i. Gradient descent on a quadratic splits into independent modes along the eigenvectors of H_ε. One step multiplies a mode's distance to its minimum by (1 − ηλ): for 0 < ηλ < 1 the mode moves part of the way on the same side; for 1 < ηλ < 2 it overshoots but ends closer; for ηλ > 2 it ends farther away than it started. After T steps a mode has closed the fraction f(λ) = 1 − (1 − ηλ)^T. Only the excited modes move; a mode with no g0 component stays at zero. The formula matches the solver to four decimals on a linear chain (report Section 5.4).

### 3.2 The stability bound

At every T, η·λ_max < 2 is required, or the top mode's distance grows every step. On a linear chain at unit precision H_ε = I + JᵀJ and g0 = Jᵀr, so the excited modes are the d_y directions of the row space of J and their eigenvalues are eig(S). λ_max = 1 + σ_max(J)² grows with the product of the downstream weights, and that product grows during training, so a fixed η eventually crosses the bound. λ_max is exponential in depth for expanding weights; muPC's forward scales hold it near 10 to 40 at init on the demo graphs. Normalization layers and weight decay bound the weight scale. The demo has no normalization layers, and the control runs (Section 4.6) show that on this graph the growth is ePC's own rather than the weight scale's: λ_max ran away only in the ePC cells with the larger η·T while the weight norms fell in every run. The control runs establish that attribution, not the mechanism; how relaxation beyond the backprop-like regime raises the curvature of the ε-energy is not derived here (Section 7).

### 3.3 The output-gradient reversal at odd T

Along a mode with eigenvalue λ, the output residual after T steps is r_T = (r/λ)·[1 + (λ − 1)(1 − ηλ)^T]; its equilibrium value r/λ per mode is r S⁻¹. At T = 1 this is r_1 = (1 − η(λ − 1))·r. The output layer's local weight gradient is proportional to r_1, so it reverses sign along the top mode once η(λ_max − 1) > 1, which for λ_max > 2 comes before the iteration bound η·λ_max > 2. The hidden errors ε_T = f(λ)·ε* keep the sign of the equilibrium error ε* for every ηλ < 2, so the reversal is confined to the output layer. For general T the condition is (1 − ηλ)^T < −1/(λ − 1), which can hold only at odd T because (1 − ηλ)^T ≥ 0 at even T. This is the flag users act on; η·λ_max > 2 remains the iteration bound. In the 2-epoch sweep the (0.1, 1) cell at η(λ_max − 1) = 1.5 collapsed and the (0.03, 1) cell at η(λ_max − 1) = 0.46 did not. In both collapsing control runs the flag fired one probe (50 updates) before the crossing.

### 3.4 Measuring the spectrum: Lanczos from g0

`fabricpc/core/epsilon_spectrum.py` runs the three-term Lanczos recurrence from v0 = g0 using only Hessian-vector products, `jax.jvp` of `jax.grad` of `EPCInference.error_energy`. Lanczos builds an orthonormal basis of span{g0, H g0, H² g0, …} and represents H_ε in it as a k × k tridiagonal matrix whose eigenvalues, the Ritz values θ_k, converge fastest at both ends of the spectrum. The weight w_k is the fraction of ‖g0‖² that Ritz mode carries, and the residual of an extreme is β_k·|s_{k−1}|, with s_{k−1} the last of the k components of that Ritz vector in the Lanczos basis, zero after a breakdown. One run therefore returns λ_max (the bound 2/λ_max), λ_min (negative means indefinite), and the distribution of the gradient over the spectrum, from which f̄ is computed. Three vectors are carried, so memory is independent of the iteration count.

Why not power iteration. On a graph with gelu activations and a softmax cross-entropy output, H_ε contains a sum over output components o of (∂E/∂μ_o)·∂²μ_o/∂ε², the output-loss gradient weighting the second derivatives of the network map, and that term makes H_ε indefinite once the weights are large. On the ResNet-18 it is indefinite already at init (λ_min = −0.42 with 1.2% of the gradient weight on negative curvature). Power iteration from a random start converges to the eigenvalue of largest magnitude: on the collapsing (1e-2, 1) run it returned −9 399 where Lanczos gives λ_max = 6 132 and λ_min = −354, and the first regime label read that negative value as "backprop-like" while the crossing detector never fired. It also carries no convergence indicator and needed 300 iterations for four digits on 4-node graphs where the probe ran 30; its Rayleigh quotient is a lower bound on λ_max, so an unconverged value overstates the safe rate.

Why the weights, not only the extremes. On a linear graph g0 lies in the row space of J, so every w_k sits on one of the d_y eigenvalues of S and the two extremes describe the excited band. On a nonlinear graph the second-derivative term couples g0 to nearly every direction: on a gelu MLP 943 of 1024 modes are excited, the smallest excited eigenvalue lies below the unit-precision floor, and 80% of the gradient weight lies between 1.06 and 1.39 (report Appendix B). A band on λ_min would say the slowest mode is far from relaxed while the modes carrying the gradient are done. f̄ from 30 Lanczos steps equals f̄ from the exact eigendecomposition to three digits.

Numerics. With eps(dtype) the machine epsilon of the dtype: when β_j ≤ √eps(dtype)·max(max_i |α_i|, max_i β_i) the Krylov space is exhausted and the recurrence stops. An exact-zero test fails: on a rank-3 excited spectrum β_3 is about 1e-6·|α| in float32, the recurrence continues on rounding noise, and the unexcited floor λ = 1 appears as a Ritz value (report Appendix B). The guard alone is not sufficient either: on a depth-5 chain with 10 excited modes the rounding accumulated over ten steps kept β_10 above the threshold and the floor was reported as λ_min. So `lanczos_extremes` takes λ_max and λ_min over the Ritz modes whose weight w_k exceeds eps(dtype). This is the same cutoff the guard expresses through β: a direction that enters the recurrence with β/|α| below √eps has a squared component in g0, its weight, below eps. `ritz_values` and `ritz_weights` are returned unfiltered. After the fix λ_min on that chain matches the oracle's excited minimum to 4e-8. A batch's λ_max is the per-sample maximum, since `error_energy` sums per-sample energies and H_ε is block-diagonal over samples; the 64-sample probe batch is therefore conservative relative to the 256-sample training batch. When ‖g0‖ = 0 a random start is used and flagged. Ghost eigenvalues from the absence of reorthogonalization duplicate converged extremes and split their weight; they move neither the extremes nor the weighted sums, so full reorthogonalization (k vectors of ε size) was not adopted. Cyclic graphs: the warm-started carried latents make the unrolled ε-energy not a pure function of ε, so the oracle excludes them and the estimator is unvalidated there.

### 3.5 API

- `make_epsilon_spectrum(structure, iters=30)` compiles `(params, state, clamps, key) -> EpsilonSpectrum`; `epsilon_spectrum(...)` is the one-shot form; `lanczos_extremes(hvp, v0, iters)` is the recurrence on any Hessian-vector product. `EpsilonSpectrum` carries `lambda_max`, `lambda_min`, `ritz_values`, `ritz_weights`, `residual_max`, `residual_min`, `negative_weight`, `gradient_norm`, `random_start`, `iters`, `k`. `weighted_relaxed_fraction(spectrum, eta, steps)` is f̄.
- `EPCInference.regime(spectrum)` reads the stability side into `Regime.unstable` (η·λ_max > 2) and `Regime.output_gradient_reverses` (Section 3.3), beside the regime fields of Section 4.4.
- `scripts/epc_analysis.py --section stability` checks Lanczos against the oracle's excited extremes and weighted fraction on linear chains; `--resnet18` measures the spectrum at init on the demo graph (17 s including compile); `examples/resnet18_cifar10_demo.py --track_regime N` records it every N updates during training.
- The module lives in `fabricpc/core` because it calls `EPCInference.begin_segment` and `error_energy`, and no module in `fabricpc/core` imports from `fabricpc/utils`; the oracle stays in `fabricpc/utils` and is NumPy-only.

## 4. Regime probe

### 4.1 Three regimes

When η·T·λ ≪ 1 on every excited mode, the T-step result is ε ≈ −η·T·g0, and the local weight gradients computed from those errors are backprop's, hidden layers scaled by η·T and the output layer unscaled (Goemaere et al., Theorem C.9; T = 1 gives the paper's Case 1 with scale η). That is the backprop-like regime. When every excited mode has relaxed, ε is the PC equilibrium and the output error is r S⁻¹, which damps the learning signal along each excited mode by its eigenvalue of S, a matrix rescaling a per-parameter optimizer cannot undo; on the ResNet-18 up to 16×. That is the PC-equilibrium regime, and it is consistent with the 31% plateau trailing the 38.8% backprop-like plateau in the 2-epoch sweep. Between them a setting is partially relaxed. Two optimizer caveats: under Adam the η·T scaling of the hidden-layer gradients is normalized away only while the scaled gradient magnitude stays far above Adam's epsilon of 1e-8; without Adam, hidden layers learn η·T times slower than the output layer.

### 4.2 Why η·T alone cannot label the regime

The paper's condition η·T ≪ 1 carries an implicit O(1) Jacobian scale; the graph-dependent form is η·T·λ ≪ 1 on the excited modes. A label computed from η and T alone reads the slowest mode, so it can certify "not backprop" but never "backprop", and against the sweep it mislabels by about 10×: (1e-2, 5) at 35.0%, already leaving the backprop value, read backprop-like, while (1e-2, 16) on the 31% plateau read partially relaxed. `Regime` therefore takes a measured spectrum.

### 4.3 f̄ and f_max: what each predicts

f̄ = Σ_{θ_k > 0} w_k·f(θ_k) / Σ_{θ_k > 0} w_k is the equilibrium criterion: the energy is at its minimum only when the modes carrying the gradient have relaxed, and the slowest of them sets the pace. A mode with θ_k ≤ 0 has no minimum to relax toward, so it is left out of f̄ and judged by its weight and growth instead. f_max = f(λ_max) is the relaxed fraction of the fastest excited mode.

On the ResNet-18 the two disagree at init: f̄ = 0.010 against f_max = 0.080 at the defaults, so the weight-averaged eigenvalue is about 2, close to the unit-precision floor at 1 and far from λ_max = 16.45. The 2-epoch accuracy follows f_max, not f̄. At η = 0.01, T = 3 to 5 the accuracy has left the backprop value while f̄ still reads backprop-like; at η = 0.03, T = 5 to 64 the accuracy sits on the PC plateau while f̄ reads partially relaxed; a one-eigenvalue fit of the η ≤ 0.01 cells (accuracy mapped linearly from 38.8% → 0 to 31.0% → 1, collapsed cells excluded) lands at λ_eff = 12.0, near λ_max. Mechanism: the bulk of the gradient weight sits on modes with λ near 1 or 2, whose relaxation moves each hidden error's component along such a mode from −η·T·g0 toward −g0/λ, a change of magnitude with the direction preserved, which Adam's per-parameter normalization absorbs; the top modes couple the hidden errors to the output prediction, and their relaxation re-weights the output error by S⁻¹, which Adam cannot undo. During training, f̄ crossed 0.1 two to three epochs before each collapse's reversal flag, while f_max had crossed it epochs earlier with accuracy still improving (report Section 5.9).

Disposition: the `Regime` band is on f̄, the equilibrium criterion and the tracked early warning; f_max is reported beside it as the predictor of the accuracy penalty at init. Open: a two-sided band, backprop-like on f_max < 0.1 and near equilibrium on f̄ > 0.9 (Section 7).

### 4.4 `Regime`

`EPCInference.regime(spectrum) -> Regime`, a NamedTuple:

| Field | Meaning |
|---|---|
| `eta`, `steps` | η and T of the solver |
| `eta_lambda_max`, `eta_T_lambda_max` | η·λ_max and η·T·λ_max |
| `unstable` | η·λ_max > 2 |
| `output_gradient_reverses` | (1 − ηλ_max)^T < −1/(λ_max − 1); always False at even T |
| `f_max`, `f_weighted` | f(λ_max) and f̄ |
| `band` | on f̄: "backprop-like" below 0.1, "near PC equilibrium" above 0.9, "partially relaxed" between |
| `negative_weight`, `growth_min` | the fraction of ‖g0‖² on negative-curvature modes, and (1 + η·max(0, −λ_min))^T, the growth of the most negative mode over T steps |
| `lambda_max`, `lambda_min` | the extremes |

`str(regime)` is a one-line label with precedence: `unstable`; then `growth_min > 1.1` ("indefinite: negative curvature carrying X% of the gradient grows Y× over T steps"); then the band with f̄, f_max, and the reversal note. Indefiniteness is judged by weight and growth rather than by a boolean because one ePC step from ε = 0 is exactly −η·g0 whatever the curvature sign; the sign enters at the second step, and a negative mode grows by (1 + η|λ_min|)^T over T steps, 1.002 at the defaults on the ResNet-18. A boolean that outranked the band would tell a user whose setting is exact backprop that it is broken.

### 4.5 `RegimeProbe`

`fabricpc/training/regime_probe.py`, exported from `fabricpc.training`: `RegimeProbe(structure, probe_clamps=None, *, every, inference=None, iters=30, key, csv_path=None)`. `on_iter(ctx)` runs every `every` updates: it initializes a graph state at `ctx.params`, runs the compiled spectrum estimator, and records per-edge Frobenius weight norms, the training energy, and `inference.regime(spectrum)` when `inference` is an `EPCInference`. `probe_clamps` are fixed clamps built once by the caller (the demo uses a 64-sample test batch); `None` measures on the training batch. `on_epoch(ctx, accuracy=None)` records the epoch row. Readouts: `first_reversal()`, `first_crossing()`, `first_chance(chance, margin)`, `growth_phases()` (the per-epoch λ_max maximum and its ratio to the previous epoch), `summary()`, `write_csv()`; `read_regime_csv` reads the file back. The CSV carries `trainer, eta_infer, infer_steps, every, probe_batch` first, then the spectrum, the regime flags, energy, accuracy, and one `wnorm:<edge_key>` column per weight, so readers take η, T, and the trainer from the columns. It works under `algorithm="backprop"` as the control (spectrum and norms; regime columns empty).

The probe runs on the `IterContext` iteration callback of release 0.5.2 (`docs/dev_plans_archive/iter_callback_context.md`), which gives `iter_callback(ctx)` the parameters, optimizer state, batch, and state. The training step donates the parameter buffers, so `ctx.params` is valid only during the callback; the probe reads it there and stores floats. Supplying `iter_callback` forces a device sync on every batch, not only probed ones. A per-epoch callback would not do: in the defaults' collapse λ_max grew 2.75× within 50 updates.

### 4.6 What was measured

All measurements are on the muPC ResNet-18 CIFAR-10 demo (gelu, no normalization layers, weight decay 1e-2, AdamW), one RTX 3090, with the spectrum probed on a 64-sample test batch; the report holds the tables.

| Conclusion | Report |
|---|---|
| λ_max = 16.4 at init, so η < 0.12; H_ε is indefinite from init with 1.2% of the gradient weight on negative curvature | Section 5.8 |
| The defaults (1e-3, 5) are backprop-like at init: f̄ = 0.010, f_max = 0.080, η·T·λ_max = 0.08 | Section 5.8 |
| Over 2 epochs accuracy falls from ePC's small-η·T value (38.8%) to the PC plateau (31%) following f_max; sPC-120 sits between (34.6%) because 120 state-based steps do not reach equilibrium | Section 5.8, Appendix A |
| Over 100 epochs (1e-3, 1) and (1e-3, 2) trained through (76.7% and 75.8%, against the backprop trainer's 77.1%); the defaults collapsed to chance by epoch 20; 1e-2 collapsed by epoch 10 at every step count | Section 5.8 |
| Four 30-epoch control runs on the normalized trainer, seed 42, probed every 50 updates: backprop, ePC (1e-3, 1), the defaults (1e-3, 5), and ePC (1e-2, 1). From λ_max = 16.4 at init, the per-epoch maximum was 22 at epoch 1 in all four; it drifted about 1.01× per epoch to 24 under backprop and 29 under (1e-3, 1), both reaching 68 to 69%; at the defaults it ran 27 (epoch 8) → 31 000 (epoch 15) and the run collapsed at epoch 15; at (1e-2, 1) it reached 6 132 at epoch 7 and the run collapsed there. Every convolution weight's Frobenius norm fell through epoch 14 in all four runs and rose only in a collapse epoch. The growth is ePC's relaxation, not the weight scale | Section 5.9 |
| The reversal flag fired one probe (50 updates) before the stability crossing in both collapsing cells; Lanczos reported λ_max = 6 132 and λ_min = −354 where power iteration had returned −9 399 | Section 5.9 |
| f̄ crossed 0.1 two to three epochs before each reversal; f_max crossed it earlier with accuracy still improving | Section 5.9 |

The 2-epoch sweep and the 100-epoch runs predate release 0.5.1 and used batch-summed PC weight gradients. The inference energy is untouched by the normalization, so H_ε and the regime at init are the same under both trainers; the training trajectories are not, and the control runs are the first tracking data from the normalized trainer, on which the backprop and ePC cells share one learning-rate scale.

Reading rule and its outcome. The rule, fixed before the runs: comparable growth of λ_max and the weight norms under backprop and (1e-3, 1) means a weight-scale effect of this parameterization, and the remedy is weight-norm control or a rate that follows λ_max; growth only in the collapsing cells means ePC's relaxation feeds it, and the remedy is solver-side. The second case held. Goemaere et al. trained ResNet-18 at the same η = 1e-3, T = 5 for 50 epochs without instability, with batch normalization after every convolution, ReLU, weight decay ≤ 1e-3, and a standard parameterization (report Section 5.11).

Why the collapsing default was kept. Any fixed η collapses once λ_max grows past 2/η, so a new constant only moves the horizon: (1e-3, 2) survived 100 epochs and (1e-4, 5) was never run that long. The defaults are backprop-like at init on the demo graph, every touch point (class docstring, guide, FAQ, demo) states the regime and how to measure it, and the follow-up is a rate that follows λ_max (Section 7).

## 5. Tests: what they pin

- `tests/test_topological_schedule.py`: DAG order equals the legacy Kahn order across insertion permutations with and without `unroll`; `GraphCycleError` on cycles without `unroll`; the `x→a⇄b→y` schedules at U = 1 and 2; bool and `unroll < 1` rejection; SCC topologies beyond two nodes (3-node, overlapping, disjoint, multi-entry, entry-less, clamped member).
- `tests/test_state_initializer.py`: feedforward through cycles matches a hand-computed propagation at U = 2 (not a replay of pass 2, so a wrong schedule cannot pass both sides); U = 2 differs from U = 1; every source leaves every initializer with z_mu == z_latent and error == 0, asserted against a garbage-writing initializer so the check is falsifiable.
- `tests/test_node_contract.py`: every registered node class (walked recursively, so `LinearExplicitGrad` and the pooling subclasses are included) resolves the five templates to `NodeBase`'s; the source guard; `energy(..., aux=None)` does not raise for any energy-overriding node; the Hopfield attractor value against a hand computation; a `predict` returning a bare array fails at trace time.
- `tests/test_inference_epc.py`: ε = 0 ⇔ feedforward init; the gradient against a closed form on a 2-layer chain and against a hand-rolled `jax.grad`; energy descent; sPC equivalence on a convex DAG with an unclamped prior (per-node state, stationarity, and `compute_local_weight_gradients` agree) and on a tanh chain with a cross-entropy output (slow); the muPC fixed-point divergence (slow); `begin_segment` preserves a distribution-initialized state; template branch coverage (clamped and unclamped × source and internal, Gaussian and Hopfield readouts, an int-token embedding graph); cyclic smoke with the warm-start assertion (two steps at U = 1 differ from one step at U = 2); `TestBackpropCorrespondence` (Section 2.6); `TestRegime` (bands from constructed spectra with the same extremes and different weights; reversal at T = 1 and 5, none at T = 2; `growth_min` precedence).
- `tests/test_inference_schedule.py`: `segments()` flattening, and a nested schedule executed bit-identical to its flattened form; a single-solver schedule equals the plain solver; ePC then sPC equals manual sequential calls; states cross the boundary bit-identical; runs inside `jax.jit(train_step)`; strict energy descent across the handoff; tracking parity under a schedule; the raising stubs raise.
- `tests/test_linear_pc_oracle.py`: the oracle against a hand-built scalar chain (x = 1, W1 = 2, W2 = 3, y = 1 gives z_h* = 0.5, E* = 1.25, S = [[10]]), Theorem 1 against least squares, the precision-weighted pull-back, the explicit H_ε form; both solvers on twelve graphs; stability brackets on the plain and muPC chains; the Hessian-vector product; Lanczos extremes and f̄ against the oracle in both precisions.
- `tests/test_epsilon_spectrum.py`: an explicit matrix with eigenvalues {−50, −1, 1, 3, 10} and a start vector of known components; breakdown at step 2 inside a 2-dimensional invariant subspace; a zero start vector; the eps weight floor hides the unexcited floor on a rank-10 fixture; tanh and gelu MLP fixtures against `jax.hessian`.
- `tests/test_regime_probe.py`: a tanh MLP trained two epochs with `every=2` under both trainers; the CSV round-trips through `read_regime_csv`.
- `tests/test_fabricpc.py` and `tests/test_trainer.py`: the sPC side of the readout fix (error kept, `latent_grad` untouched, z_latent converges to z_mu, a Hopfield readout keeps its attractor energy; eval energy finite and signed with `target_energy ≥ 0`). The full suite passed with no expectation edits at the contract-split commit, which pins sPC bit-identity.

## 6. Alternatives considered

| Decision | Chosen | Rejected and why |
|---|---|---|
| ε storage | `NodeState.error` | an extra fori_loop-carry dict (invisible to tracking, handoff, and the weight path); a new `epsilon` field (schema churn, redundant with `error`) |
| ε gradient | one global `value_and_grad` over `{node: ε}` | per-node grads stitched through the schedule (re-derives reverse AD, wrong on repeated visits); grad w.r.t. the whole `GraphState` (differentiates int latents and clamps) |
| Derive rule | one rule, the clamp decides the free side, all unclamped nodes ε-relaxed | a five-way partition by degree and clamp role (its latent-relaxed source handling froze top-down priors) |
| Node contract | predict / energy with base-owned templates | a second `forward()` call in the derive pass (two different z_mu when `predict` reads state; correctness would rest on XLA CSE); `energy_functional` directly (drops in-forward terms); a resolve-latent callable injected into `forward` (boilerplate retained; a forgetful body breaks only ePC); a staged energy closure; an overridable pair, or a pair owned by `EnergyFunctional` today (a non-volume-preserving pair breaks the equivalence unless the energy moves with it; `EnergyFunctional` is the future home if precision-weighted ε is wanted, not built) |
| Readout | ordinary ε-relaxation, energy as assigned | forcing error = 0 and energy = 0 (hard-codes the Gaussian assumption; Hopfield readouts cannot settle); a skip-the-derive special case (negligible saving) |
| Segment handoff | `begin_segment` ε resync | identity pass-through (the plan's first choice; overwrote distribution-initialized latents and dropped sPC's final update) |
| Dispatch | classmethods plus instance `run_inference` | threading `cls` parameters (still mis-passable); instance methods everywhere |
| Schedule location | new `schedule` field beside `node_order` | replacing `node_order` (every one-visit consumer must dedup); stashing in `structure.config` |
| Unroll algorithm | Tarjan SCC, Kahn on the condensation, entry-first BFS | repeating the full sweep U times (multiplies cost on the acyclic majority); feedback-arc-set removal (back edges never carry information) |
| Schedule API and ownership | `_topological_sort(unroll=)` in place; graph-owned U | a scheduler class hierarchy carrying one integer; solver-owned U (initializer and solver could disagree) |
| Initializer fix | one post-pass in `initialize_graph_state` | repeating the copy in each initializer; fixing `FeedforwardStateInit` alone |
| `InferenceSchedule` type | `InferenceBase` subclass with `segments()` | a separate protocol (two type surfaces); `graph(inference=[...])` (composition in the wrong layer) |
| Equal-wall-clock comparison | the T sweep as the time axis | per-epoch checkpoint curves (ties the axis to checkpoint cadence); a stopping rule inside `train_pcn`; assumed step-count parity; a bespoke trial loop |
| Spectrum estimator | Lanczos from g0 with Ritz weights | shifted power iteration (one number, slow on compact spectra); Lanczos extremes with a band on λ_min (mislabels nonlinear graphs); full reorthogonalization (k vectors for ghosts that move nothing) |
| Regime verdict | `Regime` NamedTuple, band on f̄, indefiniteness by weight and growth | a string label (substring parsing, no flags); a boolean `indefinite` outranking the band; bands on f_max alone (reads the fastest mode, not the equilibrium criterion; the two-sided variant stays open, Section 7); bands on f(λ_min) |
| Probe placement | `RegimeProbe` on `train`'s iteration callback | `epoch_callback` only (too coarse); a trainer `probe_every` flag (a second per-batch mechanism); the script's custom loop (not reusable); η and T parsed from filenames |
| Control runs | four cells including a re-run of the defaults | three runs without the defaults (leaves the headline series on the old estimator and trainer) |

## 7. Open questions and follow-ups

- Band definition: a two-sided band, backprop-like on f_max < 0.1 and near equilibrium on f̄ > 0.9, would match the 2-epoch accuracy on this graph on the backprop side while keeping the energy criterion on the other. Undecided; the measurement is one batch at one seed.
- Adaptive rate: the control runs assign the remedy to the solver, a rate that follows λ_max or η·T kept in the backprop regime. Not built. The mechanism by which relaxation beyond the backprop-like regime raises λ_max is not derived; the control runs give the attribution only.
- Cyclic ePC has no oracle, and the spectrum estimator is unvalidated on unrolled graphs.
- Whether an ePC warm start helps a given architecture, and whether a cycle is better refined by sPC, unrolled into ePC, or both, are experimental questions the benchmark script can answer; only the resnet18 DAG has been measured.
- PR description: cyclic graphs previously received partial-order feedforward init (cycle members skipped), so training curves on cyclic graphs shift even where tests hold; `examples/mnist_cyclic_graph.py` is the visible case.

## 8. Record

Six cycles, in order. Each states what was implemented, what the review found, what changed and why, and what the cycle bought.

**Cycle 1, 2026-08-14 to 08-22: plan, implementation, first review.** Implemented: the six components in sequence, the suite green at each gate: classmethod dispatch and segment hooks; the unrolled schedule and `GraphStructure.schedule`; the contract split with sPC bit-identity (no test-expectation edits); `EPCInference`; `InferenceSchedule` and segment-aware tracking; the resnet18 refactor and the compare script. Review found (a full branch-diff review on 2026-08-22; its file was never committed): the identity segment handoff overwrote distribution-initialized internal latents and dropped an sPC segment's final update; a `predict` that reads z_latent makes the two solvers minimize different energies; the η default rested on an unsupported overshoot claim; the sweep's derived metrics were tested with hand-rolled t-tests; `unroll=True` built silently at degree 1; source error was not zeroed at init; `evaluate_transformer` read z_latent; `StorkeyHopfield.energy` had no aux=None path; a list of test gaps. Changed and why: `begin_segment` resyncs ε at the carried latents, one forward pass per boundary for exact preservation of any incoming state; the z_latent read prohibition and the aux rules entered the contract; η = 1e-2 became the default as a margin below the fastest measured rate (104, 11, 4, and 1 ε updates to reach sPC's final 120-step energy at η = 1e-3, 1e-2, 3e-2, 0.1); descriptive statistics replaced the t-tests, since interpolation and min-over-arms selections are not planned contrasts (one of two places where the implementation deviated from the review's suggested remedy); bool rejection; the source post-pass; z_mu reads at every eval site; the base-term fallback on aux=None because sources have empty params, so the review's suggested recomputation of `(W, strength)` was impossible (the other deviation); the test-gap list became the audit and branch-coverage tests of Section 5. Value: two correctness fixes no test had caught, the handoff and the z_latent read, and a smaller review surface, since the templates are not override points.

**Cycle 2, 2026-09-01: the 100-epoch runs.** Implemented: six 100-epoch cells at η ∈ {1e-3, 1e-2} × T ∈ {1, 2, 5}, and the convergence-mode probes. Found: every 1e-2 cell at chance by epoch 10; the (1e-3, 5) default at chance by epoch 20; (1e-3, 1) and (1e-3, 2) trained through. The convergence figure showed a phantom step-0 energy because eager init and jitted tracking selected different cuDNN algorithms. Changed and why: the default returned to 1e-3, since a margin default did not hold over the training horizon; `make_tracked_probe` compiles init and tracking into one program. Value: the collapse became the question the next cycles answered, and the convergence figure became trustworthy.

**Cycle 3, 2026-09-03 to 09-04: second review round, first iteration.** Reviewer requests after the convergence figure: an exact oracle both solvers are checked against, itself checked against hand computation and Innocenti et al. 2024; the analytical questions (why sPC struggles with depth, what sets the equilibrium energy spacing, the largest stable η in deep networks, whether the collapses occur under muPC, how many steps sPC needs); whether the `infer_steps=1` caution said the right thing and where users would see it. User decisions: keep the defaults and document the regime at every touch point; no `warnings.warn`, the demos print the regime; the oracle in a shared module; a graph-aware label taking λ_max; test the collapse hypothesis on the GPU demo, not a CPU MLP; build the weight-gradient parity test now. Implemented: `linear_pc_oracle`, `error_energy`, a regime label on power iteration, `scripts/epc_analysis.py`, the guide's backprop-regime paragraph. The oracle's design review corrected three claims (Section 2.6), and the collapses were confirmed to occur under muPC, the demo's parameterization. The caution was found to key on the wrong criterion: T = 5 at η = 1e-3 is equally backprop, and the tuning table recommended 1 to 5 two paragraphs below it. Spin-off PR: the parity test compared batch-summed PC weight gradients against a batch-mean backprop objective, so the two trainers did not share a learning-rate scale. Mean-gradient normalization shipped as release 0.5.1 on 2026-09-07 (`docs/dev_plans_archive/mean_gradient_normalization.md`): the trainer divides the PC weight gradients and the backprop objective by one global prediction count, so one learning rate, clipping threshold, or Adam epsilon means the same under both trainers and across batch sizes and sequence lengths, and the parity test runs through `pc_weight_gradients`. Value: the solvers are pinned to an independent closed form including their stability bounds, so two wrong solvers agreeing is ruled out; and the backprop control run of cycle 5 is comparable with the ePC cells, which the reading rule requires.

**Cycle 4, 2026-09-08: second review round, the critical review.** Verdict: correctness verification strong; tuning diagnostics weak. Found: (1) the estimator returned the largest-magnitude eigenvalue, so the collapsing (1e-2, 1) run's −9 399 read "backprop-like" and the crossing detector never fired; (2) the equilibrium band read the fastest mode, and the printed condition multiplied by λ_max and divided by λ_min; (3) the T = 1 danger threshold was too lenient, since the output gradient reverses at η(λ_max − 1) > 1, before the bound; (4) the tracking tool re-implemented the training loop because `iter_callback(epoch_idx, batch_idx, metrics)` could not see the parameters and the epoch callback was too coarse; (5) both tracked cells collapsed, so no control separated cause from ordering, and the paper's contrary ResNet-18 result was undocumented. Smaller: a "threefold per epoch" growth claim where 1.13× was measured, the Adam caveat's limit, the r S⁻¹ damping unshown, the sPC+muPC equilibrium test unable to pin the gradient scale, the oracle's cyclic and per-sample limits undocumented, the λ_eff fit called agreement. Spin-off PR: `train`'s iteration callback gained an `IterContext` carrying the parameters, optimizer state, batch, and state, shipped as release 0.5.2 on 2026-09-08 (`docs/dev_plans_archive/iter_callback_context.md`) with every three-argument caller migrated and the dashboarding callbacks moved onto `train`. Value: the first tool would have told a user that a collapsing run was safe; the new hook lets the probe run inside `train` on any graph instead of a copied training loop.

**Cycle 5, 2026-09-08: replacement design, implementation, control runs.** Design: Lanczos from g0 with Ritz weights; `Regime`; `RegimeProbe` on the 0.5.2 `IterContext`, after rebasing the branch onto that release; four control runs with a reading rule fixed in advance. The design's first-draft review ran two CPU experiments (report Appendix B) that changed it: f̄ over the gradient-weighted modes instead of a band on λ_min (943 of 1024 modes excited on a gelu MLP, λ_min below the floor); a relative breakdown guard instead of an exact-zero test; indefiniteness by weight and growth instead of a boolean; the fourth control run, a re-run of the defaults, kept so the headline series comes from the new estimator and the normalized trainer. Implemented: five deliverables in one commit each, with the power-iteration estimator, the string label, and the script's tracking loop deleted and every caller migrated. Findings, each as expected, measured, changed: (1) the guard alone excludes the unexcited floor; on a depth-5 chain it did not, β_10 stayed above the threshold on accumulated rounding; the eps weight floor on the extremes. (2) The excited band is compact on the ResNet-18; f̄ = 0.010 against f_max = 0.080 and the accuracy follows f_max; f_max reported beside the band, the band kept on f̄ as the equilibrium criterion and early warning. (3) The reading rule; λ_max ran away only in the ePC cells with larger η·T while weight norms fell in all four, and the reversal flag fired one probe before the crossing in both collapsing cells; in (1e-2, 1) the design had predicted the flag at update 1100 from the old trainer's trajectory, but on the normalized trainer η(λ_max − 1) was 0.85 there, so the flag fired at the next probe, 1150, with the crossing at 1200; the remedy is solver-side. The trackers were reconciled with 0.5.2. The sweep, the 100-epoch runs, and the two older CSVs predate the 0.5.1 normalization and were accepted as recorded; the inference energy is untouched by the normalization, so H_ε and the regime at init are the same under both trainers; the training trajectories are not, and the control runs are the first tracking data from the normalized trainer. Value: a sign-correct estimator, a band on the right quantity, a probe that runs on any graph, and a measured cause for the collapse.

**Cycle 6, 2026-09-09: consolidation.** Numbers moved to the report, whose Appendices A and B receive the sweep and evidence tables; the changelog, the guide, and the demo docstring reduced to conclusion plus pointer; this record rewritten as a final design document; the four intermediate development documents deleted.
