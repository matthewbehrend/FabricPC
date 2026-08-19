# ePC inference solver, composable inference scheduling, and the graph topology scheduler

## Context

FabricPC's two inference solvers (`InferenceSGD`, `InferenceSGDNormClip`) implement state-based predictive coding (sPC): latent states relax by local gradient descent, so the output-loss signal attenuates by the state learning rate per layer per step and deep graphs need 100s of inference steps (resnet18 demo: 120 steps, 476 s/epoch). The ePC paper (Goemaere et al., arXiv 2505.20137, `docs/dev_plans/2505.20137v5.pdf`) reparameterizes PC over prediction errors: one reverse-mode AD pass through the whole network delivers the loss signal to every layer unattenuated, reaching the same equilibrium in ~several steps. This plan adds ePC as an `InferenceBase` subclass, a composable inference scheduler (intent: a few ePC steps to near-equilibrium, then sPC refinement on the true arbitrary-graph energy), a graph-topology-scheduler abstraction so ePC also accepts cyclic/self-recurrent graphs by unrolling, and an ePC-vs-sPC benchmark on the resnet18 demo. sPC remains the general solver for arbitrary graphs; ePC is the efficient solver for DAG (or unrolled-cyclic) representations.

Along the way it fixes two latent defects the design exposed: (1) `InferenceBase` template methods re-resolve their own class via `type(structure.config["inference"])`, which breaks any composition; (2) on cyclic graphs `_topological_sort` silently returns a partial order — on `x→a⇄b→y` the order is just `("x",)`, so feedforward init leaves cycle members *and everything downstream* at random init, and muPC silently attaches no scaling to them.

## The composed solver in one picture

ePC solves the DAG; state-based PC (sPC, today's settling path) refines around cycles; a cycle can instead be unrolled into ePC's own graph. Both cycle paths are built this quarter. Symbols: `T` = settling ticks per update in sPC alone, `T1` = ePC steps, `T2` = sPC refinement steps, `U` = unrolled cycle traversals, `(H)` = a Hopfield (recurrent) node.

**1 — Composable inference schedule.** Solvers are schedule entries composed per weight update, not trainer modes:
While sPC optimizes the exact graph as-is, ePC is a warm-started approximation of the energy minimization on cyclic graphs. Even with unrolling of the cycles, ePC is an approximation of the graph to the extent of the unrolling. For arbitrary graphs, is warm start with ePC helpful, and at what depth? Experiments are needed to form user guidance on choice of inference suited to the graph architecture.
```
schedule = [ ePC(T1), sPC(T2), WeightUpdate ]

clamp ──► ePC: T1 steps ──► sPC: T2 steps ──────────► weight update ──► next batch
          solves the DAG,   minimizes full-graph
          back-edges        energy, back-edges in;
          excluded or       warm-started from ePC's solution
          unrolled
```

**2 — Where the error lives.** From a feedforward start, sPC moves error one hop per tick with per-hop damping: after `T` ticks the error profile decays exponentially from the output clamp and deep layers have seen almost nothing. ePC pushes the output error through the full depth in every step, so ~5 steps leave signal at every layer. What remains is the residual the back-edges introduce; it enters at the cycle, diffuses a few local hops per sPC tick, and the Hopfield node falls onto its fixed-point attractor within a few iterations:

```
error magnitude by depth — deep chain with one Hopfield cycle mid-network

sPC alone (T ticks,          ePC (~5 steps):            + sPC refinement (~5 ticks):
feedforward start):          full depth reached         residual local to the cycle

out ████████                 out ██████                 out ·
    ████                         ██████                     ·
    ██                           ██████                     ▪
    █                            ██████                  ┌─►(H)──┐  error hops the cycle;
    ▏                            ██████                  └───▪───┘  (H) settles on its attractor
    ▏                            ██████                     ▪
in  ▏                        in  ██████                 in  ·
```

**3 — The alternative cycle path: unroll into ePC's DAG.** `U` traversals of the cycle become `U` copies of the cyclic subgraph inside one differentiable program — no separate solver, no stopping rule, no handoff; `U` is fixed at graph-build time:

```
cyclic graph                     unrolled into ePC's DAG (U = 2)

a ──► b ──► c ──► d              a ──► b₀ ──► c₀ ──► b₁ ──► c₁ ──► d
      ▲     │
      └─────┘                    the back-edge c→b becomes the feedforward edge c₀→b₁
```

Implementation is PC-faithful:
- Errors are tied between unrolled cycles by construction: the relaxed pytree in `EPCInference.forward_value_and_grad` keys ε by node name (one leaf per node), `GraphState` carries one `NodeState` per node, and `forward_from_error` re-injects the carried `state.error` on every visit. The numbered subscripts illustrate repeated forward passes through a node (e.g. b0 and b1) while the same error_b term is injected to each.
- Because each visit overwrites the node's single `NodeState`, node *i*'s energy term enters the total E once, evaluated at the final visit's derived state. E is therefore the cyclic graph's own Σ_i E_i evaluated at the U-traversal derived point — not a sum over unroll copies.

Which inference schedule for cyclic graphs is settled by cycle depth in a particular topology and comparative refine-vs-unroll measurements. There's also a middle ground of unrolled ePC + sPC that is also possible with the composable solvers. Depth of the cycles will be meaningful to both inference solvers as it impacts sPC signal decay and ePC unrolling cost. 


## Formulation

Symbols (one meaning throughout): for node *i*, **z_mu_i** is the node's prediction of its own latent computed by `forward()` from its in-edge sources; **z_latent_i** is the latent state; **ε_i** is the prediction error, stored in `NodeState.error`; **E** is the total energy Σ over nodes with `in_degree > 0` of the node's energy functional `energy(z_latent_i, z_mu_i)`; **η** (`eta_infer`) is the inference rate; **T** (`infer_steps`) the step count.

sPC relaxes z_latent with ε derived (`error = z_latent − z_mu`, recomputed in every node `forward()`). ePC inverts the parameterization: ε is the first-class relaxed variable and z_latent is derived by a forward pass in schedule order, `z_latent_i := z_mu_i + ε_i`. Because z_mu_i depends on upstream z_latents, every node's energy — including the clamped output node's `energy(y_clamp, z_mu)`, which is the paper's output loss — depends on all upstream ε. One `jax.value_and_grad` over the ε pytree through the whole derived forward gives exact ∇_ε E; ε steps down that gradient. The ε↔z_latent map is a bijection with unit-determinant triangular Jacobian (paper Appendix C): identical energies, identical equilibria, and the final derived state feeds the existing local weight-gradient path unchanged.

Node partition in ePC (computed at trace time from static structure + clamp keys):
- **clamped, in_degree > 0**: z_latent stays the clamp; `forward()` refreshes z_mu/error/energy; `energy(clamp, z_mu)` is the output loss; gradient reaches upstream ε through z_mu.
- **in_degree == 0, clamped** (data/token sources): untouched constants, never differentiated (int dtypes stay out of the AD pytree).
- **in_degree == 0, unclamped** (top-down priors): Computes z_latent = z_mu + error for this node like other nodes. Initialize z_latent from distribution and then z_mu ← z_latent. This provides the same pattern of initial error=0 in feed-forward initializtion without forcing z_latent=0. Unclamped terminal input nodes are typically intedned to be initialized from a distribution rather than inputting zeros to the network.
- **out_degree == 0, unclamped** (eval readout): ε reamins a 0 in absence of a gradient, z_latent ← z_mu, Forcing error=0 and energy=0 is too strong an assumption (breaks hopfield output nodes) - relax it in `nodes/base.py:514-529` where it originated.
- **all others**: ε-relaxed.

At ε = 0 the derived states equal `FeedforwardStateInit`'s output, so the paper's zero-init is the existing default initializer for free.

muPC: `scale_inputs` applies inside the differentiated forward exactly where the sPC loop applies it, and the global AD supplies the chain-rule factors automatically. The gradient preconditioners (`jacobian_gain` in `topdown_grad_scale`, `self_grad_scale`) are per-hop conditioners for sPC; they are deliberately not replicated in ePC which is a global backward pass (decided; documented on `EPCInference`). `scale_weight_grads` at learning time is untouched.

Cyclic graphs: the forward-from-errors pass iterates `structure.schedule` (below). Repeated visits of a cycle member recompute z_mu from the latest source latents and re-derive z_latent with the *same* ε — ePC on the unrolled network with tied errors; the single AD pass differentiates through all repeats. StorkeyHopfield needs no special handling: its z_mu comes purely from the input probe and its self-recurrence enters only through the Hopfield energy term on its own z_latent, which the derived forward evaluates at z_mu + ε (verified against `nodes/storkey_hopfield.py:333-419`).

## Component 1 — Graph topology scheduler

New module `fabricpc/graph_assembly/scheduling.py` (name avoids colliding with `fabricpc/core/topology.py`):

- `TopologySchedulerBase(ABC)`: house style — `__init__(**config)` storing `MappingProxyType`, abstract `@staticmethod compute_schedule(nodes, edges, config) -> Tuple[str, ...]`. A schedule is the full node visit sequence; cycle members may repeat; every node appears at least once.
- `first_occurrence_order(schedule) -> Tuple[str, ...]`: dedup keeping first occurrences (the unique node order).
- `GraphCycleError(ValueError)`.
- `DAGScheduler`: exact port of `_topological_sort`'s Kahn/BFS loop (same queue seeding from dict order, same successor order → bit-identical order on every existing DAG). On cycles it raises `GraphCycleError` naming the unordered nodes and directing the user to select a cycle-capable scheduler explicitly — even for a degenerate single-visit schedule (decided). `_topological_sort` (`graph_construction.py:65-105`) is deleted; no other importer exists.
- `UnrolledCycleScheduler(num_unrolls: int = 3)`, validated ≥ 1 (`num_unrolls=1` = each cycle member visited once — the explicit degenerate choice): iterative Tarjan SCC → Kahn on the condensation DAG (same seeding/order rules, so on a DAG the output equals `DAGScheduler`'s exactly) → each nontrivial SCC's members emitted `num_unrolls` times, intra-SCC order = BFS from entry nodes (members with an in-edge from outside the SCC, dict order; first member if none). `x→a⇄b→y`, K=2 → `("x","a","b","a","b","y")`. Self-edge rejection at build (`graph_construction.py:152-153`) stays.

Integration:
- `GraphStructure` (`core/types.py:149-172`) gains a `schedule: Tuple[str, ...]` field after `node_order`, with the build-enforced invariant `node_order == first_occurrence_order(schedule)` (equal on DAGs). Pytree static aux (`types.py:213-217`) updated. Construction sites: `graph_construction.py` + the unflatten lambda only (grep-verified).
- `graph(..., topology_scheduler: Optional[TopologySchedulerBase] = None)` → default `DAGScheduler()`. Compute `schedule = type(scheduler).compute_schedule(finalized_nodes, edge_infos, scheduler.config)`; validate coverage (missing/unknown nodes raise); `node_order = first_occurrence_order(schedule)`; store the scheduler in `gs_config` beside `inference`.
- Consumer migrations (no fallbacks): `FeedforwardStateInit` pass 2 (`state_initializer.py:269`) iterates `structure.schedule` — the loop body is already revisit-correct, so cyclic graphs gain true feedforward init through cycles (bundled defect fix). muPC (`core/mupc.py`) stays on the unique `node_order`: depth L models one merge-sum energy term per merge node regardless of visit count, and back edges contribute depth 0 naturally (`skip_counts.get(source, 0)` returns 0 for later-ordered sources); add a duplicate-entry raise to `compute_mupc_scalings`/`_count_skip_connections_depth` as hardening. Cyclic call sites gain the explicit scheduler: `tests/test_inference_order.py:105-126`, `examples/mnist_cyclic_graph.py:89-108`, `docs/user_guides/04_building_models.md:456-472`.

## Component 2 — InferenceBase dispatch refactor

`fabricpc/core/inference.py`: template methods become `@classmethod` dispatching on `cls`; `run_inference` becomes an instance method (it needs `self.config`). No dual-mode paths; all callers migrate.

| Method | Change |
|---|---|
| `inference_step` (:86) | `@classmethod`; body uses `cls.zero_grads / cls.forward_value_and_grad / cls.update_latents` (drops the `structure.config` re-resolution at :100-101) |
| `zero_grads` (:114) | unchanged static |
| `forward_value_and_grad` (:134) | `@classmethod` (body unchanged; subclass overrides need `cls`) |
| `update_latents` (:201) | `@classmethod`; uses `cls.compute_new_latent` (drops :212-213) |
| `compute_new_latent` (:229) | unchanged abstract static |
| `run_inference` (:245) | instance method: `cls = type(self)`; `state = cls.begin_segment(...)`; `lax.fori_loop(0, self.config["infer_steps"], ...)` stepping `cls.inference_step(..., self.config)`; `return cls.finalize_state(...)` |
| new `begin_segment` / `finalize_state` | `@classmethod`, default identity — segment-boundary hooks (ePC uses them; sPC untouched) |
| new `segments()` | instance method, default `((self, int(self.config["infer_steps"])),)` |

Call-site migrations (complete, grep-verified): module-level `run_inference` (:364-389) delegates to `structure.config["inference"].run_inference(...)` — its own signature is unchanged, so `train.py` (:152, :516, :598, :744) and `train_autoregressive.py` (:208, :435, :627) need no change. Tracking (`utils/dashboarding/inference_tracking.py` :50-59, :143-153) migrates to segment iteration (Component 5). Tests calling the old static form: `tests/test_fabricpc.py` (:185, :190, :227, :428), `tests/test_ndim_shapes.py` (:64, :105, :152), `tests/test_auto_node_grad.py` (:315, :318). Adjacent audit: `StateInitBase` dispatch (`state_initializer.py:384`) dispatches on the object it was handed, not via structure — not the same defect, no change; no other `type(structure.config[...])` self-resolution exists.

## Component 3 — NodeBase.forward_from_error

New node-level override point (decided: new sibling method, not a mode on `forward_and_latent_grads`) in `fabricpc/nodes/base.py` beside `forward_and_latent_grads`, dispatched as `node_info.node_class.forward_from_error(...)`:

```python
@staticmethod
def forward_from_error(params, inputs, state, node_info, is_clamped) -> NodeState:
    # ePC state derivation: state.error is the relaxed epsilon; derive
    # z_latent := z_mu + error and populate z_mu/error/energy consistently.
    # Runs inside EPCInference's global jax.grad — must stay differentiable
    # w.r.t. inputs and state.error. Never writes latent_grad.
    node_class = node_info.node_class
    if node_info.in_degree == 0:
        new_state = state._replace(z_mu=state.z_latent.astype(state.z_mu.dtype),
                                   error=jnp.zeros_like(state.error))
        return node_class.energy_functional(new_state, node_info)
    if node_info.out_degree == 0 and not is_clamped:      # eval readout
        s = node_class.forward(params, inputs, state, node_info)
        return s._replace(z_latent=s.z_mu, error=jnp.zeros_like(s.error),
                          energy=jnp.zeros_like(s.energy))
    if is_clamped:                                        # energy = E(clamp, z_mu)
        return node_class.forward(params, inputs, state, node_info)
    eps = state.error                                     # eps-relaxed
    predicted = node_class.forward(params, inputs, state, node_info)
    derived = state._replace(z_latent=predicted.z_mu + eps)
    derived = node_class.forward(params, inputs, derived, node_info)
    return derived._replace(error=eps)
```

The second `forward()` call evaluates the node's energy at the derived latent *including* any custom in-forward energy terms (StorkeyHopfield's attractor term); the duplicated z_mu computation is an identical subexpression XLA CSE removes under jit. `_replace(error=eps)` keeps the carried variable bit-exact rather than the float-noise `(z_mu + ε) − z_mu`. Override audit: `EmbeddingNode` — ePC never differentiates w.r.t. inputs and the token source is a clamped constant, default works; `LinearExplicitGrad` — its override is a gradient shortcut, irrelevant to the global-AD path, default works; all other nodes are pure `forward()` implementations. Docstring notes record the audit on both classes; the `forward()` contract docstring (`base.py:328-382`) gains one paragraph on ePC evaluation at the derived latent.

## Component 4 — EPCInference

New file `fabricpc/core/inference_epc.py`:

```python
class EPCInference(InferenceBase):
    def __init__(self, eta_infer=0.1, infer_steps=5, latent_decay=0.0): ...
```

Inherits `inference_step` (template correct after Component 2), `zero_grads`, `run_inference`, `segments`. Overrides:

- `_relaxed_partition(structure, clamps) -> (eps_nodes, latent_nodes)`: trace-time Python over static structure (partition table above).
- `derive_states(params, state, clamps, structure)`: iterate `structure.schedule`; per visit `gather_inputs` → `scale_inputs` → `node_class.forward_from_error(..., is_clamped=(name in clamps))`.
- `forward_value_and_grad`: build the relaxed pytree `{"error": {name: eps for eps_nodes}, "z_latent": {name: z for latent_nodes}}`; one `jax.value_and_grad(energy_of, has_aux=True)` where `energy_of` writes the relaxed leaves into the state, runs `derive_states`, and sums `jnp.sum(energy)` over `in_degree > 0` nodes (the same set as `train.py:155-160`, so equilibria match sPC). Grads land in `latent_grad` by accumulation (preserves the accumulate-don't-replace invariant of `tests/test_inference_order.py`). Int token latents and clamps never enter the AD pytree; `GraphState` stays the sole fori_loop carry with invariant shapes/dtypes.
- `update_latents`: ε-relaxed nodes get `compute_new_error` (`error*(1 − η·decay) − η·latent_grad`); latent-relaxed sources get `compute_new_latent` (same rule on z_latent, satisfying the ABC meaningfully).
- `begin_segment`: sync ε from the incoming state — for ε-relaxed nodes run `forward()` at current latents so `error = z_latent − z_mu` (order-independent; all z_latent fixed). With `FeedforwardStateInit` this yields ε = 0, the paper's init; after an sPC segment it makes the bijection exact (sPC's final `error` field is one half-step stale).
- `finalize_state`: one detached `derive_states` rebuild so the final state satisfies `z_latent = z_mu + ε` with energies at the final point — the paper's weight rule; `compute_local_weight_gradients`, the train-loop energy, `eval_step`'s readout, and all dashboard readers of `.error` then work unchanged.

`NodeState` schema is untouched; `error`'s docstring (`core/types.py:118`) is updated: "Prediction errors (z_latent − z_mu); under `EPCInference` the first-class relaxed variable ε, with z_latent derived as z_mu + ε."

## Component 5 — InferenceSchedule

In `fabricpc/core/inference.py`:

```python
inference = InferenceSchedule(
    EPCInference(eta_infer=0.1, infer_steps=5),    # cheap global steps to near-equilibrium
    InferenceSGD(eta_infer=0.05, infer_steps=20),  # refine on the true arbitrary-graph energy
)
```

- `__init__(*solvers)`: validates non-empty, all `InferenceBase`; stores `solvers=tuple(...)` in config (plain `MappingProxyType`, holds objects fine — `InferenceBase` is not `FrozenConfig`).
- `run_inference(self, ...)`: fold the state through each solver's `run_inference` in order (each applies its own `begin_segment`/`finalize_state`, so ePC→sPC and sPC→ePC handoffs are state-consistent by construction).
- `segments()`: flattens component `segments()` — nested schedules compose.
- `inference_step`/`compute_new_latent`: raise `NotImplementedError` with directions to `segments()` — a schedule has no single per-step rule, and a loud failure beats silently running the wrong solver.

Tracking (`inference_tracking.py`): both variants iterate `structure.config["inference"].segments()`; per segment run `begin_segment` → `lax.scan`/Python loop of `inference_step` for that segment's steps → `finalize_state`; concatenate per-step metric stacks along axis 0 (metric structure is identical across segments). Single-solver graphs produce one segment and byte-identical output to today. This transitively fixes `trackers.py:488` and removes both `config["infer_steps"]` reads.

## Component 6 — Benchmark: ePC vs sPC on resnet18/CIFAR-10

Both wall-clock comparisons read off one sweep of T1 — ePC inference steps per minibatch (`infer_steps`), against sPC's `--spc_steps` inference steps per minibatch. All arms train the same `--num_epochs`, so each arm is one point (total training wall-clock, final test accuracy) and the T1 grid — not training checkpoints — sets the granularity of the time axis.

- `examples/resnet18_cifar10_demo.py`: `build_resnet18(...)` takes a required `inference: InferenceBase` replacing the `infer_steps`/`eta_infer` kwargs and the hardcoded `InferenceSGDNormClip` (:321-323); migrate `_create_mupc_model` and `run_single_mupc` (CLI behavior and docstring reference numbers unchanged).
- New `examples/epc_spc_resnet18_compare.py` (importlib load of the demo builder, per `examples/PC_backprop_compare.py:52-58`). CLI: `--mode {sweep,convergence}` (default `sweep`), `--n_trials 3`, `--num_epochs 2`, `--batch_size 256`, `--spc_steps 120`, `--spc_eta 0.1`, `--epc_eta 0.1`, `--lr`, `--weight_decay`, `--epc_step_sweep 1,2,3,4,5,6,7,8,9,10,16,32,64,128`, `--track_steps 120`.
  - **sweep** (default) — one `PlannedMultiContrastExperiment` with an arm per T1 in `--epc_step_sweep` plus an sPC-`{spc_steps}` baseline arm, empty contrast family (the runner supplies the paired trial loop; `TrialResult.metric_value`/`train_time` already carry everything needed), all arms at the same `--num_epochs` and each arm's adamw warmup-cosine schedule from `num_epochs × len(train_loader)`. Grid: dense 1–10 where accuracy moves fastest, log-spaced 16–128 above; 128 > `spc_steps` so sPC's wall-clock lands inside the ePC time range (if ePC-128 still finishes faster than sPC-120, extend the grid — the interpolation needs the sPC time point bracketed). Derived per-trial metrics: (i) **accuracy at equal wall-clock** — linear interpolation of the trial's ePC (train_time, accuracy) points at the trial's sPC train_time; (ii) **wall-clock to equal accuracy** — the smallest-T1 arm whose accuracy ≥ the trial's sPC accuracy, reporting its train_time and the ratio to sPC's. Paired t-test and Cohen's d on both across trials via `fabricpc.experiments.statistics` (`paired_ttest`, `cohens_d`). Chart (plotly, output convention of `examples/scaling/scaling_analysis_plots.py:943-950`): x = T1 on a log axis; top panel y = final test accuracy with mean ± SE error bars over trials and the sPC baseline as a horizontal line with SE band; bottom panel y = total training wall-clock per arm with sPC's as the horizontal reference — the equal-wall-clock crossing and the time-to-equal-accuracy gap are both readable off the two panels. Written to `epc_step_sweep.html` (+ `.png` when kaleido is installed, matching the scaling script's guard). Cost: per-minibatch inference steps summed across arms = 295 (ePC grid) + 120 (sPC) ≈ 3.5× a lone sPC-120 run per trial; n=3 ≈ 3 h on the reference 3090.
  - **convergence** (single seed, no training): identical params/initial state for both solvers on one test batch; `run_inference_with_history` per solver. Reports **per-node** energy-vs-step, not the global sum: total energy is dominated by output-adjacent nodes (the energy imbalance reported in Pinchetti et al.'s PC benchmarking paper, arXiv 2407.01163), so a global curve can read as sPC near-convergence while deep nodes have received no gradient signal. Plot: log10 per-node energy vs step, one line per node colored by schedule depth, side-by-side sPC/ePC panels (per-node series come directly from the per-step history dicts). The global sum is retained for exactly one purpose: **E\*** = sPC's final total energy, with ePC's steps-to-reach ≤ E\* as the head-to-head criterion. Also reports post-warmup per-step wall-clock for both solvers and their ratio — the direct measurement of the per-step cost multiple.

## File-by-file change list

Modified: `fabricpc/core/inference.py` (refactor + hooks + `InferenceSchedule`), `fabricpc/core/types.py` (`schedule` field + pytree + docstrings), `fabricpc/graph_assembly/graph_construction.py` (scheduler integration; delete `_topological_sort`), `fabricpc/graph_assembly/__init__.py` (exports), `fabricpc/graph_initialization/state_initializer.py` (:269 → schedule; docstring), `fabricpc/core/mupc.py` (dup-raise hardening + docs), `fabricpc/nodes/base.py` (`forward_from_error` + contract docs), `fabricpc/nodes/transformer_v2.py` + `fabricpc/nodes/linear_explicit_grad.py` (audit docstrings), `fabricpc/utils/dashboarding/inference_tracking.py` (segment iteration), `fabricpc/core/__init__.py` (export `EPCInference`, `InferenceSchedule`), `examples/resnet18_cifar10_demo.py`, `examples/mnist_cyclic_graph.py`, `tests/conftest.py` (`with_inference(structure, inference=None, **kwargs)`), migrated tests (`test_fabricpc.py`, `test_ndim_shapes.py`, `test_auto_node_grad.py`, `test_inference_order.py`), `docs/user_guides/12_api_inference.md`, `docs/user_guides/04_building_models.md`, `docs/user_guides/03_how_predictive_coding_works.md`, `CHANGELOG.md`.

New: `fabricpc/graph_assembly/scheduling.py`, `fabricpc/core/inference_epc.py`, `examples/epc_spc_resnet18_compare.py`, `tests/test_topology_scheduler.py`, `tests/test_inference_epc.py`, `tests/test_inference_schedule.py`.

Sequencing: (1) dispatch refactor + tracking segments + test migrations — pure refactor, full suite green; (2) topology scheduler + `schedule` field + consumer migrations + its tests; (3) `forward_from_error` + `EPCInference` + tests; (4) `InferenceSchedule` + tests + docs/exports; (5) resnet18 refactor + compare script.

## Test plan

- `tests/test_topology_scheduler.py`: DAGScheduler equals legacy Kahn orders across insertion permutations; raises `GraphCycleError` naming `{a,b,y}` on `x→a⇄b→y`; `UnrolledCycleScheduler(2)` → `("x","a","b","a","b","y")`, K=1 single-visit, `node_order` invariant; on DAGs UnrolledCycleScheduler == DAGScheduler for several K; `graph()` rejects incomplete/unknown schedules; determinism.
- `tests/test_state_initializer.py` additions: feedforward-through-cycles equals a manual replay on `x→a⇄b→y` (K=2); K=2 ≠ K=1 result (propagation proof); existing DAG suite unchanged.
- `tests/test_mupc.py` additions: dup-order raise; cyclic graph + muPC + UnrolledCycleScheduler yields non-None scalings with correct `K_slot`; merge-in-cycle L identical for K ∈ {1, 5}; all existing muPC tests pass unchanged.
- `tests/test_inference_epc.py`: ε=0 ⇔ feedforward init (`begin_segment` gives error ≈ 0; `derive_states` preserves z_latent); gradient correctness vs closed form on a 2-layer linear chain (ε_h + Wᵀ·output-residual) and vs a hand-rolled `jax.grad`; energy decreases over steps; **sPC equivalence**: small convex DAG run to convergence — per-node z_latent/z_mu/energy agree and `compute_local_weight_gradients` at both fixed points agree; branch coverage (CrossEntropy-clamped output, eval readout, int-token EmbeddingNode graph); cyclic smoke under the unrolled schedule (jit-compiles, finite decreasing energy); muPC-scaled graph z_mu matches manual scaling; insertion-order independence of one-step grads; `z_latent == z_mu + error` after `run_inference`.
- `tests/test_inference_schedule.py`: `segments()` flattening incl. nesting; single-solver schedule ≡ plain solver (allclose); ePC→sPC schedule ≡ manual sequential calls; runs inside `jax.jit(train_step)`; round-trip handoff energy-non-increasing; tracking parity (Σ steps rows, final state ≡ `run_inference`); raising stubs raise.
- Regression: full suite green after step (1) with only the nine migrated call sites touched; `test_doc_snippets.py` gates the rewritten user-guide examples.

## Verification

1. `pytest tests/` green at each sequencing step.
2. `python examples/mnist_cyclic_graph.py` runs with the explicit `UnrolledCycleScheduler` and trains.
3. `python examples/resnet18_cifar10_demo.py` (unchanged defaults) reproduces the documented single-run behavior.
4. `python examples/epc_spc_resnet18_compare.py --mode convergence` — per-node energy-vs-step panels, ePC steps-to-reach ≤ E*, and measured post-warmup per-step wall-clock for both solvers, on one batch (minutes).
5. `python examples/epc_spc_resnet18_compare.py --mode sweep --n_trials 3` (~3 h on the reference 3090) — accuracy-vs-T1 chart written to `epc_step_sweep.html`/`.png`, plus paired accuracy-at-equal-wall-clock and wall-clock-to-equal-accuracy tables; paste the tables into the script docstring per house convention.

## Decisions taken

User-confirmed 2026-08-14:

1. Default `DAGScheduler` raises `GraphCycleError` on cyclic graphs, directing users to select a cycle-capable scheduler explicitly even for a degenerate single-visit schedule.
2. New `NodeBase.forward_from_error` sibling method (not a mode on `forward_and_latent_grads`).
3. ePC applies muPC forward scaling only; gradient preconditioners are not replicated.
4. Benchmark defaults: n_trials=3; `sweep` and `convergence` modes (revised 2026-08-18, below).

Benchmark revisions, user-directed 2026-08-18:

5. No step-count/wall-clock equivalence is assumed between ePC and sPC. Per-step costs are close (a reference torch ePC ran ~20% slower per step than sPC), so ePC's ~10× fewer steps make it far cheaper per weight update, not wall-clock-equal at 10-vs-120 steps. The benchmark measures wall-clock and compares accuracy at equal measured wall-clock (primary), plus wall-clock to reach equal accuracy.
6. Convergence mode reports energy per node, not only globally: the global sum is dominated by output-adjacent nodes (the energy imbalance in Pinchetti et al.'s PC benchmarking, arXiv 2407.01163), which can read as sPC near-convergence while deep nodes have received no gradient signal. The global E* is retained solely as the ePC steps-to-threshold criterion.
7. The T1 sweep is the benchmark's time axis. T1 = ePC inference steps per minibatch; each T1 trained at fixed `--num_epochs` is one (total wall-clock, final accuracy) point, so the grid `1,2,…,10,16,32,64,128` — not training checkpoints — sets the time-axis granularity, and both wall-clock comparisons interpolate along the sweep. The sweep chart is task performance across the full grid against the sPC baseline.

## Alternatives considered

- **ε storage**: chosen `NodeState.error` (write-mostly today; verified readers are dashboards + `LinearExplicitGrad`). Rejected: extra fori_loop-carry dict (invisible to tracking/handoff/weight path, fails the first-class requirement); new `epsilon` field (schema churn through every constructor, redundant with `error`).
- **Gradient computation**: chosen one global `jax.value_and_grad` over a `{field: {node: array}}` dict. Rejected: per-node grads hand-stitched through the schedule (re-derives reverse AD, wrong for repeated visits, reintroduces the decay ePC removes); grad w.r.t. whole `GraphState` (differentiates int latents and clamps; needs masking).
- **Scheduler type**: chosen `InferenceBase` subclass with `segments()` + raising per-step stubs. Rejected: separate protocol (breaks `graph()`'s contract, two type surfaces); `graph(inference=[...])` list support (composition in the wrong layer, no nesting).
- **Dispatch fix**: chosen classmethods + instance `run_inference`. Rejected: threading `cls` parameters (noisy, still mis-passable); instance methods everywhere (abandons the static-pure-function house style without need).
- **Schedule location**: chosen new `schedule` field beside `node_order` (each consumer takes the semantically right projection; logging readers untouched; muPC misuse blocked by dup-raise). Rejected: replace `node_order` (every one-visit consumer must dedup); stash in `structure.config` (core topology data stringly-typed).
- **Unroll algorithm**: chosen Tarjan SCC + Kahn-on-condensation + entry-first intra-SCC BFS (minimal schedule length, exact DAG equality, deterministic). Rejected: repeat full sweep K times (multiplies ePC forward cost on the acyclic majority); feedback-arc-set removal (back edges never carry information, fails the requirement).
- **Node energy at derived point**: chosen second `forward()` call (CSE-deduped). Rejected: `energy_functional` directly (silently drops in-forward energy terms like StorkeyHopfield's); splitting the node contract into predict()/energy() (whole-node-class migration, out of scope).
- **Equal-wall-clock mechanism**: chosen the T1 sweep as the time axis — every arm trains the same `--num_epochs`, each T1 yields one (total wall-clock, final accuracy) point, both comparisons interpolate along the sweep, and `TrialResult.metric_value`/`train_time` already carry all the data. Rejected: per-epoch checkpoint curves with budget-matched epoch counts (ties the time axis to checkpoint/minibatch cadence, which is not the quantity under study, and needs a calibration pass plus a `TrialResult` extension); wall-clock stopping rule inside `train_pcn` (perturbs the shared trainer for one benchmark; nondeterministic epoch boundaries break trial pairing and warmup-cosine schedule construction); assumed step-count parity between ePC-10 and sPC-120 (struck 2026-08-18 — per-step costs are close, so the ratio must be measured); bespoke trial loop in the compare script (duplicates `PlannedMultiContrastExperiment`'s pairing-by-seed machinery).
