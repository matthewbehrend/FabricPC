# ePC inference solver, composable inference schedule, and the unrolled topological schedule

## Context

FabricPC's two inference solvers (`InferenceSGD`, `InferenceSGDNormClip`) implement state-based predictive coding (sPC): latent states relax by local gradient descent, so the output-loss signal attenuates by the state learning rate per layer per step and deep graphs need 100s of inference steps (resnet18 demo: 120 steps, 476 s/epoch). The ePC paper (Goemaere et al., arXiv 2505.20137) reparameterizes PC over prediction errors: one reverse-mode AD pass through the whole network delivers the loss signal to every layer unattenuated, reaching the same equilibrium in ~several steps. sPC remains the general solver for arbitrary graphs; ePC is the efficient solver for DAG (or unrolled-cyclic) representations.

This plan adds five things: a split of the node forward contract into `predict` / `pair` / `energy` (Component 3), so one node-level prediction pass serves both parameterizations; ePC as an `InferenceBase` subclass; a composable inference schedule (a few ePC steps to near-equilibrium, then sPC refinement on the true arbitrary-graph energy); a generalization of the topological-order method to an unroll degree `U`, so ePC also accepts cyclic/self-recurrent graphs by unrolling; and an ePC-vs-sPC benchmark on the resnet18 demo.

It also fixes five latent defects the design exposed: (1) `InferenceBase` template methods re-resolve their own class via `type(structure.config["inference"])` (`core/inference.py:100-101`, `:212-213`, `:258-259`), which breaks any composition (Component 2); (2) on cyclic graphs `_topological_sort` returns a partial order behind a print warning (`graph_construction.py:102-103`) — on `x→a⇄b→y` the order is just `("x",)`, so feedforward init leaves cycle members *and everything downstream* at random init, and muPC attaches no scaling to them (Component 1); (3) every state initializer leaves `in_degree == 0` nodes with z_mu = 0 while error = 0, violating error = z_latent − z_mu at init — sPC masks this on the first forward (`nodes/base.py:495-507`), ePC would read the invalid z_mu directly (Formulation, Component 1); (4) sPC forces unclamped readouts to error = 0, energy = 0, zeroing a Hopfield readout's attractor energy (Component 3); (5) the node contract fuses prediction, the additive error map, and energy scoring in one `forward()` — `error = z_latent − z_mu` is copy-pasted into every node body while `GaussianEnergy` recomputes the difference itself and never reads `state.error`, custom energy terms are post-hoc `state._replace` patches (StorkeyHopfield; `ScaledSumNode`, `tests/test_external_custom_node.py:150-153`), and source semantics live in the solver branch (`nodes/base.py:495-507`) so `IdentityNode.forward` crashes if ever called with `in_degree == 0` (Component 3).

A second review round (2026-09-03 to 2026-09-08) followed the resnet18 results. It added an exact linear oracle for both solvers and regime and stability diagnostics for ePC, and it is recorded in full under "Review round 2" below.

## Design overview

ePC solves the DAG; state-based PC (sPC, today's settling path) refines around cycles; a cycle can instead be unrolled into ePC's own graph. Symbols: `T` = settling ticks per update in sPC alone, `T1` = ePC steps, `T2` = sPC refinement steps, `U` = unrolled cycle traversals, `(H)` = a Hopfield (recurrent) node.

**1 — Composable inference schedule.** Solvers are schedule entries composed per weight update, not trainer modes. sPC minimizes the exact graph energy as-is; on a cyclic graph ePC minimizes an approximation whose fidelity is set by the unroll degree.
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

**3 — The alternative cycle path: unroll into ePC's DAG.** `U` traversals of the cycle become `U` copies of the cyclic subgraph inside one differentiable program — no separate solver, no stopping rule, no handoff; `U` is fixed at graph-build time (`graph(..., unroll=U)`):

```
cyclic graph                     unrolled into ePC's DAG (U = 2)

a ──► b ──► c ──► d              a ──► b₀ ──► c₀ ──► b₁ ──► c₁ ──► d
      ▲     │
      └─────┘                    the back-edge c→b becomes the feedforward edge c₀→b₁
```

The subscripts b₀, b₁ are repeated forward passes through node b, each re-injecting the same ε_b: errors are tied between traversals by construction (mechanism in Formulation).

Two questions are experimental, and Component 6 supplies the first measurements: whether an ePC warm start helps a given architecture and at what depth, and whether a cycle is better refined by sPC, unrolled into ePC, or both (an unrolled ePC segment followed by sPC refinement is already expressible with the composed solvers). Cycle depth drives both sPC's signal decay and ePC's unrolling cost.

## Formulation

Symbols (one meaning throughout): for node *i*, **z_mu_i** is the node's prediction of its own latent computed by `predict()` from its in-edge sources; **z_latent_i** is the latent state; **ε_i** is the prediction error, stored in `NodeState.error`; **E** is the total energy Σ over nodes with `in_degree > 0` of the node's `energy()` method (functional default plus any custom terms, Component 3); **η** (`eta_infer`) is the inference rate — per-solver; sPC's local rate and ePC's global rate differ in scale (below); **T** (`infer_steps`) the step count; **aux** is an arbitrary pytree of intermediates a node's `predict` returns for its `energy` (Component 3).

sPC relaxes z_latent with ε derived (`error = pair_error(z_latent, z_mu) = z_latent − z_mu`, computed once in the base `forward()` template — Component 3). ePC inverts the parameterization: ε is the first-class relaxed variable and z_latent is derived by a forward pass in schedule order, `z_latent_i := z_mu_i + ε_i`. Because z_mu_i depends on upstream z_latents, every node's energy — including the clamped output node's `energy(y_clamp, z_mu)`, which is the paper's output loss — depends on all upstream ε. One `jax.value_and_grad` over the ε pytree through the whole derived forward gives exact ∇_ε E; ε steps down that gradient. The ε↔z_latent map is a bijection with unit-determinant triangular Jacobian (paper Appendix C): identical energies, identical equilibria, and the final derived state feeds the existing local weight-gradient path unchanged.

Mechanically this is the same pattern `train_backprop` (`fabricpc/training/train_backprop.py:118`) already uses — one feedforward pass, one `jax.value_and_grad` — with the differentiated variable swapped. The backprop trainer differentiates the output loss with respect to the weights; `EPCInference` walks the DAG representation of the network at the chosen unroll degree `U` (`structure.schedule`), injects the carried ε at every visit (`z_latent := z_mu + ε`), and differentiates the total energy E with respect to ε.

Because the ε gradient is taken through the full network's transfer function — a change in one node's ε moves every downstream derived latent — η must be tuned like a weight learning rate, not like sPC's local per-node rate. Measured on the resnet18/CIFAR-10 convergence benchmark (Component 6), the ε updates to reach sPC's final 120-step energy: 104 at η = 1e-3, 11 at 1e-2, 4 at 3e-2, 1 at 0.1. The default is 1e-2 as a robustness margin, not the fastest measured rate: one batch on one architecture is thin evidence for 0.1's stability across models, and 1e-2 already converges in about a dozen updates. (Superseded: the default returned to 1e-3 in commit 0d7c4c2 after the 100-epoch resnet18 runs collapsed at 1e-2 for every step count; see the interpretation under the results tables.)

One derive rule for every node — z_latent = `pair_latent(z_mu, ε)` = z_mu + ε, the inverse of `pair_error` — with the clamp deciding which side is free (computed at trace time from static structure + clamp keys):

- **Unclamped**: ε is the relaxed variable; the derive pass writes z_latent := z_mu + ε. This holds regardless of degree — top-down priors (in_degree == 0) and eval readouts (out_degree == 0) included. A readout needs no special case: an unclamped out_degree == 0 node takes the ordinary path — at the ε = 0 init a pure-Gaussian readout's gradient ∇_ε E = precision·ε is zero, so it stays put, while a Hopfield readout receives its attractor gradient; its energy is whatever `forward()` assigned. The sPC branch that forced readouts to zero (`nodes/base.py:514-529`) is deleted in the same change (Component 3).
- **Clamped**: z_latent stays the clamp and ε is derived — where in-edges exist, the template recomputes error = `pair_error(clamp, z_mu)`, so the invariant holds with the clamp fixed, and the output clamp's `energy(clamp, z_mu)` is the paper's output loss with gradient reaching upstream ε through z_mu; a clamped source keeps its init state (z_mu = clamp, error = 0) with nothing to recompute. Clamped nodes never enter the relaxed pytree, which also keeps int-dtype token sources out of the AD pytree.

The pair is base-owned and additive, not a node override point (Component 3): the ε↔z_latent equivalence needs a volume-preserving bijection (paper Appendix C), and today's ε is unweighted everywhere — if a precision-weighted ε is ever wanted, the pair moves to `EnergyFunctional` so both directions and the energy stay consistent. `predict` must not read `state.z_latent` values (shape/dtype reads are fine): sPC differentiates z_mu through such a read — the grad wrapper re-binds z_latent and calls `forward` — while ePC evaluates z_mu at the carried latent, so a z_latent-reading `predict` makes the two solvers minimize different energies. Other `NodeState` fields may be read; during ePC derivation they carry the previous step's values, so z_latent = z_mu + ε holds by construction and at a fixed point the carried and derived states coincide. aux is snapshotted when `predict` runs — under ePC, before z_latent is derived. An energy term that needs the node's own z_latent must read `state.z_latent` inside `energy()` (as the Hopfield attractor term does), never an aux entry computed from it: an aux entry frozen at the carried latent, combined with the derived latent in the rest of the term, makes ePC's node energy a function of two different latents while sPC's is a function of one, so the solvers minimize different energies and the shared-equilibrium argument fails.

z_mu is recomputed by `predict()` from in-edge sources where in_degree > 0. in_degree == 0 nodes have no in-edges to project a z_mu, so initialization assigns it: after the configured initializer and its clamp overlay run, the shared `initialize_graph_state` dispatch (`state_initializer.py:349-386`) copies z_mu ← z_latent for every in_degree == 0 node, cast to z_mu's float dtype so int token clamps keep the float-carry invariant (`state_initializer.py:137-141`), and zeroes the node's error in the same `_replace`, giving error = z_latent − z_mu = 0 at init by enforcement rather than by the initializer happening to zero the field. This is a shared-infrastructure fix: all three concrete initializers (`GlobalStateInit:150`, `NodeDistributionStateInit:200`, `FeedforwardStateInit:259`) leave source z_mu at zeros today. From the fixed state either solver runs on the same invariant: sPC derives error = z_latent − z_mu and relaxes z_latent; ePC derives z_latent = z_mu + ε and relaxes ε. During ePC a source's z_mu stays fixed and its ε receives gradient through downstream z_mu; since E excludes in_degree == 0 nodes, that is the same signal sPC accumulates into the source's latent_grad (`core/inference.py:192-197`), so the two solvers share equilibria on graphs with unclamped priors. No node needs special handling during inference.

At ε = 0 the derived states equal `FeedforwardStateInit`'s output, so the existing default initializer implements the paper's zero-init; the initializer iterates the same `structure.schedule` that `EPCInference.derive_states` iterates (Component 1), so this holds at any unroll degree.

Memory usage: each ePC step's single `value_and_grad` stores activations for the whole derived forward — full depth times U cycle copies — reverse-mode memory at backprop scale, versus sPC's per-node closures. `train_backprop.py` already runs the same reverse pass on these demos, so resnet18 at batch 256 fits.

muPC: `scale_inputs` (`core/scaling.py:33`) applies inside the differentiated forward exactly where the sPC loop applies it, and the global AD supplies the chain-rule factors automatically. The gradient preconditioners (`jacobian_gain` in `topdown_grad_scale`, `self_grad_scale`) are per-hop conditioners for sPC and are not replicated in ePC, a global backward pass; the decision is documented on `EPCInference`. Consequence: with muPC attached the two solvers settle to different fixed points — `jacobian_gain` reshapes sPC's latent flow while ePC descends plain ∇_ε E — and a slow-marked test pins the divergence (Test plan). `scale_weight_grads` (`core/scaling.py:87`) at learning time is untouched.

Cyclic graphs: the forward-from-errors pass iterates `structure.schedule` (Component 1). Errors are tied between traversals by construction: the relaxed pytree in `EPCInference.forward_value_and_grad` keys ε by node name (one leaf per node), `GraphState` carries one `NodeState` per node, and `forward_from_error` re-injects the carried `state.error` on every visit. A repeated visit recomputes z_mu from the latest source latents and re-derives z_latent with the *same* ε, and each visit overwrites the node's single `NodeState`, so node *i*'s energy term enters E once, evaluated at node *i*'s final visit — not summed over unroll copies. That final-visit energy is the output of the computational graph threaded through every traversal, so E carries the causal path through all U traversals and the single AD pass differentiates through all of them. StorkeyHopfield needs no special handling: its z_mu comes purely from the input probe and its self-recurrence enters only through the Hopfield energy term on its own z_latent, which its `energy()` override (Component 3) evaluates at the derived latent (`nodes/storkey_hopfield.py:333-363`).

## Component 1 — Generalized topological schedule

`_topological_sort` (`graph_construction.py:65-105`) is generalized in place — no new module, no scheduler class:

- `_topological_sort(nodes, edges, unroll: Optional[int] = None) -> Tuple[str, ...]` returns the full node visit schedule; cycle members may repeat; every node appears at least once.
  - On a DAG: the existing Kahn/BFS loop unchanged (same queue seeding from dict order, same successor order → bit-identical order on every existing DAG), whatever `unroll` is.
  - On a cyclic graph with `unroll=None`: raise `GraphCycleError` naming the unordered nodes and directing the user to pass `unroll` explicitly — even for a degenerate single-visit schedule. This replaces the warn-and-return partial order (the `x→a⇄b→y` → `("x",)` defect).
  - On a cyclic graph with `unroll=U ≥ 1`: iterative Tarjan SCC → Kahn on the condensation DAG (same seeding/order rules) → each nontrivial SCC's members emitted `U` times, intra-SCC order = BFS from entry nodes (members with an in-edge from outside the SCC, dict order; first member if none). `x→a⇄b→y`, U=2 → `("x","a","b","a","b","y")`. `U=1` = each cycle member visited once — the explicit degenerate choice. Self-edge rejection at build (`graph_construction.py:152-153`) stays.
- `first_occurrence_order(schedule) -> Tuple[str, ...]`: dedup keeping first occurrences (the unique node order).
- `GraphCycleError(ValueError)`, exported from `fabricpc.graph_assembly`.

Integration:
- `GraphStructure` (`core/types.py:149-172`) gains a `schedule: Tuple[str, ...]` field after `node_order`, with the build-enforced invariant `node_order == first_occurrence_order(schedule)` (equal on DAGs). Pytree static aux (`types.py:213-217`) updated. Construction sites: `graph_construction.py` and the unflatten lambda only.
- `graph(..., unroll: Optional[int] = None)` — validated ≥ 1 when given, bools rejected (`isinstance(True, int)` is true, so `unroll=True` would otherwise silently build at degree 1); `schedule = _topological_sort(finalized_nodes, edge_infos, unroll)`; `node_order = first_occurrence_order(schedule)`; `unroll` recorded in `gs_config` beside `inference`.
- Consumer migrations (no fallbacks):
  - `FeedforwardStateInit` pass 2 (`state_initializer.py:269`) iterates `structure.schedule`. The loop body is already revisit-correct, so cyclic graphs gain true feedforward init through cycles (bundled defect fix), and pass 2 walks the same unrolled schedule `EPCInference.derive_states` walks, so initialization is the derived forward at ε = 0 at the graph's unroll degree.
  - `initialize_graph_state` (`state_initializer.py:349-386`) gains the shared post-pass assigning z_mu ← z_latent and error ← 0 for every `in_degree == 0` node after the dispatched initializer returns (Formulation), fixing all three concrete initializers at one point. Zeroing error in the same `_replace` makes the source invariant enforced by the post-pass rather than inherited from the built-in initializers happening to zero the field.
  - muPC (`core/mupc.py`) stays on the unique `node_order`: depth L models one merge-sum energy term per merge node regardless of visit count, and back edges contribute depth 0 naturally (`skip_counts.get(source, 0)` returns 0 for later-ordered sources); add a duplicate-entry raise to `compute_mupc_scalings`/`_count_skip_connections_depth` as hardening.
  - Cyclic call sites gain the explicit `unroll` argument: the `_build_cycle` helper (`tests/test_inference_order.py:105-126`) and `examples/mnist_cyclic_graph.py:89-108`. The cyclic-graph section of `docs/user_guides/04_building_models.md` (:456-472) gains a complete `graph(..., unroll=U)` example — its snippet today is two bare `Edge(...)` lines with no `graph()` call, and `test_doc_snippets.py` AST-checks fenced blocks, so the example must be complete.

## Component 2 — InferenceBase dispatch refactor

`fabricpc/core/inference.py`: template methods become `@classmethod` dispatching on `cls`; `run_inference` becomes an instance method (it needs `self.config`). No dual-mode paths; all callers migrate.

| Method | Change |
|---|---|
| `inference_step` (:86) | `@classmethod`; body uses `cls.zero_grads / cls.forward_value_and_grad / cls.update_latents` (drops the `structure.config` re-resolution at :100-101) |
| `zero_grads` (:114) | unchanged static |
| `forward_value_and_grad` (:134) | `@classmethod` (body unchanged; subclass overrides need `cls`) |
| `update_latents` (:201) | `@classmethod`; uses `cls.compute_new_latent` (drops :212-213) |
| `compute_new_latent` (:229) | unchanged abstract static |
| `run_inference` (:245) | instance method: `cls = type(self)` (drops the re-resolution at :258-259); `state = cls.begin_segment(...)`; `lax.fori_loop(0, self.config["infer_steps"], ...)` stepping `cls.inference_step(..., self.config)`; `return cls.finalize_state(...)` |
| new `begin_segment` / `finalize_state` | `@classmethod`, default identity — segment-boundary hooks (ePC overrides both; sPC untouched) |
| new `segments()` | instance method, default `((self, int(self.config["infer_steps"])),)` |

Call-site migrations (complete): module-level `run_inference` (:364-389) delegates to `structure.config["inference"].run_inference(...)` — its own signature is unchanged, so `train.py` (:152, :516, :598, :744) and `train_autoregressive.py` (:208, :435, :627) need no change. Tracking (`utils/dashboarding/inference_tracking.py` :50-59, :143-153) migrates to segment iteration (Component 5). Tests calling the old static form: `tests/test_fabricpc.py` (:185, :190, :227, :428), `tests/test_ndim_shapes.py` (:64, :105, :152), `tests/test_auto_node_grad.py` (:315, :318). Adjacent audit: `StateInitBase` dispatch (`state_initializer.py:384`) dispatches on the object it was handed, not via structure — not the same defect, no change; no other `type(structure.config[...])` self-resolution exists.

## Component 3 — Node contract split: predict / pair / energy

`forward()` (`fabricpc/nodes/base.py:328-382`) fuses three stages behind the sPC dataflow direction: **predict** — z_mu from params and in-edge inputs, the expensive stage; **pair** — error = z_latent − z_mu, inlined per contract step 2 in every node body; **score** — the energy functional plus any in-forward custom terms. sPC consumes z_latent → ε; ePC needs ε → z_latent, and needs z_mu before z_latent exists. With the monolithic method the ePC derive pass would call `forward()` twice per unclamped internal node — once for z_mu, once for the energy at the derived latent. That double call is rejected (Alternatives): a `predict` that reads the node's own `NodeState` gets two different z_mu — the second call re-evaluates at the derived latent — and the returned state silently violates z_latent = z_mu + error; where the two calls do coincide, correctness rests on XLA CSE merging them, which is backend behavior rather than contract, and the traced jaxpr and eager-debug cost double regardless.

The contract splits instead. Node authors implement two staticmethods:

```python
@staticmethod
@abstractmethod
def predict(params, inputs, state, node_info) -> Tuple[jnp.ndarray, Any]:
    # All parameterized computation: z_mu (shape (batch,) + node_info.shape)
    # plus aux, an arbitrary pytree of intermediates for energy() (None if
    # unused). predict must not read state.z_latent values (shape/dtype
    # reads are fine); other state fields carry the previous step's values
    # under EPCInference.derive_states.
    ...

@staticmethod
def energy(params, inputs, state, aux, node_info) -> jnp.ndarray:
    # Per-sample energy (batch,) at (state.z_latent, state.z_mu). Override to
    # add terms (StorkeyHopfield's attractor; ScaledSumNode's weighting).
    # z_latent-dependent terms read state.z_latent here; aux is snapshotted at
    # predict time — under ePC before z_latent is derived — so an aux entry
    # computed from z_latent evaluates the energy at two different latents
    # (Formulation). Overrides must tolerate aux=None: the in_degree == 0
    # template branch passes it (sources get no predict call).
    energy_obj = node_info.energy
    return type(energy_obj).energy(state.z_latent, state.z_mu, energy_obj.config)
```

`energy_functional` (`base.py:596-617`) is deleted; its body is the default `energy()`, and the templates write `state.energy`.

`NodeBase` owns the pair and the assembly; these are not node override points (audit-tested, since Python cannot mark them final):

```python
@staticmethod
def pair_error(z_latent, z_mu):  # sPC direction
    return z_latent - z_mu

@staticmethod
def pair_latent(z_mu, error):    # ePC direction — the inverse
    return z_mu + error

@staticmethod
def forward_with_aux(params, inputs, state, node_info) -> Tuple[NodeState, Any]:
    node_class = node_info.node_class
    if node_info.in_degree == 0:
        # Source semantics, one owner (previously inlined in the sPC solver
        # branch, base.py:495-507): no in-edges project a z_mu, so it mirrors
        # z_latent (cast — a source z_latent may carry an int clamp dtype).
        new_state = state._replace(
            z_mu=state.z_latent.astype(state.z_mu.dtype),
            error=jnp.zeros_like(state.error))
        return new_state._replace(
            energy=node_class.energy(params, inputs, new_state, None, node_info)), None
    z_mu, aux = node_class.predict(params, inputs, state, node_info)
    new_state = state._replace(
        z_mu=z_mu, error=node_class.pair_error(state.z_latent, z_mu))
    return new_state._replace(
        energy=node_class.energy(params, inputs, new_state, aux, node_info)), aux

@staticmethod
def forward(params, inputs, state, node_info) -> NodeState:
    # Signature and semantics unchanged: sPC, FeedforwardStateInit, and both
    # grad wrappers call it as before; output is bit-identical to the
    # pre-split bodies (same z_mu ops, same subtraction, same functional call).
    new_state, _ = node_info.node_class.forward_with_aux(params, inputs, state, node_info)
    return new_state

@staticmethod
def forward_from_error(params, inputs, state, node_info, is_clamped) -> NodeState:
    # ePC state derivation: state.error is the relaxed ε; one predict per
    # visit. Runs inside EPCInference's global jax.grad — differentiable
    # w.r.t. inputs and state.error. Never writes latent_grad.
    node_class = node_info.node_class
    if node_info.in_degree == 0:
        if is_clamped:
            return state          # clamp fixed at init; nothing to recompute
        # top-down prior: z_mu is the constant assigned at initialization
        return state._replace(z_latent=node_class.pair_latent(state.z_mu, state.error))
    if is_clamped:
        # z_latent stays the clamp; the sPC-direction template derives
        # error = pair_error(clamp, z_mu) and energy(clamp, z_mu) — the output loss.
        new_state, _ = node_class.forward_with_aux(params, inputs, state, node_info)
        return new_state
    z_mu, aux = node_class.predict(params, inputs, state, node_info)
    eps = state.error             # the relaxed variable, written back bit-exact
    new_state = state._replace(
        z_latent=node_class.pair_latent(z_mu, eps), z_mu=z_mu, error=eps)
    return new_state._replace(
        energy=node_class.energy(params, inputs, new_state, aux, node_info))
```

Properties: one `predict` call per visit in both solvers; z_latent = z_mu + ε holds by construction for arbitrary predicts (z_latent reads are forbidden by contract; other state fields follow Formulation's lagged rule); custom in-forward energy terms live in `energy()`, so the derive path evaluates them with no second pass; there is no readout branch — an unclamped `out_degree == 0` node takes the ordinary path (Formulation); `forward_from_error` is a base template, not a node override point, so the per-node audit surface is the two-method contract. aux is created and consumed within one trace — never stored in `NodeState`, never in the fori_loop carry.

**Why aux must not carry z_latent-derived entries** — the Formulation rule in detail:

```
sPC forward():                        ePC forward_from_error():
  z_mu, aux = predict(state)            z_mu, aux = predict(state)      ← state.z_latent = carried (old) value
  ε = z_latent − z_mu                   z_latent := z_mu + ε            ← z_latent REPLACED after predict
  E = energy(state, aux)                E = energy(state′, aux)         ← state′ has the new z_latent,
      ← same z_latent throughout            but aux was computed from the old one
```

aux is snapshotted at `predict` time. Under sPC that's harmless — z_latent doesn't change within the call. Under ePC, z_latent is replaced after `predict` and before `energy`, so anything in aux that was computed from `state.z_latent` is frozen at the old latent while the rest of the energy uses the new one.

**Why that breaks equivalence.** Say a node author writes `predict` to compute `gate = sigmoid(state.z_latent)` — a read the predict contract now forbids outright — stashes it in aux, and `energy()` returns `gate * ‖z_latent − z_mu‖²`. sPC's energy at latent z is sigmoid(z)·‖z − μ‖² — one function of one point z. ePC's energy at derived latent z is sigmoid(z_old)·‖z − μ‖² — a function of two different latents, the carried one inside the gate and the derived one in the quadratic. That's the "different points": the two solvers are no longer minimizing the same energy function, so the shared-equilibrium argument (sPC and ePC reach the same fixed points because they minimize the same E in two coordinate systems) fails for that node.

**The correct pattern the rule is pointing at.** A z_latent-dependent energy term belongs inside `energy()`, reading `state.z_latent` directly — then both solvers evaluate it at the same latent the rest of the energy uses. The Hopfield attractor term does exactly this: it reads `state.z_latent` in `energy()`, while its aux carries only W and strength, which come from params. That's also why no existing node violates the rule — every current aux entry (pre_activation, W, strength) depends only on inputs and params, which don't change between `predict` and `energy`.

**Migration (complete, no fallbacks — a breaking change to the documented custom-node API, recorded in `CHANGELOG.md`):**

- The 13 library `forward()` bodies become `predict()` by deleting the pair/energy tail: `linear.py` (`_forward_with_preact` is deleted; `Linear.predict` returns `aux=pre_activation`), `identity.py`, `skip_connection.py`, `linear_residual.py`, `convolutional.py`, `pooling.py`, `storkey_hopfield.py`, `transformer.py`, and the five `transformer_v2.py` nodes. `LinearExplicitGrad` inherits `Linear.predict`; its analytic overrides call `forward_with_aux` — one owner of the tail — instead of `_forward_with_preact`.
- `StorkeyHopfield.predict` returns `aux=(W, strength)`; `accumulate_hopfield_energy` becomes an `energy()` override — the default term first, then `+ strength * E_hop`, preserving today's op order for bit-exactness. On `aux is None` (source use) the override falls back to the base PC term, matching main: a source node's params are empty (`params_initializer.py:39-41`), so `(W, strength)` cannot be recomputed there.
- Custom nodes outside the library: `examples/jpc_fc_resnet_compare.py` (three nodes, :252, :341, :422), `tests/test_external_custom_node.py` (`ScaledSumNode`, :123 — its post-hoc energy weighting becomes a three-line `energy()` override), `tests/test_mupc.py:632`.
- `docs/user_guides/06_custom_nodes.md` is rewritten around the two-method contract; its five-step `forward()` recipe with the full ConvNode example is AST-checked by `test_doc_snippets.py`, so the rewritten example must be complete. Both it and the node API reference (`docs/user_guides/10_api_nodes.md`) state the predict z_latent prohibition, state source semantics per solver direction — under sPC the templates mirror z_mu ← z_latent with zero error, while `forward_from_error` derives an unclamped source's z_latent from the frozen z_mu — and document the aux pattern and its anti-pattern: aux carries intermediates that depend only on params and inputs (`pre_activation`; StorkeyHopfield's `(W, strength)`), and an energy term that needs the node's own z_latent reads `state.z_latent` inside `energy()` — the StorkeyHopfield attractor term is the worked example; the anti-pattern, an aux entry computed from `state.z_latent` in `predict`, freezes the carried latent into an energy otherwise evaluated at the derived latent, so sPC and ePC minimize different energies (Formulation).
- sPC solver consolidation: the `in_degree == 0` branch of `forward_and_latent_grads` (`base.py:495-512`) delegates its state computation to `node_class.forward(...)` — whose source guard is now the one owner of source semantics — and keeps its zero `input_grads`/`self_grad`. Value-identical; removes the coupling where `IdentityNode.forward` crashes if called on a source.

Bundled sPC fix, same file: `forward_and_latent_grads`' unclamped-readout branch (`base.py:514-529`) forces z_latent = z_mu, error = 0, energy = 0, and zeroes `latent_grad` — contradicting the method's own contract ("`latent_grad` is *not* modified here", `base.py:485-486`) and discarding in-forward energy terms, so a Hopfield readout can never settle onto its attractor as an output node. Delete the branch; unclamped readouts take the ordinary autodiff path: error = z_latent − z_mu, energy as `forward()` assigns, z_latent relaxed like any other node. Impact: predictions read z_mu at every eval site — `eval_step` (`train.py:531`), `evaluate_pcn` (:644), and `evaluate_transformer` (:781, :800), the last migrated from z_latent in this change (the two reads coincide only under zero-error init on a DAG; otherwise z_latent lags); reported eval energy changes because `eval_step` sums energy over all nodes with no in_degree filter (`train.py:519-521`) and the readout's energy was previously zeroed. The `in_degree == 0` branch (`base.py:495-512`) stays as the grad short-circuit, with its state computation delegated to the template `forward` (Migration above): sPC re-syncs z_mu ← z_latent as the source's z_latent relaxes, keeping error = 0, while ePC holds z_mu fixed and moves ε; both move the source's z_latent by the downstream gradient, so equilibria match. After the readout-branch deletion the base body branches only on `in_degree`, so `is_clamped` is unread there; the parameter stays in the signature for solver gating and the `EmbeddingNode` override. `examples/storkey_hopfield_recall.py:137-142` documents the constraint this removes, under a stale line reference (`base.py:382-400`); rewrite that comment — a StorkeyHopfield readout now relaxes onto its attractor.

Override audit: `EmbeddingNode`'s `forward_and_latent_grads` override (`transformer_v2.py:110`) belongs to the sPC path, which ePC never calls, and calls the template `forward` — unaffected. `LinearExplicitGrad`'s analytic overrides (`linear_explicit_grad.py:49, :115`) migrate from `Linear._forward_with_preact` to `forward_with_aux` and are otherwise unchanged. `TransformerBlock.forward_and_weight_grads` (`transformer.py:403-440`) closes over the template `forward` — unaffected. `forward`, `forward_with_aux`, `forward_from_error`, `pair_error`, and `pair_latent` are not override points; `tests/test_node_contract.py` asserts every registered node class resolves them to `NodeBase`'s. The `predict`/`energy` contract docstrings carry the z_latent read prohibition, the lagged-state rule for other fields, the aux snapshot rule with its anti-pattern, and the aux = None tolerance requirement (Formulation).

## Component 4 — EPCInference

New file `fabricpc/core/inference_epc.py`:

```python
class EPCInference(InferenceBase):
    def __init__(self, eta_infer=1e-2, infer_steps=5, latent_decay=0.0): ...
```

`eta_infer` defaults to 1e-2 (superseded: 1e-3 since commit 0d7c4c2, see the interpretation under the results tables); the constructor docstring directs tuning it like a weight learning rate (the ε step descends the full-transfer-function gradient), quotes the measured convergence table (104/11/4/1 ε updates at η = 1e-3/1e-2/3e-2/0.1), and states the robustness-margin rationale for not defaulting to the fastest measured rate (Formulation).

Inherits `inference_step` (template correct after Component 2), `zero_grads`, `run_inference`, `segments`. Overrides:

- `_relaxed_errors(structure, clamps) -> Tuple[str, ...]`: trace-time Python over static structure — the unclamped node names, every degree included (Formulation). Clamped nodes are never relaxed.
- `derive_states(params, state, clamps, structure)`: iterate `structure.schedule`; per visit `gather_inputs` → `scale_inputs` → `node_class.forward_from_error(..., is_clamped=(name in clamps))`. Warm-start and truncation semantics, documented on the method: each inference step starts from the carried `GraphState`, so a cycle member's first visit reads the previous step's last-visit latent — effective traversal depth grows as T1 × U across a segment and `energy_of` is not a pure function of ε — and the gradient treats the carried latents as constants (truncation at the step boundary).
- `forward_value_and_grad`: build the relaxed pytree `{name: state.error for name in _relaxed_errors(...)}`; one `jax.value_and_grad(energy_of, has_aux=True)` where `energy_of` writes the relaxed ε leaves into the state, runs `derive_states`, and sums `jnp.sum(energy)` over `in_degree > 0` nodes (the same set as `train.py:155-161`, so equilibria match sPC). A source's ε gradient arrives purely through downstream z_mu, since E excludes in_degree == 0 nodes. Grads land in `latent_grad` by accumulation (preserves the accumulate-don't-replace invariant of `tests/test_inference_order.py:291-335`). Int token latents and clamps never enter the AD pytree; `GraphState` stays the sole fori_loop carry with invariant shapes/dtypes.
- `update_latents`: every relaxed node's ε steps by a new `compute_new_error` static method (`error*(1 − η·decay) − η·latent_grad`). `compute_new_latent` has no ePC caller; it raises `NotImplementedError` with the same message pattern as `InferenceSchedule`'s stubs.
- `begin_segment`: resync ε to the incoming latents — one sPC-direction sweep over `structure.node_order` recomputing each node's z_mu from its sources' carried latents via the template `forward` and setting ε := z_latent − z_mu (z_latent never changes during the sweep, so one visit per node suffices at any unroll degree). The first `derive_states` then reproduces the incoming z_latent exactly on DAGs: in schedule order, z_mu is recomputed at the already-preserved upstream latents, so z_mu + ε = z_latent node by node. This is the inverse direction of the bijection — a distribution initializer's random internal latents and a preceding sPC segment's final latent update both survive the handoff, where an identity `begin_segment` would overwrite every internal latent with a derive from stale ε (ε = 0 after every built-in initializer; one latent update stale after an sPC segment, whose per-step order is forward then update). On cyclic graphs cycle members are preserved at their first visit only; the unrolled parameterization has no exact inverse there. With `FeedforwardStateInit` on a DAG the sweep yields ε = 0, the paper's init. Cost: one forward pass per segment entry.
- `finalize_state`: one detached `derive_states` rebuild so the final state satisfies `z_latent = z_mu + ε` with energies at the final point — the paper's weight rule; `compute_local_weight_gradients`, the train-loop energy, `eval_step`'s readout, and all dashboard readers of `.error` then work unchanged.

`NodeState` schema is untouched; `error`'s docstring (`core/types.py:118`) is updated: "Prediction errors (z_latent − z_mu); under `EPCInference` the first-class relaxed variable ε, with z_latent derived as z_mu + ε." `latent_grad`'s docstring (`core/types.py:120`) and field comment (:127) are updated in the same change: under sPC the one-hop accumulated dE/dz_latent; under `EPCInference` ∇_ε E through the full derived forward.

## Component 5 — InferenceSchedule

In `fabricpc/core/inference.py`:

```python
inference = InferenceSchedule(
    EPCInference(eta_infer=1e-2, infer_steps=5),   # cheap global steps to near-equilibrium
    InferenceSGD(eta_infer=0.05, infer_steps=20),  # refine on the true arbitrary-graph energy
)
```

Chained execution contract:
1. Node states are initialized once, by the graph's configured initializer, before the first segment (the existing path); no segment re-initializes. Every `in_degree == 0` node leaves init with error = 0 (Formulation).
2. Each solver receives z_latent, z_mu, and error exactly as the previous segment (or the initializer) left them; the schedule itself never resyncs. A solver's own `begin_segment` then establishes its parameterization from that state — ePC resyncs ε := z_latent − z_mu at the carried latents (Component 4), sPC's is identity — so the incoming latents are preserved exactly, not merely taken as-is.
3. An ePC segment repeats `infer_steps` times: forward-project z_mu and derive z_latent := z_mu + ε in place along `structure.schedule` — later visits read already-updated upstream latents — then one global ∇_ε E and ε ← ε − η·∇_ε E (`compute_new_error`, with decay).
4. The next solver continues from the resulting state (after ePC's `finalize_state` rebuild).

- `__init__(*solvers)`: validates non-empty, all `InferenceBase`; stores `solvers=tuple(...)` in config (plain `MappingProxyType`, holds objects fine — `InferenceBase` is not `FrozenConfig`).
- `run_inference(self, ...)`: fold the state through each solver's `run_inference` in order under the handoff contract above (each applies its own `begin_segment`/`finalize_state`; both boundaries pass z_latent/z_mu/error as-is).
- `segments()`: flattens component `segments()` — nested schedules compose.
- `inference_step`/`compute_new_latent`: raise `NotImplementedError` directing callers to `segments()` — a schedule has no single per-step rule.

Tracking (`inference_tracking.py`): both variants iterate `structure.config["inference"].segments()`; per segment run `begin_segment` → `lax.scan`/Python loop of `inference_step` for that segment's steps → `finalize_state`; concatenate per-step metric stacks along axis 0 (metric structure is identical across segments). Single-solver graphs produce one segment and byte-identical output to today. This transitively fixes `AimTracker.track_inference_dynamics` (`trackers.py:406-470`) — `trackers.py:488` is a usage example inside `StateHistoryCollector`'s docstring, not executable code — and removes both `config["infer_steps"]` reads. Under a composed schedule the concatenated `latent_grad_norm` series (`inference_tracking.py:67-68`) carries each segment's own gradient semantics — sPC's one-hop dE/dz_latent, ePC's full-forward ∇_ε E; the tracking module documents the per-segment meaning.

Follow-up (2026-09-01, `docs/dev_plans/epc_convergence_init_precision_artifact.md`): `make_tracked_probe` compiles `initialize_graph_state` + `run_inference_with_history` into one jitted program — eager init and jitted tracking can select different cuDNN conv algorithms (TF32 vs FP32, per conv shape), recording a phantom step-0 energy up to ~1e-3 on unclamped nodes; the convergence-mode probes in `epc_spc_resnet18_compare.py` use it.

## Component 6 — Benchmark: ePC vs sPC on resnet18/CIFAR-10

Both wall-clock comparisons read off one sweep of T1 — ePC inference steps per minibatch (`infer_steps`), against sPC's `--spc_steps` inference steps per minibatch. All arms train the same `--num_epochs`, so each arm is one point (total training wall-clock, final test accuracy) and the T1 grid — not training checkpoints — sets the granularity of the time axis.

- `examples/resnet18_cifar10_demo.py`: `build_resnet18(...)` takes a required `inference: InferenceBase` replacing the `infer_steps`/`eta_infer` kwargs and the hardcoded `InferenceSGDNormClip` (:321-323); migrate `_create_mupc_model` and `run_single_mupc` (CLI behavior and docstring reference numbers unchanged); `make_optimizer` moves into the demo beside the builder, and the compare script imports both — no verbatim copy.
- New `examples/epc_spc_resnet18_compare.py` (importlib load of the demo builder, per the pattern at `examples/PC_backprop_compare.py:52-58`). CLI: `--mode {sweep,convergence}` (default `sweep`), `--n_trials 5`, `--num_epochs 2`, `--batch_size 256`, `--spc_steps 120`, `--spc_eta 0.1`, `--epc_eta` (default `None` → `EPCInference`'s default; convergence mode accepts a comma-separated list, sweep mode a single value), `--lr`, `--weight_decay`, `--epc_step_sweep 1,2,3,4,5,6,7,8,9,10,16,32,64,128,160`, `--track_steps 120`.
  - **sweep** (default) — one `PlannedMultiContrastExperiment` with an arm per T1 in `--epc_step_sweep` plus an sPC-`{spc_steps}` baseline arm, empty contrast family (the runner supplies the paired trial loop; `TrialResult.metric_value`/`train_time` already carry everything needed), all arms at the same `--num_epochs` and each arm's adamw warmup-cosine schedule from `num_epochs × len(train_loader)`. Grid: dense 1–10 where accuracy moves fastest, log-spaced 16–160 above. The grid must bracket sPC's wall-clock point: at the measured ~0.9× ePC/sPC per-step ratio, sPC-120's wall-clock lands near ePC T1 ≈ 130, hence the 160 top entry; extend past 160 if ePC-160 still finishes faster (the interpolation needs the sPC time point bracketed). Derived per-trial metrics: (i) **accuracy at equal wall-clock** — linear interpolation of the trial's ePC (train_time, accuracy) points at the trial's sPC train_time; (ii) **wall-clock to equal accuracy** — the smallest-T1 arm whose accuracy ≥ the trial's sPC accuracy, reporting its train_time and the ratio to sPC's. Both derived metrics are selection/interpolation statistics — wall-clock-to-equal-accuracy is a min over arms quantized to the T1 grid — so they are reported descriptively (per-trial values, mean ± SE), never tested; the `ContrastResult`/`DescriptiveDelta` split exists to keep exactly this kind of comparison out of the testing path. The docstring points to a confirmatory follow-up run with `contrasts=[("ePC-<T1>", spc_name)]` once the sweep identifies the T1 of interest. Chart (plotly, html+png output convention of `examples/scaling/scaling_analysis_plots.py:943-950`): x = T1 on a log axis; top panel y = final test accuracy with mean ± SE error bars over trials and the sPC baseline as a horizontal line with SE band; bottom panel y = total training wall-clock per arm with sPC's as the horizontal reference — the equal-wall-clock crossing and the time-to-equal-accuracy gap are both readable off the two panels. Written to `epc_step_sweep.html` always, and `epc_step_sweep.png` behind a kaleido import guard (the scaling script's `write_image` is unconditional and fails without kaleido; the compare script guards it). Cost: per-minibatch inference steps summed across arms = 455 (ePC grid) + 120 (sPC) ≈ 4.8× a lone sPC-120 run per trial.
  - **convergence** (single seed, no training): identical params/initial state for both solvers on one test batch; `run_inference_with_history` per solver. Reports **per-node** energy-vs-step, not the global sum: total energy is dominated by output-adjacent nodes (the energy imbalance reported in Pinchetti et al.'s PC benchmarking paper, arXiv 2407.01163), so a global curve can read as sPC near-convergence while deep nodes have received no gradient signal. Plot: log10 per-node energy vs step, one line per node colored by schedule depth, side-by-side sPC/ePC panels (per-node series come directly from the per-step history dicts). The global sum is retained for exactly one purpose: **E\*** = sPC's final total energy, with ePC's steps-to-reach ≤ E\* as the head-to-head criterion. `--epc_eta` accepts a comma-separated list here, reproducing the whole eta table in one invocation; reported step counts count ε updates (history index i is the energy after i updates). Wall-clock is the min over 5 post-warmup repeats, reported twice per solver: asymptotic ms/step (a full `--track_steps` run divided by its step count) and ms per update at T1 = 1 — ePC's `run_inference` brackets the fori_loop with the `begin_segment` and `finalize_state` forwards, so its small-T1 per-update cost exceeds what the asymptotic ratio implies, and the sweep's small-T1 arms must not be costed from the asymptotic number.

## File-by-file change list

Modified: `fabricpc/core/inference.py` (refactor + hooks + `InferenceSchedule`), `fabricpc/core/types.py` (`schedule` field + pytree + docstrings), `fabricpc/graph_assembly/graph_construction.py` (generalize `_topological_sort` with `unroll`; `GraphCycleError`; `first_occurrence_order`; `graph(unroll=...)`), `fabricpc/graph_assembly/__init__.py` (exports), `fabricpc/graph_initialization/state_initializer.py` (pass 2 :269 → `structure.schedule`; shared z_mu ← z_latent, error ← 0 post-pass for `in_degree == 0` nodes in `initialize_graph_state`; docstrings), `fabricpc/training/train.py` (`evaluate_transformer` predictions and external-energy term read z_mu, :781/:800, matching `eval_step` and `evaluate_pcn`), `fabricpc/core/mupc.py` (dup-raise hardening + docs), `fabricpc/nodes/base.py` (contract split: `predict`/`energy` + `pair_error`/`pair_latent` + `forward_with_aux`/`forward`/`forward_from_error` templates; `energy_functional` deleted into the default `energy()`; unclamped-readout forcing branch in `forward_and_latent_grads` deleted, :514-529; source branch delegates to `forward`), the node modules `fabricpc/nodes/linear.py`, `identity.py`, `skip_connection.py`, `linear_residual.py`, `convolutional.py`, `pooling.py`, `storkey_hopfield.py`, `transformer.py`, `transformer_v2.py`, `linear_explicit_grad.py` (forward → predict migrations; StorkeyHopfield `energy()` override with the `aux=None` fallback and its stale module docstring corrected — it described the deleted `energy_functional`/`accumulate_hopfield_energy` path; `forward_with_aux` adaptation of the analytic overrides), `examples/jpc_fc_resnet_compare.py` + `tests/test_external_custom_node.py` + `tests/test_mupc.py` (custom-node migrations; `ScaledSumNode`'s energy weighting becomes an `energy()` override), `docs/user_guides/06_custom_nodes.md` (rewritten around the two-method contract) + `docs/user_guides/10_api_nodes.md` (contract split; aux pattern and anti-pattern), `fabricpc/utils/dashboarding/inference_tracking.py` (segment iteration), `fabricpc/core/__init__.py` (export `EPCInference`, `InferenceSchedule`), `examples/resnet18_cifar10_demo.py`, `examples/mnist_cyclic_graph.py` (explicit `unroll`; correct the summary print at :143 — it claims 6 nodes / 7 edges, the graph builds 5 / 5; the trials help text said 10, the default is 3), `examples/storkey_hopfield_recall.py` (rewrite the readout-constraint comment at :137-142, whose `base.py:382-400` reference is stale; correct the docstring — recall reads the Hopfield node's own z_latent at :277, not a plain projection, and the now-relaxing unclamped output feeds gradients back into the Hopfield latent during recall, so recorded numbers shift), `tests/conftest.py` (extend the existing `with_inference` helper (:23-27) to accept an inference object: `with_inference(structure, inference=None, **kwargs)`), migrated tests (`test_fabricpc.py`, `test_ndim_shapes.py`, `test_auto_node_grad.py`, `test_inference_order.py`), `docs/user_guides/12_api_inference.md` (overview phase 3 says ePC updates ε, not z_latent; tuning table covers ePC's `infer_steps` default; extension-point list includes `begin_segment`/`finalize_state`), `docs/user_guides/04_building_models.md`, `docs/user_guides/03_how_predictive_coding_works.md` (the sPC/ePC distinction restated — sPC ignores `structure.schedule` and minimizes the exact graph energy, ePC minimizes the unrolled approximation; "cycles that are not unrolled" can no longer be constructed, `GraphCycleError`), `docs/user_guides/05_initialization_and_scaling.md` (drop the "forward initialization not applicable" row for cyclic graphs — `FeedforwardStateInit` now walks the schedule), `pyproject.toml` (register the `slow` marker), `CHANGELOG.md` (the breaking-change entries plus a full `### New` block — `EPCInference`, `InferenceSchedule`, the `begin_segment`/`finalize_state` hooks, segment-aware tracking, `graph(..., unroll=U)`, the benchmark — and an updated release header).

New: `fabricpc/core/inference_epc.py`, `examples/epc_spc_resnet18_compare.py`, `tests/test_topological_schedule.py`, `tests/test_node_contract.py`, `tests/test_inference_epc.py`, `tests/test_inference_schedule.py`.

Sequencing: (1) dispatch refactor + tracking segments + test migrations — pure refactor, full suite green; (2) generalized `_topological_sort` + `schedule` field + consumer migrations + its tests; (3) node contract split (Component 3, incl. the `forward_from_error` template) — pure refactor, full suite green with no test-expectation edits (sPC bit-identity), including the guide 06 rewrite, `CHANGELOG.md`, and the audit tests; (4) `EPCInference` + tests; (5) `InferenceSchedule` + tests + docs/exports; (6) resnet18 refactor + compare script.

## Test plan

- `tests/test_topological_schedule.py`: on DAGs the generalized `_topological_sort` equals the legacy Kahn order across insertion permutations, with and without `unroll`; cyclic + `unroll=None` raises `GraphCycleError` naming `{a,b,y}` on `x→a⇄b→y`; `unroll=2` → `("x","a","b","a","b","y")`, `unroll=1` single-visit, `node_order` invariant; `graph()` rejects `unroll < 1`, bools, and incomplete/unknown schedules; determinism; SCC topology coverage beyond 2-node cycles — a 3-node SCC, overlapping and disjoint cycles, a multi-entry SCC, an entry-less SCC (the `entries or [members[0]]` fallback), a clamped node inside a cycle.
- `tests/test_state_initializer.py` additions: feedforward-through-cycles matches a hand-computed propagation on `x→a⇄b→y` (U=2) — hand-computed rather than a replay of pass 2, so a wrong schedule cannot pass both sides — plus a muPC-scaled variant exercising `scale_inputs`; U=2 ≠ U=1 result (propagation proof); every `in_degree == 0` node — clamped and unclamped, under each of `GlobalStateInit`, `NodeDistributionStateInit`, and `FeedforwardStateInit` — leaves init with z_mu == z_latent and error == 0, asserted against a garbage-writing initializer that puts nonzero z_mu/error on sources so the invariant assertions are falsifiable; existing DAG suite unchanged.
- `tests/test_mupc.py` additions: dup-order raise; cyclic graph + muPC + `graph(..., unroll=U)` yields non-None scalings with correct `K_slot`; merge-in-cycle L identical for U ∈ {1, 5}; all existing muPC tests pass unchanged.
- `tests/test_node_contract.py`: audit — every registered node class resolves `forward`, `forward_with_aux`, `forward_from_error`, `pair_error`, and `pair_latent` to `NodeBase`'s (the templates are not override points and Python cannot mark them final), with the audited class set anchored to the package export list rather than a count floor; source guard — `forward()` on an `in_degree == 0` node returns z_mu = z_latent cast to z_mu's dtype, error = 0, functional energy (regression for the consolidated solver branch); source-path coverage for every energy-overriding node — `energy(..., aux=None, ...)` must not raise (the StorkeyHopfield base-term fallback); the Hopfield attractor formula value-pinned against a hand computation; `forward_with_aux` surfaces the aux the analytic path consumes (`Linear` pre_activation; StorkeyHopfield `(W, strength)`); a `predict` returning a bare array fails at trace time (tuple unpack). sPC bit-identity is pinned by the full suite passing with no expectation edits at sequencing step (3); the existing `test_auto_node_grad.py` explicit-vs-autodiff equivalence doubles as the aux-flow regression.
- `tests/test_inference_epc.py`: ε=0 ⇔ feedforward init (`begin_segment`'s resync yields ε = 0 from a feedforward-initialized state; `derive_states` preserves z_latent); gradient correctness vs closed form on a 2-layer linear chain (ε_h + Wᵀ·output-residual) and vs a hand-rolled `jax.grad`; energy decreases over steps; **sPC equivalence**: small convex DAG including an unclamped top-down prior, run to convergence — per-node z_latent/z_mu/energy agree, stationarity holds at the shared fixed point, and `compute_local_weight_gradients` including bias gradients agrees at both fixed points (the prior is the case that distinguishes ε-relaxed sources from frozen ones); nonlinear equivalence — tanh chain with a CrossEntropy-clamped output, run to convergence, slow-marked; **muPC fixed-point divergence** pinned, slow-marked — sPC+muPC and ePC+muPC settle to different fixed points because ePC omits `jacobian_gain`/`self_grad_scale`; the graph needs an unclamped node feeding a tanh node, since `jacobian_gain` derives from the edge's target activation; `begin_segment` preservation — a distribution-initialized state survives segment entry (the first derive reproduces the incoming z_latent) and a zero-step segment hands state through unchanged; `forward_from_error` template branch coverage, clamped/unclamped × source/internal (CrossEntropy-clamped output, pure-Gaussian readout staying at ε = 0, StorkeyHopfield readout retaining nonzero attractor energy and receiving the attractor gradient, int-token EmbeddingNode graph); cyclic smoke under the unrolled schedule (jit-compiles, finite decreasing energy; asserts the warm-start semantics — the derived state after two steps at U=1 differs from one step at U=2); muPC-scaled graph z_mu matches manual scaling; insertion-order independence of one-step grads; `z_latent == z_mu + error` after `run_inference`.
- `tests/test_fabricpc.py` additions (sPC side of the readout fix): an unclamped pure-Gaussian readout keeps error = z_latent − z_mu with `latent_grad` untouched through inference (regression for the deleted `base.py:514-529` forcing) and predictions still read z_mu; the readout's z_latent converges to z_mu over inference — the property every z_mu-reading eval site now rests on (`evaluate_transformer` regression); a StorkeyHopfield readout retains its nonzero attractor energy and relaxes onto the attractor.
- `tests/test_trainer.py` (readout-fix fallout, applied after the v0.5.0 trainer consolidation landed): the readout-branch deletion invalidates the eval non-negativity assertions. With the target free during `evaluate()`, the readout's `CrossEntropyEnergy` term −Σᵢ zᵢ·log μᵢ — z the readout's free z_latent, μ its softmax z_mu — is linear in z and unbounded below, so reported eval energy is signed and descends with every inference step. `test_train_and_evaluate_smoke` drops `eval energy >= 0` (finiteness and `target_energy >= 0`, one-hot y scored against z_mu, remain); `test_eval_energy_matches_graph_energy` replaces `expected > 0` with a non-vacuousness check, keeping the parity assertion metric ≡ `graph_energy`/batch_size; the docstrings claiming a free feedforward output sits at its zero-error fixed point are rewritten — the readout relaxes.
- `tests/test_inference_schedule.py`: `segments()` flattening incl. nesting, and a nested schedule executed end-to-end bit-identical to its flattened equivalent; single-solver schedule ≡ plain solver (allclose); ePC→sPC schedule ≡ manual sequential calls; z_latent/z_mu/error pass the segment boundary bit-identical (the schedule never resyncs; ePC's `begin_segment` does); runs inside `jax.jit(train_step)`; the handoff-energy test executes the actual schedule and asserts strict descent (a build-only test passes on zero progress); tracking parity including `run_inference_with_full_history` under a schedule (Σ steps rows, final state ≡ `run_inference`); raising stubs raise. New solvers imported from the public `fabricpc.core` surface across the new test files, not private module paths.
- Regression: full suite green after step (1) with only the nine migrated call sites touched; `test_doc_snippets.py` gates the rewritten user-guide examples.

## Verification

1. `pytest tests/` green at each sequencing step; step (3), the contract split, passes with no test-expectation edits — sPC outputs bit-identical.
2. `python examples/mnist_cyclic_graph.py` runs with the explicit `graph(..., unroll=...)` argument and trains. Expected behavior drift, not a regression: today `node_order == ("pixels",)` makes `FeedforwardStateInit` pass 2 a no-op, so all five nodes start from their `latent_init` draws with z_mu = 0; after migration they get true feedforward init through the cycle and muPC can attach scalings, so the loss curve shifts.
3. `python examples/resnet18_cifar10_demo.py` (unchanged defaults) reproduces the documented single-run behavior.
4. `python examples/epc_spc_resnet18_compare.py --mode convergence --epc_eta 0.001,0.01,0.03,0.1` — per-node energy-vs-step panels, the full eta table of ε updates to reach ≤ E* in one invocation, and min-over-5-repeats wall-clock (asymptotic ms/step and ms per update at T1 = 1) for both solvers, on one batch (minutes).
5. `python examples/epc_spc_resnet18_compare.py --mode sweep --n_trials 5` (~5 h on the reference 3090) — accuracy-vs-T1 chart written to `epc_step_sweep.html`/`.png`, plus accuracy-at-equal-wall-clock and wall-clock-to-equal-accuracy tables with per-trial values and mean ± SE (descriptive — Component 6); paste the tables into the script docstring per house convention.

## Alternatives considered

- **ε storage**: chosen `NodeState.error` (write-mostly today; verified readers are dashboards + `LinearExplicitGrad`). Rejected: extra fori_loop-carry dict (invisible to tracking/handoff/weight path, fails the first-class requirement); new `epsilon` field (schema churn through every constructor, redundant with `error`).
- **Gradient computation**: chosen one global `jax.value_and_grad` over a `{field: {node: array}}` dict. Rejected: per-node grads hand-stitched through the schedule (re-derives reverse AD, wrong for repeated visits, reintroduces the decay ePC removes); grad w.r.t. whole `GraphState` (differentiates int latents and clamps; needs masking).
- **Node derive rule**: chosen one rule for every node — z_latent = z_mu + ε, the clamp deciding the free side — with all unclamped nodes ε-relaxed regardless of degree. Rejected: a five-way partition by degree and clamp role — its latent-relaxed source handling froze top-down priors (`derive_states` overwrote the relaxed z_latent leaf with the constant z_mu + error, zeroing the prior's gradient).
- **InferenceSchedule type**: chosen `InferenceBase` subclass with `segments()` + raising per-step stubs. Rejected: separate protocol (breaks `graph()`'s contract, two type surfaces); `graph(inference=[...])` list support (composition in the wrong layer, no nesting).
- **Dispatch fix**: chosen classmethods + instance `run_inference`. Rejected: threading `cls` parameters (noisy, still mis-passable); instance methods everywhere (abandons the static-pure-function house style without need).
- **Schedule location**: chosen new `schedule` field beside `node_order` (each consumer takes the semantically right projection; logging readers untouched; muPC misuse blocked by dup-raise). Rejected: replace `node_order` (every one-visit consumer must dedup); stash in `structure.config` (core topology data stringly-typed).
- **Unroll algorithm**: chosen Tarjan SCC + Kahn-on-condensation + entry-first intra-SCC BFS (minimal schedule length, exact DAG equality, deterministic). Rejected: repeat full sweep U times (multiplies ePC forward cost on the acyclic majority); feedback-arc-set removal (back edges never carry information, fails the requirement).
- **Schedule API**: chosen generalizing `_topological_sort` in place with an `unroll` argument surfaced as `graph(..., unroll=...)`. Rejected: `TopologySchedulerBase`/`DAGScheduler`/`UnrolledCycleScheduler` class hierarchy (a new module, base class, and config-stored object carrying a single integer; the unroll algorithm is unchanged, only its packaging).
- **Schedule ownership** (binding time): chosen graph-owned — `graph(..., unroll=U)` binds one integer read by both consumers, `FeedforwardStateInit` at initialization and `EPCInference.derive_states` at every inference step, so initialization is the derived forward at ε = 0 on the identical unrolled schedule; changing U rebuilds the graph. Rejected: solver-owned schedule (an `EPCInference` constructor argument) — the initializer needs the same schedule, and two owners can disagree, breaking the ε = 0 ⇔ feedforward-init correspondence.
- **Initializer z_mu fix placement**: chosen one shared post-pass in `initialize_graph_state` after the dispatched `initialize_state` returns — one implementation point covers all three concrete initializers and any future one, beside the existing shared validation (`_validate_clamp_dtypes`). Rejected: repeating the copy inside each initializer (three copies of one loop); fixing `FeedforwardStateInit` alone (leaves `GlobalStateInit`/`NodeDistributionStateInit` invalid under ePC, which reads the initialized z_mu directly).
- **Readout handling**: chosen ordinary ε-relaxation with energy kept as `forward()` assigns — one code path in each solver, Hopfield readouts supported. Rejected: forcing error = 0 / energy = 0 (hard-codes the pure-Gaussian assumption; a Hopfield readout's attractor energy is discarded and it never settles onto its attractor); keeping a skip-the-derive special case for computation savings (negligible saving, one more path to maintain).
- **Segment handoff**: chosen `begin_segment` ε resync — one `forward()` sweep at the carried latents per ePC segment entry (Component 4). Rejected: as-is pass-through of z_latent/z_mu/error — the plan's original choice, reversed in review (B1). Every built-in initializer zeroes `error`, so identity handoff made ePC's first derive overwrite all internal latents with the feedforward pass (distribution initializers nullified on internal nodes), and after an sPC segment the adopted ε was one latent update stale, dropping sPC's final update. One forward pass per boundary buys exact preservation of any incoming state.
- **Node contract for the two dataflow directions**: chosen the predict/pair/energy split with base-owned templates (Component 3) — one `predict` per visit in both solvers, z_latent = z_mu + ε by construction for arbitrary predicts, one owner for the additive pair, custom energy terms first-class. Rejected: a second `forward()` call in `forward_from_error` (a predict that reads the node's own `NodeState` yields two different z_mu — the second call re-evaluates at the derived latent — and the returned state silently violates z_latent = z_mu + error; where the calls do coincide, correctness rests on XLA CSE merging them, backend behavior rather than contract, and the traced jaxpr and eager-debug cost double regardless); `energy_functional` directly (silently drops in-forward energy terms like StorkeyHopfield's); injecting a resolve-latent callable into `forward()` (every body must call it at the right point, so the boilerplate is retained rather than deleted, and a body that forgets breaks only ePC, silently); `forward()` returning a staged energy closure (a closure return from the data-in/data-out staticmethod contract; aux carries the same intermediates as plain data); an overridable pair, or a pair owned by `EnergyFunctional` (a non-volume-preserving pair breaks the ε↔z_latent equivalence; today's ε is unweighted everywhere — `EnergyFunctional` is noted as the future home if precision-weighted ε is wanted, not built).
- **Equal-wall-clock mechanism**: chosen the T1 sweep as the time axis — every arm trains the same `--num_epochs`, each T1 yields one (total wall-clock, final accuracy) point, both comparisons interpolate along the sweep, and `TrialResult.metric_value`/`train_time` already carry all the data. Rejected: per-epoch checkpoint curves with budget-matched epoch counts (ties the time axis to checkpoint/minibatch cadence, which is not the quantity under study, and needs a calibration pass plus a `TrialResult` extension); wall-clock stopping rule inside `train_pcn` (perturbs the shared trainer for one benchmark; nondeterministic epoch boundaries break trial pairing and warmup-cosine schedule construction); assumed step-count parity between ePC-10 and sPC-120 (per-step costs are close — a reference torch ePC ran ~20% slower per step than sPC — so ~10× fewer steps make ePC far cheaper per weight update, not wall-clock-equal, and the ratio must be measured); bespoke trial loop in the compare script (duplicates `PlannedMultiContrastExperiment`'s pairing-by-seed machinery).

## Post-review status (2026-08-22)

`docs/dev_plans/epc_pr_review.md` reviewed the full branch diff; every recommended change is applied, and the sections above describe the as-implemented design. Review-driven reversals and refinements folded into the body: the `begin_segment` ε resync replacing the identity handoff (Component 4, Alternatives "Segment handoff"); the `predict` z_latent read prohibition (Formulation, Component 3); the η = 1e-2 default justified by the measured convergence table as a robustness margin, replacing the unsupported overshoot claim (Formulation, Component 4); descriptive sweep statistics replacing hand-rolled t-tests (Component 6); the 160-topped T1 grid (Component 6); ε-update step counting and min-over-repeats timing with a separate T1 = 1 reading (Component 6); bool-rejecting `unroll` (Component 1); the source error ← 0 post-pass (Component 1); `evaluate_transformer` reading z_mu (Component 3); the aux = None contract (Component 3); and the review's test-gap list (Test plan).

Two deviations from the review's suggested remedies, both forced by facts found during implementation (also recorded at the top of the review file):

- **A2** — the review suggested `StorkeyHopfield.energy` recompute `(W, strength)` — the Storkey coupling matrix and the scalar weight on its attractor term — from params when `aux is None`. That is impossible: the param initializer gives every source node empty params (`params_initializer.py:39-41`), so `W` does not exist on a source. The fix falls back to the base PC energy on `aux is None` (matching main's behavior), and the `energy()` contract now states that overrides must tolerate it, with a test.
- **C2** — the derived sweep metrics (interpolated accuracy-at-equal-wall-clock, min-over-arms wall-clock-to-equal-accuracy) cannot be honest planned contrasts, so the hand-rolled t-tests were removed in favor of descriptive mean ± SE summaries, with the docstring pointing to `contrasts=[("ePC-<T1>", spc_name)]` for a confirmatory follow-up run.

Remaining before the PR merges:

- The PR description must state that cyclic graphs previously got partial-order feedforward init (cycle members skipped), so training curves on cyclic graphs shift even where tests hold (Verification item 2).

## Review round 2 (2026-09-03 to 2026-09-08): ePC oracle, regime test, and stability analysis

The second review round covered two things the first round had not: an independent check that `EPCInference` and the state-based solvers reach the correct equilibrium, and a way for a user to choose `eta_infer` and `infer_steps` inside a stable regime on any graph. It ran in three iterations, recorded here in order so the design decisions can be read without the intermediate documents:

1. The reviewer's requests after the resnet18 convergence figure, and the response that built an exact linear oracle, a regime label, a power-iteration eigenvalue estimator, and an analysis script.
2. A critical review of that implementation. Verdict: the correctness verification is sound; the tuning diagnostics are defective.
3. The replacement design: a Lanczos spectrum estimator, a structured `Regime` verdict, a `RegimeProbe` training callback usable on any graph, and four GPU control runs with a reading rule fixed in advance.

Iterations 1 and 2 are implemented on this branch. Iteration 3 is designed and sequenced; its first two commits are independent of the trainer, and the rest waits on a separate PR that gives `train(iter_callback=...)` a context object carrying the parameters (Status, below). The measurements behind every number here are in the technical report `docs/reports/epc_regime_and_stability_report.md`.

### Symbols

| Symbol | Meaning |
|---|---|
| z_t, μ_t, ε_t | node t's latent (`NodeState.z_latent`), its prediction from in-edge sources (`NodeState.z_mu`), and the prediction error z_t − μ_t (`NodeState.error`), the variable `EPCInference` relaxes |
| p_t, E | node t's Gaussian precision (1.0 unless set), and the total energy Σ over in_degree > 0 nodes of ½·p_t·‖z_t − μ_t‖²; a cross-entropy output replaces its quadratic term |
| H_z, H_ε | Hessians of E in latent coordinates and in error coordinates, at the feedforward point ε = 0 |
| g0 | ∇_ε E at ε = 0, which equals the backprop activation gradient at every node |
| excited mode | an eigenvector of H_ε along which g0 has a nonzero component; the only modes gradient descent from ε = 0 moves |
| λ_max, λ_min | the largest and smallest eigenvalues among the excited modes |
| J, S, r, d_y | on a linear chain: J maps the stacked hidden ε to the output prediction μ_y; S = I + JJᵀ (Innocenti et al. 2024, Theorem 1); r = y − μ_y is the feedforward output residual; d_y is the output width |
| σ_max(J) | the largest singular value of J, so λ_max = 1 + σ_max(J)² at unit precision |
| η, T | `eta_infer`, `infer_steps` |
| f(λ), f_max | relaxed fraction 1 − (1 − ηλ)^T of a mode with eigenvalue λ after T steps; f_max = f(λ_max) |
| θ_k, w_k | the k-th Ritz value (an eigenvalue of the Lanczos tridiagonal matrix T_k) and the fraction of ‖g0‖² that Ritz mode carries; Σ_k w_k = 1 |
| f̄ | gradient-weighted relaxed fraction Σ_{θ_k > 0} w_k·f(θ_k) / Σ_{θ_k > 0} w_k over the positive Ritz modes |
| α_j, β_j | the Lanczos recurrence coefficients, the diagonal and off-diagonal of T_k |

### Background: what sets ePC's regime

Every FabricPC run starts inference at the feedforward state, ε = 0. Near that point E is a quadratic in ε, and gradient descent on a quadratic splits into independent modes along the eigenvectors of H_ε. One step multiplies a mode's distance to its minimum by (1 − ηλ): for 0 < ηλ < 1 the mode moves part of the way on the same side; for 1 < ηλ < 2 it overshoots to the other side but ends closer; for ηλ > 2 it ends farther away than it started. After T steps a mode has closed the fraction f(λ) = 1 − (1 − ηλ)^T of its distance. Only the excited modes move at all.

Two regimes and one bound follow. When η·T·λ ≪ 1 on every excited mode, the T-step result is ε ≈ −η·T·g0, and the local weight gradients computed from those errors are backprop's, hidden layers scaled by η·T and the output layer unscaled (Goemaere et al., Theorem C.9). That is the backprop-like regime. When every excited mode has relaxed, ε is the PC equilibrium and the output error is r S⁻¹. That is the PC-equilibrium regime. At every T, η·λ_max < 2 is required for the iteration not to diverge.

On a linear chain at unit precision H_ε = I + JᵀJ and g0 = Jᵀr, so the excited modes are the d_y directions of the row space of J and their eigenvalues are eig(S). λ_max = 1 + σ_max(J)² grows with the product of the downstream weights, and that product grows during training, so a fixed η eventually crosses the bound.

### Iteration 1: the reviewer's requests and the first response

**Requests.** The reviewer replied to the observation bullets that accompanied the resnet18 convergence figure (`examples/epc_spc_resnet18_compare.py --mode convergence`) with three groups of requests.

- Test-suite invariants: a linear oracle that returns the exact equilibrium state and energy for a range of linear networks, both `EPCInference` and the state-based solvers checked against it, and the oracle itself checked against hand-computed numbers or Innocenti et al. 2024, Theorem 1. A warning that 1-step ePC is backprop with gradients scaled by `eta_infer`.
- Analytical questions: why sPC struggles with deep layers; what sets the equilibrium energy spacing across layers; ePC stability in deep networks and how to pick the largest stable `eta_infer`; whether the training collapses also occur under muPC (they do; the resnet18 demo is a muPC model); how many steps sPC needs to reach equilibrium and why oracle checks should stay at five layers or fewer.
- Warning adequacy: whether the existing caution against `infer_steps=1` said the right thing and appeared where users would see it.

**Facts established from the existing data** (the 2-epoch sweep in the Appendix and the six 100-epoch logs `sweep_eta{0.001,0.01}_steps{1,2,5}.log` in the project root):

- The ePC paper (Appendix C.3, Theorem C.9) gives two backprop-equivalence conditions: T = 1 yields backprop gradients scaled by η, and η·T ≪ 1 yields backprop gradients scaled by η·T. The second condition carries an implicit O(1) Jacobian scale; the graph-dependent form is η·T·λ_max ≪ 1.
- In the sweep, accuracy departs from the small-η·T limit (38.8%) at η·T ≈ 0.01 to 0.02 for every η tested and reaches the PC plateau (about 31%) by η·T ≈ 0.15. The library defaults `EPCInference(eta_infer=1e-3, infer_steps=5)` sit at the backprop value at 2 epochs (38.70% against 38.84% at T = 1), and the 100-epoch demo at T = 1 (76.73%) matches the backprop trainer on the same graph (77.11%).
- Over 100 epochs, (1e-3, 1) reached 76.73% and (1e-3, 2) 75.76%; the defaults (1e-3, 5) reached 54.76% at epoch 10 and 9.68% at epoch 20; every (1e-2, T) cell was at chance by epoch 10. Collapse is a training-horizon effect: the weights grow, λ_max grows with them, and a fixed η eventually exceeds 2/λ_max.
- The existing caution ("Use infer_steps > 1") keyed on the wrong criterion, since T = 5 at η = 1e-3 is equally backprop, and the tuning table recommended `infer_steps` 1 to 5 two paragraphs below it.

**Three claims corrected by the design review of the oracle math.** Before: an unclamped source node contributes a curvature floor in ε coordinates. After: it contributes none; in the quadratic form it has columns but no residual row, because a source has no energy term. Before: ePC is better conditioned than sPC. After: the statement holds per regime; λ_min(H_z) decays with depth even for benign weights (sPC's slow mode), while λ_max(H_ε) grows with the downstream weight products (ePC's shrinking stability bound). Before: 1-step weight gradients equal η × backprop. After: the ε identity ε_1 = −η·g0 is exact, but the weight-gradient identity holds only to first order in η·λ_max, because `finalize_state` re-derives the latents before the local weight gradient is taken; the paper's Case 1 proof evaluates ∂ŝ/∂θ at the unperturbed point.

**User decisions (2026-09-03 and 2026-09-04).** Keep the defaults and document the regime at every touch point. No `warnings.warn`; the demos print the regime instead. The oracle lives in a package module shared by tests and the analysis script. The regime label is graph-aware and takes λ_max. Test the collapse hypothesis by tracking λ_max during resnet18 training behind a GPU flag, not on a CPU MNIST MLP, and let that tracking decide an adaptive-rate follow-up. Build the weight-gradient parity test in this work rather than deferring it to the mean-gradient-normalization plan; that normalization shipped in release 0.5.1 (2026-09-07) as `pc_weight_gradients` and `grad_denominator`, and the parity test runs through it.

**Built.**

- `fabricpc/utils/linear_pc_oracle.py`. Part one is the exact-equilibrium core in NumPy float64: it assembles E = ½‖A z_free − c‖² over the stacked free latents from params, edges, muPC forward scales, the IdentityNode scale, biases, and precision, and never calls node or solver code. `linear_equilibrium` solves by least squares; `theorem1_energy` gives the chain closed form E* = ½·r S⁻¹ rᵀ; `epsilon_hessian` gives H_ε = MᵀAᵀAM with M = (I − B)⁻¹ the ε → z map and B the strictly lower-triangular edge map; `stability_bound`, `excited_eigenvalues`, `relaxed_fraction`, and `steps_to_contract` are the diagnostics; `validate_linear_gaussian` rejects nonlinear, non-Gaussian, and cyclic graphs. Part two is `top_epsilon_eigenvalue`: power iteration on Hessian-vector products through the solver's own ε-energy, usable on any graph the solver accepts.
- `fabricpc/core/inference_epc.py`. `error_energy` is the ε-energy closure extracted from `forward_value_and_grad` (same ops, bit-identical gradients) so that Hessian-vector products can be taken through it. `regime_label(lambda_max) -> str` bands on f_max: below 0.1 "backprop-like", 0.1 to 0.9 "partially relaxed", above 0.9 "near PC equilibrium", and "unstable" when η·λ_max > 2.
- Tests. `tests/test_linear_pc_oracle.py`: oracle self-checks, including a hand-built scalar chain (x = 1, W1 = 2, W2 = 3, y = 1 gives z_h* = 0.5, E* = 1.25, S = [[10]]), Theorem 1 against the least-squares energy, the precision-weighted error pull-back, and the explicit H_ε form; both solvers reach the oracle on twelve graph shapes (chains to depth 4, biases, precisions, fork-merge, clamped internal node, unclamped prior source, unclamped readout, muPC chain); stability brackets at 0.95× and 1.05× of 2/λ_max for both solvers on the depth-3 chain, which pin the gradient's scale and not only its direction; the Hessian-vector product against H_ε. `TestBackpropCorrespondence` in `tests/test_inference_epc.py`: g0 equals a hand-written backprop recursion; after one step `error == −η·g0` exactly; the local weight gradients through `pc_weight_gradients` match a backprop reference divided by the same `grad_denominator`, the first hidden layer exactly (its input is the clamp) and the downstream layers with a remainder linear in η.
- `scripts/epc_analysis.py`. CPU sections under two minutes: `backprop_regime` (the O(η) approach of 1-step gradients to η × backprop; the one-eigenvalue fit of the sweep), `equilibrium_profile` (per-layer equilibrium energies from the oracle against sPC's transient), `convergence_spectra` (H_z and H_ε spectra and steps to contract against depth), `stability` (λ_max against weight scale and depth; a gelu bracket at 0.9× and 1.1× of 2/λ_max). GPU opt-ins: `--resnet18` (λ_max at init on the demo graph) and `--track_lambda_max N` (λ_max every N updates during demo training).
- Documentation: a "Backprop regime" paragraph and tuning-table rows in `docs/user_guides/12_api_inference.md`, the troubleshooting FAQ, the demo and compare-script docstrings with the 100-epoch table, and the CHANGELOG.

**Measured** (2026-09-04, one RTX 3090, batch-summed weight gradients; epochs 1-indexed):

| Quantity | Value |
|---|---|
| λ_max at init, muPC resnet18, 64-sample test batch | 16.4, so 2/λ_max = 0.12 |
| One eigenvalue fitted to the 45 sweep cells with η ≤ 0.01 | λ_eff = 12.0, rms residual 0.05 in normalized accuracy |
| Defaults at init | η·T·λ_max = 0.08; fastest mode 8% relaxed |
| (1e-3, 5) tracked every 50 updates | λ_max 15 to 27 through epoch 8; 51 at epoch 10; 84, 131, 471, 3508, 12539 at epochs 11 to 15; η·λ_max > 2 first at update 2700; chance from epoch 15 |
| (1e-2, 1) tracked | λ_max 101 at update 1100 and 220 at update 1150 (epoch 6); chance in epoch 7 |
| Steps to contract by 1e-3, linear chain, depth 20 | sPC 30 343, ePC 75 |
| Power iteration against the oracle's λ_max, depth-5 chain | relative error 6e-8 |
| First hidden layer's 1-step weight gradient | exactly η × backprop at every η; the O(η²) remainder appears only downstream |

### Iteration 2: critical review of the implementation

The reviewer re-derived the oracle math, ran the two test files, and read the two tracking CSVs. Two questions: does the work verify FabricPC's ePC code, and does it help a user pick η and T inside a stable regime.

**Verdict.** Correctness verification: strong. The oracle is independent of node and solver code; the assembly and the precision-weighted Theorem 1 re-derive correctly; the solver tests pin the equilibrium and the gradient scale, so two wrong solvers agreeing is ruled out; the 1-step identity and its first-order weight-gradient consequence are right. Tuning utility: weak, for the defects below.

**Defect 1: the estimator returns the largest-magnitude eigenvalue, not the largest positive one.** Power iteration from a random start converges to the eigenvalue of largest magnitude. On a graph with gelu activations and a softmax cross-entropy output, H_ε contains the term Σ_k (∂L/∂μ_k)·∂²μ_k/∂ε², the loss gradient weighting the second derivatives of the network map, and that term makes H_ε indefinite once the weights are large. The (1e-2, 1) tracking CSV records λ_max = −9398.9 at update 1200. Consequences: `regime_label(−9399)` returned "backprop-like", because a negative η·λ makes f_max negative and below the 0.1 band edge; the crossing detector tested η·λ > 2 and never fired; the demo would have printed a negative 2/λ_max. Power iteration also reports no convergence indicator, and for a positive-definite H its Rayleigh quotient is a lower bound on λ_max, so an unconverged value overstates the safe rate.

**Defect 2: the equilibrium band reads the fastest mode, and the condition is misprinted.** Before: "near PC equilibrium" was declared from f_max > 0.9, which says the fastest excited mode has relaxed 90%. After: equilibrium needs the modes that carry the gradient to relax, and the slowest of them sets the pace. Before: the class docstring, the guide paragraph, and the tuning table stated the condition as "η·T·λ_max ≳ 3/λ_min,excited", which as written multiplies by λ_max and divides by λ_min, a dimensional error. After: η·T·λ ≳ 3 on the gradient-carrying modes. The label also reported a "slowest" mode from a hard-coded floor eigenvalue of 1, wrong for any precision other than 1. On the resnet18 the excited band happened to be compact (λ_eff = 12 against λ_max = 16.4), so the label matched the sweep there; that is not general.

**Defect 3: the T = 1 danger threshold is too lenient.** On the unit-precision chain, gradient descent from ε = 0 leaves the output residual after T steps at r_T = (r/λ)·[1 + (λ − 1)(1 − ηλ)^T] along a mode with eigenvalue λ; the equilibrium value r/λ per mode is r S⁻¹. At T = 1 this is r_1 = (1 − η(λ − 1))·r. The output layer's local weight gradient is proportional to r_1, so it reverses sign along the top mode at η(λ_max − 1) > 1, before the iteration bound η·λ_max > 2. The hidden errors ε_T = f(λ)·ε* keep the sign of their equilibrium value for every ηλ < 2, so the reversal is confined to the output layer. For general T the flip condition is (1 − ηλ)^T < −1/(λ − 1), which can hold only at odd T because (1 − ηλ)^T ≥ 0 at even T. The data agree: in the (1e-2, 1) run λ_max passed 100 at update 1100, where η(λ − 1) ≈ 1, test accuracy fell that epoch from 43.2% to 40.6%, and the network was at chance one epoch later; in the 2-epoch sweep the (0.1, 1) cell at η·λ_max = 1.6 (η(λ_max − 1) = 1.5) collapsed and the (0.03, 1) cell at η·λ_max = 0.49 did not. Before: the docs described the T = 1 danger as "lands farther from equilibrium than it started" at η·λ > 2 and marked η·λ > 1 only as "overshooting". After: the output-gradient reversal at η(λ_max − 1) > 1 is the warning users act on; η·λ_max > 2 remains the iteration bound.

**Defect 4: the tracking tool cannot run on a user's graph.** `--track_lambda_max` re-implemented the training loop over `make_train_step` (about 120 lines) and reached into the demo module for the model builder, the optimizer, and the data loader. It existed because `train()`'s `iter_callback(epoch_idx, batch_idx, metrics)` cannot see the parameters. `epoch_callback(ctx)` can, but per-epoch access is too coarse: in the defaults collapse λ_max grew from 584 to 1608 within 50 updates.

**Defect 5: no control runs, and the paper's contrary result is undocumented.** Both tracked cells collapsed, so "η·λ_max crossed 2 before the collapse" is an ordering, not a cause. Separating "λ_max grows regardless of the solver and a fixed η eventually crosses the bound" from "ePC's relaxation beyond the backprop-like regime drives the weight growth" needs the cells that did not collapse: (1e-3, 1) and the backprop trainer. Goemaere et al. (Tables E.9 and E.10) trained ResNet-18 at the same error rate 1e-3 with zero error momentum, T = 5, SGD on the errors and Adam on the weights, for 25 epochs in the sweep and 50 in the final run, and reported no instability. Their network has batch normalization after every convolution, ReLU, weight decay searched in [1e-6, 1e-3], and a standard parameterization; the demo uses the muPC parameterization, no normalization layers (none exist in `fabricpc/nodes`), gelu, weight decay 1e-2, and a 100-epoch schedule. λ_max is a weight-scale quantity, so normalization and weight decay are levers the guide must name.

**Smaller defects.**

- Before: the demo docstring said λ_max "grew about threefold per epoch". After: the CSV shows growth of about 1.13× per epoch through epoch 10 (16 to 51) and a runaway from epoch 11.
- "Under Adam the η scaling is normalized away" holds only while η·|g| ≫ Adam's ε of 1e-8; at η = 1e-4, hidden-layer gradients of order 1e-7 are damped by ε. Without Adam, hidden layers learn η times slower than the output layer.
- At equilibrium the output error is r S⁻¹, which damps the learning signal along mode λ_S by 1/λ_S, up to 16× on this graph, a matrix rescaling a per-parameter optimizer cannot undo. This linear-chain mechanism is consistent with the 31% plateau trailing the 38.8% plateau, and the analysis script did not show it.
- The sPC+muPC equilibrium test cannot pin the gradient's scale, because a positive diagonal preconditioner shares the fixed point with plain gradient descent; only a stability bracket does.
- The oracle docstring did not say that cyclic graphs are outside it (the unrolled ε-energy is quadratic for linear nodes, but the warm-started carried latents make it not a pure function of ε), nor that a batch's λ_max is the per-sample maximum (`error_energy` sums per-sample energies, so H_ε is block-diagonal over samples).
- The one-eigenvalue sweep fit is a heuristic; the guide must not call it agreement.
- The parity test asserted linear scaling in η at one unknown λ_max; it now measures λ_max and asserts d(η) ≤ C·η·λ_max.

### Iteration 3: the replacement design

A second review of this design's first draft is folded into it. Its two CPU experiments are the Evidence below, and its findings set the estimator's second statistic (f̄, not λ_min), the breakdown guard, the indefiniteness rule, and the control-run set.

**Spectrum estimator: Lanczos from g0 with Ritz weights** (`fabricpc/core/epsilon_spectrum.py`, new).

Chosen: `lanczos_extremes(hvp, v0, iters)` runs the three-term Lanczos recurrence from v0 = g0 using only Hessian-vector products (`jax.jvp` of `jax.grad` of the ε-energy). Lanczos builds an orthonormal basis of span{g0, H g0, H² g0, …} and represents H_ε in that basis as a k × k tridiagonal matrix T_k. The Ritz values θ_k approximate eigenvalues of H_ε and converge fastest at both ends of the spectrum; the weight w_k is the fraction of ‖g0‖² that Ritz mode carries. One run therefore returns the three things the regime needs: λ_max (the bound 2/λ_max), λ_min (negative means indefinite), and the distribution {(θ_k, w_k)} of the gradient over the spectrum, from which f̄ is the relaxed fraction of the gradient that drives the weight update. A mode with θ_k ≤ 0 has no minimum to relax toward, so it is left out of f̄ and judged by its weight and growth instead. Three vectors are carried, so memory is independent of `iters`. `make_epsilon_spectrum(structure, iters=30)` compiles `(params, state, clamps, key) -> EpsilonSpectrum` (fields: `lambda_max`, `lambda_min`, `ritz_values`, `ritz_weights`, the residuals of both extremes, `negative_weight`, `gradient_norm`, `random_start`, `k`); when ‖g0‖ = 0 a random start is used and flagged, and the extremes then describe the full spectrum.

Why the weights, not only the extremes. On a linear graph g0 = Jᵀr has components only along the row space of J, so every w_k lies on one of the d_y eigenvalues eig(S), a compact band, and the two extremes describe it. On a nonlinear graph the second-derivative term in H_ε couples g0 to every direction, so nearly every mode is excited and λ_min is the bottom of the full spectrum, far from where the gradient sits. On the analysis script's gelu MLP (Evidence, second table) 943 of 1024 modes are excited, the smallest excited eigenvalue is 0.555, below the unit-precision floor, and 80% of the gradient's weight lies between 1.06 and 1.39. A band on λ_min would say the slowest mode is far from relaxed while the modes carrying the gradient are done. f̄ from 30 Lanczos steps equals f̄ from the exact 1024 × 1024 eigendecomposition to three digits. On a linear graph f̄ averages f over eig(S), so it agrees with the slowest-excited-mode condition whenever that band is compact, which is the resnet18's case.

Breakdown guard. When β_j ≤ √eps(dtype)·max(max_i |α_i|, max_i β_i), the Krylov space is exhausted and the recurrence freezes; frozen steps decouple from T_k as 1 × 1 blocks with zero weight, and `k` counts the valid steps. An exact-zero guard fails: on the linear test fixtures the Krylov space has dimension at most d_y = 3, so β_3 is zero in exact arithmetic but about 1e-6·|α| in float32 and 1e-15·|α| in float64, the recurrence continues on rounding noise, and the minimum Ritz value converges to the unexcited floor of 1.0 (Evidence, first table). λ_max is unaffected, which is why power iteration never exposed this.

Placement. The module is solver machinery: it calls `EPCInference.begin_segment` and `error_energy`, and no module in `fabricpc/core` imports from `fabricpc/utils` today. The linear oracle stays in `fabricpc/utils/linear_pc_oracle.py` and becomes NumPy-only; `top_epsilon_eigenvalue` and `regime_label` are deleted in one commit with every caller migrated (the demo's settings line, the compare script, the analysis script, and the oracle, parity, and regime tests).

Rejected:

- Shifted power iteration (a second pass on H + |λ_1|·I when the first returns λ_1 < 0). Ten lines and a sign fix, but it yields one number: no λ_min, no weights, so the equilibrium band would have to be dropped, and it converges slowly on a compact spectrum.
- Lanczos extremes with the equilibrium band on λ_min. Exact on linear graphs. On nonlinear graphs λ_min is the bottom of the full spectrum, near or below 1, so f(λ_min) > 0.9 would need η·T ≳ 3 and most sweep cells on the 31% plateau would be labelled "partially relaxed": at η = 0.03, T = 5 (31.4%), f(1) = 0.14. The label would contradict the data it was built to explain.
- Full reorthogonalization. Keeps k vectors of ε size to suppress ghost eigenvalues that duplicate converged extremes and split their weight; ghosts move neither the extremes nor the weighted sums.
- Module in `fabricpc/utils`. `EPCInference.regime(spectrum)` would make core import utils, the first such edge in the package.

**`Regime`: a structured verdict** (`EPCInference.regime(spectrum) -> Regime`, a NamedTuple).

| Field | Meaning |
|---|---|
| `eta_lambda_max`, `eta_T_lambda_max` | η·λ_max and η·T·λ_max |
| `unstable` | η·λ_max > 2: the top mode's distance grows every step |
| `output_gradient_reverses` | (1 − ηλ_max)^T < −1/(λ_max − 1) at unit precision (T = 1: η(λ_max − 1) > 1); always False at even T |
| `f_max`, `f_weighted` | f(λ_max) and f̄ |
| `band` | on f̄: "backprop-like" below 0.1, "near PC equilibrium" above 0.9, "partially relaxed" between |
| `negative_weight`, `growth_min` | the fraction of ‖g0‖² on negative-curvature modes, and (1 + η·max(0, −λ_min))^T, the growth of the most negative mode over T steps |
| `lambda_max`, `lambda_min` | the extremes |

`__str__` is the one-line label with precedence: `unstable`, then `growth_min > 1.1` ("indefinite: negative curvature carrying X% of the gradient grows Y× over T steps"), then the band with f̄, f_max, and the reversal note. Indefiniteness is judged by weight and growth, not by a boolean: at weight std 2.0 the gelu MLP is indefinite at init, with 16 negative eigenvalues down to −14.8 carrying 22.6% of ‖g0‖², yet one ePC step from ε = 0 is still exactly −η·g0; curvature sign enters at the second step, and over T steps a negative mode grows by (1 + η|λ_min|)^T, which at the defaults is 1.08. The reversal formula is the linear unit-precision chain's; on a nonlinear graph it is the local quadratic model's prediction at ε = 0, and the (1e-2, 1) control run tests it (the flag should fire at update 1100).

Rejected: a string label with an optional `lambda_min_excited` argument (keeps substring parsing in the analysis script and cannot carry the flags); a boolean `indefinite` that outranks the band (any negative eigenvalue would print "indefinite" over "backprop-like", telling a user whose setting is exact backprop that it is broken); bands on f_max (Defect 2); bands on f(λ_min) (above).

**`RegimeProbe`: the diagnostic as a `train()` callback** (`fabricpc/training/regime_probe.py`, new, exported from `fabricpc.training`).

Chosen: `RegimeProbe(structure, probe_clamps=None, *, every, inference=None, iters=30, key, csv_path=None)`. `on_iter(ctx)` runs every `every` updates: it initializes a graph state at `ctx.params`, runs the compiled spectrum estimator, records per-edge Frobenius weight norms, the training energy, and `inference.regime(spectrum)` when `inference` is an `EPCInference`. `probe_clamps` are fixed clamps built once by the caller (the demo uses a 64-sample test batch); `None` measures on the training batch. `on_epoch(ctx, accuracy=None)` records the epoch row. Readouts: `first_reversal()`, `first_crossing()`, `first_chance(chance, margin=0.05)`, `growth_phases()` (the per-epoch λ_max maximum and its ratio to the previous epoch), `write_csv()`, `summary()`. The CSV carries metadata columns first (`trainer, eta_infer, infer_steps, every, probe_batch`), then the spectrum, the regime flags, energy, accuracy, and one `wnorm:<edge_key>` column per weight, so readers take η, T, and the trainer from the columns and the filename is only a label. It works under `algorithm="backprop"` (spectrum and norms; regime columns empty). Cost, stated in the docstring: supplying `iter_callback` forces a device sync on every batch, not only on probed ones.

The probe consumes a separate PR that changes `iter_callback` to receive an `IterContext` with `params`, `opt_state`, `step`, `batch`, and `metrics`. The training step donates the parameter buffers, so `ctx.params` is valid only during the callback; the probe reads it there and stores floats. That PR migrates every three-argument caller in the same change (the dashboarding callbacks, the transformer demo, the Bayesian tuner, and the trainer tests).

Rejected: `epoch_callback` only (no trainer change, but per-epoch access misses a collapse in which λ_max grows 2.75× within 50 updates); a trainer flag `probe_every=N, probe=callable` (a second per-batch mechanism beside `iter_callback`, with its own sync and return semantics); keeping the script's custom loop (not reusable); η and T parsed from the CSV filename (the backprop file has neither); the name `LambdaMaxProbe` (names one column of a probe that records a spectrum, weight norms, and flags).

**Control runs: four cells and a reading rule fixed in advance.**

Chosen: `python examples/resnet18_cifar10_demo.py --num_epochs 30 --schedule_epochs 100 --augment --activation gelu --track_regime 50 --eval_every 1` for four runs, about 20 minutes each on the 3090: (1e-3, 5), the defaults collapse the report rests on; (1e-3, 1), the 100-epoch survivor; `--trainer backprop`; and (1e-2, 1), the run whose CSV holds the −9399. All four train under the 0.5.1 per-prediction gradient normalization, which the earlier tracked cells and the 100-epoch runs predate, so the reading rule compares the four runs with one another, and the (1e-3, 5) re-run, not the old CSV, becomes the report's collapse series. Reading rule: if λ_max and the weight norms grow at a comparable rate in the backprop and (1e-3, 1) runs as in the collapsing cells, growth is a weight-scale effect of this parameterization (no normalization, weight decay 1e-2) and the remedy is a rate that follows λ_max or weight-norm control; if they grow only in the collapsing cells, ePC's relaxation feeds the growth. Also recorded: whether λ_max grows in the runs that do not collapse; λ_min and `negative_weight` through the collapses; for (1e-2, 1) a positive λ_max at update 1200 and the reversal flag at update 1100.

Rejected: three runs without (1e-3, 5). That leaves the report's headline series in the old CSV schema, from the sign-ambiguous estimator and the batch-summed trainer, without λ_min or weight norms through the collapse.

**Evidence from the first-draft review (CPU).**

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

**Deliverables** (one commit each, suite green at each gate; the first two do not touch the trainer):

1. `fabricpc/core/epsilon_spectrum.py` with `tests/test_epsilon_spectrum.py` (an explicit matrix with eigenvalues {−50, −1, 1, 3, 10} and a start vector of known eigen-components; a start vector inside a 2-dimensional invariant subspace triggers the breakdown at step 2; a zero start vector; the tanh and gelu MLP fixtures against `jax.hessian`); the oracle docstring gains the cyclic-graph limit and the per-sample-maximum sentence; `stability_bound` raises when λ_max ≤ 0.
2. `Regime` and `regime(spectrum)`; deletion of `regime_label`, `top_epsilon_eigenvalue`, and `section_track_lambda_max` with every caller migrated; `spectrum_at_init` in the demo and the compare script's regime column; the analysis script's `stability` section checks Lanczos against the oracle's two extremes and against the oracle's weighted fraction Σ_{λ > 0} (v_λᵀg0)² f(λ) / Σ_{λ > 0} (v_λᵀg0)² over the excited modes, v_λ the unit eigenvector of eigenvalue λ; `TestRegime` (bands from constructed spectra with the weight at the top versus at the floor, same extremes, different bands; reversal at T = 1 and 5, none at T = 2; `growth_min` precedence); `TestStabilityBracket` parametrized over the plain and muPC depth-3 chains, so the muPC bracket pins sPC+muPC's gradient scale; the class docstring rewritten with the f̄ condition, the reversal mechanism, the Adam and SGD caveats, λ_max as a weight-scale quantity, and the paper's ResNet-18 setting.
3. After the `IterContext` rebase: `RegimeProbe` with `tests/test_regime_probe.py` (a tanh MLP trained two epochs with `every=2` under both trainers; the CSV round-trips through the plot reader); the demo's `--track_regime N` and `--schedule_epochs`; the analysis script's `--plot_track` (four panels: λ_max and |λ_min| with the 2/η line only when η is present, f̄ with `negative_weight`, weight norms, test accuracy; vertical lines at the first reversal and the first crossing), the r S⁻¹ damping table in `equilibrium_profile`, and the normalized gradient comparison in `backprop_regime`.
4. Documentation: the guide's backprop-regime paragraph and tuning-table rows (η below 1/(λ_max − 1) at odd T for the output gradient and below 2/λ_max at every T, measured with `epsilon_spectrum` and tracked with `RegimeProbe`; `infer_steps` by f̄); a `RegimeProbe` fence; the FAQ; the CHANGELOG (all three replaced names are in the unreleased section, so no breaking-change bullet); the technical report's tracking section regenerated from the control runs, with new subsections on the paper's ResNet-18 setting and on the r S⁻¹ damping.
5. The four GPU control runs, with the reading rule applied and the outcome recorded in the demo docstring, the guide, and the report.

### Status (2026-09-08)

| Item | State                                                                                                                                                                                                                                                                                                                             |
|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Linear oracle, `error_energy`, oracle and backprop-correspondence tests, analysis script CPU sections, guide and CHANGELOG touch points | shipped on this branch                                                                                                                                                                                                                                                                                                            |
| `regime_label`, `top_epsilon_eigenvalue`, `--track_lambda_max` | shipped; to be deleted by deliverable 2 above                                                                                                                                                                                                                                                                                     |
| `epsilon_spectrum`, `Regime`, `RegimeProbe`, `--track_regime`, `--plot_track`, control runs | designed and implemented; deliverables 1 and 2 done, 3 to 5 wait on the `IterContext` PR                                                                                                                                                                                                                                          |
| 2-epoch sweep, 100-epoch runs, the two tracking CSVs | measured before the 0.5.1 gradient normalization; accepted as recorded, not re-collected. The inference energy is untouched by the normalization, so H_ε and the regime at init are the same under both trainers; the training trajectories are not, and the control runs are the first tracking data from the normalized trainer |

## Appendix: resnet18/CIFAR-10 2-epoch sweep results

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

Interpretation (2026-09-04, revised 2026-09-08; the mechanism and the diagnostics are under "Review round 2" above): across the five tables accuracy declines monotonically with η·T·λ from ePC's small-η·T limit (38.8%; no backprop arm was run, and the 100-epoch demo holds the only measured backprop number, 77.11%) toward the PC equilibrium (31.0% at every η for T ≥ 32), with sPC-120 at 34.6% between them because 120 state-based steps do not reach equilibrium. The regime parameter is η·T·λ on the excited modes of H_ε, the energy's Hessian in error coordinates: each excited mode relaxes by 1 − (1 − ηλ)^T over T steps. A one-eigenvalue fit of the η ≤ 0.01 cells gives λ_eff = 12.0 (rms residual 0.05 in normalized accuracy), a heuristic; power iteration on the resnet18 graph at init gives λ_max = 16.4 (`scripts/epc_analysis.py --resnet18`), and the two are consistent within a factor of 1.4. At equilibrium the output error is r S⁻¹, which damps the learning signal along each excited mode by its eigenvalue of S; that linear-chain mechanism is consistent with the 31% plateau trailing the 38.8% plateau. The η = 0.1 cells that collapsed at T ≤ 3 sit at η·λ_max = 1.6 at init: at T = 1 the output layer's weight gradient reverses sign on the top mode once η(λ_max − 1) > 1 (1.5 here), while the (0.03, 1) cell at η·λ_max = 0.49 did not collapse. The 100-epoch runs (project-root `sweep_eta*_steps*.log`) reach the bound late: only η·T ≤ 0.002 survived, the library default (1e-3, 5) collapsed at epoch 20, and 1e-2 collapsed by epoch 10 at every step count; λ_max tracked through the default's collapse grew from 16 to 51 over the first ten epochs and then ran away (84, 131, 471, 3508, 12539 at epochs 11 to 15).
