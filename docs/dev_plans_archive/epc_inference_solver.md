# ePC inference solver, composable inference schedule, and the unrolled topological schedule

## Context

FabricPC's two inference solvers (`InferenceSGD`, `InferenceSGDNormClip`) implement state-based predictive coding (sPC): latent states relax by local gradient descent, so the output-loss signal attenuates by the state learning rate per layer per step and deep graphs need 100s of inference steps (resnet18 demo: 120 steps, 476 s/epoch). The ePC paper (Goemaere et al., arXiv 2505.20137) reparameterizes PC over prediction errors: one reverse-mode AD pass through the whole network delivers the loss signal to every layer unattenuated, reaching the same equilibrium in ~several steps. sPC remains the general solver for arbitrary graphs; ePC is the efficient solver for DAG (or unrolled-cyclic) representations.

This plan adds five things: a split of the node forward contract into `predict` / `pair` / `energy` (Component 3), so one node-level prediction pass serves both parameterizations; ePC as an `InferenceBase` subclass; a composable inference schedule (a few ePC steps to near-equilibrium, then sPC refinement on the true arbitrary-graph energy); a generalization of the topological-order method to an unroll degree `U`, so ePC also accepts cyclic/self-recurrent graphs by unrolling; and an ePC-vs-sPC benchmark on the resnet18 demo.

It also fixes five latent defects the design exposed: (1) `InferenceBase` template methods re-resolve their own class via `type(structure.config["inference"])` (`core/inference.py:100-101`, `:212-213`, `:258-259`), which breaks any composition (Component 2); (2) on cyclic graphs `_topological_sort` returns a partial order behind a print warning (`graph_construction.py:102-103`) — on `x→a⇄b→y` the order is just `("x",)`, so feedforward init leaves cycle members *and everything downstream* at random init, and muPC attaches no scaling to them (Component 1); (3) every state initializer leaves `in_degree == 0` nodes with z_mu = 0 while error = 0, violating error = z_latent − z_mu at init — sPC masks this on the first forward (`nodes/base.py:495-507`), ePC would read the invalid z_mu directly (Formulation, Component 1); (4) sPC forces unclamped readouts to error = 0, energy = 0, zeroing a Hopfield readout's attractor energy (Component 3); (5) the node contract fuses prediction, the additive error map, and energy scoring in one `forward()` — `error = z_latent − z_mu` is copy-pasted into every node body while `GaussianEnergy` recomputes the difference itself and never reads `state.error`, custom energy terms are post-hoc `state._replace` patches (StorkeyHopfield; `ScaledSumNode`, `tests/test_external_custom_node.py:150-153`), and source semantics live in the solver branch (`nodes/base.py:495-507`) so `IdentityNode.forward` crashes if ever called with `in_degree == 0` (Component 3).

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

Because the ε gradient is taken through the full network's transfer function — a change in one node's ε moves every downstream derived latent — η must be tuned like a weight learning rate, not like sPC's local per-node rate. Measured on the resnet18/CIFAR-10 convergence benchmark (Component 6), the ε updates to reach sPC's final 120-step energy: 104 at η = 1e-3, 11 at 1e-2, 4 at 3e-2, 1 at 0.1. The default is 1e-2 as a robustness margin, not the fastest measured rate: one batch on one architecture is thin evidence for 0.1's stability across models, and 1e-2 already converges in about a dozen updates.

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

`eta_infer` defaults to 1e-2; the constructor docstring directs tuning it like a weight learning rate (the ε step descends the full-transfer-function gradient), quotes the measured convergence table (104/11/4/1 ε updates at η = 1e-3/1e-2/3e-2/0.1), and states the robustness-margin rationale for not defaulting to the fastest measured rate (Formulation).

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
- The sweep table is pending its ~5 h run; the equal-wall-clock claim is unmeasured until then and the PR text must not imply otherwise.


Resnet18/CIFAR-10 Results:
======================================================================
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
