# FabricPC Roadmap

**A JAX-native predictive coding framework.** FabricPC trains predictive-coding networks on arbitrary graph topologies — feedforward, recurrent, skip, cyclic — with heterogeneous nodes in one energy graph, using node-local learning rules and JAX transformations end to end.

Status date: 2026-08-29. Plain numbers NN refer to issue drafts in `docs/github_issues/` (posted manually after review); #NN are filed GitHub issues.

## Shipped

### Core architecture
- Pure-functional core: the graph is data (`GraphParams`, `GraphState`, `GraphStructure` pytrees); nodes, edges, and updates are the three public abstractions.
- N-dimensional latents: node shapes are arbitrary `(batch, *dims)` tensors, channels-last.
- Object node API: nodes are Python objects (`NodeBase` subclasses) passed to `graph(nodes, edges, task_map)`; edges wire by object reference through named slots. Custom nodes subclass `NodeBase` without touching library code; the contract is pinned by an external-node test and a user guide.
- Energy functionals: Gaussian, Bernoulli, cross-entropy, Laplacian, Huber, KL divergence, plus a custom-energy interface; per-node configuration.
- Inference: `InferenceBase` solver interface with SGD and norm-clipped SGD implementations (this family is sPC — state-based PC, where error propagates one hop per inference step).
- muPC scaling: width- and depth-transferable initialization and update scaling on arbitrary DAGs, including the merge-node depth rule and depth-free readout.

### Node library
- Linear, LinearResidual, SkipConnection, Identity.
- Conv (one class covering 1D/2D/3D) and MaxPool/AvgPool (including global).
- Transformer: a monolithic block node and a decomposed pipeline (embedding, multi-head attention with rotary position encoding and causal masking, layer-normed MLP stages, vocabulary projection).
- StorkeyHopfield: associative-memory node whose attractor dynamics fall out of the PC inference loop.

### Training and release infrastructure
- Unified trainer (v0.5.0, on main): one `train()`/`evaluate()`/`make_train_step()` for `algorithm="pc"` and `"backprop"` (backprop framed in the same energy), resumable, with epoch/iteration callbacks, pluggable metrics, and `generate()` for autoregressive sampling.
- Multi-device training: jit + `NamedSharding` mesh data parallelism via a `mesh=` argument; the `"model"` axis is reserved for future model parallelism.
- PyPI publication (v0.4.0): `pip install fabricpc`, Trusted Publishing (OIDC) pipeline with build, smoke-install, and sdist gates; JAX setup moved into the package (`setup_jax`).
- CI: test workflow (Python 3.11/3.13, a JAX-floor leg, and a two-device sharding-parity step), lint workflow (black + ruff), publish workflow (TestPyPI rehearsal + PyPI release).
- Experiments: paired A/B and N-arm framework with statistics, two-phase Bayesian tuner, Aim experiment tracking.
- Docs and examples: 17 user guides; examples spanning MNIST (dense, conv, cyclic, lateral, multi-GPU), ResNet-18/CIFAR-10, character- and BPE-level transformers, Hopfield studies, PC-vs-backprop and scaling-law comparisons.

## Shipping now

- **ePC solver** (draft PR #47): error-parameterized PC, where the output error traverses the full DAG depth each inference step instead of one hop, reaching equilibrium in a few steps where sPC needs hundreds (Goemaere et al., arXiv:2505.20137). Scope: node contract split into predict/pair/energy, an `EPCInference` solver, a composable `InferenceSchedule` that hands off ePC settling to sPC refinement of the true full-graph energy before each weight update, and cyclic-graph support by unrolling cycles into ePC's DAG. ResNet-18 benchmark evidence recorded on the ePC branches.
- **Model checkpointing** (PR #38): Orbax save/load of parameters, optimizer state, and structure.
- **v0.5.0 PyPI release**: main already carries 0.5.0 (unified trainer); tag and publish pending.

## Planned

### v0.5.0 — September 2026
- XLA flag profiles: production default plus deterministic opt-in (01).
- Stop-gradients on any edge or node output (02); lands after ePC merges so one mechanism serves sPC, ePC, and backprop.
- Node parallelism: group-vmap over stackable nodes, reducing per-step cost from a Python loop over nodes to batched kernels (07).

### v0.6.0 — hackathon release, October 2026
- Starter-kit template project (10), Colab quickstart (11), issue and PR templates (14).

### Q4 2026
- `fabricpc.bench` reproducible benchmark suite (17) and, building on it: model zoo (18), migration guides from pcx and jpc (19).
- Generative graph fuzzer with tolerance tiers and an invariant suite (20).
- Adaptive termination for sPC refinement, replacing fixed step counts (21).
- Measurement: sPC refinement vs cycle unrolling on cyclic graphs (22); structural active-set instrumentation (24, stretch).

### Backlog
- GPU CI runners, a nightly full-suite job, and benchmark trend tracking (04–06; owner wanted).
- Triage the total-graph-energy branch (26) — the `graph_energy()` core helper is already on main.
- Subgraph containers: a reusable block abstraction (28) — `GraphNamespace` ships the naming primitive; the container depends on group-vmap (07).
- Natural-gradient trainer maturation (30).
- Filed issues: random-access out-of-core data path on Grain #46, integration test suite #33, transformer demo perplexity #16, Aim install on Python 3.13 #4.

## Dropped

- **iPC** — superseded by ePC: the composed ePC→sPC schedule delivers the fast-settling inference that incremental per-node updates targeted, without a second solver family.
- Precision weighting (25); hackathon-support drafts 09, 12, 15, 16, 23.
- A `PCNetwork`-style facade — the object graph API is the public API.
- Standalone LayerNorm/BatchNorm nodes — layer normalization lives inside the transformer nodes.
- Flax.linen adoption — the native object node API fills that role.

## References

1. Rao & Ballard (1999) — Predictive coding in visual cortex.
2. Whittington & Bogacz (2017) — Approximation of backprop by PC.
3. Millidge et al. (2022) — Predictive coding: a theoretical and experimental review.
4. Goemaere et al. (2025), arXiv:2505.20137 — error-parameterized predictive coding (ePC).
