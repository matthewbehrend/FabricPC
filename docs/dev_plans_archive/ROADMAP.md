# FabricPC Roadmap

**A JAX-native predictive coding framework.** FabricPC trains predictive-coding networks on arbitrary graph topologies — feedforward, recurrent, skip, cyclic — with heterogeneous nodes in one energy graph, using node-local learning rules and JAX transformations end to end.

Status date: 2026-08-29. Work items are tracked as filed GitHub issues. Future milestones are dated; version numbers are assigned at release time.

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
- Docs and examples: 17 user guides (including writing custom nodes) and a contributor guide (CONTRIBUTING.md); examples spanning MNIST (dense, conv, cyclic, lateral, multi-GPU), ResNet-18/CIFAR-10, character- and BPE-level transformers, Hopfield studies, PC-vs-backprop and scaling-law comparisons.

## Shipping now

- **ePC solver** (draft PR #47): error-parameterized PC, where the output error traverses the full DAG depth each inference step instead of one hop, reaching equilibrium in a few steps where sPC needs hundreds (Goemaere et al., arXiv:2505.20137). Scope: node contract split into predict/pair/energy, an `EPCInference` solver, a composable `InferenceSchedule` that hands off ePC settling to sPC refinement of the true full-graph energy before each weight update, and cyclic-graph support by unrolling cycles into ePC's DAG. ResNet-18 benchmark evidence recorded on the ePC branches. Lands with a demo-level energy/accuracy comparison against sPC and one cyclic demo where the composed schedule decreases full-graph energy.
- **Model checkpointing** (PR #38): Orbax save/load of parameters, optimizer state, the structure snapshot, and the resume triple (rng_key, epoch, step), with save-every-N / keep-best-K retention (design: `model_checkpointing.md`); demonstrated, not just supported — resume-mid-training and save-then-load-then-generate examples.
- **v0.5.0 Unified trainer release**: main already carries 0.5.0 (unified trainer); tag and publish pending.

## Planned

### October 2026 — hackathon release
- XLA flag profiles: production default plus deterministic opt-in.
- Stop-gradients on any edge or node output
- Starter-kit template project, Colab quickstart, issue and PR templates.
- The event is late October; the final two weeks before it admit only docs, demos, and packaging — no engine or API changes, so the two engine items above land early in the month.

### November 2026
- Node parallelism: group-vmap over stackable nodes, reducing per-step cost from a Python loop over nodes to batched kernels. Gate metrics: equivalence against the sequential engine, plus wall-clock, compile time, and executable size on deep chains.

### December 2026
- `fabricpc.bench` reproducible benchmark suite: match the pcx suite's tasks and reference numbers first, then the arms only FabricPC can run — muPC-scaled 100+ layer graphs, and ePC vs sPC vs backprop on one graph in both arms. Every row is a paired N-arm experiment reporting effect sizes and significance, runs from one command, and records wall-clock, realized step counts, the PC-to-backprop matmul ratio, compile time, peak memory, XLA flag profile, and solver configuration.
- Generative graph fuzzer with tolerance tiers and an invariant suite; built first among the efficiency items — the deep benchmark arms and the refine-vs-unroll row stand on its equivalence substrate, and it retroactively validates the group-vmap packed engine.
- Migration guides from pcx and jpc.
- Stretch, slips first: structural active-set instrumentation — after ePC the update at refinement tick r is confined to the r-hop moral-graph neighborhood of the back-edge target nodes (each node's energy term couples it to all sources of its in-edges), known from topology at compile time; record it in cyclic benchmark rows.

### January 2027
- Adaptive termination for sPC refinement: replaces the composed schedule's fixed refinement step count T2 with a tolerance predicate; realized ticks R become the reported cost of cyclicity.
- Refine-vs-unroll measurement: the same cyclic graph solved by the composed schedule and by unrolling, compared on full-graph energy, task metric, matmul count, and wall-clock; settles the default cycle path.
- Model zoo: pretrained checkpoints with one-line load, each with a Colab that reproduces one figure.

### Backlog
- GPU CI runners, a nightly full-suite job, and benchmark trend tracking (owner wanted); the benchmark suite's nightly wall-clock regression testing depends on them.
- Close the superseded total-graph-energy and benchmark branches — `graph_energy()` on main is a strict superset of the former; the latter's tuned defaults already landed.
- Subgraph containers: a reusable block abstraction — `GraphNamespace` ships the naming primitive; containers flatten through `graph()`, and the group-vmap engine packs their shape-identical instances automatically.
- Natural-gradient optimizer transforms: maturation.
- v1.0 API stabilization driven by hackathon feedback.
- Deliberately deferred until the benchmark suite establishes credibility: distributed training beyond data parallelism, a plugin ecosystem, Hugging Face Hub integration.
- Filed issues: a random-access out-of-core data path on Grain, an integration test suite, transformer demo perplexity, Aim install on Python 3.13.
- Research branches (Navier–Stokes, Bayesian transformers, Hopfield variants) continue on their own tracks.

## Dropped

- **iPC** — superseded by ePC: the composed ePC→sPC schedule delivers the fast-settling inference that incremental per-node updates targeted, without a second solver family.
- Precision weighting
- A `PCNetwork`-style facade — the object graph API is the public API.
- Flax.linen adoption — the native object node API fills that role.

## References

1. Rao & Ballard (1999) — Predictive coding in visual cortex.
2. Whittington & Bogacz (2017) — Approximation of backprop by PC.
3. Millidge et al. (2022) — Predictive coding: a theoretical and experimental review.
4. Goemaere et al. (2025), arXiv:2505.20137 — error-parameterized predictive coding (ePC).
5. pcx benchmark suite (arXiv:2407.01163, ICLR 2025) — reference tasks and numbers for `fabricpc.bench`.
