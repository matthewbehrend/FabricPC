"""
StorkeyHopfield Diagnostic Tool
================================
Diagnostic investigation of Hopfield attractor dynamics in PC classification
under few-shot + noise conditions (Fashion-MNIST, K=50, noise_std=2.0).

Architecture (compared models)::

    Hopfield: pixels(784) ──→ hidden(128,tanh) ──→ hopfield(128,tanh) ──→ output(10)
    MLP:      pixels(784) ──→ hidden(128,tanh) ──→ linear(128,tanh)  ──→ output(10)

Four diagnostic phases:
  Phase 1: Strength sweep — ABExperiment-based sweep of hopfield_strength
           vs MLP control under K=50, noise=2.0 (default)
  Phase 2: Per-step inference dynamics — decompose gradient signals
  Phase 3: W matrix analysis — eigenvalue spectrum and evolution
  Phase 5: Latent distribution comparison — detect representation collapse

Usage:
    python examples/storkey_hopfield_diagnostic.py
    python examples/storkey_hopfield_diagnostic.py --phase 1 --n_trials 5
    python examples/storkey_hopfield_diagnostic.py --phase 2
    python examples/storkey_hopfield_diagnostic.py --phase all
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax

from fabricpc.nodes import Linear, IdentityNode, StorkeyHopfield
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import SoftmaxActivation, TanhActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD, gather_inputs, run_inference
from fabricpc.core.initializers import XavierInitializer
from fabricpc.training import train, evaluate, make_train_step
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.state_ops import update_node_in_state
from fabricpc.utils.data.dataloader import (
    FashionMnistLoader,
    FewShotLoader,
    NoisyTestLoader,
)
from fabricpc.experiments import ExperimentArm, ABExperiment
from fabricpc.experiments.statistics import paired_ttest, cohens_d
from fabricpc import setup_jax

setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")

# ============================================================================
# Shared Helpers
# ============================================================================

OPTIMIZER = optax.adamw(0.001, weight_decay=0.1)
BATCH_SIZE = 64

# Default few-shot conditions (where Hopfield advantage is strongest)
DEFAULT_K = 50
DEFAULT_NOISE = 2.0


def make_hopfield_factory(hopfield_strength=None):
    """Return a model factory closure with the given hopfield_strength."""

    def create_hopfield_model(rng_key):
        pixels = IdentityNode(shape=(784,), name="pixels")
        hidden = Linear(
            shape=(128,),
            activation=TanhActivation(),
            name="hidden",
            weight_init=XavierInitializer(),
        )
        hopfield = StorkeyHopfield(
            shape=(128,),
            name="hopfield",
            hopfield_strength=hopfield_strength,
        )
        output = Linear(
            shape=(10,),
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            name="class",
            weight_init=XavierInitializer(),
        )
        structure = graph(
            nodes=[pixels, hidden, hopfield, output],
            edges=[
                Edge(source=pixels, target=hidden.slot("in")),
                Edge(source=hidden, target=hopfield.slot("in")),
                Edge(source=hopfield, target=output.slot("in")),
            ],
            task_map=TaskMap(x=pixels, y=output),
            inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return create_hopfield_model


def create_hopfield_model_with_strength(rng_key, strength_value):
    """Create PC model with StorkeyHopfield at a given fixed strength."""
    return make_hopfield_factory(strength_value)(rng_key)


def create_mlp_model(rng_key):
    """Create standard MLP baseline."""
    pixels = IdentityNode(shape=(784,), name="pixels")
    hidden1 = Linear(
        shape=(128,),
        activation=TanhActivation(),
        name="hidden1",
        weight_init=XavierInitializer(),
    )
    hidden2 = Linear(
        shape=(128,),
        activation=TanhActivation(),
        name="hidden2",
        weight_init=XavierInitializer(),
    )
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        name="class",
        weight_init=XavierInitializer(),
    )
    structure = graph(
        nodes=[pixels, hidden1, hidden2, output],
        edges=[
            Edge(source=pixels, target=hidden1.slot("in")),
            Edge(source=hidden1, target=hidden2.slot("in")),
            Edge(source=hidden2, target=output.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=output),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
    )
    params = initialize_params(structure, rng_key)
    return params, structure


def make_data_factory(k_per_class, noise_std, batch_size):
    """Return a data_loader_factory(seed) for ABExperiment."""

    def factory(seed):
        train_loader = FewShotLoader(
            dataset_name="fashion_mnist",
            split="train",
            k_per_class=k_per_class,
            batch_size=batch_size,
            num_classes=10,
            shuffle=True,
            seed=seed,
            tensor_format="flat",
            normalize_mean=0.2860,
            normalize_std=0.3530,
        )
        base_test_loader = FashionMnistLoader(
            split="test",
            batch_size=batch_size,
            shuffle=False,
            tensor_format="flat",
        )
        test_loader = NoisyTestLoader(
            base_loader=base_test_loader,
            noise_std=noise_std,
            seed=seed,
        )
        return train_loader, test_loader

    return factory


def _make_train_loader(seed=42):
    """Create a FewShotLoader for Fashion-MNIST at default K/noise."""
    return FewShotLoader(
        dataset_name="fashion_mnist",
        split="train",
        k_per_class=DEFAULT_K,
        batch_size=BATCH_SIZE,
        num_classes=10,
        shuffle=True,
        seed=seed,
        tensor_format="flat",
        normalize_mean=0.2860,
        normalize_std=0.3530,
    )


def _make_test_loader(seed=42):
    """Create a NoisyTestLoader for Fashion-MNIST at default noise."""
    base = FashionMnistLoader(
        split="test",
        batch_size=BATCH_SIZE,
        shuffle=False,
        tensor_format="flat",
    )
    return NoisyTestLoader(base_loader=base, noise_std=DEFAULT_NOISE, seed=seed)


def _get_learned_strength(params, structure):
    """Extract the effective hopfield_strength from trained params."""
    for node_name, node_params in params.nodes.items():
        if "hopfield_strength" in node_params.biases:
            raw = node_params.biases["hopfield_strength"]
            return float(jax.nn.softplus(raw))
    return None


def custom_train_loop(
    params, structure, train_loader, optimizer, rng_key, max_batches=50
):
    """Minimal training loop returning params after N batches.

    Loops over the loader multiple times if needed (important for small
    few-shot loaders that may have fewer batches than max_batches).
    """
    opt_state = optimizer.init(params)
    step = make_train_step(structure, optimizer)
    batch_count = 0
    while batch_count < max_batches:
        for batch_data in train_loader:
            if batch_count >= max_batches:
                break
            batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
            rng_key, subkey = jax.random.split(rng_key)
            params, opt_state, _, _ = step(params, opt_state, batch, subkey)
            batch_count += 1
    return params


def custom_train_loop_with_snapshots(
    params,
    structure,
    train_loader,
    optimizer,
    rng_key,
    snapshot_every=5,
    max_batches=50,
):
    """Training loop that snapshots W matrix at regular intervals.

    Loops over the loader multiple times if needed.
    """
    opt_state = optimizer.init(params)
    step = make_train_step(structure, optimizer)
    w_snapshots = []
    batch_count = 0
    while batch_count < max_batches:
        for batch_data in train_loader:
            if batch_count >= max_batches:
                break
            batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
            rng_key, subkey = jax.random.split(rng_key)
            params, opt_state, metrics, _ = step(params, opt_state, batch, subkey)
            if batch_count % snapshot_every == 0:
                analysis = analyze_W_matrix(params, structure)
                analysis["batch"] = batch_count
                analysis["energy"] = float(metrics["energy"])
                w_snapshots.append(analysis)
            batch_count += 1
    return params, w_snapshots


def get_hopfield_node_name(structure):
    """Find the hopfield node name in the structure."""
    for name in structure.nodes:
        node = structure.nodes[name]
        if isinstance(node, StorkeyHopfield) or (
            hasattr(node, "node_info")
            and node.node_info is not None
            and issubclass(node.node_info.node_class, StorkeyHopfield)
        ):
            return name
    raise ValueError("No StorkeyHopfield node found in structure")


# ============================================================================
# Phase 1: Hopfield Strength Sweep (ABExperiment under K=50, noise=2.0)
# ============================================================================


def phase1_strength_sweep(n_trials, num_epochs):
    """Sweep hopfield_strength vs MLP baseline under K=50, noise=2.0.

    Uses ABExperiment for paired statistical comparisons at each strength.
    """
    K = DEFAULT_K
    NOISE_STD = DEFAULT_NOISE
    STRENGTHS = [0.0, 0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 32.0, None]

    optimizer = optax.adamw(0.001, weight_decay=0.1)
    train_config = {"num_epochs": num_epochs}

    print("=" * 70)
    print("Strength Sweep: Hopfield vs MLP")
    print("=" * 70)
    print(f"Dataset: Fashion-MNIST, K={K} shots/class, noise_std={NOISE_STD}")
    print("Hopfield: 784 -> 128(tanh) -> 128(StorkeyHopfield, tanh) -> 10(softmax, CE)")
    print("MLP:      784 -> 128(tanh) -> 128(tanh) -> 10(softmax, CE)")
    print(f"Trials: {n_trials}, Epochs: {num_epochs}")
    print(f"Strengths: {[s if s is not None else 'learnable' for s in STRENGTHS]}")
    print()

    arm_mlp = ExperimentArm(
        name="MLP",
        model_factory=create_mlp_model,
        train_fn=train,
        eval_fn=evaluate,
        optimizer=optimizer,
        train_config=train_config,
    )

    data_factory = make_data_factory(K, NOISE_STD, BATCH_SIZE)
    sweep_results = []

    for s in STRENGTHS:
        label = "learnable" if s is None else f"{s}"
        print(f"\n{'─'*70}")
        print(f"  hopfield_strength = {label}")
        print(f"{'─'*70}")

        arm_hop = ExperimentArm(
            name=f"Hop(s={label})",
            model_factory=make_hopfield_factory(s),
            train_fn=train,
            eval_fn=evaluate,
            optimizer=optimizer,
            train_config=train_config,
        )

        experiment = ABExperiment(
            arm_a=arm_hop,
            arm_b=arm_mlp,
            metric="accuracy",
            data_loader_factory=data_factory,
            n_trials=n_trials,
            verbose=False,
        )
        result = experiment.run()

        hop_acc = result.arm_a_metrics
        mlp_acc = result.arm_b_metrics
        delta = hop_acc - mlp_acc

        row = {
            "strength": s,
            "label": label,
            "hop_mean": float(np.mean(hop_acc)),
            "hop_se": (
                float(np.std(hop_acc, ddof=1) / np.sqrt(len(hop_acc)))
                if len(hop_acc) > 1
                else 0.0
            ),
            "mlp_mean": float(np.mean(mlp_acc)),
            "mlp_se": (
                float(np.std(mlp_acc, ddof=1) / np.sqrt(len(mlp_acc)))
                if len(mlp_acc) > 1
                else 0.0
            ),
            "delta_mean": float(np.mean(delta)),
        }

        if n_trials >= 2:
            ttest = paired_ttest(hop_acc, mlp_acc)
            effect = cohens_d(hop_acc, mlp_acc)
            row["p_value"] = ttest.p_value
            row["significant"] = ttest.significant_at_05
            row["cohens_d"] = effect.d
        else:
            row["p_value"] = float("nan")
            row["significant"] = False
            row["cohens_d"] = float("nan")

        # For learnable strength, extract the learned value
        learned_str = None
        if s is None:
            key = jax.random.PRNGKey(42)
            params, structure = make_hopfield_factory(None)(key)
            train_loader, _ = data_factory(42)
            result = train(
                params, structure, train_loader, optimizer, train_config, key
            )
            learned_str = _get_learned_strength(result.params, structure)

        row["learned_str"] = learned_str
        sweep_results.append(row)

        p_str = f"{row['p_value']:.4f}" if not np.isnan(row["p_value"]) else "n/a"
        sig = "*" if row.get("significant") else ""
        print(
            f"  -> Hopfield: {row['hop_mean']*100:.2f}%  "
            f"MLP: {row['mlp_mean']*100:.2f}%  "
            f"Delta: {row['delta_mean']*100:+.2f}%  "
            f"p={p_str} {sig}"
        )

    # Summary table
    print("\n")
    print("=" * 85)
    print("STRENGTH SWEEP SUMMARY")
    print("=" * 85)
    print(f"  K={K}, noise_std={NOISE_STD}, Trials: {n_trials}, Epochs: {num_epochs}")
    print()
    header = (
        f"{'Strength':<12} {'Hopfield%':>12} {'MLP%':>12} "
        f"{'Delta%':>10} {'p-value':>10} {'Sig':>5} {'d':>8} {'Learned':>10}"
    )
    print(header)
    print("─" * len(header))
    for r in sweep_results:
        hop_str = f"{r['hop_mean']*100:.2f}+/-{r['hop_se']*100:.2f}"
        mlp_str = f"{r['mlp_mean']*100:.2f}+/-{r['mlp_se']*100:.2f}"
        delta_str = f"{r['delta_mean']*100:+.2f}"
        p_str = f"{r['p_value']:.4f}" if not np.isnan(r["p_value"]) else "n/a"
        sig_str = "*" if r.get("significant") else ""
        d_str = (
            f"{r['cohens_d']:.3f}"
            if not np.isnan(r.get("cohens_d", float("nan")))
            else "n/a"
        )
        learned = f"{r['learned_str']:.3f}" if r["learned_str"] is not None else ""
        print(
            f"{r['label']:<12} {hop_str:>12} {mlp_str:>12} "
            f"{delta_str:>10} {p_str:>10} {sig_str:>5} {d_str:>8} {learned:>10}"
        )
    print("─" * len(header))

    return sweep_results


# ============================================================================
# Phase 2: Per-Step Inference Dynamics
# ============================================================================


def instrumented_forward_value_and_grad(params, state, clamps, structure):
    """Replicate forward_value_and_grad with gradient decomposition snapshots.

    Captures hopfield.latent_grad at two key moments:
      1. After hopfield's own forward_and_latent_grads (contains PC_self + Hop_self)
      2. After class node accumulates its backward grad (adds Top_down)
    """
    snapshots = {}
    hop_node_name = get_hopfield_node_name(structure)

    for node_name in structure.nodes:
        node = structure.nodes[node_name]
        node_info = node.node_info
        node_class = node_info.node_class
        node_state = state.nodes[node_name]
        node_params = params.nodes[node_name]

        in_edges_data = gather_inputs(node_info, structure, state)

        node_state, inedge_grads, self_grad = node_class.forward_and_latent_grads(
            node_params,
            in_edges_data,
            node_state,
            node_info,
            is_clamped=(node_name in clamps),
        )
        node_state = node_state._replace(latent_grad=node_state.latent_grad + self_grad)

        state = state._replace(nodes={**state.nodes, node_name: node_state})

        # Snapshot after hopfield processes itself (before class backward)
        if node_name == hop_node_name:
            snapshots["after_hopfield_forward"] = state.nodes[hop_node_name].latent_grad

        # Accumulate backward grads to pre-synaptic nodes
        for edge_key, grad in inedge_grads.items():
            source_name = structure.edges[edge_key].source
            latent_grad = state.nodes[source_name].latent_grad + grad
            state = update_node_in_state(state, source_name, latent_grad=latent_grad)

        # Snapshot after class accumulates its top-down grad to hopfield
        if node_name == "class":
            snapshots["after_class_backward"] = state.nodes[hop_node_name].latent_grad

    return state, snapshots


def decompose_hopfield_gradients(snapshots, hop_state):
    """Extract the three gradient components from inference snapshots."""
    grad_after_self = snapshots["after_hopfield_forward"]
    grad_final = snapshots["after_class_backward"]

    # Top_down = what class node added
    top_down = grad_final - grad_after_self

    # PC_self = precision * (z - z_mu), precision=1.0 for GaussianEnergy default
    pc_self = hop_state.z_latent - hop_state.z_mu

    # Hop_self = everything else in the self-gradient
    hop_self = grad_after_self - pc_self

    return pc_self, hop_self, top_down


def diagnostic_inference_step(params, state, clamps, structure, config):
    """Single inference step with full gradient decomposition diagnostics."""
    inference_obj = structure.config["inference"]
    cls = type(inference_obj)
    hop_node_name = get_hopfield_node_name(structure)

    # Phase 1: Zero grads
    state = cls.zero_grads(params, state, clamps, structure)

    # Phase 2: Forward + grad with snapshots
    state, snapshots = instrumented_forward_value_and_grad(
        params,
        state,
        clamps,
        structure,
    )

    # Extract diagnostics BEFORE latent update
    hop_state = state.nodes[hop_node_name]
    pc_self, hop_self, top_down = decompose_hopfield_gradients(snapshots, hop_state)

    hidden_state = state.nodes["hidden"]

    # Compute E_hop directly
    hop_params = params.nodes[hop_node_name]
    edge_key = list(hop_params.weights.keys())[0]
    W_raw = hop_params.weights[edge_key]
    node_config = structure.nodes[hop_node_name].node_info.node_config
    W = StorkeyHopfield._prepare_W(W_raw, node_config)

    if "hopfield_strength" in hop_params.biases:
        strength = float(jax.nn.softplus(hop_params.biases["hopfield_strength"]))
    else:
        strength = node_config.get("hopfield_strength", 1.0)

    z = hop_state.z_latent
    D = z.shape[-1]
    wz = z @ W
    E_hop_per_sample = (0.5 / D) * jnp.sum(wz * (wz - z), axis=-1)

    diagnostics = {
        "z_norm": float(jnp.mean(jnp.linalg.norm(hop_state.z_latent, axis=-1))),
        "z_mu_norm": float(jnp.mean(jnp.linalg.norm(hop_state.z_mu, axis=-1))),
        "E_pc": float(
            jnp.mean(0.5 * jnp.sum((hop_state.z_latent - hop_state.z_mu) ** 2, axis=-1))
        ),
        "E_hop": float(jnp.mean(strength * E_hop_per_sample)),
        "pc_self_norm": float(jnp.mean(jnp.linalg.norm(pc_self, axis=-1))),
        "hop_self_norm": float(jnp.mean(jnp.linalg.norm(hop_self, axis=-1))),
        "top_down_norm": float(jnp.mean(jnp.linalg.norm(top_down, axis=-1))),
        "total_grad_norm": float(
            jnp.mean(jnp.linalg.norm(hop_state.latent_grad, axis=-1))
        ),
        "ratio_hop_over_pc": float(
            jnp.mean(jnp.linalg.norm(hop_self, axis=-1))
            / (jnp.mean(jnp.linalg.norm(pc_self, axis=-1)) + 1e-10)
        ),
        "ratio_hop_over_topdown": float(
            jnp.mean(jnp.linalg.norm(hop_self, axis=-1))
            / (jnp.mean(jnp.linalg.norm(top_down, axis=-1)) + 1e-10)
        ),
        "tanh_saturation_frac": float(
            jnp.mean(jnp.abs(hop_state.z_mu) > jnp.tanh(2.0))
        ),
        "act_mean_abs": float(jnp.mean(jnp.abs(hop_state.z_mu))),
        "hidden_grad_norm": float(
            jnp.mean(jnp.linalg.norm(hidden_state.latent_grad, axis=-1))
        ),
        "strength_effective": strength,
    }

    # Cross-check: recompute hop_self directly
    hop_self_direct = (strength / D) * (wz @ W - wz)
    diagnostics["hop_self_crosscheck_err"] = float(
        jnp.max(jnp.abs(hop_self - hop_self_direct))
    )

    # Phase 3: Update latents
    state = cls.update_latents(params, state, clamps, structure, config)

    return state, diagnostics


def run_diagnostic_inference(params, initial_state, clamps, structure, n_steps=20):
    """Full inference loop with per-step diagnostics (no JIT)."""
    config = structure.config["inference"].config
    state = initial_state
    all_diagnostics = []

    for t in range(n_steps):
        state, diag = diagnostic_inference_step(
            params,
            state,
            clamps,
            structure,
            config,
        )
        diag["step"] = t
        all_diagnostics.append(diag)

    return state, all_diagnostics


def phase2_inference_dynamics():
    """Decompose gradient signals during inference for s=1.0 and s=0.0."""
    for strength_label, strength_val in [
        ("Hopfield s=1.0", 1.0),
        ("Hopfield s=0.0 (baseline)", 0.0),
    ]:
        print(f"\n{'='*110}")
        print(f"  {strength_label}")
        print(f"{'='*110}")

        rng = jax.random.PRNGKey(42)
        graph_key, train_key, eval_key = jax.random.split(rng, 3)
        params, structure = create_hopfield_model_with_strength(graph_key, strength_val)

        train_loader = _make_train_loader(seed=42)

        # Train for 50 batches to get non-trivial W
        params = custom_train_loop(
            params,
            structure,
            train_loader,
            OPTIMIZER,
            train_key,
            max_batches=50,
        )

        # Take one test batch
        test_loader = _make_test_loader(seed=42)
        batch_data = next(iter(test_loader))
        batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}

        # Clamps (both x and y, as in training)
        task_map = structure.config.get("task_map", structure.task_map)
        clamps = {}
        for task_name, task_value in batch.items():
            if task_name in task_map:
                clamps[task_map[task_name]] = task_value

        batch_size = batch["x"].shape[0]
        rng, subkey = jax.random.split(eval_key)
        init_state = initialize_graph_state(
            structure,
            batch_size,
            subkey,
            clamps=clamps,
            params=params,
        )

        # Run diagnostic inference
        _, step_logs = run_diagnostic_inference(
            params,
            init_state,
            clamps,
            structure,
            n_steps=20,
        )

        # Print per-step table
        print(
            f"\n{'Step':>4} {'E_pc':>8} {'E_hop':>8} {'|z|':>7} "
            f"{'PC_grad':>9} {'Hop_grad':>9} {'TD_grad':>9} "
            f"{'Hop/PC':>8} {'Hop/TD':>8} {'Sat%':>6} {'|act|':>8} "
            f"{'xcheck':>8}"
        )
        print("-" * 115)
        for d in step_logs:
            print(
                f"{d['step']:>4} "
                f"{d['E_pc']:>8.4f} {d['E_hop']:>8.4f} "
                f"{d['z_norm']:>7.3f} "
                f"{d['pc_self_norm']:>9.4f} "
                f"{d['hop_self_norm']:>9.4f} "
                f"{d['top_down_norm']:>9.4f} "
                f"{d['ratio_hop_over_pc']:>8.2f} "
                f"{d['ratio_hop_over_topdown']:>8.2f} "
                f"{d['tanh_saturation_frac']*100:>5.1f}% "
                f"{d['act_mean_abs']:>8.3f} "
                f"{d['hop_self_crosscheck_err']:>8.2e}"
            )

    return step_logs


# ============================================================================
# Phase 3: W Matrix Analysis
# ============================================================================


def analyze_W_matrix(params, structure):
    """Full analysis of the Hopfield W matrix."""
    hop_node_name = get_hopfield_node_name(structure)
    hop_params = params.nodes[hop_node_name]
    edge_key = list(hop_params.weights.keys())[0]
    W_raw = hop_params.weights[edge_key]
    node_config = structure.nodes[hop_node_name].node_info.node_config
    W = StorkeyHopfield._prepare_W(W_raw, node_config)
    W_np = np.array(W)

    eigenvalues = np.linalg.eigvalsh(W_np)
    attractor_eigs = eigenvalues**2 - eigenvalues

    return {
        "eigenvalues": eigenvalues,
        "W_frobenius": float(np.linalg.norm(W_np, "fro")),
        "W_operator_norm": float(np.max(np.abs(eigenvalues))),
        "max_eigenvalue": float(np.max(eigenvalues)),
        "min_eigenvalue": float(np.min(eigenvalues)),
        "frac_eigs_in_0_1": float(np.mean((eigenvalues > 0) & (eigenvalues < 1))),
        "frac_repelling": float(np.mean(attractor_eigs > 0)),
        "W2_minus_W_frobenius": float(np.linalg.norm(W_np @ W_np - W_np, "fro")),
        "mean_attractor_eig": float(np.mean(attractor_eigs)),
        "max_attractor_eig": float(np.max(attractor_eigs)),
    }


def phase3_W_analysis():
    """Track W matrix properties over training."""
    rng = jax.random.PRNGKey(42)
    graph_key, train_key = jax.random.split(rng)
    params, structure = create_hopfield_model_with_strength(graph_key, 1.0)

    train_loader = _make_train_loader(seed=42)

    params, w_trajectory = custom_train_loop_with_snapshots(
        params,
        structure,
        train_loader,
        OPTIMIZER,
        train_key,
        snapshot_every=5,
        max_batches=50,
    )

    # Print trajectory
    print(
        f"\n{'Batch':>6} {'||W||_F':>9} {'||W||_op':>9} "
        f"{'max(eig)':>9} {'min(eig)':>9} "
        f"{'%repel':>7} {'||W2-W||':>9} {'energy':>10}"
    )
    print("-" * 80)
    for w in w_trajectory:
        print(
            f"{w['batch']:>6} {w['W_frobenius']:>9.4f} {w['W_operator_norm']:>9.4f} "
            f"{w['max_eigenvalue']:>9.4f} {w['min_eigenvalue']:>9.4f} "
            f"{w['frac_repelling']*100:>6.1f}% {w['W2_minus_W_frobenius']:>9.4f} "
            f"{w['energy']:>10.2f}"
        )

    # Final detailed analysis
    final = analyze_W_matrix(params, structure)
    print("\nFinal W Analysis (after training):")
    print(f"  Frobenius norm:        {final['W_frobenius']:.4f}")
    print(f"  Operator norm:         {final['W_operator_norm']:.4f}")
    print(
        f"  Eigenvalue range:      [{final['min_eigenvalue']:.4f}, "
        f"{final['max_eigenvalue']:.4f}]"
    )
    print(f"  Fraction in (0,1):     {final['frac_eigs_in_0_1']*100:.1f}%")
    print(f"  Fraction repelling:    {final['frac_repelling']*100:.1f}%")
    print(f"  ||W^2 - W||_F:        {final['W2_minus_W_frobenius']:.4f}")
    print(f"  Mean attractor eig:    {final['mean_attractor_eig']:.6f}")
    print(f"  Max attractor eig:     {final['max_attractor_eig']:.6f}")

    # Eigenvalue histogram
    eigs = final["eigenvalues"]
    print("\n  Eigenvalue distribution (128 eigenvalues):")
    bins = [
        (-2, -1),
        (-1, -0.5),
        (-0.5, 0),
        (0, 0.5),
        (0.5, 1.0),
        (1.0, 2.0),
        (2.0, 5.0),
    ]
    for lo, hi in bins:
        count = int(np.sum((eigs >= lo) & (eigs < hi)))
        bar = "#" * count
        print(f"    [{lo:>5.1f}, {hi:>4.1f})  {count:>3}  {bar}")

    return w_trajectory, final


# ============================================================================
# Phase 5: Latent Distribution Comparison
# ============================================================================


def collect_latents(params, structure, test_loader, node_name, rng_key):
    """Collect z_latent from a specific node across all test batches (eval mode)."""
    all_z = []
    all_labels = []

    for batch_data in test_loader:
        batch = {"x": jnp.array(batch_data[0]), "y": jnp.array(batch_data[1])}
        batch_size = batch["x"].shape[0]

        # Eval mode: only clamp x (not y)
        task_map = structure.config.get("task_map", structure.task_map)
        clamps = {task_map["x"]: batch["x"]}

        rng_key, subkey = jax.random.split(rng_key)
        state = initialize_graph_state(
            structure,
            batch_size,
            subkey,
            clamps=clamps,
            params=params,
        )
        final_state = run_inference(params, state, clamps, structure)

        all_z.append(np.array(final_state.nodes[node_name].z_latent))
        all_labels.append(np.argmax(np.array(batch["y"]), axis=1))

    return np.concatenate(all_z), np.concatenate(all_labels)


def compute_latent_statistics(z, labels, n_classes=10):
    """Compute representation quality metrics."""
    overall_mean = np.mean(z, axis=0)
    S_B = 0.0
    S_W = 0.0
    class_means = {}

    for c in range(n_classes):
        mask = labels == c
        z_c = z[mask]
        if len(z_c) == 0:
            continue
        mu_c = np.mean(z_c, axis=0)
        class_means[c] = mu_c
        n_c = z_c.shape[0]
        S_B += n_c * np.sum((mu_c - overall_mean) ** 2)
        S_W += np.sum((z_c - mu_c) ** 2)

    fisher_ratio = S_B / (S_W + 1e-10)

    # Participation ratio (effective dimensionality)
    z_centered = z - np.mean(z, axis=0)
    cov = (z_centered.T @ z_centered) / z.shape[0]
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    total_var = np.sum(eigenvalues)
    if total_var > 0:
        participation = total_var**2 / np.sum(eigenvalues**2)
    else:
        participation = 0.0

    # PCA dims for 95% variance
    sorted_eigs = np.sort(eigenvalues)[::-1]
    cumvar = np.cumsum(sorted_eigs) / (total_var + 1e-10)
    dims_95 = int(np.searchsorted(cumvar, 0.95)) + 1

    # Inter-class centroid distances
    centroid_dists = []
    classes = sorted(class_means.keys())
    for i, c1 in enumerate(classes):
        for c2 in classes[i + 1 :]:
            d = np.linalg.norm(class_means[c1] - class_means[c2])
            centroid_dists.append(d)
    mean_centroid_dist = np.mean(centroid_dists) if centroid_dists else 0.0

    z_abs = np.abs(z)

    return {
        "fisher_ratio": fisher_ratio,
        "participation_ratio": participation,
        "dims_for_95pct_var": dims_95,
        "mean_centroid_distance": mean_centroid_dist,
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "z_abs_mean": float(np.mean(z_abs)),
        "frac_near_saturation": float(np.mean(z_abs > 0.95)),
        "total_variance": float(total_var),
    }


def phase5_latent_analysis():
    """Compare latent representations across Hopfield (s=1, s=0) and MLP."""
    train_config = {"num_epochs": 1}

    models = {
        "Hopfield s=1.0": (
            lambda key: create_hopfield_model_with_strength(key, 1.0),
            "hopfield",
        ),
        "Hopfield s=0.0": (
            lambda key: create_hopfield_model_with_strength(key, 0.0),
            "hopfield",
        ),
        "MLP baseline": (
            create_mlp_model,
            "hidden2",
        ),
    }

    for name, (factory, latent_node) in models.items():
        rng = jax.random.PRNGKey(42)
        graph_key, train_key, eval_key = jax.random.split(rng, 3)
        params, structure = factory(graph_key)

        train_loader = _make_train_loader(seed=42)

        result = train(
            params,
            structure,
            train_loader,
            OPTIMIZER,
            train_config,
            train_key,
            verbose=False,
        )
        trained_params = result.params

        test_loader = _make_test_loader(seed=42)
        z, labels = collect_latents(
            trained_params,
            structure,
            test_loader,
            latent_node,
            eval_key,
        )

        stats = compute_latent_statistics(z, labels)
        print(f"\n  {name}:")
        print(f"    Fisher ratio:          {stats['fisher_ratio']:.4f}")
        print(f"    Participation ratio:   {stats['participation_ratio']:.1f} / 128")
        print(f"    Dims for 95% var:      {stats['dims_for_95pct_var']}")
        print(f"    Mean centroid dist:    {stats['mean_centroid_distance']:.4f}")
        print(
            f"    z mean / std:          {stats['z_mean']:.4f} / {stats['z_std']:.4f}"
        )
        print(f"    Frac near saturation:  {stats['frac_near_saturation']*100:.1f}%")
        print(f"    Total variance:        {stats['total_variance']:.4f}")


# ============================================================================
# Main CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="StorkeyHopfield diagnostic tool — "
        "investigate Hopfield attractor dynamics under few-shot + noise conditions.",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="1",
        choices=["1", "2", "3", "5", "all"],
        help="Which diagnostic phase to run (default: 1)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of paired trials for Phase 1 strength sweep (default: 10)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help="Training epochs per trial for Phase 1 (default: 5)",
    )
    args = parser.parse_args()

    phases = {
        "1": (
            "Phase 1: Strength Sweep (K=50, noise=2.0)",
            lambda: phase1_strength_sweep(args.n_trials, args.num_epochs),
        ),
        "2": ("Phase 2: Per-Step Inference Dynamics", phase2_inference_dynamics),
        "3": ("Phase 3: W Matrix Analysis", phase3_W_analysis),
        "5": ("Phase 5: Latent Distribution Comparison", phase5_latent_analysis),
    }

    if args.phase == "all":
        for key in ["1", "2", "3", "5"]:
            name, fn = phases[key]
            print(f"\n{'#'*70}")
            print(f"# {name}")
            print(f"{'#'*70}")
            fn()
    else:
        name, fn = phases[args.phase]
        print(f"\n{'#'*70}")
        print(f"# {name}")
        print(f"{'#'*70}")
        fn()


if __name__ == "__main__":
    main()
