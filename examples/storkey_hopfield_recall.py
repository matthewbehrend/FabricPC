"""
Hopfield Pattern Recall Demo
=============================

Tests the StorkeyHopfield node's ability to recall stored patterns from noisy
probes — the classical Hopfield network benchmark.

Architecture (3-node graph)::

    probe ──→ StorkeyHopfield ──→ output
               (pattern recall via attractor dynamics)

Training: clamp x=noisy_probe, y=clean_pattern. W learns associative structure
    via PC energy minimization.
Recall: clamp x=noisy_probe only. The Hopfield energy gradient pulls z_latent
    toward the nearest stored attractor during inference.

Experiments:
    A) Random binary ±1 patterns (D=64, P=7)
    B) Binarized MNIST digit prototypes (D=196, P=10)

Usage:
    python examples/storkey_hopfield_recall.py
    python examples/storkey_hopfield_recall.py --experiment binary
    python examples/storkey_hopfield_recall.py --experiment mnist
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import optax

from fabricpc.nodes import IdentityNode, StorkeyHopfield
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.initializers import NormalInitializer
from fabricpc.training import train
from fabricpc import setup_jax

setup_jax(platform="cpu")
jax.config.update("jax_default_prng_impl", "threefry2x32")

NOISE_LEVELS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Hopfield Pattern Recall Demo")
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["binary", "mnist", "all"],
        help="Which experiment to run (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------


class HopfieldRecallLoader:
    """Generates (noisy_probe, clean_pattern) pairs for PC training.

    Each epoch: for every stored pattern, generates `samples_per_pattern`
    noisy copies by flipping bits with probability `noise_prob`. Shuffles
    and yields minibatches.

    Args:
        patterns: (P, D) array with values in {-1, +1}.
        noise_prob: Probability of flipping each bit.
        samples_per_pattern: Number of noisy copies per pattern per epoch.
        batch_size: Minibatch size.
        seed: Base random seed (incremented each epoch for variety).
    """

    def __init__(
        self,
        patterns,
        noise_prob=0.15,
        samples_per_pattern=100,
        batch_size=64,
        seed=None,
    ):
        self.patterns = np.array(patterns, dtype=np.float32)
        self.noise_prob = noise_prob
        self.samples_per_pattern = samples_per_pattern
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0
        self.num_samples = len(patterns) * samples_per_pattern
        self._num_batches = self.num_samples // batch_size

    def __iter__(self):
        rng = np.random.default_rng(
            self.seed + self._epoch if self.seed is not None else None
        )
        self._epoch += 1

        P, D = self.patterns.shape
        clean = np.repeat(self.patterns, self.samples_per_pattern, axis=0)
        flip_mask = rng.random((len(clean), D)) < self.noise_prob
        noisy = clean.copy()
        noisy[flip_mask] *= -1.0

        indices = rng.permutation(len(clean))
        clean = clean[indices]
        noisy = noisy[indices]

        for start in range(0, len(clean) - self.batch_size + 1, self.batch_size):
            end = start + self.batch_size
            yield noisy[start:end], clean[start:end]

    def __len__(self):
        return self._num_batches


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------


def build_recall_graph(D, hopfield_strength=1.0, infer_steps=20, eta_infer=0.05):
    """Build 3-node recall graph: probe -> hopfield -> output.

    Recall quality is read from the Hopfield node's own settled z_latent —
    the attractor lives in that node's z-space. The unclamped output readout
    relaxes like any other node (error = z_latent - z_mu, its energy in the
    total): as the attractor pulls the Hopfield latent, the readout's z_mu
    moves ahead of its z_latent, and the resulting error feeds a transient
    top-down gradient back into the Hopfield latent until the readout
    catches up.
    """
    probe = IdentityNode(shape=(D,), name="probe")
    hopfield = StorkeyHopfield(
        shape=(D,),
        name="hopfield",
        hopfield_strength=hopfield_strength,
        use_bias=False,
        enforce_symmetry=True,
        zero_diagonal=False,
        weight_init=NormalInitializer(mean=0.0, std=0.01),
    )
    output = IdentityNode(shape=(D,), name="output")

    structure = graph(
        nodes=[probe, hopfield, output],
        edges=[
            Edge(source=probe, target=hopfield.slot("in")),
            Edge(source=hopfield, target=output.slot("in")),
        ],
        task_map=TaskMap(x=probe, y=output),
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
    )
    return structure


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_recall_model(
    patterns,
    D,
    hopfield_strength,
    num_epochs,
    batch_size,
    lr,
    noise_prob,
    samples_per_pattern,
    rng_key,
):
    """Train a Hopfield recall model on the given patterns."""
    structure = build_recall_graph(
        D,
        hopfield_strength=hopfield_strength,
        infer_steps=20,
        eta_infer=0.05,
    )

    rng_key, init_key = jax.random.split(rng_key)
    params = initialize_params(structure, init_key)
    optimizer = optax.adamw(lr, weight_decay=0.01)

    train_loader = HopfieldRecallLoader(
        patterns,
        noise_prob=noise_prob,
        samples_per_pattern=samples_per_pattern,
        batch_size=batch_size,
        seed=42,
    )

    train_config = {"num_epochs": num_epochs}

    rng_key, train_key = jax.random.split(rng_key)
    result = train(
        params,
        structure,
        train_loader,
        optimizer,
        train_config,
        train_key,
        verbose=True,
    )
    return result.params, structure


# ---------------------------------------------------------------------------
# Recall Evaluation
# ---------------------------------------------------------------------------


def evaluate_recall(
    params,
    train_structure,
    patterns,
    noise_levels,
    num_trials=50,
    infer_steps=100,
    eta_infer=0.05,
    hopfield_strength=1.0,
    rng_key=None,
):
    """Evaluate recall accuracy across noise levels.

    Builds a recall-specific graph with higher infer_steps, then for each
    noise level corrupts all patterns and runs inference-only.
    """
    D = patterns.shape[1]
    num_patterns = patterns.shape[0]

    recall_structure = build_recall_graph(
        D,
        hopfield_strength=hopfield_strength,
        infer_steps=infer_steps,
        eta_infer=eta_infer,
    )

    results = []
    for noise_prob in noise_levels:
        exact_recalls = 0
        total_cosine = 0.0
        total_bit_acc = 0.0
        total_samples = 0

        for trial in range(num_trials):
            rng_key, subkey, init_key = jax.random.split(rng_key, 3)

            rng_np = np.random.default_rng(int(subkey[0]) + trial)
            clean = patterns.copy()
            flip_mask = rng_np.random((num_patterns, D)) < noise_prob
            noisy = clean.copy()
            noisy[flip_mask] *= -1.0

            clamps = {"probe": jnp.array(noisy)}

            state = initialize_graph_state(
                recall_structure,
                num_patterns,
                init_key,
                clamps=clamps,
                params=params,
            )

            final_state = run_inference(params, state, clamps, recall_structure)

            recalled = np.array(final_state.nodes["hopfield"].z_latent)
            recalled_binary = np.sign(recalled)
            recalled_binary[recalled_binary == 0] = 1.0

            for i in range(num_patterns):
                # Exact recall: nearest stored pattern is the correct one
                distances = np.sum((patterns - recalled_binary[i : i + 1]) ** 2, axis=1)
                nearest_idx = np.argmin(distances)
                if nearest_idx == i:
                    exact_recalls += 1

                # Cosine similarity with correct pattern
                norm_r = np.linalg.norm(recalled[i])
                norm_p = np.linalg.norm(patterns[i])
                cos = np.dot(recalled[i], patterns[i]) / (norm_r * norm_p + 1e-10)
                total_cosine += cos

                # Bit accuracy
                bit_acc = np.mean(recalled_binary[i] == patterns[i])
                total_bit_acc += bit_acc

            total_samples += num_patterns

        results.append(
            {
                "noise": noise_prob,
                "exact_recall": exact_recalls / total_samples,
                "cosine_sim": total_cosine / total_samples,
                "bit_accuracy": total_bit_acc / total_samples,
            }
        )

        print(
            f"  noise={noise_prob:.2f}  exact={exact_recalls/total_samples*100:5.1f}%  "
            f"cosine={total_cosine/total_samples:.4f}  "
            f"bit_acc={total_bit_acc/total_samples*100:5.1f}%"
        )

    return results


# ---------------------------------------------------------------------------
# Pattern Generation
# ---------------------------------------------------------------------------


def generate_binary_patterns(num_patterns, D, rng):
    """Generate random binary ±1 patterns."""
    return 2.0 * rng.integers(0, 2, size=(num_patterns, D)).astype(np.float32) - 1.0


def generate_mnist_prototypes(num_digits=10, downsample_to=14):
    """Load MNIST, compute per-class mean, downsample, binarize to ±1.

    Returns (num_digits, downsample_to^2) array with values in {-1, +1}.
    """
    import tensorflow_datasets as tfds
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")

    ds = tfds.load("mnist:3.0.1", split="train", as_supervised=True)

    class_sums = np.zeros((10, 28, 28), dtype=np.float64)
    class_counts = np.zeros(10, dtype=np.int64)
    for img, label in ds:
        img_np = img.numpy().astype(np.float64).squeeze() / 255.0
        label_np = int(label.numpy())
        class_sums[label_np] += img_np
        class_counts[label_np] += 1

    prototypes = class_sums / class_counts[:, None, None]

    # Downsample via 2x2 block averaging (28->14)
    if downsample_to == 14:
        prototypes = prototypes.reshape(num_digits, 14, 2, 14, 2).mean(axis=(2, 4))

    flat = prototypes.reshape(num_digits, -1).astype(np.float32)
    binary = 2.0 * (flat > 0.5).astype(np.float32) - 1.0
    return binary


# ---------------------------------------------------------------------------
# Results Printing
# ---------------------------------------------------------------------------


def print_results_table(title, D, num_patterns, results):
    """Print a formatted table of recall results."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"  D={D}, P={num_patterns}, capacity≈{0.14 * D:.0f}")
    print(f"{'=' * 70}")
    print(f"{'Noise':>8} {'Exact Recall':>14} {'Cosine Sim':>12} {'Bit Accuracy':>14}")
    print(f"{'-' * 70}")
    for r in results:
        print(
            f"{r['noise']:>8.2f} {r['exact_recall']*100:>13.1f}% "
            f"{r['cosine_sim']:>12.4f} {r['bit_accuracy']*100:>13.1f}%"
        )
    print(f"{'-' * 70}")


# ---------------------------------------------------------------------------
# Experiment Runners
# ---------------------------------------------------------------------------


def run_binary_experiment(rng_key):
    """Experiment A: Random binary ±1 patterns."""
    D = 64
    num_patterns = 7
    hopfield_strength = 1.0

    print("\n" + "=" * 70)
    print("  Experiment A: Random Binary Patterns")
    print(f"  D={D}, P={num_patterns}, hopfield_strength={hopfield_strength}")
    print("=" * 70)

    rng = np.random.default_rng(42)
    patterns = generate_binary_patterns(num_patterns, D, rng)

    print("\nTraining...")
    rng_key, train_key = jax.random.split(rng_key)
    trained_params, structure = train_recall_model(
        patterns,
        D,
        hopfield_strength,
        num_epochs=30,
        batch_size=64,
        lr=0.001,
        noise_prob=0.15,
        samples_per_pattern=100,
        rng_key=train_key,
    )

    print("\nEvaluating recall...")
    rng_key, eval_key = jax.random.split(rng_key)
    results = evaluate_recall(
        trained_params,
        structure,
        patterns,
        NOISE_LEVELS,
        num_trials=50,
        infer_steps=100,
        eta_infer=0.05,
        hopfield_strength=hopfield_strength,
        rng_key=eval_key,
    )

    print_results_table(
        "Experiment A: Random Binary Patterns", D, num_patterns, results
    )
    return results


def run_mnist_experiment(rng_key):
    """Experiment B: Binarized MNIST digit prototypes."""
    D = 196
    num_patterns = 10
    hopfield_strength = 1.0

    print("\n" + "=" * 70)
    print("  Experiment B: MNIST Digit Prototypes (14x14)")
    print(f"  D={D}, P={num_patterns}, hopfield_strength={hopfield_strength}")
    print("=" * 70)

    print("Loading MNIST and generating prototypes...")
    patterns = generate_mnist_prototypes(num_digits=num_patterns, downsample_to=14)

    print(f"Generated {num_patterns} binarized prototypes, D={D}")
    for i in range(num_patterns):
        frac_pos = np.mean(patterns[i] == 1.0)
        print(f"  digit {i}: {frac_pos*100:.0f}% +1, {(1-frac_pos)*100:.0f}% -1")

    print("\nTraining...")
    rng_key, train_key = jax.random.split(rng_key)
    trained_params, structure = train_recall_model(
        patterns,
        D,
        hopfield_strength,
        num_epochs=50,
        batch_size=100,
        lr=0.001,
        noise_prob=0.15,
        samples_per_pattern=80,
        rng_key=train_key,
    )

    print("\nEvaluating recall...")
    rng_key, eval_key = jax.random.split(rng_key)
    results = evaluate_recall(
        trained_params,
        structure,
        patterns,
        NOISE_LEVELS,
        num_trials=30,
        infer_steps=100,
        eta_infer=0.05,
        hopfield_strength=hopfield_strength,
        rng_key=eval_key,
    )

    print_results_table(
        "Experiment B: MNIST Digit Prototypes (14x14)", D, num_patterns, results
    )
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    rng_key = jax.random.PRNGKey(args.seed)

    if args.experiment in ("binary", "all"):
        rng_key, exp_key = jax.random.split(rng_key)
        run_binary_experiment(exp_key)

    if args.experiment in ("mnist", "all"):
        rng_key, exp_key = jax.random.split(rng_key)
        run_mnist_experiment(exp_key)


if __name__ == "__main__":
    main()
