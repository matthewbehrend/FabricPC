"""
StorkeyHopfield vs MLP — Depth, Few-Shot, and Noise-Robustness Demo
====================================================================

Demonstrates that StorkeyHopfield behaves as a composable graph element in
FabricPC: substituting one or more Linear hidden layers with StorkeyHopfield
should produce accuracy gains that accumulate with the *number* of
substitutions, particularly when the input is noisy and the training set
has enough samples for the attractors to form usable class prototypes.

Architecture (same depth-4 graph across all arms, hidden width = 64)::

    "MLP"             pixels(784) -> 64(Linear) -> 64(Linear)          -> 64(Linear)          -> 10(softmax+CE)
    "1hopfield"       pixels(784) -> 64(Linear) -> 64(StorkeyHopfield) -> 64(Linear)          -> 10(softmax+CE)
    "1hopfield-late"  pixels(784) -> 64(Linear) -> 64(Linear)          -> 64(StorkeyHopfield) -> 10(softmax+CE)
    "2hopfield"       pixels(784) -> 64(Linear) -> 64(StorkeyHopfield) -> 64(StorkeyHopfield) -> 10(softmax+CE)

Position 0 is always Linear (the 784 -> 64 feature-extraction projection;
StorkeyHopfield is width-preserving and cannot reduce dimension). The
classifier head is always Linear with Softmax + CrossEntropy energy.

Hypothesis
----------
Accuracy gains are cumulative in the number of Linear -> StorkeyHopfield
substitutions in the hidden stack, and driven by the substitution count
rather than the substituted position. Three planned paired contrasts on the
per-trial accuracy differences:

    substitution 1:   acc(1hopfield) - acc(MLP)            > 0
    substitution 2:   acc(2hopfield) - acc(1hopfield)      > 0
    position:         acc(1hopfield-late) - acc(1hopfield) ~ 0

The 1hopfield-late arm is the position control: without it, substitution 2
would conflate adding a second Hopfield layer with occupying position 2 for
the first time (only the 2hopfield arm places a Hopfield layer there). The
per-cell cumulative claim ("substituting more layers helps further") is the
conjunction: both increment contrasts positive AND each starred (two-sided
paired t-test p<0.05). The K x noise grid is a *sensitivity analysis* of
where the per-substitution gain is positive: it grows with input noise,
requires K large enough for the attractors to form class prototypes, and
data scarcity alone is not sufficient to produce a Hopfield advantage. The
reported 2hopfield - MLP delta is the *total* effect (both substitutions
combined), which is informative descriptively but not tested as a planned
contrast.

Experiment grid: K (shots per class) x noise_std
    - K controls data scarcity: how many samples each arm trains on.
    - noise_std controls input corruption: more noise -> more reliance
      on attractor denoising.

Setup (one-time)::

    pip install -e ".[tfds]"

(installs ``tensorflow-datasets`` and ``tensorflow``, the dataset backend
used by FashionMnistLoader / FewShotLoader; declared as the ``tfds`` extra
in pyproject.toml. Both loaders call ``tf.config.set_visible_devices([],
"GPU")`` at construction so TF stays off the GPU and does not contend with
JAX.)

Usage::

    # Default sweep: all four arms, K in {500}, noise in {2.0}, n_trials=5,
    # epochs=1. Bare invocation is intentionally fast so that JIT
    # compilation, the tfds download, and the printed pipeline are visible
    # in well under a minute on a GPU. The documented Results block below
    # was produced with --n_trials 10 --num_epochs 20 on the full grid (see
    # the explicit command beneath it).
    python examples/storkey_hopfield_demo.py

    # All four arms, all three planned contrasts; K x noise grid; long run:
    python examples/storkey_hopfield_demo.py \\
        --networks 1hopfield 1hopfield-late 2hopfield \\
        --k_values 10,50,100,500 --noise_levels 0.0,0.5,1.0,2.0 \\
        --n_trials 10 --num_epochs 20

    # 2hopfield only (total-effect contrast against MLP, no increment or
    # position tests):
    python examples/storkey_hopfield_demo.py --networks 2hopfield \\
        --k_values 50,500 --noise_levels 0.0,2.0

Results (Fashion-MNIST; --strength 2.0, n_trials=10, num_epochs=20, the
full-grid command above). Each trial's seed flows through all four arms
and the runner rebuilds the loaders per arm from that seed, so the
per-trial difference vector is a clean within-trial paired contrast.
Stars are two-sided paired t-tests at p<0.05, uncorrected across
16 cells x 3 contrasts. The heatmaps are the three *planned* contrasts;
the total effect (2hopfield - MLP) is reported descriptively in the demo
output but is not tested.

    Delta Accuracy Heatmap (1hopfield - MLP, percentage points):

         K  n=0.0  n=0.5  n=1.0  n=2.0
    ----------------------------------
        10  -0.9*  -0.5   +1.2*  +4.1*
        50  -0.1   +0.8*  +3.3*  +8.1*
       100  -0.3   +0.9*  +3.3*  +8.5*
       500  +0.3   +1.3*  +3.3*  +7.0*

    Delta Accuracy Heatmap (2hopfield - 1hopfield, percentage points):

         K  n=0.0  n=0.5  n=1.0  n=2.0
    ----------------------------------
        10  -0.8   -0.3   +0.3   +1.9*
        50  +0.1   +0.2   +1.1*  +2.9*
       100  +0.3   +0.6   +1.4*  +3.1*
       500  +0.4*  +0.8*  +2.5*  +6.0*

    Delta Accuracy Heatmap (1hopfield-late - 1hopfield, percentage points):

         K  n=0.0  n=0.5  n=1.0  n=2.0
    ----------------------------------
        10  +0.7   +0.8*  +0.2   -0.9
        50  +0.3   -0.1   -0.8*  -3.1*
       100  +0.2   -0.2   -1.2*  -3.0*
       500  +0.1   -0.0   -0.2   +0.4

      * = significant at p<0.05 (two-sided paired t-test, uncorrected)

Cumulative gain (BOTH increment contrasts starred-positive in the same
cell) is attained in 8 of 16 cells: K in {50, 100, 500} crossed with
n in {1.0, 2.0}, plus K=500 at n=0.5 and K=10 at n=2.0. Both increments
grow with noise at every K, and at clean inputs (n=0.0) both are near
zero: with no noise to denoise, the attractor dynamics contribute
nothing, and at K=10 the first substitution significantly costs accuracy
(-0.9 pp). The position contrast supports count-over-position where the
increments are largest: at K=500 and at clean inputs the single
substitution is position-insensitive (deltas within +/-0.4 pp). At
K in {50, 100} under noise the late placement is significantly worse
(-0.8 to -3.1 pp): an early attractor denoises the representation
directly after the 784 -> 64 projection, before it propagates through
the remaining Linear layers. Even there the second increment stays
positive -- adding the late Hopfield on top of the early one helps,
although the late position alone underperforms. The descriptive total
effect (2hopfield - MLP) peaks at +13.0 pp at K=500, n=2.0.

Quick run results (default args, RTX3090, cuda13, jax 0.10.2)
  Cell summary:
    MLP       : 58.32%
    1hopfield : 62.86%
    1hopfield-late: 61.68%
    2hopfield : 64.26%
"""

import argparse
from typing import Dict, List, Sequence, Tuple

import numpy as np
import jax
import optax

from fabricpc.nodes import Linear, IdentityNode, NodeBase, StorkeyHopfield
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core import TanhActivation
from fabricpc.core.activations import SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import XavierInitializer
from fabricpc.training import train, evaluate
from fabricpc.experiments import ExperimentArm, PlannedMultiContrastExperiment
from fabricpc.utils.data.dataloader import (
    FashionMnistLoader,
    FewShotLoader,
    NoisyTestLoader,
)
from fabricpc import setup_jax

setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

BATCH_SIZE = 64
HIDDEN_WIDTH = 64

# Each entry lists the hidden-layer types between the input and the Linear
# softmax readout. Position 0 must be Linear (784 -> HIDDEN_WIDTH projection);
# StorkeyHopfield is width-preserving and cannot reduce dimension.
ARCH_CONFIGS: Dict[str, List[str]] = {
    "MLP": ["Linear", "Linear", "Linear"],
    "1hopfield": ["Linear", "StorkeyHopfield", "Linear"],
    "1hopfield-late": ["Linear", "Linear", "StorkeyHopfield"],
    "2hopfield": ["Linear", "StorkeyHopfield", "StorkeyHopfield"],
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Depth + Few-Shot + Noise: StorkeyHopfield as a composable graph "
            "element on Fashion-MNIST."
        )
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="500",
        help="Comma-separated K (shots per class) values (default: 500)",
    )
    parser.add_argument(
        "--noise_levels",
        type=str,
        default="2.0",
        help="Comma-separated noise std values (default: 2.0)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of paired trials per condition (default: 5)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help=(
            "Training epochs per trial (default: 1). The bare command is "
            "intentionally fast; the documented Results block uses 20."
        ),
    )
    parser.add_argument(
        "--strength",
        type=str,
        default="2.0",
        help="Hopfield strength: float >= 0 or 'None' for learnable (default: 2.0)",
    )
    parser.add_argument(
        "--networks",
        nargs="+",
        type=str,
        default=["1hopfield", "1hopfield-late", "2hopfield"],
        choices=["1hopfield", "1hopfield-late", "2hopfield"],
        help=(
            "Which Hopfield-substituted variants to run alongside the MLP "
            "baseline. MLP is always included implicitly. Default: all three "
            "(1hopfield-late is the position control for the second "
            "increment)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-epoch training output",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------


def make_model_factory(
    layer_types: Sequence[str],
    hopfield_strength,
    width: int = HIDDEN_WIDTH,
):
    """Return a ``model_factory(rng_key)`` for a classifier whose hidden
    stack is defined by ``layer_types``.

    Architecture (depth-3 hidden + readout)::

        pixels(784) -> hidden0(width) -> hidden1(width) -> hidden2(width) -> class(10)

    Each hidden layer is either ``Linear(tanh)`` or ``StorkeyHopfield`` per
    the corresponding entry of ``layer_types``. **Position 0 must be
    Linear** because ``StorkeyHopfield`` is width-preserving (square W,
    ``in_dim == out_dim``) and cannot perform the 784 -> width projection
    -- the call below raises ``ValueError`` instead of failing later with
    an opaque shape error from graph wiring.

    The classifier head is always Linear with Softmax + CrossEntropy energy.
    """
    if not layer_types:
        raise ValueError("layer_types must contain at least one layer.")
    if layer_types[0] != "Linear":
        raise ValueError(
            f"Position 0 of layer_types must be 'Linear' (the 784 -> {width} "
            f"feature-extraction projection); got {layer_types[0]!r}. "
            "StorkeyHopfield is width-preserving and cannot change dimension."
        )

    def create_model(rng_key):
        pixels = IdentityNode(shape=(784,), name="pixels")
        # Annotate as List[NodeBase] so Pylance allows appending Linear /
        # StorkeyHopfield without inferring the narrower IdentityNode type
        # from the initial single-element list.
        nodes: List[NodeBase] = [pixels]
        edges: List[Edge] = []
        prev: NodeBase = pixels

        for i, ltype in enumerate(layer_types):
            if ltype == "StorkeyHopfield":
                layer = StorkeyHopfield(
                    shape=(width,),
                    name=f"hidden{i}",
                    hopfield_strength=hopfield_strength,
                )
            elif ltype == "Linear":
                layer = Linear(
                    shape=(width,),
                    activation=TanhActivation(),
                    name=f"hidden{i}",
                    weight_init=XavierInitializer(),
                )
            else:
                raise ValueError(
                    f"Unknown layer type {ltype!r} at position {i}; "
                    "expected 'Linear' or 'StorkeyHopfield'."
                )
            nodes.append(layer)
            edges.append(Edge(source=prev, target=layer.slot("in")))
            prev = layer

        output = Linear(
            shape=(10,),
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            name="class",
            weight_init=XavierInitializer(),
        )
        nodes.append(output)
        edges.append(Edge(source=prev, target=output.slot("in")))

        structure = graph(
            nodes=nodes,
            edges=edges,
            task_map=TaskMap(x=pixels, y=output),
            inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    return create_model


def arch_str(layer_types: Sequence[str], width: int = HIDDEN_WIDTH) -> str:
    """Human-readable architecture string for the header print."""
    parts = ["784"]
    for ltype in layer_types:
        node = "StorkeyHopfield" if ltype == "StorkeyHopfield" else "Linear"
        parts.append(f"{width}({node}, tanh)")
    parts.append("10(softmax, CE)")
    return " -> ".join(parts)


# ---------------------------------------------------------------------------
# Data Loader Factory
# ---------------------------------------------------------------------------


def make_data_factory(k_per_class, noise_std, batch_size):
    """Return a ``data_loader_factory(seed)`` for the experiment runner.

    Each trial seed produces a deterministic K-shot subsample (function of
    trial seed only) and an independent test-noise realization (function of
    trial seed and noise_std). Identical training data is reused across
    noise levels for a clean isolation of the noise effect, while different
    noise levels draw independent realizations rather than rescaled copies
    of one shared noise stream.

    Pairing of the *minibatch order* across arms within a trial holds
    because the runner calls this factory once per arm with the same trial
    seed and the loaders are deterministic in their seeds.
    """
    # Stable integer fingerprint of noise_std used as RNG entropy.
    noise_level_id = int(round(noise_std * 1_000_000))

    def factory(seed):
        train_seed = int(np.random.SeedSequence([seed, 0]).generate_state(1)[0])
        noise_seed = int(
            np.random.SeedSequence([seed, 1, noise_level_id]).generate_state(1)[0]
        )

        train_loader = FewShotLoader(
            dataset_name="fashion_mnist",
            split="train",
            k_per_class=k_per_class,
            batch_size=batch_size,
            num_classes=10,
            shuffle=True,
            seed=train_seed,
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
            seed=noise_seed,
        )

        return train_loader, test_loader

    return factory


# ---------------------------------------------------------------------------
# Planned contrasts
# ---------------------------------------------------------------------------


def build_contrasts(hopfield_arms_to_run: List[str]) -> List[Tuple[str, str]]:
    """Build the planned-contrast list from the selected Hopfield arms.

    The full hypothesis decomposes into two ordered adjacent increments plus
    one position control::

        substitution 1:   1hopfield - MLP
        substitution 2:   2hopfield - 1hopfield
        position:         1hopfield-late - 1hopfield

    Partial ``--networks`` selections keep each contrast adjacent in
    substitution count where possible:

    - Without 1hopfield, 1hopfield-late takes over as the single-substitution
      arm: its contrast is against MLP, and 2hopfield contrasts against it
      (the second increment along the position-2 path).
    - ``--networks 2hopfield`` alone can test neither increment; it falls
      back to the total-effect contrast ``2hopfield - MLP``, which is
      reported and starred. main() prints a caveat distinguishing this case
      from the increment test.
    """
    has_1hop = "1hopfield" in hopfield_arms_to_run
    has_late = "1hopfield-late" in hopfield_arms_to_run
    has_2hop = "2hopfield" in hopfield_arms_to_run

    contrasts: List[Tuple[str, str]] = []
    if has_1hop:
        contrasts.append(("1hopfield", "MLP"))
        if has_2hop:
            contrasts.append(("2hopfield", "1hopfield"))
        if has_late:
            contrasts.append(("1hopfield-late", "1hopfield"))
    elif has_late:
        contrasts.append(("1hopfield-late", "MLP"))
        if has_2hop:
            contrasts.append(("2hopfield", "1hopfield-late"))
    elif has_2hop:
        contrasts.append(("2hopfield", "MLP"))
    return contrasts


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}"


def print_per_arm_table(arm_names: List[str], grid: List[Dict]):
    print()
    print("Per-arm accuracy (mean +/- SE, %):")
    cols = [f"{'K':>5}", f"{'Noise':>6}"]
    for name in arm_names:
        cols.append(f"{name:>14}")
    header = "  ".join(cols)
    print(header)
    print("-" * len(header))
    for r in grid:
        parts = [f"{r['k']:>5}", f"{r['noise']:>6.1f}"]
        for name in arm_names:
            cell = f"{r[f'{name}_mean'] * 100:6.2f}+/-" f"{r[f'{name}_se'] * 100:.2f}"
            parts.append(f"{cell:>14}")
        print("  ".join(parts))


def print_contrasts_table(
    contrasts: List[Tuple[str, str]], grid: List[Dict], n_trials: int
):
    if not contrasts:
        return
    print()
    print("Planned contrasts (paired two-sided t-test on per-trial differences):")
    # Delta-column width per contrast: long labels (e.g.
    # "1hopfield-late-1hopfield D%") widen their own column only.
    delta_widths = [max(len(f"{a}-{b} D%"), 16) for a, b in contrasts]
    cols = [f"{'K':>5}", f"{'Noise':>6}"]
    for (a, b), w in zip(contrasts, delta_widths):
        label = f"{a}-{b}"
        cols.append(f"{label + ' D%':>{w}}")
        cols.append(f"{'p':>8}")
        cols.append(f"{'sig':>4}")
        cols.append(f"{'d':>7}")
    header = "  ".join(cols)
    print(header)
    print("-" * len(header))
    for r in grid:
        parts = [f"{r['k']:>5}", f"{r['noise']:>6.1f}"]
        for (a, b), w in zip(contrasts, delta_widths):
            key = f"{a}-{b}"
            delta_str = _pct(r[f"{key}_delta"])
            parts.append(f"{delta_str:>{w}}")
            p = r[f"{key}_pval"]
            parts.append(f"{p:8.4f}" if not np.isnan(p) else f"{'n/a':>8}")
            parts.append(f"{'*' if r[f'{key}_sig'] else '':>4}")
            d = r[f"{key}_d"]
            parts.append(f"{d:7.3f}" if not np.isnan(d) else f"{'n/a':>7}")
        print("  ".join(parts))
    print()
    print(_legend_text(n_trials, len(grid), contrasts))


def print_descriptive_total_delta(grid: List[Dict]):
    """Print the descriptive 2hopfield - MLP total-effect delta when it is
    NOT a planned contrast (i.e. when 2hopfield runs alongside a
    single-substitution arm, so 2hopfield's planned contrast is an increment
    and the total effect is reported only)."""
    if not grid or "2hopfield-MLP_delta_descriptive" not in grid[0]:
        return
    print()
    print("Reported-only (descriptive, NOT a planned contrast):")
    print(f"{'K':>5}  {'Noise':>6}  {'2hopfield-MLP D%':>18}  {'SE':>8}")
    print("-" * 44)
    for r in grid:
        d = r["2hopfield-MLP_delta_descriptive"]
        se = r["2hopfield-MLP_se_descriptive"]
        print(f"{r['k']:>5}  {r['noise']:>6.1f}  " f"{_pct(d):>18}  {se * 100:>7.2f}")


def _legend_text(n_trials: int, n_cells: int, contrasts: List[Tuple[str, str]]) -> str:
    n_contrasts = len(contrasts)
    total_tests = n_cells * n_contrasts
    cumulative_rule = ""
    if ("1hopfield", "MLP") in contrasts and ("2hopfield", "1hopfield") in contrasts:
        cumulative_rule = (
            "  Per-cell criterion for 'cumulative gain': BOTH increment contrasts\n"
            "  (1hopfield-MLP and 2hopfield-1hopfield) starred with positive\n"
            "  deltas (intersection-union test of the conjunction, valid at\n"
            "  level alpha under arbitrary dependence).\n"
        )
    return (
        f"Legend:\n"
        f"  * = two-sided paired t-test p<0.05 on the per-trial difference\n"
        f"      vector, uncorrected across {n_cells} cells x {n_contrasts} "
        f"contrasts = {total_tests} tests.\n"
        f"      Per-cell stars are exploratory across the grid.\n"
        f"{cumulative_rule}"
        f"  n_trials = {n_trials}; Cohen's d is the paired d on the same vector."
    )


def print_heatmaps(
    contrasts: List[Tuple[str, str]],
    k_values: List[int],
    noise_levels: List[float],
    grid: List[Dict],
):
    if len(k_values) <= 1 or len(noise_levels) <= 1:
        return
    for a, b in contrasts:
        print()
        print(f"Delta Accuracy Heatmap ({a} - {b}, percentage points):")
        print()
        noise_header = f"{'K':>6}" + "".join(f"  n={n:.1f}" for n in noise_levels)
        print(noise_header)
        print("-" * len(noise_header))
        for k in k_values:
            row_str = f"{k:>6}"
            for noise_std in noise_levels:
                match = [r for r in grid if r["k"] == k and r["noise"] == noise_std]
                if match:
                    key = f"{a}-{b}"
                    d = match[0][f"{key}_delta"] * 100
                    sig = "*" if match[0][f"{key}_sig"] else " "
                    row_str += f" {d:+5.1f}{sig}"
                else:
                    row_str += "    n/a"
            print(row_str)
    print()
    print("  * = significant at p<0.05 (two-sided paired t-test, uncorrected)")


# ---------------------------------------------------------------------------
# Main Experiment
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]
    noise_levels = [float(n) for n in args.noise_levels.split(",")]

    if args.strength.lower() == "none":
        hopfield_strength = None
    else:
        hopfield_strength = float(args.strength)

    optimizer = optax.adamw(0.001, weight_decay=0.1)
    train_config = {"num_epochs": args.num_epochs}

    strength_label = (
        "learnable" if hopfield_strength is None else f"{hopfield_strength}"
    )

    # MLP is always the baseline. argparse choices restricts to known names;
    # de-duplicate while preserving input order.
    seen = set()
    hopfield_arms_to_run: List[str] = []
    for name in args.networks:
        if name not in seen:
            seen.add(name)
            hopfield_arms_to_run.append(name)
    arm_names_in_run: List[str] = ["MLP"] + hopfield_arms_to_run

    def build_arm(arm_name: str) -> ExperimentArm:
        return ExperimentArm(
            name=arm_name,
            model_factory=make_model_factory(
                ARCH_CONFIGS[arm_name], hopfield_strength, HIDDEN_WIDTH
            ),
            train_fn=train,
            eval_fn=evaluate,
            optimizer=optimizer,
            train_config=train_config,
        )

    arms = [build_arm(name) for name in arm_names_in_run]
    contrasts = build_contrasts(hopfield_arms_to_run)

    print("=" * 78)
    print("StorkeyHopfield depth comparison on Fashion-MNIST")
    print("=" * 78)
    print("Dataset: Fashion-MNIST")
    for name in arm_names_in_run:
        print(f"  {name:<10}: {arch_str(ARCH_CONFIGS[name], HIDDEN_WIDTH)}")
    print(f"Hopfield strength: {strength_label}")
    print(f"Epochs per trial: {args.num_epochs}")
    print(f"Trials per condition: {args.n_trials}")
    print(f"K values (number of examples per class): {k_values}")
    print(f"Noise levels (std dev): {noise_levels}")
    print(f"Planned contrasts: {contrasts}")
    if ("2hopfield", "MLP") in contrasts:
        print()
        print(
            "NOTE: --networks 2hopfield without a single-substitution arm\n"
            "falls back to the total-effect contrast (2hopfield - MLP), not\n"
            "the per-substitution increment. Pass '--networks 1hopfield\n"
            "2hopfield' (or add 1hopfield-late) to test the ordered\n"
            "increments separately."
        )
    print()

    # ---- run the K x noise grid -----------------------------------------------

    grid_results: List[Dict] = []
    # The 2hopfield-MLP total effect is reported descriptively whenever
    # 2hopfield's planned contrast is an increment (i.e. not itself vs MLP).
    report_descriptive_total = (
        "2hopfield" in hopfield_arms_to_run and ("2hopfield", "MLP") not in contrasts
    )

    for k in k_values:
        for noise_std in noise_levels:
            print(f"\n{'='*78}")
            print(f"  K={k} shots/class, noise_std={noise_std}")
            print(f"{'='*78}")

            data_factory = make_data_factory(k, noise_std, BATCH_SIZE)

            runner = PlannedMultiContrastExperiment(
                arms=arms,
                contrasts=contrasts,
                metric="accuracy",
                data_loader_factory=data_factory,
                n_trials=args.n_trials,
                seed_offset=0,
                verbose=args.verbose,
            )
            results = runner.run()

            row: Dict = {"k": k, "noise": noise_std}
            for name in arm_names_in_run:
                acc = results.per_arm_metrics(name)
                row[f"{name}_mean"] = float(np.mean(acc))
                row[f"{name}_se"] = (
                    float(np.std(acc, ddof=1) / np.sqrt(len(acc)))
                    if len(acc) > 1
                    else 0.0
                )

            # One entry per declared contrast (these get stars). With a
            # single trial the runner reports NaN test statistics, which the
            # print helpers render as n/a.
            for c in results.contrast_results():
                key = f"{c.arm_a}-{c.arm_b}"
                row[f"{key}_delta"] = c.mean_diff
                row[f"{key}_se"] = c.se_diff
                row[f"{key}_pval"] = c.p_value
                row[f"{key}_sig"] = c.significant_at_05
                row[f"{key}_d"] = c.cohens_d

            # Reported-only descriptive total effect (only when 2hopfield's
            # planned contrast is an increment rather than vs MLP).
            if report_descriptive_total:
                d = results.delta("2hopfield", "MLP")
                row["2hopfield-MLP_delta_descriptive"] = d.mean
                row["2hopfield-MLP_se_descriptive"] = d.se

            grid_results.append(row)

            # Per-cell summary line.
            print("  Cell summary:")
            for name in arm_names_in_run:
                print(f"    {name:<10}: {row[f'{name}_mean'] * 100:.2f}%")
            for a, b in contrasts:
                key = f"{a}-{b}"
                p = row[f"{key}_pval"]
                p_str = "p=n/a" if np.isnan(p) else f"p={p:.4f}"
                star = "*" if row[f"{key}_sig"] else ""
                print(
                    f"    {key:<22} delta={row[f'{key}_delta'] * 100:+.2f}%  "
                    f"{p_str} {star}"
                )

    # ---- summary tables -------------------------------------------------------

    print("\n\n")
    print("=" * 78)
    print(
        "RESULTS SUMMARY  "
        f"(strength={strength_label}, n_trials={args.n_trials}, "
        f"epochs={args.num_epochs})"
    )
    print("=" * 78)

    print_per_arm_table(arm_names_in_run, grid_results)
    print_contrasts_table(contrasts, grid_results, args.n_trials)
    if report_descriptive_total:
        print_descriptive_total_delta(grid_results)
    print_heatmaps(contrasts, k_values, noise_levels, grid_results)

    print()
    print("=" * 78)


if __name__ == "__main__":
    main()
