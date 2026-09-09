"""
sPC vs ePC Inference on a Cyclic MNIST Graph
============================================

Statistically compares predictive coding inference methods on one cyclic
5-node graph:

- sPC (``InferenceSGD``): relaxes the latents z by gradient descent on the
  exact graph energy; cycles need no approximation.
- ePC (``EPCInference``): relaxes the errors ε (one leaf per node) on the
  unrolled visit schedule. On a cyclic graph the schedule visits each cycle
  ``unroll`` times, the same ε is re-injected at every visit, and each node's
  energy enters at its final visit, so ePC minimizes an unrolled
  approximation whose fidelity grows with ``unroll``. For sPC, ``unroll``
  only shapes the feedforward state-initialization schedule.
- ePC→sPC (``InferenceSchedule``): the ePC stage settles cheaply, then a
  short sPC stage refines on the exact graph energy starting from ePC's
  finalized state (the segment boundary re-derives the latents so
  z_latent = z_mu + ε at ePC's settled point).

The chosen solver is built into the graph (``graph(..., inference=...)``),
so training and evaluation both use it.

Architecture::

    pixels ──→ hidden1 ──→ hidden2 ──→ class
                 ↑             │
                 └── h2_back ←─┘  (cycle)

Node schedule:
p -> h1 -> h2 -> h2_back -> h1 (cycle) -> h2 (cycle) -> h2_back (cycle) -> h1 (cycle) -> h2 (cycle)...  -> class
Nodes are not duplicated, only revisited with updated inputs. The cycles are unrolled to a fixed depth.

Statistical design
------------------
All arms run in one ``PlannedMultiContrastExperiment``: trial i uses seed
i*1000 for every arm's loaders and model RNG, so arms are paired by
construction. Only the method question carries planned t-tests; the unroll
grid is a dose-response sweep and is reported descriptively, because
successive-pair tests share arms, are positively correlated, and chain
low-power steps that can hide an endpoint difference.

- ``--inference both``: one planned contrast (ePC-u vs sPC) per unroll
  value, Holm-Bonferroni corrected across that family. Unroll deltas are
  descriptive (mean ± SE vs the smallest unroll, no test).
- ``--inference epc_spc``: sPC baseline plus, per unroll value, an ePC
  reference arm and a composed ePC→sPC arm. Planned family: (ePC+sPC-u vs
  sPC) per unroll, Holm-corrected; composed-vs-ePC and ePC-vs-sPC deltas
  are descriptive.
- ``--inference epc`` with exactly two unroll values: one planned endpoint
  contrast (u_hi vs u_lo). With more values: descriptive only; confirm a
  chosen pair in a follow-up ``--inference epc --unroll <a> <b>`` run.
- ``--inference spc``: single arm, descriptive statistics only.

The comparison metric is accuracy: the eval ``energy`` metric drifts with
``infer_steps × eta_infer`` and cannot compare solvers.

Usage:
    python examples/mnist_cyclic_graph.py                        # both sPC and ePC, unroll 2
    python examples/mnist_cyclic_graph.py --unroll 1 2 4 --n_trials 10
    python examples/mnist_cyclic_graph.py --inference epc --unroll 1 4
    python examples/mnist_cyclic_graph.py --inference spc
    python examples/mnist_cyclic_graph.py --inference epc_spc --unroll 1 --refine_steps 5

sPC and ePC carry independent, per-graph-tuned CLI defaults: sPC eta 0.15 /
50 steps, ePC eta 0.05 / 5 steps.

--- Per-arm accuracy (mean +/- SE over trials) ---
arm          accuracy%          infer steps
sPC-u2       92.36 +/- 0.17     50
ePC-u1       88.41 +/- 0.22     5
ePC-u2       92.89 +/- 0.18     5
ePC-u4       92.85 +/- 0.11     5

--- Planned contrasts (paired t-tests, Holm-corrected family of 3, alpha=0.05) ---
contrast                 diff pp            t         p         cohens_d   holm
ePC-u1 - sPC-u2          -3.95 +/- 0.23     -16.848   0.0001    -7.535     YES
ePC-u2 - sPC-u2          +0.54 +/- 0.16     3.384     0.0277    1.513      YES
ePC-u4 - sPC-u2          +0.50 +/- 0.14     3.609     0.0226    1.614      YES

The second cycle traversal matters, further traversals are within noise.

Composed ePC→sPC (``--inference epc_spc``)
--- Per-arm accuracy (mean +/- SE over trials) ---
arm          accuracy%          infer steps
sPC-u2       92.36 +/- 0.17     50
ePC-u1       88.41 +/- 0.22     5
ePC+sPC-u1   91.20 +/- 0.11     10

State settles quickly on a single unroll with ePC then refines with 5 steps of sPC.
The sPC refinement recovers most of the task performance lost to the ePC approximation,
but not as well as two unroll traversals of ePC alone.
"""

import argparse

import jax
import numpy as np
import optax

from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import (
    SigmoidActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core import EPCInference, InferenceSGD, InferenceSchedule
from fabricpc.training import train, evaluate
from fabricpc.experiments import ExperimentArm, PlannedMultiContrastExperiment
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc import setup_jax

setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")

# Training hyperparameters (shared by all arms)
optimizer = optax.adamw(0.001, weight_decay=0.001)
batch_size = 200


def parse_args():
    parser = argparse.ArgumentParser(
        description="Statistical comparison of sPC vs ePC inference on a "
        "cyclic MNIST graph"
    )
    parser.add_argument(
        "--inference",
        choices=["spc", "epc", "epc_spc", "both"],
        default="both",
        help="Inference method for training and eval; 'both' runs an sPC arm "
        "plus one ePC arm per --unroll value; 'epc_spc' runs the sPC "
        "baseline plus, per --unroll value, an ePC reference arm and a "
        "composed ePC->sPC refinement arm (default: both)",
    )
    parser.add_argument(
        "--unroll",
        type=int,
        nargs="+",
        default=[2],
        help="Cycle traversal counts for the ePC arms; one arm per value "
        "(default: 2)",
    )
    parser.add_argument(
        "--spc_unroll",
        type=int,
        default=2,
        help="Cycle traversal count for the sPC arm's graph build; affects "
        "only state initialization (default: 2)",
    )
    parser.add_argument(
        "--spc_eta",
        type=float,
        default=0.15,
        help="sPC inference learning rate (default: 0.15)",
    )
    parser.add_argument(
        "--spc_steps",
        type=int,
        default=50,
        help="sPC inference steps (default: 50)",
    )
    parser.add_argument(
        "--epc_eta",
        type=float,
        default=0.05,
        help="ePC inference learning rate (default: 0.05, tuned for this graph)",
    )
    parser.add_argument(
        "--epc_steps",
        type=int,
        default=5,
        help="ePC inference steps (default: 5, tuned for this graph)",
    )
    parser.add_argument(
        "--refine_eta",
        type=float,
        default=0.15,
        help="sPC refinement-stage learning rate for the composed epc_spc "
        "arms (default: 0.15)",
    )
    parser.add_argument(
        "--refine_steps",
        type=int,
        default=5,
        help="sPC refinement-stage steps for the composed arms (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Training epochs per trial (default: 1)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=5,
        help="Number of paired trials (default: 5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-epoch training output for each trial",
    )
    return parser.parse_args()


def make_spc(args):
    return InferenceSGD(eta_infer=args.spc_eta, infer_steps=args.spc_steps)


def make_epc(args):
    return EPCInference(eta_infer=args.epc_eta, infer_steps=args.epc_steps)


def make_epc_spc(args):
    """ePC stage then a short sPC refinement on the exact graph energy."""
    return InferenceSchedule(
        make_epc(args),
        InferenceSGD(eta_infer=args.refine_eta, infer_steps=args.refine_steps),
    )


# fmt: off
def make_model_factory(inference, unroll):
    """Cyclic-graph factory closed over the solver and traversal count; the
    experiment runner calls it with only the trial's RNG key."""
    def factory(rng_key):
        pixels           = Linear(shape=(784,), activation=SigmoidActivation(), name="pixels")
        hidden1          = Linear(shape=(256,), activation=SigmoidActivation(), name="hidden1")
        hidden2_back     = Linear(shape=(64,),  activation=SigmoidActivation(), name="hidden2_back")
        hidden2          = Linear(shape=(64,),  activation=SigmoidActivation(), name="hidden2")
        output           = Linear(shape=(10,),  activation=SoftmaxActivation(), energy=CrossEntropyEnergy(), name="class")

        structure = graph(
            nodes=[pixels, hidden1, hidden2_back, hidden2, output],
            edges=[
                Edge(source=pixels,          target=hidden1.slot("in")),
                Edge(source=hidden1,         target=hidden2.slot("in")),
                Edge(source=hidden2,         target=output.slot("in")),
                # Cycle
                Edge(source=hidden2,         target=hidden2_back.slot("in")),
                Edge(source=hidden2_back,    target=hidden1.slot("in")),
            ],
            task_map=TaskMap(x=pixels, y=output),
            inference=inference,
            unroll=unroll,
        )
        params = initialize_params(structure, rng_key)
        return params, structure
    return factory
# fmt: on


def build_arms(args, train_config):
    """Arm specs plus the planned-contrast family.

    Only the method question is tested: 'both' declares (ePC-u, sPC) per
    unroll value and 'epc_spc' declares (ePC+sPC-u, sPC) per unroll value
    (each Holm-corrected in the report); 'epc' with exactly two unroll
    values declares the endpoint contrast. Larger unroll grids and the
    epc_spc component decompositions are reported descriptively.
    """
    unrolls = sorted(set(args.unroll))
    for u in unrolls + [args.spc_unroll]:
        if u < 1:
            raise ValueError(f"unroll values must be >= 1; got {u}")

    spc_name = f"sPC-u{args.spc_unroll}"
    specs = []  # (name, inference, unroll)
    if args.inference in ("spc", "both", "epc_spc"):
        specs.append((spc_name, make_spc(args), args.spc_unroll))
    if args.inference in ("epc", "both", "epc_spc"):
        epc = make_epc(args)
        specs.extend((f"ePC-u{u}", epc, u) for u in unrolls)
    if args.inference == "epc_spc":
        composed = make_epc_spc(args)
        specs.extend((f"ePC+sPC-u{u}", composed, u) for u in unrolls)

    contrasts = []
    if args.inference == "both":
        contrasts = [(f"ePC-u{u}", spc_name) for u in unrolls]
    elif args.inference == "epc_spc":
        contrasts = [(f"ePC+sPC-u{u}", spc_name) for u in unrolls]
    elif args.inference == "epc" and len(unrolls) == 2:
        contrasts = [(f"ePC-u{unrolls[1]}", f"ePC-u{unrolls[0]}")]

    arms = [
        ExperimentArm(
            name=name,
            model_factory=make_model_factory(inference, unroll),
            train_fn=train,
            eval_fn=evaluate,
            optimizer=optimizer,
            train_config=train_config,
        )
        for name, inference, unroll in specs
    ]
    return arms, specs, contrasts, unrolls


def holm_reject(p_values, alpha=0.05):
    """Holm-Bonferroni step-down rejection decisions for one test family."""
    m = len(p_values)
    order = np.argsort(p_values)
    reject = [False] * m
    for rank, idx in enumerate(order):
        p = p_values[idx]
        if np.isnan(p) or p > alpha / (m - rank):
            break
        reject[idx] = True
    return reject


def print_report(results, specs, contrasts, unrolls, args):
    n = results.n_trials
    steps_per_update = {
        name: sum(steps for _, steps in inference.segments())
        for name, inference, _ in specs
    }

    print()
    print("--- Per-arm accuracy (mean +/- SE over trials) ---")
    print(f"{'arm':<12} {'accuracy%':<18} {'infer steps'}")
    for name in results.arm_names:
        acc = results.per_arm_metrics(name) * 100
        se = acc.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        print(
            f"{name:<12} {f'{acc.mean():.2f} +/- {se:.2f}':<18} "
            f"{steps_per_update[name]}"
        )

    print()
    print("--- Per-trial accuracy% ---")
    print(
        f"{'trial':<7}{'seed':<8}"
        + "".join(f"{name:<12}" for name in results.arm_names)
    )
    for i in range(n):
        row = f"{i + 1:<7}{results.seeds[i]:<8}"
        for name in results.arm_names:
            row += f"{results.per_arm_metrics(name)[i] * 100:<12.2f}"
        print(row)

    if contrasts:
        contrast_results = results.contrast_results()
        reject = holm_reject([c.p_value for c in contrast_results])
        print()
        print(
            "--- Planned contrasts (paired t-tests, Holm-corrected "
            f"family of {len(contrast_results)}, alpha=0.05) ---"
        )
        print(
            f"{'contrast':<24} {'diff pp':<18} {'t':<9} {'p':<9} "
            f"{'cohens_d':<10} {'holm'}"
        )
        for c, rej in zip(contrast_results, reject):
            diff = f"{c.mean_diff * 100:+.2f} +/- {c.se_diff * 100:.2f}"
            print(
                f"{c.arm_a + ' - ' + c.arm_b:<24} {diff:<18} "
                f"{c.t_statistic:<9.3f} {c.p_value:<9.4f} "
                f"{c.cohens_d:<10.3f} {'YES' if rej else 'no'}"
            )

    if args.inference == "epc_spc":
        spc_name = f"sPC-u{args.spc_unroll}"
        print()
        print("--- Composed vs components (descriptive, no test) ---")
        for u in unrolls:
            for a, b in [
                (f"ePC+sPC-u{u}", f"ePC-u{u}"),
                (f"ePC-u{u}", spc_name),
            ]:
                d = results.delta(a, b)
                print(f"{f'{a} - {b}':<24} {d.mean * 100:+.2f} +/- {d.se * 100:.2f} pp")

    # Unroll dose-response: descriptive deltas, no test. Skipped when the
    # endpoint pair is already the declared contrast (epc mode, 2 values).
    endpoint_tested = args.inference == "epc" and len(unrolls) == 2
    if (
        args.inference in ("epc", "both", "epc_spc")
        and len(unrolls) > 1
        and not endpoint_tested
    ):
        prefixes = ["ePC"]
        if args.inference == "epc_spc":
            prefixes.append("ePC+sPC")
        print()
        print("--- Unroll dose-response (descriptive, no test) ---")
        for prefix in prefixes:
            base = f"{prefix}-u{unrolls[0]}"
            for u in unrolls[1:]:
                d = results.delta(f"{prefix}-u{u}", base)
                print(
                    f"{f'{prefix}-u{u} - {base}':<24} "
                    f"{d.mean * 100:+.2f} +/- {d.se * 100:.2f} pp"
                )
        print(
            "Confirm a chosen pair with a follow-up run: "
            "--inference epc --unroll <a> <b>"
        )


def main():
    args = parse_args()
    train_config = {"num_epochs": args.epochs}
    arms, specs, contrasts, unrolls = build_arms(args, train_config)

    print("=" * 70)
    print("sPC vs ePC inference on a cyclic MNIST graph")
    print("=" * 70)
    print("Graph: 784 -> [256 -> 64 -> 64_lat -> 256 cycle] -> 10  (5 nodes, 5 edges)")
    print("Arms:")
    for name, inference, unroll in specs:
        desc = " -> ".join(
            f"{type(solver).__name__}(eta_infer={solver.config['eta_infer']}, "
            f"infer_steps={steps})"
            for solver, steps in inference.segments()
        )
        print(f"  {name}: {desc}, unroll={unroll}")
    print(
        f"Epochs per trial: {args.epochs} | Trials: {args.n_trials} | "
        f"Batch size: {batch_size}"
    )
    if contrasts:
        print("Planned contrasts: " + ", ".join(f"{a} vs {b}" for a, b in contrasts))
    print()

    runner = PlannedMultiContrastExperiment(
        arms=arms,
        contrasts=contrasts,
        metric="accuracy",
        data_loader_factory=lambda seed: (
            MnistLoader(
                "train",
                batch_size=batch_size,
                tensor_format="flat",
                shuffle=True,
                seed=seed,
            ),
            MnistLoader(
                "test",
                batch_size=batch_size,
                tensor_format="flat",
                shuffle=False,
            ),
        ),
        n_trials=args.n_trials,
        verbose=args.verbose,
    )

    results = runner.run()
    print_report(results, specs, contrasts, unrolls, args)


if __name__ == "__main__":
    main()
