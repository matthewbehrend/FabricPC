"""
ePC vs sPC on ResNet-18 / CIFAR-10
==================================

Benchmarks error-parameterized PC inference (``EPCInference``) against the
state-based baseline (``InferenceSGDNormClip``) on the resnet18 demo graph.

Two modes:

``--mode sweep`` (default) — one ``PlannedMultiContrastExperiment`` with an
arm per ePC step count T1 in ``--epc_step_sweep`` plus an sPC baseline arm
at ``--spc_steps``. All arms train the same ``--num_epochs``, so each arm is
one (total training wall-clock, final test accuracy) point and the T1 grid
sets the granularity of the time axis. Derived per-trial metrics:

- **accuracy at equal wall-clock** — linear interpolation of the trial's
  ePC (train_time, accuracy) points at the trial's sPC train_time. The grid
  must bracket sPC's time point; extend ``--epc_step_sweep`` past its top
  entry if the largest ePC arm still finishes faster than sPC.
- **wall-clock to equal accuracy** — the smallest-T1 arm whose accuracy
  reaches the trial's sPC accuracy, reporting its train_time and the ratio
  to sPC's.

Both derived metrics interpolate or select over the arm grid, so they are
reported descriptively (per-trial table plus mean +/- SE), with no test
attached — the runner's contrast family is declared empty. To confirm a
chosen operating point, declare it as a planned contrast on accuracy in a
follow-up run, e.g. ``contrasts=[("ePC-8", "sPC-120")]``.

``--mode convergence`` — single seed, no training: identical params and
initial state for all solvers on one test batch, tracked with
``run_inference_with_history``. Reports **per-node** energy-vs-step (total
energy is dominated by output-adjacent nodes — the energy imbalance in
Pinchetti et al., arXiv 2407.01163 — so a global curve can read as sPC
near-convergence while deep nodes have received no signal), the E*
head-to-head criterion (E* = sPC's final recorded total energy; each ePC
eta's updates to reach <= E*), and post-warmup wall-clock. Step counting:
the tracked history records each step's energy before that step's latent/ε
update (phase 2 runs before phase 3), so history index i is the energy
after i updates, and "updates to reach E*" counts updates, not tracked
steps. ``--epc_eta`` accepts a comma-separated list here, producing the
whole eta table in one invocation. Wall-clock is the min over repeated
post-warmup runs, reported two ways: asymptotic ms/step (a full
``--track_steps`` run divided by its step count) and ms per update at
T1 = 1 — ePC's ``run_inference`` brackets its step loop with a
``begin_segment`` resync forward and a ``finalize_state`` derive forward,
which the asymptotic number amortizes but which dominate a T1 = 1 arm.
Chart written to ``epc_convergence.html`` (and ``.png`` when kaleido is
installed).

Usage:
    python examples/epc_spc_resnet18_compare.py --mode convergence \
        --epc_eta 0.001,0.01,0.03,0.1
    python examples/epc_spc_resnet18_compare.py --mode sweep --n_trials 5

Convergence results (RTX 3090, cuda13; batch 256, 120 tracked steps,
sPC = InferenceSGDNormClip @ eta 0.1; E* = sPC's final recorded total
energy):

    epc_eta   ePC updates to reach <= E*
    0.001     104
    0.01      11
    0.03      4
    0.1       1

    asymptotic per-step wall-clock ratio (ePC / sPC): 0.88-0.95x

Sweep results (paste the per-trial tables here after running per house
convention):
    (pending — run --mode sweep --n_trials 5, ~5 h on the reference 3090)
"""

from jax_setup import set_jax_flags_before_importing_jax

set_jax_flags_before_importing_jax()

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.inference_epc import EPCInference
from fabricpc.experiments import ExperimentArm, PlannedMultiContrastExperiment
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training import evaluate_pcn, train_pcn
from fabricpc.utils.data.dataloader import Cifar10Loader
from fabricpc.utils.dashboarding.inference_tracking import run_inference_with_history

# Load the demo module directly (avoids triggering examples/__init__.py).
_demo_path = Path(__file__).parent / "resnet18_cifar10_demo.py"
_spec = importlib.util.spec_from_file_location("resnet18_cifar10_demo", _demo_path)
_demo = importlib.util.module_from_spec(_spec)
sys.modules["resnet18_cifar10_demo"] = _demo
_spec.loader.exec_module(_demo)


# =============================================================================
# Shared model / data plumbing
# =============================================================================


def parse_epc_etas(args):
    """--epc_eta as a list of floats, falling back to EPCInference's default."""
    if args.epc_eta is None:
        return [EPCInference().config["eta_infer"]]
    return [float(s) for s in args.epc_eta.split(",")]


def make_model_factory(inference, activation_name):
    activation = _demo.get_activation(activation_name)

    def factory(rng_key):
        return _demo._create_mupc_model(
            rng_key, inference=inference, activation=activation
        )

    return factory


def make_loader_factory(batch_size):
    def factory(seed):
        train_loader = Cifar10Loader(
            "train", batch_size=batch_size, shuffle=True, seed=seed
        )
        test_loader = Cifar10Loader("test", batch_size=batch_size, shuffle=False)
        return train_loader, test_loader

    return factory


def _write_chart(fig, stem):
    """html always; png only behind a kaleido import guard."""
    fig.write_html(f"{stem}.html")
    print(f"  Saved: {stem}.html")
    try:
        import kaleido  # noqa: F401

        fig.write_image(f"{stem}.png", scale=2)
        print(f"  Saved: {stem}.png")
    except ImportError:
        print("  (kaleido not installed; skipping png export)")


# =============================================================================
# Sweep mode
# =============================================================================


def run_sweep(args):
    epc_steps = [int(s) for s in args.epc_step_sweep.split(",")]
    epc_etas = parse_epc_etas(args)
    if len(epc_etas) != 1:
        raise ValueError(
            "--mode sweep takes a single --epc_eta; the comma-separated list "
            "is a convergence-mode input."
        )
    epc_eta = epc_etas[0]
    spc_name = f"sPC-{args.spc_steps}"

    steps_per_epoch = len(
        Cifar10Loader("train", batch_size=args.batch_size, shuffle=True, seed=0)
    )
    optimizer = _demo.make_optimizer(
        args.lr, args.weight_decay, args.num_epochs, steps_per_epoch
    )
    train_config = {"num_epochs": args.num_epochs}

    arms = [
        ExperimentArm(
            name=spc_name,
            model_factory=make_model_factory(
                InferenceSGDNormClip(
                    eta_infer=args.spc_eta, infer_steps=args.spc_steps, max_norm=1.0
                ),
                args.activation,
            ),
            train_fn=train_pcn,
            eval_fn=evaluate_pcn,
            optimizer=optimizer,
            train_config=train_config,
        )
    ]
    for t1 in epc_steps:
        arms.append(
            ExperimentArm(
                name=f"ePC-{t1}",
                model_factory=make_model_factory(
                    EPCInference(eta_infer=epc_eta, infer_steps=t1),
                    args.activation,
                ),
                train_fn=train_pcn,
                eval_fn=evaluate_pcn,
                optimizer=optimizer,
                train_config=train_config,
            )
        )

    print("=" * 70)
    print("ePC step sweep vs sPC baseline — ResNet-18 / CIFAR-10")
    print("=" * 70)
    print(
        f"T1 grid: {epc_steps}  |  sPC: {args.spc_steps} steps @ eta {args.spc_eta}\n"
        f"ePC eta: {epc_eta}  |  epochs/arm: {args.num_epochs}  |  "
        f"trials: {args.n_trials}"
    )

    # The runner supplies the paired trial loop. The contrast family is
    # declared empty on purpose: the comparisons below interpolate or select
    # over the arm grid, so they are reported descriptively (no p-values).
    # A confirmatory comparison of a chosen T1 belongs in a follow-up run
    # with contrasts=[("ePC-<T1>", spc_name)] on accuracy.
    runner = PlannedMultiContrastExperiment(
        arms=arms,
        contrasts=[],
        metric="accuracy",
        data_loader_factory=make_loader_factory(args.batch_size),
        n_trials=args.n_trials,
    )
    results = runner.run()

    _report_sweep(results, epc_steps, spc_name, args)


def _report_sweep(results, epc_steps, spc_name, args):
    n_trials = results.n_trials
    spc_acc = results.per_arm_metrics(spc_name)
    spc_time = results.per_arm_times(spc_name)
    epc_acc = {t1: results.per_arm_metrics(f"ePC-{t1}") for t1 in epc_steps}
    epc_time = {t1: results.per_arm_times(f"ePC-{t1}") for t1 in epc_steps}

    # -- derived per-trial metrics ------------------------------------------
    acc_at_equal_time = np.full(n_trials, np.nan)
    time_to_equal_acc = np.full(n_trials, np.nan)
    t1_to_equal_acc = np.full(n_trials, np.nan)
    for i in range(n_trials):
        points = sorted((epc_time[t1][i], epc_acc[t1][i]) for t1 in epc_steps)
        times = np.array([p[0] for p in points])
        accs = np.array([p[1] for p in points])
        if times[0] <= spc_time[i] <= times[-1]:
            acc_at_equal_time[i] = float(np.interp(spc_time[i], times, accs))
        else:
            print(
                f"  WARNING trial {i + 1}: sPC wall-clock {spc_time[i]:.0f}s is "
                f"outside the ePC range [{times[0]:.0f}, {times[-1]:.0f}]s — "
                f"extend --epc_step_sweep to bracket it."
            )
        reaching = [t1 for t1 in sorted(epc_steps) if epc_acc[t1][i] >= spc_acc[i]]
        if reaching:
            t1_to_equal_acc[i] = reaching[0]
            time_to_equal_acc[i] = epc_time[reaching[0]][i]
        else:
            print(
                f"  WARNING trial {i + 1}: no ePC arm reached the sPC accuracy "
                f"{spc_acc[i] * 100:.2f}%."
            )

    # -- per-trial tables ----------------------------------------------------
    print()
    print("--- Per-arm results (mean +/- SE over trials) ---")
    print(f"{'arm':<12} {'accuracy%':<18} {'train time (s)':<18}")
    for name in [spc_name] + [f"ePC-{t1}" for t1 in epc_steps]:
        acc = results.per_arm_metrics(name) * 100
        t = results.per_arm_times(name)
        se = acc.std(ddof=1) / np.sqrt(n_trials) if n_trials > 1 else 0.0
        acc_field = f"{acc.mean():.2f} +/- {se:.2f}"
        print(f"{name:<12} {acc_field:<18} {t.mean():.1f}")

    print()
    print("--- Accuracy at equal wall-clock (per trial) ---")
    print(f"{'trial':<7} {'sPC acc%':<10} {'ePC acc% @ sPC time':<22} {'diff%':<8}")
    for i in range(n_trials):
        print(
            f"{i + 1:<7} {spc_acc[i] * 100:<10.2f} "
            f"{acc_at_equal_time[i] * 100:<22.2f} "
            f"{(acc_at_equal_time[i] - spc_acc[i]) * 100:<+8.2f}"
        )

    print()
    print("--- Wall-clock to equal accuracy (per trial) ---")
    print(f"{'trial':<7} {'sPC time s':<12} {'ePC time s':<12} {'T1':<6} {'ratio':<8}")
    for i in range(n_trials):
        ratio = time_to_equal_acc[i] / spc_time[i]
        print(
            f"{i + 1:<7} {spc_time[i]:<12.1f} {time_to_equal_acc[i]:<12.1f} "
            f"{t1_to_equal_acc[i]:<6.0f} {ratio:<8.2f}"
        )

    # -- descriptive summaries ------------------------------------------------
    # These metrics interpolate (acc @ equal time) or select the smallest
    # qualifying arm (time to equal accuracy), so no test is attached; see
    # the module docstring for the planned-contrast route.
    def _descriptive(label, diffs):
        mask = ~np.isnan(diffs)
        n = int(mask.sum())
        if n == 0:
            print(f"{label}: no complete trials.")
            return
        d = diffs[mask]
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        print(f"{label}: mean {d.mean():+.4f} +/- {se:.4f} SE (n = {n})")

    print()
    print("--- Descriptive summaries (no test; see docstring) ---")
    _descriptive("accuracy @ equal wall-clock (ePC - sPC)", acc_at_equal_time - spc_acc)
    _descriptive(
        "wall-clock to equal accuracy (ePC - sPC, s)", time_to_equal_acc - spc_time
    )

    _plot_sweep(results, epc_steps, spc_name, n_trials)


def _plot_sweep(results, epc_steps, spc_name, n_trials):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    def _mean_se(values):
        mean = values.mean()
        se = values.std(ddof=1) / np.sqrt(n_trials) if n_trials > 1 else 0.0
        return mean, se

    t1s = sorted(epc_steps)
    acc_mean, acc_se, time_mean = [], [], []
    for t1 in t1s:
        m, s = _mean_se(results.per_arm_metrics(f"ePC-{t1}") * 100)
        acc_mean.append(m)
        acc_se.append(s)
        time_mean.append(results.per_arm_times(f"ePC-{t1}").mean())
    spc_acc_mean, spc_acc_se = _mean_se(results.per_arm_metrics(spc_name) * 100)
    spc_time_mean = results.per_arm_times(spc_name).mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Final test accuracy vs ePC steps per minibatch (T1)",
            "Total training wall-clock vs T1",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=t1s,
            y=acc_mean,
            error_y=dict(type="data", array=acc_se),
            mode="lines+markers",
            name="ePC",
        ),
        row=1,
        col=1,
    )
    # sPC baseline: horizontal line with SE band.
    fig.add_hline(
        y=spc_acc_mean,
        line_dash="dash",
        annotation_text=f"{spc_name} accuracy",
        row=1,
        col=1,
    )
    fig.add_hrect(
        y0=spc_acc_mean - spc_acc_se,
        y1=spc_acc_mean + spc_acc_se,
        fillcolor="gray",
        opacity=0.2,
        line_width=0,
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=t1s, y=time_mean, mode="lines+markers", name="ePC wall-clock"),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=spc_time_mean,
        line_dash="dash",
        annotation_text=f"{spc_name} wall-clock",
        row=2,
        col=1,
    )
    fig.update_xaxes(type="log", title_text="T1 (ePC inference steps)", row=2, col=1)
    fig.update_yaxes(title_text="test accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="training wall-clock (s)", row=2, col=1)
    fig.update_layout(height=700, width=900)

    _write_chart(fig, "epc_step_sweep")


# =============================================================================
# Convergence mode
# =============================================================================


def _time_min(solver, structure, params, init_state, clamps, repeats=5):
    """Post-warmup wall-clock of one run_inference call: min over repeats."""
    runner = jax.jit(
        lambda p, s, solver=solver, structure=structure: solver.run_inference(
            p, s, clamps, structure
        )
    )
    jax.block_until_ready(runner(params, init_state))  # compile
    best = np.inf
    for _ in range(repeats):
        t0 = time.time()
        jax.block_until_ready(runner(params, init_state))
        best = min(best, time.time() - t0)
    return best


def run_convergence(args):
    epc_etas = parse_epc_etas(args)
    activation = _demo.get_activation(args.activation)
    track_steps = args.track_steps

    solvers = {
        "sPC": InferenceSGDNormClip(
            eta_infer=args.spc_eta, infer_steps=track_steps, max_norm=1.0
        )
    }
    for eta in epc_etas:
        solvers[f"ePC@{eta:g}"] = EPCInference(eta_infer=eta, infer_steps=track_steps)
    epc_labels = [label for label in solvers if label != "sPC"]

    print("=" * 70)
    print("ePC vs sPC inference convergence — ResNet-18 / CIFAR-10, one batch")
    print("=" * 70)
    print(
        f"tracked steps: {track_steps}  |  sPC eta: {args.spc_eta}  |  "
        f"ePC etas: {epc_etas}"
    )

    # One structure per solver over identical params and initial state: the
    # graph differs only in config["inference"].
    master_key = jax.random.PRNGKey(42)
    graph_key, state_key = jax.random.split(master_key)
    params, base_structure = _demo._create_mupc_model(
        graph_key, inference=solvers["sPC"], activation=activation
    )

    test_loader = Cifar10Loader("test", batch_size=args.batch_size, shuffle=False)
    images, labels_onehot = next(iter(test_loader))  # labels arrive one-hot
    batch_size = images.shape[0]
    clamps = {
        base_structure.task_map["x"]: jnp.asarray(images),
        base_structure.task_map["y"]: jnp.asarray(labels_onehot),
    }

    init_state = initialize_graph_state(
        base_structure, batch_size, state_key, clamps=clamps, params=params
    )

    in_degree_nodes = [
        name
        for name in base_structure.nodes
        if base_structure.nodes[name].node_info.in_degree > 0
    ]

    histories = {}
    for label, solver in solvers.items():
        structure = base_structure._replace(
            config={**base_structure.config, "inference": solver}
        )
        tracked = jax.jit(
            lambda p, s, structure=structure: run_inference_with_history(
                p, s, clamps, structure
            )
        )
        _, metrics = tracked(params, init_state)  # warmup + result
        jax.block_until_ready(metrics)
        histories[label] = jax.tree_util.tree_map(np.asarray, metrics)

    # Post-warmup wall-clock, min over repeats, on the untracked path.
    # Two readings per solver: asymptotic ms/step (full track_steps run /
    # track_steps) and ms per update at T1 = 1. ePC's run_inference brackets
    # the step loop with a begin_segment resync forward and a finalize_state
    # derive forward; the asymptotic number amortizes them, the T1 = 1
    # number shows them — it is the real per-update cost of a small-T1 arm.
    # The per-step cost is eta-independent, so one ePC solver suffices.
    print()
    timed = {"sPC": solvers["sPC"], "ePC": solvers[epc_labels[0]]}
    per_step = {}
    for label, solver in timed.items():
        full = _time_min(solver, base_structure, params, init_state, clamps)
        one_step = type(solver)(**{**solver.config, "infer_steps": 1})
        single = _time_min(one_step, base_structure, params, init_state, clamps)
        per_step[label] = full / track_steps
        print(
            f"  {label}: {per_step[label] * 1000:.1f} ms/step asymptotic "
            f"({track_steps} steps, min over 5 runs); "
            f"{single * 1000:.1f} ms per update at T1=1"
        )
    ratio = per_step["ePC"] / per_step["sPC"]
    print(f"  asymptotic per-step cost ratio (ePC / sPC): {ratio:.2f}x")

    # Total energy series over in_degree > 0 nodes. History index i records
    # the energy computed before that step's update (phase 2 runs before
    # phase 3), i.e. the energy after i updates.
    totals = {
        label: np.sum(
            [history[name]["energy"] * batch_size for name in in_degree_nodes],
            axis=0,
        )
        for label, history in histories.items()
    }
    e_star = totals["sPC"][-1]
    print(
        f"  E* = sPC final recorded total energy ({track_steps}-step run) "
        f"= {e_star:.4g}"
    )
    for label in epc_labels:
        reached = np.nonzero(totals[label] <= e_star)[0]
        if reached.size:
            print(f"  {label}: reaches <= E* after {reached[0]} eps updates")
        else:
            print(
                f"  {label}: did not reach <= E* within {track_steps} steps "
                f"(final {totals[label][-1]:.4g})"
            )

    _plot_convergence(histories, base_structure, in_degree_nodes, track_steps)


def _plot_convergence(histories, structure, in_degree_nodes, track_steps):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.colors as pcolors

    # Color per node by schedule depth (first-occurrence position).
    depth = {name: i for i, name in enumerate(structure.node_order)}
    max_depth = max(depth.values())
    # History index i records the energy after i updates.
    updates = np.arange(track_steps)

    labels = list(histories.keys())
    fig = make_subplots(
        rows=1, cols=len(labels), shared_yaxes=True, subplot_titles=labels
    )
    for col, label in enumerate(labels, start=1):
        history = histories[label]
        for name in in_degree_nodes:
            energy = np.maximum(history[name]["energy"], 1e-12)
            color = pcolors.sample_colorscale("Viridis", depth[name] / max_depth)[0]
            fig.add_trace(
                go.Scatter(
                    x=updates,
                    y=np.log10(energy),
                    mode="lines",
                    line=dict(color=color, width=1),
                    name=name,
                    legendgroup=name,
                    showlegend=(col == 1),
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(title_text="updates applied", row=1, col=col)
    fig.update_yaxes(title_text="log10 per-node energy (batch mean)", row=1, col=1)
    fig.update_layout(
        height=550,
        width=550 * len(labels),
        title="Per-node energy vs updates applied (color = schedule depth)",
    )

    _write_chart(fig, "epc_convergence")


# =============================================================================
# CLI
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="ePC vs sPC benchmark on ResNet-18 / CIFAR-10"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="sweep",
        choices=["sweep", "convergence"],
        help="sweep: trained accuracy/wall-clock comparison; "
        "convergence: single-batch inference dynamics (default: sweep)",
    )
    parser.add_argument("--n_trials", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--spc_steps", type=int, default=120, help="sPC inference steps per minibatch"
    )
    parser.add_argument("--spc_eta", type=float, default=0.1)
    parser.add_argument(
        "--epc_eta",
        type=str,
        default=None,
        help="ePC inference rate (default: EPCInference's default, 1e-2; the "
        "epsilon step descends the full-transfer-function gradient, so tune "
        "it like a weight learning rate). Convergence mode accepts a "
        "comma-separated list and reports one E* row per eta; sweep mode "
        "takes a single value.",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--epc_step_sweep",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,16,32,64,128,160",
        help="Comma-separated T1 grid: dense 1-10 where accuracy moves "
        "fastest, log-spaced above; must bracket sPC's wall-clock point "
        "(at the ~0.9x per-step ratio, sPC-120 lands near ePC T1~130, "
        "hence the 160 top entry)",
    )
    parser.add_argument(
        "--track_steps",
        type=int,
        default=120,
        help="Steps tracked per solver in convergence mode",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "gelu", "leaky_relu"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "sweep":
        run_sweep(args)
    else:
        run_convergence(args)


if __name__ == "__main__":
    main()
