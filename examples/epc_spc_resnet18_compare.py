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

Both are tested with a paired t-test and Cohen's d across trials; at the
default n = 5 the test has 4 degrees of freedom, so read the per-trial
table alongside the p-value. Chart written to ``epc_step_sweep.html``
(and ``.png`` when kaleido is installed).

``--mode convergence`` — single seed, no training: identical params and
initial state for both solvers on one test batch, tracked with
``run_inference_with_history``. Reports **per-node** energy-vs-step (total
energy is dominated by output-adjacent nodes — the energy imbalance in
Pinchetti et al., arXiv 2407.01163 — so a global curve can read as sPC
near-convergence while deep nodes have received no signal), the E*
head-to-head criterion (E* = sPC's final total energy; ePC's steps to reach
<= E*), and post-warmup per-step wall-clock for both solvers. Chart written
to ``epc_convergence.html`` (and ``.png`` when kaleido is installed).

Usage:
    python examples/epc_spc_resnet18_compare.py --mode convergence
    python examples/epc_spc_resnet18_compare.py --mode sweep --n_trials 5

Convergence results (RTX 3090, cuda13; batch 256, 120 tracked steps,
sPC = InferenceSGDNormClip @ eta 0.1; E* = sPC's final total energy):

    epc_eta   ePC steps to reach <= E*
    0.001     105
    0.01      12
    0.03      5
    0.1       2

    per-step wall-clock ratio (ePC / sPC): 0.88-0.95x

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
import optax

from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.inference_epc import EPCInference
from fabricpc.experiments import ExperimentArm, PlannedMultiContrastExperiment
from fabricpc.experiments.statistics import cohens_d, paired_ttest
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


def resolve_epc_eta(args):
    """--epc_eta, falling back to EPCInference's constructor default."""
    if args.epc_eta is not None:
        return args.epc_eta
    return EPCInference().config["eta_infer"]


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


def make_optimizer(lr, weight_decay, num_epochs, steps_per_epoch):
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.01,
    )
    return optax.adamw(schedule, weight_decay=weight_decay)


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
    epc_eta = resolve_epc_eta(args)
    spc_name = f"sPC-{args.spc_steps}"

    steps_per_epoch = len(
        Cifar10Loader("train", batch_size=args.batch_size, shuffle=True, seed=0)
    )
    optimizer = make_optimizer(
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

    # The runner supplies the paired trial loop; the contrast family is
    # empty — the derived equal-wall-clock / equal-accuracy comparisons
    # below are computed from the per-trial (train_time, accuracy) pairs.
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
        print(f"{name:<12} {acc.mean():.2f} +/- {se:.2f}     {t.mean():.1f}")

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

    # -- paired tests ---------------------------------------------------------
    def _paired_report(label, a, b):
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() >= 2:
            tt = paired_ttest(a[mask], b[mask])
            eff = cohens_d(a[mask], b[mask])
            print(
                f"{label}: mean diff {tt.mean_difference:+.4f}, "
                f"t = {tt.t_statistic:.3f}, p = {tt.p_value:.4f} "
                f"(n = {tt.n}), Cohen's d = {eff.d:.3f}"
            )
        else:
            print(f"{label}: fewer than 2 complete trials; no test.")

    print()
    print(f"--- Paired tests across trials (n = {n_trials}, df = {n_trials - 1}) ---")
    _paired_report(
        "accuracy @ equal wall-clock (ePC - sPC)", acc_at_equal_time, spc_acc
    )
    _paired_report(
        "wall-clock to equal accuracy (ePC - sPC, s)", time_to_equal_acc, spc_time
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


def run_convergence(args):
    epc_eta = resolve_epc_eta(args)
    activation = _demo.get_activation(args.activation)
    track_steps = args.track_steps

    solvers = {
        "sPC": InferenceSGDNormClip(
            eta_infer=args.spc_eta, infer_steps=track_steps, max_norm=1.0
        ),
        "ePC": EPCInference(eta_infer=epc_eta, infer_steps=track_steps),
    }

    print("=" * 70)
    print("ePC vs sPC inference convergence — ResNet-18 / CIFAR-10, one batch")
    print("=" * 70)
    print(
        f"tracked steps: {track_steps}  |  sPC eta: {args.spc_eta}  |  "
        f"ePC eta: {epc_eta}"
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
    clamps = {
        base_structure.task_map["x"]: jnp.asarray(images),
        base_structure.task_map["y"]: jnp.asarray(labels_onehot),
    }

    init_state = initialize_graph_state(
        base_structure, images.shape[0], state_key, clamps=clamps, params=params
    )

    in_degree_nodes = [
        name
        for name in base_structure.nodes
        if base_structure.nodes[name].node_info.in_degree > 0
    ]

    histories = {}
    step_seconds = {}
    for label, solver in solvers.items():
        structure = base_structure._replace(
            config={**base_structure.config, "inference": solver}
        )

        tracked = jax.jit(
            lambda p, s, structure=structure: run_inference_with_history(
                p, s, clamps, structure
            )
        )
        final_state, metrics = tracked(params, init_state)  # warmup + result
        jax.block_until_ready(metrics)
        histories[label] = jax.tree_util.tree_map(np.asarray, metrics)

        # Post-warmup per-step wall-clock on the untracked path.
        runner = jax.jit(
            lambda p, s, solver=solver, structure=structure: solver.run_inference(
                p, s, clamps, structure
            )
        )
        jax.block_until_ready(runner(params, init_state))  # compile
        t0 = time.time()
        jax.block_until_ready(runner(params, init_state))
        step_seconds[label] = (time.time() - t0) / track_steps
        print(f"  {label}: {step_seconds[label] * 1000:.1f} ms/step (post-warmup)")

    ratio = step_seconds["ePC"] / step_seconds["sPC"]
    print(f"  per-step cost ratio (ePC / sPC): {ratio:.2f}x")

    # Total energy series over in_degree > 0 nodes.
    totals = {
        label: np.sum(
            [history[name]["energy"] * args.batch_size for name in in_degree_nodes],
            axis=0,
        )
        for label, history in histories.items()
    }
    e_star = totals["sPC"][-1]
    reached = np.nonzero(totals["ePC"] <= e_star)[0]
    if reached.size:
        print(
            f"  E* = sPC total energy after {track_steps} steps = {e_star:.4f}; "
            f"ePC reaches <= E* at step {reached[0] + 1}"
        )
    else:
        print(
            f"  E* = {e_star:.4f}; ePC did not reach <= E* within "
            f"{track_steps} steps (final {totals['ePC'][-1]:.4f})"
        )

    _plot_convergence(histories, base_structure, in_degree_nodes, track_steps)


def _plot_convergence(histories, structure, in_degree_nodes, track_steps):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    import plotly.colors as pcolors

    # Color per node by schedule depth (first-occurrence position).
    depth = {name: i for i, name in enumerate(structure.node_order)}
    max_depth = max(depth.values())
    steps = np.arange(1, track_steps + 1)

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, subplot_titles=("sPC", "ePC")
    )
    for col, label in ((1, "sPC"), (2, "ePC")):
        history = histories[label]
        for name in in_degree_nodes:
            energy = np.maximum(history[name]["energy"], 1e-12)
            color = pcolors.sample_colorscale("Viridis", depth[name] / max_depth)[0]
            fig.add_trace(
                go.Scatter(
                    x=steps,
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
    fig.update_xaxes(title_text="inference step", row=1, col=1)
    fig.update_xaxes(title_text="inference step", row=1, col=2)
    fig.update_yaxes(title_text="log10 per-node energy (batch mean)", row=1, col=1)
    fig.update_layout(
        height=550,
        width=1100,
        title="Per-node energy vs inference step (color = schedule depth)",
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
        type=float,
        default=None,
        help="ePC inference rate (default: EPCInference's default, 1e-2; the "
        "epsilon step descends the full-transfer-function gradient, so tune "
        "it like a weight learning rate)",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--epc_step_sweep",
        type=str,
        default="1,2,3,4,5,6,7,8,9,10,16,32,64,128",
        help="Comma-separated T1 grid: dense 1-10 where accuracy moves "
        "fastest, log-spaced above; must bracket sPC's wall-clock point",
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
