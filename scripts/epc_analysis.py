"""
ePC analysis: the backprop regime, equilibrium energy profiles, convergence
spectra, and stability, answering the reviewer's bullets on the resnet18
ePC-vs-sPC convergence figure (``examples/epc_spc_resnet18_compare.py``).

Everything on linear graphs is exact, from ``fabricpc.utils.linear_pc_oracle``;
the same diagnostics run on nonlinear graphs through
``EPCInference.error_energy`` (Hessian-vector products, the Lanczos estimator
``fabricpc.core.epsilon_spectrum``).

Sections (``--section``; the default set runs on CPU in about a minute):

  backprop_regime      bullet 5. 1-step ePC weight gradients approach
                       eta * backprop (hidden) and backprop (output) at first
                       order in eta, both sides means per prediction through
                       the trainer's normalization; Adam removes the eta
                       scaling; the 2-epoch resnet18 sweep is fitted by one
                       effective error-Hessian eigenvalue lambda_eff through
                       the relaxed fraction 1 - (1 - eta*lambda)^T (a
                       heuristic); the formula is validated against the
                       solver on a linear chain.
  equilibrium_profile  bullets 2, 3. Per-layer equilibrium energies of linear
                       chains: the spread across layers and its slope follow
                       the downstream gain (eps_l* = eps_y* P_l^T); the sPC
                       transient is top-heavy early and approaches the oracle
                       late; ePC moves every layer at once. The equilibrium
                       output error r S^-1 damps the learning signal along
                       mode lambda_S by 1/lambda_S (Innocenti et al. 2024,
                       Theorem 1), tabulated per depth and init.
  convergence_spectra  bullets 1, 7. lambda_max / lambda_min of H_z and of the
                       excited H_eps versus depth, and the steps each solver
                       needs to contract by 1e-3 at eta = 1/lambda_max, with
                       two depths measured.
  stability            bullets 4, 6. lambda_max(H_eps) and the bound
                       2/lambda_max versus weight scale and depth (the
                       mechanism behind horizon-dependent collapse); the
                       Lanczos spectrum agrees with the oracle's excited
                       extremes and gradient-weighted relaxed fraction; on a
                       gelu MLP ePC descends at 0.9 eta_max and grows at
                       1.1 eta_max.

GPU, opt-in (CIFAR-10 via tfds, the demo's muPC resnet18):

  --resnet18           the excited spectrum at init on one CIFAR batch
                       (lambda_max, lambda_min, the gradient weight on
                       negative curvature, the Ritz residuals) -> eta_max and
                       the regime of the defaults, compared with the sweep's
                       fitted lambda_eff; the predicted gradient-weighted
                       relaxed fraction and regime letter per recorded sweep
                       cell beside its measured accuracy.

Tracking the spectrum during training is the demo's job:
``examples/resnet18_cifar10_demo.py --track_regime N`` runs
``fabricpc.training.RegimeProbe`` and writes ``epc_regime_track__*.csv``;
``--plot_track CSV...`` here renders those files (four panels: the extremes
against 2/eta, the relaxed fraction and the negative weight, the weight norms,
the test accuracy).

Sweep numbers are the recorded 2-epoch tables in
``docs/dev_plans_archive/epc_inference_solver.md``; the 100-epoch outcomes are
the six ``sweep_eta*_steps*.log`` files in the project root. Both predate
release 0.5.1's per-prediction gradient normalization.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc import setup_jax
from fabricpc.core import EPCInference, EpsilonSpectrum, InferenceSGD, epsilon_spectrum
from fabricpc.core.activations import (
    GeluActivation,
    IdentityActivation,
    SoftmaxActivation,
    TanhActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy, graph_energy
from fabricpc.core.epsilon_spectrum import weighted_relaxed_fraction
from fabricpc.core.initializers import (
    MuPCInitializer,
    NormalInitializer,
    XavierInitializer,
)
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode
from fabricpc.training import grad_denominator, pc_weight_gradients, read_regime_csv
from fabricpc.training.regime_probe import WNORM_PREFIX
from fabricpc.utils import linear_pc_oracle as oracle
from fabricpc.utils.dashboarding.inference_tracking import run_inference_with_history

# =============================================================================
# Recorded resnet18 data
# =============================================================================

SWEEP_T = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64, 128, 160]
# 2-epoch test accuracy (%), mean over 5 trials, muPC resnet18 / CIFAR-10.
SWEEP_ACC = {
    0.1: [10.22, 10.57, 10.05, 12.12, 14.28, 15.82, 18.57, 21.51, 25.92, 28.63,
          30.72, 31.09, 31.05, 31.03, 31.02],
    0.03: [36.90, 33.87, 32.62, 31.92, 31.45, 31.21, 31.04, 30.92, 30.78, 30.73,
           30.73, 30.98, 31.17, 31.17, 31.19],
    0.01: [38.54, 37.97, 36.91, 35.84, 34.99, 34.35, 33.86, 33.44, 33.02, 32.71,
           31.81, 30.93, 30.89, 31.12, 31.15],
    0.001: [38.84, 38.83, 38.78, 38.74, 38.70, 38.65, 38.65, 38.60, 38.55, 38.49,
            38.18, 36.68, 34.25, 32.29, 31.83],
    0.0001: [38.77, 38.83, 38.84, 38.83, 38.83, 38.84, 38.85, 38.85, 38.82, 38.81,
             38.83, 38.78, 38.65, 38.34, 38.16],
}  # fmt: skip
SWEEP_SPC_120 = 34.64
# The small-eta*T limit of ePC itself; no backprop arm was run at 2 epochs.
SWEEP_SMALL_ETA_T_LIMIT = 38.8
SWEEP_PC_PLATEAU = 31.0
SWEEP_COLLAPSED_BELOW = 20.0

# 100-epoch runs (project-root logs): (eta, T) -> final test accuracy (%).
HUNDRED_EPOCH = {
    (1e-3, 1): 76.73,
    (1e-3, 2): 75.76,
    (1e-3, 5): 9.75,  # 54.76% at epoch 10, 9.68% at epoch 20
    (1e-2, 1): 9.92,  # at chance by epoch 10
    (1e-2, 2): 9.88,
    (1e-2, 5): 10.13,
}

# Categorical palette, documented order (dataviz reference palette, light mode).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SEQUENTIAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]


# =============================================================================
# Printing
# =============================================================================


def header(title, bullet=None):
    print()
    print("=" * 78)
    print(title if bullet is None else f"{title}   [reviewer bullet {bullet}]")
    print("=" * 78)


def table(headers, rows):
    widths = [len(h) for h in headers]
    cells = [[str(c) for c in row] for row in rows]
    for row in cells:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    fmt = "  ".join("{:>" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print("  ".join("-" * w for w in widths))
    for row in cells:
        print(fmt.format(*row))


def g(x):
    return f"{x:.3g}"


# =============================================================================
# Graph builders
# =============================================================================


def build_chain(
    depth,
    width,
    d_in,
    d_out,
    *,
    weight_std=None,
    mupc=False,
    activation=None,
    output_activation=None,
    output_energy=None,
    inference=None,
):
    """x(d_in) -> depth x Linear(width) -> y(d_out).

    Plain: NormalInitializer(std=weight_std / sqrt(fan_in)) on every edge.
    muPC: MuPCInitializer on the hidden edges with MuPCConfig scaling and the
    demo's Xavier readout (``include_output=False``), the demo's
    parameterization.
    """
    activation = activation or IdentityActivation()
    output_activation = output_activation or IdentityActivation()
    x = IdentityNode(shape=(d_in,), name="x")
    nodes = [x]
    fan_in = d_in
    for i in range(depth):
        w_init = (
            MuPCInitializer()
            if mupc
            else NormalInitializer(std=weight_std / math.sqrt(fan_in))
        )
        nodes.append(
            Linear(
                shape=(width,),
                name=f"h{i + 1}",
                activation=activation,
                weight_init=w_init,
            )
        )
        fan_in = width
    out_kwargs = {"energy": output_energy} if output_energy is not None else {}
    nodes.append(
        Linear(
            shape=(d_out,),
            name="y",
            activation=output_activation,
            weight_init=(
                XavierInitializer()
                if mupc
                else NormalInitializer(std=weight_std / math.sqrt(fan_in))
            ),
            **out_kwargs,
        )
    )
    edges = [Edge(source=a, target=b.slot("in")) for a, b in zip(nodes[:-1], nodes[1:])]
    return graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=x, y=nodes[-1]),
        inference=inference or EPCInference(),
        scaling=MuPCConfig(include_output=False) if mupc else None,
    )


def with_solver(structure, inference):
    return structure._replace(config={**structure.config, "inference": inference})


def random_clamps(structure, key, batch, *, one_hot_target=True):
    d_in = structure.nodes["x"].node_info.shape[0]
    d_out = structure.nodes["y"].node_info.shape[0]
    kx, ky = jax.random.split(key)
    x = jax.random.normal(kx, (batch, d_in))
    if one_hot_target:
        y = jax.nn.one_hot(jax.random.randint(ky, (batch,), 0, d_out), d_out)
    else:
        y = jax.random.normal(ky, (batch, d_out))
    return {"x": x, "y": y}


def hidden_names(structure):
    return [n for n in structure.node_order if n.startswith("h")]


def in_degree_nodes(structure):
    return [
        n for n in structure.node_order if structure.nodes[n].node_info.in_degree > 0
    ]


def total_energy_series(history, structure):
    return sum(np.asarray(history[n]["energy"]) for n in in_degree_nodes(structure))


# =============================================================================
# Sweep fit (shared by backprop_regime and --resnet18)
# =============================================================================


def normalized_accuracy(acc):
    """0 at the small-eta*T limit, 1 at the PC plateau."""
    return (SWEEP_SMALL_ETA_T_LIMIT - acc) / (
        SWEEP_SMALL_ETA_T_LIMIT - SWEEP_PC_PLATEAU
    )


def relaxed(eta, steps, lam):
    return 1.0 - abs(1.0 - eta * lam) ** steps


def fit_sweep_lambda():
    """One effective eigenvalue lambda_eff fitted by least squares to the
    normalized accuracy of the eta <= 0.01 cells (45 cells; every fit candidate
    keeps eta*lambda < 1 there, so the cell set does not change with lambda).
    A heuristic: it maps accuracy linearly onto the relaxed fraction and
    reads one eigenvalue off it. Returns (lambda_eff, rms_residual, per-cell
    rows)."""
    cells = [
        (eta, T, acc)
        for eta, accs in SWEEP_ACC.items()
        if eta <= 0.01
        for T, acc in zip(SWEEP_T, accs)
    ]
    grid = np.logspace(0.0, 2.0, 801)  # lambda in [1, 100]
    best = None
    for lam in grid:
        pred = np.array([relaxed(eta, T, lam) for eta, T, _ in cells])
        meas = np.array([normalized_accuracy(acc) for _, _, acc in cells])
        rms = float(np.sqrt(np.mean((pred - meas) ** 2)))
        if best is None or rms < best[1]:
            best = (float(lam), rms)
    lam_eff, rms = best
    rows = []
    for eta, accs in SWEEP_ACC.items():
        for T, acc in zip(SWEEP_T, accs):
            rows.append(
                (eta, T, acc, normalized_accuracy(acc), relaxed(eta, T, lam_eff))
            )
    return lam_eff, rms, rows


def single_mode_spectrum(lam):
    """The spectrum of one eigenvalue carrying the whole gradient, for
    reading a regime off a fitted lambda_eff."""
    return EpsilonSpectrum.from_modes([lam], [1.0])


def regime_letter(eta, steps, spectrum):
    """B backprop-like, P partially relaxed, E near PC equilibrium (bands on
    the gradient-weighted relaxed fraction), U unstable (eta*lambda_max > 2);
    an appended r marks an output-gradient reversal on the top mode."""
    regime = EPCInference(eta_infer=eta, infer_steps=steps).regime(spectrum)
    if regime.unstable:
        return "U"
    letter = {"backprop-like": "B", "near PC equilibrium": "E"}.get(regime.band, "P")
    return letter + ("r" if regime.output_gradient_reverses else "")


def print_sweep_regime_table(spectrum, title):
    print(title)
    print(
        "  cell: measured accuracy % | predicted gradient-weighted relaxed fraction "
        "f_bar | fastest mode f_max | regime letter on f_bar (B backprop-like, P "
        "partially relaxed, E near equilibrium, U unstable; r: the output-layer "
        "gradient reverses on the top mode)"
    )
    headers = ["eta"] + [f"T={T}" for T in SWEEP_T]
    rows = []
    for eta, accs in SWEEP_ACC.items():
        row = [g(eta)]
        for T, acc in zip(SWEEP_T, accs):
            regime = EPCInference(eta_infer=eta, infer_steps=T).regime(spectrum)
            row.append(
                f"{acc:.1f}|{regime.f_weighted:.2f}|{regime.f_max:.2f}"
                f"{regime_letter(eta, T, spectrum)}"
            )
        rows.append(row)
    table(headers, rows)
    print(
        f"  sPC-120 (eta 0.1): {SWEEP_SPC_120}%  |  small-eta*T limit "
        f"{SWEEP_SMALL_ETA_T_LIMIT}% (ePC's own; no backprop arm was run)  |  "
        f"PC plateau {SWEEP_PC_PLATEAU}%"
    )


# =============================================================================
# Section 1 — backprop regime
# =============================================================================


def section_backprop_regime(args):
    header("backprop_regime: 1-step ePC against backprop", bullet=5)
    key = jax.random.PRNGKey(0)
    batch = 8
    structure = build_chain(
        3,
        32,
        16,
        10,
        weight_std=1.0,
        activation=TanhActivation(),
        output_activation=SoftmaxActivation(),
        output_energy=CrossEntropyEnergy(),
    )
    params = initialize_params(structure, key)
    clamps = random_clamps(structure, jax.random.fold_in(key, 1), batch)
    denom = grad_denominator(structure, clamps)

    def loss(p):
        state = initialize_graph_state(structure, batch, key, clamps, params=p)
        return graph_energy(state, structure, node_names=["y"]) / denom

    g_bp = jax.grad(loss)(params)
    layers = hidden_names(structure) + ["y"]

    def one_step_grads(eta):
        s = with_solver(structure, EPCInference(eta_infer=eta, infer_steps=1))
        state = initialize_graph_state(s, batch, key, clamps, params=params)
        final = s.config["inference"].run_inference(params, state, clamps, s)
        return pc_weight_gradients(params, final, s, clamps)

    print(
        "Relative deviation of the 1-step local weight gradient from eta*backprop\n"
        "(hidden layers) and from backprop (output), per edge weight. Both sides\n"
        "are means per prediction: pc_weight_gradients against jax.grad of the\n"
        f"output energy divided by grad_denominator (N = {denom}), the trainer's\n"
        "normalization. h1's input is the clamp, so its identity is exact (float32\n"
        "noise only); downstream layers see their input latent re-derived at the\n"
        "perturbed upstream state, an O(eta) remainder."
    )
    rows = []
    g_pc_by_eta = {}
    for eta in (1e-4, 1e-3, 1e-2, 1e-1):
        g_pc = one_step_grads(eta)
        g_pc_by_eta[eta] = g_pc
        row = [g(eta)]
        for name in layers:
            scale = 1.0 if name == "y" else eta
            ((edge_key, ref),) = g_bp.nodes[name].weights.items()
            got = g_pc.nodes[name].weights[edge_key] / scale
            row.append(
                f"{float(jnp.linalg.norm(got - ref) / jnp.linalg.norm(ref)):.2e}"
            )
        rows.append(row)
    table(["eta"] + layers, rows)

    adam = optax.adam(1e-3)
    opt_state = adam.init(params)
    upd_bp, _ = adam.update(g_bp, opt_state, params)
    print("\nCosine similarity of one Adam update from eta*backprop-scaled ePC grads")
    print("vs from backprop grads (the eta scaling cancels under Adam):")
    rows = []
    for eta in (1e-3, 1e-2):
        upd_pc, _ = adam.update(g_pc_by_eta[eta], opt_state, params)
        row = [g(eta)]
        for name in layers:
            ((edge_key, a),) = upd_bp.nodes[name].weights.items()
            b = upd_pc.nodes[name].weights[edge_key]
            cos = float(jnp.sum(a * b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))
            row.append(f"{cos:.4f}")
        rows.append(row)
    table(["eta"] + layers, rows)

    lam_eff, rms, _ = fit_sweep_lambda()
    print(
        f"\nSweep fit (heuristic): one effective error-Hessian eigenvalue for the muPC\n"
        f"resnet18, lambda_eff = {lam_eff:.1f} (rms residual {rms:.3f} in normalized\n"
        f"accuracy, fitted on the 45 cells with eta <= 0.01; accuracy mapped linearly\n"
        f"onto the relaxed fraction). Regime per cell with the whole gradient on\n"
        f"lambda_eff:"
    )
    print_sweep_regime_table(single_mode_spectrum(lam_eff), "")
    print(
        "\n100-epoch outcomes (project-root logs, batch-summed gradients before\n"
        "release 0.5.1), (eta, T) -> final accuracy %:\n  "
        + "  ".join(f"({g(e)}, {T}) {acc}" for (e, T), acc in HUNDRED_EPOCH.items())
    )
    print(
        f"  At lambda_eff the defaults (1e-3, 5) read\n"
        f"  '{EPCInference().regime(single_mode_spectrum(lam_eff))}'\n"
        f"  at init; they collapsed at epoch 20."
    )

    print(
        "\nRelaxed-fraction formula against the solver on a linear chain\n"
        "(x16 -> 3 x h16 -> y4, eta relative to lambda_max(H_eps)): remaining\n"
        "distance ||eps_T - eps*|| / ||eps*|| predicted from the eigen-decomposition\n"
        "versus measured after T ePC steps."
    )
    chain = build_chain(3, 16, 16, 4, weight_std=1.0)
    c_params = initialize_params(chain, jax.random.fold_in(key, 2))
    c_clamps = random_clamps(
        chain, jax.random.fold_in(key, 3), batch, one_hot_target=False
    )
    eq = oracle.linear_equilibrium(c_params, chain, c_clamps)
    H = oracle.epsilon_hessian(eq.quad)
    eigs, vecs = np.linalg.eigh(H)
    eps_star = oracle.flatten_free(eq.quad, eq.error_star)  # (D, batch); eps_0 = 0
    coeff = vecs.T @ eps_star
    lam_max = eigs[-1]
    rows = []
    for eta_rel, T in ((0.1, 1), (0.1, 5), (0.5, 3), (1.0, 5), (1.5, 4)):
        eta = eta_rel / lam_max
        remaining = vecs @ ((1.0 - eta * eigs)[:, None] ** T * coeff)
        predicted = np.linalg.norm(remaining) / np.linalg.norm(eps_star)
        s = with_solver(chain, EPCInference(eta_infer=eta, infer_steps=T))
        state = initialize_graph_state(s, batch, key, c_clamps, params=c_params)
        final = s.config["inference"].run_inference(c_params, state, c_clamps, s)
        eps_T = oracle.flatten_free(
            eq.quad, {n: final.nodes[n].error for n in eq.quad.free}
        )
        measured = np.linalg.norm(eps_T - eps_star) / np.linalg.norm(eps_star)
        rows.append([g(eta_rel), T, f"{predicted:.4f}", f"{measured:.4f}"])
    table(["eta*lambda_max", "T", "predicted remaining", "measured remaining"], rows)

    if args.plot:
        plot_sweep_fit(lam_eff)


# =============================================================================
# Section 2 — equilibrium profile
# =============================================================================


def per_layer_log_energy(eq, structure):
    return {n: float(np.log10(max(np.mean(eq.node_energy[n]), 1e-300)))
            for n in in_degree_nodes(structure)}  # fmt: skip


def section_equilibrium_profile(args):
    header("equilibrium_profile: per-layer equilibrium energies", bullet="2, 3")
    key = jax.random.PRNGKey(1)
    batch = 8
    width, d_in, d_out = 32, 32, 10
    inits = [("std 0.5", 0.5, False), ("std 1.0", 1.0, False), ("std 1.5", 1.5, False),
             ("muPC", None, True)]  # fmt: skip
    print(
        "Oracle per-layer log10 E_l* (batch mean). spread = max - min over hidden\n"
        "layers in decades; slope = least-squares slope of log10 E_l versus layer\n"
        "index over the hidden layers (~ 2 log10 of the per-layer downstream gain,\n"
        "from eps_l* = eps_y* P_l^T)."
    )
    rows = []
    profiles = {}
    for depth in (3, 5, 10, 20):
        for label, std, mupc in inits:
            structure = build_chain(
                depth, width, d_in, d_out, weight_std=std, mupc=mupc
            )
            params = initialize_params(structure, jax.random.fold_in(key, depth))
            clamps = random_clamps(
                structure, jax.random.fold_in(key, 100 + depth), batch
            )
            eq = oracle.linear_equilibrium(params, structure, clamps)
            logs = per_layer_log_energy(eq, structure)
            hidden = hidden_names(structure)
            hv = np.array([logs[n] for n in hidden])
            slope = (
                float(np.polyfit(np.arange(len(hv)), hv, 1)[0]) if len(hv) > 1 else 0.0
            )
            profiles[(depth, label)] = (hidden, hv, logs["y"])
            rows.append([
                depth, label, f"{hv[0]:.2f}", f"{hv[len(hv) // 2]:.2f}", f"{hv[-1]:.2f}",
                f"{logs['y']:.2f}", f"{hv.max() - hv.min():.2f}", f"{slope:+.3f}",
            ])  # fmt: skip
    table(
        ["depth", "init", "log10 E h1", "log10 E mid", "log10 E last", "log10 E y",
         "spread", "slope/layer"],
        rows,
    )  # fmt: skip

    print(
        "\nEquilibrium output error (Innocenti et al. 2024, Theorem 1): eps_y* = r S^-1\n"
        "with S = I + sum_l P_l^T P_l, so the learning signal along an eigenmode of S\n"
        "with eigenvalue lambda_S is damped by 1/lambda_S, a matrix rescaling no\n"
        "per-parameter optimizer undoes. ||r S^-1|| / ||r|| is the batch mean of the\n"
        "per-sample ratio. This linear-chain mechanism, applied to the nonlinear\n"
        "cross-entropy resnet18, is consistent with the 2-epoch sweep's PC-equilibrium\n"
        "plateau (31%) trailing its backprop-like plateau (38.8%)."
    )
    rows = []
    for depth in (3, 5, 10, 20):
        for label, std, mupc in inits:
            structure = build_chain(
                depth, width, d_in, d_out, weight_std=std, mupc=mupc
            )
            params = initialize_params(structure, jax.random.fold_in(key, depth))
            clamps = random_clamps(
                structure, jax.random.fold_in(key, 100 + depth), batch
            )
            _, S, r = oracle.theorem1_energy(params, structure, clamps)
            damped = r @ np.linalg.inv(S)
            ratio = float(
                np.mean(np.linalg.norm(damped, axis=1) / np.linalg.norm(r, axis=1))
            )
            eig_s = np.linalg.eigvalsh(S)
            rows.append([depth, label, f"{ratio:.3f}", g(eig_s[0]), g(eig_s[-1])])
    table(["depth", "init", "||r S^-1||/||r||", "lambda_min(S)", "lambda_max(S)"], rows)

    depth, std = 10, 1.0
    structure = build_chain(depth, width, d_in, d_out, weight_std=std)
    params = initialize_params(structure, jax.random.fold_in(key, depth))
    clamps = random_clamps(structure, jax.random.fold_in(key, 100 + depth), batch)
    eq = oracle.linear_equilibrium(params, structure, clamps)
    logs_star = per_layer_log_energy(eq, structure)
    nodes = in_degree_nodes(structure)

    eta_spc = min(0.1, 0.9 * oracle.stability_bound(oracle.latent_hessian(eq.quad)))
    checkpoints = [10, 50, 200, 1000, 4999]
    s = with_solver(structure, InferenceSGD(eta_infer=eta_spc, infer_steps=5000))
    state = initialize_graph_state(s, batch, key, clamps, params=params)
    _, hist = run_inference_with_history(params, state, clamps, s)
    print(
        f"\nsPC transient, depth {depth}, std {std}, eta {eta_spc:.3g}: per-layer log10\n"
        "energy after k updates (history index k), last row the oracle equilibrium.\n"
        "Early on, energy sits in the output-adjacent layers; deep layers fill in\n"
        "only after thousands of steps."
    )
    rows = []
    for k in checkpoints:
        rows.append(
            [f"after {k}"]
            + [
                f"{np.log10(max(float(hist[n]['energy'][k]), 1e-300)):.2f}"
                for n in nodes
            ]
        )
    rows.append(["oracle E*"] + [f"{logs_star[n]:.2f}" for n in nodes])
    table(["sPC"] + nodes, rows)

    eta_epc = 1.0 / float(np.linalg.eigvalsh(oracle.epsilon_hessian(eq.quad))[-1])
    s = with_solver(structure, EPCInference(eta_infer=eta_epc, infer_steps=20))
    state = initialize_graph_state(s, batch, key, clamps, params=params)
    _, hist = run_inference_with_history(params, state, clamps, s)
    print(
        f"\nePC at eta = 1/lambda_max(H_eps) = {eta_epc:.3g}: every layer moves from the\n"
        "first update."
    )
    rows = []
    for k in (1, 5, 19):
        rows.append(
            [f"after {k}"]
            + [
                f"{np.log10(max(float(hist[n]['energy'][k]), 1e-300)):.2f}"
                for n in nodes
            ]
        )
    rows.append(["oracle E*"] + [f"{logs_star[n]:.2f}" for n in nodes])
    table(["ePC"] + nodes, rows)

    if args.plot:
        plot_equilibrium_profile(profiles)


# =============================================================================
# Section 3 — convergence spectra
# =============================================================================


def section_convergence_spectra(args):
    header("convergence_spectra: Hessian spectra and steps to contract", bullet="1, 7")
    key = jax.random.PRNGKey(2)
    batch = 8
    width, d_in, d_out = 16, 16, 4
    print(
        "H_z = A^T A governs the state-based solver; H_eps = M^T H_z M governs ePC,\n"
        "of which only the excited modes (overlap with the initial gradient) matter\n"
        "from eps = 0. steps = smallest T with max |1 - eta*lambda|^T <= 1e-3 at\n"
        "eta = 1/lambda_max. kappa = lambda_max / lambda_min."
    )
    rows = []
    spectra = {}
    for depth in (2, 3, 4, 6, 8, 12, 16, 20):
        for label, mupc in (("plain", False), ("muPC", True)):
            structure = build_chain(
                depth, width, d_in, d_out, weight_std=1.0, mupc=mupc
            )
            params = initialize_params(structure, jax.random.fold_in(key, depth))
            clamps = random_clamps(
                structure, jax.random.fold_in(key, 100 + depth), batch
            )
            eq = oracle.linear_equilibrium(params, structure, clamps)
            Hz = oracle.latent_hessian(eq.quad)
            He = oracle.epsilon_hessian(eq.quad)
            ez = np.linalg.eigvalsh(Hz)
            ee = oracle.excited_eigenvalues(
                He, oracle.epsilon_gradient_at_zero(eq.quad)
            )
            steps_z = oracle.steps_to_contract(1.0 / ez[-1], ez, 1e-3)
            steps_e = oracle.steps_to_contract(1.0 / ee[-1], ee, 1e-3)
            spectra[(depth, label)] = (steps_z, steps_e)
            rows.append([
                depth, label, g(ez[-1]), g(ez[0]), g(ez[-1] / ez[0]), steps_z,
                g(ee[-1]), g(ee[0]), g(ee[-1] / ee[0]), steps_e, len(ee),
            ])  # fmt: skip
    table(
        ["depth", "init", "lmax(H_z)", "lmin(H_z)", "kappa_z", "steps sPC",
         "lmax(H_eps)", "lmin exc", "kappa_eps", "steps ePC", "#excited"],
        rows,
    )  # fmt: skip

    print(
        "\nMeasured (plain init): relative latent error ||z_T - z*|| / ||z_ff - z*||\n"
        "after the predicted T and after T/4, each solver at eta = 1/lambda_max."
    )
    rows = []
    for depth in (4, 12):
        structure = build_chain(depth, width, d_in, d_out, weight_std=1.0)
        params = initialize_params(structure, jax.random.fold_in(key, depth))
        clamps = random_clamps(structure, jax.random.fold_in(key, 100 + depth), batch)
        eq = oracle.linear_equilibrium(params, structure, clamps)
        z_star = oracle.flatten_free(eq.quad, eq.z_star)
        z_ff = oracle.flatten_free(eq.quad, eq.quad.z_ff)
        dist0 = np.linalg.norm(z_ff - z_star)
        for label, make, H, eigs in (
            ("sPC", InferenceSGD, oracle.latent_hessian(eq.quad), None),
            ("ePC", EPCInference, oracle.epsilon_hessian(eq.quad), None),
        ):
            if label == "sPC":
                eigs = np.linalg.eigvalsh(H)
            else:
                eigs = oracle.excited_eigenvalues(
                    H, oracle.epsilon_gradient_at_zero(eq.quad)
                )
            eta = 1.0 / float(np.linalg.eigvalsh(H)[-1])
            T = oracle.steps_to_contract(eta, eigs, 1e-3)
            ratios = []
            for steps in (T, max(1, T // 4)):
                s = with_solver(structure, make(eta_infer=eta, infer_steps=steps))
                state = initialize_graph_state(s, batch, key, clamps, params=params)
                final = s.config["inference"].run_inference(params, state, clamps, s)
                z_T = oracle.flatten_free(
                    eq.quad, {n: final.nodes[n].z_latent for n in eq.quad.free}
                )
                ratios.append(np.linalg.norm(z_T - z_star) / dist0)
            rows.append(
                [
                    depth,
                    label,
                    T,
                    f"{ratios[0]:.2e}",
                    max(1, T // 4),
                    f"{ratios[1]:.2e}",
                ]
            )
    table(["depth", "solver", "T predicted", "ratio at T", "T/4", "ratio at T/4"], rows)

    if args.plot:
        plot_spectra(spectra)


# =============================================================================
# Section 4 — stability
# =============================================================================


def section_stability(args):
    header("stability: the bound 2/lambda_max(H_eps)", bullet="4, 6")
    key = jax.random.PRNGKey(3)
    batch = 8
    width, d_in, d_out = 32, 32, 10
    print(
        "lambda_max(H_eps) = 1 + sigma_max(J)^2 grows with the product of the\n"
        "downstream gains, so a fixed eta crosses the bound 2/lambda_max as the\n"
        "weights grow during training: the mechanism behind a collapse that appears\n"
        "only after many epochs. At odd T the output layer is hit earlier: its\n"
        "weight gradient follows the output residual after T steps, which along the\n"
        "top mode is (1 - eta*(lambda_max - 1))*r at T = 1 and reverses sign once\n"
        "eta*(lambda_max - 1) > 1."
    )
    rows = []
    for depth in (3, 5, 10):
        for label, std, mupc in (("std 0.5", 0.5, False), ("std 1.0", 1.0, False),
                                 ("std 1.5", 1.5, False), ("std 2.0", 2.0, False),
                                 ("muPC", None, True)):  # fmt: skip
            structure = build_chain(
                depth, width, d_in, d_out, weight_std=std, mupc=mupc
            )
            params = initialize_params(structure, jax.random.fold_in(key, depth))
            clamps = random_clamps(
                structure, jax.random.fold_in(key, 100 + depth), batch
            )
            eq = oracle.linear_equilibrium(params, structure, clamps)
            He = oracle.epsilon_hessian(eq.quad)
            lam = float(np.linalg.eigvalsh(He)[-1])
            rows.append([depth, label, g(lam), g(2.0 / lam)])
    table(["depth", "init", "lambda_max(H_eps)", "eta_max = 2/lambda_max"], rows)

    structure = build_chain(5, width, d_in, d_out, weight_std=1.0)
    params = initialize_params(structure, jax.random.fold_in(key, 5))
    clamps = random_clamps(structure, jax.random.fold_in(key, 105), batch)
    eq = oracle.linear_equilibrium(params, structure, clamps)
    He = oracle.epsilon_hessian(eq.quad)
    g0 = oracle.epsilon_gradient_at_zero(eq.quad)
    eigs, weights = oracle.gradient_weights(He, g0)
    excited = oracle.excited_eigenvalues(He, g0)
    state = initialize_graph_state(structure, batch, key, clamps, params=params)
    t0 = time.time()
    spectrum = epsilon_spectrum(params, state, clamps, structure, iters=30, key=key)
    elapsed = time.time() - t0
    eta_probe, steps_probe = 0.5 / eigs[-1], 5
    fbar_lanczos = weighted_relaxed_fraction(spectrum, eta_probe, steps_probe)
    fbar_oracle = oracle.weighted_relaxed_fraction(
        eigs, weights, eta_probe, steps_probe
    )
    guard_note = (
        f"the breakdown guard froze the recurrence after k = {spectrum.k} steps"
        if spectrum.k < spectrum.iters
        else "float32 rounding kept beta above the breakdown guard, so the eps weight\n"
        "floor selected the extremes over the modes carrying gradient weight"
    )
    print(
        f"\nLanczos through EPCInference.error_energy on the depth-5 chain (30 steps,\n"
        f"{elapsed:.1f}s incl. compile; the Krylov space has dimension d_out = {d_out};\n"
        f"{guard_note}):"
    )
    table(
        ["quantity", "Lanczos", "oracle", "relative error"],
        [
            ["lambda_max", f"{spectrum.lambda_max:.6g}", f"{eigs[-1]:.6g}",
             f"{abs(spectrum.lambda_max - eigs[-1]) / eigs[-1]:.2e}"],
            ["lambda_min (excited)", f"{spectrum.lambda_min:.6g}", f"{excited.min():.6g}",
             f"{abs(spectrum.lambda_min - excited.min()) / excited.min():.2e}"],
            [f"f_bar at eta = 0.5/lambda_max, T = {steps_probe}", f"{fbar_lanczos:.6f}",
             f"{fbar_oracle:.6f}", f"{abs(fbar_lanczos - fbar_oracle):.2e}"],
        ],
    )  # fmt: skip
    print(
        f"  full spectrum: {len(eigs)} modes, {len(excited)} excited; the unit-precision\n"
        f"  floor (lambda = {eigs[0]:.4g}) is never excited from eps = 0 and Lanczos does\n"
        f"  not report it."
    )

    batch = 16
    mlp = build_chain(
        4, 64, 32, 10, weight_std=1.0, activation=GeluActivation(),
        output_activation=SoftmaxActivation(), output_energy=CrossEntropyEnergy(),
    )  # fmt: skip
    params = initialize_params(mlp, jax.random.fold_in(key, 9))
    clamps = random_clamps(mlp, jax.random.fold_in(key, 10), batch)
    state = initialize_graph_state(mlp, batch, key, clamps, params=params)
    spectrum = epsilon_spectrum(params, state, clamps, mlp, iters=30, key=key)
    lam = spectrum.lambda_max
    eta_max = 2.0 / lam
    print(
        f"\ngelu MLP x32 -> 4 x h64 -> y10 (softmax + CE), batch {batch}: Lanczos gives\n"
        f"lambda_max = {lam:.4g}, lambda_min = {spectrum.lambda_min:.4g} (gradient weight\n"
        f"on negative curvature {spectrum.negative_weight:.3f}) at init, eta_max =\n"
        f"2/lambda_max = {eta_max:.4g}. ePC for 200 steps:"
    )
    rows = []
    for factor in (0.9, 1.1):
        s = with_solver(mlp, EPCInference(eta_infer=factor * eta_max, infer_steps=200))
        st = initialize_graph_state(s, batch, key, clamps, params=params)
        _, hist = run_inference_with_history(params, st, clamps, s)
        total = total_energy_series(hist, s)
        rise = total[-1] / total.min()
        verdict = (
            "settles at its minimum"
            if rise < 1.001
            else f"rises after its minimum (final/min = {rise:.3g})"
        )
        rows.append([
            f"{factor} eta_max", f"{total[0]:.4g}", f"{total.min():.4g}", f"{total[-1]:.4g}",
            "finite" if np.all(np.isfinite(total)) else "non-finite", verdict,
        ])  # fmt: skip
    table(["eta", "E after 0", "min E", "E after 199", "finite", "trend"], rows)
    print(
        "  (nonlinear energy: the linear bound is local; the outcome is reported as observed)"
    )


# =============================================================================
# GPU sections
# =============================================================================


def load_demo():
    demo_path = (
        Path(__file__).resolve().parent.parent / "examples" / "resnet18_cifar10_demo.py"
    )
    spec = importlib.util.spec_from_file_location("resnet18_cifar10_demo", demo_path)
    demo = importlib.util.module_from_spec(spec)
    sys.modules["resnet18_cifar10_demo"] = demo
    spec.loader.exec_module(demo)
    return demo


def cifar_probe_batch(structure, batch_size):
    from fabricpc.utils.data.dataloader import Cifar10Loader

    # Slice the split to exactly one batch and read it fully (a half-read tfds
    # iterator warns on teardown).
    loader = Cifar10Loader(f"test[:{batch_size}]", batch_size=batch_size, shuffle=False)
    [(images, labels)] = list(loader)  # labels arrive one-hot
    return {
        structure.task_map["x"]: jnp.asarray(images),
        structure.task_map["y"]: jnp.asarray(labels),
    }


def section_resnet18(args):
    header("--resnet18: the excited spectrum at init on the muPC resnet18 (GPU)")
    demo = load_demo()
    # The demo's key split for trial seed --seed (its default trial seed is
    # 42), so the graph is the one the demo trains.
    graph_key, _train_key, state_key = jax.random.split(
        jax.random.PRNGKey(args.seed), 3
    )
    params, structure = demo._create_mupc_model(
        graph_key,
        inference=EPCInference(),
        activation=demo.get_activation(args.activation),
    )
    clamps = cifar_probe_batch(structure, args.probe_batch)
    state = initialize_graph_state(
        structure, args.probe_batch, state_key, clamps=clamps, params=params
    )
    t0 = time.time()
    spectrum = epsilon_spectrum(
        params, state, clamps, structure, iters=args.lanczos_iters, key=state_key
    )
    elapsed = time.time() - t0
    lam = spectrum.lambda_max
    lam_eff, rms, _ = fit_sweep_lambda()
    defaults = EPCInference()
    regime = defaults.regime(spectrum)
    print(
        f"batch {args.probe_batch}, {args.lanczos_iters} Lanczos steps ({elapsed:.1f}s incl. compile;\n"
        f"k = {spectrum.k} valid steps, ||g0|| = {spectrum.gradient_norm:.4g})\n"
        f"  lambda_max(H_eps) at init = {lam:.4g}   (Ritz residual {spectrum.residual_max:.2e})   "
        f"eta_max = 2/lambda_max = {2.0 / lam:.4g}\n"
        f"  lambda_min (excited)      = {spectrum.lambda_min:.4g}   (Ritz residual {spectrum.residual_min:.2e})\n"
        f"  gradient weight on negative curvature = {spectrum.negative_weight:.4f}\n"
        f"  f_bar of the defaults     = {regime.f_weighted:.4f}   (fastest mode {regime.f_max:.4f})\n"
        f"  sweep-fitted lambda_eff   = {lam_eff:.4g}   (rms residual {rms:.3f}; heuristic)\n"
        f"  ratio measured / fitted   = {lam / lam_eff:.3g}"
    )
    if spectrum.lambda_min < 0:
        floor_note = (
            f"negative: H_eps is indefinite at init (the second-derivative term of the\n"
            f"  gelu + cross-entropy map); the negative modes carry "
            f"{100 * spectrum.negative_weight:.1f}% of the gradient and grow by\n"
            f"  {regime.growth_min:.4g}x over the defaults' {defaults.config['infer_steps']} steps"
        )
    elif spectrum.lambda_min <= 1.05:
        floor_note = (
            "at or below the unit-precision floor, as predicted for a nonlinear graph\n"
            "  (the second-derivative term couples g0 to nearly every mode)"
        )
    else:
        floor_note = (
            "above the unit-precision floor: the excited band is compact on this graph"
        )
    print(f"  lambda_min at init is {floor_note}.")
    print(
        "  The fit reads one eigenvalue off accuracy; the measurement is the top of\n"
        "  the spectrum at init. A ratio near 1 is consistent with the accuracy\n"
        "  following the top modes. f_bar against f_max says where the gradient\n"
        "  weight sits: f_bar well below f_max means most of it is on modes far\n"
        "  below lambda_max. Either outcome is reported as observed."
    )
    print()
    print_sweep_regime_table(
        spectrum,
        "Regime per recorded sweep cell from the measured spectrum (f_bar and f_max\n"
        "per cell from the init spectrum beside the measured 2-epoch accuracy):",
    )
    print(f"\n  defaults at init: {regime}")


# =============================================================================
# Plots (--plot): plotly html always, png behind the kaleido guard
# =============================================================================


def write_chart(fig, stem):
    fig.write_html(f"{stem}.html")
    print(f"  Saved: {stem}.html")
    try:
        import kaleido  # noqa: F401

        fig.write_image(f"{stem}.png", scale=2)
        print(f"  Saved: {stem}.png")
    except ImportError:
        print("  (kaleido not installed; skipping png export)")


def _base_layout(fig, title, xaxis, yaxis):
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
        font=dict(color="#0b0b0b"),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=60, r=30, t=60, b=80),
        hovermode="closest",
    )
    fig.update_xaxes(title_text=xaxis, gridcolor="#e6e5e1", zeroline=False)
    fig.update_yaxes(title_text=yaxis, gridcolor="#e6e5e1", zeroline=False)


def plot_sweep_fit(lam_eff):
    import plotly.graph_objects as go

    fig = go.Figure()
    for slot, (eta, accs) in enumerate(SWEEP_ACC.items()):
        xs = [relaxed(eta, T, lam_eff) for T in SWEEP_T]
        ys = [normalized_accuracy(a) for a in accs]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers", name=f"eta {eta:g}",
                marker=dict(color=PALETTE[slot], size=9, line=dict(color="#fcfcfb", width=2)),
                text=[f"eta {eta:g}, T {T}: {a:.1f}%" for T, a in zip(SWEEP_T, accs)],
                hovertemplate="%{text}<br>predicted %{x:.2f}, measured %{y:.2f}<extra></extra>",
            )
        )  # fmt: skip
    fig.add_trace(
        go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="measured = predicted",
                   line=dict(color="#52514e", width=2))
    )  # fmt: skip
    _base_layout(
        fig,
        f"2-epoch sweep: normalized accuracy vs predicted relaxed fraction (lambda_eff = {lam_eff:.0f})",
        "predicted relaxed fraction 1 - (1 - eta*lambda_eff)^T",
        "normalized accuracy: 0 = small-eta*T limit 38.8%, 1 = PC plateau 31%",
    )
    write_chart(fig, "epc_analysis_sweep_fit")


def plot_equilibrium_profile(profiles):
    import plotly.graph_objects as go

    fig = go.Figure()
    depths = sorted({d for d, label in profiles if label == "std 1.0"})
    for i, depth in enumerate(depths):
        hidden, hv, y_log = profiles[(depth, "std 1.0")]
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(hv) + 1)) + [len(hv) + 1], y=list(hv) + [y_log],
                mode="lines+markers", name=f"depth {depth}",
                line=dict(color=SEQUENTIAL_BLUE[i % len(SEQUENTIAL_BLUE)], width=2),
                marker=dict(size=8, line=dict(color="#fcfcfb", width=2)),
                hovertemplate="layer %{x}: log10 E* = %{y:.2f}<extra>depth " + str(depth) + "</extra>",
            )
        )  # fmt: skip
    _base_layout(
        fig,
        "Equilibrium energy per layer, linear chains at std 1.0 (last point: output)",
        "layer index (hidden 1..L, then output)",
        "log10 equilibrium energy (batch mean)",
    )
    write_chart(fig, "epc_analysis_equilibrium_profile")


def plot_spectra(spectra):
    import plotly.graph_objects as go

    fig = go.Figure()
    depths = sorted({d for d, _ in spectra})
    series = [("sPC plain", 0, "plain", 0, "solid"), ("ePC plain", 1, "plain", 1, "solid"),
              ("sPC muPC", 0, "muPC", 0, "dot"), ("ePC muPC", 1, "muPC", 1, "dot")]  # fmt: skip
    for name, idx, init, slot, dash in series:
        ys = [spectra[(d, init)][idx] for d in depths]
        fig.add_trace(
            go.Scatter(x=depths, y=ys, mode="lines+markers", name=name,
                       line=dict(color=PALETTE[slot], width=2, dash=dash),
                       marker=dict(size=8, line=dict(color="#fcfcfb", width=2)),
                       hovertemplate="depth %{x}: %{y} steps<extra>" + name + "</extra>")
        )  # fmt: skip
    fig.update_yaxes(type="log")
    _base_layout(
        fig,
        "Steps to contract the latent error by 1e-3 at eta = 1/lambda_max",
        "hidden depth",
        "steps (log)",
    )
    write_chart(fig, "epc_analysis_spectra")


def plot_track(csv_path):
    """Four stacked panels from a ``RegimeProbe`` CSV (the demo's
    ``--track_regime N``): the excited extremes lambda_max and |lambda_min| on
    the probe batch (log scale) against the stability bound 2/eta, drawn only
    when the file records an ePC eta_infer; the gradient-weighted relaxed
    fraction f_bar with the gradient weight on negative curvature; the
    Frobenius norm of every weight; the test accuracy per epoch. Vertical
    lines mark the first output-gradient reversal and the first crossing of
    eta*lambda_max = 2. eta, T, and the trainer come from the metadata
    columns, not from the file name."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    csv_path = Path(csv_path)
    metadata, rows = read_regime_csv(csv_path)
    probes = [r for r in rows if r["lambda_max"] is not None]
    evals = [r for r in rows if r["test_accuracy"] is not None]
    eta, steps = metadata["eta_infer"], metadata["infer_steps"]
    trainer = metadata["trainer"]
    updates = [r["update"] for r in probes]
    wnorm_columns = sorted(
        c for c in (probes[0] if probes else {}) if c.startswith(WNORM_PREFIX)
    )

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.3, 0.2, 0.3, 0.2],
        subplot_titles=(
            "excited extremes of the error Hessian on the probe batch (log scale)",
            "gradient-weighted relaxed fraction f_bar and weight on negative curvature",
            "Frobenius norm per weight",
            "test accuracy after each epoch",
        ),
    )
    marker = dict(size=5, line=dict(color="#fcfcfb", width=1.5))
    fig.add_trace(
        go.Scatter(
            x=updates, y=[r["lambda_max"] for r in probes], mode="lines+markers",
            name="lambda_max", line=dict(color=PALETTE[0], width=2), marker=marker,
            hovertemplate="update %{x}: lambda_max %{y:.4g}<extra></extra>",
        ),
        row=1, col=1,
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=updates, y=[abs(r["lambda_min"]) for r in probes], mode="lines+markers",
            name="|lambda_min|",
            line=dict(color=PALETTE[2], width=2, dash="dot"), marker=marker,
            text=["negative" if r["lambda_min"] < 0 else "positive" for r in probes],
            hovertemplate="update %{x}: |lambda_min| %{y:.4g} (%{text})<extra></extra>",
        ),
        row=1, col=1,
    )  # fmt: skip
    if eta is not None:
        fig.add_hline(
            y=2.0 / eta,
            line=dict(color="#52514e", width=2),
            annotation_text=f"2/eta = {2.0 / eta:g}: eta*lambda_max = 2",
            annotation_position="top left",
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=updates, y=[r["f_weighted"] for r in probes], mode="lines+markers",
                name="f_bar", line=dict(color=PALETTE[0], width=2), marker=marker,
                hovertemplate="update %{x}: f_bar %{y:.3f}<extra></extra>",
            ),
            row=2, col=1,
        )  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=updates, y=[r["negative_weight"] for r in probes], mode="lines+markers",
            name="negative-curvature weight",
            line=dict(color=PALETTE[3], width=2, dash="dot"), marker=marker,
            hovertemplate="update %{x}: negative weight %{y:.3f}<extra></extra>",
        ),
        row=2, col=1,
    )  # fmt: skip
    for i, column in enumerate(wnorm_columns):
        label = column[len(WNORM_PREFIX) :]
        fig.add_trace(
            go.Scatter(
                x=updates, y=[r[column] for r in probes], mode="lines", name=label,
                line=dict(color=SEQUENTIAL_BLUE[i % len(SEQUENTIAL_BLUE)], width=1.2),
                showlegend=False,
                hovertemplate="update %{x}: " + label + " %{y:.4g}<extra></extra>",
            ),
            row=3, col=1,
        )  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=[r["update"] for r in evals], y=[100.0 * r["test_accuracy"] for r in evals],
            mode="lines+markers", name="test accuracy",
            line=dict(color=PALETTE[1], width=2),
            marker=dict(size=8, line=dict(color="#fcfcfb", width=2)),
            hovertemplate="update %{x}: %{y:.2f}%<extra></extra>",
        ),
        row=4, col=1,
    )  # fmt: skip
    events = []
    if eta is not None:
        reversal = next((r for r in probes if r["output_gradient_reverses"]), None)
        crossing = next((r for r in probes if r["unstable"]), None)
        if reversal is not None:
            events.append((reversal["update"], "output gradient reverses", PALETTE[4]))
        if crossing is not None:
            events.append((crossing["update"], "eta*lambda_max = 2", "#52514e"))
    for update, text, color in events:
        fig.add_vline(
            x=update,
            line=dict(color=color, width=1.5, dash="dash"),
            annotation_text=f"{text} (update {update})",
            annotation_position="top right",
        )
    fig.update_yaxes(type="log", title_text="eigenvalue", row=1, col=1)
    fig.update_yaxes(title_text="fraction", range=[0, 1.05], row=2, col=1)
    fig.update_yaxes(type="log", title_text="||W||_F", row=3, col=1)
    fig.update_yaxes(title_text="accuracy (%)", row=4, col=1)
    fig.update_xaxes(title_text="weight updates", row=4, col=1)
    solver = (
        f"ePC eta_infer={eta:g}, infer_steps={steps}" if eta is not None else "backprop"
    )
    fig.update_layout(
        title=f"{solver} (trainer {trainer}): spectrum, relaxation, and weight norms during training",
        template="plotly_white",
        paper_bgcolor="#fcfcfb",
        plot_bgcolor="#fcfcfb",
        font=dict(color="#0b0b0b"),
        legend=dict(orientation="h", y=-0.08),
        margin=dict(l=60, r=30, t=70, b=70),
        height=1200,
    )
    fig.update_xaxes(gridcolor="#e6e5e1", zeroline=False)
    fig.update_yaxes(gridcolor="#e6e5e1", zeroline=False)
    write_chart(fig, str(csv_path.with_suffix("")))


# =============================================================================
# CLI
# =============================================================================


SECTIONS = {
    "backprop_regime": section_backprop_regime,
    "equilibrium_profile": section_equilibrium_profile,
    "convergence_spectra": section_convergence_spectra,
    "stability": section_stability,
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--section", nargs="+", choices=list(SECTIONS), default=list(SECTIONS),
                   help="CPU sections to run (default: all four)")  # fmt: skip
    p.add_argument(
        "--plot",
        action="store_true",
        help="write plotly charts (html; png with kaleido)",
    )
    p.add_argument(
        "--resnet18",
        action="store_true",
        help="GPU: the excited spectrum at init on the demo's muPC resnet18",
    )
    p.add_argument(
        "--activation", default="gelu", choices=["relu", "tanh", "gelu", "leaky_relu"]
    )
    p.add_argument(
        "--probe_batch",
        type=int,
        default=64,
        help="CIFAR batch for the --resnet18 spectrum (default: 64)",
    )
    p.add_argument(
        "--plot_track",
        nargs="+",
        default=None,
        metavar="CSV",
        help="render four-panel charts from RegimeProbe CSVs (the demo's "
        "--track_regime output) and exit",
    )
    p.add_argument(
        "--lanczos_iters",
        type=int,
        default=30,
        help="Lanczos steps for the --resnet18 spectrum (default: 30)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="trial seed, the demo's key split (default: 42, the demo's first trial)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.plot_track:
        for path in args.plot_track:
            plot_track(path)
        return
    if args.resnet18:
        setup_jax()
    else:
        setup_jax(platform="cpu")
    t0 = time.time()
    if args.resnet18:
        section_resnet18(args)
    else:
        for name in args.section:
            SECTIONS[name](args)
    print(f"\ntotal {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
