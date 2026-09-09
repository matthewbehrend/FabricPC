"""
Exact predictive-coding equilibria on linear-Gaussian DAGs, and the spectral
diagnostics that set how fast each solver reaches them.

Reference. Francesco Innocenti, El Mehdi Achour, Ryan Singh, and Christopher
L. Buckley. *Only Strict Saddles in the Energy Landscape of Predictive Coding
Networks?* Advances in Neural Information Processing Systems 37 (NeurIPS
2024), pp. 53649–53683. arXiv:2408.11979. Theorem 1: the equilibrated energy
of a deep linear network is a rescaled mean-squared error with
S = I + Σ_l P_lᵀP_l.

Part 1 — exact equilibrium (pure NumPy, float64)
================================================

Purpose. Inference relaxes the free latents toward the minimum of the graph
energy. On a DAG whose nodes are ``Linear`` or ``IdentityNode`` with
``IdentityActivation`` and ``GaussianEnergy`` the energy is a quadratic, so
its minimum has a closed form. This module computes that minimum from the
parameters and the graph description alone. It calls no node, solver, or
JAX code, so it is an independent reference for ``EPCInference`` and the
state-based solvers (``InferenceSGD`` and variants).

What "a model matches the oracle" means. ``tests/test_linear_pc_oracle.py``
(``TestEPCReachesOracle`` and ``TestSPCReachesOracle``, both through
``_assert_matches_oracle``) builds a graph, initializes the state with
``initialize_graph_state`` (feedforward latents, ε = 0, every source's
``z_mu`` equal to its latent), passes those source ``z_mu`` values in as
``source_means``, runs the solver at a rate and step count taken from the
Part 2 Hessians, and asserts per sample at rtol = atol = 1e-4:

    final.nodes[t].z_latent   ≈ z_star[t]         every node t
    final.nodes[t].error      ≈ error_star[t]     every in_degree > 0 node
    final.nodes[t].energy     ≈ node_energy[t]    every in_degree > 0 node
    Σ_t final.nodes[t].energy ≈ total_energy      over in_degree > 0 nodes

Under ``EPCInference`` the ``error`` of each unclamped source is checked
too. ``InferenceSGD`` re-syncs a source's ``z_mu`` to its latent every
step, so its source error is 0 by construction and is skipped.

How the oracle is produced. ``linear_equilibrium`` calls
``assemble_linear_quadratic`` to write the energy as E = ½‖A z_free − c‖²
(steps 1–4), solves for z*_free with ``numpy.linalg.lstsq`` (step 5), and
recomputes predictions, errors, and energies from z* with the same weights
and biases (step 6). The steps are labelled at the matching lines of both
functions.

Symbols. Every stacked matrix holds samples as columns. A per-node array is
(batch, d) as the library stores it; its transpose is a stacked column
block.

    t, s          node names; every edge s → t runs forward in ``node_order``
    d_t           width of node t (rank-1 shapes only)
    rows          the ``in_degree > 0`` nodes in ``node_order``: the nodes
                  that own a Gaussian energy term. R = Σ_{t∈rows} d_t.
    free          the unclamped nodes in ``node_order``: the variables.
                  D = Σ_{t∈free} d_t.
    z_t, ẑ_t      (d_t,) latent of node t, and its clamp value when clamped
    W_eff[s→t]    (d_s, d_t) effective weight of edge s → t: the edge's muPC
                  ``forward_scale`` (1 when absent) times the ``Linear``
                  weight, or times the ``IdentityNode`` ``scale`` times I
                  (``effective_edge_matrices``)
    b_t           (d_t,) bias of node t, zeros when absent
    p_t           ``GaussianEnergy`` precision of node t (default 1)
    μ_t           prediction of t, μ_t = b_t + Σ_{s→t} W_eff[s→t]ᵀ z_s; the
                  library's row form is ``z_mu = Σ_s z_s @ W_eff + b``
    ε_t           prediction error z_t − μ_t
    z_free        (D, batch) stacked latents of the free nodes
    A, c          (R, D) and (R, batch): E = ½‖A z_free − c‖² per column
    B, M          (D, D): the free-to-free part of μ, and M = (I − B)⁻¹
    k             (D, batch) constant part of the free nodes' μ
    z_ff          the latents at ε = 0, the feedforward point

(1) Energy. Each row node t contributes

        E_t = ½ p_t ‖z_t − μ_t‖² = ½ ‖√p_t (z_t − μ_t)‖²,    E = Σ_{t∈rows} E_t,

    the ``GaussianEnergy`` of every ``in_degree > 0`` node. Sources own no
    term. This is the sum ``EPCInference.error_energy`` and the training
    loop take.

(2) Residual blocks (``assemble_linear_quadratic``, one pass over rows).
    z_t − μ_t is affine in z_free. Its free part fills row block
    ``row_offsets[t]`` of A; its constant part fills the same rows of c:

        A[row_t, col_t] = √p_t · I                       t free
        A[row_t, col_s] = −√p_t · W_eff[s→t]ᵀ            each edge s→t, s free
        c[row_t]        = √p_t · (b_t − ẑ_t·[t clamped]
                                  + Σ_{s→t, s clamped} W_eff[s→t]ᵀ ẑ_s)

    so that (A z_free − c)[row_t] = √p_t (z_t − μ_t) = √p_t ε_t, and
    E = ½‖A z_free − c‖² column by column. A clamped row node has a row
    block and no column block. A free source has a column block and no row
    block: it enters A only through the −√p_t W_eff[s→t]ᵀ blocks in its
    targets' rows.

(3) Error coordinates (same pass). For a free node t, z_t = ε_t + μ_t.
    Splitting μ_t into the part carried by free latents and the rest,

        B[col_t, col_s] = W_eff[s→t]ᵀ           each edge s→t, s and t free
        k[col_t]        = b_t + Σ_{s→t, s clamped} W_eff[s→t]ᵀ ẑ_s
        z_free = ε + B z_free + k,   so   z_free = M (ε + k),   M = (I − B)⁻¹.

    B is strictly block-lower-triangular because every edge runs forward
    in ``node_order``, so I − B is unit lower triangular, M exists, and
    det M = 1. A free source s has no in-edges: its B row block is zero,
    its k block is ``source_means[s]`` (its constant ``z_mu``), and
    ε_s = z_s − source_means[s].

(4) Feedforward point. z_ff is the latent at ε = 0, z_ff,free = M k,
    computed as a forward pass along ``node_order``: clamps, then
    ``source_means``, then μ_t for each row node. This is the state
    ``initialize_graph_state`` produces when ``source_means`` is read from
    it, and the state both solvers start from.

(5) Solve (``linear_equilibrium``). ∇_{z_free} E = Aᵀ(A z_free − c) = 0, so

        z*_free = argmin ‖A z_free − c‖ = numpy.linalg.lstsq(A, c),

    unique iff rank A = D. Rank drops when a free source's outgoing maps
    are not jointly injective; ``linear_equilibrium`` raises then.
    ``min_singular_value`` = σ_min(A) is the margin from that failure.
    Clamped nodes keep z*_t = ẑ_t.

(6) Readouts (``linear_equilibrium``). From z*, using W_eff and b again
    rather than A:

        z_mu_star[t]   = b_t + Σ_{s→t} W_eff[s→t]ᵀ z*_s       t ∈ rows
        z_mu_star[s]   = ẑ_s if clamped, else source_means[s]   s a source
        error_star[t]  = z*_t − z_mu_star[t]                    every node
        node_energy[t] = ½ p_t ‖error_star[t]‖²                 t ∈ rows
        total_energy   = Σ_{t∈rows} node_energy[t]

    Two identities tie the readouts to the quadratic of step 2 and are
    asserted by ``test_readouts_agree_with_quadratic``:
    error_star[t] = (A z*_free − c)[row_t] / √p_t for every row node, and
    total_energy = ½‖A z*_free − c‖² per column. ``source_means`` shifts
    error_star of a free source and z_ff, never z* or E*, because a source
    has no row in A or c.

Part 2 — spectral diagnostics
=============================

The Hessian in latent coordinates is H_z = AᵀA. In error coordinates
(step 3, z_free = M(ε + k)) it is H_ε = Mᵀ H_z M. The two Hessians govern
the two solvers:

- λ_min(H_z) decays with depth even for benign weights: the state-based
  solver's slow mode, which needs ~κ(H_z) steps to relax.
- λ_max(H_ε) = 1 + σ_max(J)² on a chain with unit precision, J the map from
  the stacked errors to the output prediction, grows with the product of
  downstream weights: ePC's stability bound 2/λ_max(H_ε) shrinks as the
  weights grow. From ε = 0 with uniform precision and no unclamped source,
  ePC's trajectory lives in the d_y-dimensional row space of J and sees
  only eig(S), S = I + Σ_l P_lᵀP_l (Innocenti et al. 2024, Theorem 1).

Regime. After T gradient steps at rate η from ε = 0, each excited eigenmode
λ of H_ε has relaxed toward equilibrium by f(λ) = 1 − (1 − ηλ)^T.
Backprop-like behavior (ε ≈ −η·T·∇E, the paper's Theorem C.9) requires
η·T·λ ≪ 1 on the modes that carry the gradient; the PC equilibrium requires
f(λ) > 0.9 on those modes, which on a chain are the eig(S) modes, so the
slowest of them sets the step count. Stability requires η·λ_max < 2 at
every T. At odd T the output-layer weight gradient reverses sign along the
top mode once (1 − ηλ_max)^T < −1/(λ_max − 1) (T = 1: η(λ_max − 1) > 1),
before the iteration bound: the output residual after T steps is
r_T = (r/λ)·[1 + (λ − 1)(1 − ηλ)^T] per mode, while the hidden errors
(1 − (1 − ηλ)^T)·ε* keep their sign for every ηλ < 2.

Scope. The oracle is defined on DAGs: ``validate_linear_gaussian`` rejects
cycles, whose unrolled ε-energy carries warm-started latents and is not a
pure function of ε. The quadratic is per sample (samples are columns of c),
so the solver's batch Hessian is block-diagonal over samples with this H as
every block: a batch's λ_max is the per-sample maximum, and a larger batch's
bound is at least as tight. The same diagnostics on any graph the solver
accepts, nonlinear included, are ``fabricpc.core.epsilon_spectrum``
(Lanczos on Hessian-vector products through ``EPCInference.error_energy``).
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, NamedTuple, Optional, Tuple

import numpy as np

from fabricpc.core.activations import IdentityActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.nodes.base import NodeBase
from fabricpc.nodes.identity import IdentityNode
from fabricpc.nodes.linear import Linear

Array = np.ndarray
PerNode = Dict[str, Array]


# =============================================================================
# Part 1 — exact equilibrium
# =============================================================================


class LinearQuadratic(NamedTuple):
    """E = ½‖A z_free − c‖² and the ε → z map, from ``assemble_linear_quadratic``
    (module docstring, Part 1, steps 2–4). Samples are columns.

    Index sets and offsets:
        free: unclamped node names in ``node_order``; z_free stacks their
            latents.
        rows: ``in_degree > 0`` node names in ``node_order``; each owns one
            residual block of A and c.
        col_offsets: name -> slice of that node's d_t columns in z_free
            (D total).
        row_offsets: name -> slice of that node's d_t rows in A z_free − c
            (R total).

    The quadratic (step 2):
        A: (R, D). A[row_t, col_t] = √p_t I for free t;
            A[row_t, col_s] = −√p_t W_eff[s→t]ᵀ for each edge s→t with s free.
        c: (R, batch). c[row_t] = √p_t (b_t − ẑ_t·[t clamped]
            + Σ_{s→t, s clamped} W_eff[s→t]ᵀ ẑ_s).
        precision: name -> p_t of each row node.

    Error coordinates (steps 3–4):
        B_lower: (D, D). B[col_t, col_s] = W_eff[s→t]ᵀ for each edge s→t with
            s and t free; strictly block-lower-triangular.
        M: (D, D) = (I − B_lower)⁻¹, so z_free = M (ε + k).
        z_ff: name -> (batch, d) latents at ε = 0, clamps included.
        source_means: name -> (batch, d) constant z_mu of each unclamped
            source; its block of k.
    """

    free: Tuple[str, ...]
    rows: Tuple[str, ...]
    col_offsets: Dict[str, slice]
    row_offsets: Dict[str, slice]
    A: Array
    c: Array
    B_lower: Array
    M: Array
    z_ff: PerNode
    precision: Dict[str, float]
    source_means: PerNode


class LinearEquilibrium(NamedTuple):
    """The minimizer of E and its readouts (Part 1, steps 5–6), per sample.

    Attributes:
        z_star: name -> (batch, d). z*_free from least squares for free
            nodes; ẑ_t for clamped nodes.
        z_mu_star: name -> (batch, d). b_t + Σ_s z*_s @ W_eff[s→t] for row
            nodes; a source's clamp or ``source_means`` entry.
        error_star: name -> (batch, d) = z_star − z_mu_star, every node.
        node_energy: name -> (batch,) = ½ p_t ‖error_star‖², every row node.
        total_energy: (batch,) = Σ node_energy = ½‖A z*_free − c‖².
        min_singular_value: σ_min(A), the margin from a non-unique
            equilibrium.
        quad: the assembled quadratic.
    """

    z_star: PerNode
    z_mu_star: PerNode
    error_star: PerNode
    node_energy: PerNode
    total_energy: Array
    min_singular_value: float
    quad: LinearQuadratic


def _node_info(structure: GraphStructure, name: str):
    return structure.nodes[name].node_info


def validate_linear_gaussian(structure: GraphStructure) -> None:
    """Raise ``ValueError`` unless the graph's energy is the quadratic above.

    Requirements: a DAG (every edge runs forward in ``node_order``; the
    schedule length is not a DAG test, since a cycle unrolled once visits
    each member once); node classes ``Linear`` or ``IdentityNode`` exactly;
    rank-1 node shapes; and on every ``in_degree > 0`` node
    ``IdentityActivation``, ``GaussianEnergy``, ``flatten_input=False``, and
    no ``energy()`` override.
    """
    order = {name: i for i, name in enumerate(structure.node_order)}
    for edge in structure.edges.values():
        if order[edge.source] >= order[edge.target]:
            raise ValueError(
                f"edge '{edge.key}' runs against node_order: the graph has a "
                f"cycle, and the oracle is defined on DAGs only"
            )
    for name in structure.node_order:
        info = _node_info(structure, name)
        node_class = info.node_class
        if node_class not in (Linear, IdentityNode):
            raise ValueError(
                f"node '{name}' is {node_class.__name__}; the oracle accepts "
                f"Linear and IdentityNode only"
            )
        if len(info.shape) != 1:
            raise ValueError(
                f"node '{name}' has shape {info.shape}; the oracle accepts "
                f"rank-1 node shapes only"
            )
        if info.in_degree == 0:
            continue
        if not isinstance(info.activation, IdentityActivation):
            raise ValueError(
                f"node '{name}' has activation "
                f"{type(info.activation).__name__}; the energy is quadratic "
                f"only under IdentityActivation"
            )
        if not isinstance(info.energy, GaussianEnergy):
            raise ValueError(
                f"node '{name}' has energy {type(info.energy).__name__}; the "
                f"oracle requires GaussianEnergy"
            )
        if info.node_config.get("flatten_input", False):
            raise ValueError(
                f"node '{name}' sets flatten_input=True; the oracle reads "
                f"last-axis weights only"
            )
        if node_class.energy is not NodeBase.energy:
            raise ValueError(
                f"node '{name}' ({node_class.__name__}) overrides energy(); "
                f"the oracle knows the Gaussian term only"
            )


def _precision(structure: GraphStructure, name: str) -> float:
    config = _node_info(structure, name).energy.config
    return float(config.get("precision", 1.0)) if config else 1.0


def _bias(params: GraphParams, name: str, dim: int) -> Array:
    biases = params.nodes[name].biases
    if "b" in biases and biases["b"].size > 0:
        return np.asarray(biases["b"], dtype=np.float64).reshape(dim)
    return np.zeros(dim)


def effective_edge_matrices(
    params: GraphParams, structure: GraphStructure
) -> Dict[str, Array]:
    """Row-convention W_eff per edge key: z_mu_t = Σ_s z_s @ W_eff[s→t] + b_t.

    muPC's ``scale_inputs`` multiplies the source latent by the edge's
    ``forward_scale`` before the node's matmul, so the scale folds into the
    matrix. A source ``IdentityNode`` never runs ``predict``, so its
    ``scale`` is inert; the identity map is built for in-degree > 0 targets.
    """
    matrices: Dict[str, Array] = {}
    for key, edge in structure.edges.items():
        info = _node_info(structure, edge.target)
        scaling = info.scaling_config
        forward_scale = 1.0
        if scaling is not None and key in scaling.forward_scale:
            forward_scale = float(scaling.forward_scale[key])
        d_s = _node_info(structure, edge.source).shape[0]
        d_t = info.shape[0]
        if info.node_class is Linear:
            w = np.asarray(params.nodes[edge.target].weights[key], dtype=np.float64)
        else:
            if d_s != d_t:
                raise ValueError(
                    f"IdentityNode '{edge.target}' receives dim {d_s} from "
                    f"'{edge.source}' but has dim {d_t}"
                )
            w = float(info.node_config["scale"]) * np.eye(d_t)
        matrices[key] = forward_scale * w
    return matrices


def _clamp_array(clamps: Mapping[str, object], name: str) -> Array:
    return np.asarray(clamps[name], dtype=np.float64)


def assemble_linear_quadratic(
    params: GraphParams,
    structure: GraphStructure,
    clamps: Mapping[str, object],
    *,
    source_means: Optional[PerNode] = None,
) -> LinearQuadratic:
    """Steps 1–4 of Part 1: A, c, B_lower, M, and z_ff for the graph.

    One pass over ``rows`` writes row block t of A and c (step 2) and, when t
    is free, row block t of B (step 3). A forward pass along ``node_order``
    then fills z_ff (step 4). ``clamps`` maps node names to (batch, d) arrays.
    ``source_means`` maps each unclamped source to its constant z_mu
    (``initialize_graph_state`` sets it to the source's initial latent);
    missing entries default to zeros.
    """
    validate_linear_gaussian(structure)
    clamped = set(clamps)
    source_means = {
        k: np.asarray(v, dtype=np.float64) for k, v in (source_means or {}).items()
    }
    batch = int(next(iter(clamps.values())).shape[0])
    dims = {name: _node_info(structure, name).shape[0] for name in structure.node_order}
    w_eff = effective_edge_matrices(params, structure)
    in_edges = {
        name: [structure.edges[k] for k in _node_info(structure, name).in_edges]
        for name in structure.node_order
    }

    free = tuple(n for n in structure.node_order if n not in clamped)
    rows = tuple(
        n for n in structure.node_order if _node_info(structure, n).in_degree > 0
    )
    col_offsets: Dict[str, slice] = {}
    start = 0
    for name in free:
        col_offsets[name] = slice(start, start + dims[name])
        start += dims[name]
    D = start
    row_offsets: Dict[str, slice] = {}
    start = 0
    for name in rows:
        row_offsets[name] = slice(start, start + dims[name])
        start += dims[name]
    R = start

    precision = {name: _precision(structure, name) for name in rows}
    A = np.zeros((R, D))
    c = np.zeros((R, batch))
    B = np.zeros((D, D))
    for t in rows:
        # Step 2: row block t of A and c, (A z_free − c)[row_t] = √p_t (z_t − μ_t).
        sp = math.sqrt(precision[t])
        r = row_offsets[t]
        # offset = constant part of μ_t − z_t. c[row_t] = √p_t · offset.
        offset = _bias(params, t, dims[t])[:, None] * np.ones((1, batch))
        if t in clamped:
            offset = offset - _clamp_array(clamps, t).T  # −ẑ_t
        else:
            A[r, col_offsets[t]] = sp * np.eye(dims[t])  # √p_t z_t
        for edge in in_edges[t]:
            wt = w_eff[edge.key].T  # (d_t, d_s): acts on a stacked column
            if edge.source in clamped:
                offset = offset + wt @ _clamp_array(clamps, edge.source).T  # W_effᵀ ẑ_s
            else:
                A[r, col_offsets[edge.source]] += -sp * wt  # −√p_t W_effᵀ z_s
                if t not in clamped:
                    B[col_offsets[t], col_offsets[edge.source]] += wt  # step 3
        c[r] = sp * offset

    M = np.linalg.inv(np.eye(D) - B)  # Step 3: z_free = M (ε + k)

    # Step 4: z_ff = M k by a forward pass along node_order at ε = 0.
    z_ff: PerNode = {}
    for name in structure.node_order:
        if name in clamped:
            z_ff[name] = _clamp_array(clamps, name)
        elif _node_info(structure, name).in_degree == 0:
            z_ff[name] = source_means.get(name, np.zeros((batch, dims[name])))
        else:
            mu = np.ones((batch, 1)) * _bias(params, name, dims[name])[None, :]
            for edge in in_edges[name]:
                mu = mu + z_ff[edge.source] @ w_eff[edge.key]
            z_ff[name] = mu

    return LinearQuadratic(
        free=free,
        rows=rows,
        col_offsets=col_offsets,
        row_offsets=row_offsets,
        A=A,
        c=c,
        B_lower=B,
        M=M,
        z_ff=z_ff,
        precision=precision,
        source_means={
            n: source_means.get(n, np.zeros((batch, dims[n])))
            for n in free
            if _node_info(structure, n).in_degree == 0
        },
    )


def flatten_free(quad: LinearQuadratic, per_node: Mapping[str, object]) -> Array:
    """Stack (batch, d) arrays of the free nodes into a (D, batch) matrix in
    ``node_order`` (never ``tree_leaves``, which sorts dict keys)."""
    return np.concatenate(
        [np.asarray(per_node[n], dtype=np.float64).T for n in quad.free], axis=0
    )


def unflatten_free(quad: LinearQuadratic, stacked: Array) -> PerNode:
    """Inverse of ``flatten_free``: (D, batch) -> name -> (batch, d)."""
    return {n: stacked[quad.col_offsets[n]].T for n in quad.free}


def linear_equilibrium(
    params: GraphParams,
    structure: GraphStructure,
    clamps: Mapping[str, object],
    *,
    source_means: Optional[PerNode] = None,
) -> LinearEquilibrium:
    """Steps 5–6 of Part 1: the exact minimizer of E and its readouts.

    Assembles the quadratic, solves z*_free = lstsq(A, c), then recomputes
    z_mu_star, error_star, node_energy, and total_energy from z* with W_eff
    and the biases. Raises ``ValueError`` when A is rank-deficient (non-unique
    equilibrium, usually an unclamped source whose outgoing maps are not
    jointly injective).
    """
    quad = assemble_linear_quadratic(
        params, structure, clamps, source_means=source_means
    )
    D = quad.A.shape[1]
    # Step 5: least squares; rank D is uniqueness.
    z_free, _, rank, singular = np.linalg.lstsq(quad.A, quad.c, rcond=None)
    if rank < D:
        raise ValueError(
            f"non-unique equilibrium: A has rank {rank} < {D} free "
            f"dimensions (smallest singular value {singular[-1]:.3e})"
        )
    min_sv = float(singular[-1]) if D > 0 else float("inf")
    z_star = dict(quad.z_ff)
    z_star.update(unflatten_free(quad, z_free))

    # Step 6: readouts from z* with W_eff and b.
    w_eff = effective_edge_matrices(params, structure)
    dims = {name: _node_info(structure, name).shape[0] for name in structure.node_order}
    batch = quad.c.shape[1]
    z_mu_star: PerNode = {}
    error_star: PerNode = {}
    node_energy: PerNode = {}
    for name in structure.node_order:
        info = _node_info(structure, name)
        if info.in_degree == 0:
            z_mu_star[name] = (
                z_star[name] if name in clamps else quad.source_means[name]
            )
            error_star[name] = z_star[name] - z_mu_star[name]
            continue
        mu = np.ones((batch, 1)) * _bias(params, name, dims[name])[None, :]
        for key in info.in_edges:
            mu = mu + z_star[structure.edges[key].source] @ w_eff[key]
        z_mu_star[name] = mu
        error_star[name] = z_star[name] - mu
        node_energy[name] = (
            0.5 * quad.precision[name] * np.sum(error_star[name] ** 2, axis=1)
        )
    total = sum(node_energy.values()) if node_energy else np.zeros(batch)
    return LinearEquilibrium(
        z_star=z_star,
        z_mu_star=z_mu_star,
        error_star=error_star,
        node_energy=node_energy,
        total_energy=total,
        min_singular_value=min_sv,
        quad=quad,
    )


def theorem1_energy(
    params: GraphParams, structure: GraphStructure, clamps: Mapping[str, object]
) -> Tuple[Array, Array, Array]:
    """Closed-form equilibrium energy of a clamped chain (Innocenti et al.
    2024, Theorem 1, extended to per-node precisions and biases).

    For x → h_1 → ⋯ → h_L → y with x and y clamped and every hidden node
    free, the residual at the feedforward point is r = y − μ_y(ff) (one row
    per sample), P_l = W_eff[l+1] ⋯ W_eff[L+1] maps ε_l to the output
    prediction, and

        S = I + Σ_l (p_y / p_l) · P_lᵀ P_l,     E* = ½ · p_y · r S⁻¹ rᵀ.

    Biases and muPC scales enter through r and P_l. Returns
    ``(E_star (batch,), S, r)``.
    """
    validate_linear_gaussian(structure)
    order = structure.node_order
    if len(order) < 3:
        raise ValueError("theorem1_energy needs at least x → h → y")
    x, y = order[0], order[-1]
    hidden = order[1:-1]
    if x not in clamps or y not in clamps:
        raise ValueError("theorem1_energy requires the chain's ends clamped")
    if any(h in clamps for h in hidden):
        raise ValueError("theorem1_energy requires unclamped hidden nodes")
    for prev, name in zip(order[:-1], order[1:]):
        edges = [structure.edges[k] for k in _node_info(structure, name).in_edges]
        if len(edges) != 1 or edges[0].source != prev:
            raise ValueError(
                f"theorem1_energy requires a chain; node '{name}' does not "
                f"receive exactly one edge from '{prev}'"
            )
    w_eff = effective_edge_matrices(params, structure)
    chain_w = [
        w_eff[_node_info(structure, name).in_edges[0]] for name in order[1:]
    ]  # chain_w[i] maps order[i] -> order[i+1]

    # Feedforward prediction of the output.
    z = _clamp_array(clamps, x)
    for name, w in zip(order[1:], chain_w):
        z = z @ w + _bias(params, name, _node_info(structure, name).shape[0])[None, :]
    r = _clamp_array(clamps, y) - z

    p_y = _precision(structure, y)
    d_y = _node_info(structure, y).shape[0]
    S = np.eye(d_y)
    for i, h in enumerate(hidden):
        P = np.eye(_node_info(structure, h).shape[0])
        for w in chain_w[i + 1 :]:
            P = P @ w
        S = S + (p_y / _precision(structure, h)) * (P.T @ P)
    E_star = 0.5 * p_y * np.einsum("nd,nd->n", r @ np.linalg.inv(S), r)
    return E_star, S, r


# =============================================================================
# Part 2 — spectral diagnostics
# =============================================================================


def latent_hessian(quad: LinearQuadratic) -> Array:
    """H_z = AᵀA: curvature seen by the state-based solvers."""
    return quad.A.T @ quad.A


def epsilon_hessian(quad: LinearQuadratic) -> Array:
    """H_ε = Mᵀ AᵀA M: curvature seen by ``EPCInference``."""
    AM = quad.A @ quad.M
    return AM.T @ AM


def latent_gradient_at_feedforward(quad: LinearQuadratic) -> Array:
    """∇_z E at the feedforward point, (D, batch): sPC's initial gradient."""
    z_ff = flatten_free(quad, quad.z_ff)
    return quad.A.T @ (quad.A @ z_ff - quad.c)


def epsilon_gradient_at_zero(quad: LinearQuadratic) -> Array:
    """∇_ε E at ε = 0, (D, batch): ePC's initial gradient, and the backprop
    activation gradient at the feedforward point."""
    return quad.M.T @ latent_gradient_at_feedforward(quad)


def stability_bound(H: Array) -> float:
    """Largest stable gradient-descent rate on ½ xᵀHx: 2 / λ_max(H).

    Raises ``ValueError`` when λ_max ≤ 0: with no positive curvature there
    is no descent direction to bound, and 2/λ_max would be a negative rate.
    """
    lam_max = float(np.linalg.eigvalsh(H)[-1])
    if lam_max <= 0.0:
        raise ValueError(
            f"no positive curvature: lambda_max = {lam_max:.3g} <= 0, so the "
            f"quadratic has no stable gradient-descent rate"
        )
    return 2.0 / lam_max


def excited_eigenvalues(H: Array, g0: Array, rel_tol: float = 1e-8) -> Array:
    """Eigenvalues of H whose eigenvectors overlap the initial gradient.

    Gradient descent from x_0 on ½ xᵀHx + gᵀx evolves each eigencomponent
    independently, so components with zero initial gradient stay zero.
    Returns the eigenvalues whose eigenvector has relative overlap with
    ``g0`` (D, batch) above ``rel_tol`` in at least one sample.
    """
    eigs, vecs = np.linalg.eigh(H)
    overlap = np.abs(vecs.T @ g0)  # (D, batch)
    scale = np.linalg.norm(g0, axis=0, keepdims=True) + 1e-300
    excited = (overlap / scale > rel_tol).any(axis=1)
    return eigs[excited]


def gradient_weights(H: Array, g0: Array) -> Tuple[Array, Array]:
    """Every eigenvalue of H with the fraction of ‖g0‖² its eigenvector carries.

    The batch Hessian is block-diagonal over samples with H as every block,
    so the weight on eigenvalue λ sums the squared overlaps over the samples:
    w_λ = Σ_n (v_λᵀ g0[:, n])² / ‖g0‖_F². Returns ``(eigs, weights)``, both
    (D,) in ascending eigenvalue order with the weights summing to 1; an
    unexcited mode has weight 0. These are the exact counterparts of the
    Lanczos Ritz values and weights in ``fabricpc.core.epsilon_spectrum``.
    """
    eigs, vecs = np.linalg.eigh(H)
    overlap = vecs.T @ g0  # (D, batch)
    weights = np.sum(overlap**2, axis=1)
    return eigs, weights / weights.sum()


def weighted_relaxed_fraction(eigs, weights, eta: float, steps: int) -> float:
    """Σ_{λ > 0} w_λ·f(λ) / Σ_{λ > 0} w_λ with f(λ) = 1 − (1 − eta·λ)^steps,
    the exact form of the Lanczos f̄ for eigenvalues and weights from
    :func:`gradient_weights`."""
    eigs = np.asarray(eigs, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    positive = eigs > 0
    total = float(weights[positive].sum())
    if total <= 0.0:
        return float("nan")
    f = relaxed_fraction(eta, steps, eigs[positive])
    return float((weights[positive] * f).sum() / total)


def relaxed_fraction(eta: float, steps: int, eigs) -> Array:
    """Fraction of each eigenmode's distance to equilibrium closed after
    ``steps`` gradient steps at rate ``eta``: 1 − (1 − eta·eigs)^steps."""
    return 1.0 - (1.0 - eta * np.asarray(eigs, dtype=np.float64)) ** steps


def steps_to_contract(eta: float, eigs, ratio: float) -> int:
    """Smallest step count after which every mode's distance to equilibrium
    has shrunk by at least ``ratio``. Raises if some mode does not contract
    (eta·λ ≤ 0 or ≥ 2)."""
    factors = np.abs(1.0 - eta * np.asarray(eigs, dtype=np.float64))
    if factors.size == 0:
        return 1
    worst = float(factors.max())
    if worst >= 1.0:
        raise ValueError(
            f"a mode does not contract at eta={eta}: max |1 - eta*lambda| = {worst}"
        )
    if worst == 0.0 or ratio >= 1.0:
        return 1
    return int(math.ceil(math.log(ratio) / math.log(worst)))
