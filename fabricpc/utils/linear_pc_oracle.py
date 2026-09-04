"""
Exact predictive-coding equilibria on linear-Gaussian DAGs, and the spectral
diagnostics that set how fast each solver reaches them.

Part 1 — exact equilibrium (pure NumPy, float64). On a graph whose nodes are
``Linear`` or ``IdentityNode`` with ``IdentityActivation`` and
``GaussianEnergy``, the total energy over the ``in_degree > 0`` nodes,

    E = Σ_t ½·p_t·‖z_t − μ_t‖²,      μ_t = Σ_s z_s W_eff[s→t] + b_t,

is a quadratic in the stacked free latents z_free: E = ½‖A z_free − c‖²,
one residual row block per in-degree > 0 node, one column block per
unclamped node. The equilibrium is the least-squares solution
z* = argmin ‖A z − c‖; the exact energy, per-node energies, and errors
follow from it. Nothing here calls node or solver code: the assembly reads
params, edges, muPC forward scales, biases, and precisions only, so it is an
independent reference for ``EPCInference`` and the state-based solvers.

Conventions. Nodes are enumerated in ``structure.node_order`` (topological
on a DAG); samples are columns of c; the library's row convention
``z_mu = z_s @ W`` becomes ``W_eff[s→t]ᵀ`` acting on a stacked column.
W_eff[s→t] is the muPC ``forward_scale`` for the edge (1.0 when absent)
times the Linear weight for that edge key, or times the IdentityNode
``scale`` times the identity. An unclamped source (in_degree 0) has a
column block but no row block: it carries no energy term, and under
``EPCInference`` its ``z_mu`` stays at the constant ``begin_segment``
assigned (``source_means``), which affects its error ε* but not z* or E*.

Part 2 — spectral diagnostics. The Hessian in latent coordinates is
H_z = AᵀA. Error coordinates are z_free = M(ε + const) with
M = (I − B)⁻¹, where B is the strictly block-lower-triangular map through
the edges, so H_ε = Mᵀ H_z M. The two Hessians govern the two solvers:

- λ_min(H_z) decays with depth even for benign weights: the state-based
  solver's slow mode, which needs ~κ(H_z) steps to relax.
- λ_max(H_ε) = 1 + σ_max(J)² on a chain with unit precision, J the map from
  the stacked errors to the output prediction, grows with the product of
  downstream weights: ePC's stability bound 2/λ_max(H_ε) shrinks as the
  weights grow. From ε = 0 with uniform precision and no unclamped source,
  ePC's trajectory lives in the d_y-dimensional row space of J and sees
  only eig(S), S = I + Σ_l P_lᵀP_l (Innocenti et al. 2024, Theorem 1).

Regime. After T gradient steps at rate η from ε = 0, each excited eigenmode
λ of H_ε has relaxed toward equilibrium by 1 − (1 − ηλ)^T. Backprop-like
behaviour (ε ≈ −η·T·∇E, the paper's Theorem C.9) requires η·T·λ_max ≪ 1;
near-equilibrium requires η·T·λ_min,excited ≳ 3; stability requires
η·λ_max < 2, T = 1 included: one step lands each mode at ε_1 = η·λ·ε*_λ,
so a mode with η·λ > 2 ends farther from equilibrium than it started.
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
    """E = ½‖A z_free − c‖² over the stacked free latents (samples as columns).

    Attributes:
        free: unclamped node names in ``node_order`` (the column blocks).
        rows: ``in_degree > 0`` node names in ``node_order`` (the row blocks).
        col_offsets: name -> slice of that node's columns in z_free.
        row_offsets: name -> slice of that node's rows in the residual.
        A: (R, D) residual Jacobian, precision-weighted.
        c: (R, batch) residual offset (biases, clamps), precision-weighted.
        B_lower: (D, D) strictly block-lower-triangular map z_free <- z_free.
        M: (D, D) = (I − B_lower)⁻¹, the ε -> z map.
        z_ff: name -> (batch, d) feedforward latents (the ε = 0 point).
        precision: name -> precision of each row node.
        source_means: name -> (batch, d) constant z_mu of each unclamped source.
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
    """The exact minimizer of E and its readouts, all per sample.

    Attributes:
        z_star: name -> (batch, d) equilibrium latents (clamps included).
        z_mu_star: name -> (batch, d) predictions at equilibrium (a source's
            is its clamp or ``source_means`` entry).
        error_star: name -> (batch, d) = z_star − z_mu_star.
        node_energy: name -> (batch,) for every ``in_degree > 0`` node.
        total_energy: (batch,) sum of ``node_energy``.
        min_singular_value: smallest singular value of A (uniqueness margin).
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
    """Build A, c, B_lower, M, and the feedforward point for the graph.

    ``clamps`` maps node names to (batch, d) arrays. ``source_means`` maps
    each unclamped source to its constant z_mu (the state's ``z_mu`` after
    ``EPCInference.begin_segment``); missing entries default to zeros.
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
        sp = math.sqrt(precision[t])
        r = row_offsets[t]
        offset = _bias(params, t, dims[t])[:, None] * np.ones((1, batch))
        if t in clamped:
            offset = offset - _clamp_array(clamps, t).T
        else:
            A[r, col_offsets[t]] = sp * np.eye(dims[t])
        for edge in in_edges[t]:
            wt = w_eff[edge.key].T  # (d_t, d_s): acts on a stacked column
            if edge.source in clamped:
                offset = offset + wt @ _clamp_array(clamps, edge.source).T
            else:
                A[r, col_offsets[edge.source]] += -sp * wt
                if t not in clamped:
                    B[col_offsets[t], col_offsets[edge.source]] += wt
        c[r] = sp * offset

    M = np.linalg.inv(np.eye(D) - B)

    # Feedforward point: derive along node_order at ε = 0.
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
    """Exact minimizer of the linear-Gaussian energy by least squares.

    Raises ``ValueError`` when A is rank-deficient (non-unique equilibrium,
    usually an unclamped source whose outgoing maps are not jointly
    injective).
    """
    quad = assemble_linear_quadratic(
        params, structure, clamps, source_means=source_means
    )
    D = quad.A.shape[1]
    z_free, _, rank, singular = np.linalg.lstsq(quad.A, quad.c, rcond=None)
    if rank < D:
        raise ValueError(
            f"non-unique equilibrium: A has rank {rank} < {D} free "
            f"dimensions (smallest singular value {singular[-1]:.3e})"
        )
    min_sv = float(singular[-1]) if D > 0 else float("inf")
    z_star = dict(quad.z_ff)
    z_star.update(unflatten_free(quad, z_free))

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
    """Largest stable gradient-descent rate on ½ xᵀHx: 2 / λ_max(H)."""
    return 2.0 / float(np.linalg.eigvalsh(H)[-1])


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


def relaxed_fraction(eta: float, steps: int, eigs) -> Array:
    """Fraction of each eigenmode's distance to equilibrium closed after
    ``steps`` gradient steps at rate ``eta``: 1 − (1 − eta·eigs)^steps."""
    return 1.0 - (1.0 - eta * np.asarray(eigs, dtype=np.float64)) ** steps


def steps_to_contract(eta: float, eigs, ratio: float) -> int:
    """Smallest step count after which every mode's distance to equilibrium
    has shrunk by at least ``ratio``. Raises if some mode does not contract
    (eta·λ ≤ 0 or ≥ 2)."""
    factors = np.abs(1.0 - eta * np.asarray(eigs, dtype=np.float64))
    worst = float(factors.max())
    if worst >= 1.0:
        raise ValueError(
            f"a mode does not contract at eta={eta}: max |1 - eta*lambda| = {worst}"
        )
    if worst == 0.0:
        return 1
    return int(math.ceil(math.log(ratio) / math.log(worst)))
