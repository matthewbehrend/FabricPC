"""
Lanczos estimate of the curvature ePC's starting gradient excites.

``EPCInference`` is gradient descent on the total energy in error
coordinates. Every FabricPC run starts inference at the feedforward state,
ε = 0, where the energy is locally a quadratic with Hessian H_ε and the
starting gradient is g0 = ∇_ε E, the backprop activation gradient. Gradient
descent on a quadratic splits into independent modes along the eigenvectors
of H_ε, and only the modes along which g0 has a component ever move: the
excited modes. Three facts about them set the solver's regime:

- λ_max, the largest excited eigenvalue: the iteration diverges unless
  eta_infer·λ_max < 2, and at odd step counts the output-layer weight
  gradient reverses sign once eta_infer·(λ_max − 1) > 1.
- λ_min, the smallest excited eigenvalue: negative means the energy is
  indefinite at ε = 0 (the second-derivative term Σ_k ∂L/∂μ_k·∂²μ_k/∂ε² of a
  nonlinear graph), and a mode with negative curvature grows by
  (1 + eta_infer·|λ|)^T over T steps instead of relaxing.
- the distribution of ‖g0‖² over the spectrum: after T steps a mode with
  eigenvalue λ has relaxed by f(λ) = 1 − (1 − eta_infer·λ)^T, and the
  gradient-weighted relaxed fraction f̄ = Σ_k w_k f(θ_k) / Σ_k w_k over the
  positive modes says how far the errors that drive the weight update have
  moved from backprop's (f̄ ≪ 0.1) toward the PC equilibrium (f̄ > 0.9).

Lanczos delivers all three from Hessian-vector products alone. Started at
v0 = g0 it builds an orthonormal basis of the Krylov space
span{g0, H g0, H² g0, …} and represents H_ε there as a k × k tridiagonal
matrix T_k with diagonal α_j and off-diagonal β_j. The eigenvalues of T_k,
the Ritz values θ_k, approximate eigenvalues of H_ε and converge fastest at
both ends of the spectrum; the squared first component of each eigenvector
of T_k, w_k, is the fraction of ‖g0‖² that Ritz mode carries (Σ_k w_k = 1).
Three vectors of ε size are carried, so memory is independent of ``iters``.

Breakdown. When β_j ≤ √eps(dtype)·max(max_i |α_i|, max_i β_i), the Krylov
space is exhausted (on a linear graph its dimension is the number of
distinct excited eigenvalues, at most the output dimension) and the
recurrence freezes: ``k`` counts the valid steps, and the remaining steps
carry α_j = α_0 (the Rayleigh quotient of g0, inside the excited spectrum by
construction) and β_j = 0, so they enter T_k as 1 × 1 blocks with zero
weight that move neither extreme nor f̄. Continuing past this point on
rounding noise would converge to unexcited modes (the unit-precision floor
on a linear graph) and misreport λ_min. A direction entering with β/|α|
below √eps carries weight below eps in g0, the same cutoff the linear
oracle's ``excited_eigenvalues`` applies through its overlap tolerance.

Weight floor. The guard fires when the Krylov space is small (three
excited modes, E1 of the design plan). With ten excited modes in float32
the rounding noise accumulated over ten steps keeps β above the threshold,
the recurrence continues on that noise, and the unexcited floor appears as
a Ritz value carrying weight of order 1e-14 (measured on the depth-5 chain
of ``scripts/epc_analysis.py --section stability``: floor ghosts at weight
4e-14 against 6e-4 on the smallest excited mode). λ_max and λ_min are
therefore taken over the Ritz modes whose weight exceeds eps(dtype), the
same cutoff the guard expresses through β: a mode below it is
indistinguishable from rounding noise, and within T inference steps it
cannot grow from that seed to anything the weight update sees. The full
``ritz_values`` and ``ritz_weights`` are returned unfiltered.

Ghosts. Without reorthogonalization, lost orthogonality in long runs
duplicates converged extreme eigenvalues and splits their weight between
the copies. The duplicates move neither the extremes nor any weighted sum
over the Ritz values, which is all the regime reads.

Scope. The Hessian is evaluated at the state's latents on one batch.
``error_energy`` sums per-sample energies, so H_ε is block-diagonal over
samples and a batch's λ_max is the per-sample maximum; a training batch's
bound is at least as tight as a smaller probe batch's. On a nonlinear graph
the quadratic model is local, and every statement above holds at ε = 0.
Cyclic graphs are outside the description: their unrolled ε-energy carries
warm-started latents, so it is not a pure function of ε.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from fabricpc.core.types import GraphParams, GraphState, GraphStructure

Pytree = Any


class LanczosResult(NamedTuple):
    """Ritz decomposition of a Lanczos run.

    Attributes:
        ritz_values: (iters,) eigenvalues θ_k of the tridiagonal T_k,
            ascending; frozen steps after a breakdown contribute α_0.
        ritz_weights: (iters,) fraction of ‖v0‖² carried by each Ritz mode,
            the squared first component of T_k's eigenvectors; sums to 1,
            zero on frozen steps.
        lambda_max, lambda_min: the largest and smallest Ritz values among
            the modes whose weight exceeds eps(dtype) (the weight floor of
            the module docstring).
        residual_min, residual_max: β_k·|s_{k−1}| for those two Ritz pairs,
            the norm of H y − θ y for that Ritz vector; zero after a
            breakdown.
        breakdown: the recurrence froze before ``iters`` steps.
        k: valid Lanczos steps (the dimension of the Krylov space found).
    """

    ritz_values: jnp.ndarray
    ritz_weights: jnp.ndarray
    lambda_max: jnp.ndarray
    lambda_min: jnp.ndarray
    residual_min: jnp.ndarray
    residual_max: jnp.ndarray
    breakdown: jnp.ndarray
    k: jnp.ndarray


class EpsilonSpectrum(NamedTuple):
    """The excited spectrum of H_ε at one state on one batch.

    Attributes:
        lambda_max, lambda_min: the largest and smallest Ritz values among
            the modes carrying weight above eps(dtype), the extremes of the
            excited spectrum.
        ritz_values, ritz_weights: as in :class:`LanczosResult`.
        residual_max, residual_min: Ritz residuals of the two extremes.
        negative_weight: Σ_{θ_k < 0} w_k, the fraction of ‖g0‖² on
            negative-curvature modes.
        gradient_norm: ‖g0‖, the norm of the starting gradient.
        random_start: g0 was zero, so a random start vector was used and the
            extremes describe the full spectrum rather than the excited one.
        iters: Lanczos steps run.
        k: valid steps (see :class:`LanczosResult`).
    """

    lambda_max: Any
    lambda_min: Any
    ritz_values: Any
    ritz_weights: Any
    residual_max: Any
    residual_min: Any
    negative_weight: Any
    gradient_norm: Any
    random_start: Any
    iters: Any
    k: Any

    def host(self) -> "EpsilonSpectrum":
        """The same spectrum with NumPy arrays and Python scalars."""
        out = []
        for value in self:
            array = np.asarray(value)
            out.append(array.item() if array.ndim == 0 else array)
        return EpsilonSpectrum(*out)

    @classmethod
    def from_modes(
        cls, values: Sequence[float], weights: Sequence[float]
    ) -> "EpsilonSpectrum":
        """A spectrum built from explicit eigenvalues and gradient weights
        (normalized to sum to 1), for analyses and tests that start from a
        known spectrum rather than a graph."""
        values_arr = np.asarray(values, dtype=np.float64).reshape(-1)
        weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
        if values_arr.shape != weights_arr.shape or values_arr.size == 0:
            raise ValueError(
                f"from_modes needs equal-length non-empty values and weights, "
                f"got {values_arr.shape} and {weights_arr.shape}"
            )
        weights_arr = weights_arr / weights_arr.sum()
        order = np.argsort(values_arr)
        values_arr, weights_arr = values_arr[order], weights_arr[order]
        return cls(
            lambda_max=float(values_arr[-1]),
            lambda_min=float(values_arr[0]),
            ritz_values=values_arr,
            ritz_weights=weights_arr,
            residual_max=0.0,
            residual_min=0.0,
            negative_weight=float(weights_arr[values_arr < 0].sum()),
            gradient_norm=float("nan"),
            random_start=False,
            iters=int(values_arr.size),
            k=int(values_arr.size),
        )


def weighted_relaxed_fraction(
    spectrum: EpsilonSpectrum, eta: float, steps: int
) -> float:
    """f̄ = Σ_{θ_k > 0} w_k·f(θ_k) / Σ_{θ_k > 0} w_k with f(θ) = 1 − (1 − eta·θ)^steps.

    The relaxed fraction of the gradient that drives the weight update,
    averaged over the positive-curvature Ritz modes by the weight each
    carries. Modes with θ_k ≤ 0 have no minimum to relax toward and are left
    out; :attr:`EpsilonSpectrum.negative_weight` and the regime's growth
    factor judge them. Returns ``nan`` when no positive mode carries weight.
    """
    theta = np.asarray(spectrum.ritz_values, dtype=np.float64)
    w = np.asarray(spectrum.ritz_weights, dtype=np.float64)
    positive = theta > 0
    total = float(w[positive].sum())
    if total <= 0.0:
        return float("nan")
    f = 1.0 - (1.0 - eta * theta[positive]) ** int(steps)
    return float((w[positive] * f).sum() / total)


# -----------------------------------------------------------------------------
# Pytree arithmetic
# -----------------------------------------------------------------------------


def _leaves(tree: Pytree):
    return jax.tree_util.tree_leaves(tree)


def _dot(a: Pytree, b: Pytree) -> jnp.ndarray:
    return sum(jnp.sum(x * y) for x, y in zip(_leaves(a), _leaves(b)))


def _axpy(alpha, x: Pytree, y: Pytree) -> Pytree:
    """alpha·x + y."""
    return jax.tree_util.tree_map(lambda xi, yi: alpha * xi + yi, x, y)


def _scale(s, x: Pytree) -> Pytree:
    return jax.tree_util.tree_map(lambda xi: s * xi, x)


def _random_like(key: jax.Array, tree: Pytree) -> Pytree:
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    return treedef.unflatten(
        [jax.random.normal(k, leaf.shape, leaf.dtype) for k, leaf in zip(keys, leaves)]
    )


# -----------------------------------------------------------------------------
# Lanczos
# -----------------------------------------------------------------------------


def lanczos_extremes(
    hvp: Callable[[Pytree], Pytree], v0: Pytree, iters: int
) -> LanczosResult:
    """Run ``iters`` steps of the three-term Lanczos recurrence from ``v0``.

    ``hvp(v)`` returns H v for a pytree ``v`` shaped like ``v0``; ``v0`` must
    be nonzero. Only three vectors of ``v0``'s size are live at once. The
    relative breakdown guard of the module docstring freezes the recurrence
    once the Krylov space is exhausted. Traceable under ``jax.jit``.
    """
    if iters < 1:
        raise ValueError(f"lanczos_extremes needs iters >= 1, got {iters}")
    dtype = _leaves(v0)[0].dtype
    tol = jnp.sqrt(jnp.finfo(dtype).eps).astype(dtype)

    v0_norm = jnp.sqrt(_dot(v0, v0))
    q_cur = _scale(1.0 / v0_norm, v0)
    q_prev = jax.tree_util.tree_map(jnp.zeros_like, v0)
    alphas = jnp.zeros((iters,), dtype)
    betas = jnp.zeros((iters,), dtype)
    carry = (
        q_prev,
        q_cur,
        jnp.zeros((), dtype),
        alphas,
        betas,
        jnp.asarray(False),
        jnp.asarray(0, jnp.int32),
    )

    def active(j, carry):
        q_prev, q_cur, beta_prev, alphas, betas, _frozen, k = carry
        w = hvp(q_cur)
        alpha = _dot(q_cur, w)
        w = _axpy(-alpha, q_cur, w)
        w = _axpy(-beta_prev, q_prev, w)
        beta = jnp.sqrt(_dot(w, w))
        alphas = alphas.at[j].set(alpha)
        scale_ref = jnp.maximum(
            jnp.max(jnp.abs(alphas)), jnp.maximum(jnp.max(betas), beta)
        )
        breakdown = beta <= tol * scale_ref
        beta = jnp.where(breakdown, jnp.zeros((), dtype), beta)
        betas = betas.at[j].set(beta)
        q_next = _scale(1.0 / jnp.where(breakdown, jnp.ones((), dtype), beta), w)
        return (q_cur, q_next, beta, alphas, betas, breakdown, k + 1)

    def frozen(j, carry):
        q_prev, q_cur, beta_prev, alphas, betas, frozen_flag, k = carry
        alphas = alphas.at[j].set(alphas[0])
        return (q_prev, q_cur, beta_prev, alphas, betas, frozen_flag, k)

    def body(j, carry):
        return lax.cond(carry[5], frozen, active, j, carry)

    _, _, _, alphas, betas, frozen_flag, k = lax.fori_loop(0, iters, body, carry)

    off = betas[:-1]
    tridiagonal = jnp.diag(alphas) + jnp.diag(off, k=1) + jnp.diag(off, k=-1)
    theta, vectors = jnp.linalg.eigh(tridiagonal)
    weights = vectors[0, :] ** 2
    residuals = betas[-1] * jnp.abs(vectors[-1, :])
    # Extremes over the modes that carry gradient weight: a Ritz value with
    # weight at or below eps is rounding noise (the weight floor of the
    # module docstring). The start vector itself always exceeds it, so the
    # mask is never empty.
    carries = weights > jnp.finfo(dtype).eps
    i_max = jnp.argmax(jnp.where(carries, theta, -jnp.inf))
    i_min = jnp.argmin(jnp.where(carries, theta, jnp.inf))
    return LanczosResult(
        ritz_values=theta,
        ritz_weights=weights,
        lambda_max=theta[i_max],
        lambda_min=theta[i_min],
        residual_min=residuals[i_min],
        residual_max=residuals[i_max],
        breakdown=frozen_flag,
        k=k,
    )


# -----------------------------------------------------------------------------
# The spectrum through the solver's ε-energy
# -----------------------------------------------------------------------------


def make_epsilon_spectrum(
    structure: GraphStructure, iters: int = 30
) -> Callable[[GraphParams, GraphState, Mapping[str, Any], jax.Array], EpsilonSpectrum]:
    """Compile ``(params, state, clamps, key) -> EpsilonSpectrum`` for one graph.

    The state is passed through ``EPCInference.begin_segment`` so the Hessian
    is evaluated at the state's latents; ``EPCInference.error_energy`` gives
    the ε-energy, ``jax.grad`` of it the start vector g0, and ``jax.jvp`` of
    that gradient the Hessian-vector product. When g0 is zero (no energy at
    the state, for instance an unclamped target at the feedforward point) a
    random start vector drawn from ``key`` replaces it and
    ``random_start`` is set; the extremes then describe the full spectrum.
    The returned callable is reused across calls, so a probe during training
    pays one compile.
    """
    from fabricpc.core.inference_epc import EPCInference

    def run(params, state, clamps, key):
        synced = EPCInference.begin_segment(params, state, clamps, structure)
        energy_of, errors = EPCInference.error_energy(params, synced, clamps, structure)
        grad_fn = jax.grad(lambda e: energy_of(e)[0])

        def hvp(v):
            return jax.jvp(grad_fn, (errors,), (v,))[1]

        g0 = grad_fn(errors)
        gradient_norm = jnp.sqrt(_dot(g0, g0))
        random_start = gradient_norm == 0
        v0 = lax.cond(random_start, lambda: _random_like(key, g0), lambda: g0)
        result = lanczos_extremes(hvp, v0, iters)
        theta, weights = result.ritz_values, result.ritz_weights
        return EpsilonSpectrum(
            lambda_max=result.lambda_max,
            lambda_min=result.lambda_min,
            ritz_values=theta,
            ritz_weights=weights,
            residual_max=result.residual_max,
            residual_min=result.residual_min,
            negative_weight=jnp.sum(jnp.where(theta < 0, weights, 0.0)),
            gradient_norm=gradient_norm,
            random_start=random_start,
            iters=jnp.asarray(iters, jnp.int32),
            k=result.k,
        )

    return jax.jit(run)


def epsilon_spectrum(
    params: GraphParams,
    state: GraphState,
    clamps: Mapping[str, Any],
    structure: GraphStructure,
    iters: int = 30,
    key: Optional[jax.Array] = None,
) -> EpsilonSpectrum:
    """The excited spectrum of H_ε at the state's latents, on the host (one
    compile per call; use :func:`make_epsilon_spectrum` for repeated probes).
    ``EPCInference.regime(spectrum)`` turns it into a verdict on the solver's
    ``eta_infer`` and ``infer_steps``."""
    key = jax.random.PRNGKey(0) if key is None else key
    return make_epsilon_spectrum(structure, iters)(params, state, clamps, key).host()
