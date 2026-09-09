"""
Tests for ``fabricpc.core.epsilon_spectrum``: the Lanczos estimate of the
curvature ePC's starting gradient excites.

``lanczos_extremes`` is pinned on an explicit matrix (Ritz values, gradient
weights, the breakdown guard, frozen steps). ``epsilon_spectrum`` is pinned
against a dense ``jax.hessian`` of ``EPCInference.error_energy`` on a tanh
MLP (positive-definite and indefinite weight scales) and on a gelu MLP with
softmax cross-entropy at weight std 2.0, where the ε-Hessian is indefinite
at init. The linear-oracle comparison lives in ``test_linear_pc_oracle.py``.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from conftest import inject_biases
from fabricpc.core import EPCInference
from fabricpc.core.activations import (
    GeluActivation,
    IdentityActivation,
    SoftmaxActivation,
    TanhActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.epsilon_spectrum import (
    EpsilonSpectrum,
    epsilon_spectrum,
    lanczos_extremes,
    make_epsilon_spectrum,
    weighted_relaxed_fraction,
)
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode

# =============================================================================
# Fixtures
# =============================================================================


EIGS = np.array([-50.0, -1.0, 1.0, 3.0, 10.0])


def _rotated(eigs, seed=0):
    """H = Q diag(eigs) Qᵀ with a random orthogonal Q; returns (H, Q)."""
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(len(eigs), len(eigs))))
    return Q @ np.diag(eigs) @ Q.T, Q


def _x64(enabled):
    jax.config.update("jax_enable_x64", enabled)


@pytest.fixture
def float64():
    _x64(True)
    try:
        yield
    finally:
        _x64(False)


def _tanh_mlp(output, std, batch, key):
    """x(4) -> h1(3, tanh) -> h2(3, tanh) -> y(2); biases drawn."""
    w_init = NormalInitializer(std=std)
    x = IdentityNode(shape=(4,), name="x")
    h1 = Linear(shape=(3,), name="h1", activation=TanhActivation(), weight_init=w_init)
    h2 = Linear(shape=(3,), name="h2", activation=TanhActivation(), weight_init=w_init)
    if output == "gaussian":
        y = Linear(
            shape=(2,), name="y", activation=IdentityActivation(), weight_init=w_init
        )
    else:
        y = Linear(
            shape=(2,),
            name="y",
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            weight_init=w_init,
        )
    structure = graph(
        nodes=[x, h1, h2, y],
        edges=[
            Edge(source=x, target=h1.slot("in")),
            Edge(source=h1, target=h2.slot("in")),
            Edge(source=h2, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=EPCInference(),
    )
    params = inject_biases(
        initialize_params(structure, key), jax.random.fold_in(key, 7)
    )
    xs = jax.random.normal(key, (batch, 4))
    if output == "gaussian":
        ys = jax.random.normal(jax.random.PRNGKey(1), (batch, 2))
    else:
        ys = jax.nn.one_hot(
            jax.random.randint(jax.random.PRNGKey(1), (batch,), 0, 2), 2
        )
    clamps = {"x": xs, "y": ys}
    state = initialize_graph_state(structure, batch, key, clamps, params=params)
    return structure, params, clamps, state


def _gelu_mlp(d_in, width, depth, d_out, std, batch, key):
    """x -> depth × Linear(width, gelu) -> y(softmax + cross-entropy), weights
    N(0, std²/fan_in): at std 2.0 the ε-Hessian is indefinite at init."""
    x = IdentityNode(shape=(d_in,), name="x")
    nodes = [x]
    fan_in = d_in
    for i in range(depth):
        nodes.append(
            Linear(
                shape=(width,),
                name=f"h{i + 1}",
                activation=GeluActivation(),
                weight_init=NormalInitializer(std=std / math.sqrt(fan_in)),
            )
        )
        fan_in = width
    nodes.append(
        Linear(
            shape=(d_out,),
            name="y",
            activation=SoftmaxActivation(),
            energy=CrossEntropyEnergy(),
            weight_init=NormalInitializer(std=std / math.sqrt(fan_in)),
        )
    )
    edges = [Edge(source=a, target=b.slot("in")) for a, b in zip(nodes[:-1], nodes[1:])]
    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=x, y=nodes[-1]),
        inference=EPCInference(),
    )
    params = initialize_params(structure, jax.random.fold_in(key, 9))
    kx, ky = jax.random.split(jax.random.fold_in(key, 10))
    clamps = {
        "x": jax.random.normal(kx, (batch, d_in)),
        "y": jax.nn.one_hot(jax.random.randint(ky, (batch,), 0, d_out), d_out),
    }
    state = initialize_graph_state(structure, batch, key, clamps, params=params)
    return structure, params, clamps, state


def _dense_spectrum(structure, params, clamps, state):
    """Eigenvalues of the dense ε-Hessian (``jax.hessian`` of ``error_energy``
    at the state's latents) and the fraction of ‖g0‖² on each eigenvector."""
    synced = EPCInference.begin_segment(params, state, clamps, structure)
    energy_of, errors = EPCInference.error_energy(params, synced, clamps, structure)
    leaves, treedef = jax.tree_util.tree_flatten(errors)
    sizes = [leaf.size for leaf in leaves]

    def unflatten(v):
        out, start = [], 0
        for leaf, size in zip(leaves, sizes):
            out.append(v[start : start + size].reshape(leaf.shape))
            start += size
        return treedef.unflatten(out)

    def energy_flat(v):
        return energy_of(unflatten(v))[0]

    v0 = jnp.concatenate([jnp.ravel(leaf) for leaf in leaves])
    H = np.asarray(jax.hessian(energy_flat)(v0), dtype=np.float64)
    g0 = np.asarray(jax.grad(energy_flat)(v0), dtype=np.float64)
    eigs, vecs = np.linalg.eigh(0.5 * (H + H.T))
    weights = (vecs.T @ g0) ** 2
    return eigs, weights / weights.sum()


def _exact_fbar(eigs, weights, eta, steps):
    positive = eigs > 0
    f = 1.0 - (1.0 - eta * eigs[positive]) ** steps
    return float((weights[positive] * f).sum() / weights[positive].sum())


# =============================================================================
# lanczos_extremes on an explicit matrix
# =============================================================================


class TestLanczosExplicitMatrix:
    def test_extremes_and_weights_at_exact_krylov_dimension(self):
        """Five distinct eigenvalues, a start vector with all five components:
        five float32 steps reproduce the spectrum and the squared normalized
        components as Ritz weights; negative_weight is the weight on the two
        negative eigenvalues."""
        H, Q = _rotated(EIGS)
        components = np.array([0.5, 0.2, 0.1, 0.3, 0.7])
        v0 = Q @ components
        Hj = jnp.asarray(H, jnp.float32)
        result = lanczos_extremes(lambda v: Hj @ v, jnp.asarray(v0, jnp.float32), 5)
        np.testing.assert_allclose(result.ritz_values, EIGS, rtol=1e-4, atol=1e-4)
        expected_w = components**2 / np.sum(components**2)
        np.testing.assert_allclose(result.ritz_weights, expected_w, atol=1e-4)
        theta, w = np.asarray(result.ritz_values), np.asarray(result.ritz_weights)
        assert w[theta < 0].sum() == pytest.approx(expected_w[:2].sum(), abs=1e-4)
        assert int(result.k) == 5
        assert float(result.residual_max) < 1e-2 and float(result.residual_min) < 1e-2

    def test_breakdown_freezes_with_zero_weight(self, float64):
        """Past the Krylov dimension the guard fires: k stays at 5, the frozen
        steps carry the Rayleigh quotient of v0 with zero weight, and the
        extremes and the weights on the true eigenvalues are unchanged."""
        H, Q = _rotated(EIGS)
        components = np.array([0.5, 0.2, 0.1, 0.3, 0.7])
        v0 = Q @ components
        Hj = jnp.asarray(H, jnp.float64)
        result = lanczos_extremes(lambda v: Hj @ v, jnp.asarray(v0, jnp.float64), 8)
        assert bool(result.breakdown) and int(result.k) == 5
        theta, w = np.asarray(result.ritz_values), np.asarray(result.ritz_weights)
        rayleigh = float(v0 @ H @ v0 / (v0 @ v0))
        frozen = np.isclose(theta, rayleigh) & (w < 1e-12)
        assert frozen.sum() == 3
        np.testing.assert_allclose(np.sort(theta[~frozen]), EIGS, rtol=1e-10)
        expected_w = components**2 / np.sum(components**2)
        np.testing.assert_allclose(np.sort(w[~frozen]), np.sort(expected_w), atol=1e-12)
        assert theta[-1] == pytest.approx(10.0) and theta[0] == pytest.approx(-50.0)
        assert float(result.residual_max) == 0.0 and float(result.residual_min) == 0.0

    def test_invariant_subspace_breaks_down_at_step_two(self):
        """A start vector inside the span of two eigenvectors exhausts the
        Krylov space at step 2 in float32."""
        H, Q = _rotated(EIGS)
        v0 = Q @ np.array([0.0, 0.0, 0.6, 0.0, 0.8])  # eigenvalues 1 and 10
        Hj = jnp.asarray(H, jnp.float32)
        result = lanczos_extremes(lambda v: Hj @ v, jnp.asarray(v0, jnp.float32), 6)
        assert bool(result.breakdown) and int(result.k) == 2
        theta, w = np.asarray(result.ritz_values), np.asarray(result.ritz_weights)
        assert theta[0] == pytest.approx(1.0, abs=1e-4)
        assert theta[-1] == pytest.approx(10.0, abs=1e-4)
        assert w[0] == pytest.approx(0.36, abs=1e-5) and w[-1] == pytest.approx(
            0.64, abs=1e-5
        )
        assert np.all(w[1:-1] == 0.0)

    def test_rejects_zero_iters(self):
        with pytest.raises(ValueError):
            lanczos_extremes(lambda v: v, jnp.ones(3), 0)

    def test_weight_floor_hides_the_unexcited_floor(self):
        """H = I + JᵀJ with J of rank 10 on 40 coordinates, started at
        g0 = Jᵀr: the excited spectrum is eig(S), S = I + JJᵀ, and the 30
        unexcited coordinates sit on the floor λ = 1. Ten excited modes in
        float32 accumulate enough rounding over ten steps that the breakdown
        guard may not fire and the floor appears as a Ritz value of weight
        far below eps; the extremes must still be eig(S)'s."""
        rng = np.random.default_rng(1)
        J = rng.normal(size=(10, 40)) / np.sqrt(40) * 3.0
        H = np.eye(40) + J.T @ J
        r = rng.normal(size=10)
        g0 = J.T @ r
        eig_s = np.linalg.eigvalsh(np.eye(10) + J @ J.T)
        Hj = jnp.asarray(H, jnp.float32)
        result = lanczos_extremes(lambda v: Hj @ v, jnp.asarray(g0, jnp.float32), 30)
        assert float(result.lambda_max) == pytest.approx(eig_s[-1], rel=1e-4)
        assert float(result.lambda_min) == pytest.approx(eig_s[0], rel=1e-3)
        assert eig_s[0] > 1.05  # the floor is separated from the excited minimum
        theta, w = np.asarray(result.ritz_values), np.asarray(result.ritz_weights)
        carrying = w > np.finfo(np.float32).eps
        assert theta[carrying].min() == pytest.approx(eig_s[0], rel=1e-3)
        assert w[carrying].sum() == pytest.approx(1.0, abs=1e-5)


# =============================================================================
# EpsilonSpectrum helpers
# =============================================================================


class TestSpectrumHelpers:
    def test_from_modes_normalizes_and_sorts(self):
        s = EpsilonSpectrum.from_modes([3.0, -1.0, 2.0], [2.0, 1.0, 1.0])
        np.testing.assert_allclose(s.ritz_values, [-1.0, 2.0, 3.0])
        np.testing.assert_allclose(s.ritz_weights, [0.25, 0.25, 0.5])
        assert s.lambda_max == 3.0 and s.lambda_min == -1.0
        assert s.negative_weight == 0.25 and s.k == 3
        with pytest.raises(ValueError):
            EpsilonSpectrum.from_modes([1.0], [1.0, 2.0])

    def test_weighted_relaxed_fraction_positive_modes_only(self):
        s = EpsilonSpectrum.from_modes([-4.0, 1.0, 4.0], [0.5, 0.25, 0.25])
        f1, f4 = 1 - 0.9**3, 1 - 0.6**3
        assert weighted_relaxed_fraction(s, 0.1, 3) == pytest.approx(0.5 * (f1 + f4))
        assert math.isnan(
            weighted_relaxed_fraction(EpsilonSpectrum.from_modes([-1.0], [1.0]), 0.1, 3)
        )


# =============================================================================
# epsilon_spectrum through the solver's ε-energy
# =============================================================================


class TestEpsilonSpectrumOnGraphs:
    def test_zero_gradient_uses_random_start(self, rng_key):
        """With the target unclamped the feedforward state has zero energy and
        g0 = 0; the estimator falls back to a random start and reports it.
        In ε coordinates that energy is ½Σ‖ε‖², so every eigenvalue is 1."""
        x = IdentityNode(shape=(4,), name="x")
        h = Linear(shape=(3,), name="h", activation=IdentityActivation())
        y = Linear(shape=(2,), name="y", activation=IdentityActivation())
        structure = graph(
            nodes=[x, h, y],
            edges=[
                Edge(source=x, target=h.slot("in")),
                Edge(source=h, target=y.slot("in")),
            ],
            task_map=TaskMap(x=x, y=y),
            inference=EPCInference(),
        )
        params = initialize_params(structure, rng_key)
        clamps = {"x": jax.random.normal(rng_key, (3, 4))}
        state = initialize_graph_state(structure, 3, rng_key, clamps, params=params)
        spectrum = epsilon_spectrum(
            params, state, clamps, structure, iters=10, key=rng_key
        )
        assert spectrum.random_start and spectrum.gradient_norm == 0.0
        assert spectrum.lambda_max == pytest.approx(1.0, abs=1e-5)
        assert spectrum.lambda_min == pytest.approx(1.0, abs=1e-5)

    @pytest.mark.parametrize("output", ["gaussian", "ce"])
    @pytest.mark.parametrize("std", [0.3, 1.5])
    def test_tanh_mlp_matches_dense_hessian(self, rng_key, output, std):
        """Extremes of the excited spectrum, f̄, and the negative weight from
        30 float32 Lanczos steps against the dense ε-Hessian (18 entries).
        At std 1.5 the Hessian is indefinite."""
        structure, params, clamps, state = _tanh_mlp(output, std, 3, rng_key)
        eigs, weights = _dense_spectrum(structure, params, clamps, state)
        spectrum = epsilon_spectrum(
            params, state, clamps, structure, iters=30, key=rng_key
        )
        excited = eigs[weights > 1e-10]
        np.testing.assert_allclose(spectrum.lambda_max, excited[-1], rtol=1e-3)
        np.testing.assert_allclose(spectrum.lambda_min, excited[0], rtol=1e-3)
        eta, steps = 0.05, 5
        np.testing.assert_allclose(
            weighted_relaxed_fraction(spectrum, eta, steps),
            _exact_fbar(eigs, weights, eta, steps),
            atol=1e-3,
        )
        np.testing.assert_allclose(
            spectrum.negative_weight, weights[eigs < 0].sum(), atol=1e-3
        )
        assert (spectrum.lambda_min < 0) == (std == 1.5)
        assert spectrum.gradient_norm > 0 and not spectrum.random_start

    def test_gelu_mlp_at_std_two_is_indefinite(self, rng_key):
        """x8 -> 3 × h16 (gelu) -> y4 (softmax + cross-entropy), batch 2, 96 ε
        entries: the loss-gradient-weighted second derivatives of the network
        map make H_ε indefinite at init. λ_min < 0 and the gradient weight on
        the negative modes agree with the dense Hessian."""
        structure, params, clamps, state = _gelu_mlp(8, 16, 3, 4, 2.0, 2, rng_key)
        eigs, weights = _dense_spectrum(structure, params, clamps, state)
        assert eigs[0] < 0 and weights[eigs < 0].sum() > 0.05
        spectrum = epsilon_spectrum(
            params, state, clamps, structure, iters=30, key=rng_key
        )
        assert spectrum.lambda_min < 0
        np.testing.assert_allclose(spectrum.lambda_min, eigs[0], rtol=1e-3)
        np.testing.assert_allclose(spectrum.lambda_max, eigs[-1], rtol=1e-3)
        np.testing.assert_allclose(
            spectrum.negative_weight, weights[eigs < 0].sum(), atol=1e-3
        )
        np.testing.assert_allclose(
            weighted_relaxed_fraction(spectrum, 0.03, 10),
            _exact_fbar(eigs, weights, 0.03, 10),
            atol=1e-3,
        )
        regime = EPCInference(eta_infer=0.03, infer_steps=10).regime(spectrum)
        assert regime.negative_weight > 0.05 and regime.growth_min > 1.1
        assert str(regime).startswith("indefinite")

    def test_compiled_probe_is_reusable_and_matches_convenience(self, rng_key):
        structure, params, clamps, state = _tanh_mlp("ce", 0.3, 3, rng_key)
        compiled = make_epsilon_spectrum(structure, iters=12)
        a = compiled(params, state, clamps, rng_key).host()
        b = compiled(params, state, clamps, rng_key).host()
        c = epsilon_spectrum(params, state, clamps, structure, iters=12, key=rng_key)
        assert a.lambda_max == b.lambda_max == pytest.approx(c.lambda_max)
        assert a.iters == 12 and isinstance(a.lambda_max, float)
        assert isinstance(a.ritz_values, np.ndarray) and a.ritz_values.shape == (12,)
        assert a.ritz_weights.sum() == pytest.approx(1.0, abs=1e-5)
