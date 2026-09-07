"""Natural-gradient-style optimizer transforms for predictive coding training.

Both transforms divide the gradient by an online diagonal Fisher estimate, an
exponential moving average (EMA) of the squared gradient, and compose with
Optax chains, typically followed by ``optax.scale(-lr)``. They are research
baselines, not tuned optimizers; see the regimes below and
https://github.com/trueagi-io/FabricPC/issues/68.

Gradient scale. The trainer hands Optax mean gradients per prediction
(``fabricpc.training.pc_weight_gradients`` divides the batch-summed gradients
once by the prediction count), so ``damping`` is compared against Fisher
entries on that scale: per-prediction gradients of g per weight give
Fisher entries of g**2.

Regimes. The update is ``g / (f + damping)``, with ``f`` the bias-corrected
EMA of ``g**2``. Where ``damping`` dominates ``f`` the update is
``g / damping``: plain SGD with learning rate ``scale / damping``. Where ``f``
dominates, ``f`` is about ``g**2`` because it is built from the squared *mean*
gradient of the batch rather than from per-sample gradients, so the update is
about ``1 / g``: the entries with the largest gradients move least, and the
step grows relative to the gradient as training shrinks it. Neither regime is
a natural-gradient step, and no damping value produces one: a smaller value
moves more entries into the ``1 / g`` regime, a larger one into SGD. The
default ``damping`` is the value that trained best on
``examples/mnist_advanced.py``; at that value the damping term exceeds almost
every Fisher entry from the first step (measured fraction in the demo's preset
comment), so the transforms act as SGD on almost every parameter. Choosing
``damping`` chooses the regime. The defect is the estimator: a Fisher needs
per-sample gradients at latents drawn from each node's predictive
distribution, not the squared batch mean. That design is specified in issue 68.

Bias correction. The EMA starts at zero, so after ``t`` steps it holds only
``1 - fisher_decay**t`` of a stationary ``g**2``. Both transforms divide by
that factor (``t`` is the step count held in the state), so the first steps
are not over-scaled.
"""

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax

DEFAULT_DAMPING = 1e-8


class DiagonalNaturalGradientState(NamedTuple):
    """State for diagonal natural-gradient preconditioning.

    ``count`` is the int32 number of updates applied so far; ``fisher_diag``
    is the uncorrected EMA of ``g**2`` with the parameters' structure.
    """

    count: jax.Array
    fisher_diag: Any


class LayerwiseNaturalGradientState(NamedTuple):
    """State for layer-wise natural-gradient preconditioning.

    ``count`` is the int32 number of updates applied so far;
    ``fisher_scalar`` holds one scalar per leaf, the uncorrected EMA of
    ``mean(g**2)`` over that leaf.
    """

    count: jax.Array
    fisher_scalar: Any


def scale_by_natural_gradient_diag(
    fisher_decay: float = 0.95,
    damping: float = DEFAULT_DAMPING,
) -> optax.GradientTransformation:
    """Precondition updates with a bias-corrected EMA diagonal Fisher.

    Each entry is divided by ``f + damping``, where ``f`` is the
    bias-corrected EMA of ``g**2`` for that entry.

    Args:
        fisher_decay: EMA decay for the Fisher estimate in [0, 1).
        damping: Positive constant added to every Fisher entry. Where it
            dominates ``f`` the update is ``g / damping``, SGD with rate
            ``scale / damping``; where ``f`` dominates the update is about
            ``1 / g``. See the module docstring.

    Returns:
        Optax gradient transformation.
    """
    _validate_hparams(fisher_decay, damping)
    one_minus_decay = 1.0 - fisher_decay

    def init_fn(params):
        return DiagonalNaturalGradientState(
            count=jnp.zeros((), dtype=jnp.int32),
            fisher_diag=jax.tree_util.tree_map(jnp.zeros_like, params),
        )

    def update_fn(updates, state, params=None):
        del params
        count = optax.safe_int32_increment(state.count)
        fisher_diag = jax.tree_util.tree_map(
            lambda f, g: fisher_decay * f + one_minus_decay * jnp.square(g),
            state.fisher_diag,
            updates,
        )
        fisher_hat = _bias_corrected(fisher_diag, fisher_decay, count)
        preconditioned_updates = jax.tree_util.tree_map(
            lambda g, f: g / (f + damping), updates, fisher_hat
        )
        return preconditioned_updates, DiagonalNaturalGradientState(
            count=count, fisher_diag=fisher_diag
        )

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_natural_gradient_layerwise(
    fisher_decay: float = 0.95,
    damping: float = DEFAULT_DAMPING,
) -> optax.GradientTransformation:
    """Precondition each tensor by one bias-corrected scalar Fisher per leaf.

    A cheap layer-wise approximation: each leaf's scalar is the EMA of
    ``mean(g**2)`` over the leaf, and the leaf is divided by
    ``f + damping``.

    Args:
        fisher_decay: EMA decay for the Fisher estimate in [0, 1).
        damping: Positive constant added to each leaf's Fisher scalar. The
            regimes are those of :func:`scale_by_natural_gradient_diag`.

    Returns:
        Optax gradient transformation.
    """
    _validate_hparams(fisher_decay, damping)
    one_minus_decay = 1.0 - fisher_decay

    def init_fn(params):
        return LayerwiseNaturalGradientState(
            count=jnp.zeros((), dtype=jnp.int32),
            fisher_scalar=jax.tree_util.tree_map(
                lambda p: jnp.zeros((), dtype=p.dtype), params
            ),
        )

    def update_fn(updates, state, params=None):
        del params
        count = optax.safe_int32_increment(state.count)
        fisher_scalar = jax.tree_util.tree_map(
            lambda f, g: fisher_decay * f + one_minus_decay * jnp.mean(jnp.square(g)),
            state.fisher_scalar,
            updates,
        )
        fisher_hat = _bias_corrected(fisher_scalar, fisher_decay, count)
        preconditioned_updates = jax.tree_util.tree_map(
            lambda g, f: g / (f + damping), updates, fisher_hat
        )
        return preconditioned_updates, LayerwiseNaturalGradientState(
            count=count, fisher_scalar=fisher_scalar
        )

    return optax.GradientTransformation(init_fn, update_fn)


def _bias_corrected(fisher, fisher_decay: float, count: jax.Array):
    """Divide every Fisher leaf by ``1 - fisher_decay**count``.

    ``count`` is the incremented step count (1 after the first update), so
    the factor is positive. It is computed once as a float32 scalar and cast
    to each leaf's dtype.
    """
    factor = 1.0 - fisher_decay**count
    return jax.tree_util.tree_map(lambda f: f / factor.astype(f.dtype), fisher)


def _validate_hparams(fisher_decay: float, damping: float) -> None:
    """Validate natural-gradient hyperparameters."""
    if not 0.0 <= fisher_decay < 1.0:
        raise ValueError(f"fisher_decay must be in [0, 1). got {fisher_decay}")
    if damping <= 0.0:
        raise ValueError(f"damping must be > 0. got {damping}")
