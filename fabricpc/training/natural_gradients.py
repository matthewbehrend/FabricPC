"""Natural-gradient-style optimizer transforms for predictive coding training.

Both transforms precondition the gradient with an online diagonal Fisher
approximation, an exponential moving average (EMA) of the squared gradient,
and compose with Optax chains, typically followed by ``optax.scale(-lr)``.

Gradient scale. The trainer hands Optax mean gradients per prediction
(``fabricpc.training.pc_weight_gradients`` divides the batch-summed gradients
once by the prediction count), so the gradients reaching these transforms sit
on the scale of standard mean-loss training.

Covariance. The natural-gradient update ``g / f`` is covariant: scaling every
gradient by ``c`` scales ``f`` by ``c**2`` and the update by ``1/c``. Damping
is therefore expressed relative to the Fisher itself. The damping reference is
``r = trace(F) / dim``, the mean bias-corrected Fisher entry over the whole
parameter pytree, and the update is
``g / (f + relative_damping * r + damping)``. With ``damping = 0`` (the
default) both denominator terms scale by ``c**2``, so the update scales by
exactly ``1/c`` and the set of entries where damping dominates the Fisher does
not depend on the gradient scale. No constant is added in that case: a zero
denominator occurs only when every gradient seen so far is zero, and then the
update is zero.

Absolute damping. ``damping > 0`` adds a constant to the denominator. It
breaks the covariance above and ties the transform to one gradient scale:
wherever ``damping`` dominates ``f + relative_damping * r``, the update is
``g / damping``, plain SGD with learning rate ``scale / damping``. It exists
as an explicit opt-in because of the limitation below.

Bias correction. The EMA starts at zero, so after ``t`` steps it holds only
``1 - fisher_decay**t`` of a stationary ``g**2``. Both transforms divide by
that factor (``t`` is the step count held in the state), so the first steps
are not over-scaled.

Limitations of the present implementations. ``F`` is built from the squared
*mean* gradient of the batch, not from per-sample gradients, so ``g / F`` is
about ``1 / g``: the entries with the largest gradients receive the smallest
steps, and the step size grows relative to the gradient as training reduces
it. On ``examples/mnist_advanced.py`` (a four-layer sigmoid MLP) neither
transform left chance accuracy in 10 epochs with relative damping alone, at
any of 48 combinations of ``optax.scale``, ``relative_damping``, and a global
norm clip, while ``optax.adamw`` reaches 97% on the same graph. The demo
presets that do train (weakly: 24% at 10 epochs) use ``relative_damping = 0``
with an absolute ``damping`` that, measured on the diagonal transform, lies
above 96.5% of the Fisher entries from the first step and above 99.7% by
step 300: those entries are updated as SGD with rate ``scale / damping``,
while the few large-gradient entries that hold the Fisher trace receive the
``1 / g`` step. Treat these transforms as research baselines, not as tuned
optimizers; the measurements are recorded in
``docs/dev_plans_archive/mean_gradient_normalization.md``.
"""

import operator
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax


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
    relative_damping: float = 0.1,
    damping: float = 0.0,
) -> optax.GradientTransformation:
    """Precondition updates with a bias-corrected EMA diagonal Fisher.

    Each entry is divided by ``f + relative_damping * r + damping``, where
    ``f`` is the bias-corrected EMA of ``g**2`` for that entry and ``r`` is
    the mean of ``f`` over the whole parameter pytree. With ``damping = 0``,
    scaling the gradients by ``c`` scales the update by ``1/c``.

    Args:
        fisher_decay: EMA decay for the Fisher estimate in [0, 1).
        relative_damping: Damping as a fraction of the mean Fisher entry,
            >= 0. Entries whose Fisher is well below ``relative_damping * r``
            are updated as by SGD with step ``1 / (relative_damping * r)``.
        damping: Absolute damping added to the denominator, >= 0. Where it
            dominates, the update is ``g / damping`` (SGD with rate
            ``scale / damping``); it ties the transform to one gradient
            scale. At least one of ``relative_damping`` and ``damping`` must
            be positive.

    Returns:
        Optax gradient transformation.
    """
    _validate_hparams(fisher_decay, relative_damping, damping)
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
        leaf_traces = jax.tree_util.tree_map(jnp.sum, fisher_hat)
        total_damping = (
            relative_damping * _mean_fisher_entry(leaf_traces, updates) + damping
        )
        preconditioned_updates = jax.tree_util.tree_map(
            lambda g, f: _divide_guarded(g, f + total_damping.astype(f.dtype)),
            updates,
            fisher_hat,
        )
        return preconditioned_updates, DiagonalNaturalGradientState(
            count=count, fisher_diag=fisher_diag
        )

    return optax.GradientTransformation(init_fn, update_fn)


def scale_by_natural_gradient_layerwise(
    fisher_decay: float = 0.95,
    relative_damping: float = 0.1,
    damping: float = 0.0,
) -> optax.GradientTransformation:
    """Precondition each tensor by one bias-corrected scalar Fisher per leaf.

    A cheap layer-wise approximation: each leaf's scalar is the EMA of
    ``mean(g**2)`` over the leaf. The leaf is divided by
    ``f + relative_damping * r + damping`` with ``r`` the mean Fisher entry
    over the whole pytree, where each leaf's scalar counts with weight equal
    to the leaf's size (the same ``trace(F) / dim`` the diagonal transform
    uses). With ``damping = 0``, scaling the gradients by ``c`` scales the
    update by ``1/c``.

    Args:
        fisher_decay: EMA decay for the Fisher estimate in [0, 1).
        relative_damping: Damping as a fraction of the mean Fisher entry, >= 0.
        damping: Absolute damping added to the denominator, >= 0; where it
            dominates, the update is ``g / damping``. At least one of
            ``relative_damping`` and ``damping`` must be positive.

    Returns:
        Optax gradient transformation.
    """
    _validate_hparams(fisher_decay, relative_damping, damping)
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
        # Each leaf scalar stands for g.size Fisher entries.
        leaf_traces = jax.tree_util.tree_map(
            lambda f, g: f * g.size, fisher_hat, updates
        )
        total_damping = (
            relative_damping * _mean_fisher_entry(leaf_traces, updates) + damping
        )
        preconditioned_updates = jax.tree_util.tree_map(
            lambda g, f: _divide_guarded(g, f + total_damping.astype(f.dtype)),
            updates,
            fisher_hat,
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


def _mean_fisher_entry(leaf_traces, updates):
    """``trace(F) / dim``: the mean Fisher entry over the whole pytree.

    ``leaf_traces`` holds one scalar per leaf, the sum of that leaf's Fisher
    entries. ``dim`` is the static parameter count read from ``updates``.
    """
    trace = jax.tree_util.tree_reduce(operator.add, leaf_traces)
    dim = sum(g.size for g in jax.tree_util.tree_leaves(updates))
    return trace / dim


def _divide_guarded(g, denominator):
    """``g / denominator`` with a zero denominator mapped to a zero update.

    With ``damping = 0``, ``f + relative_damping * r`` is zero only when
    every Fisher entry in the pytree is zero, which requires every gradient
    seen so far, including this step's, to be zero. Then ``g`` is zero too,
    and dividing by 1 instead gives the correct update without forming
    ``0 / 0`` and without adding any constant to the denominator.
    """
    return g / jnp.where(denominator > 0, denominator, 1.0)


def _validate_hparams(
    fisher_decay: float, relative_damping: float, damping: float
) -> None:
    """Validate natural-gradient hyperparameters."""
    if not 0.0 <= fisher_decay < 1.0:
        raise ValueError(f"fisher_decay must be in [0, 1). got {fisher_decay}")
    if relative_damping < 0.0:
        raise ValueError(f"relative_damping must be >= 0. got {relative_damping}")
    if damping < 0.0:
        raise ValueError(f"damping must be >= 0. got {damping}")
    if relative_damping == 0.0 and damping == 0.0:
        raise ValueError(
            "relative_damping and damping are both 0: at least one must be > 0, "
            "or the update g / F has no bound as gradients shrink."
        )
