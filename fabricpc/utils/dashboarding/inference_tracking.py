"""Modified inference loop that returns state history for tracking.

This module provides alternative inference and training functions that
collect intermediate states for detailed tracking and debugging.

Both history variants iterate ``structure.config["inference"].segments()``
inside one jitted program, so composed schedules (e.g. an ePC segment
followed by an sPC segment) are tracked segment by segment: each segment
runs its solver's ``begin_segment`` before its steps and ``finalize_state``
after them, as ``run_inference`` does. ``run_inference_with_history``
concatenates the per-step metric stacks along the step axis;
``make_inference_history`` samples states at global step multiples across
the segment boundaries. Metric semantics are per-segment:
``latent_grad_norm`` is the norm of whatever that segment's solver
accumulates into ``latent_grad`` — under state-based solvers the one-hop
dE/dz_latent, under ``EPCInference`` the full-forward gradient of the total
energy with respect to the relaxed errors.
"""

from typing import Callable, Dict, List, Tuple, cast
import jax
import jax.numpy as jnp
import optax

from fabricpc.core.types import (
    GraphParams,
    GraphState,
    GraphStructure,
)
from fabricpc.core.energy import graph_energy
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training.trainer import (
    batch_size_of,
    build_clamps,
    grad_denominator,
    pc_weight_gradients,
)


# TODO clarify collect_every refers to either batches or inference steps
def run_inference_with_history(
    params: GraphParams,
    initial_state: GraphState,
    clamps: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    collect_every: int = 1,
) -> Tuple[GraphState, List[Dict[str, Dict[str, jnp.ndarray]]]]:
    """Run inference and collect state history at specified intervals.

    This function uses jax.lax.scan instead of fori_loop to return
    intermediate states. The history is collected as lightweight
    dictionaries rather than full GraphState objects to manage memory.

    Note: This is more memory-intensive than run_inference. Use only
    when you need to track inference dynamics.

    Args:
        params: Model parameters.
        initial_state: Initial graph state.
        clamps: Dictionary of clamped values.
        structure: Graph structure.
        collect_every: Collect state every N steps (1 = every step).

    Returns:
        Tuple of (final_state, state_history):
            - final_state: GraphState after convergence
            - state_history: List of dicts containing key metrics per step
    """
    state = initial_state
    segment_metrics = []
    for solver, n_steps in structure.config["inference"].segments():
        solver_cls = type(solver)
        config = solver.config

        def scan_fn(
            state: GraphState, _: None
        ) -> Tuple[GraphState, Dict[str, Dict[str, jnp.ndarray]]]:
            new_state = solver_cls.inference_step(
                params, state, clamps, structure, config
            )
            # Extract key metrics for history (lightweight)
            # Reduce over batch dimension to get scalar metrics per step
            step_metrics = {
                node_name: {
                    "energy": jnp.mean(node_state.energy),
                    "latent_grad_norm": jnp.mean(
                        jnp.linalg.norm(node_state.latent_grad, axis=-1)
                    ),
                    "error_norm": jnp.mean(jnp.linalg.norm(node_state.error, axis=-1)),
                    "z_latent_mean": jnp.mean(node_state.z_latent),
                    "z_latent_std": jnp.mean(jnp.std(node_state.z_latent, axis=-1)),
                }
                for node_name, node_state in new_state.nodes.items()
            }
            return new_state, step_metrics

        state = solver_cls.begin_segment(params, state, clamps, structure)
        state, metrics = jax.lax.scan(scan_fn, state, xs=None, length=n_steps)
        state = solver_cls.finalize_state(params, state, clamps, structure)
        segment_metrics.append(metrics)

    # Concatenate per-segment stacks along the step axis. The metric
    # structure (node names, metric keys) is identical across segments.
    all_metrics = jax.tree_util.tree_map(
        lambda *xs: jnp.concatenate(xs, axis=0), *segment_metrics
    )

    # Return stacked metrics - unstacking must happen outside JIT
    # all_metrics is a nested dict with stacked arrays of shape (total_steps,)
    return state, all_metrics


def make_tracked_probe(
    structure: GraphStructure,
) -> Callable[
    [GraphParams, jax.Array, Dict[str, jnp.ndarray]],
    Tuple[GraphState, Dict[str, Dict[str, jnp.ndarray]]],
]:
    """Jitted ``probe(params, key, clamps) -> (final_state, stacked_metrics)``.

    The per-step-metrics twin of
    :func:`fabricpc.utils.dashboarding.callbacks.make_tracked_settle`: latent
    initialization from ``clamps`` and ``key`` (the batch size is the
    clamps' leading axis), then :func:`run_inference_with_history`, compiled
    into one XLA program. Splitting them — eager init, jitted tracking —
    breaks the feedforward-init invariant on GPU: at default matmul
    precision the two programs can select different cuDNN conv algorithms
    (TF32 vs FP32, per conv shape), so an unclamped node records the squared
    difference between the two conv paths (up to ~1e-3) as its step-0
    energy instead of 0.

    One compiled program per structure and clamp shape; call it with
    different ``params`` (training checkpoints, or ``ctx.params`` from
    ``train``'s iteration callback) to compare histories that differ only in
    the parameters. Build it once per structure.
    """

    def probe(params, key, clamps):
        batch_size = next(iter(clamps.values())).shape[0]
        init_state = initialize_graph_state(
            structure, batch_size, key, clamps=clamps, params=params
        )
        return run_inference_with_history(params, init_state, clamps, structure)

    return jax.jit(probe)


def _unstack_metrics(
    stacked_metrics: Dict[str, Dict[str, jnp.ndarray]],
    collect_every: int = 1,
) -> List[Dict[str, Dict[str, float]]]:
    """Convert stacked metrics from scan into list of per-step dicts.

    Args:
        stacked_metrics: Dict of node -> metric -> stacked array (num_steps, ...)
        collect_every: Subsample by taking every Nth step.

    Returns:
        List of dicts with per-step metrics.
    """
    # Get number of steps from any array
    sample_node = next(iter(stacked_metrics.keys()))
    sample_metric = next(iter(stacked_metrics[sample_node].keys()))
    num_steps = stacked_metrics[sample_node][sample_metric].shape[0]

    history = []
    for step in range(0, num_steps, collect_every):
        step_dict: Dict[str, Dict[str, float]] = {}
        for node_name, node_metrics in stacked_metrics.items():
            step_dict[node_name] = {
                metric_name: float(metric_arr[step])
                for metric_name, metric_arr in node_metrics.items()
            }
        history.append(step_dict)

    return history


def make_inference_history(
    structure: GraphStructure, *, every: int = 1
) -> Callable[
    [GraphParams, GraphState, Dict[str, jnp.ndarray]], Tuple[GraphState, GraphState]
]:
    """Build a jitted settle that also returns the states at inference steps
    ``0, every, 2*every, ...`` up to the total step count.

    Returns ``history(params, initial_state, clamps) -> (final_state,
    states)``. ``final_state`` is the settle after every segment of
    ``structure.config["inference"].segments()`` has run, the same state
    :func:`fabricpc.core.inference.run_inference` returns. ``states`` is a
    GraphState pytree whose every leaf carries a new leading axis of length
    ``total_steps // every + 1``, ``total_steps`` the sum of the segments'
    step counts: index ``i`` is the state after ``i * every`` inference
    steps, so index 0 is ``initial_state`` and, when ``total_steps`` is a
    multiple of ``every``, the last index is ``final_state``. Read step ``i``
    with ``jax.tree_util.tree_map(lambda a: a[i], states)``.

    Each segment runs its solver's ``begin_segment`` before its steps and
    ``finalize_state`` after them, and every sampled state is passed through
    the current segment's ``finalize_state`` too, so a sample is the state
    ``run_inference`` would return if the schedule stopped there: under
    ``EPCInference`` that is the derived state (latents and energies at the
    sampled errors, not one ε update behind), under the state-based solvers
    the identity. Sample points are counted across segment boundaries, so a
    schedule of 3 ePC steps then 5 sPC steps at ``every=2`` samples after
    steps 0, 2, 4, 6, 8.

    The loop is a ``lax.scan`` over blocks of ``every`` steps, each block a
    ``fori_loop``, with partial blocks at the segment boundaries, so the
    compiled program holds the sampled states only, not one per step, and
    the whole settle is one XLA program. Build once per structure and reuse
    the returned function; each factory call compiles anew on its first
    invocation.
    """
    if every < 1:
        raise ValueError(f"every must be >= 1, got {every}")
    segments = tuple(structure.config["inference"].segments())

    def history(
        params: GraphParams,
        initial_state: GraphState,
        clamps: Dict[str, jnp.ndarray],
    ) -> Tuple[GraphState, GraphState]:
        def stacked(state):
            return jax.tree_util.tree_map(lambda a: a[None], state)

        samples = [stacked(initial_state)]
        state = initial_state
        done = 0  # inference steps completed before the current segment
        for solver, n_steps in segments:
            solver_cls, config = type(solver), solver.config

            def step(_t, s, solver_cls=solver_cls, config=config):
                return solver_cls.inference_step(params, s, clamps, structure, config)

            def sample(s, solver_cls=solver_cls):
                return solver_cls.finalize_state(params, s, clamps, structure)

            state = solver_cls.begin_segment(params, state, clamps, structure)
            to_next = every - done % every  # steps to the next sample point
            if to_next <= n_steps:
                state = jax.lax.fori_loop(0, to_next, step, state)
                samples.append(stacked(sample(state)))
                n_full, tail = divmod(n_steps - to_next, every)
                if n_full:

                    def block(s, _, step=step, sample=sample):
                        s = jax.lax.fori_loop(0, every, step, s)
                        return s, sample(s)

                    state, sampled = jax.lax.scan(block, state, xs=None, length=n_full)
                    samples.append(sampled)
                state = jax.lax.fori_loop(0, tail, step, state)
            else:
                state = jax.lax.fori_loop(0, n_steps, step, state)
            state = solver_cls.finalize_state(params, state, clamps, structure)
            done += n_steps

        states = jax.tree_util.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0), *samples
        )
        return state, states

    return jax.jit(history)


def train_step_with_history(
    params: GraphParams,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    structure: GraphStructure,
    optimizer: optax.GradientTransformation,
    rng_key: jax.Array,
    collect_every: int = 1,
) -> Tuple[
    GraphParams,
    optax.OptState,
    jnp.ndarray,
    GraphState,
    Dict[str, Dict[str, jnp.ndarray]],
]:
    """PC training step that also returns inference history.

    Same step as the trainer's PC path (build_clamps -> initialize_graph_state
    -> inference -> graph_energy -> pc_weight_gradients -> optax), with
    run_inference swapped for run_inference_with_history. Use this when you
    need to track inference dynamics.

    Note: This function is designed to be JIT-compiled. The returned energy and
    inference_history are JAX arrays. Use unstack_inference_history() to convert
    the stacked metrics to a list of per-step dicts after the JIT call.

    Args:
        params: Current model parameters.
        opt_state: Optimizer state.
        batch: Batch of data with task-specific keys.
        structure: Graph structure.
        optimizer: Optax optimizer.
        rng_key: JAX random key for state initialization.
        collect_every: Collect history every N inference steps (note: currently
            ignored inside JIT; subsample after with unstack_inference_history).

    Returns:
        Tuple of (params, opt_state, energy, final_state, stacked_inference_history).
        ``energy`` is the training objective per prediction: ``graph_energy``
        over in_degree>0 nodes divided by the prediction count
        (``fabricpc.training.grad_denominator``).
        Call unstack_inference_history() on stacked_inference_history outside JIT.
    """
    batch_size = batch_size_of(batch, structure)

    clamps = build_clamps(batch, structure, clamp_target=True)
    denom = grad_denominator(structure, clamps)

    init_state = initialize_graph_state(
        structure,
        batch_size,
        rng_key,
        clamps=clamps,
        params=params,
    )

    # Run inference WITH history collection (returns stacked metrics)
    final_state, stacked_history = run_inference_with_history(
        params, init_state, clamps, structure, collect_every
    )

    energy = graph_energy(final_state, structure) / denom

    # Mean gradients per prediction (summed gradients / denom), as in train().
    grads = pc_weight_gradients(params, final_state, structure, clamps)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = cast(GraphParams, optax.apply_updates(params, updates))

    return params, opt_state, energy, final_state, stacked_history


def unstack_inference_history(
    stacked_metrics: Dict[str, Dict[str, jnp.ndarray]],
    collect_every: int = 1,
) -> List[Dict[str, Dict[str, float]]]:
    """Convert stacked metrics from JIT to list of per-step dicts.

    Call this function OUTSIDE of JIT on the stacked_inference_history
    returned by train_step_with_history.

    Args:
        stacked_metrics: Dict of node -> metric -> stacked array (num_steps,)
        collect_every: Subsample by taking every Nth step.

    Returns:
        List of dicts with per-step metrics as Python floats.
    """
    return _unstack_metrics(stacked_metrics, collect_every)


def extract_history_for_plotting(
    inference_history: List[Dict[str, Dict[str, float]]],
    node_name: str,
    metric_name: str = "energy",
) -> List[float]:
    """Extract a single metric series from inference history for plotting.

    Args:
        inference_history: History from run_inference_with_history.
        node_name: Name of the node.
        metric_name: Name of the metric to extract.

    Returns:
        List of metric values, one per inference step.
    """
    return [step[node_name][metric_name] for step in inference_history]


def summarize_inference_convergence(
    inference_history: List[Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    """Summarize inference convergence statistics.

    Args:
        inference_history: History from run_inference_with_history.

    Returns:
        Dict of node -> convergence metrics (final energy, energy reduction, etc.)
    """
    if not inference_history:
        return {}

    first_step = inference_history[0]
    last_step = inference_history[-1]

    summary = {}
    for node_name in first_step.keys():
        initial_energy = first_step[node_name].get("energy", 0.0)
        final_energy = last_step[node_name].get("energy", 0.0)
        initial_grad = first_step[node_name].get("latent_grad_norm", 0.0)
        final_grad = last_step[node_name].get("latent_grad_norm", 0.0)

        # Handle case where initial_energy is 0
        if initial_energy > 0:
            energy_reduction = (initial_energy - final_energy) / initial_energy
        else:
            energy_reduction = 0.0

        summary[node_name] = {
            "initial_energy": initial_energy,
            "final_energy": final_energy,
            "energy_reduction_ratio": energy_reduction,
            "initial_grad_norm": initial_grad,
            "final_grad_norm": final_grad,
            "converged": final_grad < 0.01 * initial_grad if initial_grad > 0 else True,
        }

    return summary
