"""Autoregressive token generation for trained FabricPC graphs.

Sampling runs a jitted ``lax.scan`` over a fixed-size sliding context
window: each step clamps the window into the input node (plus the graph's
causal mask, when its ``TaskMap`` declares one), settles the graph via
``run_inference``, and samples the next token from the output node's
post-activation probabilities (``z_mu``) at the last position.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.inference import run_inference
from fabricpc.core.types import GraphParams, GraphStructure
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.training.trainer import build_clamps


def _generation_step(
    carry: Tuple[jnp.ndarray, jnp.ndarray, jax.Array],
    step_idx: int,
    params: GraphParams,
    structure: GraphStructure,
    input_node: str,
    output_node: str,
    vocab_size: int,
    batch_size: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jax.Array], jnp.ndarray]:
    """Single ``lax.scan`` generation step over a fixed-size sliding window.

    ``carry`` is ``(context_window, output_buffer, rng_key)``; static args
    are closed over by :func:`generate`. Returns ``(new_carry, next_token)``.
    """
    context_window, output_buffer, rng_key = carry
    rng_key, sample_key, init_key = jax.random.split(rng_key, 3)

    # A 1D input node (seq_len,) takes int token indices (EmbeddingNode); a
    # 2D input node (seq_len, vocab) takes one-hot vectors (Linear).
    input_shape = structure.nodes[input_node].node_info.shape
    if len(input_shape) == 1:
        input_data = context_window
    else:
        input_data = jax.nn.one_hot(context_window, vocab_size)

    # Input only — the output runs free. build_clamps injects the causal
    # mask when the graph's TaskMap declares one (v1); v2 masks internally.
    clamps = build_clamps({"x": input_data}, structure, clamp_target=False)
    state = initialize_graph_state(
        structure, batch_size, init_key, clamps=clamps, params=params
    )
    final_state = run_inference(params, state, clamps, structure)

    # z_mu is post-activation (softmax) probabilities; take the last position.
    output_probs = final_state.nodes[output_node].z_mu
    output_last = output_probs[:, -1, :]
    logits = jnp.log(output_last + 1e-10) / temperature

    # top-k filtering (run unconditionally with effective k = vocab when unset)
    effective_top_k = top_k if top_k is not None else vocab_size
    top_k_logits, top_k_indices = jax.lax.top_k(logits, effective_top_k)
    neg_inf_mask = jnp.full_like(logits, float("-inf"))
    logits = neg_inf_mask.at[jnp.arange(batch_size)[:, None], top_k_indices].set(
        top_k_logits
    )

    # top-p (nucleus) filtering
    if top_p is not None:
        sorted_indices = jnp.argsort(-logits, axis=-1)
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)
        cutoff_mask = cumsum_probs > top_p
        # Shift so at least one token survives.
        cutoff_mask = jnp.concatenate(
            [jnp.zeros((batch_size, 1), dtype=bool), cutoff_mask[:, :-1]], axis=-1
        )
        sorted_logits = jnp.where(cutoff_mask, float("-inf"), sorted_logits)
        unsort_indices = jnp.argsort(sorted_indices, axis=-1)
        logits = jnp.take_along_axis(sorted_logits, unsort_indices, axis=-1)

    next_token = jax.random.categorical(sample_key, logits, axis=-1)
    new_context = jnp.concatenate([context_window[:, 1:], next_token[:, None]], axis=1)
    new_output_buffer = output_buffer.at[:, step_idx].set(next_token)
    return (new_context, new_output_buffer, rng_key), next_token


def generate(
    params: GraphParams,
    structure: GraphStructure,
    prompt: jnp.ndarray,
    max_new_tokens: int,
    rng_key: jax.Array,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> jnp.ndarray:
    """Autoregressively sample ``max_new_tokens`` from a trained model.

    JIT-compiled; the inner loop is a ``lax.scan`` over fixed-size buffers.
    ``prompt`` may be ``(seq_len,)`` or ``(batch, seq_len)`` integer token
    indices; it is left-padded (or truncated) to the input node's sequence
    length. Returns the prompt concatenated with the generated tokens (the
    batch dim is dropped if the input was 1D).

    Args:
        params: Trained model parameters.
        structure: Graph structure with ``x`` and ``y`` task keys.
        prompt: Initial token indices.
        max_new_tokens: Number of tokens to sample.
        rng_key: Sampling key.
        temperature: Sampling temperature (>1 flattens, <1 sharpens).
        top_k: If set, sample only from the k highest-probability tokens.
        top_p: If set, nucleus sampling with this cumulative-probability cap.
    """
    if prompt.ndim == 1:
        prompt = prompt[None, :]
        unbatch = True
    else:
        unbatch = False

    batch_size, prompt_len = prompt.shape
    input_node = structure.task_map.get("x")
    output_node = structure.task_map.get("y")
    if input_node is None or output_node is None:
        raise ValueError("Structure must have 'x' and 'y' in task_map")

    vocab_size = structure.nodes[output_node].node_info.shape[-1]
    seq_len = structure.nodes[input_node].node_info.shape[0]

    # Build the initial context window (pad-left or truncate to seq_len).
    if prompt_len >= seq_len:
        context_window = prompt[:, -seq_len:]
    else:
        context_window = jnp.pad(
            prompt, ((0, 0), (seq_len - prompt_len, 0)), constant_values=0
        )

    @jax.jit
    def jit_generate_loop(context: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
        def scan_fn(carry, step_idx):
            return _generation_step(
                carry,
                step_idx,
                params=params,
                structure=structure,
                input_node=input_node,
                output_node=output_node,
                vocab_size=vocab_size,
                batch_size=batch_size,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        output_buffer = jnp.zeros((batch_size, max_new_tokens), dtype=jnp.int32)
        init_carry = (context, output_buffer, rng)
        (_, final_output_buffer, _), _ = jax.lax.scan(
            scan_fn, init_carry, jnp.arange(max_new_tokens)
        )
        return final_output_buffer

    generated_tokens = jit_generate_loop(context_window, rng_key)
    result = jnp.concatenate([prompt, generated_tokens], axis=1)
    if unbatch:
        result = result[0]
    return result
