"""
Transformer Predictive Coding Demo

Character-level language modeling on TinyShakespeare with PC or backprop training.
PC training still suffers from some poor variance scaling — treat as a starting point for experimentation.

Architecture (per block: TransformerBlock + SkipConnection)::

    input ──→ Embedding ──→ TransformerBlock_0 ──→ SkipConnection_0 ──→ ... ──→ output
                                  │                   ↑
                                  └── (skip) ─────────┘

    Each SkipConnection sums the transformer output with the previous
    node's output. The stream enters the unscaled "skip" slot at scale 1.0;
    the block's output enters the "in" slot, where muPC damps it once by
    1/sqrt(L).

Usage:
    python examples/transformer_demo.py
    python examples/transformer_demo.py --mode backprop --lr 1e-3 --num_epochs 3
    python examples/transformer_demo.py --mode pc --num_blocks 2

Results: PC training (cuda13, rtx3090, jax 0.10.1, can vary a few points in perplexity in different jax versions / hardware due to sensitivity to floating point rounding)
Final train energy: 2.2656 (internal energy per token)
Test loss: 2.6846, Perplexity: 14.65
Prompt: 'ROMEO: '
----------------------------------------
ROMEO: hirifo!Borerenoooroo
----------------------------------------

Backprop Training (python examples/transformer_demo.py --mode backprop)
Test loss: 1.8846, Perplexity: 6.58
Prompt: 'ROMEO: '
----------------------------------------
ROMEO: his.fe!

CARILARE:
M
----------------------------------------
"""

import argparse
import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import Tuple, List
from tqdm.auto import tqdm

from fabricpc.nodes import (
    Linear,
    TransformerBlock,
    IdentityNode,
    SkipConnection,
    EmbeddingNode,
)
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params, FeedforwardStateInit
from fabricpc.core.mupc import MuPCConfig
from fabricpc.core.activations import (
    SoftmaxActivation,
    GeluActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.initializers import (
    NormalInitializer,
)
from fabricpc.core.inference import InferenceSGDNormClip
import optax
from fabricpc.training import EpochContext, evaluate, generate, train
from fabricpc.utils.dashboarding import (
    AimExperimentTracker,
    TrackingConfig,
    create_iter_callback,
    is_aim_available,
)
from fabricpc.utils.data import CharDataLoader
from fabricpc import setup_jax

setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")

TRACKED_NODES = ["embed", "transformer_0"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transformer PC/Backprop demo on TinyShakespeare"
    )
    parser.add_argument(
        "--mode",
        choices=["pc", "backprop"],
        default="pc",
        help="Training mode: predictive coding or backpropagation (default: pc)",
    )
    parser.add_argument("--seq_len", type=int, default=128, help="Sequence length")
    parser.add_argument(
        "--embed_dim", type=int, default=128, help="Embedding dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_blocks", type=int, default=1, help="Number of transformer blocks"
    )
    parser.add_argument(
        "--ff_dim", type=int, default=512, help="Feed-forward hidden dimension"
    )
    parser.add_argument("--rope_theta", type=float, default=500.0, help="RoPE theta")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--num_epochs",
        type=float,
        default=1.0,
        help="Number of epochs (supports fractional)",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=None,
        help="PC inference steps (default: auto)",
    )
    parser.add_argument(
        "--eta_infer", type=float, default=0.1, help="PC inference step size"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Peak learning rate (default: 3e-5 for pc, 1e-4 for backprop)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    if args.lr is None:
        # Gradients are means per token, so the 0.8 clip no longer normalizes
        # every PC step as it did on batch-summed gradients; at 1e-4 the PC
        # run destabilized after ~3500 steps, and 3e-5 holds the full epoch
        # (perplexity 14.65 vs 14.86 before the normalization). Backprop is
        # unaffected by the rescale and keeps 1e-4.
        args.lr = 3e-5 if args.mode == "pc" else 1e-4
    return args


# --- Model Configuration ---


def create_transformer_model(
    vocab_size: int,
    seq_len: int,
    embed_dim: int,
    num_heads: int,
    num_blocks: int,
    ff_dim: int,
    rope_theta: float,
    rng_key: jax.Array,
    infer_steps: int = None,
    eta_infer: float = 0.1,
) -> Tuple:
    """Create a transformer language model. Returns (structure, params)."""
    if infer_steps is None:
        infer_steps = 3 * (2 * num_blocks + 2)

    input_node = IdentityNode(shape=(seq_len,), name="input")
    # Use EmbeddingNode (table lookup) instead of Linear with one-hot input.
    # Linear + one-hot + muPC produces Var ~ 1/V (embedding variance collapse)
    # because muPC assumes dense input with fan_in active features. Use unit normal weight initialization and no muPC scaling for the embedding node.
    embed = EmbeddingNode(
        shape=(seq_len, embed_dim),
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        weight_init=NormalInitializer(std=1.0),
        name="embed",
    )
    mask_node = IdentityNode(shape=(1, seq_len, seq_len), name="mask")

    nodes = [input_node, embed, mask_node]
    edges = [Edge(source=input_node, target=embed.slot("in"))]

    prev_node = embed
    for i in range(num_blocks):
        new_block = TransformerBlock(
            shape=(seq_len, embed_dim),
            num_heads=num_heads,
            ff_dim=ff_dim,
            internal_activation=GeluActivation(),
            rope_theta=rope_theta,
            name=f"transformer_{i}",
        )
        new_skip = SkipConnection(
            shape=(seq_len, embed_dim),
            name=f"skip_{i}",
        )
        nodes.append(new_block)
        edges.append(Edge(source=prev_node, target=new_block.slot("in")))
        edges.append(Edge(source=mask_node, target=new_block.slot("mask")))
        nodes.append(new_skip)
        edges.append(Edge(source=prev_node, target=new_skip.slot("skip")))
        edges.append(Edge(source=new_block, target=new_skip.slot("in")))
        prev_node = new_skip

    output_node = Linear(
        shape=(seq_len, vocab_size),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        weight_init=NormalInitializer(std=np.sqrt(1.0 / embed_dim)),
        name="output",
    )
    nodes.append(output_node)
    edges.append(Edge(source=prev_node, target=output_node.slot("in")))

    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=input_node, y=output_node, causal_mask=mask_node),
        graph_state_initializer=FeedforwardStateInit(),
        inference=InferenceSGDNormClip(
            eta_infer=eta_infer, infer_steps=infer_steps, max_norm=5.0, latent_decay=0.0
        ),
        scaling=MuPCConfig(include_output=False),
    )
    params = initialize_params(structure, rng_key)
    return structure, params


# --- Text Generation ---


def generate_text(
    params,
    structure,
    dataset: CharDataLoader,
    prompts: List[str],
    max_new_tokens: int = 100,
    rng_key: jax.Array = None,
    temperature: float = 0.8,
    top_k: int = None,
    top_p: float = None,
    algorithm: str = "pc",
) -> List[str]:
    """Generate text autoregressively from batched prompts."""
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    seq_len = structure.nodes["input"].node_info.shape[0]
    pad_char = dataset.char_to_idx.get(" ", 0)

    batch_indices = []
    for prompt in prompts:
        prompt_indices = [dataset.char_to_idx.get(ch, 0) for ch in prompt]
        if len(prompt_indices) > seq_len:
            prompt_indices = prompt_indices[-seq_len:]
        elif len(prompt_indices) < seq_len:
            prompt_indices = [pad_char] * (
                seq_len - len(prompt_indices)
            ) + prompt_indices
        batch_indices.append(prompt_indices)

    prompt_tokens = jnp.array(batch_indices)  # (batch_size, seq_len)

    generated_tokens = generate(
        params=params,
        structure=structure,
        prompt=prompt_tokens,
        max_new_tokens=max_new_tokens,
        rng_key=rng_key,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        algorithm=algorithm,
    )

    generated_texts = []
    for i, prompt in enumerate(prompts):
        tokens = np.array(generated_tokens[i])

        pad_len = seq_len - len(prompt)
        if pad_len > 0:
            tokens = tokens[pad_len:]

        text = dataset.decode(tokens)
        generated_texts.append(text)

    return generated_texts


# --- Main Experiment ---


def main(args=None):
    if args is None:
        args = parse_args()

    use_pc = args.mode == "pc"

    master_key = jax.random.PRNGKey(args.seed)
    graph_key, train_key, gen_key = jax.random.split(master_key, 3)

    # Data
    train_loader = CharDataLoader(
        "train", seq_len=args.seq_len, batch_size=args.batch_size, shuffle=True, seed=0
    )
    test_loader = CharDataLoader(
        "test", seq_len=args.seq_len, batch_size=args.batch_size, shuffle=False
    )

    # EmbeddingNode takes integer token indices directly.
    vocab_size = train_loader.vocab_size

    class _IndexLoader:
        """Repackage (x, y) tuples as {'x': ..., 'y': ...} dicts.

        Both x and y are integer token ids (batch, seq_len); y is one-hot
        encoded later by build_clamps (non-float targets are one-hot by
        dtype).
        """

        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __iter__(self):
            for x_idx, y_idx in self.base:
                yield {"x": x_idx, "y": y_idx}

    train_batches = _IndexLoader(train_loader)
    test_batches = _IndexLoader(test_loader)

    print(
        f"Vocab: {vocab_size}, Train batches: {len(train_loader)}, Test batches: {len(test_loader)}"
    )

    # Model
    structure, params = create_transformer_model(
        vocab_size=train_loader.vocab_size,
        seq_len=args.seq_len,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        ff_dim=args.ff_dim,
        rope_theta=args.rope_theta,
        rng_key=graph_key,
        infer_steps=args.infer_steps,
        eta_infer=args.eta_infer,
    )

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Model created: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    print(f"Total parameters: {total_params:,}")

    # Aim tracking (optional)
    if is_aim_available():
        tracking_config = TrackingConfig(
            experiment_name="transformer_pc_shakespeare",
            run_name=f"{'PC' if use_pc else 'BP'}_{args.num_blocks}blk_{args.embed_dim}d",
            track_energy=True,
            track_weight_distributions=True,
            track_state_distributions=True,
            nodes_to_track=TRACKED_NODES,
            distribution_nodes=TRACKED_NODES,
            tracking_every_n_batches=50,
            state_tracking_every_n_infer_steps=5,
        )
        tracker = AimExperimentTracker(config=tracking_config)
        tracker.log_hyperparams(
            {
                "model_config": {
                    "seq_len": args.seq_len,
                    "embed_dim": args.embed_dim,
                    "num_heads": args.num_heads,
                    "num_blocks": args.num_blocks,
                    "ff_dim": args.ff_dim,
                    "rope_theta": args.rope_theta,
                    "total_params": total_params,
                },
                "training_method": "PC" if use_pc else "Backprop",
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "infer_steps": args.infer_steps,
                "eta_infer": args.eta_infer,
                "lr": args.lr,
            }
        )
        tracker.log_graph_structure(structure)
    else:
        tracker = None

    # Training
    # Cosine decay from args.lr over the full run; alpha is the
    # final-to-peak learning-rate ratio.
    lr_schedule = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=max(1, round(args.num_epochs * len(train_batches))),
        alpha=0.01,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.8),
        optax.adamw(lr_schedule, weight_decay=0.1),
    )
    train_config = {"num_epochs": args.num_epochs}

    def epoch_callback(ctx: EpochContext):
        metrics = evaluate(
            ctx.params,
            ctx.structure,
            test_batches,
            {},
            ctx.epoch_key,
            algorithm=ctx.algorithm,
        )
        tqdm.write(
            f"  Test - Loss: {metrics['cross_entropy']:.4f}, "
            f"Perplexity: {metrics['perplexity']:.2f}, "
            f"Acc: {metrics['accuracy']:.4f}"
        )
        return metrics

    print(
        f"\nTraining ({'PC' if use_pc else 'Backprop'}, {args.num_epochs} epochs, lr={args.lr})..."
    )

    start_time = time.time()

    # train's tqdm bar shows the per-batch energy: the internal energy per
    # token under PC, the per-token cross-entropy (target_energy under
    # CrossEntropyEnergy) under backprop. The tracking callback logs energy,
    # per-node energy for nodes_to_track, weight distributions for
    # distribution_nodes and, on tracked batches under PC, the states of a
    # jitted re-settle every state_tracking_every_n_infer_steps steps, as
    # tracking_config asks. Per-batch keys follow the trainer's fold_in
    # stream.
    result = train(
        params,
        structure,
        train_batches,
        optimizer,
        train_config,
        train_key,
        algorithm="pc" if use_pc else "backprop",
        verbose=True,
        iter_callback=create_iter_callback(tracker) if tracker is not None else None,
        epoch_callback=epoch_callback,
    )
    energy_history = result.iter_results
    eval_results = result.epoch_results

    trained_params = result.params
    train_time = time.time() - start_time

    print(
        f"\nTraining completed in {train_time:.1f}s ({train_time/args.num_epochs:.1f}s per epoch)"
    )

    # Generate samples
    prompts = [
        "ROMEO: ",
        "Know, Rome, that",
        "MENENIUS:",
        "the more virtuous",
        "by his looks",
        "To be or not to be",
        "The king",
    ]

    generated_texts = generate_text(
        trained_params,
        structure,
        train_loader,
        prompts=prompts,
        max_new_tokens=20,
        rng_key=gen_key,
        temperature=0.8,
        algorithm="pc" if use_pc else "backprop",
    )

    for prompt, generated in zip(prompts, generated_texts):
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        print(generated)
        print("-" * 40)

    if tracker is not None:
        tracker.close()

    # Results
    print(f"\nFinal train energy: {energy_history[-1][-1]['energy']:.4f}")
    if eval_results and eval_results[-1]:
        final_eval = eval_results[-1]
        print(
            f"Test loss: {final_eval['cross_entropy']:.4f}, Perplexity: {final_eval['perplexity']:.2f}"
        )

    return trained_params, structure, train_loader, test_loader


if __name__ == "__main__":
    args = parse_args()
    main(args)
