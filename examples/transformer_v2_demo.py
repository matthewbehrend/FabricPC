"""
FabricPC Transformer V2 Execution Script

Trains the decomposed PC Transformer model on the Tiny Shakespeare dataset,
evaluates its performance, and generates sample text using temperature sampling.

Architecture (each transformer block decomposed into separate PC nodes)::

    input ──→ Embedding ──→ MhaResidual_0 ──→ LnMlp1_0 ──→ Mlp2Residual_0 ──→ ... ──→ VocabProjection
                                  ↑                              ↑
                             mask │                       (skip from MhaResidual)

    Each block: MhaResidual(in + mask) ──→ LnMlp1 ──→ Mlp2Residual(in + skip)

Supports two training modes:
  - Predictive Coding (PC): local Hebbian learning
  - Backpropagation: standard end-to-end gradient training
Both run through fabricpc.training.train; pass a mesh for data parallelism
(see examples/mnist_multi_gpu.py).

Usage:
    PYTHONPATH=. python examples/transformer_v2_demo.py
    PYTHONPATH=. python examples/transformer_v2_demo.py --mode backprop --lr 1e-3
    PYTHONPATH=. python examples/transformer_v2_demo.py --mode pc --depth 6 --num_epochs 10


Results (pre-0.5 trainer, default call, cuda12, rtx3090, jax 0.8.1, can vary
a few points in perplexity in different jax versions / hardware due to
sensitivity to floating point rounding). The 0.5 unified trainer prints
`Epoch i/N — energy: ..., target_energy: ...` per epoch, changes the
per-epoch energy normalization, and fixes the transformer eval numerics
(double softmax, external SSE term — see the 0.5.0 CHANGELOG), so these
numbers are a historical baseline until re-measured under 0.5:
Model parameters: 108,353
Vocab Size: 65
Train Epoch 1/5, Energy: 274.3637, Loss: 2.1401, Perplexity: 8.50
Train Epoch 2/5, Energy: 260.0219, Loss: 2.0280, Perplexity: 7.60
Train Epoch 3/5, Energy: 250.6280, Loss: 1.9546, Perplexity: 7.06
Train Epoch 4/5, Energy: 244.7030, Loss: 1.9089, Perplexity: 6.75
Train Epoch 5/5, Energy: 242.0046, Loss: 1.8878, Perplexity: 6.61
Training completed in 8772.4s
Evaluation completed in 55.2s
Test Accuracy:   35.92%
Test CE Loss:    2.2108
Test Perplexity: 9.12
--- Generating ---
ROMEO: whou sarone the bro beariers thas tray sucas a st my lo the to ate.
"""

import argparse
import jax
import jax.numpy as jnp
from fabricpc.graph_initialization import initialize_params
from fabricpc.training import train, evaluate, generate
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.models import create_deep_transformer
from fabricpc.utils.data import CharDataLoader, BpeDataLoader
import optax
import time
from fabricpc import setup_jax

setup_jax()

# Tuned on Tiny Shakespeare, BPE tokenizer (50k-sequence subset, val PPL ~1133).
# NOTE: severe overfitting on this small corpus (train PPL ~85 vs test ~721),
# from two compounding causes: BPE needs a larger corpus than Tiny Shakespeare,
# and the model has no regularization (dropout / weight decay) yet. Placeholder
# default, not a strong config; char-level remains recommended here. See Future
# Work (regularization, larger BPE-suited datasets).
BPE_DEFAULTS = {
    "embed_dim": 128,
    "num_heads": 4,
    "mlp_dim": 512,
    "depth": 4,
    "seq_len": 64,
    "batch_size": 32,
    "num_epochs": 5,
    "infer_steps": 23,
    "lr": 1.676456563307537e-05,
    "eta_infer": 0.06558512264378524,
    "weight_init_std": 0.039890499730518045,
}

# Tuned on Tiny Shakespeare, char tokenizer (two-phase search, val PPL 12.22).
# Phase 1 fixed the architecture; Phase 2 refined lr / eta_infer / infer_steps.
CHAR_DEFAULTS = {
    "embed_dim": 64,
    "num_heads": 8,
    "mlp_dim": 256,
    "depth": 2,
    "seq_len": 128,
    "batch_size": 16,
    "num_epochs": 5,
    "infer_steps": 12,
    "lr": 0.00012108621644524519,
    "eta_infer": 0.0174852165627398,
    "weight_init_std": 0.015166293102182283,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decomposed PC Transformer on Tiny Shakespeare"
    )
    parser.add_argument(
        "--mode",
        choices=["pc", "backprop"],
        default="pc",
        help="Training mode: predictive coding or backpropagation (default: pc)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=None,
        help="Number of transformer layers (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=None,
        help="Embedding dimension (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=None,
        help="Number of attention heads (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--mlp_dim",
        type=int,
        default=None,
        help="MLP hidden dimension (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Sequence length (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--num_epochs",
        type=float,
        default=5,
        help="Number of training epochs (supports fractional)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=None,
        help="PC inference steps (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--eta_infer",
        type=float,
        default=None,
        help="PC inference learning rate (default: tuned per --tokenizer)",
    )
    parser.add_argument(
        "--weight_init_std",
        type=float,
        default=None,
        help="Weight init std (default: tuned per --tokenizer)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--tokenizer",
        choices=["char", "bpe"],
        default="char",
        help="Tokenizer to use",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print per-batch training energy"
    )
    return parser.parse_args(), parser


def main(args=None):
    if args is None:
        args, _ = parse_args()

    use_pc = args.mode == "pc"

    # --- Resolve tuned defaults for any arg left unset (None sentinel) ---
    # An explicit CLI value overrides; otherwise we fill from the tokenizer's
    # tuned constants. Done before batch_size so the tuned batch_size applies.
    use_bpe = args.tokenizer == "bpe"
    defaults = BPE_DEFAULTS if use_bpe else CHAR_DEFAULTS
    for key, val in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

    # --- Batch size ---
    batch_size = args.batch_size
    print(f"{'PC' if use_pc else 'Backprop'} mode: batch_size={batch_size}")

    # --- Data ---
    if use_bpe:
        train_loader = BpeDataLoader(
            "train",
            seq_len=args.seq_len,
            batch_size=batch_size,
            shuffle=True,
            seed=args.seed,
        )
        test_loader = BpeDataLoader(
            "test", seq_len=args.seq_len, batch_size=batch_size, shuffle=False
        )
    else:
        train_loader = CharDataLoader(
            "train",
            seq_len=args.seq_len,
            batch_size=batch_size,
            shuffle=True,
            seed=args.seed,
        )
        test_loader = CharDataLoader(
            "test", seq_len=args.seq_len, batch_size=batch_size, shuffle=False
        )

    vocab_size = train_loader.vocab_size
    char_to_ix = train_loader.token_to_idx if use_bpe else train_loader.char_to_idx

    # --- Model ---
    structure = create_deep_transformer(
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_dim=args.mlp_dim,
        seq_len=args.seq_len,
        vocab_size=vocab_size,
        inference=InferenceSGDNormClip(
            eta_infer=args.eta_infer,
            infer_steps=args.infer_steps,
            max_norm=5.0,
            latent_decay=0.0,
        ),
        weight_init={"type": "normal", "std": args.weight_init_std},
    )

    # --- Train & Evaluate ---
    master_rng_key = jax.random.PRNGKey(args.seed)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    params = initialize_params(structure, graph_key)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"Model parameters: {n_params:,}")

    train_config = {"num_epochs": args.num_epochs}
    steps_per_epoch = train_loader.num_sequences // batch_size
    schedule = optax.cosine_decay_schedule(
        init_value=args.lr,
        decay_steps=max(1, round(args.num_epochs * steps_per_epoch)),
        alpha=0.1,
    )
    optimizer = optax.adam(schedule)

    print(f"Vocab Size: {vocab_size}")
    start = time.time()

    def iter_callback(epoch_idx, batch_idx, metrics):
        if args.verbose and (batch_idx + 1) % 50 == 0:
            print(
                f"Epoch {epoch_idx + 1} | Batch {batch_idx + 1} | "
                f"Energy: {metrics['energy']:.4f}"
            )

    result = train(
        params,
        structure,
        train_loader,
        optimizer,
        train_config,
        train_key,
        algorithm="pc" if use_pc else "backprop",
        verbose=True,
        iter_callback=iter_callback if args.verbose else None,
    )
    trained_params = result.params

    print(f"Training completed in {time.time() - start:.1f}s")

    # --- Evaluate ---
    eval_start = time.time()
    metrics = evaluate(
        trained_params,
        structure,
        test_loader,
        train_config,
        eval_key,
        algorithm="pc" if use_pc else "backprop",
    )
    print(f"Evaluation completed in {time.time() - eval_start:.1f}s")

    print(f"Test Accuracy:   {metrics['accuracy'] * 100:.2f}%")
    print(f"Test CE Loss:    {metrics['cross_entropy']:.4f}")
    print(f"Test Perplexity: {metrics['perplexity']:.2f}")

    # --- Text Generation ---
    gen_key = jax.random.PRNGKey(99)
    prompt_text = "ROMEO: "
    if use_bpe:
        seed_indices = train_loader._tok.encode(prompt_text).ids
    else:
        seed_indices = [char_to_ix.get(c, 0) for c in prompt_text]

    current_indices = ([0] * (args.seq_len - len(seed_indices)) + seed_indices)[
        -args.seq_len :
    ]
    prompt = jnp.array(current_indices, dtype=jnp.int32)

    print("--- Generating ---")
    generated = generate(
        trained_params,
        structure,
        prompt,
        max_new_tokens=200,
        rng_key=gen_key,
        temperature=0.8,
        algorithm="pc" if use_pc else "backprop",
    )

    # Decode - skip the prompt padding, decode only from where prompt starts
    generated_ids = generated[args.seq_len :]
    print(prompt_text + train_loader.decode(generated_ids))


if __name__ == "__main__":
    main()
