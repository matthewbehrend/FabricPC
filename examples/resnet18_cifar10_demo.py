"""
ResNet-18 — CIFAR-10 Demo (Predictive Coding)
===============================================

Demonstrates a ResNet-18 architecture built as a predictive coding graph
using muPC scaling on CIFAR-10.

Architecture (CIFAR-10 variant — no 7x7 conv or maxpool):
    input(32,32,3) -> stem(32,32,32, 3x3)
    -> Stage 1: 2 residual blocks (32,32,32)
    -> Stage 2: 2 residual blocks (16,16,64)
    -> Stage 3: 2 residual blocks (8,8,128)
    -> Stage 4: 2 residual blocks (4,4,256)
    -> GlobalAvgPool -> Linear(10, softmax+CE)

Each residual block:
    x -> conv_a(3x3, act) -> conv_b(3x3, act) -> skip(sum)

Skip connections use SkipConnection (same dims) or 1x1 conv (downsample).

Supports multiple activation functions (--activation):
    relu      — baseline, fast but dead neurons during PC inference
    tanh      — bounded, non-zero gradients everywhere, best for PC
    gelu      — smooth middle ground between relu and tanh
    leaky_relu — avoids dead neurons with minimal change from relu

Includes cosine LR schedule with warmup and optional data augmentation
(random horizontal flip + random crop with padding).

--trainer backprop trains the identical graph (muPC init and edge scaling
included) with end-to-end autodiff instead of iterative PC inference;
--eta_infer and --infer_steps have no effect in that mode.

Usage:
    python examples/resnet18_cifar10_demo.py                      # 2-epoch smoke test
    python examples/resnet18_cifar10_demo.py --activation gelu    # with gelu instead of relu
    python examples/resnet18_cifar10_demo.py --trainer backprop   # backprop baseline on the same model
    python examples/resnet18_cifar10_demo.py --num_epochs 100 --eval_every 10 --augment --activation gelu # full training with augmentation and gelu activation


Results (RTX3090, cuda13, jax 0.10.2; can vary a few points in accuracy across
jax versions and hardware, from sensitivity to floating point rounding)

python examples/resnet18_cifar10_demo.py  # smoke test
Model: 31 nodes, 38 edges
Total parameters: 2,795,210
Train energy: 0.4792
Test Accuracy: 33.71%
Training time: 952.3s (476.2s per epoch)

python examples/resnet18_cifar10_demo.py --num_epochs 100 --eval_every 10 --augment --activation gelu --trainer backprop
Trainer: backprop  |  Activation: gelu  |  Epochs: 100  |  LR: 0.001  |  Augment: True
Training for 100 epochs (JIT compilation on first batch)...
  Epoch 10: accuracy=56.21%
  Epoch 20: accuracy=63.77%
  Epoch 30: accuracy=69.02%
  Epoch 40: accuracy=70.94%
  Epoch 50: accuracy=73.01%
  Epoch 60: accuracy=75.41%
  Epoch 70: accuracy=75.71%
  Epoch 80: accuracy=76.87%
  Epoch 90: accuracy=77.07%
  Epoch 100: accuracy=77.11%
Training time: 557.0s (5.6s per epoch)
Final evaluation...
Test Accuracy: 77.11%
"""

import jax
import numpy as np
import optax
import argparse
import time
from fabricpc.nodes import ConvNode, Linear, IdentityNode, SkipConnection, AvgPool
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.core import InferenceSGDNormClip
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    TanhActivation,
    GeluActivation,
    LeakyReLUActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.initializers import (
    MuPCInitializer,
    XavierInitializer,
)
from fabricpc.core.mupc import MuPCConfig
from fabricpc.training import EpochContext, evaluate, train
from fabricpc.utils.data.dataloader import Cifar10Loader
from fabricpc import setup_jax

setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")


# =============================================================================
# Activation Factory
# =============================================================================


def get_activation(name):
    factories = {
        "relu": ReLUActivation,
        "tanh": TanhActivation,
        "gelu": GeluActivation,
        "leaky_relu": lambda: LeakyReLUActivation(alpha=0.1),
    }
    if name not in factories:
        raise ValueError(f"Unknown activation: {name}. Choose from {list(factories)}")
    return factories[name]()


def make_optimizer(lr, weight_decay, num_epochs, steps_per_epoch):
    """adamw on a warmup-cosine schedule: 5% linear warmup to lr, cosine decay to lr/100."""
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = int(0.05 * total_steps)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=lr * 0.01,
    )
    return optax.adamw(schedule, weight_decay=weight_decay)


# =============================================================================
# Data Augmentation
# =============================================================================


class AugmentedCifar10Loader:
    """Wraps Cifar10Loader with random horizontal flip and random crop+pad."""

    def __init__(self, base_loader, seed=42, pad=4):
        self.base_loader = base_loader
        self.seed = seed
        self.pad = pad
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        pad = self.pad
        for images, labels in self.base_loader:
            flip_mask = rng.random(images.shape[0]) > 0.5
            images[flip_mask] = images[flip_mask, :, ::-1, :]

            padded = np.pad(
                images, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect"
            )
            B, H, W, C = images.shape
            crop_y = rng.integers(0, 2 * pad + 1, size=B)
            crop_x = rng.integers(0, 2 * pad + 1, size=B)
            for i in range(B):
                images[i] = padded[
                    i, crop_y[i] : crop_y[i] + H, crop_x[i] : crop_x[i] + W, :
                ]

            yield images, labels

    def __len__(self):
        return len(self.base_loader)


# =============================================================================
# Custom Nodes
# =============================================================================


# AvgPool now lives in fabricpc.nodes.pooling (imported above). Use
# global_pool=True for the (B, H, W, C) -> (B, C) global average pooling here.


# =============================================================================
# ResNet-18 Graph Builder
# =============================================================================


def make_residual_block(
    prev_node,
    channels,
    stride,
    block_name,
    weight_init,
    activation=ReLUActivation(),
):
    """
    Create one residual block: conv_a -> conv_b(act) -> skip(sum).

    Activation is applied on the main path before summation. The skip path
    passes through without activation, preserving gradient flow.

    Returns:
        (nodes_list, edges_list, skip_node) where skip_node is the block output.
    """
    in_h, in_w, in_channels = prev_node._shape

    if stride == 1:
        out_h, out_w = in_h, in_w
    else:
        out_h, out_w = in_h // stride, in_w // stride

    nodes = []
    edges = []

    conv_a = ConvNode(
        shape=(out_h, out_w, channels),
        kernel_size=(3, 3),
        stride=(stride, stride),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
        name=f"{block_name}_conv_a",
    )

    conv_b = ConvNode(
        shape=(out_h, out_w, channels),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
        name=f"{block_name}_conv_b",
    )

    skip_node = SkipConnection(
        shape=(out_h, out_w, channels),
        name=f"{block_name}_skip_sum",
    )

    nodes.extend([conv_a, conv_b, skip_node])

    # Main path edges
    edges.append(Edge(source=prev_node, target=conv_a.slot("in")))
    edges.append(Edge(source=conv_a, target=conv_b.slot("in")))
    edges.append(Edge(source=conv_b, target=skip_node.slot("in")))

    # Skip connection: the stream enters the merge's unscaled "skip" slot
    # (via a 1x1 projection when the block downsamples).
    needs_downsample = (stride != 1) or (in_channels != channels)
    if needs_downsample:
        conv_skip = ConvNode(
            shape=(out_h, out_w, channels),
            kernel_size=(1, 1),
            stride=(stride, stride),
            padding="SAME",
            activation=IdentityActivation(),
            weight_init=weight_init,
            name=f"{block_name}_skip",
        )
        nodes.append(conv_skip)
        edges.append(Edge(source=prev_node, target=conv_skip.slot("in")))
        edges.append(Edge(source=conv_skip, target=skip_node.slot("skip")))
    else:
        edges.append(Edge(source=prev_node, target=skip_node.slot("skip")))

    return nodes, edges, skip_node


def build_resnet18(
    weight_init,
    inference,
    scaling=None,
    output_weight_init=XavierInitializer(),
    activation=ReLUActivation(),
):
    """
    Build ResNet-18 for CIFAR-10 as a predictive coding graph.

    Args:
        weight_init: InitializerBase for conv/linear weights.
        inference: InferenceBase instance (a plain solver or an
            InferenceSchedule) driving PC inference.
        scaling: Optional MuPCConfig for muPC parameterization.
        output_weight_init: InitializerBase for the output layer
            (default: XavierInitializer).
        activation: Activation for hidden conv layers (default: ReLU).

    Returns:
        GraphStructure ready for initialize_params().
    """
    # Input
    input_node = IdentityNode(shape=(32, 32, 3), name="input")

    # Stem convolution: 3x3, 32 channels, no maxpool (CIFAR is 32x32)
    stem = ConvNode(
        shape=(32, 32, 32),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=activation,
        weight_init=weight_init,
        name="stem",
    )

    all_nodes = [input_node, stem]
    all_edges = [Edge(source=input_node, target=stem.slot("in"))]

    # Build 4 stages with [2, 2, 2, 2] blocks
    stage_configs = [
        (32, 1, 2),  # (channels, first_stride, num_blocks)
        (64, 2, 2),
        (128, 2, 2),
        (256, 2, 2),
    ]

    prev = stem
    for stage_idx, (channels, first_stride, num_blocks) in enumerate(stage_configs, 1):
        for block_idx in range(num_blocks):
            stride = first_stride if block_idx == 0 else 1
            block_name = f"s{stage_idx}b{block_idx + 1}"

            nodes, edges, add_node = make_residual_block(
                prev_node=prev,
                channels=channels,
                stride=stride,
                block_name=block_name,
                weight_init=weight_init,
                activation=activation,
            )
            all_nodes.extend(nodes)
            all_edges.extend(edges)
            prev = add_node

    # Global average pooling: (B, 4, 4, 256) -> (B, 256)
    avg_pool = AvgPool(shape=(256,), name="avgpool", global_pool=True)
    all_nodes.append(avg_pool)
    all_edges.append(Edge(source=prev, target=avg_pool.slot("in")))

    # Output: Linear(10) with softmax + cross-entropy
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        weight_init=output_weight_init,
        name="output",
    )
    all_nodes.append(output)
    all_edges.append(Edge(source=avg_pool, target=output.slot("in")))

    # Build graph
    structure = graph(
        nodes=all_nodes,
        edges=all_edges,
        task_map=TaskMap(x=input_node, y=output),
        inference=inference,
        scaling=scaling,
    )

    return structure


# =============================================================================
# Model Factory
# =============================================================================


def _create_mupc_model(rng_key, *, inference, activation=ReLUActivation()):
    """Create ResNet-18 with muPC parameterization."""
    structure = build_resnet18(
        weight_init=MuPCInitializer(),
        inference=inference,
        scaling=MuPCConfig(include_output=False),
        output_weight_init=XavierInitializer(),
        activation=activation,
    )
    params = initialize_params(structure, rng_key)
    return params, structure


# =============================================================================
# Training
# =============================================================================


def run_single_mupc(args):
    """Default mode: single muPC training run with progress bar."""
    activation = get_activation(args.activation)

    if args.trainer == "backprop":
        trainer_label = "Backprop"
        trainer_mode = "backprop"
    else:
        trainer_label = "Predictive Coding"
        trainer_mode = "pc"

    print("=" * 60)
    print(f"ResNet-18 on CIFAR-10 ({trainer_label} + muPC)")
    print("=" * 60)
    print(
        f"Trainer: {args.trainer}  |  Activation: {args.activation}  |  "
        f"Epochs: {args.num_epochs}  |  LR: {args.lr}  |  Augment: {args.augment}"
    )

    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    # Build model
    params, structure = _create_mupc_model(
        graph_key,
        inference=InferenceSGDNormClip(
            eta_infer=args.eta_infer, infer_steps=args.infer_steps, max_norm=1.0
        ),
        activation=activation,
    )

    # Print model summary
    print(f"\nModel: {len(structure.nodes)} nodes, {len(structure.edges)} edges")

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")

    # Data
    base_train_loader = Cifar10Loader(
        "train", batch_size=args.batch_size, shuffle=True, seed=42
    )
    if args.augment:
        train_loader = AugmentedCifar10Loader(base_train_loader, seed=42)
    else:
        train_loader = base_train_loader
    test_loader = Cifar10Loader("test", batch_size=args.batch_size, shuffle=False)

    # Cosine LR schedule with warmup
    steps_per_epoch = len(train_loader)
    optimizer = make_optimizer(
        args.lr, args.weight_decay, args.num_epochs, steps_per_epoch
    )
    train_config = {"num_epochs": args.num_epochs}

    # Periodic evaluation callback
    eval_every = args.eval_every

    def epoch_callback(ctx: EpochContext):
        epoch_num = ctx.epoch_idx + 1
        if eval_every > 0 and (
            epoch_num % eval_every == 0 or epoch_num == args.num_epochs
        ):
            metrics = evaluate(
                ctx.params,
                ctx.structure,
                test_loader,
                ctx.config,
                eval_key,
                algorithm=trainer_mode,
            )
            print(f"  Epoch {epoch_num}: accuracy={metrics['accuracy'] * 100:.2f}%")
            return metrics
        return None

    print(
        f"\nTraining for {args.num_epochs} epochs "
        f"(JIT compilation on first batch)..."
    )
    start_time = time.time()

    result = train(
        params=params,
        structure=structure,
        train_loader=train_loader,
        optimizer=optimizer,
        config=train_config,
        rng_key=train_key,
        algorithm=trainer_mode,
        verbose=False,
        epoch_callback=epoch_callback,
    )
    trained_params = result.params

    elapsed = time.time() - start_time
    print(
        f"\nTraining time: {elapsed:.1f}s ({elapsed / args.num_epochs:.1f}s per epoch)"
    )

    # Final evaluation
    print("Final evaluation...")
    metrics = evaluate(
        trained_params,
        structure,
        test_loader,
        train_config,
        eval_key,
        algorithm=trainer_mode,
    )
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")


# =============================================================================
# CLI and Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="ResNet-18 on CIFAR-10 (Predictive Coding)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=2, help="Training epochs (default: 2)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size (default: 256)"
    )
    parser.add_argument(
        "--infer_steps", type=int, default=120, help="Inference steps (default: 120)"
    )
    parser.add_argument(
        "--eta_infer", type=float, default=0.1, help="Inference rate (default: 0.1)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay (default: 0.01)"
    )
    parser.add_argument(
        "--trainer",
        type=str,
        default="pc",
        choices=["pc", "backprop"],
        help="Training algorithm: pc (predictive coding, default) or backprop",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="gelu",
        choices=["relu", "tanh", "gelu", "leaky_relu"],
        help="Activation function for hidden layers (default: relu)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation (random crop + horizontal flip)",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=0,
        help="Evaluate on test set every N epochs (0 to disable; default: 10)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch output")
    return parser.parse_args()


def main():
    args = parse_args()
    run_single_mupc(args)


if __name__ == "__main__":
    main()
