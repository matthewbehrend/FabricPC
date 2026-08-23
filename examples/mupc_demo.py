"""
muPC Scaling — FC-ResNet on MNIST
==================================

Demonstrates muPC (Maximal Update Parameterization for Predictive Coding)
on a deep fully-connected residual network trained on MNIST, using only
FabricPC's native graph components. muPC scaling is computed automatically
from the graph topology — no manual scaling needed.

Two residual block styles are supported (--mode flag):

  skip: Linear + SkipConnection (2 PC nodes per block)
      prev -> Linear(W, Tanh) -> SkipConnection(sum)
        |                              ^
        +------------------------------+  (identity skip path)


  linear_residual (default): LinearResidual (1 PC node per block, half graph depth)
      prev -> LinearResidual(W, Tanh, skip=prev)

Both use edge-based muPC scaling: "in" slot edges get full variance
scaling, "skip" slot edges pass through at scale 1.0.

Usage:
    python examples/mupc_demo.py
    python examples/mupc_demo.py --mode skip --num_blocks 32
    python examples/mupc_demo.py --num_blocks 4 --verbose

    for d in 8 16 32 64 128; do
      python examples/mupc_demo.py --num_blocks $d
    done

| Depth   | accuracy   |
|---------|------------|
| 8       | 92.0       |
| 16      | 89.7%      |
| 32      | 85.6%      |
| 64      | 84.1%      |
| 128     | 82.2%      |
"""

import argparse
import time
import jax
import optax

from fabricpc.nodes import Linear, IdentityNode, LinearResidual
from fabricpc.nodes.skip_connection import SkipConnection
from fabricpc.core.topology import Edge, GraphNamespace
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import TanhActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import MuPCInitializer, XavierInitializer
from fabricpc.core.mupc import MuPCConfig
from fabricpc.training import train, evaluate
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc import setup_jax

setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")


def parse_args():
    parser = argparse.ArgumentParser(description="muPC demo: FC-ResNet on MNIST")
    parser.add_argument(
        "--num_blocks",
        type=int,
        default=16,
        help="Number of residual blocks (default: 16)",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=64,
        help="Hidden layer width (default: 64)",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--eta_infer",
        type=float,
        default=0.1,
        help="Inference rate (default: 0.1)",
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=None,
        help="Inference steps per sample (default: 6*(num_blocks+2))",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.002,
        help="Learning rate (default: 0.002)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (default: 0.01)",
    )
    parser.add_argument(
        "--mode",
        choices=["skip", "linear_residual"],
        default="linear_residual",
        help="Residual block style: 'skip' (Linear+SkipConnection, 2 nodes/block) "
        "or 'linear_residual' (LinearResidual, 1 node/block). Default: linear_residual",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print per-epoch training output",
    )
    return parser.parse_args()


def build_fc_resnet(num_blocks, hidden_dim, infer_steps=None, *, eta_infer):
    """
    Build an FC-ResNet for MNIST with muPC scaling.

    Architecture:
        input(784) -> stem(hidden_dim) -> [N residual blocks] -> output(10)

    Each residual block has a Linear transform path and a SkipConnection
    that sums the transform output with the identity skip. The stream
    enters the SkipConnection's unscaled "skip" slot, preserving the
    identity mapping; the branch enters its "in" slot, where muPC damps
    it once by 1/sqrt(L).

    Args:
        num_blocks: Number of residual blocks.
        hidden_dim: Width of hidden layers.
        infer_steps: Inference steps per sample. Default: max(20, 4*(num_blocks+2)).
        eta_infer: Inference rate.

    Returns:
        GraphStructure with muPC scaling.
    """
    if infer_steps is None:
        infer_steps = max(20, 3 * (2 * num_blocks + 2))
    mupc_init = MuPCInitializer()

    # Input node
    input_node = IdentityNode(shape=(784,), name="input")

    # Stem: projects 784 -> hidden_dim (no skip connection)
    stem = Linear(
        shape=(hidden_dim,),
        weight_init=mupc_init,
        flatten_input=True,
        name="stem",
    )

    all_nodes = [input_node, stem]
    all_edges = [Edge(source=input_node, target=stem.slot("in"))]

    # Residual blocks
    prev = stem
    for i in range(num_blocks):
        with GraphNamespace(f"block{i}"):
            linear = Linear(
                shape=(hidden_dim,),
                activation=TanhActivation(),
                weight_init=mupc_init,
                name="linear",
            )
            skip = SkipConnection(
                shape=(hidden_dim,),
                name="sum",
            )

        all_nodes.extend([linear, skip])
        all_edges.extend(
            [
                Edge(source=prev, target=linear.slot("in")),  # transform path
                Edge(source=prev, target=skip.slot("skip")),  # stream/identity path
                Edge(source=linear, target=skip.slot("in")),  # branch joins stream
            ]
        )
        prev = skip

    # Output classifier
    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        weight_init=XavierInitializer(),
        name="output",
    )
    all_nodes.append(output)
    all_edges.append(Edge(source=prev, target=output.slot("in")))

    structure = graph(
        nodes=all_nodes,
        edges=all_edges,
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
        scaling=MuPCConfig(include_output=False),
    )

    return structure


def build_fc_resnet_linear_residual(
    num_blocks, hidden_dim, infer_steps=None, *, eta_infer
):
    """
    Build an FC-ResNet using LinearResidual nodes (1 PC node per block).

    Architecture:
        input(784) -> stem(hidden_dim) -> [N LinearResidual blocks] -> output(10)

    Each LinearResidual has two slots:
      - "in"   (scalable): receives input, applies W @ x + b then activation
      - "skip" (non-scalable): receives identity skip, summed after activation

    This halves the graph depth compared to build_fc_resnet: N+2 nodes
    instead of 2N+2.
    """
    if infer_steps is None:
        infer_steps = max(20, 3 * (num_blocks + 2))
    mupc_init = MuPCInitializer()

    input_node = IdentityNode(shape=(784,), name="input")

    stem = Linear(
        shape=(hidden_dim,),
        weight_init=mupc_init,
        flatten_input=True,
        name="stem",
    )

    all_nodes = [input_node, stem]
    all_edges = [Edge(source=input_node, target=stem.slot("in"))]

    prev = stem
    for i in range(num_blocks):
        with GraphNamespace(f"block{i}"):
            res = LinearResidual(
                shape=(hidden_dim,),
                activation=TanhActivation(),
                weight_init=mupc_init,
                name="res",
            )

        all_nodes.append(res)
        all_edges.extend(
            [
                Edge(source=prev, target=res.slot("in")),  # transform path
                Edge(source=prev, target=res.slot("skip")),  # identity skip
            ]
        )
        prev = res

    output = Linear(
        shape=(10,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        weight_init=XavierInitializer(),
        name="output",
    )
    all_nodes.append(output)
    all_edges.append(Edge(source=prev, target=output.slot("in")))

    structure = graph(
        nodes=all_nodes,
        edges=all_edges,
        task_map=TaskMap(x=input_node, y=output),
        inference=InferenceSGD(eta_infer=eta_infer, infer_steps=infer_steps),
        scaling=MuPCConfig(include_output=False),
    )

    return structure


def main():
    args = parse_args()

    print("=" * 60)
    print("muPC Demo: FC-ResNet on MNIST")
    print("=" * 60)

    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    # Build model
    if args.mode == "linear_residual":
        structure = build_fc_resnet_linear_residual(
            num_blocks=args.num_blocks,
            hidden_dim=args.hidden_dim,
            infer_steps=args.infer_steps,
            eta_infer=args.eta_infer,
        )
        mode_label = "LinearResidual (1 node/block)"
    else:
        structure = build_fc_resnet(
            num_blocks=args.num_blocks,
            hidden_dim=args.hidden_dim,
            infer_steps=args.infer_steps,
            eta_infer=args.eta_infer,
        )
        mode_label = "Linear+SkipConnection (2 nodes/block)"
    params = initialize_params(structure, graph_key)

    print(f"\nMode: {mode_label}")
    print(
        f"Architecture: input(784) -> stem({args.hidden_dim})"
        f" -> {args.num_blocks} residual blocks -> output(10)"
    )
    print(f"Model: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    # for name, node in structure.nodes.items():
    #     ni = node.node_info
    # if ni.scaling_config is not None:
    #     fwd_scales = list(ni.scaling_config.forward_scale.values())
    #     tag = f"fwd_scale={fwd_scales[0]:.6f}" if fwd_scales else ""
    # else:
    #     tag = "no scaling"
    # print(f"  {name}: shape={ni.shape}, {tag}")

    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")

    # Data
    train_loader = MnistLoader(
        "train",
        batch_size=args.batch_size,
        tensor_format="flat",
        shuffle=True,
        seed=42,
    )
    test_loader = MnistLoader(
        "test",
        batch_size=args.batch_size,
        tensor_format="flat",
        shuffle=False,
    )

    # Train
    optimizer = optax.adamw(args.lr, weight_decay=args.weight_decay)
    train_config = {"num_epochs": args.num_epochs}

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
        verbose=args.verbose,
    )
    trained_params = result.params

    elapsed = time.time() - start_time
    print(f"Training time: {elapsed:.1f}s ({elapsed / args.num_epochs:.1f}s per epoch)")

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate(trained_params, structure, test_loader, train_config, eval_key)
    print(f"Test Accuracy: {metrics['accuracy'] * 100:.2f}%")

    if metrics["accuracy"] >= 0.85:
        print("PASS: accuracy >= 85%")
    else:
        print(f"BELOW TARGET: {metrics['accuracy']*100:.1f}% < 90%")


if __name__ == "__main__":
    main()
