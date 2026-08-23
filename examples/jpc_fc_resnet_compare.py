"""
jpc FC-ResNet Comparison Demo
==============================

Replicates the FC-ResNet architecture from the jpc library (thebuckleylab/jpc)
using FabricPC's predictive coding mechanics, enabling a fair comparison of the
two implementations.

The PreActResBlock bundles a linear path and an identity skip connection inside one node:
 z_mu = scale * matmul(act_x, W) + x   # linear + unscaled skip

In FabricPC's graph-based architecture, this summation would be an explicit IdentityNode with in_degree=2,
and mupc.py:229 would compute a = gain / sqrt(fan_in * K) with K=2 for each incoming edge.
Effective scales folded into the monolithic PreActResBlock:
 - Linear path: a_linear * a_identity = sqrt(2)/sqrt(N) * 1/sqrt(2) = 1/sqrt(N)
   - equivalently: gain / sqrt(fan_in * K) = sqrt(2) / sqrt(N * 2) = 1/sqrt(N)
 - Skip path: a_identity = 1/sqrt(2)

Architecture (matching jpc mupc.ipynb)::

    input(784) -> FCInput(width) -> PreActResBlock(width) x (depth-2) -> Readout(10)

Each PreActResBlock computes:
    z_mu = hidden_scale * (W @ act(x)) + x     (pre-activation + identity skip)

This is a fully-connected ResNet — no convolutions, no spatial structure. The
skip connection is internal to each block (one z_latent per layer, matching jpc).

Scaling modes (--scaling flag):
    jpc:         in=1/√D, hidden=1/√(N*L), out=1/N, skip=1      (depth L compensates)
    fabricpc:    in=1/√D, hidden=gain/√(N*K), out=1/N, skip=1/√K (K=2 in_degree — BROKEN at depth>16)
    fabricpc_v2: in=1/√D, hidden=gain/√(N*L), out=1/N, skip=1    (depth L + Kaiming gain)

Key difference from jpc: FabricPC uses fixed-step SGD inference with gradient
norm clipping (InferenceSGDNormClip), while jpc uses an adaptive ODE solver
(diffrax Heun + PID controller). The ODE solver handles stiff dynamics from
muPC scaling natively. For FabricPC's SGD, we need gradient clipping and a
lower parameter learning rate to compensate.

Results (3 epochs, MNIST, jpc scaling, width=128):
    depth=5:  ~93% (matches jpc reference)
    depth=10: ~93%
    depth=30: ~86% (limited by SGD inference convergence)

Usage:
    # Default: depth=10 FC-ResNet with jpc scaling
    python examples/jpc_fc_resnet_compare.py

    # Compare scaling modes
    python examples/jpc_fc_resnet_compare.py --scaling jpc --depth 10
    python examples/jpc_fc_resnet_compare.py --scaling fabricpc --depth 10

    # Test depth scaling
    for d in 8 16 32 64 128; do
      python examples/jpc_fc_resnet_compare.py --scaling jpc --depth $d
    done

    # Deep network with tuned hyperparams
    python examples/jpc_fc_resnet_compare.py --depth 30 --param_lr 0.0003 \\
        --eta_infer 0.1 --max_norm 0.1 --num_epochs 5

References:
    - jpc: https://github.com/thebuckleylab/jpc
    - Innocenti et al., "muPC: scaling for predictive coding networks"

Results
Architecture: FC-ResNet, depth=*, width=128
Inference: eta=0.2, steps=3*depth, max_norm=0.2
Training: lr=0.001, batch_size=256, epochs=3

| Depth | jpc scaling | fabricpc scaling | fabricpc_v2 scaling |
|-------|-------------|------------------|---------------------|
| 8     | ~90.8%      | ~92.6%           | ~91.6%              |
| 16    | ~87.8%      | ~85.6%           | ~88.4%              |
| 32    | ~85.2%      | ~43.4%           | ~86.8%              |
| 64    | ~83.5%      | ~11.5%           | ~85.4%              |
| 128   | ~83.9%      | ~10.3%           | (not tested)        |

Root cause of fabricpc collapse: skip_scale = 1/sqrt(2) ≈ 0.707 causes
exponential signal decay through the identity path. Over L layers, the
coherent signal decays as 0.707^L (SNR ~ 1e-5 at depth 32) while noise
from the linear path maintains O(1) variance. The network can't propagate
gradients through pure noise.

Fix (fabricpc_v2): skip_scale = 1.0 preserves the identity mapping.
hidden_scale includes a depth factor (gain/sqrt(N*L)) instead of in-degree
factor (gain/sqrt(N*K)), bounding variance growth to (1+1/L)^L ~ e.

"""

import argparse
import math
import time
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fabricpc.nodes import IdentityNode
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import IdentityActivation, ReLUActivation
from fabricpc.core.energy import GaussianEnergy
from fabricpc.core.inference import InferenceSGDNormClip
from fabricpc.core.initializers import NormalInitializer, initialize
from fabricpc.core.types import NodeParams
from fabricpc.training import train, evaluate
from fabricpc.utils.data.dataloader import MnistLoader
from fabricpc import setup_jax

setup_jax()
jax.config.update("jax_default_prng_impl", "threefry2x32")


# =============================================================================
# Scaling Factor Computation
# =============================================================================


def compute_scaling_factors(
    input_dim: int, width: int, depth: int, mode: str
) -> Tuple[float, float, float, float]:
    """
    Compute per-layer-type scaling factors.

    Args:
        input_dim: Input dimensionality (784 for MNIST).
        width: Hidden layer width.
        depth: Total number of parameterized layers.
        mode: 'jpc', 'fabricpc', or 'fabricpc_v2'.

    Returns:
        (in_scale, hidden_scale, out_scale, skip_scale)
    """
    if mode == "jpc":
        # jpc mupc.ipynb: exact formulas from Innocenti et al.
        # The depth factor in hidden_scale makes the linear path's contribution
        # O(1/L), so the unscaled skip dominates and variance stays ~O(1).
        in_scale = 1.0 / math.sqrt(input_dim)
        hidden_scale = 1.0 / math.sqrt(width * depth)
        skip_scale = 1.0
        out_scale = 1.0 / width
    elif mode == "fabricpc":
        # FabricPC muPC: a = gain / sqrt(fan_in * K)
        # Each PreActResBlock sums a linear path and an identity skip,
        # equivalent to an IdentityNode with in_degree K=2 in FabricPC's
        # graph. Both paths must be scaled by 1/sqrt(K) to preserve O(1)
        # variance at the summation point (see mupc.py:229).
        #
        # BUG: skip_scale = 1/sqrt(2) ≈ 0.707 causes exponential signal
        # decay through the identity path. Over L layers, the coherent
        # signal decays as 0.707^L while noise is maintained at O(1) by
        # the variance-preserving linear path. SNR collapses for L > 16.
        # See fabricpc_v2 mode for the fix.
        K = 2  # effective in_degree for residual blocks (linear + skip)
        relu_gain = math.sqrt(2.0)
        in_scale = 1.0 / math.sqrt(input_dim)
        hidden_scale = relu_gain / math.sqrt(width * K)
        skip_scale = 1.0 / math.sqrt(K)
        out_scale = 1.0 / width
    elif mode == "fabricpc_v2":
        # Fixed residual scaling: unattenuated skip + depth-compensated linear.
        #
        # The skip connection must remain at scale 1.0 to preserve the
        # identity mapping that carries the signal through deep networks.
        # The linear path is depth-compensated: its per-layer variance
        # contribution is O(gain²/(2L)), so total variance over L layers
        # grows as (1 + gain²/(2L))^L → sqrt(e) ≈ 1.65 — bounded.
        #
        # Includes activation-aware gain (sqrt(2) for ReLU) from Kaiming
        # convention, which JPC's raw 1/sqrt(N*L) omits.
        relu_gain = math.sqrt(2.0)
        in_scale = 1.0 / math.sqrt(input_dim)
        hidden_scale = relu_gain / math.sqrt(width * depth)
        skip_scale = 1.0
        out_scale = 1.0 / width
    else:
        raise ValueError(
            f"Unknown scaling mode: {mode!r}. "
            "Use 'jpc', 'fabricpc', or 'fabricpc_v2'."
        )

    return in_scale, hidden_scale, out_scale, skip_scale


# =============================================================================
# Custom Nodes — Pre-activation FC-ResNet blocks
# =============================================================================


class FCInputNode(NodeBase):
    """
    First layer of the FC-ResNet: z_mu = in_scale * (W @ x).

    No activation, no skip connection. Matches jpc's ScaledLinear input layer.
    """

    def __init__(
        self,
        shape,
        name,
        activation=IdentityActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(mean=0.0, std=1.0),
        scale=1.0,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            scale=scale,
            use_bias=False,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def get_variance_factor(source_shape, config, weight_init):
        return float(np.prod(source_shape))

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init=None, config=None):
        if config is None:
            config = {}
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=1.0)

        weights_dict = {}
        keys = jax.random.split(key, len(input_shapes))

        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_features = int(np.prod(in_shape))
            out_features = int(np.prod(node_shape))
            weight_shape = (in_features, out_features)
            weights_dict[edge_key] = initialize(keys[i], weight_shape, weight_init)

        return NodeParams(weights=weights_dict, biases={})

    @staticmethod
    def forward(params, inputs, state, node_info):
        config = node_info.node_config
        scale = config.get("scale", 1.0)
        batch_size = state.z_latent.shape[0]

        # Single input, flatten to (batch, input_dim)
        edge_key, x = next(iter(inputs.items()))
        x_flat = x.reshape(batch_size, -1)

        # z_mu = scale * (x @ W)
        W = params.weights[edge_key]
        z_mu = scale * jnp.matmul(x_flat, W)

        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        return state


class PreActResBlock(NodeBase):
    """
    Pre-activation residual block:
        z_mu = hidden_scale * (W @ act(x)) + skip_scale * x

    Matches jpc's ResNetBlock architecture:
    - Activation applied BEFORE linear transformation (pre-activation)
    - Identity skip connection adds raw input (NOT the activated/scaled version)

    The skip connection is internal to this node — it does NOT appear as a
    graph edge. This means there is one z_latent per block, matching jpc.

    In FabricPC's graph, this block decomposes into LinearNode -> SkipConnection.
    hidden_scale = gain/sqrt(fan_in*K) folds the linear
    forward scale with the summation scaling, and skip_scale = 1/sqrt(K)
    scales the identity path. Both are needed to preserve O(1) variance.
    """

    def __init__(
        self,
        shape,
        name,
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(mean=0.0, std=1.0),
        scale=1.0,
        skip_scale=1.0,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            scale=scale,
            skip_scale=skip_scale,
            use_bias=False,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def get_variance_factor(source_shape, config, weight_init):
        return float(source_shape[-1])

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init=None, config=None):
        if config is None:
            config = {}
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=1.0)

        weights_dict = {}
        keys = jax.random.split(key, len(input_shapes))

        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_features = in_shape[-1]
            out_features = node_shape[-1]
            weight_shape = (in_features, out_features)
            weights_dict[edge_key] = initialize(keys[i], weight_shape, weight_init)

        return NodeParams(weights=weights_dict, biases={})

    @staticmethod
    def forward(params, inputs, state, node_info):
        config = node_info.node_config
        scale = config.get("scale", 1.0)
        skip_scale = config.get("skip_scale", 1.0)
        activation = node_info.activation

        # Single input
        edge_key, x = next(iter(inputs.items()))
        W = params.weights[edge_key]

        # Pre-activation: apply activation BEFORE linear transform
        act_x = type(activation).forward(x, activation.config)

        # z_mu = scale * (act(x) @ W) + skip_scale * x
        # scale = gain/sqrt(fan_in*K) for the linear path
        # skip_scale = 1/sqrt(K) for the identity skip path (K = in_degree)
        z_mu = scale * jnp.matmul(act_x, W) + skip_scale * x

        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        return state


class PreActReadout(NodeBase):
    """
    Pre-activation output layer: z_mu = out_scale * (W @ act(x)).

    Matches jpc's Readout layer: pre-activation linear with scaling, no skip.
    """

    def __init__(
        self,
        shape,
        name,
        activation=ReLUActivation(),
        energy=GaussianEnergy(),
        latent_init=NormalInitializer(),
        weight_init=NormalInitializer(mean=0.0, std=1.0),
        scale=1.0,
    ):
        super().__init__(
            shape=shape,
            name=name,
            activation=activation,
            energy=energy,
            latent_init=latent_init,
            weight_init=weight_init,
            scale=scale,
            use_bias=False,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec(name="in", is_multi_input=False)}

    @staticmethod
    def get_variance_factor(source_shape, config, weight_init):
        return float(source_shape[-1])

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init=None, config=None):
        if config is None:
            config = {}
        if weight_init is None:
            weight_init = NormalInitializer(mean=0.0, std=1.0)

        weights_dict = {}
        keys = jax.random.split(key, len(input_shapes))

        for i, (edge_key, in_shape) in enumerate(input_shapes.items()):
            in_features = in_shape[-1]
            out_features = int(np.prod(node_shape))
            weight_shape = (in_features, out_features)
            weights_dict[edge_key] = initialize(keys[i], weight_shape, weight_init)

        return NodeParams(weights=weights_dict, biases={})

    @staticmethod
    def forward(params, inputs, state, node_info):
        config = node_info.node_config
        scale = config.get("scale", 1.0)
        activation = node_info.activation

        # Single input
        edge_key, x = next(iter(inputs.items()))
        W = params.weights[edge_key]

        # Pre-activation: apply activation BEFORE linear transform
        act_x = type(activation).forward(x, activation.config)

        # z_mu = scale * (act(x) @ W)   (no skip connection)
        z_mu = scale * jnp.matmul(act_x, W)

        error = state.z_latent - z_mu
        state = state._replace(z_mu=z_mu, error=error)

        node_class = node_info.node_class
        state = node_class.energy_functional(state, node_info)
        return state


# =============================================================================
# FC-ResNet Graph Builder
# =============================================================================


def build_fc_resnet(
    width,
    depth,
    scaling_mode,
    eta_infer,
    infer_steps,
    max_norm,
    input_dim=784,
    output_dim=10,
):
    """
    Build a fully-connected ResNet matching jpc's mupc.ipynb architecture.

    Architecture:
        input(input_dim) -> FCInput(width) -> PreActResBlock(width) x (depth-2)
                         -> PreActReadout(output_dim)

    Uses InferenceSGDNormClip for stable convergence with muPC scaling.
    jpc uses an adaptive ODE solver; FabricPC's fixed-step SGD needs gradient
    clipping to handle the stiff dynamics introduced by small scaling factors.

    Args:
        input_dim: Input dimensionality (784 for MNIST).
        width: Hidden layer width.
        depth: Total parameterized layers (input_linear + resblocks + readout).
        output_dim: Output dimensionality (10 for MNIST).
        scaling_mode: 'jpc', 'fabricpc', or 'fabricpc_v2'.
        eta_infer: Inference rate.
        infer_steps: Inference steps (default: 3*depth).
        max_norm: Maximum gradient norm for inference clipping.

    Returns:
        GraphStructure (no MuPCConfig — scaling is internal to nodes).
    """
    if infer_steps is None:
        infer_steps = 3 * depth

    in_scale, hidden_scale, out_scale, skip_scale = compute_scaling_factors(
        input_dim, width, depth, scaling_mode
    )

    weight_init = NormalInitializer(mean=0.0, std=1.0)

    # Input identity node (clamped to data)
    input_node = IdentityNode(shape=(input_dim,), name="input")

    # Layer 0: FCInputNode (input_dim -> width, no activation, no skip)
    layer_0 = FCInputNode(
        shape=(width,),
        name="layer_0",
        weight_init=weight_init,
        scale=in_scale,
    )

    all_nodes = [input_node, layer_0]
    all_edges = [Edge(source=input_node, target=layer_0.slot("in"))]

    # Layers 1 to depth-2: PreActResBlock (width -> width, ReLU, skip)
    prev = layer_0
    for i in range(1, depth - 1):
        block = PreActResBlock(
            shape=(width,),
            name=f"layer_{i}",
            weight_init=weight_init,
            scale=hidden_scale,
            skip_scale=skip_scale,
        )
        all_nodes.append(block)
        all_edges.append(Edge(source=prev, target=block.slot("in")))
        prev = block

    # Layer depth-1: PreActReadout (width -> output_dim, ReLU pre-act, no skip)
    readout = PreActReadout(
        shape=(output_dim,),
        name="output",
        weight_init=weight_init,
        scale=out_scale,
    )
    all_nodes.append(readout)
    all_edges.append(Edge(source=prev, target=readout.slot("in")))

    # Build graph — NO MuPCConfig (scaling is internal to nodes)
    structure = graph(
        nodes=all_nodes,
        edges=all_edges,
        task_map=TaskMap(x=input_node, y=readout),
        inference=InferenceSGDNormClip(
            eta_infer=eta_infer, infer_steps=infer_steps, max_norm=max_norm
        ),
    )

    return structure


# =============================================================================
# CLI and Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="jpc FC-ResNet comparison on MNIST")
    parser.add_argument(
        "--scaling",
        type=str,
        default="jpc",
        choices=["jpc", "fabricpc", "fabricpc_v2"],
        help="Scaling formula: 'jpc' (1/sqrt(N*L)), 'fabricpc' (gain/sqrt(N*K), K=2), "
        "or 'fabricpc_v2' (gain/sqrt(N*L), skip=1)",
    )
    parser.add_argument(
        "--depth", type=int, default=10, help="Total parameterized layers (default: 10)"
    )
    parser.add_argument(
        "--width", type=int, default=128, help="Hidden layer width (default: 128)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size (default: 64)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=3, help="Training epochs (default: 3)"
    )
    parser.add_argument(
        "--eta_infer", type=float, default=0.2, help="Inference rate (default: 0.2)"
    )
    parser.add_argument(
        "--param_lr", type=float, default=0.001, help="Parameter LR (default: 0.001)"
    )
    parser.add_argument(
        "--infer_steps",
        type=int,
        default=None,
        help="Inference steps (default: 3*depth)",
    )
    parser.add_argument(
        "--max_norm",
        type=float,
        default=0.2,
        help="Gradient norm clipping for inference (default: 0.2)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-epoch output")
    parser.add_argument(
        "--probe_variance",
        action="store_true",
        help="Print per-layer variance diagnostics after training",
    )
    return parser.parse_args()


def probe_variance(params, structure, test_loader, rng_key):
    """
    Measure per-layer variance diagnostics after inference converges.

    Prints a table showing z_mu variance, latent_grad variance, and
    z_latent variance at each layer, revealing signal propagation issues.
    """
    from fabricpc.graph_initialization.state_initializer import initialize_graph_state
    from fabricpc.core.inference import run_inference

    # Get one batch
    batch = next(iter(test_loader))
    x_batch, y_batch = batch
    x_batch = jnp.array(x_batch)
    y_batch = jnp.array(y_batch)
    batch_size = x_batch.shape[0]

    task_map = structure.task_map
    clamps = {task_map["x"]: x_batch, task_map["y"]: y_batch}

    state = initialize_graph_state(
        structure, batch_size, rng_key, clamps=clamps, params=params
    )
    final_state = run_inference(params, state, clamps, structure)

    # Collect per-layer stats
    node_order = list(structure.nodes.keys())
    print("\n" + "=" * 72)
    print("Per-Layer Variance Diagnostics (after inference)")
    print("=" * 72)
    print(
        f"{'Layer':<12} {'Var(z_mu)':>12} {'Var(z_lat)':>12} "
        f"{'Var(grad)':>12} {'Var(error)':>12}"
    )
    print("-" * 72)

    for node_name in node_order:
        ns = final_state.nodes[node_name]
        z_mu_var = float(jnp.var(ns.z_mu))
        z_lat_var = float(jnp.var(ns.z_latent))
        grad_var = float(jnp.var(ns.latent_grad))
        error_var = float(jnp.var(ns.error))
        print(
            f"{node_name:<12} {z_mu_var:>12.4f} {z_lat_var:>12.4f} "
            f"{grad_var:>12.4f} {error_var:>12.4f}"
        )

    print("=" * 72)


def main():
    args = parse_args()

    print("=" * 60)
    print("jpc FC-ResNet Comparison on MNIST")
    print("=" * 60)
    print(f"Scaling mode: {args.scaling}")
    print(f"Architecture: FC-ResNet, depth={args.depth}, width={args.width}")
    steps = args.infer_steps or 3 * args.depth
    print(
        f"Inference: eta={args.eta_infer}, steps={steps}, " f"max_norm={args.max_norm}"
    )
    print(
        f"Training: lr={args.param_lr}, batch_size={args.batch_size}, "
        f"epochs={args.num_epochs}"
    )

    master_rng_key = jax.random.PRNGKey(42)
    graph_key, train_key, eval_key = jax.random.split(master_rng_key, 3)

    # Build model
    structure = build_fc_resnet(
        input_dim=784,
        width=args.width,
        depth=args.depth,
        output_dim=10,
        scaling_mode=args.scaling,
        eta_infer=args.eta_infer,
        infer_steps=args.infer_steps,
        max_norm=args.max_norm,
    )
    params = initialize_params(structure, graph_key)

    # Print scaling factors
    in_s, hid_s, out_s, skip_s = compute_scaling_factors(
        784, args.width, args.depth, args.scaling
    )
    print(f"\nScaling factors ({args.scaling}):")
    print(f"  input layer:   {in_s:.6f}")
    print(f"  hidden layers: {hid_s:.6f}")
    print(f"  skip path:     {skip_s:.6f}")
    print(f"  output layer:  {out_s:.6f}")

    # Print model summary
    print(f"\nModel: {len(structure.nodes)} nodes, {len(structure.edges)} edges")
    total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"Total parameters: {total_params:,}")

    # Data — MNIST, flattened, with one-hot labels for MSE
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

    # Train — Adam (not AdamW) matching jpc
    optimizer = optax.adam(args.param_lr)
    train_config = {"num_epochs": args.num_epochs}

    print(
        f"\nTraining for {args.num_epochs} epoch(s) "
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
    print(f"Training time: {elapsed:.1f}s")

    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate(trained_params, structure, test_loader, train_config, eval_key)
    accuracy = metrics["accuracy"] * 100
    print(f"Test Accuracy: {accuracy:.2f}%")

    # Comparison
    print("\n--- Comparison ---")
    print("jpc reference:   ~93% (depth=30, width=128, ODE solver)")
    print(
        f"This run:        {accuracy:.2f}% (depth={args.depth}, "
        f"width={args.width}, scaling={args.scaling})"
    )

    # Variance diagnostics
    if args.probe_variance:
        probe_key = jax.random.PRNGKey(99)
        probe_variance(trained_params, structure, test_loader, probe_key)


if __name__ == "__main__":
    main()
