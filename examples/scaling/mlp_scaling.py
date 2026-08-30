"""
MLP Scaling Law Experiments for FabricPC
========================================

Measures training time and GPU memory usage versus model width and depth
for MLP architectures using both Predictive Coding (PC) and Backpropagation
training modes.

Outputs:
- CSV file with results
- Console summary table
- Plotly plots saved as PNG (requires plotly and kaleido + Chrome browser)

Expected runtime: ~30 minutes on a high-end consumer GPU (RTX 4090 class)

How it works:
  - Parent process iterates over (width, depth, mode) grid
  - For each experiment, spawns python mlp_scaling.py --run-single --width X --depth Y ...
  - Subprocess gets fresh JAX memory pool, runs experiment, outputs JSON
  - Parent parses JSON and collects ScalingResult objects

# TODO:
Investigate node parallelization via vmap
potential strategy during inference phase and weight learning update phase (independent over nodes):
- vmap the nodes within each the device
- shard the batch over multiple devices (if available) for data parallelism

  Current Implementation:
  - Nodes are processed sequentially in a Python for-loop (inference.py:63-84)
  - No vmap at the node level
  - Multi-GPU uses data parallelism via the mesh argument, sharding each batch over the "data" mesh axis
  - Topological ordering is computed and stored (structure.node_order) but not actively exploited

  Node Independence in Predictive Coding:
  - Nodes within the same topological level are independent during initialization forward pass
  - Nodes are independent in gradient computation and reduction dependencies on outneighbor nodes only
  - Source nodes (in_degree == 0) are always independent
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"

import jax
import jax.numpy as jnp
import time
import csv
import subprocess
import sys
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params, FeedforwardStateInit
from fabricpc.core.activations import IdentityActivation, SigmoidActivation
from fabricpc.core.inference import InferenceSGD
import optax
from fabricpc.training import make_train_step

# Reproducibility
jax.config.update("jax_default_prng_impl", "threefry2x32")

# =============================================================================
# Configuration
# =============================================================================

# Model configurations to sweep
MLP_WIDTHS = [
    128,
    256,
]  # 512, 1024, 2048, 4096]  # hidden layer widths
MLP_DEPTHS = [
    4,
    8,
    16,
]  # 32, 64, 128]  # number of layers after input

# Fixed parameters
BATCH_SIZE = 256
NUM_TRAINING_STEPS = 15
NUM_WARMUP_STEPS = 2  # Ignore for timing (JIT compilation)
INFER_STEPS = 10  # Reduced for faster iteration

# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ScalingResult:
    """Result from a single scaling experiment."""

    architecture: str
    training_mode: str
    width: int
    depth: int
    num_params: int
    avg_step_time_ms: float
    memory_mb: float
    batch_size: int
    infer_steps: int


# =============================================================================
# Utility Functions
# =============================================================================


def create_mlp_model(
    input_dim: int,
    hidden_width: int,
    num_layers: int,
    output_dim: int,
    rng_key: jax.Array,
):
    """
    Create MLP model with specified width and depth.

    Architecture: input -> [hidden x num_layers] -> output
    Uses Linear with sigmoid activations.

    Returns:
        (params, structure) tuple
    """
    input_node = Linear(
        shape=(input_dim,), activation=IdentityActivation(), name="input"
    )

    nodes = [input_node]
    edges = []
    prev_node = input_node

    for i in range(num_layers):
        hidden = Linear(
            shape=(hidden_width,), activation=SigmoidActivation(), name=f"hidden_{i}"
        )
        nodes.append(hidden)
        edges.append(Edge(source=prev_node, target=hidden.slot("in")))
        prev_node = hidden

    output_node = Linear(
        shape=(output_dim,), activation=SigmoidActivation(), name="output"
    )
    nodes.append(output_node)
    edges.append(Edge(source=prev_node, target=output_node.slot("in")))

    structure = graph(
        nodes=nodes,
        edges=edges,
        task_map=TaskMap(x=input_node, y=output_node),
        graph_state_initializer=FeedforwardStateInit(),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=INFER_STEPS),
    )
    params = initialize_params(structure, rng_key)
    return params, structure


def generate_synthetic_data(
    rng_key: jax.Array,
    batch_size: int,
    input_dim: int,
    output_dim: int,
    num_batches: int,
) -> List[Dict[str, jnp.ndarray]]:
    """Generate random data batches for MLP training."""
    keys = jax.random.split(rng_key, num_batches * 2)
    batches = []
    for i in range(num_batches):
        x = jax.random.normal(keys[2 * i], (batch_size, input_dim))
        # One-hot encoded random labels
        labels = jax.random.randint(keys[2 * i + 1], (batch_size,), 0, output_dim)
        y = jax.nn.one_hot(labels, output_dim)
        batches.append({"x": x, "y": y})
    return batches


def get_memory_bytes() -> int:
    """Get current GPU memory usage in bytes.

    Note: We use bytes_in_use (current) instead of peak_bytes_in_use because
    peak includes temporary XLA compilation buffers that don't reflect
    actual runtime memory usage and can vary unpredictably with model size.
    """
    # Force synchronization
    jax.block_until_ready(jnp.zeros(1))

    device = jax.local_devices()[0]
    if hasattr(device, "memory_stats"):
        try:
            stats = device.memory_stats()
            return stats.get("bytes_in_use", 0)
        except Exception:
            pass
    return 0  # Fallback for CPU or unavailable stats


def run_timed_training_pc(
    params,
    structure,
    batches: List[Dict],
    train_config: dict,
    rng_key: jax.Array,
    num_steps: int,
    num_warmup: int,
) -> Tuple[float, int]:
    """Run PC training steps with timing, handling JIT warmup separately."""
    optimizer = optax.adam(train_config.get("lr", 1e-3))
    opt_state = optimizer.init(params)

    # JIT compile the training step
    step = make_train_step(structure, optimizer)

    keys = jax.random.split(rng_key, num_steps + num_warmup)

    # Warmup (JIT compilation happens here)
    for i in range(num_warmup):
        batch = batches[i % len(batches)]
        params, opt_state, _, _ = step(params, opt_state, batch, keys[i])
    jax.block_until_ready(params)  # Force sync

    # Measure memory after warmup (accurate in isolated subprocess)
    memory = get_memory_bytes()

    # Timed runs
    start_time = time.perf_counter()
    for i in range(num_steps):
        batch = batches[(i + num_warmup) % len(batches)]
        params, opt_state, _, _ = step(params, opt_state, batch, keys[i + num_warmup])
    jax.block_until_ready(params)  # Ensure all computation complete
    end_time = time.perf_counter()

    avg_step_time = (end_time - start_time) / num_steps

    return avg_step_time, memory


def run_timed_training_backprop(
    params,
    structure,
    batches: List[Dict],
    train_config: dict,
    rng_key: jax.Array,
    num_steps: int,
    num_warmup: int,
) -> Tuple[float, int]:
    """Run backprop training steps with timing, handling JIT warmup separately."""
    optimizer = optax.adam(train_config.get("lr", 1e-3))
    opt_state = optimizer.init(params)

    # JIT compile the training step
    step = make_train_step(structure, optimizer, algorithm="backprop")

    keys = jax.random.split(rng_key, num_steps + num_warmup)

    # Warmup (JIT compilation happens here)
    for i in range(num_warmup):
        batch = batches[i % len(batches)]
        params, opt_state, _, _ = step(params, opt_state, batch, keys[i])
    jax.block_until_ready(params)  # Force sync

    # Measure memory after warmup (accurate in isolated subprocess)
    memory = get_memory_bytes()

    # Timed runs
    start_time = time.perf_counter()
    for i in range(num_steps):
        batch = batches[(i + num_warmup) % len(batches)]
        params, opt_state, _, _ = step(params, opt_state, batch, keys[i + num_warmup])
    jax.block_until_ready(params)  # Ensure all computation complete
    end_time = time.perf_counter()

    avg_step_time = (end_time - start_time) / num_steps

    return avg_step_time, memory


def count_params(params) -> int:
    """Count total number of parameters."""
    return sum(p.size for p in jax.tree_util.tree_leaves(params))


# =============================================================================
# Experiment Runner (Subprocess-based for memory isolation)
# =============================================================================


def run_single_experiment(
    width: int,
    depth: int,
    mode: str,
    seed: int,
    batch_size: int,
    num_steps: int,
    num_warmup: int,
    infer_steps: int,
) -> ScalingResult:
    """
    Run a single experiment in the current process.

    This should be called in an isolated subprocess for accurate memory measurement.
    """
    rng_key = jax.random.PRNGKey(seed)

    # Create model
    rng_key, graph_key, data_key, train_key = jax.random.split(rng_key, 4)

    params, structure = create_mlp_model(width, width, depth, width, graph_key)
    num_params = count_params(params)

    # Generate data
    batches = generate_synthetic_data(
        data_key, batch_size, width, width, num_steps + num_warmup
    )

    # Training config
    train_config = {
        "lr": 1e-3,
    }

    # Run timed training based on mode
    if mode == "pc":
        avg_time, mem = run_timed_training_pc(
            params, structure, batches, train_config, train_key, num_steps, num_warmup
        )
    else:
        avg_time, mem = run_timed_training_backprop(
            params, structure, batches, train_config, train_key, num_steps, num_warmup
        )

    return ScalingResult(
        architecture="mlp",
        training_mode=mode,
        width=width,
        depth=depth,
        num_params=num_params,
        avg_step_time_ms=avg_time * 1000,
        memory_mb=mem / (1024 * 1024),
        batch_size=batch_size,
        infer_steps=infer_steps,
    )


def run_experiment_subprocess(
    width: int,
    depth: int,
    mode: str,
    seed: int,
    batch_size: int,
    num_steps: int,
    num_warmup: int,
    infer_steps: int,
) -> ScalingResult:
    """
    Run a single experiment in a subprocess for memory isolation.

    Each subprocess gets a fresh JAX memory pool, ensuring accurate
    memory measurement per experiment.
    """
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--run-single",
        "--width",
        str(width),
        "--depth",
        str(depth),
        "--mode",
        mode,
        "--seed",
        str(seed),
        "--batch-size",
        str(batch_size),
        "--num-steps",
        str(num_steps),
        "--num-warmup",
        str(num_warmup),
        "--infer-steps",
        str(infer_steps),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Handle subprocess failure
        error_msg = result.stderr.strip() if result.stderr else "Unknown error"
        print(f"FAILED: {error_msg[:50]}")
        return ScalingResult(
            architecture="mlp",
            training_mode=mode,
            width=width,
            depth=depth,
            num_params=0,
            avg_step_time_ms=float("nan"),
            memory_mb=float("nan"),
            batch_size=batch_size,
            infer_steps=infer_steps,
        )

    # Parse JSON output from subprocess
    try:
        data = json.loads(result.stdout.strip())
        return ScalingResult(**data)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"FAILED: Could not parse result: {e}")
        return ScalingResult(
            architecture="mlp",
            training_mode=mode,
            width=width,
            depth=depth,
            num_params=0,
            avg_step_time_ms=float("nan"),
            memory_mb=float("nan"),
            batch_size=batch_size,
            infer_steps=infer_steps,
        )


def run_all_experiments(
    widths: List[int],
    depths: List[int],
    modes: List[str],
    batch_size: int,
    num_steps: int,
    num_warmup: int,
    infer_steps: int,
    base_seed: int = 42,
) -> List[ScalingResult]:
    """
    Run all experiments via subprocesses for memory isolation.

    Each experiment runs in a fresh subprocess with its own JAX memory pool.
    """
    results = []
    total = len(widths) * len(depths) * len(modes)
    idx = 0

    for mode in modes:
        print()
        print("=" * 70)
        mode_name = "Predictive Coding" if mode == "pc" else "Backpropagation"
        print(f"Running MLP Scaling ({mode_name} Mode)")
        print("=" * 70)

        for depth in depths:
            for width in widths:
                idx += 1
                # Use different seed for each experiment
                seed = base_seed + idx
                print(
                    f"  [{idx}/{total}] mode={mode}, depth={depth}, width={width}",
                    end=" ",
                    flush=True,
                )

                result = run_experiment_subprocess(
                    width=width,
                    depth=depth,
                    mode=mode,
                    seed=seed,
                    batch_size=batch_size,
                    num_steps=num_steps,
                    num_warmup=num_warmup,
                    infer_steps=infer_steps,
                )
                results.append(result)

                if result.num_params > 0:
                    print(
                        f"params={result.num_params:,}, time={result.avg_step_time_ms:.2f}ms, mem={result.memory_mb:.1f}MB"
                    )

    return results


def parse_single_experiment_args() -> Optional[argparse.Namespace]:
    """Parse CLI arguments for single experiment mode."""
    parser = argparse.ArgumentParser(description="MLP Scaling Experiment")
    parser.add_argument(
        "--run-single", action="store_true", help="Run single experiment mode"
    )
    parser.add_argument("--width", type=int, help="Hidden layer width")
    parser.add_argument("--depth", type=int, help="Number of hidden layers")
    parser.add_argument(
        "--mode", type=str, choices=["pc", "backprop"], help="Training mode"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--batch-size", type=int, help="Batch size")
    parser.add_argument("--num-steps", type=int, help="Number of training steps")
    parser.add_argument("--num-warmup", type=int, help="Number of warmup steps")
    parser.add_argument("--infer-steps", type=int, help="PC inference steps")

    args = parser.parse_args()

    if args.run_single:
        return args
    return None


# =============================================================================
# Output Functions
# =============================================================================


def save_results_csv(results: List[ScalingResult], filename: str):
    """Save results to CSV file."""
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "architecture",
                "training_mode",
                "width",
                "depth",
                "num_params",
                "avg_step_time_ms",
                "memory_mb",
                "batch_size",
                "infer_steps",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    print(f"Results saved to: {filename}")


def print_summary_table(results: List[ScalingResult]):
    """Print a formatted summary table."""
    print("\nSummary Table:")
    print("-" * 95)
    print(
        f"{'Mode':<10} | {'Width':<7} | {'Depth':<6} | {'Params':<12} | {'Time (ms)':<12} | {'Memory (MB)':<12}"
    )
    print("-" * 95)

    for r in results:
        if r.num_params > 0:  # Skip failed experiments
            print(
                f"{r.training_mode:<10} | {r.width:<7} | {r.depth:<6} | {r.num_params:<12,} | {r.avg_step_time_ms:<12.2f} | {r.memory_mb:<12.1f}"
            )
    print("-" * 95)


def plot_results(results: List[ScalingResult], output_dir: str = "."):
    """Generate scaling law plots using Plotly."""
    try:
        import plotly.express as px
        import pandas as pd
    except ImportError:
        print("plotly/pandas not available, skipping plots")
        return

    df = pd.DataFrame([asdict(r) for r in results if r.num_params > 0])

    if df.empty:
        print("No valid results to plot")
        return

    # Create a combined label for legend grouping
    df["mode_depth"] = df["training_mode"] + ", d=" + df["depth"].astype(str)
    df["mode_width"] = df["training_mode"] + ", w=" + df["width"].astype(str)

    # Symbol mapping for training modes
    symbol_map = {"pc": "circle", "backprop": "square"}

    # Plot 1: Training Time vs Width (by depth)
    fig1 = px.line(
        df,
        x="width",
        y="avg_step_time_ms",
        color="depth",
        symbol="training_mode",
        symbol_map=symbol_map,
        markers=True,
        line_dash="training_mode",
        line_dash_map={"pc": "solid", "backprop": "dash"},
        title="Training Time vs Width",
        labels={
            "width": "Width (hidden units)",
            "avg_step_time_ms": "Step Time (ms)",
            "depth": "Depth",
            "training_mode": "Mode",
        },
        log_x=True,
    )
    fig1.update_layout(
        title_font_size=16,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig1.update_xaxes(dtick=1, tickformat="d")
    path1 = os.path.join(output_dir, "mlp_scaling_time_vs_width.png")
    fig1.write_image(path1, scale=2)
    print(f"Plot saved to: {path1}")

    # Plot 2: GPU Memory vs Width (by depth)
    fig2 = px.line(
        df,
        x="width",
        y="memory_mb",
        color="depth",
        symbol="training_mode",
        symbol_map=symbol_map,
        markers=True,
        line_dash="training_mode",
        line_dash_map={"pc": "solid", "backprop": "dash"},
        title="GPU Memory vs Width",
        labels={
            "width": "Width (hidden units)",
            "memory_mb": "Memory (MB)",
            "depth": "Depth",
            "training_mode": "Mode",
        },
        log_x=True,
    )
    fig2.update_layout(
        title_font_size=16,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig2.update_xaxes(dtick=1, tickformat="d")
    path2 = os.path.join(output_dir, "mlp_scaling_memory_vs_width.png")
    fig2.write_image(path2, scale=2)
    print(f"Plot saved to: {path2}")

    # Plot 3: Training Time vs Depth (by width)
    fig3 = px.line(
        df,
        x="depth",
        y="avg_step_time_ms",
        color="width",
        symbol="training_mode",
        symbol_map=symbol_map,
        markers=True,
        line_dash="training_mode",
        line_dash_map={"pc": "solid", "backprop": "dash"},
        title="Training Time vs Depth",
        labels={
            "depth": "Depth (layers)",
            "avg_step_time_ms": "Step Time (ms)",
            "width": "Width",
            "training_mode": "Mode",
        },
    )
    fig3.update_layout(
        title_font_size=16,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    path3 = os.path.join(output_dir, "mlp_scaling_time_vs_depth.png")
    fig3.write_image(path3, scale=2)
    print(f"Plot saved to: {path3}")

    # Plot 4: GPU Memory vs Depth (by width)
    fig4 = px.line(
        df,
        x="depth",
        y="memory_mb",
        color="width",
        symbol="training_mode",
        symbol_map=symbol_map,
        markers=True,
        line_dash="training_mode",
        line_dash_map={"pc": "solid", "backprop": "dash"},
        title="GPU Memory vs Depth",
        labels={
            "depth": "Depth (layers)",
            "memory_mb": "Memory (MB)",
            "width": "Width",
            "training_mode": "Mode",
        },
    )
    fig4.update_layout(
        title_font_size=16,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    path4 = os.path.join(output_dir, "mlp_scaling_memory_vs_depth.png")
    fig4.write_image(path4, scale=2)
    print(f"Plot saved to: {path4}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all scaling experiments using subprocess isolation."""
    print("=" * 70)
    print("FabricPC MLP Scaling Law Experiments")
    print("=" * 70)

    # Print device info
    devices = jax.devices()
    print(f"Device: {devices[0]}")
    print(f"JAX version: {jax.__version__}")
    print()

    # Configuration summary
    print(f"Widths: {MLP_WIDTHS}")
    print(f"Depths: {MLP_DEPTHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Training steps: {NUM_TRAINING_STEPS} (+ {NUM_WARMUP_STEPS} warmup)")
    print(f"PC inference steps: {INFER_STEPS}")
    print()
    print(
        "Note: Each experiment runs in isolated subprocess for accurate memory measurement."
    )

    # Run all experiments via subprocesses
    all_results = run_all_experiments(
        widths=MLP_WIDTHS,
        depths=MLP_DEPTHS,
        modes=["pc", "backprop"],
        batch_size=BATCH_SIZE,
        num_steps=NUM_TRAINING_STEPS,
        num_warmup=NUM_WARMUP_STEPS,
        infer_steps=INFER_STEPS,
        base_seed=42,
    )

    # Output results
    print()
    print("=" * 70)
    print("Results")
    print("=" * 70)

    # Get output directory (same as script location)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(output_dir, "mlp_scaling_results.csv")

    save_results_csv(all_results, csv_path)
    print_summary_table(all_results)
    plot_results(all_results, output_dir)

    print()
    print("Experiment complete!")


if __name__ == "__main__":
    # Check if running in single-experiment subprocess mode
    args = parse_single_experiment_args()

    if args is not None:
        # Subprocess mode: run single experiment and output JSON
        result = run_single_experiment(
            width=args.width,
            depth=args.depth,
            mode=args.mode,
            seed=args.seed,
            batch_size=args.batch_size,
            num_steps=args.num_steps,
            num_warmup=args.num_warmup,
            infer_steps=args.infer_steps,
        )
        # Output result as JSON for parent process to parse
        print(json.dumps(asdict(result)))
    else:
        # Normal mode: orchestrate all experiments
        main()
