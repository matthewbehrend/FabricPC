"""
Graph-level integration test for the conv/pool nodes.

Unlike test_convolutional.py / test_pooling.py (which exercise node methods in
isolation), this builds a real predictive-coding graph
(input -> conv -> maxpool -> linear), initializes parameters, and runs a couple
of full train steps (inference to convergence + local weight update). It guards
the conv/pool/muPC wiring end-to-end — the kind of test that would have caught
the non-functional ``slots=`` parameter, since a broken slot contract surfaces
only when the graph is actually assembled and run.
"""

import jax
import jax.numpy as jnp
import optax

from fabricpc.nodes import ConvNode, MaxPool, Linear, IdentityNode
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.activations import ReLUActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.training import make_train_step


def _build_structure():
    """input(8,8,1) -> conv(8,8,4) -> maxpool(4,4,4) -> linear(3)."""
    pixels = IdentityNode(shape=(8, 8, 1), name="pixels")
    conv = ConvNode(
        shape=(8, 8, 4),
        kernel_size=(3, 3),
        stride=(1, 1),
        padding="SAME",
        activation=ReLUActivation(),
        name="conv",
    )
    pool = MaxPool(
        shape=(4, 4, 4),
        window_shape=(2, 2),
        stride=(2, 2),
        padding="VALID",
        name="pool",
    )
    out = Linear(
        shape=(3,),
        activation=SoftmaxActivation(),
        energy=CrossEntropyEnergy(),
        flatten_input=True,
        name="out",
    )
    structure = graph(
        nodes=[pixels, conv, pool, out],
        edges=[
            Edge(source=pixels, target=conv.slot("in")),
            Edge(source=conv, target=pool.slot("in")),
            Edge(source=pool, target=out.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=out),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=5),
    )
    return structure


def test_conv_pool_linear_param_shapes(rng_key):
    """Parameters initialize with the shapes the architecture implies."""
    structure = _build_structure()
    params = initialize_params(structure, rng_key)

    # Conv kernel: (kH, kW, C_in, C_out) = (3, 3, 1, 4); bias (1, 1, 1, 4).
    conv_w = next(iter(params.nodes["conv"].weights.values()))
    assert conv_w.shape == (3, 3, 1, 4)
    assert params.nodes["conv"].biases["b"].shape == (1, 1, 1, 4)

    # Pooling is parameter-free.
    assert params.nodes["pool"].weights == {}
    assert params.nodes["pool"].biases == {}

    # Linear flattens the (4, 4, 4) pool output -> 64 -> 3.
    out_w = next(iter(params.nodes["out"].weights.values()))
    assert out_w.shape == (64, 3)


def test_conv_pool_linear_trains(rng_key):
    """A couple of train steps run, energy stays finite, and weights update."""
    structure = _build_structure()
    params = initialize_params(structure, rng_key)

    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)

    batch_size = 8
    key_x, key_y = jax.random.split(rng_key)
    labels = jax.nn.one_hot(jax.random.randint(key_y, (batch_size,), 0, 3), 3)
    batch = {
        "x": jax.random.normal(key_x, (batch_size, 8, 8, 1)),
        "y": labels,
    }

    step = make_train_step(structure, optimizer)
    p, os = params, opt_state
    energies = []
    for _ in range(3):
        p, os, metrics, final_state = step(p, os, batch, rng_key)
        energies.append(metrics["energy"])

    # Energy is finite at every step (no NaN/Inf through conv/pool/inference).
    for e in energies:
        assert jnp.isfinite(e)

    # The conv weights actually moved — the learning signal reached the conv node
    # through the pooling node (this is what guards the conv/pool wiring).
    conv_before = next(iter(params.nodes["conv"].weights.values()))
    conv_after = next(iter(p.nodes["conv"].weights.values()))
    assert not jnp.allclose(conv_before, conv_after)

    # Final inference state has the right per-node latent shapes.
    assert final_state.nodes["conv"].z_latent.shape == (batch_size, 8, 8, 4)
    assert final_state.nodes["pool"].z_latent.shape == (batch_size, 4, 4, 4)
    assert final_state.nodes["out"].z_latent.shape == (batch_size, 3)


def test_avgpool_preserves_variance_into_the_head(rng_key):
    """conv -> global AvgPool -> head: the head's input stays O(1).

    Global average pooling over n spatial cells divides a sum of n terms by n.
    Without the 1/n variance factor the pool reports, muPC leaves its in-edge
    at scale 1.0 and the head sees a variance reduced by up to n — 1/16 for
    the 4x4 map here, and 1/64 for the 8x8 map a stride-1 stem would produce.
    """
    from fabricpc.nodes import AvgPool
    from fabricpc.core.mupc import MuPCConfig
    from fabricpc.core.initializers import MuPCInitializer
    from fabricpc.graph_initialization.state_initializer import initialize_graph_state

    pixels = IdentityNode(shape=(8, 8, 3), name="pixels")
    conv = ConvNode(
        shape=(4, 4, 32),
        kernel_size=(3, 3),
        stride=(2, 2),
        padding="SAME",
        activation=ReLUActivation(),
        weight_init=MuPCInitializer(),
        name="conv",
    )
    pool = AvgPool(shape=(32,), name="pool", global_pool=True)
    out = Linear(shape=(3,), name="out", weight_init=MuPCInitializer())

    structure = graph(
        nodes=[pixels, conv, pool, out],
        edges=[
            Edge(source=pixels, target=conv.slot("in")),
            Edge(source=conv, target=pool.slot("in")),
            Edge(source=pool, target=out.slot("in")),
        ],
        task_map=TaskMap(x=pixels, y=out),
        inference=InferenceSGD(eta_infer=0.1, infer_steps=5),
        scaling=MuPCConfig(),
    )

    # 4x4 spatial map collapsed to (32,): n = 16, so the edge scale is sqrt(16).
    pool_scale = structure.nodes["pool"].node_info.scaling_config.forward_scale[
        "conv->pool:in"
    ]
    assert abs(pool_scale - 4.0) < 1e-10

    batch = 128
    params = initialize_params(structure, rng_key)
    x_data = jax.random.normal(rng_key, (batch, 8, 8, 3))
    state = initialize_graph_state(
        structure, batch, rng_key, clamps={"x": x_data}, params=params
    )
    var_conv = float(jnp.var(state.nodes["conv"].z_mu))
    var_pool = float(jnp.var(state.nodes["pool"].z_mu))
    # Spatially correlated features make the realized reduction milder than
    # 1/n, so sqrt(n) can over-correct; the guard is that the pool no longer
    # divides the variance reaching the head by a depth-independent factor.
    assert var_pool > 0.3 * var_conv, f"conv {var_conv} -> pool {var_pool}"
