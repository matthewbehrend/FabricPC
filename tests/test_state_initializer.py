#!/usr/bin/env python3
"""
Test suite for the State Initializer system.

Tests distribution-based init, feedforward init, clamp handling,
and the zero-error invariant of feedforward initialization.
"""

import pytest
import jax
import jax.numpy as jnp

from fabricpc.nodes import Linear
from fabricpc.nodes.transformer import TransformerBlock
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.inference import InferenceSGD, run_inference
from fabricpc.core.activations import (
    IdentityActivation,
    ReLUActivation,
    SoftmaxActivation,
    GeluActivation,
)
from fabricpc.core.initializers import NormalInitializer
from fabricpc.graph_initialization.state_initializer import (
    GlobalStateInit,
    NodeDistributionStateInit,
    FeedforwardStateInit,
    initialize_graph_state,
)


@pytest.fixture
def simple_graph_structure(rng_key):
    """Simple 3-layer graph structure for testing."""
    input_node = Linear(shape=(784,), name="input")
    hidden_node = Linear(shape=(128,), activation=ReLUActivation(), name="hidden")
    output_node = Linear(shape=(10,), name="output")

    structure = graph(
        nodes=[input_node, hidden_node, output_node],
        edges=[
            Edge(source=input_node, target=hidden_node.slot("in")),
            Edge(source=hidden_node, target=output_node.slot("in")),
        ],
        task_map=TaskMap(x=input_node, y=output_node),
        inference=InferenceSGD(),
    )
    return structure


class TestDistributionStateInit:
    """Test suite for GlobalStateInit."""

    def test_distribution_init_graph_level_config(
        self, simple_graph_structure, rng_key
    ):
        """Test distribution init with graph-level default initializer."""
        structure = simple_graph_structure

        batch_size = 8
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=GlobalStateInit(initializer=NormalInitializer(std=0.1)),
        )

        assert state.batch_size == batch_size
        assert "input" in state.nodes
        assert "hidden" in state.nodes
        assert "output" in state.nodes

        assert state.nodes["input"].z_latent.shape == (batch_size, 784)
        assert state.nodes["hidden"].z_latent.shape == (batch_size, 128)
        assert state.nodes["output"].z_latent.shape == (batch_size, 10)

        hidden_std = jnp.std(state.nodes["hidden"].z_latent)
        assert hidden_std > 0.05 and hidden_std < 0.2


class TestFeedforwardStateInit:
    """Test suite for FeedforwardStateInit."""

    def test_feedforward_init_requires_params(self, simple_graph_structure, rng_key):
        """Test that feedforward init raises error without params."""
        structure = simple_graph_structure

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        with pytest.raises(ValueError, match="requires params"):
            initialize_graph_state(
                structure,
                batch_size,
                rng_key,
                clamps,
                state_init=FeedforwardStateInit(),
                params=None,
            )

    def test_feedforward_init_with_params(self, simple_graph_structure, rng_key):
        """Test feedforward init propagates through network."""
        structure = simple_graph_structure
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 784))
        y = jax.random.normal(rng_key, (batch_size, 10))
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        assert state.batch_size == batch_size
        assert state.nodes["input"].z_latent.shape == (batch_size, 784)
        assert state.nodes["hidden"].z_latent.shape == (batch_size, 128)
        assert state.nodes["output"].z_latent.shape == (batch_size, 10)

        assert not jnp.allclose(state.nodes["hidden"].z_latent, 0.0)


class TestClampHandling:
    """Test clamp handling in state initialization."""

    def test_distribution_init_respects_clamps(self, simple_graph_structure, rng_key):
        """Test that distribution init respects clamped values."""
        structure = simple_graph_structure

        batch_size = 4
        x = jnp.ones((batch_size, 784)) * 5.0
        y = jnp.ones((batch_size, 10)) * -3.0
        clamps = {"input": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=NodeDistributionStateInit(),
        )

        assert jnp.allclose(state.nodes["input"].z_latent, x)
        assert jnp.allclose(state.nodes["output"].z_latent, y)


class TestFeedforwardZeroError:
    """Test that feedforward initialization produces zero error at all nodes."""

    def test_feedforward_zero_error_mlp(self, rng_key):
        """Test that feedforward init produces zero error for MLP architecture."""
        input_node = Linear(shape=(32,), name="input")
        h1_node = Linear(shape=(64,), activation=ReLUActivation(), name="h1")
        h2_node = Linear(shape=(32,), activation=ReLUActivation(), name="h2")
        output_node = Linear(shape=(10,), activation=SoftmaxActivation(), name="output")

        structure = graph(
            nodes=[input_node, h1_node, h2_node, output_node],
            edges=[
                Edge(source=input_node, target=h1_node.slot("in")),
                Edge(source=h1_node, target=h2_node.slot("in")),
                Edge(source=h2_node, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 32))
        clamps = {"input": x}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        for node_name in structure.nodes:
            error = state.nodes[node_name].error
            assert jnp.allclose(
                error, 0.0, atol=1e-6
            ), f"Node {node_name} has non-zero error after feedforward init: max={jnp.max(jnp.abs(error))}"

            if node_name not in clamps:
                z_latent = state.nodes[node_name].z_latent
                z_mu = state.nodes[node_name].z_mu
                assert jnp.allclose(
                    z_latent, z_mu, atol=1e-6
                ), f"Node {node_name}: z_latent != z_mu after feedforward init"

    def test_feedforward_zero_error_transformer(self, rng_key):
        """Test that feedforward init produces zero error for transformer architecture."""
        seq_len = 8
        embed_dim = 16
        vocab_size = 10

        input_node = Linear(
            shape=(seq_len, vocab_size),
            activation=IdentityActivation(),
            name="input",
        )
        embed_node = Linear(
            shape=(seq_len, embed_dim),
            activation=IdentityActivation(),
            name="embed",
        )
        mask_node = Linear(
            shape=(1, seq_len, seq_len),
            activation=IdentityActivation(),
            name="mask",
        )
        transformer_node = TransformerBlock(
            shape=(seq_len, embed_dim),
            num_heads=2,
            ff_dim=32,
            internal_activation=GeluActivation(),
            rope_theta=100.0,
            name="transformer_0",
        )
        output_node = Linear(
            shape=(seq_len, vocab_size),
            activation=SoftmaxActivation(),
            name="output",
        )

        structure = graph(
            nodes=[input_node, embed_node, mask_node, transformer_node, output_node],
            edges=[
                Edge(source=input_node, target=embed_node.slot("in")),
                Edge(source=embed_node, target=transformer_node.slot("in")),
                Edge(source=mask_node, target=transformer_node.slot("mask")),
                Edge(source=transformer_node, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node, causal_mask=mask_node),
            inference=InferenceSGD(),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 2
        x_indices = jax.random.randint(rng_key, (batch_size, seq_len), 0, vocab_size)
        x = jax.nn.one_hot(x_indices, vocab_size)

        causal_mask = jnp.tril(jnp.ones((1, seq_len, seq_len)))
        causal_mask = jnp.broadcast_to(causal_mask, (batch_size, 1, seq_len, seq_len))

        clamps = {"input": x, "mask": causal_mask}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        for node_name in structure.nodes:
            error = state.nodes[node_name].error
            max_error = jnp.max(jnp.abs(error))
            assert jnp.allclose(
                error, 0.0, atol=1e-5
            ), f"Node {node_name} has non-zero error after feedforward init: max={max_error}"

            if node_name not in clamps:
                z_latent = state.nodes[node_name].z_latent
                z_mu = state.nodes[node_name].z_mu
                assert jnp.allclose(
                    z_latent, z_mu, atol=1e-5
                ), f"Node {node_name}: z_latent != z_mu after feedforward init"

    def test_feedforward_no_change_after_inference_without_output_clamp(self, rng_key):
        """
        Test that inference with no output clamp does not change latent states
        when error is zero after feedforward init.
        """
        input_node = Linear(shape=(16,), name="input")
        hidden_node = Linear(shape=(32,), activation=ReLUActivation(), name="hidden")
        output_node = Linear(shape=(8,), name="output")

        structure = graph(
            nodes=[input_node, hidden_node, output_node],
            edges=[
                Edge(source=input_node, target=hidden_node.slot("in")),
                Edge(source=hidden_node, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGD(eta_infer=0.1, infer_steps=10),
        )

        params = initialize_params(structure, rng_key)

        batch_size = 2
        x = jax.random.normal(rng_key, (batch_size, 16))
        clamps = {"input": x}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        original_latents = {
            name: state.nodes[name].z_latent for name in structure.nodes
        }

        final_state = run_inference(params, state, clamps, structure)

        for node_name in structure.nodes:
            original = original_latents[node_name]
            final = final_state.nodes[node_name].z_latent
            max_diff = jnp.max(jnp.abs(original - final))
            assert jnp.allclose(
                original, final, atol=1e-5
            ), f"Node {node_name} changed after inference despite zero error: max_diff={max_diff}"


def _two_source_graph():
    """Two sources (one clamped, one an unclamped prior) -> hidden -> output."""
    inp = Linear(shape=(12,), name="inp")
    prior = Linear(shape=(6,), name="prior")
    hidden = Linear(shape=(8,), activation=ReLUActivation(), name="hidden")
    output = Linear(shape=(4,), name="output")
    return graph(
        nodes=[inp, prior, hidden, output],
        edges=[
            Edge(source=inp, target=hidden.slot("in")),
            Edge(source=prior, target=hidden.slot("in")),
            Edge(source=hidden, target=output.slot("in")),
        ],
        task_map=TaskMap(x=inp, y=output),
        inference=InferenceSGD(),
    )


class TestSourceZmuInvariant:
    """The shared post-pass in initialize_graph_state assigns z_mu <- z_latent
    (cast to z_mu's float dtype) for every in_degree == 0 node, so
    error = z_latent - z_mu = 0 holds at init under every initializer."""

    @pytest.mark.parametrize(
        "state_init",
        [
            GlobalStateInit(initializer=NormalInitializer(std=0.1)),
            NodeDistributionStateInit(),
            FeedforwardStateInit(),
        ],
        ids=["global", "node_distribution", "feedforward"],
    )
    def test_source_zmu_mirrors_latent(self, state_init, rng_key):
        structure = _two_source_graph()
        params = initialize_params(structure, rng_key)
        batch_size = 4
        x = jax.random.normal(rng_key, (batch_size, 12))
        y = jax.random.normal(jax.random.PRNGKey(1), (batch_size, 4))
        clamps = {"inp": x, "output": y}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=state_init,
            params=params,
        )

        # Both the clamped source and the unclamped prior leave init with
        # z_mu mirroring z_latent and zero error.
        for name in ("inp", "prior"):
            node_state = state.nodes[name]
            assert jnp.array_equal(
                node_state.z_mu, node_state.z_latent.astype(node_state.z_mu.dtype)
            ), f"{name}: z_mu != z_latent at init"
            assert jnp.all(node_state.error == 0), f"{name}: error != 0 at init"

    def test_int_clamped_source_zmu_stays_float(self, rng_key):
        """An int source clamp mirrors into z_mu cast to float, keeping the
        float-only carry invariant."""
        structure = _two_source_graph()
        params = initialize_params(structure, rng_key)
        batch_size = 4
        x = jnp.arange(batch_size * 12, dtype=jnp.int32).reshape(batch_size, 12)
        clamps = {"inp": x}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=GlobalStateInit(initializer=NormalInitializer(std=0.1)),
            params=params,
        )

        node_state = state.nodes["inp"]
        assert node_state.z_latent.dtype == jnp.int32
        assert jnp.issubdtype(node_state.z_mu.dtype, jnp.floating)
        assert jnp.array_equal(node_state.z_mu, x.astype(node_state.z_mu.dtype))
        assert jnp.all(node_state.error == 0)


def _cycle_graph(unroll):
    """x -> a <-> b -> y with the 2-node cycle unrolled `unroll` times."""
    from fabricpc.core.activations import TanhActivation
    from fabricpc.nodes.identity import IdentityNode

    w_init = NormalInitializer(std=0.1)
    x = IdentityNode(shape=(6,), name="x")
    a = Linear(shape=(8,), name="a", activation=TanhActivation(), weight_init=w_init)
    b = Linear(shape=(8,), name="b", activation=TanhActivation(), weight_init=w_init)
    y = Linear(
        shape=(4,), name="y", activation=IdentityActivation(), weight_init=w_init
    )
    return graph(
        nodes=[x, a, b, y],
        edges=[
            Edge(source=x, target=a.slot("in")),
            Edge(source=a, target=b.slot("in")),
            Edge(source=b, target=a.slot("in")),
            Edge(source=b, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=1),
        unroll=unroll,
    )


class TestFeedforwardThroughCycles:
    """FeedforwardStateInit pass 2 walks structure.schedule, so cyclic graphs
    get true feedforward initialization through the cycle."""

    def test_cycle_feedforward_matches_manual_replay(self, rng_key):
        structure = _cycle_graph(unroll=2)
        params = initialize_params(structure, rng_key)
        batch_size = 3
        x = jax.random.normal(rng_key, (batch_size, 6))
        clamps = {"x": x}

        state = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=FeedforwardStateInit(),
            params=params,
        )

        # Pass 1 of FeedforwardStateInit draws each node's latent_init with
        # the same rng split NodeDistributionStateInit uses, so that
        # initializer reproduces the pre-propagation latents.
        pass1 = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps,
            state_init=NodeDistributionStateInit(),
            params=params,
        )

        z = {name: pass1.nodes[name].z_latent for name in structure.nodes}
        for node_name in structure.schedule:
            info = structure.nodes[node_name].node_info
            if info.in_degree == 0:
                continue
            inputs = {ek: z[structure.edges[ek].source] for ek in info.in_edges}
            projected = info.node_class.forward(
                params.nodes[node_name], inputs, pass1.nodes[node_name], info
            )
            if node_name not in clamps:
                z[node_name] = projected.z_mu

        for node_name in structure.nodes:
            assert jnp.allclose(
                state.nodes[node_name].z_latent, z[node_name], atol=1e-6
            ), f"{node_name}: feedforward init != manual schedule replay"

    def test_unroll_degree_changes_init(self, rng_key):
        """U=2 propagates one more traversal through the cycle than U=1, so
        the initialized latents differ (propagation proof)."""
        batch_size = 3
        x = jax.random.normal(rng_key, (batch_size, 6))
        clamps = {"x": x}

        states = {}
        for unroll in (1, 2):
            structure = _cycle_graph(unroll=unroll)
            params = initialize_params(structure, rng_key)
            states[unroll] = initialize_graph_state(
                structure,
                batch_size,
                rng_key,
                clamps,
                state_init=FeedforwardStateInit(),
                params=params,
            )

        for node_name in ("a", "y"):
            assert not jnp.allclose(
                states[1].nodes[node_name].z_latent,
                states[2].nodes[node_name].z_latent,
            ), f"{node_name}: U=2 init should differ from U=1"
