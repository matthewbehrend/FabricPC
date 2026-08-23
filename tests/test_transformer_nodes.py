"""
Test suite for EmbeddingNode and decomposed TransformerV2 node functionality in FabricPC.
"""

import os

# Set JAX to CPU to avoid potential OOM on small test runners
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import (
    initialize_graph_state,
    FeedforwardStateInit,
)
from fabricpc.core.inference import run_inference, InferenceSGDNormClip
from fabricpc.training import evaluate, make_train_step
import optax
from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph

from fabricpc.nodes.transformer_v2 import (
    EmbeddingNode,
    MhaResidualNode,
    LnMlp1Node,
    Mlp2ResidualNode,
)
from fabricpc.models import create_deep_transformer


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


class TestEmbeddingNode:

    @pytest.fixture
    def embedding_graph(self, rng_key):
        """Creates a simple graph: Input (Linear) -> Embedding -> Output (Linear)"""
        vocab_size = 100
        embed_dim = 8
        seq_len = 5

        input_node = Linear(shape=(seq_len,), name="indices")
        embed_node = EmbeddingNode(
            shape=(seq_len, embed_dim),
            name="embed",
            vocab_size=vocab_size,
            embed_dim=embed_dim,
        )
        output_node = Linear(shape=(seq_len, 10), name="output")

        structure = graph(
            nodes=[input_node, embed_node, output_node],
            edges=[
                Edge(source=input_node, target=embed_node.slot("in")),
                Edge(source=embed_node, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=5, max_norm=5.0),
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    def test_registration_and_creation(self, embedding_graph):
        """Test that the node initializes params correctly."""
        params, structure = embedding_graph

        embed_params = params.nodes["embed"]
        assert "embeddings" in embed_params.weights

        # Check Shape: (vocab_size, embed_dim)
        expected_shape = (100, 8)
        assert embed_params.weights["embeddings"].shape == expected_shape

        # Check biases
        assert len(embed_params.biases) == 0

    def test_forward_lookup(self, embedding_graph, rng_key):
        """Test that z_mu correctly retrieves embeddings."""
        params, structure = embedding_graph

        batch_size = 3
        seq_len = 5

        input_indices = jax.random.randint(rng_key, (batch_size, seq_len), 0, 100)
        dummy_y = jnp.zeros((batch_size, seq_len, 10))

        clamps = {"indices": input_indices, "output": dummy_y}
        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        W = params.nodes["embed"].weights["embeddings"]

        expected_vectors = W[input_indices]  # (batch, seq, embed_dim)

        embed_state = state.nodes["embed"]

        # Verify z_mu matches lookup
        assert jnp.allclose(embed_state.z_mu, expected_vectors, atol=1e-5)

        # Verify shape
        assert embed_state.z_mu.shape == (batch_size, seq_len, 8)

    def test_gradient_blocking(self, embedding_graph, rng_key):
        """
        Critical Test: Ensure forward_and_latent_grads returns 0 gradients for inputs.
        Discrete inputs cannot receive gradients.
        """
        params, structure = embedding_graph
        batch_size = 2

        input_indices = jnp.ones((batch_size, 5), dtype=jnp.int32)
        clamps = {"indices": input_indices}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        node_info = structure.nodes["embed"].node_info
        node_state = state.nodes["embed"]
        node_params = params.nodes["embed"]

        # Inputs gathered from the "indices" node
        inputs = {"indices->embed:in": input_indices}

        # Call forward_and_latent_grads directly
        new_state, input_grads, self_grad = EmbeddingNode.forward_and_latent_grads(
            node_params, inputs, node_state, node_info
        )

        assert jnp.all(self_grad == 0.0), "Embedding node must return zero self-grad"
        assert self_grad.shape == node_state.z_latent.shape

        # Check input gradients
        grad = input_grads["indices->embed:in"]

        assert jnp.all(
            grad == 0.0
        ), "Embedding node must return zero gradients for inputs"
        assert grad.shape == input_indices.shape

    def test_learning_updates_embeddings(self, embedding_graph, rng_key):
        """
        Test that training actually updates the embedding matrix.
        PC Error (z_latent - z_mu) should drive updates to W[indices].
        """
        params, structure = embedding_graph

        optimizer = optax.sgd(1.0)
        opt_state = optimizer.init(params)

        batch_size = 1
        idx = 5

        # Token indices flow through as int32; state init propagates the dtype
        # to z_mu/error/etc. on the source node so the loop is type-consistent.
        input_indices = jnp.full((batch_size, 5), idx, dtype=jnp.int32)

        batch = {"x": input_indices, "y": jnp.zeros((batch_size, 5, 10))}

        # Snapshot old weights
        old_embeddings = params.nodes["embed"].weights["embeddings"]
        old_row_5 = old_embeddings[idx].copy()
        old_row_0 = old_embeddings[0].copy()

        # Run one training step
        rng_key, step_key = jax.random.split(rng_key)
        step = make_train_step(structure, optimizer)
        new_params, _, _, final_state = step(params, opt_state, batch, step_key)

        new_embeddings = new_params.nodes["embed"].weights["embeddings"]
        new_row_5 = new_embeddings[idx]
        new_row_0 = new_embeddings[0]

        # The row corresponding to the input index SHOULD change
        diff = jnp.abs(new_row_5 - old_row_5)
        assert jnp.max(diff) > 1e-5, "Embedding weights for active index did not update"

        # The row corresponding to unused index SHOULD NOT change
        diff_unused = jnp.abs(new_row_0 - old_row_0)
        assert (
            jnp.max(diff_unused) < 1e-6
        ), "Embedding weights for inactive index changed unexpectedly"

    def test_forward_squeeze_logic(self, embedding_graph, rng_key):
        """Test the logic handling (batch, seq, 1) inputs."""
        params, structure = embedding_graph

        # Create input with extra dimension (batch, seq, 1)
        input_expanded = jnp.zeros((2, 5, 1), dtype=jnp.int32)
        inputs = {"mock_edge": input_expanded}

        # Clamp the source with int indices so FeedforwardStateInit's forward
        # pass through EmbeddingNode receives integer upstream z_latent.
        clamps = {"indices": jnp.zeros((2, 5), dtype=jnp.int32)}
        state = initialize_graph_state(
            structure, 2, rng_key, clamps=clamps, params=params
        ).nodes["embed"]
        node_params = params.nodes["embed"]
        node_info = structure.nodes["embed"].node_info

        # Should not crash and should squeeze internally
        new_state = EmbeddingNode.forward(node_params, inputs, state, node_info)

        assert new_state.z_mu.shape == (2, 5, 8)


class TestTransformerBlock:

    @pytest.fixture
    def single_block_graph(self, rng_key):
        """Build a single MHA+MLP block: input -> mha -> mlp1 -> mlp2 -> output."""
        seq_len = 10
        embed_dim = 32
        ff_dim = 64

        input_node = Linear(shape=(seq_len, embed_dim), name="input")
        mha = MhaResidualNode(
            shape=(seq_len, embed_dim),
            name="mha",
            embed_dim=embed_dim,
            num_heads=4,
            use_rope=True,
        )
        mlp1 = LnMlp1Node(
            shape=(seq_len, ff_dim),
            name="mlp1",
            embed_dim=embed_dim,
            ff_dim=ff_dim,
        )
        mlp2 = Mlp2ResidualNode(
            shape=(seq_len, embed_dim),
            name="mlp2",
            embed_dim=embed_dim,
            ff_dim=ff_dim,
        )
        output_node = Linear(shape=(seq_len, embed_dim), name="output")

        structure = graph(
            nodes=[input_node, mha, mlp1, mlp2, output_node],
            edges=[
                Edge(source=input_node, target=mha.slot("in")),
                Edge(source=mha, target=mlp1.slot("in")),
                Edge(source=mlp1, target=mlp2.slot("in")),
                Edge(source=mha, target=mlp2.slot("residual")),
                Edge(source=mlp2, target=output_node.slot("in")),
            ],
            task_map=TaskMap(x=input_node, y=output_node),
            inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=5, max_norm=5.0),
        )
        params = initialize_params(structure, rng_key)
        return params, structure

    def test_block_forward_shapes(self, single_block_graph, rng_key):
        """Verify output shapes are preserved through Attention and MLP."""
        params, structure = single_block_graph
        batch_size = 2

        # Random inputs
        x = jax.random.normal(rng_key, (batch_size, 10, 32))
        clamps = {"input": x, "output": jnp.zeros_like(x)}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        final_state = run_inference(params, state, clamps, structure)

        block_latent = final_state.nodes["mlp2"].z_latent
        assert block_latent.shape == (batch_size, 10, 32)
        assert jnp.abs(block_latent).mean() > 0.0

    def test_causal_masking(self, single_block_graph, rng_key):
        """Verify future tokens do not affect past tokens."""
        params, structure = single_block_graph
        batch_size = 1

        x_base = jax.random.normal(rng_key, (batch_size, 10, 32))
        clamps_base = {"input": x_base, "output": jnp.zeros_like(x_base)}

        state_1 = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps=clamps_base,
            state_init=FeedforwardStateInit(),
            params=params,
        )
        out_1 = state_1.nodes["mha"].z_mu

        # Modified run: Change ONLY the last token
        x_mod = x_base.at[:, -1, :].add(5.0)
        clamps_mod = {"input": x_mod, "output": jnp.zeros_like(x_base)}

        state_2 = initialize_graph_state(
            structure,
            batch_size,
            rng_key,
            clamps=clamps_mod,
            state_init=FeedforwardStateInit(),
            params=params,
        )
        out_2 = state_2.nodes["mha"].z_mu

        # Check First Token (Should be Identical - Masking Working)
        diff_first = jnp.abs(out_1[:, 0, :] - out_2[:, 0, :]).max()
        assert diff_first < 1e-5, f"Causal mask failed! Past changed by {diff_first}"

        # Check Last Token (Should Change - Self Attention Working)
        diff_last = jnp.abs(out_1[:, -1, :] - out_2[:, -1, :]).max()
        assert (
            diff_last > 1e-4
        ), "Self-attention failed! Last token ignored input change."

    def test_block_learning(self, single_block_graph, rng_key):
        """Verify gradients propagate and loss decreases (overfitting test)."""
        params, structure = single_block_graph
        optimizer = optax.adam(0.01)
        opt_state = optimizer.init(params)

        target = jax.random.normal(rng_key, (4, 10, 32))
        batch = {"x": target, "y": target}

        step = make_train_step(structure, optimizer)
        losses = []
        for _ in range(5):
            rng_key, step_key = jax.random.split(rng_key)
            params, opt_state, metrics, _ = step(params, opt_state, batch, step_key)
            losses.append(metrics["energy"])

        assert losses[-1] < losses[0], "Transformer block failed to learn."

    def test_factory_structure(self, rng_key):
        """Verify create_deep_transformer generates correct graph topology."""
        depth = 3
        structure = create_deep_transformer(
            depth=depth,
            embed_dim=16,
            num_heads=2,
            mlp_dim=32,
            seq_len=10,
            vocab_size=10,
            inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=2, max_norm=5.0),
        )

        # Decomposed architecture per layer: MhaResidualNode, LnMlp1Node, Mlp2ResidualNode
        # Nodes: input_ids + embed + depth*(mha + mlp1 + mlp2) + logits
        assert len(structure.nodes) == 3 + 3 * depth  # depth=3 -> 12

        # Edges per layer (5): mha<-in, mha<-skip,
        #   mlp1<-mha:in, mlp2<-mlp1:in, mlp2<-mha:residual
        # Plus 2 global: input_ids->embed:in, last_mlp2->logits:in
        assert len(structure.edges) == 2 + 5 * depth  # depth=3 -> 17

        # Assert the per-block skip/mask wiring by intent, not just by count,
        # so a future topology change is caught structurally rather than arithmetically.
        def has_edge(target, slot):
            return any(
                e.target == target and e.slot == slot for e in structure.edges.values()
            )

        for i in range(depth):
            mha = f"L{i}_mha"
            assert has_edge(mha, "in"), f"{mha} missing 'in' edge"
            assert has_edge(mha, "skip"), f"{mha} missing residual 'skip' edge"
            assert has_edge(
                f"L{i}_mlp2", "residual"
            ), f"L{i}_mlp2 missing 'residual' edge"

        # Check node types by name. The input_ids source node has no fixed
        # position in node_order, so assert by lookup rather than by index.
        assert structure.nodes["input_ids"].node_info.node_type == "Linear"
        assert structure.nodes["embed"].node_info.node_type == "EmbeddingNode"
        assert structure.nodes["logits"].node_info.node_type == "VocabProjectionNode"

        # Each block contributes MhaResidualNode -> LnMlp1Node -> Mlp2ResidualNode,
        # in that topological order.
        block_types = [
            structure.nodes[n].node_info.node_type
            for n in structure.node_order
            if n.startswith("L")
        ]
        for i in range(depth):
            assert block_types[i * 3] == "MhaResidualNode"
            assert block_types[i * 3 + 1] == "LnMlp1Node"
            assert block_types[i * 3 + 2] == "Mlp2ResidualNode"

    def test_deep_network_inference(self, rng_key):
        """Integration test: Build deep network via factory and run data through it."""
        vocab_size = 50
        seq_len = 8
        embed_dim = 16

        structure = create_deep_transformer(
            depth=2,
            embed_dim=embed_dim,
            num_heads=4,
            mlp_dim=32,
            seq_len=seq_len,
            vocab_size=vocab_size,
            inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=5, max_norm=5.0),
        )
        params = initialize_params(structure, rng_key)

        batch_size = 2
        x_indices = jax.random.randint(rng_key, (batch_size, seq_len), 0, vocab_size)
        y_dummy = jnp.zeros((batch_size, seq_len, vocab_size))
        clamps = {"input_ids": x_indices, "logits": y_dummy}

        state = initialize_graph_state(
            structure, batch_size, rng_key, clamps=clamps, params=params
        )
        final_state = run_inference(params, state, clamps, structure)

        # Check signal reached the end
        output = final_state.nodes["logits"].z_mu

        assert output.shape == (batch_size, seq_len, vocab_size)
        assert jnp.abs(output).mean() > 0.0


class TestEvaluateTransformer:

    def test_smoke(self, rng_key):
        """evaluate on a transformer graph returns the CE-target key set with
        finite values."""
        vocab_size = 20
        seq_len = 6
        embed_dim = 16

        structure = create_deep_transformer(
            depth=1,
            embed_dim=embed_dim,
            num_heads=2,
            mlp_dim=32,
            seq_len=seq_len,
            vocab_size=vocab_size,
            inference=InferenceSGDNormClip(eta_infer=0.1, infer_steps=5, max_norm=5.0),
        )
        params = initialize_params(structure, rng_key)

        batch_size = 4
        x_data = jax.random.randint(rng_key, (batch_size, seq_len), 0, vocab_size)
        y_data = jax.random.randint(rng_key, (batch_size, seq_len), 0, vocab_size)

        test_loader = [{"x": x_data, "y": y_data}]

        metrics = evaluate(params, structure, test_loader, {}, rng_key)

        assert set(metrics.keys()) == {
            "accuracy",
            "cross_entropy",
            "perplexity",
            "energy",
            "target_energy",
        }
        for k, v in metrics.items():
            assert np.isfinite(v), f"{k} is not finite: {v}"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main(["-v", __file__]))
