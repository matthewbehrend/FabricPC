"""
Transformer components for JAX predictive coding networks.

Decomposed transformer pipeline — each stage is a separate PC node:

  tokens → Embedding → MhaResidual(+) → LnMlp1 → Mlp2Residual(+) → VocabProjection → logits
                        │ (skip)   ↑              │ (skip)    ↑
                        └──────────┘              └───────────┘
"""

from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp

from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.core.types import NodeParams, NodeState, NodeInfo
from fabricpc.core.initializers import (
    InitializerBase,
    initialize,
    NormalInitializer,
    XavierInitializer,
    KaimingInitializer,
)
from fabricpc.core.positional import precompute_freqs_cis, apply_rotary_emb
from fabricpc.utils.helpers import layernorm
from fabricpc.core.activations import (
    GeluActivation,
    IdentityActivation,
    SoftmaxActivation,
)
from fabricpc.core.energy import CrossEntropyEnergy, GaussianEnergy

# ==============================================================================
# EMBEDDING NODE
# ==============================================================================


class EmbeddingNode(NodeBase):
    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape,
        name,
        vocab_size,
        embed_dim,
        weight_init=NormalInitializer(std=0.02),
        latent_init=NormalInitializer(),
        energy=GaussianEnergy(),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            weight_init=weight_init,
            latent_init=latent_init,
            energy=energy,
            **kwargs,
        )

    @staticmethod
    def get_slots() -> Dict[str, SlotSpec]:
        # Discrete token indices, not a continuous signal — keep muPC scaling off.
        return {
            "in": SlotSpec(name="in", is_multi_input=False, is_variance_scalable=False)
        }

    @staticmethod
    def initialize_params(
        key: jax.Array,
        node_shape: Tuple[int, ...],
        input_shapes: Dict[str, Tuple[int, ...]],
        weight_init: InitializerBase,
        config: Dict[str, Any],
    ) -> NodeParams:
        vocab_size = config["vocab_size"]
        embed_dim = config["embed_dim"]

        weights = {"embeddings": initialize(key, (vocab_size, embed_dim), weight_init)}
        return NodeParams(weights=weights, biases={})

    @staticmethod
    def predict(
        params: NodeParams,
        inputs: Dict[str, jnp.ndarray],
        state: NodeState,
        node_info: NodeInfo,
    ) -> Tuple[jnp.ndarray, None]:
        edge_key = list(inputs.keys())[0]
        indices = inputs[edge_key]

        if indices.ndim == 3 and indices.shape[-1] == 1:
            indices = jnp.squeeze(indices, axis=-1)

        # Token indices must reach this node as an integer dtype.
        # Source-node clamps now propagate dtype through state init, so callers
        # should clamp with int (e.g. jnp.int32), not float.
        z_mu = params.weights["embeddings"][indices]
        return z_mu, None

    @staticmethod
    def forward_and_latent_grads(params, inputs, state, node_info, is_clamped=False):
        new_state = node_info.node_class.forward(params, inputs, state, node_info)
        # Discrete indices: no gradient flows back through the input edge.
        input_grads = {
            edge_key: jnp.zeros_like(inp) for edge_key, inp in inputs.items()
        }
        # Self-grad anchors z_latent to the lookup z_mu. Computed explicitly
        # because this override skips the base autodiff path.
        energy_obj = node_info.energy
        self_grad = type(energy_obj).grad_latent(
            new_state.z_latent, new_state.z_mu, energy_obj.config
        )
        return new_state, input_grads, self_grad


# ==============================================================================
# MULTI-HEAD ATTENTION + RESIDUAL NODE
# ==============================================================================


class MhaResidualNode(NodeBase):
    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape,
        name,
        embed_dim,
        num_heads,
        use_rope=True,
        rope_theta=10000.0,
        is_causal=True,
        weight_init=XavierInitializer(),
        latent_init=NormalInitializer(),
        energy=GaussianEnergy(),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_rope=use_rope,
            rope_theta=rope_theta,
            is_causal=is_causal,
            weight_init=weight_init,
            latent_init=latent_init,
            energy=energy,
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {
            "in": SlotSpec("in", False),
            "skip": SlotSpec(
                "skip", False, is_variance_scalable=False, is_skip_connection=True
            ),
        }

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        dim = config["embed_dim"]
        keys = jax.random.split(key, 6)

        def init_w(k, s):
            return initialize(k, s, weight_init)

        weights = {
            "ln_gamma": jnp.ones((dim,)),
            "W_q": init_w(keys[0], (dim, dim)),
            "W_k": init_w(keys[1], (dim, dim)),
            "W_v": init_w(keys[2], (dim, dim)),
            "W_o": init_w(keys[3], (dim, dim)),
        }
        biases = {
            "ln_beta": jnp.zeros((dim,)),
            "b_q": jnp.zeros((dim,)),
            "b_k": jnp.zeros((dim,)),
            "b_v": jnp.zeros((dim,)),
            "b_o": jnp.zeros((dim,)),
        }
        return NodeParams(weights, biases)

    @staticmethod
    def predict(params, inputs, state, node_info):
        x = inputs[next(k for k in inputs if k.endswith(":in"))]
        skip_key = next((k for k in inputs if k.endswith(":skip")), None)
        skip = inputs[skip_key] if skip_key else x

        cfg = node_info.node_config
        B, L, D = x.shape
        num_heads = cfg["num_heads"]
        head_dim = D // num_heads

        x_norm = layernorm(x, params.weights["ln_gamma"], params.biases["ln_beta"])

        def proj(h, w_name, b_name):
            return jnp.dot(h, params.weights[w_name]) + params.biases[b_name]

        Q = proj(x_norm, "W_q", "b_q").reshape(B, L, num_heads, head_dim)
        K = proj(x_norm, "W_k", "b_k").reshape(B, L, num_heads, head_dim)
        V = proj(x_norm, "W_v", "b_v").reshape(B, L, num_heads, head_dim)

        if cfg.get("use_rope"):
            freqs_cis = precompute_freqs_cis(
                head_dim, L, theta=cfg.get("rope_theta", 10000.0)
            )
            Q, K = apply_rotary_emb(Q, K, freqs_cis)

        Q, K, V = (
            Q.transpose(0, 2, 1, 3),
            K.transpose(0, 2, 1, 3),
            V.transpose(0, 2, 1, 3),
        )
        scores = jnp.matmul(Q, K.swapaxes(-1, -2)) / jnp.sqrt(head_dim)

        if cfg.get("is_causal", True):
            causal_mask = jnp.tril(jnp.ones((L, L)))
            scores = jnp.where(causal_mask == 0, -1e9, scores)

        attn = jax.nn.softmax(scores, axis=-1)
        mha = jnp.matmul(attn, V).transpose(0, 2, 1, 3).reshape(B, L, D)
        mha = proj(mha, "W_o", "b_o")

        z_mu = skip + mha
        return z_mu, None


# ==============================================================================
# LAYERNORM + MLP1 NODE
# ==============================================================================


class LnMlp1Node(NodeBase):
    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = GeluActivation

    def __init__(
        self,
        shape,
        name,
        embed_dim,
        ff_dim,
        activation=GeluActivation(),
        weight_init=KaimingInitializer(),
        latent_init=NormalInitializer(),
        energy=GaussianEnergy(),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            activation=activation,
            weight_init=weight_init,
            latent_init=latent_init,
            energy=energy,
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec("in", False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        embed_dim, ff_dim = config["embed_dim"], config["ff_dim"]
        keys = jax.random.split(key, 2)

        weights = {
            "ln_gamma": jnp.ones((embed_dim,)),
            "W_ff1": initialize(keys[0], (embed_dim, ff_dim), weight_init),
        }
        biases = {"ln_beta": jnp.zeros((embed_dim,)), "b_ff1": jnp.zeros((ff_dim,))}
        return NodeParams(weights, biases)

    @staticmethod
    def predict(params, inputs, state, node_info):
        x = inputs[list(inputs.keys())[0]]
        x_norm = layernorm(x, params.weights["ln_gamma"], params.biases["ln_beta"])
        h = jnp.dot(x_norm, params.weights["W_ff1"]) + params.biases["b_ff1"]

        act_obj = node_info.activation
        z_mu = type(act_obj).forward(h, act_obj.config)
        return z_mu, None


# ==============================================================================
# MLP2 + RESIDUAL NODE
# ==============================================================================


class Mlp2ResidualNode(NodeBase):
    DEFAULT_ENERGY = GaussianEnergy
    DEFAULT_ACTIVATION = IdentityActivation

    def __init__(
        self,
        shape,
        name,
        embed_dim,
        ff_dim,
        weight_init=XavierInitializer(),
        latent_init=NormalInitializer(),
        energy=GaussianEnergy(),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            weight_init=weight_init,
            latent_init=latent_init,
            energy=energy,
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {
            "in": SlotSpec("in", False),
            "residual": SlotSpec(
                "residual", False, is_variance_scalable=False, is_skip_connection=True
            ),
        }

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        embed_dim, ff_dim = config["embed_dim"], config["ff_dim"]
        weights = {"W_ff2": initialize(key, (ff_dim, embed_dim), weight_init)}
        return NodeParams(weights, {"b_ff2": jnp.zeros((embed_dim,))})

    @staticmethod
    def predict(params, inputs, state, node_info):
        mlp1_in = next(val for key, val in inputs.items() if key.endswith(":in"))
        res_in = next(val for key, val in inputs.items() if key.endswith(":residual"))

        mlp2 = jnp.dot(mlp1_in, params.weights["W_ff2"]) + params.biases["b_ff2"]
        z_mu = res_in + mlp2
        return z_mu, None


# ==============================================================================
# VOCAB PROJECTION NODE
# ==============================================================================


class VocabProjectionNode(NodeBase):
    DEFAULT_ENERGY = CrossEntropyEnergy
    DEFAULT_ACTIVATION = SoftmaxActivation

    def __init__(
        self,
        shape,
        name,
        vocab_size,
        embed_dim,
        activation=SoftmaxActivation(),
        weight_init=XavierInitializer(),
        latent_init=NormalInitializer(),
        energy=CrossEntropyEnergy(),
        **kwargs,
    ):
        super().__init__(
            shape=shape,
            name=name,
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            activation=activation,
            weight_init=weight_init,
            latent_init=latent_init,
            energy=energy,
            **kwargs,
        )

    @staticmethod
    def get_slots():
        return {"in": SlotSpec("in", False)}

    @staticmethod
    def initialize_params(key, node_shape, input_shapes, weight_init, config):
        vocab, dim = config["vocab_size"], config["embed_dim"]
        weights = {"W_out": initialize(key, (dim, vocab), weight_init)}
        return NodeParams(weights, {"b_out": jnp.zeros((vocab,))})

    @staticmethod
    def predict(params, inputs, state, node_info):
        x = inputs[list(inputs.keys())[0]]
        logits = jnp.dot(x, params.weights["W_out"]) + params.biases["b_out"]

        act_obj = node_info.activation
        z_mu = type(act_obj).forward(logits, act_obj.config)
        return z_mu, None
