"""
Audit tests for the predict/pair/energy node contract.

The error pair (``pair_error``/``pair_latent``) and the assembly templates
(``forward``, ``forward_with_aux``, ``forward_from_error``) are base-owned:
the ePC <-> sPC equivalence requires one volume-preserving bijection between
error and z_latent shared by every node, and Python cannot mark methods
final, so this audit enforces it. Also pins the template source guard (the
one owner of in_degree == 0 semantics) and the aux surface the analytic
gradient path consumes.
"""

import jax
import jax.numpy as jnp
import pytest

from fabricpc.core.activations import IdentityActivation
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.nodes import Linear, StorkeyHopfield
from fabricpc.nodes.base import NodeBase, SlotSpec
from fabricpc.nodes.identity import IdentityNode

TEMPLATE_METHODS = (
    "forward",
    "forward_with_aux",
    "forward_from_error",
    "pair_error",
    "pair_latent",
)


def _all_node_classes():
    """Recursive NodeBase subclass walk (direct __subclasses__ misses
    grandchildren like LinearExplicitGrad, MaxPool, AvgPool)."""
    # Import the library modules so every shipped subclass is registered.
    import fabricpc.nodes  # noqa: F401
    import fabricpc.nodes.linear_explicit_grad  # noqa: F401

    seen = set()
    stack = [NodeBase]
    classes = []
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub not in seen:
                seen.add(sub)
                classes.append(sub)
                stack.append(sub)
    return classes


class TestTemplatesAreNotOverridden:
    def test_all_node_classes_resolve_templates_to_base(self):
        classes = _all_node_classes()
        # The audit must cover every node class the package exports —
        # anchored to the export list, not a count, so dropping a module
        # from fabricpc.nodes.__init__ cannot silently shrink the audit.
        import fabricpc.nodes as nodes_pkg

        exported = {
            getattr(nodes_pkg, name)
            for name in nodes_pkg.__all__
            if isinstance(getattr(nodes_pkg, name), type)
            and issubclass(getattr(nodes_pkg, name), NodeBase)
            and getattr(nodes_pkg, name) is not NodeBase
        }
        missing = exported - set(classes)
        assert not missing, f"subclass walk missed exported node classes: {missing}"
        for cls in classes:
            for method in TEMPLATE_METHODS:
                # staticmethod access off the class yields the underlying
                # function; an inherited template is the identical object.
                assert getattr(cls, method) is getattr(NodeBase, method), (
                    f"{cls.__name__}.{method} overrides a base-owned template; "
                    f"the pair and assembly templates are not override points."
                )


def _chain_structure():
    """x (Linear source) -> h (Linear) -> hop (StorkeyHopfield)."""
    w_init = NormalInitializer(std=0.1)
    x = Linear(shape=(6,), name="x", weight_init=w_init)
    h = Linear(
        shape=(6,), name="h", activation=IdentityActivation(), weight_init=w_init
    )
    hop = StorkeyHopfield(shape=(6,), name="hop", hopfield_strength=1.0)
    structure = graph(
        nodes=[x, h, hop],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=hop.slot("in")),
        ],
        task_map=TaskMap(x=x, y=hop),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=1),
    )
    return structure


def _node_state(rng_key, batch_size, shape):
    from fabricpc.core.types import NodeState

    z_latent = jax.random.normal(rng_key, (batch_size, *shape))
    return NodeState(
        z_latent=z_latent,
        z_mu=jnp.zeros((batch_size, *shape)),
        error=jnp.zeros((batch_size, *shape)),
        energy=jnp.zeros((batch_size,)),
        latent_grad=jnp.zeros((batch_size, *shape)),
    )


class TestSourceGuard:
    def test_forward_on_source_mirrors_latent(self, rng_key):
        """The template forward() owns source semantics: z_mu = z_latent cast
        to z_mu's dtype, error = 0, functional energy at (z, z)."""
        structure = _chain_structure()
        params = initialize_params(structure, rng_key)
        info = structure.nodes["x"].node_info
        assert info.in_degree == 0

        state = _node_state(rng_key, 4, info.shape)
        # int-dtype latent (as an int clamp would leave it)
        int_state = state._replace(
            z_latent=jnp.arange(4 * 6, dtype=jnp.int32).reshape(4, 6)
        )

        for s in (state, int_state):
            out = Linear.forward(params.nodes["x"], {}, s, info)
            assert out.z_mu.dtype == s.z_mu.dtype
            assert jnp.array_equal(out.z_mu, s.z_latent.astype(s.z_mu.dtype))
            assert jnp.all(out.error == 0)
            energy_obj = info.energy
            expected = type(energy_obj).energy(
                s.z_latent, s.z_latent.astype(s.z_mu.dtype), energy_obj.config
            )
            assert jnp.allclose(out.energy, expected)

    def test_energy_override_tolerates_source_aux_none(self, rng_key):
        """The source branch calls energy(aux=None) — predict() never ran and
        the param initializer gives sources empty params — so an
        energy-overriding node used as a source must fall back to its base
        energy instead of unpacking a missing aux (StorkeyHopfield's memory
        matrix lives under its input edge key, which a source cannot have)."""
        w_init = NormalInitializer(std=0.1)
        hop = StorkeyHopfield(shape=(6,), name="hop", hopfield_strength=1.0)
        out = Linear(shape=(4,), name="out", weight_init=w_init)
        structure = graph(
            nodes=[hop, out],
            edges=[Edge(source=hop, target=out.slot("in"))],
            task_map=TaskMap(x=hop, y=out),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)
        info = structure.nodes["hop"].node_info
        assert info.in_degree == 0
        state = _node_state(rng_key, 3, info.shape)

        result = StorkeyHopfield.forward(params.nodes["hop"], {}, state, info)
        energy_obj = info.energy
        expected = type(energy_obj).energy(
            state.z_latent, state.z_latent, energy_obj.config
        )
        assert jnp.allclose(result.energy, expected)

    def test_source_never_calls_predict(self, rng_key):
        """IdentityNode.predict would crash on empty inputs; the template
        guard means a source IdentityNode still forwards fine."""
        w_init = NormalInitializer(std=0.1)
        src = IdentityNode(shape=(5,), name="src")
        out = Linear(shape=(3,), name="out", weight_init=w_init)
        structure = graph(
            nodes=[src, out],
            edges=[Edge(source=src, target=out.slot("in"))],
            task_map=TaskMap(x=src, y=out),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)
        info = structure.nodes["src"].node_info
        state = _node_state(rng_key, 2, info.shape)
        result = IdentityNode.forward(params.nodes["src"], {}, state, info)
        assert jnp.array_equal(result.z_mu, state.z_latent)


class TestAuxSurface:
    def test_linear_aux_is_pre_activation(self, rng_key):
        """forward_with_aux surfaces Linear's pre_activation (with identity
        activation and zero bias, aux == z_mu)."""
        structure = _chain_structure()
        params = initialize_params(structure, rng_key)
        info = structure.nodes["h"].node_info
        state = _node_state(rng_key, 4, info.shape)
        inputs = {info.in_edges[0]: jax.random.normal(jax.random.PRNGKey(7), (4, 6))}

        new_state, aux = Linear.forward_with_aux(params.nodes["h"], inputs, state, info)
        assert aux is not None
        assert jnp.allclose(aux, new_state.z_mu)
        # The pair was applied by the template.
        assert jnp.allclose(new_state.error, state.z_latent - new_state.z_mu)

    def test_storkey_hopfield_aux_is_w_and_strength(self, rng_key):
        structure = _chain_structure()
        params = initialize_params(structure, rng_key)
        info = structure.nodes["hop"].node_info
        state = _node_state(rng_key, 4, info.shape)
        inputs = {info.in_edges[0]: jax.random.normal(jax.random.PRNGKey(8), (4, 6))}

        new_state, aux = StorkeyHopfield.forward_with_aux(
            params.nodes["hop"], inputs, state, info
        )
        W, strength = aux
        assert W.shape == (6, 6)
        assert jnp.allclose(W, W.T)  # enforce_symmetry default
        assert jnp.asarray(strength).ndim == 0 or jnp.asarray(strength).size == 1
        # The attractor term entered the energy via the energy() override.
        energy_obj = info.energy
        pc_energy = type(energy_obj).energy(
            new_state.z_latent, new_state.z_mu, energy_obj.config
        )
        assert not jnp.allclose(new_state.energy, pc_energy)

    def test_source_forward_with_aux_returns_none(self, rng_key):
        structure = _chain_structure()
        params = initialize_params(structure, rng_key)
        info = structure.nodes["x"].node_info
        state = _node_state(rng_key, 4, info.shape)
        _, aux = Linear.forward_with_aux(params.nodes["x"], {}, state, info)
        assert aux is None


class TestPredictContract:
    def test_predict_ignores_z_latent_values(self, rng_key):
        """The contract forbids predict() from reading state.z_latent values
        (shape/dtype reads allowed): sPC differentiates through such a read
        while ePC evaluates z_mu at the carried latent, so a z_latent-
        dependent prediction makes the two solvers minimize different
        energies. Perturbing z_latent (same shape) must not move z_mu or aux
        for any shipped node exercised here."""
        structure = _chain_structure()
        params = initialize_params(structure, rng_key)

        for name in ("h", "hop"):
            info = structure.nodes[name].node_info
            node_class = info.node_class
            state_a = _node_state(rng_key, 4, info.shape)
            state_b = state_a._replace(z_latent=state_a.z_latent + 3.7)
            inputs = {
                info.in_edges[0]: jax.random.normal(jax.random.PRNGKey(9), (4, 6))
            }
            mu_a, aux_a = node_class.predict(params.nodes[name], inputs, state_a, info)
            mu_b, aux_b = node_class.predict(params.nodes[name], inputs, state_b, info)
            assert jnp.array_equal(mu_a, mu_b), f"{name}: z_mu depends on z_latent"
            for leaf_a, leaf_b in zip(
                jax.tree_util.tree_leaves(aux_a), jax.tree_util.tree_leaves(aux_b)
            ):
                assert jnp.array_equal(
                    leaf_a, leaf_b
                ), f"{name}: aux depends on z_latent"

    def test_bare_array_predict_fails(self, rng_key):
        """A predict that returns a bare array instead of (z_mu, aux) fails
        when the template unpacks it."""

        class BareArrayNode(NodeBase):
            @staticmethod
            def get_slots():
                return {"in": SlotSpec(name="in", is_multi_input=True)}

            @staticmethod
            def initialize_params(key, node_shape, input_shapes, weight_init, config):
                from fabricpc.core.types import NodeParams

                return NodeParams(weights={}, biases={})

            @staticmethod
            def predict(params, inputs, state, node_info):
                return sum(inputs.values())  # WRONG: must return (z_mu, aux)

        src = IdentityNode(shape=(5,), name="src")
        bad = BareArrayNode(shape=(5,), name="bad")
        structure = graph(
            nodes=[src, bad],
            edges=[Edge(source=src, target=bad.slot("in"))],
            task_map=TaskMap(x=src, y=bad),
            inference=InferenceSGD(),
        )
        params = initialize_params(structure, rng_key)
        info = structure.nodes["bad"].node_info
        # batch of 3: unpacking a (3, 5) array as (z_mu, aux) raises
        state = _node_state(rng_key, 3, info.shape)
        inputs = {info.in_edges[0]: jax.random.normal(rng_key, (3, 5))}
        with pytest.raises((ValueError, TypeError)):
            BareArrayNode.forward(params.nodes["bad"], inputs, state, info)
