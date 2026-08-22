"""Shared test fixtures and configuration for FabricPC test suite."""

import os

# Settings outside setup_jax's scope; everything it covers is set through it below.
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
os.environ.setdefault("JAX_TRACEBACK_FILTERING", "off")

from typing import Iterator

import jax
import jax.numpy as jnp
import pytest

from fabricpc import setup_jax
from fabricpc.core.activations import SigmoidActivation, SoftmaxActivation
from fabricpc.core.energy import CrossEntropyEnergy
from fabricpc.core.inference import InferenceSGD
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.nodes import Linear

# Binds at backend initialization (the first JAX computation), so it may follow
# the imports; none of them initialize the backend.
setup_jax("cpu")


@pytest.fixture
def rng_key():
    """Fixture to provide a JAX random key."""
    return jax.random.PRNGKey(42)


def with_inference(structure, inference=None, **kwargs):
    """Return structure with modified inference config for testing.

    Pass an ``InferenceBase`` object via ``inference``, or keyword arguments
    to construct an ``InferenceSGD``.
    """
    new_config = dict(structure.config)
    new_config["inference"] = (
        inference if inference is not None else InferenceSGD(**kwargs)
    )
    return structure._replace(config=new_config)


class ListLoader:
    """Deterministic loader: same batches every time it is iterated."""

    def __init__(self, batches):
        self._batches = batches

    def __len__(self) -> int:
        return len(self._batches)

    def __iter__(self) -> Iterator:
        return iter(self._batches)


def max_param_diff(a, b) -> float:
    """Max absolute elementwise difference across two GraphParams pytrees."""
    diffs = jax.tree_util.tree_map(lambda p, q: jnp.max(jnp.abs(p - q)), a, b)
    return float(jax.tree_util.tree_reduce(jnp.maximum, diffs, jnp.array(0.0)))


def make_classification_structure(
    output_energy=None, output_activation=None, state_initializer=None
):
    """3-node Linear chain x(6) -> h(8, sigmoid) -> y(3, softmax + CE).

    The default FeedforwardStateInit satisfies both algorithms and
    InferenceSGD drives PC settling; pass ``state_initializer`` (e.g.
    ``GlobalStateInit()``) for RNG-sensitive latent initialization.
    """
    x = Linear(shape=(6,), name="x")
    h = Linear(shape=(8,), activation=SigmoidActivation(), name="h")
    y = Linear(
        shape=(3,),
        activation=output_activation or SoftmaxActivation(),
        energy=output_energy or CrossEntropyEnergy(),
        name="y",
    )
    return graph(
        nodes=[x, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=InferenceSGD(eta_infer=0.05, infer_steps=10),
        graph_state_initializer=state_initializer,
    )
