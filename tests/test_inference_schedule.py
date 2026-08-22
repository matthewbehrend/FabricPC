"""
Tests for InferenceSchedule — composable per-update solver segments.

A schedule folds the state through its component solvers in order; the
handoff passes z_latent/z_mu/error exactly as the previous segment left
them (each solver applies its own begin_segment/finalize_state). segments()
flattens for per-step consumers; the per-step stubs raise.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from fabricpc.core.activations import IdentityActivation, TanhActivation
from fabricpc.core.inference import InferenceSGD, InferenceSchedule, run_inference
from fabricpc.core.inference_epc import EPCInference
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode
from fabricpc.training import train_step
from fabricpc.utils.dashboarding.inference_tracking import run_inference_with_history

W_INIT = NormalInitializer(std=0.3)


def _graph(inference):
    x = IdentityNode(shape=(5,), name="x")
    h = Linear(shape=(4,), name="h", activation=TanhActivation(), weight_init=W_INIT)
    y = Linear(
        shape=(3,), name="y", activation=IdentityActivation(), weight_init=W_INIT
    )
    return graph(
        nodes=[x, h, y],
        edges=[
            Edge(source=x, target=h.slot("in")),
            Edge(source=h, target=y.slot("in")),
        ],
        task_map=TaskMap(x=x, y=y),
        inference=inference,
    )


def _setup(inference, rng_key, batch_size=4):
    structure = _graph(inference)
    params = initialize_params(structure, rng_key)
    clamps = {
        "x": jax.random.normal(rng_key, (batch_size, 5)),
        "y": jax.random.normal(jax.random.PRNGKey(1), (batch_size, 3)),
    }
    state = initialize_graph_state(
        structure, batch_size, rng_key, clamps, params=params
    )
    return structure, params, clamps, state


def _total_energy(state, structure):
    return sum(
        jnp.sum(state.nodes[name].energy)
        for name in structure.nodes
        if structure.nodes[name].node_info.in_degree > 0
    )


class TestConstruction:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            InferenceSchedule()

    def test_non_inference_raises(self):
        with pytest.raises(TypeError, match="InferenceBase"):
            InferenceSchedule(InferenceSGD(), "not a solver")

    def test_per_step_stubs_raise(self):
        schedule = InferenceSchedule(InferenceSGD())
        with pytest.raises(NotImplementedError, match="segments"):
            schedule.inference_step(None, None, None, None, None)
        with pytest.raises(NotImplementedError, match="segments"):
            schedule.compute_new_latent(None, None, None)


class TestSegments:
    def test_segments_flatten(self):
        epc = EPCInference(infer_steps=5)
        spc = InferenceSGD(infer_steps=20)
        schedule = InferenceSchedule(epc, spc)
        assert schedule.segments() == ((epc, 5), (spc, 20))

    def test_nested_schedules_flatten(self):
        epc = EPCInference(infer_steps=5)
        spc1 = InferenceSGD(infer_steps=20)
        spc2 = InferenceSGD(infer_steps=7)
        nested = InferenceSchedule(InferenceSchedule(epc, spc1), spc2)
        assert nested.segments() == ((epc, 5), (spc1, 20), (spc2, 7))

    def test_plain_solver_is_single_segment(self):
        spc = InferenceSGD(infer_steps=13)
        assert spc.segments() == ((spc, 13),)


class TestExecution:
    def test_single_solver_schedule_equals_plain_solver(self, rng_key):
        solver = InferenceSGD(eta_infer=0.05, infer_steps=10)
        structure, params, clamps, state = _setup(InferenceSchedule(solver), rng_key)
        via_schedule = run_inference(params, state, clamps, structure)
        direct = solver.run_inference(params, state, clamps, structure)
        for name in structure.nodes:
            for field in ("z_latent", "z_mu", "error", "energy", "latent_grad"):
                assert jnp.array_equal(
                    getattr(via_schedule.nodes[name], field),
                    getattr(direct.nodes[name], field),
                ), f"{name}.{field} differs between schedule and plain solver"

    def test_epc_then_spc_equals_manual_sequential_calls(self, rng_key):
        """The schedule is the manual fold — z_latent/z_mu/error pass every
        segment boundary bit-identical (no resync)."""
        epc = EPCInference(eta_infer=0.01, infer_steps=3)
        spc = InferenceSGD(eta_infer=0.05, infer_steps=5)
        structure, params, clamps, state = _setup(InferenceSchedule(epc, spc), rng_key)

        via_schedule = run_inference(params, state, clamps, structure)
        manual = epc.run_inference(params, state, clamps, structure)
        manual = spc.run_inference(params, manual, clamps, structure)

        for name in structure.nodes:
            for field in ("z_latent", "z_mu", "error", "energy", "latent_grad"):
                assert jnp.array_equal(
                    getattr(via_schedule.nodes[name], field),
                    getattr(manual.nodes[name], field),
                ), f"{name}.{field} differs between schedule and manual fold"

    def test_handoff_energy_non_increasing(self, rng_key):
        """sPC refinement from ePC's finalized state descends the same
        energy: the round trip does not undo ePC's progress."""
        epc = EPCInference(eta_infer=0.01, infer_steps=5)
        spc = InferenceSGD(eta_infer=0.02, infer_steps=10)
        structure, params, clamps, state = _setup(InferenceSchedule(epc, spc), rng_key)

        after_epc = epc.run_inference(params, state, clamps, structure)
        after_spc = spc.run_inference(params, after_epc, clamps, structure)
        e_init = _total_energy(
            EPCInference.derive_states(params, state, clamps, structure), structure
        )
        e_epc = _total_energy(after_epc, structure)
        e_spc = _total_energy(after_spc, structure)
        assert e_epc <= e_init + 1e-6
        assert e_spc <= e_epc + 1e-6

    def test_runs_inside_jit_train_step(self, rng_key):
        schedule = InferenceSchedule(
            EPCInference(eta_infer=0.01, infer_steps=2),
            InferenceSGD(eta_infer=0.05, infer_steps=3),
        )
        structure = _graph(schedule)
        params = initialize_params(structure, rng_key)
        optimizer = optax.adamw(1e-3)
        opt_state = optimizer.init(params)
        batch = {
            "x": jax.random.normal(rng_key, (4, 5)),
            "y": jax.random.normal(jax.random.PRNGKey(1), (4, 3)),
        }

        step_fn = jax.jit(
            lambda p, o, b, k: train_step(p, o, b, structure, optimizer, k)
        )
        new_params, new_opt_state, energy, final_state = step_fn(
            params, opt_state, batch, rng_key
        )
        assert jnp.isfinite(energy)
        assert not jnp.any(jnp.isnan(final_state.nodes["h"].z_latent))


class TestTrackingParity:
    def test_history_rows_and_final_state(self, rng_key):
        epc = EPCInference(eta_infer=0.01, infer_steps=3)
        spc = InferenceSGD(eta_infer=0.05, infer_steps=5)
        structure, params, clamps, state = _setup(InferenceSchedule(epc, spc), rng_key)

        final_tracked, metrics = run_inference_with_history(
            params, state, clamps, structure
        )
        # One metric row per step across all segments.
        n_rows = metrics["h"]["energy"].shape[0]
        assert n_rows == 3 + 5

        final_plain = run_inference(params, state, clamps, structure)
        for name in structure.nodes:
            assert jnp.allclose(
                final_tracked.nodes[name].z_latent,
                final_plain.nodes[name].z_latent,
                atol=1e-6,
            ), f"{name}: tracked final state differs from run_inference"

    def test_single_solver_tracking_unchanged(self, rng_key):
        solver = InferenceSGD(eta_infer=0.05, infer_steps=4)
        structure, params, clamps, state = _setup(solver, rng_key)
        final_tracked, metrics = run_inference_with_history(
            params, state, clamps, structure
        )
        assert metrics["h"]["energy"].shape[0] == 4
        final_plain = run_inference(params, state, clamps, structure)
        assert jnp.array_equal(
            final_tracked.nodes["h"].z_latent, final_plain.nodes["h"].z_latent
        )
