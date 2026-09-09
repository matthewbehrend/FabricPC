"""
Tests for InferenceSchedule — composable per-update solver segments.

A schedule folds the state through its component solvers in order; each
segment receives the previous segment's latents, and each solver's
begin_segment adapts the derived fields to its own parameterization without
moving them (ePC resyncs ε := z_latent - z_mu at the carried latents).
segments() flattens for per-step consumers; the per-step stubs raise.
"""

import jax
import jax.numpy as jnp
import optax
import pytest

from conftest import total_energy
from fabricpc.core import (
    EPCInference,
    InferenceSGD,
    InferenceSchedule,
    run_inference,
)
from fabricpc.core.activations import IdentityActivation, TanhActivation
from fabricpc.core.initializers import NormalInitializer
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.graph_initialization.state_initializer import initialize_graph_state
from fabricpc.nodes import Linear
from fabricpc.nodes.identity import IdentityNode
from fabricpc.training import make_train_step
from fabricpc.utils.dashboarding.inference_tracking import (
    make_inference_history,
    run_inference_with_history,
)

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


class TestConstruction:
    def test_empty_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            InferenceSchedule()

    def test_non_inference_raises(self):
        with pytest.raises(TypeError, match="InferenceBase"):
            InferenceSchedule(InferenceSGD(), "not a solver")

    def test_class_instead_of_instance_raises(self):
        with pytest.raises(TypeError, match="InferenceBase"):
            InferenceSchedule(InferenceSGD)

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
        """The schedule is the manual fold — every field passes each segment
        boundary bit-identical (each solver's own begin_segment/finalize run
        inside its run_inference either way)."""
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

    def test_schedule_descends_energy(self, rng_key):
        """The composed schedule strictly descends the energy: ePC makes real
        progress from init, and the sPC segment strictly improves on ePC's
        handoff (both clamps are active, so neither segment starts at the
        minimum)."""
        epc = EPCInference(eta_infer=0.01, infer_steps=5)
        spc = InferenceSGD(eta_infer=0.02, infer_steps=10)
        schedule = InferenceSchedule(epc, spc)
        structure, params, clamps, state = _setup(schedule, rng_key)

        after_epc = epc.run_inference(params, state, clamps, structure)
        final = run_inference(params, state, clamps, structure)  # the schedule

        e_init = float(total_energy(state, structure))
        e_epc = float(total_energy(after_epc, structure))
        e_final = float(total_energy(final, structure))
        assert e_epc < e_init, "ePC segment made no progress from init"
        assert e_final < e_epc, "sPC segment made no progress from ePC's handoff"

    def test_nested_schedule_executes_like_flat(self, rng_key):
        """Nesting is execution-transparent: a schedule containing a schedule
        folds the state through the same segment sequence bit-identically."""

        def solvers():
            return (
                EPCInference(eta_infer=0.01, infer_steps=3),
                InferenceSGD(eta_infer=0.05, infer_steps=4),
                InferenceSGD(eta_infer=0.02, infer_steps=2),
            )

        epc, spc1, spc2 = solvers()
        nested = InferenceSchedule(InferenceSchedule(epc, spc1), spc2)
        flat = InferenceSchedule(*solvers())
        structure, params, clamps, state = _setup(nested, rng_key)

        via_nested = nested.run_inference(params, state, clamps, structure)
        via_flat = flat.run_inference(params, state, clamps, structure)
        for name in structure.nodes:
            for field in ("z_latent", "z_mu", "error", "energy", "latent_grad"):
                assert jnp.array_equal(
                    getattr(via_nested.nodes[name], field),
                    getattr(via_flat.nodes[name], field),
                ), f"{name}.{field} differs between nested and flat schedules"

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

        step_fn = make_train_step(structure, optimizer)
        new_params, new_opt_state, metrics, final_state = step_fn(
            params, opt_state, batch, rng_key
        )
        assert jnp.isfinite(metrics["energy"])
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

    @staticmethod
    def _at(states, i):
        return jax.tree_util.tree_map(lambda a: a[i], states)

    @staticmethod
    def _assert_states_close(a, b, what):
        for name in a.nodes:
            for field in ("z_latent", "z_mu", "error", "energy"):
                assert jnp.allclose(
                    getattr(a.nodes[name], field),
                    getattr(b.nodes[name], field),
                    atol=1e-6,
                ), f"{name}.{field}: {what}"

    def test_sampled_history_schedule_rows_and_final(self, rng_key):
        """``make_inference_history`` iterates the segments inside one jitted
        program: with ``every=1`` a 3-step ePC then 5-step sPC schedule
        yields 3 + 5 + 1 sampled states, index 0 the initialization, index 3
        the ePC segment's finalized state (what ``run_inference`` with the
        ePC solver alone returns), and the last the full settle."""
        epc = EPCInference(eta_infer=0.01, infer_steps=3)
        spc = InferenceSGD(eta_infer=0.05, infer_steps=5)
        structure, params, clamps, state = _setup(InferenceSchedule(epc, spc), rng_key)

        final_tracked, states = make_inference_history(structure, every=1)(
            params, state, clamps
        )
        assert jax.tree_util.tree_leaves(states)[0].shape[0] == 3 + 5 + 1
        final_plain = run_inference(params, state, clamps, structure)
        self._assert_states_close(final_tracked, final_plain, "final vs run_inference")
        self._assert_states_close(self._at(states, 8), final_plain, "last sample")
        self._assert_states_close(self._at(states, 0), state, "index 0")
        epc_only = structure._replace(config={**structure.config, "inference": epc})
        self._assert_states_close(
            self._at(states, 3),
            run_inference(params, state, clamps, epc_only),
            "sample at the segment boundary",
        )

    def test_sampling_interval_crosses_segment_boundaries(self, rng_key):
        """``every=2`` on the same schedule samples after steps 0, 2, 4, 6, 8:
        the sample after step 4 (one sPC step past the ePC segment) equals
        the ``every=1`` sample at index 4."""
        epc = EPCInference(eta_infer=0.01, infer_steps=3)
        spc = InferenceSGD(eta_infer=0.05, infer_steps=5)
        structure, params, clamps, state = _setup(InferenceSchedule(epc, spc), rng_key)
        _, dense = make_inference_history(structure, every=1)(params, state, clamps)
        final, sparse = make_inference_history(structure, every=2)(
            params, state, clamps
        )
        assert jax.tree_util.tree_leaves(sparse)[0].shape[0] == 5
        for i in range(5):
            self._assert_states_close(
                self._at(sparse, i), self._at(dense, 2 * i), f"every=2 sample {i}"
            )
        self._assert_states_close(self._at(sparse, 4), final, "last sample is final")

    def test_epc_samples_are_finalized_states(self, rng_key):
        """Under ePC a sample after i steps is the derived state
        ``run_inference`` returns at ``infer_steps=i``, not the raw state one
        ε update behind."""
        structure, params, clamps, state = _setup(
            EPCInference(eta_infer=0.01, infer_steps=3), rng_key
        )
        final, states = make_inference_history(structure, every=1)(
            params, state, clamps
        )
        for i in (1, 2, 3):
            truncated = structure._replace(
                config={
                    **structure.config,
                    "inference": EPCInference(eta_infer=0.01, infer_steps=i),
                }
            )
            self._assert_states_close(
                self._at(states, i),
                run_inference(params, state, clamps, truncated),
                f"ePC sample after {i} steps",
            )
        self._assert_states_close(self._at(states, 3), final, "last sample")

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
