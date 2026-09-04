"""
Error-parameterized predictive coding (ePC) inference.

State-based PC (sPC — ``InferenceSGD`` and variants) relaxes the latents
z_latent by local gradient descent, so the output-loss signal attenuates by
the inference rate per layer per step and deep graphs need many steps. ePC
(Goemaere et al., arXiv 2505.20137) reparameterizes the same energy over the
prediction errors: the error ε is the first-class relaxed variable and each
z_latent is derived by a forward pass in schedule order,
``z_latent := z_mu + ε``. Because every node's z_mu depends on all upstream
latents, one ``jax.value_and_grad`` over the ε pytree through the whole
derived forward delivers the loss signal to every layer unattenuated. The
ε ↔ z_latent map is a bijection with unit-determinant triangular Jacobian
(paper Appendix C): identical energies, identical equilibria, and the final
derived state feeds the existing local weight-gradient path unchanged.

muPC note: ``scale_inputs`` applies inside the differentiated forward
exactly where the sPC loop applies it, and the global reverse pass supplies
the chain-rule factors automatically. The per-hop gradient preconditioners
(the ``jacobian_gain`` factor inside ``topdown_grad_scale``, and
``self_grad_scale``) condition sPC's one-hop updates and are not replicated
here — a global backward pass has no per-hop damping to compensate.
``scale_weight_grads`` at learning time is untouched.

Memory: each step's single ``value_and_grad`` stores activations for the
whole derived forward — full depth times the unroll degree — reverse-mode
memory at backprop scale, versus sPC's per-node closures.
"""

from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

from fabricpc.core.inference import InferenceBase, gather_inputs
from fabricpc.core.scaling import scale_inputs
from fabricpc.core.state_ops import update_node_in_state
from fabricpc.core.types import (
    GraphParams,
    GraphState,
    GraphStructure,
    NodeState,
)


class EPCInference(InferenceBase):
    """
    ePC inference: relax the prediction errors, derive the latents.

    Args:
        eta_infer: Inference rate on ε (default: 1e-3). The ε gradient is
            taken through the full network's transfer function — a change in
            one node's ε moves every downstream derived latent — so tune it
            like a weight learning rate, not like sPC's local per-node rate.
            Gradient descent on the error-coordinate energy is stable only
            for eta_infer < 2/λ_max(H_ε), the top eigenvalue of that energy's
            Hessian; ``fabricpc.utils.linear_pc_oracle.top_epsilon_eigenvalue``
            measures it on any graph.
        infer_steps: Number of inference iterations (default: 5). One
            reverse pass per step reaches every layer, so a few steps replace
            sPC's hundreds on deep DAGs.
        latent_decay: Decay factor on ε in the update (default: 0.0).

    Backprop regime. One step from ε = 0 leaves ε_t = −eta_infer·∂L/∂z_t
    exactly, the backprop activation gradient at the feedforward point. The
    local weight gradients are then taken at the re-derived latents
    (``finalize_state``), so they match backprop's to first order in
    eta_infer·λ_max(H_ε): hidden layers scaled by eta_infer, the output
    layer unscaled (Goemaere et al., Theorem C.9, Case 1). The remainder
    comes from a node's input latent being re-derived at the perturbed
    upstream state, so a layer fed only by clamped nodes matches exactly.
    After T steps
    each excited error mode with Hessian eigenvalue λ has relaxed toward
    equilibrium by 1 − (1 − eta_infer·λ)^T, so the regime parameter is
    eta_infer·T·λ_max: ≪ 1 is backprop-like (Case 2), ≳ 3/λ_min,excited
    reaches the PC equilibrium, and eta_infer·λ_max < 2 is required for
    stability at every T, T = 1 included, since one step lands each mode at
    eta_infer·λ times its equilibrium value. ``regime_label`` names the
    regime for a measured λ_max. Under Adam the eta_infer scaling of the
    hidden-layer gradients is normalized away, so 1-step ePC with Adam
    trains as backprop with Adam.

    Measured on the muPC resnet18 demo (``examples/resnet18_cifar10_demo.py``):
    the 2-epoch sweep implies an effective excited eigenvalue of order
    10¹–10² at init, and the defaults (eta_infer·T = 0.005) train as backprop
    for tens of epochs but collapsed at epoch 20 of a 100-epoch run while
    infer_steps ∈ {1, 2} survived, consistent with λ_max growing past
    2/eta_infer as the weights grow. The defaults are kept pending a
    stability-aware rate; ``scripts/epc_analysis.py`` measures λ_max and
    tracks it during training.
    """

    def __init__(self, eta_infer=1e-3, infer_steps=5, latent_decay=0.0):
        super().__init__(
            eta_infer=eta_infer, infer_steps=infer_steps, latent_decay=latent_decay
        )

    def regime_label(self, lambda_max: float) -> str:
        """Name the regime of this solver's (eta_infer, infer_steps) on a
        graph whose error-coordinate Hessian has top eigenvalue
        ``lambda_max`` (``top_epsilon_eigenvalue`` or the linear oracle).

        The fastest excited mode has relaxed by f_max = 1 − |1 − η·λ_max|^T
        and the slowest possible mode, at the unit-precision floor λ = 1, by
        f_min = 1 − (1 − η)^T. Bands on f_max: below 0.1 "backprop-like",
        0.1 to 0.9 "partially relaxed", above 0.9 "near PC equilibrium".
        η·λ_max > 2 is "unstable" (the mode diverges, T = 1 included) and
        1 < η·λ_max ≤ 2 overshoots. λ_max grows with the weights during
        training, so a label computed at init describes init.
        """
        eta = float(self.config["eta_infer"])
        steps = int(self.config["infer_steps"])
        x = eta * float(lambda_max)
        if x > 2.0:
            return f"unstable: eta*lambda_max = {x:.3g} > 2"
        f_max = 1.0 - abs(1.0 - x) ** steps
        f_min = 1.0 - abs(1.0 - eta) ** steps
        if f_max < 0.1:
            band = "backprop-like"
        elif f_max <= 0.9:
            band = "partially relaxed"
        else:
            band = "near PC equilibrium"
        if x > 1.0:
            band += ", overshooting"
        return (
            f"eta*T*lambda_max = {x * steps:.3g} (fastest error mode relaxed "
            f"{100 * f_max:.0f}%, slowest {100 * f_min:.2g}%): {band}"
        )

    @staticmethod
    def _relaxed_errors(
        structure: GraphStructure, clamps: Dict[str, jnp.ndarray]
    ) -> Tuple[str, ...]:
        """The relaxed node names: every unclamped node, whatever its degree.

        Trace-time Python over static structure. Clamped nodes are never
        relaxed — their z_latent stays the clamp and their error is derived —
        which also keeps int-dtype token sources out of the AD pytree (the
        error field itself is always float).
        """
        return tuple(name for name in structure.nodes if name not in clamps)

    @classmethod
    def derive_states(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Forward-from-errors pass: iterate ``structure.schedule`` and derive
        each node's state from the carried ε via ``forward_from_error``
        (z_mu from predict at the latest source latents; z_latent = z_mu + ε
        for unclamped nodes; clamped nodes keep the clamp and derive ε).

        Warm-start and truncation semantics: each inference step starts from
        the carried ``GraphState``, so a cycle member's first visit reads the
        previous step's last-visit latent — effective traversal depth grows
        as steps x unroll across a segment, and the total energy is not a
        pure function of ε alone. The gradient treats the carried latents as
        constants (truncation at the step boundary).

        On a repeated visit (cyclic schedule) the same ε is re-injected and
        the node's single ``NodeState`` is overwritten, so each node's energy
        term enters the total once, evaluated at its final visit — the output
        of the computational graph threaded through every traversal.
        """
        for node_name in structure.schedule:
            node_info = structure.nodes[node_name].node_info
            in_edges_data = gather_inputs(node_info, structure, state)
            scaled_inputs = scale_inputs(in_edges_data, node_info.scaling_config)
            new_node_state = node_info.node_class.forward_from_error(
                params.nodes[node_name],
                scaled_inputs,
                state.nodes[node_name],
                node_info,
                is_clamped=(node_name in clamps),
            )
            state = state._replace(nodes={**state.nodes, node_name: new_node_state})
        return state

    @classmethod
    def error_energy(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ):
        """
        The total energy as a function of the relaxed errors.

        Returns ``(energy_of, errors)``: ``errors`` is the relaxed pytree
        {node name: ε} read from ``state``, and ``energy_of(errors)`` writes
        those ε into the state, derives all latents along the schedule, and
        returns ``(total, derived_state)`` with ``total`` the sum of the
        per-sample energies of every ``in_degree > 0`` node (the same set
        the training loop sums, so equilibria match sPC — a source's ε
        gradient arrives purely through downstream z_mu). One owner of the
        ε-energy for the solver's gradient, Hessian-vector products
        (``jax.jvp(jax.grad(...))``), and power iteration.
        """
        relaxed = cls._relaxed_errors(structure, clamps)

        def energy_of(errors):
            inner = state
            for name in relaxed:
                inner = update_node_in_state(inner, name, error=errors[name])
            inner = cls.derive_states(params, inner, clamps, structure)
            total = jnp.asarray(0.0)
            for name in structure.nodes:
                if structure.nodes[name].node_info.in_degree > 0:
                    total = total + jnp.sum(inner.nodes[name].energy)
            return total, inner

        errors = {name: state.nodes[name].error for name in relaxed}
        return energy_of, errors

    @classmethod
    def forward_value_and_grad(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        One global energy gradient with respect to the relaxed errors.

        Differentiates ``error_energy``'s scalar with respect to the relaxed
        pytree; gradients accumulate into ``latent_grad`` (never replace it).
        """
        relaxed = cls._relaxed_errors(structure, clamps)
        energy_of, errors = cls.error_energy(params, state, clamps, structure)
        (_, new_state), grads = jax.value_and_grad(energy_of, has_aux=True)(errors)

        for name in relaxed:
            latent_grad = new_state.nodes[name].latent_grad + grads[name]
            new_state = update_node_in_state(new_state, name, latent_grad=latent_grad)
        return new_state

    @classmethod
    def update_latents(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
        config: Dict[str, Any],
    ) -> GraphState:
        """Step every relaxed node's ε down the accumulated gradient."""
        for node_name in cls._relaxed_errors(structure, clamps):
            node_state = state.nodes[node_name]
            new_error = cls.compute_new_error(node_name, node_state, config)
            state = update_node_in_state(state, node_name, error=new_error)
        return state

    @staticmethod
    def compute_new_error(
        node_name: str,
        node_state: NodeState,
        config: Dict[str, Any],
    ) -> jnp.ndarray:
        """ε update: ε * (1 - eta * decay) - eta * latent_grad."""
        eta_infer = config["eta_infer"]
        latent_decay = config["latent_decay"]
        return (
            node_state.error * (1.0 - eta_infer * latent_decay)
            - eta_infer * node_state.latent_grad
        )

    @staticmethod
    def compute_new_latent(node_name, node_state, config):
        raise NotImplementedError(
            "EPCInference relaxes errors, not latents; the per-step update is "
            "compute_new_error()."
        )

    @classmethod
    def begin_segment(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        Resync ε to the incoming latents before the first ε update.

        One sPC-direction forward pass at the carried z_latents: each node's
        z_mu is recomputed from its sources' carried latents via the template
        ``forward`` and ε := z_latent - z_mu. The first ``derive_states``
        then reproduces the incoming z_latent exactly on DAGs — in schedule
        order, z_mu is recomputed at the already-preserved upstream latents,
        so z_mu + ε = z_latent node by node. A distribution initializer's
        random internal latents and a preceding sPC segment's final latent
        update both survive the handoff instead of being overwritten by a
        derive from stale ε. On cyclic graphs, repeated visits re-inject the
        same ε at updated latents, so cycle members are preserved at their
        first visit only (the unrolled parameterization has no exact inverse
        there).

        z_latent never changes during this sweep, so one visit per node
        suffices whatever the schedule's unroll degree.
        """
        for node_name in structure.node_order:
            node_info = structure.nodes[node_name].node_info
            in_edges_data = gather_inputs(node_info, structure, state)
            scaled_inputs = scale_inputs(in_edges_data, node_info.scaling_config)
            new_node_state = node_info.node_class.forward(
                params.nodes[node_name],
                scaled_inputs,
                state.nodes[node_name],
                node_info,
            )
            state = state._replace(nodes={**state.nodes, node_name: new_node_state})
        return state

    @classmethod
    def finalize_state(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        One detached ``derive_states`` rebuild after the last ε update, so
        the returned state satisfies z_latent = z_mu + ε with energies at the
        final point — the paper's weight rule. The local weight-gradient
        path, the train-loop energy, eval readouts of z_mu, and dashboard
        readers of ``error`` then work unchanged.
        """
        return cls.derive_states(params, state, clamps, structure)
