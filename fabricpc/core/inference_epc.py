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
            like a weight learning rate, starting from the weight optimizer's
            (the demos' adamw uses 1e-3), not like sPC's local rate: sPC's
            typical 0.05-0.1 conditions a per-node step against that node's
            own energy and overshoots the minimum along the global gradient.
        infer_steps: Number of inference iterations (default: 5). One
            reverse pass per step reaches every layer, so a few steps replace
            sPC's hundreds on deep DAGs.
        latent_decay: Decay factor on ε in the update (default: 0.0).
    """

    def __init__(self, eta_infer=1e-3, infer_steps=5, latent_decay=0.0):
        super().__init__(
            eta_infer=eta_infer, infer_steps=infer_steps, latent_decay=latent_decay
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
    def forward_value_and_grad(
        cls,
        params: GraphParams,
        state: GraphState,
        clamps: Dict[str, jnp.ndarray],
        structure: GraphStructure,
    ) -> GraphState:
        """
        One global energy gradient with respect to the relaxed errors.

        Builds the relaxed pytree {node name: ε}, derives all states from it
        along the schedule, sums the per-sample energies of every
        ``in_degree > 0`` node (the same set the training loop sums, so
        equilibria match sPC — a source's ε gradient arrives purely through
        downstream z_mu), and differentiates that scalar. Gradients
        accumulate into ``latent_grad`` (never replace it).
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
