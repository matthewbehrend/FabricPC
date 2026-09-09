"""Core JAX predictive coding components."""

# Type definitions
from fabricpc.core.types import (
    GraphParams,
    GraphState,
    GraphStructure,
    NodeInfo,
    EdgeInfo,
    SlotInfo,
    NodeParams,
    NodeState,
)

# Activation functions
from fabricpc.core.activations import (
    ActivationBase,
    IdentityActivation,
    SigmoidActivation,
    TanhActivation,
    ReLUActivation,
    LeakyReLUActivation,
    GeluActivation,
    SoftmaxActivation,
    HardTanhActivation,
)

# Energy functions
from fabricpc.core.energy import (
    EnergyFunctional,
    GaussianEnergy,
    BernoulliEnergy,
    CrossEntropyEnergy,
    LaplacianEnergy,
    HuberEnergy,
    KLDivergenceEnergy,
    compute_energy,
    compute_energy_gradient,
    get_energy_and_gradient,
)

# Inference functions and classes
from fabricpc.core.inference import (
    InferenceBase,
    InferenceSGD,
    InferenceSGDNormClip,
    InferenceSchedule,
    gather_inputs,
    run_inference,
)
from fabricpc.core.inference_epc import EPCInference, Regime
from fabricpc.core.epsilon_spectrum import (
    EpsilonSpectrum,
    epsilon_spectrum,
    make_epsilon_spectrum,
    weighted_relaxed_fraction,
)

# Initializers
from fabricpc.core.initializers import (
    InitializerBase,
    ZerosInitializer,
    OnesInitializer,
    NormalInitializer,
    UniformInitializer,
    XavierInitializer,
    KaimingInitializer,
    MuPCInitializer,
    initialize,
)

__all__ = [
    # Types
    "GraphParams",
    "GraphState",
    "GraphStructure",
    "NodeInfo",
    "EdgeInfo",
    "SlotInfo",
    "NodeParams",
    "NodeState",
    # Activation functions
    "ActivationBase",
    "IdentityActivation",
    "SigmoidActivation",
    "TanhActivation",
    "ReLUActivation",
    "LeakyReLUActivation",
    "GeluActivation",
    "SoftmaxActivation",
    "HardTanhActivation",
    # Energy functions
    "EnergyFunctional",
    "GaussianEnergy",
    "BernoulliEnergy",
    "CrossEntropyEnergy",
    "LaplacianEnergy",
    "HuberEnergy",
    "KLDivergenceEnergy",
    "compute_energy",
    "compute_energy_gradient",
    "get_energy_and_gradient",
    # Inference
    "InferenceBase",
    "InferenceSGD",
    "InferenceSGDNormClip",
    "EPCInference",
    "Regime",
    "EpsilonSpectrum",
    "epsilon_spectrum",
    "make_epsilon_spectrum",
    "weighted_relaxed_fraction",
    "InferenceSchedule",
    "gather_inputs",
    "run_inference",
    # Initializers
    "InitializerBase",
    "ZerosInitializer",
    "OnesInitializer",
    "NormalInitializer",
    "UniformInitializer",
    "XavierInitializer",
    "KaimingInitializer",
    "MuPCInitializer",
    "initialize",
]
