# FabricPC

**State-of-the-art predictive coding, made easy.**

FabricPC is an easy-to-use, high-performance open-source Python library for building and training predictive coding networks. It is designed to get researchers from idea to running experiment as fast as possible, eliminating algorithm boilerplate. A single directed edge between nodes is all that's needed to define a connection. Local derivatives are built in, following graph topology. The framework handles inference and learning dynamics automatically for whatever you write in a node's `forward()` method.

Built on JAX for GPU and multi-GPU acceleration with local (node-level) automatic differentiation.

## What It Does

FabricPC supports arbitrary graph topologies: feedforward, recurrent, skip connections, and cyclic architectures. Heterogeneous components such as linear, convolutional, and pooling nodes, transformer blocks, and Storkey-Hopfield associative memory coexist within the same energy-minimization graph. The same graph topology can be trained by predictive coding (`train(..., algorithm="pc")`) or by backpropagation (`train(..., algorithm="backprop")`), so controlled PC-vs-backprop comparisons reuse one model definition instead of two. See `examples/PC_backprop_compare.py`.

Internally, everything is organized around three abstractions: nodes (state and computation), edges (connections between nodes), and updates (inference and learning algorithms).

## Installation

Python 3.11–3.13. Install into a virtual environment, not the system Python. Create and activate the environment, then one command installs FabricPC, its optional dependencies, and a version-matched JAX backend — pick the line for your hardware:

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U "fabricpc[all,cuda13]"   # GPU, CUDA 13 (NVIDIA driver ≥580)
pip install -U "fabricpc[all,cuda12]"   # GPU, CUDA 12
pip install -U "fabricpc[all]"          # CPU only
pip install fabricpc                    # core library only, CPU
```

`nvidia-smi` reports the CUDA version your driver supports.

**Platform:** GPU acceleration requires **Linux** (x86_64 or aarch64) — JAX publishes CUDA wheels for Linux only. On native Windows or macOS, install CPU-only; for GPU on Windows use WSL2 (JAX marks WSL2 GPU support experimental). The optional Aim experiment tracker in `[viz]`/`[all]` is Linux/macOS only and supports Python ≤3.12; on Windows or Python 3.13 it is skipped automatically and everything else installs normally.

See the [installation guide](https://github.com/trueagi-io/FabricPC/blob/main/docs/user_guides/01_installation.md) for details.

### From source (contributors)

```bash
git clone https://github.com/trueagi-io/FabricPC.git
cd FabricPC
python3 -m venv .venv && source .venv/bin/activate
pip install -U -e ".[all,dev]"    # add a backend extra for GPU: ".[all,dev,cuda12]"

# Install pre-commit hooks for code quality
pre-commit install

# Run an example
python examples/mnist_demo.py
```

## Build a Model

Define the graph. Initialize the parameters. Start experimenting.

```python
import jax
from fabricpc.nodes import Linear
from fabricpc.core.topology import Edge
from fabricpc.graph_assembly import TaskMap, graph
from fabricpc.graph_initialization import initialize_params
from fabricpc.core.inference import InferenceSGD
from fabricpc import setup_jax

setup_jax()

layer1 = Linear(shape=(784,), name="input")
layer2 = Linear(shape=(256,), name="hidden")
layer3 = Linear(shape=(10,), name="output")

structure = graph(
    nodes=[layer1, layer2, layer3],
    edges=[Edge(source=layer1, target=layer2.slot("in")),
           Edge(source=layer2, target=layer3.slot("in"))],
    task_map=TaskMap(x=layer1, y=layer3),
    inference=InferenceSGD(eta_infer=0.05, infer_steps=20),
)

rng_key = jax.random.PRNGKey(0)
params = initialize_params(structure, rng_key)
```

## Demos

The [`examples`](https://github.com/trueagi-io/FabricPC/tree/main/examples) folder includes working demonstrations across image classification, sequence modeling, depth scaling (`examples/scaling/`), associative memory, and architectural probes. Start with [`mnist_demo.py`](https://github.com/trueagi-io/FabricPC/blob/main/examples/mnist_demo.py) (over 98% accuracy on MNIST) and explore from there:

- [`mnist_conv_demo.py`](https://github.com/trueagi-io/FabricPC/blob/main/examples/mnist_conv_demo.py) — convolutional MNIST classifier with `ConvNode` and `MaxPool`
- [`resnet18_cifar10_demo.py`](https://github.com/trueagi-io/FabricPC/blob/main/examples/resnet18_cifar10_demo.py) — ResNet-18 as a PC graph, with global average pooling
- [`transformer_v2_demo.py`](https://github.com/trueagi-io/FabricPC/blob/main/examples/transformer_v2_demo.py) — character- or BPE-level language modeling with text generation
- [`transformer_tuning.py`](https://github.com/trueagi-io/FabricPC/blob/main/examples/transformer_tuning.py) — two-phase hyperparameter search minimizing validation perplexity

## Documentation

User guides, API reference, and tutorials live in [`docs/user_guides`](https://github.com/trueagi-io/FabricPC/blob/main/docs/user_guides/00_index.md). Development plans and technical design documents are in [`docs/dev_plans_archive`](https://github.com/trueagi-io/FabricPC/tree/main/docs/dev_plans_archive).

## Extending FabricPC

### Custom Nodes

Create custom node types by subclassing `NodeBase`. Implement the `get_slots()`, `initialize_params()`, and `forward()` methods. Nodes have a single output. Slots define incoming connections and are referenced in edges when building the graph.

See [`docs/user_guides/06_custom_nodes.md`](https://github.com/trueagi-io/FabricPC/blob/main/docs/user_guides/06_custom_nodes.md) for the node contract and a Conv2D teaching example (the production node is `fabricpc.nodes.ConvNode`).

## Contributing

Contributions are welcome! Please open issues or pull requests on the GitHub repository. See [CONTRIBUTING.md](https://github.com/trueagi-io/FabricPC/blob/main/CONTRIBUTING.md) for the development setup, the pull-request expectations, and the design-first workflow.

This is a research-first project.
- APIs may change frequently until the v1.0 release.
- Any breaking changes are documented in the changelog.

## Team

FabricPC is actively maintained by SingularityNET as part of the Artificial Superintelligence Alliance. Project lead: Dr. Matthew Behrend.

## License

This project is licensed under the [MIT License](https://github.com/trueagi-io/FabricPC/blob/main/LICENSE).
