# Troubleshooting

## Installation Issues

First run NVIDIA-SMI to view system driver version and cuda version.
```bash
nvidia-smi
```

**JAX/CUDA version conflict**

If you see CUDA-related errors, install the backend matching your driver and re-run with
`-U` so JAX's coupled wheels (`jax`, `jaxlib`, plugin, pjrt) upgrade together. See
[Why `-U`?](01_installation.md#why--u) for the reason.
```bash
pip install -U "jax[cuda12]"   # or "jax[cuda13]" for driver ≥580
```

**Two CUDA backends installed (`ALREADY_EXISTS: PJRT_Api already exists for device type cuda`)**

At startup JAX scans the `jax_plugins` entry-point group and calls `initialize()` on every
plugin it finds. If both the `cuda12` and `cuda13` backend sets are present, each registers
a PJRT plugin for device type `cuda`; the first succeeds, the second is rejected and JAX
aborts backend init. This is a local environment issue — it comes from installing more than
one backend extra (`.[cuda12]` and `.[cuda13]`) into the same environment, not from the
package spec.

The reliable fix is to recreate the environment with exactly one backend — the CUDA wheels
of both versions install into the *same* `site-packages/nvidia/<component>/lib/` directories
(the `-cu12`/`-cu13` suffix is on the package name, not the import path), so a plain
`pip uninstall` of one set deletes shared `.so` files the other set still lists. That leaves
the surviving backend broken (e.g. `RuntimeError: cuDNN not found`) and JAX falls back to CPU.
Keep the backend matching your driver (`cuda13` for driver ≥580):
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U -e ".[all,cuda13]"
python -c "import jax; print(jax.default_backend(), jax.devices())"   # -> gpu [CudaDevice(id=0)]
```
To repair in place without recreating the venv, uninstall the unwanted set and then
force-reinstall the wanted set's wheels so the shared files are restored:
```bash
pip uninstall -y jax-cuda12-plugin jax-cuda12-pjrt \
  $(pip list --format=freeze | grep -oE '^nvidia-[a-z0-9-]+-cu12')
pip install --force-reinstall --no-deps \
  $(pip list --format=freeze | grep -iE '^nvidia-')   # restore cu13 files clobbered by the uninstall
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

**GPU install fails on Windows / macOS**

If `pip install -U -e ".[all,cuda12]"` (or `cuda13`) fails with
`No matching distribution found for jax-cuda12-plugin`, you are on a platform without
JAX CUDA wheels — JAX publishes them for Linux x86_64/aarch64 only (see the
[Requirements](01_installation.md#requirements) summary). Install CPU-only, or use WSL2
for GPU on Windows (JAX marks WSL2 GPU support experimental).
```bash
pip install -U -e ".[all]"   # CPU-only; works on Windows and macOS
```

**Editable install fails: `missing the 'build_editable' hook`**

`pip install -e .` with the system Python fails before building:
```
Project ... has a 'pyproject.toml' and its build backend is missing the
'build_editable' hook. Since it does not have a 'setup.py' nor a 'setup.cfg', it
cannot be installed in editable mode. Consider using a build backend that supports PEP 660.
```
Editable installs with the system Python can fail with a `build_editable` error on
distros that package an old setuptools. Install into a virtual environment — its
module path omits the distro's stale `dist-packages`:
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -U -e ".[all,dev,cuda12]"
```

**Triton GEMM XLA errors**

If XLA compilation fails with Triton-related errors:
```bash
export FABRICPC_DISABLE_TRITON_GEMM=1
```

**`Unknown flag in XLA_FLAGS`**

The process dies at the first JAX computation with no Python traceback:
```
F0000 00:00:00.000000 parse_flags_from_env.cc:234] Unknown flag in XLA_FLAGS: --xla_gpu_...
```
XLA aborts rather than raising when it does not recognize a flag. `setup_jax` writes
`--xla_gpu_deterministic_ops` and `--xla_gpu_enable_triton_gemm`, which jaxlib accepts
but does not list in its stable flag set (`XLA_FLAGS=--help`), so a jaxlib release that
drops either name produces this abort. The parser rejects the flag's *name*, so
presetting your own value in `XLA_FLAGS` aborts identically. Two remedies: stop
`setup_jax` from writing any XLA flag, or pin jax to the last version that accepts the
flag:

```bash
export FABRICPC_SKIP_XLA_FLAGS=1   # setup_jax leaves XLA_FLAGS untouched
# or
pip install "jax==<last working version>"
```

**`setup_jax()` had no effect**

`setup_jax` must run before the first JAX computation or device query; afterwards the
backend exists and the call returns without changing anything. It emits a
`RuntimeWarning` in that case ("setup_jax() ran after the JAX backend was
initialized"). Move the call above the
first `jax.devices()`, `jnp` operation, or `initialize_params` in your script. Import
order does not matter — only computation order.

**Python version**

FabricPC supports Python 3.11–3.13. Only the optional Aim experiment tracker (in `[viz]`/`[all]`) is limited to Python ≤3.12; on Python 3.13 it is skipped automatically and the rest installs normally.

---

## Training Issues

**Energy not decreasing**
- Check `eta_infer` — too high causes oscillation, too low causes slow convergence. Try 0.01–0.2.
- Check `infer_steps` — too few steps prevent convergence. Try 20–50.
- Verify activation/energy pairing — use `SoftmaxActivation` + `CrossEntropyEnergy` for classification output nodes.
- For deep networks, enable muPC: `scaling=MuPCConfig()` in `graph()`.

**NaN values during training**
- Reduce the optimizer learning rate (try 1e-4 or lower).
- Check weight initialization scale — use `MuPCInitializer` with `MuPCConfig` for deep networks.
- Try `InferenceSGDNormClip` for gradient stability.
- Ensure inputs are properly normalized.

**Low accuracy despite low energy**
- Verify that `TaskMap(x=..., y=...)` correctly maps to your input and output nodes.
- Ensure the output node uses `SoftmaxActivation` + `CrossEntropyEnergy` for classification.
- Check that labels are one-hot encoded.

---

## Performance Issues

**First batch is very slow (5–10 seconds)**

This is expected — JAX JIT-compiles the training step on first invocation. Subsequent batches run at full speed.

**JAX memory flags**

Set before the first JAX computation (import order does not matter):
```python
from fabricpc import setup_jax
setup_jax()
```

Or manually:
```python
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
```

---

## Graph Construction Issues

**Self-edges are not allowed**

A node cannot connect to itself via an Edge. The graph builder will raise an error.

See the `StorkeyHopfield` node for an example of self-feedback lateral connections; implement self-connection inside a node.

**Duplicate node names**

Each node must have a unique name. Use `GraphNamespace` for scoped naming:
```python
from fabricpc.core.topology import GraphNamespace
with GraphNamespace("encoder"):
    h1 = Linear(shape=(256,), name="h1")  # becomes "encoder/h1"
```

**Single-input slot receiving multiple edges**

If a slot has `is_multi_input=False` (e.g., StorkeyHopfield's `"in"`), it accepts only one edge. Use `IdentityNode` as a merge point if you need to combine multiple sources.

---

## muPC Issues

**When to set `include_output=True` vs `False`**

- `False` (default): For classification with softmax + CrossEntropy. Output node uses standard initialization.
- `True`: For regression with identity + Gaussian energy. Output node gets muPC scaling.

**Using MuPCInitializer without MuPCConfig**

`MuPCInitializer` alone just draws weights from `N(0, 1)`. Without `MuPCConfig` on the graph, there is no forward scaling — activations will explode. Always use both together.

---

## FAQ

**Is PC equivalent to backpropagation?**

Under certain conditions (infinite inference steps, specific energy functionals), PC converges to the same gradients as backprop. In practice, PC with finite inference steps and Hebbian learning produces similar but not identical results. FabricPC provides both modes for comparison.

`EPCInference` has a second route to backprop: a single step, or a small `eta_infer × infer_steps × λ_max` (λ_max the top eigenvalue of the energy's Hessian in error coordinates), leaves the errors at the backprop activation gradient, so the weight gradients are backprop's with rescaled magnitude. `EPCInference.regime_label(lambda_max)` names the regime for a measured λ_max (`fabricpc.utils.linear_pc_oracle.top_epsilon_eigenvalue`); see the backprop-regime paragraph in the [Inference Algorithms API](12_api_inference.md).

**Why is PC slower than backprop?**

PC requires iterative inference (typically 3 to 5 steps per layer of depth per batch) before weight updates. Each inference step is a forward pass through the network. This overhead is inherent to the algorithm. Potential speedups include neuromorphic hardware that runs inference in parallel, or incremental PC methods that amortize inference across batches.

**Can I use FabricPC for non-classification tasks?**

Yes. Set the output node to `IdentityActivation()` + `GaussianEnergy()` for regression. The `TaskMap(x=..., y=...)` pattern works the same way — `y` is clamped to continuous targets during training.

**How do I save and load trained parameters?**

Use Orbax (included in dependencies):
```python
import orbax.checkpoint as ocp

# Save
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save("/path/to/checkpoint", params)

# Load
params = checkpointer.restore("/path/to/checkpoint")
```
