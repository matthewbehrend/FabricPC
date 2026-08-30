# Data Loaders API

All data loaders are in `fabricpc.utils.data`.

## Summary

| Loader | Dataset | Image Shape | Classes | Install Group |
|--------|---------|:-----------:|:-------:|:-------------:|
| `MnistLoader` | MNIST | 28x28x1 | 10 | `tfds` |
| `FashionMnistLoader` | Fashion-MNIST | 28x28x1 | 10 | `tfds` |
| `Cifar10Loader` | CIFAR-10 | 32x32x3 | 10 | `tfds` |
| `Cifar100Loader` | CIFAR-100 | 32x32x3 | 100 | `tfds` |
| `CharDataLoader` | TinyShakespeare | — | vocab_size | `tfds` |
| `BpeDataLoader` | TinyShakespeare (BPE) | — | 11711 (default) | `tfds` (includes `tokenizers`) |
| `FewShotLoader` | Any TFDS dataset | varies | varies | `tfds` |
| `NoisyTestLoader` | Wraps any loader | same as base | same | — |

## MnistLoader

```python
from fabricpc.utils.data import MnistLoader

loader = MnistLoader(
    split="train",          # "train" or "test"
    batch_size=200,
    tensor_format="flat",   # "flat" (784,) or "NHWC" (28,28,1)
    shuffle=True,
    seed=42,
    normalize_mean=0.1307,  # MNIST mean
    normalize_std=0.3081,   # MNIST std
)

for images, labels in loader:
    # images: ndarray[batch, 784] (flat) or [batch, 28, 28, 1] (NHWC)
    # labels: ndarray[batch, 10] (one-hot)
    pass
```

## FashionMnistLoader

Same interface as `MnistLoader`. Default normalization: `mean=0.2860`, `std=0.3530`.

```python
from fabricpc.utils.data import FashionMnistLoader
loader = FashionMnistLoader("train", batch_size=200, tensor_format="flat", shuffle=True, seed=42)
```

## Cifar10Loader

```python
from fabricpc.utils.data import Cifar10Loader

loader = Cifar10Loader(
    split="train",
    batch_size=128,
    tensor_format="NHWC",   # (32,32,3) or "flat" (3072,)
    shuffle=True,
    seed=42,
    normalize_mean=(0.4914, 0.4822, 0.4465),  # per-channel
    normalize_std=(0.2470, 0.2435, 0.2616),
)
```

## Cifar100Loader

Same interface as `Cifar10Loader` with 100 classes. Default normalization: `mean=(0.5071, 0.4867, 0.4408)`, `std=(0.2675, 0.2565, 0.2761)`.

## CharDataLoader

Character-level text loader for language modeling.

```python
from fabricpc.utils.data import CharDataLoader

loader = CharDataLoader(
    split="train",       # "train", "validation", or "test"
    seq_len=128,         # characters per input sequence
    batch_size=64,
    shuffle=True,
    seed=42,
    max_samples=None,    # cap sequences for fast tuning
)

for x_indices, y_indices in loader:
    # x_indices: ndarray[batch, seq_len] int32
    # y_indices: ndarray[batch, seq_len] int32 — x shifted one position left
    pass

# Decode indices to text
text = loader.decode(x_indices[0])
```

Both `x` and `y` are integer token ids; one-hot encoding of the target happens inside the training step (see [Training and Evaluation](08_training_and_evaluation.md)).

## BpeDataLoader

Subword text loader for language modeling: byte-pair encoding on TinyShakespeare. One token covers a word piece, so the same text becomes a shorter sequence over a larger vocabulary than character-level. Yields the same int32 `(x, y)` id pairs as `CharDataLoader`.

```python
from fabricpc.utils.data import BpeDataLoader

loader = BpeDataLoader(
    split="train",       # "train", "validation", or "test"
    seq_len=128,         # tokens per input sequence
    batch_size=64,
)

for x_indices, y_indices in loader:
    # x_indices, y_indices: ndarray[batch, seq_len] int32
    pass

text = loader.decode(x_indices[0])
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split` | `str` | required | `"train"`, `"validation"`, or `"test"` |
| `seq_len` | `int` | required | Tokens per input sequence |
| `batch_size` | `int` | required | Sequences per batch |
| `shuffle` | `bool` | `True` | Shuffle sequence order each epoch |
| `seed` | `int` | `None` | Random seed for shuffling |
| `max_samples` | `int` | `None` | Cap sequences for fast tuning |
| `bpe_data_dir` | `str` | `"data/bpe_tokenized"` | Cache directory for tokenizer and token arrays |
| `vocab_size` | `int` | `11711` | BPE vocabulary size |
| `verbose` | `bool` | `True` | Print tokenization progress |

On first use the loader trains a HuggingFace `tokenizers` BPE tokenizer on TinyShakespeare and caches `tokenizer.json` plus one `{split}.npy` token array per split under `bpe_data_dir`; later runs load the cache. Requires the optional `tokenizers>=0.15.0` dependency (part of the `tfds` install group) and raises `ImportError` with an install hint when it is missing.

## FewShotLoader

Class-balanced K-shot loader. Subsamples exactly K examples per class.

```python
from fabricpc.utils.data import FewShotLoader

loader = FewShotLoader(
    dataset_name="fashion_mnist",
    split="train",
    k_per_class=50,       # examples per class
    batch_size=64,
    num_classes=10,
    shuffle=True,
    seed=42,
    tensor_format="flat",
    normalize_mean=0.2860,
    normalize_std=0.3530,
)
```

## NoisyTestLoader

Wraps any loader and adds Gaussian noise to images.

```python
from fabricpc.utils.data import NoisyTestLoader, FashionMnistLoader

base = FashionMnistLoader("test", batch_size=64, tensor_format="flat", shuffle=False)
noisy = NoisyTestLoader(base_loader=base, noise_std=2.0, seed=42)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_loader` | iterable | required | Base loader yielding (images, labels) |
| `noise_std` | `float` | `0.0` | Std of Gaussian noise (in normalized units) |
| `seed` | `int` | `None` | Random seed for reproducible noise |

## Using Custom Data

`train()` accepts any iterable yielding batches in one of two formats. Language-model training uses the same loader contract with integer-id targets: both `x` and `y` are `(batch, seq_len)` int32 token ids.

**Tuple format** (recommended):
```python
def my_loader():
    for i in range(num_batches):
        x = np.random.randn(batch_size, 784).astype(np.float32)
        y = np.eye(10)[np.random.randint(0, 10, batch_size)].astype(np.float32)
        yield x, y
```

**Dict format**:
```python
yield {"x": x_array, "y": y_array}
```

The loader must support `len()` (return number of batches) and `__iter__()`.
