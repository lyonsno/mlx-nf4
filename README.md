# mlx-nf4 — native NF4 for MLX

`mlx-nf4` brings NormalFloat4 weight quantization and a native Metal
quantized-matrix-multiply path to stock
[MLX](https://github.com/ml-explore/mlx). It stores eight 4-bit weights in each
`uint32`, keeps those weights compressed through the fast path, and exposes the
kernel through both a functional API and an `NF4Linear` layer.

The implementation began as an MLX core patch and was extracted into a
standalone custom extension. The extracted kernel retained essentially the
performance of the core implementation in the original M4 Max assay while
avoiding a permanent MLX fork:

| Path | Median latency |
| --- | ---: |
| standalone `mlx-nf4` Metal kernel | 0.245 ms |
| original MLX core patch | 0.228 ms |
| explicit dequantize then matmul | 0.514 ms |

That assay used `x=(128, 2048)`, packed weights `(2048, 256)`, group size 64,
and 20 timed iterations. It shows that the extension extraction preserved the
interesting mechanism: direct multiplication against packed NF4 weights, about
2.1× the speed of materializing the weight and then calling matmul in that
workload. It is not a universal performance promise; shapes, dtype, MLX
version, hardware, and warm-up still matter.

## What it provides

- NF4 quantization and dequantization with the standard 16-value codebook
- native Metal quantized matmul over packed NF4 weights
- `NF4Linear` construction from an MLX dense layer or externally packed weights
- group sizes 32, 64, and 128
- float32, float16, and bfloat16 activations
- explicit low-first and bitsandbytes-compatible high-first nibble intake
- reconstruction of plain and nested/double-quantized bitsandbytes absmax scales
- a separately named dequantize-then-matmul reference for correctness checks

The package ships and loads its own C++ binding, dynamic library, and compiled
Metal library. It uses MLX as a dependency; no patched MLX checkout is required.

## Install from GitHub

`mlx-nf4` currently builds from source on Apple silicon. It requires Python
3.10 or newer, MLX 0.32.2 or newer, and Xcode command-line tools:

```sh
python -m pip install \
  "git+https://github.com/lyonsno/mlx-nf4.git"
```

For a reproducible install, pin a tag or commit:

```sh
python -m pip install \
  "git+https://github.com/lyonsno/mlx-nf4.git@<tag-or-commit>"
```

The build installs `_ext`, `libmlx_nf4_native.dylib`, and
`mlx_nf4.metallib` inside the Python package. See
[`docs/compatibility.md`](docs/compatibility.md) for verified systems, the
native support boundary, and the clean-build procedure.

## Quantize and multiply

```python
import mlx.core as mx
import mlx_nf4 as nf4

weight = mx.random.normal((32, 64))
x = mx.random.normal((3, 64))

packed_weight, scales = nf4.quantize(weight, group_size=64)
y = nf4.quantized_matmul(
    x,
    packed_weight,
    scales,
    transpose=True,
    group_size=64,
)
mx.eval(y)
```

`quantized_matmul` is always the native extension path. The correctness oracle
is explicit rather than a silent fallback:

```python
y_reference = nf4.reference_quantized_matmul(
    x,
    packed_weight,
    scales,
    transpose=True,
    group_size=64,
)
```

## Use `NF4Linear`

```python
import mlx.core as mx
import mlx.nn as nn
import mlx_nf4 as nf4

dense = nn.Linear(64, 32)
linear = nf4.NF4Linear.from_linear(dense, group_size=64)

x = mx.random.normal((3, 64))
y = linear(x)
mx.eval(y)
```

`NF4Linear` can also consume package-native `uint32` weights or byte-packed
`uint8` weights:

```python
linear = nf4.NF4Linear.from_packed(
    packed_weight,
    scales,
    bias=bias,
    group_size=64,
)
```

For high-nibble-first bitsandbytes-style bytes:

```python
linear = nf4.NF4Linear.from_bitsandbytes(
    byte_weight,
    scales,
    bias=bias,
    group_size=64,
)
```

`NF4Linear.from_bitsandbytes` accepts reconstructed float32 absolute-maximum
scales. Use `reconstruct_bitsandbytes_scales` to obtain them from either plain
absmax tensors or nested/double-quantized bitsandbytes quant state before
constructing the layer. Model-file parsing remains the loader's job. The
nibble-order contract is checked against an observed Ideogram4 NF4 tensor prefix in
[`tests/fixtures/ideogram4_input_proj_bitsandbytes_nf4.json`](tests/fixtures/ideogram4_input_proj_bitsandbytes_nf4.json).

## GPT-2 bitsandbytes consumer

The repository includes a complete consumer for the public, double-quantized
`manu02/gpt2-bnb-4bit-nf4-dq` checkpoint. It verifies the checkpoint revision,
reconstructs nested scales, checks the stored NF4 codebook, replaces all 48
GPT-2 projections with `NF4Linear`, compares one native projection against the
explicit reference path, and writes a generation receipt.

From a repository checkout:

```sh
python -m pip install -e ".[gpt2]"
hf download manu02/gpt2-bnb-4bit-nf4-dq \
  --revision 7744ff22be99f562bdaa444612a35a20bf995999 \
  --local-dir ./gpt2-bnb-4bit-nf4-dq
hf cache verify manu02/gpt2-bnb-4bit-nf4-dq \
  --revision 7744ff22be99f562bdaa444612a35a20bf995999 \
  --local-dir ./gpt2-bnb-4bit-nf4-dq \
  --fail-on-missing-files
python examples/gpt2_bitsandbytes.py \
  --model-dir ./gpt2-bnb-4bit-nf4-dq \
  --receipt ./gpt2-nf4-receipt.json
```

## API

| API | Purpose |
| --- | --- |
| `quantize` | Convert dense weights to packed NF4 plus float32 group scales |
| `dequantize` | Reconstruct dense values from packed NF4 weights |
| `quantized_matmul` | Run the native Metal fast path |
| `reference_quantized_matmul` | Dequantize explicitly and multiply for comparison |
| `pack_uint8_to_uint32` | Convert byte-packed NF4 into the native word layout |
| `reconstruct_bitsandbytes_scales` | Reconstruct float32 scales from plain or nested bitsandbytes absmax state |
| `NF4Linear` | Frozen MLX layer backed by the native NF4 kernel |

## Native scope in 0.1

The native kernel currently supports:

- activations with any number of leading dimensions; the package flattens
  those dimensions for the native kernel and restores them on output
- two-dimensional packed weights and scale tensors
- transposed weights (`transpose=True`)
- group sizes 32, 64, and 128
- float32, float16, and bfloat16 activations
- packed `uint32` weights and float32 absolute-maximum scales

Batched quantized matmul, non-transposed weights, gather quantized matmul, and
loader-owned reconstruction of nested quantization metadata are not yet part of
the native API.

## Verified systems

The same source revision has independently built a native wheel, installed into
a separate build-tool-free environment, and passed the complete native matrix
on both of these systems:

| Hardware | macOS | Python | MLX |
| --- | --- | --- | --- |
| Apple M4 Max (`Mac16,5`) | 15.6 | 3.12.12 | 0.32.2 |
| Apple M2 Pro (`Mac14,9`) | 26.5.1 | 3.12.12 | 0.32.2 |

Each run covered 45 native/reference combinations across all three activation
dtypes, all three group sizes, aligned and tail output widths, and zero-scale
groups, followed by the installed package test suite. These are verified
combinations, not a claim that every Python ABI or macOS release has already
been exercised.

## Develop and test

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

The repository also includes a clean-install harness that builds from an exact
Git revision, uses separate builder and runtime environments, audits the native
wheel, and exercises the installed extension:

```sh
python tools/install_smoke.py \
  --source "$PWD" \
  --expected-revision "$(git rev-parse HEAD)" \
  --mlx-version 0.32.2 \
  --report /absolute/path/to/mlx-nf4-install-smoke.json
```

The harness is described in [`docs/compatibility.md`](docs/compatibility.md).

## Lineage and license

The package is the extension-route continuation of an NF4 implementation first
built inside MLX core. It retains the MLX-derived MIT kernel substrate and its
Apple notices while identifying the NF4 implementation and extraction work
separately. See [`docs/provenance.md`](docs/provenance.md), [`NOTICE`](NOTICE),
and [`LICENSE`](LICENSE).
