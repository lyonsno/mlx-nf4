# mlx-nf4

Package-local NF4 / NormalFloat4 quantization and native Metal matmul kernels
for [MLX](https://github.com/ml-explore/mlx).

`mlx-nf4` carries the NF4 work as an MLX custom extension: it does not patch the
installed `mlx` package, and it does not claim that
`mlx.core.quantized_matmul(..., mode="nf4")` exists. The fast path loads a C++
binding, dynamic library, and compiled Metal library shipped inside this
package. The explicit reference path dequantizes first and exists for correctness
checks.

## Install

The package builds from source on Apple silicon and requires Xcode command-line
tools, Python 3.10 or newer, and MLX 0.32.2 or newer:

```sh
git clone https://github.com/lyonsno/mlx-nf4.git
cd mlx-nf4
python -m pip install .
```

For an exact revision without a persistent checkout:

```sh
python -m pip install \
  "git+https://github.com/lyonsno/mlx-nf4.git@<commit>"
```

The build produces `_ext`, `libmlx_nf4_native.dylib`, and
`mlx_nf4.metallib` inside the installed `mlx_nf4` package. Editable installs
are useful for development, but the compatibility receipts use newly built,
non-editable wheels in fresh environments. See
[`docs/compatibility.md`](docs/compatibility.md) for the exact command and the
evidence contract.

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

For a correctness oracle, call the slower path by name:

```python
y_reference = nf4.reference_quantized_matmul(
    x,
    packed_weight,
    scales,
    transpose=True,
    group_size=64,
)
```

`quantized_matmul` fails if the native extension is absent. It never silently
falls back to dequantize-then-matmul and therefore cannot turn missing native
code into a false performance result.

## NF4Linear

```python
import mlx.nn as nn
import mlx_nf4 as nf4

linear = nn.Linear(64, 32)
quantized = nf4.NF4Linear.from_linear(linear, group_size=64)
y = quantized(x)
```

The module also accepts already packed tensors. Its native layout stores eight
NF4 indices per `uint32`:

```python
layer = nf4.NF4Linear.from_packed(packed_weight, scales, bias=bias)
```

For byte-packed weights, `source_order` states the nibble order explicitly:

```python
packed_weight = nf4.pack_uint8_to_uint32(
    byte_weight,
    source_order="high_first",
)
layer = nf4.NF4Linear.from_packed(packed_weight, scales, bias=bias)
```

`NF4Linear.from_bitsandbytes(byte_weight, scales, bias=bias)` is the compact
form of the high-nibble-first bitsandbytes NF4 boundary. That packing contract
is checked against a real Ideogram4 NF4 safetensors prefix in
[`tests/fixtures/ideogram4_input_proj_bitsandbytes_nf4.json`](tests/fixtures/ideogram4_input_proj_bitsandbytes_nf4.json),
with the model revision, source blob, tensor quant state, expected logical
indices, and the pinned upstream bitsandbytes kernel source recorded beside the
bytes. The constructor expects float32 absolute-maximum scales that have already
been reconstructed. It does not parse safetensors or resolve nested/double-
quantized `quant_state`.

## Native scope in 0.1

- two-dimensional activation, weight, and scale tensors
- `transpose=True`
- group sizes 32, 64, and 128
- float32, float16, and bfloat16 activations
- packed `uint32` weights and float32 absolute-maximum scales

Batched quantized matmul, non-transposed weights, and gather quantized matmul
are not implemented in this release surface.

## Development and verification

```sh
python -m pip install -e .
python -m unittest discover -s tests -v
```

The release-grade clean install is stricter:

```sh
python tools/install_smoke.py \
  --source "$PWD" \
  --expected-revision "$(git rev-parse HEAD)" \
  --mlx-version 0.32.2 \
  --report /absolute/path/to/mlx-nf4-install-smoke.json
```

That check builds a fresh source distribution and wheel in builder environment
A, rejects absolute builder paths in the wheel's Mach-O payloads, then creates
a separate build-tool-free runtime environment B containing only stock MLX and
the non-editable wheel. Runtime B exercises 45 native/reference cases spanning
float32, float16, bfloat16, every supported group size, aligned and tail output
rows, and zero-scale groups before running the installed core tests. The JSON
receipt records the effective Python, macOS, architecture, hardware model,
Metal device, both environments, load commands, artifacts, and per-case errors.
It fails loud on fallback imports, stale revisions, version substitution,
missing binaries, builder/runtime aliasing, leaked rpaths, incomplete case
matrices, blank output, or zero-test pseudo-success.

## Performance context

An early extraction smoke on an M4 Max measured the following single-call
latencies for `x=(128, 2048)`, packed `w=(2048, 256)`, group size 64, and 20
timed iterations:

| Path | Median latency |
| --- | ---: |
| package-local native NF4 | 0.245 ms |
| explicit dequantize then matmul | 0.514 ms |
| original MLX-core NF4 patch | 0.228 ms |

Those numbers are provenance for the extraction decision, not a portable
benchmark claim. Hardware, MLX revision, shapes, dtypes, and warm-up determine
the result; run the same workload in the consuming environment before making a
deployment decision.

## Provenance and license

The implementation began as an MLX core patch and was extracted after the
maintainers declined the additional core maintenance surface and identified
custom extensions as the appropriate route. The package preserves the MIT
license and Apple notices inherited from MLX while separately identifying the
NF4 implementation and extraction authorship. See
[`docs/provenance.md`](docs/provenance.md), [`NOTICE`](NOTICE), and
[`LICENSE`](LICENSE).
