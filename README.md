# mlx-nf4

NF4 / NormalFloat4 quantization helpers and package-local Metal kernels for MLX.

This package is the extension-route continuation of the declined MLX core NF4 PR. It does not monkey-patch MLX and does not claim built-in `mx.quantized_matmul(..., mode="nf4")` support.

Current slice:

- `mlx_nf4.quantize`
- `mlx_nf4.dequantize`
- `mlx_nf4.pack_uint8_to_uint32`
- `mlx_nf4.quantized_matmul`
- `mlx_nf4.reference_quantized_matmul`
- `mlx_nf4.NF4Linear`

`mlx_nf4.quantized_matmul` dispatches to a package-local C++/Metal extension. The explicit `reference_quantized_matmul` helper is the slow dequantize-then-matmul path for correctness tests.

## Packed Weight Intake

The package-native layout stores eight NF4 indices per `uint32`. For loaders that receive byte-packed weights, use:

```python
import mlx_nf4 as nf4

wq = nf4.pack_uint8_to_uint32(byte_weight, source_order="high_first")
layer = nf4.NF4Linear.from_packed(wq, scales, bias=bias)
```

`source_order="high_first"` handles the nibble order used by bitsandbytes NF4 tensors observed in the companion model smoke. `source_order="low_first"` is the package-native logical byte order.

For the common bitsandbytes tensor boundary:

```python
layer = nf4.NF4Linear.from_bitsandbytes(byte_weight, scales, bias=bias)
```

`from_bitsandbytes` expects already reconstructed float32 absmax scales. It does not parse safetensors files or resolve bitsandbytes nested/double-quantized `quant_state`; loader code should reconstruct those tensors before calling into `mlx_nf4`.

## Build

```sh
python -m pip install -e .
```

For local development against an MLX source checkout:

```sh
PYTHONPATH=/path/to/mlx/python python setup.py build_ext --inplace
```

The build produces `_ext`, `libmlx_nf4_native.dylib`, and `mlx_nf4.metallib` inside the `mlx_nf4` package directory.

## Current Native Scope

The first native port supports 2D NF4 qmm on Metal:

- activation dtypes: `float32`, `float16`, `bfloat16`
- group sizes: `32`, `64`, `128`
- `transpose=True`
- no batched qmm yet
- no `gather_qmm`

## Test Smoke

After installation:

```text
python -m unittest discover -s tests -v
Ran 14 tests
OK
```

Bounded timing smoke for `x=(128, 2048)`, packed `w=(2048, 256)`, `group_size=64`, 20 iterations:

```text
mlx_nf4.quantized_matmul: 0.245 ms
reference_dequantize_matmul: 0.514 ms
core_mx.quantized_matmul_nf4: 0.228 ms
```
