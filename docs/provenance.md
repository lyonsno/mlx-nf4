# Provenance

## From MLX core kernel to standalone extension

`mlx-nf4` began as an NF4 implementation inside an MLX fork. That work added
the NormalFloat4 codebook, packed-weight handling, host dispatch, and Metal
quantized-matmul kernels directly to MLX core. The source lineage begins with
Noah Lyons's NF4 work at commit
`eccb857825439dd8c24c5cc5d27e9489fc2f4eef`.

The MLX maintainers preferred not to add the continuing maintenance surface to
core and pointed the implementation toward MLX custom extensions and custom
Metal kernels. The standalone package is that continuation: the kernel,
dispatch, and Python layer now build and load beside `mlx_nf4` while consuming
an ordinary installed MLX release.

The extraction preserved the central result. In the original M4 Max assay, the
standalone kernel measured 0.245 ms against 0.228 ms for the MLX-core version
and 0.514 ms for explicit dequantization followed by matmul at the tested
shape. The package therefore carries the native packed-weight mechanism rather
than replacing the core patch with a Python-only compatibility layer.

## NF4 representation

The 16-entry lookup table is the standard NormalFloat4 codebook. Native package
weights store eight four-bit indices per `uint32`, accompanied by one float32
absolute-maximum scale per quantization group.

The package can also ingest byte-packed weights. `source_order="low_first"`
describes its native logical order; `source_order="high_first"` handles the
nibble convention used by observed bitsandbytes NF4 tensors.

The bitsandbytes boundary is checked against a prefix from the public
`ideogram-ai/ideogram-4-nf4` safetensors artifact. The fixture records the model
revision, source blob, tensor and quant-state shapes, observed bytes, decoded
logical indices, expected native words, and the upstream bitsandbytes source
revision:

[`tests/fixtures/ideogram4_input_proj_bitsandbytes_nf4.json`](../tests/fixtures/ideogram4_input_proj_bitsandbytes_nf4.json)

`NF4Linear.from_bitsandbytes` intentionally begins after loader work has
reconstructed float32 absolute-maximum scales. Safetensors parsing and nested
or double-quantized scale reconstruction are separate model-loader concerns.

## Attribution and licensing

The package-local C++ and Metal implementation derives from MLX extension and
quantized-kernel interfaces distributed under the MIT License. Apple copyright
headers remain in the inherited kernel sources. NF4 implementation and package
extraction authorship are identified separately in [`NOTICE`](../NOTICE).

The complete distribution terms are in [`LICENSE`](../LICENSE).
