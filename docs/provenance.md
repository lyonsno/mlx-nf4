# Provenance

This package is the extension-route continuation of an NF4 implementation that
was first developed as an MLX core patch. The core proposal was not accepted:
maintainers preferred to avoid the additional maintenance surface and pointed
the work toward MLX custom extensions and custom Metal kernels. `mlx-nf4`
follows that boundary. It neither monkey-patches MLX nor advertises a built-in
`mlx.core.quantized_matmul(..., mode="nf4")` capability.

The source lineage begins with NF4 work authored by Noah Lyons in the MLX fork
at `eccb857825439dd8c24c5cc5d27e9489fc2f4eef`; the extracted native package was
subsequently adapted to load its own dynamic library and Metal library beside
the Python module. The Metal code retains the Apple copyright headers inherited
from the MIT-licensed MLX kernel substrate. See `LICENSE` and `NOTICE` for the
distribution terms and attribution boundary.

The 16-entry NormalFloat4 lookup table is the standard NF4 codebook. The
package's intake helpers support its native low-nibble-first byte layout and a
high-nibble-first boundary for bitsandbytes-style packed tensors. The latter
expects already reconstructed float32 absolute-maximum scales; safetensors
parsing and nested/double-quantization reconstruction remain loader concerns.
