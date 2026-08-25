#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "mlx_nf4/nf4.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(_ext, m) {
  m.doc() = "Native NF4 extension kernels for MLX";
  m.def(
      "quantized_matmul",
      &mlx_nf4::quantized_matmul,
      "x"_a,
      "w"_a,
      "scales"_a,
      nb::kw_only(),
      "transpose"_a = true,
      "group_size"_a = 64,
      "stream"_a = nb::none(),
      R"(
        Package-local NF4 quantized matmul.

        This dispatches to the mlx-nf4 Metal kernels and does not use the
        reference dequantize-then-matmul path.
      )");
}
