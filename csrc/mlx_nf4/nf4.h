#pragma once

#include <optional>

#include "mlx/array.h"
#include "mlx/stream.h"
#include "mlx/utils.h"

namespace mlx_nf4 {

mlx::core::array quantized_matmul(
    mlx::core::array x,
    mlx::core::array w,
    mlx::core::array scales,
    bool transpose = true,
    int group_size = 64,
    mlx::core::StreamOrDevice s = {});

} // namespace mlx_nf4
