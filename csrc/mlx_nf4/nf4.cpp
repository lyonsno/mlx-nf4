#include "mlx_nf4/nf4.h"

#include <dlfcn.h>

#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/dtype.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mx = mlx::core;

namespace mlx_nf4 {
namespace {

std::string current_binary_dir() {
  static std::string binary_dir = []() {
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&current_binary_dir), &info)) {
      throw std::runtime_error("Unable to get mlx-nf4 binary directory.");
    }
    return std::filesystem::path(info.dli_fname).parent_path().string();
  }();
  return binary_dir;
}

std::string metal_type_name(mx::Dtype dtype) {
  if (dtype == mx::float32) {
    return "float";
  }
  if (dtype == mx::float16) {
    return "float16_t";
  }
  if (dtype == mx::bfloat16) {
    return "bfloat16_t";
  }
  std::ostringstream msg;
  msg << "[mlx_nf4.quantized_matmul] unsupported activation dtype " << dtype
      << ".";
  throw std::invalid_argument(msg.str());
}

void validate_group_size(int group_size) {
  if (group_size != 32 && group_size != 64 && group_size != 128) {
    std::ostringstream msg;
    msg << "[mlx_nf4.quantized_matmul] group_size must be 32, 64, or 128 but "
        << "got " << group_size << ".";
    throw std::invalid_argument(msg.str());
  }
}

void validate_shapes(
    const mx::array& x,
    const mx::array& w,
    const mx::array& scales,
    bool transpose,
    int group_size) {
  if (!transpose) {
    throw std::invalid_argument(
        "[mlx_nf4.quantized_matmul] transpose=False is outside the 0.1 "
        "native surface.");
  }
  if (x.ndim() != 2 || w.ndim() != 2 || scales.ndim() != 2) {
    throw std::invalid_argument(
        "[mlx_nf4.quantized_matmul] initial native port supports 2D x, w, "
        "and scales only.");
  }
  if (w.dtype() != mx::uint32) {
    throw std::invalid_argument(
        "[mlx_nf4.quantized_matmul] packed NF4 weights must be uint32.");
  }
  if (scales.dtype() != mx::float32) {
    throw std::invalid_argument(
        "[mlx_nf4.quantized_matmul] NF4 scales must be float32 absmax "
        "scales.");
  }
  const int pack_factor = 8;
  int K = w.shape(-1) * pack_factor;
  int N = w.shape(-2);
  int scale_groups = K / group_size;
  if (K % group_size != 0) {
    throw std::invalid_argument(
        "[mlx_nf4.quantized_matmul] K must be divisible by group_size.");
  }
  if (x.shape(-1) != K) {
    std::ostringstream msg;
    msg << "[mlx_nf4.quantized_matmul] x.shape[-1] must equal unpacked K="
        << K << " but got " << x.shape(-1) << ".";
    throw std::invalid_argument(msg.str());
  }
  if (scales.shape(-2) != N || scales.shape(-1) != scale_groups) {
    throw std::invalid_argument(
        "[mlx_nf4.quantized_matmul] transpose=True scales shape must be "
        "(N, K / group_size).");
  }
}

class NF4QuantizedMatmul : public mx::UnaryPrimitive {
 public:
  NF4QuantizedMatmul(mx::Stream stream, int group_size, bool transpose)
      : mx::UnaryPrimitive(stream),
        group_size_(group_size),
        transpose_(transpose) {}

  void eval_cpu(const std::vector<mx::array>&, mx::array&) override {
    throw std::runtime_error(
        "[mlx_nf4.quantized_matmul] native NF4 qmm is Metal-only.");
  }

  void eval_gpu(const std::vector<mx::array>& inputs, mx::array& out) override {
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    const auto& scales = inputs[2];

    out.set_data(mx::allocator::malloc(out.nbytes()));

    auto& s = stream();
    auto& d = mx::metal::device(s.device);

    int M = x.shape(-2);
    int K = x.shape(-1);
    int N = out.shape(-1);
    int B = out.size() / M / N;
    if (B != 1) {
      throw std::runtime_error(
          "[mlx_nf4.quantized_matmul] batched qmm is not wired yet.");
    }

    int wm = 2;
    int wn = 2;
    int bm = 32;
    int bn = 32;
    MTL::Size group_dims(32, wn, wm);
    MTL::Size grid_dims((N + bn - 1) / bn, (M + bm - 1) / bm, B);

    bool aligned = N % 32 == 0;
    bool batched = false;
    std::string type_string = metal_type_name(x.dtype());
    std::string kname;
    mx::concatenate(
        kname,
        std::string("nf4_") + (transpose_ ? "qmm_t_" : "qmm_n_"),
        type_string,
        "_gs_",
        group_size_,
        "_b_4",
        transpose_ ? (aligned ? "_alN_true" : "_alN_false") : "",
        batched ? "_batch_1" : "_batch_0");

    auto lib = d.get_library("mlx_nf4", current_binary_dir());
    auto kernel = d.get_kernel(kname, lib);
    auto& compute_encoder = mx::metal::get_command_encoder(s);
    compute_encoder.set_compute_pipeline_state(kernel);

    int c = 0;
    compute_encoder.set_input_array(w, c++);
    compute_encoder.set_input_array(scales, c++);
    compute_encoder.set_input_array(x, c++);
    compute_encoder.set_output_array(out, c++);
    compute_encoder.set_bytes(K, c++);
    compute_encoder.set_bytes(N, c++);
    compute_encoder.set_bytes(M, c++);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  bool is_equivalent(const mx::Primitive& other) const override {
    const auto& qmm_other = static_cast<const NF4QuantizedMatmul&>(other);
    return group_size_ == qmm_other.group_size_ &&
        transpose_ == qmm_other.transpose_;
  }

  std::vector<mx::Shape> output_shapes(
      const std::vector<mx::array>& inputs) override {
    const auto& x = inputs[0];
    const auto& w = inputs[1];
    auto out_shape = x.shape();
    out_shape.back() = transpose_ ? w.shape(-2) : w.shape(-1);
    return {out_shape};
  }

  const char* name() const override {
    return "mlx_nf4::NF4QuantizedMatmul";
  }

 private:
  int group_size_;
  bool transpose_;
};

} // namespace

mx::array quantized_matmul(
    mx::array x,
    mx::array w,
    mx::array scales,
    bool transpose,
    int group_size,
    mx::StreamOrDevice s) {
  validate_group_size(group_size);
  validate_shapes(x, w, scales, transpose, group_size);
  metal_type_name(x.dtype());

  auto stream = mx::to_stream(s);
  if (!(stream.device == mx::Device::gpu)) {
    throw std::invalid_argument(
        "[mlx_nf4.quantized_matmul] native NF4 qmm is Metal-only.");
  }

  auto out_shape = x.shape();
  out_shape.back() = transpose ? w.shape(-2) : w.shape(-1);
  return mx::array(
      std::move(out_shape),
      x.dtype(),
      std::make_shared<NF4QuantizedMatmul>(stream, group_size, transpose),
      {std::move(x), std::move(w), std::move(scales)});
}

} // namespace mlx_nf4
