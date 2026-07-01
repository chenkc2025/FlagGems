#include <algorithm>

#include <c10/core/TensorImpl.h>

#include "flag_gems/operators.h"

namespace flag_gems {
namespace {

int64_t check_split_dim(const at::Tensor& self, int64_t dim) {
  const int64_t ndim = self.dim();
  TORCH_CHECK(ndim > 0, "split expects at least a 1-dimensional tensor");
  return at::maybe_wrap_dim(dim, ndim);
}

at::Tensor split_view(
    const at::Tensor& self,
    int64_t dim,
    int64_t start,
    int64_t length) {
  std::vector<int64_t> sizes = self.sizes().vec();
  std::vector<int64_t> strides = self.strides().vec();
  sizes[dim] = length;
  const int64_t storage_offset = self.storage_offset() + start * self.stride(dim);
  return self.as_strided(sizes, strides, storage_offset);
}

void reset_version_counters(std::vector<at::Tensor>& outs) {
  for (at::Tensor& out : outs) {
    if (!out.is_inference()) {
      out.unsafeGetTensorImpl()->set_version_counter(c10::VariableVersion(0));
    }
  }
}

}  // namespace

std::vector<at::Tensor> unsafe_split(
    const at::Tensor& self,
    c10::SymInt split_size,
    int64_t dim) {
  const int64_t split_size_int = split_size.expect_int();
  TORCH_CHECK(split_size_int >= 0,
              "split expects split_size be non-negative, but got split_size=",
              split_size_int);

  dim = check_split_dim(self, dim);
  const int64_t dim_size = self.size(dim);
  TORCH_CHECK(split_size_int != 0 || dim_size == 0,
              "split_size can only be 0 if dimension size is 0, but got dimension size of ",
              dim_size);

  if (dim_size == 0) {
    std::vector<at::Tensor> outs = {split_view(self, dim, 0, 0)};
    reset_version_counters(outs);
    return outs;
  }

  const int64_t num_splits = (dim_size + split_size_int - 1) / split_size_int;
  std::vector<at::Tensor> outs;
  outs.reserve(num_splits);
  for (int64_t start = 0; start < dim_size; start += split_size_int) {
    const int64_t length = std::min(split_size_int, dim_size - start);
    outs.emplace_back(split_view(self, dim, start, length));
  }
  reset_version_counters(outs);
  return outs;
}

std::vector<at::Tensor> unsafe_split_with_sizes(
    const at::Tensor& self,
    c10::SymIntArrayRef split_sizes,
    int64_t dim) {
  dim = check_split_dim(self, dim);

  int64_t total = 0;
  std::vector<int64_t> split_sizes_int;
  split_sizes_int.reserve(split_sizes.size());
  for (const c10::SymInt& split_size : split_sizes) {
    const int64_t split_size_int = split_size.expect_int();
    TORCH_CHECK(split_size_int >= 0,
                "split_with_sizes expects split_sizes have only non-negative entries");
    split_sizes_int.push_back(split_size_int);
    total += split_size_int;
  }

  const int64_t dim_size = self.size(dim);
  TORCH_CHECK(total == dim_size,
              "split_with_sizes expects split_sizes to sum exactly to ",
              dim_size,
              " (input tensor's size at dimension ",
              dim,
              "), but got split_sizes=",
              split_sizes_int);

  std::vector<at::Tensor> outs;
  outs.reserve(split_sizes_int.size());
  int64_t start = 0;
  for (const int64_t length : split_sizes_int) {
    outs.emplace_back(split_view(self, dim, start, length));
    start += length;
  }
  reset_version_counters(outs);
  return outs;
}

}  // namespace flag_gems
