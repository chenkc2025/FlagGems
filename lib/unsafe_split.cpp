#include <algorithm>

#include <c10/core/TensorImpl.h>
#include <c10/util/SmallVector.h>

#include "flag_gems/operators.h"

namespace flag_gems {
namespace {

int64_t check_split_dim(const at::Tensor& self, int64_t dim) {
  const int64_t ndim = self.dim();
  TORCH_CHECK(ndim > 0, "split expects at least a 1-dimensional tensor");
  return at::maybe_wrap_dim(dim, ndim);
}

at::Tensor split_view(const at::Tensor& self,
                      c10::SmallVector<int64_t, 8>& sizes,
                      const c10::SmallVector<int64_t, 8>& strides,
                      int64_t dim,
                      int64_t start,
                      int64_t length,
                      int64_t base_storage_offset,
                      int64_t dim_stride) {
  sizes[dim] = length;
  const int64_t storage_offset = base_storage_offset + start * dim_stride;
  return self.as_strided(sizes, strides, storage_offset);
}

void reset_version_counter(at::Tensor& out) {
  if (!out.is_inference()) {
    out.unsafeGetTensorImpl()->set_version_counter(c10::VariableVersion(0));
  }
}

}  // namespace

std::vector<at::Tensor> unsafe_split(
    const at::Tensor& self,
    int64_t split_size_int,
    int64_t dim) {
  TORCH_CHECK(split_size_int >= 0,
              "split expects split_size be non-negative, but got split_size=",
              split_size_int);

  dim = check_split_dim(self, dim);
  const int64_t dim_size = self.size(dim);
  TORCH_CHECK(split_size_int != 0 || dim_size == 0,
              "split_size can only be 0 if dimension size is 0, but got dimension size of ",
              dim_size);

  c10::SmallVector<int64_t, 8> sizes(self.sizes().begin(), self.sizes().end());
  const c10::SmallVector<int64_t, 8> strides(self.strides().begin(), self.strides().end());
  const int64_t base_storage_offset = self.storage_offset();
  const int64_t dim_stride = self.stride(dim);

  if (dim_size == 0) {
    std::vector<at::Tensor> outs;
    outs.emplace_back(split_view(
        self, sizes, strides, dim, 0, 0, base_storage_offset, dim_stride));
    reset_version_counter(outs.back());
    return outs;
  }

  const int64_t num_splits = (dim_size + split_size_int - 1) / split_size_int;
  std::vector<at::Tensor> outs;
  outs.reserve(num_splits);
  for (int64_t start = 0; start < dim_size; start += split_size_int) {
    const int64_t length = std::min(split_size_int, dim_size - start);
    outs.emplace_back(split_view(
        self, sizes, strides, dim, start, length, base_storage_offset, dim_stride));
    reset_version_counter(outs.back());
  }
  return outs;
}

std::vector<at::Tensor> unsafe_split_with_sizes(
    const at::Tensor& self,
    at::IntArrayRef split_sizes,
    int64_t dim) {
  dim = check_split_dim(self, dim);
  const int64_t dim_size = self.size(dim);

  c10::SmallVector<int64_t, 8> sizes(self.sizes().begin(), self.sizes().end());
  const c10::SmallVector<int64_t, 8> strides(self.strides().begin(), self.strides().end());
  const int64_t base_storage_offset = self.storage_offset();
  const int64_t dim_stride = self.stride(dim);

  std::vector<at::Tensor> outs;
  outs.reserve(split_sizes.size());
  int64_t start = 0;
  for (const int64_t length : split_sizes) {
    TORCH_CHECK(length >= 0,
                "split_with_sizes expects split_sizes have only non-negative entries");
    outs.emplace_back(split_view(
        self, sizes, strides, dim, start, length, base_storage_offset, dim_stride));
    reset_version_counter(outs.back());
    start += length;
  }
  TORCH_CHECK(start == dim_size,
              "split_with_sizes expects split_sizes to sum exactly to ",
              dim_size,
              " (input tensor's size at dimension ",
              dim,
              ")");
  return outs;
}

}  // namespace flag_gems
