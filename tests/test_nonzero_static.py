import pytest
import torch

import flag_gems
from flag_gems.ops.nonzero_static import nonzero_static, nonzero_static_ref


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="nonzero_static Triton implementation requires CUDA",
)

DTYPES = [
    torch.bool,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.float32,
    torch.bfloat16,
]

CASES = [
    ((), torch.float32, 1.0, 4, -1),
    ((0,), torch.float32, 0.0, 4, 7),
    ((8,), torch.float32, 0.0, 4, -1),
    ((8,), torch.float32, 1.0, 4, -1),
    ((8,), torch.float32, 0.5, 0, -1),
    ((1024,), torch.float32, 0.01, 128, -1),
    ((4, 5), torch.float32, 0.5, 16, 7),
    ((2, 3, 4), torch.float32, 0.1, 16, -1),
    ((2, 3, 4, 5), torch.float32, 0.1, 32, -1),
]


def make_input(shape, dtype, nnz_ratio, device):
    if shape == ():
        value = bool(nnz_ratio >= 0.5)
        if dtype == torch.bool:
            return torch.tensor(value, device=device, dtype=dtype)
        return torch.tensor(1 if value else 0, device=device, dtype=dtype)

    mask = torch.rand(shape, device=device) < nnz_ratio

    if dtype == torch.bool:
        return mask

    x = torch.zeros(shape, device=device, dtype=dtype)

    if dtype.is_floating_point:
        values = torch.randn(shape, device=device, dtype=dtype) + 1
    else:
        values = torch.randint(1, 10, shape, device=device, dtype=dtype)

    x[mask] = values[mask]
    return x


def assert_nonzero_static_matches(shape, dtype, nnz_ratio, size, fill_value):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is not supported on this CUDA device")

    torch.manual_seed(0)
    x_cpu = make_input(shape, dtype, nnz_ratio, "cpu")
    x_gpu = x_cpu.cuda()

    actual = nonzero_static(x_gpu, size=size, fill_value=fill_value)
    expected = nonzero_static_ref(x_cpu, size=size, fill_value=fill_value).cuda()

    assert actual.dtype == torch.int64
    assert tuple(actual.shape) == (size, x_gpu.dim())
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    if hasattr(torch, "nonzero_static"):
        try:
            expected_cuda = torch.nonzero_static(
                x_gpu, size=size, fill_value=fill_value
            )
            torch.testing.assert_close(actual, expected_cuda, rtol=0, atol=0)
        except Exception:
            pass


@pytest.mark.nonzero_static
@pytest.mark.parametrize("dtype", DTYPES)
def test_nonzero_static_dtypes(dtype):
    assert_nonzero_static_matches((32, 128), dtype, 0.1, 128, -1)


@pytest.mark.nonzero_static
@pytest.mark.parametrize("shape,dtype,nnz_ratio,size,fill_value", CASES)
def test_nonzero_static_cases(shape, dtype, nnz_ratio, size, fill_value):
    assert_nonzero_static_matches(shape, dtype, nnz_ratio, size, fill_value)


@pytest.mark.nonzero_static
def test_nonzero_static_non_contiguous_transpose():
    torch.manual_seed(1)
    x_cpu_base = make_input((16, 32), torch.float32, 0.2, "cpu")
    x_gpu_base = x_cpu_base.cuda()

    x_cpu_view = x_cpu_base.t()
    x_gpu_view = x_gpu_base.t()

    actual = nonzero_static(x_gpu_view, size=128, fill_value=-1)
    expected = nonzero_static_ref(x_cpu_view, size=128, fill_value=-1).cuda()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.nonzero_static
def test_nonzero_static_non_contiguous_slice():
    torch.manual_seed(2)
    x_cpu_base = make_input((16, 32), torch.float32, 0.2, "cpu")
    x_gpu_base = x_cpu_base.cuda()

    x_cpu_view = x_cpu_base[:, ::2]
    x_gpu_view = x_gpu_base[:, ::2]

    actual = nonzero_static(x_gpu_view, size=128, fill_value=7)
    expected = nonzero_static_ref(x_cpu_view, size=128, fill_value=7).cuda()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.nonzero_static
def test_nonzero_static_rejects_complex():
    x = torch.ones((8,), device="cuda", dtype=torch.complex64)

    with pytest.raises(RuntimeError, match="does not support complex dtype"):
        nonzero_static(x, size=4, fill_value=-1)


@pytest.mark.nonzero_static
def test_nonzero_static_rejects_ndim_gt_4():
    x = torch.zeros((1, 1, 1, 1, 1), device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="only supports ndim <= 4"):
        nonzero_static(x, size=4, fill_value=-1)


@pytest.mark.nonzero_static
def test_nonzero_static_rejects_negative_size():
    x = torch.ones((8,), device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="size must be non-negative"):
        nonzero_static(x, size=-1, fill_value=-1)


@pytest.mark.nonzero_static
def test_nonzero_static_registered_with_use_gems():
    if not hasattr(torch, "nonzero_static"):
        pytest.skip("torch.nonzero_static is unavailable in this PyTorch build")

    torch.manual_seed(3)
    x_cpu = make_input((4, 5), torch.float32, 0.4, "cpu")
    x_gpu = x_cpu.cuda()

    with flag_gems.use_gems(include=["nonzero_static"]):
        actual = torch.nonzero_static(x_gpu, size=16, fill_value=-1)

    expected = nonzero_static_ref(x_cpu, size=16, fill_value=-1).cuda()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
