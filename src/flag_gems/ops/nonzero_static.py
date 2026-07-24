import logging

import flag_gems
import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

DEFAULT_BLOCK_SIZE = 1024


@triton.jit
def _load_nonzero_flags(
    x_ptr,
    offsets,
    mask,
    IS_COMPLEX: tl.constexpr,
):
    if IS_COMPLEX:
        complex_offsets = offsets * 2
        real = tl.load(x_ptr + complex_offsets, mask=mask, other=0)
        imag = tl.load(x_ptr + complex_offsets + 1, mask=mask, other=0)
        return (real != 0) | (imag != 0)

    values = tl.load(x_ptr + offsets, mask=mask, other=0)
    return values != 0


@libentry()
@triton.jit
def _nonzero_static_count_kernel(
    x_ptr,
    counts_ptr,
    numel: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    flags = _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)

    cnt = tl.sum(flags.to(tl.int32), axis=0)
    tl.store(counts_ptr + pid, cnt.to(tl.int64))


@libentry()
@triton.jit
def _nonzero_static_fill_kernel(
    out_ptr,
    total_out: tl.constexpr,
    fill_value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out

    vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + offsets, vals, mask=mask)


@libentry()
@triton.jit
def _nonzero_static_fill_tail_kernel(
    out_ptr,
    prefix_ptr,
    num_blocks: tl.constexpr,
    size: tl.constexpr,
    ndim: tl.constexpr,
    fill_value: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    total_nnz = tl.load(prefix_ptr + num_blocks - 1)
    valid_rows = tl.minimum(total_nnz, size)
    tail_start = valid_rows * ndim
    total_out = size * ndim

    pid = tl.program_id(0)
    offsets = tail_start + pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out

    vals = tl.full((BLOCK_SIZE,), fill_value, tl.int64)
    tl.store(out_ptr + offsets, vals, mask=mask)


@libentry()
@triton.jit
def _nonzero_static_write_kernel(
    x_ptr,
    prefix_ptr,
    counts_ptr,
    out_ptr,
    shape_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    ndim: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    flags = _load_nonzero_flags(x_ptr, offsets, mask, IS_COMPLEX)

    block_nnz = tl.load(counts_ptr + pid)
    prefix = tl.load(prefix_ptr + pid) - block_nnz

    local_rank = tl.cumsum(flags.to(tl.int32), 0) - 1
    global_rank = prefix + local_rank.to(tl.int64)

    write_mask = mask & flags & (global_rank < size)
    linear = offsets.to(tl.int64)

    for dim in range(ndim - 1, -1, -1):
        dim_size = tl.load(shape_ptr + dim)
        coord = linear % dim_size
        linear //= dim_size
        tl.store(out_ptr + global_rank * ndim + dim, coord, mask=write_mask)


def nonzero_static_ref(x: torch.Tensor, size: int, fill_value: int = -1):
    if size < 0:
        raise RuntimeError("nonzero_static: size must be non-negative")

    size = int(size)
    fill_value = int(fill_value)
    ndim = x.dim()
    out = torch.empty((size, ndim), device=x.device, dtype=torch.long)

    if size == 0:
        return out

    if ndim == 0:
        return out

    nz = torch.nonzero(x, as_tuple=False)
    copy_len = min(size, nz.shape[0])

    if copy_len > 0:
        out[:copy_len].copy_(nz[:copy_len])

    if copy_len < size:
        out[copy_len:].fill_(fill_value)

    return out


def nonzero_static(input: torch.Tensor, size: int, fill_value: int = -1):
    logger.debug("GEMS NONZERO_STATIC")

    if size < 0:
        raise RuntimeError("nonzero_static: size must be non-negative")

    size = int(size)
    fill_value = int(fill_value)
    ndim = input.dim()

    if not input.is_cuda:
        return nonzero_static_ref(input, size=size, fill_value=fill_value)

    out = torch.empty((size, ndim), device=input.device, dtype=torch.int64)

    if size == 0:
        return out

    if ndim == 0:
        return out

    x = input.contiguous()
    numel = x.numel()
    is_complex = x.is_complex()

    block_size = DEFAULT_BLOCK_SIZE
    total_out = size * ndim

    if numel == 0:
        fill_grid = (triton.cdiv(total_out, block_size),)
        with torch_device_fn.device(input.device):
            _nonzero_static_fill_kernel[fill_grid](
                out,
                total_out,
                fill_value,
                BLOCK_SIZE=block_size,
            )
        return out

    shape = torch.tensor(input.shape, dtype=torch.int64, device=input.device)
    if is_complex:
        x = torch.view_as_real(x).reshape(-1)
    num_blocks = triton.cdiv(numel, block_size)
    counts = torch.empty((num_blocks,), device=input.device, dtype=torch.int64)

    with torch_device_fn.device(input.device):
        _nonzero_static_count_kernel[(num_blocks,)](
            x,
            counts,
            numel,
            IS_COMPLEX=is_complex,
            BLOCK_SIZE=block_size,
        )

    prefix = flag_gems.cumsum(counts, dim=0)

    with torch_device_fn.device(input.device):
        _nonzero_static_write_kernel[(num_blocks,)](
            x,
            prefix,
            counts,
            out,
            shape,
            size,
            numel,
            ndim,
            IS_COMPLEX=is_complex,
            BLOCK_SIZE=block_size,
        )

    fill_grid = (triton.cdiv(total_out, block_size),)
    with torch_device_fn.device(input.device):
        _nonzero_static_fill_tail_kernel[fill_grid](
            out,
            prefix,
            num_blocks,
            size,
            ndim,
            fill_value,
            BLOCK_SIZE=block_size,
        )

    return out
