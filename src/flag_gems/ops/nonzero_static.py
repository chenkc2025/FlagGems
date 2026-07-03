import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _nonzero_static_count_kernel(
    x_ptr,
    counts_ptr,
    numel: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    flags = vals != 0

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
def _nonzero_static_write_kernel(
    x_ptr,
    prefix_ptr,
    out_ptr,
    size: tl.constexpr,
    numel: tl.constexpr,
    ndim: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    D2: tl.constexpr,
    D3: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    vals = tl.load(x_ptr + offsets, mask=mask, other=0)
    flags = vals != 0

    prefix = tl.load(prefix_ptr + pid)

    local_rank = tl.cumsum(flags.to(tl.int32), 0) - 1
    global_rank = prefix + local_rank.to(tl.int64)

    write_mask = mask & flags & (global_rank < size)
    linear = offsets.to(tl.int64)

    if ndim == 1:
        c0 = linear
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)

    if ndim == 2:
        c0 = linear // D1
        c1 = linear % D1
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)

    if ndim == 3:
        s0 = D1 * D2
        c0 = linear // s0
        r0 = linear % s0
        c1 = r0 // D2
        c2 = r0 % D2
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 2, c2, mask=write_mask)

    if ndim == 4:
        s0 = D1 * D2 * D3
        s1 = D2 * D3
        c0 = linear // s0
        r0 = linear % s0
        c1 = r0 // s1
        r1 = r0 % s1
        c2 = r1 // D3
        c3 = r1 % D3
        tl.store(out_ptr + global_rank * ndim, c0, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 1, c1, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 2, c2, mask=write_mask)
        tl.store(out_ptr + global_rank * ndim + 3, c3, mask=write_mask)


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

    if ndim > 4:
        raise RuntimeError(
            "nonzero_static Triton implementation only supports ndim <= 4"
        )

    if input.is_complex():
        raise RuntimeError(
            "nonzero_static Triton implementation does not support complex dtype"
        )

    if not input.is_cuda:
        return nonzero_static_ref(input, size=size, fill_value=fill_value)

    out = torch.empty((size, ndim), device=input.device, dtype=torch.int64)

    if size == 0:
        return out

    if ndim == 0:
        return out

    x = input.contiguous()
    numel = x.numel()

    block_size = 1024
    total_out = size * ndim
    fill_grid = (triton.cdiv(total_out, block_size),)

    with torch_device_fn.device(input.device):
        _nonzero_static_fill_kernel[fill_grid](
            out,
            total_out,
            fill_value,
            BLOCK_SIZE=block_size,
        )

    if numel == 0:
        return out

    num_blocks = triton.cdiv(numel, block_size)
    counts = torch.empty((num_blocks,), device=input.device, dtype=torch.int64)

    with torch_device_fn.device(input.device):
        _nonzero_static_count_kernel[(num_blocks,)](
            x,
            counts,
            numel,
            BLOCK_SIZE=block_size,
        )

    prefix = torch.cumsum(counts, dim=0) - counts
    shape = list(x.shape) + [1] * (4 - ndim)

    with torch_device_fn.device(input.device):
        _nonzero_static_write_kernel[(num_blocks,)](
            x,
            prefix,
            out,
            size,
            numel,
            ndim,
            shape[0],
            shape[1],
            shape[2],
            shape[3],
            BLOCK_SIZE=block_size,
        )

    return out
