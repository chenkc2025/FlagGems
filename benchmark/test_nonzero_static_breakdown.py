import argparse
import csv
import sys
import time
from pathlib import Path

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flag_gems.ops.nonzero_static import (  # noqa: E402
    DEFAULT_BLOCK_SIZE,
    _nonzero_static_count_kernel,
    _nonzero_static_fill_kernel,
    _nonzero_static_fill_tail_kernel,
    _nonzero_static_write_kernel,
    nonzero_static,
)
from flag_gems.runtime import torch_device_fn  # noqa: E402


DEFAULT_SHAPES = [(1048576,), (1024, 4096)]
DEFAULT_NNZ_RATIOS = [0.001, 0.1, 1.0]
DEFAULT_SIZES = [128, 1024, 4096]
DEFAULT_FILL_VALUE = -1


def parse_csv(value, cast):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_shapes(value):
    specs = parse_csv(value, str)
    if specs is None:
        return None
    return [tuple(int(dim) for dim in spec.split("x") if dim) for spec in specs]


def make_input(shape, nnz_ratio, device):
    mask = torch.rand(shape, device=device) < nnz_ratio
    x = torch.zeros(shape, device=device, dtype=torch.float32)
    values = torch.randn(shape, device=device, dtype=torch.float32) + 1
    x[mask] = values[mask]
    return x


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def bench(fn, warmup=20, repeat=100):
    for _ in range(warmup):
        fn()
    sync()

    start = time.perf_counter()
    for _ in range(repeat):
        fn()
    sync()
    end = time.perf_counter()

    return (end - start) * 1000 / repeat


def baseline_nonzero_static(x, size, fill_value):
    try:
        return torch.nonzero_static(x, size=size, fill_value=fill_value)
    except Exception:
        ndim = x.dim()
        out = torch.empty((size, ndim), device=x.device, dtype=torch.long)
        nz = torch.nonzero(x, as_tuple=False)
        copy_len = min(size, nz.shape[0])
        if copy_len > 0:
            out[:copy_len].copy_(nz[:copy_len])
        if copy_len < size:
            out[copy_len:].fill_(fill_value)
        return out


def run_count(x, counts, block_size):
    numel = x.numel()
    num_blocks = triton.cdiv(numel, block_size)
    with torch_device_fn.device(x.device):
        _nonzero_static_count_kernel[(num_blocks,)](
            x,
            counts,
            numel,
            BLOCK_SIZE=block_size,
        )


def run_write(x, prefix, out, size, block_size):
    ndim = x.dim()
    numel = x.numel()
    num_blocks = triton.cdiv(numel, block_size)
    shape = list(x.shape) + [1] * (4 - ndim)
    with torch_device_fn.device(x.device):
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


def run_fill_tail(out, prefix, counts, size, fill_value, block_size):
    ndim = out.shape[1]
    total_out = size * ndim
    fill_grid = (triton.cdiv(total_out, block_size),)
    with torch_device_fn.device(out.device):
        _nonzero_static_fill_tail_kernel[fill_grid](
            out,
            prefix,
            counts,
            counts.numel(),
            size,
            ndim,
            fill_value,
            BLOCK_SIZE=block_size,
        )


def run_fill_all(out, size, fill_value, block_size):
    ndim = out.shape[1]
    total_out = size * ndim
    fill_grid = (triton.cdiv(total_out, block_size),)
    with torch_device_fn.device(out.device):
        _nonzero_static_fill_kernel[fill_grid](
            out,
            total_out,
            fill_value,
            BLOCK_SIZE=block_size,
        )


def collect_rows(shapes, ratios, sizes, fill_value, warmup, repeat):
    rows = []
    device = torch.device("cuda")
    for shape in shapes:
        for ratio in ratios:
            torch.manual_seed(0)
            x = make_input(shape, ratio, device).contiguous()
            num_blocks = triton.cdiv(x.numel(), DEFAULT_BLOCK_SIZE)
            counts = torch.empty((num_blocks,), device=device, dtype=torch.int64)

            run_count(x, counts, DEFAULT_BLOCK_SIZE)
            sync()
            prefix = torch.cumsum(counts, dim=0) - counts
            sync()

            for size in sizes:
                out = torch.empty((size, x.dim()), device=device, dtype=torch.int64)

                count_ms = bench(
                    lambda: run_count(x, counts, DEFAULT_BLOCK_SIZE),
                    warmup,
                    repeat,
                )
                cumsum_ms = bench(
                    lambda: torch.cumsum(counts, dim=0) - counts,
                    warmup,
                    repeat,
                )
                write_ms = bench(
                    lambda: run_write(x, prefix, out, size, DEFAULT_BLOCK_SIZE),
                    warmup,
                    repeat,
                )
                fill_tail_ms = bench(
                    lambda: run_fill_tail(
                        out,
                        prefix,
                        counts,
                        size,
                        fill_value,
                        DEFAULT_BLOCK_SIZE,
                    ),
                    warmup,
                    repeat,
                )
                fill_all_ms = bench(
                    lambda: run_fill_all(out, size, fill_value, DEFAULT_BLOCK_SIZE),
                    warmup,
                    repeat,
                )
                total_ms = bench(
                    lambda: nonzero_static(x, size=size, fill_value=fill_value),
                    warmup,
                    repeat,
                )
                torch_ms = bench(
                    lambda: baseline_nonzero_static(x, size, fill_value),
                    warmup,
                    repeat,
                )

                rows.append(
                    {
                        "shape": "x".join(str(dim) for dim in shape),
                        "numel": x.numel(),
                        "nnz_ratio": ratio,
                        "size": size,
                        "count_ms": f"{count_ms:.6f}",
                        "cumsum_ms": f"{cumsum_ms:.6f}",
                        "write_ms": f"{write_ms:.6f}",
                        "fill_tail_ms": f"{fill_tail_ms:.6f}",
                        "fill_all_ms": f"{fill_all_ms:.6f}",
                        "total_ms": f"{total_ms:.6f}",
                        "torch_ms": f"{torch_ms:.6f}",
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--shapes", default=None, help="Example: 1048576,1024x4096")
    parser.add_argument("--ratios", default=None, help="Example: 0.001,0.1,1")
    parser.add_argument("--sizes", default=None, help="Example: 128,1024,4096")
    parser.add_argument("--fill-value", type=int, default=DEFAULT_FILL_VALUE)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return

    rows = collect_rows(
        shapes=parse_shapes(args.shapes) or DEFAULT_SHAPES,
        ratios=parse_csv(args.ratios, float) or DEFAULT_NNZ_RATIOS,
        sizes=parse_csv(args.sizes, int) or DEFAULT_SIZES,
        fill_value=args.fill_value,
        warmup=args.warmup,
        repeat=args.repeat,
    )

    fields = [
        "shape",
        "numel",
        "nnz_ratio",
        "size",
        "count_ms",
        "cumsum_ms",
        "write_ms",
        "fill_tail_ms",
        "fill_all_ms",
        "total_ms",
        "torch_ms",
    ]
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
