import argparse
import csv
import sys
import time
from pathlib import Path

import pytest
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

try:
    from benchmark import conftest as bench_cfg
except Exception:
    bench_cfg = None


DEFAULT_DTYPES = [torch.float32]
FULL_DTYPES = [torch.bool, torch.int32, torch.float16, torch.float32]
DEFAULT_SHAPES = [
    (1024,),
    (16384,),
    (262144,),
    (1048576,),
    (32, 1024),
    (128, 4096),
    (1024, 4096),
    (16, 64, 64),
    (32, 128, 128),
]
DEFAULT_NNZ_RATIOS = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]
DEFAULT_SIZES = [16, 128, 1024, 4096]
DEFAULT_FILL_VALUE = -1

BREAKDOWN_SHAPES = [(1048576,), (1024, 4096)]
BREAKDOWN_NNZ_RATIOS = [0.001, 0.1, 1.0]
BREAKDOWN_SIZES = [128, 1024, 4096]

DTYPE_BY_NAME = {
    "bool": torch.bool,
    "int32": torch.int32,
    "float16": torch.float16,
    "float32": torch.float32,
}


def dtype_name(dtype):
    return str(dtype).replace("torch.", "")


def shape_name(shape):
    return "x".join(str(dim) for dim in shape)


def parse_csv(value, cast):
    if value is None:
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_dtypes(value):
    if value == "full":
        return FULL_DTYPES
    names = parse_csv(value, str)
    if names is None:
        return None
    dtypes = []
    for name in names:
        if name not in DTYPE_BY_NAME:
            raise ValueError(f"Unsupported dtype '{name}'")
        dtypes.append(DTYPE_BY_NAME[name])
    return dtypes


def parse_shapes(value):
    shape_specs = parse_csv(value, str)
    if shape_specs is None:
        return None

    shapes = []
    for spec in shape_specs:
        dims = tuple(int(dim) for dim in spec.split("x") if dim)
        if not dims:
            raise ValueError(f"Invalid shape '{spec}'")
        shapes.append(dims)
    return shapes


def make_input(shape, dtype, nnz_ratio, device):
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


def baseline_nonzero_static_fallback(x, size: int, fill_value: int = -1):
    ndim = x.dim()
    out = torch.empty((size, ndim), device=x.device, dtype=torch.long)

    if size == 0 or ndim == 0:
        return out

    nz = torch.nonzero(x, as_tuple=False)
    copy_len = min(size, nz.shape[0])

    if copy_len > 0:
        out[:copy_len].copy_(nz[:copy_len])

    if copy_len < size:
        out[copy_len:].fill_(fill_value)

    return out


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


def print_rows(rows, fields, output_format):
    if output_format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return

    widths = {}
    for field in fields:
        widths[field] = max(len(field), *(len(str(row[field])) for row in rows))

    header = " | ".join(field.ljust(widths[field]) for field in fields)
    sep = "-+-".join("-" * widths[field] for field in fields)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(str(row[field]).ljust(widths[field]) for field in fields))


def measure_torch_baseline(x, size, fill_value, warmup, repeat):
    try:
        torch_ms = bench(
            lambda: torch.nonzero_static(x, size=size, fill_value=fill_value),
            warmup,
            repeat,
        )
        return torch_ms, "torch.nonzero_static"
    except Exception:
        try:
            torch_ms = bench(
                lambda: baseline_nonzero_static_fallback(
                    x, size=size, fill_value=fill_value
                ),
                warmup,
                repeat,
            )
            return torch_ms, "torch.nonzero + truncate + fill"
        except Exception:
            return None, "baseline unavailable"


def benchmark_rows(
    shapes,
    dtypes,
    nnz_ratios,
    sizes,
    fill_value,
    warmup,
    repeat,
    max_cases=None,
):
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return []

    rows = []
    case_count = 0
    device = torch.device("cuda")
    for dtype in dtypes:
        for shape in shapes:
            for nnz_ratio in nnz_ratios:
                torch.manual_seed(0)
                x = make_input(shape, dtype, nnz_ratio, device)
                for size in sizes:
                    flaggems_ms = bench(
                        lambda: nonzero_static(
                            x, size=size, fill_value=fill_value
                        ),
                        warmup,
                        repeat,
                    )
                    torch_ms, baseline_status = measure_torch_baseline(
                        x, size, fill_value, warmup, repeat
                    )
                    speedup = None if torch_ms is None else torch_ms / flaggems_ms
                    rows.append(
                        {
                            "shape": shape_name(shape),
                            "dtype": dtype_name(dtype),
                            "numel": x.numel(),
                            "nnz_ratio": nnz_ratio,
                            "size": size,
                            "fill_value": fill_value,
                            "flaggems_ms": f"{flaggems_ms:.6f}",
                            "torch_ms": (
                                "None" if torch_ms is None else f"{torch_ms:.6f}"
                            ),
                            "speedup": (
                                "None" if speedup is None else f"{speedup:.6f}"
                            ),
                            "baseline_status": baseline_status,
                        }
                    )

                    case_count += 1
                    if max_cases is not None and case_count >= max_cases:
                        return rows
    return rows


def _prepare_workspace(x, size, block_size):
    ndim = x.dim()
    out = torch.empty((size, ndim), device=x.device, dtype=torch.int64)

    if size == 0 or ndim == 0:
        return out, x, None

    x = x.contiguous()
    numel = x.numel()

    if numel == 0:
        return out, x, None

    num_blocks = triton.cdiv(numel, block_size)
    counts = torch.empty((num_blocks,), device=x.device, dtype=torch.int64)
    return out, x, counts


def _run_count(x, counts, block_size):
    numel = x.numel()
    num_blocks = triton.cdiv(numel, block_size)
    with torch_device_fn.device(x.device):
        _nonzero_static_count_kernel[(num_blocks,)](
            x,
            counts,
            numel,
            BLOCK_SIZE=block_size,
        )


def _run_cumsum(counts):
    return torch.cumsum(counts, dim=0) - counts


def _run_write(x, prefix, out, size, block_size):
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


def _run_fill_tail(out, prefix, counts, size, fill_value, block_size):
    ndim = out.shape[1]
    total_out = size * ndim
    num_blocks = counts.numel()
    fill_grid = (triton.cdiv(total_out, block_size),)
    with torch_device_fn.device(out.device):
        _nonzero_static_fill_tail_kernel[fill_grid](
            out,
            prefix,
            counts,
            num_blocks,
            size,
            ndim,
            fill_value,
            BLOCK_SIZE=block_size,
        )


def _run_fill_all(out, size, fill_value, block_size):
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


def breakdown_rows(
    shapes,
    dtypes,
    nnz_ratios,
    sizes,
    fill_value,
    warmup,
    repeat,
    max_cases=None,
    block_size=DEFAULT_BLOCK_SIZE,
):
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return []

    rows = []
    case_count = 0
    device = torch.device("cuda")
    for dtype in dtypes:
        for shape in shapes:
            for nnz_ratio in nnz_ratios:
                torch.manual_seed(0)
                x = make_input(shape, dtype, nnz_ratio, device)
                for size in sizes:
                    out, x_contig, counts = _prepare_workspace(x, size, block_size)

                    if size == 0 or x.dim() == 0:
                        count_ms = cumsum_ms = fill_ms = write_ms = 0.0
                    elif x_contig.numel() == 0:
                        count_ms = cumsum_ms = write_ms = 0.0
                        fill_ms = bench(
                            lambda: _run_fill_all(
                                out, size, fill_value, block_size
                            ),
                            warmup,
                            repeat,
                        )
                    else:
                        _run_count(x_contig, counts, block_size)
                        sync()
                        prefix = _run_cumsum(counts)
                        sync()

                        count_ms = bench(
                            lambda: _run_count(x_contig, counts, block_size),
                            warmup,
                            repeat,
                        )
                        cumsum_ms = bench(
                            lambda: _run_cumsum(counts),
                            warmup,
                            repeat,
                        )
                        write_ms = bench(
                            lambda: _run_write(
                                x_contig, prefix, out, size, block_size
                            ),
                            warmup,
                            repeat,
                        )
                        fill_ms = bench(
                            lambda: _run_fill_tail(
                                out, prefix, counts, size, fill_value, block_size
                            ),
                            warmup,
                            repeat,
                        )

                    total_ms = bench(
                        lambda: nonzero_static(
                            x, size=size, fill_value=fill_value
                        ),
                        warmup,
                        repeat,
                    )
                    torch_ms, baseline_status = measure_torch_baseline(
                        x, size, fill_value, warmup, repeat
                    )

                    rows.append(
                        {
                            "shape": shape_name(shape),
                            "dtype": dtype_name(dtype),
                            "numel": x.numel(),
                            "nnz_ratio": nnz_ratio,
                            "size": size,
                            "count_ms": f"{count_ms:.6f}",
                            "cumsum_ms": f"{cumsum_ms:.6f}",
                            "fill_ms": f"{fill_ms:.6f}",
                            "write_ms": f"{write_ms:.6f}",
                            "total_ms": f"{total_ms:.6f}",
                            "torch_ms": (
                                "None" if torch_ms is None else f"{torch_ms:.6f}"
                            ),
                            "baseline_status": baseline_status,
                        }
                    )

                    case_count += 1
                    if max_cases is not None and case_count >= max_cases:
                        return rows
    return rows


@pytest.mark.nonzero_static
def test_perf_nonzero_static():
    if not torch.cuda.is_available():
        pytest.skip("nonzero_static benchmark requires CUDA")

    warmup = 20
    repeat = 100
    if bench_cfg is not None and bench_cfg.Config is not None:
        warmup = int(bench_cfg.Config.warm_up)
        repeat = int(bench_cfg.Config.repetition)

    rows = benchmark_rows(
        shapes=DEFAULT_SHAPES,
        dtypes=DEFAULT_DTYPES,
        nnz_ratios=DEFAULT_NNZ_RATIOS,
        sizes=DEFAULT_SIZES,
        fill_value=DEFAULT_FILL_VALUE,
        warmup=warmup,
        repeat=repeat,
    )
    print_rows(rows, BENCH_FIELDS, output_format="table")


BENCH_FIELDS = [
    "shape",
    "dtype",
    "numel",
    "nnz_ratio",
    "size",
    "fill_value",
    "flaggems_ms",
    "torch_ms",
    "speedup",
    "baseline_status",
]

BREAKDOWN_FIELDS = [
    "shape",
    "dtype",
    "numel",
    "nnz_ratio",
    "size",
    "count_ms",
    "cumsum_ms",
    "fill_ms",
    "write_ms",
    "total_ms",
    "torch_ms",
    "baseline_status",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--breakdown", action="store_true")
    parser.add_argument("--format", choices=["table", "csv"], default="table")
    parser.add_argument(
        "--dtypes", default=None, help="Example: bool,int32,float32 or full"
    )
    parser.add_argument("--shapes", default=None, help="Example: 1024,32x1024")
    parser.add_argument("--ratios", default=None, help="Example: 0,0.001,0.1")
    parser.add_argument("--sizes", default=None, help="Example: 16,128,1024")
    parser.add_argument("--fill-value", type=int, default=DEFAULT_FILL_VALUE)
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    if args.breakdown:
        shapes = parse_shapes(args.shapes) or BREAKDOWN_SHAPES
        ratios = parse_csv(args.ratios, float) or BREAKDOWN_NNZ_RATIOS
        sizes = parse_csv(args.sizes, int) or BREAKDOWN_SIZES
    else:
        shapes = parse_shapes(args.shapes) or DEFAULT_SHAPES
        ratios = parse_csv(args.ratios, float) or DEFAULT_NNZ_RATIOS
        sizes = parse_csv(args.sizes, int) or DEFAULT_SIZES

    dtypes = parse_dtypes(args.dtypes) or DEFAULT_DTYPES

    if args.breakdown:
        rows = breakdown_rows(
            shapes=shapes,
            dtypes=dtypes,
            nnz_ratios=ratios,
            sizes=sizes,
            fill_value=args.fill_value,
            warmup=args.warmup,
            repeat=args.repeat,
            max_cases=args.max_cases,
        )
        print_rows(rows, BREAKDOWN_FIELDS, output_format=args.format)
    else:
        rows = benchmark_rows(
            shapes=shapes,
            dtypes=dtypes,
            nnz_ratios=ratios,
            sizes=sizes,
            fill_value=args.fill_value,
            warmup=args.warmup,
            repeat=args.repeat,
            max_cases=args.max_cases,
        )
        print_rows(rows, BENCH_FIELDS, output_format=args.format)


if __name__ == "__main__":
    main()
