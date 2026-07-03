import argparse
import csv
import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from flag_gems.ops.nonzero_static import nonzero_static  # noqa: E402

try:
    from benchmark import conftest as bench_cfg
except Exception:
    bench_cfg = None


BENCH_DTYPES = [torch.float32]
BENCH_SHAPES = [
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
NNZ_RATIOS = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0]
SIZES = [16, 128, 1024, 4096]
FILL_VALUE = -1


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


def run_benchmark(warmup=20, repeat=100):
    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
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
        ],
    )
    writer.writeheader()

    device = torch.device("cuda")
    for dtype in BENCH_DTYPES:
        for shape in BENCH_SHAPES:
            for nnz_ratio in NNZ_RATIOS:
                torch.manual_seed(0)
                x = make_input(shape, dtype, nnz_ratio, device)
                for size in SIZES:
                    flaggems_ms = bench(
                        lambda: nonzero_static(
                            x, size=size, fill_value=FILL_VALUE
                        ),
                        warmup,
                        repeat,
                    )

                    try:
                        torch_ms = bench(
                            lambda: torch.nonzero_static(
                                x, size=size, fill_value=FILL_VALUE
                            ),
                            warmup,
                            repeat,
                        )
                        baseline_status = "torch.nonzero_static"
                    except Exception:
                        try:
                            torch_ms = bench(
                                lambda: baseline_nonzero_static_fallback(
                                    x, size=size, fill_value=FILL_VALUE
                                ),
                                warmup,
                                repeat,
                            )
                            baseline_status = "torch.nonzero + truncate + fill"
                        except Exception:
                            torch_ms = None
                            baseline_status = "baseline unavailable"

                    speedup = None if torch_ms is None else torch_ms / flaggems_ms
                    writer.writerow(
                        {
                            "shape": "x".join(str(dim) for dim in shape),
                            "dtype": str(dtype),
                            "numel": x.numel(),
                            "nnz_ratio": nnz_ratio,
                            "size": size,
                            "fill_value": FILL_VALUE,
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


@pytest.mark.nonzero_static
def test_perf_nonzero_static():
    if not torch.cuda.is_available():
        pytest.skip("nonzero_static benchmark requires CUDA")

    warmup = 20
    repeat = 100
    if bench_cfg is not None and bench_cfg.Config is not None:
        warmup = int(bench_cfg.Config.warm_up)
        repeat = int(bench_cfg.Config.repetition)

    run_benchmark(warmup=warmup, repeat=repeat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()

    run_benchmark(warmup=args.warmup, repeat=args.repeat)


if __name__ == "__main__":
    main()
