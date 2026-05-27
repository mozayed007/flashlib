"""Benchmark KNN TileLang vs Triton.

Phase: 8
Usage:
    python benchmarks/tilelang/bench_knn.py --backend all
    modal run scripts/modal/bench_primitive.py --primitive knn
"""
from __future__ import annotations

import argparse
import sys


def benchmark_knn(backend: str, N: int = 1024, M: int = 1024, D: int = 64,
                  k: int = 5, n_iters: int = 100, warmup: int = 10) -> float:
    """Benchmark KNN top-K. Returns average time in ms."""
    import torch

    x = torch.randn(1, N, D, device="cuda", dtype=torch.float32)
    c = torch.randn(1, M, D, device="cuda", dtype=torch.float32)

    if backend == "tilelang":
        from flashlib.primitives.knn.tilelang.knn import tilelang_flash_knn
        fn = lambda: tilelang_flash_knn(x, c, k)
    elif backend == "triton":
        from flashlib.primitives.knn.triton.dispatch import flash_knn_triton
        fn = lambda: flash_knn_triton(x, c, k)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(n_iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    return start_event.elapsed_time(end_event) / n_iters


def main():
    parser = argparse.ArgumentParser(description="Benchmark KNN")
    parser.add_argument("--backend", default="all", choices=["all", "tilelang", "triton"])
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sizes", default="")
    args = parser.parse_args()

    # TODO: Phase 8 — add more shape configurations
    shapes = [
        (1024, 1024, 64, 5),
        (4096, 4096, 64, 5),
        (4096, 4096, 128, 10),
        (1024, 1024, 64, 32),
    ]

    if args.sizes:
        parts = [int(x) for x in args.sizes.split(",")]
        if len(parts) == 4:
            shapes = [tuple(parts)]
        else:
            print(f"Error: --sizes must be N,M,D,k (got {args.sizes})")
            sys.exit(1)

    backends = ["triton", "tilelang"] if args.backend == "all" else [args.backend]

    print(f"{'Shape (N,M,D,k)':<25} {'Backend':<10} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 60)

    for N, M, D, k in shapes:
        times = {}
        for backend in backends:
            try:
                t = benchmark_knn(backend, N=N, M=M, D=D, k=k,
                                  n_iters=args.n_iters, warmup=args.warmup)
                times[backend] = t
            except Exception as e:
                print(f"  {backend}: FAILED ({e})")

        if "triton" in times and "tilelang" in times:
            speedup = times["triton"] / times["tilelang"]
            for backend in backends:
                s = f"{speedup:.2f}x" if backend == "tilelang" else ""
                print(f"{str((N, M, D, k)):<25} {backend:<10} {times[backend]:<12.3f} {s:<10}")
        else:
            for backend in backends:
                if backend in times:
                    print(f"{str((N, M, D, k)):<25} {backend:<10} {times[backend]:<12.3f}")


if __name__ == "__main__":
    main()
