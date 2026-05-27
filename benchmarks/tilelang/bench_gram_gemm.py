"""Benchmark gram_gemm TileLang vs Triton.

Phase: 8
Usage:
    python benchmarks/tilelang/bench_gram_gemm.py --backend all
    modal run scripts/modal/bench_primitive.py --primitive gram_gemm
"""
from __future__ import annotations

import argparse
import sys


def benchmark_gram_gemm(backend: str, N: int = 256, D: int = 4096,
                        n_iters: int = 100, warmup: int = 10) -> float:
    """Benchmark X @ X^T. Returns average time in ms."""
    import torch

    X = torch.randn(N, D, device="cuda", dtype=torch.float32)

    if backend == "tilelang":
        from flashlib.linalg.gram_gemm.tilelang.gram_gemm import tilelang_gram_gemm
        fn = lambda: tilelang_gram_gemm(X)
    elif backend == "triton":
        from flashlib.linalg.gram_gemm.triton import gram_gemm as triton_gram_gemm
        fn = lambda: triton_gram_gemm(X)
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
    parser = argparse.ArgumentParser(description="Benchmark gram_gemm (X @ X^T)")
    parser.add_argument("--backend", default="all", choices=["all", "tilelang", "triton"])
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sizes", default="")
    args = parser.parse_args()

    # TODO: Phase 8 — add more shape configurations
    shapes = [
        (64, 4096),
        (128, 4096),
        (256, 4096),
        (128, 8192),
        (256, 8192),
    ]

    if args.sizes:
        parts = [int(x) for x in args.sizes.split(",")]
        if len(parts) == 2:
            shapes = [tuple(parts)]
        else:
            print(f"Error: --sizes must be N,D (got {args.sizes})")
            sys.exit(1)

    backends = ["triton", "tilelang"] if args.backend == "all" else [args.backend]

    print(f"{'Shape (N,D)':<20} {'Backend':<10} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 55)

    for N, D in shapes:
        times = {}
        for backend in backends:
            try:
                t = benchmark_gram_gemm(backend, N=N, D=D,
                                        n_iters=args.n_iters, warmup=args.warmup)
                times[backend] = t
            except Exception as e:
                print(f"  {backend}: FAILED ({e})")

        if "triton" in times and "tilelang" in times:
            speedup = times["triton"] / times["tilelang"]
            for backend in backends:
                s = f"{speedup:.2f}x" if backend == "tilelang" else ""
                print(f"{str((N, D)):<20} {backend:<10} {times[backend]:<12.3f} {s:<10}")
        else:
            for backend in backends:
                if backend in times:
                    print(f"{str((N, D)):<20} {backend:<10} {times[backend]:<12.3f}")


if __name__ == "__main__":
    main()
