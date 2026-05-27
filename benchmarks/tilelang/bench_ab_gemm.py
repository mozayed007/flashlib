"""Benchmark ab_gemm TileLang vs Triton.

Phase: 8
Usage:
    python benchmarks/tilelang/bench_ab_gemm.py --backend all
    modal run scripts/modal/bench_primitive.py --primitive ab_gemm
"""
from __future__ import annotations

import argparse
import sys


def benchmark_ab_gemm(backend: str, N: int = 4096, D1: int = 128, D2: int = 64,
                      n_iters: int = 100, warmup: int = 10) -> float:
    """Benchmark A^T @ B. Returns average time in ms."""
    import torch

    A = torch.randn(N, D1, device="cuda", dtype=torch.float32)
    B = torch.randn(N, D2, device="cuda", dtype=torch.float32)

    if backend == "tilelang":
        from flashlib.linalg.ab_gemm.tilelang.ab_gemm import tilelang_ab_gemm
        fn = lambda: tilelang_ab_gemm(A, B)
    elif backend == "triton":
        from flashlib.linalg.ab_gemm.triton import ab_gemm as triton_ab_gemm
        fn = lambda: triton_ab_gemm(A, B)
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
    parser = argparse.ArgumentParser(description="Benchmark ab_gemm (A^T @ B)")
    parser.add_argument("--backend", default="all", choices=["all", "tilelang", "triton"])
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sizes", default="")
    args = parser.parse_args()

    # TODO: Phase 8 — add more shape configurations
    shapes = [
        (4096, 128, 64),
        (4096, 256, 128),
        (8192, 128, 64),
        (8192, 256, 128),
    ]

    if args.sizes:
        parts = [int(x) for x in args.sizes.split(",")]
        if len(parts) == 3:
            shapes = [tuple(parts)]
        else:
            print(f"Error: --sizes must be N,D1,D2 (got {args.sizes})")
            sys.exit(1)

    backends = ["triton", "tilelang"] if args.backend == "all" else [args.backend]

    print(f"{'Shape (N,D1,D2)':<25} {'Backend':<10} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 60)

    for N, D1, D2 in shapes:
        times = {}
        for backend in backends:
            try:
                t = benchmark_ab_gemm(backend, N=N, D1=D1, D2=D2,
                                      n_iters=args.n_iters, warmup=args.warmup)
                times[backend] = t
            except Exception as e:
                print(f"  {backend}: FAILED ({e})")

        if "triton" in times and "tilelang" in times:
            speedup = times["triton"] / times["tilelang"]
            for backend in backends:
                s = f"{speedup:.2f}x" if backend == "tilelang" else ""
                print(f"{str((N, D1, D2)):<25} {backend:<10} {times[backend]:<12.3f} {s:<10}")
        else:
            for backend in backends:
                if backend in times:
                    print(f"{str((N, D1, D2)):<25} {backend:<10} {times[backend]:<12.3f}")


if __name__ == "__main__":
    main()
