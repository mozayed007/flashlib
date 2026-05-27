"""Benchmark KMeans TileLang vs Triton.

Phase: 8
Usage:
    python benchmarks/tilelang/bench_kmeans.py --backend all
    python benchmarks/tilelang/bench_kmeans.py --backend tilelang --sizes "4096,128,64"
    modal run scripts/modal/bench_primitive.py --primitive kmeans
"""
from __future__ import annotations

import argparse
import sys
import time


def benchmark_assign(backend: str, N: int = 4096, D: int = 128, K: int = 64,
                     n_iters: int = 100, warmup: int = 10) -> float:
    """Benchmark KMeans assign kernel. Returns average time in ms."""
    import torch

    x = torch.randn(1, N, D, device="cuda", dtype=torch.float32)
    c = torch.randn(1, K, D, device="cuda", dtype=torch.float32)

    if backend == "tilelang":
        from flashlib.primitives.kmeans.tilelang.assign import tilelang_assign_euclid
        fn = lambda: tilelang_assign_euclid(x, c)
    elif backend == "triton":
        from flashlib.primitives.kmeans.triton.assign import euclid_assign_triton
        fn = lambda: euclid_assign_triton(x, c)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Warmup (includes JIT compilation for TileLang)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(n_iters):
        fn()
    end_event.record()
    torch.cuda.synchronize()

    total_ms = start_event.elapsed_time(end_event)
    return total_ms / n_iters


def main():
    parser = argparse.ArgumentParser(description="Benchmark KMeans assign")
    parser.add_argument("--backend", default="all", choices=["all", "tilelang", "triton"])
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sizes", default="",
                        help="Comma-separated N,D,K (e.g. '4096,128,64')")
    args = parser.parse_args()

    # TODO: Phase 8 — add more shape configurations
    shapes = [
        (4096, 128, 64),
        (4096, 256, 256),
        (8192, 128, 64),
        (8192, 256, 256),
        (16384, 128, 64),
    ]

    if args.sizes:
        parts = [int(x) for x in args.sizes.split(",")]
        if len(parts) == 3:
            shapes = [tuple(parts)]
        else:
            print(f"Error: --sizes must be N,D,K (got {args.sizes})")
            sys.exit(1)

    backends = ["triton", "tilelang"] if args.backend == "all" else [args.backend]

    print(f"{'Shape (N,D,K)':<25} {'Backend':<10} {'Time (ms)':<12} {'Speedup':<10}")
    print("-" * 60)

    for N, D, K in shapes:
        times = {}
        for backend in backends:
            try:
                t = benchmark_assign(
                    backend, N=N, D=D, K=K,
                    n_iters=args.n_iters, warmup=args.warmup,
                )
                times[backend] = t
            except Exception as e:
                print(f"  {backend}: FAILED ({e})")

        if "triton" in times and "tilelang" in times:
            speedup = times["triton"] / times["tilelang"]
            for backend in backends:
                s = f"{speedup:.2f}x" if backend == "tilelang" else ""
                print(f"{str((N, D, K)):<25} {backend:<10} {times[backend]:<12.3f} {s:<10}")
        else:
            for backend in backends:
                if backend in times:
                    print(f"{str((N, D, K)):<25} {backend:<10} {times[backend]:<12.3f}")


if __name__ == "__main__":
    main()
