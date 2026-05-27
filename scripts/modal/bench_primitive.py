"""Run benchmarks on Modal H100.

Usage::

    modal run scripts/modal/bench_primitive.py --primitive kmeans
    modal run scripts/modal/bench_primitive.py --primitive knn --backend tilelang
    modal run scripts/modal/bench_primitive.py --primitive cov_gemm --sizes "4096,256"

Benchmarks compare TileLang vs Triton (vs CuteDSL when available)
backends for the specified primitive.
"""
from __future__ import annotations

import subprocess
import sys

import modal

from scripts.modal.setup_env import app, image


@app.function(
    image=image,
    gpu="H100",
    timeout=3600,
    concurrency_limit=1,
)
def run_bench(primitive: str, backend: str = "all", sizes: str = "",
              n_iters: int = 100, warmup: int = 10):
    """Execute a benchmark script inside the H100 container.

    Args:
        primitive: Name of the primitive (kmeans, knn, cov_gemm, etc.).
        backend: "tilelang", "triton", "cutedsl", or "all" (default).
        sizes: Comma-separated shape overrides (e.g. "4096,256").
        n_iters: Number of timed iterations.
        warmup: Number of warmup iterations.
    """
    script = f"benchmarks/tilelang/bench_{primitive}.py"
    cmd = [
        sys.executable, script,
        "--backend", backend,
        "--n-iters", str(n_iters),
        "--warmup", str(warmup),
    ]
    if sizes:
        cmd += ["--sizes", sizes]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


@app.local_entrypoint()
def main(primitive: str = "kmeans", backend: str = "all",
         sizes: str = "", n_iters: int = 100, warmup: int = 10):
    """Benchmark a primitive on Modal H100.

    Args:
        primitive: Primitive name (kmeans, knn, cov_gemm, etc.).
        backend: Backend to benchmark (tilelang, triton, cutedsl, all).
        sizes: Shape string (e.g. "4096,128,64" for N,D,K).
        n_iters: Timed iterations.
        warmup: Warmup iterations.
    """
    exit_code = run_bench.remote(
        primitive=primitive, backend=backend, sizes=sizes,
        n_iters=n_iters, warmup=warmup,
    )
    if exit_code != 0:
        print(f"Benchmark failed (exit code {exit_code})")
        raise SystemExit(exit_code)
    print("Benchmark complete.")
