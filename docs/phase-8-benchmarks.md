# Phase 8: Benchmarks

## What You're Building

Benchmark scripts comparing TileLang vs Triton (vs CuteDSL) performance. Update routing heuristics based on measured speedups.

## Background Reading

| File | What to Study |
|------|---------------|
| `benchmarks/tilelang/` | Benchmark directory (created in Phase 0) |
| `benchmarks/bench_knn_components.py` | Reference benchmark script |
| `scripts/modal/bench_primitive.py` | Modal benchmark runner |
| `docs/modal-workflow.md` | Modal usage |

## Benchmark Scripts

Create one script per real TileLang kernel:

| Script | Primitive | What to Measure |
|--------|-----------|-----------------|
| `bench_kmeans.py` | KMeans assign | Assign kernel time (ms) vs Triton |
| `bench_knn.py` | KNN | Top-K time vs Triton |
| `bench_cov_gemm.py` | cov_gemm | X^T @ X time vs Triton |
| `bench_gram_gemm.py` | gram_gemm | X @ X^T time vs Triton |
| `bench_ab_gemm.py` | ab_gemm | A^T @ B time vs Triton |
| `bench_standard_scaler.py` | StandardScaler | Fit+transform time vs Triton |

Skip benchmarks for stubs (hdbscan, umap, tsne, random_forest) — they delegate to Triton.

## Step-by-Step

### Step 1: Create `bench_kmeans.py`

```python
"""Benchmark KMeans TileLang vs Triton."""
import torch
import time
import argparse

def benchmark_assign(backend, N=4096, D=128, K=64, n_iters=100, warmup=10):
    x = torch.randn(1, N, D, device="cuda", dtype=torch.float32)
    c = torch.randn(1, K, D, device="cuda", dtype=torch.float32)

    if backend == "tilelang":
        from flashlib.primitives.kmeans.tilelang.assign import tilelang_assign_euclid
        fn = lambda: tilelang_assign_euclid(x, c)
    else:
        from flashlib.primitives.kmeans.triton.assign import euclid_assign_triton
        fn = lambda: euclid_assign_triton(x, c)

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    total_ms = start.elapsed_time(end)
    avg_ms = total_ms / n_iters
    return avg_ms

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="all")
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sizes", default="")
    args = parser.parse_args()

    shapes = [
        (4096, 128, 64),
        (4096, 256, 256),
        (8192, 128, 64),
        (8192, 256, 256),
    ]

    if args.sizes:
        parts = [int(x) for x in args.sizes.split(",")]
        shapes = [tuple(parts)]

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
                continue

        if "triton" in times and "tilelang" in times:
            speedup = times["triton"] / times["tilelang"]
            for backend in backends:
                s = f"{speedup:.2f}x" if backend == "tilelang" else ""
                print(f"{str((N,D,K)):<25} {backend:<10} {times[backend]:<12.3f} {s:<10}")
        else:
            for backend in backends:
                if backend in times:
                    print(f"{str((N,D,K)):<25} {backend:<10} {times[backend]:<12.3f}")

if __name__ == "__main__":
    main()
```

### Step 2: Create other benchmark scripts

Follow the same pattern for each primitive.

### Step 3: Run benchmarks on Modal

```bash
modal run scripts/modal/bench_primitive.py --primitive kmeans
modal run scripts/modal/bench_primitive.py --primitive knn
modal run scripts/modal/bench_primitive.py --primitive cov_gemm
```

### Step 4: Analyze results

Compare TileLang vs Triton speedup across shapes:

| Shape | Triton (ms) | TileLang (ms) | Speedup |
|-------|-------------|---------------|---------|
| (4096, 128, 64) | ? | ? | ? |
| (4096, 256, 256) | ? | ? | ? |
| (8192, 128, 64) | ? | ? | ? |
| (8192, 256, 256) | ? | ? | ? |

### Step 5: Update routing heuristics

If TileLang is faster for certain shapes, update the `_route()` functions in `impl.py`:

```python
def _route(*, B, N, D, K, backend=None, variant=None, hw=None):
    if backend is not None:
        return backend
    hw = hw or _hw.current()
    if not hw.is_cuda:
        return "torch"
    # Auto-route to TileLang for certain shapes
    if D <= 256 and K >= 64:
        return "tilelang", variant
    return "triton", variant
```

**Warning:** Only auto-route if TileLang compilation is cached (first-call compile is too slow for auto-routing).

## Validation

```bash
modal run scripts/modal/bench_primitive.py --primitive kmeans
modal run scripts/modal/bench_primitive.py --primitive knn
modal run scripts/modal/bench_primitive.py --primitive cov_gemm
```

## Budget Impact

Benchmarks consume ~6.4 GPU hours ($25.28). Run efficiently:
- Use `--n-iters 50` for quick checks, `--n-iters 200` for final numbers
- Run all shapes in one invocation (not separate calls)
- Share warmup across shapes when possible

## Checkpoint

- [ ] Benchmark scripts exist for all 6 real TileLang kernels
- [ ] Benchmarks run on Modal H100 without error
- [ ] Speedup numbers collected for representative shapes
- [ ] Routing heuristics updated (if TileLang wins for certain shapes)
- [ ] Final parity tests still pass after routing changes
