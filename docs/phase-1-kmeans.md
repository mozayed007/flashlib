# Phase 1: KMeans TileLang Kernel

## What You're Building

A TileLang kernel for KMeans Euclidean assign — the most important phase. This is the reference implementation that teaches you all the TileLang patterns.

**End result:** `flash_kmeans(x, K, backend="tilelang")` returns `(cluster_ids, centroids, n_iter)` matching the Triton output.

## Background Reading

| File | Lines | What to Study |
|------|-------|---------------|
| `flashlib/primitives/kmeans/triton/assign.py` | 823-924 | `_euclid_assign_kernel` — the Triton kernel to port |
| `flashlib/primitives/kmeans/triton/assign.py` | 939-1034 | Split-D variant (port later for large D) |
| `flashlib/primitives/kmeans/triton/update.py` | full | `triton_lloyd_centroid_step_euclid` — reuse in TileLang path |
| `flashlib/primitives/kmeans/cutedsl/assign.py` | 140-170 | `_try_init_cutedsl()` pattern |
| `flashlib/primitives/kmeans/cutedsl/assign.py` | 454-607 | Kernel wrapper with fallback |
| `flashlib/primitives/kmeans/cutedsl/assign.py` | 614-669 | Lloyd loop — must match this interface |
| `flashlib/primitives/kmeans/impl.py` | full | Dispatcher with `_route()` |
| `flashlib/primitives/kmeans/__init__.py` | full | Lazy imports pattern |
| `docs/tilelang-api.md` | full | TileLang API reference |

## Key Concepts

### The Euclidean Assign Problem

For each point `x[i]`, find the nearest centroid `c[j]` by squared Euclidean distance:

```
dist(i, j) = ||x[i] - c[j]||^2 = ||x[i]||^2 + ||c[j]||^2 - 2 * <x[i], c[j]>
```

Since `||x[i]||^2` is the same for all centroids, we only need:
```
argmin_j ( ||c[j]||^2 - 2 * <x[i], c[j]> )
```

### Tiling Strategy

- Each block handles BLOCK_N points for one batch element
- Iterate over K centroids in chunks of BLOCK_K
- For each chunk:
  1. Load X tile (BLOCK_N, D) to shared memory
  2. Load C tile (BLOCK_K, D) to shared memory
  3. Compute cross term: `T.gemm(X_shared, C_shared, cross_frag, transpose_B=True)`
  4. Compute distance: `dist = c_sq - 2 * cross`
  5. Update running argmin

### The `T.gemm` Accumulation

`T.gemm` accumulates (adds to the result). For KMeans, this is fine because we compute one GEMM per K-chunk. But you must `T.clear(cross_frag)` if reusing the fragment.

### Lloyd Loop Interface

The full KMeans function must return `(cluster_ids, centroids, n_iter)` matching:
```python
cluster_ids, centroids, n_iter = batch_kmeans_Euclid_torch_native(
    x, n_clusters, max_iters=max_iters, tol=tol,
    init_centroids=init_centroids, verbose=verbose,
)
```

The TileLang path reuses `triton_lloyd_centroid_step_euclid` from `flashlib/primitives/kmeans/triton/update.py` for the centroid update step (only the assign step is ported to TileLang).

## Step-by-Step

### Step 1: Create directory

```bash
mkdir -p flashlib/primitives/kmeans/tilelang
```

### Step 2: Create `tilelang/__init__.py`

```python
"""KMeans TileLang backend."""
from flashlib.primitives.kmeans.tilelang.assign import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_assign_euclid,
    tilelang_kmeans_Euclid,
)

__all__ = [
    "tilelang_assign_euclid",
    "tilelang_kmeans_Euclid",
]
```

### Step 3: Create `tilelang/assign.py`

Structure:

1. TileLang availability guard (`_try_init_tilelang`)
2. Triton fallback imports
3. TileLang assign kernel (`tilelang_assign_euclid`)
4. TileLang Lloyd loop (`tilelang_kmeans_Euclid`)

The assign kernel:

```python
import tilelang
from tilelang import T

@tilelang.jit(out_idx=[-1])
def _build_assign_kernel(N, D, K, BLOCK_N, BLOCK_K):
    @T.prim_func
    def kernel(
        X: T.Tensor[[N, D], "float16"],
        C: T.Tensor[[K, D], "float16"],
        C_sq: T.Tensor[[K], "float32"],
        Out: T.Tensor[[N], "int32"],
    ):
        # Allocate shared memory
        X_shared = T.alloc_shared([BLOCK_N, D], dtype="float16")
        C_shared = T.alloc_shared([BLOCK_K, D], dtype="float16")
        cross_frag = T.alloc_fragment([BLOCK_N, BLOCK_K], dtype="float32")
        best_dist = T.alloc_fragment([BLOCK_N], dtype="float32")
        best_idx = T.alloc_fragment([BLOCK_N], dtype="int32")

        # Get block index
        pid_n = T.get_block_idx(0)

        # Initialize
        T.fill(best_dist, T.infinity("float32"))
        T.clear(best_idx)

        # Load X tile
        n_start = pid_n * BLOCK_N
        T.copy(X[n_start:n_start+BLOCK_N, :], X_shared)

        # Iterate over centroids
        for k_start in T.serial(0, K, BLOCK_K):
            # Load C tile
            T.copy(C[k_start:k_start+BLOCK_K, :], C_shared)

            # Clear and compute cross term
            T.clear(cross_frag)
            T.gemm(X_shared, C_shared, cross_frag, transpose_B=True)

            # Epilogue: compute distances and update argmin
            for local_i in T.serial(BLOCK_N):
                for k in T.serial(BLOCK_K):
                    dist = C_sq[k_start + k] - 2.0 * cross_frag[local_i, k]
                    if dist < best_dist[local_i]:
                        best_dist[local_i] = dist
                        best_idx[local_i] = k_start + k

        # Write output
        T.copy(best_idx, Out[n_start:n_start+BLOCK_N])

    return kernel
```

The wrapper:

```python
def tilelang_assign_euclid(x, centroids):
    """TileLang euclidean assign. Falls back to Triton if unavailable."""
    if not _try_init_tilelang():
        return euclid_assign_triton(x, centroids)

    B, N, D = x.shape
    K = centroids.shape[1]
    BLOCK_N = min(64, N)
    BLOCK_K = min(64, K)

    # Precompute centroid squared norms
    c_sq = (centroids * centroids).sum(dim=-1).float()

    # Cast to float16 for TileLang
    x_f16 = x.half().contiguous()
    c_f16 = centroids.half().contiguous()

    kernel = _build_assign_kernel(N, D, K, BLOCK_N, BLOCK_K)
    out = kernel(x_f16, c_f16, c_sq)
    return out
```

The Lloyd loop:

```python
def tilelang_kmeans_Euclid(x, n_clusters, *, max_iters=100, tol=0.0,
                           init_centroids=None, verbose=False, **kwargs):
    """Full KMeans with TileLang assign + Triton centroid update."""
    from flashlib.primitives.kmeans.triton.update import (
        triton_lloyd_centroid_step_euclid,
        triton_centroid_finalize,
    )

    B, N, D = x.shape
    K = n_clusters

    # Initialize centroids (random or provided)
    if init_centroids is not None:
        centroids = init_centroids.clone()
    else:
        idx = torch.randperm(N, device=x.device)[:K]
        centroids = x[0, idx].unsqueeze(0).expand(B, -1, -1).clone()

    for it in range(max_iters):
        # Assign using TileLang
        cluster_ids = tilelang_assign_euclid(x, centroids)

        # Update centroids using Triton (reuse existing)
        new_centroids, converged = triton_lloyd_centroid_step_euclid(
            x, cluster_ids, K,
        )
        centroids = triton_centroid_finalize(
            new_centroids, cluster_ids, K,
        )

        if converged and tol > 0:
            break

    return cluster_ids, centroids, it + 1
```

### Step 4: Update `flashlib/primitives/kmeans/impl.py`

Add import and `"tilelang"` branch:

```python
from flashlib.primitives.kmeans.tilelang import tilelang_kmeans_Euclid

# In flash_kmeans():
    elif chosen == "tilelang":
        if metric != "euclidean":
            raise NotImplementedError(
                f"tilelang backend currently supports euclidean only "
                f"(got {metric!r}); fall back to backend='triton' for cosine/dot."
            )
        cluster_ids, centroids, n_iter = tilelang_kmeans_Euclid(
            x_b, n_clusters, max_iters=max_iters, tol=tol,
            init_centroids=init_centroids, verbose=verbose, **kwargs,
        )
```

### Step 5: Update `flashlib/primitives/kmeans/__init__.py`

Add lazy imports:

```python
from flashlib._lazy import lazy_attr

tilelang_assign_euclid = lazy_attr(
    "flashlib.primitives.kmeans.tilelang", "tilelang_assign_euclid",
)
tilelang_kmeans_Euclid = lazy_attr(
    "flashlib.primitives.kmeans.tilelang", "tilelang_kmeans_Euclid",
)

# Add to __all__:
    "tilelang_assign_euclid",
    "tilelang_kmeans_Euclid",
```

### Step 6: Update `flashlib/__init__.py`

Add to `_LAZY_ATTRS`:

```python
    "flash_kmeans_tilelang":     ("flashlib.primitives.kmeans", "tilelang_kmeans_Euclid"),
```

### Step 7: Run parity test

```bash
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_kmeans
```

## Tests

### Unit test: assign correctness

```python
def test_assign_matches_torch_reference():
    B, N, D, K = 1, 4096, 128, 64
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    c = torch.randn(B, K, D, device="cuda", dtype=torch.float32)

    tl_ids = tilelang_assign_euclid(x, c)
    ref_ids = _torch_kmeans_assign(x, c)

    mismatch_rate = (tl_ids != ref_ids).float().mean().item()
    assert mismatch_rate < 5e-3
```

### Integration test: full Lloyd loop

```python
def test_full_lloyd_loop():
    B, N, D, K = 1, 2048, 64, 16
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float32)

    cluster_ids, centroids, n_iter = flash_kmeans(
        x, K, max_iters=10, backend="tilelang",
    )

    assert cluster_ids.shape == (B, N)
    assert centroids.shape == (B, K, D)
    assert 1 <= n_iter <= 10
    assert cluster_ids.min() >= 0 and cluster_ids.max() < K
```

## Validation on Modal

```bash
# Test assign kernel
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_assign_matches

# Test full loop
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_full_lloyd_loop

# All kmeans tests
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_kmeans
```

## Troubleshooting

**SMEM overflow:** Reduce BLOCK_N and BLOCK_K. Start with 64, 64.

**dtype error:** Ensure x and centroids are cast to float16 before passing to TileLang kernel.

**Wrong assignments:** Check that `T.clear(cross_frag)` is called before `T.gemm`, and that argmin uses `k_start + k` (global index).

**Slow compilation:** First call takes 30-120s. This is normal. Subsequent calls are instant.

**Triton fallback not working:** Check `_try_init_tilelang()` returns `False` when tilelang is not installed.

## Checkpoint

- [ ] `tilelang_assign_euclid` returns correct assignments (mismatch < 0.5%)
- [ ] `tilelang_kmeans_Euclid` runs full Lloyd loop without error
- [ ] `flash_kmeans(x, K, backend="tilelang")` works end-to-end
- [ ] Parity test passes on Modal H100
- [ ] Triton fallback works when tilelang is not installed
