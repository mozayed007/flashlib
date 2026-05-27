# Phase 2: KNN TileLang Kernel

## What You're Building

A TileLang kernel for brute-force KNN (top-K nearest neighbors). Multi-kernel design: distance computation + top-K selection.

**End result:** `flash_knn(x, c, k, backend="tilelang")` returns `(vals, idxs)` matching Triton output.

## Background Reading

| File | Lines | What to Study |
|------|-------|---------------|
| `flashlib/primitives/knn/triton/dispatch.py` | full | Triton KNN dispatcher |
| `flashlib/primitives/knn/impl.py` | full | KNN dispatcher with `_route()` |
| `flashlib/primitives/knn/__init__.py` | full | Lazy imports |
| `flashlib/primitives/knn/cutedsl/` | full | CuteDSL FA3 reference |
| `docs/phase-1-kmeans.md` | full | Patterns from Phase 1 (reuse here) |

## Key Concepts

### KNN Computation

For each query point `x[i]`, find the K nearest corpus points `c[j]` by L2 distance:

```
dist(i, j) = ||x[i] - c[j]||^2 = ||x[i]||^2 + ||c[j]||^2 - 2 * <x[i], c[j]>
```

### Multi-Kernel Approach

1. **Distance kernel:** Compute `cross = x @ c^T` using `T.gemm`, then compute distances
2. **Top-K kernel:** Iterative insert to find K smallest distances per query

### Sub-16 D Padding

KNN requires D >= 16 for `T.gemm`. The `_prepare_inputs` function in `impl.py` pads smaller D with zeros.

## Step-by-Step

### Step 1: Create directory

```bash
mkdir -p flashlib/primitives/knn/tilelang
```

### Step 2: Create `tilelang/__init__.py`

```python
"""KNN TileLang backend."""
from flashlib.primitives.knn.tilelang.knn import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_flash_knn,
)

__all__ = ["tilelang_flash_knn"]
```

### Step 3: Create `tilelang/knn.py`

Structure:
1. TileLang availability guard
2. Triton fallback import
3. TileLang distance kernel
4. TileLang top-K selection
5. Wrapper function

```python
"""TileLang KNN — brute-force top-K nearest neighbors."""
from __future__ import annotations
import torch

_TILELANG_AVAILABLE = False
_TILELANG_IMPORT_ERROR = None

def _try_init_tilelang() -> bool:
    global _TILELANG_AVAILABLE, _TILELANG_IMPORT_ERROR
    if _TILELANG_AVAILABLE:
        return True
    if _TILELANG_IMPORT_ERROR is not None:
        return False
    try:
        import tilelang
        _TILELANG_AVAILABLE = True
        return True
    except Exception as e:
        _TILELANG_IMPORT_ERROR = e
        return False

from flashlib.primitives.knn.triton.dispatch import flash_knn_triton


def tilelang_flash_knn(x, c, k, **kwargs):
    """TileLang KNN. Falls back to Triton if unavailable.

    Args:
        x: (B, N, D) query tensor.
        c: (B, M, D) corpus tensor.
        k: number of neighbors.

    Returns:
        (B, N, k) int64 indices of top-K nearest corpus points.
    """
    if not _try_init_tilelang():
        return flash_knn_triton(x, c, k, **kwargs)

    import tilelang
    from tilelang import T

    B, N, D = x.shape
    M = c.shape[1]
    BLOCK_N = min(64, N)
    BLOCK_M = min(64, M)

    # Compute squared norms
    x_sq = (x * x).sum(dim=-1).float()  # (B, N)
    c_sq = (c * c).sum(dim=-1).float()  # (B, M)

    x_f16 = x.half().contiguous()
    c_f16 = c.half().contiguous()

    # Distance kernel: compute full (N, M) distance matrix
    @tilelang.jit(out_idx=[-1])
    def _build_distance_kernel(N, M, D, BLOCK_N, BLOCK_M):
        @T.prim_func
        def kernel(
            X: T.Tensor[[N, D], "float16"],
            C: T.Tensor[[M, D], "float16"],
            X_sq: T.Tensor[[N], "float32"],
            C_sq: T.Tensor[[M], "float32"],
            Dist: T.Tensor[[N, M], "float32"],
        ):
            X_shared = T.alloc_shared([BLOCK_N, D], dtype="float16")
            C_shared = T.alloc_shared([BLOCK_M, D], dtype="float16")
            cross_frag = T.alloc_fragment([BLOCK_N, BLOCK_M], dtype="float32")

            pid_n = T.get_block_idx(0)
            pid_m = T.get_block_idx(1)

            n_start = pid_n * BLOCK_N
            m_start = pid_m * BLOCK_M

            T.clear(cross_frag)
            T.copy(X[n_start:n_start+BLOCK_N, :], X_shared)
            T.copy(C[m_start:m_start+BLOCK_M, :], C_shared)
            T.gemm(X_shared, C_shared, cross_frag, transpose_B=True)

            # Epilogue: dist = x_sq + c_sq - 2*cross
            for local_i in T.serial(BLOCK_N):
                for local_j in T.serial(BLOCK_M):
                    cross_frag[local_i, local_j] = (
                        X_sq[n_start + local_i]
                        + C_sq[m_start + local_j]
                        - 2.0 * cross_frag[local_i, local_j]
                    )

            T.copy(cross_frag, Dist[n_start:n_start+BLOCK_N, m_start:m_start+BLOCK_M])

        return kernel

    # Top-K selection (serial per query point — simple, correct)
    def _topk(dist, k):
        """Select top-K smallest distances and their indices."""
        return torch.topk(dist, k, dim=-1, largest=False)

    # Compute distance matrix
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    grid_m = (M + BLOCK_M - 1) // BLOCK_M

    kernel = _build_distance_kernel(N, M, D, BLOCK_N, BLOCK_M)
    dist = kernel(x_f16[0], c_f16[0], x_sq[0], c_sq[0])  # (N, M)

    # Top-K on distance matrix
    vals, idxs = _topk(dist, k)
    return idxs.unsqueeze(0)  # (1, N, k) — batch dim
```

### Step 4: Update `flashlib/primitives/knn/impl.py`

Add import and `"tilelang"` branch:

```python
from flashlib.primitives.knn.tilelang import tilelang_flash_knn

# In flash_knn_dispatch():
    if chosen == "tilelang":
        idxs = tilelang_flash_knn(x_p, c_p, k, **kwargs)
```

Also update `_route()` to accept "tilelang":

```python
def _route(*, B, N, D, k, backend=None, hw=None):
    if backend is not None:
        return backend  # "tilelang" passes through here
    ...
```

### Step 5: Update `flashlib/primitives/knn/__init__.py`

```python
tilelang_flash_knn = lazy_attr(
    "flashlib.primitives.knn.tilelang", "tilelang_flash_knn",
)

__all__ = [..., "tilelang_flash_knn"]
```

### Step 6: Update `flashlib/__init__.py`

```python
    "flash_knn_tilelang":          ("flashlib.primitives.knn", "tilelang_flash_knn"),
```

## Tests

```python
def test_topk_matches_triton():
    B, N, D, k = 1, 1024, 64, 5
    x = torch.randn(B, N, D, device="cuda", dtype=torch.float32)
    c = torch.randn(B, N, D, device="cuda", dtype=torch.float32)

    vals_tl, idxs_tl = flash_knn_dispatch(x, c, k, backend="tilelang")
    vals_tr, idxs_tr = flash_knn_dispatch(x, c, k, backend="triton")

    match_rate = (idxs_tl == idxs_tr).float().mean().item()
    assert match_rate > 0.9
```

## Validation on Modal

```bash
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_knn
```

## Troubleshooting

**D < 16 error:** The `_prepare_inputs` function in `impl.py` handles padding. Ensure you call it before passing to the TileLang kernel.

**Large distance matrix:** For large N x M, the full distance matrix may not fit in SMEM. Use the kernel-per-block approach (each block handles a tile).

**Top-K is slow:** The serial `_topk` is correct but slow. For production, write a TileLang top-K kernel with iterative insert (similar to the Triton approach).

## Checkpoint

- [ ] `tilelang_flash_knn` returns correct top-K indices
- [ ] Index match rate with Triton > 90%
- [ ] `flash_knn(x, c, k, backend="tilelang")` works end-to-end
- [ ] Parity test passes on Modal H100
