# Phase 4: GEMM Variants

## What You're Building

TileLang backends for the three linalg GEMM primitives:
- `cov_gemm(X)` → `X^T @ X` (started in Phase 3, complete here)
- `gram_gemm(X)` → `X @ X^T` (dual of cov_gemm)
- `ab_gemm(A, B)` → `A^T @ B` (generalized)

## Background Reading

| File | What to Study |
|------|---------------|
| `flashlib/linalg/cov_gemm/triton/cov_gemm.py` | 123-line Triton kernel |
| `flashlib/linalg/gram_gemm/triton/` | Gram GEMM kernel |
| `flashlib/linalg/ab_gemm/triton/` | AB GEMM kernel |
| `docs/phase-3-pca.md` | cov_gemm TileLang kernel (start from there) |

## Key Concepts

### Symmetric Optimization

`X^T @ X` is symmetric — only compute the upper triangle, then mirror:
```python
result = torch.triu(result) + torch.triu(result, diagonal=1).T
```

### gram_gemm: X @ X^T

For D >> N (wide matrices), compute `X @ X^T` instead:
- Output is (N, N) — may be large
- Same tiling strategy as cov_gemm but transposed

### ab_gemm: A^T @ B

Generalized form — `A` and `B` share the first dimension N:
```
Out[i, j] = sum_k A[k, i] * B[k, j]
```

## Step-by-Step

### Step 1: Complete cov_gemm (if not done in Phase 3)

Verify `flashlib/linalg/cov_gemm/tilelang/cov_gemm.py` works:
```python
from flashlib.linalg.cov_gemm.tilelang import tilelang_cov_gemm
X = torch.randn(4096, 128, device="cuda")
C = tilelang_cov_gemm(X)  # (128, 128)
```

### Step 2: Create gram_gemm TileLang backend

```bash
mkdir -p flashlib/linalg/gram_gemm/tilelang
```

`tilelang/__init__.py`:
```python
"""gram_gemm TileLang backend."""
from flashlib.linalg.gram_gemm.tilelang.gram_gemm import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_gram_gemm,
)

__all__ = ["tilelang_gram_gemm"]
```

`tilelang/gram_gemm.py`:
```python
"""TileLang gram_gemm: X @ X^T for wide X (D >> N)."""
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

from flashlib.linalg.gram_gemm.triton import gram_gemm as triton_gram_gemm


def tilelang_gram_gemm(X):
    """X @ X^T using TileLang. Falls back to Triton if unavailable.

    Args:
        X: (N, D) float32 CUDA tensor, D >> N.

    Returns:
        (N, N) float32 Gram matrix.
    """
    if not _try_init_tilelang():
        return triton_gram_gemm(X)

    import tilelang
    from tilelang import T

    N, D = X.shape
    X = X.contiguous()

    BLOCK_N = min(64, N)
    BLOCK_D = min(128, D)

    @tilelang.jit(out_idx=[-1])
    def _build_gram_gemm(N, D, BLOCK_N, BLOCK_D):
        @T.prim_func
        def kernel(
            X: T.Tensor[[N, D], "float32"],
            Out: T.Tensor[[N, N], "float32"],
        ):
            X_shared_i = T.alloc_shared([BLOCK_N, BLOCK_D], dtype="float32")
            X_shared_j = T.alloc_shared([BLOCK_N, BLOCK_D], dtype="float32")
            acc_frag = T.alloc_fragment([BLOCK_N, BLOCK_N], dtype="float32")

            pid_i = T.get_block_idx(0)
            pid_j = T.get_block_idx(1)

            ni_start = pid_i * BLOCK_N
            nj_start = pid_j * BLOCK_N

            if nj_start + BLOCK_N <= ni_start:
                return

            T.clear(acc_frag)

            for d_start in T.serial(0, D, BLOCK_D):
                T.copy(X[ni_start:ni_start+BLOCK_N, d_start:d_start+BLOCK_D], X_shared_i)
                T.copy(X[nj_start:nj_start+BLOCK_N, d_start:d_start+BLOCK_D], X_shared_j)
                T.gemm(X_shared_i, X_shared_j, acc_frag, transpose_A=True)

            T.copy(acc_frag, Out[ni_start:ni_start+BLOCK_N, nj_start:nj_start+BLOCK_N])

        return kernel

    result = kernel(X)
    result = torch.triu(result) + torch.triu(result, diagonal=1).T
    return result
```

### Step 3: Create ab_gemm TileLang backend

```bash
mkdir -p flashlib/linalg/ab_gemm/tilelang
```

Same pattern. `A^T @ B` where A is (N, D1), B is (N, D2), output is (D1, D2).

### Step 4: Update `__init__.py` files

For each GEMM variant, add the tilelang import to its `__init__.py`:

```python
from flashlib.linalg.<op>.tilelang import tilelang_<op>
```

## Tests

```python
def test_cov_gemm_matches_triton():
    N, D = 4096, 128
    X = torch.randn(N, D, device="cuda", dtype=torch.float32)
    out_tl = tilelang_cov_gemm(X)
    out_tr = triton_cov_gemm(X)
    torch.testing.assert_close(out_tl, out_tr, rtol=1e-2, atol=1e-1)

def test_gram_gemm_matches_triton():
    N, D = 256, 4096
    X = torch.randn(N, D, device="cuda", dtype=torch.float32)
    out_tl = tilelang_gram_gemm(X)
    out_tr = triton_gram_gemm(X)
    torch.testing.assert_close(out_tl, out_tr, rtol=1e-2, atol=1e-1)

def test_ab_gemm_matches_triton():
    N, D1, D2 = 4096, 128, 64
    A = torch.randn(N, D1, device="cuda", dtype=torch.float32)
    B = torch.randn(N, D2, device="cuda", dtype=torch.float32)
    out_tl = tilelang_ab_gemm(A, B)
    out_tr = triton_ab_gemm(A, B)
    torch.testing.assert_close(out_tl, out_tr, rtol=1e-2, atol=1e-1)
```

## Validation on Modal

```bash
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_gram
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_ab
```

## Troubleshooting

**Accumulation error in gram_gemm:** The `T.gemm(X_i, X_j, acc, transpose_A=True)` accumulates across D chunks. This is correct — we want `sum_d X_i[:, d] * X_j[:, d]`.

**Large N*N output:** For N > 4096, the (N, N) output may be large. Consider streaming the output write.

## Checkpoint

- [ ] `tilelang_cov_gemm` matches Triton output
- [ ] `tilelang_gram_gemm` matches Triton output
- [ ] `tilelang_ab_gemm` matches Triton output
- [ ] All GEMM parity tests pass on Modal H100
