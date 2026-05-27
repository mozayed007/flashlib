# Phase 3: PCA + Truncated SVD

## What You're Building

TileLang backends for PCA and Truncated SVD. Both delegate to a TileLang `cov_gemm` kernel (X^T @ X) plus `torch.linalg.eigh`.

**End result:**
- `flash_pca(X, K, backend="tilelang")` returns `(eigenvalues, eigenvectors)`
- `flash_truncated_svd(X, K, backend="tilelang")` returns `(S, Vh)`

## Background Reading

| File | What to Study |
|------|---------------|
| `flashlib/linalg/cov_gemm/triton/cov_gemm.py` | Triton cov_gemm kernel (123 lines) |
| `flashlib/linalg/cov_gemm/__init__.py` | Simple 5-line re-export |
| `flashlib/primitives/pca/triton/` | How PCA delegates to cov_gemm + eigh |
| `flashlib/primitives/pca/impl.py` | 41-line dispatcher |
| `flashlib/primitives/truncated_svd/impl.py` | 67-line dispatcher |
| `docs/phase-4-gemm.md` | GEMM patterns (read if doing phase 4 first) |

## Key Concepts

### PCA Computation

1. Compute covariance matrix: `C = X^T @ X` (or `X @ X^T` for D >> N)
2. Eigendecompose: `eigenvalues, eigenvectors = torch.linalg.eigh(C)`
3. Return top-K eigenpairs

### cov_gemm: Tall-Skinny GEMM

For N >> D (common case), compute `X^T @ X` where X is (N, D):
- Output is (D, D) — small, fits in L2 cache
- Stream X in panels of BLOCK_N rows
- Symmetric optimization: compute upper triangle only

## Step-by-Step

### Step 1: Create `flashlib/linalg/cov_gemm/tilelang/`

```bash
mkdir -p flashlib/linalg/cov_gemm/tilelang
```

### Step 2: Create `tilelang/__init__.py`

```python
"""cov_gemm TileLang backend."""
from flashlib.linalg.cov_gemm.tilelang.cov_gemm import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_cov_gemm,
)

__all__ = ["tilelang_cov_gemm"]
```

### Step 3: Create `tilelang/cov_gemm.py`

```python
"""TileLang cov_gemm: X^T @ X for tall-skinny X (N >> D)."""
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

from flashlib.linalg.cov_gemm.triton import cov_gemm as triton_cov_gemm


def tilelang_cov_gemm(X):
    """X^T @ X using TileLang. Falls back to Triton if unavailable.

    Args:
        X: (N, D) float32 CUDA tensor, N >> D.

    Returns:
        (D, D) float32 covariance matrix.
    """
    if not _try_init_tilelang():
        return triton_cov_gemm(X)

    import tilelang
    from tilelang import T

    N, D = X.shape
    X = X.contiguous()

    BLOCK_N = min(128, N)
    BLOCK_D = min(64, D)

    @tilelang.jit(out_idx=[-1])
    def _build_cov_gemm(N, D, BLOCK_N, BLOCK_D):
        @T.prim_func
        def kernel(
            X: T.Tensor[[N, D], "float32"],
            Out: T.Tensor[[D, D], "float32"],
        ):
            X_shared = T.alloc_shared([BLOCK_N, BLOCK_D], dtype="float32")
            acc_frag = T.alloc_fragment([BLOCK_D, BLOCK_D], dtype="float32")

            pid_i = T.get_block_idx(0)
            pid_j = T.get_block_idx(1)

            di_start = pid_i * BLOCK_D
            dj_start = pid_j * BLOCK_D

            # Skip lower triangle
            if dj_start + BLOCK_D <= di_start:
                return

            T.clear(acc_frag)

            # Stream X in panels
            for n_start in T.serial(0, N, BLOCK_N):
                T.copy(X[n_start:n_start+BLOCK_N, di_start:di_start+BLOCK_D], X_shared)
                # Need second shared buffer for j-dimension
                X_shared_j = T.alloc_shared([BLOCK_N, BLOCK_D], dtype="float32")
                T.copy(X[n_start:n_start+BLOCK_N, dj_start:dj_start+BLOCK_D], X_shared_j)
                T.gemm(X_shared, X_shared_j, acc_frag, transpose_A=True)

            T.copy(acc_frag, Out[di_start:di_start+BLOCK_D, dj_start:dj_start+BLOCK_D])

        return kernel

    out = torch.zeros(D, D, device=X.device, dtype=torch.float32)
    grid_d = (D + BLOCK_D - 1) // BLOCK_D

    kernel = _build_cov_gemm(N, D, BLOCK_N, BLOCK_D)
    result = kernel(X)

    # Symmetric: mirror upper to lower
    result = torch.triu(result) + torch.triu(result, diagonal=1).T
    return result
```

### Step 4: Update `flashlib/linalg/cov_gemm/__init__.py`

```python
"""cov_gemm(X) -> X.T @ X, tall-skinny GEMM optimized for N >> D."""
from flashlib.linalg.cov_gemm.triton import cov_gemm, full_gemm
from flashlib.linalg.cov_gemm.tilelang import tilelang_cov_gemm
from flashlib.linalg.cov_gemm import cost

__all__ = ["cov_gemm", "full_gemm", "tilelang_cov_gemm", "cost"]
```

### Step 5: Create PCA TileLang backend

```bash
mkdir -p flashlib/primitives/pca/tilelang
```

`tilelang/__init__.py`:
```python
"""PCA TileLang backend."""
from flashlib.primitives.pca.tilelang.pca import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_pca,
)

__all__ = ["tilelang_pca"]
```

`tilelang/pca.py`:
```python
"""TileLang PCA — delegates to TileLang cov_gemm + torch.linalg.eigh."""
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

from flashlib.primitives.pca.triton import triton_pca


def tilelang_pca(X, K, **kwargs):
    """PCA using TileLang cov_gemm + eigh. Falls back to Triton if unavailable."""
    if not _try_init_tilelang():
        return triton_pca(X, K, **kwargs)

    from flashlib.linalg.cov_gemm.tilelang import tilelang_cov_gemm

    N, D = X.shape
    if N > 4 * D:
        # Tall-skinny: cov path
        C = tilelang_cov_gemm(X)
    else:
        # Wide: Gram path (X @ X^T)
        C = X @ X.T  # TODO: port gram_gemm to TileLang

    eigenvalues, eigenvectors = torch.linalg.eigh(C)
    # eigh returns ascending; we want top-K
    eigenvalues = eigenvalues[-K:]
    eigenvectors = eigenvectors[:, -K:]

    return eigenvalues, eigenvectors
```

### Step 6: Create Truncated SVD TileLang backend

Same pattern as PCA — delegates to TileLang cov_gemm + eigh:

```bash
mkdir -p flashlib/primitives/truncated_svd/tilelang
```

### Step 7: Update dispatchers

Update `pca/impl.py`:
```python
if backend == "tilelang":
    return tilelang_pca(X, K)
```

Update `truncated_svd/impl.py`:
```python
if backend == "tilelang":
    return tilelang_truncated_svd(X, K)
```

### Step 8: Update `__init__.py` files and `_LAZY_ATTRS`

Add lazy imports for `tilelang_pca`, `tilelang_truncated_svd`, `tilelang_cov_gemm`.

## Tests

```python
def test_pca_matches_triton():
    N, D, K = 1024, 64, 8
    X = torch.randn(N, D, device="cuda", dtype=torch.float32)
    (evals_tl, evecs_tl) = flash_pca(X, K, backend="tilelang")
    (evals_tr, evecs_tr) = flash_pca(X, K, backend="triton")
    torch.testing.assert_close(evals_tl, evals_tr, rtol=1e-2, atol=1e-2)
```

## Validation on Modal

```bash
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_pca
```

## Troubleshooting

**eigh returns wrong order:** `torch.linalg.eigh` returns ascending eigenvalues. Slice `[-K:]` for top-K.

**Symmetric optimization bug:** Ensure `torch.triu(result) + torch.triu(result, diagonal=1).T` correctly mirrors the matrix.

## Checkpoint

- [ ] `tilelang_cov_gemm(X)` matches Triton cov_gemm output
- [ ] `flash_pca(X, K, backend="tilelang")` returns correct eigenpairs
- [ ] `flash_truncated_svd(X, K, backend="tilelang")` returns correct SVD
- [ ] Parity tests pass on Modal H100
