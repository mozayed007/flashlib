# How to Port a Primitive to TileLang

## The Recipe

Follow these steps for every primitive. Adjust based on whether it has a cutedsl backend or not.

### Step 1: Study the Existing Backend

Read the Triton kernel to understand:
- Input/output shapes and dtypes
- Tiling strategy (BLOCK_N, BLOCK_K, etc.)
- The core computation (GEMM, elementwise, reduction)
- Epilogue logic (argmin, softmax, etc.)

Key files to read:
- `flashlib/primitives/<op>/triton/<kernel>.py` — the Triton kernel
- `flashlib/primitives/<op>/impl.py` — the dispatcher
- `flashlib/primitives/<op>/__init__.py` — the public API

### Step 2: Create the TileLang Directory

```bash
mkdir -p flashlib/primitives/<op>/tilelang
```

### Step 3: Create `tilelang/__init__.py`

Follow the cutedsl re-export pattern:

```python
"""<Op> TileLang backend."""
from flashlib.primitives.<op>.tilelang.<kernel> import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_<op>,
)

__all__ = ["tilelang_<op>"]
```

### Step 4: Create `tilelang/<kernel>.py`

Structure:

```python
"""TileLang kernel for <Op>."""
from __future__ import annotations
from typing import Optional

import torch

# ---------------------------------------------------------------------------
# TileLang availability guard
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Triton fallback (always available)
# ---------------------------------------------------------------------------
from flashlib.primitives.<op>.triton.<kernel> import triton_<op>

# ---------------------------------------------------------------------------
# TileLang kernel
# ---------------------------------------------------------------------------

def tilelang_<op>(x, ..., **kwargs):
    """TileLang implementation of <Op>.

    Falls back to Triton if tilelang is unavailable.
    """
    if not _try_init_tilelang():
        return triton_<op>(x, ..., **kwargs)

    import tilelang
    from tilelang import T

    # Define kernel
    @tilelang.jit(out_idx=[-1])
    def _kernel(N, D, BLOCK_N, BLOCK_D):
        @T.prim_func
        def kernel(
            X: T.Tensor[[N, D], "float16"],
            Out: T.Tensor[[N], "float32"],
        ):
            # ... kernel body ...
            pass
        return kernel

    # Call kernel
    compiled = _kernel(N, D, BLOCK_N, BLOCK_D)
    return compiled(x)
```

### Step 5: Update `impl.py`

Add the `"tilelang"` branch to the dispatcher:

```python
# Simple pattern (most primitives):
def flash_<op>(..., backend=None):
    if backend == "cutedsl":
        return cutedsl_<op>(...)
    if backend == "tilelang":
        return tilelang_<op>(...)
    return triton_<op>(...)
```

```python
# Complex pattern (kmeans, knn) with _route():
def _route(..., backend=None):
    if backend is not None:
        return backend
    # ... shape-based heuristics ...
```

### Step 6: Update `__init__.py`

Add lazy imports for TileLang entry points:

```python
from flashlib._lazy import lazy_attr

tilelang_<op> = lazy_attr(
    "flashlib.primitives.<op>.tilelang", "tilelang_<op>",
)

__all__ = [
    ...,
    "tilelang_<op>",
]
```

### Step 7: Update `flashlib/__init__.py`

Add entries to `_LAZY_ATTRS`:

```python
_LAZY_ATTRS = {
    ...,
    "flash_<op>_tilelang": ("flashlib.primitives.<op>", "tilelang_<op>"),
}
```

### Step 8: Write Parity Test

Add a test class to `tests/test_tilelang_parity.py`:

```python
class Test<Op>TileLang:
    def test_matches_triton(self):
        _seeded()
        # ... setup ...
        from flashlib.primitives.<op>.impl import flash_<op>
        out_tl = flash_<op>(..., backend="tilelang")
        out_tr = flash_<op>(..., backend="triton")
        torch.testing.assert_close(out_tl, out_tr, rtol=1e-2, atol=1e-1)
```

### Step 9: Test on Modal H100

```bash
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_<op>
```

## Patterns by Primitive Type

### GEMM-based (cov_gemm, gram_gemm, ab_gemm)
- Core: `T.gemm` with appropriate transpose flags
- Tile: (BLOCK_N, BLOCK_D) for input, (BLOCK_D, BLOCK_D) for output
- Symmetric optimization: compute upper triangle only

### Assign-based (kmeans)
- Core: `T.gemm` for cross term + serial epilogue for argmin
- Tile: (BLOCK_N, D) for X, (BLOCK_K, D) for C
- Epilogue: `T.serial` for running minimum

### Top-K (knn)
- Core: `T.gemm` for distances + iterative insert for top-K
- Multiple kernels: distance kernel + gather kernel

### Reduction-based (standard_scaler)
- Core: `T.Parallel` reduction for mean, then elementwise for scaling

### Stub (hdbscan, umap, tsne, random_forest)
- Just delegate to Triton — no TileLang kernel needed
- Add the `"tilelang"` branch in impl.py that calls the Triton function

## Common Pitfalls

1. **`T.gemm` accumulates** — always `T.clear()` before first GEMM
2. **JIT caches by signature** — don't re-wrap `@tilelang.jit` inside a function called per-invocation
3. **Shared memory limits** — check `BLOCK_N * D * dtype_bytes` fits
4. **dtype mismatches** — TileLang kernels are dtype-strict; cast inputs before passing
5. **Serial epilogues are slow** — start with serial for correctness, optimize later
