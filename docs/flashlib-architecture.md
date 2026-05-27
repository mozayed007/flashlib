# FlashLib Architecture

## Overview

FlashLib is a multi-backend GPU ML library. Every primitive has:
1. A **Triton** backend (default, always available on CUDA)
2. An optional **CuteDSL** backend (Hopper-only, opt-in)
3. A new **TileLang** backend (sm_80+, opt-in)

## Multi-Backend Pattern

### The Dispatcher (`impl.py`)

Every primitive has an `impl.py` that acts as the dispatcher:

```python
# flashlib/primitives/<op>/impl.py
def flash_<op>(..., backend=None):
    if backend == "cutedsl":
        return cutedsl_<op>(...)
    return triton_<op>(...)
```

The dispatcher pattern varies by primitive:
- **Simple** (pca, dbscan, linear_regression, ridge, logistic_regression, multinomial_nb, spectral_clustering, hdbscan, umap): direct `if backend == "cutedsl"` branch
- **Helper** (standard_scaler): `_use_cutedsl(backend)` helper
- **Complex** (kmeans, knn): `_route()` function with shape-based heuristics

### The `__init__.py` (Lazy Imports)

Each primitive's `__init__.py` re-exports the dispatcher and uses `lazy_attr()` for backend-specific entry points:

```python
from flashlib._lazy import lazy_attr
from flashlib.primitives.<op>.impl import flash_<op>

# Lazy — only imported when accessed
cutedsl_<op> = lazy_attr("flashlib.primitives.<op>.cutedsl", "cutedsl_<op>")

__all__ = ["flash_<op>", "cutedsl_<op>", ...]
```

### The Top-Level `_LAZY_ATTRS`

`flashlib/__init__.py` has a `_LAZY_ATTRS` dict mapping attribute names to `(module_path, attr_name)`:

```python
_LAZY_ATTRS = {
    "flash_<op>":            ("flashlib.primitives.<op>", "flash_<op>"),
    "flash_<op>_triton":     ("flashlib.primitives.<op>", "flash_<op>_triton"),
    "flash_<op>_cutedsl":    ("flashlib.primitives.<op>", "flash_<op>_cutedsl"),
    ...
}
```

### The `_lazy.py` Helper

`flashlib/_lazy.py` provides `lazy_attr(module_path, name)` which returns a callable that resolves `getattr(importlib.import_module(module_path), name)(*args, **kwargs)` on first call. No changes needed to this file for TileLang.

### The `_hw.py` Hardware Fingerprint

`flashlib/_hw.py` provides `HwProps` with `sm_arch`, `is_cuda`, `is_hopper` properties. Routing rules use `hw = hw or _hw.current()`. TileLang works on sm_80+ so routing is simpler than CuteDSL (which requires sm_90+).

## The CuteDSL Pattern (Template for TileLang)

Every `cutedsl/__init__.py` follows this pattern:

```python
# flashlib/primitives/<op>/cutedsl/__init__.py
from flashlib.primitives.<op>.cutedsl.<file> import (
    _CUTEDSL_AVAILABLE,
    _try_init_cutedsl,
    cutedsl_<op>,
)
```

And every `cutedsl/<file>.py` has:

```python
_CUTEDSL_AVAILABLE = False
_CUTE_IMPORT_ERROR = None

def _try_init_cutedsl() -> bool:
    global _CUTEDSL_AVAILABLE, _CUTE_IMPORT_ERROR
    if _CUTEDSL_AVAILABLE:
        return True
    if _CUTE_IMPORT_ERROR is not None:
        return False
    try:
        import cutlass
        # ... setup ...
        _CUTEDSL_AVAILABLE = True
        return True
    except Exception as e:
        _CUTE_IMPORT_ERROR = e
        return False

# Triton fallback imports (always available)
from flashlib.primitives.<op>.triton.<file> import triton_<op>

def cutedsl_<op>(*args, **kwargs):
    if not _try_init_cutedsl():
        return triton_<op>(*args, **kwargs)  # fallback
    # ... CuteDSL kernel logic ...
```

## TileLang Backend Structure

The TileLang backend follows the same contract:

```
flashlib/primitives/<op>/
├── __init__.py          # Add tilelang lazy imports
├── impl.py              # Add "tilelang" branch in dispatcher
├── tilelang/
│   ├── __init__.py      # Re-export tilelang wrappers
│   └── <kernel>.py      # TileLang kernel implementation
├── triton/
│   └── ...
└── cutedsl/
    └── ...
```

For linalg primitives (cov_gemm, gram_gemm, ab_gemm) which have no cutedsl backend:

```
flashlib/linalg/<op>/
├── __init__.py          # Add tilelang re-export
├── tilelang/
│   ├── __init__.py
│   └── <kernel>.py
├── triton/
│   └── ...
└── cost.py
```

## Dispatcher Integration Checklist

For each primitive:

1. Create `tilelang/__init__.py` with re-exports
2. Create `tilelang/<kernel>.py` with TileLang kernel
3. Update `impl.py` to add `backend="tilelang"` branch
4. Update `__init__.py` to add `lazy_attr()` for tilelang entry points
5. Update `flashlib/__init__.py` `_LAZY_ATTRS` to add tilelang entries

## Routing Rules

TileLang doesn't need hardware gating (works on sm_80+), so routing is simpler:

```python
# In impl.py _route():
if backend is not None:
    return backend
# TileLang is opt-in only — never auto-routed
# (first-call compile is too slow for auto-routing)
```

## `flashlib/_tilelang.py`

Created in Phase 0. Provides:
- `tilelang_available() -> bool` — cached import check
- `tilelang()` — returns module or raises ImportError
- `_try_init_tilelang() -> bool` — the low-level guard
