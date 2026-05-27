# Phase 6: Dispatcher Integration

## What You're Building

Verification that every `impl.py` correctly accepts `backend="tilelang"` and routes to the TileLang backend (or stub).

## Background Reading

| File | What to Check |
|------|---------------|
| All `flashlib/primitives/*/impl.py` | `"tilelang"` branch exists |
| All `flashlib/linalg/*/tilelang/` | TileLang modules exist |
| `flashlib/__init__.py` | All `_LAZY_ATTRS` entries present |

## Checklist

For each primitive, verify:

### Primitives with Dispatchers

| Primitive | impl.py Branch | Expected Behavior |
|-----------|---------------|-------------------|
| kmeans | `elif chosen == "tilelang":` | Calls `tilelang_kmeans_Euclid` |
| knn | `if chosen == "tilelang":` | Calls `tilelang_flash_knn` |
| pca | `if backend == "tilelang":` | Calls `tilelang_pca` |
| truncated_svd | `if backend == "tilelang":` | Calls `tilelang_truncated_svd` |
| standard_scaler | `_use_tilelang(backend)` | Calls `tilelang_standard_scaler_*` |
| multinomial_nb | `if backend == "tilelang":` | Calls `tilelang_multinomial_nb` |
| linear_regression | `if backend == "tilelang":` | Calls `tilelang_linear_regression` |
| ridge | `if backend == "tilelang":` | Calls `tilelang_ridge_regression` |
| logistic_regression | `if backend == "tilelang":` | Calls `tilelang_logistic_regression` |
| dbscan | `if backend == "tilelang":` | Calls `tilelang_dbscan` |
| spectral_clustering | `if backend == "tilelang":` | Calls `tilelang_spectral_clustering` |
| hdbscan | `if backend == "tilelang":` | Calls `tilelang_hdbscan` (stub) |
| umap | `if backend == "tilelang":` | Calls `tilelang_flash_umap` (stub) |
| tsne | `if backend == "tilelang":` | Calls `tilelang_tsne` (stub) |

### Special Cases

| Primitive | Location | Expected Behavior |
|-----------|----------|-------------------|
| random_forest | `flashlib/primitives/random_forest/__init__.py` | `tilelang_random_forest` lazy import |

### Linalg Primitives

| Primitive | Location | Expected Behavior |
|-----------|----------|-------------------|
| cov_gemm | `flashlib/linalg/cov_gemm/__init__.py` | `tilelang_cov_gemm` import |
| gram_gemm | `flashlib/linalg/gram_gemm/__init__.py` | `tilelang_gram_gemm` import |
| ab_gemm | `flashlib/linalg/ab_gemm/__init__.py` | `tilelang_ab_gemm` import |

## Step-by-Step

### 1. Verify all impl.py files accept "tilelang"

```bash
grep -r "tilelang" flashlib/primitives/*/impl.py flashlib/linalg/*/tilelang/
```

### 2. Verify all lazy imports work

```python
python -c "
from flashlib._tilelang import tilelang_available
print(f'tilelang available: {tilelang_available()}')

# Test lazy imports (should not crash even without tilelang)
import flashlib
print(hasattr(flashlib, 'flash_kmeans_tilelang'))
"
```

### 3. Verify stubs delegate without error

Run the stub tests:
```bash
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_stubs
```

## Validation on Modal

```bash
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py
```

## Troubleshooting

**`AttributeError: module has no attribute`:** Check that the `_LAZY_ATTRS` entry in `flashlib/__init__.py` is correct.

**Import error in lazy_attr:** Check that the module path in `lazy_attr()` matches the actual file location.

**Stub raises NotImplementedError:** Stubs should delegate to Triton, not raise. Fix the stub to call the Triton function.

## Checkpoint

- [ ] Every `impl.py` accepts `backend="tilelang"` without error
- [ ] Every `__init__.py` has tilelang lazy imports
- [ ] `flashlib.__init__.py._LAZY_ATTRS` has all tilelang entries
- [ ] Stubs delegate to Triton silently
- [ ] `python -c "import flashlib"` works without error
