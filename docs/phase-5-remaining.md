# Phase 5: Remaining Primitives

## What You're Building

TileLang backends (or stubs) for the remaining 10 primitives. Some have real TileLang kernels, others delegate to Triton.

## Primitive Categories

### Real TileLang Kernels (5)

| Primitive | Kernel Type | Complexity |
|-----------|-------------|------------|
| `standard_scaler` | Elementwise + reduction | Low |
| `multinomial_nb` | Count + log-prob | Low |
| `linear_regression` | GEMM + solve | Medium |
| `ridge` | GEMM + solve + regularization | Medium |
| `logistic_regression` | GEMM + sigmoid + BCE | Medium |

### Stubs — Delegate to Triton (4)

| Primitive | Reason |
|-----------|--------|
| `hdbscan` | Complex multi-stage pipeline (MRD + Boruvka + condensation) |
| `umap` | Complex (KNN graph + fuzzy simplicial set + SGD) |
| `tsne` | Complex (P-matrix + gradient SGD) |
| `random_forest` | No `impl.py` dispatcher — direct class |

### Delegates to eigh + KMeans (1)

| Primitive | Approach |
|-----------|----------|
| `spectral_clustering` | Delegates to eigh + flash_kmeans |

## StandardScaler

### Background

`flashlib/primitives/standard_scaler/impl.py` uses `_use_cutedsl(backend)` helper. Follow the same pattern.

### Step-by-Step

1. Create `tilelang/__init__.py` and `tilelang/scaler.py`
2. Implement elementwise mean/variance reduction + scaling
3. Update `impl.py` to add `_use_tilelang(backend)` helper
4. Update `__init__.py` and `_LAZY_ATTRS`

### Kernel Design

```python
# Two-pass: (1) compute mean + variance, (2) scale
# Pass 1: Parallel reduction for mean and variance
# Pass 2: Elementwise (X - mean) * inv_std
```

## MultinomialNB

### Background

MultinomialNB is count-based. The TileLang kernel can accelerate the predict step (log-prob computation).

### Approach

Delegate to Triton for fit, implement TileLang predict kernel:
```python
# log_proba = X_test @ log_proba_per_feature.T + log_prior
# This is a GEMM — use T.gemm
```

## Linear Regression

### Background

Normal equations: `w = (X^T X)^{-1} X^T y`. The dominant cost is `X^T @ X` (already ported as cov_gemm).

### Approach

Delegate to TileLang cov_gemm + torch.linalg.solve:
```python
XtX = tilelang_cov_gemm(X)
Xty = X.T @ y
w = torch.linalg.solve(XtX, Xty)
```

## Ridge Regression

### Background

Like linear regression but with regularization: `w = (X^T X + alpha * I)^{-1} X^T y`.

### Approach

```python
XtX = tilelang_cov_gemm(X)
XtX += alpha * torch.eye(D, device=X.device)
Xty = X.T @ y
w = torch.linalg.solve(XtX, Xty)
```

## Logistic Regression

### Background

L-BFGS optimization with GEMV-bound forward pass. The TileLang kernel can accelerate the forward GEMV.

### Approach

Implement TileLang forward GEMV (matrix-vector multiply), delegate L-BFGS loop to Python.

## DBSCAN

### Background

DBSCAN uses flash_knn for radius search + connected components. The TileLang backend delegates both.

### Approach

```python
# 1. KNN radius search (reuse TileLang KNN)
# 2. Connected components (reuse existing)
```

## Spectral Clustering

### Background

Delegates to eigh + flash_kmeans. If both support TileLang, the whole pipeline uses TileLang.

### Approach

```python
# 1. Build KNN graph (reuse flash_knn)
# 2. Compute Laplacian eigenvectors (torch.linalg.eigh)
# 3. Run flash_kmeans on eigenvectors (reuse TileLang KMeans)
```

## Stubs (hdbscan, umap, tsne, random_forest)

These simply delegate to Triton when `backend="tilelang"`:

```python
def tilelang_<op>(*args, **kwargs):
    """TileLang stub — delegates to Triton."""
    return triton_<op>(*args, **kwargs)
```

### random_forest Special Case

`random_forest` has no `impl.py` dispatcher — it's a direct `FlashRandomForestClassifier` class. The stub is a module-level function:

```python
# flashlib/primitives/random_forest/tilelang/__init__.py
def tilelang_random_forest(*args, backend=None, **kwargs):
    """TileLang stub — delegates to Triton."""
    from flashlib.primitives.random_forest.impl import FlashRandomForestClassifier
    return FlashRandomForestClassifier(*args, **kwargs)
```

## Step-by-Step (All Primitives)

For each primitive:

1. Create `tilelang/__init__.py` with re-exports
2. Create `tilelang/<kernel>.py` with TileLang kernel or stub
3. Update `impl.py` to add `"tilelang"` branch
4. Update `__init__.py` to add `lazy_attr()` for tilelang entry points
5. Update `flashlib/__init__.py` `_LAZY_ATTRS`

## Update `flashlib/__init__.py`

Add all TileLang entries to `_LAZY_ATTRS`:

```python
    # TileLang backends
    "flash_kmeans_tilelang":           ("flashlib.primitives.kmeans", "tilelang_kmeans_Euclid"),
    "flash_knn_tilelang":              ("flashlib.primitives.knn", "tilelang_flash_knn"),
    "flash_pca_tilelang":              ("flashlib.primitives.pca", "tilelang_pca"),
    "flash_truncated_svd_tilelang":    ("flashlib.primitives.truncated_svd", "tilelang_truncated_svd"),
    "flash_standard_scaler_tilelang":  ("flashlib.primitives.standard_scaler", "tilelang_standard_scaler_fit_transform"),
    "flash_multinomial_nb_tilelang":   ("flashlib.primitives.multinomial_nb", "tilelang_multinomial_nb"),
    "flash_linear_regression_tilelang":("flashlib.primitives.linear_regression", "tilelang_linear_regression"),
    "flash_ridge_tilelang":            ("flashlib.primitives.ridge", "tilelang_ridge_regression"),
    "flash_logistic_regression_tilelang": ("flashlib.primitives.logistic_regression", "tilelang_logistic_regression"),
    "flash_dbscan_tilelang":           ("flashlib.primitives.dbscan", "tilelang_dbscan"),
    "flash_spectral_clustering_tilelang": ("flashlib.primitives.spectral_clustering", "tilelang_spectral_clustering"),
    "flash_hdbscan_tilelang":          ("flashlib.primitives.hdbscan", "tilelang_hdbscan"),
    "flash_umap_tilelang":             ("flashlib.primitives.umap", "tilelang_flash_umap"),
    "flash_tsne_tilelang":             ("flashlib.primitives.tsne", "tilelang_tsne"),
    "flash_random_forest_tilelang":    ("flashlib.primitives.random_forest", "tilelang_random_forest"),
    # TileLang linalg
    "tilelang_cov_gemm":               ("flashlib.linalg.cov_gemm", "tilelang_cov_gemm"),
    "tilelang_gram_gemm":              ("flashlib.linalg.gram_gemm", "tilelang_gram_gemm"),
    "tilelang_ab_gemm":                ("flashlib.linalg.ab_gemm", "tilelang_ab_gemm"),
```

## Tests

Each primitive gets a test in `tests/test_tilelang_parity.py`. See the existing test skeleton for the pattern.

## Validation on Modal

```bash
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py
```

## Troubleshooting

**Stub crashes:** Ensure the stub function accepts `**kwargs` and passes them through to the Triton function.

**Missing `__init__.py`:** Every `tilelang/` directory needs an `__init__.py`.

**Import cycle:** The `tilelang/<kernel>.py` file imports from `triton/` at module level. This is fine — it's the same pattern as cutedsl.

## Checkpoint

- [ ] All 10 primitives have `tilelang/` directories
- [ ] All `impl.py` files accept `backend="tilelang"`
- [ ] All `__init__.py` files have tilelang lazy imports
- [ ] `flashlib/__init__.py` has all tilelang `_LAZY_ATTRS` entries
- [ ] Stubs delegate to Triton without error
- [ ] Real TileLang kernels pass parity tests
