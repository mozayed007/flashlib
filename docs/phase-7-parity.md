# Phase 7: Full Parity Test Suite

## What You're Building

Complete parity test coverage for all 15 primitives. Fix any failures.

## Background Reading

| File | What to Study |
|------|---------------|
| `tests/test_tilelang_parity.py` | Skeleton tests (created in Phase 0) |
| `tests/test_backend_parity.py` | Reference pattern for parity tests |
| `docs/debugging.md` | Common TileLang errors |

## Test Matrix

| Test Class | Primitive | Test Methods |
|------------|-----------|--------------|
| `TestKMeansTileLang` | kmeans | assign_matches_torch, assign_matches_triton, full_lloyd_loop |
| `TestKNNTileLang` | knn | topk_matches_triton |
| `TestPCATileLang` | pca | pca_matches_triton |
| `TestTruncatedSVDTileLang` | truncated_svd | svd_matches_triton |
| `TestCovGEMMTileLang` | cov_gemm | cov_gemm_matches_triton |
| `TestGramGEMMTileLang` | gram_gemm | gram_gemm_matches_triton |
| `TestAbGEMMTileLang` | ab_gemm | ab_gemm_matches_triton |
| `TestStandardScalerTileLang` | standard_scaler | fit_transform_matches_triton |
| `TestMultinomialNBTileLang` | multinomial_nb | tilelang_delegates_to_triton |
| `TestLinearRegressionTileLang` | linear_regression | matches_triton |
| `TestRidgeTileLang` | ridge | matches_triton |
| `TestLogisticRegressionTileLang` | logistic_regression | matches_triton |
| `TestDBSCANTileLang` | dbscan | labels_match_triton |
| `TestSpectralClusteringTileLang` | spectral_clustering | labels_match_triton |
| `TestStubsDelegateToTriton` | hdbscan, umap, tsne, random_forest | stub_delegates |

## Step-by-Step

### 1. Run the full suite

```bash
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py
```

### 2. Fix failures

For each failing test:

1. Read the error message
2. Check `docs/debugging.md` for the error pattern
3. Fix the kernel or wrapper
4. Re-run the specific test: `modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_<name>`

### 3. Common failure patterns

**Mismatch rate too high:**
- Check dtype casting (float16 vs float32)
- Check that `T.clear()` is called before `T.gemm`
- Check argmin uses global index (`k_start + k`)

**Shape mismatch:**
- Check that squeeze/unsqueeze is handled correctly
- Check B=1 assumption in kernels

**Import error:**
- Check `lazy_attr()` module path matches actual file location
- Check `__init__.py` exports match function names

**Timeout:**
- TileLang first-call compilation is slow (30-120s)
- Increase Modal timeout if needed

### 4. Add missing tests

If a primitive doesn't have a test yet, add it following the existing pattern:

```python
class Test<Op>TileLang:
    def test_matches_triton(self):
        _seeded()
        # ... setup data ...
        from flashlib.primitives.<op>.impl import flash_<op>
        out_tl = flash_<op>(..., backend="tilelang")
        out_tr = flash_<op>(..., backend="triton")
        # ... compare ...
```

## Validation

```bash
# Full suite
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py

# With verbose output
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py --tb long

# Specific phase
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k "TestKMeans or TestKNN"
```

## Troubleshooting

**Tests skip:** Check that tilelang is installed in the Modal container. The `setup_env.py` image includes `tilelang>=0.1.6`.

**Random failures:** Set `_seeded()` at the start of each test for reproducibility.

**Tolerance too tight:** Adjust `rtol` and `atol` in `torch.testing.assert_close()`. Default: `rtol=1e-2, atol=1e-1`.

**Tests take too long:** Reduce problem sizes (N, D, K) for faster iteration. Only test with large shapes for the final validation.

## Checkpoint

- [ ] All 15 primitive test classes exist
- [ ] All tests pass on Modal H100
- [ ] No tests skip (except on CPU CI)
- [ ] Test suite completes in < 10 minutes
