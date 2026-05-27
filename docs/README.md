# TileLang Backend for FlashLib — Learning Guide

## What This Project Is

You are porting 15 FlashLib ML primitives from Triton to TileLang. The goal is to learn TileLang by writing real GPU kernels — not toy examples.

## GPU Requirements

TileLang needs an NVIDIA GPU (sm_80+) to compile and run kernels. You have two options:

**Option A — Modal (recommended for beginners):**
Sign up at https://modal.com and install the CLI (`pip install modal`). The branch includes ready-to-use Modal scripts for H100 access. Free tier includes some credits; H100 costs $3.95/hr. See `modal-workflow.md` for details.

**Option B — Your own GPU:**
Any NVIDIA GPU with compute capability 8.0+ works (A100, H100, H200, RTX 3090, RTX 4090, L4, etc.). Install tilelang (`pip install tilelang`) and run tests locally.

Phase 0 (infrastructure verification) does not need a GPU. Phases 1–8 require one.

## How to Use These Docs

Read in this order:

1. **This file** — orientation and phase overview
2. `tilelang-api.md` — TileLang API reference (read before Phase 1)
3. `flashlib-architecture.md` — how FlashLib's multi-backend system works
4. `how-to-port-a-primitive.md` — the porting recipe (reference during all phases)
5. `modal-workflow.md` — how to use Modal H100
6. `debugging.md` — when something breaks

Then follow the phase guides in order:

```
phase-0-setup.md          Infrastructure verification (no GPU needed)
phase-1-kmeans.md         KMeans — THE reference phase (most detailed)
phase-2-knn.md            KNN — multi-kernel, top-K epilogue
phase-3-pca.md            PCA + Truncated SVD — cov_gemm + eigh delegation
phase-4-gemm.md           GEMM variants — cov_gemm, gram_gemm, ab_gemm
phase-5-remaining.md      10 remaining primitives (easy, stubs, GEMM wrappers)
phase-6-dispatcher.md     Dispatcher integration + verification
phase-7-parity.md         Full parity test suite
phase-8-benchmarks.md     Benchmarking + route heuristic updates
```

## Phase Summary

| Phase | What You Build | GPU Hours | Cost |
|-------|---------------|-----------|------|
| 0 | Verify infrastructure | 0.9 | $3.56 |
| 1 | KMeans TileLang kernel | 3.0 | $11.85 |
| 2 | KNN TileLang kernel | 4.0 | $15.80 |
| 3 | PCA + Truncated SVD | 2.6 | $10.27 |
| 4 | GEMM variants | 3.5 | $13.83 |
| 5 | 10 remaining primitives | 9.0 | $35.55 |
| 6 | Dispatcher polish | 1.4 | $5.53 |
| 7 | Parity tests | 4.6 | $18.17 |
| 8 | Benchmarks | 6.4 | $25.28 |
| Buffer | Debugging | 8.0 | $31.60 |
| **Total** | | **43.4** | **$171.44** |

## Quick Reference

```bash
# Verify Phase 0 (no GPU needed)
python -c "from flashlib._tilelang import _try_init_tilelang; print(_try_init_tilelang())"

# Run test on Modal H100
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_kmeans

# Run test locally (if you have a GPU)
pytest tests/test_tilelang_parity.py -k test_kmeans -v

# Interactive H100 shell (Modal)
modal run scripts/modal/shell.py

# Benchmark (Modal)
modal run scripts/modal/bench_primitive.py --primitive kmeans

# Check Modal budget
# https://modal.com/settings/usage
```

## Key Concepts

- **TileLang** is a Python DSL for GPU kernels that compiles to TIR (TVM IR). It targets NVIDIA GPUs with sm_80+ support.
- **FlashLib** uses a multi-backend pattern: every primitive has a Triton backend (default), an optional CuteDSL backend (Hopper-only), and now a TileLang backend.
- **The cutedsl pattern** is the template: `_try_init_cutedsl()` guard, fallback to Triton, `lazy_attr()` in `__init__.py`, entry in `_LAZY_ATTRS`.
- **Parity** means the TileLang output must match the Triton output within precision tolerances.
