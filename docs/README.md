# TileLang Backend for FlashLib — Learning Guide

## What This Project Is

You are porting 15 FlashLib ML primitives from Triton to TileLang. The goal is to learn TileLang by writing real GPU kernels — not toy examples.

## Setup

**Prerequisites:**
- Python 3.9+
- NVIDIA GPU with compute capability 8.0+ (A100, H100, H200, RTX 3090, RTX 4090, L4, etc.)
- CUDA toolkit

**Install dependencies:**

```bash
# Option A — pip (venv)
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Option B — conda
conda create -n flashlib-tilelang python=3.11
conda activate flashlib-tilelang
pip install -r requirements.txt

# Option C — install flashlib in editable mode (recommended for development)
pip install -e ".[tilelang,dev]"
```

Phase 0 (infrastructure verification) works on CPU. Phases 1–8 require a CUDA GPU.

**No local GPU?** You can use Modal (https://modal.com) for cloud H100 access. The branch includes ready-to-run Modal scripts. See `modal-workflow.md` for details.

## How to Use These Docs

Read in this order:

1. **This file** — orientation and phase overview
2. `tilelang-api.md` — TileLang API reference (read before Phase 1)
3. `flashlib-architecture.md` — how FlashLib's multi-backend system works
4. `how-to-port-a-primitive.md` — the porting recipe (reference during all phases)
5. `modal-workflow.md` — how to use Modal H100 (skip if using your own GPU)
6. `debugging.md` — when something breaks

## Warmup: TileLang Puzzles

Before starting the phases, solve the first 7 [TileLang Puzzles](https://github.com/tile-ai/tilelang-puzzles). They teach the exact patterns you need for the FlashLib port:

| Puzzle | What It Teaches | FlashLib Use |
|--------|----------------|--------------|
| 01-copy | `T.copy`, `T.Kernel`, basic structure | Every kernel loads data |
| 02-vector-add | `T.Parallel`, elementwise ops | standard_scaler transform |
| 03-outer-vec-add | 2D `T.Parallel`, broadcast | Distance matrix in KNN |
| 04-backward-op | Multi-output kernels | — |
| 05-reduce-sum | `T.Parallel` reduction | Mean/variance in standard_scaler |
| 06-softmax | Multi-pass reduction + epilogue | KMeans argmin epilogue |
| 07-scalar-flash-attn | Fused GEMM + epilogue | KMeans assign kernel |

Puzzles 8-10 (GEMM, conv, dequant GEMM) are optional — they teach advanced patterns but the FlashLib port doesn't require them.

```bash
# Clone and solve
git clone https://github.com/tile-ai/tilelang-puzzles
cd tilelang-puzzles
python3 puzzles/01-copy.py       # read the puzzle
python3 ans/01-copy.py           # check the answer
```

After puzzles 1-7, Phase 1 (KMeans) will take 2-3 hours instead of 4-6.

## Phase Guides

Follow the phase guides in order:

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

| Phase | What You Build | Est. GPU Hours |
|-------|---------------|----------------|
| 0 | Verify infrastructure | ~1 |
| 1 | KMeans TileLang kernel | ~3 |
| 2 | KNN TileLang kernel | ~4 |
| 3 | PCA + Truncated SVD | ~3 |
| 4 | GEMM variants | ~4 |
| 5 | 10 remaining primitives | ~9 |
| 6 | Dispatcher polish | ~1 |
| 7 | Parity tests | ~5 |
| 8 | Benchmarks | ~6 |
| Buffer | Debugging | ~8 |
| **Total** | | **~44** |

## Quick Reference

```bash
# Verify Phase 0 (no GPU needed)
python -c "from flashlib._tilelang import _try_init_tilelang; print(_try_init_tilelang())"

# Run tests locally
pytest tests/test_tilelang_parity.py -k test_kmeans -v

# Run tests on Modal H100
modal run scripts/modal/run_test.py --path tests/test_tilelang_parity.py -k test_kmeans

# Interactive shell on Modal H100
modal run scripts/modal/shell.py

# Benchmark on Modal
modal run scripts/modal/bench_primitive.py --primitive kmeans
```

## Key Concepts

- **TileLang** is a Python DSL for GPU kernels that compiles to TIR (TVM IR). It targets NVIDIA GPUs with sm_80+ support.
- **FlashLib** uses a multi-backend pattern: every primitive has a Triton backend (default), an optional CuteDSL backend (Hopper-only), and now a TileLang backend.
- **The cutedsl pattern** is the template: `_try_init_cutedsl()` guard, fallback to Triton, `lazy_attr()` in `__init__.py`, entry in `_LAZY_ATTRS`.
- **Parity** means the TileLang output must match the Triton output within precision tolerances.
