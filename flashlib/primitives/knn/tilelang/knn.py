"""TileLang brute-force KNN — top-K nearest neighbors.

Phase: 2
Reference: flashlib/primitives/knn/triton/dispatch.py
Guide: docs/phase-2-knn.md
"""
from __future__ import annotations

from typing import Optional

import torch

# ---------------------------------------------------------------------------
# TileLang availability guard
# ---------------------------------------------------------------------------

_TILELANG_AVAILABLE = False
_TILELANG_IMPORT_ERROR: Optional[Exception] = None


def _try_init_tilelang() -> bool:
    global _TILELANG_AVAILABLE, _TILELANG_IMPORT_ERROR
    if _TILELANG_AVAILABLE:
        return True
    if _TILELANG_IMPORT_ERROR is not None:
        return False
    try:
        import tilelang  # noqa: F401
        _TILELANG_AVAILABLE = True
        return True
    except Exception as e:  # pragma: no cover
        _TILELANG_IMPORT_ERROR = e
        return False


# ---------------------------------------------------------------------------
# Triton fallback (always available)
# ---------------------------------------------------------------------------
from flashlib.primitives.knn.triton.dispatch import flash_knn_triton  # noqa: E402


# ---------------------------------------------------------------------------
# TileLang KNN kernel
# ---------------------------------------------------------------------------

def tilelang_flash_knn(
    x: torch.Tensor,
    c: torch.Tensor,
    k: int,
    **kwargs,
) -> torch.Tensor:
    """TileLang brute-force KNN — returns top-K indices.

    Falls back to Triton if tilelang is unavailable.

    Parameters
    ----------
    x : (B, N, D) query tensor (padded to D >= 16 by impl.py)
    c : (B, M, D) corpus tensor
    k : int — number of neighbors

    Returns
    -------
    (B, N, k) int64 indices of K nearest corpus points per query
    """
    if not _try_init_tilelang():
        return flash_knn_triton(x, c, k, **kwargs)

    import tilelang
    from tilelang import T

    B, N, D = x.shape
    M = c.shape[1]

    # TODO: Phase 2 — implement TileLang KNN kernel
    #
    # Design (multi-kernel):
    #   Kernel 1 — Distance matrix:
    #     - Compute cross = x @ c^T via T.gemm (one block per (N_tile, M_tile))
    #     - dist = x_sq + c_sq - 2 * cross
    #     - Write to (N, M) distance matrix in global memory
    #
    #   Kernel 2 — Top-K selection:
    #     - Serial iterative insert per query point
    #     - Or use torch.topk on the distance matrix (simpler, correct)
    #
    # Key gotchas:
    #   - D must be >= 16 for T.gemm (impl.py _prepare_inputs handles padding)
    #   - Precompute x_sq = (x*x).sum(-1) and c_sq = (c*c).sum(-1)
    #   - Cast to float16 for TileLang GEMM, accumulate distances in float32
    #   - For large N x M, tile the distance matrix (don't materialize all at once)
    #
    # Simpler approach (correct but uses more memory):
    #   1. Compute full (N, M) distance matrix via TileLang GEMM kernel
    #   2. Use torch.topk(dist, k, largest=False) for top-K selection
    #
    # Production approach (memory-efficient):
    #   1. Tile M in chunks of BLOCK_M
    #   2. For each chunk, compute distances and maintain running top-K
    #   3. Iterative insert into sorted top-K buffer
    raise NotImplementedError(
        "TileLang KNN kernel not yet implemented — see docs/phase-2-knn.md"
    )
