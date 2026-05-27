"""TileLang euclidean assign kernel for KMeans.

Phase: 1
Reference: flashlib/primitives/kmeans/triton/assign.py (lines 823-924)
Guide: docs/phase-1-kmeans.md
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
from flashlib.primitives.kmeans.triton.assign import euclid_assign_triton  # noqa: E402
from flashlib.primitives.kmeans.triton.update import (  # noqa: E402
    triton_centroid_finalize,
    triton_lloyd_centroid_step_euclid,
)


# ---------------------------------------------------------------------------
# TileLang assign kernel
# ---------------------------------------------------------------------------

def tilelang_assign_euclid(
    x: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """TileLang euclidean assign — argmin_j ||x_i - c_j||^2 per point.

    Falls back to Triton if tilelang is unavailable.

    Parameters
    ----------
    x : (B, N, D) float32/float16 tensor
    centroids : (B, K, D) float32/float16 tensor

    Returns
    -------
    (B, N) int32 cluster assignments
    """
    if not _try_init_tilelang():
        return euclid_assign_triton(x, centroids)

    import tilelang
    from tilelang import T

    B, N, D = x.shape
    K = centroids.shape[1]

    # TODO: Phase 1 — implement TileLang assign kernel
    #
    # Design:
    #   - Each block handles BLOCK_N points for one batch element
    #   - Iterate over K centroids in chunks of BLOCK_K
    #   - For each chunk:
    #     1. Load X tile (BLOCK_N, D) to shared memory
    #     2. Load C tile (BLOCK_K, D) to shared memory
    #     3. T.clear(cross_frag) then T.gemm(X_shared, C_shared, cross_frag, transpose_B=True)
    #     4. Epilogue: dist = c_sq[k] - 2.0 * cross_frag[i, k], serial argmin
    #   - Write best_idx to output
    #
    # Key gotchas:
    #   - T.gemm ACCUMULATES — always T.clear() before first GEMM
    #   - argmin must use GLOBAL index: k_start + k (not just k)
    #   - Cast x/centroids to float16 for TileLang; accumulate in float32
    #   - Precompute c_sq = (centroids * centroids).sum(dim=-1).float()
    #   - BLOCK_N=64, BLOCK_K=64 is a safe starting point
    #
    # Reference kernel structure:
    #   @tilelang.jit(out_idx=[-1])
    #   def _build_assign_kernel(N, D, K, BLOCK_N, BLOCK_K):
    #       @T.prim_func
    #       def kernel(X, C, C_sq, Out):
    #           X_shared = T.alloc_shared([BLOCK_N, D], "float16")
    #           C_shared = T.alloc_shared([BLOCK_K, D], "float16")
    #           cross_frag = T.alloc_fragment([BLOCK_N, BLOCK_K], "float32")
    #           best_dist = T.alloc_fragment([BLOCK_N], "float32")
    #           best_idx = T.alloc_fragment([BLOCK_N], "int32")
    #           ...
    #       return kernel
    raise NotImplementedError(
        "TileLang assign kernel not yet implemented — see docs/phase-1-kmeans.md"
    )


# ---------------------------------------------------------------------------
# TileLang full Lloyd loop
# ---------------------------------------------------------------------------

def tilelang_kmeans_Euclid(
    x: torch.Tensor,
    n_clusters: int,
    *,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids=None,
    verbose: bool = False,
    **kwargs,
):
    """Full KMeans Lloyd loop — TileLang assign + Triton centroid update.

    Falls back to Triton if tilelang is unavailable.

    Parameters
    ----------
    x : (B, N, D) tensor
    n_clusters : int
    max_iters : int
    tol : float — convergence tolerance on centroid shift
    init_centroids : optional (B, K, D) tensor
    verbose : bool

    Returns
    -------
    (cluster_ids, centroids, n_iter) — same interface as batch_kmeans_Euclid
    """
    if not _try_init_tilelang():
        from flashlib.primitives.kmeans.triton.kmeans import batch_kmeans_Euclid
        return batch_kmeans_Euclid(
            x, n_clusters, max_iters=max_iters, tol=tol,
            init_centroids=init_centroids, verbose=verbose, **kwargs,
        )

    # TODO: Phase 1 — implement TileLang Lloyd loop
    #
    # Design:
    #   1. Initialize centroids (random or from init_centroids)
    #   2. For each iteration:
    #      a. cluster_ids = tilelang_assign_euclid(x, centroids)
    #      b. new_centroids, converged = triton_lloyd_centroid_step_euclid(x, cluster_ids, K)
    #      c. centroids = triton_centroid_finalize(new_centroids, cluster_ids, K)
    #      d. If converged and tol > 0: break
    #   3. Return (cluster_ids, centroids, n_iter)
    #
    # Note: Reuse triton_lloyd_centroid_step_euclid and triton_centroid_finalize
    #       from flashlib.primitives.kmeans.triton.update (already imported above).
    #       Only the assign step is ported to TileLang.
    raise NotImplementedError(
        "TileLang Lloyd loop not yet implemented — see docs/phase-1-kmeans.md"
    )
