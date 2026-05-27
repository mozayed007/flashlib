"""TileLang DBSCAN — KNN radius search + connected components.

Phase: 5
Reference: flashlib/primitives/dbscan/triton/dbscan.py
Guide: docs/phase-5-remaining.md
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
from flashlib.primitives.dbscan.triton.dbscan import (  # noqa: E402
    flash_dbscan as triton_dbscan,
)


# ---------------------------------------------------------------------------
# TileLang DBSCAN
# ---------------------------------------------------------------------------

def tilelang_dbscan(
    X: torch.Tensor,
    eps: float,
    min_samples: int = 5,
    max_neighbors: int = 32,
    *,
    tol: Optional[float] = None,
    **kwargs,
):
    """DBSCAN using TileLang KNN + connected components.

    Falls back to Triton if tilelang is unavailable.

    Parameters
    ----------
    X : (N, D) float32 CUDA tensor
    eps : float — neighborhood radius
    min_samples : int — minimum points for core point
    max_neighbors : int — max neighbors per point
    tol : optional precision tolerance for KNN

    Returns
    -------
    (N,) int32 cluster labels (-1 for noise)
    """
    if not _try_init_tilelang():
        return triton_dbscan(
            X, eps=eps, min_samples=min_samples,
            max_neighbors=max_neighbors, tol=tol, **kwargs,
        )

    # TODO: Phase 5 — implement TileLang DBSCAN
    #
    # Design:
    #   1. KNN radius search: find all neighbors within eps
    #      - Reuse tilelang_flash_knn (Phase 2) with k=max_neighbors
    #      - Filter by distance <= eps^2
    #   2. Core point identification: count neighbors >= min_samples
    #   3. Connected components: union-find on core-core edges
    #      - Reuse flashlib.kernels.connected_components (already exists)
    #   4. Border point assignment: assign non-core points to nearest core's cluster
    #
    # Note: The connected components step is CPU-bound (atomic UF),
    #       not worth porting to TileLang. Focus on the KNN radius search.
    raise NotImplementedError(
        "TileLang DBSCAN not yet implemented — see docs/phase-5-remaining.md"
    )
