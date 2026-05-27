"""TileLang Truncated SVD — delegates to TileLang cov_gemm + eigh.

Phase: 3
Reference: flashlib/primitives/truncated_svd/triton/
Guide: docs/phase-3-pca.md
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
from flashlib.primitives.truncated_svd.triton import triton_truncated_svd  # noqa: E402


# ---------------------------------------------------------------------------
# TileLang Truncated SVD
# ---------------------------------------------------------------------------

def tilelang_truncated_svd(
    X: torch.Tensor,
    K: int,
    *,
    tol: Optional[float] = None,
    **kwargs,
):
    """Truncated SVD using TileLang cov_gemm + eigh.

    Falls back to Triton if tilelang is unavailable.

    Parameters
    ----------
    X : (N, D) float32 CUDA tensor
    K : int — number of singular components
    tol : optional residual tolerance

    Returns
    -------
    (S, Vh) — (K,) singular values descending, (K, D) right singular vectors
    """
    if not _try_init_tilelang():
        return triton_truncated_svd(X, K, tol=tol, **kwargs)

    # TODO: Phase 3 — implement TileLang Truncated SVD
    #
    # Design:
    #   1. Compute covariance: C = tilelang_cov_gemm(X)  -> (D, D)
    #   2. Eigendecompose: eigenvalues, eigenvectors = torch.linalg.eigh(C)
    #   3. Top-K eigenvalues are singular values squared: S = sqrt(eigenvalues[-K:])
    #   4. Right singular vectors: Vh = eigenvectors[:, -K:].T
    #   5. Return (S.flip(0), Vh.flip(0))  — descending order
    #
    # Note: SVD and PCA share the cov_gemm + eigh pipeline.
    #       The difference is the post-processing (sqrt + reordering).
    raise NotImplementedError(
        "TileLang Truncated SVD not yet implemented — see docs/phase-3-pca.md"
    )
