"""TileLang PCA — delegates to TileLang cov_gemm + torch.linalg.eigh.

Phase: 3
Reference: flashlib/primitives/pca/triton/
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
from flashlib.primitives.pca.triton import triton_pca  # noqa: E402


# ---------------------------------------------------------------------------
# TileLang PCA
# ---------------------------------------------------------------------------

def tilelang_pca(
    X: torch.Tensor,
    K: int,
    *,
    tol: Optional[float] = None,
    **kwargs,
):
    """PCA top-K eigenpairs using TileLang cov_gemm + eigh.

    Falls back to Triton if tilelang is unavailable.

    Parameters
    ----------
    X : (N, D) float32 CUDA tensor
    K : int — number of components
    tol : optional residual tolerance (unused by TileLang path currently)

    Returns
    -------
    (eigenvalues, eigenvectors) — (K,) and (D, K) tensors, ascending order
    """
    if not _try_init_tilelang():
        return triton_pca(X, K, tol=tol, **kwargs)

    # TODO: Phase 3 — implement TileLang PCA
    #
    # Design:
    #   1. Compute covariance: C = tilelang_cov_gemm(X)  -> (D, D)
    #      - For N > 4*D: use cov path (X^T @ X)
    #      - For N <= 4*D: use Gram path (X @ X^T) then convert eigenvalues
    #   2. Eigendecompose: eigenvalues, eigenvectors = torch.linalg.eigh(C)
    #      - eigh returns ASCENDING order
    #   3. Return top-K: eigenvalues[-K:], eigenvectors[:, -K:]
    #
    # Note: tilelang_cov_gemm is in flashlib.linalg.cov_gemm.tilelang
    #       It must be implemented first (Phase 3-4).
    #
    # For the Gram path (N <= 4*D):
    #   - Compute G = X @ X^T  -> (N, N)
    #   - eigh(G) gives eigenvalues directly; eigenvectors are (N, K)
    #   - Convert to right eigenvectors: V = X^T @ U / sqrt(lambda)
    raise NotImplementedError(
        "TileLang PCA not yet implemented — see docs/phase-3-pca.md"
    )
