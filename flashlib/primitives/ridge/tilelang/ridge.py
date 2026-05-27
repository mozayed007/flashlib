"""TileLang Ridge Regression — cov_gemm + regularized solve.

Phase: 5
Reference: flashlib/primitives/ridge/triton/ridge.py
Guide: docs/phase-5-remaining.md
"""
from __future__ import annotations

from typing import Optional, Tuple

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
from flashlib.primitives.ridge.triton.ridge import (  # noqa: E402
    triton_ridge_regression,
)


# ---------------------------------------------------------------------------
# TileLang Ridge Regression
# ---------------------------------------------------------------------------

def tilelang_ridge_regression(
    X: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 1.0,
    *,
    tol: Optional[float] = None,
    n_refine: int = 1,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ridge regression via normal equations using TileLang cov_gemm.

    Falls back to Triton if tilelang is unavailable.

    Parameters
    ----------
    X : (N, D) float32 CUDA tensor
    y : (N,) or (N, 1) float32 target
    alpha : float — regularization strength
    tol : optional precision tolerance
    n_refine : int — iterative refinement steps

    Returns
    -------
    (w, b) — (D,) weight vector, scalar bias
    """
    if not _try_init_tilelang():
        return triton_ridge_regression(
            X, y, alpha=alpha, tol=tol, n_refine=n_refine, **kwargs,
        )

    # TODO: Phase 5 — implement TileLang Ridge Regression
    #
    # Design:
    #   1. XtX = tilelang_cov_gemm(X)                    -> (D, D)
    #   2. XtX += alpha * torch.eye(D, device=X.device)   -> regularize
    #   3. Xty = X.T @ y                                  -> (D,)
    #   4. w = torch.linalg.solve(XtX, Xty)               -> (D,)
    #   5. b = y.mean() - X.mean(0) @ w
    #   6. Optional: iterative refinement (same as linear_regression)
    #
    # Note: Almost identical to linear_regression — only difference is
    #       adding alpha * I to XtX before solving.
    raise NotImplementedError(
        "TileLang ridge regression not yet implemented — see docs/phase-5-remaining.md"
    )
