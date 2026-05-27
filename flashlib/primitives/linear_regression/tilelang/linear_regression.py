"""TileLang Linear Regression — cov_gemm + solve.

Phase: 5
Reference: flashlib/primitives/linear_regression/triton/
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
from flashlib.primitives.linear_regression.triton import (  # noqa: E402
    triton_linear_regression,
)


# ---------------------------------------------------------------------------
# TileLang Linear Regression
# ---------------------------------------------------------------------------

def tilelang_linear_regression(
    X: torch.Tensor,
    y: torch.Tensor,
    n_refine: int = 1,
    *,
    tol: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Linear regression via normal equations using TileLang cov_gemm.

    Falls back to Triton if tilelang is unavailable.

    Parameters
    ----------
    X : (N, D) float32 CUDA tensor
    y : (N,) or (N, 1) float32 target
    n_refine : int — iterative refinement steps
    tol : optional precision tolerance

    Returns
    -------
    (w, b) — (D,) weight vector, scalar bias
    """
    if not _try_init_tilelang():
        return triton_linear_regression(X, y, n_refine=n_refine, tol=tol, **kwargs)

    # TODO: Phase 5 — implement TileLang Linear Regression
    #
    # Design:
    #   1. XtX = tilelang_cov_gemm(X)          -> (D, D)  [Phase 3-4 must be done]
    #   2. Xty = X.T @ y                        -> (D,)
    #   3. w = torch.linalg.solve(XtX, Xty)     -> (D,)
    #   4. b = y.mean() - X.mean(0) @ w
    #   5. Optional: iterative refinement
    #      residual = y - X @ w - b
    #      dw = torch.linalg.solve(XtX, X.T @ residual)
    #      w = w + dw
    #
    # Note: The dominant cost is step 1 (cov_gemm). Steps 2-5 are cheap.
    raise NotImplementedError(
        "TileLang linear regression not yet implemented — see docs/phase-5-remaining.md"
    )
