"""TileLang Logistic Regression — GEMV + L-BFGS.

Phase: 5
Reference: flashlib/primitives/logistic_regression/triton/logistic_regression.py
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
from flashlib.primitives.logistic_regression.triton.logistic_regression import (  # noqa: E402
    triton_logistic_regression,
)


# ---------------------------------------------------------------------------
# TileLang Logistic Regression
# ---------------------------------------------------------------------------

def tilelang_logistic_regression(
    X: torch.Tensor,
    y: torch.Tensor,
    n_iter: int = 100,
    lr: Optional[float] = None,
    C: float = 1.0,
    gtol: float = 1e-4,
    m_lbfgs: int = 10,
    *,
    tol: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, float]:
    """L-BFGS Logistic Regression with TileLang forward GEMV.

    Falls back to Triton if tilelang is unavailable.

    Parameters
    ----------
    X : (N, D) float32 CUDA tensor
    y : (N,) float32 binary labels (0/1)
    n_iter : int — max L-BFGS iterations
    lr : optional learning rate
    C : float — inverse regularization strength
    gtol : float — gradient sup-norm convergence threshold
    m_lbfgs : int — L-BFGS history size
    tol : optional precision tolerance

    Returns
    -------
    (w, b) — (D,) weight vector, scalar bias
    """
    if not _try_init_tilelang():
        return triton_logistic_regression(
            X, y, n_iter=n_iter, lr=lr, C=C, gtol=gtol, m_lbfgs=m_lbfgs, tol=tol,
            **kwargs,
        )

    # TODO: Phase 5 — implement TileLang Logistic Regression
    #
    # Design:
    #   The L-BFGS loop is Python-level (not GPU-bound). The bottleneck
    #   is the forward pass GEMV: logits = X @ w + b
    #
    #   Option A — TileLang forward GEMV only:
    #     - Implement a TileLang matrix-vector multiply kernel
    #     - Keep L-BFGS loop in Python with torch autograd-style gradient
    #
    #   Option B — Full delegation (simpler, still correct):
    #     - Just call triton_logistic_regression (the L-BFGS loop is the same)
    #     - Only the forward GEMV differs, but Triton's is already fast
    #
    #   Recommendation: Start with Option B (stub that delegates to Triton).
    #   If benchmarks show the GEMV is a bottleneck, implement Option A.
    raise NotImplementedError(
        "TileLang logistic regression not yet implemented — see docs/phase-5-remaining.md"
    )
