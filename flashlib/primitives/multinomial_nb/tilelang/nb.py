"""TileLang Multinomial NB — count + log-prob GEMM.

Phase: 5
Reference: flashlib/primitives/multinomial_nb/triton/nb.py
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
from flashlib.primitives.multinomial_nb.triton.nb import (  # noqa: E402
    flash_multinomial_nb as triton_multinomial_nb,
)


# ---------------------------------------------------------------------------
# TileLang Multinomial NB
# ---------------------------------------------------------------------------

def tilelang_multinomial_nb(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    n_classes: int,
    alpha: float = 1.0,
    predict_dtype=None,
    **kwargs,
):
    """Multinomial NB using TileLang predict GEMM.

    Falls back to Triton if tilelang is unavailable.

    Parameters
    ----------
    X_train : (N_train, D) float tensor
    y_train : (N_train,) int tensor
    X_test : (N_test, D) float tensor
    n_classes : int
    alpha : float — Laplace smoothing
    predict_dtype : optional dtype for predict GEMM

    Returns
    -------
    (N_test,) int64 predicted labels
    """
    if not _try_init_tilelang():
        return triton_multinomial_nb(
            X_train, y_train, X_test, n_classes,
            alpha=alpha, predict_dtype=predict_dtype, **kwargs,
        )

    # TODO: Phase 5 — implement TileLang Multinomial NB
    #
    # Design:
    #   Fit (can use torch — not the bottleneck):
    #     1. Count features per class: count[c, d] = sum(X_train[y==c, d]) + alpha
    #     2. log_proba_per_feature = log(count / count.sum(axis=1, keepdim=True))
    #     3. log_prior = log(class_counts / N_train)
    #
    #   Predict (the GEMM-bound part — TileLang this):
    #     1. log_proba = X_test @ log_proba_per_feature.T + log_prior
    #     2. labels = argmax(log_proba, dim=1)
    #
    #   The predict step is a standard GEMM — use T.gemm.
    #   Or delegate fit to torch and only TileLang the predict GEMM.
    raise NotImplementedError(
        "TileLang Multinomial NB not yet implemented — see docs/phase-5-remaining.md"
    )
