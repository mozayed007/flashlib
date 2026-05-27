"""TileLang StandardScaler — elementwise mean/variance + scaling.

Phase: 5
Reference: flashlib/primitives/standard_scaler/triton/
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
from flashlib.primitives.standard_scaler.triton import (  # noqa: E402
    flash_standard_scaler_fit as triton_standard_scaler_fit,
    flash_standard_scaler_transform as triton_standard_scaler_transform,
    flash_standard_scaler_fit_transform as triton_standard_scaler_fit_transform,
)


def tilelang_available() -> bool:
    """Check if TileLang is available."""
    return _try_init_tilelang()


# ---------------------------------------------------------------------------
# TileLang StandardScaler
# ---------------------------------------------------------------------------

def tilelang_standard_scaler_fit(
    X: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute mean, std, inv_std using TileLang.

    Falls back to Triton if tilelang is unavailable.

    Parameters
    ----------
    X : (N, D) float32 CUDA tensor

    Returns
    -------
    (mean, std, inv_std) — each (D,) float32
    """
    if not _try_init_tilelang():
        return triton_standard_scaler_fit(X, **kwargs)

    # TODO: Phase 5 — implement TileLang standard_scaler fit
    #
    # Design:
    #   - Parallel reduction along N dimension for mean and variance
    #   - Two-pass: (1) compute mean, (2) compute variance
    #   - Or one-pass Welford algorithm
    #   - TileLang T.Parallel reduction for each feature column
    #
    # Simpler approach (correct, may be slower than Triton):
    #   mean = X.mean(dim=0)
    #   std = X.std(dim=0)
    #   inv_std = 1.0 / (std + 1e-8)
    #   return mean, std, inv_std
    raise NotImplementedError(
        "TileLang standard_scaler fit not yet implemented — see docs/phase-5-remaining.md"
    )


def tilelang_standard_scaler_transform(
    X: torch.Tensor,
    mean: torch.Tensor,
    inv_std: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """Scale X using precomputed mean and inv_std.

    Falls back to Triton if tilelang is unavailable.
    """
    if not _try_init_tilelang():
        return triton_standard_scaler_transform(X, mean, inv_std, **kwargs)

    # TODO: Phase 5 — implement TileLang standard_scaler transform
    #
    # Design:
    #   - Elementwise: Y = (X - mean) * inv_std
    #   - Tile with T.Parallel over (N, D)
    #   - Or use torch directly (this is memory-bound, not compute-bound)
    raise NotImplementedError(
        "TileLang standard_scaler transform not yet implemented — see docs/phase-5-remaining.md"
    )


def tilelang_standard_scaler_fit_transform(
    X: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Fit and transform in one pass.

    Falls back to Triton if tilelang is unavailable.
    """
    if not _try_init_tilelang():
        return triton_standard_scaler_fit_transform(X, **kwargs)

    # TODO: Phase 5 — implement TileLang standard_scaler fit_transform
    #
    # Design:
    #   1. Fit: mean, std, inv_std = tilelang_standard_scaler_fit(X)
    #   2. Transform: Y = tilelang_standard_scaler_transform(X, mean, inv_std)
    #   3. Return (Y, (mean, std))
    raise NotImplementedError(
        "TileLang standard_scaler fit_transform not yet implemented — see docs/phase-5-remaining.md"
    )
