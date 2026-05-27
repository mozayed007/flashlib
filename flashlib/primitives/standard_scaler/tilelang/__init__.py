"""StandardScaler TileLang backend.

Re-exports the public Python wrappers. ``@tilelang.jit`` kernels stay
private to their file.
"""
from flashlib.primitives.standard_scaler.tilelang.scaler import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_available,
    tilelang_standard_scaler_fit,
    tilelang_standard_scaler_transform,
    tilelang_standard_scaler_fit_transform,
)

__all__ = [
    "tilelang_available",
    "tilelang_standard_scaler_fit",
    "tilelang_standard_scaler_transform",
    "tilelang_standard_scaler_fit_transform",
]
