"""Ridge Regression TileLang backend.

Re-exports the public Python wrappers. ``@tilelang.jit`` kernels stay
private to their file.
"""
from flashlib.primitives.ridge.tilelang.ridge import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_ridge_regression,
)

__all__ = [
    "tilelang_ridge_regression",
]
