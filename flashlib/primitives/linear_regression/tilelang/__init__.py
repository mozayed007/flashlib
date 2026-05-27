"""Linear Regression TileLang backend.

Re-exports the public Python wrappers. ``@tilelang.jit`` kernels stay
private to their file.
"""
from flashlib.primitives.linear_regression.tilelang.linear_regression import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_linear_regression,
)

__all__ = [
    "tilelang_linear_regression",
]
