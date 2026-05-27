"""Logistic Regression TileLang backend.

Re-exports the public Python wrappers. ``@tilelang.jit`` kernels stay
private to their file.
"""
from flashlib.primitives.logistic_regression.tilelang.logistic_regression import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_logistic_regression,
)

__all__ = [
    "tilelang_logistic_regression",
]
