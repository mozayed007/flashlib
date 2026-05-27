"""Truncated SVD TileLang backend.

Re-exports the public Python wrappers. ``@tilelang.jit`` kernels stay
private to their file.
"""
from flashlib.primitives.truncated_svd.tilelang.svd import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_truncated_svd,
)

__all__ = [
    "tilelang_truncated_svd",
]
