"""KNN TileLang backend.

Re-exports the public Python wrappers. ``@tilelang.jit`` kernels stay
private to their file.
"""
from flashlib.primitives.knn.tilelang.knn import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_flash_knn,
)

__all__ = [
    "tilelang_flash_knn",
]
