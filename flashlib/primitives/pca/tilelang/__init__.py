"""PCA TileLang backend.

Re-exports the public Python wrappers. ``@tilelang.jit`` kernels stay
private to their file.
"""
from flashlib.primitives.pca.tilelang.pca import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_pca,
)

__all__ = [
    "tilelang_pca",
]
