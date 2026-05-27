"""KMeans TileLang backend.

Re-exports the public Python wrappers. ``@tilelang.jit`` kernels stay
private to their file.
"""
from flashlib.primitives.kmeans.tilelang.assign import (
    _TILELANG_AVAILABLE,
    _try_init_tilelang,
    tilelang_assign_euclid,
    tilelang_kmeans_Euclid,
)

__all__ = [
    "tilelang_assign_euclid",
    "tilelang_kmeans_Euclid",
]
